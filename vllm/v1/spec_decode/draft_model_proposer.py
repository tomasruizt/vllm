from dataclasses import is_dataclass, replace
from typing import Any, Optional

from attr import dataclass
import numpy as np
import torch
from transformers import AutoTokenizer

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch


logger = init_logger(__name__)


class DraftModelProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        # FIXME: The runner is only used to build attention metadata for drafting
        # Ideally there is an easier way to build this without accessing the full runner.
        runner: "GPUModelRunner",
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len
        # lazily initialized
        self.layer_names: list[str] = []
        # FOR DEBUGGING
        self.tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.speculative_config.model
        )

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        common_attn_metadata: CommonAttentionMetadata,
        input_batch: InputBatch,
    ) -> torch.Tensor:
        next_token_ids = torch.tensor(
            [tokens[-1] for tokens in sampled_token_ids],
            dtype=torch.int32,
            device=self.device,
        )
        batch_size = len(next_token_ids)
        num_speculative_tokens: int = (
            self.vllm_config.speculative_config.num_speculative_tokens
        )
        assert num_speculative_tokens == 1, (
            "Only one speculative token is supported for now"
        )
        output_ids = torch.empty(
            (batch_size, num_speculative_tokens),
            dtype=torch.long,
            device=self.device,
        )

        # Get the last position for each request from common_attn_metadata
        last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        flat_inputs = to_flat_inputs(input_batch).cuda()
        draft_attn_metadata = self._build_draft_attention_metadata(
            flat_inputs=flat_inputs,
            input_batch=input_batch,
        )
        with set_forward_context(
            draft_attn_metadata,
            self.vllm_config,
            num_tokens=flat_inputs.num_tokens(),
        ):
            logits = self.model(
                input_ids=flat_inputs.input_ids,
                positions=flat_inputs.positions,
            )
        toks = self.decode(flat_inputs.input_ids)
        logger.info("Draft.forward() on %d tokens: %s", len(toks), toks)

        draft_logits = logits[last_token_indices]
        first_draft = torch.argmax(draft_logits, dim=-1).to(torch.int32)
        output_ids[:, 0] = first_draft
        return output_ids

    def decode(self, ids: torch.Tensor) -> list[str]:
        return [self.tokenizer.decode(id) for id in ids]

    def _build_draft_attention_metadata(
        self,
        flat_inputs: "FlatInputs",
        input_batch: InputBatch,
    ) -> dict[str, Any]:
        kv_cache_group_id = 1  # draft model kv cache group
        blk_table = input_batch.block_table[kv_cache_group_id]
        blk_table_tensor = blk_table.get_device_tensor()[
            : flat_inputs.num_reqs()
        ]
        slot_mapping = blk_table.slot_mapping[: flat_inputs.num_tokens()]

        # Fill unused with -1. Needed for reshape_and_cache in full cuda
        # graph mode.
        blk_table.slot_mapping[flat_inputs.num_tokens() :].fill_(-1)
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=flat_inputs.query_start_loc,
            query_start_loc_cpu=flat_inputs.query_start_loc,
            seq_lens=torch.tensor(
                flat_inputs.seq_lens_cpu, device=self.device, dtype=torch.int32
            ),
            seq_lens_cpu=torch.tensor(
                flat_inputs.seq_lens_cpu, dtype=torch.int32
            ),
            num_computed_tokens_cpu=torch.tensor(
                flat_inputs.seq_lens_cpu, dtype=torch.int32
            ),
            num_reqs=flat_inputs.num_reqs(),
            num_actual_tokens=flat_inputs.num_tokens(),
            # NOTE: For draft models, we process the entire sequence as "query"
            # to generate the next token, so max_query_len equals max_seq_len.
            # This is different from the main model which typically only
            # processes new tokens in decode phase.
            max_query_len=max(flat_inputs.seq_lens_cpu),
            max_seq_len=max(flat_inputs.seq_lens_cpu),
            block_table_tensor=blk_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
        )

        attn_group = self.runner.attn_groups[1]  # draft model attn group
        assert len(attn_group) == 1
        attn_metadata = attn_group[0].metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )
        return attn_metadata

    def load_model(self, *args, **kwargs) -> None:
        draft_vllm_config = replace(
            self.vllm_config,
            model_config=self.vllm_config.speculative_config.draft_model_config,
        )
        self.model = get_model(
            vllm_config=draft_vllm_config, prefix="draft_model"
        )
        self.layer_names = list(
            get_layers_from_vllm_config(draft_vllm_config, Attention).keys()
        )


@dataclass
class FlatInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    last_token_ids: torch.Tensor
    seq_lens_cpu: list[int]  # Maybe tensor?
    query_start_loc: torch.Tensor

    def num_tokens(self) -> int:
        return len(self.input_ids)

    def num_reqs(self) -> int:
        return len(self.seq_lens_cpu)

    def cuda(self) -> "FlatInputs":
        return FlatInputs(
            input_ids=self.input_ids.cuda(non_blocking=True),
            positions=self.positions.cuda(non_blocking=True),
            last_token_ids=self.last_token_ids.cuda(non_blocking=True),
            seq_lens_cpu=self.seq_lens_cpu,
            query_start_loc=self.query_start_loc.cuda(non_blocking=True),
        )


def to_flat_inputs(input_batch: InputBatch) -> FlatInputs:
    """
    Prepares inputs for draft model inference.
    
    For draft models, we process the entire sequence (prompt + all generated 
    tokens) to predict the next token. This is different from the main model 
    which typically only processes new tokens in decode phase.
    
    Returns FlatInputs where seq_lens_cpu contains the full sequence lengths,
    and query_start_loc reflects processing the entire sequences as "queries".
    """
    input_ids_list: list[torch.Tensor] = []
    positions_list: list[torch.Tensor] = []
    seq_lens: list[int] = []

    req_idxs = [input_batch.req_id_to_index[i] for i in input_batch.req_ids]
    for req_idx in req_idxs:
        # Get the full sequence including all generated tokens 
        # (no speculative tokens)
        base_len = input_batch.num_tokens_no_spec[req_idx]
        seq_ids_np = input_batch.token_ids_cpu[req_idx, :base_len]
        seq_ids = torch.from_numpy(seq_ids_np).to(dtype=torch.int32)
        seq_pos = torch.arange(seq_ids.shape[0], dtype=torch.int64)

        input_ids_list.append(seq_ids)
        positions_list.append(seq_pos)
        seq_lens.append(len(seq_ids))

    input_ids = torch.cat(input_ids_list, dim=0)
    positions = torch.cat(positions_list, dim=0)

    # The last token of each request is used to sample the next token
    last_token_idxs = torch.cumsum(torch.tensor(seq_lens), dim=0) - 1

    # Create proper query_start_loc tensor
    query_start_loc = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    for i, seq_len in enumerate(seq_lens):
        query_start_loc[i + 1] = query_start_loc[i] + seq_len

    return FlatInputs(
        input_ids=input_ids,
        positions=positions,
        last_token_ids=last_token_idxs,
        seq_lens_cpu=seq_lens,
        query_start_loc=query_start_loc,
    )
