from dataclasses import is_dataclass, replace
from typing import Any, Optional

import torch
from transformers import AutoTokenizer

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.worker.gpu_input_batch import InputBatch


logger = init_logger(__name__)

class DraftModelProposer:
    def __init__(self,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 runner: Optional[Any] = None):
        self.vllm_config = vllm_config
        self.device = device
        # Access to runner is needed to build attention metadata for drafting.
        # We keep it optional for construction, but propose() requires it.
        self.runner = runner
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len
        # lazily initialized
        self.attn_layer_names: list[str] = []
        self.tokenizer = AutoTokenizer.from_pretrained(vllm_config.speculative_config.model)

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        common_attn_metadata: CommonAttentionMetadata,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_batch: InputBatch,
    ) -> torch.Tensor:
        """
        Generate draft tokens using autoregressive generation with the draft
        model.
        
        Args:
            sampled_token_ids: The tokens just sampled by the target model
                [batch_size][tokens]
            attn_metadata: Layer-to-attention metadata mapping from target model
            
        Returns:
            torch.Tensor: Draft tokens of shape [batch_size,
                num_speculative_tokens]
        """
        assert self.runner is not None, (
            "DraftModelProposer requires runner to build attention metadata.")

        # Convert ragged sampled_token_ids (list[list[int]]) to last-token
        # tensor [B]. Avoid torch.tensor on ragged lists by extracting the last
        # element per request.
        next_token_ids_list = [tokens[-1] for tokens in sampled_token_ids]
        next_token_ids = torch.tensor(
            next_token_ids_list,
            dtype=torch.int32,
            device=self.device,
        )
        batch_size = int(next_token_ids.shape[0])
        num_speculative_tokens: int = (
            self.vllm_config.speculative_config.num_speculative_tokens)

        # Use the provided CommonAttentionMetadata and adapt it for drafting.

        # Build per-layer attention metadata for drafting.
        # Use the first group's builder (same assumption as EAGLE).
        builder: AttentionMetadataBuilder = self.runner.attn_groups[0][0].metadata_builder
        attn_meta = builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0)
        
        # Indices of last tokens per request in the flat target input.
        last_token_indices = common_attn_metadata.query_start_loc[1:] - 1
        # Gather the last positions and advance by 1 for the first draft step.
        cur_positions = positions[last_token_indices] + 1

        # Output buffer [B, num_speculative_tokens]
        output_ids = torch.empty((batch_size, num_speculative_tokens),
                                 dtype=torch.long, device=self.device)

        # Convenience buffers
        arange_bszp1 = torch.arange(batch_size + 1,
                                     device=self.device,
                                     dtype=torch.int32)
        
        # Build per-request full contexts (prompt + accepted tokens) from the
        # input batch. Note: `num_tokens_no_spec` already includes the
        # just-sampled token from the target model for this step, so we must
        # NOT append it again here to avoid duplicating the last token.
        req_idxs = [input_batch.req_id_to_index[i] for i in input_batch.req_ids]

        full_input_ids_list: list[torch.Tensor] = []
        full_positions_list: list[torch.Tensor] = []
        new_lens: list[int] = []
        for i, req_idx in enumerate(req_idxs):
            base_len = int(input_batch.num_tokens_no_spec[req_idx])
            base_ids_np = input_batch.token_ids_cpu[req_idx, :base_len]
            base_ids = torch.from_numpy(base_ids_np).to(self.device,
                                                        dtype=torch.int32)

            # Use the existing context exactly as-is (already includes the
            # just-sampled token). Do not append `next_id` again.
            seq_ids = base_ids

            # Text models commonly use 1D positions starting from 0.
            seq_pos = torch.arange(seq_ids.shape[0],
                                   device=self.device,
                                   dtype=torch.int64)

            full_input_ids_list.append(seq_ids)
            full_positions_list.append(seq_pos)
            new_lens.append(int(seq_ids.shape[0]))

        full_input_ids = torch.cat(full_input_ids_list, dim=0)
        full_positions = torch.cat(full_positions_list, dim=0)
        num_tokens = len(full_input_ids)

        last_token_indices = (torch.cumsum(
            torch.tensor(new_lens, device=self.device), dim=0) - 1)

        with set_forward_context(self._per_layer(attn_meta), self.vllm_config,
                                 num_tokens=num_tokens):
            full_hidden_states = self.model(
                input_ids=full_input_ids,
                positions=full_positions,
            )
            full_logits = self.model.compute_logits(
                full_hidden_states[last_token_indices], None)
            first_draft = torch.argmax(full_logits, dim=-1).to(torch.int32)
            output_ids[:, 0] = first_draft
            logger.info("Draft.forward() on %d tokens: %s", len(full_input_ids), [self.tokenizer.decode(t) for t in full_input_ids])

        # Step-by-step speculative rollout for remaining tokens.
        draft_inputs = first_draft  # [B]
        for step in range(1, num_speculative_tokens):
            attn_meta = _update_attn_metadata_for_draft(
                attn_meta=attn_meta,
                positions_1d=cur_positions,
                block_size=self.block_size,
                max_model_len=self.max_model_len,
                batch_size=batch_size,
                arange_bszp1=arange_bszp1,
            )

            with set_forward_context(self._per_layer(attn_meta),
                                      self.vllm_config,
                                      num_tokens=batch_size):
                hidden_states = self.model(
                    input_ids=draft_inputs,
                    positions=cur_positions,
                )
                logits = self.model.compute_logits(hidden_states, None)
                next_tokens = torch.argmax(logits, dim=-1).to(torch.int32)
                output_ids[:, step] = next_tokens
                logger.info("Draft.forward() on %d tokens: %s", len(draft_inputs), [self.tokenizer.decode(t) for t in draft_inputs])

            # Next step inputs/positions
            draft_inputs = next_tokens
            cur_positions = cur_positions + 1

        return output_ids

    def load_model(self, *args, **kwargs) -> None:
        draft_vllm_config = replace(
            self.vllm_config, 
            model_config=self.vllm_config.speculative_config.draft_model_config
        )
        self.model = get_model(
            vllm_config=draft_vllm_config, 
            prefix="draft_model"
        )
        self.attn_layer_names = list(
            get_layers_from_vllm_config(draft_vllm_config, Attention).keys()
        )

    def _per_layer(self, attn_meta_obj: Any) -> dict[str, Any]:
        return {layer_name: attn_meta_obj
                for layer_name in self.attn_layer_names}


def _update_attn_metadata_for_draft(
    attn_meta: Any,
    positions_1d: torch.Tensor,
    block_size: int,
    max_model_len: int,
    batch_size: int,
    arange_bszp1: torch.Tensor,
) -> Any:
    """Pure helper that returns a new attention metadata object for a
    single-token-per-request draft step.

    - Does not mutate the input attn_meta
    - Requires attn_meta to be a dataclass (backend metadata are dataclasses)
    """
    assert is_dataclass(attn_meta), "Attention metadata must be a dataclass"

    exceeds = positions_1d >= max_model_len
    clamped_positions = torch.where(exceeds, 0, positions_1d)

    # Determine block table tensor
    block_table = getattr(attn_meta, "block_table",
                          getattr(attn_meta, "block_table_tensor", None))
    if block_table is None:
        raise AssertionError("Attention metadata missing block_table")

    block_numbers = (clamped_positions // block_size).view(-1, 1)
    block_ids = block_table.gather(dim=1, index=block_numbers).view(-1)
    slot_mapping = (block_ids * block_size
                    + (clamped_positions % block_size)).to(torch.int64)
    slot_mapping = slot_mapping.masked_fill(exceeds, -1)

    # Update lengths
    new_seq_lens = getattr(attn_meta, "seq_lens", None)
    if new_seq_lens is not None:
        new_seq_lens = new_seq_lens + 1
        # Optionally clamp for requests exceeding max length, like EAGLE
        new_seq_lens = new_seq_lens.masked_fill(exceeds, 1)

    # Optional max_seq_len
    max_seq_len = getattr(attn_meta, "max_seq_len", None)
    if max_seq_len is not None:
        max_seq_len = min(int(max_seq_len) + 1, max_model_len)

    # Build replacement kwargs
    kwargs: dict[str, Any] = {
        "slot_mapping": slot_mapping,
        "num_actual_tokens": batch_size,
        "max_query_len": 1,
        "query_start_loc": arange_bszp1[:batch_size + 1],
    }
    if new_seq_lens is not None:
        kwargs["seq_lens"] = new_seq_lens
    if max_seq_len is not None:
        kwargs["max_seq_len"] = max_seq_len

    return replace(attn_meta, **kwargs)
