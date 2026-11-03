# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint loading and output parity for the fused QKVZ/BA projection."""

import pytest
import torch
import torch.nn.functional as F

from tests.v1.attention.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="input projection fusion requires CUDA"
)


@pytest.mark.parametrize(
    "dtype,checkpoint",
    [
        (torch.bfloat16, "hf"),
        (torch.float16, "hf"),
        (torch.bfloat16, "packed"),
        (torch.float16, "packed"),
        (torch.bfloat16, "fused"),
        (torch.bfloat16, "fused_shards"),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 4, 9, 256])
@torch.inference_mode()
def test_input_projection_loads_checkpoint_and_matches_separate_gemms(
    dist_init, dtype: torch.dtype, checkpoint: str, num_tokens: int
) -> None:
    """Exercise real weight loading and preserve QKVZ, b and a ordering exactly."""
    config = create_vllm_config(
        dtype=dtype,
        hf_config_override={
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        },
    )
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(dtype),
        torch.device("cuda"),
    ):
        layer = QwenGatedDeltaNetAttention(
            config.model_config.hf_text_config,
            config,
            prefix="model.layers.0.linear_attn",
        )
        model = torch.nn.ModuleDict({"linear_attn": layer})
        torch.manual_seed(0)
        sizes = [layer.key_dim * 2 + layer.value_dim, layer.value_dim, 4, 4]
        weights = [torch.randint(-2, 3, (n, 64)).to(dtype) / 8 for n in sizes]
        qkvz_weight = torch.cat(weights[:2])
        ba_weight = torch.cat(weights[2:])
        if checkpoint == "hf":
            names = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
            state = [(f"linear_attn.{n}.weight", w) for n, w in zip(names, weights)]
            mapper = Qwen3_5Model.hf_to_vllm_mapper
        elif checkpoint == "packed":
            state = [
                ("linear_attn.in_proj_qkvz.weight", qkvz_weight),
                ("linear_attn.in_proj_ba.weight", ba_weight),
            ]
            mapper = None
        elif checkpoint == "fused_shards":
            state = []
            for weight, shard_id in zip(weights, [(0, 1, 2), 3, 4, 5]):
                weight.shard_id = shard_id
                state.append(("linear_attn.in_proj_qkvz.weight", weight))
            mapper = None
        else:
            state = [("linear_attn.in_proj_qkvz.weight", torch.cat(weights))]
            mapper = None
        AutoWeightsLoader(model).load_weights(state, mapper=mapper)
        x = torch.randint(-2, 3, (num_tokens, 64)).to(dtype) / 8
        qkvz, ba = layer._input_projection(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_qkvz, graph_ba = layer._input_projection(x)
        graph.replay()

        assert layer.fuse_ba_proj == (dtype == torch.bfloat16)
        torch.testing.assert_close(qkvz, F.linear(x, qkvz_weight), rtol=0, atol=0)
        torch.testing.assert_close(ba, F.linear(x, ba_weight), rtol=0, atol=0)
        torch.testing.assert_close(graph_qkvz, qkvz, rtol=0, atol=0)
        torch.testing.assert_close(graph_ba, ba, rtol=0, atol=0)
        b, a = layer.split_ba(ba)
        torch.testing.assert_close(b, F.linear(x, weights[2]), rtol=0, atol=0)
        torch.testing.assert_close(a, F.linear(x, weights[3]), rtol=0, atol=0)
