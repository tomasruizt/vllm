# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FMMS (Fused Matrix Multiplication & Sampling) sampler for vLLM.

Fuses the lm_head matmul with Gumbel-max categorical sampling in a single
Triton/Helion kernel, avoiding materialization of the full [B, V] logits
tensor. Only supports temperature-only sampling (no top-k/top-p/penalties).
"""

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata


class FMMSSampler:
    def __init__(self, provider: str = "fused-triton"):
        from fused_mm_sampling.core import get_sampler

        # get_sampler needs a weights tensor for some providers (e.g. JL),
        # but fused-triton and helion don't, so pass a dummy.
        dummy = torch.empty(1, 1)
        sampler = get_sampler(provider, weights=dummy)
        self.sampler = sampler.prepare()

    def __call__(
        self,
        lm_head_weight: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        temperature = sampling_metadata.temperature
        if temperature is None:
            temperature = torch.ones(1, device=hidden_states.device)

        sampled = self.sampler.sample(
            weights=lm_head_weight,
            hidden_states=hidden_states,
            num_samples=1,
            temperature=temperature[0],
        )  # [B, 1] torch.long

        return SamplerOutput(
            sampled_token_ids=sampled.to(torch.int32),
            logprobs_tensors=None,
        )
