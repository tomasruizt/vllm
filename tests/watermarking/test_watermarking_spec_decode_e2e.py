# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.v1.e2e.spec_decode.utils import get_spec_decode_metric_value
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.watermarking import DualKeyGumbelWatermarkDetector


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="requires CUDA and Model Runner V2"
)
def test_speculative_generation_respects_request_watermarking(vllm_runner, monkeypatch):
    """Real MTP generation must remain detectable and honor per-request opt-out."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    watermark_config = {"key": 42, "context_width": 4}
    detector = DualKeyGumbelWatermarkDetector(**watermark_config)
    prompt = "Tell me a story about an explorer who discovers a mysterious island."
    sampling_params = [
        SamplingParams(
            temperature=1.0,
            max_tokens=256,
            seed=0,
            watermarking=enabled,
        )
        for enabled in (True, False)
    ]

    with vllm_runner(
        "Qwen/Qwen3.5-0.8B-Base",
        watermark_config={"algorithm": "dual_key_gumbel", **watermark_config},
        seed=0,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "draft_sample_method": "probabilistic",
        },
        language_model_only=True,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=2,
        gpu_memory_utilization=0.2,
        disable_log_stats=False,  # enables llm.get_metrics()
    ) as runner:
        outputs = runner.llm.generate([prompt, prompt], sampling_params)
        metrics = runner.llm.get_metrics()

    num_draft_tokens = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_draft_tokens"
    )
    num_accepted_tokens = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_accepted_tokens"
    )
    assert num_draft_tokens > 0, "Speculative decoding did not propose tokens"
    assert 0 < num_accepted_tokens < num_draft_tokens, (
        "Expected accepted and rejected drafts: "
        f"{num_accepted_tokens}/{num_draft_tokens}"
    )
    assert len(outputs) == 2
    for output, params in zip(outputs, sampling_params):
        token_ids = list(output.outputs[0].token_ids)
        detection = detector.detect(token_ids)
        print(f"watermarking={params.watermarking}: {detection}")
        assert detection.is_watermarked == params.watermarking, detection
