# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure watermark detection on C4 continuations and unwatermarked controls.

Run: pytest tests/watermarking/test_detection_e2e.py -s

Requires a GPU and Hugging Face access for Qwen3.6-27B and English C4.
Uses general C4 validation prose, not a news-only subset. Generation stops
at EOS or 512 tokens; short and empty completions count in the detection rate.
"""

from statistics import mean

import pytest
import torch
from datasets import load_dataset

from vllm import SamplingParams
from vllm.v1.watermarking import (
    DualKeyGumbelWatermarkDetector,
    GumbelWatermarkDetector,
    WatermarkDetection,
    WatermarkDetector,
)

NUM_COMPLETIONS = 100
PROMPT_CHARS = 512
MAX_NEW_TOKENS = 512
P_VALUE_THRESHOLD = 0.01

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")


@pytest.mark.parametrize("algorithm", ["gumbel", "dual_key_gumbel"])
def test_c4_watermark_detection(vllm_runner, c4_prompts, algorithm):
    watermark_config, detector = watermarking_inputs(algorithm)

    detections: dict[bool, list[WatermarkDetection]] = {}
    with vllm_runner(
        "Qwen/Qwen3.6-27B",
        max_model_len=2048,
        max_num_seqs=NUM_COMPLETIONS,
        watermark_config=watermark_config,
        enable_chunked_prefill=True,  # required for mamba/GDN in Qwen3.6-27B
    ) as model:
        for use_watermarking in (True, False):
            sampling_params = SamplingParams(
                temperature=1.0,
                max_tokens=MAX_NEW_TOKENS,
                seed=0,
                watermarking=use_watermarking,
            )
            outputs = model.llm.generate(c4_prompts, sampling_params, use_tqdm=False)
            assert len(outputs) == len(c4_prompts)
            assert all(len(output.outputs) == 1 for output in outputs)
            detections[use_watermarking] = [
                detector.detect(list(output.outputs[0].token_ids)) for output in outputs
            ]

    true_positive_pct = 100 * mean(d.is_watermarked for d in detections[True])
    false_positive_pct = 100 * mean(d.is_watermarked for d in detections[False])
    mean_scored_tokens = mean(d.num_scored_tokens for d in detections[True])
    mean_watermarked_p_value = mean(d.p_value for d in detections[True])
    mean_control_p_value = mean(d.p_value for d in detections[False])
    summary = (
        f"{algorithm}: true positives={true_positive_pct:.1f}%, "
        f"false positives={false_positive_pct:.1f}%, "
        f"mean scored tokens={mean_scored_tokens:.1f}, "
        f"n={len(c4_prompts)}\n"
        f"Average p-value (watermarked): {mean_watermarked_p_value:.6g}\n"
        f"Average p-value (control): {mean_control_p_value:.6g}\n"
        f"Detection threshold: p <= {P_VALUE_THRESHOLD}"
    )
    print(summary)
    # Provisional regression bounds; validate these on GPU before enabling CI.
    assert true_positive_pct >= 90.0, summary
    assert false_positive_pct <= 5.0, summary


def watermarking_inputs(
    algorithm: str,
) -> tuple[dict, WatermarkDetector]:
    config = {
        "algorithm": algorithm,
        "key": 42,
        "context_width": 4,
        "prf": "philox",
        "deduplicate_contexts": "single_turn",
    }
    detector_kwargs = {
        "key": config["key"],
        "context_width": config["context_width"],
        "prf": config["prf"],
        "deduplicate_contexts": True,
        "p_value_threshold": P_VALUE_THRESHOLD,
    }
    if algorithm == "dual_key_gumbel":
        config["alpha"] = 0.1
        detector = DualKeyGumbelWatermarkDetector(
            **detector_kwargs, alpha=config["alpha"]
        )
    else:
        detector = GumbelWatermarkDetector(**detector_kwargs)
    return config, detector


@pytest.fixture(scope="module")
def c4_prompts() -> list[str]:
    # Stream only enough validation documents for this test, not the C4 corpus.
    corpus = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts = []
    for doc in corpus:
        text = doc["text"]
        if len(text) >= 2 * PROMPT_CHARS:
            prompts.append(text[:PROMPT_CHARS])
        if len(prompts) == NUM_COMPLETIONS:
            break
    assert len(prompts) == NUM_COMPLETIONS
    return prompts
