#!/usr/bin/env python3
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

def compute_acceptance_rate(metrics) -> float:
    name2metric = {metric.name: metric for metric in metrics}
    n_draft_toks = name2metric["vllm:spec_decode_num_draft_tokens"].value
    if n_draft_toks == 0:
        return float("nan")
    n_accepted_toks = name2metric["vllm:spec_decode_num_accepted_tokens"].value
    return n_accepted_toks / n_draft_toks

print("\n" + "="*80)
print("Testing: target=20b, draft=20b")
print("="*80 + "\n")

test_prompts = [[{"role": "user", "content": "please repeat the word 'test' 10 times."}]]
sampling_params = SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)

spec_llm = LLM(
    model="openai/gpt-oss-20b",
    speculative_config={
        "model": "openai/gpt-oss-20b",
        "method": "draft_model",
        "num_speculative_tokens": 3,
        "max_model_len": 1024,
        "enforce_eager": True,
        "draft_tensor_parallel_size": 1,
        "max_num_seqs": 100,
    },
    max_model_len=1024,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    enforce_eager=True,
    disable_log_stats=False,
    attention_config={"backend": "FLASH_ATTN"},
)

outputs = spec_llm.chat(test_prompts, sampling_params)
metrics = spec_llm.get_metrics()
acceptance_rate = compute_acceptance_rate(metrics)

print(f"\nMetrics:")
print(f"  Acceptance rate: {acceptance_rate:.4f}")
print(f"\nOutput: {outputs[0].outputs[0].text}")

del spec_llm
torch.cuda.empty_cache()
cleanup_dist_env_and_memory()
