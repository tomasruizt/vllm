#!/usr/bin/env python3
"""
Test script that:
1. First runs standalone 20b to prime the Triton cache
2. Then runs 120b/20b speculative decoding

This tests whether Triton kernel caching is the root cause.
"""

import os
import gc
import torch

os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

PROMPT = "User asks: \"please repeat the first 3 sentences of the declaration of independence, in correct english, capitalized correctly\"\n\nassistant:\n"

def clear_gpu():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def main():
    print("=" * 80)
    print("PHASE 1: Warmup with standalone 20b model")
    print("=" * 80)

    # First, run standalone 20b to prime the Triton cache
    from vllm.config import AttentionConfig
    warmup_llm = LLM(
        model="openai/gpt-oss-20b",
        max_model_len=1024,
        gpu_memory_utilization=0.4,  # Use less memory for warmup
        enforce_eager=True,
        disable_log_stats=True,
        attention_config=AttentionConfig(backend="FLASH_ATTN"),
    )

    # Run one inference to compile kernels
    warmup_params = SamplingParams(max_tokens=10, temperature=0.0)
    warmup_llm.generate([PROMPT], warmup_params)
    print("Warmup complete!")

    # Cleanup
    del warmup_llm
    clear_gpu()

    print("\n" + "=" * 80)
    print("PHASE 2: Now running 120b/20b speculative decoding")
    print("=" * 80)

    # Now run 120b/20b speculative decoding
    llm = LLM(
        model="openai/gpt-oss-120b",
        max_model_len=1024,
        enforce_eager=True,
        attention_config=AttentionConfig(backend="FLASH_ATTN"),
        speculative_config={
            "model": "openai/gpt-oss-20b",
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "max_model_len": 1024,
            "enforce_eager": True,
            "draft_tensor_parallel_size": 1,
            "max_num_seqs": 100,
        },
    )

    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

    outputs = llm.generate([PROMPT], sampling_params)

    for output in outputs:
        print("\nMetrics:")
        if hasattr(output, 'metrics') and output.metrics:
            print(f"  Acceptance rate: {output.metrics.spec_decode_metrics.draft_acceptance_rate:.4f}")
        print(f"Output: {output.outputs[0].text[:100]}")

    # Get final metrics
    if outputs and hasattr(outputs[0], 'metrics') and outputs[0].metrics:
        acceptance = outputs[0].metrics.spec_decode_metrics.draft_acceptance_rate
        print(f"\nFinal acceptance rate: {acceptance:.4f}")
        if acceptance > 0.5:
            print("SUCCESS: Acceptance rate is good after warmup!")
        else:
            print("FAILED: Acceptance rate still low even after warmup")
    else:
        print("\nNo metrics available")

if __name__ == "__main__":
    main()
