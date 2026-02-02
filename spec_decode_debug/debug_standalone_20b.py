#!/usr/bin/env python3
"""Debug script to test draft model forward pass in isolation."""
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch
from vllm import LLM, SamplingParams

def test_draft_standalone():
    """Test the 20b model standalone (no speculative decoding)."""
    print("\n" + "="*80)
    print("Testing 20b model STANDALONE (no speculative decoding)")
    print("="*80 + "\n")

    test_prompts = [[{"role": "user", "content": "please repeat the word 'test' 10 times."}]]
    sampling_params = SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)

    llm = LLM(
        model="openai/gpt-oss-20b",
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        enforce_eager=True,
        attention_config={"backend": "FLASH_ATTN"},
    )

    outputs = llm.chat(test_prompts, sampling_params)
    print(f"\nOutput: {outputs[0].outputs[0].text}")
    print(f"Token IDs: {outputs[0].outputs[0].token_ids}")

    del llm
    torch.cuda.empty_cache()
    return outputs[0].outputs[0].token_ids

if __name__ == "__main__":
    token_ids = test_draft_standalone()
    print(f"\nExpected first token: 35644 (the correct answer)")
    print(f"Actual first token: {token_ids[0]}")
