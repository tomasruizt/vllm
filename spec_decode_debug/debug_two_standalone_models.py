#!/usr/bin/env python3
"""
Test if 20b model output is affected by 120b model initialization/execution.
This tests WITHOUT speculative decoding to isolate the issue.

Test cases:
1. Run 20b standalone (baseline)
2. Initialize 120b, run it, then initialize 20b and run it
3. Compare outputs
"""

import os
import gc
import torch

os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig

PROMPT = "User asks: \"please repeat the first 3 sentences of the declaration of independence, in correct english, capitalized correctly\"\n\nassistant:\n"

def clear_gpu():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def run_model(model_name, prefix=""):
    """Run a model and return the output"""
    print(f"\n{'='*80}")
    print(f"{prefix}Running model: {model_name}")
    print(f"{'='*80}")

    llm = LLM(
        model=model_name,
        max_model_len=1024,
        enforce_eager=True,
        attention_config=AttentionConfig(backend="FLASH_ATTN"),
    )

    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
    outputs = llm.generate([PROMPT], sampling_params)

    output_text = outputs[0].outputs[0].text
    print(f"{prefix}Output: {output_text[:100]}...")

    # Cleanup
    del llm
    clear_gpu()

    return output_text

def main():
    print("=" * 80)
    print("TEST 1: Run 20b STANDALONE (baseline)")
    print("=" * 80)

    baseline_output = run_model("openai/gpt-oss-20b", prefix="[BASELINE] ")

    print("\n" + "=" * 80)
    print("TEST 2: Run 120b first, then 20b")
    print("=" * 80)

    # Run 120b first
    _ = run_model("openai/gpt-oss-120b", prefix="[120b FIRST] ")

    # Now run 20b after 120b
    after_120b_output = run_model("openai/gpt-oss-20b", prefix="[20b AFTER 120b] ")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\nBaseline 20b output:      {baseline_output[:80]}...")
    print(f"20b after 120b output:    {after_120b_output[:80]}...")

    if baseline_output == after_120b_output:
        print("\n✅ SUCCESS: Outputs are IDENTICAL")
    else:
        print("\n❌ FAILURE: Outputs DIFFER!")
        print(f"\nBaseline length: {len(baseline_output)}")
        print(f"After 120b length: {len(after_120b_output)}")

        # Find first difference
        for i, (a, b) in enumerate(zip(baseline_output, after_120b_output)):
            if a != b:
                print(f"First difference at position {i}: '{a}' vs '{b}'")
                print(f"Context: ...{baseline_output[max(0,i-10):i+20]}...")
                break

if __name__ == "__main__":
    main()
