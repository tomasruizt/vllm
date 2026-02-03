#!/usr/bin/env python3
"""
Test script to verify if target and draft models share the same
static_all_moe_layers list, causing the draft model to use the
wrong MoE layer weights.
"""

import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig

PROMPT = "User asks: \"please repeat the first 3 sentences of the declaration of independence, in correct english, capitalized correctly\"\n\nassistant:\n"

def main():
    print("=" * 80)
    print("DEBUG: Checking MoE layer order in static_all_moe_layers")
    print("=" * 80)

    # Create the LLM with speculative decoding
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

    # Access the vllm_config
    vllm_config = llm.llm_engine.vllm_config
    compilation_config = vllm_config.compilation_config

    print("\n" + "=" * 80)
    print("STATIC_ALL_MOE_LAYERS:")
    print("=" * 80)

    all_moe_layers = compilation_config.static_all_moe_layers
    print(f"Total layers in static_all_moe_layers: {len(all_moe_layers)}")

    # Count target vs draft layers
    target_layers = [l for l in all_moe_layers if not l.startswith("draft_model")]
    draft_layers = [l for l in all_moe_layers if l.startswith("draft_model")]

    print(f"\nTarget model layers: {len(target_layers)}")
    print(f"Draft model layers: {len(draft_layers)}")

    print("\nFirst 10 layers in order:")
    for i, layer in enumerate(all_moe_layers[:10]):
        print(f"  [{i}] {layer}")

    print("\n...")

    print("\nLast 10 layers in order:")
    for i, layer in enumerate(all_moe_layers[-10:]):
        print(f"  [{len(all_moe_layers) - 10 + i}] {layer}")

    print("\n" + "=" * 80)
    print("STATIC_FORWARD_CONTEXT:")
    print("=" * 80)

    static_forward_context = compilation_config.static_forward_context
    print(f"Total entries: {len(static_forward_context)}")

    # Check if target and draft layers have different FusedMoE objects
    print("\nChecking if layer objects are unique:")

    layer_obj_ids = {}
    duplicates = []
    for layer_name, layer_obj in static_forward_context.items():
        obj_id = id(layer_obj)
        if obj_id in layer_obj_ids:
            duplicates.append((layer_name, layer_obj_ids[obj_id]))
        else:
            layer_obj_ids[obj_id] = layer_name

    if duplicates:
        print(f"  WARNING: Found {len(duplicates)} duplicate layer objects!")
        for dup in duplicates[:5]:
            print(f"    {dup[0]} shares object with {dup[1]}")
    else:
        print("  All layer objects are unique.")

    # Check first MoE layer for target and draft
    print("\n" + "=" * 80)
    print("COMPARING FIRST MOE LAYER:")
    print("=" * 80)

    target_first = "layers.0.experts"
    draft_first = "draft_model.layers.0.experts"

    if target_first in static_forward_context:
        target_moe = static_forward_context[target_first]
        print(f"\nTarget layer '{target_first}':")
        print(f"  Object ID: {id(target_moe)}")
        print(f"  local_num_experts: {getattr(target_moe, 'local_num_experts', 'N/A')}")
        if hasattr(target_moe, 'w13_weight'):
            w = target_moe.w13_weight
            print(f"  w13_weight shape: {w.shape if hasattr(w, 'shape') else 'N/A'}")
            print(f"  w13_weight id: {id(w)}")
    else:
        print(f"  Target layer '{target_first}' not found!")

    if draft_first in static_forward_context:
        draft_moe = static_forward_context[draft_first]
        print(f"\nDraft layer '{draft_first}':")
        print(f"  Object ID: {id(draft_moe)}")
        print(f"  local_num_experts: {getattr(draft_moe, 'local_num_experts', 'N/A')}")
        if hasattr(draft_moe, 'w13_weight'):
            w = draft_moe.w13_weight
            print(f"  w13_weight shape: {w.shape if hasattr(w, 'shape') else 'N/A'}")
            print(f"  w13_weight id: {id(w)}")
    else:
        print(f"  Draft layer '{draft_first}' not found!")

    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST:")
    print("=" * 80)
    print("""
When the draft model runs:
1. moe_layer_index starts at 0
2. Draft model's first MoE layer calls encode_layer_name() -> "from_forward_context"
3. get_layer_from_name() returns all_moe_layers[0] = '{}'
4. This is the TARGET model's layer, not the draft model's!

This means the draft model is using the WRONG weights!
""".format(all_moe_layers[0] if all_moe_layers else "N/A"))

    # Run inference to trigger the bug
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE...")
    print("=" * 80)

    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
    outputs = llm.generate([PROMPT], sampling_params)

    for output in outputs:
        print(f"\nOutput: {output.outputs[0].text[:100]}...")
        if hasattr(output, 'metrics') and output.metrics:
            print(f"Acceptance rate: {output.metrics.spec_decode_metrics.draft_acceptance_rate:.4f}")

if __name__ == "__main__":
    main()
