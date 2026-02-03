# Speculative Decoding Divergence Investigation

## Problem Summary
- **120b/20b speculative decoding**: 0% acceptance rate
- **20b/20b speculative decoding**: 100% acceptance rate
- **Standalone 20b**: Produces correct output

## ROOT CAUSE IDENTIFIED (2026-02-03 17:45)

**The draft model is using the TARGET model's MoE layer weights instead of its own!**

### Evidence

When MoE layers are registered during model loading:
- Target model (120b) layers: `model.layers.0.mlp.experts` through `model.layers.35.mlp.experts` (indices 0-35)
- Draft model (20b) layers: `draft_model.model.layers.0.mlp.experts` through `draft_model.model.layers.23.mlp.experts` (indices 36-59)

All 60 layers are stored in `static_all_moe_layers` list.

**During forward pass with `enforce_eager=True`:**
1. Each MoE layer calls `encode_layer_name()` which returns `"from_forward_context"` (because `all_moe_layers is not None`)
2. `get_layer_from_name("from_forward_context")` uses `moe_layer_index` counter to lookup layer
3. When draft model forward is called, `moe_layer_index` starts at 0
4. `all_moe_layers[0]` = `model.layers.0.mlp.experts` (TARGET model's layer!)
5. The draft model ends up using indices 0-23 (target model layers) instead of 36-59 (draft model layers)

### Log Evidence

```
DEBUG FusedMoE REGISTERED: prefix=draft_model.model.layers.0.mlp.experts, total_layers=37
DEBUG FusedMoE REGISTERED: prefix=draft_model.model.layers.1.mlp.experts, total_layers=38
...
DEBUG get_layer_from_name: original=from_forward_context, resolved=model.layers.0.mlp.experts, moe_layer_index=-1, total_layers=60
DEBUG get_layer_from_name: original=from_forward_context, resolved=model.layers.1.mlp.experts, moe_layer_index=0, total_layers=60
```

The draft model is resolving to `model.layers.X` (target) instead of `draft_model.model.layers.X` (draft)!

### Why This Happens

#### Background: The Counter Mechanism for torch.compile

When using `torch.compile` (`enforce_eager=False`), you cannot hardcode string constants (like layer names) into the compiled graph - it causes cold start issues. The MoE custom ops (`vllm.moe_forward`) need to know which layer they're operating on to access the correct weights.

The workaround is a counter-based lookup:
1. `ForwardContext` stores `all_moe_layers` (list of layer names) and `moe_layer_index` (counter starting at 0)
2. Instead of passing layer names, MoE layers return `"from_forward_context"`
3. `get_layer_from_name()` looks up `all_moe_layers[moe_layer_index]` and increments the counter
4. This assumes MoE layers execute in consistent order during each forward pass

#### The Bug

The issue is in `vllm/model_executor/layers/fused_moe/layer.py` in the `encode_layer_name()` function:

```python
def encode_layer_name() -> str:
    if (
        is_forward_context_available()
        and get_forward_context().all_moe_layers is not None
    ):
        return "from_forward_context"  # <-- This is the problem!
    return self.layer_name  # <-- This would be correct
```

When `all_moe_layers` is set (which happens when both models share the same `vllm_config`), ALL MoE calls use the `moe_layer_index` counter instead of their actual layer name. But the counter doesn't know which MODEL is calling!

When draft model starts its forward pass:
1. New `ForwardContext` created with `moe_layer_index = 0`
2. Draft layer 0 returns `"from_forward_context"`
3. `get_layer_from_name()` looks up `all_moe_layers[0]` → **target's layer**, not draft's!
4. Draft model uses wrong weights → wrong outputs → 0% acceptance

### Fix Implemented (2026-02-03 17:47)

Modified `encode_layer_name()` in `vllm/model_executor/layers/fused_moe/layer.py` to bypass the counter mechanism for draft model layers:

```python
def encode_layer_name() -> str:
    # IMPORTANT: For draft model layers in speculative decoding, we must
    # return the actual layer name directly. The "from_forward_context"
    # mechanism relies on a counter that doesn't account for multiple
    # models sharing the same vllm_config.
    if "draft_model" in self.layer_name:
        return self.layer_name
    if (
        is_forward_context_available()
        and get_forward_context().all_moe_layers is not None
    ):
        return "from_forward_context"
    return self.layer_name
```

**Result: Acceptance rate improved from 0% to 66.67%!**

Before fix:
- Layer resolution: `model.layers.0.mlp.experts` (TARGET model's weights)
- Acceptance rate: 0%

After fix:
- Layer resolution: `draft_model.model.layers.0.mlp.experts` (DRAFT model's weights)
- Acceptance rate: 66.67%

### Verification (2026-02-03 17:48)

- **120b/20b**: Acceptance rate improved from **0%** to **66.67%** ✅
- **20b/20b**: Acceptance rate remains at **100%** (no regression) ✅

The fix correctly identifies draft model layers by checking for "draft_model" in the layer name prefix, and bypasses the broken counter mechanism that was causing draft model MoE layers to use target model weights.

### Why The Fix Works

The fix is safe because:
1. **Draft models always run with `enforce_eager=True`** - they don't use torch.compile
2. Since there's no compiled graph, hardcoding the layer name string is fine
3. Target models can still use the counter mechanism for torch.compile compatibility

**Future consideration**: If draft models ever need to support `torch.compile`, this fix would need to be revisited - perhaps by maintaining separate counters or separate `all_moe_layers` lists for target vs draft models.

## Key Finding (Previous - 2026-02-03 17:03)

**EVERYTHING is IDENTICAL between both setups, yet MoE outputs differ drastically!**

### Verified Identical Components (Draft Model Layer 0, Prefill, 79 tokens)

| Component | 120b/20b | 20b/20b | Match? |
|-----------|----------|---------|--------|
| **PRE_MLP hidden states** | | | |
| first_5 | [0.1796875, -0.1357421875, -0.416015625, -0.1279296875, -0.0556640625] | [0.1796875, -0.1357421875, -0.416015625, -0.1279296875, -0.0556640625] | ✅ IDENTICAL |
| last_5 | [-0.1796875, -0.408203125, -0.54296875, -0.2734375, 0.1884765625] | [-0.1796875, -0.408203125, -0.54296875, -0.2734375, 0.1884765625] | ✅ IDENTICAL |
| **Router decisions** | | | |
| topk_indices | [5, 9, 6, 18] | [5, 9, 6, 18] | ✅ IDENTICAL |
| **Weight scales** | | | |
| ws_sum | 2012478720.0 | 2012478720.0 | ✅ IDENTICAL |
| first_10 | [122, 121, 121, 121, 121, 121, 121, 122, 120, 121] | same | ✅ IDENTICAL |
| **Raw quantized weights** | | | |
| w13 first_32bytes | [16, 66, 170, 84, 69, 33, 225, 67, 17, 17, 10, 18, 209, 161, 35, 226, 80, 89, 146, 137, 108, 136, 172, 218, 169, 82, 150, 72, 184, 142, 96, 25] | same | ✅ IDENTICAL |

### MoE Output Comparison - WHERE DIVERGENCE OCCURS

| Metric | 120b/20b | 20b/20b | Delta |
|--------|----------|---------|-------|
| mean | 0.0062 | 0.0255 | 4.1x |
| std | **0.2690** | **1.1531** | **4.3x** |
| min | -9.31 | -49.75 | 5.3x |
| max | 20.75 | **136.0** | **6.5x** |
| first_5 | [0.129, -0.030, 0.067, -0.011, 0.072] | [0.160, 0.122, -0.049, 0.125, 0.984] | Different |

### Cascade Effect

The decode step shows different **embedding inputs** because the draft predicted wrong tokens:
- 120b/20b decode INPUT: `[0.466796875, 0.35546875, -1.1015625, ...]`
- 20b/20b decode INPUT: `[1.3203125, -0.51171875, -1.2734375, ...]`

This confirms the **root cause is in the prefill MoE output** - wrong output → wrong token → chain of errors.

### Buffer Zeroing Test Result

**VLLM_ZERO_MOE_BUFFERS=1 did NOT fix the issue** - still 0% acceptance rate.

This rules out the buffer contamination hypothesis.

## Narrowed Hypothesis (Updated 2026-02-03 17:15)

All configuration state is also IDENTICAL:
- `_opt_flags_constraints = {'split_k': 1}` in BOTH setups
- `_opt_flags = None` in BOTH setups
- `enforce_bitwise_invariance = False` in BOTH setups
- `FlexCtx.rhs_data = InFlexData(dtype=None, scale=None)` in BOTH setups

Since inputs, weights, scales, raw quantized weights, AND all configuration state are identical, the divergence MUST be in:

### Additional Verified Identical Components
- `num_warps = 8` for both target and draft model
- `is_batched_moe = False` for both
- `enforce_bitwise_invariance` has no effect

### Key Observation
- **Standalone 20b** produces: mean=0.0256, std=1.1565, max=136.0
- **20b/20b draft** produces: mean=0.0255, std=1.1531, max=136.0 (MATCHES standalone)
- **120b/20b draft** produces: mean=0.0062, std=0.2690, max=20.75 (4x smaller!)

The presence of the 120b target model somehow affects the draft model's MoE computation, even though all traced state is identical.

### Additional Verified Identical Components (2026-02-03 17:32)
- Tensor strides: `(8294400, 1, 5760)` in both setups
- Tensor contiguity: `is_contiguous=False` in both
- First 32 bytes: Identical

### Remaining Hypotheses
1. **Triton kernel cache poisoning** - Compiled kernels may be cached with wrong parameters when 120b runs first
2. **PTX/CUBIN generation differences** - Different compilation paths for different model sizes
3. **CUDA memory state** - Hidden GPU state affected by running 120b model first
4. **TMA descriptor caching** - Tensor Memory Accelerator descriptors may be cached incorrectly

## Critical Finding (2026-02-03 17:37)

**Running 120b then 20b as STANDALONE models (no speculative decoding) produces IDENTICAL outputs!**

This means:
- The Triton kernel caching is NOT the issue
- The global opt_flags state is NOT the issue
- The issue is **SPECIFIC to the speculative decoding framework**

The problem must be in how:
1. The draft model shares state with the target model during spec decode
2. KV cache management between target and draft
3. The speculative decoding worker's handling of the draft model

## Summary of Investigation

### What We Verified as IDENTICAL
1. PRE_MLP hidden states (first_5, last_5 values)
2. Router logits and top-k indices
3. Weight scales (sum, mean, std, first_10)
4. Raw quantized weights (first_32bytes of uint8 data)
5. Tensor strides and contiguity
6. opt_flags_constraints (`{'split_k': 1}`)
7. _opt_flags (`None`)
8. enforce_bitwise_invariance (`False`)
9. FlexCtx.rhs_data (`dtype=None, scale=None`)
10. num_warps (`8`)
11. is_batched_moe (`False`)

### The Divergence Point
Despite ALL the above being identical, the MLP OUTPUT differs:
- **Correct (standalone/20b-20b)**: mean=0.0255, std=1.15, max=136
- **Wrong (120b/20b)**: mean=0.0062, std=0.27, max=20.75 (4x smaller!)

### Test Results
- `VLLM_ZERO_MOE_BUFFERS=1`: No effect
- `enforce_bitwise_invariance=True`: No effect
- Fresh Triton cache (`TRITON_CACHE_DIR=/tmp/...`): No effect

### Likely Root Cause
The issue is deep in the Triton `matmul_ogs` kernel execution path. Since all Python-level state is identical, the difference must be in:
1. GPU-side kernel compilation/caching behavior
2. TMA (Tensor Memory Accelerator) descriptor state
3. Undocumented Triton kernel internal state

### Next Steps for Further Investigation
1. Add TRITON_DEBUG=1 logging to see kernel compilation details
2. Compare kernel signatures between 120b and 20b model calls
3. Check TMA descriptor creation in the matmul_ogs kernel
4. Bisect the triton_kernels code to find the divergence point

## Evidence Chain

### Identical Inputs
- Router logits: mean=-0.1077, std=1.0264
- Top-k experts: [5, 9, 6, 18]
- PRE_MLP hidden states: mean=0.0012, std=0.3457, first_5/last_5 verified identical

### Identical Weights & Scales & Raw Data
- w13_shape: [32, 2880, 5760]
- w2_shape: [32, 2880, 2880]
- ws_shape: [32, 90, 5760]
- Weight scale values: **IDENTICAL**
- Raw quantized weight bytes: **IDENTICAL**

### Divergent Output
- POST_MLP outputs differ by 4-6x in std and max values
- Root cause is in the Triton matmul_ogs kernel execution

## Hypothesis

Since weights and scales are identical, the divergence must be in:

1. **Triton kernel autotuning state** - The `matmul_ogs` kernel from triton_kernels may have internal state affected by running the target model first
2. **Flex context** - The `FlexCtx(rhs_data=w13_flex)` used in `PrecisionConfig` may have state that differs
3. **Intermediate buffer reuse** - `_resize_cache` or intermediate tensors may be contaminated
4. **Routing data computation** - The `routing()` function's output may differ despite same inputs

## Next Steps to Investigate

1. **Compare FlexCtx contents** - Log the `flex_ctx.rhs_data` values in both setups
2. **Add logging to routing()** - Check if `RoutingData` differs between runs
3. **Check intermediate_cache** - Log the intermediate tensor after first matmul
4. **Disable triton autotuning** - See if `TRITON_CACHE_DIR` affects results
5. **Run with torch.backends.cudnn.deterministic=True**

## New Debug Logging Added (2024-02-03)

Added detailed logging in `triton_kernel_fused_experts()` to capture:
- Hidden states input statistics
- Routing data (n_expts_tot, n_expts_act, gate_scal)
- Gather/scatter indices
- **Intermediate cache state BEFORE matmul** (to detect stale data contamination)
- **Output tensor state BEFORE matmul** (to detect stale data contamination)
- Intermediate cache state AFTER first matmul
- Output tensor state AFTER second matmul

### Buffer Contamination Theory

The `_resize_cache()` function reuses buffers **without zeroing them**:
```python
def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    return x.flatten()[: prod(v)].view(*v)  # No zeroing!
```

If the target model (120b) runs first and leaves stale data in GPU memory, the draft model (20b) may reuse those buffers with contaminated values.

### Experimental Fix

Set `VLLM_ZERO_MOE_BUFFERS=1` to test if zeroing buffers before use fixes the issue:
```bash
VLLM_ZERO_MOE_BUFFERS=1 PYTHONUNBUFFERED=1 python spec_decode_debug/debug_120b_20b.py
```

### How to Run New Debug Tests

```bash
# Kill any existing GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}

# Run with detailed logging (look for DEBUG FUSED_EXPERTS lines)
PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python spec_decode_debug/debug_120b_20b.py 2>&1 | tee debug_120b_20b_buffers.log

# Test the buffer zeroing fix
VLLM_ZERO_MOE_BUFFERS=1 PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python spec_decode_debug/debug_120b_20b.py 2>&1 | tee debug_120b_20b_zeroed.log

# Compare key DEBUG lines
grep "DEBUG FUSED_EXPERTS" debug_120b_20b_buffers.log
```

## Reproduction

```bash
# 120b/20b - 0% acceptance (BROKEN)
PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python spec_decode_debug/debug_120b_20b.py

# 20b/20b - 100% acceptance (WORKING)
PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python spec_decode_debug/debug_20b_20b.py
```

## Relevant Files Modified

- `vllm/model_executor/models/gpt_oss.py` - Added debug logging in MLPBlock.forward()
- `vllm/model_executor/layers/quantization/mxfp4.py` - Added debug logging in apply_monolithic()

## Log Files

Key log files in `/home/shadeform/code/logs/`:
- `debug_120b_20b_triton3_*.log` - Latest 120b/20b run with weight scale logging
- `debug_20b_20b_triton3_*.log` - Latest 20b/20b run with weight scale logging

## Architecture Notes

### Triton MoE Path
```
MLPBlock.forward()
  → self.router(x) - linear layer for gating
  → self.experts(hidden_states=x, router_logits=g)
    → FusedMoE.forward_impl()
      → Mxfp4MoEMethod.apply_monolithic()
        → triton_kernel_moe_forward()
          → routing() - computes RoutingData, gather_idx, scatter_idx
          → triton_kernel_fused_experts()
            → matmul_ogs(x, w1, ..., precision_config=quant_config.w1_precision)
            → matmul_ogs(intermediate, w2, ..., precision_config=quant_config.w2_precision)
```

### Key Data Structures
- `PrecisionConfig(weight_scale=..., flex_ctx=FlexCtx(rhs_data=...))`
- `FusedMoEQuantConfig(_w1=FusedMoEQuantDesc("mxfp4", ..., scale=PrecisionConfig))`
- `RoutingData`, `GatherIndx`, `ScatterIndx` from triton_kernels.routing
