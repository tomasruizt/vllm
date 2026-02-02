# Speculative Decoding Divergence Investigation

## Problem Summary
- **120b/20b speculative decoding**: 0% acceptance rate
- **20b/20b speculative decoding**: 100% acceptance rate
- **Standalone 20b**: Produces correct output

## Key Finding (Latest)

**The weight scales are IDENTICAL between both setups, yet MoE outputs differ drastically.**

### Weight Scale Comparison (Draft Model Layer 0)
| Metric | 120b/20b | 20b/20b |
|--------|----------|---------|
| ws_sum | 2012478720.0 | 2012478720.0 |
| ws_mean | 118.678 | 118.678 |
| ws_std | 17.702 | 17.702 |
| ws_max | 126.0 | 126.0 |
| first_10 | [122, 121, 121, 121, 121, 121, 121, 122, 120, 121] | [122, 121, 121, 121, 121, 121, 121, 122, 120, 121] |

### MoE Output Comparison (Draft Model Layer 0, 79 tokens)
| Metric | 120b/20b | 20b/20b | Delta |
|--------|----------|---------|-------|
| mean | 0.0062 | 0.0254 | 4.1x |
| std | 0.2690 | **1.1545** | **4.3x** |
| min | -9.31 | -49.75 | 5.3x |
| max | 20.75 | **136.0** | **6.5x** |
| first_5 | [0.129, -0.030, 0.067, -0.011, 0.072] | [0.160, 0.122, -0.049, 0.125, 0.984] | Different |

## Evidence Chain

### Identical Inputs
- Router logits: mean=-0.1077, std=1.0264
- Top-k experts: [5, 9, 6, 18]
- PRE_MLP hidden states: mean=0.0012, std=0.3457

### Identical Weights & Scales
- w13_shape: [32, 2880, 5760]
- w2_shape: [32, 2880, 2880]
- ws_shape: [32, 90, 5760]
- Weight scale values: **IDENTICAL**

### Divergent Output
- POST_MLP outputs differ by 4-6x in std and max values

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
