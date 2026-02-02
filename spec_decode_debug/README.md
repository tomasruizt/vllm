# Speculative Decoding Debug Package

This folder contains scripts and logs for debugging the speculative decoding divergence issue where:
- 120b/20b setup produces 0% acceptance rate
- 20b/20b setup produces 100% acceptance rate

## Quick Start

```bash
# Kill any existing GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}

# Run 120b/20b test (broken)
PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python spec_decode_debug/debug_120b_20b.py

# Run 20b/20b test (working)
PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python spec_decode_debug/debug_20b_20b.py

# Run standalone 20b test (working)
PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python spec_decode_debug/debug_standalone_20b.py
```

## Files

- `HYPOTHESIS.md` - Latest hypothesis with evidence
- `HYPOTHESIS_ORIGINAL.md` - Initial hypothesis before detailed investigation
- `debug_120b_20b.py` - Test script for 120b target + 20b draft
- `debug_20b_20b.py` - Test script for 20b target + 20b draft
- `debug_standalone_20b.py` - Test script for 20b standalone (no spec decode)
- `logs/` - Captured debug logs

## Key Finding

**Weight scales are IDENTICAL between both setups, yet MoE outputs differ by 4-6x**

This suggests the issue is in:
1. Triton kernel autotuning state
2. FlexCtx contents
3. Intermediate buffer reuse
4. Routing data computation

## Debug Logging Locations

Debug logging was added to:
- `vllm/model_executor/models/gpt_oss.py` - MLPBlock.forward()
- `vllm/model_executor/layers/quantization/mxfp4.py` - apply_monolithic()

Look for logs starting with:
- `DEBUG MLP ROUTER:` - Router logits and top-k experts
- `DEBUG MLP WEIGHTS:` - Weight shapes and types
- `DEBUG MLP WEIGHT_SCALE:` - Weight scale values (fingerprint)
- `DEBUG MLP OUTPUT:` - MoE output statistics
- `DEBUG TRITON PATH:` - Triton kernel entry points

## Next Steps

1. Add logging to `routing()` function in triton_kernels
2. Compare FlexCtx.rhs_data between setups
3. Log intermediate tensor after first matmul
4. Try disabling triton autotuning with `TRITON_CACHE_DIR`
