# Speculative Decoding Divergence Investigation

## Problem
- **120b/20b speculative decoding**: 0% acceptance rate
- **20b/20b speculative decoding**: 100% acceptance rate
- **Standalone 20b**: Produces correct output

## Hypothesis

The divergence in draft model outputs between 120b/20b and 20b/20b setups occurs in the **MLP (MoE) layer** after layer 0's attention output.

## Evidence

### Layer 0 Computations - IDENTICAL
| Metric | 120b/20b | 20b/20b |
|--------|----------|---------|
| Layer 0 key input mean | 0.0995 | 0.0995 |
| Layer 0 key input std | 7.7164 | 7.7164 |
| Layer 0 attn output mean | -0.0033 | -0.0033 |
| Layer 0 attn output std | 0.5563 | 0.5563 |

### Layer 1 Computations - DIVERGENT
| Metric | 120b/20b | 20b/20b | Difference |
|--------|----------|---------|------------|
| Layer 1 key input mean | -0.8059 | -0.8285 | 2.7% |
| Layer 1 key input std | 9.8004 | 9.9984 | 2.0% |
| Layer 1 attn output std | 0.5581 | 0.4866 | 14.7% |

### Final Output
| Setup | First Draft Token | Expected Token |
|-------|-------------------|----------------|
| 120b/20b | 6151 | 35644 |
| 20b/20b | 35644 | 35644 |
| Standalone 20b | 35644 | 35644 |

## Processing Chain Analysis

```
Layer 0 attention output [IDENTICAL]
  → o_proj (linear layer) - same weights, same input
  → post_attention_layernorm (fused RMSNorm + residual)
  → MLP (FusedMoE) [SUSPECTED DIVERGENCE POINT]
  → Layer 1 input_layernorm (fused RMSNorm + residual)
  → QKV projection → key [DIVERGENT]
```

## KV Cache Group Structure

### 120b/20b (6 groups)
- gid 0-3: Target model (120b)
- gid 4-5: Draft model (20b)
- Draft layer 0 slot_mapping: starts at 336
- Draft layer 1 slot_mapping: starts at 416

### 20b/20b (4 groups)
- gid 0-1: Target model (20b)
- gid 2-3: Draft model (20b)
- Draft layer 0 slot_mapping: starts at 176
- Draft layer 1 slot_mapping: starts at 256

## Possible Root Causes

1. **FusedMoE state contamination**: Triton autotuning cache or buffers affected by target model run
2. **Memory layout differences**: Different KV cache allocation affecting MoE computation
3. **Numerical precision accumulation**: Small differences in MoE routing/computation
4. **RMSNorm fused residual**: Issue with residual stream management

## Relevant Log Files
- `/home/shadeform/code/logs/debug_120b_20b_20260202_161937.log`
- `/home/shadeform/code/logs/debug_20b_20b_20260202_162650.log`
- `/home/shadeform/code/logs/debug_draft_standalone_20260202_163656.log`

## Reproduction
```bash
# 120b/20b - 0% acceptance
PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python /home/shadeform/code/vllm/debug_120b_20b.py

# 20b/20b - 100% acceptance
PYTHONUNBUFFERED=1 VLLM_BATCH_INVARIANT=1 python /tmp/debug_20b_20b.py
```

## Next Steps
1. Add debug logging to MLP output in gpt_oss.py
2. Check FusedMoE for persistent state
3. Verify VLLM_BATCH_INVARIANT is respected in draft model's MoE path
