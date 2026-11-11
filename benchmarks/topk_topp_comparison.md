# Top-k/top-p masking: Triton vs. FlashInfer

Kernel time in µs on B200, V=248,320, synthetic Zipf logits (rank^-1.1, rows shuffled). Each call restores the input and flushes the caches outside the timed region, so only the masking call is timed.

| case | rows | main Triton | new Triton | fi-fastpath | fi-naive-renorm | fi-mask |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top-k 50 | 8 | 232.4 | **26.6** | 29.7 | 137.3 | 44.6 |
| | 64 | 238.2 | **49.6** | 87.0 | 272.3 | 120.0 |
| | 512 | 1009.0 | **222.4** | 388.2 | 1498.0 | 679.1 |
| | 2048 | 3511.7 | **786.1** | 1364.3 | 5760.4 | 2659.1 |
| top-k 100 + top-p 0.9 | 8 | 283.2 | 29.1 | **27.6** | 297.3 | 309.3 |
| | 64 | 310.5 | **51.7** | 81.0 | 677.1 | 694.5 |
| | 512 | 1152.8 | **229.9** | 383.1 | 3696.1 | 3576.0 |
| | 2048 | 4020.0 | **813.2** | 1370.2 | 13899.3 | 12888.9 |
| top-p 0.9 | 8 | 107.2¹ | **77.3** | main Triton | 256.3 | 285.3 |
| | 64 | 617.5¹ | **167.2** | main Triton | 582.0 | 807.4 |
| | 512 | 1828.5¹ | **790.3** | main Triton | 3293.7 | 4303.0 |
| | 2048 | 6610.0¹ | **2829.8** | main Triton | 12238.0 | 15898.2 |

¹ Not exact (fails the PyTorch reference check). Every other number passes it.

Implementations (all return the masked logits):

- **main Triton:** the current `apply_top_k_top_p_triton` on main (`topk_topp_triton_main.py`).
- **new Triton:** the kernel of #66792.
- **fi-fastpath:** the FlashInfer path of #65649 (`topk_topp_flashinfer_fastpath.py`): FlashInfer's top-k selects up to 256 candidates, then a Triton kernel cuts them at k and p. It applies only when every row has top-k ≤ 256, and otherwise falls back to main's Triton kernel.
- **fi-naive-renorm:** softmax, then FlashInfer's `top_k_renorm_probs` and `top_p_renorm_probs`, then the logits where the probability is nonzero.
- **fi-mask:** FlashInfer's `top_k_mask_logits` on the logits, then (with top-p) softmax, `top_p_renorm_probs` and the logits where the probability is nonzero.

To reproduce, from the root of this branch with vLLM installed from it (torch 2.14.0+cu130, Triton 3.8.0, FlashInfer 0.7.3, driver 610.43.02), on one GPU:

```bash
python benchmarks/benchmark_topk_topp.py \
  --batch-sizes 8 64 512 2048 --vocab-sizes 131072 248320 \
  --scenarios topk_whole topp_whole topk_topp_whole \
  --implementations triton triton-main fi-fastpath fi-naive-renorm flashinfer-mask \
  --warmup-iters 5 --benchmark-iters 20 --check --quiet --output results/topk_topp_comparison.jsonl
```

The table shows the `topk_whole` (top-k 50), `topk_topp_whole` (top-k 100, top-p 0.9) and `topp_whole` (top-p 0.9) rows at V=248,320, mean time per call over the 20 timed iterations.

V=131,072 shows the same ranking.
