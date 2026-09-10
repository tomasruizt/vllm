# Repeated-context kernel speedups

These results supersede the separate-process measurements for speedup comparisons. Original and current kernels use the exact same input tensors and pinned-memory allocations for each case. Measurement order alternates original/current, current/original, original/current. Each timing is 100 ms of CUDA-graph replay; the reported latency is the median of the three timings for that implementation. This reduces order effects and differences between separate host-memory allocations.

Hardware/software: NVIDIA H100 80GB HBM3, GPU 6 reserved through canhazgpu; PyTorch 2.13.0+cu130, Triton 3.7.1.

Original: the block-vectorized one-program-per-request kernel at PR head 4eab7a072. Current: one scan kernel with batch-dependent grid, bool output padded to whole atomic words, unconditional masked atomic OR through an int32 pointer, and no output conversion kernel. Programs per request for batches 1/32/256/1024 are 32/4/1/1.

Memory: history and prompt lengths are pinned-host UVA views; total lengths, request indices and contexts reside on the GPU. Each row has a 32-token prompt and 65,536-token output capacity. All requests in a case have the same history length; half match near the end. Both implementations assert the expected boolean mask on the shared inputs in every case.

Timings include zero-initialization and scan/atomic work for the current version. They exclude input preparation, compilation and Python allocation/dispatch. Speedup = original/current; below 1 means slower. These are mask-operation timings, not full sampler or serving timings. Repeated fixed-buffer replay does not reproduce model-induced cache pressure.

Original source SHA-256: `f366685ba3ddb1f6c4c520e703365a1c1c74595fc1a0ce8eade4d43d4163c732`.

Current source SHA-256: `544a34e2f02b150fdbe7fa03f98d82047115514c1a54204f9ff994bbc6685213`.

## Context width 4

| Batch | History | Original µs | Current µs | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 8.30 | 5.55 | 1.50× |
| 1 | 8192 | 25.45 | 5.77 | 4.41× |
| 1 | 32768 | 94.56 | 8.06 | 11.74× |
| 32 | 2048 | 10.04 | 11.21 | 0.90× |
| 32 | 8192 | 27.97 | 30.05 | 0.93× |
| 32 | 32768 | 105.88 | 106.15 | 1.00× |
| 256 | 2048 | 51.05 | 52.11 | 0.98× |
| 256 | 8192 | 199.38 | 200.25 | 1.00× |
| 256 | 32768 | 789.96 | 790.68 | 1.00× |
| 1024 | 2048 | 205.09 | 205.82 | 1.00× |
| 1024 | 8192 | 824.38 | 824.62 | 1.00× |
| 1024 | 32768 | 3244.83 | 3245.73 | 1.00× |

## Context width 16

| Batch | History | Original µs | Current µs | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 25.21 | 9.39 | 2.69× |
| 1 | 8192 | 91.18 | 9.77 | 9.34× |
| 1 | 32768 | 359.15 | 16.67 | 21.54× |
| 32 | 2048 | 26.65 | 16.21 | 1.64× |
| 32 | 8192 | 93.74 | 41.09 | 2.28× |
| 32 | 32768 | 362.70 | 135.94 | 2.67× |
| 256 | 2048 | 73.20 | 73.49 | 1.00× |
| 256 | 8192 | 277.15 | 277.46 | 1.00× |
| 256 | 32768 | 1085.63 | 1085.55 | 1.00× |
| 1024 | 2048 | 284.70 | 286.05 | 1.00× |
| 1024 | 8192 | 1127.89 | 1128.07 | 1.00× |
| 1024 | 32768 | 4463.87 | 4463.06 | 1.00× |

## Configuration retained after tuning

Retain `BLOCK=512` and `PROGRAMS_PER_REQ=max(1, min(32, 128 // n,
ceil(token_capacity / 512)))`. This is a conservative choice across context
widths, not the fastest configuration for every individual case. Large batches
retain one program per request. No dtype changes are required in consumers:
`repeated` is a padded boolean allocation, updated through int32 atomic words.

The final comparison used the same pinned allocations, alternating measurement
order, and two 30 ms CUDA-graph measurements per configuration on an H100 with
132 SMs. Batch-one results (microseconds):

| Context width | History | BLOCK=512, P=32 (retained) | BLOCK=256, P=132 | BLOCK=128, P=264 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 2048 | 5.54 | 7.02 | 7.18 |
| 4 | 32768 | 8.17 | 8.06 | 8.59 |
| 16 | 2048 | 9.58 | 9.95 | 9.29 |
| 16 | 32768 | 16.68 | 12.41 | 13.14 |

More parallelism helps the long width-16 case but costs about 27% in the short
width-4 case with BLOCK=256. Doubling the program budget also regressed batch-32,
width-16, 32768-token history from 135.81 to 146.25 us. These tradeoffs motivated
retaining the existing policy rather than introducing further special cases.

Nsight Compute measured average SM active cycles of 20.36%, 69.37%, and 70.34%
for the three batch-one configurations at width 16 and history 32768. More
blocks improve SM activity, but do not guarantee lower latency or full GPU
utilization. The retained policy prioritizes the measured latency tradeoff.

## Validation

The retained kernel passed 15 focused existing watermarking/sampler tests.
An external scratch check covered 44 boolean atomic/padding/contention cases;
Compute Sanitizer reported zero errors. No repository tests were modified.
These are kernel microbenchmarks, not model quality or end-to-end serving evals.

## Reproduce individual source measurements

From the repository root, with the vLLM environment active and a reserved GPU:

```bash
git show 4eab7a072:vllm/v1/worker/gpu/sample/watermark.py > /tmp/watermark-original.py
/home/tomasruizt/.venv/bin/python benchmarks/kernels/benchmark_repeated_context.py \
  --source /tmp/watermark-original.py --output /tmp/original.json \
  --widths 4 16 --batches 1 32 256 1024 --lengths 2048 8192 32768 \
  --capacity 65536 --uva-library /path/to/vllm/_C_stable_libtorch.abi3.so
/home/tomasruizt/.venv/bin/python benchmarks/kernels/benchmark_repeated_context.py \
  --output /tmp/current.json \
  --widths 4 16 --batches 1 32 256 1024 --lengths 2048 8192 32768 \
  --capacity 65536 --uva-library /path/to/vllm/_C_stable_libtorch.abi3.so
```

The commands above run each source separately. The tables use paired input
allocations as described above; separate-process ratios can drift due to host
memory placement. The benchmark exposes `load_mask` and `make_inputs` for paired
measurement: load both sources into separate temporary directories, create inputs
once per case, retain the pinned owners, and alternate timing the two functions.
