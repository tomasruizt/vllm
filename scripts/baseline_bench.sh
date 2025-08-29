#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TORCH_PROFILER_DIR=./profiles

vllm bench throughput \
  --model Qwen/Qwen3-4B \
  --dataset-name=hf \
  --dataset-path=likaixin/InstructCoder \
  --input-len=1000 \
  --output-len=10 \
  --num-prompts=10 \
  --max_num_seqs=20 \
  --max-model-len=2048 \
  --gpu_memory_utilization=0.8 \
  --profile \
