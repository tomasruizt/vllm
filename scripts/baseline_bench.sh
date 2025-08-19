#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm bench throughput \
  --model Qwen/Qwen3-4B \
  --dataset-name=hf \
  --dataset-path=likaixin/InstructCoder \
  --input-len=1000 \
  --output-len=100 \
  --num-prompts=100 \
  --max-model-len=2048 \
  --max_num_seqs=10 \
  --gpu_memory_utilization=0.8 \
  --enforce-eager
