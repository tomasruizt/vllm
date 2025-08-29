#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TORCH_PROFILER_DIR=./profiles

SPEC_CFG='{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3,"max_model_len":2048}'

vllm bench throughput \
  --model Qwen/Qwen3-4B \
  --speculative-config "$SPEC_CFG" \
  --dataset-name=hf \
  --dataset-path=likaixin/InstructCoder \
  --input-len=1000 \
  --output-len=10 \
  --num-prompts=10 \
  --max_num_seqs=20 \
  --max-model-len=2048 \
  --gpu_memory_utilization=0.6 \
  --print-acceptance-rate \
  --profile \
