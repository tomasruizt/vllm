#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SPEC_CFG='{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3,"max_model_len":2048}'

vllm bench throughput \
  --model Qwen/Qwen3-4B \
  --dataset-name=hf \
  --dataset-path=likaixin/InstructCoder \
  --input-len=1000 \
  --output-len=100 \
  --num-prompts=100 \
  --max_num_seqs=10 \
  --max-model-len=2048 \
  --speculative-config "$SPEC_CFG" \
  --gpu_memory_utilization=0.8 \
  --print-acceptance-rate \
  --enforce-eager
