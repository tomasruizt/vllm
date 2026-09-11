#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
export PORT=${PORT:-18001}

# Run from the repository root. Capture ten worker iterations after five
# initial iterations, with enough output tokens to keep decoding active.
bash benchmarks/watermarking/run.sh \
  --serve-params benchmarks/watermarking/profile_params.json \
  --bench-cmd 'vllm bench serve
    --model Qwen/Qwen3.5-9B
    --backend vllm --endpoint /v1/completions
    --dataset-name hf --dataset-path openai/gsm8k
    --hf-subset main --hf-split test --hf-output-len 128
    --num-prompts 8 --num-warmups 8
    --max-concurrency 8 --request-rate inf
    --temperature 1 --ignore-eos --seed 42 --profile'" --port $PORT" \
  --output-dir benchmarks/watermarking/results/profiles \
  --num-runs 1 \
  "$@"
