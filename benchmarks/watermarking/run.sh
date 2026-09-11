#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Run from the repository root with its vLLM environment activated.
# Append --dry-run to preview the commands without starting a server.
export VLLM_USE_V2_MODEL_RUNNER=1

vllm bench sweep serve \
  --serve-cmd 'vllm serve Qwen/Qwen3.5-9B
    --language-model-only
    --max-model-len 4096
    --no-enable-prefix-caching
    --generation-config vllm
    --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}' \
  --bench-cmd 'vllm bench serve
    --model Qwen/Qwen3.5-9B
    --backend vllm --endpoint /v1/completions
    --dataset-name hf --dataset-path openai/gsm8k
    --hf-subset main --hf-split test --hf-output-len 512
    --num-prompts 200 --num-warmups 8
    --max-concurrency 8 --request-rate inf
    --temperature 1 --ignore-eos --seed 42' \
  --serve-params benchmarks/watermarking/serve_params.json \
  --num-runs 3 \
  --server-ready-timeout 1200 \
  --output-dir benchmarks/watermarking/results \
  "$@"
