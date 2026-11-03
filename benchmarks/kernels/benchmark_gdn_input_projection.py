# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Qwen3.6 BF16 projections with cold rotating weights and CUDA graphs."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.utils.multi_stream_utils import execute_in_parallel


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Write metadata.json and projection.csv here."
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    weights = [
        torch.randn(12352, 2048, device="cuda", dtype=torch.bfloat16) / 64
        for _ in range(4)
    ]
    stream = torch.cuda.Stream()
    events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in weights]
    samples = []
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "weight_sets": 4,
        "dtype": "bf16",
        "hidden": 2048,
        "qkvz": 12288,
        "ba": 64,
        "replays": 100,
        "timing": "CUDA events over graphs, rotating 202 MB weights",
    }
    print(json.dumps(metadata), flush=True)
    for tokens in [1, 4, 8, 32, 128, 256]:
        xs = [
            torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
            for _ in weights
        ]
        calls = {
            name: make_call(name, xs, weights, stream, events)
            for name in ["separate", "overlap", "fused"]
        }
        reference = calls["separate"]()
        for call in calls.values():
            for actual, expected in zip(call(), reference):
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, rtol=0.016, atol=0.016)
        graphs = {name: capture(call) for name, call in calls.items()}
        for trial in range(5):
            order = list(graphs) if trial % 2 == 0 else list(reversed(graphs))
            for name in order:
                graph, outputs = graphs[name]
                us = time_graph(graph)
                samples.append(
                    {"tokens": tokens, "method": name, "trial": trial, "us": us}
                )
        for name in calls:
            values = [
                r["us"]
                for r in samples
                if r["tokens"] == tokens and r["method"] == name
            ]
            median = statistics.median(values)
            print(f"tokens={tokens} method={name} median_us={median:.3f}", flush=True)
    if args.output_dir is not None:
        save_results(args.output_dir, metadata, samples)


def save_results(output_dir: Path, metadata: dict, samples: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output_dir / "projection.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def make_call(name, xs, weights, stream, events):
    def call():
        return [
            project(name, x, w, stream, event)
            for x, w, event in zip(xs, weights, events)
        ]

    return call


def project(name, x, weight, stream, events):
    if name == "fused":
        return F.linear(x, weight).split([12288, 64], dim=-1)
    if name == "separate":
        return F.linear(x, weight[:12288]), F.linear(x, weight[12288:])
    start, done = events
    qkvz, (ba,) = execute_in_parallel(
        lambda: F.linear(x, weight[:12288]),
        [lambda: F.linear(x, weight[12288:])],
        start,
        [done],
        [stream],
        enable=torch.cuda.is_current_stream_capturing(),
    )
    return qkvz, ba


def capture(call):
    call()
    call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = call()
    torch.accelerator.synchronize()
    return graph, outputs


def time_graph(graph) -> float:
    for _ in range(10):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / 400


if __name__ == "__main__":
    main()
