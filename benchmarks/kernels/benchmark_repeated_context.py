# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark duplicate-context masking without loading the vLLM engine.

Pass --source to compare a saved pre-change watermark.py with the current file.
Times include every GPU kernel in repeated_context_mask, using CUDA graphs.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import statistics
import tempfile
from pathlib import Path

import torch

from vllm.triton_utils import triton


def load_mask(source, directory):
    tree = ast.parse(source.read_text())
    functions = [
        ast.get_source_segment(source.read_text(), node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and "repeated_context" in node.name
    ]
    # Include decorators, which ast.get_source_segment excludes.
    code = "import torch\nimport triton\nimport triton.language as tl\n"
    code += "\n\n".join(
        ("@triton.jit\n" if "_kernel(" in fn else "") + fn for fn in functions
    )
    path = Path(directory) / "mask_module.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location("mask_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(batch, length, width, capacity=None):
    prompt = 32
    tokens = torch.randint(
        100,
        32000,
        (batch, prompt + (capacity or length)),
        dtype=torch.int32,
        device="cuda",
    )
    contexts = torch.arange(1, width + 1, device="cuda").expand(batch, -1)
    # Half the requests match, including matches near the end of the scan.
    tokens[::2, prompt + length - width - 1 : prompt + length - 1] = contexts[::2]
    return (
        tokens,
        torch.arange(batch, dtype=torch.int32, device="cuda"),
        torch.full((batch,), prompt, dtype=torch.int32, device="cuda"),
        torch.full((batch,), prompt + length, dtype=torch.int32, device="cuda"),
        contexts.contiguous(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("vllm/v1/worker/gpu/sample/watermark.py")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[128, 1024, 8192, 32768]
    )
    parser.add_argument(
        "--uva-library",
        type=Path,
        help="Load vLLM's CUDA extension and use pinned-host token and prompt buffers",
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.uva_library:
        torch.ops.load_library(str(args.uva_library))
    results = []
    with tempfile.TemporaryDirectory() as directory:
        module = load_mask(args.source, directory)
        for width in args.widths:
            for batch in args.batches:
                for length in args.lengths:
                    if args.capacity is not None and args.capacity < length:
                        continue
                    inputs = make_inputs(batch, length, width, args.capacity)
                    # Keep pinned owners alive through correctness checks and timing.
                    pinned_owners = []
                    if args.uva_library:
                        inputs = list(inputs)
                        for index in (0, 2):
                            pinned = inputs[index].cpu().pin_memory()
                            pinned_owners.append(pinned)
                            inputs[index] = torch.ops._C.get_cuda_view_from_cpu_tensor(
                                pinned
                            )
                            assert inputs[index].is_cuda and pinned.is_pinned()
                            assert inputs[index].data_ptr() == pinned.data_ptr()
                    actual = module.repeated_context_mask(*inputs)
                    expected = torch.arange(batch, device="cuda") % 2 == 0
                    torch.testing.assert_close(actual, expected)
                    samples = [
                        1000
                        * triton.testing.do_bench_cudagraph(
                            lambda inputs=inputs: module.repeated_context_mask(*inputs),
                            rep=100,
                        )
                        for _ in range(3)
                    ]
                    for _ in range(20):
                        module.repeated_context_mask(*inputs)
                    torch.accelerator.synchronize()
                    events = [
                        (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        for _ in range(100)
                    ]
                    for start, end in events:
                        start.record()
                        module.repeated_context_mask(*inputs)
                        end.record()
                    torch.accelerator.synchronize()
                    eager_us = statistics.mean(
                        start.elapsed_time(end) * 1000 for start, end in events
                    )
                    result = dict(
                        batch=batch,
                        length=length,
                        width=width,
                        capacity=args.capacity or length,
                        us=statistics.median(samples),
                        samples_us=samples,
                        eager_mean_us=eager_us,
                    )
                    results.append(result)
                    print(json.dumps(result), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                triton=triton.__version__,
                source=str(args.source),
                source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                memory="uva" if args.uva_library else "cuda",
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
