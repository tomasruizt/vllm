# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request adaptive-verification trace for offline allocation replay.

Experiment-only. Enabled by `AV_TRACE_PATH` (JSONL output). With
`AV_TRACE_VERIFY_ALL=1` every scheduled draft is verified, so the trace holds
the uncensored accepted length of each request, while the policy's own budget
is still recorded.
"""

import json
import os

import numpy as np
import torch

from vllm.utils.gpu_sync_debug import gpu_sync_allowed

TRACE_PATH = os.environ.get("AV_TRACE_PATH")
VERIFY_ALL = os.environ.get("AV_TRACE_VERIFY_ALL") == "1"
_handle = None


def emit(kind: str, **fields: object) -> None:
    global _handle
    if _handle is None:
        _handle = open(str(TRACE_PATH), "a", buffering=1)  # noqa: SIM115
    _handle.write(json.dumps(dict(kind=kind, **fields), separators=(",", ":")))
    _handle.write("\n")


def emit_costs(cost_tables: tuple[np.ndarray, np.ndarray]) -> None:
    draft, verify = cost_tables
    emit("costs", draft_ms=draft.round(4).tolist(), verify_ms=verify.round(4).tolist())


def emit_step(pending: dict, num_sampled: torch.Tensor) -> None:
    """One verification step: per-request confidences, widths and outcomes."""
    with gpu_sync_allowed():
        _emit_step(pending, num_sampled)


def _emit_step(pending: dict, num_sampled: torch.Tensor) -> None:
    scheduled = pending.pop("scheduled")
    rows = np.flatnonzero(scheduled > 0)
    n = len(scheduled)
    probs = pending.pop("probs").cpu().numpy()[rows]
    admitted = pending.pop("admitted").cpu().numpy()[rows]
    accepted = (num_sampled[:n].cpu().numpy()[rows] - 1).clip(min=0)
    req_ids = pending.pop("req_ids")
    emit(
        "step",
        **pending,
        req_ids=[req_ids[i] for i in rows],
        scheduled=scheduled[rows].tolist(),
        admitted=admitted.tolist(),
        accepted=accepted.tolist(),
        p=probs.round(5).tolist(),
    )
