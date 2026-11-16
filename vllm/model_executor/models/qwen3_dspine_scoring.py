# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental DSpine transition-scoring variants (selected by env vars).

VLLM_DSPINE_SCORING: "ref" (default), "compile", "lowrank", "lowrank_compile" or
"fused".
VLLM_DSPINE_SCORING_TF32=1 runs the scoring's fp32 matmuls on TF32 tensor cores.
"""

import contextlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton

SCORING = os.environ.get("VLLM_DSPINE_SCORING", "ref")
TF32 = os.environ.get("VLLM_DSPINE_SCORING_TF32", "0") == "1"


@dataclass
class LowRankTables:
    pred_gate: torch.Tensor  # [V, M] predecessor_gate(codebook[v] * sqrt(T))
    message: torch.Tensor  # [V, M] message(codebook[v] * sqrt(T))
    successor: torch.Tensor  # [V, M] (lm_head[v] * g) @ W, W = writes[-1].weight
    gram: torch.Tensor  # [M, M] W^T W
    write: torch.Tensor  # [H, M] W in fp32
    write_bf16: torch.Tensor  # [H, M] W as stored


def score(model, candidate_ids, final_pre_injection, anchor_token_ids, ref_fn):
    with _tf32() if TF32 else contextlib.nullcontext():
        if SCORING in ("lowrank", "lowrank_compile", "fused"):
            fn = {
                "lowrank": _lowrank,
                "lowrank_compile": _lowrank_compiled,
                "fused": _fused,
            }[SCORING]
            return fn(
                model.model.injection,
                model.model.norm,
                model.lm_head,
                model._dspine_tables,
                candidate_ids,
                final_pre_injection,
                anchor_token_ids,
            )
        fn = _compiled(ref_fn) if SCORING == "compile" else ref_fn
        return fn(
            model.model.injection,
            model.model.norm,
            model.lm_head,
            model.model.transfer_codebook,
            candidate_ids,
            final_pre_injection,
            anchor_token_ids,
            len(model.model.layers) - 1,
        )


@torch.no_grad()
def build_tables(model) -> LowRankTables | None:
    if SCORING not in ("lowrank", "lowrank_compile", "fused"):
        return None
    assert model.lm_head.tp_size == 1, "lowrank scoring supports TP=1 only"
    inj, norm = model.model.injection, model.model.norm
    write = inj.writes[len(model.model.layers) - 1].weight.float()
    codebook, lm_head = model.model.transfer_codebook, model.lm_head.weight
    vocab, msg_dim = codebook.shape[0], write.shape[1]
    tables = [torch.empty(vocab, msg_dim, device=write.device) for _ in range(3)]
    for lo in range(0, vocab, 16384):
        hi = min(lo + 16384, vocab)
        rows = slice(lo, hi)
        x = codebook[rows].float() * math.sqrt(inj.transfer_dim)
        tables[0][rows] = F.linear(x, inj.predecessor_gate.weight.float())
        tables[1][rows] = F.linear(x, inj.message.weight.float())
        tables[2][rows] = (lm_head[rows].float() * norm.weight.float()) @ write
    write_bf16 = inj.writes[len(model.model.layers) - 1].weight.contiguous()
    pred_gate, message, successor = tables
    gram = write.T @ write
    return LowRankTables(pred_gate, message, successor, gram, write, write_bf16)


def _lowrank(
    injection,
    norm,
    lm_head,
    t: LowRankTables,
    candidate_ids,
    final_pre_injection,
    anchor_token_ids,
):
    """score[k, j] = (r.(g*e_j) + s * u_k.t_j) / sqrt(mean(|r + s W u_k|^2) + eps).

    The same math as `_score_transitions`, with the per-predecessor update
    `W u_k` never expanded to the hidden size.
    """
    top_k = candidate_ids.shape[-1]
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]),
        dim=1,
    )
    receiver_features = injection.project(
        len(injection.writes) - 1, final_pre_injection
    )
    r = final_pre_injection.float()
    s = injection.rms(final_pre_injection)
    gate_in = (
        injection.receiver_gate(receiver_features).float() + injection.gate_bias.float()
    )
    gate = torch.sigmoid(gate_in[:, :, None, :] + t.pred_gate[predecessor_ids])
    u = gate * t.message[predecessor_ids]
    unary = torch.einsum(
        "bnh,bnjh->bnj", r * norm.weight.float(), lm_head.weight[candidate_ids].float()
    )
    cross = torch.einsum("bnkm,bnjm->bnkj", u, t.successor[candidate_ids])
    q = r @ t.write
    quad = ((u @ t.gram) * u).sum(-1)
    sq = (
        r.square().sum(-1, keepdim=True)
        + 2 * s * (u * q[:, :, None]).sum(-1)
        + s.square() * quad
    )
    inv = (sq / r.shape[-1] + norm.variance_epsilon).rsqrt()
    return (unary[:, :, None, :] + s[..., None] * cross) * inv[..., None]


@contextlib.contextmanager
def _tf32():
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


_COMPILED: dict[Callable, Callable] = {}


def _compiled(fn: Callable) -> Callable:
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn, dynamic=True)
    return _COMPILED[fn]


_lowrank_compiled = torch.compile(_lowrank, dynamic=True)


def _fused(
    injection,
    norm,
    lm_head,
    t: LowRankTables,
    candidate_ids,
    final_pre_injection,
    anchor_token_ids,
):
    """`_lowrank` with the per-position gathers and dot products in one Triton
    kernel, and u^T G u as a TF32 GEMM.

    u^T G u only scales row k of the scores, so TF32 cannot change the argmax
    over successors, and its ~1e-3 relative error barely changes the softmax.
    """
    batch, steps, top_k = candidate_ids.shape
    positions, hidden = batch * steps, final_pre_injection.shape[-1]
    msg_dim = t.gram.shape[0]
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]),
        dim=1,
    )
    receiver_features = injection.project(
        len(injection.writes) - 1, final_pre_injection
    )
    gate_in = (
        injection.receiver_gate(receiver_features).float() + injection.gate_bias.float()
    )
    r = final_pre_injection.reshape(positions, hidden)
    q = torch.mm(r, t.write_bf16, out_dtype=torch.float32)  # exact products
    u = torch.empty(positions * top_k, msg_dim, device=r.device)
    numerator = torch.empty(positions, top_k, top_k, device=r.device)
    lin = torch.empty(positions, top_k, device=r.device)
    s = torch.empty(positions, device=r.device)
    _position_kernel[(positions,)](
        r,
        norm.weight,
        gate_in.reshape(positions, msg_dim),
        q,
        predecessor_ids.reshape(positions, top_k).contiguous(),
        candidate_ids.reshape(positions, top_k).contiguous(),
        t.pred_gate,
        t.message,
        t.successor,
        lm_head.weight,
        u,
        numerator,
        lin,
        s,
        hidden,
        lm_head.weight.stride(0),
        K=top_k,
        M=msg_dim,
        BLOCK_H=512,
        BLOCK_M=64,
        num_warps=4,
    )
    with _tf32():
        quad = (torch.mm(u, t.gram) * u).sum(-1).view(positions, top_k)
    inv = ((lin + s[:, None].square() * quad) / hidden + norm.variance_epsilon).rsqrt()
    return (numerator * inv[..., None]).view(batch, steps, top_k, top_k)


@triton.jit
def _position_kernel(
    r_ptr,
    g_ptr,
    gate_ptr,
    q_ptr,
    pred_ptr,
    cand_ptr,
    pg_ptr,
    msg_ptr,
    succ_ptr,
    lm_ptr,
    u_ptr,
    numerator_ptr,
    lin_ptr,
    s_ptr,
    H,
    stride_lm,
    K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, K)
    pred = tl.load(pred_ptr + pid * K + ks)
    cand = tl.load(cand_ptr + pid * K + ks)
    # r.(g*e_j) and |r|^2 over the hidden size.
    hs = tl.arange(0, BLOCK_H)
    unary = tl.zeros((K,), tl.float32)
    rr = tl.zeros((BLOCK_H,), tl.float32)
    for h in range(0, H, BLOCK_H):
        r = tl.load(r_ptr + pid * H + h + hs).to(tl.float32)
        rr += r * r
        rg = r * tl.load(g_ptr + h + hs).to(tl.float32)
        e = tl.load(lm_ptr + cand[:, None] * stride_lm + h + hs[None, :]).to(tl.float32)
        unary += tl.sum(e * rg[None, :], 1)
    r_sq = tl.sum(rr, 0)
    s = tl.sqrt(tl.maximum(r_sq / H, 1.1754943508222875e-38))
    # u_k (stored for u^T G u), u_k.t_j and q.u_k, in BLOCK_M chunks.
    cross = tl.zeros((K, K), tl.float32)
    uq = tl.zeros((K,), tl.float32)
    for c in range(0, M, BLOCK_M):
        cs = c + tl.arange(0, BLOCK_M)
        gate = tl.load(gate_ptr + pid * M + cs)
        pg = tl.load(pg_ptr + pred[:, None] * M + cs[None, :])
        msg = tl.load(msg_ptr + pred[:, None] * M + cs[None, :])
        u_c = tl.sigmoid(gate[None, :] + pg) * msg
        tl.store(u_ptr + (pid * K + ks[:, None]) * M + cs[None, :], u_c)
        succ = tl.load(succ_ptr + cand[:, None] * M + cs[None, :])
        cross += tl.dot(u_c, tl.trans(succ), input_precision="tf32x3")
        uq += tl.sum(u_c * tl.load(q_ptr + pid * M + cs)[None, :], 1)
    tl.store(
        numerator_ptr + pid * K * K + ks[:, None] * K + ks[None, :],
        unary[None, :] + s * cross,
    )
    tl.store(lin_ptr + pid * K + ks, r_sq + 2 * s * uq)
    tl.store(s_ptr + pid, s)
