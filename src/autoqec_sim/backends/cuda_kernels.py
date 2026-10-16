# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Fused CUDA kernels behind the torch backend, compiled through CuPy (NVRTC).

Two kernels replace chains of torch ops on CUDA tensors:

- ``apply_actions``: one launch applies a per-element gate (any of the nine
  gates, per-element qubit indices) to every row of every selected tableau,
  in place. The torch path groups the batch by gate type and issues several
  gather/scatter ops per group.
- ``parity_codes``: bit-packed popcount symplectic products of an error set
  against the code rows, reduced in the kernel to one byte per (element,
  error): 0 detected, 1 undetected, 2 in the stabilizer group. The torch path
  is a float32 GEMM, a remainder, and a second GEMM.

Opt in with ``AUTOQEC_CUDA_KERNELS=1`` or :func:`enable`. CPU tensors, a
missing CuPy, or a disabled flag fall back to the torch path. Agreement with
that path is tested in ``tests/test_cuda_kernels.py``.
"""
from __future__ import annotations

import os

import torch

try:  # CuPy is optional; the module imports without it.
    import cupy as cp
except Exception:  # pragma: no cover - depends on the machine
    cp = None

GATE_CODES = {"H": 0, "S": 1, "S_DAG": 2, "X": 3, "Y": 4, "Z": 5, "CNOT": 6, "CZ": 7, "SWAP": 8}

_enabled = os.environ.get("AUTOQEC_CUDA_KERNELS", "") not in ("", "0")


def enable(flag: bool = True) -> None:
    """Turn the kernels on or off for CUDA tensors (default: the environment variable)."""
    global _enabled
    _enabled = bool(flag)


def active(t: torch.Tensor) -> bool:
    """True when ``t`` is a CUDA tensor and the kernels are enabled and available."""
    return _enabled and cp is not None and t.is_cuda


_GATE_SRC = r"""
extern "C" __global__
void apply_actions(unsigned char* tab, const long long* sel, const int* gate,
                   const long long* q1, const long long* q2, int m, int n)
{
    const int rows = 2 * n;
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)m * rows) return;
    const int i = (int)(idx / rows);
    const int r = (int)(idx % rows);
    const int g = gate[i];
    if (g < 0) return;
    unsigned char* row = tab + (sel[i] * rows + r) * (long long)(rows + 1);
    const int a = (int)q1[i];
    const int c = (int)q2[i];
    const unsigned char xa = row[a], za = row[n + a];
    const unsigned char ph = row[rows];
    switch (g) {
    case 0: row[rows] = ph ^ (xa & za); row[a] = za; row[n + a] = xa; break;          // H
    case 1: row[rows] = ph ^ (xa & za); row[n + a] = za ^ xa; break;                   // S
    case 2: row[rows] = ph ^ (xa & (za ^ 1)); row[n + a] = za ^ xa; break;             // S_DAG
    case 3: row[rows] = ph ^ za; break;                                                 // X
    case 4: row[rows] = ph ^ xa ^ za; break;                                            // Y
    case 5: row[rows] = ph ^ xa; break;                                                 // Z
    case 6: {                                                                           // CNOT a -> c
        const unsigned char xc = row[c], zc = row[n + c];
        row[rows] = ph ^ (xa & zc & (xc ^ za ^ 1));
        row[c] = xc ^ xa; row[n + a] = za ^ zc; break; }
    case 7: {                                                                           // CZ
        const unsigned char xc = row[c], zc = row[n + c];
        row[rows] = ph ^ (xa & xc & (za ^ zc));
        row[n + a] = za ^ xc; row[n + c] = zc ^ xa; break; }
    case 8: {                                                                           // SWAP
        const unsigned char xc = row[c], zc = row[n + c];
        row[a] = xc; row[n + a] = zc; row[c] = xa; row[n + c] = za; break; }
    }
}
"""

_PARITY_SRC = r"""
extern "C" __global__
void parity_codes(const unsigned long long* err, const unsigned long long* rows,
                  unsigned char* out, int M, int B, int R, int W, int nstab)
{
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)B * M) return;
    const int b = (int)(idx / M);
    const int m = (int)(idx % M);
    const unsigned long long* e = err + (long long)m * W;
    const unsigned long long* rb = rows + (long long)b * R * W;
    int anti_stab = 0, anti_log = 0;
    for (int r = 0; r < R; ++r) {
        const unsigned long long* row = rb + (long long)r * W;
        int acc = 0;
        for (int w = 0; w < W; ++w) acc ^= __popcll(e[w] & row[w]);
        if (r < nstab) anti_stab |= (acc & 1); else anti_log |= (acc & 1);
    }
    out[idx] = anti_stab ? 0 : (anti_log ? 1 : 2);
}
"""

_kernels: dict = {}


def _kernel(name: str):
    if name not in _kernels:
        src = _GATE_SRC if name == "apply_actions" else _PARITY_SRC
        _kernels[name] = cp.RawKernel(src, name)
    return _kernels[name]


def _cp(t: torch.Tensor):
    return cp.from_dlpack(t.contiguous())


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """``(..., L)`` 0/1 uint8 to ``(..., W)`` int64 words, 64 bits per word, little-endian."""
    L = bits.shape[-1]
    W = (L + 63) // 64
    if W * 64 != L:
        pad = torch.zeros((*bits.shape[:-1], W * 64 - L), dtype=torch.uint8, device=bits.device)
        bits = torch.cat([bits.to(torch.uint8), pad], dim=-1)
    weights = (1 << torch.arange(8, device=bits.device, dtype=torch.int16)).to(torch.uint8)
    by = (bits.view((*bits.shape[:-1], W, 8, 8)) * weights).sum(-1, dtype=torch.int16).to(torch.uint8)
    return by.contiguous().view(torch.int64).reshape((*bits.shape[:-1], W))


def apply_actions(tab: torch.Tensor, sel: torch.Tensor, codes: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> None:
    """Apply gate ``codes[i]`` (``GATE_CODES``, -1 for none) on qubits ``q1[i]``, ``q2[i]``
    to tableau ``sel[i]``, every row, one launch."""
    n = (tab.shape[-1] - 1) // 2
    m = int(sel.numel())
    if m == 0:
        return
    total = m * 2 * n
    threads = 256
    _kernel("apply_actions")(
        ((total + threads - 1) // threads,), (threads,),
        (_cp(tab), _cp(sel.to(torch.int64)), _cp(codes.to(torch.int32)),
         _cp(q1.to(torch.int64)), _cp(q2.to(torch.int64)), cp.int32(m), cp.int32(n)),
    )


_errors_cache: dict = {}


def packed_errors(errors: torch.Tensor) -> torch.Tensor:
    """``(M, W)`` int64 words of a ``(M, 2n)`` error set, cached per tensor storage."""
    key = (errors.data_ptr(), tuple(errors.shape), str(errors.device))
    hit = _errors_cache.get(key)
    if hit is None:
        if len(_errors_cache) > 8:
            _errors_cache.clear()
        hit = _errors_cache[key] = pack_bits(errors.to(torch.uint8))
    return hit


def parity_codes(rows_swapped: torch.Tensor, errors: torch.Tensor, nstab: int, chunk: int = 1 << 20) -> torch.Tensor:
    """``(B, M)`` uint8: 0 detected, 1 undetected, 2 in the stabilizer group.

    ``rows_swapped`` is ``(B, R, 2n)`` uint8, the code rows with X and Z halves
    swapped (stabilizers first, ``nstab`` of them, then logicals); ``errors``
    is ``(M, 2n)`` uint8 on the same device.
    """
    B, R, _ = rows_swapped.shape
    rp = pack_bits(rows_swapped.to(torch.uint8))  # (B, R, W)
    ep = packed_errors(errors)  # (M, W)
    W = rp.shape[-1]
    M = ep.shape[0]
    out = torch.empty((B, M), dtype=torch.uint8, device=rows_swapped.device)
    threads = 256
    for start in range(0, M, chunk):
        e = ep[start : start + chunk].contiguous()
        mc = e.shape[0]
        o = torch.empty((B, mc), dtype=torch.uint8, device=rows_swapped.device)
        _kernel("parity_codes")(
            ((B * mc + threads - 1) // threads,), (threads,),
            (_cp(e), _cp(rp), _cp(o), cp.int32(mc), cp.int32(B), cp.int32(R), cp.int32(W), cp.int32(nstab)),
        )
        out[:, start : start + mc] = o
    return out
