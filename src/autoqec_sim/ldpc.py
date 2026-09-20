# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""LDPC sparsity of a stabilizer group, batched over tableaux.

Two quantities of a stabilizer group ``S`` with ``m = n - k`` generators:

``w*`` (minimum bottleneck weight)
    The smallest ``w`` such that the elements of ``S`` of weight at most
    ``w`` generate ``S``; equivalently the smallest possible maximum
    generator weight over all generating sets. For ``m <= exact_max_generators``
    (default 14) it is computed exactly: all ``2^m`` group elements are
    enumerated as bit-packed ``uint64`` rows, sorted by weight, and a batched
    GF(2) elimination gives the rank of the elements of weight ``<= w``.
    The pivot rows of that elimination, restricted to weight ``<= w*``, are
    a generating set with maximum weight ``w*`` (the *sparsest basis*
    reported here; it is a minimum-bottleneck basis, not necessarily a
    minimum-total-weight one). For larger ``m`` the fallback is a greedy
    pairwise reduction of the given generators (replace ``g_i`` by
    ``g_i g_j`` whenever that lowers its weight, to convergence); its
    maximum weight is an upper bound on ``w*``.

``deg`` (maximum qubit degree)
    The largest number of generators of the sparsest basis acting on one
    qubit. The degree depends on the basis; the value reported is that of
    the basis above, so it is an upper bound on the minimum achievable
    maximum degree at bottleneck weight ``w*``.

The exact path costs ``O(2^m)`` memory per batch element and ``2n``
vectorized elimination steps; at ``n = 9`` and batch 512 one call takes a
few milliseconds, cheap enough to run every environment step. Bit packing
needs ``2n <= 64``.
"""

from __future__ import annotations

import numpy as np

EXACT_MAX_GENERATORS = 14


# --------------------------------------------------------------------------
# packed rows
# --------------------------------------------------------------------------
def pack_rows(rows: np.ndarray) -> np.ndarray:
    """``(..., 2n)`` bits to ``(...)`` uint64 (bit ``i`` is column ``i``)."""
    rows = np.asarray(rows, dtype=np.uint64)
    width = rows.shape[-1]
    if width > 64:
        raise ValueError("bit packing needs 2n <= 64")
    shifts = np.arange(width, dtype=np.uint64)
    return np.bitwise_or.reduce(rows << shifts, axis=-1)


def unpack_rows(v: np.ndarray, width: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.uint64)
    shifts = np.arange(width, dtype=np.uint64)
    return ((v[..., None] >> shifts) & np.uint64(1)).astype(np.uint8)


def packed_weight(v: np.ndarray, n: int) -> np.ndarray:
    """Pauli weight of packed rows (X bits ``0..n-1``, Z bits ``n..2n-1``)."""
    v = np.asarray(v, dtype=np.uint64)
    mask = np.uint64((1 << n) - 1)
    return np.bitwise_count((v & mask) | ((v >> np.uint64(n)) & mask)).astype(np.int64)


def group_elements_packed(gens: np.ndarray) -> np.ndarray:
    """``(B, 2^m)`` packed products of the ``(B, m)`` packed generators
    (index 0 is the identity)."""
    b, m = gens.shape
    out = np.zeros((b, 1 << m), dtype=np.uint64)
    for i in range(m):
        half = 1 << i
        out[:, half : 2 * half] = out[:, :half] ^ gens[:, i : i + 1]
    return out


def batched_rank_packed(rows: np.ndarray, mask: np.ndarray, ncols: int):
    """GF(2) rank of the masked rows of every batch element.

    ``rows`` is ``(B, R)`` uint64, ``mask`` ``(B, R)`` bool. Returns
    ``(rank (B,), pivots (B, ncols))`` where ``pivots[b, c]`` is the index
    of the row chosen as pivot for column ``c`` (``-1`` if none). The
    original rows at the pivot indices are a basis of the masked span.
    Ties are broken toward the lowest row index, so rows sorted by weight
    give a light basis.
    """
    rows = np.where(mask, rows, np.uint64(0))
    b, r = rows.shape
    used = np.zeros((b, r), dtype=bool)
    rank = np.zeros(b, dtype=np.int64)
    pivots = np.full((b, ncols), -1, dtype=np.int64)
    ar = np.arange(b)
    one = np.uint64(1)
    for c in range(ncols):
        bit = ((rows >> np.uint64(c)) & one).astype(bool) & ~used
        piv = bit.argmax(axis=1)
        has = bit[ar, piv]
        if not has.any():
            continue
        pv = np.where(has, rows[ar, piv], np.uint64(0))
        target = bit
        target[ar, piv] = False
        rows ^= np.where(target, pv[:, None], np.uint64(0))
        used[ar[has], piv[has]] = True
        pivots[has, c] = piv[has]
        rank += has
    return rank, pivots


def _generators_packed(tab: np.ndarray, k: int) -> np.ndarray:
    n = (tab.shape[-1] - 1) // 2
    return pack_rows(tab[:, n + k :, : 2 * n])


def _degrees(basis: np.ndarray, n: int) -> np.ndarray:
    """``(B, n)`` number of basis rows ``(B, m)`` acting on each qubit."""
    mask = np.uint64((1 << n) - 1)
    support = (basis & mask) | ((basis >> np.uint64(n)) & mask)  # (B, m)
    shifts = np.arange(n, dtype=np.uint64)
    bits = ((support[..., None] >> shifts) & np.uint64(1)).astype(np.int64)  # (B, m, n)
    return bits.sum(axis=1)


# --------------------------------------------------------------------------
# sparsest basis
# --------------------------------------------------------------------------
def _exact_stats(gens: np.ndarray, n: int):
    """Exact ``w*`` and a bottleneck basis: the rank of the elements of
    weight ``<= w`` is taken for ``w = 1, 2, ...`` until it reaches ``m``;
    only elements still undecided are re-examined at the next ``w``."""
    b, m = gens.shape
    elems = group_elements_packed(gens)[:, 1:]  # drop identity
    wts = packed_weight(elems, n)
    order = np.argsort(wts, axis=1, kind="stable")
    ar = np.arange(b)[:, None]
    elems = elems[ar, order]
    wts = wts[ar, order]
    w_star = np.full(b, n, dtype=np.int64)
    basis = np.zeros((b, m), dtype=np.uint64)
    undecided = np.ones(b, dtype=bool)
    for w in range(int(wts[:, 0].min()), n + 1):
        if not undecided.any():
            break
        idx = np.flatnonzero(undecided)
        rank, piv = batched_rank_packed(elems[idx], wts[idx] <= w, 2 * n)
        ok = rank == m
        sel = idx[ok]
        w_star[sel] = w
        basis[sel] = _basis_from_pivots(elems[sel], piv[ok], m)
        undecided[sel] = False
    return w_star, basis


def _basis_from_pivots(elems: np.ndarray, pivots: np.ndarray, m: int) -> np.ndarray:
    """Gather the ``m`` pivot rows (``pivots`` has ``-1`` in non-pivot
    columns) into ``(B, m)``."""
    b = elems.shape[0]
    out = np.zeros((b, m), dtype=np.uint64)
    for i in range(b):
        p = pivots[i][pivots[i] >= 0]
        out[i, : p.size] = elems[i, p]
    return out


def _greedy_stats(gens: np.ndarray, n: int, max_passes: int = 20):
    gens = gens.copy()
    _, m = gens.shape
    wts = packed_weight(gens, n)
    for _ in range(max_passes):
        changed = False
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                cand = gens[:, i] ^ gens[:, j]
                cw = packed_weight(cand, n)
                better = (cw < wts[:, i]) & (cw > 0)
                if better.any():
                    gens[better, i] = cand[better]
                    wts[better, i] = cw[better]
                    changed = True
        if not changed:
            break
    return wts.max(axis=1), gens


def sparsest_basis_stats(
    tab: np.ndarray, k: int, w_max: int | None = None, exact_max_generators: int = EXACT_MAX_GENERATORS
) -> dict:
    """``{"w_star": (B,), "max_degree": (B,), "basis": (B, m) packed,
    "exact": bool}`` for every tableau in the batch. See the module
    docstring for what is exact and what is a bound."""
    n = (tab.shape[-1] - 1) // 2
    gens = _generators_packed(tab, k)
    m = gens.shape[1]
    if m == 0:
        b = tab.shape[0]
        return {"w_star": np.zeros(b, dtype=np.int64), "max_degree": np.zeros(b, dtype=np.int64), "basis": gens, "exact": True}
    exact = m <= exact_max_generators
    if exact:
        w_star, basis = _exact_stats(gens, n)
    else:
        w_star, basis = _greedy_stats(gens, n)
    deg = _degrees(basis, n).max(axis=1)
    return {"w_star": w_star, "max_degree": deg, "basis": basis, "exact": exact}


def sparsest_basis_rows(tab: np.ndarray, k: int, index: int = 0, **kw) -> np.ndarray:
    """The sparsest basis of batch element ``index`` as ``(m, 2n)`` bits."""
    n = (tab.shape[-1] - 1) // 2
    st = sparsest_basis_stats(tab[index : index + 1], k, **kw)
    return unpack_rows(st["basis"][0], 2 * n)


# --------------------------------------------------------------------------
# constraint and penalty
# --------------------------------------------------------------------------
def ldpc_feasible(tab: np.ndarray, k: int, w_max: int, exact_max_generators: int = EXACT_MAX_GENERATORS) -> np.ndarray:
    """``(B,)`` bool: the stabilizer group has a generating set with every
    generator of weight ``<= w_max``. Exact for ``n - k <= exact_max_generators``
    (rank of the elements of weight ``<= w_max`` equals ``n - k``); the
    greedy fallback for larger codes is sufficient but not necessary."""
    n = (tab.shape[-1] - 1) // 2
    gens = _generators_packed(tab, k)
    m = gens.shape[1]
    if m == 0:
        return np.ones(tab.shape[0], dtype=bool)
    if m <= exact_max_generators:
        elems = group_elements_packed(gens)[:, 1:]
        wts = packed_weight(elems, n)
        rank, _ = batched_rank_packed(elems, wts <= w_max, 2 * n)
        return rank == m
    w_star, _ = _greedy_stats(gens, n)
    return w_star <= w_max


def ldpc_penalty(
    tab: np.ndarray,
    k: int,
    w_max: int,
    deg_max: int | None = None,
    scale: float = 1.0,
    exact_max_generators: int = EXACT_MAX_GENERATORS,
) -> np.ndarray:
    """``-scale * (max(0, w* - w_max) + max(0, deg - deg_max))`` per
    element, with ``w*`` and ``deg`` from :func:`sparsest_basis_stats`.
    Zero iff the sparsest basis meets both bounds."""
    st = sparsest_basis_stats(tab, k, w_max, exact_max_generators)
    excess = np.maximum(0, st["w_star"] - w_max).astype(float)
    if deg_max is not None:
        excess += np.maximum(0, st["max_degree"] - deg_max)
    return -scale * excess


def make_ldpc_penalty(w_max: int, deg_max: int | None = None, scale: float = 1.0):
    def f(tab, k):
        return ldpc_penalty(tab, k, w_max, deg_max, scale)

    return f
