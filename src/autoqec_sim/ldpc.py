# Copyright 2026 Unitary Foundation
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

:class:`LdpcObjective` combines the Knill-Laflamme term, the padding
penalty, and the sparsity constraint into the reward and the success
criterion of an LDPC-constrained ``[[n, k, d]]`` target, in one of the
forms of ``LDPC_FORMS`` (compared at ``n = 12`` and ``16`` in
``experiments/ldpc_scaling/REPORT.md``; ``DEFAULT_LDPC_FORM`` is the one
that comparison selected; the ``shaped`` and ``terminal`` forms and
:func:`greedy_action_mask` are the sparsity variants of
``experiments/ldpc_sparsity/REPORT.md``; its ``"exact"`` method, the
feasibility test after every two-qubit action on the pairs the greedy
bound rejects, is the exact-mask section of that report).
"""

from __future__ import annotations

import time

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


def heavy_generator_count(
    tab: np.ndarray, k: int, w_max: int, exact_max_generators: int = EXACT_MAX_GENERATORS
) -> np.ndarray:
    """``(B,)`` number of generators of the sparsest basis with weight above
    ``w_max``: ``m`` minus the rank of the group elements of weight at most
    ``w_max``. Zero iff the group is feasible under ``w_max``. Basis
    independent, since every bottleneck basis built lightest first contains
    exactly a basis of the light span. Exact for ``m <= exact_max_generators``;
    for larger ``m`` the count of heavy rows of the greedy basis (an upper
    bound)."""
    n = (tab.shape[-1] - 1) // 2
    gens = _generators_packed(tab, k)
    m = gens.shape[1]
    if m == 0:
        return np.zeros(tab.shape[0], dtype=np.int64)
    if m <= exact_max_generators:
        elems = group_elements_packed(gens)[:, 1:]
        wts = packed_weight(elems, n)
        rank, _ = batched_rank_packed(elems, wts <= w_max, 2 * n)
        return m - rank
    _, basis = _greedy_stats(gens, n)
    return (packed_weight(basis, n) > w_max).sum(axis=1)


def _packed_cnot(gens: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
    """CNOT applied to packed Pauli rows (X bit ``q``, Z bit ``n + q``):
    ``x_t ^= x_c`` and ``z_c ^= z_t``."""
    one = np.uint64(1)
    g = gens.copy()
    xc = (g >> np.uint64(control)) & one
    g ^= xc << np.uint64(target)
    zt = (g >> np.uint64(n + target)) & one
    g ^= zt << np.uint64(n + control)
    return g


def _greedy_within(gens: np.ndarray, n: int, w_max: int, max_passes: int = 20) -> np.ndarray:
    """``(B,)`` bool: the greedy pairwise reduction of ``gens`` (``(B, m)``
    packed) brings every row to weight ``<= w_max``. Only rows above the
    bound are reduced, and only on the batch elements that still have one,
    so the cost scales with the number of heavy rows rather than with
    ``m^2 B``. Same bound as :func:`_greedy_stats` thresholded at ``w_max``
    (an upper bound on ``w*``: true is sufficient for feasibility)."""
    gens = gens.copy()
    _, m = gens.shape
    wts = packed_weight(gens, n)
    for _ in range(max_passes):
        heavy_any = (wts > w_max).any(axis=1)
        if not heavy_any.any():
            break
        changed = False
        for i in range(m):
            idx = np.flatnonzero(wts[:, i] > w_max)
            if idx.size == 0:
                continue
            gi = gens[idx, i]
            wi = wts[idx, i]
            for j in range(m):
                if i == j:
                    continue
                cand = gi ^ gens[idx, j]
                cw = packed_weight(cand, n)
                better = (cw < wi) & (cw > 0)
                if better.any():
                    gi = np.where(better, cand, gi)
                    wi = np.where(better, cw, wi)
                    changed = True
            gens[idx, i] = gi
            wts[idx, i] = wi
        if not changed:
            break
    return ~(wts > w_max).any(axis=1)


def _packed_cnot_pairs(rows: np.ndarray, n: int, control: np.ndarray, target: np.ndarray) -> np.ndarray:
    """:func:`_packed_cnot` with one ``(control, target)`` pair per row
    block: ``rows`` is ``(P, R)`` packed Paulis, ``control`` and ``target``
    are ``(P,)``."""
    one = np.uint64(1)
    c = np.asarray(control, dtype=np.uint64)[:, None]
    t = np.asarray(target, dtype=np.uint64)[:, None]
    g = rows.copy()
    xc = (g >> c) & one
    g ^= xc << t
    zt = (g >> (np.uint64(n) + t)) & one
    g ^= zt << (np.uint64(n) + c)
    return g


class ExactMaskCache:
    """Bounded cache of the light group elements of a stabilizer group for
    :func:`greedy_action_mask` with ``method="exact"``, keyed by the bytes
    of the packed greedy basis (the same tableau gives the same basis, so a
    state seen again, such as the initial state after a reset, is not
    enumerated twice). Oldest entries are evicted above ``maxsize``."""

    def __init__(self, maxsize: int = 4096):
        self.maxsize = int(maxsize)
        self.data: dict = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: bytes):
        v = self.data.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, key: bytes, value) -> None:
        if len(self.data) >= self.maxsize:
            for old in list(self.data)[: max(1, self.maxsize // 8)]:
                del self.data[old]
        self.data[key] = value


# weight change on the pair (control, target) of a CNOT, by the local Pauli
# codes (0 = I, 1 = X, 2 = Z, 3 = Y) of the control and the target: the CNOT
# maps X_c -> X_c X_t, Z_c -> Z_c, X_t -> X_t, Z_t -> Z_c Z_t
def _cnot_weight_delta_table() -> np.ndarray:
    tab = np.zeros(16, dtype=np.int8)
    for pc in range(4):
        for pt in range(4):
            xc, zc, xt, zt = pc & 1, pc >> 1, pt & 1, pt >> 1
            xc2, zc2, xt2, zt2 = xc, zc ^ zt, xt ^ xc, zt
            before = (pc != 0) + (pt != 0)
            after = ((xc2 | zc2) != 0) + ((xt2 | zt2) != 0)
            tab[pc * 4 + pt] = after - before
    return tab


_CNOT_DELTA = _cnot_weight_delta_table()


def _light_elements(basis: np.ndarray, n: int, w_lim: int, cache: ExactMaskCache | None = None):
    """For every row of the ``(B, m)`` packed basis: the group elements of
    weight at most ``w_lim`` (identity excluded) as ``(coefficient vectors
    (R,) uint16, weights (R,) uint8, local Pauli codes (R, n) uint8)``. The
    coefficient vector of an element is its index in
    :func:`group_elements_packed` (bit ``i`` set iff generator ``i`` is a
    factor), so a GF(2) rank over these ``m``-bit integers is the rank of
    the elements in the group. The local code of qubit ``q`` is
    ``x_q + 2 z_q``. Returns three lists over the batch. States are
    enumerated in chunks of 64 to bound memory at ``2^m`` elements per
    state."""
    b, _m = basis.shape
    idx_out: list = [None] * b
    wt_out: list = [None] * b
    loc_out: list = [None] * b
    todo = []
    for i in range(b):
        hit = cache.get(basis[i].tobytes()) if cache is not None else None
        if hit is not None:
            idx_out[i], wt_out[i], loc_out[i] = hit
        else:
            todo.append(i)
    shifts = np.arange(n, dtype=np.uint64)
    one = np.uint64(1)
    for start in range(0, len(todo), 64):
        sel = todo[start : start + 64]
        elems = group_elements_packed(basis[sel])[:, 1:]
        wts = packed_weight(elems, n)
        for j, i in enumerate(sel):
            keep = np.flatnonzero(wts[j] <= w_lim)
            el = elems[j, keep]
            xs = ((el[:, None] >> shifts) & one).astype(np.uint8)
            zs = ((el[:, None] >> (shifts + np.uint64(n))) & one).astype(np.uint8)
            val = ((keep + 1).astype(np.uint16), wts[j, keep].astype(np.uint8), xs | (zs << 1))
            idx_out[i], wt_out[i], loc_out[i] = val
            if cache is not None:
                cache.put(basis[i].tobytes(), val)
    return idx_out, wt_out, loc_out


def _rank_small(rows: np.ndarray, mask: np.ndarray, ncols: int) -> np.ndarray:
    """GF(2) rank ``(P,)`` of the masked rows of every batch element of the
    ``(P, R)`` small-integer rows (uint16), as :func:`batched_rank_packed`
    without the pivot bookkeeping; a column with no set bit in the whole
    chunk costs one pass."""
    rows = np.where(mask, rows, 0)
    p, r = rows.shape
    used = np.zeros((p, r), dtype=bool)
    rank = np.zeros(p, dtype=np.int64)
    ar = np.arange(p)
    for c in range(ncols):
        bit = ((rows >> c) & 1).astype(bool)
        bit &= ~used
        piv = bit.argmax(axis=1)
        has = bit[ar, piv]
        if not has.any():
            continue
        pv = np.where(has, rows[ar, piv], 0)
        bit[ar, piv] = False
        rows ^= np.where(bit, pv[:, None], 0)
        used[ar[has], piv[has]] = True
        rank += has
    return rank


def _exact_cnot_admits(
    light: tuple, pair_b: np.ndarray, pair_c: np.ndarray, pair_t: np.ndarray, heavy_bits: np.ndarray, n: int, m: int, w_max: int, chunk_rows: int
) -> np.ndarray:
    """``(P,)`` bool: for every pair (state ``pair_b[p]``, CNOT
    ``pair_c[p] -> pair_t[p]``), the group after the CNOT has a generating
    set with every weight ``<= w_max``. Exact. ``light`` is the output of
    :func:`_light_elements` at ``w_max + 1`` (a CNOT changes the weight of
    a Pauli by at most one, so every element light after the gate is among
    them); ``heavy_bits[p]`` marks the rows of the gate-transformed basis
    with weight above ``w_max``. The light span contains every light row
    of that basis, so it is the whole group iff the coefficient vectors of
    the light elements, restricted to the heavy rows, have rank equal to
    the number of heavy rows: the elimination runs over the ``m``
    coefficient bits with the light rows zeroed, so only the heavy columns
    cost a full pass. The weight after the gate is the weight before plus
    a table lookup on the local Paulis of the control and the target.
    Pairs are sorted by the number of light elements of their state and
    processed in chunks of at most ``chunk_rows`` padded rows."""
    light_idx, light_wt, light_loc = light
    p_total = pair_b.size
    out = np.zeros(p_total, dtype=bool)
    if p_total == 0:
        return out
    sizes_b = np.array([x.size for x in light_idx], dtype=np.int64)
    r_all = int(sizes_b.max()) if sizes_b.size else 0
    need = np.bitwise_count(heavy_bits.astype(np.uint64)).astype(np.int64)
    if r_all == 0:
        out[:] = need == 0
        return out
    b = len(light_idx)
    idx_pad = np.zeros((b, r_all), dtype=np.uint16)
    wt_pad = np.full((b, r_all), 255, dtype=np.uint8)
    loc_pad = np.zeros((b, r_all, n), dtype=np.uint8)
    for i in range(b):
        r = sizes_b[i]
        idx_pad[i, :r] = light_idx[i]
        wt_pad[i, :r] = light_wt[i]
        loc_pad[i, :r] = light_loc[i]
    sizes = sizes_b[pair_b]
    order = np.argsort(sizes, kind="stable")
    pos = 0
    while pos < p_total:
        end = pos + 1
        while end < p_total and (end + 1 - pos) * max(int(sizes[order[end]]), 1) <= chunk_rows:
            end += 1
        sel = order[pos:end]
        r = int(sizes[sel[-1]])
        if r == 0:
            out[sel] = need[sel] == 0
            pos = end
            continue
        sb = pair_b[sel]
        ar = np.arange(r)
        pc = loc_pad[sb[:, None], ar[None, :], pair_c[sel][:, None]]
        pt = loc_pad[sb[:, None], ar[None, :], pair_t[sel][:, None]]
        w_after = wt_pad[sb, :r].astype(np.int16) + _CNOT_DELTA[(pc << 2) | pt]
        light_mask = w_after <= w_max  # padding has weight 255
        rows = idx_pad[sb, :r] & heavy_bits[sel].astype(np.uint16)[:, None]
        out[sel] = _rank_small(rows, light_mask, m) == need[sel]
        pos = end
    return out


def greedy_action_mask(
    tab: np.ndarray,
    k: int,
    action_table,
    w_max: int,
    method: str = "basis",
    exact_max_generators: int = EXACT_MAX_GENERATORS,
    cache: ExactMaskCache | None = None,
    stats: dict | None = None,
    chunk_rows: int = 1 << 22,
) -> np.ndarray:
    """``(B, num_actions)`` bool: false where the action would take the
    stabilizer group outside the bound ``w* <= w_max``. Single-qubit gates
    never change the support of a stabilizer, so they are always allowed.
    Three methods; the first two are upper bounds on ``w*`` (the mask can
    forbid an action whose exact ``w*`` is within the bound; it never
    allows one outside), the third is exact:

    ``"tableau"``
        every two-qubit action is applied to a copy of the whole tableau
        batch and :func:`_greedy_stats` runs from the tableau's generators
        on all copies in one call (``0.13 s`` for 132 CNOT actions at
        ``n = 12``, batch 256; ``7 s`` for 600 at ``n = 25``, most of it
        the copies and the packing). The method of the runs of
        ``experiments/ldpc_sparsity/REPORT.md`` at ``n = 12``.
    ``"basis"``
        the current state's greedy basis is computed once (``(B, m)``
        packed rows), every CNOT is applied to those packed rows with two
        bit operations (:func:`_packed_cnot`), and :func:`_greedy_within`
        re-reduces only the rows the gate made heavy (a CNOT raises the
        weight of a Pauli by at most one). No tableau copies; the cost
        scales with the number of heavy rows. Requires the two-qubit
        actions to be CNOTs (falls back to ``"tableau"`` otherwise).
    ``"exact"``
        the ``"basis"`` bound as a pre-filter, then the exact feasibility
        test on every (state, action) pair the bound rejected: the group
        after the gate has a generating set within ``w_max`` iff the
        elements the gate leaves at weight ``<= w_max`` have rank
        ``m = n - k``, decided by :func:`_exact_cnot_admits` over the
        elements of weight ``<= w_max + 1`` of the current group (the only
        ones a CNOT can bring within the bound), enumerated once per state
        and shared by all its actions (``cache`` keeps them across calls
        for states seen again); the rank test is over the heavy rows of the
        gate-transformed basis only, since the light rows are in the light
        span already. Needs ``m <= exact_max_generators``; above
        that the method is ``"basis"`` (``stats["method"]`` records which
        ran). Two-qubit gates other than CNOT are tested exactly on tableau
        copies of the rejected pairs (:func:`ldpc_feasible`).

    ``stats``, when given, receives the counts of the call: two-qubit
    pairs, pairs the greedy bound rejected, pairs tested exactly (and how
    many of them were distinct, since identical states in the batch share
    one test per action), pairs the exact test admitted, cache hits and
    misses, and the seconds spent in the greedy and the exact parts.
    """
    b = tab.shape[0]
    n = (tab.shape[-1] - 1) // 2
    m = n - k
    mask = np.ones((b, len(action_table)), dtype=bool)
    two = np.flatnonzero(action_table.q2 >= 0)
    if two.size == 0 or b == 0:
        return mask
    entries = [action_table.entries[a] for a in two]
    all_cnot = all(g == "CNOT" for g, _, _ in entries)
    exact = method == "exact" and m <= exact_max_generators
    if stats is not None:
        stats.update(method="exact" if exact else ("basis" if method in ("basis", "exact") and all_cnot else "tableau"), pairs=two.size * b,
                     greedy_rejected=0, exact_tested=0, exact_admitted=0, cache_hits=0, cache_misses=0, greedy_s=0.0, exact_s=0.0)
    t0 = time.perf_counter()
    basis = None
    if method in ("basis", "exact") and all_cnot:
        _, basis = _greedy_stats(_generators_packed(tab, k), n)
        trials = np.stack([_packed_cnot(basis, n, c, t) for _, c, t in entries], axis=0)  # (A, B, m)
        ok = _greedy_within(trials.reshape(two.size * b, -1), n, w_max)
        mask[:, two] = ok.reshape(two.size, b).T
    else:
        trials = np.repeat(tab[None], two.size, axis=0).reshape(two.size * b, *tab.shape[1:])
        action_table.apply(trials, np.repeat(two, b))
        w_star, _ = _greedy_stats(_generators_packed(trials, k), n)
        mask[:, two] = (w_star <= w_max).reshape(two.size, b).T
    t1 = time.perf_counter()
    rejected_b, rejected_j = np.nonzero(~mask[:, two])  # state index, position in ``two``
    if stats is not None:
        stats["greedy_rejected"] = int(rejected_b.size)
        stats["greedy_s"] = t1 - t0
    if not exact or rejected_b.size == 0:
        return mask
    if all_cnot:
        hits0, misses0 = (cache.hits, cache.misses) if cache is not None else (0, 0)
        light = _light_elements(basis, n, w_max + 1, cache)
        # identical states (a collapsed policy puts many episodes in the same
        # state at the same step) share one test per action
        _, state_uid = np.unique(basis.view(np.dtype((np.void, basis.dtype.itemsize * basis.shape[1]))).ravel(), return_inverse=True)
        pair_key = state_uid.ravel()[rejected_b] * two.size + rejected_j
        uniq, first, inverse = np.unique(pair_key, return_index=True, return_inverse=True)
        ub, uj = rejected_b[first], rejected_j[first]
        ctrl = np.array([c for _, c, _ in entries], dtype=np.int64)[uj]
        targ = np.array([t for _, _, t in entries], dtype=np.int64)[uj]
        heavy_rows = packed_weight(trials[uj, ub], n) > w_max  # (P, m) rows of the gate-transformed basis above the bound
        heavy_bits = (heavy_rows.astype(np.uint32) << np.arange(m, dtype=np.uint32)).sum(axis=1)
        admit = _exact_cnot_admits(light, ub, ctrl, targ, heavy_bits, n, m, w_max, chunk_rows)[inverse.ravel()]
        if stats is not None:
            stats["exact_unique"] = int(uniq.size)
            if cache is not None:
                stats["cache_hits"], stats["cache_misses"] = cache.hits - hits0, cache.misses - misses0
    else:
        admit = np.zeros(rejected_b.size, dtype=bool)
        per = max(1, chunk_rows // max(1 << m, 1))
        for start in range(0, rejected_b.size, per):
            sl = slice(start, start + per)
            trial = tab[rejected_b[sl]].copy()
            action_table.apply(trial, two[rejected_j[sl]])
            admit[sl] = ldpc_feasible(trial, k, w_max, exact_max_generators)
    mask[rejected_b, two[rejected_j]] = admit
    if stats is not None:
        stats["exact_tested"] = int(rejected_b.size)
        stats["exact_admitted"] = int(admit.sum())
        stats["exact_s"] = time.perf_counter() - t1
    return mask


# --------------------------------------------------------------------------
# objective of an LDPC-constrained target
# --------------------------------------------------------------------------
LDPC_FORMS = ("penalty", "unnormalized", "final", "schedule", "shaped", "terminal")
# The form the comparison of experiments/ldpc_scaling/REPORT.md selected: the
# only one of the four that leaves the distance-1 trap of the penalty form at
# n = 12 and 16 (distance-3 codes in about two thirds of the episodes on three
# of four cells) and the only one that produced codes within the bound there;
# its trained policies still sit at w* one to four above the bound, so the
# sparsity part remains open (see the report).
DEFAULT_LDPC_FORM = "unnormalized"
OBJECTIVE_EXACT_GENERATORS = 15


class LdpcObjective:
    """Reward ``f(tab, k) -> (batch,)`` and success criterion
    ``success(tab, k, state_reward) -> (batch,) bool`` of an LDPC-constrained
    ``[[n, k, d]]`` target with bottleneck bound ``w_max`` (and optionally
    a qubit-degree bound ``deg_max``).

    Base terms on every state: the Knill-Laflamme term over the Paulis of
    weight below ``d`` (:func:`autoqec_sim.reward.kl_reward`) and the
    padding penalty (:func:`autoqec_sim.reward.low_weight_stabilizer_penalty`
    at weight 1). A *candidate* is a state with every error detected and
    no idle qubit. Forms:

    ``"penalty"``
        The rediscovery regression's form: normalized base (fraction of
        undetected errors, fraction of weight-1 Paulis in ``S``) plus
        ``-scale * excess`` on every state. Success iff the reward is 0.
        At ``n >= 12`` one undetected error is worth ``1/630`` to
        ``1/1128`` against ``scale = 0.25`` per unit of ``w*`` excess, and
        PPO converges to sparse groups of distance 1
        (``experiments/scaling/REPORT.md``).
    ``"unnormalized"``
        Per-error base: the number of undetected errors and the number of
        weight-1 stabilizers, one unit each, plus ``-scale * excess``, so
        that one undetected error costs ``1 / scale`` units of ``w*``
        excess. The sum is divided by ``r0``, the number of undetected
        errors of the initial state (``|0>`` ancillas, bare logical
        qubits), a global constant that keeps the reward within about
        ``[-1, 0]`` for the value head and changes neither the optimum nor
        the normalized-advantage policy gradient. Success iff the reward
        is 0.
    ``"final"``
        Normalized base only; the constraint is hard and terminal:
        success iff the state is a candidate with exact ``w* <= w_max``.
        No sparsity signal on the trajectory.
    ``"schedule"``
        Distance first. Normalized base until :meth:`activate` is called
        (the trainer does so once the success rate at the target distance
        exceeds a threshold); from then on the penalty form. Before
        activation success is any candidate, after it the reward must be 0.
    ``"shaped"``
        The unnormalized base divided by ``r0`` minus ``shaping * excess``
        in reward units on every state (greedy bound on ordinary states,
        exact on candidates), so that one unit of ``w*`` excess costs
        ``shaping * r0`` undetected errors (3.75 to 15 at ``[[12, 2]]`` for
        ``shaping`` 0.05 to 0.2, against 0.25 in the unnormalized form).
        Success iff the reward is 0.
    ``"terminal"``
        Count-unit terminal signal. Ordinary states get the unnormalized
        base divided by ``r0`` and no sparsity term; a candidate ends the
        episode with reward ``1 - h / m``, where ``h`` is the number of
        generators of the sparsest basis with weight above ``w_max``
        (:func:`heavy_generator_count`, exact) and ``m = n - k``, so one
        excess generator costs one unit out of ``m`` and a feasible code
        scores 1. Success (episode end) is any candidate; feasibility is
        ``h = 0``, which :meth:`feasible` and the experiment check.

    Cost. On ordinary states ``w*`` is the greedy pairwise-reduction bound
    of this module (a few ms per batch; equal to the exact value on 95 to
    99 percent of random states); on candidate states the exact ``w*``
    replaces it (``2^(n - k)`` group elements, ``exact_generators`` at
    most), where it decides success. ``last`` holds the terms of the last
    evaluated batch for diagnostics.
    """

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        w_max: int,
        form: str = DEFAULT_LDPC_FORM,
        scale: float = 0.25,
        deg_max: int | None = None,
        exact_generators: int = OBJECTIVE_EXACT_GENERATORS,
        shaping: float = 0.1,
        device=None,
        chunk: int = 1 << 16,
    ):
        from .errors import ErrorSet

        if form not in LDPC_FORMS:
            raise ValueError(f"form must be one of {LDPC_FORMS}, got {form!r}")
        self.n, self.k, self.d = int(n), int(k), int(d)
        self.w_max = int(w_max)
        self.deg_max = deg_max
        self.form = form
        self.scale = float(scale)
        self.shaping = float(shaping)
        self.exact_generators = int(exact_generators)
        self.errors = ErrorSet.up_to_weight(self.n, self.d - 1)
        self.num_errors = len(self.errors)
        self.active = form in ("penalty", "unnormalized", "shaped")
        self.exact_evals = 0
        self.last: dict = {}
        self.r0 = self._initial_count()
        # Optional torch device for the base terms (the Knill-Laflamme and
        # padding counts, the bulk of the cost above n = 16): the error set
        # and the weight-1 Paulis live on the device as float32, the tableau
        # batch is copied once per call, and the counts come back as numpy.
        # The w* terms stay numpy (candidates only, packed uint64 rows).
        self.device = None
        self.chunk = int(chunk)
        if device is not None:
            import torch

            from .backends import torch_backend as tb
            from .errors import paulis_up_to_weight

            self.torch, self.tb = torch, tb
            self.device = torch.device(device)
            self._errors_t = torch.as_tensor(self.errors.paulis, device=self.device).to(torch.float32)
            w1, _ = paulis_up_to_weight(self.n, 1)
            self._weight1_t = torch.as_tensor(w1, device=self.device).to(torch.float32)

    def _initial_count(self) -> int:
        from .reward import undetected_mask
        from .tableau import identity_tableau

        return int(undetected_mask(identity_tableau(self.n, 1), self.k, self.errors.paulis).sum())

    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Switch the LDPC term on (``"schedule"`` form)."""
        self.active = True

    @property
    def uses_penalty(self) -> bool:
        return self.active

    def base_terms(self, tab: np.ndarray, k: int):
        """``(undetected counts (B,), weight-1 stabilizer counts (B,))``,
        on the torch device when one was given, else numpy."""
        if self.device is not None:
            tabt = self.tb.from_numpy(tab, self.device)
            und = self.tb.undetected_mask(tabt, k, self._errors_t, self.chunk).sum(dim=1)
            pad = self.tb.in_stabilizer_mask(tabt, k, self._weight1_t, self.chunk).sum(dim=1)
            return und.cpu().numpy().astype(np.int64), pad.cpu().numpy().astype(np.int64)
        from .errors import paulis_up_to_weight
        from .reward import in_stabilizer_mask, undetected_mask

        und = undetected_mask(tab, k, self.errors.paulis).sum(axis=1)
        w1, _ = paulis_up_to_weight(self.n, 1)
        pad = in_stabilizer_mask(tab, k, w1).sum(axis=1)
        return und, pad

    def excess(self, tab: np.ndarray, k: int, exact: bool) -> np.ndarray:
        st = sparsest_basis_stats(tab, k, self.w_max, self.exact_generators if exact else 0)
        ex = np.maximum(0, st["w_star"] - self.w_max).astype(float)
        if self.deg_max is not None:
            ex += np.maximum(0, st["max_degree"] - self.deg_max)
        if exact:
            self.exact_evals += int(tab.shape[0])
        return ex

    def heavy(self, tab: np.ndarray, k: int) -> np.ndarray:
        """Exact number of heavy generators (:func:`heavy_generator_count`)."""
        self.exact_evals += int(tab.shape[0])
        return heavy_generator_count(tab, k, self.w_max, self.exact_generators)

    def __call__(self, tab: np.ndarray, k: int) -> np.ndarray:
        und, pad = self.base_terms(tab, k)
        cand = (und == 0) & (pad == 0)
        if self.form in ("unnormalized", "shaped", "terminal"):
            base = -(und + pad).astype(float)
            unit = 1.0 / max(self.r0, 1)
        else:
            base = -und / float(self.num_errors) - pad / float(3 * self.n)
            unit = 1.0
        r = base.copy()
        if self.active:
            ex = self.excess(tab, k, exact=False)
            if cand.any():
                idx = np.flatnonzero(cand)
                ex[idx] = self.excess(tab[idx], k, exact=True)
            if self.form == "shaped":
                r *= unit
                r -= self.shaping * ex
                unit = 1.0
            else:
                r -= self.scale * ex
        r *= unit
        if self.form == "terminal" and cand.any():
            idx = np.flatnonzero(cand)
            m = self.n - self.k
            r[idx] = 1.0 - self.heavy(tab[idx], k) / float(max(m, 1))
        self.last = {"undetected": und, "weight1": pad, "candidate": cand}
        return r

    def success(self, tab: np.ndarray, k: int, state_reward: np.ndarray) -> np.ndarray:
        """Success criterion for ``CodeDiscoveryEnv(success_fn=...)``. With
        the penalty active a zero reward already certifies exact
        feasibility; in the ``"final"`` form (and before activation of the
        schedule) candidates are checked exactly here. In the ``"terminal"``
        form every candidate ends the episode (reward ``>= 0`` iff
        candidate), feasible or not."""
        ok = np.asarray(state_reward) >= 0.0
        if self.form == "final" and ok.any():
            idx = np.flatnonzero(ok)
            ok = ok.copy()
            ok[idx] = self.excess(tab[idx], k, exact=True) == 0
        return ok

    def feasible(self, tab: np.ndarray, k: int) -> np.ndarray:
        """Exact feasibility of arbitrary states (for verification)."""
        return self.excess(tab, k, exact=True) == 0

    def describe(self) -> dict:
        return {
            "form": self.form,
            "scale": self.scale,
            "shaping": self.shaping if self.form == "shaped" else None,
            "w_max": self.w_max,
            "deg_max": self.deg_max,
            "num_errors": self.num_errors,
            "r0": self.r0,
            "exact_generators": self.exact_generators,
            # how w* is decided on candidates: exact enumeration of the 2^(n-k)
            # group elements, or the greedy pairwise-reduction bound (an upper
            # bound on w*, so feasibility under it is sufficient, not necessary)
            "w_star_test": "exact" if self.n - self.k <= self.exact_generators else "greedy bound",
            "reward_backend": "numpy" if self.device is None else f"torch:{self.device}",
        }
