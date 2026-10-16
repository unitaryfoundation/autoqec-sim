# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""GF(2) linear algebra and the symplectic representation of Pauli operators.

A Pauli operator on ``n`` qubits (up to phase) is a binary vector of length
``2n``: the first ``n`` entries are the X bits, the last ``n`` the Z bits.
Two Paulis commute iff their symplectic product ``x . z' + z . x'`` is zero
mod 2. Batched products are computed with float32 matrix products followed
by a parity reduction, which maps to a GEMM kernel; the entries never exceed
``2n`` so float32 is exact.
"""

from __future__ import annotations

import numpy as np

_PAULI_BITS = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_BITS_PAULI = {v: k for k, v in _PAULI_BITS.items()}


def pauli_to_vec(s: str) -> np.ndarray:
    """``"XZIY"`` to a length ``2n`` binary vector. A leading sign is ignored."""
    s = s.lstrip("+-")
    n = len(s)
    v = np.zeros(2 * n, dtype=np.uint8)
    for i, ch in enumerate(s.upper()):
        x, z = _PAULI_BITS[ch]
        v[i] = x
        v[n + i] = z
    return v


def vec_to_pauli(v: np.ndarray) -> str:
    v = np.asarray(v)
    n = v.shape[-1] // 2
    return "".join(_BITS_PAULI[(int(v[i]), int(v[n + i]))] for i in range(n))


def pauli_weight(v: np.ndarray) -> np.ndarray:
    """Number of non-identity sites of each Pauli (last axis has length 2n)."""
    v = np.asarray(v)
    n = v.shape[-1] // 2
    return np.count_nonzero(v[..., :n] | v[..., n:], axis=-1)


def swap_halves(v: np.ndarray) -> np.ndarray:
    """Apply the symplectic form: ``(x | z) -> (z | x)``."""
    n = v.shape[-1] // 2
    return np.concatenate([v[..., n:], v[..., :n]], axis=-1)


def symplectic_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise symplectic products mod 2.

    ``a`` has shape ``(..., M, 2n)`` and ``b`` shape ``(..., K, 2n)``; the
    leading axes broadcast. Returns ``(..., M, K)`` uint8 with 1 where the
    two Paulis anticommute.
    """
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(swap_halves(np.asarray(b)), dtype=np.float32)
    prod = np.matmul(a32, np.swapaxes(b32, -1, -2))
    return (prod.astype(np.int64) & 1).astype(np.uint8)


# --------------------------------------------------------------------------
# GF(2) elimination
# --------------------------------------------------------------------------
def rref(m: np.ndarray):
    """Reduced row echelon form over GF(2). Returns ``(R, pivots)`` where
    ``R`` holds only the nonzero rows."""
    m = (np.asarray(m, dtype=np.uint8) & 1).copy()
    rows, cols = m.shape
    r = 0
    pivots = []
    for c in range(cols):
        if r >= rows:
            break
        piv = np.flatnonzero(m[r:, c])
        if piv.size == 0:
            continue
        p = r + piv[0]
        if p != r:
            m[[r, p]] = m[[p, r]]
        others = np.flatnonzero(m[:, c])
        others = others[others != r]
        m[others] ^= m[r]
        pivots.append(c)
        r += 1
    return m[:r], pivots


def rank(m: np.ndarray) -> int:
    return rref(m)[0].shape[0]


def kernel(m: np.ndarray) -> np.ndarray:
    """Basis of ``{v : m v = 0}`` over GF(2), as rows."""
    m = np.asarray(m, dtype=np.uint8)
    cols = m.shape[1]
    r, pivots = rref(m)
    free = [c for c in range(cols) if c not in pivots]
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for row, p in zip(r, pivots):
            if row[f]:
                basis[i, p] = 1
    return basis


def in_rowspace(v: np.ndarray, m: np.ndarray) -> bool:
    m = np.asarray(m, dtype=np.uint8)
    if m.shape[0] == 0:
        return not np.any(v)
    return rank(np.vstack([m, np.asarray(v, dtype=np.uint8)[None]])) == rank(m)


def independent_rows(m: np.ndarray) -> np.ndarray:
    """Indices of a maximal linearly independent subset of the rows of ``m``,
    chosen greedily in order."""
    keep = []
    cur = np.zeros((0, m.shape[1]), dtype=np.uint8)
    for i, row in enumerate(np.asarray(m, dtype=np.uint8)):
        if not in_rowspace(row, cur):
            keep.append(i)
            cur = np.vstack([cur, row[None]])
    return np.array(keep, dtype=np.intp)


# --------------------------------------------------------------------------
# normalizer, logical operators, and tableau completion
# --------------------------------------------------------------------------
def normalizer_basis(stabilizers: np.ndarray) -> np.ndarray:
    """Basis (rows) of the Paulis commuting with every row of ``stabilizers``."""
    s = np.asarray(stabilizers, dtype=np.uint8)
    n2 = s.shape[1]
    if s.shape[0] == 0:
        return np.eye(n2, dtype=np.uint8)
    return kernel(swap_halves(s))


def symplectic_gram_schmidt(vectors: np.ndarray):
    """Pair up ``vectors`` (rows spanning a symplectic subspace) into
    ``(u_i, v_i)`` with ``<u_i, v_j> = delta_ij`` and all other products 0.

    Returns ``(pairs, isotropic)`` where ``pairs`` is a list of ``(u, v)``
    and ``isotropic`` holds the vectors that commute with everything
    remaining (the isotropic part). Implements the standard algorithm:
    take the first vector ``u``, find a partner ``v`` with ``<u, v> = 1``,
    then project every other vector ``w`` as ``w += <w, v> u + <w, u> v``.
    """
    rest = [np.asarray(v, dtype=np.uint8) & 1 for v in vectors]
    pairs = []
    isotropic = []
    while rest:
        u = rest.pop(0)
        if not np.any(u):
            continue
        partner = None
        for j, w in enumerate(rest):
            if symplectic_product(u[None], w[None])[0, 0]:
                partner = j
                break
        if partner is None:
            isotropic.append(u)
            continue
        v = rest.pop(partner)
        new_rest = []
        for w in rest:
            cu = symplectic_product(w[None], v[None])[0, 0]
            cv = symplectic_product(w[None], u[None])[0, 0]
            w2 = w ^ (u if cu else 0) ^ (v if cv else 0)
            if np.any(w2):
                new_rest.append(w2.astype(np.uint8))
        rest = new_rest
        pairs.append((u, v))
    return pairs, isotropic


def logical_operators(stabilizers: np.ndarray):
    """``(logical_x, logical_z)`` for the code with the given stabilizer
    generators. Each has shape ``(k, 2n)``; ``<X_i, Z_j> = delta_ij`` and
    all other pairs commute."""
    s = np.asarray(stabilizers, dtype=np.uint8)
    n2 = s.shape[1]
    norm = normalizer_basis(s)
    # Quotient N(S)/S: drop normalizer vectors that lie in the stabilizer span.
    quotient = []
    cur = s.copy() if s.shape[0] else np.zeros((0, n2), dtype=np.uint8)
    for v in norm:
        if not in_rowspace(v, cur):
            quotient.append(v)
            cur = np.vstack([cur, v[None]])
    pairs, iso = symplectic_gram_schmidt(quotient)
    if iso:
        raise RuntimeError("normalizer quotient is not symplectic")
    lx = np.array([u for u, _ in pairs], dtype=np.uint8).reshape(len(pairs), n2)
    lz = np.array([v for _, v in pairs], dtype=np.uint8).reshape(len(pairs), n2)
    return lx, lz


def gf2_solve(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    """One solution ``x`` of ``m x = t`` over GF(2) for every column of ``t``.

    ``m`` is ``(r, c)``, ``t`` is ``(r, s)``; returns ``(c, s)``. Raises
    ``ValueError`` if some column has no solution.
    """
    m = np.asarray(m, dtype=np.uint8) & 1
    t = np.asarray(t, dtype=np.uint8) & 1
    _, c = m.shape
    aug = np.concatenate([m, t], axis=1)
    red, pivots = rref(aug)
    if any(p >= c for p in pivots):
        raise ValueError("inconsistent GF(2) system")
    x = np.zeros((c, t.shape[1]), dtype=np.uint8)
    for row, p in zip(red, pivots):
        x[p] = row[c:]
    return x


def complete_tableau(stabilizers: np.ndarray, signs=None):
    """Full ``(2n, 2n + 1)`` tableau in the encoding convention from a set of
    ``n - k`` independent commuting stabilizer generators.

    Rows ``0 .. k-1``: logical X (destabilizer block), rows ``k .. n-1``:
    destabilizers of the code stabilizers, rows ``n .. n+k-1``: logical Z,
    rows ``n+k .. 2n-1``: the given stabilizers (with ``signs``).

    The destabilizers solve the linear system ``<D_i, S_j> = delta_ij`` and
    ``<D_i, L> = 0`` for every logical ``L`` (consistent because the rows
    ``S, L`` are independent), then are made mutually commuting by adding
    ``S_i`` to ``D_j`` whenever ``<D_i, D_j> = 1`` for ``i < j``; adding
    ``S_i`` changes no other product already fixed.
    """
    s = np.asarray(stabilizers, dtype=np.uint8) & 1
    m, n2 = s.shape
    n = n2 // 2
    if rank(s) != m:
        raise ValueError("stabilizer generators are not independent")
    if np.any(symplectic_product(s, s)):
        raise ValueError("stabilizer generators do not commute")
    k = n - m
    lx, lz = logical_operators(s)
    if lx.shape[0] != k:
        raise RuntimeError("logical extraction returned the wrong count")
    rows = np.vstack([s, lx, lz]) if k else s
    target = np.zeros((n + k, m), dtype=np.uint8)
    target[:m, :m] = np.eye(m, dtype=np.uint8)
    dest = gf2_solve(swap_halves(rows), target).T  # (m, 2n)
    for j in range(m):
        for i in range(j):
            if symplectic_product(dest[i][None], dest[j][None])[0, 0]:
                dest[j] ^= s[i]
    tab = np.zeros((2 * n, 2 * n + 1), dtype=np.uint8)
    tab[:k, :n2] = lx
    tab[k:n, :n2] = dest
    tab[n : n + k, :n2] = lz
    tab[n + k :, :n2] = s
    if signs is not None:
        tab[n + k :, n2] = np.asarray(signs, dtype=np.uint8)
    if not is_valid_tableau(tab):
        raise RuntimeError("tableau completion failed the symplectic check")
    return tab


def is_valid_tableau(tab: np.ndarray) -> bool:
    """Check the symplectic constraints of one ``(2n, 2n + 1)`` tableau."""
    n = (tab.shape[1] - 1) // 2
    body = tab[:, : 2 * n]
    prod = symplectic_product(body, body)
    expect = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    expect[:n, n:] = np.eye(n, dtype=np.uint8)
    expect[n:, :n] = np.eye(n, dtype=np.uint8)
    return bool(np.array_equal(prod, expect))
