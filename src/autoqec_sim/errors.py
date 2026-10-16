# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Pauli error sets in symplectic form.

An error set is a pair ``(E, lam)``: ``E`` is a ``(M, 2n)`` uint8 array of
Paulis and ``lam`` a ``(M,)`` float array of weights (probabilities or
hyperparameters, the ``lambda_mu`` of Olle et al.).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

# X, Y, Z as (x bit, z bit)
_PAULI_XZ = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)


def count_paulis(n: int, max_weight: int, min_weight: int = 1) -> int:
    return sum(3**w * comb(n, w) for w in range(min_weight, max_weight + 1))


def paulis_of_weight(n: int, w: int) -> np.ndarray:
    """All ``3^w C(n, w)`` Paulis of weight exactly ``w`` on ``n`` qubits."""
    if w == 0:
        return np.zeros((1, 2 * n), dtype=np.uint8)
    supports = np.array(list(combinations(range(n), w)), dtype=np.intp)  # (C, w)
    c = supports.shape[0]
    # every assignment of {X, Y, Z} to the w sites, as base-3 digits
    kinds = np.indices((3,) * w).reshape(w, -1).T  # (3^w, w)
    m = c * kinds.shape[0]
    sup = np.repeat(supports, kinds.shape[0], axis=0)  # (m, w)
    kd = np.tile(kinds, (c, 1))  # (m, w)
    xz = _PAULI_XZ[kd]  # (m, w, 2)
    out = np.zeros((m, 2 * n), dtype=np.uint8)
    rows = np.repeat(np.arange(m), w)
    out[rows, sup.ravel()] = xz[:, :, 0].ravel()
    out[rows, n + sup.ravel()] = xz[:, :, 1].ravel()
    return out


def paulis_up_to_weight(n: int, max_weight: int, min_weight: int = 1):
    """``(E, weights)`` for all Paulis with ``min_weight <= wt <= max_weight``,
    ordered by increasing weight."""
    blocks = []
    wts = []
    for w in range(min_weight, max_weight + 1):
        p = paulis_of_weight(n, w)
        blocks.append(p)
        wts.append(np.full(p.shape[0], w, dtype=np.int64))
    if not blocks:
        return np.zeros((0, 2 * n), dtype=np.uint8), np.zeros(0, dtype=np.int64)
    return np.concatenate(blocks), np.concatenate(wts)


def pauli_probabilities(errors: np.ndarray, px, py, pz) -> np.ndarray:
    """Probability of each Pauli under independent single-qubit channels.

    ``px, py, pz`` are scalars or length ``n`` arrays; the identity on a site
    has probability ``1 - px - py - pz``.
    """
    n = errors.shape[1] // 2
    px = np.broadcast_to(np.asarray(px, dtype=float), (n,))
    py = np.broadcast_to(np.asarray(py, dtype=float), (n,))
    pz = np.broadcast_to(np.asarray(pz, dtype=float), (n,))
    pi = 1.0 - px - py - pz
    x = errors[:, :n].astype(bool)
    z = errors[:, n:].astype(bool)
    site = np.where(
        x & z, py, np.where(x, px, np.where(z, pz, pi))
    )  # (M, n)
    return np.prod(site, axis=1)


@dataclass
class ErrorSet:
    """A set of Pauli errors with weights.

    ``paulis`` has shape ``(M, 2n)``; ``lam`` has shape ``(M,)``. ``weights``
    (Pauli weights, integers) is kept for the distance reward.
    """

    paulis: np.ndarray
    lam: np.ndarray
    weights: np.ndarray

    @property
    def n(self) -> int:
        return self.paulis.shape[1] // 2

    def __len__(self) -> int:
        return self.paulis.shape[0]

    @classmethod
    def up_to_weight(cls, n: int, max_weight: int, min_weight: int = 1) -> ErrorSet:
        """All Paulis of weight in ``[min_weight, max_weight]`` with unit
        weights. For a target distance ``d`` use ``max_weight = d - 1``:
        detecting every such error is equivalent to the Knill-Laflamme
        conditions for correcting all errors of weight at most
        ``(d - 1) // 2``."""
        p, w = paulis_up_to_weight(n, max_weight, min_weight)
        return cls(p, np.ones(p.shape[0]), w)

    @classmethod
    def from_channel(
        cls, n: int, px, py, pz, max_weight: int, top: int | None = None
    ) -> ErrorSet:
        """Paulis up to ``max_weight`` weighted by their probability under
        independent per-qubit Pauli channels (``lam = p_mu``); optionally
        only the ``top`` most probable. This is the noise-dominant
        error set of the noise-aware reward."""
        p, w = paulis_up_to_weight(n, max_weight, 1)
        lam = pauli_probabilities(p, px, py, pz)
        if top is not None and top < p.shape[0]:
            order = np.argsort(-lam, kind="stable")[:top]
            order.sort()
            p, w, lam = p[order], w[order], lam[order]
        return cls(p, lam, w)

    @classmethod
    def from_paulis(cls, paulis, lam=None) -> ErrorSet:
        from .symplectic import pauli_to_vec, pauli_weight

        arr = np.array(
            [pauli_to_vec(p) if isinstance(p, str) else np.asarray(p, dtype=np.uint8) for p in paulis]
        )
        lam = np.ones(arr.shape[0]) if lam is None else np.asarray(lam, dtype=float)
        return cls(arr, lam, pauli_weight(arr))
