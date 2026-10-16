# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Knill-Laflamme rewards, vectorized over a batch of tableaux.

Following Olle et al. (arXiv:2311.04750, Eq. 10), an error ``E_mu`` is
detected by the code with stabilizer group ``S`` if it anticommutes with
some stabilizer generator or lies in ``S`` (a degenerate error acts
trivially). ``K_mu = 0`` in that case and ``1`` otherwise, and the reward is
``r = -sum_mu lam_mu K_mu``. Detecting every Pauli of weight ``< d`` is
equivalent to the Knill-Laflamme conditions for correcting all errors of
weight at most ``(d - 1) // 2``, since the conditions involve products
``E_a^dagger E_b`` of two such errors.

Membership in ``S`` is decided without linear algebra: a Pauli that commutes
with all ``n - k`` stabilizer generators lies in the normalizer, and it lies
in ``S`` iff it also commutes with all ``2k`` logical operators, because the
logicals form a symplectic basis of ``N(S) / S``. So ``K_mu = 1`` iff
``E_mu`` commutes with every stabilizer row and anticommutes with at least
one logical row. Both tests are one batched symplectic product of the
error set against the ``n + k`` relevant rows of the tableau.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .errors import ErrorSet
from .symplectic import symplectic_product


def code_rows(tab: np.ndarray, k: int) -> np.ndarray:
    """``(batch, n + k, 2n)``: stabilizers ``k..n-1`` then logical X and Z.

    The stabilizer block comes first so that the first ``n - k`` columns of
    the product matrix are the stabilizer syndromes and the last ``2k`` the
    logical commutation pattern.
    """
    n = (tab.shape[-1] - 1) // 2
    stab = tab[:, n + k : 2 * n, : 2 * n]
    lx = tab[:, :k, : 2 * n]
    lz = tab[:, n : n + k, : 2 * n]
    return np.concatenate([stab, lx, lz], axis=1)


def undetected_mask(
    tab: np.ndarray, k: int, errors: np.ndarray, chunk: int = 1 << 16
) -> np.ndarray:
    """``K_mu`` for every batch element and error: ``(batch, M)`` bool.

    ``errors`` is ``(M, 2n)``; the error set is chunked so that the
    intermediate product ``(batch, chunk, n + k)`` stays bounded.
    """
    n = (tab.shape[-1] - 1) // 2
    rows = code_rows(tab, k)  # (B, n+k, 2n)
    batch = tab.shape[0]
    m = errors.shape[0]
    out = np.empty((batch, m), dtype=bool)
    n_stab = n - k
    for start in range(0, m, chunk):
        e = errors[start : start + chunk]
        prod = symplectic_product(e[None], rows)  # (B, m_c, n+k)
        commutes_all_stab = ~prod[:, :, :n_stab].any(axis=2)
        hits_logical = prod[:, :, n_stab:].any(axis=2)
        out[:, start : start + e.shape[0]] = commutes_all_stab & hits_logical
    return out


def kl_reward(
    tab: np.ndarray,
    k: int,
    errors: ErrorSet,
    normalize: bool = True,
    chunk: int = 1 << 16,
) -> np.ndarray:
    """``r = -sum_mu lam_mu K_mu`` per batch element (shape ``(batch,)``).

    With ``normalize`` the sum is divided by ``sum_mu lam_mu`` so the reward
    lies in ``[-1, 0]`` and ``0`` means every error in the set is detected.
    """
    und = undetected_mask(tab, k, errors.paulis, chunk)
    r = -(und.astype(np.float64) @ errors.lam)
    if normalize:
        total = float(errors.lam.sum())
        if total > 0:
            r = r / total
    return r


def detected_fraction(tab: np.ndarray, k: int, errors: ErrorSet, chunk: int = 1 << 16):
    """Unweighted fraction of the error set that is detected, ``(batch,)``."""
    und = undetected_mask(tab, k, errors.paulis, chunk)
    return 1.0 - und.mean(axis=1)


def batched_distance(
    tab: np.ndarray, k: int, max_weight: int, chunk: int = 1 << 16
) -> np.ndarray:
    """Exact distance of every batch element, capped at ``max_weight + 1``.

    Enumerates Paulis by increasing weight and records the first weight at
    which some undetected error exists. Elements with no undetected error up
    to ``max_weight`` receive ``max_weight + 1`` (a lower bound on their
    distance). Cost grows as ``3^w C(n, w)``; intended for ``n <= 20`` and
    small ``max_weight``.
    """
    from .errors import paulis_of_weight

    batch = tab.shape[0]
    n = (tab.shape[-1] - 1) // 2
    d = np.full(batch, max_weight + 1, dtype=np.int64)
    undecided = np.ones(batch, dtype=bool)
    for w in range(1, max_weight + 1):
        if not undecided.any():
            break
        e = paulis_of_weight(n, w)
        idx = np.flatnonzero(undecided)
        und = undetected_mask(tab[idx], k, e, chunk).any(axis=1)
        d[idx[und]] = w
        undecided[idx[und]] = False
    return d


def distance_reward(
    tab: np.ndarray, k: int, target_distance: int, chunk: int = 1 << 16
) -> np.ndarray:
    """``min(d, target) / target`` per element, where ``d`` is the exact
    distance computed up to ``target_distance - 1``. Reaches 1 iff the code
    has distance at least ``target_distance``."""
    d = batched_distance(tab, k, target_distance - 1, chunk)
    return np.minimum(d, target_distance) / float(target_distance)


def in_stabilizer_mask(tab: np.ndarray, k: int, paulis: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """``(batch, M)`` bool: which Paulis lie in the stabilizer group (they
    commute with every stabilizer and every logical row)."""
    rows = code_rows(tab, k)
    batch = tab.shape[0]
    m = paulis.shape[0]
    out = np.empty((batch, m), dtype=bool)
    for start in range(0, m, chunk):
        e = paulis[start : start + chunk]
        prod = symplectic_product(e[None], rows)
        out[:, start : start + e.shape[0]] = ~prod.any(axis=2)
    return out


def low_weight_stabilizer_penalty(
    tab: np.ndarray, k: int, max_weight: int, chunk: int = 1 << 16
) -> np.ndarray:
    """Minus the fraction of Paulis of weight ``1..max_weight`` that are
    stabilizer elements, ``(batch,)``. Zero iff the stabilizer group has no
    element of weight at most ``max_weight``. With ``max_weight = 1`` this
    penalizes idle qubits (a code padded with unused qubits has a weight-1
    ``Z`` stabilizer), which the Knill-Laflamme reward alone does not
    exclude."""
    from .errors import paulis_up_to_weight

    e, _ = paulis_up_to_weight((tab.shape[-1] - 1) // 2, max_weight)
    if e.shape[0] == 0:
        return np.zeros(tab.shape[0])
    return -in_stabilizer_mask(tab, k, e, chunk).mean(axis=1)


def make_low_weight_penalty(max_weight: int):
    def f(tab, k):
        return low_weight_stabilizer_penalty(tab, k, max_weight)

    return f


class CompositeReward:
    """Weighted sum of reward callables ``f(tab, k) -> (batch,)``."""

    def __init__(self, terms: list[tuple[float, Callable[[np.ndarray, int], np.ndarray]]]):
        self.terms = terms

    def __call__(self, tab: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros(tab.shape[0])
        for wgt, fn in self.terms:
            out += wgt * fn(tab, k)
        return out


def make_kl_reward(errors: ErrorSet, normalize: bool = True):
    """Closure ``f(tab, k)`` for use as an environment reward."""

    def f(tab, k):
        return kl_reward(tab, k, errors, normalize=normalize)

    return f


def make_distance_reward(target_distance: int):
    def f(tab, k):
        return distance_reward(tab, k, target_distance)

    return f
