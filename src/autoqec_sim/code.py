# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Stabilizer codes extracted from encoding tableaux.

Works for general (non-CSS) stabilizer codes in the symplectic
representation. Distances are exact for ``n <= 20`` (enumeration of the
normalizer or of low-weight Paulis, whichever is cheaper) and randomized
upper bounds otherwise, following the information-set search of
``qldpc-challenge/research/kit/surrogate.py`` adapted to symplectic form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from . import symplectic as sp
from .errors import count_paulis, paulis_of_weight
from .reward import undetected_mask
from .tableau import BatchTableau

EXACT_MAX_N = 20


@dataclass
class StabilizerCode:
    """An ``[[n, k]]`` stabilizer code with a fixed choice of logicals.

    Attributes
    ----------
    stabilizers
        ``(n - k, 2n)`` binary symplectic rows.
    logical_x, logical_z
        ``(k, 2n)`` rows each.
    signs
        ``(n - k,)``, 1 for a negative generator (``None``: all positive).
    destabilizers
        ``(n - k, 2n)`` rows when known, else ``None``.
    """

    stabilizers: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    signs: np.ndarray = field(default=None)
    destabilizers: np.ndarray = field(default=None)

    def __post_init__(self):
        self.stabilizers = np.asarray(self.stabilizers, dtype=np.uint8)
        self.logical_x = np.asarray(self.logical_x, dtype=np.uint8)
        self.logical_z = np.asarray(self.logical_z, dtype=np.uint8)
        if self.signs is None:
            self.signs = np.zeros(self.stabilizers.shape[0], dtype=np.uint8)
        self.signs = np.asarray(self.signs, dtype=np.uint8)

    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.stabilizers.shape[1] // 2

    @property
    def k(self) -> int:
        return self.logical_x.shape[0]

    @property
    def logicals(self) -> np.ndarray:
        return np.concatenate([self.logical_x, self.logical_z], axis=0)

    @classmethod
    def from_tableau(cls, tab: np.ndarray, k: int, index: int = 0) -> StabilizerCode:
        """Read the code off batch element ``index`` of an encoding tableau
        ``(batch, 2n, 2n + 1)`` with ``k`` logical qubits (first ``k`` input
        qubits carry the logical state)."""
        t = tab[index]
        n = (t.shape[1] - 1) // 2
        return cls(
            stabilizers=t[n + k :, : 2 * n],
            logical_x=t[:k, : 2 * n],
            logical_z=t[n : n + k, : 2 * n],
            signs=t[n + k :, 2 * n],
            destabilizers=t[k:n, : 2 * n],
        )

    @classmethod
    def from_gates(cls, n: int, k: int, gates) -> StabilizerCode:
        """Read the code off the encoding circuit ``gates`` applied to the
        identity tableau on ``n`` qubits. ``gates`` are ``(name, a, b)``
        triples as in ``SearchResult.gates``, with ``b < 0`` for a
        single-qubit gate."""
        tab = BatchTableau(n, 1)
        for gate, a, b in gates:
            tab.apply(gate, a, None if b is None or b < 0 else b)
        return cls.from_tableau(tab.array, k=k)

    @classmethod
    def from_stabilizers(cls, stabilizers, signs=None) -> StabilizerCode:
        """From Pauli strings (``"XZZXI"``, optional leading sign) or
        symplectic rows; logicals and destabilizers are computed."""
        rows = []
        sgn = []
        for s in stabilizers:
            if isinstance(s, str):
                sgn.append(1 if s.startswith("-") else 0)
                rows.append(sp.pauli_to_vec(s))
            else:
                sgn.append(0)
                rows.append(np.asarray(s, dtype=np.uint8))
        rows = np.array(rows, dtype=np.uint8)
        if signs is not None:
            sgn = list(signs)
        tab = sp.complete_tableau(rows, sgn)
        n = rows.shape[1] // 2
        k = n - rows.shape[0]
        return cls.from_tableau(tab[None], k)

    def to_tableau(self) -> np.ndarray:
        """``(1, 2n, 2n + 1)`` encoding tableau of this code."""
        n, k = self.n, self.k
        if self.destabilizers is None:
            tab = sp.complete_tableau(self.stabilizers, self.signs)
            return tab[None]
        tab = np.zeros((2 * n, 2 * n + 1), dtype=np.uint8)
        tab[:k, : 2 * n] = self.logical_x
        tab[k:n, : 2 * n] = self.destabilizers
        tab[n : n + k, : 2 * n] = self.logical_z
        tab[n + k :, : 2 * n] = self.stabilizers
        tab[n + k :, 2 * n] = self.signs
        return tab[None]

    def stabilizer_strings(self) -> list[str]:
        return [
            ("-" if s else "+") + sp.vec_to_pauli(r)
            for r, s in zip(self.stabilizers, self.signs)
        ]

    # ------------------------------------------------------------------
    def is_css(self) -> bool:
        """True if the stabilizer group has a generating set of pure X and
        pure Z operators: ``dim(S cap X-type) + dim(S cap Z-type) = dim(S)``.
        The X-type subgroup is the set of generator combinations whose Z
        part vanishes, the kernel of the Z-part matrix; likewise for Z."""
        n = self.n
        s = self.stabilizers
        dim_sx = sp.kernel(s[:, n:].T).shape[0]
        dim_sz = sp.kernel(s[:, :n].T).shape[0]
        return dim_sx + dim_sz == s.shape[0]

    def stabilizer_weights(self) -> np.ndarray:
        """Weight of each generator, ``(n - k,)``."""
        return sp.pauli_weight(self.stabilizers)

    # ------------------------------------------------------------------
    def _group_elements(self, gens: np.ndarray, chunk: int = 1 << 18):
        """Yield ``(2^m, 2n)`` chunks of all products of ``gens``."""
        m = gens.shape[0]
        total = 1 << m
        g32 = gens.astype(np.float32)
        for start in range(0, total, chunk):
            idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
            coef = ((idx[:, None] >> np.arange(m)) & 1).astype(np.float32)
            yield (coef @ g32).astype(np.int64) & 1, coef

    def distance_exact(self, max_weight: int | None = None, force: bool = False) -> int:
        """Exact distance: minimum weight over ``N(S) \\ S``.

        Chooses between enumerating the ``2^(n+k)`` normalizer elements and
        enumerating low-weight Paulis up to ``max_weight`` (default ``n``),
        whichever is cheaper. Refuses ``n > 20`` unless ``force``.
        """
        n, k = self.n, self.k
        if n > EXACT_MAX_N and not force:
            raise ValueError(f"exact distance limited to n <= {EXACT_MAX_N}; use distance_upper_bound")
        if k == 0:
            return 0
        if max_weight is None:
            max_weight = n
        normalizer_cost = 1 << (n + k)
        weight_cost = count_paulis(n, max_weight)
        if normalizer_cost <= weight_cost:
            return self._distance_by_normalizer()
        return self._distance_by_weight(max_weight)

    def _distance_by_normalizer(self) -> int:
        n = self.n
        gens = np.concatenate([self.stabilizers, self.logicals], axis=0)
        m_stab = self.stabilizers.shape[0]
        best = n + 1
        for elems, coef in self._group_elements(gens):
            nontrivial = coef[:, m_stab:].any(axis=1)
            if not nontrivial.any():
                continue
            w = sp.pauli_weight(elems[nontrivial])
            best = min(best, int(w.min()))
        return best

    def _distance_by_weight(self, max_weight: int) -> int:
        tab = self.to_tableau()
        n = self.n
        for w in range(1, max_weight + 1):
            e = paulis_of_weight(n, w)
            if undetected_mask(tab, self.k, e).any():
                return w
        return max_weight + 1

    def distance_upper_bound(self, trials: int = 2000, seed: int = 0, pair_depth: int = 10):
        """Randomized information-set upper bound on the distance and a
        witness logical operator (symplectic row).

        Each trial row-reduces a basis of ``N(S)`` with the ``2n`` columns
        visited in random order; the reduced rows are normalizer elements
        and the lightest one outside ``S`` (anticommuting with some logical)
        bounds the distance. Pairwise sums of the lightest rows close most of
        the gap to minima realized only as short combinations. Monte Carlo:
        raise ``trials`` until the value stops dropping.
        """
        n = self.n
        basis = np.concatenate([self.stabilizers, self.logicals], axis=0)
        logs = self.logicals
        rng = np.random.default_rng(seed)
        best_w, best_v = n + 1, None

        def consider(rows):
            nonlocal best_w, best_v
            if rows.shape[0] == 0:
                return
            w = sp.pauli_weight(rows)
            nz = sp.symplectic_product(rows, logs).any(axis=1) & (w > 0)
            cand = np.flatnonzero(nz)
            if cand.size:
                i = cand[np.argmin(w[cand])]
                if int(w[i]) < best_w:
                    best_w, best_v = int(w[i]), rows[i].copy()

        for _ in range(trials):
            perm = rng.permutation(2 * n)
            red, _ = sp.rref(basis[:, perm])
            red = red[:, np.argsort(perm)]
            consider(red)
            if pair_depth > 1 and red.shape[0] >= 2:
                w = sp.pauli_weight(red)
                light = red[np.argsort(w)[: min(pair_depth, red.shape[0])]]
                iu = np.triu_indices(light.shape[0], 1)
                consider(light[iu[0]] ^ light[iu[1]])
        return best_w, best_v

    # ------------------------------------------------------------------
    def weight_enumerators(self):
        """``(A, B)``: weight distributions of ``S`` (length ``n + 1``) and of
        ``N(S)``. Invariant under qubit permutations and local Cliffords, so
        equal enumerators are a necessary condition for equivalence. Needs
        ``n + k <= 24`` or so."""
        n = self.n
        a = np.zeros(n + 1, dtype=np.int64)
        for elems, _ in self._group_elements(self.stabilizers):
            a += np.bincount(sp.pauli_weight(elems), minlength=n + 1)
        b = np.zeros(n + 1, dtype=np.int64)
        gens = np.concatenate([self.stabilizers, self.logicals], axis=0)
        for elems, _ in self._group_elements(gens):
            b += np.bincount(sp.pauli_weight(elems), minlength=n + 1)
        return a, b

    def same_weight_enumerators(self, other: StabilizerCode) -> bool:
        if (self.n, self.k) != (other.n, other.k):
            return False
        a1, b1 = self.weight_enumerators()
        a2, b2 = other.weight_enumerators()
        return bool(np.array_equal(a1, a2) and np.array_equal(b1, b2))

    def parameters(self, **kw) -> tuple[int, int, int]:
        return self.n, self.k, self.distance_exact(**kw)

    # ------------------------------------------------------------------
    def to_stim_tableau(self):
        from .tableau import to_stim_tableau

        return to_stim_tableau(self.to_tableau(), 0)

    def stim_stabilizers(self):
        """Stabilizer generators as ``stim.PauliString`` objects."""
        from .tableau import row_to_pauli_string

        rows = np.concatenate([self.stabilizers, self.signs[:, None]], axis=1)
        return [row_to_pauli_string(r) for r in rows]



@dataclass
class SubsystemCode:
    """A subsystem code given by independent generators of its gauge group.

    Attributes
    ----------
    gauge
        ``(g, 2n)`` binary symplectic rows, independent, every generator
        taken with sign ``+1``.
    stabilizers
        ``(s, 2n)`` rows spanning the center of the gauge group, the
        stabilizer group of the code.
    logical_x, logical_z
        ``(k, 2n)`` bare logical operators: they commute with every gauge
        generator and are not in the gauge group.
    fixed
        The stabilizer code with generators ``stabilizers`` plus a maximal
        commuting set of gauge generators (one gauge fixing). It carries
        the same bare logicals and stands in for the distance and the
        hook-order search of a memory circuit.

    The memory circuit of :mod:`autoqec_sim.memory_objective` measures the
    gauge generators, each with its own ancilla, and forms each stabilizer
    as the parity of the gauge outcomes whose product it is
    (:meth:`stabilizer_factors`).
    """

    gauge: np.ndarray
    stabilizers: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    fixed: StabilizerCode

    @classmethod
    def from_gauge(cls, gauge) -> SubsystemCode:
        """From gauge generators as Pauli strings (a leading sign is
        ignored) or symplectic rows; dependent generators are dropped."""
        rows = np.array([sp.pauli_to_vec(g) if isinstance(g, str) else (np.asarray(g, dtype=np.uint8) & 1) for g in gauge], dtype=np.uint8)
        G = rows[sp.independent_rows(rows)]
        n2 = G.shape[1]
        # the center: products c G of generators that commute with every generator
        gram = sp.symplectic_product(G, G)
        coeff = sp.kernel(gram)
        center = ((coeff.astype(np.int64) @ G.astype(np.int64)) % 2).astype(np.uint8) if coeff.shape[0] else np.zeros((0, n2), dtype=np.uint8)
        S = center[sp.independent_rows(center)] if center.shape[0] else center
        # bare logicals: the normalizer of the gauge group modulo the center
        quotient = []
        cur = S.copy()
        for v in sp.normalizer_basis(G):
            if not sp.in_rowspace(v, cur):
                quotient.append(v)
                cur = np.vstack([cur, v[None]])
        pairs, iso = sp.symplectic_gram_schmidt(quotient)
        if iso:
            raise ValueError("the normalizer of the gauge group modulo its center is not symplectic")
        lx = np.array([u for u, _ in pairs], dtype=np.uint8).reshape(len(pairs), n2)
        lz = np.array([v for _, v in pairs], dtype=np.uint8).reshape(len(pairs), n2)
        # one gauge fixing: a maximal commuting set of generators independent of the center
        fixed_rows = S.copy()
        chosen: list[np.ndarray] = []
        for g in G:
            if chosen and np.any(sp.symplectic_product(g[None], np.array(chosen))):
                continue
            if sp.in_rowspace(g, fixed_rows):
                continue
            chosen.append(g)
            fixed_rows = np.vstack([fixed_rows, g[None]])
        fixed = StabilizerCode.from_stabilizers(fixed_rows)
        if fixed.k != lx.shape[0]:
            raise ValueError("gauge fixing left a different number of logical qubits than the bare logicals")
        return cls(G, S, lx, lz, fixed)

    @property
    def n(self) -> int:
        return self.gauge.shape[1] // 2

    @property
    def k(self) -> int:
        return self.logical_x.shape[0]

    @property
    def num_gauge(self) -> int:
        return self.gauge.shape[0]

    def gauge_strings(self) -> list[str]:
        return [sp.vec_to_pauli(row) for row in self.gauge]

    def stabilizer_strings(self) -> list[str]:
        return [sp.vec_to_pauli(row) for row in self.stabilizers]

    def stabilizer_factors(self) -> tuple[list[list[int]], list[int]]:
        """For each stabilizer, the indices of the gauge generators whose
        product it is (unique, the generators being independent) and the
        sign of that product, ``+1`` or ``-1``, relative to the stabilizer
        as a positive Pauli. Raises ``ValueError`` when the factors of some
        stabilizer do not pairwise commute, since their outcomes measured in
        one round would not multiply to the stabilizer's eigenvalue."""
        import stim

        if self.stabilizers.shape[0] == 0:
            return [], []
        coeff = sp.gf2_solve(self.gauge.T, self.stabilizers.T)  # (g, s)
        factors, signs = [], []
        for i in range(self.stabilizers.shape[0]):
            idx = [j for j in range(self.num_gauge) if coeff[j, i]]
            sub = self.gauge[idx]
            if np.any(sp.symplectic_product(sub, sub)):
                raise ValueError(f"stabilizer {i} is a product of gauge generators that do not all commute")
            prod = stim.PauliString(self.n)
            for j in idx:
                prod = prod * stim.PauliString(sp.vec_to_pauli(self.gauge[j]))
            signs.append(int(prod.sign.real))
            factors.append(idx)
        return factors, signs

    def parameters(self) -> tuple[int, int, int]:
        """``(n, k, d)`` with ``d`` the exact distance of the gauge-fixed
        stabilizer code."""
        return (self.n, self.k, self.fixed.distance_exact())

def codes_from_batch(tab: np.ndarray, k: int) -> list[StabilizerCode]:
    return [StabilizerCode.from_tableau(tab, k, b) for b in range(tab.shape[0])]


def expected_normalizer_size(n: int, k: int) -> int:
    return 1 << (n + k)


def weight_count(n: int, w: int) -> int:
    return 3**w * comb(n, w)
