# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Multi-round memory objective: score a stabilizer code by the logical
error rate of a syndrome-extraction memory experiment, not by its encoder.

The encoder single-fault objective of :mod:`autoqec_sim.circuit_noise`
cannot see a local-Clifford deformation of a code (a Hadamard at the end of
the encoder only relabels weight-1 faults, all detected), so it cannot
express bias tailoring (``experiments/noise_aware/runs/REPORT.md``, verdict items 2
to 4 and 6). This module builds, for a code given by its stabilizer
generators and logicals, an ``R``-round syndrome-extraction circuit with
one ancilla per generator, applies a ``autoqec_sim.noise.Profile`` with its
timing model, derives stim's detector error model, and estimates the
logical error rate per round with one of four fidelity tiers:

``proxy``
    closed form from the DEM: the failure probability of the
    maximum-likelihood decoder restricted to fault sets of weight at most
    two (:func:`pfail_low_weight`, ``O(M^2)`` in the number ``M`` of DEM
    mechanisms, exact to ``O(p^3)``), plus the exact circuit-level
    distance ``d_circ`` up to weight 4 and the mass of its minimum-weight
    undetectable logical fault sets (:func:`exact_dcirc`, ported from
    ``ion-search/lib.py``). No sampling, no decoder dependency.
``sampled``
    small-shot Monte Carlo from the DEM with an exact maximum-likelihood
    decoder when ``detectors + observables <= exact_ml_bits`` (dynamic
    program over the joint syndrome distribution, ``2^(D+O)`` table; the
    decoder's failure probability is then exact and no shots are needed),
    otherwise a lookup decoder built from the same weight-``<= 2`` fault
    sets with a one-mechanism completion for unseen syndromes.
``bposd``
    full sampling with the board's pinned BP+OSD decoder (``ldpc``,
    optional dependency, validation only).
``tesseract``
    full sampling with Tesseract (``tesseract-decoder``, optional
    dependency), a most-likely-error search decoder that takes the DEM as
    it is, without decomposition; distinct syndromes are decoded once.

:func:`undetectable_sets` lists the minimum-weight undetectable logical
fault sets of a DEM up to weight 3 by probability, for the fault report
of :func:`autoqec_sim.evaluate_code`.

Circuits (:func:`build_memory_circuit`). Every generator ``g_j`` gets an
ancilla prepared in ``|+>``, controlled-``P`` gates (``CX``, ``CY``,
``CZ``) onto its support in a fixed order, then ``H`` and a ``Z``-basis
measurement. Generators that share a qubit with different Pauli letters
cannot be interleaved (their controlled gates on that qubit do not
commute), so the generators are colored into batches of mutually
interleavable generators (for a CSS code: the X block and the Z block)
which run one after another; inside a batch the gates are packed into
layers as soon as possible with one gate per qubit per layer. Under the
sequential timing of the ion presets every two-qubit gate is its own time
slot anyway, so the batch structure costs nothing there. Preparation and
final readout are noiseless ``MPP`` measurements of every generator and of
the logical operators of the chosen basis (``Z`` or ``X`` memory);
detectors compare consecutive measurements of each generator, and the
observables compare the initial and final logical readout. With
``flags="one"`` each generator of weight at least three gets a flag qubit
(``|0>``, two ``CX`` from the ancilla around the middle of the support,
measured and declared a detector) in the Chao-Reichardt pattern; flags
need all-to-all connectivity here.

Connectivity (:class:`Layout`). ``all_to_all_layout`` (the ion chain)
places ancilla ``j`` on qubit ``n + j``. ``square_layout(rows, cols)``
puts the data qubits on the lattice and ancilla sites at every plaquette
center including the boundary ring, with edges only between a site and
its corner data qubits (the rotated-surface-code hardware layout);
generators are assigned to sites by a greedy minimum of the path lengths
to their support, and a controlled gate to a non-adjacent qubit is routed
along a shortest path with the nearest-neighbour CNOT ladder of the
noise-aware experiment (intermediate qubits restored), so non-local
generators pay in gates and depth. A generator that needs routing runs in
a batch of its own. The profile's connectivity graph is replaced by the
layout's graph, so the noise module's placement check applies to the
routed circuit.

Reward (:class:`MemoryReward`). The environment reward of the noise-aware
experiment with the memory term in place of the encoder term:
``min(d, d*)/d* - 1 - q_1`` on every state and, on states at the target
distance with no weight-1 stabilizer, ``weight * log10(p_ref / p_round)``
clipped to ``+-clip`` decades, where ``p_round`` is the per-round logical
error estimate of the chosen tier summed over the memory bases. Such a
state ends the episode (the reward exposes ``success_fn``, which
:class:`autoqec_sim.env.CodeDiscoveryEnv` picks up). Results are cached
by the reduced row echelon form of the stabilizer block, so a code is
evaluated once per run. The memory term depends only on the code, not on
the encoding circuit that produced it.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import stim

from autoqec_sim import noise as nm
from autoqec_sim.noise import Profile

from .code import StabilizerCode
from .graphs import Graph
from .reward import batched_distance, low_weight_stabilizer_penalty
from .symplectic import rref

__all__ = [
    "BASES",
    "TIERS",
    "Layout",
    "MemoryCircuit",
    "MemoryObjective",
    "MemoryResult",
    "MemoryReward",
    "all_to_all_layout",
    "build_memory_circuit",
    "code_key",
    "dem_signature",
    "exact_dcirc",
    "exact_ml_failure",
    "hook_orders",
    "make_memory_reward",
    "memory_dem",
    "per_round",
    "pfail_low_weight",
    "sample_failures",
    "square_layout",
    "undetectable_sets",
]

BASES = ("Z", "X")
TIERS = ("proxy", "sampled", "bposd", "tesseract")
_LETTER = {(1, 0): "X", (1, 1): "Y", (0, 1): "Z"}
_CONTROLLED = {"X": "CX", "Y": "CY", "Z": "CZ"}
# basis change so that a controlled-P becomes a routed controlled-X:
# P = U X U^dagger, time order U^dagger, CX, U
_BASIS_PRE = {"X": (), "Y": ("S_DAG",), "Z": ("H",)}
_BASIS_POST = {"X": (), "Y": ("S",), "Z": ("H",)}


# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    """Physical register: data qubits ``0..n-1``, ancilla sites ``sites``
    (physical indices; flag qubits, when any, take the spare sites), and a
    connectivity graph over every physical qubit (``None`` for
    all-to-all)."""

    n: int
    sites: tuple
    graph: Graph | None
    name: str = ""

    @property
    def num_qubits(self) -> int:
        return max((self.n, *[s + 1 for s in self.sites]))

    def adjacent(self, a: int, b: int) -> bool:
        return True if self.graph is None else self.graph.has_edge(a, b)

    def shortest_path(self, a: int, b: int) -> list[int]:
        if self.graph is None or self.adjacent(a, b):
            return [a, b]
        adj = {q: set() for q in range(self.graph.n)}
        for u, v in self.graph.edges:
            adj[u].add(v)
            adj[v].add(u)
        prev = {a: None}
        frontier = [a]
        while frontier and b not in prev:
            nxt = []
            for u in frontier:
                for v in sorted(adj[u]):
                    if v not in prev:
                        prev[v] = u
                        nxt.append(v)
            frontier = nxt
        if b not in prev:
            raise ValueError(f"no path from {a} to {b} in layout {self.name!r}")
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        return path[::-1]

    def distances_from(self, a: int) -> dict[int, int]:
        """BFS distance from ``a`` to every node (all-to-all: 1 everywhere)."""
        if self.graph is None:
            return {q: (0 if q == a else 1) for q in range(self.num_qubits)}
        adj = {q: set() for q in range(self.graph.n)}
        for u, v in self.graph.edges:
            adj[u].add(v)
            adj[v].add(u)
        dist = {a: 0}
        frontier = [a]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        return dist


def all_to_all_layout(n: int, m: int, flags: int = 0) -> Layout:
    """Ion-chain layout: data ``0..n-1``, ancillas ``n..n+m-1``, flag
    qubits after them, every pair connected."""
    return Layout(n, tuple(range(n, n + m + flags)), None, name=f"all_to_all:{n}+{m + flags}")


def square_layout(rows: int, cols: int) -> Layout:
    """Data qubits row-major on a ``rows x cols`` lattice, one ancilla site
    per plaquette center ``(i - 1/2, j - 1/2)`` for ``i in 0..rows``,
    ``j in 0..cols`` (the boundary ring included), each adjacent to its
    corner data qubits and to nothing else."""
    n = rows * cols
    sites = []
    edges = []
    s = n
    for i in range(rows + 1):
        for j in range(cols + 1):
            for di, dj in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
                r, c = i + di, j + dj
                if 0 <= r < rows and 0 <= c < cols:
                    edges.append((r * cols + c, s))
            sites.append(s)
            s += 1
    graph = Graph(s, tuple(sorted({(min(a, b), max(a, b)) for a, b in edges})), name=f"square_sites:{rows}x{cols}")
    return Layout(n, tuple(sites), graph, name=graph.name)


def layout_for(code_n: int, m: int, connectivity, flags: int = 0) -> Layout:
    """``connectivity`` is ``"all_to_all"``, ``"square:RxC"``, or a
    :class:`Layout`."""
    if isinstance(connectivity, Layout):
        return connectivity
    if connectivity == "all_to_all":
        return all_to_all_layout(code_n, m, flags)
    if isinstance(connectivity, str) and connectivity.startswith("square:"):
        r, c = (int(v) for v in connectivity.split(":")[1].split("x"))
        if r * c != code_n:
            raise ValueError(f"square:{r}x{c} has {r * c} data qubits, the code has {code_n}")
        if flags:
            raise ValueError("flag qubits need all-to-all connectivity in this layout")
        return square_layout(r, c)
    raise ValueError(f"unknown connectivity {connectivity!r}")


def _long_range_cnot(path: list[int]) -> list[tuple]:
    """CNOT from ``path[0]`` to ``path[-1]`` with nearest-neighbour CNOTs
    along the path: the four-gate identity through the middle qubit,
    applied recursively, or a SWAP chain (three CNOTs per SWAP) when that
    is shorter; the intermediate qubits are restored. (The construction of
    the noise-aware experiment's baseline router.)"""
    steps = len(path) - 1
    if steps == 1:
        return [("CX", path[0], path[1])]

    def recursive(p):
        if len(p) == 2:
            return [("CX", p[0], p[1])]
        m = len(p) // 2
        left, right = p[: m + 1], p[m:]
        return recursive(right) + recursive(left) + recursive(right) + recursive(left)

    def swap(u, v):
        return [("CX", u, v), ("CX", v, u), ("CX", u, v)]

    chain = []
    for i in range(steps - 1):
        chain += swap(path[i], path[i + 1])
    chain += [("CX", path[steps - 1], path[steps])]
    for i in range(steps - 2, -1, -1):
        chain += swap(path[i], path[i + 1])
    rec = recursive(path)
    return rec if len(rec) <= len(chain) else chain


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
def _support(row: np.ndarray, n: int, order: str, rng) -> list[tuple[int, str]]:
    """``[(qubit, letter), ...]`` of one symplectic row in the gate order."""
    sup = [(q, _LETTER[(int(row[q]), int(row[n + q]))]) for q in range(n) if row[q] or row[n + q]]
    if order == "random":
        rng.shuffle(sup)
    return sup


class _CompletionCost:
    """``cost(E)``: the fewest extra single-qubit Paulis that complete the
    Pauli ``E`` to an undetected nontrivial logical of the code (0 when
    ``E`` itself is one), computed exactly up to ``max_cost`` with the
    Knill-Laflamme kernel and cached per Pauli."""

    def __init__(self, code: StabilizerCode, max_cost: int):
        from .errors import paulis_of_weight
        from .reward import undetected_mask

        self.tab = code.to_tableau()
        self.k = code.k
        self.n = code.n
        self.max_cost = int(max_cost)
        self._mask = undetected_mask
        self._extra = [paulis_of_weight(code.n, w) for w in range(1, self.max_cost + 1)]
        self.cache: dict[bytes, int] = {}

    def __call__(self, e: np.ndarray) -> int:
        key = e.tobytes()
        c = self.cache.get(key)
        if c is None:
            c = self.max_cost + 1
            if self._mask(self.tab, self.k, e[None])[0, 0]:
                c = 0
            else:
                for w, extra in enumerate(self._extra, start=1):
                    if self._mask(self.tab, self.k, extra ^ e[None]).any():
                        c = w
                        break
            self.cache[key] = c
        return c


def _hook_order(sup: list[tuple[int, str]], n: int, cost: _CompletionCost, target: int, rng, max_perms: int = 300) -> list[tuple[int, str]]:
    """The gate order of one generator whose ancilla hook errors are
    harmless at ``target``: for every proper suffix ``S`` of the order (the
    data error of an ancilla fault after that gate) ``1 + cost(S) >=
    target``, and for every contiguous segment ``S`` between two gates
    (two ancilla faults) ``2 + cost(S) >= target`` (the criterion of the
    ion-search schedule search). Permutations are enumerated for weight
    ``<= 6`` and sampled otherwise; the first acceptable order wins, else
    the one with the largest minimum criterion value."""
    w = len(sup)
    if w <= 2:
        return list(sup)

    def vec(items):
        v = np.zeros(2 * n, dtype=np.uint8)
        for q, p in items:
            if p in ("X", "Y"):
                v[q] ^= 1
            if p in ("Z", "Y"):
                v[n + q] ^= 1
        return v

    if w <= 6:
        candidates = itertools.permutations(sup)
    else:
        candidates = (tuple(sup[i] for i in rng.permutation(w)) for _ in range(max_perms))
    best, best_val = list(sup), -1
    for order in candidates:
        val = 10**6
        for t in range(1, w):
            val = min(val, 1 + cost(vec(order[t:])))
            if val < target:
                break
        if val >= target:
            for i in range(1, w):
                for j in range(i + 1, w):
                    val = min(val, 2 + cost(vec(order[i:j])))
                    if val < target:
                        break
                if val < target:
                    break
        if val > best_val:
            best, best_val = list(order), val
        if val >= target:
            break
    return best


def hook_orders(code: StabilizerCode, target: int | None = None, seed: int = 0) -> list[list[tuple[int, str]]]:
    """Hook-aware gate orders of every generator (:func:`_hook_order`);
    ``target`` defaults to the code's exact distance."""
    n = code.n
    if target is None:
        target = code.distance_exact()
    cost = _CompletionCost(code, max(1, target - 1))
    rng = np.random.default_rng(seed)
    return [_hook_order(_support(code.stabilizers[j], n, "sorted", rng), n, cost, target, rng) for j in range(code.stabilizers.shape[0])]


def _conflict(sa: dict, sb: dict) -> bool:
    """Two generators conflict when they share a qubit with different letters."""
    return any(sb.get(q, p) != p for q, p in sa.items())


def _batches(supports: list[list[tuple[int, str]]], routed: set) -> list[list[int]]:
    """Greedy coloring of the conflict graph into batches of mutually
    interleavable generators; routed generators get a batch of their own.
    Generators are taken in order, so the CSS X block comes before the Z
    block when the generators are listed that way."""
    dicts = [dict(s) for s in supports]
    batches: list[list[int]] = []
    for j in range(len(supports)):
        if j in routed:
            batches.append([j])
            continue
        for b in batches:
            if b[0] in routed:
                continue
            if all(not _conflict(dicts[j], dicts[i]) for i in b):
                b.append(j)
                break
        else:
            batches.append([j])
    return batches


def _assign_sites(layout: Layout, supports: list[list[tuple[int, str]]]) -> list[int]:
    """One ancilla site per generator. All-to-all: site ``j``. Graph:
    greedy by decreasing weight, each generator takes the free site with
    the smallest summed distance to its support (then smallest maximum
    distance, then index)."""
    m = len(supports)
    if layout.graph is None:
        if m > len(layout.sites):
            raise ValueError("layout has fewer ancilla sites than generators")
        return list(layout.sites[:m])
    if m > len(layout.sites):
        raise ValueError(f"layout {layout.name!r} has {len(layout.sites)} sites for {m} generators")
    dist = {s: layout.distances_from(s) for s in layout.sites}
    free = list(layout.sites)
    site = [-1] * m
    for j in sorted(range(m), key=lambda j: (-len(supports[j]), j)):
        qs = [q for q, _ in supports[j]]
        best = min(free, key=lambda s: (sum(dist[s].get(q, 10**6) for q in qs), max(dist[s].get(q, 10**6) for q in qs), s))
        site[j] = best
        free.remove(best)
    return site


# --------------------------------------------------------------------------
# circuit
# --------------------------------------------------------------------------
@dataclass
class MemoryCircuit:
    """The noiseless memory experiment of one code and basis.

    ``body`` is the noisy part (``R`` rounds of syndrome extraction) as a
    TICK-layered skeleton on ``num_qubits`` physical qubits, without
    detectors; ``prefix`` and ``suffix`` are the noiseless ``MPP``
    preparation and readout. :meth:`assemble` joins a (noisy) body with
    them and appends the detectors and observables.
    """

    n: int
    k: int
    rounds: int
    basis: str
    body: stim.Circuit
    prefix: stim.Circuit
    suffix: stim.Circuit
    num_qubits: int
    ancilla: list[int]
    flag: list[int]
    batches: list[list[int]]
    supports: list[list[tuple[int, str]]]
    routed_gates: int
    two_qubit_gates: int
    layers: int
    # body measurement index of (round, generator) and of (round, flag j)
    meas_index: dict = field(default_factory=dict)
    flag_index: dict = field(default_factory=dict)
    layout_name: str = ""

    @property
    def num_generators(self) -> int:
        return len(self.supports)

    def circuit(self) -> stim.Circuit:
        """The noiseless full circuit."""
        return self.assemble(self.body)

    def assemble(self, body: stim.Circuit) -> stim.Circuit:
        m, k = self.num_generators, self.k
        n_pre = self.prefix.num_measurements
        n_body = body.num_measurements
        if n_body != self.body.num_measurements:
            raise ValueError("the noisy body has extra measurement records (heralded channels are not supported here)")
        total = n_pre + n_body + self.suffix.num_measurements
        c = stim.Circuit()
        c += self.prefix
        c += body
        c += self.suffix

        def rec(i):
            return stim.target_rec(i - total)

        for r in range(self.rounds):
            for j in range(m):
                cur = n_pre + self.meas_index[(r, j)]
                prev = j if r == 0 else n_pre + self.meas_index[(r - 1, j)]
                c.append("DETECTOR", [rec(cur), rec(prev)], [j, r])
            for j in range(m):
                if (r, j) in self.flag_index:
                    c.append("DETECTOR", [rec(n_pre + self.flag_index[(r, j)])], [j, r, 1])
        for j in range(m):
            c.append("DETECTOR", [rec(n_pre + n_body + j), rec(n_pre + self.meas_index[(self.rounds - 1, j)])], [j, self.rounds])
        for i in range(k):
            c.append("OBSERVABLE_INCLUDE", [rec(m + i), rec(n_pre + n_body + m + i)], i)
        return c


def _mpp_targets(row: np.ndarray, n: int):
    targets = []
    for q in range(n):
        x, z = int(row[q]), int(row[n + q])
        if not (x or z):
            continue
        if targets:
            targets.append(stim.target_combiner())
        targets.append({(1, 0): stim.target_x, (1, 1): stim.target_y, (0, 1): stim.target_z}[(x, z)](q))
    return targets


def build_memory_circuit(
    code: StabilizerCode,
    rounds: int,
    basis: str = "Z",
    connectivity="all_to_all",
    flags: str = "none",
    order: str = "hook",
    seed: int = 0,
    hook_target: int | None = None,
    orders: list | None = None,
) -> MemoryCircuit:
    """The memory circuit of ``code`` (see the module docstring).

    Parameters
    ----------
    code
        Stabilizer code.
    rounds
        Noisy rounds, at least 1.
    basis
        ``"Z"`` or ``"X"``: the logical operators read out.
    connectivity
        ``"all_to_all"``, ``"square:RxC"``, or a :class:`Layout`.
    flags
        ``"none"`` or ``"one"``.
    order
        ``"hook"`` (hook-aware order per generator, :func:`hook_orders`, at
        ``hook_target``, default the code's exact distance), ``"sorted"``
        (support by qubit index), or ``"random"`` (per generator, from
        ``seed``).
    seed
        Seed of the random orders.
    hook_target
        Target distance of the hook-aware order.
    orders
        Overrides the per-generator orders (``[(qubit, letter), ...]`` per
        generator, as :func:`hook_orders` returns them).

    Returns
    -------
    MemoryCircuit
    """
    if basis not in BASES:
        raise ValueError("basis must be Z or X")
    if flags not in ("none", "one"):
        raise ValueError("flags must be none or one")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if order not in ("hook", "sorted", "random"):
        raise ValueError("order must be hook, sorted, or random")
    n, k = code.n, code.k
    m = code.stabilizers.shape[0]
    rng = np.random.default_rng(seed)
    if orders is not None:
        supports = [list(o) for o in orders]
    elif order == "hook":
        supports = hook_orders(code, hook_target, seed)
    else:
        supports = [_support(code.stabilizers[j], n, order, rng) for j in range(m)]
    flagged = [j for j in range(m) if flags == "one" and len(supports[j]) >= 3]
    layout = layout_for(n, m, connectivity, flags=len(flagged))
    ancilla = _assign_sites(layout, supports)
    flag_q = {}
    if flagged:
        spare = [s for s in layout.sites if s not in ancilla]
        if len(spare) < len(flagged):
            raise ValueError("not enough spare qubits for the flag variant")
        flag_q = dict(zip(flagged, spare))

    # per generator: the list of atoms (gate, qubits) in time order, and
    # whether any controlled gate had to be routed
    atoms: list[list[tuple]] = []
    routed = set()
    routed_gates = 0
    for j in range(m):
        a = ancilla[j]
        seq: list[tuple] = []
        w = len(supports[j])
        for t, (q, p) in enumerate(supports[j]):
            if layout.adjacent(a, q):
                seq.append((_CONTROLLED[p], a, q))
            else:
                path = layout.shortest_path(a, q)
                routed.add(j)
                seq += [(g, q) for g in _BASIS_PRE[p]]
                ladder = _long_range_cnot(path)
                routed_gates += len(ladder) - 1
                seq += ladder
                seq += [(g, q) for g in _BASIS_POST[p]]
            if j in flag_q and t in (0, w - 2):
                seq.append(("CX", a, flag_q[j]))
        atoms.append(seq)
    batches = _batches(supports, routed)

    lines: list[str] = [f"QUBIT_COORDS({q}) {q}" for q in range(layout.num_qubits)]
    meas_index: dict = {}
    flag_index: dict = {}
    n_meas = 0
    layers = 0
    two_q = 0

    def emit(layer_lines):
        nonlocal layers
        if layers:
            lines.append("TICK")
        lines.extend(layer_lines)
        layers += 1

    for r in range(rounds):
        for batch in batches:
            anc = [ancilla[j] for j in batch]
            fl = [flag_q[j] for j in batch if j in flag_q]
            emit(["R " + " ".join(map(str, anc + fl))])
            emit(["H " + " ".join(map(str, anc))])
            # ASAP packing of the batch's atoms with one gate per qubit per layer
            ready: dict[int, int] = {}
            packed: list[list[str]] = []
            for j in batch:
                for atom in atoms[j]:
                    qs = atom[1:]
                    layer = max((ready.get(q, 0) for q in qs), default=0)
                    while len(packed) <= layer:
                        packed.append([])
                    packed[layer].append(atom[0] + " " + " ".join(map(str, qs)))
                    for q in qs:
                        ready[q] = layer + 1
                    if len(qs) == 2:
                        two_q += 1
            for layer in packed:
                emit(layer)
            emit(["H " + " ".join(map(str, anc))])
            emit(["M " + " ".join(map(str, anc + fl))])
            for j in batch:
                meas_index[(r, j)] = n_meas
                n_meas += 1
            for j in batch:
                if j in flag_q:
                    flag_index[(r, j)] = n_meas
                    n_meas += 1
    body = stim.Circuit("\n".join(lines))

    logicals = code.logical_z if basis == "Z" else code.logical_x
    prefix = stim.Circuit()
    suffix = stim.Circuit()
    for row in code.stabilizers:
        prefix.append("MPP", _mpp_targets(row, n))
        suffix.append("MPP", _mpp_targets(row, n))
    for row in logicals:
        prefix.append("MPP", _mpp_targets(row, n))
        suffix.append("MPP", _mpp_targets(row, n))
    return MemoryCircuit(
        n=n, k=k, rounds=rounds, basis=basis, body=body, prefix=prefix, suffix=suffix,
        num_qubits=layout.num_qubits, ancilla=ancilla, flag=[flag_q.get(j, -1) for j in range(m)],
        batches=batches, supports=supports, routed_gates=routed_gates, two_qubit_gates=two_q,
        layers=layers, meas_index=meas_index, flag_index=flag_index, layout_name=layout.name,
    )


# --------------------------------------------------------------------------
# noise and DEM
# --------------------------------------------------------------------------
def _profile_for(profile: Profile, layout_graph: Graph | None) -> Profile:
    return profile.with_graph(None if layout_graph is None else layout_graph.to_topology())


def noisy_memory_circuit(mc: MemoryCircuit, profile: Profile, scale: float = 1.0, layout: Layout | None = None) -> stim.Circuit:
    """The full noisy circuit: ``profile`` at ``scale`` on the body (its
    graph replaced by the layout's), noiseless preparation and readout."""
    graph = None
    if layout is not None:
        graph = layout.graph
    elif mc.layout_name.startswith("square_sites"):
        r, c = (int(v) for v in mc.layout_name.split(":")[1].split("x"))
        graph = square_layout(r, c).graph
    prof = _profile_for(profile, graph)
    kw = {}
    if prof.params.idle.scope == "data":
        kw["n_data"] = mc.n
    noisy = nm.apply(mc.body, prof, scale, **kw)
    return mc.assemble(noisy)


def memory_dem(mc: MemoryCircuit, profile: Profile, scale: float = 1.0, layout: Layout | None = None) -> stim.DetectorErrorModel:
    """Detector error model of the noisy memory circuit (loops flattened,
    no decomposition; stim's approximate conversion only when the profile
    emits channels the exact one refuses)."""
    return nm.dem(noisy_memory_circuit(mc, profile, scale, layout))


@dataclass
class Signature:
    """One DEM as arrays: ``det`` ``(M, W)`` uint64 detector bit words,
    ``obs`` ``(M,)`` uint64 observable mask, ``probs`` ``(M,)``,
    ``num_detectors``, ``num_observables``."""

    det: np.ndarray
    obs: np.ndarray
    probs: np.ndarray
    num_detectors: int
    num_observables: int

    def __len__(self):
        return self.probs.shape[0]

    @property
    def words(self) -> int:
        return self.det.shape[1]

    def matrices(self):
        """Dense ``(H, L)``: detectors x mechanisms and observables x
        mechanisms, ``uint8``."""
        m = len(self)
        h = np.zeros((self.num_detectors, m), dtype=np.uint8)
        lg = np.zeros((self.num_observables, m), dtype=np.uint8)
        for w in range(self.words):
            for b in range(min(64, self.num_detectors - 64 * w)):
                h[64 * w + b] = (self.det[:, w] >> np.uint64(b)) & np.uint64(1)
        for o in range(self.num_observables):
            lg[o] = (self.obs >> np.uint64(o)) & np.uint64(1)
        return h, lg


def dem_signature(dem: stim.DetectorErrorModel) -> Signature:
    d, o = dem.num_detectors, dem.num_observables
    if o > 64:
        raise ValueError("at most 64 observables")
    words = max(1, -(-d // 64))
    det, obs, probs = [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dw = [0] * words
        om = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                v = int(t.val)
                dw[v // 64] ^= 1 << (v % 64)
            elif t.is_logical_observable_id():
                om ^= 1 << int(t.val)
        det.append(dw)
        obs.append(om)
        probs.append(inst.args_copy()[0])
    if not det:
        return Signature(np.zeros((0, words), dtype=np.uint64), np.zeros(0, dtype=np.uint64), np.zeros(0), d, o)
    return Signature(
        np.array(det, dtype=np.uint64), np.array(obs, dtype=np.uint64), np.array(probs, dtype=float), d, o
    )


# --------------------------------------------------------------------------
# closed-form proxy: low-weight ML failure and exact d_circ
# --------------------------------------------------------------------------
def _pairs(m: int):
    iu = np.triu_indices(m, 1)
    return iu[0].astype(np.int64), iu[1].astype(np.int64)


def _joint_ids(*word_arrays):
    """Integer ids for detector signatures given as ``(N_i, W)`` word
    arrays, consistent across the arrays (equal rows get equal ids)."""
    if not word_arrays:
        return []
    if word_arrays[0].shape[1] == 1:
        return [a[:, 0] for a in word_arrays]
    allw = np.concatenate(word_arrays, axis=0)
    # Rows are labelled by their rank in lexicographic order (a lexsort on
    # the word columns; ``np.unique(axis=0)`` argsorts a structured view and
    # is 20 to 50 times slower on the 2M pair rows of an n = 16 DEM). Callers
    # test ``id == 0`` for the zero syndrome, so the ids are shifted by one
    # when the all-zero row is absent.
    order = np.lexsort([allw[:, w] for w in range(allw.shape[1] - 1, -1, -1)])
    srt = allw[order]
    change = np.concatenate([[True], (srt[1:] != srt[:-1]).any(axis=1)])
    ranks = np.cumsum(change) - 1
    if srt.shape[0] and not (srt[0] == 0).all():
        ranks = ranks + 1
    inv = np.empty(allw.shape[0], dtype=np.int64)
    inv[order] = ranks
    out, start = [], 0
    for a in word_arrays:
        out.append(inv[start : start + a.shape[0]])
        start += a.shape[0]
    return out


def _ml_failure_from_sets(sid: np.ndarray, obs: np.ndarray, weight: np.ndarray) -> float:
    """Failure mass of the maximum-likelihood decision over fault sets
    grouped by syndrome id: for every syndrome the decoder returns the
    observable pattern with the largest summed weight and every set with
    another pattern fails (syndrome 0 decodes to no flip)."""
    if sid.shape[0] == 0:
        return 0.0
    order = np.lexsort((obs, sid))
    s, b, w = sid[order], obs[order], weight[order]
    sub = np.flatnonzero(np.concatenate([[True], (s[1:] != s[:-1]) | (b[1:] != b[:-1])]))
    sub_mass = np.add.reduceat(w, sub)
    sub_s = s[sub]
    sub_b = b[sub]
    grp = np.flatnonzero(np.concatenate([[True], sub_s[1:] != sub_s[:-1]]))
    total = np.add.reduceat(sub_mass, grp)
    best = np.maximum.reduceat(sub_mass, grp)
    fail = total - best
    zero = sub_s[grp] == 0
    if zero.any():
        g = int(np.flatnonzero(zero)[0])
        end = grp[g + 1] if g + 1 < grp.shape[0] else sub_mass.shape[0]
        seg_b = sub_b[grp[g] : end]
        seg_m = sub_mass[grp[g] : end]
        fail[g] = float(seg_m[seg_b != 0].sum())
    return float(fail.sum())


def pfail_low_weight(sig: Signature, max_weight: int = 2, max_triples: int = 2_000_000, exclusive: bool = False) -> float:
    """Leading-order failure probability of the maximum-likelihood decoder
    restricted to fault sets of weight at most ``max_weight``.

    Sets are grouped by syndrome, the decoder picks the observable pattern of
    largest summed weight per syndrome, and the rest fails. This is the
    leading term of the failure rate of a code whose circuit-level distance
    is at most ``2 max_weight``, meaningful while the expected number of
    faults per shot (``sig.probs.sum()``) is below about one.

    Parameters
    ----------
    sig
        :class:`Signature` of the detector error model.
    max_weight
        2, or 3 when the number of triples is at most ``max_triples``.
    max_triples
        Cap on the weight-3 enumeration.
    exclusive
        Weight a set by the exact probability of that set alone,
        ``prod p_i prod_{j not in set} (1 - p_j)``, instead of the low-``p``
        asymptote ``prod_{i in set} p_i``.

    Returns
    -------
    float
    """
    m = len(sig)
    if m == 0:
        return 0.0
    p = np.clip(sig.probs, 0.0, 1.0 - 1e-15)
    if exclusive:
        p0 = float(np.prod(1.0 - p))
        q = p / (1.0 - p)
    else:
        p0 = 1.0
        q = p
    words = [sig.det]
    obs = [sig.obs]
    wgt = [q * p0]
    if m >= 2:
        i, j = _pairs(m)
        words.append(sig.det[i] ^ sig.det[j])
        obs.append(sig.obs[i] ^ sig.obs[j])
        wgt.append(q[i] * q[j] * p0)
        if max_weight >= 3 and m >= 3 and m * (m - 1) * (m - 2) // 6 <= max_triples:
            cnt = m * (m - 1) * (m - 2) // 6
            flat = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(m), 3)), dtype=np.int64, count=3 * cnt)
            t = flat.reshape(cnt, 3)
            words.append(sig.det[t[:, 0]] ^ sig.det[t[:, 1]] ^ sig.det[t[:, 2]])
            obs.append(sig.obs[t[:, 0]] ^ sig.obs[t[:, 1]] ^ sig.obs[t[:, 2]])
            wgt.append(q[t[:, 0]] * q[t[:, 1]] * q[t[:, 2]] * p0)
    ids = _joint_ids(*words)
    return _ml_failure_from_sets(np.concatenate(ids), np.concatenate(obs), np.concatenate(wgt))


def _group_hits(dsorted, bsorted, psorted, idx_sorted, weight):
    n = len(dsorted)
    if n < 2:
        return None
    starts = np.flatnonzero(np.concatenate([[True], dsorted[1:] != dsorted[:-1]]))
    ends = np.concatenate([starts[1:], [n]])
    big = np.flatnonzero(ends - starts > 1)
    count, mass, witness = 0, 0.0, None
    splits = {2: 1, 4: 3}[weight]
    for g in big:
        s, e = starts[g], ends[g]
        b = bsorted[s:e]
        if (b == b[0]).all():
            continue
        p = psorted[s:e]
        _, inv = np.unique(b, return_inverse=True)
        cnt_v = np.bincount(inv)
        mass_v = np.bincount(inv, weights=p)
        tot_c, tot_m = cnt_v.sum(), mass_v.sum()
        count += int((cnt_v * (tot_c - cnt_v)).sum() // 2)
        mass += float((mass_v * (tot_m - mass_v)).sum() / 2)
        if witness is None:
            i1 = s + int(np.flatnonzero(b != b[0])[0])
            w = sorted(set(idx_sorted[s].tolist()) ^ set(idx_sorted[i1].tolist()))
            witness = w if len(w) == weight else None
    if count == 0:
        return None
    return {"d_circ": weight, "count": count // splits, "mass": mass / splits, "witness": witness}


def _cross_hits(dA, bA, pA, idxA, dB, bB, pB, idxB, weight):
    nA = len(dA)
    lo = np.searchsorted(dA, dB, side="left")
    hi = np.searchsorted(dA, dB, side="right")
    has = hi > lo
    if not has.any():
        return None
    starts = np.flatnonzero(np.concatenate([[True], dA[1:] != dA[:-1]]))
    ends = np.concatenate([starts[1:], [nA]])
    cand = np.flatnonzero(has)
    count, mass, witness = 0, 0.0, None
    splits = {3: 3, 5: 10}[weight]
    gid = np.searchsorted(starts, lo[cand], side="right") - 1
    for g in np.unique(gid):
        s, e = starts[g], ends[g]
        b, p = bA[s:e], pA[s:e]
        members = cand[gid == g]
        vals, inv = np.unique(b, return_inverse=True)
        cnt_v = np.bincount(inv, minlength=len(vals)).astype(np.int64)
        mass_v = np.bincount(inv, weights=p, minlength=len(vals))
        tot_c, tot_m = cnt_v.sum(), mass_v.sum()
        bm = bB[members]
        pos = np.searchsorted(vals, bm)
        pos_ok = pos < len(vals)
        pos_c = np.where(pos_ok, pos, 0)
        match = pos_ok & (vals[pos_c] == bm)
        sub_c = np.where(match, cnt_v[pos_c], 0)
        sub_m = np.where(match, mass_v[pos_c], 0.0)
        hits_c = tot_c - sub_c
        count += int(hits_c.sum())
        mass += float(((tot_m - sub_m) * pB[members]).sum())
        if witness is None and (hits_c > 0).any():
            mb = members[int(np.flatnonzero(hits_c > 0)[0])]
            for i in range(s, e):
                if bA[i] != bB[mb]:
                    w = sorted(set(idxA[i].tolist()) ^ set(idxB[mb].tolist()))
                    if len(w) == weight:
                        witness = w
                    break
    if count == 0:
        return None
    return {"d_circ": weight, "count": count // splits, "mass": mass / splits, "witness": witness}


def exact_dcirc(sig: Signature, wmax: int = 4) -> dict:
    """Exact circuit-level distance of a DEM up to ``wmax``.

    Parameters
    ----------
    sig
        :class:`Signature` of the detector error model; detector signatures
        wider than 64 bits are mapped to integer ids first.
    wmax
        Largest fault-set weight searched, 4 or 5.

    Returns
    -------
    dict
        ``d_circ`` (the smallest number of mechanisms whose combined effect
        flips no detector and some observable, ``None`` when no such set of
        weight ``<= wmax`` exists), ``count`` of such minimum-weight sets,
        ``mass`` (their summed probability product), and ``witness`` (one set
        as mechanism indices).
    """
    m = len(sig)
    det1 = sig.det
    obs = sig.obs
    probs = sig.probs
    if m == 0:
        return {"d_circ": None, "count": 0, "mass": 0.0, "witness": None}
    if m >= 2:
        i, j = _pairs(m)
        det2w = det1[i] ^ det1[j]
        d1, d2 = _joint_ids(det1, det2w)
    else:
        (d1,) = _joint_ids(det1)
        d2 = None
    hit = np.flatnonzero((d1 == 0) & (obs != 0)) if sig.words == 1 else np.flatnonzero((~det1.any(axis=1)) & (obs != 0))
    if len(hit):
        return {"d_circ": 1, "count": len(hit), "mass": float(probs[hit].sum()), "witness": [int(hit[0])]}
    if m < 2:
        return {"d_circ": None, "count": 0, "mass": 0.0, "witness": None}
    o1 = np.argsort(d1, kind="stable")
    s1, b1, p1 = d1[o1], obs[o1], probs[o1]
    res = _group_hits(s1, b1, p1, o1.reshape(-1, 1), 2)
    if res is not None:
        return res
    b2 = obs[i] ^ obs[j]
    p2 = probs[i] * probs[j]
    idx2 = np.stack([i, j], 1)
    res = _cross_hits(s1, b1, p1, o1.reshape(-1, 1), d2, b2, p2, idx2, 3)
    if res is not None:
        return res
    o2 = np.argsort(d2, kind="stable")
    res = _group_hits(d2[o2], b2[o2], p2[o2], idx2[o2], 4)
    if res is not None:
        return res
    if wmax < 5 or m < 3:
        return {"d_circ": None, "count": 0, "mass": 0.0, "witness": None}
    cnt = m * (m - 1) * (m - 2) // 6
    flat = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(m), 3)), dtype=np.int64, count=cnt * 3)
    t = flat.reshape(cnt, 3)
    det3w = det1[t[:, 0]] ^ det1[t[:, 1]] ^ det1[t[:, 2]]
    d2b, d3 = _joint_ids(det2w, det3w)
    o2 = np.argsort(d2b, kind="stable")
    b3 = obs[t[:, 0]] ^ obs[t[:, 1]] ^ obs[t[:, 2]]
    p3 = probs[t[:, 0]] * probs[t[:, 1]] * probs[t[:, 2]]
    res = _cross_hits(d2b[o2], b2[o2], p2[o2], idx2[o2], d3, b3, p3, t, 5)
    if res is not None:
        return res
    return {"d_circ": None, "count": 0, "mass": 0.0, "witness": None}


def _top_sets(sets: list[tuple[np.ndarray, np.ndarray]], limit: int) -> list[tuple[tuple[int, ...], float]]:
    idx = np.concatenate([s for s, _ in sets], axis=0)
    prob = np.concatenate([p for _, p in sets])
    order = np.argsort(-prob, kind="stable")[:limit]
    return [(tuple(int(v) for v in idx[r]), float(prob[r])) for r in order]


def undetectable_sets(sig: Signature, wmax: int = 3, limit: int = 10, chunk: int = 1_000_000) -> dict:
    """The minimum-weight undetectable logical fault sets of a DEM up to
    weight ``wmax`` (at most 3), listed by probability: the smallest ``w``
    such that some ``w`` mechanisms together flip no detector and some
    observable, the number of such sets, and the ``limit`` most probable
    of them as ``(mechanism indices, probability product)`` pairs in
    decreasing probability. ``weight`` is ``None`` and the list empty when
    no set of weight ``<= wmax`` exists. The weight is the ``d_circ`` of
    :func:`exact_dcirc` whenever that is at most ``wmax``; the count agrees
    with it. Triples are found as a pair plus a single with the pair's
    detector signature, scanned ``chunk`` pairs at a time."""
    if wmax > 3:
        raise ValueError("undetectable_sets enumerates up to weight 3")
    m = len(sig)
    none = {"weight": None, "count": 0, "sets": []}
    if m == 0:
        return none
    det, obs, probs = sig.det, sig.obs, sig.probs
    zero = (det == 0).all(axis=1)
    hit = np.flatnonzero(zero & (obs != 0))
    if len(hit):
        sets = [(hit.reshape(-1, 1), probs[hit])]
        return {"weight": 1, "count": len(hit), "sets": _top_sets(sets, limit)}
    if wmax < 2 or m < 2:
        return none
    (d1,) = _joint_ids(det)
    o1 = np.argsort(d1, kind="stable")
    s1 = d1[o1]
    # weight 2: two mechanisms with the same detector signature and
    # different observable flips
    starts = np.flatnonzero(np.concatenate([[True], s1[1:] != s1[:-1]]))
    ends = np.concatenate([starts[1:], [m]])
    found, count = [], 0
    for s, e in zip(starts[ends - starts > 1], ends[ends - starts > 1]):
        members = o1[s:e]
        a, b = np.triu_indices(len(members), 1)
        i, j = members[a], members[b]
        keep = obs[i] != obs[j]
        if keep.any():
            i, j = i[keep], j[keep]
            count += len(i)
            found.append((np.stack([np.minimum(i, j), np.maximum(i, j)], 1), probs[i] * probs[j]))
    if count:
        return {"weight": 2, "count": count, "sets": _top_sets(found, limit)}
    if wmax < 3 or m < 3:
        return none
    # weight 3: a pair (i < j) plus a single k > j whose detector signature
    # equals the pair's; each triple is counted once, from its two
    # smallest members
    i_all, j_all = _pairs(m)
    d1_ids, d2_ids = _joint_ids(det, det[i_all] ^ det[j_all])
    o1 = np.argsort(d1_ids, kind="stable")
    s1 = d1_ids[o1]
    best_idx, best_prob = np.zeros((0, 3), dtype=np.int64), np.zeros(0)
    for start in range(0, len(i_all), chunk):
        i, j, d2 = i_all[start : start + chunk], j_all[start : start + chunk], d2_ids[start : start + chunk]
        lo = np.searchsorted(s1, d2, side="left")
        hi = np.searchsorted(s1, d2, side="right")
        n_cand = hi - lo
        has = np.flatnonzero(n_cand > 0)
        if not len(has):
            continue
        reps = n_cand[has]
        pair = np.repeat(has, reps)
        offset = np.arange(int(reps.sum())) - np.repeat(np.cumsum(reps) - reps, reps)
        k = o1[lo[pair] + offset]
        ii, jj = i[pair], j[pair]
        keep = (k > jj) & ((obs[ii] ^ obs[jj] ^ obs[k]) != 0)
        if not keep.any():
            continue
        ii, jj, k = ii[keep], jj[keep], k[keep]
        count += len(k)
        best_idx = np.concatenate([best_idx, np.stack([ii, jj, k], 1)], axis=0)
        best_prob = np.concatenate([best_prob, probs[ii] * probs[jj] * probs[k]])
        order = np.argsort(-best_prob, kind="stable")[:limit]
        best_idx, best_prob = best_idx[order], best_prob[order]
    if count:
        return {"weight": 3, "count": count, "sets": _top_sets([(best_idx, best_prob)], limit)}
    return none


# --------------------------------------------------------------------------
# decoders
# --------------------------------------------------------------------------
def exact_ml_failure(sig: Signature, max_bits: int = 20) -> tuple[float, np.ndarray]:
    """Exact failure probability of the optimal (degenerate maximum
    likelihood) decoder for a DEM with ``D + O <= max_bits``, by the
    dynamic program ``P <- (1 - p_i) P + p_i P[index xor v_i]`` over the
    joint distribution of (syndrome, observable flips) (the construction of
    ``ion-search/decoder/exact_decoder.py``). Returns ``(p_fail, table)``
    with ``table`` of shape ``(2^O, 2^D)``. The decision for syndrome ``s``
    is ``argmax_o table[o, s]``; ``p_fail = 1 - sum_s max_o table[o, s]``."""
    d, o = sig.num_detectors, sig.num_observables
    if d + o > max_bits:
        raise ValueError(f"exact ML needs D + O <= {max_bits}, got {d + o}")
    if sig.words != 1:
        raise ValueError("exact ML needs at most 64 detectors")
    size = 1 << (d + o)
    table = np.zeros(size)
    table[0] = 1.0
    idx = np.arange(size, dtype=np.int64)
    for det, obs, p in zip(sig.det[:, 0].astype(np.int64), sig.obs.astype(np.int64), sig.probs):
        v = det | (obs << d)
        if v == 0:
            continue
        table = (1.0 - p) * table + p * table[idx ^ v]
    tab2 = table.reshape(1 << o, 1 << d)
    return float(1.0 - tab2.max(axis=0).sum()), tab2


def _pack_words(bits: np.ndarray, words: int) -> np.ndarray:
    """``(N, D)`` bool to ``(N, words)`` uint64, bit ``i`` of detector ``i``
    in word ``i // 64``."""
    n, d = bits.shape
    pad = np.zeros((n, 64 * words), dtype=np.uint8)
    pad[:, :d] = bits
    return np.packbits(pad, axis=1, bitorder="little").view(np.uint64).reshape(n, words)


class LookupDecoder:
    """Table decoder from the fault sets of weight at most two: syndrome to
    the most likely observable pattern (summed weight per pattern). An
    unseen syndrome is completed with one more mechanism (the best
    ``table[s xor det_i]`` times ``q_i``); if none fits, no flip."""

    def __init__(self, sig: Signature):
        self.sig = sig
        m = len(sig)
        p = np.clip(sig.probs, 0.0, 1.0 - 1e-15)
        self.q = p / (1.0 - p)
        words = [sig.det]
        obs = [sig.obs]
        wgt = [self.q]
        if m >= 2:
            i, j = _pairs(m)
            words.append(sig.det[i] ^ sig.det[j])
            obs.append(sig.obs[i] ^ sig.obs[j])
            wgt.append(self.q[i] * self.q[j])
        allw = np.concatenate(words)
        allo = np.concatenate(obs)
        allg = np.concatenate(wgt)
        keys = [w.tobytes() for w in allw]
        agg: dict[bytes, dict[int, float]] = {}
        for key, o, g in zip(keys, allo.tolist(), allg.tolist()):
            agg.setdefault(key, {})
            agg[key][o] = agg[key].get(o, 0.0) + g
        self.table: dict[bytes, tuple[int, float]] = {}
        for key, per in agg.items():
            best = max(per.items(), key=lambda kv: kv[1])
            self.table[key] = (best[0], best[1])
        zero = np.zeros(sig.words, dtype=np.uint64).tobytes()
        self.table[zero] = (0, float("inf"))
        self.det_words = sig.det
        self.fallbacks = 0

    def decode_words(self, words: np.ndarray) -> int:
        key = words.tobytes()
        hit = self.table.get(key)
        if hit is not None:
            return hit[0]
        self.fallbacks += 1
        cand = self.det_words ^ words[None, :]
        best_o, best_w = 0, -1.0
        for i in range(cand.shape[0]):
            h = self.table.get(cand[i].tobytes())
            if h is not None:
                w = h[1] * self.q[i] if math.isfinite(h[1]) else self.q[i]
                if w > best_w:
                    best_w, best_o = w, h[0] ^ int(self.sig.obs[i])
        return best_o


def _bposd_decoder(sig: Signature):
    try:
        from ldpc import BpOsdDecoder
        from scipy.sparse import csr_matrix
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the bposd tier needs the `ldpc` package (uv run --with ldpc ...)") from exc
    h, _ = sig.matrices()
    # OSD order clamped to the redundancy of H (ldpc 2.4.1 does not check it)
    a = h.copy()
    rank, rows = 0, a.shape[0]
    for col in range(a.shape[1]):
        piv = next((r for r in range(rank, rows) if a[r, col]), None)
        if piv is None:
            continue
        a[[rank, piv]] = a[[piv, rank]]
        hit = np.nonzero(a[:, col])[0]
        a[hit[hit != rank]] ^= a[rank]
        rank += 1
    order = max(0, min(10, a.shape[1] - rank))
    return BpOsdDecoder(
        csr_matrix(h), error_channel=list(map(float, sig.probs)), max_iter=30, bp_method="minimum_sum",
        ms_scaling_factor=0.625, osd_method="osd_cs", osd_order=order,
    )


def _tesseract_decoder(dem: stim.DetectorErrorModel):
    try:
        from tesseract_decoder import tesseract as T
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the tesseract tier needs the `tesseract-decoder` package "
            "(uv run --with tesseract-decoder ..., or the `tesseract` extra)"
        ) from exc
    return T.TesseractDecoder(T.TesseractConfig(dem=dem))


def _decode_batch_cached(dec, dets: np.ndarray, words: np.ndarray, cache: dict, num_observables: int) -> np.ndarray:
    """Observable masks predicted by a batch decoder (``decode_batch`` on
    ``(shots, D)`` booleans to ``(shots, O)`` booleans), each distinct
    syndrome decoded once through ``cache``."""
    keys = [words[s].tobytes() for s in range(dets.shape[0])]
    todo: dict[bytes, int] = {}
    for s, key in enumerate(keys):
        if key not in cache:
            todo.setdefault(key, s)
    if todo:
        rows = np.fromiter(todo.values(), dtype=np.int64, count=len(todo))
        flips = np.asarray(dec.decode_batch(dets[rows].astype(bool))).reshape(len(rows), -1)
        shifts = np.arange(num_observables, dtype=np.int64)
        vals = (flips.astype(np.int64)[:, :num_observables] << shifts).sum(axis=1)
        for key, v in zip(todo, vals):
            cache[key] = int(v)
    return np.fromiter((cache[k] for k in keys), dtype=np.int64, count=len(keys))


def sample_failures(
    dem: stim.DetectorErrorModel,
    sig: Signature,
    decoder: str,
    shots: int,
    seed: int = 0,
    min_failures: int = 0,
    max_shots: int | None = None,
) -> dict:
    """Monte Carlo logical failures of ``dem`` under ``decoder``.

    Distinct syndromes are decoded once (every decoder here is a
    deterministic function of the syndrome). Shots grow in doublings until
    ``min_failures`` failures or ``max_shots``.

    Parameters
    ----------
    dem
        Detector error model sampled.
    sig
        Its :class:`Signature`.
    decoder
        ``"lookup"``, ``"bposd"``, or ``"tesseract"``.
    shots
        Initial number of shots.
    seed
        Sampler seed.
    min_failures
        Stop once this many failures are seen (``0``: a single pass).
    max_shots
        Cap on the doublings.

    Returns
    -------
    dict
        ``failures``, ``shots``, ``p_fail``, ``distinct_syndromes``,
        ``fallbacks`` (lookup only), ``seconds``.
    """
    t0 = time.perf_counter()
    sampler = dem.compile_sampler(seed=seed)
    lg = None
    if decoder == "lookup":
        dec = LookupDecoder(sig)
    elif decoder == "bposd":
        dec = _bposd_decoder(sig)
        _, lg = sig.matrices()
    elif decoder == "tesseract":
        dec = _tesseract_decoder(dem)
    else:
        raise ValueError("decoder must be lookup, bposd, or tesseract")
    max_shots = shots if max_shots is None else max_shots
    failures = 0
    done = 0
    cache: dict[bytes, int] = {}
    batch = shots
    while True:
        dets, obs, _ = sampler.sample(shots=batch)
        dets = dets.astype(np.uint8)
        words = _pack_words(dets, sig.words)
        obs_mask = np.zeros(batch, dtype=np.int64)
        for o in range(sig.num_observables):
            obs_mask |= obs[:, o].astype(np.int64) << o
        if decoder == "tesseract":
            pred = _decode_batch_cached(dec, dets, words, cache, sig.num_observables)
        else:
            pred = np.empty(batch, dtype=np.int64)
            for s in range(batch):
                key = words[s].tobytes()
                v = cache.get(key)
                if v is None:
                    if decoder == "lookup":
                        v = dec.decode_words(words[s])
                    else:
                        e = dec.decode(dets[s])
                        flips = (lg.astype(np.int64) @ e.astype(np.int64)) % 2
                        v = int(sum(int(f) << o for o, f in enumerate(flips)))
                    cache[key] = v
                pred[s] = v
        failures += int((pred != obs_mask).sum())
        done += batch
        if failures >= min_failures or done >= max_shots:
            break
        batch = min(batch, max_shots - done)
        if batch <= 0:
            break
    return {
        "failures": failures,
        "shots": done,
        "p_fail": failures / done if done else float("nan"),
        "distinct_syndromes": len(cache),
        "fallbacks": getattr(dec, "fallbacks", None),
        "seconds": round(time.perf_counter() - t0, 3),
    }


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
def per_round(p_shot: float, rounds: int) -> float:
    """Per-round failure probability from the failure probability of an
    ``R``-round experiment: ``1 - (1 - p)^(1/R)``."""
    p = min(max(p_shot, 0.0), 1.0)
    if p >= 1.0:
        return 1.0
    return 1.0 - (1.0 - p) ** (1.0 / rounds)


def code_key(code: StabilizerCode) -> bytes:
    """Canonical key of the stabilizer group (reduced row echelon form)."""
    red, _ = rref(code.stabilizers)
    return red.tobytes()


@dataclass
class MemoryResult:
    """Evaluation of one code under one objective: per-basis dictionaries
    and the summary ``p_round`` (sum over bases of the per-round
    failure probability of the chosen tier)."""

    tier: str
    rounds: int
    bases: tuple
    per_basis: dict
    p_round: float
    seconds: float

    def to_dict(self):
        return {
            "tier": self.tier, "rounds": self.rounds, "bases": list(self.bases), "per_basis": self.per_basis,
            "p_round": self.p_round, "seconds": self.seconds,
        }


class MemoryObjective:
    """Configuration and evaluator of the memory objective.

    Parameters
    ----------
    profile
        ``autoqec_sim.noise.Profile`` (its graph is replaced by the layout's).
    scale
        Multiplies every rate of the profile.
    rounds
        Noisy syndrome-extraction rounds (``None``: the target distance
        given to :meth:`evaluate`, else fixed).
    tier
        ``"proxy"``, ``"sampled"``, ``"bposd"``, or ``"tesseract"``.
    connectivity
        ``"all_to_all"``, ``"square:RxC"``, or a :class:`Layout`.
    flags, order, seed
        Passed to :func:`build_memory_circuit` (``order`` defaults to the
        hook-aware order at target ``rounds``).
    bases
        Memory bases evaluated and summed (default both).
    proxy_max_weight
        2 or 3 for :func:`pfail_low_weight`.
    with_dcirc
        Also compute :func:`exact_dcirc` (up to weight 4) in the proxy tier.
    exact_ml_bits
        Largest ``D + O`` for the exact ML decoder of the sampled tier.
    shots, min_failures, max_shots
        Sampling budget of the sampled, bposd, and tesseract tiers.
    schedule_trials
        With ``order="random"``, the number of random gate orders tried
        per code (seeds ``seed .. seed + trials - 1``, the same order for
        every basis); the best summed ``p_round`` is kept.
    """

    def __init__(
        self,
        profile: Profile,
        scale: float = 1.0,
        rounds: int | None = None,
        tier: str = "proxy",
        connectivity="all_to_all",
        flags: str = "none",
        order: str = "hook",
        seed: int = 0,
        bases=BASES,
        proxy_max_weight: int = 2,
        with_dcirc: bool = False,
        exact_ml_bits: int = 20,
        shots: int = 2000,
        min_failures: int = 0,
        max_shots: int | None = None,
        schedule_trials: int = 1,
    ):
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        self.profile = profile
        self.scale = float(scale)
        self.rounds = rounds
        self.tier = tier
        self.connectivity = connectivity
        self.flags = flags
        self.order = order
        self.seed = int(seed)
        self.bases = tuple(bases)
        self.proxy_max_weight = int(proxy_max_weight)
        self.with_dcirc = bool(with_dcirc)
        self.exact_ml_bits = int(exact_ml_bits)
        self.shots = int(shots)
        self.min_failures = int(min_failures)
        self.max_shots = max_shots
        self.schedule_trials = int(schedule_trials)
        self.cache: dict[bytes, MemoryResult] = {}
        self.evaluations = 0
        self.cache_hits = 0

    def describe(self) -> dict:
        return {
            "profile": self.profile.name, "timing": self.profile.timing.name, "scale": self.scale, "rounds": self.rounds,
            "tier": self.tier, "connectivity": self.connectivity if isinstance(self.connectivity, str) else self.connectivity.name,
            "flags": self.flags, "order": self.order, "bases": list(self.bases), "proxy_max_weight": self.proxy_max_weight,
            "exact_ml_bits": self.exact_ml_bits, "shots": self.shots, "min_failures": self.min_failures, "max_shots": self.max_shots,
            "schedule_trials": self.schedule_trials,
        }

    # ------------------------------------------------------------------
    def circuit(self, code: StabilizerCode, basis: str, rounds: int, seed: int | None = None, orders=None) -> MemoryCircuit:
        seed = self.seed if seed is None else seed
        return build_memory_circuit(code, rounds, basis, self.connectivity, self.flags, self.order, seed, hook_target=rounds, orders=orders)

    def dem(self, code: StabilizerCode, basis: str, rounds: int, seed: int | None = None, orders=None):
        mc = self.circuit(code, basis, rounds, seed, orders)
        return mc, memory_dem(mc, self.profile, self.scale)

    def evaluate_basis(self, code: StabilizerCode, basis: str, rounds: int, tier: str | None = None, seed: int | None = None, orders=None) -> dict:
        mc, dem = self.dem(code, basis, rounds, seed, orders)
        return self.score_basis(mc, dem, tier)

    def score_basis(self, mc: MemoryCircuit, dem: stim.DetectorErrorModel, tier: str | None = None) -> dict:
        """The per-basis dictionary of ``tier`` for a memory circuit and
        its detector error model (``evaluate_basis`` without the build)."""
        tier = self.tier if tier is None else tier
        basis, rounds = mc.basis, mc.rounds
        t0 = time.perf_counter()
        sig = dem_signature(dem)
        out = {
            "basis": basis, "rounds": rounds, "mechanisms": len(sig), "detectors": sig.num_detectors,
            "observables": sig.num_observables, "two_qubit_gates": mc.two_qubit_gates, "routed_gates": mc.routed_gates,
            "layers": mc.layers, "batches": len(mc.batches), "num_qubits": mc.num_qubits,
            "expected_faults": float(sig.probs.sum()),
        }
        if tier == "proxy":
            out["p_fail"] = pfail_low_weight(sig, self.proxy_max_weight)
            out["proxy_weight"] = self.proxy_max_weight
            if self.with_dcirc:
                dc = exact_dcirc(sig, wmax=4)
                out["d_circ"] = dc["d_circ"]
                out["d_circ_count"] = dc["count"]
                out["d_circ_mass"] = dc["mass"]
        elif tier == "sampled":
            if sig.num_detectors + sig.num_observables <= self.exact_ml_bits and sig.words == 1:
                p, _ = exact_ml_failure(sig, self.exact_ml_bits)
                out["p_fail"] = p
                out["decoder"] = "exact_ml"
            else:
                r = sample_failures(dem, sig, "lookup", self.shots, self.seed, self.min_failures, self.max_shots)
                out["p_fail"] = r["p_fail"]
                out["decoder"] = "lookup"
                out["sampling"] = r
        elif tier in ("bposd", "tesseract"):
            r = sample_failures(dem, sig, tier, self.shots, self.seed, self.min_failures, self.max_shots)
            out["p_fail"] = r["p_fail"]
            out["decoder"] = tier
            out["sampling"] = r
        else:
            raise ValueError(tier)
        out["p_round"] = per_round(out["p_fail"], rounds)
        out["seconds"] = round(time.perf_counter() - t0, 4)
        return out

    def evaluate(self, code: StabilizerCode, target_distance: int | None = None, tier: str | None = None, use_cache: bool = True) -> MemoryResult:
        """Evaluate ``code``; ``rounds`` defaults to ``target_distance``
        (else the code's exact distance when ``rounds`` is ``None``)."""
        tier = self.tier if tier is None else tier
        rounds = self.rounds
        if rounds is None:
            rounds = target_distance if target_distance is not None else code.distance_exact()
        key = (code_key(code), tier, rounds)
        if use_cache and key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        t0 = time.perf_counter()
        trials = self.schedule_trials if self.order == "random" else 1
        per, p_round = None, None
        for s in range(self.seed, self.seed + trials):
            # the gate order is a property of the code, shared by the bases
            orders = hook_orders(code, rounds, s) if self.order == "hook" else None
            cand = {b: self.evaluate_basis(code, b, rounds, tier, s, orders) for b in self.bases}
            p = float(sum(cand[b]["p_round"] for b in self.bases))
            if p_round is None or p < p_round:
                per, p_round = cand, p
                for b in self.bases:
                    per[b]["order_seed"] = s
        res = MemoryResult(tier, rounds, self.bases, per, p_round, round(time.perf_counter() - t0, 4))
        self.evaluations += 1
        if use_cache:
            self.cache[key] = res
        return res


# --------------------------------------------------------------------------
# environment reward
# --------------------------------------------------------------------------
class MemoryReward:
    """Environment reward with the memory term (module docstring).

    Use as ``CodeDiscoveryEnv(reward_fn=r)``; the environment adopts
    ``r.success_fn`` unless a ``success_fn`` is given explicitly.

    Parameters
    ----------
    objective
        :class:`MemoryObjective` that scores a terminal code.
    target_distance
        Distance a state must reach before the memory term is added.
    weight
        Multiplier of the memory term ``log10(p_ref / p_round)``.
    p_ref
        Reference per-round failure probability (term zero at ``p_ref``).
    clip
        Bound on the log ratio before weighting.
    p_floor
        Lower bound on ``p_round`` inside the logarithm.
    """

    def __init__(
        self,
        objective: MemoryObjective,
        target_distance: int,
        weight: float = 0.1,
        p_ref: float = 1e-2,
        clip: float = 3.0,
        p_floor: float = 1e-9,
    ):
        self.objective = objective
        self.d = int(target_distance)
        self.weight = float(weight)
        self.p_ref = float(p_ref)
        self.clip = float(clip)
        self.p_floor = float(p_floor)
        self.last_success = None
        self.terminal_calls = 0

    def term(self, code: StabilizerCode) -> float:
        p = max(self.objective.evaluate(code, self.d).p_round, self.p_floor)
        v = math.log10(self.p_ref / p)
        return self.weight * max(-self.clip, min(self.clip, v))

    def __call__(self, tab: np.ndarray, k: int) -> np.ndarray:
        d = batched_distance(tab, k, self.d - 1)
        pad = low_weight_stabilizer_penalty(tab, k, 1)
        r = np.minimum(d, self.d) / float(self.d) - 1.0 + pad
        succ = (d >= self.d) & (pad == 0.0)
        for b in np.flatnonzero(succ):
            code = StabilizerCode.from_tableau(tab, k, int(b))
            r[b] += self.term(code)
            self.terminal_calls += 1
        self.last_success = succ
        return r

    def success_fn(self, tab, k, r):
        if self.last_success is not None and self.last_success.shape == r.shape:
            return self.last_success
        return r >= 0.0


def make_memory_reward(objective: MemoryObjective, target_distance: int, **kw) -> Callable:
    """:class:`MemoryReward` as a reward callable for the environment."""
    return MemoryReward(objective, target_distance, **kw)
