# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Connectivity graphs for the two-qubit action space.

The graph definitions live in ``autoqec_sim.noise.topology``, so the environment's action space and the noise
profile's placement check are built from one edge set. :class:`Graph` is
the environment-side view: ``n`` qubits numbered ``0..n-1`` and a sorted
tuple of undirected edges ``(a, b)`` with ``a < b``. :func:`from_topology`
and :meth:`Graph.to_topology` convert in both directions and round-trip
exactly; every builder here delegates to the ``autoqec_sim.noise`` builder of the
same name (``all_to_all`` to ``complete``, ``linear`` to ``path``).
"""

from __future__ import annotations

from dataclasses import dataclass

from autoqec_sim.noise import topology as topo


@dataclass(frozen=True)
class Graph:
    n: int
    edges: tuple
    name: str = ""

    def directed_edges(self):
        out = []
        for a, b in self.edges:
            out.append((a, b))
            out.append((b, a))
        return out

    def degree(self):
        d = [0] * self.n
        for a, b in self.edges:
            d[a] += 1
            d[b] += 1
        return d

    def has_edge(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.edges

    def is_connected(self) -> bool:
        if self.n == 0:
            return True
        adj = {i: set() for i in range(self.n)}
        for a, b in self.edges:
            adj[a].add(b)
            adj[b].add(a)
        seen = {0}
        stack = [0]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == self.n

    def to_topology(self) -> topo.Graph:
        """The same graph as a ``autoqec_sim.noise.topology.Graph`` (nodes
        ``0..n-1``, same edges, same name), for ``Profile.with_graph``."""
        return topo.Graph(self.name, tuple(range(self.n)), frozenset(frozenset(e) for e in self.edges))

    def same_edges(self, other) -> bool:
        """True if ``other`` (either graph class) has the same node count and
        edge set."""
        other = as_graph(other)
        return self.n == other.n and self.edges == other.edges


def _mk(n, edges, name):
    edges = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})
    return Graph(n, tuple(edges), name)


def from_topology(g: topo.Graph) -> Graph:
    """Environment graph from a ``autoqec_sim.noise.topology.Graph``. The nodes
    must be ``0..n-1`` (every ``autoqec_sim.noise`` builder numbers them so)."""
    if tuple(g.nodes) != tuple(range(g.n)):
        raise ValueError("topology nodes must be 0..n-1 to serve as qubit indices")
    return _mk(g.n, g.edge_list(), g.name)


def as_graph(g) -> Graph:
    """Accept either graph class."""
    if isinstance(g, Graph):
        return g
    if isinstance(g, topo.Graph):
        return from_topology(g)
    raise TypeError(f"not a graph: {type(g).__name__}")


def all_to_all(n: int) -> Graph:
    return from_topology(topo.complete(n))


def linear(n: int) -> Graph:
    return from_topology(topo.path(n))


def square(rows: int, cols: int) -> Graph:
    """Nearest-neighbour square lattice, qubits indexed row-major."""
    return from_topology(topo.square(rows, cols))


def triangular(rows: int, cols: int) -> Graph:
    """Square lattice plus one diagonal per cell (degree up to 6)."""
    return from_topology(topo.triangular(rows, cols))


def heavy_hex(rows: int, cols: int) -> Graph:
    """Heavy-hex lattice of ``rows x cols`` hexagons with a qubit on every
    edge (degree at most 3), the ``autoqec_sim.noise`` construction."""
    return from_topology(topo.heavy_hex(rows, cols))


BUILDERS = {
    "all_to_all": all_to_all,
    "linear": linear,
    "square": square,
    "triangular": triangular,
    "heavy_hex": heavy_hex,
}


def from_spec(spec: str) -> Graph:
    """``"all_to_all:7"``, ``"linear:5"``, ``"square:3x3"``,
    ``"triangular:2x3"``, ``"heavy_hex:1x1"``."""
    kind, _, arg = spec.partition(":")
    if kind not in BUILDERS:
        raise ValueError(f"unknown graph kind {kind!r}")
    if "x" in arg:
        r, c = (int(v) for v in arg.split("x"))
        return BUILDERS[kind](r, c)
    return BUILDERS[kind](int(arg))
