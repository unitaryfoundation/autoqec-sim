"""Connectivity graphs and the gate-placement check.

Qubit indices of a circuit are graph nodes unless a `layout` mapping
(qubit -> node) is given. Builders: `square`, `triangular`, `heavy_hex`,
`complete` (an all-to-all ion chain), `path`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import stim

from .skeleton import ANNOTATIONS, segments, two_qubit_pairs


@dataclass(frozen=True)
class Graph:
    name: str
    nodes: tuple[int, ...]
    edges: frozenset[frozenset[int]]
    positions: dict[int, tuple[float, float]] = field(default_factory=dict,
                                                      compare=False, hash=False)

    def __post_init__(self):
        node_set = set(self.nodes)
        for e in self.edges:
            if len(e) != 2 or not e <= node_set:
                raise ValueError(f"bad edge {set(e)} for graph {self.name}")

    @property
    def n(self) -> int:
        return len(self.nodes)

    def has_edge(self, a: int, b: int) -> bool:
        return frozenset((a, b)) in self.edges

    def neighbors(self, q: int) -> list[int]:
        return sorted(next(iter(e - {q})) for e in self.edges if q in e)

    def degree(self, q: int) -> int:
        return sum(1 for e in self.edges if q in e)

    def max_degree(self) -> int:
        return max((self.degree(q) for q in self.nodes), default=0)

    def edge_list(self) -> list[tuple[int, int]]:
        return sorted(tuple(sorted(e)) for e in self.edges)


def _graph(name, nodes, edges, positions=None):
    return Graph(name, tuple(sorted(nodes)),
                 frozenset(frozenset(e) for e in edges), positions or {})


def complete(n: int, name: str | None = None) -> Graph:
    """All-to-all coupling: a single trapped-ion chain of n ions."""
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return _graph(name or f"complete-{n}", range(n), edges,
                  {q: (float(q), 0.0) for q in range(n)})


ion_chain = complete


def path(n: int) -> Graph:
    """Nearest-neighbour line."""
    return _graph(f"path-{n}", range(n), [(q, q + 1) for q in range(n - 1)],
                  {q: (float(q), 0.0) for q in range(n)})


def square(rows: int, cols: int) -> Graph:
    """rows x cols square lattice, node (r, c) -> r * cols + c, degree 4."""
    def idx(r, c):
        return r * cols + c
    edges, pos = [], {}
    for r in range(rows):
        for c in range(cols):
            pos[idx(r, c)] = (float(c), float(r))
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return _graph(f"square-{rows}x{cols}", range(rows * cols), edges, pos)


def triangular(rows: int, cols: int) -> Graph:
    """Square lattice plus one diagonal per cell ((r, c) to (r + 1, c + 1)),
    degree 6 in the bulk."""
    g = square(rows, cols)

    def idx(r, c):
        return r * cols + c
    edges = set(g.edges)
    for r in range(rows - 1):
        for c in range(cols - 1):
            edges.add(frozenset((idx(r, c), idx(r + 1, c + 1))))
    return Graph(f"triangular-{rows}x{cols}", g.nodes, frozenset(edges),
                 dict(g.positions))


def heavy_hex(rows: int, cols: int) -> Graph:
    """Heavy-hexagon lattice of `rows` x `cols` hexagons (IBM Eagle and
    Heron style): a honeycomb with a qubit on every edge, maximum degree 3.

    Construction: rows + 1 horizontal lines of 4 cols + 1 qubits; vertical
    connector qubits join line i to line i + 1 at horizontal position x with
    x = 0 mod 4 on even i and x = 2 mod 4 on odd i. Every hexagon then has 6
    degree-3 vertices and 6 degree-2 edge qubits (12 qubits). Line qubits
    are numbered first, row by row, then connectors. IBM's devices trim the
    outermost lines differently (Eagle 127, Heron r2 156), so sizes match
    only approximately; the local structure is the same.
    """
    width = 4 * cols + 1
    lines = rows + 1
    nodes, edges, pos = [], [], {}

    def line(i, x):
        return i * width + x
    for i in range(lines):
        for x in range(width):
            q = line(i, x)
            nodes.append(q)
            pos[q] = (float(x), 2.0 * i)
            if x + 1 < width:
                edges.append((q, line(i, x + 1)))
    nxt = lines * width
    for i in range(rows):
        start = 0 if i % 2 == 0 else 2
        for x in range(start, width, 4):
            q = nxt
            nxt += 1
            nodes.append(q)
            pos[q] = (float(x), 2.0 * i + 1.0)
            edges.append((line(i, x), q))
            edges.append((q, line(i + 1, x)))
    return _graph(f"heavy-hex-{rows}x{cols}", nodes, edges, pos)


BUILDERS = {
    "square": square,
    "triangular": triangular,
    "heavy_hex": heavy_hex,
    "complete": complete,
    "ion_chain": complete,
    "path": path,
}


@dataclass(frozen=True)
class Violation:
    layer: int
    gate: str
    qubits: tuple[int, int]
    reason: str

    def __str__(self):
        return (f"layer {self.layer}: {self.gate} {self.qubits[0]} {self.qubits[1]}: "
                f"{self.reason}")


def check_connectivity(circuit: stim.Circuit, graph: Graph,
                       layout: dict[int, int] | None = None) -> list[Violation]:
    """Every two-qubit gate whose (mapped) qubit pair is not an edge of the
    graph, or touches a qubit not in the graph. [] iff the circuit respects
    the topology. Single-qubit operations are unconstrained."""
    node_set = set(graph.nodes)
    out = []
    for li, seg in enumerate(segments(circuit.flattened())):
        for name, a, b in two_qubit_pairs(seg):
            na = layout.get(a, a) if layout else a
            nb = layout.get(b, b) if layout else b
            missing = [q for q in (na, nb) if q not in node_set]
            if missing:
                out.append(Violation(li, name, (a, b),
                                     f"node(s) {missing} not in graph {graph.name}"))
            elif not graph.has_edge(na, nb):
                out.append(Violation(li, name, (a, b),
                                     f"({na}, {nb}) is not an edge of {graph.name}"))
    return out


def qubits_used(circuit: stim.Circuit) -> list[int]:
    qs = set()
    for inst in circuit.flattened():
        if inst.name in ANNOTATIONS:
            continue
        for t in inst.targets_copy():
            if t.qubit_value is not None:
                qs.add(t.qubit_value)
    return sorted(qs)
