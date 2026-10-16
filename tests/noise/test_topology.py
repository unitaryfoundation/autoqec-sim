"""Graph builders and the gate-placement check."""

import pytest
import stim
from conftest import needs_913

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets


def test_square():
    g = nm.square(3, 4)
    assert g.n == 12 and len(g.edges) == 3 * 3 + 2 * 4
    assert g.max_degree() == 4
    assert g.has_edge(0, 1) and g.has_edge(0, 4) and not g.has_edge(0, 5)


def test_triangular():
    g = nm.triangular(3, 3)
    assert g.n == 9 and len(g.edges) == 12 + 4
    assert g.has_edge(0, 4) and not g.has_edge(1, 3)
    assert g.max_degree() == 6 or g.degree(4) == 6


def test_heavy_hex_single_cell_is_a_12_cycle():
    g = nm.heavy_hex(1, 1)
    assert g.n == 12 and len(g.edges) == 12
    assert all(g.degree(q) == 2 for q in g.nodes)


def test_heavy_hex_degrees():
    g = nm.heavy_hex(4, 4)
    assert g.max_degree() == 3
    degs = [g.degree(q) for q in g.nodes]
    # interior vertex qubits have degree 3, edge qubits degree 2
    assert degs.count(3) > 0 and degs.count(2) > degs.count(3)
    # bipartite-like: no two degree-3 qubits adjacent
    for e in g.edges:
        a, b = tuple(e)
        assert not (g.degree(a) == 3 and g.degree(b) == 3)


def test_complete_and_path():
    g = nm.complete(6)
    assert len(g.edges) == 15 and g.max_degree() == 5
    assert nm.ion_chain is nm.complete
    p = nm.path(5)
    assert len(p.edges) == 4 and p.neighbors(2) == [1, 3]


@needs_913
def test_check_connectivity(skel_913):
    skel = skel_913["one_x"]
    assert nm.check_connectivity(skel, nm.complete(17)) == []
    bad = nm.check_connectivity(skel, nm.complete(12))
    assert bad and all("not in graph" in v.reason for v in bad)
    bad = nm.check_connectivity(skel, nm.path(17))
    assert bad and all("not an edge" in v.reason for v in bad)
    assert all(isinstance(v.layer, int) and v.gate == "CX" for v in bad)
    # a layout mapping can repair it
    sq = nm.square(5, 5)
    # map every CX pair onto some edge: an all-to-all layout cannot exist on a
    # square lattice for this circuit, so at least one violation remains
    assert nm.check_connectivity(skel, sq, layout={q: q for q in range(17)})


@needs_913
def test_apply_reports_violations(skel_913):
    skel = skel_913["one_x"]
    prof = presets.heron(graph=nm.heavy_hex(2, 2))
    with pytest.raises(nm.ConnectivityError) as ei:
        nm.apply(skel, prof, 1.0)
    assert len(ei.value.violations) == 24
    assert "violate the connectivity graph" in str(ei.value)
    # and can be bypassed explicitly
    nm.apply(skel, prof, 1.0, check_graph=False)
    # the ion chain accepts everything
    nm.apply(skel, presets.forte(graph=presets.forte_graph(36)), 1.0)


def test_layout_respected():
    c = stim.Circuit("R 0 1\nTICK\nCX 0 1\nTICK\nM 0 1")
    g = nm.path(3)
    assert nm.check_connectivity(c, g) == []
    assert nm.check_connectivity(c, g, layout={0: 0, 1: 2})
    assert nm.check_connectivity(c, g, layout={0: 1, 1: 2}) == []


def test_presets_registry():
    for name in presets.PRESETS:
        prof = presets.preset(name)
        assert prof.name == name
        assert nm.summarize(prof)
    with pytest.raises(ValueError):
        presets.preset("nope")
    h = presets.heron()
    assert h.graph is not None and h.graph.max_degree() == 3
    assert h.params.idle.kind == "duration"
