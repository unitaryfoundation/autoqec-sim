"""Correlated gate-pair terms and nearest-neighbour crosstalk."""

import pytest
from conftest import needs_913

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets
from autoqec_sim.noise.channels import exclusive_chain
from autoqec_sim.noise.skeleton import segments, two_qubit_pairs


def test_exclusive_chain():
    q = exclusive_chain([0.1, 0.2, 0.3])
    assert q[0] == pytest.approx(0.1)
    assert q[1] == pytest.approx(0.2 / 0.9)
    assert q[2] == pytest.approx(0.3 / 0.7)
    with pytest.raises(ValueError):
        exclusive_chain([0.6, 0.5])


def _n_2q(skel):
    return sum(len(two_qubit_pairs(s)) for s in segments(skel.flattened()))


@needs_913
def test_gate_pair_terms(skel_913):
    skel = skel_913["one_x"]
    corr = nm.Correlated(gate_pairs=(("ZZ", 2e-4), ("XX", 1e-4)))
    prof = presets.forte().with_params(correlated=corr)
    noisy = nm.apply(skel, prof, 1.0)
    es = [i for i in noisy.flattened() if i.name == "E"]
    assert len(es) == 2 * _n_2q(skel)
    assert not any(i.name == "ELSE_CORRELATED_ERROR" for i in noisy.flattened())
    zz = [i for i in es if abs(i.gate_args_copy()[0] - 2e-4) < 1e-18]
    assert len(zz) == _n_2q(skel)
    t = zz[0].targets_copy()
    assert all(x.is_z_target for x in t) and len(t) == 2
    assert prof.dem_exact
    nm.dem(noisy, approximate=False)
    assert str(nm.strip_noise(noisy)) == str(skel)
    # scale multiplies the correlated rates too
    half = nm.apply(skel, prof, 0.5)
    assert [i.gate_args_copy()[0] for i in half.flattened() if i.name == "E"] == \
        pytest.approx([0.5 * i.gate_args_copy()[0] for i in es])


@needs_913
def test_exclusive_gate_pair_terms(skel_913):
    skel = skel_913["one_x"]
    corr = nm.Correlated(gate_pairs=(("ZZ", 2e-4), ("XX", 1e-4), ("YY", 1e-4)), exclusive=True)
    prof = presets.forte().with_params(correlated=corr)
    noisy = nm.apply(skel, prof, 1.0)
    es = [i for i in noisy.flattened() if i.name == "E"]
    elses = [i for i in noisy.flattened() if i.name == "ELSE_CORRELATED_ERROR"]
    assert len(es) == _n_2q(skel) and len(elses) == 2 * _n_2q(skel)
    assert elses[0].gate_args_copy()[0] == pytest.approx(1e-4 / (1 - 2e-4))
    assert not prof.dem_exact
    with pytest.raises(ValueError):
        nm.dem(noisy, approximate=False)
    nm.dem(noisy)


def _expected_crosstalk_terms(skel, graph):
    n = 0
    for seg in segments(skel.flattened()):
        for _, a, b in two_qubit_pairs(seg):
            for gq in (a, b):
                n += sum(1 for c in graph.neighbors(gq) if c not in (a, b))
    return n


@needs_913
def test_crosstalk_needs_graph(skel_913):
    prof = presets.forte().with_params(correlated=nm.Correlated(crosstalk=1e-4))
    with pytest.raises(ValueError):
        nm.apply(skel_913["one_x"], prof, 1.0)


@needs_913
@pytest.mark.parametrize("graph", [nm.complete(17), nm.path(17)])
def test_crosstalk_terms_and_mechanism_count(skel_913, graph):
    """One CORRELATED_ERROR per (gate qubit, spectator neighbour) pair, and
    the DEM grows by exactly the crosstalk signatures not already present
    (stim merges equal-signature mechanisms)."""
    skel = skel_913["one_x"]
    base = presets.forte().with_graph(graph)
    xt = base.with_params(correlated=nm.Correlated(crosstalk=1e-4, crosstalk_pauli="ZZ"))
    if nm.check_connectivity(skel, graph):
        base = base.replace(graph=None)
        xt_only_prof = xt
        noisy_xt = nm.apply(skel, xt, 1.0, check_graph=False)
    else:
        xt_only_prof = xt
        noisy_xt = nm.apply(skel, xt, 1.0)
    noisy_base = nm.apply(skel, base, 1.0, check_graph=False)
    es = [i for i in noisy_xt.flattened() if i.name == "E"]
    assert len(es) == _expected_crosstalk_terms(skel, graph)
    assert all(len(i.targets_copy()) == 2 and all(t.is_z_target for t in i.targets_copy())
               for i in es)

    # crosstalk-only circuit: every other rate zero
    zero = nm.Params(p2=0.0, p1=0.0, p_reset=0.0, p_meas=0.0,
                     correlated=nm.Correlated(crosstalk=1e-4))
    only = nm.apply(skel, nm.Profile("xt-only", zero, graph=graph), 1.0, check_graph=False)

    def cols(c):
        out = set()
        for inst in nm.dem(c).flattened():
            if inst.type == "error":
                out.add(tuple(str(t) for t in inst.targets_copy()))
        return out
    c_base, c_xt, c_both = cols(noisy_base), cols(only), cols(noisy_xt)
    assert len(c_xt) > 0
    assert c_both == c_base | c_xt
    assert nm.mechanism_count(nm.dem(noisy_xt)) == len(c_base | c_xt)
    assert nm.mechanism_count(nm.dem(noisy_xt)) > nm.mechanism_count(nm.dem(noisy_base))
    assert str(nm.strip_noise(noisy_xt)) == str(skel)
    del xt_only_prof


def test_bad_correlated_specs():
    with pytest.raises(ValueError):
        nm.Correlated(gate_pairs=(("ZZZ", 1e-4),))
    with pytest.raises(ValueError):
        nm.Correlated(crosstalk_pauli="Z")
