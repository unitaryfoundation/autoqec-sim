"""Biased channels: construction, emission form, and the direction a bias
sweep moves the X- and Z-basis failure rates."""

import math

import pytest
from _decoder import LookupDecoder
from conftest import needs_913

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets
from autoqec_sim.noise.channels import PAULI2, biased_1q, biased_2q, marginal_bias_2q


@pytest.mark.parametrize("eta", [0.01, 0.5, 1.0, 10.0, 100.0])
def test_biased_1q(eta):
    px, py, pz = biased_1q(1e-3, eta)
    assert abs(px + py + pz - 1e-3) < 1e-18
    assert px == py
    assert abs(pz / (px + py) - eta) < 1e-12


def test_biased_1q_limits():
    assert biased_1q(1e-3, 0.5) == pytest.approx((1e-3 / 3,) * 3)
    assert biased_1q(1e-3, math.inf) == (0.0, 0.0, 1e-3)


@pytest.mark.parametrize("eta", [0.05, 0.5, 3.0, 50.0])
def test_biased_2q(eta):
    v = biased_2q(2e-3, eta)
    assert len(v) == 15
    assert abs(sum(v) - 2e-3) < 1e-17
    assert abs(marginal_bias_2q(v) - eta) < 1e-9
    # symmetric under swapping the two qubits
    d = dict(zip(PAULI2, v))
    for s in PAULI2:
        assert d[s] == pytest.approx(d[s[::-1]])


def test_biased_2q_depolarizing_is_uniform():
    v = biased_2q(1.5e-3, 0.5)
    assert all(abs(x - 1.5e-3 / 15) < 1e-18 for x in v)
    assert dict(zip(PAULI2, biased_2q(1e-3, math.inf)))["ZZ"] == 1e-3


def test_biased_2q_orders_zz_first_at_high_bias():
    d = dict(zip(PAULI2, biased_2q(1e-3, 20.0)))
    # product weights: ZZ (4 eta^2), then the six terms with one Z factor
    # (2 eta each, ZI = IZ = ZX = ZY = XZ = YZ), then the eight Z-free terms (1)
    assert d["ZZ"] > d["ZI"] == d["IZ"] == d["ZX"] == d["YZ"] > d["XX"] == d["XY"]


@needs_913
def test_emission_forms(skel_913):
    skel = skel_913["one_x"]
    base = presets.forte()
    dep = nm.apply(skel, base, 1.0)
    names = {i.name for i in dep.flattened()}
    assert "DEPOLARIZE2" in names and "PAULI_CHANNEL_2" not in names
    biased = nm.apply(skel, base.with_bias(10.0), 1.0)
    names = {i.name for i in biased.flattened()}
    assert "PAULI_CHANNEL_2" in names and "DEPOLARIZE2" not in names
    assert "DEPOLARIZE1" not in names          # single-qubit gates are biased too
    pc1 = [i for i in biased.flattened() if i.name == "PAULI_CHANNEL_1"]
    assert pc1
    inf = nm.apply(skel, base.with_bias(math.inf, eta2=0.5), 1.0)
    names = {i.name for i in inf.flattened()}
    assert "Z_ERROR" in names and "DEPOLARIZE2" in names
    # skeleton unchanged either way
    assert str(nm.strip_noise(biased)) == str(skel)
    assert str(nm.strip_noise(inf)) == str(skel)
    # explicit 15-vector override
    custom = base.with_params(pauli2=tuple(biased_2q(4e-3, 3.0)))
    c = nm.apply(skel, custom, 0.5)
    pc2 = next(i for i in c.flattened() if i.name == "PAULI_CHANNEL_2")
    assert pc2.gate_args_copy() == pytest.approx([0.5 * x for x in biased_2q(4e-3, 3.0)])


@needs_913
def test_dem_conversion_exact_or_approximate(skel_913):
    skel = skel_913["one_x"]
    p = presets.forte()
    assert p.dem_exact
    nm.dem(nm.apply(skel, p, 1.0), approximate=False)
    b = p.with_bias(5.0)
    assert not b.dem_exact
    with pytest.raises(ValueError):
        nm.dem(nm.apply(skel, b, 1.0), approximate=False)
    d = nm.dem(nm.apply(skel, b, 1.0))     # falls back to approximate
    assert nm.mechanism_count(d) > 0
    # single-qubit bias alone stays exact (all three components positive)
    s = p.with_bias(5.0, eta2=0.5)
    assert s.dem_exact
    nm.dem(nm.apply(skel, s, 1.0), approximate=False)


def _observable_flip_mass(circuit):
    """Sum of DEM mechanism probabilities that flip the observable."""
    d = nm.dem(circuit)
    tot = 0.0
    for inst in d.flattened():
        if inst.type != "error":
            continue
        if any(t.is_logical_observable_id() for t in inst.targets_copy()):
            tot += inst.args_copy()[0]
    return tot


@needs_913
def test_bias_sweep_moves_bases_in_opposite_directions(skel_913):
    """Z-biased noise at fixed total rate loads the X-basis memory (Z
    errors flip logical X) and unloads the Z-basis one. Checked on the
    DEM (mass of observable-flipping mechanisms) and on sampled failures."""
    base = presets.depolarizing(5e-3)
    etas = (0.05, 0.5, 5.0, 50.0)
    mass = {}
    for key in ("one_x", "one_z"):
        skel = skel_913[key]
        mass[key] = [_observable_flip_mass(nm.apply(skel, base.with_bias(e), 1.0)) for e in etas]
    assert mass["one_x"] == sorted(mass["one_x"]), mass["one_x"]
    assert mass["one_z"] == sorted(mass["one_z"], reverse=True), mass["one_z"]

    shots = 200_000
    fails = {}
    for key in ("one_x", "one_z"):
        skel = skel_913[key]
        dep = nm.apply(skel, base, 1.0)
        dec = LookupDecoder.train(dep, 200_000, seed=3)
        fails[key] = {e: dec.failures(nm.apply(skel, base.with_bias(e), 1.0), shots, seed=7)
                      for e in (0.5, 50.0)}
    # measured 2026-09-20 (lookup decoder, 2e5 shots): X 5050 -> 7296,
    # Z 4993 -> 382; thresholds leave room for sampler changes
    assert fails["one_x"][50.0] > 1.2 * fails["one_x"][0.5], fails
    assert fails["one_z"][50.0] < 0.5 * fails["one_z"][0.5], fails
    # the biased idle is emitted as PAULI_CHANNEL_1, the board form stays
    # DEPOLARIZE1 when no bias is set
    names = {i.name for i in nm.apply(skel_913["one_x"], base.with_bias(50.0), 1.0).flattened()}
    assert "PAULI_CHANNEL_1" in names and "DEPOLARIZE1" not in names
    names = {i.name for i in nm.apply(skel_913["one_x"], base, 1.0).flattened()}
    assert "DEPOLARIZE1" in names and "PAULI_CHANNEL_1" not in names
