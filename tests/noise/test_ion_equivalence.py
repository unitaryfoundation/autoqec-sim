"""The ion presets through `apply` are byte-identical to
ion_noise.apply_ion_noise: both parameter sets, both timings, scales 1 and
0.25, on every [[9,1,3]] ion-experiment circuit (one-round, MCM,
multi-round; X and Z)."""

import pytest
from conftest import needs_913, needs_board

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets

SETS = (("forte-2308.05071", presets.forte), ("chain-2606.06455", presets.chain))
TIMINGS = ("parallel", "sequential")
SCALES = (1.0, 0.25)


@needs_board
@needs_913
@pytest.mark.parametrize("name,maker", SETS, ids=[s[0] for s in SETS])
@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("scale", SCALES)
def test_byte_identical_to_ion_noise(skel_913, ion_noise, name, maker, timing, scale):
    prof = maker(timing=timing)
    assert prof.name == name
    for cname, skel in skel_913.items():
        want = str(ion_noise.apply_ion_noise(skel, scale, params=name, timing=timing))
        have = str(nm.apply(skel, prof, scale))
        assert want == have, (cname, name, timing, scale)


@needs_board
@needs_913
def test_reslot_1_is_serialize_two_qubit(skel_913, ion_noise):
    for skel in skel_913.values():
        assert str(nm.reslot(skel, 1)) == str(ion_noise.serialize_two_qubit(skel))
        assert str(nm.reslot(skel, None)) == str(skel)


@needs_board
@needs_913
def test_ion_params_match_presets(ion_noise):
    for name, maker in SETS:
        p = ion_noise.ion_params(1.0, name)
        q = nm.scaled(maker().params, 1.0)
        for k in ("p2", "p1", "p_reset", "p_meas", "t_2q", "t_1q", "t_meas", "t_reset"):
            assert p[k] == q[k], (name, k)
        if p["idle"] == "duration":
            assert p["z_rate"] == q["z_rate"] and p["xy_rate"] == q["xy_rate"]
        else:
            assert p["idle_gate"] == q["idle_gate"] and p["idle_meas"] == q["idle_meas"]
        assert p["leak_mcm"] == q["leak_meas"]


@needs_913
def test_strip_round_trip_and_records(skel_913):
    for skel in skel_913.values():
        for prof in (presets.forte(), presets.chain(), presets.chain("parallel")):
            noisy = nm.apply(skel, prof, 1.0)
            if prof.timing.gates_per_slot is None:
                assert str(nm.strip_noise(noisy)) == str(skel)
            else:
                assert str(nm.strip_noise(noisy)) == str(nm.reslot(skel, 1))
            assert noisy.num_detectors == skel.num_detectors
            assert noisy.num_observables == skel.num_observables
            assert noisy.num_measurements == skel.num_measurements
            # fixed point on its own skeleton
            assert str(nm.apply(nm.strip_noise(noisy), prof.with_timing("parallel"), 1.0)) == str(noisy)


@needs_board
@needs_913
def test_skeleton_helpers_match_board(skel_913, ct):
    for skel in skel_913.values():
        noisy = ct.apply_noise(skel, 9)
        assert str(nm.strip_noise(noisy)) == str(ct.strip_noise(noisy))
        a = [[str(i) for i in s] for s in nm.skeleton.segments(noisy.flattened())]
        b = [[str(i) for i in s] for s in ct._segments(noisy.flattened())]
        assert a == b


def test_linear_in_scale(skel_913):
    skel = skel_913["one_x"]
    for prof in (presets.forte(), presets.chain(), presets.heron(graph=None)):
        full = nm.apply(skel, prof, 1.0)
        half = nm.apply(skel, prof, 0.5)
        for a, b in zip(full.flattened(), half.flattened()):
            assert a.name == b.name
            if a.name in nm.skeleton.ANNOTATIONS:
                continue
            for x, y in zip(a.gate_args_copy(), b.gate_args_copy()):
                assert abs(x - 2 * y) <= 1e-15 * max(1.0, x)
        zero = nm.apply(skel, prof, 0.0)
        assert nm.mechanism_count(nm.dem(zero)) == 0
