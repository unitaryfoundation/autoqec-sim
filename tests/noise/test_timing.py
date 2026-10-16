"""Timing models: parallel, sequential, gates_per_slot=k."""

import pytest
from conftest import needs_913

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets
from autoqec_sim.noise.skeleton import segments, two_qubit_pairs


def test_timing_names():
    assert nm.Timing.from_name("parallel") is nm.PARALLEL
    assert nm.Timing.from_name("sequential") is nm.SEQUENTIAL
    assert nm.Timing.from_name(3) == nm.partial(3)
    assert nm.partial(2).name == "gates_per_slot=2"
    with pytest.raises(ValueError):
        nm.Timing(0)
    with pytest.raises(ValueError):
        nm.Timing.from_name("fast")


@needs_913
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_reslot_properties(skel_913, ct, k):
    skel = skel_913["one_x"]
    r = nm.reslot(skel, k)
    segs = segments(r.flattened())
    counts = [len(two_qubit_pairs(s)) for s in segs]
    assert max(counts) <= k
    assert sum(counts) == sum(len(two_qubit_pairs(s)) for s in segments(skel.flattened()))
    assert ct.layer_conflict_errors(r) == []
    assert r.num_measurements == skel.num_measurements
    assert r.num_detectors == skel.num_detectors
    # non-two-qubit instructions never move out of their original layer order
    a = [i for i in skel.flattened() if i.name not in ("TICK", "CX")]
    b = [i for i in r.flattened() if i.name not in ("TICK", "CX")]
    assert [str(x) for x in a] == [str(x) for x in b]


@needs_913
def test_idle_budget_monotone_in_parallelism(skel_913):
    """Fewer gates per slot means more layers and more idle exposure."""
    skel = skel_913["one_x"]
    prof = presets.forte()

    def idle_mass(k):
        noisy = nm.apply(skel, prof.replace(timing=nm.Timing(k)), 1.0)
        return sum(sum(i.gate_args_copy()) * len(i.targets_copy())
                   for i in noisy.flattened() if i.name == "PAULI_CHANNEL_1")
    masses = [idle_mass(k) for k in (None, 4, 3, 2, 1)]
    assert masses == sorted(masses)
    assert masses[0] < masses[-1]
    # k >= the widest layer is the parallel skeleton
    assert str(nm.apply(skel, prof.with_timing(4), 1.0)) == str(nm.apply(skel, prof, 1.0))
