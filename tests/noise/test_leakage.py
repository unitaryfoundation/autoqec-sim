"""Leakage: the Pauli proxy fires only after measurement layers (and per
two-qubit gate when asked); the heralded form adds flag records without
breaking detector references."""

import pytest
from conftest import needs_913

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets
from autoqec_sim.noise.skeleton import layer_classes, segments


def _leak_segments(noisy, gate, rate):
    out = []
    for si, seg in enumerate(segments(noisy.flattened())):
        hits = [i for i in seg if i.name == gate and abs(i.gate_args_copy()[0] - rate) < 1e-15]
        out.append((si, hits, "meas" in layer_classes(seg)))
    return out


@needs_913
@pytest.mark.parametrize("key", ["mcm_x", "mcm_z", "multi_x", "multi_z", "one_x", "one_z"])
def test_pauli_proxy_only_after_measurement_layers(skel_913, key):
    skel = skel_913[key]
    prof = presets.chain(timing="parallel")
    rate = 0.75 * prof.params.leakage.per_meas_layer
    noisy = nm.apply(skel, prof, 1.0)
    n_leak_layers = 0
    for si, hits, is_meas in _leak_segments(noisy, "DEPOLARIZE1", rate):
        if hits:
            assert is_meas, f"leak proxy in non-measurement layer {si}"
            assert len(hits) == 1
            n_leak_layers += 1
            # after the idle channel, on the same qubits
            seg = segments(noisy.flattened())[si]
            idle = [i for i in seg if i.name == "PAULI_CHANNEL_1"]
            assert idle and idle[0].targets_copy() == hits[0].targets_copy()
            assert seg.index(idle[0]) < seg.index(hits[0])
    meas_layers_with_idle = 0
    all_q = set(nm.circuit_qubits(skel))
    for seg in segments(skel.flattened()):
        if "meas" in layer_classes(seg):
            touched = {t.qubit_value for i in seg if i.name not in nm.skeleton.ANNOTATIONS
                       for t in i.targets_copy()}
            if all_q - touched:
                meas_layers_with_idle += 1
    assert n_leak_layers == meas_layers_with_idle
    if key.startswith("one_"):
        assert n_leak_layers == 0      # the last layer measures every qubit


@needs_913
def test_per_2q_leakage(skel_913):
    skel = skel_913["one_x"]
    prof = presets.heron(graph=None).with_params(leakage=nm.Leakage(per_2q=1e-3, mode="pauli"))
    noisy = nm.apply(skel, prof, 1.0)
    for seg in segments(noisy.flattened()):
        pairs = nm.skeleton.two_qubit_pairs(seg)
        hits = [i for i in seg if i.name == "DEPOLARIZE1"
                and abs(i.gate_args_copy()[0] - 0.75e-3) < 1e-15]
        assert len(hits) == (1 if pairs else 0) * len([i for i in seg if i.name == "CX"])
        for h in hits:
            assert len(h.targets_copy()) % 2 == 0
    assert str(nm.strip_noise(noisy)) == str(skel)


@needs_913
@pytest.mark.parametrize("key", ["mcm_x", "multi_z"])
def test_heralded_leakage(skel_913, key):
    skel = skel_913[key]
    prof = presets.chain(timing="parallel").with_params(
        leakage=nm.Leakage(per_meas_layer=1.7e-3, mode="heralded"))
    noisy = nm.apply(skel, prof, 1.0)
    heralds = [i for i in noisy.flattened() if i.name == "HERALDED_ERASE"]
    assert heralds
    n_flags = sum(len(i.targets_copy()) for i in heralds)
    for si, hits, is_meas in _leak_segments(noisy, "HERALDED_ERASE", 1.7e-3):
        if hits:
            assert is_meas
    assert noisy.num_measurements == skel.num_measurements + n_flags
    assert noisy.num_detectors == skel.num_detectors + n_flags
    assert noisy.num_observables == skel.num_observables
    # the skeleton's detectors still reference the right records
    assert str(nm.strip_noise(noisy)) == str(skel)
    assert not prof.dem_exact
    d = nm.dem(noisy)
    assert d.num_detectors == noisy.num_detectors
    # herald mechanisms fire a herald detector
    herald_dets = set(range(skel.num_detectors, noisy.num_detectors))
    n_with_flag = 0
    for inst in d.flattened():
        if inst.type == "error":
            ds = {int(t.val) for t in inst.targets_copy() if t.is_relative_detector_id()}
            if ds & herald_dets:
                n_with_flag += 1
    assert n_with_flag > 0
    noisy.compile_detector_sampler(seed=1).sample(50)
    # noiseless-but-heralds circuit has deterministic detectors otherwise
    quiet = nm.apply(skel, prof, 0.0)
    assert "HERALDED_ERASE" not in {i.name for i in quiet.flattened()}
    assert str(quiet.without_noise()) == str(skel.without_noise()) or \
        str(nm.strip_noise(quiet)) == str(skel)


@needs_913
def test_heralded_without_detectors(skel_913):
    skel = skel_913["mcm_x"]
    prof = presets.chain(timing="parallel").with_params(
        leakage=nm.Leakage(per_meas_layer=1e-3, mode="heralded", herald_detectors=False))
    noisy = nm.apply(skel, prof, 1.0)
    assert noisy.num_detectors == skel.num_detectors
    assert noisy.num_measurements > skel.num_measurements
    assert str(nm.strip_noise(noisy)) == str(skel)


def test_bad_leakage_mode():
    with pytest.raises(ValueError):
        nm.Leakage(mode="erasure")
