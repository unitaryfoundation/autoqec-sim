"""The board preset reproduces circuit_tools.apply_noise, and the sampled
logical error rate at p = 1e-3 agrees within statistics (independent
seeds, one lookup decoder for both)."""

import os

import pytest
import stim
from _decoder import LookupDecoder, two_proportion_z
from conftest import N_DATA_913, QLDPC_ROOT, needs_913, needs_board

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets


@needs_board
@needs_913
def test_board_byte_identical(skel_913, ct):
    for skel in skel_913.values():
        want = str(ct.apply_noise(skel, N_DATA_913, ct.P_REF))
        have = str(nm.apply(skel, presets.board(), 1.0, n_data=N_DATA_913))
        assert want == have
        # scale multiplies p
        assert str(ct.apply_noise(skel, N_DATA_913, 2e-3)) == \
            str(nm.apply(skel, presets.board(), 2.0, n_data=N_DATA_913))
        # explicit idle set gives the same thing
        assert have == str(nm.apply(skel, presets.board(), 1.0, idle_qubits=range(N_DATA_913)))


@needs_board
def test_board_byte_identical_on_committed_board_circuit(ct):
    path = os.path.join(QLDPC_ROOT, "circuits", "40-10-4", "memory_x.stim")
    if not os.path.exists(path):
        pytest.skip("board circuit not present")
    raw = stim.Circuit.from_file(path)
    skel = ct.strip_noise(raw)
    have = nm.apply(skel, presets.board(), 1.0, n_data=40)
    assert str(have) == str(ct.apply_noise(skel, 40, ct.P_REF))
    assert ct.noise_recipe_errors(have, 40) == []


def test_board_needs_n_data(skel_913):
    with pytest.raises(ValueError):
        nm.apply(skel_913["one_x"], presets.board(), 1.0)


@needs_board
@needs_913
def test_dem_equal_and_exact(skel_913, ct):
    for key in ("one_x", "one_z"):
        skel = skel_913[key]
        board = ct.apply_noise(skel, N_DATA_913)
        ours = nm.apply(skel, presets.board(), 1.0, n_data=N_DATA_913)
        assert ct.dem_matches(ct.derive_dem(ours), ct.derive_dem(board))
        assert presets.board().dem_exact
        assert nm.dem(ours, approximate=False) is not None


@needs_board
@needs_913
@pytest.mark.parametrize("key", ["one_x", "one_z"])
def test_ler_matches_board_within_statistics(skel_913, ct, key):
    """Sampled failure fractions of the board recipe and of the board
    profile at p = 1e-3, decoded with one lookup table trained on the board
    circuit, agree within 3 sigma (two-proportion z). Same channel,
    different sampler seeds, so this is a check of the sampling path, and
    of the decoder harness the bias tests then reuse."""
    skel = skel_913[key]
    board = ct.apply_noise(skel, N_DATA_913, ct.P_REF)
    ours = nm.apply(skel, presets.board(), 1.0, n_data=N_DATA_913)
    dec = LookupDecoder.train(board, 200_000, seed=11)
    shots = 400_000
    f_board = dec.failures(board, shots, seed=101)
    f_ours = dec.failures(ours, shots, seed=202)
    assert f_board > 20 and f_ours > 20, (f_board, f_ours)
    z = two_proportion_z(f_board, shots, f_ours, shots)
    assert abs(z) < 3.0, (key, f_board, f_ours, z)


@needs_913
def test_uniform_depolarizing_profile_is_board_plus_single_qubit_channel(skel_913):
    """presets.depolarizing(p) differs from the board recipe only by the
    DEPOLARIZE1(p) after single-qubit unitaries and by idling every qubit."""
    skel = skel_913["one_x"]
    a = nm.apply(skel, presets.depolarizing(1e-3, idle_scope="data"), 1.0, n_data=N_DATA_913)
    b = nm.apply(skel, presets.board(), 1.0, n_data=N_DATA_913)
    extra = [i for i in a.flattened() if i.name == "DEPOLARIZE1"]
    base = [i for i in b.flattened() if i.name == "DEPOLARIZE1"]
    n_1q = sum(len(i.targets_copy()) for i in skel.flattened()
               if i.name not in nm.skeleton.ANNOTATIONS and i.name not in ("R", "RX", "M", "MX")
               and not stim.gate_data(i.name).is_two_qubit_gate)
    assert n_1q > 0
    assert len(extra) > len(base)
    assert str(nm.strip_noise(a)) == str(skel)
