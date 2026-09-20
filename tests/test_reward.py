import numpy as np
import pytest

from autoqec_sim import known_codes
from autoqec_sim.errors import ErrorSet, count_paulis, paulis_up_to_weight
from autoqec_sim.reward import batched_distance, detected_fraction, distance_reward, kl_reward, undetected_mask
from autoqec_sim.symplectic import pauli_to_vec, pauli_weight
from autoqec_sim.tableau import identity_tableau


def test_error_enumeration_counts_and_weights():
    for n, w in [(4, 2), (7, 3), (9, 2)]:
        e, wt = paulis_up_to_weight(n, w)
        assert e.shape == (count_paulis(n, w), 2 * n)
        assert np.array_equal(pauli_weight(e), wt)
        assert len({row.tobytes() for row in e}) == e.shape[0]


def test_channel_weighted_error_set_orders_by_probability():
    es = ErrorSet.from_channel(5, px=0.01, py=0.01, pz=0.1, max_weight=2, top=20)
    assert len(es) == 20
    # the most probable errors are single Z's
    top = es.paulis[np.argsort(-es.lam)[:5]]
    assert all(pauli_weight(r) == 1 and r[:5].sum() == 0 for r in top)


@pytest.mark.parametrize("name", list(known_codes.KNOWN))
def test_kl_reward_is_zero_iff_distance_is_reached(name):
    code = known_codes.get(name)
    n, k, d = known_codes.parameters(name)
    tab = code.to_tableau()
    assert kl_reward(tab, k, ErrorSet.up_to_weight(n, d - 1))[0] == 0.0
    if d < n:
        assert kl_reward(tab, k, ErrorSet.up_to_weight(n, d))[0] < 0.0


def test_degenerate_errors_count_as_detected():
    # Shor code: ZZ on qubits 0,1 is a stabilizer, so it is not a logical
    # error even though it commutes with every generator.
    code = known_codes.get("shor")
    tab = code.to_tableau()
    e = np.array([pauli_to_vec("ZZIIIIIII"), pauli_to_vec("XIIIIIIII"), pauli_to_vec("ZIIZIIIII")])
    und = undetected_mask(tab, 1, e)[0]
    assert und.tolist() == [False, False, False]
    # weight-3 logical Z of Shor: Z on qubits 0,3,6 -> undetected
    e = np.array([pauli_to_vec("ZIIZIIZII")])
    assert undetected_mask(tab, 1, e)[0, 0]


def test_trivial_encoding_detects_nothing_on_logical_qubit():
    n, k = 5, 1
    tab = identity_tableau(n, 3)
    es = ErrorSet.up_to_weight(n, 1)
    frac = detected_fraction(tab, k, es)
    # errors on ancilla qubits anticommute with Z stabilizers or are Z (in S);
    # the 3 Paulis on the logical qubit are undetected
    assert np.allclose(frac, 1 - 3 / len(es))


def test_batched_distance_matches_exact():
    codes = [known_codes.get(nm) for nm in ("shor", "rotated_surface_9")]
    tab = np.concatenate([c.to_tableau() for c in codes] + [identity_tableau(9, 1)])
    d = batched_distance(tab, 1, max_weight=3)
    assert d.tolist() == [3, 3, 1]
    r = distance_reward(tab, 1, target_distance=3)
    assert np.allclose(r, [1.0, 1.0, 1 / 3])
