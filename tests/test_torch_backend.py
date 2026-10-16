"""The torch backend agrees exactly with the numpy path on CPU. Skipped
without torch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autoqec_sim import graphs, known_codes
from autoqec_sim import reward as R
from autoqec_sim import symplectic as sp
from autoqec_sim import tableau as T
from autoqec_sim.backends import torch_backend as tb
from autoqec_sim.env import build_action_table
from autoqec_sim.errors import ErrorSet, paulis_up_to_weight

NS = (5, 9, 16)
BATCH = 48
K = 1


def _pair(n, batch, rng, steps=25):
    """Same random tableau in both backends, driven through every gate type."""
    table = build_action_table(graphs.all_to_all(n), T.GATES)
    tt = tb.ActionTable(table)
    tab = T.identity_tableau(n, batch)
    tabt = tb.identity_tableau(n, batch)
    for _ in range(steps):
        a = rng.integers(-1, len(table), size=batch)  # -1 is a no-op
        table.apply(tab, a)
        tt.apply(tabt, torch.from_numpy(a))
    return tab, tabt


@pytest.mark.parametrize("n", NS)
def test_action_table_apply_matches_numpy(n):
    rng = np.random.default_rng(n)
    tab, tabt = _pair(n, BATCH, rng)
    assert np.array_equal(tab, tb.to_numpy(tabt))
    assert sp.is_valid_tableau(tab[0])


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("gate", T.GATES)
def test_single_gate_kernels_match_numpy(n, gate):
    rng = np.random.default_rng(hash((n, gate)) % 2**32)
    tab, tabt = _pair(n, BATCH, rng, steps=10)
    b = np.flatnonzero(rng.random(BATCH) < 0.6)
    q1 = rng.integers(n, size=b.size)
    q2 = (q1 + rng.integers(1, n, size=b.size)) % n if gate in T.GATES_2Q else None
    T.apply_gate(tab, gate, q1, q2, where=b)
    tb.apply_gate(tabt, gate, torch.from_numpy(q1), None if q2 is None else torch.from_numpy(q2), where=torch.from_numpy(b))
    assert np.array_equal(tab, tb.to_numpy(tabt))
    # scalar qubit on a boolean mask, the other calling convention
    mask = rng.random(BATCH) < 0.5
    T.apply_gate(tab, gate, 0, 1 if q2 is not None else None, where=mask)
    tb.apply_gate(tabt, gate, 0, 1 if q2 is not None else None, where=torch.from_numpy(mask))
    assert np.array_equal(tab, tb.to_numpy(tabt))


@pytest.mark.parametrize("n", NS)
def test_symplectic_product_matches_numpy(n):
    rng = np.random.default_rng(n)
    a = rng.integers(2, size=(7, 2 * n), dtype=np.uint8)
    b = rng.integers(2, size=(4, 11, 2 * n), dtype=np.uint8)
    ref = sp.symplectic_product(a[None], b)
    got = tb.symplectic_product(torch.from_numpy(a)[None], torch.from_numpy(b))
    assert got.dtype == torch.uint8
    assert np.array_equal(ref, tb.to_numpy(got))


@pytest.mark.parametrize("n", NS)
def test_masks_and_distance_match_numpy_on_random_tableaux(n):
    rng = np.random.default_rng(100 + n)
    tab, tabt = _pair(n, BATCH, rng)
    e = ErrorSet.up_to_weight(n, 2)
    assert np.array_equal(R.undetected_mask(tab, K, e.paulis), tb.to_numpy(tb.undetected_mask(tabt, K, e.paulis)))
    assert np.array_equal(R.in_stabilizer_mask(tab, K, e.paulis), tb.to_numpy(tb.in_stabilizer_mask(tabt, K, e.paulis)))
    # chunking does not change the result
    assert np.array_equal(
        R.undetected_mask(tab, K, e.paulis, chunk=100), tb.to_numpy(tb.undetected_mask(tabt, K, e.paulis, chunk=100))
    )
    assert np.array_equal(R.batched_distance(tab, K, 2), tb.to_numpy(tb.batched_distance(tabt, K, 2)))
    np.testing.assert_allclose(R.kl_reward(tab, K, e), tb.to_numpy(tb.kl_reward(tabt, K, e)))
    np.testing.assert_allclose(
        R.low_weight_stabilizer_penalty(tab, K, 1), tb.to_numpy(tb.low_weight_stabilizer_penalty(tabt, K, 1))
    )


@pytest.mark.parametrize("name", list(known_codes.KNOWN))
def test_masks_and_distance_match_numpy_on_known_codes(name):
    code = known_codes.get(name)
    n, k, d = known_codes.parameters(name)
    tab = np.repeat(code.to_tableau(), 6, axis=0)
    rng = np.random.default_rng(7)
    # a few random Clifford gates on half the batch so the batch is not uniform
    table = build_action_table(graphs.all_to_all(n), T.GATES)
    tabt = tb.from_numpy(tab)
    for _ in range(5):
        a = rng.integers(len(table), size=6)
        a[:3] = -1
        table.apply(tab, a)
        tb.ActionTable(table).apply(tabt, torch.from_numpy(a))
    e, _ = paulis_up_to_weight(n, d)
    ref_u = R.undetected_mask(tab, k, e)
    ref_s = R.in_stabilizer_mask(tab, k, e)
    assert np.array_equal(ref_u, tb.to_numpy(tb.undetected_mask(tabt, k, e)))
    assert np.array_equal(ref_s, tb.to_numpy(tb.in_stabilizer_mask(tabt, k, e)))
    ref_d = R.batched_distance(tab, k, d)
    got_d = tb.batched_distance(tabt, k, d)
    assert np.array_equal(ref_d, tb.to_numpy(got_d))
    assert ref_d[0] == d  # the untouched copies keep the code's distance
    np.testing.assert_allclose(R.distance_reward(tab, k, d), tb.to_numpy(tb.distance_reward(tabt, k, d)))


def test_numpy_reward_adapter_and_composite():
    n = 5
    rng = np.random.default_rng(3)
    tab, _ = _pair(n, 16, rng)
    e = ErrorSet.up_to_weight(n, 2)
    ref = R.CompositeReward([(1.0, R.make_kl_reward(e)), (0.5, R.make_low_weight_penalty(1))])(tab, K)
    torch_reward = tb.CompositeReward([(1.0, tb.make_kl_reward(e, device="cpu")), (0.5, tb.make_low_weight_penalty(1))])
    got = tb.numpy_reward_adapter(torch_reward, device="cpu")(tab, K)
    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(ref, got)


def test_backends_package_does_not_import_torch():
    import autoqec_sim.backends as pkg

    # the package namespace is torch-free; only the torch_backend module imports torch
    assert "torch" not in pkg.__dict__
    assert tb.GATES == T.GATES
