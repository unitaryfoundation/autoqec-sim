import numpy as np
import pytest
import stim

from autoqec_sim import tableau as T


def _random_gates(rng, n, length):
    gates = []
    for _ in range(length):
        g = rng.choice(T.GATES)
        if g in T.GATES_2Q:
            a, b = rng.choice(n, size=2, replace=False)
            gates.append((g, int(a), int(b)))
        else:
            gates.append((g, int(rng.integers(n)), -1))
    return gates


@pytest.mark.parametrize("n,batch,length", [(3, 8, 20), (6, 32, 60), (9, 16, 120)])
def test_random_circuits_agree_with_stim(n, batch, length):
    rng = np.random.default_rng(n * 1000 + length)
    circuits = [_random_gates(rng, n, length) for _ in range(batch)]
    tab = T.BatchTableau(n, batch)
    # Different circuit per batch element: apply step t of every element at
    # once through an ActionTable, which exercises the masked per-element path.
    entries = sorted({g for c in circuits for g in c})
    table = T.ActionTable(entries)
    lookup = {e: i for i, e in enumerate(table.entries)}
    for t in range(length):
        acts = np.array([lookup[c[t]] for c in circuits])
        table.apply(tab.array, acts)
    for b in range(batch):
        ref = stim.Tableau.from_circuit(T.gates_to_stim_circuit(circuits[b], n))
        assert tab.to_stim(b) == ref
        back = T.BatchTableau.from_stim(ref)
        assert np.array_equal(back.array[0], tab.array[b])


def test_masked_application_only_touches_selected_elements():
    tab = T.BatchTableau(4, 5)
    before = tab.array.copy()
    tab.h(0, where=np.array([False, True, False, True, False]))
    changed = np.any(tab.array != before, axis=(1, 2))
    assert changed.tolist() == [False, True, False, True, False]


def test_per_element_qubits():
    n, batch = 5, 5
    tab = T.BatchTableau(n, batch)
    tab.h(np.arange(batch))
    for b in range(batch):
        ref = stim.Tableau.from_circuit(T.gates_to_stim_circuit([("H", b, -1)], n))
        assert tab.to_stim(b) == ref


def test_two_qubit_same_qubit_rejected():
    tab = T.BatchTableau(3, 1)
    with pytest.raises(ValueError):
        tab.cnot(1, 1)


def test_pauli_frames_flip_signs_only():
    tab = T.BatchTableau(3, 1)
    body = tab.array[0, :, :6].copy()
    tab.pauli("X", 1).pauli("Z", 2).pauli("Y", 0)
    assert np.array_equal(tab.array[0, :, :6], body)
    ref = stim.Tableau.from_circuit(
        T.gates_to_stim_circuit([("X", 1, -1), ("Z", 2, -1), ("Y", 0, -1)], 3)
    )
    assert tab.to_stim(0) == ref
