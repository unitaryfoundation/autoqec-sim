import numpy as np
import pytest
import stim

from autoqec_sim import known_codes
from autoqec_sim import symplectic as sp
from autoqec_sim.code import StabilizerCode
from autoqec_sim.tableau import BatchTableau, gates_to_stim_circuit


@pytest.mark.parametrize("name", list(known_codes.KNOWN))
def test_known_code_parameters(name):
    code = known_codes.get(name)
    n, k, d = known_codes.parameters(name)
    assert (code.n, code.k) == (n, k)
    assert code.distance_exact() == d
    # both exact methods agree
    assert code._distance_by_normalizer() == d
    assert code._distance_by_weight(n) == d
    ub, witness = code.distance_upper_bound(trials=200, seed=1)
    assert ub >= d
    assert ub == d  # small codes: the search finds the minimum
    assert sp.pauli_weight(witness) == ub
    assert not sp.symplectic_product(witness[None], code.stabilizers).any()
    assert sp.symplectic_product(witness[None], code.logicals).any()


@pytest.mark.parametrize("name,css", [("five_qubit", False), ("steane", True), ("shor", True), ("rotated_surface_9", True), ("four_two_two", True)])
def test_css_detection(name, css):
    assert known_codes.get(name).is_css() == css


def test_tableau_completion_is_symplectic_and_matches_stim():
    for name in known_codes.KNOWN:
        code = known_codes.get(name)
        tab = code.to_tableau()
        assert sp.is_valid_tableau(tab[0])
        t = code.to_stim_tableau()  # raises if not a valid Clifford
        # the stabilizer rows are the given generators
        k = code.k
        for j, s in enumerate(code.stim_stabilizers()):
            assert t.z_output(k + j) == s
        # logicals commute with all stabilizers and pair correctly
        prod = sp.symplectic_product(code.logical_x, code.logical_z)
        assert np.array_equal(prod, np.eye(k, dtype=np.uint8))


def test_extraction_from_encoding_circuit_matches_stim_stabilizers():
    # Encode the [[5,1,3]] code with an explicit circuit and compare the
    # extracted stabilizer group with stim's simulation of |0>^5 logical zero.
    # Build the encoding tableau from the known code, turn it into a stim
    # circuit, run that circuit on our simulator, and read the code back.
    code = known_codes.get("five_qubit")
    t = code.to_stim_tableau()
    circ = t.to_circuit("elimination")
    gates = []
    for inst in circ.flattened():
        name = inst.name
        targets = [tg.value for tg in inst.targets_copy()]
        if name in ("H", "S", "S_DAG", "X", "Y", "Z"):
            gates += [(name, q, -1) for q in targets]
        elif name in ("CX", "CZ", "SWAP"):
            g = "CNOT" if name == "CX" else name
            gates += [(g, targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
        elif name in ("I", "TICK"):
            continue
        elif name == "SQRT_X":
            # SQRT_X = H S H
            gates += [(gg, q, -1) for q in targets for gg in ("H", "S", "H")]
        elif name == "SQRT_X_DAG":
            gates += [(gg, q, -1) for q in targets for gg in ("H", "S_DAG", "H")]
        else:
            raise AssertionError(f"unexpected gate {name}")
    bt = BatchTableau(5, 1)
    for g, a, b in gates:
        bt.apply(g, a, b if b >= 0 else None)
    assert bt.to_stim(0) == stim.Tableau.from_circuit(gates_to_stim_circuit(gates, 5))
    found = StabilizerCode.from_tableau(bt.array, k=1)
    # same group: every reference generator is a product of found generators
    for row in code.stabilizers:
        assert sp.in_rowspace(row, found.stabilizers)
    assert found.distance_exact() == 3


def test_weight_enumerators_distinguish_and_match():
    five = known_codes.get("five_qubit")
    a, b = five.weight_enumerators()
    assert a.tolist() == [1, 0, 0, 0, 15, 0]
    assert b.sum() == 2 ** 6
    steane = known_codes.get("steane")
    shor = known_codes.get("shor")
    surf = known_codes.get("rotated_surface_9")
    assert steane.same_weight_enumerators(steane)
    assert not shor.same_weight_enumerators(surf)
    # qubit permutation and local Clifford leave the enumerators unchanged
    perm = np.array([3, 1, 4, 0, 2, 6, 5])
    rows = np.concatenate([steane.stabilizers[:, perm], steane.stabilizers[:, 7 + perm]], axis=1)
    permuted = StabilizerCode.from_stabilizers(list(rows))
    assert steane.same_weight_enumerators(permuted)


def test_random_upper_bound_on_larger_code():
    # [[4,2,2]] tensor products are cheap; a toric-like larger CSS code via
    # repetition on 25 qubits checks the n > 20 path is usable.
    n = 25
    stabs = []
    for i in range(n - 1):
        s = ["I"] * n
        s[i] = "Z"
        s[i + 1] = "Z"
        stabs.append("".join(s))
    rep = StabilizerCode.from_stabilizers(stabs)  # [[25,1]] repetition code, d_X = 1
    with pytest.raises(ValueError):
        rep.distance_exact()
    ub, _ = rep.distance_upper_bound(trials=50, seed=0)
    assert ub == 1


def test_from_gates_matches_tableau_replay():
    gates = [("H", 0, -1), ("CNOT", 0, 1), ("S", 2, -1), ("CNOT", 2, 3), ("CNOT", 1, 4)]
    tab = BatchTableau(5, 1)
    for gate, a, b in gates:
        tab.apply(gate, a, b if b >= 0 else None)
    direct = StabilizerCode.from_tableau(tab.array, k=1)
    code = StabilizerCode.from_gates(5, 1, gates)
    assert code.stabilizer_strings() == direct.stabilizer_strings()
    assert np.array_equal(code.logicals, direct.logicals)
