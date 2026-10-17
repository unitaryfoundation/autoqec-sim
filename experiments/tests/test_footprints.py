"""The footprints experiment: the two CSS references, the distance and
minimum-weight-logical count from the weight enumerators, and the CNF
sampler (when pycmsgen is installed)."""

import pytest

from autoqec_sim import known_codes
from autoqec_sim.code import StabilizerCode
from autoqec_sim.symplectic import pauli_to_vec, symplectic_product
from experiments.footprints import run as fpr


def test_references_have_the_expected_parameters():
    steane = fpr.footprint("steane")
    assert len(steane["faces"]) == 3 and all(len(f) == 4 for f in steane["faces"])
    code = StabilizerCode.from_stabilizers(steane["reference"])
    assert fpr.distance_and_count(code) == (3, 21) and code.parameters() == (7, 1, 3)
    color = fpr.footprint("color488")
    faces = color["faces"]
    assert sorted(len(f) for f in faces) == [4] * 7 + [8]
    qubits = [q for f in faces for q in f]
    assert len(set(qubits)) == 17
    # three corner qubits in one face, nine boundary qubits in two, five bulk qubits in three
    assert sorted(qubits.count(q) for q in range(17)) == [1] * 3 + [2] * 9 + [3] * 5
    for i, f in enumerate(faces):
        for g in faces[:i]:
            assert len(set(f) & set(g)) in (0, 2)
    code = StabilizerCode.from_stabilizers(color["reference"])
    assert fpr.distance_and_count(code) == (5, 153) and code.parameters() == (17, 1, 5)


def test_distance_and_count_matches_the_exact_distance_on_known_codes():
    for name in ("five_qubit", "steane", "rotated_surface_9"):
        code = known_codes.get(name)
        d, count = fpr.distance_and_count(code)
        assert d == code.distance_exact() and count > 0


def test_sampler_yields_commuting_distinct_codes_on_the_faces():
    pytest.importorskip("pycmsgen")
    fp = fpr.footprint("steane")
    codes = fpr.sample(fp, seed=3, count=8, known=set(), log=lambda *_: None)
    assert len(codes) == 8
    for _tag, stabs in codes:
        rows = [pauli_to_vec(s) for s in stabs]
        assert not symplectic_product(rows, rows).any()
        for i, face in enumerate(fp["faces"]):
            a, b = stabs[2 * i], stabs[2 * i + 1]
            assert a != b
            for s in (a, b):
                assert all((ch != "I") == (q in face) for q, ch in enumerate(s))
        assert StabilizerCode.from_stabilizers(stabs).k == 1
    keys = {fpr.code_key(StabilizerCode.from_stabilizers(stabs)) for _, stabs in codes}
    assert len(keys) == 8
