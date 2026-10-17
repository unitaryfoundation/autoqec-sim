"""Subsystem codes: the center, the bare logicals, and the gauge fixing from
the gauge generators; the memory circuit that measures the gauge
generators and reads each stabilizer as a parity of their outcomes."""

import numpy as np
import pytest

from autoqec_sim import SubsystemCode, evaluate_code, known_codes
from autoqec_sim import memory_objective as mo
from autoqec_sim.noise import presets
from autoqec_sim.symplectic import symplectic_product


def _check_structure(code: SubsystemCode, n, k, num_gauge, num_stabilizers):
    assert (code.n, code.k, code.num_gauge, code.stabilizers.shape[0]) == (n, k, num_gauge, num_stabilizers)
    # the center commutes with every gauge generator and lies in their span
    assert not symplectic_product(code.stabilizers, code.gauge).any()
    # the bare logicals commute with every gauge generator and pair up symplectically
    assert not symplectic_product(code.logical_x, code.gauge).any()
    assert not symplectic_product(code.logical_z, code.gauge).any()
    assert np.array_equal(symplectic_product(code.logical_x, code.logical_z), np.eye(k, dtype=np.uint8))
    assert code.fixed.k == k and code.fixed.n == n
    factors, signs = code.stabilizer_factors()
    assert len(factors) == num_stabilizers and set(signs) <= {1, -1}
    for i, idx in enumerate(factors):
        prod = np.bitwise_xor.reduce(code.gauge[idx], axis=0)
        assert np.array_equal(prod, code.stabilizers[i])


def _check_circuit(code, rounds, detectors_per_round):
    for basis in "ZX":
        mc = mo.build_memory_circuit(code, rounds, basis)
        assert mc.extraction == "batched" and mc.num_generators == code.num_gauge
        c = mc.circuit()
        assert c.num_detectors == detectors_per_round * (rounds + 1)
        # every stabilizer parity is deterministic in the noiseless circuit
        assert c.detector_error_model().num_errors == 0
    with pytest.raises(ValueError):
        mo.build_memory_circuit(code, 1, "Z", extraction="interleaved")


def test_bacon_shor_from_gauge():
    code = known_codes.get_subsystem("bacon_shor_9")
    _check_structure(code, 9, 1, 12, 4)
    assert code.fixed.distance_exact() == 3 and code.parameters() == (9, 1, 3)
    assert sorted(code.stabilizer_strings()) == sorted(["XXXXXXIII", "IIIXXXXXX", "ZZIZZIZZI", "IZZIZZIZZ"])
    assert all(len(f) == 3 for f in code.stabilizer_factors()[0])
    _check_circuit(code, 2, 4)
    # weight-2 gauge measurements have no hook error: the circuit keeps the distance
    ev = evaluate_code(code, presets.chain(), scale=0.25, rounds=2)
    assert ev.d_circ == {"Z": 3, "X": 3} and ev.p_round > 0
    assert all(len(f.mechanisms) == 3 for f in ev.dominant_faults)
    assert all(ev.per_basis[b]["extraction"] == "batched" and ev.per_basis[b]["batches"] == 2 for b in "ZX")


def test_heavy_hex_from_gauge():
    code = known_codes.get_subsystem("heavy_hex_9")
    _check_structure(code, 9, 1, 10, 6)
    assert code.parameters() == (9, 1, 3)
    _check_circuit(code, 2, 6)
    ev = evaluate_code(code, presets.depolarizing(1e-3), rounds=2)
    assert ev.d_circ == {"Z": 3, "X": 3} and ev.p_round > 0


def test_subsystem_code_in_the_memory_objective_cache():
    code = known_codes.get_subsystem("bacon_shor_9")
    obj = mo.MemoryObjective(presets.chain(), 0.25, rounds=1, tier="proxy")
    a = obj.evaluate(code)
    b = obj.evaluate(code)
    assert a is b and obj.cache_hits == 1
    assert mo.code_key(code) != mo.code_key(code.fixed)
    assert mo.code_distance(code) == 3
