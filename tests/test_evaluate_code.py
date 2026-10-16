"""`evaluate_code`: the rotated surface code under uniform depolarizing noise
at the proxy tier (error per round, exact circuit distance, dominant faults
explained to circuit locations), the fault enumeration against `exact_dcirc`,
the documentation example, and the Tesseract tier against the sampled tier
(skipped without `tesseract_decoder`)."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import re
from pathlib import Path

import numpy as np
import pytest

from autoqec_sim import CodeEvaluation, DominantFault, FaultLocation, evaluate_code, known_codes
from autoqec_sim import memory_objective as mo
from autoqec_sim.noise import presets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def surface_proxy():
    return evaluate_code(known_codes.get("rotated_surface_9"), presets.depolarizing(1e-3))


def test_proxy_tier_rates_distance_and_structure(surface_proxy):
    ev = surface_proxy
    assert isinstance(ev, CodeEvaluation)
    assert ev.tier == "proxy" and ev.rounds == 3
    assert 0 < ev.p_round < 0.1
    assert ev.d_circ == {"Z": 3, "X": 3}
    assert all(ev.d_circ_mass[b] > 0 for b in ("Z", "X"))
    for basis in ("Z", "X"):
        d = ev.per_basis[basis]
        assert d["d_circ"] == 3 and d["d_circ_mass"] == ev.d_circ_mass[basis] and d["d_circ_count"] > 0
        assert d["p_round"] > 0 and "decoder" not in d
    assert abs(ev.p_round - sum(ev.per_basis[b]["p_round"] for b in ("Z", "X"))) < 1e-15
    with pytest.raises(AttributeError):
        ev.p_round = 0.0  # frozen


def test_dominant_faults_are_minimum_weight_and_located(surface_proxy):
    faults = surface_proxy.dominant_faults
    assert faults and len(faults) == 20  # ten per basis
    assert all(isinstance(f, DominantFault) for f in faults)
    assert {f.basis for f in faults} == {"Z", "X"}
    assert all(faults[i].probability >= faults[i + 1].probability for i in range(len(faults) - 1))
    for f in faults:
        assert len(f.mechanisms) == 3 and len(f.locations) == 3 and f.probability > 0
        for loc in f.locations:
            assert isinstance(loc, FaultLocation)
            assert loc.gate != "?" and loc.qubits and loc.tick >= 0
            assert re.fullmatch(r"(flip )?[XYZ]\d+(\*[XYZ]\d+)*", loc.pauli)
            assert re.fullmatch(r"(D\d+ )*(L\d+ ?)*", loc.targets + " ")
    # the most probable set is three data-qubit faults of the memory basis
    # that together flip the logical and no detector
    top = faults[0]
    assert sum("L0" in loc.targets for loc in top.locations) % 2 == 1
    assert all(f.probability <= surface_proxy.d_circ_mass[f.basis] for f in faults)
    assert surface_proxy.to_dict()["dominant_faults"][0]["mechanisms"] == top.mechanisms


@pytest.mark.parametrize("name", ["five_qubit", "steane", "shor"])
def test_undetectable_sets_agree_with_exact_dcirc(name):
    code = known_codes.get(name)
    mc = mo.build_memory_circuit(code, 2, "Z")
    sig = mo.dem_signature(mo.memory_dem(mc, presets.chain(), 0.25))
    dc = mo.exact_dcirc(sig, 4)
    us = mo.undetectable_sets(sig, 3, limit=5)
    assert us["weight"] == dc["d_circ"] and us["count"] == dc["count"]
    assert 0 < len(us["sets"]) <= 5
    for idx, p in us["sets"]:
        det = np.bitwise_xor.reduce(sig.det[list(idx)], axis=0)
        obs = np.bitwise_xor.reduce(sig.obs[list(idx)])
        assert not det.any() and obs != 0 and len(set(idx)) == len(idx) == dc["d_circ"]
        assert p == pytest.approx(float(np.prod(sig.probs[list(idx)])))


def test_stabilizer_strings_and_bad_tier():
    ev = evaluate_code(known_codes.FIVE_QUBIT, presets.chain(), scale=0.25, rounds=2)
    assert ev.rounds == 2 and ev.d_circ == {"Z": 2, "X": 2} and ev.dominant_faults
    assert all(len(f.mechanisms) == 2 for f in ev.dominant_faults)
    with pytest.raises(ValueError):
        evaluate_code(known_codes.FIVE_QUBIT, presets.chain(), tier="exact")


def test_docs_example_proxy_section_runs(capsys):
    text = (ROOT / "docs" / "examples" / "evaluate_code.py").read_text()
    m = re.search(r"# --8<-- \[start:proxy\]\n(.*?)# --8<-- \[end:proxy\]", text, re.S)
    assert m is not None
    exec(compile(m.group(1), "evaluate_code.py", "exec"), {})
    out = capsys.readouterr().out
    assert "error per round" in out and "d_circ {'Z': 3, 'X': 3}" in out


def test_tesseract_tier_matches_sampled_tier():
    pytest.importorskip("tesseract_decoder")
    code = known_codes.get("rotated_surface_9")
    profile = presets.depolarizing(1e-3)
    kw = dict(rounds=3, shots=20_000, seed=0)
    sampled = evaluate_code(code, profile, tier="sampled", **kw)
    tess = evaluate_code(code, profile, tier="tesseract", **kw)
    assert tess.tier == "tesseract" and tess.d_circ == sampled.d_circ == {"Z": 3, "X": 3}
    for basis in ("Z", "X"):
        s = tess.per_basis[basis]
        assert s["decoder"] == "tesseract" and s["sampling"]["shots"] == 20_000 and s["sampling"]["failures"] > 0
    assert 0.5 < tess.p_round / sampled.p_round < 2.0
