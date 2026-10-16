"""The SAT population experiment: preparation from a source directory,
scoring, and the report, on a three-code stand-in for the source data."""

import csv
import json

import stim

from autoqec_sim.code import StabilizerCode
from autoqec_sim.experiments import sat_population
from autoqec_sim.experiments.memory_objective import xzzx_9
from autoqec_sim.known_codes import get as known


def _source_format(strings):
    return ["+" + s.replace("I", "_") for s in strings]


def _write_source(root):
    surface = known("rotated_surface_9").stabilizer_strings()
    product = str(stim.PauliString(surface[0]) * stim.PauliString(surface[1])).lstrip("+-").replace("_", "I")
    same_group = [product, *surface[1:]]  # another generator list of the same stabilizer group
    codes = [surface, same_group, xzzx_9()]
    (root / "surface_codes.json").write_text(json.dumps([_source_format(c) for c in codes]))
    (root / "data").mkdir()
    with open(root / "data" / "output_0.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "eta", "ler"])
        for i in range(len(codes)):
            for eta in range(1, 11):
                w.writerow([f"D{i}", f"{eta}.0", 0.001 * eta * (i + 1)])
    with open(root / "bias_sampled_lers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "eta", "ler"])
        for name in ("CSS", "XZZX"):
            for eta in range(1, 11):
                w.writerow([name, f"{eta}.0", 0.002 * eta])


def test_prepare_run_and_report(tmp_path):
    src = tmp_path / "surface"
    src.mkdir()
    _write_source(src)
    population = tmp_path / "population.json"
    pop = sat_population.prepare(src, population, n_random=2, n_top=1, n_bottom=1, seed=0)
    assert pop["source_codes"] == 3 and pop["distinct_strings"] == 3 and pop["distinct_groups"] == 2
    ids = [e["id"] for e in pop["codes"]]
    assert ids[-2:] == ["rotated_surface", "xzzx"]
    for e in pop["codes"]:
        assert StabilizerCode.from_stabilizers(e["stabilizers"]).parameters() == (9, 1, 3)
    out = tmp_path / "runs"
    records = sat_population.run(out, population, rounds=3, tiers=["proxy"], shots=1000, min_failures=5, max_shots=2000, seed=0,
                                 workers=1, cases=["idle_eta_10", "forte"], limit=None, verbose=False)
    recs = sat_population.load_records(out)
    assert len(recs) == len(pop["codes"]) and records.exists()
    assert {r["id"] for r in recs[:2]} == {"rotated_surface", "xzzx"}
    for r in recs:
        for case in ("idle_eta_10", "forte"):
            assert 0 < r["cases"][case]["proxy"]["p_round"] < 0.1
    report = sat_population.write_report(out, population, figure=None)
    text = report.read_text()
    assert "Rank agreement with the source" in text and "Best codes of the population" in text
    an = json.loads((out / "analysis.json").read_text())
    assert an["counts"]["scored"] == len(pop["codes"])
