"""The SAT-designed code selection loads, verifies, and scores at the proxy tier."""

from pathlib import Path

from autoqec_sim.code import StabilizerCode
from experiments.memory_objective.run import DEPOLARIZING_SCALE, biased_idle_profile, evaluate_tiers
from experiments.sat_codes import run as sat_codes

ROOT = Path(__file__).resolve().parents[2]


def test_selection_loads_as_distance_3_codes():
    codes = sat_codes.load_codes(ROOT / sat_codes.SELECTION)
    assert [c["id"] for c in codes[-2:]] == ["rotated_surface", "xzzx"]
    assert len(codes) == 17
    for c in codes:
        code = StabilizerCode.from_stabilizers(c["stabilizers"])
        assert code.parameters() == (9, 1, 3), c["id"]


def test_proxy_tier_scores_a_selected_code():
    codes = sat_codes.load_codes(ROOT / sat_codes.SELECTION)
    code = StabilizerCode.from_stabilizers(codes[0]["stabilizers"])
    rec = evaluate_tiers(code, biased_idle_profile(5.0, "parallel"), 3, ["proxy"], 0, 1000, 5, 2000, scale=DEPOLARIZING_SCALE)
    assert 0 < rec["proxy"]["p_round"] < 0.1
