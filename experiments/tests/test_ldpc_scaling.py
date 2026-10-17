"""LDPC constraint forms above n = 9 (experiments.ldpc_scaling): one tiny
run per form and the report tables. Skipped without torch."""

import pytest

torch = pytest.importorskip("torch")

from experiments.ldpc_scaling import run as L


@pytest.mark.parametrize("form", ["unnormalized", "final", "schedule"])
def test_tiny_run_records_everything(form):
    rec = L.run_form(
        form, "n12_k2_all_to_all", seed=0, minutes=0.5, threads=1, batch=32, max_env_steps=3 * 16 * 32,
        post_success_updates=1, schedule_window=1, schedule_rate=0.0,
    )
    assert rec["kind"] == "ldpc_form" and rec["form"] == form and rec["policy"] == "equivariant"
    assert (rec["n"], rec["k"], rec["d"], rec["w_max"]) == (12, 2, 3, 4)
    assert rec["updates"] == 3 and rec["env_steps"] == 3 * 16 * 32
    assert len(rec["history"]) == 3 and all("terminal_w_star" in h for h in rec["history"])
    assert rec["objective"]["form"] == form
    if form == "schedule":
        assert rec["switch"] is not None and rec["switch"]["update"] == 1
        assert rec["history"][0]["ldpc_active"] is False and rec["history"][-1]["ldpc_active"] is True
    else:
        assert rec["switch"] is None
    assert rec["terminal_last"]["n_terminal"] >= 0 and "sampled" in rec["final_policy_probe"]
    rec["_file"] = f"{form}_n12_k2_all_to_all_seed0.json"
    for table in (L.summary_table([rec]), L.run_table([rec]), L.sources_table([rec])):
        assert form in table and ("[[12, 2, 3]]" in table or "n12_k2" in table)
