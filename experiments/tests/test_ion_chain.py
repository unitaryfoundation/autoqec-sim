"""All-to-all ion-chain experiment on the CPU: the settings are the 4x4
setting with the graph and gate set changed, one tiny cell runs end to end
and writes the record the report reads, and the report renders with the
lattice runs and baselines absent. Skipped without torch."""

import pytest

torch = pytest.importorskip("torch")

from experiments.ion_chain import run as ic
from experiments.noise_aware import run as na


def test_settings_match_the_4x4_protocol():
    base = na.SETTINGS["4x4"]
    for s in ic.SETTINGS.values():
        assert (s.n, s.k, s.d, s.max_steps, s.fractional) == (base.n, base.k, base.d, base.max_steps, base.fractional)
        assert s.graph_spec == "all_to_all:16"
    assert ic.SETTINGS["a2a16"].gates == ("H", "S", "CNOT")
    assert ic.SETTINGS["a2a16_hc"].gates == base.gates
    assert set(na.SETTINGS).isdisjoint(ic.SETTINGS)


def test_smoke_cell_and_report(tmp_path):
    calib = na.calibrate(ic.SETTINGS["a2a16"], count=4)
    rec = ic.run_one("a2a16", "ion", 0, tmp_path, device="cpu", batch=32, steps=1024, threads=1, calib=calib)
    for key in ("setting", "profile", "seed", "started_at", "finished_at", "env_steps", "wall_time_s", "hit_wall_cap", "success", "best", "modal", "history", "calibration"):
        assert key in rec
    assert rec["num_actions"] == 16 + 16 + 16 * 15
    assert rec["ppo_config"]["policy_resolved"] == "equivariant"
    assert (tmp_path / "run_a2a16_ion_seed0.json").exists()
    assert (tmp_path / "logs" / "run_a2a16_ion_seed0.log").exists()
    text = ic.write_report(tmp_path, lattice_dir=tmp_path / "none", baselines_path=tmp_path / "none.json").read_text()
    assert "Setting a2a16" in text
    assert "All-to-all against the lattice" in text
    assert (tmp_path / "figures" / "curves_a2a16.png").exists()
