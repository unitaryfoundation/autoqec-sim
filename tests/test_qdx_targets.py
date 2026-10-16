"""The QDX-targets experiment on its smallest target with a small budget."""

import pytest

torch = pytest.importorskip("torch")

from autoqec_sim.experiments import qdx_targets


def test_qdx_target_table_is_consistent():
    assert len(qdx_targets.QDX_TARGETS) == 51
    for (n, k, d), (codes, lo, med, hi) in qdx_targets.QDX_TARGETS.items():
        assert 5 <= n <= 15 and 1 <= k < n and d in (3, 4, 5)
        assert codes >= 1 and lo <= med <= hi


def test_five_qubit_target_is_found_and_verified():
    rec = qdx_targets.run_target(5, 1, 3, seed=0, max_env_steps=200_000, minutes=2.0, threads=2)
    assert rec["success"] and rec["verified"] and rec["verified_d"] == 3
    assert rec["qdx"]["codes"] == 930
