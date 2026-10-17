"""Sparsity variants (experiments.ldpc_sparsity): one tiny run per variant
and the report tables. Skipped without torch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.ldpc_sparsity import run as S


@pytest.mark.parametrize("variant", S.VARIANTS)
def test_tiny_run_records_everything(variant):
    rec = S.run_variant(variant, "n12_k2_all_to_all", seed=0, c=0.05, max_env_steps=3 * 16 * 32, minutes=1, threads=1, batch=32)
    assert rec["kind"] == "ldpc_sparsity" and rec["variant"] == variant and rec["policy"] == "equivariant"
    assert rec["form"] == {"mask": "unnormalized", "mask_terminal": "terminal"}.get(variant, variant)
    assert rec["updates"] == 3 and rec["env_steps"] == 3 * 16 * 32 and rec["budget_env_steps"] == 3 * 16 * 32
    for h in rec["history"]:
        assert {"terminal_joint", "terminal_at_d", "terminal_inside_at_d", "terminal_heavy_at_d"} <= set(h)
        assert h["terminal_inside_at_d"] <= h["terminal_at_d"] <= sum(h["terminal_d"])
    for key in ("terminal_last", "terminal_all"):
        assert {"at_d", "inside", "inside_frac", "heavy_at_d_hist"} <= set(rec[key])
    if variant in S.MASKED:
        assert rec["mask"]["calls"] == 3 * 16 + 3 and 0.0 <= rec["mask"]["two_masked_frac_last"] <= 1.0
        assert "mask_two_masked_frac" in rec["history"][0]
    else:
        assert rec["mask"] is None
    assert (rec["c"] == 0.05) == (variant == "shaped")
    rec["_file"] = S.run_name(variant, "n12_k2_all_to_all", 0, 0.05, 3 * 16 * 32) + ".json"
    for table in (S.summary_table([rec]), S.run_table([rec]), S.sources_table([rec])):
        assert variant in table and "[[12, 2, 3]]" in table


def test_inside_bound_from_joint_counts():
    joint = np.zeros((4, 13), dtype=int)
    joint[3, 4] = 6  # d = 3, w* = 4
    joint[3, 6] = 2  # d = 3, w* = 6
    joint[2, 3] = 5  # d = 2, ignored
    out = S.inside_bound({"joint": joint.tolist()}, 3, 4)
    assert out == {"at_d": 8, "inside": 6, "inside_frac": 0.75}
    assert S.inside_bound({"joint": []}, 3, 4)["inside_frac"] is None
    assert S.run_name("shaped", "n12_k2_square", 2, 0.2) == "shaped_c0.2_n12_k2_square_seed2"
    assert S.run_name("unnormalized", "n12_k2_all_to_all", 1, budget=3_000_000) == "unnormalized_3M_n12_k2_all_to_all_seed1"


def test_torch_reward_backend_matches_numpy_and_records_device():
    # the torch base terms on cpu reproduce the numpy counts on random states
    import numpy as np

    from autoqec_sim import graphs
    from autoqec_sim.env import CodeDiscoveryEnv
    from autoqec_sim.ldpc import LdpcObjective

    env = CodeDiscoveryEnv(n=12, k=2, graph=graphs.from_spec("all_to_all:12"), batch_size=64, max_steps=30, seed=7)
    env.reset()
    for _ in range(15):
        env.step(env.sample_actions())
    ref = LdpcObjective(12, 2, 3, 4, form="terminal")
    dev = LdpcObjective(12, 2, 3, 4, form="terminal", device="cpu", chunk=500)
    assert np.array_equal(ref(env.tab, 2), dev(env.tab, 2))
    u0, p0 = ref.base_terms(env.tab, 2)
    u1, p1 = dev.base_terms(env.tab, 2)
    assert np.array_equal(u0, u1) and np.array_equal(p0, p1)
    assert ref.describe()["reward_backend"] == "numpy" and dev.describe()["reward_backend"] == "torch:cpu"
    assert ref.describe()["w_star_test"] == "exact"
    assert LdpcObjective(20, 4, 3, 4, form="terminal").describe()["w_star_test"] == "greedy bound"
    # a tiny device-path run on the smallest cell records the device fields and the torch terminal distance
    rec = S.run_variant("mask_terminal", "n12_k2_all_to_all", seed=0, max_env_steps=2 * 16 * 32, minutes=1, threads=1, batch=32, device="cpu", reward_backend="torch")
    assert rec["device"] == "cpu" and rec["reward_backend"] == "torch" and rec["w_star_test"] == "exact"
    assert rec["reward_calls"] >= 2 * 16 and rec["reward_time_s"] >= 0 and rec["update_time_s"] >= 0
    assert rec["max_steps"] == 40 and rec["d"] == 3


def test_cell_specs_and_capped_description():
    n, k, d, graph, gates, steps = S.CELL_SPECS["n25_k1_d5_all_to_all"]
    assert (n, k, d, graph, steps) == (25, 1, 5, "all_to_all:25", 80) and "S" in gates
    assert S.CELL_SPECS["n20_k4_all_to_all"][2] == 3 and S.CELL_SPECS["n16_k4_all_to_all"][5] == 40
    assert set(S.TARGETS) == set(S.CELLS)
    from autoqec_sim import known_codes

    tab = known_codes.get("rotated_surface_9").to_tableau()[0]
    assert S.describe_code_capped(tab, 1, 3)["d"] == 3  # n <= 20: the exact path
    from autoqec_sim.tableau import identity_tableau

    desc = S.describe_code_capped(identity_tableau(25, 1)[0], 1, 4)
    assert desc["d"] == 1 and desc["d_capped_at"] == 5 and desc["w_star_test"] == "greedy bound"


def test_run_name_and_label_carry_mask_method_and_batch():
    assert S.run_name("mask_terminal", "n16_k4_square", 1, budget=3_000_000, mask_method="exact") == "mask_terminal_exact_3M_n16_k4_square_seed1"
    assert S.run_name("mask_terminal", "n16_k4_square", 0, budget=3_000_000, batch=1024) == "mask_terminal_3M_b1024_n16_k4_square_seed0"
    assert S.run_name("mask", "n12_k2_square", 2) == "mask_n12_k2_square_seed2"
    rec = {"variant": "mask_terminal", "c": None, "budget_env_steps": 3_000_000, "batch": 1024, "n": 16, "mask": {"method": "exact"}}
    assert S.label(rec) == "mask_terminal (exact mask) (3M steps, batch 1024)"
    assert S.base_label(S.label(rec)) == "mask_terminal (exact mask)"
    assert S.base_label("mask_terminal (3M steps)") == "mask_terminal"
    assert S.base_label("shaped (c = 0.05)") == "shaped (c = 0.05)"
    rec = {"variant": "mask_terminal", "c": None, "budget_env_steps": 3_000_000, "batch": 256, "n": 16, "mask": {"method": "tableau"}}
    assert S.label(rec) == "mask_terminal (3M steps, tableau mask)" and S.base_label(S.label(rec)) == "mask_terminal"
    assert S.label(dict(rec, n=12, budget_env_steps=S.DEFAULT_BUDGET)) == "mask_terminal"


def test_exact_mask_run_records_counters_and_bench():
    rec = S.run_variant("mask_terminal", "n12_k2_square", seed=0, max_env_steps=2 * 16 * 16, minutes=1, threads=1, batch=16, mask_method="exact")
    m = rec["mask"]
    assert m["method"] == "exact" and m["method_effective"] == "exact" and m["calls"] == rec["env_steps"] // 16 + 2
    assert m["pairs"] == m["calls"] * 16 * 34 and m["exact_tested"] == m["greedy_rejected"] >= m["exact_admitted"] >= 0
    assert 0 < m["cache_hits"] + m["cache_misses"] <= m["calls"] * 16  # states are enumerated only on calls with a rejected pair
    assert "mask_two_greedy_masked_frac" in rec["history"][-1] and m["greedy_two_masked_frac_last"] >= m["two_masked_frac_last"]
    bench = S.bench_mask("n12_k2_square", 8, steps=3)
    assert set(bench) >= {"basis", "exact"} and bench["exact"]["tested_per_call"] >= 0 and bench["two_qubit_actions"] == 34
