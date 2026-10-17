"""GPU-scale experiment module on the CPU: the torch distance kernel and the
torch noise-aware reward agree with the numpy path, the code verifier
recognizes the rotated surface code, and one tiny cell of each target runs
end to end and writes the JSON schema the report reads. Skipped without
torch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autoqec_sim import graphs
from autoqec_sim.errors import paulis_of_weight
from autoqec_sim.ldpc import sparsest_basis_stats
from autoqec_sim.reward import batched_distance, low_weight_stabilizer_penalty, undetected_mask
from experiments.gpu_scale import run as gs
from experiments.noise_aware import run as na

COMMON_KEYS = {
    "kind", "target", "profile", "seed", "device", "device_name", "torch", "platform", "batch",
    "budget_env_steps", "wall_cap_s", "threads", "chunk", "ppo_config", "num_parameters", "num_actions",
    "env_steps", "episodes", "updates", "wall_time_s", "env_steps_per_s", "reward_time_s", "update_time_s",
    "hit_wall_cap", "success", "first_success", "history", "reward",
}
NOISE_KEYS = {
    "setting", "profile_description", "scale", "m0", "calibration", "successful_episodes", "distinct_success_codes",
    "partial_distance", "noise_evals", "cache_hits", "first_success_env_steps", "final_success_rate", "selection",
    "best", "best_reward", "modal",
}
SURFACE_KEYS = {"n", "k", "d_target", "graph", "gates", "max_steps", "step_penalty", "ldpc", "successes", "terminal_d_counts", "max_terminal_d_seen", "code", "best_state"}


def _random_env(n, spec, gates, max_steps, batch, seed):
    env = na.make_env(na.SETTINGS["4x4"], seed, batch) if n == 16 else None
    if env is None:
        from autoqec_sim.env import CodeDiscoveryEnv

        env = CodeDiscoveryEnv(n=n, k=1, graph=graphs.from_spec(spec), batch_size=batch, gates=gates, max_steps=max_steps, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)
    for _ in range(rng.integers(5, max_steps)):
        env.step(env.sample_actions())
    return env


def _numpy_reference(tab, k, d_max):
    d = batched_distance(tab, k, d_max - 1)
    pad = low_weight_stabilizer_penalty(tab, k, 1)
    n = (tab.shape[-1] - 1) // 2
    frac = np.ones(tab.shape[0])
    for w in range(1, d_max):
        idx = np.flatnonzero(d == w)
        if idx.size:
            frac[idx] = 1.0 - undetected_mask(tab[idx], k, paulis_of_weight(n, w)).mean(axis=1)
    return d, frac, pad


@pytest.mark.parametrize("n,spec,d_max,chunk", [(9, "square:3x3", 3, 50), (16, "square:4x4", 4, 4096)])
def test_distance_kernel_matches_numpy(n, spec, d_max, chunk):
    env = _random_env(n, spec, ("H", "CNOT"), 40, 48, seed=n)
    tab = env.tab
    kernel = gs.DistanceKernel(n, 1, d_max, "cpu", chunk=chunk)
    d, frac, pad = kernel(tab)
    d0, frac0, pad0 = _numpy_reference(tab, 1, d_max)
    assert np.array_equal(d, d0)
    np.testing.assert_allclose(frac, frac0)
    np.testing.assert_allclose(pad, pad0)
    assert set(np.unique(d)) <= set(range(1, d_max + 1))
    assert kernel.calls == 1


def test_torch_noise_aware_reward_matches_numpy():
    setting = na.SETTINGS["4x4"]
    calib = na.calibrate(setting, count=4)
    env = na.make_env(setting, 0, 32)
    models = na.calibrated_models(setting, calib, env.graph)
    ref = na.NoiseAwareReward(env, setting, models["ion"], calib["m0"])
    tor = gs.TorchNoiseAwareReward(env, setting, models["ion"], calib["m0"], device="cpu", chunk=1024)
    env.reward_fn = ref
    env.success_fn = ref.success_fn
    env.reset()
    for _ in range(20):
        env.step(env.sample_actions())
    r_ref = ref(env.tab, 1)
    r_tor = tor(env.tab, 1)
    np.testing.assert_allclose(r_tor, r_ref)
    assert np.array_equal(tor.last_success, ref.last_success)
    assert not ref.last_success.any()
    # a routed rotated-surface [[16,1,4]] encoder reaches the target: both rewards flag it and agree
    graph = env.graph
    gates = na.route_gates(na.css_standard_encoder(na.rotated_surface(4), graph=graph), graph)
    acts = {tuple(e): i for i, e in enumerate(env.action_table.entries)}
    env.max_steps = len(gates)
    env.history = np.full((32, env.max_steps), -1, dtype=np.int64)
    seq = np.full((32, len(gates)), -1, dtype=np.int64)
    seq[0] = [acts[tuple(g)] for g in gates]
    env.auto_reset = False  # keep the terminal tableau in env.tab (make_env turns auto-reset on)
    _, ok, _ = env.rollout(seq)
    assert ok[0] and not ok[1:].any()
    r_ref = ref(env.tab, 1)
    r_tor = tor(env.tab, 1)
    np.testing.assert_allclose(r_tor, r_ref)
    assert tor.last_success[0] and -1.0 < r_tor[0] < 0.0


def test_level_reward_numpy_and_kernel_agree():
    n, d = 9, 3
    env = _random_env(n, "square:3x3", ("H", "CNOT"), 30, 40, seed=5)
    numpy_level = gs.LevelReward(n, 1, d, device=None, chunk=64)
    torch_level = gs.LevelReward(n, 1, d, device="cpu", chunk=64)  # "cpu" stays on the numpy path
    assert numpy_level.kernel is None and torch_level.kernel is None
    kernel_level = gs.LevelReward(n, 1, d, device=torch.device("cpu"), chunk=64)
    kernel_level.kernel = gs.DistanceKernel(n, 1, d, "cpu", chunk=64)
    r0 = numpy_level(env.tab, 1)
    r1 = kernel_level(env.tab, 1)
    np.testing.assert_allclose(r0, r1)
    assert np.array_equal(numpy_level.last_d, kernel_level.last_d)
    assert np.all(r0 <= 0.0)


def test_verify_code_recognizes_rotated_surface_3x3():
    graph = graphs.from_spec("square:3x3")
    stabs = na.rotated_surface(3)
    gates = na.route_gates(na.css_standard_encoder(stabs, graph=graph), graph)
    v = gs.verify_code(gates, 9, 1, 3, stabs, "rotated surface [[9,1,3]]")
    assert v["distance_exact"] == 3 and v["meets_target"] and v["no_weight1_stabilizer"]
    assert v["is_css"] and v["enumerators_match_reference"]
    assert v["w_star"] == 4 and v["w_star_exact"]
    assert v["gate_count"] == len(gates) and v["cnot_count"] == sum(g[0] == "CNOT" for g in gates)
    assert len(v["weight_enumerator_S"]) == 10 and v["weight_enumerator_S"][0] == 1
    # a padded code (one idle qubit) is rejected by the weight-1 test
    v2 = gs.verify_code(gates[:3], 9, 1, 3, None)
    assert not v2["no_weight1_stabilizer"] and not v2["meets_target"]


def test_smoke_noise4x4_cpu(tmp_path):
    rec = gs.run_one("noise4x4", "board", 0, tmp_path, device="cpu", batch=64, steps=4096, threads=1, calib_count=4)
    assert set(rec) >= COMMON_KEYS and set(rec) >= NOISE_KEYS
    assert rec["target"] == "noise4x4" and rec["profile"] == "board" and rec["device"] == "cpu"
    assert rec["ppo_config"]["policy"] == "auto" and rec["ppo_config"]["policy_resolved"] == "equivariant"
    assert rec["env_steps"] >= 4096 and rec["updates"] == len(rec["history"]) >= 1
    assert rec["setting"]["name"] == "4x4" and rec["calibration"]["profiles"]["board"]["scale"] == 1.0
    assert rec["reward"]["fractional_distance"] is True
    h = rec["history"][-1]
    assert {"successes_in_update", "distinct_success_codes", "noise_evals", "cache_hits", "entropy"} <= set(h)
    if rec["best"] is not None:
        assert set(rec["best"]["mass"]) == set(gs.PROFILES)
        assert "reward_terms" in rec["best"]
        for g, a, b in rec["best"]["gates"]:
            if g == "CNOT":
                assert graphs.from_spec("square:4x4").has_edge(a, b)
    assert (tmp_path / "run_noise4x4_board_seed0.json").exists()
    assert (tmp_path / "logs" / "run_noise4x4_board_seed0.log").exists()


def test_smoke_surface5x5_cpu(tmp_path):
    rec = gs.run_one("surface5x5", None, 0, tmp_path, device="cpu", batch=64, steps=4096, threads=1, max_steps=40)
    assert set(rec) >= COMMON_KEYS and set(rec) >= SURFACE_KEYS
    assert rec["target"] == "surface5x5" and rec["profile"] is None
    assert (rec["n"], rec["k"], rec["d_target"], rec["max_steps"]) == (25, 1, 5, 40)
    assert rec["ppo_config"]["policy_resolved"] == "equivariant" and rec["num_actions"] == 25 + 2 * 40
    assert rec["env_steps"] >= 4096 or rec["success"]
    assert len(rec["terminal_d_counts"]) == 6
    assert {"terminal_d", "max_terminal_d_seen", "best_terminal_reward"} <= set(rec["history"][-1])
    if rec["success"]:
        assert rec["code"]["distance_exact"] >= 5 and rec["code"]["no_weight1_stabilizer"]
        assert "enumerators_match_reference" in rec["code"]
    else:
        assert rec["code"] is None and rec["best_state"] is not None
        assert rec["best_state"]["distance_capped"] <= 4
    path = gs.write_report(tmp_path)
    text = path.read_text()
    assert "## Target A" in text and "## Target B" in text and "run_surface5x5_seed0.json" in text


@pytest.mark.parametrize("n,spec", [(9, "square:3x3"), (12, "square:3x4")])
def test_sparsity_check_is_exact_and_backends_agree(n, spec):
    env = _random_env(n, spec, ("H", "CNOT"), 40, 256, seed=20 + n)
    st = sparsest_basis_stats(env.tab, 1)  # exact: n - k <= 14 generators
    for w in (2, 3, 4, 6):
        got = gs.SparsityCheck(n, 1, w).feasible(env.tab)
        assert np.array_equal(got, st["w_star"] <= w), w
        if w % 2 == 0:
            assert np.array_equal(gs.SparsityCheck(n, 1, w, device="cpu").feasible(env.tab), got), w
    ws = np.array([gs.exact_w_star(t, 1)[0] for t in env.tab[:32]])
    assert np.array_equal(ws, st["w_star"][:32])


def test_sparsity_check_accepts_rotated_surface_25_where_greedy_does_not():
    from autoqec_sim.ldpc import ldpc_feasible
    from autoqec_sim.regress import replay

    graph = graphs.from_spec("square:5x5")
    tab = replay(na.route_gates(na.css_standard_encoder(na.rotated_surface(5), graph=graph), graph), 25).to_tableau()
    assert not ldpc_feasible(tab, 1, 4)[0]  # the greedy bound above 14 generators rejects the target
    assert gs.SparsityCheck(25, 1, 4).feasible(tab)[0]
    assert gs.SparsityCheck(25, 1, 4, device="cpu").feasible(tab)[0]
    assert gs.exact_w_star(tab[0], 1) == (4, True)
    env = _random_env(25, "square:5x5", ("H", "CNOT"), 60, 64, seed=3)
    a = gs.SparsityCheck(25, 1, 4).feasible(env.tab)
    b = gs.SparsityCheck(25, 1, 4, device="cpu").feasible(env.tab)
    assert np.array_equal(a, b)


def test_sparse_env_mask_mode_keeps_every_state_sparse():
    n, d, w_max = 9, 3, 4
    level = gs.LevelReward(n, 1, d)
    check = gs.SparsityCheck(n, 1, w_max)
    env = gs.SparseLevelEnv(
        level, check, "mask", n=n, k=1, graph=graphs.from_spec("square:3x3"), batch_size=64, max_steps=30,
        gates=("H", "CNOT"), reward_fn=level, seed=1, auto_reset=True,
    )
    env.reset()
    reverted = 0
    for _ in range(30):
        _, _, _, _, info = env.step(env.sample_actions())
        reverted += int(info["ldpc_violation"].sum())
        assert (sparsest_basis_stats(env.tab, 1)["w_star"] <= w_max).all()
    assert reverted == env.violations > 0
    assert (env.history[env.history >= 0].size + reverted) >= 0


def test_smoke_surface5x5_sparse_cpu(tmp_path):
    rec = gs.run_one("surface5x5_sparse", None, 0, tmp_path, device="cpu", batch=32, steps=1024, threads=1, max_steps=20)
    assert set(rec) >= COMMON_KEYS and set(rec) >= SURFACE_KEYS
    assert rec["target"] == "surface5x5_sparse" and rec["ldpc"] == {
        "w_max": 4, "mode": "mask", "test": rec["ldpc"]["test"], "device": "cpu (numpy)",
    }
    assert rec["step_penalty"] == 0.0 and {"ldpc_check_time_s", "ldpc_check_calls", "ldpc_violations"} <= set(rec)
    assert rec["ldpc_check_calls"] >= rec["updates"] * rec["ppo_config"]["rollout_steps"]
    assert "ldpc_violation_frac" in rec["history"][-1]
    if not rec["success"]:
        assert rec["best_state"]["w_star"] <= 4 and rec["best_state"]["w_star_exact"]
    # the second mode goes to its own out-dir, as on the pod; report reads it as an extra directory
    rec2 = gs.run_one("surface5x5_sparse", None, 0, tmp_path / "final_mode", device="cpu", batch=32, steps=512, threads=1, max_steps=20, ldpc_mode="final")
    assert rec2["ldpc"]["mode"] == "final" and rec2["step_penalty"] == -0.01
    recs = gs.load_all(tmp_path)
    assert sorted(r["_file"] for r in recs) == ["final_mode/run_surface5x5_sparse_seed0.json", "run_surface5x5_sparse_seed0.json"]
    assert len(gs.load_all(tmp_path, extra_dirs=[])) == 1
    text = gs.write_report(tmp_path).read_text()
    assert "## Target C" in text and "run_surface5x5_sparse_seed0.json" in text and "`final_mode/run_surface5x5_sparse_seed0.json`" in text
    assert "| mask |" in text and "| final |" in text and "Outcome:" in text
