"""PPO agent: masking, update mechanics, and a reduced-budget [[5,1,3]]
rediscovery (target: well under two minutes). Skipped without torch."""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autoqec_sim import graphs, regress
from autoqec_sim.agents import ppo
from autoqec_sim.agents.ppo import (
    EnvAdapter,
    PPOAgent,
    PPOConfig,
    graph_action_mask,
)
from autoqec_sim.code import StabilizerCode
from autoqec_sim.env import CodeDiscoveryEnv, build_action_table


def test_graph_mask_from_action_table():
    table = build_action_table(graphs.all_to_all(4), gates=("H", "CNOT"))
    mask = graph_action_mask(table, graphs.linear(4))
    allowed = {(g, a, b) for (g, a, b), m in zip(table.entries, mask) if m}
    assert {e for e in allowed if e[0] == "CNOT"} == {
        ("CNOT", 0, 1), ("CNOT", 1, 0), ("CNOT", 1, 2), ("CNOT", 2, 1), ("CNOT", 2, 3), ("CNOT", 3, 2)
    }
    assert all(m for (g, _, _), m in zip(table.entries, mask) if g == "H")


def test_undo_mask_forbids_repeating_involutions():
    env = CodeDiscoveryEnv(n=3, k=1, batch_size=2, gates=("H", "S", "CNOT"), max_steps=5, seed=0)
    env.reset()
    adapter = EnvAdapter(env, mask_undo=True)
    assert adapter.action_mask() is None  # nothing to undo at the start
    h0 = env.action_table.entries.index(("H", 0, -1))
    s0 = env.action_table.entries.index(("S", 0, -1))
    env.step(np.array([h0, s0]))
    mask = adapter.action_mask()
    assert mask.shape == (2, env.num_actions)
    assert not mask[0, h0]  # H H = I is masked
    assert mask[1, s0]  # S is not an involution
    assert mask[0].sum() == env.num_actions - 1 and mask[1].all()


def test_masked_logits_never_sampled():
    env = CodeDiscoveryEnv(n=3, k=1, batch_size=8, gates=("H", "CNOT"), max_steps=4, seed=1)
    agent = PPOAgent(env, PPOConfig(hidden=(16,), rollout_steps=4), seed=1)
    h0 = env.action_table.entries.index(("H", 0, -1))
    env.reset()
    env.step(np.full(8, h0))
    for _ in range(20):
        a, _, _ = agent._policy(agent.adapter.encode(env.tab), agent.adapter.action_mask())
        assert not np.any(a == h0)


def test_update_runs_and_changes_parameters():
    env = CodeDiscoveryEnv(n=4, k=1, batch_size=16, max_steps=6, seed=2)
    agent = PPOAgent(env, PPOConfig(hidden=(32, 32), rollout_steps=6, minibatch_size=32, epochs=2), seed=2)
    before = [p.detach().clone() for p in agent.model.parameters()]
    buf = agent.collect(6)
    assert buf["obs"].shape == (6, 16, agent.adapter.obs_dim)
    adv, ret = agent.gae(buf)
    assert adv.shape == ret.shape == (6, 16)
    stats = agent.update(buf)
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"}
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.model.parameters()))
    assert agent.env_steps == 96 and agent.updates == 1


def test_ppo_rediscovers_five_qubit_code():
    target = regress.TARGETS["five_qubit"]
    t0 = time.perf_counter()
    r = ppo_regress_run(target)
    wall = time.perf_counter() - t0
    assert r["rediscovered"], r["attempts"]
    v = r["result"]["verification"]
    assert v["parameters_match"] and v["nontrivial"] and v["distance_exact"] == 3
    # [[5,1,3]] is unique up to equivalence
    assert v["enumerators_match_reference"]
    code = regress.replay([tuple(g) for g in r["result"]["gates"]], 5)
    assert isinstance(code, StabilizerCode) and code.distance_exact() == 3
    assert wall < 120


def ppo_regress_run(target):
    from autoqec_sim.agents import ppo_regress

    return ppo_regress.run_target(
        target, seed=0, batch_size=256, max_env_steps=400_000, max_time_s=90.0, max_attempts=1, require_match=False
    )


def test_package_imports_without_touching_torch():
    # the agents package and the env stack must not import torch eagerly
    import importlib
    import sys

    import autoqec_sim.agents  # noqa: F401

    assert "autoqec_sim.agents.ppo" not in sys.modules or ppo._torch is not None
    mod = importlib.import_module("autoqec_sim.env")
    assert "torch" not in mod.__dict__


# ----------------------------------------------------------------------
# policy selection (agents.policies)
# ----------------------------------------------------------------------
def _env(n, k=1, batch=8, seed=0):
    if n in (9, 16):  # the 3x3 and 4x4 lattice targets of the regression and the scaling probe
        side = 3 if n == 9 else 4
        return CodeDiscoveryEnv(n=n, k=k, graph=graphs.from_spec(f"square:{side}x{side}"), batch_size=batch, max_steps=6, gates=("H", "CNOT"), seed=seed)
    return CodeDiscoveryEnv(n=n, k=k, graph=graphs.all_to_all(n), batch_size=batch, max_steps=6, gates=("H", "S", "CNOT"), seed=seed)


@pytest.mark.parametrize("policy", ["mlp", "equivariant", "auto"])
def test_policy_option_constructs(policy):
    agent = PPOAgent(_env(5), PPOConfig(policy=policy, hidden=(16,), max_threads=2), seed=0)
    assert agent.policy in ("mlp", "equivariant")
    assert agent.model.num_parameters() == agent.num_parameters() > 0
    with pytest.raises(ValueError):
        PPOAgent(_env(5), PPOConfig(policy="transformer", max_threads=2), seed=0)


def test_auto_policy_is_mlp_below_twelve_qubits_and_equivariant_above():
    assert ppo.resolve_policy("auto", 9) == "mlp" and ppo.resolve_policy("auto", 12) == "equivariant"
    small = PPOAgent(_env(9), PPOConfig(max_threads=2), seed=0)
    assert small.policy == "mlp"
    assert small.num_parameters() == 157_730  # the 3x3 regression number (README, "PPO agent")
    large = PPOAgent(_env(16), PPOConfig(max_threads=2), seed=0)
    assert large.policy == "equivariant"
    assert large.num_parameters() == 123_907  # experiments/scaling/REPORT.md, 4x4 equivariant runs
    assert PPOAgent(_env(16), PPOConfig(policy="mlp", max_threads=2), seed=0).num_parameters() == 344_897


def test_default_mlp_architecture_unchanged():
    # the regression targets must keep their networks (results/regression_ppo.json)
    assert PPOAgent(_env(5), PPOConfig(max_threads=2), seed=0).num_parameters() == 99_615
    agent = PPOAgent(_env(5), PPOConfig(max_threads=2), seed=0)
    assert type(agent.model).__name__ == "ActorCritic" and agent.model.trunk[0].in_features == 100


@pytest.mark.parametrize("observation", ["tableau", "check_matrix"])
def test_equivariant_forward_shapes_and_mask(observation):
    env = _env(6, batch=8, seed=3)
    agent = PPOAgent(env, PPOConfig(policy="equivariant", observation=observation, max_threads=2), seed=0)
    env.reset()
    h0 = env.action_table.entries.index(("H", 0, -1))
    env.step(np.full(8, h0))
    mask = agent.adapter.action_mask()
    obs = torch.from_numpy(agent.adapter.encode(env.tab))
    assert obs.shape == (8, agent.adapter.obs_dim)
    with torch.no_grad():
        logits, value = agent.model(obs, torch.from_numpy(mask))
    assert logits.shape == (8, env.num_actions) and value.shape == (8,)
    assert torch.all(logits[:, h0] <= ppo.MASK_LOGIT)
    a, logp, v = agent._policy(agent.adapter.encode(env.tab), mask)
    assert a.shape == (8,) and logp.shape == (8,) and v.shape == (8,) and not np.any(a == h0)


def test_equivariant_logits_permute_with_the_qubits():
    env = _env(6, batch=4, seed=3)
    agent = PPOAgent(env, PPOConfig(policy="equivariant", max_threads=2), seed=0)
    env.reset()
    for _ in range(4):
        env.step(env.sample_actions())
    n, tab = env.n, env.tab.copy()
    perm = np.random.default_rng(0).permutation(n)
    ptab = tab[:, :, list(perm) + list(n + perm) + [2 * n]]  # relabel the qubit columns
    with torch.no_grad():
        l1, v1 = agent.model(torch.from_numpy(agent.adapter.encode(tab)))
        l2, v2 = agent.model(torch.from_numpy(agent.adapter.encode(ptab)))
    entries = env.action_table.entries
    index = {e: i for i, e in enumerate(entries)}
    match = [index[(g, int(perm[q1]), int(perm[q2]) if q2 >= 0 else -1)] for g, q1, q2 in entries]
    assert torch.allclose(l2, l1[:, match], atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)


def test_equivariant_training_smoke():
    env = _env(5, batch=16, seed=4)
    agent = PPOAgent(env, PPOConfig(policy="equivariant", equivariant_hidden=32, rollout_steps=4, minibatch_size=32, epochs=2, max_threads=2), seed=4)
    before = [p.detach().clone() for p in agent.model.parameters()]
    res = agent.train(max_env_steps=3 * 4 * 16, stop_on_success=False)
    assert res.updates == 3 and len(res.history) == 3 and res.env_steps == 192
    assert res.config["policy"] == "equivariant" and res.config["policy_resolved"] == "equivariant"
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.model.parameters()))
    assert all(np.isfinite(h["policy_loss"]) and np.isfinite(h["value_loss"]) for h in res.history)


# ----------------------------------------------------------------------
# PPO-only 4x4 rotated-surface target (agents.ppo_regress)
# ----------------------------------------------------------------------
def test_rotated_surface_16_target_and_reference():
    from autoqec_sim import known_codes, ldpc
    from autoqec_sim.agents import ppo_regress

    t = ppo_regress.TARGETS["rotated_surface_16"]
    assert (t.n, t.k, t.d, t.graph_spec, t.gates, t.max_steps) == (16, 1, 4, "square:4x4", ("H", "CNOT"), 50)
    assert "rotated_surface_16" not in regress.TARGETS  # PPO only; the evolutionary regression does not attempt it
    assert "rotated_surface_16" in ppo_regress.REDUCED_SKIP
    assert ppo_regress.target_budget("rotated_surface_16", "full")["max_env_steps"] == 1_000_000
    assert ppo_regress.target_budget("five_qubit", "full") == ppo_regress.BUDGETS["full"]
    assert ppo.resolve_policy(PPOConfig().policy, t.n) == "equivariant"
    ref = known_codes.get("rotated_surface_16")
    assert (ref.n, ref.k) == (16, 1) and ref.is_css() and ref.distance_exact() == 4
    st = ldpc.sparsest_basis_stats(ref.to_tableau(), 1, exact_max_generators=15)
    assert (int(st["w_star"][0]), int(st["max_degree"][0])) == (4, 4)
    ver = regress.verify(ref, t)
    assert ver["parameters_match"] and ver["nontrivial"] and ver["enumerators_match_reference"]
    assert ver["min_max_generator_weight"] == 4  # exact at n - k = 15 (VERIFY_EXACT_GENERATORS)
    # the d = 3 builder reproduces the 3x3 reference group
    from autoqec_sim.code import StabilizerCode

    assert StabilizerCode.from_stabilizers(known_codes.rotated_surface(3)).same_weight_enumerators(
        known_codes.get("rotated_surface_9")
    )


def test_reduced_budget_skips_the_4x4_target():
    from autoqec_sim.agents import ppo_regress

    # reduced: every evolutionary target and nothing else; full: plus the 4x4 target
    reduced = [n for n in ppo_regress.TARGETS if n not in ppo_regress.REDUCED_SKIP]
    assert reduced == list(regress.TARGETS)
    assert list(ppo_regress.TARGETS) == [*regress.TARGETS, "rotated_surface_16"]
