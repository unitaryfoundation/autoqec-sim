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
