# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""RL scaling probe: how far the PPO agent scales in ``n``.

Three questions, each answered by a small controlled run with a fixed
budget (``experiments/scaling/runs/`` holds the outputs, ``experiments/scaling/
REPORT.md`` the write-up):

1. ``diag``: why the 4x4 setting (``n = 16``, ``k = 1``, target ``d = 4``,
   H and CNOT on the square lattice, 50-gate episodes) never produces a
   distance-4 state. The run trains the default agent on the plain level
   distance reward plus the padding penalty and logs, per update, the
   distribution of terminal distances, the distance reached at every step
   of the episode, the policy entropy, the action-type frequencies, the
   number of idle qubits, and the exact bottleneck weight ``w*`` of a
   sample of terminal states. ``evo`` runs the model-free evolutionary
   search on the same environment as a reachability control.
2. ``variant``: one change at a time on the 4x4 target: ``curriculum``
   (target distance ramped 2, 3, 4 within one run), ``actions`` (CZ and
   compound plaquette-preparation macros added to the action set),
   ``equivariant`` (per-qubit shared MLP over the qubit's tableau column
   plus a pooled context, with per-qubit and per-edge heads; now
   :func:`autoqec_sim.agents.policies.equivariant_actor_critic`),
   ``bigbatch`` (1024 parallel episodes, entropy coefficient 0.001), and
   ``long`` (the baseline with a longer episode cap).
3. ``ldpc``: the LDPC-constrained target ``[[n, k, 3]]`` with ``w_max = 4``
   in the penalty form of :mod:`autoqec_sim.ldpc` on all-to-all and on
   the square graph, ``n = 12`` to ``16``, ``k = 2`` to ``4``.

Every run writes one JSON with its configuration, its per-update history,
and its outcome. ``python -m autoqec_sim.experiments.scaling --help``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .. import graphs
from ..agents.policies import equivariant_actor_critic
from ..code import StabilizerCode
from ..env import CodeDiscoveryEnv
from ..errors import ErrorSet, paulis_of_weight
from ..ldpc import ldpc_penalty, sparsest_basis_stats
from ..reward import (
    batched_distance,
    kl_reward,
    low_weight_stabilizer_penalty,
    undetected_mask,
)
from ..symplectic import rref
from ..tableau import ActionTable, BatchTableau, apply_gate

DEFAULT_OUT = Path("experiments/scaling/runs")
EXACT_GENERATORS = 15  # exact w* up to n - k = 15 (the 4x4 target)
W_STAR_SAMPLE = 16  # terminal states per sampled update for the w* statistic
W_STAR_EVERY = 10  # updates between w* samples


# --------------------------------------------------------------------------
# the 4x4 target
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    n: int = 16
    k: int = 1
    d: int = 4
    graph_spec: str = "square:4x4"
    gates: tuple = ("H", "CNOT")
    max_steps: int = 50

    def to_dict(self):
        d = asdict(self)
        d["gates"] = list(self.gates)
        return d


TARGET_4X4 = Target()
TARGET_3X3 = Target(n=9, d=3, graph_spec="square:3x3", max_steps=30)


class LevelDistanceReward:
    """``min(d, d_t) / d_t - 1`` plus the padding penalty (minus the
    fraction of weight-1 Paulis in the stabilizer group), so that the
    reward is 0 iff the state has distance at least ``d_t`` and no idle
    qubit. ``d_t`` is mutable (the curriculum ramps it). ``last_d`` holds
    the distances of the last evaluated batch (capped at ``d_max``)."""

    def __init__(self, d_target: int, d_max: int, fractional: bool = False):
        self.d_target = int(d_target)
        self.d_max = int(d_max)
        self.fractional = bool(fractional)
        self.last_d = None
        self.last_pad = None

    def __call__(self, tab: np.ndarray, k: int) -> np.ndarray:
        d = batched_distance(tab, k, self.d_max - 1)
        pad = low_weight_stabilizer_penalty(tab, k, 1)
        r = np.minimum(d, self.d_target) / float(self.d_target) - 1.0
        if self.fractional:
            n = (tab.shape[-1] - 1) // 2
            for w in range(1, self.d_target):
                idx = np.flatnonzero(d == w)
                if idx.size:
                    frac = 1.0 - undetected_mask(tab[idx], k, paulis_of_weight(n, w)).mean(axis=1)
                    r[idx] = (w - 1 + frac) / float(self.d_target) - 1.0
        self.last_d = d
        self.last_pad = pad
        return r + pad


# --------------------------------------------------------------------------
# compound actions
# --------------------------------------------------------------------------
def plaquettes(rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    """Row-major corner indices ``(top-left, top-right, bottom-left,
    bottom-right)`` of every unit square of a ``rows x cols`` grid."""
    idx = lambda r, c: r * cols + c  # noqa: E731
    return [
        (idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1))
        for r in range(rows - 1)
        for c in range(cols - 1)
    ]


def plaquette_macros(rows: int, cols: int) -> list[tuple[str, int, list[tuple]]]:
    """Two four-gate macros per plaquette, both nearest-neighbour on the
    lattice. ``PLAQX`` prepares the X-type check ``XXXX`` on fresh qubits
    (H on the top-left corner, then CNOT fan-out along edges); ``PLAQZ``
    is the same fan-out without the H (a Z-type parity spread). These are
    the building blocks of the standard surface-code encoders."""
    out = []
    for p, (a, b, c, d) in enumerate(plaquettes(rows, cols)):
        fan = [("CNOT", a, b), ("CNOT", a, c), ("CNOT", b, d)]
        out.append(("PLAQX", p, [("H", a, -1), *fan]))
        out.append(("PLAQZ", p, fan))
    return out


class MacroActionTable:
    """An :class:`ActionTable` extended with compound actions. Entries
    ``(name, index, -1)`` of a macro expand to a gate list through
    :meth:`expand`; ``apply`` runs the base table on atomic actions and
    each macro's gates on the elements that chose it."""

    def __init__(self, base: ActionTable, macros: list[tuple[str, int, list[tuple]]]):
        self.base = base
        self.n_base = len(base)
        self.macro_gates = {self.n_base + i: list(g) for i, (_, _, g) in enumerate(macros)}
        self.entries = list(base.entries) + [(name, idx, -1) for name, idx, _ in macros]

    def __len__(self):
        return len(self.entries)

    def apply(self, tab: np.ndarray, actions, where=None) -> None:
        actions = np.asarray(actions, dtype=np.int64)
        active = np.ones(actions.shape[0], dtype=bool) if where is None else np.asarray(where, dtype=bool)
        atomic = actions < self.n_base
        self.base.apply(tab, np.where(atomic, actions, -1), where=active & atomic)
        for a in np.unique(actions[active & ~atomic]):
            sel = np.flatnonzero(active & (actions == a))
            for g, q1, q2 in self.macro_gates[int(a)]:
                apply_gate(tab, g, q1, q2 if q2 >= 0 else None, where=sel)

    def expand(self, gates) -> list[tuple]:
        """Atomic gate list of a list of table entries."""
        out = []
        lookup = {tuple(e): i for i, e in enumerate(self.entries)}
        for g in gates:
            i = lookup[tuple(g)]
            out += self.macro_gates[i] if i >= self.n_base else [tuple(g)]
        return out


def expand_gates(env: CodeDiscoveryEnv, gates) -> list[tuple]:
    table = env.action_table
    return table.expand(gates) if isinstance(table, MacroActionTable) else [tuple(g) for g in gates]


# --------------------------------------------------------------------------
# diagnostic environment
# --------------------------------------------------------------------------
class DiagEnv(CodeDiscoveryEnv):
    """``CodeDiscoveryEnv`` that accumulates, between ``flush`` calls, the
    distribution of terminal distances, the distances reached at every
    step of the episode, the action-type frequencies, the number of idle
    qubits of terminal states, and the first environment step at which a
    state of the target distance appeared (with or without idle qubits)."""

    def __init__(self, *args, d_max: int, **kw):
        super().__init__(*args, **kw)
        self.d_max = int(d_max)
        self.env_steps = 0
        self.first_d_max_any = None
        self.first_d_max_clean = None
        self.terminal_samples: list = []
        self.set_table(self.action_table)

    def set_table(self, table):
        """Install ``table`` as the action table and rebuild the kind index."""
        self.action_table = table
        self.action_kinds = sorted({e[0] for e in table.entries})
        self.kind_index = np.array([self.action_kinds.index(e[0]) for e in table.entries], dtype=np.int64)
        self.flush()

    def flush(self):
        rec = getattr(self, "_acc", None)
        self._acc = dict(
            terminal_d=np.zeros(self.d_max + 1, dtype=np.int64),
            terminal_idle=np.zeros(self.n + 1, dtype=np.int64),
            terminal_steps=np.zeros(self.max_steps + 1, dtype=np.int64),
            live_d=np.zeros((self.max_steps + 1, self.d_max + 1), dtype=np.int64),
            actions=np.zeros(len(self.action_kinds), dtype=np.int64),
            n_terminal=0,
        )
        self.terminal_samples = []
        return rec

    def reset(self, seed=None, where=None):
        # a partial reset (auto-reset of finished episodes inside ``step``)
        # re-evaluates the reward on the reset elements only; keep the
        # full-batch distances of the step for the statistics below
        if where is None:
            return super().reset(seed, where)
        keep = (self.reward_fn.last_d, self.reward_fn.last_pad)
        out = super().reset(seed, where)
        self.reward_fn.last_d, self.reward_fn.last_pad = keep
        return out

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        active = self.steps < self.max_steps
        obs, reward, terminated, truncated, info = super().step(actions)
        self.env_steps += self.batch_size
        acc = self._acc
        chosen = actions[active]
        chosen = chosen[chosen >= 0]
        acc["actions"] += np.bincount(self.kind_index[chosen], minlength=len(self.action_kinds))
        d = self.reward_fn.last_d
        pad = self.reward_fn.last_pad
        if d is not None:
            reached = d >= self.d_max
            if reached.any():
                if self.first_d_max_any is None:
                    self.first_d_max_any = self.env_steps
                if self.first_d_max_clean is None and (reached & (pad == 0.0)).any():
                    self.first_d_max_clean = self.env_steps
            done = terminated | truncated
            step_idx = info["steps"]  # steps after this action
            np.add.at(acc["live_d"], (np.minimum(step_idx, self.max_steps), np.minimum(d, self.d_max)), 1)
            if done.any():
                dd = np.minimum(d[done], self.d_max)
                acc["terminal_d"] += np.bincount(dd, minlength=self.d_max + 1)
                idle = np.rint(-pad[done] * 3 * self.n).astype(int)  # weight-1 Paulis in S
                acc["terminal_idle"] += np.bincount(np.minimum(idle, self.n), minlength=self.n + 1)
                acc["terminal_steps"] += np.bincount(np.minimum(info["final_steps"], self.max_steps), minlength=self.max_steps + 1)
                acc["n_terminal"] += int(done.sum())
                if "final_tableau" in info and len(self.terminal_samples) < 4 * W_STAR_SAMPLE:
                    best = dd.max()
                    sel = np.flatnonzero(dd == best)[:W_STAR_SAMPLE]
                    self.terminal_samples.append((int(best), info["final_tableau"][sel].copy()))
        return obs, reward, terminated, truncated, info


def w_star_sample(env: DiagEnv) -> dict | None:
    """Exact ``w*``, maximum qubit degree, and generator-row weights of the
    sampled terminal states of largest distance."""
    if not env.terminal_samples:
        return None
    best = max(b for b, _ in env.terminal_samples)
    tabs = np.concatenate([t for b, t in env.terminal_samples if b == best])[:W_STAR_SAMPLE]
    st = sparsest_basis_stats(tabs, env.k, exact_max_generators=EXACT_GENERATORS)
    n, k = env.n, env.k
    rows = tabs[:, n + k :, : 2 * n]
    wts = (rows[:, :, :n] | rows[:, :, n:]).sum(axis=2)
    return {
        "d": best,
        "count": int(tabs.shape[0]),
        "w_star_mean": float(st["w_star"].mean()),
        "w_star_min": int(st["w_star"].min()),
        "max_degree_mean": float(st["max_degree"].mean()),
        "row_weight_mean": float(wts.mean()),
        "row_weight_max": int(wts.max()),
    }


# --------------------------------------------------------------------------
# equivariant policy: now ``autoqec_sim.agents.policies.equivariant_actor_critic``
# (``PPOConfig(policy="equivariant")``). The runs below still build the agent
# with ``policy="mlp"`` and swap the network in afterwards, exactly as the
# recorded runs in experiments/scaling/runs/ did, so that their torch RNG stream (MLP
# initialization first, then the equivariant one) and hence their trajectories
# are reproducible.
# --------------------------------------------------------------------------
def swap_in_equivariant(agent, config) -> None:
    """Replace ``agent.model`` by the equivariant network (recorded-run protocol)."""
    agent.policy = "equivariant"
    agent.model = equivariant_actor_critic(agent.env.n, agent.env.action_table.entries)
    agent.optimizer = agent.torch.optim.Adam(agent.model.parameters(), lr=config.learning_rate, eps=1e-5)


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------
VARIANTS = ("baseline", "curriculum", "actions", "equivariant", "bigbatch", "long", "fractional")


def make_env(target: Target, seed: int, batch: int, variant: str, d_start: int | None = None) -> DiagEnv:
    graph = graphs.from_spec(target.graph_spec)
    gates = target.gates + (("CZ",) if variant == "actions" else ())
    max_steps = target.max_steps
    if variant == "long":
        max_steps = 2 * target.max_steps
    reward = LevelDistanceReward(d_start or target.d, target.d, fractional=(variant == "fractional"))
    env = DiagEnv(
        n=target.n, k=target.k, graph=graph, batch_size=batch, max_steps=max_steps, gates=gates,
        reward_fn=reward, seed=seed, auto_reset=True, d_max=target.d,
    )
    if variant == "actions":
        rows, cols = (int(x) for x in target.graph_spec.split(":")[1].split("x"))
        env.set_table(MacroActionTable(env.action_table, plaquette_macros(rows, cols)))
    env.reset()
    return env


def make_agent(env: DiagEnv, variant: str, seed: int, threads: int):
    from ..agents.ppo import PPOAgent, PPOConfig

    kw = dict(max_threads=threads, policy="mlp")
    if variant == "bigbatch":
        kw["ent_coef"] = 0.001
    config = PPOConfig(**kw)
    agent = PPOAgent(env, config=config, seed=seed)
    if variant == "equivariant":
        swap_in_equivariant(agent, config)
    return agent


def describe_code(tab: np.ndarray, k: int) -> dict:
    code = StabilizerCode.from_tableau(tab[None], k)
    st = sparsest_basis_stats(tab[None], k, exact_max_generators=EXACT_GENERATORS)
    return {
        "stabilizers": code.stabilizer_strings(),
        "d": int(code.distance_exact()),
        "is_css": bool(code.is_css()),
        "w_star": int(st["w_star"][0]),
        "max_degree": int(st["max_degree"][0]),
    }


def run_variant(
    variant: str,
    seed: int,
    minutes: float,
    threads: int = 2,
    target: Target = TARGET_4X4,
    batch: int | None = None,
    max_env_steps: int = 10**9,
    verbose: bool = False,
    curriculum_rate: float = 0.5,
    curriculum_window: int = 10,
    stop_on_success: bool = True,
) -> dict:
    """One PPO run of ``variant`` on ``target``; stops at the first clean
    target-distance state (``stop_on_success``), the wall cap, or the step
    cap."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    batch = batch or (1024 if variant == "bigbatch" else 256)
    d_start = 2 if variant == "curriculum" else None
    env = make_env(target, seed, batch, variant, d_start)
    agent = make_agent(env, variant, seed, threads)
    reward: LevelDistanceReward = env.reward_fn
    t0 = time.perf_counter()
    history = []
    stages = [{"d_target": reward.d_target, "env_steps": 0, "update": 0}]
    recent = []

    def cb(agent, new):
        ws = w_star_sample(env) if agent.updates % W_STAR_EVERY == 0 else None
        acc = env.flush()
        rec = agent.history[-1]
        nt = max(acc["n_terminal"], 1)
        rec["terminal_d"] = acc["terminal_d"].tolist()
        rec["terminal_d_frac"] = (acc["terminal_d"] / nt).round(4).tolist()
        rec["terminal_idle_mean"] = float((acc["terminal_idle"] * np.arange(env.n + 1)).sum() / nt)
        rec["terminal_steps_mean"] = float((acc["terminal_steps"] * np.arange(env.max_steps + 1)).sum() / nt)
        rec["action_frac"] = dict(zip(env.action_kinds, (acc["actions"] / max(acc["actions"].sum(), 1)).round(4).tolist()))
        live = acc["live_d"]
        # first within-episode step at which half of the live states have d >= 2, 3
        rec["step_half_d2"] = _first_step_with_frac(live, 2, 0.5)
        rec["step_half_d3"] = _first_step_with_frac(live, 3, 0.5)
        rec["live_d3_frac_last_step"] = float(live[env.max_steps, 3:].sum() / max(live[env.max_steps].sum(), 1))
        rec["d_target"] = reward.d_target
        rec["first_d_max_any"] = env.first_d_max_any
        rec["first_d_max_clean"] = env.first_d_max_clean
        if ws is not None:
            rec["w_star_sample"] = ws
        history.append(rec)
        if verbose:
            print(
                f"upd {rec['update']:5d} steps {rec['env_steps']:9d} r {rec['mean_reward']:+.3f} "
                f"ent {rec['entropy']:.3f} d_t {reward.d_target} term_d {rec['terminal_d']} "
                f"idle {rec['terminal_idle_mean']:.1f} succ {rec['success_rate']:.3f} t {rec['wall_time_s']:.0f}s",
                flush=True,
            )
        # curriculum: advance when the stage target is reached reliably
        if variant == "curriculum" and reward.d_target < target.d:
            recent.append(rec["success_rate"])
            if len(recent) >= curriculum_window and np.mean(recent[-curriculum_window:]) >= curriculum_rate:
                reward.d_target += 1
                recent.clear()
                agent.successes.clear()
                stages.append({"d_target": reward.d_target, "env_steps": agent.env_steps, "update": agent.updates})
                if verbose:
                    print(f"    curriculum: target distance -> {reward.d_target}", flush=True)
            return False
        return bool(stop_on_success and reward.d_target >= target.d and new)

    agent.train(max_env_steps, max_time_s=minutes * 60.0, stop_on_success=False, callback=cb, verbose=False)
    wall = time.perf_counter() - t0
    first = None
    if reward.d_target >= target.d and agent.successes:
        fc = agent.successes[0]
        first = fc.to_dict()
        first["gates"] = [list(g) for g in expand_gates(env, fc.gates)]
        first["atomic_gate_count"] = len(first["gates"])
        code = replay_gates(first["gates"], target.n, target.k)
        first["code"] = describe_code(code, target.k)
    return {
        "kind": "variant",
        "variant": variant,
        "target": target.to_dict(),
        "seed": seed,
        "threads": threads,
        "batch": batch,
        "num_actions": env.num_actions,
        "action_kinds": env.action_kinds,
        "max_steps": env.max_steps,
        "ppo_config": agent.config.to_dict(),
        "num_parameters": agent.num_parameters(),
        "minutes_cap": minutes,
        "wall_time_s": round(wall, 1),
        "env_steps": agent.env_steps,
        "episodes": agent.episodes,
        "updates": agent.updates,
        "reached_d_target": first is not None,
        "first_success": first,
        "first_d_max_any_env_steps": env.first_d_max_any,
        "first_d_max_clean_env_steps": env.first_d_max_clean,
        "curriculum_stages": stages,
        "final": history[-1] if history else None,
        "platform": platform.platform(),
        "history": history,
    }


def _first_step_with_frac(live: np.ndarray, d: int, frac: float):
    tot = live.sum(axis=1)
    ok = np.flatnonzero((tot > 0) & (live[:, d:].sum(axis=1) >= frac * np.maximum(tot, 1)))
    return int(ok[0]) if ok.size else None


def replay_gates(gates, n: int, k: int) -> np.ndarray:
    bt = BatchTableau(n, 1)
    for g, a, b in gates:
        bt.apply(g, a, b if b >= 0 else None)
    return bt.array[0]


# --------------------------------------------------------------------------
# evolutionary control
# --------------------------------------------------------------------------
def run_evo(seed: int, minutes: float, target: Target = TARGET_4X4, max_steps: int | None = None, batch: int = 512, verbose=False) -> dict:
    """Model-free evolutionary search on the 4x4 target with the same
    reward: does any distance-4 state exist within the episode length?"""
    from ..search import EvolutionarySearch

    max_steps = max_steps or target.max_steps
    reward = LevelDistanceReward(target.d, target.d)
    env = CodeDiscoveryEnv(
        n=target.n, k=target.k, graph=graphs.from_spec(target.graph_spec), batch_size=batch,
        max_steps=max_steps, gates=target.gates, reward_fn=reward, seed=seed,
    )
    search = EvolutionarySearch(env, seed=seed)
    t0 = time.perf_counter()
    best_d = 0
    evaluations = 0
    generations = 0
    trace = []
    pop = search.random_population(batch)
    best_reward = -np.inf
    best_seq = None
    while time.perf_counter() - t0 < minutes * 60.0:
        rew, succ, used = env.rollout(pop)
        evaluations += batch
        generations += 1
        d = reward.last_d
        best_d = max(best_d, int(d.max()))
        i = int(np.argmax(rew))
        if rew[i] > best_reward:
            best_reward = float(rew[i])
            best_seq = pop[i, : used[i]].copy()  # the prefix that produced the state
        if generations % 20 == 0 or succ.any():
            trace.append({"generation": generations, "evaluations": evaluations, "best_reward": best_reward, "best_d": best_d, "d_hist": np.bincount(np.minimum(d, target.d), minlength=target.d + 1).tolist(), "wall_time_s": round(time.perf_counter() - t0, 1)})
            if verbose:
                print(trace[-1], flush=True)
        if succ.any():
            break
        pop = search.next_population(pop, rew)
    out = {
        "kind": "evo",
        "target": target.to_dict(),
        "max_steps": max_steps,
        "seed": seed,
        "batch": batch,
        "minutes_cap": minutes,
        "wall_time_s": round(time.perf_counter() - t0, 1),
        "evaluations": evaluations,
        "generations": generations,
        "best_reward": best_reward,
        "best_d": best_d,
        "reached_d_target": bool(best_reward >= 0.0),
        "trace": trace,
    }
    if best_seq is not None:
        gates = [env.action_table.entries[a] for a in best_seq if a >= 0]
        tab = replay_gates(gates, target.n, target.k)
        out["best_code"] = describe_code(tab, target.k)
        out["best_gates"] = [list(g) for g in gates]
    return out


def run_random(caps=(50, 60, 70, 80, 100), circuits: int = 8192, seed: int = 0, target: Target = TARGET_4X4) -> dict:
    """Terminal-distance histogram of uniformly random gate sequences of
    each length on the target (the reachability of the uniform policy)."""
    graph = graphs.from_spec(target.graph_spec)
    reward = LevelDistanceReward(target.d, target.d)
    out = {"kind": "random", "target": target.to_dict(), "circuits": circuits, "seed": seed, "caps": {}}
    for cap in caps:
        env = CodeDiscoveryEnv(
            n=target.n, k=target.k, graph=graph, batch_size=circuits, max_steps=cap, gates=target.gates,
            reward_fn=reward, seed=seed,
        )
        t0 = time.perf_counter()
        for _ in range(cap):
            env.step(env.sample_actions())
        d = reward.last_d
        clean = reward.last_pad == 0.0
        out["caps"][str(cap)] = {
            "d_hist": np.bincount(np.minimum(d, target.d), minlength=target.d + 1).tolist(),
            "d_hist_clean": np.bincount(np.minimum(d[clean], target.d), minlength=target.d + 1).tolist(),
            "idle_mean": float(np.rint(-reward.last_pad * 3 * target.n).mean()),
            "wall_time_s": round(time.perf_counter() - t0, 1),
        }
    return out


def random_table(recs: list[dict]) -> str:
    lines = [
        "| episode length | circuits | d = 1 | d = 2 | d = 3 | d = 4 | d = 4 with no idle qubit | mean idle qubits | file |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        if r.get("kind") != "random":
            continue
        for cap, c in r["caps"].items():
            h = c["d_hist"]
            lines.append(f"| {cap} | {r['circuits']:,} | {h[1]:,} | {h[2]:,} | {h[3]:,} | {h[4]:,} | {c['d_hist_clean'][4]:,} | {c['idle_mean']:.1f} | `{Path(r['_file']).name}` |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# LDPC scaling
# --------------------------------------------------------------------------
class LdpcReward:
    """Regression reward (Knill-Laflamme fraction of detected errors of
    weight ``< d`` plus the padding penalty) plus the LDPC penalty of
    :func:`autoqec_sim.ldpc.ldpc_penalty` in two tiers. On every state the
    penalty uses the greedy pairwise-reduction bound on ``w*`` (the
    ``ldpc.py`` fallback, 2 to 6 ms per batch of 256 at these sizes; equal
    to the exact ``w*`` on 95 to 99 percent of random states), which shapes
    the trajectory toward sparse groups as the every-state penalty of the
    regression does. On candidate states (every error detected, no idle
    qubit) the exact ``w*`` replaces it (up to 0.7 s per batch of 256 at
    ``n - k = 14``, affordable only there), so the reward is 0 iff the state
    is a distance-``d`` code with no idle qubit and exact ``w* <= w_max``."""

    form = "greedy bound on every state, exact w* on candidate states"

    def __init__(self, n: int, d: int, w_max: int, scale: float = 0.25, deg_max: int | None = None):
        self.errors = ErrorSet.up_to_weight(n, d - 1)
        self.w_max = int(w_max)
        self.scale = float(scale)
        self.deg_max = deg_max
        self.ldpc_evals = 0
        self.last_candidates = None

    def __call__(self, tab: np.ndarray, k: int) -> np.ndarray:
        base = kl_reward(tab, k, self.errors) + low_weight_stabilizer_penalty(tab, k, 1)
        r = base + ldpc_penalty(tab, k, self.w_max, self.deg_max, self.scale, exact_max_generators=0)
        cand = base >= 0.0
        self.last_candidates = cand
        if cand.any():
            idx = np.flatnonzero(cand)
            r[idx] = ldpc_penalty(tab[idx], k, self.w_max, self.deg_max, self.scale, exact_max_generators=EXACT_GENERATORS)
            self.ldpc_evals += int(idx.size)
        return r


def probe_policy(agent, env: CodeDiscoveryEnv, reward: LdpcReward, d: int) -> dict:
    """What the trained policy produces: one greedy episode and one batch
    of sampled episodes from a fresh reset, without masks or auto-reset,
    described by the distance (capped at ``d``), idle qubits, exact ``w*``,
    and the reward terms of the final states."""
    n, k = env.n, env.k
    out = {}
    for mode in ("greedy", "sampled"):
        probe = CodeDiscoveryEnv(
            n=n, k=k, graph=env.graph, batch_size=env.batch_size, max_steps=env.max_steps, gates=env.gates,
            reward_fn=lambda tab, k: np.zeros(tab.shape[0]), seed=1234,
        )
        probe.reset()
        for _ in range(env.max_steps):
            probe.step(agent.act(probe.tab, greedy=(mode == "greedy")))
        tab = probe.tab
        dist = batched_distance(tab, k, d - 1)
        pad = low_weight_stabilizer_penalty(tab, k, 1)
        kl = kl_reward(tab, k, reward.errors)
        st = sparsest_basis_stats(tab, k, exact_max_generators=EXACT_GENERATORS)
        rows = tab[:, n + k :, : 2 * n]
        wts = (rows[:, :, :n] | rows[:, :, n:]).sum(axis=2)
        out[mode] = {
            "d_hist": np.bincount(np.minimum(dist, d), minlength=d + 1).tolist(),
            "idle_mean": float(np.rint(-pad * 3 * n).mean()),
            "kl_mean": float(kl.mean()),
            "w_star_hist": np.bincount(st["w_star"], minlength=n + 1).tolist(),
            "w_star_at_target_d": np.bincount(st["w_star"][dist >= d], minlength=n + 1).tolist(),
            "row_weight_mean": float(wts.mean()),
            "clean_target_states": int(((dist >= d) & (pad == 0.0)).sum()),
            "feasible_clean_target_states": int(((dist >= d) & (pad == 0.0) & (st["w_star"] <= reward.w_max)).sum()),
        }
        if mode == "greedy":
            out[mode]["code"] = describe_code(tab[0], k)
            out[mode]["gates"] = [list(g) for g in probe.gate_lists()[0]]
    return out


def run_ldpc(
    n: int,
    k: int,
    graph_spec: str,
    seed: int,
    minutes: float,
    threads: int = 2,
    d: int = 3,
    w_max: int = 4,
    max_steps: int = 40,
    batch: int = 256,
    gates=("H", "S", "CNOT"),
    post_success_updates: int = 25,
    verbose: bool = False,
    scale: float = 0.25,
    policy: str = "mlp",
) -> dict:
    from ..agents.ppo import PPOAgent, PPOConfig

    graph = graphs.from_spec(graph_spec)
    gates = tuple(gates)
    reward = LdpcReward(n, d, w_max, scale=scale)
    env = CodeDiscoveryEnv(
        n=n, k=k, graph=graph, batch_size=batch, max_steps=max_steps, gates=gates, reward_fn=reward, seed=seed,
        auto_reset=True,
    )
    config = PPOConfig(max_threads=threads, policy="mlp")
    agent = PPOAgent(env, config=config, seed=seed)
    if policy == "equivariant":
        swap_in_equivariant(agent, config)
    elif policy != "mlp":
        raise ValueError("policy must be mlp or equivariant")
    t0 = time.perf_counter()
    state = {"first_update": None, "candidate_frac": []}

    def cb(agent, new):
        rec = agent.history[-1]
        rec["ldpc_evals"] = reward.ldpc_evals
        if verbose:
            print(
                f"upd {rec['update']:5d} steps {rec['env_steps']:9d} r {rec['mean_reward']:+.3f} "
                f"ent {rec['entropy']:.3f} succ {rec['success_rate']:.3f} ldpc_evals {reward.ldpc_evals} t {rec['wall_time_s']:.0f}s",
                flush=True,
            )
        if new and state["first_update"] is None:
            state["first_update"] = agent.updates
        return state["first_update"] is not None and agent.updates >= state["first_update"] + post_success_updates

    agent.train(10**9, max_time_s=minutes * 60.0, stop_on_success=False, callback=cb, verbose=False)
    wall = time.perf_counter() - t0
    first = None
    distinct = {}
    for fc in agent.successes:
        key = rref(fc.tableau[n + k :, : 2 * n])[0].tobytes()
        distinct[key] = distinct.get(key, 0) + 1
    if agent.successes:
        fc = agent.successes[0]
        first = fc.to_dict()
        first["code"] = describe_code(fc.tableau, k)
    tail = agent.history[-post_success_updates:] if first else []
    probe = probe_policy(agent, env, reward, d)
    return {
        "kind": "ldpc",
        "final_policy_probe": probe,
        "penalty_form": LdpcReward.form,
        "ldpc_scale": scale,
        "policy": policy,
        "n": n,
        "k": k,
        "d": d,
        "w_max": w_max,
        "graph": graph_spec,
        "gates": list(gates),
        "max_steps": max_steps,
        "batch": batch,
        "seed": seed,
        "threads": threads,
        "ppo_config": agent.config.to_dict(),
        "num_parameters": agent.num_parameters(),
        "minutes_cap": minutes,
        "wall_time_s": round(wall, 1),
        "env_steps": agent.env_steps,
        "episodes": agent.episodes,
        "updates": agent.updates,
        "success": first is not None,
        "first_success": first,
        "first_success_wall_s": None if first is None else next((h["wall_time_s"] for h in agent.history if h["env_steps"] >= first["env_steps"]), None),
        "successes": len(agent.successes),
        "distinct_codes": len(distinct),
        "post_success_rate": float(np.mean([h["success_rate"] for h in tail])) if tail else None,
        "final": agent.history[-1] if agent.history else None,
        "ldpc_evals": reward.ldpc_evals,
        "platform": platform.platform(),
        "history": agent.history,
    }


# --------------------------------------------------------------------------
# report tables
# --------------------------------------------------------------------------
def load_all(out_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(out_dir.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        r["_file"] = str(p)
        recs.append(r)
    return recs


def _fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def terminal_fractions(history: list[dict], window: int = 50) -> list[float]:
    """Fractions of terminal states at each distance, summed over the last
    ``window`` updates (episodes end synchronously, so single updates can
    have no terminal state)."""
    counts = np.sum([h["terminal_d"] for h in history[-window:]], axis=0)
    tot = counts.sum()
    return (counts / max(tot, 1)).round(4).tolist()


def variant_table(recs: list[dict]) -> str:
    lines = [
        "| variant | seed | actions | episode cap | parameters | env steps | wall (s) | max terminal d | first d = 4 state (env steps) | first clean d = 4 (env steps) | final entropy | terminal d fractions over the last 50 updates (1, 2, 3, 4) | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        if r.get("kind") != "variant":
            continue
        fin = r["final"] or {}
        maxd = max((i for h in r["history"] for i, c in enumerate(h["terminal_d"]) if c > 0), default=0)
        lines.append(
            f"| {r['variant']} | {r['seed']} | {r['num_actions']} | {r['max_steps']} | {r['num_parameters']:,} | {r['env_steps']:,} | {r['wall_time_s']:.0f} "
            f"| {maxd} | {_fmt(r['first_d_max_any_env_steps'])} | {_fmt(r['first_d_max_clean_env_steps'])} "
            f"| {_fmt(fin.get('entropy'))} | {terminal_fractions(r['history'])[1:]} | `{Path(r['_file']).name}` |"
        )
    return "\n".join(lines)


def ldpc_table(recs: list[dict]) -> str:
    lines = [
        "| n | k | graph | policy | penalty scale | seed | success | env steps to first success | wall to first success (s) | post-success success rate | distinct codes | first code w*, deg, CSS, gates | final policy, 256 sampled episodes: at d = 3, mode of w*, mean idle qubits, percent of errors undetected | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted((r for r in recs if r.get("kind") == "ldpc"), key=lambda r: (r["n"], r["k"], r["graph"], r.get("policy", "mlp"), -r.get("ldpc_scale", 0.25), r["seed"])):
        fs = r["first_success"]
        code = fs["code"] if fs else None
        first = _fmt(fs["env_steps"]) if fs else f"not reached in {r['env_steps']:,}"
        probe = r.get("final_policy_probe", {}).get("sampled")
        if probe:
            ws = probe["w_star_hist"]
            ptxt = f"{sum(probe['d_hist'][r['d']:])}, {max(range(len(ws)), key=ws.__getitem__)}, {probe['idle_mean']:.1f}, {-100 * probe['kl_mean']:.1f}"
        else:
            ptxt = "n/a"
        form = "final-state penalty only" if "finalonly" in r["_file"] else f"{r.get('ldpc_scale', 0.25):g}"
        lines.append(
            f"| {r['n']} | {r['k']} | {r['graph']} | {r.get('policy', 'mlp')} | {form} | {r['seed']} | {'yes' if r['success'] else 'no'} "
            f"| {first} | {_fmt(r['first_success_wall_s'], 0)} "
            f"| {_fmt(r['post_success_rate'], 3)} | {r['distinct_codes']} "
            f"| {(str(code['w_star']) + ', ' + str(code['max_degree']) + ', ' + ('yes' if code['is_css'] else 'no') + ', ' + str(fs['circuit_length'])) if code else 'n/a'} | {ptxt} | `{Path(r['_file']).name}` |"
        )
    return "\n".join(lines)


def _decile(history, i, key=None):
    n = len(history)
    return history[i * n // 10 : (i + 1) * n // 10]


def diag_table(recs: list[dict]) -> str:
    """Per-run diagnosis numbers of the baseline (and curriculum) runs."""
    lines = [
        "| run | seed | episodes | terminal d counts (1, 2, 3, 4) | d >= 2 fraction, first decile | d >= 2 fraction, last decile | entropy start, min, end | idle qubits at termination, first, last decile | CNOT fraction of actions, last decile | mean step reward, first, last decile | w* of sampled terminal states (d, mean w*, mean row weight) | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        if r.get("kind") != "variant" or r["variant"] not in ("baseline", "curriculum") or r["max_steps"] != 50 or r["target"]["n"] != 16:
            continue
        H = r["history"]
        if len(H) < 20:
            continue
        tot = np.sum([h["terminal_d"] for h in H], axis=0)
        def frac2(hs):
            if not hs:
                return float("nan")
            c = np.sum([h["terminal_d"] for h in hs], axis=0)
            return c[2:].sum() / max(c.sum(), 1)
        def idle(hs):
            v = [h["terminal_idle_mean"] for h in hs if sum(h["terminal_d"]) > 0]
            return np.mean(v) if v else float("nan")
        ent = [h["entropy"] for h in H]
        ws = [h["w_star_sample"] for h in H if h.get("w_star_sample")]
        wtxt = "n/a"
        if ws:
            late = ws[-20:]
            wtxt = f"d = {max(w['d'] for w in late)}, {np.mean([w['w_star_mean'] for w in late]):.1f}, {np.mean([w['row_weight_mean'] for w in late]):.1f} ({sum(w['count'] for w in late)} states)"
        name = r["variant"] + (" (w* rerun, 5 min)" if "wstar" in r["_file"] else "")
        lines.append(
            f"| {name} | {r['seed']} | {r['episodes']:,} | {tot[1:].tolist()} | {frac2(_decile(H, 0)):.4f} | {frac2(_decile(H, 9)):.4f} "
            f"| {ent[0]:.2f}, {min(ent):.2f}, {ent[-1]:.2f} | {idle(_decile(H, 0)):.1f}, {idle(_decile(H, 9)):.1f} "
            f"| {np.mean([h['action_frac'].get('CNOT', 0) for h in _decile(H, 9)]):.2f} "
            f"| {np.mean([h['mean_reward'] for h in _decile(H, 0)]):.3f}, {np.mean([h['mean_reward'] for h in _decile(H, 9)]):.3f} | {wtxt} | `{Path(r['_file']).name}` |"
        )
    return "\n".join(lines)


def evo_table(recs: list[dict]) -> str:
    lines = [
        "| episode cap | seed | evaluations (circuits) | generations | wall (s) | best d | reached d = 4 | file |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        if r.get("kind") != "evo":
            continue
        lines.append(
            f"| {r['max_steps']} | {r['seed']} | {r['evaluations']:,} | {r['generations']} | {r['wall_time_s']:.0f} | {r['best_d']} | {'yes' if r['reached_d_target'] else 'no'} | `{Path(r['_file']).name}` |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figure_diag(recs: list[dict], stem: Path):
    """Diagnosis figure for the baseline runs: (a) fraction of terminal
    states at each distance against env steps (25-update window; solid
    seed 0, dashed seed 1), (b) policy entropy (trailing mean over 25
    updates), one color per seed."""
    from . import figstyle

    plt = figstyle.apply()
    base = sorted(
        (r for r in recs if r.get("kind") == "variant" and r["variant"] == "baseline" and r["max_steps"] == 50 and "wstar" not in r["_file"]),
        key=lambda r: r["seed"],
    )
    if not base:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE))
    styles = ["-", "--", ":"]
    xmax = max(r["env_steps"] for r in base) / 1e6
    for j, r in enumerate(base):
        steps = np.array([h["env_steps"] for h in r["history"]]) / 1e6
        counts = np.array([h["terminal_d"] for h in r["history"]], dtype=float)
        win = 25
        csum = np.cumsum(np.vstack([np.zeros((1, counts.shape[1])), counts]), axis=0)
        lo = np.maximum(np.arange(len(counts)) + 1 - win, 0)
        window = csum[1:] - csum[lo]
        frac = window / np.maximum(window.sum(axis=1, keepdims=True), 1)
        for d in range(1, frac.shape[1]):
            axes[0].plot(steps, frac[:, d], styles[j % 3], color=figstyle.DISTANCE_COLORS[d], label=f"d = {d}" if j == 0 else None)
        axes[1].plot(steps, _smooth([h["entropy"] for h in r["history"]]), "-", color=figstyle.SERIES_COLORS[j], label=f"seed {r['seed']}")
    axes[0].set_xlabel("env steps (millions)")
    axes[0].set_ylabel("fraction of terminal states")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlim(0, xmax * 1.35)
    axes[0].legend(loc="center right")
    axes[1].set_xlabel("env steps (millions)")
    axes[1].set_ylabel("policy entropy (nats)")
    axes[1].set_xlim(0, xmax * 1.05)
    axes[1].axhline(np.log(base[0]["num_actions"]), color="#bbbbbb", lw=figstyle.THIN, ls=":")
    axes[1].legend(loc="upper right")
    figstyle.panel_label(axes[0], "a")
    figstyle.panel_label(axes[1], "b")
    fig.tight_layout()
    return figstyle.save(fig, stem)


def _smooth(x, w=25):
    x = np.asarray(x, dtype=float)
    if x.size < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = x.copy()
    out[w - 1 :] = (c[w:] - c[:-w]) / w
    for i in range(min(w - 1, x.size)):
        out[i] = x[: i + 1].mean()
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
FIGURE = Path("docs/assets/figures/scaling")


def _graph_label(spec: str) -> str:
    kind, _, size = spec.partition(":")
    return "all-to-all" if kind == "all_to_all" else f"{kind} {size}"


def _step_axis(ax, ticks):
    """Log axis of env steps with explicit k/M tick labels."""
    from matplotlib.ticker import NullFormatter

    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1e6:g}M" if t >= 1e6 else f"{t / 1e3:g}k" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def make_figure(runs_dir: Path, out_stem: Path | str = FIGURE) -> Path | None:
    """Headline figure. (a) Env steps to the first clean distance-4 state on
    the 4x4 target at the 50-gate cap for the equivariant policy, the flat
    MLP with 1024 parallel episodes and entropy coefficient 0.001, and the
    default flat MLP, one marker per seed; a run without a success is an
    open right-pointing marker at the steps it used before its wall cap.
    (b) The LDPC cells at n = 12 to 16: the steps every run used, none of
    which reached a distance-3 code with w* <= 4. Writes ``out_stem.png``
    and ``out_stem.pdf``; returns the PNG path, or ``None`` without
    matplotlib."""
    try:
        from matplotlib.lines import Line2D

        from . import figstyle
    except ImportError:  # pragma: no cover
        return None
    recs = load_all(Path(runs_dir))
    plt = figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE + 0.3))
    color = {"equivariant": figstyle.SERIES_COLORS[0], "mlp": figstyle.SERIES_COLORS[1]}

    # (a) the 4x4 target at the 50-gate cap, the three variants of the recommendation
    ax = axes[0]
    variants = [
        r for r in recs
        if r.get("kind") == "variant" and r["target"]["n"] == 16 and r["max_steps"] == 50
        and "nostop" not in r["_file"] and "wstar" not in r["_file"]
    ]
    rows = [
        ("equivariant policy", "equivariant", [r for r in variants if r["variant"] == "equivariant"]),
        ("flat MLP, 1024 episodes,\nentropy 0.001", "mlp", [r for r in variants if r["variant"] == "bigbatch"]),
        ("flat MLP, default", "mlp", [r for r in variants if r["variant"] == "baseline"]),
    ]
    for i, (_, pol, rs) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, r in enumerate(sorted(rs, key=lambda r: r["seed"])):
            yj = y + 0.15 * (j - (len(rs) - 1) / 2)
            if r["first_d_max_clean_env_steps"] is not None:
                ax.plot([r["first_d_max_clean_env_steps"]], [yj], "o", color=color[pol])
            else:
                ax.plot([r["env_steps"]], [yj], ">", color=color[pol], mfc="none")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    _step_axis(ax, [1e5, 3e5, 1e6, 3e6, 1e7, 3e7])
    ax.set_xlim(8e4, 6e7)
    ax.set_xlabel("env steps to first clean d = 4 (state evaluations)")
    ax.set_title("[[16,1,4]] on square 4x4, 50-gate episodes")

    # (b) the LDPC cells, every run censored at the steps it used
    ax = axes[1]
    ldpc = [r for r in recs if r.get("kind") == "ldpc"]
    cells = sorted({(r["n"], r["graph"]) for r in ldpc}, key=lambda c: (c[0], not c[1].startswith("all_to_all")))
    for i, cell in enumerate(cells):
        y = len(cells) - 1 - i
        rs = sorted((r for r in ldpc if (r["n"], r["graph"]) == cell), key=lambda r: (r.get("policy", "mlp"), r["seed"]))
        for j, r in enumerate(rs):
            yj = y + 0.5 * ((j + 0.5) / len(rs) - 0.5)
            pol = r.get("policy", "mlp")
            if r["first_success"]:
                ax.plot([r["first_success"]["env_steps"]], [yj], "o", color=color[pol])
            else:
                ax.plot([r["env_steps"]], [yj], ">", color=color[pol], mfc="none")
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"n = {n}, {_graph_label(g)}" for n, g in reversed(cells)])
    ax.set_ylim(-0.6, len(cells) - 0.4)
    _step_axis(ax, [1e6, 2e6, 5e6, 1e7, 2e7, 5e7])
    ax.set_xlim(8e5, 5e7)
    ax.set_xlabel("env steps used, no success (state evaluations)")
    ax.set_title("[[n,k,3]] with w* <= 4, k = 2 and 4")
    fig.legend(
        [Line2D([], [], marker="o", color=figstyle.NEUTRAL, ls="none"),
         Line2D([], [], marker=">", color=figstyle.NEUTRAL, mfc="none", ls="none"),
         Line2D([], [], color=color["equivariant"]),
         Line2D([], [], color=color["mlp"])],
        ["first clean d = 4, one marker per seed", "no success; steps used before the wall cap",
         "equivariant policy", "flat MLP"],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14),
    )
    figstyle.panel_label(axes[0], "a")
    figstyle.panel_label(axes[1], "b")
    fig.tight_layout()
    figstyle.save(fig, out_stem)
    return Path(out_stem).with_suffix(".png")


def _write(out: Path, rec: dict):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=_json_default)
    print("wrote", out)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description="RL scaling probe.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("variant", help="one PPO run of a 4x4 variant")
    p.add_argument("--variant", choices=VARIANTS, default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--target", choices=("4x4", "3x3"), default="4x4")
    p.add_argument("--max-steps", type=int, default=None, help="episode cap override (the `long` variant doubles the target's cap)")
    p.add_argument("--no-stop", action="store_true", help="keep training after the first success")
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("evo", help="evolutionary-search reachability control")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("random", help="terminal distances of uniformly random gate sequences on the 4x4 target")
    p.add_argument("--circuits", type=int, default=8192)
    p.add_argument("--caps", nargs="*", type=int, default=[50, 60, 70, 80, 100])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)

    p = sub.add_parser("ldpc", help="LDPC-constrained [[n, k, 3]] target, w_max = 4")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--graph", required=True, help="graph spec, e.g. all_to_all:12 or square:3x4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--gates", nargs="*", default=None)
    p.add_argument("--ldpc-scale", type=float, default=0.25, help="penalty per unit of w* excess (regression: 0.25)")
    p.add_argument("--policy", choices=("mlp", "equivariant"), default="mlp")
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("report", help="print the tables and draw the figures from experiments/scaling/runs")
    p.add_argument("--dir", default=str(DEFAULT_OUT))
    p.add_argument("--figures", default="experiments/scaling/figures")

    p = sub.add_parser("figure", help="draw the headline figure (4x4 variants and LDPC cells) from experiments/scaling/runs")
    p.add_argument("--dir", default=str(DEFAULT_OUT))
    p.add_argument("--stem", default=str(FIGURE), help="output path without extension; .png and .pdf are written")

    args = ap.parse_args(argv)
    if args.cmd == "variant":
        target = TARGET_4X4 if args.target == "4x4" else TARGET_3X3
        suffix = ""
        if args.max_steps is not None:
            target = replace(target, max_steps=args.max_steps)
            suffix = f"_cap{args.max_steps}"
        rec = run_variant(args.variant, args.seed, args.minutes, args.threads, target, args.batch, verbose=args.verbose, stop_on_success=not args.no_stop)
        out = Path(args.out or DEFAULT_OUT / f"variant_{args.target}_{args.variant}{suffix}_seed{args.seed}.json")
        _write(out, rec)
        print(f"reached d = {target.d}: {rec['reached_d_target']} env_steps={rec['env_steps']:,} wall={rec['wall_time_s']}s")
        return 0
    if args.cmd == "evo":
        rec = run_evo(args.seed, args.minutes, max_steps=args.max_steps, verbose=args.verbose)
        out = Path(args.out or DEFAULT_OUT / f"evo_4x4_steps{rec['max_steps']}_seed{args.seed}.json")
        _write(out, rec)
        print(f"best d {rec['best_d']} reached {rec['reached_d_target']} evaluations {rec['evaluations']:,}")
        return 0
    if args.cmd == "random":
        rec = run_random(tuple(args.caps), args.circuits, args.seed)
        out = Path(args.out or DEFAULT_OUT / f"random_4x4_seed{args.seed}.json")
        _write(out, rec)
        for cap, c in rec["caps"].items():
            print(cap, c["d_hist"], "clean d4", c["d_hist_clean"][4], "idle", c["idle_mean"])
        return 0
    if args.cmd == "ldpc":
        gates = tuple(args.gates) if args.gates else (("H", "CNOT") if args.graph.startswith("square") else ("H", "S", "CNOT"))
        rec = run_ldpc(
            args.n, args.k, args.graph, args.seed, args.minutes, args.threads, max_steps=args.max_steps, gates=gates,
            verbose=args.verbose, scale=args.ldpc_scale, policy=args.policy,
        )
        suffix = ("" if args.ldpc_scale == 0.25 else f"_scale{args.ldpc_scale:g}") + ("" if args.policy == "mlp" else f"_{args.policy}")
        out = Path(args.out or DEFAULT_OUT / f"ldpc_n{args.n}_k{args.k}_{args.graph.replace(':', '_')}{suffix}_seed{args.seed}.json")
        _write(out, rec)
        print(f"success {rec['success']} first {rec['first_success']['env_steps'] if rec['first_success'] else None} post-rate {rec['post_success_rate']}")
        return 0
    if args.cmd == "report":
        recs = load_all(Path(args.dir))
        print("## Diagnosis\n")
        print(diag_table(recs))
        print("\n## Variants\n")
        print(variant_table(recs))
        print("\n## Random policy\n")
        print(random_table(recs))
        print("\n## Evolutionary control\n")
        print(evo_table(recs))
        print("\n## LDPC\n")
        print(ldpc_table(recs))
        fig = figure_diag(recs, Path(args.figures) / "diag_4x4")
        if fig:
            print("\nfigure:", fig[0])
        return 0
    if args.cmd == "figure":
        png = make_figure(Path(args.dir), Path(args.stem))
        print("wrote", png if png else "nothing (matplotlib missing)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
