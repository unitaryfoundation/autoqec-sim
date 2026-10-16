# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Known-code rediscovery regression with the PPO agent.

Same targets, reward (Knill-Laflamme plus low-weight-stabilizer penalty),
replay, and verification (exact distance, weight enumerators of ``S`` and
``N(S)`` against the reference) as :mod:`autoqec_sim.regress`; only the
search agent differs. A PPO run keeps training after its first success and
every distinct code the policy produces is verified, so one seed can cover
several code families. The ``full`` budget stops at the first code whose
enumerators match the reference and moves to the next seed when a run
exhausts its step or time budget; ``reduced`` stops at the first verified
nontrivial ``[[n, 1, 3]]`` code.

Cost axes recorded per attempt: ``env_steps`` (state reward evaluations,
batch elements times steps), ``episodes`` (circuits), and wall time. One
evolutionary-search evaluation in :mod:`autoqec_sim.regress` is one
circuit and costs up to ``max_steps`` state evaluations.

PPO-only targets. ``PPO_ONLY_TARGETS`` holds targets that the evolutionary
regression does not attempt: the 4x4 rotated surface code ``[[16, 1, 4]]``
on the square lattice with H and CNOT, 50-gate episodes, budget 1,000,000
env steps per attempt (the RL scaling probe reached a clean distance-4
state in 217k to 258k env steps with the equivariant policy, which
``PPOConfig(policy="auto")`` selects at 16 qubits). Its reward is the one
of every other target (normalized Knill-Laflamme term over the 16,248
Paulis of weight at most 3 plus the padding penalty) and the match
criterion is the same (enumerators of ``S`` and ``N(S)`` against
``known_codes.ROTATED_SURFACE_16``). The ``reduced`` budget skips it.

Run as ``python -m autoqec_sim.agents.ppo_regress --budget full --out
results/regression_ppo.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import numpy as np

from .. import graphs
from ..env import CodeDiscoveryEnv
from ..regress import TARGETS as EVO_TARGETS
from ..regress import Target, env_kwargs, make_reward, replay, verify
from ..symplectic import rref
from .ppo import PPOAgent, PPOConfig

PPO_ONLY_TARGETS = {
    # 4x4 rotated surface code: n = 16, d = 4, square lattice, H and CNOT,
    # 50-gate episodes as in the scaling probe; only weight-1 stabilizers
    # (idle qubits) are forbidden, as for the 3x3 target.
    "rotated_surface_16": Target(
        "rotated_surface_16", 16, 1, 4, "square:4x4", "rotated_surface_16", 50, 1, ("H", "CNOT")
    ),
}
TARGETS = {**EVO_TARGETS, **PPO_ONLY_TARGETS}

# Per-target overrides of the budget: env steps and wall cap per attempt.
# The 4x4 target runs at about 1k env steps per second at 2 threads (the
# reward evaluates 16,248 errors per state), so 1M steps need up to about
# 20 minutes; the wall cap is set so that the step budget is the binding one.
TARGET_BUDGETS = {
    "rotated_surface_16": {"full": dict(max_env_steps=1_000_000, max_time_s=3600.0, max_attempts=1)},
}
# Targets outside the reduced budget (too expensive for the quick check).
REDUCED_SKIP = frozenset({"rotated_surface_16"})

BUDGETS = {
    # batch, state evaluations per attempt, wall cap per attempt, attempts,
    # require enumerator match
    "full": dict(
        batch_size=256,
        max_env_steps=3_000_000,
        max_time_s=1200.0,
        max_attempts=3,
        require_match=True,
    ),
    "reduced": dict(
        batch_size=256,
        max_env_steps=400_000,
        max_time_s=90.0,
        max_attempts=1,
        require_match=False,
    ),
}

MAX_VERIFY_PER_UPDATE = 256


def code_key(tab: np.ndarray, k: int) -> bytes:
    """Canonical key of the stabilizer group: GF(2) reduced row echelon form
    of the stabilizer block (signs ignored)."""
    n = (tab.shape[-1] - 1) // 2
    r, _ = rref(tab[n + k :, : 2 * n])
    return r.tobytes()


class Verifier:
    """Verifies the distinct codes a PPO agent finds and stops it on a
    target-specific criterion."""

    def __init__(self, target: Target, require_match: bool, t0: float):
        self.target = target
        self.require_match = require_match
        self.t0 = t0
        self.seen: set[bytes] = set()
        self.families: dict[tuple, int] = {}  # enumerators -> count
        self.verified = 0
        self.skipped = 0
        self.outcome: dict | None = None
        self.first_good: dict | None = None

    def __call__(self, agent: PPOAgent, new) -> bool:
        checked = 0
        for fc in new:
            key = code_key(fc.tableau, self.target.k)
            if key in self.seen:
                continue
            if checked >= MAX_VERIFY_PER_UPDATE:
                self.skipped += 1
                continue
            self.seen.add(key)
            checked += 1
            code = replay(fc.gates, self.target.n)
            ver = verify(code, self.target)
            self.verified += 1
            fam = (tuple(ver["weight_enumerator_S"]), tuple(ver["weight_enumerator_N"]))
            self.families[fam] = self.families.get(fam, 0) + 1
            good = ver["parameters_match"] and ver["nontrivial"]
            if not good:
                continue
            rec = dict(fc.to_dict(), wall_time_s=round(time.perf_counter() - self.t0, 3), verification=ver)
            if self.first_good is None:
                self.first_good = rec
            if ver["enumerators_match_reference"] or not self.require_match:
                self.outcome = rec
                return True
        return False


def run_target(
    target: Target,
    seed: int = 0,
    batch_size: int = 256,
    max_env_steps: int = 3_000_000,
    max_time_s: float = 1200.0,
    max_attempts: int = 1,
    require_match: bool = False,
    config: PPOConfig | None = None,
    gates=None,
    verbose: bool = False,
) -> dict:
    gates = tuple(target.gates if gates is None else gates)
    graph = graphs.from_spec(target.graph_spec)
    reward = make_reward(target)
    config = config or PPOConfig()
    t0 = time.perf_counter()
    attempts = []
    total_steps = 0
    total_episodes = 0
    outcome = None
    num_parameters = None
    for attempt in range(max_attempts):
        s = seed + attempt
        env = CodeDiscoveryEnv(
            n=target.n,
            k=target.k,
            graph=graph,
            batch_size=batch_size,
            max_steps=target.max_steps,
            gates=gates,
            reward_fn=reward,
            seed=s,
            **env_kwargs(target, reward),
        )
        agent = PPOAgent(env, config=config, seed=s)
        num_parameters = agent.num_parameters()
        ta = time.perf_counter()
        ver = Verifier(target, require_match, ta)
        res = agent.train(
            max_env_steps, max_time_s=max_time_s, stop_on_success=False, callback=ver, verbose=verbose
        )
        total_steps += res.env_steps
        total_episodes += res.episodes
        rec = {
            "seed": s,
            "env_steps": res.env_steps,
            "episodes": res.episodes,
            "updates": res.updates,
            "wall_time_s": round(res.wall_time_s, 3),
            "successful_episodes": res.successes,
            "first_success_env_steps": res.first_success.env_steps if res.first_success else None,
            "first_success_episodes": res.first_success.episodes if res.first_success else None,
            "distinct_codes_verified": ver.verified,
            "distinct_codes_skipped": ver.skipped,
            "enumerator_families": [
                {"S": list(a), "N": list(b), "count": c} for (a, b), c in ver.families.items()
            ],
            "matched": ver.outcome is not None and ver.outcome["verification"]["enumerators_match_reference"],
            "first_nontrivial": ver.first_good,
            "result": ver.outcome,
            "final_success_rate": res.history[-1]["success_rate"] if res.history else None,
            "history": res.history,
        }
        attempts.append(rec)
        if ver.outcome is not None:
            outcome = rec
            break
    return {
        "target": target.name,
        "agent": "ppo",
        "parameters": [target.n, target.k, target.d],
        "graph": target.graph_spec,
        "gates": list(gates),
        "reference": target.reference,
        "forbid_stabilizer_weight": target.forbid_stabilizer_weight,
        "max_steps": target.max_steps,
        "budget": {
            "batch_size": batch_size,
            "max_env_steps_per_attempt": max_env_steps,
            "max_time_s_per_attempt": max_time_s,
            "max_attempts": max_attempts,
            "require_enumerator_match": require_match,
        },
        "ppo_config": config.to_dict(),
        "num_parameters": num_parameters,
        "seed": seed,
        "rediscovered": outcome is not None,
        "attempts_used": len(attempts),
        "total_env_steps": total_steps,
        "total_episodes": total_episodes,
        "wall_time_s": round(time.perf_counter() - t0, 3),
        "result": outcome["result"] if outcome else None,
        "attempts": attempts,
    }


def run_all(budget: str = "reduced", seed: int = 0, targets=None, verbose=False, config=None) -> dict:
    import torch

    cfg = BUDGETS[budget]
    names = [n for n in TARGETS if budget != "reduced" or n not in REDUCED_SKIP] if targets is None else list(targets)
    out = {
        "agent": "ppo",
        "budget": budget,
        "budget_config": cfg,
        "seed": seed,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "ppo_threads": (config or PPOConfig()).max_threads,
        "targets": {},
    }
    for name in names:
        out["targets"][name] = run_target(TARGETS[name], seed=seed, verbose=verbose, config=config, **target_budget(name, budget))
    return out


def target_budget(name: str, budget: str) -> dict:
    """The budget of ``name`` under ``budget``: ``BUDGETS[budget]`` with the
    target's ``TARGET_BUDGETS`` overrides applied."""
    return {**BUDGETS[budget], **TARGET_BUDGETS.get(name, {}).get(budget, {})}


def comparison_table(ppo: dict, evo_path: str | Path | None = None) -> str:
    """Markdown table of PPO against the evolutionary-search results JSON
    (``results/regression.json`` by default) on the same targets."""
    evo = None
    if evo_path is not None and Path(evo_path).exists():
        with open(evo_path) as f:
            evo = json.load(f)
    lines = [
        "| Target | Agent | Seeds | Circuits | State evaluations | Wall time | Enumerators match |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in ppo["targets"].items():
        t = TARGETS[name]
        if evo and name in evo["targets"]:
            e = evo["targets"][name]
            ev = e["result"]["verification"] if e["result"] else None
            lines.append(
                f"| {name} | evolutionary | {e['attempts_used']} | {e['total_evaluations']:,} "
                f"| about {e['total_evaluations'] * t.max_steps:,} | {e['wall_time_s']:.1f} s "
                f"| {'yes' if ev and ev['enumerators_match_reference'] else 'no'} |"
            )
        v = r["result"]["verification"] if r["result"] else None
        lines.append(
            f"| {name} | PPO | {r['attempts_used']} | {r['total_episodes']:,} "
            f"| {r['total_env_steps']:,} | {r['wall_time_s']:.1f} s "
            f"| {'yes' if v and v['enumerators_match_reference'] else 'no'} |"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Known-code rediscovery regression with PPO.")
    ap.add_argument("--budget", choices=list(BUDGETS), default="reduced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", nargs="*", default=None, choices=list(TARGETS))
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--compare", default="results/regression.json", help="evolutionary results JSON")
    ap.add_argument("--threads", type=int, default=None, help="torch and BLAS thread cap (default 6)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    # torch threads; BLAS threads are read from the environment at numpy
    # import, so set OMP_NUM_THREADS and friends in the shell as well
    config = None if args.threads is None else PPOConfig(max_threads=args.threads)
    res = run_all(args.budget, args.seed, args.targets, args.verbose, config=config)
    for name, r in res["targets"].items():
        v = r["result"]["verification"] if r["result"] else None
        line = (
            f"{name:18s} rediscovered={r['rediscovered']} attempts={r['attempts_used']} "
            f"env_steps={r['total_env_steps']} episodes={r['total_episodes']} wall={r['wall_time_s']}s"
        )
        if v:
            line += f" d={v['distance_exact']} css={v['is_css']} enumerators_match={v['enumerators_match_reference']}"
        fam = sum(len(a["enumerator_families"]) for a in r["attempts"])
        line += f" families={fam}"
        print(line)
    print()
    print(comparison_table(res, args.compare))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", args.out)
    # Exclusion targets (expect_success=False) pass when nothing is found.
    return 0 if all(r["rediscovered"] == TARGETS[n].expect_success for n, r in res["targets"].items()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
