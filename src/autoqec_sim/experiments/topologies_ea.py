# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Evolutionary-search benchmark on the three-topology matrix.

The same cells, objective, gate set, episode cap, and environment-step
budget as :mod:`autoqec_sim.experiments.topologies` (the PPO matrix), with
:class:`autoqec_sim.search.EvolutionarySearch` in place of the agent: one
batched rollout per generation, budget in evaluations = env steps / episode
cap. ``report`` joins the two result sets into a comparison table.

    python -m autoqec_sim.experiments.topologies_ea run --topology square --target-n 9 --seed 0
    python -m autoqec_sim.experiments.topologies_ea report
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from ..search import EvolutionarySearch
from . import topologies as T

DEFAULT_OUT = Path("results/topologies_ea")


def run_cell(topology: str, target_n: int, seed: int, max_evaluations: int | None = None,
             minutes: float = T.MINUTES, batch: int = T.BATCH, verbose: bool = False) -> dict:
    graph = T.resolve_graph(topology, target_n)
    tier = min(T.TIERS, key=lambda t: abs(T.TIERS[t] - target_n))
    budget_steps = T.BUDGETS[tier]
    if max_evaluations is None:
        max_evaluations = budget_steps // T.MAX_STEPS
    reward = T.LevelDistanceReward(T.D_TARGET, T.D_TARGET)
    env = T.TrackingEnv(n=graph.n, k=1, graph=graph, batch_size=batch, max_steps=T.MAX_STEPS, gates=T.GATES,
                        reward_fn=reward, seed=seed, auto_reset=True)
    search = EvolutionarySearch(env, seed=seed)
    t0 = time.perf_counter()
    res = search.run(max_evaluations=max_evaluations, stop_on_success=True, verbose=verbose) if "verbose" in search.run.__code__.co_varnames \
        else search.run(max_evaluations=max_evaluations, stop_on_success=True)
    wall = time.perf_counter() - t0
    models = T.noise_models(graph)
    best = None
    gates = getattr(res, "gates", None)
    if res.success and gates:
        tab = T.replay_gates(gates, graph.n, 1)
        best = T.describe_code(tab, 1, gates, models)
        best["source"] = "fittest successful element of the stopping generation"
    elif env.best_tableau is not None:
        best = T.describe_code(env.best_tableau, 1, env.best_gates, models)
        best["source"] = "best terminal state (no success)"
    return {"kind": "ea_run", "agent": "evolutionary", "topology": topology, "tier": tier, "target_n": target_n,
            "graph": T.graph_summary(graph), "n": graph.n, "k": 1, "d_target": T.D_TARGET, "gates": list(T.GATES),
            "max_steps": T.MAX_STEPS, "batch": batch, "num_actions": env.num_actions, "seed": seed,
            "budget_env_steps": budget_steps, "budget_evaluations": max_evaluations,
            "evaluations": res.evaluations, "env_steps_equivalent": res.evaluations * T.MAX_STEPS,
            "generations": getattr(res, "generations", None), "wall_time_s": round(wall, 1),
            "success": bool(res.success), "best_reward": getattr(res, "best_reward", None),
            "first_success": {"evaluations": res.evaluations, "env_steps_equivalent": res.evaluations * T.MAX_STEPS,
                              "wall_time_s": round(wall, 1), "gate_count": len(gates)} if res.success and gates else None,
            "terminal_d_counts": env.terminal_d.tolist(), "best_code": best, "platform": platform.platform()}


def run_path(out_dir: Path, topology: str, n: int, seed: int) -> Path:
    return out_dir / f"ea_{topology}_n{n}_seed{seed}.json"


def load_all(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("ea_*.json"))]


def report(out_dir: Path, ppo_dir: Path) -> Path:
    ea = load_all(out_dir)
    ppo = [r for r in T.load_all(ppo_dir) if r.get("kind") == "run"]

    def cell(r):
        return (r["topology"], r["n"])

    cells = sorted({cell(r) for r in ea} | {cell(r) for r in ppo}, key=lambda c: (c[1], c[0]))
    lines = ["# Evolutionary search against PPO on the three-topology matrix", "",
             "Same objective ([[n, 1, 3]] with no idle qubit, H and CNOT on the lattice edges, 50-gate episodes), "
             "same environment-step budget per tier. Evolutionary search spends its budget as whole episodes "
             "(evaluations x 50 steps); PPO as environment steps. Success = a distance-3 code with no idle qubit within the budget.", "",
             "| lattice | n | PPO successes | PPO median steps to success | EA successes | EA median steps to success | EA median wall |", "|---|---|---|---|---|---|---|"]
    import statistics
    for top, n in cells:
        p = [r for r in ppo if cell(r) == (top, n)]
        e = [r for r in ea if cell(r) == (top, n)]
        ps = [r for r in p if r.get("success")]
        es = [r for r in e if r.get("success")]

        def med(rows, key):
            vals = [key(r) for r in rows]
            return f"{statistics.median(vals):,.0f}" if vals else "none"

        pmed = med(ps, lambda r: r["first_success"]["env_steps"]) if p else "no run"
        emed = med(es, lambda r: r["first_success"]["env_steps_equivalent"]) if e else "no run"
        ewall = f"{statistics.median(r['wall_time_s'] for r in e):.0f} s" if e else "no run"
        lines.append(f"| {top} | {n} | {len(ps)}/{len(p)} | {pmed} | {len(es)}/{len(e)} | {emed} | {ewall} |")
    tot_p = sum(1 for r in ppo if r.get("success"))
    tot_e = sum(1 for r in ea if r.get("success"))
    lines += ["", f"Totals: PPO {tot_p} of {len(ppo)} runs succeed, evolutionary search {tot_e} of {len(ea)}. "
              "The evolutionary search is autoqec-sim's own (population of gate sequences, mutation, one batched rollout per generation), "
              "not a tuned evolutionary agent; the comparison fixes the budget, objective, and action space, nothing else."]
    out = out_dir / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--topology", choices=T.TOPOLOGIES, required=True)
    p.add_argument("--target-n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-evaluations", type=int, default=None)
    p.add_argument("--batch", type=int, default=T.BATCH)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--verbose", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--out-dir", default=str(DEFAULT_OUT))
    r.add_argument("--ppo-dir", default=str(T.DEFAULT_OUT))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        rec = run_cell(a.topology, a.target_n, a.seed, a.max_evaluations, batch=a.batch, verbose=a.verbose)
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = run_path(out, a.topology, rec["n"], a.seed)
        path.write_text(json.dumps(rec, indent=1, default=T._json_default) + "\n")
        fs = rec["first_success"]
        print(f"{a.topology} n={rec['n']} seed={a.seed}: success={rec['success']} evaluations={rec['evaluations']:,} "
              f"({rec['env_steps_equivalent']:,} env steps) wall={rec['wall_time_s']}s" + (f" first success at {fs['evaluations']:,} evaluations, {fs['gate_count']} gates" if fs else ""))
    else:
        print("wrote", report(Path(a.out_dir), Path(a.ppo_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
