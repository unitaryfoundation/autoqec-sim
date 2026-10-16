# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Three-topology run matrix: the PPO agent alone on square, triangular,
and heavy-hex lattices with one objective and one metric set.

Milestone M2 of the AutoQEC roadmap names a three-topology run matrix with
common metrics on heavy-hex, square, and triangular lattices. This module
runs that matrix for the RL agent of :mod:`autoqec_sim.agents.ppo` by
itself (not the LLM meta-agent the milestone also names): one objective on
every graph, three seeds per cell, two qubit-count tiers, and the same
metrics per cell.

Objective, identical on every topology: a ``[[n, 1, d >= 3]]`` code with
no idle qubit, H and CNOT restricted to the lattice edges, 50-gate episode
cap, the level distance reward of :class:`autoqec_sim.experiments.scaling.
LevelDistanceReward` (``min(d, 3) / 3 - 1`` plus the padding penalty, so the
reward is 0 iff the state is a code of distance at least 3 with no idle
qubit). ``PPOConfig`` defaults with ``policy = "auto"`` (flat MLP below 12
qubits, per-qubit equivariant network from 12 on), 256 parallel episodes.

Tiers: ``small`` targets ``n = 9`` and ``large`` targets ``n = 16``; each
topology takes the available lattice whose qubit count is nearest the
target (:func:`resolve_graph`). Square and triangular have exact sizes
(``3x3`` and ``4x4``). Heavy-hex sizes are fixed by the honeycomb
construction of ``autoqec_sim.noise.topology.heavy_hex`` (12, 18, 21, 25, ...
qubits), so the small tier takes ``heavy_hex:1x1`` (12 qubits, which is a
single hexagon with a qubit on every edge, that is a 12-cycle of degree 2)
and the large tier takes ``heavy_hex:2x1`` (18 qubits, two stacked
hexagons, two degree-3 qubits).

Metrics per cell: success (a distance-3 code reached), env steps and wall
time to first success, total wall time, the best code found (``n``, ``k``,
exact ``d``, exact bottleneck weight ``w*`` and maximum qubit degree of a
sparsest basis, CSS or not, weight enumerators of ``S`` and ``N(S)``, gate
count and depth of its encoder), the undetected fault mass of that encoder
under the Heron preset placed on the cell's graph and under the board
preset (both at scale 1; :mod:`autoqec_sim.circuit_noise`), and, where a
rotated surface code of the cell's size exists (``n = 9`` and ``16`` on the
square and triangular lattices, the latter containing the square edges),
the same masses for that code's standard-form encoder routed onto the
graph (best over logical qubit and pivot choice, unpolished).

Every run writes one JSON under ``results/topologies/``; ``report`` writes
``REPORT.md`` from the JSONs. ``python -m autoqec_sim.experiments.topologies
--help``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from autoqec_sim.noise import presets

from .. import graphs
from ..circuit_noise import (
    circuit_depth,
    fault_set,
    gates_to_layered_circuit,
    resolve_noise_model,
    undetected_fault_mass,
)
from ..code import StabilizerCode
from ..env import CodeDiscoveryEnv
from ..ldpc import sparsest_basis_stats
from .noise_aware import code_from_circuit, css_standard_encoder, rotated_surface, route_gates
from .scaling import LevelDistanceReward, _json_default, replay_gates

DEFAULT_OUT = Path("results/topologies")
TOPOLOGIES = ("square", "triangular", "heavy_hex")
TIERS = {"small": 9, "large": 16}
BUDGETS = {"small": 2_000_000, "large": 4_000_000}
SEEDS = (0, 1, 2)
MINUTES = 12.0
THREADS = 2
LOAD_LIMIT = 40.0  # 1-minute load above which the matrix runs on one thread
D_TARGET = 3
GATES = ("H", "CNOT")
MAX_STEPS = 50
BATCH = 256
EXACT_GENERATORS = 17  # exact w* up to n - k = 17 (the 18-qubit heavy-hex cell)
PROFILES = ("heron", "board")


# --------------------------------------------------------------------------
# graphs
# --------------------------------------------------------------------------
def candidate_specs(topology: str) -> list[str]:
    """Lattice specs of ``topology`` considered by :func:`resolve_graph`."""
    if topology in ("square", "triangular"):
        return [f"{topology}:{r}x{c}" for r in range(2, 7) for c in range(r, 7)]
    if topology == "heavy_hex":
        return [f"heavy_hex:{r}x{c}" for r in range(1, 4) for c in range(1, 4)]
    raise ValueError(f"unknown topology {topology!r}")


def resolve_graph(topology: str, target_n: int) -> graphs.Graph:
    """The lattice of ``topology`` whose qubit count is nearest ``target_n``
    (ties: the squarer lattice, then the larger one)."""
    best = None
    for spec in candidate_specs(topology):
        g = graphs.from_spec(spec)
        r, c = (int(v) for v in spec.split(":")[1].split("x"))
        key = (abs(g.n - target_n), abs(r - c), -g.n)
        if best is None or key < best[0]:
            best = (key, spec, g)
    return graphs.Graph(best[2].n, best[2].edges, best[1])


def graph_summary(g: graphs.Graph) -> dict:
    return {"spec": g.name, "n": g.n, "edges": len(g.edges), "max_degree": int(max(g.degree()))}


# --------------------------------------------------------------------------
# environment with a record of the best terminal state
# --------------------------------------------------------------------------
class TrackingEnv(CodeDiscoveryEnv):
    """``CodeDiscoveryEnv`` that keeps the best terminal state seen (highest
    state reward, then fewer gates) with its gate list."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.best_reward = -np.inf
        self.best_steps = None
        self.best_tableau = None
        self.best_gates = None
        self.terminal_d = np.zeros(D_TARGET + 1, dtype=np.int64)

    def reset(self, seed=None, where=None):
        # a partial reset (auto-reset inside ``step``) re-evaluates the reward
        # on the reset elements only; keep the full-batch distances of the step
        if where is None:
            return super().reset(seed, where)
        keep = (self.reward_fn.last_d, self.reward_fn.last_pad)
        out = super().reset(seed, where)
        self.reward_fn.last_d, self.reward_fn.last_pad = keep
        return out

    def step(self, actions):
        out = super().step(actions)
        info = out[4]
        if "final_tableau" in info:
            r = info["final_state_reward"]
            steps = info["final_steps"]
            d = self.reward_fn.last_d
            done = out[2] | out[3]
            if d is not None:
                self.terminal_d += np.bincount(np.minimum(d[done], D_TARGET), minlength=D_TARGET + 1)
            order = np.lexsort((steps, -r))
            i = order[0]
            if r[i] > self.best_reward or (r[i] == self.best_reward and steps[i] < self.best_steps):
                self.best_reward = float(r[i])
                self.best_steps = int(steps[i])
                self.best_tableau = info["final_tableau"][i].copy()
                self.best_gates = [self.action_table.entries[a] for a in info["final_history"][i] if a >= 0]
        return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def noise_models(graph: graphs.Graph) -> dict:
    """``{"heron": model | None, "board": model | None}`` at scale 1 on
    ``graph``; a profile that refuses the graph gives ``None`` and the
    reason in ``"_refused"``."""
    out = {"_refused": {}}
    builders = {"heron": lambda: presets.heron(graph=None), "board": lambda: presets.board()}
    for name in PROFILES:
        try:
            out[name] = resolve_noise_model(builders[name](), scale=1.0, graph=graph, n=graph.n)
        except Exception as exc:
            out[name] = None
            out["_refused"][name] = f"{type(exc).__name__}: {exc}"
    return out


def masses(gates, n: int, tab: np.ndarray, models: dict) -> dict:
    """Undetected and total fault mass of the encoder ``gates`` of the code
    with encoding tableau ``tab`` (``(2n, 2n+1)``) under every model."""
    out = {}
    for name in PROFILES:
        model = models.get(name)
        if model is None:
            out[name] = {"refused": models["_refused"].get(name, "unavailable")}
            continue
        try:
            faults = fault_set(model(gates_to_layered_circuit(gates, n)), n)
        except Exception as exc:
            out[name] = {"refused": f"{type(exc).__name__}: {exc}"}
            continue
        out[name] = {"undetected": undetected_fault_mass(tab[None], 1, faults), "total": faults.total}
    return out


def describe_code(tab: np.ndarray, k: int, gates, models: dict) -> dict:
    """Code parameters, sparsity, CSS test, weight enumerators, encoder
    statistics, and noise masses of one terminal tableau."""
    n = (tab.shape[-1] - 1) // 2
    code = StabilizerCode.from_tableau(tab[None], k)
    st = sparsest_basis_stats(tab[None], k, exact_max_generators=EXACT_GENERATORS)
    a, b = code.weight_enumerators()
    gates = [tuple(g) for g in gates]
    return {
        "n": n,
        "k": k,
        "d": int(code.distance_exact()),
        "w_star": int(st["w_star"][0]),
        "w_star_exact": bool(st["exact"]),
        "max_degree": int(st["max_degree"][0]),
        "is_css": bool(code.is_css()),
        "stabilizers": code.stabilizer_strings(),
        "weight_enumerator_S": a.tolist(),
        "weight_enumerator_NS": b.tolist(),
        "gates": [list(g) for g in gates],
        "gate_count": len(gates),
        "cnot_count": sum(g[0] == "CNOT" for g in gates),
        "depth": circuit_depth(gates, n),
        "mass": masses(gates, n, tab, models),
    }


# --------------------------------------------------------------------------
# rotated surface baseline
# --------------------------------------------------------------------------
def surface_baseline(graph: graphs.Graph, models: dict) -> dict | None:
    """Rotated surface code ``[[d^2, 1, d]]`` on a square or triangular
    ``d x d`` lattice: standard-form CSS encoder routed onto the graph,
    over every logical input qubit and pivot choice, the lowest-mass
    circuit per profile (unpolished). ``None`` where no construction
    exists (heavy-hex, or ``n`` not a square)."""
    kind, _, size = graph.name.partition(":")
    if kind not in ("square", "triangular"):
        return None
    r, c = (int(v) for v in size.split("x"))
    if r != c:
        return None
    d = r
    n = d * d
    stabs = rotated_surface(d)
    code = StabilizerCode.from_stabilizers(stabs)
    cands = []
    seen = set()
    for logical in range(n):
        for sd in [None, *list(range(12))]:
            try:
                base = css_standard_encoder(stabs, logical=logical, seed=sd, graph=graph)
            except RuntimeError:
                continue
            routed = route_gates(base, graph)
            key = tuple(routed)
            if key in seen:
                continue
            seen.add(key)
            tab = code_from_circuit(routed, n, logical).to_tableau()[0]
            cands.append({"logical": logical, "seed": sd, "gates": routed, "mass": masses(routed, n, tab, models)})
    if not cands:
        return None
    out = {
        "code": f"rotated surface [[{n},1,{d}]]",
        "d": int(code.distance_exact()),
        "candidates": len(cands),
        "best_by_profile": {},
    }
    for name in PROFILES:
        ok = [c for c in cands if "undetected" in c["mass"][name]]
        if not ok:
            out["best_by_profile"][name] = {"refused": cands[0]["mass"][name].get("refused", "unavailable")}
            continue
        best = min(ok, key=lambda c: c["mass"][name]["undetected"])
        out["best_by_profile"][name] = {
            "undetected": best["mass"][name]["undetected"],
            "total": best["mass"][name]["total"],
            "gate_count": len(best["gates"]),
            "cnot_count": sum(g[0] == "CNOT" for g in best["gates"]),
            "depth": circuit_depth(best["gates"], n),
            "logical_qubit": best["logical"],
            "pivot_seed": best["seed"],
            "gates": [list(g) for g in best["gates"]],
        }
    return out


# --------------------------------------------------------------------------
# one cell, one seed
# --------------------------------------------------------------------------
def load_1min() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):  # pragma: no cover
        return None


def pick_threads(threads: int = THREADS) -> tuple[int, float | None]:
    """``threads``, or 1 when the 1-minute load exceeds ``LOAD_LIMIT``."""
    load = load_1min()
    if load is not None and load > LOAD_LIMIT:
        return 1, load
    return threads, load


def run_cell(
    topology: str,
    target_n: int,
    seed: int,
    max_env_steps: int | None = None,
    minutes: float = MINUTES,
    threads: int | None = None,
    batch: int = BATCH,
    verbose: bool = False,
    log_every: int = 25,
) -> dict:
    """One PPO run on ``topology`` at the lattice nearest ``target_n``;
    stops at the first distance-3 code with no idle qubit, the step
    budget, or the wall cap."""
    from ..agents.ppo import PPOAgent, PPOConfig

    if threads is None:
        threads, load = pick_threads()
    else:
        load = load_1min()
    graph = resolve_graph(topology, target_n)
    tier = min(TIERS, key=lambda t: abs(TIERS[t] - target_n))
    if max_env_steps is None:
        max_env_steps = BUDGETS[tier]
    reward = LevelDistanceReward(D_TARGET, D_TARGET)
    env = TrackingEnv(
        n=graph.n, k=1, graph=graph, batch_size=batch, max_steps=MAX_STEPS, gates=GATES,
        reward_fn=reward, seed=seed, auto_reset=True,
    )
    config = PPOConfig(max_threads=threads)
    agent = PPOAgent(env, config=config, seed=seed)

    def cb(agent, new):
        if verbose and (agent.updates % log_every == 0 or new):
            h = agent.history[-1]
            print(
                f"{topology} n={graph.n} seed={seed} upd {h['update']:5d} steps {h['env_steps']:9,d} "
                f"r {h['mean_reward']:+.3f} ent {h['entropy']:.3f} term_d {env.terminal_d.tolist()} "
                f"best {env.best_reward:+.3f} t {h['wall_time_s']:.0f}s",
                flush=True,
            )
        return False

    t0 = time.perf_counter()
    res = agent.train(max_env_steps, max_time_s=minutes * 60.0, stop_on_success=True, callback=cb, verbose=False)
    wall = time.perf_counter() - t0
    models = noise_models(graph)
    first = None
    best = None
    if res.first_success is not None:
        fc = res.first_success
        first = {
            "env_steps": fc.env_steps,
            "episodes": fc.episodes,
            "update": fc.update,
            "wall_time_s": next((h["wall_time_s"] for h in agent.history if h["env_steps"] >= fc.env_steps), None),
            "gate_count": len(fc.gates),
        }
        # best code: among the successes of the stopping update, the shortest encoder
        cands = sorted(agent.successes, key=lambda c: len(c.gates))
        fc_best = cands[0]
        tab = replay_gates(fc_best.gates, graph.n, 1)
        best = describe_code(tab, 1, fc_best.gates, models)
        best["source"] = f"shortest of {len(agent.successes)} successful episodes of update {fc_best.update + 1}"
    elif env.best_tableau is not None:
        best = describe_code(env.best_tableau, 1, env.best_gates, models)
        best["source"] = "best terminal state (no success)"
    return {
        "kind": "run",
        "topology": topology,
        "tier": tier,
        "target_n": target_n,
        "graph": graph_summary(graph),
        "n": graph.n,
        "k": 1,
        "d_target": D_TARGET,
        "gates": list(GATES),
        "max_steps": MAX_STEPS,
        "batch": batch,
        "num_actions": env.num_actions,
        "seed": seed,
        "threads": threads,
        "load_1min_at_start": load,
        "ppo_config": dict(agent.config.to_dict(), policy_resolved=agent.policy),
        "num_parameters": agent.num_parameters(),
        "budget_env_steps": max_env_steps,
        "minutes_cap": minutes,
        "wall_time_s": round(wall, 1),
        "env_steps": agent.env_steps,
        "episodes": agent.episodes,
        "updates": agent.updates,
        "success": res.success,
        "successes": len(agent.successes),
        "first_success": first,
        "terminal_d_counts": env.terminal_d.tolist(),
        "best_code": best,
        "noise_refused": models["_refused"],
        "platform": platform.platform(),
        "history": agent.history,
    }


def run_baseline(topology: str, target_n: int) -> dict:
    graph = resolve_graph(topology, target_n)
    models = noise_models(graph)
    t0 = time.perf_counter()
    base = surface_baseline(graph, models)
    return {
        "kind": "baseline",
        "topology": topology,
        "target_n": target_n,
        "graph": graph_summary(graph),
        "n": graph.n,
        "noise_refused": models["_refused"],
        "surface": base,
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def run_path(out_dir: Path, topology: str, n: int, seed: int) -> Path:
    return out_dir / f"run_{topology}_n{n}_seed{seed}.json"


def baseline_path(out_dir: Path, topology: str, n: int) -> Path:
    return out_dir / f"baseline_{topology}_n{n}.json"


def write(out: Path, rec: dict):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=_json_default)
    print("wrote", out, flush=True)


def load_all(out_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(out_dir.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        r["_file"] = str(p)
        recs.append(r)
    return recs


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def _mass(m: dict | None, key="undetected") -> str:
    if not m:
        return "n/a"
    if "refused" in m:
        return "refused"
    return f"{m[key]:.3g}"


def _cells(recs):
    runs = [r for r in recs if r.get("kind") == "run"]
    keys = sorted({(r["tier"], r["topology"], r["n"]) for r in runs}, key=lambda t: (TIERS[t[0]], TOPOLOGIES.index(t[1])))
    return [(k, sorted((r for r in runs if (r["tier"], r["topology"], r["n"]) == k), key=lambda r: r["seed"])) for k in keys]


def matrix_table(recs: list[dict]) -> str:
    lines = [
        "| tier | topology | graph | n | edges | max degree | actions | policy | parameters | seeds | successes | env steps to first success | update of first success | wall to first success (s) | total wall (s) | env steps used |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (tier, topo, n), rs in _cells(recs):
        r0 = rs[0]
        g = r0["graph"]
        steps = ", ".join(_fmt(r["first_success"]["env_steps"]) if r["first_success"] else "none" for r in rs)
        walls = ", ".join(_fmt(r["first_success"]["wall_time_s"], 1) if r["first_success"] else "none" for r in rs)
        upd = ", ".join(str(r["first_success"]["update"] + 1) if r["first_success"] else "none" for r in rs)
        lines.append(
            f"| {tier} | {topo} | `{g['spec']}` | {n} | {g['edges']} | {g['max_degree']} | {r0['num_actions']} "
            f"| {r0['ppo_config'].get('policy_resolved', '?')} | {r0['num_parameters']:,} | {', '.join(str(r['seed']) for r in rs)} "
            f"| {sum(r['success'] for r in rs)} of {len(rs)} | {steps} | {upd} | {walls} "
            f"| {', '.join(_fmt(r['wall_time_s'], 1) for r in rs)} | {', '.join(_fmt(r['env_steps']) for r in rs)} |"
        )
    return "\n".join(lines)


def code_table(recs: list[dict]) -> str:
    bases = {(b["topology"], b["n"]): b for b in recs if b.get("kind") == "baseline"}
    lines = [
        "| tier | topology | n | seed | code | w* | max degree | CSS | gates (CNOT), depth | heron undetected / total | board undetected / total | surface code heron undetected (gates) | surface code board undetected (gates) | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (tier, topo, n), rs in _cells(recs):
        base = bases.get((topo, n), {}).get("surface")
        for r in rs:
            c = r["best_code"]
            if c is None:
                lines.append(f"| {tier} | {topo} | {n} | {r['seed']} | none | | | | | | | | | `{Path(r['_file']).name}` |")
                continue
            code = f"[[{c['n']},{c['k']},{c['d']}]]" + ("" if r["success"] else " (no success; best terminal state)")
            ws = str(c["w_star"]) + ("" if c["w_star_exact"] else " (bound)")
            m = c["mass"]
            heron = f"{_mass(m.get('heron'))} / {_mass(m.get('heron'), 'total')}"
            board = f"{_mass(m.get('board'))} / {_mass(m.get('board'), 'total')}"
            if base:
                bh = base["best_by_profile"].get("heron", {})
                bb = base["best_by_profile"].get("board", {})
                sh = f"{_mass(bh)} ({bh.get('gate_count', '')})" if "undetected" in bh else _mass(bh)
                sb = f"{_mass(bb)} ({bb.get('gate_count', '')})" if "undetected" in bb else _mass(bb)
            else:
                sh = sb = "n/a"
            lines.append(
                f"| {tier} | {topo} | {n} | {r['seed']} | {code} | {ws} | {c['max_degree']} | {'yes' if c['is_css'] else 'no'} "
                f"| {c['gate_count']} ({c['cnot_count']}), {c['depth']} | {heron} | {board} | {sh} | {sb} | `{Path(r['_file']).name}` |"
            )
    return "\n".join(lines)


def enumerator_table(recs: list[dict]) -> str:
    lines = [
        "| tier | topology | n | seed | code | A(S), weights 0..n | B(N(S)), weights 0..n |",
        "|---|---|---|---|---|---|---|",
    ]
    for (tier, topo, n), rs in _cells(recs):
        for r in rs:
            c = r["best_code"]
            if c is None:
                continue
            lines.append(
                f"| {tier} | {topo} | {n} | {r['seed']} | [[{c['n']},{c['k']},{c['d']}]] "
                f"| {c['weight_enumerator_S']} | {c['weight_enumerator_NS']} |"
            )
    return "\n".join(lines)


def baseline_table(recs: list[dict]) -> str:
    lines = [
        "| topology | graph | n | code | candidates | heron: undetected / total, gates (CNOT), depth, logical qubit | board: undetected / total, gates (CNOT), depth, logical qubit | file |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b in sorted((b for b in recs if b.get("kind") == "baseline"), key=lambda b: (b["n"], TOPOLOGIES.index(b["topology"]))):
        s = b["surface"]
        if s is None:
            lines.append(f"| {b['topology']} | `{b['graph']['spec']}` | {b['n']} | n/a (no rotated surface code of this size on this graph) | | | | `{Path(b['_file']).name}` |")
            continue
        cols = []
        for name in PROFILES:
            p = s["best_by_profile"].get(name, {})
            if "undetected" in p:
                cols.append(f"{p['undetected']:.3g} / {p['total']:.3g}, {p['gate_count']} ({p['cnot_count']}), {p['depth']}, {p['logical_qubit']}")
            else:
                cols.append("refused")
        lines.append(f"| {b['topology']} | `{b['graph']['spec']}` | {b['n']} | {s['code']} (exact d = {s['d']}) | {s['candidates']} | {cols[0]} | {cols[1]} | `{Path(b['_file']).name}` |")
    return "\n".join(lines)


def summary_text(recs: list[dict]) -> str:
    runs = [r for r in recs if r.get("kind") == "run"]
    if not runs:
        return "No runs recorded yet."
    cells = _cells(recs)
    bases = {(b["topology"], b["n"]): b for b in recs if b.get("kind") == "baseline"}
    parts = []
    early = []
    ratios = []
    for (tier, topo, n), rs in cells:
        ok = [r for r in rs if r["success"]]
        if ok:
            st = [r["first_success"]["env_steps"] for r in ok]
            up = [r["first_success"]["update"] + 1 for r in ok]
            parts.append(f"{topo} n = {n}: {len(ok)} of {len(rs)} seeds, first success at {min(st):,} to {max(st):,} env steps")
            if max(up) <= 3:
                early.append(f"{topo} n = {n} (update {min(up)} to {max(up)})" if min(up) != max(up) else f"{topo} n = {n} (update {max(up)})")
        else:
            bd = max((r["best_code"]["d"] for r in rs if r["best_code"]), default=0)
            parts.append(f"{topo} n = {n}: 0 of {len(rs)} seeds (best terminal distance {bd})")
        base = (bases.get((topo, n)) or {}).get("surface")
        if base and ok:
            for name in PROFILES:
                b = base["best_by_profile"].get(name, {})
                vals = [r["best_code"]["mass"][name].get("undetected") for r in ok]
                if "undetected" in b and all(v is not None for v in vals):
                    ratios.append((topo, n, name, min(vals) / b["undetected"], max(vals) / b["undetected"]))
    threads = sorted({r["threads"] for r in runs})
    loads = [r["load_1min_at_start"] for r in runs if r.get("load_1min_at_start") is not None]
    wall_total = sum(r["wall_time_s"] for r in runs)
    cap = max(r["wall_time_s"] for r in runs)
    text = (
        "This is the PPO agent of `autoqec_sim.agents.ppo` by itself on three lattices, "
        "not the LLM meta-agent that milestone M2 also names. "
        f"Cells: {'; '.join(parts)}. "
    )
    if early:
        text += (
            "The first success came within the first three PPO updates on "
            + ", ".join(early)
            + ", that is under a policy still close to its random initialization, so those cells measure "
            "reachability under near-uniform sampling of lattice gates, not learning. "
        )
    if ratios:
        text += "Undetected fault mass of the agent's encoder relative to the routed rotated surface encoder on the same graph: " + "; ".join(
            f"{topo} n = {n}, {name}: {lo:.2f}x to {hi:.2f}x" for topo, n, name, lo, hi in ratios
        ) + ". "
        bigger = [b["surface"] for b in bases.values() if b.get("surface") and b["surface"]["d"] > D_TARGET]
        if bigger:
            gc = [p["gate_count"] for s in bigger for p in s["best_by_profile"].values() if "gate_count" in p]
            text += (
                f"Where the surface code of the cell's size has distance {max(s['d'] for s in bigger)}, the baseline protects "
                f"against more than the objective asks and its routed encoder is {min(gc)} to {max(gc)} gates long "
                f"against the agent's at most {MAX_STEPS}. "
            )
    text += f"Runs used {', '.join(str(t) for t in threads)} torch thread{'s' if max(threads) > 1 else ''} each, sequentially"
    if loads:
        text += f", with a 1-minute load of {min(loads):.0f} to {max(loads):.0f} at run start on a shared workstation"
    text += f"; total wall time {wall_total / 60:.0f} min over {len(runs)} runs, longest run {cap:.0f} s, so no run reached its step budget or the {MINUTES:.0f} min wall cap."
    return text


def write_report(out_dir: Path) -> Path:
    recs = load_all(out_dir)
    runs = [r for r in recs if r.get("kind") == "run"]
    refused = {}
    for r in recs:
        for name, why in (r.get("noise_refused") or {}).items():
            refused.setdefault(name, why)
    L = [
        "# Three-topology run matrix: PPO on square, triangular, and heavy-hex lattices",
        "",
        "Status: complete for the cells listed. Every number comes from a JSON under",
        f"`{out_dir}/` (named in the last column of each table) and the tables",
        "regenerate with `uv run --extra rl python -m autoqec_sim.experiments.topologies report`.",
        "Code: `src/autoqec_sim/experiments/topologies.py`.",
        "",
        "## Summary",
        "",
        summary_text(recs),
        "",
        "## Setup",
        "",
        "Objective, identical on every topology: a `[[n, 1, d >= 3]]` code with no idle",
        "qubit (the padding penalty of the regression, minus the fraction of weight-1",
        "Paulis in the stabilizer group), H and CNOT on the lattice edges only, 50-gate",
        "episode cap, level distance reward `min(d, 3) / 3 - 1` plus the padding penalty",
        "(`experiments.scaling.LevelDistanceReward`; reward 0 iff the target is met).",
        "`PPOConfig` defaults with `policy = \"auto\"` (flat MLP below 12 qubits, per-qubit",
        f"equivariant network from 12 on), {BATCH} parallel episodes, seeds {', '.join(map(str, SEEDS))}.",
        f"Budgets: {BUDGETS['small']:,} env steps for the small tier, {BUDGETS['large']:,} for the",
        f"large tier, {MINUTES:.0f} min wall cap per run; a run stops at its first success.",
        "",
        "Graphs. Square and triangular lattices have exact sizes at both tiers (`3x3`,",
        "`4x4`). Heavy-hex sizes are fixed by the honeycomb construction of",
        "`autoqec_sim.noise.topology.heavy_hex` (12, 18, 21, 25, ... qubits), so the small",
        "tier takes `heavy_hex:1x1` (12 qubits; a single hexagon with a qubit on every",
        "edge is a 12-cycle, every qubit of degree 2) and the large tier takes",
        "`heavy_hex:2x1` (18 qubits, two stacked hexagons, two qubits of degree 3).",
        "The heavy-hex small cell therefore has more qubits than the square and",
        "triangular small cells, and its large cell has two more than theirs.",
        "",
        "Metrics. `w*` is the exact minimum bottleneck generator weight of the",
        "stabilizer group and `max degree` the maximum qubit degree of the sparsest",
        "basis (`autoqec_sim.ldpc`, exact up to `n - k = 17`). Noise masses are the",
        "undetected fault mass of the encoder under `presets.heron(graph=None)` placed",
        "on the cell's graph (IBM Heron r2 rates, parallel timing) and under",
        "`presets.board()` (one depolarizing rate per two-qubit gate and idle qubit per",
        "layer), both at scale 1 and unnormalized (`circuit_noise.undetected_fault_mass`),",
        "with the total mass beside it. The rotated surface code column is the same",
        "mass for the standard-form CSS encoder of the rotated surface code of the",
        "cell's size, routed onto the cell's graph, best over the logical input qubit",
        "and pivot choice, without the deletion polish of the noise-aware experiment;",
        "it exists only where `n` is a perfect square on the square and triangular",
        "lattices (the triangular lattice contains the square edges).",
        "",
        "## Matrix",
        "",
        matrix_table(recs),
        "",
        "## Best code per run",
        "",
        code_table(recs),
        "",
        "## Weight enumerators",
        "",
        enumerator_table(recs),
        "",
        "## Rotated surface baselines",
        "",
        baseline_table(recs),
        "",
        "## Caveats",
        "",
        "- RL agent only. The milestone's LLM meta-agent (proposing rewards, action",
        "  sets, or targets) is not part of this matrix; every cell uses one fixed",
        "  objective and the default agent.",
        "- The objective is distance 3 with no idle qubit, not a code tuned for the",
        "  noise profile; the noise masses are measured after the fact on the",
        "  encoder the agent happened to produce (the first success, shortest",
        "  circuit among the successes of the stopping update). The surface code",
        "  baseline is optimized over encoder choices for each profile, so the",
        "  comparison favours the baseline.",
        "- A run stops at its first success, so success rate after training and",
        "  the diversity of codes are not measured here.",
        "- Heavy-hex cells are not size-matched to the other topologies (12 and 18",
        "  qubits against 9 and 16); the 12-qubit heavy-hex graph is a cycle.",
        "- Three seeds per cell; the env-step counts are three samples, not a",
        "  distribution.",
    ]
    if refused:
        L.append("- Noise presets that refused a graph: " + "; ".join(f"`{k}`: {v}" for k, v in refused.items()) + ".")
    else:
        L.append("- No noise preset refused any of the graphs.")
    if runs:
        L += ["- Platform: `" + runs[0]["platform"] + "`."]
    L += [
        "",
        "## Regenerate",
        "",
        "```",
        "uv run --extra rl python -m autoqec_sim.experiments.topologies matrix",
        "uv run --extra rl python -m autoqec_sim.experiments.topologies report",
        "```",
        "",
        "One cell, one seed, and one baseline:",
        "",
        "```",
        "uv run --extra rl python -m autoqec_sim.experiments.topologies run --topology heavy_hex --target-n 16 --seed 0",
        "uv run --extra rl python -m autoqec_sim.experiments.topologies baseline --topology square --target-n 9",
        "```",
        "",
    ]
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(L))
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Three-topology run matrix for the PPO agent.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="one PPO run on one topology and seed")
    p.add_argument("--topology", choices=TOPOLOGIES, required=True)
    p.add_argument("--target-n", type=int, required=True, help="qubit count; the nearest available lattice is used")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-steps", type=int, default=None, help="budget override (default by tier)")
    p.add_argument("--minutes", type=float, default=MINUTES)
    p.add_argument("--threads", type=int, default=None, help="default 2, or 1 when the 1-minute load exceeds 40")
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("baseline", help="rotated surface code masses on one topology")
    p.add_argument("--topology", choices=TOPOLOGIES, required=True)
    p.add_argument("--target-n", type=int, required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))

    p = sub.add_parser("matrix", help="every baseline and every cell and seed, sequentially")
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    p.add_argument("--tiers", nargs="*", choices=list(TIERS), default=list(TIERS))
    p.add_argument("--topologies", nargs="*", choices=TOPOLOGIES, default=list(TOPOLOGIES))
    p.add_argument("--minutes", type=float, default=MINUTES)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--skip-existing", action="store_true", help="skip cells whose JSON exists")
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("report", help="write REPORT.md from the JSONs")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))

    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.cmd == "run":
        rec = run_cell(
            args.topology, args.target_n, args.seed, max_env_steps=args.env_steps, minutes=args.minutes,
            threads=args.threads, batch=args.batch, verbose=args.verbose,
        )
        write(run_path(out_dir, args.topology, rec["n"], args.seed), rec)
        fs = rec["first_success"]
        print(f"success={rec['success']} first={fs['env_steps'] if fs else None} env_steps={rec['env_steps']:,} wall={rec['wall_time_s']}s")
        return 0
    if args.cmd == "baseline":
        rec = run_baseline(args.topology, args.target_n)
        write(baseline_path(out_dir, args.topology, rec["n"]), rec)
        return 0
    if args.cmd == "matrix":
        t0 = time.perf_counter()
        for tier in args.tiers:
            for topo in args.topologies:
                n = resolve_graph(topo, TIERS[tier]).n
                bp = baseline_path(out_dir, topo, n)
                if not (args.skip_existing and bp.exists()):
                    write(bp, run_baseline(topo, TIERS[tier]))
        for tier in args.tiers:
            for topo in args.topologies:
                n = resolve_graph(topo, TIERS[tier]).n
                for seed in args.seeds:
                    rp = run_path(out_dir, topo, n, seed)
                    if args.skip_existing and rp.exists():
                        continue
                    rec = run_cell(topo, TIERS[tier], seed, minutes=args.minutes, threads=args.threads, verbose=args.verbose)
                    write(rp, rec)
                    fs = rec["first_success"]
                    print(
                        f"[{(time.perf_counter() - t0) / 60:.1f} min] {tier} {topo} n={n} seed={seed}: "
                        f"success={rec['success']} first={fs['env_steps'] if fs else None} wall={rec['wall_time_s']}s",
                        flush=True,
                    )
        print("report:", write_report(out_dir))
        return 0
    if args.cmd == "report":
        print("report:", write_report(out_dir))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
