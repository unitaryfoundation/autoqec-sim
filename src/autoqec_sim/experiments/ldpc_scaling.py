# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""LDPC constraint forms above ``n = 9``.

The RL scaling probe (``experiments/scaling/REPORT.md``) found that the
penalty form of the rediscovery regression traps every PPO policy at
``n = 12`` to ``16`` in sparse stabilizer groups of distance 1: the
normalized Knill-Laflamme term pays ``1/630`` to ``1/1128`` per detected
error while one unit of ``w*`` excess costs 0.25. This module compares the
forms of :class:`autoqec_sim.ldpc.LdpcObjective` on the LDPC-constrained
targets ``[[12, 2, 3]]`` and ``[[16, 4, 3]]`` with ``w_max = 4`` on the
all-to-all and the square graph, three seeds each, with the equivariant
policy:

``unnormalized``
    per-error Knill-Laflamme term: one undetected error costs four units
    of ``w*`` excess;
``final``
    hard final-state constraint: success requires exact ``w* <= w_max``,
    the reward carries no LDPC term;
``schedule``
    distance first: the penalty is switched on once the success rate at
    the target distance over ``schedule_window`` updates reaches
    ``schedule_rate``;
``penalty``
    the regression's form, for reference (the scaling probe's runs of it
    are in ``results/scaling/``).

Every run gets the same budget, 1,500,000 env steps or 20 minutes,
whichever comes first, and no early stop, so the success rate of the
trained policy is measured on the same footing for every form. Per run
the JSON records the first feasible success (env steps, wall time,
circuit, exact ``d`` and ``w*``), every success, the success rate over
the updates after the first success and over the last 25 updates, the
distribution of exact ``d`` and ``w*`` over the terminal states of the
last 25 updates, a probe of the trained policy (256 sampled episodes from
a fresh reset), and the per-update history. ``report`` prints the tables of
``experiments/ldpc_scaling/REPORT.md`` from ``results/ldpc_scaling/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from .. import graphs
from ..env import CodeDiscoveryEnv
from ..ldpc import LDPC_FORMS, LdpcObjective, sparsest_basis_stats
from ..reward import batched_distance, low_weight_stabilizer_penalty
from ..symplectic import rref
from .scaling import describe_code, probe_policy, replay_gates

DEFAULT_OUT = Path("results/ldpc_scaling")
EXACT_GENERATORS = 15
TERMINAL_WINDOW = 25  # updates over which the terminal-state statistics of the report are summed

TARGETS = {
    # name: (n, k, graph spec, gates), d = 3 and w_max = 4 throughout
    "n12_k2_all_to_all": (12, 2, "all_to_all:12", ("H", "S", "CNOT")),
    "n12_k2_square": (12, 2, "square:3x4", ("H", "CNOT")),
    "n16_k4_all_to_all": (16, 4, "all_to_all:16", ("H", "S", "CNOT")),
    "n16_k4_square": (16, 4, "square:4x4", ("H", "CNOT")),
}


# --------------------------------------------------------------------------
# environment with terminal-state statistics
# --------------------------------------------------------------------------
class StatsEnv(CodeDiscoveryEnv):
    """``CodeDiscoveryEnv`` that accumulates, between ``flush`` calls, the
    exact distance (capped at ``d``), the exact ``w*``, the idle-qubit count,
    and the feasibility of every terminal state."""

    def __init__(self, *args, d: int, w_max: int, **kw):
        super().__init__(*args, **kw)
        self.d = int(d)
        self.w_max = int(w_max)
        # optional replacement for the numpy exact distance of terminal
        # states: ``f(tab) -> (B,) int`` capped at ``d`` (a torch kernel on a
        # device, for the cells above n = 16)
        self.distance_fn = None
        self.flush()

    def terminal_distance(self, ft: np.ndarray) -> np.ndarray:
        if self.distance_fn is not None:
            return np.minimum(np.asarray(self.distance_fn(ft), dtype=np.int64), self.d)
        return np.minimum(batched_distance(ft, self.k, self.d - 1), self.d)

    def flush(self) -> dict:
        rec = getattr(self, "_acc", None)
        self._acc = dict(
            terminal_d=np.zeros(self.d + 1, dtype=np.int64),
            terminal_w_star=np.zeros(self.n + 1, dtype=np.int64),
            terminal_joint=np.zeros((self.d + 1, self.n + 1), dtype=np.int64),
            terminal_idle=0,
            terminal_feasible_codes=0,
            n_terminal=0,
        )
        return rec

    def step(self, actions):
        out = super().step(actions)
        info = out[4]
        if "final_tableau" in info:
            ft = info["final_tableau"]
            dist = self.terminal_distance(ft)
            st = sparsest_basis_stats(ft, self.k, exact_max_generators=EXACT_GENERATORS)
            ws = np.minimum(st["w_star"], self.n)
            idle = np.rint(-low_weight_stabilizer_penalty(ft, self.k, 1) * 3 * self.n).astype(int)
            acc = self._acc
            acc["terminal_d"] += np.bincount(dist, minlength=self.d + 1)
            acc["terminal_w_star"] += np.bincount(ws, minlength=self.n + 1)
            np.add.at(acc["terminal_joint"], (dist, ws), 1)
            acc["terminal_idle"] += int(idle.sum())
            acc["terminal_feasible_codes"] += int(((dist >= self.d) & (ws <= self.w_max) & (idle == 0)).sum())
            acc["n_terminal"] += int(ft.shape[0])
        return out


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def run_form(
    form: str,
    target: str,
    seed: int,
    minutes: float = 20.0,
    threads: int = 2,
    d: int = 3,
    w_max: int = 4,
    max_steps: int = 40,
    batch: int = 256,
    post_success_updates: int = 0,
    schedule_rate: float = 0.5,
    schedule_window: int = 10,
    scale: float = 0.25,
    policy: str = "equivariant",
    max_env_steps: int = 10**9,
    verbose: bool = False,
) -> dict:
    """One PPO run of ``form`` on ``target`` until ``max_env_steps`` or the
    wall cap ``minutes``, whichever comes first (with ``post_success_updates
    > 0`` the run also stops that many updates after the first feasible
    success). The comparison uses the fixed budget so that the success rate
    of the trained policy is measured on the same footing for every form."""
    from ..agents.ppo import PPOAgent, PPOConfig

    if form not in LDPC_FORMS:
        raise ValueError(f"form must be one of {LDPC_FORMS}")
    n, k, graph_spec, gates = TARGETS[target]
    objective = LdpcObjective(n, k, d, w_max, form=form, scale=scale, exact_generators=EXACT_GENERATORS)
    env = StatsEnv(
        n=n, k=k, graph=graphs.from_spec(graph_spec), batch_size=batch, max_steps=max_steps, gates=gates,
        reward_fn=objective, success_fn=objective.success, seed=seed, auto_reset=True, d=d, w_max=w_max,
    )
    agent = PPOAgent(env, config=PPOConfig(max_threads=threads, policy=policy), seed=seed)
    t0 = time.perf_counter()
    state = {"first_feasible": None, "first_feasible_update": None, "first_any": None, "switch": None, "feasible": 0, "distinct": {}}
    recent: list[float] = []
    history: list[dict] = []

    def cb(agent, new):
        rec = agent.history[-1]
        acc = env.flush()
        nt = max(acc["n_terminal"], 1)
        rec["terminal_d"] = acc["terminal_d"].tolist()
        rec["terminal_w_star"] = acc["terminal_w_star"].tolist()
        rec["terminal_joint"] = acc["terminal_joint"].tolist()
        rec["terminal_idle_mean"] = acc["terminal_idle"] / nt
        rec["terminal_feasible_codes"] = acc["terminal_feasible_codes"]
        rec["n_terminal"] = acc["n_terminal"]
        rec["ldpc_active"] = objective.active
        rec["exact_evals"] = objective.exact_evals
        for fc in new:
            if state["first_any"] is None:
                state["first_any"] = dict(fc.to_dict(), wall_time_s=rec["wall_time_s"])
            if objective.feasible(fc.tableau[None], k)[0]:
                state["feasible"] += 1
                key = rref(fc.tableau[n + k :, : 2 * n])[0].tobytes()
                state["distinct"][key] = state["distinct"].get(key, 0) + 1
                if state["first_feasible"] is None:
                    state["first_feasible"] = dict(fc.to_dict(), wall_time_s=rec["wall_time_s"])
                    state["first_feasible_update"] = agent.updates
        history.append(rec)
        if verbose:
            print(
                f"upd {rec['update']:5d} steps {rec['env_steps']:9d} r {rec['mean_reward']:+.4f} ent {rec['entropy']:.3f} "
                f"succ {rec['success_rate']:.3f} term_d {rec['terminal_d']} feas {rec['terminal_feasible_codes']} "
                f"active {int(objective.active)} t {rec['wall_time_s']:.0f}s",
                flush=True,
            )
        if form == "schedule" and not objective.active:
            recent.append(rec["success_rate"])
            if len(recent) >= schedule_window and float(np.mean(recent[-schedule_window:])) >= schedule_rate:
                objective.activate()
                state["switch"] = {"env_steps": agent.env_steps, "update": agent.updates, "wall_time_s": rec["wall_time_s"]}
                if verbose:
                    print("    schedule: LDPC term switched on", flush=True)
            return False
        fu = state["first_feasible_update"]
        return post_success_updates > 0 and fu is not None and agent.updates >= fu + post_success_updates

    agent.train(max_env_steps, max_time_s=minutes * 60.0, stop_on_success=False, callback=cb, verbose=False)
    wall = time.perf_counter() - t0
    first = state["first_feasible"]
    if first is not None:
        tab = replay_gates([tuple(g) for g in first["gates"]], n, k)
        first["code"] = describe_code(tab, k)
    fu = state["first_feasible_update"]
    tail = [h for h in history if h["update"] > fu] if fu is not None else []
    last = history[-TERMINAL_WINDOW:]
    probe = probe_policy(agent, env, objective, d)
    return {
        "kind": "ldpc_form",
        "form": form,
        "objective": objective.describe(),
        "target": target,
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
        "policy": agent.policy,
        "ppo_config": agent.config.to_dict(),
        "num_parameters": agent.num_parameters(),
        "minutes_cap": minutes,
        "post_success_updates": post_success_updates,
        "schedule": {"rate": schedule_rate, "window": schedule_window} if form == "schedule" else None,
        "switch": state["switch"],
        "wall_time_s": round(wall, 1),
        "env_steps": agent.env_steps,
        "episodes": agent.episodes,
        "updates": agent.updates,
        "success": first is not None,
        "first_success": first,
        "first_any_success": state["first_any"],
        "successes_env": len(agent.successes),
        "feasible_successes": state["feasible"],
        "distinct_feasible_codes": len(state["distinct"]),
        "post_success_rate": float(np.mean([h["success_rate"] for h in tail])) if tail else None,
        "last_window_success_rate": float(np.mean([h["success_rate"] for h in last])) if last else None,
        "max_env_steps": max_env_steps,
        "terminal_last": summarize_terminal(last, d, w_max),
        "final_policy_probe": probe,
        "final": history[-1] if history else None,
        "platform": platform.platform(),
        "history": history,
    }


def summarize_terminal(hist: list[dict], d: int, w_max: int) -> dict:
    """Terminal-state statistics summed over ``hist``: distance histogram,
    ``w*`` histogram, joint counts, feasible-code count, idle mean."""
    if not hist:
        return {"n_terminal": 0}
    td = np.sum([h["terminal_d"] for h in hist], axis=0)
    tw = np.sum([h["terminal_w_star"] for h in hist], axis=0)
    tj = np.sum([h["terminal_joint"] for h in hist], axis=0)
    nt = int(td.sum())
    return {
        "updates": len(hist),
        "n_terminal": nt,
        "d_hist": td.tolist(),
        "w_star_hist": tw.tolist(),
        "joint": tj.tolist(),
        "feasible_codes": int(sum(h["terminal_feasible_codes"] for h in hist)),
        "at_target_d": int(td[d:].sum()),
        "w_star_le_w_max_frac": float(tw[: w_max + 1].sum() / max(nt, 1)),
        "idle_mean": float(np.sum([h["terminal_idle_mean"] * sum(h["terminal_d"]) for h in hist]) / max(nt, 1)),
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def load_all(out_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(out_dir.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        r["_file"] = str(p)
        recs.append(r)
    return [r for r in recs if r.get("kind") == "ldpc_form"]


def _fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def _mode(hist: list[int]) -> str:
    if not hist or sum(hist) == 0:
        return "n/a"
    return str(max(range(len(hist)), key=hist.__getitem__))


def _pct(a, b) -> str:
    return f"{100.0 * a / b:.0f}" if b else "n/a"


FORM_ORDER = {f: i for i, f in enumerate(("penalty", "unnormalized", "final", "schedule"))}


def _form_key(r: dict):
    return (FORM_ORDER.get(r["form"], 9), r.get("objective", {}).get("scale", 0.25))


def run_table(recs: list[dict]) -> str:
    """One row per run."""
    lines = [
        "| target | graph | form | seed | success | env steps to first success | wall to first success (s) | env steps in run | success rate after first success | success rate, last 25 updates | feasible successes (distinct codes) | first code: gates, d, w*, degree, CSS | terminal states of the last 25 updates: n, at d = 3 (percent), w* <= 4 (percent), feasible codes (percent), mode of w*, mean idle | switch (env steps) | probe: 256 sampled episodes, at d = 3, feasible codes, mode of w* | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(recs, key=lambda r: (r["n"], r["graph"], _form_key(r), r["seed"])):
        fs = r["first_success"]
        code = fs.get("code") if fs else None
        tl = r["terminal_last"]
        nt = tl.get("n_terminal", 0)
        term = (
            f"{nt:,}, {_pct(tl.get('at_target_d', 0), nt)}, {_pct(sum(tl['w_star_hist'][: r['w_max'] + 1]), nt) if nt else 'n/a'}, "
            f"{_pct(tl.get('feasible_codes', 0), nt)}, {_mode(tl.get('w_star_hist', []))}, {_fmt(tl.get('idle_mean'), 1)}"
            if nt
            else "n/a"
        )
        probe = (r.get("final_policy_probe") or {}).get("sampled")
        ptxt = (
            f"{sum(probe['d_hist'][r['d']:])}, {probe['feasible_clean_target_states']}, {_mode(probe['w_star_hist'])}"
            if probe
            else "n/a"
        )
        sw = r.get("switch")
        lines.append(
            f"| [[{r['n']}, {r['k']}, {r['d']}]] | {r['graph']} | {form_label(r)} | {r['seed']} | {'yes' if r['success'] else 'no'} "
            f"| {_fmt(fs['env_steps']) if fs else 'not reached'} | {_fmt(fs['wall_time_s'], 0) if fs else 'n/a'} | {r['env_steps']:,} "
            f"| {_fmt(r['post_success_rate'], 3)} | {_fmt(r.get('last_window_success_rate'), 3)} | {r['feasible_successes']:,} ({r['distinct_feasible_codes']:,}) "
            f"| {(str(fs['circuit_length']) + ', ' + str(code['d']) + ', ' + str(code['w_star']) + ', ' + str(code['max_degree']) + ', ' + ('yes' if code['is_css'] else 'no')) if code else 'n/a'} "
            f"| {term} | {_fmt(sw['env_steps']) if sw else ('never' if r['form'] == 'schedule' else 'n/a')} | {ptxt} | `{Path(r['_file']).name}` |"
        )
    return "\n".join(lines)


def summary_table(recs: list[dict]) -> str:
    """One row per (target, form): seeds succeeded, steps to first success,
    post-success rate, and the terminal-state distribution pooled over seeds."""
    lines = [
        "| target | graph | form | seeds succeeded | env steps to first success (per seed) | success rate over the last 25 updates (per seed) | feasible successes (per seed) | terminal states, last 25 updates of every seed: at d = 3 (percent), w* <= 4 (percent), feasible codes (percent) | terminal w* histogram (w* = 1, 2, ...) | terminal d histogram (d = 1, 2, 3) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        groups.setdefault((r["n"], r["k"], r["d"], r["graph"], form_label(r)), []).append(r)
    for (n, k, d, graph, form), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][3], _form_key(kv[1][0]))):
        rs = sorted(rs, key=lambda r: r["seed"])
        succ = sum(r["success"] for r in rs)
        steps = ", ".join(_fmt(r["first_success"]["env_steps"]) if r["first_success"] else "none" for r in rs)
        rates = ", ".join(_fmt(r.get("last_window_success_rate"), 3) if r.get("last_window_success_rate") is not None else "n/a" for r in rs)
        feas = ", ".join(f"{r['feasible_successes']:,}" for r in rs)
        tl = [r["terminal_last"] for r in rs if r["terminal_last"].get("n_terminal")]
        nt = sum(t["n_terminal"] for t in tl)
        if nt:
            at_d = sum(t["at_target_d"] for t in tl)
            wle = sum(sum(t["w_star_hist"][: rs[0]["w_max"] + 1]) for t in tl)
            feas = sum(t["feasible_codes"] for t in tl)
            wh = np.sum([t["w_star_hist"] for t in tl], axis=0)
            dh = np.sum([t["d_hist"] for t in tl], axis=0)
            term = f"{_pct(at_d, nt)}, {_pct(wle, nt)}, {_pct(feas, nt)}"
            whist = str(wh[1 : max(np.flatnonzero(wh)) + 1].tolist()) if wh.any() else "n/a"
            dhist = str(dh[1:].tolist())
        else:
            term, whist, dhist = "n/a", "n/a", "n/a"
        lines.append(
            f"| [[{n}, {k}, {d}]] | {graph} | {form} | {succ} of {len(rs)} | {steps} | {rates} | {feas} | {term} | {whist} | {dhist} |"
        )
    return "\n".join(lines)


def sources_table(recs: list[dict]) -> str:
    lines = ["| file | contents |", "|---|---|"]
    for r in sorted(recs, key=lambda r: r["_file"]):
        lines.append(
            f"| `results/ldpc_scaling/{Path(r['_file']).name}` | {form_label(r)} form, [[{r['n']}, {r['k']}, {r['d']}]] w_max {r['w_max']} on {r['graph']}, "
            f"seed {r['seed']}, {r['env_steps']:,} env steps, {r['wall_time_s']:.0f} s, {'success' if r['success'] else 'no success'} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
FORM_COLORS = {"penalty": "#777777", "unnormalized": "#4477AA", "final": "#EE6677", "schedule": "#228833"}
SEED_STYLES = ["-", "--", ":"]
FIG_WINDOW = 25  # updates


def _windowed_fraction(history: list[dict], key: str, window: int = FIG_WINDOW):
    """Fraction of terminal states with property ``key`` over a trailing
    window of updates (``key`` is ``"at_d"`` for the target distance or
    ``"feasible"``), with the env steps of every update."""
    steps = np.array([h["env_steps"] for h in history], dtype=float) / 1e6
    tot = np.array([sum(h["terminal_d"]) for h in history], dtype=float)
    if key == "at_d":
        num = np.array([h["terminal_d"][-1] for h in history], dtype=float)
    else:
        num = np.array([h["terminal_feasible_codes"] for h in history], dtype=float)
    c_num = np.cumsum(np.insert(num, 0, 0.0))
    c_tot = np.cumsum(np.insert(tot, 0, 0.0))
    lo = np.maximum(np.arange(len(history)) + 1 - window, 0)
    frac = (c_num[1:] - c_num[lo]) / np.maximum(c_tot[1:] - c_tot[lo], 1.0)
    return steps, frac


def figure_terminal(recs: list[dict], stem: Path, key: str = "at_d", scale: float = 0.25):
    """One panel per target; the fraction of terminal states at the target
    distance (``key="at_d"``) or that are feasible codes (``"feasible"``)
    over a trailing 25-update window against env steps, one color per
    form (runs at penalty ``scale``), one line style per seed."""
    from . import figstyle

    plt = figstyle.apply()
    recs = [r for r in recs if r.get("objective", {}).get("scale", 0.25) == scale]
    targets = [t for t in TARGETS if any(r["target"] == t for r in recs)]
    if not targets:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(figstyle.WIDTH, figstyle.HEIGHT_DOUBLE), sharey=True)
    axes = axes.ravel()
    seen = set()
    for ax, target in zip(axes, targets):
        n, k, graph, _ = TARGETS[target]
        sw_marked = False
        for r in sorted((r for r in recs if r["target"] == target), key=lambda r: (FORM_ORDER.get(r["form"], 9), r["seed"])):
            steps, frac = _windowed_fraction(r["history"], key)
            label = r["form"] if r["form"] not in seen else None
            seen.add(r["form"])
            ax.plot(steps, frac, SEED_STYLES[r["seed"] % 3], color=FORM_COLORS.get(r["form"], "#000000"), label=label, lw=figstyle.THIN)
            if r.get("switch"):
                ax.axvline(r["switch"]["env_steps"] / 1e6, color=FORM_COLORS["schedule"], lw=figstyle.THIN, ls=SEED_STYLES[r["seed"] % 3], alpha=0.5)
                sw_marked = True
        ax.set_title(f"[[{n}, {k}, 3]] on {graph}" + (" (vertical: schedule switch)" if sw_marked else ""))
        ax.set_xlabel("env steps (millions)")
        ax.set_ylim(0, 1.02)
    for ax in axes[len(targets):]:
        ax.set_visible(False)
    ylabel = "fraction of terminal states at d = 3" if key == "at_d" else "feasible fraction of terminal states"
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    axes[0].legend(loc="upper left", title="form (line style: seed 0 solid, 1 dashed, 2 dotted)")
    for ax, letter in zip(axes, "abcd"):
        figstyle.panel_label(ax, letter)
    fig.tight_layout()
    return figstyle.save(fig, stem)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run_name(form: str, target: str, seed: int, scale: float = 0.25) -> str:
    tag = "" if scale == 0.25 else f"_scale{scale:g}"
    return f"{form}{tag}_{target}_seed{seed}"


def form_label(r: dict) -> str:
    """Form name with the penalty scale when it is not the regression's 0.25."""
    scale = r.get("objective", {}).get("scale", 0.25)
    return r["form"] if scale == 0.25 else f"{r['form']} (scale {scale:g})"


def main(argv=None):
    ap = argparse.ArgumentParser(description="LDPC constraint forms above n = 9.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="one PPO run of one form on one target")
    p.add_argument("--form", choices=LDPC_FORMS, required=True)
    p.add_argument("--target", choices=list(TARGETS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max-env-steps", type=int, default=1_500_000)
    p.add_argument("--post-success-updates", type=int, default=0, help="> 0: stop that many updates after the first feasible success")
    p.add_argument("--schedule-rate", type=float, default=0.5)
    p.add_argument("--schedule-window", type=int, default=10)
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--policy", choices=("mlp", "equivariant", "auto"), default="equivariant")
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("report", help="print the tables from results/ldpc_scaling and draw the figures")
    p.add_argument("--dir", default=str(DEFAULT_OUT))
    p.add_argument("--figures", default="experiments/ldpc_scaling/figures", help="directory for the figures ('' to skip)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        rec = run_form(
            args.form, args.target, args.seed, args.minutes, args.threads, max_steps=args.max_steps, batch=args.batch,
            post_success_updates=args.post_success_updates, schedule_rate=args.schedule_rate,
            schedule_window=args.schedule_window, scale=args.scale, policy=args.policy,
            max_env_steps=args.max_env_steps, verbose=args.verbose,
        )
        out = Path(args.out or DEFAULT_OUT / f"{run_name(args.form, args.target, args.seed, args.scale)}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=_json_default)
        fs = rec["first_success"]
        print(
            f"wrote {out}: success {rec['success']} first {fs['env_steps'] if fs else None} "
            f"post-rate {rec['post_success_rate']} env_steps {rec['env_steps']:,} wall {rec['wall_time_s']}s"
        )
        return 0
    recs = load_all(Path(args.dir))
    print("## Summary per form\n")
    print(summary_table(recs))
    print("\n## Runs\n")
    print(run_table(recs))
    print("\n## Sources\n")
    print(sources_table(recs))
    if args.figures:
        for key in ("at_d", "feasible"):
            out = figure_terminal(recs, Path(args.figures) / f"terminal_{key}", key)
            if out:
                print("figure:", out[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
