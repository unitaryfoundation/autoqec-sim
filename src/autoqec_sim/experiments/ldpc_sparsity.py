# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Making ``w* <= w_max`` a property of the trained policy.

The comparison of ``experiments/ldpc_scaling/REPORT.md`` left the sparsity
half of the LDPC constraint open: the ``unnormalized`` form learns
distance-3 codes at ``n = 12`` to ``16`` but its policies sit at ``w*`` one
to four above the bound, and the codes within the bound were single
episodes. This module compares, against ``unnormalized`` at equal budget
in env steps, three ways of pricing or enforcing the bound, on
``[[12, 2, 3]]`` (all-to-all and ``square:3x4``) and ``[[16, 4, 3]]``
(``square:4x4``) with ``w_max = 4``, equivariant policy, three seeds:

``shaped`` (``--c``)
    a dense per-step term ``-c max(0, w* - w_max)`` in reward units, the
    greedy bound on ordinary states and the exact ``w*`` on candidates
    (``LdpcObjective(form="shaped", shaping=c)``);
``terminal``
    a count-unit terminal signal: any candidate (every error detected, no
    idle qubit) ends the episode with reward ``1 - h / (n - k)``, ``h``
    the number of generators of the sparsest basis with weight above
    ``w_max`` (``LdpcObjective(form="terminal")``);
``mask``
    an action-space constraint: a two-qubit gate that would take the
    greedy ``w*`` bound above ``w_max`` is masked out of the policy's
    logits (:func:`autoqec_sim.ldpc.greedy_action_mask`), so the
    trajectory never leaves the sparse set; the reward is the
    ``unnormalized`` form's;
``mask_terminal``
    the same mask on top of the ``terminal`` reward: every candidate the
    mask admits has ``h = 0``, so the terminal bonus is 1 for every
    feasible code and the path reward carries no sparsity term.

The measured quantity is the fraction of terminal states at the target
distance whose exact ``w*`` is within the bound, over the last 25 updates
of every run (2,560 terminal states) and per update along the run. The
``unnormalized`` form is also run at 3M env steps on one cell, the
longer-budget question the earlier report left open. ``report`` prints
the tables of ``experiments/ldpc_sparsity/REPORT.md`` from
``experiments/ldpc_sparsity/runs/`` and draws its figures.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
from pathlib import Path

import numpy as np

from .. import graphs
from ..code import StabilizerCode
from ..ldpc import ExactMaskCache, LdpcObjective, greedy_action_mask, heavy_generator_count, sparsest_basis_stats
from ..reward import batched_distance
from ..symplectic import rref
from .ldpc_scaling import EXACT_GENERATORS, TERMINAL_WINDOW, StatsEnv, _json_default, summarize_terminal
from .scaling import describe_code, probe_policy, replay_gates

DEFAULT_OUT = Path("experiments/ldpc_sparsity/runs")
VARIANTS = ("unnormalized", "shaped", "terminal", "mask", "mask_terminal")
MASKED = ("mask", "mask_terminal")  # variants with the action mask
MASK_METHODS = ("basis", "tableau", "exact")  # ldpc.greedy_action_mask methods; "exact" is the exact feasibility test on the pairs the greedy bound rejects
DEFAULT_BATCH = 256
# name: (n, k, d, graph spec, gates, episode length); w_max = 4 throughout.
# The first three are the cells of experiments/ldpc_scaling (d = 3, 40 gates);
# the all-to-all cells from n = 16 on are the scaling-in-n question of the
# ion-chain connectivity (batch 1024 on a GPU). Above n - k = 15 generators
# the exact w* enumeration is out of reach and the greedy bound decides
# feasibility (recorded as ``w_star_test`` in the JSON).
CELL_SPECS = {
    "n12_k2_all_to_all": (12, 2, 3, "all_to_all:12", ("H", "S", "CNOT"), 40),
    "n12_k2_square": (12, 2, 3, "square:3x4", ("H", "CNOT"), 40),
    "n16_k4_square": (16, 4, 3, "square:4x4", ("H", "CNOT"), 40),
    "n16_k4_all_to_all": (16, 4, 3, "all_to_all:16", ("H", "S", "CNOT"), 40),
    "n20_k4_all_to_all": (20, 4, 3, "all_to_all:20", ("H", "S", "CNOT"), 50),
    "n25_k1_d4_all_to_all": (25, 1, 4, "all_to_all:25", ("H", "S", "CNOT"), 80),
    "n25_k1_d5_all_to_all": (25, 1, 5, "all_to_all:25", ("H", "S", "CNOT"), 80),
}
CELLS = tuple(CELL_SPECS)
TARGETS = {name: (n, k, graph, gates) for name, (n, k, d, graph, gates, _) in CELL_SPECS.items()}  # (n, k, graph, gates), as ldpc_scaling.TARGETS
DEFAULT_BUDGET = 1_500_000
SHAPING_SWEEP = (0.05, 0.1, 0.2)
PROBE_MAX_ERRORS = 20_000  # above this many Knill-Laflamme Paulis the numpy probe of the trained policy is skipped


# --------------------------------------------------------------------------
# environment: terminal statistics plus the optional greedy action mask
# --------------------------------------------------------------------------
class SparsityEnv(StatsEnv):
    """:class:`StatsEnv` that also accumulates the heavy-generator count of
    every terminal state at the target distance and, with ``mask_w_max``
    set, exposes :meth:`action_mask` (picked up by the PPO agent's
    ``EnvAdapter``) forbidding two-qubit gates that would take the greedy
    ``w*`` bound above ``mask_w_max``."""

    def __init__(self, *args, mask_w_max: int | None = None, mask_method: str = "basis", **kw):
        self.mask_w_max = None if mask_w_max is None else int(mask_w_max)
        self.mask_method = mask_method
        self.mask_calls = 0
        self.mask_time_s = 0.0
        # whole-run counters of the exact method (ldpc.greedy_action_mask stats)
        self.mask_totals = dict(pairs=0, greedy_rejected=0, exact_tested=0, exact_unique=0, exact_admitted=0, cache_hits=0, cache_misses=0, greedy_s=0.0, exact_s=0.0)
        self.mask_method_effective = None
        self.mask_cache = ExactMaskCache() if mask_method == "exact" else None
        super().__init__(*args, **kw)

    def flush(self) -> dict:
        rec = super().flush()
        self._acc["terminal_heavy_at_d"] = np.zeros(self.n - self.k + 1, dtype=np.int64)
        self._acc["mask_two_masked"] = 0
        self._acc["mask_two_greedy_masked"] = 0
        self._acc["mask_two_total"] = 0
        self._acc["mask_exact_tested"] = 0
        self._acc["mask_exact_admitted"] = 0
        return rec

    def action_mask(self) -> np.ndarray | None:
        if self.mask_w_max is None:
            return None
        st: dict = {}
        t0 = time.perf_counter()
        mask = greedy_action_mask(
            self.tab, self.k, self.action_table, self.mask_w_max, self.mask_method,
            exact_max_generators=EXACT_GENERATORS, cache=self.mask_cache, stats=st,
        )
        self.mask_time_s += time.perf_counter() - t0
        self.mask_calls += 1
        two = self.action_table.q2 >= 0
        self._acc["mask_two_masked"] += int((~mask[:, two]).sum())
        self._acc["mask_two_total"] += int(mask[:, two].size)
        if st:
            self.mask_method_effective = st["method"]
            self._acc["mask_two_greedy_masked"] += st["greedy_rejected"]
            self._acc["mask_exact_tested"] += st["exact_tested"]
            self._acc["mask_exact_admitted"] += st["exact_admitted"]
            for key in self.mask_totals:
                self.mask_totals[key] += st.get(key, 0)
        return mask

    def mask_record(self, last: list[dict]) -> dict:
        """The ``mask`` entry of the run's JSON: method requested and run,
        calls and seconds, masked fractions over the last updates (final
        mask and greedy pre-filter), and the exact method's counters."""
        tot = self.mask_totals
        return {
            "method": self.mask_method,
            "method_effective": self.mask_method_effective,
            "calls": self.mask_calls,
            "time_s": round(self.mask_time_s, 1),
            "greedy_time_s": round(tot["greedy_s"], 1),
            "exact_time_s": round(tot["exact_s"], 1),
            "two_masked_frac_last": float(np.mean([h["mask_two_masked_frac"] for h in last])) if last else None,
            "greedy_two_masked_frac_last": float(np.mean([h["mask_two_greedy_masked_frac"] for h in last])) if last and "mask_two_greedy_masked_frac" in last[-1] else None,
            "pairs": tot["pairs"],
            "greedy_rejected": tot["greedy_rejected"],
            "exact_tested": tot["exact_tested"],
            "exact_unique": tot["exact_unique"],
            "exact_admitted": tot["exact_admitted"],
            "exact_tested_per_call": (tot["exact_tested"] / self.mask_calls) if self.mask_calls else None,
            "cache_hits": tot["cache_hits"],
            "cache_misses": tot["cache_misses"],
        }

    def step(self, actions):
        out = super().step(actions)
        info = out[4]
        if "final_tableau" in info:
            ft = info["final_tableau"]
            dist = np.minimum(batched_distance(ft, self.k, self.d - 1), self.d)
            at_d = dist >= self.d
            if at_d.any():
                h = heavy_generator_count(ft[at_d], self.k, self.w_max, EXACT_GENERATORS)
                self._acc["terminal_heavy_at_d"] += np.bincount(h, minlength=self.n - self.k + 1)
        return out


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def make_objective(variant: str, n: int, k: int, d: int, w_max: int, c: float, device=None, chunk: int = 1 << 16) -> LdpcObjective:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    form = {"unnormalized": "unnormalized", "shaped": "shaped", "terminal": "terminal", "mask": "unnormalized", "mask_terminal": "terminal"}[variant]
    return LdpcObjective(n, k, d, w_max, form=form, shaping=c, exact_generators=EXACT_GENERATORS, device=device, chunk=chunk)


class Timed:
    """Callable wrapper accumulating wall time and call count."""

    def __init__(self, fn):
        self.fn = fn
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, *args, **kw):
        t = time.perf_counter()
        out = self.fn(*args, **kw)
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return out


def resolve_device(device: str):
    """``torch.device`` for ``"cpu"`` or ``"cuda"``; refuses ``cuda`` when
    torch has no CUDA device (as ``gpu_scale.resolve_device``)."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")
    return dev


def describe_code_capped(tab: np.ndarray, k: int, d: int) -> dict:
    """``scaling.describe_code`` for ``n <= 20``; above that the exact
    distance is out of reach, so ``d`` is the exact distance capped at
    ``d + 1`` (Paulis of weight up to ``d`` enumerated) and ``w*`` is the
    greedy bound, both marked."""
    n = (tab.shape[-1] - 1) // 2
    if n <= 20:
        return describe_code(tab, k)
    code = StabilizerCode.from_tableau(tab[None], k)
    st = sparsest_basis_stats(tab[None], k, exact_max_generators=EXACT_GENERATORS)
    return {
        "stabilizers": code.stabilizer_strings(),
        "d": int(batched_distance(tab[None], k, d)[0]),
        "d_capped_at": d + 1,
        "is_css": bool(code.is_css()),
        "w_star": int(st["w_star"][0]),
        "w_star_test": "exact" if st["exact"] else "greedy bound",
        "max_degree": int(st["max_degree"][0]),
    }


def run_variant(
    variant: str,
    cell: str,
    seed: int,
    c: float = 0.1,
    max_env_steps: int = DEFAULT_BUDGET,
    minutes: float = 180.0,
    threads: int = 4,
    d: int | None = None,
    w_max: int = 4,
    max_steps: int | None = None,
    batch: int = 256,
    policy: str = "equivariant",
    device: str = "cpu",
    reward_backend: str | None = None,
    chunk: int | None = None,
    mask_method: str = "basis",
    verbose: bool = False,
) -> dict:
    """One PPO run of ``variant`` on ``cell`` for ``max_env_steps`` env steps
    (the wall cap ``minutes`` is a safety net, not the budget). ``device``
    is the torch device of the policy and the PPO update; the reward's
    base terms run on it too when ``reward_backend`` is ``"torch"`` (the
    default on cuda; ``"numpy"`` is the default on cpu and the path of the
    earlier runs). The greedy action mask and the ``w*`` terms stay numpy
    on the host. ``d`` and ``max_steps`` default to the cell's."""
    from ..agents.ppo import PPOAgent, PPOConfig

    if cell not in CELLS:
        raise ValueError(f"cell must be one of {CELLS}, got {cell!r}")
    n, k, cell_d, graph_spec, gates, cell_steps = CELL_SPECS[cell]
    d = cell_d if d is None else int(d)
    max_steps = cell_steps if max_steps is None else int(max_steps)
    dev = resolve_device(device)
    backend = reward_backend or ("torch" if dev.type == "cuda" else "numpy")
    if backend not in ("numpy", "torch"):
        raise ValueError("reward_backend must be numpy or torch")
    chunk = int(chunk or (8192 if dev.type == "cuda" else 1 << 16))
    objective = make_objective(variant, n, k, d, w_max, c, device=dev if backend == "torch" else None, chunk=chunk)
    reward_timer = Timed(objective)
    env = SparsityEnv(
        n=n, k=k, graph=graphs.from_spec(graph_spec), batch_size=batch, max_steps=max_steps, gates=gates,
        reward_fn=reward_timer, success_fn=objective.success, seed=seed, auto_reset=True, d=d, w_max=w_max,
        mask_w_max=w_max if variant in MASKED else None, mask_method=mask_method,
    )
    if backend == "torch":
        from ..backends import torch_backend as tb

        env.distance_fn = lambda ft: tb.batched_distance(tb.from_numpy(ft, dev), k, d - 1, chunk).cpu().numpy()
    agent = PPOAgent(env, config=PPOConfig(max_threads=threads, policy=policy), seed=seed, device=dev)
    update_timer = Timed(agent.update)
    agent.update = update_timer
    t0 = time.perf_counter()
    state = {"first_feasible": None, "first_feasible_update": None, "first_any": None, "feasible": 0, "distinct": {}}
    history: list[dict] = []

    def cb(agent, new):
        rec = agent.history[-1]
        acc = env.flush()
        nt = max(acc["n_terminal"], 1)
        joint = acc["terminal_joint"]
        rec["terminal_d"] = acc["terminal_d"].tolist()
        rec["terminal_w_star"] = acc["terminal_w_star"].tolist()
        rec["terminal_joint"] = joint.tolist()
        rec["terminal_heavy_at_d"] = acc["terminal_heavy_at_d"].tolist()
        rec["terminal_at_d"] = int(joint[d:].sum())
        rec["terminal_inside_at_d"] = int(joint[d:, : w_max + 1].sum())
        rec["terminal_idle_mean"] = acc["terminal_idle"] / nt
        rec["terminal_feasible_codes"] = acc["terminal_feasible_codes"]
        rec["n_terminal"] = acc["n_terminal"]
        rec["exact_evals"] = objective.exact_evals
        if variant in MASKED:
            rec["mask_two_masked_frac"] = acc["mask_two_masked"] / max(acc["mask_two_total"], 1)
            if mask_method == "exact":
                rec["mask_two_greedy_masked_frac"] = acc["mask_two_greedy_masked"] / max(acc["mask_two_total"], 1)
                rec["mask_exact_tested"] = acc["mask_exact_tested"]
                rec["mask_exact_admitted"] = acc["mask_exact_admitted"]
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
                f"succ {rec['success_rate']:.3f} term_d {rec['terminal_d']} at_d {rec['terminal_at_d']} "
                f"inside {rec['terminal_inside_at_d']} feas {rec['terminal_feasible_codes']} t {rec['wall_time_s']:.0f}s",
                flush=True,
            )
        return False

    agent.train(max_env_steps, max_time_s=minutes * 60.0, stop_on_success=False, callback=cb, verbose=False)
    wall = time.perf_counter() - t0
    first = state["first_feasible"]
    if first is not None:
        tab = replay_gates([tuple(g) for g in first["gates"]], n, k)
        first["code"] = describe_code_capped(tab, k, d)
    last = history[-TERMINAL_WINDOW:]
    if objective.num_errors <= PROBE_MAX_ERRORS:
        probe = probe_policy(agent, env, objective, d)
    else:
        probe = {"skipped": f"numpy probe skipped above {PROBE_MAX_ERRORS:,} Knill-Laflamme Paulis ({objective.num_errors:,} here)"}
    import torch

    rec = {
        "kind": "ldpc_sparsity",
        "variant": variant,
        "c": c if variant == "shaped" else None,
        "form": objective.form,
        "objective": objective.describe(),
        "w_star_test": objective.describe()["w_star_test"],
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "reward_backend": backend,
        "chunk": chunk,
        "reward_time_s": round(reward_timer.seconds, 1),
        "reward_calls": reward_timer.calls,
        "update_time_s": round(update_timer.seconds, 1),
        "cell": cell,
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
        "budget_env_steps": max_env_steps,
        "minutes_cap": minutes,
        "wall_time_s": round(wall, 1),
        "env_steps": agent.env_steps,
        "env_steps_per_s": round(agent.env_steps / max(wall, 1e-9), 1),
        "episodes": agent.episodes,
        "updates": agent.updates,
        "success": first is not None,
        "first_success": first,
        "first_any_success": state["first_any"],
        "successes_env": len(agent.successes),
        "feasible_successes": state["feasible"],
        "distinct_feasible_codes": len(state["distinct"]),
        "last_window_success_rate": float(np.mean([h["success_rate"] for h in last])) if last else None,
        "terminal_last": summarize_terminal(last, d, w_max),
        "terminal_all": summarize_terminal(history, d, w_max),
        "mask": env.mask_record(last) if variant in MASKED else None,
        "final_policy_probe": probe,
        "final": history[-1] if history else None,
        "platform": platform.platform(),
        "history": history,
    }
    for key in ("terminal_last", "terminal_all"):
        rec[key].update(inside_bound(rec[key], d, w_max))
        rec[key]["heavy_at_d_hist"] = np.sum(
            [h["terminal_heavy_at_d"] for h in (last if key == "terminal_last" else history)], axis=0
        ).tolist() if history else []
    return rec


def inside_bound(term: dict, d: int, w_max: int) -> dict:
    """``at_d`` (terminal states at the target distance), ``inside`` (those
    with exact ``w* <= w_max``), and their ratio, from the joint
    ``(d, w*)`` counts of :func:`summarize_terminal`."""
    joint = np.asarray(term.get("joint", []), dtype=np.int64)
    if joint.size == 0:
        return {"at_d": 0, "inside": 0, "inside_frac": None}
    at_d = int(joint[d:].sum())
    inside = int(joint[d:, : w_max + 1].sum())
    return {"at_d": at_d, "inside": inside, "inside_frac": (inside / at_d) if at_d else None}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def load_all(out_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(out_dir.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        if r.get("kind") != "ldpc_sparsity":
            continue
        r["_file"] = str(p)
        recs.append(r)
    return recs


def mask_method(r: dict) -> str | None:
    """Mask method of a run record, or ``None`` for an unmasked variant.
    Records written before ``mask_method`` existed used the tableau method."""
    m = r.get("mask")
    if not m:
        return None
    return m.get("method") or "tableau"


def label(r: dict) -> str:
    """Variant label of a run: the variant (with ``c`` for ``shaped`` and
    ``(exact mask)`` for the exact mask method), then the qualifiers that
    differ from the comparison's defaults (budget, batch)."""
    v = r["variant"]
    if v == "shaped":
        v = f"shaped (c = {r['c']:g})"
    if mask_method(r) == "exact":
        v += " (exact mask)"
    quals = []
    if r["budget_env_steps"] != DEFAULT_BUDGET:
        quals.append(f"{r['budget_env_steps'] / 1e6:g}M steps")
    if r.get("batch", DEFAULT_BATCH) != DEFAULT_BATCH:
        quals.append(f"batch {r['batch']}")
    if mask_method(r) == "tableau" and r["n"] > 12:
        # the n = 12 runs all used the tableau method (the comparison's documented
        # default); above that the basis method is the default and a tableau run is tagged
        quals.append("tableau mask")
    if quals:
        v += f" ({', '.join(quals)})"
    return v


def base_label(lab: str) -> str:
    """``label`` without the budget, batch, and tableau-mask qualifiers."""
    return re.sub(r" \((?:(?:[0-9.]+M steps|batch \d+|tableau mask)(?:, )?)+\)$", "", lab)


VARIANT_ORDER = {"unnormalized": 0, "shaped": 1, "terminal": 2, "mask": 3, "mask_terminal": 4}


def _key(r: dict):
    return (CELLS.index(r["cell"]), VARIANT_ORDER[r["variant"]], r["c"] or 0.0, r["budget_env_steps"], r["seed"])


def _pct(a, b) -> str:
    return f"{100.0 * a / b:.0f}" if b else "n/a"


def _frac_str(term: dict) -> str:
    f = term.get("inside_frac")
    return "n/a" if f is None else f"{100 * f:.0f}"


def _hist_str(h, lo=1):
    h = np.asarray(h, dtype=np.int64)
    if h.size == 0 or not h.any():
        return "n/a"
    return str(h[lo : max(np.flatnonzero(h)) + 1].tolist())


def groups(recs: list[dict]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = {}
    for r in recs:
        g.setdefault((r["cell"], label(r)), []).append(r)
    return {key: sorted(v, key=lambda r: r["seed"]) for key, v in sorted(g.items(), key=lambda kv: _key(kv[1][0]))}


def summary_table(recs: list[dict]) -> str:
    """One row per (cell, variant); the terminal columns are over the last
    25 updates of every seed."""
    lines = [
        "| target | graph | variant | seeds | env steps per run | at-d terminal states inside the bound, percent (per seed; pooled) | terminal states at d = 3, percent (per seed) | first feasible code, env steps (per seed) | feasible successes (per seed) | w* histogram of at-d terminal states (w* = 1, 2, ...) | heavy-generator histogram of at-d terminal states (h = 0, 1, ...) | wall (s, per seed) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (cell, lab), rs in groups(recs).items():
        n, k, graph, _ = TARGETS[cell]
        d = rs[0]["d"]
        tl = [r["terminal_last"] for r in rs]
        at_d = sum(t["at_d"] for t in tl)
        inside = sum(t["inside"] for t in tl)
        joint = np.sum([np.asarray(t["joint"]) for t in tl], axis=0)
        wh = joint[d:].sum(axis=0)
        heavy = np.sum([t.get("heavy_at_d_hist", []) for t in tl], axis=0) if all(t.get("heavy_at_d_hist") for t in tl) else np.zeros(1, dtype=int)
        lines.append(
            f"| [[{n}, {k}, {d}]] | {graph} | {lab} | {len(rs)} | {', '.join(f'{r['env_steps']:,}' for r in rs)} "
            f"| {', '.join(_frac_str(t) for t in tl)}; {_pct(inside, at_d)} "
            f"| {', '.join(_pct(t['at_d'], t['n_terminal']) for t in tl)} "
            f"| {', '.join(f'{r['first_success']['env_steps']:,}' if r['first_success'] else 'none' for r in rs)} "
            f"| {', '.join(str(r['feasible_successes']) for r in rs)} | {_hist_str(wh)} | {_hist_str(heavy, 0)} "
            f"| {', '.join(f'{r['wall_time_s']:.0f}' for r in rs)} |"
        )
    return "\n".join(lines)


def run_table(recs: list[dict]) -> str:
    lines = [
        "| target | graph | variant | seed | env steps | updates | wall (s) | env steps/s | first feasible code: env steps, gates, d, w*, degree, CSS | feasible successes (distinct) | terminal states, last 25 updates: n, at d = 3, inside the bound, inside percent of at-d, mode of w* at d | whole run: at d = 3, inside, percent | mask: fraction of two-qubit actions masked (last 25 updates), mask time (s) | probe (256 sampled episodes): at d = 3, feasible, mode of w* | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(recs, key=_key):
        fs = r["first_success"]
        code = fs.get("code") if fs else None
        tl, ta = r["terminal_last"], r["terminal_all"]
        d = r["d"]
        joint = np.asarray(tl["joint"]) if tl.get("joint") else None
        mode = str(int(np.argmax(joint[d:].sum(axis=0)))) if joint is not None and joint[d:].any() else "n/a"
        first = (
            f"{fs['env_steps']:,}, {fs['circuit_length']}, {code['d']}, {code['w_star']}, {code['max_degree']}, {'yes' if code['is_css'] else 'no'}"
            if code
            else "none"
        )
        m = r.get("mask")
        mtxt = f"{100 * m['two_masked_frac_last']:.1f} percent, {m['time_s']:.0f}" if m and m.get("two_masked_frac_last") is not None else "n/a"
        probe = (r.get("final_policy_probe") or {}).get("sampled")
        ptxt = (
            f"{sum(probe['d_hist'][d:])}, {probe['feasible_clean_target_states']}, {int(np.argmax(probe['w_star_hist']))}"
            if probe
            else ("skipped" if (r.get("final_policy_probe") or {}).get("skipped") else "n/a")
        )
        lines.append(
            f"| [[{r['n']}, {r['k']}, {d}]] | {r['graph']} | {label(r)} | {r['seed']} | {r['env_steps']:,} | {r['updates']} | {r['wall_time_s']:.0f} "
            f"| {r['env_steps_per_s']:,.0f} | {first} | {r['feasible_successes']} ({r['distinct_feasible_codes']}) "
            f"| {tl.get('n_terminal', 0):,}, {tl.get('at_d', 0):,}, {tl.get('inside', 0):,}, {_frac_str(tl)}, {mode} "
            f"| {ta.get('at_d', 0):,}, {ta.get('inside', 0):,}, {_frac_str(ta)} | {mtxt} | {ptxt} | `{Path(r['_file']).name}` |"
        )
    return "\n".join(lines)


def sources_table(recs: list[dict]) -> str:
    lines = ["| file | contents |", "|---|---|"]
    for r in sorted(recs, key=lambda r: r["_file"]):
        lines.append(
            f"| `experiments/ldpc_sparsity/runs/{Path(r['_file']).name}` | {label(r)}, [[{r['n']}, {r['k']}, {r['d']}]] w_max {r['w_max']} on {r['graph']}, "
            f"seed {r['seed']}, {r['env_steps']:,} env steps, {r['wall_time_s']:.0f} s, "
            f"{'first feasible code at ' + format(r['first_success']['env_steps'], ',') if r['first_success'] else 'no feasible code'} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
VARIANT_COLORS = {
    "unnormalized": "#4477AA",
    "shaped (c = 0.05)": "#CCBB44",
    "shaped (c = 0.1)": "#EE6677",
    "shaped (c = 0.2)": "#AA3377",
    "terminal": "#228833",
    "mask": "#66CCEE",
    "mask_terminal": "#000000",
    "unnormalized (3M steps)": "#777777",
}
SEED_STYLES = ["-", "--", ":"]


def _windowed(history: list[dict], num_key: str, den_key: str, window: int = TERMINAL_WINDOW):
    steps = np.array([h["env_steps"] for h in history], dtype=float) / 1e6
    num = np.array([h[num_key] for h in history], dtype=float)
    den = np.array([h[den_key] if den_key else sum(h["terminal_d"]) for h in history], dtype=float)
    cn = np.cumsum(np.insert(num, 0, 0.0))
    cd = np.cumsum(np.insert(den, 0, 0.0))
    lo = np.maximum(np.arange(len(history)) + 1 - window, 0)
    dwin = cd[1:] - cd[lo]
    frac = np.where(dwin > 0, (cn[1:] - cn[lo]) / np.maximum(dwin, 1.0), np.nan)
    return steps, frac


def figure_cells(recs: list[dict]) -> list[str]:
    """Cells drawn in the comparison figures: those with more than one
    variant. Cells run with the masked variants only (the scaling cells and
    the exact-mask cells) are in the tables and in :func:`figure_exact`."""
    return [c for c in CELLS if len({r["variant"] for r in recs if r["cell"] == c}) > 1]


EXACT_CELLS = ("n16_k4_all_to_all", "n16_k4_square")  # the cells of the exact-mask section


def exact_section_recs(recs: list[dict]) -> list[dict]:
    """Runs of the exact-mask section: the masked variants on the 16-qubit
    cells (exact mask and greedy controls, every batch)."""
    return sorted((r for r in recs if r["cell"] in EXACT_CELLS and r["variant"] in MASKED), key=_key)


def _run_style(r: dict):
    """Color by mask method and batch, line style by seed, for :func:`figure_exact`."""
    from . import figstyle

    exact = mask_method(r) == "exact"
    big = r.get("batch", DEFAULT_BATCH) != DEFAULT_BATCH
    color = {(True, False): "#000000", (True, True): "#EE6677", (False, False): "#4477AA", (False, True): figstyle.NEUTRAL}[(exact, big)]
    name = ("exact mask" if exact else "greedy mask") + (f", batch {r['batch']}" if big else "")
    if not exact and mask_method(r) == "tableau":
        name += " (tableau method)"
        color = "#CCBB44"
    return color, SEED_STYLES[r["seed"] % 3], name


def _rolling(values, window: int = TERMINAL_WINDOW):
    """Trailing mean over ``window`` entries (``nan`` entries skipped)."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.size, np.nan)
    for i in range(v.size):
        w = v[max(0, i + 1 - window) : i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if w.size else np.nan
    return out


def figure_exact(recs: list[dict], stem: Path):
    """Exact-mask section. Top row: fraction of terminal states at distance
    2 or more over a trailing 25-update window (distance 3, the target, is
    reached by at most one state per run, so the row shows the progress
    below it); bottom row: fraction of two-qubit actions masked, trailing
    25-update mean (episodes are synchronized, so the per-update value
    oscillates with the step in the episode; for the exact runs the thin
    line of the same color is the fraction the greedy pre-filter rejected
    before the exact test admitted some back). One column per 16-qubit
    cell, color by mask method and batch, line style by seed."""
    from . import figstyle

    rs = exact_section_recs(recs)
    cells = [c for c in EXACT_CELLS if any(r["cell"] == c for r in rs)]
    if not cells:
        return None
    plt = figstyle.apply()
    fig, axes = plt.subplots(2, len(cells), figsize=(figstyle.WIDTH, figstyle.HEIGHT_DOUBLE), sharey="row", squeeze=False)
    seen = {}
    for j, cell in enumerate(cells):
        n, k, graph, _ = TARGETS[cell]
        for r in (r for r in rs if r["cell"] == cell):
            color, ls, name = _run_style(r)
            for h in r["history"]:
                h["_terminal_ge2"] = int(sum(h["terminal_d"][2:]))
            s, f = _windowed(r["history"], "_terminal_ge2", None)
            line, = axes[0, j].plot(s, f, ls, color=color, lw=figstyle.THIN)
            seen.setdefault(name, line)
            steps = np.array([h["env_steps"] for h in r["history"]]) / 1e6
            axes[1, j].plot(steps, _rolling([h.get("mask_two_masked_frac", np.nan) for h in r["history"]]), ls, color=color, lw=figstyle.THIN)
            if "mask_two_greedy_masked_frac" in r["history"][-1]:
                axes[1, j].plot(steps, _rolling([h.get("mask_two_greedy_masked_frac", np.nan) for h in r["history"]]), ls, color=color, lw=0.35, alpha=0.6)
        axes[0, j].set_title(f"[[{n}, {k}, {CELL_SPECS[cell][2]}]] on {graph}")
        axes[1, j].set_xlabel("env steps (millions)")
        for ax in axes[:, j]:
            ax.set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("terminal states\nat distance 2 or more")
    axes[1, 0].set_ylabel("two-qubit actions masked\n(thin: greedy pre-filter)")
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=min(3, len(seen)), bbox_to_anchor=(0.5, -0.13), title="mask (seed 0 solid, 1 dashed, 2 dotted)")
    for ax, letter in zip(axes.ravel(), "abcdefgh"):
        figstyle.panel_label(ax, letter)
    fig.tight_layout()
    return figstyle.save(fig, stem)


def figure_exact_distance(recs: list[dict], stem: Path):
    """Distance distribution of the terminal states of the last 25 updates,
    one stacked bar per run of the exact-mask section, one panel per
    16-qubit cell."""
    from . import figstyle

    rs = exact_section_recs(recs)
    cells = [c for c in EXACT_CELLS if any(r["cell"] == c for r in rs)]
    if not cells:
        return None
    plt = figstyle.apply()
    fig, axes = plt.subplots(1, len(cells), figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, cell in zip(axes, cells):
        n, k, graph, _ = TARGETS[cell]
        d = CELL_SPECS[cell][2]
        cell_rs = [r for r in rs if r["cell"] == cell]
        labels = []
        for i, r in enumerate(cell_rs):
            joint = np.asarray(r["terminal_last"]["joint"], dtype=float)
            by_d = joint.sum(axis=1)
            tot = by_d.sum()
            frac = by_d / tot if tot else by_d
            bottom = 0.0
            for dd in range(1, d + 1):
                ax.bar(i, frac[dd], bottom=bottom, color=figstyle.DISTANCE_COLORS.get(dd, "#000000"), width=0.7, label=f"d = {dd}" if i == 0 else None)
                bottom += frac[dd]
            kind = "exact" if mask_method(r) == "exact" else "greedy"
            qual = "tableau" if mask_method(r) == "tableau" else (f"batch {r['batch']}" if r.get("batch", DEFAULT_BATCH) != DEFAULT_BATCH else "")
            labels.append(f"{kind}\n{qual}\nseed {r['seed']}")
        ax.set_xticks(range(len(cell_rs)))
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_title(f"[[{n}, {k}, {d}]] on {graph}")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("terminal states, last 25 updates")
    axes[-1].legend(loc="upper right", fontsize=6)
    for ax, letter in zip(axes, "abcd"):
        figstyle.panel_label(ax, letter)
    fig.tight_layout()
    return figstyle.save(fig, stem)


def figure_inside(recs: list[dict], stem: Path):
    """Top row: fraction of at-d terminal states inside the bound over a
    trailing 25-update window; bottom row: fraction of terminal states at
    the target distance. One column per comparison cell, one color per
    variant, one line style per seed."""
    from . import figstyle

    plt = figstyle.apply()
    cells = figure_cells(recs)
    if not cells:
        return None
    fig, axes = plt.subplots(2, len(cells), figsize=(figstyle.WIDTH, figstyle.HEIGHT_DOUBLE), sharey="row", squeeze=False)
    seen = set()
    for j, cell in enumerate(cells):
        n, k, graph, _ = TARGETS[cell]
        for r in sorted((r for r in recs if r["cell"] == cell), key=_key):
            lab = label(r)
            color = VARIANT_COLORS.get(lab, "#000000")
            ls = SEED_STYLES[r["seed"] % 3]
            s, f = _windowed(r["history"], "terminal_inside_at_d", "terminal_at_d")
            axes[0, j].plot(s, f, ls, color=color, lw=figstyle.THIN, label=lab if lab not in seen else None)
            seen.add(lab)
            s, f = _windowed(r["history"], "terminal_at_d", None)
            axes[1, j].plot(s, f, ls, color=color, lw=figstyle.THIN)
        axes[0, j].set_title(f"[[{n}, {k}, {CELL_SPECS[cell][2]}]] on {graph}")
        axes[1, j].set_xlabel("env steps (millions)")
        for ax in axes[:, j]:
            ax.set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("at-d terminal states\ninside the bound")
    axes[1, 0].set_ylabel("terminal states\nat the target distance")
    handles, labels = [], []
    for ax in axes[0]:
        hs, ls = ax.get_legend_handles_labels()
        handles += hs
        labels += ls
    order = sorted(range(len(labels)), key=lambda i: (VARIANT_ORDER.get(labels[i].split(" (")[0], 9), labels[i]))
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(len(labels), 1)), bbox_to_anchor=(0.5, -0.13), title="variant (seed 0 solid, 1 dashed, 2 dotted)")
    for ax, letter in zip(axes.ravel(), "abcdefgh"):
        figstyle.panel_label(ax, letter)
    fig.tight_layout()
    return figstyle.save(fig, stem)


def figure_wstar(recs: list[dict], stem: Path):
    """Distribution of exact ``w*`` over the terminal states at d = 3 of the
    last 25 updates, pooled over seeds; one panel per cell, one bar group
    per ``w*``, one color per variant."""
    from . import figstyle

    plt = figstyle.apply()
    cells = figure_cells(recs)
    if not cells:
        return None
    fig, axes = plt.subplots(1, len(cells), figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, cell in zip(axes, cells):
        n, k, graph, _ = TARGETS[cell]
        d, w_max = CELL_SPECS[cell][2], 4
        cell_groups = [(lab, rs) for (c, lab), rs in groups(recs).items() if c == cell]
        width = 0.8 / max(len(cell_groups), 1)
        ws = np.arange(1, n + 1)
        for i, (lab, rs) in enumerate(cell_groups):
            joint = np.sum([np.asarray(r["terminal_last"]["joint"]) for r in rs], axis=0)
            wh = joint[d:].sum(axis=0)[1 : n + 1]
            tot = wh.sum()
            frac = wh / tot if tot else wh
            # variants without any state at distance 3 have nothing to draw; the tables report them
            ax.bar(ws + (i - (len(cell_groups) - 1) / 2) * width, frac, width=width, color=VARIANT_COLORS.get(lab, "#000000"), label=f"{lab} (n = {tot:,})" if tot else None)
        ax.axvline(w_max + 0.5, color=figstyle.NEUTRAL, lw=figstyle.THIN, ls="--")
        ax.set_title(f"[[{n}, {k}, {CELL_SPECS[cell][2]}]] on {graph}")
        ax.set_xlabel("exact w* of terminal states at the target distance")
        ax.set_xlim(1.5, min(n, 10) + 0.5)
        ax.legend(loc="upper right", fontsize=6)
    axes[0].set_ylabel("fraction (last 25 updates, all seeds)")
    for ax, letter in zip(axes, "abcd"):
        figstyle.panel_label(ax, letter)
    fig.tight_layout()
    return figstyle.save(fig, stem)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def run_name(variant: str, cell: str, seed: int, c: float = 0.1, budget: int = DEFAULT_BUDGET, mask_method: str = "basis", batch: int = DEFAULT_BATCH) -> str:
    """File stem of a run. The exact mask method adds ``_exact`` and a
    batch other than 256 adds ``_b<batch>`` (the batch-1024 records of the
    all-to-all scaling section predate the tag and were renamed to it; the
    tableau-method record on ``square:4x4`` carries ``_tableau`` by hand,
    since the method is not part of the name and the basis-method control
    of the exact-mask section has the same name otherwise)."""
    tag = f"_c{c:g}" if variant == "shaped" else ""
    if variant in MASKED and mask_method == "exact":
        tag += "_exact"
    if budget != DEFAULT_BUDGET:
        tag += f"_{budget / 1e6:g}M"
    if batch != DEFAULT_BATCH:
        tag += f"_b{batch}"
    return f"{variant}{tag}_{cell}_seed{seed}"


def bench_mask(cell: str, batch: int, steps: int | None = None, seed: int = 0, methods=("basis", "exact")) -> dict:
    """Cost per env step of the action-mask methods on ``cell`` at
    ``batch``: a random policy under the exact mask (uniform over the
    admitted actions) for ``steps`` env steps (default the cell's episode
    length), every method evaluated on every state. Reports mean and max
    seconds per call, the masked fractions, and the exact method's
    counters (tests per call, admitted, cache hit rate)."""
    from ..env import CodeDiscoveryEnv

    n, k, _d, graph_spec, gates, cell_steps = CELL_SPECS[cell]
    steps = cell_steps if steps is None else int(steps)
    env = CodeDiscoveryEnv(n=n, k=k, batch_size=batch, max_steps=steps + 1, gates=gates, graph=graphs.from_spec(graph_spec), seed=seed, auto_reset=True)
    env.reset()
    rng = np.random.default_rng(seed)
    cache = ExactMaskCache()
    times = {mth: [] for mth in methods}
    masked = {mth: [] for mth in methods}
    totals = dict(greedy_rejected=0, exact_tested=0, exact_unique=0, exact_admitted=0, cache_hits=0, cache_misses=0, pairs=0)
    two = env.action_table.q2 >= 0
    for _ in range(steps):
        masks = {}
        for mth in methods:
            st: dict = {}
            t0 = time.perf_counter()
            masks[mth] = greedy_action_mask(env.tab, k, env.action_table, 4, mth, exact_max_generators=EXACT_GENERATORS, cache=cache if mth == "exact" else None, stats=st)
            times[mth].append(time.perf_counter() - t0)
            masked[mth].append(float((~masks[mth][:, two]).mean()))
            if mth == "exact":
                for key in totals:
                    totals[key] += st.get(key, 0)
        m = masks["exact" if "exact" in masks else methods[-1]]
        u = np.where(m, rng.random(m.shape), -1.0)
        env.step(u.argmax(axis=1))
    out = {"cell": cell, "batch": batch, "steps": steps, "seed": seed, "graph": graph_spec, "two_qubit_actions": int(two.sum()), "cpu": platform.processor() or platform.machine(), "threads": os.environ.get("OMP_NUM_THREADS")}
    if os.path.exists("/proc/loadavg"):
        with open("/proc/loadavg") as f:
            out["loadavg"] = [float(x) for x in f.read().split()[:3]]
        out["cpu_count"] = os.cpu_count()
    for mth in methods:
        out[mth] = {"mean_s": float(np.mean(times[mth])), "max_s": float(np.max(times[mth])), "masked_frac_mean": float(np.mean(masked[mth])), "masked_frac_last": masked[mth][-1]}
    if "exact" in methods:
        out["exact"].update(dict(totals.items()))
        out["exact"]["tested_per_call"] = totals["exact_tested"] / steps
        out["exact"]["s_per_tested_pair"] = (sum(times["exact"]) - sum(times.get("basis", [0.0]))) / max(totals["exact_tested"], 1)
    return out


def _report_work(rec: dict, out: Path) -> None:
    """Report the run to the compute ledger's ``wrap`` through its work file
    when one is set (``COMPUTE_LEDGER_WORK_FILE``)."""
    path = os.environ.get("COMPUTE_LEDGER_WORK_FILE")
    if not path:
        return
    work = {
        "env_steps": rec["env_steps"],
        "episodes": rec["episodes"],
        "updates": rec["updates"],
        "threads": rec["threads"],
        "variant": label(rec),
        "cell": rec["cell"],
        "seed": rec["seed"],
        "batch": rec["batch"],
        "budget_env_steps": rec["budget_env_steps"],
        "inside_frac_last": rec["terminal_last"].get("inside_frac"),
        "at_d_frac_last": (rec["terminal_last"]["at_d"] / rec["terminal_last"]["n_terminal"]) if rec["terminal_last"].get("n_terminal") else None,
        "feasible_successes": rec["feasible_successes"],
        "artifacts": [str(out)],
        "outcome": "success" if rec["success"] else "negative_result",
    }
    with open(path, "w") as f:
        json.dump(work, f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sparsity as a property of the trained policy.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="one PPO run of one variant on one cell")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--cell", choices=CELLS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c", type=float, default=0.1, help="shaping coefficient of the shaped variant")
    p.add_argument("--max-env-steps", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--minutes", type=float, default=180.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None, help="episode length (default: the cell's)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--policy", choices=("mlp", "equivariant", "auto"), default="equivariant")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="torch device of the policy and the PPO update")
    p.add_argument("--reward-backend", choices=("numpy", "torch"), default=None, help="base terms of the reward on numpy or on the torch device (default: torch on cuda, numpy on cpu)")
    p.add_argument("--chunk", type=int, default=None, help="Paulis per streamed chunk of the torch reward (default 8192 on cuda, 65536 on cpu)")
    p.add_argument("--mask-method", choices=MASK_METHODS, default="basis", help="action mask: the greedy bound from the current state's greedy basis with packed CNOTs (default), from full tableau copies (the method of the n = 12 runs), or the exact feasibility test on the pairs the greedy bound rejects (n - k <= 15)")
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("bench-mask", help="cost per env step of the mask methods on one cell (random policy under the exact mask)")
    p.add_argument("--cell", choices=CELLS, required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--methods", default="basis,exact")
    p = sub.add_parser("report", help="print the tables from experiments/ldpc_sparsity/runs and draw the figures")
    p.add_argument("--dir", default=str(DEFAULT_OUT))
    p.add_argument("--figures", default="experiments/ldpc_sparsity/figures", help="directory for the figures ('' to skip)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        rec = run_variant(
            args.variant, args.cell, args.seed, c=args.c, max_env_steps=args.max_env_steps, minutes=args.minutes,
            threads=args.threads, max_steps=args.max_steps, batch=args.batch, policy=args.policy, device=args.device,
            reward_backend=args.reward_backend, chunk=args.chunk, mask_method=args.mask_method, verbose=args.verbose,
        )
        out = Path(args.out or DEFAULT_OUT / f"{run_name(args.variant, args.cell, args.seed, args.c, args.max_env_steps, args.mask_method, args.batch)}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=_json_default)
        _report_work(rec, out)
        tl = rec["terminal_last"]
        print(
            f"wrote {out}: env_steps {rec['env_steps']:,} wall {rec['wall_time_s']}s at_d {tl.get('at_d')} inside {tl.get('inside')} "
            f"inside_frac {tl.get('inside_frac')} feasible_successes {rec['feasible_successes']}"
        )
        return 0
    if args.cmd == "bench-mask":
        from ..agents.ppo import set_threads

        set_threads(args.threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        print(json.dumps(bench_mask(args.cell, args.batch, args.steps, args.seed, tuple(args.methods.split(","))), indent=1))
        return 0
    recs = load_all(Path(args.dir))
    print("## Summary per variant\n")
    print(summary_table(recs))
    print("\n## Runs\n")
    print(run_table(recs))
    print("\n## Sources\n")
    print(sources_table(recs))
    if args.figures and recs:
        for fn, stem in ((figure_inside, "inside_bound"), (figure_wstar, "w_star_at_d"), (figure_exact, "exact_mask"), (figure_exact_distance, "exact_mask_distance")):
            out = fn(recs, Path(args.figures) / stem)
            if out:
                print("figure:", out[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
