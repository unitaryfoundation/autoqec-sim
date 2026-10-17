"""QDX's published target set through autoqec-sim's PPO at matched settings.

QDX (github.com/jolle-ag/qdx, Olle, Zen, Puviani, Marquardt, npj Quantum
Information 10, 126 (2024)) ships every ``[[n, k, d]]`` it discovered on
all-to-all connectivity for ``n <= 15`` as ``rl_discovered_codes.zip``;
``QDX_TARGETS`` records, per target, how many codes that archive holds and
the shortest, median, and longest encoding circuit among them (read on
2026-10-16). This module runs the same targets through
:class:`autoqec_sim.agents.ppo.PPOAgent` on :class:`CodeDiscoveryEnv` with
the default Knill-Laflamme reward over all Paulis of weight below ``d``,
gates H, S, and CNOT, all-to-all connectivity, and an episode cap that
grows with ``d``, and verifies each result by exact distance. QDX weights
the same error set by a depolarizing channel (``p = 0.1``) and ran four
agents per target on a GPU; the comparison fixes the target, the gate
set, the connectivity, and the success criterion (every error of weight
below ``d`` detected or in the stabilizer group), nothing else.

    python -m experiments.qdx_targets.run run --tier quick --seed 0
    python -m experiments.qdx_targets.run run --targets 7,1,3 9,3,3
    python -m experiments.qdx_targets.run report
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from autoqec_sim.code import StabilizerCode
from autoqec_sim.env import CodeDiscoveryEnv

DEFAULT_OUT = Path("experiments/qdx_targets/runs")
GATES = ("H", "S", "CNOT")
BATCH = 512

# (n, k, d): (codes in QDX's archive, shortest, median, longest circuit).
QDX_TARGETS = {
    (5, 1, 3): (930, 10, 12, 21), (6, 1, 3): (3215, 11, 15, 30), (7, 1, 3): (1393, 10, 14, 28),
    (8, 1, 3): (1376, 11, 15, 22), (8, 2, 3): (666, 16, 18, 27), (8, 3, 3): (164, 19, 22, 28),
    (9, 1, 3): (9032, 11, 15, 29), (9, 2, 3): (2148, 14, 18, 26), (9, 3, 3): (1336, 18, 22, 30),
    (10, 1, 3): (1021, 9, 15, 22), (10, 1, 4): (622, 20, 26, 34), (10, 2, 3): (2949, 15, 19, 24),
    (10, 2, 4): (111, 26, 31, 37), (10, 3, 3): (3014, 18, 22, 29), (10, 4, 3): (928, 21, 26, 35),
    (11, 1, 3): (696, 9, 19, 23), (11, 1, 4): (1189, 18, 24, 32), (11, 1, 5): (7, 32, 35, 44),
    (11, 2, 3): (384, 17, 20, 25), (11, 2, 4): (698, 26, 32, 39), (11, 3, 3): (366, 20, 24, 29),
    (11, 4, 3): (3488, 22, 26, 34), (11, 5, 3): (768, 25, 30, 37), (12, 1, 4): (448, 20, 24, 33),
    (12, 2, 3): (224, 18, 21, 25), (12, 2, 4): (749, 25, 31, 38), (12, 3, 3): (375, 20, 25, 29),
    (12, 3, 4): (304, 33, 40, 49), (12, 4, 3): (3006, 23, 27, 34), (12, 5, 3): (2535, 26, 29, 38),
    (12, 6, 3): (998, 28, 33, 41), (13, 1, 5): (4, 44, 60, 60), (13, 3, 4): (689, 31, 38, 48),
    (13, 4, 3): (364, 25, 29, 34), (13, 5, 3): (215, 27, 30, 38), (13, 6, 3): (1299, 28, 32, 40),
    (13, 7, 3): (540, 32, 38, 47), (14, 1, 5): (46, 36, 40, 46), (14, 3, 4): (58, 31, 36, 43),
    (14, 5, 3): (214, 28, 31, 37), (14, 6, 3): (159, 29, 33, 36), (14, 7, 3): (748, 32, 36, 44),
    (14, 8, 3): (377, 36, 42, 55), (15, 1, 5): (14, 39, 40, 42), (15, 2, 5): (1, 49, 49, 49),
    (15, 5, 3): (238, 29, 33, 38), (15, 6, 3): (156, 31, 34, 38), (15, 7, 3): (91, 33, 37, 42),
    (15, 8, 3): (545, 35, 39, 47), (15, 9, 3): (328, 38, 47, 57),
}

TIERS = {
    "quick": lambda n, k, d: d == 3 and n <= 10,
    "d3": lambda n, k, d: d == 3,
    "d4": lambda n, k, d: d == 4,
    "d5": lambda n, k, d: d == 5,
    "all": lambda n, k, d: True,
}


def max_steps_for(d: int) -> int:
    return {3: 30, 4: 40}.get(d, 60)


def run_target(n: int, k: int, d: int, seed: int = 0, max_env_steps: int = 3_000_000, minutes: float = 10.0, threads: int = 6) -> dict:
    """One target: train PPO until a code with every error of weight below
    ``d`` detected or in the group, then verify its exact distance."""
    from autoqec_sim.agents.ppo import PPOAgent, PPOConfig

    env = CodeDiscoveryEnv(n=n, k=k, batch_size=BATCH, gates=GATES, max_steps=max_steps_for(d), target_distance=d, seed=seed)
    agent = PPOAgent(env, config=PPOConfig(max_threads=threads), seed=seed)
    t0 = time.perf_counter()
    res = agent.train(max_env_steps, max_time_s=minutes * 60.0, stop_on_success=True, verbose=False)
    wall = time.perf_counter() - t0
    found = res.first_success
    gates = list(found.gates) if found is not None else []
    rec = {
        "target": [n, k, d], "seed": seed, "gates": list(GATES), "graph": f"all_to_all:{n}", "max_steps": max_steps_for(d),
        "batch": BATCH, "policy": res.config.get("policy") if isinstance(res.config, dict) else None,
        "budget": {"max_env_steps": max_env_steps, "minutes": minutes},
        "success": bool(res.success), "env_steps": int(res.env_steps), "episodes": int(res.episodes), "updates": int(res.updates),
        "wall_s": round(wall, 1), "circuit_length": len(gates) if gates else None,
        "qdx": dict(zip(("codes", "min_len", "median_len", "max_len"), QDX_TARGETS.get((n, k, d), (None,) * 4), strict=True)),
        "platform": platform.platform(), "numpy": np.__version__,
    }
    if res.success and gates:
        code = StabilizerCode.from_gates(n, k, gates)
        dist = code.distance_exact() if n <= 20 else None
        rec.update({"verified_d": dist, "verified": dist is not None and dist >= d, "css": code.is_css(), "stabilizers": code.stabilizer_strings()})
    return rec


def run_path(out_dir: Path, n: int, k: int, d: int, seed: int) -> Path:
    return out_dir / f"{n}_{k}_{d}_seed{seed}.json"


def load_all(out_dir: Path) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_seed*.json"))]
    return sorted(recs, key=lambda r: (r["target"][2], r["target"][0], r["target"][1], r["seed"]))


def report_table(recs: list[dict]) -> str:
    lines = ["| target | QDX codes | QDX circuit length (min / median) | autoqec PPO | env steps | wall | circuit length | verified d |",
             "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        n, k, d = r["target"]
        q = r["qdx"]
        status = "found" if r.get("verified") else ("found, unverified" if r["success"] else "not found in budget")
        lines.append(f"| [[{n},{k},{d}]] | {q['codes']} | {q['min_len']} / {q['median_len']} | {status} | {r['env_steps']:,} | {r['wall_s']:.0f} s | "
                     f"{r['circuit_length'] if r['circuit_length'] is not None else '-'} | {r.get('verified_d', '-')} |")
    return "\n".join(lines)


def write_report(out_dir: Path) -> Path:
    recs = load_all(out_dir)
    found = sum(1 for r in recs if r.get("verified"))
    capped = [r for r in recs if not r["success"] and r["wall_s"] >= 0.98 * r["budget"]["minutes"] * 60]
    cap_note = ""
    if capped:
        lo, hi = min(r["env_steps"] for r in capped), max(r["env_steps"] for r in capped)
        cap_note = (f" {len(capped)} of the unsuccessful runs stopped at the time cap after {lo:,} to {hi:,} environment "
                    f"steps, so they used between {lo / capped[0]['budget']['max_env_steps']:.0%} and "
                    f"{hi / capped[0]['budget']['max_env_steps']:.0%} of the step budget; the env steps column says which. "
                    "This run shared the machine with other jobs, which cut the step rate by up to eightfold from n = 12 on.")
    text = f"""# QDX's targets through autoqec-sim

Status: {found} of {len(recs)} targets run so far found and verified. Every
number comes from a JSON under `experiments/qdx_targets/runs/`; this file
regenerates with `python -m experiments.qdx_targets.run report`.
Code: `experiments/qdx_targets/run.py`.

QDX (Olle, Zen, Puviani, Marquardt, npj Quantum Information 10, 126 (2024);
github.com/jolle-ag/qdx) publishes every [[n, k, d]] its agents found on
all-to-all connectivity for n <= 15. The QDX columns count the codes in
that archive and give the shortest and median encoding circuit among them.
The autoqec columns are one PPO run per target (seed in the JSON), gates H,
S, CNOT, all-to-all, batch {BATCH}, episode cap 30 / 40 / 60 gates for
d = 3 / 4 / 5, the default Knill-Laflamme reward over all Paulis of weight
below d, and a budget of 3M environment steps or 10 minutes, whichever
comes first. Success means every such error is detected or in the group;
"verified" means the exact distance of the resulting code is at least d.
QDX weighted the same error set by a depolarizing channel and ran four
agents per target on a GPU, so wall times are not comparable across the two;
success within budget and circuit length are.{cap_note}

{report_table(recs)}
"""
    path = out_dir.parents[0] / "REPORT.md"
    path.write_text(text)
    return path


def _steps_label(steps: int) -> str:
    return f"{steps / 1e6:.1f}M" if steps >= 1e6 else f"{round(steps / 1e3)}k"


def make_figure(out_dir: Path = DEFAULT_OUT, out_stem: Path | str = "docs/assets/figures/qdx_targets"):
    """The target grid: one cell per published ``[[n, k, d]]`` triple at
    ``d = 3`` and ``d = 4``, filled when this package's PPO found and
    verified a code within the budget (annotated with the environment
    steps to success), open when the run hit the budget first. The
    ``d = 5`` targets, not run, are listed in the legend."""
    from ..common import figstyle

    plt = figstyle.apply()
    recs = {tuple(r["target"]): r for r in load_all(out_dir)}
    panels = [("d = 3", [t for t in QDX_TARGETS if t[2] == 3]), ("d = 4", [t for t in QDX_TARGETS if t[2] == 4])]
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE + 0.4), gridspec_kw={"width_ratios": [11, 5]})
    for ax, (title, targets), letter in zip(axes, panels, "ab", strict=True):
        for n, k, d in targets:
            r = recs.get((n, k, d))
            if r is None:
                ax.add_patch(plt.Rectangle((n - 0.45, k - 0.45), 0.9, 0.9, fill=False, edgecolor=figstyle.NEUTRAL, linestyle=":", linewidth=figstyle.THIN))
            elif r["success"] and r.get("verified"):
                ax.add_patch(plt.Rectangle((n - 0.45, k - 0.45), 0.9, 0.9, facecolor=figstyle.SERIES_COLORS[0], edgecolor="none"))
                ax.text(n, k, _steps_label(r["env_steps"]), ha="center", va="center", color="white", fontsize=5.5)
            else:
                ax.add_patch(plt.Rectangle((n - 0.45, k - 0.45), 0.9, 0.9, fill=False, edgecolor=figstyle.SERIES_COLORS[1], linewidth=figstyle.LINE))
        ns = sorted({t[0] for t in targets})
        ks = sorted({t[1] for t in targets})
        ax.set_xlim(min(ns) - 0.6, max(ns) + 0.6)
        ax.set_ylim(min(ks) - 0.6, max(ks) + 0.6)
        ax.set_xticks(ns)
        ax.set_yticks(ks)
        ax.set_xlabel("n")
        ax.set_ylabel("k")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(False)
        figstyle.panel_label(ax, letter)
    not_run = ", ".join(f"[[{n},{k},{d}]]" for n, k, d in QDX_TARGETS if (n, k, d) not in recs)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=figstyle.SERIES_COLORS[0], edgecolor="none", label="found and verified (env steps to success)"),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=figstyle.SERIES_COLORS[1], label="not found within 3M env steps or 10 min"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    if not_run:
        fig.text(0.5, -0.1, f"not run: {not_run}", ha="center", va="top", fontsize=7, color=figstyle.NEUTRAL)
    fig.tight_layout()
    return figstyle.save(fig, out_stem)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tier", choices=sorted(TIERS), default=None)
    r.add_argument("--targets", nargs="*", default=None, help="n,k,d triples")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-env-steps", type=int, default=3_000_000)
    r.add_argument("--minutes", type=float, default=10.0)
    r.add_argument("--threads", type=int, default=6)
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--force", action="store_true", help="rerun targets that have a result")
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    f = sub.add_parser("figure")
    f.add_argument("--out", default=str(DEFAULT_OUT))
    f.add_argument("--stem", default="docs/assets/figures/qdx_targets")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "report":
        print("wrote", write_report(out))
        return
    if a.cmd == "figure":
        print("wrote", *make_figure(out, a.stem))
        return
    out.mkdir(parents=True, exist_ok=True)
    if a.targets:
        targets = [tuple(int(x) for x in t.split(",")) for t in a.targets]
    else:
        pred = TIERS[a.tier or "quick"]
        targets = [t for t in QDX_TARGETS if pred(*t)]
    for n, k, d in targets:
        path = run_path(out, n, k, d, a.seed)
        if path.exists() and not a.force:
            print("skip", path.name, flush=True)
            continue
        rec = run_target(n, k, d, a.seed, a.max_env_steps, a.minutes, a.threads)
        path.write_text(json.dumps(rec, indent=1))
        print(f"[[{n},{k},{d}]] success={rec['success']} verified={rec.get('verified')} env_steps={rec['env_steps']:,} wall={rec['wall_s']} s len={rec['circuit_length']}", flush=True)
    print("wrote", write_report(out))


if __name__ == "__main__":
    main()
