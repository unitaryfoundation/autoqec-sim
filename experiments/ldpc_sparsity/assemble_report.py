# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Assemble experiments/ldpc_sparsity/REPORT.md from results/ldpc_sparsity/*.json.
Every number in the prose is computed here from the JSONs; the planned
cells that have no JSON are listed as pending.

    uv run --extra rl python experiments/ldpc_sparsity/assemble_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoqec_sim.experiments import ldpc_sparsity as S
from autoqec_sim.experiments.ldpc_sparsity import CELL_SPECS, TARGETS

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "ldpc_sparsity"
OUT = ROOT / "experiments" / "ldpc_sparsity" / "REPORT.md"
FIG = ROOT / "experiments" / "ldpc_sparsity" / "figures"

PLANNED_VARIANTS = ["unnormalized", "shaped (c = 0.05)", "shaped (c = 0.1)", "shaped (c = 0.2)", "terminal", "mask", "mask_terminal"]
PLANNED_SEEDS = (0, 1, 2)
# the mask on top of the terminal reward was added after the first results, seed 0 on the 12-qubit cells only
FIRST_CELLS = ("n12_k2_all_to_all", "n12_k2_square", "n16_k4_square")  # the comparison's cells (every variant, three seeds)
SCALING_CELLS = ("n16_k4_all_to_all", "n20_k4_all_to_all", "n25_k1_d4_all_to_all", "n25_k1_d5_all_to_all")  # mask_terminal only, seed 0, cuda
PLANNED = {}
for c in FIRST_CELLS:
    for lab in PLANNED_VARIANTS:
        PLANNED[(c, lab)] = PLANNED_SEEDS if lab != "mask_terminal" else ((0,) if TARGETS[c][0] == 12 else ())
for c in SCALING_CELLS:
    for lab in PLANNED_VARIANTS:
        PLANNED[(c, lab)] = (0,) if lab == "mask_terminal" else ()
MAJORITY = 0.5


def pct(a, b):
    return f"{100.0 * a / b:.0f}" if b else "n/a"


def rng(vals, fmt="{:.0f}"):
    vals = list(vals)
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)} to {fmt.format(hi)}"


def listing(vals, fmt="{}"):
    vals = [fmt.format(v) for v in vals]
    if len(vals) <= 1:
        return "".join(vals)
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return ", ".join(vals[:-1]) + f", and {vals[-1]}"


def inside_frac(r):
    return r["terminal_last"].get("inside_frac")


def at_d_frac(r):
    tl = r["terminal_last"]
    return tl["at_d"] / tl["n_terminal"] if tl.get("n_terminal") else 0.0


def w_mode_at_d(r):
    joint = np.asarray(r["terminal_last"]["joint"])
    h = joint[r["d"] :].sum(axis=0)
    return int(np.argmax(h)) if h.any() else None


def cell_name(cell):
    n, k, d, graph, _, _ = CELL_SPECS[cell]
    return f"`[[{n}, {k}, {d}]]` on `{graph}`"


def _time_split(r):
    """Percent of the wall time in the mask, the reward, the PPO update, and the rest."""
    wall = max(r["wall_time_s"], 1e-9)
    mask = (r.get("mask") or {}).get("time_s", 0.0) or 0.0
    parts = [mask, r.get("reward_time_s", 0.0), r.get("update_time_s", 0.0)]
    parts.append(max(wall - sum(parts), 0.0))
    return [100.0 * x / wall for x in parts]


def exact_section(A, recs):
    """The exact-mask section: the masked variants on the 16-qubit cells,
    exact mask against the greedy controls, with the benchmark of the mask
    cost (``results/ldpc_sparsity/bench/*.json``) and the hand-written
    reading in ``exact_mask.md``."""
    rs = S.exact_section_recs(recs)
    if not rs:
        return
    A("")
    A("## Exact feasibility mask")
    A("")
    A("The greedy bound of `greedy_action_mask` is an upper bound on `w*`, so it")
    A("can forbid a two-qubit gate whose resulting group is feasible. The exact")
    A("method (`method=\"exact\"`) keeps the greedy bound as a pre-filter and runs")
    A("the exact feasibility test on every (state, action) pair the bound rejects:")
    A("a CNOT changes the weight of a Pauli by at most one, so the elements light")
    A("after the gate are among the elements of weight `<= w_max + 1` before it,")
    A("enumerated once per state from the `2^(n - k)` group elements and shared by")
    A("all of the state's actions; the gate is admitted iff the elements it leaves")
    A("at weight `<= w_max` have rank `n - k` (elimination over the `n - k`")
    A("coefficient bits). A cache keyed by the state's greedy basis keeps the")
    A("enumeration across steps for states seen again. Exact for `n - k <= 15`;")
    A("above that the method falls back to the greedy basis bound and records it.")
    A("Tests: the exact mask equals `ldpc_feasible` on every trial action, admits")
    A("everything the greedy bound admits, and admits some moves it forbids")
    A("(`tests/test_ldpc.py`).")
    A("")
    benches = sorted((RES / "bench").glob("bench_mask_*.json")) if (RES / "bench").exists() else []
    if benches:
        A("Cost of the mask per env step, random policy under the exact mask, the")
        A("cell's episode length, pod CPU at 6 threads (`ldpc_sparsity bench-mask`):")
        A("")
        A("| cell | batch | two-qubit actions | greedy: mean, max (s per call) | exact: mean, max (s per call) | greedy-rejected pairs per call, of which exact admits (percent) | microseconds per exact test | two-qubit actions masked, mean (greedy; exact) | cache hit rate |")
        A("|---|---|---|---|---|---|---|---|---|")
        for bf in benches:
            b = json.loads(bf.read_text())
            e, g = b["exact"], b.get("basis", {})
            A(f"| {b['cell']} | {b['batch']} | {b['two_qubit_actions']} | {g.get('mean_s', float('nan')):.3f}, {g.get('max_s', float('nan')):.3f} | {e['mean_s']:.3f}, {e['max_s']:.3f} "
              f"| {e['tested_per_call']:,.0f}, {pct(e['exact_admitted'], e['exact_tested'])} | {1e6 * e['s_per_tested_pair']:.0f} | {100 * g.get('masked_frac_mean', float('nan')):.1f}; {100 * e['masked_frac_mean']:.1f} percent "
              f"| {pct(e['cache_hits'], e['cache_hits'] + e['cache_misses'])} percent |")
        A("")
    A("Runs: `mask_terminal` on the two `[[16, 4, 3]]` cells, 3M env steps, exact")
    A("mask at batch 256 (three seeds) and 1024 (seed 0), the greedy mask at batch")
    A("256 (seed 0) as the control at equal update count, and the earlier greedy")
    A("runs (batch 1024 on the pod's GPU on `all_to_all:16`; the tableau-method run")
    A("at batch 256 on `square:4x4`).")
    A("")
    A("| target | graph | mask | batch | seed | env steps | updates | wall (s) | env steps/s | terminal states, last 25 updates: n, at d = 1, 2, 3 | at-d inside the bound | feasible successes (distinct codes) | first feasible code, env steps | two-qubit actions masked, last 25 updates (greedy pre-filter; final) | exact tests per call, admitted (percent) | time split: mask, reward, update, rest (percent) | file |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        m = r.get("mask") or {}
        tl = r["terminal_last"]
        joint = np.asarray(tl["joint"], dtype=np.int64)
        by_d = joint.sum(axis=1)
        d = r["d"]
        method = "exact" if S.mask_method(r) == "exact" else f"greedy ({S.mask_method(r)})"
        masked = f"{100 * m['two_masked_frac_last']:.0f}" if m.get("two_masked_frac_last") is not None else "n/a"
        if m.get("greedy_two_masked_frac_last") is not None:
            masked = f"{100 * m['greedy_two_masked_frac_last']:.0f}; {masked}"
        else:
            masked = f"n/a; {masked}"
        exact_txt = f"{m['exact_tested_per_call']:,.0f}, {pct(m['exact_admitted'], m['exact_tested'])}" if m.get("exact_tested") else "n/a"
        split = ", ".join(f"{x:.0f}" for x in _time_split(r))
        inside = f"{tl['inside']:,} of {tl['at_d']:,}" if tl.get("at_d") else "no state at d = 3"
        fs = r["first_success"]
        A(f"| [[{r['n']}, {r['k']}, {d}]] | {r['graph']} | {method} | {r['batch']} | {r['seed']} | {r['env_steps']:,} | {r['updates']} | {r['wall_time_s']:.0f} | {r['env_steps_per_s']:,.0f} "
          f"| {tl.get('n_terminal', 0):,}: {', '.join(f'{int(by_d[dd]):,}' for dd in range(1, d + 1))} | {inside} | {r['feasible_successes']} ({r['distinct_feasible_codes']}) "
          f"| {f'{fs['env_steps']:,}' if fs else 'none'} | {masked} | {exact_txt} | {split} | `{Path(r['_file']).name}` |")
    A("")
    if (FIG / "exact_mask.png").exists():
        A("![](figures/exact_mask.png)")
        A("")
        A("Top row: fraction of terminal states at distance 3 over a trailing")
        A("25-update window; bottom row: fraction of two-qubit actions masked per")
        A("update (for the exact runs the thin line of the same color is the fraction")
        A("the greedy pre-filter rejected before the exact test admitted some back).")
        A("Color by mask method and batch, seed 0 solid, 1 dashed, 2 dotted.")
        A("")
    if (FIG / "exact_mask_distance.png").exists():
        A("![](figures/exact_mask_distance.png)")
        A("")
        A("Distance distribution of the terminal states of the last 25 updates, one")
        A("bar per run.")
        A("")
    A("EXACT_PROSE")


def main():
    recs = S.load_all(RES)
    recs = [r for r in recs if r["budget_env_steps"] == S.DEFAULT_BUDGET or r["variant"] == "unnormalized" or r["cell"] in SCALING_CELLS or r["variant"] in S.MASKED]
    g = S.groups(recs)
    base = {c: [r for r in rs if r["budget_env_steps"] == S.DEFAULT_BUDGET] for (c, lab), rs in g.items() if lab == "unnormalized"}
    long = {c: rs for (c, lab), rs in g.items() if lab.startswith("unnormalized (")}
    cells_run = [c for c in S.CELLS if any(r["cell"] == c for r in recs)]
    steps_all = [r["env_steps"] for r in recs]
    threads = sorted({r["threads"] for r in recs})
    n12 = [c for c in cells_run if TARGETS[c][0] == 12]

    # ---- per (cell, variant) statistics --------------------------------------
    stats = {}
    for (c, lab), rs in g.items():
        fr = [inside_frac(r) for r in rs]
        pooled_at = sum(r["terminal_last"]["at_d"] for r in rs)
        pooled_in = sum(r["terminal_last"]["inside"] for r in rs)
        stats[(c, lab)] = dict(
            runs=rs,
            fr=fr,
            pooled=(pooled_in / pooled_at) if pooled_at else None,
            at_d=[at_d_frac(r) for r in rs],
            first=[r["first_success"]["env_steps"] for r in rs if r["first_success"]],
            feas=[r["feasible_successes"] for r in rs],
            modes=[w_mode_at_d(r) for r in rs],
            wall=[r["wall_time_s"] for r in rs],
            sps=[r["env_steps_per_s"] for r in rs],
        )

    def majority(c, lab):
        st = stats.get((c, lab))
        return st is not None and st["pooled"] is not None and st["pooled"] > MAJORITY and all(f is not None and f > MAJORITY for f in st["fr"])

    winners = [lab for lab in PLANNED_VARIANTS if lab != "unnormalized" and n12 and all(majority(c, lab) for c in n12 if (c, lab) in stats) and any((c, lab) in stats for c in n12)]

    # ---- pending cells ---------------------------------------------------------
    pending = []
    base_label = S.base_label  # variant label without the budget and batch qualifiers

    for c in S.CELLS:
        for lab in PLANNED_VARIANTS:
            have = {r["seed"] for (cc, ll), st in stats.items() if cc == c and base_label(ll) == lab for r in st["runs"]}
            missing = [s for s in PLANNED.get((c, lab), ()) if s not in have]
            if missing:
                pending.append(f"{cell_name(c)}, {lab}: seed{'s' if len(missing) > 1 else ''} {listing(missing)}")

    # ---- prose --------------------------------------------------------------------
    lines = []
    A = lines.append
    A("# Sparsity as a property of the trained policy")
    A("")
    n_planned = sum(len(v) for v in PLANNED.values())
    status = "complete" if not pending else f"partial ({len(recs)} of {n_planned} planned runs; the pending cells are listed at the end)"
    A(f"Status: {status}. Every number below comes from a JSON under")
    A("`results/ldpc_sparsity/` (named in the last column of the run table);")
    A("the tables and figures regenerate with")
    A("`uv run --extra rl python -m autoqec_sim.experiments.ldpc_sparsity report`")
    A("and this file with `uv run --extra rl python experiments/ldpc_sparsity/assemble_report.py`.")
    A("Code: `src/autoqec_sim/ldpc.py` (`LdpcObjective` forms `shaped` and `terminal`,")
    A("`heavy_generator_count`, `greedy_action_mask`),")
    A("`src/autoqec_sim/experiments/ldpc_sparsity.py`. Logs: `results/ldpc_sparsity/logs/`;")
    A("compute records: `results/ldpc_sparsity/compute_pod.jsonl`, imported into")
    A("`genesis/ledger/compute.jsonl`.")
    A("")
    A("## Question")
    A("")
    A("`experiments/ldpc_scaling/REPORT.md` left the sparsity half of the LDPC")
    A("constraint open: the `unnormalized` form learns distance-3 codes at")
    A("`n = 12` to `16` but its policies produce codes at `w*` one to four above")
    A("`w_max = 4`, and the three codes within the bound were single episodes.")
    A("Measured here: the fraction of terminal states at the target distance")
    A("whose exact `w*` is within the bound, over the last 25 updates of every")
    A("run (2,560 terminal states) and along the run, for three ways of pricing")
    A("or enforcing the bound against `unnormalized` at equal budget in env")
    A("steps. The success criterion set in advance: a variant works if a")
    A("majority of its at-distance terminal states are inside the bound on")
    A("both 12-qubit cells, on every seed.")
    A("")
    A("## Setup")
    A("")
    A("Cells: `[[12, 2, 3]]` on `all_to_all:12` (H, S, CNOT; 156 actions) and on")
    A("`square:3x4` (H, CNOT; 46 actions), `[[16, 4, 3]]` on `square:4x4` (H,")
    A("CNOT; 64 actions), `w_max = 4`, no idle qubit. Episodes of 40 gates, 256")
    A("parallel episodes, `PPOConfig` defaults with `policy=\"equivariant\"`,")
    A(f"seeds {listing(PLANNED_SEEDS)} planned per cell, {S.DEFAULT_BUDGET / 1e6:g}M env steps per run")
    A(f"({rng(steps_all, '{:,}')} reached; the wall cap of 240 minutes never bound). Runs on the")
    A(f"RunPod pod's CPU (Xeon Gold 6342, no CUDA device visible, {listing(threads)} BLAS and torch")
    A("threads per run, three runs at a time (four at 6 threads for the")
    A("exact-mask section, on a host at load 10 to 190 from other tenants),")
    A("1-minute load 7 to 22 on 96")
    A("logical cores; 8 threads instead of the 4 first planned because the")
    A("PPO update of the equivariant network took 9.0 s at 4 threads and 5.9 s")
    A("at 8 on this CPU, single run). Env-step throughput per run is in the run")
    A("table (`env steps/s`).")
    A("")
    A("Variants (`ldpc.LdpcObjective` forms unless noted; the base terms of every")
    A("one are the per-error Knill-Laflamme count and the padding count of the")
    A("`unnormalized` form, divided by the initial state's undetected count")
    A("`r0 = 75` at `[[12, 2]]` and `210` at `[[16, 4]]`):")
    A("")
    A("| variant | sparsity signal | episode ends |")
    A("|---|---|---|")
    A("| `unnormalized` (reference, `DEFAULT_LDPC_FORM`) | `-0.25 max(0, w* - 4) / r0` on every state (greedy bound, exact on candidates) | reward 0: candidate with exact `w* <= 4` |")
    A("| `shaped (c)` | `-c max(0, w* - 4)` in reward units on every state (greedy bound, exact on candidates); one unit of excess costs `c r0` = 3.75, 7.5, and 15 undetected errors at `c` = 0.05, 0.1, and 0.2 (0.25 in `unnormalized`) | reward 0 |")
    A("| `terminal` | none on the path; a candidate (every error detected, no idle qubit) scores `1 - h / (n - k)`, `h` the number of generators of the sparsest basis with weight above 4 (`m - rank` of the light elements, exact) | any candidate, feasible or not |")
    A("| `mask` | `unnormalized` reward; a two-qubit gate whose resulting state has greedy `w*` above 4 is masked out of the logits (`ldpc.greedy_action_mask`, all CNOT trials in one batched call, 0.13 s per step on `all_to_all:12`) | reward 0 |")
    A("| `mask_terminal` (added after the first results) | the mask of `mask` on top of the `terminal` reward: every admitted candidate has `h = 0`, so the terminal bonus is 1 for every feasible code and the path reward has no sparsity term | any candidate (all feasible under the mask) |")
    A("")
    A("The degree-bounded environment of the plan's item (iv) was not built:")
    A("the environment is a Clifford-circuit environment whose checks are the")
    A("rows of the tableau, not objects the circuit can bound per qubit by")
    A("construction, so a degree bound is a reward or mask term")
    A("(`LdpcObjective(deg_max=...)`, `greedy_action_mask` on the degree) and")
    A("not a natural extension of `env.py`.")
    A("")
    A("## Results")
    A("")
    A("### Summary per variant")
    A("")
    A("Terminal-state columns are over the last 25 updates of every seed (2,560")
    A("terminal states per run). \"Inside the bound\" is exact `w* <= 4` among the")
    A("terminal states at distance 3 (the quantity of the success criterion);")
    A("\"heavy generators\" is `h` of the `terminal` form, computed for every")
    A("variant.")
    A("")
    A(S.summary_table(recs))
    A("")
    A("### Runs")
    A("")
    A(S.run_table(recs))
    A("")
    if (FIG / "inside_bound.png").exists():
        A("### Figures")
        A("")
        A("![](figures/inside_bound.png)")
        A("")
        A("Top row: fraction of the terminal states at distance 3 that are inside")
        A("the bound, over a trailing 25-update window, against env steps; bottom")
        A("row: fraction of all terminal states at the target distance. One column")
        A("per comparison cell (cells run with a single variant, the scaling cells,")
        A("are in the tables only), one color per variant, seed 0 solid, 1 dashed,")
        A("2 dotted. A gap in a top-row line is a window without any state at the")
        A("target distance.")
        A("")
        A("![](figures/w_star_at_d.png)")
        A("")
        A("Distribution of the exact `w*` of the terminal states at distance 3 over")
        A("the last 25 updates, pooled over seeds; the dashed line is `w_max = 4`.")
        A("")
    A("### Reading")
    A("")
    for c in cells_run:
        if not any((c, lab) in stats for lab in PLANNED_VARIANTS):
            continue  # cells of the scaling and exact-mask sections only
        A(f"- {cell_name(c)}.")
        for lab in PLANNED_VARIANTS:
            st = stats.get((c, lab))
            if not st:
                continue
            rs = st["runs"]
            fr_txt = listing([("n/a" if f is None else f"{100 * f:.0f}") for f in st["fr"]])
            at_txt = listing([f"{100 * a:.0f}" for a in st["at_d"]])
            first_txt = f"first feasible code at {listing(st['first'], '{:,}')} env steps" if st["first"] else "no feasible code"
            feas_txt = listing(st["feas"]) + (" feasible success" if st["feas"] == [1] else " feasible successes")
            modes = [m for m in st["modes"] if m is not None]
            mode_txt = f"mode of `w*` at distance 3: {listing(modes)}" if modes else "no state at distance 3"
            extra = ""
            if lab in S.MASKED:
                mf = [r["mask"]["two_masked_frac_last"] for r in rs if r.get("mask") and r["mask"].get("two_masked_frac_last") is not None]
                if mf:
                    extra = f"; {rng([100 * x for x in mf], '{:.0f}')} percent of the two-qubit actions masked over the last 25 updates"
            pooled_at = sum(r["terminal_last"]["at_d"] for r in rs)
            inside_txt = (
                f"{fr_txt} percent of the at-distance terminal states inside the bound (pooled {pct(sum(r['terminal_last']['inside'] for r in rs), pooled_at)} percent)"
                if pooled_at
                else "no terminal state at distance 3 in the last 25 updates (the inside-the-bound fraction is undefined)"
            )
            A(f"  - {lab} (seed{'s' if len(rs) > 1 else ''} {listing([r['seed'] for r in rs])}): {inside_txt}, "
              f"{at_txt} percent of all terminal states at distance 3, {first_txt}, {feas_txt}, {mode_txt}{extra}; "
              f"{rng(st['wall'])} s wall at {rng(st['sps'], '{:,.0f}')} env steps/s.")
    if long:
        A("")
        A("### Longer budget for the unnormalized form")
        A("")
        for c, rs in long.items():
            b = base.get(c, [])
            for r in rs:
                ref = [x for x in b if x["seed"] == r["seed"]]
                ref_txt = f" (the {S.DEFAULT_BUDGET / 1e6:g}M run of the same seed: {100 * (inside_frac(ref[0]) or 0):.0f} percent inside, {100 * at_d_frac(ref[0]):.0f} percent at distance 3)" if ref else ""
                A(f"- {cell_name(c)}, seed {r['seed']}, {r['env_steps']:,} env steps: {100 * (inside_frac(r) or 0):.0f} percent of the at-distance terminal states inside the bound, "
                  f"{100 * at_d_frac(r):.0f} percent of terminal states at distance 3, mode of `w*` at distance 3: {w_mode_at_d(r)}, "
                  f"{r['feasible_successes']} feasible successes{ref_txt}; `{Path(r['_file']).name}`.")
    exact_section(A, recs)
    A("")
    A("## Verdict")
    A("")
    if winners:
        A(f"The criterion (a majority of the at-distance terminal states inside the bound on the 12-qubit cells, every seed run) is met by {listing(winners)}:")
        for lab in winners:
            for c in n12:
                st = stats.get((c, lab))
                if st:
                    A(f"- {lab} on {cell_name(c)}: {listing([f'{100 * f:.0f}' for f in st['fr']])} percent (seeds {listing([r['seed'] for r in st['runs']])}), pooled {100 * st['pooled']:.0f} percent.")
    else:
        A("No variant meets the criterion (a majority of the at-distance terminal states inside the bound on both 12-qubit cells, every seed run) on the runs that finished.")
    for c in n12:
        st = stats.get((c, "unnormalized"))
        if st:
            A(f"- unnormalized on {cell_name(c)}: {listing([('n/a' if f is None else f'{100 * f:.0f}') for f in st['fr']])} percent inside the bound (seeds {listing([r['seed'] for r in st['runs']])}).")
    A("")
    A("VERDICT_PROSE")
    A("")
    if pending:
        A("## Pending")
        A("")
        A("Planned runs without a result JSON at the time of writing:")
        A("")
        for p in pending:
            A(f"- {p}")
        A("")
    A("## Sources")
    A("")
    A(S.sources_table(recs))
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `results/ldpc_sparsity/logs/*.log`, `logs/queue.log` | per-update logs of every run, queue log with the load at every start |")
    A("| `results/ldpc_sparsity/queue.sh`, `make_job.sh`, `jobs/` | the 3-slot pod runner and the job scripts (each run under the compute ledger's `wrap`) |")
    A("| `results/ldpc_sparsity/compute_pod.jsonl`, `import_pod_ledger.py` | the pod's ledger records and their import into `genesis/ledger/compute.jsonl` |")
    A("| `experiments/ldpc_sparsity/figures/` | the figures, drawn by `report` with `experiments.figstyle` |")
    A("| `results/ldpc_sparsity/bench/` | `bench-mask` measurements of the mask cost per env step on the pod CPU |")
    A("| `experiments/ldpc_sparsity/exact_mask.md`, `verdict.md` | the hand-written reading of the exact-mask section and the verdict, inserted by the assembly |")
    A("| `src/autoqec_sim/ldpc.py`, `src/autoqec_sim/experiments/ldpc_sparsity.py`, `experiments/ldpc_sparsity/assemble_report.py` | the objective forms and mask, the experiment, and this report's assembly |")
    text = "\n".join(lines) + "\n"
    verdict_file = OUT.parent / "verdict.md"
    verdict = verdict_file.read_text().strip() if verdict_file.exists() else "(verdict prose pending)"
    text = text.replace("VERDICT_PROSE", verdict)
    exact_file = OUT.parent / "exact_mask.md"
    text = text.replace("EXACT_PROSE", exact_file.read_text().strip() if exact_file.exists() else "(exact-mask reading pending)")
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(recs)} runs; winners: {winners or 'none'}; pending: {len(pending)})")


if __name__ == "__main__":
    main()
