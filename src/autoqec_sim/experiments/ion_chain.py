# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""PPO code discovery under the trapped-ion profile on all-to-all
connectivity at 16 qubits, against the 4x4 lattice cells of
``experiments.gpu_scale``.

Question. The noise-aware 4x4 cells (``results/gpu_scale/REPORT.md``) train
on the square lattice under three profiles, one of them the IonQ Forte
sequential profile. An ion chain has all-to-all connectivity, so the
lattice is an artificial restriction for that profile. Does lifting it
change the code the agent finds, or its undetected fault mass under the
ion profile?

Settings (both ``n = 16``, ``k = 1``, target ``d = 4``, ``all_to_all:16``,
50-gate episodes, the fractional distance term of the 4x4 setting):

``a2a16``
    gates H, S, CNOT. On an ion, S is a virtual frame rotation on the
    hardware; in the Forte preset it is charged like every single-qubit
    gate (``p1 = 2e-4`` and a 110 us single-qubit layer of idle
    dephasing), a twentieth of a two-qubit gate.
``a2a16_hc``
    gates H, CNOT: the gate set of the 4x4 cells, so the only difference
    to ``noise4x4`` is the graph.

Everything else is the ``noise4x4`` protocol of :mod:`gpu_scale`: the
reward of :class:`noise_aware.NoiseAwareReward` (fractional distance term
below the target, padding penalty, encoder undetected fault mass under the
training profile, LDPC penalty above ``w* = 4``) with its distance and
padding terms on the GPU (:class:`gpu_scale.TorchNoiseAwareReward`), the
three profiles ``board``, ``ion``, ``heron`` calibrated by
:func:`noise_aware.calibrate` to equal total fault mass on random circuits
of the new graph, ``PPOConfig(policy="auto")`` (equivariant), 1024
parallel episodes, 8M env steps, a 40 min wall cap, three seeds, the same
best-code selection and code analysis (:func:`noise_aware.analyze_circuit`,
with the undetected mass under all three profiles).

The report writes the cross-evaluation matrix (training profile against
evaluation profile) per setting and, for the ion profile, compares the
all-to-all codes with (a) the 4x4 lattice codes of
``results/gpu_scale/run_noise4x4_*.json`` (read, not rerun), (b) the
rotated surface ``[[16, 1, 4]]`` with its polished lattice encoder from
``results/noise_aware/baselines_4x4.json`` and, when
``results/ion_chain/baselines_<setting>.json`` exists (``baselines``
subcommand: :func:`noise_aware.run_baselines` on the all-to-all setting,
where stim's elimination encoder is a further seed), its polished
all-to-all encoder, and (c) the ``n = 16`` codes of
:mod:`autoqec_sim.known_codes`.

Files: ``results/ion_chain/run_<setting>_<profile>_seed<k>.json`` and a log
under ``results/ion_chain/logs/``; ``report`` writes
``results/ion_chain/REPORT.md`` and the figures under
``results/ion_chain/figures/``. Pod commands as in ``docs/gpu_run.md``
with this module in place of ``gpu_scale``::

    python -m autoqec_sim.experiments.ion_chain matrix --device cuda --threads 8 --profiles ion
    python -m autoqec_sim.experiments.ion_chain report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import socket
import time
import warnings
from pathlib import Path

import numpy as np

from .. import graphs
from ..code import StabilizerCode
from . import gpu_scale as gs
from . import noise_aware as na
from .noise_aware import _fmt, _json_default, _mean_range

DEFAULT_OUT = Path("results/ion_chain")
DEFAULT_LATTICE = Path("results/gpu_scale")
DEFAULT_BASELINES = Path("results/noise_aware/baselines_4x4.json")
PROFILES = na.PROFILES
SEEDS = (0, 1, 2)
BATCH = 1024
STEPS = 8_000_000
WALL_MIN = 40.0
THREADS = 8
LOG_EVERY = 25

SETTINGS = {
    "a2a16": na.Setting("a2a16", 16, 1, 4, "all_to_all:16", ("H", "S", "CNOT"), 50, STEPS, fractional=True),
    "a2a16_hc": na.Setting("a2a16_hc", 16, 1, 4, "all_to_all:16", ("H", "CNOT"), 50, STEPS, fractional=True),
}
SETTING_LABELS = {"a2a16": "all-to-all, H S CNOT", "a2a16_hc": "all-to-all, H CNOT"}
LATTICE_LABEL = "4x4 lattice, H CNOT (gpu_scale noise4x4)"


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------
def run_cell(
    setting: na.Setting,
    profile_name: str,
    seed: int,
    device: str = "cpu",
    batch: int = BATCH,
    steps: int | None = None,
    wall_min: float = WALL_MIN,
    threads: int = THREADS,
    chunk: int | None = None,
    verbose: bool = False,
    log: gs.RunLog | None = None,
    calib: dict | None = None,
    log_every: int = LOG_EVERY,
) -> dict:
    """One (setting, profile, seed) cell: the ``noise4x4`` training loop of
    :func:`gpu_scale.run_noise4x4` on the given setting. Trains the full
    budget and analyses the best and the modal code; the record keeps the
    field names of the ``noise4x4`` records so the same report code reads
    both."""
    import torch

    steps = int(setting.budget if steps is None else steps)
    wall_s = 60.0 * float(wall_min)
    dev = gs.resolve_device(device)
    chunk = int(chunk or gs.default_chunk(dev))
    log = log or gs.RunLog(None, verbose)
    started = _now()
    log(
        f"{setting.name} {profile_name} seed {seed}: device {dev} ({gs.device_name(dev)}), graph {setting.graph_spec}, "
        f"gates {', '.join(setting.gates)}, batch {batch}, budget {steps:,} env steps, wall cap {wall_s / 60:.0f} min"
    )
    t_setup = time.perf_counter()
    calib = calib or na.calibrate(setting)
    env = na.make_env(setting, seed, batch)
    models = na.calibrated_models(setting, calib, env.graph)
    if dev.type == "cpu":
        reward = na.NoiseAwareReward(env, setting, models[profile_name], calib["m0"])
        kernel = None
    else:
        reward = gs.TorchNoiseAwareReward(env, setting, models[profile_name], calib["m0"], device=dev, chunk=chunk)
        kernel = reward.kernel
    reward_timer = gs.Timed(reward)
    env.reward_fn = reward_timer
    env.success_fn = reward.success_fn
    tracker = na.CandidateTracker(setting.d, fractional=setting.fractional)
    env.tracker = tracker
    env.reset()
    agent, config, update_timer = gs._make_agent(env, seed, dev, threads, steps, wall_s)
    log(f"  policy {agent.policy}, {agent.num_parameters():,} parameters, {env.num_actions} actions, setup {time.perf_counter() - t_setup:.1f} s")

    def cb(agent, new):
        tracker.snapshot()
        rec = agent.history[-1]
        rec["successes_in_update"] = len(new)
        rec["distinct_success_codes"] = len(tracker.success)
        rec["best_terminal_reward"] = tracker.best_reward()
        rec["noise_evals"] = reward.noise_evals
        rec["cache_hits"] = reward.cache_hits
        if agent.updates % log_every == 0 or new:
            log(
                f"  upd {rec['update']:5d} steps {rec['env_steps']:11,d} r {rec['mean_reward']:+.4f} succ {rec['success_rate']:.3f} "
                f"ent {rec['entropy']:.3f} distinct {len(tracker.success)} best {tracker.best_reward() if tracker.success else float('nan'):+.4f} "
                f"t {rec['wall_time_s']:.0f}s ({rec['env_steps'] / max(rec['wall_time_s'], 1e-9):,.0f} steps/s)"
            )
        return False

    res = agent.train(steps, max_time_s=wall_s, stop_on_success=False, callback=cb, verbose=False)
    timing = gs._timing(agent, res, reward_timer, update_timer, kernel)
    log(f"  trained: {timing['env_steps']:,} env steps in {timing['wall_time_s']} s ({timing['env_steps_per_s']:,} steps/s); reward {timing['reward_time_s']} s, update {timing['update_time_s']} s")

    t1 = time.perf_counter()
    if tracker.success:
        best = max(tracker.success.values(), key=lambda v: v["reward"])
        selection = "highest reward among states that reached the target distance"
    else:
        cands = list(tracker.partial.values())
        for c in cands:
            c["mass_train"] = reward.mass([tuple(g) for g in c["gates"]])
        cands.sort(key=lambda c: c["mass_train"])
        best = cands[0] if cands else None
        selection = (
            f"no state reached distance {setting.d}; smallest undetected mass among the "
            f"{len(cands)} distinct distance-{tracker.partial_d} terminal states kept"
        )
    analysis = na.analyze_circuit(best["gates"], setting, models) if best else None
    if analysis is not None:
        m = analysis["mass"][profile_name]["undetected"]
        analysis["reward_terms"] = {
            "distance_term": min(analysis["distance"], setting.d) / setting.d - 1.0,
            "padding_term": -analysis["weight_enumerator_S"][1] / (3 * setting.n),
            "noise_term": -na.NOISE_WEIGHT * m / calib["m0"],
            "ldpc_term": -na.LDPC_SCALE * max(0, analysis["w_star"] - na.W_MAX),
        }
    modal, modal_visits, window_visits = tracker.modal_code()
    modal_analysis = None
    if modal is not None:
        modal_analysis = na.analyze_circuit(modal["gates"], setting, models)
        modal_analysis["reward"] = modal["reward"]
        modal_analysis["visits_last_20_updates"] = modal_visits
        modal_analysis["successes_last_20_updates"] = window_visits
    if analysis:
        log(
            f"  best: d={analysis['distance']} dX={analysis['distance_x']} dZ={analysis['distance_z']} w*={analysis['w_star']} "
            f"css={analysis['is_css']} S gates={analysis['circuit']['s_gates']} mass board/ion/heron="
            + "/".join(f"{analysis['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
            + f" ({analysis['circuit']['gates']} gates); analysis {time.perf_counter() - t1:.1f} s"
        )
    return {
        "kind": "run",
        "target": "ion_chain",
        "target_description": f"noise-aware {setting.name} on {setting.graph_spec} with policy=auto",
        "setting": setting.to_dict(),
        "profile": profile_name,
        "seed": seed,
        "started_at": started,
        "finished_at": _now(),
        "host": socket.gethostname(),
        "device": str(dev),
        "device_name": gs.device_name(dev),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if dev.type == "cuda" else None,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "batch": batch,
        "budget_env_steps": steps,
        "wall_cap_s": wall_s,
        "threads": threads,
        "chunk": chunk,
        "profile_description": na.profile_description(profile_name),
        "scale": calib["profiles"][profile_name]["scale"],
        "m0": calib["m0"],
        "calibration": calib,
        "reward": gs.reward_description(setting),
        "ppo_config": dict(config.to_dict(), policy_resolved=agent.policy),
        "num_parameters": res.num_parameters,
        "num_actions": env.num_actions,
        **timing,
        "success": bool(tracker.success),
        "successful_episodes": tracker.success_episodes,
        "distinct_success_codes": len(tracker.success),
        "partial_distance": tracker.partial_d if not tracker.success else None,
        "distinct_partial_codes": len(tracker.partial) if not tracker.success else None,
        "noise_evals": reward.noise_evals,
        "cache_hits": reward.cache_hits,
        "first_success": gs._first_success(agent, res),
        "first_success_env_steps": res.first_success.env_steps if res.first_success else None,
        "final_success_rate": res.history[-1]["success_rate"] if res.history else None,
        "selection": selection,
        "best": analysis,
        "best_reward": best["reward"] if best and tracker.success else None,
        "modal": modal_analysis,
        "analysis_time_s": round(time.perf_counter() - t1, 1),
        "history": res.history,
    }


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def run_name(setting: str, profile: str, seed: int) -> str:
    return f"run_{setting}_{profile}_seed{seed}"


def run_path(out_dir: Path, setting: str, profile: str, seed: int) -> Path:
    return out_dir / (run_name(setting, profile, seed) + ".json")


def log_path(out_dir: Path, setting: str, profile: str, seed: int) -> Path:
    return out_dir / "logs" / (run_name(setting, profile, seed) + ".log")


def run_one(setting_name: str, profile: str, seed: int, out_dir: Path, **kw) -> dict:
    log = gs.RunLog(log_path(out_dir, setting_name, profile, seed), verbose=kw.pop("verbose", False))
    try:
        rec = run_cell(SETTINGS[setting_name], profile, seed, log=log, **kw)
    finally:
        log.close()
    gs.write(run_path(out_dir, setting_name, profile, seed), rec)
    return rec


def _load_json(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"skipping unreadable {p} (still being written?)")
        return None


def load_runs(out_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(out_dir.glob("run_*.json")):
        r = _load_json(p)
        if r is None:
            continue
        r["_file"] = p.name
        recs.append(r)
    return recs


def load_lattice_runs(lattice_dir: Path) -> list[dict]:
    """The ``noise4x4`` records of ``gpu_scale`` (the lattice comparison)."""
    recs = []
    for p in sorted(lattice_dir.glob("run_noise4x4_*.json")):
        r = _load_json(p)
        if r is None:
            continue
        r["_file"] = str(p)
        recs.append(r)
    return recs


def load_baselines(out_dir: Path, lattice_path: Path) -> dict:
    """``{"4x4": ..., "a2a16": ..., ...}``: the lattice baselines and any
    all-to-all baselines written by the ``baselines`` subcommand."""
    out = {}
    if lattice_path.exists():
        b = _load_json(lattice_path)
        if b:
            out[b["setting"]["name"]] = b
    for p in sorted(out_dir.glob("baselines_*.json")):
        b = _load_json(p)
        if b:
            out[b["setting"]["name"]] = b
    return out


def known_codes_16() -> dict:
    """The ``n = 16`` entries of :mod:`autoqec_sim.known_codes` with their
    parameters, stabilizers, and, for each, the name of the baseline code
    with the same stabilizer group (if any)."""
    from .. import known_codes

    out = {}
    for name, (stabs, (n, k, d)) in known_codes.KNOWN.items():
        if n != 16:
            continue
        code = StabilizerCode.from_stabilizers(stabs)
        out[name] = {"n": n, "k": k, "d": d, "stabilizers": list(stabs), "key": na.code_key(code.to_tableau()[0], k)}
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _yes(b) -> str:
    return "yes" if b else "no"


def _steps(x) -> str:
    return "n/a" if x is None else f"{int(x):,}"


def _by_profile(recs: list[dict]) -> dict:
    return {p: sorted([r for r in recs if r["profile"] == p], key=lambda r: r["seed"]) for p in PROFILES}


def where_table(recs: list[dict]) -> list[str]:
    L = [
        "| setting | profile | seed | device | policy | parameters | actions | batch | budget (env steps) | env steps | updates | wall (s) | env steps/s | reward share | update share | wall cap hit | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        wall = max(float(r["wall_time_s"]), 1e-9)
        L.append(
            f"| {r['setting']['name']} | {r['profile']} | {r['seed']} | {r['device']} ({r['device_name']}) | {r['ppo_config'].get('policy_resolved', '?')} | "
            f"{r['num_parameters']:,} | {r['num_actions']} | {r['batch']} | {r['budget_env_steps']:,} | {r['env_steps']:,} | {r['updates']} | {r['wall_time_s']} | "
            f"{r['env_steps_per_s']:,} | {100 * r['reward_time_s'] / wall:.0f}% | {100 * r['update_time_s'] / wall:.0f}% | {_yes(r['hit_wall_cap'])} | `{r['_file']}` |"
        )
    L.append("")
    return L


def calibration_table(recs_by_setting: dict) -> list[str]:
    L = [
        "| setting | profile | preset | scale | total mass at scale 1 (mean over 64 random circuits) | M_0 |",
        "|---|---|---|---|---|---|",
    ]
    for sname, recs in recs_by_setting.items():
        c = recs[0]["calibration"]
        for p in PROFILES:
            cp = c["profiles"][p]
            L.append(f"| {sname} | {p} | {cp['description']} | {cp['scale']:.3f} | {cp['mass_scale1_mean']:.4f} | {c['m0']:.4f} |")
    L.append("")
    return L


def setting_sections(sname: str, recs: list[dict], baselines: dict, lattice: list[dict]) -> list[str]:
    """Runs table, cross-evaluation matrix, per-seed table, mechanism
    breakdown, modal codes, and the seed-level comparison for one setting."""
    st = SETTINGS[sname]
    prof = _by_profile(recs)
    L = [f"## Setting {sname}: n = {st.n}, k = 1, target d = {st.d}, `{st.graph_spec}`, gates {', '.join(st.gates)}\n"]
    L.append("### Runs\n")
    L.append("| profile | seeds | budget (env steps) | env steps reached | wall per seed (s) | env steps/s | seeds reaching d* | first success (env steps) | final success rate | distinct target codes per seed | noise evaluations per seed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PROFILES:
        lst = prof[p]
        if not lst:
            continue
        reached = sum(1 for r in lst if r["distinct_success_codes"] > 0)
        L.append(
            f"| {p} | {len(lst)} | {lst[0]['budget_env_steps']:,} | {_mean_range([float(r['env_steps']) for r in lst])} | "
            f"{_mean_range([float(r['wall_time_s']) for r in lst])} | {_mean_range([float(r['env_steps_per_s']) for r in lst if r['env_steps_per_s']])} | "
            f"{reached}/{len(lst)} | {_mean_range([r['first_success_env_steps'] for r in lst])} | "
            f"{_mean_range([r['final_success_rate'] for r in lst])} | {_mean_range([float(r['distinct_success_codes']) for r in lst])} | "
            f"{_mean_range([float(r['noise_evals']) for r in lst])} |"
        )
    L.append("")
    caps = [f"{r['profile']}/seed {r['seed']}" for r in recs if r["hit_wall_cap"]]
    L.append(("Runs stopped by the wall cap before their step budget: " + ", ".join(caps) + ".") if caps else "No run hit its wall cap.")
    L.append("")

    have = {p: [r for r in prof[p] if r["best"] and r["success"]] for p in PROFILES}
    lat = _by_profile(lattice)
    lat_have = {p: [r for r in lat[p] if r["best"] and r["success"]] for p in PROFILES}
    L.append("### Cross-evaluation matrix\n")
    L.append(
        "Rows: training profile (the best code of each seed that reached d = 4, its own encoding circuit). Columns: "
        "evaluation profile at the calibrated scale of this setting. Entries: undetected fault mass of the encoder "
        "(`circuit_noise.undetected_fault_mass`, absolute probability), mean over seeds with [min, max]. Smaller is better. "
        "The `lattice` rows are the 4x4 cells of `results/gpu_scale/` (same reward, profiles, batch, budget, and wall cap; "
        "graph `square:4x4`, gates H, CNOT), read from their JSON files. The `known` rows are the polished encoders of "
        "`results/noise_aware/baselines_4x4.json` (routed on the lattice) and, where present, of "
        f"`baselines_{sname}.json` (all-to-all encoders of the same codes).\n"
    )
    L.append("| trained on \\ evaluated on | " + " | ".join(PROFILES) + " | seeds with d = 4 | gates (mean) | S gates (mean) |")
    L.append("|---|" + "---|" * (len(PROFILES) + 3))
    for p in PROFILES:
        lst = have[p]
        if not lst:
            L.append(f"| {p} | " + " | ".join("n/a" for _ in PROFILES) + f" | 0 of {len(prof[p])} | n/a | n/a |")
            continue
        cells = [_mean_range([r["best"]["mass"][ev]["undetected"] for r in lst]) for ev in PROFILES]
        L.append(
            f"| {p} | " + " | ".join(cells) + f" | {len(lst)} of {len(prof[p])} | "
            f"{np.mean([r['best']['circuit']['gates'] for r in lst]):.1f} | {np.mean([r['best']['circuit']['s_gates'] for r in lst]):.1f} |"
        )
    for p in PROFILES:
        lst = lat_have[p]
        if not lst:
            continue
        cells = [_mean_range([r["best"]["mass"][ev]["undetected"] for r in lst]) for ev in PROFILES]
        L.append(f"| lattice 4x4, {p} | " + " | ".join(cells) + f" | {len(lst)} of {len(lat[p])} | {np.mean([r['best']['circuit']['gates'] for r in lst]):.1f} | 0 |")
    for bname, b in baselines.items():
        if bname not in ("4x4", sname):
            continue
        where = "lattice encoder" if bname == "4x4" else "all-to-all encoder"
        for cname, c in b["codes"].items():
            bb = c["encoders"]["best_by_profile"]
            cells = [f"{bb[ev]['undetected']:.4g} ({bb[ev]['circuit']['gates']} gates)" if ev in bb else "no encoder" for ev in PROFILES]
            L.append(f"| known: {cname}, {where} | " + " | ".join(cells) + f" | d = {c['distance']}, w* = {c['w_star']}, CSS {_yes(c['is_css'])} | | |")
    L.append("")

    L.append("Per seed (undetected mass under board / ion / heron of the best code; a seed without a d = 4 state shows its best lower-distance state and is not comparable):\n")
    L.append("| trained on | seed | reached d* | d | d_X | d_Z | w* | CSS | frame | board | ion | heron | gates | CNOTs | H | S | depth | first success (env steps) | file |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PROFILES:
        for r in prof[p]:
            a = r["best"]
            if not a:
                L.append(f"| {p} | {r['seed']} | no | none | | | | | | | | | | | | | | n/a | `{r['_file']}` |")
                continue
            fr = a["frame"]
            frame = "CSS" if a["is_css"] else (f"{fr['search']} frame, {fr['frame_size']} qubits" if fr.get("css_frame_found") else "none")
            c = a["circuit"]
            L.append(
                f"| {p} | {r['seed']} | {_yes(r['success'])} | {a['distance']} | {_fmt(a['distance_x'])} | {_fmt(a['distance_z'])} | {a['w_star']} | "
                f"{_yes(a['is_css'])} | {frame} | " + " | ".join(f"{a['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                + f" | {c['gates']} | {c['two_qubit_gates']} | {c['h_gates']} | {c['s_gates']} | {c['depth']} | {_steps(r['first_success_env_steps'])} | `{r['_file']}` |"
            )
    L.append("")

    L.append("### Mechanism breakdown of the undetected mass\n")
    if any(have.values()):
        L += na.breakdown_table({p: have[p] for p in PROFILES})
    modal = [(p, r) for p in PROFILES for r in prof[p] if r.get("modal")]
    if modal:
        L.append("Most visited target-distance code of the last 20 updates (the converged policy's code):\n")
        L.append("| trained on | seed | visits (of successes in the window) | d | d_X | d_Z | w* | CSS | board | ion | heron | gates | S gates | same as best |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p, r in modal:
            m = r["modal"]
            same = r["best"] is not None and m["stabilizers"] == r["best"]["stabilizers"]
            L.append(
                f"| {p} | {r['seed']} | {m['visits_last_20_updates']} ({m['successes_last_20_updates']}) | {m['distance']} | {_fmt(m['distance_x'])} | "
                f"{_fmt(m['distance_z'])} | {m['w_star']} | {_yes(m['is_css'])} | " + " | ".join(f"{m['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                + f" | {m['circuit']['gates']} | {m['circuit']['s_gates']} | {_yes(same)} |"
            )
        L.append("")

    L.append("### Does training on the evaluation profile help? Seed-level comparison\n")
    L.append(
        "For each evaluation profile B: undetected mass of the codes trained on B against those trained on A, evaluated on B. "
        "`pairs` counts seed pairs (i, j) with mass(B-trained_i) < mass(A-trained_j); p is the exact two-sided permutation "
        "p-value of the difference of means. With three seeds per cell there are 20 relabelings, so the smallest attainable "
        "p is 0.1 and no entry of this table is significant on its own; the direction and the pair counts are what it shows.\n"
    )
    L.append("| evaluated on B | trained on B (mean) | trained on A | A-trained mean | ratio A/B | pairs B < A | p |")
    L.append("|---|---|---|---|---|---|---|")
    for ev in PROFILES:
        own = [r["best"]["mass"][ev]["undetected"] for r in have[ev]]
        for other in PROFILES:
            if other == ev:
                continue
            oth = [r["best"]["mass"][ev]["undetected"] for r in have[other]]
            t = na.permutation_test(own, oth)
            if "mean_a" not in t:
                continue
            L.append(
                f"| {ev} | {t['mean_a']:.4g} | {other} | {t['mean_b']:.4g} | {t['ratio_b_over_a']:.2f} | "
                f"{t['pairs_a_below_b']}/{t['pairs']} | {t['p_value']:.3f} |"
            )
    L.append("")
    return L


def ion_families(recs: list[dict], lattice: list[dict], baselines: dict, known: dict) -> list[dict]:
    """The code families compared under the ion profile: every all-to-all
    cell trained on ion, the lattice cells trained on ion, and the known
    codes with their lattice and all-to-all encoders (the ion-polished
    encoder of each). Each family: label, kind, and a list of members with
    ``mass`` (under the three profiles) and code descriptors."""
    fams = []

    def member(a, seed=None, file=None):
        c = a["circuit"]
        return {
            "seed": seed,
            "d": a["distance"],
            "d_x": a["distance_x"],
            "d_z": a["distance_z"],
            "w_star": a["w_star"],
            "css": a["is_css"],
            "gates": c["gates"],
            "cnots": c["two_qubit_gates"],
            "h": c["h_gates"],
            "s": c["s_gates"],
            "depth": c["depth"],
            "mass": {ev: a["mass"][ev]["undetected"] for ev in PROFILES},
            "circuit": [tuple(g) for g in a["gates"]],
            "logical": 0,
            "file": file,
        }

    for sname in SETTINGS:
        lst = [r for r in recs if r["setting"]["name"] == sname and r["profile"] == "ion" and r["best"] and r["success"]]
        fams.append({"label": SETTING_LABELS[sname], "kind": "rl", "setting": sname, "members": [member(r["best"], r["seed"], r["_file"]) for r in sorted(lst, key=lambda r: r["seed"])]})
    lst = [r for r in lattice if r["profile"] == "ion" and r["best"] and r["success"]]
    fams.append({"label": LATTICE_LABEL, "kind": "rl", "setting": "4x4", "members": [member(r["best"], r["seed"], r["_file"]) for r in sorted(lst, key=lambda r: r["seed"])]})
    for bname, b in baselines.items():
        if bname not in ("4x4", "a2a16"):
            continue  # the H, CNOT all-to-all encoders coincide with the a2a16 ones (no encoder uses S)
        where = "lattice encoder" if bname == "4x4" else "all-to-all encoder"
        for cname, c in b["codes"].items():
            bb = c["encoders"]["best_by_profile"]
            if "ion" not in bb:
                continue
            e = bb["ion"]
            cc = e["circuit"]
            fams.append(
                {
                    "label": f"known: {cname}, {where}",
                    "kind": "known",
                    "setting": bname,
                    "code": cname,
                    "members": [
                        {
                            "seed": None,
                            "d": c["distance"],
                            "d_x": c["distance_x"],
                            "d_z": c["distance_z"],
                            "w_star": c["w_star"],
                            "css": c["is_css"],
                            "gates": cc["gates"],
                            "cnots": cc["two_qubit_gates"],
                            "h": cc["h_gates"],
                            "s": cc["s_gates"],
                            "depth": cc["depth"],
                            "mass": {ev: e["mass_all_profiles"][ev]["undetected"] for ev in PROFILES},
                            "circuit": [tuple(g) for g in e["gates"]],
                            "logical": e["logical_qubit"],
                            "file": None,
                        }
                    ],
                }
            )
    return fams


def common_calibration(recs: list[dict]) -> dict:
    """The calibration of the ``a2a16`` runs (or, before any exists, a fresh
    :func:`noise_aware.calibrate` of that setting): the common scale at
    which every code of the comparison is evaluated."""
    for r in recs:
        if r["setting"]["name"] == "a2a16":
            return r["calibration"]
    return na.calibrate(SETTINGS["a2a16"])


def add_common_masses(fams: list[dict], calib: dict) -> None:
    """Evaluate every member's encoder under the three profiles at the
    ``a2a16`` calibration (``mass_common``). The native ``mass`` of a
    lattice code or a lattice baseline was computed at the 4x4
    calibration, whose ion scale is about 8 percent lower, so the native
    numbers of different settings are not comparable; these are."""
    from ..circuit_noise import fault_set, gates_to_layered_circuit, undetected_fault_mass

    setting = SETTINGS["a2a16"]
    models = na.calibrated_models(setting, calib, graphs.from_spec(setting.graph_spec))
    for f in fams:
        for m in f["members"]:
            gates = m["circuit"]
            tab = na.code_from_circuit(gates, setting.n, m["logical"]).to_tableau()
            m["mass_common"] = {
                ev: undetected_fault_mass(tab, 1, fault_set(model(gates_to_layered_circuit(gates, setting.n)), setting.n)) for ev, model in models.items()
            }


def _known_note(known: dict, baselines: dict) -> str:
    if not known:
        return "`known_codes` has no code at n = 16."
    parts = []
    b = baselines.get("4x4")
    for name, kc in known.items():
        same = None
        if b:
            for cname, c in b["codes"].items():
                key = na.code_key(StabilizerCode.from_stabilizers(c["stabilizers"]).to_tableau()[0], 1)
                if key == kc["key"]:
                    same = cname
                    break
        parts.append(
            f"`known_codes.{name}` [[{kc['n']},{kc['k']},{kc['d']}]]"
            + (f" has the same stabilizer group as the baseline `{same}`, so its numbers are that row's" if same else " has no baseline encoder and is listed without a mass")
        )
    return "; ".join(parts) + "."


def comparison_section(recs: list[dict], fams: list[dict], baselines: dict, known: dict, calib: dict) -> list[str]:
    L = ["## All-to-all against the lattice under the ion profile\n"]
    L.append(
        "Codes trained on the ion profile (best code of each seed that reached d = 4) and the known codes with their "
        "ion-polished encoders. Every encoder here is re-evaluated under the three profiles at one calibration, that of "
        f"the `a2a16` setting (ion scale {calib['profiles']['ion']['scale']:.3f}, heron {calib['profiles']['heron']['scale']:.3f}, "
        f"M_0 = {calib['m0']:.4f}); the `native` column is the mass at the code's own setting's calibration (the 4x4 "
        "calibration has an ion scale of 0.391, so native lattice numbers sit about 8 percent below the common ones and "
        "must not be compared across settings). Smaller is better. " + _known_note(known, baselines) + "\n"
    )
    L.append("| family | seeds | d | d_X | d_Z | w* | CSS | gates (mean) | CNOTs | H | S | uses S | ion, common calibration: mean [min, max] | board, common | heron, common | ion, native |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in fams:
        ms = f["members"]
        if not ms:
            L.append(f"| {f['label']} | 0 | | | | | | | | | | | n/a | n/a | n/a | n/a |")
            continue

        def col(key):
            vals = sorted({str(m[key]) for m in ms})
            return ", ".join(vals)

        L.append(
            f"| {f['label']} | {len(ms)} | {col('d')} | {col('d_x')} | {col('d_z')} | {col('w_star')} | "
            f"{col('css')} | {np.mean([m['gates'] for m in ms]):.1f} | {np.mean([m['cnots'] for m in ms]):.1f} | {np.mean([m['h'] for m in ms]):.1f} | "
            f"{np.mean([m['s'] for m in ms]):.1f} | {_yes(any(m['s'] > 0 for m in ms))} | "
            f"{_mean_range([m['mass_common']['ion'] for m in ms])} | {_mean_range([m['mass_common']['board'] for m in ms])} | "
            f"{_mean_range([m['mass_common']['heron'] for m in ms])} | {_mean_range([m['mass']['ion'] for m in ms])} |"
        )
    L.append("")
    L.append("Per seed (common calibration; the last column is the native ion mass):\n")
    L.append("| family | seed | d | d_X | d_Z | w* | CSS | gates | CNOTs | H | S | depth | ion | board | heron | ion, native |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in fams:
        for m in f["members"]:
            L.append(
                f"| {f['label']} | {'-' if m['seed'] is None else m['seed']} | {m['d']} | {_fmt(m['d_x'])} | {_fmt(m['d_z'])} | {m['w_star']} | {_yes(m['css'])} | "
                f"{m['gates']} | {m['cnots']} | {m['h']} | {m['s']} | {m['depth']} | {m['mass_common']['ion']:.4g} | {m['mass_common']['board']:.4g} | "
                f"{m['mass_common']['heron']:.4g} | {m['mass']['ion']:.4g} |"
            )
    L.append("")

    rl = {f["setting"]: f for f in fams if f["kind"] == "rl"}
    L.append("### Seed-level tests under the ion profile (common calibration)\n")
    L.append(
        "Exact two-sided permutation test of the difference of means (20 relabelings for 3 against 3 seeds, so p >= 0.1 "
        "always; the pair count and whether the seed intervals overlap carry the information).\n"
    )
    L.append("| A | B | mean A | mean B | ratio B/A | pairs A < B | intervals overlap | p |")
    L.append("|---|---|---|---|---|---|---|---|")
    verdict = []
    for a_key, b_key in (("a2a16", "4x4"), ("a2a16_hc", "4x4"), ("a2a16", "a2a16_hc")):
        fa, fb = rl.get(a_key), rl.get(b_key)
        if not fa or not fb or not fa["members"] or not fb["members"]:
            continue
        ma = [m["mass_common"]["ion"] for m in fa["members"]]
        mb = [m["mass_common"]["ion"] for m in fb["members"]]
        t = na.permutation_test(ma, mb)
        overlap = not (max(ma) < min(mb) or max(mb) < min(ma))
        L.append(
            f"| {fa['label']} | {fb['label']} | {t['mean_a']:.4g} | {t['mean_b']:.4g} | {t['ratio_b_over_a']:.2f} | "
            f"{t['pairs_a_below_b']}/{t['pairs']} | {_yes(overlap)} | {t['p_value']:.3f} |"
        )
        if b_key == "4x4":
            verdict.append((fa["label"], t, overlap, ma, mb))
    L.append("")
    L.append("### Verdict\n")
    if not verdict:
        L.append("No ion-trained all-to-all cell has reached d = 4 yet, so there is nothing to compare against the lattice.\n")
    else:
        for label, t, overlap, ma, mb in verdict:
            lattice_mean, a2a_mean = t["mean_b"], t["mean_a"]
            change = 100 * (a2a_mean / lattice_mean - 1.0)
            L.append(
                f"- {label} against the lattice: ion mass {a2a_mean:.4g} [{min(ma):.3g}, {max(ma):.3g}] against "
                f"{lattice_mean:.4g} [{min(mb):.3g}, {max(mb):.3g}], a change of {change:+.0f} percent in the mean; "
                f"{t['pairs_a_below_b']} of {t['pairs']} seed pairs favour all-to-all; seed intervals "
                f"{'overlap' if overlap else 'are disjoint'}; p = {t['p_value']:.2f}."
            )
        best_a2a = min((m["mass_common"]["ion"], f["label"]) for f in fams if f["kind"] == "rl" and f["setting"] != "4x4" for m in f["members"])
        best_lat = min((m["mass_common"]["ion"], f["label"]) for f in fams if f["kind"] == "rl" and f["setting"] == "4x4" for m in f["members"])
        best_known = min(((m["mass_common"]["ion"], f["label"]) for f in fams if f["kind"] == "known" for m in f["members"]), default=None)
        L.append(
            f"- Best single code under ion (common calibration): all-to-all {best_a2a[0]:.4g} ({best_a2a[1]}) against lattice {best_lat[0]:.4g}"
            + (f"; the best known-code encoder is {best_known[0]:.4g} ({best_known[1]})" if best_known else "")
            + "."
        )
        L.append("")
    return L


def make_figures(recs: list[dict], fams: list[dict], out_dir: Path) -> list[str]:
    """(1) training curves per setting, as in ``noise_aware.make_figures``;
    (2) undetected mass by code family under the ion profile at the common
    calibration, and the same codes under board and heron."""
    from . import figstyle

    plt = figstyle.apply()
    paths = []
    by_setting = {s: [r for r in recs if r["setting"]["name"] == s] for s in SETTINGS}
    for sname, lst_all in by_setting.items():
        if not lst_all:
            continue
        prof = _by_profile(lst_all)
        fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE))
        max_steps = max(r["env_steps"] for r in lst_all)
        grid = np.linspace(0, max_steps, 300)
        any_best = False
        for pname in PROFILES:
            lst = prof[pname]
            if not lst:
                continue
            color = figstyle.PROFILE_COLORS[pname]
            series = {"a": [], "b": [], "c": []}
            for r in lst:
                h = r["history"]
                steps = np.array([x["env_steps"] for x in h], dtype=float)
                series["a"].append(na._on_grid(steps, [x["mean_reward"] for x in h], grid))
                series["b"].append(na._on_grid(steps, [x["success_rate"] for x in h], grid))
                best = np.array([np.nan if x.get("best_terminal_reward") is None else -x["best_terminal_reward"] for x in h])
                any_best |= bool(np.isfinite(best).any())
                c = np.interp(grid, steps, best)
                c[grid > steps[-1]] = np.nan
                series["c"].append(c)
            for ax, key in zip(axes, "abc"):
                arr = np.array(series[key])
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices before the first success
                    med = np.nanmedian(arr, axis=0)
                    lo = np.nanmin(arr, axis=0)
                    hi = np.nanmax(arr, axis=0)
                x = grid / 1e6
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
                ax.plot(x, med, color=color, lw=figstyle.LINE, label=f"{figstyle.PROFILE_LABELS[pname]} ({len(lst)} seeds)" if key == "b" else None)
        axes[0].set_ylabel("mean step reward (25-update mean)")
        axes[1].set_ylabel("success rate per update (25-update mean)")
        axes[2].set_ylabel("-(best terminal reward so far)")
        axes[1].set_ylim(-0.03, 1.03)
        if any_best:
            axes[2].set_yscale("log")
            axes[2].set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
            axes[2].yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axes[2].yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        for ax, letter in zip(axes, "abc"):
            ax.set_xlabel("env steps (millions)")
            figstyle.panel_label(ax, letter)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
        st = SETTINGS[sname]
        fig.suptitle(f"{sname}: n = {st.n}, target d = {st.d}, {st.graph_spec}, gates {', '.join(st.gates)}; median over seeds, band = min to max", y=1.02)
        fig.tight_layout()
        png, _ = figstyle.save(fig, out_dir / "figures" / f"curves_{sname}")
        plt.close(fig)
        paths.append(str(png.relative_to(out_dir)))

    # family comparison: one panel per evaluation profile, codes trained on that profile
    short = {"all-to-all, H S CNOT": "a2a HSC", "all-to-all, H CNOT": "a2a HC", LATTICE_LABEL: "lattice HC"}
    fl = []
    for f in fams:
        if f["kind"] == "rl":
            fl.append((short.get(f["label"], f["label"]), "rl", f["members"]))
        elif f.get("code") == "rotated_surface" or (f.get("code") == "transposed_shor" and f["setting"] == "a2a16"):
            enc = "lattice" if f["setting"] == "4x4" else "a2a"
            name = "RS" if f["code"] == "rotated_surface" else "tShor"
            fl.append((f"{name} {enc} enc.", "known", f["members"]))
    if any(ms for _, _, ms in fl):
        fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE + 0.7), sharey=True)
        for ax, ev, letter in zip(axes, ("ion", "board", "heron"), "abc"):
            for i, (_, kind, ms) in enumerate(fl):
                vals = [m["mass_common"][ev] for m in ms]
                if not vals:
                    continue
                color = figstyle.PROFILE_COLORS["ion"] if kind == "rl" else figstyle.NEUTRAL
                ax.plot([i] * len(vals), vals, "o" if kind == "rl" else "s", color=color, ms=figstyle.MARKER, alpha=0.8, lw=0)
                if len(vals) > 1:
                    ax.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=color, lw=figstyle.LINE)
            ax.set_xticks(range(len(fl)))
            ax.set_xticklabels([lbl for lbl, _, _ in fl], rotation=45, ha="right", fontsize=6.5)
            ax.set_xlim(-0.6, len(fl) - 0.4)
            ax.set_yscale("log")
            ax.set_title(f"evaluated on {ev}")
            figstyle.panel_label(ax, letter)
        axes[0].set_ylabel("undetected fault mass of the encoder")
        fig.suptitle("Ion-trained codes (best per seed, circles; bar = seed mean) and known-code encoders (squares), a2a16 calibration", y=1.02)
        fig.tight_layout()
        png, _ = figstyle.save(fig, out_dir / "figures" / "ion_families")
        plt.close(fig)
        paths.append(str(png.relative_to(out_dir)))
    return paths


def write_report(out_dir: Path, lattice_dir: Path = DEFAULT_LATTICE, baselines_path: Path = DEFAULT_BASELINES) -> Path:
    recs = load_runs(out_dir)
    lattice = load_lattice_runs(lattice_dir)
    baselines = load_baselines(out_dir, baselines_path)
    known = known_codes_16()
    calib = common_calibration(recs)
    fams = ion_families(recs, lattice, baselines, known)
    add_common_masses(fams, calib)
    figs = make_figures(recs, fams, out_dir) if recs else []
    by_setting = {s: [r for r in recs if r["setting"]["name"] == s] for s in SETTINGS}
    by_setting = {s: lst for s, lst in by_setting.items() if lst}
    L = ["# PPO code discovery on all-to-all connectivity at n = 16 under the trapped-ion profile\n"]
    L.append(
        f"Status: {len(recs)} runs recorded. Every number comes from a JSON under `results/ion_chain/` (last column of the "
        "run tables), from `results/gpu_scale/run_noise4x4_*.json` (the lattice cells, not rerun), or from "
        "`results/noise_aware/baselines_4x4.json`; the tables regenerate with "
        "`uv run --extra rl python -m autoqec_sim.experiments.ion_chain report`. Code: "
        "`src/autoqec_sim/experiments/ion_chain.py`.\n"
    )
    verdict = out_dir / "verdict.md"
    if verdict.exists():
        L.append("## Verdict\n")
        L.append(verdict.read_text().strip() + "\n")
    L.append("## Question and protocol\n")
    L.append(
        "The 4x4 cells of `results/gpu_scale/REPORT.md` train under the IonQ Forte sequential profile on the square lattice, "
        "a connectivity an ion chain does not have. The two settings here keep everything of those cells (n = 16, k = 1, "
        "target d = 4, 50-gate episodes, the fractional distance term, padding penalty, encoder undetected fault mass under "
        "the training profile, LDPC penalty above w* = 4, equivariant policy, batch 1024, 8M env steps, 40 min wall cap, "
        "three profiles calibrated to equal total fault mass on 64 random circuits of the setting) and change the graph to "
        "`all_to_all:16`. `a2a16` adds S to the gate set (H, S, CNOT); `a2a16_hc` keeps H, CNOT so that the only change "
        "against the lattice is the graph. S is not free in the Forte preset: every single-qubit gate is charged "
        "`p1 = 2e-4` (a twentieth of a two-qubit gate) and a 110 us single-qubit layer of idle dephasing; under the board "
        "preset single-qubit gates carry no channel. The all-to-all action space has 272 actions (16 H, 16 S, 240 ordered "
        "CNOTs) or 256 without S, against 64 on the lattice.\n"
    )
    L.append(
        "The undetected mass of a distance-4 encoder is a sum over few fault mechanisms (about ten of the two to three "
        "thousand of a 50-gate circuit), each with a probability from a small discrete set (a two-qubit gate fault at "
        "`scale p2 / 15`, an idle fault of a given slot count), so distinct codes with distinct stabilizer groups tie "
        "exactly, and a difference between two codes is a difference in the count and kind of surviving mechanisms, "
        "not a continuous quantity. Identical masses in the tables below are such ties, checked by recomputing them "
        "from the gate lists.\n"
    )
    L.append("### Profiles and calibration\n")
    if by_setting:
        L += calibration_table(by_setting)
    L.append("## What ran where\n")
    L.append(
        "`reward share` and `update share` are the fractions of the training wall time spent in the reward (GPU distance "
        "kernel plus the host-side terminal terms: stim fault propagation and exact w* on target-distance states) and in "
        "the PPO update.\n"
    )
    L += where_table(recs)
    for fpath in figs:
        L.append(f"![]({fpath})\n")
    if figs:
        L.append(
            "Training curves per setting (median over seeds, band = min to max, color by profile): (a) mean step reward, "
            "trailing mean over 25 updates, (b) fraction of finished episodes that reached the target distance, (c) minus the "
            "best terminal reward so far (the noise plus LDPC penalty of the best target-distance code, log axis). The last "
            "figure: undetected fault mass of the best code of every ion-trained seed by code family under (a) ion, (b) board, "
            "(c) heron at the a2a16 calibration (`a2a HSC` and `a2a HC` are the two all-to-all settings, `lattice HC` the 4x4 "
            "cells); the horizontal bar is the seed mean, squares are the polished encoders of the rotated surface code (`RS`) "
            "on the lattice and on all-to-all and of the transposed Shor code (`tShor`) on all-to-all.\n"
        )
    for sname, lst in by_setting.items():
        L += setting_sections(sname, lst, baselines, lattice)
    L += comparison_section(recs, fams, baselines, known, calib)
    L.append("## Best code per seed\n")
    for sname, lst in by_setting.items():
        prof = _by_profile(lst)
        for p in PROFILES:
            for r in prof[p]:
                a = r["best"]
                L.append(f"**{sname} / {p} / seed {r['seed']}**: {r['selection']}.")
                if not a:
                    L.append("")
                    continue
                c = a["circuit"]
                L.append(
                    f"[[{a['n']},{a['k']},{a['distance']}]], d_X = {_fmt(a['distance_x'])}, d_Z = {_fmt(a['distance_z'])}, w* = {a['w_star']} "
                    f"(degree {a['max_qubit_degree']}), CSS: {_yes(a['is_css'])}. Circuit: {c['gates']} gates "
                    f"({c['two_qubit_gates']} CNOT, {c['h_gates']} H, {c['s_gates']} S), depth {c['depth']}. "
                    "Undetected mass board/ion/heron = " + "/".join(f"{a['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                    + (
                        f". Reward terms: distance {a['reward_terms']['distance_term']:+.3f}, padding {a['reward_terms']['padding_term']:+.3f}, "
                        f"noise {a['reward_terms']['noise_term']:+.4f}, LDPC {a['reward_terms']['ldpc_term']:+.3f}."
                        if "reward_terms" in a
                        else "."
                    )
                )
                L.append("")
                L.append("Stabilizers (sparsest basis): `" + "`, `".join(a["sparsest_basis"]) + "`  ")
                L.append(f"Weight enumerator of S: {a['weight_enumerator_S']}; of N(S): {a['weight_enumerator_N']}  ")
                L.append("Gates: `" + " ".join(f"{g[0]}({g[1]}{',' + str(g[2]) if g[2] >= 0 else ''})" for g in a["gates"]) + "`")
                L.append("")
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(L))
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _common(p):
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--steps", type=int, default=None, help="env-step budget (default 8M)")
    p.add_argument("--wall", type=float, default=WALL_MIN, help="wall cap in minutes")
    p.add_argument("--chunk", type=int, default=None, help="Paulis per streamed chunk of the GPU distance kernel")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--verbose", action="store_true")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO code discovery on all-to-all connectivity at n = 16 under the ion profile.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate", help="print the profile scales of a setting")
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p = sub.add_parser("run", help="one cell: setting, profile, seed")
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p.add_argument("--profile", choices=list(PROFILES), required=True)
    p.add_argument("--seed", type=int, default=0)
    _common(p)
    p = sub.add_parser("matrix", help="the cells of the given settings, profiles, and seeds, in sequence")
    p.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    p.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--skip-existing", action="store_true")
    _common(p)
    p = sub.add_parser("baselines", help="known codes with all-to-all encoders for a setting (noise_aware.run_baselines)")
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p.add_argument("--polish-time", type=float, default=90.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("report", help="write REPORT.md and the figures")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--lattice-dir", default=str(DEFAULT_LATTICE))
    p.add_argument("--baselines", default=str(DEFAULT_BASELINES))
    args = ap.parse_args(argv)

    if args.cmd == "calibrate":
        print(json.dumps(na.calibrate(SETTINGS[args.setting]), indent=1))
        return 0
    if args.cmd == "report":
        print("wrote", write_report(Path(args.out_dir), Path(args.lattice_dir), Path(args.baselines)))
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "baselines":
        setting = SETTINGS[args.setting]
        b = na.run_baselines(setting, threads=args.threads, verbose=args.verbose, polish_time_s=args.polish_time)
        path = out_dir / f"baselines_{args.setting}.json"
        with open(path, "w") as f:
            json.dump(b, f, indent=1, default=_json_default)
        for name, c in b["codes"].items():
            bb = c["encoders"]["best_by_profile"]
            print(f"{args.setting} {name}: d={c['distance']} w*={c['w_star']} mass=" + "/".join(f"{bb[ev]['undetected']:.4g}" if ev in bb else "none" for ev in PROFILES))
        print("wrote", path)
        return 0
    kw = dict(device=args.device, threads=args.threads, batch=args.batch, steps=args.steps, wall_min=args.wall, chunk=args.chunk, verbose=args.verbose)
    if args.cmd == "run":
        run_one(args.setting, args.profile, args.seed, out_dir, **kw)
        return 0
    if args.cmd == "matrix":
        t0 = time.perf_counter()
        calibs = {}
        for seed in args.seeds:
            for sname in args.settings:
                for profile in args.profiles:
                    path = run_path(out_dir, sname, profile, seed)
                    if args.skip_existing and path.exists():
                        print(f"skip {path.name} (exists)", flush=True)
                        continue
                    if sname not in calibs:
                        calibs[sname] = na.calibrate(SETTINGS[sname])
                    rec = run_one(sname, profile, seed, out_dir, calib=calibs[sname], **kw)
                    print(
                        f"[{(time.perf_counter() - t0) / 60:.1f} min] {path.stem}: success={rec['success']} first={rec['first_success_env_steps']} "
                        f"env_steps={rec['env_steps']:,} wall={rec['wall_time_s']}s ({rec['env_steps_per_s']} steps/s)",
                        flush=True,
                    )
                    try:
                        print("report:", write_report(out_dir), flush=True)
                    except Exception as e:  # a report failure must not stop the matrix
                        print(f"report failed: {type(e).__name__}: {e}", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
