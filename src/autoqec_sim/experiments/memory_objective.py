# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Benchmark, rank-correlation, bias-sensitivity, lattice, and PPO smoke
runs of the memory objective (:mod:`autoqec_sim.memory_objective`).

Subcommands (``python -m autoqec_sim.experiments.memory_objective``):

``benchmark``
    the five reference codes ([[5,1,3]], Steane, rotated surface [[9,1,3]],
    the board's [[16,4,4]], and the non-CSS [[9,1,3]] of the noise-aware
    ion run) under the three profiles, every tier, ``R = d`` rounds, with
    the wall time of each tier per code.
``random``
    at least 30 random distance-3 codes at ``n = 9`` (random 150-gate
    Clifford circuits, no weight-1 stabilizer), proxy against bposd under
    the three profiles; Spearman rank correlation per profile.
``bias``
    the rotated surface code against its XZZX deformation at ``n = 9``
    under Z-biased idle channels: the two ion presets and a depolarizing
    profile whose slot idle bias ``eta`` is swept, sequential and parallel
    timing.
``lattice``
    the square-lattice layout with routing against all-to-all for the
    rotated surface code and the non-CSS 3x3 code under board and heron.
``ppo``
    PPO on the 3x3 lattice under the ion profile with the memory reward,
    smoke budget; reports whether distance 3 was reached and the memory
    cost per terminal code.
``report``
    ``experiments/memory_objective/REPORT.md`` and the figures from the
    JSON files under ``results/memory_objective/``.

Profiles: ``ion`` is ``presets.chain()`` (arXiv:2606.06455 rates, sequential
gates), ``forte`` is ``presets.forte("sequential")``, ``board`` is the
qldpc-challenge recipe, ``heron`` the Heron r2 rates with the layout's
connectivity. Rates at scale 1 unless stated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


def _threads_from_argv(argv) -> int:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return 2


_THREADS = _threads_from_argv(sys.argv)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import numpy as np

from autoqec_sim.noise import Idle, presets

from .. import graphs, known_codes
from ..code import StabilizerCode
from ..env import CodeDiscoveryEnv
from ..memory_objective import MemoryObjective, MemoryReward, build_memory_circuit, code_key
from ..reward import batched_distance, low_weight_stabilizer_penalty

DEFAULT_OUT = Path("results/memory_objective")
REPORT_DIR = Path("experiments/memory_objective")

# ----------------------------------------------------------------------
# codes
# ----------------------------------------------------------------------
# qldpc-challenge/codes/16-4-4.json ("[[16,4,4]] SAT t3 w6 (a)", MathysRennela,
# 2026-09-06): X and Z check supports.
BOARD_16_4_4_X = [[1, 4, 5, 6, 11, 12], [1, 4, 6, 8, 10, 13], [3, 4, 6, 7, 11, 14], [2, 5, 6, 9, 13, 14], [3, 6, 9, 10, 11, 15], [0, 1, 2, 3, 6, 10]]
BOARD_16_4_4_Z = [[0, 5, 8, 9, 10, 12], [2, 4, 7, 10, 11, 14], [1, 3, 4, 7, 9, 14], [2, 5, 6, 8, 14, 15], [2, 6, 9, 11, 13, 15], [0, 1, 4, 5, 6, 10]]

# results/noise_aware/runs/3x3_ion_seed0.json, best code (non-CSS [[9,1,3]],
# w* = 4, 15-gate encoder), stabilizers as reported (signs dropped).
NONCSS_9_ION = ["XXXIIIIII", "IZZIZZIII", "ZZIXIIXII", "IXXYYXIII", "IIIIZZIII", "IIIZZIZZI", "IIIIIIXXX", "IIIIIIIZZ"]


def _css_strings(n, supports, letter):
    out = []
    for s in supports:
        row = ["I"] * n
        for q in s:
            row[q] = letter
        out.append("".join(row))
    return out


def hadamard_conjugate(stabilizers, qubits):
    swap = {"X": "Z", "Z": "X", "Y": "Y", "I": "I"}
    qs = set(qubits)
    return ["".join(swap[ch] if q in qs else ch for q, ch in enumerate(s)) for s in stabilizers]


def xzzx_9() -> list[str]:
    """Rotated surface code with H on the checkerboard qubits (r + c odd)."""
    checker = [r * 3 + c for r in range(3) for c in range(3) if (r + c) % 2 == 1]
    return hadamard_conjugate(known_codes.ROTATED_SURFACE_9, checker)


BENCHMARK_CODES = {
    "five_qubit": ("[[5,1,3]]", lambda: known_codes.get("five_qubit"), 3),
    "steane": ("Steane [[7,1,3]]", lambda: known_codes.get("steane"), 3),
    "rotated_surface_9": ("rotated surface [[9,1,3]]", lambda: known_codes.get("rotated_surface_9"), 3),
    "board_16_4_4": ("board [[16,4,4]] (SAT t3 w6 a)", lambda: StabilizerCode.from_stabilizers(_css_strings(16, BOARD_16_4_4_X, "X") + _css_strings(16, BOARD_16_4_4_Z, "Z")), 4),
    "noncss_9_ion": ("non-CSS [[9,1,3]], noise-aware 3x3 ion seed 0", lambda: StabilizerCode.from_stabilizers(NONCSS_9_ION), 3),
}

PROFILE_FACTORIES = {
    "ion": lambda: presets.chain(),
    "board": lambda: presets.board(),
    "heron": lambda: presets.heron(graph=None),
    "forte": lambda: presets.forte("sequential"),
}
BENCHMARK_PROFILES = ("ion", "board", "heron")
BENCHMARK_SHORT = {"five_qubit": "[[5,1,3]]", "steane": "Steane", "rotated_surface_9": "rotated surface", "board_16_4_4": "[[16,4,4]]", "noncss_9_ion": "non-CSS [[9,1,3]]"}
PROFILE_TITLES = {"ion": "chain, sequential", "board": "board depolarizing", "heron": "Heron rates", "forte": "Forte, sequential"}
# Scale per profile: the closed-form proxy is a leading-order expansion and
# the sampled tiers saturate once the expected number of faults per shot
# passes about one. At scale 1 the chain preset gives 0.6 to 3.4 expected
# faults per shot on the benchmark circuits and the Heron preset (reset and
# readout at 1.5e-2, 1.5e-2 dephasing per idle data qubit per readout
# layer) 2.8 to 8.8; the board recipe at p = 1e-3 gives 0.5 to 2.1. The
# chain preset at 0.25 is the labelled point of the large-code ion study.
PROFILE_SCALES = {"ion": 0.25, "board": 1.0, "heron": 0.1, "forte": 0.25}
DEPOLARIZING_SCALE = 0.25  # the biased-idle sweep: every rate 2.5e-4


def profile(name: str):
    return PROFILE_FACTORIES[name]()


def biased_idle_profile(eta: float, timing: str, p: float = 1e-3):
    """Depolarizing profile at rate ``p`` on gates, resets, and measurements
    with a slot idle channel of rate ``p`` and bias ``eta`` (``0.5`` is
    depolarizing, ``inf`` pure Z)."""
    prof = presets.depolarizing(p, timing=timing)
    return prof.with_params(idle=Idle(kind="slot", gate=p, meas=p, scope="all", bias=eta))


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.hex()
    raise TypeError(type(o).__name__)


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))
    return path


def spearman(x, y) -> float:
    """Spearman rank correlation with average ranks for ties."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] < 2:
        return float("nan")

    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(v.shape[0])
        r[order] = np.arange(1, v.shape[0] + 1)
        # average ranks for ties
        _, inv = np.unique(v, return_inverse=True)
        sums = np.bincount(inv, weights=r)
        counts = np.bincount(inv)
        return (sums / counts)[inv]

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _platform():
    return {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(), "threads": _THREADS}


def evaluate_tiers(code, prof, rounds, tiers, seed, shots, min_failures, max_shots, connectivity="all_to_all", flags="none", scale=1.0, proxy3_max_mechanisms=300):
    """One code under one profile with every requested tier; returns a
    dict tier -> MemoryResult.to_dict() plus per-tier wall time."""
    out = {}
    for tier in tiers:
        obj = MemoryObjective(
            prof, scale=scale, rounds=rounds, tier=tier, connectivity=connectivity, flags=flags, seed=seed,
            with_dcirc=(tier == "proxy"), shots=shots, min_failures=min_failures, max_shots=max_shots,
        )
        t0 = time.perf_counter()
        res = obj.evaluate(code, use_cache=False)
        rec = res.to_dict()
        rec["wall_s"] = round(time.perf_counter() - t0, 4)
        if tier == "proxy":
            m = max(rec["per_basis"][b]["mechanisms"] for b in rec["per_basis"])
            if m <= proxy3_max_mechanisms:
                obj3 = MemoryObjective(prof, scale=scale, rounds=rounds, tier="proxy", connectivity=connectivity, flags=flags, seed=seed, proxy_max_weight=3)
                t0 = time.perf_counter()
                r3 = obj3.evaluate(code, use_cache=False)
                rec["proxy3_p_round"] = r3.p_round
                rec["proxy3_wall_s"] = round(time.perf_counter() - t0, 4)
        out[tier] = rec
    return out


# ----------------------------------------------------------------------
# benchmark
# ----------------------------------------------------------------------
def run_benchmark(out_dir: Path, profiles, tiers, shots: int, min_failures: int, max_shots: int, seed: int, flags: str, verbose: bool):
    out = {"kind": "benchmark", "profiles": list(profiles), "scales": {p: PROFILE_SCALES[p] for p in profiles}, "tiers": list(tiers), "shots": shots, "min_failures": min_failures,
           "max_shots": max_shots, "seed": seed, "flags": flags, "platform": _platform(), "codes": {}}
    for key, (label, factory, d) in BENCHMARK_CODES.items():
        code = factory()
        rec = {"label": label, "n": code.n, "k": code.k, "d": d, "rounds": d, "profiles": {}}
        for pname in profiles:
            prof = profile(pname)
            t0 = time.perf_counter()
            rec["profiles"][pname] = evaluate_tiers(code, prof, d, tiers, seed, shots, min_failures, max_shots, flags=flags, scale=PROFILE_SCALES[pname])
            rec["profiles"][pname]["scale"] = PROFILE_SCALES[pname]
            if verbose:
                per = rec["profiles"][pname]
                line = " ".join(f"{t}={per[t]['p_round']:.3g}({per[t]['wall_s']:.2f}s)" for t in tiers)
                print(f"{key:18s} {pname:6s} {line}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        out["codes"][key] = rec
    path = _dump(out, out_dir / f"benchmark_flags_{flags}.json" if flags != "none" else out_dir / "benchmark.json")
    print("wrote", path)
    return out


# ----------------------------------------------------------------------
# random distance-3 codes at n = 9
# ----------------------------------------------------------------------
def random_distance3_codes(n: int, count: int, seed: int, length: int = 150, batch: int = 512):
    """Distinct ``[[n, 1, 3]]`` codes with no weight-1 stabilizer from
    random ``length``-gate Clifford circuits (H, S, CNOT, all-to-all)."""
    env = CodeDiscoveryEnv(n=n, k=1, graph=graphs.all_to_all(n), batch_size=batch, gates=("H", "S", "CNOT"), max_steps=length, seed=seed)
    codes, keys = [], set()
    tried = 0
    while len(codes) < count:
        env.reset()
        for _ in range(length):
            env.step(env.sample_actions())
        tried += batch
        d = batched_distance(env.tab, 1, 2)
        pad = low_weight_stabilizer_penalty(env.tab, 1, 1)
        for b in np.flatnonzero((d >= 3) & (pad == 0.0)):
            code = StabilizerCode.from_tableau(env.tab, 1, int(b))
            k = code_key(code)
            if k in keys:
                continue
            keys.add(k)
            codes.append(code)
            if len(codes) >= count:
                break
    return codes, tried


def run_random(out_dir: Path, count: int, profiles, shots: int, min_failures: int, max_shots: int, seed: int, verbose: bool):
    codes, tried = random_distance3_codes(9, count, seed)
    out = {"kind": "random", "n": 9, "count": len(codes), "circuits_tried": tried, "seed": seed, "profiles": list(profiles), "scales": {p: PROFILE_SCALES[p] for p in profiles},
           "shots": shots, "min_failures": min_failures, "max_shots": max_shots, "platform": _platform(), "codes": []}
    for i, code in enumerate(codes):
        rec = {"index": i, "stabilizers": code.stabilizer_strings(), "is_css": code.is_css(),
               "weights": code.stabilizer_weights().tolist(), "profiles": {}}
        for pname in profiles:
            rec["profiles"][pname] = evaluate_tiers(code, profile(pname), 3, ("proxy", "bposd"), seed, shots, min_failures, max_shots, scale=PROFILE_SCALES[pname])
        if verbose:
            line = " ".join(f"{p}: {rec['profiles'][p]['proxy']['p_round']:.3g}/{rec['profiles'][p]['bposd']['p_round']:.3g}" for p in profiles)
            print(f"code {i:2d} css={rec['is_css']!s:5s} {line}", flush=True)
        out["codes"].append(rec)
    out["spearman"] = {}
    for pname in profiles:
        px = [c["profiles"][pname]["proxy"]["p_round"] for c in out["codes"]]
        bp = [c["profiles"][pname]["bposd"]["p_round"] for c in out["codes"]]
        dc = [min(c["profiles"][pname]["proxy"]["per_basis"][b].get("d_circ") or 9 for b in "ZX") for c in out["codes"]]
        p3 = [c["profiles"][pname]["proxy"].get("proxy3_p_round") for c in out["codes"]]
        out["spearman"][pname] = {
            "proxy_vs_bposd": spearman(px, bp),
            "dcirc_mass_vs_bposd": spearman([-d for d in dc], bp),
            "proxy3_vs_bposd": spearman(p3, bp) if all(v is not None for v in p3) else None,
            "count": len(px),
        }
    path = _dump(out, out_dir / "random.json")
    print("wrote", path, json.dumps(out["spearman"]))
    return out


def run_random_check(out_dir: Path, shots: int, min_failures: int, max_shots: int, seed: int, verbose: bool, profiles=None, tag: str = ""):
    """Where the proxy-bposd disagreement on the random codes comes from:
    the same codes under the sampled tier (lookup decoder, near-ML) and
    under bposd with a second sampling seed. Spearman of proxy against
    lookup isolates the truncation error of the proxy; bposd seed against
    seed gives the shot-noise ceiling of the correlation."""
    rnd = _load(out_dir / "random.json")
    if rnd is None:
        raise SystemExit("run `random` first")
    profiles = list(rnd["profiles"]) if not profiles else list(profiles)
    out = {"kind": "random_check", "seed": seed, "shots": shots, "min_failures": min_failures, "max_shots": max_shots, "profiles": profiles, "codes": []}
    for c in rnd["codes"]:
        code = StabilizerCode.from_stabilizers([s.lstrip("+-") for s in c["stabilizers"]])
        rec = {"index": c["index"], "profiles": {}}
        for pname in profiles:
            prof = profile(pname)
            sc = PROFILE_SCALES[pname]
            smp = MemoryObjective(prof, scale=sc, rounds=3, tier="sampled", seed=seed, shots=shots, min_failures=min_failures, max_shots=max_shots).evaluate(code)
            bp2 = MemoryObjective(prof, scale=sc, rounds=3, tier="bposd", seed=seed + 1000, shots=shots, min_failures=min_failures, max_shots=max_shots).evaluate(code)
            rec["profiles"][pname] = {"sampled": smp.to_dict(), "bposd_seed2": bp2.to_dict(), "proxy": c["profiles"][pname]["proxy"]["p_round"], "bposd": c["profiles"][pname]["bposd"]["p_round"]}
        if verbose:
            print("code", c["index"], {p: (round(rec["profiles"][p]["proxy"], 5), round(rec["profiles"][p]["sampled"]["p_round"], 5), round(rec["profiles"][p]["bposd"], 5), round(rec["profiles"][p]["bposd_seed2"]["p_round"], 5)) for p in profiles}, flush=True)
        out["codes"].append(rec)
    out["spearman"] = {}
    for pname in profiles:
        px = [r["profiles"][pname]["proxy"] for r in out["codes"]]
        sm = [r["profiles"][pname]["sampled"]["p_round"] for r in out["codes"]]
        b1 = [r["profiles"][pname]["bposd"] for r in out["codes"]]
        b2 = [r["profiles"][pname]["bposd_seed2"]["p_round"] for r in out["codes"]]
        bm = [(a + b) / 2 for a, b in zip(b1, b2)]
        out["spearman"][pname] = {
            "proxy_vs_sampled": spearman(px, sm), "sampled_vs_bposd": spearman(sm, b1), "bposd_seed_vs_seed": spearman(b1, b2),
            "proxy_vs_bposd_mean": spearman(px, bm), "sampled_vs_bposd_mean": spearman(sm, bm), "count": len(px),
        }
    path = _dump(out, out_dir / f"random_check{tag}.json")
    print("wrote", path, json.dumps(out["spearman"]))
    return out


# ----------------------------------------------------------------------
# bias sensitivity
# ----------------------------------------------------------------------
BIAS_ETAS = (0.5, 2.0, 5.0, 10.0, 30.0, 100.0, float("inf"))


def run_bias(out_dir: Path, tiers, shots: int, min_failures: int, max_shots: int, seed: int, rounds: int, verbose: bool):
    plain = known_codes.get("rotated_surface_9")
    xzzx = StabilizerCode.from_stabilizers(xzzx_9())
    cases = [("forte", presets.forte("sequential"), None, "sequential", PROFILE_SCALES["forte"]), ("ion", presets.chain(), None, "sequential", PROFILE_SCALES["ion"])]
    for timing in ("sequential", "parallel"):
        for eta in BIAS_ETAS:
            cases.append((f"depol_idle_eta_{eta:g}_{timing}", biased_idle_profile(eta, timing), eta, timing, DEPOLARIZING_SCALE))
    out = {"kind": "bias", "rounds": rounds, "tiers": list(tiers), "shots": shots, "min_failures": min_failures, "max_shots": max_shots,
           "seed": seed, "platform": _platform(), "cases": []}
    for name, prof, eta, timing, scale in cases:
        rec = {"name": name, "profile": prof.name, "timing": timing, "eta": eta, "scale": scale, "codes": {}}
        for cname, code in (("plain", plain), ("xzzx", xzzx)):
            rec["codes"][cname] = evaluate_tiers(code, prof, rounds, tiers, seed, shots, min_failures, max_shots, scale=scale)
        rec["ratio"] = {t: rec["codes"]["plain"][t]["p_round"] / rec["codes"]["xzzx"][t]["p_round"] if rec["codes"]["xzzx"][t]["p_round"] > 0 else None for t in tiers}
        if verbose:
            print(name, {t: (round(rec["codes"]["plain"][t]["p_round"], 6), round(rec["codes"]["xzzx"][t]["p_round"], 6)) for t in tiers}, flush=True)
        out["cases"].append(rec)
    path = _dump(out, out_dir / "bias.json")
    print("wrote", path)
    return out


# ----------------------------------------------------------------------
# lattice layout
# ----------------------------------------------------------------------
def run_lattice(out_dir: Path, seed: int, verbose: bool):
    codes = {
        "rotated_surface_9": known_codes.get("rotated_surface_9"),
        "xzzx_9": StabilizerCode.from_stabilizers(xzzx_9()),
        "noncss_9_ion": StabilizerCode.from_stabilizers(NONCSS_9_ION),
        "shor_9": known_codes.get("shor"),
    }
    out = {"kind": "lattice", "seed": seed, "platform": _platform(), "codes": {}}
    for cname, code in codes.items():
        rec = {}
        for conn in ("all_to_all", "square:3x3"):
            mc = build_memory_circuit(code, 3, "Z", connectivity=conn)
            rec[conn] = {"two_qubit_gates": mc.two_qubit_gates, "routed_gates": mc.routed_gates, "layers": mc.layers,
                         "batches": mc.batches, "num_qubits": mc.num_qubits, "ancilla": mc.ancilla, "profiles": {}}
            for pname in ("board", "heron", "ion"):
                obj = MemoryObjective(profile(pname), scale=PROFILE_SCALES[pname], rounds=3, tier="proxy", connectivity=conn, seed=seed, with_dcirc=True)
                rec[conn]["profiles"][pname] = obj.evaluate(code).to_dict()
        if verbose:
            print(cname, {c: (rec[c]["two_qubit_gates"], rec[c]["routed_gates"], {p: round(rec[c]["profiles"][p]["p_round"], 5) for p in rec[c]["profiles"]}) for c in rec}, flush=True)
        out["codes"][cname] = rec
    path = _dump(out, out_dir / "lattice.json")
    print("wrote", path)
    return out


# ----------------------------------------------------------------------
# PPO smoke regression
# ----------------------------------------------------------------------
def run_ppo(out_dir: Path, seed: int, max_env_steps: int, max_time_s: float, rounds: int, batch: int, threads: int, verbose: bool):
    from ..agents.ppo import PPOAgent, PPOConfig
    from ..regress import replay

    prof = profile("ion")
    obj = MemoryObjective(prof, scale=PROFILE_SCALES["ion"], rounds=rounds, tier="proxy", connectivity="all_to_all", seed=seed)
    reward = MemoryReward(obj, target_distance=3, weight=0.1, p_ref=1e-2)
    env = CodeDiscoveryEnv(
        n=9, k=1, graph=graphs.square(3, 3), batch_size=batch, gates=("H", "CNOT"), max_steps=30,
        reward_fn=reward, reward_mode="dense", auto_reset=True, seed=seed,
    )
    agent = PPOAgent(env, PPOConfig(max_threads=threads), seed=seed)
    t0 = time.perf_counter()
    res = agent.train(max_env_steps=max_env_steps, max_time_s=max_time_s, stop_on_success=False, verbose=verbose)
    wall = time.perf_counter() - t0
    found = []
    for fc in agent.successes[:2000]:
        code = replay([tuple(g) for g in fc.gates], 9)
        key = code_key(code)
        if key in {f["key"] for f in found}:
            continue
        r = obj.evaluate(code, 3)
        found.append({"key": key.hex(), "env_steps": fc.env_steps, "gates": len(fc.gates), "d": code.distance_exact(), "css": code.is_css(),
                      "p_round": r.p_round, "state_reward": fc.state_reward, "stabilizers": code.stabilizer_strings()})
    best = min(found, key=lambda f: f["p_round"]) if found else None
    ref = obj.evaluate(known_codes.get("rotated_surface_9"), 3).p_round
    out = {
        "kind": "ppo", "seed": seed, "profile": prof.name, "scale": PROFILE_SCALES["ion"], "rounds": rounds, "batch": batch, "max_env_steps": max_env_steps,
        "max_time_s": max_time_s, "env_steps": res.env_steps, "episodes": res.episodes, "updates": res.updates, "wall_time_s": round(wall, 1),
        "reached_distance_3": res.first_success is not None,
        "first_success_env_steps": res.first_success.env_steps if res.first_success else None,
        "successes": res.successes, "distinct_codes_verified": len(found), "memory_evaluations": obj.evaluations,
        "memory_cache_hits": obj.cache_hits, "terminal_reward_calls": reward.terminal_calls,
        "memory_seconds_total": round(sum(r.seconds for r in obj.cache.values()), 2),
        "memory_seconds_per_code": round(float(np.mean([r.seconds for r in obj.cache.values()])), 4) if obj.cache else None,
        "final_success_rate": float(np.mean([h["success_rate"] for h in res.history[-10:]])) if res.history else None,
        "best_code": best, "reference_rotated_surface_p_round": ref,
        "history": res.history, "platform": _platform(), "num_parameters": res.num_parameters,
    }
    path = _dump(out, out_dir / f"ppo_3x3_ion_seed{seed}.json")
    print("wrote", path, "reached d=3:", out["reached_distance_3"], "first at", out["first_success_env_steps"], "best p_round", best["p_round"] if best else None, "surface", ref)
    return out


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(x, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) < 1e-3 or abs(x) >= 1e4:
            return f"{x:.{digits - 1}e}"
        return f"{x:.{digits}g}"
    return str(x)


def make_figures(fig_dir: Path, bench, rnd, bias) -> list[str]:
    from . import figstyle

    plt = figstyle.apply()
    figs = []
    if rnd is not None or bench is not None:
        fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE))
        for ax, pname in zip(axes, BENCHMARK_PROFILES):
            xs, ys = [], []
            allv = []
            if rnd is not None:
                for c in rnd["codes"]:
                    xs.append(c["profiles"][pname]["proxy"]["p_round"])
                    ys.append(c["profiles"][pname]["bposd"]["p_round"])
                ax.plot(xs, ys, "o", color=figstyle.NEUTRAL, label=f"random [[9,1,3]] ({len(xs)})", markersize=figstyle.MARKER)
                allv += xs + ys
            if bench is not None:
                for i, (key, rec) in enumerate(bench["codes"].items()):
                    per = rec["profiles"].get(pname)
                    if per is None or "bposd" not in per:
                        continue
                    x, y = per["proxy"]["p_round"], per["bposd"]["p_round"]
                    allv += [x, y]
                    ax.plot([x], [y], "s", color=figstyle.SERIES_COLORS[i % len(figstyle.SERIES_COLORS)], label=BENCHMARK_SHORT[key], markersize=figstyle.MARKER + 1)
            lo = min([v for v in allv if v > 0] or [1e-4]) * 0.6
            hi = max([*allv, 1e-2]) * 1.6
            ax.plot([lo, hi], [lo, hi], "-", color="#bbbbbb", linewidth=figstyle.THIN)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            from matplotlib.ticker import NullFormatter

            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("proxy p_round")
            if ax is axes[0]:
                ax.set_ylabel("bposd p_round")
            rho = rnd["spearman"][pname]["proxy_vs_bposd"] if rnd is not None else float("nan")
            ax.set_title(f"{PROFILE_TITLES[pname]} (scale {PROFILE_SCALES[pname]:g})", pad=10)
            ax.text(0.97, 0.04, f"Spearman {rho:.2f}\n(random set)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        axes[0].legend(loc="upper left", fontsize=6)
        for ax, letter in zip(axes, "abc"):
            figstyle.panel_label(ax, letter)
        figs.append(str(figstyle.save(fig, fig_dir / "tiers")[0]))
        plt.close(fig)
    if bias is not None:
        fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE))
        for ax, timing in zip(axes, ("sequential", "parallel")):
            cases = [c for c in bias["cases"] if c["eta"] is not None and c["timing"] == timing]
            etas = [c["eta"] if c["eta"] != float("inf") else 1000.0 for c in cases]
            for cname, color, marker in (("plain", figstyle.SERIES_COLORS[0], "o"), ("xzzx", figstyle.SERIES_COLORS[1], "s")):
                for tier, ls in (("proxy", "-"), ("bposd", "--")):
                    if tier not in bias["tiers"]:
                        continue
                    ys = [c["codes"][cname][tier]["p_round"] for c in cases]
                    ax.plot(etas, ys, marker=marker, linestyle=ls, color=color, label=f"{cname}, {tier}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("idle bias eta (1000 = pure Z)")
            if ax is axes[0]:
                ax.set_ylabel("p_round (both bases)")
            ax.set_title(f"every rate 2.5e-4, idle bias eta, {timing} timing", pad=10)
        axes[1].legend(fontsize=6)
        for ax, letter in zip(axes, "ab"):
            figstyle.panel_label(ax, letter)
        figs.append(str(figstyle.save(fig, fig_dir / "bias")[0]))
        plt.close(fig)
    return figs


def write_report(out_dir: Path, report_dir: Path) -> Path:
    bench = _load(out_dir / "benchmark.json")
    bench_flags = _load(out_dir / "benchmark_flags_one.json")
    rnd = _load(out_dir / "random.json")
    bias = _load(out_dir / "bias.json")
    lat = _load(out_dir / "lattice.json")
    ppo = sorted(out_dir.glob("ppo_3x3_ion_seed*.json"))
    ppo = [_load(p) for p in ppo]
    figs = make_figures(report_dir / "figures", bench, rnd, bias)
    lines = ["# Memory objective: tiers, benchmark, bias sensitivity, PPO smoke run", ""]
    lines.append(f"Every number below is read from the JSON files under `{out_dir}/` (regenerate with `python -m autoqec_sim.experiments.memory_objective <subcommand>`; this file with `report`). Module `src/autoqec_sim/memory_objective.py`, design note `docs/memory_objective.md`.")
    lines.append("")
    verdict = report_dir / "verdict.md"
    if verdict.exists():
        lines.append(verdict.read_text().strip())
        lines.append("")
    chk = _load(out_dir / "random_check.json")
    if bench is not None:
        lines += ["## Benchmark: five codes, three profiles, three tiers", ""]
        scales = ", ".join(f"{p} {v:g}" for p, v in bench.get("scales", {}).items())
        lines.append(f"Rounds `R = d`, scales {scales}, all-to-all layout, bare ancillas, hook-aware gate order. Sampled tiers: {bench['shots']} shots per basis, grown to at least {bench['min_failures']} failures or {bench['max_shots']} shots. `p_round` is the per-round failure probability summed over the Z and X memory bases; `E` is the expected number of faults per shot (Z basis). Times are wall seconds per code for both bases on {bench['platform']['machine']}, one thread, one process.")
        lines.append("")
        lines.append("| code | profile | E | mechanisms (Z/X) | detectors | 2q gates per round | d_circ (Z/X) | proxy p_round | proxy s | proxy w<=3 p_round | sampled p_round | sampled decoder | sampled s | bposd p_round | bposd shots | bposd s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key, rec in bench["codes"].items():
            for pname, per in rec["profiles"].items():
                pz, px = per["proxy"]["per_basis"]["Z"], per["proxy"]["per_basis"]["X"]
                smp = per.get("sampled", {})
                bp = per.get("bposd", {})
                sd = smp.get("per_basis", {}).get("Z", {}).get("decoder", "n/a") if smp else "n/a"
                shots = bp["per_basis"]["Z"]["sampling"]["shots"] if bp else None
                lines.append(
                    f"| {rec['label']} | {pname} | {_fmt(pz['expected_faults'], 2)} | {pz['mechanisms']}/{px['mechanisms']} | {pz['detectors']} | {pz['two_qubit_gates'] // rec['rounds']} | {pz.get('d_circ')}/{px.get('d_circ')} | {_fmt(per['proxy']['p_round'])} | {_fmt(per['proxy']['wall_s'])} | {_fmt(per['proxy'].get('proxy3_p_round'))} | {_fmt(smp.get('p_round')) if smp else 'n/a'} | {sd} | {_fmt(smp.get('wall_s')) if smp else 'n/a'} | {_fmt(bp.get('p_round')) if bp else 'n/a'} | {shots if shots else 'n/a'} | {_fmt(bp.get('wall_s')) if bp else 'n/a'} |"
                )
        lines.append("")
        # rank correlation across the benchmark set
        lines.append("Spearman rank correlation of the tiers across the benchmark codes, per profile:")
        lines.append("")
        lines.append("| profile | codes | proxy vs bposd | sampled vs bposd | proxy vs sampled |")
        lines.append("|---|---|---|---|---|")
        for pname in bench["profiles"]:
            px = [rec["profiles"][pname]["proxy"]["p_round"] for rec in bench["codes"].values() if pname in rec["profiles"]]
            sm = [rec["profiles"][pname]["sampled"]["p_round"] for rec in bench["codes"].values() if pname in rec["profiles"] and "sampled" in rec["profiles"][pname]]
            bp = [rec["profiles"][pname]["bposd"]["p_round"] for rec in bench["codes"].values() if pname in rec["profiles"] and "bposd" in rec["profiles"][pname]]
            lines.append(f"| {pname} | {len(px)} | {_fmt(spearman(px, bp)) if bp else 'n/a'} | {_fmt(spearman(sm, bp)) if sm and bp else 'n/a'} | {_fmt(spearman(px, sm)) if sm else 'n/a'} |")
        lines.append("")
    if bench_flags is not None and bench is not None:
        lines += ["### One-flag variant", ""]
        lines.append("Same codes and profiles with one flag qubit per generator of weight at least three (`flags=\"one\"`), proxy tier: d_circ and p_round against the bare-ancilla circuit.")
        lines.append("")
        lines.append("| code | profile | bare d_circ (Z/X) | flag d_circ (Z/X) | bare proxy p_round | flag proxy p_round | flag/bare | bare 2q per round | flag 2q per round |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for key, rec in bench_flags["codes"].items():
            for pname, per in rec["profiles"].items():
                b = bench["codes"][key]["profiles"][pname]["proxy"]
                f = per["proxy"]
                bz, bx = b["per_basis"]["Z"], b["per_basis"]["X"]
                fz, fx = f["per_basis"]["Z"], f["per_basis"]["X"]
                ratio = f["p_round"] / b["p_round"] if b["p_round"] > 0 else None
                lines.append(f"| {rec['label']} | {pname} | {bz.get('d_circ')}/{bx.get('d_circ')} | {fz.get('d_circ')}/{fx.get('d_circ')} | {_fmt(b['p_round'])} | {_fmt(f['p_round'])} | {_fmt(ratio)} | {bz['two_qubit_gates'] // rec['rounds']} | {fz['two_qubit_gates'] // rec['rounds']} |")
        lines.append("")
    if rnd is not None:
        lines += ["## Random distance-3 codes at n = 9", ""]
        lines.append(f"{rnd['count']} distinct `[[9,1,3]]` codes with no weight-1 stabilizer from random 150-gate H, S, CNOT circuits on 9 qubits ({rnd['circuits_tried']} circuits drawn, seed {rnd['seed']}); {sum(c['is_css'] for c in rnd['codes'])} of them CSS. `R = 3`, all-to-all, proxy against bposd ({rnd['shots']} shots per basis grown to {rnd['min_failures']} failures or {rnd['max_shots']} shots).")
        lines.append("")
        lines.append("| profile | codes | Spearman proxy vs bposd | Spearman proxy (w<=3) vs bposd | Spearman -d_circ vs bposd | proxy/bposd ratio, median [min, max] | proxy s per code, mean | bposd s per code, mean |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for pname, sp in rnd["spearman"].items():
            ratios = [c["profiles"][pname]["proxy"]["p_round"] / c["profiles"][pname]["bposd"]["p_round"] for c in rnd["codes"] if c["profiles"][pname]["bposd"]["p_round"] > 0]
            tp = np.mean([c["profiles"][pname]["proxy"]["wall_s"] for c in rnd["codes"]])
            tb = np.mean([c["profiles"][pname]["bposd"]["wall_s"] for c in rnd["codes"]])
            lines.append(f"| {pname} | {sp['count']} | {_fmt(sp['proxy_vs_bposd'])} | {_fmt(sp['proxy3_vs_bposd'])} | {_fmt(sp['dcirc_mass_vs_bposd'])} | {_fmt(float(np.median(ratios)))} [{_fmt(min(ratios))}, {_fmt(max(ratios))}] | {_fmt(float(tp))} | {_fmt(float(tb))} |")
        lines.append("")
        dcs = {}
        for c in rnd["codes"]:
            for pname in rnd["profiles"]:
                d = min(c["profiles"][pname]["proxy"]["per_basis"][b].get("d_circ") or 9 for b in "ZX")
                dcs.setdefault(pname, []).append(d)
        if chk is not None:
            lines.append(f"Where the disagreement comes from (`random-check`, the same codes under the sampled tier with the lookup decoder and under bposd with a second sampling seed, {chk['shots']} shots grown to {chk['min_failures']} failures or {chk['max_shots']}):")
            lines.append("")
            lines.append("| profile | Spearman proxy vs sampled (lookup) | sampled vs bposd | bposd seed vs seed | proxy vs bposd (mean of two seeds) | sampled vs bposd (mean of two seeds) |")
            lines.append("|---|---|---|---|---|---|")
            for pname, sp in chk["spearman"].items():
                lines.append(f"| {pname} | {_fmt(sp['proxy_vs_sampled'])} | {_fmt(sp['sampled_vs_bposd'])} | {_fmt(sp['bposd_seed_vs_seed'])} | {_fmt(sp['proxy_vs_bposd_mean'])} | {_fmt(sp['sampled_vs_bposd_mean'])} |")
            lines.append("")
        chk2 = _load(out_dir / "random_check_ion400.json")
        if chk2 is not None:
            sp = chk2["spearman"]["ion"]
            lines.append(f"Same check on the ion profile at higher statistics ({chk2['shots']} shots grown to {chk2['min_failures']} failures or {chk2['max_shots']}): proxy vs sampled {_fmt(sp['proxy_vs_sampled'])}, sampled vs bposd {_fmt(sp['sampled_vs_bposd'])}, bposd seed vs seed {_fmt(sp['bposd_seed_vs_seed'])}, proxy vs bposd (mean of two seeds) {_fmt(sp['proxy_vs_bposd_mean'])}, over {sp['count']} codes.")
            lines.append("")
        lines.append("Circuit-level distance (minimum over bases) of the random codes with bare ancillas: " + "; ".join(f"{p}: " + ", ".join(f"d_circ = {d}: {dcs[p].count(d)}" for d in sorted(set(dcs[p]))) for p in dcs) + ".")
        lines.append("")
        if figs:
            lines.append(f"![]({os.path.relpath(figs[0], report_dir)})")
            lines.append("")
            lines.append("Proxy against bposd per-round failure probability, log axes, one panel per profile: random `[[9,1,3]]` codes (grey) and the benchmark codes (colored squares); the line is equality; the corner text is the Spearman correlation over the random set.")
            lines.append("")
    if bias is not None:
        lines += ["## Bias sensitivity: XZZX against the plain rotated surface code at n = 9", ""]
        lines.append(f"`R = {bias['rounds']}`, all-to-all, bare ancillas. `ratio` is `p_round(plain) / p_round(xzzx)`; above 1 the objective prefers the XZZX deformation. The depolarizing profile has rate 1e-3 on every gate, reset, and measurement and a slot idle channel of rate 1e-3 per layer with bias `eta = pZ / (pX + pY)` (0.5 is depolarizing; inf is pure Z).")
        lines.append("")
        tiers = bias["tiers"]
        head = "| profile | timing | eta | " + " | ".join(f"plain {t} | xzzx {t} | ratio {t}" for t in tiers) + " |"
        lines.append(head)
        lines.append("|" + "---|" * (3 + 3 * len(tiers)))
        for c in bias["cases"]:
            eta = "native" if c["eta"] is None else ("inf" if c["eta"] == float("inf") else f"{c['eta']:g}")
            cells = []
            for t in tiers:
                cells += [_fmt(c["codes"]["plain"][t]["p_round"]), _fmt(c["codes"]["xzzx"][t]["p_round"]), _fmt(c["ratio"][t])]
            lines.append(f"| {c['profile']} | {c['timing']} | {eta} | " + " | ".join(cells) + " |")
        lines.append("")
        bsc = _load(out_dir / "bias_sampled_check.json")
        if bsc is not None:
            lines.append("Sampled tier (lookup decoder, near maximum likelihood) on the two cases where the proxy and bposd disagree or are the primary target, `R = 3`, scale 0.25:")
            lines.append("")
            lines.append("| case | plain p_round | xzzx p_round | ratio | shots Z / X (plain) | failures Z / X (plain) |")
            lines.append("|---|---|---|---|---|---|")
            for case in ("forte", "depol_eta_inf_seq"):
                pl, xz = bsc.get(f"{case}/plain"), bsc.get(f"{case}/xzzx")
                if pl is None or xz is None:
                    continue
                sz, sx = pl["per_basis"]["Z"]["sampling"], pl["per_basis"]["X"]["sampling"]
                lines.append(f"| {case} | {_fmt(pl['p_round'])} | {_fmt(xz['p_round'])} | {_fmt(pl['p_round'] / xz['p_round'])} | {sz['shots']} / {sx['shots']} | {sz['failures']} / {sx['failures']} |")
            lines.append("")
        if len(figs) > 1:
            lines.append(f"![]({os.path.relpath(figs[1], report_dir)})")
            lines.append("")
            lines.append("Per-round failure probability of the plain (circles) and XZZX (squares) codes against the idle bias, proxy (solid) and bposd (dashed), sequential (a) and parallel (b) timing.")
            lines.append("")
    if lat is not None:
        lines += ["## Lattice layout with routing", ""]
        lines.append("`square:3x3`: data qubits on the lattice, 16 ancilla sites at the plaquette centers including the boundary ring, site-to-corner edges only; generators whose site is not adjacent to a support qubit are routed with nearest-neighbour CNOT ladders and run in their own batch. Proxy tier, `R = 3`, Z basis circuit statistics.")
        lines.append("")
        lines.append("| code | layout | 2q gates (3 rounds) | routed extra 2q | layers | batches | board p_round | heron p_round | ion p_round | d_circ board (Z/X) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for cname, rec in lat["codes"].items():
            for conn, r in rec.items():
                pb = r["profiles"]["board"]
                lines.append(f"| {cname} | {conn} | {r['two_qubit_gates']} | {r['routed_gates']} | {r['layers']} | {len(r['batches'])} | {_fmt(pb['p_round'])} | {_fmt(r['profiles']['heron']['p_round'])} | {_fmt(r['profiles']['ion']['p_round'])} | {pb['per_basis']['Z'].get('d_circ')}/{pb['per_basis']['X'].get('d_circ')} |")
        lines.append("")
    if ppo:
        lines += ["## PPO smoke run: 3x3 lattice, ion profile, memory reward", ""]
        lines.append("`CodeDiscoveryEnv(n=9, k=1, square:3x3, H and CNOT, 30 gates, batch 256)` with `MemoryReward(MemoryObjective(chain, rounds=R, tier=proxy), d*=3, weight 0.1, p_ref 1e-2)`, `PPOConfig` defaults, dense reward, no early stop.")
        lines.append("")
        lines.append("| seed | rounds | env steps | episodes | wall s | reached d = 3 | first d = 3 (env steps) | successes | distinct codes | memory evaluations | memory s per code | memory s total | final success rate | best p_round | rotated surface p_round |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in ppo:
            best = r["best_code"]["p_round"] if r["best_code"] else None
            lines.append(f"| {r['seed']} | {r['rounds']} | {r['env_steps']} | {r['episodes']} | {r['wall_time_s']} | {r['reached_distance_3']} | {r['first_success_env_steps']} | {r['successes']} | {r['distinct_codes_verified']} | {r['memory_evaluations']} | {_fmt(r['memory_seconds_per_code'])} | {_fmt(r['memory_seconds_total'])} | {_fmt(r['final_success_rate'])} | {_fmt(best)} | {_fmt(r['reference_rotated_surface_p_round'])} |")
        lines.append("")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "REPORT.md"
    path.write_text("\n".join(lines))
    print("wrote", path, *figs)
    return path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=_THREADS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--profiles", nargs="+", default=list(BENCHMARK_PROFILES))
    b.add_argument("--tiers", nargs="+", default=["proxy", "sampled", "bposd"])
    b.add_argument("--shots", type=int, default=5000)
    b.add_argument("--min-failures", type=int, default=200)
    b.add_argument("--max-shots", type=int, default=100_000)
    b.add_argument("--flags", default="none", choices=["none", "one"])
    r = sub.add_parser("random")
    r.add_argument("--count", type=int, default=32)
    r.add_argument("--profiles", nargs="+", default=list(BENCHMARK_PROFILES))
    r.add_argument("--shots", type=int, default=5000)
    r.add_argument("--min-failures", type=int, default=100)
    r.add_argument("--max-shots", type=int, default=50_000)
    rc = sub.add_parser("random-check")
    rc.add_argument("--shots", type=int, default=5000)
    rc.add_argument("--min-failures", type=int, default=100)
    rc.add_argument("--max-shots", type=int, default=50_000)
    rc.add_argument("--profiles", nargs="+", default=None)
    rc.add_argument("--tag", default="")
    bi = sub.add_parser("bias")
    bi.add_argument("--tiers", nargs="+", default=["proxy", "bposd"])
    bi.add_argument("--shots", type=int, default=5000)
    bi.add_argument("--min-failures", type=int, default=200)
    bi.add_argument("--max-shots", type=int, default=200_000)
    bi.add_argument("--rounds", type=int, default=3)
    sub.add_parser("lattice")
    p = sub.add_parser("ppo")
    p.add_argument("--max-env-steps", type=int, default=400_000)
    p.add_argument("--max-time", type=float, default=600.0)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--batch", type=int, default=256)
    rp = sub.add_parser("report")
    rp.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = ap.parse_args(argv)
    if args.cmd == "benchmark":
        run_benchmark(args.out, args.profiles, args.tiers, args.shots, args.min_failures, args.max_shots, args.seed, args.flags, args.verbose)
    elif args.cmd == "random":
        run_random(args.out, args.count, args.profiles, args.shots, args.min_failures, args.max_shots, args.seed, args.verbose)
    elif args.cmd == "random-check":
        run_random_check(args.out, args.shots, args.min_failures, args.max_shots, args.seed, args.verbose, args.profiles, args.tag)
    elif args.cmd == "bias":
        run_bias(args.out, args.tiers, args.shots, args.min_failures, args.max_shots, args.seed, args.rounds, args.verbose)
    elif args.cmd == "lattice":
        run_lattice(args.out, args.seed, args.verbose)
    elif args.cmd == "ppo":
        run_ppo(args.out, args.seed, args.max_env_steps, args.max_time, args.rounds, args.batch, args.threads, args.verbose)
    elif args.cmd == "report":
        write_report(args.out, args.report_dir)


if __name__ == "__main__":
    main()
