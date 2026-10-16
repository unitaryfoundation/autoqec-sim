# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""PPO on the 4x4 setting with the memory objective as the noise term.

Question. ``results/gpu_scale/REPORT.md`` trains the noise-aware 4x4 cells
(``n = 16``, ``k = 1``, target ``d = 4``, ``square:4x4``, H and CNOT,
50-gate episodes, equivariant policy, 1024 parallel episodes) with the
encoder objective: the undetected fault mass of the encoding circuit. That
objective scores the encoder, not the code, and cannot separate a code from
its local-Clifford deformations (``docs/memory_objective.md``). Here the
same cells train with the memory objective of
:mod:`autoqec_sim.memory_objective` in its place: the per-round logical
failure probability of an ``R = 4`` round syndrome-extraction memory
experiment of the code, proxy tier, evaluated on every state that reaches
the target distance with no weight-1 stabilizer. Does training on the
memory objective produce codes with lower memory error than training on
the encoder objective, and does bias tailoring (a Hadamard-frame pattern,
``d_X != d_Z``) appear under the ion profile?

Reward. ``r = (d - 1 + f)/d* - 1 - q_1`` on every state (the fractional
distance term of the 4x4 setting and the padding penalty of
``noise_aware``) and, on target-distance states with ``q_1 = 0``,
``+ 0.1 clip(log10(1e-2 / p_round), -3, 3) - 0.25 max(0, w* - 4)``: the
memory term of ``MemoryReward`` (weight 0.1, ``p_ref = 1e-2``) and the LDPC
term of ``noise_aware``. Such a state ends the episode. The distance,
fractional, and padding terms run on the GPU (:class:`gpu_scale.
DistanceKernel`); the memory term runs on the host in a process pool
(:class:`MemoryTermPool`), one evaluation per distinct stabilizer group
(cached by the reduced row echelon form of the group, as in
``MemoryObjective``), synchronously: the training loop waits for the terms
of the new codes of each reward call, so the objective is exactly the one
``MemoryReward`` defines and the pool only adds parallelism across the new
codes of a call.

Profiles (cells). Every cell uses the all-to-all layout of the memory
circuit (one ancilla per generator, hook-aware order, commutation batches;
the ion chain has all-to-all connectivity and the flag variant refuses the
lattice layout), ``R = 4`` rounds, both memory bases summed:

``ion``
    ``presets.chain()`` (arXiv:2606.06455 rates, sequential timing) at
    scale 0.25, bare ancillas.
``ion_flag``
    the same with one flag qubit per generator of weight at least three
    (``flags="one"``).
``board``
    the qldpc-challenge depolarizing recipe at scale 0.25 (``p = 2.5e-4``).
``heron``
    the Heron r2 rates at scale 0.05.

Scales. The memory objective's tiers need the expected number of faults
per shot ``E`` below about one (``docs/memory_objective.md``, regime of
validity). The scales of the ``n <= 9``, ``R = 3`` benchmark (chain 0.25,
board 1, Heron 0.1) do not carry over to ``n = 16``, ``R = 4``: measured
on the pod before the run, the rotated surface ``[[16,1,4]]`` has ``E =
0.65`` under chain 0.25, ``2.02`` under board 1, and ``0.91`` under Heron
0.1, and the dense encoder-objective ion seed-0 code of ``gpu_scale``
(316 two-qubit gates per round on the all-to-all layout) ``1.11``, ``5.78``,
and ``2.78``. Board 0.25 and Heron 0.05 put the surface code at ``E =
0.50`` and ``0.45`` and the dense code at ``1.45`` and ``1.39``, in line
with the chain profile. The scales are recorded in every run JSON
(``objective.scale``) and can be overridden with ``--scale``.

Selection and analysis. The best code of a seed is the highest-reward
distinct target-distance state (the selection of ``noise_aware.run_seed``
and ``gpu_scale``); the run also records the target-distance code with the
lowest training ``p_round`` and the modal code of the last 20 updates. Each
is analysed by :func:`noise_aware.analyze_circuit` (exact ``d``, ``d_X``,
``d_Z``, ``w*``, CSS test, Hadamard-frame search, weight enumerators, and
the encoder undetected mass under the three ``noise_aware`` profiles at
their calibrated scales, the numbers of the ``gpu_scale`` tables) and
scored with the memory objective under all four cells' profiles at the
proxy tier. ``score`` then evaluates, in a host process pool, the best code
of every run, the best codes of the nine encoder-objective cells of
``results/gpu_scale``, and the rotated surface ``[[16,1,4]]`` with its
XZZX deformation under the four profiles at the proxy tier and at the
bposd tier (the board's pinned BP+OSD decoder, sampled, validation);
``report`` writes the tables of ``experiments/memory_ppo/REPORT.md`` from
the JSON files.

Run 2 (``--variant run2``, ``VARIANTS``). The LDPC term is dropped (``w*``
is reported only), the selection is the target-distance code with the
lowest training ``p_round``, a gate-count term ``-GATE_WEIGHT max(0, g /
GATE_REF - 1)`` penalises memory circuits above 1.25 times the surface
code's two-qubit gates per round, a candidate above ``SAMPLED_ABOVE``
expected faults per shot is evaluated at the sampled tier instead of the
proxy (cached per code), and every code is evaluated on the sparsest
basis of its stabilizer group (:func:`sparsest_code`): the memory circuit
measures the generators it is given, and the basis read off an encoder
tableau is denser than the sparsest one (the replayed rotated surface
code: 224 against 192 two-qubit gates per round). Run 1 evaluated the
discovered codes on their tableau bases; ``score --sparsify`` re-scores
them on the sparsest basis for the run-2 comparison.

Files. ``results/memory_ppo/run_<profile>_seed<k>.json`` and a log under
``results/memory_ppo/logs/``; ``results/memory_ppo/scores.json``; run 2
under ``results/memory_ppo/run2/``; ``experiments/memory_ppo/REPORT.md``
(two sections, verdicts from ``verdict.md`` and ``verdict_run2.md``) and
``figures/``.

    python -m autoqec_sim.experiments.memory_ppo run --profile ion --seed 0 --device cuda --threads 1 --pool 1
    python -m autoqec_sim.experiments.memory_ppo score --workers 4
    python -m autoqec_sim.experiments.memory_ppo report
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import multiprocessing
import os
import platform
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _threads_from_argv(argv) -> int:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return 1


_THREADS = _threads_from_argv(sys.argv)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(_THREADS))

import numpy as np

from autoqec_sim.noise import presets

from ..code import StabilizerCode
from ..ldpc import ldpc_penalty, sparsest_basis_rows
from ..memory_objective import MemoryObjective
from ..regress import replay
from . import gpu_scale as gs
from . import noise_aware as na
from .noise_aware import _json_default, _mean_range

DEFAULT_OUT = Path("results/memory_ppo")
REPORT_DIR = Path("experiments/memory_ppo")
ENCODER_DIR = Path("results/gpu_scale")
SETTING = na.SETTINGS["4x4"]
SEEDS = (0, 1, 2)
BATCH = 1024
STEPS = 3_000_000
WALL_MIN = 150.0
THREADS = 1
POOL = 1
LOG_EVERY = 25
ROUNDS = SETTING.d
CONNECTIVITY = "all_to_all"
TRAIN_TIER = "proxy"
MEMORY_WEIGHT = 0.1
P_REF = 1e-2
CLIP = 3.0
P_FLOOR = 1e-9
# bposd validation tier (score subcommand)
SCORE_SHOTS = 5000
SCORE_MIN_FAILURES = 100
SCORE_MAX_SHOTS = 50_000

# run-2 variant (second section of the report): no LDPC term, a gate-count term above 1.25 times the
# surface code's two-qubit gates per round, the sampled tier for candidates outside the proxy's regime,
# selection by the training p_round.
SURFACE_2Q_GATES = 192  # rotated surface [[16,1,4]] memory circuit, two-qubit gates per round
GATE_REF = 240  # 1.25 x SURFACE_2Q_GATES
GATE_WEIGHT = 0.2
SAMPLED_ABOVE = 0.8  # expected faults per shot above which the sampled tier replaces the proxy
TRAIN_SAMPLED = {"shots": 2000, "min_failures": 50, "max_shots": 20_000}
VARIANTS = {
    "run1": {
        "ldpc_scale": na.LDPC_SCALE,
        "gate_ref": None,
        "gate_weight": 0.0,
        "sampled_above": None,
        "select": "reward",
        "sparsify": False,
    },
    "run2": {
        "ldpc_scale": 0.0,
        "gate_ref": GATE_REF,
        "gate_weight": GATE_WEIGHT,
        "sampled_above": SAMPLED_ABOVE,
        "select": "p_round",
        "sparsify": True,
    },
}
RUN2_DIR = DEFAULT_OUT / "run2"

PROFILES = {
    "ion": {"preset": "chain", "scale": 0.25, "flags": "none", "label": "ion chain 0.25, sequential, bare ancillas"},
    "ion_flag": {
        "preset": "chain",
        "scale": 0.25,
        "flags": "one",
        "label": "ion chain 0.25, sequential, one flag per generator",
    },
    "board": {"preset": "board", "scale": 0.25, "flags": "none", "label": "board depolarizing 0.25 (p = 2.5e-4)"},
    "heron": {"preset": "heron", "scale": 0.05, "flags": "none", "label": "Heron r2 rates 0.05"},
}
PROFILE_ORDER = tuple(PROFILES)
SHORT = {"ion": "ion", "ion_flag": "ion+flag", "board": "board", "heron": "heron"}
TITLES = {
    "ion": "ion chain 0.25, bare",
    "ion_flag": "ion chain 0.25, one flag",
    "board": "board 0.25",
    "heron": "Heron 0.05",
}
FIGURE_SMOOTH = 10  # updates in the moving average of the success-rate curves
KNOWN = ("rotated_surface", "xzzx")
ENCODER_PROFILES = ("board", "ion", "heron")  # cells of results/gpu_scale


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_profile(name: str):
    preset = PROFILES[name]["preset"]
    if preset == "chain":
        return presets.chain()  # sequential timing by default
    if preset == "board":
        return presets.board()
    if preset == "heron":
        return presets.heron(graph=None)
    raise ValueError(preset)


def make_objective(
    name: str,
    tier: str = TRAIN_TIER,
    scale: float | None = None,
    seed: int = 0,
    shots: int = SCORE_SHOTS,
    min_failures: int = SCORE_MIN_FAILURES,
    max_shots: int | None = SCORE_MAX_SHOTS,
) -> MemoryObjective:
    spec = PROFILES[name]
    return MemoryObjective(
        make_profile(name),
        scale=spec["scale"] if scale is None else float(scale),
        rounds=ROUNDS,
        tier=tier,
        connectivity=CONNECTIVITY,
        flags=spec["flags"],
        seed=seed,
        shots=shots,
        min_failures=min_failures,
        max_shots=max_shots,
    )


def memory_term(
    p_round: float, weight: float = MEMORY_WEIGHT, p_ref: float = P_REF, clip: float = CLIP, p_floor: float = P_FLOOR
) -> float:
    """``weight * clip(log10(p_ref / p_round))``, the term of ``MemoryReward``."""
    v = math.log10(p_ref / max(float(p_round), p_floor))
    return weight * max(-clip, min(clip, v))


def compact_result(res) -> dict:
    """The parts of a ``MemoryResult`` the runs keep per code."""
    keep = (
        "p_round",
        "p_fail",
        "mechanisms",
        "detectors",
        "expected_faults",
        "two_qubit_gates",
        "routed_gates",
        "layers",
        "batches",
        "num_qubits",
        "decoder",
        "seconds",
    )
    per = {b: {k: v for k, v in d.items() if k in keep} for b, d in res.per_basis.items()}
    for b, d in res.per_basis.items():
        if "sampling" in d:
            per[b]["shots"] = d["sampling"].get("shots")
            per[b]["failures"] = d["sampling"].get("failures")
    return {
        "tier": res.tier,
        "rounds": res.rounds,
        "p_round": float(res.p_round),
        "seconds": res.seconds,
        "per_basis": per,
    }


# --------------------------------------------------------------------------
# process pool for the memory term
# --------------------------------------------------------------------------
_WORKER_OBJ: MemoryObjective | None = None
_WORKER_SAMPLED: MemoryObjective | None = None
_WORKER_SAMPLED_ABOVE: float | None = None
_WORKER_SPARSIFY = False


def _worker_init(
    name: str,
    tier: str,
    scale: float | None,
    seed: int,
    shots: int,
    min_failures: int,
    max_shots: int | None,
    threads: int,
    sampled_above: float | None = None,
    sampled_sampling: dict | None = None,
    sparsify: bool = False,
):
    global _WORKER_OBJ, _WORKER_SAMPLED, _WORKER_SAMPLED_ABOVE, _WORKER_SPARSIFY
    _WORKER_SPARSIFY = bool(sparsify)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    _WORKER_OBJ = make_objective(name, tier, scale, seed, shots, min_failures, max_shots)
    _WORKER_SAMPLED_ABOVE = sampled_above
    _WORKER_SAMPLED = (
        make_objective(name, "sampled", scale, seed, **(sampled_sampling or TRAIN_SAMPLED))
        if sampled_above is not None
        else None
    )


def sparsest_code(code: StabilizerCode) -> StabilizerCode:
    """``code`` with its stabilizer generators replaced by the sparsest basis
    of the group (``ldpc.sparsest_basis_rows``, exact up to 15 generators).
    The memory circuit measures the given generators, one ancilla each, so
    a code read off an encoder tableau would otherwise be charged for the
    tableau's dense basis: the replayed rotated surface code has 224
    two-qubit gates per round in the tableau basis and 192 in the sparsest."""
    rows = sparsest_basis_rows(code.to_tableau(), code.k, exact_max_generators=na.EXACT_GENERATORS)
    return StabilizerCode(stabilizers=rows, logical_x=code.logical_x, logical_z=code.logical_z)


def evaluate_with_fallback(
    obj: MemoryObjective,
    sampled: MemoryObjective | None,
    code: StabilizerCode,
    d: int,
    sampled_above: float | None,
    sparsify: bool = False,
) -> dict:
    """The proxy result of ``obj`` or, when the expected number of faults per
    shot of either basis exceeds ``sampled_above``, the sampled-tier result
    of ``sampled`` for the same code (``tier_used`` says which; the proxy
    value is kept as ``proxy_p_round``). With ``sparsify`` the generators are
    first replaced by the sparsest basis (:func:`sparsest_code`)."""
    if sparsify:
        code = sparsest_code(code)
    res = compact_result(obj.evaluate(code, d, use_cache=False))
    res["sparsified"] = bool(sparsify)
    res["tier_used"] = res["tier"]
    res["expected_faults_max"] = max(b["expected_faults"] for b in res["per_basis"].values())
    if sampled is not None and sampled_above is not None and res["expected_faults_max"] > sampled_above:
        srs = compact_result(sampled.evaluate(code, d, use_cache=False))
        srs["proxy_p_round"] = res["p_round"]
        srs["proxy_seconds"] = res["seconds"]
        srs["seconds"] = round(res["seconds"] + srs["seconds"], 4)
        srs["tier_used"] = "sampled"
        srs["expected_faults_max"] = res["expected_faults_max"]
        srs["sparsified"] = bool(sparsify)
        return srs
    return res


def _code_payload(code: StabilizerCode) -> tuple:
    return (
        code.stabilizers.tobytes(),
        code.stabilizers.shape,
        code.logical_x.tobytes(),
        code.logical_z.tobytes(),
        code.logical_x.shape,
    )


def _code_from_payload(payload: tuple) -> StabilizerCode:
    s, ss, lx, lz, ls = payload
    return StabilizerCode(
        stabilizers=np.frombuffer(s, dtype=np.uint8).reshape(ss),
        logical_x=np.frombuffer(lx, dtype=np.uint8).reshape(ls),
        logical_z=np.frombuffer(lz, dtype=np.uint8).reshape(ls),
    )


def _worker_eval(payload: tuple, d: int) -> dict:
    return evaluate_with_fallback(
        _WORKER_OBJ, _WORKER_SAMPLED, _code_from_payload(payload), d, _WORKER_SAMPLED_ABOVE, _WORKER_SPARSIFY
    )


class MemoryTermPool:
    """Memory-objective evaluations of distinct codes in worker processes,
    cached by code key. ``workers = 0`` evaluates inline in this process.
    ``evaluate(codes)`` takes ``{key: StabilizerCode}`` and returns ``{key:
    compact result}``; the new keys of a call are submitted together and
    waited for, so a call costs one evaluation's wall time per ``workers``
    new codes."""

    def __init__(
        self,
        name: str,
        d: int,
        workers: int = POOL,
        tier: str = TRAIN_TIER,
        scale: float | None = None,
        seed: int = 0,
        threads: int = 1,
        sampled_above: float | None = None,
        sampled_sampling: dict | None = None,
        sparsify: bool = False,
    ):
        self.name, self.d, self.workers = name, int(d), int(workers)
        self.sparsify = bool(sparsify)
        self.cache: dict[bytes, dict] = {}
        self.evals = 0
        self.hits = 0
        self.sampled_evals = 0
        self.seconds = 0.0  # summed worker seconds
        self.wall = 0.0  # wall time the caller waited
        self.objective = make_objective(name, tier, scale, seed)
        self.sampled_above = sampled_above
        self.sampled_sampling = dict(sampled_sampling or TRAIN_SAMPLED)
        self.sampled = (
            make_objective(name, "sampled", scale, seed, **self.sampled_sampling) if sampled_above is not None else None
        )
        self.pool = None
        if self.workers > 0:
            ctx = multiprocessing.get_context("spawn")
            self.pool = ProcessPoolExecutor(
                self.workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(
                    name,
                    tier,
                    scale,
                    seed,
                    SCORE_SHOTS,
                    SCORE_MIN_FAILURES,
                    SCORE_MAX_SHOTS,
                    threads,
                    sampled_above,
                    self.sampled_sampling,
                    self.sparsify,
                ),
            )

    def evaluate(self, codes: dict) -> dict:
        t0 = time.perf_counter()
        new = {k: c for k, c in codes.items() if k not in self.cache}
        self.hits += len(codes) - len(new)
        if new:
            if self.pool is None:
                for k, c in new.items():
                    self.cache[k] = evaluate_with_fallback(
                        self.objective, self.sampled, c, self.d, self.sampled_above, self.sparsify
                    )
            else:
                futs = {k: self.pool.submit(_worker_eval, _code_payload(c), self.d) for k, c in new.items()}
                for k, f in futs.items():
                    self.cache[k] = f.result()
            self.evals += len(new)
            self.sampled_evals += sum(self.cache[k].get("tier_used") == "sampled" for k in new)
            self.seconds += sum(self.cache[k]["seconds"] for k in new)
        self.wall += time.perf_counter() - t0
        return {k: self.cache[k] for k in codes}

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def describe(self) -> dict:
        return {
            **self.objective.describe(),
            "workers": self.workers,
            "sampled_above": self.sampled_above,
            "sampled_sampling": self.sampled_sampling if self.sampled is not None else None,
            "sparsify": self.sparsify,
        }


# --------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------
class TorchMemoryReward:
    """The reward of the module docstring: distance, fractional, and padding
    terms through :class:`gpu_scale.DistanceKernel` on ``device``; on
    target-distance states with no weight-1 stabilizer the memory term from
    ``pool`` and the LDPC term of ``noise_aware``. The terminal terms are
    skipped on reward calls over a sub-batch (the environment's re-evaluation
    of reset elements), as in ``noise_aware.NoiseAwareReward``."""

    def __init__(
        self,
        env,
        setting: na.Setting,
        pool: MemoryTermPool,
        device,
        chunk: int | None = None,
        weight: float = MEMORY_WEIGHT,
        p_ref: float = P_REF,
        clip: float = CLIP,
        w_max: int = na.W_MAX,
        ldpc_scale: float = na.LDPC_SCALE,
        gate_ref: int | None = None,
        gate_weight: float = 0.0,
    ):
        self.env = env
        self.d, self.n, self.k = setting.d, setting.n, setting.k
        self.fractional = bool(setting.fractional)
        self.pool = pool
        self.weight, self.p_ref, self.clip = float(weight), float(p_ref), float(clip)
        self.w_max, self.ldpc_scale = int(w_max), float(ldpc_scale)
        self.gate_ref, self.gate_weight = gate_ref, float(gate_weight)
        self.kernel = gs.DistanceKernel(setting.n, setting.k, setting.d, device, chunk)
        self.visits: dict[bytes, int] = {}
        self.terminal_calls = 0
        self.last_success = None

    def term(self, p_round: float) -> float:
        return memory_term(p_round, self.weight, self.p_ref, self.clip)

    def gate_term(self, two_qubit_gates: int) -> float:
        """``-gate_weight * max(0, g / gate_ref - 1)`` on the memory circuit's
        two-qubit gates per round; zero without ``gate_ref``."""
        if self.gate_ref is None or self.gate_weight == 0.0:
            return 0.0
        return -self.gate_weight * max(0.0, two_qubit_gates / self.gate_ref - 1.0)

    def __call__(self, tab: np.ndarray, k: int) -> np.ndarray:
        d, frac, pad = self.kernel(tab)
        r = np.minimum(d, self.d) / float(self.d) - 1.0
        if self.fractional:
            below = d < self.d
            r[below] = (d[below] - 1 + frac[below]) / float(self.d) - 1.0
        r = r + pad
        succ = (d >= self.d) & (pad == 0.0)
        if succ.any() and tab.shape[0] == self.env.batch_size:
            idx = np.flatnonzero(succ)
            keys, codes = [], {}
            for b in idx:
                key = na.code_key(tab[b], k)
                keys.append(key)
                if key not in codes:
                    codes[key] = StabilizerCode.from_tableau(tab, k, int(b))
                self.visits[key] = self.visits.get(key, 0) + 1
            res = self.pool.evaluate(codes)
            mem = np.array(
                [
                    self.term(res[key]["p_round"]) + self.gate_term(res[key]["per_basis"]["Z"]["two_qubit_gates"])
                    for key in keys
                ],
                dtype=float,
            )
            r[idx] += mem
            if self.ldpc_scale > 0.0:
                r[idx] += ldpc_penalty(
                    tab[idx], k, self.w_max, scale=self.ldpc_scale, exact_max_generators=na.EXACT_GENERATORS
                )
            self.terminal_calls += len(idx)
        self.last_success = succ
        return r

    def success_fn(self, tab, k, r):
        if self.last_success is not None and self.last_success.shape == r.shape:
            return self.last_success
        return r >= 0.0


def reward_description(variant: str = "run1") -> dict:
    v = VARIANTS[variant]
    if variant == "run1":
        form = (
            "(d - 1 + f)/d* - 1 - q_1 + [d >= d*, q_1 = 0](0.1 clip(log10(1e-2 / p_round), -3, 3) - 0.25 max(0, w* - 4)), "
            "f = detected fraction of weight-d Paulis, q_1 = fraction of weight-1 Paulis in S, "
            "p_round = memory objective (proxy tier, R = 4, both bases) under the training profile"
        )
    else:
        form = (
            f"(d - 1 + f)/d* - 1 - q_1 + [d >= d*, q_1 = 0](0.1 clip(log10(1e-2 / p_round), -3, 3) - {v['gate_weight']} max(0, g / {v['gate_ref']} - 1)), "
            "f = detected fraction of weight-d Paulis, q_1 = fraction of weight-1 Paulis in S, "
            f"g = two-qubit gates per round of the memory circuit, p_round = memory objective (R = 4, both bases; proxy tier, "
            f"or the sampled tier when the expected faults per shot exceed {v['sampled_above']}) under the training profile; no w* term"
        )
    return {
        "variant": variant,
        "form": form,
        "padding_penalty": "reward.low_weight_stabilizer_penalty(max_weight=1); success requires no weight-1 stabilizer element",
        "fractional_distance": SETTING.fractional,
        "memory_weight": MEMORY_WEIGHT,
        "p_ref": P_REF,
        "clip": CLIP,
        "w_max": na.W_MAX,
        "ldpc_scale": v["ldpc_scale"],
        "gate_ref": v["gate_ref"],
        "gate_weight": v["gate_weight"],
        "sampled_above": v["sampled_above"],
        "selection": "highest reward" if v["select"] == "reward" else "lowest training p_round",
        "source": "autoqec_sim.experiments.memory_ppo.TorchMemoryReward (memory_objective.MemoryReward term, noise_aware LDPC term)",
    }


# --------------------------------------------------------------------------
# code analysis
# --------------------------------------------------------------------------
def memory_scores(
    code: StabilizerCode, tier: str = TRAIN_TIER, names=PROFILE_ORDER, seed: int = 0, sparsify: bool = False, **sampling
) -> dict:
    """``{profile: compact result}`` of ``code`` under the cells' profiles."""
    if sparsify:
        code = sparsest_code(code)
    out = {}
    for name in names:
        obj = make_objective(name, tier, None, seed, **sampling)
        out[name] = compact_result(obj.evaluate(code, SETTING.d, use_cache=False))
    return out


def analyze_code(
    gates, models: dict, train_profile: str | None, train_result: dict | None, variant: str = "run1"
) -> dict:
    """:func:`noise_aware.analyze_circuit` plus the proxy memory scores under
    the four profiles and the reward terms of the training cell."""
    v = VARIANTS[variant]
    a = na.analyze_circuit(gates, SETTING, models)
    code = replay([tuple(g) for g in gates], SETTING.n)
    a["memory_proxy"] = memory_scores(code, sparsify=v["sparsify"])
    a["memory_proxy_sparsified"] = v["sparsify"]
    if train_profile is not None:
        p = train_result["p_round"] if train_result is not None else a["memory_proxy"][train_profile]["p_round"]
        g = a["memory_proxy"][train_profile]["per_basis"]["Z"]["two_qubit_gates"]
        a["reward_terms"] = {
            "distance_term": min(a["distance"], SETTING.d) / SETTING.d - 1.0,
            "padding_term": -a["weight_enumerator_S"][1] / (3 * SETTING.n),
            "memory_term": memory_term(p),
            "ldpc_term": -v["ldpc_scale"] * max(0, a["w_star"] - na.W_MAX),
            "ldpc_excess": max(0, a["w_star"] - na.W_MAX),
            "gate_term": -v["gate_weight"] * max(0.0, g / v["gate_ref"] - 1.0) if v["gate_ref"] else 0.0,
            "two_qubit_gates": g,
            "train_p_round": p,
            "train_tier": (train_result or {}).get("tier_used", "proxy"),
            "train_proxy_p_round": (train_result or {}).get("proxy_p_round", p if train_result else None),
            "expected_faults_max": (train_result or {}).get("expected_faults_max"),
        }
    return a


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def run_cell(
    profile_name: str,
    seed: int,
    device: str = "cpu",
    batch: int = BATCH,
    steps: int = STEPS,
    wall_min: float = WALL_MIN,
    threads: int = THREADS,
    pool_workers: int = POOL,
    chunk: int | None = None,
    scale: float | None = None,
    verbose: bool = False,
    log: gs.RunLog | None = None,
    calib: dict | None = None,
    log_every: int = LOG_EVERY,
    max_recorded_codes: int = 5000,
    variant: str = "run1",
) -> dict:
    """One (profile, seed) cell: the ``noise4x4`` training loop of
    :func:`gpu_scale.run_noise4x4` with :class:`TorchMemoryReward`. Trains
    the full budget and analyses the best, the lowest-``p_round``, and the
    modal code. ``variant`` selects the reward terms and the selection rule
    (``VARIANTS``)."""
    import torch

    setting = SETTING
    vp = VARIANTS[variant]
    wall_s = 60.0 * float(wall_min)
    dev = gs.resolve_device(device)
    chunk = int(chunk or gs.default_chunk(dev))
    log = log or gs.RunLog(None, verbose)
    started = _now()
    pool = MemoryTermPool(
        profile_name,
        setting.d,
        pool_workers,
        TRAIN_TIER,
        scale,
        seed,
        threads,
        sampled_above=vp["sampled_above"],
        sparsify=vp["sparsify"],
    )
    log(
        f"memory_ppo {profile_name} seed {seed} ({variant}): device {dev} ({gs.device_name(dev)}), batch {batch}, budget {steps:,} env steps, "
        f"wall cap {wall_s / 60:.0f} min, memory pool {pool_workers} worker(s), objective {pool.describe()}, reward {vp}"
    )
    t_setup = time.perf_counter()
    calib = calib or na.calibrate(setting)
    env = na.make_env(setting, seed, batch)
    models = na.calibrated_models(setting, calib, env.graph)
    reward = TorchMemoryReward(
        env,
        setting,
        pool,
        device=dev,
        chunk=chunk,
        ldpc_scale=vp["ldpc_scale"],
        gate_ref=vp["gate_ref"],
        gate_weight=vp["gate_weight"],
    )
    reward_timer = gs.Timed(reward)
    env.reward_fn = reward_timer
    env.success_fn = reward.success_fn
    tracker = na.CandidateTracker(setting.d, fractional=setting.fractional)
    env.tracker = tracker
    env.reset()
    agent, config, update_timer = gs._make_agent(env, seed, dev, threads, steps, wall_s)
    log(
        f"  policy {agent.policy}, {agent.num_parameters():,} parameters, {env.num_actions} actions, setup {time.perf_counter() - t_setup:.1f} s"
    )

    def best_p():
        vals = [v["p_round"] for v in pool.cache.values()]
        return min(vals) if vals else None

    def cb(agent, new):
        tracker.snapshot()
        rec = agent.history[-1]
        rec["successes_in_update"] = len(new)
        rec["distinct_success_codes"] = len(tracker.success)
        rec["best_terminal_reward"] = tracker.best_reward()
        rec["memory_evals"] = pool.evals
        rec["memory_cache_hits"] = pool.hits
        rec["memory_sampled_evals"] = pool.sampled_evals
        rec["memory_wall_s"] = round(pool.wall, 1)
        rec["best_p_round"] = best_p()
        if agent.updates % log_every == 0 or new:
            bp = rec["best_p_round"]
            log(
                f"  upd {rec['update']:5d} steps {rec['env_steps']:11,d} r {rec['mean_reward']:+.4f} succ {rec['success_rate']:.3f} "
                f"ent {rec['entropy']:.3f} distinct {len(tracker.success)} evals {pool.evals} ({pool.sampled_evals} sampled) best r {tracker.best_reward() if tracker.success else float('nan'):+.4f} "
                f"best p {bp if bp is None else f'{bp:.3e}'} mem {pool.wall:.0f}s t {rec['wall_time_s']:.0f}s ({rec['env_steps'] / max(rec['wall_time_s'], 1e-9):,.0f} steps/s)"
            )
        return False

    try:
        res = agent.train(steps, max_time_s=wall_s, stop_on_success=False, callback=cb, verbose=False)
    finally:
        pool.close()
    timing = gs._timing(agent, res, reward_timer, update_timer, reward.kernel)
    timing["memory_wall_s"] = round(pool.wall, 1)
    timing["memory_worker_s"] = round(pool.seconds, 1)
    log(
        f"  trained: {timing['env_steps']:,} env steps in {timing['wall_time_s']} s ({timing['env_steps_per_s']:,} steps/s); "
        f"reward {timing['reward_time_s']} s of which memory {timing['memory_wall_s']} s ({pool.evals} evaluations, {pool.hits} cache hits), update {timing['update_time_s']} s"
    )

    # codes: join the tracker (gates, reward) with the memory cache (p_round)
    t1 = time.perf_counter()
    codes = []
    for key, v in tracker.success.items():
        m = pool.cache.get(key)
        codes.append(
            {
                "key": key.hex(),
                "reward": v["reward"],
                "gates": v["gates"],
                "gate_count": len(v["gates"]),
                "visits": reward.visits.get(key, 0),
                "p_round": m["p_round"] if m else None,
                "p_round_basis": {b: m["per_basis"][b]["p_round"] for b in m["per_basis"]} if m else None,
                "expected_faults": {b: m["per_basis"][b]["expected_faults"] for b in m["per_basis"]} if m else None,
                "two_qubit_gates": m["per_basis"]["Z"]["two_qubit_gates"] if m else None,
                "batches": m["per_basis"]["Z"]["batches"] if m else None,
                "memory_seconds": m["seconds"] if m else None,
                "tier": m.get("tier_used", "proxy") if m else None,
                "proxy_p_round": m.get("proxy_p_round", m["p_round"]) if m else None,
                "gate_term": reward.gate_term(m["per_basis"]["Z"]["two_qubit_gates"]) if m else None,
            }
        )
    codes.sort(key=lambda c: -c["reward"])
    for c in codes:
        code = replay([tuple(g) for g in c["gates"]], setting.n)
        c["is_css"] = bool(code.is_css())
    if len(codes) > max_recorded_codes:
        kept = codes[:max_recorded_codes]
        dropped = len(codes) - len(kept)
    else:
        kept, dropped = codes, 0

    if tracker.success:
        scored = [c for c in codes if c["p_round"] is not None]
        low_entry = min(scored, key=lambda c: c["p_round"]) if scored else None
        if vp["select"] == "p_round" and low_entry is not None:
            best_entry = low_entry
            low_entry = codes[0]  # the highest-reward code is reported in its place
            selection = "lowest training p_round among states that reached the target distance (the highest-reward code is reported as the alternate)"
        else:
            best_entry = codes[0]
            selection = "highest reward among states that reached the target distance"
    else:
        best_entry, low_entry = None, None
        selection = f"no state reached distance {setting.d}; best distance seen {tracker.partial_d} ({len(tracker.partial)} distinct states kept)"
    analysis = (
        analyze_code(
            best_entry["gates"], models, profile_name, pool.cache.get(bytes.fromhex(best_entry["key"])), variant
        )
        if best_entry
        else None
    )
    if analysis is not None:
        analysis["reward"] = best_entry["reward"]
    low_analysis = None
    if low_entry is not None and low_entry["key"] != best_entry["key"]:
        low_analysis = analyze_code(
            low_entry["gates"], models, profile_name, pool.cache.get(bytes.fromhex(low_entry["key"])), variant
        )
        low_analysis["reward"] = low_entry["reward"]
    modal, modal_visits, window_visits = tracker.modal_code()
    modal_analysis = None
    if modal is not None:
        mkey = next(k for k, v in tracker.success.items() if v is modal)
        modal_analysis = analyze_code(modal["gates"], models, profile_name, pool.cache.get(mkey), variant)
        modal_analysis["reward"] = modal["reward"]
        modal_analysis["visits_last_20_updates"] = modal_visits
        modal_analysis["successes_last_20_updates"] = window_visits
        modal_analysis["same_as_best"] = bool(mkey.hex() == best_entry["key"]) if best_entry else False
    if analysis:
        log(
            f"  best: d={analysis['distance']} dX={analysis['distance_x']} dZ={analysis['distance_z']} w*={analysis['w_star']} "
            f"css={analysis['is_css']} frame={analysis['frame'].get('css_frame_found')} p_round ion/ion_flag/board/heron="
            + "/".join(f"{analysis['memory_proxy'][p]['p_round']:.3e}" for p in PROFILE_ORDER)
            + f" ({analysis['circuit']['gates']} gates); analysis {time.perf_counter() - t1:.1f} s"
        )
    partial_best = None
    if not tracker.success and tracker.partial:
        pb = min(tracker.partial.values(), key=lambda v: -v["reward"])
        partial_best = gs.describe_state(
            replay([tuple(g) for g in pb["gates"]], setting.n).to_tableau()[0], pb["gates"], setting.k, setting.d
        )
    return {
        "kind": "run",
        "target": "memory_ppo",
        "target_description": "noise-aware 4x4 setting with the memory objective as the noise term, policy=auto",
        "variant": variant,
        "variant_params": vp,
        "setting": setting.to_dict(),
        "profile": profile_name,
        "profile_label": PROFILES[profile_name]["label"],
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
        "objective": pool.describe(),
        "profiles": {p: {**PROFILES[p], "objective": make_objective(p).describe()} for p in PROFILE_ORDER},
        "reward": reward_description(variant),
        "encoder_calibration": calib,
        "ppo_config": dict(config.to_dict(), policy_resolved=agent.policy),
        "num_parameters": res.num_parameters,
        "num_actions": env.num_actions,
        **timing,
        "success": bool(tracker.success),
        "successful_episodes": tracker.success_episodes,
        "distinct_success_codes": len(tracker.success),
        "partial_distance": tracker.partial_d if not tracker.success else None,
        "distinct_partial_codes": len(tracker.partial) if not tracker.success else None,
        "memory_evals": pool.evals,
        "memory_cache_hits": pool.hits,
        "memory_sampled_evals": pool.sampled_evals,
        "memory_terminal_calls": reward.terminal_calls,
        "memory_seconds_per_code": round(pool.seconds / pool.evals, 3) if pool.evals else None,
        "first_success": gs._first_success(agent, res),
        "first_success_env_steps": res.first_success.env_steps if res.first_success else None,
        "final_success_rate": res.history[-1]["success_rate"] if res.history else None,
        "selection": selection,
        "best": analysis,
        "best_reward": best_entry["reward"] if best_entry else None,
        "lowest_p_round": low_analysis,  # run2: the highest-reward code when it differs from the selected one
        "modal": modal_analysis,
        "partial_best": partial_best,
        "codes": kept,
        "codes_dropped": dropped,
        "analysis_time_s": round(time.perf_counter() - t1, 1),
        "history": res.history,
    }


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def run_name(profile: str, seed: int) -> str:
    return f"run_{profile}_seed{seed}"


def run_path(out_dir: Path, profile: str, seed: int) -> Path:
    return out_dir / (run_name(profile, seed) + ".json")


def log_path(out_dir: Path, profile: str, seed: int) -> Path:
    return out_dir / "logs" / (run_name(profile, seed) + ".log")


def _report_work(work: dict):
    """Report counters to the compute ledger's wrap through its work file."""
    path = os.environ.get("COMPUTE_LEDGER_WORK_FILE")
    if not path:
        return
    with contextlib.suppress(OSError):
        Path(path).write_text(json.dumps(work, default=_json_default))


def run_one(profile: str, seed: int, out_dir: Path, **kw) -> dict:
    log = gs.RunLog(log_path(out_dir, profile, seed), verbose=kw.pop("verbose", False))
    try:
        rec = run_cell(profile, seed, log=log, **kw)
    finally:
        log.close()
    path = run_path(out_dir, profile, seed)
    gs.write(path, rec)
    _report_work(
        {
            "gpu_seconds": rec["wall_time_s"],
            "env_steps": rec["env_steps"],
            "episodes": rec["episodes"],
            "updates": rec["updates"],
            "memory_evals": rec["memory_evals"],
            "distinct_success_codes": rec["distinct_success_codes"],
            "first_success_env_steps": rec["first_success_env_steps"],
            "artifacts": [str(path)],
            "outcome": "success" if rec["success"] else "negative_result",
        }
    )
    return rec


def load_runs(out_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(out_dir.glob("run_*.json")):
        try:
            with open(p) as f:
                r = json.load(f)
        except json.JSONDecodeError:
            print(f"skipping unreadable {p}")
            continue
        r["_file"] = p.name
        recs.append(r)
    return recs


def load_encoder_runs(encoder_dir: Path = ENCODER_DIR) -> list[dict]:
    recs = []
    for p in sorted(encoder_dir.glob("run_noise4x4_*.json")):
        with open(p) as f:
            r = json.load(f)
        r["_file"] = str(p)
        recs.append(r)
    return recs


# --------------------------------------------------------------------------
# score: proxy and bposd under the four profiles for every code of interest
# --------------------------------------------------------------------------
def _score_worker(item: tuple) -> tuple:
    name, payload, profile, tier, shots, min_failures, max_shots, seed, sparsify = item
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    code = _code_from_payload(payload)
    if sparsify:
        code = sparsest_code(code)
    obj = make_objective(profile, tier, None, seed, shots, min_failures, max_shots)
    res = compact_result(obj.evaluate(code, SETTING.d, use_cache=False))
    res["sparsified"] = bool(sparsify)
    return name, profile, tier, res


GROUP_LABELS = {
    "memory": "memory-trained ",
    "memory_low": "lowest-proxy ",
    "memory2": "run-2 memory-trained ",
    "memory2_alt": "run-2 highest-reward ",
    "encoder": "encoder-trained ",
}
GROUP_ORDER = {"memory2": 0, "memory2_alt": 1, "memory": 2, "memory_low": 3, "encoder": 4, "known": 5}
GROUP_PREFIX = {"memory": "mem_", "memory_low": "low_", "memory2": "mem2_", "memory2_alt": "alt2_", "encoder": "enc_"}


def _run_codes(runs: list[dict], field: str, group: str, rel: str) -> dict:
    out = {}
    for r in runs:
        a = r.get(field)
        if a:
            out[f"{GROUP_PREFIX[group]}{r['profile']}_seed{r['seed']}"] = {
                "label": f"{GROUP_LABELS[group]}{SHORT[r['profile']]} seed {r['seed']}",
                "source": f"{rel}/{r['_file']}" + (f" ({field})" if field != "best" else ""),
                "group": group,
                "train_profile": r["profile"],
                "seed": r["seed"],
                "gates": a["gates"],
            }
    return out


def collect_codes(
    out_dir: Path,
    encoder_dir: Path = ENCODER_DIR,
    which=("runs", "encoder", "known"),
    run1_dir: Path | None = None,
    runs_group: str = "memory",
) -> dict:
    """``{name: {label, source, gates or stabilizers, ...}}`` of the codes to
    score. Groups: ``runs`` (best code per run of ``out_dir``, group
    ``runs_group``), ``lowest`` (the alternate code per run: the
    lowest-proxy code in run 1, the highest-reward code in run 2), ``run1``
    (best and lowest-proxy codes of ``run1_dir``), ``encoder``, ``known``."""
    codes = {}
    rel = "results/memory_ppo" if out_dir == DEFAULT_OUT else str(out_dir)
    if "runs" in which:
        codes.update(_run_codes(load_runs(out_dir), "best", runs_group, rel))
    if "lowest" in which:
        codes.update(
            _run_codes(
                load_runs(out_dir), "lowest_p_round", "memory_low" if runs_group == "memory" else "memory2_alt", rel
            )
        )
    if "run1" in which and run1_dir is not None:
        r1 = load_runs(run1_dir)
        codes.update(_run_codes(r1, "best", "memory", "results/memory_ppo"))
        codes.update(_run_codes(r1, "lowest_p_round", "memory_low", "results/memory_ppo"))
    if "encoder" in which:
        for r in load_encoder_runs(encoder_dir):
            if r.get("best"):
                codes[f"enc_{r['profile']}_seed{r['seed']}"] = {
                    "label": f"encoder-trained {r['profile']} seed {r['seed']}",
                    "source": r["_file"],
                    "group": "encoder",
                    "train_profile": r["profile"],
                    "seed": r["seed"],
                    "gates": r["best"]["gates"],
                }
    if "known" in which:
        known = na.known_codes_for(SETTING)
        for name in KNOWN:
            codes[name] = {
                "label": known[name]["description"],
                "source": "noise_aware.known_codes_for",
                "group": "known",
                "stabilizers": known[name]["stabilizers"],
            }
    return codes


def code_of(entry: dict) -> StabilizerCode:
    if "gates" in entry:
        return replay([tuple(g) for g in entry["gates"]], SETTING.n)
    return StabilizerCode.from_stabilizers(entry["stabilizers"])


def describe_code(code: StabilizerCode) -> dict:
    tab = code.to_tableau()
    dx, _ = na.pure_type_distance(tab, SETTING.k, "X")
    dz, _ = na.pure_type_distance(tab, SETTING.k, "Z")
    st = na.sparsest_basis_stats(tab, SETTING.k, exact_max_generators=na.EXACT_GENERATORS)
    fr = na.css_frames(code)
    return {
        "n": code.n,
        "k": code.k,
        "distance": int(code.distance_exact()),
        "distance_x": dx,
        "distance_z": dz,
        "w_star": int(st["w_star"][0]),
        "is_css": bool(code.is_css()),
        "stabilizers": code.stabilizer_strings(),
        "frame": {k: fr[k] for k in ("css_frame_found", "frames", "frame_size", "frame", "hadamard_only") if k in fr},
    }


def run_score(
    out_dir: Path,
    encoder_dir: Path = ENCODER_DIR,
    workers: int = 4,
    tiers=("proxy", "bposd"),
    shots: int = SCORE_SHOTS,
    min_failures: int = SCORE_MIN_FAILURES,
    max_shots: int | None = SCORE_MAX_SHOTS,
    seed: int = 0,
    which=("runs", "encoder", "known"),
    profiles=PROFILE_ORDER,
    verbose: bool = False,
    run1_dir: Path | None = None,
    runs_group: str = "memory",
    sparsify: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Score every code of :func:`collect_codes` under ``profiles`` at
    ``tiers``; merges into ``out_dir/scores.json`` (existing entries are
    kept unless recomputed)."""
    path = out_dir / "scores.json"
    prev = json.loads(path.read_text()) if path.exists() else {"codes": {}, "scores": {}}
    codes = collect_codes(out_dir, encoder_dir, which, run1_dir, runs_group)
    items = []
    t0 = time.perf_counter()
    for name, entry in codes.items():
        code = code_of(entry)
        prev["codes"][name] = {**entry, **describe_code(code), "sparsified": bool(sparsify)}
        prev["scores"].setdefault(name, {})
        for prof in profiles:
            prev["scores"][name].setdefault(prof, {})
            for tier in tiers:
                if skip_existing and tier in prev["scores"][name][prof]:
                    continue
                items.append((name, _code_payload(code), prof, tier, shots, min_failures, max_shots, seed, sparsify))
    if verbose:
        print(
            f"scoring {len(codes)} codes x {len(profiles)} profiles x {len(tiers)} tiers = {len(items)} evaluations on {workers} worker(s)",
            flush=True,
        )
    # bposd first (the long ones), so the pool stays busy
    items.sort(key=lambda it: it[3] != "bposd")
    ctx = multiprocessing.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for name, prof, tier, res in ex.map(_score_worker, items):
            prev["scores"][name][prof][tier] = res
            done += 1
            if verbose:
                print(
                    f"  [{done}/{len(items)}] {name} {prof} {tier} p_round {res['p_round']:.3e} ({res['seconds']:.0f} s) t {time.perf_counter() - t0:.0f} s",
                    flush=True,
                )
            prev["meta"] = {
                "updated_at": _now(),
                "host": socket.gethostname(),
                "shots": shots,
                "min_failures": min_failures,
                "max_shots": max_shots,
                "seed": seed,
                "rounds": ROUNDS,
                "connectivity": CONNECTIVITY,
                "sparsify": bool(sparsify),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prev, indent=1, default=_json_default))
    return prev


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _yes(b) -> str:
    return "yes" if b else "no"


def _p(x, digits=3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}g}"


def _by_profile(recs: list[dict]) -> dict:
    out = {}
    for r in recs:
        out.setdefault(r["profile"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["seed"])
    return out


def _score(scores: dict, name: str, prof: str, tier: str):
    try:
        return scores["scores"][name][prof][tier]["p_round"]
    except KeyError:
        return None


def _frame_str(a: dict) -> str:
    fr = a.get("frame") or {}
    if not fr.get("css_frame_found"):
        return "none"
    return f"H on {fr.get('frame_size')} qubit(s)" + (f" ({fr.get('frames')} frames)" if fr.get("frames") else "")


def where_table(recs: list[dict]) -> list[str]:
    lines = [
        "| profile | seed | device | batch | budget (env steps) | env steps | updates | wall (s) | env steps/s | reward share | memory share | update share | memory evals | s per code | cache hits | wall cap hit | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(recs, key=lambda r: (PROFILE_ORDER.index(r["profile"]), r["seed"])):
        w = max(r["wall_time_s"], 1e-9)
        lines.append(
            f"| {r['profile']} | {r['seed']} | {r['device']} ({r['device_name']}) | {r['batch']} | {r['budget_env_steps']:,} | {r['env_steps']:,} | {r['updates']} | "
            f"{r['wall_time_s']:.0f} | {r['env_steps_per_s']:,.0f} | {r['reward_time_s'] / w:.0%} | {r.get('memory_wall_s', 0) / w:.0%} | {r['update_time_s'] / w:.0%} | "
            f"{r['memory_evals']} | {_p(r.get('memory_seconds_per_code'))} | {r['memory_cache_hits']} | {_yes(r['hit_wall_cap'])} | `{r['_file']}` |"
        )
    return lines


def runs_table(recs: list[dict]) -> list[str]:
    lines = [
        "| profile | seeds | seeds reaching d* | first success (env steps) | final success rate | distinct target codes per seed | memory evaluations per seed | best training p_round per seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for prof, rs in sorted(_by_profile(recs).items(), key=lambda kv: PROFILE_ORDER.index(kv[0])):
        ok = [r for r in rs if r["success"]]
        best = [r["best"]["reward_terms"]["train_p_round"] for r in ok if r.get("best")]
        lines.append(
            f"| {prof} | {len(rs)} | {len(ok)}/{len(rs)} | {_mean_range([r['first_success_env_steps'] for r in ok]) if ok else '-'} | "
            f"{_mean_range([r['final_success_rate'] for r in rs])} | {_mean_range([r['distinct_success_codes'] for r in rs])} | "
            f"{_mean_range([r['memory_evals'] for r in rs])} | {_mean_range(best) if best else '-'} |"
        )
    return lines


def codes_table(recs: list[dict], scores: dict | None, name_prefix: str = "mem_") -> list[str]:
    lines = [
        "| trained on | seed | reached d* | d | d_X | d_Z | w* | CSS | H frame | gates | 2q gates per round | batches | E (train profile) | train p_round | tier | "
        + " | ".join(f"bposd {SHORT[p]}" for p in PROFILE_ORDER)
        + " | file |",
        "|" + "---|" * (16 + len(PROFILE_ORDER)) + "---|",
    ]
    for r in sorted(recs, key=lambda r: (PROFILE_ORDER.index(r["profile"]), r["seed"])):
        a = r.get("best")
        if not a:
            pb = r.get("partial_best") or {}
            lines.append(
                f"| {r['profile']} | {r['seed']} | no | {pb.get('distance_capped', '-')} | - | - | {pb.get('w_star', '-')} | {_yes(pb.get('is_css'))} | - | {pb.get('gate_count', '-')} | - | - | - | - | - | "
                + " | ".join("-" for _ in PROFILE_ORDER)
                + f" | `{r['_file']}` |"
            )
            continue
        name = f"{name_prefix}{r['profile']}_seed{r['seed']}"
        bp = [_p(_score(scores, name, p, "bposd")) if scores else "-" for p in PROFILE_ORDER]
        mp = a["memory_proxy"][r["profile"]]["per_basis"]["Z"]
        lines.append(
            f"| {r['profile']} | {r['seed']} | yes | {a['distance']} | {a['distance_x']} | {a['distance_z']} | {a['w_star']} | {_yes(a['is_css'])} | {_frame_str(a)} | "
            f"{a['circuit']['gates']} | {mp['two_qubit_gates']} | {mp['batches']} | {mp['expected_faults']:.2f} | {_p(a['reward_terms']['train_p_round'])} | {a['reward_terms'].get('train_tier', 'proxy')} | "
            + " | ".join(bp)
            + f" | `{r['_file']}` |"
        )
    return lines


def matrix_rows(scores: dict, tier: str) -> list[tuple[str, dict, int]]:
    """``(row label, {profile: [p_round per code]}, count)`` for the memory
    cells, the encoder cells, and the known codes."""
    rows = []
    codes = scores["codes"]
    for group in sorted(GROUP_LABELS, key=GROUP_ORDER.get):
        trained = ENCODER_PROFILES if group == "encoder" else PROFILE_ORDER
        for tp in trained:
            names = [n for n, c in codes.items() if c.get("group") == group and c.get("train_profile") == tp]
            if not names:
                continue
            vals = {p: [_score(scores, n, p, tier) for n in names] for p in PROFILE_ORDER}
            vals = {p: [v for v in vs if v is not None] for p, vs in vals.items()}
            label = GROUP_LABELS[group] + (tp if group == "encoder" else SHORT[tp])
            rows.append((label, vals, len(names)))
    for n in KNOWN:
        if n in codes:
            vals = {p: [v for v in [_score(scores, n, p, tier)] if v is not None] for p in PROFILE_ORDER}
            rows.append((f"known: {n}", vals, 1))
    return rows


def matrix_table(scores: dict, tier: str) -> list[str]:
    lines = [
        "| trained on \\ evaluated under | " + " | ".join(SHORT[p] for p in PROFILE_ORDER) + " | codes |",
        "|" + "---|" * (len(PROFILE_ORDER) + 2),
    ]
    for label, vals, count in matrix_rows(scores, tier):
        cells = []
        for p in PROFILE_ORDER:
            vs = vals[p]
            if not vs:
                cells.append("-")
            elif len(vs) == 1:
                cells.append(_p(vs[0]))
            else:
                cells.append(_mean_range(vs))
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {count} |")
    return lines


def per_code_table(scores: dict, tier: str) -> list[str]:
    lines = [
        "| code | d | d_X | d_Z | w* | CSS | H frame | " + " | ".join(SHORT[p] for p in PROFILE_ORDER) + " |",
        "|" + "---|" * (7 + len(PROFILE_ORDER)),
    ]
    for name, c in sorted(scores["codes"].items(), key=lambda kv: (GROUP_ORDER.get(kv[1].get("group"), 9), kv[0])):
        lines.append(
            f"| {c['label']} | {c['distance']} | {c['distance_x']} | {c['distance_z']} | {c['w_star']} | {_yes(c['is_css'])} | {_frame_str(c)} | "
            + " | ".join(_p(_score(scores, name, p, tier)) for p in PROFILE_ORDER)
            + " |"
        )
    return lines


TIER_PAIRS = (("proxy", "bposd"), ("proxy", "sampled"), ("sampled", "bposd"))


def tier_agreement(scores: dict) -> list[str]:
    """Per profile and pair of tiers, over every scored code: Spearman rank
    correlation and the median and range of the ratio second / first."""
    from .memory_objective import spearman

    lines = ["| profile | tiers | codes | Spearman | median ratio | range |", "|---|---|---|---|---|---|"]
    for p in PROFILE_ORDER:
        for t1, t2 in TIER_PAIRS:
            pairs = [(_score(scores, n, p, t1), _score(scores, n, p, t2)) for n in scores["codes"]]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None and a > 0]
            if len(pairs) < 3:
                continue
            a = np.array([x for x, _ in pairs])
            b = np.array([y for _, y in pairs])
            ratio = b / a
            lines.append(
                f"| {p} | {t2} / {t1} | {len(pairs)} | {spearman(a, b):.2f} | {np.median(ratio):.2f} | {ratio.min():.2f} to {ratio.max():.2f} |"
            )
    return lines


def comparison_lines(
    recs: list[dict], scores: dict, name_prefix: str = "mem_", label: str = "memory-trained"
) -> list[str]:
    """Plain statements per training profile: the selected codes against
    the encoder-trained codes, the surface code, and XZZX, all at the bposd
    tier under the training profile."""
    out = []
    enc_names = [n for n, c in scores["codes"].items() if c.get("group") == "encoder"]
    for prof in PROFILE_ORDER:
        rs = [r for r in recs if r["profile"] == prof and r.get("best")]
        if not rs:
            continue
        mine = [(r["seed"], _score(scores, f"{name_prefix}{prof}_seed{r['seed']}", prof, "bposd")) for r in rs]
        mine = [(s, v) for s, v in mine if v is not None]
        enc = [v for v in (_score(scores, n, prof, "bposd") for n in enc_names) if v is not None]
        surf = _score(scores, "rotated_surface", prof, "bposd")
        xz = _score(scores, "xzzx", prof, "bposd")
        if not mine:
            continue
        best_s, best_v = min(mine, key=lambda t: t[1])
        parts = [f"{SHORT[prof]}: {label} best per seed {', '.join(f'{_p(v)} (seed {s})' for s, v in mine)}"]
        if enc:
            parts.append(
                f"encoder-trained codes {_p(min(enc))} to {_p(max(enc))} (best of {len(enc)}); {sum(v < min(enc) for _, v in mine)} of {len(mine)} {label} seeds below the best encoder-trained code"
            )
        if surf is not None:
            parts.append(f"ratio to the surface code {', '.join(f'{v / surf:.1f}x' for _, v in mine)}")
        if surf is not None:
            parts.append(f"rotated surface {_p(surf)} ({sum(v < surf for _, v in mine)} of {len(mine)} seeds below)")
        if xz is not None:
            parts.append(f"XZZX {_p(xz)} ({sum(v < xz for _, v in mine)} of {len(mine)} seeds below)")
        parts.append(f"best seed {best_s}: {_p(best_v)}")
        out.append("- " + "; ".join(parts) + ".")
    return out


def make_figures(recs: list[dict], scores: dict | None, fig_dir: Path, prefix: str = "") -> list[str]:
    from . import figstyle

    plt = figstyle.apply()
    written = []
    byp = _by_profile(recs)
    profs = [p for p in PROFILE_ORDER if p in byp]
    if profs:
        fig, axes = plt.subplots(
            2, len(profs), figsize=(figstyle.WIDTH, figstyle.HEIGHT_DOUBLE), sharex=True, squeeze=False
        )
        for row in axes:
            for ax in row[1:]:
                ax.sharey(row[0])
        kern = np.ones(FIGURE_SMOOTH) / FIGURE_SMOOTH
        for j, prof in enumerate(profs):
            for i, r in enumerate(byp[prof]):
                col = figstyle.SERIES_COLORS[i % len(figstyle.SERIES_COLORS)]
                steps = np.array([h["env_steps"] for h in r["history"]]) / 1e6
                sr = np.array([h["success_rate"] for h in r["history"]], dtype=float)
                if sr.size >= FIGURE_SMOOTH:
                    axes[0, j].plot(
                        steps[FIGURE_SMOOTH - 1 :],
                        np.convolve(sr, kern, mode="valid"),
                        color=col,
                        label=f"seed {r['seed']}",
                    )
                else:
                    axes[0, j].plot(steps, sr, color=col, label=f"seed {r['seed']}")
                bp = np.array([h.get("best_p_round") or np.nan for h in r["history"]], dtype=float)
                axes[1, j].plot(steps, bp, color=col)
            axes[0, j].set_title(TITLES[prof], fontsize=7)
            axes[1, j].set_yscale("log")
            axes[1, j].set_xlabel("env steps (millions)")
            if scores:
                for name, ls in (("rotated_surface", "--"), ("xzzx", ":")):
                    v = _score(scores, name, prof, "proxy")
                    if v is not None:
                        axes[1, j].axhline(
                            v,
                            color=figstyle.NEUTRAL,
                            linestyle=ls,
                            linewidth=figstyle.THIN,
                            label=name if j == 0 else None,
                        )
            figstyle.panel_label(axes[0, j], "abcd"[j])
            figstyle.panel_label(axes[1, j], "efgh"[j])
        axes[0, 0].set_ylabel(f"success rate ({FIGURE_SMOOTH}-update mean)")
        axes[1, 0].set_ylabel("best training p_round (proxy)")
        axes[0, 0].legend(loc="lower right")
        if scores:
            axes[1, 0].legend(loc="upper right")
        stem = fig_dir / f"{prefix}curves"
        figstyle.save(fig, stem)
        plt.close(fig)
        written.append(stem.name + ".png")
    if scores and scores.get("scores"):
        rows = matrix_rows(scores, "bposd")
        fig, axes = plt.subplots(
            1, len(PROFILE_ORDER), figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE), squeeze=False, sharey=True
        )
        for j, p in enumerate(PROFILE_ORDER):
            ax = axes[0, j]
            labels, means, lo, hi, colors = [], [], [], [], []
            for label, vals, _ in rows:
                vs = vals[p]
                if not vs:
                    continue
                labels.append(
                    label.replace("run-2 memory-trained ", "r2 ")
                    .replace("run-2 highest-reward ", "r2 alt ")
                    .replace("memory-trained ", "mem ")
                    .replace("lowest-proxy ", "low ")
                    .replace("encoder-trained ", "enc ")
                    .replace("known: rotated_surface", "surface")
                    .replace("known: ", "")
                )
                means.append(np.mean(vs))
                lo.append(np.mean(vs) - min(vs))
                hi.append(max(vs) - np.mean(vs))
                colors.append(
                    figstyle.SERIES_COLORS[3]
                    if label.startswith("run-2")
                    else figstyle.SERIES_COLORS[0]
                    if label.startswith("memory")
                    else figstyle.SERIES_COLORS[4]
                    if label.startswith("lowest")
                    else figstyle.SERIES_COLORS[1]
                    if label.startswith("encoder")
                    else figstyle.NEUTRAL
                )
            x = np.arange(len(labels))
            ax.bar(
                x, means, yerr=[lo, hi], color=colors, width=0.7, error_kw={"elinewidth": figstyle.THIN, "capsize": 1.5}
            )
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=5.5)
            ax.set_title(f"under {SHORT[p]}", fontsize=7)
            figstyle.panel_label(ax, "abcd"[j])
        axes[0, 0].set_ylabel("p_round (bposd)")
        stem = fig_dir / f"{prefix}matrix"
        figstyle.save(fig, stem)
        plt.close(fig)
        written.append(stem.name + ".png")
    return written


def _sections(
    recs: list[dict],
    scores: dict | None,
    figs: list[str],
    fig_prefix: str = "",
    name_prefix: str = "mem_",
    label: str = "memory-trained",
    h: str = "##",
) -> list[str]:
    """The data sections of one experiment (run 1 or run 2)."""
    L = [
        f"{h} What ran where",
        "",
        "`reward share` is the fraction of the training wall time spent in the reward call (GPU distance kernel plus the host-side terminal terms), `memory share` the part of it spent waiting for the memory pool, `update share` the PPO update.",
        "",
        *where_table(recs),
        "",
        f"{h} Runs",
        "",
        *runs_table(recs),
        "",
    ]
    capped = [f"{r['profile']}/seed {r['seed']}" for r in recs if r["hit_wall_cap"]]
    if capped:
        L += ["Runs stopped by the wall cap before their step budget: " + ", ".join(capped) + ".", ""]
    sampled = [r for r in recs if r.get("memory_sampled_evals")]
    if sampled:
        L += [
            "Sampled-tier evaluations (candidates above the expected-fault threshold): "
            + ", ".join(
                f"{r['profile']}/seed {r['seed']}: {r['memory_sampled_evals']} of {r['memory_evals']}" for r in sampled
            )
            + ".",
            "",
        ]
    L += [
        f"{h} Codes found",
        "",
        "Selected code per seed (`selection` in the JSON; `H frame` is the smallest Hadamard frame that makes the stabilizer group CSS, `none` when no such frame exists; "
        "`2q gates per round`, `batches`, and `E` (expected faults per shot) are the memory circuit's under the training profile; `train p_round` is the value the reward saw at the tier in `tier`; the `bposd` columns are from `scores.json`).",
        "",
        *codes_table(recs, scores, name_prefix),
        "",
    ]
    for r in sorted(recs, key=lambda r: (PROFILE_ORDER.index(r["profile"]), r["seed"])):
        a = r.get("best")
        if not a:
            continue
        rt = a["reward_terms"]
        L += [
            f"**{r['profile']} / seed {r['seed']}**: [[{a['n']},{a['k']},{a['distance']}]], d_X = {a['distance_x']}, d_Z = {a['distance_z']}, w* = {a['w_star']} (degree {a['max_qubit_degree']}), CSS: {_yes(a['is_css'])}, Hadamard frame: {_frame_str(a)}. "
            f"Encoder: {a['circuit']['gates']} gates, depth {a['circuit']['depth']}. Memory p_round (proxy) ion/ion+flag/board/heron = "
            + "/".join(_p(a["memory_proxy"][p]["p_round"]) for p in PROFILE_ORDER)
            + "; encoder undetected mass board/ion(forte)/heron = "
            + "/".join(_p(a["mass"][p]["undetected"]) for p in na.PROFILES)
            + f". Reward terms: memory {rt['memory_term']:+.4f}, LDPC {rt['ldpc_term']:+.3f} (w* excess {rt.get('ldpc_excess', max(0, a['w_star'] - na.W_MAX))}), gate {rt.get('gate_term', 0.0):+.4f}. "
            f"Visits of the selected code: {next((c['visits'] for c in r['codes'] if c['gates'] == a['gates']), '-')}; "
            f"distinct target codes {r['distinct_success_codes']}, of which {sum(c['is_css'] for c in r['codes'])} CSS.",
            "",
            "Stabilizers (sparsest basis): `" + "`, `".join(a["sparsest_basis"]) + "`  ",
            "",
        ]
        lo = r.get("lowest_p_round")
        if lo:
            what = (
                "Highest-reward code of the seed (a different code)"
                if r.get("variant") == "run2"
                else "Lowest training p_round of the seed (a different code)"
            )
            L += [
                f"{what}: p_round {_p(lo['reward_terms']['train_p_round'])} ({lo['reward_terms'].get('train_tier', 'proxy')}), w* = {lo['w_star']}, d_X/d_Z = {lo['distance_x']}/{lo['distance_z']}, CSS {_yes(lo['is_css'])}, frame {_frame_str(lo)}, "
                f"2q gates {lo['reward_terms'].get('two_qubit_gates', '-')}, reward {lo['reward']:+.4f} (LDPC {lo['reward_terms']['ldpc_term']:+.3f}, gate {lo['reward_terms'].get('gate_term', 0.0):+.4f}).",
                "",
            ]
        mo_ = r.get("modal")
        if mo_:
            L += [
                f"Modal code of the last 20 updates: {mo_['visits_last_20_updates']} of {mo_['successes_last_20_updates']} successes, same as selected: {_yes(mo_.get('same_as_best'))}; "
                f"p_round {_p(mo_['reward_terms']['train_p_round'])}, w* = {mo_['w_star']}, d_X/d_Z = {mo_['distance_x']}/{mo_['distance_z']}, CSS {_yes(mo_['is_css'])}, frame {_frame_str(mo_)}, 2q gates {mo_['reward_terms'].get('two_qubit_gates', '-')}.",
                "",
            ]
    if scores:
        meta = scores.get("meta", {})
        has_sampled = any("sampled" in v for row in scores["scores"].values() for v in row.values())
        L += [
            f"{h} Cross-evaluation",
            "",
            f"Rows: training cell (selected code of each seed), the alternate codes, the encoder-objective cells of `results/gpu_scale` (their best codes, re-scored with the memory objective), and the known codes; columns: evaluation profile. "
            f"Entries: memory p_round (both bases summed, `R = {ROUNDS}`), mean over codes with [min, max]. Smaller is better. bposd tier: the pinned BP+OSD decoder, "
            f"{meta.get('shots')} initial shots grown to {meta.get('min_failures')} failures or {meta.get('max_shots')} shots per basis.",
            "",
            "bposd tier (validation):",
            "",
            *matrix_table(scores, "bposd"),
            "",
            "Proxy tier:",
            "",
            *matrix_table(scores, "proxy"),
            "",
        ]
        if has_sampled:
            L += [
                "Sampled tier (weight-2 lookup decoder, near maximum likelihood):",
                "",
                *matrix_table(scores, "sampled"),
                "",
            ]
        L += [
            "Per code, bposd tier:",
            "",
            *per_code_table(scores, "bposd"),
            "",
            "Per code, proxy tier:",
            "",
            *per_code_table(scores, "proxy"),
            "",
            "Tier agreement over every scored code (ratio is the second tier over the first):",
            "",
            *tier_agreement(scores),
            "",
            f"{h} Comparison under the training profile (bposd)",
            "",
            *comparison_lines(recs, scores, name_prefix, label),
            "",
        ]
    if figs:
        L += [f"{h} Figures", ""]
        if f"{fig_prefix}curves.png" in figs:
            L += [
                f"`figures/{fig_prefix}curves.png`: success rate among the episodes finished in an update, moving average over {FIGURE_SMOOTH} updates (top), and the lowest training p_round seen so far at the tier the reward used (bottom, log scale shared across cells; dashed: rotated surface, dotted: XZZX under the cell's profile at the proxy tier) against env steps, one column per cell, seeds as series.",
                "",
            ]
        if f"{fig_prefix}matrix.png" in figs:
            L += [
                f"`figures/{fig_prefix}matrix.png`: bposd p_round per evaluation profile (one panel each), bars are the mean over codes of a row of the cross-evaluation table with min to max whiskers (yellow: run-2 cells, blue: run-1 memory-trained cells, light blue: their lowest-proxy codes, red: encoder-trained cells, grey: known codes).",
                "",
            ]
    return L


def write_report(
    out_dir: Path = DEFAULT_OUT,
    report_dir: Path = REPORT_DIR,
    encoder_dir: Path = ENCODER_DIR,
    run2_dir: Path | None = None,
) -> Path:
    recs = load_runs(out_dir)
    spath = out_dir / "scores.json"
    scores = json.loads(spath.read_text()) if spath.exists() else None
    report_dir.mkdir(parents=True, exist_ok=True)
    figs = make_figures(recs, scores, report_dir / "figures")
    verdict_path = report_dir / "verdict.md"
    run2_dir = RUN2_DIR if run2_dir is None and out_dir == DEFAULT_OUT else run2_dir
    recs2 = load_runs(run2_dir) if run2_dir is not None and run2_dir.exists() else []
    L = [
        "# PPO with the memory objective on the 4x4 setting",
        "",
        f"Status: {len(recs)} runs of run 1 and {len(recs2)} runs of run 2 recorded. Every number below is read from a JSON under `results/memory_ppo/` "
        f"(`run_*.json`, `scores.json`, `run2/`) or `results/gpu_scale/run_noise4x4_*.json`; the tables regenerate with "
        "`python -m autoqec_sim.experiments.memory_ppo report`. Code: `src/autoqec_sim/experiments/memory_ppo.py` "
        "(module docstring: reward, profiles, scales, selection); objective: `src/autoqec_sim/memory_objective.py`, `docs/memory_objective.md`.",
        "",
        "# Run 1: memory term with the LDPC penalty, proxy tier, highest-reward selection",
        "",
    ]
    if verdict_path.exists():
        L += [verdict_path.read_text().rstrip(), ""]
    L += [
        "## Setting",
        "",
        f"`n = {SETTING.n}`, `k = {SETTING.k}`, target `d = {SETTING.d}`, `{SETTING.graph_spec}`, gates {', '.join(SETTING.gates)}, {SETTING.max_steps}-gate episodes, "
        f'equivariant policy (`PPOConfig(policy="auto")`), batch {BATCH}, budget {STEPS:,} env steps or {WALL_MIN:.0f} min per run, seeds {", ".join(map(str, SEEDS))}. '
        f"Reward: `{reward_description()['form']}`. Memory objective: `R = {ROUNDS}` rounds, `{CONNECTIVITY}` layout, hook-aware order, both bases summed, proxy tier in training; "
        "the memory term is evaluated once per distinct stabilizer group (cache keyed by the reduced row echelon form) in a host process pool, synchronously with the training loop.",
        "",
        "| cell | profile | scale | flags | objective |",
        "|---|---|---|---|---|",
    ]
    for p in PROFILE_ORDER:
        d = make_objective(p).describe()
        L.append(
            f"| {p} | {d['profile']} ({d['timing']}) | {d['scale']} | {d['flags']} | rounds {d['rounds']}, {d['connectivity']}, order {d['order']}, bases {'+'.join(d['bases'])} |"
        )
    L += [
        "",
        "Scales: the tiers of the memory objective need the expected number of faults per shot below about one; the benchmark scales of `docs/memory_objective.md` "
        "(board 1, Heron 0.1) put the rotated surface `[[16,1,4]]` at 2.02 and 0.91 expected faults per shot in the `R = 4` circuit and the dense encoder-objective codes at 5.8 and 2.8, "
        "so board runs at 0.25 and Heron at 0.05 (surface 0.50 and 0.45, dense code 1.45 and 1.39; chain 0.25 gives 0.65 and 1.11). The expected fault counts of every scored circuit are in `scores.json` (`per_basis.*.expected_faults`).",
        "",
    ]
    L += _sections(recs, scores, figs)
    if recs2:
        spath2 = run2_dir / "scores.json"
        scores2 = json.loads(spath2.read_text()) if spath2.exists() else None
        figs2 = make_figures(recs2, scores2, report_dir / "figures", prefix="run2_")
        v2 = report_dir / "verdict_run2.md"
        rd = reward_description("run2")
        cells2 = sorted({r["profile"] for r in recs2}, key=PROFILE_ORDER.index)
        L += [
            "# Run 2: proxy selection inside its valid regime, no LDPC term, gate-count term",
            "",
        ]
        if v2.exists():
            L += [v2.read_text().rstrip(), ""]
        L += [
            "## Setting",
            "",
            f"Cells {', '.join(cells2)} (same profiles and scales as run 1), seeds {', '.join(str(x) for x in sorted({r['seed'] for r in recs2}))}, "
            f"wall cap {recs2[0]['wall_cap_s'] / 60:.0f} min per run, budget {recs2[0]['budget_env_steps']:,} env steps, batch {recs2[0]['batch']}; everything else as in run 1. "
            f"Reward: `{rd['form']}`. The LDPC term is removed (w* is reported as a statistic); the gate-count term penalises memory circuits above {GATE_REF} two-qubit gates per round "
            f"(1.25 times the surface code's {SURFACE_2Q_GATES}) with weight {GATE_WEIGHT}; a candidate whose expected faults per shot exceed {SAMPLED_ABOVE} in either basis is evaluated at the sampled tier "
            f"(weight-2 lookup decoder, {TRAIN_SAMPLED['shots']} shots grown to {TRAIN_SAMPLED['min_failures']} failures or {TRAIN_SAMPLED['max_shots']} shots per basis) instead of the proxy, cached per code. "
            "Selection: the target-distance code with the lowest training p_round (at the tier the reward used); the highest-reward code is reported as the alternate.",
            "",
        ]
        L += _sections(recs2, scores2, figs2, "run2_", "mem2_", "run-2 memory-trained", "##")
    path = report_dir / "REPORT.md"
    path.write_text("\n".join(L).rstrip() + "\n")
    print("wrote", path)
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one (profile, seed) cell")
    r.add_argument("--profile", choices=PROFILE_ORDER, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--batch", type=int, default=BATCH)
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--minutes", type=float, default=WALL_MIN)
    r.add_argument("--threads", type=int, default=THREADS)
    r.add_argument("--pool", type=int, default=POOL, help="memory-term worker processes (0: inline)")
    r.add_argument("--chunk", type=int, default=None)
    r.add_argument("--scale", type=float, default=None, help="override the cell's profile scale")
    r.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    r.add_argument(
        "--variant", choices=tuple(VARIANTS), default="run1", help="reward terms and selection rule (VARIANTS)"
    )
    r.add_argument("--verbose", action="store_true")
    s = sub.add_parser(
        "score", help="proxy and bposd scores of the run, encoder-objective, and known codes under the four profiles"
    )
    s.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    s.add_argument("--encoder-dir", type=Path, default=ENCODER_DIR)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--tiers", nargs="+", default=["proxy", "bposd"])
    s.add_argument(
        "--which",
        nargs="+",
        default=["runs", "lowest", "encoder", "known"],
        help="code groups: runs (best code per run), lowest (lowest training p_round per run), encoder, known",
    )
    s.add_argument(
        "--run1-dir",
        type=Path,
        default=None,
        help="with `run1` in --which: the run-1 directory whose best and lowest-proxy codes are scored too",
    )
    s.add_argument(
        "--runs-group",
        default="memory",
        choices=("memory", "memory2"),
        help="group of the `runs` codes (memory2 for a run-2 directory)",
    )
    s.add_argument(
        "--sparsify",
        action="store_true",
        help="replace every code's generators by the sparsest basis before scoring (run 2)",
    )
    s.add_argument(
        "--skip-existing",
        action="store_true",
        help="do not recompute (code, profile, tier) entries already in scores.json",
    )
    s.add_argument("--profiles", nargs="+", default=list(PROFILE_ORDER))
    s.add_argument("--shots", type=int, default=SCORE_SHOTS)
    s.add_argument("--min-failures", type=int, default=SCORE_MIN_FAILURES)
    s.add_argument("--max-shots", type=int, default=SCORE_MAX_SHOTS)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--verbose", action="store_true")
    q = sub.add_parser("report", help="write experiments/memory_ppo/REPORT.md and the figures")
    q.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    q.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    q.add_argument("--encoder-dir", type=Path, default=ENCODER_DIR)
    q.add_argument("--run2-dir", type=Path, default=None)
    a = p.parse_args(argv)
    if a.cmd == "run":
        run_one(
            a.profile,
            a.seed,
            a.out_dir,
            device=a.device,
            batch=a.batch,
            steps=a.steps,
            wall_min=a.minutes,
            threads=a.threads,
            pool_workers=a.pool,
            chunk=a.chunk,
            variant=a.variant,
            scale=a.scale,
            verbose=a.verbose,
        )
    elif a.cmd == "score":
        run_score(
            a.out_dir,
            a.encoder_dir,
            a.workers,
            tuple(a.tiers),
            a.shots,
            a.min_failures,
            a.max_shots,
            a.seed,
            tuple(a.which),
            tuple(a.profiles),
            a.verbose,
            a.run1_dir,
            a.runs_group,
            a.sparsify,
            a.skip_existing,
        )
    elif a.cmd == "report":
        write_report(a.out_dir, a.report_dir, a.encoder_dir, a.run2_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
