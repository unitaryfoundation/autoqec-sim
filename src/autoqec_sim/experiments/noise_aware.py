# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Noise-aware code discovery with PPO under three noise profiles.

Question (the experiment the proposal cites Olle et al., arXiv:2311.04750,
for): does the code an RL agent discovers depend on the noise model it is
trained against, and in the direction the model's structure predicts?

Design
------
Three profiles from ``autoqec_sim.noise.presets``, each at a scale that gives
the same mean total fault mass on a calibration set of random circuits of
the setting (the board preset at scale 1 is the anchor):

- ``board``: ``presets.board()``, one depolarizing rate on every two-qubit
  gate and every idle qubit per layer, no single-qubit-gate channel,
  parallel timing (unbiased);
- ``ion``: ``presets.forte("sequential")``, IonQ Forte rates with every
  two-qubit gate in its own time slot, so idle dephasing (``T2 = 1 s``
  against ``T1 = 10 s``, ``pz : px : py = 20 : 1 : 1``) is charged to
  every other qubit for 900 us per gate, all-to-all connectivity;
- ``heron``: ``presets.heron(graph=None)``, IBM Heron r2 rates; the
  preset's gate bias is depolarizing (``bias1 = bias2 = 1/2``, no public
  Pauli decomposition) and its idle is the ``T1, T2`` channel (``T2 = 120
  us``, ``T1 = 213 us``, so the idle term is mildly Z-biased, ``eta =
  1.8``) but small (68 ns gates). The preset's heavy-hex graph is
  replaced by the setting's graph, since the three settings below fix the
  connectivity; ``heron`` therefore differs from ``board`` in its rate
  structure, not in its connectivity.

Three settings, all ``k = 1``: 7 qubits all-to-all with gates H, S, CNOT
and target distance 3; the 3x3 square lattice with H, CNOT and target
distance 3; the 4x4 square lattice with H, CNOT and target distance 4.

Reward. The state reward of a tableau ``s`` produced by the gate list
``c`` is

    r(s) = min(d(s), d*) / d* - 1 - q_1(s)
           + [d(s) >= d*, q_1(s) = 0] ( - 2 m(c) / M_0 - 0.25 max(0, w*(s) - 4) ),

where ``q_1(s)`` is the fraction of weight-1 Paulis that lie in the
stabilizer group (:func:`autoqec_sim.reward.low_weight_stabilizer_penalty`
with ``max_weight = 1``, the padding penalty of the rediscovery
regression: without it the agent embeds a smaller code and leaves qubits
idle, which the circuit-level objective rewards since an idle ``|0>`` qubit
contributes only trivially detected faults; see
``results/noise_aware/pilot/no_padding_penalty/``), ``d(s)`` is the exact
distance computed up to ``d* - 1``
(:func:`autoqec_sim.reward.distance_reward`, shifted by -1 so that every
non-target state has a negative dense reward and termination is never
disfavoured), ``m(c)`` is the circuit-level undetected fault mass of the
encoding circuit under the training profile
(:func:`autoqec_sim.circuit_noise.undetected_fault_mass`, ``normalize =
False``, absolute probability), ``M_0`` is the calibrated common total
fault mass of the setting (so ``m / M_0`` is the undetected fraction of a
typical circuit's fault budget and the weight 2 makes a typical value of
0.1 cost 0.2, under one distance quantum), and ``w*(s)`` is the exact
minimum bottleneck generator weight of the stabilizer group
(:func:`autoqec_sim.ldpc.ldpc_penalty` with ``w_max = 4``, scale 0.25).
The noise and LDPC terms are evaluated only on states that reached the
target distance with no idle qubit; such a state ends the episode (the
``success_fn`` of the environment), as in the rediscovery regression. The reward is dense
(state reward after every gate).

Per (profile, setting) five PPO seeds are trained at a fixed step budget.
The best code of a seed is the distinct terminal state with the highest
reward; if no state reached the target distance, the best is chosen among
the terminal states with the largest distance by the smallest undetected
mass. Every best code is replayed and analysed: exact ``d``, the pure-X
and pure-Z distances, CSS test and Hadamard-frame search (the smallest
qubit subset ``A`` such that ``H`` on ``A`` makes the code CSS; the full
local-Clifford frame search for ``n <= 8``), ``w*`` and qubit degree,
weight enumerators of ``S`` and ``N(S)``, X against Z component counts
over the whole stabilizer group, and the undetected fault mass of its
encoding circuit under all three profiles (the cross-evaluation matrix).

Known codes at the same ``(n, k)`` are built here (Steane; rotated
surface, XZZX, Hadamard-deformed, Shor, transposed Shor at ``n = 9`` and
``n = 16``) and get an encoding circuit on the setting's graph from a
PPO search whose reward is the dimension of the intersection of the
current stabilizer group with the target group; among the distinct
encoders found, the one with the smallest undetected mass under each
profile is the baseline for that profile.

Run one cell (setting, profile, seed)::

    uv run --extra rl python -m autoqec_sim.experiments.noise_aware run \\
        --setting 3x3 --profile ion --seed 0 --threads 2

then ``baselines --setting 3x3``, and ``report`` to write
``results/noise_aware/REPORT.md`` and the figures.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


def _threads_from_argv(argv) -> int:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return 6


_THREADS = _threads_from_argv(sys.argv)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import numpy as np

from autoqec_sim.noise import Profile, presets
from autoqec_sim.noise.skeleton import ANNOTATIONS

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
from ..ldpc import (
    batched_rank_packed,
    group_elements_packed,
    ldpc_penalty,
    pack_rows,
    sparsest_basis_stats,
    unpack_rows,
)
from ..regress import replay
from ..reward import batched_distance, low_weight_stabilizer_penalty, undetected_mask
from ..symplectic import pauli_to_vec, rref, vec_to_pauli

W_MAX = 4
LDPC_SCALE = 0.25
NOISE_WEIGHT = 2.0
EXACT_GENERATORS = 15  # exact w* up to n - k = 15 (the 4x4 setting)
CALIBRATION_CIRCUITS = 64
CALIBRATION_SEED = 0
BATCH = 256
DEFAULT_OUT = Path("results/noise_aware")


# --------------------------------------------------------------------------
# settings and profiles
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Setting:
    name: str
    n: int
    k: int
    d: int
    graph_spec: str
    gates: tuple
    max_steps: int
    budget: int  # PPO env steps (state evaluations) per seed, chosen so a run fits 25 min
    # distance term: False gives min(d, d*)/d* (three or four levels); True adds,
    # below the target, the detected fraction of the Paulis of weight d, so
    # that the term is (d - 1 + f)/d* and increases continuously toward the
    # next level. The 4x4 setting needs it: with the level reward PPO stayed
    # at d = 1 to 2 for 32.5M steps (25 min) in the pilot.
    fractional: bool = False

    def to_dict(self):
        d = asdict(self)
        d["gates"] = list(self.gates)
        return d


SETTINGS = {
    "n7": Setting("n7", 7, 1, 3, "all_to_all:7", ("H", "S", "CNOT"), 20, 6_000_000),
    "3x3": Setting("3x3", 9, 1, 3, "square:3x3", ("H", "CNOT"), 30, 8_000_000),
    "4x4": Setting("4x4", 16, 1, 4, "square:4x4", ("H", "CNOT"), 50, 16_000_000, fractional=True),
}

PROFILE_FACTORIES = {
    "board": lambda: presets.board(),
    "ion": lambda: presets.forte("sequential"),
    "heron": lambda: presets.heron(graph=None),
}
PROFILES = tuple(PROFILE_FACTORIES)


def profile(name: str) -> Profile:
    return PROFILE_FACTORIES[name]()


def profile_description(name: str) -> str:
    p = profile(name)
    return f"{p.name} [{p.timing.name}]"


# --------------------------------------------------------------------------
# calibration: equal total fault mass
# --------------------------------------------------------------------------
def calibration_circuits(setting: Setting, count: int = CALIBRATION_CIRCUITS, seed: int = CALIBRATION_SEED):
    """``count`` random gate lists of ``setting.max_steps`` gates, uniform
    over the action table."""
    env = CodeDiscoveryEnv(
        n=setting.n, k=setting.k, graph=graphs.from_spec(setting.graph_spec), batch_size=count,
        gates=setting.gates, max_steps=setting.max_steps, seed=seed,
    )
    env.reset()
    for _ in range(setting.max_steps):
        env.step(env.sample_actions())
    return env.gate_lists(), env.graph


def total_mass(gates, n: int, model) -> float:
    return fault_set(model(gates_to_layered_circuit(gates, n)), n).total


def calibrate(setting: Setting, count: int = CALIBRATION_CIRCUITS, seed: int = CALIBRATION_SEED) -> dict:
    """Per profile: the mean total fault mass at scale 1 on the calibration
    circuits and the scale that equalizes it to the board's. ``m0`` is the
    common mass."""
    circuits, graph = calibration_circuits(setting, count, seed)
    out = {"setting": setting.name, "count": count, "seed": seed, "profiles": {}}
    masses = {}
    for name in PROFILES:
        model = resolve_noise_model(profile(name), scale=1.0, graph=graph, n=setting.n)
        m = np.array([total_mass(g, setting.n, model) for g in circuits])
        masses[name] = m
    m0 = float(masses["board"].mean())
    out["m0"] = m0
    for name in PROFILES:
        mean = float(masses[name].mean())
        out["profiles"][name] = {
            "mass_scale1_mean": mean,
            "mass_scale1_std": float(masses[name].std()),
            "scale": m0 / mean,
            "description": profile_description(name),
        }
    return out


def calibrated_models(setting: Setting, calib: dict, graph) -> dict:
    return {
        name: resolve_noise_model(profile(name), scale=calib["profiles"][name]["scale"], graph=graph, n=setting.n)
        for name in PROFILES
    }


# --------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------
class NoiseAwareReward:
    """The composite reward of the module docstring, bound to an environment
    so that it can read the gate list behind every terminal state."""

    def __init__(
        self,
        env: CodeDiscoveryEnv,
        setting: Setting,
        model,
        m0: float,
        noise_weight: float = NOISE_WEIGHT,
        w_max: int = W_MAX,
        ldpc_scale: float = LDPC_SCALE,
    ):
        self.env = env
        self.d = setting.d
        self.n = setting.n
        self.fractional = bool(setting.fractional)
        self.model = model
        self.m0 = float(m0)
        self.noise_weight = float(noise_weight)
        self.w_max = int(w_max)
        self.ldpc_scale = float(ldpc_scale)
        self.cache: dict[tuple, float] = {}
        self.noise_evals = 0
        self.cache_hits = 0
        self.last_success = None

    def mass(self, gates) -> float:
        key = tuple(gates)
        m = self.cache.get(key)
        if m is None:
            faults = fault_set(self.model(gates_to_layered_circuit(gates, self.n)), self.n)
            tab = replay(gates, self.n).to_tableau()
            m = undetected_fault_mass(tab, 1, faults)
            self.cache[key] = m
            self.noise_evals += 1
        else:
            self.cache_hits += 1
        return m

    def terminal_terms(self, tab_sel: np.ndarray, gate_lists) -> tuple[np.ndarray, np.ndarray]:
        """``(noise term, ldpc term)`` for the selected tableaux."""
        m = np.array([self.mass(g) for g in gate_lists], dtype=float)
        noise = -self.noise_weight * m / self.m0
        ldpc = ldpc_penalty(tab_sel, 1, self.w_max, scale=self.ldpc_scale, exact_max_generators=EXACT_GENERATORS)
        return noise, ldpc

    def distance_term(self, tab: np.ndarray, k: int, d: np.ndarray) -> np.ndarray:
        """``min(d, d*)/d* - 1``, plus, when ``fractional``, the detected
        fraction ``f`` of the weight-``d`` Paulis for states below the
        target (``(d - 1 + f)/d* - 1``)."""
        r = np.minimum(d, self.d) / float(self.d) - 1.0
        if not self.fractional:
            return r
        from ..errors import paulis_of_weight

        for w in range(1, self.d):
            idx = np.flatnonzero(d == w)
            if idx.size == 0:
                continue
            e = paulis_of_weight(self.n, w)
            frac = 1.0 - undetected_mask(tab[idx], k, e).mean(axis=1)
            r[idx] = (w - 1 + frac) / float(self.d) - 1.0
        return r

    def __call__(self, tab: np.ndarray, k: int) -> np.ndarray:
        d = batched_distance(tab, k, self.d - 1)
        pad = low_weight_stabilizer_penalty(tab, k, 1)  # -(fraction of weight-1 Paulis in S)
        r = self.distance_term(tab, k, d) + pad
        succ = (d >= self.d) & (pad == 0.0)
        if succ.any() and tab.shape[0] == self.env.batch_size:
            idx = np.flatnonzero(succ)
            noise, ldpc = self.terminal_terms(tab[idx], self.env.gate_lists(succ))
            r[idx] += noise + ldpc
        self.last_success = succ
        return r

    def success_fn(self, tab, k, r):
        if self.last_success is not None and self.last_success.shape == r.shape:
            return self.last_success
        return r >= 0.0


def code_key(tab_b: np.ndarray, k: int) -> bytes:
    n = (tab_b.shape[-1] - 1) // 2
    red, _ = rref(tab_b[n + k :, : 2 * n])
    return red.tobytes()


class CandidateTracker:
    """Distinct terminal codes of a run: every state that reached the
    target distance with its reward, and (capped) the states of the
    largest distance seen when none did."""

    def __init__(self, d_target: int, cap: int = 512, fractional: bool = False):
        self.d_target = d_target
        self.fractional = fractional
        self.cap = cap
        self.success: dict[bytes, dict] = {}
        self.partial: dict[bytes, dict] = {}
        self.partial_d = 0
        self.episodes = 0
        self.success_episodes = 0
        self.current_counts: dict[bytes, int] = {}  # visits per code since the last snapshot
        self.windows: list[dict[bytes, int]] = []  # one snapshot per PPO update

    def record(self, info, done, table, d_final):
        """``d_final`` is the exact distance (capped at ``d_target - 1``) of
        every finished episode's terminal tableau, computed by the caller
        from ``info["final_tableau"]`` (the reward object's own distance
        array cannot be reused: ``reset(where=...)`` re-evaluates the reward
        on the reset elements, and when every episode truncates at the same
        step that sub-batch is the whole batch)."""
        tabs = info["final_tableau"]
        hist = info["final_history"]
        rew = info["final_state_reward"]
        succ = info["success"][done]
        dl = d_final
        self.episodes += tabs.shape[0]
        for i in range(tabs.shape[0]):
            key = code_key(tabs[i], 1)
            if succ[i]:
                self.success_episodes += 1
                self.current_counts[key] = self.current_counts.get(key, 0) + 1
                cur = self.success.get(key)
                if cur is None or rew[i] > cur["reward"]:
                    acts = [int(a) for a in hist[i] if a >= 0]
                    self.success[key] = {"reward": float(rew[i]), "gates": [list(table.entries[a]) for a in acts], "d": self.d_target}
                continue
            if self.success:
                continue
            d = min(int(dl[i]), self.d_target - 1)
            if d > self.partial_d:
                self.partial = {}
                self.partial_d = d
            if d == self.partial_d and key not in self.partial and len(self.partial) < self.cap:
                acts = [int(a) for a in hist[i] if a >= 0]
                self.partial[key] = {"reward": float(rew[i]), "gates": [list(table.entries[a]) for a in acts], "d": d}

    def best_reward(self):
        if not self.success:
            return None
        return max(v["reward"] for v in self.success.values())

    def snapshot(self):
        """Close the current window of visit counts (called once per update)."""
        self.windows.append(self.current_counts)
        self.current_counts = {}

    def modal_code(self, last_updates: int = 20):
        """The most visited target-distance code over the last ``last_updates``
        windows (the code the converged policy produces most often), with its
        visit count and the number of visits in the window."""
        counts: dict[bytes, int] = {}
        for w in self.windows[-last_updates:]:
            for key, c in w.items():
                counts[key] = counts.get(key, 0) + c
        if not counts:
            return None, 0, 0
        key = max(counts, key=counts.get)
        return self.success[key], counts[key], sum(counts.values())


class TrackingEnv(CodeDiscoveryEnv):
    """Environment that hands every finished episode to a tracker."""

    tracker: CandidateTracker | None = None

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        if self.tracker is not None and "final_tableau" in info:
            d_final = batched_distance(info["final_tableau"], self.k, self.tracker.d_target - 1)
            self.tracker.record(info, terminated | truncated, self.action_table, d_final)
        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------------
# code analysis
# --------------------------------------------------------------------------
def pure_type_distance(tab_b: np.ndarray, k: int, kind: str) -> tuple[int | None, int]:
    """Minimum weight of an undetected pure-X (``kind="X"``) or pure-Z
    Pauli, and the number of such minimum-weight Paulis; ``None`` if no
    pure-type Pauli is undetected (no pure-type logical operator)."""
    n = (tab_b.shape[-1] - 1) // 2
    bits = ((np.arange(1, 1 << n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    e = np.zeros((bits.shape[0], 2 * n), dtype=np.uint8)
    if kind == "X":
        e[:, :n] = bits
    else:
        e[:, n:] = bits
    und = undetected_mask(tab_b, k, e)[0]
    if not und.any():
        return None, 0
    w = bits.sum(axis=1)[und]
    return int(w.min()), int((w == w.min()).sum())


# GL(2, 2): the six single-qubit local Cliffords as maps (x, z) -> (x', z')
# with x' = a x + b z, z' = c x + d z. Index 0 is the identity, 1 is H.
LC_MATRICES = [
    ((1, 0), (0, 1)),  # I
    ((0, 1), (1, 0)),  # H
    ((1, 0), (1, 1)),  # S (X -> Y)
    ((1, 1), (0, 1)),  # H S H (Z -> Y)
    ((0, 1), (1, 1)),  # S H
    ((1, 1), (1, 0)),  # H S
]


def css_frames(code: StabilizerCode, full_lc: bool | None = None, max_frames: int = 64) -> dict:
    """Local-Clifford frames that make the stabilizer group CSS.

    With ``full_lc`` false only Hadamard frames are tried (``2^n`` subsets
    ``A``); with true all ``6^n`` products of single-qubit Cliffords
    (default: true for ``n <= 8``). Returns the smallest frame (by the
    number of qubits acted on nontrivially), the number of frames found,
    and, for the smallest frame, the transformed stabilizers.
    """
    n, m = code.n, code.stabilizers.shape[0]
    if full_lc is None:
        full_lc = n <= 8
    s = code.stabilizers
    x = pack_rows(s[:, :n])  # (m,) n-bit
    z = pack_rows(s[:, n:])
    if full_lc:
        base = 6
        count = base**n
        digits = (np.arange(count)[:, None] // (base ** np.arange(n))) % base  # (count, n)
    else:
        base = 2
        count = 1 << n
        digits = (np.arange(count)[:, None] >> np.arange(n)) & 1
    mats = np.array(LC_MATRICES, dtype=np.uint64)  # (6, 2, 2)
    xp = np.zeros((count, m), dtype=np.uint64)
    zp = np.zeros((count, m), dtype=np.uint64)
    for q in range(n):
        xq = ((x >> np.uint64(q)) & np.uint64(1))[None, :]  # (1, m)
        zq = ((z >> np.uint64(q)) & np.uint64(1))[None, :]
        mq = mats[digits[:, q]]  # (count, 2, 2)
        a, b = mq[:, 0, 0][:, None], mq[:, 0, 1][:, None]
        c, d = mq[:, 1, 0][:, None], mq[:, 1, 1][:, None]
        xp |= ((a & xq) ^ (b & zq)) << np.uint64(q)
        zp |= ((c & xq) ^ (d & zq)) << np.uint64(q)
    ones = np.ones((count, m), dtype=bool)
    rx, _ = batched_rank_packed(xp, ones, n)
    rz, _ = batched_rank_packed(zp, ones, n)
    css = (rx + rz) == m
    found = np.flatnonzero(css)
    if found.size == 0:
        return {"css_frame_found": False, "frames": 0, "search": "local Clifford" if full_lc else "Hadamard"}
    size = (digits[found] != 0).sum(axis=1)
    order = np.lexsort((found, size))
    best = found[order[0]]
    frame = [(q, int(digits[best, q])) for q in range(n) if digits[best, q]]
    names = {1: "H", 2: "S", 3: "HSH", 4: "SH", 5: "HS"}
    trans = np.concatenate([unpack_rows(xp[best], n), unpack_rows(zp[best], n)], axis=1)
    x_type = trans[trans[:, n:].sum(axis=1) == 0]
    return {
        "css_frame_found": True,
        "frames": int(found.size),
        "search": "local Clifford" if full_lc else "Hadamard",
        "frame_size": int(size[order[0]]),
        "frame": [[q, names[g]] for q, g in frame],
        "hadamard_only": all(g == 1 for _, g in frame),
        "frame_stabilizers": [vec_to_pauli(r) for r in trans],
        "frame_x_type_generators_in_basis": int(x_type.shape[0]),
        "frame_examples": [
            [[q, names[int(digits[f, q])]] for q in range(n) if digits[f, q]] for f in found[order[:max_frames]]
        ],
    }


def group_component_counts(code: StabilizerCode) -> dict:
    """Over all ``2^m`` stabilizer group elements: the number of pure-X and
    pure-Z elements (sizes of the X-type and Z-type subgroups minus the
    identity) and their weight distributions. Basis independent, and in
    the physical frame (a Hadamard on a qubit moves elements between the
    two counts). The total X and Z component counts over the group are not
    reported: a nonzero linear functional is 1 on exactly half the group,
    so both equal ``2^m n / 2`` for every code whose stabilizers touch
    every qubit in both bases, and carry no information."""
    n = code.n
    gens = pack_rows(code.stabilizers)[None, :]
    elems = group_elements_packed(gens)[0]  # (2^m,)
    mask = np.uint64((1 << n) - 1)
    xpart = elems & mask
    zpart = (elems >> np.uint64(n)) & mask
    xw = np.bitwise_count(xpart).astype(np.int64)
    zw = np.bitwise_count(zpart).astype(np.int64)
    pure_x = (zpart == 0) & (xpart != 0)
    pure_z = (xpart == 0) & (zpart != 0)
    return {
        "pure_x_elements": int(pure_x.sum()),
        "pure_z_elements": int(pure_z.sum()),
        "pure_x_weights": np.bincount(xw[pure_x], minlength=n + 1).tolist(),
        "pure_z_weights": np.bincount(zw[pure_z], minlength=n + 1).tolist(),
    }


def mass_breakdown(gates, n: int, model, tab: np.ndarray) -> dict:
    """Undetected and total fault mass split by mechanism class (two-qubit
    gate channel, single-qubit gate channel, idle channel) and by the
    Pauli type of the inserted fault (pure Z, pure X, has a Y or mixed
    two-qubit label). A gate channel is the noise instruction that
    immediately follows a unitary on the same targets; every other noise
    instruction of a segment is idle."""
    import stim

    from ..circuit_noise import _channel_terms, _is_noise

    noisy = model(gates_to_layered_circuit(gates, n))
    faults = fault_set(noisy, n)
    und = undetected_mask(tab, 1, faults.paulis)[0] & ~faults.heralded
    insts = list(noisy.flattened())
    classes, labels = [], []
    prev = None
    for i, inst in enumerate(insts):
        if inst.name in ANNOTATIONS:
            continue
        if not _is_noise(inst):
            prev = inst
            continue
        targets = tuple(t.value for t in inst.targets_copy())
        gate_channel = (
            prev is not None
            and stim.gate_data(prev.name).is_unitary
            and tuple(t.value for t in prev.targets_copy()) == targets
        )
        prev = None
        if gate_channel:
            cls = "two-qubit gate" if stim.gate_data(inst.name).is_two_qubit_gate else "single-qubit gate"
        else:
            cls = "idle"
        for _, lab, _, _ in _channel_terms(inst, 1.0):
            classes.append(cls)
            core = lab.replace("I", "")
            labels.append("Z" if set(core) == {"Z"} else "X" if set(core) == {"X"} else "Y or mixed")
    assert len(classes) == len(faults), (len(classes), len(faults))
    classes = np.array(classes)
    labels = np.array(labels)
    out = {"total": faults.total, "undetected": float(faults.probs[und].sum()), "by_class": {}, "by_type": {}, "by_class_and_type": {}}
    for c in ("two-qubit gate", "single-qubit gate", "idle"):
        m = classes == c
        out["by_class"][c] = {"total": float(faults.probs[m].sum()), "undetected": float(faults.probs[m & und].sum())}
        for t in ("Z", "X", "Y or mixed"):
            mt = m & (labels == t)
            if mt.any():
                out["by_class_and_type"][f"{c} / {t}"] = {"total": float(faults.probs[mt].sum()), "undetected": float(faults.probs[mt & und].sum())}
    for t in ("Z", "X", "Y or mixed"):
        m = labels == t
        out["by_type"][t] = {"total": float(faults.probs[m].sum()), "undetected": float(faults.probs[m & und].sum())}
    return out


def circuit_stats(gates, n: int) -> dict:
    two = [g for g in gates if g[0] in ("CNOT", "CZ", "SWAP")]
    return {
        "gates": len(gates),
        "two_qubit_gates": len(two),
        "single_qubit_gates": len(gates) - len(two),
        "h_gates": sum(g[0] == "H" for g in gates),
        "s_gates": sum(g[0] in ("S", "S_DAG") for g in gates),
        "depth": circuit_depth(gates, n),
    }


def analyze_circuit(gates, setting: Setting, models: dict) -> dict:
    """Everything the report needs about one encoding circuit and its code."""
    gates = [tuple(g) for g in gates]
    n, k = setting.n, setting.k
    code = replay(gates, n)
    tab = code.to_tableau()
    d = code.distance_exact()
    dx, nx = pure_type_distance(tab, k, "X")
    dz, nz = pure_type_distance(tab, k, "Z")
    st = sparsest_basis_stats(tab, k, exact_max_generators=EXACT_GENERATORS)
    a, b = code.weight_enumerators()
    masses = {}
    for name, model in models.items():
        faults = fault_set(model(gates_to_layered_circuit(gates, n)), n)
        masses[name] = {
            "undetected": undetected_fault_mass(tab, k, faults),
            "total": faults.total,
            "mechanisms": len(faults),
            "breakdown": mass_breakdown(gates, n, model, tab),
        }
    basis = unpack_rows(st["basis"][0], 2 * n)
    return {
        "gates": [list(g) for g in gates],
        "circuit": circuit_stats(gates, n),
        "n": n,
        "k": k,
        "distance": int(d),
        "distance_x": dx,
        "distance_x_count": nx,
        "distance_z": dz,
        "distance_z_count": nz,
        "is_css": bool(code.is_css()),
        "stabilizers": code.stabilizer_strings(),
        "sparsest_basis": [vec_to_pauli(r) for r in basis],
        "w_star": int(st["w_star"][0]),
        "max_qubit_degree": int(st["max_degree"][0]),
        "ldpc_exact": bool(st["exact"]),
        "weight_enumerator_S": a.tolist(),
        "weight_enumerator_N": b.tolist(),
        "components": group_component_counts(code),
        "frame": css_frames(code),
        "mass": masses,
    }


# --------------------------------------------------------------------------
# one PPO run
# --------------------------------------------------------------------------
def make_env(setting: Setting, seed: int, batch: int = BATCH) -> TrackingEnv:
    return TrackingEnv(
        n=setting.n, k=setting.k, graph=graphs.from_spec(setting.graph_spec), batch_size=batch,
        gates=setting.gates, max_steps=setting.max_steps, seed=seed, auto_reset=True,
    )


def run_seed(
    setting: Setting,
    profile_name: str,
    seed: int,
    budget: int | None = None,
    max_time_s: float = 25 * 60,
    threads: int = 6,
    batch: int = BATCH,
    verbose: bool = False,
    calib: dict | None = None,
) -> dict:
    from ..agents.ppo import PPOAgent, PPOConfig

    budget = setting.budget if budget is None else int(budget)
    calib = calib or calibrate(setting)
    env = make_env(setting, seed, batch)
    models = calibrated_models(setting, calib, env.graph)
    reward = NoiseAwareReward(env, setting, models[profile_name], calib["m0"])
    env.reward_fn = reward
    env.success_fn = reward.success_fn
    tracker = CandidateTracker(setting.d, fractional=setting.fractional)
    env.tracker = tracker
    env.reset()
    config = PPOConfig(max_threads=threads)
    agent = PPOAgent(env, config=config, seed=seed)
    t0 = time.perf_counter()

    def cb(agent, new):
        tracker.snapshot()
        rec = agent.history[-1]
        rec["successes_in_update"] = len(new)
        rec["distinct_success_codes"] = len(tracker.success)
        rec["best_terminal_reward"] = tracker.best_reward()
        rec["noise_evals"] = reward.noise_evals
        rec["cache_hits"] = reward.cache_hits
        if verbose and new:
            print(f"    best terminal reward {tracker.best_reward():+.4f} distinct codes {len(tracker.success)}")
        return False

    res = agent.train(budget, max_time_s=max_time_s, stop_on_success=False, callback=cb, verbose=verbose)
    train_time = time.perf_counter() - t0
    hit_time_cap = res.env_steps < budget

    # best code of the seed
    t1 = time.perf_counter()
    if tracker.success:
        best = max(tracker.success.values(), key=lambda v: v["reward"])
        selection = "highest reward among states that reached the target distance"
        ranked = sorted(tracker.success.values(), key=lambda v: -v["reward"])
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
        ranked = cands
    analysis = analyze_circuit(best["gates"], setting, models) if best else None
    modal, modal_visits, window_visits = tracker.modal_code()
    modal_analysis = None
    if modal is not None:
        modal_analysis = analyze_circuit(modal["gates"], setting, models)
        modal_analysis["reward"] = modal["reward"]
        modal_analysis["visits_last_20_updates"] = modal_visits
        modal_analysis["successes_last_20_updates"] = window_visits
    # the reward decomposition of the best code under the training profile
    if analysis is not None:
        m = analysis["mass"][profile_name]["undetected"]
        analysis["reward_terms"] = {
            "distance_term": min(analysis["distance"], setting.d) / setting.d - 1.0,
            "padding_term": -analysis["weight_enumerator_S"][1] / (3 * setting.n),
            "noise_term": -NOISE_WEIGHT * m / calib["m0"],
            "ldpc_term": -LDPC_SCALE * max(0, analysis["w_star"] - W_MAX),
        }
    # a short list of runner-up codes (distinct groups) for the code list
    runners = []
    for v in ranked[1:6]:
        c = replay([tuple(g) for g in v["gates"]], setting.n)
        st = sparsest_basis_stats(c.to_tableau(), 1, exact_max_generators=EXACT_GENERATORS)
        runners.append(
            {
                "reward": v["reward"],
                "d": v["d"],
                "gates": len(v["gates"]),
                "w_star": int(st["w_star"][0]),
                "stabilizers": c.stabilizer_strings(),
                "is_css": bool(c.is_css()),
            }
        )
    return {
        "setting": setting.to_dict(),
        "profile": profile_name,
        "profile_description": profile_description(profile_name),
        "scale": calib["profiles"][profile_name]["scale"],
        "m0": calib["m0"],
        "reward": {
            "form": (
                "(d - 1 + f)/d* - 1 - q_1 + [d >= d*, q_1 = 0](-2 m / M_0 - 0.25 max(0, w* - 4)), f = detected fraction of weight-d Paulis, q_1 = fraction of weight-1 Paulis in S"
                if setting.fractional
                else "min(d, d*)/d* - 1 - q_1 + [d >= d*, q_1 = 0](-2 m / M_0 - 0.25 max(0, w* - 4)), q_1 = fraction of weight-1 Paulis in S"
            ),
            "padding_penalty": "reward.low_weight_stabilizer_penalty(max_weight=1); success requires no weight-1 stabilizer element",
            "fractional_distance": setting.fractional,
            "noise_weight": NOISE_WEIGHT,
            "w_max": W_MAX,
            "ldpc_scale": LDPC_SCALE,
        },
        "seed": seed,
        "budget_env_steps": budget,
        "env_steps": res.env_steps,
        "episodes": res.episodes,
        "updates": res.updates,
        "wall_time_s": round(train_time, 1),
        "analysis_time_s": round(time.perf_counter() - t1, 1),
        "hit_time_cap": bool(hit_time_cap),
        "threads": threads,
        "batch": batch,
        "ppo_config": config.to_dict(),
        "num_parameters": res.num_parameters,
        "successful_episodes": tracker.success_episodes,
        "distinct_success_codes": len(tracker.success),
        "partial_distance": tracker.partial_d if not tracker.success else None,
        "distinct_partial_codes": len(tracker.partial) if not tracker.success else None,
        "noise_evals": reward.noise_evals,
        "cache_hits": reward.cache_hits,
        "first_success_env_steps": res.first_success.env_steps if res.first_success else None,
        "final_success_rate": res.history[-1]["success_rate"] if res.history else None,
        "selection": selection,
        "best": analysis,
        "best_reward": best["reward"] if best and tracker.success else None,
        "modal": modal_analysis,
        "runners_up": runners,
        "history": res.history,
        "platform": platform.platform(),
    }


def write_history_csv(run: dict, path: Path):
    cols = [
        "update", "env_steps", "episodes", "wall_time_s", "mean_reward", "success_rate",
        "successes_in_update", "distinct_success_codes", "best_terminal_reward", "entropy", "approx_kl",
    ]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for rec in run["history"]:
            vals = []
            for c in cols:
                v = rec.get(c)
                vals.append("" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)))
            f.write(",".join(vals) + "\n")


def run_name(setting: str, profile_name: str, seed: int) -> str:
    return f"{setting}_{profile_name}_seed{seed}"


# --------------------------------------------------------------------------
# known codes and their encoders
# --------------------------------------------------------------------------
def _grid_index(rows: int, cols: int):
    return lambda r, c: r * cols + c


def rotated_surface(d: int) -> list[str]:
    """Rotated surface code of distance ``d`` on a ``d x d`` grid (row-major
    indices). Plaquette ``(r, c)`` covers ``{(r, c), (r, c+1), (r+1, c),
    (r+1, c+1)}`` and is X-type iff ``r + c`` is even; boundary weight-2
    checks have the type opposite to the adjacent plaquette. ``d = 3``
    reproduces ``known_codes.ROTATED_SURFACE_9``."""
    n = d * d
    idx = _grid_index(d, d)
    out = []

    def pauli(kind, qubits):
        s = ["I"] * n
        for q in qubits:
            s[q] = kind
        return "".join(s)

    for r in range(d - 1):
        for c in range(d - 1):
            kind = "X" if (r + c) % 2 == 0 else "Z"
            out.append(pauli(kind, [idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)]))
    for c in range(d - 1):
        if (0 + c) % 2 == 1:  # plaquette (0, c) is Z: X boundary on top
            out.append(pauli("X", [idx(0, c), idx(0, c + 1)]))
        if (d - 2 + c) % 2 == 1:  # plaquette (d-2, c) is Z: X boundary at the bottom
            out.append(pauli("X", [idx(d - 1, c), idx(d - 1, c + 1)]))
    for r in range(d - 1):
        if (r + 0) % 2 == 0:  # plaquette (r, 0) is X: Z boundary on the left
            out.append(pauli("Z", [idx(r, 0), idx(r + 1, 0)]))
        if (r + d - 2) % 2 == 0:  # plaquette (r, d-2) is X: Z boundary on the right
            out.append(pauli("Z", [idx(r, d - 1), idx(r + 1, d - 1)]))
    return out


def shor_code(rows: int) -> list[str]:
    """Generalized Shor code on ``rows`` blocks of ``rows`` qubits (the rows
    of the grid): ``ZZ`` on adjacent qubits of a row, ``X`` on two adjacent
    rows. ``rows = 3`` is ``known_codes.SHOR``. ``[[r^2, 1, r]]``,
    ``w* = 2r``."""
    n = rows * rows
    out = []
    for r in range(rows):
        for c in range(rows - 1):
            s = ["I"] * n
            s[r * rows + c] = s[r * rows + c + 1] = "Z"
            out.append("".join(s))
    for r in range(rows - 1):
        s = ["I"] * n
        for q in range(r * rows, (r + 2) * rows):
            s[q] = "X"
        out.append("".join(s))
    return out


def hadamard_conjugate(stabilizers: list[str], qubits) -> list[str]:
    """Swap X and Z on ``qubits`` in every string (Y is fixed)."""
    swap = {"X": "Z", "Z": "X", "Y": "Y", "I": "I"}
    qs = set(qubits)
    return ["".join(swap[ch] if q in qs else ch for q, ch in enumerate(s)) for s in stabilizers]


def known_codes_for(setting: Setting) -> dict[str, dict]:
    """Known codes at the setting's ``(n, k)``: stabilizers, a description,
    and the Hadamard frame ``A`` (qubits conjugated by H) that turns the
    CSS parent code into the listed one."""
    n = setting.n
    if n == 7:
        from ..known_codes import STEANE

        return {"steane": {"stabilizers": STEANE, "description": "Steane [[7,1,3]], CSS, self-dual", "frame": []}}
    d = round(n**0.5)
    idx = _grid_index(d, d)
    checker = sorted(idx(r, c) for r in range(d) for c in range(d) if (r + c) % 2 == 1)
    outer_cols = sorted(idx(r, c) for r in range(d) for c in (0, d - 1))
    rs = rotated_surface(d)
    sh = shor_code(d)
    return {
        "rotated_surface": {"stabilizers": rs, "description": f"rotated surface [[{n},1,{d}]]", "frame": []},
        "xzzx": {
            "stabilizers": hadamard_conjugate(rs, checker),
            "description": f"XZZX surface [[{n},1,{d}]]: H on the checkerboard {checker}",
            "frame": checker,
        },
        "surface_h_outer_cols": {
            "stabilizers": hadamard_conjugate(rs, outer_cols),
            "description": (
                f"rotated surface with H on the outer columns A={outer_cols} "
                "(the best Forte deformation of ion-search/deform at d = 3)"
            ),
            "frame": outer_cols,
        },
        "shor": {"stabilizers": sh, "description": f"Shor [[{n},1,{d}]]: ZZ along rows, X on row pairs (w* = {2 * d})", "frame": []},
        "transposed_shor": {
            "stabilizers": hadamard_conjugate(sh, range(n)),
            "description": f"transposed Shor [[{n},1,{d}]]: XX along rows, Z on row pairs (ion-search shor:3x3T at d = 3)",
            "frame": list(range(n)),
        },
    }


def _rref_with_order(m: np.ndarray, order):
    """GF(2) row reduction of ``m`` choosing pivot columns in the given
    column preference ``order``. Returns ``(reduced rows, pivot columns)``
    with the rows in the original column indexing; every pivot column is
    zero in all other rows."""
    a = np.array(m, dtype=np.uint8) % 2
    rows, _ = a.shape
    pivots = []
    r = 0
    for c in order:
        if r >= rows:
            break
        nz = np.flatnonzero(a[r:, c]) + r
        if nz.size == 0:
            continue
        p = nz[0]
        if p != r:
            a[[r, p]] = a[[p, r]]
        for i in range(rows):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        pivots.append(int(c))
        r += 1
    return a[:r], pivots


def code_from_circuit(gates, n: int, logical: int = 0) -> StabilizerCode:
    """The code encoded by ``gates`` when the logical input is qubit
    ``logical`` (the environment convention is ``logical = 0``): the
    stabilizers are the images of ``Z_q`` for ``q != logical``, the
    logicals the images of ``X_logical`` and ``Z_logical``."""
    from ..tableau import BatchTableau

    bt = BatchTableau(n, 1)
    for g, a, b in gates:
        bt.apply(g, a, b if b >= 0 else None)
    t = bt.array[0]
    others = [q for q in range(n) if q != logical]
    return StabilizerCode(
        stabilizers=t[[n + q for q in others], : 2 * n],
        logical_x=t[[logical], : 2 * n],
        logical_z=t[[n + logical], : 2 * n],
        signs=t[[n + q for q in others], 2 * n],
        destabilizers=t[others, : 2 * n],
    )


def _exclusive_pivots(rows: np.ndarray, forbidden: set, graph=None) -> list[int] | None:
    """One pivot per row such that the pivot lies in no other row's support
    and not in ``forbidden``; ``None`` if some row has no such qubit. With
    a graph, the pivot adjacent to most of its row's support is preferred
    (fewer routed CNOTs)."""
    supports = [set(np.flatnonzero(r)) for r in rows]
    pivots = []
    for i, sup in enumerate(supports):
        others = set().union(*(supports[j] for j in range(len(supports)) if j != i)) if len(supports) > 1 else set()
        cands = sorted(sup - others - forbidden)
        if not cands:
            return None
        if graph is not None:
            cands.sort(key=lambda q: -sum(graph.has_edge(q, r) for r in sup if r != q))
        pivots.append(cands[0])
    return pivots


def css_standard_encoder(stabilizers: list[str], logical: int = 0, seed: int | None = None, graph=None) -> list[tuple]:
    """Gottesman standard-form encoder (H and CNOT, all-to-all) of a CSS
    code with ``k = 1``, the logical input on qubit ``logical``.

    With X generators ``g_i`` having exclusive pivots ``p_i`` (a qubit of
    ``g_i`` in no other X generator) and Z generators ``h_q`` with
    exclusive pivots ``q`` outside the X pivots and the logical qubit, the
    circuit is: CNOT from the logical qubit to every Z pivot ``q`` whose
    generator ``h_q`` acts on the logical qubit (this spreads the logical
    X), then for every X generator ``H`` on its pivot followed by CNOTs
    from the pivot to the rest of its support. The Z generators come out
    right by the commutation relations (Gottesman, thesis, Sec. 4.2, where
    the generators are in reduced row echelon form; exclusive pivots are
    all the argument uses). With ``seed=None`` the given generators are
    used when they admit exclusive pivots (plaquettes of a lattice code do,
    which keeps the CNOTs short), otherwise, or with an integer ``seed``,
    a randomized GF(2) row reduction supplies reduced generators.
    """
    rows = np.array([pauli_to_vec(s) for s in stabilizers], dtype=np.uint8)
    n = rows.shape[1] // 2
    is_x = rows[:, n:].sum(axis=1) == 0
    is_z = rows[:, :n].sum(axis=1) == 0
    if not np.all(is_x | is_z):
        raise ValueError("css_standard_encoder needs pure X and pure Z generators")
    hx = rows[is_x, :n]
    hz = rows[is_z, n:]
    target = StabilizerCode.from_stabilizers(stabilizers)
    tkey = code_key(target.to_tableau()[0], 1)
    others = [q for q in range(n) if q != logical]

    def build(xr, px, zr, pz):
        gates = []
        for row, q in zip(zr, pz):
            if row[logical]:
                gates.append(("CNOT", int(logical), int(q)))
        for row, p in zip(xr, px):
            gates.append(("H", int(p), -1))
            for q in np.flatnonzero(row):
                if q != p:
                    gates.append(("CNOT", int(p), int(q)))
        return gates if code_key(code_from_circuit(gates, n, logical).to_tableau()[0], 1) == tkey else None

    if seed is None:
        px = _exclusive_pivots(hx, {logical}, graph)
        if px is not None:
            pz = _exclusive_pivots(hz, set(px) | {logical}, graph)
            if pz is None:
                free = [q for q in others if q not in px]
                zr, pz = _rref_with_order(hz, free + px + [logical])
                if len(pz) == hz.shape[0] and all(q in free for q in pz):
                    g = build(hx, px, zr, pz)
                    if g is not None:
                        return g
            else:
                g = build(hx, px, hz, pz)
                if g is not None:
                    return g
        seed = 0
    rng = np.random.default_rng(seed)
    for _ in range(200):
        order_x = [*list(rng.permutation(others)), logical]
        xr, px = _rref_with_order(hx, order_x)
        if len(px) != hx.shape[0] or logical in px:
            continue
        free = [q for q in others if q not in px]
        order_z = list(rng.permutation(free)) + px + [logical]
        zr, pz = _rref_with_order(hz, order_z)
        if len(pz) != hz.shape[0] or any(q not in free for q in pz):
            continue
        g = build(xr, px, zr, pz)
        if g is not None:
            return g
    raise RuntimeError("no standard-form encoder found for this pivot/logical choice")


def _shortest_path(graph, a: int, b: int) -> list[int]:
    adj = {q: set() for q in range(graph.n)}
    for u, v in graph.edges:
        adj[u].add(v)
        adj[v].add(u)
    prev = {a: None}
    frontier = [a]
    while frontier and b not in prev:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    nxt.append(v)
        frontier = nxt
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def _long_range_cnot(path: list[int]) -> list[tuple]:
    """CNOT from ``path[0]`` to ``path[-1]`` with nearest-neighbour CNOTs
    along the path: the four-gate identity through the middle qubit,
    applied recursively, or a SWAP chain (three CNOTs per SWAP) when that
    is shorter; the intermediate qubits are restored."""
    L = len(path) - 1
    if L == 1:
        return [("CNOT", path[0], path[1])]

    def recursive(p):
        if len(p) == 2:
            return [("CNOT", p[0], p[1])]
        m = len(p) // 2
        left, right = p[: m + 1], p[m:]
        # CNOT(a, c) = CNOT(m, c) CNOT(a, m) CNOT(m, c) CNOT(a, m) in time order
        return recursive(right) + recursive(left) + recursive(right) + recursive(left)

    def swap(u, v):
        return [("CNOT", u, v), ("CNOT", v, u), ("CNOT", u, v)]

    chain = []
    for i in range(L - 1):
        chain += swap(path[i], path[i + 1])
    chain += [("CNOT", path[L - 1], path[L])]
    for i in range(L - 2, -1, -1):
        chain += swap(path[i], path[i + 1])
    rec = recursive(path)
    return rec if len(rec) <= len(chain) else chain


def route_gates(gates, graph) -> list[tuple]:
    """Replace every CNOT on a non-edge of ``graph`` by a nearest-neighbour
    implementation along a shortest path."""
    out = []
    for g in gates:
        if g[0] == "CNOT" and not graph.has_edge(g[1], g[2]):
            out += _long_range_cnot(_shortest_path(graph, g[1], g[2]))
        else:
            out.append(tuple(g))
    return out


def polish_encoder(gates, setting: Setting, model, target_key: bytes, env_table, max_time_s: float = 90.0, seed: int = 0, logical: int = 0) -> dict:
    """Greedy local search on an encoding circuit that keeps the stabilizer
    group fixed and lowers the undetected fault mass under ``model``:
    passes of single-gate deletions, then adjacent swaps, then
    replacements of one gate by any other action of the setting, then
    deletions again, each accepted when the group is unchanged and the
    mass does not increase (strictly lower mass, or equal mass with fewer
    gates)."""
    n = setting.n
    rng = np.random.default_rng(seed)
    actions = [tuple(e) for e in env_table.entries]

    def score(gs):
        tab = code_from_circuit(gs, n, logical).to_tableau()
        if code_key(tab[0], 1) != target_key:
            return None
        faults = fault_set(model(gates_to_layered_circuit(gs, n)), n)
        return undetected_fault_mass(tab, 1, faults)

    cur = [tuple(g) for g in gates]
    cur_m = score(cur)
    assert cur_m is not None
    t0 = time.perf_counter()
    evals = 0
    accepted = {"delete": 0, "replace": 0, "swap": 0}

    def time_left():
        return time.perf_counter() - t0 < max_time_s

    def deletion_pass():
        nonlocal cur, cur_m, evals
        changed = True
        while changed and time_left():
            changed = False
            for i in rng.permutation(len(cur)):
                cand = cur[:i] + cur[i + 1 :]
                m = score(cand)
                evals += 1
                if m is not None and m <= cur_m + 1e-15:
                    cur, cur_m = cand, m
                    accepted["delete"] += 1
                    changed = True
                    break

    deletion_pass()
    for i in rng.permutation(max(len(cur) - 1, 0)):
        if not time_left():
            break
        cand = [*cur[:i], cur[i + 1], cur[i], *cur[i + 2:]]
        m = score(cand)
        evals += 1
        if m is not None and m < cur_m - 1e-15:
            cur, cur_m = cand, m
            accepted["swap"] += 1
    for i in rng.permutation(len(cur)):
        if not time_left():
            break
        for a in rng.permutation(len(actions)):
            if not time_left():
                break
            g = actions[a]
            if g == cur[i]:
                continue
            cand = [*cur[:i], g, *cur[i + 1:]]
            m = score(cand)
            evals += 1
            if m is not None and m < cur_m - 1e-15:
                cur, cur_m = cand, m
                accepted["replace"] += 1
                break
    deletion_pass()
    return {"gates": [list(g) for g in cur], "mass": cur_m, "evaluations": evals, "accepted": accepted, "wall_time_s": round(time.perf_counter() - t0, 1)}


def stim_encoder(stabilizers: list[str], gate_set) -> list[tuple] | None:
    """stim's elimination encoder of the code, expanded into the gate set;
    ``None`` if it needs a gate outside it (only meaningful on all-to-all)."""
    code = StabilizerCode.from_stabilizers(stabilizers)
    circ = code.to_stim_tableau().to_circuit("elimination")
    gates = []
    for inst in circ.flattened():
        tg = [t.value for t in inst.targets_copy()]
        name = inst.name
        if name in ("H", "S"):
            gates += [(name, q, -1) for q in tg]
        elif name == "S_DAG":
            gates += [("S", q, -1) for q in tg for _ in range(3)]
        elif name == "CX":
            gates += [("CNOT", tg[i], tg[i + 1]) for i in range(0, len(tg), 2)]
        elif name == "SQRT_X":
            gates += [(g, q, -1) for q in tg for g in ("H", "S", "H")]
        elif name == "SQRT_X_DAG":
            gates += [(g, q, -1) for q in tg for g in ("H", "S", "S", "S", "H")]
        elif name in ("X", "Y", "Z", "I"):
            continue  # Pauli frame: irrelevant to the code and to detection
        else:
            return None
    if any(g[0] not in gate_set for g in gates):
        return None
    return gates


def known_code_encoders(setting: Setting, spec: dict, models: dict, polish_time_s: float = 90.0, verbose: bool = False, n_seeds: int = 3) -> dict:
    """Encoders of a known code on the setting's graph: the standard-form
    CSS encoder of the parent code followed by H on the frame, routed onto
    the graph, over every choice of logical input qubit and of pivots (the
    given generators when they admit exclusive pivots, else randomized
    row reductions); the ``n_seeds`` shortest routed circuits are polished
    per profile; on all-to-all stim's elimination encoder is a further
    seed. Returns the best encoder per profile. The RL agent's encoders
    always take qubit 0 as the logical input; the baselines may use any
    qubit, a freedom that can only favour them."""
    graph = graphs.from_spec(setting.graph_spec)
    n = setting.n
    table = CodeDiscoveryEnv(n=n, k=1, graph=graph, batch_size=1, gates=setting.gates, max_steps=1).action_table
    frame = list(spec.get("frame", []))
    parent = hadamard_conjugate(spec["stabilizers"], frame) if frame else list(spec["stabilizers"])
    target_key = code_key(StabilizerCode.from_stabilizers(spec["stabilizers"]).to_tableau()[0], 1)
    cands = []
    for logical in range(n):
        for sd in [None, *list(range(12))]:
            try:
                base = css_standard_encoder(parent, logical=logical, seed=sd, graph=graph)
            except RuntimeError:
                continue
            routed = route_gates(base + [("H", q, -1) for q in frame], graph)
            src = "given generators" if sd is None else f"RREF seed {sd}"
            cands.append({"source": f"standard form ({src}), logical qubit {logical}, routed", "gates": routed, "logical": logical})
    cands.sort(key=lambda c: (len(c["gates"]), circuit_depth(c["gates"], n)))
    seeds = []
    seen = set()
    for c in cands:
        key = tuple(c["gates"])
        if key in seen:
            continue
        seen.add(key)
        assert code_key(code_from_circuit(c["gates"], n, c["logical"]).to_tableau()[0], 1) == target_key
        seeds.append(c)
        if len(seeds) >= n_seeds:
            break
    if setting.graph_spec.startswith("all_to_all"):
        g = stim_encoder(spec["stabilizers"], setting.gates)
        if g is not None:
            seeds.append({"source": "stim elimination, logical qubit 0", "gates": g, "logical": 0})
    out = {
        "candidates": len(cands),
        "seeds": [{"source": sd["source"], "circuit": circuit_stats(sd["gates"], n)} for sd in seeds],
        "best_by_profile": {},
    }
    for pname, model in models.items():
        best = None
        for sd in seeds:
            pol = polish_encoder(sd["gates"], setting, model, target_key, table, max_time_s=polish_time_s, logical=sd["logical"])
            raw_tab = code_from_circuit(sd["gates"], n, sd["logical"]).to_tableau()
            raw_mass = undetected_fault_mass(raw_tab, 1, fault_set(model(gates_to_layered_circuit(sd["gates"], n)), n))
            rec = {"source": sd["source"], "logical_qubit": sd["logical"], "unpolished_mass": raw_mass, "unpolished_gates": len(sd["gates"]), **pol}
            if best is None or rec["mass"] < best["mass"]:
                best = rec
        gates = [tuple(g) for g in best["gates"]]
        tab = code_from_circuit(gates, n, best["logical_qubit"]).to_tableau()
        best["circuit"] = circuit_stats(gates, n)
        best["undetected"] = best.pop("mass")
        best["mass_all_profiles"] = {}
        for other, om in models.items():
            faults = fault_set(om(gates_to_layered_circuit(gates, n)), n)
            best["mass_all_profiles"][other] = {"undetected": undetected_fault_mass(tab, 1, faults), "total": faults.total}
        best["total"] = best["mass_all_profiles"][pname]["total"]
        out["best_by_profile"][pname] = best
        if verbose:
            print(
                f"    {pname}: {best['source']} {best['unpolished_gates']} -> {best['circuit']['gates']} gates, "
                f"mass {best['unpolished_mass']:.4g} -> {best['undetected']:.4g} ({best['evaluations']} evals, {best['wall_time_s']} s)"
            )
    return out


def run_baselines(setting: Setting, threads: int, verbose: bool, calib: dict | None = None, polish_time_s: float = 90.0) -> dict:
    calib = calib or calibrate(setting)
    graph = graphs.from_spec(setting.graph_spec)
    models = calibrated_models(setting, calib, graph)
    out = {"setting": setting.to_dict(), "calibration": calib, "polish_time_s": polish_time_s, "codes": {}}
    for name, spec in known_codes_for(setting).items():
        t0 = time.perf_counter()
        code = StabilizerCode.from_stabilizers(spec["stabilizers"])
        tab = code.to_tableau()
        st = sparsest_basis_stats(tab, 1, exact_max_generators=EXACT_GENERATORS)
        dx, nx = pure_type_distance(tab, 1, "X")
        dz, nz = pure_type_distance(tab, 1, "Z")
        rec = {
            "description": spec["description"],
            "stabilizers": spec["stabilizers"],
            "frame": spec.get("frame", []),
            "distance": int(code.distance_exact()),
            "distance_x": dx,
            "distance_x_count": nx,
            "distance_z": dz,
            "distance_z_count": nz,
            "w_star": int(st["w_star"][0]),
            "max_qubit_degree": int(st["max_degree"][0]),
            "is_css": bool(code.is_css()),
            "weight_enumerator_S": code.weight_enumerators()[0].tolist(),
            "components": group_component_counts(code),
        }
        if verbose:
            print(f"[baseline] {setting.name} {name}: d={rec['distance']} dX={dx} dZ={dz} w*={rec['w_star']}; building encoders")
        rec["encoders"] = known_code_encoders(setting, spec, models, polish_time_s=polish_time_s, verbose=verbose)
        rec["wall_time_s"] = round(time.perf_counter() - t0, 1)
        out["codes"][name] = rec
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def load_runs(out_dir: Path) -> dict:
    runs = {}
    for p in sorted((out_dir / "runs").glob("*.json")):
        try:
            with open(p) as f:
                r = json.load(f)
        except json.JSONDecodeError:
            print(f"skipping unreadable {p} (still being written?)")
            continue
        runs.setdefault(r["setting"]["name"], {}).setdefault(r["profile"], []).append(r)
    for s in runs.values():
        for lst in s.values():
            lst.sort(key=lambda r: r["seed"])
    return runs


def load_baselines(out_dir: Path) -> dict:
    out = {}
    for p in sorted(out_dir.glob("baselines_*.json")):
        try:
            with open(p) as f:
                b = json.load(f)
        except json.JSONDecodeError:
            print(f"skipping unreadable {p}")
            continue
        out[b["setting"]["name"]] = b
    return out


def _json_default(o):
    """numpy scalars and arrays as plain Python for ``json.dump``."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def _mean_range(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3g}"
    return f"{np.mean(vals):.3g} [{min(vals):.3g}, {max(vals):.3g}]"


def _on_grid(steps, values, grid, window: int = 25):
    """Rolling mean over ``window`` updates, then linear interpolation onto
    ``grid`` (env steps); NaN beyond the run's last step."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.full(grid.shape, np.nan)
    # trailing mean over the last ``window`` updates (shorter at the start)
    c = np.concatenate([[0.0], np.cumsum(v)])
    idx = np.arange(1, v.size + 1)
    lo = np.maximum(0, idx - window)
    sm = (c[idx] - c[lo]) / (idx - lo)
    out = np.interp(grid, steps, sm)
    out[grid > steps[-1]] = np.nan
    return out


def make_figures(runs: dict, out_dir: Path) -> list[str]:
    """One figure per setting: (a) mean step reward, (b) success rate per
    update, (c) minus the best terminal reward so far (log axis), each as
    the median over seeds (line) and the min-max band across seeds, rolling
    mean over 25 updates for (a) and (b)."""
    from . import figstyle

    plt = figstyle.apply()
    paths = []
    for sname, prof_runs in runs.items():
        fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE))
        max_steps = max(r["env_steps"] for lst in prof_runs.values() for r in lst)
        grid = np.linspace(0, max_steps, 300)
        for pname in PROFILES:
            lst = prof_runs.get(pname, [])
            if not lst:
                continue
            color = figstyle.PROFILE_COLORS[pname]
            series = {"a": [], "b": [], "c": []}
            for r in lst:
                h = r["history"]
                steps = np.array([x["env_steps"] for x in h], dtype=float)
                series["a"].append(_on_grid(steps, [x["mean_reward"] for x in h], grid))
                series["b"].append(_on_grid(steps, [x["success_rate"] for x in h], grid))
                best = np.array([np.nan if x.get("best_terminal_reward") is None else -x["best_terminal_reward"] for x in h])
                c = np.interp(grid, steps, np.where(np.isnan(best), np.nan, best))
                c[grid > steps[-1]] = np.nan
                series["c"].append(c)
            for ax, key in zip(axes, "abc"):
                arr = np.array(series[key])
                with np.errstate(all="ignore"):
                    med = np.nanmedian(arr, axis=0)
                    lo = np.nanmin(arr, axis=0)
                    hi = np.nanmax(arr, axis=0)
                x = grid / 1e6
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
                ax.plot(x, med, color=color, lw=figstyle.LINE, label=f"{figstyle.PROFILE_LABELS[pname]} ({len(lst)} seeds)" if key == "b" else None)
        axes[0].set_ylabel("mean step reward (25-update mean)")
        axes[1].set_ylabel("success rate per update (25-update mean)")
        axes[2].set_ylabel("-(best terminal reward so far)")
        if any(x.get("best_terminal_reward") is not None for lst in prof_runs.values() for r in lst for x in r["history"]):
            axes[2].set_yscale("log")
        else:
            axes[2].text(0.5, 0.5, "no run reached\nthe target distance", transform=axes[2].transAxes, ha="center", va="center")
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            axes[2].grid(False)
        axes[1].set_ylim(-0.03, 1.03)
        for ax, letter in zip(axes, "abc"):
            ax.set_xlabel("env steps (millions)")
            figstyle.panel_label(ax, letter)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
        if axes[2].get_yscale() == "log":
            axes[2].set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
            axes[2].yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axes[2].yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        st = SETTINGS[sname]
        fig.suptitle(
            f"{sname}: n = {st.n}, target d = {st.d}, {st.graph_spec}, gates {', '.join(st.gates)}; median over seeds, band = min to max",
            y=1.02,
        )
        fig.tight_layout()
        png, _ = figstyle.save(fig, out_dir / "figures" / f"curves_{sname}")
        plt.close(fig)
        paths.append(str(png.relative_to(out_dir)))
    return paths


def bias_flags(a: dict) -> dict:
    """Structural-bias indicators of an analysed code."""
    dx, dz = a["distance_x"], a["distance_z"]
    comp = a["components"]
    return {
        "dz_gt_dx": (dz is not None and dx is not None and dz > dx) or (dz is None and dx is not None),
        "dx_gt_dz": (dz is not None and dx is not None and dx > dz) or (dx is None and dz is not None),
        "min_z_logicals": a["distance_z_count"],
        "min_x_logicals": a["distance_x_count"],
        "pure_z_elements": comp["pure_z_elements"],
        "pure_x_elements": comp["pure_x_elements"],
        "css": a["is_css"],
        "frame": a["frame"].get("css_frame_found", False),
        "frame_size": a["frame"].get("frame_size"),
    }


def permutation_test(a, b, n_max: int = 20000, seed: int = 0) -> dict:
    """Two-sided exact (or Monte Carlo above ``n_max`` relabelings)
    permutation test of the difference of means between two small samples.
    Returns the means, their difference and ratio, the number of ``(i, j)``
    pairs with ``a_i < b_j`` out of ``len(a) len(b)``, and the p-value."""
    from itertools import combinations
    from math import comb

    a = [float(x) for x in a if x is not None]
    b = [float(x) for x in b if x is not None]
    if not a or not b:
        return {"n_a": len(a), "n_b": len(b)}
    pooled = np.array(a + b)
    na = len(a)
    obs = float(np.mean(a) - np.mean(b))
    if comb(len(pooled), na) <= n_max:
        count = 0
        total = 0
        for idx in combinations(range(len(pooled)), na):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(idx)] = True
            diff = pooled[mask].mean() - pooled[~mask].mean()
            count += abs(diff) >= abs(obs) - 1e-12
            total += 1
        pval = count / total
        exact = True
    else:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_max):
            perm = rng.permutation(len(pooled))
            diff = pooled[perm[:na]].mean() - pooled[perm[na:]].mean()
            count += abs(diff) >= abs(obs) - 1e-12
        pval = count / n_max
        exact = False
    pairs = sum(1 for x in a for y in b if x < y)
    return {
        "n_a": na,
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "difference": obs,
        "ratio_b_over_a": float(np.mean(b) / np.mean(a)) if np.mean(a) else None,
        "pairs_a_below_b": pairs,
        "pairs": na * len(b),
        "p_value": pval,
        "exact": exact,
    }


def ensure_breakdowns(runs: dict) -> None:
    """Fill in ``mass[profile]["breakdown"]`` of every best and modal code
    that lacks it (runs written before the breakdown existed)."""
    for sname, prof_runs in runs.items():
        setting = SETTINGS[sname]
        models = None
        for lst in prof_runs.values():
            for r in lst:
                for key in ("best", "modal"):
                    a = r.get(key)
                    if not a:
                        continue
                    for ev in PROFILES:
                        if "breakdown" in a["mass"][ev]:
                            continue
                        if models is None:
                            calib = calibrate(setting)
                            models = calibrated_models(setting, calib, graphs.from_spec(setting.graph_spec))
                        gates = [tuple(g) for g in a["gates"]]
                        tab = replay(gates, setting.n).to_tableau()
                        a["mass"][ev]["breakdown"] = mass_breakdown(gates, setting.n, models[ev], tab)


def breakdown_table(prof_runs: dict) -> list[str]:
    L = [
        "Undetected fault mass of the best codes split by the mechanism that caused it (mean over seeds): "
        "two-qubit gate channel, single-qubit gate channel, idle channel with a Z fault, idle channel with an X or Y fault. "
        "The board profile has no single-qubit gate channel and a depolarizing idle; the ion profile's idle is 95 percent Z.\n",
        "| trained on | evaluated on | two-qubit gate | single-qubit gate | idle Z | idle X or Y | total undetected | total fault mass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pname in PROFILES:
        lst = [r for r in prof_runs.get(pname, []) if r["best"]]
        if not lst:
            continue
        for ev in PROFILES:
            bs = [r["best"]["mass"][ev]["breakdown"] for r in lst]

            def m(getter):
                return float(np.mean([getter(b) for b in bs]))

            def cat(b, key):
                return b["by_class_and_type"].get(key, {}).get("undetected", 0.0)

            L.append(
                f"| {pname} | {ev} | {m(lambda b: b['by_class']['two-qubit gate']['undetected']):.2e} | "
                f"{m(lambda b: b['by_class']['single-qubit gate']['undetected']):.2e} | "
                f"{m(lambda b: cat(b, 'idle / Z')):.2e} | "
                f"{m(lambda b: cat(b, 'idle / X') + cat(b, 'idle / Y or mixed')):.2e} | "
                f"{m(lambda b: b['undetected']):.2e} | {m(lambda b: b['total']):.3f} |"
            )
    L.append("")
    return L


def analysis_tables(prof_runs: dict, base: dict | None) -> list[str]:
    """Markdown for the three analysis questions of one setting."""
    L = []
    have = {p: [r for r in prof_runs.get(p, []) if r["best"]] for p in PROFILES}
    # (2) cross matrix: for every evaluation profile B, codes trained on B against codes trained on A
    L.append("#### (2) Does training on the evaluation profile help? Seed-level comparison\n")
    L.append(
        "For each evaluation profile B: undetected mass of the five codes trained on B against the five "
        "trained on A, evaluated on B. `pairs` counts seed pairs (i, j) with mass(B-trained_i) < mass(A-trained_j) "
        "out of 25; p is the exact two-sided permutation p-value of the difference of means (252 relabelings, "
        "so the smallest attainable p is 0.0079). With five seeds a ratio near 1 with p above 0.1 is not distinguishable from no effect.\n"
    )
    L.append("| evaluated on B | trained on B (mean) | trained on A | A-trained mean | ratio A/B | pairs B < A | p |")
    L.append("|---|---|---|---|---|---|---|")
    for ev in PROFILES:
        own = [r["best"]["mass"][ev]["undetected"] for r in have[ev]]
        for other in PROFILES:
            if other == ev:
                continue
            oth = [r["best"]["mass"][ev]["undetected"] for r in have[other]]
            t = permutation_test(own, oth)
            if "mean_a" not in t:
                continue
            L.append(
                f"| {ev} | {t['mean_a']:.4g} | {other} | {t['mean_b']:.4g} | {t['ratio_b_over_a']:.2f} | "
                f"{t['pairs_a_below_b']}/{t['pairs']} | {t['p_value']:.3f} |"
            )
    L.append("")
    # (1) bias
    L.append("#### (1) Structural bias\n")
    L.append(
        "Counts over seeds of codes with d_Z > d_X and d_X > d_Z (pure-type distances in the physical frame), "
        "and the per-seed pure-Z : pure-X stabilizer element counts. The ion profile's idle noise is Z-dephasing, so the "
        "structural prediction is more Z-detecting structure (more X-type stabilizer weight, larger or more numerous "
        "minimum-weight pure-Z logicals detected, that is fewer minimum-weight pure-Z logicals) under ion than under board.\n"
    )
    for pname in PROFILES:
        lst = have[pname]
        if not lst:
            continue
        flags = [bias_flags(r["best"]) for r in lst]
        L.append(
            f"- {pname}: d_Z > d_X in {sum(f['dz_gt_dx'] for f in flags)}/{len(lst)} seeds, d_X > d_Z in "
            f"{sum(f['dx_gt_dz'] for f in flags)}/{len(lst)}; minimum-weight pure-Z : pure-X logical counts "
            + ", ".join(f"{f['min_z_logicals']}:{f['min_x_logicals']}" for f in flags)
            + "; pure-Z : pure-X stabilizer elements "
            + ", ".join(f"{f['pure_z_elements']}:{f['pure_x_elements']}" for f in flags)
            + f"; CSS in {sum(f['css'] for f in flags)}/{len(lst)}, non-CSS with a local-Clifford CSS frame in "
            f"{sum(1 for f in flags if f['frame'] and not f['css'])}/{len(lst)}."
        )
    L.append("")
    # (3) known codes
    if base:
        L.append("#### (3) Discovered against known codes\n")
        L.append("| evaluated on | best known code (polished encoder) | its mass | discovered, trained on the same profile: mean [min, max] | seeds beating the best known | seeds beating every known code |")
        L.append("|---|---|---|---|---|---|")
        for ev in PROFILES:
            known = {cn: c["encoders"]["best_by_profile"][ev]["undetected"] for cn, c in base["codes"].items() if ev in c["encoders"]["best_by_profile"]}
            if not known:
                continue
            kbest = min(known, key=known.get)
            kworst_val = max(known.values())
            own = [r["best"]["mass"][ev]["undetected"] for r in have[ev]]
            if not own:
                continue
            L.append(
                f"| {ev} | {kbest} | {known[kbest]:.4g} | {_mean_range(own)} | "
                f"{sum(1 for m in own if m < known[kbest])}/{len(own)} | {sum(1 for m in own if m < kworst_val)}/{len(own)} |"
            )
        L.append("")
    return L


def write_report(out_dir: Path) -> Path:
    runs = load_runs(out_dir)
    ensure_breakdowns(runs)
    bases = load_baselines(out_dir)
    figs = make_figures(runs, out_dir) if runs else []
    L = []
    L.append("# Noise-aware code discovery with PPO under three noise profiles\n")
    L.append(
        "Question: does the code an RL agent discovers depend on the noise model it is "
        "trained against, and in the direction the model's structure predicts (the experiment "
        "the proposal cites Olle et al., arXiv:2311.04750, for)? Module: "
        "`src/autoqec_sim/experiments/noise_aware.py`; one cell reproduces with "
        "`uv run --extra rl python -m autoqec_sim.experiments.noise_aware run --setting S --profile P --seed K`.\n"
    )
    verdict = out_dir / "verdict.md"
    if verdict.exists():
        L.append("## Verdict\n")
        L.append(verdict.read_text().strip() + "\n")
    L.append("## Reward and protocol\n")
    L.append(
        "State reward of a tableau `s` reached by the gate list `c`: "
        "`r(s) = min(d(s), d*)/d* - 1 - q_1(s) + [d(s) >= d*, q_1(s) = 0] (-2 m(c)/M_0 - 0.25 max(0, w*(s) - 4))`. "
        "`d(s)` is the exact distance up to `d* - 1` (`reward.distance_reward`, shifted by -1); `q_1(s)` is the "
        "fraction of weight-1 Paulis in the stabilizer group (`reward.low_weight_stabilizer_penalty`, the padding "
        "penalty of the rediscovery regression: a first pass without it, kept under `pilot/no_padding_penalty/`, "
        "had the 3x3 agent embed a [[7,1,3]] code and leave two qubits idle in 2 of 3 profiles at seed 0, since an "
        "idle qubit adds only trivially detected faults; the 7-qubit cells of that pass were not padded); "
        "`m(c)` is the circuit-level undetected fault mass of the encoding circuit under the "
        "training profile (`circuit_noise.undetected_fault_mass`, `normalize=False`, absolute "
        "probability, every fault mechanism of the noisy circuit propagated to the output and "
        "tested with the Knill-Laflamme kernel); `M_0` is the setting's common total fault mass "
        "(below); `w*` is the exact minimum bottleneck generator weight (`ldpc.ldpc_penalty`, "
        "`w_max = 4`, scale 0.25). Noise and LDPC terms are evaluated only on states that reach "
        "the target distance with no weight-1 stabilizer, and such a state ends the episode. In the 4x4 setting the distance term is "
        "refined below the target to `(d - 1 + f)/d* - 1` with `f` the detected fraction of the weight-`d` "
        "Paulis (the level reward alone left PPO at d = 1 to 2 for 32.5M steps, 25 minutes, in the pilot). "
        "Dense reward (state reward after every gate), PPO defaults of `agents.ppo.PPOConfig` (256 parallel episodes, 16-step "
        "rollouts, 4 epochs, lr 3e-4, gamma 0.98, entropy 0.01), undo mask on.\n"
    )
    L.append("### Profiles and calibration\n")
    L.append(
        "Each profile is scaled so that the mean total fault mass over 64 random circuits of the "
        "setting's maximum length equals the board preset's at scale 1 (`M_0`). The Heron preset's "
        "heavy-hex graph is replaced by the setting's graph; the settings fix connectivity, so "
        "`heron` differs from `board` in rate structure (p1 = 2e-4 on single-qubit gates, T1/T2 "
        "idle with eta = 1.8, gate bias 1/2), not in connectivity.\n"
    )
    L.append("| setting | profile | preset | scale | total mass at scale 1 (mean over 64 random circuits) | M_0 |")
    L.append("|---|---|---|---|---|---|")
    for sname, b in bases.items():
        c = b["calibration"]
        for pname in PROFILES:
            cp = c["profiles"][pname]
            L.append(f"| {sname} | {pname} | {cp['description']} | {cp['scale']:.3f} | {cp['mass_scale1_mean']:.4f} | {c['m0']:.4f} |")
    for sname in runs:
        if sname in bases:
            continue
        c = calibrate(SETTINGS[sname])
        for pname in PROFILES:
            cp = c["profiles"][pname]
            L.append(f"| {sname} | {pname} | {cp['description']} | {cp['scale']:.3f} | {cp['mass_scale1_mean']:.4f} | {c['m0']:.4f} |")
    L.append("")

    L.append("## Runs\n")
    L.append("| setting | profile | seeds | budget (env steps) | wall time per seed, s | seeds reaching d* | first success (env steps) | final success rate | distinct target codes per seed | noise evaluations per seed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for sname, prof_runs in runs.items():
        for pname in PROFILES:
            lst = prof_runs.get(pname, [])
            if not lst:
                continue
            reached = sum(1 for r in lst if r["distinct_success_codes"] > 0)
            L.append(
                f"| {sname} | {pname} | {len(lst)} | {lst[0]['budget_env_steps']:,} | "
                f"{_mean_range([r['wall_time_s'] for r in lst])} | {reached}/{len(lst)} | "
                f"{_mean_range([r['first_success_env_steps'] for r in lst])} | "
                f"{_mean_range([r['final_success_rate'] for r in lst])} | "
                f"{_mean_range([float(r['distinct_success_codes']) for r in lst])} | "
                f"{_mean_range([float(r['noise_evals']) for r in lst])} |"
            )
    caps = [(s, p, r["seed"]) for s, pr in runs.items() for p, lst in pr.items() for r in lst if r["hit_time_cap"]]
    L.append("")
    if caps:
        L.append("Runs stopped by the 25-minute cap before their step budget: " + ", ".join(f"{s}/{p}/seed {k}" for s, p, k in caps) + ".\n")
    else:
        L.append("No run hit the 25-minute cap; every run used its full step budget.\n")
    L.append(
        "Budgets: 6M env steps for n7 and 8M for 3x3 (about 5 and 10 minutes per seed at 2 threads; PPO reaches a "
        "success rate near 1 within the first 0.5M to 1M steps in both, so the rest of the budget explores distinct "
        "target-distance codes), 16M for 4x4 with the 25-minute cap. A 4x4 pilot with the plain level distance reward "
        "(`pilot/4x4_board_level_reward_seed1.json`, board profile, 32.5M steps in 25 minutes) never produced a "
        "distance-4 state and stayed at mean step reward -0.75 to -0.66 (d = 1 to 2), which is why the 4x4 cells use "
        "the fractional distance term.\n"
    )

    for fpath in figs:
        L.append(f"![]({fpath})\n")
    if figs:
        L.append(
            "Training curves per setting, median over seeds (line) with the min-to-max band across seeds, "
            "color by profile: (a) mean step reward over the rollout, trailing mean over 25 updates (dense "
            "reward, so it rises from -(d* - 1)/d* as the policy reaches higher-distance states earlier; "
            "the band width after convergence reflects seeds whose success rate collapses and recovers, "
            "which the raw per-update curves show as spikes to 0), (b) fraction of finished episodes that "
            "reached the target distance, same smoothing, (c) minus the best terminal reward seen so far, "
            "that is the noise plus LDPC penalty of the best target-distance code, log axis. In a setting "
            "where no run reached the target the profile never enters the reward and the three profiles' curves "
            "coincide (only the last drawn is visible). The raw per-update values are in the CSV files under `runs/`.\n"
        )

    for sname, prof_runs in runs.items():
        s = SETTINGS[sname]
        b = bases.get(sname)
        L.append(f"## Setting {sname}: n = {s.n}, k = 1, target d = {s.d}, {s.graph_spec}, gates {', '.join(s.gates)}\n")
        reached = sum(1 for lst in prof_runs.values() for r in lst if r["distinct_success_codes"] > 0)
        total_runs = sum(len(lst) for lst in prof_runs.values())
        if reached == 0:
            L.append(
                f"No run of this setting reached distance {s.d} ({total_runs} runs finished). The largest "
                "distance of a terminal state per run and the mean step reward are in the runs table and the "
                "training curves; the best state per run below is the terminal state of the largest distance with "
                "the smallest undetected mass under the training profile, which is not a distance-"
                f"{s.d} code, so no cross-evaluation, bias, or known-code comparison is made for this setting. "
                "Until a state reaches the target distance the profile does not enter the reward, so runs of the "
                "three profiles with the same seed are the same trajectory (identical mean-reward curves and "
                "terminal states); the profile column below is therefore not a treatment.\n"
            )
            L.append("| trained on | seed | largest terminal d | distinct states at that d | final mean step reward | d of best state | d_X | d_Z | gates |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for pname in PROFILES:
                for r in prof_runs.get(pname, []):
                    a = r["best"]
                    L.append(
                        f"| {pname} | {r['seed']} | {r['partial_distance']} | {r['distinct_partial_codes']} | "
                        f"{r['history'][-1]['mean_reward']:+.3f} | {a['distance'] if a else 'n/a'} | {_fmt(a['distance_x']) if a else ''} | "
                        f"{_fmt(a['distance_z']) if a else ''} | {a['circuit']['gates'] if a else ''} |"
                    )
            L.append("")
            if b:
                L.append("### Known codes at the same (n, k)\n")
                L.append("Baselines for the setting (polished encoder per profile), for reference:\n")
                L.append("| code | d | d_X | d_Z | w* | CSS | " + " | ".join(f"{ev} (gates, unpolished mass)" for ev in PROFILES) + " |")
                L.append("|---|---|---|---|---|---|" + "---|" * len(PROFILES))
                for cname, c in b["codes"].items():
                    bb = c["encoders"]["best_by_profile"]
                    cells = [
                        f"{bb[ev]['undetected']:.4g} ({bb[ev]['circuit']['gates']}, {bb[ev]['unpolished_mass']:.4g})" if ev in bb else "no encoder"
                        for ev in PROFILES
                    ]
                    L.append(
                        f"| {cname} ({c['description']}) | {c['distance']} | {_fmt(c['distance_x'])} | {_fmt(c['distance_z'])} | "
                        f"{c['w_star']} | {'yes' if c['is_css'] else 'no'} | " + " | ".join(cells) + " |"
                    )
                L.append("")
            continue
        # cross matrix
        L.append("### Cross-evaluation matrix\n")
        L.append(
            "Rows: training profile (the best code of each seed, its own encoding circuit). "
            "Columns: evaluation profile at the calibrated scale. Entries: undetected fault mass, "
            "mean over seeds with [min, max]; per seed values in the code list below. Smaller is better.\n"
        )
        L.append("| trained on \\ evaluated on | " + " | ".join(PROFILES) + " | d reached (seeds) | gates (mean) |")
        L.append("|---|" + "---|" * (len(PROFILES) + 2))
        for pname in PROFILES:
            lst = [r for r in prof_runs.get(pname, []) if r["best"]]
            if not lst:
                continue
            cells = []
            for ev in PROFILES:
                cells.append(_mean_range([r["best"]["mass"][ev]["undetected"] for r in lst]))
            ds = [r["best"]["distance"] for r in lst]
            L.append(
                f"| {pname} | " + " | ".join(cells) + f" | {', '.join(map(str, ds))} | "
                f"{np.mean([r['best']['circuit']['gates'] for r in lst]):.1f} |"
            )
        L.append("")
        # per-seed matrix (compact)
        L.append("Per seed (undetected mass under board / ion / heron):\n")
        L.append("| trained on | seed | d | d_X | d_Z | w* | CSS | frame | board | ion | heron | gates | CNOTs | depth |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pname in PROFILES:
            for r in prof_runs.get(pname, []):
                a = r["best"]
                if not a:
                    L.append(f"| {pname} | {r['seed']} | none | | | | | | | | | | | |")
                    continue
                fr = a["frame"]
                frame = ("CSS" if a["is_css"] else (f"{fr['search']} frame, {fr['frame_size']} qubits" if fr.get("css_frame_found") else "none"))
                L.append(
                    f"| {pname} | {r['seed']} | {a['distance']} | {_fmt(a['distance_x'])} | {_fmt(a['distance_z'])} | {a['w_star']} | "
                    f"{'yes' if a['is_css'] else 'no'} | {frame} | "
                    + " | ".join(f"{a['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                    + f" | {a['circuit']['gates']} | {a['circuit']['two_qubit_gates']} | {a['circuit']['depth']} |"
                )
        L.append("")
        L.append("### Mechanism breakdown of the undetected mass\n")
        L += breakdown_table(prof_runs)
        # modal (converged-policy) code
        L.append("Most visited target-distance code of the last 20 updates (the converged policy's code), undetected mass under board / ion / heron:\n")
        L.append("| trained on | seed | visits (of successes in the window) | d | d_X | d_Z | w* | CSS | board | ion | heron | gates | same as best |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pname in PROFILES:
            for r in prof_runs.get(pname, []):
                m = r.get("modal")
                if not m:
                    continue
                same = r["best"] is not None and m["stabilizers"] == r["best"]["stabilizers"]
                L.append(
                    f"| {pname} | {r['seed']} | {m['visits_last_20_updates']} ({m['successes_last_20_updates']}) | {m['distance']} | "
                    f"{_fmt(m['distance_x'])} | {_fmt(m['distance_z'])} | {m['w_star']} | {'yes' if m['is_css'] else 'no'} | "
                    + " | ".join(f"{m['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                    + f" | {m['circuit']['gates']} | {'yes' if same else 'no'} |"
                )
        L.append("")
        # baselines
        if b:
            L.append("### Known codes at the same (n, k)\n")
            L.append(
                "Encoders: the Gottesman standard-form encoder of the CSS parent code followed by H on the "
                "frame, routed onto the setting's graph (nearest-neighbour CNOT chains), then polished per "
                "profile by a greedy local search that keeps the stabilizer group fixed and lowers the "
                "undetected mass (single-gate deletions, adjacent swaps, single-gate replacements, "
                f"{b.get('polish_time_s', 90)} s per profile); on all-to-all also stim's elimination encoder. "
                "The entry is the polished encoder's undetected mass under that profile (a different circuit "
                "per column). The RL codes have encoders co-optimized with the code by the agent, so this "
                "comparison is fair only to the extent the polishing reaches comparable circuits; the "
                "unpolished mass and gate counts are given to show how much the polishing did.\n"
            )
            L.append("| code | d | d_X | d_Z | w* | CSS | " + " | ".join(f"{ev} (gates, unpolished mass)" for ev in PROFILES) + " |")
            L.append("|---|---|---|---|---|---|" + "---|" * len(PROFILES))
            for cname, c in b["codes"].items():
                bb = c["encoders"]["best_by_profile"]
                cells = [
                    f"{bb[ev]['undetected']:.4g} ({bb[ev]['circuit']['gates']}, {bb[ev]['unpolished_mass']:.4g})" if ev in bb else "no encoder"
                    for ev in PROFILES
                ]
                L.append(
                    f"| {cname} ({c['description']}) | {c['distance']} | {_fmt(c['distance_x'])} | {_fmt(c['distance_z'])} | "
                    f"{c['w_star']} | {'yes' if c['is_css'] else 'no'} | " + " | ".join(cells) + " |"
                )
            L.append("")
        # code list
        L.append("### Best code per seed\n")
        for pname in PROFILES:
            for r in prof_runs.get(pname, []):
                a = r["best"]
                L.append(f"**{sname} / {pname} / seed {r['seed']}**: {r['selection']}.")
                if not a:
                    L.append("")
                    continue
                comp = a["components"]
                fr = a["frame"]
                L.append(
                    f"[[{a['n']},{a['k']},{a['distance']}]], d_X = {_fmt(a['distance_x'])} ({a['distance_x_count']} min-weight), "
                    f"d_Z = {_fmt(a['distance_z'])} ({a['distance_z_count']}), w* = {a['w_star']} (degree {a['max_qubit_degree']}), "
                    f"CSS: {'yes' if a['is_css'] else 'no'}"
                    + (
                        f"; {fr['search']} CSS frame on {fr['frame_size']} qubit(s) {fr['frame']}"
                        if fr.get("css_frame_found") and not a["is_css"]
                        else ("" if a["is_css"] else f"; no {fr['search']} frame makes it CSS")
                    )
                    + f". Pure-X stabilizer elements {comp['pure_x_elements']} (weights {comp['pure_x_weights']}), "
                    f"pure-Z {comp['pure_z_elements']} (weights {comp['pure_z_weights']}). "
                    f"Circuit: {a['circuit']['gates']} gates ({a['circuit']['two_qubit_gates']} CNOT, {a['circuit']['h_gates']} H, "
                    f"{a['circuit']['s_gates']} S), depth {a['circuit']['depth']}. "
                    f"Undetected mass board/ion/heron = " + "/".join(f"{a['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                    + (
                        f". Reward terms: distance {a['reward_terms']['distance_term']:+.3f}, padding {a['reward_terms']['padding_term']:+.3f}, noise {a['reward_terms']['noise_term']:+.4f}, "
                        f"LDPC {a['reward_terms']['ldpc_term']:+.3f}."
                        if "reward_terms" in a
                        else "."
                    )
                )
                L.append("")
                L.append("Stabilizers (sparsest basis): `" + "`, `".join(a["sparsest_basis"]) + "`  ")
                L.append(f"Weight enumerator of S: {a['weight_enumerator_S']}; of N(S): {a['weight_enumerator_N']}  ")
                L.append("Gates: `" + " ".join(f"{g[0]}({g[1]}{',' + str(g[2]) if g[2] >= 0 else ''})" for g in a["gates"]) + "`")
                L.append("")
        # per-setting analysis summaries
        L.append("### Analysis\n")
        L += analysis_tables(prof_runs, b)
        L.append("### Structural bias by training profile\n")
        L.append(
            "| trained on | seeds with a code | d_Z > d_X | d_X > d_Z | min-weight pure-Z : pure-X logicals (per seed) | "
            "pure-Z : pure-X stabilizer elements (per seed) | CSS | LC/H frame (non-CSS) | H gates (mean) | S gates (mean) | CNOTs (mean) | depth (mean) |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pname in PROFILES:
            lst = [r for r in prof_runs.get(pname, []) if r["best"]]
            if not lst:
                continue
            flags = [bias_flags(r["best"]) for r in lst]
            L.append(
                f"| {pname} | {len(lst)} | {sum(f['dz_gt_dx'] for f in flags)} | {sum(f['dx_gt_dz'] for f in flags)} | "
                + ", ".join(f"{f['min_z_logicals']}:{f['min_x_logicals']}" for f in flags) + " | "
                + ", ".join(f"{f['pure_z_elements']}:{f['pure_x_elements']}" for f in flags) + " | "
                f"{sum(f['css'] for f in flags)} | {sum(1 for f in flags if f['frame'] and not f['css'])} | "
                f"{np.mean([r['best']['circuit']['h_gates'] for r in lst]):.1f} | {np.mean([r['best']['circuit']['s_gates'] for r in lst]):.1f} | "
                f"{np.mean([r['best']['circuit']['two_qubit_gates'] for r in lst]):.1f} | {np.mean([r['best']['circuit']['depth'] for r in lst]):.1f} |"
            )
        L.append("")
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(L))
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Noise-aware PPO code discovery under three noise profiles.")
    ap.add_argument("--threads", type=int, default=_THREADS, help="CPU thread cap for torch and BLAS")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="results directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate", help="print the profile scales of a setting")
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p = sub.add_parser("run", help="train one (setting, profile, seed) cell")
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p.add_argument("--profile", choices=list(PROFILES), required=True)
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--budget", type=int, default=None, help="env steps (default: the setting's)")
    p.add_argument("--max-time", type=float, default=25 * 60)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("baselines", help="known codes and their encoders for a setting")
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p.add_argument("--polish-time", type=float, default=90.0, help="seconds of local search per code and profile")
    p.add_argument("--verbose", action="store_true")
    sub.add_parser("report", help="write REPORT.md and the figures from the results directory")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "calibrate":
        c = calibrate(SETTINGS[args.setting])
        print(json.dumps(c, indent=1))
        return 0
    if args.cmd == "run":
        setting = SETTINGS[args.setting]
        calib = calibrate(setting)
        (out / "runs").mkdir(exist_ok=True)
        for seed in args.seed:
            r = run_seed(
                setting, args.profile, seed, budget=args.budget, max_time_s=args.max_time,
                threads=args.threads, batch=args.batch, verbose=args.verbose, calib=calib,
            )
            name = run_name(args.setting, args.profile, seed)
            with open(out / "runs" / f"{name}.json", "w") as f:
                json.dump(r, f, indent=1, default=_json_default)
            write_history_csv(r, out / "runs" / f"{name}.csv")
            a = r["best"]
            line = (
                f"{name}: env_steps={r['env_steps']} wall={r['wall_time_s']}s episodes={r['episodes']} "
                f"successes={r['successful_episodes']} distinct={r['distinct_success_codes']}"
            )
            if a:
                line += (
                    f" best d={a['distance']} dX={a['distance_x']} dZ={a['distance_z']} w*={a['w_star']} css={a['is_css']} "
                    f"mass=" + "/".join(f"{a['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                )
            print(line, flush=True)
        return 0
    if args.cmd == "baselines":
        setting = SETTINGS[args.setting]
        b = run_baselines(setting, threads=args.threads, verbose=args.verbose, polish_time_s=args.polish_time)
        with open(out / f"baselines_{args.setting}.json", "w") as f:
            json.dump(b, f, indent=1, default=_json_default)
        for name, c in b["codes"].items():
            bb = c["encoders"]["best_by_profile"]
            print(
                f"{args.setting} {name}: d={c['distance']} dX={c['distance_x']} dZ={c['distance_z']} w*={c['w_star']} mass="
                + "/".join(f"{bb[ev]['undetected']:.4g}" if ev in bb else "none" for ev in PROFILES)
            )
        return 0
    if args.cmd == "report":
        path = write_report(out)
        print("wrote", path)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
