# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""PPO at GPU scale: the noise-aware 4x4 cells with the equivariant policy,
and the ``[[25, 1, 5]]`` rediscovery on the 5x5 lattice.

Two targets, both ``k = 1`` with H and CNOT on the edges of a square
lattice, both trained with ``PPOConfig(policy="auto")`` (the per-qubit
permutation-equivariant network of :mod:`autoqec_sim.agents.policies` at
``n >= 12``) and otherwise the ``PPOConfig`` defaults:

``noise4x4``
    The 4x4 setting of :mod:`autoqec_sim.experiments.noise_aware` again
    (``n = 16``, target ``d = 4``, ``square:4x4``, 50-gate episodes): the
    same reward (fractional distance term, padding penalty, and, on
    target-distance states, the circuit-level undetected fault mass under
    the training profile plus the LDPC term), the same three profiles
    (``board``, ``ion``, ``heron``) at the same calibrated scales, three
    seeds, 1024 parallel episodes and 20M env steps by default. In
    ``experiments/noise_aware/runs/REPORT.md`` every 4x4 cell failed with the flat
    MLP at 256 episodes and 16M steps. Everything the reward and the code
    analysis need is imported from ``noise_aware``; only the training loop
    and the device plumbing are here. The output keeps the field names of
    ``noise_aware.run_seed`` (``best``, ``modal``, ``mass`` under all three
    profiles) so the cross-evaluation matrix of the earlier report can be
    extended with these runs, against the known-code baselines at ``n =
    16`` already computed in ``experiments/noise_aware/runs/baselines_4x4.json``.
``surface5x5``
    ``n = 25``, target ``d = 5``, ``square:5x5``, 80-gate episodes, the
    level distance reward ``min(d, 5)/5 - 1`` plus the padding penalty
    (:class:`LevelReward`, the objective of ``experiments.scaling`` and
    ``experiments.topologies`` at ``d* = 5``), 2048 parallel episodes and
    30M env steps by default, three seeds, stop at the first success. A
    success is a terminal state of exact distance 5 with no weight-1
    stabilizer element; :func:`verify_code` replays its circuit, computes
    the exact distance (normalizer enumeration), the weight enumerators of
    ``S`` and ``N(S)``, ``w*`` (a bound above 14 generators), the CSS test,
    and whether the enumerators equal those of the rotated surface code
    ``[[25, 1, 5]]`` (the match criterion of the rediscovery regression:
    necessary for equivalence, not a proof of it).
``surface5x5_sparse``
    ``surface5x5`` plus the hard LDPC constraint ``w* <= 4`` of the
    environment, with the exact test of :class:`SparsityCheck` in place of
    ``ldpc.ldpc_feasible`` (whose greedy bound above 14 generators rejects
    the rotated surface code itself). ``--ldpc-mode mask`` (default): a
    gate that would take the stabilizer group above the bound becomes a
    no-op step, so every visited state is sparse (the ``mask`` mode of
    ``CodeDiscoveryEnv``); ``--ldpc-mode final``: the path is free and only
    a sparse target-distance state ends the episode, with a step penalty
    of -0.01 as the incentive. The penalty form of the constraint is not
    offered: at ``n = 12`` to ``16`` it trapped every run at distance 1
    (``experiments/scaling/REPORT.md``). Defaults 2048 episodes, 40M env
    steps, 150 min. Success is exact ``d = 5``, exact ``w* <= 4``, and no
    weight-1 stabilizer; the report also says whether the enumerators
    match the rotated surface code.

Device handling
---------------
``--device cuda`` puts two things on the GPU. (i) The policy network, its
forward passes over the rollout, and the PPO update (``PPOAgent(device=
"cuda")``: observations, masks, and rollout buffers are copied to the
device, actions and values back). (ii) The state reward: the exact
distance up to ``d* - 1``, the fraction of detected Paulis at the deciding
weight (the fractional term of the noise-aware reward), and the padding
penalty are evaluated by :class:`DistanceKernel` through
:mod:`autoqec_sim.backends.torch_backend` with one host-to-device copy of
the ``(batch, 2n, 2n + 1)`` tableau per environment step and three
``(batch,)`` arrays copied back (a fused equivalent of
``torch_backend.numpy_reward_adapter`` that moves the tableau once for
all terms and never materializes the full ``(batch, M)`` mask: the error
set is streamed in chunks of ``--chunk`` Paulis and only the per-element
counts are kept, which bounds the GPU intermediate to ``4 B_u chunk (n +
k)`` bytes). (iii) For ``surface5x5_sparse`` the exact LDPC feasibility
test of every step (syndrome product, sort, collision products, and the
GF(2) rank of :class:`SparsityCheck`) runs on the GPU as well, on a trial
copy of the tableau; the numpy version of the same test costs 0.6 s per
2048-element batch on the laptop and would dominate a CPU run.

What still runs on the CPU with ``--device cuda``, measured per run as
``reward_time_s`` (the reward, including its host-side terminal terms) and
``update_time_s`` (the PPO update) against ``wall_time_s``:

- the numpy environment ``CodeDiscoveryEnv``: gate application, action
  history, auto-reset, and the ``{-1, +1}`` observation encoding of
  ``EnvAdapter``;
- the terminal terms of the noise-aware reward: circuit-level fault
  propagation with stim and the exact ``w*`` (``noise_aware.
  NoiseAwareReward.terminal_terms``), evaluated only on states that
  reached the target distance and cached per gate list;
- the candidate tracker of the noise-aware target (one numpy distance
  evaluation and one GF(2) canonical form per finished episode);
- the verification of every recorded code: exact distance, weight
  enumerators, CSS test, sparsest basis, and the cross-evaluation of the
  best 4x4 code under the three profiles.

With ``--device cpu`` every component is the numpy path of the earlier
experiments (``PPOAgent`` on CPU torch, ``noise_aware.NoiseAwareReward``,
numpy ``reward.batched_distance``), so a CPU run is the earlier experiment
with the equivariant policy and a larger batch.

Files. Every run writes ``experiments/gpu_scale/runs/run_<target>[_<profile>]_seed
<k>.json`` and a log under ``experiments/gpu_scale/runs/logs/``; ``report`` writes
``experiments/gpu_scale/runs/REPORT.md`` from the JSON files. ``python -m
autoqec_sim.experiments.gpu_scale --help``; ``docs/gpu_run.md`` has the
pod commands.
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
    return 6


_THREADS = _threads_from_argv(sys.argv)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(_THREADS))

import numpy as np

from .. import graphs
from ..circuit_noise import circuit_depth
from ..code import StabilizerCode
from ..env import CodeDiscoveryEnv
from ..errors import paulis_of_weight, paulis_up_to_weight
from ..ldpc import _greedy_stats, batched_rank_packed, pack_rows, packed_weight, sparsest_basis_stats, unpack_rows
from ..regress import replay
from ..reward import batched_distance, code_rows, low_weight_stabilizer_penalty
from ..symplectic import symplectic_product, vec_to_pauli
from . import noise_aware as na
from .noise_aware import _fmt, _json_default, _mean_range

DEFAULT_OUT = Path("experiments/gpu_scale/runs")
DEFAULT_BASELINES = Path("experiments/noise_aware/runs/baselines_4x4.json")
PROFILES = na.PROFILES
SEEDS = (0, 1, 2)
THREADS = 6
LOG_EVERY = 25
CHUNK = {"cpu": 2048, "cuda": 8192}  # Paulis per streamed chunk of the torch distance kernel

TARGETS = {
    "noise4x4": {
        "description": "noise-aware 4x4 setting of experiments.noise_aware with policy=auto",
        "n": 16,
        "k": 1,
        "d": 4,
        "graph": "square:4x4",
        "gates": ("H", "CNOT"),
        "max_steps": 50,
        "batch": 1024,
        "steps": 20_000_000,
        "wall_min": 60.0,
        "profiles": PROFILES,
    },
    "surface5x5": {
        "description": "[[25,1,5]] rediscovery on square:5x5 with the level distance reward",
        "n": 25,
        "k": 1,
        "d": 5,
        "graph": "square:5x5",
        "gates": ("H", "CNOT"),
        "max_steps": 80,
        "batch": 2048,
        "steps": 30_000_000,
        "wall_min": 120.0,
        "profiles": (None,),
        "ldpc_w_max": None,
    },
    "surface5x5_sparse": {
        "description": "[[25,1,5]] rediscovery on square:5x5 under the hard LDPC constraint w* <= 4 (exact test)",
        "n": 25,
        "k": 1,
        "d": 5,
        "graph": "square:5x5",
        "gates": ("H", "CNOT"),
        "max_steps": 80,
        "batch": 2048,
        "steps": 40_000_000,
        "wall_min": 150.0,
        "profiles": (None,),
        "ldpc_w_max": 4,
    },
}
SURFACE_TARGETS = ("surface5x5", "surface5x5_sparse")
LDPC_MODES = ("mask", "final")
EXACT_W_STAR_MAX = 6  # exact w* by syndrome collisions up to this value; above it the greedy bound is reported


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------
def resolve_device(device: str):
    """``torch.device`` for ``"cpu"`` or ``"cuda"``; refuses ``cuda`` when
    torch has no CUDA device."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")
    return dev


def device_name(dev) -> str:
    import torch

    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    return platform.processor() or platform.machine()


def default_chunk(dev) -> int:
    return CHUNK["cuda" if dev.type == "cuda" else "cpu"]


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


class RunLog:
    """Appends timestamped lines to a file and, with ``verbose``, to stdout."""

    def __init__(self, path: Path | None, verbose: bool = False):
        self.path = path
        self.verbose = verbose
        self.f = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "a")  # noqa: SIM115 - long-lived log handle, closed by close()

    def __call__(self, msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        if self.f is not None:
            self.f.write(line + "\n")
            self.f.flush()
        if self.verbose:
            print(line, flush=True)

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


# --------------------------------------------------------------------------
# rewards on a torch device
# --------------------------------------------------------------------------
class DistanceKernel:
    """Exact distance capped at ``d_max``, the detected fraction of the
    Paulis at the deciding weight, and the padding penalty of a numpy
    tableau batch, computed on a torch device.

    ``__call__(tab) -> (d, frac, pad)`` with ``d`` ``(batch,)`` int64 (the
    first weight ``w < d_max`` with an undetected Pauli, else ``d_max``),
    ``frac`` ``(batch,)`` float64 (``1 - undetected/total`` over the
    Paulis of weight ``d`` for elements with ``d < d_max``, ``1.0``
    otherwise), and ``pad`` ``(batch,)`` float64 (minus the fraction of
    weight-1 Paulis in the stabilizer group). The numbers are those of
    ``reward.batched_distance``, the fractional term of
    ``noise_aware.NoiseAwareReward.distance_term``, and
    ``reward.low_weight_stabilizer_penalty(max_weight=1)``; the error sets
    live on the device as float32 and are streamed in chunks of ``chunk``
    Paulis, so the largest intermediate is ``4 B_u chunk (n + k)`` bytes
    for the ``B_u`` elements still undecided at that weight.
    """

    def __init__(self, n: int, k: int, d_max: int, device, chunk: int | None = None):
        import torch

        from ..backends import torch_backend as tb

        self.torch = torch
        self.tb = tb
        self.n, self.k, self.d_max = int(n), int(k), int(d_max)
        self.device = torch.device(device)
        self.chunk = int(chunk or default_chunk(self.device))
        self.errors = {
            w: torch.as_tensor(paulis_of_weight(self.n, w), device=self.device).to(torch.float32)
            for w in range(1, self.d_max)
        }
        e1, _ = paulis_up_to_weight(self.n, 1)
        self.weight1 = torch.as_tensor(e1, device=self.device).to(torch.float32)
        self.calls = 0
        self.top_weight_elements = 0  # elements evaluated at the most expensive weight

    def __call__(self, tab: np.ndarray):
        torch, tb, dev, k = self.torch, self.tb, self.device, self.k
        tabt = tb.from_numpy(tab, dev)
        batch = tabt.shape[0]
        d = torch.full((batch,), self.d_max, dtype=torch.long, device=dev)
        frac = torch.ones(batch, dtype=torch.float64, device=dev)
        undecided = torch.ones(batch, dtype=torch.bool, device=dev)
        for w in range(1, self.d_max):
            idx = torch.nonzero(undecided, as_tuple=True)[0]
            if idx.numel() == 0:
                break
            if w == self.d_max - 1:
                self.top_weight_elements += int(idx.numel())
            sub = tabt[idx]
            e = self.errors[w]
            cnt = torch.zeros(idx.numel(), dtype=torch.long, device=dev)
            for start in range(0, e.shape[0], self.chunk):
                cnt += tb.undetected_mask(sub, k, e[start : start + self.chunk]).sum(dim=1)
            hit = cnt > 0
            hidx = idx[hit]
            d[hidx] = w
            frac[hidx] = 1.0 - cnt[hit].to(torch.float64) / float(e.shape[0])
            undecided[hidx] = False
        pad = -tb.in_stabilizer_mask(tabt, k, self.weight1).to(torch.float64).mean(dim=1)
        self.calls += 1
        return d.cpu().numpy(), frac.cpu().numpy(), pad.cpu().numpy()


class TorchNoiseAwareReward(na.NoiseAwareReward):
    """``noise_aware.NoiseAwareReward`` with the distance, fractional, and
    padding terms evaluated by :class:`DistanceKernel` on a device; the
    terminal terms (fault mass, LDPC) are the parent's, on the host."""

    def __init__(self, env, setting, model, m0, device, chunk=None, **kw):
        super().__init__(env, setting, model, m0, **kw)
        self.kernel = DistanceKernel(setting.n, setting.k, setting.d, device, chunk)

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
            noise, ldpc = self.terminal_terms(tab[idx], self.env.gate_lists(succ))
            r[idx] += noise + ldpc
        self.last_success = succ
        return r


class LevelReward:
    """``min(d, d*)/d* - 1`` plus the padding penalty (the objective of
    ``experiments.scaling.LevelDistanceReward``), on numpy with a bounded
    error-set chunk (``device`` ``None`` or ``cpu``) or through
    :class:`DistanceKernel` on a device. ``last_d`` and ``last_pad`` hold
    the last evaluated batch."""

    def __init__(self, n: int, k: int, d_target: int, device=None, chunk: int | None = None):
        self.n, self.k, self.d_target = int(n), int(k), int(d_target)
        self.kernel = None
        if device is not None and str(device) != "cpu":
            self.kernel = DistanceKernel(n, k, d_target, device, chunk)
        self.chunk = int(chunk or CHUNK["cpu"])
        self.last_d = None
        self.last_pad = None

    def __call__(self, tab: np.ndarray, k: int) -> np.ndarray:
        if self.kernel is None:
            d = batched_distance(tab, k, self.d_target - 1, chunk=self.chunk)
            pad = low_weight_stabilizer_penalty(tab, k, 1)
        else:
            d, _, pad = self.kernel(tab)
        self.last_d = d
        self.last_pad = pad
        return np.minimum(d, self.d_target) / float(self.d_target) - 1.0 + pad


class SurfaceTrackingEnv(CodeDiscoveryEnv):
    """``CodeDiscoveryEnv`` over a :class:`LevelReward` that counts terminal
    distances and keeps the best terminal state (highest state reward,
    then fewer gates) with its gate list."""

    def __init__(self, level: LevelReward, *args, **kw):
        self.level = level
        self.terminal_d = np.zeros(level.d_target + 1, dtype=np.int64)
        self.window_d = np.zeros(level.d_target + 1, dtype=np.int64)
        self.best_reward = -np.inf
        self.best_steps = None
        self.best_tableau = None
        self.best_gates = None
        super().__init__(*args, **kw)

    def reset(self, seed=None, where=None):
        if where is None:
            return super().reset(seed, where)
        keep = (self.level.last_d, self.level.last_pad)
        out = super().reset(seed, where)
        self.level.last_d, self.level.last_pad = keep
        return out

    def flush_window(self) -> list[int]:
        out = self.window_d.tolist()
        self.window_d[:] = 0
        return out

    def step(self, actions):
        out = super().step(actions)
        info = out[4]
        if "final_tableau" in info:
            done = out[2] | out[3]
            d = self.level.last_d
            if d is not None:
                counts = np.bincount(np.minimum(d[done], self.level.d_target), minlength=self.level.d_target + 1)
                self.terminal_d += counts
                self.window_d += counts
            r = info["final_state_reward"]
            steps = info["final_steps"]
            i = np.lexsort((steps, -r))[0]
            if r[i] > self.best_reward or (r[i] == self.best_reward and steps[i] < self.best_steps):
                self.best_reward = float(r[i])
                self.best_steps = int(steps[i])
                self.best_tableau = info["final_tableau"][i].copy()
                self.best_gates = [self.action_table.entries[a] for a in info["final_history"][i] if a >= 0]
        return out


# --------------------------------------------------------------------------
# exact LDPC feasibility: w* <= w_max by syndrome collisions
# --------------------------------------------------------------------------
class SparsityCheck:
    """Exact test of ``w* <= w_max`` for every tableau of a batch, at any
    generator count with ``2n <= 64``.

    ``ldpc.ldpc_feasible`` is exact only up to 14 generators and falls back
    to the greedy pairwise-reduction bound above that; at ``n = 25`` (24
    generators) the bound calls the rotated surface code itself infeasible
    (greedy ``w* = 6``, true ``w* = 4``), so a hard constraint built on it
    would exclude the target. This test is exact: an element of ``S`` of
    weight at most ``w_max`` is a product ``a b`` of two Paulis of weight
    at most ``h = ceil(w_max / 2)`` (disjoint supports; ``b = I`` allowed),
    and ``a b`` lies in ``S`` iff ``a`` and ``b`` have the same syndrome
    against the ``n + k`` code rows (stabilizers and logicals; an element
    commuting with all of them is in ``S``). So the low-weight elements of
    ``S`` are read off the syndrome collisions among the ``M`` Paulis of
    weight at most ``h`` (``M = 2776`` at ``n = 25``, ``h = 2``): one
    symplectic product ``(batch, M, n + k)``, one sort per element, the
    products ``p_1 p_j`` of every collision class (they span the same
    space as all pairwise products), a weight filter to ``w_max``, and a
    batched GF(2) rank on packed rows. ``w* <= w_max`` iff the rank equals
    ``n - k``. The syndrome product runs on ``device`` when given (torch),
    the rest in numpy.
    """

    def __init__(self, n: int, k: int, w_max: int = 4, device=None, batch_chunk: int = 256):
        self.n, self.k, self.w_max = int(n), int(k), int(w_max)
        if 2 * self.n > 64:
            raise ValueError("SparsityCheck needs 2n <= 64 (packed rows)")
        half = (self.w_max + 1) // 2
        e, _ = paulis_up_to_weight(self.n, half)
        self.e = np.concatenate([np.zeros((1, 2 * self.n), dtype=np.uint8), e], axis=0)  # identity first
        self.packed = pack_rows(self.e)
        self.pow2 = (np.int64(1) << np.arange(self.n + self.k, dtype=np.int64)).astype(np.int64)
        self.batch_chunk = int(batch_chunk)
        self.torch = None
        if device is not None:
            import torch

            from ..backends import torch_backend as tb

            self.torch, self.tb = torch, tb
            self.device = torch.device(device)
            self.e_t = torch.as_tensor(self.e, device=self.device)
            self.packed_t = torch.as_tensor(self.packed.astype(np.int64), device=self.device)
            self.pow2_lo = torch.as_tensor((2.0 ** np.arange(13)).astype(np.float32), device=self.device)
        self.calls = 0
        self.elements = 0
        self.seconds = 0.0

    def syndromes(self, tab: np.ndarray) -> np.ndarray:
        """``(batch, M)`` int64: the commutation pattern of every low-weight
        Pauli against the code rows, packed as an integer."""
        b = tab.shape[0]
        out = np.empty((b, self.e.shape[0]), dtype=np.int64)
        for s in range(0, b, self.batch_chunk):
            sub = tab[s : s + self.batch_chunk]
            prod = symplectic_product(self.e[None], code_rows(sub, self.k))  # (b, M, n+k) uint8
            out[s : s + self.batch_chunk] = prod.astype(np.int64) @ self.pow2
        return out

    # ---- torch path (any torch device; "cpu" is used by the tests as a reference)
    def _syndromes_torch(self, tabt):
        torch, tb = self.torch, self.tb
        outs = []
        for s in range(0, tabt.shape[0], self.batch_chunk):
            rows = tb.code_rows(tabt[s : s + self.batch_chunk], self.k)
            prod = tb.symplectic_product(self.e_t.unsqueeze(0), rows).to(torch.float32)  # (b, M, n+k) in {0, 1}
            # integer matmul is not available on cuda: two float32 halves (exact below 2^13 each)
            lo = prod[..., :13] @ self.pow2_lo[: min(13, prod.shape[-1])]
            hi = prod[..., 13:] @ self.pow2_lo[: max(prod.shape[-1] - 13, 0)] if prod.shape[-1] > 13 else torch.zeros_like(lo)
            outs.append(lo.to(torch.int64) + (hi.to(torch.int64) << 13))
        return torch.cat(outs, dim=0)

    def _low_weight_elements_torch(self, syn):
        torch = self.torch
        b, m = syn.shape
        shift = max(1, (m - 1).bit_length())
        pos = torch.arange(m, device=syn.device).expand(b, m)
        key, _ = torch.sort((syn << shift) | pos, dim=1)
        s = key >> shift
        order = key & ((1 << shift) - 1)
        is_start = torch.ones((b, m), dtype=torch.bool, device=syn.device)
        is_start[:, 1:] = s[:, 1:] != s[:, :-1]
        start = torch.cummax(torch.where(is_start, pos, torch.zeros_like(pos)), dim=1).values
        p = self.packed_t[order]
        rows = p ^ torch.gather(p, 1, start)
        rows = torch.where(is_start, torch.zeros_like(rows), rows)
        rows, _ = torch.sort(rows, dim=1)
        keep = rows != 0
        keep[:, 1:] &= rows[:, 1:] != rows[:, :-1]
        r_max = int(keep.sum(dim=1).max()) if b else 0
        if r_max == 0:
            return rows[:, :0], keep[:, :0]
        comp = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :r_max]
        return torch.gather(rows, 1, comp), torch.gather(keep, 1, comp)

    def _rank_torch(self, rows, mask, ncols: int):
        """Port of ``ldpc.batched_rank_packed`` (rank only) on int64 rows."""
        torch = self.torch
        rows = torch.where(mask, rows, torch.zeros_like(rows))
        b, r = rows.shape
        used = torch.zeros((b, r), dtype=torch.bool, device=rows.device)
        rank = torch.zeros(b, dtype=torch.int64, device=rows.device)
        ar = torch.arange(b, device=rows.device)
        for c in range(ncols):
            bit = (((rows >> c) & 1) != 0) & ~used
            piv = bit.to(torch.int8).argmax(dim=1)
            has = bit[ar, piv]
            if not bool(has.any()):
                continue
            pv = torch.where(has, rows[ar, piv], torch.zeros_like(rank))
            target = bit.clone()
            target[ar, piv] = False
            rows = rows ^ torch.where(target, pv[:, None], torch.zeros_like(rows))
            used[ar[has], piv[has]] = True
            rank += has.to(torch.int64)
        return rank

    def _feasible_torch(self, tab: np.ndarray) -> np.ndarray:
        m = self.n - self.k
        tabt = self.tb.from_numpy(tab, self.device)
        rows, mask = self._low_weight_elements_torch(self._syndromes_torch(tabt))
        if rows.shape[1] == 0:
            return np.full(tab.shape[0], m == 0, dtype=bool)
        return (self._rank_torch(rows, mask, 2 * self.n) == m).cpu().numpy()

    def low_weight_elements(self, syn: np.ndarray, w_max: int | None = None):
        """Distinct elements of ``S`` of weight at most ``w_max`` per batch
        element as packed rows ``(batch, R)`` with a validity mask."""
        w_max = self.w_max if w_max is None else int(w_max)
        b, m = syn.shape
        order = np.argsort(syn, axis=1, kind="stable")
        s = np.take_along_axis(syn, order, axis=1)
        is_start = np.ones((b, m), dtype=bool)
        is_start[:, 1:] = s[:, 1:] != s[:, :-1]
        pos = np.broadcast_to(np.arange(m), (b, m))
        start = np.maximum.accumulate(np.where(is_start, pos, 0), axis=1)
        p = self.packed[order]
        rows = p ^ np.take_along_axis(p, start, axis=1)  # p_j * p_first of its collision class
        mask = ~is_start & (packed_weight(rows, self.n) <= w_max)
        # drop duplicates (one element can arise from several decompositions)
        rows = np.where(mask, rows, np.uint64(0))
        rows.sort(axis=1)
        keep = rows != 0
        keep[:, 1:] &= rows[:, 1:] != rows[:, :-1]
        r_max = int(keep.sum(axis=1).max()) if b else 0
        if r_max == 0:
            return np.zeros((b, 0), dtype=np.uint64), np.zeros((b, 0), dtype=bool)
        comp = np.argsort(~keep, axis=1, kind="stable")[:, :r_max]
        return np.take_along_axis(rows, comp, axis=1), np.take_along_axis(keep, comp, axis=1)

    def feasible(self, tab: np.ndarray, w_max: int | None = None) -> np.ndarray:
        """``(batch,)`` bool: ``w*(tab) <= w_max`` (default the check's)."""
        t0 = time.perf_counter()
        w_max = self.w_max if w_max is None else int(w_max)
        if w_max > 2 * ((self.w_max + 1) // 2):
            raise ValueError("w_max above the enumerated half weight")
        m = self.n - self.k
        if self.torch is not None and w_max == 2 * ((self.w_max + 1) // 2):
            out = self._feasible_torch(tab)  # even w_max: every pair product qualifies, no weight filter needed
        else:
            rows, mask = self.low_weight_elements(self.syndromes(tab), w_max)
            if rows.shape[1] == 0:
                out = np.zeros(tab.shape[0], dtype=bool) if m > 0 else np.ones(tab.shape[0], dtype=bool)
            else:
                rank, _ = batched_rank_packed(rows, mask, 2 * self.n)
                out = rank == m
        self.calls += 1
        self.elements += tab.shape[0]
        self.seconds += time.perf_counter() - t0
        return out


def exact_w_star(tab_b: np.ndarray, k: int, max_w: int = EXACT_W_STAR_MAX) -> tuple[int, bool]:
    """``(w*, exact)`` of one tableau ``(2n, 2n + 1)``: exact by
    :class:`SparsityCheck` when ``w* <= max_w``, else the greedy bound of
    ``ldpc`` with ``exact = False``."""
    n = (tab_b.shape[-1] - 1) // 2
    m = n - k
    if m <= 14:
        st = sparsest_basis_stats(tab_b[None], k)
        return int(st["w_star"][0]), True
    tab = tab_b[None]
    for half in range(1, (max_w + 1) // 2 + 1):
        check = SparsityCheck(n, k, 2 * half)
        syn = check.syndromes(tab)
        for w in (2 * half - 1, 2 * half):
            if w > max_w or w < 1:
                continue
            rows, mask = check.low_weight_elements(syn, w)
            if rows.shape[1] and batched_rank_packed(rows, mask, 2 * n)[0][0] == m:
                return w, True
    gens = pack_rows(tab[:, n + k :, : 2 * n])
    return int(_greedy_stats(gens, n)[0][0]), False


class SparseLevelEnv(SurfaceTrackingEnv):
    """:class:`SurfaceTrackingEnv` under the hard constraint ``w* <= w_max``
    tested exactly by :class:`SparsityCheck`.

    ``mode = "mask"`` (the ``mask`` mode of ``CodeDiscoveryEnv`` with the
    exact test in place of ``ldpc.ldpc_feasible``): the chosen gate is
    applied to a trial copy, and an element whose new group would violate
    the bound takes a no-op step instead (history ``-1``, the step counts,
    the reward is that of the unchanged state), so every visited state is
    sparse and a target-distance state is a success as soon as it
    appears. ``mode = "final"``: the path is free and only a final state
    that meets the bound counts as a success (the ``final`` mode of the
    environment); a dense target-distance state does not end the episode.
    The penalty form of the constraint is not offered: at ``n = 12`` to
    ``16`` it trapped every run at distance 1 (``experiments/scaling/
    REPORT.md``, LDPC section).
    """

    def __init__(self, level: LevelReward, check: SparsityCheck, mode: str, *args, **kw):
        if mode not in LDPC_MODES:
            raise ValueError(f"mode must be one of {LDPC_MODES}")
        self.check = check
        self.mode = mode
        self.violations = 0
        self.window_violations = 0
        self.window_actions = 0
        super().__init__(level, *args, **kw)
        if mode == "final":
            self.success_fn = self._final_success

    def _final_success(self, tab, k, r):
        ok = r >= self.success_threshold
        idx = np.flatnonzero(ok)
        if idx.size:
            ok[idx] = self.check.feasible(tab[idx])
        return ok

    def flush_violations(self) -> tuple[int, int]:
        out = (self.window_violations, self.window_actions)
        self.window_violations = 0
        self.window_actions = 0
        return out

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        violated = np.zeros(self.batch_size, dtype=bool)
        if self.mode == "mask":
            active = (self.steps < self.max_steps) & (actions >= 0)
            trial = self.tab.copy()
            self.action_table.apply(trial, actions, where=active)
            violated = active & ~self.check.feasible(trial)
            actions = np.where(violated, -1, actions)
            self.violations += int(violated.sum())
            self.window_violations += int(violated.sum())
            self.window_actions += int(active.sum())
        out = super().step(actions)
        out[4]["ldpc_violation"] = violated
        return out


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def gate_stats(gates, n: int) -> dict:
    gates = [tuple(g) for g in gates]
    return {
        "gate_count": len(gates),
        "cnot_count": sum(g[0] == "CNOT" for g in gates),
        "h_count": sum(g[0] == "H" for g in gates),
        "depth": circuit_depth(gates, n),
    }


def verify_code(gates, n: int, k: int, d_target: int, reference: list[str] | None, reference_name: str = "", w_max: int | None = None) -> dict:
    """Replay ``gates`` and verify the code independently of the reward:
    exact distance (``StabilizerCode.distance_exact``, forced above 20
    qubits), weight enumerators of ``S`` and ``N(S)``, exact ``w*`` up to
    ``EXACT_W_STAR_MAX`` (:func:`exact_w_star`; the greedy bound above), the
    greedy sparsest basis (a basis of maximum weight equal to the greedy
    bound, not necessarily ``w*``), CSS test, the enumerator comparison
    with ``reference`` (stabilizer strings), and, with ``w_max``, whether
    ``w* <= w_max``."""
    t0 = time.perf_counter()
    gates = [tuple(g) for g in gates]
    code = replay(gates, n)
    tab = code.to_tableau()
    d = int(code.distance_exact(force=True))
    a, b = code.weight_enumerators()
    st = sparsest_basis_stats(tab, k)
    w_star, w_exact = exact_w_star(tab[0], k)
    basis = [vec_to_pauli(r) for r in unpack_rows(st["basis"][0], 2 * n)]
    sparse_ok = w_max is None or w_star <= w_max
    out = {
        "n": n,
        "k": k,
        "d_target": d_target,
        "w_max": w_max,
        "distance_exact": d,
        "no_weight1_stabilizer": bool(a[1] == 0),
        "sparse": bool(sparse_ok),
        "meets_target": bool(d >= d_target and a[1] == 0 and sparse_ok),
        "is_css": bool(code.is_css()),
        "stabilizers": code.stabilizer_strings(),
        "sparsest_basis": basis,
        "sparsest_basis_is_greedy": True,
        "w_star": int(w_star),
        "w_star_exact": bool(w_exact),
        "max_qubit_degree": int(st["max_degree"][0]),
        "weight_enumerator_S": a.tolist(),
        "weight_enumerator_N": b.tolist(),
        "gates": [list(g) for g in gates],
        **gate_stats(gates, n),
    }
    if reference is not None:
        ref = StabilizerCode.from_stabilizers(reference)
        ra, rb = ref.weight_enumerators()
        out["reference"] = reference_name
        out["reference_weight_enumerator_S"] = ra.tolist()
        out["reference_weight_enumerator_N"] = rb.tolist()
        out["enumerators_match_reference"] = bool(np.array_equal(a, ra) and np.array_equal(b, rb))
    out["verify_time_s"] = round(time.perf_counter() - t0, 1)
    return out


def describe_state(tab: np.ndarray, gates, k: int, d_cap: int) -> dict:
    """Light description of a terminal state that is not a success: the
    distance capped at ``d_cap`` (no normalizer enumeration), the greedy
    ``w*`` bound, CSS test, stabilizers, and gate statistics."""
    n = (tab.shape[-1] - 1) // 2
    code = StabilizerCode.from_tableau(tab[None], k)
    st = sparsest_basis_stats(tab[None], k)
    w_star, w_exact = exact_w_star(tab, k)
    d = int(batched_distance(tab[None], k, d_cap - 1, chunk=CHUNK["cpu"])[0])
    return {
        "n": n,
        "k": k,
        "distance_capped": d,
        "distance_cap": d_cap,
        "padding_penalty": float(low_weight_stabilizer_penalty(tab[None], k, 1)[0]),
        "is_css": bool(code.is_css()),
        "stabilizers": code.stabilizer_strings(),
        "w_star": int(w_star),
        "w_star_exact": bool(w_exact),
        "max_qubit_degree": int(st["max_degree"][0]),
        "gates": [list(g) for g in gates],
        **gate_stats(gates, n),
    }


# --------------------------------------------------------------------------
# common run plumbing
# --------------------------------------------------------------------------
def _run_header(target: str, profile, seed: int, dev, batch: int, steps: int, wall_s: float, threads: int, chunk: int) -> dict:
    import torch

    return {
        "kind": "run",
        "target": target,
        "target_description": TARGETS[target]["description"],
        "profile": profile,
        "seed": seed,
        "device": str(dev),
        "device_name": device_name(dev),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if dev.type == "cuda" else None,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "batch": batch,
        "budget_env_steps": steps,
        "wall_cap_s": wall_s,
        "threads": threads,
        "chunk": chunk,
    }


def _first_success(agent, res) -> dict | None:
    if res.first_success is None:
        return None
    fc = res.first_success
    return {
        "env_steps": fc.env_steps,
        "episodes": fc.episodes,
        "update": fc.update,
        "wall_time_s": next((h["wall_time_s"] for h in agent.history if h["env_steps"] >= fc.env_steps), None),
        "gate_count": len(fc.gates),
    }


def _timing(agent, res, reward_timer: Timed, update_timer: Timed, kernel: DistanceKernel | None) -> dict:
    wall = res.wall_time_s
    out = {
        "env_steps": res.env_steps,
        "episodes": res.episodes,
        "updates": res.updates,
        "wall_time_s": round(wall, 1),
        "env_steps_per_s": round(res.env_steps / wall, 1) if wall > 0 else None,
        "reward_time_s": round(reward_timer.seconds, 1),
        "reward_calls": reward_timer.calls,
        "update_time_s": round(update_timer.seconds, 1),
        "other_time_s": round(max(wall - reward_timer.seconds - update_timer.seconds, 0.0), 1),
        "hit_wall_cap": bool(res.env_steps < agent_budget(agent) and wall >= agent.wall_cap_s),
    }
    if kernel is not None:
        out["kernel_calls"] = kernel.calls
        out["kernel_top_weight_elements"] = kernel.top_weight_elements
    return out


def agent_budget(agent) -> int:
    return getattr(agent, "budget_env_steps", 0)


def _make_agent(env, seed: int, dev, threads: int, steps: int, wall_s: float):
    from ..agents.ppo import PPOAgent, PPOConfig

    config = PPOConfig(policy="auto", max_threads=threads)
    agent = PPOAgent(env, config=config, seed=seed, device=dev)
    agent.budget_env_steps = int(steps)
    agent.wall_cap_s = float(wall_s)
    update_timer = Timed(agent.update)
    agent.update = update_timer
    return agent, config, update_timer


# --------------------------------------------------------------------------
# target A: noise-aware 4x4
# --------------------------------------------------------------------------
def reward_description(setting) -> dict:
    return {
        "form": (
            "(d - 1 + f)/d* - 1 - q_1 + [d >= d*, q_1 = 0](-2 m / M_0 - 0.25 max(0, w* - 4)), "
            "f = detected fraction of weight-d Paulis, q_1 = fraction of weight-1 Paulis in S"
            if setting.fractional
            else "min(d, d*)/d* - 1 - q_1 + [d >= d*, q_1 = 0](-2 m / M_0 - 0.25 max(0, w* - 4)), q_1 = fraction of weight-1 Paulis in S"
        ),
        "padding_penalty": "reward.low_weight_stabilizer_penalty(max_weight=1); success requires no weight-1 stabilizer element",
        "fractional_distance": setting.fractional,
        "noise_weight": na.NOISE_WEIGHT,
        "w_max": na.W_MAX,
        "ldpc_scale": na.LDPC_SCALE,
        "source": "autoqec_sim.experiments.noise_aware.NoiseAwareReward",
    }


def run_noise4x4(
    profile_name: str,
    seed: int,
    device: str = "cpu",
    batch: int | None = None,
    steps: int | None = None,
    wall_min: float | None = None,
    threads: int = THREADS,
    chunk: int | None = None,
    verbose: bool = False,
    log: RunLog | None = None,
    calib: dict | None = None,
    calib_count: int = na.CALIBRATION_CIRCUITS,
    log_every: int = LOG_EVERY,
) -> dict:
    """One (profile, seed) cell of the noise-aware 4x4 setting with the
    equivariant policy; trains the full budget (no stop at success, as in
    ``noise_aware.run_seed``) and analyses the best and the modal code."""
    spec = TARGETS["noise4x4"]
    setting = na.SETTINGS["4x4"]
    batch = int(batch or spec["batch"])
    steps = int(steps or spec["steps"])
    wall_s = 60.0 * float(spec["wall_min"] if wall_min is None else wall_min)
    dev = resolve_device(device)
    chunk = int(chunk or default_chunk(dev))
    log = log or RunLog(None, verbose)
    log(f"noise4x4 {profile_name} seed {seed}: device {dev} ({device_name(dev)}), batch {batch}, budget {steps:,} env steps, wall cap {wall_s / 60:.0f} min")

    t_setup = time.perf_counter()
    calib = calib or na.calibrate(setting, count=calib_count)
    env = na.make_env(setting, seed, batch)
    models = na.calibrated_models(setting, calib, env.graph)
    if dev.type == "cpu":
        reward = na.NoiseAwareReward(env, setting, models[profile_name], calib["m0"])
        kernel = None
    else:
        reward = TorchNoiseAwareReward(env, setting, models[profile_name], calib["m0"], device=dev, chunk=chunk)
        kernel = reward.kernel
    reward_timer = Timed(reward)
    env.reward_fn = reward_timer
    env.success_fn = reward.success_fn
    tracker = na.CandidateTracker(setting.d, fractional=setting.fractional)
    env.tracker = tracker
    env.reset()
    agent, config, update_timer = _make_agent(env, seed, dev, threads, steps, wall_s)
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
    timing = _timing(agent, res, reward_timer, update_timer, kernel)
    log(f"  trained: {timing['env_steps']:,} env steps in {timing['wall_time_s']} s ({timing['env_steps_per_s']:,} steps/s); reward {timing['reward_time_s']} s, update {timing['update_time_s']} s")

    # best code of the seed (the selection of noise_aware.run_seed)
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
            f"css={analysis['is_css']} mass board/ion/heron=" + "/".join(f"{analysis['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
            + f" ({analysis['circuit']['gates']} gates); analysis {time.perf_counter() - t1:.1f} s"
        )
    return {
        **_run_header("noise4x4", profile_name, seed, dev, batch, steps, wall_s, threads, chunk),
        "setting": setting.to_dict(),
        "profile_description": na.profile_description(profile_name),
        "scale": calib["profiles"][profile_name]["scale"],
        "m0": calib["m0"],
        "calibration": calib,
        "reward": reward_description(setting),
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
        "first_success": _first_success(agent, res),
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
# targets B and C: surface 5x5, free and under the hard LDPC constraint
# --------------------------------------------------------------------------
def run_surface(
    target: str,
    seed: int,
    device: str = "cpu",
    batch: int | None = None,
    steps: int | None = None,
    wall_min: float | None = None,
    threads: int = THREADS,
    chunk: int | None = None,
    max_steps: int | None = None,
    ldpc_mode: str = "mask",
    step_penalty: float | None = None,
    verbose: bool = False,
    log: RunLog | None = None,
    log_every: int = LOG_EVERY,
) -> dict:
    """One seed of the ``[[25, 1, 5]]`` rediscovery (``surface5x5``) or of
    the same search under the hard constraint ``w* <= 4``
    (``surface5x5_sparse``, :class:`SparseLevelEnv` with ``ldpc_mode``);
    stops at the first success, the step budget, or the wall cap, and
    verifies the code. ``step_penalty`` (added to every step's reward)
    defaults to 0 in ``mask`` mode and to -0.01 in ``final`` mode, where a
    dense target-distance state does not end the episode and the penalty
    is the only term that prefers a sparse one."""
    if target not in SURFACE_TARGETS:
        raise ValueError(f"target must be one of {SURFACE_TARGETS}")
    spec = TARGETS[target]
    n, k, d = spec["n"], spec["k"], spec["d"]
    w_max = spec["ldpc_w_max"]
    batch = int(batch or spec["batch"])
    steps = int(steps or spec["steps"])
    max_steps = int(max_steps or spec["max_steps"])
    wall_s = 60.0 * float(spec["wall_min"] if wall_min is None else wall_min)
    dev = resolve_device(device)
    chunk = int(chunk or default_chunk(dev))
    if step_penalty is None:
        step_penalty = -0.01 if (w_max is not None and ldpc_mode == "final") else 0.0
    log = log or RunLog(None, verbose)
    constraint = f", hard LDPC w* <= {w_max} ({ldpc_mode} mode, exact test), step penalty {step_penalty}" if w_max else ""
    log(f"{target} seed {seed}: device {dev} ({device_name(dev)}), batch {batch}, budget {steps:,} env steps, episode cap {max_steps}, wall cap {wall_s / 60:.0f} min{constraint}")

    t_setup = time.perf_counter()
    graph = graphs.from_spec(spec["graph"])
    level = LevelReward(n, k, d, device=dev if dev.type == "cuda" else None, chunk=chunk)
    reward_timer = Timed(level)
    env_kw = dict(
        n=n, k=k, graph=graph, batch_size=batch, max_steps=max_steps, gates=spec["gates"],
        reward_fn=reward_timer, seed=seed, auto_reset=True, step_penalty=step_penalty,
    )
    check = None
    if w_max is None:
        env = SurfaceTrackingEnv(level, **env_kw)
    else:
        check = SparsityCheck(n, k, w_max, device=dev if dev.type == "cuda" else None)
        env = SparseLevelEnv(level, check, ldpc_mode, **env_kw)
    agent, config, update_timer = _make_agent(env, seed, dev, threads, steps, wall_s)
    log(f"  policy {agent.policy}, {agent.num_parameters():,} parameters, {env.num_actions} actions, setup {time.perf_counter() - t_setup:.1f} s")
    max_d_seen = 0

    def cb(agent, new):
        nonlocal max_d_seen
        rec = agent.history[-1]
        window = env.flush_window()
        rec["terminal_d"] = window
        if any(window[1:]):
            max_d_seen = max(max_d_seen, max(i for i, c in enumerate(window) if c > 0))
        rec["max_terminal_d_seen"] = max_d_seen
        rec["best_terminal_reward"] = env.best_reward if env.best_reward > -np.inf else None
        extra = ""
        if check is not None:
            viol, acts = env.flush_violations()
            rec["ldpc_violation_frac"] = round(viol / acts, 4) if acts else None
            rec["ldpc_check_time_s"] = round(check.seconds, 1)
            extra = f" viol {rec['ldpc_violation_frac']}" if acts else ""
        if agent.updates % log_every == 0 or new:
            log(
                f"  upd {rec['update']:5d} steps {rec['env_steps']:11,d} r {rec['mean_reward']:+.4f} ent {rec['entropy']:.3f} "
                f"term_d {window} best {env.best_reward:+.3f}{extra} t {rec['wall_time_s']:.0f}s ({rec['env_steps'] / max(rec['wall_time_s'], 1e-9):,.0f} steps/s)"
            )
        return False

    res = agent.train(steps, max_time_s=wall_s, stop_on_success=True, callback=cb, verbose=False)
    timing = _timing(agent, res, reward_timer, update_timer, level.kernel)
    if check is not None:
        timing["ldpc_check_time_s"] = round(check.seconds, 1)
        timing["ldpc_check_calls"] = check.calls
        timing["ldpc_violations"] = env.violations
        timing["other_time_s"] = round(max(timing["other_time_s"] - check.seconds, 0.0), 1)
    log(
        f"  trained: {timing['env_steps']:,} env steps in {timing['wall_time_s']} s ({timing['env_steps_per_s']:,} steps/s); "
        f"reward {timing['reward_time_s']} s, update {timing['update_time_s']} s"
        + (f", LDPC check {timing['ldpc_check_time_s']} s ({env.violations:,} reverted actions)" if check is not None else "")
        + f"; success {res.success}"
    )

    t1 = time.perf_counter()
    reference = na.rotated_surface(5)
    code = None
    best_state = None
    if res.first_success is not None:
        # the shortest encoder among the successes of the stopping update
        fc = min(agent.successes, key=lambda c: len(c.gates))
        code = verify_code(fc.gates, n, k, d, reference, "rotated surface [[25,1,5]] (noise_aware.rotated_surface(5))", w_max=w_max)
        code["source"] = f"shortest of {len(agent.successes)} successful episodes of update {fc.update + 1}"
        log(
            f"  code: d={code['distance_exact']} no_w1={code['no_weight1_stabilizer']} w*={code['w_star']}{'' if code['w_star_exact'] else ' (bound)'} "
            f"sparse={code['sparse']} css={code['is_css']} rotated_surface_enumerators={code['enumerators_match_reference']} {code['gate_count']} gates; verify {code['verify_time_s']} s"
        )
    elif env.best_tableau is not None:
        best_state = describe_state(env.best_tableau, env.best_gates, k, d)
        best_state["source"] = "best terminal state (no success)"
        log(f"  best terminal state: d={best_state['distance_capped']} (cap {d}) w*={best_state['w_star']}{'' if best_state['w_star_exact'] else ' (bound)'} padding {best_state['padding_penalty']:.3f}")
    return {
        **_run_header(target, None, seed, dev, batch, steps, wall_s, threads, chunk),
        "n": n,
        "k": k,
        "d_target": d,
        "graph": spec["graph"],
        "gates": list(spec["gates"]),
        "max_steps": max_steps,
        "step_penalty": step_penalty,
        "ldpc": None
        if w_max is None
        else {
            "w_max": w_max,
            "mode": ldpc_mode,
            "test": "gpu_scale.SparsityCheck (exact w* <= w_max by syndrome collisions of weight <= 2 Paulis)",
            "device": str(check.device) if check.torch is not None else "cpu (numpy)",
        },
        "reward": {
            "form": "min(d, 5)/5 - 1 - q_1 + step_penalty, q_1 = fraction of weight-1 Paulis in S; success iff the state reward is 0"
            + (f" and w* <= {w_max}" if w_max else ""),
            "source": "autoqec_sim.experiments.gpu_scale.LevelReward (experiments.scaling.LevelDistanceReward at d* = 5)",
        },
        "ppo_config": dict(config.to_dict(), policy_resolved=agent.policy),
        "num_parameters": res.num_parameters,
        "num_actions": env.num_actions,
        **timing,
        "success": bool(res.success and code is not None and code["meets_target"]),
        "successes": len(agent.successes),
        "first_success": _first_success(agent, res),
        "terminal_d_counts": env.terminal_d.tolist(),
        "max_terminal_d_seen": max_d_seen,
        "code": code,
        "best_state": best_state,
        "analysis_time_s": round(time.perf_counter() - t1, 1),
        "history": res.history,
    }


def run_surface5x5(seed: int, **kw) -> dict:
    return run_surface("surface5x5", seed, **kw)


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def run_name(target: str, profile, seed: int) -> str:
    return f"run_{target}_{profile}_seed{seed}" if profile else f"run_{target}_seed{seed}"


def run_path(out_dir: Path, target: str, profile, seed: int) -> Path:
    return out_dir / (run_name(target, profile, seed) + ".json")


def log_path(out_dir: Path, target: str, profile, seed: int) -> Path:
    return out_dir / "logs" / (run_name(target, profile, seed) + ".log")


def write(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=_json_default)
    print("wrote", path, flush=True)


EXTRA_DIRS = ("final_mode",)  # subdirectories of out_dir that `report` also reads by default


def load_all(out_dir: Path, extra_dirs=None) -> list[dict]:
    """Every ``run_*.json`` of ``out_dir`` plus those of ``extra_dirs``
    (paths, relative to ``out_dir`` unless absolute; default the
    ``EXTRA_DIRS`` subdirectories that exist, where a second mode of a
    target was run with its own ``--out-dir``). ``_file`` is the path
    relative to ``out_dir``."""
    dirs = [out_dir]
    extra = EXTRA_DIRS if extra_dirs is None else extra_dirs
    for e in extra:
        d = Path(e) if Path(e).is_absolute() else out_dir / e
        if d.exists() and d != out_dir:
            dirs.append(d)
    recs = []
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            try:
                with open(p) as f:
                    r = json.load(f)
            except json.JSONDecodeError:
                print(f"skipping unreadable {p} (still being written?)")
                continue
            try:
                r["_file"] = str(p.relative_to(out_dir))
            except ValueError:
                r["_file"] = str(p)
            recs.append(r)
    return recs


def run_one(target: str, profile, seed: int, out_dir: Path, **kw) -> dict:
    """Run one cell, write its JSON and log, return the record."""
    log = RunLog(log_path(out_dir, target, profile, seed), verbose=kw.pop("verbose", False))
    try:
        if target == "noise4x4":
            kw.pop("max_steps", None)
            kw.pop("ldpc_mode", None)
            kw.pop("step_penalty", None)
            rec = run_noise4x4(profile, seed, log=log, **kw)
        elif target in SURFACE_TARGETS:
            kw.pop("calib", None)
            kw.pop("calib_count", None)
            rec = run_surface(target, seed, log=log, **kw)
        else:
            raise ValueError(f"unknown target {target!r}")
    finally:
        log.close()
    write(run_path(out_dir, target, profile, seed), rec)
    return rec


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _yes(b) -> str:
    return "yes" if b else "no"


def _steps(x) -> str:
    return "n/a" if x is None else f"{int(x):,}"


def where_table(recs: list[dict]) -> str:
    L = [
        "| target | profile | seed | device | torch | policy | parameters | batch | budget (env steps) | env steps | updates | wall (s) | env steps/s | reward share | update share | wall cap hit | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        wall = max(r["wall_time_s"], 1e-9)
        L.append(
            f"| {r['target']} | {r['profile'] or '-'} | {r['seed']} | {r['device']} ({r['device_name']}) | {r['torch']} | "
            f"{r['ppo_config'].get('policy_resolved', '?')} | {r['num_parameters']:,} | {r['batch']} | {r['budget_env_steps']:,} | "
            f"{r['env_steps']:,} | {r['updates']} | {r['wall_time_s']:.0f} | {_steps(r['env_steps_per_s'])} | "
            f"{r['reward_time_s'] / wall:.0%} | {r['update_time_s'] / wall:.0%} | {_yes(r['hit_wall_cap'])} | `{r['_file']}` |"
        )
    return "\n".join(L)


def noise_sections(recs: list[dict], baselines: dict | None) -> list[str]:
    runs = [r for r in recs if r["target"] == "noise4x4"]
    if not runs:
        return ["No noise4x4 run recorded yet.", ""]
    prof = {p: sorted([r for r in runs if r["profile"] == p], key=lambda r: r["seed"]) for p in PROFILES}
    setting = runs[0]["setting"]
    L = []
    L.append(
        f"Setting 4x4 of `experiments.noise_aware`: n = {setting['n']}, k = 1, target d = {setting['d']}, "
        f"`{setting['graph_spec']}`, gates {', '.join(setting['gates'])}, {setting['max_steps']}-gate episodes, "
        f"reward `{runs[0]['reward']['form']}`, profiles calibrated to M_0 = {runs[0]['m0']:.4f} "
        f"(scales: " + ", ".join(f"{p} {runs[0]['calibration']['profiles'][p]['scale']:.3f}" for p in PROFILES) + "). "
        "The runs train the full budget and the best code of a seed is the highest-reward distinct target-distance state "
        "(`selection` in the JSON), as in `noise_aware.run_seed`.\n"
    )
    L.append("### Runs\n")
    L.append("| profile | seeds | device | budget (env steps) | wall per seed (s) | env steps/s | seeds reaching d* | first success (env steps) | final success rate | distinct target codes per seed | noise evaluations per seed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PROFILES:
        lst = prof[p]
        if not lst:
            continue
        reached = sum(1 for r in lst if r["distinct_success_codes"] > 0)
        L.append(
            f"| {p} | {len(lst)} | {', '.join(sorted({r['device'] for r in lst}))} | {lst[0]['budget_env_steps']:,} | "
            f"{_mean_range([float(r['wall_time_s']) for r in lst])} | {_mean_range([float(r['env_steps_per_s']) for r in lst if r['env_steps_per_s']])} | "
            f"{reached}/{len(lst)} | {_mean_range([r['first_success_env_steps'] for r in lst])} | "
            f"{_mean_range([r['final_success_rate'] for r in lst])} | {_mean_range([float(r['distinct_success_codes']) for r in lst])} | "
            f"{_mean_range([float(r['noise_evals']) for r in lst])} |"
        )
    L.append("")
    caps = [f"{r['profile']}/seed {r['seed']}" for r in runs if r["hit_wall_cap"]]
    L.append(("Runs stopped by the wall cap before their step budget: " + ", ".join(caps) + ".") if caps else "No run hit its wall cap.")
    L.append("")

    have = {p: [r for r in prof[p] if r["best"] and r["success"]] for p in PROFILES}
    L.append("### Cross-evaluation matrix\n")
    L.append(
        "Rows: training profile (the best code of each seed that reached d = 4, its own encoding circuit). Columns: evaluation "
        "profile at the calibrated scale. Entries: undetected fault mass of the encoder (`circuit_noise.undetected_fault_mass`, "
        "absolute probability), mean over seeds with [min, max]. Smaller is better. Seeds that did not reach d = 4 are excluded "
        "from the rows and counted in the runs table. The last rows are the known codes at n = 16 from "
        "`experiments/noise_aware/runs/baselines_4x4.json` (polished standard-form encoder per profile, the same numbers as the 4x4 table of "
        "`experiments/noise_aware/runs/REPORT.md`).\n"
    )
    L.append("| trained on \\ evaluated on | " + " | ".join(PROFILES) + " | seeds with d = 4 | gates (mean) |")
    L.append("|---|" + "---|" * (len(PROFILES) + 2))
    for p in PROFILES:
        lst = have[p]
        if not lst:
            L.append(f"| {p} | " + " | ".join("n/a" for _ in PROFILES) + f" | 0 of {len(prof[p])} | n/a |")
            continue
        cells = [_mean_range([r["best"]["mass"][ev]["undetected"] for r in lst]) for ev in PROFILES]
        L.append(
            f"| {p} | " + " | ".join(cells) + f" | {len(lst)} of {len(prof[p])} | "
            f"{np.mean([r['best']['circuit']['gates'] for r in lst]):.1f} |"
        )
    if baselines:
        for cname, c in baselines["codes"].items():
            bb = c["encoders"]["best_by_profile"]
            cells = [f"{bb[ev]['undetected']:.4g} ({bb[ev]['circuit']['gates']} gates)" if ev in bb else "no encoder" for ev in PROFILES]
            L.append(f"| known: {cname} | " + " | ".join(cells) + f" | d = {c['distance']}, w* = {c['w_star']}, CSS {_yes(c['is_css'])} | |")
    L.append("")
    if baselines and any(have.values()):
        rs = baselines["codes"].get("rotated_surface")
        if rs:
            bb = rs["encoders"]["best_by_profile"]
            parts = []
            for ev in PROFILES:
                own = [r["best"]["mass"][ev]["undetected"] for r in have[ev]]
                if own and ev in bb:
                    parts.append(f"{ev}: {sum(1 for m in own if m < bb[ev]['undetected'])} of {len(own)} seeds below the rotated surface code's {bb[ev]['undetected']:.4g}")
            if parts:
                L.append("Discovered codes trained on a profile against the rotated surface code evaluated on that profile: " + "; ".join(parts) + ".\n")

    L.append("Per seed (undetected mass under board / ion / heron of the best code; a seed without a d = 4 state shows its best lower-distance state and is not comparable):\n")
    L.append("| trained on | seed | reached d* | d | d_X | d_Z | w* | CSS | frame | board | ion | heron | gates | CNOTs | depth | first success (env steps) | file |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PROFILES:
        for r in prof[p]:
            a = r["best"]
            if not a:
                L.append(f"| {p} | {r['seed']} | no | none | | | | | | | | | | | | n/a | `{r['_file']}` |")
                continue
            fr = a["frame"]
            frame = "CSS" if a["is_css"] else (f"{fr['search']} frame, {fr['frame_size']} qubits" if fr.get("css_frame_found") else "none")
            L.append(
                f"| {p} | {r['seed']} | {_yes(r['success'])} | {a['distance']} | {_fmt(a['distance_x'])} | {_fmt(a['distance_z'])} | {a['w_star']} | "
                f"{_yes(a['is_css'])} | {frame} | " + " | ".join(f"{a['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                + f" | {a['circuit']['gates']} | {a['circuit']['two_qubit_gates']} | {a['circuit']['depth']} | {_steps(r['first_success_env_steps'])} | `{r['_file']}` |"
            )
    L.append("")
    modal = [(p, r) for p in PROFILES for r in prof[p] if r.get("modal")]
    if modal:
        L.append("Most visited target-distance code of the last 20 updates (the converged policy's code):\n")
        L.append("| trained on | seed | visits (of successes in the window) | d | d_X | d_Z | w* | CSS | board | ion | heron | gates | same as best |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p, r in modal:
            m = r["modal"]
            same = r["best"] is not None and m["stabilizers"] == r["best"]["stabilizers"]
            L.append(
                f"| {p} | {r['seed']} | {m['visits_last_20_updates']} ({m['successes_last_20_updates']}) | {m['distance']} | {_fmt(m['distance_x'])} | "
                f"{_fmt(m['distance_z'])} | {m['w_star']} | {_yes(m['is_css'])} | " + " | ".join(f"{m['mass'][ev]['undetected']:.4g}" for ev in PROFILES)
                + f" | {m['circuit']['gates']} | {_yes(same)} |"
            )
        L.append("")
    L.append("### Best code per seed\n")
    for p in PROFILES:
        for r in prof[p]:
            a = r["best"]
            L.append(f"**noise4x4 / {p} / seed {r['seed']}**: {r['selection']}.")
            if not a:
                L.append("")
                continue
            L.append(
                f"[[{a['n']},{a['k']},{a['distance']}]], d_X = {_fmt(a['distance_x'])}, d_Z = {_fmt(a['distance_z'])}, w* = {a['w_star']} "
                f"(degree {a['max_qubit_degree']}), CSS: {_yes(a['is_css'])}. Circuit: {a['circuit']['gates']} gates "
                f"({a['circuit']['two_qubit_gates']} CNOT, {a['circuit']['h_gates']} H), depth {a['circuit']['depth']}. "
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
    return L


def sparse_outcome(runs: list[dict]) -> str:
    """One sentence on the constrained target's outcome, from the JSON."""
    ok = [r for r in runs if r["success"]]
    if ok:
        return (
            f"Outcome: {len(ok)} of {len(runs)} runs produced a state with exact d = 5 and w* <= 4 "
            f"(modes {', '.join(sorted({r['ldpc']['mode'] for r in ok}))})."
        )
    parts = []
    for r in sorted(runs, key=lambda r: (r["ldpc"]["mode"], r["seed"])):
        m = r["ldpc"]["mode"]
        if m == "mask":
            fracs = [h["ldpc_violation_frac"] for h in r["history"] if h.get("ldpc_violation_frac") is not None]
            parts.append(
                f"`mask` seed {r['seed']} reached terminal distance {r['max_terminal_d_seen']} at most in {r['env_steps']:,} env steps"
                + (f" with {np.mean(fracs):.1%} of actions reverted by the bound" if fracs else "")
            )
        else:
            n5 = r["terminal_d_counts"][-1] if r.get("terminal_d_counts") else 0
            best = r.get("best_state") or {}
            w = best.get("w_star")
            parts.append(
                f"`{m}` seed {r['seed']} reached d = 5 in {n5} episodes over {r['env_steps']:,} env steps but none passed the exact w* <= 4 test"
                + (f" (best terminal state w* = {w}{'' if best.get('w_star_exact') else ', bound'})" if w is not None else "")
            )
    seeds = sorted({r["seed"] for r in runs})
    stopped = f"; the runs were stopped after seed {seeds[0]}" if seeds == [0] else ""
    capped = all(r["hit_wall_cap"] for r in runs)
    return (
        "Outcome: negative. No run produced a state with exact d = 5 and w* <= 4: "
        + "; ".join(parts)
        + (". Every run stopped at its wall cap" if capped else "")
        + stopped
        + "."
    )


def surface_sections(recs: list[dict], target: str = "surface5x5") -> list[str]:
    runs = sorted([r for r in recs if r["target"] == target], key=lambda r: ((r.get("ldpc") or {}).get("mode", ""), r["seed"]))
    if not runs:
        return [f"No {target} run recorded yet.", ""]
    r0 = runs[0]
    L = []
    L.append(
        f"Target [[{r0['n']},1,{r0['d_target']}]] on `{r0['graph']}` with gates {', '.join(r0['gates'])}, {r0['max_steps']}-gate episodes, "
        f"reward `{r0['reward']['form']}`. A run stops at its first success; the code is the shortest encoder among the successes of the "
        "stopping update, replayed and verified independently of the reward (exact distance by normalizer enumeration, weight enumerators "
        f"of S and N(S), exact w* up to {EXACT_W_STAR_MAX} by syndrome collisions and the greedy bound above, CSS test). `rotated surface` means the enumerators of S and N(S) "
        "equal those of the rotated surface code [[25,1,5]], the match criterion of the rediscovery regression (necessary for equivalence under "
        "qubit permutations and local Cliffords, not a proof of it).\n"
    )
    has_ldpc = bool(r0.get("ldpc"))
    if has_ldpc:
        modes = sorted({r["ldpc"]["mode"] for r in runs})
        L.append(
            f"Hard LDPC constraint w* <= {r0['ldpc']['w_max']}, tested exactly by `gpu_scale.SparsityCheck` (the stock `ldpc.ldpc_feasible` "
            "falls back to a greedy bound above 14 generators and calls the rotated surface code itself infeasible). Mode "
            f"{', '.join(f'`{m}`' for m in modes)}: in `mask` mode a gate that would take the group above the bound is replaced by a no-op step, so every "
            "visited state is sparse and `actions reverted` measures how often the policy pushes against the bound; in `final` "
            "mode the path is free and only a sparse target-distance state ends the episode, with the step penalty as the incentive. The penalty form of "
            "the constraint is not used (it trapped every run at n = 12 to 16, `experiments/scaling/REPORT.md`).\n"
        )
        L.append(sparse_outcome(runs) + "\n")
    L.append("### Runs\n")
    ldpc_cols = " LDPC mode | actions reverted | check time share |" if has_ldpc else ""
    L.append(f"| seed | device | policy | parameters | batch | budget (env steps) | env steps | episodes | updates | wall (s) | env steps/s |{ldpc_cols} success | first success (env steps) | update | wall to first success (s) | largest terminal d seen | terminal d counts (d = 0..5) | file |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|" + ("---|---|---|" if has_ldpc else "") + "---|---|---|---|---|---|---|")
    for r in runs:
        fs = r["first_success"]
        fs_wall = f"{fs['wall_time_s']:.0f}" if fs and fs.get("wall_time_s") is not None else "none"
        lc = ""
        if has_ldpc:
            fracs = [h["ldpc_violation_frac"] for h in r["history"] if h.get("ldpc_violation_frac") is not None]
            frac = f" ({np.mean(fracs):.1%} of actions)" if fracs else ""
            lc = f" {r['ldpc']['mode']} | {r.get('ldpc_violations', 0):,}{frac} | {r.get('ldpc_check_time_s', 0) / max(r['wall_time_s'], 1e-9):.0%} |"
        L.append(
            f"| {r['seed']} | {r['device']} | {r['ppo_config'].get('policy_resolved', '?')} | {r['num_parameters']:,} | {r['batch']} | {r['budget_env_steps']:,} | "
            f"{r['env_steps']:,} | {r['episodes']:,} | {r['updates']} | {r['wall_time_s']:.0f} | {_steps(r['env_steps_per_s'])} |{lc} {_yes(r['success'])} | "
            f"{_steps(fs['env_steps']) if fs else 'none'} | {fs['update'] + 1 if fs else 'none'} | {fs_wall} | "
            f"{r['max_terminal_d_seen']} | {r['terminal_d_counts']} | `{r['_file']}` |"
        )
    L.append("")
    L.append("### Codes\n")
    L.append("| seed | outcome | exact d | no weight-1 stabilizer | w* | w* <= 4 | max degree (greedy basis) | CSS | rotated surface enumerators | gates (CNOT), depth | verify time (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        c = r["code"]
        if c:
            ws = str(c["w_star"]) + ("" if c["w_star_exact"] else " (bound)")
            L.append(
                f"| {r['seed']} | [[{c['n']},{c['k']},{c['distance_exact']}]] | {c['distance_exact']} | {_yes(c['no_weight1_stabilizer'])} | {ws} | {_yes(c['w_star'] <= 4)} | "
                f"{c['max_qubit_degree']} | {_yes(c['is_css'])} | {_yes(c.get('enumerators_match_reference'))} | {c['gate_count']} ({c['cnot_count']}), {c['depth']} | {c['verify_time_s']} |"
            )
        elif r["best_state"]:
            b = r["best_state"]
            ws = str(b["w_star"]) + ("" if b["w_star_exact"] else " (bound)")
            L.append(
                f"| {r['seed']} | no success; best terminal state d = {b['distance_capped']} (cap {b['distance_cap']}), padding {b['padding_penalty']:.3f} | "
                f"{b['distance_capped']} | {_yes(b['padding_penalty'] == 0.0)} | {ws} | {_yes(b['w_star'] <= 4)} | {b['max_qubit_degree']} | {_yes(b['is_css'])} | no | {b['gate_count']} ({b['cnot_count']}), {b['depth']} | |"
            )
        else:
            L.append(f"| {r['seed']} | no terminal state recorded | | | | | | | | | |")
    L.append("")
    for r in runs:
        c = r["code"]
        if not c:
            continue
        L.append(f"**{target} / seed {r['seed']}**: {c['source']}.  ")
        L.append("Stabilizers (sparsest basis): `" + "`, `".join(c["sparsest_basis"]) + "`  ")
        L.append(f"Weight enumerator of S: {c['weight_enumerator_S']}  ")
        L.append(f"Weight enumerator of N(S): {c['weight_enumerator_N']}  ")
        if "reference_weight_enumerator_S" in c:
            L.append(f"Rotated surface [[25,1,5]], S: {c['reference_weight_enumerator_S']}  ")
            L.append(f"Rotated surface [[25,1,5]], N(S): {c['reference_weight_enumerator_N']}  ")
        L.append("Gates: `" + " ".join(f"{g[0]}({g[1]}{',' + str(g[2]) if g[2] >= 0 else ''})" for g in c["gates"]) + "`")
        L.append("")
    return L


def write_report(out_dir: Path, baselines_path: Path = DEFAULT_BASELINES, extra_dirs=None) -> Path:
    recs = load_all(out_dir, extra_dirs)
    baselines = None
    if baselines_path.exists():
        with open(baselines_path) as f:
            baselines = json.load(f)
    devices = sorted({f"{r['device']} ({r['device_name']}, torch {r['torch']})" for r in recs})
    L = [
        "# PPO at GPU scale: noise-aware 4x4 with the equivariant policy, and [[25,1,5]] on the 5x5 lattice",
        "",
        "Status: " + (f"{len(recs)} runs recorded" if recs else "no runs recorded") + ". Every number comes from a JSON under",
        f"`{out_dir}/` or one of its extra directories (named in the last column of each table, relative to `{out_dir}/`;",
        f"by default the subdirectories {list(EXTRA_DIRS)} hold a second mode of a target run with its own `--out-dir`); the tables regenerate with",
        "`uv run --extra rl python -m autoqec_sim.experiments.gpu_scale report`. Code:",
        "`src/autoqec_sim/experiments/gpu_scale.py`; pod commands in `docs/gpu_run.md`.",
        "",
        "This report extends, and does not replace, `experiments/noise_aware/runs/REPORT.md` (the 45-run",
        "noise-aware experiment, whose 4x4 cells all failed with the flat MLP at 256 episodes) and",
        "`experiments/scaling/REPORT.md` (the equivariant policy reaching [[16,1,4]] at batch 256).",
        "The 4x4 numbers here are the same reward, profiles, calibration, and code analysis with",
        "`policy = \"auto\"` (equivariant), a larger batch, and a larger budget; the known-code",
        "baselines at n = 16 are read from `experiments/noise_aware/runs/baselines_4x4.json` and not recomputed.",
        "",
        "## What ran where",
        "",
        ("Devices: " + "; ".join(devices) + ". " if devices else "")
        + "On `cuda` the policy network, the PPO update, and the state reward (exact distance up to d* - 1, "
        "fractional term, padding penalty) run on the GPU through `backends.torch_backend`; the numpy environment, "
        "the observation encoding, the terminal terms of the noise-aware reward (stim fault propagation and exact w*, "
        "cached per gate list), the candidate tracker, and every verification run on the host. `reward share` and "
        "`update share` are the fractions of the training wall time spent in the reward (host-side terminal terms "
        "included) and in the PPO update; the rest is the numpy environment step, the observation encoding, and the "
        "policy forward pass. On `cpu` every component is the numpy path of the earlier experiments.",
        "",
        where_table(recs),
        "",
        "## Target A: noise-aware 4x4 with the equivariant policy",
        "",
        *noise_sections(recs, baselines),
        "## Target B: [[25,1,5]] on the 5x5 lattice",
        "",
        *surface_sections(recs, "surface5x5"),
        "## Target C: [[25,1,5]] on the 5x5 lattice under the hard constraint w* <= 4",
        "",
        *surface_sections(recs, "surface5x5_sparse"),
        "## Caveats",
        "",
        "- Three seeds per cell; env-step counts and masses are three samples, not a distribution.",
        "- The noise4x4 cross-evaluation is only over seeds whose best code reached d = 4; a profile with fewer",
        "  such seeds has a smaller sample. The known-code rows are polished encoders optimized per profile",
        "  (see the 4x4 section of `experiments/noise_aware/runs/REPORT.md` for how much the polishing did).",
        "- The surface5x5 and surface5x5_sparse runs stop at their first success, so success rate after training",
        f"  and code diversity are not measured; `w*` is exact up to {EXACT_W_STAR_MAX} (syndrome collisions) and the greedy",
        "  bound of `autoqec_sim.ldpc` above that (marked `bound`).",
        "- surface5x5_sparse in `mask` mode restricts the search to groups with w* <= 4 at every step. The",
        "  standard-form encoders of the rotated surface code (routed or not) leave that set early, because the",
        "  boundary X checks have no exclusive pivot and the construction uses dense reduced generators; whether",
        "  a lattice circuit reaches the surface code through sparse intermediate groups is not established, so",
        "  a failure of this target does not show that no such circuit exists.",
        "- CPU-side work (numpy environment, terminal terms, verification) is unchanged by `--device cuda`;",
        "  the `reward share` column shows how much of the wall time it still is.",
        "",
        "## Regenerate",
        "",
        "```",
        "uv run --extra rl python -m autoqec_sim.experiments.gpu_scale matrix --device cuda",
        "uv run --extra rl python -m autoqec_sim.experiments.gpu_scale report",
        "```",
        "",
        "One cell:",
        "",
        "```",
        "uv run --extra rl python -m autoqec_sim.experiments.gpu_scale run --target noise4x4 --profile ion --seed 0 --device cuda",
        "uv run --extra rl python -m autoqec_sim.experiments.gpu_scale run --target surface5x5 --seed 0 --device cuda",
        "uv run --extra rl python -m autoqec_sim.experiments.gpu_scale run --target surface5x5_sparse --seed 0 --device cuda --ldpc-mode mask",
        "```",
        "",
    ]
    path = out_dir / "REPORT.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _common(p):
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=THREADS, help="CPU thread cap for torch and BLAS")
    p.add_argument("--chunk", type=int, default=None, help=f"Paulis per streamed chunk of the torch distance kernel (default {CHUNK})")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--verbose", action="store_true", help="also print the log lines to stdout")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO at GPU scale: noise-aware 4x4 (equivariant) and [[25,1,5]] on 5x5.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="one cell: target, profile (noise4x4 only), seed")
    p.add_argument("--target", choices=list(TARGETS), required=True)
    p.add_argument("--profile", choices=list(PROFILES), default=None, help="noise4x4 only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=None, help="parallel episodes (default 1024 noise4x4, 2048 surface5x5)")
    p.add_argument("--steps", type=int, default=None, help="env-step budget (default 20M noise4x4, 30M surface5x5)")
    p.add_argument("--wall", type=float, default=None, help="wall cap in minutes (default 60 noise4x4, 120 surface5x5)")
    p.add_argument("--max-steps", type=int, default=None, help="surface5x5 episode cap (default 80)")
    p.add_argument("--calib-count", type=int, default=na.CALIBRATION_CIRCUITS, help="noise4x4 calibration circuits (64 reproduces the earlier scales)")
    p.add_argument("--ldpc-mode", choices=LDPC_MODES, default="mask", help="surface5x5_sparse: hard-constraint mode")
    p.add_argument("--step-penalty", type=float, default=None, help="surface targets: added to every step's reward (default 0, or -0.01 in final mode)")
    _common(p)

    p = sub.add_parser("matrix", help="every cell of both targets, sequentially, then the report")
    p.add_argument("--targets", nargs="*", choices=list(TARGETS), default=list(TARGETS))
    p.add_argument("--profiles", nargs="*", choices=list(PROFILES), default=list(PROFILES))
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    for t in TARGETS:
        flag = t.replace("_", "-")
        p.add_argument(f"--batch-{flag}", type=int, default=None, dest=f"batch_{t}")
        p.add_argument(f"--steps-{flag}", type=int, default=None, dest=f"steps_{t}")
        p.add_argument(f"--wall-{flag}", type=float, default=None, dest=f"wall_{t}", help="minutes")
    p.add_argument("--ldpc-mode", choices=LDPC_MODES, default="mask", help="surface5x5_sparse: hard-constraint mode")
    p.add_argument("--step-penalty", type=float, default=None, help="surface targets: added to every step's reward")
    p.add_argument("--skip-existing", action="store_true", help="skip cells whose JSON exists")
    _common(p)

    p = sub.add_parser("report", help="write REPORT.md from the JSON files")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--baselines", default=str(DEFAULT_BASELINES), help="noise_aware baselines JSON for the n = 16 known codes")
    p.add_argument(
        "--extra-dirs", nargs="*", default=None,
        help=f"further directories of run_*.json to include (relative to --out-dir unless absolute; default the existing ones of {list(EXTRA_DIRS)})",
    )

    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.cmd == "run":
        if args.target == "noise4x4" and args.profile is None:
            ap.error("--profile is required for --target noise4x4")
        kw = dict(device=args.device, batch=args.batch, steps=args.steps, wall_min=args.wall, threads=args.threads, chunk=args.chunk, verbose=args.verbose)
        if args.target == "noise4x4":
            kw["calib_count"] = args.calib_count
        else:
            kw.update(max_steps=args.max_steps, ldpc_mode=args.ldpc_mode, step_penalty=args.step_penalty)
        rec = run_one(args.target, args.profile if args.target == "noise4x4" else None, args.seed, out_dir, **kw)
        fs = rec["first_success"]
        print(
            f"{run_name(rec['target'], rec['profile'], rec['seed'])}: success={rec['success']} first={fs['env_steps'] if fs else None} "
            f"env_steps={rec['env_steps']:,} wall={rec['wall_time_s']}s ({rec['env_steps_per_s']} steps/s) device={rec['device']}"
        )
        return 0
    if args.cmd == "matrix":
        t0 = time.perf_counter()
        cells = []
        if "noise4x4" in args.targets:
            cells += [("noise4x4", p, s) for p in args.profiles for s in args.seeds]
        for t in SURFACE_TARGETS:
            if t in args.targets:
                cells += [(t, None, s) for s in args.seeds]
        calib = None
        for target, profile, seed in cells:
            path = run_path(out_dir, target, profile, seed)
            if args.skip_existing and path.exists():
                print(f"skip {path.name} (exists)", flush=True)
                continue
            kw = dict(device=args.device, threads=args.threads, chunk=args.chunk, verbose=args.verbose)
            if target == "noise4x4":
                if calib is None:
                    calib = na.calibrate(na.SETTINGS["4x4"])
                kw.update(batch=args.batch_noise4x4, steps=args.steps_noise4x4, wall_min=args.wall_noise4x4, calib=calib)
            else:
                kw.update(
                    batch=getattr(args, f"batch_{target}"), steps=getattr(args, f"steps_{target}"), wall_min=getattr(args, f"wall_{target}"),
                    ldpc_mode=args.ldpc_mode, step_penalty=args.step_penalty,
                )
            rec = run_one(target, profile, seed, out_dir, **kw)
            fs = rec["first_success"]
            print(
                f"[{(time.perf_counter() - t0) / 60:.1f} min] {run_name(target, profile, seed)}: success={rec['success']} "
                f"first={fs['env_steps'] if fs else None} env_steps={rec['env_steps']:,} wall={rec['wall_time_s']}s ({rec['env_steps_per_s']} steps/s)",
                flush=True,
            )
            print("report:", write_report(out_dir), flush=True)
        print("report:", write_report(out_dir))
        return 0
    if args.cmd == "report":
        print("report:", write_report(out_dir, Path(args.baselines), args.extra_dirs))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
