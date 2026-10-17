# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Syndrome-extraction schedule synthesis for a fixed CSS code, as an
episodic MDP whose reward is the circuit's fault distance.

The code is fixed. What the agent chooses is the CX layer structure of one
extraction round: which (check, data qubit) couplings happen in which time
slot. That is the quantity the board's circuit tier scores
(``qldpc-challenge/verify/circuit_tools.build_css_memory``: "The schedule --
the thing the tier scores -- is the CX layer structure"), and the one that
moved the measured logical error rate by 5 to 300x in the project's matched
pairs under a fixed code, noise model, and decoder.

Episode. The Tanner edges of ``H_X`` and ``H_Z`` are visited in a fixed
order; at each step the agent picks a time slot for the current edge. A slot
is feasible when no edge already in it shares the check or the data qubit
(the layer must be a matching), and, without ``interleave``, X edges use the
first ``slots_x`` slots and Z edges the following ``slots_z`` so the two
blocks never overlap in time and every detector is deterministic. With
``interleave`` the two blocks share one set of slots; a schedule whose
detectors are not deterministic (the ordering criterion of the board's
interleaved builder fails) is invalid and scored ``invalid_reward``.

Reward. At the end of the episode the schedule is built into the board's
memory experiment (``research/circuit_autogen.build_memory``), the canonical
noise recipe is applied, stim derives the detector error model, and the
fault distance is read either exactly up to weight ``exact_wmax``
(:func:`autoqec_sim.memory_objective.exact_dcirc`, the right tier when the
code distance is at most ``exact_wmax``, which can go to 6) or by the
board's own randomized
information-set search (``circuit_tools.ris_dem``, an upper bound). The
reward is ``min_basis d_circ / d`` minus ``depth_penalty`` per CX layer
beyond the lower bound, so a schedule that reaches the code distance in the
fewest layers scores 1. ``d_circ`` is clamped to ``d``: nothing above the
code distance is credited, since the tier does not credit it either.

Everything the reward depends on is the board's code, imported from a
checkout of ``qldpc-challenge`` (``QLDPC_CHALLENGE`` environment variable,
default ``../qldpc-challenge`` beside this repository), so a schedule found
here is scored exactly as the board would score it on submission.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ScheduleEnv",
    "coloring_baseline",
    "layers_from_slots",
    "load_board_code",
    "score_layers",
]

# --------------------------------------------------------------------------
# board code import
# --------------------------------------------------------------------------


def _qldpc_root() -> Path:
    root = os.environ.get("QLDPC_CHALLENGE")
    if root:
        return Path(root)
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "qldpc-challenge"
        if (cand / "verify" / "circuit_tools.py").exists():
            return cand
    raise ImportError("set QLDPC_CHALLENGE to a qldpc-challenge checkout")


_CT = None
_CA = None


def _board():
    """The board's circuit tools and circuit builder, imported lazily."""
    global _CT, _CA
    if _CT is None:
        root = _qldpc_root()
        for sub in ("verify", "research"):
            s = str(root / sub)
            if s not in sys.path:
                sys.path.insert(0, s)
        import circuit_autogen as ca
        import circuit_tools as ct
        _CT, _CA = ct, ca
    return _CT, _CA


@dataclass
class BoardCode:
    slug: str
    n: int
    k: int
    d: int
    HX: np.ndarray
    HZ: np.ndarray
    d_circ_board: int | None = None
    schedule_note: str = ""


def load_board_code(slug: str) -> BoardCode:
    """Read ``codes/<slug>.json`` from the board checkout."""
    root = _qldpc_root()
    with open(root / "codes" / f"{slug}.json") as f:
        doc = json.load(f)
    n = int(doc["n"])

    def mat(rows):
        out = np.zeros((len(rows), n), dtype=np.int8)
        for i, r in enumerate(rows):
            if len(r) == n and max(r) <= 1 and len(r) > 1:
                out[i] = np.asarray(r, dtype=np.int8)
            else:
                out[i, list(r)] = 1
        return out

    circ = doc.get("circuit") or {}
    dc = circ.get("d_circ")
    board_dc = None
    if isinstance(dc, dict):
        vals = [v.get("value") for v in dc.values() if isinstance(v, dict) and isinstance(v.get("value"), int)]
        board_dc = min(vals) if vals else None
    elif isinstance(dc, int):
        board_dc = dc
    return BoardCode(
        slug=slug,
        n=n,
        k=int(doc["k"]),
        d=int(doc["distance"]["d"]),
        HX=mat(doc["checks"]["X"]),
        HZ=mat(doc["checks"]["Z"]),
        d_circ_board=board_dc,
        schedule_note=(circ.get("notes") or "")[:200],
    )


# --------------------------------------------------------------------------
# schedule scoring
# --------------------------------------------------------------------------


def layers_from_slots(edges, slots, num_slots) -> list[list[tuple]]:
    """Group ``edges`` (``(kind, check, qubit)``) by their slot into the
    layer list ``build_memory`` takes; empty slots are dropped."""
    layers = [[] for _ in range(num_slots)]
    for e, s in zip(edges, slots):
        if s >= 0:
            layers[int(s)].append(e)
    return [layer for layer in layers if layer]


def score_layers(code: BoardCode, layers, rounds=None, bases=("Z", "X"), tier="exact",
                 exact_wmax=4, ris_trials=2000, ris_seed=0, ris_seconds=None):
    """Fault distance of a schedule under the board's own pipeline.

    Parameters
    ----------
    code
        :class:`BoardCode`.
    layers
        CX layers, each a list of ``(kind, check, qubit)`` edges.
    rounds
        Extraction rounds scored (default ``code.d``).
    bases
        Memory bases scored.
    tier
        ``"exact"`` (:func:`autoqec_sim.memory_objective.exact_dcirc` up to
        ``exact_wmax``) or ``"ris"`` (randomized information set search with
        ``ris_trials`` trials, ``ris_seed``, and ``ris_seconds`` cap).

    Returns
    -------
    dict
        ``{"valid": bool, "d_circ": {basis: int | None}, "min": int | None,
        "layers": int, "seconds": float, "detail": {...}}``. In the ``exact``
        tier a basis whose search finds nothing up to ``exact_wmax`` reports
        ``exact_wmax + 1`` as a floor; in the ``ris`` tier ``None`` from the
        search is reported as ``code.d``.
    """
    ct, ca = _board()
    rounds = code.d if rounds is None else int(rounds)
    t0 = time.time()
    out = {"valid": True, "d_circ": {}, "layers": len(layers), "detail": {}}
    for basis in bases:
        skel = ca.build_memory(code.HX, code.HZ, rounds, basis, layers)
        if not ca.deterministic(skel):
            out["valid"] = False
            out["d_circ"][basis] = None
            out["detail"][basis] = "detectors not deterministic"
            break
        noisy = ct.apply_noise(skel, code.n)
        dem = ct.derive_dem(noisy)
        if tier == "exact":
            from autoqec_sim import memory_objective as mo
            sig = mo.dem_signature(dem)
            r = mo.exact_dcirc(sig, wmax=exact_wmax)
            dc = r["d_circ"] if r["d_circ"] is not None else exact_wmax + 1
            out["detail"][basis] = {"count": r["count"], "mass": r["mass"], "mechanisms": len(sig),
                                    "floor": r["d_circ"] is None}
        else:
            H, L = ct.dem_matrices(dem)
            w, _wit = ct.ris_dem(H, L, trials=ris_trials, seed=ris_seed, max_seconds=ris_seconds)
            dc = int(w) if w is not None else code.d
            out["detail"][basis] = {"mechanisms": int(H.shape[1]), "found": w is not None}
        out["d_circ"][basis] = int(dc)
    vals = [v for v in out["d_circ"].values() if v is not None]
    out["min"] = min(vals) if (vals and out["valid"]) else None
    out["seconds"] = time.time() - t0
    return out


def _edges_of(H, kind):
    rows, cols = np.nonzero(H)
    return [(kind, int(r), int(c)) for r, c in zip(rows, cols)]


def coloring_baseline(code: BoardCode, samples=20, seed=0, **score_kw):
    """The board builder's own schedule family: a greedy proper edge coloring of
    each Tanner graph, X block then Z block, shuffled by seed.

    Parameters
    ----------
    code
        :class:`BoardCode`.
    samples
        Number of colorings scored.
    seed
        First seed; sample ``i`` uses ``seed + i``.
    **score_kw
        Passed to :func:`score_layers`.

    Returns
    -------
    list[dict]
        One :func:`score_layers` result per sample with ``layers_list`` and
        ``seed`` attached.
    """
    ct, _ = _board()
    out = []
    for i in range(samples):
        rng = np.random.default_rng(seed + i)
        lx = ct.edge_coloring(code.HX, rng)
        lz = ct.edge_coloring(code.HZ, rng)
        layers = [[("X", int(c), int(q)) for c, q in layer] for layer in lx if layer] + \
                 [[("Z", int(c), int(q)) for c, q in layer] for layer in lz if layer]
        s = score_layers(code, layers, **score_kw)
        s["layers_list"] = layers
        s["seed"] = seed + i
        out.append(s)
    return out


# --------------------------------------------------------------------------
# the environment
# --------------------------------------------------------------------------


@dataclass
class ScheduleEnv:
    """Vectorized schedule-synthesis environment over ``batch_size``
    episodes of a fixed board code.

    Parameters
    ----------
    code
        :class:`BoardCode` (see :func:`load_board_code`).
    batch_size
        Parallel episodes.
    slots_x, slots_z
        Time slots offered to the X and Z blocks. Default: the block's
        chromatic lower bound (max of check weight and qubit degree) plus
        ``slack``. More slots make every episode feasible but let the agent
        waste depth, which ``depth_penalty`` prices.
    interleave
        Share one slot set between the blocks (depth can halve; detectors
        may become non-deterministic, which is scored ``invalid_reward``).
    tier, exact_wmax, ris_trials, rounds, bases
        Passed to :func:`score_layers`.
    depth_penalty
        Reward per CX layer beyond the lower bound, subtracted.
    edge_order
        ``"fixed"`` (checks in row order, qubits ascending) or
        ``"shuffle"`` (a new random visiting order per episode, which makes
        the policy learn a slot rule rather than a lookup table).
    """

    code: BoardCode
    batch_size: int = 16
    slots_x: int | None = None
    slots_z: int | None = None
    slack: int = 1
    interleave: bool = False
    tier: str = "exact"
    exact_wmax: int = 4
    ris_trials: int = 2000
    rounds: int | None = None
    bases: tuple = ("Z", "X")
    depth_penalty: float = 0.02
    invalid_reward: float = -1.0
    edge_order: str = "fixed"
    seed: int = 0
    log: object = None
    # state
    rng: np.random.Generator = field(init=False, repr=False)
    edges: list = field(init=False, repr=False)

    def __post_init__(self):
        HX, HZ = self.code.HX, self.code.HZ
        self.rng = np.random.default_rng(self.seed)
        self.edges_x = _edges_of(HX, "X")
        self.edges_z = _edges_of(HZ, "Z")
        self.edges = self.edges_x + self.edges_z
        self.E = len(self.edges)
        self.Ex = len(self.edges_x)
        lb_x = int(max(HX.sum(1).max(), HX.sum(0).max()))
        lb_z = int(max(HZ.sum(1).max(), HZ.sum(0).max()))
        self.lower_bound_x, self.lower_bound_z = lb_x, lb_z
        if self.interleave:
            lb = int(max(lb_x, lb_z, (HX.sum(0) + HZ.sum(0)).max()))
            T = (self.slots_x or lb) + self.slack if self.slots_x is None else self.slots_x
            self.T = int(T)
            self.block_slots = (np.arange(self.T), np.arange(self.T))
            self.lower_bound_layers = lb
        else:
            Tx = (lb_x + self.slack) if self.slots_x is None else int(self.slots_x)
            Tz = (lb_z + self.slack) if self.slots_z is None else int(self.slots_z)
            self.T = Tx + Tz
            self.block_slots = (np.arange(Tx), Tx + np.arange(Tz))
            self.lower_bound_layers = lb_x + lb_z
        # per-edge check id (block-local) and data qubit, for conflict masks
        self.edge_check = np.array([c for _, c, _ in self.edges], dtype=np.int64)
        self.edge_qubit = np.array([q for _, _, q in self.edges], dtype=np.int64)
        self.edge_kind = np.array([0 if k == "X" else 1 for k, _, _ in self.edges], dtype=np.int64)
        self.rX = HX.shape[0]
        self.rZ = HZ.shape[0]
        B = self.batch_size
        self.slots = np.full((B, self.E), -1, dtype=np.int64)
        self.order = np.tile(np.arange(self.E), (B, 1))
        self.ptr = np.zeros(B, dtype=np.int64)
        # occupancy: (B, T, checks_total) and (B, T, n) booleans
        self.occ_check = np.zeros((B, self.T, self.rX + self.rZ), dtype=bool)
        self.occ_qubit = np.zeros((B, self.T, self.code.n), dtype=bool)
        self.last_scores = [None] * B
        self.episodes = 0

    # ---- interface ------------------------------------------------------
    @property
    def num_actions(self) -> int:
        return self.T

    @property
    def obs_dim(self) -> int:
        # slot occupancy per edge (E x T) + current edge one-hot (E) + kind flag
        return self.E * self.T + self.E + 1

    def observation(self) -> np.ndarray:
        B = self.batch_size
        onehot = np.zeros((B, self.E, self.T), dtype=np.float32)
        b, e = np.nonzero(self.slots >= 0)
        onehot[b, e, self.slots[b, e]] = 1.0
        cur = np.zeros((B, self.E), dtype=np.float32)
        done = self.ptr >= self.E
        idx = np.flatnonzero(~done)
        cur[idx, self.order[idx, self.ptr[idx]]] = 1.0
        kind = np.zeros((B, 1), dtype=np.float32)
        if idx.size:
            kind[idx, 0] = self.edge_kind[self.order[idx, self.ptr[idx]]]
        return np.concatenate([onehot.reshape(B, -1) * 2 - 1, cur, kind], axis=1)

    def action_mask(self) -> np.ndarray:
        """``(B, T)`` feasible slots for each episode's current edge."""
        B = self.batch_size
        mask = np.zeros((B, self.T), dtype=bool)
        self.dead = np.zeros(B, dtype=bool)
        for b in range(B):
            if self.ptr[b] >= self.E:
                mask[b, 0] = True
                continue
            e = self.order[b, self.ptr[b]]
            kind = self.edge_kind[e]
            cand = self.block_slots[kind]
            chk = self.edge_check[e] + (0 if kind == 0 else self.rX)
            ok = ~self.occ_check[b, cand, chk] & ~self.occ_qubit[b, cand, self.edge_qubit[e]]
            mask[b, cand] = ok
            if not mask[b].any():
                # dead end: no slot is free for this edge. Offer one slot so a
                # sampler has something to pick, and let step() record the
                # episode as infeasible whatever it picks.
                self.dead[b] = True
                mask[b, cand[0]] = True
        return mask

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        B = self.batch_size
        self.slots[:] = -1
        self.ptr[:] = 0
        self.occ_check[:] = False
        self.occ_qubit[:] = False
        self.infeasible = np.zeros(B, dtype=bool)
        if self.edge_order == "shuffle":
            for b in range(B):
                ox = self.rng.permutation(self.Ex)
                oz = self.Ex + self.rng.permutation(self.E - self.Ex)
                self.order[b] = np.concatenate([ox, oz])
        else:
            self.order[:] = np.arange(self.E)
        return self.observation(), {}

    def step(self, actions):
        """Assign the chosen slot to each episode's current edge. Returns
        ``(obs, reward, done, info)``; reward is zero until the final step,
        where every episode is scored at once."""
        actions = np.asarray(actions, dtype=np.int64)
        B = self.batch_size
        mask = self.action_mask()
        for b in range(B):
            if self.ptr[b] >= self.E:
                continue
            e = self.order[b, self.ptr[b]]
            a = int(actions[b])
            if self.dead[b] or not mask[b, a]:
                self.infeasible[b] = True
            self.slots[b, e] = a
            kind = self.edge_kind[e]
            chk = self.edge_check[e] + (0 if kind == 0 else self.rX)
            self.occ_check[b, a, chk] = True
            self.occ_qubit[b, a, self.edge_qubit[e]] = True
            self.ptr[b] += 1
        done = self.ptr >= self.E
        reward = np.zeros(B, dtype=np.float32)
        info = {}
        if done.all():
            scores = self.score_all()
            for b, s in enumerate(scores):
                reward[b] = s["reward"]
            info["scores"] = scores
            self.episodes += B
        return self.observation(), reward, done, info

    # ---- scoring --------------------------------------------------------
    def layers(self, b: int) -> list[list[tuple]]:
        return layers_from_slots(self.edges, self.slots[b], self.T)

    def reward_of(self, s: dict) -> float:
        """Reward of a :func:`score_layers` result. The fault distance is
        clamped to the code distance. In the exact tier the number of
        minimum-weight undetected fault sets at the limiting basis is a
        dense tie-breaker inside one distance level: ``(d_circ + 1 /
        (1 + count)) / (d + 1)``, so one fewer such set is progress and any
        schedule at ``d_circ + 1`` beats every schedule at ``d_circ``. In
        the ris tier no count exists and the term is a constant."""
        if not s.get("valid") or s.get("min") is None:
            return float(self.invalid_reward)
        dc = min(s["min"], self.code.d)
        count = 0
        if self.tier == "exact":
            counts = [v.get("count", 0) for b, v in s["detail"].items()
                      if isinstance(v, dict) and s["d_circ"].get(b) == s["min"]]
            count = max(counts) if counts else 0
        s["count_min"] = int(count)
        value = (dc + 1.0 / (1.0 + count)) / (self.code.d + 1.0)
        return float(value - self.depth_penalty * max(0, s["layers"] - self.lower_bound_layers))

    def score_all(self) -> list[dict]:
        out = []
        for b in range(self.batch_size):
            if self.infeasible[b]:
                out.append({"valid": False, "min": None, "reward": self.invalid_reward,
                            "layers": None, "detail": "infeasible slot chosen"})
                continue
            layers = self.layers(b)
            s = score_layers(self.code, layers, rounds=self.rounds, bases=self.bases, tier=self.tier,
                             exact_wmax=self.exact_wmax, ris_trials=self.ris_trials,
                             ris_seed=int(self.rng.integers(2**31)))
            s["reward"] = self.reward_of(s)
            s["layers_list"] = layers
            out.append(s)
            if self.log is not None:
                self.log(f"episode {self.episodes + b}: d_circ {s['d_circ']} sets {s.get('count_min')} layers {s['layers']} reward {s['reward']:.3f} ({s.get('seconds', 0):.1f}s)")
        self.last_scores = out
        return out
