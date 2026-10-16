# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Batched code-discovery environment with gymnasium-style semantics.

The state is one batched tableau of shape ``(batch, 2n, 2n + 1)``; every
batch element is an independent episode. Actions are Clifford gates: the
single-qubit gates on every qubit and the two-qubit gates on both
orientations of every edge of a connectivity graph. The reward is any
callable ``f(tab, k) -> (batch,)``, by default the Knill-Laflamme reward of
:mod:`autoqec_sim.reward`. There is no dependency on gymnasium; ``reset``
and ``step`` follow its calling convention (``observation``, ``reward``,
``terminated``, ``truncated``, ``info``).

An optional LDPC constraint (``ldpc_w_max``) restricts the search to
states whose stabilizer group has a generating set with every generator of
weight at most ``w_max`` (:func:`autoqec_sim.ldpc.ldpc_feasible`, exact
for ``n - k <= 14``). Mode ``"terminate"`` ends the episode of an element
that leaves the feasible set with ``ldpc_violation_reward``; mode
``"mask"`` reverts such an action to a no-op (the step is still counted)
and :meth:`CodeDiscoveryEnv.feasible_action_mask` gives agents the full
per-action mask; mode ``"final"`` leaves the path free and only withholds
success from an infeasible state (the constraint binds the solution, not
the trajectory). A soft alternative is the reward term
:func:`autoqec_sim.ldpc.make_ldpc_penalty`. ``terminated`` is true when an
episode ends by success or by violation; ``info["success"]`` distinguishes
them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .errors import ErrorSet
from .graphs import Graph, all_to_all, as_graph
from .ldpc import ldpc_feasible
from .reward import kl_reward
from .tableau import GATES_1Q, GATES_2Q, ActionTable, identity_tableau

RewardFn = Callable[[np.ndarray, int], np.ndarray]


def build_action_table(graph: Graph, gates=("H", "S", "CNOT")) -> ActionTable:
    """Single-qubit gates on every qubit, two-qubit gates on both directions
    of every edge (``CZ`` and ``SWAP`` are symmetric and get one entry per
    edge)."""
    entries = []
    for g in gates:
        if g in GATES_1Q:
            entries += [(g, q, -1) for q in range(graph.n)]
        elif g in GATES_2Q:
            if g == "CNOT":
                entries += [(g, a, b) for a, b in graph.directed_edges()]
            else:
                entries += [(g, a, b) for a, b in graph.edges]
        else:
            raise ValueError(f"unknown gate {g!r}")
    return ActionTable(entries)


class CodeDiscoveryEnv:
    """Vectorized environment for stabilizer-code discovery.

    Parameters
    ----------
    n, k
        Physical and logical qubit counts. Qubits ``0..k-1`` carry the
        logical state, ``k..n-1`` start in ``|0>``.
    graph
        Connectivity for two-qubit gates (default all-to-all on ``n``);
        either :class:`autoqec_sim.graphs.Graph` or
        ``autoqec_sim.noise.topology.Graph``. ``env.topology`` is the
        ``autoqec_sim.noise`` view of the same graph for noise profiles.
    batch_size
        Number of parallel episodes.
    reward_fn
        ``f(tab, k) -> (batch,)``. Default: normalized KL reward for all
        Paulis of weight ``< target_distance``.
    target_distance
        Used only to build the default reward.
    gates
        Gate names offered as actions.
    max_steps
        Episode length cap (``truncated`` after this many steps).
    reward_mode
        ``"dense"``: reward of the new state at every step (Olle et al.);
        ``"delta"``: change in state reward; ``"sparse"``: state reward
        only on the step that ends the episode, otherwise ``step_penalty``.
    step_penalty
        Added to every step's reward (negative favours short circuits).
    success_threshold
        A state whose reward is ``>= success_threshold`` ends the episode
        with ``terminated = True``.
    success_fn
        Optional ``f(tab, k, state_reward) -> (batch,) bool`` replacing the
        threshold test, for rewards whose success criterion is not a level
        of the reward itself (a reward that adds a negative noise term to
        states that reached the target distance, for example). When it is
        ``None`` and ``reward_fn`` has a ``success_fn`` attribute, that
        method is used (:class:`autoqec_sim.memory_objective.MemoryReward`).
    observation
        ``"tableau"`` (``(batch, 2n, 2n + 1)`` bits), ``"check_matrix"``
        (``(batch, n - k, 2n)`` code stabilizers, the observation of Olle
        et al.), or ``"flat"`` (tableau flattened per element).
    auto_reset
        If true, finished episodes are reset to the identity tableau at the
        end of ``step``; the returned observation is then the fresh one and
        ``info["final_tableau"]`` holds the terminal tableaux.
    ldpc_w_max, ldpc_mode, ldpc_violation_reward
        Hard LDPC constraint (see the module docstring); ``ldpc_w_max=None``
        disables it. ``ldpc_mode`` is ``"terminate"``, ``"mask"``, or
        ``"final"``.
    """

    def __init__(
        self,
        n: int,
        k: int,
        graph: Graph | None = None,
        batch_size: int = 64,
        reward_fn: RewardFn | None = None,
        target_distance: int = 3,
        gates=("H", "S", "CNOT"),
        max_steps: int = 30,
        reward_mode: str = "dense",
        step_penalty: float = 0.0,
        success_threshold: float = 0.0,
        observation: str = "tableau",
        auto_reset: bool = False,
        seed: int | None = None,
        ldpc_w_max: int | None = None,
        ldpc_mode: str = "terminate",
        ldpc_violation_reward: float = -1.0,
        success_fn: Callable[[np.ndarray, int, np.ndarray], np.ndarray] | None = None,
    ):
        self.n = int(n)
        self.k = int(k)
        self.graph = as_graph(graph) if graph is not None else all_to_all(self.n)
        if self.graph.n != self.n:
            raise ValueError("graph size does not match n")
        self.topology = self.graph.to_topology()
        self.ldpc_w_max = None if ldpc_w_max is None else int(ldpc_w_max)
        if ldpc_mode not in ("terminate", "mask", "final"):
            raise ValueError("ldpc_mode must be terminate, mask, or final")
        self.ldpc_mode = ldpc_mode
        self.ldpc_violation_reward = float(ldpc_violation_reward)
        self.batch_size = int(batch_size)
        if reward_fn is None:
            errors = ErrorSet.up_to_weight(self.n, target_distance - 1)
            reward_fn = lambda tab, k: kl_reward(tab, k, errors)  # noqa: E731
        self.reward_fn = reward_fn
        self.target_distance = target_distance
        self.gates = tuple(gates)
        self.action_table = build_action_table(self.graph, self.gates)
        self.max_steps = int(max_steps)
        if reward_mode not in ("dense", "delta", "sparse"):
            raise ValueError("reward_mode must be dense, delta, or sparse")
        self.reward_mode = reward_mode
        self.step_penalty = float(step_penalty)
        self.success_threshold = float(success_threshold)
        # a reward object may carry its own success criterion (the memory
        # reward of :mod:`autoqec_sim.memory_objective` and rewards built
        # like it); an explicit ``success_fn`` takes precedence
        if success_fn is None:
            success_fn = getattr(reward_fn, "success_fn", None)
        self.success_fn = success_fn
        if observation not in ("tableau", "check_matrix", "flat"):
            raise ValueError("unknown observation kind")
        self.observation_kind = observation
        self.auto_reset = bool(auto_reset)
        self.rng = np.random.default_rng(seed)
        self.tab = identity_tableau(self.n, self.batch_size)
        self.steps = np.zeros(self.batch_size, dtype=np.int64)
        self.history = np.full((self.batch_size, self.max_steps), -1, dtype=np.int64)
        self.state_reward = self.reward_fn(self.tab, self.k)

    # ------------------------------------------------------------------
    @property
    def num_actions(self) -> int:
        return len(self.action_table)

    @property
    def observation_shape(self) -> tuple:
        n, k = self.n, self.k
        if self.observation_kind == "tableau":
            return (2 * n, 2 * n + 1)
        if self.observation_kind == "check_matrix":
            return (n - k, 2 * n)
        return (2 * n * (2 * n + 1),)

    def observation(self) -> np.ndarray:
        n, k = self.n, self.k
        if self.observation_kind == "tableau":
            return self.tab.copy()
        if self.observation_kind == "check_matrix":
            return self.tab[:, n + k :, : 2 * n].copy()
        return self.tab.reshape(self.batch_size, -1).copy()

    def sample_actions(self) -> np.ndarray:
        return self.rng.integers(self.num_actions, size=self.batch_size)

    def action_description(self, a: int) -> tuple:
        return self.action_table.entries[int(a)]

    def ldpc_feasible(self, tab=None) -> np.ndarray:
        """``(batch,)`` bool feasibility of ``tab`` (default the current
        state) under the LDPC constraint; all true when it is disabled."""
        tab = self.tab if tab is None else tab
        if self.ldpc_w_max is None:
            return np.ones(tab.shape[0], dtype=bool)
        return ldpc_feasible(tab, self.k, self.ldpc_w_max)

    def feasible_action_mask(self) -> np.ndarray:
        """``(batch, num_actions)`` bool: actions whose resulting state
        satisfies the LDPC constraint. One batched feasibility check per
        action; all true when the constraint is disabled."""
        mask = np.ones((self.batch_size, self.num_actions), dtype=bool)
        if self.ldpc_w_max is None:
            return mask
        for a in range(self.num_actions):
            trial = self.tab.copy()
            self.action_table.apply(trial, np.full(self.batch_size, a, dtype=np.int64))
            mask[:, a] = ldpc_feasible(trial, self.k, self.ldpc_w_max)
        return mask

    def gate_lists(self, where=None) -> list[list[tuple]]:
        """Per-element gate lists ``(gate, q1, q2)`` from the action history."""
        idx = np.arange(self.batch_size) if where is None else np.flatnonzero(where)
        out = []
        for b in idx:
            acts = self.history[b, : self.steps[b]]
            out.append([self.action_table.entries[a] for a in acts if a >= 0])
        return out

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, where=None):
        """Reset all episodes (or those selected by ``where``). Returns the
        observation and an info dict."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if where is None:
            self.tab = identity_tableau(self.n, self.batch_size)
            self.steps[:] = 0
            self.history[:] = -1
            self.state_reward = self.reward_fn(self.tab, self.k)
        else:
            idx = np.flatnonzero(np.asarray(where, dtype=bool))
            if idx.size:
                self.tab[idx] = identity_tableau(self.n, 1)[0]
                self.steps[idx] = 0
                self.history[idx] = -1
                self.state_reward[idx] = self.reward_fn(self.tab[idx], self.k)
        return self.observation(), {"state_reward": self.state_reward.copy()}

    def step(self, actions):
        """Apply one action per batch element. Negative actions are no-ops
        (the element still counts a step unless it is already finished).

        Returns ``(observation, reward, terminated, truncated, info)``; all
        four arrays have shape ``(batch,)``. ``info`` holds
        ``state_reward`` (reward of the new state), ``steps``, and, with
        ``auto_reset``, ``final_tableau`` and ``final_state_reward``.
        """
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.batch_size,):
            raise ValueError("actions must have shape (batch,)")
        active = self.steps < self.max_steps
        prev_tab = self.tab.copy() if self.ldpc_w_max is not None else None
        self.action_table.apply(self.tab, actions, where=active)
        valid = active & (actions >= 0)
        violated = np.zeros(self.batch_size, dtype=bool)
        if self.ldpc_w_max is not None:
            violated = active & ~ldpc_feasible(self.tab, self.k, self.ldpc_w_max)
            if self.ldpc_mode == "mask" and violated.any():
                self.tab[violated] = prev_tab[violated]
                actions = np.where(violated, -1, actions)
                valid &= ~violated
        self.history[np.flatnonzero(active), self.steps[active]] = actions[active]
        self.steps[active] += 1
        prev = self.state_reward
        new = self.reward_fn(self.tab, self.k)
        if self.success_fn is not None:
            success = np.asarray(self.success_fn(self.tab, self.k, new), dtype=bool)
        else:
            success = new >= self.success_threshold
        if self.ldpc_mode == "terminate":
            success &= ~violated
            terminated = success | violated
        elif self.ldpc_mode == "final":
            success &= ~violated
            terminated = success
        else:
            terminated = success
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated
        if self.reward_mode == "dense":
            reward = new.copy()
        elif self.reward_mode == "delta":
            reward = new - prev
        else:
            reward = np.where(done, new, 0.0)
        reward = reward + self.step_penalty
        if self.ldpc_mode == "terminate":
            reward = np.where(violated, self.ldpc_violation_reward, reward)
        reward = np.where(active, reward, 0.0)
        self.state_reward = new
        info = {
            "state_reward": new.copy(),
            "steps": self.steps.copy(),
            "valid_action": valid,
            "success": success,
            "ldpc_violation": violated,
        }
        if self.auto_reset and done.any():
            info["final_tableau"] = self.tab[done].copy()
            info["final_state_reward"] = new[done].copy()
            info["final_steps"] = self.steps[done].copy()
            info["final_history"] = self.history[done].copy()
            self.reset(where=done)
        return self.observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def rollout(self, action_matrix: np.ndarray):
        """Run fixed action sequences ``(batch, L)`` (``-1`` = no-op) from a
        fresh reset and return ``(state_reward, success, used_steps)``.

        Episodes stop contributing once they succeed, so the returned
        tableau of a successful element is the first successful state.
        """
        action_matrix = np.asarray(action_matrix, dtype=np.int64)
        if action_matrix.shape[0] != self.batch_size:
            raise ValueError("action_matrix batch does not match env batch")
        self.reset()
        success = np.zeros(self.batch_size, dtype=bool)
        used = np.zeros(self.batch_size, dtype=np.int64)
        dead = np.zeros(self.batch_size, dtype=bool)  # ended by violation
        for t in range(min(action_matrix.shape[1], self.max_steps)):
            acts = np.where(success | dead, -1, action_matrix[:, t])
            _, _, terminated, _, info = self.step(acts)
            newly = info["success"] & ~success
            used[newly] = t + 1
            success |= info["success"]
            dead |= terminated & ~info["success"]
            if (success | dead).all():
                break
        used[~success] = min(action_matrix.shape[1], self.max_steps)
        reward = self.state_reward.copy()
        if self.ldpc_mode == "terminate":
            reward[dead & ~success] = self.ldpc_violation_reward
        return reward, success, used
