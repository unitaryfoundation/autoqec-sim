# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Search agents over gate sequences, evaluated in batches by the environment.

:class:`EvolutionarySearch` keeps a population of fixed-length action
sequences (``-1`` entries are no-ops, so sequences may be shorter than the
cap). Every generation is one batched rollout. The next population is the
elite, mutated children of tournament-selected parents (point replacement,
insertion, deletion, one-point crossover), and a fraction of fresh random
sequences. With ``random_frac = 1`` it degenerates to random search.

A PPO agent over the same environment is in :mod:`autoqec_sim.agents.ppo`
(optional torch dependency).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .env import CodeDiscoveryEnv


@dataclass
class SearchResult:
    success: bool
    best_reward: float
    actions: list[int]
    gates: list[tuple]
    evaluations: int
    generations: int
    wall_time_s: float
    seed: int
    history: list[float] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["gates"] = [list(g) for g in self.gates]
        return d


class EvolutionarySearch:
    """Random search with evolutionary improvement over batched episodes.

    Parameters
    ----------
    env
        Environment whose action table and reward are searched.
    seed
        Generator seed.
    length
        Gate-sequence length (default and cap ``env.max_steps``).
    elite_frac
        Fraction of each generation kept unchanged.
    random_frac
        Fraction of each generation drawn fresh at random.
    mutation_rate
        Per-position probability of replacing an action.
    crossover_prob
        Probability of one-point crossover between tournament winners.
    tournament
        Tournament size of parent selection.
    length_penalty
        Fitness penalty per gate, which favors shorter sequences at equal
        reward.
    """

    def __init__(
        self,
        env: CodeDiscoveryEnv,
        seed: int = 0,
        length: int | None = None,
        elite_frac: float = 0.1,
        random_frac: float = 0.2,
        mutation_rate: float = 0.15,
        crossover_prob: float = 0.3,
        tournament: int = 3,
        length_penalty: float = 1e-4,
    ):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.length = env.max_steps if length is None else min(length, env.max_steps)
        self.elite_frac = elite_frac
        self.random_frac = random_frac
        self.mutation_rate = mutation_rate
        self.crossover_prob = crossover_prob
        self.tournament = tournament
        self.length_penalty = length_penalty

    # ------------------------------------------------------------------
    def random_population(self, size: int) -> np.ndarray:
        return self.rng.integers(self.env.num_actions, size=(size, self.length))

    def _mutate(self, seq: np.ndarray) -> np.ndarray:
        seq = seq.copy()
        na = self.env.num_actions
        L = self.length
        m = self.rng.random(L) < self.mutation_rate
        seq[m] = self.rng.integers(na, size=int(m.sum()))
        op = self.rng.random()
        if op < 0.3 and L > 1:  # insertion: shift right from a random point
            i = self.rng.integers(L)
            seq[i + 1 :] = seq[i:-1].copy()
            seq[i] = self.rng.integers(na)
        elif op < 0.6 and L > 1:  # deletion: shift left, pad with no-op
            i = self.rng.integers(L)
            seq[i:-1] = seq[i + 1 :].copy()
            seq[-1] = -1
        return seq

    def _select(self, fitness: np.ndarray) -> int:
        cand = self.rng.integers(fitness.shape[0], size=self.tournament)
        return int(cand[np.argmax(fitness[cand])])

    def next_population(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        size = pop.shape[0]
        n_elite = max(1, round(self.elite_frac * size))
        n_random = round(self.random_frac * size)
        n_child = size - n_elite - n_random
        order = np.argsort(-fitness)
        new = [pop[order[:n_elite]]]
        children = np.empty((n_child, self.length), dtype=np.int64)
        for c in range(n_child):
            a = pop[self._select(fitness)]
            if self.rng.random() < self.crossover_prob:
                b = pop[self._select(fitness)]
                cut = self.rng.integers(1, self.length) if self.length > 1 else 0
                a = np.concatenate([a[:cut], b[cut:]])
            children[c] = self._mutate(a)
        new.append(children)
        if n_random:
            new.append(self.random_population(n_random))
        return np.concatenate(new)

    # ------------------------------------------------------------------
    def run(
        self,
        max_evaluations: int,
        target_reward: float = 0.0,
        stop_on_success: bool = True,
        verbose: bool = False,
    ) -> SearchResult:
        env = self.env
        size = env.batch_size
        t0 = time.perf_counter()
        pop = self.random_population(size)
        best_reward = -np.inf
        best_seq = None
        best_used = self.length
        best_ok = False
        evaluations = 0
        generations = 0
        history = []
        success = False
        while evaluations < max_evaluations:
            reward, ok, used = env.rollout(pop)
            evaluations += size
            generations += 1
            fitness = reward - self.length_penalty * used
            # the best element is the fittest successful one if any element
            # succeeded (an LDPC-constrained environment can hold a state at
            # the reward threshold without granting success), else the fittest
            if ok.any():
                cand = np.flatnonzero(ok)
                i = int(cand[np.argmax(fitness[cand])])
            else:
                i = int(np.argmax(fitness))
            better = (
                best_seq is None
                or (bool(ok[i]) and not best_ok)
                or (bool(ok[i]) == best_ok and reward[i] > best_reward)
                or (bool(ok[i]) == best_ok and reward[i] == best_reward and used[i] < best_used)
            )
            if better:
                best_reward = float(reward[i])
                best_ok = bool(ok[i])
                # the executed history, not the proposed sequence: with the
                # environment's LDPC mask some proposed actions become no-ops
                best_seq = env.history[i, : used[i]].copy()
                best_used = int(used[i])
            history.append(float(reward.max()))
            if verbose:
                print(f"gen {generations:4d} evals {evaluations:7d} best {best_reward:+.4f} mean {reward.mean():+.4f}")
            if best_ok and best_reward >= target_reward:
                success = True
                if stop_on_success:
                    break
            pop = self.next_population(pop, fitness)
        wall = time.perf_counter() - t0
        actions = [int(a) for a in best_seq if a >= 0] if best_seq is not None else []
        gates = [env.action_table.entries[a] for a in actions]
        return SearchResult(
            success=bool(success and best_reward >= target_reward),
            best_reward=best_reward,
            actions=actions,
            gates=gates,
            evaluations=evaluations,
            generations=generations,
            wall_time_s=wall,
            seed=self.seed,
            history=history,
        )


class RandomSearch(EvolutionarySearch):
    """Pure random search: every generation is a fresh random population.

    Parameters
    ----------
    env
        Environment searched.
    seed
        Generator seed.
    length
        Gate-sequence length (default and cap ``env.max_steps``).
    """

    def __init__(self, env: CodeDiscoveryEnv, seed: int = 0, length: int | None = None):
        super().__init__(env, seed=seed, length=length, elite_frac=0.0, random_frac=1.0)

    def next_population(self, pop, fitness):
        return self.random_population(pop.shape[0])
