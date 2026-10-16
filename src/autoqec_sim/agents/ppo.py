# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Proximal policy optimization over the batched code-discovery environment.

The agent follows Olle et al. (arXiv:2311.04750) in spirit: a policy over
discrete gate choices, conditioned on the current stabilizer tableau, and
trained on the Knill-Laflamme reward that the environment returns at every
step. The batch dimension of :class:`autoqec_sim.env.CodeDiscoveryEnv` is
the vectorized environment; one call to ``env.step`` advances every
episode by one gate.

Components (Schulman et al., arXiv:1707.06347):

- clipped surrogate objective with ratio clip ``clip_eps``;
- generalized advantage estimation with ``gamma`` and ``gae_lambda``;
- a value head sharing the trunk with the policy, squared-error loss
  weighted by ``vf_coef``;
- an entropy bonus weighted by ``ent_coef``.

Action masking. The environment's action table already contains only the
gates the connectivity graph allows, so nothing there needs masking. The
agent still applies a mask through :class:`EnvAdapter`: (i) a static mask
from :func:`graph_action_mask` when the table has actions on edges outside
a given graph (a policy trained on all-to-all and evaluated on a lattice),
(ii) a dynamic mask that forbids undoing the previous gate when it is an
involution (H, CNOT, CZ, SWAP, Paulis), and (iii) any mask the environment
itself exposes through an ``action_mask()`` method returning a ``(batch,
num_actions)`` boolean array. Masked actions receive the logit
``MASK_LOGIT`` (a large negative number) before sampling.

Policies. ``PPOConfig.policy`` chooses the actor-critic from
:mod:`autoqec_sim.agents.policies`: the flat MLP (``"mlp"``), the per-qubit
permutation-equivariant network (``"equivariant"``), or ``"auto"`` (the
default: equivariant from 12 qubits on, mlp below). Both take the same
observation and mask from :class:`EnvAdapter` and return logits in the
order of the environment's action table plus a value.

torch is imported lazily so that the rest of the package works without it.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..env import CodeDiscoveryEnv
from ..graphs import Graph
from ..tableau import ActionTable
from .policies import (
    EQUIVARIANT_MIN_QUBITS,
    MASK_LOGIT,
    POLICIES,
    equivariant_actor_critic,
    mlp_actor_critic,
    mlp_actor_critic_class,
    resolve_policy,
)

__all__ = [
    "EQUIVARIANT_MIN_QUBITS",
    "INVOLUTIONS",
    "MASK_LOGIT",
    "POLICIES",
    "EnvAdapter",
    "FoundCode",
    "PPOAgent",
    "PPOConfig",
    "PPOResult",
    "equivariant_actor_critic",
    "graph_action_mask",
    "mlp_actor_critic",
    "resolve_policy",
    "set_threads",
]
INVOLUTIONS = frozenset({"H", "CNOT", "CZ", "SWAP", "X", "Y", "Z"})


# ----------------------------------------------------------------------
# lazy torch
# ----------------------------------------------------------------------
def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "autoqec_sim.agents.ppo needs torch; install with `uv sync --extra rl`"
        ) from exc
    return torch


def set_threads(max_threads: int = 6) -> int:
    """Cap BLAS and torch threads (CPU only). Returns the thread count."""
    n = max(1, min(int(max_threads), os.cpu_count() or 1))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
    torch = _torch()
    torch.set_num_threads(n)
    return n


# ----------------------------------------------------------------------
# environment adapter
# ----------------------------------------------------------------------
def graph_action_mask(table: ActionTable, graph: Graph) -> np.ndarray:
    """Boolean ``(num_actions,)`` mask, true where the action is allowed on
    ``graph``: every single-qubit gate on a qubit of the graph, every
    two-qubit gate whose pair is an edge (either orientation)."""
    allowed = {(min(a, b), max(a, b)) for a, b in graph.edges}
    mask = np.zeros(len(table), dtype=bool)
    for i, (g, a, b) in enumerate(table.entries):
        if b < 0:
            mask[i] = a < graph.n
        else:
            mask[i] = (min(a, b), max(a, b)) in allowed
    return mask


class EnvAdapter:
    """Observation encoding and action masks for :class:`CodeDiscoveryEnv`.

    ``observation``: ``"tableau"`` encodes the ``2n x 2n`` X and Z bits of
    the whole tableau (destabilizers and stabilizers; the sign column is
    dropped since the reward does not depend on it), ``"check_matrix"``
    encodes only the ``(n - k) x 2n`` stabilizer block (the observation of
    Olle et al.). Bits are mapped to ``{-1, +1}`` floats.
    """

    def __init__(
        self,
        env: CodeDiscoveryEnv,
        observation: str = "tableau",
        mask_undo: bool = True,
        mask_graph: Graph | None = None,
    ):
        self.env = env
        if observation not in ("tableau", "check_matrix"):
            raise ValueError("observation must be tableau or check_matrix")
        self.observation = observation
        self.mask_undo = bool(mask_undo)
        n, k = env.n, env.k
        self.obs_dim = (2 * n) * (2 * n) if observation == "tableau" else (n - k) * 2 * n
        self.num_actions = env.num_actions
        table = env.action_table
        self.involution = np.array([g in INVOLUTIONS for g, _, _ in table.entries], dtype=bool)
        self.static_mask = (
            graph_action_mask(table, mask_graph) if mask_graph is not None else None
        )
        if self.static_mask is not None and not self.static_mask.any():
            raise ValueError("graph mask forbids every action")
        self.env_mask_fn = getattr(env, "action_mask", None)
        if self.env_mask_fn is not None and not callable(self.env_mask_fn):
            self.env_mask_fn = None

    def encode(self, tab: np.ndarray) -> np.ndarray:
        n, k = self.env.n, self.env.k
        bits = tab[:, :, : 2 * n] if self.observation == "tableau" else tab[:, n + k : 2 * n, : 2 * n]
        return (bits.reshape(tab.shape[0], -1).astype(np.float32) * 2.0) - 1.0

    def action_mask(self) -> np.ndarray | None:
        """``(batch, num_actions)`` boolean mask of allowed actions, or
        ``None`` if nothing is masked."""
        env = self.env
        mask = None
        if self.static_mask is not None:
            mask = np.broadcast_to(self.static_mask, (env.batch_size, self.num_actions)).copy()
        if self.mask_undo:
            has_prev = env.steps > 0
            if has_prev.any():
                if mask is None:
                    mask = np.ones((env.batch_size, self.num_actions), dtype=bool)
                idx = np.flatnonzero(has_prev)
                last = env.history[idx, env.steps[idx] - 1]
                ok = (last >= 0) & self.involution[np.maximum(last, 0)]
                mask[idx[ok], last[ok]] = False
        if self.env_mask_fn is not None:
            m = np.asarray(self.env_mask_fn(), dtype=bool)
            if m.shape == (env.batch_size, self.num_actions):
                mask = m if mask is None else (mask & m)
        if mask is not None:
            # never mask everything for an element
            empty = ~mask.any(axis=1)
            if empty.any():
                mask[empty] = True
        return mask


# ----------------------------------------------------------------------
# network (see policies.py)
# ----------------------------------------------------------------------
_actor_critic_class = mlp_actor_critic_class  # former name of the flat MLP class builder


# ----------------------------------------------------------------------
# configuration and results
# ----------------------------------------------------------------------
@dataclass
class PPOConfig:
    """Hyperparameters of :class:`PPOAgent`.

    ``policy`` selects the actor-critic (:mod:`autoqec_sim.agents.policies`):
    ``"mlp"`` (flat MLP with layers ``hidden``), ``"equivariant"`` (per-qubit
    permutation-equivariant network of width ``equivariant_hidden``), or
    ``"auto"`` (equivariant when the environment has at least
    ``EQUIVARIANT_MIN_QUBITS`` = 12 qubits, mlp otherwise). The flat MLP at
    ``n >= 12`` needs ``rollout_steps * batch`` of about 16 x 1024 and
    ``ent_coef = 0.001`` to make progress (``experiments/scaling/REPORT.md``).
    """

    hidden: tuple = (256, 256)
    observation: str = "tableau"
    policy: str = "auto"
    equivariant_hidden: int = 128
    rollout_steps: int = 16  # env steps per update; transitions = batch * rollout_steps
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    mask_undo: bool = True
    max_threads: int = 6

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


@dataclass
class FoundCode:
    """One terminated episode whose state reached the success threshold."""

    env_steps: int  # cumulative state evaluations (batch elements) when found
    episodes: int  # cumulative finished episodes when found
    update: int
    actions: list[int]
    gates: list[tuple]
    tableau: np.ndarray
    state_reward: float

    def to_dict(self):
        return {
            "env_steps": self.env_steps,
            "episodes": self.episodes,
            "update": self.update,
            "actions": self.actions,
            "gates": [list(g) for g in self.gates],
            "state_reward": self.state_reward,
            "circuit_length": len(self.gates),
        }


@dataclass
class PPOResult:
    success: bool
    seed: int
    env_steps: int
    episodes: int
    updates: int
    wall_time_s: float
    first_success: FoundCode | None
    successes: int
    num_parameters: int
    config: dict
    history: list[dict] = field(default_factory=list)

    def to_dict(self):
        d = {
            "success": self.success,
            "seed": self.seed,
            "env_steps": self.env_steps,
            "episodes": self.episodes,
            "updates": self.updates,
            "wall_time_s": round(self.wall_time_s, 3),
            "first_success": self.first_success.to_dict() if self.first_success else None,
            "successes": self.successes,
            "num_parameters": self.num_parameters,
            "config": self.config,
            "history": self.history,
        }
        return d


# ----------------------------------------------------------------------
# agent
# ----------------------------------------------------------------------
class PPOAgent:
    """PPO over a :class:`CodeDiscoveryEnv` with ``auto_reset`` semantics.

    The environment is switched to ``auto_reset = True`` and dense reward;
    the episode ends when the state reward reaches
    ``env.success_threshold`` (a code satisfying every term of the reward)
    or after ``env.max_steps`` gates. Truncated episodes are treated as
    terminal for the value bootstrap.

    ``device`` (default ``"cpu"``) is the torch device of the network and
    of the update: observations, masks, and rollout buffers are copied to
    it, actions and values copied back. The environment and its reward
    stay numpy on the host either way. On ``"cpu"`` every copy is a no-op
    and the trajectory is the one of the version without the argument.
    """

    def __init__(self, env: CodeDiscoveryEnv, config: PPOConfig | None = None, seed: int = 0, device=None):
        self.config = config or PPOConfig()
        self.threads = set_threads(self.config.max_threads)
        torch = _torch()
        self.torch = torch
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.env = env
        env.auto_reset = True
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.adapter = EnvAdapter(env, self.config.observation, self.config.mask_undo)
        self.policy = resolve_policy(self.config.policy, env.n)
        self.model = self.build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.env_steps = 0
        self.episodes = 0
        self.updates = 0
        self.successes: list[FoundCode] = []
        self.history: list[dict] = []
        self._obs = None

    # ------------------------------------------------------------------
    def build_model(self):
        """Fresh actor-critic for ``self.policy`` over the adapter's observation
        and the environment's action table (torch RNG state is consumed)."""
        if self.policy == "mlp":
            return mlp_actor_critic(self.adapter.obs_dim, self.adapter.num_actions, tuple(self.config.hidden))
        rows = self.adapter.obs_dim // (2 * self.env.n)
        return equivariant_actor_critic(
            self.env.n, self.env.action_table.entries, hidden=int(self.config.equivariant_hidden), rows=rows
        )

    def num_parameters(self) -> int:
        return self.model.num_parameters()

    def _policy(self, obs_np: np.ndarray, mask_np: np.ndarray | None):
        torch = self.torch
        obs = torch.from_numpy(obs_np).to(self.device)
        mask = torch.from_numpy(mask_np).to(self.device) if mask_np is not None else None
        with torch.no_grad():
            logits, value = self.model(obs, mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    def act(self, tab: np.ndarray, greedy: bool = False) -> np.ndarray:
        """Actions for an arbitrary batch of tableaux (no masking, used for
        inspection of a trained policy)."""
        torch = self.torch
        obs = torch.from_numpy(self.adapter.encode(tab)).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(obs)
            if greedy:
                return logits.argmax(-1).cpu().numpy()
            return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

    # ------------------------------------------------------------------
    def collect(self, steps: int, record_successes: bool = True, max_record: int = 4096):
        """Run ``steps`` environment steps for every batch element and return
        the rollout buffers. Successful terminations are appended to
        ``self.successes`` (at most ``max_record`` per call)."""
        env = self.env
        B = env.batch_size
        if self._obs is None:
            tab, _ = env.reset()
            self._obs = self.adapter.encode(tab)
        obs_buf = np.empty((steps, B, self.adapter.obs_dim), dtype=np.float32)
        mask_buf = np.ones((steps, B, self.adapter.num_actions), dtype=bool)
        act_buf = np.empty((steps, B), dtype=np.int64)
        logp_buf = np.empty((steps, B), dtype=np.float32)
        val_buf = np.empty((steps, B), dtype=np.float32)
        rew_buf = np.empty((steps, B), dtype=np.float32)
        done_buf = np.empty((steps, B), dtype=bool)
        any_mask = False
        recorded = 0
        n_success = 0
        n_done = 0
        for t in range(steps):
            mask = self.adapter.action_mask()
            if mask is not None:
                any_mask = True
                mask_buf[t] = mask
            action, logp, value = self._policy(self._obs, mask)
            tab, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            obs_buf[t] = self._obs
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = value
            rew_buf[t] = reward
            done_buf[t] = done
            self.env_steps += B
            nd = int(done.sum())
            n_done += nd
            self.episodes += nd
            if terminated.any():
                n_success += int(terminated.sum())
                if record_successes and "final_tableau" in info:
                    sel = terminated[done]  # which of the finished ones succeeded
                    ft = info["final_tableau"][sel]
                    fh = info["final_history"][sel]
                    fr = info["final_state_reward"][sel]
                    for i in range(ft.shape[0]):
                        if recorded >= max_record:
                            break
                        acts = [int(a) for a in fh[i] if a >= 0]
                        self.successes.append(
                            FoundCode(
                                env_steps=self.env_steps,
                                episodes=self.episodes,
                                update=self.updates,
                                actions=acts,
                                gates=[env.action_table.entries[a] for a in acts],
                                tableau=ft[i].copy(),
                                state_reward=float(fr[i]),
                            )
                        )
                        recorded += 1
            self._obs = self.adapter.encode(tab)
        # bootstrap value of the last observation
        _, _, last_value = self._policy(self._obs, self.adapter.action_mask())
        return dict(
            obs=obs_buf,
            mask=mask_buf if any_mask else None,
            actions=act_buf,
            logp=logp_buf,
            values=val_buf,
            rewards=rew_buf,
            dones=done_buf,
            last_value=last_value.astype(np.float32),
            n_success=n_success,
            n_done=n_done,
        )

    def gae(self, buf):
        cfg = self.config
        T, B = buf["rewards"].shape
        adv = np.zeros((T, B), dtype=np.float32)
        last = np.zeros(B, dtype=np.float32)
        next_value = buf["last_value"]
        for t in reversed(range(T)):
            nonterminal = 1.0 - buf["dones"][t].astype(np.float32)
            delta = buf["rewards"][t] + cfg.gamma * next_value * nonterminal - buf["values"][t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
            adv[t] = last
            next_value = buf["values"][t]
        returns = adv + buf["values"]
        return adv, returns

    def update(self, buf) -> dict:
        torch = self.torch
        cfg = self.config
        adv, ret = self.gae(buf)
        T, B = buf["rewards"].shape
        N = T * B
        dev = self.device
        obs = torch.from_numpy(buf["obs"].reshape(N, -1)).to(dev)
        mask = torch.from_numpy(buf["mask"].reshape(N, -1)).to(dev) if buf["mask"] is not None else None
        actions = torch.from_numpy(buf["actions"].reshape(N)).to(dev)
        old_logp = torch.from_numpy(buf["logp"].reshape(N)).to(dev)
        adv_t = torch.from_numpy(adv.reshape(N)).to(dev)
        ret_t = torch.from_numpy(ret.reshape(N)).to(dev)
        if cfg.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0, clip_frac=0.0)
        count = 0
        mb = min(cfg.minibatch_size, N)
        for _ in range(cfg.epochs):
            perm = torch.from_numpy(self.rng.permutation(N)).to(dev)
            for start in range(0, N, mb):
                idx = perm[start : start + mb]
                logits, value = self.model(obs[idx], mask[idx] if mask is not None else None)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv_t[idx]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * a
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (value - ret_t[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    stats["policy_loss"] += float(policy_loss)
                    stats["value_loss"] += float(value_loss)
                    stats["entropy"] += float(entropy)
                    stats["approx_kl"] += float((old_logp[idx] - logp).mean())
                    stats["clip_frac"] += float(((ratio - 1.0).abs() > cfg.clip_eps).float().mean())
                count += 1
        for k in stats:
            stats[k] /= max(count, 1)
        self.updates += 1
        return stats

    # ------------------------------------------------------------------
    def train(
        self,
        max_env_steps: int,
        max_time_s: float = 20 * 60,
        stop_on_success: bool = True,
        callback=None,
        verbose: bool = False,
    ) -> PPOResult:
        """Alternate rollouts and updates until ``max_env_steps`` state
        evaluations, ``max_time_s`` seconds, or (with ``stop_on_success``)
        the first success. ``callback(agent, new_successes) -> bool`` runs
        after every update; returning true stops training.

        ``env_steps`` counts batch elements times steps, that is, state
        reward evaluations. One evolutionary-search "evaluation" (one
        circuit) costs up to ``max_steps`` of these.
        """
        t0 = time.perf_counter()
        first = None
        stop = False
        while self.env_steps < max_env_steps and not stop:
            n_before = len(self.successes)
            buf = self.collect(self.config.rollout_steps)
            stats = self.update(buf)
            new = self.successes[n_before:]
            if first is None and new:
                first = new[0]
            rec = dict(
                update=self.updates,
                env_steps=self.env_steps,
                episodes=self.episodes,
                mean_reward=float(buf["rewards"].mean()),
                success_rate=(buf["n_success"] / buf["n_done"]) if buf["n_done"] else 0.0,
                wall_time_s=round(time.perf_counter() - t0, 2),
                **{k: round(v, 5) for k, v in stats.items()},
            )
            self.history.append(rec)
            if verbose:
                print(
                    f"upd {rec['update']:4d} steps {rec['env_steps']:8d} eps {rec['episodes']:7d} "
                    f"r {rec['mean_reward']:+.4f} succ {rec['success_rate']:.3f} "
                    f"ent {rec['entropy']:.3f} kl {rec['approx_kl']:.4f} t {rec['wall_time_s']:.1f}s"
                )
            if callback is not None and callback(self, new):
                stop = True
            if stop_on_success and first is not None:
                stop = True
            if time.perf_counter() - t0 > max_time_s:
                stop = True
        return PPOResult(
            success=first is not None,
            seed=self.seed,
            env_steps=self.env_steps,
            episodes=self.episodes,
            updates=self.updates,
            wall_time_s=time.perf_counter() - t0,
            first_success=first,
            successes=len(self.successes),
            num_parameters=self.num_parameters(),
            config=dict(self.config.to_dict(), policy_resolved=self.policy),
            history=self.history,
        )


def main(argv=None):  # pragma: no cover - thin CLI
    """Train PPO on one ``[[n, k, d]]`` target and print the first code found."""
    import argparse

    from .. import graphs
    from ..code import StabilizerCode

    ap = argparse.ArgumentParser(description="PPO code discovery.")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--graph", default=None, help="graph spec, default all_to_all:n")
    ap.add_argument("--gates", nargs="*", default=["H", "S", "CNOT"])
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--env-steps", type=int, default=2_000_000)
    ap.add_argument("--max-time", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forbid-weight", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    from ..errors import ErrorSet
    from ..reward import CompositeReward, make_kl_reward, make_low_weight_penalty

    graph = graphs.from_spec(args.graph) if args.graph else graphs.all_to_all(args.n)
    reward = CompositeReward(
        [
            (1.0, make_kl_reward(ErrorSet.up_to_weight(args.n, args.d - 1))),
            (1.0, make_low_weight_penalty(args.forbid_weight)),
        ]
    )
    env = CodeDiscoveryEnv(
        n=args.n, k=args.k, graph=graph, batch_size=args.batch, max_steps=args.max_steps,
        gates=tuple(args.gates), reward_fn=reward, seed=args.seed,
    )
    agent = PPOAgent(env, seed=args.seed)
    res = agent.train(args.env_steps, max_time_s=args.max_time, verbose=args.verbose)
    print(f"success={res.success} env_steps={res.env_steps} episodes={res.episodes} wall={res.wall_time_s:.1f}s params={res.num_parameters}")
    if res.first_success:
        code = StabilizerCode.from_tableau(res.first_success.tableau[None], args.k)
        print("gates:", res.first_success.gates)
        print("stabilizers:", code.stabilizer_strings(), "d =", code.distance_exact())
    return 0 if res.success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
