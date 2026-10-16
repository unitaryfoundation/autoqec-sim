# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Actor-critic networks for the PPO agent.

Two policies share the interface ``model(obs, mask=None) -> (logits,
value)`` with ``obs`` the flat ``{-1, +1}`` observation of
:class:`autoqec_sim.agents.ppo.EnvAdapter`, ``mask`` an optional boolean
``(batch, num_actions)`` tensor, ``logits`` of shape ``(batch,
num_actions)`` in the order of the environment's action table, and
``value`` of shape ``(batch,)``. Masked actions receive the logit
``MASK_LOGIT``.

``mlp``
    Flat MLP trunk over the whole observation with a linear policy head
    and a linear value head. Its first layer has ``obs_dim x hidden``
    weights, so the parameter count grows as ``n^2`` and the policy has no
    notion of which input bit belongs to which qubit.
``equivariant``
    Per-qubit shared MLP over the qubit's two tableau columns plus a
    mean-pooled context, per-gate heads on the qubit embedding
    (single-qubit gates) and on the ordered pair of embeddings (two-qubit
    gates), value head on the mean and max pooled embeddings. Relabelling
    the qubits permutes the logits accordingly. This is the policy the RL
    scaling probe found to reach ``[[16, 1, 4]]`` on every seed where the
    flat MLP at batch 256 never did (``experiments/scaling/REPORT.md``).

:func:`resolve_policy` implements the ``"auto"`` choice of
:class:`autoqec_sim.agents.ppo.PPOConfig`: the equivariant network from
``EQUIVARIANT_MIN_QUBITS`` qubits on, the flat MLP below.

torch is imported inside the builders so that the module imports without
it.
"""

from __future__ import annotations

import functools

import numpy as np

MASK_LOGIT = -1.0e9
POLICIES = ("mlp", "equivariant", "auto")
EQUIVARIANT_MIN_QUBITS = 12


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("autoqec_sim.agents.policies needs torch; install with `uv sync --extra rl`") from exc
    return torch


def resolve_policy(policy: str, n: int) -> str:
    """``"mlp"`` or ``"equivariant"`` for a ``PPOConfig.policy`` value and
    the qubit count of the environment."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if policy == "auto":
        return "equivariant" if int(n) >= EQUIVARIANT_MIN_QUBITS else "mlp"
    return policy


# ----------------------------------------------------------------------
# flat MLP
# ----------------------------------------------------------------------
@functools.cache
def mlp_actor_critic_class():
    torch = _torch()
    nn = torch.nn

    class ActorCritic(nn.Module):
        """MLP trunk with a policy head (logits) and a value head."""

        def __init__(self, obs_dim: int, num_actions: int, hidden=(256, 256)):
            super().__init__()
            layers = []
            d = obs_dim
            for h in hidden:
                lin = nn.Linear(d, h)
                nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
                nn.init.zeros_(lin.bias)
                layers += [lin, nn.Tanh()]
                d = h
            self.trunk = nn.Sequential(*layers)
            self.pi = nn.Linear(d, num_actions)
            nn.init.orthogonal_(self.pi.weight, gain=0.01)
            nn.init.zeros_(self.pi.bias)
            self.v = nn.Linear(d, 1)
            nn.init.orthogonal_(self.v.weight, gain=1.0)
            nn.init.zeros_(self.v.bias)

        def forward(self, obs, mask=None):
            h = self.trunk(obs)
            logits = self.pi(h)
            if mask is not None:
                logits = torch.where(mask, logits, torch.full_like(logits, MASK_LOGIT))
            return logits, self.v(h).squeeze(-1)

        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

    return ActorCritic


def mlp_actor_critic(obs_dim: int, num_actions: int, hidden=(256, 256)):
    """Flat MLP actor-critic over a flat observation of ``obs_dim`` bits."""
    return mlp_actor_critic_class()(obs_dim, num_actions, tuple(hidden))


# ----------------------------------------------------------------------
# permutation-equivariant per-qubit policy
# ----------------------------------------------------------------------
def equivariant_actor_critic(n: int, entries: list[tuple], hidden: int = 128, rows: int | None = None):
    """Permutation-equivariant actor-critic over the tableau.

    ``entries`` is the environment's action table (``(gate, q1, q2)`` with
    ``q2 = -1`` for single-qubit gates); logits are assembled in its order.
    The observation is the flat ``rows x 2n`` block of X and Z bits
    (``rows = 2n`` for the ``tableau`` observation, ``n - k`` for
    ``check_matrix``). Qubit ``q`` gets the ``2 rows`` features of its two
    columns (X bit and Z bit of every row), a shared MLP embeds them, a
    mean over qubits gives the context, and a second shared MLP maps
    ``[own, context]`` to the qubit embedding. Single-qubit gate logits
    come from a per-gate linear head on the qubit embedding; two-qubit gate
    logits from a head on the concatenated embeddings of the ordered pair;
    the value from the mean and max pooled embeddings.
    """
    torch = _torch()
    nn = torch.nn
    n = int(n)
    rows = 2 * n if rows is None else int(rows)
    entries = [(g, int(a), int(b)) for g, a, b in entries]
    kinds_1q = sorted({e[0] for e in entries if e[2] < 0})
    kinds_2q = sorted({e[0] for e in entries if e[2] >= 0})

    class Equivariant(nn.Module):
        def __init__(self):
            super().__init__()
            h = hidden
            self.n = n
            self.rows = rows
            self.phi = nn.Sequential(nn.Linear(2 * rows, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
            self.psi = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh())
            self.head_1q = nn.Linear(h, max(len(kinds_1q), 1))
            self.head_2q = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh(), nn.Linear(h, max(len(kinds_2q), 1)))
            self.v = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh(), nn.Linear(h, 1))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.head_1q.weight, gain=0.01)
            nn.init.orthogonal_(self.head_2q[-1].weight, gain=0.01)
            # every action is (gate slot, qubit) or (gate slot, ordered pair)
            pairs = sorted({(e[1], e[2]) for e in entries if e[2] >= 0})
            self.register_buffer("pair_a", torch.tensor([a for a, _ in pairs] or [0], dtype=torch.long))
            self.register_buffer("pair_b", torch.tensor([b for _, b in pairs] or [0], dtype=torch.long))
            pair_index = {p: i for i, p in enumerate(pairs)}
            # flat gather indices into the concatenation of the (n, n1q) and (E, n2q) logit blocks
            n1 = max(len(kinds_1q), 1)
            n2 = max(len(kinds_2q), 1)
            flat = []
            for e in entries:
                if e[2] < 0:
                    flat.append(e[1] * n1 + kinds_1q.index(e[0]))
                else:
                    flat.append(n * n1 + pair_index[(e[1], e[2])] * n2 + kinds_2q.index(e[0]))
            self.register_buffer("gather", torch.tensor(flat, dtype=torch.long))

        def embed(self, obs):
            b = obs.shape[0]
            n = self.n
            x = obs.view(b, self.rows, 2 * n)
            feats = torch.cat([x[:, :, :n].transpose(1, 2), x[:, :, n:].transpose(1, 2)], dim=2)  # (b, n, 2 rows)
            own = self.phi(feats)
            ctx = own.mean(dim=1, keepdim=True).expand(-1, n, -1)
            return self.psi(torch.cat([own, ctx], dim=2))

        def forward(self, obs, mask=None):
            e = self.embed(obs)
            b = e.shape[0]
            one = self.head_1q(e).reshape(b, -1)  # (b, n * n1q)
            pair = torch.cat([e[:, self.pair_a], e[:, self.pair_b]], dim=2)  # (b, E, 2h)
            two = self.head_2q(pair).reshape(b, -1)  # (b, E * n2q)
            logits = torch.cat([one, two], dim=1)[:, self.gather]
            if mask is not None:
                logits = torch.where(mask, logits, torch.full_like(logits, MASK_LOGIT))
            pooled = torch.cat([e.mean(dim=1), e.max(dim=1).values], dim=1)
            return logits, self.v(pooled).squeeze(-1)

        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

    return Equivariant()
