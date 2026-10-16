# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""torch versions of the tableau gate kernels and of the parity masks.

Every function here has the name and the semantics of its numpy counterpart
in ``autoqec_sim.tableau``, ``autoqec_sim.symplectic``, and
``autoqec_sim.reward``, but takes and returns ``torch.Tensor`` objects on
whatever device the tableau lives on (``cpu`` or ``cuda``). The tableau
layout is the same ``(batch, 2n, 2n + 1)`` ``uint8`` array, so
``torch.from_numpy(tab)`` and ``tab.cpu().numpy()`` move a state between
the two backends without conversion.

Parity test. The numpy path computes the float32 product of the error set
against the code rows, casts it to int64, masks the low bit, and runs two
boolean ``any()`` reductions; on the CPU the cast and the reductions are
about 80 percent of the reward (``docs/profile.md``). Here the product is
reduced mod 2 in place in float32 (``remainder_(2)``, exact because the
entries are integers below ``2n``) and the two commutation tests are one
matrix-vector product against a weight vector: stabilizer columns weigh
``2k + 1`` and logical columns ``1``, so the score of an error is ``s (2k
+ 1) + l`` with ``s`` the number of anticommuting stabilizers and ``l`` the
number of anticommuting logicals, and the error is undetected iff ``1 <=
score <= 2k`` (``s = 0`` and ``l >= 1``). The intermediate is one float32
``(batch, chunk, n + k)`` block, a third of the numpy path's peak, and no
boolean ``(batch, chunk, n + k)`` array exists at all.

Gate kernels are the gather and scatter operations of the numpy module on
torch tensors. Grouping a vector of actions by gate type needs one
``nonzero`` per gate type, which on ``cuda`` synchronizes with the host;
the gate kernels are about one percent of an environment step, so this is
accepted rather than fused.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ..errors import ErrorSet, paulis_of_weight, paulis_up_to_weight
from ..tableau import GATES
from ..tableau import ActionTable as NumpyActionTable

_TWO_QUBIT_NAMES = ("CNOT", "CZ", "SWAP")


def identity_tableau(n: int, batch_size: int, device=None) -> torch.Tensor:
    """Tableau tensor of the identity circuit, replicated ``batch_size`` times."""
    tab = torch.zeros((batch_size, 2 * n, 2 * n + 1), dtype=torch.uint8, device=device)
    tab[:, :, : 2 * n] = torch.eye(2 * n, dtype=torch.uint8, device=device)
    return tab


def from_numpy(tab: np.ndarray, device=None) -> torch.Tensor:
    """A numpy tableau (or any uint8 array) as a tensor on ``device``. On
    ``cpu`` the tensor shares the array's memory; on ``cuda`` it is a copy."""
    return torch.as_tensor(np.ascontiguousarray(tab), device=device)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def num_qubits(tab: torch.Tensor) -> int:
    return (tab.shape[-1] - 1) // 2


def _select(tab: torch.Tensor, where) -> torch.Tensor:
    if where is None:
        return torch.arange(tab.shape[0], device=tab.device)
    where = torch.as_tensor(where, device=tab.device)
    if where.dtype == torch.bool:
        return torch.nonzero(where, as_tuple=True)[0]
    return where.to(torch.long)


def _per_element(q, m: int, device) -> torch.Tensor:
    q = torch.as_tensor(q, dtype=torch.long, device=device)
    if q.ndim == 0:
        return q.expand(m)
    return q


# --------------------------------------------------------------------------
# single-qubit gates
# --------------------------------------------------------------------------
def apply_h(tab, b, q):
    n = num_qubits(tab)
    x = tab[b, :, q]
    z = tab[b, :, n + q]
    tab[b, :, 2 * n] ^= x & z
    tab[b, :, q] = z
    tab[b, :, n + q] = x


def apply_s(tab, b, q):
    n = num_qubits(tab)
    x = tab[b, :, q]
    z = tab[b, :, n + q]
    tab[b, :, 2 * n] ^= x & z
    tab[b, :, n + q] = z ^ x


def apply_s_dag(tab, b, q):
    n = num_qubits(tab)
    x = tab[b, :, q]
    z = tab[b, :, n + q]
    tab[b, :, 2 * n] ^= x & (z ^ 1)
    tab[b, :, n + q] = z ^ x


def apply_x(tab, b, q):
    n = num_qubits(tab)
    tab[b, :, 2 * n] ^= tab[b, :, n + q]


def apply_z(tab, b, q):
    n = num_qubits(tab)
    tab[b, :, 2 * n] ^= tab[b, :, q]


def apply_y(tab, b, q):
    n = num_qubits(tab)
    tab[b, :, 2 * n] ^= tab[b, :, q] ^ tab[b, :, n + q]


# --------------------------------------------------------------------------
# two-qubit gates
# --------------------------------------------------------------------------
def apply_cnot(tab, b, c, t):
    n = num_qubits(tab)
    xc = tab[b, :, c]
    zc = tab[b, :, n + c]
    xt = tab[b, :, t]
    zt = tab[b, :, n + t]
    tab[b, :, 2 * n] ^= xc & zt & (xt ^ zc ^ 1)
    tab[b, :, t] = xt ^ xc
    tab[b, :, n + c] = zc ^ zt


def apply_cz(tab, b, a, c):
    n = num_qubits(tab)
    xa = tab[b, :, a]
    za = tab[b, :, n + a]
    xc = tab[b, :, c]
    zc = tab[b, :, n + c]
    tab[b, :, 2 * n] ^= xa & xc & (za ^ zc)
    tab[b, :, n + a] = za ^ xc
    tab[b, :, n + c] = zc ^ xa


def apply_swap(tab, b, a, c):
    n = num_qubits(tab)
    xa = tab[b, :, a].clone()
    za = tab[b, :, n + a].clone()
    tab[b, :, a] = tab[b, :, c]
    tab[b, :, n + a] = tab[b, :, n + c]
    tab[b, :, c] = xa
    tab[b, :, n + c] = za


_ONE_QUBIT = {
    "H": apply_h,
    "S": apply_s,
    "S_DAG": apply_s_dag,
    "X": apply_x,
    "Y": apply_y,
    "Z": apply_z,
}
_TWO_QUBIT = {"CNOT": apply_cnot, "CZ": apply_cz, "SWAP": apply_swap}


def apply_gate(tab: torch.Tensor, gate: str, q1, q2=None, where=None) -> None:
    """Apply ``gate`` in place to the batch elements selected by ``where``
    (same contract as ``tableau.apply_gate``)."""
    b = _select(tab, where)
    if b.numel() == 0:
        return
    m = b.numel()
    q1 = _per_element(q1, m, tab.device)
    if gate in _ONE_QUBIT:
        _ONE_QUBIT[gate](tab, b, q1)
    elif gate in _TWO_QUBIT:
        if q2 is None:
            raise ValueError(f"{gate} needs two qubits")
        q2 = _per_element(q2, m, tab.device)
        if bool((q1 == q2).any()):
            raise ValueError("two-qubit gate on identical qubits")
        _TWO_QUBIT[gate](tab, b, q1, q2)
    else:
        raise ValueError(f"unknown gate {gate!r}")


class ActionTable:
    """``tableau.ActionTable`` with its three lookup arrays on a device.

    Construct from an existing numpy ``ActionTable`` or from the entry list.
    ``apply`` takes the batch of action indices as a tensor on the same
    device as the tableau (a numpy array or list is moved there first).
    """

    def __init__(self, entries, device=None):
        base = entries if isinstance(entries, NumpyActionTable) else NumpyActionTable(entries)
        self.base = base
        self.entries = base.entries
        self.gate_names = base.gate_names
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.gate_id = torch.as_tensor(base.gate_id, device=self.device)
        self.q1 = torch.as_tensor(base.q1, dtype=torch.long, device=self.device)
        self.q2 = torch.as_tensor(base.q2, dtype=torch.long, device=self.device)

    def __len__(self):
        return len(self.entries)

    def apply(self, tab: torch.Tensor, actions, where=None) -> None:
        actions = torch.as_tensor(actions, dtype=torch.long, device=tab.device)
        active = actions >= 0
        if where is not None:
            active &= torch.as_tensor(where, dtype=torch.bool, device=tab.device)
        if not bool(active.any()):
            return
        gid = torch.full(actions.shape, -1, dtype=torch.long, device=tab.device)
        gid[active] = self.gate_id[actions[active]]
        for g_idx, gate in enumerate(self.gate_names):
            sel = torch.nonzero(gid == g_idx, as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            acts = actions[sel]
            q1 = self.q1[acts]
            q2 = self.q2[acts] if gate in _TWO_QUBIT_NAMES else None
            apply_gate(tab, gate, q1, q2, where=sel)


# --------------------------------------------------------------------------
# symplectic product and parity masks
# --------------------------------------------------------------------------
def swap_halves(v: torch.Tensor) -> torch.Tensor:
    n = v.shape[-1] // 2
    return torch.cat([v[..., n:], v[..., :n]], dim=-1)


def symplectic_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise symplectic products mod 2, ``(..., M, K)`` uint8; the
    leading axes broadcast as in ``symplectic.symplectic_product``."""
    a32 = a.to(torch.float32)
    b32 = swap_halves(b).to(torch.float32)
    prod = torch.matmul(a32, b32.transpose(-1, -2))
    prod.remainder_(2)
    return prod.to(torch.uint8)


def code_rows(tab: torch.Tensor, k: int) -> torch.Tensor:
    """``(batch, n + k, 2n)``: stabilizers ``k..n-1`` then logical X and Z."""
    n = num_qubits(tab)
    stab = tab[:, n + k : 2 * n, : 2 * n]
    lx = tab[:, :k, : 2 * n]
    lz = tab[:, n : n + k, : 2 * n]
    return torch.cat([stab, lx, lz], dim=1)


def _rows32_swapped(tab: torch.Tensor, k: int) -> torch.Tensor:
    """``(batch, 2n, n + k)`` float32: the code rows with their halves
    swapped and transposed, the right factor of the product."""
    return swap_halves(code_rows(tab, k)).to(torch.float32).transpose(-1, -2)


def _errors32(errors, device) -> torch.Tensor:
    e = torch.as_tensor(errors, device=device)
    return e.to(torch.float32)


def _score_weights(n: int, k: int, device) -> torch.Tensor:
    w = torch.ones(n + k, dtype=torch.float32, device=device)
    w[: n - k] = 2 * k + 1
    return w


def undetected_mask(tab: torch.Tensor, k: int, errors, chunk: int = 1 << 16) -> torch.Tensor:
    """``K_mu`` for every batch element and error: ``(batch, M)`` bool.

    ``errors`` is ``(M, 2n)`` (uint8 tensor on the tableau's device, or a
    numpy array, which is moved there). Peak intermediate: one float32
    ``(batch, min(M, chunk), n + k)`` block.
    """
    n = num_qubits(tab)
    rows_t = _rows32_swapped(tab, k)  # (B, 2n, n+k)
    e32 = _errors32(errors, tab.device)  # (M, 2n)
    w = _score_weights(n, k, tab.device)
    batch, m = tab.shape[0], e32.shape[0]
    out = torch.empty((batch, m), dtype=torch.bool, device=tab.device)
    for start in range(0, m, chunk):
        e = e32[start : start + chunk]
        prod = torch.matmul(e.unsqueeze(0), rows_t)  # (B, m_c, n+k)
        prod.remainder_(2)
        score = torch.matmul(prod, w)  # (B, m_c)
        out[:, start : start + e.shape[0]] = (score >= 1) & (score <= 2 * k)
    return out


def in_stabilizer_mask(tab: torch.Tensor, k: int, paulis, chunk: int = 1 << 16) -> torch.Tensor:
    """``(batch, M)`` bool: which Paulis lie in the stabilizer group."""
    n = num_qubits(tab)
    rows_t = _rows32_swapped(tab, k)
    e32 = _errors32(paulis, tab.device)
    w = torch.ones(n + k, dtype=torch.float32, device=tab.device)
    batch, m = tab.shape[0], e32.shape[0]
    out = torch.empty((batch, m), dtype=torch.bool, device=tab.device)
    for start in range(0, m, chunk):
        e = e32[start : start + chunk]
        prod = torch.matmul(e.unsqueeze(0), rows_t)
        prod.remainder_(2)
        out[:, start : start + e.shape[0]] = torch.matmul(prod, w) == 0
    return out


def reward_intermediate_bytes(n: int, k: int, m: int) -> int:
    """Peak bytes per batch element inside ``undetected_mask``: the float32
    product ``(chunk, n + k)`` reduced in place (no int64 copy)."""
    return m * (n + k) * 4


def kl_reward(tab: torch.Tensor, k: int, errors: ErrorSet, normalize: bool = True, chunk: int = 1 << 16):
    """``r = -sum_mu lam_mu K_mu`` per batch element, ``(batch,)`` float64."""
    und = undetected_mask(tab, k, errors.paulis, chunk)
    lam = torch.as_tensor(errors.lam, dtype=torch.float64, device=tab.device)
    r = -(und.to(torch.float64) @ lam)
    if normalize:
        total = float(errors.lam.sum())
        if total > 0:
            r = r / total
    return r


def batched_distance(tab: torch.Tensor, k: int, max_weight: int, chunk: int = 1 << 16) -> torch.Tensor:
    """Exact distance of every batch element, capped at ``max_weight + 1``
    (``(batch,)`` int64 on the tableau's device)."""
    batch = tab.shape[0]
    n = num_qubits(tab)
    d = torch.full((batch,), max_weight + 1, dtype=torch.long, device=tab.device)
    undecided = torch.ones(batch, dtype=torch.bool, device=tab.device)
    for w in range(1, max_weight + 1):
        if not bool(undecided.any()):
            break
        e = torch.as_tensor(paulis_of_weight(n, w), device=tab.device)
        idx = torch.nonzero(undecided, as_tuple=True)[0]
        und = undetected_mask(tab[idx], k, e, chunk).any(dim=1)
        hit = idx[und]
        d[hit] = w
        undecided[hit] = False
    return d


def distance_reward(tab: torch.Tensor, k: int, target_distance: int, chunk: int = 1 << 16) -> torch.Tensor:
    d = batched_distance(tab, k, target_distance - 1, chunk)
    return torch.clamp(d, max=target_distance).to(torch.float64) / float(target_distance)


def low_weight_stabilizer_penalty(tab: torch.Tensor, k: int, max_weight: int, chunk: int = 1 << 16) -> torch.Tensor:
    e, _ = paulis_up_to_weight(num_qubits(tab), max_weight)
    if e.shape[0] == 0:
        return torch.zeros(tab.shape[0], dtype=torch.float64, device=tab.device)
    return -in_stabilizer_mask(tab, k, e, chunk).to(torch.float64).mean(dim=1)


class CompositeReward:
    """Weighted sum of reward callables ``f(tab, k) -> (batch,)`` tensors."""

    def __init__(self, terms: list[tuple[float, Callable[[torch.Tensor, int], torch.Tensor]]]):
        self.terms = terms

    def __call__(self, tab: torch.Tensor, k: int) -> torch.Tensor:
        out = torch.zeros(tab.shape[0], dtype=torch.float64, device=tab.device)
        for wgt, fn in self.terms:
            out += wgt * fn(tab, k)
        return out


def make_kl_reward(errors: ErrorSet, normalize: bool = True, device=None):
    """Closure ``f(tab, k)`` with the error set and its weights resident on
    ``device`` (the tableau's device when ``None``)."""
    cache: dict = {}

    def f(tab, k):
        dev = tab.device
        if dev not in cache:
            cache[dev] = (
                torch.as_tensor(errors.paulis, device=dev),
                torch.as_tensor(errors.lam, dtype=torch.float64, device=dev),
                float(errors.lam.sum()),
            )
        e, lam, total = cache[dev]
        r = -(undetected_mask(tab, k, e).to(torch.float64) @ lam)
        if normalize and total > 0:
            r = r / total
        return r

    if device is not None:
        dev = torch.device(device)
        cache[dev] = (
            torch.as_tensor(errors.paulis, device=dev),
            torch.as_tensor(errors.lam, dtype=torch.float64, device=dev),
            float(errors.lam.sum()),
        )
    return f


def make_low_weight_penalty(max_weight: int):
    def f(tab, k):
        return low_weight_stabilizer_penalty(tab, k, max_weight)

    return f


def make_distance_reward(target_distance: int):
    def f(tab, k):
        return distance_reward(tab, k, target_distance)

    return f


def numpy_reward_adapter(fn, device=None):
    """Wrap a torch reward ``f(tab, k) -> tensor`` as a numpy reward
    ``f(tab: np.ndarray, k) -> np.ndarray`` for ``CodeDiscoveryEnv``: the
    tableau is copied to ``device`` per call and the result copied back."""

    def g(tab: np.ndarray, k: int) -> np.ndarray:
        return to_numpy(fn(from_numpy(tab, device), k))

    return g


__all__ = [
    "GATES",
    "ActionTable",
    "CompositeReward",
    "apply_cnot",
    "apply_cz",
    "apply_gate",
    "apply_h",
    "apply_s",
    "apply_s_dag",
    "apply_swap",
    "apply_x",
    "apply_y",
    "apply_z",
    "batched_distance",
    "code_rows",
    "distance_reward",
    "from_numpy",
    "identity_tableau",
    "in_stabilizer_mask",
    "kl_reward",
    "low_weight_stabilizer_penalty",
    "make_distance_reward",
    "make_kl_reward",
    "make_low_weight_penalty",
    "num_qubits",
    "numpy_reward_adapter",
    "reward_intermediate_bytes",
    "swap_halves",
    "symplectic_product",
    "to_numpy",
    "undetected_mask",
]
