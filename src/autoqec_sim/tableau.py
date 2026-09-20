# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Batched Aaronson-Gottesman stabilizer tableaux.

The state of the whole simulator is one dense array of shape
``(batch, 2n, 2n + 1)`` with dtype ``uint8``. Row ``r < n`` is destabilizer
``r`` and row ``n + r`` is stabilizer ``r``. Columns ``[0, n)`` hold the X
bits, columns ``[n, 2n)`` the Z bits, and column ``2n`` the sign bit (1 means
the Pauli carries a minus sign). A qubit with X bit 1 and Z bit 1 is ``Y``.

The tableau of the identity circuit has destabilizers ``X_0, ..., X_{n-1}``
and stabilizers ``Z_0, ..., Z_{n-1}``. Applying a Clifford circuit ``C``
turns stabilizer row ``r`` into ``C Z_r C^dagger`` and destabilizer row
``r`` into ``C X_r C^dagger``. In the encoding convention of Olle et al.
(arXiv:2311.04750) the first ``k`` qubits carry the logical state and the
remaining ``n - k`` start in ``|0>``, so after the circuit the stabilizer
rows ``k, ..., n - 1`` generate the code and rows ``0, ..., k - 1`` of the
stabilizer and destabilizer blocks are the logical ``Z`` and ``X`` operators.

Every gate routine takes an index array of batch elements and, per selected
element, the qubit(s) to act on. The routines use only gather and scatter
operations along the batch axis and elementwise XOR and AND, so each one
maps to a single array kernel; there is no Python loop over the batch.
"""

from __future__ import annotations

import numpy as np

GATES_1Q = ("H", "S", "S_DAG", "X", "Y", "Z")
GATES_2Q = ("CNOT", "CZ", "SWAP")
GATES = GATES_1Q + GATES_2Q


def identity_tableau(n: int, batch_size: int) -> np.ndarray:
    """Tableau array of the identity circuit, replicated ``batch_size`` times."""
    eye = np.eye(2 * n, dtype=np.uint8)
    tab = np.zeros((batch_size, 2 * n, 2 * n + 1), dtype=np.uint8)
    tab[:, :, : 2 * n] = eye
    return tab


def num_qubits(tab: np.ndarray) -> int:
    return (tab.shape[-1] - 1) // 2


def _select(tab: np.ndarray, where):
    """Batch index array for ``where`` (``None`` means all, bool mask, or ints)."""
    if where is None:
        return np.arange(tab.shape[0])
    where = np.asarray(where)
    if where.dtype == bool:
        return np.flatnonzero(where)
    return where


def _per_element(q, m: int) -> np.ndarray:
    q = np.asarray(q, dtype=np.intp)
    if q.ndim == 0:
        return np.broadcast_to(q, (m,))
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
    xa = tab[b, :, a].copy()
    za = tab[b, :, n + a].copy()
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


def apply_gate(tab: np.ndarray, gate: str, q1, q2=None, where=None) -> None:
    """Apply ``gate`` in place to the batch elements selected by ``where``.

    ``q1`` and ``q2`` are ints (same qubit on every selected element) or
    arrays with one entry per selected element. ``where`` is ``None`` (all
    elements), a boolean mask over the batch, or an integer index array.
    """
    b = _select(tab, where)
    if b.size == 0:
        return
    m = b.size
    q1 = _per_element(q1, m)
    if gate in _ONE_QUBIT:
        _ONE_QUBIT[gate](tab, b, q1)
    elif gate in _TWO_QUBIT:
        if q2 is None:
            raise ValueError(f"{gate} needs two qubits")
        q2 = _per_element(q2, m)
        if np.any(q1 == q2):
            raise ValueError("two-qubit gate on identical qubits")
        _TWO_QUBIT[gate](tab, b, q1, q2)
    else:
        raise ValueError(f"unknown gate {gate!r}")


class ActionTable:
    """Enumeration of the discrete actions of an environment.

    ``entries`` is a list of ``(gate, q1, q2)`` with ``q2 = -1`` for
    single-qubit gates. The table stores three parallel arrays so that a
    batch of action indices can be resolved with one gather each.
    """

    def __init__(self, entries):
        self.entries = [(g, int(a), int(b)) for g, a, b in entries]
        self.gate_names = sorted({g for g, _, _ in self.entries}, key=GATES.index)
        self.gate_id = np.array(
            [self.gate_names.index(g) for g, _, _ in self.entries], dtype=np.int64
        )
        self.q1 = np.array([a for _, a, _ in self.entries], dtype=np.intp)
        self.q2 = np.array([b for _, _, b in self.entries], dtype=np.intp)

    def __len__(self):
        return len(self.entries)

    def apply(self, tab: np.ndarray, actions, where=None) -> None:
        """Apply per-element actions. ``actions`` has one index per batch
        element; a negative index is a no-op. The loop is over gate types
        (at most nine), never over the batch.
        """
        actions = np.asarray(actions, dtype=np.int64)
        active = actions >= 0
        if where is not None:
            active &= np.asarray(where, dtype=bool)
        if not active.any():
            return
        gid = np.full(actions.shape, -1, dtype=np.int64)
        gid[active] = self.gate_id[actions[active]]
        for g_idx, gate in enumerate(self.gate_names):
            sel = np.flatnonzero(gid == g_idx)
            if sel.size == 0:
                continue
            acts = actions[sel]
            q1 = self.q1[acts]
            q2 = self.q2[acts] if gate in _TWO_QUBIT else None
            apply_gate(tab, gate, q1, q2, where=sel)


# --------------------------------------------------------------------------
# stim interoperability (cross-checks only; the hot path never imports stim)
# --------------------------------------------------------------------------
def row_to_pauli_string(row: np.ndarray):
    """One tableau row (length ``2n + 1``) as a ``stim.PauliString``."""
    import stim

    n = (row.shape[0] - 1) // 2
    return stim.PauliString.from_numpy(
        xs=row[:n].astype(bool),
        zs=row[n : 2 * n].astype(bool),
        sign=-1 if row[2 * n] else +1,
    )


def pauli_string_to_row(p) -> np.ndarray:
    xs, zs = p.to_numpy()
    n = xs.shape[0]
    row = np.zeros(2 * n + 1, dtype=np.uint8)
    row[:n] = xs
    row[n : 2 * n] = zs
    if p.sign == -1:
        row[2 * n] = 1
    elif p.sign != 1:
        raise ValueError("tableau rows must be Hermitian (sign +1 or -1)")
    return row


def to_stim_tableau(tab: np.ndarray, index: int = 0):
    """Batch element ``index`` as a ``stim.Tableau`` (conjugation map)."""
    import stim

    n = num_qubits(tab)
    t = tab[index]
    xs = [row_to_pauli_string(t[r]) for r in range(n)]
    zs = [row_to_pauli_string(t[n + r]) for r in range(n)]
    return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)


def from_stim_tableau(t) -> np.ndarray:
    """A ``stim.Tableau`` as a tableau array of batch size 1."""
    n = len(t)
    tab = np.zeros((1, 2 * n, 2 * n + 1), dtype=np.uint8)
    for r in range(n):
        tab[0, r] = pauli_string_to_row(t.x_output(r))
        tab[0, n + r] = pauli_string_to_row(t.z_output(r))
    return tab


_STIM_NAMES = {
    "H": "H",
    "S": "S",
    "S_DAG": "S_DAG",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "CNOT": "CX",
    "CZ": "CZ",
    "SWAP": "SWAP",
}


def gates_to_stim_circuit(gates, n: int):
    """A list of ``(gate, q1, q2)`` (``q2 = -1`` for one-qubit gates) as a
    ``stim.Circuit`` on ``n`` qubits, one instruction per gate."""
    import stim

    c = stim.Circuit()
    for g, a, b in gates:
        if g in _TWO_QUBIT:
            c.append(_STIM_NAMES[g], [int(a), int(b)])
        else:
            c.append(_STIM_NAMES[g], [int(a)])
    if n > 0:
        c.append("I", [n - 1])
    return c


class BatchTableau:
    """Thin object wrapper around the tableau array.

    Holds ``array`` of shape ``(batch, 2n, 2n + 1)``. Methods mutate in
    place and return ``self``.
    """

    def __init__(self, n: int, batch_size: int = 1, array: np.ndarray | None = None):
        self.n = int(n)
        if array is None:
            array = identity_tableau(self.n, batch_size)
        if array.shape[1:] != (2 * self.n, 2 * self.n + 1):
            raise ValueError("array shape does not match n")
        self.array = array

    @property
    def batch_size(self) -> int:
        return self.array.shape[0]

    def copy(self) -> BatchTableau:
        return BatchTableau(self.n, array=self.array.copy())

    def reset(self, where=None) -> BatchTableau:
        b = _select(self.array, where)
        self.array[b] = identity_tableau(self.n, 1)[0]
        return self

    def apply(self, gate: str, q1, q2=None, where=None) -> BatchTableau:
        apply_gate(self.array, gate, q1, q2, where)
        return self

    def h(self, q, where=None):
        return self.apply("H", q, where=where)

    def s(self, q, where=None):
        return self.apply("S", q, where=where)

    def cnot(self, c, t, where=None):
        return self.apply("CNOT", c, t, where=where)

    def cz(self, a, b, where=None):
        return self.apply("CZ", a, b, where=where)

    def swap(self, a, b, where=None):
        return self.apply("SWAP", a, b, where=where)

    def pauli(self, gate: str, q, where=None):
        if gate not in ("X", "Y", "Z"):
            raise ValueError("pauli must be X, Y, or Z")
        return self.apply(gate, q, where=where)

    def stabilizers(self, k: int = 0) -> np.ndarray:
        """View ``(batch, n - k, 2n)`` of the code stabilizers (no sign bit)."""
        n = self.n
        return self.array[:, n + k : 2 * n, : 2 * n]

    def logical_x(self, k: int) -> np.ndarray:
        return self.array[:, :k, : 2 * self.n]

    def logical_z(self, k: int) -> np.ndarray:
        n = self.n
        return self.array[:, n : n + k, : 2 * n]

    def logicals(self, k: int) -> np.ndarray:
        """``(batch, 2k, 2n)``: logical X rows followed by logical Z rows."""
        return np.concatenate([self.logical_x(k), self.logical_z(k)], axis=1)

    def to_stim(self, index: int = 0):
        return to_stim_tableau(self.array, index)

    @classmethod
    def from_stim(cls, t) -> BatchTableau:
        arr = from_stim_tableau(t)
        return cls(len(t), array=arr)
