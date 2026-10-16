"""Pauli channels parametrized by a total rate and a bias.

The bias of a single-qubit Pauli channel is eta = pZ / (pX + pY). The
depolarizing channel has eta = 1/2. For two-qubit channels the package
uses a product-weight construction: a weight w(I) = w(X) = w(Y) = 1,
w(Z) = 2 eta per factor and probability proportional to w(P1) w(P2) for
every non-identity pair. Its single-qubit marginals have bias exactly eta
and eta = 1/2 gives the uniform two-qubit depolarizing channel. As eta
grows the channel concentrates on ZZ (weight 4 eta^2), then equally on
the six terms with one Z factor (ZI, IZ, ZX, ZY, XZ, YZ, weight 2 eta),
then the eight Z-free terms (weight 1). This is a modelling choice, not a
measured decomposition: it does not distinguish ZI from ZX, and it is not
the product of two independent single-qubit biased channels (which would
have no correlation). Callers with a measured 15-vector pass it through
`Params.pauli2`.
"""

from __future__ import annotations

import math

import stim

DEPOLARIZING_BIAS = 0.5
PAULI1 = ("X", "Y", "Z")
# stim's PAULI_CHANNEL_2 argument order (first letter acts on the first target).
PAULI2 = ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ",
          "ZI", "ZX", "ZY", "ZZ")


def biased_1q(p: float, eta: float) -> tuple[float, float, float]:
    """(px, py, pz) with px + py + pz = p and pz / (px + py) = eta."""
    if p < 0:
        raise ValueError("p must be non-negative")
    if math.isinf(eta):
        return (0.0, 0.0, p)
    if eta < 0:
        raise ValueError("bias must be non-negative")
    pxy = p / (2.0 * (1.0 + eta))
    return (pxy, pxy, p * eta / (1.0 + eta))


def biased_2q(p: float, eta: float) -> tuple[float, ...]:
    """The 15 PAULI_CHANNEL_2 probabilities of the product-weight biased
    channel with total rate p and marginal bias eta."""
    if p < 0:
        raise ValueError("p must be non-negative")
    if math.isinf(eta):
        # eta -> inf: ZZ, ZI, IZ survive in ratio 4 eta^2 : 2 eta : 2 eta, so
        # the limit is pure ZZ.
        return tuple(p if s == "ZZ" else 0.0 for s in PAULI2)
    if eta < 0:
        raise ValueError("bias must be non-negative")
    w = {"I": 1.0, "X": 1.0, "Y": 1.0, "Z": 2.0 * eta}
    tot = (3.0 + 2.0 * eta) ** 2 - 1.0
    return tuple(p * w[s[0]] * w[s[1]] / tot for s in PAULI2)


def marginal_bias_2q(probs) -> float:
    """eta of the first-qubit marginal of a 15-vector (I on the first qubit
    excluded), for checking a supplied decomposition."""
    pz = sum(q for s, q in zip(PAULI2, probs) if s[0] == "Z")
    pxy = sum(q for s, q in zip(PAULI2, probs) if s[0] in "XY")
    return math.inf if pxy == 0 else pz / pxy


def is_depolarizing(eta: float) -> bool:
    return eta == DEPOLARIZING_BIAS


def pauli_targets(pauli: str, qubits) -> list[stim.GateTarget]:
    """Targets for CORRELATED_ERROR: one Pauli letter per qubit."""
    if len(pauli) != len(qubits):
        raise ValueError(f"pauli string {pauli!r} does not match {len(qubits)} qubits")
    out = []
    for s, q in zip(pauli, qubits):
        if s == "X":
            out.append(stim.target_x(q))
        elif s == "Y":
            out.append(stim.target_y(q))
        elif s == "Z":
            out.append(stim.target_z(q))
        elif s == "I":
            continue
        else:
            raise ValueError(f"bad Pauli letter {s!r}")
    if not out:
        raise ValueError("a correlated term must act on at least one qubit")
    return out


def exclusive_chain(probs) -> list[float]:
    """Conditional probabilities that make a CORRELATED_ERROR followed by
    ELSE_CORRELATED_ERROR terms fire with the given marginal probabilities,
    mutually exclusively: q_i = p_i / (1 - sum_{j < i} p_j)."""
    out, used = [], 0.0
    for p in probs:
        rem = 1.0 - used
        if rem <= 0 or p > rem:
            raise ValueError("exclusive correlated terms sum to more than 1")
        out.append(p / rem)
        used += p
    return out
