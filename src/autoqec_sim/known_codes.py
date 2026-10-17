# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Reference stabilizer codes used by the tests and the rediscovery regression."""

from __future__ import annotations

from .code import StabilizerCode, SubsystemCode

FIVE_QUBIT = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]

STEANE = [
    "IIIXXXX",
    "IXXIIXX",
    "XIXIXIX",
    "IIIZZZZ",
    "IZZIIZZ",
    "ZIZIZIZ",
]

SHOR = [
    "ZZIIIIIII",
    "IZZIIIIII",
    "IIIZZIIII",
    "IIIIZZIII",
    "IIIIIIZZI",
    "IIIIIIIZZ",
    "XXXXXXIII",
    "IIIXXXXXX",
]

# Rotated surface code, distance 3, qubits on a 3x3 grid indexed row-major.
#   0 1 2
#   3 4 5
#   6 7 8
# Bulk plaquettes alternate X and Z; two-body boundary checks complete them.
ROTATED_SURFACE_9 = [
    "XXIXXIIII",  # X plaquette on {0,1,3,4}
    "IIIIXXIXX",  # X plaquette on {4,5,7,8}
    "IXXIIIIII",  # X boundary {1,2}
    "IIIIIIXXI",  # X boundary {6,7}
    "IZZIZZIII",  # Z plaquette on {1,2,4,5}
    "IIIZZIZZI",  # Z plaquette on {3,4,6,7}
    "ZIIZIIIII",  # Z boundary {0,3}
    "IIIIIZIIZ",  # Z boundary {5,8}
]

FOUR_TWO_TWO = ["XXXX", "ZZZZ"]


def rotated_surface(d: int) -> list[str]:
    """Rotated surface code ``[[d^2, 1, d]]`` on a ``d x d`` grid.

    Parameters
    ----------
    d
        Distance; qubit indices are row-major.

    Returns
    -------
    StabilizerCode

    Notes
    -----
    Plaquette ``(r, c)`` covers ``{(r, c), (r, c+1), (r+1, c), (r+1, c+1)}``
    and is X-type iff ``r + c`` is even; the weight-2 boundary checks have
    the type opposite to the adjacent plaquette. ``d = 3`` reproduces
    ``ROTATED_SURFACE_9`` up to the order of the generators.
    """
    n = d * d
    idx = lambda r, c: r * d + c  # noqa: E731

    def pauli(kind, qubits):
        s = ["I"] * n
        for q in qubits:
            s[q] = kind
        return "".join(s)

    out = []
    for r in range(d - 1):
        for c in range(d - 1):
            kind = "X" if (r + c) % 2 == 0 else "Z"
            out.append(pauli(kind, [idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)]))
    for c in range(d - 1):
        if c % 2 == 1:  # plaquette (0, c) is Z: X boundary on top
            out.append(pauli("X", [idx(0, c), idx(0, c + 1)]))
        if (d - 2 + c) % 2 == 1:  # plaquette (d-2, c) is Z: X boundary at the bottom
            out.append(pauli("X", [idx(d - 1, c), idx(d - 1, c + 1)]))
    for r in range(d - 1):
        if r % 2 == 0:  # plaquette (r, 0) is X: Z boundary on the left
            out.append(pauli("Z", [idx(r, 0), idx(r + 1, 0)]))
        if (r + d - 2) % 2 == 0:  # plaquette (r, d-2) is X: Z boundary on the right
            out.append(pauli("Z", [idx(r, d - 1), idx(r + 1, d - 1)]))
    return out


# Rotated surface code, distance 4, on the 4x4 grid (the target of the
# 4x4 PPO rediscovery target and of the RL scaling probe); w* = 4, qubit
# degree 4.
ROTATED_SURFACE_16 = rotated_surface(4)

KNOWN = {
    "five_qubit": (FIVE_QUBIT, (5, 1, 3)),
    "steane": (STEANE, (7, 1, 3)),
    "shor": (SHOR, (9, 1, 3)),
    "rotated_surface_9": (ROTATED_SURFACE_9, (9, 1, 3)),
    "rotated_surface_16": (ROTATED_SURFACE_16, (16, 1, 4)),
    "four_two_two": (FOUR_TWO_TWO, (4, 2, 2)),
}


def get(name: str) -> StabilizerCode:
    strings, _ = KNOWN[name]
    return StabilizerCode.from_stabilizers(strings)


def _bacon_shor_gauge(rows: int = 3, cols: int = 3) -> list[str]:
    """Gauge generators of the Bacon-Shor code on a ``rows x cols`` grid,
    qubits row-major: ``XX`` on vertically adjacent qubits and ``ZZ`` on
    horizontally adjacent ones."""
    n = rows * cols
    out = []
    for c in range(cols):
        for r in range(rows - 1):
            s = ["I"] * n
            s[r * cols + c] = s[(r + 1) * cols + c] = "X"
            out.append("".join(s))
    for r in range(rows):
        for c in range(cols - 1):
            s = ["I"] * n
            s[r * cols + c] = s[r * cols + c + 1] = "Z"
            out.append("".join(s))
    return out


# Bacon-Shor [[9,1,3]]: twelve weight-2 gauge generators; the stabilizers are
# the weight-6 products on pairs of adjacent rows (X) and columns (Z).
BACON_SHOR_9_GAUGE = _bacon_shor_gauge(3, 3)

# The distance-3 heavy-hexagon subsystem code of Sundaresan et al. (its
# equations 1 to 4, qubits 1 to 9 as indices 0 to 8): weight-2 and weight-4
# Z gauge generators and weight-2 X gauge generators; the center holds two
# weight-6 Z stabilizers and four X stabilizers of weight 2 and 4.
HEAVY_HEX_9_GAUGE = [
    "ZZIIIIIII", "IZZIZZIII", "IIIZZIZZI", "IIIIIIIZZ",
    "XIIXIIIII", "IXIIXIIII", "IIXIIXIII", "IIIXIIXII", "IIIIXIIXI", "IIIIIXIIX",
]

SUBSYSTEM = {
    "bacon_shor_9": (BACON_SHOR_9_GAUGE, (9, 1, 3)),
    "heavy_hex_9": (HEAVY_HEX_9_GAUGE, (9, 1, 3)),
}


def get_subsystem(name: str) -> SubsystemCode:
    """A :class:`SubsystemCode` from its gauge generators."""
    gauge, _ = SUBSYSTEM[name]
    return SubsystemCode.from_gauge(gauge)


def parameters(name: str) -> tuple[int, int, int]:
    return KNOWN[name][1]
