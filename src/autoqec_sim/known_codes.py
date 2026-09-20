# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Reference stabilizer codes used by the tests and the rediscovery regression."""

from __future__ import annotations

from .code import StabilizerCode

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

KNOWN = {
    "five_qubit": (FIVE_QUBIT, (5, 1, 3)),
    "steane": (STEANE, (7, 1, 3)),
    "shor": (SHOR, (9, 1, 3)),
    "rotated_surface_9": (ROTATED_SURFACE_9, (9, 1, 3)),
    "four_two_two": (FOUR_TWO_TWO, (4, 2, 2)),
}


def get(name: str) -> StabilizerCode:
    strings, _ = KNOWN[name]
    return StabilizerCode.from_stabilizers(strings)


def parameters(name: str) -> tuple[int, int, int]:
    return KNOWN[name][1]
