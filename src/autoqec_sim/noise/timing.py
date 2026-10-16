"""Timing models: how a TICK layer of the skeleton maps onto time slots.

`parallel`: every TICK layer is one slot (the board's idealization).
`sequential`: every two-qubit gate is a slot of its own (the chain model
of arXiv:2503.22071 and arXiv:2606.06455 Appendix C). `gates_per_slot=k`:
the two-qubit gates of a layer run k at a time, the partial parallelism
the ion work found decisive for the idle budget. In every case the
non-two-qubit instructions of a layer (resets, single-qubit gates,
measurements, annotations) stay together in a first slot, and single-qubit
layers are never serialized, so `reslot(c, 1)` equals
ion_noise.serialize_two_qubit(c) instruction for instruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import stim

from .skeleton import ANNOTATIONS, segments


@dataclass(frozen=True)
class Timing:
    """gates_per_slot None means fully parallel; 1 means sequential."""
    gates_per_slot: int | None = None

    def __post_init__(self):
        k = self.gates_per_slot
        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError("gates_per_slot must be a positive int or None")

    @property
    def name(self) -> str:
        if self.gates_per_slot is None:
            return "parallel"
        if self.gates_per_slot == 1:
            return "sequential"
        return f"gates_per_slot={self.gates_per_slot}"

    @classmethod
    def from_name(cls, name) -> Timing:
        if isinstance(name, Timing):
            return name
        if name == "parallel":
            return PARALLEL
        if name == "sequential":
            return SEQUENTIAL
        if isinstance(name, int):
            return cls(name)
        raise ValueError(f"unknown timing {name!r}")


PARALLEL = Timing(None)
SEQUENTIAL = Timing(1)


def partial(k: int) -> Timing:
    return Timing(k)


def reslot(circuit: stim.Circuit, gates_per_slot: int | None) -> stim.Circuit:
    """Rewrite the skeleton so each TICK layer holds at most `gates_per_slot`
    two-qubit gates. None returns the input unchanged."""
    if gates_per_slot is None:
        return circuit
    k = gates_per_slot
    out = stim.Circuit()
    first = True
    for seg in segments(circuit.flattened()):
        two, rest = [], []
        for inst in seg:
            if inst.name not in ANNOTATIONS and \
                    stim.gate_data(inst.name).is_two_qubit_gate:
                two.append(inst)
            else:
                rest.append(inst)
        if not first:
            out.append("TICK")
        first = False
        for inst in rest:
            out.append(inst)
        pairs = []
        for inst in two:
            t = [x.qubit_value for x in inst.targets_copy()]
            pairs.extend((inst.name, a, b) for a, b in zip(t[0::2], t[1::2]))
        for j in range(0, len(pairs), k):
            if j or rest:
                out.append("TICK")
            for name, a, b in pairs[j:j + k]:
                out.append(name, [a, b])
    return out
