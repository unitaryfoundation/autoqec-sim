"""Skeleton conventions shared with qldpc-challenge/verify/circuit_tools.py.

A skeleton is a noiseless stim circuit whose TICK-delimited segments are
parallel layers, with the gate set R, RX, M, MX, stim's unitary gates, and
the annotations below. The helpers here are re-implementations of the
board's, kept dependency-free so the package installs without the
qldpc-challenge checkout; tests check them against the originals.
"""

from __future__ import annotations

import stim

RESET_FLIP = {"R": "X_ERROR", "RX": "Z_ERROR"}
MEASURES = ("M", "MX")
ANNOTATIONS = ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS",
               "SHIFT_COORDS")


def is_pure_noise(name: str) -> bool:
    gd = stim.gate_data(name)
    return gd.is_noisy_gate and not gd.produces_measurements


def segments(flat_circuit) -> list[list[stim.CircuitInstruction]]:
    """TICK-delimited segments of a flattened circuit, TICKs removed."""
    segs, cur = [], []
    for inst in flat_circuit:
        if inst.name == "TICK":
            segs.append(cur)
            cur = []
        else:
            cur.append(inst)
    segs.append(cur)
    return segs


def strip_noise(circuit: stim.Circuit) -> stim.Circuit:
    """The noiseless skeleton of a noisy circuit produced by `apply`: noise
    channels removed, measurement flip arguments dropped, herald records of
    HERALDED_ERASE removed with the detector and observable record indices
    rewritten to the shorter record. Equal to the board's strip_noise on
    circuits without heralds."""
    out = stim.Circuit()
    n_in = 0                      # measurement records consumed in the input
    out_index: list[int | None] = []   # input record -> output record or None
    n_out = 0
    for inst in circuit.flattened():
        name = inst.name
        gd = stim.gate_data(name)
        if gd.produces_measurements and gd.is_noisy_gate and name not in MEASURES:
            # heralded channel: its records vanish from the skeleton
            k = len([t for t in inst.targets_copy() if t.is_qubit_target])
            out_index.extend([None] * k)
            n_in += k
            continue
        if is_pure_noise(name):
            continue
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            targets = []
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    abs_in = n_in + t.value        # value is negative
                    abs_out = out_index[abs_in]
                    if abs_out is None:
                        continue                  # a herald detector
                    targets.append(stim.target_rec(abs_out - n_out))
                else:
                    targets.append(t)
            if name == "DETECTOR" and not targets:
                continue
            out.append(name, targets, inst.gate_args_copy())
            continue
        if name in MEASURES:
            targets = inst.targets_copy()
            k = len(targets)
            out_index.extend(range(n_out, n_out + k))
            n_in += k
            n_out += k
            out.append(name, targets, [])
            continue
        if gd.produces_measurements:
            k = len([t for t in inst.targets_copy() if t.is_qubit_target])
            out_index.extend(range(n_out, n_out + k))
            n_in += k
            n_out += k
        out.append(inst)
    return out


def circuit_qubits(circuit: stim.Circuit) -> list[int]:
    """Every qubit index the circuit operates on (data and ancilla)."""
    qs = set()
    for inst in circuit.flattened():
        if inst.name in ANNOTATIONS:
            continue
        for t in inst.targets_copy():
            if t.qubit_value is not None:
                qs.add(t.qubit_value)
    return sorted(qs)


def layer_classes(seg) -> set[str]:
    """Instruction classes present in a segment: 'meas', 'reset', '2q',
    '1q'. Annotations and noise channels do not count."""
    cls = set()
    for inst in seg:
        name = inst.name
        if name in ANNOTATIONS:
            continue
        gd = stim.gate_data(name)
        if gd.is_noisy_gate and name not in MEASURES:
            continue
        if name in MEASURES:
            cls.add("meas")
        elif name in RESET_FLIP:
            cls.add("reset")
        elif gd.is_two_qubit_gate:
            cls.add("2q")
        else:
            cls.add("1q")
    return cls


def two_qubit_pairs(seg) -> list[tuple[str, int, int]]:
    """(gate name, a, b) for every two-qubit gate atom in a segment."""
    out = []
    for inst in seg:
        if inst.name in ANNOTATIONS:
            continue
        gd = stim.gate_data(inst.name)
        if gd.is_noisy_gate or not gd.is_two_qubit_gate:
            continue
        t = [x.qubit_value for x in inst.targets_copy()]
        out.extend((inst.name, a, b) for a, b in zip(t[0::2], t[1::2]))
    return out
