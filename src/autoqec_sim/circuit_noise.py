# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Circuit-level noise for the reward, wired to ``genesis/noise-model``.

A *noise model* is any callable ``stim.Circuit -> stim.Circuit`` that
inserts Pauli noise into a noiseless encoding circuit. Three forms are
accepted wherever a noise model is expected (see :func:`resolve_noise_model`):

- a ``noise_model.Profile`` (``presets.board()``, ``presets.chain()``,
  ``presets.forte()``, ``presets.heron()``, ``presets.depolarizing(p)``,
  or a variant built with ``with_bias``, ``with_timing``, ``with_params``),
  applied through ``noise_model.noisifier(profile, scale)``;
- a preset name (``"board-rfc0001"``, ``"chain-2606.06455"``,
  ``"forte-2308.05071"``, ``"heron-r2"``);
- a plain callable, such as the placeholder
  :func:`depolarizing_noise_model`, which is kept for tests and as the
  profile-free fallback.

The encoding circuit handed to the noise model is the gate list laid out
in the TICK-layer convention of the noise module (:func:`gates_to_layered_circuit`,
as-soon-as-possible scheduling: a gate joins the earliest layer after the
last gate on any of its qubits). The profile's timing model then decides
how a layer maps onto time slots (parallel, sequential, or ``k`` two-qubit
gates per slot), and its idle channel is charged to every qubit not
operated in a slot. The profile's connectivity graph, when present, is the
environment's graph (``Graph.to_topology``); a two-qubit gate on a
non-edge makes the noise module raise ``ConnectivityError``.

The reward enumerates every fault mechanism of the noisy circuit (each
Pauli term of each noise instruction with its probability, the same
mechanism set that stim's detector-error-model derivation walks, before
merging by detector signature), propagates the mechanism through the
remaining gates (a GF(2) product against the symplectic map of the
circuit suffix; :func:`fault_set_reference` does the same with stim's
``PauliString.after`` and is kept as the cross-check), and scores the
resulting Pauli on the encoded state with the batched Knill-Laflamme
kernel. Heralded mechanisms
(``HERALDED_ERASE``) count as detected. The reward is
``-sum_f p_f K_f`` over mechanisms, divided by ``sum_f p_f`` when
normalized. A detector-error-model derivation of the same undetected mass
is provided for cross-checks (:func:`dem_undetected_mass`); it is not used
in the reward because an encoding circuit has no detectors of its own and
the logical ``X`` and ``Z`` readouts needed to see both kinds of logical
fault anticommute, so they need two circuits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import stim

from noise_model import ConnectivityError, Profile, noisifier, presets
from noise_model.skeleton import ANNOTATIONS

from .graphs import as_graph
from .reward import undetected_mask
from .tableau import _STIM_NAMES, _TWO_QUBIT, pauli_string_to_row, row_to_pauli_string

NoiseModel = Callable[[stim.Circuit], stim.Circuit]

_PAULI1 = ("X", "Y", "Z")
_PAULI2 = [a + b for a in "IXYZ" for b in "IXYZ"][1:]  # 15 non-identity pairs, stim order

__all__ = [
    "ConnectivityError",
    "FaultSet",
    "NoiseModel",
    "circuit_depth",
    "circuit_level_reward",
    "dem_undetected_mass",
    "depolarizing_noise_model",
    "direct_undetected_mass",
    "fault_set",
    "fault_set_reference",
    "gates_to_layered_circuit",
    "make_circuit_level_reward",
    "readout_circuit",
    "resolve_noise_model",
    "undetected_fault_mass",
]


# --------------------------------------------------------------------------
# skeleton layout
# --------------------------------------------------------------------------
def gates_to_layered_circuit(gates, n: int) -> stim.Circuit:
    """``(gate, q1, q2)`` list as a TICK-layered noiseless skeleton on ``n``
    qubits. As-soon-as-possible scheduling: each gate goes into the first
    layer after the last layer that touched one of its qubits. A
    ``QUBIT_COORDS`` line per qubit fixes ``num_qubits = n`` so that idle
    channels reach qubits the circuit never touches. The circuit is built
    from its text form in one parse (an order of magnitude faster than
    appending instruction by instruction)."""
    layers: list[list[str]] = []
    ready = [0] * n
    for g, a, b in gates:
        qs = (int(a), int(b)) if g in _TWO_QUBIT else (int(a),)
        layer = max(ready[q] for q in qs)
        while len(layers) <= layer:
            layers.append([])
        layers[layer].append(_STIM_NAMES[g] + " " + " ".join(map(str, qs)))
        for q in qs:
            ready[q] = layer + 1
    lines = [f"QUBIT_COORDS({q}) {q}" for q in range(n)]
    for i, layer in enumerate(layers):
        if i:
            lines.append("TICK")
        lines.extend(layer)
    return stim.Circuit("\n".join(lines))


def circuit_depth(gates, n: int) -> int:
    """Number of TICK layers of the as-soon-as-possible layout."""
    ready = [0] * n
    depth = 0
    for g, a, b in gates:
        qs = (int(a), int(b)) if g in _TWO_QUBIT else (int(a),)
        layer = max(ready[q] for q in qs)
        for q in qs:
            ready[q] = layer + 1
        depth = max(depth, layer + 1)
    return depth


# --------------------------------------------------------------------------
# placeholder model
# --------------------------------------------------------------------------
def depolarizing_noise_model(p1: float | None = 1e-3, p2: float = 1e-2, idle: float = 0.0) -> NoiseModel:
    """Placeholder: ``DEPOLARIZE1(p1)`` after every single-qubit gate
    (``None`` for no single-qubit channel), ``DEPOLARIZE2(p2)`` after every
    two-qubit gate, and, if ``idle > 0``, ``DEPOLARIZE1(idle)`` on every
    qubit of ``circuit.num_qubits`` not operated in a TICK layer. With
    ``p1=None, p2=p, idle=p`` on a layered skeleton it equals the board
    preset ``presets.board(p)`` under parallel timing."""

    def apply(circuit):
        out = stim.Circuit()
        n = circuit.num_qubits
        insts = list(circuit.flattened())
        segs: list[list] = [[]]
        for inst in insts:
            if inst.name == "TICK":
                segs.append([])
            else:
                segs[-1].append(inst)
        for si, seg in enumerate(segs):
            if si:
                out.append("TICK")
            touched = set()
            for inst in seg:
                out.append(inst)
                if inst.name in ANNOTATIONS:
                    continue
                targets = [t.value for t in inst.targets_copy()]
                touched.update(targets)
                if inst.name in ("CX", "CZ", "SWAP", "CNOT"):
                    out.append("DEPOLARIZE2", targets, p2)
                elif p1 is not None and inst.name in (
                    "H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG"
                ):
                    out.append("DEPOLARIZE1", targets, p1)
            if idle > 0 and touched:
                idle_q = [q for q in range(n) if q not in touched]
                if idle_q:
                    out.append("DEPOLARIZE1", idle_q, idle)
        return out

    return apply


# --------------------------------------------------------------------------
# resolving profiles
# --------------------------------------------------------------------------
def resolve_noise_model(spec, scale: float = 1.0, graph=None, n: int | None = None, **apply_kw) -> NoiseModel:
    """Turn ``spec`` into a ``stim.Circuit -> stim.Circuit`` callable.

    ``spec`` is a ``noise_model.Profile``, a preset name, or a callable
    (returned unchanged; ``scale`` and ``graph`` then do not apply). For a
    profile, ``graph`` (either graph class) becomes the profile's
    connectivity graph when the profile has none, and must have the same
    edge set as the profile's graph otherwise, so that the environment's
    action space and the placement check agree. ``n`` is passed as
    ``n_data`` for profiles whose idle channel is restricted to data qubits
    (the board recipe); the whole register of an encoding circuit is data.
    """
    if callable(spec) and not isinstance(spec, Profile):
        return spec
    if isinstance(spec, str):
        spec = presets.preset(spec)
    if not isinstance(spec, Profile):
        raise TypeError(f"cannot resolve a noise model from {type(spec).__name__}")
    profile = spec
    if graph is not None:
        g = as_graph(graph)
        if profile.graph is None:
            profile = profile.with_graph(g.to_topology())
        elif not g.same_edges(profile.graph):
            raise ValueError(
                f"profile graph {profile.graph.name!r} and environment graph {g.name!r} differ"
            )
    kw = dict(apply_kw)
    if n is not None and profile.params.idle.scope == "data":
        kw.setdefault("n_data", n)
    return noisifier(profile, scale, **kw)


# --------------------------------------------------------------------------
# fault mechanisms
# --------------------------------------------------------------------------
def _pauli_targets(inst) -> tuple[tuple[int, ...], str]:
    qs, labs = [], []
    for t in inst.targets_copy():
        qs.append(t.qubit_value)
        labs.append("X" if t.is_x_target else "Y" if t.is_y_target else "Z")
    return tuple(qs), "".join(labs)


def _channel_terms(inst, else_scale: float = 1.0):
    """Yield ``(qubits, pauli_label, prob, heralded)`` for every fault
    mechanism of one noise instruction. ``else_scale`` is the probability
    that no earlier term of an ``ELSE_CORRELATED_ERROR`` chain fired."""
    name = inst.name
    args = inst.gate_args_copy()
    if name in ("CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"):
        qs, lab = _pauli_targets(inst)
        yield qs, lab, args[0] * else_scale, False
        return
    targets = [t.value for t in inst.targets_copy()]
    if name == "DEPOLARIZE1":
        p = args[0] / 3
        for q in targets:
            for lab in _PAULI1:
                yield (q,), lab, p, False
    elif name == "DEPOLARIZE2":
        p = args[0] / 15
        for i in range(0, len(targets), 2):
            for lab in _PAULI2:
                yield (targets[i], targets[i + 1]), lab, p, False
    elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        for q in targets:
            yield (q,), name[0], args[0], False
    elif name == "PAULI_CHANNEL_1":
        for q in targets:
            for lab, p in zip(_PAULI1, args):
                if p > 0:
                    yield (q,), lab, p, False
    elif name == "PAULI_CHANNEL_2":
        for i in range(0, len(targets), 2):
            for lab, p in zip(_PAULI2, args):
                if p > 0:
                    yield (targets[i], targets[i + 1]), lab, p, False
    elif name == "HERALDED_ERASE":
        # erasure: the qubit is replaced by the maximally mixed state with
        # probability p, i.e. I, X, Y, Z each with p/4; the flag is known
        p = args[0] / 4
        for q in targets:
            for lab in _PAULI1:
                yield (q,), lab, p, True
    elif name == "HERALDED_PAULI_CHANNEL_1":
        for q in targets:
            for lab, p in zip(_PAULI1, args[1:]):
                if p > 0:
                    yield (q,), lab, p, True
    else:
        raise ValueError(f"unsupported noise instruction {name}")


def _is_noise(inst) -> bool:
    return inst.name not in ANNOTATIONS and stim.gate_data(inst.name).is_noisy_gate


@dataclass
class FaultSet:
    """Propagated single-fault mechanisms of a noisy encoding circuit.

    ``paulis`` is ``(M, 2n)`` (the fault as a Pauli on the output state),
    ``probs`` is ``(M,)``, ``heralded`` is ``(M,)`` bool (mechanisms whose
    occurrence is flagged in the circuit and count as detected), and
    ``source`` is ``(M,)`` the index of the noise instruction in the
    flattened noisy circuit. Unpacks as ``paulis, probs``.
    """

    paulis: np.ndarray
    probs: np.ndarray
    heralded: np.ndarray
    source: np.ndarray

    def __iter__(self):
        yield self.paulis
        yield self.probs

    def __len__(self):
        return self.paulis.shape[0]

    @property
    def total(self) -> float:
        return float(self.probs.sum())


def _gate_rows(name: str, qs, n: int):
    """Row updates of the suffix map for one Clifford gate, as a list of
    ``(row, source rows)``: the image of generator ``row`` under the gate
    followed by the current suffix is the XOR of the current images of
    ``source rows`` (signs ignored). Generators are indexed ``q`` for
    ``X_q`` and ``n + q`` for ``Z_q``."""
    if name == "H":
        (q,) = qs
        return [(q, (n + q,)), (n + q, (q,))]
    if name in ("S", "S_DAG"):
        (q,) = qs
        return [(q, (q, n + q))]
    if name in ("X", "Y", "Z", "I"):
        return []
    if name in ("CX", "CNOT"):
        c, t = qs
        return [(c, (c, t)), (n + t, (n + t, n + c))]
    if name == "CZ":
        a, b = qs
        return [(a, (a, n + b)), (b, (b, n + a))]
    if name == "SWAP":
        a, b = qs
        return [(a, (b,)), (b, (a,)), (n + a, (n + b,)), (n + b, (n + a,))]
    raise ValueError(f"unsupported gate {name} in the encoding circuit")


_LABEL_COMPONENTS = {"I": (), "X": (0,), "Y": (0, 1), "Z": (1,)}


def _mechanism_matrix(inst, n: int, else_scale: float):
    """``(coef (m, 2n) uint8, probs (m,), heralded (m,))`` of the fault
    mechanisms of one noise instruction, ``coef`` being the symplectic
    vectors of the mechanisms *before* propagation (the same terms, in the
    same order, as :func:`_channel_terms`)."""
    rows, probs, her = [], [], []
    for qubits, lab, p, h in _channel_terms(inst, else_scale):
        v = np.zeros(2 * n, dtype=np.uint8)
        for q, ch in zip(qubits, lab):
            for comp in _LABEL_COMPONENTS[ch]:
                v[q + comp * n] ^= 1
        rows.append(v)
        probs.append(p)
        her.append(h)
    if not rows:
        return np.zeros((0, 2 * n), dtype=np.uint8), np.zeros(0), np.zeros(0, dtype=bool)
    return np.array(rows, dtype=np.uint8), np.array(probs, dtype=float), np.array(her, dtype=bool)


def fault_set(noisy_circuit: stim.Circuit, n: int) -> FaultSet:
    """Propagate every single fault of ``noisy_circuit`` to the end.

    Handles ``DEPOLARIZE1/2``, ``X/Y/Z_ERROR``, ``PAULI_CHANNEL_1/2``,
    ``CORRELATED_ERROR`` and ``ELSE_CORRELATED_ERROR`` chains (absolute
    probabilities), and ``HERALDED_ERASE`` (heralded). Annotations and
    ``TICK`` are ignored.

    The propagation is symplectic: walking the circuit backwards keeps the
    ``2n x 2n`` GF(2) map from a Pauli inserted at the current position to
    the output Pauli (signs dropped, they do not affect detection), each
    Clifford gate being a few row operations on that map, and the
    mechanisms of a noise instruction are one small GF(2) product against
    it. :func:`fault_set_reference` is the instruction-by-instruction stim
    version it replaces; the two agree mechanism for mechanism.
    """
    insts = list(noisy_circuit.flattened())
    # forward pass: absolute probabilities of ELSE_CORRELATED_ERROR chains
    else_scales = {}
    else_scale = 1.0
    for i, inst in enumerate(insts):
        if not _is_noise(inst):
            continue
        if inst.name != "ELSE_CORRELATED_ERROR":
            else_scale = 1.0
        else_scales[i] = else_scale
        if inst.name in ("CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"):
            else_scale *= 1.0 - inst.gate_args_copy()[0]
    # backward pass: suffix maps
    suffix = np.eye(2 * n, dtype=np.uint8)
    chunks_p, chunks_pr, chunks_h, chunks_s = [], [], [], []
    for i in range(len(insts) - 1, -1, -1):
        inst = insts[i]
        name = inst.name
        if name in ANNOTATIONS:
            continue
        if _is_noise(inst):
            coef, probs, her = _mechanism_matrix(inst, n, else_scales[i])
            if coef.shape[0]:
                out = (coef.astype(np.int64) @ suffix.astype(np.int64)) & 1
                chunks_p.append(out.astype(np.uint8))
                chunks_pr.append(probs)
                chunks_h.append(her)
                chunks_s.append(np.full(probs.shape[0], i, dtype=np.int64))
            continue
        gd = stim.gate_data(name)
        if not gd.is_unitary:
            raise ValueError(f"unsupported instruction {name} in an encoding circuit")
        targets = [t.value for t in inst.targets_copy()]
        width = 2 if gd.is_two_qubit_gate else 1
        for j in range(0, len(targets), width):
            qs = tuple(targets[j : j + width])
            ups = _gate_rows(name, qs, n)
            if not ups:
                continue
            new = {row: np.bitwise_xor.reduce(suffix[list(src)], axis=0) for row, src in ups}
            for row, v in new.items():
                suffix[row] = v
    if not chunks_p:
        return FaultSet(
            np.zeros((0, 2 * n), dtype=np.uint8), np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
        )
    # the backward walk visited noise instructions last-to-first
    return FaultSet(
        np.concatenate(chunks_p[::-1]),
        np.concatenate(chunks_pr[::-1]),
        np.concatenate(chunks_h[::-1]),
        np.concatenate(chunks_s[::-1]),
    )


def fault_set_reference(noisy_circuit: stim.Circuit, n: int) -> FaultSet:
    """Reference implementation of :func:`fault_set`: each mechanism is a
    ``stim.PauliString`` conjugated through the suffix circuit with
    ``PauliString.after``. Kept for cross-checks; about twenty times
    slower."""
    insts = list(noisy_circuit.flattened())
    rows, probs, heralded, source = [], [], [], []
    else_scale = 1.0
    for i, inst in enumerate(insts):
        if not _is_noise(inst):
            continue
        if inst.name in ("CORRELATED_ERROR", "E") or inst.name != "ELSE_CORRELATED_ERROR":
            else_scale = 1.0
        suffix = stim.Circuit()
        for later in insts[i + 1 :]:
            if later.name not in ANNOTATIONS and not _is_noise(later):
                suffix.append(later)
        terms = list(_channel_terms(inst, else_scale))
        for qubits, lab, p, her in terms:
            ps = stim.PauliString(n)
            for q, ch in zip(qubits, lab):
                if ch != "I":
                    ps[q] = ch
            out = ps.after(suffix)
            rows.append(pauli_string_to_row(out)[: 2 * n])
            probs.append(p)
            heralded.append(her)
            source.append(i)
        if inst.name in ("CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"):
            else_scale *= 1.0 - inst.gate_args_copy()[0]
    if not rows:
        return FaultSet(
            np.zeros((0, 2 * n), dtype=np.uint8), np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
        )
    return FaultSet(
        np.array(rows, dtype=np.uint8),
        np.array(probs, dtype=float),
        np.array(heralded, dtype=bool),
        np.array(source, dtype=np.int64),
    )


def undetected_fault_mass(tab_b: np.ndarray, k: int, faults: FaultSet) -> float:
    """``sum_f p_f K_f`` for one tableau ``(1, 2n, 2n+1)``; heralded
    mechanisms count as detected."""
    if len(faults) == 0:
        return 0.0
    und = undetected_mask(tab_b, k, faults.paulis)[0] & ~faults.heralded
    return float(und @ faults.probs)


# --------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------
def circuit_level_reward(
    tab: np.ndarray,
    k: int,
    gate_lists: list[list[tuple]],
    noise_model,
    normalize: bool = True,
    scale: float = 1.0,
    graph=None,
    on_violation: str = "raise",
    violation_reward: float = -1.0,
) -> np.ndarray:
    """Per-element ``-sum_f p_f K_f`` (divided by ``sum_f p_f`` when
    ``normalize``) for the encoding circuits ``gate_lists`` whose final
    tableaux are the rows of ``tab``.

    ``noise_model`` is anything :func:`resolve_noise_model` accepts;
    ``scale`` and ``graph`` apply to profiles. A two-qubit gate on a
    non-edge of the profile's graph raises ``ConnectivityError``
    (``on_violation="raise"``) or scores ``violation_reward``
    (``on_violation="penalize"``).
    """
    n = (tab.shape[-1] - 1) // 2
    model = resolve_noise_model(noise_model, scale=scale, graph=graph, n=n)
    out = np.zeros(tab.shape[0])
    for b, gates in enumerate(gate_lists):
        try:
            noisy = model(gates_to_layered_circuit(gates, n))
        except ConnectivityError:
            if on_violation == "raise":
                raise
            out[b] = violation_reward
            continue
        faults = fault_set(noisy, n)
        if len(faults) == 0:
            continue
        r = -undetected_fault_mass(tab[b : b + 1], k, faults)
        if normalize and faults.total > 0:
            r /= faults.total
        out[b] = r
    return out


def make_circuit_level_reward(env, noise_model, normalize: bool = True, scale: float = 1.0, **kw):
    """Closure ``f(tab, k)`` bound to an environment so it can read the
    action histories that produced ``tab``; the profile's graph is the
    environment's graph. Use as a terminal-only reward
    (``reward_mode="sparse"``); it is not batched over circuits."""
    n = env.n
    model = resolve_noise_model(noise_model, scale=scale, graph=env.graph, n=n)

    def f(tab, k):
        gates = env.gate_lists()
        return circuit_level_reward(tab, k, gates, model, normalize, **kw)

    return f


# --------------------------------------------------------------------------
# detector-error-model cross-check
# --------------------------------------------------------------------------
def readout_circuit(noisy_circuit: stim.Circuit, tab_b: np.ndarray, k: int, basis: str = "Z") -> stim.Circuit:
    """``noisy_circuit`` followed by an ideal readout of the code: ``MPP``
    of every stabilizer generator, each declared a ``DETECTOR``, and of
    every logical ``Z`` (``basis="Z"``, input logical ``|0>``) or logical
    ``X`` (``basis="X"``, the logical qubits are put in ``|+>`` by a
    noiseless ``H`` before the circuit), each declared an observable."""
    n = (tab_b.shape[-1] - 1) // 2
    t = tab_b[0]
    c = stim.Circuit()
    if basis == "X":
        c.append("H", list(range(k)))
    c += noisy_circuit
    stabs = [row_to_pauli_string(t[n + k + j]) for j in range(n - k)]
    if basis == "Z":
        logs = [row_to_pauli_string(t[n + j]) for j in range(k)]
    elif basis == "X":
        logs = [row_to_pauli_string(t[j]) for j in range(k)]
    else:
        raise ValueError("basis must be X or Z")
    for s in stabs:
        c.append("MPP", _mpp_targets(s))
        c.append("DETECTOR", [stim.target_rec(-1)])
    for j, lg in enumerate(logs):
        c.append("MPP", _mpp_targets(lg))
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], j)
    return c


def _mpp_targets(ps: stim.PauliString):
    targets = []
    first = True
    for q in range(len(ps)):
        v = ps[q]
        if v == 0:
            continue
        if not first:
            targets.append(stim.target_combiner())
        first = False
        targets.append({1: stim.target_x, 2: stim.target_y, 3: stim.target_z}[v](q))
    if ps.sign == -1:
        targets[0] = stim.target_inv(targets[0])
    return targets


def dem_undetected_mass(noisy_circuit: stim.Circuit, tab_b: np.ndarray, k: int, basis: str = "Z") -> float:
    """Probability mass, from stim's detector error model of
    :func:`readout_circuit`, of error mechanisms that flip no stabilizer
    detector and flip some logical observable in ``basis``. Mechanisms
    with the same signature are merged by stim (XOR of probabilities), so
    this agrees with the direct enumeration to ``O(p^2)``."""
    c = readout_circuit(noisy_circuit, tab_b, k, basis)
    try:
        d = c.detector_error_model(flatten_loops=True)
    except ValueError:
        d = c.detector_error_model(flatten_loops=True, approximate_disjoint_errors=True)
    mass = 0.0
    for inst in d.flattened():
        if inst.type != "error":
            continue
        dets = [t for t in inst.targets_copy() if t.is_relative_detector_id()]
        obs = [t for t in inst.targets_copy() if t.is_logical_observable_id()]
        if not dets and obs:
            mass += inst.args_copy()[0]
    return mass


def direct_undetected_mass(faults: FaultSet, tab_b: np.ndarray, k: int, basis: str = "Z") -> float:
    """Direct-enumeration counterpart of :func:`dem_undetected_mass`: mass of
    unheralded mechanisms whose propagated Pauli commutes with every
    stabilizer and anticommutes with some logical of ``basis``."""
    from .symplectic import symplectic_product

    n = (tab_b.shape[-1] - 1) // 2
    t = tab_b[0]
    stabs = t[n + k :, : 2 * n]
    logs = t[n : n + k, : 2 * n] if basis == "Z" else t[:k, : 2 * n]
    if len(faults) == 0:
        return 0.0
    comm = ~symplectic_product(faults.paulis, stabs).any(axis=1)
    hit = symplectic_product(faults.paulis, logs).any(axis=1)
    sel = comm & hit & ~faults.heralded
    return float(faults.probs[sel].sum())


def is_two_qubit(gate: str) -> bool:
    return gate in _TWO_QUBIT
