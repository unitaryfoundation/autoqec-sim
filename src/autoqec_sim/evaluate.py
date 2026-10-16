# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""One-call evaluation of a stabilizer code under a device noise profile.

:func:`evaluate_code` builds the ``R``-round memory experiment of
:mod:`autoqec_sim.memory_objective` in both bases, applies the profile, and
returns a :class:`CodeEvaluation`: the logical error per round of the
chosen tier summed over the bases, the per-basis dictionaries, the exact
circuit-level distance and the mass of its minimum-weight undetectable
fault sets per basis (computed from the detector error model whatever the
tier), and the most probable minimum-weight undetectable fault sets
explained back to circuit locations with stim's
``explain_detector_error_model_errors``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import stim

from autoqec_sim import noise as nm
from autoqec_sim.noise import Profile

from .code import StabilizerCode
from .memory_objective import (
    BASES,
    TIERS,
    MemoryObjective,
    dem_signature,
    exact_dcirc,
    hook_orders,
    noisy_memory_circuit,
    undetectable_sets,
)

__all__ = ["CodeEvaluation", "DominantFault", "FaultLocation", "evaluate_code", "explain_mechanisms"]


@dataclass(frozen=True)
class FaultLocation:
    """One DEM mechanism as a circuit error: the noise instruction ``gate``
    (stim name, such as ``DEPOLARIZE2`` or ``M``), its ``qubits``, the
    ``pauli`` product it flips (``"Y10"``, ``"Z10*X7"``, or ``"flip Z9"``
    for a measurement flip), the number of ``TICK`` layers before it, and
    the DEM ``targets`` it produces (``"D1 D4 L0"``). ``gate`` is ``"?"``
    when stim cannot attribute the mechanism to one circuit error."""

    gate: str
    qubits: tuple[int, ...]
    pauli: str
    tick: int
    targets: str


@dataclass(frozen=True)
class DominantFault:
    """A minimum-weight undetectable logical fault set of one basis: the
    DEM mechanism indices (into ``dem.flattened()`` error instructions),
    the product of their probabilities, and one circuit location per
    mechanism."""

    basis: str
    mechanisms: tuple[int, ...]
    probability: float
    locations: tuple[FaultLocation, ...]


@dataclass(frozen=True)
class CodeEvaluation:
    """Result of :func:`evaluate_code`.

    ``p_round`` is the per-round logical error of ``tier`` summed over the
    bases; ``per_basis`` maps each basis to the dictionary of
    :meth:`MemoryObjective.score_basis` with ``d_circ``, ``d_circ_count``,
    and ``d_circ_mass`` added; ``d_circ`` and ``d_circ_mass`` repeat the
    exact circuit-level distance (``None`` above 4) and the summed
    probability product of its minimum-weight undetectable fault sets per
    basis; ``dominant_faults`` holds the most probable such sets of weight
    at most 3 over both bases, most probable first (empty for a basis whose
    ``d_circ`` is 4 or more).
    """

    p_round: float
    per_basis: dict
    d_circ: dict
    d_circ_mass: dict
    tier: str
    rounds: int
    dominant_faults: tuple[DominantFault, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _target_key(targets) -> frozenset:
    return frozenset(str(t) for t in targets if not t.is_separator())


def _pauli_string(gate_targets) -> str:
    return "*".join(f"{t.gate_target.pauli_type}{t.gate_target.qubit_value}" for t in gate_targets)


def _location(loc: stim.CircuitErrorLocation, targets: str) -> FaultLocation:
    inside = loc.instruction_targets
    qubits = tuple(int(t.gate_target.qubit_value) for t in inside.targets_in_range if t.gate_target.is_qubit_target)
    if loc.flipped_measurement is not None:
        pauli = "flip " + _pauli_string(loc.flipped_measurement.observable)
    else:
        pauli = _pauli_string(loc.flipped_pauli_product)
    return FaultLocation(inside.gate, qubits, pauli, int(loc.tick_offset), targets)


def explain_mechanisms(circuit: stim.Circuit, dem: stim.DetectorErrorModel, indices) -> dict[int, FaultLocation]:
    """One circuit error per DEM mechanism: ``indices`` into the error
    instructions of ``dem.flattened()`` (the mechanism order of
    :func:`autoqec_sim.memory_objective.dem_signature`), explained on the
    noisy ``circuit`` the DEM came from with
    ``explain_detector_error_model_errors``. A mechanism stim cannot
    attribute to a single circuit error gets ``gate="?"``."""
    errors = [inst for inst in dem.flattened() if inst.type == "error"]
    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return {}
    dem_filter = stim.DetectorErrorModel()
    for i in wanted:
        dem_filter.append(errors[i])
    by_key: dict[frozenset, stim.ExplainedError] = {}
    try:
        for e in circuit.explain_detector_error_model_errors(dem_filter=dem_filter, reduce_to_one_representative_error=True):
            by_key[frozenset(str(t.dem_target) for t in e.dem_error_terms)] = e
    except ValueError:
        pass
    out = {}
    for i in wanted:
        targets = " ".join(sorted(_target_key(errors[i].targets_copy()), key=lambda s: (s[0], int(s[1:]))))
        e = by_key.get(_target_key(errors[i].targets_copy()))
        if e is None or not e.circuit_error_locations:
            out[i] = FaultLocation("?", (), "", -1, targets)
        else:
            out[i] = _location(e.circuit_error_locations[0], targets)
    return out


def evaluate_code(
    code,
    profile: Profile,
    *,
    rounds: int | None = None,
    scale: float = 1.0,
    tier: str = "proxy",
    shots: int = 20_000,
    min_failures: int = 0,
    max_shots: int | None = None,
    seed: int = 0,
    connectivity="all_to_all",
    max_faults: int = 10,
) -> CodeEvaluation:
    """Evaluate ``code`` under ``profile`` at ``scale`` by an ``R``-round
    memory experiment in both bases.

    Parameters
    ----------
    code
        A :class:`StabilizerCode`, or its stabilizer generators as Pauli
        strings for :meth:`StabilizerCode.from_stabilizers`.
    profile
        ``autoqec_sim.noise.Profile``; every rate is multiplied by ``scale``.
    rounds
        Noisy syndrome-extraction rounds; default the code's exact distance.
    tier
        ``"proxy"`` (closed form, no decoder), ``"sampled"`` (exact
        maximum likelihood or the lookup decoder), ``"bposd"`` (``ldpc``),
        or ``"tesseract"`` (``tesseract-decoder``); see
        :class:`MemoryObjective`.
    shots, min_failures, max_shots, seed
        Sampling budget of the sampling tiers (shots grow in doublings up
        to ``max_shots`` until ``min_failures`` failures) and the seed of
        the sampler and of the gate-order search.
    connectivity
        ``"all_to_all"``, ``"square:RxC"``, or a
        :class:`autoqec_sim.memory_objective.Layout`.
    max_faults
        Minimum-weight undetectable fault sets listed per basis.
    """
    if not isinstance(code, StabilizerCode):
        code = StabilizerCode.from_stabilizers(code)
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    rounds = code.distance_exact() if rounds is None else int(rounds)
    objective = MemoryObjective(
        profile, scale, rounds, tier, connectivity, seed=seed, shots=shots, min_failures=min_failures, max_shots=max_shots,
    )
    orders = hook_orders(code, rounds, seed)
    per_basis, d_circ, d_circ_mass, faults = {}, {}, {}, []
    for basis in BASES:
        mc = objective.circuit(code, basis, rounds, seed, orders)
        noisy = noisy_memory_circuit(mc, profile, scale)
        dem = nm.dem(noisy)
        out = objective.score_basis(mc, dem, tier)
        sig = dem_signature(dem)
        dc = exact_dcirc(sig, wmax=4)
        out["d_circ"], out["d_circ_count"], out["d_circ_mass"] = dc["d_circ"], dc["count"], dc["mass"]
        d_circ[basis], d_circ_mass[basis] = dc["d_circ"], dc["mass"]
        sets = undetectable_sets(sig, wmax=3, limit=max_faults)
        if sets["sets"]:
            where = explain_mechanisms(noisy, dem, {i for s, _ in sets["sets"] for i in s})
            faults.extend(DominantFault(basis, s, p, tuple(where[i] for i in s)) for s, p in sets["sets"])
        per_basis[basis] = out
    faults.sort(key=lambda f: -f.probability)
    p_round = float(sum(per_basis[b]["p_round"] for b in BASES))
    return CodeEvaluation(p_round, per_basis, d_circ, d_circ_mass, tier, rounds, tuple(faults))
