"""apply(circuit, profile, scale): the noisy circuit of a skeleton.

Placement per TICK segment of the (re-slotted) skeleton, in this order:

  - each instruction unchanged, followed immediately by its channel:
      two-qubit gate:  DEPOLARIZE2(p2) or PAULI_CHANNEL_2(15 biased args),
                       then the correlated gate-pair terms, then crosstalk
                       terms on graph neighbours, then per-gate leakage;
      single-qubit unitary: DEPOLARIZE1(p1) / PAULI_CHANNEL_1 / Z_ERROR;
      R / RX:          X_ERROR(p_reset) / Z_ERROR(p_reset);
      M / MX:          flip probability p_meas as the gate argument;
  - if the segment did anything: one idle channel on every in-scope qubit
    not operated in the segment; then, if the segment measures, the
    per-measurement-layer leakage term on the same qubits.

With the ion presets this reproduces ion_noise.apply_ion_noise byte for
byte (tests/test_ion_equivalence.py) and with the board preset
circuit_tools.apply_noise (tests/test_board_equivalence.py).
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import stim

from . import channels as ch
from .profile import Profile, scaled
from .skeleton import ANNOTATIONS, MEASURES, RESET_FLIP, circuit_qubits, layer_classes, segments
from .timing import reslot
from .topology import Violation, check_connectivity


class ConnectivityError(ValueError):
    def __init__(self, violations: list[Violation]):
        self.violations = violations
        shown = "\n  ".join(str(v) for v in violations[:10])
        more = "" if len(violations) <= 10 else f"\n  ... {len(violations) - 10} more"
        super().__init__(f"{len(violations)} two-qubit gate(s) violate the "
                         f"connectivity graph:\n  {shown}{more}")


def _layer_duration(seg, r) -> float:
    dur = {"meas": r["t_meas"], "reset": r["t_reset"], "2q": r["t_2q"], "1q": r["t_1q"]}
    return max((dur[c] for c in layer_classes(seg)), default=0.0)


def _idle_channel(profile: Profile, r, seg):
    """The idle channel of a segment: ("DEPOLARIZE1", rate) or
    ("PAULI_CHANNEL_1", (px, py, pz)) or None."""
    idle = profile.params.idle
    if idle.kind == "none":
        return None
    if idle.kind == "duration":
        dur = _layer_duration(seg, r)
        if dur <= 0:
            return None
        px = py = min(0.25, r["xy_rate"] * dur)
        pz = min(0.5, r["z_rate"] * dur)
        return ("PAULI_CHANNEL_1", (px, py, pz))
    cls = layer_classes(seg)
    if not cls:
        return None
    c = r["idle_meas"] if "meas" in cls else r["idle_gate"]
    if isinstance(c, tuple):
        return ("PAULI_CHANNEL_1", c)
    eta = idle.bias
    if eta is None or ch.is_depolarizing(eta):
        return ("DEPOLARIZE1", c)
    if eta == float("inf"):
        return ("Z_ERROR", c)
    return ("PAULI_CHANNEL_1", ch.biased_1q(c, eta))


class _Recorder:
    """Tracks measurement-record indices so herald records inserted by
    HERALDED_ERASE do not break the skeleton's rec[-k] references."""

    def __init__(self):
        self.n_in = 0
        self.n_out = 0
        self.map: list[int] = []

    def measured(self, k: int):
        self.map.extend(range(self.n_out, self.n_out + k))
        self.n_in += k
        self.n_out += k

    def heralded(self, k: int):
        self.n_out += k

    def retarget(self, targets):
        out = []
        for t in targets:
            if t.is_measurement_record_target:
                out.append(stim.target_rec(self.map[self.n_in + t.value] - self.n_out))
            else:
                out.append(t)
        return out


def apply(circuit: stim.Circuit, profile: Profile, scale: float = 1.0, *,
          n_data: int | None = None, idle_qubits=None,
          check_graph: bool = True) -> stim.Circuit:
    """The noisy circuit for a noiseless skeleton in the board convention.

    Parameters
    ----------
    circuit
        Noiseless skeleton with ``TICK`` layers.
    profile
        Noise profile.
    scale
        Multiplies every rate of the profile.
    n_data
        Number of data qubits (``0..n_data-1``); required when the profile's
        idle scope is ``"data"``, the board rule.
    idle_qubits
        Overrides the idle set outright.
    check_graph
        With a connectivity graph on the profile, check every two-qubit gate
        first and raise :class:`ConnectivityError` listing the violations.
        Crosstalk still needs the graph when this is false.

    Returns
    -------
    stim.Circuit
    """
    p = profile.params
    if profile.graph is not None and check_graph:
        bad = check_connectivity(circuit, profile.graph)
        if bad:
            raise ConnectivityError(bad)
    if p.correlated.crosstalk > 0 and profile.graph is None:
        raise ValueError("crosstalk needs a connectivity graph on the profile")
    circuit = reslot(circuit, profile.timing.gates_per_slot)
    r = scaled(p, scale)

    if idle_qubits is not None:
        scope = list(idle_qubits)
    elif p.idle.scope == "data":
        if n_data is None:
            raise ValueError(f"profile {profile.name!r} idles data qubits only; pass n_data")
        scope = list(range(n_data))
    else:
        scope = circuit_qubits(circuit)

    leak_meas = 0.75 * r["leak_meas"] if p.leakage.mode == "pauli" else r["leak_meas"]
    leak_2q = 0.75 * r["leak_2q"] if p.leakage.mode == "pauli" else r["leak_2q"]
    leak_gate = "DEPOLARIZE1" if p.leakage.mode == "pauli" else "HERALDED_ERASE"
    corr_probs = [q for _, q in r["corr"]]
    if p.correlated.exclusive:
        corr_probs = ch.exclusive_chain(corr_probs)
    two_q_channel = None
    if r["pauli2"] is not None:
        two_q_channel = ("PAULI_CHANNEL_2", list(r["pauli2"]))
    elif r["p2"] is not None:
        if ch.is_depolarizing(p.bias2):
            two_q_channel = ("DEPOLARIZE2", r["p2"])
        else:
            two_q_channel = ("PAULI_CHANNEL_2", list(ch.biased_2q(r["p2"], p.bias2)))
    one_q_channel = None
    if r["p1"] is not None:
        if ch.is_depolarizing(p.bias1):
            one_q_channel = ("DEPOLARIZE1", r["p1"])
        elif p.bias1 == float("inf"):
            one_q_channel = ("Z_ERROR", r["p1"])
        else:
            one_q_channel = ("PAULI_CHANNEL_1", list(ch.biased_1q(r["p1"], p.bias1)))

    rec = _Recorder()
    heralds = p.leakage.mode == "heralded" and p.leakage.active

    def leak(out, qubits, rate):
        if rate <= 0:
            return
        out.append(leak_gate, qubits, rate)
        if heralds:
            rec.heralded(len(qubits))
            if p.leakage.herald_detectors:
                for i in range(len(qubits)):
                    out.append("DETECTOR", [stim.target_rec(-len(qubits) + i)])

    out = stim.Circuit()
    for si, seg in enumerate(segments(circuit.flattened())):
        if si:
            out.append("TICK")
        touched, active = set(), False
        for inst in seg:
            name = inst.name
            if name in ANNOTATIONS:
                if heralds and name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                    out.append(name, rec.retarget(inst.targets_copy()), inst.gate_args_copy())
                else:
                    out.append(inst)
                continue
            targets = inst.targets_copy()
            qubits = [t.qubit_value for t in targets if t.qubit_value is not None]
            touched.update(qubits)
            active = True
            if name in MEASURES:
                if r["p_meas"] is None:
                    out.append(inst)
                else:
                    out.append(name, targets, r["p_meas"])
                rec.measured(len(qubits))
            elif name in RESET_FLIP:
                out.append(inst)
                if r["p_reset"] is not None:
                    out.append(RESET_FLIP[name], qubits, r["p_reset"])
            else:
                gd = stim.gate_data(name)
                if gd.is_noisy_gate:
                    raise ValueError(f"input is not noiseless: {name}")
                out.append(inst)
                if gd.is_two_qubit_gate:
                    if two_q_channel is not None:
                        out.append(two_q_channel[0], targets, two_q_channel[1])
                    pairs = list(zip(qubits[0::2], qubits[1::2]))
                    for a, b in pairs:
                        for i, ((pauli, _), q) in enumerate(zip(r["corr"], corr_probs)):
                            kind = "ELSE_CORRELATED_ERROR" if (p.correlated.exclusive and i) \
                                else "CORRELATED_ERROR"
                            out.append(kind, ch.pauli_targets(pauli, (a, b)), q)
                    if r["crosstalk"] > 0:
                        g = profile.graph
                        pg, ps = p.correlated.crosstalk_pauli
                        for a, b in pairs:
                            for gq in (a, b):
                                for c in g.neighbors(gq):
                                    if c in (a, b):
                                        continue
                                    out.append("CORRELATED_ERROR",
                                               ch.pauli_targets(pg + ps, (gq, c)),
                                               r["crosstalk"])
                    if leak_2q > 0:
                        leak(out, qubits, leak_2q)
                elif gd.is_unitary:
                    if one_q_channel is not None:
                        out.append(one_q_channel[0], qubits, one_q_channel[1])
                else:
                    raise ValueError(f"unsupported instruction {name}")
        if active:
            idle = [q for q in scope if q not in touched]
            if idle:
                c = _idle_channel(profile, r, seg)
                if c is not None:
                    kind, arg = c
                    total = sum(arg) if isinstance(arg, tuple) else arg
                    if total > 0:
                        out.append(kind, idle, list(arg) if isinstance(arg, tuple) else arg)
                if leak_meas > 0 and "meas" in layer_classes(seg):
                    leak(out, idle, leak_meas)
    return out


def noisifier(profile: Profile, scale: float = 1.0, **kw) -> Callable[[stim.Circuit], stim.Circuit]:
    """A callable ``circuit -> noisy circuit`` with the profile and scale bound,
    the form a reward engine plugs in.

    Parameters
    ----------
    profile, scale, **kw
        Passed to :func:`apply`.
    """
    return functools.partial(apply, profile=profile, scale=scale, **kw)


def dem(circuit: stim.Circuit, approximate: bool | None = None,
        decompose_errors: bool = False) -> stim.DetectorErrorModel:
    """Detector error model with the board's pinned options (no decomposition,
    loops flattened).

    Parameters
    ----------
    circuit
        Noisy circuit.
    approximate
        Use ``approximate_disjoint_errors``. Channels stim cannot convert
        exactly (``PAULI_CHANNEL_2``, ``ELSE_CORRELATED_ERROR``,
        ``HERALDED_ERASE``, a Pauli channel with a zero component) need it;
        with ``None`` the exact conversion is tried first and the approximate
        one used only if stim refuses.
    decompose_errors
        Passed to stim.

    Returns
    -------
    stim.DetectorErrorModel
    """
    kw = dict(decompose_errors=decompose_errors, flatten_loops=True)
    if approximate:
        return circuit.detector_error_model(approximate_disjoint_errors=True, **kw)
    try:
        return circuit.detector_error_model(**kw)
    except ValueError as e:
        if approximate is False or "approximate_disjoint_errors" not in str(e):
            raise
        return circuit.detector_error_model(approximate_disjoint_errors=True, **kw)


def mechanism_count(d: stim.DetectorErrorModel) -> int:
    return sum(1 for inst in d.flattened() if inst.type == "error")


def fault_expectation(d: stim.DetectorErrorModel) -> float:
    return sum(inst.args_copy()[0] for inst in d.flattened() if inst.type == "error")
