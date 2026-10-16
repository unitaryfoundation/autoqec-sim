"""Profile = parameter set + timing + connectivity graph.

Every probability in a Params is a rate at scale 1; `apply` multiplies all
of them by `scale`. Durations and coherence times do not scale. A field set
to None means "no channel of that class" (the board recipe has no
single-qubit-gate channel); 0.0 means a channel that is emitted with
probability 0 (so `scale` sweeps stay structurally identical, as in
ion_noise).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .channels import DEPOLARIZING_BIAS, PAULI2
from .timing import PARALLEL, Timing
from .topology import Graph


@dataclass(frozen=True)
class Idle:
    """Idle channel of a layer, applied to every qubit in `scope` not
    operated in an active layer.

    kind "duration": PAULI_CHANNEL_1(t / (4 T1), t / (4 T1), t / (2 T2)) with
      t the layer's wall-clock duration (slowest instruction class present).
      Pure dephasing from T2 plus a Pauli-twirled amplitude term, linear in
      t (t << T1, T2). The T2 here is the measured T2 (Hahn or Ramsey as the
      source states), used as a total dephasing time: the pz formula does
      not subtract the T1 contribution, which is the same convention as
      ion_noise.
    kind "slot": a fixed channel per layer class, `gate` for layers without
      a measurement and `meas` for layers with one. Each is a scalar
      (emitted as DEPOLARIZE1(rate), the board's form) or a 3-tuple
      (PAULI_CHANNEL_1(px, py, pz)).
    kind "none": no idle channel.
    scope "all": every qubit the circuit uses; "data": qubits 0..n_data-1
      (the board rule; `apply` then needs n_data).
    bias: for scalar slot rates only, eta of the emitted channel (None or
      1/2: DEPOLARIZE1(rate); otherwise PAULI_CHANNEL_1 with pz/(px+py) =
      eta; inf: Z_ERROR(rate)). Duration idles carry their own T1/T2
      structure and tuple slot idles are taken as given.
    """
    kind: str = "none"
    T1: float | None = None
    T2: float | None = None
    gate: float | tuple[float, float, float] | None = None
    meas: float | tuple[float, float, float] | None = None
    scope: str = "all"
    bias: float | None = None

    def __post_init__(self):
        if self.kind not in ("none", "duration", "slot"):
            raise ValueError(f"unknown idle kind {self.kind!r}")
        if self.bias is not None and not (self.bias >= 0 or math.isinf(self.bias)):
            raise ValueError("idle bias must be non-negative")
        if self.scope not in ("all", "data"):
            raise ValueError(f"unknown idle scope {self.scope!r}")
        if self.kind == "duration" and (not self.T1 or not self.T2):
            raise ValueError("duration idle needs T1 and T2")
        if self.kind == "slot" and (self.gate is None or self.meas is None):
            raise ValueError("slot idle needs gate and meas channels")


@dataclass(frozen=True)
class Leakage:
    """Leakage, in a stim-expressible form.

    Rates: `per_meas_layer` fires on every qubit not measured in a layer
    that measures (the mid-circuit-measurement leakage of arXiv:2606.06455
    Table V); `per_2q` fires on both qubits of every two-qubit gate, after
    the gate's Pauli channel.

    mode "pauli": the erasure-to-Pauli proxy, DEPOLARIZE1(3/4 rate), i.e.
      the qubit is replaced by the maximally mixed state with probability
      `rate` and nothing is heralded. This is the simulation proxy of
      arXiv:2606.06455 Appendix E.
    mode "heralded": HERALDED_ERASE(rate), stim's erasure channel with a
      flag bit in the measurement record; with `herald_detectors` a
      DETECTOR on each flag so the DEM carries the herald. This models
      detected leakage (leakage detection units or post-selection); the
      board's conformance check does not accept heralded channels, and the
      DEM conversion needs approximate_disjoint_errors.

    Both are approximations: a leaked qubit is neither erased nor
    depolarized, it acts as a lost qubit whose gates with neighbours apply
    non-Pauli errors. Stim has no leaked state.
    """
    per_meas_layer: float = 0.0
    per_2q: float = 0.0
    mode: str = "pauli"
    herald_detectors: bool = True

    def __post_init__(self):
        if self.mode not in ("pauli", "heralded"):
            raise ValueError(f"unknown leakage mode {self.mode!r}")

    @property
    def active(self) -> bool:
        return self.per_meas_layer > 0 or self.per_2q > 0


@dataclass(frozen=True)
class Correlated:
    """Correlated Pauli terms beyond the per-gate channel.

    `gate_pairs`: ((pauli2, rate), ...) applied after every two-qubit gate
      on its (first, second) target as CORRELATED_ERROR; with `exclusive`
      the terms after the first become ELSE_CORRELATED_ERROR with
      conditional probabilities so exactly one of them can fire (the DEM
      conversion of the ELSE form needs approximate_disjoint_errors).
    `crosstalk`: for every two-qubit gate (a, b) and every graph neighbour
      c of a or b outside the gate, CORRELATED_ERROR(crosstalk) with
      `crosstalk_pauli[0]` on the gate qubit and `crosstalk_pauli[1]` on c.
      Stochastic stand-in for always-on ZZ coupling or spectator drive;
      needs `Profile.graph`.
    """
    gate_pairs: tuple[tuple[str, float], ...] = ()
    exclusive: bool = False
    crosstalk: float = 0.0
    crosstalk_pauli: str = "ZZ"

    def __post_init__(self):
        for s, r in self.gate_pairs:
            if s not in PAULI2:
                raise ValueError(f"bad two-qubit Pauli {s!r}")
            if r < 0:
                raise ValueError("correlated rates must be non-negative")
        if len(self.crosstalk_pauli) != 2 or any(c not in "XYZ" for c in self.crosstalk_pauli):
            raise ValueError("crosstalk_pauli must be two letters from XYZ")

    @property
    def active(self) -> bool:
        return bool(self.gate_pairs) or self.crosstalk > 0


@dataclass(frozen=True)
class Params:
    """Rates at scale 1 (probabilities) and durations (seconds).

    p2, p1: total Pauli error per two-qubit gate and per single-qubit
      unitary, with biases `bias2`, `bias1` (eta = pZ / (pX + pY);
      1/2 is depolarizing and is emitted as DEPOLARIZE1/2, any other value
      as PAULI_CHANNEL_1/2). `pauli2`, if given, is an explicit 15-vector in
      stim's PAULI_CHANNEL_2 order that overrides p2 and bias2.
    p_reset: X_ERROR after R, Z_ERROR after RX. p_meas: flip on M and MX.
    t_2q, t_1q, t_meas, t_reset: layer durations by instruction class.
    """
    p2: float | None
    p1: float | None
    p_reset: float | None
    p_meas: float | None
    bias1: float = DEPOLARIZING_BIAS
    bias2: float = DEPOLARIZING_BIAS
    pauli2: tuple[float, ...] | None = None
    idle: Idle = field(default_factory=Idle)
    leakage: Leakage = field(default_factory=Leakage)
    correlated: Correlated = field(default_factory=Correlated)
    t_2q: float = 0.0
    t_1q: float = 0.0
    t_meas: float = 0.0
    t_reset: float = 0.0
    notes: str = ""

    def __post_init__(self):
        for k in ("p2", "p1", "p_reset", "p_meas"):
            v = getattr(self, k)
            if v is not None and v < 0:
                raise ValueError(f"{k} must be non-negative")
        for k in ("bias1", "bias2"):
            v = getattr(self, k)
            if not (v >= 0 or math.isinf(v)):
                raise ValueError(f"{k} must be non-negative")
        if self.pauli2 is not None and len(self.pauli2) != 15:
            raise ValueError("pauli2 must have 15 entries (stim PAULI_CHANNEL_2 order)")

    def replace(self, **kw) -> Params:
        return replace(self, **kw)

    @property
    def dem_exact(self) -> bool:
        """Whether stim's exact DEM conversion (approximate_disjoint_errors
        False, the board's pinned derivation) accepts the channels this
        parameter set emits."""
        if self.pauli2 is not None or (self.p2 is not None and self.bias2 != DEPOLARIZING_BIAS):
            return False
        if self.p1 is not None and math.isinf(self.bias1):
            pass                                   # emitted as Z_ERROR, exact
        if self.leakage.mode == "heralded" and self.leakage.active:
            return False
        if self.correlated.exclusive and len(self.correlated.gate_pairs) > 1:
            return False
        if self.idle.kind == "slot":
            for ch in (self.idle.gate, self.idle.meas):
                if isinstance(ch, tuple) and any(x == 0 for x in ch) and sum(ch) > 0:
                    return False
        return True


@dataclass(frozen=True)
class Profile:
    name: str
    params: Params
    timing: Timing = PARALLEL
    graph: Graph | None = None
    source: str = ""

    def replace(self, **kw) -> Profile:
        return replace(self, **kw)

    def with_timing(self, timing) -> Profile:
        return replace(self, timing=Timing.from_name(timing))

    def with_graph(self, graph: Graph | None) -> Profile:
        return replace(self, graph=graph)

    def with_params(self, **kw) -> Profile:
        return replace(self, params=self.params.replace(**kw))

    def with_bias(self, eta: float, eta2: float | None = None,
                  idle: bool = True) -> Profile:
        """Set the single-qubit bias to eta and the two-qubit bias to eta2
        (default: the same eta). With `idle`, a slot idle given as scalar
        rates is biased by eta as well; duration and tuple idles are
        unchanged."""
        kw = dict(bias1=eta, bias2=eta if eta2 is None else eta2)
        p = self.params
        if idle and p.idle.kind == "slot" and not isinstance(p.idle.gate, tuple) \
                and not isinstance(p.idle.meas, tuple):
            kw["idle"] = replace(p.idle, bias=eta)
        return self.with_params(**kw)

    @property
    def dem_exact(self) -> bool:
        return self.params.dem_exact


def scaled(params: Params, scale: float) -> dict:
    """Every rate of `params` multiplied by `scale`, durations unchanged.
    The arithmetic mirrors ion_noise.ion_params so the ion presets stay
    bit-identical: per-second idle rates are scale / (2 T2) and
    scale / (4 T1), multiplied by the layer duration at emission time."""
    if scale < 0:
        raise ValueError("scale must be non-negative")

    def s(v):
        return None if v is None else scale * v

    def s_idle(ch):
        if ch is None:
            return None
        if isinstance(ch, tuple):
            return tuple(scale * x for x in ch)
        return scale * ch
    out = {
        "scale": scale,
        "p2": s(params.p2), "p1": s(params.p1),
        "p_reset": s(params.p_reset), "p_meas": s(params.p_meas),
        "pauli2": None if params.pauli2 is None else tuple(scale * x for x in params.pauli2),
        "leak_meas": scale * params.leakage.per_meas_layer,
        "leak_2q": scale * params.leakage.per_2q,
        "corr": tuple((p, scale * r) for p, r in params.correlated.gate_pairs),
        "crosstalk": scale * params.correlated.crosstalk,
        "t_2q": params.t_2q, "t_1q": params.t_1q,
        "t_meas": params.t_meas, "t_reset": params.t_reset,
    }
    idle = params.idle
    if idle.kind == "duration":
        out["z_rate"] = scale / (2.0 * idle.T2)
        out["xy_rate"] = scale / (4.0 * idle.T1)
    elif idle.kind == "slot":
        out["idle_gate"] = s_idle(idle.gate)
        out["idle_meas"] = s_idle(idle.meas)
    return out


def summarize(profile: Profile, scale: float = 1.0) -> str:
    """One-line human summary of the rates at `scale`."""
    r = scaled(profile.params, scale)
    parts = [f"{profile.name} @ scale {scale:g} [{profile.timing.name}"
             f"{', ' + profile.graph.name if profile.graph else ''}]:"]
    for k in ("p2", "p1", "p_reset", "p_meas"):
        if r[k] is not None:
            parts.append(f"{k}={r[k]:.3g}")
    p = profile.params
    if p.bias1 != DEPOLARIZING_BIAS or p.bias2 != DEPOLARIZING_BIAS:
        parts.append(f"bias1={p.bias1:g} bias2={p.bias2:g}")
    parts.append(f"idle={p.idle.kind}/{p.idle.scope}")
    if p.leakage.active:
        parts.append(f"leak_meas={r['leak_meas']:.3g} leak_2q={r['leak_2q']:.3g} ({p.leakage.mode})")
    if p.correlated.active:
        parts.append(f"corr={r['corr']} crosstalk={r['crosstalk']:.3g}")
    return " ".join(parts)
