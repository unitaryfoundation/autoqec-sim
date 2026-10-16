"""Trapped-ion Pauli noise recipe for qLDPC memory circuits (see SPEC.md).

Takes a noiseless stim memory circuit in the qldpc-challenge board's
TICK-layer convention (verify/circuit_tools.build_css_memory) and applies
the ion model of SPEC.md, every parameter multiplied by one scalar `scale`.
The board's apply_noise puts one rate p on everything; this recipe puts
distinct rates on two-qubit gates, single-qubit gates, resets, measurements
and idles, with the idle rate set by the wall-clock duration (or slot
class) of each layer.

Two named parameter sets (SPEC.md tables):

  "forte-2308.05071"  171Yb+ Forte, duration-based idle from T2/T1 (default)
  "chain-2606.06455"  133Ba+ 40-ion chain of arXiv:2606.06455, per-slot
                      idle, MCM-round idle and a leakage-to-Pauli proxy

Two timings:

  "parallel"    every TICK layer runs in one gate time (default)
  "sequential"  every two-qubit gate in its own time slot, idle on every
                other qubit for the slot; measurement lasts its full
                duration (the chain model of arXiv:2503.22071)

Run from the qldpc-challenge checkout so the pinned stim is used:

    cd <qldpc-challenge> && uv run --with stim python <this file> [--scale S]
        [--circuit F] [--params NAME] [--timing parallel|sequential] [--test]

Set QLDPC_ROOT if the checkout is not the sibling directory of this file's
parent.
"""

import argparse
import importlib.util
import os
import sys

import stim

QLDPC_ROOT = os.environ.get(
    "QLDPC_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qldpc-challenge"))
sys.path.insert(0, os.path.join(QLDPC_ROOT, "verify"))
import circuit_tools as ct

# ------------------------------------------------------------ nominal values
# scale = 1 reproduces these. Sources and labelled assumptions: SPEC.md.
P2_REF = 4.0e-3          # two-qubit (MS-compiled ZZ / CX) error, IonQ Forte
P1_REF = 2.0e-4          # single-qubit gate error, IonQ Forte
P_RESET_REF = 1.0e-3     # state-preparation flip     (split of 0.5% SPAM,
P_MEAS_REF = 4.0e-3      # measurement flip            assumption)
T2_REF = 1.0             # seconds, IonQ Forte product page ("~1s")
T1_REF = 10.0            # seconds, low end of the page's "10-100s"

# Layer durations in seconds. A layer lasts as long as its slowest
# instruction class; idle qubits dephase for that long.
T_2Q = 900e-6            # IonQ Forte average two-qubit gate duration
T_1Q = 110e-6            # IonQ Forte single-qubit gate duration
T_MEAS = 200e-6          # ASSUMPTION: Yb+ fluorescence detection at ~1e-3
T_RESET = 20e-6          # ASSUMPTION: optical-pumping state preparation

# Named parameter sets. "idle" selects how the idle channel of a layer is
# formed: "duration" (pz = t / 2 T2, px = py = t / 4 T1 with t the layer
# duration) or "slot" (fixed (px, py, pz) per layer class, as tabulated by
# arXiv:2606.06455 Table V, which does not distinguish one- and two-qubit
# gate durations). Durations of the chain set are derived, not measured
# (SPEC.md); they are reported, not used by the recipe.
PARAM_SETS = {
    "forte-2308.05071": {
        "p2": P2_REF, "p1": P1_REF, "p_reset": P_RESET_REF, "p_meas": P_MEAS_REF,
        "idle": "duration", "T2": T2_REF, "T1": T1_REF,
        "t_2q": T_2Q, "t_1q": T_1Q, "t_meas": T_MEAS, "t_reset": T_RESET,
        "leak_mcm": 0.0,
    },
    "chain-2606.06455": {
        "p2": 8.5e-3,        # Fig. 6 mean of fully-spanning two-qubit DRB (read
                             # off Fig. 7's reference line; ASSUMPTION +-1e-3)
        "p1": 3.2e-4,        # Table V one-qubit depolarizing
        "p_reset": 3.2e-4,   # ASSUMPTION: not tabulated; single-qubit-op rate
        "p_meas": 9.0e-3,    # Table V measurement bit-flip
        "idle": "slot",
        # Table V idle (6 2S1/2) per gate slot: X error 1e-5, Z error 4e-5.
        # The bit-flip class is split equally over X and Y (ASSUMPTION, as
        # the Forte set's twirled-T1 term); stim's exact DEM conversion
        # needs all three components positive, and a Y term of zero would
        # force approximate_disjoint_errors, which the board's pinned
        # derive_dem does not enable.
        "idle_gate": (5.0e-6, 5.0e-6, 4.0e-5),
        "idle_meas": (5.0e-5, 5.0e-5, 4.0e-4),  # Table V idle per MCM round: X 1e-4, Z 4e-4
        "leak_mcm": 1.7e-3,  # Table V leakage per MCM round; modelled as the
                             # paper's own simulation proxy: full depolarization
                             # of every non-measured qubit with this probability
        "T2star": 1.28,      # Appendix J, 1.28(16) s (Gaussian Ramsey fit)
        "t_2q": 460e-6,      # DERIVED from Table VI cycle times (SPEC.md)
        "t_1q": 460e-6,      # same slot as a two-qubit gate (Table V remark)
        "t_meas": 9.7e-3,    # DERIVED MCM-round duration from Table VI
        "t_reset": 460e-6,   # ASSUMPTION: one slot
    },
}
DEFAULT_PARAMS = "forte-2308.05071"
TIMINGS = ("parallel", "sequential")

RESET_FLIP = ct.RESET_FLIP
MEASURES = ct.MEASURES
ANNOTATIONS = ct.ANNOTATIONS


def ion_params(scale=1.0, params=DEFAULT_PARAMS):
    """All rates of the model at a given scale (rates, not durations, scale).

    The reference rate is p2 = scale * P2; every other rate is a fixed
    multiple of it, so plotting LER against p2 (or against `scale`) is the
    ion analogue of plotting against the board's p.
    """
    if scale < 0:
        raise ValueError("scale must be non-negative")
    if isinstance(params, str):
        if params not in PARAM_SETS:
            raise ValueError(f"unknown parameter set {params!r}; "
                             f"choose from {sorted(PARAM_SETS)}")
        base = PARAM_SETS[params]
        name = params
    else:
        base, name = dict(params), "custom"
    out = {
        "set": name,
        "scale": scale,
        "p2": scale * base["p2"],
        "p1": scale * base["p1"],
        "p_reset": scale * base["p_reset"],
        "p_meas": scale * base["p_meas"],
        "idle": base["idle"],
        "leak_mcm": scale * base.get("leak_mcm", 0.0),
        "t_2q": base["t_2q"], "t_1q": base["t_1q"],
        "t_meas": base["t_meas"], "t_reset": base["t_reset"],
    }
    if base["idle"] == "duration":
        # per-second idle rates; multiplied by the layer duration below
        out["z_rate"] = scale / (2.0 * base["T2"])      # Z error prob per second
        out["xy_rate"] = scale / (4.0 * base["T1"])     # X and Y error prob per second
    else:
        out["idle_gate"] = tuple(scale * x for x in base["idle_gate"])
        out["idle_meas"] = tuple(scale * x for x in base["idle_meas"])
    return out


def idle_channel(params, duration):
    """(px, py, pz) for a qubit idling `duration` seconds under a
    duration-based set: pure dephasing from T2 plus a Pauli-twirled T1 term.
    Linear in duration (t << T1, T2)."""
    px = py = min(0.25, params["xy_rate"] * duration)
    pz = min(0.5, params["z_rate"] * duration)
    return px, py, pz


def layer_classes(seg):
    """Set of instruction classes present in a TICK segment: 'meas',
    'reset', '2q', '1q'. Annotations do not count."""
    cls = set()
    for inst in seg:
        name = inst.name
        if name in ANNOTATIONS:
            continue
        if name in MEASURES:
            cls.add("meas")
        elif name in RESET_FLIP:
            cls.add("reset")
        elif stim.gate_data(name).is_two_qubit_gate:
            cls.add("2q")
        else:
            cls.add("1q")
    return cls


def layer_duration(seg, params=None):
    """Wall-clock length of one TICK segment: the slowest instruction class
    present. Annotations do not count. 0 if the layer does nothing. With
    `params` None the Forte module constants are used."""
    t_2q, t_1q, t_meas, t_reset = (T_2Q, T_1Q, T_MEAS, T_RESET) if params is None \
        else (params["t_2q"], params["t_1q"], params["t_meas"], params["t_reset"])
    dur = {"meas": t_meas, "reset": t_reset, "2q": t_2q, "1q": t_1q}
    cls = layer_classes(seg)
    return max((dur[c] for c in cls), default=0.0)


def idle_for_layer(params, seg):
    """(px, py, pz) applied to every qubit idle during `seg`."""
    if params["idle"] == "duration":
        dur = layer_duration(seg, params)
        return idle_channel(params, dur) if dur > 0 else (0.0, 0.0, 0.0)
    cls = layer_classes(seg)
    if not cls:
        return (0.0, 0.0, 0.0)
    return params["idle_meas"] if "meas" in cls else params["idle_gate"]


def circuit_qubits(circuit):
    """Every qubit index the circuit operates on (data and ancilla)."""
    qs = set()
    for inst in circuit.flattened():
        if inst.name in ANNOTATIONS:
            continue
        for t in inst.targets_copy():
            if t.qubit_value is not None:
                qs.add(t.qubit_value)
    return sorted(qs)


def serialize_two_qubit(circuit):
    """The sequential-timing skeleton: within every TICK segment the
    non-two-qubit instructions (and annotations) stay together in a first
    layer, then every two-qubit gate gets a layer of its own, in listed
    order. Equal to ion-search/lib.serialize_cx on circuits whose only
    two-qubit gate is CX (checked by --test)."""
    out = stim.Circuit()
    first = True
    for seg in ct._segments(circuit.flattened()):
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
        j = 0
        for inst in two:
            t = [x.qubit_value for x in inst.targets_copy()]
            for a, b in zip(t[0::2], t[1::2]):
                if j or rest:
                    out.append("TICK")
                out.append(inst.name, [a, b])
                j += 1
    return out


def apply_ion_noise(circuit, scale=1.0, idle_qubits=None, params=DEFAULT_PARAMS,
                    timing="parallel"):
    """The ion-model noisy circuit for a noiseless skeleton.

    `circuit` must be noiseless (run circuit_tools.strip_noise first if it is
    a committed board circuit). Placement per SPEC.md:

      - DEPOLARIZE2(p2) after every two-qubit gate;
      - DEPOLARIZE1(p1) after every single-qubit unitary;
      - X_ERROR(p_reset) after R, Z_ERROR(p_reset) after RX;
      - measurement flip p_meas on M / MX;
      - PAULI_CHANNEL_1(px, py, pz) on every qubit not operated in an active
        layer, with (px, py, pz) set by that layer's duration (duration-based
        set) or class (per-slot set);
      - per-slot sets with a leakage rate: DEPOLARIZE1(3/4 leak_mcm) on every
        qubit not operated in a measurement layer, after the idle channel
        (a fully depolarized state with probability leak_mcm).

    timing "sequential" first rewrites the skeleton with serialize_two_qubit
    so every two-qubit gate is its own layer; strip_noise of the result is
    then that serialized skeleton, not the input.

    Idle noise goes on every qubit the circuit uses (data and ancilla),
    unlike the board's data-only rule: in a chain every ion waits. Idle
    errors on an ancilla between its measurement and its next reset have
    no detector or observable effect and stim drops them from the DEM.
    `idle_qubits` overrides the idle set (e.g. range(n) for the board rule).
    """
    if timing not in TIMINGS:
        raise ValueError(f"timing must be one of {TIMINGS}")
    if timing == "sequential":
        circuit = serialize_two_qubit(circuit)
    p = ion_params(scale, params)
    all_qubits = list(idle_qubits) if idle_qubits is not None \
        else circuit_qubits(circuit)
    leak = 0.75 * p["leak_mcm"]
    out = stim.Circuit()
    segs = ct._segments(circuit.flattened())
    for si, seg in enumerate(segs):
        if si:
            out.append("TICK")
        touched, active = set(), False
        for inst in seg:
            name = inst.name
            if name in ANNOTATIONS:
                out.append(inst)
                continue
            targets = inst.targets_copy()
            qubits = [t.qubit_value for t in targets
                      if t.qubit_value is not None]
            touched.update(qubits)
            active = True
            if name in MEASURES:
                out.append(name, targets, p["p_meas"])
            elif name in RESET_FLIP:
                out.append(inst)
                out.append(RESET_FLIP[name], qubits, p["p_reset"])
            else:
                gd = stim.gate_data(name)
                if gd.is_noisy_gate:
                    raise ValueError(f"input is not noiseless: {name}")
                out.append(inst)
                if gd.is_two_qubit_gate:
                    out.append("DEPOLARIZE2", targets, p["p2"])
                elif gd.is_unitary:
                    out.append("DEPOLARIZE1", qubits, p["p1"])
                else:
                    raise ValueError(f"unsupported instruction {name}")
        if active:
            idle = [q for q in all_qubits if q not in touched]
            if idle:
                px, py, pz = idle_for_layer(p, seg)
                if px + py + pz > 0:
                    out.append("PAULI_CHANNEL_1", idle, [px, py, pz])
                if leak > 0 and "meas" in layer_classes(seg):
                    out.append("DEPOLARIZE1", idle, leak)
    return out


# ------------------------------------------------------------------ summary

def fault_expectation(dem):
    """Sum of mechanism probabilities: expected faults per shot."""
    return sum(inst.args_copy()[0] for inst in dem.flattened()
               if inst.type == "error")


def mechanism_count(dem):
    return sum(1 for inst in dem.flattened() if inst.type == "error")


def channel_budget(circuit):
    """Expected noise-channel firings per shot by class: two-qubit gates,
    single-qubit gates, resets, measurements, idles, leakage proxy. Linear
    in scale, unlike the DEM total, which merges equal-signature mechanisms.
    A DEPOLARIZE1 not preceded by a unitary in the same layer is the
    leakage proxy; under the board recipe DEPOLARIZE1 also carries idles."""
    b = {"2q": 0.0, "1q": 0.0, "reset": 0.0, "meas": 0.0, "idle": 0.0, "leak": 0.0}
    for seg in ct._segments(circuit.flattened()):
        unitary_qubits = set()
        for inst in seg:
            name = inst.name
            if name in ANNOTATIONS:
                continue
            gd = stim.gate_data(name)
            if gd.is_unitary and not gd.is_two_qubit_gate:
                unitary_qubits.update(t.qubit_value for t in inst.targets_copy())
        for inst in seg:
            name = inst.name
            args = inst.gate_args_copy()
            if not args or name in ANNOTATIONS:
                continue
            tq = [t.qubit_value for t in inst.targets_copy()]
            nt = len(tq)
            if name in MEASURES:
                b["meas"] += args[0] * nt
            elif name in ("X_ERROR", "Z_ERROR"):
                b["reset"] += args[0] * nt
            elif name == "DEPOLARIZE1":
                if unitary_qubits and set(tq) <= unitary_qubits:
                    b["1q"] += args[0] * nt
                elif any(i.name == "PAULI_CHANNEL_1" for i in seg) and \
                        "meas" in layer_classes(seg):
                    b["leak"] += args[0] * nt    # ion recipe: after the idle channel
                else:
                    b["1q"] += args[0] * nt      # board recipe: also its idles
            elif name == "DEPOLARIZE2":
                b["2q"] += args[0] * (nt // 2)
            elif name == "PAULI_CHANNEL_1":
                b["idle"] += sum(args) * nt
    return b


def channel_faults(circuit):
    return sum(channel_budget(circuit).values())


def summarize(circuit, scale, label):
    dem = circuit.detector_error_model(decompose_errors=False,
                                       flatten_loops=True)
    b = channel_budget(circuit)
    tot = sum(b.values())
    share = " ".join(f"{k}={v / tot:.0%}" for k, v in b.items() if v)
    print(f"{label:<26} scale={scale:<6g} mechanisms={mechanism_count(dem):>6}"
          f"  channel faults/shot={tot:.4f} ({share})"
          f"  DEM faults/shot={fault_expectation(dem):.4f}"
          f"  detectors={dem.num_detectors}  observables={dem.num_observables}")
    return dem


# -------------------------------------------------------------------- tests

def _linear_in_scale(full, half):
    for a, b in zip(full.flattened(), half.flattened()):
        assert a.name == b.name
        if a.name in ANNOTATIONS:
            continue
        for x, y in zip(a.gate_args_copy(), b.gate_args_copy()):
            assert abs(x - 2 * y) <= 1e-15 * max(1.0, x), ("linearity", a, b)


def run_tests(skeleton, n_data):
    """Round-trip and structural checks; raises AssertionError on failure."""
    noisy = apply_ion_noise(skeleton, 1.0)
    # 1. strip(apply(skeleton)) == skeleton
    assert str(ct.strip_noise(noisy)) == str(skeleton), "strip round-trip"
    # 2. the skeleton still passes the board's structural checks
    assert ct.skeleton_errors(skeleton) == []
    assert ct.layer_conflict_errors(skeleton) == []
    # 3. detectors/observables/measurement record unchanged
    assert noisy.num_detectors == skeleton.num_detectors
    assert noisy.num_observables == skeleton.num_observables
    assert noisy.num_measurements == skeleton.num_measurements
    # 4. apply is idempotent on strip: apply(strip(apply(s))) == apply(s)
    assert str(apply_ion_noise(ct.strip_noise(noisy), 1.0)) == str(noisy)
    # 5. scale = 0 gives a noiseless circuit (DEM has no mechanisms)
    zero = apply_ion_noise(skeleton, 0.0)
    assert mechanism_count(zero.detector_error_model()) == 0, "scale 0"
    # 6. every channel argument is exactly linear in scale (the DEM's totals
    #    are not: stim merges equal-signature mechanisms as p + q - 2pq)
    half = apply_ion_noise(skeleton, 0.5)
    _linear_in_scale(noisy, half)
    assert abs(channel_faults(noisy) - 2 * channel_faults(half)) < 1e-9
    # 7. board-rule idle set reproduces the board's own layer/idle structure
    #    when rates coincide: same qubits idle in every active layer
    board = ct.apply_noise(skeleton, n_data, 1e-3)
    ours = apply_ion_noise(skeleton, 1.0, idle_qubits=range(n_data))
    b_idle = [set(t.qubit_value for t in i.targets_copy())
              for i in board.flattened() if i.name == "DEPOLARIZE1"]
    o_idle = [set(t.qubit_value for t in i.targets_copy())
              for i in ours.flattened() if i.name == "PAULI_CHANNEL_1"]
    assert b_idle == o_idle, "idle placement differs from the board's"
    # 8. the noisy circuit samples
    noisy.compile_detector_sampler(seed=1).sample(shots=200)
    # 9. the default arguments are the Forte set with parallel timing
    assert str(apply_ion_noise(skeleton, 1.0, params="forte-2308.05071",
                               timing="parallel")) == str(noisy)
    # 10. chain set: round trip, linearity, scale 0, samples; on a circuit
    #     with a mid-circuit measurement layer the leakage proxy adds
    #     DEPOLARIZE1 on the idle qubits of that layer and nowhere else
    chain = apply_ion_noise(skeleton, 1.0, params="chain-2606.06455")
    assert str(ct.strip_noise(chain)) == str(skeleton), "chain strip round-trip"
    _linear_in_scale(chain, apply_ion_noise(skeleton, 0.5, params="chain-2606.06455"))
    assert mechanism_count(apply_ion_noise(skeleton, 0.0, params="chain-2606.06455")
                           .detector_error_model()) == 0
    chain.compile_detector_sampler(seed=1).sample(shots=200)
    pc = ion_params(1.0, "chain-2606.06455")
    for seg in ct._segments(chain.flattened()):
        cls = layer_classes(ct.strip_noise(stim.Circuit("\n".join(map(str, seg)))))
        leaks = [i for i in seg if i.name == "DEPOLARIZE1"
                 and abs(i.gate_args_copy()[0] - 0.75 * pc["leak_mcm"]) < 1e-15]
        idles = [i for i in seg if i.name == "PAULI_CHANNEL_1"]
        if "meas" in cls:
            assert len(leaks) == len(idles) <= 1
            if idles:
                assert list(idles[0].gate_args_copy()) == list(pc["idle_meas"])
                assert idles[0].targets_copy() == leaks[0].targets_copy()
        else:
            assert not leaks
            for i in idles:
                assert list(i.gate_args_copy()) == list(pc["idle_gate"])
    # 11. sequential timing: strip gives the serialized skeleton, one
    #     two-qubit gate per layer, same records; idle set of a CX layer is
    #     everything but the two gate qubits
    seq = apply_ion_noise(skeleton, 1.0, timing="sequential")
    ser = serialize_two_qubit(skeleton)
    assert str(ct.strip_noise(seq)) == str(ser)
    assert ct.layer_conflict_errors(ser) == []
    assert seq.num_measurements == skeleton.num_measurements
    assert seq.num_detectors == skeleton.num_detectors
    n_cx = sum(len(i.targets_copy()) // 2 for i in skeleton.flattened()
               if i.name != "TICK" and i.name not in ANNOTATIONS
               and stim.gate_data(i.name).is_two_qubit_gate)
    cx_layers = [s for s in ct._segments(ser.flattened())
                 if any(i.name not in ANNOTATIONS and stim.gate_data(i.name).is_two_qubit_gate
                        for i in s)]
    assert len(cx_layers) == n_cx
    assert all(len(s) == 1 for s in cx_layers)
    nq = len(circuit_qubits(skeleton))
    for s in ct._segments(seq.flattened()):
        names = [i.name for i in s]
        if "CX" in names:
            pc1 = [i for i in s if i.name == "PAULI_CHANNEL_1"]
            assert len(pc1) == 1 and len(pc1[0].targets_copy()) == nq - 2
    print("tests: ok")


def run_ion_search_agreement():
    """The sequential timing must reproduce ion-search/lib.dem_of(skel, 1,
    'seq') (that tool serializes CX layers itself and calls this recipe with
    default arguments) on the [[9,1,3]] one-round circuits, both bases.
    Skipped with a message when ion-search is not checked out alongside."""
    here = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(os.path.dirname(here), "ion-search", "lib.py")
    if not os.path.exists(lib_path):
        print("ion-search agreement: skipped (no ../ion-search/lib.py)")
        return
    import json
    spec = importlib.util.spec_from_file_location("ion_search_lib", lib_path)
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    _, HX, HZ = L.IE.load_code("9-1-3")
    sched = json.load(open(os.path.join(L.IONEXP, "circuits", "9-1-3", "schedule.json")))
    lx = [[tuple(e) for e in l] for l in sched["lay_x"]]
    lz = [[tuple(e) for e in l] for l in sched["lay_z"]]
    for b in "XZ":
        skel, _ = L.build(HX, HZ, b, lx, lz)
        for scale in (1.0, 0.25):
            theirs = L.dem_of(skel, scale, "seq")
            ours = ct.derive_dem(apply_ion_noise(skel, scale, timing="sequential"))
            assert ct.dem_matches(ours, theirs), ("sequential DEM differs", b, scale)
            assert str(ion_circuit := apply_ion_noise(L.serialize_cx(skel), scale)) == \
                str(apply_ion_noise(skel, scale, timing="sequential")), ("circuit", b)
            del ion_circuit
            par_theirs = L.dem_of(skel, scale, "par")
            par_ours = ct.derive_dem(apply_ion_noise(skel, scale))
            assert ct.dem_matches(par_ours, par_theirs), ("parallel DEM differs", b)
        # the chain set on the one-round circuit: identical mechanism
        # signatures (the leakage proxy never fires: the last layer measures
        # every qubit), different probabilities
        for timing in TIMINGS:
            f = ct.dem_columns(ct.derive_dem(apply_ion_noise(skel, 1.0, timing=timing)))
            c = ct.dem_columns(ct.derive_dem(apply_ion_noise(
                skel, 1.0, params="chain-2606.06455", timing=timing)))
            assert f == c, ("chain/forte mechanism sets differ", b, timing)
    print("ion-search agreement: ok ([[9,1,3]] one-round, X and Z, scales 1 and 0.25)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--circuit", default=os.path.join(
        QLDPC_ROOT, "circuits", "40-10-4", "memory_x.stim"),
        help="a committed (noisy) or noiseless board memory circuit")
    ap.add_argument("--n", type=int, default=None,
                    help="number of data qubits (default: from the slug)")
    ap.add_argument("--scale", type=float, nargs="*", default=[1.0, 0.25],
                    help="scale values to summarize")
    ap.add_argument("--params", default=DEFAULT_PARAMS, choices=sorted(PARAM_SETS))
    ap.add_argument("--timing", default="parallel", choices=TIMINGS)
    ap.add_argument("--test", action="store_true", help="run the checks")
    ap.add_argument("--out", default=None,
                    help="write the scale[0] noisy circuit here")
    a = ap.parse_args()

    raw = stim.Circuit.from_file(a.circuit)
    skeleton = ct.strip_noise(raw)
    n = a.n
    if n is None:
        slug = os.path.basename(os.path.dirname(os.path.abspath(a.circuit)))
        try:
            n = int(slug.split("-")[0])
        except ValueError:
            sys.exit("pass --n: could not read n from the circuit path")

    print(f"stim {stim.__version__}; circuit {a.circuit}; n_data={n}; "
          f"layers={len(ct._segments(skeleton.flattened()))}")
    for name in PARAM_SETS:
        p = ion_params(1.0, name)
        if p["idle"] == "duration":
            idle = (f"idle pz/layer: 2q={idle_channel(p, p['t_2q'])[2]:.2e}"
                    f" meas={idle_channel(p, p['t_meas'])[2]:.2e}"
                    f" 1q={idle_channel(p, p['t_1q'])[2]:.2e}")
        else:
            idle = (f"idle (px,py,pz)/slot: gate={p['idle_gate']} meas={p['idle_meas']}"
                    f" leak/MCM={p['leak_mcm']:g}")
        print(f"{name}: p2={p['p2']:g} p1={p['p1']:g} p_reset={p['p_reset']:g} "
              f"p_meas={p['p_meas']:g}; {idle}")
    if a.test:
        run_tests(skeleton, n)
        run_ion_search_agreement()
    summarize(ct.apply_noise(skeleton, n, ct.P_REF), ct.P_REF, "board recipe")
    for s in a.scale:
        for name in PARAM_SETS:
            for timing in TIMINGS:
                summarize(apply_ion_noise(skeleton, s, params=name, timing=timing), s,
                          f"{name} {timing[:3]}")
    if a.out:
        apply_ion_noise(skeleton, a.scale[0], params=a.params, timing=a.timing).to_file(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
