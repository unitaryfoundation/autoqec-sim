# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""The evaluator against five published memory experiments.

Each row takes a code and a noise profile whose every parameter is read
from the paper (the assumptions are listed per row), builds the package's
own memory circuit, and scores it with the proxy tier at one to three
rounds and with Tesseract sampling (and minimum-weight matching where
``pymatching`` is importable and the detector error model decomposes) at
the paper's round count. The paper's own per-round logical error, with its
own decoder, sits beside it. The comparison is between the package's
noise model on the package's circuit and the number the paper reports; it
is not a prediction about a device, and each row names what the circuit
and the model leave out.

    uv run --with pymatching python -m experiments.hardware_comparison.run run
    uv run python -m experiments.hardware_comparison.run report
    uv run python -m experiments.hardware_comparison.run figure
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from autoqec_sim import memory_objective as mo
from autoqec_sim.code import StabilizerCode
from autoqec_sim.known_codes import get, get_subsystem
from autoqec_sim.noise import Profile, presets
from autoqec_sim.noise.profile import Idle, Params
from autoqec_sim.noise.timing import PARALLEL, partial
from autoqec_sim.symplectic import independent_rows

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "runs"
REPORT = HERE / "REPORT.md"
FIGURE = Path("docs/assets/figures/hardware_comparison")


# ------------------------------------------------------------------ codes
def _pauli_rows(h: np.ndarray, letter: str) -> list[str]:
    return ["".join(letter if v else "I" for v in row) for row in h]


def toric_18() -> StabilizerCode:
    """The [[18,2,3]] toric code on a 3 x 3 torus: horizontal edge (i, j)
    is qubit 3i + j, vertical edge (i, j) is qubit 9 + 3i + j. The paper's
    parity-check matrices are a relabeling; the evaluator depends on the
    stabilizer group and its own gate order only."""
    size = 3

    def h(i, j):
        return (i % size) * size + (j % size)

    def v(i, j):
        return 9 + (i % size) * size + (j % size)

    hx, hz = [], []
    for i in range(size):
        for j in range(size):
            x = np.zeros(18, dtype=np.uint8)
            for q in (h(i, j), h(i, j - 1), v(i, j), v(i - 1, j)):
                x[q] ^= 1
            hx.append(x)
            z = np.zeros(18, dtype=np.uint8)
            for q in (h(i, j), h(i + 1, j), v(i, j), v(i, j + 1)):
                z[q] ^= 1
            hz.append(z)
    hx, hz = np.array(hx), np.array(hz)
    hx, hz = hx[independent_rows(hx)], hz[independent_rows(hz)]
    return StabilizerCode.from_stabilizers(_pauli_rows(hx, "X") + _pauli_rows(hz, "Z"))


def bb_18_4_3() -> StabilizerCode:
    """The bivariate-bicycle [[18,4,3]] code of the 40-ion paper (its Table
    III): l = m = 3, A = 1 + x, B = 1 + y + x y^2, HX = [A | B],
    HZ = [B^T | A^T]."""
    size = 3
    shift = np.roll(np.eye(size, dtype=np.uint8), 1, axis=1)
    x = np.kron(shift, np.eye(size, dtype=np.uint8))
    y = np.kron(np.eye(size, dtype=np.uint8), shift)
    eye = np.eye(size * size, dtype=np.uint8)
    a = (eye + x) % 2
    b = (eye + y + x @ y @ y) % 2
    hx = np.concatenate([a, b], axis=1) % 2
    hz = np.concatenate([b.T, a.T], axis=1) % 2
    hx, hz = hx[independent_rows(hx)], hz[independent_rows(hz)]
    return StabilizerCode.from_stabilizers(_pauli_rows(hx, "X") + _pauli_rows(hz, "Z"))


# --------------------------------------------------------------- profiles
def google_2023() -> Profile:
    """Google Sycamore 2023 (arXiv:2207.06431): the component Pauli error
    rates of supplementary Table I (SQ 1.09e-3, CZ 6.05e-3 including its
    2.0e-4 leakage and 9.5e-4 crosstalk as Pauli noise, reset 1.86e-3,
    readout 1.96e-2) and the main text's mean T1 = 20 us and T2 CPMG =
    30 us. Leakage from heating (6.4e-4 per cycle) is omitted. Durations:
    the 921 ns cycle of Fig. 1 with 500 ns measurement and 160 ns reset
    (the supplement's error budget uses 896 ns); the CZ (30 ns) and
    single-qubit (25 ns) durations are not stated and are ASSUMPTIONS."""
    p = Params(p2=6.05e-3, p1=1.09e-3, p_reset=1.86e-3, p_meas=1.96e-2,
               idle=Idle(kind="duration", T1=20e-6, T2=30e-6, scope="all"),
               t_2q=30e-9, t_1q=25e-9, t_meas=500e-9, t_reset=160e-9,
               notes="Google Sycamore 2023 distance-3 surface code")
    return Profile("google-2023", p, PARALLEL, None, source="arXiv:2207.06431 supplementary Table III")


def google_2024_72q() -> Profile:
    """Google 72-qubit processor 2024 (arXiv:2408.13687), supplementary
    Table S4: CZ 2.8e-3 plus 5.5e-4 crosstalk plus 2.0e-4 leakage as Pauli
    noise, single-qubit 6.2e-4, readout 0.8e-2, reset 1.5e-3; CZ 37 ns and
    pi pulse 35 ns on this processor (supplement section I); 1.1 us cycle.
    ASSUMPTIONS: T1 = 73 us is the median physical qubit lifetime of
    Fig. S9, a quantity derived from T1 and T2 CPMG, used here as T1;
    T2 = 89 us is the 105-qubit processor's T2 CPMG from the main text;
    measurement 500 ns and reset 160 ns are carried over from the 2023
    paper, the 2024 paper stating neither. Leakage heating 2.5e-4 omitted."""
    p = Params(p2=3.55e-3, p1=6.2e-4, p_reset=1.5e-3, p_meas=0.8e-2,
               idle=Idle(kind="duration", T1=73e-6, T2=89e-6, scope="all"),
               t_2q=37e-9, t_1q=35e-9, t_meas=500e-9, t_reset=160e-9,
               notes="Google 72-qubit processor 2024 distance-3 surface code")
    return Profile("google-2024-72q", p, PARALLEL, None, source="arXiv:2408.13687 supplementary Table S4")


def quantinuum_h1_2021(idle: str = "none") -> Profile:
    """Quantinuum H1 2021 (arXiv:2107.07505), Table IV: two-qubit
    depolarizing 3.1e-3 plus spontaneous emission 2.75e-4 (folded in as
    depolarizing, an ASSUMPTION for its leakage half), single-qubit 7e-5
    plus 1.25e-5, measurement bit flip 2.4e-3 (its 5e-3 leakage omitted,
    the ancilla being re-initialized), initialization 1.66e-6 plus 3.33e-5
    leakage folded in as a bit flip. Three gates run at a time, as in the
    paper's three-parallel protocol. ``idle="none"`` is gates and SPAM only;
    ``idle="bound"`` adds the paper's depth-one memory error bound 5e-4 per
    slot as a Z-dominated channel (ASSUMPTION) and the 2.3e-4 measurement
    crosstalk in the measurement slot. The paper's own dephasing term is a
    fitted parameter, not a measurement, so the two variants bracket it.
    The device ran a flagged adaptive protocol (two flagged three-parallel
    circuits with three ancillas, then all six stabilizers unflagged when
    a flag or syndrome fired: 36 two-qubit gates, 60 when triggered, by
    our count from the paper's Fig. 7) in under 200 ms per cycle, most of
    it transport and cooling; this circuit is the bare 24-gate extraction."""
    if idle == "none":
        idl = Idle(kind="none")
    else:
        idl = Idle(kind="slot", gate=(1e-5, 1e-5, 4.8e-4), meas=(1e-5, 1e-5, 4.8e-4 + 2.3e-4), scope="all")
    p = Params(p2=3.1e-3 + 2.75e-4, p1=7e-5 + 1.25e-5, p_reset=1.66e-6 + 3.33e-5, p_meas=2.4e-3,
               idle=idl, t_2q=8e-3, t_1q=1e-4, t_meas=1e-3, t_reset=1e-4,
               notes="Quantinuum H1 2021 Table IV")
    return Profile(f"h1-2021-{idle}", p, partial(3), None, source="arXiv:2107.07505 Table IV")


def ion_chain_2026() -> Profile:
    """The repository's chain preset: Table V of the 40-ion paper
    (arXiv:2606.06455), sequential timing. Its two-qubit rate 8.5e-3 is
    read from the paper's Fig. 6 (preset ASSUMPTION, about 1e-3 either
    way) and its reset rate 3.2e-4 is a preset ASSUMPTION."""
    return presets.chain()


def ibm_peekskill_2022(kind: str = "decoder") -> Profile:
    """ibm_peekskill 2022 (arXiv:2203.07205). ``kind="decoder"``: the
    per-operation error vector the paper gives its analytical matching
    decoder (supplementary section C: cx 0.0126, h 0.000266, id 0.001 per
    gate layer, idm 0.002 per measurement layer, measure 0.00713, reset
    0.0290; the 0.0142 initialization term is not modeled separately),
    used here as a circuit-level noise model. ``kind="median"``: medians of
    the per-qubit Tables S3 and S4 (CX error, single-qubit error, readout,
    reset, T1, T2, CX duration) with the duration idle model; the paper's
    own simulation used the per-qubit values directly and reports 0.059
    (Z) and 0.152 (X) per round, or 0.038 and 0.106 with an averaged
    model. Measurement with its conditional reset takes 768 ns in total."""
    if kind == "decoder":
        p = Params(p2=0.0126, p1=0.000266, p_reset=0.0290, p_meas=0.00713,
                   idle=Idle(kind="slot", gate=0.001, meas=0.002, scope="all"),
                   t_2q=430e-9, t_1q=35.55e-9, t_meas=768e-9, t_reset=0.0,
                   notes="ibm_peekskill circuit-level model of arXiv:2203.07205 supplementary section C")
    else:
        t1 = [420.3, 354.8, 331.7, 124.8, 131.7, 424.5, 249.4, 271.7, 357.0, 283.8, 280.9, 349.8, 399.3, 226.6, 259.8, 234.4, 195.5, 319.6, 278.1, 206.9, 278.3, 258.7, 364.1]
        t2 = [118.4, 119.8, 25.8, 77.3, 215.5, 59.7, 228.8, 316.0, 72.0, 188.8, 353.0, 345.0, 99.7, 217.4, 209.2, 311.7, 34.7, 167.6, 308.0, 132.4, 145.0, 14.6, 327.5]
        epg1 = [0.0143, 0.0171, 0.0315, 0.0145, 0.0212, 0.0310, 0.0287, 0.0207, 0.0217, 0.0367, 0.0410, 0.0694, 0.0473, 0.0280, 0.0324, 0.0118, 0.0485, 0.0143, 0.0177, 0.0169, 0.0265]
        readout = [1.24, 0.69, 0.99, 0.46, 0.84, 0.55, 0.48, 1.24, 0.26, 1.59, 0.36, 0.10, 1.19, 0.34, 0.75, 0.61, 0.28, 1.17, 0.50, 0.62, 0.23, 0.34, 0.40]
        reset = [3.3, 5.6, 5.4, 2.0, 1.2, 3.3, 1.1, 1.7, 1.8, 2.6, 1.2, 0.9, 4.6, 3.2, 1.2, 1.6, 1.2, 7.2, 0.7, 1.8, 0.8, 2.6, 0.8]
        cx = [0.95, 1.42, 1.02, 1.03, 1.49, 1.10, 0.82, 1.14, 3.96, 0.73, 0.55, 0.67, 0.94, 0.57, 0.81, 0.91, 0.53, 0.64, 0.49, 0.71, 0.76, 0.54, 1.39, 0.84]
        cx_ns = [483.6, 433.8, 334.2, 519.1, 504.9, 469.3, 440.9, 696.9, 426.7, 348.4, 362.7, 519.1, 362.7, 412.4, 384.0, 384.0, 426.7, 391.1, 547.6, 348.4, 362.7, 426.7, 526.2, 384.0]
        p = Params(p2=statistics.median(cx) / 100, p1=statistics.median(epg1) / 100,
                   p_reset=statistics.median(reset) / 100, p_meas=statistics.median(readout) / 100,
                   idle=Idle(kind="duration", T1=statistics.median(t1) * 1e-6, T2=statistics.median(t2) * 1e-6, scope="all"),
                   t_2q=statistics.median(cx_ns) * 1e-9, t_1q=35.55e-9, t_meas=768e-9, t_reset=0.0,
                   notes="ibm_peekskill medians of Tables S3 and S4")
    return Profile(f"peekskill-{kind}", p, PARALLEL, None, source="arXiv:2203.07205 Tables S3, S4, supplementary section C")


# ------------------------------------------------------------------ rows
# Measured per-round logical error (fractions) with the paper's decoder, per
# basis where the paper gives it. Every value is to be read from the cited
# table or figure, never inferred.
MEASURED = {
    "google_2023": {"paper": "acharya2023", "rounds": 25, "decoder": "tensor network (belief matching 3.118e-2)",
                    "basis_average": 3.028e-2, "note": "main text: 3.028 +- 0.023 percent per cycle, the distance-3 subset codes averaged over the Z and X bases, fit over cycles 3 to 25"},
    "google_2024_72q": {"paper": "acharya2025", "rounds": 25, "decoder": "neural network (Libra 6.1e-3, real-time matching 7.1e-3)",
                        "basis_average": 5.9e-3, "note": "72-qubit processor, supplementary Table S1 (Fig. 4c columns): 5.9 +- 0.1 e-3 per cycle, both bases averaged"},
    "quantinuum_h1_2021": {"paper": "ryananderson2021", "rounds": 6, "decoder": "lookup table on flagged syndromes",
                           "Z": 1.05e-2, "X": 1.75e-2, "note": "Table I: |0> 1.16e-2 and |1> 0.94e-2 averaged; |+> 1.8e-2 and |-> 1.7e-2 averaged"},
    "ion_chain_toric_18": {"paper": "tham2026", "rounds": 6, "decoder": "beam decoder",
                           "Z": 1.51e-2, "X": 2.88e-2, "note": "Table I per logical qubit (Z 0.76e-2, X 1.45e-2) converted to the two-qubit block"},
    "ion_chain_bb_18_4_3": {"paper": "tham2026", "rounds": 6, "decoder": "beam decoder",
                            "Z": 4.33e-2, "X": 7.58e-2, "note": "Table I per logical qubit (Z 1.10e-2, X 1.95e-2) converted to the four-qubit block"},
    "ibm_peekskill_2022": {"paper": "sundaresan2023", "rounds": 10,
                           "decoder": "matching, all shots (leakage post-selected Z 4.25e-2, X 8.65e-2; maximum likelihood on post-selected data 0.037 and 0.087 in the published abstract)",
                           "Z": 5.85e-2, "X": 1.10e-1, "note": "main text fits over rounds 0 to 10, 500,000 shots per run: |0> 0.059 and |1> 0.058 averaged; |+> 0.113 and |-> 0.107 averaged"},
}

ROWS = [
    # name, measured key, code, profile, connectivity, flags, note
    ("google_2023", "google_2023", lambda: get("rotated_surface_9"), google_2023, "square:3x3", "none",
     "two ASSUMED gate durations; leakage from heating omitted"),
    ("google_2024_72q", "google_2024_72q", lambda: get("rotated_surface_9"), google_2024_72q, "square:3x3", "none",
     "T1, T2, measurement and reset durations ASSUMED; leakage heating omitted"),
    ("quantinuum_h1_2021_noidle", "quantinuum_h1_2021", lambda: get("steane"), lambda: quantinuum_h1_2021("none"), "all_to_all", "none",
     "bare 24-gate extraction against the paper's flagged adaptive 36- to 60-gate protocol; no transport or cooling dephasing"),
    ("quantinuum_h1_2021_idlebound", "quantinuum_h1_2021", lambda: get("steane"), lambda: quantinuum_h1_2021("bound"), "all_to_all", "none",
     "as above, with the depth-one memory error bound as a Z-dominated idle channel (ASSUMPTION)"),
    ("ion_chain_toric_18", "ion_chain_toric_18", toric_18, ion_chain_2026, "all_to_all", "none",
     "two-qubit rate read from a figure (preset ASSUMPTION); the paper remaps qubits to ions per code"),
    ("ion_chain_bb_18_4_3", "ion_chain_bb_18_4_3", bb_18_4_3, ion_chain_2026, "all_to_all", "none",
     "as above"),
    ("ibm_peekskill_2022_flagged", "ibm_peekskill_2022", lambda: get_subsystem("heavy_hex_9"), lambda: ibm_peekskill_2022("decoder"), "all_to_all", "one",
     "gauge generators measured with one flag on each weight-4 Z gauge where the device uses two and ignores the flags of its weight-2 Z gauges; the paper's matching-decoder error vector as the noise model; no leakage"),
    ("ibm_peekskill_2022_bare", "ibm_peekskill_2022", lambda: get_subsystem("heavy_hex_9"), lambda: ibm_peekskill_2022("decoder"), "all_to_all", "none",
     "as above without flags"),
    ("ibm_peekskill_2022_medians", "ibm_peekskill_2022", lambda: get_subsystem("heavy_hex_9"), lambda: ibm_peekskill_2022("median"), "all_to_all", "one",
     "medians of the per-qubit tables with the duration idle model; the paper's averaged-model simulation gives 0.038 (Z) and 0.106 (X) per round"),
]


# ------------------------------------------------------------- evaluation
NOT_MATCHABLE = "not a matching code: a single data error flips three checks of one type"


def _matchable(code) -> bool:
    """True when every single data error flips at most two checks of each
    type, the condition under which the matching graph is the natural
    decoder (the surface, toric, and heavy-hexagon codes, not the Steane
    or bivariate-bicycle codes, where a data error flips three)."""
    stab = np.asarray(code.stabilizers, dtype=np.uint8).astype(bool)
    n = code.n
    x_part, z_part = stab[:, :n], stab[:, n:]
    x_rows, z_rows = x_part.any(1) & ~z_part.any(1), z_part.any(1) & ~x_part.any(1)
    if not (x_rows | z_rows).all():
        return False
    return int(x_part[x_rows].sum(0).max()) <= 2 and int(z_part[z_rows].sum(0).max()) <= 2


def _mwpm(circ, rounds: int, shots: int, seed: int) -> dict | None:
    try:
        import pymatching
    except ImportError:
        return None
    try:
        dem = circ.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True, flatten_loops=True)
    except Exception as e:  # hyperedges that do not decompose
        return {"error": f"no decomposition: {type(e).__name__}"}
    matcher = pymatching.Matching.from_detector_error_model(dem)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    pred = matcher.decode_batch(det)
    fails = int(np.any(pred != obs, axis=1).sum())
    pf = fails / shots
    return {"shots": shots, "failures": fails, "p_fail": pf, "p_round": mo.per_round(pf, rounds)}


def evaluate_row(name: str, measured_key: str, code, profile: Profile, connectivity: str, flags: str, note: str,
                 proxy_rounds=(1, 2, 3), shots: int = 20_000, seed: int = 0, mwpm: bool = True) -> dict:
    meas = MEASURED[measured_key]
    rounds = meas["rounds"]
    t0 = time.perf_counter()
    orders = mo.hook_orders(code, None, seed, flags=flags)
    rec = {"experiment": name, "measured_key": measured_key, "profile": profile.name, "timing": profile.timing.name,
           "connectivity": connectivity, "flags": flags, "n": code.n, "k": code.k, "note": note,
           "paper_rounds": rounds, "proxy": {}, "sampled": {}, "mwpm": {}}
    for R in proxy_rounds:
        obj = mo.MemoryObjective(profile, rounds=R, tier="proxy", connectivity=connectivity, flags=flags, with_dcirc=True)
        rec["proxy"][str(R)] = {}
        for basis in mo.BASES:
            pb = obj.evaluate_basis(code, basis, R, "proxy", orders=orders)
            rec["proxy"][str(R)][basis] = {k: pb[k] for k in ("p_round", "p_fail", "expected_faults", "mechanisms", "detectors",
                                                             "two_qubit_gates", "layers", "batches", "num_qubits", "extraction", "d_circ")}
    obj = mo.MemoryObjective(profile, rounds=rounds, tier="tesseract", connectivity=connectivity, flags=flags, shots=shots, seed=seed)
    for basis in mo.BASES:
        mc = obj.circuit(code, basis, rounds, orders=orders)
        circ = mo.noisy_memory_circuit(mc, profile, obj.scale)
        pb = obj.score_basis(mc, mo.memory_dem(mc, profile, obj.scale), "tesseract")
        rec["sampled"][basis] = {"p_round": pb["p_round"], "p_fail": pb["p_fail"], "shots": pb["sampling"]["shots"],
                                 "failures": pb["sampling"]["failures"], "seconds": pb["seconds"]}
        if mwpm:
            if not _matchable(code):
                rec["mwpm"][basis] = {"skipped": NOT_MATCHABLE}
            elif (r := _mwpm(circ, rounds, shots, seed + 1)) is not None:
                rec["mwpm"][basis] = r
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def run(out_dir: Path, rows=None, shots: int = 20_000, seed: int = 0, mwpm: bool = True, force: bool = False, verbose: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, key, code_f, prof_f, conn, flags, note in ROWS:
        if rows and name not in rows:
            continue
        path = out_dir / f"{name}.json"
        if path.exists() and not force:
            paths.append(path)
            continue
        rec = evaluate_row(name, key, code_f(), prof_f(), conn, flags, note, shots=shots, seed=seed, mwpm=mwpm)
        path.write_text(json.dumps(rec, indent=1, default=float))
        paths.append(path)
        if verbose:
            s = rec["sampled"]
            print(f"{name}: Tesseract per round Z {s['Z']['p_round']:.4f} X {s['X']['p_round']:.4f} "
                  f"({s['Z']['failures']} and {s['X']['failures']} failures), {rec['seconds']} s", flush=True)
    return paths


def refresh_mwpm(out_dir: Path, shots: int = 20_000, seed: int = 0, verbose: bool = True) -> list[Path]:
    """Recompute the matching column of every existing record on the same
    circuit, leaving the proxy and Tesseract entries alone."""
    paths = []
    for name, _key, code_f, prof_f, conn, flags, _note in ROWS:
        path = out_dir / f"{name}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        code, profile, rounds = code_f(), prof_f(), rec["paper_rounds"]
        orders = mo.hook_orders(code, None, seed, flags=flags)
        obj = mo.MemoryObjective(profile, rounds=rounds, tier="tesseract", connectivity=conn, flags=flags, shots=shots, seed=seed)
        rec["mwpm"] = {}
        for basis in mo.BASES:
            if not _matchable(code):
                rec["mwpm"][basis] = {"skipped": NOT_MATCHABLE}
                continue
            mc = obj.circuit(code, basis, rounds, orders=orders)
            r = _mwpm(mo.noisy_memory_circuit(mc, profile, obj.scale), rounds, shots, seed + 1)
            if r is not None:
                rec["mwpm"][basis] = r
        path.write_text(json.dumps(rec, indent=1, default=float))
        paths.append(path)
        if verbose:
            print(name, {b: (f"{v['p_round']:.4f}" if "p_round" in v else "skipped") for b, v in rec["mwpm"].items()}, flush=True)
    return paths


def load_all(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


# ------------------------------------------------------------------ report
def _measured(meas: dict, basis: str) -> float | None:
    return meas.get(basis, meas.get("basis_average"))


def write_report(out_dir: Path = DEFAULT_OUT, report: Path = REPORT) -> Path:
    recs = load_all(out_dir)
    order = {name: i for i, (name, *_) in enumerate(ROWS)}
    recs.sort(key=lambda r: order.get(r["experiment"], 99))
    L = [
        "# The evaluator against published memory experiments",
        "",
        "Every number comes from the JSON files under `experiments/hardware_comparison/runs/`; this file regenerates with "
        "`uv run python -m experiments.hardware_comparison.run report`. Code: `experiments/hardware_comparison/run.py`.",
        "",
        "Each row builds the package's own memory circuit for the paper's code, applies a noise profile whose parameters are "
        "read from the paper (assumptions listed per row), and scores it with the proxy tier at one to three rounds and with "
        "Tesseract sampling (and minimum-weight matching for the surface, toric, and heavy-hexagon codes, where a single data "
        "error flips at most two checks of each type) at the paper's round count; per-round values use the memory-decay "
        "convention of the papers. The measured column is the paper's own number "
        "with the paper's decoder. This is a comparison of the package's noise model on the package's circuit with a reported "
        "number, not a prediction about a device: the circuits differ from the devices' (bare or one-flag extraction, no "
        "adaptive protocol, no transport), several parameters are assumptions named below, and leakage is omitted everywhere.",
        "",
        "## Summary",
        "",
        "| row | code | rounds | measured per round | proxy, one round | Tesseract at the paper's rounds | matching | Tesseract over measured |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        meas = MEASURED[r["measured_key"]]
        R = r["paper_rounds"]
        measured = ", ".join(f"{b} {_measured(meas, b):.2e}" if b in meas else "" for b in "ZX" if b in meas) or f"{meas['basis_average']:.2e} (both bases averaged)"
        proxy1 = r["proxy"].get("1", {})
        p1 = ", ".join(f"{b} {proxy1[b]['p_round']:.2e}" for b in "ZX" if b in proxy1)
        tess = ", ".join(f"{b} {r['sampled'][b]['p_round']:.2e} ({r['sampled'][b]['failures']})" for b in "ZX")
        mw = ", ".join(f"{b} {r['mwpm'][b]['p_round']:.2e}" for b in "ZX" if "p_round" in r["mwpm"].get(b, {}))
        mw = mw or ("not a matching code" if any("skipped" in v for v in r["mwpm"].values()) else "n/a")
        ratios = []
        for b in "ZX":
            m = _measured(meas, b)
            if m:
                ratios.append(f"{b} {r['sampled'][b]['p_round'] / m:.2f}")
        L.append(f"| {r['experiment']} | [[{r['n']},{r['k']}]] {r['profile']} | {R} | {measured} | {p1} | {tess} | {mw} | {', '.join(ratios)} |")
    L += ["", "Failure counts in parentheses are Tesseract failures out of the shots per basis; a count below about 50 leaves a standard error above 15 percent.", ""]
    L += [
        "## Sources",
        "",
        "Every measured value and every profile parameter was read from the paper named in the row's profile and checked "
        "against the arXiv text on 2026-10-17; the per-row notes above name the quantities the papers do not state and "
        "the choices made for them. The papers: Google Quantum AI 2023 (arXiv:2207.06431, Nature 614, 676), Google Quantum "
        "AI 2024 (arXiv:2408.13687, Nature 638, 920), Ryan-Anderson et al. 2021 (arXiv:2107.07505, Physical Review X 11, "
        "041058), Tham et al. 2026 (arXiv:2606.06455), and Sundaresan et al. 2023 (arXiv:2203.07205, Nature Communications "
        "14, 2852).",
        "",
        "## Rows",
        "",
    ]
    for r in recs:
        meas = MEASURED[r["measured_key"]]
        L += [f"### {r['experiment']}", "",
              f"Code [[{r['n']},{r['k']}]], profile `{r['profile']}` ({r['timing']} timing, connectivity `{r['connectivity']}`, flags `{r['flags']}`), "
              f"extraction `{r['proxy']['1']['Z']['extraction']}`. Measured: {meas['note']}; decoder {meas['decoder']}. Circuit and model: {r['note']}.", ""]
        L += ["| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |", "|---|---|---|---|---|---|---|"]
        for R, row in r["proxy"].items():
            for b in "ZX":
                v = row[b]
                L.append(f"| {R} | {b} | {v['p_round']:.3e} | {v['expected_faults']:.2f} | {v['mechanisms']} | {v['layers']} | {v['d_circ']} |")
        L.append("")
    report.write_text("\n".join(L))
    return report


LABELS = {
    "google_2023": "Google 2023, d = 3 surface",
    "google_2024_72q": "Google 2024 (72 qubits), d = 3 surface",
    "quantinuum_h1_2021_noidle": "Quantinuum H1 2021, Steane, no idle",
    "quantinuum_h1_2021_idlebound": "Quantinuum H1 2021, Steane, idle bound",
    "ion_chain_toric_18": "Tham et al. 2026, 40-ion chain, toric [[18,2,3]]",
    "ion_chain_bb_18_4_3": "Tham et al. 2026, 40-ion chain, bicycle [[18,4,3]]",
    "ibm_peekskill_2022_flagged": "IBM Peekskill 2022, heavy-hex, flagged gauges",
    "ibm_peekskill_2022_bare": "IBM Peekskill 2022, heavy-hex, bare gauges",
    "ibm_peekskill_2022_medians": "IBM Peekskill 2022, heavy-hex, median parameters",
}


def make_figure(out_dir: Path = DEFAULT_OUT, out_stem: Path | str = FIGURE):
    """One line per row and basis: the paper's per-round number (black)
    and the evaluator's Tesseract number at the paper's round count
    (color), joined by a gray segment; hollow markers are matching on the
    same circuit where the code is a matching code. The Google papers
    report the two bases averaged, drawn at both."""
    from matplotlib.lines import Line2D

    from ..common import figstyle

    plt = figstyle.apply()
    recs = {r["experiment"]: r for r in load_all(out_dir)}
    names = [n for n, *_ in ROWS if n in recs]
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, figstyle.HEIGHT_DOUBLE))
    color = {"Z": figstyle.SERIES_COLORS[0], "X": figstyle.SERIES_COLORS[1]}
    marker = {"Z": "o", "X": "s"}
    offset = {"Z": 0.18, "X": -0.18}
    for y, name in enumerate(names):
        r = recs[name]
        meas = MEASURED[r["measured_key"]]
        for b in "ZX":
            m, ev = _measured(meas, b), r["sampled"][b]["p_round"]
            yy = y + offset[b]
            ax.plot([m, ev], [yy, yy], "-", color=figstyle.NEUTRAL, linewidth=figstyle.THIN, zorder=1)
            ax.plot([m], [yy], marker[b], color="black", zorder=3)
            err = ev / np.sqrt(max(r["sampled"][b]["failures"], 1))
            ax.errorbar([ev], [yy], xerr=[err], fmt=marker[b], color=color[b], capsize=1.5, linewidth=figstyle.THIN, zorder=3)
            mw = r["mwpm"].get(b, {})
            if "p_round" in mw:
                ax.plot([mw["p_round"]], [yy], marker[b], color=color[b], markerfacecolor="none", zorder=2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("logical error per round")
    ax.grid(axis="y", visible=False)
    handles = [
        Line2D([], [], marker="o", color="black", ls="none", label="paper's number, its decoder (circle Z, square X)"),
        Line2D([], [], marker="o", color=color["Z"], ls="none", label="evaluator, Tesseract, Z basis"),
        Line2D([], [], marker="s", color=color["X"], ls="none", label="evaluator, Tesseract, X basis"),
        Line2D([], [], marker="o", color=figstyle.NEUTRAL, markerfacecolor="none", ls="none", label="evaluator, matching, same circuit"),
    ]
    ax.legend(handles=handles, loc="lower left")
    fig.tight_layout()
    return figstyle.save(fig, out_stem)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rows", nargs="*", default=None)
    r.add_argument("--shots", type=int, default=20_000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-mwpm", action="store_true")
    r.add_argument("--force", action="store_true")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    f = sub.add_parser("figure")
    f.add_argument("--out", default=str(DEFAULT_OUT))
    f.add_argument("--stem", default=str(FIGURE))
    m = sub.add_parser("mwpm", help="recompute the matching column of the existing records")
    m.add_argument("--shots", type=int, default=20_000)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "run":
        run(out, a.rows, a.shots, a.seed, not a.no_mwpm, a.force)
        print("wrote", write_report(out))
    elif a.cmd == "report":
        print("wrote", write_report(out))
    elif a.cmd == "mwpm":
        refresh_mwpm(out, a.shots, a.seed)
        print("wrote", write_report(out))
    else:
        print("wrote", *make_figure(out, a.stem))


if __name__ == "__main__":
    main()
