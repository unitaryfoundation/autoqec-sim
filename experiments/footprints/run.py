"""Letter assignments on two footprints that are not surface codes.

The SAT-population experiment (:mod:`experiments.sat_population`) found
that on the 3x3 rotated-surface footprint every commuting independent
assignment of Pauli letters to the stabilizer supports is a [[9,1,3]]
code with the same 24 weight-3 logicals, that the circuit-level rate
still spreads by 2.6x, that about half the random assignments beat XZZX
under the Forte preset, and that the two-basis mass of the minimum-weight
undetectable fault sets predicts the rate. This module asks the same
questions on two footprints that are not surface codes: the three
weight-4 faces of the Steane code and the eight faces of the [[17,1,5]]
triangular 4.8.8 color code, with two generators per face.

Sampling. CMSGen (``pycmsgen``) samples models of a CNF with one Pauli
string per generator: zero outside its face, a nonzero letter on every
face qubit, the two generators of a face distinct, and every pair of
generators commuting (Tseitin product variables and one XOR clause per
pair). Dependent generating sets are rejected and codes are deduplicated
by stabilizer group, so the population is one of groups, not of
generating sets. No distance clause is imposed: the distance distribution
is one of the results.

Scoring. The exact distance and the number of minimum-weight logical
operators come from the weight enumerators. The proxy tier with the exact
circuit distance, and the count and mass of the minimum-weight
undetectable fault sets, under two cases: the Forte preset (sequential
timing, scale 0.25, a noise model assembled from public specifications,
not a device measurement) and a depolarizing profile with a biased idle
channel (bias 5, parallel timing); rounds equal to the footprint's
distance, both memory bases. The reference on each footprint is the CSS
code. Confirmation: Tesseract sampling under the Forte case for the
reference and the best full-distance codes.

    uv run --with pycmsgen python -m experiments.footprints.run run --footprint steane --count 300
    uv run --with pycmsgen python -m experiments.footprints.run run --footprint color488 --count 120 --workers 4
    uv run --with pycmsgen python -m experiments.footprints.run run --footprint steane --count 300 --extraction batched
    uv run python -m experiments.footprints.run confirm --footprint steane --top 5
    uv run python -m experiments.footprints.run confirm --footprint color488 --top 4
    uv run python -m experiments.footprints.run report
    uv run python -m experiments.footprints.run figure
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from autoqec_sim import known_codes
from autoqec_sim.code import StabilizerCode
from autoqec_sim.memory_objective import MemoryObjective, code_key

from ..memory_objective.run import DEPOLARIZING_SCALE, PROFILE_SCALES, biased_idle_profile, spearman

DEFAULT_OUT = Path("experiments/footprints/runs")
REPORT = Path("experiments/footprints/REPORT.md")
FIGURE = Path("docs/assets/figures/footprints")

# The [[17,1,5]] triangular 4.8.8 color code on 17 qubits: one octagon,
# three squares, and four weight-4 faces cut by the boundary. Faces overlap
# in zero or two qubits, the three corner qubits lie in one face, nine
# boundary qubits in two, and five bulk qubits in three. The CSS code on
# these faces has parameters (17, 1, 5) with 153 weight-5 logicals
# (checked in experiments/tests/test_footprints.py).
COLOR488_FACES = [
    [2, 3, 5, 7, 9, 11, 13, 14],
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [9, 10, 11, 12],
    [0, 2, 4, 5],
    [6, 7, 13, 15],
    [8, 10, 12, 16],
    [11, 12, 14, 16],
]

CASES = ("forte", "idle_eta_5")
CASE_TITLES = {"forte": "Forte preset, sequential, scale 0.25", "idle_eta_5": "biased idle, bias 5, parallel"}


def css_strings(n: int, faces: list[list[int]]) -> list[str]:
    out = []
    for letter in "XZ":
        for face in faces:
            row = ["I"] * n
            for q in face:
                row[q] = letter
            out.append("".join(row))
    return out


def footprint(name: str) -> dict:
    """``n``, ``faces``, ``rounds`` (the CSS distance), and the CSS
    reference strings of a footprint."""
    if name == "steane":
        x_type = [s for s in known_codes.STEANE if set(s) <= {"I", "X"}]
        faces = [[q for q, ch in enumerate(s) if ch != "I"] for s in x_type]
        return {"name": name, "n": 7, "faces": faces, "rounds": 3, "reference": css_strings(7, faces), "title": "Steane [[7,1,3]] footprint"}
    if name == "color488":
        return {"name": name, "n": 17, "faces": COLOR488_FACES, "rounds": 5, "reference": css_strings(17, COLOR488_FACES), "title": "4.8.8 color [[17,1,5]] footprint"}
    raise ValueError(name)


FOOTPRINTS = ("steane", "color488")


# ---------------------------------------------------------------- sampling
class _Cnf:
    def __init__(self):
        self.num_vars = 0
        self.clauses: list[list[int]] = []
        self.xors: list[tuple[list[int], bool]] = []

    def var(self) -> int:
        self.num_vars += 1
        return self.num_vars

    def product(self, a: int, b: int) -> int:
        """A fresh variable equal to ``a AND b``."""
        p = self.var()
        self.clauses += [[-p, a], [-p, b], [p, -a, -b]]
        return p


def build_cnf(n: int, faces: list[list[int]]) -> tuple[_Cnf, list[tuple[list[int], list[int]]]]:
    """Two Pauli strings per face, zero outside the face and nonzero on it,
    distinct within a face, all pairs commuting."""
    cnf = _Cnf()
    gens: list[tuple[list[int], list[int]]] = []
    for face in faces:
        fs = set(face)
        pair = []
        for _ in range(2):
            xs, zs = [cnf.var() for _ in range(n)], [cnf.var() for _ in range(n)]
            for q in range(n):
                if q in fs:
                    cnf.clauses.append([xs[q], zs[q]])
                else:
                    cnf.clauses += [[-xs[q]], [-zs[q]]]
            pair.append((xs, zs))
        (ax, az), (bx, bz) = pair
        diffs = []
        for q in face:
            for a, b in ((ax[q], bx[q]), (az[q], bz[q])):
                d = cnf.var()
                cnf.xors.append(([d, a, b], False))  # d = a xor b
                diffs.append(d)
        cnf.clauses.append(diffs)  # the two generators differ somewhere
        gens.extend(pair)
    supports = [set(faces[i // 2]) for i in range(len(gens))]
    for i in range(len(gens)):
        for j in range(i):
            shared = supports[i] & supports[j]
            if not shared:
                continue
            (xi, zi), (xj, zj) = gens[i], gens[j]
            prods = [cnf.product(a, b) for q in sorted(shared) for a, b in ((xi[q], zj[q]), (zi[q], xj[q]))]
            cnf.xors.append((prods, False))  # symplectic product zero
    return cnf, gens


def _strings_from_model(model, gens, n: int) -> list[str]:
    letters = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    return ["".join(letters[int(model[xs[q]]), int(model[zs[q]])] for q in range(n)) for xs, zs in gens]


def sample(fp: dict, seed: int, count: int, known: set[bytes], log=print, max_draws: int | None = None) -> list[tuple[str, list[str]]]:
    """``count`` distinct stabilizer groups (keys not in ``known``) from
    CMSGen models of the footprint CNF."""
    from pycmsgen import Solver

    n, faces = fp["n"], fp["faces"]
    cnf, gens = build_cnf(n, faces)
    solver = Solver(seed=seed)
    for c in cnf.clauses:
        solver.add_clause(c)
    for v, rhs in cnf.xors:
        solver.add_xor_clause(v, rhs)
    out: list[tuple[str, list[str]]] = []
    seen = set(known)
    draws = rejected = dup = wrong_k = 0
    cap = max_draws or (50 * count + 1000)
    t0 = time.perf_counter()
    while len(out) < count and draws < cap:
        ok, model = solver.solve()
        if not ok:
            log("unsat")
            break
        draws += 1
        stabs = _strings_from_model(model, gens, n)
        try:
            code = StabilizerCode.from_stabilizers(stabs)
        except ValueError:
            rejected += 1
            continue
        if code.k != 1:
            wrong_k += 1
            continue
        key = code_key(code)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append((f"S{seed}_{len(known) + len(out)}", stabs))
    log(f"{fp['name']}: {len(out)} new distinct groups from {draws} draws ({rejected} dependent, {wrong_k} wrong k, {dup} duplicate) "
        f"in {time.perf_counter() - t0:.0f} s; cnf {cnf.num_vars} vars, {len(cnf.clauses)} clauses, {len(cnf.xors)} xors")
    return out


# ----------------------------------------------------------------- scoring
def distance_and_count(code: StabilizerCode) -> tuple[int, int]:
    """Exact distance and the number of minimum-weight logical operators
    (elements of the normalizer outside the stabilizer group), from the
    weight enumerators."""
    a, b = code.weight_enumerators()
    diff = b - a
    d = int(np.nonzero(diff[1:])[0][0] + 1)
    return d, int(diff[d])


def make_cases(rounds: int, extraction: str = "auto") -> dict[str, MemoryObjective]:
    from autoqec_sim.noise import presets

    kw = {"rounds": rounds, "tier": "proxy", "with_dcirc": True, "extraction": extraction}
    return {
        "forte": MemoryObjective(presets.forte("sequential"), scale=PROFILE_SCALES["forte"], **kw),
        "idle_eta_5": MemoryObjective(biased_idle_profile(5.0, "parallel"), scale=DEPOLARIZING_SCALE, **kw),
    }


_CASES: dict[str, MemoryObjective] = {}


def _init(rounds: int, extraction: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _CASES.update(make_cases(rounds, extraction))


_BASIS_KEYS = ("p_round", "p_fail", "d_circ", "d_circ_count", "d_circ_mass", "mechanisms", "expected_faults", "layers", "extraction", "proxy_weight")


def score(item: tuple[str, list[str]]) -> str:
    tag, stabs = item
    code = StabilizerCode.from_stabilizers(stabs)
    t0 = time.perf_counter()
    d, count = distance_and_count(code)
    rec = {"id": tag, "stabilizers": stabs, "n": code.n, "k": code.k, "d": d, "min_weight_logicals": count, "css": code.is_css(),
           "distance_s": round(time.perf_counter() - t0, 2), "cases": {}}
    for name, obj in _CASES.items():
        t0 = time.perf_counter()
        res = obj.evaluate(code, use_cache=False)
        per = {b: {k: res.per_basis[b].get(k) for k in _BASIS_KEYS} for b in "ZX"}
        rec["cases"][name] = {
            "p_round": res.p_round,
            "mass": sum((per[b]["d_circ_mass"] or 0.0) for b in "ZX"),
            "count": sum((per[b]["d_circ_count"] or 0) for b in "ZX"),
            "per_basis": per,
            "wall_s": round(time.perf_counter() - t0, 1),
        }
    return json.dumps(rec, default=float)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def records_path(out_dir: Path, name: str, extraction: str = "auto") -> Path:
    return out_dir / (f"{name}.jsonl" if extraction == "auto" else f"{name}_{extraction}.jsonl")


def run(out_dir: Path, name: str, seed: int, count: int, workers: int, extraction: str = "auto", verbose: bool = True) -> Path:
    """Sample and score; ``extraction="batched"`` scores the same sample
    (same seed) with every code measured in batches, as the CSS Steane
    code is, for a like-for-like comparison."""
    fp = footprint(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = records_path(out_dir, name, extraction)
    have = load_records(path)
    known = {code_key(StabilizerCode.from_stabilizers(r["stabilizers"])) for r in have}
    log = print if verbose else (lambda *_: None)
    items: list[tuple[str, list[str]]] = []
    if not any(r["id"] == "css" for r in have):
        items.append(("css", fp["reference"]))
    need = count - sum(r["id"] != "css" for r in have)
    if need > 0:
        items += sample(fp, seed, need, known, log)
    if not items:
        log(f"{name}: {len(have)} records already present")
        return path
    log(f"{name}: n={fp['n']} faces={len(fp['faces'])} rounds={fp['rounds']} extraction={extraction} scoring {len(items)} codes on {workers} workers")
    t0 = time.perf_counter()
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(fp["rounds"], extraction)) as pool, open(path, "a") as fh:
        for i, line in enumerate(pool.imap(score, items, chunksize=1)):
            fh.write(line + "\n")
            fh.flush()
            r = json.loads(line)
            if i < 3 or (i + 1) % 20 == 0 or i == len(items) - 1:
                c = r["cases"]["forte"]
                z, x = c["per_basis"]["Z"], c["per_basis"]["X"]
                log(f"{i + 1}/{len(items)} {r['id']} d={r['d']} logicals={r['min_weight_logicals']} forte {c['p_round']:.3e} "
                    f"d_circ {z['d_circ']}/{x['d_circ']} {z['extraction']} {c['wall_s']} s ({time.perf_counter() - t0:.0f} s)")
    return path


# ------------------------------------------------------------ confirmation
def confirm(out_dir: Path, name: str, top: int, shots: int, seed: int, spread: bool = True, verbose: bool = True) -> Path:
    """Tesseract under the Forte case for the CSS reference and the ``top``
    full-distance codes by the Forte proxy; with ``spread``, also the
    median and the worst full-distance code and the best distance-3 code,
    so the check spans the range of the proxy."""
    from autoqec_sim.noise import presets

    fp = footprint(name)
    recs = load_records(out_dir / f"{name}.jsonl")
    rate = lambda r: r["cases"]["forte"]["p_round"]  # noqa: E731
    full = sorted((r for r in recs if r["d"] == fp["rounds"] and r["id"] != "css"), key=rate)
    chosen = [r for r in recs if r["id"] == "css"] + full[:top]
    if spread and full:
        d3 = sorted((r for r in recs if r["d"] == 3 and r["d"] != fp["rounds"]), key=rate)
        for r in [full[len(full) // 2], full[-1], *d3[:1]]:
            if r["id"] not in {c["id"] for c in chosen}:
                chosen.append(r)
    path = out_dir / f"{name}_tesseract.jsonl"
    done = {r["id"] for r in load_records(path)}
    obj = MemoryObjective(presets.forte("sequential"), scale=PROFILE_SCALES["forte"], rounds=fp["rounds"], tier="tesseract", shots=shots, seed=seed)
    with open(path, "a") as fh:
        for r in chosen:
            if r["id"] in done:
                continue
            code = StabilizerCode.from_stabilizers(r["stabilizers"])
            t0 = time.perf_counter()
            res = obj.evaluate(code, use_cache=False)
            per = {b: {"p_round": res.per_basis[b]["p_round"], "p_fail": res.per_basis[b]["p_fail"],
                       "shots": res.per_basis[b]["sampling"]["shots"], "failures": res.per_basis[b]["sampling"]["failures"]} for b in "ZX"}
            rec = {"id": r["id"], "d": r["d"], "proxy_p_round": r["cases"]["forte"]["p_round"], "tesseract_p_round": res.p_round, "per_basis": per,
                   "shots": shots, "seconds": round(time.perf_counter() - t0, 1)}
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            if verbose:
                print(f"{name} {r['id']}: proxy {rec['proxy_p_round']:.3e} tesseract {res.p_round:.3e} "
                      f"({per['Z']['failures']} and {per['X']['failures']} failures) {rec['seconds']} s", flush=True)
    return path


# ----------------------------------------------------------------- analysis
def corner_clash(fp: dict, stabs: list[str]) -> bool:
    """True when some qubit lying in a single face carries the same letter
    in both generators of that face: the single-qubit Pauli of that letter
    then commutes with every generator and the distance is 1."""
    faces = fp["faces"]
    multiplicity = {q: sum(q in f for f in faces) for q in range(fp["n"])}
    for i, face in enumerate(faces):
        a, b = stabs[2 * i], stabs[2 * i + 1]
        if any(multiplicity[q] == 1 and a[q] == b[q] for q in face):
            return True
    return False


def analyze(name: str, recs: list[dict], conf: list[dict]) -> dict:
    fp = footprint(name)
    d_full = fp["rounds"]
    ref = next(r for r in recs if r["id"] == "css")
    samples = [r for r in recs if r["id"] != "css"]
    by_d: dict[int, int] = {}
    for r in samples:
        by_d[r["d"]] = by_d.get(r["d"], 0) + 1
    full = [r for r in samples if r["d"] == d_full]
    atleast3 = [r for r in samples if r["d"] >= 3]
    classes: dict[int, int] = {}
    for r in full:
        classes[r["min_weight_logicals"]] = classes.get(r["min_weight_logicals"], 0) + 1
    d1 = [r for r in samples if r["d"] == 1]
    out = {"n": fp["n"], "faces": len(fp["faces"]), "rounds": d_full, "samples": len(samples), "by_distance": dict(sorted(by_d.items())),
           "full": len(full), "classes": dict(sorted(classes.items())), "reference": {"d": ref["d"], "min_weight_logicals": ref["min_weight_logicals"]},
           "corner_clash": sum(corner_clash(fp, r["stabilizers"]) for r in d1), "cases": {}}
    for case in CASES:
        rates = np.array([r["cases"][case]["p_round"] for r in full])
        ref_rate = ref["cases"][case]["p_round"]
        rz, rx = ref["cases"][case]["per_basis"]["Z"], ref["cases"][case]["per_basis"]["X"]
        c = {"reference": ref_rate, "reference_Z": rz["p_round"], "reference_X": rx["p_round"],
             "reference_d_circ": (rz["d_circ"], rx["d_circ"]), "reference_extraction": rz["extraction"]}
        if len(full):
            masses = np.array([r["cases"][case]["mass"] for r in full])
            c.update({"min": float(rates.min()), "median": float(np.median(rates)), "max": float(rates.max()), "spread": float(rates.max() / rates.min()),
                      "best_over_reference": float(rates.min() / ref_rate), "below_reference": int((rates < ref_rate).sum()),
                      "d_circ": sorted({(r["cases"][case]["per_basis"]["Z"]["d_circ"], r["cases"][case]["per_basis"]["X"]["d_circ"]) for r in full}),
                      "extraction": sorted({r["cases"][case]["per_basis"]["Z"]["extraction"] for r in full}),
                      "spearman_mass": spearman(masses, rates) if len(full) > 2 else float("nan"),
                      "expected_faults": float(np.median([r["cases"][case]["per_basis"]["Z"]["expected_faults"] for r in full])),
                      "wall_s": float(np.median([r["cases"][case]["wall_s"] for r in full]))})
            best = full[int(rates.argmin())]
            c["best"] = {"id": best["id"], "p_round": float(rates.min()), "Z": best["cases"][case]["per_basis"]["Z"]["p_round"],
                         "X": best["cases"][case]["per_basis"]["X"]["p_round"], "stabilizers": best["stabilizers"]}
        if len(atleast3) > 2:
            c["spearman_mass_d_ge_3"] = spearman([r["cases"][case]["mass"] for r in atleast3], [r["cases"][case]["p_round"] for r in atleast3])
        out["cases"][case] = c
    if len(full) > 2:
        out["spearman_between_cases"] = spearman([r["cases"]["forte"]["p_round"] for r in full], [r["cases"]["idle_eta_5"]["p_round"] for r in full])
    if conf:
        ratios = [r["tesseract_p_round"] / r["proxy_p_round"] for r in conf if r["proxy_p_round"] > 0]
        out["confirm"] = {"codes": len(conf), "shots": conf[0]["shots"], "ratio_median": float(np.median(ratios)), "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios)),
                          "spearman": spearman([r["proxy_p_round"] for r in conf], [r["tesseract_p_round"] for r in conf]) if len(conf) > 2 else float("nan"),
                          "failures_median": float(np.median([r["per_basis"]["Z"]["failures"] + r["per_basis"]["X"]["failures"] for r in conf])),
                          "rows": conf}
    return out


def _fmt(x) -> str:
    return "" if x is None else (f"{x:.2e}" if isinstance(x, float) and (abs(x) < 1e-2 or abs(x) >= 1e4) else (f"{x:.2f}" if isinstance(x, float) else str(x)))


def write_report(out_dir: Path = DEFAULT_OUT, report: Path = REPORT) -> Path:
    L = [
        "# Letter assignments on the Steane and 4.8.8 footprints",
        "",
        "Every number comes from the JSON lines under `experiments/footprints/runs/`; this file regenerates with "
        "`uv run python -m experiments.footprints.run report`. Code: `experiments/footprints/run.py`.",
        "",
        "Each footprint is a set of faces; a code is an assignment of two Pauli strings per face, nonzero on the face, zero "
        "outside it, distinct within the face, with every pair commuting, sampled by CMSGen and deduplicated by stabilizer "
        "group. The reference is the CSS code on the faces. Scores are the proxy tier with the exact circuit distance under the "
        "Forte preset (sequential timing, scale 0.25, a model assembled from public specifications, not a device measurement) and "
        "a depolarizing profile with a biased idle channel (bias 5, parallel timing), with rounds equal to the footprint's "
        "distance and both memory bases. The confirmation is Tesseract sampling under the Forte case.",
        "",
    ]
    for name, extraction in [(n, e) for n in FOOTPRINTS for e in ("auto", "batched")]:
        recs = load_records(records_path(out_dir, name, extraction))
        if not recs:
            continue
        conf = load_records(out_dir / f"{name}_tesseract.jsonl") if extraction == "auto" else []
        a = analyze(name, recs, conf)
        fp = footprint(name)
        title = fp["title"] + ("" if extraction == "auto" else ", every code measured in batches")
        L += [f"## {title}", "",
              f"{a['samples']} sampled groups on {a['faces']} faces, {a['rounds']} rounds. Distance distribution: "
              + ", ".join(f"d = {d}: {c}" for d, c in a["by_distance"].items())
              + f" ({a['corner_clash']} of the d = 1 codes have the same letter in both generators of a face on a qubit of that face only)"
              + f". Full-distance codes: {a['full']}, with "
              + ", ".join(f"{c} having {k} weight-{a['rounds']} logicals" for k, c in a["classes"].items())
              + f" (the CSS reference has {a['reference']['min_weight_logicals']}).", ""]
        L += ["| full-distance codes | " + " | ".join(CASE_TITLES[c] for c in CASES) + " |", "|---|---|---|"]
        rows = [
            ("CSS reference p_round (Z, X)", lambda c: f"{c['reference']:.3e} ({c['reference_Z']:.2e}, {c['reference_X']:.2e})"),
            ("CSS reference d_circ (Z, X), extraction", lambda c: f"{c['reference_d_circ']}, {c['reference_extraction']}"),
            ("random min / median / max", lambda c: f"{c['min']:.3e} / {c['median']:.3e} / {c['max']:.3e}" if "min" in c else ""),
            ("spread max / min", lambda c: _fmt(c.get("spread"))),
            ("best / CSS", lambda c: _fmt(c.get("best_over_reference"))),
            ("codes below CSS", lambda c: f"{c['below_reference']} of {a['full']}" if "below_reference" in c else ""),
            ("d_circ (Z, X) over the codes", lambda c: ", ".join(str(t) for t in c.get("d_circ", []))),
            ("extraction", lambda c: ", ".join(c.get("extraction", []))),
            ("Spearman, two-basis minimum-weight fault mass against p_round", lambda c: _fmt(c.get("spearman_mass"))),
            ("the same over every code with d >= 3", lambda c: _fmt(c.get("spearman_mass_d_ge_3"))),
            ("expected faults per shot, median", lambda c: _fmt(c.get("expected_faults"))),
            ("seconds per evaluation (both bases), median", lambda c: _fmt(c.get("wall_s"))),
        ]
        for label, f in rows:
            L.append(f"| {label} | " + " | ".join(f(a["cases"][c]) for c in CASES) + " |")
        L.append("")
        if "spearman_between_cases" in a:
            L += [f"Spearman between the two cases over the full-distance codes: {a['spearman_between_cases']:.2f}.", ""]
        for case in CASES:
            b = a["cases"][case].get("best")
            if b:
                L += [f"Best full-distance code, {CASE_TITLES[case]} ({b['id']}, p_round {b['p_round']:.3e}, Z {b['Z']:.2e}, X {b['X']:.2e}):", "",
                      "    " + "  ".join(b["stabilizers"]), ""]
        if "confirm" in a:
            k = a["confirm"]
            L += [f"Tesseract confirmation, Forte case, {k['shots']:,} shots per basis, {k['codes']} codes: Tesseract over proxy median "
                  f"{k['ratio_median']:.2f} (range {k['ratio_min']:.2f} to {k['ratio_max']:.2f}), Spearman {k['spearman']:.2f}, "
                  f"{k['failures_median']:.0f} failures per code (median).", "",
                  "| code | d | proxy p_round | Tesseract p_round | ratio | failures (Z, X) |", "|---|---|---|---|---|---|"]
            for r in sorted(k["rows"], key=lambda r: r["proxy_p_round"]):
                L.append(f"| {r['id']} | {r.get('d', '')} | {r['proxy_p_round']:.3e} | {r['tesseract_p_round']:.3e} | {r['tesseract_p_round'] / r['proxy_p_round']:.2f} | "
                         f"{r['per_basis']['Z']['failures']}, {r['per_basis']['X']['failures']} |")
            L.append("")
    report.write_text("\n".join(L))
    return report


# ------------------------------------------------------------------- figure
def make_figure(out_dir: Path = DEFAULT_OUT, out_stem: Path | str = FIGURE):
    """One panel per footprint: the Forte proxy rate of every sampled code
    with d >= 3, ranked, colored by distance, the CSS reference as a line,
    and the Tesseract confirmation as hollow markers."""
    from ..common import figstyle

    plt = figstyle.apply()
    panels = [(name, load_records(out_dir / f"{name}.jsonl"), load_records(out_dir / f"{name}_tesseract.jsonl")) for name in FOOTPRINTS]
    panels = [p for p in panels if p[1]]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE + 0.3), squeeze=False)
    for ax, letter, (name, recs, conf) in zip(axes[0], "abc", panels, strict=False):
        fp = footprint(name)
        ref = next(r for r in recs if r["id"] == "css")
        samples = sorted((r for r in recs if r["id"] != "css" and r["d"] >= 3), key=lambda r: r["cases"]["forte"]["p_round"])
        rank = {r["id"]: i + 1 for i, r in enumerate(samples)}
        for d, color in ((fp["rounds"], figstyle.SERIES_COLORS[0]), (3, figstyle.SERIES_COLORS[1])):
            pts = [(rank[r["id"]], r["cases"]["forte"]["p_round"]) for r in samples if r["d"] == d]
            if pts and not (d == 3 and fp["rounds"] == 3 and color == figstyle.SERIES_COLORS[1]):
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=color, label=f"proxy, d = {d} ({len(pts)})")
        ax.axhline(ref["cases"]["forte"]["p_round"], color=figstyle.NEUTRAL, linestyle="--", linewidth=figstyle.LINE, label="CSS reference, proxy")
        for r in conf:
            x = 0 if r["id"] == "css" else rank.get(r["id"])
            if x is None:
                continue
            fails = max(r["per_basis"]["Z"]["failures"] + r["per_basis"]["X"]["failures"], 1)
            y = r["tesseract_p_round"]
            ax.errorbar([x], [y], yerr=[y / np.sqrt(fails)], fmt="o", color="black", markerfacecolor="none", capsize=1.5, linewidth=figstyle.THIN)
        if conf:
            ax.plot([], [], "o", color="black", markerfacecolor="none", label="Tesseract (rank 0: CSS)")
        ax.set_yscale("log")
        ax.set_xlabel("code, ranked by the Forte proxy rate")
        ax.set_ylabel("logical error per round, Forte case")
        ax.set_title(fp["title"])
        ax.legend(loc="lower right")
        figstyle.panel_label(ax, letter)
    fig.tight_layout()
    return figstyle.save(fig, out_stem)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--footprint", choices=FOOTPRINTS, required=True)
    r.add_argument("--count", type=int, default=300)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--extraction", choices=("auto", "batched"), default="auto")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--quiet", action="store_true")
    c = sub.add_parser("confirm")
    c.add_argument("--footprint", choices=FOOTPRINTS, required=True)
    c.add_argument("--top", type=int, default=5)
    c.add_argument("--no-spread", action="store_true", help="only the reference and the top codes")
    c.add_argument("--shots", type=int, default=50_000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default=str(DEFAULT_OUT))
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    f = sub.add_parser("figure")
    f.add_argument("--out", default=str(DEFAULT_OUT))
    f.add_argument("--stem", default=str(FIGURE))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "run":
        run(out, a.footprint, a.seed, a.count, a.workers, a.extraction, not a.quiet)
        print("wrote", write_report(out))
    elif a.cmd == "confirm":
        confirm(out, a.footprint, a.top, a.shots, a.seed, not a.no_spread)
        print("wrote", write_report(out))
    elif a.cmd == "report":
        print("wrote", write_report(out))
    else:
        print("wrote", *make_figure(out, a.stem))


if __name__ == "__main__":
    main()
