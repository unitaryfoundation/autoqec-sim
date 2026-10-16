"""Circuit-level scores for a population of SAT-sampled codes.

DalFavero, Watkins, LaBorde, Russo, Egger, Quiroz, and LaRose
(arXiv:2608.23460; github.com/bdalfavero/satisfying-codes) sample 100,000
9-qubit codes on the stabilizer footprint of the rotated surface code by
SAT, each Pauli letter free, and rank them by the exact maximum-likelihood
logical error rate under code-capacity biased noise (``p_X = eta P``,
``p_Y = p_Z = P``). The pilot in :mod:`autoqec_sim.experiments.sat_codes`
scored fifteen of them. This module scores a population: a uniform random
sample of the distinct codes plus the best and the worst by the source's
rate at ``eta = 10``, so that the questions are statistical ones.

* How much of the circuit-level ranking does the code-capacity ranking
  predict (rank correlation, overlap of the best sets)?
* Do the codes the source ranks best score better than random ones once
  the syndrome-extraction circuit and its noise are included?
* Which structural feature of a code predicts its circuit-level rate: the
  source rate, or the batch structure of its extraction circuit (groups
  of generators whose controlled gates cannot interleave), its depth, or
  its minimum-weight circuit faults?
* Which code of the population is best under a device preset, and does
  it beat the rotated surface code and its XZZX deformation?

Each code is Hadamard-conjugated on every qubit so its X tailoring becomes
the Z tailoring autoqec-sim's bias parameter expresses, then scored by
:class:`autoqec_sim.memory_objective.MemoryObjective` over ``R`` rounds at
the proxy and sampled tiers under four cases: a depolarizing profile with
a biased idle channel at source ``eta`` 4 and 10 (the pilot's cases), the
same profile with every channel biased at source ``eta = 10``, and the
IonQ Forte preset. Exact distances are recomputed.

    python -m autoqec_sim.experiments.sat_population prepare --source <satisfying-codes>/experiments/surface
    python -m autoqec_sim.experiments.sat_population run --workers 4
    python -m autoqec_sim.experiments.sat_population report
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import platform
import random
import time
from pathlib import Path

from ..code import StabilizerCode
from ..known_codes import get as known
from ..memory_objective import code_key, rref
from .memory_objective import (
    DEPOLARIZING_SCALE,
    PROFILE_SCALES,
    biased_idle_profile,
    evaluate_tiers,
    hadamard_conjugate,
    spearman,
    xzzx_9,
)

DEFAULT_OUT = Path("experiments/sat_population/runs")
POPULATION = Path("experiments/sat_population/population.json")
FIGURE = Path("docs/assets/figures/sat_population")
N = 9
RANK_ETA = 10.0  # the source's eta that defines "best" and "worst"
CASES = ("idle_eta_4", "idle_eta_10", "all_eta_10", "forte")
CASE_TITLES = {
    "idle_eta_4": "biased idle, source eta = 4",
    "idle_eta_10": "biased idle, source eta = 10",
    "all_eta_10": "every channel biased, source eta = 10",
    "forte": "IonQ Forte, sequential",
}
CASE_SOURCE_ETA = {"idle_eta_4": 4.0, "idle_eta_10": 10.0, "all_eta_10": 10.0, "forte": 1.0}
PROXY_KEYS = ("p_round", "layers", "batches", "two_qubit_gates", "d_circ", "d_circ_mass", "mechanisms")
SAMPLED_KEYS = ("p_round", "decoder")


def _strings(stabs: list[str]) -> list[str]:
    return [s.lstrip("+-").replace("_", "I") for s in stabs]


def make_case(name: str):
    """The profile and rate scale of a case; built where it is used so
    worker processes need not unpickle a profile."""
    from ..noise import presets

    if name == "forte":
        return presets.forte("sequential"), PROFILE_SCALES["forte"]
    eta = float(name.rsplit("_", 1)[1]) / 2  # the source's eta is twice autoqec's bias parameter
    prof = biased_idle_profile(eta, "parallel")
    if name.startswith("all_"):
        prof = prof.with_bias(eta)
    return prof, DEPOLARIZING_SCALE


# ----------------------------------------------------------------------
# prepare: the population file
# ----------------------------------------------------------------------
def read_source(source: Path):
    codes = json.loads((source / "surface_codes.json").read_text())
    ler: dict[int, dict[float, float]] = {}
    for f in sorted(glob.glob(str(source / "data" / "output_*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                ler.setdefault(int(r["code"][1:]), {})[float(r["eta"])] = float(r["ler"])
    ref: dict[str, dict[float, float]] = {}
    with open(source / "bias_sampled_lers.csv") as fh:
        for r in csv.DictReader(fh):
            ref.setdefault(r["code"], {})[float(r["eta"])] = float(r["ler"])
    return codes, ler, ref


def distinct_groups(codes: list[list[str]]) -> dict[bytes, list[int]]:
    """Source indices grouped by stabilizer group (signs dropped)."""
    by_strings: dict[tuple, list[int]] = {}
    for i, stabs in enumerate(codes):
        by_strings.setdefault(tuple(_strings(stabs)), []).append(i)
    groups: dict[bytes, list[int]] = {}
    for strs, ids in by_strings.items():
        key = code_key(StabilizerCode.from_stabilizers(list(strs)))
        groups.setdefault(key, []).extend(ids)
    return groups


def prepare(source: Path, path: Path, n_random: int, n_top: int, n_bottom: int, seed: int) -> dict:
    codes, ler, ref = read_source(source)
    groups = distinct_groups(codes)
    reps = sorted(min(ids) for ids in groups.values())
    dup_count = {min(ids): len(ids) for ids in groups.values()}
    ranked = sorted(reps, key=lambda i: ler[i][RANK_ETA])
    rank = {i: r + 1 for r, i in enumerate(ranked)}
    chosen: dict[int, list[str]] = {}
    for i in random.Random(seed).sample(reps, n_random):
        chosen.setdefault(i, []).append("random")
    for i in ranked[:n_top]:
        chosen.setdefault(i, []).append("top")
    for i in ranked[-n_bottom:]:
        chosen.setdefault(i, []).append("bottom")
    entries = []
    for i in sorted(chosen):
        entries.append({
            "id": f"D{i}", "groups": chosen[i], "source_rank": rank[i], "duplicates": dup_count[i],
            "source_ler": {f"{e:g}": v for e, v in sorted(ler[i].items())},
            "stabilizers": hadamard_conjugate(_strings(codes[i]), range(N)),
        })
    entries.append({"id": "rotated_surface", "groups": ["reference"], "source_rank": None, "duplicates": None,
                    "source_ler": {f"{e:g}": v for e, v in sorted(ref["CSS"].items())},
                    "stabilizers": known("rotated_surface_9").stabilizer_strings()})
    entries.append({"id": "xzzx", "groups": ["reference"], "source_rank": None, "duplicates": None,
                    "source_ler": {f"{e:g}": v for e, v in sorted(ref["XZZX"].items())},
                    "stabilizers": xzzx_9()})
    out = {
        "source": "bdalfavero/satisfying-codes experiments/surface (surface_codes.json, data/output_*.csv, bias_sampled_lers.csv), read 2026-10-16",
        "bias": "p_X = eta P, p_Y = p_Z = P; code-capacity exact maximum-likelihood LER, Z basis; stabilizers here are Hadamard-conjugated on every qubit",
        "source_codes": len(codes), "distinct_strings": len({tuple(_strings(c)) for c in codes}), "distinct_groups": len(groups),
        "rank_eta": RANK_ETA, "seed": seed, "n_random": n_random, "n_top": n_top, "n_bottom": n_bottom,
        "codes": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------
# run: score every code of the population
# ----------------------------------------------------------------------
_SETTINGS: dict = {}


def _init(settings: dict):
    _SETTINGS.update(settings)


def score(entry: dict, settings: dict | None = None) -> dict:
    s = settings or _SETTINGS
    code = StabilizerCode.from_stabilizers(entry["stabilizers"])
    rec = {
        "id": entry["id"], "groups": entry["groups"], "source_rank": entry["source_rank"],
        "source_ler": entry["source_ler"], "n": code.n, "k": code.k, "d": code.distance_exact(),
        "css": code.is_css(), "num_y": sum(g.count("Y") for g in entry["stabilizers"]), "cases": {},
    }
    for name in s["cases"]:
        prof, scale = make_case(name)
        t0 = time.perf_counter()
        tiers = evaluate_tiers(code, prof, s["rounds"], s["tiers"], s["seed"], s["shots"], s["min_failures"], s["max_shots"], scale=scale)
        case = {"wall_s": round(time.perf_counter() - t0, 2)}
        if "proxy" in tiers:
            z = tiers["proxy"]["per_basis"]["Z"]
            case["proxy"] = {k: z[k] for k in PROXY_KEYS if k in z}
            case["proxy"]["p_round"] = tiers["proxy"]["p_round"]
            if "proxy3_p_round" in tiers["proxy"]:
                case["proxy"]["proxy3_p_round"] = tiers["proxy"]["proxy3_p_round"]
        if "sampled" in tiers:
            z = tiers["sampled"]["per_basis"]["Z"]
            case["sampled"] = {"p_round": tiers["sampled"]["p_round"], "decoder": z["decoder"],
                               "shots": sum(tiers["sampled"]["per_basis"][b]["sampling"]["shots"] for b in tiers["sampled"]["per_basis"]),
                               "failures": sum(tiers["sampled"]["per_basis"][b]["sampling"]["failures"] for b in tiers["sampled"]["per_basis"])}
        rec["cases"][name] = case
    return rec


def run(out_dir: Path, population: Path, rounds: int, tiers, shots: int, min_failures: int, max_shots: int, seed: int,
        workers: int, cases, limit: int | None, verbose: bool) -> Path:
    pop = json.loads(population.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    records = out_dir / "records.jsonl"
    done = set()
    if records.exists():
        with open(records) as fh:
            done = {json.loads(line)["id"] for line in fh if line.strip()}
    todo = [e for e in pop["codes"] if e["id"] not in done]
    todo.sort(key=lambda e: "reference" not in e["groups"])  # references first, so partial reports have them
    if limit is not None:
        todo = todo[:limit]
    settings = {"rounds": rounds, "tiers": list(tiers), "shots": shots, "min_failures": min_failures, "max_shots": max_shots,
                "seed": seed, "cases": list(cases)}
    (out_dir / "settings.json").write_text(json.dumps({**settings, "population": str(population), "platform": platform.platform()}, indent=1))
    t0 = time.perf_counter()
    with open(records, "a") as fh:
        if workers <= 1:
            _init(settings)
            for i, e in enumerate(todo):
                rec = score(e)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if verbose:
                    print(f"{i + 1}/{len(todo)} {rec['id']} {time.perf_counter() - t0:.0f} s", flush=True)
        else:
            import multiprocessing as mp

            saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
            os.environ.update(dict.fromkeys(saved, "1"))
            try:
                ctx = mp.get_context("spawn")
                with ctx.Pool(workers, initializer=_init, initargs=(settings,)) as pool:
                    for i, rec in enumerate(pool.imap_unordered(score, todo, chunksize=1)):
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        if verbose and (i + 1) % 25 == 0:
                            print(f"{i + 1}/{len(todo)} {time.perf_counter() - t0:.0f} s", flush=True)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
    return records


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def load_records(out_dir: Path) -> list[dict]:
    recs = []
    with open(out_dir / "records.jsonl") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _p(rec: dict, case: str, tier: str) -> float:
    return rec["cases"][case][tier]["p_round"]


def _src(rec: dict, eta: float) -> float:
    return rec["source_ler"][f"{eta:g}"]


def _median(xs):
    xs = sorted(xs)
    m = len(xs)
    return xs[m // 2] if m % 2 else 0.5 * (xs[m // 2 - 1] + xs[m // 2])


def _quantile(xs, q):
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


def r_squared(y, columns) -> float:
    """Coefficient of determination of an ordinary least-squares fit of
    ``y`` on the given columns plus an intercept."""
    import numpy as np

    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in columns])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid**2).sum() / tot) if tot > 0 else float("nan")


def weight3_logical_mask(code: StabilizerCode):
    """Boolean mask over all weight-3 Paulis: commutes with every stabilizer
    and is not a stabilizer, so a minimum-weight logical of a distance-3 code."""
    import numpy as np

    from ..errors import paulis_of_weight

    n = code.n
    e3 = paulis_of_weight(n, 3)
    s = code.stabilizers.astype(np.int64)
    ex, ez = e3[:, :n].astype(np.int64), e3[:, n:].astype(np.int64)
    commutes = ~(((ex @ s[:, n:].T + ez @ s[:, :n].T) % 2).any(axis=1))
    red, _ = rref(code.stabilizers)
    red = red[red.any(axis=1)]
    mask = commutes.copy()
    for i in np.flatnonzero(commutes):  # in the group iff adding the row keeps the rank
        aug, _ = rref(np.vstack([red, e3[i]]))
        if aug.any(axis=1).sum() == red.shape[0]:
            mask[i] = False
    return e3, mask


def cc_mass(code: StabilizerCode, eta: float) -> float:
    """Low-rate code-capacity mass of the weight-3 logicals: the sum over
    them of the product of letter weights 1 (X, Y) and ``eta`` (Z), the
    bias of the source's channel after the Hadamard conjugation."""
    import numpy as np

    e3, mask = weight3_logical_mask(code)
    n = code.n
    x, z = e3[:, :n].astype(bool), e3[:, n:].astype(bool)
    w = np.where(z & ~x, eta, 1.0)
    return float(np.prod(w, axis=1)[mask].sum())


def analyze(recs: list[dict], tiers, stabilizers: dict[str, list[str]] | None = None) -> dict:
    by_id = {r["id"]: r for r in recs}
    rnd = [r for r in recs if "random" in r["groups"]]
    top = [r for r in recs if "top" in r["groups"]]
    bottom = [r for r in recs if "bottom" in r["groups"]]
    refs = {r["id"]: r for r in recs if "reference" in r["groups"]}
    cases = [c for c in CASES if all(c in r["cases"] for r in recs)]
    out = {"counts": {"scored": len(recs), "random": len(rnd), "top": len(top), "bottom": len(bottom)}, "cases": {}}
    for case in cases:
        eta = CASE_SOURCE_ETA[case]
        c: dict = {"source_eta": eta, "tiers": {}}
        for tier in tiers:
            t: dict = {}
            xs = [_src(r, eta) for r in rnd]
            ys = [_p(r, case, tier) for r in rnd]
            t["spearman_random"] = spearman(xs, ys)
            k = 20
            best_src = {r["id"] for r in sorted(rnd, key=lambda r: _src(r, eta))[:k]}
            best_circ = {r["id"] for r in sorted(rnd, key=lambda r: _p(r, case, tier))[:k]}
            t["top20_overlap"] = len(best_src & best_circ)
            med_rnd = _median(ys)
            t["groups"] = {}
            for gname, g in (("random", rnd), ("top", top), ("bottom", bottom)):
                ps = [_p(r, case, tier) for r in g]
                if ps:
                    t["groups"][gname] = {"median": _median(ps), "q25": _quantile(ps, 0.25), "q75": _quantile(ps, 0.75),
                                          "below_random_median": sum(p < med_rnd for p in ps) / len(ps)}
            t["references"] = {rid: _p(r, case, tier) for rid, r in refs.items()}
            allc = [r for r in recs if "reference" not in r["groups"]]
            ranked = sorted(allc, key=lambda r: _p(r, case, tier))
            t["best"] = [{"id": r["id"], "p_round": _p(r, case, tier), "source_rank": r["source_rank"], "css": r["css"],
                          "batches": r["cases"][case]["proxy"]["batches"] if "proxy" in r["cases"][case] else None,
                          "num_y": r["num_y"]} for r in ranked[:8]]
            n_better = {rid: sum(_p(r, case, tier) < t["references"][rid] for r in allc) for rid in refs}
            t["codes_better_than_reference"] = {rid: [n_better[rid], len(allc)] for rid in refs}
            if rnd and "proxy" in rnd[0]["cases"][case]:
                raw = {
                    "cc_mass": [cc_mass(StabilizerCode.from_stabilizers(stabilizers[r["id"]]), eta) for r in rnd] if stabilizers else [],
                    "batches": [r["cases"][case]["proxy"]["batches"] for r in rnd],
                    "layers": [r["cases"][case]["proxy"]["layers"] for r in rnd],
                    "d_circ": [r["cases"][case]["proxy"]["d_circ"] for r in rnd],
                    "d_circ_mass": [r["cases"][case]["proxy"]["d_circ_mass"] for r in rnd],
                    "num_y": [r["num_y"] for r in rnd],
                    "css": [1.0 if r["css"] else 0.0 for r in rnd],
                }
                raw = {f: v for f, v in raw.items() if v}
                t["constant_features"] = {f: v[0] for f, v in raw.items() if len(set(v)) == 1}
                feats = {f: v for f, v in raw.items() if f not in t["constant_features"]}
                t["feature_spearman"] = {f: spearman(v, ys) for f, v in feats.items()}
                logy = [math.log10(y) for y in ys]
                logx = [math.log10(x) for x in xs]
                t["r2"] = {
                    "source_ler": r_squared(logy, [logx]),
                    "structure": r_squared(logy, list(feats.values())) if feats else float("nan"),
                    "both": r_squared(logy, [logx, *feats.values()]),
                }
                css_p = [_p(r, case, tier) for r in rnd if r["css"]]
                non_p = [_p(r, case, tier) for r in rnd if not r["css"]]
                t["css_median"] = {"css": _median(css_p) if css_p else None, "non_css": _median(non_p) if non_p else None,
                                   "css_count": len(css_p), "non_css_count": len(non_p)}
            c["tiers"][tier] = t
        out["cases"][case] = c
    out["by_id"] = {rid: by_id[rid]["source_rank"] for rid in by_id}
    return out


def _f(x, digits=5):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def report_tables(an: dict, pop: dict, tiers) -> str:
    L = []
    cnt = an["counts"]
    L.append(f"Scored {cnt['scored']} codes: {cnt['random']} uniform random distinct codes, the {cnt['top']} best and the "
             f"{cnt['bottom']} worst distinct codes by the source's rate at eta = {pop['rank_eta']:g} (overlaps counted once), "
             f"and the two reference codes. The source holds {pop['source_codes']:,} codes, {pop['distinct_strings']:,} distinct "
             f"generator lists, and {pop['distinct_groups']:,} distinct stabilizer groups.")
    L.append("\n## Rank agreement with the source\n")
    L.append("Over the random sample. Spearman is between the source's code-capacity rate at the matching eta (eta = 1 for Forte) "
             "and the circuit-level p_round; the overlap counts how many of the 20 best codes by one score are among the 20 best by the other.\n")
    L.append("| case | tier | Spearman | top-20 overlap |")
    L.append("|---|---|---|---|")
    for case, c in an["cases"].items():
        for tier in tiers:
            t = c["tiers"][tier]
            L.append(f"| {CASE_TITLES[case]} | {tier} | {t['spearman_random']:.2f} | {t['top20_overlap']} / 20 |")
    L.append("\n## Do the source's best codes score better than random ones?\n")
    L.append("Median p_round (interquartile range) per group, and the fraction of each group below the random sample's median.\n")
    L.append("| case | tier | random | best by source | worst by source | best below random median | worst below random median |")
    L.append("|---|---|---|---|---|---|---|")
    for case, c in an["cases"].items():
        for tier in tiers:
            g = c["tiers"][tier]["groups"]
            def cell(name):
                if name not in g:
                    return "n/a"
                x = g[name]
                return f"{x['median']:.5f} ({x['q25']:.5f}, {x['q75']:.5f})"
            L.append(f"| {CASE_TITLES[case]} | {tier} | {cell('random')} | {cell('top')} | {cell('bottom')} | "
                     f"{g['top']['below_random_median']:.0%} | {g['bottom']['below_random_median']:.0%} |" if "top" in g and "bottom" in g else
                     f"| {CASE_TITLES[case]} | {tier} | {cell('random')} | {cell('top')} | {cell('bottom')} | n/a | n/a |")
    L.append("\n## What predicts the circuit-level rate\n")
    L.append("Over the random sample at the sampled tier. Spearman of p_round with each feature: cc_mass, the code-capacity mass of "
             "the weight-3 logical operators under the case's bias (letter weight 1 for X and Y, eta for Z), computed from the "
             "stabilizers alone; batches, groups of generators whose controlled gates cannot interleave; layers, the circuit depth of "
             "one round; d_circ, the exact circuit distance; d_circ_mass, the probability mass of the minimum-weight undetectable "
             "circuit fault sets in the Z-basis memory; num_y, Y letters among the generators; css. Then the R^2 of a least-squares "
             "fit of log10 p_round on log10 source rate alone, on the non-constant features alone, and on both.\n")
    tier = "sampled" if "sampled" in tiers else tiers[0]
    first = next((c["tiers"][tier] for c in an["cases"].values() if "feature_spearman" in c["tiers"][tier]), None)
    if first is not None:
        names = list(first["feature_spearman"])
        if first["constant_features"]:
            L.append("Constant across the random sample, so carrying no information here: "
                     + ", ".join(f"{f} = {v:g}" if isinstance(v, (int, float)) else f"{f} = {v}" for f, v in first["constant_features"].items()) + ".\n")
        L.append("| case | " + " | ".join(names) + " | R^2 source | R^2 structure | R^2 both |")
        L.append("|---|" + "---|" * (len(names) + 3))
        for case, c in an["cases"].items():
            t = c["tiers"][tier]
            if "feature_spearman" not in t:
                continue
            fs, r2 = t["feature_spearman"], t["r2"]
            L.append(f"| {CASE_TITLES[case]} | " + " | ".join(_f(fs.get(f), 2) for f in names)
                     + f" | {_f(r2['source_ler'], 2)} | {_f(r2['structure'], 2)} | {_f(r2['both'], 2)} |")
    L.append("\n## Best codes of the population\n")
    L.append("Sampled tier. The source rank is among the distinct stabilizer groups (1 = best at eta = 10). "
             "The last two columns count population codes with a lower p_round than each reference code.\n")
    for case, c in an["cases"].items():
        t = c["tiers"][tier]
        refs = t["references"]
        better = t["codes_better_than_reference"]
        L.append(f"\n### {CASE_TITLES[case]}\n")
        if "rotated_surface" in refs and "xzzx" in refs:
            L.append(f"rotated surface p_round {refs['rotated_surface']:.5f}, XZZX p_round {refs['xzzx']:.5f}; "
                     f"codes below rotated surface: {better['rotated_surface'][0]} of {better['rotated_surface'][1]}, "
                     f"below XZZX: {better['xzzx'][0]} of {better['xzzx'][1]}.\n")
        L.append("| code | p_round | source rank | CSS | batches | Y letters |")
        L.append("|---|---|---|---|---|---|")
        for b in t["best"]:
            L.append(f"| {b['id']} | {b['p_round']:.5f} | {b['source_rank']} | {'yes' if b['css'] else 'no'} | {b['batches']} | {b['num_y']} |")
    return "\n".join(L)


def confirm(out_dir: Path, population: Path, top_k: int, rounds: int, min_failures: int, max_shots: int, seed: int, verbose: bool) -> Path:
    """Re-score the best ``top_k`` codes of every case (sampled tier) and
    the references with a large shot budget, so the ranking at the top is
    not shot noise."""
    pop = {e["id"]: e for e in json.loads(population.read_text())["codes"]}
    an = json.loads((out_dir / "analysis.json").read_text())
    ids: list[str] = [rid for rid, e in pop.items() if "reference" in e["groups"]]
    for c in an["cases"].values():
        for b in c["tiers"]["sampled"]["best"][:top_k]:
            if b["id"] not in ids:
                ids.append(b["id"])
    cases = list(an["cases"])
    out: dict = {"top_k": top_k, "rounds": rounds, "min_failures": min_failures, "max_shots": max_shots, "seed": seed, "cases": {c: {} for c in cases}}
    for rid in ids:
        code = StabilizerCode.from_stabilizers(pop[rid]["stabilizers"])
        for case in cases:
            prof, scale = make_case(case)
            t0 = time.perf_counter()
            res = evaluate_tiers(code, prof, rounds, ["sampled"], seed, min_failures * 100, min_failures, max_shots, scale=scale)["sampled"]
            per = res["per_basis"]
            out["cases"][case][rid] = {
                "p_round": res["p_round"],
                "shots": sum(per[b]["sampling"]["shots"] for b in per), "failures": sum(per[b]["sampling"]["failures"] for b in per),
                "wall_s": round(time.perf_counter() - t0, 1),
            }
        if verbose:
            print(rid, {c: round(out["cases"][c][rid]["p_round"], 5) for c in cases}, flush=True)
    path = out_dir / "confirm.json"
    path.write_text(json.dumps(out, indent=1))
    return path


def confirm_tables(conf: dict, recs: list[dict]) -> str:
    L = ["\n## Confirmation of the best codes\n"]
    L.append(f"The {conf['top_k']} best codes of each case at the sampled tier, and the references, re-scored with at least "
             f"{conf['min_failures']} failures per basis (up to {conf['max_shots']:,} shots) on the same circuit as the first pass. "
             f"The relative standard error of each rate is about 1 / sqrt(failures). Codes are sorted by the confirmed rate; "
             f"the first-pass rate is the one the ranking used.\n")
    ranks = {r["id"]: "reference" if r["source_rank"] is None else r["source_rank"] for r in recs}
    for case, rows in conf["cases"].items():
        first = {r["id"]: _p(r, case, "sampled") for r in recs if case in r["cases"] and "sampled" in r["cases"][case]}
        L.append(f"\n### {CASE_TITLES[case]}\n")
        L.append("| code | confirmed p_round | rel. std. err. | first pass | source rank |")
        L.append("|---|---|---|---|---|")
        for rid, r in sorted(rows.items(), key=lambda kv: kv[1]["p_round"]):
            se = 1 / math.sqrt(r["failures"]) if r["failures"] else float("nan")
            L.append(f"| {rid} | {r['p_round']:.5f} | {se:.0%} | {_f(first.get(rid))} | {ranks.get(rid, 'n/a')} |")
    return "\n".join(L)


def make_figure(recs: list[dict], stem: Path, tier: str = "sampled") -> Path | None:
    try:
        from . import figstyle
    except ImportError:  # pragma: no cover
        return None
    try:
        plt = figstyle.apply()
    except ImportError:  # pragma: no cover
        return None
    rnd = [r for r in recs if "random" in r["groups"] and all(c in r["cases"] for c in ("idle_eta_10", "forte"))]
    refs = {r["id"]: r for r in recs if "reference" in r["groups"]}
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE + 0.3))
    panels = (("idle_eta_10", 10.0, "source LER, eta = 10"), ("forte", 1.0, "source LER, eta = 1"))
    batches = sorted({r["cases"]["idle_eta_10"]["proxy"]["batches"] for r in rnd})
    colors = {b: figstyle.SERIES_COLORS[i % len(figstyle.SERIES_COLORS)] for i, b in enumerate(batches)}
    for ax, (case, eta, xlabel), letter in zip(axes, panels, "ab"):
        for b in batches:
            pts = [(r["source_ler"][f"{eta:g}"], r["cases"][case][tier]["p_round"]) for r in rnd if r["cases"][case]["proxy"]["batches"] == b]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=6, color=colors[b], alpha=0.6, linewidths=0, label=f"{b} batches")
        for rid, marker, label in (("rotated_surface", "s", "rotated surface"), ("xzzx", "D", "XZZX")):
            if rid in refs:
                ax.scatter([refs[rid]["source_ler"][f"{eta:g}"]], [refs[rid]["cases"][case][tier]["p_round"]], s=30, marker=marker,
                           color="black", facecolors="none", linewidths=1.0, label=label, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"circuit-level p_round, {tier} tier")
        ax.set_title(CASE_TITLES[case])
        figstyle.panel_label(ax, letter)
    axes[0].legend(loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    figstyle.save(fig, stem)
    return stem.with_suffix(".png")


def write_report(out_dir: Path, population: Path, figure: Path | None = FIGURE) -> Path:
    pop = json.loads(population.read_text())
    settings = json.loads((out_dir / "settings.json").read_text())
    recs = load_records(out_dir)
    tiers = [t for t in settings["tiers"] if all(t in r["cases"][c] for r in recs for c in r["cases"])]
    an = analyze(recs, tiers, {e["id"]: e["stabilizers"] for e in pop["codes"]})
    (out_dir / "analysis.json").write_text(json.dumps(an, indent=1))
    fig_line = ""
    if figure is not None:
        png = make_figure(recs, figure)
        if png is not None:
            rel = os.path.relpath(png, out_dir.parent)
            fig_line = f"\n![source rate against circuit-level rate]({rel})\n\nFigure: the random sample's source rate against its circuit-level rate under the biased idle case at eta = 10 (a) and under Forte (b), colored by the number of batches in the extraction circuit; the reference codes are the open markers.\n"
    conf_text = ""
    if (out_dir / "confirm.json").exists():
        conf_text = confirm_tables(json.loads((out_dir / "confirm.json").read_text()), recs)
    text = f"""# A population of SAT-designed codes under the memory objective

Status: {'complete' if an['counts']['scored'] >= len(pop['codes']) else f"partial, {an['counts']['scored']} of {len(pop['codes'])} codes"}.
Every number comes from `experiments/sat_population/runs/records.jsonl`;
this file regenerates with `python -m autoqec_sim.experiments.sat_population report`.
Code: `src/autoqec_sim/experiments/sat_population.py`; population:
`experiments/sat_population/population.json`.

Source: DalFavero et al., arXiv:2608.23460, `experiments/surface/` of
github.com/bdalfavero/satisfying-codes: 100,000 SAT-sampled 9-qubit codes
on the stabilizer footprint of the rotated surface code, each scored by
the exact maximum-likelihood logical error rate under code-capacity
biased noise (`p_X = eta P`, `p_Y = p_Z = P`) in the Z basis. The
reference codes are the rotated surface code and its XZZX deformation,
with the source's rates for both.

Here each code is Hadamard-conjugated on every qubit so its X tailoring
becomes Z tailoring, and scored by the circuit-level memory objective,
{settings['rounds']} rounds, under four cases: a depolarizing profile with a
biased idle channel at source eta 4 and 10 (bias parameter eta / 2, so the
Z rate is eta times each of the X and Y rates), the same profile with
every channel biased at source eta 10, and the IonQ Forte preset with
sequential timing. Tiers: proxy (maximum-likelihood failure over fault
sets of weight at most two, closed form) and sampled (Monte Carlo from the
detector error model with a lookup decoder built from the same fault
sets, {settings['shots']:,} shots and at least {settings['min_failures']} failures
up to {settings['max_shots']:,} shots). Every code, the references included, uses the hook-aware gate
order, whose random tie-breaking left the proxy rate unchanged over five
seeds on this footprint. Distances are recomputed exactly.
Sampled-tier differences below about ten percent are within shot noise.
{fig_line}
{report_tables(an, pop, tiers)}
{conf_text}
"""
    path = out_dir.parent / "REPORT.md"
    path.write_text(text)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True, help="satisfying-codes/experiments/surface")
    p.add_argument("--population", default=str(POPULATION))
    p.add_argument("--random", type=int, default=2000)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--bottom", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--population", default=str(POPULATION))
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--tiers", nargs="+", default=["proxy", "sampled"])
    r.add_argument("--cases", nargs="+", default=list(CASES))
    r.add_argument("--shots", type=int, default=20_000)
    r.add_argument("--min-failures", type=int, default=50)
    r.add_argument("--max-shots", type=int, default=200_000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--quiet", action="store_true")
    c = sub.add_parser("confirm")
    c.add_argument("--population", default=str(POPULATION))
    c.add_argument("--out", default=str(DEFAULT_OUT))
    c.add_argument("--top", type=int, default=8)
    c.add_argument("--rounds", type=int, default=3)
    c.add_argument("--min-failures", type=int, default=500)
    c.add_argument("--max-shots", type=int, default=4_000_000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--quiet", action="store_true")
    q = sub.add_parser("report")
    q.add_argument("--population", default=str(POPULATION))
    q.add_argument("--out", default=str(DEFAULT_OUT))
    q.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        out = prepare(Path(a.source), Path(a.population), a.random, a.top, a.bottom, a.seed)
        print(f"{out['distinct_groups']:,} distinct groups of {out['source_codes']:,} codes; wrote {len(out['codes'])} entries to {a.population}")
        return
    out = Path(a.out)
    if a.cmd == "run":
        run(out, Path(a.population), a.rounds, a.tiers, a.shots, a.min_failures, a.max_shots, a.seed, a.workers, a.cases, a.limit, not a.quiet)
    elif a.cmd == "confirm":
        confirm(out, Path(a.population), a.top, a.rounds, a.min_failures, a.max_shots, a.seed, not a.quiet)
    print("wrote", write_report(out, Path(a.population), None if getattr(a, "no_figure", False) else FIGURE))


if __name__ == "__main__":
    main()
