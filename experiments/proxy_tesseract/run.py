# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""The proxy tier against Tesseract at circuit distance 5.

Every code of ``codes.json`` (the 5x5 rotated surface code, its XZZX
deformation, and 19 random single-qubit Clifford deformations of it, all of
distance 5) is scored in both memory bases under a noise profile at a
scale: the proxy tier (closed form, no shots; at circuit distance 5 the
weight-3 form from the weight-5 and weight-6 undetectable sets) and
Tesseract sampling with a fixed number of shots. The report gives the rank
correlation between the two, the ratio of the sampled rate to the proxy,
the share of the proxy carried by the weight-5 sets alone, and the cost of
each. Runs are JSON lines under ``runs/`` (one record per code, resumable).

    uv run python -m experiments.proxy_tesseract.run run --rounds 3 --shots 50000
    uv run python -m experiments.proxy_tesseract.run report
    uv run python -m experiments.proxy_tesseract.run figure
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
from autoqec_sim.noise import presets

HERE = Path(__file__).resolve().parent
CODES = HERE / "codes.json"
DEFAULT_OUT = HERE / "runs"
REPORT = HERE / "REPORT.md"
FIGURE = Path("docs/assets/figures/proxy_tesseract")
PROFILES = {"forte": lambda: presets.forte("sequential"), "depolarizing": lambda: presets.depolarizing(1e-3)}


def load_codes(path: Path = CODES) -> list[dict]:
    return json.loads(path.read_text())["codes"]


def records_path(out_dir: Path, profile: str, rounds: int, scale: float) -> Path:
    return out_dir / f"{profile}_r{rounds}_s{scale:g}.jsonl"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_code(entry: dict, profile, rounds: int, scale: float, shots: int, seed: int) -> dict:
    """Both bases of one code: proxy tier with its weight-5-only variant,
    the weight-5 and weight-6 set counts and masses, and Tesseract."""
    code = StabilizerCode.from_stabilizers(entry["stabilizers"])
    t0 = time.perf_counter()
    orders = mo.hook_orders(code, 5, seed)
    rec = {"id": entry["id"], "rounds": rounds, "scale": scale, "t_orders": round(time.perf_counter() - t0, 1), "bases": {}}
    for basis in mo.BASES:
        mc = mo.build_memory_circuit(code, rounds, basis, hook_target=5, orders=orders)
        dem = mo.memory_dem(mc, profile, scale)
        sig = mo.dem_signature(dem)
        t1 = time.perf_counter()
        pr = mo._proxy_failure(sig, None, 2_000_000, False)
        found = mo.undetectable_logicals(sig, 6)
        q = np.clip(sig.probs, 0.0, 1.0 - 1e-15)
        weight5_only = mo._pfail_weight3(mo._cluster_search(sig), {5: found[5]}, q, 1.0) if 5 in found else 0.0
        t2 = time.perf_counter()
        ts = mo.sample_failures(dem, sig, "tesseract", shots, seed, 0, None)
        t3 = time.perf_counter()
        rec["bases"][basis] = {
            "mechanisms": len(sig), "expected_faults": float(sig.probs.sum()), "extraction": mc.extraction, "layers": mc.layers,
            "proxy": pr["p_fail"], "proxy_weight": pr["weight"], "d_circ": pr["d_circ"]["d_circ"] if pr["d_circ"] else None,
            "proxy_weight5_only": weight5_only,
            "counts": {str(w): len(v[0]) for w, v in found.items()}, "mass": {str(w): float(v[1].sum()) for w, v in found.items()},
            "tesseract": ts["p_fail"], "failures": ts["failures"], "shots": ts["shots"],
            "t_proxy": round(t2 - t1, 1), "t_tesseract": round(t3 - t2, 1),
        }
        mo._CLUSTER_CACHE.clear()
    return rec


def run(out_dir: Path, profile: str, rounds: int, scale: float, shots: int, seed: int, verbose: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = records_path(out_dir, profile, rounds, scale)
    done = {r["id"] for r in load_records(path)}
    prof = PROFILES[profile]()
    with path.open("a") as fh:
        for entry in load_codes():
            if entry["id"] in done:
                continue
            rec = score_code(entry, prof, rounds, scale, shots, seed)
            rec["profile"] = profile
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if verbose:
                b = rec["bases"]
                print(
                    f"{entry['id']}: proxy {b['Z']['proxy'] + b['X']['proxy']:.3e} tesseract {b['Z']['tesseract'] + b['X']['tesseract']:.3e} "
                    f"({b['Z']['failures'] + b['X']['failures']} failures) proxy {b['X']['t_proxy']:.0f} s, tesseract {b['X']['t_tesseract']:.0f} s per basis",
                    flush=True,
                )
    return path


# ------------------------------------------------------------------ analysis
def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).correlation)


def analyze(recs: list[dict]) -> dict:
    """Rank agreement and ratios over the codes of one run."""
    out = {"codes": len(recs)}
    for key in ("Z", "X", "total"):
        if key == "total":
            px = np.array([r["bases"]["Z"]["proxy"] + r["bases"]["X"]["proxy"] for r in recs])
            tx = np.array([r["bases"]["Z"]["tesseract"] + r["bases"]["X"]["tesseract"] for r in recs])
            fails = np.array([r["bases"]["Z"]["failures"] + r["bases"]["X"]["failures"] for r in recs])
            w5 = np.array([r["bases"]["Z"]["proxy_weight5_only"] + r["bases"]["X"]["proxy_weight5_only"] for r in recs])
            mass = np.array([sum(r["bases"][b]["mass"].get("5", 0.0) for b in "ZX") for r in recs])
        else:
            px = np.array([r["bases"][key]["proxy"] for r in recs])
            tx = np.array([r["bases"][key]["tesseract"] for r in recs])
            fails = np.array([r["bases"][key]["failures"] for r in recs])
            w5 = np.array([r["bases"][key]["proxy_weight5_only"] for r in recs])
            mass = np.array([r["bases"][key]["mass"].get("5", 0.0) for r in recs])
        ok = tx > 0
        ratio = tx[ok] / px[ok]
        out[key] = {
            "spearman": _spearman(px, tx) if len(recs) > 2 else None,
            "spearman_weight5_only": _spearman(w5, tx) if len(recs) > 2 else None,
            "spearman_weight5_mass": _spearman(mass, tx) if len(recs) > 2 else None,
            "ratio_median": float(np.median(ratio)) if ok.any() else None,
            "ratio_min": float(ratio.min()) if ok.any() else None,
            "ratio_max": float(ratio.max()) if ok.any() else None,
            "failures_median": int(np.median(fails)),
            "failures_min": int(fails.min()),
            "weight5_share_median": float(np.median(w5 / px)),
        }
    out["expected_faults"] = float(np.mean([r["bases"]["X"]["expected_faults"] for r in recs]))
    out["seconds_per_basis"] = {
        "proxy": float(statistics.median(r["bases"][b]["t_proxy"] for r in recs for b in "ZX")),
        "tesseract": float(statistics.median(r["bases"][b]["t_tesseract"] for r in recs for b in "ZX")),
    }
    out["extraction"] = sorted({r["bases"][b]["extraction"] for r in recs for b in "ZX"})
    return out


def write_report(out_dir: Path = DEFAULT_OUT, report: Path = REPORT) -> Path:
    runs = sorted(out_dir.glob("*.jsonl"))
    L = [
        "# The proxy tier against Tesseract at circuit distance 5",
        "",
        "Every number comes from the JSON lines under `experiments/proxy_tesseract/runs/`; this file regenerates with "
        "`uv run python -m experiments.proxy_tesseract.run report`. Code: `experiments/proxy_tesseract/run.py`; codes: `codes.json` "
        "(the 5x5 rotated surface code, its XZZX deformation, and 19 random single-qubit Clifford deformations of it, all of distance 5).",
        "",
        "The proxy tier is the closed-form maximum-likelihood failure over fault sets of weight at most 3 (at circuit distance 5 the weight-3 "
        "form built from the weight-5 and weight-6 undetectable sets), with no shots and no decoder; Tesseract is a most-likely-error "
        "search decoder on the same detector error model, sampled with a fixed number of shots. Both score the same circuit. Where a "
        "profile is named Forte, it is the package's Forte preset, a noise model assembled from public specifications, not a device "
        "measurement.",
        "",
    ]
    for path in runs:
        recs = load_records(path)
        if len(recs) < 3:
            continue
        an = analyze(recs)
        r0 = recs[0]
        L += [
            f"## {r0.get('profile', path.stem)}, {r0['rounds']} rounds, scale {r0['scale']:g}",
            "",
            f"{an['codes']} codes, {recs[0]['bases']['X']['shots']:,} Tesseract shots per basis, {an['expected_faults']:.2f} expected faults per shot, "
            f"extraction {', '.join(an['extraction'])}. Seconds per basis: proxy {an['seconds_per_basis']['proxy']:.0f}, Tesseract {an['seconds_per_basis']['tesseract']:.0f}.",
            "",
            "| quantity | Z basis | X basis | both bases |",
            "|---|---|---|---|",
        ]
        rows = [
            ("Spearman, proxy against Tesseract", "spearman", "{:.2f}"),
            ("Spearman, weight-5 sets only against Tesseract", "spearman_weight5_only", "{:.2f}"),
            ("Spearman, weight-5 fault mass against Tesseract", "spearman_weight5_mass", "{:.2f}"),
            ("Tesseract over proxy, median", "ratio_median", "{:.2f}"),
            ("Tesseract over proxy, range", None, None),
            ("Tesseract failures per code, median (min)", None, None),
            ("share of the proxy from the weight-5 sets, median", "weight5_share_median", "{:.2f}"),
        ]
        for label, key, fmt in rows:
            cells = []
            for b in ("Z", "X", "total"):
                a = an[b]
                if key is not None:
                    v = a[key]
                    cells.append(fmt.format(v) if v is not None else "n/a")
                elif label.startswith("Tesseract over proxy, range"):
                    cells.append(f"{a['ratio_min']:.2f} to {a['ratio_max']:.2f}" if a["ratio_min"] is not None else "n/a")
                else:
                    cells.append(f"{a['failures_median']} ({a['failures_min']})")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L += ["", "| code | proxy Z | Tesseract Z | proxy X | Tesseract X | both, proxy | both, Tesseract | failures |", "|---|---|---|---|---|---|---|---|"]
        for r in sorted(recs, key=lambda r: r["bases"]["Z"]["tesseract"] + r["bases"]["X"]["tesseract"]):
            z, x = r["bases"]["Z"], r["bases"]["X"]
            L.append(
                f"| {r['id']} | {z['proxy']:.2e} | {z['tesseract']:.2e} | {x['proxy']:.2e} | {x['tesseract']:.2e} | "
                f"{z['proxy'] + x['proxy']:.2e} | {z['tesseract'] + x['tesseract']:.2e} | {z['failures'] + x['failures']} |"
            )
        L.append("")
    report.write_text("\n".join(L))
    return report


def make_figure(out_dir: Path = DEFAULT_OUT, out_stem: Path | str = FIGURE):
    """Per run, the Tesseract rate against the proxy per code and basis
    (log-log, Poisson error bars from the failure counts, identity line)."""
    from ..common import figstyle

    plt = figstyle.apply()
    runs = [p for p in sorted(out_dir.glob("*.jsonl")) if len(load_records(p)) >= 3]
    if not runs:
        return None
    fig, axes = plt.subplots(1, len(runs), figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE), squeeze=False)
    for ax, path, letter in zip(axes[0], runs, "abcdef", strict=False):
        recs = load_records(path)
        an = analyze(recs)
        lo, hi = np.inf, 0.0
        for basis, color, marker in (("X", figstyle.SERIES_COLORS[0], "o"), ("Z", figstyle.SERIES_COLORS[1], "s")):
            px = np.array([r["bases"][basis]["proxy"] for r in recs])
            tx = np.array([r["bases"][basis]["tesseract"] for r in recs])
            fails = np.array([r["bases"][basis]["failures"] for r in recs])
            ok = (tx > 0) & (px > 0)
            err = tx[ok] / np.sqrt(np.maximum(fails[ok], 1))
            ax.errorbar(px[ok], tx[ok], yerr=err, fmt=marker, color=color, label=f"{basis} basis", capsize=1.5, linewidth=figstyle.THIN, alpha=0.85)
            lo, hi = min(lo, px[ok].min(), tx[ok].min()), max(hi, px[ok].max(), tx[ok].max())
        ax.plot([lo * 0.7, hi * 1.4], [lo * 0.7, hi * 1.4], "-", color=figstyle.NEUTRAL, linewidth=figstyle.THIN)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("proxy p_fail per shot")
        ax.set_ylabel("Tesseract p_fail per shot")
        r0 = recs[0]
        ax.set_title(f"{r0.get('profile', path.stem)}, {r0['rounds']} rounds, scale {r0['scale']:g}")
        ax.text(0.97, 0.05, f"Spearman {an['total']['spearman']:.2f} (both bases)\n{an['codes']} codes", transform=ax.transAxes, ha="right", va="bottom")
        ax.legend(loc="upper left")
        if len(runs) > 1:
            figstyle.panel_label(ax, letter)
    fig.tight_layout()
    return figstyle.save(fig, out_stem)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", choices=sorted(PROFILES), default="forte")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--scale", type=float, default=0.25)
    r.add_argument("--shots", type=int, default=50_000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--quiet", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    f = sub.add_parser("figure")
    f.add_argument("--out", default=str(DEFAULT_OUT))
    f.add_argument("--stem", default=str(FIGURE))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "run":
        run(out, a.profile, a.rounds, a.scale, a.shots, a.seed, not a.quiet)
        print("wrote", write_report(out))
    elif a.cmd == "report":
        print("wrote", write_report(out))
    else:
        print("wrote", *make_figure(out, a.stem))


if __name__ == "__main__":
    main()
