"""SAT-designed biased-noise codes under the circuit-level memory objective.

DalFavero, Watkins, LaBorde, Russo, Egger, Quiroz, and LaRose
(arXiv:2608.23460; github.com/bdalfavero/satisfying-codes) sample 9-qubit
surface-like codes by SAT and score each by its exact maximum-likelihood
logical error rate under code-capacity biased noise, ``p_X = eta P``,
``p_Y = p_Z = P``, against the rotated surface code (CSS) and its XZZX
deformation. ``experiments/sat_codes/selection.json`` holds the twelve
best distinct codes at ``eta = 10`` and three of the worst, with their
reported rates, read from that repository on 2026-10-16.

This module asks whether that code-capacity ranking survives a
circuit-level memory experiment: each code, Hadamard-conjugated on every
qubit so its X tailoring becomes the Z tailoring autoqec-sim's bias
parameter expresses, is scored by :class:`autoqec_sim.memory_objective.MemoryObjective`
over ``R`` rounds under a depolarizing profile with a biased idle channel
(``eta_autoqec = eta / 2``, so that the Z rate is ``eta`` times each of
the X and Y rates), at the proxy tier and the sampled tier, alongside the
two reference codes and under one device preset. The exact distance of
every code is recomputed here.

    python -m autoqec_sim.experiments.sat_codes run
    python -m autoqec_sim.experiments.sat_codes report
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from ..code import StabilizerCode
from ..known_codes import get as known
from .memory_objective import (
    DEPOLARIZING_SCALE,
    PROFILE_SCALES,
    biased_idle_profile,
    evaluate_tiers,
    hadamard_conjugate,
    spearman,
    xzzx_9,
)

DEFAULT_OUT = Path("experiments/sat_codes/runs")
SELECTION = Path("experiments/sat_codes/selection.json")
ETAS = (1.0, 4.0, 10.0)  # the source's eta; autoqec's bias parameter is eta / 2


def _strings(stabs: list[str]) -> list[str]:
    """Drop the sign prefix and write identities as I; signs do not affect
    the code's distance or its memory experiment."""
    return [s.lstrip("+-").replace("_", "I") for s in stabs]


def load_codes(selection: Path) -> list[dict]:
    sel = json.loads(selection.read_text())
    codes = []
    for c in sel["codes"]:
        strs = hadamard_conjugate(_strings(c["stabilizers"]), range(9))
        codes.append({"id": c["id"], "source_ler": {float(k): v for k, v in c["ler"].items()}, "stabilizers": strs})
    ref = sel["reference"]
    codes.append({"id": "rotated_surface", "source_ler": {float(k): v for k, v in ref["CSS"].items()}, "stabilizers": known("rotated_surface_9").stabilizer_strings()})
    codes.append({"id": "xzzx", "source_ler": {float(k): v for k, v in ref["XZZX"].items()}, "stabilizers": xzzx_9()})
    return codes


def run(out_dir: Path, selection: Path, rounds: int, tiers, shots: int, min_failures: int, max_shots: int, seed: int, verbose: bool) -> dict:
    from ..noise import presets

    codes = load_codes(selection)
    cases = [(f"depol_idle_eta_{eta:g}", biased_idle_profile(eta / 2, "parallel"), eta, DEPOLARIZING_SCALE) for eta in ETAS]
    cases.append(("forte_sequential", presets.forte("sequential"), None, PROFILE_SCALES["forte"]))
    out = {"kind": "sat_codes", "rounds": rounds, "tiers": list(tiers), "shots": shots, "min_failures": min_failures, "max_shots": max_shots,
           "seed": seed, "platform": platform.platform(), "selection": str(selection), "codes": {}}
    for c in codes:
        code = StabilizerCode.from_stabilizers(c["stabilizers"])
        rec = {"source_ler": c["source_ler"], "stabilizers": c["stabilizers"], "n": code.n, "k": code.k,
               "d": code.distance_exact(), "css": code.is_css(), "cases": {}}
        for name, prof, eta, scale in cases:
            t0 = time.perf_counter()
            rec["cases"][name] = evaluate_tiers(code, prof, rounds, tiers, seed, shots, min_failures, max_shots, scale=scale)
            rec["cases"][name]["eta"] = eta
            rec["cases"][name]["wall_s"] = round(time.perf_counter() - t0, 2)
        out["codes"][c["id"]] = rec
        if verbose:
            print(c["id"], f"d={rec['d']}", {name: {t: round(rec["cases"][name][t]["p_round"], 5) for t in tiers} for name in rec["cases"]}, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sat_codes.json").write_text(json.dumps(out, indent=1))
    return out


def report_tables(out: dict) -> str:
    tiers = out["tiers"]
    ids = list(out["codes"])
    lines = []
    for name in next(iter(out["codes"].values()))["cases"]:
        eta = out["codes"][ids[0]]["cases"][name]["eta"]
        head = f"source eta = {eta:g}" if eta is not None else "Forte preset (public-spec model), sequential"
        lines.append(f"\n### {head}\n")
        cols = " | ".join(f"{t} p_round" for t in tiers)
        lines.append(f"| code | d | CSS | source LER (eta = {eta:g}) | {cols} |" if eta is not None else f"| code | d | CSS | {cols} |")
        lines.append("|---|---|---|---|" + "---|" * len(tiers) if eta is not None else "|---|---|---|" + "---|" * len(tiers))
        for cid in ids:
            r = out["codes"][cid]
            c = r["cases"][name]
            src = f"{r['source_ler'].get(float(eta), float('nan')):.4f}" if eta is not None else None
            vals = " | ".join(f"{c[t]['p_round']:.5f}" for t in tiers)
            lines.append(f"| {cid} | {r['d']} | {'yes' if r['css'] else 'no'} | {src} | {vals} |" if eta is not None else f"| {cid} | {r['d']} | {'yes' if r['css'] else 'no'} | {vals} |")
        if eta is not None:
            xs = [out["codes"][cid]["source_ler"].get(float(eta)) for cid in ids]
            for t in tiers:
                ys = [out["codes"][cid]["cases"][name][t]["p_round"] for cid in ids]
                ok = [i for i, x in enumerate(xs) if x is not None]
                lines.append(f"\nSpearman(source LER, {t} p_round) over {len(ok)} codes: {spearman([xs[i] for i in ok], [ys[i] for i in ok]):.2f}")
    return "\n".join(lines)


def write_report(out_dir: Path) -> Path:
    out = json.loads((out_dir / "sat_codes.json").read_text())
    for rec in out["codes"].values():  # JSON turns the float keys of the source rates into strings
        rec["source_ler"] = {float(k): v for k, v in rec["source_ler"].items()}
    text = f"""# SAT-designed codes under the memory objective

Status: complete for the selection below. Every number comes from
`experiments/sat_codes/runs/sat_codes.json`; this file regenerates with
`python -m autoqec_sim.experiments.sat_codes report`. Code:
`src/autoqec_sim/experiments/sat_codes.py`.

Source: DalFavero et al., arXiv:2608.23460, `experiments/surface/` of
github.com/bdalfavero/satisfying-codes: 100,000 SAT-sampled 9-qubit
surface-like codes scored by exact maximum-likelihood logical error rate
under code-capacity biased noise (`p_X = eta P`, `p_Y = p_Z = P`) in the Z
basis, against the rotated surface code and its XZZX deformation. The
selection is the twelve best distinct codes at eta = 10 and three of the
worst. The source LER column is theirs; the p_round columns are
autoqec-sim's circuit-level memory objective, {out['rounds']} rounds, under a
depolarizing profile with a biased idle channel whose Z rate is eta times
each of the X and Y rates (the codes are Hadamard-conjugated on every qubit
so their X tailoring becomes Z tailoring), and under the package's Forte
preset with sequential timing, a noise model assembled from public
specifications rather than a device measurement. Distances are recomputed
exactly here.

Two structural differences between the two scores. The source's channel is
a single biased code-capacity channel with an exact maximum-likelihood
decoder; here only the idle channel carries the bias, while gate, reset,
and measurement errors stay unbiased, so the effective bias a code sees
over a round is weaker, and the decoder is the tier's (a low-weight
maximum-likelihood proxy, or sampling with the exact decoder where the
detector count allows). The sampled tier stops at the shot and failure
counts in the JSON, so differences below about ten percent are within its
noise; the proxy tier is deterministic.
{report_tables(out)}
"""
    path = out_dir.parents[0] / "REPORT.md"
    path.write_text(text)
    return path


def make_figure(out_dir: Path = DEFAULT_OUT, out_stem: Path | str = "docs/assets/figures/sat_codes"):
    """(a) The source's code-capacity logical error rate at bias 10 against
    the circuit-level per-round rate of the sampled tier under the matching
    biased idle profile; (b) the same circuit-level rate per case for every
    code: the source's twelve best, its three worst, and the two references."""
    from . import figstyle

    plt = figstyle.apply()
    out = json.loads((out_dir / "sat_codes.json").read_text())
    codes = out["codes"]
    # the selection: the source's twelve best distinct codes at bias 10 and three of its worst
    ranked = sorted((c for c in codes if c not in ("rotated_surface", "xzzx")), key=lambda c: codes[c]["source_ler"]["10.0"])
    best, worst = ranked[:12], ranked[12:]
    cases = [(f"depol_idle_eta_{int(e)}", f"idle bias {int(e)}") for e in ETAS] + [("forte_sequential", "Forte preset")]
    cases = [(key, label) for key, label in cases if key in codes["xzzx"]["cases"]]
    tier = "sampled" if "sampled" in codes["xzzx"]["cases"][cases[0][0]] else "proxy"

    def rate(c, key):
        return codes[c]["cases"][key][tier]["p_round"]

    groups = [("source's best 12", best, figstyle.SERIES_COLORS[0], "o"), ("source's worst 3", worst, figstyle.SERIES_COLORS[1], "s"),
              ("rotated surface", ["rotated_surface"], figstyle.SERIES_COLORS[2], "D"), ("XZZX", ["xzzx"], figstyle.SERIES_COLORS[3], "^")]
    yticks = [4e-4, 6e-4, 1e-3, 1.5e-3]
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE))
    ax = axes[0]
    key10 = "depol_idle_eta_10"
    for label, members, color, marker in groups:
        xs = [codes[c]["source_ler"]["10.0"] for c in members]
        ys = [rate(c, key10) for c in members]
        ax.plot(xs, ys, marker, color=color, label=label, alpha=0.85, linestyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.008, 0.25)
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1", "0.2"])
    ax.set_xticks([], minor=True)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    ax.set_yticks([], minor=True)
    ax.set_xlabel("source code-capacity LER, bias 10")
    ax.set_ylabel(f"circuit-level p_round ({tier} tier)")
    ax.set_title("idle bias 10, three rounds")
    ax.legend(loc="upper center")
    figstyle.panel_label(ax, "a")
    ax = axes[1]
    offsets = {"source's best 12": -0.18, "source's worst 3": 0.18, "rotated surface": 0.0, "XZZX": 0.0}
    for label, members, color, marker in groups:
        for i, (key, _) in enumerate(cases):
            ys = [rate(c, key) for c in members]
            ax.plot([i + offsets[label]] * len(ys), ys, marker, color=color, alpha=0.85, linestyle="none", label=label if i == 0 else None)
        if len(members) == 1:
            ax.plot(range(len(cases)), [rate(members[0], key) for key, _ in cases], "-", color=color, linewidth=figstyle.THIN)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([label.replace("idle ", "") for _, label in cases])
    ax.set_yscale("log")
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    ax.set_yticks([], minor=True)
    ax.set_ylabel(f"circuit-level p_round ({tier} tier)")
    ax.set_title("every code, per noise case")
    figstyle.panel_label(ax, "b")
    fig.tight_layout()
    return figstyle.save(fig, out_stem)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--selection", default=str(SELECTION))
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--tiers", nargs="+", default=["proxy", "sampled"])
    r.add_argument("--shots", type=int, default=20_000)
    r.add_argument("--min-failures", type=int, default=50)
    r.add_argument("--max-shots", type=int, default=200_000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--quiet", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    f = sub.add_parser("figure")
    f.add_argument("--out", default=str(DEFAULT_OUT))
    f.add_argument("--stem", default="docs/assets/figures/sat_codes")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "figure":
        print("wrote", *make_figure(out, a.stem))
        return
    if a.cmd == "run":
        run(out, Path(a.selection), a.rounds, a.tiers, a.shots, a.min_failures, a.max_shots, a.seed, not a.quiet)
    print("wrote", write_report(out))


if __name__ == "__main__":
    main()
