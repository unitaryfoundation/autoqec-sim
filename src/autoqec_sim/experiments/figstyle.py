# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Shared matplotlib style for every figure of the package.

Scripts import this module, call :func:`apply` once, size their canvas
with :data:`WIDTH` and one of the standard heights, label panels with
:func:`panel_label`, and save with :func:`save`. Nothing about fonts,
grids, legends, markers, or colors is set in the plotting scripts
themselves. Figures are generated at the document text width (6.5 in)
and are meant to be included at that width.
"""

from __future__ import annotations

from pathlib import Path

WIDTH = 6.5  # inches, the document text width
HEIGHT_SINGLE = 2.4  # one row of panels
HEIGHT_DOUBLE = 4.4  # two rows of panels
MARKER = 3.5
LINE = 1.0
THIN = 0.7

# Tol bright palette (colorblind safe), one color per noise profile.
PROFILE_COLORS = {"board": "#4477AA", "ion": "#EE6677", "heron": "#228833"}
PROFILE_LABELS = {
    "board": "board depolarizing",
    "ion": "Forte ion chain, sequential",
    "heron": "Heron heavy-hex rates",
}
NEUTRAL = "#777777"
# Tol bright, for series that are not noise profiles (seeds, variants).
SERIES_COLORS = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377")
DISTANCE_COLORS = {1: "#BBBBBB", 2: "#4477AA", 3: "#EE6677", 4: "#228833"}


def apply():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": LINE,
            "lines.markersize": MARKER,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "pdf.fonttype": 42,
        }
    )
    return plt


def panel_label(ax, letter: str):
    """Uniform panel label: bold lowercase letter in parentheses at the
    top-left corner outside the axes."""
    ax.text(
        -0.12,
        1.05,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(fig, stem: str | Path):
    """Write ``stem.png`` and ``stem.pdf``."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")
