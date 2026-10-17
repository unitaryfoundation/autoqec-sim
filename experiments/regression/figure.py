# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Rediscovery cost of the two agents, from the full-budget regression JSONs.

For each target both agents rediscover, the states evaluated up to a verified
rediscovery: the evolutionary search's rollouts times the episode cap, and
PPO's environment steps, both from the records in this directory. The wall
time of each run is written next to its marker. Output goes to
``docs/assets/figures/rediscovery.{png,pdf}`` for the README and the docs.

    uv run python -m experiments.regression.figure
"""

from __future__ import annotations

import json
from pathlib import Path

from autoqec_sim.regress import TARGETS

from ..common import figstyle

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "docs" / "assets" / "figures" / "rediscovery"

LABELS = {
    "five_qubit": "[[5,1,3]], all-to-all",
    "steane": "Steane [[7,1,3]], all-to-all",
    "rotated_surface_9": "rotated surface [[9,1,3]], 3x3 lattice",
}


def _seconds(s: float) -> str:
    return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s"


def main() -> None:
    ea = json.loads((HERE / "regression.json").read_text())["targets"]
    ppo = json.loads((HERE / "regression_ppo.json").read_text())["targets"]
    rows = []
    for name, label in LABELS.items():
        e, p = ea[name], ppo[name]
        ea_states = e["total_evaluations"] * TARGETS[name].max_steps
        ppo_states = p.get("total_env_steps") or p["result"]["env_steps"]
        rows.append((label, ea_states, e["wall_time_s"], ppo_states, p.get("wall_time_s") or p["result"]["wall_time_s"]))

    plt = figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, figstyle.HEIGHT_SINGLE))
    ys = range(len(rows))[::-1]
    c_ea, c_ppo = figstyle.SERIES_COLORS[0], figstyle.SERIES_COLORS[1]
    for y, (label, ea_states, ea_wall, ppo_states, ppo_wall) in zip(ys, rows, strict=True):
        ax.plot([ppo_states, ea_states], [y, y], color=figstyle.NEUTRAL, linewidth=figstyle.THIN, zorder=1)
        ax.plot(ea_states, y, "o", color=c_ea, markersize=figstyle.MARKER + 2, zorder=2)
        ax.plot(ppo_states, y, "o", color=c_ppo, markersize=figstyle.MARKER + 2, zorder=2)
        ax.annotate(_seconds(ea_wall), (ea_states, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7, color=c_ea)
        ax.annotate(_seconds(ppo_wall), (ppo_states, y), xytext=(-7, 0), textcoords="offset points", va="center", ha="right", fontsize=7, color=c_ppo)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(2e4, 4e6)
    ax.set_xlabel("states evaluated up to a verified rediscovery (full budget, seed 0)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=c_ea, label="evolutionary search")
    ax.plot([], [], "o", color=c_ppo, label="PPO")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(loc="upper right", handletextpad=0.3)
    png, pdf = figstyle.save(fig, OUT)
    print(png, pdf)
    for label, ea_states, ea_wall, ppo_states, ppo_wall in rows:
        print(f"{label:40s} evolutionary {ea_states:>9,d} states {ea_wall:6.1f} s | PPO {ppo_states:>9,d} states {ppo_wall:5.1f} s | ratio {ea_states / ppo_states:4.1f}x")


if __name__ == "__main__":
    main()
