#!/usr/bin/env python3
"""Thin wrapper kept for the experiment report; the runner is autoqec_sim.schedule_cli.

    uv run --frozen python experiments/schedule/run.py 49-13-4 --updates 40
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from autoqec_sim.schedule_cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] + ([] if any(a.startswith("--out") for a in sys.argv[1:]) else ["--out", str(HERE / "out" / sys.argv[1])])))
