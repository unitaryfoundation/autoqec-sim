# Experiments

The experiments that produced the reports in this directory. They build on
the installed package (`autoqec_sim`) but are not part of it: a pip install
of `autoqec-sim` does not include this directory, and the code here runs
from a clone of the repository, from its root.

## Layout

```
experiments/
  __init__.py              makes python -m experiments.<name>.run work from the repository root
  common/figstyle.py       the one matplotlib style every figure shares (canvas width, heights, panel labels, colors)
  tests/                   the experiment tests, collected with the package suite by uv run pytest
  <name>/run.py            the experiment: its cells, its subcommands, and its report and figure writers
  <name>/REPORT.md         the write-up, regenerated from runs/ by the report subcommand (tracked)
  <name>/figures/          the figures, regenerated from runs/ by the figure or report subcommand (tracked)
  <name>/runs/             raw output: one JSON per run and logs/ (not tracked)
  regression/              the rediscovery regression JSONs and figure.py, which draws the README's rediscovery figure
  schedule/                the schedule-synthesis runs (run.py wraps autoqec-schedule)
```

Some directories carry extra scripts next to `run.py`: `assemble_report.py`
(`ldpc_scaling`, `ldpc_sparsity`) rebuilds the report with every number
asserted against the JSONs, `local_search.py` (`sat_population`) is the
SAT-constrained local search, and `heavy_hex_overnight.sh` launches the
unattended heavy-hex run. They run as modules the same way
(`python -m experiments.ldpc_scaling.assemble_report`).

## Running an experiment

From the repository root:

```
uv run python -m experiments.<name>.run --help
```

lists the subcommands. Every experiment has `report`, which reads
`experiments/<name>/runs/*.json` and rewrites `REPORT.md`; the newer ones
have `figure`, which draws the figures into `figures/`, and the older ones
draw them from `report`. The run subcommands (`run`, `matrix`, `baselines`,
`benchmark`, and so on, by experiment) write the JSONs the report reads. For
the memory objective runs, for example:

```
uv run python -m experiments.memory_objective.run benchmark
uv run python -m experiments.memory_objective.run report
```

The PPO-based experiments (`noise_aware`, `scaling`, `ldpc_scaling`,
`ldpc_sparsity`, `gpu_scale`, `ion_chain`, `topologies`, `topologies_ea`,
`heavy_hex_overnight`, `memory_ppo`, `qdx_targets`, and the `ppo`
subcommand of `memory_objective`) need torch, which is the `rl` extra:

```
uv run --extra rl python -m experiments.topologies.run matrix
uv run --extra rl python -m experiments.topologies.run report
uv run --extra rl python -m experiments.topologies.run figure
```

Paths inside the experiments (`experiments/<name>/runs`, the population and
selection JSONs) are relative to the repository root, so the commands run
from there. Each `docs/experiments/<name>.md` page gives the commands that
reproduce one run of that experiment, and each `REPORT.md` states which
runs it was built from.
