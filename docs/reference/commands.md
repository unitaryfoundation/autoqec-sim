# Commands

Three console scripts install with the package. The first runs on numpy
alone; the other two need the `rl` extra.

## autoqec-regress

The known-code rediscovery regression with the evolutionary search.
`--budget reduced` is what the test suite runs; `--budget full` produced
`experiments/regression/regression.json`. [Rediscovery regression](../regression.md) lists
the targets and [Reproduce](../reproduce.md) the expected outputs.

```
usage: autoqec-regress [-h] [--budget {full,reduced}] [--seed SEED]
                       [--targets [TARGET ...]] [--out OUT] [--verbose]

options:
  --budget {full,reduced}
  --seed SEED
  --targets [TARGET ...]  five_qubit, steane, rotated_surface_9,
                          rotated_surface_9_ldpc, rotated_surface_9_ldpc_penalty,
                          steane_ldpc_w4, seven_qubit_w3_excluded,
                          five_qubit_w3_excluded
  --out OUT               results JSON path
  --verbose
```

## autoqec-ppo

The same regression with PPO in place of the evolutionary search, plus the
PPO-only 4x4 rotated-surface target `rotated_surface_16` at the `full`
budget. `--compare` takes the evolutionary results JSON for a side-by-side
table.

```
usage: autoqec-ppo [-h] [--budget {full,reduced}] [--seed SEED]
                   [--targets [TARGET ...]] [--out OUT] [--compare COMPARE]
                   [--threads THREADS] [--verbose]

options:
  --budget {full,reduced}
  --seed SEED
  --targets [TARGET ...]  the autoqec-regress targets and rotated_surface_16
  --out OUT               results JSON path
  --compare COMPARE       evolutionary results JSON
  --threads THREADS       torch and BLAS thread cap (default 6)
  --verbose
```

## autoqec-schedule

CX-layer schedule synthesis for the syndrome-extraction circuit of a fixed
code from a qldpc-challenge board. The command scores the board builder's
own schedules as baselines, trains a PPO policy on the exact circuit fault
distance, and confirms the best schedule at `rounds = d` with the board's
search tier. On the [[49,13,4]] circuit it raised the fault distance from
2 to 3.

```
usage: autoqec-schedule [-h] [--updates UPDATES] [--batch BATCH]
                        [--baseline BASELINE] [--tier {exact,ris}]
                        [--ris-trials RIS_TRIALS] [--rounds ROUNDS]
                        [--interleave] [--slack SLACK]
                        [--depth-penalty DEPTH_PENALTY]
                        [--edge-order {fixed,shuffle}] [--seed SEED]
                        [--confirm-trials CONFIRM_TRIALS] [--out OUT]
                        [--board BOARD]
                        slug

positional arguments:
  slug                  board code slug, for example 49-13-4

options:
  --rounds ROUNDS       extraction rounds scored (default: screen at 2,
                        confirm at d)
  --tier {exact,ris}    fault-distance tier
  --out OUT             output directory (default schedule_out/<slug>)
  --board BOARD         qldpc-challenge checkout (default: QLDPC_CHALLENGE or
                        a sibling directory)
```

`autoqec_sim.schedule_env` and `autoqec_sim.schedule_cli` are documented
in the [Python API](../api/schedule.md).
