# PPO at GPU scale: noise-aware 4x4 with the equivariant policy, and [[25,1,5]] on the 5x5 lattice

Status: 2 runs recorded. Every number comes from a JSON under
`/private/var/folders/y6/ht74t4s93l55nf4mqctl4qw40000gn/T/pytest-of-vincent.russo/pytest-91/test_smoke_surface5x5_sparse_c0/` or one of its extra directories (named in the last column of each table, relative to `/private/var/folders/y6/ht74t4s93l55nf4mqctl4qw40000gn/T/pytest-of-vincent.russo/pytest-91/test_smoke_surface5x5_sparse_c0/`;
by default the subdirectories ['final_mode'] hold a second mode of a target run with its own `--out-dir`); the tables regenerate with
`uv run --extra rl python -m experiments.gpu_scale.run report`. Code:
`experiments/gpu_scale/run.py`; pod commands in `docs/gpu.md`.

This report extends, and does not replace, `experiments/noise_aware/REPORT.md` (the 45-run
noise-aware experiment, whose 4x4 cells all failed with the flat MLP at 256 episodes) and
`experiments/scaling/REPORT.md` (the equivariant policy reaching [[16,1,4]] at batch 256).
The 4x4 numbers here are the same reward, profiles, calibration, and code analysis with
`policy = "auto"` (equivariant), a larger batch, and a larger budget; the known-code
baselines at n = 16 are read from `experiments/noise_aware/runs/baselines_4x4.json` and not recomputed.

## What ran where

Devices: cpu (arm, torch 2.14.0). On `cuda` the policy network, the PPO update, and the state reward (exact distance up to d* - 1, fractional term, padding penalty) run on the GPU through `backends.torch_backend`; the numpy environment, the observation encoding, the terminal terms of the noise-aware reward (stim fault propagation and exact w*, cached per gate list), the candidate tracker, and every verification run on the host. `reward share` and `update share` are the fractions of the training wall time spent in the reward (host-side terminal terms included) and in the PPO update; the rest is the numpy environment step, the observation encoding, and the policy forward pass. On `cpu` every component is the numpy path of the earlier experiments.

| target | profile | seed | device | torch | policy | parameters | batch | budget (env steps) | env steps | updates | wall (s) | env steps/s | reward share | update share | wall cap hit | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| surface5x5_sparse | - | 0 | cpu (arm) | 2.14.0 | equivariant | 128,515 | 32 | 1,024 | 1,024 | 2 | 1 | 984 | 0% | 60% | no | `run_surface5x5_sparse_seed0.json` |
| surface5x5_sparse | - | 0 | cpu (arm) | 2.14.0 | equivariant | 128,515 | 32 | 512 | 512 | 1 | 0 | 1,471 | 0% | 100% | no | `final_mode/run_surface5x5_sparse_seed0.json` |

## Target A: noise-aware 4x4 with the equivariant policy

No noise4x4 run recorded yet.

## Target B: [[25,1,5]] on the 5x5 lattice

No surface5x5 run recorded yet.

## Target C: [[25,1,5]] on the 5x5 lattice under the hard constraint w* <= 4

Target [[25,1,5]] on `square:5x5` with gates H, CNOT, 20-gate episodes, reward `min(d, 5)/5 - 1 - q_1 + step_penalty, q_1 = fraction of weight-1 Paulis in S; success iff the state reward is 0 and w* <= 4`. A run stops at its first success; the code is the shortest encoder among the successes of the stopping update, replayed and verified independently of the reward (exact distance by normalizer enumeration, weight enumerators of S and N(S), exact w* up to 6 by syndrome collisions and the greedy bound above, CSS test). `rotated surface` means the enumerators of S and N(S) equal those of the rotated surface code [[25,1,5]], the match criterion of the rediscovery regression (necessary for equivalence under qubit permutations and local Cliffords, not a proof of it).

Hard LDPC constraint w* <= 4, tested exactly by `gpu_scale.SparsityCheck` (the stock `ldpc.ldpc_feasible` falls back to a greedy bound above 14 generators and calls the rotated surface code itself infeasible). Mode `final`, `mask`: in `mask` mode a gate that would take the group above the bound is replaced by a no-op step, so every visited state is sparse and `actions reverted` measures how often the policy pushes against the bound; in `final` mode the path is free and only a sparse target-distance state ends the episode, with the step penalty as the incentive. The penalty form of the constraint is not used (it trapped every run at n = 12 to 16, `experiments/scaling/REPORT.md`).

Outcome: negative. No run produced a state with exact d = 5 and w* <= 4: `final` seed 0 reached d = 5 in 0 episodes over 512 env steps but none passed the exact w* <= 4 test; `mask` seed 0 reached terminal distance 1 at most in 1,024 env steps with 0.0% of actions reverted by the bound; the runs were stopped after seed 0.

### Runs

| seed | device | policy | parameters | batch | budget (env steps) | env steps | episodes | updates | wall (s) | env steps/s | LDPC mode | actions reverted | check time share | success | first success (env steps) | update | wall to first success (s) | largest terminal d seen | terminal d counts (d = 0..5) | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | cpu | equivariant | 128,515 | 32 | 512 | 512 | 0 | 1 | 0 | 1,471 | final | 0 | 0% | no | none | none | none | 0 | [0, 0, 0, 0, 0, 0] | `final_mode/run_surface5x5_sparse_seed0.json` |
| 0 | cpu | equivariant | 128,515 | 32 | 1,024 | 1,024 | 32 | 2 | 1 | 984 | mask | 0 (0.0% of actions) | 40% | no | none | none | none | 1 | [0, 32, 0, 0, 0, 0] | `run_surface5x5_sparse_seed0.json` |

### Codes

| seed | outcome | exact d | no weight-1 stabilizer | w* | w* <= 4 | max degree (greedy basis) | CSS | rotated surface enumerators | gates (CNOT), depth | verify time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | no terminal state recorded | | | | | | | | | |
| 0 | no success; best terminal state d = 1 (cap 5), padding -0.213 | 1 | no | 4 | yes | 4 | yes | no | 20 (17), 6 | |

## Caveats

- Three seeds per cell; env-step counts and masses are three samples, not a distribution.
- The noise4x4 cross-evaluation is only over seeds whose best code reached d = 4; a profile with fewer
  such seeds has a smaller sample. The known-code rows are polished encoders optimized per profile
  (see the 4x4 section of `experiments/noise_aware/REPORT.md` for how much the polishing did).
- The surface5x5 and surface5x5_sparse runs stop at their first success, so success rate after training
  and code diversity are not measured; `w*` is exact up to 6 (syndrome collisions) and the greedy
  bound of `autoqec_sim.ldpc` above that (marked `bound`).
- surface5x5_sparse in `mask` mode restricts the search to groups with w* <= 4 at every step. The
  standard-form encoders of the rotated surface code (routed or not) leave that set early, because the
  boundary X checks have no exclusive pivot and the construction uses dense reduced generators; whether
  a lattice circuit reaches the surface code through sparse intermediate groups is not established, so
  a failure of this target does not show that no such circuit exists.
- CPU-side work (numpy environment, terminal terms, verification) is unchanged by `--device cuda`;
  the `reward share` column shows how much of the wall time it still is.

## Regenerate

```
uv run --extra rl python -m experiments.gpu_scale.run matrix --device cuda
uv run --extra rl python -m experiments.gpu_scale.run report
```

One cell:

```
uv run --extra rl python -m experiments.gpu_scale.run run --target noise4x4 --profile ion --seed 0 --device cuda
uv run --extra rl python -m experiments.gpu_scale.run run --target surface5x5 --seed 0 --device cuda
uv run --extra rl python -m experiments.gpu_scale.run run --target surface5x5_sparse --seed 0 --device cuda --ldpc-mode mask
```
