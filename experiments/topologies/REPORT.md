# Three-topology run matrix: PPO on square, triangular, and heavy-hex lattices

Status: complete for the cells listed. Every number comes from a JSON under
`results/topologies/` (named in the last column of each table) and the tables
regenerate with `uv run --extra rl python -m autoqec_sim.experiments.topologies report`.
Code: `src/autoqec_sim/experiments/topologies.py`.

## Summary

This is the PPO agent of `autoqec_sim.agents.ppo` by itself on three lattices, not the LLM meta-agent that milestone M2 also names. Cells: square n = 9: 3 of 3 seeds, first success at 9,472 to 12,288 env steps; triangular n = 9: 3 of 3 seeds, first success at 6,144 to 8,192 env steps; heavy_hex n = 12: 3 of 3 seeds, first success at 49,408 to 74,240 env steps; square n = 16: 3 of 3 seeds, first success at 100,864 to 112,896 env steps; triangular n = 16: 3 of 3 seeds, first success at 62,720 to 99,840 env steps; heavy_hex n = 18: 3 of 3 seeds, first success at 307,200 to 344,832 env steps. The first success came within the first three PPO updates on square n = 9 (update 3), triangular n = 9 (update 2), that is under a policy still close to its random initialization, so those cells measure reachability under near-uniform sampling of lattice gates, not learning. Undetected fault mass of the agent's encoder relative to the routed rotated surface encoder on the same graph: square n = 9, heron: 1.60x to 2.02x; square n = 9, board: 2.18x to 4.47x; triangular n = 9, heron: 1.09x to 1.46x; triangular n = 9, board: 2.36x to 7.79x; square n = 16, heron: 0.67x to 0.71x; square n = 16, board: 0.33x to 1.12x; triangular n = 16, heron: 0.60x to 0.86x; triangular n = 16, board: 0.88x to 2.27x. Where the surface code of the cell's size has distance 4, the baseline protects against more than the objective asks and its routed encoder is 53 to 74 gates long against the agent's at most 50. Runs used 2 torch threads each, sequentially, with a 1-minute load of 10 to 13 at run start on a shared workstation; total wall time 13 min over 18 runs, longest run 197 s, so no run reached its step budget or the 12 min wall cap.

## Setup

Objective, identical on every topology: a `[[n, 1, d >= 3]]` code with no idle
qubit (the padding penalty of the regression, minus the fraction of weight-1
Paulis in the stabilizer group), H and CNOT on the lattice edges only, 50-gate
episode cap, level distance reward `min(d, 3) / 3 - 1` plus the padding penalty
(`experiments.scaling.LevelDistanceReward`; reward 0 iff the target is met).
`PPOConfig` defaults with `policy = "auto"` (flat MLP below 12 qubits, per-qubit
equivariant network from 12 on), 256 parallel episodes, seeds 0, 1, 2.
Budgets: 2,000,000 env steps for the small tier, 4,000,000 for the
large tier, 12 min wall cap per run; a run stops at its first success.

Graphs. Square and triangular lattices have exact sizes at both tiers (`3x3`,
`4x4`). Heavy-hex sizes are fixed by the honeycomb construction of
`noise_model.topology.heavy_hex` (12, 18, 21, 25, ... qubits), so the small
tier takes `heavy_hex:1x1` (12 qubits; a single hexagon with a qubit on every
edge is a 12-cycle, every qubit of degree 2) and the large tier takes
`heavy_hex:2x1` (18 qubits, two stacked hexagons, two qubits of degree 3).
The heavy-hex small cell therefore has more qubits than the square and
triangular small cells, and its large cell has two more than theirs.

Metrics. `w*` is the exact minimum bottleneck generator weight of the
stabilizer group and `max degree` the maximum qubit degree of the sparsest
basis (`autoqec_sim.ldpc`, exact up to `n - k = 17`). Noise masses are the
undetected fault mass of the encoder under `presets.heron(graph=None)` placed
on the cell's graph (IBM Heron r2 rates, parallel timing) and under
`presets.board()` (one depolarizing rate per two-qubit gate and idle qubit per
layer), both at scale 1 and unnormalized (`circuit_noise.undetected_fault_mass`),
with the total mass beside it. The rotated surface code column is the same
mass for the standard-form CSS encoder of the rotated surface code of the
cell's size, routed onto the cell's graph, best over the logical input qubit
and pivot choice, without the deletion polish of the noise-aware experiment;
it exists only where `n` is a perfect square on the square and triangular
lattices (the triangular lattice contains the square edges).

## Matrix

| tier | topology | graph | n | edges | max degree | actions | policy | parameters | seeds | successes | env steps to first success | update of first success | wall to first success (s) | total wall (s) | env steps used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | square | `square:3x3` | 9 | 12 | 4 | 33 | mlp | 157,730 | 0, 1, 2 | 3 of 3 | 9,472, 9,472, 12,288 | 3, 3, 3 | 0.4, 0.3, 0.3 | 0.4, 0.3, 0.3 | 12,288, 12,288, 12,288 |
| small | triangular | `triangular:3x3` | 9 | 16 | 6 | 41 | mlp | 159,786 | 0, 1, 2 | 3 of 3 | 8,192, 7,424, 6,144 | 2, 2, 2 | 0.2, 0.2, 0.2 | 0.2, 0.2, 0.2 | 8,192, 8,192, 8,192 |
| small | heavy_hex | `heavy_hex:1x1` | 12 | 12 | 2 | 36 | equivariant | 121,859 | 0, 1, 2 | 3 of 3 | 63,488, 74,240, 49,408 | 16, 19, 13 | 14.6, 17.4, 11.8 | 14.6, 17.4, 11.8 | 65,536, 77,824, 53,248 |
| large | square | `square:4x4` | 16 | 24 | 4 | 64 | equivariant | 123,907 | 0, 1, 2 | 3 of 3 | 100,864, 112,896, 112,128 | 25, 28, 28 | 39.6, 41.2, 44.2 | 39.6, 41.2, 44.2 | 102,400, 114,688, 114,688 |
| large | triangular | `triangular:4x4` | 16 | 33 | 6 | 82 | equivariant | 123,907 | 0, 1, 2 | 3 of 3 | 99,840, 62,720, 75,776 | 25, 16, 19 | 59.7, 43.9, 50.0 | 59.7, 43.9, 50.0 | 102,400, 65,536, 77,824 |
| large | heavy_hex | `heavy_hex:2x1` | 18 | 18 | 3 | 54 | equivariant | 124,931 | 0, 1, 2 | 3 of 3 | 307,200, 340,480, 344,832 | 75, 84, 85 | 65.6, 175.5, 196.8 | 65.6, 175.5, 196.8 | 307,200, 344,064, 348,160 |

## Best code per run

| tier | topology | n | seed | code | w* | max degree | CSS | gates (CNOT), depth | heron undetected / total | board undetected / total | surface code heron undetected (gates) | surface code board undetected (gates) | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | square | 9 | 0 | [[9,1,3]] | 6 | 5 | no | 37 (28), 17 | 0.00744 / 0.137 | 0.005 / 0.116 | 0.00378 (20) | 0.00113 (20) | `run_square_n9_seed0.json` |
| small | square | 9 | 1 | [[9,1,3]] | 5 | 3 | no | 37 (24), 14 | 0.00605 / 0.114 | 0.00247 / 0.089 | 0.00378 (20) | 0.00113 (20) | `run_square_n9_seed1.json` |
| small | square | 9 | 2 | [[9,1,3]] | 6 | 4 | no | 48 (33), 23 | 0.00763 / 0.171 | 0.00507 / 0.159 | 0.00378 (20) | 0.00113 (20) | `run_square_n9_seed2.json` |
| small | triangular | 9 | 0 | [[9,1,3]] | 4 | 4 | no | 32 (25), 14 | 0.00365 / 0.119 | 0.00273 / 0.094 | 0.00327 (14) | 0.000933 (14) | `run_triangular_n9_seed0.json` |
| small | triangular | 9 | 1 | [[9,1,3]] | 4 | 4 | no | 29 (22), 16 | 0.00357 / 0.116 | 0.00727 / 0.115 | 0.00327 (14) | 0.000933 (14) | `run_triangular_n9_seed1.json` |
| small | triangular | 9 | 2 | [[9,1,3]] | 6 | 4 | no | 24 (17), 9 | 0.00478 / 0.0786 | 0.0022 / 0.057 | 0.00327 (14) | 0.000933 (14) | `run_triangular_n9_seed2.json` |
| small | heavy_hex | 12 | 0 | [[12,1,3]] | 6 | 4 | no | 48 (37), 12 | 0.00473 / 0.158 | 0.0024 / 0.096 | n/a | n/a | `run_heavy_hex_n12_seed0.json` |
| small | heavy_hex | 12 | 1 | [[12,1,3]] | 6 | 4 | no | 40 (28), 11 | 0.00478 / 0.129 | 0.0016 / 0.092 | n/a | n/a | `run_heavy_hex_n12_seed1.json` |
| small | heavy_hex | 12 | 2 | [[12,1,3]] | 6 | 4 | no | 43 (27), 14 | 0.0053 / 0.138 | 0.00247 / 0.125 | n/a | n/a | `run_heavy_hex_n12_seed2.json` |
| large | square | 16 | 0 | [[16,1,3]] | 6 | 5 | no | 44 (33), 13 | 0.00496 / 0.172 | 0.0048 / 0.164 | 0.00746 (74) | 0.00427 (74) | `run_square_n16_seed0.json` |
| large | square | 16 | 1 | [[16,1,3]] | 7 | 4 | no | 41 (35), 13 | 0.00528 / 0.182 | 0.00227 / 0.167 | 0.00746 (74) | 0.00427 (74) | `run_square_n16_seed1.json` |
| large | square | 16 | 2 | [[16,1,3]] | 6 | 5 | no | 38 (29), 8 | 0.00497 / 0.13 | 0.0014 / 0.09 | 0.00746 (74) | 0.00427 (74) | `run_square_n16_seed2.json` |
| large | triangular | 16 | 0 | [[16,1,3]] | 12 | 4 | no | 40 (31), 13 | 0.00399 / 0.171 | 0.00287 / 0.168 | 0.00661 (53) | 0.00327 (53) | `run_triangular_n16_seed0.json` |
| large | triangular | 16 | 1 | [[16,1,3]] | 6 | 4 | no | 45 (36), 17 | 0.00568 / 0.212 | 0.00453 / 0.227 | 0.00661 (53) | 0.00327 (53) | `run_triangular_n16_seed1.json` |
| large | triangular | 16 | 2 | [[16,1,3]] | 5 | 4 | no | 46 (37), 16 | 0.00457 / 0.208 | 0.0074 / 0.21 | 0.00661 (53) | 0.00327 (53) | `run_triangular_n16_seed2.json` |
| large | heavy_hex | 18 | 0 | [[18,1,3]] | 6 | 3 | no | 50 (34), 12 | 0.00357 / 0.181 | 0.001 / 0.166 | n/a | n/a | `run_heavy_hex_n18_seed0.json` |
| large | heavy_hex | 18 | 1 | [[18,1,3]] | 8 | 4 | no | 30 (22), 10 | 0.00359 / 0.135 | 0.00213 / 0.15 | n/a | n/a | `run_heavy_hex_n18_seed1.json` |
| large | heavy_hex | 18 | 2 | [[18,1,3]] | 5 | 3 | no | 47 (34), 11 | 0.00565 / 0.169 | 0.00387 / 0.151 | n/a | n/a | `run_heavy_hex_n18_seed2.json` |

## Weight enumerators

| tier | topology | n | seed | code | A(S), weights 0..n | B(N(S)), weights 0..n |
|---|---|---|---|---|---|---|
| small | square | 9 | 0 | [[9,1,3]] | [1, 0, 2, 8, 8, 12, 54, 96, 63, 12] | [1, 0, 2, 26, 44, 90, 222, 318, 243, 78] |
| small | square | 9 | 1 | [[9,1,3]] | [1, 0, 5, 2, 15, 26, 31, 86, 76, 14] | [1, 0, 5, 27, 39, 131, 223, 233, 244, 121] |
| small | square | 9 | 2 | [[9,1,3]] | [1, 0, 4, 2, 10, 30, 44, 78, 69, 18] | [1, 0, 4, 20, 46, 132, 212, 252, 249, 108] |
| small | triangular | 9 | 0 | [[9,1,3]] | [1, 0, 1, 1, 9, 31, 55, 79, 62, 17] | [1, 0, 1, 11, 39, 127, 235, 289, 236, 85] |
| small | triangular | 9 | 1 | [[9,1,3]] | [1, 0, 2, 2, 12, 18, 62, 86, 51, 22] | [1, 0, 2, 14, 52, 102, 238, 298, 219, 98] |
| small | triangular | 9 | 2 | [[9,1,3]] | [1, 0, 4, 0, 22, 0, 100, 0, 129, 0] | [1, 0, 4, 24, 22, 192, 100, 408, 129, 144] |
| small | heavy_hex | 12 | 0 | [[12,1,3]] | [1, 0, 6, 6, 31, 40, 108, 148, 327, 456, 526, 374, 25] | [1, 0, 6, 32, 43, 208, 356, 880, 1383, 1840, 1846, 1136, 461] |
| small | heavy_hex | 12 | 1 | [[12,1,3]] | [1, 0, 5, 2, 26, 32, 98, 148, 413, 496, 409, 346, 72] | [1, 0, 5, 16, 56, 168, 322, 968, 1417, 1848, 1833, 1096, 462] |
| small | heavy_hex | 12 | 2 | [[12,1,3]] | [1, 0, 9, 6, 32, 52, 126, 160, 285, 396, 537, 410, 34] | [1, 0, 9, 36, 48, 256, 434, 904, 1281, 1632, 1781, 1268, 542] |
| large | square | 16 | 0 | [[16,1,3]] | [1, 0, 9, 10, 47, 100, 233, 486, 993, 1848, 3139, 4982, 6205, 6564, 5451, 2394, 306] | [1, 0, 9, 30, 71, 316, 673, 1842, 4113, 7816, 13939, 20258, 24821, 25212, 19299, 10062, 2610] |
| large | square | 16 | 1 | [[16,1,3]] | [1, 0, 12, 17, 67, 130, 330, 575, 967, 1820, 2920, 4895, 6453, 6274, 5090, 2673, 544] | [1, 0, 12, 36, 135, 392, 902, 2324, 4535, 8208, 13312, 18412, 23929, 25384, 19726, 10780, 2984] |
| large | square | 16 | 2 | [[16,1,3]] | [1, 0, 7, 6, 32, 80, 167, 378, 838, 1920, 3445, 5162, 6632, 6544, 4733, 2294, 529] | [1, 0, 7, 15, 67, 210, 591, 1509, 3629, 8012, 14221, 20905, 25753, 25282, 18973, 9603, 2294] |
| large | triangular | 16 | 0 | [[16,1,3]] | [1, 0, 41, 4, 317, 152, 1125, 812, 2355, 2064, 2859, 3228, 3407, 1752, 6279, 8372, 0] | [1, 0, 41, 84, 317, 1424, 1445, 6596, 6483, 12352, 13611, 12828, 12303, 13872, 17991, 18380, 13344] |
| large | triangular | 16 | 1 | [[16,1,3]] | [1, 0, 12, 8, 66, 108, 256, 572, 1016, 1976, 3228, 4800, 5990, 6172, 5272, 2748, 543] | [1, 0, 12, 28, 102, 356, 820, 2136, 4456, 8328, 13956, 19660, 23786, 23508, 19004, 11520, 3399] |
| large | triangular | 16 | 2 | [[16,1,3]] | [1, 0, 8, 3, 37, 50, 158, 425, 865, 2028, 3484, 5157, 6603, 6210, 4574, 2511, 654] | [1, 0, 8, 9, 67, 186, 598, 1687, 3815, 8172, 14140, 20711, 25813, 24634, 18310, 10137, 2784] |
| large | heavy_hex | 18 | 0 | [[18,1,3]] | [1, 0, 12, 8, 96, 88, 524, 584, 2234, 2648, 6980, 9176, 16040, 20936, 24868, 23000, 14781, 9096, 0] | [1, 0, 12, 36, 96, 468, 748, 3348, 5050, 14788, 23588, 45228, 65448, 88636, 95684, 83228, 60861, 26412, 10656] |
| large | heavy_hex | 18 | 1 | [[18,1,3]] | [1, 0, 20, 10, 172, 158, 828, 1050, 2558, 4062, 6156, 10702, 13564, 19530, 22500, 21006, 18009, 9018, 1728] | [1, 0, 20, 48, 212, 672, 1600, 4240, 8446, 16448, 29272, 43600, 64004, 77472, 84096, 81648, 59049, 38016, 15444] |
| large | heavy_hex | 18 | 2 | [[18,1,3]] | [1, 0, 7, 16, 47, 120, 267, 688, 1453, 2808, 5445, 9776, 15749, 22632, 27305, 24592, 14558, 4904, 704] | [1, 0, 7, 33, 75, 281, 827, 2085, 5217, 11469, 22525, 41187, 67465, 93163, 103977, 90647, 57738, 23279, 4312] |

## Rotated surface baselines

| topology | graph | n | code | candidates | heron: undetected / total, gates (CNOT), depth, logical qubit | board: undetected / total, gates (CNOT), depth, logical qubit | file |
|---|---|---|---|---|---|---|---|
| square | `square:3x3` | 9 | rotated surface [[9,1,3]] (exact d = 3) | 111 | 0.00378 / 0.0767, 20 (16), 9, 4 | 0.00113 / 0.052, 20 (16), 8, 4 | `baseline_square_n9.json` |
| triangular | `triangular:3x3` | 9 | rotated surface [[9,1,3]] (exact d = 3) | 111 | 0.00327 / 0.0451, 14 (10), 5, 4 | 0.000933 / 0.031, 14 (10), 5, 4 | `baseline_triangular_n9.json` |
| heavy_hex | `heavy_hex:1x1` | 12 | n/a (no rotated surface code of this size on this graph) | | | | `baseline_heavy_hex_n12.json` |
| square | `square:4x4` | 16 | rotated surface [[16,1,4]] (exact d = 4) | 192 | 0.00746 / 0.464, 74 (67), 41, 10 | 0.00427 / 0.582, 74 (67), 41, 10 | `baseline_square_n16.json` |
| triangular | `triangular:4x4` | 16 | rotated surface [[16,1,4]] (exact d = 4) | 192 | 0.00661 / 0.321, 53 (46), 29, 10 | 0.00327 / 0.411, 53 (46), 29, 10 | `baseline_triangular_n16.json` |
| heavy_hex | `heavy_hex:2x1` | 18 | n/a (no rotated surface code of this size on this graph) | | | | `baseline_heavy_hex_n18.json` |

## Caveats

- RL agent only. The milestone's LLM meta-agent (proposing rewards, action
  sets, or targets) is not part of this matrix; every cell uses one fixed
  objective and the default agent.
- The objective is distance 3 with no idle qubit, not a code tuned for the
  noise profile; the noise masses are measured after the fact on the
  encoder the agent happened to produce (the first success, shortest
  circuit among the successes of the stopping update). The surface code
  baseline is optimized over encoder choices for each profile, so the
  comparison favours the baseline.
- A run stops at its first success, so success rate after training and
  the diversity of codes are not measured here.
- Heavy-hex cells are not size-matched to the other topologies (12 and 18
  qubits against 9 and 16); the 12-qubit heavy-hex graph is a cycle.
- Three seeds per cell; the env-step counts are three samples, not a
  distribution.
- No noise preset refused any of the graphs.
- Platform: `macOS-26.5.1-arm64-arm-64bit`.

## Regenerate

```
uv run --extra rl python -m autoqec_sim.experiments.topologies matrix
uv run --extra rl python -m autoqec_sim.experiments.topologies report
```

One cell, one seed, and one baseline:

```
uv run --extra rl python -m autoqec_sim.experiments.topologies run --topology heavy_hex --target-n 16 --seed 0
uv run --extra rl python -m autoqec_sim.experiments.topologies baseline --topology square --target-n 9
```
