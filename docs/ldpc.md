# LDPC constraint

`ldpc.sparsest_basis_stats` computes, for each batch element, the exact
minimum bottleneck generator weight `w*` (the smallest `w` such that the
stabilizer elements of weight at most `w` generate the group) and the
maximum qubit degree of a sparsest basis, by bit-packed GF(2) elimination
over the enumerated group for `n - k <= 14` (about 4 ms at `n = 9`), with a
greedy pairwise reduction as the bound for larger codes. The environment
takes `ldpc_w_max` with three modes: `terminate` (episode ends on a
violating state with `ldpc_violation_reward`), `mask` (the violating
action reverts to a no-op; `feasible_action_mask()` gives agents the mask
to apply to logits), and `final` (only success is gated). `ldpc_penalty`
is the soft form for the reward. `w_max = 3` provably excludes every
5-qubit distance-3 code and empirically every 7-qubit one the search
reaches; `w_max = 4` admits [[5,1,3]] and Steane and excludes Shor's code
(`w* = 6`).

Above `n = 9` the penalty form fails (`experiments/scaling/REPORT.md`):
the normalized Knill-Laflamme term pays `1/630` to `1/1128` per detected
error against 0.25 per unit of `w*` excess, and PPO converges to sparse
groups of distance 1. `ldpc.LdpcObjective(n, k, d, w_max, form=...)` is
the reward and success criterion of an LDPC-constrained target in one of
four forms (`ldpc.LDPC_FORMS`), all built from the same base terms
(Knill-Laflamme term over the Paulis of weight below `d`, padding penalty)
with the exact `w*` evaluated only on candidate states (every error
detected, no idle qubit) and the greedy bound elsewhere:

| form | reward on every state | success |
|---|---|---|
| `penalty` | normalized base minus `scale * excess` (the regression's form) | reward 0 |
| `unnormalized` | one unit per undetected error and per weight-1 stabilizer, minus `scale * excess`, the sum divided by the initial state's undetected count; one undetected error costs `1 / scale` units of `w*` excess | reward 0 |
| `final` | the normalized base alone (hard final-state constraint) | candidate with exact `w* <= w_max` |
| `schedule` | the normalized base until `activate()`, then the penalty form (the trainer switches once the success rate at the target distance reaches a threshold) | any candidate before the switch, reward 0 after |

`regress.LdpcSpec(w_max, mode="objective", form=...)` puts one of them on a
regression target (`make_reward` returns the objective, `env_kwargs`
hands its `success` method to the environment as `success_fn`); the
`n <= 9` targets keep `mode="penalty"`. `experiments.ldpc_scaling` compares
the forms on `[[12, 2, 3]]` and `[[16, 4, 3]]` with `w_max = 4` on the
all-to-all and the square graph, three seeds each, equivariant policy,
1.5M env steps or 20 minutes per run (`experiments/ldpc_scaling/REPORT.md`,
runs in `results/ldpc_scaling/`). Result (36 runs plus 4 scale variants, 0.62M to 1.5M env steps each on a
shared machine): the `unnormalized` form is the default
(`ldpc.DEFAULT_LDPC_FORM`). It learns distance where the penalty form
never did (61 to 69 percent of terminal states at distance 3 on five of
six `[[12, 2, 3]]` runs, 64 percent on the best 4x4-lattice seed) and it is
the only form that produced codes within the bound: 3 of 12 runs, all
`[[12, 2, 3]]` (303k and 354k env steps all-to-all, 950k on the lattice,
23 to 31 gates, `w* = 4`), each a single episode after which the policy
keeps producing distance-3 codes with `w*` 5 to 6 (`n = 12`) or 7 to 8
(`n = 16`). The `final` form reaches distance 3 (14 to 68 percent at
`n = 12`) and never a sparse code (0 of 12); the `schedule` form switches
its penalty on at 0.63M to 0.91M env steps on `[[12, 2, 3]]` and then falls
into the penalty form's distance-1 trap (0 of 12); at `[[16, 4, 3]]`
all-to-all no form reaches distance 3 in the budget. Pricing the excess at
0.5 or 1 unit per error instead of 0.25 costs distance and buys no `w*`.
So the constraint above `n = 9` is half fixed: the distance-1 trap is gone
and the bound is reachable, but the sparsity part is not yet a property of
the trained policy.

`experiments.ldpc_sparsity` takes the sparsity part further
(`experiments/ldpc_sparsity/REPORT.md`, runs in `results/ldpc_sparsity/`;
the 12-qubit cells, `[[16, 4, 3]]` pending): a dense per-step term in
reward units (`LdpcObjective(form="shaped", shaping=c)`, `c` = 0.05, 0.1,
0.2) traps at distance 1 like the penalty form; a count-unit terminal
signal (`form="terminal"`: any candidate ends the episode with
`1 - h / (n - k)`, `h` the heavy generators of the sparsest basis,
`ldpc.heavy_generator_count`) is learned fast but optimizes `h` rather
than `w*` and collapses onto one dense code; the action mask
`ldpc.greedy_action_mask` (two-qubit gates whose resulting greedy `w*`
bound exceeds `w_max` are masked out of the logits) makes every terminal
state feasible by construction and, on `all_to_all:12`, still learns
distance 3 as the policy's typical output on three seeds (68 to 91
percent of terminal states at distance 3 over the last 25 updates, 769 to
2,248 distinct feasible codes per run) while trapping at `w* = 4`,
distance 1 on `square:3x4`; the mask on top of the `terminal` reward
(`mask_terminal`, seed 0) does it on both cells (95 and 98 percent at
distance 3, all at `w* = 4`). The mask costs 1.3 to 2 times the wall
time per env step. The default form is unchanged. On the all-to-all
graph at `n = 16` and `20` (`--device cuda`, batch 1024, 3M env steps)
the same variant never reaches distance 3 (the report's scaling section),
and on `square:4x4` at batch 256 neither: the greedy bound over-masks as
`n` grows (36 to 63 percent of the two-qubit actions at the end). The
exact feasibility test in the mask (`greedy_action_mask(method="exact")`,
`--mask-method exact`; the greedy bound as a pre-filter, then the exact
rank test on the pairs it rejects, `n - k <= 15`) does not change that
at `n = 16` in 3M env steps on three seeds per graph: on `all_to_all:16`
it admits 4 to 7 percent of the rejected moves (the greedy bound was
nearly exact) and the policy reaches one feasible `[[16, 4, 3]]` in 9M env
steps; on `square:4x4` it admits 12 to 30 percent and every run ends at
distance 1. The mask costs 0.6 s per env step on the all-to-all cell (27
percent of the wall time) and 0.08 s on the lattice (the report's "Exact
feasibility mask" section).

```
uv run --extra rl python -m autoqec_sim.experiments.ldpc_sparsity run --variant mask_terminal --cell n12_k2_square --seed 0
uv run --extra rl python -m autoqec_sim.experiments.ldpc_sparsity report
```

```
uv run --extra rl python -m autoqec_sim.experiments.ldpc_scaling run --form unnormalized --target n12_k2_all_to_all --seed 0
uv run --extra rl python -m autoqec_sim.experiments.ldpc_scaling report
```
