Two variants make the bound a property of the trained policy, and they are the
two that enforce it in the action space rather than price it in the reward;
the one that does it on both 12-qubit cells is the greedy action mask on top
of the `terminal` reward (`mask_terminal`).

The action mask alone (`mask`) does it on `all_to_all:12` on all three seeds:
with the greedy `w*` bound enforced at every step every terminal state has `w*
<= 4` by construction (100 percent of the at-distance terminal states inside
the bound on every seed, all 10,864 terminal states of the three
last-25-update windows at `w* = 4` exactly), and the policy still learns
distance 3 inside the constrained set as its typical output: 91, 68, and 89
percent of the terminal states of the last 25 updates at distance 3 (seeds 0,
1, 2; the `unnormalized` reference on the same cell: 50, 54, and 62 percent,
none inside), 769, 1,523, and 2,248 distinct feasible `[[12, 2, 3]]` codes per
run (the reference: 0, 0, and 2 feasible episodes, single samples), the first
feasible code at 81k to 121k env steps, and the at-distance fraction over a
25-update window first passing one half at 0.68M to 1.02M env steps (the
reference: 0.97M to 1.04M). The seed-1 run dips in its last updates (episode
success 0.63 over the last 25 updates against 0.91 and 0.96; entropy back up
to 0.4 from 0.15), so the interval is wide; the median is 89 percent. On
`square:3x4` the same mask traps: every terminal state of the last 25 updates
sits exactly at `w* = 4` with distance 1, 45 percent of the two-qubit actions
masked, two feasible codes in the whole run.

The mask on top of the `terminal` reward (`mask_terminal`, added after the
first results, seed 0) removes the lattice trap: on `square:3x4` 98 percent of
the terminal states of the last 25 updates are at distance 3, every one at `w*
= 4`, 166 distinct feasible codes, 37,213 feasible episodes, the at-distance
fraction passing one half at 0.49M env steps (the `unnormalized` reference on
this cell: 0.63M; the plain mask never passes 1 percent), 26 percent of the
two-qubit actions masked at the end (the plain mask: 45 percent, the policy
sitting on the boundary of the feasible set). The two runs share the
constrained set, so the difference is the reward: a dense negative per-step
count that is small on the sparse distance-1 groups inside the set, against an
episode-ending bonus of 1 that only a candidate earns (and under the mask
every candidate is feasible, `h = 0`). On `all_to_all:12` the combination also
does at least as well as the plain mask: 95 percent of the terminal states of
the last 25 updates at distance 3, every one at `w* = 4`, 2,151 distinct
feasible codes, the at-distance fraction passing one half at 0.67M env steps
(the plain mask, seeds 0 to 2: 0.68M to 1.02M), 40 percent of the two-qubit
actions masked at the end. So on seed 0 `mask_terminal` meets the pre-set
criterion on both 12-qubit cells (a majority of the at-distance terminal
states inside the bound, here all of them, with 95 and 98 percent of all
terminal states at distance 3); its seeds 1 and 2 and the `[[16, 4, 3]]` cell
are pending.

The two reward-only variants fail, each in an identifiable way:

- `shaped`. Pricing the excess in reward units reproduces the penalty form's
  trap at every `c` run: the policy leaves the distance-3 states it had begun
  to find (the at-distance fraction peaks in the first third of the run and
  falls to zero) and settles in sparse groups of distance 1, where the shaping
  term is zero and the Knill-Laflamme count is small. The ratio that traps is
  `c r0` undetected errors per unit of excess (3.75 at `c = 0.05`), between
  the unnormalized form's 0.25 (no trap, no sparsity) and the penalty form's
  157 (trap); the earlier scale sweep (0.5 and 1 error per unit) had already
  cost distance without buying `w*`. On `all_to_all:12` the trap is the same
  at `c` = 0.05 and 0.2 (peak at-distance fraction 17 and 8 percent at 0.24M
  and 0.34M env steps, then 2,362 and 2,519 of the last 2,560 terminal states
  at distance 1).
- `terminal` without the mask. The count-unit signal is learned, and fast: the
  policy ends nearly every episode at a candidate within 0.5M env steps, the
  entropy collapses below 0.2, and the terminal states concentrate on one
  code. But the signal it optimizes is `h`, the number of heavy generators,
  not `w*`: on `all_to_all:12` the code it collapses onto has `h = 2` and `w*
  = 7` (reward 0.8), a code the `unnormalized` form never produces (its codes
  have `w*` 5 to 6 and `h` 3 to 10); on `square:3x4` it collapses onto `h =
  3`, `w*` 5 to 6. The step from `h = 2` or `3` to `h = 0` is one or two
  feasible codes in some 50,000 to 70,000 candidates, too rare against a
  collapsed policy. A count-unit signal is not a substitute for a bottleneck
  signal, but it is the right terminal signal once the mask has made every
  candidate feasible.

Against these the `unnormalized` reference behaves as in the earlier report
(half to three quarters of the terminal states at distance 3, `w*` 5 to 6, no
state inside the bound in the last 25 updates), and doubling its budget to 3M
env steps on `square:3x4` does not move the `w*` distribution (the at-distance
fraction peaks at 94 percent mid-run and the codes stay at `w*` 5 to 6).

Cost of the mask: 155 to 178 env steps per second against 224 to 306 for the
reference on `all_to_all:12` (1.3 to 2.0 times the wall time per env step; the
132 CNOT trials cost about 2,900 s of a run, the rest of the gap is contention
with three or four concurrent runs on the pod), and 431 to 601 against 815 on
`square:3x4` (1.4 to 1.9 times), for a policy that reaches the bound where the
reference reaches it only by sampling.

`DEFAULT_LDPC_FORM` stays `unnormalized`: the mask is an environment
constraint, not a reward form, and on its own it fails on the lattice cell.
The mask is exposed as `ldpc.greedy_action_mask`, the combination as the
`mask_terminal` variant of the experiment; the regression's
`LdpcSpec(mode="mask")` remains the exact per-step constraint of the `n <= 9`
targets. A 5x5 rediscovery under `w* <= 4` on the GPU is not yet justified:
`mask_terminal` has one seed per cell, the `[[16, 4, 3]]` cell (24 generators
at `n = 25` are above the exact range, so the greedy bound would be the
success test too) has not been run, and the mask costs one greedy evaluation
per CNOT direction (80 on the 5x5 lattice) per step, which `gpu_scale` does
not implement (`--ldpc-mode mask` there is the exact per-step feasibility test
in the environment, a different mechanism). The order is: seeds 1 and 2 of
`mask_terminal` on both 12-qubit cells and seed 0 on `square:4x4` (on the
pod's CPU, about 1 h and 2.5 h per run at three concurrent), then the wiring
of `greedy_action_mask` and the `terminal` form into `gpu_scale`, then the 5x5
run.

What it would take to get `w* <= 4` learned on a lattice, from these runs: a
hard constraint in the action space plus a reward whose only gradient is
toward distance. The evidence is the pair of lattice runs with the same
constrained set and different rewards: under the `unnormalized` reward the
masked policy settles on the sparse distance-1 groups inside the set (all
2,560 terminal states at `w* = 4`, distance 1), because the dense per-step
count is already small there and the mask removes the dense codes that the
reward's gradient led to on the free cell; under the `terminal` reward the
same masked policy reaches distance 3 in 98 percent of its episodes, because
the only reward above the per-step negatives is the bonus at a candidate and
every candidate the mask admits is feasible. Pricing sparsity in the reward
fails in both directions (the penalty and `shaped` forms trap at distance 1;
the `unnormalized` and `terminal` forms learn dense codes), and the reward-
only `terminal` form shows the bonus alone finds the dense codes first. What
remains open for larger lattices is whether the constrained set stays
connected enough for exploration: on `square:3x4` 26 percent of the CNOT
directions are masked at the end and the policy still finds 166 distinct
feasible codes; at `[[16, 4, 3]]` and beyond the greedy bound also masks
feasible moves (it is an upper bound on `w*`), so the exact feasibility test
(`n - k <= 14`) or a wider gate set (CZ, SWAP) may be needed to keep the set
traversable.

Scaling in n on the all-to-all graph (the connectivity of a trapped-ion
chain), `mask_terminal`, seed 0, batch 1024, 3M env steps, on the pod's A40
shared with another agent's GPU stream: negative at `n = 16` and `n = 20`.
`[[16, 4, 3]]` on `all_to_all:16` (240 CNOT directions, exact `w*`) ends with
8,821 of 10,240 terminal states at distance 1 and 1,419 at distance 2, none at
distance 3 at any update of the run, 57 percent of the two-qubit actions
masked at the end; `[[20, 4, 3]]` on `all_to_all:20` (380 CNOT directions,
greedy bound as the success test) ends with 4,270 at distance 1 and 3,922 at
distance 2, none at distance 3, 59 percent masked. Every terminal state is
inside the mask's bound by construction; at `n = 20` the terminal statistic's
own greedy bound (from the tableau generators, without the mask's extra
reduction step) places 37 percent of them at `w*` 5 to 6, which is the looser
of two valid upper bounds, so the true `w*` of those states is at most 4. The
same variant on `square:4x4` (`[[16, 4, 3]]`, run by another session on the
pod's CPU at batch 256 with the tableau-copy mask, 3M steps) is negative too:
no distance-3 state, 42 percent masked. Two confounds against the `n = 12`
runs: at batch 1024 a 3M-step run is 293 (`n = 16`) or 183 (`n = 20`) PPO
updates against 367 for the 1.5M-step runs at batch 256, and the masked
fraction of two-qubit actions grows with `n` (26 and 40 percent at `n = 12`,
57 to 59 at `n = 16` to `20`), so the constrained set the policy can explore
shrinks as the action space grows. Cost on the shared A40: 823 and 520 env
steps per second, the mask (numpy, on the host) 41 and 65 percent of the wall
time and the PPO update 44 and 30 percent, the reward under 1 percent. The two
`[[25, 1]]` cells are queued on the pod behind the RIS stream (they start one
at a time once no `ris_gpu` process remains) and are not reported here.

For the lattice hypothesis above this adds the size condition: the mask plus
the terminal bonus works where the constrained set is still rich in distance-3
codes reachable in the episode length (`n = 12`, both graphs) and fails where
the mask removes most two-qubit moves before the policy has found any (`n =
16` to `20`, both graphs). Next: the exact feasibility test in the mask at `n
= 16` (12 generators are within the exact range, so the mask can stop
forbidding feasible moves), a longer episode, and batch 256 at the same env-
step budget to separate the update count from the size effect.

Exact feasibility mask at `n = 16` (the "Exact feasibility mask" section):
negative on both graphs, and it closes the over-masking hypothesis on the
all-to-all graph. With `greedy_action_mask(method="exact")` every two-qubit
action the greedy bound rejects gets the exact rank test, so the mask
forbids exactly the moves that leave the feasible set. On `all_to_all:16`
that admits 4 to 7 percent of the rejected pairs and moves the masked
fraction from 57 to 61 percent down to 53 to 58: the greedy bound was
already nearly exact there, and the policy still never reaches distance 3
as a typical outcome (one feasible `[[16, 4, 3]]` in 3M env steps on seed
0, none on seeds 1 and 2), though it gets closer than the greedy runs (51,
1, and 18 percent of its terminal states at distance 2 at the end against
6 percent for the greedy control at the same batch). On `square:4x4` the
exact test admits 12 to 30 percent of the rejected pairs and the masked
fraction drops from 42 to 31 to 39 percent, but every run ends with all or
all but 15 of its 2,560 last-window terminal states at distance 1 and no
distance-3 state at any update. The greedy control
at batch 256 is as negative as the earlier batch-1024 run, so the update
count was not the confound. The mask costs 0.6 s per env step on
`all_to_all:16` (27 percent of the wall time) and 0.09 s on `square:4x4` (9
percent), 1.1 to 1.25 times the greedy runs' wall. Conclusion for the
lattice hypothesis: at `n = 16` the bound's slack is not what blocks
distance 3, the constrained set under a 40-gate H, S, CNOT episode is; `n =
20` (`n - k = 16`, beyond the exact range) would need a different route
regardless of the test: a gate set that can move weight inside the set (CZ,
SWAP), a longer episode, or a curriculum from the distance-2 groups the
exact-mask policy does reach.
