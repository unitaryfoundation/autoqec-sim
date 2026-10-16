Reading. The exact mask does not make `[[16, 4, 3]]` learnable on either
graph in 3M env steps: of the eight exact-mask runs (three seeds per graph at batch 256, one per graph at batch 1024), one produced a distance-3 terminal state, once, and every run ends with all or nearly all of its terminal states below distance 3. What it changes is the
distance distribution below the target and, on the lattice, the size of the
constrained set.

- `all_to_all:16`. The greedy bound is nearly exact here: the exact test
  admits 4 to 7 percent of the pairs the bound rejects (17.6M of 421M,
  28.4M of 433M, and 26.7M of 418M over seeds 0, 1, and 2), so the masked
  fraction of two-qubit actions ends at 55, 58, and 53 percent against 57,
  61, and 56 percent for the greedy pre-filter and 58 percent for the greedy
  control at the same batch. The constrained set itself is what
  the policy cannot traverse to distance 3 in 40 gates, not the bound's
  slack. Inside it the exact-mask policy gets further than the greedy one:
  on seed 0, 51 percent of the terminal states of the last 25 updates at
  distance 2 (1,316 of 2,560; greedy control at batch 256: 155 of 2,560;
  greedy at batch 1024: 1,419 of 10,240), a trailing-window fraction at
  distance 2 or more of 0.6 to 0.85 between 1.5M and 2.6M env steps
  (greedy: at most 0.4), and one distance-3 terminal state in the whole
  run, at 1.57M env steps, a feasible `[[16, 4, 3]]` code (the greedy
  runs: none). Seeds 1 and 2 end lower, 15 and 459 of 2,560 at distance 2
  (1 and 18 percent) and no distance-3 state in either run, with 14,784
  and 11,730 distance-2 terminal states over the run against 10,692 for
  the greedy control; the three seeds share the same masked fraction and
  differ in where the policy settles. The unmasked probe of the final
  seed-0 policy (256 sampled episodes, no mask) ends 6 at distance 3, none
  feasible.
- `square:4x4`. The exact test admits 30, 12, and 29 percent of the
  rejected pairs (seeds 0, 1, and 2; 18.7M of 61.8M on seed 0), so the
  final masked fraction is 31, 39, and 32 percent against 48, 43, and 45
  percent for the pre-filter and 42 percent for both greedy controls; the
  peak over the run drops from 84 to 53 percent (seed 0; seed 2 peaks at 92
  percent for the pre-filter and 63 for the mask). The wider set does not
  reach distance 3 either: 2,285, 395, and 2,633 distance-2 terminal states
  over the run (the greedy basis control: 255, the tableau control: 753),
  at most 51, 29, and 88 in one update (greedy: 16), and the last 25
  updates at distance 1 on every state (seeds 0 and 1) or all but 15 (seed
  2). The policy collapses (entropy 0.55, 0.44, and 0.75 at the end) onto
  sparse distance-1 groups, as the greedy runs do.
- Batch. The greedy control at batch 256 (733 updates, the update count of
  the 12-qubit runs) is as negative as the batch-1024 run (184 updates):
  the update-count confound of the scaling section is closed. The exact
  mask at batch 1024 (184 updates) is negative on both graphs as well. On
  `all_to_all:16` it ends with 26 percent of the last-window terminal
  states at distance 2 (2,666 of 10,240; the greedy batch-1024 run: 14
  percent), 54 percent of two-qubit actions masked against 57 for the
  pre-filter, and no distance-3 state at any update. On `square:4x4` the
  policy drifts into a region where almost nothing is masked (0.6 percent
  of two-qubit actions at the end, 4 percent for the pre-filter, after a
  peak of 54 percent at update 50), with all 10,240 last-window terminal
  states at distance 1 and entropy 1.69: the plain distance-1 trap, without
  the sparse-boundary behavior of the batch-256 runs.

Cost. On the pod's CPU at 6 threads with four runs concurrent and a host
load of 30 to 190 from other tenants, the exact mask took 0.60 s per env
step on `all_to_all:16` (0.20 to 0.22 s greedy pre-filter, 0.38 to 0.40 s
exact test, 33,600 to 34,700 tests per call, 78 to 82 percent of them
distinct after merging identical states, cache hit rate 9 to 21 percent),
27 percent of the wall time against 54 to 59 percent for the PPO update;
the runs took 1.25 to 1.27 times the wall of the greedy control (27,719 to
28,135 against 22,212 s). On `square:4x4` it took 0.08 s per env step
(5,000 to 5,300 tests per call, 33 to 42 percent distinct, cache hit rate
49 to 61 percent), 9 to 10 percent of the wall time, 0.9 to 1.1 times the
greedy control's wall (9,502 to 12,088 against 10,971 s). The benchmark table above (random
policy under the exact mask, so fewer rejected pairs than a trained policy
produces) puts the mean at 0.53 s per call at batch 256 and 2.25 s at
batch 1024 on the all-to-all cell; the two reductions the plan named for a
cost above 2 s per step are both in the implementation already (all
actions of all states go through one batched elimination, and only the
pairs the greedy bound rejects are tested, which are exactly the CNOTs
that leave a heavy generator), so the batch-1024 runs were made as they
are. In the runs the exact mask took 1.66 s per env step on `all_to_all:16`
at batch 1024 (0.69 s pre-filter, 0.96 s exact test, 126,000 tests per
call, 99 percent distinct, cache hit rate 7 percent), 29 percent of the
wall time (17,949 s against 3,665 s for the greedy batch-1024 run on the
GPU-hosted policy of the scaling section, so the two are not comparable in
wall), and 0.24 s per env step on `square:4x4` (14,400 tests per call),
16 percent of the wall time. The wall-time comparisons
between runs are confounded by the pod's load: the same greedy
configuration on `square:4x4` took 2,840 s alone at 8 threads on
2026-09-22 and 10,971 s here.

Probe column. The probe of the final policy samples without the mask, so
for the masked variants it describes the unmasked policy, not the
constrained one; its distance-3 states need not be feasible.

What this says for `n = 20`. The exact test is available up to `n - k =
15`; `[[20, 4, 3]]` has 16 generators and would need either a smaller `k`
or a different test. But the all-to-all result makes the range question
secondary: at `n = 16` the greedy bound is already within 2 percentage
points of the exact mask, so the exact test would not change the `n = 20`
outcome either. What blocks distance 3 at `n >= 16` is the constrained set
under a 40-gate episode with H, S, and CNOT, not the bound's slack. The
routes left are a wider gate set inside the set (CZ, SWAP, so that a step
can move weight without passing through a heavy generator), a longer
episode, and a curriculum from the distance-2 groups the exact-mask policy
does reach on all-to-all (half its terminal states at the end).
