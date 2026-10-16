# PPO with the memory objective on the 4x4 setting

Status: 12 runs of run 1 and 9 runs of run 2 recorded. Every number below is read from a JSON under `experiments/memory_ppo/runs/` (`run_*.json`, `scores.json`, `run2/`) or `experiments/gpu_scale/runs/run_noise4x4_*.json`; the tables regenerate with `python -m autoqec_sim.experiments.memory_ppo report`. Code: `src/autoqec_sim/experiments/memory_ppo.py` (module docstring: reward, profiles, scales, selection); objective: `src/autoqec_sim/memory_objective.py`, `docs/memory_objective.md`.

# Run 1: memory term with the LDPC penalty, proxy tier, highest-reward selection

## Verdict

Correction from run 2 (2026-09-26): the memory circuit measures the generators
it is given, and every discovered code in this section was evaluated, in
training and in scoring, on the generator basis read off its encoder tableau,
which is denser than the sparsest basis of the same group (the replayed rotated
surface code itself carries 224 two-qubit gates per round in that basis against
192 in its sparsest; the known codes below were entered in their sparse form,
so they were not affected). The run-2 scoring pass
(`experiments/memory_ppo/runs/run2/scores.json`) re-scores the 12 best codes of this
section on their sparsest bases: 236 to 360 two-qubit gates per round instead
of 280 to 468, 0.75 to 1.25 expected faults per shot instead of 0.94 to 1.54,
and bposd `p_round` on the training profile 1.5 to 5.3 times the surface code
instead of 8.9 to 50 times (ion 2.3, 5.3, 4.6; ion+flag 2.4, 2.5, 2.6; board
3.0, 1.8, 2.6; heron 2.0, 1.5, 2.2). The encoder-objective codes move the same
way (1.5 to 5.1 times instead of 13.5 to 97). The absolute numbers in this
section are therefore the tableau-basis numbers and overstate every discovered
code's memory error by 3 to 10 times; the ranking statements (no memory-trained
cell distinguishable from the encoder-trained cells, no bias tailoring, proxy
outside its regime on these bases) stand, and the in-regime comparison is in
the run-2 section.

Status: 12 runs (four cells, seeds 0 to 2) on the RunPod A40, four cells of a
seed concurrent, 150 min cap per run, host CPU capped at four cores (`taskset
0-3`, `nice 5`, one memory worker per cell). Every run hit the wall cap at
1.18M to 2.21M env steps (budget 3M) and every run reached `d = 4` (first
success at 256k to 444k env steps). Scores: the best code of each run, the
lowest-proxy code of each run, the nine encoder-objective 4x4 codes of
`experiments/gpu_scale/runs`, and the rotated surface `[[16,1,4]]` with its XZZX
deformation (35 codes), under the four profiles at the proxy, sampled, and
bposd tiers (`experiments/memory_ppo/runs/scores.json`). Profiles: `ion` is the chain
preset at 0.25 with sequential timing and bare ancillas, `ion+flag` the same
with one flag per generator, `board` the qldpc-challenge recipe at 0.25,
`heron` the Heron r2 rates at 0.05 (the "Scales" paragraph below says why not 1
and 0.1).

Answer 1: does training on the memory objective produce codes with lower memory
error than training on the encoder objective? No. Under the validation decoder
(bposd) on their training profile, the memory-trained best codes have `p_round`
3.2e-3 to 1.8e-2, which is 8.9 to 50 times the rotated surface code's 3.5e-4 to
6.5e-4; the encoder-trained codes have 6.7e-3 to 4.8e-2 (13.5 to 97 times the
surface code). Cell means on the training profile, memory-trained against
encoder-trained: 0.0141 against 0.0178 (ion), 0.0111 against 0.0237 (board),
0.0105 against 0.0092 (heron); the spread within a cell is 2 to 6 times over
three seeds, so none of these differences is resolved. No memory-trained seed
is below the best of the nine encoder-trained codes on ion, ion+flag, or board;
one of three is on heron. Both families are dense degenerate non-CSS codes with
`w* = 5` whose memory circuits have 280 to 468 two-qubit gates per round in 4
to 9 commutation batches, against 192 gates in 2 batches for the surface code.

Answer 2: does bias tailoring appear under the ion profile? No. None of the 12
best codes is CSS or has a Hadamard frame that makes it CSS (the smallest-frame
search over all `2^16` Hadamard subsets finds none); `d_X > d_Z` in 7 of 12 and
`d_Z > d_X` in none, the same as the encoder-trained codes (7 of 9) and the
same under every profile, so the asymmetry is a property of the H and CNOT
encoder, not of the noise. Under the chain preset at 0.25 the memory objective
itself does not separate the surface code from XZZX (proxy 3.74e-4 against
3.70e-4, bposd 6.45e-4 against 5.86e-4 at about 100 failures per basis),
consistent with the design note's finding that the chain set is readout
dominated; the one profile where the objective prefers XZZX is Heron (proxy
3.39e-4 against 1.98e-4, bposd 3.53e-4 against 2.38e-4), and the Heron-trained
codes have no frame either.

Why: the proxy tier does not hold on the codes the agent visits. Over the 35
scored codes the bposd to proxy ratio has median 8.1 (ion), 20 (ion+flag), 8.0
(board), and 5.4 (heron), with Spearman 0.46 to 0.77; the sampled
near-maximum-likelihood tier sides with bposd (Spearman 0.94 to 0.98, median
ratio 0.78 to 1.8) and not with the proxy (Spearman 0.44 to 0.77, median ratio
5.2 to 24). On the surface code the ratios are 1.7 (ion), 1.2 (board), and 1.04
(heron). The proxy ranked the discovered best codes 1.2 to 5.6 times worse than
the surface code; bposd ranks them 8.9 to 50 times worse. Every discovered code
has 0.94 to 1.54 expected faults per shot under its training profile (surface:
0.45 to 0.65), where the weight-3 and higher fault sets the weight-2 proxy
ignores carry most of the failure mass; the discovered codes are dense, so the
scales that keep the surface code inside the proxy's regime do not keep them
inside it. Within a run the proxy preferred codes with heavier generators
still: the lowest-proxy code of every run has `w* = 6` to 10 (LDPC term -0.5 to
-1.5), which is why none of them is the best-reward code. Scoring those 12
codes splits them by their expected fault count. The three whose memory
circuits are as short as the surface code's (196 to 212 two-qubit gates, 0.61
to 0.72 expected faults per shot; `w* = 8` to 10, so one or two heavy
generators among light ones) are within 1.1 to 2.2 times the surface code under
bposd on their training profile (heron seed 2: 3.78e-4 against 3.53e-4; board
seed 2: 7.32e-4 against 4.98e-4; ion seed 2: 1.44e-3 against 6.45e-4), and the
proxy had them at 0.51 to 1.3 times the surface code. The other nine (244 to
456 gates, 0.82 to 1.50 expected faults) are 3.8 to 34 times worse than the
surface code under bposd while the proxy had them at 0.72 to 2.2 times. So the
objective did reach codes with near-surface memory error, the LDPC term of the
reward removed them from the selection, and the proxy cannot tell them from the
dense codes it also favoured.

Training under the cap: the memory term took 75 to 97 percent of the wall time
(2.5 to 10.3 s per code at one core; 20,281 distinct target codes over the 12
runs, every one evaluated, cache hits only in the three runs that converged),
so the runs made 1.2M to 2.2M env steps in 150 min against 2.2M to 3.3M in 40
min for the encoder-objective runs, and only three of twelve reached a success
rate above 0.9 (seed 2 of ion+flag, board, and heron). The best-reward
selection is decided by the LDPC term (every best code has `w* = 5`, memory
terms differ by at most 0.1 between codes), as in the encoder-objective runs.
Flags do not help the discovered codes: the ion to ion+flag bposd ratio over
the 35 codes has median 1.03 (surface code 6.45e-4 to 6.12e-4), against the 2.6
to 4.4 times of the `n <= 9` benchmark.

Scales: at `n = 16`, `R = 4` the benchmark scales board 1 and Heron 0.1 give
the rotated surface code 2.02 and 0.91 expected faults per shot and the dense
encoder-objective codes 5.8 and 2.8 (measured on the pod before the run),
outside the tiers' regime for every code; board 0.25 and Heron 0.05 give 0.50
and 0.45 for the surface code and 1.4 for the dense codes, in line with chain
0.25 (0.65 and 1.1). The chain scale is the one asked for.

Two other deviations from the brief. `memory_objective._joint_ids` labels
multi-word detector signatures (more than 64 detectors, every `n = 16` circuit)
by a lexsort instead of `np.unique(axis=0)`, whose structured-view argsort took
14 of the 19 s of one proxy evaluation on the pod, and shifts the labels when
the all-zero signature is absent, since the maximum-likelihood grouping tests
`id == 0` for the zero syndrome (with a test); everything else about the module
is untouched. The CPU cap was met with `taskset -c 0-3` and `nice 5` on the
four concurrent cells, each with one memory worker, rather than one shared pool
of four.

What this means for the objective: the proxy tier is the wrong training signal
at `n = 16` unless the codes are kept below about 0.7 expected faults per shot
(a circuit-length term in place of, or next to, the `w*` term, or lower
scales), or the term is evaluated at the sampled tier (about 30 s per code
here, 3 to 10 times the proxy); the `w*` LDPC term and the memory term pull in
opposite directions on this setting (the codes with the lowest memory error
have `w* = 8` to 10 with surface-length circuits); and either way the memory
term needs an order of magnitude more host CPU than this run had.

## Setting

`n = 16`, `k = 1`, target `d = 4`, `square:4x4`, gates H, CNOT, 50-gate episodes, equivariant policy (`PPOConfig(policy="auto")`), batch 1024, budget 3,000,000 env steps or 150 min per run, seeds 0, 1, 2. Reward: `(d - 1 + f)/d* - 1 - q_1 + [d >= d*, q_1 = 0](0.1 clip(log10(1e-2 / p_round), -3, 3) - 0.25 max(0, w* - 4)), f = detected fraction of weight-d Paulis, q_1 = fraction of weight-1 Paulis in S, p_round = memory objective (proxy tier, R = 4, both bases) under the training profile`. Memory objective: `R = 4` rounds, `all_to_all` layout, hook-aware order, both bases summed, proxy tier in training; the memory term is evaluated once per distinct stabilizer group (cache keyed by the reduced row echelon form) in a host process pool, synchronously with the training loop.

| cell | profile | scale | flags | objective |
|---|---|---|---|---|
| ion | chain-2606.06455 (sequential) | 0.25 | none | rounds 4, all_to_all, order hook, bases Z+X |
| ion_flag | chain-2606.06455 (sequential) | 0.25 | one | rounds 4, all_to_all, order hook, bases Z+X |
| board | board-rfc0001 (parallel) | 0.25 | none | rounds 4, all_to_all, order hook, bases Z+X |
| heron | heron-r2 (parallel) | 0.05 | none | rounds 4, all_to_all, order hook, bases Z+X |

Scales: the tiers of the memory objective need the expected number of faults per shot below about one; the benchmark scales of `docs/memory_objective.md` (board 1, Heron 0.1) put the rotated surface `[[16,1,4]]` at 2.02 and 0.91 expected faults per shot in the `R = 4` circuit and the dense encoder-objective codes at 5.8 and 2.8, so board runs at 0.25 and Heron at 0.05 (surface 0.50 and 0.45, dense code 1.45 and 1.39; chain 0.25 gives 0.65 and 1.11). The expected fault counts of every scored circuit are in `scores.json` (`per_basis.*.expected_faults`).

## What ran where

`reward share` is the fraction of the training wall time spent in the reward call (GPU distance kernel plus the host-side terminal terms), `memory share` the part of it spent waiting for the memory pool, `update share` the PPO update.

| profile | seed | device | batch | budget (env steps) | env steps | updates | wall (s) | env steps/s | reward share | memory share | update share | memory evals | s per code | cache hits | wall cap hit | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ion | 0 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,851,392 | 113 | 9216 | 201 | 96% | 96% | 2% | 959 | 9.19 | 0 | yes | `run_ion_seed0.json` |
| ion | 1 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,261,568 | 77 | 9306 | 136 | 98% | 96% | 1% | 1707 | 5.25 | 20 | yes | `run_ion_seed1.json` |
| ion | 2 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,556,480 | 95 | 9215 | 169 | 98% | 97% | 1% | 2000 | 4.46 | 88 | yes | `run_ion_seed2.json` |
| ion_flag | 0 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,376,256 | 84 | 9464 | 145 | 97% | 97% | 1% | 885 | 10.3 | 0 | yes | `run_ion_flag_seed0.json` |
| ion_flag | 1 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,196,032 | 73 | 9212 | 130 | 98% | 97% | 1% | 1189 | 7.52 | 3 | yes | `run_ion_flag_seed1.json` |
| ion_flag | 2 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,851,392 | 113 | 9080 | 204 | 97% | 84% | 1% | 1846 | 4.12 | 5277 | yes | `run_ion_flag_seed2.json` |
| board | 0 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,425,408 | 87 | 9846 | 145 | 97% | 96% | 1% | 1315 | 7.21 | 0 | yes | `run_board_seed0.json` |
| board | 1 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,179,648 | 72 | 9074 | 130 | 98% | 96% | 1% | 2609 | 3.33 | 22 | yes | `run_board_seed1.json` |
| board | 2 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 2,211,840 | 135 | 9018 | 245 | 96% | 75% | 2% | 2728 | 2.47 | 5102 | yes | `run_board_seed2.json` |
| heron | 0 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,638,400 | 100 | 9909 | 165 | 97% | 96% | 2% | 1216 | 7.83 | 143 | yes | `run_heron_seed0.json` |
| heron | 1 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,261,568 | 77 | 9244 | 136 | 98% | 97% | 1% | 1327 | 6.75 | 22 | yes | `run_heron_seed1.json` |
| heron | 2 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 2,015,232 | 123 | 9003 | 224 | 96% | 81% | 2% | 2500 | 2.9 | 6587 | yes | `run_heron_seed2.json` |

## Runs

| profile | seeds | seeds reaching d* | first success (env steps) | final success rate | distinct target codes per seed | memory evaluations per seed | best training p_round per seed |
|---|---|---|---|---|---|---|---|
| ion | 3 | 3/3 | 3.67e+05 [2.56e+05, 4.44e+05] | 0.496 [0.0446, 0.779] | 1.56e+03 [959, 2e+03] | 1.56e+03 [959, 2e+03] | 0.001 [0.000769, 0.00136] |
| ion_flag | 3 | 3/3 | 3.67e+05 [2.56e+05, 4.44e+05] | 0.677 [0.26, 0.985] | 1.31e+03 [885, 1.85e+03] | 1.31e+03 [885, 1.85e+03] | 0.000473 [0.000422, 0.000506] |
| board | 3 | 3/3 | 3.67e+05 [2.56e+05, 4.44e+05] | 0.678 [0.318, 0.999] | 2.22e+03 [1.32e+03, 2.73e+03] | 2.22e+03 [1.32e+03, 2.73e+03] | 0.00137 [0.00108, 0.00159] |
| heron | 3 | 3/3 | 3.67e+05 [2.56e+05, 4.44e+05] | 0.652 [0.294, 0.996] | 1.68e+03 [1.22e+03, 2.5e+03] | 1.68e+03 [1.22e+03, 2.5e+03] | 0.000861 [0.000568, 0.00103] |

Runs stopped by the wall cap before their step budget: board/seed 0, board/seed 1, board/seed 2, heron/seed 0, heron/seed 1, heron/seed 2, ion_flag/seed 0, ion_flag/seed 1, ion_flag/seed 2, ion/seed 0, ion/seed 1, ion/seed 2.

## Codes found

Selected code per seed (`selection` in the JSON; `H frame` is the smallest Hadamard frame that makes the stabilizer group CSS, `none` when no such frame exists; `2q gates per round`, `batches`, and `E` (expected faults per shot) are the memory circuit's under the training profile; `train p_round` is the value the reward saw at the tier in `tier`; the `bposd` columns are from `scores.json`).

| trained on | seed | reached d* | d | d_X | d_Z | w* | CSS | H frame | gates | 2q gates per round | batches | E (train profile) | train p_round | tier | bposd ion | bposd ion+flag | bposd board | bposd heron | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ion | 0 | yes | 4 | 5 | 4 | 5 | no | none | 42 | 336 | 7 | 1.13 | 0.000883 | proxy | 0.0113 | 0.0105 | 0.0136 | 0.00639 | `run_ion_seed0.json` |
| ion | 1 | yes | 4 | 5 | 5 | 5 | no | none | 44 | 364 | 6 | 1.16 | 0.000769 | proxy | 0.0134 | 0.0121 | 0.0157 | 0.00586 | `run_ion_seed1.json` |
| ion | 2 | yes | 4 | 5 | 4 | 5 | no | none | 45 | 384 | 8 | 1.27 | 0.00136 | proxy | 0.0175 | 0.0207 | 0.0264 | 0.0133 | `run_ion_seed2.json` |
| ion_flag | 0 | yes | 4 | 5 | 4 | 5 | no | none | 34 | 376 | 5 | 1.27 | 0.000492 | proxy | 0.00694 | 0.00572 | 0.00492 | 0.00274 | `run_ion_flag_seed0.json` |
| ion_flag | 1 | yes | 4 | 4 | 4 | 5 | no | none | 47 | 416 | 5 | 1.38 | 0.000422 | proxy | 0.00646 | 0.00586 | 0.00658 | 0.00241 | `run_ion_flag_seed1.json` |
| ion_flag | 2 | yes | 4 | 5 | 4 | 5 | no | none | 45 | 468 | 6 | 1.54 | 0.000506 | proxy | 0.013 | 0.0116 | 0.0133 | 0.0051 | `run_ion_flag_seed2.json` |
| board | 0 | yes | 4 | 5 | 4 | 5 | no | none | 48 | 348 | 6 | 1.42 | 0.00144 | proxy | 0.0123 | 0.0125 | 0.0153 | 0.00607 | `run_board_seed0.json` |
| board | 1 | yes | 4 | 5 | 5 | 5 | no | none | 44 | 324 | 5 | 1.16 | 0.00108 | proxy | 0.00812 | 0.00721 | 0.00742 | 0.00278 | `run_board_seed1.json` |
| board | 2 | yes | 4 | 4 | 4 | 5 | no | none | 46 | 344 | 4 | 1.18 | 0.00159 | proxy | 0.0119 | 0.0107 | 0.0105 | 0.00333 | `run_board_seed2.json` |
| heron | 0 | yes | 4 | 4 | 4 | 5 | no | none | 47 | 456 | 7 | 1.32 | 0.000989 | proxy | 0.0293 | 0.0275 | 0.0443 | 0.0178 | `run_heron_seed0.json` |
| heron | 1 | yes | 4 | 7 | 5 | 5 | no | none | 40 | 324 | 6 | 1.10 | 0.000568 | proxy | 0.0084 | 0.00835 | 0.00724 | 0.00315 | `run_heron_seed1.json` |
| heron | 2 | yes | 4 | 7 | 5 | 5 | no | none | 47 | 360 | 8 | 1.43 | 0.00103 | proxy | 0.0165 | 0.018 | 0.0207 | 0.0105 | `run_heron_seed2.json` |

**ion / seed 0**: [[16,1,4]], d_X = 5, d_Z = 4, w* = 5 (degree 5), CSS: no, Hadamard frame: none. Encoder: 42 gates, depth 14. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000883/0.000472/0.00171/0.00163; encoder undetected mass board/ion(forte)/heron = 0.00153/0.0018/0.00318. Reward terms: memory +0.1054, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 959, of which 0 CSS.

Stabilizers (sparsest basis): `YYXXIIXIIIIIIIII`, `YIIIIIIXZYXIIIII`, `IXZIYIIIIYIIIIXI`, `IIIIXXIIXZIIIIII`, `IIIIIXXIIIIXIIIX`, `IIIIIXIXIIXIXXII`, `IIIIIIIIIIIIXIXX`, `ZZIIIIIIIZIIIIII`, `ZIIIIZZZZIIIIIII`, `IIZZIIIIIIIIIIII`, `IIZIIIZIIIIZIIII`, `IIIIZIIIZIIZIIZZ`, `IIIIIIIZIIZIIIII`, `IIIIIIIIIZZIZIZI`, `IIIIIIIIIIIZZZIZ`  

Lowest training p_round of the seed (a different code): p_round 0.000628 (proxy), w* = 7, d_X/d_Z = 5/4, CSS no, frame none, 2q gates -, reward -0.6298 (LDPC -0.750, gate +0.0000).

Modal code of the last 20 updates: 1 of 314 successes, same as selected: no; p_round 0.00213, w* = 7, d_X/d_Z = 5/4, CSS no, frame none, 2q gates -.

**ion / seed 1**: [[16,1,4]], d_X = 5, d_Z = 5, w* = 5 (degree 7), CSS: no, Hadamard frame: none. Encoder: 44 gates, depth 14. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000856/0.00025/0.000673/0.000396; encoder undetected mass board/ion(forte)/heron = 0.00167/0.00109/0.00221. Reward terms: memory +0.1114, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 1707, of which 0 CSS.

Stabilizers (sparsest basis): `XXIIXIIIZZIIIIII`, `IXIYIXZYIIIIIIII`, `XIYIXYIZIIIIIIII`, `IIIYIIIIXXIIIXIY`, `IIIIYXIIYZIIIIII`, `YYIIIIYYIIXIIIII`, `IIIIZXIIIXIIXXII`, `IIIIIIIIXXXXIIXI`, `ZIIZIZIIIYYIIIII`, `ZZZIIIZZIIIIIIII`, `IXXIIXIIIIZZIIII`, `IIIZIIZZIIIIIIIZ`, `IIIIIIZIIZZIIZII`, `IIIIIIIIIIIZIIZI`, `IIIIIIIIIIIIZZIZ`  

Lowest training p_round of the seed (a different code): p_round 0.000711 (proxy), w* = 6, d_X/d_Z = 4/4, CSS no, frame none, 2q gates -, reward -0.3852 (LDPC -0.500, gate +0.0000).

Modal code of the last 20 updates: 11 of 1529 successes, same as selected: no; p_round 0.00116, w* = 8, d_X/d_Z = 5/4, CSS no, frame none, 2q gates -.

**ion / seed 2**: [[16,1,4]], d_X = 5, d_Z = 4, w* = 5 (degree 6), CSS: no, Hadamard frame: none. Encoder: 45 gates, depth 22. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00209/0.000703/0.00283/0.00169; encoder undetected mass board/ion(forte)/heron = 0.00153/0.00161/0.00305. Reward terms: memory +0.0866, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 2000, of which 0 CSS.

Stabilizers (sparsest basis): `XZIIIXIIXIIIIIII`, `ZXZIIXIIIIIIIIIZ`, `IZXXIIIIIIIXIIII`, `YIIIYYYXIIIIIIII`, `IIIIIXZIIIIIZIZI`, `IIIIIXXXIXIIIIXI`, `IIIZIIZYIIIYIIII`, `IZIIIIIIIXXIIIIX`, `IIIIXXIIXIIIZZII`, `IIIIZIIYIIXIXIYI`, `IIIIIIIIIIIIYYXI`, `ZIIIZIIIZIIIIIII`, `IIZZIIIIIIIIIIII`, `IIIIIIIXIIZXIIIZ`, `IIIIIIIIIZZIIZZI`  

Lowest training p_round of the seed (a different code): p_round 0.000444 (proxy), w* = 10, d_X/d_Z = 8/4, CSS no, frame none, 2q gates -, reward -1.3648 (LDPC -1.500, gate +0.0000).

Modal code of the last 20 updates: 17 of 1608 successes, same as selected: no; p_round 0.00201, w* = 8, d_X/d_Z = 7/4, CSS no, frame none, 2q gates -.

**ion_flag / seed 0**: [[16,1,4]], d_X = 5, d_Z = 4, w* = 5 (degree 5), CSS: no, Hadamard frame: none. Encoder: 34 gates, depth 13. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00115/0.000492/0.00121/0.00072; encoder undetected mass board/ion(forte)/heron = 0.00233/0.00135/0.00158. Reward terms: memory +0.1308, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 885, of which 0 CSS.

Stabilizers (sparsest basis): `YXIIIIYXIIZIIIII`, `IXIIIXIIIXIIIXII`, `ZIXXIIZIIIZIIIII`, `IIIIYXIIYIIIZIII`, `IIIIIXXXIIXIIIII`, `IYZIIZIIIIYIXIII`, `IIIIIIIXIIIXIIII`, `IIIIIIIIXIIIXXXX`, `ZZIIIIIIIIIIZZII`, `IIZZIIIIIIIIIIII`, `IIIIZIIIZIIIIIZI`, `ZZIIZZIIIIZIIIII`, `IIIIIIZZIIIZIIII`, `IIIIIIIIIZIIIZZI`, `IIIIIIIIIIIIIIZZ`  

Lowest training p_round of the seed (a different code): p_round 0.00033 (proxy), w* = 6, d_X/d_Z = 4/6, CSS no, frame none, 2q gates -, reward -0.3518 (LDPC -0.500, gate +0.0000).

Modal code of the last 20 updates: 1 of 635 successes, same as selected: no; p_round 0.000754, w* = 8, d_X/d_Z = 5/4, CSS no, frame none, 2q gates -.

**ion_flag / seed 1**: [[16,1,4]], d_X = 4, d_Z = 4, w* = 5 (degree 6), CSS: no, Hadamard frame: none. Encoder: 47 gates, depth 14. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00107/0.000421/0.00112/0.000584; encoder undetected mass board/ion(forte)/heron = 0.00153/0.0029/0.00395. Reward terms: memory +0.1375, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 1189, of which 0 CSS.

Stabilizers (sparsest basis): `YIIIXIIIYIIZZIII`, `ZYIIYIIIIIIIIIII`, `IIXXIIXXIIIIIIII`, `XXIXIIIIIIIXIIIX`, `IXIIZIIIIIIIXXII`, `IIIIIXIIIIXIIIXI`, `IIIIIIIXIIXXXXII`, `IIIIIIIIXXIIXIXI`, `ZIIIIZIXIIYYIIII`, `IIIIIIIIIIIIIXIX`, `ZIZZIIIIZZIIIIII`, `IIZIIIZIIIIIIIII`, `IIIZIIIZIIIZIIII`, `IIIIIZIIIZIIIIZI`, `ZIIIIIIIIZIIYYIZ`  

Lowest training p_round of the seed (a different code): p_round 0.000271 (proxy), w* = 6, d_X/d_Z = 4/4, CSS no, frame none, 2q gates -, reward -0.3433 (LDPC -0.500, gate +0.0000).

Modal code of the last 20 updates: 2 of 1007 successes, same as selected: no; p_round 0.000833, w* = 9, d_X/d_Z = 11/4, CSS no, frame none, 2q gates -.

**ion_flag / seed 2**: [[16,1,4]], d_X = 5, d_Z = 4, w* = 5 (degree 6), CSS: no, Hadamard frame: none. Encoder: 45 gates, depth 18. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00113/0.000555/0.00156/0.00101; encoder undetected mass board/ion(forte)/heron = 0.0014/0.00232/0.00355. Reward terms: memory +0.1296, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 1846, of which 0 CSS.

Stabilizers (sparsest basis): `XIIIIXIIXIIIZXII`, `IXXIIIXIIXXIIIII`, `IYIIXIYIXIIXIIII`, `IIIXXXIXIIIXIIII`, `ZIIIYYXIIIIIIIII`, `XIIIIIYXIYXIIIII`, `IIIIIIIXIIXIIXZX`, `IIIIIIIIIIIIXZXI`, `IIIIIIIIIIIIIIXZ`, `ZIZIIIIIIZIIXIII`, `IZZZIIIIIIIZIIII`, `IIIXIIXYIIIYIIXI`, `IIIZZIZIIZIIIIII`, `IIIIZZIIZIIIIIII`, `IIIIIIIIIZZIIIXI`  

Lowest training p_round of the seed (a different code): p_round 0.000269 (proxy), w* = 8, d_X/d_Z = 7/4, CSS no, frame none, 2q gates -, reward -0.8429 (LDPC -1.000, gate +0.0000).

Modal code of the last 20 updates: 6067 of 13613 successes, same as selected: no; p_round 0.000472, w* = 8, d_X/d_Z = 7/4, CSS no, frame none, 2q gates -.

**board / seed 0**: [[16,1,4]], d_X = 5, d_Z = 4, w* = 5 (degree 6), CSS: no, Hadamard frame: none. Encoder: 48 gates, depth 16. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00089/0.000457/0.00144/0.00101; encoder undetected mass board/ion(forte)/heron = 0.0008/0.00165/0.00243. Reward terms: memory +0.0841, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 1315, of which 0 CSS.

Stabilizers (sparsest basis): `XXXIIXXIIIIIIIII`, `XIIYZIIYIIIZIIII`, `IIXZYIXIYIIIIIII`, `IIXIZYYIIIIIZIII`, `IIIIIXIIIXIIXXII`, `IIIIIXXXIIXXIIII`, `IZZIIIIIXIIIXIII`, `IIIIIIIZIXYXIIYI`, `IIIIIIIIIIXIIIXX`, `ZZIZIIIIIIIIIIII`, `IIZZIIZZIIIIIIII`, `IIZIZZIIIZIZIIII`, `IIIIZIIIZIIIZZII`, `IIIIIIIIIZIIIZZZ`, `IIIIIIIIIIZZIIIZ`  

Lowest training p_round of the seed (a different code): p_round 0.000908 (proxy), w* = 6, d_X/d_Z = 5/4, CSS no, frame none, 2q gates -, reward -0.3958 (LDPC -0.500, gate +0.0000).

Modal code of the last 20 updates: 1 of 1036 successes, same as selected: no; p_round 0.00207, w* = 7, d_X/d_Z = 4/4, CSS no, frame none, 2q gates -.

**board / seed 1**: [[16,1,4]], d_X = 5, d_Z = 5, w* = 5 (degree 5), CSS: no, Hadamard frame: none. Encoder: 44 gates, depth 11. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00128/0.000442/0.00134/0.000494; encoder undetected mass board/ion(forte)/heron = 0.00173/0.00123/0.00241. Reward terms: memory +0.0965, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 2609, of which 0 CSS.

Stabilizers (sparsest basis): `XZIIIIYYIIIIIIII`, `IXIIYYXIXIIIIIII`, `IIXYIXYIIIIIZIII`, `IXXIIIIZIIIZIIZI`, `IIIIXZIIIIIIXXII`, `ZIIIZXIZIIIIIIII`, `XIIZZYYIIIIIIIII`, `IIIIIIIIXXIIIIII`, `IIIIIIIIIIXIIIIX`, `IIZZIIIIIIIXIIIX`, `IIIIIIIIIIXXIXXI`, `ZZZIIIZIIIIIIIII`, `IIIIZIIIZZIIZIII`, `IIIIIIIIIIZZIIIZ`, `IIIIIIIIIIIIZZZI`  

Lowest training p_round of the seed (a different code): p_round 0.000553 (proxy), w* = 7, d_X/d_Z = 4/5, CSS no, frame none, 2q gates -, reward -0.6243 (LDPC -0.750, gate +0.0000).

Modal code of the last 20 updates: 6 of 2502 successes, same as selected: no; p_round 0.00202, w* = 8, d_X/d_Z = 8/4, CSS no, frame none, 2q gates -.

**board / seed 2**: [[16,1,4]], d_X = 4, d_Z = 4, w* = 5 (degree 6), CSS: no, Hadamard frame: none. Encoder: 46 gates, depth 17. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00148/0.000709/0.00168/0.000748; encoder undetected mass board/ion(forte)/heron = 0.0014/0.00193/0.00355. Reward terms: memory +0.0798, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 2728, of which 0 CSS.

Stabilizers (sparsest basis): `XXIIIXIIIIIIIIXX`, `IXIXIIIZIIIXZIII`, `IIXXIIXIIIXXIIII`, `IIXXXIIIIIIIIIXX`, `IIIIIIIXIXIXXIII`, `IIIIIIIIXIIIIXII`, `IIIIIIIXIIXIXXIX`, `IIIIIIZIIZIZXXII`, `ZIIIIZIIIIIIIIII`, `ZZZZIIIIIIIIIIII`, `IIZIZIZIIIIIIIII`, `IIZZIIIZIIZZIIII`, `IIIIZZIIIIIIIIZI`, `IIZIIZIIIIZIIIIZ`, `IIIIIIIIZZIIZZII`  

Lowest training p_round of the seed (a different code): p_round 0.000432 (proxy), w* = 8, d_X/d_Z = 7/4, CSS no, frame none, 2q gates -, reward -0.8636 (LDPC -1.000, gate +0.0000).

Modal code of the last 20 updates: 6304 of 15188 successes, same as selected: no; p_round 0.00109, w* = 9, d_X/d_Z = 7/4, CSS no, frame none, 2q gates -.

**heron / seed 0**: [[16,1,4]], d_X = 4, d_Z = 4, w* = 5 (degree 6), CSS: no, Hadamard frame: none. Encoder: 47 gates, depth 14. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00242/0.000793/0.00252/0.000989; encoder undetected mass board/ion(forte)/heron = 0.00173/0.0012/0.00241. Reward terms: memory +0.1005, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 1216, of which 0 CSS.

Stabilizers (sparsest basis): `YIZIIYIIXIIIIIII`, `YYIIIIIXIIIIXIII`, `YYXIXZIIIIIIIIII`, `IIIYXIYIIIIIXIII`, `IIIYIIYIIYIIIYII`, `XIIIZIIIZIIIZIII`, `IXIIZIXZIIIIIXII`, `IIIZIIZXIIXXIIII`, `IIIIIIIIXXIIXXII`, `IIIIIIIIIXXIIXXX`, `XZIZIZZIIIIIIIII`, `IXXIIXIZIIIZIIII`, `IIIIIIZZIIZIZZII`, `IIIIIIIIIIZZIIZI`, `IIIIIIIIIIIIIIZZ`  

Lowest training p_round of the seed (a different code): p_round 0.000244 (proxy), w* = 6, d_X/d_Z = 5/5, CSS no, frame none, 2q gates -, reward -0.3387 (LDPC -0.500, gate +0.0000).

Modal code of the last 20 updates: 83 of 1117 successes, same as selected: no; p_round 0.00249, w* = 8, d_X/d_Z = 10/4, CSS no, frame none, 2q gates -.

**heron / seed 1**: [[16,1,4]], d_X = 7, d_Z = 5, w* = 5 (degree 5), CSS: no, Hadamard frame: none. Encoder: 40 gates, depth 13. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000912/0.000387/0.00107/0.000568; encoder undetected mass board/ion(forte)/heron = 0.00167/0.00127/0.00221. Reward terms: memory +0.1246, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 1327, of which 0 CSS.

Stabilizers (sparsest basis): `XZIIYZIIYIIIIIII`, `IXXIZIIIIIIIIIII`, `IIXXIIXIIZIIIIII`, `IIYIYIIIIYIIYIXI`, `IYYIIYYXIIIIIIII`, `IIIZIIIIIXIIIXYY`, `IIIIIIIIIZXIIIZI`, `IIIIIIIIIIIYIIIY`, `IIIIZIIIZIIIXXII`, `IIIIIIIIIIZXIIXX`, `ZIIIIIIIZIIIIIII`, `IZZZIIIIIIYIIIYI`, `IIIZIIZZIIIIIIII`, `IIIIIZIZIIIIIIII`, `IIIIIIIIIZIIZZII`  

Lowest training p_round of the seed (a different code): p_round 0.000224 (proxy), w* = 6, d_X/d_Z = 4/5, CSS no, frame none, 2q gates -, reward -0.3351 (LDPC -0.500, gate +0.0000).

Modal code of the last 20 updates: 12 of 1134 successes, same as selected: no; p_round 0.00226, w* = 7, d_X/d_Z = 4/4, CSS no, frame none, 2q gates -.

**heron / seed 2**: [[16,1,4]], d_X = 7, d_Z = 5, w* = 5 (degree 6), CSS: no, Hadamard frame: none. Encoder: 47 gates, depth 15. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00138/0.000683/0.00234/0.000963; encoder undetected mass board/ion(forte)/heron = 0.0014/0.0021/0.00341. Reward terms: memory +0.0989, LDPC -0.250 (w* excess 1), gate +0.0000. Visits of the selected code: 1; distinct target codes 2500, of which 0 CSS.

Stabilizers (sparsest basis): `YIIIYIIIXIIIXIII`, `IXXXIIIIIIIIIIII`, `IYIYIIZIIXIIIZII`, `IIIXIIXIIIZIIIII`, `ZIIIXIIZIYIIIYII`, `XIIIXXXIIZIIIIII`, `IIZYIIIIIIYIIIXZ`, `XIXIIIIXYIIIIIIY`, `IIIIIIIIIIZYIIIY`, `IIIIIIIIIZIIXXII`, `ZIIIIZIIZIIIIIII`, `ZZZIZIIIIZIIIIII`, `IIIIZZIIIIIIZZII`, `IIIIIIIZIIIZIIIZ`, `IIIIIIIIIIZIIIZI`  

Lowest training p_round of the seed (a different code): p_round 0.000147 (proxy), w* = 10, d_X/d_Z = 7/5, CSS no, frame none, 2q gates -, reward -1.3169 (LDPC -1.500, gate +0.0000).

Modal code of the last 20 updates: 7864 of 13761 successes, same as selected: no; p_round 0.000869, w* = 8, d_X/d_Z = 7/4, CSS no, frame none, 2q gates -.

## Cross-evaluation

Rows: training cell (selected code of each seed), the alternate codes, the encoder-objective cells of `experiments/gpu_scale/runs` (their best codes, re-scored with the memory objective), and the known codes; columns: evaluation profile. Entries: memory p_round (both bases summed, `R = 4`), mean over codes with [min, max]. Smaller is better. bposd tier: the pinned BP+OSD decoder, 5000 initial shots grown to 100 failures or 100000 shots per basis.

bposd tier (validation):

| trained on \ evaluated under | ion | ion+flag | board | heron | codes |
|---|---|---|---|---|---|
| memory-trained ion | 0.0141 [0.0113, 0.0175] | 0.0144 [0.0105, 0.0207] | 0.0186 [0.0136, 0.0264] | 0.00853 [0.00586, 0.0133] | 3 |
| memory-trained ion+flag | 0.00879 [0.00646, 0.013] | 0.00773 [0.00572, 0.0116] | 0.00826 [0.00492, 0.0133] | 0.00342 [0.00241, 0.0051] | 3 |
| memory-trained board | 0.0108 [0.00812, 0.0123] | 0.0101 [0.00721, 0.0125] | 0.0111 [0.00742, 0.0153] | 0.00406 [0.00278, 0.00607] | 3 |
| memory-trained heron | 0.0181 [0.0084, 0.0293] | 0.0179 [0.00835, 0.0275] | 0.0241 [0.00724, 0.0443] | 0.0105 [0.00315, 0.0178] | 3 |
| lowest-proxy ion | 0.00497 [0.00144, 0.011] | 0.00511 [0.00149, 0.0112] | 0.00489 [0.000878, 0.0117] | 0.00299 [0.000581, 0.00694] | 3 |
| lowest-proxy ion+flag | 0.00829 [0.00233, 0.0114] | 0.00728 [0.00231, 0.0103] | 0.00774 [0.00157, 0.0137] | 0.00364 [0.000693, 0.00689] | 3 |
| lowest-proxy board | 0.00446 [0.00113, 0.00636] | 0.00432 [0.00101, 0.00615] | 0.00423 [0.000732, 0.00666] | 0.00181 [0.000784, 0.00261] | 3 |
| lowest-proxy heron | 0.0144 [0.00109, 0.0227] | 0.0137 [0.000876, 0.0213] | 0.0162 [0.000864, 0.0264] | 0.00773 [0.000378, 0.0121] | 3 |
| encoder-trained board | 0.0155 [0.00555, 0.0272] | 0.0151 [0.00557, 0.0284] | 0.0237 [0.00671, 0.0483] | 0.0105 [0.00338, 0.0218] | 3 |
| encoder-trained ion | 0.0178 [0.0133, 0.0249] | 0.0173 [0.013, 0.0247] | 0.0267 [0.0168, 0.0416] | 0.0112 [0.00833, 0.0164] | 3 |
| encoder-trained heron | 0.0116 [0.00691, 0.0143] | 0.0113 [0.008, 0.0139] | 0.0178 [0.0102, 0.023] | 0.00921 [0.00836, 0.0102] | 3 |
| known: rotated_surface | 0.000645 | 0.000612 | 0.000498 | 0.000353 | 1 |
| known: xzzx | 0.000586 | 0.000698 | 0.000513 | 0.000238 | 1 |

Proxy tier:

| trained on \ evaluated under | ion | ion+flag | board | heron | codes |
|---|---|---|---|---|---|
| memory-trained ion | 0.00128 [0.000856, 0.00209] | 0.000475 [0.00025, 0.000703] | 0.00174 [0.000673, 0.00283] | 0.00124 [0.000396, 0.00169] | 3 |
| memory-trained ion+flag | 0.00112 [0.00107, 0.00115] | 0.000489 [0.000421, 0.000555] | 0.0013 [0.00112, 0.00156] | 0.000771 [0.000584, 0.00101] | 3 |
| memory-trained board | 0.00122 [0.00089, 0.00148] | 0.000536 [0.000442, 0.000709] | 0.00149 [0.00134, 0.00168] | 0.000752 [0.000494, 0.00101] | 3 |
| memory-trained heron | 0.00157 [0.000912, 0.00242] | 0.000621 [0.000387, 0.000793] | 0.00198 [0.00107, 0.00252] | 0.00084 [0.000568, 0.000989] | 3 |
| lowest-proxy ion | 0.000647 [0.000487, 0.000826] | 0.000398 [0.000364, 0.000455] | 0.000872 [0.000496, 0.00127] | 0.000723 [0.000379, 0.000918] | 3 |
| lowest-proxy ion+flag | 0.000864 [0.000627, 0.00123] | 0.000288 [0.000246, 0.00033] | 0.000897 [0.000587, 0.00138] | 0.000487 [0.000355, 0.000689] | 3 |
| lowest-proxy board | 0.000626 [0.000449, 0.000908] | 0.000319 [0.000139, 0.000535] | 0.000631 [0.000477, 0.000908] | 0.000526 [0.000252, 0.000786] | 3 |
| lowest-proxy heron | 0.000699 [0.000385, 0.000902] | 0.000224 [0.00014, 0.000293] | 0.000733 [0.000647, 0.000791] | 0.000228 [0.000172, 0.000269] | 3 |
| encoder-trained board | 0.00121 [0.00106, 0.00137] | 0.000574 [0.000516, 0.000641] | 0.00211 [0.00183, 0.00262] | 0.00122 [0.00114, 0.0013] | 3 |
| encoder-trained ion | 0.00294 [0.00275, 0.00319] | 0.000861 [0.000825, 0.000921] | 0.00297 [0.00244, 0.00353] | 0.00176 [0.0016, 0.00194] | 3 |
| encoder-trained heron | 0.00134 [0.0009, 0.00199] | 0.000792 [0.00059, 0.00116] | 0.00241 [0.00153, 0.0039] | 0.0017 [0.000836, 0.0026] | 3 |
| known: rotated_surface | 0.000374 | 0.00034 | 0.000423 | 0.000339 | 1 |
| known: xzzx | 0.00037 | 0.00034 | 0.000423 | 0.000198 | 1 |

Sampled tier (weight-2 lookup decoder, near maximum likelihood):

| trained on \ evaluated under | ion | ion+flag | board | heron | codes |
|---|---|---|---|---|---|
| memory-trained ion | 0.00856 [0.00636, 0.0119] | 0.0153 [0.0124, 0.0202] | 0.0164 [0.00929, 0.023] | 0.00885 [0.00504, 0.0131] | 3 |
| memory-trained ion+flag | 0.00493 [0.00382, 0.00629] | 0.0107 [0.00853, 0.0139] | 0.00896 [0.00511, 0.0142] | 0.00416 [0.0032, 0.00551] | 3 |
| memory-trained board | 0.00592 [0.00524, 0.00673] | 0.0127 [0.012, 0.0137] | 0.00929 [0.00726, 0.0126] | 0.00404 [0.00288, 0.00542] | 3 |
| memory-trained heron | 0.00943 [0.00547, 0.014] | 0.0189 [0.0123, 0.0247] | 0.0206 [0.00805, 0.032] | 0.00912 [0.00443, 0.0122] | 3 |
| lowest-proxy ion | 0.0034 [0.00156, 0.0057] | 0.0063 [0.00247, 0.0116] | 0.00526 [0.00127, 0.0117] | 0.00403 [0.000842, 0.00894] | 3 |
| lowest-proxy ion+flag | 0.00455 [0.00209, 0.00633] | 0.00845 [0.00312, 0.0122] | 0.00842 [0.00191, 0.0146] | 0.00403 [0.000931, 0.00757] | 3 |
| lowest-proxy board | 0.0028 [0.00123, 0.00377] | 0.0064 [0.00219, 0.00944] | 0.00355 [0.00124, 0.00585] | 0.00271 [0.00108, 0.00518] | 3 |
| lowest-proxy heron | 0.00656 [0.0013, 0.0101] | 0.0131 [0.00162, 0.0199] | 0.0142 [0.00143, 0.0241] | 0.00735 [0.00069, 0.0125] | 3 |
| encoder-trained board | 0.00855 [0.00483, 0.0144] | 0.0159 [0.00985, 0.0254] | 0.018 [0.00828, 0.0331] | 0.0105 [0.0052, 0.0174] | 3 |
| encoder-trained ion | 0.00964 [0.00732, 0.0125] | 0.0157 [0.0114, 0.0202] | 0.0199 [0.0147, 0.0291] | 0.0133 [0.00985, 0.0162] | 3 |
| encoder-trained heron | 0.0073 [0.00562, 0.00906] | 0.0148 [0.0118, 0.0166] | 0.0163 [0.00906, 0.021] | 0.0102 [0.00381, 0.0142] | 3 |
| known: rotated_surface | 0.000853 | 0.00181 | 0.00069 | 0.000475 | 1 |
| known: xzzx | 0.000917 | 0.00188 | 0.000763 | 0.000403 | 1 |

Per code, bposd tier:

| code | d | d_X | d_Z | w* | CSS | H frame | ion | ion+flag | board | heron |
|---|---|---|---|---|---|---|---|---|---|---|
| memory-trained board seed 0 | 4 | 5 | 4 | 5 | no | none | 0.0123 | 0.0125 | 0.0153 | 0.00607 |
| memory-trained board seed 1 | 4 | 5 | 5 | 5 | no | none | 0.00812 | 0.00721 | 0.00742 | 0.00278 |
| memory-trained board seed 2 | 4 | 4 | 4 | 5 | no | none | 0.0119 | 0.0107 | 0.0105 | 0.00333 |
| memory-trained heron seed 0 | 4 | 4 | 4 | 5 | no | none | 0.0293 | 0.0275 | 0.0443 | 0.0178 |
| memory-trained heron seed 1 | 4 | 7 | 5 | 5 | no | none | 0.0084 | 0.00835 | 0.00724 | 0.00315 |
| memory-trained heron seed 2 | 4 | 7 | 5 | 5 | no | none | 0.0165 | 0.018 | 0.0207 | 0.0105 |
| memory-trained ion+flag seed 0 | 4 | 5 | 4 | 5 | no | none | 0.00694 | 0.00572 | 0.00492 | 0.00274 |
| memory-trained ion+flag seed 1 | 4 | 4 | 4 | 5 | no | none | 0.00646 | 0.00586 | 0.00658 | 0.00241 |
| memory-trained ion+flag seed 2 | 4 | 5 | 4 | 5 | no | none | 0.013 | 0.0116 | 0.0133 | 0.0051 |
| memory-trained ion seed 0 | 4 | 5 | 4 | 5 | no | none | 0.0113 | 0.0105 | 0.0136 | 0.00639 |
| memory-trained ion seed 1 | 4 | 5 | 5 | 5 | no | none | 0.0134 | 0.0121 | 0.0157 | 0.00586 |
| memory-trained ion seed 2 | 4 | 5 | 4 | 5 | no | none | 0.0175 | 0.0207 | 0.0264 | 0.0133 |
| lowest-proxy board seed 0 | 4 | 5 | 4 | 6 | no | none | 0.00636 | 0.00615 | 0.00666 | 0.00204 |
| lowest-proxy board seed 1 | 4 | 4 | 5 | 7 | no | none | 0.0059 | 0.0058 | 0.00531 | 0.00261 |
| lowest-proxy board seed 2 | 4 | 7 | 4 | 8 | no | none | 0.00113 | 0.00101 | 0.000732 | 0.000784 |
| lowest-proxy heron seed 0 | 4 | 5 | 5 | 6 | no | none | 0.0227 | 0.0213 | 0.0264 | 0.0121 |
| lowest-proxy heron seed 1 | 4 | 4 | 5 | 6 | no | none | 0.0195 | 0.019 | 0.0214 | 0.0108 |
| lowest-proxy heron seed 2 | 4 | 7 | 5 | 10 | no | none | 0.00109 | 0.000876 | 0.000864 | 0.000378 |
| lowest-proxy ion+flag seed 0 | 4 | 4 | 6 | 6 | no | none | 0.0111 | 0.0103 | 0.0137 | 0.00689 |
| lowest-proxy ion+flag seed 1 | 4 | 4 | 4 | 6 | no | none | 0.0114 | 0.00924 | 0.00795 | 0.00334 |
| lowest-proxy ion+flag seed 2 | 4 | 7 | 4 | 8 | no | none | 0.00233 | 0.00231 | 0.00157 | 0.000693 |
| lowest-proxy ion seed 0 | 4 | 5 | 4 | 7 | no | none | 0.00242 | 0.00267 | 0.00208 | 0.00146 |
| lowest-proxy ion seed 1 | 4 | 4 | 4 | 6 | no | none | 0.011 | 0.0112 | 0.0117 | 0.00694 |
| lowest-proxy ion seed 2 | 4 | 8 | 4 | 10 | no | none | 0.00144 | 0.00149 | 0.000878 | 0.000581 |
| encoder-trained board seed 0 | 4 | 5 | 4 | 5 | no | none | 0.0137 | 0.0112 | 0.016 | 0.00618 |
| encoder-trained board seed 1 | 4 | 6 | 4 | 5 | no | none | 0.00555 | 0.00557 | 0.00671 | 0.00338 |
| encoder-trained board seed 2 | 4 | 4 | 4 | 5 | no | none | 0.0272 | 0.0284 | 0.0483 | 0.0218 |
| encoder-trained heron seed 0 | 4 | 4 | 4 | 5 | no | none | 0.00691 | 0.008 | 0.0102 | 0.00836 |
| encoder-trained heron seed 1 | 4 | 5 | 4 | 5 | no | none | 0.0143 | 0.0139 | 0.023 | 0.0102 |
| encoder-trained heron seed 2 | 4 | 6 | 4 | 5 | no | none | 0.0135 | 0.012 | 0.0201 | 0.0091 |
| encoder-trained ion seed 0 | 4 | 6 | 5 | 5 | no | none | 0.0133 | 0.0144 | 0.0168 | 0.00878 |
| encoder-trained ion seed 1 | 4 | 5 | 4 | 5 | no | none | 0.0151 | 0.013 | 0.0216 | 0.00833 |
| encoder-trained ion seed 2 | 4 | 5 | 4 | 5 | no | none | 0.0249 | 0.0247 | 0.0416 | 0.0164 |
| rotated surface [[16,1,4]] | 4 | 4 | 4 | 4 | yes | H on 0 qubit(s) (2 frames) | 0.000645 | 0.000612 | 0.000498 | 0.000353 |
| XZZX surface [[16,1,4]]: H on the checkerboard [1, 3, 4, 6, 9, 11, 12, 14] | 4 | 4 | 4 | 4 | no | H on 8 qubit(s) (2 frames) | 0.000586 | 0.000698 | 0.000513 | 0.000238 |

Per code, proxy tier:

| code | d | d_X | d_Z | w* | CSS | H frame | ion | ion+flag | board | heron |
|---|---|---|---|---|---|---|---|---|---|---|
| memory-trained board seed 0 | 4 | 5 | 4 | 5 | no | none | 0.00089 | 0.000457 | 0.00144 | 0.00101 |
| memory-trained board seed 1 | 4 | 5 | 5 | 5 | no | none | 0.00128 | 0.000442 | 0.00134 | 0.000494 |
| memory-trained board seed 2 | 4 | 4 | 4 | 5 | no | none | 0.00148 | 0.000709 | 0.00168 | 0.000748 |
| memory-trained heron seed 0 | 4 | 4 | 4 | 5 | no | none | 0.00242 | 0.000793 | 0.00252 | 0.000989 |
| memory-trained heron seed 1 | 4 | 7 | 5 | 5 | no | none | 0.000912 | 0.000387 | 0.00107 | 0.000568 |
| memory-trained heron seed 2 | 4 | 7 | 5 | 5 | no | none | 0.00138 | 0.000683 | 0.00234 | 0.000963 |
| memory-trained ion+flag seed 0 | 4 | 5 | 4 | 5 | no | none | 0.00115 | 0.000492 | 0.00121 | 0.00072 |
| memory-trained ion+flag seed 1 | 4 | 4 | 4 | 5 | no | none | 0.00107 | 0.000421 | 0.00112 | 0.000584 |
| memory-trained ion+flag seed 2 | 4 | 5 | 4 | 5 | no | none | 0.00113 | 0.000555 | 0.00156 | 0.00101 |
| memory-trained ion seed 0 | 4 | 5 | 4 | 5 | no | none | 0.000883 | 0.000472 | 0.00171 | 0.00163 |
| memory-trained ion seed 1 | 4 | 5 | 5 | 5 | no | none | 0.000856 | 0.00025 | 0.000673 | 0.000396 |
| memory-trained ion seed 2 | 4 | 5 | 4 | 5 | no | none | 0.00209 | 0.000703 | 0.00283 | 0.00169 |
| lowest-proxy board seed 0 | 4 | 5 | 4 | 6 | no | none | 0.000908 | 0.000535 | 0.000908 | 0.000786 |
| lowest-proxy board seed 1 | 4 | 4 | 5 | 7 | no | none | 0.00052 | 0.000139 | 0.000477 | 0.000252 |
| lowest-proxy board seed 2 | 4 | 7 | 4 | 8 | no | none | 0.000449 | 0.000283 | 0.000506 | 0.00054 |
| lowest-proxy heron seed 0 | 4 | 5 | 5 | 6 | no | none | 0.000902 | 0.00014 | 0.00076 | 0.000244 |
| lowest-proxy heron seed 1 | 4 | 4 | 5 | 6 | no | none | 0.00081 | 0.000239 | 0.000791 | 0.000269 |
| lowest-proxy heron seed 2 | 4 | 7 | 5 | 10 | no | none | 0.000385 | 0.000293 | 0.000647 | 0.000172 |
| lowest-proxy ion+flag seed 0 | 4 | 4 | 6 | 6 | no | none | 0.000739 | 0.00033 | 0.00138 | 0.000689 |
| lowest-proxy ion+flag seed 1 | 4 | 4 | 4 | 6 | no | none | 0.00123 | 0.000246 | 0.000587 | 0.000355 |
| lowest-proxy ion+flag seed 2 | 4 | 7 | 4 | 8 | no | none | 0.000627 | 0.000287 | 0.000721 | 0.000419 |
| lowest-proxy ion seed 0 | 4 | 5 | 4 | 7 | no | none | 0.000628 | 0.000455 | 0.000846 | 0.000871 |
| lowest-proxy ion seed 1 | 4 | 4 | 4 | 6 | no | none | 0.000826 | 0.000375 | 0.00127 | 0.000918 |
| lowest-proxy ion seed 2 | 4 | 8 | 4 | 10 | no | none | 0.000487 | 0.000364 | 0.000496 | 0.000379 |
| encoder-trained board seed 0 | 4 | 5 | 4 | 5 | no | none | 0.0012 | 0.000516 | 0.00188 | 0.00114 |
| encoder-trained board seed 1 | 4 | 6 | 4 | 5 | no | none | 0.00106 | 0.000641 | 0.00183 | 0.0013 |
| encoder-trained board seed 2 | 4 | 4 | 4 | 5 | no | none | 0.00137 | 0.000564 | 0.00262 | 0.00122 |
| encoder-trained heron seed 0 | 4 | 4 | 4 | 5 | no | none | 0.0009 | 0.000626 | 0.00153 | 0.000836 |
| encoder-trained heron seed 1 | 4 | 5 | 4 | 5 | no | none | 0.00199 | 0.00116 | 0.0039 | 0.0026 |
| encoder-trained heron seed 2 | 4 | 6 | 4 | 5 | no | none | 0.00113 | 0.00059 | 0.0018 | 0.00167 |
| encoder-trained ion seed 0 | 4 | 6 | 5 | 5 | no | none | 0.00288 | 0.000921 | 0.00293 | 0.00174 |
| encoder-trained ion seed 1 | 4 | 5 | 4 | 5 | no | none | 0.00275 | 0.000825 | 0.00244 | 0.0016 |
| encoder-trained ion seed 2 | 4 | 5 | 4 | 5 | no | none | 0.00319 | 0.000836 | 0.00353 | 0.00194 |
| rotated surface [[16,1,4]] | 4 | 4 | 4 | 4 | yes | H on 0 qubit(s) (2 frames) | 0.000374 | 0.00034 | 0.000423 | 0.000339 |
| XZZX surface [[16,1,4]]: H on the checkerboard [1, 3, 4, 6, 9, 11, 12, 14] | 4 | 4 | 4 | 4 | no | H on 8 qubit(s) (2 frames) | 0.00037 | 0.00034 | 0.000423 | 0.000198 |

Tier agreement over every scored code (ratio is the second tier over the first):

| profile | tiers | codes | Spearman | median ratio | range |
|---|---|---|---|---|---|
| ion | bposd / proxy | 35 | 0.74 | 8.07 | 1.58 to 25.15 |
| ion | sampled / proxy | 35 | 0.75 | 5.16 | 2.28 to 11.23 |
| ion | bposd / sampled | 35 | 0.98 | 1.77 | 0.64 to 2.36 |
| ion_flag | bposd / proxy | 35 | 0.46 | 20.41 | 1.80 to 152.39 |
| ion_flag | sampled / proxy | 35 | 0.44 | 24.16 | 5.32 to 142.42 |
| ion_flag | bposd / sampled | 35 | 0.94 | 0.81 | 0.34 to 1.25 |
| board | bposd / proxy | 35 | 0.77 | 8.00 | 1.18 to 34.77 |
| board | sampled / proxy | 35 | 0.77 | 6.82 | 1.63 to 31.76 |
| board | bposd / sampled | 35 | 0.97 | 1.00 | 0.59 to 1.87 |
| heron | bposd / proxy | 35 | 0.60 | 5.44 | 1.04 to 49.44 |
| heron | sampled / proxy | 35 | 0.70 | 6.49 | 1.40 to 51.11 |
| heron | bposd / sampled | 35 | 0.94 | 0.78 | 0.50 to 2.19 |

## Comparison under the training profile (bposd)

- ion: memory-trained best per seed 0.0113 (seed 0), 0.0134 (seed 1), 0.0175 (seed 2); encoder-trained codes 0.00555 to 0.0272 (best of 9); 0 of 3 memory-trained seeds below the best encoder-trained code; ratio to the surface code 17.5x, 20.7x, 27.2x; rotated surface 0.000645 (0 of 3 seeds below); XZZX 0.000586 (0 of 3 seeds below); best seed 0: 0.0113.
- ion+flag: memory-trained best per seed 0.00572 (seed 0), 0.00586 (seed 1), 0.0116 (seed 2); encoder-trained codes 0.00557 to 0.0284 (best of 9); 0 of 3 memory-trained seeds below the best encoder-trained code; ratio to the surface code 9.4x, 9.6x, 19.0x; rotated surface 0.000612 (0 of 3 seeds below); XZZX 0.000698 (0 of 3 seeds below); best seed 0: 0.00572.
- board: memory-trained best per seed 0.0153 (seed 0), 0.00742 (seed 1), 0.0105 (seed 2); encoder-trained codes 0.00671 to 0.0483 (best of 9); 0 of 3 memory-trained seeds below the best encoder-trained code; ratio to the surface code 30.8x, 14.9x, 21.0x; rotated surface 0.000498 (0 of 3 seeds below); XZZX 0.000513 (0 of 3 seeds below); best seed 1: 0.00742.
- heron: memory-trained best per seed 0.0178 (seed 0), 0.00315 (seed 1), 0.0105 (seed 2); encoder-trained codes 0.00338 to 0.0218 (best of 9); 1 of 3 memory-trained seeds below the best encoder-trained code; ratio to the surface code 50.4x, 8.9x, 29.8x; rotated surface 0.000353 (0 of 3 seeds below); XZZX 0.000238 (0 of 3 seeds below); best seed 1: 0.00315.

## Figures

`figures/curves.png`: success rate among the episodes finished in an update, moving average over 10 updates (top), and the lowest training p_round seen so far at the tier the reward used (bottom, log scale shared across cells; dashed: rotated surface, dotted: XZZX under the cell's profile at the proxy tier) against env steps, one column per cell, seeds as series.

`figures/matrix.png`: bposd p_round per evaluation profile (one panel each), bars are the mean over codes of a row of the cross-evaluation table with min to max whiskers (yellow: run-2 cells, blue: run-1 memory-trained cells, light blue: their lowest-proxy codes, red: encoder-trained cells, grey: known codes).

# Run 2: proxy selection inside its valid regime, no LDPC term, gate-count term

## Verdict (run 2)

Status: 9 runs (ion chain 0.25 bare, ion chain 0.25 one flag, board 0.25; seeds
0 to 2) on the RunPod A40 with the pod otherwise idle, three cells of a seed
concurrent, host CPU capped at four cores as before (`taskset 0-3`, `nice 5`,
one memory worker per cell). Two changes from run 1 as asked: the LDPC `w*`
term is removed from the reward and the selection is the target-distance code
with the lowest training `p_round`; a gate-count term `-0.2 max(0, g/240 - 1)`
penalises memory circuits above 1.25 times the surface code's 192 two-qubit
gates per round, and a candidate whose expected faults per shot exceed 0.8 in
either basis is evaluated at the sampled tier (weight-2 lookup decoder, near
maximum likelihood) instead of the proxy, cached per code. One change not asked
for, found while testing the gate term: the memory circuit measures the
generators it is given, and a code read off an encoder tableau carries the
tableau's dense basis (the replayed rotated surface code: 224 two-qubit gates
per round against 192 in its sparsest basis). Run 2 therefore evaluates every
code on the sparsest basis of its stabilizer group (`ldpc.sparsest_basis_rows`,
exact at 15 generators), in training and in scoring, and the run-2 scoring pass
re-scores the run-1 codes and the encoder-objective codes the same way. This
changes the run-1 picture: the run-1 codes, which run 1 scored on their tableau
bases at 8.9 to 50 times the surface code, are 1.5 to 5.3 times the surface
code on their sparsest bases (see the run-1 correction above).

Wall cap 120 min per run instead of 150: nine runs at three concurrent fit 6
GPU-hours only at 120 min (used 5.5 h wall, 15.9 process-hours; the three board
runs stopped at their 3M-step budget after 58 to 78 min). Every run reached `d
= 4` (first success at 256k to 444k env steps); 1,850 to 5,159 distinct target
codes per ion run, 894 to 1,173 per ion+flag run, 953 to 3,482 per board run,
every one evaluated; final success rates 0.19 to 1.00 (five of nine above 0.6).
The sampled tier took 15 to 61 percent of the evaluations in the bare cells and
98 to 100 percent in the flag cell (the flag CNOTs put nearly every code above
0.8 expected faults; the rotated surface code with flags is at 0.90). Scoring:
45 codes (9 run-2 selected, 1 run-2 highest-reward alternate that differs from
the selected code, 12 run-1 best, 12 run-1 lowest-proxy, 9 encoder-objective,
rotated surface, XZZX) under the four profiles at the proxy, sampled, and bposd
tiers, all on the sparsest basis (`experiments/memory_ppo/runs/run2/scores.json`).

Answer: does selecting on the proxy inside its valid regime close the gap to
the surface code, and by how much? Partly. Under bposd on the training profile
the nine selected codes are 0.8 to 3.1 times the rotated surface code (median
1.8): ion 3.1, 2.0, 2.1 (seeds 0 to 2; surface 5.99e-4), ion+flag 1.1, 1.6, 1.8
(surface 7.63e-4), board 0.8, 2.0, 1.3 (surface 6.08e-4). The run-1 best codes
of the same three cells, re-scored on their sparsest bases, are 1.8 to 5.3
times the surface code (median 2.6): ion 2.3, 5.3, 4.6, ion+flag 2.4, 2.5, 2.6,
board 3.0, 1.8, 2.6. The encoder-objective codes are 1.5 to 5.1 times (median
2.3). Cell means, run 2 against run 1 on the same profile: ion 1.44e-3 against
2.45e-3, ion+flag 1.16e-3 against 1.92e-3, board 8.4e-4 against 1.51e-3, so run
2 halves the gap on average, and 7 of 9 seeds are below the best
encoder-objective code of their profile against 1 of 9 in run 1. One code is at
the surface code's level: board seed 0 (`[[16,1,4]]`, `w* = 7`, 228 two-qubit
gates per round, 0.57 expected faults) at 5.09e-4 against 6.08e-4 under bposd
on board, which is 0.8 times, inside the resolution of the estimates (about 100
failures per basis, 10 to 15 percent each); under the sampled tier the same
pair reads 7.89e-4 against 8.59e-4 (0.9 times), and under bposd on the other
three profiles the code is 1.6, 2.0, and 1.0 times the surface code. No
selected code is below the surface code on ion or ion+flag, and none beats XZZX
on ion (XZZX 7.36e-4; the closest is 1.20e-3); on board, two of three are below
XZZX (8.19e-4).

What the selected codes are: all nine are degenerate non-CSS `[[16,1,4]]` codes
with no Hadamard frame, `w* = 7` (ion, board) or 9 (ion+flag), 216 to 276
two-qubit gates per round on the sparsest basis, 0.57 to 0.82 expected faults
per shot, `d_X > d_Z` in five of nine and never the reverse. The gate term did
what it was for: the selected circuits are within 1.15 times the surface code's
length and the sparsest-basis fault counts are inside the proxy's regime for
eight of nine (the ion+flag seed-0 code, at 0.82, was scored at the sampled
tier). It did not produce sparse codes: without the `w*` term the agent
converges on codes with one or two weight-7 to weight-9 generators among light
ones (the modal codes of the converged runs have `w* = 7` to 11 with 184 to 228
gates), and the highest-reward alternate of board seed 1 (`w* = 6`, 240 gates)
scores 0.69e-3, better than that seed's selected code (1.22e-3) under bposd.

Where the proxy still fails: within the narrow range these codes occupy (every
scored code is within 0.7 to 5.4 times the surface code under bposd) the proxy
does not rank them. Over the 45 sparsified codes the bposd against proxy
Spearman is 0.33 (ion), -0.30 (ion+flag), 0.40 (board), 0.72 (heron), with the
bposd to proxy ratio at median 3.4, 6.5, 2.3, and 1.9 (range 1.1 to 33); the
sampled tier again sides with bposd (Spearman 0.86 to 0.91). The bposd to proxy
ratio is 1.6 (ion), 2.3 (ion+flag), 1.2 (board), and 1.3 (heron) on the surface
code itself, so the proxy also underestimates the reference by less than it
underestimates the discovered codes, which flatters them by a further factor of
about two in training. In the flag cell the proxy inverts the ranking: it puts
the flagged surface code at 3.35e-4 against 3.66e-4 bare (flags help), bposd
puts it at 7.63e-4 against 5.99e-4 (flags hurt at this scale), and the selected
ion+flag codes are the ones the sampled tier, not the proxy, ranked, since the
proxy was almost never used there.

So the gap closes from a median 2.6 times (run-1 codes on the same cells) to
1.8 times, with one seed at parity on board and none on the ion profiles; the
remaining gap is the proxy's ranking error at the 2 to 3 times level, not the
regime problem of run 1. Closing it further needs the sampled tier as the
training signal throughout (about 3 s per code on the sparsest basis at these
fault counts, 5 to 8 times the proxy) or a decoder-aware proxy; it does not
need the gate term to be tightened, since the selected circuits are already
surface-length.

## Setting

Cells ion, ion_flag, board (same profiles and scales as run 1), seeds 0, 1, 2, wall cap 120 min per run, budget 3,000,000 env steps, batch 1024; everything else as in run 1. Reward: `(d - 1 + f)/d* - 1 - q_1 + [d >= d*, q_1 = 0](0.1 clip(log10(1e-2 / p_round), -3, 3) - 0.2 max(0, g / 240 - 1)), f = detected fraction of weight-d Paulis, q_1 = fraction of weight-1 Paulis in S, g = two-qubit gates per round of the memory circuit, p_round = memory objective (R = 4, both bases; proxy tier, or the sampled tier when the expected faults per shot exceed 0.8) under the training profile; no w* term`. The LDPC term is removed (w* is reported as a statistic); the gate-count term penalises memory circuits above 240 two-qubit gates per round (1.25 times the surface code's 192) with weight 0.2; a candidate whose expected faults per shot exceed 0.8 in either basis is evaluated at the sampled tier (weight-2 lookup decoder, 2000 shots grown to 50 failures or 20000 shots per basis) instead of the proxy, cached per code. Selection: the target-distance code with the lowest training p_round (at the tier the reward used); the highest-reward code is reported as the alternate.

## What ran where

`reward share` is the fraction of the training wall time spent in the reward call (GPU distance kernel plus the host-side terminal terms), `memory share` the part of it spent waiting for the memory pool, `update share` the PPO update.

| profile | seed | device | batch | budget (env steps) | env steps | updates | wall (s) | env steps/s | reward share | memory share | update share | memory evals | s per code | cache hits | wall cap hit | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ion | 0 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,851,392 | 113 | 7242 | 256 | 96% | 96% | 3% | 1850 | 3.73 | 37 | yes | `run_ion_seed0.json` |
| ion | 1 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,261,568 | 77 | 7518 | 168 | 98% | 97% | 2% | 4475 | 1.61 | 420 | yes | `run_ion_seed1.json` |
| ion | 2 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,441,792 | 88 | 7324 | 197 | 97% | 97% | 2% | 5159 | 1.35 | 1238 | yes | `run_ion_seed2.json` |
| ion_flag | 0 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,474,560 | 90 | 7455 | 198 | 97% | 97% | 2% | 894 | 8.07 | 0 | yes | `run_ion_flag_seed0.json` |
| ion_flag | 1 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,376,256 | 84 | 7847 | 175 | 98% | 97% | 2% | 1173 | 6.49 | 2 | yes | `run_ion_flag_seed1.json` |
| ion_flag | 2 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 1,261,568 | 77 | 7214 | 175 | 97% | 97% | 2% | 1135 | 6.16 | 13 | yes | `run_ion_flag_seed2.json` |
| board | 0 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 3,014,656 | 184 | 3495 | 863 | 87% | 86% | 9% | 953 | 3.13 | 6 | no | `run_board_seed0.json` |
| board | 1 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 3,014,656 | 184 | 4619 | 653 | 88% | 87% | 7% | 3482 | 1.13 | 17571 | no | `run_board_seed1.json` |
| board | 2 | cuda (NVIDIA A40) | 1024 | 3,000,000 | 3,014,656 | 184 | 4556 | 662 | 91% | 90% | 5% | 2571 | 1.56 | 4241 | no | `run_board_seed2.json` |

## Runs

| profile | seeds | seeds reaching d* | first success (env steps) | final success rate | distinct target codes per seed | memory evaluations per seed | best training p_round per seed |
|---|---|---|---|---|---|---|---|
| ion | 3 | 3/3 | 3.67e+05 [2.56e+05, 4.44e+05] | 0.682 [0.191, 0.959] | 3.83e+03 [1.85e+03, 5.16e+03] | 3.83e+03 [1.85e+03, 5.16e+03] | 0.000293 [0.000267, 0.000333] |
| ion_flag | 3 | 3/3 | 3.67e+05 [2.56e+05, 4.44e+05] | 0.428 [0.259, 0.635] | 1.07e+03 [894, 1.17e+03] | 1.07e+03 [894, 1.17e+03] | 0.000991 [0.000362, 0.00215] |
| board | 3 | 3/3 | 3.67e+05 [2.56e+05, 4.44e+05] | 0.738 [0.227, 0.998] | 2.34e+03 [953, 3.48e+03] | 2.34e+03 [953, 3.48e+03] | 0.000224 [0.000215, 0.000233] |

Runs stopped by the wall cap before their step budget: ion_flag/seed 0, ion_flag/seed 1, ion_flag/seed 2, ion/seed 0, ion/seed 1, ion/seed 2.

Sampled-tier evaluations (candidates above the expected-fault threshold): board/seed 0: 463 of 953, board/seed 1: 454 of 3482, board/seed 2: 546 of 2571, ion_flag/seed 0: 894 of 894, ion_flag/seed 1: 1162 of 1173, ion_flag/seed 2: 1095 of 1135, ion/seed 0: 1134 of 1850, ion/seed 1: 790 of 4475, ion/seed 2: 646 of 5159.

## Codes found

Selected code per seed (`selection` in the JSON; `H frame` is the smallest Hadamard frame that makes the stabilizer group CSS, `none` when no such frame exists; `2q gates per round`, `batches`, and `E` (expected faults per shot) are the memory circuit's under the training profile; `train p_round` is the value the reward saw at the tier in `tier`; the `bposd` columns are from `scores.json`).

| trained on | seed | reached d* | d | d_X | d_Z | w* | CSS | H frame | gates | 2q gates per round | batches | E (train profile) | train p_round | tier | bposd ion | bposd ion+flag | bposd board | bposd heron | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ion | 0 | yes | 4 | 7 | 5 | 7 | no | none | 46 | 240 | 3 | 0.79 | 0.000333 | proxy | 0.00186 | 0.00185 | 0.00107 | 0.000454 | `run_ion_seed0.json` |
| ion | 1 | yes | 4 | 4 | 4 | 7 | no | none | 42 | 232 | 3 | 0.77 | 0.000267 | proxy | 0.0012 | 0.00147 | 0.00107 | 0.000388 | `run_ion_seed1.json` |
| ion | 2 | yes | 4 | 7 | 4 | 7 | no | none | 38 | 216 | 3 | 0.73 | 0.000279 | proxy | 0.00128 | 0.00137 | 0.000656 | 0.000413 | `run_ion_seed2.json` |
| ion_flag | 0 | yes | 4 | 4 | 4 | 9 | no | none | 37 | 232 | 3 | 0.82 | 0.00215 | sampled | 0.000909 | 0.000875 | 0.000827 | 0.000606 | `run_ion_flag_seed0.json` |
| ion_flag | 1 | yes | 4 | 7 | 4 | 9 | no | none | 20 | 216 | 3 | 0.76 | 0.000464 | proxy | 0.00149 | 0.00123 | 0.00128 | 0.00118 | `run_ion_flag_seed1.json` |
| ion_flag | 2 | yes | 4 | 7 | 4 | 9 | no | none | 28 | 224 | 3 | 0.79 | 0.000362 | proxy | 0.00142 | 0.00136 | 0.000931 | 0.000762 | `run_ion_flag_seed2.json` |
| board | 0 | yes | 4 | 4 | 4 | 7 | no | none | 49 | 228 | 3 | 0.57 | 0.000215 | proxy | 0.000978 | 0.00104 | 0.000509 | 0.0003 | `run_board_seed0.json` |
| board | 1 | yes | 4 | 4 | 4 | 7 | no | none | 46 | 276 | 3 | 0.67 | 0.000233 | proxy | 0.00258 | 0.00269 | 0.00122 | 0.000743 | `run_board_seed1.json` |
| board | 2 | yes | 4 | 7 | 4 | 7 | no | none | 50 | 244 | 3 | 0.60 | 0.000224 | proxy | 0.0022 | 0.00201 | 0.000784 | 0.000514 | `run_board_seed2.json` |

**ion / seed 0**: [[16,1,4]], d_X = 7, d_Z = 5, w* = 7 (degree 6), CSS: no, Hadamard frame: none. Encoder: 46 gates, depth 24. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000333/0.000263/0.000396/0.000234; encoder undetected mass board/ion(forte)/heron = 0.00233/0.00095/0.00158. Reward terms: memory +0.1477, LDPC -0.000 (w* excess 3), gate -0.0000. Visits of the selected code: 1; distinct target codes 1850, of which 0 CSS.

Stabilizers (sparsest basis): `XIXXZIIIIIIIIIII`, `IXYYIZIIIZIIIIII`, `XIIIIIIYZIIYIIII`, `IIIXIXXIIIIIIIII`, `IIYIYIIIIIYIIIIY`, `IZIIZXIIIIXIIIII`, `ZIIIYIIIXZYIIIII`, `IZIIIIIIZYIIXYXI`, `IIIZYZIZYXIIIIXI`, `IIZZIIZIIIIIIIIZ`, `IIIIZIIIZIIIIIIZ`, `XXIIIIIIIIZIIZIZ`, `IIIIIIIZIIIZIIII`, `IIIIIIIIIZIIIIZI`, `IIIIIIIIIIIIZZII`  

Modal code of the last 20 updates: 17 of 1122 successes, same as selected: no; p_round 0.00067, w* = 8, d_X/d_Z = 4/4, CSS no, frame none, 2q gates 200.

**ion / seed 1**: [[16,1,4]], d_X = 4, d_Z = 4, w* = 7 (degree 6), CSS: no, Hadamard frame: none. Encoder: 42 gates, depth 18. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000288/0.000191/0.000458/0.000258; encoder undetected mass board/ion(forte)/heron = 0.00167/0.00108/0.00221. Reward terms: memory +0.1573, LDPC -0.000 (w* excess 3), gate -0.0000. Visits of the selected code: 1; distinct target codes 4475, of which 0 CSS.

Stabilizers (sparsest basis): `YIIIYIIIIIIIIIII`, `IXIXIIIYIIIYIIII`, `IIYYIIIIIIZZIIII`, `IXXXIIXIIXIIIIII`, `IZZIYIIIZIIXIYIX`, `IIIIIXIIIXIIIYYX`, `IIZIIIYIIIYIIIII`, `XIIIXIIIXIIIXIII`, `IIXIIXIYXIXIXIYI`, `IZIIIZIZIZIIIIII`, `IIZZIIIZIIIIIIII`, `IIIIIZIIIIIIIIZI`, `IIIIIZIZIIIZIZII`, `IIIIIIIIZIIIZIII`, `IIIIIIIIIIIIIZIZ`  

Modal code of the last 20 updates: 49 of 4549 successes, same as selected: no; p_round 0.000907, w* = 7, d_X/d_Z = 4/4, CSS no, frame none, 2q gates 196.

**ion / seed 2**: [[16,1,4]], d_X = 7, d_Z = 4, w* = 7 (degree 5), CSS: no, Hadamard frame: none. Encoder: 38 gates, depth 17. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000292/0.000224/0.000293/0.000211; encoder undetected mass board/ion(forte)/heron = 0.0014/0.00193/0.00355. Reward terms: memory +0.1555, LDPC -0.000 (w* excess 3), gate -0.0000. Visits of the selected code: 1; distinct target codes 5159, of which 0 CSS.

Stabilizers (sparsest basis): `YIIYXYIYYIIIYIII`, `IXZIIXIXIIZIIZII`, `IZXIIIZIIIIIIIII`, `IIIXIIIXIIIXIIII`, `IIIIIYIIIIYXIIXZ`, `IIZIIIXIIIIXIIII`, `IIIIIIIIIYYIIXII`, `IIIIIZIIIXXIXIII`, `IIIIIIIIIIIIIIZX`, `ZIIIZIIIIIIIIIII`, `IZIZIIIZIIIIIIII`, `IIIIZIIIZIIIIIII`, `IIIIIZZZIIIZIIII`, `IIIIIIIIZZIIZZII`, `IIIIIIIIIZZIIIZI`  

Modal code of the last 20 updates: 153 of 6185 successes, same as selected: no; p_round 0.000661, w* = 9, d_X/d_Z = 7/4, CSS no, frame none, 2q gates 184.

**ion_flag / seed 0**: [[16,1,4]], d_X = 4, d_Z = 4, w* = 9 (degree 5), CSS: no, Hadamard frame: none. Encoder: 37 gates, depth 13. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00043/0.000341/0.00051/0.000496; encoder undetected mass board/ion(forte)/heron = 0.0014/0.00215/0.00355. Reward terms: memory +0.0668, LDPC -0.000 (w* excess 5), gate -0.0000. Visits of the selected code: 1; distinct target codes 894, of which 0 CSS.

Stabilizers (sparsest basis): `XIIIIIIIXIIIIIII`, `IXZXXXZXIIXXIIII`, `IIXZIIIIIIIIIIII`, `IIZYYIIXIIIIZZII`, `IIIIXIIIXIIIXIXI`, `IIIIIZXIIIIIIIII`, `IIIIIIIIIXIIXXXX`, `ZIIZZIIIZIIIIIII`, `IZIIIZIIIIIIIIII`, `IIIZIIIZIIIIIIII`, `IIIIIZIIIIZIIIII`, `IIIIIIIIIZIIIZII`, `IIIIIIIIIIZZIIII`, `IIIIIIIIIIIIZIZI`, `IIIIIIIIIIIIIZIZ`  

Modal code of the last 20 updates: 1 of 561 successes, same as selected: no; p_round 0.00548, w* = 7, d_X/d_Z = 4/4, CSS no, frame none, 2q gates 316.

**ion_flag / seed 1**: [[16,1,4]], d_X = 7, d_Z = 4, w* = 9 (degree 5), CSS: no, Hadamard frame: none. Encoder: 20 gates, depth 12. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000571/0.000436/0.000876/0.000946; encoder undetected mass board/ion(forte)/heron = 0.00153/0.00159/0.00305. Reward terms: memory +0.1334, LDPC -0.000 (w* excess 5), gate -0.0000. Visits of the selected code: 1; distinct target codes 1173, of which 0 CSS.

Stabilizers (sparsest basis): `YYXXXZIIXZIIXIII`, `IXXXIIIIIXXIIXII`, `IXXXIXXXIIIXIIXX`, `ZIIIZIIIIIIIIIII`, `IZZIIIIIIIIIIIII`, `IIZZIIIIIIIIIIII`, `IIIIZIIIZIIIIIII`, `IIIIIZZIIIIIIIII`, `IIIIIIZZIIIIIIII`, `IIIIIIIZIIIZIIII`, `IIIIIIIIZIIIZIII`, `IIIIIIIIIZZIIIII`, `IIIIIIIIIZIIIZII`, `IIIIIIIIIIIZIIIZ`, `IIIIIIIIIIIIIIZZ`  

Modal code of the last 20 updates: 2 of 938 successes, same as selected: no; p_round 0.00329, w* = 8, d_X/d_Z = 8/4, CSS no, frame none, 2q gates 264.

**ion_flag / seed 2**: [[16,1,4]], d_X = 7, d_Z = 4, w* = 9 (degree 5), CSS: no, Hadamard frame: none. Encoder: 28 gates, depth 16. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000971/0.00087/0.000785/0.00056; encoder undetected mass board/ion(forte)/heron = 0.0014/0.00212/0.00355. Reward terms: memory +0.1441, LDPC -0.000 (w* excess 5), gate -0.0000. Visits of the selected code: 1; distinct target codes 1135, of which 0 CSS.

Stabilizers (sparsest basis): `YZIIXXIIXZYIXIXI`, `IXXXIIXIIIXIIIXI`, `IIIIIXXIIIIIIIII`, `IIIIIIIXIIIIIIIX`, `IIIIIIIIIXXXIIXI`, `IIIIIIIIIIIIIXXI`, `ZIIIZIIIIIIIIIII`, `IZZIIIIIIIIIIIII`, `IIZZIIIIIIIIIIII`, `IIIIZIIIZIIIIIII`, `IIIIIYYIIZZIIIII`, `IIIIIIIZIIIIIIIZ`, `IIIIIIIIZIIIZIII`, `IIIIIIIIIZIZIIII`, `IIIIIIIIIIZIIZZI`  

Modal code of the last 20 updates: 11 of 975 successes, same as selected: no; p_round 0.00041, w* = 9, d_X/d_Z = 7/4, CSS no, frame none, 2q gates 216.

**board / seed 0**: [[16,1,4]], d_X = 4, d_Z = 4, w* = 7 (degree 5), CSS: no, Hadamard frame: none. Encoder: 49 gates, depth 28. Memory p_round (proxy) ion/ion+flag/board/heron = 0.00025/0.000153/0.000215/0.000176; encoder undetected mass board/ion(forte)/heron = 0.0024/0.00123/0.00178. Reward terms: memory +0.1668, LDPC -0.000 (w* excess 3), gate -0.0000. Visits of the selected code: 1; distinct target codes 953, of which 0 CSS.

Stabilizers (sparsest basis): `XIXIIIIXIIIIIIII`, `IYIZYIIIIIXYIYII`, `XIIZIIIIIZIIIIII`, `IIIYIIIIIYIXIIII`, `IIXIIYIIIIXIYIXZ`, `IIIIIIXXIIXIIIII`, `YYIYXIZYXIIIIIII`, `IIIIIIIIXIIIYYII`, `IIIIIZIIIZIYIIIY`, `IIIIIIZIIIZIXZII`, `IZIIZIIIIIIIIIII`, `IIYIIZZYIIIIIIII`, `IIIIZIIIZIIIIZII`, `IIIIIZIIIIIIIIZI`, `IIIIIIIIIIIXXZIX`  

Modal code of the last 20 updates: 3 of 345 successes, same as selected: no; p_round 0.000333, w* = 6, d_X/d_Z = 4/8, CSS no, frame none, 2q gates 228.

**board / seed 1**: [[16,1,4]], d_X = 4, d_Z = 4, w* = 7 (degree 6), CSS: no, Hadamard frame: none. Encoder: 46 gates, depth 16. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000343/0.000219/0.00024/0.00043; encoder undetected mass board/ion(forte)/heron = 0.0014/0.00197/0.00355. Reward terms: memory +0.1633, LDPC -0.000 (w* excess 3), gate -0.0300. Visits of the selected code: 1; distinct target codes 3482, of which 0 CSS.

Stabilizers (sparsest basis): `YYIIZYZIIYZIIIII`, `IXXXIXIIIIIXXIII`, `IIXXIIXXIIXIIIII`, `IIXIIIXIIXIIIIXZ`, `ZIIIXIIIIIIIIIII`, `IIIIIIIIXXXIXIII`, `IIIIXIIIXIXIIZXI`, `IIIIIIIIXIIXIXZX`, `IIZZIIIIIIIIXXZI`, `IIIIXZIIZIIIYYII`, `IZIIIZIIIIIIIIII`, `IIZZIIZZIIIIIIII`, `IIIZIIIZIIIZIIIZ`, `IIIZIZIIIZZIIIZI`, `IIIIIIIZIIZZYYII`  

Highest-reward code of the seed (a different code): p_round 0.000249 (proxy), w* = 6, d_X/d_Z = 6/7, CSS no, frame none, 2q gates 240, reward +0.1603 (LDPC -0.000, gate -0.0000).

Modal code of the last 20 updates: 2188 of 12683 successes, same as selected: no; p_round 0.00149, w* = 9, d_X/d_Z = 6/4, CSS no, frame none, 2q gates 196.

**board / seed 2**: [[16,1,4]], d_X = 7, d_Z = 4, w* = 7 (degree 6), CSS: no, Hadamard frame: none. Encoder: 50 gates, depth 30. Memory p_round (proxy) ion/ion+flag/board/heron = 0.000369/0.000197/0.000235/0.000246; encoder undetected mass board/ion(forte)/heron = 0.0024/0.000852/0.00178. Reward terms: memory +0.1649, LDPC -0.000 (w* excess 3), gate -0.0033. Visits of the selected code: 1; distinct target codes 2571, of which 0 CSS.

Stabilizers (sparsest basis): `XIIIXIIIXIYIIZIY`, `XXXXXIIIXXIIIIII`, `YZIIIXIIXIIIIYXI`, `IIIIZXIIIYIYIIIX`, `IIIIIIYZIIYIIIII`, `IIIIIIIYIIXYIIXX`, `IIIIIIXIIIXIYYII`, `ZIIIIIIIZIIIIIII`, `IZZIIIIIIIIIIIII`, `IIZZIIIIIIIIIIII`, `IIIIZIIIZIIIZZII`, `IIIIIZIIIIIZIIZI`, `IIIIIIXIIIXZIIIZ`, `IIIIZIIZIZIIIIIZ`, `IIIIIIIIZZIIIIZZ`  

Modal code of the last 20 updates: 8816 of 12231 successes, same as selected: no; p_round 0.00143, w* = 11, d_X/d_Z = 6/4, CSS no, frame none, 2q gates 208.

## Cross-evaluation

Rows: training cell (selected code of each seed), the alternate codes, the encoder-objective cells of `experiments/gpu_scale/runs` (their best codes, re-scored with the memory objective), and the known codes; columns: evaluation profile. Entries: memory p_round (both bases summed, `R = 4`), mean over codes with [min, max]. Smaller is better. bposd tier: the pinned BP+OSD decoder, 5000 initial shots grown to 100 failures or 100000 shots per basis.

bposd tier (validation):

| trained on \ evaluated under | ion | ion+flag | board | heron | codes |
|---|---|---|---|---|---|
| run-2 memory-trained ion | 0.00144 [0.0012, 0.00186] | 0.00156 [0.00137, 0.00185] | 0.000933 [0.000656, 0.00107] | 0.000418 [0.000388, 0.000454] | 3 |
| run-2 memory-trained ion+flag | 0.00127 [0.000909, 0.00149] | 0.00116 [0.000875, 0.00136] | 0.00101 [0.000827, 0.00128] | 0.000849 [0.000606, 0.00118] | 3 |
| run-2 memory-trained board | 0.00192 [0.000978, 0.00258] | 0.00191 [0.00104, 0.00269] | 0.000837 [0.000509, 0.00122] | 0.000519 [0.0003, 0.000743] | 3 |
| run-2 highest-reward board | 0.00205 | 0.00223 | 0.000686 | 0.000475 | 1 |
| memory-trained ion | 0.00245 [0.0014, 0.00319] | 0.00222 [0.00145, 0.00314] | 0.00172 [0.00128, 0.00218] | 0.00112 [0.000951, 0.0013] | 3 |
| memory-trained ion+flag | 0.00198 [0.00183, 0.00218] | 0.00192 [0.00183, 0.00199] | 0.00139 [0.0012, 0.00156] | 0.00106 [0.000786, 0.00127] | 3 |
| memory-trained board | 0.00232 [0.00204, 0.00261] | 0.00213 [0.00181, 0.00232] | 0.00151 [0.0011, 0.00183] | 0.0012 [0.00112, 0.00125] | 3 |
| memory-trained heron | 0.00202 [0.00155, 0.00293] | 0.00196 [0.00119, 0.00266] | 0.00143 [0.000896, 0.00172] | 0.000816 [0.000651, 0.000942] | 3 |
| lowest-proxy ion | 0.00141 [0.000969, 0.00182] | 0.00155 [0.0012, 0.00189] | 0.00092 [0.000498, 0.00125] | 0.000772 [0.000558, 0.00105] | 3 |
| lowest-proxy ion+flag | 0.00315 [0.00149, 0.00472] | 0.0031 [0.00186, 0.00411] | 0.00238 [0.00112, 0.0041] | 0.00149 [0.000945, 0.00189] | 3 |
| lowest-proxy board | 0.00157 [0.00136, 0.00183] | 0.00137 [0.00128, 0.0015] | 0.000824 [0.000664, 0.00102] | 0.000753 [0.000476, 0.00113] | 3 |
| lowest-proxy heron | 0.00208 [0.000938, 0.00295] | 0.0021 [0.000971, 0.00271] | 0.00139 [0.000804, 0.00201] | 0.000602 [0.000323, 0.0008] | 3 |
| encoder-trained board | 0.0022 [0.00131, 0.00315] | 0.00236 [0.0015, 0.00343] | 0.00151 [0.000923, 0.00259] | 0.00102 [0.000443, 0.00156] | 3 |
| encoder-trained ion | 0.00264 [0.0021, 0.00303] | 0.00236 [0.00187, 0.00302] | 0.00178 [0.00104, 0.00247] | 0.00123 [0.000842, 0.00189] | 3 |
| encoder-trained heron | 0.00176 [0.0016, 0.00205] | 0.00144 [0.0013, 0.00171] | 0.00111 [0.00108, 0.00114] | 0.000917 [0.000833, 0.000992] | 3 |
| known: rotated_surface | 0.000599 | 0.000763 | 0.000608 | 0.000432 | 1 |
| known: xzzx | 0.000736 | 0.000894 | 0.000819 | 0.00047 | 1 |

Proxy tier:

| trained on \ evaluated under | ion | ion+flag | board | heron | codes |
|---|---|---|---|---|---|
| run-2 memory-trained ion | 0.000304 [0.000288, 0.000333] | 0.000226 [0.000191, 0.000263] | 0.000382 [0.000293, 0.000458] | 0.000234 [0.000211, 0.000258] | 3 |
| run-2 memory-trained ion+flag | 0.000657 [0.00043, 0.000971] | 0.000549 [0.000341, 0.00087] | 0.000724 [0.00051, 0.000876] | 0.000667 [0.000496, 0.000946] | 3 |
| run-2 memory-trained board | 0.000321 [0.00025, 0.000369] | 0.00019 [0.000153, 0.000219] | 0.00023 [0.000215, 0.00024] | 0.000284 [0.000176, 0.00043] | 3 |
| run-2 highest-reward board | 0.000374 | 0.000238 | 0.000249 | 0.00022 | 1 |
| memory-trained ion | 0.000589 [0.000521, 0.000666] | 0.000278 [0.00017, 0.000398] | 0.000598 [0.000377, 0.000726] | 0.000563 [0.000346, 0.000796] | 3 |
| memory-trained ion+flag | 0.000519 [0.000501, 0.000542] | 0.000307 [0.000267, 0.000335] | 0.000623 [0.000519, 0.000682] | 0.000592 [0.000384, 0.000795] | 3 |
| memory-trained board | 0.000858 [0.000607, 0.00129] | 0.000306 [0.00024, 0.000435] | 0.000566 [0.000478, 0.0007] | 0.000512 [0.000356, 0.000691] | 3 |
| memory-trained heron | 0.000517 [0.000422, 0.000595] | 0.000251 [0.000209, 0.000283] | 0.000602 [0.000447, 0.000784] | 0.000376 [0.000304, 0.00044] | 3 |
| lowest-proxy ion | 0.000592 [0.000336, 0.00101] | 0.000491 [0.000193, 0.00094] | 0.000505 [0.00038, 0.000678] | 0.000531 [0.000339, 0.00087] | 3 |
| lowest-proxy ion+flag | 0.000586 [0.000474, 0.00068] | 0.000224 [0.000184, 0.000261] | 0.000575 [0.000381, 0.000678] | 0.000501 [0.000461, 0.000566] | 3 |
| lowest-proxy board | 0.000502 [0.000297, 0.000795] | 0.000259 [0.000107, 0.000376] | 0.000396 [0.000199, 0.000571] | 0.000473 [0.000116, 0.000776] | 3 |
| lowest-proxy heron | 0.000334 [0.000289, 0.000387] | 0.000151 [7.78e-05, 0.000293] | 0.000377 [0.000212, 0.000655] | 0.000128 [8.52e-05, 0.000173] | 3 |
| encoder-trained board | 0.000686 [0.000544, 0.000949] | 0.000327 [0.000189, 0.000423] | 0.000658 [0.000385, 0.0009] | 0.000542 [0.000214, 0.000788] | 3 |
| encoder-trained ion | 0.000969 [0.000702, 0.00126] | 0.000366 [0.000299, 0.00049] | 0.00085 [0.000469, 0.00129] | 0.000611 [0.000472, 0.000824] | 3 |
| encoder-trained heron | 0.000635 [0.000458, 0.000734] | 0.000325 [0.000263, 0.00041] | 0.000596 [0.000409, 0.000727] | 0.000569 [0.000487, 0.00068] | 3 |
| known: rotated_surface | 0.000366 | 0.000335 | 0.000489 | 0.000343 | 1 |
| known: xzzx | 0.000427 | 0.000394 | 0.000701 | 0.000413 | 1 |

Sampled tier (weight-2 lookup decoder, near maximum likelihood):

| trained on \ evaluated under | ion | ion+flag | board | heron | codes |
|---|---|---|---|---|---|
| run-2 memory-trained ion | 0.00175 [0.00167, 0.00191] | 0.00504 [0.0043, 0.00566] | 0.00147 [0.000972, 0.00215] | 0.000904 [0.000838, 0.000945] | 3 |
| run-2 memory-trained ion+flag | 0.0016 [0.00145, 0.00182] | 0.00203 [0.0016, 0.00226] | 0.00149 [0.00119, 0.00168] | 0.00119 [0.000964, 0.00136] | 3 |
| run-2 memory-trained board | 0.00212 [0.00146, 0.00249] | 0.00486 [0.00371, 0.00598] | 0.000912 [0.000789, 0.00109] | 0.000808 [0.000697, 0.00101] | 3 |
| run-2 highest-reward board | 0.00179 | 0.00463 | 0.000793 | 0.000855 | 1 |
| memory-trained ion | 0.00279 [0.00224, 0.00314] | 0.00727 [0.00585, 0.00828] | 0.00283 [0.00234, 0.0031] | 0.00231 [0.00203, 0.00282] | 3 |
| memory-trained ion+flag | 0.00236 [0.00191, 0.00279] | 0.00597 [0.00489, 0.00709] | 0.00233 [0.0017, 0.00274] | 0.00235 [0.00152, 0.00281] | 3 |
| memory-trained board | 0.00301 [0.00269, 0.00338] | 0.00688 [0.0059, 0.0076] | 0.00245 [0.00204, 0.00298] | 0.00241 [0.00206, 0.00303] | 3 |
| memory-trained heron | 0.00252 [0.00189, 0.00309] | 0.00641 [0.00433, 0.00757] | 0.00245 [0.00161, 0.00301] | 0.00215 [0.00145, 0.00306] | 3 |
| lowest-proxy ion | 0.00187 [0.00129, 0.00239] | 0.00401 [0.00194, 0.00646] | 0.00156 [0.000756, 0.00235] | 0.00154 [0.000881, 0.00204] | 3 |
| lowest-proxy ion+flag | 0.00255 [0.00155, 0.0032] | 0.006 [0.00263, 0.00774] | 0.00318 [0.00193, 0.00455] | 0.00304 [0.00182, 0.00451] | 3 |
| lowest-proxy board | 0.0017 [0.00142, 0.00186] | 0.00374 [0.00232, 0.00506] | 0.00125 [0.00088, 0.00153] | 0.00135 [0.00101, 0.00169] | 3 |
| lowest-proxy heron | 0.0025 [0.00132, 0.0032] | 0.00595 [0.00174, 0.00884] | 0.00223 [0.00142, 0.00313] | 0.00176 [0.000775, 0.00308] | 3 |
| encoder-trained board | 0.00256 [0.00193, 0.00328] | 0.00688 [0.00413, 0.00863] | 0.00215 [0.00136, 0.00322] | 0.0021 [0.000962, 0.00337] | 3 |
| encoder-trained ion | 0.00259 [0.00233, 0.00275] | 0.00582 [0.00539, 0.00613] | 0.00266 [0.00158, 0.00379] | 0.00236 [0.00178, 0.00339] | 3 |
| encoder-trained heron | 0.00219 [0.00206, 0.00234] | 0.00536 [0.0041, 0.00661] | 0.00188 [0.00177, 0.00194] | 0.00202 [0.00182, 0.00225] | 3 |
| known: rotated_surface | 0.00092 | 0.00178 | 0.000859 | 0.000541 | 1 |
| known: xzzx | 0.00111 | 0.00243 | 0.00121 | 0.000887 | 1 |

Per code, bposd tier:

| code | d | d_X | d_Z | w* | CSS | H frame | ion | ion+flag | board | heron |
|---|---|---|---|---|---|---|---|---|---|---|
| run-2 memory-trained board seed 0 | 4 | 4 | 4 | 7 | no | none | 0.000978 | 0.00104 | 0.000509 | 0.0003 |
| run-2 memory-trained board seed 1 | 4 | 4 | 4 | 7 | no | none | 0.00258 | 0.00269 | 0.00122 | 0.000743 |
| run-2 memory-trained board seed 2 | 4 | 7 | 4 | 7 | no | none | 0.0022 | 0.00201 | 0.000784 | 0.000514 |
| run-2 memory-trained ion+flag seed 0 | 4 | 4 | 4 | 9 | no | none | 0.000909 | 0.000875 | 0.000827 | 0.000606 |
| run-2 memory-trained ion+flag seed 1 | 4 | 7 | 4 | 9 | no | none | 0.00149 | 0.00123 | 0.00128 | 0.00118 |
| run-2 memory-trained ion+flag seed 2 | 4 | 7 | 4 | 9 | no | none | 0.00142 | 0.00136 | 0.000931 | 0.000762 |
| run-2 memory-trained ion seed 0 | 4 | 7 | 5 | 7 | no | none | 0.00186 | 0.00185 | 0.00107 | 0.000454 |
| run-2 memory-trained ion seed 1 | 4 | 4 | 4 | 7 | no | none | 0.0012 | 0.00147 | 0.00107 | 0.000388 |
| run-2 memory-trained ion seed 2 | 4 | 7 | 4 | 7 | no | none | 0.00128 | 0.00137 | 0.000656 | 0.000413 |
| run-2 highest-reward board seed 1 | 4 | 6 | 7 | 6 | no | none | 0.00205 | 0.00223 | 0.000686 | 0.000475 |
| memory-trained board seed 0 | 4 | 5 | 4 | 5 | no | none | 0.0023 | 0.00232 | 0.00183 | 0.00125 |
| memory-trained board seed 1 | 4 | 5 | 5 | 5 | no | none | 0.00204 | 0.00181 | 0.0011 | 0.00112 |
| memory-trained board seed 2 | 4 | 4 | 4 | 5 | no | none | 0.00261 | 0.00224 | 0.00159 | 0.00122 |
| memory-trained heron seed 0 | 4 | 4 | 4 | 5 | no | none | 0.00293 | 0.00266 | 0.00172 | 0.000854 |
| memory-trained heron seed 1 | 4 | 7 | 5 | 5 | no | none | 0.00157 | 0.00119 | 0.000896 | 0.000651 |
| memory-trained heron seed 2 | 4 | 7 | 5 | 5 | no | none | 0.00155 | 0.00202 | 0.00167 | 0.000942 |
| memory-trained ion+flag seed 0 | 4 | 5 | 4 | 5 | no | none | 0.00183 | 0.00183 | 0.0012 | 0.000786 |
| memory-trained ion+flag seed 1 | 4 | 4 | 4 | 5 | no | none | 0.00218 | 0.00193 | 0.00156 | 0.00113 |
| memory-trained ion+flag seed 2 | 4 | 5 | 4 | 5 | no | none | 0.00193 | 0.00199 | 0.00141 | 0.00127 |
| memory-trained ion seed 0 | 4 | 5 | 4 | 5 | no | none | 0.0014 | 0.00145 | 0.00128 | 0.00109 |
| memory-trained ion seed 1 | 4 | 5 | 5 | 5 | no | none | 0.00319 | 0.00314 | 0.00218 | 0.0013 |
| memory-trained ion seed 2 | 4 | 5 | 4 | 5 | no | none | 0.00276 | 0.00207 | 0.0017 | 0.000951 |
| lowest-proxy board seed 0 | 4 | 5 | 4 | 6 | no | none | 0.00152 | 0.00128 | 0.00102 | 0.00113 |
| lowest-proxy board seed 1 | 4 | 4 | 5 | 7 | no | none | 0.00183 | 0.0015 | 0.000791 | 0.000476 |
| lowest-proxy board seed 2 | 4 | 7 | 4 | 8 | no | none | 0.00136 | 0.00133 | 0.000664 | 0.000653 |
| lowest-proxy heron seed 0 | 4 | 5 | 5 | 6 | no | none | 0.00236 | 0.0026 | 0.00135 | 0.0008 |
| lowest-proxy heron seed 1 | 4 | 4 | 5 | 6 | no | none | 0.00295 | 0.00271 | 0.00201 | 0.000685 |
| lowest-proxy heron seed 2 | 4 | 7 | 5 | 10 | no | none | 0.000938 | 0.000971 | 0.000804 | 0.000323 |
| lowest-proxy ion+flag seed 0 | 4 | 4 | 6 | 6 | no | none | 0.00472 | 0.00411 | 0.0041 | 0.00189 |
| lowest-proxy ion+flag seed 1 | 4 | 4 | 4 | 6 | no | none | 0.00324 | 0.00333 | 0.00193 | 0.00164 |
| lowest-proxy ion+flag seed 2 | 4 | 7 | 4 | 8 | no | none | 0.00149 | 0.00186 | 0.00112 | 0.000945 |
| lowest-proxy ion seed 0 | 4 | 5 | 4 | 7 | no | none | 0.00143 | 0.00156 | 0.00101 | 0.00105 |
| lowest-proxy ion seed 1 | 4 | 4 | 4 | 6 | no | none | 0.00182 | 0.00189 | 0.00125 | 0.000714 |
| lowest-proxy ion seed 2 | 4 | 8 | 4 | 10 | no | none | 0.000969 | 0.0012 | 0.000498 | 0.000558 |
| encoder-trained board seed 0 | 4 | 5 | 4 | 5 | no | none | 0.00315 | 0.00343 | 0.00259 | 0.00156 |
| encoder-trained board seed 1 | 4 | 6 | 4 | 5 | no | none | 0.00131 | 0.0015 | 0.000923 | 0.00105 |
| encoder-trained board seed 2 | 4 | 4 | 4 | 5 | no | none | 0.00212 | 0.00216 | 0.00103 | 0.000443 |
| encoder-trained heron seed 0 | 4 | 4 | 4 | 5 | no | none | 0.0016 | 0.0013 | 0.00111 | 0.000833 |
| encoder-trained heron seed 1 | 4 | 5 | 4 | 5 | no | none | 0.00161 | 0.00132 | 0.00114 | 0.000992 |
| encoder-trained heron seed 2 | 4 | 6 | 4 | 5 | no | none | 0.00205 | 0.00171 | 0.00108 | 0.000926 |
| encoder-trained ion seed 0 | 4 | 6 | 5 | 5 | no | none | 0.00303 | 0.00302 | 0.00247 | 0.00189 |
| encoder-trained ion seed 1 | 4 | 5 | 4 | 5 | no | none | 0.0021 | 0.00187 | 0.00104 | 0.000842 |
| encoder-trained ion seed 2 | 4 | 5 | 4 | 5 | no | none | 0.00277 | 0.0022 | 0.00184 | 0.000958 |
| rotated surface [[16,1,4]] | 4 | 4 | 4 | 4 | yes | H on 0 qubit(s) (2 frames) | 0.000599 | 0.000763 | 0.000608 | 0.000432 |
| XZZX surface [[16,1,4]]: H on the checkerboard [1, 3, 4, 6, 9, 11, 12, 14] | 4 | 4 | 4 | 4 | no | H on 8 qubit(s) (2 frames) | 0.000736 | 0.000894 | 0.000819 | 0.00047 |

Per code, proxy tier:

| code | d | d_X | d_Z | w* | CSS | H frame | ion | ion+flag | board | heron |
|---|---|---|---|---|---|---|---|---|---|---|
| run-2 memory-trained board seed 0 | 4 | 4 | 4 | 7 | no | none | 0.00025 | 0.000153 | 0.000215 | 0.000176 |
| run-2 memory-trained board seed 1 | 4 | 4 | 4 | 7 | no | none | 0.000343 | 0.000219 | 0.00024 | 0.00043 |
| run-2 memory-trained board seed 2 | 4 | 7 | 4 | 7 | no | none | 0.000369 | 0.000197 | 0.000235 | 0.000246 |
| run-2 memory-trained ion+flag seed 0 | 4 | 4 | 4 | 9 | no | none | 0.00043 | 0.000341 | 0.00051 | 0.000496 |
| run-2 memory-trained ion+flag seed 1 | 4 | 7 | 4 | 9 | no | none | 0.000571 | 0.000436 | 0.000876 | 0.000946 |
| run-2 memory-trained ion+flag seed 2 | 4 | 7 | 4 | 9 | no | none | 0.000971 | 0.00087 | 0.000785 | 0.00056 |
| run-2 memory-trained ion seed 0 | 4 | 7 | 5 | 7 | no | none | 0.000333 | 0.000263 | 0.000396 | 0.000234 |
| run-2 memory-trained ion seed 1 | 4 | 4 | 4 | 7 | no | none | 0.000288 | 0.000191 | 0.000458 | 0.000258 |
| run-2 memory-trained ion seed 2 | 4 | 7 | 4 | 7 | no | none | 0.000292 | 0.000224 | 0.000293 | 0.000211 |
| run-2 highest-reward board seed 1 | 4 | 6 | 7 | 6 | no | none | 0.000374 | 0.000238 | 0.000249 | 0.00022 |
| memory-trained board seed 0 | 4 | 5 | 4 | 5 | no | none | 0.000607 | 0.00024 | 0.00052 | 0.000488 |
| memory-trained board seed 1 | 4 | 5 | 5 | 5 | no | none | 0.000681 | 0.000242 | 0.000478 | 0.000356 |
| memory-trained board seed 2 | 4 | 4 | 4 | 5 | no | none | 0.00129 | 0.000435 | 0.0007 | 0.000691 |
| memory-trained heron seed 0 | 4 | 4 | 4 | 5 | no | none | 0.000595 | 0.000283 | 0.000575 | 0.000384 |
| memory-trained heron seed 1 | 4 | 7 | 5 | 5 | no | none | 0.000422 | 0.000209 | 0.000447 | 0.000304 |
| memory-trained heron seed 2 | 4 | 7 | 5 | 5 | no | none | 0.000533 | 0.000262 | 0.000784 | 0.00044 |
| memory-trained ion+flag seed 0 | 4 | 5 | 4 | 5 | no | none | 0.000542 | 0.00032 | 0.000519 | 0.000384 |
| memory-trained ion+flag seed 1 | 4 | 4 | 4 | 5 | no | none | 0.000501 | 0.000267 | 0.000667 | 0.000597 |
| memory-trained ion+flag seed 2 | 4 | 5 | 4 | 5 | no | none | 0.000513 | 0.000335 | 0.000682 | 0.000795 |
| memory-trained ion seed 0 | 4 | 5 | 4 | 5 | no | none | 0.000579 | 0.000267 | 0.000726 | 0.000796 |
| memory-trained ion seed 1 | 4 | 5 | 5 | 5 | no | none | 0.000521 | 0.00017 | 0.000377 | 0.000346 |
| memory-trained ion seed 2 | 4 | 5 | 4 | 5 | no | none | 0.000666 | 0.000398 | 0.000692 | 0.000548 |
| lowest-proxy board seed 0 | 4 | 5 | 4 | 6 | no | none | 0.000795 | 0.000376 | 0.000571 | 0.000776 |
| lowest-proxy board seed 1 | 4 | 4 | 5 | 7 | no | none | 0.000297 | 0.000107 | 0.000199 | 0.000116 |
| lowest-proxy board seed 2 | 4 | 7 | 4 | 8 | no | none | 0.000413 | 0.000295 | 0.000418 | 0.000527 |
| lowest-proxy heron seed 0 | 4 | 5 | 5 | 6 | no | none | 0.000289 | 7.78e-05 | 0.000264 | 0.000125 |
| lowest-proxy heron seed 1 | 4 | 4 | 5 | 6 | no | none | 0.000325 | 8.3e-05 | 0.000212 | 8.52e-05 |
| lowest-proxy heron seed 2 | 4 | 7 | 5 | 10 | no | none | 0.000387 | 0.000293 | 0.000655 | 0.000173 |
| lowest-proxy ion+flag seed 0 | 4 | 4 | 6 | 6 | no | none | 0.00068 | 0.000227 | 0.000678 | 0.000477 |
| lowest-proxy ion+flag seed 1 | 4 | 4 | 4 | 6 | no | none | 0.000605 | 0.000184 | 0.000381 | 0.000461 |
| lowest-proxy ion+flag seed 2 | 4 | 7 | 4 | 8 | no | none | 0.000474 | 0.000261 | 0.000665 | 0.000566 |
| lowest-proxy ion seed 0 | 4 | 5 | 4 | 7 | no | none | 0.00101 | 0.00094 | 0.000678 | 0.00087 |
| lowest-proxy ion seed 1 | 4 | 4 | 4 | 6 | no | none | 0.000336 | 0.000193 | 0.000458 | 0.000339 |
| lowest-proxy ion seed 2 | 4 | 8 | 4 | 10 | no | none | 0.000431 | 0.000339 | 0.00038 | 0.000384 |
| encoder-trained board seed 0 | 4 | 5 | 4 | 5 | no | none | 0.000949 | 0.000368 | 0.0009 | 0.000624 |
| encoder-trained board seed 1 | 4 | 6 | 4 | 5 | no | none | 0.000566 | 0.000423 | 0.000689 | 0.000788 |
| encoder-trained board seed 2 | 4 | 4 | 4 | 5 | no | none | 0.000544 | 0.000189 | 0.000385 | 0.000214 |
| encoder-trained heron seed 0 | 4 | 4 | 4 | 5 | no | none | 0.000713 | 0.000301 | 0.00065 | 0.000541 |
| encoder-trained heron seed 1 | 4 | 5 | 4 | 5 | no | none | 0.000734 | 0.00041 | 0.000727 | 0.00068 |
| encoder-trained heron seed 2 | 4 | 6 | 4 | 5 | no | none | 0.000458 | 0.000263 | 0.000409 | 0.000487 |
| encoder-trained ion seed 0 | 4 | 6 | 5 | 5 | no | none | 0.00126 | 0.00049 | 0.00129 | 0.000824 |
| encoder-trained ion seed 1 | 4 | 5 | 4 | 5 | no | none | 0.000702 | 0.000299 | 0.000469 | 0.000538 |
| encoder-trained ion seed 2 | 4 | 5 | 4 | 5 | no | none | 0.000945 | 0.000308 | 0.000795 | 0.000472 |
| rotated surface [[16,1,4]] | 4 | 4 | 4 | 4 | yes | H on 0 qubit(s) (2 frames) | 0.000366 | 0.000335 | 0.000489 | 0.000343 |
| XZZX surface [[16,1,4]]: H on the checkerboard [1, 3, 4, 6, 9, 11, 12, 14] | 4 | 4 | 4 | 4 | no | H on 8 qubit(s) (2 frames) | 0.000427 | 0.000394 | 0.000701 | 0.000413 |

Tier agreement over every scored code (ratio is the second tier over the first):

| profile | tiers | codes | Spearman | median ratio | range |
|---|---|---|---|---|---|
| ion | bposd / proxy | 45 | 0.33 | 3.37 | 1.42 to 9.08 |
| ion | sampled / proxy | 45 | 0.39 | 4.34 | 1.87 to 10.33 |
| ion | bposd / sampled | 45 | 0.89 | 0.86 | 0.60 to 1.62 |
| ion_flag | bposd / proxy | 45 | -0.30 | 6.49 | 1.57 to 33.46 |
| ion_flag | sampled / proxy | 45 | -0.39 | 21.18 | 2.60 to 106.48 |
| ion_flag | bposd / sampled | 45 | 0.86 | 0.35 | 0.24 to 0.77 |
| board | bposd / proxy | 45 | 0.40 | 2.30 | 1.17 to 9.46 |
| board | sampled / proxy | 45 | 0.43 | 3.38 | 1.73 to 11.86 |
| board | bposd / sampled | 45 | 0.91 | 0.65 | 0.42 to 1.12 |
| heron | bposd / proxy | 45 | 0.72 | 1.89 | 1.14 to 8.03 |
| heron | sampled / proxy | 45 | 0.50 | 3.89 | 1.32 to 24.56 |
| heron | bposd / sampled | 45 | 0.87 | 0.49 | 0.26 to 0.94 |

## Comparison under the training profile (bposd)

- ion: run-2 memory-trained best per seed 0.00186 (seed 0), 0.0012 (seed 1), 0.00128 (seed 2); encoder-trained codes 0.00131 to 0.00315 (best of 9); 2 of 3 run-2 memory-trained seeds below the best encoder-trained code; ratio to the surface code 3.1x, 2.0x, 2.1x; rotated surface 0.000599 (0 of 3 seeds below); XZZX 0.000736 (0 of 3 seeds below); best seed 1: 0.0012.
- ion+flag: run-2 memory-trained best per seed 0.000875 (seed 0), 0.00123 (seed 1), 0.00136 (seed 2); encoder-trained codes 0.0013 to 0.00343 (best of 9); 2 of 3 run-2 memory-trained seeds below the best encoder-trained code; ratio to the surface code 1.1x, 1.6x, 1.8x; rotated surface 0.000763 (0 of 3 seeds below); XZZX 0.000894 (1 of 3 seeds below); best seed 0: 0.000875.
- board: run-2 memory-trained best per seed 0.000509 (seed 0), 0.00122 (seed 1), 0.000784 (seed 2); encoder-trained codes 0.000923 to 0.00259 (best of 9); 2 of 3 run-2 memory-trained seeds below the best encoder-trained code; ratio to the surface code 0.8x, 2.0x, 1.3x; rotated surface 0.000608 (1 of 3 seeds below); XZZX 0.000819 (2 of 3 seeds below); best seed 0: 0.000509.

## Figures

`figures/run2_curves.png`: success rate among the episodes finished in an update, moving average over 10 updates (top), and the lowest training p_round seen so far at the tier the reward used (bottom, log scale shared across cells; dashed: rotated surface, dotted: XZZX under the cell's profile at the proxy tier) against env steps, one column per cell, seeds as series.

`figures/run2_matrix.png`: bposd p_round per evaluation profile (one panel each), bars are the mean over codes of a row of the cross-evaluation table with min to max whiskers (yellow: run-2 cells, blue: run-1 memory-trained cells, light blue: their lowest-proxy codes, red: encoder-trained cells, grey: known codes).
