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
