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
tiers, all on the sparsest basis (`results/memory_ppo/run2/scores.json`).

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
