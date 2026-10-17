# Memory objective: design note

Module `src/autoqec_sim/memory_objective.py`; experiment
`src/autoqec_sim/experiments/memory_objective.py`; results
`experiments/memory_objective/REPORT.md` (every number below is copied from
there, which regenerates from `experiments/memory_objective/runs/*.json`).

## Why

The circuit-level reward of `circuit_noise` scores the encoding circuit:
every single fault of the noisy encoder is propagated to the output and
tested with the Knill-Laflamme [@knill1997] kernel. A Hadamard on a data qubit at the
end of the encoder relabels the weight-1 faults of the last layer, all of
which the code detects, so the objective cannot distinguish a code from its
local-Clifford deformations (the rotated surface code, XZZX [@bonillaataides2021], and the
outer-column deformation tie exactly in `experiments/noise_aware/REPORT.md`,
verdict items 3, 4, and 6). Bias tailoring is a statement about how the
code's stabilizers align with the dominant fault type during repeated
syndrome extraction, so the reward has to be a memory experiment.

## What the objective computes

Input: a `StabilizerCode` (stabilizer generators, logical operators; CSS or
not), a `autoqec_sim.noise.Profile` with its timing model, a scale, a number of
rounds `R` (default the target distance), a layout, and a tier.

1. Circuit (`build_memory_circuit`). One ancilla per generator, prepared
   in `|+>`, controlled-X, -Y, or -Z from the ancilla onto each support
   qubit in a chosen order, `H`, measurement in the Z basis. By default
   every generator runs in one interleaved block with one measurement per
   round (`extraction="auto"`): the orders are chosen jointly among the
   hook-safe ones so that no qubit sees two gates in one layer and, for
   every pair of generators sharing qubits with different letters, an even
   number of those qubits sees one generator first, the condition for each
   ancilla to measure its own generator (an odd number multiplies the
   measured operator by `Z` of the other ancilla, which starts in `|+>`).
   The standard four-layer surface-code cycle is one such schedule, and
   every surface code tried, the Shor code, and the [[5,1,3]] code
   schedule; the Steane code does not within the search cap and falls back
   to batches of mutually interleavable generators (for a CSS code the X
   block and the Z block), one after the other with their own reset and
   measurement layers, which `extraction="batched"` forces and which was
   the only schedule before this option (the numbers of this page come
   from it). Inside a batch the gates are packed into layers as soon as
   possible with one gate per qubit per layer. Preparation and final readout are noiseless `MPP`
   measurements of every generator and of the logical operators of the
   memory basis (`Z` or `X`); detectors compare consecutive measurements of
   each generator (round 0 against the preparation, the final readout
   against the last round), the observables compare the initial and final
   logical readout. Both bases are evaluated and the per-round failure
   probabilities are summed. The noiseless circuit has deterministic
   detectors for every code (test).
2. Gate order (`hook_orders`, default). An X fault on the ancilla after
   gate `t` lands on the data as the product of the remaining letters of
   the generator (the hook). For each generator the permutations of the
   support are enumerated (sampled above weight 6) and the first is taken
   whose every proper suffix `S` satisfies `1 + cost(S) >= R` and every
   contiguous segment (two ancilla faults) `2 + cost(S) >= R`, with
   `cost(S)` the fewest extra single-qubit Paulis that complete `S` to an
   undetected nontrivial logical (exact, from the Knill-Laflamme kernel,
   cached per Pauli). This is the criterion of the ion-search schedule
   search. Without it the naive sorted order gives the Shor code a
   circuit-level distance of 1 in the Z basis (the hook of the weight-6
   check is a logical operator) and the rotated surface code 2 in the X
   basis; with it both are 3. The order is a property of the code and is
   shared by the two bases.
3. Flags (`flags="one"`). Each generator of weight at least three gets a
   flag qubit in `|0>` with a CNOT from the ancilla after the first data
   gate and another before the last, measured and declared a detector
   (Chao-Reichardt). Lifts the bare-ancilla `d_circ` of the [[5,1,3]] code
   from 2 to 3 (test). Needs all-to-all connectivity in this layout.
4. Layout. `all_to_all` (the ion chain: ancilla `j` on qubit `n + j`) or
   `square:RxC` (data on the lattice, ancilla sites at every plaquette
   center including the boundary ring, edges only between a site and its
   corner data qubits, the rotated-surface hardware layout). Generators
   take sites by a greedy minimum of the path lengths to their support; a
   controlled gate to a non-adjacent qubit is routed along a shortest path
   with the nearest-neighbour CNOT ladder of the noise-aware baselines
   (intermediate qubits restored), and a routed generator runs in a batch
   of its own. The rotated surface code and XZZX route nothing; the Shor
   code and the discovered non-CSS 3x3 code pay in gates and depth. The
   profile's graph is replaced by the layout's graph, so
   `autoqec_sim.noise.apply` checks the routed circuit.
5. Noise and DEM. The body (the `R` rounds) goes through
   `autoqec_sim.noise.apply` with the profile's timing (`sequential` for the ion
   presets: every two-qubit gate its own slot, idle noise on every other
   qubit), `n_data = n` for the board recipe; stim's [@gidney2021] detector error model
   (`autoqec_sim.noise.dem`, loops flattened, no decomposition) gives `M`
   mechanisms with detector and observable signatures.
6. Tiers.
   - `proxy`: `pfail_low_weight`, the failure probability of the
     maximum-likelihood decoder restricted to fault sets of weight at most
     two: singles and pairs are grouped by syndrome, the decoder takes the
     observable pattern of largest summed weight per syndrome (no flip for
     the zero syndrome), everything else fails. Weight of a set:
     `prod p_i` (the low-`p` asymptote; the `exclusive=True` form uses the
     exact probability `prod p_i prod (1 - p_j)` and is used in the test
     against the exact decoder). `O(M^2)`. Plus `exact_dcirc`: the exact
     circuit-level distance up to weight 4 with the count and summed
     probability product of the minimum-weight undetectable logical fault
     sets (ported from `ion-search/lib.py`; signatures wider than 64
     detectors go through integer ids). When no undetectable logical set
     of weight at most 4 exists (a circuit distance of 5 or 6) the
     weight-2 form is zero, and the proxy escalates to weight 3 without a
     triple table: `undetectable_logicals` lists every weight-5 and
     weight-6 undetectable logical set by a cluster search over the
     detector hypergraph (a set is grown from its smallest member by
     resolving the lowest flipped detector with one of its incident
     mechanisms, the last two members come from the pair table, and a
     canonical rule gives each set one derivation), and their 3-subsets
     are the only triples the decoder can get wrong. The search visits
     about `M (c / 2)^3` states for `M` mechanisms and detector degree
     `c`: about a second for the 5x5 surface code at one round (475
     mechanisms) and 20 s per basis at five rounds (2,107 mechanisms,
     2.3e7 states), where 20,000 Tesseract shots take 45 s.
     `exact_dcirc(sig, wmax=6)` and `undetectable_sets(sig, wmax=6)` use
     the same search. Weight-3 sets are also enumerated directly when the
     number of triples is small (`proxy_max_weight=3`).
   - `sampled`: when `D + O <= 20` (detectors plus observables) the exact
     maximum-likelihood decoder by dynamic program over the joint
     (syndrome, observable) distribution, whose failure probability is
     exact without shots (`exact_ml_failure`; the construction of
     `ion-search/decoder/exact_decoder.py`); otherwise a lookup decoder
     built from the same weight-`<= 2` fault sets with a one-mechanism
     completion for unseen syndromes, sampled from the DEM with distinct
     syndromes decoded once.
   - `bposd`: the board's pinned BP+OSD decoder [@roffe2020; @panteleev2021] (`ldpc`, minimum-sum,
     30 iterations, scaling 0.625, OSD-CS order 10 clamped to the
     redundancy), sampled; validation only, optional dependency
     (`uv run --with ldpc`).
   - `tesseract`: Tesseract [@beni2025], a most-likely-error search
     decoder that takes the DEM without decomposition, sampled with every
     distinct syndrome decoded once; optional dependency (`tesseract`
     extra). On the rotated surface code under uniform depolarizing noise
     at 1e-3 it agrees with the sampled tier within shot noise (0.0085
     against 0.0088 per round at 20,000 shots per basis).
7. Per-round rate: `p_round = 1 - (1 - p_shot)^(1/R)` per basis, summed
   over the two bases. `MemoryResult` carries the per-basis dictionaries
   (mechanisms, detectors, gates, layers, batches, expected faults per
   shot, `d_circ`, decoder, shots) and the summary.

## Reward

`MemoryReward(objective, d*, weight=0.1, p_ref=1e-2, clip=3)`:
`r = min(d, d*)/d* - 1 - q_1` on every state (the distance and padding
terms of the noise-aware experiment) and, on states at the target distance
with no weight-1 stabilizer, `+ weight * clip(log10(p_ref / p_round))`;
such a state ends the episode (the reward exposes `success_fn`, which
`CodeDiscoveryEnv` adopts when none is given). A tenfold better memory rate
is worth 0.1, a third of one distance quantum at `d* = 3`, so the term
orders target-distance codes without competing with distance. Results are
cached by the reduced row echelon form of the stabilizer group, so each
distinct code is evaluated once per run. The term depends only on the code,
not on the encoding circuit that produced it, which is the point: the
encoder objective rewarded degenerate codes with weight-2 stabilizers and
saturated at about ten undetected mechanisms (`experiments/ion_chain/runs/REPORT.md`).

## Regime of validity and scales

The proxy is a leading-order expansion and the sampled tiers saturate once
the expected number of faults per shot passes about one. At scale 1 the
chain preset gives 0.6 to 3.4 expected faults per shot on the benchmark
circuits, the Heron preset 2.8 to 8.8 (reset and readout flips at 1.5e-2
and a 3.5 us readout layer against `T2 = 120 us`, 1.5 percent dephasing
per idle data qubit per readout layer), the board recipe 0.5 to 2.1. In the
saturated regime the proxy is meaningless (the non-CSS 3x3 code under Heron
at scale 1 on the lattice: proxy 1.8e-4, bposd 0.49) and bposd fails on a
third to a half of the shots. The experiment therefore evaluates the chain
preset at scale 0.25 (the labeled point of the large-code ion study), the
board recipe at 1 (its canonical `p = 1e-3`), Heron at 0.1, and the
biased-idle sweep at every rate 2.5e-4. `MemoryResult` reports the expected
fault count per shot so a caller can see when a code is outside the regime.

## Costs per state

Measured on the laptop (arm64, one BLAS thread, one process) for both
bases at `R = d`, all-to-all layout, hook-aware order; from
`experiments/memory_objective/runs/benchmark.json`. The DEM derivation is 40 to 60
ms per basis at `n = 9` and dominates the proxy tier; the hook-order search
is 1 to 2 ms at `n <= 9` and 0.1 s for the weight-6 checks of the
[[16,4,4]] code.

| code | profile | mechanisms | detectors | proxy s | sampled s (decoder) | bposd s (shots) |
|---|---|---|---|---|---|---|
| [[5,1,3]], `R = 3` | ion / board / heron | 225 to 255 | 16 | 0.08 to 0.14 | 0.16 to 0.23 (exact ML, `D + O = 17`) | 0.2 to 0.5 (10k to 15k) |
| Steane, `R = 3` | ion / board / heron | 405 | 24 | 0.08 to 0.16 | 0.17 to 0.25 (lookup) | 0.5 to 0.7 (10k to 30k) |
| rotated surface [[9,1,3]], `R = 3` | ion / board / heron | 359 | 32 | 0.10 to 0.19 | 0.17 to 0.28 (lookup) | 1.3 to 1.5 (20k to 100k) |
| non-CSS [[9,1,3]], `R = 3` | ion / board / heron | 363 to 370 | 32 | 0.10 to 0.22 | 0.18 to 0.31 (lookup) | 1.1 (10k to 40k) |
| board [[16,4,4]], `R = 4` | ion / board / heron | 1988 to 1992 | 60 | 1.7 to 2.3 | 4.8 to 5.6 (lookup) | 36 to 42 (5k to 30k) |

Inside PPO on the 3x3 lattice (`R = 3`, chain preset, all-to-all layout)
the reward evaluated 92 distinct target-distance codes at 0.47 s each
(hook-order search, two circuits, two DEMs, two proxies, `d_circ` off),
44 s of a 62 s run at one thread; the 92 cache hits cost nothing. The
noise-aware experiment saw about 23k distinct target-distance codes per
8M-step 3x3 run, so at 0.47 s per code the memory term adds about 3 hours
per such run on one thread (the DEM derivation is single-threaded stim;
the term parallelizes over codes across processes).

## Benchmark and rank correlation

Benchmark (`experiments/memory_objective/runs/benchmark.json`): five codes
([[5,1,3]], Steane, rotated surface [[9,1,3]], the board's [[16,4,4]] SAT
code, the non-CSS [[9,1,3]] of the noise-aware ion run), three profiles,
`R = d`, all-to-all, bare ancillas, hook-aware order. `p_round` per code
(proxy / sampled / bposd), with `E` the expected faults per shot:

| code | ion (chain 0.25) | board (1) | heron (0.1) |
|---|---|---|---|
| [[5,1,3]], `d_circ` 2 | 0.0099 / 0.0086 / 0.0112 (`E` 0.16) | 0.0255 / 0.0201 / 0.0293 (0.50) | 0.0100 / 0.0086 / 0.0092 (0.34) |
| Steane, `d_circ` 2 | 0.0040 / 0.0031 / 0.0067 (0.22) | 0.0097 / 0.0088 / 0.0173 (0.45) | 0.0031 / 0.0028 / 0.0100 (0.28) |
| rotated surface, `d_circ` 3 | 0.00077 / 0.00079 / 0.00088 (0.24) | 0.0077 / 0.0074 / 0.0074 (0.64) | 0.0022 / 0.0021 / 0.0022 (0.36) |
| [[16,4,4]], `d_circ` 2 | 0.0074 / 0.0091 / 0.0148 (0.85) | 0.024 / 0.075 / 0.056 (2.13) | 0.0054 / 0.0080 / 0.0085 (0.90) |
| non-CSS [[9,1,3]], `d_circ` 2 | 0.0024 / 0.0022 / 0.0028 (0.27) | 0.018 / 0.016 / 0.016 (0.80) | 0.0063 / 0.0063 / 0.0065 (0.49) |

The proxy is within 25 percent of the sampled tier (exact ML for the
[[5,1,3]] code, the lookup decoder otherwise) on every code below one
expected fault per shot, and the two rank the codes the same way on every
profile (Spearman 0.9 on five codes; the [[5,1,3]] and Steane rows are
within 1.15x of each other under Heron and swap). bposd deviates in two
places: Steane, where it is 1.7 to 3.2 times worse than the ML tiers on
every profile (the degenerate Steane DEM is a known weak case for BP), and
the [[16,4,4]] code under the ion preset (2x). The [[16,4,4]] row under
the board recipe (`E = 2.1`) is outside the regime: the lookup decoder
sees mostly unseen syndromes and the proxy underestimates bposd by 2.3x.
With bare ancillas every code but the rotated surface has `d_circ = 2`
(hook errors of the weight-4 and weight-6 checks), so the benchmark ranks
codes mostly by the count and mass of their weight-2 failures, which the
proxy computes exactly.

Random codes (`random.json`): 32 distinct `[[9,1,3]]` codes with no
weight-1 stabilizer from random 150-gate H, S, CNOT circuits on 9 qubits
(512 circuits drawn; none CSS), `R = 3`, all-to-all, proxy against bposd
(5000 shots grown to 100 failures or 50k shots). Spearman proxy against
bposd: 0.79 (ion), 0.90 (board), 0.78 (heron). All 32 have `d_circ = 2`
in both bases with bare ancillas, so `d_circ` alone does not rank them
(correlation undefined); the ranking is entirely in the weight-2 failure
mass. To see where the disagreement comes from
(`random_check.json`, the same codes under the sampled lookup tier and
under bposd with a second sampling seed, 100 failures per basis): proxy
against sampled 0.90 / 0.81 / 0.70 (ion / board / heron), sampled against
bposd 0.67 / 0.83 / 0.70, and bposd seed against seed 0.53 / 0.95 / 0.88.
On the ion profile a single 100-failure bposd sample ranks the codes less
consistently with a second sample than the proxy does with either, so the
0.79 is a shot-noise ceiling, not a proxy error; on board and heron the
proxy against the two-seed bposd mean is 0.90 and 0.74. Two caveats. The
random codes are dense (generator weights 3 to 9), so at the profile
scales their expected fault counts are 0.5 (ion), 1.8 to 2.5 (board), 0.9
to 1.2 (heron): the board and heron random sets sit at the edge of or
outside the leading-order regime, and the proxy's absolute values are
1.2 times the sampled tier on ion, equal on board, and 0.8 on heron
(medians). The per-round rates within a profile span a factor of 2.5, so
rank correlations of about 0.9 are what estimates with 10 percent
relative error allow. At 400 failures per basis on the ion profile
(`random_check_ion400.json`) the picture sharpens: proxy against sampled
0.93, sampled against bposd 0.70, bposd seed against seed 0.58 (the
original 100-failure sample against a 400-failure one; their per-code
ratios range from 0.77 to 1.35 while the 32 rates span 0.010 to 0.020),
proxy against the two-seed bposd mean 0.69. The two maximum-likelihood
estimates agree with each other on the ranking and bposd agrees with
neither better than it agrees with itself, so the proxy meets the
"ranks the codes the same way as tier (c)" test up to the resolution
tier (c) has on this set. The lookup decoder fell back to the
one-mechanism completion on 10 percent of the distinct syndromes; its
absolute values are 1.19 times the proxy's (median).

## Bias sensitivity

Rotated surface against XZZX at `n = 9`, `R = 3`, all-to-all, bare
ancillas (`bias.json`). The synthetic profile has every gate, reset, and
measurement at 2.5e-4 and a slot idle channel of 2.5e-4 per layer with
bias `eta = pZ / (pX + pY)`. `ratio = p_round(plain) / p_round(xzzx)`,
above 1 means the objective prefers the XZZX deformation:

| eta | sequential: proxy | sequential: bposd | parallel: proxy | parallel: bposd |
|---|---|---|---|---|
| 0.5 (depolarizing) | 1.000 | 1.08 | 1.000 | 1.05 |
| 2 | 1.21 | 1.23 | 1.21 | 1.18 |
| 5 | 1.55 | 1.25 | 1.53 | 1.59 |
| 10 | 1.80 | 1.89 | 1.75 | 1.84 |
| 30 | 2.06 | 1.87 | 1.96 | 2.32 |
| 100 | 2.20 | 2.28 | 2.06 | 2.95 |
| inf (pure Z) | 2.27 | 2.19 | 2.10 | 2.35 |

At `eta = 0.5` the two codes tie to nine digits in the proxy (the
deformation is a relabeling of a symmetric channel), which is the check
that the preference is the bias and not the circuit. Absolute values under
sequential timing: plain 0.00130 to 0.00172, XZZX 0.00130 to 0.00076
(proxy), 200k bposd shots per basis. Under the Forte set at 0.25 (duration
idle with `eta = 10`, `E = 0.20`) the proxy prefers XZZX by 1.22 and the
near-ML lookup decoder by 1.21 (0.000567 against 0.000467, 220k to 300k
shots per basis); bposd does not (0.78: 0.000698 against 0.000894), so
the pinned decoder fails to exploit the XZZX structure there. Under the
chain set at 0.25 the codes tie (0.99 proxy, 0.98 bposd): its slot idle
is 1e-5 per gate slot at this scale while its readout flips 2.25e-3 per
ancilla dominate, consistent with the ion-search finding that the chain
set makes measurement error dominant. The encoder objective could not
separate the two codes under any profile (exact tie,
`experiments/noise_aware/REPORT.md` item 4).

## Lattice layout

`lattice.json`, `R = 3`, proxy, Z-basis circuit statistics over three
rounds:

| code | layout | 2q gates | routed extra | layers | batches | `d_circ` Z/X (board) | board | heron | ion |
|---|---|---|---|---|---|---|---|---|---|
| rotated surface | all-to-all or `square:3x3` | 72 | 0 | 66 | 2 | 3/3 | 0.0077 | 0.0022 | 0.00077 |
| XZZX | all-to-all or `square:3x3` | 72 | 0 | 66 | 2 | 3/3 | 0.0077 | 0.0019 | 0.00078 |
| non-CSS [[9,1,3]] | all-to-all | 81 | 0 | 84 | 3 | 2/2 | 0.018 | 0.0063 | 0.0024 |
| non-CSS [[9,1,3]] | `square:3x3` | 270 | 63 | 315 | 7 | 2/2 | 0.26 (`E` 2.8) | 0.040 | 0.028 |
| Shor | all-to-all | 72 | 0 | 60 | 2 | 3/3 | 0.0061 | 0.0025 | 0.00077 |
| Shor | `square:3x3` | 207 | 45 | 204 | 4 | 2/2 | 0.072 (`E` 1.8) | 0.013 | 0.010 |

The surface code and its deformation embed without routing (the greedy
site assignment finds the standard layout); the non-local codes pay in
gates, depth, and, through the ladders' hook errors, in `d_circ`. The
routed board rows are outside the proxy's regime and are listed for the
gate counts.

## PPO regression

`ppo_3x3_ion_seed0.json`: `CodeDiscoveryEnv(n=9, k=1, square:3x3, H and
CNOT, 30 gates, batch 256)`, `MemoryReward(MemoryObjective(chain, scale
0.25, rounds=3, tier=proxy), d*=3, weight 0.1, p_ref 1e-2)`, `PPOConfig`
defaults (flat MLP, 157,730 parameters), dense reward, no early stop,
seed 0, one thread: 401k env steps, 13,313 episodes, 62 s. Distance 3
reached at 83,200 env steps (the noise-aware runs on the same setting
reached it at 83k to 161k); 92 successful episodes, all distinct codes,
92 memory evaluations at 0.47 s each; final success rate 0.07 (the run is
far from converged at this budget; the noise-aware runs needed 0.5M to
1M steps to approach 1). The best code is a non-CSS `[[9,1,3]]` with
`p_round` 0.0016 under the objective, against 0.00077 for the rotated
surface code and 0.0024 for the noise-aware ion seed-0 code. The
test-suite regression (`test_ppo_smoke_regression_3x3_ion_memory`, `R =
2`, 300k steps or 240 s, stop at first success) passes. The
memory-objective term does not change the exploration problem: the
distance term drives the search to `d = 3`, the memory term only reorders
target-distance codes, as in the noise-aware experiment.

## What a full run needs

A real noise-aware PPO run with the memory objective at the 3x3 scale is
the noise-aware protocol (8M env steps, 5 seeds, 3 profiles) with the
memory term: 23k distinct target-distance codes per run at 0.47 s is
about 3 process-hours per run on top of the 10 minutes of PPO, 45 runs
about 135 process-hours, or 6 hours on 24 cores with the memory term
evaluated in a process pool (the term is per code, embarrassingly
parallel, and stim's DEM derivation is single-threaded). The 4x4 setting
(`d = 4`, `R = 4`, about 2000 mechanisms per basis) costs 1.7 to 2.3 s per
code in the proxy tier; with the equivariant policy's 350k to 400k steps to
the first `d = 4` state and the `gpu_scale` success rates, a 3M-step run
evaluates a few thousand codes, about 2 process-hours per run. The PPO
part runs on the GPU as in `docs/gpu.md`; the memory term stays on the
CPU, so a pod with one A40 and 16 to 32 cores covers the 4x4 matrix (9
cells) in about 4 hours. Two things to decide before that run: the
scale (0.25 for the chain set keeps `E` below one at `n = 16`; the [[16,4,4]]
row has `E = 0.85`), and whether to train with flags (2.6 to 4.4 times
lower `p_round` under the ion preset for every bare-`d_circ`-2 code, at
1.3 to 1.5 times the gates) or without, since the two objectives may
prefer different codes.

## Tests

`tests/test_memory_objective.py`:

- noiseless circuits of the [[5,1,3]], Steane, rotated surface, and the
  non-CSS 3x3 code have deterministic detectors in both bases, `m (R + 1)`
  detectors, `k` observables, `n + m` qubits;
- batches follow commutation (Steane X block then Z block, [[5,1,3]] four
  singleton batches, the non-CSS code's batches share no qubit with
  different letters);
- the hook-aware order lifts the Shor code's Z-basis `d_circ` from 1 to 3
  and the rotated surface code's X-basis `d_circ` from 2 to 3;
- one flag per generator lifts the [[5,1,3]] `d_circ` from 2 to 3 and adds
  one detector per flagged generator per round;
- the square layout routes nothing for the rotated surface code, routes the
  Shor code with deterministic detectors, and the routed circuit passes
  the noise module's connectivity check; flags on the lattice are refused;
- the proxy is within 10 to 20 percent of the exact maximum-likelihood
  failure probability on the [[5,1,3]] one-round DEM, the exclusive
  weight-2 and weight-3 sums increase toward it, and the lookup decoder's
  sampled rate agrees with it;
- the objective prefers XZZX over the plain surface code under a Z-biased
  idle channel (`eta = 30` and pure Z) and ties them under a depolarizing
  idle;
- `MemoryReward` through `CodeDiscoveryEnv.rollout`: the [[5,1,3]] encoder
  succeeds with the memory term applied once, the idle element does not,
  and the environment adopts the reward's `success_fn`;
- PPO smoke regression: 3x3 lattice, H and CNOT, chain profile at 0.25,
  `R = 2`, batch 256, 300k env steps or 240 s, must reach a verified
  distance-3 code (skipped without torch).
