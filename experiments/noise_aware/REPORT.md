# Noise-aware code discovery with PPO under three noise profiles

Question: does the code an RL agent discovers depend on the noise model it is trained against, and in the direction the model's structure predicts (the experiment the proposal cites Olle et al., arXiv:2311.04750, for)? Module: `src/autoqec_sim/experiments/noise_aware.py`; one cell reproduces with `uv run --extra rl python -m autoqec_sim.experiments.noise_aware run --setting S --profile P --seed K`.

## Verdict

Status: complete. 45 PPO runs (3 settings x 3 profiles x 5 seeds, padding penalty on), three baseline sets, 31 process-hours in about 3.5 hours of wall time on 6 threads. Every number below is copied from the tables of this report, which regenerate from `runs/*.json` and `baselines_*.json` with `python -m autoqec_sim.experiments.noise_aware report`.

Yes, the code depends on the noise profile, and the dependence is large where the profiles differ in structure, but it is not the dependence the memory-noise picture predicts. What the agent adapts is the encoding circuit's fault propagation, not the code's X/Z distance structure.

1. Cross evaluation (question 2). At n = 7 the board-trained codes have 3.4x less undetected mass on the board profile than the ion- or heron-trained codes (0.00069 against 0.0023, 25 of 25 seed pairs, exact permutation p = 0.008), and the ion-trained codes have 1.6x less on the ion profile than the board-trained ones (0.00108 against 0.00172, 25/25, p = 0.008). The ion and heron profiles are interchangeable at n = 7: ion- and heron-trained codes are within 2 to 9 percent of each other on either profile (p = 0.87 and 0.16), and several seeds of the two converge to the same code (0.002333/0.001032/0.001412 on board/ion/heron). On the 3x3 lattice the board-versus-ion effect repeats (3.3x on board, 25/25, p = 0.008; 1.3x on ion, 22/25, p = 0.048), heron-trained codes beat board- and ion-trained ones on heron by 1.3x and 1.1x (25/25, p = 0.008; 16/25, p = 0.40), and board-versus-heron on board is 2.6x on the mean but only 19/25 pairs (p = 0.14) because one heron seed found a 14-gate code with 0.0004 on board, below every board-trained code (0.00067 to 0.00093). Five seeds resolve ratios of about 1.3 and above; ratios of 1.0 to 1.1 are not distinguishable from no effect.

2. Mechanism (why the matrix is asymmetric). The board profile has a depolarizing idle channel of the same rate as the gate channel and no single-qubit-gate channel; the ion profile's idle is 95 percent Z at a comparable total mass; the Heron profile at equal total mass is gate dominated with a small, mildly Z-biased idle. The mechanism-breakdown tables show the board-trained codes reach zero undetected idle mass on board (idle faults stay weight 1 in their circuits and are caught by the distance-3 code) and lose on ion through the two-qubit-gate channel (1.4e-3 against 0.75e-3 for the ion-trained codes); the ion-trained codes lose on board almost entirely through idle X and Y faults (1.3e-3 to 1.5e-3 of their 2.4e-3 to 2.6e-3), mechanisms the ion profile hardly emits and the agent never had to detect. The adaptation is to which fault mechanisms exist and where in the circuit they land.

3. Structural bias (question 1). No. d_Z > d_X occurs in 0/5 seeds under every profile at n = 7 and in 1/5 seeds under every profile on the 3x3 lattice (d_X > d_Z equally often), none of the 30 codes is CSS, and none is local-Clifford equivalent to a CSS code, so there is no Hadamard-frame pattern to read off. The discovered codes are degenerate non-CSS [[7,1,3]] and [[9,1,3]] codes with w* = 4 and several weight-2 stabilizer elements, at 10 to 30 gates; the pure-Z to pure-X stabilizer element counts range from 1:1 to 31:3 with no profile ordering (3x3: board 3:3, 1:1, 15:3, 15:7, 7:1; ion 15:3, 1:3, 31:3, 7:3, 7:7; heron 7:3, 31:3, 15:1, 3:3, 3:3). A Z-biased idle channel in the encoding circuit does not translate into a Z-tailored code because an idle Z fault propagates through the later CNOTs and Hadamards of the same circuit and reaches the output as a mixed error; the reward sees the propagated error, not the inserted one.

4. Known codes (question 3). The discovered codes beat every known code at the same (n, k) under every profile once the known code gets its best polished encoder. At n = 7 against Steane: 0.00069 [0.00053, 0.00093] against 0.0008 on board (4/5 seeds), 0.00108 against 0.00233 on ion (5/5), 0.00135 against 0.00218 on heron (5/5). On the 3x3 lattice against the rotated surface code (the best known code under every profile; XZZX and the Hadamard-deformed surface code tie it exactly, Shor and transposed Shor are 1.1x to 1.5x worse): 0.00075 [0.00067, 0.00107] against 0.00087 on board (4/5), 0.00136 against 0.00316 on ion (5/5), 0.00164 against 0.00246 on heron (5/5). Two caveats. The discovered codes are degenerate with weight-2 stabilizers, which this objective rewards and a memory experiment would not; and the known-code encoders come from a routed standard-form construction plus a local search, so part of the gap is encoder quality (the polishing lowered their mass by 1.2x to 4x). The exact tie between the surface code and its XZZX and outer-column deformations is structural: a Hadamard on a data qubit at the end of the encoder only relabels single faults of the last layer, all of weight 1 and detected, so the circuit-level single-fault objective cannot see a local-Clifford deformation of the code. That is a second reason no Hadamard-frame pattern appears in the discovered codes.

5. What did not fit the budget. The 4x4 setting (n = 16, target d = 4) did not produce a single distance-4 state in any of its 15 runs (16M env steps, 12 to 17 minutes each): the largest terminal distance was 3 in every run, the mean step reward stayed at -1.0 (a d = 1 to 2 state with untouched qubits), and, since the profile only enters the reward through a target-distance state, the three profiles' runs at the same seed are the same trajectory. Two variants were tried before settling on the reported one: the plain level distance reward without the padding penalty (`pilot/4x4_board_level_reward_seed1.json`, 32.5M steps in 25 minutes, largest terminal d = 3, no d = 4), and the reported fractional distance term with the padding penalty. The MLP policy over the 1024-bit tableau with 64 actions and 50-gate episodes is the limitation, not the reward evaluation (16M steps ran in under 18 minutes). The known-code baselines at n = 16 are in the 4x4 table for reference (rotated surface 0.0019 on board, 0.0059 on ion, 0.0051 on heron with 64- to 66-gate polished encoders; Shor and transposed Shor 3x to 2x worse), so a future agent that reaches d = 4 has numbers to beat.

6. Two objective-level findings. Without the padding penalty of the rediscovery regression, the circuit-level reward makes the 3x3 agent embed a [[7,1,3]] code and leave two qubits idle (2 of 3 profiles at seed 0; `pilot/no_padding_penalty/`), because an idle qubit adds only trivially detected faults; every reported cell uses the penalty (the n7 cells of that first pass were not padded and gave the same cross matrix to within seed noise). A tracker bug in an earlier 4x4 pass (the distance label of truncated episodes was read after the batch reset when every episode truncated in the same step) mislabelled terminal states as d = 1; the runs were redone with the fix (`pilot/tracker_bug/`) and success detection, which decides every number in items 1 to 4, was never affected.

## Reward and protocol

State reward of a tableau `s` reached by the gate list `c`: `r(s) = min(d(s), d*)/d* - 1 - q_1(s) + [d(s) >= d*, q_1(s) = 0] (-2 m(c)/M_0 - 0.25 max(0, w*(s) - 4))`. `d(s)` is the exact distance up to `d* - 1` (`reward.distance_reward`, shifted by -1); `q_1(s)` is the fraction of weight-1 Paulis in the stabilizer group (`reward.low_weight_stabilizer_penalty`, the padding penalty of the rediscovery regression: a first pass without it, kept under `pilot/no_padding_penalty/`, had the 3x3 agent embed a [[7,1,3]] code and leave two qubits idle in 2 of 3 profiles at seed 0, since an idle qubit adds only trivially detected faults; the 7-qubit cells of that pass were not padded); `m(c)` is the circuit-level undetected fault mass of the encoding circuit under the training profile (`circuit_noise.undetected_fault_mass`, `normalize=False`, absolute probability, every fault mechanism of the noisy circuit propagated to the output and tested with the Knill-Laflamme kernel); `M_0` is the setting's common total fault mass (below); `w*` is the exact minimum bottleneck generator weight (`ldpc.ldpc_penalty`, `w_max = 4`, scale 0.25). Noise and LDPC terms are evaluated only on states that reach the target distance with no weight-1 stabilizer, and such a state ends the episode. In the 4x4 setting the distance term is refined below the target to `(d - 1 + f)/d* - 1` with `f` the detected fraction of the weight-`d` Paulis (the level reward alone left PPO at d = 1 to 2 for 32.5M steps, 25 minutes, in the pilot). Dense reward (state reward after every gate), PPO defaults of `agents.ppo.PPOConfig` (256 parallel episodes, 16-step rollouts, 4 epochs, lr 3e-4, gamma 0.98, entropy 0.01), undo mask on.

### Profiles and calibration

Each profile is scaled so that the mean total fault mass over 64 random circuits of the setting's maximum length equals the board preset's at scale 1 (`M_0`). The Heron preset's heavy-hex graph is replaced by the setting's graph; the settings fix connectivity, so `heron` differs from `board` in rate structure (p1 = 2e-4 on single-qubit gates, T1/T2 idle with eta = 1.8, gate bias 1/2), not in connectivity.

| setting | profile | preset | scale | total mass at scale 1 (mean over 64 random circuits) | M_0 |
|---|---|---|---|---|---|
| 3x3 | board | board-rfc0001 [parallel] | 1.000 | 0.0867 | 0.0867 |
| 3x3 | ion | forte-2308.05071 [sequential] | 0.508 | 0.1709 | 0.0867 |
| 3x3 | heron | heron-r2 [parallel] | 0.810 | 0.1071 | 0.0867 |
| 4x4 | board | board-rfc0001 [parallel] | 1.000 | 0.1653 | 0.1653 |
| 4x4 | ion | forte-2308.05071 [sequential] | 0.391 | 0.4224 | 0.1653 |
| 4x4 | heron | heron-r2 [parallel] | 0.868 | 0.1903 | 0.1653 |
| n7 | board | board-rfc0001 [parallel] | 1.000 | 0.0554 | 0.0554 |
| n7 | ion | forte-2308.05071 [sequential] | 0.546 | 0.1013 | 0.0554 |
| n7 | heron | heron-r2 [parallel] | 0.777 | 0.0713 | 0.0554 |

## Runs

| setting | profile | seeds | budget (env steps) | wall time per seed, s | seeds reaching d* | first success (env steps) | final success rate | distinct target codes per seed | noise evaluations per seed |
|---|---|---|---|---|---|---|---|---|---|
| 3x3 | board | 5 | 8,000,000 | 593 [443, 810] | 5/5 | 1.2e+05 [8.32e+04, 1.61e+05] | 0.969 [0.881, 1] | 3.19e+03 [948, 3.92e+03] | 2.31e+04 [1.68e+04, 2.69e+04] |
| 3x3 | ion | 5 | 8,000,000 | 639 [570, 706] | 5/5 | 1.2e+05 [8.32e+04, 1.61e+05] | 0.8 [0, 1] | 2.84e+03 [1.53e+03, 3.31e+03] | 2.28e+04 [1.58e+04, 2.91e+04] |
| 3x3 | heron | 5 | 8,000,000 | 599 [391, 843] | 5/5 | 1.2e+05 [8.32e+04, 1.61e+05] | 0.999 [0.997, 1] | 2.77e+03 [1.2e+03, 4.44e+03] | 2.36e+04 [1.63e+04, 2.97e+04] |
| 4x4 | board | 5 | 16,000,000 | 878 [720, 1.04e+03] | 0/5 | n/a | 0 [0, 0] | 0 [0, 0] | 319 [17, 512] |
| 4x4 | ion | 5 | 16,000,000 | 877 [713, 1.04e+03] | 0/5 | n/a | 0 [0, 0] | 0 [0, 0] | 319 [17, 512] |
| 4x4 | heron | 5 | 16,000,000 | 878 [720, 1.04e+03] | 0/5 | n/a | 0 [0, 0] | 0 [0, 0] | 319 [17, 512] |
| n7 | board | 5 | 6,000,000 | 242 [224, 267] | 5/5 | 7.98e+04 [3.58e+04, 1.08e+05] | 1 [0.998, 1] | 950 [565, 1.75e+03] | 5.59e+03 [4e+03, 8.59e+03] |
| n7 | ion | 5 | 6,000,000 | 255 [237, 284] | 5/5 | 7.98e+04 [3.58e+04, 1.08e+05] | 0.918 [0.591, 1] | 730 [516, 1.1e+03] | 4.95e+03 [4.5e+03, 5.45e+03] |
| n7 | heron | 5 | 6,000,000 | 243 [223, 273] | 5/5 | 7.98e+04 [3.58e+04, 1.08e+05] | 0.987 [0.941, 1] | 612 [498, 689] | 4.75e+03 [4.02e+03, 5.29e+03] |

No run hit the 25-minute cap; every run used its full step budget.

Budgets: 6M env steps for n7 and 8M for 3x3 (about 5 and 10 minutes per seed at 2 threads; PPO reaches a success rate near 1 within the first 0.5M to 1M steps in both, so the rest of the budget explores distinct target-distance codes), 16M for 4x4 with the 25-minute cap. A 4x4 pilot with the plain level distance reward (`pilot/4x4_board_level_reward_seed1.json`, board profile, 32.5M steps in 25 minutes) never produced a distance-4 state and stayed at mean step reward -0.75 to -0.66 (d = 1 to 2), which is why the 4x4 cells use the fractional distance term.

![](figures/curves_3x3.png)

![](figures/curves_4x4.png)

![](figures/curves_n7.png)

Training curves per setting, median over seeds (line) with the min-to-max band across seeds, color by profile: (a) mean step reward over the rollout, trailing mean over 25 updates (dense reward, so it rises from -(d* - 1)/d* as the policy reaches higher-distance states earlier; the band width after convergence reflects seeds whose success rate collapses and recovers, which the raw per-update curves show as spikes to 0), (b) fraction of finished episodes that reached the target distance, same smoothing, (c) minus the best terminal reward seen so far, that is the noise plus LDPC penalty of the best target-distance code, log axis. In a setting where no run reached the target the profile never enters the reward and the three profiles' curves coincide (only the last drawn is visible). The raw per-update values are in the CSV files under `runs/`.

## Setting 3x3: n = 9, k = 1, target d = 3, square:3x3, gates H, CNOT

### Cross-evaluation matrix

Rows: training profile (the best code of each seed, its own encoding circuit). Columns: evaluation profile at the calibrated scale. Entries: undetected fault mass, mean over seeds with [min, max]; per seed values in the code list below. Smaller is better.

| trained on \ evaluated on | board | ion | heron | d reached (seeds) | gates (mean) |
|---|---|---|---|---|---|
| board | 0.000747 [0.000667, 0.00107] | 0.00179 [0.00148, 0.00234] | 0.00216 [0.00189, 0.00313] | 3, 3, 3, 3, 3 | 23.8 |
| ion | 0.00249 [0.002, 0.00333] | 0.00136 [0.00121, 0.00183] | 0.00179 [0.00147, 0.00226] | 3, 3, 3, 3, 3 | 20.8 |
| heron | 0.00193 [0.0004, 0.00327] | 0.00178 [0.00152, 0.00216] | 0.00164 [0.00151, 0.00182] | 3, 3, 3, 3, 3 | 22.6 |

Per seed (undetected mass under board / ion / heron):

| trained on | seed | d | d_X | d_Z | w* | CSS | frame | board | ion | heron | gates | CNOTs | depth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| board | 0 | 3 | 3 | 3 | 4 | no | none | 0.0006667 | 0.001612 | 0.00189 | 23 | 17 | 9 |
| board | 1 | 3 | 4 | 5 | 4 | no | none | 0.0006667 | 0.001612 | 0.00189 | 21 | 12 | 7 |
| board | 2 | 3 | 3 | 3 | 4 | no | none | 0.0006667 | 0.001914 | 0.001944 | 27 | 21 | 13 |
| board | 3 | 3 | 3 | 3 | 4 | no | none | 0.0006667 | 0.00148 | 0.001944 | 21 | 17 | 11 |
| board | 4 | 3 | 8 | 3 | 4 | no | none | 0.001067 | 0.002344 | 0.003132 | 27 | 21 | 12 |
| ion | 0 | 3 | 3 | 3 | 4 | no | none | 0.002333 | 0.001215 | 0.001473 | 15 | 11 | 7 |
| ion | 1 | 3 | 3 | 4 | 4 | no | none | 0.002333 | 0.001244 | 0.001527 | 28 | 21 | 14 |
| ion | 2 | 3 | 3 | 3 | 4 | no | none | 0.003333 | 0.001268 | 0.001832 | 15 | 10 | 9 |
| ion | 3 | 3 | 5 | 3 | 4 | no | none | 0.002467 | 0.001242 | 0.001851 | 18 | 13 | 11 |
| ion | 4 | 3 | 3 | 3 | 4 | no | none | 0.002 | 0.001834 | 0.002257 | 28 | 20 | 14 |
| heron | 0 | 3 | 3 | 3 | 4 | no | none | 0.0009333 | 0.001632 | 0.00182 | 29 | 24 | 12 |
| heron | 1 | 3 | 3 | 3 | 4 | no | none | 0.0004 | 0.002031 | 0.001512 | 14 | 9 | 7 |
| heron | 2 | 3 | 6 | 3 | 4 | no | none | 0.002667 | 0.001517 | 0.001537 | 21 | 11 | 9 |
| heron | 3 | 3 | 3 | 3 | 4 | no | none | 0.003267 | 0.002157 | 0.001668 | 30 | 24 | 13 |
| heron | 4 | 3 | 3 | 4 | 4 | no | none | 0.0024 | 0.001574 | 0.001662 | 19 | 13 | 9 |

### Mechanism breakdown of the undetected mass

Undetected fault mass of the best codes split by the mechanism that caused it (mean over seeds): two-qubit gate channel, single-qubit gate channel, idle channel with a Z fault, idle channel with an X or Y fault. The board profile has no single-qubit gate channel and a depolarizing idle; the ion profile's idle is 95 percent Z.

| trained on | evaluated on | two-qubit gate | single-qubit gate | idle Z | idle X or Y | total undetected | total fault mass |
|---|---|---|---|---|---|---|---|
| board | board | 7.47e-04 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 7.47e-04 | 0.070 |
| board | ion | 1.52e-03 | 2.71e-05 | 2.33e-04 | 1.68e-05 | 1.79e-03 | 0.068 |
| board | heron | 2.12e-03 | 4.32e-05 | 0.00e+00 | 0.00e+00 | 2.16e-03 | 0.070 |
| ion | board | 4.27e-04 | 0.00e+00 | 6.00e-04 | 1.47e-03 | 2.49e-03 | 0.078 |
| ion | ion | 8.66e-04 | 2.03e-05 | 4.10e-04 | 6.41e-05 | 1.36e-03 | 0.058 |
| ion | heron | 1.21e-03 | 3.24e-05 | 3.16e-04 | 2.30e-04 | 1.79e-03 | 0.065 |
| heron | board | 4.00e-04 | 0.00e+00 | 4.00e-04 | 1.13e-03 | 1.93e-03 | 0.067 |
| heron | ion | 8.12e-04 | 5.41e-05 | 8.16e-04 | 1.00e-04 | 1.78e-03 | 0.063 |
| heron | heron | 1.13e-03 | 8.64e-05 | 2.27e-04 | 1.92e-04 | 1.64e-03 | 0.064 |

Most visited target-distance code of the last 20 updates (the converged policy's code), undetected mass under board / ion / heron:

| trained on | seed | visits (of successes in the window) | d | d_X | d_Z | w* | CSS | board | ion | heron | gates | same as best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| board | 0 | 3659 (4213) | 3 | 4 | 3 | 4 | no | 0.001667 | 0.001395 | 0.002059 | 16 | no |
| board | 1 | 4562 (4871) | 3 | 4 | 3 | 4 | no | 0.0006667 | 0.001612 | 0.00189 | 14 | no |
| board | 2 | 4101 (5053) | 3 | 4 | 3 | 5 | no | 0.0006667 | 0.001457 | 0.001944 | 19 | no |
| board | 3 | 5095 (5120) | 3 | 3 | 3 | 4 | no | 0.0007333 | 0.001593 | 0.002133 | 18 | no |
| board | 4 | 7443 (7444) | 3 | 5 | 3 | 6 | no | 0.0007333 | 0.001593 | 0.002133 | 11 | no |
| ion | 0 | 5487 (5686) | 3 | 3 | 3 | 4 | no | 0.002333 | 0.001215 | 0.001473 | 15 | yes |
| ion | 1 | 4232 (4257) | 3 | 5 | 3 | 5 | no | 0.001733 | 0.001554 | 0.002248 | 17 | no |
| ion | 2 | 6213 (6253) | 3 | 3 | 3 | 4 | no | 0.003333 | 0.001268 | 0.001832 | 15 | yes |
| ion | 3 | 3111 (4518) | 3 | 7 | 3 | 4 | no | 0.001 | 0.001697 | 0.002174 | 18 | no |
| ion | 4 | 6800 (6816) | 3 | 5 | 3 | 6 | no | 0.001533 | 0.00205 | 0.002849 | 13 | no |
| heron | 0 | 7166 (7174) | 3 | 5 | 3 | 6 | no | 0.0004 | 0.001574 | 0.001512 | 13 | no |
| heron | 1 | 6709 (6724) | 3 | 3 | 3 | 4 | no | 0.0004 | 0.002031 | 0.001512 | 14 | yes |
| heron | 2 | 5629 (6123) | 3 | 3 | 3 | 4 | no | 0.0006667 | 0.001914 | 0.001944 | 15 | no |
| heron | 3 | 4931 (5540) | 3 | 3 | 3 | 4 | no | 0.0006667 | 0.001469 | 0.001944 | 13 | no |
| heron | 4 | 5360 (6013) | 3 | 3 | 3 | 4 | no | 0.0006667 | 0.001469 | 0.001944 | 14 | no |

### Known codes at the same (n, k)

Encoders: the Gottesman standard-form encoder of the CSS parent code followed by H on the frame, routed onto the setting's graph (nearest-neighbour CNOT chains), then polished per profile by a greedy local search that keeps the stabilizer group fixed and lowers the undetected mass (single-gate deletions, adjacent swaps, single-gate replacements, 60.0 s per profile); on all-to-all also stim's elimination encoder. The entry is the polished encoder's undetected mass under that profile (a different circuit per column). The RL codes have encoders co-optimized with the code by the agent, so this comparison is fair only to the extent the polishing reaches comparable circuits; the unpolished mass and gate counts are given to show how much the polishing did.

| code | d | d_X | d_Z | w* | CSS | board (gates, unpolished mass) | ion (gates, unpolished mass) | heron (gates, unpolished mass) |
|---|---|---|---|---|---|---|---|---|
| rotated_surface (rotated surface [[9,1,3]]) | 3 | 3 | 3 | 4 | yes | 0.0008667 (19, 0.001333) | 0.00316 (19, 0.003659) | 0.002457 (19, 0.003065) |
| xzzx (XZZX surface [[9,1,3]]: H on the checkerboard [1, 3, 5, 7]) | 3 | 3 | 3 | 4 | no | 0.0008667 (23, 0.001333) | 0.00316 (23, 0.003687) | 0.002457 (23, 0.003065) |
| surface_h_outer_cols (rotated surface with H on the outer columns A=[0, 2, 3, 5, 6, 8] (the best Forte deformation of ion-search/deform at d = 3)) | 3 | 3 | 3 | 4 | no | 0.0008667 (25, 0.001333) | 0.003216 (25, 0.003771) | 0.002457 (25, 0.003065) |
| shor (Shor [[9,1,3]]: ZZ along rows, X on row pairs (w* = 6)) | 3 | 3 | 3 | 6 | yes | 0.001333 (25, 0.003267) | 0.003803 (23, 0.005258) | 0.002768 (23, 0.004212) |
| transposed_shor (transposed Shor [[9,1,3]]: XX along rows, Z on row pairs (ion-search shor:3x3T at d = 3)) | 3 | 3 | 3 | 6 | yes | 0.001333 (34, 0.003267) | 0.003575 (32, 0.005286) | 0.002768 (32, 0.004212) |

### Best code per seed

**3x3 / board / seed 0**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (1), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 0, 0, 1, 2, 0, 0, 0, 0]), pure-Z 3 (weights [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]). Circuit: 23 gates (17 CNOT, 6 H, 0 S), depth 9. Undetected mass board/ion/heron = 0.0006667/0.001612/0.00189. Reward terms: distance +0.000, padding +0.000, noise -0.0154, LDPC -0.000.

Stabilizers (sparsest basis): `YYIIIXZII`, `IYIYXIIXI`, `IIYYIIIIZ`, `ZZXZIIIII`, `IIXIXXIIX`, `ZIIIIIXXI`, `IZIIZZIII`, `IIIIZIZZZ`  
Weight enumerator of S: [1, 0, 0, 3, 6, 27, 64, 81, 57, 17]; of N(S): [1, 0, 0, 10, 42, 114, 232, 318, 237, 70]  
Gates: `H(4) H(1) CNOT(0,3) H(7) CNOT(1,0) H(0) CNOT(4,3) CNOT(3,6) CNOT(4,5) CNOT(1,2) CNOT(0,3) CNOT(7,8) H(6) CNOT(3,0) CNOT(4,1) H(1) CNOT(8,5) CNOT(5,2) CNOT(1,4) CNOT(7,4) CNOT(1,2) CNOT(6,7) CNOT(4,7)`

**3x3 / board / seed 1**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 4 (1 min-weight), d_Z = 5 (1), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), pure-Z 1 (weights [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]). Circuit: 21 gates (12 CNOT, 9 H, 0 S), depth 7. Undetected mass board/ion/heron = 0.0006667/0.001612/0.00189. Reward terms: distance +0.000, padding +0.000, noise -0.0154, LDPC -0.000.

Stabilizers (sparsest basis): `YXYIIIIII`, `IXIXXXIII`, `YIIZYIZII`, `ZIZYYIIII`, `IYXIZIYII`, `IIIIIIIXZ`, `IIIIIIIZX`, `IZZIIZIII`  
Weight enumerator of S: [1, 0, 3, 2, 9, 30, 49, 78, 66, 18]; of N(S): [1, 0, 3, 17, 45, 129, 217, 267, 246, 99]  
Gates: `CNOT(0,3) H(7) H(6) CNOT(6,3) H(1) CNOT(1,0) CNOT(3,4) CNOT(1,2) CNOT(4,1) H(2) H(0) H(4) CNOT(7,8) H(7) CNOT(2,5) CNOT(0,3) CNOT(1,0) H(1) CNOT(4,3) H(0) CNOT(1,2)`

**3x3 / board / seed 2**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (3), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]), pure-Z 15 (weights [0, 0, 1, 3, 1, 4, 5, 1, 0, 0]). Circuit: 27 gates (21 CNOT, 6 H, 0 S), depth 13. Undetected mass board/ion/heron = 0.0006667/0.001914/0.001944. Reward terms: distance +0.000, padding +0.000, noise -0.0154, LDPC -0.000.

Stabilizers (sparsest basis): `XZIXIIIIX`, `IYIIYIXIZ`, `IIXXIIXII`, `IIIIXYIYX`, `ZIIIIZIIZ`, `IZZIIIZII`, `IXYYXIIII`, `IIIIIZIZI`  
Weight enumerator of S: [1, 0, 1, 4, 9, 22, 55, 88, 62, 14]; of N(S): [1, 0, 1, 17, 39, 109, 235, 307, 236, 79]  
Gates: `CNOT(0,3) H(1) CNOT(1,0) H(3) CNOT(3,6) H(4) CNOT(4,3) H(3) CNOT(4,3) H(3) CNOT(8,5) H(8) CNOT(1,2) CNOT(4,1) CNOT(7,6) CNOT(0,3) CNOT(0,1) CNOT(8,7) CNOT(3,0) CNOT(6,7) CNOT(4,5) CNOT(3,6) CNOT(3,0) CNOT(5,8) CNOT(3,0) CNOT(7,4) CNOT(4,5)`

**3x3 / board / seed 3**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (5), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 7 (weights [0, 0, 0, 1, 2, 3, 1, 0, 0, 0]), pure-Z 15 (weights [0, 0, 1, 0, 8, 0, 5, 0, 1, 0]). Circuit: 21 gates (17 CNOT, 4 H, 0 S), depth 11. Undetected mass board/ion/heron = 0.0006667/0.00148/0.001944. Reward terms: distance +0.000, padding +0.000, noise -0.0154, LDPC -0.000.

Stabilizers (sparsest basis): `XXIXIXIII`, `IXIIXIIXI`, `IIXXIIIXX`, `YZIIYIXII`, `ZIZZIIZII`, `IZIIZZZII`, `IIZIIIIIZ`, `ZIIZZIIZI`  
Weight enumerator of S: [1, 0, 1, 1, 13, 27, 47, 87, 66, 13]; of N(S): [1, 0, 1, 15, 31, 123, 251, 285, 228, 89]  
Gates: `CNOT(0,3) H(1) CNOT(1,0) H(8) CNOT(8,5) H(4) CNOT(4,1) CNOT(1,2) H(3) CNOT(4,7) CNOT(3,6) CNOT(4,1) CNOT(5,4) CNOT(4,3) CNOT(3,0) CNOT(5,4) CNOT(2,5) CNOT(4,3) CNOT(3,4) CNOT(5,2) CNOT(0,3)`

**3x3 / board / seed 4**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 8 (2 min-weight), d_Z = 3 (2), w* = 4 (degree 3), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 1, 1, 2, 0, 1, 0, 0]). Circuit: 27 gates (21 CNOT, 6 H, 0 S), depth 12. Undetected mass board/ion/heron = 0.001067/0.002344/0.003132. Reward terms: distance +0.000, padding +0.000, noise -0.0246, LDPC -0.000.

Stabilizers (sparsest basis): `YIIXIIYII`, `IYYIIIYYI`, `IIIIXIIIX`, `IZIIXXIZI`, `ZIIIIZXXI`, `ZIIZIIIII`, `IZZIIIIII`, `IIIIZZIIZ`  
Weight enumerator of S: [1, 0, 3, 4, 11, 22, 45, 88, 68, 14]; of N(S): [1, 0, 3, 23, 41, 115, 225, 277, 242, 97]  
Gates: `CNOT(0,1) CNOT(1,4) H(0) H(4) CNOT(0,1) CNOT(4,5) CNOT(0,3) CNOT(7,4) CNOT(2,1) CNOT(3,6) CNOT(4,5) CNOT(1,2) H(8) CNOT(4,7) H(3) CNOT(3,0) CNOT(5,8) H(7) CNOT(1,4) CNOT(8,7) CNOT(6,3) CNOT(7,4) CNOT(3,6) H(6) CNOT(4,7) CNOT(8,5) CNOT(4,5)`

**3x3 / ion / seed 0**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (4), w* = 4 (degree 3), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 0, 2, 0, 0, 1, 0, 0, 0]), pure-Z 15 (weights [0, 0, 3, 0, 7, 0, 5, 0, 0, 0]). Circuit: 15 gates (11 CNOT, 4 H, 0 S), depth 7. Undetected mass board/ion/heron = 0.002333/0.001215/0.001473. Reward terms: distance +0.000, padding +0.000, noise -0.0280, LDPC -0.000.

Stabilizers (sparsest basis): `XXXIIIIII`, `XIIYYXIII`, `ZZIXIIXII`, `IIIIIIXXX`, `IZZIIIIII`, `IIIZZIZZI`, `IIIIZZIII`, `IIIIIIIZZ`  
Weight enumerator of S: [1, 0, 3, 4, 11, 22, 45, 88, 68, 14]; of N(S): [1, 0, 3, 23, 41, 115, 225, 277, 242, 97]  
Gates: `H(4) CNOT(4,3) CNOT(0,3) H(1) CNOT(1,0) CNOT(4,5) CNOT(1,2) H(3) CNOT(3,6) H(7) CNOT(7,6) CNOT(5,2) CNOT(4,1) CNOT(4,3) CNOT(7,8)`

**3x3 / ion / seed 1**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (2 min-weight), d_Z = 4 (1), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 0, 1, 0, 0, 1, 1, 0, 0]), pure-Z 1 (weights [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]). Circuit: 28 gates (21 CNOT, 7 H, 0 S), depth 14. Undetected mass board/ion/heron = 0.002333/0.001244/0.001527. Reward terms: distance +0.000, padding +0.000, noise -0.0287, LDPC -0.000.

Stabilizers (sparsest basis): `XXZIIIIII`, `IXIIIIXXI`, `IYYIXIIZI`, `ZYIXIIYII`, `YIYZIXIII`, `IIIIZIIXZ`, `IIIZIIZZX`, `IIZIZZIII`  
Weight enumerator of S: [1, 0, 0, 4, 10, 20, 56, 92, 61, 12]; of N(S): [1, 0, 0, 16, 34, 104, 248, 320, 229, 72]  
Gates: `H(1) CNOT(1,4) H(3) CNOT(0,1) H(0) CNOT(3,6) CNOT(0,3) CNOT(3,4) CNOT(0,1) CNOT(4,7) CNOT(7,6) CNOT(1,2) H(1) H(7) CNOT(6,7) CNOT(4,5) H(6) CNOT(4,3) CNOT(4,5) CNOT(7,8) CNOT(4,5) H(8) CNOT(6,3) CNOT(4,1) CNOT(2,5) CNOT(1,0) CNOT(6,7) CNOT(3,6)`

**3x3 / ion / seed 2**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (10), w* = 4 (degree 3), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 0, 0, 2, 0, 1, 0, 0, 0]), pure-Z 31 (weights [0, 0, 5, 0, 13, 0, 11, 0, 2, 0]). Circuit: 15 gates (10 CNOT, 5 H, 0 S), depth 9. Undetected mass board/ion/heron = 0.003333/0.001268/0.001832. Reward terms: distance +0.000, padding +0.000, noise -0.0293, LDPC -0.000.

Stabilizers (sparsest basis): `XXXXIIIII`, `XIIIXXIIX`, `IZIYIIYXI`, `ZIIZZIZII`, `IZZIIIIII`, `IIIIZZIII`, `IIIIIZIIZ`, `IIIIIIZZI`  
Weight enumerator of S: [1, 0, 5, 0, 23, 0, 95, 0, 132, 0]; of N(S): [1, 0, 5, 27, 23, 195, 95, 393, 132, 153]  
Gates: `H(3) CNOT(3,6) H(3) CNOT(0,3) H(3) H(1) CNOT(1,0) CNOT(6,7) H(4) CNOT(4,3) CNOT(4,1) CNOT(1,2) CNOT(0,3) CNOT(4,5) CNOT(5,8)`

**3x3 / ion / seed 3**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 5 (2 min-weight), d_Z = 3 (2), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 1, 0, 0, 2, 0, 0, 0, 0]), pure-Z 7 (weights [0, 0, 1, 2, 1, 2, 1, 0, 0, 0]). Circuit: 18 gates (13 CNOT, 5 H, 0 S), depth 11. Undetected mass board/ion/heron = 0.002467/0.001242/0.001851. Reward terms: distance +0.000, padding +0.000, noise -0.0286, LDPC -0.000.

Stabilizers (sparsest basis): `YXXIYIIII`, `YIIIIXYXI`, `IZIYZIYII`, `ZIIIXIZII`, `IIIIIXIIX`, `IZZIIIIII`, `IIIZIIZZI`, `IIIIIZIZZ`  
Weight enumerator of S: [1, 0, 2, 4, 10, 22, 50, 88, 65, 14]; of N(S): [1, 0, 2, 20, 40, 112, 230, 292, 239, 88]  
Gates: `H(4) CNOT(4,1) CNOT(1,0) CNOT(4,5) CNOT(0,3) H(3) CNOT(3,6) CNOT(4,7) CNOT(3,4) H(1) CNOT(1,0) CNOT(0,3) H(3) CNOT(3,4) CNOT(1,2) H(8) CNOT(4,3) CNOT(8,5)`

**3x3 / ion / seed 4**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (3 min-weight), d_Z = 3 (2), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 7 (weights [0, 0, 1, 0, 2, 0, 3, 0, 1, 0]), pure-Z 7 (weights [0, 0, 1, 0, 3, 0, 3, 0, 0, 0]). Circuit: 28 gates (20 CNOT, 8 H, 0 S), depth 14. Undetected mass board/ion/heron = 0.002/0.001834/0.002257. Reward terms: distance +0.000, padding +0.000, noise -0.0423, LDPC -0.000.

Stabilizers (sparsest basis): `XIXXIIXII`, `XYIXYIIII`, `IXIIXXXII`, `IIIYXZYII`, `IIIIIIIYY`, `ZZZIIZIII`, `ZZIZZIIII`, `IIIIIIIZZ`  
Weight enumerator of S: [1, 0, 3, 0, 21, 0, 105, 0, 126, 0]; of N(S): [1, 0, 3, 21, 21, 189, 105, 423, 126, 135]  
Gates: `H(4) CNOT(8,5) CNOT(4,3) H(4) CNOT(0,1) CNOT(4,3) H(0) H(5) CNOT(7,8) CNOT(4,3) CNOT(0,3) H(7) CNOT(3,0) CNOT(3,4) H(8) H(1) H(7) CNOT(5,2) CNOT(5,4) CNOT(8,7) CNOT(3,6) CNOT(6,3) CNOT(1,2) CNOT(3,0) CNOT(4,3) CNOT(1,4) CNOT(3,0) CNOT(0,1)`

**3x3 / heron / seed 0**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (1), w* = 4 (degree 5), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 0, 0, 3, 0, 0, 0, 0, 0]), pure-Z 7 (weights [0, 0, 0, 2, 3, 2, 0, 0, 0, 0]). Circuit: 29 gates (24 CNOT, 5 H, 0 S), depth 12. Undetected mass board/ion/heron = 0.0009333/0.001632/0.00182. Reward terms: distance +0.000, padding +0.000, noise -0.0420, LDPC -0.000.

Stabilizers (sparsest basis): `YXIIYIXII`, `IXXIIXIIX`, `IIXXIIIXX`, `IIIIXIZXX`, `IIIZIIXZI`, `ZZIIIZIII`, `IZZIZIIZI`, `IIIIIZIZZ`  
Weight enumerator of S: [1, 0, 0, 3, 10, 23, 56, 89, 61, 13]; of N(S): [1, 0, 0, 14, 34, 110, 248, 314, 229, 74]  
Gates: `H(4) H(5) CNOT(7,8) CNOT(0,3) H(1) CNOT(4,3) CNOT(7,6) CNOT(1,2) H(0) CNOT(1,4) CNOT(0,1) CNOT(4,7) CNOT(7,4) CNOT(3,6) CNOT(1,4) CNOT(2,5) CNOT(1,2) CNOT(4,5) CNOT(7,8) CNOT(5,2) CNOT(1,2) CNOT(5,2) CNOT(4,3) CNOT(7,4) H(6) CNOT(4,3) CNOT(5,2) CNOT(2,1) CNOT(5,8)`

**3x3 / heron / seed 1**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (10), w* = 4 (degree 3), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 0, 0, 2, 0, 1, 0, 0, 0]), pure-Z 31 (weights [0, 0, 5, 0, 13, 0, 11, 0, 2, 0]). Circuit: 14 gates (9 CNOT, 5 H, 0 S), depth 7. Undetected mass board/ion/heron = 0.0004/0.002031/0.001512. Reward terms: distance +0.000, padding +0.000, noise -0.0349, LDPC -0.000.

Stabilizers (sparsest basis): `YYXIIIZII`, `XIIXIIXXI`, `IIIXXXIIX`, `ZZIZZIIII`, `IZZIIIIII`, `IIIIZZIII`, `IIIIIZIIZ`, `IIIIIIZZI`  
Weight enumerator of S: [1, 0, 5, 0, 23, 0, 95, 0, 132, 0]; of N(S): [1, 0, 5, 27, 23, 195, 95, 393, 132, 153]  
Gates: `H(6) CNOT(6,3) H(0) CNOT(6,7) H(0) CNOT(0,3) H(0) CNOT(0,1) CNOT(1,2) H(5) CNOT(5,4) CNOT(3,0) CNOT(4,3) CNOT(5,8)`

**3x3 / heron / seed 2**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 6 (1 min-weight), d_Z = 3 (5), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), pure-Z 15 (weights [0, 0, 2, 0, 5, 0, 8, 0, 0, 0]). Circuit: 21 gates (11 CNOT, 10 H, 0 S), depth 9. Undetected mass board/ion/heron = 0.002667/0.001517/0.001537. Reward terms: distance +0.000, padding +0.000, noise -0.0355, LDPC -0.000.

Stabilizers (sparsest basis): `YIXYXIIII`, `XXZIXIIII`, `IIIYIIYXI`, `IIIIXXIIX`, `ZZIZIIZII`, `XYIIYZIII`, `IIIIIZIIZ`, `IIIIIIZZI`  
Weight enumerator of S: [1, 0, 2, 4, 12, 20, 46, 92, 67, 12]; of N(S): [1, 0, 2, 22, 36, 110, 238, 290, 235, 90]  
Gates: `H(1) H(3) CNOT(3,6) H(3) CNOT(1,4) H(3) CNOT(1,0) CNOT(1,2) H(1) CNOT(6,7) CNOT(1,4) H(1) CNOT(0,3) H(1) H(3) CNOT(0,3) H(3) CNOT(1,0) H(5) CNOT(5,8) CNOT(5,4)`

**3x3 / heron / seed 3**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (2), w* = 4 (degree 5), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 0, 0, 2, 0, 1, 0, 0, 0]), pure-Z 3 (weights [0, 0, 0, 0, 1, 2, 0, 0, 0, 0]). Circuit: 30 gates (24 CNOT, 6 H, 0 S), depth 13. Undetected mass board/ion/heron = 0.003267/0.002157/0.001668. Reward terms: distance +0.000, padding +0.000, noise -0.0385, LDPC -0.000.

Stabilizers (sparsest basis): `XXIIIIXXI`, `IXIXIYIIY`, `IIYIIYIXI`, `IYIYXIZII`, `IIIIZIXII`, `IIZIIZIIX`, `ZZIIIIXIX`, `IZZZIIIZI`  
Weight enumerator of S: [1, 0, 1, 2, 11, 26, 51, 86, 64, 14]; of N(S): [1, 0, 1, 15, 35, 119, 243, 293, 232, 85]  
Gates: `H(3) H(8) CNOT(5,2) CNOT(8,7) CNOT(8,5) H(4) CNOT(1,4) CNOT(4,1) CNOT(3,6) CNOT(3,0) H(4) CNOT(4,1) CNOT(8,5) H(7) CNOT(4,3) CNOT(0,3) CNOT(4,5) CNOT(1,2) CNOT(1,0) CNOT(0,1) CNOT(4,1) CNOT(8,5) CNOT(5,8) H(8) CNOT(8,7) CNOT(2,1) CNOT(7,6) CNOT(6,7) CNOT(5,4) CNOT(2,5)`

**3x3 / heron / seed 4**: highest reward among states that reached the target distance.
[[9,1,3]], d_X = 3 (1 min-weight), d_Z = 4 (1), w* = 4 (degree 4), CSS: no; no Hadamard frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 1, 1, 0, 1, 0, 0, 0, 0]), pure-Z 3 (weights [0, 0, 1, 1, 0, 1, 0, 0, 0, 0]). Circuit: 19 gates (13 CNOT, 6 H, 0 S), depth 9. Undetected mass board/ion/heron = 0.0024/0.001574/0.001662. Reward terms: distance +0.000, padding +0.000, noise -0.0383, LDPC -0.000.

Stabilizers (sparsest basis): `XZIXIIXII`, `XXZZIIIII`, `IZXIZIIZI`, `YIYIYIYII`, `IIIIXIXXI`, `IIIIIXIIX`, `IIIZZIZII`, `IIIIIZIIZ`  
Weight enumerator of S: [1, 0, 3, 2, 9, 30, 49, 78, 66, 18]; of N(S): [1, 0, 3, 17, 45, 129, 217, 267, 246, 99]  
Gates: `H(3) H(7) H(1) CNOT(1,2) CNOT(0,1) CNOT(3,0) CNOT(7,6) CNOT(1,4) H(1) CNOT(3,6) H(5) CNOT(4,3) CNOT(7,4) CNOT(2,5) CNOT(1,2) CNOT(0,1) H(1) CNOT(5,8) CNOT(1,0)`

### Analysis

#### (2) Does training on the evaluation profile help? Seed-level comparison

For each evaluation profile B: undetected mass of the five codes trained on B against the five trained on A, evaluated on B. `pairs` counts seed pairs (i, j) with mass(B-trained_i) < mass(A-trained_j) out of 25; p is the exact two-sided permutation p-value of the difference of means (252 relabelings, so the smallest attainable p is 0.0079). With five seeds a ratio near 1 with p above 0.1 is not distinguishable from no effect.

| evaluated on B | trained on B (mean) | trained on A | A-trained mean | ratio A/B | pairs B < A | p |
|---|---|---|---|---|---|---|
| board | 0.0007467 | ion | 0.002493 | 3.34 | 25/25 | 0.008 |
| board | 0.0007467 | heron | 0.001933 | 2.59 | 19/25 | 0.135 |
| ion | 0.001361 | board | 0.001793 | 1.32 | 22/25 | 0.048 |
| ion | 0.001361 | heron | 0.001782 | 1.31 | 22/25 | 0.040 |
| heron | 0.00164 | board | 0.00216 | 1.32 | 25/25 | 0.008 |
| heron | 0.00164 | ion | 0.001788 | 1.09 | 16/25 | 0.397 |

#### (1) Structural bias

Counts over seeds of codes with d_Z > d_X and d_X > d_Z (pure-type distances in the physical frame), and the per-seed pure-Z : pure-X stabilizer element counts. The ion profile's idle noise is Z-dephasing, so the structural prediction is more Z-detecting structure (more X-type stabilizer weight, larger or more numerous minimum-weight pure-Z logicals detected, that is fewer minimum-weight pure-Z logicals) under ion than under board.

- board: d_Z > d_X in 1/5 seeds, d_X > d_Z in 1/5; minimum-weight pure-Z : pure-X logical counts 1:2, 1:1, 3:2, 5:1, 2:2; pure-Z : pure-X stabilizer elements 3:3, 1:1, 15:3, 15:7, 7:1; CSS in 0/5, non-CSS with a local-Clifford CSS frame in 0/5.
- ion: d_Z > d_X in 1/5 seeds, d_X > d_Z in 1/5; minimum-weight pure-Z : pure-X logical counts 4:1, 1:2, 10:1, 2:2, 2:3; pure-Z : pure-X stabilizer elements 15:3, 1:3, 31:3, 7:3, 7:7; CSS in 0/5, non-CSS with a local-Clifford CSS frame in 0/5.
- heron: d_Z > d_X in 1/5 seeds, d_X > d_Z in 1/5; minimum-weight pure-Z : pure-X logical counts 1:1, 10:1, 5:1, 2:2, 1:1; pure-Z : pure-X stabilizer elements 7:3, 31:3, 15:1, 3:3, 3:3; CSS in 0/5, non-CSS with a local-Clifford CSS frame in 0/5.

#### (3) Discovered against known codes

| evaluated on | best known code (polished encoder) | its mass | discovered, trained on the same profile: mean [min, max] | seeds beating the best known | seeds beating every known code |
|---|---|---|---|---|---|
| board | rotated_surface | 0.0008667 | 0.000747 [0.000667, 0.00107] | 4/5 | 5/5 |
| ion | rotated_surface | 0.00316 | 0.00136 [0.00121, 0.00183] | 5/5 | 5/5 |
| heron | rotated_surface | 0.002457 | 0.00164 [0.00151, 0.00182] | 5/5 | 5/5 |

### Structural bias by training profile

| trained on | seeds with a code | d_Z > d_X | d_X > d_Z | min-weight pure-Z : pure-X logicals (per seed) | pure-Z : pure-X stabilizer elements (per seed) | CSS | LC/H frame (non-CSS) | H gates (mean) | S gates (mean) | CNOTs (mean) | depth (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| board | 5 | 1 | 1 | 1:2, 1:1, 3:2, 5:1, 2:2 | 3:3, 1:1, 15:3, 15:7, 7:1 | 0 | 0 | 6.2 | 0.0 | 17.6 | 10.4 |
| ion | 5 | 1 | 1 | 4:1, 1:2, 10:1, 2:2, 2:3 | 15:3, 1:3, 31:3, 7:3, 7:7 | 0 | 0 | 5.8 | 0.0 | 15.0 | 11.0 |
| heron | 5 | 1 | 1 | 1:1, 10:1, 5:1, 2:2, 1:1 | 7:3, 31:3, 15:1, 3:3, 3:3 | 0 | 0 | 6.4 | 0.0 | 16.2 | 10.0 |

## Setting 4x4: n = 16, k = 1, target d = 4, square:4x4, gates H, CNOT

No run of this setting reached distance 4 (15 runs finished). The largest distance of a terminal state per run and the mean step reward are in the runs table and the training curves; the best state per run below is the terminal state of the largest distance with the smallest undetected mass under the training profile, which is not a distance-4 code, so no cross-evaluation, bias, or known-code comparison is made for this setting. Until a state reaches the target distance the profile does not enter the reward, so runs of the three profiles with the same seed are the same trajectory (identical mean-reward curves and terminal states); the profile column below is therefore not a treatment.

| trained on | seed | largest terminal d | distinct states at that d | final mean step reward | d of best state | d_X | d_Z | gates |
|---|---|---|---|---|---|---|---|---|
| board | 0 | 3 | 17 | -1.021 | 3 | 3 | 3 | 50 |
| board | 1 | 3 | 512 | -0.988 | 3 | 3 | 3 | 50 |
| board | 2 | 3 | 512 | -1.013 | 3 | 7 | 3 | 50 |
| board | 3 | 3 | 512 | -1.003 | 3 | 4 | 3 | 50 |
| board | 4 | 3 | 43 | -1.016 | 3 | 4 | 3 | 50 |
| ion | 0 | 3 | 17 | -1.021 | 3 | 4 | 3 | 50 |
| ion | 1 | 3 | 512 | -0.988 | 3 | 3 | 3 | 50 |
| ion | 2 | 3 | 512 | -1.013 | 3 | 7 | 3 | 50 |
| ion | 3 | 3 | 512 | -1.003 | 3 | 4 | 3 | 50 |
| ion | 4 | 3 | 43 | -1.016 | 3 | 4 | 3 | 50 |
| heron | 0 | 3 | 17 | -1.021 | 3 | 3 | 3 | 50 |
| heron | 1 | 3 | 512 | -0.988 | 3 | 3 | 4 | 50 |
| heron | 2 | 3 | 512 | -1.013 | 3 | 4 | 3 | 50 |
| heron | 3 | 3 | 512 | -1.003 | 3 | 3 | 3 | 50 |
| heron | 4 | 3 | 43 | -1.016 | 3 | 4 | 3 | 50 |

### Known codes at the same (n, k)

Baselines for the setting (polished encoder per profile), for reference:

| code | d | d_X | d_Z | w* | CSS | board (gates, unpolished mass) | ion (gates, unpolished mass) | heron (gates, unpolished mass) |
|---|---|---|---|---|---|---|---|---|
| rotated_surface (rotated surface [[16,1,4]]) | 4 | 4 | 4 | 4 | yes | 0.001933 (64, 0.004267) | 0.005927 (66, 0.007316) | 0.005108 (64, 0.006476) |
| xzzx (XZZX surface [[16,1,4]]: H on the checkerboard [1, 3, 4, 6, 9, 11, 12, 14]) | 4 | 4 | 4 | 4 | no | 0.001933 (72, 0.004267) | 0.005949 (74, 0.007359) | 0.005108 (72, 0.006476) |
| surface_h_outer_cols (rotated surface with H on the outer columns A=[0, 3, 4, 7, 8, 11, 12, 15] (the best Forte deformation of ion-search/deform at d = 3)) | 4 | 4 | 4 | 4 | no | 0.001933 (72, 0.004267) | 0.005927 (74, 0.007381) | 0.005108 (72, 0.006476) |
| shor (Shor [[16,1,4]]: ZZ along rows, X on row pairs (w* = 8)) | 4 | 4 | 4 | 8 | yes | 0.006533 (97, 0.02567) | 0.01123 (87, 0.02581) | 0.008812 (87, 0.02374) |
| transposed_shor (transposed Shor [[16,1,4]]: XX along rows, Z on row pairs (ion-search shor:3x3T at d = 3)) | 4 | 4 | 4 | 8 | yes | 0.005733 (112, 0.02567) | 0.01123 (103, 0.02592) | 0.008537 (109, 0.02549) |

## Setting n7: n = 7, k = 1, target d = 3, all_to_all:7, gates H, S, CNOT

### Cross-evaluation matrix

Rows: training profile (the best code of each seed, its own encoding circuit). Columns: evaluation profile at the calibrated scale. Entries: undetected fault mass, mean over seeds with [min, max]; per seed values in the code list below. Smaller is better.

| trained on \ evaluated on | board | ion | heron | d reached (seeds) | gates (mean) |
|---|---|---|---|---|---|
| board | 0.000693 [0.000533, 0.000933] | 0.00172 [0.00145, 0.00228] | 0.00178 [0.00161, 0.00186] | 3, 3, 3, 3, 3 | 15.4 |
| ion | 0.00235 [0.00233, 0.0024] | 0.00108 [0.00103, 0.00121] | 0.00147 [0.00141, 0.00165] | 3, 3, 3, 3, 3 | 14.0 |
| heron | 0.00231 [0.00227, 0.00233] | 0.00109 [0.00103, 0.00118] | 0.00135 [0.00123, 0.00141] | 3, 3, 3, 3, 3 | 13.8 |

Per seed (undetected mass under board / ion / heron):

| trained on | seed | d | d_X | d_Z | w* | CSS | frame | board | ion | heron | gates | CNOTs | depth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| board | 0 | 3 | 3 | 3 | 4 | no | none | 0.0006667 | 0.001569 | 0.001864 | 10 | 7 | 4 |
| board | 1 | 3 | 3 | 3 | 4 | no | none | 0.0009333 | 0.001449 | 0.001745 | 19 | 14 | 13 |
| board | 2 | 3 | 3 | 3 | 4 | no | none | 0.0006667 | 0.001569 | 0.001864 | 14 | 10 | 6 |
| board | 3 | 3 | 3 | 3 | 4 | no | none | 0.0005333 | 0.002283 | 0.001605 | 19 | 15 | 9 |
| board | 4 | 3 | 3 | 3 | 4 | no | none | 0.0006667 | 0.001736 | 0.001813 | 15 | 11 | 7 |
| ion | 0 | 3 | 3 | 3 | 4 | no | none | 0.002333 | 0.001032 | 0.001412 | 15 | 12 | 11 |
| ion | 1 | 3 | 3 | 3 | 4 | no | none | 0.002333 | 0.001032 | 0.001412 | 12 | 9 | 9 |
| ion | 2 | 3 | 3 | 3 | 4 | no | none | 0.0024 | 0.001214 | 0.001645 | 14 | 10 | 12 |
| ion | 3 | 3 | 3 | 3 | 4 | no | none | 0.002333 | 0.001068 | 0.001464 | 13 | 10 | 12 |
| ion | 4 | 3 | 3 | 3 | 4 | no | none | 0.002333 | 0.001032 | 0.001412 | 16 | 9 | 9 |
| heron | 0 | 3 | 3 | 3 | 4 | no | none | 0.002267 | 0.001181 | 0.001283 | 10 | 7 | 6 |
| heron | 1 | 3 | 3 | 3 | 4 | no | none | 0.002333 | 0.001044 | 0.001412 | 20 | 15 | 12 |
| heron | 2 | 3 | 3 | 3 | 4 | no | none | 0.002267 | 0.001169 | 0.001231 | 11 | 8 | 7 |
| heron | 3 | 3 | 5 | 3 | 4 | no | none | 0.002333 | 0.001044 | 0.001412 | 16 | 12 | 12 |
| heron | 4 | 3 | 3 | 3 | 4 | no | none | 0.002333 | 0.001032 | 0.001412 | 12 | 9 | 9 |

### Mechanism breakdown of the undetected mass

Undetected fault mass of the best codes split by the mechanism that caused it (mean over seeds): two-qubit gate channel, single-qubit gate channel, idle channel with a Z fault, idle channel with an X or Y fault. The board profile has no single-qubit gate channel and a depolarizing idle; the ion profile's idle is 95 percent Z.

| trained on | evaluated on | two-qubit gate | single-qubit gate | idle Z | idle X or Y | total undetected | total fault mass |
|---|---|---|---|---|---|---|---|
| board | board | 6.27e-04 | 0.00e+00 | 0.00e+00 | 6.67e-05 | 6.93e-04 | 0.039 |
| board | ion | 1.37e-03 | 4.37e-05 | 2.88e-04 | 1.99e-05 | 1.72e-03 | 0.041 |
| board | heron | 1.70e-03 | 6.21e-05 | 0.00e+00 | 1.24e-05 | 1.78e-03 | 0.041 |
| ion | board | 3.47e-04 | 0.00e+00 | 6.67e-04 | 1.33e-03 | 2.35e-03 | 0.060 |
| ion | ion | 7.58e-04 | 1.46e-05 | 2.76e-04 | 2.76e-05 | 1.08e-03 | 0.036 |
| ion | heron | 9.43e-04 | 2.07e-05 | 3.24e-04 | 1.82e-04 | 1.47e-03 | 0.043 |
| heron | board | 3.07e-04 | 0.00e+00 | 6.67e-04 | 1.33e-03 | 2.31e-03 | 0.051 |
| heron | ion | 6.70e-04 | 7.29e-06 | 3.74e-04 | 4.23e-05 | 1.09e-03 | 0.037 |
| heron | heron | 8.34e-04 | 1.04e-05 | 3.24e-04 | 1.82e-04 | 1.35e-03 | 0.041 |

Most visited target-distance code of the last 20 updates (the converged policy's code), undetected mass under board / ion / heron:

| trained on | seed | visits (of successes in the window) | d | d_X | d_Z | w* | CSS | board | ion | heron | gates | same as best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| board | 0 | 8188 (8190) | 3 | 3 | 3 | 4 | no | 0.0006667 | 0.001569 | 0.001864 | 12 | no |
| board | 1 | 7369 (7463) | 3 | 5 | 3 | 6 | no | 0.001333 | 0.001403 | 0.001458 | 11 | no |
| board | 2 | 9087 (9089) | 3 | 5 | 3 | 6 | no | 0.0006667 | 0.001569 | 0.001864 | 12 | no |
| board | 3 | 8192 (8192) | 3 | 5 | 3 | 6 | no | 0.0009333 | 0.002082 | 0.001851 | 12 | no |
| board | 4 | 7758 (7789) | 3 | 3 | 3 | 4 | no | 0.0006667 | 0.001569 | 0.001864 | 13 | no |
| ion | 0 | 4266 (6123) | 3 | 3 | 3 | 4 | no | 0.002333 | 0.001032 | 0.001412 | 11 | no |
| ion | 1 | 8188 (8201) | 3 | 3 | 3 | 6 | no | 0.002333 | 0.001032 | 0.001412 | 10 | no |
| ion | 2 | 9096 (9097) | 3 | 5 | 3 | 6 | no | 0.0024 | 0.001226 | 0.001645 | 9 | no |
| ion | 3 | 6749 (6825) | 3 | 3 | 3 | 4 | no | 0.002333 | 0.001068 | 0.001464 | 13 | yes |
| ion | 4 | 7445 (7445) | 3 | 5 | 3 | 4 | no | 0.002333 | 0.001032 | 0.001412 | 12 | no |
| heron | 0 | 2895 (4982) | 3 | 3 | 3 | 4 | no | 0.002267 | 0.001181 | 0.001283 | 10 | yes |
| heron | 1 | 8169 (8171) | 3 | 3 | 3 | 6 | no | 0.001333 | 0.001122 | 0.001406 | 15 | no |
| heron | 2 | 7318 (7340) | 3 | 3 | 3 | 4 | no | 0.002267 | 0.001169 | 0.001231 | 11 | yes |
| heron | 3 | 8598 (8601) | 3 | 3 | 3 | 6 | no | 0.002333 | 0.001067 | 0.001412 | 12 | no |
| heron | 4 | 7436 (7437) | 3 | 5 | 3 | 4 | no | 0.002333 | 0.001032 | 0.001412 | 13 | no |

### Known codes at the same (n, k)

Encoders: the Gottesman standard-form encoder of the CSS parent code followed by H on the frame, routed onto the setting's graph (nearest-neighbour CNOT chains), then polished per profile by a greedy local search that keeps the stabilizer group fixed and lowers the undetected mass (single-gate deletions, adjacent swaps, single-gate replacements, 60.0 s per profile); on all-to-all also stim's elimination encoder. The entry is the polished encoder's undetected mass under that profile (a different circuit per column). The RL codes have encoders co-optimized with the code by the agent, so this comparison is fair only to the extent the polishing reaches comparable circuits; the unpolished mass and gate counts are given to show how much the polishing did.

| code | d | d_X | d_Z | w* | CSS | board (gates, unpolished mass) | ion (gates, unpolished mass) | heron (gates, unpolished mass) |
|---|---|---|---|---|---|---|---|---|
| steane (Steane [[7,1,3]], CSS, self-dual) | 3 | 3 | 3 | 4 | yes | 0.0008 (14, 0.001133) | 0.002328 (20, 0.004186) | 0.002175 (14, 0.002395) |

### Best code per seed

**n7 / board / seed 0**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (6), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 1, 0, 2, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 0, 3, 0, 2, 0]). Circuit: 10 gates (7 CNOT, 3 H, 0 S), depth 4. Undetected mass board/ion/heron = 0.0006667/0.001569/0.001864. Reward terms: distance +0.000, padding +0.000, noise -0.0241, LDPC -0.000.

Stabilizers (sparsest basis): `YYIIXIZ`, `IXXXIIX`, `IIIXIXI`, `ZIIIZII`, `ZZIZIZI`, `IIZIIIZ`  
Weight enumerator of S: [1, 0, 3, 0, 15, 0, 45, 0]; of N(S): [1, 0, 3, 30, 15, 108, 45, 54]  
Gates: `CNOT(0,3) H(6) CNOT(6,3) H(0) CNOT(0,4) CNOT(3,1) CNOT(6,2) H(5) CNOT(5,3) CNOT(0,1)`

**n7 / board / seed 1**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (1), w* = 4 (degree 4), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 0, 0, 0, 1, 0, 0]), pure-Z 1 (weights [0, 0, 0, 1, 0, 0, 0, 0]). Circuit: 19 gates (14 CNOT, 5 H, 0 S), depth 13. Undetected mass board/ion/heron = 0.0009333/0.001449/0.001745. Reward terms: distance +0.000, padding +0.000, noise -0.0337, LDPC -0.000.

Stabilizers (sparsest basis): `YXIIIYX`, `IYIYXXI`, `IIYYIIX`, `IZZXIXI`, `IIXIZZZ`, `ZZIIZII`  
Weight enumerator of S: [1, 0, 0, 2, 9, 24, 22, 6]; of N(S): [1, 0, 0, 17, 45, 78, 82, 33]  
Gates: `H(5) CNOT(0,3) CNOT(5,3) CNOT(5,4) H(0) CNOT(0,5) CNOT(3,2) CNOT(5,6) H(0) H(2) CNOT(3,6) CNOT(4,6) H(3) CNOT(1,4) CNOT(3,1) CNOT(4,3) CNOT(0,3) CNOT(3,6) CNOT(3,4)`

**n7 / board / seed 2**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (3), w* = 4 (degree 4), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 0 (weights [0, 0, 0, 0, 0, 0, 0, 0]), pure-Z 3 (weights [0, 0, 1, 0, 2, 0, 0, 0]). Circuit: 14 gates (10 CNOT, 4 H, 0 S), depth 6. Undetected mass board/ion/heron = 0.0006667/0.001569/0.001864. Reward terms: distance +0.000, padding +0.000, noise -0.0241, LDPC -0.000.

Stabilizers (sparsest basis): `XIIXZIX`, `IXIIIZI`, `ZIXXXII`, `IZYIZYI`, `ZIIIIIZ`, `ZIZZIZI`  
Weight enumerator of S: [1, 0, 2, 0, 17, 0, 44, 0]; of N(S): [1, 0, 2, 27, 17, 114, 44, 51]  
Gates: `CNOT(0,5) H(5) H(2) CNOT(2,0) CNOT(5,1) CNOT(0,4) H(1) CNOT(5,6) CNOT(2,3) H(0) CNOT(0,6) CNOT(5,2) CNOT(0,3) CNOT(5,6)`

**n7 / board / seed 3**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (6), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 1, 0, 0, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 0, 3, 0, 2, 0]). Circuit: 19 gates (15 CNOT, 4 H, 0 S), depth 9. Undetected mass board/ion/heron = 0.0005333/0.002283/0.001605. Reward terms: distance +0.000, padding +0.000, noise -0.0193, LDPC -0.000.

Stabilizers (sparsest basis): `YYXIXII`, `ZXIXIIX`, `IIIIXXI`, `ZIZIIII`, `IZIIZZZ`, `IIIZIIZ`  
Weight enumerator of S: [1, 0, 3, 0, 15, 0, 45, 0]; of N(S): [1, 0, 3, 30, 15, 108, 45, 54]  
Gates: `H(4) CNOT(4,5) H(0) CNOT(0,2) H(6) CNOT(6,3) CNOT(6,0) CNOT(0,1) CNOT(3,2) CNOT(3,4) H(1) CNOT(0,5) CNOT(6,3) CNOT(1,3) CNOT(6,3) CNOT(3,4) CNOT(2,5) CNOT(3,4) CNOT(1,6)`

**n7 / board / seed 4**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (1), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 1, 0, 2, 0, 0, 0]), pure-Z 3 (weights [0, 0, 2, 0, 1, 0, 0, 0]). Circuit: 15 gates (11 CNOT, 4 H, 0 S), depth 7. Undetected mass board/ion/heron = 0.0006667/0.001736/0.001813. Reward terms: distance +0.000, padding +0.000, noise -0.0241, LDPC -0.000.

Stabilizers (sparsest basis): `YIIZYZI`, `IXXIIII`, `ZZYIIYI`, `IIXXIXX`, `ZIIIZII`, `IIIZIIZ`  
Weight enumerator of S: [1, 0, 3, 0, 15, 0, 45, 0]; of N(S): [1, 0, 3, 30, 15, 108, 45, 54]  
Gates: `CNOT(0,1) H(2) H(4) CNOT(4,0) H(0) H(5) CNOT(2,1) CNOT(5,6) CNOT(0,3) CNOT(3,4) CNOT(0,5) CNOT(5,2) CNOT(3,0) CNOT(4,0) CNOT(5,3)`

**n7 / ion / seed 0**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (6), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 1, 0, 0, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 0, 3, 0, 2, 0]). Circuit: 15 gates (12 CNOT, 3 H, 0 S), depth 11. Undetected mass board/ion/heron = 0.002333/0.001032/0.001412. Reward terms: distance +0.000, padding +0.000, noise -0.0373, LDPC -0.000.

Stabilizers (sparsest basis): `YIIYXIX`, `IXIIIIX`, `ZIXXIXI`, `ZIIIZII`, `IZIZIZZ`, `IIZIIZI`  
Weight enumerator of S: [1, 0, 3, 0, 15, 0, 45, 0]; of N(S): [1, 0, 3, 30, 15, 108, 45, 54]  
Gates: `H(6) CNOT(6,1) CNOT(0,1) H(5) CNOT(5,0) CNOT(5,2) CNOT(0,3) H(0) CNOT(0,3) CNOT(0,4) CNOT(3,1) CNOT(0,4) CNOT(0,1) CNOT(0,4) CNOT(3,6)`

**n7 / ion / seed 1**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (4), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 0, 0, 0, 0, 1, 0]), pure-Z 7 (weights [0, 0, 2, 0, 5, 0, 0, 0]). Circuit: 12 gates (9 CNOT, 3 H, 0 S), depth 9. Undetected mass board/ion/heron = 0.002333/0.001032/0.001412. Reward terms: distance +0.000, padding +0.000, noise -0.0373, LDPC -0.000.

Stabilizers (sparsest basis): `YXIIXIY`, `YZIXYII`, `IIXYXYI`, `ZIIZIZZ`, `IZIIIIZ`, `IIZIIZI`  
Weight enumerator of S: [1, 0, 2, 0, 17, 0, 44, 0]; of N(S): [1, 0, 2, 27, 17, 114, 44, 51]  
Gates: `H(5) CNOT(5,2) CNOT(5,0) CNOT(0,4) H(1) CNOT(1,0) H(0) CNOT(5,3) CNOT(1,6) CNOT(0,3) CNOT(6,0) CNOT(0,4)`

**n7 / ion / seed 2**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (6), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 1, 0, 2, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 0, 3, 0, 2, 0]). Circuit: 14 gates (10 CNOT, 3 H, 1 S), depth 12. Undetected mass board/ion/heron = 0.0024/0.001214/0.001645. Reward terms: distance +0.000, padding +0.000, noise -0.0439, LDPC -0.000.

Stabilizers (sparsest basis): `YXIIYZI`, `IIXIIIX`, `IIXXXXI`, `ZZIIIII`, `ZIZIZIZ`, `IIIZIZI`  
Weight enumerator of S: [1, 0, 3, 0, 15, 0, 45, 0]; of N(S): [1, 0, 3, 30, 15, 108, 45, 54]  
Gates: `H(4) CNOT(4,5) CNOT(0,5) H(0) CNOT(0,1) CNOT(0,6) H(2) CNOT(0,5) CNOT(5,3) CNOT(0,6) S(6) CNOT(0,6) CNOT(2,6) CNOT(4,2)`

**n7 / ion / seed 3**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (4), w* = 4 (degree 4), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 0 (weights [0, 0, 0, 0, 0, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 0, 5, 0, 0, 0]). Circuit: 13 gates (10 CNOT, 3 H, 0 S), depth 12. Undetected mass board/ion/heron = 0.002333/0.001068/0.001464. Reward terms: distance +0.000, padding +0.000, noise -0.0386, LDPC -0.000.

Stabilizers (sparsest basis): `XIXYYII`, `IXIYIYX`, `IZIXXIX`, `ZIZIIII`, `IZIIIZI`, `IZZIZIZ`  
Weight enumerator of S: [1, 0, 2, 0, 17, 0, 44, 0]; of N(S): [1, 0, 2, 27, 17, 114, 44, 51]  
Gates: `H(3) CNOT(3,4) CNOT(0,4) H(0) CNOT(0,2) CNOT(2,6) CNOT(1,6) CNOT(3,1) H(1) CNOT(1,5) CNOT(1,6) CNOT(5,3) CNOT(4,6)`

**n7 / ion / seed 4**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (4), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 0 (weights [0, 0, 0, 0, 0, 0, 0, 0]), pure-Z 3 (weights [0, 0, 2, 0, 1, 0, 0, 0]). Circuit: 16 gates (9 CNOT, 6 H, 1 S), depth 9. Undetected mass board/ion/heron = 0.002333/0.001032/0.001412. Reward terms: distance +0.000, padding +0.000, noise -0.0373, LDPC -0.000.

Stabilizers (sparsest basis): `XIZXIXI`, `IXXYYII`, `IIIIYIY`, `ZIIZZIX`, `ZIIIIZI`, `IZZIIII`  
Weight enumerator of S: [1, 0, 3, 0, 15, 0, 45, 0]; of N(S): [1, 0, 3, 30, 15, 108, 45, 54]  
Gates: `H(4) CNOT(4,6) CNOT(4,0) CNOT(0,1) CNOT(4,3) H(0) CNOT(0,5) CNOT(0,3) H(4) CNOT(4,6) H(4) H(0) S(2) H(0) CNOT(2,1) CNOT(1,2)`

**n7 / heron / seed 0**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (6), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 3 (weights [0, 0, 1, 0, 2, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 0, 3, 0, 2, 0]). Circuit: 10 gates (7 CNOT, 3 H, 0 S), depth 6. Undetected mass board/ion/heron = 0.002267/0.001181/0.001283. Reward terms: distance +0.000, padding +0.000, noise -0.0463, LDPC -0.000.

Stabilizers (sparsest basis): `YYXIIZI`, `IXIXXXI`, `IIIXIIX`, `ZIZIIII`, `ZZIZIIZ`, `IIIIZZI`  
Weight enumerator of S: [1, 0, 3, 0, 15, 0, 45, 0]; of N(S): [1, 0, 3, 30, 15, 108, 45, 54]  
Gates: `H(5) CNOT(5,3) CNOT(0,3) H(0) CNOT(3,1) CNOT(0,1) CNOT(0,2) CNOT(5,4) H(6) CNOT(6,3)`

**n7 / heron / seed 1**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (3), w* = 4 (degree 5), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 0, 0, 1, 0, 0, 0]), pure-Z 3 (weights [0, 0, 0, 0, 3, 0, 0, 0]). Circuit: 20 gates (15 CNOT, 4 H, 1 S), depth 12. Undetected mass board/ion/heron = 0.002333/0.001044/0.001412. Reward terms: distance +0.000, padding +0.000, noise -0.0510, LDPC -0.000.

Stabilizers (sparsest basis): `XXIYIIX`, `XIIZXIZ`, `YIYXIYI`, `IIIYXXX`, `YIZZYII`, `IZZIIZZ`  
Weight enumerator of S: [1, 0, 0, 0, 21, 0, 42, 0]; of N(S): [1, 0, 0, 21, 21, 126, 42, 45]  
Gates: `H(5) CNOT(5,2) CNOT(5,0) CNOT(5,4) CNOT(0,1) H(1) H(6) CNOT(3,4) CNOT(6,5) CNOT(1,6) CNOT(0,3) CNOT(6,3) CNOT(1,4) CNOT(5,4) H(0) S(3) CNOT(5,4) CNOT(6,4) CNOT(1,4) CNOT(0,4)`

**n7 / heron / seed 2**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (1 min-weight), d_Z = 3 (5), w* = 4 (degree 4), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 0, 0, 1, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 0, 5, 0, 0, 0]). Circuit: 11 gates (8 CNOT, 3 H, 0 S), depth 7. Undetected mass board/ion/heron = 0.002267/0.001169/0.001231. Reward terms: distance +0.000, padding +0.000, noise -0.0445, LDPC -0.000.

Stabilizers (sparsest basis): `YIYIXIX`, `IXIXIXX`, `XIIXIZZ`, `ZIZZIZI`, `IZIIIZI`, `IIZIZII`  
Weight enumerator of S: [1, 0, 2, 0, 17, 0, 44, 0]; of N(S): [1, 0, 2, 27, 17, 114, 44, 51]  
Gates: `H(2) CNOT(2,4) H(5) CNOT(5,6) CNOT(2,0) CNOT(0,6) CNOT(5,1) H(0) CNOT(0,3) CNOT(2,0) CNOT(5,3)`

**n7 / heron / seed 3**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 5 (2 min-weight), d_Z = 3 (4), w* = 4 (degree 4), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 0, 0, 1, 0, 0, 0]), pure-Z 3 (weights [0, 0, 2, 0, 1, 0, 0, 0]). Circuit: 16 gates (12 CNOT, 4 H, 0 S), depth 12. Undetected mass board/ion/heron = 0.002333/0.001044/0.001412. Reward terms: distance +0.000, padding +0.000, noise -0.0510, LDPC -0.000.

Stabilizers (sparsest basis): `XIIIXXX`, `IXYYIXI`, `ZIXZZII`, `ZIZIXZI`, `ZIIIIIZ`, `IZIZIII`  
Weight enumerator of S: [1, 0, 2, 0, 17, 0, 44, 0]; of N(S): [1, 0, 2, 27, 17, 114, 44, 51]  
Gates: `H(3) CNOT(3,1) CNOT(3,0) CNOT(0,5) H(0) CNOT(0,4) CNOT(0,2) CNOT(1,6) CNOT(2,0) H(0) CNOT(0,5) CNOT(3,2) CNOT(1,6) H(2) CNOT(0,6) CNOT(0,4)`

**n7 / heron / seed 4**: highest reward among states that reached the target distance.
[[7,1,3]], d_X = 3 (2 min-weight), d_Z = 3 (6), w* = 4 (degree 3), CSS: no; no local Clifford frame makes it CSS. Pure-X stabilizer elements 1 (weights [0, 0, 1, 0, 0, 0, 0, 0]), pure-Z 7 (weights [0, 0, 2, 0, 3, 0, 2, 0]). Circuit: 12 gates (9 CNOT, 3 H, 0 S), depth 9. Undetected mass board/ion/heron = 0.002333/0.001032/0.001412. Reward terms: distance +0.000, padding +0.000, noise -0.0510, LDPC -0.000.

Stabilizers (sparsest basis): `YIXXIYI`, `IXXIIII`, `ZIIIXXX`, `ZIIZIII`, `IZZIZZI`, `IIIIZIZ`  
Weight enumerator of S: [1, 0, 3, 0, 15, 0, 45, 0]; of N(S): [1, 0, 3, 30, 15, 108, 45, 54]  
Gates: `H(2) H(4) CNOT(4,6) CNOT(4,0) CNOT(0,1) CNOT(0,5) H(0) CNOT(0,3) CNOT(0,5) CNOT(2,1) CNOT(3,2) CNOT(4,2)`

### Analysis

#### (2) Does training on the evaluation profile help? Seed-level comparison

For each evaluation profile B: undetected mass of the five codes trained on B against the five trained on A, evaluated on B. `pairs` counts seed pairs (i, j) with mass(B-trained_i) < mass(A-trained_j) out of 25; p is the exact two-sided permutation p-value of the difference of means (252 relabelings, so the smallest attainable p is 0.0079). With five seeds a ratio near 1 with p above 0.1 is not distinguishable from no effect.

| evaluated on B | trained on B (mean) | trained on A | A-trained mean | ratio A/B | pairs B < A | p |
|---|---|---|---|---|---|---|
| board | 0.0006933 | ion | 0.002347 | 3.38 | 25/25 | 0.008 |
| board | 0.0006933 | heron | 0.002307 | 3.33 | 25/25 | 0.008 |
| ion | 0.001076 | board | 0.001721 | 1.60 | 25/25 | 0.008 |
| ion | 0.001076 | heron | 0.001094 | 1.02 | 14/25 | 0.873 |
| heron | 0.00135 | board | 0.001778 | 1.32 | 25/25 | 0.008 |
| heron | 0.00135 | ion | 0.001469 | 1.09 | 18/25 | 0.159 |

#### (1) Structural bias

Counts over seeds of codes with d_Z > d_X and d_X > d_Z (pure-type distances in the physical frame), and the per-seed pure-Z : pure-X stabilizer element counts. The ion profile's idle noise is Z-dephasing, so the structural prediction is more Z-detecting structure (more X-type stabilizer weight, larger or more numerous minimum-weight pure-Z logicals detected, that is fewer minimum-weight pure-Z logicals) under ion than under board.

- board: d_Z > d_X in 0/5 seeds, d_X > d_Z in 0/5; minimum-weight pure-Z : pure-X logical counts 6:1, 1:1, 3:1, 6:2, 1:2; pure-Z : pure-X stabilizer elements 7:3, 1:1, 3:0, 7:1, 3:3; CSS in 0/5, non-CSS with a local-Clifford CSS frame in 0/5.
- ion: d_Z > d_X in 0/5 seeds, d_X > d_Z in 0/5; minimum-weight pure-Z : pure-X logical counts 6:2, 4:2, 6:1, 4:1, 4:1; pure-Z : pure-X stabilizer elements 7:1, 7:1, 7:3, 7:0, 3:0; CSS in 0/5, non-CSS with a local-Clifford CSS frame in 0/5.
- heron: d_Z > d_X in 0/5 seeds, d_X > d_Z in 1/5; minimum-weight pure-Z : pure-X logical counts 6:1, 3:2, 5:1, 4:2, 6:2; pure-Z : pure-X stabilizer elements 7:3, 3:1, 7:1, 3:1, 7:1; CSS in 0/5, non-CSS with a local-Clifford CSS frame in 0/5.

#### (3) Discovered against known codes

| evaluated on | best known code (polished encoder) | its mass | discovered, trained on the same profile: mean [min, max] | seeds beating the best known | seeds beating every known code |
|---|---|---|---|---|---|
| board | steane | 0.0008 | 0.000693 [0.000533, 0.000933] | 4/5 | 4/5 |
| ion | steane | 0.002328 | 0.00108 [0.00103, 0.00121] | 5/5 | 5/5 |
| heron | steane | 0.002175 | 0.00135 [0.00123, 0.00141] | 5/5 | 5/5 |

### Structural bias by training profile

| trained on | seeds with a code | d_Z > d_X | d_X > d_Z | min-weight pure-Z : pure-X logicals (per seed) | pure-Z : pure-X stabilizer elements (per seed) | CSS | LC/H frame (non-CSS) | H gates (mean) | S gates (mean) | CNOTs (mean) | depth (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| board | 5 | 0 | 0 | 6:1, 1:1, 3:1, 6:2, 1:2 | 7:3, 1:1, 3:0, 7:1, 3:3 | 0 | 0 | 4.0 | 0.0 | 11.4 | 7.8 |
| ion | 5 | 0 | 0 | 6:2, 4:2, 6:1, 4:1, 4:1 | 7:1, 7:1, 7:3, 7:0, 3:0 | 0 | 0 | 3.6 | 0.4 | 10.0 | 10.6 |
| heron | 5 | 0 | 1 | 6:1, 3:2, 5:1, 4:2, 6:2 | 7:3, 3:1, 7:1, 3:1, 7:1 | 0 | 0 | 3.4 | 0.2 | 10.2 | 9.2 |
