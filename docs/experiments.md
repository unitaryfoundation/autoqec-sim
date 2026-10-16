# Experiments

## Noise-aware discovery

`experiments/noise_aware.py` asks whether the code PPO discovers depends
on the noise profile it is trained against, and in the direction the
profile's structure predicts (the experiment the proposal cites [@olle2024] for). Three `autoqec_sim.noise` profiles scaled to the
same total fault mass (board depolarizing; Forte [@chen2024] ion chain with sequential
timing, so idle dephasing dominates; Heron r2 rates [@harper2026] on the setting's
graph), three settings (7 qubits all-to-all with H, S, CNOT and target
distance 3; 3x3 lattice with H, CNOT and distance 3; 4x4 lattice with H,
CNOT and distance 4), five PPO seeds per cell. The reward is the shifted
distance reward plus, on states at the target distance, the circuit-level
undetected fault mass (`normalize=False`) and the LDPC penalty at
`w_max = 4`; the exact combination, the calibration, the cross-evaluation
matrices, the code lists, the known-code baselines, and the verdict are in
`experiments/noise_aware/runs/REPORT.md`. One cell reproduces with

```
uv run --extra rl python -m autoqec_sim.experiments.noise_aware run --setting 3x3 --profile ion --seed 0
uv run --extra rl python -m autoqec_sim.experiments.noise_aware baselines --setting 3x3
uv run --extra rl python -m autoqec_sim.experiments.noise_aware report
```

(`--threads N` caps torch and BLAS; `experiments/noise_aware/runs/run_sweep.sh`
with the job scripts under `experiments/noise_aware/runs/jobs/` is the 3-worker,
6-thread queue the release numbers came from.)

Result (45 runs, 15 per setting): the discovered code depends strongly on
the training profile, but through the encoding circuit's fault propagation
rather than through X/Z structure. Board-trained codes have 3.4x less
undetected fault mass on the board profile than ion- or heron-trained
codes (25 of 25 seed pairs, permutation p = 0.008, both settings) and
ion-trained codes 1.3x to 1.6x less on the ion profile (p = 0.008 and
0.048); ion- and heron-trained codes are interchangeable at n = 7. No
profile produced codes with `d_Z > d_X` more often than another (0/5 or
1/5 seeds everywhere), none of the 30 codes is CSS or local-Clifford
equivalent to a CSS code, and the mechanism breakdown shows the
board-trained circuits detect every idle fault while ion-trained circuits
leave idle X and Y faults undetected, faults the ion profile hardly emits.
The discovered degenerate codes beat Steane [@steane1996] (n = 7) and the rotated
surface code and its XZZX [@bonillaataides2021] and Hadamard-deformed variants (3x3) with
polished encoders under every profile; a Hadamard deformation of the code
is invisible to this single-fault circuit objective, which is one reason
no such structure emerges. The 4x4 setting (target d = 4) did not fit the
budget: none of its 15 runs (16M steps each) produced a distance-4 state,
with either the level or the fractional distance term. Without the padding
penalty the agent embeds a smaller code and leaves qubits idle.

## Memory objective

`memory_objective.py` scores a code by the logical error rate per round of
an `R`-round syndrome-extraction memory experiment instead of by its
encoder's single-fault mass, which cannot see a local-Clifford deformation
of the code (a Hadamard at the end of the encoder relabels weight-1
faults, all detected; `experiments/noise_aware/runs/REPORT.md`). The circuit is
built for arbitrary (non-CSS) generators, one ancilla per generator with
controlled-X/Y/Z gates in a hook-aware order (`hook_orders`: without it
the sorted order gives the Shor code a circuit-level distance of 1 in the
Z basis), generators that share a qubit with different Pauli letters in
separate batches so every interleaving commutes (X block then Z block for
a CSS code), optional one-flag ancillas, noiseless `MPP` preparation and
readout, and an all-to-all or `square:RxC` layout (ancilla sites at the
plaquette centers, non-local generators routed with CNOT ladders and run
in their own batch). The noise is a `autoqec_sim.noise.Profile` with its timing
model; stim's detector error model feeds three tiers:

| tier | what | cost per code, both bases, `R = d`, one thread |
|---|---|---|
| `proxy` | leading-order failure of the maximum-likelihood decoder over fault sets of weight at most two, plus exact `d_circ` up to weight 4 (`exact_dcirc`, from `ion-search/lib.py`) | 0.08 to 0.22 s at `n <= 9` (the DEM derivation dominates), 1.7 to 2.3 s for the [[16,4,4]] code (about 2000 mechanisms, 60 detectors) |
| `sampled` | exact ML decoder by dynamic program when detectors plus observables `<= 20` (no shots), else a weight-2 lookup decoder with a one-mechanism completion, sampled | 0.16 to 0.31 s at `n <= 9` (5000 shots grown to 200 failures), 5 s at `n = 16` |
| `bposd` | the board's pinned BP+OSD [@roffe2020; @panteleev2021] (`ldpc`, `uv run --with ldpc`), sampled; validation only | 0.2 to 1.5 s at `n <= 9`, 37 to 42 s at `n = 16` |

`MemoryReward(objective, d*)` is the environment reward: the distance and
padding terms of the noise-aware experiment plus
`0.1 * log10(p_ref / p_round)` on target-distance states (cached per
code); `CodeDiscoveryEnv` adopts its `success_fn`. Design note:
`docs/memory_objective.md`; runs: `experiments/memory_objective/REPORT.md`
(`python -m autoqec_sim.experiments.memory_objective {benchmark,random,random-check,bias,lattice,ppo,report}`).

Results (chain preset at scale 0.25, board at 1, Heron at 0.1; the tiers
are leading-order or sampled estimates and need the expected number of
faults per shot below about one, which the presets at scale 1 exceed):

- Rank agreement. On 32 random `[[9,1,3]]` codes (`R = 3`) the Spearman
  correlation of the proxy against bposd is 0.79 (ion), 0.90 (board),
  0.78 (heron); on the five benchmark codes 0.9, 0.8, 0.3, where the
  outlier is Steane, for which bposd is 1.7 to 3.2 times worse than both
  the proxy and the near-ML lookup decoder (proxy against lookup 0.9 on
  every profile). The proxy is within 25 percent of the exact-ML or
  lookup sampled tier on every benchmark code below one expected fault
  per shot; bposd deviates where it decodes badly (Steane, the [[16,4,4]]
  code under the ion preset at 2x).
- Bias sensitivity. Rotated surface against XZZX at `n = 9`, `R = 3`,
  every rate 2.5e-4 with a slot idle channel of bias `eta`: the ratio
  `p_round(plain) / p_round(xzzx)` is 1.00 at `eta = 0.5` (exact tie, the
  deformation is a relabeling), 1.21 at 2, 1.80 at 10, 2.27 at pure Z
  under sequential timing (bposd 1.08, 1.23, 1.89, 2.19), and 1.00 to
  2.10 under parallel timing. Under the Forte set (duration idle,
  `eta = 10`) the proxy prefers XZZX by 1.22 and the lookup decoder by
  1.21; bposd does not (0.78). Under the chain set the two tie (0.99):
  its idle is small and its readout flips dominate.
- Flags. One flag per generator lifts `d_circ` from 2 to 3 for [[5,1,3]],
  Steane, the [[16,4,4]] code, and the non-CSS 3x3 code, and lowers the
  ion `p_round` by 2.6 to 4.4 times (0.23 to 0.38 of the bare value); on
  the board recipe the extra gates cost more than the flags gain (1.0 to
  1.5 times the bare value).
- Lattice. On `square:3x3` the surface and XZZX codes route nothing; the
  non-CSS 3x3 code needs 63 extra two-qubit gates per round (270 against
  81 in 3 rounds) and 7 sequential batches, and its board `p_round` goes
  from 0.018 to 0.26.
- PPO smoke run (3x3 lattice, H and CNOT, chain profile at 0.25, `R = 3`,
  batch 256, 400k env steps, 62 s at one thread): distance 3 reached at
  83k env steps, 92 distinct target-distance codes evaluated at 0.47 s
  each (44 s of memory evaluation in total, all-to-all layout); the best
  has `p_round` 0.0016 against 0.00077 for the rotated surface code. The
  test-suite version (`R = 2`, 300k steps) is the regression.

## RL scaling probe

`experiments/scaling.py` asks how far the PPO agent scales in `n` and what the
cheapest change is that gets it to a distance-4 code on the 4x4 lattice (the
setting the noise-aware experiment could not solve), plus how the
LDPC-constrained target behaves at `n = 12` to `16`. Findings, from
`experiments/scaling/REPORT.md`: the 4x4 failure is exploration (d = 4 states
within 50 gates exist but are too rare for the default policy to ever see
one). This holds for this setting, nearest-neighbor 4x4 connectivity, the
unrestricted H and CNOT action space, and 50-gate episodes; a flat policy
with CSS-restricted actions on a next-to-nearest-neighbor periodic
lattice reaches distance 5 at 17 qubits in [@olle2024]. Here a per-qubit
permutation-equivariant policy
(now `agents.policies.equivariant_actor_critic`, `PPOConfig(policy="equivariant")`) reaches a clean `[[16, 1, 4]]` at the
50-gate cap in 0.2M to 0.26M env steps on every seed (about 20x fewer than the
flat MLP with a 4x larger batch, which also works), and it is the recommended
default for `n >= 12`. The LDPC penalty form of the regression does not carry
over: in 18 runs at `n = 12` to `16` no PPO policy reached a distance-3 code
with `w* <= 4`; all converge to sparse groups of distance 1, so the penalty is
validated at `n = 9` only. The limits above are limits of gate-by-gate
encoder search. Code design under hardware constraints is NP-complete in
general, and a SAT formulation nonetheless designs physics-inspired codes
with up to 100 physical qubits in minutes and hardware-aware codes for
biased noise with lower logical error rates than surface codes
[@dalfavero2026]; that search space, check matrices rather than encoding
circuits, is where the rewards here (circuit-level noise, the memory
objective) and the schedule synthesizer take a code as input. One run
reproduces with

```
uv run --extra rl python -m autoqec_sim.experiments.scaling variant --variant equivariant --seed 0
uv run --extra rl python -m autoqec_sim.experiments.scaling ldpc --n 12 --k 2 --graph all_to_all:12 --seed 0
uv run --extra rl python -m autoqec_sim.experiments.scaling report
```

## Three-topology run matrix

`experiments/topologies.py` runs the PPO agent by itself (not the LLM
meta-agent) on square, triangular, and heavy-hex
lattices with one objective and one metric set: a `[[n, 1, d >= 3]]` code with
no idle qubit, H and CNOT on the lattice edges, 50-gate episodes, the level
distance reward of the scaling probe, `PPOConfig` defaults with `policy="auto"`,
three seeds per cell. Two tiers: `3x3` and `4x4` for square and triangular
(`n = 9`, `16`); heavy-hex takes the nearest sizes the honeycomb construction
allows, `1x1` (`n = 12`, a 12-cycle) and `2x1` (`n = 18`). Every cell reports
success, env steps and wall time to first success, the best code (exact `d`,
`w*`, maximum qubit degree, CSS test, weight enumerators), the undetected fault
mass of its encoder under the Heron preset on the cell's graph and under the
board preset, and the same mass for the rotated surface encoder routed onto the
graph where one exists. Results and the write-up are in
`experiments/topologies/runs/REPORT.md`: all 18 runs reached distance 3, at 6k to 12k
env steps on the 9-qubit lattices (within the first three updates, before any
learning), 49k to 113k at 12 and 16 qubits, and 307k to 345k on the 18-qubit
heavy-hex graph.

```
uv run --extra rl python -m autoqec_sim.experiments.topologies matrix
uv run --extra rl python -m autoqec_sim.experiments.topologies run --topology heavy_hex --target-n 16 --seed 0
uv run --extra rl python -m autoqec_sim.experiments.topologies report
```

## Evolutionary search against PPO

`autoqec_sim.experiments.topologies_ea` runs the three-topology matrix with the package's evolutionary search in place of the PPO agent: same cells, objective ([[n, 1, 3]] with no idle qubit), gate set, 50-gate episodes, and environment-step budget per tier (evolutionary evaluations count as whole episodes). Run 16 October 2026 on one laptop; results in `experiments/topologies_ea/runs/`.

| lattice | n | PPO successes | PPO median steps to success | EA successes | EA median steps to success | EA median wall |
|---|---|---|---|---|---|---|
| square | 9 | 3/3 | 9,472 | 0/3 | none | 22 s |
| triangular | 9 | 3/3 | 7,424 | 0/3 | none | 24 s |
| heavy_hex | 12 | 3/3 | 63,488 | 0/3 | none | 21 s |
| square | 16 | 3/3 | 112,128 | 0/3 | none | 54 s |
| triangular | 16 | 3/3 | 75,776 | 0/3 | none | 46 s |
| heavy_hex | 18 | 3/3 | 340,480 | 0/3 | none | 38 s |

Totals: PPO 18 of 18 runs succeed, evolutionary search 0 of 18. The evolutionary search is autoqec-sim's own (population of gate sequences, mutation, one batched rollout per generation), not a tuned evolutionary agent; the comparison fixes the budget, objective, and action space, nothing else.

## QDX's targets

`autoqec_sim.experiments.qdx_targets` runs the full target set that QDX, the
code of [@olle2024], publishes for all-to-all connectivity at `n <= 15`
(50 `[[n, k, d]]` triples with `d` up to 5) through this package's PPO at
matched gate set, connectivity, and success criterion, one run per target
with a budget of 3M environment steps or 10 minutes, and verifies each
result by exact distance. `experiments/qdx_targets/REPORT.md` holds the
table; the QDX columns are read from its archive of discovered codes.

```
uv run --extra rl python -m autoqec_sim.experiments.qdx_targets run --tier d3
uv run --extra rl python -m autoqec_sim.experiments.qdx_targets report
```

## SAT-designed codes under the memory objective

`autoqec_sim.experiments.sat_codes` takes the 9-qubit surface-like codes
that [@dalfavero2026] sampled by SAT and ranked by exact maximum-likelihood
logical error rate under code-capacity biased noise, selects the twelve
best distinct codes at bias 10 and three of the worst, and scores them with
the circuit-level memory objective against the rotated surface code and its
XZZX deformation under a biased idle profile at three bias values and under
the Forte preset, at the proxy and sampled tiers. The question is whether
the code-capacity ranking survives circuit-level noise.
`experiments/sat_codes/REPORT.md` holds the tables and the rank
correlations.

```
uv run python -m autoqec_sim.experiments.sat_codes run
uv run python -m autoqec_sim.experiments.sat_codes report
```

## A population of SAT-designed codes

`autoqec_sim.experiments.sat_population` scales the previous experiment
from fifteen codes to a population: a uniform random sample of 2,000 of the
43,563 distinct stabilizer groups among the 100,000 codes of
[@dalfavero2026], plus the 100 best and the 50 worst by the source's
code-capacity rate at bias 10, scored under four cases (biased idle at
source bias 4 and 10, every channel biased at bias 10, and the Forte
preset) at the proxy and sampled tiers. The report gives the rank
correlation between the two scores, the overlap of their best sets, the
circuit-level rates of the source's best codes against random ones, the
structural features of a code's extraction circuit that predict its
circuit-level rate, and the best codes of the population against the
rotated surface code and XZZX. `experiments/sat_population/REPORT.md`
holds the tables and the figure. The `prepare` step reads the source
repository's `experiments/surface/` directory and writes
`experiments/sat_population/population.json`, which is committed.

```
uv run python -m autoqec_sim.experiments.sat_population prepare --source <satisfying-codes>/experiments/surface
uv run python -m autoqec_sim.experiments.sat_population run --workers 4
uv run python -m autoqec_sim.experiments.sat_population report
```
