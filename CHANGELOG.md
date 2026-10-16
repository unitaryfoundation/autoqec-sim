# Changelog

## Unreleased

- `experiments.qdx_targets`: QDX's published all-to-all target set
  through this package's PPO at matched settings, verified by exact
  distance (`experiments/qdx_targets/REPORT.md`). `experiments.sat_codes`:
  the SAT-sampled biased-noise codes of DalFavero et al. under the
  circuit-level memory objective against the rotated surface and XZZX
  codes (`experiments/sat_codes/REPORT.md`).

## 0.2.0 (2026-10-16)

- Repository layout: `tools/` (release and reproduce scripts), `benchmarks/`
  (the CPU profile and the CUDA kernel benchmark), one `experiments/<name>/`
  directory per experiment with raw output under `experiments/<name>/runs/`
  (not tracked) and the regression JSONs under `experiments/regression/`,
  and the quick-start scripts under `docs/examples/`, snippeted into the
  documentation and checked against the README by `tests/test_examples.py`.
- Documentation bibliography: `docs/refs.bib` (every entry checked against
  arXiv or Crossref) rendered by mkdocs-bibtex; pages cite sources by key
  and end with their references instead of inline arXiv numbers.
- `experiments/regression/figure.py`: states evaluated up to a verified
  rediscovery for both agents, from the regression records, as
  `docs/assets/figures/rediscovery.png` in the README and the regression page.
- `StabilizerCode.from_gates(n, k, gates)` reads the code off a gate
  sequence such as `SearchResult.gates`, and `EvolutionarySearch` is
  exported from the package root, so the README quick start is a search,
  a code, and its stabilizers (`docs/examples/quickstart.py`). The longer
  rediscover, verify, and score walkthrough is
  `docs/examples/rediscover_and_score.py`. README after clifft's: logo, badges,
  one-line install, short quick start; CODE_OF_CONDUCT.md (Contributor
  Covenant 2.1).
- `backends.cuda_kernels`: two opt-in fused CUDA kernels (CuPy, NVRTC) behind
  the torch backend, a per-element gate kernel and a bit-packed popcount
  parity kernel; agreement tests against the torch path; benchmark script
  and A100 numbers in `docs/gpu.md`.
First tagged release. Three additions on top of 0.1.0: the noise module
in the circuit-level reward, an LDPC sparsity constraint, and a PPO
agent with its own rediscovery regression.

- The noise module ships as the `autoqec_sim.noise` subpackage: biased,
  correlated, and leakage-aware Pauli channels, device timing, topologies,
  and sourced presets for IonQ Forte, a long ion chain, IBM Heron, and the
  qldpc-challenge recipe. `circuit_noise` accepts a `autoqec_sim.noise.Profile`,
  a preset name, or any `stim.Circuit -> stim.Circuit` callable
  (`resolve_noise_model`); profiles run through `autoqec_sim.noise.noisifier`
  with `scale`. Gate lists are laid out as TICK-layered skeletons
  (`gates_to_layered_circuit`, as-soon-as-possible scheduling), so the
  profile's timing model and idle channel apply. Fault mechanisms are
  enumerated from every noise instruction the module emits (`DEPOLARIZE1/2`,
  `X/Y/Z_ERROR`, `PAULI_CHANNEL_1/2`, `CORRELATED_ERROR` and
  `ELSE_CORRELATED_ERROR` chains with absolute probabilities,
  `HERALDED_ERASE` as heralded, hence detected) and propagated through
  the remaining gates as before. `dem_undetected_mass` derives the same
  undetected mass from stim's detector error model of the circuit plus
  an ideal stabilizer and logical readout, used as a cross-check (agreement
  to `O(p^2)`, the merging error of the DEM). The placeholder
  `depolarizing_noise_model` stays and gained an `idle` rate; with
  `p1=None, p2=p, idle=p` it reproduces the board preset mechanism for
  mechanism.
- `graphs` delegates every builder to `autoqec_sim.noise.topology`, with
  `from_topology`, `Graph.to_topology`, and `as_graph`; the environment
  accepts either graph class and exposes `env.topology`. A profile
  without a graph takes the environment's graph in `resolve_noise_model`
  and `make_circuit_level_reward`; a profile whose graph differs is
  refused. `heavy_hex(rows, cols)` is now the `autoqec_sim.noise` honeycomb
  construction (rows x cols hexagons), not the earlier brick-wall one.
- `ldpc`: bit-packed batched GF(2) elimination; exact minimum bottleneck
  weight `w*` (smallest `w` such that the weight-`<= w` elements generate
  the group) for `n - k <= 14` by enumerating the `2^(n-k)` group elements,
  a greedy pairwise generator reduction as the fallback bound; the
  sparsest basis and its maximum qubit degree; `ldpc_feasible`,
  `ldpc_penalty`, `make_ldpc_penalty`.
- `env`: hard LDPC constraint `ldpc_w_max` with modes `terminate`, `mask`
  (revert to a no-op, plus `feasible_action_mask()` for agents that mask
  logits), and `final` (feasibility gates success only). `info["success"]`
  distinguishes success from termination by violation; `rollout` uses it.
- `search`: the reported best circuit is the executed history (masked
  actions dropped) and is chosen among successful elements when any
  exist; before, an element at the reward threshold but outside the
  constraint could be reported as the solution.
- `regress`: `LdpcSpec` on targets, exclusion targets
  (`expect_success=False`, `passed` field), `min_max_generator_weight`
  and `max_qubit_degree` in every verification, five new targets
  (`rotated_surface_9_ldpc`, `rotated_surface_9_ldpc_penalty`,
  `steane_ldpc_w4`, `seven_qubit_w3_excluded`, `five_qubit_w3_excluded`).
- Observation from the `w*` column of the regression: the unconstrained
  surface-code result of 0.1.0 (seed 20) has the surface code's weight
  enumerators of `S` and `N(S)` but `w* = 6` (its 26 elements of weight
  at most 4 span a rank-7 subgroup, the reference's span rank 8), an
  instance of the known fact that equal enumerators do not imply
  local-Clifford equivalence; the mask-constrained target's match has
  `w* = 4`.
- PPO agent (`agents.ppo`): clipped objective, GAE, shared value head,
  entropy bonus, action masking (static graph mask, dynamic undo mask, and
  any mask the environment exposes), PyTorch on CPU with a six-thread cap.
  torch is the optional `rl` extra and is imported lazily.
- PPO rediscovery regression (`agents.ppo_regress`, run as `python -m`):
  same targets, reward, and enumerator verification as `regress`; every
  distinct code a run produces is verified. Full-budget results in
  `experiments/regression/regression_ppo.json`: [[5,1,3]], Steane, and the rotated surface
  code all matched on seed 0 with 3 to 17 times fewer state evaluations
  than the evolutionary search.
- `tests/test_ppo.py` (6 tests, skipped without torch; 15 items with the
  policy tests below), including a reduced-budget [[5,1,3]] rediscovery
  that trains PPO in about 1.5 s.
- `autoqec-ppo` console script for the PPO regression (`uv run --extra rl autoqec-ppo`).
- Noise-aware discovery experiment (`experiments.noise_aware`, run as
  `python -m`): PPO under three `autoqec_sim.noise` profiles (board depolarizing,
  Forte ion chain with sequential timing, Heron rates on the setting's
  graph) scaled to equal total fault mass, in three settings (7 qubits
  all-to-all with H, S, CNOT; 3x3 and 4x4 square lattices with H, CNOT),
  five seeds per cell. Reward: shifted distance reward (fractional below
  the target on 4x4) and the padding penalty of the regression, plus, on
  states at the target distance with no idle qubit, the circuit-level
  undetected fault mass (`normalize=False`, weight `2 / M_0`) and the LDPC
  penalty at `w_max = 4`; such a state ends the episode through the new
  `success_fn` hook of `CodeDiscoveryEnv`. Per seed: the best and the most visited
  target-distance codes with exact `d`, pure-X and pure-Z distances,
  local-Clifford CSS-frame search, `w*`, enumerators, and the undetected
  mass under all three profiles (cross-evaluation); known codes (Steane,
  rotated surface, XZZX, Hadamard-deformed surface, Shor, transposed Shor
  at `n = 9` and `n = 16`) with standard-form encoders routed onto the
  lattice and polished per profile; training-curve CSVs and figures
  (`experiments.figstyle`, one style); `experiments/noise_aware/runs/REPORT.md`.
- RL scaling probe (`experiments.scaling`, run as `python -m`; report in
  `experiments/scaling/REPORT.md`, runs in `experiments/scaling/runs/`). Diagnosis
  of the 4x4 failure of the noise-aware experiment (`[[16, 1, 4]]`, H and
  CNOT on the lattice, 50-gate episodes): a distance-4 state is reachable
  at 50 gates (evolutionary search finds one after 369k circuits) but rare
  (0 of 8192 random 50-gate sequences, 2 at d = 3), the default agent never
  evaluates one in 21M to 24M env steps, and its terminal states are no
  better than random on the padding term (8 idle qubits, 87 to 91 percent
  CNOT actions), so the failure is exploration, not credit assignment or
  representational capacity. Variants, one change each: a target-distance
  curriculum learns d = 3 at n = 16 but not d = 4; CZ plus plaquette
  macros reach d = 4 in 23k to 38k steps by raising the gate budget to
  about 200 per episode; a per-qubit permutation-equivariant policy
  (`scaling.equivariant_actor_critic`, 124k parameters) reaches a clean
  d = 4 at the 50-gate cap in 217k to 258k env steps on 3 of 3 seeds (71
  to 110 s) and produces d = 4 codes in 56 to 64 percent of episodes after
  1M steps; 1024 parallel episodes with entropy coefficient 0.001 reach it
  with the flat MLP in 4.2M to 4.7M steps; the baseline alone needs an
  80-gate cap (3.1M and 5.4M steps) and fails at 60 and 70 in 30 minutes.
  The LDPC penalty form of the regression does not scale past n = 9: in 18
  runs at n = 12 and 16, k = 2 and 4, on all-to-all and square graphs (penalty
  scales 0.25 and 0.05, flat and equivariant policies, 20 minutes each) no run
  reached a distance-3 code with w* <= 4; every policy converges to sparse
  stabilizer groups of distance 1, because the normalized Knill-Laflamme term
  pays 1/630 to 1/1128 per detected error against 0.25 per unit of w* excess.
- Three-topology run matrix (`experiments.topologies`, run as `python -m`;
  report in `experiments/topologies/runs/REPORT.md`). The PPO agent alone, not the
  LLM meta-agent of milestone M2, on square, triangular, and heavy-hex
  lattices at two tiers (`3x3` and `4x4`; heavy-hex `1x1` with 12 qubits and
  `2x1` with 18, the nearest sizes the honeycomb construction allows), one
  objective (`[[n, 1, 3]]`, no idle qubit, H and CNOT on lattice edges,
  50-gate cap, level distance reward), `PPOConfig` defaults with
  `policy="auto"`, three seeds per cell, 2M or 4M env steps and 12 min per
  run. Common metrics per cell: steps and wall time to first success, exact
  `d`, `w*`, qubit degree, CSS test, weight enumerators, undetected fault
  mass of the encoder under the Heron preset on the cell's graph and under
  the board preset, and the routed rotated surface encoder's mass where one
  exists. All 18 runs reached distance 3: within the first three updates on
  the 9-qubit lattices, 49k to 113k env steps at 12 and 16 qubits, 307k to
  345k on the 18-qubit heavy-hex graph. `tests/test_topologies.py` (3 tests,
  skipped without torch).
- `circuit_noise.fault_set` propagates fault mechanisms with a GF(2)
  product against the symplectic map of the circuit suffix (15 to 25
  times faster than the stim `PauliString.after` loop, now
  `fault_set_reference`, which the tests compare against mechanism for
  mechanism); `gates_to_layered_circuit` builds the skeleton from text in
  one parse.
- `env.CodeDiscoveryEnv(success_fn=...)`: an optional success criterion
  replacing the reward threshold.
- `matplotlib` in the `dev` dependency group (figures of the experiment).
- Tests: 44 to 71 for these modules (noise wiring, DEM cross-check, ion
  timing ranking, heavy-hex rejection, graph round trip, LDPC statistics,
  constraint modes, regression); 78 with the PPO agent's 6 and the
  version check.
- `agents.policies`: the two actor-critics of the PPO agent, the flat MLP
  (moved out of `agents.ppo` unchanged) and the permutation-equivariant
  per-qubit network of the scaling probe (moved from
  `experiments.scaling.equivariant_actor_critic`, now also accepting the
  `check_matrix` observation). `PPOConfig.policy` with values `mlp`,
  `equivariant`, and `auto` (the default: equivariant from 12 qubits on,
  mlp below, `EQUIVARIANT_MIN_QUBITS`), `PPOConfig.equivariant_hidden`
  (128), `PPOAgent.policy` and `PPOAgent.build_model()`, and
  `policy_resolved` in the result's config. Below 12 qubits the agent
  builds the same network from the same torch RNG state as before, so the
  reduced PPO regression reproduces `docs/reproduce.md` step for step
  (counts, gates, and per-update history checked). `experiments.scaling`
  keeps its recorded-run protocol (build the MLP, then swap the equivariant
  network in) so `experiments/scaling/runs/` stays reproducible; it now passes
  `policy="mlp"` explicitly. Seven test functions (nine items) in
  `tests/test_ppo.py`, which now collects 15 items.
- `known_codes.rotated_surface(d)` (the rotated surface code of any
  distance on the `d x d` grid, the builder the noise-aware experiment
  used) and `ROTATED_SURFACE_16` (`[[16, 1, 4]]`, `w* = 4`, degree 4) in
  the known-code table.
- PPO-only rediscovery target `rotated_surface_16` in `agents.ppo_regress`
  (`PPO_ONLY_TARGETS`): `[[16, 1, 4]]` on `square:4x4` with H and CNOT,
  50-gate episodes, the regression's reward and enumerator match, a budget
  of 1,000,000 env steps in one attempt (`TARGET_BUDGETS`), skipped by the
  `reduced` budget (`REDUCED_SKIP`); `target_budget()` and `--threads`.
  Result at 2 threads (`experiments/regression/regression_ppo_4x4_seed{0,1}.json`): both
  seeds reach a verified `[[16, 1, 4]]` code (88,576 and 268,288 env
  steps, 46 and 48 gates, non-CSS, `w*` 5 and 6), 66 and 2 distinct
  codes over 1M steps, none with the rotated surface enumerators, so the
  target is not rediscovered at this budget.
- `regress.verify` computes `w*` exactly up to `n - k = 15`
  (`VERIFY_EXACT_GENERATORS`), so the 4x4 target's codes report the exact
  bottleneck weight instead of the greedy bound.
- `ldpc.LdpcObjective`: reward and success criterion of an
  LDPC-constrained `[[n, k, d]]` target in four forms (`LDPC_FORMS`):
  `penalty` (the regression's normalized form), `unnormalized` (one
  undetected error costs one unit against `scale` per unit of `w*`
  excess, divided by the initial state's undetected count),
  `final` (hard final-state constraint, no LDPC term in the reward), and
  `schedule` (distance first, penalty switched on by `activate()`). The
  exact `w*` is evaluated on candidate states only (every error detected,
  no idle qubit), the greedy bound elsewhere. `DEFAULT_LDPC_FORM` is the
  form the comparison below selected.
- `ldpc.LdpcObjective` forms `shaped` (dense per-step term `-shaping *
  excess` in reward units, greedy bound on ordinary states, exact on
  candidates) and `terminal` (no path term; any candidate ends the episode
  with reward `1 - h / (n - k)`, `h` the number of generators of the
  sparsest basis above `w_max`); `ldpc.heavy_generator_count` (`m` minus
  the rank of the light elements, basis independent) and
  `ldpc.greedy_action_mask` (per-action greedy `w*` bound, all two-qubit
  trials in one batched call, single-qubit gates always allowed).
- Sparsity comparison (`experiments.ldpc_sparsity`, run as `python -m`;
  report in `experiments/ldpc_sparsity/REPORT.md`, runs in
  `experiments/ldpc_sparsity/runs/`): `shaped` at `c` = 0.05, 0.1, 0.2, `terminal`,
  and `mask` against `unnormalized` at 1.5M env steps on `[[12, 2, 3]]`
  (all-to-all, `square:3x4`) and `[[16, 4, 3]]` (`square:4x4`), equivariant
  policy, on the RunPod pod's CPU with every run under the compute ledger's
  `wrap`. Measured: the fraction of terminal states at distance 3 with exact
  `w* <= 4`. On the two 12-qubit cells (`[[16, 4, 3]]` pending): the
  `mask` variant makes the bound a property of the policy on
  `all_to_all:12` (three seeds: 91, 68, and 89 percent of the terminal
  states of the last 25 updates at distance 3, every one at `w* = 4`, 769
  to 2,248 distinct feasible codes per run; the reference: 50 to 62
  percent at distance 3, `w*` 5 to 6, none inside) and traps on
  `square:3x4` (every terminal state at `w* = 4` and distance 1); the
  same mask on top of the `terminal` reward (`mask_terminal`, seed 0)
  does it on both cells (95 and 98 percent at distance 3, all at
  `w* = 4`, 2,151 and 166 distinct feasible codes). `shaped` traps at
  distance 1 at every `c`; `terminal` alone collapses onto one dense code
  (`h = 2`, `w* = 7` all-to-all; `h = 3`, `w*` 5 to 6 on the lattice)
  because it optimizes the heavy-generator count, not `w*`;
  `unnormalized` at 3M env steps keeps `w*` 5 to 6. `DEFAULT_LDPC_FORM`
  stays `unnormalized` (the mask is an environment constraint, not a
  form). The mask costs 1.3 to 2 times the wall time per env step.
- `experiments.ldpc_sparsity` scaling preparation: `--device cuda` puts the
  policy, the PPO update, the reward's base terms (`LdpcObjective(device=
  ...)`, the error set on the device as float32, tableau copied once per
  call) and the terminal-state distance on the torch device; four
  all-to-all cells (`n16_k4_all_to_all`, `n20_k4_all_to_all`,
  `n25_k1_d4_all_to_all`, `n25_k1_d5_all_to_all`) with per-cell `d` and
  episode length (`CELL_SPECS`), the greedy `w*` bound as the success test
  above 15 generators (`w_star_test` in the JSON), and a capped code
  description above 20 qubits. `ldpc.greedy_action_mask(method="basis")`,
  now the default, applies every CNOT to the current state's greedy basis
  as packed bit operations and re-reduces only the rows the gate made
  heavy (`_packed_cnot`, `_greedy_within`): 0.05 s against 4.1 s per step
  at `n = 25`, batch 256, same bound, sound against the exact test; the
  `n = 12` runs used `method="tableau"`, which stays available.
  First scaling runs (`mask_terminal`, seed 0, batch 1024, 3M env steps,
  A40 shared with another job): `[[16, 4, 3]]` and `[[20, 4, 3]]` on the
  all-to-all graph never reach a distance-3 terminal state (57 to 59
  percent of two-qubit actions masked at the end, 823 and 520 env
  steps/s); the `[[25, 1]]` cells are queued behind the pod's other GPU
  work.
- `ldpc.greedy_action_mask(method="exact")`: the exact feasibility test
  in the action mask for `n - k <= 15`. The greedy basis bound stays as
  the pre-filter; every (state, action) pair it rejects gets the exact
  test: a CNOT changes the weight of a Pauli by at most one, so the
  elements of weight `<= w_max + 1` of the current group (enumerated once
  per state from the `2^(n - k)` group elements, shared by all of the
  state's actions, kept across steps in an `ExactMaskCache` keyed by the
  greedy basis) are the only ones the gate can leave within the bound;
  the gate is admitted iff those it leaves at weight `<= w_max` span the
  heavy rows of the gate-transformed basis (a GF(2) rank over the
  coefficient bits, uint16 rows, weights from a 16-entry table on the
  local Paulis of the control and the target). Identical states in the
  batch share one test per action. Above 15 generators the method is the
  greedy basis bound and says so in `stats`; two-qubit gates other than
  CNOT are tested on tableau copies of the rejected pairs. Tests: equal to
  `ldpc_feasible` on every trial action, admits everything the greedy
  bound admits, admits some moves the bound forbids, cache hits on
  repeated states. `experiments.ldpc_sparsity`: `--mask-method exact`,
  the run JSON's `mask` entry with the pre-filter and final masked
  fractions, exact tests and admissions per call, and cache counts;
  `bench-mask` (cost per env step of the mask methods on one cell);
  `run_name` tags the exact method (`_exact`) and a batch other than 256
  (`_b1024`); `label` carries the mask method and batch. Cost on the pod's
  CPU (6 threads, 96-core host at load 30 to 90 from other tenants):
  0.6 s per env step on `all_to_all:16` at batch 256 (0.2 s greedy
  pre-filter, 0.4 s exact test over 34,000 rejected pairs per call), 27
  percent of the wall time; 0.08 s on `square:4x4`, 9 percent. Runs
  (`mask_terminal`, 3M env steps, exact mask at batch 256 on three seeds
  and at batch 1024 on one, greedy controls at batch 256; report section
  "Exact feasibility mask"): negative on both `[[16, 4, 3]]` cells. On
  `all_to_all:16` the exact test admits 4 to 7 percent of the pairs the
  greedy bound rejects (masked 53 to 58 percent of two-qubit actions
  against 58 for the greedy control), so the bound was nearly exact and
  the constrained set itself blocks distance 3 under a 40-gate episode;
  one feasible `[[16, 4, 3]]` in 9M env steps over three seeds, half the
  terminal states at distance 2 at the end on seed 0 (greedy control: 6
  percent). On `square:4x4` it admits 12 to 30 percent (masked 31 to 39
  against 42), and every run ends at distance 1. The greedy control at
  batch 256 is as negative as the earlier batch-1024 run, so the update
  count was not the confound. At batch 1024 the exact mask is negative on
  both graphs too (1.7 s per env step on the all-to-all cell, 29 percent
  of the wall time). `n = 20` (`n - k =
  16`) is beyond the exact range and, on this evidence, would not change
  with it; the routes left are a gate set that moves weight inside the
  feasible set (CZ, SWAP), a longer episode, or a curriculum from the
  distance-2 groups. The batch-1024 records of the scaling section were
  renamed to the `_b1024` tag; the tableau-method `square:4x4` record
  carries `_tableau`.
- LDPC-form comparison (`experiments.ldpc_scaling`, run as `python -m`;
  report in `experiments/ldpc_scaling/REPORT.md`, runs in
  `experiments/ldpc_scaling/runs/`): the forms
  `unnormalized`, `final`, and `schedule` on `[[12, 2, 3]]` and
  `[[16, 4, 3]]` with `w_max = 4`, all-to-all and square graphs, three
  seeds each, equivariant policy, 1.5M env steps or 20 minutes per run
  (36 runs plus 4 scale variants). `unnormalized` learns distance-3 codes
  at `n = 12` to `16` where the penalty form never did (61 to 69 percent
  of terminal states at distance 3 on five of six `[[12, 2, 3]]` runs) and
  is the only form that produced codes within the bound (3 of 12 runs, all
  `[[12, 2, 3]]`, at 303k, 354k, and 950k env steps), as single episodes:
  its policies' codes stay at `w*` 5 to 6 (`n = 12`) and 7 to 8 (`n = 16`).
  `final` reaches distance 3 and never a sparse code (0 of 12); `schedule`
  switches its penalty on at 0.63M to 0.91M env steps at `n = 12` and then
  falls into the penalty form's distance-1 trap (0 of 12). `unnormalized`
  is `DEFAULT_LDPC_FORM`; `regress.LdpcSpec(mode="objective", form=...)`
  puts any form on a regression target (`make_reward` returns the
  objective, `env_kwargs(target, reward)` its `success_fn`; both
  regressions pass it). Also existence
  constructions for both targets (`experiments/ldpc_scaling/runs/existence.json`:
  `[[5,1,3]]` plus Steane, and two copies of a searched `[[8,2,3]]` with
  `w* = 4`).
- Tests: the 4x4 target and reference (`tests/test_ppo.py`), the four
  objective forms on known codes and in environment rollouts
  (`tests/test_ldpc.py`), one tiny run per form and the report tables
  (`tests/test_ldpc_scaling.py`).
- `benchmarks/profile_tableau.py` and `docs/profile.md`: CPU throughput and
  memory profile of the vectorized simulator (numpy on the host, not the
  GPU version the roadmap names) for `n` in 5 to 49 and batch 256 to 4096:
  gate applications, reward evaluations, and environment steps per second,
  bytes per batch element of the tableau and of the reward's intermediate
  product, and a time breakdown of one environment step. The reward's
  `(batch, M, n + k)` float32 product and its int64 copy for the mod-2
  reduction are the memory bottleneck (GiB per batch at `n >= 25`, against
  kilobytes for the tableau) and the GEMM plus that cast are where the
  time goes.
- `backends.torch_backend`: the two hot paths of the reward engine on
  torch tensors on any device (`cpu` or `cuda`), opt-in, with the numpy
  path and its results unchanged. Gate kernels `apply_h`, `apply_s`,
  `apply_s_dag`, `apply_x/y/z`, `apply_cnot`, `apply_cz`, `apply_swap`,
  `apply_gate`, and an `ActionTable` with its lookup arrays on the device;
  `symplectic_product`, `code_rows`, `undetected_mask`,
  `in_stabilizer_mask`, `batched_distance`, `kl_reward`, `distance_reward`,
  `low_weight_stabilizer_penalty`, `CompositeReward`, and the `make_*`
  closures with the same names and semantics as the numpy ones;
  `numpy_reward_adapter` plugs a torch reward into `CodeDiscoveryEnv`. The
  parity test reduces the float32 product mod 2 in place and decides both
  commutation tests with one matrix-vector product against a weight
  vector, so the int64 copy and the two boolean `any()` reductions of the
  numpy path (about 80 percent of its reward time on the CPU) and the
  boolean `(batch, M, n + k)` array are gone; the peak intermediate is
  `4 M (n + k)` bytes per element instead of `12 M (n + k)`. On CPU torch
  the reward runs 1.9 times faster than numpy at `n = 9` and 2.3 times at
  `n = 16` (batch 1024); the gate kernels, one to two percent of a step,
  run at about half the numpy rate. `tests/test_torch_backend.py` (43
  items, skipped without torch) checks every kernel and every mask
  against numpy bit for bit on random tableaux and the known codes at
  `n` in 5, 9, 16.
- `benchmarks/profile_tableau.py`: `--backend {numpy,torch}`, `--device
  {cpu,cuda}`, `--json` (record a run), `--render` (tables from records),
  `--no-breakdown`. `docs/gpu.md`: running the profile on a RunPod GPU,
  the A40 result (`experiments/profile_pod/runs/`), and what stays cuQuantum-specific.
- Packaging for release: SPDX license metadata, ruff configuration, CI and release workflows,
  `tools/reproduce.sh`, `docs/reproduce.md`, CITATION.cff. The
  `autoqec-ppo` console script runs the PPO regression.
- All-to-all ion-chain cells (`experiments.ion_chain`, run as `python -m`;
  report in `experiments/ion_chain/runs/REPORT.md`): the noise-aware 4x4 protocol
  of `experiments.gpu_scale` (equivariant PPO, batch 1024, 8M env steps,
  40 min cap, three profiles calibrated to equal total fault mass, three
  seeds) with the graph changed to `all_to_all:16`, in two settings
  defined in the module (`a2a16` with H, S, CNOT and `a2a16_hc` with H,
  CNOT, the lattice gate set). Imports the reward, calibration, tracker,
  and code analysis from `noise_aware` and the GPU reward and run plumbing
  from `gpu_scale`; adds nothing to `noise_aware.SETTINGS`. The report
  gives the cross-evaluation matrix per setting and compares the
  all-to-all codes under the ion profile with the 4x4 lattice cells of
  `experiments/gpu_scale/runs/` (read from their JSON files), the rotated surface
  code's polished lattice encoder (`experiments/noise_aware/runs/baselines_4x4.json`)
  and, from the `baselines` subcommand, its all-to-all encoder, and the
  `n = 16` codes of `known_codes`, with seed-level permutation tests.
  `tests/test_ion_chain.py` (2 items, skipped without torch).
- Multi-round memory objective (`memory_objective`; design note
  `docs/memory_objective.md`, runs in `experiments/memory_objective/REPORT.md`).
  A code is scored by the logical error rate per round of an `R`-round
  syndrome-extraction memory experiment instead of by its encoder's
  single-fault mass, which cannot see a local-Clifford deformation of the
  code (`experiments/noise_aware/runs/REPORT.md`). `build_memory_circuit` builds
  the circuit for arbitrary (non-CSS) generators: one ancilla per
  generator with controlled-X/Y/Z gates, hook-aware gate orders
  (`hook_orders`, the ion-search suffix and segment criterion, exact
  completion cost from the Knill-Laflamme kernel), generators that share a
  qubit with different letters in separate batches so the interleaved
  gates commute (X block then Z block for CSS codes), optional one-flag
  ancillas, noiseless `MPP` preparation and readout, detectors between
  consecutive rounds. Layouts: all-to-all (ion chain) and `square:RxC`
  with ancilla sites at the plaquette centers and nearest-neighbour CNOT
  ladders for non-local generators. Tiers: `proxy` (leading-order failure
  of the maximum-likelihood decoder over fault sets of weight at most two
  from the DEM, plus exact `d_circ` up to weight 4, `exact_dcirc` ported
  from `ion-search/lib.py`), `sampled` (exact ML decoder by dynamic
  program when `D + O <= 20`, else a weight-2 lookup decoder), `bposd`
  (the board's pinned BP+OSD, needs `ldpc`, validation only).
  `MemoryReward` is the environment reward (distance and padding terms of
  the noise-aware experiment plus `weight * log10(p_ref / p_round)` on
  target-distance states, cached per code) and exposes `success_fn`;
  `CodeDiscoveryEnv` adopts a reward's `success_fn` attribute when none is
  given. `experiments.memory_objective` (run as `python -m`): tier
  benchmark on [[5,1,3]], Steane, rotated surface, the board's
  [[16,4,4]], and a discovered non-CSS [[9,1,3]] under the chain, board,
  and Heron profiles; Spearman correlation of the proxy against bposd on
  32 random [[9,1,3]] codes; the XZZX bias-sensitivity sweep; the lattice
  routing comparison; a PPO smoke run on the 3x3 lattice under the ion
  profile. `tests/test_memory_objective.py` (14 items, the PPO smoke
  regression skipped without torch).
- PPO with the memory objective on the 4x4 setting
  (`experiments.memory_ppo`, run as `python -m`; report in
  `experiments/memory_ppo/REPORT.md`, runs in `experiments/memory_ppo/runs/`): the
  noise-aware 4x4 protocol of `gpu_scale` (equivariant policy, batch 1024,
  H and CNOT on `square:4x4`, 50-gate episodes, fractional distance term,
  padding and LDPC penalties) with the memory term of `MemoryReward`
  (`R = 4` rounds, all-to-all layout, proxy tier, both bases) in place of
  the encoder's undetected fault mass, on four cells: the chain preset at
  scale 0.25 with bare and with one-flag ancillas, the board recipe at
  0.25, and the Heron rates at 0.05 (the benchmark scales board 1 and
  Heron 0.1 put the `R = 4` circuits above one expected fault per shot,
  outside the tiers' regime). `TorchMemoryReward` evaluates the distance,
  fractional, and padding terms on the device through
  `gpu_scale.DistanceKernel` and the memory term on the host in a spawn
  process pool (`MemoryTermPool`, one evaluation per distinct stabilizer
  group, synchronous with the training loop). `score` evaluates the best
  code of every run, the nine encoder-objective 4x4 codes of
  `experiments/gpu_scale/runs`, and the rotated surface `[[16,1,4]]` with its XZZX
  deformation under the four profiles at the proxy and the bposd tier;
  `report` writes the tables and figures. Result of the 12-run matrix
  (RunPod A40, 150 min per run, host CPU capped at four cores): every run
  reaches `d = 4`, but under the validation decoder the memory-trained
  best codes are 8.9 to 50 times worse than the rotated surface code on
  their training profile and not distinguishable from the encoder-trained
  codes; no Hadamard frame or `d_X`/`d_Z` asymmetry beyond the encoder's
  own appears under the ion profile; the proxy tier underestimates the
  discovered codes 5 to 20 times (sampled tier agrees with bposd) because
  their memory circuits carry 0.9 to 1.5 expected faults per shot.
  Run 2 (`--variant run2`, `experiments/memory_ppo/runs/run2/`, second section of
  the report): no LDPC term, selection by the training `p_round`, a
  gate-count term above 1.25 times the surface code's two-qubit gates per
  round, the sampled tier for candidates above 0.8 expected faults per
  shot, and every code evaluated on the sparsest basis of its stabilizer
  group (`sparsest_code`; the memory circuit measures the generators it is
  given, and the encoder tableau's basis is denser than the sparsest one,
  which inflated every run-1 discovered-code number 3 to 10 times). The
  run-2 selected codes are 0.8 to 3.1 times the surface code under bposd on
  their training profile (median 1.8) against 1.8 to 5.3 (median 2.6) for
  the run-1 codes re-scored the same way; one board seed is at parity.
  `tests/test_memory_ppo.py` (5 items, skipped without torch).
- `memory_objective._joint_ids`: the multi-word path (more than 64
  detectors, every `n = 16` memory circuit) labels the pair signatures by
  a lexsort of the word columns instead of `np.unique(axis=0)` (whose
  structured-view argsort took 14 of the 19 s of one proxy evaluation of
  a dense `[[16,1,4]]` code), and shifts the labels when the all-zero
  signature is absent, since the maximum-likelihood grouping tests
  `id == 0` for the zero syndrome (before, the smallest nonzero signature
  took that role). Test added.

## 0.1.0 (2026-09-20)

First cut.

- Batched Aaronson-Gottesman tableau (`tableau`) with H, S, S_DAG, CNOT, CZ,
  SWAP, and Pauli frames applied to selected batch elements with per-element
  qubit indices; stim conversion and random-circuit agreement tests.
- Symplectic and GF(2) helpers (`symplectic`): logical operators by
  symplectic Gram-Schmidt, tableau completion from stabilizers.
- Pauli error sets by weight and by channel probability (`errors`).
- Knill-Laflamme reward of Olle et al., batched exact-distance reward,
  low-weight-stabilizer penalty, composite rewards (`reward`).
- `StabilizerCode` with exact distance for `n <= 20`, randomized upper bound
  with witness for larger `n`, CSS test, weight enumerators (`code`);
  reference codes [[5,1,3]], Steane, Shor, rotated surface [[9,1,3]],
  [[4,2,2]] (`known_codes`).
- Connectivity graphs: all-to-all, linear, square, triangular, heavy-hex
  (`graphs`).
- Gymnasium-style batched environment (`env`), no gymnasium dependency.
- Circuit-level noise hook with single-fault propagation through stim and a
  placeholder depolarizing model (`circuit_noise`).
- Random and evolutionary search agents (`search`); rediscovery regression
  for [[5,1,3]], Steane, and the rotated surface code with results JSON
  (`regress`, `autoqec-regress` CLI).
