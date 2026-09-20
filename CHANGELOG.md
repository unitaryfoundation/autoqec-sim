# Changelog

## 0.2.0 (TBD)

First tagged release. Three M1 items on top of 0.1.0: the shared noise
module in the circuit-level reward, an LDPC sparsity constraint, and a PPO
agent with its own rediscovery regression.

- `noise-model` (`genesis/noise-model`) is an editable path dependency
  (`[tool.uv.sources]`). `circuit_noise` accepts a `noise_model.Profile`,
  a preset name, or any `stim.Circuit -> stim.Circuit` callable
  (`resolve_noise_model`); profiles run through `noise_model.noisifier`
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
- `graphs` delegates every builder to `noise_model.topology`, with
  `from_topology`, `Graph.to_topology`, and `as_graph`; the environment
  accepts either graph class and exposes `env.topology`. A profile
  without a graph takes the environment's graph in `resolve_noise_model`
  and `make_circuit_level_reward`; a profile whose graph differs is
  refused. `heavy_hex(rows, cols)` is now the `noise_model` honeycomb
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
- Finding from the `w*` column of the regression: the unconstrained
  surface-code result of 0.1.0 (seed 20) has the surface code's weight
  enumerators of `S` and `N(S)` but `w* = 6` (its 26 elements of weight
  at most 4 span a rank-7 subgroup, the reference's span rank 8), so equal
  enumerators do not imply local-Clifford equivalence; the mask-constrained
  target's match has `w* = 4`.
- PPO agent (`agents.ppo`): clipped objective, GAE, shared value head,
  entropy bonus, action masking (static graph mask, dynamic undo mask, and
  any mask the environment exposes), PyTorch on CPU with a six-thread cap.
  torch is the optional `rl` extra and is imported lazily.
- PPO rediscovery regression (`agents.ppo_regress`, run as `python -m`):
  same targets, reward, and enumerator verification as `regress`; every
  distinct code a run produces is verified. Full-budget results in
  `results/regression_ppo.json`: [[5,1,3]], Steane, and the rotated surface
  code all matched on seed 0 with 3 to 17 times fewer state evaluations
  than the evolutionary search.
- `tests/test_ppo.py` (6 tests, skipped without torch), including a
  reduced-budget [[5,1,3]] rediscovery that trains PPO in about 1.5 s.
- `autoqec-ppo` console script for the PPO regression (`uv run --extra rl autoqec-ppo`).
- Tests: 44 to 71 for these modules (noise wiring, DEM cross-check, ion
  timing ranking, heavy-hex rejection, graph round trip, LDPC statistics,
  constraint modes, regression); 78 with the PPO agent's 6 and the
  version check.
- Packaging for release: `noise-model>=0.2.0,<0.3` as the published pin
  with the sibling checkout as the uv development source, SPDX license
  metadata, ruff configuration, CI and release workflows,
  `scripts/reproduce.sh`, `docs/reproduce.md`, CITATION.cff. The
  `autoqec-ppo` console script runs the PPO regression.

## 0.1.0 (2026-09-20)

First cut of the M1 substrate.

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
