# Modules

| Module | Contents |
|---|---|
| `autoqec_sim.tableau` | Batched Aaronson-Gottesman tableau [@aaronson2004], one dense `uint8` array of shape `(batch, 2n, 2n+1)`. Gates H, S, S_DAG, CNOT, CZ, SWAP, and Pauli frames X, Y, Z, each applied to selected batch elements with per-element qubit indices through gather/scatter indexing (no Python loop over the batch). `ActionTable` resolves a vector of discrete actions with one pass per gate type. Conversion to and from `stim.Tableau` [@gidney2021] for cross-checks. |
| `autoqec_sim.symplectic` | GF(2) elimination, kernel, solve; symplectic product as a float32 GEMM plus parity; logical-operator extraction by symplectic Gram-Schmidt [@wilde2009] on N(S)/S; completion of a stabilizer set to a full tableau. |
| `autoqec_sim.errors` | Enumeration of all Paulis up to a weight, probability weighting under per-qubit Pauli channels, top-M dominant errors. |
| `autoqec_sim.reward` | Knill-Laflamme [@knill1997] reward of [@olle2024], `r = -sum_mu lam_mu K_mu`, vectorized over the batch and chunked over the error set. Detection is decided from the `n+k` code rows of the tableau: an error is undetected iff it commutes with every stabilizer and anticommutes with some logical (degenerate errors in S count as detected, no linear algebra needed). Also a batched exact-distance reward, a low-weight-stabilizer penalty (excludes codes padded with idle qubits), and a composite reward. |
| `autoqec_sim.code` | `StabilizerCode` read off an encoding tableau or built from Pauli strings (general non-CSS, symplectic form). Exact distance for `n <= 20` by enumerating N(S) or low-weight Paulis, whichever is cheaper; randomized information-set upper bound with a witness for larger `n` (the symplectic version of `research/kit/surrogate.py` in [@qldpcchallenge]); CSS test [@calderbank1996; @steane1996]; weight enumerators of S and N(S). |
| `autoqec_sim.known_codes` | [[5,1,3]] [@laflamme1996; @bennett1996], Steane [[7,1,3]] [@steane1996], Shor [[9,1,3]] [@shor1995], rotated surface [@tomita2014] [[9,1,3]] and [[16,1,4]] (`rotated_surface(d)` builds any distance), [[4,2,2]]. |
| `autoqec_sim.graphs` | all-to-all, linear chain, square, triangular, heavy-hex [@chamberland2020] builders and a `kind:size` spec parser. Every builder delegates to `autoqec_sim.noise.topology`; `from_topology`, `Graph.to_topology`, and `as_graph` convert between the environment view (`n`, sorted edge tuple) and the noise module's graph, so the action space and the noise profile's placement check share one edge set. |
| `autoqec_sim.ldpc` | LDPC sparsity of a stabilizer group, batched: exact minimum bottleneck generator weight `w*` and a sparsest basis with its maximum qubit degree (bit-packed GF(2) elimination over the enumerated group, `n - k <= 14`), greedy fallback for larger codes, feasibility test `ldpc_feasible`, reward penalty `ldpc_penalty`, per-action mask `greedy_action_mask` (greedy bound, or the exact feasibility test after every two-qubit gate for `n - k <= 15`). `LdpcObjective`: reward and success criterion of an LDPC-constrained `[[n, k, d]]` target in one of four forms (`penalty`, `unnormalized`, `final`, `schedule`), exact `w*` on candidate states only; see "LDPC constraint". |
| `autoqec_sim.env` | `CodeDiscoveryEnv`: gymnasium-style `reset`/`step`/`observation` without a gymnasium dependency. State is the batched tableau; actions are single-qubit gates on every qubit and two-qubit gates on the edges of a connectivity graph (either graph class; `env.topology` is the noise-module view); observation is the tableau bits, the `(n-k) x 2n` check matrix, or a flat vector; reward modes dense [@olle2024], delta, sparse; per-element auto-reset; `rollout` of fixed action sequences; optional hard LDPC constraint (`ldpc_w_max`, modes `terminate`, `mask`, `final`) and `feasible_action_mask()`. |
| `autoqec_sim.circuit_noise` | Circuit-level reward under a `autoqec_sim.noise.Profile` (board, ion, Heron presets and their variants), a preset name, or any `stim.Circuit -> stim.Circuit` callable (the placeholder depolarizing model is kept). The gate list is laid out as a TICK-layered skeleton so the profile's timing and idle model apply; every fault mechanism of the noisy circuit is propagated through the rest of the encoding circuit (one GF(2) product against the symplectic map of the circuit suffix; `fault_set_reference` is the stim `PauliString.after` version, about twenty times slower, kept as the cross-check) and scored with the Knill-Laflamme kernel; `dem_undetected_mass` cross-checks the result against stim's detector error model. |
| `autoqec_sim.memory_objective` | Multi-round memory objective: an `R`-round syndrome-extraction circuit for arbitrary (non-CSS) stabilizer generators with one ancilla per generator, hook-aware gate orders, commutation-respecting batches, optional one-flag ancillas, all-to-all or square-lattice layouts with routed long-range gates; the `autoqec_sim.noise` profile with its timing model, stim's detector error model, and four fidelity tiers for the logical error rate per round (closed-form low-weight maximum-likelihood proxy with exact `d_circ`, exact ML or lookup-decoder sampling, pinned BP+OSD [@roffe2020; @panteleev2021] sampling, Tesseract [@beni2025] sampling); `undetectable_sets` lists the minimum-weight undetectable fault sets by probability, and `undetectable_logicals` enumerates every irreducible one up to weight 6 by a cluster search, which carries the proxy to circuit distance 5 and 6. `MemoryReward` is the environment reward; see docs/experiments/index.md and `docs/memory_objective.md`. |
| `autoqec_sim.evaluate` | `evaluate_code(code, profile)`: one call for the logical error per round of a chosen tier, the exact circuit-level distance and its fault mass per basis, and the most probable undetectable fault sets explained to circuit locations with stim's `explain_detector_error_model_errors`; returns a frozen `CodeEvaluation`. See `docs/evaluate.md`. |
| `autoqec_sim.search` | Random search and an evolutionary search over gate sequences, one batched rollout per generation. |
| `autoqec_sim.regress` | The rediscovery regression and the `autoqec-regress` CLI. |
| `autoqec_sim.backends.torch_backend` | Opt-in torch versions of the gate kernels, `ActionTable`, the symplectic product, and the parity masks (`undetected_mask`, `in_stabilizer_mask`, `batched_distance`, the rewards) on `cpu` or `cuda` tensors, same names and semantics as the numpy functions, bit-for-bit agreement tested. The mod-2 reduction and both commutation tests run as one in-place `remainder_(2)` and one matrix-vector product, without the int64 copy and the boolean reductions of the numpy path. `numpy_reward_adapter` plugs a torch reward into the environment. See `docs/gpu.md`. |
| `autoqec_sim.agents.ppo` | PPO [@schulman2017] agent (PyTorch, CPU): clipped objective, GAE [@schulman2015gae], shared value head, entropy bonus, action masking, over the batched environment. torch is an optional dependency (`rl` extra) and is imported lazily. |
| `autoqec_sim.agents.ppo_regress` | The rediscovery regression run with PPO instead of evolutionary search, reusing the targets, reward, replay, and verification of `regress`, plus the PPO-only 4x4 rotated-surface target `[[16, 1, 4]]` (1M env steps, `full` budget only). |

Not included: Clifft integration and cuQuantum kernels.

## Experiments

The experiments are not part of the installed package. Each lives in
`experiments/<name>/run.py` in the repository, next to its `REPORT.md` and
`figures/`, and runs from the repository root as
`uv run python -m experiments.<name>.run <subcommand>` (`--extra rl` for the
PPO-based ones); `experiments/common/figstyle.py` is the one matplotlib style
every figure shares. `experiments/README.md` has the layout and the
regeneration steps, and [Experiments](experiments/index.md) has one page per
experiment with its question, result, and commands.

| Experiment | Contents |
|---|---|
| `noise_aware` | The noise-aware discovery experiment (three noise profiles, three settings, five PPO seeds each, cross-evaluation, known-code baselines). |
| `scaling` | The RL scaling probe: diagnosis of the 4x4 failure, one-change-at-a-time variants of the agent on `[[16, 1, 4]]`, and LDPC-constrained `[[n, k, 3]]` targets at `n = 12` to `16`. |
| `ldpc_scaling` | The LDPC constraint forms above `n = 9` (`unnormalized`, `final`, `schedule` against `penalty`) on `[[12, 2, 3]]` and `[[16, 4, 3]]` with `w_max = 4`. |
| `ldpc_sparsity` | Sparsity as a property of the trained policy: the `shaped`, `terminal`, and `mask` variants against `unnormalized` at equal budget. |
| `gpu_scale` | The noise-aware 4x4 protocol on the torch backend (equivariant PPO, batch 1024, three profiles, cross-evaluation) and the 5x5 targets. |
| `ion_chain` | The `gpu_scale` protocol on all-to-all connectivity at `n = 16`, with and without S in the gate set. |
| `topologies` | The three-topology PPO run matrix on square, triangular, and heavy-hex lattices at two qubit-count tiers, with common metrics per cell. |
| `topologies_ea` | The package's evolutionary search on the same matrix and budget as `topologies`. |
| `heavy_hex_overnight` | The heavy-hex column of `topologies` widened to six seeds and extended to `[[18, 1, d >= 4]]`, driven unattended under a wall cap. |
| `memory_objective` | Benchmark of the memory objective's tiers on five codes and three profiles, rank correlation on random distance-3 codes, the XZZX bias-sensitivity test, the lattice-routing comparison, and the PPO smoke run. |
| `memory_ppo` | The 4x4 protocol with the memory objective as the noise term, a second run without the LDPC term, and `score` for the discovered and reference codes under every profile and tier. |
| `qdx_targets` | QDX's published all-to-all target set through this package's PPO at matched settings. |
| `sat_codes` | Fifteen SAT-designed biased-noise codes under the circuit-level memory objective against the rotated surface and XZZX codes. |
| `sat_population` | The same question over 2,150 SAT-sampled codes, with the structural features of each code's extraction circuit. |

## Layout

```
src/autoqec_sim/         package
src/autoqec_sim/backends/  torch backend of the gate kernels and parity masks (opt-in; docs/gpu.md)
src/autoqec_sim/agents/  PPO agent, its two policies (flat MLP, permutation-equivariant), and its regression
tests/                   pytest suite (tableau vs stim, codes, rewards, env, noise hook, LDPC, regression, PPO, torch backend vs numpy, the quick-start example)
experiments/<name>/      one directory per experiment: run.py (python -m experiments.<name>.run, not installed with the package), REPORT.md, figures/, launcher scripts, and runs/ (raw output, not tracked)
experiments/common/      the matplotlib style every experiment figure shares
experiments/tests/       the experiment tests, collected with the suite by uv run pytest
experiments/regression/  regression.json (evolutionary) and regression_ppo.json (PPO), full budget; regression_ppo_4x4_seed{0,1}.json (PPO, the 4x4 target)
experiments/regression/figure.py  the rediscovery-cost figure of the README and the regression page, from those files
benchmarks/profile_tableau.py  CPU throughput and memory profile of the tableau, reward, and environment (docs/profile.md, docs/gpu.md)
benchmarks/bench_cuda_kernels.py  the fused CUDA kernels against the torch path (docs/gpu.md)
tools/reproduce.sh       fresh venv, tests, both reduced regressions (docs/reproduce.md)
tools/release.sh         the release procedure (docs/development/releasing.md)
docs/examples/           the quick-start scripts, snippeted into the documentation and checked against the README
```
