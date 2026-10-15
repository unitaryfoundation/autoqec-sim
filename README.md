# autoqec-sim

Vectorized stabilizer-code reward engine, RL environment, and known-code
rediscovery regression for the AutoQEC Genesis project (Task 1, milestone
M1). CPU-only numpy implementation; the array layout and kernels are chosen
so that NVIDIA can replace them with cuQuantum/cuStabilizer kernels without
changing the surrounding code.

Version 0.2.0 is the M1 public release of the AutoQEC project (Task 1);
`noise_model` (repository `noise-model`) is its one non-PyPI dependency.
`ORG` in the URLs below is the GitHub organization, to be fixed at release
(RELEASE.md).

## What exists

| Module | Contents |
|---|---|
| `autoqec_sim.tableau` | Batched Aaronson-Gottesman tableau, one dense `uint8` array of shape `(batch, 2n, 2n+1)`. Gates H, S, S_DAG, CNOT, CZ, SWAP, and Pauli frames X, Y, Z, each applied to selected batch elements with per-element qubit indices through gather/scatter indexing (no Python loop over the batch). `ActionTable` resolves a vector of discrete actions with one pass per gate type. Conversion to and from `stim.Tableau` for cross-checks. |
| `autoqec_sim.symplectic` | GF(2) elimination, kernel, solve; symplectic product as a float32 GEMM plus parity; logical-operator extraction by symplectic Gram-Schmidt on N(S)/S; completion of a stabilizer set to a full tableau. |
| `autoqec_sim.errors` | Enumeration of all Paulis up to a weight, probability weighting under per-qubit Pauli channels, top-M dominant errors. |
| `autoqec_sim.reward` | Knill-Laflamme reward of Olle et al. (arXiv:2311.04750, Eq. 10), `r = -sum_mu lam_mu K_mu`, vectorized over the batch and chunked over the error set. Detection is decided from the `n+k` code rows of the tableau: an error is undetected iff it commutes with every stabilizer and anticommutes with some logical (degenerate errors in S count as detected, no linear algebra needed). Also a batched exact-distance reward, a low-weight-stabilizer penalty (excludes codes padded with idle qubits), and a composite reward. |
| `autoqec_sim.code` | `StabilizerCode` read off an encoding tableau or built from Pauli strings (general non-CSS, symplectic form). Exact distance for `n <= 20` by enumerating N(S) or low-weight Paulis, whichever is cheaper; randomized information-set upper bound with a witness for larger `n` (the symplectic version of `qldpc-challenge/research/kit/surrogate.py`); CSS test; weight enumerators of S and N(S). |
| `autoqec_sim.known_codes` | [[5,1,3]], Steane [[7,1,3]], Shor [[9,1,3]], rotated surface [[9,1,3]], [[4,2,2]]. |
| `autoqec_sim.graphs` | all-to-all, linear chain, square, triangular, heavy-hex builders and a `kind:size` spec parser. Every builder delegates to `noise_model.topology`; `from_topology`, `Graph.to_topology`, and `as_graph` convert between the environment view (`n`, sorted edge tuple) and the noise module's graph, so the action space and the noise profile's placement check share one edge set. |
| `autoqec_sim.ldpc` | LDPC sparsity of a stabilizer group, batched: exact minimum bottleneck generator weight `w*` and a sparsest basis with its maximum qubit degree (bit-packed GF(2) elimination over the enumerated group, `n - k <= 14`), greedy fallback for larger codes, feasibility test `ldpc_feasible`, reward penalty `ldpc_penalty`. |
| `autoqec_sim.env` | `CodeDiscoveryEnv`: gymnasium-style `reset`/`step`/`observation` without a gymnasium dependency. State is the batched tableau; actions are single-qubit gates on every qubit and two-qubit gates on the edges of a connectivity graph (either graph class; `env.topology` is the noise-module view); observation is the tableau bits, the `(n-k) x 2n` check matrix, or a flat vector; reward modes dense (Olle et al.), delta, sparse; per-element auto-reset; `rollout` of fixed action sequences; optional hard LDPC constraint (`ldpc_w_max`, modes `terminate`, `mask`, `final`) and `feasible_action_mask()`. |
| `autoqec_sim.circuit_noise` | Circuit-level reward under a `noise_model.Profile` (board, ion, Heron presets and their variants), a preset name, or any `stim.Circuit -> stim.Circuit` callable (the placeholder depolarizing model is kept). The gate list is laid out as a TICK-layered skeleton so the profile's timing and idle model apply; every fault mechanism of the noisy circuit is propagated with stim through the rest of the encoding circuit and scored with the Knill-Laflamme kernel; `dem_undetected_mass` cross-checks the result against stim's detector error model. |
| `autoqec_sim.search` | Random search and an evolutionary search over gate sequences, one batched rollout per generation. |
| `autoqec_sim.regress` | The rediscovery regression and the `autoqec-regress` CLI. |
| `autoqec_sim.agents.ppo` | PPO agent (PyTorch, CPU): clipped objective, GAE, shared value head, entropy bonus, action masking, over the batched environment. torch is an optional dependency (`rl` extra) and is imported lazily. |
| `autoqec_sim.agents.ppo_regress` | The rediscovery regression run with PPO instead of evolutionary search, reusing the targets, reward, replay, and verification of `regress`. |

Not included: Clifft integration, cuQuantum kernels, and noise-aware PPO
training (in progress, not part of 0.2.0).

## Mapping to Task 1 and M1

The proposal's M1 deliverable is "GPU-accelerated vectorized Clifford
simulator extended to non-CSS codes and circuit-level noise; RL agent
reproduces Steane and surface-code discoveries; cuQuantum integration;
public GitHub release."

- Vectorized Clifford simulator: `tableau` (batched, CPU, numpy). The
  Knill-Laflamme reward is the reward signal of Olle et al.
- Non-CSS: the whole stack works in the symplectic representation; the
  action set includes S, so non-CSS codes are reachable, and the distance
  and enumerator tools do not assume CSS.
- Circuit-level noise: `circuit_noise` scores every single-fault
  mechanism of the encoding circuit under a `genesis/noise-model` profile
  (see "Noise profiles" below).
- LDPC sparsity (Task 1 item (ii)): `ldpc` and the environment constraint
  (see "LDPC constraint" below).
- Steane and surface-code rediscovery: `regress` (evolutionary search) and
  `agents.ppo_regress` (PPO), see below.
- Public release: this version (0.2.0), with the tagged noise-model
  0.2.0 it depends on.
- cuQuantum integration: not done (see "What is left for cuQuantum"); the
  kernel list there is the hand-off to NVIDIA.

## Noise profiles

`circuit_noise.resolve_noise_model` accepts a `noise_model.Profile`
(`presets.forte()`, `presets.chain()`, `presets.board()`,
`presets.heron()`, `presets.depolarizing(p)`, or any `with_bias`,
`with_timing`, `with_params` variant), a preset name, or a bare
`stim.Circuit -> stim.Circuit` callable. The encoding gate list is laid out
as a TICK-layered skeleton (as-soon-as-possible scheduling) so the
profile's timing model and idle channel apply; every noise instruction the
profile emits becomes a fault mechanism, propagated through the rest of
the circuit with stim and scored with the Knill-Laflamme kernel as
`-(undetected mass)`, normalized by the total mass. A profile without a
graph takes the environment's graph; a profile whose graph differs is
refused, and a two-qubit gate off the graph raises `ConnectivityError` (or
scores `violation_reward` with `on_violation="penalize"`).
`dem_undetected_mass` cross-checks the result against stim's detector error
model to `O(p^2)`. Use through `make_circuit_level_reward(env, profile)` as
a terminal reward (`reward_mode="sparse"`); it is per circuit, not batched.

## LDPC constraint

`ldpc.sparsest_basis_stats` computes, for each batch element, the exact
minimum bottleneck generator weight `w*` (the smallest `w` such that the
stabilizer elements of weight at most `w` generate the group) and the
maximum qubit degree of a sparsest basis, by bit-packed GF(2) elimination
over the enumerated group for `n - k <= 14` (about 4 ms at `n = 9`), with a
greedy pairwise reduction as the bound for larger codes. The environment
takes `ldpc_w_max` with three modes: `terminate` (episode ends on a
violating state with `ldpc_violation_reward`), `mask` (the violating
action reverts to a no-op; `feasible_action_mask()` gives agents the mask
to apply to logits), and `final` (only success is gated). `ldpc_penalty`
is the soft form for the reward. `w_max = 3` provably excludes every
5-qubit distance-3 code and empirically every 7-qubit one the search
reaches; `w_max = 4` admits [[5,1,3]] and Steane and excludes Shor's code
(`w* = 6`).

## Install

Python 3.12 to 3.14. `noise-model` is not on PyPI; install it from its tag
first, then this package (the `rl` extra adds CPU torch for the PPO agent).

```
pip install "noise-model @ git+https://github.com/ORG/noise-model@v0.2.0"
pip install "autoqec-sim[rl] @ git+https://github.com/ORG/autoqec-sim@v0.2.0"
```

With uv, in a project:

```
uv add "noise-model @ git+https://github.com/ORG/noise-model@v0.2.0"
uv add "autoqec-sim[rl] @ git+https://github.com/ORG/autoqec-sim@v0.2.0"
```

From checkouts, for development. `pyproject.toml` resolves `noise_model`
from the sibling directory `../noise-model` (`[tool.uv.sources]`, editable)
so the two evolve together; the built wheel carries the version pin
`noise-model>=0.2.0,<0.3` instead. `uv sync --no-sources` switches a
checkout to the pin.

```
git clone https://github.com/ORG/noise-model
git clone https://github.com/ORG/autoqec-sim && cd autoqec-sim
uv sync --group dev --extra rl      # plain `uv sync --group dev` skips torch and the PPO tests
uv run pytest
```

The suite (78 tests, including the reduced-budget regression for both
agents) runs in about 70 s on a laptop CPU. Without the `rl` extra the six
PPO tests are skipped. Thread use is capped at six through
`OPENBLAS_NUM_THREADS` and friends in `tests/conftest.py` and in the
regression CLIs. `scripts/reproduce.sh` does all of this from a fresh venv
and also runs both reduced regressions; `docs/reproduce.md` has the
expected outputs.

## Quick start

`examples/quickstart.py`: rediscover the [[5,1,3]] code, verify it
independently of the reward, and score its encoding circuit under a
circuit-level noise profile. About 2 s.

```python
import numpy as np

from autoqec_sim import BatchTableau, StabilizerCode, known_codes
from autoqec_sim.circuit_noise import circuit_level_reward
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.regress import TARGETS, make_reward
from autoqec_sim.search import EvolutionarySearch
from noise_model import presets

# Batched environment: 512 parallel episodes on 5 all-to-all qubits with
# H, S, and CNOT, at most 14 gates; the reward is the Knill-Laflamme term
# over all weight-1 and weight-2 Paulis plus the weight-1 stabilizer penalty.
target = TARGETS["five_qubit"]
env = CodeDiscoveryEnv(n=5, k=1, batch_size=512, gates=target.gates, max_steps=target.max_steps,
                       reward_fn=make_reward(target), seed=0)
res = EvolutionarySearch(env, seed=0).run(max_evaluations=30_000)
print(f"search: success={res.success} evaluations={res.evaluations} gates={len(res.gates)} wall={res.wall_time_s:.1f} s")

# Replay the found gate sequence on a fresh tableau and read the code off it.
tab = BatchTableau(5, 1)
for gate, a, b in res.gates:
    tab.apply(gate, a, b if b >= 0 else None)
code = StabilizerCode.from_tableau(tab.array, k=1)
ref = known_codes.get("five_qubit")
print(f"code: n={code.n} k={code.k} d={code.distance_exact()} css={code.is_css()}")
match = all(np.array_equal(a, b) for a, b in zip(code.weight_enumerators(), ref.weight_enumerators(), strict=True))
print("enumerators match [[5,1,3]]:", match)

# Circuit-level score: every single-fault mechanism of the encoding circuit
# under the IonQ Forte profile, propagated with stim, as -undetected mass.
r = circuit_level_reward(tab.array, 1, [res.gates], presets.forte(), scale=0.25)
print(f"circuit-level reward under forte @ 0.25: {r[0]:.4f}")
```

Output on seed 0: `search: success=True evaluations=13824 gates=14 wall=1.4 s`,
`code: n=5 k=1 d=3 css=False`, `enumerators match [[5,1,3]]: True`, and
`circuit-level reward under forte @ 0.25: -0.0599` (about 6 percent of the
single-fault mass of that 14-gate encoder goes undetected under the Forte
rates at a quarter scale).

## Rediscovery regression

```
uv run autoqec-regress --budget reduced          # what the pytest runs
uv run autoqec-regress --budget full --out results/regression.json
```

Targets (all `k = 1`, `d = 3`):

| Target | Graph | Gates | Max gates | Forbidden stabilizer weights |
|---|---|---|---|---|
| [[5,1,3]] | all-to-all on 5 | H, S, CNOT | 14 | 1 |
| Steane [[7,1,3]] | all-to-all on 7 | H, S, CNOT | 20 | 1 to 3 |
| rotated surface [[9,1,3]] | 3x3 square lattice | H, CNOT | 30 | 1 |
| rotated surface, LDPC `w_max = 4`, `deg_max = 4`, env mask | 3x3 square lattice | H, CNOT | 30 | 1 |
| rotated surface, LDPC `w_max = 4`, `deg_max = 4`, penalty 0.25 | 3x3 square lattice | H, CNOT | 30 | 1 |
| Steane, LDPC `w_max = 4`, penalty 0.25 | all-to-all on 7 | H, S, CNOT | 20 | 1 to 3 |
| 7 qubits, LDPC `w_max = 3`, penalty 0.25, must fail | all-to-all on 7 | H, S, CNOT | 20 | 1 |
| 5 qubits, LDPC `w_max = 3`, penalty 0.25, must fail | all-to-all on 5 | H, S, CNOT | 14 | 1 |

The two exclusion targets pass when no verified distance-3 code is found
within budget. On 5 qubits this is a theorem: the [[5,1,3]] code is the
unique 5-qubit distance-3 code and all 15 of its nontrivial stabilizers
have weight 4, so `w* = 4` and no `w_max = 3` solution exists. On 7
qubits it is empirical: Steane and the second [[7,1,3]] family the
search finds both have `w* = 4`, and no [[7,1,3]] code generated by
weight-3 elements was found in the recorded budget; the contrast target
`steane_ldpc_w4` finds Steane in the same budget. `w_max = 4` does not
exclude the [[5,1,3]] code or Steane (both have `w* = 4`), which is why
the bound for the exclusion targets is 3; on 9 qubits `w_max = 4` does
exclude Shor's code (`w* = 6`). Exclusion targets forbid only weight-1
stabilizers so weight-2 and weight-3 generators stay admissible, and use
the penalty form so that the search path is unconstrained and only the
final code is judged. Every verification reports
`min_max_generator_weight` (`w*`) and `max_qubit_degree`.

Reward during search: normalized Knill-Laflamme reward over all Paulis of
weight 1 and 2, plus the penalty on stabilizer elements of the forbidden
weights (without it the agent pads a [[5,1,3]] code with idle qubits and
still satisfies the [[7,1,3]] and [[9,1,3]] parameters). A success is
verified independently of the reward: the circuit is replayed, the code is
extracted, its distance is computed exactly, and its weight enumerators of
S and N(S) are compared with the reference code. Equal enumerators are
necessary for equivalence under qubit permutations and local Cliffords;
they are reported as `enumerators_match_reference`, not as a proof of
isomorphism. The `full` budget restarts with the next seed until the
enumerators match; the `reduced` budget stops at the first verified
nontrivial `[[n,1,3]]` code and reports whether it matched.

Results of the `full` run are in `results/regression.json` (seeds, budgets,
evaluations, wall time, the found circuits and stabilizers). Summary of the
release run (seed 0, batch 512, at most 100,000 circuit evaluations per
attempt, up to 60 attempts; exclusion targets run 5 seeds to exhaustion):

| Target | Gates | Attempts (seeds) | Circuit evaluations | Wall time | Found code | w* / degree | Enumerators match |
|---|---|---|---|---|---|---|---|
| [[5,1,3]] | H, S, CNOT | 1 (seed 0) | 13,824 | 1.1 s | d = 3, 14 gates, non-CSS | 4 / 4 | yes |
| Steane [[7,1,3]] | H, S, CNOT | 3 (seeds 0 to 2) | 54,272 | 29.8 s | d = 3, 19 gates, non-CSS as generated | 4 / 4 | yes |
| rotated surface [[9,1,3]] | H, CNOT | 21 (seeds 0 to 20) | 60,416 | 20.9 s | d = 3, 22 gates, non-CSS as generated | 6 / 4 | yes |
| rotated surface, LDPC mask `w_max = 4` | H, CNOT | 10 (seeds 0 to 9) | 78,848 | 44.3 s | d = 3, 28 gates, non-CSS as generated | 4 / 4 | yes |
| rotated surface, LDPC penalty `w_max = 4` | H, CNOT | 1 (seed 0) | 10,752 | 11.3 s | d = 3, 30 gates, non-CSS | 4 / 4 | no (not required) |
| Steane, LDPC penalty `w_max = 4` | H, S, CNOT | 1 (seed 0) | 67,584 | 47.4 s | d = 3, 18 gates, non-CSS as generated | 4 / 5 | yes |
| 7 qubits, LDPC penalty `w_max = 3` | H, S, CNOT | 5 (seeds 0 to 4) | 501,760 | 154.9 s | none (expected); best reward -3/210 on every seed | n/a | n/a |
| 5 qubits, LDPC penalty `w_max = 3` | H, S, CNOT | 5 (seeds 0 to 4) | 501,760 | 62.2 s | none (expected); best reward -3/105 on every seed | n/a | n/a |

The `w*` column is the exact minimum bottleneck generator weight of the
found code: the unconstrained surface-code run returned a code with the
surface enumerators but `w* = 6`, that is, a code whose sparsest
generating set has a weight-6 element, so equal enumerators do not
certify local-Clifford equivalence to the CSS reference (a local Clifford
preserves weights, so an equivalent code would have `w* = 4`; the
enumerators of `S` and `N(S)` coincide without equivalence here). The
mask-constrained run matched the enumerators with `w* = 4` in 10
attempts against 21 unconstrained. The penalty-constrained surface target
accepts any verified distance-3 code within the bound: in a first full
run it produced one on all 60 attempts (31 distinct enumerator classes,
none the surface code's; kept in the JSON under `first_run_attempts`),
so the surface code is one of many weight-4 [[9,1,3]] codes reachable
on the lattice and a stop-at-first search rarely lands on it. On the
exclusion targets the best reward stayed at three undetected weight-2
errors on every seed, that is, the search kept codes within `w_max = 3`
that miss distance 3 by three errors rather than taking the 0.25 penalty
of a weight-4 solution.

"Non-CSS as generated" means the found generators mix X and Z; the
enumerators equal those of the CSS reference, consistent with a local
Clifford equivalence, which the check does not prove. Attempts that ended
in a verified nontrivial [[n,1,3]] code with different enumerators are kept
in the JSON under `attempts`. The reduced budget (batch 256, 15,000
evaluations per attempt, at most 3 attempts, 1 seed for exclusion
targets, no enumerator requirement) finishes all eight targets in about
60 s and is what `tests/test_regress.py` runs.

Hit rates from tallies of independent seeds (batch 512, 60,000
evaluations per run, every run produced a verified nontrivial [[n,1,3]]
code): on 7 qubits all-to-all the search finds two families, with
stabilizer enumerators `[1,0,0,0,21,0,42,0]` (Steane) and
`[1,0,0,0,13,24,18,8]`; Steane appeared in 35 of 80 runs (H, S, CNOT and
H, CNOT gate sets behave alike). On the 3x3 lattice the [[9,1,3]] codes are
far more varied: the rotated-surface enumerators appeared in 3 of 60 runs
with H, CNOT and in 0 of 30 runs with H, S, CNOT, which is why the surface
target uses H, CNOT and the full budget allows up to 60 attempts.

## PPO agent

`agents/ppo.py` is a PPO implementation (Schulman et al., arXiv:1707.06347)
over `CodeDiscoveryEnv`, in the spirit of Olle et al. (arXiv:2311.04750): a
policy over gate choices conditioned on the current tableau, trained on the
Knill-Laflamme reward. The batch dimension of the environment is the
vectorized environment (256 parallel episodes by default), `auto_reset` is
on, and the reward is the dense mode (state reward after every gate). An
episode ends when the state reward reaches 0 (every term of the reward is
satisfied) or after the target's gate cap.

Network and hyperparameters (`PPOConfig` defaults):

| Item | Value |
|---|---|
| Observation | the `2n x 2n` X and Z bits of the whole tableau (destabilizers and stabilizers, sign column dropped), mapped to `{-1, +1}`; `check_matrix` (only the `(n - k) x 2n` stabilizer block, as in Olle et al.) is an option |
| Network | MLP trunk, two tanh layers of 256 units, orthogonal init; linear policy head over the action table and linear value head sharing the trunk. 99,615 parameters for [[5,1,3]] (100 inputs, 30 actions), 130,873 for Steane (196 inputs, 56 actions), 157,730 for the 3x3 surface target (324 inputs, 33 actions) |
| Rollout | 16 steps of 256 environments, so 4,096 transitions per update |
| Optimization | Adam, learning rate 3e-4, 4 epochs, minibatch 512, gradient norm clipped at 0.5 |
| PPO | clip 0.2, gamma 0.98, GAE lambda 0.95, value coefficient 0.5, entropy coefficient 0.01, advantages normalized per update |
| Masking | the action table already contains only the gates the graph allows, so no graph masking is needed there; `graph_action_mask` builds a static mask when a table has actions outside a graph; a dynamic mask forbids undoing the previous gate when it is an involution (H, CNOT, CZ, SWAP, Paulis); an `action_mask()` method on the environment, if present, is intersected with these. Masked actions get logit `-1e9` |
| Threads | CPU only, torch and BLAS capped at 6 threads |

Run on one target, or the regression:

```
uv run --extra rl python -m autoqec_sim.agents.ppo --n 5 --max-steps 14 --verbose
uv run --extra rl autoqec-ppo --budget reduced
uv run --extra rl autoqec-ppo --budget full --out results/regression_ppo.json
```

(`autoqec-ppo` is the console script for `autoqec_sim.agents.ppo_regress`.)

`ppo_regress` uses the same targets, reward (Knill-Laflamme plus the
low-weight-stabilizer penalty), replay, and verification (exact distance,
enumerators of `S` and `N(S)` against the reference) as `regress`. The
difference in protocol: a PPO run does not stop at its first success; every
distinct code the policy produces (deduplicated by the reduced row echelon
form of its stabilizer block) is verified, and the `full` budget stops at the
first code whose enumerators match, moving to the next seed only if a run
exhausts 3,000,000 state evaluations or 20 minutes. The `reduced` budget
(400,000 state evaluations, 90 s, one seed) stops at the first verified
nontrivial `[[n,1,3]]` code and is what `tests/test_ppo.py` runs on the
[[5,1,3]] target (1.5 s).

Results of the `full` run, seed 0, in `results/regression_ppo.json`,
against the evolutionary search's `results/regression.json` on the same
targets. Circuits are episodes for PPO and evaluated sequences for the
evolutionary search; state evaluations are reward calls per batch element
(one PPO environment step of the batch is 256 of them; one evolutionary
circuit costs up to `max gates` of them, so its count is an upper bound).

| Target | Agent | Seeds | Circuits | State evaluations | Wall time | First d = 3 code (circuits) | Enumerators match |
|---|---|---|---|---|---|---|---|
| [[5,1,3]] | evolutionary | 1 | 13,824 | about 193,000 | 1.1 s | 13,824 | yes |
| [[5,1,3]] | PPO | 1 | 4,352 | 61,440 | 1.4 s | 4,352 | yes |
| Steane [[7,1,3]] | evolutionary | 3 | 54,272 | about 1,085,000 | 28.9 s | 22,528 | yes |
| Steane [[7,1,3]] | PPO | 1 | 11,264 | 225,280 | 10.7 s | 3,841 | yes |
| rotated surface [[9,1,3]] | evolutionary | 21 | 60,416 | about 1,812,000 | 20.5 s | 1,024 | yes |
| rotated surface [[9,1,3]] | PPO | 1 | 3,333 | 106,496 | 3.4 s | 1,025 | yes |

On the fair axis (state evaluations) PPO needs 3 times fewer for
[[5,1,3]], 5 times fewer for Steane, and 17 times fewer for the surface
code; per state evaluation it is slower (network forward and backward on
top of the reward kernel), so the wall-time gap is smaller. The Steane run
saw the same two 7-qubit families as the evolutionary tallies
(`[1,0,0,0,13,24,18,8]` five times before Steane once). The surface run
produced 11 distinct enumerator families among 13 verified codes before the
rotated-surface enumerators appeared, at a per-episode success rate of 60
percent by update 26; with the evolutionary search that family appeared in
about one run in twenty. The matched surface circuit has 15 gates against 22 for the
evolutionary one. Seeds 1 to 3 at the full budget also matched every target
on the first attempt: 53,000 to 66,000 state evaluations for [[5,1,3]],
119,000 to 160,000 for Steane, and 61,000 to 176,000 for the surface code
(3 to 12 s per target). The evolutionary column is one run from one seed
sequence, not an average.

Missing for parity with Olle et al.:

- Noise-aware training. The agent takes any `reward_fn`, so a
  channel-weighted error set (`ErrorSet.from_channel`) or the circuit-level
  fault reward of `circuit_noise` plugs in; the runs are in progress and
  not part of 0.2.0.
- Larger `n` and `k > 1`. The MLP input grows as `4 n^2` and the all-to-all
  action space as `O(n^2)`; there is no permutation-equivariant or
  attention policy, no curriculum over error weight, and the reward cost
  grows as `3^w C(n, w)`. Only `k = 1`, `n <= 9` has been exercised.
- Training conveniences: no learning-rate annealing, no checkpointing or
  policy export, no evaluation of a trained policy on a different graph
  (the static graph mask exists and is unit-tested but has not been used
  for transfer).
- The undo mask is a heuristic of this implementation, not part of Olle et
  al.

## What is left for cuQuantum

The state is one array `tab` of shape `(batch, 2n, 2n+1)` with dtype
`uint8`. Everything a GPU backend has to provide:

1. Gate kernels (`tableau.apply_h`, `apply_s`, `apply_s_dag`, `apply_cnot`,
   `apply_cz`, `apply_swap`, `apply_x/y/z`): each reads two or four columns
   of the selected batch elements (`tab[b, :, q]`, a gather along axis 2
   with a per-element column index) and writes them back after elementwise
   XOR/AND. `ActionTable.apply` groups a vector of actions by gate type and
   calls these once per type; with a fused kernel the grouping disappears.
2. The symplectic product `symplectic.symplectic_product(E, rows)`: a
   batched GEMM of the shared error set `(M, 2n)` against the per-element
   code rows `(batch, n+k, 2n)` followed by a parity reduction. This is the
   whole cost of `reward.undetected_mask`, `in_stabilizer_mask`,
   `batched_distance`, and hence of every reward. Bit-packed rows and a
   popcount-based product would replace float32.
3. `reward.code_rows`: a slice-and-concatenate of the tableau; on device
   it is a view.

Not on the hot path and fine to leave on the host: GF(2) elimination in
`symplectic`, exact distance and enumerators in `code`, stim conversion,
the circuit-level fault propagation (per circuit, uses stim), and the
search agent's population update.

## Layout

```
src/autoqec_sim/         package
src/autoqec_sim/agents/  PPO agent and its regression
tests/                   pytest suite (tableau vs stim, codes, rewards, env, noise hook, LDPC, regression, PPO)
results/                 regression.json (evolutionary) and regression_ppo.json (PPO), full budget
examples/quickstart.py   the quick start above
scripts/reproduce.sh     fresh venv, tests, both reduced regressions
docs/reproduce.md        commands and expected outputs for both regressions
```

## Funding

This material is based upon work supported by the U.S. Department of
Energy, Office of Science, Office of Advanced Scientific Computing
Research, as part of the Genesis Mission National Science and Technology
Challenges under Award Number DE-SC0025336.

## Citation and license

Versions are in CHANGELOG.md. To cite the package use CITATION.cff (the
GitHub "Cite this repository" button reads it); a Zenodo DOI is minted at
the first tagged release (RELEASE.md). The reward and the training setup
follow Olle et al., arXiv:2311.04750, which is the reference to cite for
the method. Licensed under Apache-2.0 (LICENSE).
