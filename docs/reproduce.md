# Reproducing the rediscovery regressions

Two regressions rediscover the [[5,1,3]], Steane [[7,1,3]], and rotated
surface [[9,1,3]] codes (plus the LDPC-constrained and exclusion targets for
the evolutionary search): `autoqec-regress` (evolutionary search over gate
sequences) and `autoqec-ppo` (PPO). Each has a `reduced` budget, run by the
test suite and by `scripts/reproduce.sh`, and a `full` budget, whose results
ship in `results/regression.json` and `results/regression_ppo.json` and are
tabulated in README.md. Every run records seed, budget, platform, numpy and
torch versions, and thread count in its JSON.

Search is randomized but seeded: the same seed, batch size, numpy version,
and thread count reproduce the same circuits and evaluation counts. Across
numpy versions or BLAS libraries the random streams are identical (numpy
`default_rng`), but PPO's torch arithmetic can differ in the last bits and
change a trajectory; expect the PPO counts to move, not the outcome.

## Clean environment

```
uv venv --python 3.12 .venv-repro && source .venv-repro/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.14.0"   # CPU torch
uv pip install "autoqec-sim[rl] @ git+https://github.com/unitaryfoundation/autoqec-sim@v0.2.0"  # or ".[rl]" in a checkout
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
```

`scripts/reproduce.sh` does the same in a temporary venv, runs the tests and
both reduced regressions, and prints the summary below.

## Reduced budget

Evolutionary: batch 256, 15,000 evaluations per attempt, at most 3 attempts
per target, 1 seed for the exclusion targets, no enumerator requirement.
PPO: batch 256, 400,000 state evaluations or 90 s per attempt, 1 attempt,
stop at the first verified nontrivial `[[n,1,3]]` code.

```
autoqec-regress --budget reduced --seed 0 --out regression_reduced.json
autoqec-ppo     --budget reduced --seed 0 --out regression_ppo_reduced.json --compare results/regression.json
```

Expected (seed 0, macOS arm64, Python 3.12, numpy 2.5, torch 2.14, 6
threads; the machine was under other load, so wall times are upper ends):

| Target | Agent | Attempts | Evaluations | Wall time | Found | Enumerators match |
|---|---|---|---|---|---|---|
| [[5,1,3]] | evolutionary | 2 | 19,456 circuits | 1.8 s | d = 3, w* = 4 | yes |
| Steane | evolutionary | 1 | 12,544 circuits | 7.4 s | d = 3, w* = 5 | no (not required) |
| rotated surface | evolutionary | 1 | 1,536 circuits | 0.6 s | d = 3, w* = 5 | no (not required) |
| rotated surface, LDPC mask w_max 4 | evolutionary | 1 | 4,352 circuits | 2.6 s | d = 3, w* = 4 | no (not required) |
| rotated surface, LDPC penalty w_max 4 | evolutionary | 1 | 14,848 circuits | 17.2 s | d = 3, w* = 4 | no (not required) |
| Steane, LDPC penalty w_max 4 | evolutionary | 2 | 21,760 circuits | 16.0 s | d = 3, w* = 4 | yes |
| 7 qubits, LDPC penalty w_max 3 | evolutionary | 1 | 15,104 circuits | 5.2 s | none (expected) | n/a |
| 5 qubits, LDPC penalty w_max 3 | evolutionary | 1 | 15,104 circuits | 2.1 s | none (expected) | n/a |
| [[5,1,3]] | PPO | 1 | 61,440 state evaluations (4,352 episodes) | 2.3 s | d = 3 | yes |
| Steane | PPO | 1 | 81,920 (4,096 episodes) | 4.3 s | d = 3 | no (not required) |
| rotated surface | PPO | 1 | 40,960 (1,280 episodes) | 1.6 s | d = 3 | no (not required) |
| rotated surface, LDPC mask w_max 4 | PPO | 1 | 40,960 (1,280 episodes) | 1.6 s | d = 3 | no (not required) |
| rotated surface, LDPC penalty w_max 4 | PPO | 1 | 94,208 (3,072 episodes) | 6.2 s | d = 3 | no (not required) |
| Steane, LDPC penalty w_max 4 | PPO | 1 | 192,512 (9,472 episodes) | 12.3 s | d = 3 | yes |
| 7 qubits, LDPC penalty w_max 3 | PPO | 1 | 401,408 (19,968 episodes) | 17.2 s | none (expected) | n/a |
| 5 qubits, LDPC penalty w_max 3 | PPO | 1 | 401,408 (28,672 episodes) | 13.7 s | none (expected) | n/a |

Both CLIs exit 0 on this run. `scripts/reproduce.sh` took 201 s in total
on 2026-09-20: 2 s to build and install (warm uv cache; a cold cache adds
the torch download), 85 s for the 78 tests, 53 s for the evolutionary
regression, and 60 s for the PPO regression. The PPO exclusion targets run
to their 400,000-step cap, which is where most of the PPO time goes.

Pass criteria, as in `tests/test_regress.py` and `tests/test_ppo.py`: every
`expect_success` target ends with `passed` and `rediscovered` true, a
verified exact distance 3, and (for LDPC targets) `w* <= w_max` and
`max_qubit_degree <= deg_max`; the two exclusion targets end with `passed`
true and `rediscovered` false; the [[5,1,3]] enumerators match the reference
(the code is unique). The reduced budget does not require enumerator
matches for Steane or the surface code.

## Full budget

Evolutionary: batch 512, 100,000 evaluations per attempt, up to 60 attempts
(seeds 0, 1, ...), restart until the enumerators of `S` and `N(S)` match the
reference; exclusion targets run 5 seeds to exhaustion. PPO: batch 256,
3,000,000 state evaluations or 20 minutes per attempt, up to 3 attempts,
stop at the first enumerator match; the PPO-only 4x4 target
`rotated_surface_16` has its own budget of 1,000,000 state evaluations in
one attempt (`ppo_regress.TARGET_BUDGETS`) and is skipped by the reduced
budget. Run it alone with `autoqec-ppo --budget full --targets
rotated_surface_16 --seed 0 --threads 2 --out results/regression_ppo_4x4_seed0.json`
(about 20 minutes at 2 threads).

```
autoqec-regress --budget full --seed 0 --out results/regression.json
autoqec-ppo     --budget full --seed 0 --out results/regression_ppo.json --compare results/regression.json
```

Expected, from the shipped JSONs (seed 0, macOS arm64, Python 3.12.14,
numpy 2.5.3, torch 2.14.0, 6 threads):

| Target | Agent | Attempts (seeds) | Evaluations | Wall time | Found | Enumerators match |
|---|---|---|---|---|---|---|
| [[5,1,3]] | evolutionary | 1 | 13,824 circuits | 1.1 s | d = 3, 14 gates, w* = 4 | yes |
| Steane | evolutionary | 3 (0 to 2) | 54,272 circuits | 29.8 s | d = 3, 19 gates, w* = 4 | yes |
| rotated surface | evolutionary | 21 (0 to 20) | 60,416 circuits | 20.9 s | d = 3, 22 gates, w* = 6 | yes |
| rotated surface, LDPC mask w_max 4 | evolutionary | 10 (0 to 9) | 78,848 circuits | 44.3 s | d = 3, 28 gates, w* = 4 | yes |
| rotated surface, LDPC penalty w_max 4 | evolutionary | 1 | 10,752 circuits | 11.3 s | d = 3, 30 gates, w* = 4 | no (not required) |
| Steane, LDPC penalty w_max 4 | evolutionary | 1 | 67,584 circuits | 47.4 s | d = 3, 18 gates, w* = 4 | yes |
| 7 qubits, LDPC penalty w_max 3 | evolutionary | 5 (0 to 4) | 501,760 circuits | 154.9 s | none (expected) | n/a |
| 5 qubits, LDPC penalty w_max 3 | evolutionary | 5 (0 to 4) | 501,760 circuits | 62.2 s | none (expected) | n/a |
| [[5,1,3]] | PPO | 1 | 61,440 state evaluations | 1.4 s | d = 3 | yes |
| Steane | PPO | 1 | 225,280 state evaluations | 10.7 s | d = 3 | yes |
| rotated surface | PPO | 1 | 106,496 state evaluations | 3.4 s | d = 3, 15 gates | yes |
| rotated surface 4x4 `[[16,1,4]]`, seed 0 | PPO | 1 | 1,003,520 state evaluations | 1,203 s | d = 4, 46 gates, w* = 5 (first of 66 verified codes) | no |
| rotated surface 4x4 `[[16,1,4]]`, seed 1 | PPO | 1 | 1,003,520 state evaluations | 1,237 s | d = 4, 48 gates, w* = 6 (first of 2 verified codes) | no |

The whole evolutionary full run takes about 6 minutes and the PPO full run
under 20 s on a laptop CPU with 6 threads for the eight shared targets; the
4x4 target adds about 20 minutes per seed at 2 threads
(`results/regression_ppo_4x4_seed{0,1}.json`, run on a shared machine).
The 4x4 rows end without an enumerator match: both seeds reach a verified
`[[16, 1, 4]]` code, none of the 68 distinct codes is the rotated surface
code (README, "PPO agent"). One evolutionary "circuit" costs
up to `max gates` state evaluations, so the evolutionary counts are upper
bounds on the same axis as PPO's (README.md, "PPO agent").

## Reading the JSON

`targets.<name>.result.verification` holds `n`, `k`, `distance_exact`,
`parameters_match`, `nontrivial`, `is_css`, the weight enumerators of `S`
and `N(S)`, `enumerators_match_reference`, `min_max_generator_weight`
(`w*`), `max_qubit_degree`, and `ldpc_bound_met`; `result.gates` is the
encoding circuit as `[gate, q1, q2]` triples (`q2 = -1` for single-qubit
gates), replayable with `autoqec_sim.regress.replay(gates, n)`. Attempts
that found a verified code with other enumerators are under
`targets.<name>.attempts`.
