# Evolutionary search against PPO on the three-topology matrix

Same objective ([[n, 1, 3]] with no idle qubit, H and CNOT on the lattice edges, 50-gate episodes), same environment-step budget per tier. Evolutionary search spends its budget as whole episodes (evaluations x 50 steps); PPO as environment steps. Success = a distance-3 code with no idle qubit within the budget.

| lattice | n | PPO successes | PPO median steps to success | EA successes | EA median steps to success | EA median wall |
|---|---|---|---|---|---|---|
| square | 9 | 3/3 | 9,472 | 3/3 | 12,800 | 0.1 s |
| triangular | 9 | 3/3 | 7,424 | 3/3 | 12,800 | 0.2 s |
| heavy_hex | 12 | 3/3 | 63,488 | 3/3 | 217,600 | 3.6 s |
| square | 16 | 3/3 | 112,128 | 3/3 | 281,600 | 6.5 s |
| triangular | 16 | 3/3 | 75,776 | 3/3 | 179,200 | 3.6 s |
| heavy_hex | 18 | 3/3 | 340,480 | 2/3 | 2,086,400 | 64 s |

Totals: PPO 18 of 18 runs succeed, evolutionary search 17 of 18. The evolutionary search is autoqec-sim's own (population of gate sequences, mutation, one batched rollout per generation), not a tuned evolutionary agent; the comparison fixes the budget, objective, and action space, nothing else.
