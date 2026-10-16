# User Guide

A discovery run fixes four things: the search space, the reward, the agent,
and where the kernels run. The pages in this section cover each one.

## 1. Define the search space

`CodeDiscoveryEnv(n, k, graph, batch_size, reward_fn, gates, max_steps, seed)`
holds `batch_size` encoding circuits on `n` qubits as one array of
stabilizer tableaux of shape `(batch, 2n, 2n+1)`. Actions are single-qubit
gates on every qubit and two-qubit gates on the edges of a connectivity
graph from `autoqec_sim.graphs` (all-to-all, chain, square, triangular,
heavy-hex, or a `kind:size` spec). Every builder delegates to
`autoqec_sim.noise.topology`, so the action space and a noise profile's
placement check share one edge set. Observations are the tableau bits, the
`(n-k) x 2n` check matrix, or a flat vector. With `auto_reset=True`,
finished episodes restart per element while the rest of the batch runs
on.

## 2. Choose the reward

| Reward | Page |
|---|---|
| Knill-Laflamme undetected mass and exact distance, the default of the regression targets | [Rediscovery regression](regression.md) |
| LDPC sparsity, as a hard constraint (`terminate`, `mask`, `final`) or a penalty | [LDPC constraint](ldpc.md) |
| Circuit-level noise of the encoding circuit under a device profile | [Noise interface](noise.md), [Device presets](noise_presets.md) |
| Multi-round memory experiment with a decoder, three fidelity tiers | [Memory objective](memory_objective.md) |

`regress.make_reward(target)` composes the Knill-Laflamme term with the
low-weight-stabilizer penalty and, on LDPC targets, the sparsity term.
Any callable from the batched tableau to a reward vector works as
`reward_fn`.

## 3. Choose the agent

`search.EvolutionarySearch` runs one batched rollout per generation on
numpy alone. `agents.ppo` is a clipped PPO with GAE, a shared value head,
an entropy bonus, and action masking, on torch; `agents.policies` holds the
flat MLP and the permutation-equivariant actor-critic. `regress` and
`agents.ppo_regress` wrap both agents in the rediscovery regression, which
verifies every distinct code a run produces by exact distance and weight
enumerators.

[Experiments](experiments.md) records the noise-aware discovery runs, the
memory-objective runs, the RL scaling probe, and the three-topology run
matrix, with what each found and where the agent stops.

## 4. Choose where the kernels run

The numpy engine on the CPU is the reference; [CPU profile](profile.md)
gives its throughput and where an environment step spends its time. The
[torch backend](gpu.md) runs the gate kernels, the parity masks, and the
rewards on `cpu` or `cuda` tensors with the same names and bit-for-bit
agreement tests, and two opt-in fused CUDA kernels replace the gate step
and the parity reduction.

## Reproduce

[Reproduce](reproduce.md) lists the commands and expected outputs of the
two rediscovery regressions at the reduced and full budgets.
