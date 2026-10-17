# User Guide

autoqec-sim has two layers. The evaluator takes a stabilizer code and a
noise profile and returns the circuit-level logical error per round, the
exact circuit-level distance, and the dominant fault sets:
[Evaluate a code](evaluate.md) is the one-call entry point,
[Memory objective](memory_objective.md) the design note behind it, and
[Noise interface](noise.md) and [Device presets](noise_presets.md) the
noise models it applies. The proposers generate candidates for it. Codes
from a SAT sampler or from the literature go in as Pauli strings (the
[SAT population](experiments/sat_population.md) experiment is that
pipeline); the gate-by-gate search below builds encoding circuits one
Clifford gate at a time with an evolutionary search or a PPO agent. A
gate-by-gate run fixes four things: the search space, the reward, the
agent, and where the kernels run.

## 1. Define the search space

`CodeDiscoveryEnv(n, k, graph, batch_size, reward_fn, gates, max_steps, seed)`
holds `batch_size` encoding circuits on `n` qubits as one array of
stabilizer tableaux of shape `(batch, 2n, 2n+1)`. Actions are single-qubit
gates on every qubit and two-qubit gates on the edges of a connectivity
graph from `autoqec_sim.graphs` (all-to-all, chain, square, triangular,
heavy-hex [@chamberland2020], or a `kind:size` spec). Every builder delegates to
`autoqec_sim.noise.topology`, so the action space and a noise profile's
placement check share one edge set. Observations are the tableau bits, the
`(n-k) x 2n` check matrix, or a flat vector. With `auto_reset=True`,
finished episodes restart per element while the rest of the batch runs
on.

## 2. Choose the reward

| Reward | Page |
|---|---|
| Multi-round memory experiment, the evaluator's objective, four fidelity tiers | [Memory objective](memory_objective.md) |
| Circuit-level noise of the encoding circuit under a device profile | [Noise interface](noise.md), [Device presets](noise_presets.md) |
| Knill-Laflamme undetected mass [@knill1997] and exact distance, the default of the regression targets | [Rediscovery regression](regression.md) |
| LDPC sparsity, as a hard constraint (`terminate`, `mask`, `final`) or a penalty | [LDPC constraint](ldpc.md) |

`regress.make_reward(target)` composes the Knill-Laflamme term with the
low-weight-stabilizer penalty and, on LDPC targets, the sparsity term.
Any callable from the batched tableau to a reward vector works as
`reward_fn`. The encoder-based rewards cannot see a local-Clifford
deformation of a code, since a Hadamard at the end of the encoder only
relabels detected weight-1 faults; the memory objective can.

## 3. Choose the agent

`search.EvolutionarySearch` runs one batched rollout per generation on
numpy alone. `agents.ppo` is a clipped PPO [@schulman2017] with GAE [@schulman2015gae], a shared value head,
an entropy bonus, and action masking, on torch; `agents.policies` holds the
flat MLP and the permutation-equivariant actor-critic. `regress` and
`agents.ppo_regress` wrap both agents in the rediscovery regression, which
verifies every distinct code a run produces by exact distance and weight
enumerators.

[Experiments](experiments/index.md) records what each proposer has found:
the evolutionary search matches PPO on the three-topology matrix at a
fraction of the wall time, PPO is the more step-efficient of the two and
needs the permutation-equivariant policy from 12 qubits, and neither
scales past about 16 qubits under a sparsity constraint.

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
