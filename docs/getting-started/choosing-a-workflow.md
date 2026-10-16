# Choose a Workflow

The package separates the search space (qubits, gate set, connectivity),
the reward, the agent, and where the kernels run. Start from the result you
need.

| I want to... | Start with |
|---|---|
| Rediscover a known code and check the installation | `autoqec-regress --budget reduced`; [Rediscovery regression](../regression.md) |
| Train PPO on the same targets | `autoqec-ppo --budget reduced` (`rl` extra); [PPO agent](../regression.md#ppo-agent) |
| Search for a code under my own `n`, `k`, graph, and gate set | `CodeDiscoveryEnv` with `EvolutionarySearch` or `agents.ppo`; [User Guide](../guide.md) |
| Score an encoding circuit under a device noise model | `circuit_noise.circuit_level_reward` with a preset; [Noise interface](../noise.md), [Device presets](../noise_presets.md) |
| Score a code by a multi-round memory experiment with a decoder | `memory_objective.MemoryReward`; [Memory objective](../memory_objective.md) |
| Require sparse stabilizers | `ldpc_w_max` on the environment or `ldpc_penalty` in the reward; [LDPC constraint](../ldpc.md) |
| Noisify a `stim.Circuit` of my own | `autoqec_sim.noise.apply`; [Noise interface](../noise.md) |
| Assign CX layers of a syndrome-extraction circuit for a fixed code | `autoqec-schedule <slug> --board <path>`; [Commands](../reference/commands.md) |
| Run the reward engine on a GPU | `backends.torch_backend`, `backends.cuda_kernels`; [GPU backend](../gpu.md) |
| Reproduce the published tables | `scripts/reproduce.sh`; [Reproduce](../reproduce.md) |
| Read what the discovery agent has and has not found | [Experiments](../experiments.md) |

## Which agent

The evolutionary search needs only numpy, runs one batched rollout per
generation, and is the agent of `autoqec-regress`. PPO needs torch, uses
action masking (the connectivity graph, an undo mask, and any mask the
environment exposes), and matched the evolutionary search on every
rediscovery target with 3 to 17 times fewer state evaluations in the
full-budget regression. For lattices, the permutation-equivariant policy in
`agents.policies` is the one the experiments use.

## Which reward

| Reward | Module | Scores | Cost |
|---|---|---|---|
| Knill-Laflamme | `reward.kl_reward` | undetected mass over an enumerated Pauli error set | one GF(2) product per error set |
| Exact distance | `reward` and `code.StabilizerCode` | the minimum weight in the normalizer minus the stabilizer | enumeration, `n <= 20` |
| LDPC penalty or constraint | `ldpc` | minimum bottleneck generator weight `w*` and qubit degree | bit-packed elimination, `n - k <= 14` exact |
| Circuit-level noise | `circuit_noise` | undetected single-fault mass of the noisy encoder under a profile | one fault set per circuit, propagated with one GF(2) product |
| Memory objective | `memory_objective` | logical error rate per round of an `R`-round syndrome-extraction circuit | proxy, exact-ML or lookup sampling, or BP+OSD sampling |
