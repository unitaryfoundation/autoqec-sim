# Quick Start

## Score a code

`MemoryObjective` builds the syndrome-extraction circuit of any stabilizer
code, noisifies it under a device profile, and returns the logical error
per round from its detector error model. Here the three codes are known
ones; a code from any other tool goes in the same way through
`StabilizerCode.from_stabilizers`. The Forte preset runs at a quarter of
its measured rates, the scale the experiments use.

```python
--8<-- "docs/examples/quickstart.py:score"
```

```
five_qubit (5, 1, 3) error per round 4.27e-03
steane (7, 1, 3) error per round 2.10e-03
rotated_surface_9 (9, 1, 3) error per round 5.89e-04
```

## Search for a code

A `CodeDiscoveryEnv` holds `batch_size` encoding circuits on `n` qubits
encoding `k` logical qubits. Its default reward is the Knill-Laflamme
undetected mass over all Paulis below the target distance of 3, and its
default gate set is H, S, and CNOT on all-to-all connectivity. The
evolutionary search evolves gate sequences until one reaches the target
reward, and `StabilizerCode.from_gates` reads the code off the winning
circuit. This is `docs/examples/quickstart.py`; it runs in under a second.

```python
--8<-- "docs/examples/quickstart.py:search"
```

```
(5, 1, 3) ['-ZZIYX', '+IXXZX', '+XIYXX', '-ZXYIY']
```

A [[5,1,3]] code, the only distance-3 code on five qubits.

## Verify and score

The rest of this page rediscovers the [[5,1,3]] code with the
regression's target, verifies it independently of the reward, and scores
its encoding circuit under a device noise profile. The whole script is
`docs/examples/rediscover_and_score.py` and runs in about two seconds.

### Search

A `CodeDiscoveryEnv` holds `batch_size` encoding circuits as one array of
stabilizer tableaux on `n` qubits encoding `k` logical qubits. Each step
applies one gate from `gates` to every batch element. The regression's
target for the [[5,1,3]] code fixes the gate set (H, S, CNOT on five
all-to-all qubits), the gate cap, and the reward: the Knill-Laflamme term
over all weight-1 and weight-2 Paulis plus a penalty on weight-1
stabilizers.

```python
--8<-- "docs/examples/rediscover_and_score.py:search"
```

### Verify

Replay the found gate sequence on a fresh tableau and read the code off it.
`StabilizerCode` computes the exact distance and the weight enumerators of
the stabilizer group and its normalizer, which are compared with the
reference code. This check does not use the reward.

```python
--8<-- "docs/examples/rediscover_and_score.py:verify"
```

The code has distance 3, is not CSS, and its enumerators match the
reference.

### Score under device noise

`circuit_level_reward` lays the gate list out as a TICK-layered circuit,
applies a noise profile, propagates every single-fault mechanism to the
output with stim, and returns minus the undetected mass. The IonQ Forte
preset at a quarter of its measured rates is one of the
[device presets](../noise_presets.md).

```python
--8<-- "docs/examples/rediscover_and_score.py:score"
```

## Noisify a circuit of your own

The noise module also works on a plain `stim.Circuit`. A skeleton in the
TICK-layer convention is noisified under a preset, and the detector error
model gives the mechanism count and the expected number of faults. The
script is `docs/examples/noise_quickstart.py`.

```python
--8<-- "docs/examples/noise_quickstart.py:noise"
```

## Commands

```
autoqec-regress --budget reduced                           # rediscovery regression, evolutionary search
autoqec-ppo --budget reduced                               # the same targets with PPO (rl extra)
autoqec-schedule 49-13-4 --board /path/to/qldpc-challenge  # CX-layer schedule for a board code's circuit
```

## Next Steps

- [Choose a Workflow](choosing-a-workflow.md): pick the reward, agent, and entry point for the result you need.
- [Rediscovery regression](../regression.md): the targets, budgets, and verification behind the two commands.
- [Noise interface](../noise.md): profiles, timing models, topologies, and the circuit-level reward.
- [LDPC constraint](../ldpc.md): sparsity as a hard constraint, a mask, or a penalty.
- [GPU backend](../gpu.md): the same kernels on CUDA.
- [Python API](../api/autoqec_sim.md): the module reference.
