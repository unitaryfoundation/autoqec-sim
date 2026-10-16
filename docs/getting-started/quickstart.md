# Quick Start

This page rediscovers the [[5,1,3]] code with the evolutionary search,
verifies it independently of the reward, and scores its encoding circuit
under a device noise profile. The whole script is `examples/quickstart.py`
and runs in about two seconds.

## Search

A `CodeDiscoveryEnv` holds `batch_size` encoding circuits as one array of
stabilizer tableaux on `n` qubits encoding `k` logical qubits. Each step
applies one gate from `gates` to every batch element. The regression's
target for the [[5,1,3]] code fixes the gate set (H, S, CNOT on five
all-to-all qubits), the gate cap, and the reward: the Knill-Laflamme term
over all weight-1 and weight-2 Paulis plus a penalty on weight-1
stabilizers.

```python
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.regress import TARGETS, make_reward
from autoqec_sim.search import EvolutionarySearch

target = TARGETS["five_qubit"]
env = CodeDiscoveryEnv(n=5, k=1, batch_size=512, gates=target.gates, max_steps=target.max_steps,
                       reward_fn=make_reward(target), seed=0)
res = EvolutionarySearch(env, seed=0).run(max_evaluations=30_000)
print(f"search: success={res.success} evaluations={res.evaluations} gates={len(res.gates)} wall={res.wall_time_s:.1f} s")
```

## Verify

Replay the found gate sequence on a fresh tableau and read the code off it.
`StabilizerCode` computes the exact distance and the weight enumerators of
the stabilizer group and its normalizer, which are compared with the
reference code. This check does not use the reward.

```python
import numpy as np

from autoqec_sim import BatchTableau, StabilizerCode, known_codes

tab = BatchTableau(5, 1)
for gate, a, b in res.gates:
    tab.apply(gate, a, b if b >= 0 else None)
code = StabilizerCode.from_tableau(tab.array, k=1)
ref = known_codes.get("five_qubit")
print(f"code: n={code.n} k={code.k} d={code.distance_exact()} css={code.is_css()}")
match = all(np.array_equal(a, b) for a, b in zip(code.weight_enumerators(), ref.weight_enumerators(), strict=True))
print("enumerators match [[5,1,3]]:", match)
```

The code has distance 3, is not CSS, and its enumerators match the
reference.

## Score under device noise

`circuit_level_reward` lays the gate list out as a TICK-layered circuit,
applies a noise profile, propagates every single-fault mechanism to the
output with stim, and returns minus the undetected mass. The IonQ Forte
preset at a quarter of its measured rates is one of the
[device presets](../noise_presets.md).

```python
from autoqec_sim.circuit_noise import circuit_level_reward
from autoqec_sim.noise import presets

r = circuit_level_reward(tab.array, 1, [res.gates], presets.forte(), scale=0.25)
print(f"circuit-level reward under forte @ 0.25: {r[0]:.4f}")
```

## Noisify a circuit of your own

The noise module also works on a plain `stim.Circuit`. A skeleton in the
TICK-layer convention is noisified under a preset, and the detector error
model gives the mechanism count and the expected number of faults. The
script is `examples/noise_quickstart.py`.

```python
import stim

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets

skeleton = stim.Circuit("""
    R 0 1 2 3 4
    TICK
    CX 0 1 2 3
    TICK
    CX 2 1 4 3
    TICK
    M 1 3
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    TICK
    M 0 2 4
    OBSERVABLE_INCLUDE(0) rec[-3]
""")

noisy = nm.apply(skeleton, presets.forte(), scale=0.25)
dem = nm.dem(noisy)
print("mechanisms:", nm.mechanism_count(dem), "expected faults:", round(nm.fault_expectation(dem), 5))

profile = presets.heron(graph=nm.path(5)).with_bias(10.0).with_timing(nm.partial(1))
print(nm.summarize(profile, scale=1.0))
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
