---
hide:
  - navigation
title: autoqec-sim
---

# autoqec-sim { .visually-hidden }

<div class="hero" markdown>

:material-dots-hexagon:{ .hero-mark }

<p class="hero-title">autoqec-sim</p>

<p class="hero-tagline">
A batched stabilizer-code simulator and reinforcement-learning environment
for discovering small quantum error-correcting codes.
</p>

</div>

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![CI](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/unitaryfoundation/autoqec-sim/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](getting-started/installation.md)

---

## What is autoqec-sim?

autoqec-sim holds thousands of encoding circuits as one array of stabilizer
tableaux and advances them together, one Clifford gate per step, on a chosen
connectivity graph. A reward scores the code each circuit encodes. The
rewards are the Knill-Laflamme undetected-error mass, exact distance for
small codes, LDPC sparsity, circuit-level noise under a device profile, and
a multi-round memory experiment with a decoder.

Two agents search over gate sequences: PPO, with an optional
lattice-equivariant policy, and an evolutionary search. A rediscovery
regression verifies every code a run produces independently of the reward,
by exact distance and weight enumerators. A schedule synthesizer assigns the
CX layers of a syndrome-extraction circuit for a fixed code to raise its
fault distance.

The numpy engine on the CPU is the reference. The same kernels and rewards
run on CUDA through the torch backend, with two opt-in fused CUDA kernels
for the gate updates and the parity masks.

For calibration: the package rediscovers [[5,1,3]], Steane, Shor, and the
rotated surface code on every lattice tested, has found noise-adapted
non-CSS codes up to 16 qubits, and raised the fault distance of the
[[49,13,4]] circuit on the qldpc-challenge board from 2 to 3. The discovery
agent has not produced a new code above 16 qubits; [Experiments](experiments.md)
records where it stops.

## Quick Example

Install with
`pip install "autoqec-sim[rl] @ git+https://github.com/unitaryfoundation/autoqec-sim@v0.2.0"`,
then rediscover the [[5,1,3]] code, verify it, and score its encoding
circuit under a device noise profile. About two seconds.

```python
import numpy as np

from autoqec_sim import BatchTableau, StabilizerCode, known_codes
from autoqec_sim.circuit_noise import circuit_level_reward
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.noise import presets
from autoqec_sim.regress import TARGETS, make_reward
from autoqec_sim.search import EvolutionarySearch

target = TARGETS["five_qubit"]
env = CodeDiscoveryEnv(n=5, k=1, batch_size=512, gates=target.gates, max_steps=target.max_steps,
                       reward_fn=make_reward(target), seed=0)
res = EvolutionarySearch(env, seed=0).run(max_evaluations=30_000)

tab = BatchTableau(5, 1)
for gate, a, b in res.gates:
    tab.apply(gate, a, b if b >= 0 else None)
code = StabilizerCode.from_tableau(tab.array, k=1)
ref = known_codes.get("five_qubit")
print(code.distance_exact(), code.is_css(),
      all(np.array_equal(a, b) for a, b in zip(code.weight_enumerators(), ref.weight_enumerators(), strict=True)))

r = circuit_level_reward(tab.array, 1, [res.gates], presets.forte(), scale=0.25)
print(f"undetected single-fault mass under Forte noise at quarter scale: {-r[0]:.4f}")
```

## Key Features

<div class="grid cards" markdown>

- **Batched tableau simulator**

    One `(batch, 2n, 2n+1)` array of Aaronson-Gottesman tableaux. Gates
    apply to selected batch elements with per-element qubit indices through
    gather and scatter indexing, with no Python loop over the batch, and are
    cross-checked against stim.

- **Rewards for code quality**

    Knill-Laflamme undetected-error mass, exact distance, an
    [LDPC sparsity constraint](ldpc.md), circuit-level noise under
    [sourced device presets](noise_presets.md), and a
    [multi-round memory objective](memory_objective.md) with a decoder.

- **Two agents and a regression**

    PPO with action masking and a permutation-equivariant policy, and an
    evolutionary search. The [rediscovery regression](regression.md)
    verifies each found code by exact distance and weight enumerators.

- **CPU reference, GPU backend**

    The numpy engine is the reference. The [torch backend](gpu.md) runs the
    same kernels on CUDA with bit-for-bit agreement tests, and two fused
    CUDA kernels take the gate step and the parity masks further.

</div>

For a fixed code, `autoqec-schedule` trains a PPO policy on the exact
circuit fault distance of the syndrome-extraction circuit and confirms the
best CX-layer schedule at `rounds = d`. See [Commands](reference/commands.md).

## Get Started

[Quick Start](getting-started/quickstart.md){ .md-button .md-button--primary }
[Reproduce the regressions](reproduce.md){ .md-button }

## What's New in 0.2.0

Release 0.2.0 is the first tagged release. On top of 0.1.0 it adds the
[noise module](noise.md) as the `autoqec_sim.noise` subpackage, with biased,
correlated, and leakage-aware channels, device timing, topologies, and
[sourced presets](noise_presets.md) for IonQ Forte, a long ion chain, IBM
Heron, and the qldpc-challenge recipe. The circuit-level reward lays each
gate list out as a TICK-layered skeleton so a profile's timing and idle
model apply, and cross-checks its result against stim's detector error
model.

The [LDPC constraint](ldpc.md) computes the exact minimum bottleneck
generator weight per batch element and enters the environment as a hard
constraint or a reward penalty. The [PPO agent](regression.md#ppo-agent)
matches the evolutionary search on every rediscovery target with 3 to 17
times fewer state evaluations, and the [memory objective](memory_objective.md)
scores a multi-round syndrome-extraction circuit with three fidelity tiers.
The [torch backend](gpu.md) and two fused CUDA kernels run the reward
engine on a GPU.

[Full Changelog](development/changelog.md){ .md-button }
