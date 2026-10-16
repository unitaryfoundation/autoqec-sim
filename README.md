# autoqec-sim

A batched stabilizer-code simulator with a reinforcement-learning
environment for discovering and evaluating small quantum error-correcting
codes. Thousands of encoding circuits are held as one array of stabilizer
tableaux and advanced together, one gate per step, on a chosen connectivity
graph. Rewards score the code each circuit encodes: the Knill-Laflamme
undetected-error mass, exact distance for small codes, LDPC sparsity,
circuit-level noise under a device profile, and a multi-round memory
experiment with a decoder. Two agents are included, PPO with an optional
lattice-equivariant policy and an evolutionary search. A schedule
synthesizer assigns the CX layers of a syndrome-extraction circuit to raise
its fault distance.

What it has done so far, for calibration: it rediscovers [[5,1,3]], Steane,
Shor, and the rotated surface code on every lattice tested; it has found
noise-adapted non-CSS codes up to 16 qubits; and the schedule synthesizer
raised the fault distance of the [[49,13,4]] circuit on the qldpc-challenge
board from 2 to 3. The RL discovery agent has not produced a new code above
16 qubits; `docs/experiments.md` records where it stops.

## Install

```
pip install "autoqec-sim[rl] @ git+https://github.com/unitaryfoundation/autoqec-sim@v0.2.0"
```

Python 3.12 to 3.14, `numpy`, and `stim`. The `rl` extra adds torch for
the PPO agent; without it, the evolutionary search and every reward still
run.

## Quick start

Rediscover the [[5,1,3]] code, verify it independently of the reward, and
score its encoding circuit under a device noise profile. About two seconds.

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

The full script is `examples/quickstart.py`.

## Commands

```
autoqec-regress --budget reduced                           # rediscovery regression, evolutionary search
autoqec-ppo --budget reduced                               # the same targets with PPO (rl extra)
autoqec-schedule 49-13-4 --board /path/to/qldpc-challenge  # CX-layer schedule for a board code's circuit
```

`autoqec-schedule` scores the board builder's own schedules, trains a PPO
policy on the exact circuit fault distance, and confirms the best schedule
at `rounds = d` with the board's search tier. Expected outputs for the two
regressions are in `docs/reproduce.md`.

## Modules

| Module | Contents |
|---|---|
| `tableau`, `symplectic` | batched Aaronson-Gottesman tableau; GF(2) and symplectic linear algebra |
| `reward`, `errors`, `code`, `known_codes` | Knill-Laflamme and distance rewards; Pauli error sets; `StabilizerCode` with exact and randomized distance; reference codes |
| `env`, `graphs`, `ldpc` | the discovery environment; connectivity graphs; LDPC sparsity constraint |
| `circuit_noise`, `memory_objective`, `noise` | circuit-level fault propagation; multi-round memory experiment; noise channels, timing, topologies, and device presets |
| `search`, `agents.ppo`, `regress` | evolutionary search; PPO; the rediscovery regression |
| `schedule_env`, `schedule_cli` | CX-layer schedule synthesis for a fixed code |
| `backends.torch_backend` | the gate kernels and rewards on CPU or CUDA tensors, bit-identical to numpy |

`docs/modules.md` has the full table.

## GPU

The numpy engine is the reference. `backends.torch_backend` runs the same
kernels and rewards on CUDA with agreement tests against numpy, and
`backends.cuda_kernels` adds two opt-in fused CUDA kernels for the gate
updates and the parity masks, five times and up to thirteen times the torch
path on an A100. `docs/gpu.md` has the measurements.

## Documentation

https://unitaryfoundation.github.io/autoqec-sim/ (MkDocs from `docs/`;
`uv run --group docs mkdocs serve` renders it locally). `stable/` is the
latest release and `dev/` tracks `main`.

## License

Apache-2.0. The reward and training setup follow Olle et al.,
arXiv:2311.04750. Supported by the U.S. Department of Energy, Office of
Science, Office of Advanced Scientific Computing Research, Genesis Mission,
award DE-SC0025336.
