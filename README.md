<p align="center">
  <img src="docs/assets/logos/autoqec-sim.svg" alt="autoqec-sim" width="420">
</p>

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://unitaryfoundation.github.io/autoqec-sim/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://unitaryfoundation.github.io/autoqec-sim/stable/getting-started/installation/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

**autoqec-sim** evaluates stabilizer codes and their circuits under device
noise, and searches for codes, encoders, and syndrome-extraction schedules
that score well under it.

Developed with support from the U.S. Department of Energy Genesis Mission
and hosted by the [Unitary Foundation](https://unitary.foundation).

The evaluation layer scores a code three ways: the undetected single-fault
mass of its encoding circuit under a device noise profile (IonQ Forte, IBM
Heron, a long ion chain, or the qldpc-challenge recipe, every parameter
sourced), a multi-round memory experiment with a decoder, and the exact
distance and weight enumerators that verify any claim independently of the
objective. The search layer runs on a batched Aaronson-Gottesman tableau
that holds thousands of encoding circuits as one array and advances them
one Clifford gate per step on a chosen connectivity graph. PPO, with an
optional lattice-equivariant policy, and an evolutionary search build
encoders gate by gate; a schedule synthesizer assigns the CX layers of a
fixed code's syndrome-extraction circuit against the exact circuit-level
fault distance.

The gate-by-gate discovery mode follows Olle, Zen, Puviani, and Marquardt
([npj Quantum Information 10, 126 (2024)](https://doi.org/10.1038/s41534-024-00920-y)),
whose code, [QDX](https://github.com/jolle-ag/qdx), introduced it and has
reached larger codes than this package has. For code design at larger
sizes, the SAT formulation of DalFavero et al.
([arXiv:2608.23460](https://arxiv.org/abs/2608.23460)) designs codes with up
to 100 qubits; the evaluation and schedule tools here take such codes as
input.

What the package has produced: rediscovery of [[5,1,3]], Steane, Shor, and
the distance-3 rotated surface code on every lattice tested; a [[25,1,5]]
code on the 5x5 square lattice in 10 of 10 seeds (median 1.8M environment
steps and 263 s on one A40), none of them with the rotated surface code's
stabilizer group; distance-3 codes on square, triangular, and heavy-hex
lattices at 9 to 18 qubits in 18 of 18 PPO runs, where the package's own
evolutionary search found none in 18; noise-adapted non-CSS codes up to 16
qubits; and syndrome-extraction schedules that raised the circuit fault
distance of qldpc-challenge board circuits from 2 to 3 ([[49,13,4]], merged;
[[38,10,3]] and [[53,20,3]], filed). Under the memory objective, the codes
it trains have 0.8 to 3.1 times the rotated surface code's logical error
rate under BP+OSD on the training noise (median 1.8, one seed at parity),
and 7 of 9 beat the best code from the encoder objective.

What it has not produced: the rotated surface code's stabilizer group at
distance 4 or 5 (its [[16,1,4]] and [[25,1,5]] codes are non-CSS); a code
with every stabilizer generator of weight at most 4 beyond 12 qubits, since
the constrained action set stops being traversable at 16 qubits on every
graph tried; a code resolvably below the surface code under the memory
objective; and no new entry on the qldpc-challenge board. The
[experiments page](https://unitaryfoundation.github.io/autoqec-sim/stable/experiments/)
records each stopping point.

## Why autoqec-sim?

- **Device-level evaluation**: circuit-level noise of the encoder under
  sourced device presets (IonQ Forte, a long ion chain, IBM Heron, the
  qldpc-challenge recipe), cross-checked against Stim's detector error model,
  and a multi-round memory objective with three decoder tiers. Codes designed
  elsewhere, by SAT or by hand, can be scored here.
- **Verification independent of the objective**: exact distance, weight
  enumerators, and the sparsest generator basis for every code a run reports,
  plus a rediscovery regression that every change passes through.
- **Schedule synthesis**: PPO over the CX-layer assignment of a
  syndrome-extraction circuit with the exact circuit-level fault distance as
  the reward.
- **Gate-by-gate code and encoder search**, after Olle et al.: PPO with
  action masking and a permutation-equivariant policy, an evolutionary
  search, an LDPC sparsity constraint, and a batched tableau simulator on
  numpy with a bit-identical torch and CUDA path.

## Installation

```bash
pip install git+https://github.com/unitaryfoundation/autoqec-sim
```

Python 3.12 to 3.14; the dependencies are `numpy` and `stim`. The PPO agent
needs torch: install `"autoqec-sim[rl] @ git+https://github.com/unitaryfoundation/autoqec-sim"`.

## Quick Start

Score three known codes under a device noise profile: a three-round memory
experiment under the IonQ Forte preset, at the proxy tier.

```python
from autoqec_sim import known_codes
from autoqec_sim.memory_objective import MemoryObjective
from autoqec_sim.noise import presets

# A three-round memory experiment under the IonQ Forte preset at a quarter of
# its measured rates, scored by the low-weight maximum-likelihood proxy.
objective = MemoryObjective(presets.forte("sequential"), scale=0.25, rounds=3)
for name in ("five_qubit", "steane", "rotated_surface_9"):
    code = known_codes.get(name)
    print(name, code.parameters(), f"error per round {objective.evaluate(code).p_round:.2e}")
```

```
five_qubit (5, 1, 3) error per round 4.27e-03
steane (7, 1, 3) error per round 2.10e-03
rotated_surface_9 (9, 1, 3) error per round 5.89e-04
```

Search for a code on five qubits with the evolutionary agent, and read the
code off the circuit it found:

```python
from autoqec_sim import CodeDiscoveryEnv, EvolutionarySearch, StabilizerCode

env = CodeDiscoveryEnv(n=5, k=1, batch_size=512, seed=0)
result = EvolutionarySearch(env, seed=0).run(max_evaluations=30_000)

code = StabilizerCode.from_gates(5, 1, result.gates)
print(code.parameters(), code.stabilizer_strings())
```

```
(5, 1, 3) ['-ZZIYX', '+IXXZX', '+XIYXX', '-ZXYIY']
```

Both run in about a second on a laptop. Continue with the
[Quick Start](https://unitaryfoundation.github.io/autoqec-sim/stable/getting-started/quickstart/), which verifies a found code
against the reference and scores its encoder under a device profile, or use
[Choose a Workflow](https://unitaryfoundation.github.io/autoqec-sim/stable/getting-started/choosing-a-workflow/) to pick a
reward, agent, and entry point by the result you need. The console scripts
are listed under [Commands](https://unitaryfoundation.github.io/autoqec-sim/stable/reference/commands/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Participation is governed by the
[code of conduct](CODE_OF_CONDUCT.md).

## AI acknowledgment

Generative AI tools were used in parts of the software development and
writing for this project: code generation and review, implementation
analysis, and documentation editing. Design, validation, and release
decisions were made by the human author.

## Funding

This material is based upon work supported by the U.S. Department of
Energy, Office of Science, Office of Advanced Scientific Computing
Research, as part of the Genesis Mission National Science and Technology
Challenges under Award Number DE-SC0025336. This research used resources of
the National Energy Research Scientific Computing Center, a DOE Office of
Science User Facility supported by the Office of Science of the U.S.
Department of Energy under Contract No. DE-AC02-05CH11231.

## License

Apache-2.0
