<p align="center">
  <img src="docs/assets/logos/autoqec-sim.svg" alt="autoqec-sim" width="420">
</p>

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://unitaryfoundation.github.io/autoqec-sim/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://unitaryfoundation.github.io/autoqec-sim/stable/getting-started/installation/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

**autoqec-sim** evaluates stabilizer codes under device noise. One call
builds the syndrome-extraction memory experiment of any stabilizer code,
CSS or not, applies a noise model with sourced device parameters, and
returns the logical error per round, the exact circuit-level distance, and
the dominant fault sets explained back to gates and qubits, without
sampling or a decoder. Candidates can come from anywhere. The package also
ships three proposers, a SAT-population pipeline, an evolutionary search,
and a PPO agent, and the tools that verify what they find.

Developed with support from the U.S. Department of Energy Genesis Mission
and hosted by the [Unitary Foundation](https://unitary.foundation).

## What the evaluator does

`evaluate_code(code, profile)` takes stabilizer generators as Pauli strings
and a noise profile (IonQ Forte, IBM Heron, a long ion chain, the
qldpc-challenge recipe, or uniform depolarizing noise). It builds an
`R`-round memory circuit with one ancilla per generator in a hook-aware
gate order, derives stim's detector error model, and scores it at one of
four tiers: a closed-form maximum-likelihood proxy over low-weight fault
sets (no shots, no decoder), exact maximum likelihood or a lookup decoder,
BP+OSD, or Tesseract. The proxy is exact to leading order at circuit
distance up to 6 and agrees with the sampled decoders where they overlap:
Spearman 0.78 to 0.90 against BP+OSD over 32 random [[9,1,3]] codes, and
within the sampling error of 50,000 Tesseract shots on the 5x5 surface
code at five rounds (8 percent apart, where the shots leave Tesseract a 30
percent standard error). Beside the rate it returns
the exact circuit-level distance up to weight 6, the probability mass of
the minimum-weight undetectable fault sets, and the most probable of them
as circuit locations.

Its limits: the proxy is a leading-order estimate and holds while the
expected number of faults per shot stays below about one, where the
sampled tiers take over; the circuit-distance search stops at weight 6;
the memory circuit interleaves all checks into one block with one
measurement per round where a hook-safe schedule exists (every surface
code tried) and falls back to separate blocks with their own measurements
where none does (the Steane code); subsystem codes are measured through
their gauge generators, but flags beyond one per generator and adaptive
protocols are not expressible; and the presets are noise models assembled
from public specifications, not device measurements.

## What the proposers have found

The evaluator exists because two cheaper rankings fail. Over 2,150
SAT-designed nine-qubit codes, the code-capacity logical error rate has
rank correlation 0.00 to 0.21 with the circuit-level rate and none of the
twenty best codes by one score is among the twenty best by the other,
while the best codes of the population sit 30 to 40 percent below XZZX
under biased noise. The gate-by-gate search of Olle, Zen, Puviani, and
Marquardt
([npj Quantum Information 10, 126 (2024)](https://doi.org/10.1038/s41534-024-00920-y)),
reimplemented here with PPO and an evolutionary search, rediscovers
[[5,1,3]], Steane, Shor, and the rotated surface code, reaches distance 3
on square, triangular, and heavy-hex lattices at 9 to 18 qubits in 18 of
18 PPO runs and 17 of 18 evolutionary runs, matched 24 of the 45 published
QDX targets it ran within 10 minutes each, and found a [[25,1,5]] code on
the 5x5 lattice. It does not scale past about 16 qubits under a sparsity
constraint, and the SAT formulation of DalFavero et al.
([arXiv:2608.23460](https://arxiv.org/abs/2608.23460)) designs codes with
up to 100 qubits in minutes; such codes go into the evaluator as they are.
A schedule synthesizer assigns the CX layers of a fixed code's extraction
circuit against the exact circuit-level fault distance and raised that
distance from 2 to 3 for the qldpc-challenge [[49,13,4]] circuit. The
[experiments pages](https://unitaryfoundation.github.io/autoqec-sim/stable/experiments/)
hold each result with its figure and the commands that reproduce it.

## Why autoqec-sim?

- **One-call evaluation of any stabilizer code**: logical error per round,
  exact circuit-level distance, and dominant faults explained to gates, for
  CSS and non-CSS codes alike, without a matching decoder or shots.
- **Noise models with every parameter sourced**: IonQ Forte, IBM Heron, a
  long ion chain, and the qldpc-challenge recipe, with biased, correlated,
  and leakage-aware channels, device timing, and connectivity; the same
  models noisify a `stim.Circuit` of your own.
- **Verification independent of the objective**: exact distance, weight
  enumerators, and the sparsest generator basis for every code a run
  reports.
- **Three proposers**: a SAT-population pipeline over the codes of DalFavero
  et al., an evolutionary search, and PPO with action masking and a
  permutation-equivariant policy, on a batched tableau simulator with a
  bit-identical torch and CUDA path.
- **Schedule synthesis**: PPO over the CX-layer assignment of a
  syndrome-extraction circuit with the exact circuit-level fault distance
  as the reward.

## Installation

```bash
pip install git+https://github.com/unitaryfoundation/autoqec-sim
```

Python 3.12 to 3.14; the dependencies are `numpy` and `stim`. Tesseract
sampling needs the `tesseract` extra and the PPO agent needs torch:
install `"autoqec-sim[tesseract,rl] @ git+https://github.com/unitaryfoundation/autoqec-sim"`.

## Quick Start

Evaluate a code under device noise: the rotated surface code, three noisy
rounds of depolarizing noise at 1e-3, scored at the proxy tier, with the
most probable undetectable fault set as circuit errors.

```python
from autoqec_sim import evaluate_code
from autoqec_sim.noise import presets

# Rotated surface code [[9,1,3]], qubits on a 3x3 grid, row-major.
surface = [
    "XXIXXIIII", "IIIIXXIXX", "IXXIIIIII", "IIIIIIXXI",
    "IZZIZZIII", "IIIZZIZZI", "ZIIZIIIII", "IIIIIZIIZ",
]

# Three noisy rounds under uniform depolarizing noise at 1e-3, scored by
# the closed-form low-weight maximum-likelihood proxy (no decoder).
result = evaluate_code(surface, presets.depolarizing(1e-3), rounds=3)
print(f"error per round {result.p_round:.2e}  d_circ {result.d_circ}")

# The most probable minimum-weight undetectable fault set, as circuit errors.
top = result.dominant_faults[0]
print(f"{top.basis} basis, probability {top.probability:.1e}, {len(top.mechanisms)} mechanisms:")
for loc in top.locations:
    print(f"  {loc.pauli:7s} after {loc.gate} on {loc.qubits} at tick {loc.tick}  ->  {loc.targets}")
```

```
error per round 1.15e-03  d_circ {'Z': 3, 'X': 3}
Z basis, probability 3.6e-08, 3 mechanisms:
  X4      after DEPOLARIZE2 on (14, 4) at tick 4  ->  D12 D13
  X1      after DEPOLARIZE2 on (13, 1) at tick 2  ->  D12 L0
  X6      after DEPOLARIZE2 on (14, 6) at tick 3  ->  D13
```

Compare codes under a device preset: three known codes in a three-round
memory experiment under the IonQ Forte preset, at a quarter of its rates.

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
five_qubit (5, 1, 3) error per round 4.38e-03
steane (7, 1, 3) error per round 2.10e-03
rotated_surface_9 (9, 1, 3) error per round 5.96e-04
```

Propose a code: the evolutionary search on five qubits, and the code read
off the circuit it found.

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

All three run in a few seconds on a laptop. Continue with
[Evaluate a code](https://unitaryfoundation.github.io/autoqec-sim/stable/evaluate/)
for the result fields, the tiers, the validation, and a distance-5
example, the
[Quick Start](https://unitaryfoundation.github.io/autoqec-sim/stable/getting-started/quickstart/)
for verifying a found code against the reference, or
[Choose a Workflow](https://unitaryfoundation.github.io/autoqec-sim/stable/getting-started/choosing-a-workflow/)
to pick an entry point by the result you need. The console scripts are
listed under [Commands](https://unitaryfoundation.github.io/autoqec-sim/stable/reference/commands/).

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
