<p align="center">
  <img src="docs/assets/logos/autoqec-sim-logo.svg" alt="autoqec-sim" width="420">
</p>

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://unitaryfoundation.github.io/autoqec-sim/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://unitaryfoundation.github.io/autoqec-sim/dev/getting-started/installation/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

**autoqec-sim** is a batched stabilizer-code simulator and
reinforcement-learning environment for discovering small quantum
error-correcting codes.

Developed with support from the U.S. Department of Energy Genesis Mission
and hosted by the [Unitary Foundation](https://unitary.foundation).

autoqec-sim holds thousands of encoding circuits as one array of stabilizer
tableaux and advances them together, one Clifford gate per step, on a chosen
connectivity graph. A reward scores the code each circuit encodes: the
Knill-Laflamme undetected-error mass, exact distance for small codes, LDPC
sparsity, circuit-level noise under a device profile, or a multi-round
memory experiment with a decoder. Two agents search over gate sequences, PPO
with an optional lattice-equivariant policy and an evolutionary search, and
a rediscovery regression verifies every code a run produces independently
of the reward.

So far it rediscovers [[5,1,3]], Steane, Shor, and the rotated surface code
on every lattice tested, has found noise-adapted non-CSS codes up to 16
qubits, and its schedule synthesizer raised the fault distance of the
[[49,13,4]] circuit on the qldpc-challenge board from 2 to 3. The discovery
agent has not produced a new code above 16 qubits; the
[experiments page](https://unitaryfoundation.github.io/autoqec-sim/dev/experiments/) records where it stops.

## Why autoqec-sim?

- **Batched tableau simulator**: one `(batch, 2n, 2n+1)` array of
  Aaronson-Gottesman tableaux, gates applied by gather and scatter indexing
  with no Python loop over the batch, cross-checked against stim.
- **Rewards for code quality**: Knill-Laflamme mass, exact distance, an LDPC
  sparsity constraint, circuit-level noise under sourced device presets
  (IonQ Forte, a long ion chain, IBM Heron, the qldpc-challenge recipe), and
  a multi-round memory objective with a decoder.
- **Two agents and a regression**: PPO with action masking and a
  permutation-equivariant policy, an evolutionary search, and a rediscovery
  regression that verifies each found code by exact distance and weight
  enumerators.
- **CPU reference, GPU backend**: the numpy engine is the reference; the
  torch backend runs the same kernels on CUDA with bit-for-bit agreement
  tests, and two fused CUDA kernels take the gate step and the parity masks
  further.

## Rediscovery

autoqec-sim rediscovers the [[5,1,3]], Steane, and rotated surface codes in
between one second and half a minute on a laptop CPU. From the full-budget
regression at seed 0: the states each agent evaluated before a code was
found and verified against the reference, with the wall time of the run
next to each marker.

<p align="center">
  <img src="docs/assets/figures/rediscovery.png" alt="States evaluated up to a verified rediscovery by the evolutionary search and PPO on three targets" width="700">
</p>

PPO reaches a verified rediscovery with 3 to 17 times fewer state
evaluations than the evolutionary search. See [Rediscovery regression](https://unitaryfoundation.github.io/autoqec-sim/dev/regression/)
for the targets, budgets, LDPC-constrained variants, and the verification,
and [Reproduce](https://unitaryfoundation.github.io/autoqec-sim/dev/reproduce/) to rerun both.

## Installation

```bash
pip install git+https://github.com/unitaryfoundation/autoqec-sim
```

Python 3.12 to 3.14; the dependencies are `numpy` and `stim`. The PPO agent
needs torch: install `"autoqec-sim[rl] @ git+https://github.com/unitaryfoundation/autoqec-sim"`.
Releases are tagged on GitHub and archived on Zenodo; there is no PyPI
package.

## Quick Start

```python
from autoqec_sim import CodeDiscoveryEnv, EvolutionarySearch, StabilizerCode

env = CodeDiscoveryEnv(n=5, k=1, batch_size=512, seed=0)
result = EvolutionarySearch(env, seed=0).run(max_evaluations=30_000)

code = StabilizerCode.from_gates(5, 1, result.gates)
print(code.parameters(), code.stabilizer_strings())
```

The search returns a [[5,1,3]] code in under a second on a laptop. Continue
with the [Quick Start](https://unitaryfoundation.github.io/autoqec-sim/dev/getting-started/quickstart/), which verifies the
code against the reference and scores its encoder under a device noise
profile, or use [Choose a Workflow](https://unitaryfoundation.github.io/autoqec-sim/dev/getting-started/choosing-a-workflow/)
to pick a reward, agent, and entry point by the result you need.

## Commands

```
autoqec-regress --budget reduced                           # rediscovery regression, evolutionary search
autoqec-ppo --budget reduced                               # the same targets with PPO (rl extra)
autoqec-schedule 49-13-4 --board /path/to/qldpc-challenge  # CX-layer schedule for a board code's circuit
```

See [Commands](https://unitaryfoundation.github.io/autoqec-sim/dev/reference/commands/) for the options and
[Reproduce](https://unitaryfoundation.github.io/autoqec-sim/dev/reproduce/) for the expected outputs of the two regressions.

## Documentation

https://unitaryfoundation.github.io/autoqec-sim/ (`dev/` tracks `main`,
`stable/` is the latest release). It builds with MkDocs from `docs/`;
`uv run --group docs mkdocs serve` renders it locally.

## Citation

If you use autoqec-sim in your work, please cite the software below
([CITATION.cff](CITATION.cff) holds the same metadata).
```
@software{russo2026autoqecsim,
      title={autoqec-sim: Vectorized stabilizer-code reward engine and RL environment},
      author={Vincent Russo},
      year={2026},
      version={0.2.0},
      url={https://github.com/unitaryfoundation/autoqec-sim},
      license={Apache-2.0},
}
```

The reward and training setup follow Olle et al.,
[arXiv:2311.04750](https://arxiv.org/abs/2311.04750).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Participation is governed by the
[code of conduct](CODE_OF_CONDUCT.md).

## AI Acknowledgement

Generative AI tools were used in parts of the software development and
writing for this project: code generation and review, implementation
analysis, and documentation editing. Design, validation, and release
decisions were made by the human author.

## Funding

This material is based upon work supported by the U.S. Department of
Energy, Office of Science, Office of Advanced Scientific Computing
Research, as part of the Genesis Mission National Science and Technology
Challenges under Award Number DE-SC0025336. This material is also based
upon work supported by the U.S. Department of Energy, Office of Science,
National Quantum Information Science Research Centers, Quantum Science
Center. This research used resources of the National Energy Research
Scientific Computing Center, a DOE Office of Science User Facility
supported by the Office of Science of the U.S. Department of Energy under
Contract No. DE-AC02-05CH11231 using NERSC award ALCC-ERCAP0040673.

## License

Apache-2.0
