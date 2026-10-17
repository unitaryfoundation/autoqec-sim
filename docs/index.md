---
hide:
  - navigation
title: autoqec-sim
---

# autoqec-sim { .visually-hidden }

<div class="hero" markdown>

![autoqec-sim](assets/logos/autoqec-sim.svg){ width=420 }

<p class="hero-tagline">
Circuit-level evaluation of stabilizer codes under device noise, in one
call and without sampling, for codes from any source.
</p>

</div>

[![CI](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/unitaryfoundation/autoqec-sim/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/unitaryfoundation/autoqec-sim/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](getting-started/installation.md)

---

## What is autoqec-sim?

autoqec-sim evaluates stabilizer codes under device noise. One call builds
the syndrome-extraction memory experiment of any stabilizer code, CSS or
not, applies a noise model with sourced device parameters, and returns the
logical error per round, the exact circuit-level distance, and the
dominant fault sets explained back to gates and qubits, without sampling
or a decoder.

- **Four scoring tiers** on stim's detector error model [@gidney2021]: a
  closed-form maximum-likelihood proxy over low-weight fault sets (no
  shots, no decoder), exact maximum likelihood or a lookup decoder, BP+OSD
  [@roffe2020; @panteleev2021], and Tesseract [@beni2025]. The proxy ranks
  codes the way the sampled decoders do and holds below about one expected
  fault per shot; [Evaluate a code](evaluate.md) has the validation and
  the limits.
- **Three proposers** and the tools that verify what they find: a
  SAT-population pipeline over the codes of [@dalfavero2026], an
  evolutionary search, and a PPO agent in the gate-by-gate formulation of
  [@olle2024], on a batched Aaronson-Gottesman tableau with a torch and
  CUDA path.
- **Tracked experiments**: the [experiments](experiments/index.md) hold
  what each has found, with a figure and the commands that reproduce it.

## Quick Example

Install with `pip install git+https://github.com/unitaryfoundation/autoqec-sim`,
then evaluate the rotated surface code under three rounds of depolarizing
noise:

```python
--8<-- "docs/examples/evaluate_code.py:proxy"
```

```
error per round 1.15e-03  d_circ {'Z': 3, 'X': 3}
Z basis, probability 3.6e-08, 3 mechanisms:
  X4      after DEPOLARIZE2 on (14, 4) at tick 4  ->  D12 D13
  X1      after DEPOLARIZE2 on (13, 1) at tick 2  ->  D12 L0
  X6      after DEPOLARIZE2 on (14, 6) at tick 3  ->  D13
```

The [Quick Start](getting-started/quickstart.md) compares codes under a
device preset, proposes a code with the evolutionary search, and verifies
it against the reference.

## Key Features

<div class="grid cards" markdown>

- **Evaluate a code**

    [`evaluate_code`](evaluate.md): logical error per round, exact
    circuit-level distance up to weight 6, and the most probable
    undetectable fault sets as circuit locations, for CSS and non-CSS
    codes, at a closed-form proxy tier or with a sampled decoder.

- **Noise models with sourced parameters**

    [Device presets](noise_presets.md) for IonQ Forte, IBM Heron, a long
    ion chain, and the qldpc-challenge recipe; biased, correlated, and
    leakage-aware channels, device timing, and connectivity through one
    [noise interface](noise.md) that also noisifies a `stim.Circuit` of
    your own.

- **Verification independent of the objective**

    Exact distance, weight enumerators, and the sparsest generator basis
    for every code; the [rediscovery regression](regression.md) checks
    each found code against its reference.

- **Proposers**

    The SAT-population pipeline of the
    [experiments](experiments/sat_population.md), an evolutionary search,
    and PPO with action masking and a permutation-equivariant policy, with
    an [LDPC sparsity constraint](ldpc.md) and the
    [memory objective](memory_objective.md) as a reward; the
    [torch backend](gpu.md) runs the kernels on CUDA.

</div>

For a fixed code, `autoqec-schedule` trains a PPO policy on the exact
circuit fault distance of the syndrome-extraction circuit and confirms the
best CX-layer schedule at `rounds = d`. See [Commands](reference/commands.md).

## Get Started

[Evaluate a code](evaluate.md){ .md-button .md-button--primary }
[Quick Start](getting-started/quickstart.md){ .md-button }

## What's New

Since release 0.2.0, `main` adds [`evaluate_code`](evaluate.md), the
Tesseract tier, the proxy at circuit distance 5 and 6 through the
enumeration of weight-5 and weight-6 undetectable fault sets, the
SAT-designed-code experiments, and one documentation page per experiment.

Release 0.2.0 added the [noise module](noise.md) as the `autoqec_sim.noise`
subpackage, with biased, correlated, and leakage-aware channels, device
timing, topologies, and [sourced presets](noise_presets.md) for IonQ Forte,
a long ion chain, IBM Heron, and the qldpc-challenge recipe; the
[LDPC constraint](ldpc.md) as a hard constraint or a reward penalty; the
[PPO agent](regression.md#ppo-agent), which matches the evolutionary
search on every rediscovery target with 3 to 17 times fewer state
evaluations; the [memory objective](memory_objective.md) with its
fidelity tiers; and the [torch backend](gpu.md) with two fused CUDA
kernels.

[Full Changelog](development/changelog.md){ .md-button }
