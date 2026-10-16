# Evaluate a code under a device profile

`evaluate_code(code, profile)` builds the `R`-round syndrome-extraction
circuit of a stabilizer code in both memory bases (the circuit of the
[memory objective](memory_objective.md)), applies a noise profile (a
[device preset](noise_presets.md) or any `autoqec_sim.noise.Profile`), and
returns a `CodeEvaluation` with the logical error per round, the exact
circuit-level distance per basis, and the most probable undetectable
fault sets explained back to circuit locations. The code is a
`StabilizerCode` or its stabilizer generators as Pauli strings, CSS or not.

```python
evaluate_code(code, profile, *, rounds=None, scale=1.0, tier="proxy",
              shots=20_000, min_failures=0, max_shots=None, seed=0,
              connectivity="all_to_all", max_faults=10)
```

`rounds` defaults to the code's exact distance; `scale` multiplies every
rate of the profile; `connectivity` is `"all_to_all"`, `"square:RxC"`, or
a `memory_objective.Layout`. The gate order of the extraction circuit is
the hook-aware order of the memory objective, shared by the two bases.

## Example

The whole script is `docs/examples/evaluate_code.py`. The first part runs
in half a second.

```python
--8<-- "docs/examples/evaluate_code.py:proxy"
```

```
error per round 9.05e-03  d_circ {'Z': 3, 'X': 3}
Z basis, probability 1.4e-06, 3 mechanisms:
  X4      after DEPOLARIZE2 on (14, 4) at tick 16  ->  D12 D13
  X1      after DEPOLARIZE2 on (13, 1) at tick 14  ->  D12 L0
  X6      after DEPOLARIZE2 on (14, 6) at tick 15  ->  D13
```

The three data-qubit `X` faults, each the data half of a two-qubit
depolarizing channel after a `CZ` from a `Z`-check ancilla (qubits 13 and
14), multiply to a logical `X` (the column `{1, 4, 7}` times the boundary check
`{6, 7}`). Their detector flips cancel in pairs and only the observable
`L0` is left, so no decoder can see them. There are 176 such weight-3 sets
in the `Z` basis with a summed mass of `8.7e-6` per shot
(`result.d_circ_mass["Z"]`); the list holds the ten most probable per basis.

```python
--8<-- "docs/examples/evaluate_code.py:tesseract"
```

```
tesseract error per round 8.45e-03
  Z: 248 failures in 20000 shots, 2798 distinct syndromes
  X: 257 failures in 20000 shots, 2797 distinct syndromes
```

Tesseract [@beni2025] decodes the detector error model as it is, without
graphlike decomposition, so it applies to non-CSS codes and to the
correlated channels of the device presets. The proxy above (`9.05e-3`) and
the sampled tier (`8.79e-3` with the lookup decoder at the same shots and
seed) agree with it within shot noise on this code. Distinct syndromes are
decoded once, so the 20,000 shots per basis cost about 2,800 decoder calls
each, a few seconds on a laptop.

## What comes back

`CodeEvaluation` is a frozen dataclass; `to_dict()` gives plain Python
types for JSON.

| Field | Content |
|---|---|
| `p_round` | logical error per round of `tier`, summed over the `Z` and `X` memories |
| `per_basis` | per basis, the dictionary of the memory objective: `p_fail` per shot, `p_round`, `decoder`, `sampling` (`shots`, `failures`, `distinct_syndromes`, `seconds`) for the sampling tiers, `mechanisms`, `detectors`, `two_qubit_gates`, `layers`, `expected_faults` per shot, and `d_circ`, `d_circ_count`, `d_circ_mass` |
| `d_circ`, `d_circ_mass` | per basis, the exact circuit-level distance (`None` above 4) and the summed probability product of its minimum-weight undetectable fault sets, computed whatever the tier |
| `tier`, `rounds` | the tier used and the number of noisy rounds |
| `dominant_faults` | `DominantFault` entries over both bases, most probable first |

## Tiers

| `tier` | Estimate | Cost | Needs |
|---|---|---|---|
| `proxy` | failure probability of the maximum-likelihood decoder restricted to fault sets of weight at most two, closed form from the DEM, exact to `O(p^3)` | `O(M^2)` in the `M` mechanisms; 0.1 to 0.5 s per code at `n = 9` | stim |
| `sampled` | exact maximum-likelihood decoder when `detectors + observables <= 20`, else a lookup decoder over the same weight-2 sets; Monte Carlo | `shots` per basis, distinct syndromes decoded once | stim |
| `bposd` | the board's pinned BP+OSD decoder [@roffe2020; @panteleev2021]; Monte Carlo | as `sampled`, slower per syndrome | `ldpc` (`uv run --with ldpc`) |
| `tesseract` | Tesseract most-likely-error search [@beni2025] on the full DEM; Monte Carlo | 1 to 2 ms per distinct syndrome on the surface-code DEM | `tesseract` extra |

The proxy and the lookup decoder are leading-order estimates. They hold
while the expected number of faults per shot (`per_basis[b]["expected_faults"]`)
stays below about one; above that, use `tesseract` or `bposd`. The
[memory objective](memory_objective.md) page has the benchmark of the
tiers against each other and the regime of each preset.

Install the Tesseract tier with the extra (PyPI carries development
releases of `tesseract-decoder` only):

```bash
pip install "autoqec-sim[tesseract] @ git+https://github.com/unitaryfoundation/autoqec-sim"
uv run --extra tesseract python docs/examples/evaluate_code.py    # in this repository
```

## Dominant faults

The circuit-level distance `d_circ` of a basis is the smallest number of
DEM mechanisms whose combined effect flips no detector and some
observable, found exactly up to weight 4. `dominant_faults` lists, per
basis, the `max_faults` most probable of these minimum-weight sets when
`d_circ` is at most 3 (a basis with `d_circ` of 4 or more contributes no
entries). Each `DominantFault` carries:

| Field | Content |
|---|---|
| `basis` | `"Z"` or `"X"` |
| `mechanisms` | indices into the `error` instructions of `dem.flattened()`, the mechanism order of `memory_objective.dem_signature` |
| `probability` | product of the mechanism probabilities |
| `locations` | one `FaultLocation` per mechanism |

A `FaultLocation` comes from stim's
`explain_detector_error_model_errors(dem_filter=..., reduce_to_one_representative_error=True)`
on the noisy circuit: `gate` is the noise instruction (`DEPOLARIZE2`,
`PAULI_CHANNEL_1`, `M` for a measurement flip), `qubits` its targets,
`pauli` the flipped Pauli product (`"Y10"`, `"Z10*X7"`, or `"flip Z9"` for
a measurement flip), `tick` the number of `TICK` layers before it, and
`targets` the detectors and observables the mechanism flips (`"D12 L0"`).
Several circuit errors usually produce the same mechanism (the data half
of every two-qubit channel on a qubit in the same layer, for one); stim
reports one representative. `gate` is `"?"` when stim cannot attribute a
mechanism to a single circuit error.

The same enumeration is available on any DEM signature as
`memory_objective.undetectable_sets(sig, wmax=3, limit=10)`, which returns
the weight, the count, and the most probable sets; its weight and count
agree with `exact_dcirc` whenever the distance is at most 3.
