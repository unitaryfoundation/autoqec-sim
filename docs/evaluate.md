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
error per round 1.15e-03  d_circ {'Z': 3, 'X': 3}
Z basis, probability 3.6e-08, 3 mechanisms:
  X4      after DEPOLARIZE2 on (14, 4) at tick 4  ->  D12 D13
  X1      after DEPOLARIZE2 on (13, 1) at tick 2  ->  D12 L0
  X6      after DEPOLARIZE2 on (14, 6) at tick 3  ->  D13
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
tesseract error per round 8.49e-03
  Z: 248 failures in 20000 shots, 2798 distinct syndromes
  X: 257 failures in 20000 shots, 2797 distinct syndromes
```

Tesseract [@beni2025] decodes the detector error model as it is, without
graphlike decomposition, so it applies to non-CSS codes and to the
correlated channels of the device presets. The proxy above (`9.09e-3`) and
the sampled tier (`8.83e-3` with the lookup decoder at the same shots and
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
| `d_circ`, `d_circ_mass` | per basis, the exact circuit-level distance (`None` above 6) and the summed probability product of its minimum-weight undetectable fault sets, computed whatever the tier |
| `tier`, `rounds` | the tier used and the number of noisy rounds |
| `dominant_faults` | `DominantFault` entries over both bases, most probable first |

## Tiers

| `tier` | Estimate | Cost | Needs |
|---|---|---|---|
| `proxy` | failure probability of the maximum-likelihood decoder restricted to fault sets of weight at most two, closed form from the DEM, exact to `O(p^3)`; at circuit distance 5 or 6, where that form is zero, weight at most three through the weight-5 and weight-6 undetectable sets, exact to `O(p^4)` | `O(M^2)` in the `M` mechanisms; 0.1 to 0.5 s per code at `n = 9`; 25 to 50 s per basis for the 5x5 surface code at five rounds | stim |
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
observable, found exactly up to weight 6 (pair tables to weight 4, the
cluster search of `memory_objective.undetectable_logicals` beyond).
`dominant_faults` lists, per basis, the `max_faults` most probable of
these minimum-weight sets (a basis with `d_circ` of 7 or more contributes
no entries). Each `DominantFault` carries:

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
`memory_objective.undetectable_sets(sig, wmax=6, limit=10)`, which returns
the weight, the count, and the most probable sets; its weight and count
agree with `exact_dcirc` whenever the distance is at most `wmax`.
`memory_objective.undetectable_logicals(sig, wmax)` returns every
irreducible undetectable logical set up to `wmax` as arrays, weight by
weight.

## A distance-5 code

In a distance-preserving circuit for a distance-5 code no fault set of
weight at most 4 is undetectable, so the weight-2 proxy is zero. The proxy
tier then escalates to weight 3: `memory_objective.undetectable_logicals`
lists every weight-5 and weight-6 undetectable logical fault set by a
cluster search over the detector hypergraph, and the 3-subsets of those
sets are the only weight-3 fault sets the decoder can get wrong
(`proxy_weight` in `per_basis` reports 3). The 5x5 rotated surface code at
five rounds under depolarizing noise at 1e-3:

```python
--8<-- "docs/examples/evaluate_code.py:distance5"
```

```
error per round 9.11e-05  d_circ {'Z': 5, 'X': 5}  proxy weight 3
  Z: 12906 weight-5 undetectable sets, mass 1.4e-11, p_fail 2.28e-04
  X: 12221 weight-5 undetectable sets, mass 1.4e-11, p_fail 2.28e-04
Z basis, probability 1.2e-13, 5 mechanisms
```

The search on this 2,107-mechanism model takes 25 to 50 s per basis
depending on machine load; the hook-order search for the 25-qubit code
adds a few seconds. Tesseract with 50,000 shots per basis gives 8.4e-05
per round from 12 and 9 failures (a standard error of about 30 percent)
in three and a half minutes, 8 percent below the proxy, at 1.4 expected
faults per shot. Beyond distance 6 the proxy is zero again; the search
cost grows about twentyfold per unit of weight.


## Subsystem codes

A `SubsystemCode` is given by independent generators of its gauge group;
the center (the stabilizer group), the bare logical operators, and one
gauge fixing (a stabilizer code with the same bare logicals, used for the
distance and the gate-order search) are computed from them. Its memory
circuit measures every gauge generator with its own ancilla, in batches of
mutually interleavable generators, and reads each stabilizer as the parity
of the gauge outcomes whose product it is, with the sign of that product
carried into the reference measurements. The gauge generators of the
Bacon-Shor code [@bacon2006] and of the distance-3 heavy-hexagon code
[@sundaresan2023] ship in `known_codes` (`get_subsystem`).

```python
--8<-- "docs/examples/evaluate_code.py:subsystem"
```

```
(9, 1, 3) ['XXXXXXIII', 'IIIXXXXXX', 'ZZIZZIZZI', 'IZZIZZIZZ']
error per round 1.01e-02  d_circ {'Z': 3, 'X': 3}  gauge generators measured 12
```

A stabilizer whose factors do not all commute cannot be read in one round,
and `stabilizer_factors` raises for it; the heavy-hexagon code's two
weight-6 Z stabilizers and four X stabilizers are all products of
commuting gauge generators.

## Validation and limits

- The proxy is exact for what it computes. On the one-round 5x5 surface
  code its weight-3 form equals the enumeration of every triple, and the
  weight-5 count and mass match (`tests/test_undetectable_logicals.py`).
- It ranks like the sampled decoders where they overlap: Spearman 0.79,
  0.90, and 0.78 against BP+OSD over 32 random [[9,1,3]] codes under the
  chain, board, and Heron profiles
  ([memory objective runs](experiments/memory_objective.md)), within 25
  percent of the exact maximum-likelihood tier on every benchmark code
  below one expected fault per shot, within 8 percent of Tesseract on
  the 5x5 surface code at five rounds, where 50,000 shots per basis leave
  Tesseract itself a 30 percent standard error (above), and at Spearman
  0.89 and 0.93 against Tesseract over 21 distance-5 codes under the
  Forte preset at three and five rounds (1.7 and 2.8 expected faults per
  shot), where it runs about 30 percent low in level
  ([proxy against Tesseract](experiments/proxy_tesseract.md)).
- The proxy is a leading-order estimate. It holds while `expected_faults`
  per shot stays below about one and underestimates above that; the
  sampled tiers take over there.
- The circuit-distance search stops at weight 6, about twentyfold more
  work per unit of weight, so a basis with `d_circ` of 7 or more reports
  `None` and a zero proxy.
- The memory circuit interleaves every generator into one block with one
  measurement per round when a hook-safe schedule exists, which it does
  for every surface code tried (the standard four-layer cycle), the Shor
  code, and the [[5,1,3]] code. When none exists within the search cap
  (the Steane code), generators that share a qubit with different Pauli
  letters run in separate blocks with their own measurements, and the
  extra idle time makes such a code's absolute rate pessimistic
  (`per_basis[b]["extraction"]` says which schedule was used). Subsystem
  codes are measured through their gauge generators (above); flags beyond
  one per generator, adaptive protocols, and transport are not
  expressible.
- The presets are noise models assembled from public specifications, not
  device measurements, and `evaluate_code` builds its own circuit. A rate
  it reports is for that circuit under that model.
