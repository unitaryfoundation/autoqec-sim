"""Evaluate a code under a device profile: a code given as stabilizer strings,
the proxy tier, the Tesseract tier, then a distance-5 code. The documentation
page docs/evaluate.md includes the three marked sections;
tests/test_evaluate_code.py runs the first; the third takes about a minute."""

# --8<-- [start:proxy]
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
# --8<-- [end:proxy]

# --8<-- [start:tesseract]
# Monte Carlo with Tesseract: 20,000 shots per basis, each distinct syndrome
# decoded once. Needs the `tesseract` extra.
sampled = evaluate_code(surface, presets.depolarizing(1e-3), rounds=3, tier="tesseract", shots=20_000)
print(f"tesseract error per round {sampled.p_round:.2e}")
for basis in ("Z", "X"):
    s = sampled.per_basis[basis]["sampling"]
    print(f"  {basis}: {s['failures']} failures in {s['shots']} shots, {s['distinct_syndromes']} distinct syndromes")
# --8<-- [end:tesseract]

# --8<-- [start:distance5]
# A distance-5 code: the 5x5 rotated surface code at five rounds. No fault
# set of weight at most 4 is undetectable, so the weight-2 proxy is zero and
# the proxy tier escalates to weight 3 through the weight-5 and weight-6
# undetectable sets (a cluster search over the detector hypergraph, about
# 20 s per basis for this 2,100-mechanism model).
from autoqec_sim.known_codes import rotated_surface

d5 = evaluate_code(rotated_surface(5), presets.depolarizing(1e-3), rounds=5)
print(f"error per round {d5.p_round:.2e}  d_circ {d5.d_circ}  proxy weight {d5.per_basis['Z']['proxy_weight']}")
for basis in ("Z", "X"):
    s = d5.per_basis[basis]
    print(f"  {basis}: {s['d_circ_count']} weight-5 undetectable sets, mass {s['d_circ_mass']:.1e}, p_fail {s['p_fail']:.2e}")
top = d5.dominant_faults[0]
print(f"{top.basis} basis, probability {top.probability:.1e}, {len(top.mechanisms)} mechanisms")
# --8<-- [end:distance5]
