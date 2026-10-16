"""Evaluate a code under a device profile: a code given as stabilizer strings,
the proxy tier, then the Tesseract tier. The documentation page
docs/evaluate.md includes the two marked sections; tests/test_evaluate_code.py
runs the first."""

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
