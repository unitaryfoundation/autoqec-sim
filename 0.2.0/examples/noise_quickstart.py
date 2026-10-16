"""Quick start: noisify a one-round repetition-code skeleton under two presets.
The documentation's Quick Start includes the marked section."""

# --8<-- [start:noise]
import stim

from autoqec_sim import noise as nm
from autoqec_sim.noise import presets

# Three-qubit repetition code, one round, in the TICK-layer convention:
# data qubits 0, 2, 4 and one ancilla per Z check (qubits 1, 3).
skeleton = stim.Circuit("""
    R 0 1 2 3 4
    TICK
    CX 0 1 2 3
    TICK
    CX 2 1 4 3
    TICK
    M 1 3
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    TICK
    M 0 2 4
    OBSERVABLE_INCLUDE(0) rec[-3]
""")

# IonQ Forte preset at a quarter of its measured rates (p2 = 1e-3).
noisy = nm.apply(skeleton, presets.forte(), scale=0.25)
dem = nm.dem(noisy)
print(noisy)
print("mechanisms:", nm.mechanism_count(dem), "expected faults:", round(nm.fault_expectation(dem), 5))

# Superconducting preset on a line graph, Z-biased tenfold, one two-qubit
# gate per time slot; gates off the graph raise ConnectivityError.
profile = presets.heron(graph=nm.path(5)).with_bias(10.0).with_timing(nm.partial(1))
print(nm.summarize(profile, scale=1.0))
print("exact DEM:", profile.dem_exact, "layers:", str(nm.apply(skeleton, profile, 1.0)).count("TICK") + 1)
# --8<-- [end:noise]
