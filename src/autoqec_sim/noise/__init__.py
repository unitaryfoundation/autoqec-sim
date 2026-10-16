"""Circuit-level noise profiles for stim memory circuits.

    from autoqec_sim.noise import apply, presets
    noisy = apply(skeleton, presets.forte(), scale=0.25)

See README.md for the interface, the profiles table and the sources.
"""

from . import presets
from .apply import ConnectivityError, apply, dem, fault_expectation, mechanism_count, noisifier
from .channels import DEPOLARIZING_BIAS, PAULI2, biased_1q, biased_2q, marginal_bias_2q
from .profile import Correlated, Idle, Leakage, Params, Profile, scaled, summarize
from .skeleton import circuit_qubits, strip_noise
from .timing import PARALLEL, SEQUENTIAL, Timing, partial, reslot
from .topology import Graph, Violation, check_connectivity, complete, heavy_hex, ion_chain, path, square, triangular

__version__ = "0.2.0"

__all__ = [
                    "DEPOLARIZING_BIAS",
                    "PARALLEL",
                    "PAULI2",
                    "SEQUENTIAL",
                    "ConnectivityError",
                    "Correlated",
                    "Graph",
                    "Idle",
                    "Leakage",
                    "Params",
                    "Profile",
                    "Timing",
                    "Violation",
                    "apply",
                    "biased_1q",
                    "biased_2q",
                    "check_connectivity",
                    "circuit_qubits",
                    "complete",
                    "dem",
                    "fault_expectation",
                    "heavy_hex",
                    "ion_chain",
                    "marginal_bias_2q",
                    "mechanism_count",
                    "noisifier",
                    "partial",
                    "path",
                    "presets",
                    "reslot",
                    "scaled",
                    "square",
                    "strip_noise",
                    "summarize",
                    "triangular",
]
