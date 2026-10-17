# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Circuit-level evaluation of stabilizer codes under device noise, and the
search tools that propose candidates for it.

Modules
-------
evaluate       evaluate_code: one call for p_round, d_circ, and dominant faults
memory_objective  multi-round memory experiment, decoder tiers, exact d_circ
noise          noise profiles, channels, timing, topologies, device presets
circuit_noise  circuit-level noise reward of an encoding circuit
code           code extraction, exact and randomized distance, enumerators
known_codes    reference codes
tableau        batched Aaronson-Gottesman tableaux and Clifford gates
symplectic     GF(2) and symplectic helpers, tableau completion
errors         Pauli error sets (by weight or by noise-profile probability)
reward         Knill-Laflamme, distance, and low-weight-stabilizer rewards
graphs         connectivity graphs (shared with autoqec_sim.noise.topology)
ldpc           LDPC sparsity: sparsest generator basis, constraint, penalty
env            gymnasium-style batched environment (optional LDPC constraint)
search         random and evolutionary search agents
regress        rediscovery regression
agents.ppo     PPO agent (optional torch dependency, imported lazily)
"""

from .code import StabilizerCode
from .env import CodeDiscoveryEnv
from .errors import ErrorSet
from .evaluate import CodeEvaluation, DominantFault, FaultLocation, evaluate_code
from .ldpc import ldpc_feasible, ldpc_penalty, sparsest_basis_stats
from .reward import kl_reward, low_weight_stabilizer_penalty, undetected_mask
from .search import EvolutionarySearch
from .tableau import BatchTableau, identity_tableau

__version__ = "0.2.0"

__all__ = [
    "BatchTableau",
    "CodeDiscoveryEnv",
    "CodeEvaluation",
    "DominantFault",
    "ErrorSet",
    "EvolutionarySearch",
    "FaultLocation",
    "StabilizerCode",
    "__version__",
    "evaluate_code",
    "identity_tableau",
    "kl_reward",
    "ldpc_feasible",
    "ldpc_penalty",
    "low_weight_stabilizer_penalty",
    "sparsest_basis_stats",
    "undetected_mask",
]
