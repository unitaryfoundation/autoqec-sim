"""Quick start: score three known codes under a device noise profile, then
search for a code. The two marked sections are included verbatim by the
documentation and copied in README.md (tests/test_examples.py keeps the
copies equal and runs this file)."""

# --8<-- [start:score]
from autoqec_sim import known_codes
from autoqec_sim.memory_objective import MemoryObjective
from autoqec_sim.noise import presets

# A three-round memory experiment under the IonQ Forte preset at a quarter of
# its measured rates, scored by the low-weight maximum-likelihood proxy.
objective = MemoryObjective(presets.forte("sequential"), scale=0.25, rounds=3)
for name in ("five_qubit", "steane", "rotated_surface_9"):
    code = known_codes.get(name)
    print(name, code.parameters(), f"error per round {objective.evaluate(code).p_round:.2e}")
# --8<-- [end:score]

# --8<-- [start:search]
from autoqec_sim import CodeDiscoveryEnv, EvolutionarySearch, StabilizerCode

env = CodeDiscoveryEnv(n=5, k=1, batch_size=512, seed=0)
result = EvolutionarySearch(env, seed=0).run(max_evaluations=30_000)

code = StabilizerCode.from_gates(5, 1, result.gates)
print(code.parameters(), code.stabilizer_strings())
# --8<-- [end:search]
