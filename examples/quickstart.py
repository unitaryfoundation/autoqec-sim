"""Quick start: find a [[5,1,3]] code and print its stabilizers. Kept
identical to the snippet in README.md."""

from autoqec_sim import CodeDiscoveryEnv, EvolutionarySearch, StabilizerCode

env = CodeDiscoveryEnv(n=5, k=1, batch_size=512, seed=0)
result = EvolutionarySearch(env, seed=0).run(max_evaluations=30_000)

code = StabilizerCode.from_gates(5, 1, result.gates)
print(code.parameters(), code.stabilizer_strings())
