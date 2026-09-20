"""Quick start: rediscover the [[5,1,3]] code with the evolutionary search,
verify it independently of the reward, and score its encoding circuit under
a circuit-level noise profile. Kept identical to the snippet in README.md."""

import numpy as np

from autoqec_sim import BatchTableau, StabilizerCode, known_codes
from autoqec_sim.circuit_noise import circuit_level_reward
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.regress import TARGETS, make_reward
from autoqec_sim.search import EvolutionarySearch
from noise_model import presets

# Batched environment: 512 parallel episodes on 5 all-to-all qubits with
# H, S, and CNOT, at most 14 gates; the reward is the Knill-Laflamme term
# over all weight-1 and weight-2 Paulis plus the weight-1 stabilizer penalty.
target = TARGETS["five_qubit"]
env = CodeDiscoveryEnv(n=5, k=1, batch_size=512, gates=target.gates, max_steps=target.max_steps,
                       reward_fn=make_reward(target), seed=0)
res = EvolutionarySearch(env, seed=0).run(max_evaluations=30_000)
print(f"search: success={res.success} evaluations={res.evaluations} gates={len(res.gates)} wall={res.wall_time_s:.1f} s")

# Replay the found gate sequence on a fresh tableau and read the code off it.
tab = BatchTableau(5, 1)
for gate, a, b in res.gates:
    tab.apply(gate, a, b if b >= 0 else None)
code = StabilizerCode.from_tableau(tab.array, k=1)
ref = known_codes.get("five_qubit")
print(f"code: n={code.n} k={code.k} d={code.distance_exact()} css={code.is_css()}")
match = all(np.array_equal(a, b) for a, b in zip(code.weight_enumerators(), ref.weight_enumerators(), strict=True))
print("enumerators match [[5,1,3]]:", match)

# Circuit-level score: every single-fault mechanism of the encoding circuit
# under the IonQ Forte profile, propagated with stim, as -undetected mass.
r = circuit_level_reward(tab.array, 1, [res.gates], presets.forte(), scale=0.25)
print(f"circuit-level reward under forte @ 0.25: {r[0]:.4f}")
