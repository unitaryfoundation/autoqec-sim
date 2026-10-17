"""Memory objective: circuit construction, DEM proxy against exact ML and
sampling, hook-aware orders, flags, lattice routing, bias sensitivity,
the reward closure, and the PPO smoke regression (skipped without torch)."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pytest

from autoqec_sim import graphs, known_codes
from autoqec_sim import memory_objective as mo
from autoqec_sim import noise as nm
from autoqec_sim.code import StabilizerCode
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.experiments.memory_objective import NONCSS_9_ION, biased_idle_profile, xzzx_9
from autoqec_sim.noise import presets
from autoqec_sim.regress import replay


def _codes():
    return {
        "five": known_codes.get("five_qubit"),
        "steane": known_codes.get("steane"),
        "surface": known_codes.get("rotated_surface_9"),
        "noncss": StabilizerCode.from_stabilizers(NONCSS_9_ION),
    }


@pytest.mark.parametrize("name", ["five", "steane", "surface", "noncss"])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_noiseless_circuit_is_deterministic(name, basis):
    code = _codes()[name]
    m = code.stabilizers.shape[0]
    for rounds in (1, 3):
        mc = mo.build_memory_circuit(code, rounds, basis)
        c = mc.circuit()
        dem = c.detector_error_model()  # raises on a non-deterministic detector
        assert dem.num_errors == 0
        assert c.num_detectors == m * (rounds + 1)
        assert c.num_observables == code.k
        assert mc.num_qubits == code.n + m


def test_batches_follow_commutation():
    codes = _codes()
    # batched: CSS X block then Z block; the [[5,1,3]] generators pairwise conflict
    assert mo.build_memory_circuit(codes["steane"], 1, extraction="batched").batches == [[0, 1, 2], [3, 4, 5]]
    assert mo.build_memory_circuit(codes["five"], 1, extraction="batched").batches == [[0], [1], [2], [3]]
    # the default schedule interleaves the [[5,1,3]] generators into one block
    assert mo.build_memory_circuit(codes["five"], 1).batches == [[0, 1, 2, 3]]
    mc = mo.build_memory_circuit(codes["noncss"], 1, extraction="batched")
    letters = [dict(s) for s in mc.supports]
    for batch in mc.batches:
        for a in batch:
            for b in batch:
                assert all(letters[b].get(q, p) == p for q, p in letters[a].items())


def test_hook_order_lifts_shor_and_surface_dcirc():
    shor = known_codes.get("shor")
    surface = known_codes.get("rotated_surface_9")
    prof = presets.chain()

    def dcirc(code, basis, order):
        mc = mo.build_memory_circuit(code, 2, basis, order=order)
        return mo.exact_dcirc(mo.dem_signature(mo.memory_dem(mc, prof, 0.25)))["d_circ"]

    assert dcirc(shor, "Z", "sorted") == 1  # hook error of the weight-6 check is a logical
    assert dcirc(shor, "Z", "hook") == 3
    assert dcirc(surface, "X", "sorted") == 2
    assert dcirc(surface, "X", "hook") == 3 and dcirc(surface, "Z", "hook") == 3


def test_flags_lift_five_qubit_dcirc_and_add_detectors():
    code = _codes()["five"]
    bare = mo.build_memory_circuit(code, 2, "Z")
    flag = mo.build_memory_circuit(code, 2, "Z", flags="one")
    assert flag.circuit().num_detectors == bare.circuit().num_detectors + 4 * 2
    assert flag.circuit().detector_error_model().num_errors == 0
    prof = presets.chain()
    d_bare = mo.exact_dcirc(mo.dem_signature(mo.memory_dem(bare, prof, 0.25)))["d_circ"]
    d_flag = mo.exact_dcirc(mo.dem_signature(mo.memory_dem(flag, prof, 0.25)))["d_circ"]
    assert d_bare == 2 and d_flag == 3


def test_square_layout_routes_only_nonlocal_generators():
    layout = mo.square_layout(3, 3)
    assert layout.n == 9 and len(layout.sites) == 16
    surface = known_codes.get("rotated_surface_9")
    mc = mo.build_memory_circuit(surface, 1, "Z", connectivity="square:3x3")
    assert mc.routed_gates == 0 and mc.two_qubit_gates == 24
    shor = known_codes.get("shor")
    mc2 = mo.build_memory_circuit(shor, 1, "Z", connectivity="square:3x3")
    assert mc2.routed_gates > 0
    assert mc2.circuit().detector_error_model().num_errors == 0
    # every two-qubit gate of the routed circuit lies on the layout graph
    prof = presets.board().with_graph(layout.graph.to_topology())
    noisy = nm.apply(mc2.body, prof, 1.0, n_data=9)
    assert noisy.num_measurements == mc2.body.num_measurements
    with pytest.raises(ValueError):
        mo.build_memory_circuit(surface, 1, "Z", connectivity="square:3x3", flags="one")


def test_proxy_against_exact_ml_and_sampling():
    code = _codes()["five"]
    mc = mo.build_memory_circuit(code, 1, "Z")
    dem = mo.memory_dem(mc, presets.chain(), 0.25)
    sig = mo.dem_signature(dem)
    p2 = mo.pfail_low_weight(sig, 2)
    p_ml, table = mo.exact_ml_failure(sig, 20)
    assert table.shape == (2, 1 << sig.num_detectors)
    assert np.isclose(table.sum(), 1.0)
    assert 0.9 * p_ml < p2 < 1.2 * p_ml
    # exclusive weights are the exact probabilities of the low-weight events,
    # so the truncated sums increase toward the exact failure probability
    p2x = mo.pfail_low_weight(sig, 2, exclusive=True)
    p3x = mo.pfail_low_weight(sig, 3, max_triples=5_000_000, exclusive=True)
    assert p2x < p3x <= p_ml * (1 + 1e-9)
    assert abs(p3x - p_ml) < 0.1 * abs(p2x - p_ml)
    assert p2x < p2  # the leading-order form does not damp by the no-fault probability
    r = mo.sample_failures(dem, sig, "lookup", 20000, seed=3)
    se = np.sqrt(p_ml * (1 - p_ml) / r["shots"])
    assert abs(r["p_fail"] - p_ml) < 4 * se + 0.1 * p_ml


def test_bias_prefers_xzzx_under_z_biased_idle():
    plain = known_codes.get("rotated_surface_9")
    xzzx = StabilizerCode.from_stabilizers(xzzx_9())
    for eta, timing in ((float("inf"), "sequential"), (30.0, "sequential")):
        obj = mo.MemoryObjective(biased_idle_profile(eta, timing), scale=0.25, rounds=2, tier="proxy")
        assert obj.evaluate(xzzx).p_round < 0.8 * obj.evaluate(plain).p_round
    # depolarizing idle: the deformation is a relabeling, the two codes tie
    obj = mo.MemoryObjective(biased_idle_profile(0.5, "sequential"), scale=0.25, rounds=2, tier="proxy")
    assert np.isclose(obj.evaluate(xzzx).p_round, obj.evaluate(plain).p_round, rtol=1e-9)


def test_memory_reward_and_env_hook():
    from _helpers import gates_from_stim

    code = known_codes.get("five_qubit")
    gates = gates_from_stim(code.to_stim_tableau().to_circuit("elimination"))
    obj = mo.MemoryObjective(presets.chain(), scale=0.25, rounds=2, tier="proxy")
    reward = mo.make_memory_reward(obj, target_distance=3, weight=0.1, p_ref=1e-2)
    env = CodeDiscoveryEnv(n=5, k=1, batch_size=2, gates=("H", "S", "S_DAG", "CNOT"), max_steps=len(gates), reward_fn=reward, seed=0)
    assert env.success_fn == reward.success_fn
    env.reset()
    # element 0 runs the [[5,1,3]] encoder, element 1 idles
    table = env.action_table.entries
    acts = np.full((2, len(gates)), -1, dtype=np.int64)
    for t, g in enumerate(gates):
        acts[0, t] = table.index(tuple(g))
    r, succ, used = env.rollout(acts)
    assert succ[0] and not succ[1]
    # the episode ends at the first distance-3 state (the encoder's trailing
    # single-qubit gates only deform the code), so the scored code is the
    # state at that step
    assert used[0] < len(gates)
    found = StabilizerCode.from_tableau(env.tab, 1, 0)
    assert found.distance_exact() == 3
    term = reward.term(found)
    assert np.isclose(r[0], term) and abs(term) <= 0.3
    assert obj.evaluations == 1 and obj.cache_hits >= 1
    assert r[1] < 0 and reward.success_fn(env.tab, 1, r).shape == (2,)
    assert replay(gates, 5).distance_exact() == 3


def test_ppo_smoke_regression_3x3_ion_memory():
    pytest.importorskip("torch")
    from autoqec_sim.agents.ppo import PPOAgent, PPOConfig

    obj = mo.MemoryObjective(presets.chain(), scale=0.25, rounds=2, tier="proxy")
    reward = mo.MemoryReward(obj, target_distance=3, weight=0.1, p_ref=1e-2)
    env = CodeDiscoveryEnv(
        n=9, k=1, graph=graphs.square(3, 3), batch_size=256, gates=("H", "CNOT"), max_steps=30,
        reward_fn=reward, auto_reset=True, seed=0,
    )
    agent = PPOAgent(env, PPOConfig(max_threads=2), seed=0)
    res = agent.train(max_env_steps=300_000, max_time_s=240.0, stop_on_success=True)
    assert res.success, "PPO did not reach a distance-3 code on the 3x3 lattice within the smoke budget"
    code = replay([tuple(g) for g in res.first_success.gates], 9)
    assert code.distance_exact() == 3
    assert obj.evaluations >= 1
    assert isinstance(obj.evaluate(code, 3).p_round, float)


def test_joint_ids_match_unique_and_zero_row():
    """The multi-word joint ids group rows exactly as ``np.unique(axis=0)``
    (equal rows, equal ids; distinct rows, distinct ids), the all-zero row
    gets id 0 when present, and no row gets id 0 when it is absent."""
    rng = np.random.default_rng(3)
    base = rng.integers(0, 4, size=(40, 2), dtype=np.uint64)
    a = base[rng.integers(0, 40, size=300)]
    b = base[rng.integers(0, 40, size=200)]
    a[5] = 0
    ids = np.concatenate(mo._joint_ids(a, b))
    allw = np.concatenate([a, b])
    _, ref = np.unique(allw, axis=0, return_inverse=True)
    ref = ref.reshape(-1)
    # same partition into groups
    pairs = {(int(i), int(r)) for i, r in zip(ids, ref, strict=True)}
    assert len(pairs) == len(set(ids)) == len(set(ref))
    assert ids[5] == 0 and (ids[allw.any(axis=1)] > 0).all()
    nz = allw[allw.any(axis=1)]
    ids_nz = np.concatenate(mo._joint_ids(nz[:150], nz[150:]))
    assert (ids_nz > 0).all()
    single = rng.integers(0, 9, size=(50, 1), dtype=np.uint64)
    assert np.array_equal(mo._joint_ids(single)[0], single[:, 0])
