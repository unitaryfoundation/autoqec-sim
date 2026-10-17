import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import numpy as np
import pytest

from autoqec_sim import graphs, known_codes
from autoqec_sim.circuit_noise import fault_set, fault_set_reference, gates_to_layered_circuit, resolve_noise_model
from autoqec_sim.code import StabilizerCode
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.noise import presets
from autoqec_sim.regress import replay
from experiments.noise_aware import run as na


def _random_gate_lists(n, spec, gates, length, count, seed):
    env = CodeDiscoveryEnv(n=n, k=1, graph=graphs.from_spec(spec), batch_size=count, gates=gates, max_steps=length, seed=seed)
    env.reset()
    for _ in range(length):
        env.step(env.sample_actions())
    return env.gate_lists(), env.graph


def _fault_key(fs):
    return sorted(zip((bytes(r) for r in fs.paulis), np.round(fs.probs, 14), fs.heralded.tolist(), fs.source.tolist()))


@pytest.mark.parametrize("profile", [presets.board(), presets.forte("sequential"), presets.heron(graph=None)])
def test_symplectic_fault_set_matches_stim_reference(profile):
    gl, graph = _random_gate_lists(7, "all_to_all:7", ("H", "S", "CNOT"), 16, 4, 3)
    model = resolve_noise_model(profile, graph=graph, n=7)
    for g in gl:
        noisy = model(gates_to_layered_circuit(g, 7))
        assert _fault_key(fault_set(noisy, 7)) == _fault_key(fault_set_reference(noisy, 7))


def test_known_code_constructions():
    ref = known_codes.get("rotated_surface_9")
    rs = StabilizerCode.from_stabilizers(na.rotated_surface(3))
    assert na.code_key(rs.to_tableau()[0], 1) == na.code_key(ref.to_tableau()[0], 1)
    sh = StabilizerCode.from_stabilizers(na.shor_code(3))
    assert na.code_key(sh.to_tableau()[0], 1) == na.code_key(known_codes.get("shor").to_tableau()[0], 1)
    for name, spec in na.known_codes_for(na.SETTINGS["4x4"]).items():
        code = StabilizerCode.from_stabilizers(spec["stabilizers"])
        assert (code.n, code.k) == (16, 1)
        assert code.distance_exact() == 4, name


def test_pure_type_distances_and_frames():
    steane = known_codes.get("steane")
    tab = steane.to_tableau()
    assert na.pure_type_distance(tab, 1, "X")[0] == 3 and na.pure_type_distance(tab, 1, "Z")[0] == 3
    five = known_codes.get("five_qubit")
    assert na.pure_type_distance(five.to_tableau(), 1, "Z") == (5, 1)
    assert not na.css_frames(five)["css_frame_found"]
    xzzx = StabilizerCode.from_stabilizers(na.known_codes_for(na.SETTINGS["3x3"])["xzzx"]["stabilizers"])
    fr = na.css_frames(xzzx)
    assert fr["css_frame_found"] and fr["hadamard_only"] and fr["frame_size"] == 4
    shor = StabilizerCode.from_stabilizers(na.shor_code(3))
    comp = na.group_component_counts(shor)
    assert (comp["pure_x_elements"], comp["pure_z_elements"]) == (3, 63)


def test_standard_form_encoders_and_routing():
    for L in (2, 3, 4):
        path = list(range(L + 1))
        a = replay(na._long_range_cnot(path), L + 1).to_tableau()[0][:, : 2 * (L + 1)]
        b = replay([("CNOT", 0, L)], L + 1).to_tableau()[0][:, : 2 * (L + 1)]
        assert np.array_equal(a, b)
    setting = na.SETTINGS["3x3"]
    graph = graphs.from_spec(setting.graph_spec)
    for name, spec in na.known_codes_for(setting).items():
        frame = spec["frame"]
        parent = na.hadamard_conjugate(spec["stabilizers"], frame) if frame else spec["stabilizers"]
        target = na.code_key(StabilizerCode.from_stabilizers(spec["stabilizers"]).to_tableau()[0], 1)
        for logical in (0, 4):
            gates = na.css_standard_encoder(parent, logical=logical, graph=graph) + [("H", q, -1) for q in frame]
            routed = na.route_gates(gates, graph)
            assert all(graph.has_edge(a, b) for g, a, b in routed if g == "CNOT")
            assert na.code_key(na.code_from_circuit(routed, 9, logical).to_tableau()[0], 1) == target, (name, logical)


def test_noise_aware_reward_terms_only_on_target_states():
    setting = na.SETTINGS["n7"]
    calib = na.calibrate(setting, count=8)
    assert calib["profiles"]["board"]["scale"] == 1.0
    env = na.make_env(setting, 0, 16)
    models = na.calibrated_models(setting, calib, env.graph)
    reward = na.NoiseAwareReward(env, setting, models["ion"], calib["m0"])
    env.reward_fn = reward
    env.success_fn = reward.success_fn
    env.reset()
    r = env.state_reward
    # identity tableau: d = 1, six weight-1 Z stabilizers out of 21 weight-1 Paulis, no noise term
    assert np.allclose(r, 1 / 3 - 1 - 6 / 21)
    # a Steane encoder reaches d = 3: reward = noise + LDPC terms, both <= 0, success flagged
    gates = na.stim_encoder(known_codes.STEANE, ("H", "S", "CNOT"))
    acts = {tuple(e): i for i, e in enumerate(env.action_table.entries)}
    seq = np.full((16, len(gates)), -1, dtype=np.int64)
    seq[0] = [acts[tuple(g)] for g in gates]
    env.max_steps = len(gates)
    env.history = np.full((16, env.max_steps), -1, dtype=np.int64)
    rew, ok, _ = env.rollout(seq)
    assert ok[0] and -1.0 < rew[0] < 0.0
    assert not ok[1:].any()
