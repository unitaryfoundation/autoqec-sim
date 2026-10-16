import numpy as np
import stim

from autoqec_sim import graphs
from autoqec_sim.code import StabilizerCode
from autoqec_sim.env import CodeDiscoveryEnv, build_action_table
from autoqec_sim.tableau import BatchTableau, gates_to_stim_circuit


def test_graph_builders():
    assert graphs.all_to_all(5).edges == tuple((a, b) for a in range(5) for b in range(a + 1, 5))
    assert graphs.linear(4).edges == ((0, 1), (1, 2), (2, 3))
    sq = graphs.square(3, 3)
    assert sq.n == 9 and len(sq.edges) == 12 and sq.is_connected()
    tri = graphs.triangular(3, 3)
    assert len(tri.edges) == 12 + 4
    hh = graphs.heavy_hex(2, 3)
    assert hh.is_connected() and max(hh.degree()) <= 3
    assert graphs.from_spec("square:3x3") == sq
    assert graphs.from_spec("all_to_all:7").n == 7


def test_graphs_round_trip_with_autoqec_noise_topology():
    from autoqec_sim import noise as nm

    for g, t in [
        (graphs.square(3, 3), nm.square(3, 3)),
        (graphs.triangular(2, 3), nm.triangular(2, 3)),
        (graphs.heavy_hex(1, 2), nm.heavy_hex(1, 2)),
        (graphs.all_to_all(5), nm.complete(5)),
        (graphs.linear(4), nm.path(4)),
    ]:
        assert g.to_topology() == t
        assert graphs.from_topology(t) == g
        assert graphs.as_graph(t) == g and graphs.as_graph(g) is g
        assert set(g.edges) == set(t.edge_list())
    env = CodeDiscoveryEnv(n=9, k=1, graph=nm.square(3, 3), batch_size=2)
    assert env.graph == graphs.square(3, 3) and env.topology == nm.square(3, 3)


def test_action_table_respects_graph():
    g = graphs.linear(4)
    table = build_action_table(g, gates=("H", "S", "CNOT", "CZ"))
    two = [(a, b) for gname, a, b in table.entries if gname == "CNOT"]
    assert set(two) == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    cz = [(a, b) for gname, a, b in table.entries if gname == "CZ"]
    assert cz == [(0, 1), (1, 2), (2, 3)]
    assert len(table) == 4 + 4 + 6 + 3


def test_env_step_semantics_and_history():
    env = CodeDiscoveryEnv(n=5, k=1, batch_size=8, max_steps=6, seed=3)
    obs, info = env.reset()
    assert obs.shape == (8, 10, 11)
    assert np.all(info["state_reward"] < 0)
    for _ in range(6):
        acts = env.sample_actions()
        obs, _, term, trunc, info = env.step(acts)
    assert np.all(trunc | term)
    # replaying the history in stim gives the same tableau per element
    for b, gates in enumerate(env.gate_lists()):
        ref = stim.Tableau.from_circuit(gates_to_stim_circuit(gates, 5))
        assert BatchTableau(5, array=env.tab).to_stim(b) == ref


def test_env_terminates_when_code_found():
    # Feed the env the encoding circuit of the [[5,1,3]] code and check it
    # terminates with reward 0 exactly when the code is complete.
    from autoqec_sim import known_codes

    code = known_codes.get("five_qubit")
    circ = code.to_stim_tableau().to_circuit("elimination")
    gates = []
    for inst in circ.flattened():
        tg = [t.value for t in inst.targets_copy()]
        if inst.name in ("H", "S", "S_DAG", "X", "Y", "Z"):
            gates += [(inst.name, q, -1) for q in tg]
        elif inst.name == "CX":
            gates += [("CNOT", tg[i], tg[i + 1]) for i in range(0, len(tg), 2)]
        elif inst.name in ("CZ", "SWAP"):
            gates += [(inst.name, tg[i], tg[i + 1]) for i in range(0, len(tg), 2)]
        elif inst.name == "SQRT_X":
            gates += [(g, q, -1) for q in tg for g in ("H", "S", "H")]
        elif inst.name == "SQRT_X_DAG":
            gates += [(g, q, -1) for q in tg for g in ("H", "S_DAG", "H")]
    env = CodeDiscoveryEnv(n=5, k=1, batch_size=1, gates=("H", "S", "S_DAG", "X", "Y", "Z", "CNOT", "CZ", "SWAP"), max_steps=len(gates) + 1, reward_mode="delta")
    lookup = {e: i for i, e in enumerate(env.action_table.entries)}
    seq = np.array([[lookup[g] for g in gates]])
    reward, success, _ = env.rollout(seq)
    assert success[0] and reward[0] == 0.0
    found = StabilizerCode.from_tableau(env.tab, 1)
    assert found.distance_exact() == 3


def test_auto_reset_reports_final_tableaux():
    env = CodeDiscoveryEnv(n=4, k=1, batch_size=4, max_steps=2, auto_reset=True, seed=0)
    env.reset()
    env.step(env.sample_actions())
    obs, _, term, trunc, info = env.step(env.sample_actions())
    assert trunc.all() or term.any()
    assert "final_tableau" in info and info["final_tableau"].shape[0] == 4
    assert np.all(env.steps == 0)
    assert np.array_equal(obs[0], env.tab[0])


def test_check_matrix_observation_shape():
    env = CodeDiscoveryEnv(n=6, k=2, batch_size=3, observation="check_matrix")
    obs, _ = env.reset()
    assert obs.shape == (3, 4, 12)
    assert env.observation_shape == (4, 12)
