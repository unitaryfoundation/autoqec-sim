import numpy as np
import pytest
from conftest import gates_from_stim

from autoqec_sim import known_codes, ldpc
from autoqec_sim import symplectic as sp
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.regress import replay
from autoqec_sim.search import EvolutionarySearch
from autoqec_sim.tableau import identity_tableau

# (code, minimum bottleneck weight w*, maximum qubit degree of the basis)
EXPECTED = {
    "five_qubit": (4, 4),
    "steane": (4, 6),
    "shor": (6, 4),
    "rotated_surface_9": (4, 4),
    "four_two_two": (4, 2),
}


def test_pack_unpack_and_weight():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2, size=(5, 18), dtype=np.uint8)
    packed = ldpc.pack_rows(rows)
    assert np.array_equal(ldpc.unpack_rows(packed, 18), rows)
    assert np.array_equal(ldpc.packed_weight(packed, 9), sp.pauli_weight(rows))


def test_batched_rank_matches_rref():
    rng = np.random.default_rng(1)
    b, r, c = 6, 12, 14
    rows = rng.integers(0, 2, size=(b, r, c), dtype=np.uint8)
    mask = rng.random((b, r)) < 0.7
    rank, piv = ldpc.batched_rank_packed(ldpc.pack_rows(rows), mask, c)
    for i in range(b):
        sel = rows[i][mask[i]]
        assert rank[i] == (sp.rank(sel) if sel.shape[0] else 0)
        # the original rows at the pivot indices are a basis of the span
        p = piv[i][piv[i] >= 0]
        assert p.size == rank[i]
        assert sp.rank(rows[i][p]) == rank[i]
        assert mask[i][p].all()


@pytest.mark.parametrize("name", list(EXPECTED))
def test_known_code_sparsest_basis(name):
    code = known_codes.get(name)
    tab = code.to_tableau()
    st = ldpc.sparsest_basis_stats(tab, code.k)
    w_star, deg = EXPECTED[name]
    assert st["exact"]
    assert int(st["w_star"][0]) == w_star
    assert int(st["max_degree"][0]) == deg
    basis = ldpc.sparsest_basis_rows(tab, code.k)
    assert basis.shape == code.stabilizers.shape
    assert sp.rank(basis) == code.n - code.k
    assert sp.pauli_weight(basis).max() == w_star
    # every basis row is a stabilizer element
    for row in basis:
        assert sp.in_rowspace(row, code.stabilizers)
    # the greedy fallback gives an upper bound
    w_greedy, _ = ldpc._greedy_stats(ldpc._generators_packed(tab, code.k), code.n)
    assert w_greedy[0] >= w_star


def test_feasibility_bounds():
    five = known_codes.get("five_qubit").to_tableau()
    steane = known_codes.get("steane").to_tableau()
    shor = known_codes.get("shor").to_tableau()
    surface = known_codes.get("rotated_surface_9").to_tableau()
    assert ldpc.ldpc_feasible(five, 1, 4)[0] and not ldpc.ldpc_feasible(five, 1, 3)[0]
    assert ldpc.ldpc_feasible(steane, 1, 4)[0] and not ldpc.ldpc_feasible(steane, 1, 3)[0]
    nine = np.concatenate([shor, surface, identity_tableau(9, 1)])
    assert ldpc.ldpc_feasible(nine, 1, 4).tolist() == [False, True, True]
    assert ldpc.ldpc_feasible(nine, 1, 6).tolist() == [True, True, True]
    # the exact check and the greedy fallback agree on these codes
    assert ldpc.ldpc_feasible(nine, 1, 4, exact_max_generators=0).tolist() == [False, True, True]


def test_penalty_values():
    tabs = np.concatenate(
        [known_codes.get(n).to_tableau() for n in ("shor", "rotated_surface_9")] + [identity_tableau(9, 1)]
    )
    pen = ldpc.ldpc_penalty(tabs, 1, w_max=4)
    assert pen.tolist() == [-2.0, 0.0, 0.0]
    pen = ldpc.ldpc_penalty(tabs, 1, w_max=4, deg_max=3, scale=0.5)
    assert pen.tolist() == [-0.5 * (2 + 1), -0.5 * 1, 0.0]
    f = ldpc.make_ldpc_penalty(3)
    assert f(tabs, 1).tolist() == [-3.0, -1.0, 0.0]


def test_exact_and_greedy_on_random_states():
    env = CodeDiscoveryEnv(n=8, k=1, batch_size=64, max_steps=12, seed=5)
    env.reset()
    for _ in range(12):
        env.step(env.sample_actions())
    st = ldpc.sparsest_basis_stats(env.tab, 1)
    w_greedy, _ = ldpc._greedy_stats(ldpc._generators_packed(env.tab, 1), 8)
    assert np.all(w_greedy >= st["w_star"])
    for w in range(1, 9):
        assert np.array_equal(ldpc.ldpc_feasible(env.tab, 1, w), st["w_star"] <= w)
    # exhaustive check of w* on a few elements: minimum over all bases of
    # the maximum weight equals the threshold at which the light elements
    # span the group
    for b in range(4):
        code = replay(env.gate_lists()[b], 8)
        gens = code.stabilizers
        elems = np.array(next(r for r, _ in code._group_elements(gens)))[1:]
        wts = sp.pauli_weight(elems)
        w_star = min(w for w in range(1, 9) if sp.rank(elems[wts <= w]) == gens.shape[0])
        assert w_star == st["w_star"][b]


def test_env_terminate_mode():
    # Shor's encoder needs a weight-6 generator at the end; with w_max = 4
    # in terminate mode the episode ends with the violation reward and no
    # success.
    code = known_codes.get("shor")
    gates = gates_from_stim(code.to_stim_tableau().to_circuit("elimination"))
    gate_set = ("H", "S", "S_DAG", "CNOT", "CZ", "SWAP")
    env = CodeDiscoveryEnv(n=9, k=1, batch_size=1, gates=gate_set, max_steps=len(gates) + 1, ldpc_w_max=4, ldpc_mode="terminate")
    lookup = {e: i for i, e in enumerate(env.action_table.entries)}
    seq = np.array([[lookup[g] for g in gates]])
    reward, success, _ = env.rollout(seq)
    assert not success[0] and reward[0] == env.ldpc_violation_reward
    # without the constraint the same sequence succeeds
    env0 = CodeDiscoveryEnv(n=9, k=1, batch_size=1, gates=gate_set, max_steps=len(gates) + 1)
    reward, success, _ = env0.rollout(seq)
    assert success[0] and reward[0] == 0.0
    # with w_max = 6 the constraint is inactive on this path
    env6 = CodeDiscoveryEnv(n=9, k=1, batch_size=1, gates=gate_set, max_steps=len(gates) + 1, ldpc_w_max=6, ldpc_mode="terminate")
    reward, success, _ = env6.rollout(seq)
    assert success[0]


def test_env_mask_mode_reverts_and_matches_history():
    env = CodeDiscoveryEnv(n=7, k=1, batch_size=32, max_steps=15, seed=2, ldpc_w_max=3, ldpc_mode="mask", gates=("H", "CNOT"))
    env.reset()
    masked_any = False
    for _ in range(15):
        acts = env.sample_actions()
        mask = env.feasible_action_mask()
        _, _, _, _, info = env.step(acts)
        predicted = ~mask[np.arange(32), acts]
        assert np.array_equal(info["ldpc_violation"], predicted)
        masked_any |= bool(predicted.any())
        assert env.ldpc_feasible().all()
    assert masked_any
    # the recorded history (masked actions dropped) replays to the state
    from autoqec_sim.tableau import BatchTableau

    for b, gates in enumerate(env.gate_lists()):
        bt = BatchTableau(7, 1)
        for g, a, c in gates:
            bt.apply(g, a, c if c >= 0 else None)
        assert np.array_equal(bt.array[0], env.tab[b])
        assert ldpc.ldpc_feasible(env.tab[b : b + 1], 1, 3)[0]


def test_search_reports_executed_gates_under_mask():
    env = CodeDiscoveryEnv(n=7, k=1, batch_size=64, max_steps=12, seed=3, ldpc_w_max=4, ldpc_mode="mask", gates=("H", "CNOT"))
    res = EvolutionarySearch(env, seed=3).run(64 * 3, stop_on_success=False)
    code = replay(res.gates, 7)
    assert ldpc.ldpc_feasible(code.to_tableau(), 1, 4)[0]
    assert len(res.gates) <= 12
