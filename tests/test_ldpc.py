import numpy as np
import pytest
from _helpers import gates_from_stim

from autoqec_sim import known_codes, ldpc
from autoqec_sim import symplectic as sp
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.regress import replay
from autoqec_sim.search import EvolutionarySearch
from autoqec_sim.tableau import apply_gate, identity_tableau

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


# ----------------------------------------------------------------------
# LdpcObjective: the four constraint forms
# ----------------------------------------------------------------------
def _three_states():
    return np.concatenate(
        [known_codes.get(n).to_tableau() for n in ("shor", "rotated_surface_9")] + [identity_tableau(9, 1)]
    )


@pytest.mark.parametrize("form", ldpc.LDPC_FORMS)
def test_objective_success_only_on_feasible_codes(form):
    # Shor: distance 3, no idle qubit, w* = 6 (infeasible); rotated surface:
    # feasible; identity: not a code
    tabs = _three_states()
    obj = ldpc.LdpcObjective(9, 1, 3, 4, form=form)
    if form == "schedule":
        obj.activate()  # before activation any clean distance-3 code succeeds (tested below)
    r = obj(tabs, 1)
    ok = obj.success(tabs, 1, r)
    if form == "terminal":
        # every candidate ends the episode; the feasible one scores 1
        assert r[1] == 1.0 and ok.tolist() == [True, True, False]
    else:
        assert r[1] == 0.0 and ok.tolist() == [False, True, False]
    assert obj.feasible(tabs, 1).tolist() == [False, True, True]
    assert obj.r0 == 27  # 3 weight-1 and 3 x 8 weight-2 undetected errors of the initial state
    assert obj.describe()["form"] == form


def test_objective_penalty_and_unnormalized_values():
    tabs = _three_states()
    pen = ldpc.LdpcObjective(9, 1, 3, 4, form="penalty")
    r = pen(tabs, 1)
    assert r[0] == -0.5  # Shor: w* - w_max = 2 at scale 0.25
    assert np.isclose(r[2], -27 / 351 - 8 / 27)  # identity: 27 of 351 errors undetected, 8 idle qubits
    unn = ldpc.LdpcObjective(9, 1, 3, 4, form="unnormalized")
    r = unn(tabs, 1)
    assert np.isclose(r[0], -0.5 / 27) and np.isclose(r[2], -(27 + 8) / 27)
    # one undetected error (1 unit) outweighs one unit of w* excess (0.25)
    assert abs(r[2]) * 27 / 35 == pytest.approx(1.0)
    assert unn.exact_evals == 2  # exact w* only on the two candidate states
    with pytest.raises(ValueError):
        ldpc.LdpcObjective(9, 1, 3, 4, form="soft")


def test_objective_final_and_schedule_forms():
    tabs = _three_states()
    fin = ldpc.LdpcObjective(9, 1, 3, 4, form="final")
    r = fin(tabs, 1)
    assert r[0] == 0.0 and not fin.active  # no LDPC term in the reward, Shor scores 0
    assert fin.success(tabs, 1, r).tolist() == [False, True, False]  # but does not succeed
    sch = ldpc.LdpcObjective(9, 1, 3, 4, form="schedule")
    r = sch(tabs, 1)
    assert r[0] == 0.0 and sch.success(tabs, 1, r).tolist() == [True, True, False]  # distance first
    sch.activate()
    r = sch(tabs, 1)
    assert r[0] == -0.5 and sch.success(tabs, 1, r).tolist() == [False, True, False]


@pytest.mark.parametrize("form", ldpc.LDPC_FORMS)
def test_objective_in_env_rollout(form):
    # the Shor encoder never succeeds under w_max = 4, the surface encoder does
    gate_set = ("H", "S", "S_DAG", "CNOT", "CZ", "SWAP")
    for name, expect in (("shor", False), ("rotated_surface_9", True)):
        gates = gates_from_stim(known_codes.get(name).to_stim_tableau().to_circuit("elimination"))
        obj = ldpc.LdpcObjective(9, 1, 3, 4, form=form)
        if form == "schedule":
            obj.activate()
        if form == "terminal":
            expect = True  # any candidate ends the episode; feasibility is judged afterwards
        env = CodeDiscoveryEnv(
            n=9, k=1, batch_size=1, gates=gate_set, max_steps=len(gates) + 1, reward_fn=obj, success_fn=obj.success
        )
        lookup = {e: i for i, e in enumerate(env.action_table.entries)}
        _, success, _ = env.rollout(np.array([[lookup[g] for g in gates]]))
        assert bool(success[0]) is expect


# ----------------------------------------------------------------------
# sparsity variants: heavy-generator count, shaped and terminal forms, mask
# ----------------------------------------------------------------------
def test_heavy_generator_count_and_terminal_form():
    tabs = _three_states()
    # Shor: the two weight-6 X stabilizers cannot be made light, so 2 of 8
    # generators are heavy; the surface code and the identity have none
    assert ldpc.heavy_generator_count(tabs, 1, 4).tolist() == [2, 0, 0]
    assert ldpc.heavy_generator_count(tabs, 1, 4, exact_max_generators=0).tolist() == [2, 0, 0]  # greedy path
    assert ldpc.heavy_generator_count(tabs, 1, 6).tolist() == [0, 0, 0]
    term = ldpc.LdpcObjective(9, 1, 3, 4, form="terminal")
    r = term(tabs, 1)
    assert r[0] == pytest.approx(1 - 2 / 8) and r[1] == 1.0 and r[2] == pytest.approx(-(27 + 8) / 27)
    assert term.success(tabs, 1, r).tolist() == [True, True, False]  # every candidate ends the episode
    assert term.feasible(tabs, 1).tolist() == [False, True, True]
    assert not term.active and term.describe()["shaping"] is None


def test_shaped_form_prices_excess_in_reward_units():
    tabs = _three_states()
    sh = ldpc.LdpcObjective(9, 1, 3, 4, form="shaped", shaping=0.1)
    r = sh(tabs, 1)
    assert r[0] == pytest.approx(-0.2)  # Shor: excess 2 at c = 0.1, no division by r0
    assert r[1] == 0.0 and r[2] == pytest.approx(-(27 + 8) / 27)  # identity: greedy w* = 1, no excess
    assert sh.success(tabs, 1, r).tolist() == [False, True, False]
    assert sh.describe()["shaping"] == 0.1
    # one unit of excess costs shaping * r0 undetected errors
    assert 0.1 * sh.r0 == pytest.approx(2.7)


def test_greedy_action_mask_forbids_only_heavy_two_qubit_gates():
    env = CodeDiscoveryEnv(n=7, k=1, batch_size=16, max_steps=30, seed=5, gates=("H", "S", "CNOT"))
    env.reset()
    for _ in range(12):
        env.step(env.sample_actions())
    mask = ldpc.greedy_action_mask(env.tab, 1, env.action_table, 3, method="tableau")
    two = env.action_table.q2 >= 0
    assert mask.shape == (16, env.num_actions) and mask[:, ~two].all()  # single-qubit gates always allowed
    assert not mask[:, two].all()  # some CNOT raises the bound above 3 after 12 random gates
    # an allowed two-qubit action leads to a state with greedy w* <= 3, a forbidden one above
    for a in np.flatnonzero(two):
        trial = env.tab.copy()
        env.action_table.apply(trial, np.full(16, a))
        w, _ = ldpc._greedy_stats(ldpc._generators_packed(trial, 1), 7)
        assert np.array_equal(mask[:, a], w <= 3)


def test_greedy_action_mask_basis_method_is_sound_and_matches_tableau_method():
    env = CodeDiscoveryEnv(n=7, k=1, batch_size=48, max_steps=30, seed=11, gates=("H", "S", "CNOT"))
    env.reset()
    for _ in range(10):
        env.step(env.sample_actions())
    basis = ldpc.greedy_action_mask(env.tab, 1, env.action_table, 3, method="basis")
    tableau = ldpc.greedy_action_mask(env.tab, 1, env.action_table, 3, method="tableau")
    two = np.flatnonzero(env.action_table.q2 >= 0)
    assert basis[:, env.action_table.q2 < 0].all()  # single-qubit gates always allowed
    for a in two:  # both methods are sound against the exact test
        trial = env.tab.copy()
        env.action_table.apply(trial, np.full(48, a))
        exact = ldpc.ldpc_feasible(trial, 1, 3)
        assert (~basis[:, a] | exact).all() and (~tableau[:, a] | exact).all()
    assert (basis == tableau).mean() > 0.95
    # the packed CNOT is the tableau CNOT on the generators
    g = ldpc._generators_packed(env.tab, 1)
    t2 = env.tab.copy()
    apply_gate(t2, "CNOT", 2, 5)
    assert np.array_equal(ldpc._packed_cnot(g, 7, 2, 5), ldpc._generators_packed(t2, 1))


def _masked_rollout(n, k, graph, gates, w_max, steps, batch, seed):
    """States of a random policy under the greedy mask (sparse, so the
    greedy bound rejects a fair share of the two-qubit actions)."""
    from autoqec_sim import graphs

    env = CodeDiscoveryEnv(n=n, k=k, batch_size=batch, max_steps=steps + 5, seed=seed, gates=gates, graph=graphs.from_spec(graph))
    env.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        m = ldpc.greedy_action_mask(env.tab, k, env.action_table, w_max, method="basis")
        env.step(np.where(m, rng.random(m.shape), -1.0).argmax(axis=1))
    return env


@pytest.mark.parametrize("n, k, graph, gates, w_max, steps", [(7, 1, "all_to_all:7", ("H", "S", "CNOT"), 3, 10), (12, 2, "all_to_all:12", ("H", "S", "CNOT"), 4, 25), (16, 4, "square:4x4", ("H", "CNOT"), 4, 30)])
def test_exact_action_mask_equals_exact_feasibility_and_extends_greedy(n, k, graph, gates, w_max, steps):
    env = _masked_rollout(n, k, graph, gates, w_max, steps, 48, 3)
    cache = ldpc.ExactMaskCache()
    st = {}
    exact = ldpc.greedy_action_mask(env.tab, k, env.action_table, w_max, method="exact", exact_max_generators=15, cache=cache, stats=st)
    greedy = ldpc.greedy_action_mask(env.tab, k, env.action_table, w_max, method="basis")
    two = np.flatnonzero(env.action_table.q2 >= 0)
    assert exact[:, env.action_table.q2 < 0].all()
    assert st["method"] == "exact" and st["pairs"] == two.size * 48 and st["exact_tested"] == st["greedy_rejected"] == int((~greedy[:, two]).sum())
    opened = 0
    for a in two:
        trial = env.tab.copy()
        env.action_table.apply(trial, np.full(48, a))
        feasible = ldpc.ldpc_feasible(trial, k, w_max, exact_max_generators=15)
        assert np.array_equal(exact[:, a], feasible)  # never masks a feasible move, never admits an infeasible one
        assert (~greedy[:, a] | exact[:, a]).all()  # agrees with the greedy bound wherever the bound admits
        opened += int((~greedy[:, a] & exact[:, a] & feasible).sum())
    assert opened == st["exact_admitted"] > 0  # some move the greedy bound forbids is feasible
    # a second call on the same states hits the cache for every state
    st2 = {}
    again = ldpc.greedy_action_mask(env.tab, k, env.action_table, w_max, method="exact", exact_max_generators=15, cache=cache, stats=st2)
    assert np.array_equal(again, exact) and st2["cache_hits"] == 48 and st2["cache_misses"] == 0


def test_exact_action_mask_non_cnot_gates_and_out_of_range_fallback():
    env = CodeDiscoveryEnv(n=6, k=1, batch_size=24, max_steps=30, seed=4, gates=("H", "CNOT", "CZ"))
    env.reset()
    for _ in range(8):
        env.step(env.sample_actions())
    st = {}
    exact = ldpc.greedy_action_mask(env.tab, 1, env.action_table, 3, method="exact", stats=st)
    assert st["method"] == "exact"
    for a in np.flatnonzero(env.action_table.q2 >= 0):
        trial = env.tab.copy()
        env.action_table.apply(trial, np.full(24, a))
        assert np.array_equal(exact[:, a], ldpc.ldpc_feasible(trial, 1, 3))
    # above the exact range the method is the greedy basis bound and says so
    st = {}
    fallback = ldpc.greedy_action_mask(env.tab, 1, env.action_table, 3, method="exact", exact_max_generators=4, stats=st)
    assert st["method"] == "tableau" and st["exact_tested"] == 0  # CZ in the gate set: the tableau-copy bound
    assert np.array_equal(fallback, ldpc.greedy_action_mask(env.tab, 1, env.action_table, 3, method="tableau"))
