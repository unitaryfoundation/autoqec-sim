"""Reduced-budget rediscovery regression (target: under two minutes)."""

import time

import pytest

from autoqec_sim import regress


@pytest.fixture(scope="module")
def reduced_results():
    t0 = time.perf_counter()
    res = regress.run_all("reduced", seed=0)
    res["_wall"] = time.perf_counter() - t0
    return res


@pytest.mark.parametrize("name", [n for n, t in regress.TARGETS.items() if t.expect_success])
def test_rediscovers_target(reduced_results, name):
    r = reduced_results["targets"][name]
    assert r["passed"] and r["rediscovered"], r["attempts"]
    v = r["result"]["verification"]
    assert v["parameters_match"] and v["nontrivial"]
    assert v["distance_exact"] == 3
    spec = regress.TARGETS[name].ldpc
    if spec is not None:
        assert v["ldpc_bound_met"]
        assert v["min_max_generator_weight"] <= spec.w_max
        if spec.deg_max is not None:
            assert v["max_qubit_degree"] <= spec.deg_max


@pytest.mark.parametrize("name", [n for n, t in regress.TARGETS.items() if not t.expect_success])
def test_ldpc_bound_excludes_dense_codes(name):
    # w_max = 3: no distance-3 code on 5 qubits (a theorem: the [[5,1,3]]
    # code is unique and all its stabilizers have weight 4) and none on 7
    # qubits within the reduced budget (Steane and the other [[7,1,3]]
    # family found by the search have w* = 4).
    r = regress.run_target(regress.TARGETS[name], seed=0, **regress.BUDGETS["reduced"])
    assert r["passed"] and not r["rediscovered"], r["attempts"]
    assert all(not a["search_success"] for a in r["attempts"])


def test_five_qubit_code_is_unique(reduced_results):
    # The [[5,1,3]] code is unique up to equivalence, so its enumerators
    # must match.
    v = reduced_results["targets"]["five_qubit"]["result"]["verification"]
    assert v["enumerators_match_reference"]


def test_reduced_budget_is_quick(reduced_results):
    assert reduced_results["_wall"] < 120


def test_objective_mode_targets_at_n9():
    # LdpcSpec(mode="objective") wires an LdpcObjective as reward and success
    # criterion; at n = 9 every form finds a sparse [[9,1,3]] code quickly
    import pytest

    from autoqec_sim import ldpc
    from autoqec_sim.regress import LdpcSpec, Target, env_kwargs, make_reward, run_target

    with pytest.raises(ValueError):
        LdpcSpec(4, mode="soft")
    with pytest.raises(ValueError):
        LdpcSpec(4, mode="objective", form="soft")
    for form in ldpc.LDPC_FORMS:
        t = Target(
            f"surface_objective_{form}", 9, 1, 3, "square:3x3", "rotated_surface_9", 30, 1, ("H", "CNOT"),
            ldpc=LdpcSpec(w_max=4, mode="objective", form=form), require_reference_match=False,
        )
        reward = make_reward(t)
        assert isinstance(reward, ldpc.LdpcObjective) and reward.form == form
        assert env_kwargs(t, reward) == {"success_fn": reward.success}
        with pytest.raises(ValueError):
            env_kwargs(t)
        if form == "schedule":
            # the switch is the trainer's job (experiments.ldpc_scaling); the
            # evolutionary search has none, so only the wiring is checked here
            continue
        r = run_target(t, seed=0, batch_size=256, max_evaluations=30_000, max_attempts=2)
        assert r["rediscovered"], (form, r["attempts"])
        v = r["result"]["verification"]
        assert v["distance_exact"] == 3 and v["nontrivial"] and v["ldpc_bound_met"] and v["min_max_generator_weight"] <= 4
