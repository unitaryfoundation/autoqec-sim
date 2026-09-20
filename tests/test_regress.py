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
