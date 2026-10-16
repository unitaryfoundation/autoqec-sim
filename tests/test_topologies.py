"""Three-topology run matrix: lattice resolution, the rotated surface
baseline, and a one-cell PPO run with a tiny budget. Skipped without torch."""

import pytest

torch = pytest.importorskip("torch")

from autoqec_sim.experiments import topologies as T


def test_resolve_graph_picks_nearest_available_lattice():
    assert T.resolve_graph("square", 9).name == "square:3x3"
    assert T.resolve_graph("triangular", 16).name == "triangular:4x4"
    # heavy-hex sizes are fixed by the honeycomb construction: 12, 18, 21, ...
    small = T.resolve_graph("heavy_hex", 9)
    large = T.resolve_graph("heavy_hex", 16)
    assert (small.name, small.n, max(small.degree())) == ("heavy_hex:1x1", 12, 2)
    assert (large.name, large.n, max(large.degree())) == ("heavy_hex:2x1", 18, 3)


def test_surface_baseline_square_3x3():
    graph = T.resolve_graph("square", 9)
    models = T.noise_models(graph)
    assert not models["_refused"]
    base = T.surface_baseline(graph, models)
    assert base["d"] == 3
    for name in T.PROFILES:
        best = base["best_by_profile"][name]
        assert 0.0 < best["undetected"] < best["total"]
        assert all(graph.has_edge(a, b) for g, a, b in best["gates"] if g == "CNOT")
    assert T.surface_baseline(T.resolve_graph("heavy_hex", 9), models) is None


def test_one_cell_tiny_budget():
    rec = T.run_cell("square", 9, seed=0, max_env_steps=4096, minutes=0.5, threads=1, batch=32)
    assert rec["kind"] == "run"
    assert rec["graph"]["spec"] == "square:3x3" and rec["n"] == 9
    assert rec["ppo_config"]["policy_resolved"] == "mlp"
    assert rec["env_steps"] >= 4096 or rec["success"]
    assert rec["threads"] == 1
    code = rec["best_code"]
    assert code is not None
    assert code["n"] == 9 and code["k"] == 1
    assert len(code["weight_enumerator_S"]) == 10
    assert set(code["mass"]) == set(T.PROFILES)
    for g, a, b in code["gates"]:
        if g == "CNOT":
            assert T.resolve_graph("square", 9).has_edge(a, b)
    if rec["success"]:
        assert code["d"] >= 3
        assert rec["first_success"]["env_steps"] <= rec["env_steps"]
