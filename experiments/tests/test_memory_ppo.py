"""PPO with the memory objective (``experiments.memory_ppo``) on the CPU: the
reward's state terms agree with the numpy path and its terminal terms with
``MemoryReward`` and ``ldpc_penalty``; the spawn pool returns the inline
objective's value and caches; a tiny cell runs end to end and the report
builds from its JSON. Skipped without torch."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autoqec_sim.errors import paulis_of_weight
from autoqec_sim.ldpc import ldpc_penalty
from autoqec_sim.regress import replay
from autoqec_sim.reward import batched_distance, low_weight_stabilizer_penalty, undetected_mask
from experiments.memory_ppo import run as mp
from experiments.noise_aware import run as na


def _surface_tableau():
    stabs = na.known_codes_for(mp.SETTING)["rotated_surface"]["stabilizers"]
    gates = na.css_standard_encoder(stabs)
    code = replay(gates, mp.SETTING.n)
    return code, code.to_tableau()


def test_state_terms_match_numpy():
    env = na.make_env(mp.SETTING, seed=3, batch=16)
    env.reset()
    rng = np.random.default_rng(3)
    for _ in range(20):
        env.step(rng.integers(0, env.num_actions, size=env.batch_size))
    tab = env.tab.copy()
    pool = mp.MemoryTermPool("ion", mp.SETTING.d, workers=0)
    reward = mp.TorchMemoryReward(SimpleNamespace(batch_size=-1), mp.SETTING, pool, device="cpu")
    r = reward(tab, 1)
    d = batched_distance(tab, 1, mp.SETTING.d - 1)
    ref = np.minimum(d, 4) / 4.0 - 1.0
    for w in range(1, 4):
        idx = np.flatnonzero(d == w)
        if idx.size:
            frac = 1.0 - undetected_mask(tab[idx], 1, paulis_of_weight(16, w)).mean(axis=1)
            ref[idx] = (w - 1 + frac) / 4.0 - 1.0
    ref += low_weight_stabilizer_penalty(tab, 1, 1)
    assert np.allclose(r, ref)
    assert pool.evals == 0


def test_terminal_terms_on_the_surface_code():
    code, tab = _surface_tableau()
    pool = mp.MemoryTermPool("ion", mp.SETTING.d, workers=0)
    reward = mp.TorchMemoryReward(SimpleNamespace(batch_size=1), mp.SETTING, pool, device="cpu")
    r = reward(tab, 1)
    p = mp.make_objective("ion").evaluate(code, 4).p_round
    ldpc = ldpc_penalty(tab, 1, na.W_MAX, scale=na.LDPC_SCALE, exact_max_generators=na.EXACT_GENERATORS)[0]
    assert ldpc == 0.0
    assert r[0] == pytest.approx(mp.memory_term(p))
    assert reward.success_fn(tab, 1, r).tolist() == [True]
    assert pool.evals == 1 and pool.hits == 0
    reward(tab, 1)
    assert pool.evals == 1 and pool.hits == 1
    assert 0 < p < 1e-2  # the surface code is better than p_ref under the chain profile


def test_spawn_pool_matches_inline():
    code, tab = _surface_tableau()
    key = na.code_key(tab[0], 1)
    inline = mp.make_objective("board").evaluate(code, 4)
    pool = mp.MemoryTermPool("board", mp.SETTING.d, workers=1)
    try:
        res = pool.evaluate({key: code})[key]
        again = pool.evaluate({key: code})[key]
    finally:
        pool.close()
    assert res["p_round"] == pytest.approx(inline.p_round, rel=1e-9)
    assert res["per_basis"]["Z"]["mechanisms"] == inline.per_basis["Z"]["mechanisms"]
    assert again is res and pool.evals == 1 and pool.hits == 1


def test_tiny_cell_and_report(tmp_path):
    rec = mp.run_one("heron", 0, tmp_path, device="cpu", batch=8, steps=320, wall_min=2.0, threads=1, pool_workers=0)
    for k in (
        "kind",
        "profile",
        "objective",
        "reward",
        "env_steps",
        "memory_evals",
        "codes",
        "history",
        "best",
        "success",
        "hit_wall_cap",
    ):
        assert k in rec
    assert (
        rec["objective"]["profile"].startswith("heron") and rec["objective"]["scale"] == mp.PROFILES["heron"]["scale"]
    )
    assert (tmp_path / "run_heron_seed0.json").exists()
    path = mp.write_report(tmp_path, tmp_path / "report", mp.ENCODER_DIR)
    text = path.read_text()
    assert "## Runs" in text and "run_heron_seed0.json" in text


def test_run2_variant_terms_and_sampled_fallback():
    """Run-2 reward: no LDPC term, gate-count term above ``gate_ref``, and
    the sampled tier replacing the proxy when the expected faults exceed
    the threshold (forced here with a threshold of 0 and a tiny budget)."""
    code, tab = _surface_tableau()
    assert mp.sparsest_code(code).stabilizers.sum() < code.stabilizers.sum()  # the tableau basis is denser
    pool = mp.MemoryTermPool(
        "ion",
        mp.SETTING.d,
        workers=0,
        sampled_above=0.0,
        sampled_sampling={"shots": 500, "min_failures": 0, "max_shots": 500},
        sparsify=True,
    )
    reward = mp.TorchMemoryReward(
        SimpleNamespace(batch_size=1), mp.SETTING, pool, device="cpu", ldpc_scale=0.0, gate_ref=100, gate_weight=0.2
    )
    r = reward(tab, 1)
    key = na.code_key(tab[0], 1)
    res = pool.cache[key]
    assert res["tier_used"] == "sampled" and res["tier"] == "sampled" and pool.sampled_evals == 1
    assert res["sparsified"] is True
    assert 0 < res["proxy_p_round"] < 1e-2 and res["expected_faults_max"] > 0
    g = res["per_basis"]["Z"]["two_qubit_gates"]
    assert g == mp.SURFACE_2Q_GATES
    assert r[0] == pytest.approx(mp.memory_term(res["p_round"]) - 0.2 * (g / 100 - 1))
    d = mp.reward_description("run2")
    assert d["ldpc_scale"] == 0.0 and d["gate_ref"] == mp.GATE_REF and d["selection"] == "lowest training p_round"
