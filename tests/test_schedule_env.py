# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Schedule-synthesis environment: feasibility masks, layer export, and
scoring against the board's own circuit pipeline (skipped when no
qldpc-challenge checkout is reachable)."""
import numpy as np
import pytest

from autoqec_sim import schedule_env as se

try:
    se._qldpc_root()
    HAVE_BOARD = True
except ImportError:
    HAVE_BOARD = False

pytestmark = pytest.mark.skipif(not HAVE_BOARD, reason="needs a qldpc-challenge checkout")


def _steane():
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.int8)
    return se.BoardCode(slug="7-1-3", n=7, k=1, d=3, HX=H.copy(), HZ=H.copy())


def test_masks_give_matchings_and_layers_cover_every_edge():
    code = _steane()
    # slack 3: a random greedy assignment can dead-end at the chromatic bound
    env = se.ScheduleEnv(code, batch_size=3, slack=3, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (3, env.obs_dim)
    rng = np.random.default_rng(1)
    while True:
        mask = env.action_mask()
        assert mask.any(axis=1).all()
        a = np.array([rng.choice(np.flatnonzero(mask[b])) for b in range(3)])
        obs, _r, done, _info = env.step(a)
        if done.all():
            break
    assert not env.infeasible.any()
    for b in range(3):
        layers = env.layers(b)
        seen = set()
        for layer in layers:
            checks = [(k, c) for k, c, _ in layer]
            qubits = [q for _, _, q in layer]
            assert len(set(checks)) == len(checks), "a check appears twice in one layer"
            assert len(set(qubits)) == len(qubits), "a data qubit appears twice in one layer"
            seen.update(layer)
        assert seen == set(env.edges)
        # X block strictly before Z block without interleave
        kinds = ["".join(sorted({k for k, _, _ in layer})) for layer in layers]
        assert "XZ" not in kinds
        first_z = next((i for i, k in enumerate(kinds) if k == "Z"), len(kinds))
        assert all(k == "Z" for k in kinds[first_z:])


def test_scoring_matches_the_board_pipeline_on_steane():
    code = _steane()
    base = se.coloring_baseline(code, samples=2, seed=0, rounds=2, tier="exact", exact_wmax=4)
    for s in base:
        assert s["valid"]
        assert set(s["d_circ"]) == {"Z", "X"}
        assert 1 <= s["min"] <= 3
    env = se.ScheduleEnv(code, batch_size=2, rounds=2, tier="exact", seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    while True:
        mask = env.action_mask()
        a = np.array([rng.choice(np.flatnonzero(mask[b])) for b in range(2)])
        _, r, done, info = env.step(a)
        if done.all():
            break
    for s, rew in zip(info["scores"], r):
        assert s["valid"]
        count = s["count_min"]
        expected = (min(s["min"], code.d) + 1.0 / (1.0 + count)) / (code.d + 1.0) \
            - env.depth_penalty * max(0, s["layers"] - env.lower_bound_layers)
        assert abs(float(rew) - expected) < 1e-6
        # a schedule one distance level up always beats one with any count below it
        assert (min(s["min"], code.d) + 1) / (code.d + 1.0) > (min(s["min"], code.d) + 1.0 / (1.0 + count)) / (code.d + 1.0)


def test_infeasible_action_is_recorded_and_penalized():
    code = _steane()
    env = se.ScheduleEnv(code, batch_size=1, slack=0, seed=0)
    env.reset()
    # force every edge into slot 0: the second edge of the first check conflicts
    while True:
        _, r, done, _info = env.step(np.zeros(1, dtype=np.int64))
        if done.all():
            break
    assert env.infeasible[0]
    assert float(r[0]) == env.invalid_reward
