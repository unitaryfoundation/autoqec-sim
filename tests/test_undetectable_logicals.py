"""The cluster search for undetectable fault sets beyond weight 4 and the
weight-3 proxy it gives at circuit distance 5: completeness against
exhaustive enumeration on small sub-models, exact agreement with the
triple enumeration, and the escalation in the memory objective and in
`evaluate_code`."""

import itertools
import math

import numpy as np
import pytest

from autoqec_sim import known_codes
from autoqec_sim import memory_objective as mo
from autoqec_sim.code import StabilizerCode
from autoqec_sim.evaluate import evaluate_code
from autoqec_sim.noise import presets

SURFACE5 = known_codes.rotated_surface(5)


def _surface5_orders(z_major="col"):
    """The standard hook-safe order (X checks row-major, Z checks
    column-major); ``z_major="row"`` is the hook-vulnerable all-row-major
    order, whose X-basis circuit distance drops to 3."""
    orders = []
    for s in SURFACE5:
        sup = [(q, ch) for q, ch in enumerate(s) if ch != "I"]
        if sup[0][1] == "X" or z_major == "row":
            orders.append(sorted(sup, key=lambda qc: (qc[0] // 5, qc[0] % 5)))
        else:
            orders.append(sorted(sup, key=lambda qc: (qc[0] % 5, qc[0] // 5)))
    return orders


def _surface5(basis, rounds=1, z_major="col"):
    code = StabilizerCode.from_stabilizers(SURFACE5)
    mc = mo.build_memory_circuit(code, rounds, basis, hook_target=5, orders=_surface5_orders(z_major))
    dem = mo.memory_dem(mc, presets.chain(), 0.25)
    return mc, dem, mo.dem_signature(dem)


def _sub_signature(sig, indices):
    """The DEM restricted to the mechanisms ``indices`` (a sub-model has
    the same or a larger circuit distance), with the index map."""
    idx = np.array(sorted({int(i) for i in indices}), dtype=np.int64)
    sub = mo.Signature(sig.det[idx], sig.obs[idx], sig.probs[idx], sig.num_detectors, sig.num_observables)
    return sub, idx


def _brute_force(sig, wmax):
    """Every undetectable logical set of at most ``wmax`` mechanisms of a
    small DEM, as ``{weight: {sorted indices: probability}}``."""
    m = len(sig)
    out = {}
    for w in range(1, wmax + 1):
        n = math.comb(m, w)
        flat = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(m), w)), dtype=np.int64, count=n * w)
        c = flat.reshape(n, w)
        d, o = sig.det[c[:, 0]].copy(), sig.obs[c[:, 0]].copy()
        for k in range(1, w):
            d ^= sig.det[c[:, k]]
            o ^= sig.obs[c[:, k]]
        hit = c[(~d.any(axis=1)) & (o != 0)]
        if len(hit):
            out[w] = {tuple(int(v) for v in row): float(np.prod(sig.probs[row])) for row in hit}
    return out


def _as_dict(found):
    return {w: {tuple(int(v) for v in row): float(p) for row, p in zip(sets, probs, strict=True)} for w, (sets, probs) in found.items()}


def _check_sets(sig, found, weights):
    """Every listed set flips no detector and some observable, and has no
    undetectable proper subset of one or two mechanisms."""
    assert sorted(found) == sorted(weights)
    for w, (sets, probs) in found.items():
        assert sets.shape == (len(probs), w) and (np.diff(sets, axis=1) > 0).all()
        for row, p in zip(sets, probs, strict=True):
            assert not np.bitwise_xor.reduce(sig.det[row], axis=0).any()
            assert np.bitwise_xor.reduce(sig.obs[row]) != 0
            assert p == pytest.approx(float(np.prod(sig.probs[row])))
            assert not (sig.det[row] == 0).all(axis=1).any()
            for a, b in itertools.combinations(row, 2):
                assert (sig.det[a] != sig.det[b]).any()


def test_search_matches_exhaustive_enumeration_at_distance_three():
    # the hook-ordered 3x3 surface code at one round has circuit distance 3,
    # so every logical set of weight at most 5 is irreducible
    code = known_codes.get("rotated_surface_9")
    mc = mo.build_memory_circuit(code, 1, "X")
    sig = mo.dem_signature(mo.memory_dem(mc, presets.chain(), 0.25))
    full = mo.undetectable_logicals(sig, 4)
    _check_sets(sig, full, [3, 4])
    assert mo.exact_dcirc(sig, 4)["count"] == len(full[3][0])
    rng = np.random.default_rng(0)
    keep = set(full[3][0][:5].ravel().tolist()) | set(full[4][0][rng.choice(len(full[4][0]), 5, replace=False)].ravel().tolist())
    keep |= set(rng.choice(len(sig), 30, replace=False).tolist())
    sub, idx = _sub_signature(sig, keep)
    assert mo.exact_dcirc(sub, 4)["d_circ"] == 3
    found = _as_dict(mo.undetectable_logicals(sub, 4))
    brute = _brute_force(sub, 4)
    assert found.keys() == brute.keys() == {3, 4}
    for w in brute:
        assert found[w].keys() == brute[w].keys()
        assert all(found[w][s] == pytest.approx(brute[w][s]) for s in brute[w])
    # the sub-model lists exactly the sets of the full model that it contains
    pos = {int(v): i for i, v in enumerate(idx)}
    for w in (3, 4):
        inside = {tuple(pos[int(v)] for v in row) for row in full[w][0] if all(int(v) in pos for v in row)}
        assert inside == found[w].keys()


def test_search_at_distance_five_matches_exhaustive_enumeration_and_triples():
    _, _, sig = _surface5("X")
    assert mo.exact_dcirc(sig, 4)["d_circ"] is None and mo.pfail_low_weight(sig, 2) == 0.0
    dc = mo.exact_dcirc(sig, 6)
    full = mo.undetectable_logicals(sig, 6)
    _check_sets(sig, full, [5, 6])
    assert dc["d_circ"] == 5 and dc["count"] == len(full[5][0]) and dc["mass"] == pytest.approx(full[5][1].sum())
    assert sorted(dc["witness"]) == list(full[5][0][int(np.argmax(full[5][1]))])
    us = mo.undetectable_sets(sig, 6, limit=4)
    assert us["weight"] == 5 and us["count"] == dc["count"] and len(us["sets"]) == 4
    assert us["sets"][0] == (tuple(dc["witness"]), pytest.approx(full[5][1].max()))
    # a sub-model small enough for exhaustive enumeration to weight 5
    rng = np.random.default_rng(1)
    order5 = np.argsort(-full[5][1])
    keep = set(full[5][0][order5[:8]].ravel().tolist()) | set(rng.choice(len(sig), 15, replace=False).tolist())
    sub, idx = _sub_signature(sig, keep)
    found = _as_dict(mo.undetectable_logicals(sub, 5))
    brute = _brute_force(sub, 5)
    assert 5 in brute and found.keys() == brute.keys()
    for w in brute:
        assert found[w].keys() == brute[w].keys()
    pos = {int(v): i for i, v in enumerate(idx)}
    inside = {tuple(pos[int(v)] for v in row) for row in full[5][0] if all(int(v) in pos for v in row)}
    assert inside == found[5].keys()
    # a larger sub-model: the escalated proxy equals the maximum-likelihood
    # failure over every triple, in both weightings
    keep = set(full[5][0][order5[:10]].ravel().tolist()) | set(full[6][0][np.argsort(-full[6][1])[:10]].ravel().tolist())
    keep |= set(rng.choice(len(sig), 50, replace=False).tolist())
    sub, _ = _sub_signature(sig, keep)
    assert len(sub) < 140
    pr = mo._proxy_failure(sub, None, 2_000_000, False)
    assert pr["weight"] == 3 and pr["d_circ"]["d_circ"] == 5 and pr["p_fail"] > 0
    assert pr["p_fail"] == pytest.approx(mo.pfail_low_weight(sub, 3, max_triples=10**7), rel=1e-9)
    assert mo.pfail_low_weight(sub, None, exclusive=True) == pytest.approx(mo.pfail_low_weight(sub, 3, max_triples=10**7, exclusive=True), rel=1e-9)
    assert mo.pfail_low_weight(sub, 2) == 0.0


def test_reducible_sets_are_dropped_and_sub_models_cannot_lose_distance():
    # the all-row-major order is hook-vulnerable: the X-basis distance is 3
    _, _, sig = _surface5("X", z_major="row")
    dc = mo.exact_dcirc(sig, 4)
    found = mo.undetectable_logicals(sig, 4)
    _check_sets(sig, found, [3, 4])
    assert dc["d_circ"] == 3 and dc["count"] == len(found[3][0])
    # weight-6 sets of a distance-3 model include reducible ones (a weight-3
    # logical set plus a weight-3 loop); the search keeps only irreducible sets
    sets6 = mo.undetectable_logicals(sig, 6).get(6)
    if sets6 is not None:
        for row in sets6[0][:200]:
            for size in (1, 2, 3):
                for combo in itertools.combinations(row, size):
                    assert np.bitwise_xor.reduce(sig.det[list(combo)], axis=0).any()


def test_memory_objective_escalates_the_proxy_at_distance_five():
    mc, dem, sig = _surface5("X")
    plain = mo.MemoryObjective(presets.chain(), 0.25, rounds=1, tier="proxy")
    out = plain.score_basis(mc, dem)
    assert out["proxy_weight"] == 3 and out["d_circ"] == 5 and out["d_circ_count"] > 0
    assert out["p_fail"] == pytest.approx(mo.pfail_low_weight(sig, None)) and out["p_fail"] > 0
    assert out["p_round"] == pytest.approx(mo.per_round(out["p_fail"], 1))
    fixed = mo.MemoryObjective(presets.chain(), 0.25, rounds=1, tier="proxy", proxy_max_weight=2)
    out2 = fixed.score_basis(mc, dem)
    assert out2["proxy_weight"] == 2 and out2["p_fail"] == 0.0 and "d_circ" not in out2
    out3 = mo.MemoryObjective(presets.chain(), 0.25, rounds=1, tier="proxy", proxy_max_weight=2, with_dcirc=True).score_basis(mc, dem)
    assert out3["d_circ"] is None and out3["d_circ_count"] == 0
    assert plain.describe()["proxy_max_weight"] is None


def test_evaluate_code_at_distance_five():
    code = StabilizerCode.from_stabilizers(SURFACE5)
    assert mo.code_distance(code) == 5  # beyond the exact limit: the randomized upper bound
    ev = evaluate_code(code, presets.chain(), rounds=1, scale=0.25)
    assert ev.d_circ == {"Z": 5, "X": 5} and ev.p_round > 0
    assert all(ev.per_basis[b]["proxy_weight"] == 3 for b in "ZX")
    assert ev.dominant_faults and all(len(f.mechanisms) == 5 for f in ev.dominant_faults)
    probs = [f.probability for f in ev.dominant_faults]
    assert probs == sorted(probs, reverse=True)
    for f in ev.dominant_faults[:3]:
        assert sum("L0" in loc.targets for loc in f.locations) % 2 == 1
    # the hook-aware order built by evaluate_code reproduces the standard one
    _, _, sig = _surface5("X")
    assert ev.per_basis["X"]["d_circ_count"] == mo.exact_dcirc(sig, 6)["count"]


def test_interleaved_schedule_on_surface_codes():
    """One block, one measurement per round, four gate layers, a clean
    noiseless detector model, and the full circuit distance."""
    for name, d in (("rotated_surface_9", 3), ("rotated_surface_16", 4)):
        code = known_codes.get(name)
        for basis in "ZX":
            mc = mo.build_memory_circuit(code, 2, basis)
            assert mc.extraction == "interleaved" and len(mc.batches) == 1
            assert mc.layers == 2 * 8  # R, H, four gate layers, H, M per round
            assert sum(1 for inst in mc.body if inst.name == "M") == 2
            assert mc.circuit().detector_error_model().num_errors == 0
            sig = mo.dem_signature(mo.memory_dem(mc, presets.chain(), 0.25))
            assert mo.exact_dcirc(sig, 6)["d_circ"] == d
        batched = mo.build_memory_circuit(code, 2, "X", extraction="batched")
        assert batched.extraction == "batched" and len(batched.batches) == 2 and batched.layers > mc.layers
    # the standard surface-code pattern schedules; seven generators on one
    # qubit need seven layers, more than a round of a weight-4 code allows
    assert mo.interleavable(_surface5_orders("col"))
    assert not mo.interleavable([[(0, "X"), (1, "X"), (2, "X"), (3, "X")]] * 7)
    # the Steane code's six weight-4 checks share every qubit six ways and
    # fall back to batches
    steane = known_codes.get("steane")
    assert mo.build_memory_circuit(steane, 1, "Z").extraction == "batched"
    with pytest.raises(ValueError):
        mo.build_memory_circuit(steane, 1, "Z", extraction="interleaved")


def test_interleaved_schedule_with_flags_and_evaluate_code():
    code = known_codes.get("rotated_surface_9")
    flagged = mo.build_memory_circuit(code, 2, "Z", flags="one")
    assert flagged.extraction == "interleaved"
    c = flagged.circuit()
    assert c.detector_error_model().num_errors == 0
    assert c.num_detectors == mo.build_memory_circuit(code, 2, "Z").circuit().num_detectors + 2 * 4  # four weight-4 generators carry a flag
    ev = evaluate_code(code, presets.chain(), scale=0.25, rounds=2)
    assert all(ev.per_basis[b]["extraction"] == "interleaved" and ev.per_basis[b]["batches"] == 1 for b in "ZX")
    assert ev.d_circ == {"Z": 3, "X": 3}
