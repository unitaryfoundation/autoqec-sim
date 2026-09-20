# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Known-code rediscovery regression.

For each target the search agent, from a seeded start, must produce an
encoding circuit on the target's connectivity graph whose code has the
target parameters ``[[n, 1, 3]]`` with no stabilizer element below a
target-specific weight (so a smaller code padded with idle qubits does not
count). The result is verified independently of the reward: the circuit is
replayed, the code is read off the tableau, its distance is computed
exactly, and its weight enumerators are compared with the reference code.
Equal enumerators of ``S`` and ``N(S)`` are a necessary condition for
equivalence under qubit permutations and local Cliffords; the report labels
the match as such rather than as a proof of isomorphism.

LDPC targets add the sparsity constraint of :mod:`autoqec_sim.ldpc`
(``Target.ldpc``): a hard bound ``w_max`` on the generator weight enforced
in the environment (mode ``"mask"`` or ``"terminate"``) or as a reward
penalty (mode ``"penalty"``). Two of them are *exclusion* targets
(``expect_success=False``): the search must not find a distance-3 code
within budget. For 5 qubits with ``w_max = 3`` this is a theorem (the
[[5,1,3]] code is unique and every nontrivial stabilizer has weight 4);
for 7 qubits with ``w_max = 3`` it is an empirical statement bounded by
the budget, contrasted with ``w_max = 4``, which admits the Steane code
and is found in the same budget. The verification of every LDPC target
reports ``min_max_generator_weight`` (exact ``w*``) and the maximum qubit
degree of the sparsest basis found.

Run ``autoqec-regress --budget full`` for the release numbers (restarts
until the enumerators match) and ``--budget reduced`` for the quick check
that the pytest uses.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import numpy as np

from . import graphs, known_codes
from .code import StabilizerCode
from .env import CodeDiscoveryEnv
from .errors import ErrorSet
from .ldpc import make_ldpc_penalty, sparsest_basis_stats
from .reward import CompositeReward, make_kl_reward, make_low_weight_penalty
from .search import EvolutionarySearch
from .tableau import BatchTableau


@dataclass(frozen=True)
class LdpcSpec:
    """LDPC constraint of a target. ``mode`` is ``"mask"``, ``"terminate"``,
    or ``"final"`` (hard constraint in the environment) or ``"penalty"``
    (reward term ``-penalty_scale * excess``, see ``ldpc.ldpc_penalty``)."""

    w_max: int
    deg_max: int | None = None
    mode: str = "mask"
    penalty_scale: float = 0.25

    def to_dict(self):
        return {"w_max": self.w_max, "deg_max": self.deg_max, "mode": self.mode, "penalty_scale": self.penalty_scale}


@dataclass(frozen=True)
class Target:
    name: str
    n: int
    k: int
    d: int
    graph_spec: str
    reference: str | None
    max_steps: int
    forbid_stabilizer_weight: int  # penalize stabilizer elements of weight <= this
    gates: tuple = ("H", "S", "CNOT")
    ldpc: LdpcSpec | None = None
    expect_success: bool = True
    # False: any verified code meeting the parameters and the LDPC bound
    # counts, whatever its enumerators (the reference is still reported)
    require_reference_match: bool = True


TARGETS = {
    "five_qubit": Target("five_qubit", 5, 1, 3, "all_to_all:5", "five_qubit", 14, 1),
    "steane": Target("steane", 7, 1, 3, "all_to_all:7", "steane", 20, 3),
    # H and CNOT only: with S in the action set the rotated-surface
    # enumerators did not appear in 30 tallied runs; with H, CNOT they
    # appear in about one run in twenty.
    "rotated_surface_9": Target(
        "rotated_surface_9", 9, 1, 3, "square:3x3", "rotated_surface_9", 30, 1, ("H", "CNOT")
    ),
    # LDPC targets. The rotated surface code has w* = 4 and qubit degree 4;
    # Shor's code has w* = 6 (its X-type stabilizers all have weight 6) and
    # is excluded by w_max = 4 along with every denser [[9,1,3]] code.
    "rotated_surface_9_ldpc": Target(
        "rotated_surface_9_ldpc", 9, 1, 3, "square:3x3", "rotated_surface_9", 30, 1, ("H", "CNOT"),
        ldpc=LdpcSpec(w_max=4, deg_max=4, mode="mask"),
    ),
    # The penalty form finds a distance-3 code within the bound on nearly
    # every attempt (60 of 60 at full budget, 31 distinct enumerator
    # classes) and stops at the first one, so this target accepts any such
    # code; the surface-code enumerators appeared in 1 of 12 reduced-budget
    # seeds and 0 of 60 full-budget attempts.
    "rotated_surface_9_ldpc_penalty": Target(
        "rotated_surface_9_ldpc_penalty", 9, 1, 3, "square:3x3", "rotated_surface_9", 30, 1, ("H", "CNOT"),
        ldpc=LdpcSpec(w_max=4, deg_max=4, mode="penalty", penalty_scale=0.25),
        require_reference_match=False,
    ),
    # Steane has w* = 4, so w_max = 4 admits it; w_max = 3 admits no
    # [[7,1,3]] code found within budget (empirical, see the module
    # docstring). The exclusion targets forbid only weight-1 stabilizers
    # (idle qubits), so weight-2 and weight-3 generators are allowed and
    # the question is not settled by the padding penalty. They use the
    # penalty form so that the search path is unconstrained and only the
    # final code is judged: a hard per-step constraint would also exclude
    # feasible codes reachable only through heavier intermediate groups.
    "steane_ldpc_w4": Target(
        "steane_ldpc_w4", 7, 1, 3, "all_to_all:7", "steane", 20, 3, ldpc=LdpcSpec(w_max=4, mode="penalty")
    ),
    "seven_qubit_w3_excluded": Target(
        "seven_qubit_w3_excluded", 7, 1, 3, "all_to_all:7", None, 20, 1,
        ldpc=LdpcSpec(w_max=3, mode="penalty"), expect_success=False,
    ),
    # The [[5,1,3]] code is the unique 5-qubit distance-3 code and all 15
    # nontrivial stabilizers have weight 4, so w_max = 3 excludes it.
    "five_qubit_w3_excluded": Target(
        "five_qubit_w3_excluded", 5, 1, 3, "all_to_all:5", None, 14, 1,
        ldpc=LdpcSpec(w_max=3, mode="penalty"), expect_success=False,
    ),
}

BUDGETS = {
    # batch, evaluations per attempt, max attempts, require enumerator match,
    # seeds run to exhaustion for exclusion targets
    "full": dict(batch_size=512, max_evaluations=100_000, max_attempts=60, require_match=True, exclusion_attempts=5),
    "reduced": dict(batch_size=256, max_evaluations=15_000, max_attempts=3, require_match=False, exclusion_attempts=1),
}


def make_reward(target: Target):
    errors = ErrorSet.up_to_weight(target.n, target.d - 1)
    terms = [
        (1.0, make_kl_reward(errors)),
        (1.0, make_low_weight_penalty(target.forbid_stabilizer_weight)),
    ]
    if target.ldpc is not None and target.ldpc.mode == "penalty":
        terms.append((1.0, make_ldpc_penalty(target.ldpc.w_max, target.ldpc.deg_max, target.ldpc.penalty_scale)))
    return CompositeReward(terms)


def env_kwargs(target: Target) -> dict:
    """Environment arguments of the hard LDPC constraint, if any."""
    if target.ldpc is None or target.ldpc.mode == "penalty":
        return {}
    return {"ldpc_w_max": target.ldpc.w_max, "ldpc_mode": target.ldpc.mode}


def replay(gates, n: int) -> StabilizerCode:
    bt = BatchTableau(n, 1)
    for g, a, b in gates:
        bt.apply(g, a, b if b >= 0 else None)
    return StabilizerCode.from_tableau(bt.array, 1)


def verify(code: StabilizerCode, target: Target) -> dict:
    d = code.distance_exact()
    a, b = code.weight_enumerators()
    min_stab_w = int(np.flatnonzero(a[1:])[0] + 1) if a[1:].any() else 0
    nontrivial = min_stab_w > target.forbid_stabilizer_weight
    st = sparsest_basis_stats(code.to_tableau(), code.k)
    out = {
        "n": code.n,
        "k": code.k,
        "distance_exact": int(d),
        "parameters_match": (code.n, code.k, d) == (target.n, target.k, target.d),
        "min_stabilizer_weight": min_stab_w,
        "nontrivial": bool(nontrivial),
        "is_css": bool(code.is_css()),
        "stabilizers": code.stabilizer_strings(),
        "weight_enumerator_S": a.tolist(),
        "weight_enumerator_N": b.tolist(),
        "min_max_generator_weight": int(st["w_star"][0]),
        "max_qubit_degree": int(st["max_degree"][0]),
        "enumerators_match_reference": False,
    }
    if target.reference is not None:
        ra, rb = known_codes.get(target.reference).weight_enumerators()
        out["enumerators_match_reference"] = bool(np.array_equal(a, ra) and np.array_equal(b, rb))
    if target.ldpc is not None:
        out["ldpc_bound_met"] = bool(
            st["w_star"][0] <= target.ldpc.w_max
            and (target.ldpc.deg_max is None or st["max_degree"][0] <= target.ldpc.deg_max)
        )
    return out


def run_target(
    target: Target,
    seed: int = 0,
    batch_size: int = 512,
    max_evaluations: int = 100_000,
    max_attempts: int = 1,
    require_match: bool = False,
    exclusion_attempts: int = 1,
    gates=None,
    verbose: bool = False,
) -> dict:
    """One target. Exclusion targets (``expect_success=False``) run
    ``exclusion_attempts`` seeds (each to the full ``max_evaluations``) and
    pass when none finds a verified code."""
    gates = tuple(target.gates if gates is None else gates)
    graph = graphs.from_spec(target.graph_spec)
    reward = make_reward(target)
    t0 = time.perf_counter()
    attempts = []
    total_evals = 0
    outcome = None
    for attempt in range(max_attempts):
        s = seed + attempt
        env = CodeDiscoveryEnv(
            n=target.n,
            k=target.k,
            graph=graph,
            batch_size=batch_size,
            max_steps=target.max_steps,
            gates=gates,
            reward_fn=reward,
            seed=s,
            **env_kwargs(target),
        )
        res = EvolutionarySearch(env, seed=s).run(max_evaluations, verbose=verbose)
        total_evals += res.evaluations
        rec = {
            "seed": s,
            "search_success": res.success,
            "evaluations": res.evaluations,
            "generations": res.generations,
            "wall_time_s": round(res.wall_time_s, 3),
            "best_reward": res.best_reward,
            "circuit_length": len(res.gates),
        }
        if res.success:
            code = replay(res.gates, target.n)
            ver = verify(code, target)
            rec["verification"] = ver
            rec["gates"] = [list(g) for g in res.gates]
            good = ver["parameters_match"] and ver["nontrivial"] and ver.get("ldpc_bound_met", True)
            need_match = require_match and target.require_reference_match and target.reference is not None
            if good and (ver["enumerators_match_reference"] or not need_match):
                outcome = rec
        attempts.append(rec)
        if outcome is not None:
            break
        if not target.expect_success and attempt + 1 >= exclusion_attempts:
            break
    rediscovered = outcome is not None
    return {
        "target": target.name,
        "parameters": [target.n, target.k, target.d],
        "graph": target.graph_spec,
        "gates": list(gates),
        "reference": target.reference,
        "forbid_stabilizer_weight": target.forbid_stabilizer_weight,
        "ldpc": None if target.ldpc is None else target.ldpc.to_dict(),
        "expect_success": target.expect_success,
        "budget": {
            "batch_size": batch_size,
            "max_evaluations_per_attempt": max_evaluations,
            "max_attempts": max_attempts if target.expect_success else min(max_attempts, exclusion_attempts),
            "require_enumerator_match": bool(require_match and target.require_reference_match and target.reference is not None),
        },
        "seed": seed,
        "rediscovered": rediscovered,
        "passed": rediscovered == target.expect_success,
        "attempts_used": len(attempts),
        "total_evaluations": total_evals,
        "wall_time_s": round(time.perf_counter() - t0, 3),
        "result": outcome,
        "attempts": attempts,
    }


def run_all(budget: str = "reduced", seed: int = 0, targets=None, verbose=False) -> dict:
    cfg = BUDGETS[budget]
    names = list(TARGETS) if targets is None else list(targets)
    out = {
        "budget": budget,
        "budget_config": cfg,
        "seed": seed,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "targets": {},
    }
    for name in names:
        out["targets"][name] = run_target(TARGETS[name], seed=seed, verbose=verbose, **cfg)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Known-code rediscovery regression.")
    ap.add_argument("--budget", choices=list(BUDGETS), default="reduced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", nargs="*", default=None, choices=list(TARGETS))
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    res = run_all(args.budget, args.seed, args.targets, args.verbose)
    for name, r in res["targets"].items():
        v = r["result"]["verification"] if r["result"] else None
        line = f"{name:30s} passed={r['passed']} rediscovered={r['rediscovered']} attempts={r['attempts_used']} evals={r['total_evaluations']} wall={r['wall_time_s']}s"
        if v:
            line += f" d={v['distance_exact']} css={v['is_css']} w*={v['min_max_generator_weight']} deg={v['max_qubit_degree']} enumerators_match={v['enumerators_match_reference']}"
        print(line)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", args.out)
    return 0 if all(r["passed"] for r in res["targets"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
