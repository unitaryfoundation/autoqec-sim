"""SAT-constrained local search over the surface-footprint codes with the circuit-level proxy as objective.

Prototype of the design loop the population experiment argues for: SAT keeps every candidate a valid
distance-3 code on the footprint (the clauses are those of DalFavero et al.'s surface sampler, rebuilt here
without that package's cirq import), CMSGen samples completions under assumptions that fix all but two
generators, and the Forte proxy tier of the memory objective ranks them. Each step frees two random
generators, scores up to eight distinct completions, and accepts the best if it improves. Every evaluation is
logged so the trajectory compares with random sampling at equal evaluation counts. Needs the `qodesign`
package of github.com/bdalfavero/satisfying-codes on the path, `pycmsgen`, and `python-sat`.

    uv run --with pycmsgen --with python-sat python -m experiments.sat_population.local_search <seed> <steps> <out.json>

Results of three runs of 40 steps are in `local_search/run_{1,2,3}.json`: final proxy rates 0.000339,
0.000331, and 0.000365 after 143, 129, and 159 evaluations; the random sample's best of 2,000 draws is
0.000344 and its best-of-150 median 0.000351. At the sampled tier with 500 failures the best search result
and the population's best code are 0.000333 and 0.000337, XZZX 0.000485.
"""

from __future__ import annotations

import json
import random
import sys
import time

from pycmsgen import Solver
from qodesign.sat import (
    _CnfBuilder,
    _CnfVarString,
    _detect_or_trivial_clause,
    _symplectic_terms,
    assign_strings_from_model,
    enumerate_errors,
)

from autoqec_sim.code import StabilizerCode
from autoqec_sim.memory_objective import MemoryObjective
from autoqec_sim.noise import presets

from ..memory_objective.run import hadamard_conjugate

N = 9
FOOTPRINTS = [{0, 1, 3, 4}, {4, 5, 7, 8}, {2, 5}, {3, 6}, {1, 2, 4, 5}, {3, 4, 6, 7}, {0, 1}, {7, 8}]


def surface_clauses(errors):
    """Clauses of the source's surface sampler: letters free on the footprint, generators commute,
    every error of the list is detected or trivial."""
    builder = _CnfBuilder()
    strings = [_CnfVarString.fresh(N, builder) for _ in FOOTPRINTS]
    for cvs, fp in zip(strings, FOOTPRINTS, strict=True):
        for q in range(N):
            if q not in fp:
                for lit in (cvs.xs[q], cvs.zs[q]):
                    if not isinstance(lit, bool):
                        builder.add([-lit])
    for i in range(len(strings)):
        for j in range(i):
            lits, const = _symplectic_terms(builder, strings[i], strings[j])
            builder.assert_parity(lits, const, False)
    for err in errors:
        builder.add(_detect_or_trivial_clause(builder, strings, [], err, N))
    return builder.clauses, strings


def sol_to_model(sol):
    return [i if bit else -i for i, bit in enumerate(sol) if bit is not None]


def distance3_errors():
    errors = []
    for w in (1, 2):
        errors.extend(enumerate_errors(N, w))
    return errors


def main(seed: int, steps: int, out: str) -> None:
    rng = random.Random(seed)
    clauses, strings = surface_clauses(distance3_errors())
    solver = Solver(seed=seed)
    for clause in clauses:
        solver.add_clause(clause)
    objective = MemoryObjective(presets.forte("sequential"), scale=0.25, rounds=3, tier="proxy")

    def letters(model):
        return [str(s).lstrip("+-").replace("_", "I") for s in assign_strings_from_model(model, strings)]

    def score(code_strings):
        code = StabilizerCode.from_stabilizers(hadamard_conjugate(code_strings, range(N)))
        return objective.evaluate(code, use_cache=False).p_round

    _, sol = solver.solve()
    model = sol_to_model(sol)
    current = letters(model)
    current_p = score(current)
    log = [{"step": 0, "p": current_p, "code": current, "accepted": True}]
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        free = rng.sample(range(len(strings)), 2)
        positive = {lit for lit in model if lit > 0}
        assumptions = [
            lit if lit in positive else -lit
            for g in range(len(strings))
            if g not in free
            for lit in strings[g].variable_lits()
        ]
        candidates = {}
        for _ in range(8):
            ok, m = solver.solve(assumptions)
            if not ok:
                break
            mm = sol_to_model(m)
            s = tuple(letters(mm))
            if s not in candidates and list(s) != current:
                candidates[s] = mm
        best = None
        for s, mm in candidates.items():
            p = score(list(s))
            log.append({"step": step, "p": p, "code": list(s), "accepted": False})
            if best is None or p < best[0]:
                best = (p, s, mm)
        if best and best[0] < current_p:
            current_p, current, model = best[0], list(best[1]), best[2]
            log[-1]["accepted"] = True
            print(f"step {step}: {current_p:.6f} ({len(log)} evals, {time.perf_counter() - t0:.0f} s)", flush=True)
    with open(out, "w") as fh:
        json.dump({"seed": seed, "steps": steps, "final_p": current_p, "final": current, "evals": len(log), "log": log}, fh)
    print("final", current_p, current, "evals", len(log))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
