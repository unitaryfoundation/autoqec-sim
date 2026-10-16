# Schedule synthesis for board codes: first two runs (2026-10-07)

The environment (`src/autoqec_sim/schedule_env.py`) fixes a board code and
lets a PPO agent assign each Tanner edge to a CX time slot; the terminal
reward is the circuit fault distance of the resulting memory experiment,
scored by the board's own builder and noise recipe. `run.py` scores the
board generator's own schedule family (greedy edge colorings) as the
baseline, trains, and confirms the best schedule at rounds = d.

## Where schedule synthesis can gain at all

Of the board's 54 circuits, 44 already have d_circ = d. Six sit at d - 2:
36-16-4, 40-24-4, 49-13-4, 64-14-4, 64-42-4, 72-24-4, all distance-4 codes
with check weight 8 to 32 under the generic sequential schedule. Three sit
at d - 1: 7-1-3, 38-10-3, 53-20-3 (sequential) and 56-2-8 (interleaved,
7 against 8). Every d - 2 code and both distance-3 ones have data qubits of
check-degree 1 in both blocks; 56-2-8 has minimum degree 2.

## [[49,13,4]]: 2 to 3, confirmed

| run | reward | updates x batch | best d_circ | layers | confirmation at rounds = 4 |
|---|---|---|---|---|---|
| 1 | d_circ / d | 60 x 16 | 2 (every schedule) | 20 | exact 2, RIS 200k 2 |
| 2 | (d_circ + 1/(1 + sets)) / (d + 1) | 120 x 16 | 3 on both sides (update 116) | 18 | exact 3, RIS 200k 3 |

Baseline: 20 greedy colorings read d_circ 1 or 2 (one coloring read 1),
with 1 to 26 minimum-weight undetected sets. Run 1 was flat: every schedule
read 2, entropy moved from 1.83 to 1.68, nothing to climb. Run 2 added the
number of minimum-weight undetected fault sets (from the exact tier) as a
tie-breaker inside a distance level; the mean did not climb much (sets mean
oscillated between 10 and 20, layers fell from 20 to 18 under the depth
penalty), but the policy produced one schedule at 3 on both sides, and it
held at rounds = 4 under both the exact search and 200,000 RIS trials.
The two-fault witnesses at 2 are pairs of hook errors on two weight-8
checks whose data residues multiply to a logical; the degree-1 data qubits
make many checks "soft" in this sense. Filed as a circuit update to the
board entry (same code, same sequential structure, new layer assignment).
Code distance is 4; no schedule of this family reached it.

## [[56,2,8]]: the search-tier reward is too noisy at 5,000 trials

Baseline colorings read 5 to 7 at rounds 2 with 5,000 RIS trials (1 s per
schedule), so the signal looked dense. The agent's "best" read 8 on both
sides at update 1, but the confirmation at rounds = 8 read 6 under 200,000
RIS trials (the exact tier only certifies "no set of weight <= 4" there).
The 5,000-trial reading is an upper bound that overstates by one to two,
so the reward rewarded search misses. The training mean stayed at 6.2 to
6.4 for 80 updates. The board's interleaved schedule, at 7, was not beaten.
A usable reward here needs 20,000 to 50,000 trials (5 to 10 s per schedule)
or the exact tier at wmax 5 on a smaller code.

## What to do next

- Interleaving: `--interleave` shares slots between the blocks; a random
  assignment is mostly non-deterministic, so it needs the overlap-parity
  criterion of the board's interleaved builder as an action mask rather
  than a terminal penalty.
- Hook-aware actions: choose a per-check ordering from
  `memory_objective.hook_orders` instead of a raw slot, which shrinks the
  action space from 18 slots x 235 edges to one permutation per check.
- The six d - 2 codes share the degree-1 structure; whether any sequential
  schedule reaches 4 on them is open. The exact tier makes the check cheap
  (1.5 s per schedule), so an exhaustive-per-check local search is a fair
  comparator to PPO before more training is spent.
