# QDX's targets through autoqec-sim

Status: 24 of 45 targets run so far found and verified. Every
number comes from a JSON under `experiments/qdx_targets/runs/`; this file
regenerates with `python -m experiments.qdx_targets.run report`.
Code: `experiments/qdx_targets/run.py`.

QDX (Olle, Zen, Puviani, Marquardt, npj Quantum Information 10, 126 (2024);
github.com/jolle-ag/qdx) publishes every [[n, k, d]] its agents found on
all-to-all connectivity for n <= 15. The QDX columns count the codes in
that archive and give the shortest and median encoding circuit among them.
The autoqec columns are one PPO run per target (seed in the JSON), gates H,
S, CNOT, all-to-all, batch 512, episode cap 30 / 40 / 60 gates for
d = 3 / 4 / 5, the default Knill-Laflamme reward over all Paulis of weight
below d, and a budget of 3M environment steps or 10 minutes, whichever
comes first. Success means every such error is detected or in the group;
"verified" means the exact distance of the resulting code is at least d.
QDX weighted the same error set by a depolarizing channel and ran four
agents per target on a GPU, so wall times are not comparable across the two;
success within budget and circuit length are. 21 of the unsuccessful runs stopped at the time cap after 172,032 to 1,712,128 environment steps, so they used between 6% and 57% of the step budget; the env steps column says which. This run shared the machine with other jobs, which cut the step rate by up to eightfold from n = 12 on.

| target | QDX codes | QDX circuit length (min / median) | autoqec PPO | env steps | wall | circuit length | verified d |
|---|---|---|---|---|---|---|---|
| [[5,1,3]] | 930 | 10 / 12 | found | 16,384 | 2 s | 17 | 3 |
| [[6,1,3]] | 3215 | 11 / 15 | found | 16,384 | 2 s | 29 | 3 |
| [[7,1,3]] | 1393 | 10 / 14 | found | 16,384 | 2 s | 25 | 3 |
| [[8,1,3]] | 1376 | 11 / 15 | found | 16,384 | 2 s | 30 | 3 |
| [[8,2,3]] | 666 | 16 / 18 | found | 16,384 | 3 s | 30 | 3 |
| [[8,3,3]] | 164 | 19 / 22 | found | 876,544 | 132 s | 20 | 3 |
| [[9,1,3]] | 9032 | 11 / 15 | found | 16,384 | 3 s | 30 | 3 |
| [[9,2,3]] | 2148 | 14 / 18 | found | 122,880 | 22 s | 26 | 3 |
| [[9,3,3]] | 1336 | 18 / 22 | found | 352,256 | 64 s | 24 | 3 |
| [[10,1,3]] | 1021 | 9 / 15 | found | 49,152 | 9 s | 29 | 3 |
| [[10,2,3]] | 2949 | 15 / 19 | found | 65,536 | 12 s | 27 | 3 |
| [[10,3,3]] | 3014 | 18 / 22 | found | 262,144 | 50 s | 28 | 3 |
| [[10,4,3]] | 928 | 21 / 26 | found | 892,928 | 179 s | 29 | 3 |
| [[11,1,3]] | 696 | 9 / 19 | found | 73,728 | 15 s | 22 | 3 |
| [[11,2,3]] | 384 | 17 / 20 | found | 172,032 | 36 s | 29 | 3 |
| [[11,3,3]] | 366 | 20 / 24 | found | 327,680 | 73 s | 30 | 3 |
| [[11,4,3]] | 3488 | 22 / 26 | found | 983,040 | 215 s | 29 | 3 |
| [[11,5,3]] | 768 | 25 / 30 | found | 1,228,800 | 230 s | 30 | 3 |
| [[12,2,3]] | 224 | 18 / 21 | found | 81,920 | 138 s | 27 | 3 |
| [[12,3,3]] | 375 | 20 / 25 | found | 98,304 | 220 s | 30 | 3 |
| [[12,4,3]] | 3006 | 23 / 27 | found | 221,184 | 430 s | 30 | 3 |
| [[12,5,3]] | 2535 | 26 / 29 | not found in budget | 327,680 | 615 s | - | - |
| [[12,6,3]] | 998 | 28 / 33 | not found in budget | 319,488 | 601 s | - | - |
| [[13,4,3]] | 364 | 25 / 29 | found | 172,032 | 237 s | 29 | 3 |
| [[13,5,3]] | 215 | 27 / 30 | not found in budget | 475,136 | 610 s | - | - |
| [[13,6,3]] | 1299 | 28 / 32 | not found in budget | 458,752 | 601 s | - | - |
| [[13,7,3]] | 540 | 32 / 38 | not found in budget | 458,752 | 608 s | - | - |
| [[14,5,3]] | 214 | 28 / 31 | not found in budget | 417,792 | 602 s | - | - |
| [[14,6,3]] | 159 | 29 / 33 | not found in budget | 425,984 | 602 s | - | - |
| [[14,7,3]] | 748 | 32 / 36 | not found in budget | 442,368 | 604 s | - | - |
| [[14,8,3]] | 377 | 36 / 42 | not found in budget | 393,216 | 610 s | - | - |
| [[15,5,3]] | 238 | 29 / 33 | not found in budget | 229,376 | 604 s | - | - |
| [[15,6,3]] | 156 | 31 / 34 | not found in budget | 237,568 | 613 s | - | - |
| [[15,7,3]] | 91 | 33 / 37 | not found in budget | 262,144 | 601 s | - | - |
| [[15,8,3]] | 545 | 35 / 39 | not found in budget | 237,568 | 611 s | - | - |
| [[15,9,3]] | 328 | 38 / 47 | not found in budget | 229,376 | 609 s | - | - |
| [[10,1,4]] | 622 | 20 / 26 | not found in budget | 1,114,112 | 601 s | - | - |
| [[10,2,4]] | 111 | 26 / 31 | not found in budget | 1,712,128 | 602 s | - | - |
| [[11,1,4]] | 1189 | 18 / 24 | found | 638,976 | 403 s | 35 | 4 |
| [[11,2,4]] | 698 | 26 / 32 | not found in budget | 1,114,112 | 602 s | - | - |
| [[12,1,4]] | 448 | 20 / 24 | found | 204,800 | 526 s | 35 | 4 |
| [[12,2,4]] | 749 | 25 / 31 | not found in budget | 253,952 | 601 s | - | - |
| [[12,3,4]] | 304 | 33 / 40 | not found in budget | 294,912 | 604 s | - | - |
| [[13,3,4]] | 689 | 31 / 38 | not found in budget | 262,144 | 613 s | - | - |
| [[14,3,4]] | 58 | 31 / 36 | not found in budget | 172,032 | 632 s | - | - |
