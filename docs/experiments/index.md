# Experiments

Each experiment asks one question about code discovery or code
evaluation, runs from `experiments/<name>/run.py` in the repository
(`uv run python -m experiments.<name>.run` from the repository root;
the code is not part of the installed package), and keeps its write-up
in `experiments/<name>/REPORT.md` next to it (raw runs under
`experiments/<name>/runs/` are not tracked). Each page gives the
question, the result with its figure, and the commands that reproduce
one run.

| Experiment | Question | Result |
|---|---|---|
| [Noise-aware discovery](noise_aware.md) | Does the code PPO discovers depend on the noise profile it trains against? | Yes, by 1.3x to 3.4x in undetected fault mass, through the encoder's fault propagation rather than the code's X/Z structure |
| [Memory objective runs](memory_objective.md) | Does the closed-form proxy of the memory objective rank codes like a sampled decoder? | Spearman 0.78 to 0.90 against BP+OSD on 32 random codes; XZZX preferred under Z bias |
| [RL scaling probe](scaling.md) | Why does PPO miss the [[16,1,4]] code, and what is the cheapest fix? | Exploration; a permutation-equivariant policy finds it in 0.2M to 0.26M env steps on every seed |
| [Three-topology matrix](topologies.md) | Does PPO reach distance 3 on square, triangular, and heavy-hex lattices? | 18 of 18 runs, at 6k to 345k env steps |
| [Evolutionary search against PPO](topologies_ea.md) | Does the package's evolutionary search match PPO at the same budget? | 17 of 18 runs succeed; PPO needs 1.4x to 6x fewer env steps, the evolutionary search far less wall time |
| [QDX's targets](qdx_targets.md) | How many of the 50 targets QDX publishes does this PPO reach in 10 minutes each? | 24 of the 45 run: 22 of 36 at d = 3, 2 of 9 at d = 4 |
| [SAT-designed codes](sat_codes.md) | Does a code-capacity ranking of fifteen SAT-designed codes survive circuit-level noise? | Barely: Spearman 0.08 to 0.52; XZZX beats or ties every one of them |
| [SAT population](sat_population.md) | The same question over 2,150 codes | No: Spearman 0.00 to 0.21; the best codes sit 30 to 40 percent below XZZX |
| [Proxy against Tesseract](proxy_tesseract.md) | Does the shot-free proxy rank distance-5 codes like a near-maximum-likelihood decoder? | Yes: Spearman 0.89 and 0.93 over 21 codes at three and five rounds, about 30 percent low in level at 1.7 to 2.8 expected faults per shot, at a fifth of the cost of 50,000 shots |
| [Published experiments](hardware_comparison.md) | How far is the evaluator from five published per-round logical error rates, on its own circuit with each paper's parameters? | Below the paper's number in all nine rows: 0.4 to 0.9 of it for the surface, toric, bicycle, and heavy-hex rows, 0.2 to 0.45 for Steane, where the circuits differ most; matching on the Google circuits within 2 to 24 percent of the papers' matching figures |
| [Other footprints](footprints.md) | Does the surface-footprint picture (every letter assignment keeps the distance, most beat the CSS code) hold on the Steane and 4.8.8 color-code footprints? | Not as a rule: 56 and 28 percent of the assignments lose distance, and on the full-distance codes the gain over CSS holds on the 4.8.8 footprint (27 of 29 by the proxy, the five checked at 0.41 to 0.52 of CSS under Tesseract) but not on Steane (41 of 132, and the two noise cases disagree) |

Every number on these pages is copied from the experiment's report, which
regenerates from the runs with the module's `report` command; the figures
regenerate with its `figure` command (or `report`, for the older
experiments). The memory-objective runs recorded here used the batched
extraction schedule, the only one before the interleaved schedule became
the default; `extraction="batched"` reproduces them. Where a case is named Forte, it is the package's Forte
preset, a noise model assembled from public specifications, not a
measurement on a device.
