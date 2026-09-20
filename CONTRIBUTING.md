# Contributing

Issues and pull requests are welcome at the repository (`ORG` in URLs is
the GitHub organization placeholder; see RELEASE.md).

Setup and checks. The sibling checkout `../noise-model` is the development
source of `noise_model` (`[tool.uv.sources]` in pyproject.toml); without
it, `uv sync --no-sources` resolves the published pin.

```
git clone https://github.com/ORG/noise-model
git clone https://github.com/ORG/autoqec-sim && cd autoqec-sim
uv sync --group dev --extra rl     # plain `uv sync --group dev` skips torch and the PPO tests
uv run ruff check
uv run pytest                      # about 70 s; the reduced regressions are in the suite
```

Rules of the repository:

- The reward engine is numpy on CPU by design; keep the state as the one
  `(batch, 2n, 2n+1)` uint8 array and the hot path in the kernels listed
  under "What is left for cuQuantum" in README.md, so a GPU backend can
  replace them without touching the surrounding code.
- A change that affects search results reruns the full-budget regressions
  and updates `results/regression.json` or `results/regression_ppo.json`
  and the tables in README.md (`docs/reproduce.md` has the commands).
- New reward terms or constraints get a verification independent of the
  reward (exact distance, enumerators), as `regress` does.
- Add a CHANGELOG.md entry under the upcoming version.
- Prose: no em dashes, Oxford comma, terse.

Commits are authored by their author; no generated-with or co-author
trailers.
