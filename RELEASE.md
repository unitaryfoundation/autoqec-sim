# Release checklist for autoqec-sim 0.2.0

Everything below the line "Done in the checkout" is not yet done. Steps
marked OWNER need a decision by the owner; steps marked RUN are mechanical
once the OWNER steps before them are settled. `ORG` is the GitHub
organization placeholder used in pyproject.toml, README.md, CITATION.cff,
CONTRIBUTING.md, docs/reproduce.md, scripts/reproduce.sh, and the two
workflows; step 1 fixes it. Release noise-model first: this package pins
`noise-model>=0.2.0,<0.3` and its CI checks out `ORG/noise-model` at tag
`v0.2.0`.

## Done in the checkout

- `git init`, `.gitignore`, initial commit on `main`, no remote.
- Version 0.2.0 in `pyproject.toml` and `autoqec_sim.__version__`
  (`tests/test_version.py` pins them together); CHANGELOG.md entry dated
  TBD with the former "Unreleased" PPO items folded in.
- Dependency on noise-model resolvable two ways: `[tool.uv.sources]` path
  source (`../noise-model`, editable) for development, the version pin in
  `[project.dependencies]` for the wheel.
- `uv build` produces `dist/autoqec_sim-0.2.0.tar.gz` and
  `dist/autoqec_sim-0.2.0-py3-none-any.whl`; both install into a clean venv
  together with the noise-model wheel and pass the test suite.
- `.github/workflows/ci.yml` (ruff, pytest on 3.12 and 3.13, Linux and
  macOS without the `rl` extra; one Linux job with CPU torch; a build job)
  and `.github/workflows/release.yml` (build and attach wheels on a `v*`
  tag).
- LICENSE (Apache-2.0), CITATION.cff, CONTRIBUTING.md, `docs/reproduce.md`,
  `scripts/reproduce.sh`.

## Remaining steps

1. OWNER: choose the GitHub organization (the DMSP names Unitary
   Foundation's organization; IonQ hosting is undecided) and the
   institutional copyright holder. Then replace the placeholder and the
   license note:

   ```
   grep -rn "ORG" pyproject.toml README.md CITATION.cff CONTRIBUTING.md RELEASE.md \
     docs/reproduce.md scripts/reproduce.sh .github/workflows
   sed -i '' 's#github.com/ORG/#github.com/<org>/#g; s#repository: ORG/#repository: <org>/#g' \
     pyproject.toml README.md CITATION.cff CONTRIBUTING.md RELEASE.md docs/reproduce.md \
     scripts/reproduce.sh .github/workflows/ci.yml .github/workflows/release.yml
   # LICENSE: edit the first five lines (copyright line and the "to be confirmed" note).
   # CITATION.cff: fill the ORCID (orcid.org/0000-0000-0000-0000 is a placeholder)
   #               and set date-released to the tag date; CHANGELOG.md: replace TBD.
   git commit -am "Set organization, copyright holder, and release date for 0.2.0"
   ```

2. OWNER: decide whether the two packages also go to PyPI (both names
   were unclaimed on 2026-09-20). Without PyPI, users install noise-model
   from its git tag before this package, as README.md says; with it, the
   pin resolves on its own and `release.yml` gains a
   `pypa/gh-action-pypi-publish` step and a trusted publisher on PyPI.

3. RUN: create the remote and push (after noise-model's `v0.2.0` tag
   exists, or CI's checkout of it fails).

   ```
   gh repo create <org>/autoqec-sim --public --source . --remote origin \
     --description "Vectorized stabilizer-code reward engine and RL environment for AutoQEC (M1)" --push
   ```

4. RUN: wait for CI on `main` to pass (six jobs).

   ```
   gh run list --workflow ci.yml --limit 1
   gh run watch
   ```

5. RUN: tag and push the tag. The release workflow checks that the tag
   equals `uv version --short`.

   ```
   git tag -a v0.2.0 -m "autoqec-sim 0.2.0"
   git push origin v0.2.0
   gh run list --workflow release.yml --limit 1 && gh run watch
   gh release view v0.2.0        # sdist and wheel attached
   ```

6. OWNER then RUN: Zenodo DOI, per the DMSP (open release with a
   persistent identifier). Log in to zenodo.org with the GitHub account
   that owns `<org>`, enable the repository under GitHub integration
   before the tag is pushed (or push a `v0.2.1` metadata tag afterwards),
   then paste the concept DOI into CITATION.cff (`doi:` field) and the
   README citation line, and commit.

7. RUN: record the release in the project ledger and the M1 completion
   evidence (release URL, tag, DOI, CI run URL, `docs/reproduce.md`
   numbers). The M1 items still open after this release are the cuQuantum
   kernels (NVIDIA) and noise-aware PPO training (in progress under
   `experiments/`, not part of 0.2.0).
