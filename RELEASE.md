# Release checklist for autoqec-sim 0.2.0

Steps marked OWNER need a decision by the owner; steps marked RUN are
mechanical once the OWNER steps before them are settled.

## Done in the checkout

- Version 0.2.0 in `pyproject.toml` and `autoqec_sim.__version__`
  (`tests/test_version.py` pins them together); CHANGELOG.md entry dated TBD.
- The noise module is the `autoqec_sim.noise` subpackage; the package has no
  dependency outside PyPI.
- `uv build` produces the sdist and wheel; both install into a clean venv and
  pass the test suite.
- `.github/workflows/ci.yml` (ruff, pytest on 3.12 to 3.14, Linux and macOS
  without the `rl` extra; one Linux job with CPU torch; a build job) and
  `.github/workflows/release.yml` (build and attach wheels on a `v*` tag).
- LICENSE (Apache-2.0), CITATION.cff, CONTRIBUTING.md, `docs/reproduce.md`,
  `scripts/reproduce.sh`. Repository URLs point at `unitaryfoundation`.

## Remaining steps

1. OWNER: the institutional copyright holder (LICENSE, first lines), the
   commit identity of the public history, and whether the package goes to
   PyPI. CITATION.cff: fill the ORCID and set `date-released` to the tag
   date; CHANGELOG.md: replace TBD.

2. RUN: make the repository public, wait for CI on `main`.

   ```
   gh repo edit unitaryfoundation/autoqec-sim --visibility public --accept-visibility-change-consequences
   gh run list --workflow ci.yml --limit 1 && gh run watch
   ```

3. RUN: tag and push the tag. The release workflow checks that the tag
   equals `uv version --short`.

   ```
   git tag -a v0.2.0 -m "autoqec-sim 0.2.0"
   git push origin v0.2.0
   gh release view v0.2.0        # sdist and wheel attached
   ```

4. OWNER then RUN: if PyPI, add a `pypa/gh-action-pypi-publish` step and a
   trusted publisher; Zenodo DOI through the GitHub integration, then paste
   the concept DOI into CITATION.cff and commit.
