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
  `tools/reproduce.sh`. Repository URLs point at `unitaryfoundation`.

## Remaining steps

1. OWNER: the institutional copyright holder (LICENSE, first lines; a
   question for UF agreements and IonQ given the PI transfer). Decided: no
   PyPI (the proposal and the data-management plan ask for a public GitHub
   release with a Zenodo DOI); commit identity is vprusso throughout.
   CITATION.cff: fill the ORCID and set `date-released` to the tag date;
   CHANGELOG.md: replace TBD.

2. RUN: `tools/release.sh v0.2.0` dates CITATION.cff and CHANGELOG.md,
   commits, makes the repository public, tags, and pushes. GitHub Pages is
   already enabled on the `gh-pages` branch the docs workflow maintains (on
   this plan the Pages site is public while the repository is still
   private). The release workflow checks that the tag
   equals `uv version --short` and attaches the sdist and wheel.

4. RUN: Zenodo DOI through the GitHub integration (enable the repository
   on zenodo.org before the tag is pushed), then paste the concept DOI into
   CITATION.cff and as a `doi` field of the README's BibTeX entry, and
   commit. GitHub Pages is enabled (source: gh-pages);
   the release workflow publishes the `0.2.0` docs and `stable` from the
   tag, and the docs workflow publishes `dev` from `main`.
