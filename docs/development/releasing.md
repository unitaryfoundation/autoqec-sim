# Releasing

Releases are tagged on `main`. There is no PyPI package: the release
artifacts are the sdist and wheel attached to the GitHub release, the
Zenodo DOI, and the versioned documentation.

## Prepare

1. Set the version in `pyproject.toml` and `autoqec_sim.__version__`;
   `tests/test_version.py` pins them together.
2. Date the version's entry in `CHANGELOG.md` and set `date-released` in
   `CITATION.cff`.
3. Update "What's New" on the documentation home page (`docs/index.md`)
   with a short summary of the release and links to the new pages.
4. Build and check the documentation: `uv run --group docs mkdocs build`.
   The configuration is strict, so a broken link or an unresolved API
   reference fails the build.

## Tag

```bash
git tag -a vX.Y.Z -m "autoqec-sim X.Y.Z"
git push origin vX.Y.Z
```

The release workflow checks that the tag equals `uv version --short`,
builds the sdist and wheel, installs the wheel into a clean environment,
attaches both to a GitHub release, and then publishes the documentation
for `X.Y.Z` with the `stable` alias and makes `stable` the site's default
version.

## Documentation versions

Documentation is versioned with mike. Pushes to `main` publish the
unreleased documentation to
`https://unitaryfoundation.github.io/autoqec-sim/dev/`. A release tag
publishes `https://unitaryfoundation.github.io/autoqec-sim/X.Y.Z/` and
points `stable` at it. Until the first release, the site root redirects to
`dev`; afterwards it redirects to `stable`. The version selector in the
header lists every published version.

Deep links in the changelog and the README should carry the `stable/`
prefix, since the site only redirects at its root.

## Verify

- `https://github.com/unitaryfoundation/autoqec-sim/releases/tag/vX.Y.Z`
  has the sdist and the wheel.
- `https://unitaryfoundation.github.io/autoqec-sim/X.Y.Z/` exists and
  `stable/` shows the same content.
- The Zenodo record for the tag exists, and its concept DOI is in
  `CITATION.cff`.
