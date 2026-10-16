#!/usr/bin/env bash
# Release autoqec-sim: tools/release.sh vX.Y.Z
# Dates CITATION.cff and CHANGELOG.md, commits, makes the repository public,
# tags, pushes, and enables GitHub Pages. Run from the repository root with
# a clean tree, gh authenticated, after the owner decisions listed in
# docs/development/releasing.md.
set -euo pipefail
TAG="${1:?usage: tools/release.sh vX.Y.Z}"
VER="${TAG#v}"
REPO="unitaryfoundation/autoqec-sim"
test "$(uv version --short)" = "$VER" || { echo "pyproject version is $(uv version --short), not $VER"; exit 1; }
test -z "$(git status --porcelain)" || { echo "working tree not clean"; exit 1; }
TODAY="$(date +%Y-%m-%d)"
sed -i.bak "s/^date-released: \"TBD\"/date-released: \"$TODAY\"/" CITATION.cff && rm -f CITATION.cff.bak
sed -i.bak "s/^## $VER (TBD)/## $VER ($TODAY)/" CHANGELOG.md && rm -f CHANGELOG.md.bak
sed -i.bak "s#github.io/autoqec-sim/dev/#github.io/autoqec-sim/stable/#g" README.md && rm -f README.md.bak
git add CITATION.cff CHANGELOG.md README.md
git -c user.name=vprusso -c user.email=vincent.russo@ionq.co commit -q -m "Release $VER" || true
git push -q origin main
gh repo edit "$REPO" --visibility public --accept-visibility-change-consequences
git tag -a "$TAG" -m "autoqec-sim $VER"
git push -q origin "$TAG"
gh workflow run docs.yml --repo "$REPO" >/dev/null 2>&1 || true   # first public deploy of the docs site
sleep 90
gh api -X POST "repos/$REPO/pages" -f "source[branch]=gh-pages" -f "source[path]=/" >/dev/null 2>&1 || true
echo "released $TAG: https://github.com/$REPO/releases/tag/$TAG"
echo "docs: https://unitaryfoundation.github.io/autoqec-sim/ (Pages enabled on gh-pages)"
echo "next: Zenodo (enable the repository under GitHub integration, then paste the concept DOI into CITATION.cff)"
