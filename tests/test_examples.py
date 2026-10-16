"""The quick-start example is one file: the documentation includes its marked
section and README.md carries a copy, which this test keeps equal and runs."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _section(path: str, name: str) -> str:
    text = (ROOT / path).read_text()
    m = re.search(rf"# --8<-- \[start:{name}\]\n(.*?)# --8<-- \[end:{name}\]", text, re.S)
    assert m is not None, f"no section {name} in {path}"
    return m.group(1).strip()


def test_readme_quick_start_equals_the_example():
    readme = (ROOT / "README.md").read_text()
    m = re.search(r"## Quick Start\n\n```python\n(.*?)```", readme, re.S)
    assert m is not None
    assert m.group(1).strip() == _section("docs/examples/quickstart.py", "quickstart")


def test_quick_start_runs_and_finds_a_distance_3_code():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "docs" / "examples" / "quickstart.py")],
        capture_output=True, text=True, check=True, timeout=300,
    )
    assert proc.stdout.startswith("(5, 1, 3)"), proc.stdout
