"""The quick-start examples are files under docs/examples: the documentation
includes their marked sections and README.md carries copies, which this test
keeps equal and runs."""

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
    section = readme[readme.index("## Quick Start"):readme.index("## Contributing")]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) == 3
    assert blocks[0].strip() == _section("docs/examples/evaluate_code.py", "proxy")
    assert blocks[1].strip() == _section("docs/examples/quickstart.py", "score")
    assert blocks[2].strip() == _section("docs/examples/quickstart.py", "search")


def test_quick_start_runs_and_finds_a_distance_3_code():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "docs" / "examples" / "quickstart.py")],
        capture_output=True, text=True, check=True, timeout=300,
    )
    assert "steane (7, 1, 3)" in proc.stdout and "(5, 1, 3)" in proc.stdout, proc.stdout
