from importlib.metadata import version

import autoqec_sim


def test_version_matches_metadata():
    assert autoqec_sim.__version__ == version("autoqec-sim")
