"""Paths and fixtures. The [[9,1,3]] skeleton circuits are vendored under
tests/data. The comparison targets for the byte-identity tests
(qldpc-challenge verify/circuit_tools.py and the frozen ion_noise.py, which
imports it) are located through QLDPC_ROOT or as the sibling checkout
genesis/qldpc-challenge; tests needing them skip when absent.

    uv sync --group dev && uv run pytest              # 23 tests without the board
    QLDPC_ROOT=/path/to/qldpc-challenge uv run pytest # all 69
"""

import os
import sys

import pytest
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, "reference")
REPO_DIR = os.path.dirname(os.path.dirname(HERE))
GENESIS = os.path.dirname(REPO_DIR)
QLDPC_ROOT = os.environ.get("QLDPC_ROOT", os.path.join(GENESIS, "qldpc-challenge"))
# The six [[9,1,3]] skeletons are vendored under tests/data/9-1-3 (copied from
# genesis/ion-experiment/circuits/9-1-3 on 2026-09-20); IONEXP_ROOT points at
# a checkout of that experiment to test against its copy instead.
IONEXP_ROOT = os.environ.get("IONEXP_ROOT")
CIRC_913 = (os.path.join(IONEXP_ROOT, "circuits", "9-1-3") if IONEXP_ROOT
            else os.path.join(HERE, "data", "9-1-3"))

for p in (REFERENCE_DIR, os.path.join(QLDPC_ROOT, "verify")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _have_board():
    return os.path.exists(os.path.join(QLDPC_ROOT, "verify", "circuit_tools.py"))


def _have_913():
    return os.path.exists(os.path.join(CIRC_913, "one_x.stim"))


needs_board = pytest.mark.skipif(not _have_board(), reason="qldpc-challenge checkout not found")
needs_913 = pytest.mark.skipif(not _have_913(), reason="ion-experiment [[9,1,3]] circuits not found")


@pytest.fixture(scope="session")
def ct():
    if not _have_board():
        pytest.skip("qldpc-challenge checkout not found")
    import circuit_tools
    return circuit_tools


@pytest.fixture(scope="session")
def ion_noise():
    if not _have_board():
        pytest.skip("ion_noise needs the qldpc-challenge checkout")
    if not os.path.exists(os.path.join(REFERENCE_DIR, "ion_noise.py")):
        pytest.skip("tests/noise/reference/ion_noise.py not found")
    import ion_noise as m
    return m


def _load_913(name):
    from autoqec_sim.noise import strip_noise
    return strip_noise(stim.Circuit.from_file(os.path.join(CIRC_913, name)))


@pytest.fixture(scope="session")
def skel_913():
    """{name: noiseless skeleton} for the [[9,1,3]] ion-experiment circuits:
    one-round X and Z, mid-circuit-measurement X and Z, multi-round X and Z."""
    if not _have_913():
        pytest.skip("[[9,1,3]] circuits not found")
    return {k: _load_913(k + ".stim") for k in
            ("one_x", "one_z", "mcm_x", "mcm_z", "multi_x", "multi_z")}


N_DATA_913 = 9
