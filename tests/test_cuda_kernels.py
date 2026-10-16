"""The fused CUDA kernels agree bit for bit with the torch path (CUDA + CuPy only)."""
import numpy as np
import pytest
import torch

from autoqec_sim.backends import cuda_kernels as ck
from autoqec_sim.backends import torch_backend as tb
from autoqec_sim.errors import paulis_up_to_weight
from autoqec_sim.tableau import GATES

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and ck.cp is not None), reason="needs CUDA and CuPy")


def _entries(n):
    out = []
    for g in GATES:
        if g in ("CNOT", "CZ", "SWAP"):
            out += [(g, a, b) for a in range(n) for b in range(n) if a != b]
        else:
            out += [(g, q, -1) for q in range(n)]
    return out


def _random_tab(n, batch, steps, seed):
    rng = np.random.default_rng(seed)
    table = tb.ActionTable(_entries(n), device="cuda")
    tab = tb.identity_tableau(n, batch, device="cuda")
    ck.enable(False)
    for _ in range(steps):
        acts = torch.as_tensor(rng.integers(-1, len(table), size=batch), device="cuda")
        table.apply(tab, acts)
    return table, tab


@pytest.mark.parametrize("n,batch", [(5, 64), (9, 257), (16, 1024)])
def test_apply_actions_matches_torch(n, batch):
    table, tab = _random_tab(n, batch, 20, seed=n)
    rng = np.random.default_rng(100 + n)
    for _ in range(30):
        acts = torch.as_tensor(rng.integers(-1, len(table), size=batch), device="cuda")
        ref = tab.clone()
        ck.enable(False)
        table.apply(ref, acts)
        ck.enable(True)
        table.apply(tab, acts)
        ck.enable(False)
        assert torch.equal(tab, ref)


@pytest.mark.parametrize("n,k,batch,w", [(5, 1, 64, 2), (9, 1, 200, 2), (12, 2, 300, 3)])
def test_parity_masks_match_torch(n, k, batch, w):
    _, tab = _random_tab(n, batch, 25, seed=7 * n + k)
    errors = torch.as_tensor(paulis_up_to_weight(n, w)[0], device="cuda")
    ck.enable(False)
    und_ref = tb.undetected_mask(tab, k, errors)
    stab_ref = tb.in_stabilizer_mask(tab, k, errors)
    ck.enable(True)
    und = tb.undetected_mask(tab, k, errors)
    stab = tb.in_stabilizer_mask(tab, k, errors)
    ck.enable(False)
    assert torch.equal(und, und_ref)
    assert torch.equal(stab, stab_ref)
    assert int(und.sum()) > 0


def test_pack_bits_roundtrip():
    bits = torch.randint(0, 2, (3, 5, 130), dtype=torch.uint8, device="cuda")
    words = ck.pack_bits(bits)
    assert words.shape == (3, 5, 3)
    back = ((words.view(torch.uint8).reshape(3, 5, 24, 1) >> torch.arange(8, device="cuda", dtype=torch.uint8)) & 1).reshape(3, 5, 192)[..., :130]
    assert torch.equal(back, bits)
