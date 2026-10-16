#!/usr/bin/env python3
"""Time the torch path against the fused CUDA kernels: ActionTable.apply and
undetected_mask at a few (n, batch, error weight) cells. Writes JSON with --json."""
import argparse
import json
import time

import numpy as np
import torch

from autoqec_sim.backends import cuda_kernels as ck
from autoqec_sim.backends import torch_backend as tb
from autoqec_sim.errors import paulis_up_to_weight
from autoqec_sim.tableau import GATES


def entries(n):
    out = []
    for g in GATES:
        if g in ("CNOT", "CZ", "SWAP"):
            out += [(g, a, b) for a in range(n) for b in range(n) if a != b]
        else:
            out += [(g, q, -1) for q in range(n)]
    return out


def timeit(fn, min_time=0.5):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 0
    while time.perf_counter() - t0 < min_time:
        fn()
        reps += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--cells", default="5,1,1024,2;9,1,1024,2;16,1,1024,2;16,1,4096,2;25,1,1024,2;25,1,1024,3")
    args = ap.parse_args()
    rows = []
    for cell in args.cells.split(";"):
        n, k, batch, w = (int(x) for x in cell.split(","))
        table = tb.ActionTable(entries(n), device="cuda")
        tab = tb.identity_tableau(n, batch, device="cuda")
        rng = np.random.default_rng(0)
        acts = torch.as_tensor(rng.integers(0, len(table), size=batch), device="cuda")
        errors = torch.as_tensor(paulis_up_to_weight(n, w)[0], device="cuda")
        ck.enable(False)
        for _ in range(10):
            table.apply(tab, acts)
        t_apply_torch = timeit(lambda: table.apply(tab, acts))
        t_mask_torch = timeit(lambda: tb.undetected_mask(tab, k, errors))
        ck.enable(True)
        t_apply_k = timeit(lambda: table.apply(tab, acts))
        t_mask_k = timeit(lambda: tb.undetected_mask(tab, k, errors))
        ck.enable(False)
        row = {"n": n, "k": k, "batch": batch, "errors": int(errors.shape[0]),
               "apply_torch_us": t_apply_torch * 1e6, "apply_kernel_us": t_apply_k * 1e6,
               "mask_torch_ms": t_mask_torch * 1e3, "mask_kernel_ms": t_mask_k * 1e3}
        row["apply_speedup"] = row["apply_torch_us"] / row["apply_kernel_us"]
        row["mask_speedup"] = row["mask_torch_ms"] / row["mask_kernel_ms"]
        rows.append(row)
        print(f"n={n:2d} B={batch:5d} M={row['errors']:6d}  apply {row['apply_torch_us']:8.1f} -> {row['apply_kernel_us']:7.1f} us ({row['apply_speedup']:.1f}x)"
              f"  mask {row['mask_torch_ms']:8.3f} -> {row['mask_kernel_ms']:7.3f} ms ({row['mask_speedup']:.1f}x)", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
