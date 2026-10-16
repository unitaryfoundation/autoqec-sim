#!/usr/bin/env python
# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Throughput and memory profile of the vectorized simulator.

For every ``(n, batch)`` cell: gate applications per second of the batched
tableau (``ActionTable.apply`` on a uniform mix of H, S, and CNOT, the path
``CodeDiscoveryEnv.step`` takes), reward evaluations per second of the
Knill-Laflamme reward used by the regression (``kl_reward`` over all Paulis
of weight 1 and 2 plus the weight-1 stabilizer penalty), bytes per batch
element of the tableau and of the reward's intermediate product, and
environment steps per second (batch elements per second, the ``env_steps``
axis of the PPO agent) of ``CodeDiscoveryEnv`` under uniformly random
actions. One breakdown cell times the pieces of ``env.step`` and of
``undetected_mask``.

``--backend numpy`` (default) profiles the numpy path on the host CPU.
``--backend torch --device {cpu,cuda}`` profiles
``autoqec_sim.backends.torch_backend`` on that device; the environment
stays numpy and calls the torch reward through ``numpy_reward_adapter``
(one host-to-device copy of the tableau per step), so its env-steps column
is the hybrid path. ``--json`` records the run; ``--render`` turns one or
more such records back into the Markdown tables.

    uv run python scripts/profile_tableau.py --out docs/profile.md
    uv run --extra rl python scripts/profile_tableau.py --backend torch --device cpu --json out.json
    uv run python scripts/profile_tableau.py --render out.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import textwrap
import time
from pathlib import Path

THREADS = int(os.environ.get("AUTOQEC_PROFILE_THREADS", "6"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoqec_sim import graphs
from autoqec_sim.env import CodeDiscoveryEnv, build_action_table
from autoqec_sim.errors import ErrorSet
from autoqec_sim.reward import CompositeReward, code_rows, make_kl_reward, make_low_weight_penalty
from autoqec_sim.symplectic import swap_halves
from autoqec_sim.tableau import identity_tableau

GATES = ("H", "S", "CNOT")
K = 1
D = 3  # error set: every Pauli of weight <= d - 1 = 2
MAX_STEPS = 30


def timeit(fn, min_time: float, max_calls: int = 10_000, sync=None) -> tuple[float, int]:
    """Call ``fn`` until ``min_time`` seconds have elapsed (at least once);
    returns ``(seconds, calls)``. ``sync`` (device synchronization) runs
    after every call, inside the timed region."""
    fn()  # warm-up (allocations, BLAS thread pool, CUDA context)
    if sync is not None:
        sync()
    calls = 0
    t0 = time.perf_counter()
    while True:
        fn()
        if sync is not None:
            sync()
        calls += 1
        dt = time.perf_counter() - t0
        if dt >= min_time or calls >= max_calls:
            return dt, calls


def reward_intermediate_bytes(n: int, m: int) -> int:
    """Peak bytes per batch element inside the numpy ``undetected_mask``: the
    float32 product ``(chunk, n + k)`` plus its int64 cast for the mod-2
    reduction."""
    return m * (n + K) * (4 + 8)


def random_tableau(n: int, batch: int, table, rng, steps: int = 20) -> np.ndarray:
    tab = identity_tableau(n, batch)
    for _ in range(steps):
        table.apply(tab, rng.integers(len(table), size=batch))
    return tab


# --------------------------------------------------------------------------
# backends: the pieces of a cell that differ between numpy and torch
# --------------------------------------------------------------------------
class NumpyBackend:
    name = "numpy"
    device = "cpu"
    sync = None

    def reward_bytes(self, n: int, m: int) -> int:
        return reward_intermediate_bytes(n, m)

    def prepare(self, tab: np.ndarray, table, acts: np.ndarray):
        """Returns ``(tab, gate_call, reward, reward_fn_for_env)``."""
        i = [0]

        def gate_call():
            table.apply(tab, acts[i[0] % acts.shape[0]])
            i[0] += 1

        return tab, gate_call, self.reward, self.reward

    def make_reward(self, errors: ErrorSet):
        self.reward = CompositeReward([(1.0, make_kl_reward(errors)), (1.0, make_low_weight_penalty(1))])
        return self.reward

    def reward_pieces(self, tab: np.ndarray, errors: ErrorSet, n: int) -> dict:
        e = errors.paulis
        rows = code_rows(tab, K)
        a32 = np.asarray(e[None], dtype=np.float32)
        b32 = np.asarray(swap_halves(rows), dtype=np.float32)
        bt = np.swapaxes(b32, -1, -2)
        prod = np.matmul(a32, bt)
        p8 = (prod.astype(np.int64) & 1).astype(np.uint8)
        return {
            "      code_rows and float32 casts": lambda: (
                np.asarray(e[None], dtype=np.float32),
                np.asarray(swap_halves(code_rows(tab, K)), dtype=np.float32),
            ),
            "      GEMM (M, 2n) x (2n, n + k)": lambda: np.matmul(a32, bt),
            "      int64 cast and & 1 (mod 2)": lambda: (prod.astype(np.int64) & 1).astype(np.uint8),
            "      two any() reductions over n + k": lambda: ~p8[:, :, : n - K].any(axis=2) & p8[:, :, n - K :].any(axis=2),
            "      und @ lam and normalization": lambda: (
                -(p8[:, :, 0].astype(np.float64) @ errors.lam) / float(errors.lam.sum())
            ),
        }

    def machine_extra(self) -> str:
        return ""


class TorchBackend:
    name = "torch"

    def __init__(self, device: str):
        import torch

        from autoqec_sim.backends import torch_backend as tb

        self.torch = torch
        self.tb = tb
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
        self.device = device
        self.dev = torch.device(device)
        torch.set_num_threads(THREADS)
        self.sync = torch.cuda.synchronize if device == "cuda" else None

    def reward_bytes(self, n: int, m: int) -> int:
        return self.tb.reward_intermediate_bytes(n, K, m)

    def prepare(self, tab: np.ndarray, table, acts: np.ndarray):
        tb = self.tb
        tab_t = tb.from_numpy(tab, self.dev)
        tt = tb.ActionTable(table, device=self.dev)
        acts_t = self.torch.as_tensor(acts, device=self.dev)
        i = [0]

        def gate_call():
            tt.apply(tab_t, acts_t[i[0] % acts_t.shape[0]])
            i[0] += 1

        return tab_t, gate_call, self.reward, tb.numpy_reward_adapter(self.reward, self.dev)

    def make_reward(self, errors: ErrorSet):
        tb = self.tb
        self.reward = tb.CompositeReward(
            [(1.0, tb.make_kl_reward(errors, device=self.dev)), (1.0, tb.make_low_weight_penalty(1))]
        )
        return self.reward

    def reward_pieces(self, tab: np.ndarray, errors: ErrorSet, n: int) -> dict:
        torch, tb = self.torch, self.tb
        tab_t = tb.from_numpy(tab, self.dev)
        e32 = torch.as_tensor(errors.paulis, device=self.dev).to(torch.float32)
        rows_t = tb._rows32_swapped(tab_t, K)
        w = tb._score_weights(n, K, self.dev)
        lam = torch.as_tensor(errors.lam, dtype=torch.float64, device=self.dev)
        prod = torch.matmul(e32.unsqueeze(0), rows_t)
        p = prod.remainder(2)
        score = torch.matmul(p, w)
        und = (score >= 1) & (score <= 2 * K)
        return {
            "      host-to-device copy of the tableau": lambda: tb.from_numpy(tab, self.dev),
            "      code_rows and float32 casts": lambda: tb._rows32_swapped(tab_t, K),
            "      GEMM (M, 2n) x (2n, n + k)": lambda: torch.matmul(e32.unsqueeze(0), rows_t),
            "      remainder_(2) in place (mod 2)": lambda: prod.remainder_(2),
            "      GEMV score and two compares": lambda: (
                (lambda s: (s >= 1) & (s <= 2 * K))(torch.matmul(p, w))
            ),
            "      und @ lam and normalization": lambda: -(und.to(torch.float64) @ lam) / float(errors.lam.sum()),
        }

    def machine_extra(self) -> str:
        torch = self.torch
        s = f"; torch {torch.__version__} on {self.device}"
        if self.device == "cuda":
            p = torch.cuda.get_device_properties(0)
            s += f" ({p.name}, {p.total_memory / 2**30:.0f} GiB, CUDA {torch.version.cuda})"
        else:
            s += f" ({torch.get_num_threads()} threads)"
        return s


def make_backend(name: str, device: str):
    if name == "numpy":
        if device != "cpu":
            raise SystemExit("--backend numpy runs on the CPU only")
        return NumpyBackend()
    return TorchBackend(device)


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------
def profile_cell(be, n: int, batch: int, min_time: float, mem_cap: int, rng) -> dict:
    graph = graphs.all_to_all(n)
    table = build_action_table(graph, GATES)
    errors = ErrorSet.up_to_weight(n, D - 1)
    m = errors.paulis.shape[0]
    reward = be.make_reward(errors)
    tab_np = random_tableau(n, batch, table, rng)
    rec = dict(
        n=n,
        batch=batch,
        num_actions=len(table),
        num_errors=m,
        tableau_bytes=tab_np.nbytes // batch,
        reward_bytes=be.reward_bytes(n, m),
    )
    acts = rng.integers(len(table), size=(64, batch))
    tab, gate_call, reward, env_reward = be.prepare(tab_np, table, acts)
    dt, calls = timeit(gate_call, min_time, sync=be.sync)
    rec["gates_per_s"] = batch * calls / dt
    # reward and env: skip when the reward intermediate would exceed the cap
    need = rec["reward_bytes"] * batch
    if need > mem_cap:
        rec["reward_per_s"] = None
        rec["env_steps_per_s"] = None
        rec["skipped_gb"] = need / 2**30
        return rec
    dt, calls = timeit(lambda: reward(tab, K), min_time, sync=be.sync)
    rec["reward_per_s"] = batch * calls / dt
    env = CodeDiscoveryEnv(
        n=n,
        k=K,
        graph=graph,
        batch_size=batch,
        max_steps=MAX_STEPS,
        gates=GATES,
        reward_fn=env_reward,
        auto_reset=True,
        seed=int(rng.integers(2**31)),
    )
    env.reset()
    dt, calls = timeit(lambda: env.step(env.sample_actions()), min_time)
    rec["env_steps_per_s"] = batch * calls / dt
    return rec


def breakdown(be, n: int, batch: int, min_time: float, rng) -> dict:
    """Seconds per call of the pieces of one environment step.

    The reward and the gate kernels are timed in situ (their calls inside
    ``env.step`` are wrapped), the pieces of ``undetected_mask`` on a copy
    of the environment's tableau (the cost is independent of the content).
    With the torch backend the environment is numpy and the reward runs
    through ``numpy_reward_adapter``; the in-situ reward time includes the
    host-to-device copy, and the gate row is the numpy kernel.
    """
    graph = graphs.all_to_all(n)
    table = build_action_table(graph, GATES)
    errors = ErrorSet.up_to_weight(n, D - 1)
    reward = be.make_reward(errors)
    _, _, _, env_reward = be.prepare(random_tableau(n, batch, table, rng), table, np.zeros((1, batch), dtype=np.int64))
    env = CodeDiscoveryEnv(
        n=n,
        k=K,
        graph=graph,
        batch_size=batch,
        max_steps=MAX_STEPS,
        gates=GATES,
        reward_fn=env_reward,
        auto_reset=True,
        seed=0,
    )
    env.reset()
    acc = {"reward": 0.0, "gates": 0.0, "kl": 0.0, "penalty": 0.0}

    def timed(name, fn):
        def g(*a, **kw):
            t = time.perf_counter()
            out = fn(*a, **kw)
            if be.sync is not None:
                be.sync()
            acc[name] += time.perf_counter() - t
            return out

        return g

    kl_fn, pen_fn = (fn for _, fn in reward.terms)
    reward.terms = [(1.0, timed("kl", kl_fn)), (1.0, timed("penalty", pen_fn))]
    env.reward_fn = timed("reward", env_reward)
    env.action_table.apply = timed("gates", table.apply)
    dt, calls = timeit(lambda: env.step(env.sample_actions()), min_time)
    calls += 1  # the warm-up call also went through the timers
    gate_label = "  ActionTable.apply" if be.name == "numpy" else "  ActionTable.apply (numpy env)"
    out = {
        "env.step": dt / (calls - 1),
        gate_label: acc["gates"] / calls,
        "  reward": acc["reward"] / calls,
        "    Knill-Laflamme term": acc["kl"] / calls,
        "    weight-1 stabilizer penalty": acc["penalty"] / calls,
    }
    for name, fn in be.reward_pieces(env.tab.copy(), errors, n).items():
        dt, calls = timeit(fn, min_time, sync=be.sync)
        out[name] = dt / calls
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def fmt_rate(x) -> str:
    if x is None:
        return "not run"
    if x >= 1e6:
        return f"{x / 1e6:.2f} M"
    if x >= 1e3:
        return f"{x / 1e3:.1f} k"
    return f"{x:.0f}"


def fmt_bytes(b: int) -> str:
    if b >= 2**20:
        return f"{b / 2**20:.1f} MiB"
    if b >= 2**10:
        return f"{b / 2**10:.1f} KiB"
    return f"{b} B"


def table_md(recs: list[dict], mem_cap: int) -> str:
    lines = [
        "| n | batch | actions | errors (w <= 2) | tableau B/elem | reward intermediate B/elem | gate applications/s | reward evaluations/s | env steps/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        if r.get("skipped_gb") is not None:
            tail = f"| not run ({r['skipped_gb']:.1f} GiB intermediate > {mem_cap / 2**30:.0f} GiB cap) | not run |"
        else:
            tail = f"| {fmt_rate(r['reward_per_s'])} | {fmt_rate(r['env_steps_per_s'])} |"
        lines.append(
            f"| {r['n']} | {r['batch']} | {r['num_actions']} | {r['num_errors']:,} | {fmt_bytes(r['tableau_bytes'])} "
            f"| {fmt_bytes(r['reward_bytes'])} | {fmt_rate(r['gates_per_s'])} {tail}"
        )
    return "\n".join(lines)


def breakdown_md(bd: dict, n: int, batch: int) -> str:
    total = bd["env.step"]
    lines = [f"| piece (n = {n}, batch = {batch}) | ms per call | share of env.step |", "|---|---|---|"]
    for k, v in bd.items():
        lines.append(f"| `{k.strip()}` | {1e3 * v:.2f} | {100 * v / total:.0f}% |")
    return "\n".join(lines)


def blas_name() -> str:
    try:
        b = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return " ".join(str(b[key]) for key in ("name", "version") if b.get(key) not in (None, "", "unknown"))
    except Exception:  # pragma: no cover - older numpy
        return "unknown"


def machine_text(be, elapsed: float) -> str:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            import subprocess

            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return (
        f"Machine: {platform.platform()}, {cpu}, {os.cpu_count()} logical CPUs, BLAS threads capped at "
        f"{os.environ['OPENBLAS_NUM_THREADS']} (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, "
        f"`VECLIB_MAXIMUM_THREADS`); Python {platform.python_version()}, numpy {np.__version__} "
        f"(BLAS: {blas_name()}){be.machine_extra()}. Backend: {be.name} on {be.device}. "
        f"Whole run: {elapsed:.0f} s on {time.strftime('%Y-%m-%d')}."
    )


def machine_md(text: str) -> str:
    return textwrap.fill(text, width=76)


def record_md(rec: dict) -> str:
    """Markdown tables of one ``--json`` record."""
    parts = [machine_md(rec["machine"]), "", table_md(rec["cells"], int(rec["mem_cap_bytes"]))]
    if rec.get("breakdown"):
        b = rec["breakdown"]
        parts += ["", breakdown_md(b["pieces"], b["n"], b["batch"])]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="device of the torch backend")
    ap.add_argument("--ns", nargs="*", type=int, default=[5, 9, 16, 25, 49])
    ap.add_argument("--batches", nargs="*", type=int, default=[256, 1024, 4096])
    ap.add_argument("--min-time", type=float, default=0.3, help="seconds of repeated calls per measurement")
    ap.add_argument(
        "--mem-cap-gb", type=float, default=4.0, help="skip reward and env cells whose intermediate exceeds this"
    )
    ap.add_argument("--breakdown", nargs=2, type=int, default=[16, 1024], metavar=("N", "BATCH"))
    ap.add_argument("--no-breakdown", action="store_true")
    ap.add_argument("--breakdown-time", type=float, default=1.0, help="seconds of repeated calls per breakdown piece")
    ap.add_argument("--out", default=None, help="write docs/profile.md-style Markdown here")
    ap.add_argument("--json", default=None, help="write the run (cells, breakdown, machine) as JSON here")
    ap.add_argument(
        "--render", nargs="*", default=None, metavar="JSON", help="print the Markdown tables of these JSON records and exit"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.render is not None:
        for i, path in enumerate(args.render):
            if i:
                print()
            print(record_md(json.loads(Path(path).read_text())))
        return 0
    be = make_backend(args.backend, args.device)
    mem_cap = int(args.mem_cap_gb * 2**30)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    recs = []
    for n in args.ns:
        for batch in args.batches:
            rec = profile_cell(be, n, batch, args.min_time, mem_cap, rng)
            recs.append(rec)
            print(
                f"n={n:2d} batch={batch:4d} gates/s={fmt_rate(rec['gates_per_s']):>8} reward/s={fmt_rate(rec['reward_per_s']):>8} "
                f"env/s={fmt_rate(rec['env_steps_per_s']):>8}",
                file=sys.stderr,
                flush=True,
            )
    bn, bb = args.breakdown
    bd = None if args.no_breakdown else breakdown(be, bn, bb, args.breakdown_time, rng)
    elapsed = time.perf_counter() - t0
    machine = machine_text(be, elapsed)
    table = table_md(recs, mem_cap)
    print(table)
    bdt = None
    if bd is not None:
        bdt = breakdown_md(bd, bn, bb)
        print()
        print(bdt)
    if args.json:
        rec = dict(
            backend=be.name,
            device=be.device,
            machine=machine,
            date=time.strftime("%Y-%m-%d"),
            elapsed_s=elapsed,
            mem_cap_bytes=mem_cap,
            gates=list(GATES),
            k=K,
            d=D,
            cells=recs,
            breakdown=None if bd is None else dict(n=bn, batch=bb, pieces=bd),
        )
        Path(args.json).write_text(json.dumps(rec, indent=1))
        print(f"wrote {args.json}", file=sys.stderr)
    if args.out:
        if be.name == "numpy" and bd is not None:
            text = render_doc(table, bdt, machine_md(machine), recs, bd, bn, bb)
        else:
            text = render_generic(be, table, bdt, machine_md(machine), bn, bb)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def render_generic(be, table: str, bdt: str | None, machine: str, bn: int, bb: int) -> str:
    body = f"""# Profile of the vectorized simulator ({be.name} backend, {be.device})

Output of `scripts/profile_tableau.py --backend {be.name} --device {be.device}`.
Columns as in `docs/profile.md`; with the torch backend the reward
intermediate is the float32 `(B, M, n + k)` block reduced in place (`4 M (n +
k)` bytes per element) and the env-steps column is the numpy environment
calling the torch reward through `numpy_reward_adapter`.

{machine}

{table}
"""
    if bdt:
        body += f"\n## Where the time goes\n\n{bdt}\n"
    return body


def render_doc(table: str, bdt: str, machine: str, recs: list[dict], bd: dict, bn: int, bb: int) -> str:
    step = bd["env.step"]
    r_share = bd["  reward"] / step
    g_share = bd["  ActionTable.apply"] / step
    kl = bd["    Knill-Laflamme term"]
    gemm = bd["      GEMM (M, 2n) x (2n, n + k)"] / kl
    cast = bd["      int64 cast and & 1 (mod 2)"] / kl
    red = bd["      two any() reductions over n + k"] / kl
    return f"""# CPU profile of the vectorized simulator

Throughput and memory of the batched tableau, the Knill-Laflamme reward,
and `CodeDiscoveryEnv`, measured by `scripts/profile_tableau.py` (stdlib
and numpy; no torch). This is a numpy profile on the host CPU of one
laptop, not the GPU-vectorized simulator the roadmap's week-4 wording
names; the cuQuantum kernels have not been written (README, "What is left
for cuQuantum").

{machine}

Setting per cell: all-to-all graph, gates H, S, CNOT (`actions` is the
size of the action table), `k = 1`, reward = normalized Knill-Laflamme term
over every Pauli of weight 1 and 2 (`errors`) plus the weight-1 stabilizer
penalty, the reward of the rediscovery regression. Rates are batch elements
per second: one `ActionTable.apply` call on a batch of `B` is `B` gate
applications, one reward call is `B` reward evaluations, one `env.step`
is `B` env steps (the `env_steps` axis of the PPO agent). Each rate is the
mean over at least 0.3 s of repeated calls after one warm-up call, on a
machine that was otherwise idle. The tableau column is `2n (2n + 1)` bytes
of `uint8` per element; the reward column is the peak of the `(B, M, n +
k)` float32 product and its int64 copy inside `undetected_mask`, `12 M (n
+ k)` bytes per element. Cells whose intermediate exceeds the cap are not
run; the gate rate does not depend on the reward and is measured
everywhere.

{table}

## Where the time goes

The reward and the gate kernels are timed inside `env.step`; the pieces of
the Knill-Laflamme term are timed on their own on the same arrays, and
their sum is within measurement noise of the term.

{bdt}

At `n = {bn}`, batch {bb}, the reward is {100 * r_share:.0f}% of an environment step and
the gate kernels {100 * g_share:.0f}%; the remainder is `env.step` bookkeeping (history, masks,
reset of finished elements). Inside the Knill-Laflamme term the float32
GEMM of the error set against the code rows is {100 * gemm:.0f}%, the int64 cast for
the mod-2 reduction {100 * cast:.0f}%, and the two boolean `any()` reductions over the
`(B, M, n + k)` product {100 * red:.0f}%, so the README's list is right that the
symplectic product of the error set against the code rows is the hot path
but not that the GEMM is the whole of its cost: on this CPU the mod-2 cast
and the reductions on the product are most of it. A bit-packed popcount
product with the parity and the commutation test fused into one kernel
would remove the GEMM, the cast, the reductions, and the `(B, M, n + k)`
intermediate, which is also the memory bottleneck (GiB per batch at `n >=
25` against kilobytes for the tableau itself). The gate kernels run at 1 to
9 million gate applications per second and are under 3% of a step at every
`n` (the ratio of the env-steps and gate-applications columns); they are
not what a GPU port has to speed up first.
"""


if __name__ == "__main__":
    raise SystemExit(main())
