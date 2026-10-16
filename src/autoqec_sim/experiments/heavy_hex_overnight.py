# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Unattended heavy-hex run of the PPO agent with checkpoints, a heartbeat,
per-run diagnostics, and per-run resource accounting.

The matrix (:data:`CELLS`) is the heavy-hex column of
:mod:`autoqec_sim.experiments.topologies` widened to six seeds and extended
with a harder target: ``heavy_hex:1x1`` (12 qubits) and ``heavy_hex:2x1``
(18 qubits) at ``[[n, 1, d >= 3]]`` with no idle qubit, and ``heavy_hex:2x1``
at ``[[18, 1, d >= 4]]``. The objective, environment, reward
(:class:`autoqec_sim.experiments.scaling.LevelDistanceReward`), agent
defaults, and the metric set (:func:`autoqec_sim.experiments.topologies.
describe_code`) are those of the topology matrix; the only change is the
target distance, which this module sets per cell. The next heavy-hex size
(``heavy_hex:1x2``, 21 qubits) is excluded because
:meth:`autoqec_sim.code.StabilizerCode.distance_exact` refuses ``n > 20``.

Files under the output directory (default ``results/heavy_hex_overnight``):

- ``run_<spec>_n<n>_d<d>_seed<s>.json``: one record per finished run,
  written when the run completes; these are the checkpoints, and ``drive``
  skips every cell whose file exists.
- ``heartbeat.json``: rewritten every minute by the driver with the current
  cell, elapsed time, and progress.
- ``log.txt``: one START and one DONE line per run with steps, wall, success,
  best reward, first-success steps, and the code's ``d`` and ``w*``.
- ``logs/<run>.log``: the child's stdout (PPO progress every 25 updates).
- ``resources.jsonl``: one line per run with CPU seconds (user and system)
  and peak resident set size, from ``resource.getrusage`` in the child
  (``RUSAGE_SELF``) and in the driver (``RUSAGE_CHILDREN`` delta), plus
  wall time and exit status.
- ``driver.pid``: the driver's pid while it runs.
- ``SUMMARY.md``: written by ``report`` from the JSONs.

Each run is a separate Python process, so a crash in one run is one missing
checkpoint, not a lost night, and the peak memory is that run's alone.

    python -m autoqec_sim.experiments.heavy_hex_overnight plan
    python -m autoqec_sim.experiments.heavy_hex_overnight drive --hours 10 --threads 4
    python -m autoqec_sim.experiments.heavy_hex_overnight report
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_OUT = Path("results/heavy_hex_overnight")
SEEDS = tuple(range(6))
THREADS = 4
BATCH = 256
GATES = ("H", "CNOT")
MAX_STEPS = 50
HEARTBEAT_S = 60.0
MIN_RUN_MINUTES = 5.0  # a cell is skipped (not started) when less than this remains
SKIP_EXIT = 3  # exit status of a child that found its checkpoint already written


@dataclass(frozen=True)
class Cell:
    spec: str  # graph spec, e.g. heavy_hex:2x1
    n: int
    d_target: int
    budget_env_steps: int
    minutes: float

    def name(self, seed: int) -> str:
        return f"run_{self.spec.replace(':', '_')}_n{self.n}_d{self.d_target}_seed{seed}"


# In run order. The two d >= 3 cells are the heavy-hex rows of the topology
# matrix (same budgets; the wall cap of the 18-qubit cell is raised from 12
# to 20 min); the d >= 4 cell is the harder target with a 60 min cap.
CELLS = (
    Cell("heavy_hex:1x1", 12, 3, 2_000_000, 12.0),
    Cell("heavy_hex:2x1", 18, 3, 4_000_000, 20.0),
    Cell("heavy_hex:2x1", 18, 4, 4_000_000, 60.0),
)


def jobs(seeds=SEEDS) -> list[tuple[Cell, int]]:
    return [(c, s) for c in CELLS for s in seeds]


def run_path(out_dir: Path, cell: Cell, seed: int) -> Path:
    return out_dir / f"{cell.name(seed)}.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def rusage_dict(who) -> dict:
    ru = resource.getrusage(who)
    maxrss = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024  # bytes on macOS, KiB on Linux
    return {"utime_s": ru.ru_utime, "stime_s": ru.ru_stime, "maxrss_bytes": int(maxrss)}


# --------------------------------------------------------------------------
# one run (child process)
# --------------------------------------------------------------------------
def run_one(cell: Cell, seed: int, threads: int, minutes: float | None = None, verbose: bool = True, log_every: int = 25) -> dict:
    """One PPO run on ``cell.spec`` toward ``[[n, 1, d >= d_target]]`` with
    no idle qubit; the record of :func:`topologies.run_cell` plus the best
    terminal reward, the target distance, and the child's resource usage."""
    import numpy as np

    from .. import graphs
    from ..agents.ppo import PPOAgent, PPOConfig
    from . import topologies as T
    from .scaling import LevelDistanceReward, replay_gates

    # TrackingEnv reads the module-level target distance; one run per process
    T.D_TARGET = cell.d_target
    minutes = cell.minutes if minutes is None else min(cell.minutes, float(minutes))
    graph = graphs.from_spec(cell.spec)
    assert graph.n == cell.n, (graph.n, cell.n)
    load = T.load_1min()
    reward = LevelDistanceReward(cell.d_target, cell.d_target)
    env = T.TrackingEnv(
        n=graph.n, k=1, graph=graph, batch_size=BATCH, max_steps=MAX_STEPS, gates=GATES,
        reward_fn=reward, seed=seed, auto_reset=True,
    )
    agent = PPOAgent(env, config=PPOConfig(max_threads=threads), seed=seed)

    def cb(agent, new):
        if verbose and (agent.updates % log_every == 0 or new):
            h = agent.history[-1]
            print(
                f"{now()} {cell.spec} d>={cell.d_target} seed={seed} upd {h['update']:5d} steps {h['env_steps']:9,d} "
                f"r {h['mean_reward']:+.3f} ent {h['entropy']:.3f} term_d {env.terminal_d.tolist()} "
                f"best {env.best_reward:+.3f} t {h['wall_time_s']:.0f}s",
                flush=True,
            )
        return False

    t0 = time.perf_counter()
    res = agent.train(cell.budget_env_steps, max_time_s=minutes * 60.0, stop_on_success=True, callback=cb, verbose=False)
    wall = time.perf_counter() - t0
    models = T.noise_models(graph)
    first = None
    best = None
    if res.first_success is not None:
        fc = res.first_success
        first = {
            "env_steps": fc.env_steps,
            "episodes": fc.episodes,
            "update": fc.update,
            "wall_time_s": next((h["wall_time_s"] for h in agent.history if h["env_steps"] >= fc.env_steps), None),
            "gate_count": len(fc.gates),
        }
        cands = sorted(agent.successes, key=lambda c: len(c.gates))
        fc_best = cands[0]
        tab = replay_gates(fc_best.gates, graph.n, 1)
        best = T.describe_code(tab, 1, fc_best.gates, models)
        best["source"] = f"shortest of {len(agent.successes)} successful episodes of update {fc_best.update + 1}"
    elif env.best_tableau is not None:
        best = T.describe_code(env.best_tableau, 1, env.best_gates, models)
        best["source"] = "best terminal state (no success)"
    stopped_by = "success" if res.success else ("env_steps" if agent.env_steps >= cell.budget_env_steps else "wall_cap")
    return {
        "kind": "run",
        "experiment": "heavy_hex_overnight",
        "topology": "heavy_hex",
        "spec": cell.spec,
        "graph": T.graph_summary(graph),
        "n": graph.n,
        "k": 1,
        "d_target": cell.d_target,
        "gates": list(GATES),
        "max_steps": MAX_STEPS,
        "batch": BATCH,
        "num_actions": env.num_actions,
        "seed": seed,
        "threads": threads,
        "load_1min_at_start": load,
        "ppo_config": dict(agent.config.to_dict(), policy_resolved=agent.policy),
        "num_parameters": agent.num_parameters(),
        "budget_env_steps": cell.budget_env_steps,
        "minutes_cap": minutes,
        "wall_time_s": round(wall, 1),
        "env_steps": agent.env_steps,
        "episodes": agent.episodes,
        "updates": agent.updates,
        "success": res.success,
        "stopped_by": stopped_by,
        "successes": len(agent.successes),
        "first_success": first,
        "best_terminal_reward": None if env.best_reward == -np.inf else float(env.best_reward),
        "terminal_d_counts": env.terminal_d.tolist(),
        "best_code": best,
        "noise_refused": models["_refused"],
        "platform": platform.platform(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - wall)),
        "finished_at": now(),
        "rusage_self": rusage_dict(resource.RUSAGE_SELF),
        "history": agent.history,
    }


def cmd_run_one(args) -> int:
    from .scaling import _json_default

    cell = next(c for c in CELLS if c.spec == args.spec and c.d_target == args.d)
    out_dir = Path(args.out_dir)
    path = run_path(out_dir, cell, args.seed)
    if path.exists():
        print(f"{now()} checkpoint exists, skipping: {path}", flush=True)
        return SKIP_EXIT
    print(f"{now()} start {cell} seed={args.seed} threads={args.threads} pid={os.getpid()}", flush=True)
    rec = run_one(cell, args.seed, args.threads, minutes=args.minutes_cap, verbose=not args.quiet)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, default=_json_default)
    os.replace(tmp, path)  # the checkpoint appears atomically
    fs = rec["first_success"]
    print(
        f"{now()} done success={rec['success']} first={fs['env_steps'] if fs else None} env_steps={rec['env_steps']:,} "
        f"wall={rec['wall_time_s']}s best={rec['best_terminal_reward']} wrote {path}",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------
# driver (parent process)
# --------------------------------------------------------------------------
class Driver:
    def __init__(self, out_dir: Path, hours: float, threads: int, seeds, verbose_child: bool):
        self.out_dir = out_dir
        self.threads = threads
        self.seeds = tuple(seeds)
        self.verbose_child = verbose_child
        self.t_start = time.time()
        self.deadline = self.t_start + hours * 3600.0
        self.jobs = jobs(self.seeds)
        self.done = 0
        self.skipped = 0
        self.failed = 0
        self.current: dict | None = None
        self.last: dict | None = None
        self.child: subprocess.Popen | None = None
        self.stop = threading.Event()
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.log_path = out_dir / "log.txt"
        self.hb_path = out_dir / "heartbeat.json"
        self.res_path = out_dir / "resources.jsonl"
        self.pid_path = out_dir / "driver.pid"

    # -- files
    def log(self, msg: str):
        line = f"{now()} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def heartbeat(self, status: str = "running"):
        rec = {
            "time": now(),
            "epoch": time.time(),
            "status": status,
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t_start)),
            "elapsed_s": round(time.time() - self.t_start, 1),
            "deadline": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.deadline)),
            "remaining_s": round(self.deadline - time.time(), 1),
            "planned": len(self.jobs),
            "done_this_session": self.done,
            "skipped_existing": self.skipped,
            "failed": self.failed,
            "checkpoints_on_disk": sum(run_path(self.out_dir, c, s).exists() for c, s in self.jobs),
            "current": self.current,
            "last_completed": self.last,
            "load_1min": os.getloadavg()[0],
        }
        tmp = self.hb_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, self.hb_path)

    def heartbeat_loop(self):
        while not self.stop.wait(HEARTBEAT_S):
            try:
                self.heartbeat()
            except Exception as exc:  # the heartbeat must not kill the run
                print(f"{now()} heartbeat failed: {exc!r}", flush=True)

    # -- signals
    def on_signal(self, signum, frame):
        self.log(f"signal {signum}; terminating child and exiting")
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(30)
            except subprocess.TimeoutExpired:
                self.child.kill()
        self.finish("killed")
        sys.exit(128 + signum)

    def finish(self, status: str):
        self.stop.set()
        self.heartbeat(status)
        if self.pid_path.exists():
            self.pid_path.unlink()

    # -- runs
    def run_job(self, cell: Cell, seed: int):
        path = run_path(self.out_dir, cell, seed)
        name = cell.name(seed)
        remaining = self.deadline - time.time()
        minutes = min(cell.minutes, remaining / 60.0)
        self.current = {"run": name, "spec": cell.spec, "n": cell.n, "d_target": cell.d_target, "seed": seed,
                        "started": now(), "minutes_cap": round(minutes, 1)}
        self.heartbeat()
        self.log(f"START {name} budget={cell.budget_env_steps:,} cap={minutes:.1f}min threads={self.threads} load={os.getloadavg()[0]:.1f}")
        cmd = [sys.executable, "-m", "autoqec_sim.experiments.heavy_hex_overnight", "run-one",
               "--spec", cell.spec, "--d", str(cell.d_target), "--seed", str(seed),
               "--threads", str(self.threads), "--out-dir", str(self.out_dir), "--minutes-cap", f"{minutes:.3f}"]
        if not self.verbose_child:
            cmd.append("--quiet")
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.time()
        with open(self.out_dir / "logs" / f"{name}.log", "a") as lf:
            self.child = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            # hard stop at the global deadline (the child also caps itself)
            hard = self.deadline + 120.0 - time.time()
            try:
                rc = self.child.wait(timeout=max(hard, 1.0))
            except subprocess.TimeoutExpired:
                self.child.terminate()
                try:
                    rc = self.child.wait(30)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    rc = self.child.wait()
                self.log(f"KILLED {name} at the global deadline")
        wall = time.time() - t0
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.child = None
        rec = None
        if path.exists():
            with open(path) as f:
                rec = json.load(f)
        line = {
            "time": now(),
            "run": name,
            "spec": cell.spec,
            "n": cell.n,
            "d_target": cell.d_target,
            "seed": seed,
            "threads": self.threads,
            "exit_status": rc,
            "wall_s": round(wall, 1),
            "cpu_user_s": round(after.ru_utime - before.ru_utime, 2),
            "cpu_sys_s": round(after.ru_stime - before.ru_stime, 2),
            "children_maxrss_bytes_so_far": int(after.ru_maxrss if sys.platform == "darwin" else after.ru_maxrss * 1024),
            "child_rusage_self": rec["rusage_self"] if rec else None,
            "env_steps": rec["env_steps"] if rec else None,
            "success": rec["success"] if rec else None,
            "load_1min_at_end": os.getloadavg()[0],
        }
        with open(self.res_path, "a") as f:
            f.write(json.dumps(line) + "\n")
        if rec is None:
            self.failed += 1
            self.log(f"FAIL {name} exit={rc} wall={wall:.0f}s (no checkpoint written; see logs/{name}.log)")
            self.last = {"run": name, "status": f"failed exit={rc}"}
            return
        self.done += 1
        fs = rec["first_success"]
        bc = rec["best_code"] or {}
        cpu = line["cpu_user_s"] + line["cpu_sys_s"]
        rss = (rec["rusage_self"]["maxrss_bytes"] / 2**20) if rec.get("rusage_self") else float("nan")
        self.log(
            f"DONE {name} success={rec['success']} stopped_by={rec['stopped_by']} env_steps={rec['env_steps']:,} "
            f"updates={rec['updates']} wall={rec['wall_time_s']}s best_reward={rec['best_terminal_reward']} "
            f"first_success_steps={fs['env_steps'] if fs else None} d={bc.get('d')} w*={bc.get('w_star')} "
            f"gates={bc.get('gate_count')} cpu={cpu:.0f}s maxrss={rss:.0f}MiB terminal_d={rec['terminal_d_counts']}"
        )
        self.last = {"run": name, "success": rec["success"], "env_steps": rec["env_steps"], "wall_s": rec["wall_time_s"],
                     "finished": now()}

    def drive(self) -> int:
        if self.pid_path.exists():
            try:
                pid = int(self.pid_path.read_text().strip())
                os.kill(pid, 0)
                print(f"driver already running with pid {pid} ({self.pid_path}); refusing to start", file=sys.stderr)
                return 2
            except (ValueError, ProcessLookupError, PermissionError):
                pass  # stale pid file
        self.pid_path.write_text(f"{os.getpid()}\n")
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        self.heartbeat()
        pending = [(c, s) for c, s in self.jobs if not run_path(self.out_dir, c, s).exists()]
        self.skipped = len(self.jobs) - len(pending)
        self.log(
            f"driver pid={os.getpid()} planned={len(self.jobs)} finished_already={self.skipped} pending={len(pending)} "
            f"deadline={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.deadline))} threads={self.threads} "
            f"cells={[asdict(c) for c in CELLS]} seeds={list(self.seeds)}"
        )
        for cell, seed in pending:
            if run_path(self.out_dir, cell, seed).exists():
                self.skipped += 1
                self.log(f"SKIP {cell.name(seed)} (checkpoint appeared)")
                continue
            remaining_min = (self.deadline - time.time()) / 60.0
            if remaining_min < MIN_RUN_MINUTES:
                self.log(f"STOP {remaining_min:.1f} min left before the deadline; not starting {cell.name(seed)}")
                break
            self.run_job(cell, seed)
        self.current = None
        left = sum(not run_path(self.out_dir, c, s).exists() for c, s in self.jobs)
        self.log(f"driver exit: done={self.done} failed={self.failed} finished_already={self.skipped} still_pending={left} "
                 f"elapsed={(time.time() - self.t_start) / 3600:.2f}h")
        try:
            write_summary(self.out_dir)
        except Exception as exc:
            self.log(f"summary failed: {exc!r}")
        self.finish("finished" if left == 0 else "deadline")
        return 0


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.1f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def write_summary(out_dir: Path) -> Path:
    recs = []
    for p in sorted(out_dir.glob("run_*.json")):
        with open(p) as f:
            r = json.load(f)
        r["_file"] = p.name
        recs.append(r)
    res = {}
    if (out_dir / "resources.jsonl").exists():
        for line in (out_dir / "resources.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                res[d["run"]] = d  # the last line per run wins
    L = [
        "# Heavy-hex overnight run: summary",
        "",
        f"Written by `python -m autoqec_sim.experiments.heavy_hex_overnight report` at {now()} from",
        f"{len(recs)} run JSONs under `{out_dir}/`. Objective per cell: `[[n, 1, d >= d_target]]` with no idle",
        "qubit, H and CNOT on the heavy-hex edges, 50-gate episodes, `LevelDistanceReward`,",
        "`PPOConfig` defaults (`policy=\"auto\"`, so the equivariant network at these sizes), batch 256.",
        "",
        "## Cells",
        "",
        "| spec | n | d target | seeds planned | finished | successes | env steps to first success (per finished seed) | stopped by |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in CELLS:
        rs = sorted((r for r in recs if r["spec"] == c.spec and r["d_target"] == c.d_target), key=lambda r: r["seed"])
        steps = ", ".join(f"s{r['seed']}: {_fmt(r['first_success']['env_steps']) if r['first_success'] else 'none'}" for r in rs) or "none yet"
        stopped = ", ".join(f"s{r['seed']}: {r['stopped_by']}" for r in rs) or ""
        L.append(f"| `{c.spec}` | {c.n} | {c.d_target} | {len(SEEDS)} | {len(rs)} | {sum(r['success'] for r in rs)} of {len(rs)} | {steps} | {stopped} |")
    L += [
        "",
        "## Runs",
        "",
        "| run | success | stopped by | env steps | updates | wall (s) | first success (env steps, wall s) | best terminal reward | code | w* | max degree | gates | terminal d counts | cpu (s) | peak RSS (MiB) | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        fs = r["first_success"]
        bc = r["best_code"]
        code = f"[[{bc['n']},{bc['k']},{bc['d']}]]" + ("" if r["success"] else " (best terminal state)") if bc else "none"
        ws = (str(bc["w_star"]) + ("" if bc["w_star_exact"] else " (bound)")) if bc else "n/a"
        rr = res.get(r["_file"][:-5], {})
        cpu = (rr["cpu_user_s"] + rr["cpu_sys_s"]) if rr else None
        rss = r["rusage_self"]["maxrss_bytes"] / 2**20 if r.get("rusage_self") else None
        L.append(
            f"| {r['_file'][:-5]} | {'yes' if r['success'] else 'no'} | {r['stopped_by']} | {_fmt(r['env_steps'])} | {r['updates']} | {_fmt(r['wall_time_s'])} "
            f"| {(_fmt(fs['env_steps']) + ', ' + _fmt(fs['wall_time_s'])) if fs else 'none'} | {r['best_terminal_reward']} | {code} | {ws} "
            f"| {bc['max_degree'] if bc else 'n/a'} | {bc['gate_count'] if bc else 'n/a'} | {r['terminal_d_counts']} | {_fmt(cpu)} | {_fmt(rss)} | `{r['_file']}` |"
        )
    L += [
        "",
        "Peak RSS is `ru_maxrss` of the run's own process (`resource.getrusage(RUSAGE_SELF)`); CPU seconds are the",
        "driver's `RUSAGE_CHILDREN` delta across the run (user plus system). `terminal d counts` is the histogram of",
        "the exact distance of every terminated episode, capped at the target distance (index 0 is distance 0 or 1 for",
        "a padded state, the last index is `d >= d_target`).",
        "",
    ]
    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(L))
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unattended heavy-hex PPO run with checkpoints and diagnostics.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="print the cells and seeds and which checkpoints exist")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))

    p = sub.add_parser("run-one", help="one run (the driver's child); exits 3 if the checkpoint exists")
    p.add_argument("--spec", required=True, choices=sorted({c.spec for c in CELLS}))
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--minutes-cap", type=float, default=None, help="wall cap override, never above the cell's own")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("drive", help="run every pending cell sequentially until done or the deadline")
    p.add_argument("--hours", type=float, default=10.0, help="global wall cap from now")
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--quiet-children", action="store_true")

    p = sub.add_parser("report", help="write SUMMARY.md from the run JSONs")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))

    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.cmd == "plan":
        for c, s in jobs(args.seeds):
            print(f"{c.name(s):45s} budget={c.budget_env_steps:>9,} cap={c.minutes:5.1f}min "
                  f"{'done' if run_path(out_dir, c, s).exists() else 'pending'}")
        return 0
    if args.cmd == "run-one":
        return cmd_run_one(args)
    if args.cmd == "drive":
        return Driver(out_dir, args.hours, args.threads, args.seeds, verbose_child=not args.quiet_children).drive()
    if args.cmd == "report":
        print("wrote", write_summary(out_dir))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
