# Copyright 2026 Vincent Russo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Schedule synthesis for one board code: coloring baseline, then PPO.

    autoqec-schedule 49-13-4 --board /path/to/qldpc-challenge --updates 40

Writes ``<out>/`` (default ``schedule_out/<slug>/``) with ``baseline.json``
(the board builder's own schedule family, scored), ``train.jsonl`` (one line
per PPO update), ``best.json`` (the best schedule found, its layers, and a
deep-search confirmation of its fault distance), and ``log.txt``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import schedule_env as se
from .agents.policies import mlp_actor_critic


def train_ppo(env: se.ScheduleEnv, updates: int, log, out_path: Path, *, lr=3e-4, epochs=4,
              minibatch=256, clip=0.2, ent_coef=0.01, vf_coef=0.5, seed=0, hidden=(256, 256)):
    """Fixed-horizon PPO over whole episodes. Every update collects
    ``batch_size`` complete schedules, scores them at the terminal step, and
    fits the policy on the ``batch_size * E`` transitions with a terminal
    reward and ``gamma = 1``."""
    import torch
    torch.manual_seed(seed)
    net = mlp_actor_critic(env.obs_dim, env.num_actions, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    B, E, T = env.batch_size, env.E, env.num_actions
    best = {"reward": -np.inf}
    history = []
    for u in range(updates):
        t0 = time.time()
        obs_buf = np.zeros((E, B, env.obs_dim), dtype=np.float32)
        mask_buf = np.zeros((E, B, T), dtype=bool)
        act_buf = np.zeros((E, B), dtype=np.int64)
        logp_buf = np.zeros((E, B), dtype=np.float32)
        val_buf = np.zeros((E, B), dtype=np.float32)
        obs, _ = env.reset()
        with torch.no_grad():
            for t in range(E):
                mask = env.action_mask()
                o = torch.as_tensor(obs)
                logits, v = net(o, torch.as_tensor(mask))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                obs_buf[t], mask_buf[t], act_buf[t] = obs, mask, a.numpy()
                logp_buf[t], val_buf[t] = dist.log_prob(a).numpy(), v.numpy()
                obs, reward, _done, info = env.step(a.numpy())
        scores = info["scores"]
        R = np.asarray(reward, dtype=np.float32)  # terminal reward per episode
        ret = np.broadcast_to(R, (E, B)).copy()
        adv = ret - val_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # update
        def flat(x):
            return torch.as_tensor(np.asarray(x).reshape(E * B, *np.asarray(x).shape[2:]))

        OBS, M, A, LP, RET, ADV = map(flat, (obs_buf, mask_buf, act_buf, logp_buf, ret, adv))
        N = E * B
        stats = {}
        for _ in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, minibatch):
                idx = perm[i:i + minibatch]
                logits, v = net(OBS[idx], M[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[idx])
                ratio = torch.exp(lp - LP[idx])
                pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx]).mean()
                vl = ((v - RET[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + vf_coef * vl - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats = {"pg": pg.item(), "vl": vl.item(), "ent": ent.item()}
        mins = [s["min"] for s in scores if s.get("min") is not None]
        counts = [s["count_min"] for s in scores if s.get("count_min") is not None]
        row = {"update": u, "reward_mean": float(R.mean()), "reward_max": float(R.max()),
               "dcirc_mean": float(np.mean(mins)) if mins else None,
               "dcirc_max": int(max(mins)) if mins else None,
               "sets_mean": float(np.mean(counts)) if counts else None,
               "sets_min": int(min(counts)) if counts else None,
               "reached_d": int(sum(1 for s in scores if s.get("min") is not None and s["min"] >= env.code.d)),
               "invalid": int(sum(1 for s in scores if not s.get("valid", False))),
               "layers_mean": float(np.mean([s["layers"] for s in scores if s.get("layers")])) if mins else None,
               "seconds": time.time() - t0, **stats}
        history.append(row)
        with open(out_path / "train.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"update {u}: reward mean {row['reward_mean']:.3f} max {row['reward_max']:.3f} "
            f"d_circ mean {row['dcirc_mean']} max {row['dcirc_max']} min-sets mean {row['sets_mean']} min {row['sets_min']} "
            f"reached d {row['reached_d']}/{B} invalid {row['invalid']} layers {row['layers_mean']} ({row['seconds']:.0f}s)")
        for s in scores:
            if "d_circ" not in s or "layers_list" not in s:
                continue  # invalid schedule: no circuit was scored
            if s["reward"] > best["reward"]:
                best = {"reward": float(s["reward"]), "d_circ": s.get("d_circ"), "count_min": s.get("count_min"),
                        "layers": s["layers"], "layers_list": s["layers_list"], "update": u}
                with open(out_path / "best.json", "w") as f:
                    json.dump(best, f, indent=1)
    return best, history


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--baseline", type=int, default=20)
    ap.add_argument("--tier", default="exact", choices=["exact", "ris"])
    ap.add_argument("--ris-trials", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=None, help="extraction rounds scored (default: screen at 2, confirm at d)")
    ap.add_argument("--interleave", action="store_true")
    ap.add_argument("--slack", type=int, default=1)
    ap.add_argument("--depth-penalty", type=float, default=0.02)
    ap.add_argument("--edge-order", default="fixed", choices=["fixed", "shuffle"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirm-trials", type=int, default=200000)
    ap.add_argument("--out", default=None, help="output directory (default schedule_out/<slug>)")
    ap.add_argument("--board", default=None, help="qldpc-challenge checkout (default: QLDPC_CHALLENGE or a sibling directory)")
    args = ap.parse_args(argv)

    if args.board:
        os.environ["QLDPC_CHALLENGE"] = str(Path(args.board).resolve())
    out = Path(args.out) if args.out else Path.cwd() / "schedule_out" / args.slug
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")  # noqa: SIM115

    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    code = se.load_board_code(args.slug)
    rounds = args.rounds if args.rounds is not None else 2
    log(f"{code.slug}: n={code.n} k={code.k} d={code.d}; board circuit d_circ={code.d_circ_board} ({code.schedule_note[:80]})")
    log(f"screening at rounds={rounds}, tier={args.tier}")

    # baseline: the board builder's own family
    base = se.coloring_baseline(code, samples=args.baseline, seed=args.seed, rounds=rounds, tier=args.tier,
                                ris_trials=args.ris_trials)
    bmins = [b["min"] for b in base]
    bcounts = []
    for b in base:
        if args.tier == "exact" and b["min"] is not None:
            bcounts.append(max(v.get("count", 0) for bb, v in b["detail"].items()
                               if isinstance(v, dict) and b["d_circ"].get(bb) == b["min"]))
    log(f"baseline ({args.baseline} greedy colorings): d_circ values {sorted(bmins)}; min-weight sets {sorted(bcounts)}; "
        f"layers {sorted({b['layers'] for b in base})}; {np.mean([b['seconds'] for b in base]):.1f}s per schedule")
    with open(out / "baseline.json", "w") as f:
        json.dump([{k: v for k, v in b.items() if k != "layers_list"} for b in base], f, indent=1)

    env = se.ScheduleEnv(code, batch_size=args.batch, slack=args.slack, interleave=args.interleave, tier=args.tier,
                         ris_trials=args.ris_trials, rounds=rounds, depth_penalty=args.depth_penalty,
                         edge_order=args.edge_order, seed=args.seed)
    log(f"env: {env.E} edges, {env.T} slots (lower bound {env.lower_bound_layers} layers), obs_dim {env.obs_dim}")
    best, _hist = train_ppo(env, args.updates, log, out, seed=args.seed)
    log(f"best: reward {best['reward']:.3f} d_circ {best.get('d_circ')} layers {best.get('layers')} at update {best.get('update')}")

    # confirmation at rounds = d with the board's deep search and the exact tier
    if best.get("layers_list"):
        conf = {}
        conf["exact_rounds_d"] = se.score_layers(code, best["layers_list"], rounds=code.d, tier="exact", exact_wmax=4)
        conf["ris_rounds_d"] = se.score_layers(code, best["layers_list"], rounds=code.d, tier="ris",
                                               ris_trials=args.confirm_trials, ris_seed=args.seed + 1)
        best["confirmation"] = {k: {kk: vv for kk, vv in v.items() if kk != "layers_list"} for k, v in conf.items()}
        with open(out / "best.json", "w") as f:
            json.dump(best, f, indent=1)
        log(f"confirmation at rounds={code.d}: exact {conf['exact_rounds_d']['d_circ']} ; "
            f"ris {args.confirm_trials} trials {conf['ris_rounds_d']['d_circ']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
