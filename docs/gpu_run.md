# Running the GPU-scale matrix on the RunPod pod

`autoqec_sim.experiments.gpu_scale` trains PPO with the equivariant policy
on three targets: the noise-aware 4x4 cells of `experiments.noise_aware`
(three profiles x three seeds, 1024 episodes, 20M env steps each), the
`[[25, 1, 5]]` rediscovery on the 5x5 lattice (three seeds, 2048 episodes,
30M env steps each, stop at the first success), and the same rediscovery
under the hard LDPC constraint `w* <= 4` (`surface5x5_sparse`: three
seeds, 2048 episodes, 40M env steps, 150 min cap; `--ldpc-mode mask` by
default, `final` as the alternative; the exact feasibility test runs on
the GPU every step). The module docstring says
what runs on the GPU and what stays on the CPU. This page is the command
sequence for the pod that already has the repository at
`/root/work/genesis/autoqec-sim` and a venv at `/root/work/genesis/venv`
(created with `--system-site-packages` and `pip install -e .`, so the pod's `torch 2.8.0+cu128` is the torch in use; do not run
`uv sync --extra rl` or `pip install -e .[rl]` there, both would replace
it with a CPU torch, see `docs/gpu.md`).

## 1. Sync the code (laptop)

The editable install on the pod reads the source tree, so syncing the
files is the whole deployment. `POD_HOST` is the pod's SSH host (and
`-p PORT` if RunPod gives a nonstandard port; add `-e "ssh -p PORT"` to
every rsync below).

```sh
rsync -av --exclude .venv --exclude results --exclude '__pycache__' --exclude .git \
    ~/Projects/genesis/autoqec-sim/ root@POD_HOST:/root/work/genesis/autoqec-sim/
# the n = 16 known-code baselines the report reads (not recomputed on the pod)
rsync -av ~/Projects/genesis/autoqec-sim/results/noise_aware/baselines_4x4.json \
    root@POD_HOST:/root/work/genesis/autoqec-sim/results/noise_aware/
```

## 2. Check the environment (pod)

```sh
ssh root@POD_HOST
cd /root/work/genesis/autoqec-sim
source /root/work/genesis/venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
python -m pytest -q tests/test_torch_backend.py tests/test_gpu_scale.py
```

The tests run on the CPU. The one CUDA-specific check worth doing before
a long run is that the distance kernel on the GPU agrees with numpy:

```sh
python - <<'PY'
import numpy as np
from autoqec_sim import graphs
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.experiments.gpu_scale import DistanceKernel
from autoqec_sim.reward import batched_distance, low_weight_stabilizer_penalty
for n, spec, d in ((16, "square:4x4", 4), (25, "square:5x5", 5)):
    env = CodeDiscoveryEnv(n=n, k=1, graph=graphs.from_spec(spec), batch_size=256, gates=("H", "CNOT"), max_steps=80, seed=1)
    env.reset()
    for _ in range(60):
        env.step(env.sample_actions())
    dd, _, pad = DistanceKernel(n, 1, d, "cuda")(env.tab)
    assert np.array_equal(dd, batched_distance(env.tab, 1, d - 1, chunk=2048))
    assert np.allclose(pad, low_weight_stabilizer_penalty(env.tab, 1, 1))
    print(n, "cuda kernel agrees with numpy; terminal d histogram", np.bincount(dd, minlength=d + 1).tolist())
PY
```

## 3. Size the budget with a short pilot (pod, about 5 minutes)

Throughput on the pod decides whether the default budgets fit the night.
On the laptop CPU (Apple M5 Pro, 6 threads, 200k-step smoke, 2026-09-21)
the module runs at 3.5k env steps/s on noise4x4 at batch 1024 and 2.3k
env steps/s on surface5x5 at batch 2048, with 86 to 89 percent of the wall
time in the PPO update of the equivariant network and 2 percent in the
reward (early in training almost every state is decided at weight 1 or 2,
so the expensive top weight is rarely evaluated; the reward share grows as
the policy reaches d* - 1 states). On the GPU the update should stop being
the bottleneck and the numpy environment step plus the observation
encoding take over. Measure it:

```sh
mkdir -p results/gpu_scale/pilot
python -m autoqec_sim.experiments.gpu_scale run --target noise4x4 --profile board --seed 0 \
    --device cuda --steps 500000 --threads 8 --out-dir results/gpu_scale/pilot --verbose
python -m autoqec_sim.experiments.gpu_scale run --target surface5x5 --seed 0 \
    --device cuda --steps 500000 --threads 8 --out-dir results/gpu_scale/pilot --verbose
python -c "
import json, glob
for p in sorted(glob.glob('results/gpu_scale/pilot/run_*.json')):
    r = json.load(open(p)); print(p, r['env_steps_per_s'], 'steps/s; reward', r['reward_time_s'], 's, update', r['update_time_s'], 's, other', r['other_time_s'], 's')
"
```

Matrix cost at the default budgets: 9 x 20M + 3 x 30M + 3 x 40M = 390M
env steps. At 20k steps/s that is 5.4 h; at 10k steps/s 11 h. The wall
caps (60 min per noise4x4 cell, 120 min per surface5x5 cell, 150 min per
surface5x5_sparse cell, 22.5 h in total)
bound the worst case; a cell that hits its cap is marked `hit_wall_cap`
in its JSON and in the report. To fit a fixed window, scale `--steps-*`
or `--wall-*` from the pilot's steps/s; the surface5x5 cells stop at
their first success and are usually shorter than their budget.

## 4. Launch the matrix (pod)

```sh
cd /root/work/genesis/autoqec-sim && source /root/work/genesis/venv/bin/activate
mkdir -p results/gpu_scale/logs
nohup python -m autoqec_sim.experiments.gpu_scale matrix --device cuda --threads 8 \
    > results/gpu_scale/logs/matrix.out 2>&1 &
echo $! > results/gpu_scale/logs/matrix.pid
```

Order: the nine noise4x4 cells (board, ion, heron x seeds 0, 1, 2), then
the three surface5x5 seeds, then the three surface5x5_sparse seeds. To
run only the constrained target (the first surface5x5 seed on the pod
succeeded at 1.79M steps with a dense code, so the free target is the
cheap one):

```sh
nohup python -m autoqec_sim.experiments.gpu_scale matrix --device cuda --threads 8 --targets surface5x5_sparse \
    > results/gpu_scale/logs/matrix_sparse.out 2>&1 &
# the other mode, into the same result set, needs its own out-dir (same file names)
nohup python -m autoqec_sim.experiments.gpu_scale matrix --device cuda --threads 8 --targets surface5x5_sparse \
    --ldpc-mode final --out-dir results/gpu_scale/final_mode > results/gpu_scale/logs/matrix_sparse_final.out 2>&1 &
``` Every finished cell writes
`results/gpu_scale/run_<target>[_<profile>]_seed<k>.json` and rewrites
`results/gpu_scale/REPORT.md`, so the report is current at any time. With
a tighter window, or to run the two targets side by side on one GPU (each
process is one CPU-bound environment plus GPU bursts):

```sh
nohup python -m autoqec_sim.experiments.gpu_scale matrix --device cuda --threads 4 --targets noise4x4 \
    --steps-noise4x4 10000000 --wall-noise4x4 40 > results/gpu_scale/logs/matrix_noise4x4.out 2>&1 &
nohup python -m autoqec_sim.experiments.gpu_scale matrix --device cuda --threads 4 --targets surface5x5 \
    --wall-surface5x5 90 > results/gpu_scale/logs/matrix_surface5x5.out 2>&1 &
```

`--skip-existing` resumes a matrix after an interruption without redoing
finished cells. `--batch-<target>`, `--steps-<target>`, `--wall-<target>`
(with `surface5x5-sparse` spelled with a hyphen), `--seeds`,
`--profiles`, and `--chunk` (Paulis per streamed chunk of the GPU distance
kernel, default 8192 on cuda; the GPU intermediate is `4 x undecided
elements x chunk x (n + k)` bytes, 1.7 GiB at the worst case of the
surface5x5 cell) are the other knobs.

## 5. Check progress (pod)

```sh
tail -n 5 results/gpu_scale/logs/matrix.out          # one line per finished cell
tail -n 2 results/gpu_scale/logs/run_*.log            # last update of every cell (every 25 updates and on every success)
ls -la results/gpu_scale/*.json
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
kill "$(cat results/gpu_scale/logs/matrix.pid)"       # stop; the finished cells stay, resume with --skip-existing
```

For surface5x5_sparse the pilot also shows the cost of the exact LDPC
check (`ldpc_check_time_s` in the JSON, `viol` in the log lines: the
fraction of actions reverted by the constraint in `mask` mode). On the
laptop CPU the numpy check costs 0.6 s per 2048-element step and the
torch-on-CPU path 0.28 s; the run uses the torch path on the GPU.

A log line reads `upd 125 steps 2,048,000 r -0.7412 succ 0.000 ent 2.1
distinct 0 best nan t 95s (21,558 steps/s)` for noise4x4 (`distinct` is
the number of distinct target-distance codes so far, `best` the best
terminal reward among them) and `upd ... term_d [0, 512, 1400, 130, 6, 0]
best -0.200 ...` for surface5x5 (`term_d` is the histogram of terminal
distances 0..5 since the previous update).

## 6. Bring the results back (laptop) and regenerate the report

```sh
rsync -av root@POD_HOST:/root/work/genesis/autoqec-sim/results/gpu_scale/ \
    ~/Projects/genesis/autoqec-sim/results/gpu_scale/
cd ~/Projects/genesis/autoqec-sim
uv run --extra rl python -m autoqec_sim.experiments.gpu_scale report
```

`report` needs no torch and no GPU; it reads the JSON files and
`results/noise_aware/baselines_4x4.json`. The pilot directory is not
read (only `results/gpu_scale/run_*.json` is), so pilot runs do not enter
the report unless they are moved up one level.
