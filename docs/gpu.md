# The torch backend on a GPU

`autoqec_sim.backends.torch_backend` runs the two hot paths of the reward
engine (the tableau gate kernels and the parity masks behind every reward)
on torch tensors on `cpu` or `cuda`. This page says what it covers, how to
measure it on a RunPod GPU with `scripts/profile_tableau.py`, and what is
still specific to cuQuantum. The GPU table was measured on a RunPod A40
on 2026-09-21.

## What the backend covers

Same array layout as numpy: `(batch, 2n, 2n + 1)` `uint8`, so
`torch_backend.from_numpy(tab, device)` and `torch_backend.to_numpy(t)`
move a state between the backends without conversion. Functions have the
names and semantics of their numpy counterparts in `tableau`, `symplectic`,
and `reward`:

- gate kernels `apply_h`, `apply_s`, `apply_s_dag`, `apply_x`, `apply_y`,
  `apply_z`, `apply_cnot`, `apply_cz`, `apply_swap`, `apply_gate`, and
  `ActionTable(table_or_entries, device)` with the three lookup arrays on
  the device and `apply(tab, actions)` grouped by gate type;
- `symplectic_product`, `swap_halves`, `code_rows`;
- `undetected_mask`, `in_stabilizer_mask`, `batched_distance`, `kl_reward`,
  `distance_reward`, `low_weight_stabilizer_penalty`, `CompositeReward`,
  `make_kl_reward(errors, device=...)`, `make_low_weight_penalty`,
  `make_distance_reward`;
- `numpy_reward_adapter(fn, device)`: a torch reward as a `reward_fn` for
  `CodeDiscoveryEnv` (the environment itself stays numpy; the tableau is
  copied to the device once per step and the `(batch,)` result back).

Parity test. The numpy path computes the float32 product of the error set
against the `n + k` code rows, casts it to int64, masks the low bit, and
runs two boolean `any()` reductions; on the CPU the cast and the reductions
are about 80 percent of the reward (`docs/profile.md`). The torch path
reduces the product mod 2 in place in float32 (`remainder_(2)`, exact since
the entries are integers below `2n`) and decides both tests with one
matrix-vector product against a weight vector: stabilizer columns weigh
`2k + 1`, logical columns `1`, so the score of an error is `s (2k + 1) + l`
with `s` anticommuting stabilizers and `l` anticommuting logicals, and the
error is undetected iff `1 <= score <= 2k`. The peak intermediate is one
float32 `(batch, chunk, n + k)` block, `4 M (n + k)` bytes per element
instead of `12 M (n + k)`, and no boolean `(batch, chunk, n + k)` array
exists. `tests/test_torch_backend.py` checks every kernel and every mask
against numpy bit for bit at `n` in 5, 9, 16 on random tableaux and on the
known codes.

## CPU torch against numpy (measured)

Same machine and setting as `docs/profile.md` (Apple M5 Pro, six BLAS
threads, torch 2.14.0 CPU, all-to-all graph, gates H, S, CNOT, `k = 1`,
Knill-Laflamme reward over all Paulis of weight 1 and 2 plus the weight-1
penalty, batch 1024, 2026-09-21):

| n | backend | reward intermediate B/elem | gate applications/s | reward evaluations/s | env steps/s |
|---|---|---|---|---|---|
| 9 | numpy | 41.1 KiB | 6.51 M | 83.8 k | 94.2 k |
| 9 | torch cpu | 13.7 KiB | 3.08 M | 155.8 k | 147.3 k |
| 16 | numpy | 224.7 KiB | 4.84 M | 23.3 k | 28.0 k |
| 16 | torch cpu | 74.9 KiB | 2.82 M | 54.2 k | 53.0 k |
| 25 | numpy | 845.5 KiB | 3.57 M | 10.7 k | 11.4 k |
| 25 | torch cpu | 281.8 KiB | 2.04 M | 21.4 k | 21.2 k |

The reward is 1.9 (`n = 9`) to 2.3 (`n = 16`) times faster on CPU torch;
the gate kernels are about half the numpy rate (torch advanced-indexing
overhead on small tensors) and are one to two percent of an environment
step, so the env-steps column follows the reward. Breakdown at `n = 16`,
batch 1024, torch cpu: GEMM 5.95 ms, `remainder_(2)` 5.15 ms, score GEMV
and compares 3.02 ms, against 5.81, 10.00, and 19.94 ms for the numpy
GEMM, int64 cast, and two reductions. The `n = 25` numpy row is from
`docs/profile.md`.

## Running the profile on a RunPod pod

The pod image has torch 2.8.0+cu128 with CUDA available. One point about
the install: the package's `rl` extra pins `torch>=2.14.0` and, on Linux, resolves torch
from the CPU-only PyTorch index, so `uv sync --extra rl` or
`pip install -e .[rl]` would replace the pod's CUDA torch with a CPU one.
Install without the extra into the pod's Python and keep its torch:

```sh
# on the laptop: the repository, without build and result directories
rsync -a --exclude .venv --exclude results --exclude '__pycache__' \
    ~/Projects/genesis/autoqec-sim root@POD_HOST:/workspace/genesis/
# (or git clone into /workspace/genesis on the pod)

# on the pod
cd /workspace/genesis/autoqec-sim
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install -e . pytest
python -c "import autoqec_sim.backends.torch_backend"   # imports the pod's torch

# agreement tests on the CPU, then on the GPU
python -m pytest -q tests/test_torch_backend.py
AUTOQEC_DEVICE=cuda python - <<'PY'
import numpy as np, torch
from autoqec_sim import graphs, reward as R, tableau as T
from autoqec_sim.backends import torch_backend as tb
from autoqec_sim.env import build_action_table
from autoqec_sim.errors import ErrorSet
rng = np.random.default_rng(0)
for n in (5, 9, 16):
    table = build_action_table(graphs.all_to_all(n), T.GATES)
    tt = tb.ActionTable(table, device="cuda")
    tab = T.identity_tableau(n, 256); tabt = tb.identity_tableau(n, 256, device="cuda")
    for _ in range(30):
        a = rng.integers(-1, len(table), size=256)
        table.apply(tab, a); tt.apply(tabt, torch.as_tensor(a, device="cuda"))
    assert np.array_equal(tab, tb.to_numpy(tabt))
    e = ErrorSet.up_to_weight(n, 2)
    assert np.array_equal(R.undetected_mask(tab, 1, e.paulis), tb.to_numpy(tb.undetected_mask(tabt, 1, e.paulis)))
    assert np.array_equal(R.in_stabilizer_mask(tab, 1, e.paulis), tb.to_numpy(tb.in_stabilizer_mask(tabt, 1, e.paulis)))
    print(n, "cuda agrees with numpy")
PY

# the profile: numpy on the pod's CPU, torch on its CPU, torch on the GPU
python scripts/profile_tableau.py --backend numpy --json pod_numpy.json
python scripts/profile_tableau.py --backend torch --device cpu --json pod_torch_cpu.json
python scripts/profile_tableau.py --backend torch --device cuda --mem-cap-gb 16 --json pod_torch_cuda.json
```

`--mem-cap-gb` should stay below the GPU's memory: the reward intermediate
is `4 M (n + k)` bytes per batch element (2.0 GiB at `n = 49`, batch 1024;
8.1 GiB at batch 4096), and `undetected_mask(..., chunk=...)` bounds it by
chunking the error set if a cell has to run under a tighter cap. Timing
synchronizes the device after every call (`torch.cuda.synchronize`), so
the rates include kernel launch and, in the env-steps column, the
host-to-device copy of the tableau per step. Bring the JSON files back
and render the tables:

```sh
rsync root@POD_HOST:/workspace/genesis/autoqec-sim/pod_*.json docs/pod/
uv run python scripts/profile_tableau.py --render docs/pod/pod_numpy.json docs/pod/pod_torch_cpu.json docs/pod/pod_torch_cuda.json
```

The GPU tableau at batch 4096 and `n = 49` is 38 MiB; the gate kernels are
far from memory-bound and their rate on the GPU will be set by launch
overhead and the per-gate-type `nonzero` synchronizations in
`ActionTable.apply` (at most three with the H, S, CNOT gate set).

## GPU result (RunPod A40, 2026-09-21)

Measured on a dedicated RunPod pod: NVIDIA A40 (44 GiB, CUDA 12.8), torch
2.8.0+cu128, Python 3.12.3, numpy 2.1.2 (scipy-openblas), GPU idle before
the run (0 percent, 0 MiB). The host's CPUs were shared with other tenants
(load average 15 on 96 visible cores), so the pod numpy column is slowed
and the laptop numpy column (docs/profile.md, Apple M5 Pro, 6 threads,
unloaded) is the fair CPU reference. Raw records:
`results/profile_pod/torch_cuda_clean.json` and
`results/profile_pod/numpy_pod3.json`; an earlier run on a pod whose GPU
was already at 100 percent from another job
(`torch_cuda.json`, `numpy_pod.json`) gave 4 to 5 times lower GPU rates and
is kept as the contended reference. Render any of them with
`python scripts/profile_tableau.py --render results/profile_pod/<file>.json`.

Reward evaluations per second (the term that is 99 percent of an env step
on the CPU):

| n | batch | torch cuda (A40) | numpy, pod CPU (shared host) | numpy, laptop | GPU / laptop |
|---|---|---|---|---|---|
| 5 | 4096 | 5.77 M | 102.5 k | 197.7 k | 29x |
| 9 | 4096 | 2.40 M | 23.8 k | 79.9 k | 30x |
| 16 | 1024 | 674.0 k | 6.0 k | 24.4 k | 28x |
| 16 | 4096 | 798.4 k | 5.5 k | 24.4 k | 33x |
| 25 | 1024 | 228.2 k | 2.1 k | 10.7 k | 21x |
| 25 | 4096 | 240.4 k | 2.0 k | 10.6 k | 23x |
| 49 | 256 | 36.4 k | not run | 1.7 k | 21x |
| 49 | 1024 | 37.8 k | not run (6.1 GiB) | not run (6.1 GiB) | |

Gate applications per second on the GPU: 244 k to 282 k at batch 256,
1.1 M at 1024, 3.5 M to 4.5 M at 4096, independent of `n` (launch-bound:
one `nonzero` per gate type per call). The numpy kernels run at 1 M to
8 M on the laptop, so gates are at parity at batch 4096 and slower at
small batches; on the CPU they are 1 to 2 percent of a step, on the GPU
10 to 40 percent of the hybrid `env steps/s` column (512.5 k at n = 16,
batch 4096, against 798.4 k reward evaluations). A fused per-element gate
kernel is the first cuQuantum-specific item below.

The `(B, M, n+k)` intermediate at `n = 49`, batch 1024 is 2.0 GiB on the
torch path (float32, in place) against 6.1 GiB on the numpy path, which is
why the GPU runs that cell and the CPU paths do not; at batch 4096
(8.2 GiB) the profile's 4 GiB cap stops both. The GPU reward rate keeps
rising with the batch at every `n` up to 16 (n = 16: 358.7 k, 674.0 k,
798.4 k at 256, 1024, 4096) and is flat from 1024 at n = 25 and 49, where
the GEMM itself saturates the device.

The whole torch run took 20 s and the reduced numpy run 60 s.

## Done here versus cuQuantum-specific

Done in the torch backend (items 1 to 3 of the README's "What is left for
cuQuantum" list, on any torch device):

- the gate kernels and `ActionTable` as gather/scatter operations on device
  tensors;
- `code_rows` as a device-side slice and concatenation;
- the symplectic product and the parity masks with the mod-2 reduction in
  place and both commutation tests as one GEMV, a third of the numpy
  intermediate and none of its int64 or boolean copies;
- the profile with `--backend`, `--device`, `--json`, `--render`.

Still specific to cuQuantum (cuStabilizer) or to a custom CUDA kernel:

- the bit-packed popcount product with parity, the stabilizer test, and
  the logical test fused in one pass, which removes the GEMM and the
  `(batch, chunk, n + k)` intermediate altogether; torch has no batched
  popcount matmul, so the torch path keeps a float32 block per chunk;
- a fused per-element gate kernel: `ActionTable.apply` groups actions by
  gate type with one `nonzero` per type, each a host synchronization on
  `cuda`; a kernel that reads the gate id per batch element needs no
  grouping;
- the environment on the device: `CodeDiscoveryEnv` is numpy (history,
  masks, auto-reset), so the torch reward is reached through
  `numpy_reward_adapter` and the tableau crosses the PCIe bus once per
  step (1 KiB per element at `n = 16`, 9.5 KiB at `n = 49`);
- integration with the PPO agent's tensors (the observation encoder reads
  the numpy tableau; a device-resident environment would hand the policy
  the tableau bits without a copy).

Not on the hot path and left on the host as before: GF(2) elimination,
exact distance and enumerators in `code`, stim conversion, the
circuit-level fault propagation, and the search agent's population update.

## What is left for cuQuantum

The state is one array `tab` of shape `(batch, 2n, 2n+1)` with dtype
`uint8`. Everything a GPU backend has to provide:

1. Gate kernels (`tableau.apply_h`, `apply_s`, `apply_s_dag`, `apply_cnot`,
   `apply_cz`, `apply_swap`, `apply_x/y/z`): each reads two or four columns
   of the selected batch elements (`tab[b, :, q]`, a gather along axis 2
   with a per-element column index) and writes them back after elementwise
   XOR/AND. `ActionTable.apply` groups a vector of actions by gate type and
   calls these once per type; with a fused kernel the grouping disappears.
2. The symplectic product `symplectic.symplectic_product(E, rows)`: a
   batched GEMM of the shared error set `(M, 2n)` against the per-element
   code rows `(batch, n+k, 2n)` followed by a parity reduction. This is the
   whole cost of `reward.undetected_mask`, `in_stabilizer_mask`,
   `batched_distance`, and hence of every reward. Bit-packed rows and a
   popcount-based product would replace float32.
3. `reward.code_rows`: a slice-and-concatenate of the tableau; on device
   it is a view.

Not on the hot path and fine to leave on the host: GF(2) elimination in
`symplectic`, exact distance and enumerators in `code`, stim conversion,
the circuit-level fault propagation (per circuit, uses stim), and the
search agent's population update.

Measured on the CPU (`scripts/profile_tableau.py`, `docs/profile.md`; numpy
on one laptop, not a GPU): the reward is 99% of an environment step at
`n = 16`, batch 1024, and the gate kernels 1%, at every `n` from 5 to 49.
Inside the reward the float32 GEMM of item 2 is about 15% of the
Knill-Laflamme term; the int64 cast for the mod-2 reduction is about 30%
and the two boolean `any()` reductions over the `(batch, M, n + k)` product
more than half.
So item 2 is the hot path, but a port that replaces the GEMM alone and
keeps the cast and the reductions on the host gains little; the kernel to
write is the fused bit-packed popcount product with the parity, the
stabilizer-commutation test, and the logical-anticommutation test in one
pass, which also removes the `(batch, M, n + k)` intermediate (0.8 MiB per
batch element at `n = 25`, 6 MiB at `n = 49`, against 2.5 and 9.5 KiB for
the tableau). Throughput on the CPU at batch 1024: 7.9 M gate applications
per second and 204 k reward evaluations per second at `n = 5`, 4.8 M and
23 k at `n = 16`, 3.6 M and 11 k at `n = 25`.

The torch backend (`autoqec_sim.backends.torch_backend`, `docs/gpu.md`)
is the interim version of items 1 to 3 on any torch device: the gate
kernels and `ActionTable` as the same gather/scatter operations on torch
tensors, and the parity test as a float32 GEMM reduced mod 2 in place plus
one matrix-vector product that decides the stabilizer-commutation and
logical-anticommutation tests together (stabilizer columns weigh `2k + 1`,
logical columns `1`; undetected iff `1 <= score <= 2k`). It removes the
int64 copy and the two boolean reductions, and the intermediate is one
float32 `(batch, chunk, n + k)` block, `4 M (n + k)` bytes per element
instead of `12 M (n + k)`. On CPU torch it is 1.9 to 2.3 times faster than
numpy on the reward at `n = 9` and `16` (batch 1024; the gate kernels run
at half the numpy rate and are one to two percent of a step).
`scripts/profile_tableau.py --backend torch --device cuda --json` produces
the same table on a GPU; `docs/gpu.md` has the RunPod procedure and the
A40 measurement of 2026-09-21 (798 k reward evaluations per second at
`n = 16`, batch 4096, 33 times the laptop numpy rate; 240 k at `n = 25`,
23 times).
What remains cuQuantum-specific after
this is the bit-packed popcount kernel with parity and both tests fused,
which removes the GEMM and the intermediate entirely, and a fused
per-element gate kernel that avoids the per-gate-type `nonzero`
synchronizations.

## Fused CUDA kernels

`autoqec_sim.backends.cuda_kernels` holds the two kernels the list above
names, written as plain CUDA and compiled at first use through CuPy's NVRTC
path (no nvcc, no toolkit beyond the driver):

- `apply_actions`: one launch applies a per-element gate (any of the nine
  gates, per-element qubit indices) to every row of every selected tableau.
- `parity_codes`: bit-packed popcount symplectic products of the error set
  against the code rows, reduced in the kernel to detected, undetected, or
  in-stabilizer per (element, error).

They are opt-in: `AUTOQEC_CUDA_KERNELS=1` or `cuda_kernels.enable()`. The
torch backend's agreement tests pass with them on, and
`tests/test_cuda_kernels.py` compares them with the torch path directly.
Measured on one NVIDIA A100-SXM4-80GB (torch 2.8.0+cu129, `scripts/bench_cuda_kernels.py`,
`docs/data/bench_cuda_kernels_a100.json`):

| n | batch | errors | gate step, torch path | gate step, kernel | undetected mask, torch path | undetected mask, kernel |
|---|---|---|---|---|---|---|
| 5 | 1024 | 105 | 2415 us | 491 us (4.9x) | 0.24 ms | 0.39 ms (0.6x) |
| 9 | 1024 | 351 | 2454 us | 483 us (5.1x) | 0.25 ms | 0.38 ms (0.7x) |
| 16 | 1024 | 1,128 | 2435 us | 486 us (5.0x) | 0.90 ms | 0.38 ms (2.4x) |
| 16 | 4096 | 1,128 | 2437 us | 485 us (5.0x) | 3.49 ms | 0.41 ms (8.5x) |
| 25 | 1024 | 2,775 | 2414 us | 503 us (4.8x) | 3.18 ms | 0.40 ms (8.0x) |
| 25 | 1024 | 64,875 | 2566 us | 485 us (5.3x) | 72.33 ms | 5.59 ms (12.9x) |

The gate kernel is about five times the torch path at every size, since it
replaces a per-gate-type gather and scatter with one launch. The parity
kernel loses at a few hundred errors, where both paths are launch-bound, and
wins by 8 to 13 times once the error set has thousands of rows, which is the
regime of the Knill-Laflamme reward on 16 to 25 qubits.
