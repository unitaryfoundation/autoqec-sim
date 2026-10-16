# CPU profile of the vectorized simulator

Throughput and memory of the batched tableau, the Knill-Laflamme reward,
and `CodeDiscoveryEnv`, measured by `scripts/profile_tableau.py` (stdlib
and numpy; no torch). This is a numpy profile on the host CPU of one
laptop, not the GPU-vectorized simulator the roadmap's week-4 wording
names; the cuQuantum kernels have not been written (README, "What is left
for cuQuantum").

Machine: macOS-26.5.1-arm64-arm-64bit, Apple M5 Pro, 18 logical CPUs, BLAS
threads capped at 6 (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`); Python 3.12.14, numpy 2.5.3
(BLAS: accelerate). Whole run: 24 s on 2026-09-21.

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

| n | batch | actions | errors (w <= 2) | tableau B/elem | reward intermediate B/elem | gate applications/s | reward evaluations/s | env steps/s |
|---|---|---|---|---|---|---|---|---|
| 5 | 256 | 30 | 105 | 110 B | 7.4 KiB | 5.24 M | 206.9 k | 189.7 k |
| 5 | 1024 | 30 | 105 | 110 B | 7.4 KiB | 7.28 M | 197.4 k | 199.1 k |
| 5 | 4096 | 30 | 105 | 110 B | 7.4 KiB | 8.40 M | 197.7 k | 207.7 k |
| 9 | 256 | 90 | 351 | 342 B | 41.1 KiB | 4.34 M | 81.4 k | 91.4 k |
| 9 | 1024 | 90 | 351 | 342 B | 41.1 KiB | 6.15 M | 80.7 k | 92.1 k |
| 9 | 4096 | 90 | 351 | 342 B | 41.1 KiB | 6.88 M | 79.9 k | 100.4 k |
| 16 | 256 | 272 | 1,128 | 1.0 KiB | 224.7 KiB | 3.72 M | 23.2 k | 26.0 k |
| 16 | 1024 | 272 | 1,128 | 1.0 KiB | 224.7 KiB | 5.04 M | 24.4 k | 28.2 k |
| 16 | 4096 | 272 | 1,128 | 1.0 KiB | 224.7 KiB | 5.34 M | 24.4 k | 27.7 k |
| 25 | 256 | 650 | 2,775 | 2.5 KiB | 845.5 KiB | 3.04 M | 10.6 k | 11.3 k |
| 25 | 1024 | 650 | 2,775 | 2.5 KiB | 845.5 KiB | 3.57 M | 10.7 k | 11.4 k |
| 25 | 4096 | 650 | 2,775 | 2.5 KiB | 845.5 KiB | 3.66 M | 10.6 k | 11.2 k |
| 49 | 256 | 2450 | 10,731 | 9.5 KiB | 6.1 MiB | 1.34 M | 1.7 k | 1.7 k |
| 49 | 1024 | 2450 | 10,731 | 9.5 KiB | 6.1 MiB | 1.41 M | not run (6.1 GiB intermediate > 4 GiB cap) | not run |
| 49 | 4096 | 2450 | 10,731 | 9.5 KiB | 6.1 MiB | 1.12 M | not run (24.6 GiB intermediate > 4 GiB cap) | not run |

## Where the time goes

The reward and the gate kernels are timed inside `env.step`; the pieces of
the Knill-Laflamme term are timed on their own on the same arrays, and
their sum is within measurement noise of the term.

| piece (n = 16, batch = 1024) | ms per call | share of env.step |
|---|---|---|
| `env.step` | 37.14 | 100% |
| `ActionTable.apply` | 0.24 | 1% |
| `reward` | 36.81 | 99% |
| `Knill-Laflamme term` | 35.19 | 95% |
| `weight-1 stabilizer penalty` | 1.61 | 4% |
| `code_rows and float32 casts` | 0.30 | 1% |
| `GEMM (M, 2n) x (2n, n + k)` | 5.41 | 15% |
| `int64 cast and & 1 (mod 2)` | 10.01 | 27% |
| `two any() reductions over n + k` | 19.36 | 52% |
| `und @ lam and normalization` | 0.36 | 1% |

At `n = 16`, batch 1024, the reward is 99% of an environment step and
the gate kernels 1%; the remainder is `env.step` bookkeeping (history, masks,
reset of finished elements). Inside the Knill-Laflamme term the float32
GEMM of the error set against the code rows is 15%, the int64 cast for
the mod-2 reduction 28%, and the two boolean `any()` reductions over the
`(B, M, n + k)` product 55%, so the README's list is right that the
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
