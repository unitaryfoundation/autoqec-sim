# Installation

autoqec-sim is not published on PyPI. Install it from the repository at a
release tag:

```bash
pip install "autoqec-sim[rl] @ git+https://github.com/unitaryfoundation/autoqec-sim@v0.2.0"
```

!!! note "Python 3.12 to 3.14"
    The package runs on Python 3.12, 3.13, and 3.14. Its only dependencies
    are `numpy` and `stim`.

## Extras

| Extra | Adds | Needed for |
|---|---|---|
| `rl` | `torch` | the PPO agent, `autoqec-ppo`, `autoqec-schedule`, and the torch backend |

Without the extra, the evolutionary search, `autoqec-regress`, and every
reward run on numpy.

## GPU

The torch backend runs on any CUDA device that torch sees. Install a CUDA
build of torch for your driver, then install autoqec-sim without the `rl`
extra so the CUDA build is kept. In this repository, `uv sync --extra rl`
pins torch to the CPU-only index on Linux, so on a GPU host use
`pip install -e .` into an environment that already has CUDA torch.

The two fused CUDA kernels in `autoqec_sim.backends.cuda_kernels` are
compiled at first use through CuPy's NVRTC path and need `cupy` for the
installed CUDA version. They are opt-in: set `AUTOQEC_CUDA_KERNELS=1` or
call `cuda_kernels.enable()`. [GPU backend](../gpu.md) has the procedure
and the measurements.

## From Source

```bash
git clone https://github.com/unitaryfoundation/autoqec-sim && cd autoqec-sim
uv sync --group dev --extra rl     # plain `uv sync --group dev` skips torch and the PPO tests
uv run pytest                      # about 70 s; the reduced regressions are in the suite
```

The documentation builds with `uv run --group docs mkdocs serve`.

Next, follow the [Quick Start](quickstart.md) or pick a path from
[Choose a Workflow](choosing-a-workflow.md).
