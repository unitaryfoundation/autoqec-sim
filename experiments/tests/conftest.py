import os

# Same BLAS thread cap as tests/conftest.py, for a run of this directory alone.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
