# The proxy tier against Tesseract at circuit distance 5

Every number comes from the JSON lines under `experiments/proxy_tesseract/runs/`; this file regenerates with `uv run python -m experiments.proxy_tesseract.run report`. Code: `experiments/proxy_tesseract/run.py`; codes: `codes.json` (the 5x5 rotated surface code, its XZZX deformation, and 19 random single-qubit Clifford deformations of it, all of distance 5).

The proxy tier is the closed-form maximum-likelihood failure over fault sets of weight at most 3 (at circuit distance 5 the weight-3 form built from the weight-5 and weight-6 undetectable sets), with no shots and no decoder; Tesseract is a most-likely-error search decoder on the same detector error model, sampled with a fixed number of shots. Both score the same circuit. Where a profile is named Forte, it is the package's Forte preset, a noise model assembled from public specifications, not a device measurement.

## forte, 3 rounds, scale 0.25

21 codes, 50,000 Tesseract shots per basis, 1.70 expected faults per shot, extraction interleaved. Seconds per basis: proxy 13, Tesseract 64.

| quantity | Z basis | X basis | both bases |
|---|---|---|---|
| Spearman, proxy against Tesseract | 0.95 | 0.85 | 0.89 |
| Spearman, weight-5 sets only against Tesseract | 0.96 | 0.84 | 0.88 |
| Spearman, weight-5 fault mass against Tesseract | 0.89 | 0.70 | 0.51 |
| Tesseract over proxy, median | 1.33 | 1.19 | 1.31 |
| Tesseract over proxy, range | 0.78 to 2.61 | 0.81 to 1.97 | 0.86 to 1.58 |
| Tesseract failures per code, median (min) | 19 (1) | 27 (8) | 51 (24) |
| share of the proxy from the weight-5 sets, median | 0.37 | 0.38 | 0.38 |

| code | proxy Z | Tesseract Z | proxy X | Tesseract X | both, proxy | both, Tesseract | failures |
|---|---|---|---|---|---|---|---|
| S1_9 | 2.74e-04 | 3.20e-04 | 1.65e-04 | 1.60e-04 | 4.39e-04 | 4.80e-04 | 24 |
| S1_7 | 1.77e-04 | 3.20e-04 | 2.42e-04 | 2.40e-04 | 4.18e-04 | 5.60e-04 | 28 |
| S1_11 | 3.33e-04 | 3.80e-04 | 2.22e-04 | 2.40e-04 | 5.56e-04 | 6.20e-04 | 31 |
| S1_25 | 1.99e-04 | 1.80e-04 | 4.84e-04 | 4.40e-04 | 6.83e-04 | 6.20e-04 | 31 |
| S1_0 | 4.52e-04 | 5.40e-04 | 1.73e-04 | 3.40e-04 | 6.25e-04 | 8.80e-04 | 44 |
| S1_22 | 2.48e-04 | 3.80e-04 | 4.38e-04 | 5.20e-04 | 6.86e-04 | 9.00e-04 | 45 |
| S1_26 | 1.30e-04 | 2.00e-04 | 5.23e-04 | 7.00e-04 | 6.53e-04 | 9.00e-04 | 45 |
| S1_28 | 1.07e-04 | 2.80e-04 | 5.45e-04 | 6.20e-04 | 6.52e-04 | 9.00e-04 | 45 |
| S1_29 | 4.87e-04 | 5.20e-04 | 3.04e-04 | 4.00e-04 | 7.91e-04 | 9.20e-04 | 46 |
| S1_20 | 2.30e-04 | 3.60e-04 | 5.95e-04 | 6.40e-04 | 8.25e-04 | 1.00e-03 | 50 |
| xzzx5 | 3.75e-04 | 5.00e-04 | 4.65e-04 | 5.20e-04 | 8.41e-04 | 1.02e-03 | 51 |
| S1_12 | 6.23e-04 | 8.00e-04 | 1.88e-04 | 2.40e-04 | 8.11e-04 | 1.04e-03 | 52 |
| S1_21 | 5.18e-04 | 7.40e-04 | 2.86e-04 | 3.20e-04 | 8.04e-04 | 1.06e-03 | 53 |
| S1_8 | 3.49e-04 | 4.00e-04 | 5.55e-04 | 6.60e-04 | 9.05e-04 | 1.06e-03 | 53 |
| S1_24 | 1.44e-04 | 3.00e-04 | 5.97e-04 | 7.80e-04 | 7.41e-04 | 1.08e-03 | 54 |
| S1_27 | 2.46e-04 | 4.40e-04 | 4.70e-04 | 6.40e-04 | 7.16e-04 | 1.08e-03 | 54 |
| S1_10 | 2.35e-04 | 3.60e-04 | 5.91e-04 | 8.80e-04 | 8.26e-04 | 1.24e-03 | 62 |
| S1_15 | 7.68e-04 | 7.00e-04 | 6.68e-04 | 5.40e-04 | 1.44e-03 | 1.24e-03 | 62 |
| S1_3 | 4.98e-04 | 5.80e-04 | 6.50e-04 | 1.12e-03 | 1.15e-03 | 1.70e-03 | 85 |
| S1_6 | 7.50e-04 | 1.08e-03 | 3.65e-04 | 6.80e-04 | 1.12e-03 | 1.76e-03 | 88 |
| surface5_css | 2.55e-05 | 2.00e-05 | 2.91e-03 | 2.58e-03 | 2.93e-03 | 2.60e-03 | 130 |

## forte, 5 rounds, scale 0.25

21 codes, 50,000 Tesseract shots per basis, 2.83 expected faults per shot, extraction interleaved. Seconds per basis: proxy 16, Tesseract 92.

| quantity | Z basis | X basis | both bases |
|---|---|---|---|
| Spearman, proxy against Tesseract | 0.97 | 0.91 | 0.93 |
| Spearman, weight-5 sets only against Tesseract | 0.97 | 0.94 | 0.89 |
| Spearman, weight-5 fault mass against Tesseract | 0.92 | 0.76 | 0.65 |
| Tesseract over proxy, median | 1.28 | 1.24 | 1.29 |
| Tesseract over proxy, range | 0.95 to 2.11 | 0.89 to 1.83 | 0.90 to 1.58 |
| Tesseract failures per code, median (min) | 48 (5) | 71 (18) | 109 (56) |
| share of the proxy from the weight-5 sets, median | 0.35 | 0.36 | 0.37 |

| code | proxy Z | Tesseract Z | proxy X | Tesseract X | both, proxy | both, Tesseract | failures |
|---|---|---|---|---|---|---|---|
| S1_7 | 3.81e-04 | 4.60e-04 | 5.38e-04 | 6.60e-04 | 9.19e-04 | 1.12e-03 | 56 |
| S1_9 | 6.25e-04 | 1.04e-03 | 3.57e-04 | 3.60e-04 | 9.83e-04 | 1.40e-03 | 70 |
| S1_11 | 7.20e-04 | 9.00e-04 | 4.97e-04 | 5.20e-04 | 1.22e-03 | 1.42e-03 | 71 |
| S1_0 | 9.87e-04 | 1.24e-03 | 3.78e-04 | 5.80e-04 | 1.37e-03 | 1.82e-03 | 91 |
| S1_26 | 2.83e-04 | 5.80e-04 | 1.14e-03 | 1.24e-03 | 1.42e-03 | 1.82e-03 | 91 |
| S1_28 | 2.37e-04 | 5.00e-04 | 1.20e-03 | 1.42e-03 | 1.43e-03 | 1.92e-03 | 96 |
| S1_25 | 4.70e-04 | 5.40e-04 | 1.08e-03 | 1.46e-03 | 1.55e-03 | 2.00e-03 | 100 |
| S1_21 | 1.13e-03 | 1.36e-03 | 6.13e-04 | 6.60e-04 | 1.74e-03 | 2.02e-03 | 101 |
| S1_22 | 5.35e-04 | 8.20e-04 | 9.46e-04 | 1.24e-03 | 1.48e-03 | 2.06e-03 | 103 |
| S1_10 | 5.32e-04 | 6.80e-04 | 1.30e-03 | 1.50e-03 | 1.83e-03 | 2.18e-03 | 109 |
| S1_24 | 3.26e-04 | 5.40e-04 | 1.33e-03 | 1.64e-03 | 1.65e-03 | 2.18e-03 | 109 |
| S1_29 | 1.04e-03 | 1.30e-03 | 6.76e-04 | 9.00e-04 | 1.72e-03 | 2.20e-03 | 110 |
| S1_8 | 7.80e-04 | 9.60e-04 | 1.24e-03 | 1.32e-03 | 2.02e-03 | 2.28e-03 | 114 |
| S1_12 | 1.36e-03 | 1.76e-03 | 4.16e-04 | 6.00e-04 | 1.78e-03 | 2.36e-03 | 118 |
| S1_27 | 5.32e-04 | 9.60e-04 | 1.05e-03 | 1.42e-03 | 1.58e-03 | 2.38e-03 | 119 |
| xzzx5 | 8.78e-04 | 1.00e-03 | 1.12e-03 | 1.42e-03 | 2.00e-03 | 2.42e-03 | 121 |
| S1_20 | 4.99e-04 | 7.20e-04 | 1.28e-03 | 1.92e-03 | 1.78e-03 | 2.64e-03 | 132 |
| S1_15 | 1.69e-03 | 1.60e-03 | 1.47e-03 | 1.54e-03 | 3.16e-03 | 3.14e-03 | 157 |
| S1_3 | 1.12e-03 | 1.36e-03 | 1.45e-03 | 2.14e-03 | 2.57e-03 | 3.50e-03 | 175 |
| S1_6 | 1.63e-03 | 2.38e-03 | 7.99e-04 | 1.46e-03 | 2.43e-03 | 3.84e-03 | 192 |
| surface5_css | 5.60e-05 | 1.00e-04 | 6.52e-03 | 5.82e-03 | 6.58e-03 | 5.92e-03 | 296 |
