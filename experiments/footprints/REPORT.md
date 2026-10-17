# Letter assignments on the Steane and 4.8.8 footprints

Every number comes from the JSON lines under `experiments/footprints/runs/`; this file regenerates with `uv run python -m experiments.footprints.run report`. Code: `experiments/footprints/run.py`.

Each footprint is a set of faces; a code is an assignment of two Pauli strings per face, nonzero on the face, zero outside it, distinct within the face, with every pair commuting, sampled by CMSGen and deduplicated by stabilizer group. The reference is the CSS code on the faces. Scores are the proxy tier with the exact circuit distance under the Forte preset (sequential timing, scale 0.25, a model assembled from public specifications, not a device measurement) and a depolarizing profile with a biased idle channel (bias 5, parallel timing), with rounds equal to the footprint's distance and both memory bases. The confirmation is Tesseract sampling under the Forte case.

## Steane [[7,1,3]] footprint

300 sampled groups on 3 faces, 3 rounds. Distance distribution: d = 1: 168, d = 3: 132 (152 of the d = 1 codes have the same letter in both generators of a face on a qubit of that face only). Full-distance codes: 132, with 119 having 21 weight-3 logicals, 13 having 30 weight-3 logicals (the CSS reference has 21).

| full-distance codes | Forte preset, sequential, scale 0.25 | biased idle, bias 5, parallel |
|---|---|---|
| CSS reference p_round (Z, X) | 2.096e-03 (8.10e-04, 1.29e-03) | 9.323e-04 (2.31e-04, 7.01e-04) |
| CSS reference d_circ (Z, X), extraction | (2, 2), batched | (2, 2), batched |
| random min / median / max | 9.607e-04 / 2.251e-03 / 3.097e-03 | 3.195e-04 / 4.934e-04 / 3.306e-03 |
| spread max / min | 3.22 | 10.35 |
| best / CSS | 0.46 | 0.34 |
| codes below CSS | 41 of 132 | 100 of 132 |
| d_circ (Z, X) over the codes | (2, 2) | (2, 2) |
| extraction | batched, interleaved | batched, interleaved |
| Spearman, two-basis minimum-weight fault mass against p_round | 0.64 | 0.20 |
| the same over every code with d >= 3 | 0.64 | 0.20 |
| expected faults per shot, median | 0.19 | 0.08 |
| seconds per evaluation (both bases), median | 0.30 | 0.20 |

Spearman between the two cases over the full-distance codes: 0.31.

Best full-distance code, Forte preset, sequential, scale 0.25 (S1_256, p_round 9.607e-04, Z 4.93e-04, X 4.68e-04):

    IIIZXXY  IIIYZXY  IYYIIZX  IXYIIYX  YIXIYIZ  XIZIYIZ

Best full-distance code, biased idle, bias 5, parallel (S1_219, p_round 3.195e-04, Z 1.08e-04, X 2.11e-04):

    IIIYZXX  IIIXXZZ  IYXIIZZ  IZYIIXX  ZIYIZIX  XIXIXIZ

Tesseract confirmation, Forte case, 50,000 shots per basis, 8 codes: Tesseract over proxy median 0.98 (range 0.83 to 1.06), Spearman 0.90, 156 failures per code (median).

| code | d | proxy p_round | Tesseract p_round | ratio | failures (Z, X) |
|---|---|---|---|---|---|
| S1_256 | 3 | 9.607e-04 | 8.875e-04 | 0.92 | 73, 60 |
| S1_91 | 3 | 9.753e-04 | 9.676e-04 | 0.99 | 66, 79 |
| S1_12 | 3 | 1.024e-03 | 1.034e-03 | 1.01 | 77, 78 |
| S1_196 | 3 | 1.042e-03 | 1.054e-03 | 1.01 | 87, 71 |
| S1_241 | 3 | 1.058e-03 | 1.014e-03 | 0.96 | 59, 93 |
| css | 3 | 2.096e-03 | 2.212e-03 | 1.06 | 123, 208 |
| S1_247 | 3 | 2.257e-03 | 1.864e-03 | 0.83 | 112, 167 |
| S1_105 | 3 | 3.097e-03 | 2.681e-03 | 0.87 | 253, 148 |

## Steane [[7,1,3]] footprint, every code measured in batches

300 sampled groups on 3 faces, 3 rounds. Distance distribution: d = 1: 168, d = 3: 132 (152 of the d = 1 codes have the same letter in both generators of a face on a qubit of that face only). Full-distance codes: 132, with 119 having 21 weight-3 logicals, 13 having 30 weight-3 logicals (the CSS reference has 21).

| full-distance codes | Forte preset, sequential, scale 0.25 | biased idle, bias 5, parallel |
|---|---|---|
| CSS reference p_round (Z, X) | 2.096e-03 (8.10e-04, 1.29e-03) | 9.323e-04 (2.31e-04, 7.01e-04) |
| CSS reference d_circ (Z, X), extraction | (2, 2), batched | (2, 2), batched |
| random min / median / max | 9.607e-04 / 2.236e-03 / 3.086e-03 | 7.308e-04 / 1.214e-03 / 3.306e-03 |
| spread max / min | 3.21 | 4.52 |
| best / CSS | 0.46 | 0.78 |
| codes below CSS | 37 of 132 | 6 of 132 |
| d_circ (Z, X) over the codes | (2, 2) | (2, 2) |
| extraction | batched | batched |
| Spearman, two-basis minimum-weight fault mass against p_round | 0.68 | 0.08 |
| the same over every code with d >= 3 | 0.68 | 0.08 |
| expected faults per shot, median | 0.16 | 0.16 |
| seconds per evaluation (both bases), median | 0.20 | 0.10 |

Spearman between the two cases over the full-distance codes: 0.11.

Best full-distance code, Forte preset, sequential, scale 0.25 (S1_256, p_round 9.607e-04, Z 4.93e-04, X 4.68e-04):

    IIIZXXY  IIIYZXY  IYYIIZX  IXYIIYX  YIXIYIZ  XIZIYIZ

Best full-distance code, biased idle, bias 5, parallel (S1_218, p_round 7.308e-04, Z 3.69e-04, X 3.61e-04):

    IIIXZXZ  IIIZYYX  IZXIIXZ  IYYIIYX  YIXIZIZ  ZIYIYIX

## 4.8.8 color [[17,1,5]] footprint

120 sampled groups on 8 faces, 5 rounds. Distance distribution: d = 1: 34, d = 3: 57, d = 5: 29 (27 of the d = 1 codes have the same letter in both generators of a face on a qubit of that face only). Full-distance codes: 29, with 29 having 153 weight-5 logicals (the CSS reference has 153).

| full-distance codes | Forte preset, sequential, scale 0.25 | biased idle, bias 5, parallel |
|---|---|---|
| CSS reference p_round (Z, X) | 2.945e-04 (4.81e-05, 2.46e-04) | 1.276e-05 (2.69e-06, 1.01e-05) |
| CSS reference d_circ (Z, X), extraction | (3, 3), interleaved | (3, 3), interleaved |
| random min / median / max | 1.191e-04 / 2.191e-04 / 3.324e-04 | 7.374e-06 / 1.221e-05 / 1.893e-05 |
| spread max / min | 2.79 | 2.57 |
| best / CSS | 0.40 | 0.58 |
| codes below CSS | 27 of 29 | 20 of 29 |
| d_circ (Z, X) over the codes | (3, 3) | (3, 3) |
| extraction | interleaved | interleaved |
| Spearman, two-basis minimum-weight fault mass against p_round | 0.81 | 0.79 |
| the same over every code with d >= 3 | 0.90 | 0.95 |
| expected faults per shot, median | 1.81 | 0.42 |
| seconds per evaluation (both bases), median | 6.00 | 5.10 |

Spearman between the two cases over the full-distance codes: 0.84.

Best full-distance code, Forte preset, sequential, scale 0.25 (S1_11, p_round 1.191e-04, Z 5.64e-05, X 6.26e-05):

    IIYZIXIZIXIXIXXII  IIZXIZIYIYIYIYZII  YXYZIIIIIIIIIIIII  ZZXYIIIIIIIIIIIII  IIIIYXXZIIIIIIIII  IIIIXZYYIIIIIIIII  IIIIIIIIIYYYYIIII  IIIIIIIIIZXZZIIII  XIZIXZIIIIIIIIIII  YIYIYXIIIIIIIIIII  IIIIIIXZIIIIIXIXI  IIIIIIZXIIIIIZIYI  IIIIIIIIYIYIYIIIX  IIIIIIIIZIZIXIIIY  IIIIIIIIIIIZZIYIZ  IIIIIIIIIIIYYIZIX

Best full-distance code, biased idle, bias 5, parallel (S1_11, p_round 7.374e-06, Z 3.27e-06, X 4.10e-06):

    IIYZIXIZIXIXIXXII  IIZXIZIYIYIYIYZII  YXYZIIIIIIIIIIIII  ZZXYIIIIIIIIIIIII  IIIIYXXZIIIIIIIII  IIIIXZYYIIIIIIIII  IIIIIIIIIYYYYIIII  IIIIIIIIIZXZZIIII  XIZIXZIIIIIIIIIII  YIYIYXIIIIIIIIIII  IIIIIIXZIIIIIXIXI  IIIIIIZXIIIIIZIYI  IIIIIIIIYIYIYIIIX  IIIIIIIIZIZIXIIIY  IIIIIIIIIIIZZIYIZ  IIIIIIIIIIIYYIZIX

Tesseract confirmation, Forte case, 50,000 shots per basis, 6 codes: Tesseract over proxy median 4.35 (range 3.51 to 5.48), Spearman 0.60, 170 failures per code (median).

| code | d | proxy p_round | Tesseract p_round | ratio | failures (Z, X) |
|---|---|---|---|---|---|
| S1_11 | 5 | 1.191e-04 | 6.529e-04 | 5.48 | 63, 100 |
| S1_41 | 5 | 1.453e-04 | 6.288e-04 | 4.33 | 78, 79 |
| S1_21 | 5 | 1.614e-04 | 7.050e-04 | 4.37 | 98, 78 |
| S1_39 | 5 | 1.639e-04 | 6.128e-04 | 3.74 | 59, 94 |
| S1_111 | 5 | 2.191e-04 | 7.692e-04 | 3.51 | 85, 107 |
| css | 5 | 2.945e-04 | 1.488e-03 | 5.05 | 28, 342 |
