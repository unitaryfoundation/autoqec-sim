# The evaluator against published memory experiments

Every number comes from the JSON files under `experiments/hardware_comparison/runs/`; this file regenerates with `uv run python -m experiments.hardware_comparison.run report`. Code: `experiments/hardware_comparison/run.py`.

Each row builds the package's own memory circuit for the paper's code, applies a noise profile whose parameters are read from the paper (assumptions listed per row), and scores it with the proxy tier at one to three rounds and with Tesseract sampling (and minimum-weight matching for the surface, toric, and heavy-hexagon codes, where a single data error flips at most two checks of each type) at the paper's round count; per-round values use the memory-decay convention of the papers. The measured column is the paper's own number with the paper's decoder. This is a comparison of the package's noise model on the package's circuit with a reported number, not a prediction about a device: the circuits differ from the devices' (bare or one-flag extraction, no adaptive protocol, no transport), several parameters are assumptions named below, and leakage is omitted everywhere.

## Summary

| row | code | rounds | measured per round | proxy, one round | Tesseract at the paper's rounds | matching | Tesseract over measured |
|---|---|---|---|---|---|---|---|
| google_2023 | [[9,1]] google-2023 | 25 | 3.03e-02 (both bases averaged) | Z 9.51e-03, X 1.21e-02 | Z 2.14e-02 (6655), X 2.56e-02 (7315) | Z 2.87e-02, X 3.06e-02 | Z 0.71, X 0.85 |
| google_2024_72q | [[9,1]] google-2024-72q | 25 | 5.90e-03 (both bases averaged) | Z 1.41e-03, X 1.98e-03 | Z 3.54e-03 (1626), X 4.69e-03 (2100) | Z 5.41e-03, X 6.29e-03 | Z 0.60, X 0.80 |
| quantinuum_h1_2021_noidle | [[7,1]] h1-2021-none | 6 | Z 1.05e-02, X 1.75e-02 | Z 2.56e-03, X 2.56e-03 | Z 3.10e-03 (366), X 3.07e-03 (363) | not a matching code | Z 0.30, X 0.18 |
| quantinuum_h1_2021_idlebound | [[7,1]] h1-2021-bound | 6 | Z 1.05e-02, X 1.75e-02 | Z 2.91e-03, X 5.85e-03 | Z 3.36e-03 (397), X 7.79e-03 (899) | not a matching code | Z 0.32, X 0.45 |
| ion_chain_toric_18 | [[18,2]] chain-2606.06455 | 6 | Z 1.51e-02, X 2.88e-02 | Z 4.48e-03, X 7.08e-03 | Z 1.30e-02 (1462), X 1.78e-02 (1955) | Z 2.39e-02, X 2.85e-02 | Z 0.86, X 0.62 |
| ion_chain_bb_18_4_3 | [[18,4]] chain-2606.06455 | 6 | Z 4.33e-02, X 7.58e-02 | Z 8.35e-03, X 1.28e-02 | Z 2.94e-02 (3048), X 3.86e-02 (3825) | not a matching code | Z 0.68, X 0.51 |
| ibm_peekskill_2022_flagged | [[9,1]] peekskill-decoder | 10 | Z 5.85e-02, X 1.10e-01 | Z 2.20e-02, X 2.25e-02 | Z 4.44e-02 (6053), X 4.50e-02 (6110) | Z 5.59e-02, X 4.90e-02 | Z 0.76, X 0.41 |
| ibm_peekskill_2022_bare | [[9,1]] peekskill-decoder | 10 | Z 5.85e-02, X 1.10e-01 | Z 2.14e-02, X 1.92e-02 | Z 4.22e-02 (5855), X 4.02e-02 (5676) | Z 5.18e-02, X 4.31e-02 | Z 0.72, X 0.37 |
| ibm_peekskill_2022_medians | [[9,1]] peekskill-median | 10 | Z 5.85e-02, X 1.10e-01 | Z 1.27e-02, X 3.05e-02 | Z 3.04e-02 (4663), X 5.21e-02 (6675) | Z 3.72e-02, X 5.40e-02 | Z 0.52, X 0.47 |

Failure counts in parentheses are Tesseract failures out of the shots per basis; a count below about 50 leaves a standard error above 15 percent.

## Sources

Every measured value and every profile parameter was read from the paper named in the row's profile and checked against the arXiv text on 2026-10-17; the per-row notes above name the quantities the papers do not state and the choices made for them. The papers: Google Quantum AI 2023 (arXiv:2207.06431, Nature 614, 676), Google Quantum AI 2024 (arXiv:2408.13687, Nature 638, 920), Ryan-Anderson et al. 2021 (arXiv:2107.07505, Physical Review X 11, 041058), Tham et al. 2026 (arXiv:2606.06455), and Sundaresan et al. 2023 (arXiv:2203.07205, Nature Communications 14, 2852).

## Rows

### google_2023

Code [[9,1]], profile `google-2023` (parallel timing, connectivity `square:3x3`, flags `none`), extraction `interleaved`. Measured: main text: 3.028 +- 0.023 percent per cycle, the distance-3 subset codes averaged over the Z and X bases, fit over cycles 3 to 25; decoder tensor network (belief matching 3.118e-2). Circuit and model: two ASSUMED gate durations; leakage from heating omitted.

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 9.512e-03 | 0.61 | 135 | 8 | 3 |
| 1 | X | 1.208e-02 | 0.61 | 135 | 8 | 3 |
| 2 | Z | 1.552e-02 | 1.21 | 247 | 16 | 3 |
| 2 | X | 1.976e-02 | 1.21 | 247 | 16 | 3 |
| 3 | Z | 1.863e-02 | 1.82 | 359 | 24 | 3 |
| 3 | X | 2.369e-02 | 1.82 | 359 | 24 | 3 |

### google_2024_72q

Code [[9,1]], profile `google-2024-72q` (parallel timing, connectivity `square:3x3`, flags `none`), extraction `interleaved`. Measured: 72-qubit processor, supplementary Table S1 (Fig. 4c columns): 5.9 +- 0.1 e-3 per cycle, both bases averaged; decoder neural network (Libra 6.1e-3, real-time matching 7.1e-3). Circuit and model: T1, T2, measurement and reset durations ASSUMED; leakage heating omitted.

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 1.407e-03 | 0.25 | 135 | 8 | 3 |
| 1 | X | 1.980e-03 | 0.25 | 135 | 8 | 3 |
| 2 | Z | 2.297e-03 | 0.51 | 247 | 16 | 3 |
| 2 | X | 3.254e-03 | 0.51 | 247 | 16 | 3 |
| 3 | Z | 2.721e-03 | 0.76 | 359 | 24 | 3 |
| 3 | X | 3.851e-03 | 0.76 | 359 | 24 | 3 |

### quantinuum_h1_2021_noidle

Code [[7,1]], profile `h1-2021-none` (gates_per_slot=3 timing, connectivity `all_to_all`, flags `none`), extraction `batched`. Measured: Table I: |0> 1.16e-2 and |1> 0.94e-2 averaged; |+> 1.8e-2 and |-> 1.7e-2 averaged; decoder lookup table on flagged syndromes. Circuit and model: bare 24-gate extraction against the paper's flagged adaptive 36- to 60-gate protocol; no transport or cooling dephasing.

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 2.559e-03 | 0.09 | 149 | 20 | 2 |
| 1 | X | 2.561e-03 | 0.09 | 149 | 20 | 2 |
| 2 | Z | 2.995e-03 | 0.19 | 277 | 40 | 2 |
| 2 | X | 3.003e-03 | 0.19 | 277 | 40 | 2 |
| 3 | Z | 3.285e-03 | 0.28 | 405 | 60 | 2 |
| 3 | X | 3.290e-03 | 0.28 | 405 | 60 | 2 |

### quantinuum_h1_2021_idlebound

Code [[7,1]], profile `h1-2021-bound` (gates_per_slot=3 timing, connectivity `all_to_all`, flags `none`), extraction `batched`. Measured: Table I: |0> 1.16e-2 and |1> 0.94e-2 averaged; |+> 1.8e-2 and |-> 1.7e-2 averaged; decoder lookup table on flagged syndromes. Circuit and model: as above, with the depth-one memory error bound as a Z-dominated idle channel (ASSUMPTION).

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 2.910e-03 | 0.16 | 149 | 20 | 2 |
| 1 | X | 5.853e-03 | 0.16 | 149 | 20 | 2 |
| 2 | Z | 3.601e-03 | 0.32 | 277 | 40 | 2 |
| 2 | X | 7.985e-03 | 0.32 | 277 | 40 | 2 |
| 3 | Z | 4.070e-03 | 0.48 | 405 | 60 | 2 |
| 3 | X | 9.052e-03 | 0.48 | 405 | 60 | 2 |

### ion_chain_toric_18

Code [[18,2]], profile `chain-2606.06455` (sequential timing, connectivity `all_to_all`, flags `none`), extraction `batched`. Measured: Table I per logical qubit (Z 0.76e-2, X 1.45e-2) converted to the two-qubit block; decoder beam decoder. Circuit and model: two-qubit rate read from a figure (preset ASSUMPTION); the paper remaps qubits to ions per code.

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 4.477e-03 | 0.83 | 422 | 25 | 3 |
| 1 | X | 7.077e-03 | 0.83 | 422 | 25 | 3 |
| 2 | Z | 6.369e-03 | 1.65 | 790 | 50 | 3 |
| 2 | X | 9.706e-03 | 1.65 | 790 | 50 | 3 |
| 3 | Z | 7.121e-03 | 2.48 | 1158 | 75 | 3 |
| 3 | X | 1.079e-02 | 2.48 | 1158 | 75 | 3 |

### ion_chain_bb_18_4_3

Code [[18,4]], profile `chain-2606.06455` (sequential timing, connectivity `all_to_all`, flags `none`), extraction `batched`. Measured: Table I per logical qubit (Z 1.10e-2, X 1.95e-2) converted to the four-qubit block; decoder beam decoder. Circuit and model: as above.

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 8.351e-03 | 0.86 | 502 | 36 | 3 |
| 1 | X | 1.281e-02 | 0.86 | 502 | 36 | 3 |
| 2 | Z | 1.105e-02 | 1.73 | 950 | 72 | 3 |
| 2 | X | 1.664e-02 | 1.73 | 950 | 72 | 3 |
| 3 | Z | 1.218e-02 | 2.59 | 1398 | 108 | 3 |
| 3 | X | 1.834e-02 | 2.59 | 1398 | 108 | 3 |

### ibm_peekskill_2022_flagged

Code [[9,1]], profile `peekskill-decoder` (parallel timing, connectivity `all_to_all`, flags `one`), extraction `batched`. Measured: main text fits over rounds 0 to 10, 500,000 shots per run: |0> 0.059 and |1> 0.058 averaged; |+> 0.113 and |-> 0.107 averaged; decoder matching, all shots (leakage post-selected Z 4.25e-2, X 8.65e-2; maximum likelihood on post-selected data 0.037 and 0.087 in the published abstract). Circuit and model: gauge generators measured with one flag on each weight-4 Z gauge, the paper's own circuit-level parameters; no leakage.

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 2.196e-02 | 1.00 | 112 | 24 | 3 |
| 1 | X | 2.252e-02 | 1.00 | 112 | 24 | 3 |
| 2 | Z | 3.570e-02 | 1.99 | 205 | 48 | 3 |
| 2 | X | 3.653e-02 | 1.99 | 205 | 48 | 3 |
| 3 | Z | 4.397e-02 | 2.99 | 298 | 72 | 3 |
| 3 | X | 4.497e-02 | 2.99 | 298 | 72 | 3 |

### ibm_peekskill_2022_bare

Code [[9,1]], profile `peekskill-decoder` (parallel timing, connectivity `all_to_all`, flags `none`), extraction `batched`. Measured: main text fits over rounds 0 to 10, 500,000 shots per run: |0> 0.059 and |1> 0.058 averaged; |+> 0.113 and |-> 0.107 averaged; decoder matching, all shots (leakage post-selected Z 4.25e-2, X 8.65e-2; maximum likelihood on post-selected data 0.037 and 0.087 in the published abstract). Circuit and model: as above without flags.

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 2.139e-02 | 0.84 | 100 | 22 | 3 |
| 1 | X | 1.917e-02 | 0.84 | 100 | 22 | 3 |
| 2 | Z | 3.508e-02 | 1.69 | 181 | 44 | 3 |
| 2 | X | 3.170e-02 | 1.69 | 181 | 44 | 3 |
| 3 | Z | 4.327e-02 | 2.53 | 262 | 66 | 3 |
| 3 | X | 3.904e-02 | 2.53 | 262 | 66 | 3 |

### ibm_peekskill_2022_medians

Code [[9,1]], profile `peekskill-median` (parallel timing, connectivity `all_to_all`, flags `one`), extraction `batched`. Measured: main text fits over rounds 0 to 10, 500,000 shots per run: |0> 0.059 and |1> 0.058 averaged; |+> 0.113 and |-> 0.107 averaged; decoder matching, all shots (leakage post-selected Z 4.25e-2, X 8.65e-2; maximum likelihood on post-selected data 0.037 and 0.087 in the published abstract). Circuit and model: medians of the per-qubit tables with the duration idle model instead of the paper's circuit-level model.

| rounds | basis | proxy per round | expected faults per shot | mechanisms | layers | d_circ |
|---|---|---|---|---|---|---|
| 1 | Z | 1.268e-02 | 0.89 | 112 | 24 | 3 |
| 1 | X | 3.052e-02 | 0.89 | 112 | 24 | 3 |
| 2 | Z | 2.025e-02 | 1.79 | 205 | 48 | 3 |
| 2 | X | 5.258e-02 | 1.79 | 205 | 48 | 3 |
| 3 | Z | 2.444e-02 | 2.68 | 298 | 72 | 3 |
| 3 | X | 6.686e-02 | 2.68 | 298 | 72 | 3 |
