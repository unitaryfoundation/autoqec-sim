# SAT-designed codes under the memory objective

Status: complete for the selection below. Every number comes from
`experiments/sat_codes/runs/sat_codes.json`; this file regenerates with
`python -m autoqec_sim.experiments.sat_codes report`. Code:
`src/autoqec_sim/experiments/sat_codes.py`.

Source: DalFavero et al., arXiv:2608.23460, `experiments/surface/` of
github.com/bdalfavero/satisfying-codes: 100,000 SAT-sampled 9-qubit
surface-like codes scored by exact maximum-likelihood logical error rate
under code-capacity biased noise (`p_X = eta P`, `p_Y = p_Z = P`) in the Z
basis, against the rotated surface code and its XZZX deformation. The
selection is the twelve best distinct codes at eta = 10 and three of the
worst. The source LER column is theirs; the p_round columns are
autoqec-sim's circuit-level memory objective, 3 rounds, under a
depolarizing profile with a biased idle channel whose Z rate is eta times
each of the X and Y rates (the codes are Hadamard-conjugated on every qubit
so their X tailoring becomes Z tailoring), and under the IonQ Forte preset
with sequential timing. Distances are recomputed exactly here.

Two structural differences between the two scores. The source's channel is
a single biased code-capacity channel with an exact maximum-likelihood
decoder; here only the idle channel carries the bias, while gate, reset,
and measurement errors stay unbiased, so the effective bias a code sees
over a round is weaker, and the decoder is the tier's (a low-weight
maximum-likelihood proxy, or sampling with the exact decoder where the
detector count allows). The sampled tier stops at the shot and failure
counts in the JSON, so differences below about ten percent are within its
noise; the proxy tier is deterministic.

### source eta = 1

| code | d | CSS | source LER (eta = 1) | proxy p_round | sampled p_round |
|---|---|---|---|---|---|
| D43432 | 3 | no | 0.0057 | 0.00068 | 0.00088 |
| D40235 | 3 | no | 0.0057 | 0.00103 | 0.00097 |
| D88100 | 3 | no | 0.0057 | 0.00068 | 0.00059 |
| D91226 | 3 | no | 0.0057 | 0.00068 | 0.00071 |
| D15671 | 3 | no | 0.0057 | 0.00068 | 0.00088 |
| D2486 | 3 | no | 0.0057 | 0.00101 | 0.00100 |
| D97522 | 3 | no | 0.0057 | 0.00068 | 0.00059 |
| D98396 | 3 | no | 0.0057 | 0.00068 | 0.00083 |
| D22823 | 3 | no | 0.0057 | 0.00068 | 0.00071 |
| D26278 | 3 | no | 0.0057 | 0.00101 | 0.00096 |
| D61094 | 3 | no | 0.0057 | 0.00068 | 0.00088 |
| D50616 | 3 | no | 0.0057 | 0.00068 | 0.00078 |
| D32724 | 3 | no | 0.0108 | 0.00103 | 0.00096 |
| D28519 | 3 | no | 0.0108 | 0.00101 | 0.00099 |
| D29333 | 3 | no | 0.0108 | 0.00103 | 0.00097 |
| rotated_surface | 3 | yes | 0.0057 | 0.00057 | 0.00063 |
| xzzx | 3 | no | 0.0057 | 0.00057 | 0.00053 |

Spearman(source LER, proxy p_round) over 17 codes: 0.52

Spearman(source LER, sampled p_round) over 17 codes: 0.43

### source eta = 4

| code | d | CSS | source LER (eta = 4) | proxy p_round | sampled p_round |
|---|---|---|---|---|---|
| D43432 | 3 | no | 0.0080 | 0.00066 | 0.00073 |
| D40235 | 3 | no | 0.0080 | 0.00081 | 0.00089 |
| D88100 | 3 | no | 0.0080 | 0.00066 | 0.00069 |
| D91226 | 3 | no | 0.0080 | 0.00066 | 0.00073 |
| D15671 | 3 | no | 0.0080 | 0.00066 | 0.00073 |
| D2486 | 3 | no | 0.0080 | 0.00116 | 0.00122 |
| D97522 | 3 | no | 0.0080 | 0.00066 | 0.00069 |
| D98396 | 3 | no | 0.0080 | 0.00066 | 0.00060 |
| D22823 | 3 | no | 0.0080 | 0.00066 | 0.00061 |
| D26278 | 3 | no | 0.0080 | 0.00119 | 0.00123 |
| D61094 | 3 | no | 0.0080 | 0.00066 | 0.00073 |
| D50616 | 3 | no | 0.0080 | 0.00066 | 0.00070 |
| D32724 | 3 | no | 0.0400 | 0.00085 | 0.00097 |
| D28519 | 3 | no | 0.0400 | 0.00130 | 0.00111 |
| D29333 | 3 | no | 0.0400 | 0.00084 | 0.00078 |
| rotated_surface | 3 | yes | 0.0353 | 0.00061 | 0.00057 |
| xzzx | 3 | no | 0.0197 | 0.00050 | 0.00046 |

Spearman(source LER, proxy p_round) over 17 codes: 0.17

Spearman(source LER, sampled p_round) over 17 codes: 0.08

### source eta = 10

| code | d | CSS | source LER (eta = 10) | proxy p_round | sampled p_round |
|---|---|---|---|---|---|
| D43432 | 3 | no | 0.0115 | 0.00065 | 0.00056 |
| D40235 | 3 | no | 0.0115 | 0.00075 | 0.00074 |
| D88100 | 3 | no | 0.0115 | 0.00065 | 0.00059 |
| D91226 | 3 | no | 0.0115 | 0.00065 | 0.00056 |
| D15671 | 3 | no | 0.0115 | 0.00065 | 0.00056 |
| D2486 | 3 | no | 0.0115 | 0.00121 | 0.00134 |
| D97522 | 3 | no | 0.0115 | 0.00065 | 0.00059 |
| D98396 | 3 | no | 0.0115 | 0.00066 | 0.00066 |
| D22823 | 3 | no | 0.0115 | 0.00066 | 0.00065 |
| D26278 | 3 | no | 0.0115 | 0.00127 | 0.00118 |
| D61094 | 3 | no | 0.0115 | 0.00065 | 0.00056 |
| D50616 | 3 | no | 0.0115 | 0.00065 | 0.00059 |
| D32724 | 3 | no | 0.1401 | 0.00083 | 0.00071 |
| D28519 | 3 | no | 0.1401 | 0.00150 | 0.00135 |
| D29333 | 3 | no | 0.1401 | 0.00084 | 0.00092 |
| rotated_surface | 3 | yes | 0.1369 | 0.00068 | 0.00077 |
| xzzx | 3 | no | 0.0615 | 0.00044 | 0.00048 |

Spearman(source LER, proxy p_round) over 17 codes: 0.25

Spearman(source LER, sampled p_round) over 17 codes: 0.26

### IonQ Forte, sequential

| code | d | CSS | proxy p_round | sampled p_round |
|---|---|---|---|---|
| D43432 | 3 | no | 0.00053 | 0.00049 |
| D40235 | 3 | no | 0.00062 | 0.00072 |
| D88100 | 3 | no | 0.00053 | 0.00051 |
| D91226 | 3 | no | 0.00053 | 0.00051 |
| D15671 | 3 | no | 0.00053 | 0.00049 |
| D2486 | 3 | no | 0.00093 | 0.00099 |
| D97522 | 3 | no | 0.00053 | 0.00051 |
| D98396 | 3 | no | 0.00053 | 0.00046 |
| D22823 | 3 | no | 0.00053 | 0.00048 |
| D26278 | 3 | no | 0.00095 | 0.00083 |
| D61094 | 3 | no | 0.00053 | 0.00049 |
| D50616 | 3 | no | 0.00053 | 0.00048 |
| D32724 | 3 | no | 0.00066 | 0.00071 |
| D28519 | 3 | no | 0.00107 | 0.00104 |
| D29333 | 3 | no | 0.00066 | 0.00068 |
| rotated_surface | 3 | yes | 0.00059 | 0.00057 |
| xzzx | 3 | no | 0.00048 | 0.00048 |
