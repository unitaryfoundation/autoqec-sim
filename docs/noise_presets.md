# Preset parameters and sources

Rates at scale 1. Every number is sourced or labeled ASSUMPTION.

| Preset | p2 | p1 | p_reset | p_meas | idle | durations (2q / 1q / meas / reset) | leakage | default timing | source |
|---|---|---|---|---|---|---|---|---|---|
| `forte-2308.05071` | 4.0e-3 | 2.0e-4 | 1.0e-3 (ASSUMPTION: SPAM split) | 4.0e-3 (ASSUMPTION: SPAM split) | duration, T1 10 s, T2 1 s, all qubits | 900 us / 110 us / 200 us (ASSUMPTION) / 20 us (ASSUMPTION) | none | parallel | IonQ Forte page; Chen et al., Quantum 8, 1516 (2024), arXiv:2308.05071; SPEC.md table 1 |
| `chain-2606.06455` | 8.5e-3 (read off Fig. 7, ASSUMPTION +-1e-3) | 3.2e-4 | 3.2e-4 (ASSUMPTION) | 9.0e-3 | slot: gate (5e-6, 5e-6, 4e-5), meas (5e-5, 5e-5, 4e-4), X/Y split ASSUMPTION | 0.46 ms / 0.46 ms / 9.7 ms / 0.46 ms (DERIVED, unused by the recipe) | 1.7e-3 per measurement layer, Pauli proxy | sequential | Tham et al., arXiv:2606.06455 Table V, Appendices C and E; SPEC.md table 2 |
| `board-rfc0001` | 1e-3 | none | 1e-3 | 1e-3 | slot: DEPOLARIZE1(1e-3), data qubits only | not used | none | parallel | qldpc-challenge `verify/circuit_tools.apply_noise`, `P_REF` |
| `heron-r2` | 3.5e-3 | 2.0e-4 | 1.5e-2 (ASSUMPTION) | 1.5e-2 | duration, T1 213 us, T2 120 us, all qubits | 68 ns / 32 ns / 3.5 us (ASSUMPTION: midpoint of 3 to 4 us) / 3.53 us (ASSUMPTION) | none (unsourced) | parallel | see below |
| `depolarizing(p)` | p | p | p | p | slot: DEPOLARIZE1(p) | not used | none | parallel | uniform reference, not a device |

`heron-r2` sources, accessed 2026-09-20. IBM Heron r2: 156 fixed-frequency
transmons, tunable couplers, heavy-hex lattice, native gates CZ, RZ, SX, X,
measurement, reset (measurement followed by a conditional X).

| Parameter | Value | Source and status |
|---|---|---|
| p2 | 3.5e-3 | Harper et al., "Characterising the failure mechanisms of error-corrected quantum logic gates", arXiv:2504.07258 (Nat. Commun. 2026), Sec. IV.1.2: median two-qubit error "approximately 0.35%" on Heron-class devices. Cross-check: IBM Quantum Platform table for ibm_fez quotes a 2.77e-3 median, and 2.85e-3 is quoted for Heron r2 in secondary sources attributing it to arXiv:2410.00916 (not verified against that primary) |
| p1 | 2.0e-4 | arXiv:2504.07258 Table 1, fitted single-qubit depolarizing 0.02%. A fitted model value, not an RB median; the SX medians quoted for Heron executions elsewhere (2.2e-4) agree |
| p_meas | 1.5e-2 | IBM Quantum Platform ibm_fez median readout assignment error 1.55e-2 as quoted in secondary sources on 2026-09-20. arXiv:2504.07258 Table 1 fits 1.2% quantum plus 4.2% classical measurement error; the assignment error is used |
| p_reset | 1.5e-2 | ASSUMPTION. IBM reset is measure plus conditional X (arXiv:2510.18847 appendix), so its flip rate is taken equal to the readout error. arXiv:2504.07258 Table 1 fits 7.5% reset noise, which would make reset the dominant term; not adopted without a characterization |
| T1, T2 | 213 us, 120 us | arXiv:2504.07258 Sec. IV.1.2, "on the best Heron device". Medians of 218 us and 264 us are quoted for Heron r2 in secondary sources; the primary-paper pair is used |
| t_2q | 68 ns | CZ duration for Heron r2, quoted in secondary sources (Legato.jl issue 15, attributing arXiv:2410.00916). Not verified against the primary |
| t_1q | 32 ns | Vezvaee et al., "Surface code scaling on heavy-hex superconducting quantum processors", arXiv:2510.18847, appendix: "an X gate is 32 ns = 8 dt" on ibm_aachen (Heron r2) |
| t_meas | 3.5 us | arXiv:2504.07258 Sec. IV.1.2: measurement "between 3000 and 4000 ns"; midpoint, ASSUMPTION. Other Heron devices quote 1.3 to 2.6 us readout lengths |
| t_reset | t_meas + t_1q | ASSUMPTION from the measure-plus-X construction |
| bias | 1/2 | ASSUMPTION: no public Pauli decomposition of CZ or SX on Heron |
| leakage | 0 | not sourced: no published CZ leakage rate for Heron; left at 0 and stated |
| crosstalk | 0 | arXiv:2510.18847 names coherent ZZ crosstalk as the dominant term dynamical decoupling suppresses, with no rate; set `Correlated(crosstalk=...)` per run |

The Heron idle term at these numbers is small per layer (CZ layer pz =
2.8e-4, meas layer pz = 1.5e-2 at scale 1) because T2 is 1800 CZ
durations; the measurement layer dominates the idle budget, the reverse of
the Forte set.

## Parameters that are assumptions

Forte and chain sets: (SPAM split, t_meas, t_reset,
two-qubit Pauli structure, Yb+ leakage, simultaneous gate count; chain p2
read off a figure, p_reset, X/Y idle split). Heron set: p_reset, t_meas
(midpoint of a stated range), t_reset, the Pauli bias of CZ and SX, CZ
leakage, and the ZZ crosstalk rate; t_2q = 68 ns and the ibm_fez medians
are from secondary quotations of IBM's device table, not from a paper I
could open. The two-qubit biased channel's product-weight form is a
construction, not a measurement, for every preset.
