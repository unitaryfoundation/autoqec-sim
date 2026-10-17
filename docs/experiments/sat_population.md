# A population of SAT-designed codes

The [fifteen-code comparison](sat_codes.md) scaled to a population: does
the code-capacity ranking of DalFavero et al. [@dalfavero2026] carry any
information about circuit-level rates, and what does?

## Setup

- Codes: a uniform random sample of 2,000 of the 43,563 distinct stabilizer groups among the source's 100,000 codes (62,876 distinct generator lists), plus its 100 best and 50 worst by code-capacity rate at bias 10, and the two reference codes.
- Cases: biased idle at source bias 4 and 10, every channel biased at bias 10, and the Forte preset (a model from public specifications); `proxy` and `sampled` tiers, three rounds.
- The `prepare` step reads the source repository's `experiments/surface/` directory and writes `experiments/sat_population/population.json`, which is committed.

## Result

![Source code-capacity rate against circuit-level rate at idle bias 10, and the Forte per-round rate against the two-basis mass of minimum-weight undetectable fault sets](../assets/figures/sat_population.png)

| | idle bias 4 | idle bias 10 | all channels, bias 10 | Forte preset |
|---|---|---|---|---|
| Spearman, source rate against circuit-level rate (sampled tier) | 0.09 | 0.18 | 0.18 | 0.00 |
| overlap of the 20 best codes by each score | 0 of 20 | 0 of 20 | 0 of 20 | 0 of 20 |
| best code of the population against XZZX, confirmed at 3 percent standard error | 0.00047 vs 0.00050 | 0.00032 vs 0.00047 | 0.00025 vs 0.00042 | 0.00034 vs 0.00049 |

- The 100 codes the source ranks best score worse than the random sample under bias (median 0.00063 against 0.00051 at bias 10, sampled tier) and no better under Forte.
- Every code in the sample has exactly 24 weight-3 logical operators, so at code capacity under depolarizing noise they are identical to leading order, while their Forte rates span a factor 2.6. The spread is in how many space-time realizations of those logicals go undetected in the extraction circuit: 342 weight-3 undetectable fault sets for the best codes against 462 to 472 for the worst, a tenfold spread in probability mass.
- The mass of those fault sets over both memory bases, a quantity only a circuit-level evaluator computes, predicts the rate in every case (Spearman 0.70 to 0.73, R² 0.47 to 0.61 for a log-log fit). Under bias, the weight-3 mass computed from the stabilizers alone predicts better than the source's exact maximum-likelihood rate at its working point of 1e-2.
- About half the random sample beats XZZX under Forte and two thirds beat the rotated surface code.
- A SAT-constrained local search with the proxy as objective (`experiments/sat_population/local_search.py`: SAT keeps every candidate a valid distance-3 code on the footprint, CMSGen samples completions with all but two generators fixed) reaches 0.000331 to 0.000365 in 129 to 159 evaluations, against 0.000344 for the best of the 2,000 random draws. It reaches the random-sampling floor in about a fifteenth of the evaluations and does not pass it.

## Reproduce

```
uv run python -m experiments.sat_population.run prepare --source <satisfying-codes>/experiments/surface
uv run python -m experiments.sat_population.run run --workers 4
uv run python -m experiments.sat_population.run confirm
uv run python -m experiments.sat_population.run report
```

The write-up is
[`experiments/sat_population/REPORT.md`](https://github.com/unitaryfoundation/autoqec-sim/blob/main/experiments/sat_population/REPORT.md);
the local search needs the source repository's `qodesign` package,
`pycmsgen`, and `python-sat`.
