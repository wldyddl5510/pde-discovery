# Section 6 conic estimator: oracle μ/τ sanity grid

Status: **complete**, 25/25 cases; 1 trial per PDE/noise setting, λ=1.

**Oracle diagnostic:** μ and τ minimize coefficient E2 against the known true equation. These are best values among the searched candidates, not globally optimal hyperparameters or a data-driven tuning procedure. Do not compare these scores as a fair tuned benchmark.

Variance is estimated from each observed field (also at zero injected noise). The same noise seeds, full polynomial libraries, weak kernels and Gaussian correction as the formula sanity study are used. State, coordinates, columns and response retain physical units; rescale=False and solver equilibration=False. No coefficient thresholding or refit.

## Search and numerical checks

- Coarse μ/‖y‖∞: 0 and 10^q for integer q = −8,…,2 (12 values).
- Coarse τ/‖y‖∞: 0, 10^q for integer q = −6,…,−1, 0.2, 0.4, 0.6, 0.8, 0.95 (12 values).
- Each case has 144 coarse solves, then up to 49 points near the best eligible coarse point (duplicates skipped). No refinement if the coarse search has no eligible candidate.
- Grid ratios only construct physical μ and τ; the solver arrays and objective are not normalized. The μ reference assumes a coefficient of one in the existing physical units.
- Clarabel tolerance 1e-08, maximum 200 iterations. Eligibility requires solver success plus physical residual violation ≤ 1e-6‖y‖∞ (no unit-size floor), cone feasibility, returned objective/dual gap consistency and the feasible-zero-coefficient objective bound.
- Other validation limits use 1e-6 times max(1, relevant objective or coefficient norm). Raw solver status and each rejection reason remain in the candidate JSONL.
- E2 = ‖β−β*‖₂/‖β*‖₂ uses unmodified coefficients. Exact support uses |β|>1e-7 only for its separate support mask. E2 alone selects parameters; ties use candidate order.
- Formula μδ,τδ at δ=0.05 are reference values only. τ≥‖y‖∞ makes β=t=0 globally optimal. For μ>0 even τ<‖y‖∞ permits β=0 at t=(‖y‖∞−τ)/μ; a nonzero solution is not guaranteed.
- A positive μ is always mathematically feasible; any PrimalInfeasible status there is a numerical failure. At μ=0 an infeasibility status may reflect genuine infeasibility.
- Physical-unit L1/L2 penalties depend on dictionary units: large high-degree columns can fit a response with tiny coefficients. An eligible fit with E2≈1 and empty support can contain tiny nonzero coefficients; it is different from a failed solve (shown as —).
- One trial per setting is a sanity check, not an estimate of recovery probability. HKS and VBG use the repository’s declared resimulations of the consistency-paper equations.

Total candidates: 4200; eligible: 1496. Cases with an eligible fit: 14/25; selected exact support: 0/14 eligible cases.

## IB

True equation: $u_t=-\frac12\partial_x(u^2)$.

| Noise | Trial | μ selected | τ selected | E2 oracle | Exact support | Eligible / tried | Formula E2 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | — | — | — | — | 0 / 144 | 1 |
| 0.2 | 0 | 29.87 | 2.838e+05 | 1 | no | 102 / 187 | 1 |
| 0.5 | 0 | 1452 | 3.94e+04 | 1 | no | 68 / 187 | 1 |
| 0.75 | 0 | 3.412e+05 | 3.242e+05 | 1 | no | 89 / 187 | 1 |
| 1 | 0 | 0.003242 | 2.593e+05 | 1 | no | 82 / 189 | 1 |

**Per-case diagnostics**

| Noise | ‖y‖∞ | max absolute X entry | σ̂² | μδ reference | τδ reference | Selected τ/‖y‖∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.918e+05 | 7.164e+20 | 132.7 | 1.961e+26 | 9.83e+10 | — |
| 0.2 | 2.987e+05 | 7.265e+20 | 1.969e+04 | 4.874e+27 | 2.551e+11 | 0.95 |
| 0.5 | 3.127e+05 | 7.961e+20 | 1.228e+05 | 8.049e+28 | 5.037e+11 | 0.126 |
| 0.75 | 3.412e+05 | 8.588e+20 | 2.747e+05 | 5.197e+29 | 8.166e+11 | 0.95 |
| 1 | 3.242e+05 | 1.034e+21 | 4.897e+05 | 2.069e+30 | 9.319e+11 | 0.8 |

- Noise 0, trial 0: AlmostSolved: 9, InsufficientProgress: 13, NumericalError: 122. No eligible solution; E2 is not reported.
- Noise 0.2, trial 0: AlmostSolved: 10, InsufficientProgress: 69, MaxIterations: 1, NumericalError: 5, Solved: 102. Selected point is at an upper grid boundary; the range may limit the result.
- Noise 0.5, trial 0: AlmostSolved: 50, InsufficientProgress: 63, NumericalError: 6, Solved: 68.
- Noise 0.75, trial 0: AlmostSolved: 7, InsufficientProgress: 85, NumericalError: 6, Solved: 89. Selected point is at an upper grid boundary; the range may limit the result.
- Noise 1, trial 0: AlmostSolved: 12, InsufficientProgress: 87, NumericalError: 8, Solved: 82.

## KdV

True equation: $u_t=-\frac12\partial_x(u^2)-u_{xxx}$.

| Noise | Trial | μ selected | τ selected | E2 oracle | Exact support | Eligible / tried | Formula E2 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | 44.61 | 175.4 | 1 | no | 81 / 187 | 1 |
| 0.2 | 0 | 20.74 | 169.1 | 1 | no | 84 / 187 | 1 |
| 0.5 | 0 | 20.85 | 166.8 | 1 | no | 76 / 187 | 1 |
| 0.75 | 0 | 0.0451 | 167.5 | 1 | no | 81 / 187 | 1 |
| 1 | 0 | 0.04582 | 196.4 | 1 | no | 75 / 187 | 1 |

**Per-case diagnostics**

| Noise | ‖y‖∞ | max absolute X entry | σ̂² | μδ reference | τδ reference | Selected τ/‖y‖∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 207.1 | 1.423e+20 | 1.359e-06 | 3.296e+26 | 4.46e+07 | 0.8472 |
| 0.2 | 207.4 | 1.45e+20 | 3011 | 1.753e+27 | 1.05e+08 | 0.8151 |
| 0.5 | 208.5 | 1.527e+20 | 1.874e+04 | 1.064e+28 | 2.386e+08 | 0.8 |
| 0.75 | 209.4 | 1.641e+20 | 4.25e+04 | 2.097e+28 | 3.079e+08 | 0.8 |
| 1 | 212.7 | 1.804e+20 | 7.599e+04 | 3.26e+28 | 4.046e+08 | 0.9232 |

- Noise 0, trial 0: AlmostSolved: 22, InsufficientProgress: 84, Solved: 81.
- Noise 0.2, trial 0: AlmostSolved: 21, InsufficientProgress: 77, NumericalError: 5, Solved: 84.
- Noise 0.5, trial 0: AlmostSolved: 31, InsufficientProgress: 74, NumericalError: 6, Solved: 76.
- Noise 0.75, trial 0: AlmostSolved: 23, InsufficientProgress: 77, NumericalError: 6, Solved: 81.
- Noise 1, trial 0: AlmostSolved: 30, InsufficientProgress: 75, NumericalError: 7, Solved: 75.

## KS

True equation: $u_t=-\frac12\partial_x(u^2)-u_{xx}-u_{xxxx}$.

| Noise | Trial | μ selected | τ selected | E2 oracle | Exact support | Eligible / tried | Formula E2 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | 6.971e-11 | 3.236e-05 | 0.0001141 | no | 157 / 187 | 1 |
| 0.2 | 0 | 0.07039 | 0.3267 | 0.07016 | no | 158 / 184 | 1 |
| 0.5 | 0 | 0 | 1.481 | 0.7386 | no | 143 / 189 | 1 |
| 0.75 | 0 | 0.01558 | 1.447 | 0.4039 | no | 152 / 187 | 1 |
| 1 | 0 | 0.07866 | 0.07866 | 0.7577 | no | 148 / 184 | 1 |

**Per-case diagnostics**

| Noise | ‖y‖∞ | max absolute X entry | σ̂² | μδ reference | τδ reference | Selected τ/‖y‖∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.971 | 3696 | 1.625e-07 | 1.05e+08 | 1.452e+04 | 4.642e-06 |
| 0.2 | 7.039 | 3329 | 0.05015 | 4.403e+08 | 2.056e+04 | 0.04642 |
| 0.5 | 7.406 | 5399 | 0.3103 | 2.177e+09 | 3.925e+04 | 0.2 |
| 0.75 | 7.233 | 8407 | 0.7019 | 7.966e+09 | 5.226e+04 | 0.2 |
| 1 | 7.866 | 7270 | 1.245 | 1.266e+10 | 6.708e+04 | 0.01 |

- Noise 0, trial 0: AlmostPrimalInfeasible: 1, AlmostSolved: 27, PrimalInfeasible: 2, Solved: 157.
- Noise 0.2, trial 0: PrimalInfeasible: 6, Solved: 178.
- Noise 0.5, trial 0: PrimalInfeasible: 12, Solved: 177.
- Noise 0.75, trial 0: PrimalInfeasible: 8, Solved: 179.
- Noise 1, trial 0: PrimalInfeasible: 8, Solved: 176.

## HKS

True equation: $u_t=u_{xxxx}+0.75u_{xxxxxx}-0.5\partial_x(u^2)+0.1\partial_x^3(u^2)$.

| Noise | Trial | μ selected | τ selected | E2 oracle | Exact support | Eligible / tried | Formula E2 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | — | — | — | — | 0 / 144 | 1 |
| 0.2 | 0 | — | — | — | — | 0 / 144 | 1 |
| 0.5 | 0 | — | — | — | — | 0 / 144 | 1 |
| 0.75 | 0 | — | — | — | — | 0 / 144 | 1 |
| 1 | 0 | — | — | — | — | 0 / 144 | 1 |

**Per-case diagnostics**

| Noise | ‖y‖∞ | max absolute X entry | σ̂² | μδ reference | τδ reference | Selected τ/‖y‖∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.267e-08 | 0.001106 | 1.405e-10 | 35.57 | 0.0001609 | — |
| 0.2 | 6.327e-08 | 0.001126 | 0.03771 | 104.2 | 0.0001958 | — |
| 0.5 | 6.708e-08 | 0.001239 | 0.2336 | 719.5 | 0.0004249 | — |
| 0.75 | 6.146e-08 | 0.002336 | 0.5405 | 2586 | 0.0005589 | — |
| 1 | 6.196e-08 | 0.002969 | 0.9552 | 7662 | 0.0007133 | — |

- Noise 0, trial 0: AlmostSolved: 132, Solved: 12. No eligible solution; E2 is not reported.
- Noise 0.2, trial 0: AlmostSolved: 132, Solved: 12. No eligible solution; E2 is not reported.
- Noise 0.5, trial 0: AlmostSolved: 132, Solved: 12. No eligible solution; E2 is not reported.
- Noise 0.75, trial 0: AlmostSolved: 132, Solved: 12. No eligible solution; E2 is not reported.
- Noise 1, trial 0: AlmostSolved: 132, Solved: 12. No eligible solution; E2 is not reported.

## VBG

True equation: $u_t=0.01u_{xx}-0.5\partial_x(u^2)-u^3+2u^2+1$.

| Noise | Trial | μ selected | τ selected | E2 oracle | Exact support | Eligible / tried | Formula E2 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | — | — | — | — | 0 / 144 | 1 |
| 0.2 | 0 | — | — | — | — | 0 / 144 | 1 |
| 0.5 | 0 | — | — | — | — | 0 / 144 | 1 |
| 0.75 | 0 | — | — | — | — | 0 / 144 | 1 |
| 1 | 0 | — | — | — | — | 0 / 144 | 1 |

**Per-case diagnostics**

| Noise | ‖y‖∞ | max absolute X entry | σ̂² | μδ reference | τδ reference | Selected τ/‖y‖∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.081e-09 | 0.4683 | 3.644e-11 | 5622 | 6.977e-07 | — |
| 0.2 | 3.073e-09 | 0.5164 | 0.03664 | 8.222e+04 | 1.996e-06 | — |
| 0.5 | 3.137e-09 | 0.9481 | 0.2288 | 1.15e+06 | 4.84e-06 | — |
| 0.75 | 3.06e-09 | 2.181 | 0.5146 | 4.512e+06 | 6.521e-06 | — |
| 1 | 3.105e-09 | 3.402 | 0.9143 | 1.901e+07 | 9.5e-06 | — |

- Noise 0, trial 0: AlmostSolved: 108, Solved: 36. No eligible solution; E2 is not reported.
- Noise 0.2, trial 0: AlmostSolved: 108, Solved: 36. No eligible solution; E2 is not reported.
- Noise 0.5, trial 0: AlmostSolved: 108, Solved: 36. No eligible solution; E2 is not reported.
- Noise 0.75, trial 0: AlmostSolved: 108, Solved: 36. No eligible solution; E2 is not reported.
- Noise 1, trial 0: AlmostSolved: 108, Solved: 36. No eligible solution; E2 is not reported.

## Reproducibility

[Protocol and hashes](results.manifest.json); [summary](results.summary.json); [preservation checks](verification.json). `cases/` contains selected coefficients, truth, correction parameters and reference results. `candidates/` retains every solve and failure. `results_source/` contains the frozen sources. To rerun frozen sources from the repo root, run `python benchmark_results/eiv/oracle_grid_default_tolerance/results_source/run_oracle_grid.py --output-dir benchmark_results/eiv/oracle_grid_rerun`.

The repository root `results.md` and previous sanity/benchmark outputs are preserved.
