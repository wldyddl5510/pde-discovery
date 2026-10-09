# Section 6 conic estimator: oracle μ/τ sanity grid

Status: **complete**, 25/25 cases; 1 trial per PDE/noise setting, λ=1.

**Oracle diagnostic:** μ and τ minimize coefficient E2 against the known true equation. These are best values among the searched candidates, not globally optimal hyperparameters or a data-driven tuning procedure. Do not compare these scores as a fair tuned benchmark.

Variance is estimated from each observed field (also at zero injected noise). The same noise seeds, full polynomial libraries, weak kernels and Gaussian correction as the formula sanity study are used. State, coordinates, columns and response retain physical units; rescale=False and solver equilibration=False. No coefficient thresholding or refit.

## Search and numerical checks

- Coarse μ/‖y‖∞: 0 and 10^q for integer q = −8,…,2 (12 values).
- Coarse τ/‖y‖∞: 0, 10^q for integer q = −6,…,−1, 0.2, 0.4, 0.6, 0.8, 0.95 (12 values).
- Each case has 144 coarse solves, then up to 49 points near the best eligible coarse point (duplicates skipped). No refinement if the coarse search has no eligible candidate.
- Grid ratios only construct physical μ and τ; the solver arrays and objective are not normalized. The μ reference assumes a coefficient of one in the existing physical units.
- Clarabel tolerance 1e-12, maximum 200 iterations. Eligibility requires solver success plus physical residual violation ≤ 1e-6‖y‖∞ (no unit-size floor), cone feasibility, returned objective/dual gap consistency and the feasible-zero-coefficient objective bound.
- Other validation limits use 1e-6 times max(1, relevant objective or coefficient norm). Raw solver status and each rejection reason remain in the candidate JSONL.
- E2 = ‖β−β*‖₂/‖β*‖₂ uses unmodified coefficients. Exact support uses |β|>1e-7 only for its separate support mask. E2 alone selects parameters; ties use candidate order.
- Formula μδ,τδ at δ=0.05 are reference values only. τ≥‖y‖∞ makes β=t=0 globally optimal. For μ>0 even τ<‖y‖∞ permits β=0 at t=(‖y‖∞−τ)/μ; a nonzero solution is not guaranteed.
- One trial per setting is a sanity check, not an estimate of recovery probability. HKS and VBG use the repository’s declared resimulations of the consistency-paper equations.

Total candidates: 3983; eligible: 611. Cases with an eligible fit: 9/25; selected exact support: 0/9 eligible cases.

## IB

True equation: $u_t=-\frac12\partial_x(u^2)$.

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
| 0 | 2.918e+05 | 7.164e+20 | 132.7 | 1.961e+26 | 9.83e+10 | — |
| 0.2 | 2.987e+05 | 7.265e+20 | 1.969e+04 | 4.874e+27 | 2.551e+11 | — |
| 0.5 | 3.127e+05 | 7.961e+20 | 1.228e+05 | 8.049e+28 | 5.037e+11 | — |
| 0.75 | 3.412e+05 | 8.588e+20 | 2.747e+05 | 5.197e+29 | 8.166e+11 | — |
| 1 | 3.242e+05 | 1.034e+21 | 4.897e+05 | 2.069e+30 | 9.319e+11 | — |

- Noise 0, trial 0: AlmostSolved: 11, InsufficientProgress: 11, NumericalError: 122. No eligible solution; E2 is not reported.
- Noise 0.2, trial 0: AlmostSolved: 70, InsufficientProgress: 68, MaxIterations: 1, NumericalError: 5. No eligible solution; E2 is not reported.
- Noise 0.5, trial 0: AlmostSolved: 68, InsufficientProgress: 69, NumericalError: 6, Solved: 1. No eligible solution; E2 is not reported.
- Noise 0.75, trial 0: AlmostSolved: 53, InsufficientProgress: 82, NumericalError: 9. No eligible solution; E2 is not reported.
- Noise 1, trial 0: AlmostSolved: 49, InsufficientProgress: 89, NumericalError: 6. No eligible solution; E2 is not reported.

## KdV

True equation: $u_t=-\frac12\partial_x(u^2)-u_{xxx}$.

| Noise | Trial | μ selected | τ selected | E2 oracle | Exact support | Eligible / tried | Formula E2 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | 0 | 196.7 | 1 | no | 15 / 189 | 1 |
| 0.2 | 0 | 0 | 165.9 | 1 | no | 8 / 189 | 1 |
| 0.5 | 0 | — | — | — | — | 0 / 144 | 1 |
| 0.75 | 0 | 0 | 198.9 | 1 | no | 19 / 189 | 1 |
| 1 | 0 | 0.0002127 | 170.2 | 1 | no | 9 / 187 | 1 |

**Per-case diagnostics**

| Noise | ‖y‖∞ | max absolute X entry | σ̂² | μδ reference | τδ reference | Selected τ/‖y‖∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 207.1 | 1.423e+20 | 1.359e-06 | 3.296e+26 | 4.46e+07 | 0.95 |
| 0.2 | 207.4 | 1.45e+20 | 3011 | 1.753e+27 | 1.05e+08 | 0.8 |
| 0.5 | 208.5 | 1.527e+20 | 1.874e+04 | 1.064e+28 | 2.386e+08 | — |
| 0.75 | 209.4 | 1.641e+20 | 4.25e+04 | 2.097e+28 | 3.079e+08 | 0.95 |
| 1 | 212.7 | 1.804e+20 | 7.599e+04 | 3.26e+28 | 4.046e+08 | 0.8 |

- Noise 0, trial 0: AlmostSolved: 89, InsufficientProgress: 85, Solved: 15. Selected point is at an upper grid boundary; the range may limit the result.
- Noise 0.2, trial 0: AlmostSolved: 98, InsufficientProgress: 78, NumericalError: 5, Solved: 8.
- Noise 0.5, trial 0: AlmostSolved: 64, InsufficientProgress: 74, NumericalError: 6. No eligible solution; E2 is not reported.
- Noise 0.75, trial 0: AlmostSolved: 87, InsufficientProgress: 77, NumericalError: 6, Solved: 19. Selected point is at an upper grid boundary; the range may limit the result.
- Noise 1, trial 0: AlmostSolved: 96, InsufficientProgress: 75, NumericalError: 7, Solved: 9.

## KS

True equation: $u_t=-\frac12\partial_x(u^2)-u_{xx}-u_{xxxx}$.

| Noise | Trial | μ selected | τ selected | E2 oracle | Exact support | Eligible / tried | Formula E2 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0 | 697.1 | 5.682 | 1 | no | 62 / 189 | 1 |
| 0.2 | 0 | 0.07039 | 0.07039 | 0.07016 | no | 112 / 184 | 1 |
| 0.5 | 0 | 0.07406 | 0.07406 | 0.7521 | no | 134 / 184 | 1 |
| 0.75 | 0 | 0.01558 | 0.3357 | 0.4039 | no | 125 / 184 | 1 |
| 1 | 0 | 0.07866 | 0.07866 | 0.7577 | no | 127 / 184 | 1 |

**Per-case diagnostics**

| Noise | ‖y‖∞ | max absolute X entry | σ̂² | μδ reference | τδ reference | Selected τ/‖y‖∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.971 | 3696 | 1.625e-07 | 1.05e+08 | 1.452e+04 | 0.8151 |
| 0.2 | 7.039 | 3329 | 0.05015 | 4.403e+08 | 2.056e+04 | 0.01 |
| 0.5 | 7.406 | 5399 | 0.3103 | 2.177e+09 | 3.925e+04 | 0.01 |
| 0.75 | 7.233 | 8407 | 0.7019 | 7.966e+09 | 5.226e+04 | 0.04642 |
| 1 | 7.866 | 7270 | 1.245 | 1.266e+10 | 6.708e+04 | 0.01 |

- Noise 0, trial 0: AlmostPrimalInfeasible: 2, AlmostSolved: 125, Solved: 62. Selected point is at an upper grid boundary; the range may limit the result.
- Noise 0.2, trial 0: AlmostSolved: 52, PrimalInfeasible: 6, Solved: 126.
- Noise 0.5, trial 0: AlmostSolved: 28, PrimalInfeasible: 7, Solved: 149.
- Noise 0.75, trial 0: AlmostSolved: 30, PrimalInfeasible: 8, Solved: 146.
- Noise 1, trial 0: AlmostSolved: 27, PrimalInfeasible: 8, Solved: 149.

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

- Noise 0, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.
- Noise 0.2, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.
- Noise 0.5, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.
- Noise 0.75, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.
- Noise 1, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.

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

- Noise 0, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.
- Noise 0.2, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.
- Noise 0.5, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.
- Noise 0.75, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.
- Noise 1, trial 0: AlmostSolved: 144. No eligible solution; E2 is not reported.

## Reproducibility

[Protocol and hashes](results.manifest.json); [summary](results.summary.json); [preservation checks](verification.json). `cases/` contains selected coefficients, truth, correction parameters and reference results. `candidates/` retains every solve and failure. `results_source/` contains the frozen sources. To rerun frozen sources from the repo root, run `python benchmark_results/eiv/oracle_grid/results_source/run_oracle_grid.py --output-dir benchmark_results/eiv/oracle_grid_rerun`.

The repository root `results.md` and previous sanity/benchmark outputs are preserved.
