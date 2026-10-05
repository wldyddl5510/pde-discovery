# Debiased WSINDy (time2) + MSTLS: IB

## LASSO penalty selection

For K weak rows, normalize each nonzero library column as `Z_j=G_j/RMS(G_j)` and each response as `y_e=b_e/RMS(b_e)`. Solve `||Z theta-y_e||_2^2/(2*K) + lambda_e*||theta||_1`, without centering or an extra intercept; the constant library term is penalized.

For each trial and equation, define `lambda_max,e = max(abs(Z.T @ y_e))/K`, the smallest penalty for which the zero model is optimal. Evaluate 100 logarithmically spaced candidates from `1e-4*lambda_max,e` to `lambda_max,e`: `lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99)`, for `k=0,...,99`. Coupled equations have separate lambda_max values and use the same candidate index k.

Choose the candidate that minimizes the existing WSINDy score: matrix 2-norm prediction difference from full OLS, divided by the full OLS prediction norm, plus the fraction of nonzero coefficients. Exact ties select the smallest lambda. Selection uses the observed weak system G/b; it uses neither the true equation nor cross-validation.

The `lambda median` column gives the median selected lambda over the 100 trials, separately for each LHS equation. Lambda multiplies the L1 penalty on RMS-normalized coefficients; physical coefficients are restored afterward. Retain the penalized coefficients, including shrinkage, with no OLS support refit or coefficient cutoff. Empty supports are allowed. CSV/JSON summaries retain the full penalty statistics; MSTLS rows have no LASSO penalty.

The scikit-learn LARS Gram path is checked against KKT conditions at every candidate. Primal active-set polishing is tried first, followed by coordinate descent and final polishing if needed; violation divided by lambda_max above `1e-7` rejects the fit. These LASSO runs are additional regression comparisons to the papers' published algorithms.

Status: complete; 500/500 fits. Noise ratios: 0, 0.2, 0.5, 0.75, 1; 100 trials each.

Section 5 first-order polynomial expansion `(1-p)*pilot^p + p*pilot^(p-1)*U`; the LHS and linear columns use raw U. Pilot: other time-index parity only (`time2`), box moving average, periodic spatial extension and reflected time extension. Window: debiasing_study.py data-driven volume rule, prior 0.01, rounded up to odd and at least 3; no window selection using truth. IB selects 61 x 61 at every tested noise level. Author-code weak supports, state/coordinate scaling and physical coefficient restoration are retained. Both regressions use the same pilot/system and the existing penalty selection. All 100 trials at noise ratios 0, 0.2, 0.5, 0.75, 1 use the original benchmark's observation seeds.

Timings include noise estimation, pilot, weak system and the selected regression's entire path; data/noise generation and diagnostics are excluded. The weak system is built once per observation and its measured time is attributed to each fit.

## IB

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x}
\end{aligned}
$$

Observation shape `(256, 256)`; library 43 terms; weak half-widths `(60, 60)`, strides `(5, 5)`; pilot window `(61, 61)`; split `time2`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.166667 ± 5.58e-17 | 0.0122654 ± 0 | 27.4227 ± 7.14e-15 | 0.022 | — |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 8.0% | 0.334262 ± 0.212 | 0.0584391 ± 0.0972 | 26.0369 ± 100 | 0.0225 | — |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 25.0% | 0.477833 ± 0.312 | 0.0510718 ± 0.0322 | 15.5908 ± 25 | 0.0222 | — |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 22.0% | 0.470667 ± 0.297 | 0.0540189 ± 0.0661 | 21.8176 ± 65.5 | 0.0222 | — |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 24.0% | 0.4955 ± 0.297 | 0.0596323 ± 0.0743 | 22.9425 ± 68.3 | 0.0224 | — |

