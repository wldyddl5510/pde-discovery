# Debiased WSINDy (time2) + LASSO: VBG

## LASSO penalty selection

For K weak rows, normalize each nonzero library column as `Z_j=G_j/RMS(G_j)` and each response as `y_e=b_e/RMS(b_e)`. Solve `||Z theta-y_e||_2^2/(2*K) + lambda_e*||theta||_1`, without centering or an extra intercept; the constant library term is penalized.

For each trial and equation, define `lambda_max,e = max(abs(Z.T @ y_e))/K`, the smallest penalty for which the zero model is optimal. Evaluate 100 logarithmically spaced candidates from `1e-4*lambda_max,e` to `lambda_max,e`: `lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99)`, for `k=0,...,99`. Coupled equations have separate lambda_max values and use the same candidate index k.

Choose the candidate that minimizes the existing WSINDy score: matrix 2-norm prediction difference from full OLS, divided by the full OLS prediction norm, plus the fraction of nonzero coefficients. Exact ties select the smallest lambda. Selection uses the observed weak system G/b; it uses neither the true equation nor cross-validation.

The `lambda median` column gives the median selected lambda over the 100 trials, separately for each LHS equation. Lambda multiplies the L1 penalty on RMS-normalized coefficients; physical coefficients are restored afterward. Retain the penalized coefficients, including shrinkage, with no OLS support refit or coefficient cutoff. Empty supports are allowed. CSV/JSON summaries retain the full penalty statistics; MSTLS rows have no LASSO penalty.

The scikit-learn LARS Gram path is checked against KKT conditions at every candidate. Primal active-set polishing is tried first, followed by coordinate descent and final polishing if needed; violation divided by lambda_max above `1e-7` rejects the fit. These LASSO runs are additional regression comparisons to the papers' published algorithms.

Status: complete; 500/500 fits. Mean ± sample SD over 100 trials per noise level.

Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each.

Timings include fold-specific noise estimation, pilot, weak construction and the entire regression path. Shared weak construction time is attributed to each fit; loading, noise generation and diagnostics are excluded.

## VBG

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.01\,u_{xx} - 0.5\,\left(u^{2}\right)_{x} - u^{3} + 2\,u^{2} + 1
\end{aligned}
$$

Observation shape `(512, 451)`; library 43 terms; weak half-widths `(64, 56)`, strides `(12, 11)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.222222 ± 2.79e-17 | 1 ± 0 | 0.936931 ± 1.12e-16 | 0.0814 | u=0.006524 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.229111 ± 0.0315 | 1 ± 0 | 0.92576 ± 0.00641 | 0.0915 | u=0.01498 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.315495 ± 0.0249 | 1 ± 0 | 0.923946 ± 0.00729 | 0.0938 | u=0.02122 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.295434 ± 0.0289 | 1 ± 0 | 0.933211 ± 0.015 | 0.0905 | u=0.02093 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.299482 ± 0.0377 | 1 ± 0 | 0.928707 ± 0.0149 | 0.0894 | u=0.02878 |

