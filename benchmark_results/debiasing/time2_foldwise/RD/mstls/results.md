# Debiased WSINDy (time2) + MSTLS: RD

## LASSO penalty selection

For K weak rows, normalize each nonzero library column as `Z_j=G_j/RMS(G_j)` and each response as `y_e=b_e/RMS(b_e)`. Solve `||Z theta-y_e||_2^2/(2*K) + lambda_e*||theta||_1`, without centering or an extra intercept; the constant library term is penalized.

For each trial and equation, define `lambda_max,e = max(abs(Z.T @ y_e))/K`, the smallest penalty for which the zero model is optimal. Evaluate 100 logarithmically spaced candidates from `1e-4*lambda_max,e` to `lambda_max,e`: `lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99)`, for `k=0,...,99`. Coupled equations have separate lambda_max values and use the same candidate index k.

Choose the candidate that minimizes the existing WSINDy score: matrix 2-norm prediction difference from full OLS, divided by the full OLS prediction norm, plus the fraction of nonzero coefficients. Exact ties select the smallest lambda. Selection uses the observed weak system G/b; it uses neither the true equation nor cross-validation.

The `lambda median` column gives the median selected lambda over the 100 trials, separately for each LHS equation. Lambda multiplies the L1 penalty on RMS-normalized coefficients; physical coefficients are restored afterward. Retain the penalized coefficients, including shrinkage, with no OLS support refit or coefficient cutoff. Empty supports are allowed. CSV/JSON summaries retain the full penalty statistics; MSTLS rows have no LASSO penalty.

The scikit-learn LARS Gram path is checked against KKT conditions at every candidate. Primal active-set polishing is tried first, followed by coordinate descent and final polishing if needed; violation divided by lambda_max above `1e-7` rejects the fit. These LASSO runs are additional regression comparisons to the papers' published algorithms.

Status: complete; 500/500 fits. Mean ± sample SD over 100 trials per noise level.

Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each.

Timings include fold-specific noise estimation, pilot, weak construction and the entire regression path. Shared weak construction time is attributed to each fit; loading, noise generation and diagnostics are excluded.

## RD

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.1\,u_{xx} + 0.1\,u_{yy} - u\,v^{2} - u^{3} + v^{3} + u^{2}\,v + u \\
v_{t} &= 0.1\,v_{xx} + 0.1\,v_{yy} - u\,v^{2} - u^{3} - v^{3} - u^{2}\,v + v
\end{aligned}
$$

Observation shape `(256, 256, 201)`; library 181 terms; weak half-widths `(13, 13, 14)`, strides `(13, 13, 12)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.000441219 ± 0 | 0.000251307 ± 0 | 23.2 | — |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0202617 ± 0.004 | 0.0119097 ± 0.00217 | 24.5 | — |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0899071 ± 0.012 | 0.0583135 ± 0.00548 | 24.3 | — |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 60.0% | 0.968667 ± 0.0413 | 0.201127 ± 0.038 | 0.148 ± 0.0446 | 24.4 | — |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy (time2) + MSTLS | 100 | 24.0% | 0.927128 ± 0.0671 | 0.383092 ± 0.143 | 0.319902 ± 0.115 | 24.6 | — |

