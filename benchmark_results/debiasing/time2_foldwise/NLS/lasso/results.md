# Debiased WSINDy (time2) + LASSO: NLS

## LASSO penalty selection

For K weak rows, normalize each nonzero library column as `Z_j=G_j/RMS(G_j)` and each response as `y_e=b_e/RMS(b_e)`. Solve `||Z theta-y_e||_2^2/(2*K) + lambda_e*||theta||_1`, without centering or an extra intercept; the constant library term is penalized.

For each trial and equation, define `lambda_max,e = max(abs(Z.T @ y_e))/K`, the smallest penalty for which the zero model is optimal. Evaluate 100 logarithmically spaced candidates from `1e-4*lambda_max,e` to `lambda_max,e`: `lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99)`, for `k=0,...,99`. Coupled equations have separate lambda_max values and use the same candidate index k.

Choose the candidate that minimizes the existing WSINDy score: matrix 2-norm prediction difference from full OLS, divided by the full OLS prediction norm, plus the fraction of nonzero coefficients. Exact ties select the smallest lambda. Selection uses the observed weak system G/b; it uses neither the true equation nor cross-validation.

The `lambda median` column gives the median selected lambda over the 100 trials, separately for each LHS equation. Lambda multiplies the L1 penalty on RMS-normalized coefficients; physical coefficients are restored afterward. Retain the penalized coefficients, including shrinkage, with no OLS support refit or coefficient cutoff. Empty supports are allowed. CSV/JSON summaries retain the full penalty statistics; MSTLS rows have no LASSO penalty.

The scikit-learn LARS Gram path is checked against KKT conditions at every candidate. Primal active-set polishing is tried first, followed by coordinate descent and final polishing if needed; violation divided by lambda_max above `1e-7` rejects the fit. These LASSO runs are additional regression comparisons to the papers' published algorithms.

Status: complete; 500/500 fits. Mean ± sample SD over 100 trials per noise level.

Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each.

Timings include fold-specific noise estimation, pilot, weak construction and the entire regression path. Shared weak construction time is attributed to each fit; loading, noise generation and diagnostics are excluded.

## NLS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.5\,v_{xx} + u^{2}\,v + v^{3} \\
v_{t} &= -0.5\,u_{xx} - u\,v^{2} - u^{3}
\end{aligned}
$$

Observation shape `(256, 251)`; library 190 terms; weak half-widths `(19, 25)`, strides `(5, 5)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.6 ± 0 | 0.00103542 ± 0 | 0.000439056 ± 1.09e-19 | 0.123 | u=0.0001094; v=0.0001094 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.334231 ± 0.0355 | 0.126382 ± 0.0119 | 0.0545315 ± 0.0109 | 0.139 | u=0.006564; v=0.006564 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.174329 ± 0.0125 | 0.485489 ± 0.0895 | 0.377117 ± 0.0548 | 0.164 | u=0.007982; v=0.007991 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.17681 ± 0.0156 | 0.483523 ± 0.0589 | 0.318389 ± 0.0617 | 0.177 | u=0.01151; v=0.01153 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Debiased WSINDy + LASSO | 100 | 0.0% | 0.179367 ± 0.0243 | 0.595747 ± 0.0504 | 0.318989 ± 0.06 | 0.188 | u=0.01389; v=0.0139 |

