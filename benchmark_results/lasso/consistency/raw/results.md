# WSINDy + LASSO benchmark

## LASSO penalty selection

For K weak rows, normalize each nonzero library column as `Z_j=G_j/RMS(G_j)` and each response as `y_e=b_e/RMS(b_e)`. Solve `||Z theta-y_e||_2^2/(2*K) + lambda_e*||theta||_1`, without centering or an extra intercept; the constant library term is penalized.

For each trial and equation, define `lambda_max,e = max(abs(Z.T @ y_e))/K`, the smallest penalty for which the zero model is optimal. Evaluate 100 logarithmically spaced candidates from `1e-4*lambda_max,e` to `lambda_max,e`: `lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99)`, for `k=0,...,99`. Coupled equations have separate lambda_max values and use the same candidate index k.

Choose the candidate that minimizes the existing WSINDy score: matrix 2-norm prediction difference from full OLS, divided by the full OLS prediction norm, plus the fraction of nonzero coefficients. Exact ties select the smallest lambda. Selection uses the observed weak system G/b; it uses neither the true equation nor cross-validation.

The `lambda median` column gives the median selected lambda over the 100 trials, separately for each LHS equation. Lambda multiplies the L1 penalty on RMS-normalized coefficients; physical coefficients are restored afterward. Retain the penalized coefficients, including shrinkage, with no OLS support refit or coefficient cutoff. Empty supports are allowed. CSV/JSON summaries retain the full penalty statistics; MSTLS rows have no LASSO penalty.

The scikit-learn LARS Gram path is checked against KKT conditions at every candidate. Primal active-set polishing is tried first, followed by coordinate descent and final polishing if needed; violation divided by lambda_max above `1e-7` rejects the fit. These LASSO runs are additional regression comparisons to the papers' published algorithms.

Status: complete; 1,000/1,000 completed trials.

Schedule: 100 trials per noise level at `0, 0.2, 0.5, 0.75, 1`; root seed 0. One fixed equation and trajectory per PDE; only observation noise varies. Zero-noise repeats are identical clean checks.

HKS/VBG are declared resimulations using the consistency-paper equations; IC/BC/grids/strides are declared here. Weak systems use the C-infinity bump with physical coordinates and no rescaling. Noise ratio is relative to centered clean sample std.

TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E_inf is the maximum relative error over true coefficients; E2 is relative coefficient L2 error. Mean ± sample SD includes failed identifications. Runtime includes weak construction, normalization and the full LASSO path; loading, noise generation and reports are excluded.

## HKS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= u_{xxxx} + 0.75\,u_{xxxxxx} - 0.5\,\left(u^{2}\right)_{x} + 0.1\,\left(u^{2}\right)_{xxx}
\end{aligned}
$$

Declared IC: `cos(x/16)*(1+sin(x/16))`; boundary: periodic; x in `[0.0, 100.53096491487338]`, t in `[0.0, 82.0]`. Fine simulation: `[1024, 1025]`; observation subsampling: `[4, 4]`.

Data shape: `(256, 257)`, components `('u',)`; G shape: `(900, 73)`. `m=(26, 26)`, `s=(7, 7)`, `p=(0, 0)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.375 ± 0 | 1 ± 0 | 0.756139 ± 1.12e-16 | 0.0495 | u=0.006203 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.345298 ± 0.0456 | 1 ± 0 | 0.869025 ± 0.0339 | 0.0488 | u=0.02716 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.342877 ± 0.0493 | 1 ± 0 | 0.870665 ± 0.0232 | 0.0535 | u=0.03499 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.32475 ± 0.0595 | 1 ± 0 | 0.86543 ± 0.0319 | 0.0596 | u=0.04305 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.313516 ± 0.0651 | 1 ± 0 | 0.847748 ± 0.0341 | 0.0581 | u=0.04773 |

## VBG

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.01\,u_{xx} - 0.5\,\left(u^{2}\right)_{x} - u^{3} + 2\,u^{2} + 1
\end{aligned}
$$

Declared IC: `2*sin(pi*x)`; boundary: periodic; x in `[-1.0, 1.0]`, t in `[0.0, 1.5]`. Fine simulation: `[2048, 1801]`; observation subsampling: `[4, 4]`.

Data shape: `(512, 451)`, components `('u',)`; G shape: `(992, 43)`. `m=(64, 56)`, `s=(12, 11)`, `p=(0, 0)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.222222 ± 2.79e-17 | 1 ± 0 | 0.936933 ± 3.35e-16 | 0.102 | u=0.006524 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.222222 ± 2.79e-17 | 1 ± 0 | 0.93563 ± 0.00198 | 0.106 | u=0.008614 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.212495 ± 0.0131 | 1 ± 0 | 0.932625 ± 0.00891 | 0.115 | u=0.01643 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.206955 ± 0.0186 | 1 ± 0 | 0.933057 ± 0.0142 | 0.11 | u=0.02144 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.225084 ± 0.0342 | 1 ± 0 | 0.925098 ± 0.0241 | 0.116 | u=0.03363 |

