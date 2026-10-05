# WSINDy + LASSO benchmark

## LASSO penalty selection

For K weak rows, normalize each nonzero library column as `Z_j=G_j/RMS(G_j)` and each response as `y_e=b_e/RMS(b_e)`. Solve `||Z theta-y_e||_2^2/(2*K) + lambda_e*||theta||_1`, without centering or an extra intercept; the constant library term is penalized.

For each trial and equation, define `lambda_max,e = max(abs(Z.T @ y_e))/K`, the smallest penalty for which the zero model is optimal. Evaluate 100 logarithmically spaced candidates from `1e-4*lambda_max,e` to `lambda_max,e`: `lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99)`, for `k=0,...,99`. Coupled equations have separate lambda_max values and use the same candidate index k.

Choose the candidate that minimizes the existing WSINDy score: matrix 2-norm prediction difference from full OLS, divided by the full OLS prediction norm, plus the fraction of nonzero coefficients. Exact ties select the smallest lambda. Selection uses the observed weak system G/b; it uses neither the true equation nor cross-validation.

The `lambda median` column gives the median selected lambda over the 100 trials, separately for each LHS equation. Lambda multiplies the L1 penalty on RMS-normalized coefficients; physical coefficients are restored afterward. Retain the penalized coefficients, including shrinkage, with no OLS support refit or coefficient cutoff. Empty supports are allowed. CSV/JSON summaries retain the full penalty statistics; MSTLS rows have no LASSO penalty.

The scikit-learn LARS Gram path is checked against KKT conditions at every candidate. Primal active-set polishing is tried first, followed by coordinate descent and final polishing if needed; violation divided by lambda_max above `1e-7` rejects the fit. These LASSO runs are additional regression comparisons to the papers' published algorithms.

Status: complete; 3,000/3,000 completed trials.

Schedule: 100 trials per noise level at `0, 0.2, 0.5, 0.75, 1`; root seed 0. One fixed equation and trajectory per PDE; only observation noise varies. Zero-noise repeats are identical clean checks.

Original six polynomial PDE datasets and published weak supports/degrees/strides. Weak-system construction uses the `authors` state/coordinate scaling profile. Noise ratio is relative to clean-component RMS. RD uses the archived 181-term library.

TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E_inf is the maximum relative error over true coefficients; E2 is relative coefficient L2 error. Mean ± sample SD includes failed identifications. Runtime includes weak construction, normalization and the full LASSO path; loading, noise generation and reports are excluded.

## RD

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.1\,u_{xx} + 0.1\,u_{yy} - u\,v^{2} - u^{3} + v^{3} + u^{2}\,v + u \\
v_{t} &= 0.1\,v_{xx} + 0.1\,v_{yy} - u\,v^{2} - u^{3} - v^{3} - u^{2}\,v + v
\end{aligned}
$$

Data shape: `(256, 256, 201)`, components `('u', 'v')`; G shape: `(4860, 181)`. `m=(13, 13, 14)`, `s=(13, 13, 12)`, `p=(13, 13, 12)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.35 ± 1.12e-16 | 0.0153923 ± 0 | 0.0134158 ± 1.74e-18 | 15.3 | u=8.461e-05; v=8.463e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.285994 ± 0.0149 | 1 ± 0 | 0.911146 ± 0.0143 | 15.8 | u=0.008092; v=0.008095 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.17717 ± 0.0123 | 1 ± 0 | 0.950779 ± 0.024 | 16.3 | u=0.00882; v=0.008824 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.149377 ± 0.0173 | 1 ± 0 | 1.04107 ± 0.0289 | 18.5 | u=0.01035; v=0.01035 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.0933204 ± 0.0377 | 1 ± 0 | 1.02229 ± 0.0233 | 17.3 | u=0.009949; v=0.009947 |

## NS

**True equation (physical units):**

$$
\begin{aligned}
\omega_{t} &= -\left(\omega\,u\right)_{x} - \left(\omega\,v\right)_{y} + 0.01\,\omega_{xx} + 0.01\,\omega_{yy}
\end{aligned}
$$

Here $\omega$ is vorticity and $(u,v)$ is the observed incompressible velocity ($u_x+v_y=0$). Only the vorticity equation is identified; $-(\omega u)_x-(\omega v)_y=-u\omega_x-v\omega_y$. The viscosity is $\nu=0.01$.

Data shape: `(324, 149, 201)`, components `('omega', 'u', 'v')`; G shape: `(3872, 50)`. `m=(31, 31, 14)`, `s=(12, 12, 8)`, `p=(9, 9, 12)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.571429 ± 2.23e-16 | 0.606835 ± 1.12e-16 | 0.0425387 ± 6.97e-18 | 4.31 | omega=0.00602 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.436786 ± 0.0692 | 0.624728 ± 0.41 | 0.0488924 ± 0.0291 | 4.28 | omega=0.009585 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.407083 ± 0.0293 | 1 ± 0 | 0.131849 ± 0.0164 | 4.27 | omega=0.01675 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.336433 ± 0.0241 | 0.997563 ± 0.0244 | 0.177631 ± 0.0151 | 4.27 | omega=0.007246 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.33375 ± 0.00417 | 1 ± 0 | 0.298139 ± 0.00902 | 5.11 | omega=0.01265 |

## NLS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.5\,v_{xx} + u^{2}\,v + v^{3} \\
v_{t} &= -0.5\,u_{xx} - u\,v^{2} - u^{3}
\end{aligned}
$$

Data shape: `(256, 251)`, components `('u', 'v')`; G shape: `(1804, 190)`. `m=(19, 25)`, `s=(5, 5)`, `p=(11, 10)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.00084906 ± 0 | 0.000618916 ± 1.09e-19 | 0.185 | u=9.083e-05; v=9.087e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.359777 ± 0.0326 | 0.126324 ± 0.0103 | 0.0593371 ± 0.00781 | 0.199 | u=0.006564; v=0.006566 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.245019 ± 0.0222 | 0.243522 ± 0.0198 | 0.154701 ± 0.0104 | 0.221 | u=0.007937; v=0.007939 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.232413 ± 0.0186 | 0.378335 ± 0.0264 | 0.285532 ± 0.0192 | 0.231 | u=0.01147; v=0.0115 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.224861 ± 0.0213 | 0.538202 ± 0.0475 | 0.448255 ± 0.0328 | 0.24 | u=0.01389; v=0.01391 |

## KS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xx} - u_{xxxx}
\end{aligned}
$$

Data shape: `(256, 301)`, components `('u',)`; G shape: `(1806, 43)`. `m=(23, 22)`, `s=(5, 6)`, `p=(10, 10)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.272727 ± 5.58e-17 | 0.00920226 ± 0 | 0.00759012 ± 2.62e-18 | 0.0516 | u=5.794e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.285054 ± 0.0251 | 0.866745 ± 0.257 | 0.711172 ± 0.211 | 0.0537 | u=0.01267 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.262917 ± 0.0333 | 0.903213 ± 0.203 | 0.737055 ± 0.166 | 0.0478 | u=0.01213 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.26944 ± 0.0447 | 0.956166 ± 0.16 | 0.780472 ± 0.126 | 0.0472 | u=0.01495 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.289096 ± 0.0508 | 0.979625 ± 0.104 | 0.816684 ± 0.0799 | 0.0464 | u=0.02106 |

## KdV

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xxx}
\end{aligned}
$$

Data shape: `(400, 601)`, components `('u',)`; G shape: `(1443, 43)`. `m=(45, 80)`, `s=(8, 12)`, `p=(8, 7)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.0139033 ± 5.23e-18 | 0.0124488 ± 5.23e-18 | 0.109 | u=9.987e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 1.0% | 0.286667 ± 0.129 | 0.927385 ± 0.248 | 0.830413 ± 0.222 | 0.114 | u=0.007211 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.275 ± 0.0995 | 0.958036 ± 0.168 | 0.857904 ± 0.15 | 0.115 | u=0.007208 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.265833 ± 0.0498 | 0.995386 ± 0.0461 | 0.891908 ± 0.0414 | 0.117 | u=0.007904 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 0.0% | 0.2975 ± 0.0559 | 0.994591 ± 0.0541 | 0.892453 ± 0.0487 | 0.116 | u=0.02529 |

## IB

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x}
\end{aligned}
$$

Data shape: `(256, 256)`, components `('u',)`; G shape: `(784, 43)`. `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 100.0% | 1 ± 0 | 0.00183166 ± 0 | 0.00183166 ± 0 | 0.0335 | u=0.001789 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 25.0% | 0.625 ± 0.218 | 0.0778663 ± 0.0679 | 34.6959 ± 44.8 | 0.0378 | u=0.009538 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 21.0% | 0.561667 ± 0.261 | 0.262172 ± 0.278 | 40544 ± 4.03e+05 | 0.0388 | u=0.02094 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 21.0% | 0.540833 ± 0.272 | 0.349902 ± 0.268 | 61863.4 ± 6.16e+05 | 0.0397 | u=0.03954 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + LASSO | 100 | 5.0% | 0.416667 ± 0.203 | 0.526774 ± 0.286 | 77815.3 ± 7.72e+05 | 0.0411 | u=0.04464 |

