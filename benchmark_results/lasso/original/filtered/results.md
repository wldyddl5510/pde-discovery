# Filtered WSINDy + LASSO benchmark

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

Filtered WSINDy estimates component noise with unit-L2 sixth differences along time ([1,-6,15,-20,15,-6,1]/sqrt(924)), then applies a separable space-time moving average with symmetric reflection (`scipy.ndimage.uniform_filter`, origin 0). The coefficient-ratio prior is 0.01; sigma_est is the maximum component estimate. For D space-time axes, library degree p_max, and weak-test half-width m_d, w_d = max(1, floor(min(2*(binom(p_max,2)*sigma_est^2/prior)^(1/D), (2*m_d+1)/2))). All components share the same window. Both LHS and RHS use filtered states, with library nonlinearities evaluated after averaging. These are adaptations of [the consistency paper, Sections 4.2/5.4 and Appendix G](https://arxiv.org/pdf/2211.16000): the degree-6 factor 1500 is generalized, and each anisotropic axis is capped separately. The existing polynomial tests and state/coordinate scaling are retained; regression uses LASSO. Window selection uses observed data only, including at zero noise. This applies the preprocessing comparison to the original six-PDE polynomial benchmark protocol. Smoothing bias and signal contamination of the noise estimate can reduce accuracy.

Runtime includes filtering and noise estimation.

## RD

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.1\,u_{xx} + 0.1\,u_{yy} - u\,v^{2} - u^{3} + v^{3} + u^{2}\,v + u \\
v_{t} &= 0.1\,v_{xx} + 0.1\,v_{yy} - u\,v^{2} - u^{3} - v^{3} - u^{2}\,v + v
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(1, 10), (1, 10), (1, 10)]`.

Data shape: `(256, 256, 201)`, components `('u', 'v')`; G shape: `(4860, 181)`. `m=(13, 13, 14)`, `s=(13, 13, 12)`, `p=(13, 13, 12)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.35 ± 1.12e-16 | 0.0153923 ± 0 | 0.0134158 ± 1.74e-18 | 15.6 | u=8.461e-05; v=8.463e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.315114 ± 0.021 | 0.715256 ± 0.201 | 0.584312 ± 0.223 | 16.5 | u=0.004199; v=0.004201 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.290288 ± 0.0188 | 0.932402 ± 0.132 | 0.779452 ± 0.177 | 17.2 | u=0.01007; v=0.01008 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.295891 ± 0.0211 | 0.766422 ± 0.181 | 0.548757 ± 0.199 | 19.1 | u=0.007221; v=0.007219 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.296331 ± 0.0231 | 0.70345 ± 0.209 | 0.481669 ± 0.191 | 17.3 | u=0.00754; v=0.007545 |

## NS

**True equation (physical units):**

$$
\begin{aligned}
\omega_{t} &= -\left(\omega\,u\right)_{x} - \left(\omega\,v\right)_{y} + 0.01\,\omega_{xx} + 0.01\,\omega_{yy}
\end{aligned}
$$

Here $\omega$ is vorticity and $(u,v)$ is the observed incompressible velocity ($u_x+v_y=0$). Only the vorticity equation is identified; $-(\omega u)_x-(\omega v)_y=-u\omega_x-v\omega_y$. The viscosity is $\nu=0.01$.

Observed filter-side ranges (grid points, space then time): `[(1, 12), (1, 12), (1, 12)]`.

Data shape: `(324, 149, 201)`, components `('omega', 'u', 'v')`; G shape: `(3872, 50)`. `m=(31, 31, 14)`, `s=(12, 12, 8)`, `p=(9, 9, 12)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.571429 ± 2.23e-16 | 0.606835 ± 1.12e-16 | 0.0425387 ± 6.97e-18 | 4.65 | omega=0.00602 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 1 ± 0 | 0.0901335 ± 0.00484 | 4.79 | omega=0.02008 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.375 ± 0 | 1 ± 0 | 0.130315 ± 0.011 | 4.78 | omega=0.01504 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.320348 ± 0.0279 | 1 ± 0 | 0.259175 ± 0.0206 | 5 | omega=0.03418 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.231923 ± 0.00459 | 1 ± 0 | 0.404132 ± 0.0133 | 5.36 | omega=0.004805 |

## NLS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.5\,v_{xx} + u^{2}\,v + v^{3} \\
v_{t} &= -0.5\,u_{xx} - u\,v^{2} - u^{3}
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(1, 19), (1, 25)]`.

Data shape: `(256, 251)`, components `('u', 'v')`; G shape: `(1804, 190)`. `m=(19, 25)`, `s=(5, 5)`, `p=(11, 10)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.00084906 ± 0 | 0.000618916 ± 1.09e-19 | 0.107 | u=9.083e-05; v=9.087e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.274269 ± 0.0301 | 0.128724 ± 0.0133 | 0.105388 ± 0.00935 | 0.121 | u=0.007592; v=0.007592 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.120304 ± 0.0274 | 0.983704 ± 0.0607 | 1.05245 ± 0.0863 | 0.138 | u=0.01246; v=0.01247 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.131838 ± 0.0366 | 0.947129 ± 0.0988 | 0.98706 ± 0.122 | 0.147 | u=0.01362; v=0.01363 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.139004 ± 0.0349 | 0.918399 ± 0.127 | 0.950743 ± 0.125 | 0.141 | u=0.01361; v=0.0136 |

## KS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xx} - u_{xxxx}
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(1, 23), (1, 22)]`.

Data shape: `(256, 301)`, components `('u',)`; G shape: `(1806, 43)`. `m=(23, 22)`, `s=(5, 6)`, `p=(10, 10)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.272727 ± 5.58e-17 | 0.00920226 ± 0 | 0.00759012 ± 2.62e-18 | 0.0256 | u=5.794e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.391619 ± 0.0277 | 1 ± 0 | 0.977837 ± 0.00175 | 0.0299 | u=0.0306 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.00666667 ± 0.0328 | 1 ± 0 | 0.999641 ± 0.00198 | 0.0294 | u=0.05897 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.0317143 ± 0.0732 | 1.00071 ± 0.00502 | 0.998044 ± 0.00451 | 0.0288 | u=0.05766 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.0248095 ± 0.0665 | 1.00147 ± 0.00871 | 0.998598 ± 0.00387 | 0.0289 | u=0.05508 |

## KdV

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xxx}
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(1, 45), (1, 80)]`.

Data shape: `(400, 601)`, components `('u',)`; G shape: `(1443, 43)`. `m=(45, 80)`, `s=(8, 12)`, `p=(8, 7)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.0139033 ± 5.23e-18 | 0.0124488 ± 5.23e-18 | 0.0642 | u=9.987e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 1 ± 0 | 385.212 ± 0.817 | 0.0681 | u=0.03004 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 1 ± 0 | 384.979 ± 2.34 | 0.0742 | u=0.03003 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 1 ± 0 | 384.765 ± 3.65 | 0.0723 | u=0.03002 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 1 ± 0 | 383.727 ± 4.36 | 0.0743 | u=0.02999 |

## IB

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x}
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(60, 60), (60, 60)]`.

Data shape: `(256, 256)`, components `('u',)`; G shape: `(784, 43)`. `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.890185 ± 1.12e-16 | 870.704 ± 0 | 0.0255 | u=0.01517 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.891891 ± 0.00943 | 870.693 ± 10.7 | 0.0239 | u=0.01663 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.491667 ± 0.0365 | 0.89946 ± 0.0208 | 871.311 ± 26.3 | 0.0237 | u=0.02098 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.474167 ± 0.0623 | 0.898433 ± 0.033 | 232531 ± 2.32e+06 | 0.0247 | u=0.02407 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.446667 ± 0.108 | 0.905774 ± 0.0416 | 1.32148e+06 ± 5.85e+06 | 0.0281 | u=0.02392 |

