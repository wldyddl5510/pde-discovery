# WSINDy + LASSO benchmark

Status: in progress; 30/3,500 completed trials.

Schedule: 100 trials per noise level at `0, 0.2, 0.5, 0.75, 1`; root seed 0. One fixed equation and trajectory per PDE; only observation noise varies. Zero-noise repeats are identical clean checks.

Original seven PDE datasets and published weak supports/degrees/strides. Weak-system construction uses the `authors` state/coordinate scaling profile. Noise ratio is relative to clean-component RMS. RD uses the archived 181-term library.

LASSO replaces sequential hard thresholding. For K weak rows, normalize each nonzero column Z_j=G_j/RMS(G_j) and each response y_e=b_e/RMS(b_e), without centering or an extra intercept (the constant library term is penalized). Solve `||Z theta-y_e||_2^2/(2*K)+alpha_e*||theta||_1`, with `alpha_e=ratio*max(abs(Z.T@y_e))/K` at 100 ratios `logspace(-4,0,100)`. Coupled equations use separate alpha_max values and one selected ratio. Select using the existing WSINDy loss: matrix-2-norm prediction difference from full OLS / OLS prediction norm + fraction of nonzero coefficients; smallest ratio breaks ties. Retain the penalized coefficients, including shrinkage, and restore physical units; there is no OLS support refit or coefficient cutoff. This is a column-normalized L1 penalty, equivalently a weighted penalty in physical coefficient units. Empty supports are allowed. The scikit-learn LARS Gram path is checked against KKT conditions at every candidate; coordinate descent and primal active-set polishing are used if needed, and violation / alpha_max above 1e-7 rejects the fit. Lambda selection uses G and b only. These are additional comparisons, not the papers' published sparse-regression algorithms.

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

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 30 | 0.0% | 0.35 ± 1.13e-16 | 0.0153923 ± 0 | 0.0134158 ± 5.29e-18 | 22.4 |

## NS

**True equation (physical units):**

$$
\begin{aligned}
\omega_{t} &= -\left(\omega\,u\right)_{x} - \left(\omega\,v\right)_{y} + 0.01\,\omega_{xx} + 0.01\,\omega_{yy}
\end{aligned}
$$

Here $\omega$ is vorticity and $(u,v)$ is the observed incompressible velocity ($u_x+v_y=0$). Only the vorticity equation is identified; $-(\omega u)_x-(\omega v)_y=-u\omega_x-v\omega_y$. The viscosity is $\nu=0.01$.

Data shape: `(324, 149, 201)`, components `('omega', 'u', 'v')`; G shape: `pending`. `m=(31, 31, 14)`, `s=(12, 12, 8)`, `p=(9, 9, 12)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## SG

**True equation (physical units):**

$$
\begin{aligned}
u_{tt} &= u_{xx} + u_{yy} - \sin\left(u\right)
\end{aligned}
$$

Data shape: `(129, 403, 205)`, components `('u',)`; G shape: `pending`. `m=(40, 40, 25)`, `s=(5, 5, 8)`, `p=(8, 8, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## NLS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.5\,v_{xx} + u^{2}\,v + v^{3} \\
v_{t} &= -0.5\,u_{xx} - u\,v^{2} - u^{3}
\end{aligned}
$$

Data shape: `(256, 251)`, components `('u', 'v')`; G shape: `pending`. `m=(19, 25)`, `s=(5, 5)`, `p=(11, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## KS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xx} - u_{xxxx}
\end{aligned}
$$

Data shape: `(256, 301)`, components `('u',)`; G shape: `pending`. `m=(23, 22)`, `s=(5, 6)`, `p=(10, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## KdV

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xxx}
\end{aligned}
$$

Data shape: `(400, 601)`, components `('u',)`; G shape: `pending`. `m=(45, 80)`, `s=(8, 12)`, `p=(8, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## IB

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x}
\end{aligned}
$$

Data shape: `(256, 256)`, components `('u',)`; G shape: `pending`. `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
