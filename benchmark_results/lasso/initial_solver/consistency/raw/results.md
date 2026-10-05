# WSINDy + LASSO benchmark

Status: in progress; 230/1,000 completed trials.

Schedule: 100 trials per noise level at `0, 0.2, 0.5, 0.75, 1`; root seed 0. One fixed equation and trajectory per PDE; only observation noise varies. Zero-noise repeats are identical clean checks.

HKS/VBG are declared resimulations using the consistency-paper equations; IC/BC/grids/strides are declared here. Weak systems use the C-infinity bump with physical coordinates and no rescaling. Noise ratio is relative to centered clean sample std.

LASSO replaces sequential hard thresholding. For K weak rows, normalize each nonzero column Z_j=G_j/RMS(G_j) and each response y_e=b_e/RMS(b_e), without centering or an extra intercept (the constant library term is penalized). Solve `||Z theta-y_e||_2^2/(2*K)+alpha_e*||theta||_1`, with `alpha_e=ratio*max(abs(Z.T@y_e))/K` at 100 ratios `logspace(-4,0,100)`. Coupled equations use separate alpha_max values and one selected ratio. Select using the existing WSINDy loss: matrix-2-norm prediction difference from full OLS / OLS prediction norm + fraction of nonzero coefficients; smallest ratio breaks ties. Retain the penalized coefficients, including shrinkage, and restore physical units; there is no OLS support refit or coefficient cutoff. This is a column-normalized L1 penalty, equivalently a weighted penalty in physical coefficient units. Empty supports are allowed. The scikit-learn LARS Gram path is checked against KKT conditions at every candidate; coordinate descent and primal active-set polishing are used if needed, and violation / alpha_max above 1e-7 rejects the fit. Lambda selection uses G and b only. These are additional comparisons, not the papers' published sparse-regression algorithms.

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

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 0.0% | 0.375 ± 0 | 1 ± 0 | 0.756139 ± 1.12e-16 | 0.124 |
| 0.2 | 100 | 0.0% | 0.345298 ± 0.0456 | 1 ± 0 | 0.869025 ± 0.0339 | 1.47 |
| 0.5 | 30 | 0.0% | 0.338161 ± 0.0542 | 1 ± 0 | 0.880446 ± 0.0198 | 4.67 |

## VBG

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.01\,u_{xx} - 0.5\,\left(u^{2}\right)_{x} - u^{3} + 2\,u^{2} + 1
\end{aligned}
$$

Declared IC: `2*sin(pi*x)`; boundary: periodic; x in `[-1.0, 1.0]`, t in `[0.0, 1.5]`. Fine simulation: `[2048, 1801]`; observation subsampling: `[4, 4]`.

Data shape: `(512, 451)`, components `('u',)`; G shape: `pending`. `m=(64, 56)`, `s=(12, 11)`, `p=(0, 0)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
