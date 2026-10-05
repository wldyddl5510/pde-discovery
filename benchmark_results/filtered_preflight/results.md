# Filtered WSINDy reproduction

Target: [Messenger & Bortz, arXiv:2007.02848v3](https://arxiv.org/html/2007.02848v3). Original `U_exact` arrays from the [authors' archive](https://zenodo.org/records/20787783), with MD5 verification. No resimulation, interpolation, or coarsening.

Status: complete; 35/35 completed trials. KS, NLS, RD are primary benchmarks; IB, KdV, NS, SG are supplementary.

Schedule: 1 trials per noise level; root seed 0. Noise ratios: 0, 0.2, 0.5, 0.75, 1. Subset of the paper identification schedule.

Author-code baseline: least squares on the scaled system, physical-unit MSTLS bounds, return before an empty support, and state scale exponent `1/(beta_max-1)`. Both use published supports/degrees/strides and 50 thresholds `logspace(-4,0,50)`. SVD least squares uses SciPy GELSD, not MATLAB backslash. RD uses the archived degree-5/order-4 library (181 columns, 4860 rows); the paper's contradictory RD row count is not reproduced. See README for source discrepancies.

TPR is TP/(TP+FN+FP), E2 is relative coefficient L2 error, and E_inf is the maximum relative error over true nonzero coefficients. The table reports mean ± sample standard deviation when there are at least two trials, and the observed value for a single trial. Exact is the percentage of trials with precisely the true support. For coupled systems metrics summarize the full coefficient matrix; per-equation values are in the JSONL files. Errors include failed identifications. Runtime covers weak-system construction, scaling, and all threshold refits, excluding disk loading, noise generation, and reporting; concurrent timings are not serial MATLAB timings.

These results measure identification and coefficient accuracy as in arXiv v3. Solution-prediction metrics added in the later journal article are not evaluated. Original MATLAB random draws and runtime measurements are not reproduced.

Filtered WSINDy estimates component noise with unit-L2 sixth differences along time ([1,-6,15,-20,15,-6,1]/sqrt(924)), then applies a separable space-time moving average with symmetric reflection (`scipy.ndimage.uniform_filter`, origin 0). The coefficient-ratio prior is 0.01; sigma_est is the maximum component estimate. For D space-time axes, library degree p_max, and weak-test half-width m_d, w_d = max(1, floor(min(2*(binom(p_max,2)*sigma_est^2/prior)^(1/D), (2*m_d+1)/2))). All components share the same window. Both LHS and RHS use filtered states, with library nonlinearities evaluated after averaging. These are adaptations of [the consistency paper, Sections 4.2/5.4 and Appendix G](https://arxiv.org/pdf/2211.16000): the degree-6 factor 1500 is generalized, and each anisotropic axis is capped separately. The existing polynomial tests, 50 thresholds, scaling, and MSTLS are retained. Window selection uses observed data only, including at zero noise. This applies the preprocessing comparison to the original seven-PDE benchmark protocol. Smoothing bias and signal contamination of the noise estimate can reduce accuracy.

Runtime includes noise estimation and moving-average preprocessing.

## KS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xx} - u_{xxxx}
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(1, 23), (1, 22)]`.

Data shape: `(256, 301)`, components `('u',)`; G shape: `(1806, 43)`. `m=(23, 22)`, `s=(5, 6)`, `p=(10, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 8.18316e-07 | 5.54625e-07 | 0.0197 |
| 0.2 | 1 | 0.0% | 0.0909091 | 1.9998 | 5.65162 | 0.0352 |
| 0.5 | 1 | 0.0% | 0.111111 | 1.76635 | 1.45205 | 0.0393 |
| 0.75 | 1 | 0.0% | 0.0689655 | 2.4315 | 4.58333 | 0.0334 |
| 1 | 1 | 0.0% | 0.0416667 | 2.57109 | 5.38038 | 0.0402 |

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

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 9.28616e-08 | 5.60489e-08 | 0.0773 |
| 0.2 | 1 | 0.0% | 0.666667 | 0.191262 | 0.158484 | 0.5 |
| 0.5 | 1 | 0.0% | 0.0943396 | 2.47412 | 8.43033 | 0.94 |
| 0.75 | 1 | 0.0% | 0.12766 | 2.93607 | 5.45642 | 0.953 |
| 1 | 1 | 0.0% | 0.130435 | 3.57542 | 5.40496 | 0.899 |

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

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 4.45443e-10 | 1.79099e-10 | 10.7 |
| 0.2 | 1 | 100.0% | 1 | 0.0116659 | 0.0067902 | 11.7 |
| 0.5 | 1 | 100.0% | 1 | 0.0457645 | 0.0242156 | 10.5 |
| 0.75 | 1 | 0.0% | 0.636364 | 0.396217 | 0.306296 | 10.6 |
| 1 | 1 | 0.0% | 0.5 | 0.590645 | 0.840505 | 10.5 |

## IB

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x}
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(60, 60), (60, 60)]`.

Data shape: `(256, 256)`, components `('u',)`; G shape: `(784, 43)`. `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.0% | 0.2 | 0.258283 | 2.88961 | 0.0208 |
| 0.2 | 1 | 0.0% | 0.166667 | 0.707485 | 626.336 | 0.02 |
| 0.5 | 1 | 0.0% | 0.142857 | 0.233116 | 25.5536 | 0.0196 |
| 0.75 | 1 | 0.0% | 0.166667 | 0.253138 | 3.13319 | 0.0205 |
| 1 | 1 | 0.0% | 0.166667 | 0.228716 | 7.60997 | 0.0207 |

## KdV

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xxx}
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(1, 45), (1, 80)]`.

Data shape: `(400, 601)`, components `('u',)`; G shape: `(1443, 43)`. `m=(45, 80)`, `s=(8, 12)`, `p=(8, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 3.13515e-07 | 2.84321e-07 | 0.0605 |
| 0.2 | 1 | 0.0% | 0.5 | 5.90197 | 5.30715 | 0.067 |
| 0.5 | 1 | 0.0% | 0.5 | 5.88982 | 5.29629 | 0.0688 |
| 0.75 | 1 | 0.0% | 0.333333 | 6.13352 | 6.79984 | 0.07 |
| 1 | 1 | 0.0% | 0.666667 | 7.22982 | 6.48755 | 0.0644 |

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

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 0.00537952 | 0.000789496 | 3.53 |
| 0.2 | 1 | 0.0% | 0.333333 | 1 | 0.0943974 | 3.61 |
| 0.5 | 1 | 0.0% | 0.333333 | 1 | 0.180741 | 3.63 |
| 0.75 | 1 | 0.0% | 0.285714 | 1 | 0.170516 | 3.28 |
| 1 | 1 | 0.0% | 0.25 | 1 | 0.356278 | 3.28 |

## SG

**True equation (physical units):**

$$
\begin{aligned}
u_{tt} &= u_{xx} + u_{yy} - \sin\left(u\right)
\end{aligned}
$$

Observed filter-side ranges (grid points, space then time): `[(1, 18), (1, 18), (1, 18)]`.

Data shape: `(129, 403, 205)`, components `('u',)`; G shape: `(13000, 73)`. `m=(40, 40, 25)`, `s=(5, 5, 8)`, `p=(8, 8, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 4.36442e-05 | 2.74649e-05 | 3.56 |
| 0.2 | 1 | 100.0% | 1 | 0.0108661 | 0.00633989 | 3.68 |
| 0.5 | 1 | 100.0% | 1 | 0.0492151 | 0.0299791 | 3.9 |
| 0.75 | 1 | 100.0% | 1 | 0.087518 | 0.0540987 | 3.56 |
| 1 | 1 | 100.0% | 1 | 0.142455 | 0.0839497 | 3.77 |
