# Filtered WSINDy reproduction

Target: [Messenger & Bortz, arXiv:2007.02848v3](https://arxiv.org/html/2007.02848v3). Original `U_exact` arrays from the [authors' archive](https://zenodo.org/records/20787783), with MD5 verification. No resimulation, interpolation, or coarsening.

Status: complete; 3,500/3,500 completed trials. KS, NLS, RD are primary benchmarks; IB, KdV, NS, SG are supplementary.

Schedule: 100 trials per noise level; root seed 0. Noise ratios: 0, 0.2, 0.5, 0.75, 1. Subset of the paper identification schedule.

Author-code baseline: least squares on the scaled system, physical-unit MSTLS bounds, return before an empty support, and state scale exponent `1/(beta_max-1)`. Both use published supports/degrees/strides and 50 thresholds `logspace(-4,0,50)`. SVD least squares uses SciPy GELSD, not MATLAB backslash. RD uses the archived degree-5/order-4 library (181 columns, 4860 rows); the paper's contradictory RD row count is not reproduced. See README for source discrepancies.

TPR is TP/(TP+FN+FP), E2 is relative coefficient L2 error, and E_inf is the maximum relative error over true nonzero coefficients. The table reports mean ± sample standard deviation when there are at least two trials, and the observed value for a single trial. Exact is the percentage of trials with precisely the true support. For coupled systems metrics summarize the full coefficient matrix; per-equation values are in the JSONL files. Errors include failed identifications. Runtime covers weak-system construction, scaling, and all threshold refits, excluding disk loading, noise generation, and reporting; concurrent timings are not serial MATLAB timings.

These results measure identification and coefficient accuracy as in arXiv v3. Solution-prediction metrics added in the later journal article are not evaluated. Original MATLAB random draws and runtime measurements are not reproduced.

Reused 35 previously completed trials with unchanged numerical method and seeds. Selection is by requested noise level and trial index only. Original protocol IDs are retained in raw records; see `results.reuse.json` for provenance.

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
| 0 | 100 | 100.0% | 1 ± 0 | 8.18316e-07 ± 0 | 5.54625e-07 ± 0 | 0.0182 |
| 0.2 | 100 | 0.0% | 0.0723575 ± 0.0196 | 2.26388 ± 0.339 | 6.69491 ± 1.47 | 0.0376 |
| 0.5 | 100 | 0.0% | 0.0992914 ± 0.0272 | 2.10143 ± 0.407 | 3.50823 ± 2.15 | 0.0371 |
| 0.75 | 100 | 0.0% | 0.0895441 ± 0.0297 | 2.20814 ± 0.423 | 4.02027 ± 2.05 | 0.0371 |
| 1 | 100 | 0.0% | 0.090529 ± 0.0351 | 2.20801 ± 0.422 | 4.03428 ± 2.03 | 0.0377 |

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
| 0 | 100 | 100.0% | 1 ± 0 | 9.28616e-08 ± 0 | 5.60489e-08 ± 6.65e-24 | 0.0863 |
| 0.2 | 100 | 0.0% | 0.662469 ± 0.0953 | 0.134777 ± 0.0371 | 0.149227 ± 0.0358 | 0.551 |
| 0.5 | 100 | 0.0% | 0.129448 ± 0.0584 | 2.41777 ± 0.798 | 5.35843 ± 1.9 | 0.958 |
| 0.75 | 100 | 0.0% | 0.131142 ± 0.0416 | 2.57529 ± 0.767 | 4.97586 ± 2.05 | 1.04 |
| 1 | 100 | 0.0% | 0.123039 ± 0.0372 | 2.42934 ± 0.906 | 4.83771 ± 1.67 | 1.09 |

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
| 0 | 100 | 100.0% | 1 ± 0 | 4.45443e-10 ± 0 | 1.79099e-10 ± 2.6e-26 | 11.8 |
| 0.2 | 100 | 100.0% | 1 ± 0 | 0.0135013 ± 0.00292 | 0.00771947 ± 0.00135 | 11.5 |
| 0.5 | 100 | 74.0% | 0.959036 ± 0.0753 | 0.0999785 ± 0.0902 | 0.068204 ± 0.0731 | 10.4 |
| 0.75 | 100 | 4.0% | 0.751797 ± 0.1 | 0.426226 ± 0.126 | 0.402213 ± 0.146 | 10.5 |
| 1 | 100 | 0.0% | 0.618625 ± 0.11 | 0.598599 ± 0.0527 | 0.65562 ± 0.151 | 10.8 |

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
| 0 | 100 | 0.0% | 0.2 ± 5.58e-17 | 0.258283 ± 5.58e-17 | 2.88961 ± 4.46e-16 | 0.0215 |
| 0.2 | 100 | 0.0% | 0.170988 ± 0.0325 | 0.315689 ± 0.209 | 128.567 ± 263 | 0.0213 |
| 0.5 | 100 | 0.0% | 0.168206 ± 0.0387 | 0.340184 ± 0.278 | 135.368 ± 268 | 0.0211 |
| 0.75 | 100 | 0.0% | 0.17306 ± 0.0415 | 0.38071 ± 0.337 | 114.619 ± 260 | 0.0212 |
| 1 | 100 | 0.0% | 0.18081 ± 0.0423 | 0.449489 ± 0.433 | 105.431 ± 237 | 0.0202 |

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
| 0 | 100 | 100.0% | 1 ± 0 | 3.13515e-07 ± 0 | 2.84321e-07 ± 1.06e-22 | 0.0593 |
| 0.2 | 100 | 0.0% | 0.457302 ± 0.0955 | 5.19599 ± 1.6 | 4.97227 ± 1.8 | 0.0683 |
| 0.5 | 100 | 0.0% | 0.44746 ± 0.107 | 5.61902 ± 1.09 | 5.52692 ± 1.53 | 0.0701 |
| 0.75 | 100 | 0.0% | 0.417437 ± 0.114 | 5.75482 ± 0.741 | 7.00888 ± 11.1 | 0.0706 |
| 1 | 100 | 1.0% | 0.397802 ± 0.142 | 5.66193 ± 1.24 | 10.5294 ± 22 | 0.0705 |

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
| 0 | 100 | 100.0% | 1 ± 0 | 0.00537952 ± 2.62e-18 | 0.000789496 ± 2.18e-19 | 3.65 |
| 0.2 | 100 | 0.0% | 0.606333 ± 0.225 | 0.542068 ± 0.384 | 0.0704365 ± 0.0196 | 4.13 |
| 0.5 | 100 | 0.0% | 0.31381 ± 0.0235 | 1 ± 0 | 0.244773 ± 0.0726 | 3.96 |
| 0.75 | 100 | 0.0% | 0.309008 ± 0.102 | 1 ± 0 | 0.370325 ± 0.198 | 3.82 |
| 1 | 100 | 0.0% | 0.30867 ± 0.054 | 1 ± 0 | 0.322707 ± 0.124 | 3.8 |

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
| 0 | 100 | 100.0% | 1 ± 0 | 4.36442e-05 ± 0 | 2.74649e-05 ± 0 | 4.08 |
| 0.2 | 100 | 100.0% | 1 ± 0 | 0.0176175 ± 0.00341 | 0.0104067 ± 0.00189 | 4.33 |
| 0.5 | 100 | 97.0% | 0.9855 ± 0.0877 | 0.0743948 ± 0.133 | 0.0465172 ± 0.0927 | 4.35 |
| 0.75 | 100 | 94.0% | 0.9675 ± 0.133 | 0.139386 ± 0.199 | 0.0884601 ± 0.14 | 4.39 |
| 1 | 100 | 89.0% | 0.936667 ± 0.183 | 0.225587 ± 0.274 | 0.144765 ± 0.191 | 3.72 |
