# WSINDy reproduction

Target: [Messenger & Bortz, arXiv:2007.02848v3](https://arxiv.org/html/2007.02848v3). Original `U_exact` arrays from the [authors' archive](https://zenodo.org/records/20787783), with MD5 verification. No resimulation, interpolation, or coarsening.

Status: complete; 3,500/3,500 completed trials. KS, NLS, RD are primary benchmarks; IB, KdV, NS, SG are supplementary.

Schedule: 100 trials per noise level; root seed 0. Noise ratios: 0, 0.2, 0.5, 0.75, 1. Subset of the paper identification schedule.

Author-code baseline: least squares on the scaled system, physical-unit MSTLS bounds, return before an empty support, and state scale exponent `1/(beta_max-1)`. Both use published supports/degrees/strides and 50 thresholds `logspace(-4,0,50)`. SVD least squares uses SciPy GELSD, not MATLAB backslash. RD uses the archived degree-5/order-4 library (181 columns, 4860 rows); the paper's contradictory RD row count is not reproduced. See README for source discrepancies.

TPR is TP/(TP+FN+FP), E2 is relative coefficient L2 error, and E_inf is the maximum relative error over true nonzero coefficients. The table reports mean ± sample standard deviation when there are at least two trials, and the observed value for a single trial. Exact is the percentage of trials with precisely the true support. For coupled systems metrics summarize the full coefficient matrix; per-equation values are in the JSONL files. Errors include failed identifications. Runtime covers weak-system construction, scaling, and all threshold refits, excluding disk loading, noise generation, and reporting; concurrent timings are not serial MATLAB timings.

These results measure identification and coefficient accuracy as in arXiv v3. Solution-prediction metrics added in the later journal article are not evaluated. Original MATLAB random draws and runtime measurements are not reproduced.

Reused 3,200 previously completed trials with unchanged numerical method and seeds. Selection is by requested noise level and trial index only. Original protocol IDs are retained in raw records; see `results.reuse.json` for provenance.

## KS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xx} - u_{xxxx}
\end{aligned}
$$

Data shape: `(256, 301)`, components `('u',)`; G shape: `(1806, 43)`. `m=(23, 22)`, `s=(5, 6)`, `p=(10, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 8.18316e-07 ± 0 | 5.54625e-07 ± 0 | 0.0176 |
| 0.2 | 100 | 100.0% | 1 ± 0 | 0.0122972 ± 0.00551 | 0.0104377 ± 0.0056 | 0.0429 |
| 0.5 | 100 | 100.0% | 1 ± 0 | 0.0722229 ± 0.0168 | 0.0670009 ± 0.0173 | 0.0459 |
| 0.75 | 100 | 98.0% | 0.995 ± 0.0352 | 0.164424 ± 0.0274 | 0.15789 ± 0.0246 | 0.0504 |
| 1 | 100 | 87.0% | 0.943357 ± 0.158 | 0.326388 ± 0.222 | 0.309675 ± 0.172 | 0.0501 |

## NLS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.5\,v_{xx} + u^{2}\,v + v^{3} \\
v_{t} &= -0.5\,u_{xx} - u\,v^{2} - u^{3}
\end{aligned}
$$

Data shape: `(256, 251)`, components `('u', 'v')`; G shape: `(1804, 190)`. `m=(19, 25)`, `s=(5, 5)`, `p=(11, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 9.28616e-08 ± 0 | 5.60489e-08 ± 6.65e-24 | 0.0804 |
| 0.2 | 100 | 100.0% | 1 ± 0 | 0.031011 ± 0.00358 | 0.0200444 ± 0.00267 | 0.586 |
| 0.5 | 100 | 74.0% | 0.940207 ± 0.124 | 0.192192 ± 0.166 | 0.13637 ± 0.0799 | 0.811 |
| 0.75 | 100 | 3.0% | 0.636592 ± 0.163 | 0.326789 ± 0.143 | 0.393518 ± 0.124 | 0.887 |
| 1 | 100 | 0.0% | 0.367984 ± 0.19 | 0.587566 ± 0.228 | 1.01973 ± 0.51 | 0.862 |

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
| 0 | 100 | 100.0% | 1 ± 0 | 4.45443e-10 ± 0 | 1.79099e-10 ± 2.6e-26 | 11.1 |
| 0.2 | 100 | 97.0% | 0.998 ± 0.0114 | 0.0444576 ± 0.00371 | 0.0283019 ± 0.00161 | 14.1 |
| 0.5 | 100 | 0.0% | 0.870882 ± 0.014 | 0.106342 ± 0.0221 | 0.0932207 ± 0.00825 | 10.8 |
| 0.75 | 100 | 0.0% | 0.765429 ± 0.0469 | 0.304019 ± 0.119 | 0.260724 ± 0.0478 | 10.4 |
| 1 | 100 | 0.0% | 0.662298 ± 0.051 | 0.55438 ± 0.188 | 0.47153 ± 0.0719 | 10.5 |

## IB

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x}
\end{aligned}
$$

Data shape: `(256, 256)`, components `('u',)`; G shape: `(784, 43)`. `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 4.30846e-05 ± 0 | 4.30846e-05 ± 0 | 0.0253 |
| 0.2 | 100 | 100.0% | 1 ± 0 | 0.00408295 ± 0.00318 | 0.00408295 ± 0.00318 | 0.0253 |
| 0.5 | 100 | 99.0% | 0.991111 ± 0.0889 | 0.0109775 ± 0.0094 | 0.940651 ± 9.3 | 0.0241 |
| 0.75 | 100 | 100.0% | 1 ± 0 | 0.0182255 ± 0.0129 | 0.0182255 ± 0.0129 | 0.0236 |
| 1 | 100 | 99.0% | 0.995 ± 0.05 | 0.0214272 ± 0.0151 | 0.0219393 ± 0.016 | 0.0243 |

## KdV

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xxx}
\end{aligned}
$$

Data shape: `(400, 601)`, components `('u',)`; G shape: `(1443, 43)`. `m=(45, 80)`, `s=(8, 12)`, `p=(8, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 3.13515e-07 ± 0 | 2.84321e-07 ± 1.06e-22 | 0.0691 |
| 0.2 | 100 | 98.0% | 0.988333 ± 0.0829 | 0.0277983 ± 0.14 | 0.250095 ± 2.34 | 0.0749 |
| 0.5 | 100 | 95.0% | 0.971667 ± 0.125 | 0.0706185 ± 0.215 | 0.0631837 ± 0.192 | 0.0755 |
| 0.75 | 100 | 94.0% | 0.971667 ± 0.114 | 0.0813984 ± 0.213 | 0.146024 ± 0.751 | 0.0755 |
| 1 | 100 | 99.0% | 0.995 ± 0.05 | 0.0520222 ± 0.103 | 0.0465661 ± 0.0917 | 0.0757 |

## NS

**True equation (physical units):**

$$
\begin{aligned}
\omega_{t} &= -\left(\omega\,u\right)_{x} - \left(\omega\,v\right)_{y} + 0.01\,\omega_{xx} + 0.01\,\omega_{yy}
\end{aligned}
$$

Here $\omega$ is vorticity and $(u,v)$ is the observed incompressible velocity ($u_x+v_y=0$). Only the vorticity equation is identified; $-(\omega u)_x-(\omega v)_y=-u\omega_x-v\omega_y$. The viscosity is $\nu=0.01$.

Data shape: `(324, 149, 201)`, components `('omega', 'u', 'v')`; G shape: `(3872, 50)`. `m=(31, 31, 14)`, `s=(12, 12, 8)`, `p=(9, 9, 12)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 0.00537952 ± 2.62e-18 | 0.000789496 ± 2.18e-19 | 3.11 |
| 0.2 | 100 | 100.0% | 1 ± 0 | 0.0128688 ± 0.00914 | 0.00131589 ± 0.000509 | 3.42 |
| 0.5 | 100 | 0.0% | 0.398667 ± 0.0276 | 1 ± 0 | 0.0697783 ± 0.00971 | 3.41 |
| 0.75 | 100 | 0.0% | 0.409 ± 0.0288 | 1 ± 0 | 0.0653139 ± 0.0101 | 3.43 |
| 1 | 100 | 0.0% | 0.5 ± 0 | 1 ± 0 | 0.0468941 ± 0.00246 | 3.1 |

## SG

**True equation (physical units):**

$$
\begin{aligned}
u_{tt} &= u_{xx} + u_{yy} - \sin\left(u\right)
\end{aligned}
$$

Data shape: `(129, 403, 205)`, components `('u',)`; G shape: `(13000, 73)`. `m=(40, 40, 25)`, `s=(5, 5, 8)`, `p=(8, 8, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 4.36442e-05 ± 0 | 2.74649e-05 ± 0 | 3.79 |
| 0.2 | 100 | 100.0% | 1 ± 0 | 0.030438 ± 0.0037 | 0.017685 ± 0.00215 | 4.19 |
| 0.5 | 100 | 98.0% | 0.993333 ± 0.0469 | 0.217881 ± 0.113 | 0.126432 ± 0.0659 | 4.47 |
| 0.75 | 100 | 95.0% | 0.986667 ± 0.0589 | 0.520554 ± 0.0542 | 0.301549 ± 0.0316 | 3.89 |
| 1 | 100 | 81.0% | 0.946667 ± 0.112 | 1.06346 ± 0.0546 | 0.6153 ± 0.0318 | 3.6 |
