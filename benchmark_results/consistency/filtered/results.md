# Filtered WSINDy consistency PDE benchmark

Target: [Messenger & Bortz, arXiv:2211.16000v1, Sections 5.3.2/5.4.2](https://arxiv.org/html/2211.16000v1#S5). This is a **declared resimulation benchmark**, not an exact replication of the authors' trajectories or resolution sweeps. The published equations, domains, libraries and bump test function are used. Initial conditions, periodic boundaries, fixed observation grids, query strides and solver step sizes are explicitly chosen here because the original data/generation settings were not found in the checked public sources. The high-resolution spectral ETDRK4 trajectories are subsampled by four on both axes. Timestep refinement is checked and recorded in the manifest. No model-selection results are used to choose these settings. Noise uses the centered clean sample standard deviation (ddof=1), unlike the original suite's RMS convention. Both methods use physical coordinates without state/coordinate rescaling, the C-infinity bump exp(9/(r^2-1)), absolute hard-threshold STLS until stable, and 100 thresholds logspace(-4,0,100), with the projection-plus-sparsity loss of Eqs. 2.6–2.9.

Status: complete; 1,000/1,000 completed trials.

Schedule: 100 trials per noise ratio at `0, 0.2, 0.5, 0.75, 1`; root seed 0. One fixed trajectory per PDE; only Gaussian observation noise varies. Zero-noise repeats are identical.

TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E_inf is maximum relative error over true coefficients; E2 is relative coefficient L2 error. Mean ± sample SD includes failed identifications. Runtime excludes generation/loading/noise and includes preprocessing. Subset-support probabilities and Wilson exact-recovery intervals are saved in summaries.

Filter prior tau_star=0.01. Noise is estimated with unit-L2 sixth differences along time. For D space-time axes and m=prod(2*half_width+1), use the common side w=max(1,floor(min(2*(binom(p_max,2)*sigma_est^2/tau_star)^(1/D),m^(1/D)/2))). Average space and time with symmetric reflection before evaluating the nonlinear library; both weak LHS and RHS use filtered states. The VBG degree-6 factor is 1500. For HKS the same rule is extended to degree 8 (factor 2800); the paper does not report filtered HKS results. At zero noise the estimator still uses observed data only. Smoothing bias can worsen identification.

Runtime includes noise estimation and moving-average preprocessing.

## HKS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= u_{xxxx} + 0.75\,u_{xxxxxx} - 0.5\,\left(u^{2}\right)_{x} + 0.1\,\left(u^{2}\right)_{xxx}
\end{aligned}
$$

Declared IC: `cos(x/16)*(1+sin(x/16))`; boundary: periodic; x in `[0.0, 100.53096491487338]`, t in `[0.0, 82.0]`. Fine simulation: `[1024, 1025]`; observation subsampling: `[4, 4]`.

Observed filter-side ranges (grid points, space then time): `[(1, 26), (1, 26)]`.

Data shape: `(256, 257)`, components `('u',)`; G shape: `(900, 73)`. `m=(26, 26)`, `s=(7, 7)`, `p=(0, 0)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 1.66438e-06 ± 0 | 5.10996e-07 ± 2.13e-22 | 0.0284 |
| 0.2 | 100 | 0.0% | 0.0528484 ± 0.0162 | 103.489 ± 43.8 | 99.4648 ± 31.7 | 0.0855 |
| 0.5 | 100 | 0.0% | 0.0618768 ± 0.0164 | 64.4332 ± 16.1 | 93.6061 ± 36.2 | 0.0904 |
| 0.75 | 100 | 0.0% | 0.0622497 ± 0.0208 | 59.5208 ± 22 | 82.7057 ± 39.4 | 0.0965 |
| 1 | 100 | 0.0% | 0.0603478 ± 0.0218 | 55.3038 ± 24.3 | 79.685 ± 38 | 0.0949 |

## VBG

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.01\,u_{xx} - 0.5\,\left(u^{2}\right)_{x} - u^{3} + 2\,u^{2} + 1
\end{aligned}
$$

Declared IC: `2*sin(pi*x)`; boundary: periodic; x in `[-1.0, 1.0]`, t in `[0.0, 1.5]`. Fine simulation: `[2048, 1801]`; observation subsampling: `[4, 4]`.

Observed filter-side ranges (grid points, space then time): `[(1, 60), (1, 60)]`.

Data shape: `(512, 451)`, components `('u',)`; G shape: `(992, 43)`. `m=(64, 56)`, `s=(12, 11)`, `p=(0, 0)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 2.18747e-11 ± 0 | 1.17192e-12 ± 2.03e-28 | 0.0575 |
| 0.2 | 100 | 9.0% | 0.765746 ± 0.11 | 0.80834 ± 0.154 | 0.0920576 ± 0.102 | 0.0652 |
| 0.5 | 100 | 1.0% | 0.654307 ± 0.133 | 2.47793 ± 0.473 | 0.233236 ± 0.22 | 0.0672 |
| 0.75 | 100 | 1.0% | 0.619222 ± 0.148 | 3.85489 ± 0.908 | 0.326748 ± 0.257 | 0.0672 |
| 1 | 100 | 0.0% | 0.593655 ± 0.138 | 4.02666 ± 1.14 | 0.413761 ± 0.302 | 0.0686 |
