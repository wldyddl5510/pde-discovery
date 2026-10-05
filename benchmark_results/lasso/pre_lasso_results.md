# WSINDy benchmark comparison

Original WSINDy: 3,500/3,500 trials; Filtered WSINDy: 3,500/3,500 trials; 3,500 completed pairs. Status: complete.

Schedule: 100 trials per method at noise ratios `0, 0.2, 0.5, 0.75, 1`; root seed 0. KS, NLS, RD are primary benchmarks; IB, KdV, NS, SG are supplementary. Each PDE uses one fixed true equation, coefficients, initial condition and clean trajectory. Only observation noise changes. Zero-noise repeats use identical observations.

The two rows at each noise level use exactly the same completed trial indices and input seeds. While filtering is running, both rows summarize only those paired trials; the full original 100-trial results remain available in the individual report. Pending metrics are shown as —.

[Original results](benchmark_results/authors/results.md) · [Filtered results](benchmark_results/filtered/results.md) · [Filtered protocol](benchmark_results/filtered/results.manifest.json) · [Execution status](benchmark_results/filtered/results.status.json) · [Paired CSV summary](benchmark_results/filtered/results.paired.summary.csv)

Filtered WSINDy estimates component noise with unit-L2 sixth differences along time ([1,-6,15,-20,15,-6,1]/sqrt(924)), then applies a separable space-time moving average with symmetric reflection (`scipy.ndimage.uniform_filter`, origin 0). The coefficient-ratio prior is 0.01; sigma_est is the maximum component estimate. For D space-time axes, library degree p_max, and weak-test half-width m_d, w_d = max(1, floor(min(2*(binom(p_max,2)*sigma_est^2/prior)^(1/D), (2*m_d+1)/2))). All components share the same window. Both LHS and RHS use filtered states, with library nonlinearities evaluated after averaging. These are adaptations of [the consistency paper, Sections 4.2/5.4 and Appendix G](https://arxiv.org/pdf/2211.16000): the degree-6 factor 1500 is generalized, and each anisotropic axis is capped separately. The existing polynomial tests, 50 thresholds, scaling, and MSTLS are retained. Window selection uses observed data only, including at zero noise. This applies the preprocessing comparison to the original seven-PDE benchmark protocol. Smoothing bias and signal contamination of the noise estimate can reduce accuracy.

Both methods use the `authors` regression profile, identical original grids, libraries, test supports/degrees/strides, and 50 thresholds. Noise ratios use clean-component RMS as in the original PDE paper. Raw baseline records and its original protocol are preserved.

TPR = TP/(TP+FN+FP); Exact is the percentage of exact support recoveries. E_inf is maximum relative error over true nonzero coefficients; E2 is relative coefficient L2 error. Metrics pool coupled equations and include failed identifications. Values are mean ± sample SD. Filtered runtime includes noise estimation and averaging; loading, noise generation and reports are excluded. Timings come from separate runs with recorded concurrency and are not paired serial speed measurements.

## KS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xx} - u_{xxxx}
\end{aligned}
$$

Data shape: `(256, 301)`; components `('u',)`; `m=(23, 22)`, `s=(5, 6)`, `p=(10, 10)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 8.18316e-07 ± 0 | 5.54625e-07 ± 0 | 0.0176 |
| 0 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 8.18316e-07 ± 0 | 5.54625e-07 ± 0 | 0.0182 |
| 0.2 | WSINDy | 100 | 100.0% | 1 ± 0 | 0.0122972 ± 0.00551 | 0.0104377 ± 0.0056 | 0.0429 |
| 0.2 | Filtered WSINDy | 100 | 0.0% | 0.0723575 ± 0.0196 | 2.26388 ± 0.339 | 6.69491 ± 1.47 | 0.0376 |
| 0.5 | WSINDy | 100 | 100.0% | 1 ± 0 | 0.0722229 ± 0.0168 | 0.0670009 ± 0.0173 | 0.0459 |
| 0.5 | Filtered WSINDy | 100 | 0.0% | 0.0992914 ± 0.0272 | 2.10143 ± 0.407 | 3.50823 ± 2.15 | 0.0371 |
| 0.75 | WSINDy | 100 | 98.0% | 0.995 ± 0.0352 | 0.164424 ± 0.0274 | 0.15789 ± 0.0246 | 0.0504 |
| 0.75 | Filtered WSINDy | 100 | 0.0% | 0.0895441 ± 0.0297 | 2.20814 ± 0.423 | 4.02027 ± 2.05 | 0.0371 |
| 1 | WSINDy | 100 | 87.0% | 0.943357 ± 0.158 | 0.326388 ± 0.222 | 0.309675 ± 0.172 | 0.0501 |
| 1 | Filtered WSINDy | 100 | 0.0% | 0.090529 ± 0.0351 | 2.20801 ± 0.422 | 4.03428 ± 2.03 | 0.0377 |

## NLS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.5\,v_{xx} + u^{2}\,v + v^{3} \\
v_{t} &= -0.5\,u_{xx} - u\,v^{2} - u^{3}
\end{aligned}
$$

Data shape: `(256, 251)`; components `('u', 'v')`; `m=(19, 25)`, `s=(5, 5)`, `p=(11, 10)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 9.28616e-08 ± 0 | 5.60489e-08 ± 6.65e-24 | 0.0804 |
| 0 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 9.28616e-08 ± 0 | 5.60489e-08 ± 6.65e-24 | 0.0863 |
| 0.2 | WSINDy | 100 | 100.0% | 1 ± 0 | 0.031011 ± 0.00358 | 0.0200444 ± 0.00267 | 0.586 |
| 0.2 | Filtered WSINDy | 100 | 0.0% | 0.662469 ± 0.0953 | 0.134777 ± 0.0371 | 0.149227 ± 0.0358 | 0.551 |
| 0.5 | WSINDy | 100 | 74.0% | 0.940207 ± 0.124 | 0.192192 ± 0.166 | 0.13637 ± 0.0799 | 0.811 |
| 0.5 | Filtered WSINDy | 100 | 0.0% | 0.129448 ± 0.0584 | 2.41777 ± 0.798 | 5.35843 ± 1.9 | 0.958 |
| 0.75 | WSINDy | 100 | 3.0% | 0.636592 ± 0.163 | 0.326789 ± 0.143 | 0.393518 ± 0.124 | 0.887 |
| 0.75 | Filtered WSINDy | 100 | 0.0% | 0.131142 ± 0.0416 | 2.57529 ± 0.767 | 4.97586 ± 2.05 | 1.04 |
| 1 | WSINDy | 100 | 0.0% | 0.367984 ± 0.19 | 0.587566 ± 0.228 | 1.01973 ± 0.51 | 0.862 |
| 1 | Filtered WSINDy | 100 | 0.0% | 0.123039 ± 0.0372 | 2.42934 ± 0.906 | 4.83771 ± 1.67 | 1.09 |

## RD

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.1\,u_{xx} + 0.1\,u_{yy} - u\,v^{2} - u^{3} + v^{3} + u^{2}\,v + u \\
v_{t} &= 0.1\,v_{xx} + 0.1\,v_{yy} - u\,v^{2} - u^{3} - v^{3} - u^{2}\,v + v
\end{aligned}
$$

Data shape: `(256, 256, 201)`; components `('u', 'v')`; `m=(13, 13, 14)`, `s=(13, 13, 12)`, `p=(13, 13, 12)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 4.45443e-10 ± 0 | 1.79099e-10 ± 2.6e-26 | 11.1 |
| 0 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 4.45443e-10 ± 0 | 1.79099e-10 ± 2.6e-26 | 11.8 |
| 0.2 | WSINDy | 100 | 97.0% | 0.998 ± 0.0114 | 0.0444576 ± 0.00371 | 0.0283019 ± 0.00161 | 14.1 |
| 0.2 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 0.0135013 ± 0.00292 | 0.00771947 ± 0.00135 | 11.5 |
| 0.5 | WSINDy | 100 | 0.0% | 0.870882 ± 0.014 | 0.106342 ± 0.0221 | 0.0932207 ± 0.00825 | 10.8 |
| 0.5 | Filtered WSINDy | 100 | 74.0% | 0.959036 ± 0.0753 | 0.0999785 ± 0.0902 | 0.068204 ± 0.0731 | 10.4 |
| 0.75 | WSINDy | 100 | 0.0% | 0.765429 ± 0.0469 | 0.304019 ± 0.119 | 0.260724 ± 0.0478 | 10.4 |
| 0.75 | Filtered WSINDy | 100 | 4.0% | 0.751797 ± 0.1 | 0.426226 ± 0.126 | 0.402213 ± 0.146 | 10.5 |
| 1 | WSINDy | 100 | 0.0% | 0.662298 ± 0.051 | 0.55438 ± 0.188 | 0.47153 ± 0.0719 | 10.5 |
| 1 | Filtered WSINDy | 100 | 0.0% | 0.618625 ± 0.11 | 0.598599 ± 0.0527 | 0.65562 ± 0.151 | 10.8 |

## IB

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x}
\end{aligned}
$$

Data shape: `(256, 256)`; components `('u',)`; `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 4.30846e-05 ± 0 | 4.30846e-05 ± 0 | 0.0253 |
| 0 | Filtered WSINDy | 100 | 0.0% | 0.2 ± 5.58e-17 | 0.258283 ± 5.58e-17 | 2.88961 ± 4.46e-16 | 0.0215 |
| 0.2 | WSINDy | 100 | 100.0% | 1 ± 0 | 0.00408295 ± 0.00318 | 0.00408295 ± 0.00318 | 0.0253 |
| 0.2 | Filtered WSINDy | 100 | 0.0% | 0.170988 ± 0.0325 | 0.315689 ± 0.209 | 128.567 ± 263 | 0.0213 |
| 0.5 | WSINDy | 100 | 99.0% | 0.991111 ± 0.0889 | 0.0109775 ± 0.0094 | 0.940651 ± 9.3 | 0.0241 |
| 0.5 | Filtered WSINDy | 100 | 0.0% | 0.168206 ± 0.0387 | 0.340184 ± 0.278 | 135.368 ± 268 | 0.0211 |
| 0.75 | WSINDy | 100 | 100.0% | 1 ± 0 | 0.0182255 ± 0.0129 | 0.0182255 ± 0.0129 | 0.0236 |
| 0.75 | Filtered WSINDy | 100 | 0.0% | 0.17306 ± 0.0415 | 0.38071 ± 0.337 | 114.619 ± 260 | 0.0212 |
| 1 | WSINDy | 100 | 99.0% | 0.995 ± 0.05 | 0.0214272 ± 0.0151 | 0.0219393 ± 0.016 | 0.0243 |
| 1 | Filtered WSINDy | 100 | 0.0% | 0.18081 ± 0.0423 | 0.449489 ± 0.433 | 105.431 ± 237 | 0.0202 |

## KdV

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xxx}
\end{aligned}
$$

Data shape: `(400, 601)`; components `('u',)`; `m=(45, 80)`, `s=(8, 12)`, `p=(8, 7)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 3.13515e-07 ± 0 | 2.84321e-07 ± 1.06e-22 | 0.0691 |
| 0 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 3.13515e-07 ± 0 | 2.84321e-07 ± 1.06e-22 | 0.0593 |
| 0.2 | WSINDy | 100 | 98.0% | 0.988333 ± 0.0829 | 0.0277983 ± 0.14 | 0.250095 ± 2.34 | 0.0749 |
| 0.2 | Filtered WSINDy | 100 | 0.0% | 0.457302 ± 0.0955 | 5.19599 ± 1.6 | 4.97227 ± 1.8 | 0.0683 |
| 0.5 | WSINDy | 100 | 95.0% | 0.971667 ± 0.125 | 0.0706185 ± 0.215 | 0.0631837 ± 0.192 | 0.0755 |
| 0.5 | Filtered WSINDy | 100 | 0.0% | 0.44746 ± 0.107 | 5.61902 ± 1.09 | 5.52692 ± 1.53 | 0.0701 |
| 0.75 | WSINDy | 100 | 94.0% | 0.971667 ± 0.114 | 0.0813984 ± 0.213 | 0.146024 ± 0.751 | 0.0755 |
| 0.75 | Filtered WSINDy | 100 | 0.0% | 0.417437 ± 0.114 | 5.75482 ± 0.741 | 7.00888 ± 11.1 | 0.0706 |
| 1 | WSINDy | 100 | 99.0% | 0.995 ± 0.05 | 0.0520222 ± 0.103 | 0.0465661 ± 0.0917 | 0.0757 |
| 1 | Filtered WSINDy | 100 | 1.0% | 0.397802 ± 0.142 | 5.66193 ± 1.24 | 10.5294 ± 22 | 0.0705 |

## NS

**True equation (physical units):**

$$
\begin{aligned}
\omega_{t} &= -\left(\omega\,u\right)_{x} - \left(\omega\,v\right)_{y} + 0.01\,\omega_{xx} + 0.01\,\omega_{yy}
\end{aligned}
$$

Here $\omega$ is vorticity and $(u,v)$ is the observed incompressible velocity ($u_x+v_y=0$). Only the vorticity equation is identified; $-(\omega u)_x-(\omega v)_y=-u\omega_x-v\omega_y$. The viscosity is $\nu=0.01$.

Data shape: `(324, 149, 201)`; components `('omega', 'u', 'v')`; `m=(31, 31, 14)`, `s=(12, 12, 8)`, `p=(9, 9, 12)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 0.00537952 ± 2.62e-18 | 0.000789496 ± 2.18e-19 | 3.11 |
| 0 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 0.00537952 ± 2.62e-18 | 0.000789496 ± 2.18e-19 | 3.65 |
| 0.2 | WSINDy | 100 | 100.0% | 1 ± 0 | 0.0128688 ± 0.00914 | 0.00131589 ± 0.000509 | 3.42 |
| 0.2 | Filtered WSINDy | 100 | 0.0% | 0.606333 ± 0.225 | 0.542068 ± 0.384 | 0.0704365 ± 0.0196 | 4.13 |
| 0.5 | WSINDy | 100 | 0.0% | 0.398667 ± 0.0276 | 1 ± 0 | 0.0697783 ± 0.00971 | 3.41 |
| 0.5 | Filtered WSINDy | 100 | 0.0% | 0.31381 ± 0.0235 | 1 ± 0 | 0.244773 ± 0.0726 | 3.96 |
| 0.75 | WSINDy | 100 | 0.0% | 0.409 ± 0.0288 | 1 ± 0 | 0.0653139 ± 0.0101 | 3.43 |
| 0.75 | Filtered WSINDy | 100 | 0.0% | 0.309008 ± 0.102 | 1 ± 0 | 0.370325 ± 0.198 | 3.82 |
| 1 | WSINDy | 100 | 0.0% | 0.5 ± 0 | 1 ± 0 | 0.0468941 ± 0.00246 | 3.1 |
| 1 | Filtered WSINDy | 100 | 0.0% | 0.30867 ± 0.054 | 1 ± 0 | 0.322707 ± 0.124 | 3.8 |

## SG

**True equation (physical units):**

$$
\begin{aligned}
u_{tt} &= u_{xx} + u_{yy} - \sin\left(u\right)
\end{aligned}
$$

Data shape: `(129, 403, 205)`; components `('u',)`; `m=(40, 40, 25)`, `s=(5, 5, 8)`, `p=(8, 8, 10)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 4.36442e-05 ± 0 | 2.74649e-05 ± 0 | 3.79 |
| 0 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 4.36442e-05 ± 0 | 2.74649e-05 ± 0 | 4.08 |
| 0.2 | WSINDy | 100 | 100.0% | 1 ± 0 | 0.030438 ± 0.0037 | 0.017685 ± 0.00215 | 4.19 |
| 0.2 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 0.0176175 ± 0.00341 | 0.0104067 ± 0.00189 | 4.33 |
| 0.5 | WSINDy | 100 | 98.0% | 0.993333 ± 0.0469 | 0.217881 ± 0.113 | 0.126432 ± 0.0659 | 4.47 |
| 0.5 | Filtered WSINDy | 100 | 97.0% | 0.9855 ± 0.0877 | 0.0743948 ± 0.133 | 0.0465172 ± 0.0927 | 4.35 |
| 0.75 | WSINDy | 100 | 95.0% | 0.986667 ± 0.0589 | 0.520554 ± 0.0542 | 0.301549 ± 0.0316 | 3.89 |
| 0.75 | Filtered WSINDy | 100 | 94.0% | 0.9675 ± 0.133 | 0.139386 ± 0.199 | 0.0884601 ± 0.14 | 4.39 |
| 1 | WSINDy | 100 | 81.0% | 0.946667 ± 0.112 | 1.06346 ± 0.0546 | 0.6153 ± 0.0318 | 3.6 |
| 1 | Filtered WSINDy | 100 | 89.0% | 0.936667 ± 0.183 | 0.225587 ± 0.274 | 0.144765 ± 0.191 | 3.72 |

---

# Consistency paper: two additional PDE benchmarks

Target: [Messenger & Bortz, arXiv:2211.16000v1, Sections 5.3.2/5.4.2](https://arxiv.org/html/2211.16000v1#S5). This is a **declared resimulation benchmark**, not an exact replication of the authors' trajectories or resolution sweeps. The published equations, domains, libraries and bump test function are used. Initial conditions, periodic boundaries, fixed observation grids, query strides and solver step sizes are explicitly chosen here because the original data/generation settings were not found in the checked public sources. The high-resolution spectral ETDRK4 trajectories are subsampled by four on both axes. Timestep refinement is checked and recorded in the manifest. No model-selection results are used to choose these settings. Noise uses the centered clean sample standard deviation (ddof=1), unlike the original suite's RMS convention. Both methods use physical coordinates without state/coordinate rescaling, the C-infinity bump exp(9/(r^2-1)), absolute hard-threshold STLS until stable, and 100 thresholds logspace(-4,0,100), with the projection-plus-sparsity loss of Eqs. 2.6–2.9.

Filter prior tau_star=0.01. Noise is estimated with unit-L2 sixth differences along time. For D space-time axes and m=prod(2*half_width+1), use the common side w=max(1,floor(min(2*(binom(p_max,2)*sigma_est^2/tau_star)^(1/D),m^(1/D)/2))). Average space and time with symmetric reflection before evaluating the nonlinear library; both weak LHS and RHS use filtered states. The VBG degree-6 factor is 1500. For HKS the same rule is extended to degree 8 (factor 2800); the paper does not report filtered HKS results. At zero noise the estimator still uses observed data only. Smoothing bias can worsen identification.

WSINDy: 1,000/1,000; Filtered WSINDy: 1,000/1,000; 1,000 completed pairs. Status: complete.

Schedule: 100 trials per method at noise ratios `0, 0.2, 0.5, 0.75, 1`; root seed 0. Each pair uses the same noise seed and injected standard deviation. Rows summarize only completed matching trials. Each PDE has a fixed true equation and clean trajectory; zero-noise repeats are identical.

TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E_inf is maximum relative error over true coefficients; E2 is relative coefficient L2 error. Mean ± sample SD includes failures. Median seconds includes filtering, excludes loading/noise/generation, and is measured under the recorded concurrency. Subset-support recovery is included in CSV/JSON summaries.

[Raw results](benchmark_results/consistency/raw/results.md) · [Filtered results](benchmark_results/consistency/filtered/results.md) · [Protocol](benchmark_results/consistency/filtered/results.manifest.json) · [Execution status](benchmark_results/consistency/filtered/results.status.json)

## HKS

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= u_{xxxx} + 0.75\,u_{xxxxxx} - 0.5\,\left(u^{2}\right)_{x} + 0.1\,\left(u^{2}\right)_{xxx}
\end{aligned}
$$

Declared IC: `cos(x/16)*(1+sin(x/16))`; periodic boundary; x in `[0.0, 100.53096491487338]`, t in `[0.0, 82.0]`. Fine simulation: `[1024, 1025]`; subsampling: `[4, 4]`.

Data shape: `(256, 257)`; components `('u',)`; `m=(26, 26)`, `s=(7, 7)`, `p=(0, 0)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 1.66438e-06 ± 0 | 5.10996e-07 ± 2.13e-22 | 0.0267 |
| 0 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 1.66438e-06 ± 0 | 5.10996e-07 ± 2.13e-22 | 0.0284 |
| 0.2 | WSINDy | 100 | 75.0% | 0.933 ± 0.119 | 0.311273 ± 0.402 | 0.040731 ± 0.0349 | 0.111 |
| 0.2 | Filtered WSINDy | 100 | 0.0% | 0.0528484 ± 0.0162 | 103.489 ± 43.8 | 99.4648 ± 31.7 | 0.0855 |
| 0.5 | WSINDy | 100 | 4.0% | 0.73469 ± 0.116 | 0.986278 ± 0.177 | 0.168049 ± 0.179 | 0.122 |
| 0.5 | Filtered WSINDy | 100 | 0.0% | 0.0618768 ± 0.0164 | 64.4332 ± 16.1 | 93.6061 ± 36.2 | 0.0904 |
| 0.75 | WSINDy | 100 | 0.0% | 0.628053 ± 0.193 | 1.19907 ± 0.771 | 0.343221 ± 0.284 | 0.121 |
| 0.75 | Filtered WSINDy | 100 | 0.0% | 0.0622497 ± 0.0208 | 59.5208 ± 22 | 82.7057 ± 39.4 | 0.0965 |
| 1 | WSINDy | 100 | 0.0% | 0.606274 ± 0.209 | 1.61323 ± 2.66 | 0.483157 ± 0.366 | 0.104 |
| 1 | Filtered WSINDy | 100 | 0.0% | 0.0603478 ± 0.0218 | 55.3038 ± 24.3 | 79.685 ± 38 | 0.0949 |

## VBG

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.01\,u_{xx} - 0.5\,\left(u^{2}\right)_{x} - u^{3} + 2\,u^{2} + 1
\end{aligned}
$$

Declared IC: `2*sin(pi*x)`; periodic boundary; x in `[-1.0, 1.0]`, t in `[0.0, 1.5]`. Fine simulation: `[2048, 1801]`; subsampling: `[4, 4]`.

Data shape: `(512, 451)`; components `('u',)`; `m=(64, 56)`, `s=(12, 11)`, `p=(0, 0)`.

| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | WSINDy | 100 | 100.0% | 1 ± 0 | 2.18747e-11 ± 0 | 1.17192e-12 ± 2.03e-28 | 0.0541 |
| 0 | Filtered WSINDy | 100 | 100.0% | 1 ± 0 | 2.18747e-11 ± 0 | 1.17192e-12 ± 2.03e-28 | 0.0575 |
| 0.2 | WSINDy | 100 | 0.0% | 0.799333 ± 0.0314 | 0.935103 ± 0.238 | 0.0412677 ± 0.0531 | 0.0594 |
| 0.2 | Filtered WSINDy | 100 | 9.0% | 0.765746 ± 0.11 | 0.80834 ± 0.154 | 0.0920576 ± 0.102 | 0.0652 |
| 0.5 | WSINDy | 100 | 0.0% | 0.669655 ± 0.108 | 1.00289 ± 0.106 | 0.288737 ± 0.163 | 0.0611 |
| 0.5 | Filtered WSINDy | 100 | 1.0% | 0.654307 ± 0.133 | 2.47793 ± 0.473 | 0.233236 ± 0.22 | 0.0672 |
| 0.75 | WSINDy | 100 | 0.0% | 0.490853 ± 0.0615 | 1.00482 ± 0.0346 | 0.716361 ± 0.0989 | 0.0641 |
| 0.75 | Filtered WSINDy | 100 | 1.0% | 0.619222 ± 0.148 | 3.85489 ± 0.908 | 0.326748 ± 0.257 | 0.0672 |
| 1 | WSINDy | 100 | 0.0% | 0.490536 ± 0.0788 | 1.04512 ± 0.139 | 1.05185 ± 0.318 | 0.062 |
| 1 | Filtered WSINDy | 100 | 0.0% | 0.593655 ± 0.138 | 4.02666 ± 1.14 | 0.413761 ± 0.302 | 0.0686 |
