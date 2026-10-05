# WSINDy consistency PDE benchmark

Target: [Messenger & Bortz, arXiv:2211.16000v1, Sections 5.3.2/5.4.2](https://arxiv.org/html/2211.16000v1#S5). This is a **declared resimulation benchmark**, not an exact replication of the authors' trajectories or resolution sweeps. The published equations, domains, libraries and bump test function are used. Initial conditions, periodic boundaries, fixed observation grids, query strides and solver step sizes are explicitly chosen here because the original data/generation settings were not found in the checked public sources. The high-resolution spectral ETDRK4 trajectories are subsampled by four on both axes. Timestep refinement is checked and recorded in the manifest. No model-selection results are used to choose these settings. Noise uses the centered clean sample standard deviation (ddof=1), unlike the original suite's RMS convention. Both methods use physical coordinates without state/coordinate rescaling, the C-infinity bump exp(9/(r^2-1)), absolute hard-threshold STLS until stable, and 100 thresholds logspace(-4,0,100), with the projection-plus-sparsity loss of Eqs. 2.6–2.9.

Status: complete; 1,000/1,000 completed trials.

Schedule: 100 trials per noise ratio at `0, 0.2, 0.5, 0.75, 1`; root seed 0. One fixed trajectory per PDE; only Gaussian observation noise varies. Zero-noise repeats are identical.

TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E_inf is maximum relative error over true coefficients; E2 is relative coefficient L2 error. Mean ± sample SD includes failed identifications. Runtime excludes generation/loading/noise and includes preprocessing. Subset-support probabilities and Wilson exact-recovery intervals are saved in summaries.

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
| 0 | 100 | 100.0% | 1 ± 0 | 1.66438e-06 ± 0 | 5.10996e-07 ± 2.13e-22 | 0.0267 |
| 0.2 | 100 | 75.0% | 0.933 ± 0.119 | 0.311273 ± 0.402 | 0.040731 ± 0.0349 | 0.111 |
| 0.5 | 100 | 4.0% | 0.73469 ± 0.116 | 0.986278 ± 0.177 | 0.168049 ± 0.179 | 0.122 |
| 0.75 | 100 | 0.0% | 0.628053 ± 0.193 | 1.19907 ± 0.771 | 0.343221 ± 0.284 | 0.121 |
| 1 | 100 | 0.0% | 0.606274 ± 0.209 | 1.61323 ± 2.66 | 0.483157 ± 0.366 | 0.104 |

## VBG

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.01\,u_{xx} - 0.5\,\left(u^{2}\right)_{x} - u^{3} + 2\,u^{2} + 1
\end{aligned}
$$

Declared IC: `2*sin(pi*x)`; boundary: periodic; x in `[-1.0, 1.0]`, t in `[0.0, 1.5]`. Fine simulation: `[2048, 1801]`; observation subsampling: `[4, 4]`.

Data shape: `(512, 451)`, components `('u',)`; G shape: `(992, 43)`. `m=(64, 56)`, `s=(12, 11)`, `p=(0, 0)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100.0% | 1 ± 0 | 2.18747e-11 ± 0 | 1.17192e-12 ± 2.03e-28 | 0.0541 |
| 0.2 | 100 | 0.0% | 0.799333 ± 0.0314 | 0.935103 ± 0.238 | 0.0412677 ± 0.0531 | 0.0594 |
| 0.5 | 100 | 0.0% | 0.669655 ± 0.108 | 1.00289 ± 0.106 | 0.288737 ± 0.163 | 0.0611 |
| 0.75 | 100 | 0.0% | 0.490853 ± 0.0615 | 1.00482 ± 0.0346 | 0.716361 ± 0.0989 | 0.0641 |
| 1 | 100 | 0.0% | 0.490536 ± 0.0788 | 1.04512 ± 0.139 | 1.05185 ± 0.318 | 0.062 |
