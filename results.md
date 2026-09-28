# PDE coefficient experiments

Experiments 1–3 use the exact mass-one anisotropic porous-medium solution in
[simulation_generation.py](simulation_generation.py), seed `0`, and noise ratios
`0` and `1`. Gaussian noise has standard deviation equal to the ratio times the
RMS of the clean solution on the stated reference grid. The coefficient library
contains every nonzero spatial derivative multi-index of total order at most
`5`, applied to each of `u, ..., u^5`. Reported error is
`||beta_hat - beta_true||_2^2` over the full library. Runtime is the median of
three timed repetitions after one warm-up; it includes system construction
and fitting, but excludes data generation and penalty searches. BLAS used one
thread. These are single-seed sanity checks, not averages over independent data.

In Experiments 1–3, weak methods use translated tensor-product tests made from
`b_p(r) = (1-r^2)^p` for `|r|<1` and zero otherwise. Each factor is evaluated at
`(coordinate - center)/h`, where `h` is the stated support half-width. Tests
have peak value one; grid-based weak integrals use trapezoidal quadrature.
SINDy uses pointwise derivatives and has no test-function count `K`.
MSTLS searches `np.logspace(-4, 0, 50)` thresholds. A LASSO penalty shown below
is the only retained result for that method and noise level. Where several
penalties were tried, the retained one has the lowest *observed coefficient
error against known truth*; this is oracle selection for this seed, not a
practical tuning rule.

## Experiment 1: 2D grid

PDE: `u_t = 0.3 d_xx(u^2) - 0.8 d_xy(u^2) + d_yy(u^2)`.
The `64 x 64 x 32` grid covers `[-5,5]^2 x [0.5,2.5]`, so `n=131,072`.
There are `S=20` derivatives, `J=5` powers, and `100` coefficients.
Weak methods use `K=512 = 8 x 8 x 8` tests with exponents `(11,11,16)`,
support half-widths `(16,16,8)` grid cells, and center strides `(4,4,2)`.
The LASSO penalties here were each measured at one setting only; no 2D
penalty sweep was recorded. The zero-vector error is `1.73`.

| Noise | Method | LASSO penalty `rho_1` | Squared error | Runtime (s) |
| ---: | --- | ---: | ---: | ---: |
| 0 | SINDy (OLS) | — | 9.878746e+01 | 0.270994 |
| 0 | SINDy (LASSO) | 1e-4 | 1.256032 | 0.134706 |
| 0 | WSINDy (OLS) | — | 1.131562e+05 | 0.048332 |
| 0 | WSINDy (LASSO) | 1e-6 | 1.627932 | 0.121193 |
| 0 | WSINDy (MSTLS) | — | 1.663836e-05 | 0.124151 |
| 0 | WENDy (OLS) * | — | 9.686758e+03 | 13.496299 |
| 0 | WENDy (LASSO) | 1e-4 | 1.004261 | 20.026773 |
| 1 | SINDy (OLS) | — | 3.262772e+03 | 0.342833 |
| 1 | SINDy (LASSO) | 1e-4 | 1.873925 | 3.499125 |
| 1 | WSINDy (OLS) | — | 1.700983e+07 | 0.087451 |
| 1 | WSINDy (LASSO) | 1e-6 | 1.013724 | 0.206751 |
| 1 | WSINDy (MSTLS) | — | 2.328519e-03 | 0.114476 |
| 1 | WENDy (OLS) | — | 8.963744e+05 | 37.997836 |
| 1 | WENDy (LASSO) | 1e-4 | 0.131299 | 27.104080 |

`*` WENDy (OLS) returned an estimate at noise `0`, but stopped on its
normality test before reaching the fixed-point tolerance. WENDy-MLE has no
completed result: its zero-noise likelihood was undefined and both noise-1
variants exceeded the 600-second limit.

## Experiment 2: 3D grid, 4096 weak tests

PDE: `u_t = 0.3 d_xx(u^2) - 0.2 d_xy(u^2) + 0.1 d_xz(u^2) + 0.7 d_yy(u^2) - 0.16 d_yz(u^2) + d_zz(u^2)`.
The `32 x 32 x 32 x 16` grid covers `[-5,5]^3 x [0.5,2.5]`, so `n=524,288`.
There are `S=55` derivatives, `J=5` powers, and `275` coefficients.
Weak methods use `K=4096 = 8^4` tests with exponents `(16,16,16,28)`,
support half-widths `(8,8,8,4)` grid cells, and center strides `(2,2,2,1)`.
The LASSO penalty in each row is the lowest-error result among its completed
trials for that method and noise level. The zero-vector error is `1.6556`.

| Noise | Method | LASSO penalty `rho_1` | Squared error | Runtime (s) |
| ---: | --- | ---: | ---: | ---: |
| 0 | SINDy (OLS) | — | 4.411126e+04 | 3.062872 |
| 0 | SINDy (LASSO) | 1e-7 | 0.1588735 | 2.070837 |
| 0 | WSINDy (OLS) | — | 4.868710e+07 | 0.453482 |
| 0 | WSINDy (LASSO) | 1e-9 | 0.1594418 | 1.158624 |
| 0 | WSINDy (MSTLS) | — | 0.006125071 | 3.409501 |
| 1 | SINDy (OLS) | — | 1.095782e+06 | 3.231388 |
| 1 | SINDy (LASSO) | 1e-3 | 1.655600 | 1.980746 |
| 1 | WSINDy (OLS) | — | 8.186907e+08 | 0.434958 |
| 1 | WSINDy (LASSO) | 1e-4 | 1.655600 | 0.433031 |
| 1 | WSINDy (MSTLS) | — | 0.08238487 | 2.497880 |

WSINDy (MSTLS) selected all six true terms at noise `0`; at noise `1`, it
selected the three diagonal diffusion terms and missed the three mixed terms.

## Experiment 3: 3D iid points, ordinary versus debiased WSINDy

This uses the same 3D PDE, domain, `S=55`, `J=5`, and `K=4096` physical test
functions as Experiment 2, but draws `n=524,288` iid uniform space-time points.
Both methods form weak equations on the **same final 262,144 evaluation points**
using Monte Carlo weights. Ordinary WSINDy integrates the raw powers `U^j`.
Debiased WSINDy instead uses corrected powers from a box-kernel moving-average
pilot fitted on the independent first 262,144 points, with physical bandwidths
`(0.8,0.8,0.8,0.35)`. Thus the evaluation data and test functions are matched,
but debiased WSINDy additionally uses the training half. Noise scale uses the
clean solution's RMS on the `32 x 32 x 32 x 16` reference grid. Experiment 2
remains a different grid/quadrature design and is not directly comparable.
The current SINDy and WENDy implementations require a grid, so they are not
included in this paired iid-sample comparison.

The LASSO penalty is `lambda` in `||Y-X beta||_2^2/2 + lambda||beta||_1`
for each method's weak system. Among 25 tested log-spaced penalties from
`1e-4` to `0.2`, `lambda=0.2` gave the lowest coefficient error for both
methods and noise levels, using known truth on this seed. The zero-vector
error is `1.6556`. Runtime is the median of three fits after one warm-up;
ordinary WSINDy counts shared weak-system construction plus each regression,
while debiased WSINDy times complete calls including pilot fitting. Data
generation and penalty search are excluded.

| Noise | Method | LASSO penalty `lambda` | Squared error | Runtime (s) |
| ---: | --- | ---: | ---: | ---: |
| 0 | WSINDy, Monte Carlo (OLS) | — | 1.898895e+12 | 45.732935 |
| 0 | WSINDy, Monte Carlo (LASSO) | 0.2 | 1.655600 | 45.698647 |
| 0 | WSINDy, Monte Carlo (MSTLS) | — | 1.664912 | 47.217092 |
| 0 | Debiased WSINDy (OLS) | — | 8.871171e+11 | 54.615432 |
| 0 | Debiased WSINDy (LASSO) | 0.2 | 1.655600 | 53.463993 |
| 0 | Debiased WSINDy (MSTLS) | — | 71.946856 | 57.602692 |
| 1 | WSINDy, Monte Carlo (OLS) | — | 3.798131e+10 | 47.373257 |
| 1 | WSINDy, Monte Carlo (LASSO) | 0.2 | 1.655600 | 47.337400 |
| 1 | WSINDy, Monte Carlo (MSTLS) | — | 1.165900 | 49.389236 |
| 1 | Debiased WSINDy (OLS) | — | 1.079440e+12 | 56.455327 |
| 1 | Debiased WSINDy (LASSO) | 0.2 | 1.655600 | 56.432819 |
| 1 | Debiased WSINDy (MSTLS) | — | 62.563155 | 58.195397 |

All four LASSO fits are exactly zero. Ordinary WSINDy (MSTLS) selected one
spurious term at noise `0` and one true term at noise `1`; debiased WSINDy
(MSTLS) selected `1/6` and `2/6`
true terms plus `10` and `3` false positives. The correction did not improve
MSTLS coefficient selection on this seed. Monte Carlo integration remains noisy
even without observation noise, so these runs do not establish PDE recovery.

## Nonlinear viscous Burgers: paper-filter comparison

PDE: `u_t = 0.01 u_xx - 0.5 d_x(u^2) - u^3 + 2u^2 + 1`.
This is an adapted instance: periodic `x in [-1,1)`, `t in [0,1.5]`, and
`u(x,0)=0.5+0.7 sin(pi*x)+0.25 sin(2*pi*x+0.3)`. The [Messenger–Bortz paper](https://arxiv.org/pdf/2211.16000) does not
specify these initial/boundary data. A centered-difference, sparse-BDF
solution is the numerical truth; the grid and iid observations share it.

- Grid: `256 x 257` (`n=65792`); iid observations: `n=524288`, split equally into independent pilot and evaluation samples.
- `K=600` tests, support half-widths `(0.25,0.1875)` in `(x,t)`, support volume ratio `1/16`; translated paper bump `exp(9/(r^2-1))` on `|r|<1`.
- Library: spatial orders `0..6`, powers `u^0..u^6`; 43 nonzero columns after omitting positive derivatives of the constant. All sparse fits use MSTLS with `logspace(-4,0,100)` thresholds.
- Paper filter: sixth-difference estimate of `sigma`, then the paper's equation (5.6) per-axis box width (periodic space, reflected time). For iid pilots, the grid width is mapped to physical units with a floor giving 64 expected training neighbors in an interior box. This is an adaptation to iid data.
- `sigma_c=sqrt(0.01/3)=0.057735`. Each level has 3 independent seeds starting at `0`. Entries are mean squared coefficient error, support recovery count, mean relative system residual at the true coefficients, and median complete fit runtime (seconds). Data generation and width selection are excluded from runtime. OLS has no sparse support score.

| sigma/sigma_c | estimated sigma | filter width | iid pilot bandwidths (x,t) | pilot MSE |
| ---: | ---: | ---: | --- | ---: |
| 0 | 3.355e-05 | 1 | (0.015625, 0.011719) | 4.363e-05 |
| 1 | 0.05776 | 5 | (0.015625, 0.011719) | 9.838e-05 |
| 2 | 0.1155 | 9 | (0.03125, 0.023438) | 0.000326 |

### Grid methods

| sigma/sigma_c | Method | Squared error | Support | True-system residual | Runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | SINDy (OLS) | 0.00013701 | — | 0.01799 | 0.053 |
| 0 | SINDy (MSTLS) | 0.000911769 | 3/3 | 0.01799 | 0.532 |
| 0 | WSINDy (OLS) | 2.59235e-11 | — | 0.00102 | 0.020 |
| 0 | WSINDy (MSTLS) | 4.49928e-06 | 3/3 | 0.00102 | 0.062 |
| 0 | Paper filtered WSINDy (MSTLS) | 4.49928e-06 | 3/3 | 0.00102 | 0.022 |
| 1 | SINDy (OLS) | 26.8137 | — | 3.335 | 0.053 |
| 1 | SINDy (MSTLS) | 50.1382 | 0/3 | 3.335 | 1.403 |
| 1 | WSINDy (OLS) | 30.2943 | — | 0.0249 | 0.019 |
| 1 | WSINDy (MSTLS) | 0.789354 | 0/3 | 0.0249 | 0.051 |
| 1 | Paper filtered WSINDy (MSTLS) | 0.0719707 | 0/3 | 0.02707 | 0.049 |
| 2 | SINDy (OLS) | 30.244 | — | 3.61 | 0.052 |
| 2 | SINDy (MSTLS) | 30.9714 | 0/3 | 3.61 | 1.446 |
| 2 | WSINDy (OLS) | 177.693 | — | 0.05133 | 0.019 |
| 2 | WSINDy (MSTLS) | 2.53096 | 0/3 | 0.05133 | 0.043 |
| 2 | Paper filtered WSINDy (MSTLS) | 7.33972 | 0/3 | 0.05643 | 0.047 |

### Independent-point Monte Carlo methods

| sigma/sigma_c | Method | Squared error | Support | True-system residual | Runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Sampled WSINDy (MSTLS) | 17.0592 | 0/3 | 0.262 | 2.292 |
| 0 | Sampled plug-in WSINDy (MSTLS) | 16.7601 | 0/3 | 0.263 | 4.136 |
| 0 | Sampled debiased WSINDy (MSTLS) | 17.0596 | 0/3 | 0.262 | 4.153 |
| 1 | Sampled WSINDy (MSTLS) | 21.72 | 0/3 | 0.262 | 2.380 |
| 1 | Sampled plug-in WSINDy (MSTLS) | 21.2095 | 0/3 | 0.2639 | 4.252 |
| 1 | Sampled debiased WSINDy (MSTLS) | 16.8183 | 0/3 | 0.2619 | 4.255 |
| 2 | Sampled WSINDy (MSTLS) | 14.4543 | 0/3 | 0.2631 | 2.405 |
| 2 | Sampled plug-in WSINDy (MSTLS) | 14.2042 | 0/3 | 0.2693 | 7.235 |
| 2 | Sampled debiased WSINDy (MSTLS) | 9.75798 | 0/3 | 0.2622 | 7.306 |

Grid and iid errors are separate comparisons because their observation locations and quadrature differ. The paper reports 200 noise realizations; these runs are a smaller numerical check, not a reproduction of its figure.

The noisy grid runs did not recover the exact five-term support in any of the
three seeds: MSTLS dropped the small `0.01 u_xx` coefficient. At
`sigma/sigma_c=1`, raw grid squared errors were `0.0484, 2.2640, 0.0557` and
filtered errors were `0.0662, 0.0751, 0.0746`. At `sigma/sigma_c=2`, the
filtered errors were `0.0898, 0.1093, 21.8201`; the mean in the table is
driven by the third seed. Thus filtering was not consistently better on this
adapted, lower-resolution trajectory.

For a separate noise-free iid diagnostic with seed `0`, increasing the total
independent-point count gave:

| Total iid `n` | Evaluation points | `||Y-X beta_true|| / ||Y||` |
| ---: | ---: | ---: |
| 131,072 | 65,536 | 0.4885 |
| 524,288 | 262,144 | 0.2894 |
| 2,097,152 | 1,048,576 | 0.1435 |

The shrinking residual supports Monte Carlo integration error as a major limit
for the iid comparison. The pilot cannot remove error in the shared weak-form
target `Y`; the grid and iid tables should not be ranked against each other.

Reproduce the main comparison with:

```sh
OPENBLAS_NUM_THREADS=1 python experiments.py --instance nonlinear_viscous_burgers --noise-multipliers 0 1 2 --replicates 3 --n-observations 524288 --append
```
