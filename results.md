# PDE-discovery sanity check

WENDy (OLS) and WENDy-MLE (unpenalized) replace their MSTLS variants and were
rerun on the same data and settings. The other seven columns retain their
previous measurements. WENDy and WENDy-MLE use no thresholding in this comparison.
See [METHODS.md](METHODS.md) for estimator definitions.

## Experiment settings

- Instance: `anisotropic_porous_medium`; exact 2D anisotropic porous-medium weak solution.
- PDE: `u_t = 0.3 d_xx(u^2) - 0.8 d_xy(u^2) + d_yy(u^2)`.
- Grid: `(64, 64, 32)` on `[-5, 5]^2`, with time in `[0.5, 2.5]`; endpoints included.
- Library: all mixed spatial derivatives of total order 1 through 5, applied to
  powers 1 through 5: 100 coefficients, in `polynomial_library_terms(2)` order.
- One Gaussian-noise realization per noise ratio, seed `0`;
  `noise_std = noise_ratio * RMS(u_true)`. Each method receives the same observations.
- Error: `sum((beta_hat - beta_true)**2)` over all 100 coefficients, with no
  relative normalization. Truth is used only for evaluation; its squared norm is 1.73.
- LASSO: SINDy `rho_1=0.0001`; WSINDy `rho_1=1e-06`;
  `max_iter=200000`, `tol=1e-08`. These are fixed example penalties, not tuned values.
- OLS: `rho_1=0`. WSINDy MSTLS candidates: `np.logspace(-4, 0, 50)`.
- Weak methods: half-widths `(16, 16, 8)`, strides `(4, 4, 2)`, bump exponents `(11, 11, 16)`;
  peak-one test functions and physical quadrature weights, with no row normalization.
- Runtime: median of 3 timed calls after one untimed warm-up per method
  and noise level. Each call includes library/weak-system construction and regression;
  data generation, imports, error calculation, and report writing are excluded.
- Environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3,
  macOS-15.7.3-arm64-arm-64bit-Mach-O; `OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1`.

- Wall-time limit: 600 seconds per estimator call, including warm-up.

- WENDy: LASSO `rho_1=0.0001`, OLS `rho_1=0`; `alpha=1e-10`,
  `max_reweights=100`, `reweight_tol=1e-06`,
  normality stopping enabled (p<1e-4 after 10 reweights).
- WENDy-MLE: LASSO `rho_1=0.001`, unpenalized `rho_1=0`; `alpha=0`;
  `max_iter=1000`, `tol=1e-06` (scaled KKT tolerance).
  noise std: known generator noise std for each noise level.
- WENDy (OLS) fits unpenalized least squares on each whitened system (GLS/IRLS).
  WENDy-MLE (unpenalized) minimizes the full Gaussian weak-residual likelihood,
  including the log determinant, with no L1 penalty. Neither uses thresholding.
  Both nonsparse fits start from full-library WSINDy (OLS).

## Noise ratio 0: Squared coefficient error

| Experiment instance | SINDy (OLS) | WSINDy (OLS) | SINDy (LASSO) | WSINDy (LASSO) | WSINDy (MSTLS) | WENDy (LASSO) | WENDy (OLS) | WENDy-MLE (LASSO) | WENDy-MLE (unpenalized) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D anisotropic porous medium | 9.878746e+01 | 1.131562e+05 | 1.256032e+00 | 1.627932e+00 | 1.663836e-05 | 1.004261e+00 | 9.686758e+03 * | N/A | N/A |

## Noise ratio 1: Squared coefficient error

| Experiment instance | SINDy (OLS) | WSINDy (OLS) | SINDy (LASSO) | WSINDy (LASSO) | WSINDy (MSTLS) | WENDy (LASSO) | WENDy (OLS) | WENDy-MLE (LASSO) | WENDy-MLE (unpenalized) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D anisotropic porous medium | 3.262772e+03 | 1.700983e+07 | 1.873925e+00 | 1.013724e+00 | 2.328519e-03 | 1.312987e-01 | 8.963744e+05 | TIMEOUT | TIMEOUT |

## Noise ratio 0: Runtime (seconds)

| Experiment instance | SINDy (OLS) | WSINDy (OLS) | SINDy (LASSO) | WSINDy (LASSO) | WSINDy (MSTLS) | WENDy (LASSO) | WENDy (OLS) | WENDy-MLE (LASSO) | WENDy-MLE (unpenalized) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D anisotropic porous medium | 0.270994 | 0.048332 | 0.134706 | 0.121193 | 0.124151 | 20.026773 | 13.496299 * | N/A | N/A |

## Noise ratio 1: Runtime (seconds)

| Experiment instance | SINDy (OLS) | WSINDy (OLS) | SINDy (LASSO) | WSINDy (LASSO) | WSINDy (MSTLS) | WENDy (LASSO) | WENDy (OLS) | WENDy-MLE (LASSO) | WENDy-MLE (unpenalized) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D anisotropic porous medium | 0.342833 | 0.087451 | 3.499125 | 0.206751 | 0.114476 | 27.104080 | 37.997836 | TIMEOUT (600.041 s) | TIMEOUT (600.017 s) |

## Fit status

`*` marks a returned estimate with a warning; see the stop reason below.
`FAIL` and `TIMEOUT` mark incomplete benchmarks. Their parenthesized
times measure the failed call, not a median successful-fit runtime.
`N/A` means the sigma=0 MLE objective is undefined; no fit was attempted.

- Noise ratio 0, WENDy (OLS): WENDy stopped on the normality test after 11 reweights (p=2.24e-17); the fixed-point tolerance was not met.
- Noise ratio 0, WENDy-MLE (LASSO): The sigma=0 Gaussian likelihood is undefined; no working variance was supplied.
- Noise ratio 0, WENDy-MLE (unpenalized): The sigma=0 Gaussian likelihood is undefined; no working variance was supplied.
- Noise ratio 1, WENDy-MLE (LASSO) (warm-up): Estimator exceeded the 600 s wall-time limit.
- Noise ratio 1, WENDy-MLE (unpenalized) (warm-up): Estimator exceeded the 600 s wall-time limit.

## Interpretation

This is one fixed-grid sanity check with one seed, not a Monte Carlo comparison.
The zero coefficient vector has squared error 1.73, so errors above 1.73 are
worse than that reference on this coefficient metric. The solution has a
nonsmooth moving front and the full library is highly correlated; solving the
regression accurately does not by itself ensure accurate coefficient recovery.
LASSO penalties act on differently scaled losses, so the chosen
penalties do not represent equal regularization strength across methods.
Runtime covers this implementation, including all configured MSTLS candidates.
Repeated timings reuse the same data; they are not independent noise trials.
A timeout describes the implementation under the stated compute budget; it
does not establish nonconvergence or an accuracy ranking for that method.

## Validation

All 51 tests passed. The new nonsparse CLI options were also checked at identity
covariance: both matched WSINDy (OLS) and retained all 100 coefficients.

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python experiments.py --instance anisotropic_porous_medium \
  --nx 64 --ny 64 --nt 32 --seed 0 --repeats 3 \
  --noise-ratios 0 1 \
  --methods sindy-ols wsindy-ols sindy-lasso wsindy-lasso wsindy-mstls wendy-lasso wendy-ols wendy-mle-lasso wendy-mle \
  --sindy-rho-1 0.0001 --wsindy-rho-1 1e-06 \
  --max-iter 200000 --tol 1e-08 --output results.md \
  --timeout-seconds 600 \
  --wendy-rho-1 0.0001 --wendy-mle-rho-1 0.001 \
  --wendy-alpha 1e-10 --wendy-mle-alpha 0 \
  --max-reweights 100 --reweight-tol 1e-06 \
  --mle-max-iter 1000 --mle-tol 1e-06
```
