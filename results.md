# PDE-discovery sanity check

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
- OLS: `rho_1=0`. MSTLS: automatic selection from `np.logspace(-4, 0, 50)`.
- WSINDy: half-widths `(16, 16, 8)`, strides `(4, 4, 2)`, bump exponents `(11, 11, 16)`;
  peak-one test functions and physical quadrature weights, with no row normalization.
- Runtime: median of 3 timed calls after one untimed warm-up per method
  and noise level. Each call includes library/weak-system construction and regression;
  data generation, imports, error calculation, and report writing are excluded.
- Environment: Python 3.13.5, NumPy 2.1.3,
  macOS-15.7.3-arm64-arm-64bit-Mach-O; `OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1`.

## Noise ratio 0: Squared coefficient error

| Experiment instance | SINDy (OLS) | WSINDy (OLS) | SINDy (LASSO) | WSINDy (LASSO) | WSINDy (MSTLS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D anisotropic porous medium | 9.878746e+01 | 1.131562e+05 | 1.256032e+00 | 1.627932e+00 | 1.663836e-05 |

## Noise ratio 1: Squared coefficient error

| Experiment instance | SINDy (OLS) | WSINDy (OLS) | SINDy (LASSO) | WSINDy (LASSO) | WSINDy (MSTLS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D anisotropic porous medium | 3.262772e+03 | 1.700983e+07 | 1.873925e+00 | 1.013724e+00 | 2.328519e-03 |

## Noise ratio 0: Runtime (seconds)

| Experiment instance | SINDy (OLS) | WSINDy (OLS) | SINDy (LASSO) | WSINDy (LASSO) | WSINDy (MSTLS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D anisotropic porous medium | 0.270831 | 0.048872 | 0.130062 | 0.123297 | 0.126980 |

## Noise ratio 1: Runtime (seconds)

| Experiment instance | SINDy (OLS) | WSINDy (OLS) | SINDy (LASSO) | WSINDy (LASSO) | WSINDy (MSTLS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D anisotropic porous medium | 0.335150 | 0.051915 | 3.547145 | 0.225179 | 0.124317 |

## Interpretation

This is one fixed-grid sanity check with one seed, not a Monte Carlo comparison.
The zero coefficient vector has squared error 1.73, so errors above 1.73 are
worse than that reference on this coefficient metric. The solution has a
nonsmooth moving front and the full library is highly correlated; solving the
regression accurately does not by itself ensure accurate coefficient recovery.
LASSO penalties act on differently scaled strong and weak losses, so the chosen
penalties do not represent equal regularization strength across the two methods.
Runtime covers this implementation, including all 50 MSTLS candidates.
Repeated timings reuse the same data; they are not independent noise trials.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python experiments.py --instance anisotropic_porous_medium \
  --nx 64 --ny 64 --nt 32 --seed 0 --repeats 3 \
  --noise-ratios 0 1 \
  --methods sindy-ols wsindy-ols sindy-lasso wsindy-lasso wsindy-mstls \
  --sindy-rho-1 0.0001 --wsindy-rho-1 1e-06 \
  --max-iter 200000 --tol 1e-08 --output results.md
```
