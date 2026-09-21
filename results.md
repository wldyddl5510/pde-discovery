# PDE-discovery sanity check

WENDy (OLS) and WENDy-MLE (unpenalized) replace their MSTLS variants and were
rerun on the same data and settings. The other seven columns retain their
previous measurements. WENDy and WENDy-MLE use no thresholding in this comparison.
See [METHODS.md](METHODS.md) for estimator definitions.

## Experiment settings

- Instance: `anisotropic_porous_medium`; exact 2D anisotropic porous-medium weak solution.
- PDE: `u_t = 0.3 d_xx(u^2) - 0.8 d_xy(u^2) + d_yy(u^2)`.
- Grid: `(64, 64, 32)` on `[-5, 5]^2`, with time in `[0.5, 2.5]`; endpoints included.
- **n = 131,072** observed space-time points: `64 * 64 * 32`;
  this counts the full input grid for every method, including spatial/time endpoints.
- **K = 512** test functions / weak equations: `8 * 8 * 8`
  centers along `(x, y, t)`. All WSINDy, WENDy, and WENDy-MLE variants use the same
  tests at every noise ratio. SINDy uses no test functions, so K does not apply to it.
- **S = 20** spatial derivative operators, using the draft's indexing:
  every multi-index `alpha=(a,b)` with `a,b >= 0` and `1 <= a+b <= 5`.
  The maximum total spatial derivative order is **5**; S counts operators.
- **J = 5** polynomial powers: `u, u^2, ..., u^5`.
  J describes powers of u, separately from the test-function exponents below.
- Library: `D^alpha(u^j)`, giving `S * J = 100` coefficients, in
  `polynomial_library_terms(2)` order. No zeroth spatial derivative or constant power is included.
- SINDy retains `58 * 58 * 30 = 100,920` regression rows
  after removing 3 points at each spatial end and 1 at each time end.
  Its design matrix is `100,920 x 100`; weak design matrices are `512 x 100`.
- One Gaussian-noise realization per noise ratio, seed `0`;
  `noise_std = noise_ratio * RMS(u_true)`. Each method receives the same observations.
- Error: `sum((beta_hat - beta_true)**2)` over all 100 coefficients, with no
  relative normalization. Truth is used only for evaluation; its squared norm is 1.73.
- LASSO: SINDy `rho_1=0.0001`; WSINDy `rho_1=1e-06`;
  `max_iter=200000`, `tol=1e-08`. These are fixed example penalties, not tuned values.
- OLS: `rho_1=0`. WSINDy MSTLS candidates: `np.logspace(-4, 0, 50)`.
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

## Test functions

Every weak method starts from the same reference function, a tensor product
of compact polynomial bumps (the WSINDy family):

```text
b_p(r) = (1-r^2)^p for |r| < 1, and 0 otherwise.
phi_ref(r_x,r_y,r_t) = b_11(r_x) * b_11(r_y) * b_16(r_t).
```

The reference function is centered at `(0,0,0)`, supported on `[-1,1]^3`,
and satisfies `phi_ref(0,0,0)=1`. Construct each physical test function by
scaling its coordinates and translating its center:

```text
Delta_x = 10/(nx-1), Delta_y = 10/(ny-1), Delta_t = 2/(nt-1).
h_axis = m_axis * grid_spacing_axis.
c_k = (-5 + i_x*Delta_x, -5 + i_y*Delta_y, 0.5 + i_t*Delta_t).
phi_k(x,y,t) = phi_ref((x-c_kx)/h_x, (y-c_ky)/h_y, (t-c_kt)/h_t).
```

Take every Cartesian-product combination of the center indices `(i_x,i_y,i_t)`
listed below, with time varying fastest. This gives `8 * 8 * 8 = 512` functions.
The reference function and support half-widths are fixed; only the center changes.
Each translated function has support `[c_kx-h_x,c_kx+h_x]` times
`[c_ky-h_y,c_ky+h_y]` times `[c_kt-h_t,c_kt+h_t]`, with peak value one.
There is no volume factor `1/(h_x*h_y*h_t)` or L2 normalization.

- Bump exponents `(p_x, p_y, p_t) = (11, 11, 16)` (one-dimensional polynomial degrees `(22, 22, 32)`).
- Support half-widths in grid cells: `(m_x, m_y, m_t) = (16, 16, 8)`;
  each support spans `(33, 33, 17)` grid points including endpoints.
  Physical half-widths `(h_x, h_y, h_t)` are approximately `(2.539683, 2.539683, 0.516129)`.
- Center strides in grid cells: `(4, 4, 2)`. Zero-based center indices are
  `x: range(16, 48, 4), y: range(16, 48, 4), t: range(8, 24, 2)` (range stops excluded), giving `(8, 8, 8)` centers and `K=512`.
- Default support rule: `m_axis=max(2, axis_size//4)`; default stride: `max(1, m_axis//4)`.
  These are fixed grid-size rules; the paper's Fourier-based support selection is not used.
- Default exponent rule: the smallest integer p greater than the derivative order on that axis
  (`5` in space, `1` in time) with `(1-(1-1/m)^2)^p <= 1e-10`.
  These polynomial bumps have finite smoothness, sufficient for the derivatives used here.
- Derivatives act analytically on the test functions. Integrals use tensor-product trapezoidal
  quadrature with the physical grid spacings; there is no normalization of individual equations.
  Only supports fully inside the observed grid are used, with no padding or periodic wrapping.

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
