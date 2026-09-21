# pde-discovery

Synthetic data and numerical comparisons for PDE coefficient estimation.

## Data generation

Install the dependency with `python -m pip install -r requirements.txt`.

```python
from simulation_generation import generate_anisotropic_porous_medium

data = generate_anisotropic_porous_medium(
    nx=64, ny=64, nt=32, noise_ratio=0.05, seed=0
)
x, y = data.spatial_grid
t = data.time
u_true = data.u_true          # Shape: (nx, ny, nt).
u_observed = data.u_observed  # Same clean solution plus iid Gaussian noise.
sigma = data.noise_std       # 0.05 * RMS(u_true).
```

The first experiment is the two-dimensional anisotropic porous medium equation

```text
u_t = 0.3 * d_xx(u^2) - 0.8 * d_xy(u^2) + d_yy(u^2).
```

The generator evaluates the exact Barenblatt weak solution from
[Messenger and Bortz (2021), Appendix B.5](https://pmc.ncbi.nlm.nih.gov/articles/PMC8570254/).
Its default observation domain is `[-5, 5]^2`, with times from `0.5` to `2.5`.
The initial profile is the solution at `t=0.5`; it stays zero on the default
spatial boundary. No PDE solver or data files are needed. For the paper's grid,
use `nx=200, ny=200, nt=128`. Grid endpoints are included.

`anisotropic_porous_medium_solution(x, y, t)` also evaluates the solution at
arbitrary broadcastable coordinates with positive times. Custom bounds in the
generator select an observation window, without changing the underlying PDE.

Each experiment has its own `generate_<experiment>` function in
`simulation_generation.py` and returns a `SimulationData` object. Its
`true_coefficients` maps `(spatial_derivative, polynomial_power)` to the true
coefficient: `((1, 1), 2)` means `d_xy(u^2)`. Unlisted terms have coefficient zero.
This ground-truth mapping is for evaluation; it does not restrict the library
used by an estimator. Additional experiments can reuse `add_gaussian_noise`.

## SINDy

```python
from methods import polynomial_library_terms, sindy

beta = sindy(data.u_observed, data.spatial_grid, data.time, rho_1=0.0)
beta_lasso = sindy(
    data.u_observed, data.spatial_grid, data.time, regression="lasso", rho_1=1e-4
)
terms = polynomial_library_terms(spatial_dim=2)
coefficient_by_term = dict(zip(terms, beta))
estimated_cross_diffusion = coefficient_by_term[((1, 1), 2)]
```

The default library includes every mixed spatial derivative of total order 1
through 5, applied to each of `u, u^2, ..., u^5`. In two spatial dimensions this
gives 100 coefficients. With `regression="lasso"` (the default), the objective is

```text
||y - X beta||_2^2 / N + rho_1 * ||beta||_1,
```

where `N` is the number of retained interior grid points. `rho_1=0` (the default)
uses ordinary least squares; a positive value uses LASSO. The L1 penalty applies
to coefficients in the original units, not standardized coefficients. There is
no zeroth derivative, intercept, or smoothing. The LASSO mode has no separate
thresholding step; `regression="mstls"` selects sequential thresholding instead,
as described below.
The input is the observed data and its coordinates; the estimator does not
access `u_true` or `true_coefficients`.

`polynomial_library_terms` records the coefficient order. Derivatives are sorted
by total order, starting with `(1, 0), (0, 1), (2, 0), (1, 1), (0, 2), ...`.
For each derivative, polynomial powers run from 1 to 5. The same optional
`max_derivative_order` and `max_polynomial_degree` arguments are accepted by
the library helper and `sindy`.

SINDy uses centered finite differences that are second-order accurate for
smooth functions. It differentiates each power of the observations directly.
All columns are fitted on a common interior grid: the defaults remove three
points at each end of each spatial axis and one at each time end. Coordinates
must be uniformly spaced. `build_sindy_system` exposes the resulting `X, y`
for inspection. Ordinary least squares uses column normalization and returns
coefficients in the original units. With `rho_1=0`, a numerically rank-deficient
library raises `LinAlgError` because a unique coefficient vector cannot be
identified.

LASSO uses cyclic coordinate descent on the original columns and also accepts
rank-deficient or underdetermined systems. Such systems can have multiple LASSO
minimizers. `max_iter=10000` and `tol=1e-8` control convergence: the maximum KKT
violation must be at most `tol * max(1, rho_1)`. Failure to converge raises
`RuntimeError`; `max_iter` can be increased for strongly correlated libraries.

The porous medium reference solution has a nonsmooth moving front. Successful
least-squares optimization on that data does not imply accurate PDE coefficient
recovery, even without observation noise. Tests therefore check coefficient
recovery on smooth exact solutions separately from the porous medium pipeline.

## WSINDy

```python
from methods import build_wsindy_system, wsindy

beta_weak = wsindy(data.u_observed, data.spatial_grid, data.time, rho_1=0.0)
beta_weak_lasso = wsindy(
    data.u_observed, data.spatial_grid, data.time, regression="lasso", rho_1=1e-6
)
beta_weak_mstls = wsindy(
    data.u_observed, data.spatial_grid, data.time, regression="mstls"
)

# Optional explicit test-function settings for the 64 x 64 x 32 example.
X_weak, y_weak = build_wsindy_system(
    data.u_observed, data.spatial_grid, data.time,
    half_widths=(16, 16, 8),  # Support radii in grid cells: x, y, time.
    strides=(4, 4, 2),       # Distances between successive test centers.
    test_degrees=(11, 11, 16),
)
```

`wsindy` accepts the same observations, coordinates, library limits,
`regression`, `rho_1`, `thresholds`, `max_iter`, and `tol` as `sindy`, and returns
coefficients in the same order. The default two-dimensional library still has
all 100 terms. In the default LASSO mode it solves

```text
||y_weak - X_weak beta||_2^2 / K + rho_1 * ||beta||_1,

y_weak[k]            = -integral(d_t(phi_k) * u),
X_weak[k, (alpha,j)] = (-1)^|alpha| * integral(d^alpha(phi_k) * u^j),
```

where `K` is the number of test functions. Every derivative acts analytically
on the test function, including the time derivative. The integrals use
tensor-product trapezoidal quadrature with the actual coordinate spacings.
Direct separable sums avoid allocating a dense test-function-by-data matrix.
Only supports fully inside the data are used; there is no padding or periodic
boundary assumption. Rows follow the grid of test centers in C order, with
time varying fastest. `build_wsindy_system` exposes these matrices separately.

The test functions are translated products of the compact polynomial bump
`b_p(r) = (1-r^2)^p` for `|r|<1`, zero elsewhere, from
[Messenger and Bortz (2021), Section 4.1.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC8570254/).
They have peak one. The optional settings have one entry per spatial axis,
followed by time:

- `half_widths`: integer radii `m >= 2`, giving `2*m+1` support points.
  Defaults to `max(2, axis_size//4)` and must fit within the grid.
- `strides`: positive integer steps between test centers, defaulting to
  `max(1, m//4)`. Center indices are `range(m, axis_size-m, stride)`.
- `test_degrees`: bump exponents, each greater than the maximum derivative
  order on its axis. By default, use the smallest such integer satisfying
  `(1-(1-1/m)^2)^p <= 1e-10`, the paper's equation (4.3).

Support defaults use a fixed grid-size rule; the paper's automatic Fourier-based
support selection is not implemented. The regression can use the draft's weak
least squares with optional LASSO, or the paper's MSTLS algorithm below.
Support widths, degrees, and center spacings are experiment parameters. Narrow
or poorly resolved test functions can produce substantial quadrature error,
especially for higher derivatives.

All regression modes share the SINDy solvers and their rank/convergence checks. The L1
penalty uses the original coefficients, and weak equations are not normalized
row by row. Changing test-function support or amplitude changes the loss scale;
using the same `rho_1` in SINDy and WSINDy does not mean equal regularization
strength. The example penalties above are usage examples, not tuned values.
Weak integration also leaves the bias in nonlinear powers of noisy observations
uncorrected. A small regression residual need not imply accurate coefficients
in the full, highly correlated polynomial library.

## Sequential thresholding (MSTLS)

Both estimators accept `regression="lasso"` or `regression="mstls"`. The default
remains `"lasso"` with `rho_1=0`, so existing calls still perform ordinary least
squares. A positive `rho_1` adds the L1 penalty only in LASSO mode.

`regression="mstls"` implements the **modified sequential-thresholding least
squares** algorithm in
[Messenger and Bortz (2021), equations (4.4)-(4.7)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8570254/).
The same rule is available on the strong-form SINDy system for comparison.
For each candidate threshold `lambda`, begin with full-library OLS. Keep term
`j` only when its coefficient satisfies

```text
lambda * max(1, ||y||_2 / ||X_j||_2)
    <= |beta_j| <=
min(1, ||y||_2 / ||X_j||_2) / lambda.
```

Refit OLS on the retained terms and repeat until the support stops changing.
The rule checks both coefficient magnitude and each term's contribution
relative to the target. Coefficients are not soft-thresholded. Empty supports
return the zero model, following the paper's equations.

By default, `thresholds=None` evaluates `np.logspace(-4, 0, 50)`, as in the
paper's Section 5.2, and chooses the smallest threshold minimizing

```text
||X(beta_lambda - beta_OLS)||_2 / ||X beta_OLS||_2
    + number_of_nonzero_coefficients / number_of_library_terms.
```

Use `thresholds` to specify another positive candidate set, or a single value:

```python
beta_fixed = wsindy(
    data.u_observed, data.spatial_grid, data.time,
    regression="mstls", thresholds=(0.01,),
)
```

`rho_1` must remain zero in MSTLS mode; supplying a positive penalty raises
`ValueError`. Conversely, `thresholds` is rejected in LASSO mode.
`max_iter` limits refits per candidate threshold and failure to stabilize raises
`RuntimeError`; `tol` applies only to LASSO. MSTLS uses the same full-rank OLS
requirement as the nonsparse estimator. An all-zero target returns zero directly.

All bounds use coefficients and column norms in the original units. Numerical
column scaling for the OLS solve is undone before thresholding. The paper's
separate rescaling of the data and coordinates is not implemented, so enabling
MSTLS alone does not reproduce its complete experimental pipeline. MSTLS is a
threshold-and-refit heuristic, not a solver for the LASSO objective.

## Checks

```sh
python -m unittest discover -s tests -v
```

WSINDy checks include all 100 weak columns against independent continuous
integrals, coefficient recovery on smooth one- and two-dimensional PDEs,
weak-residual convergence across the porous-medium moving front, and the
OLS/LASSO optimality conditions on clean and noisy porous-medium observations.
MSTLS checks cover both threshold bounds, successive support reductions,
refitting, the threshold-selection loss and tie rule, and both public estimators.
