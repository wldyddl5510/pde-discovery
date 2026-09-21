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

## WENDy and WENDy-MLE

Install the updated dependencies with `python -m pip install -r requirements.txt`.
SciPy supplies sparse matrices, Cholesky solves, the normality test, and nonlinear
optimization. Both new estimators return the same coefficient vector as WSINDy
and accept its library and test-function settings. Their common observation
Jacobian is

```text
r(beta) = y - X beta,
A(beta, U)[k,i] = -d_t(phi_k)(z_i) * Delta_z
                 - sum_{alpha,j} beta[alpha,j] * (-1)^|alpha|
                   * d^alpha(phi_k)(z_i) * j * U_i^(j-1) * Delta_z,
C(beta) = (1-alpha) * A(beta,U) A(beta,U).T + alpha * I.
```

The scalar `alpha` in the covariance is the relaxation parameter; the multi-index
in the sum labels a spatial derivative. Off-diagonal covariance entries are
retained, including correlations between overlapping tests. No clean solution
or true coefficient is used to construct `A`.

```python
from methods import wendy, wendy_mle

inputs = (data.u_observed, data.spatial_grid, data.time)

beta_wendy_lasso = wendy(*inputs, regression="lasso", rho_1=1e-4)
beta_wendy_mstls = wendy(*inputs, regression="mstls")

# A positive observation noise std is required for the MLE objective.
# Here the synthetic experiment supplies its known noise scale.
beta_mle_lasso = wendy_mle(
    *inputs, noise_std=data.noise_std, regression="lasso", rho_1=1e-3,
)
beta_mle_mstls = wendy_mle(
    *inputs, noise_std=data.noise_std, regression="mstls",
)
```

These penalties illustrate the API and are not tuned for coefficient recovery.
The loss scales of WSINDy, WENDy, and WENDy-MLE differ.

### WENDy-IRLS

`wendy` implements the draft's covariance-weighted iteration, following
[Bortz, Messenger and Dukic (2023)](https://arxiv.org/abs/2302.13271).
At each iteration it freezes `C` at the current coefficients, whitens `X, y`
with a Cholesky factor, and fits the resulting linear system:

- `regression="lasso"` minimizes `r.T C^-1 r / K + rho_1 * ||beta||_1`.
  `rho_1=0` gives ordinary GLS on that iteration.
- `regression="mstls"` applies the existing MSTLS bounds, refits, and threshold
  selection to that iteration's whitened system. The covariance is then updated.

These are sparsity extensions of the draft's nonsparse IRLS, not an assertion
that sparse IRLS globally minimizes a single beta-dependent objective.
The initialization is the corresponding WSINDy fit. `initial_beta` can supply
another starting point in the original coefficient units.

Defaults are `alpha=1e-10`, `max_reweights=100`, and `reweight_tol=1e-6`.
The relative fixed-point change uses `max(||beta_old||, 1e-12)` as its denominator.
As in equation (18) of the paper, a Shapiro-Wilk check is also enabled after
10 reweights, with `normality_tol=1e-4`. Stopping on that check emits a warning
that fixed-point convergence was not reached. `normality_tol=None` disables
this rule. Exhausting the iteration budget raises `RuntimeError`.
`max_iter` and `tol` control each inner regression. With `alpha=1`, WENDy
reduces to the corresponding WSINDy fit.

### WENDy-MLE

`wendy_mle` uses the draft's first-order Gaussian weak-residual likelihood,
as developed in [Rummel et al. (2025)](https://arxiv.org/abs/2502.08881):

```text
[logdet(C(beta)) + r(beta).T C(beta)^-1 r(beta) / noise_std^2] / (2*K)
    + rho_1 * ||beta||_1.
```

Terms constant in `beta` are omitted. Dividing by `K` specifies the penalty
convention; it leaves unpenalized MLE unchanged. The default `alpha=0` matches
the draft; a positive value explicitly regularizes the covariance. This is an
approximate likelihood of weak residuals, not the exact observation likelihood.

`noise_std` is required and must be strictly positive. At zero observation
variance the Gaussian objective above is undefined; the implementation raises
an error instead of silently introducing a noise floor. A positive working
variance for a noise-free experiment must be supplied explicitly and reported
as a modeling choice.

For `regression="lasso"`, the nonlinear objective uses positive/negative parts
of the coefficients and L-BFGS-B to enforce the L1 penalty. With `rho_1=0`, it
uses BFGS without sparsity. The gradient differentiates both covariance terms,
including `logdet(C)`; it does not freeze covariance during optimization.
The default `max_iter=1000` limits iterations per nonlinear fit. With
`tol=1e-6`, the returned coefficients must satisfy the scaled KKT test:
multiply each violation in original units by `||y||/||X_j||` and bound its
maximum by `tol` (zero norms use the floors documented in the code).
These controls also apply to LASSO initialization. For `alpha=1`, the
likelihood reduces exactly to a scaled OLS/LASSO problem, which is solved by
the existing linear solver and its original-coordinate tolerance.

`regression="mstls"` is an explicitly defined **extension** of MSTLS to this
nonlinear objective:

1. Obtain a dense MLE and freeze its whitening for threshold bounds and the
   prediction-distance-plus-support-size selection score.
2. For each candidate threshold, remove terms failing those bounds and
   re-optimize the full beta-dependent likelihood on the remaining support.
   Repeat until the support stops changing.
3. Select the smallest threshold minimizing the common selection score.

Each refit updates covariance inside the likelihood; it is not a GLS refit.
The default candidate grid is the same 50 thresholds used by WSINDy;
`thresholds=(0.05,)` specifies one. `rho_1` must be zero in MSTLS mode.
This extension is not claimed as an algorithm proposed in the WENDy-MLE paper.

MLE is nonconvex: stationarity does not guarantee a global optimum, and a failed
stationarity check raises `RuntimeError`. `initial_beta` allows another starting
point. The full 100-term porous-medium library can be difficult for both IRLS
and MLE; success on smaller identifiable systems is not a recovery guarantee
for that library. In particular, sparse IRLS can fail to reach a fixed point.

Covariance methods cost more than WSINDy. The implementation stores local
support indices and sparse `A`, and evaluates the likelihood gradient in blocks,
avoiding a dense `(100, K, N)` derivative tensor. Covariance itself is a full
`K x K` matrix. Use `strides` to control the number of test functions; sufficient
independent rows are still needed for the unpenalized initialization.

### Experiment runner

`experiments.py --methods` also accepts `wendy-lasso`, `wendy-mstls`,
`wendy-mle-lasso`, and `wendy-mle-mstls`. Use `--noise-ratios 1` when comparing
MLE at the generator's known noise level. Alternatively, `--mle-noise-std`
sets an explicit positive working value. The runner records that choice.
The five existing methods remain the default selection.

Additional controls are `--wendy-rho-1`, `--wendy-mle-rho-1`, `--wendy-alpha`,
`--wendy-mle-alpha`, `--max-reweights`, `--reweight-tol`,
`--disable-normality-stop`, `--thresholds`, `--half-widths`, `--strides`, and
`--test-degrees`. The chosen settings are included in generated reports and
their reproduction commands. The existing `results.md` records the earlier
five-method experiment; adding estimators does not provide new benchmark values.

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
WENDy checks cover the full 100-column observation Jacobian, off-diagonal
covariance, independent likelihood-gradient finite differences, GLS and MLE KKT
conditions, both sparsity options on a noisy 2D diffusion equation, reduction
to WSINDy at identity covariance, and explicit stopping/failure behavior.
