"""Estimate PDE coefficients from space-time observations.

Grid-based arrays have spatial axes first and time last, as in
simulation_generation.py. The debiased method takes independent point samples.
SINDy differentiates the observations; WSINDy differentiates test functions.
Both offer least squares with an optional LASSO penalty, or the WSINDy
paper's modified sequential-thresholding least squares (MSTLS).
WENDy iteratively weights weak residuals by their covariance; WENDy-MLE
optimizes their parameter-dependent Gaussian likelihood.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import factorial
import warnings

import numpy as np
from numpy.polynomial import Polynomial
from scipy import linalg, optimize, sparse, stats
from scipy.spatial import cKDTree


def polynomial_library_terms(
    spatial_dim: int,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    *,
    min_derivative_order: int = 1,
    min_polynomial_degree: int = 1,
) -> list[tuple[tuple[int, ...], int]]:
    """List (spatial_derivative, polynomial_power) in coefficient-vector order.

    Defaults retain the existing order: positive spatial derivatives and
    powers u through u^5. Set both minimum orders to zero and both maxima to
    six for the Burgers library. The constant u^0 is listed only with the
    zeroth derivative; its positive derivatives are identically zero.
    """
    if not isinstance(spatial_dim, (int, np.integer)) or spatial_dim < 1:
        raise ValueError("spatial_dim must be a positive integer.")
    for name, minimum, maximum in (
        ("derivative_order", min_derivative_order, max_derivative_order),
        ("polynomial_degree", min_polynomial_degree, max_polynomial_degree),
    ):
        if any(not isinstance(value, (int, np.integer)) for value in (minimum, maximum)):
            raise ValueError(f"{name} limits must be integers.")
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"{name} limits must satisfy 0 <= minimum <= maximum.")

    terms = []
    for total_order in range(min_derivative_order, max_derivative_order + 1):
        for axes in combinations_with_replacement(range(spatial_dim), total_order):
            derivative = tuple(axes.count(axis) for axis in range(spatial_dim))
            for power in range(min_polynomial_degree, max_polynomial_degree + 1):
                if total_order > 0 and power == 0:
                    continue
                terms.append((derivative, power))
    if not terms:
        raise ValueError("The requested library has no nonzero terms.")
    return terms


def _grid_spacing(grid, expected_size: int, name: str) -> float:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) != expected_size or len(grid) < 2:
        raise ValueError(f"{name} must be a 1D grid matching its data axis.")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain only finite coordinates.")

    differences = np.diff(grid)
    spacing = differences[0]
    if spacing <= 0 or not np.allclose(differences, spacing, rtol=1e-7, atol=0.0):
        raise ValueError(f"{name} must be increasing and uniformly spaced.")
    return float(spacing)


def _finite_difference(values, spacing: float, axis: int, order: int):
    """Centered, second-order-accurate derivative on the valid interior.

    Differentiate directly at the requested order instead of repeatedly
    taking first differences. Boundary entries are left zero; the caller
    removes them before fitting. No periodic boundary condition is assumed.
    """
    radius = (order + 1) // 2
    offsets = np.arange(-radius, radius + 1)

    # Match derivatives of monomials to obtain the centered stencil weights.
    moments = np.array([offsets.astype(float)**k for k in range(len(offsets))])
    target = np.zeros(len(offsets))
    target[order] = factorial(order)
    weights = np.linalg.solve(moments, target)
    weights = 0.5 * (weights + (-1)**order * weights[::-1])

    interior = [slice(None)] * values.ndim
    interior[axis] = slice(radius, values.shape[axis] - radius)
    derivative = np.zeros_like(values, dtype=float)
    for offset, weight in zip(offsets, weights):
        shifted = list(interior)
        shifted[axis] = slice(radius + offset, values.shape[axis] - radius + offset)
        derivative[tuple(interior)] += weight * values[tuple(shifted)]
    return derivative / spacing**order


def build_sindy_system(
    u: np.ndarray,
    spatial_grid: tuple[np.ndarray, ...],
    time: np.ndarray,
    *,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    min_derivative_order: int = 1,
    min_polynomial_degree: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X and y for u_t = sum beta[alpha, j] * d^alpha(u**j).

    u contains observations, not derivatives. For each column, take the
    polynomial power first and then apply the spatial derivatives. Columns
    follow polynomial_library_terms(). Rows use C-order flattening of the
    retained space-time grid, so time varies fastest.

    All columns use the same interior: remove (max_derivative_order + 1) // 2
    points from each end of each spatial axis, and one from each time end.
    """
    u = np.asarray(u, dtype=float)
    spatial_dim = len(spatial_grid)
    terms = polynomial_library_terms(
        spatial_dim, max_derivative_order, max_polynomial_degree,
        min_derivative_order=min_derivative_order,
        min_polynomial_degree=min_polynomial_degree,
    )
    if u.ndim != spatial_dim + 1:
        raise ValueError("u must have one axis per spatial coordinate, then time.")
    if not np.all(np.isfinite(u)):
        raise ValueError("u must contain only finite observations.")

    spatial_steps = []
    for axis, grid in enumerate(spatial_grid):
        spatial_steps.append(_grid_spacing(grid, u.shape[axis], f"spatial_grid[{axis}]"))
    time_step = _grid_spacing(time, u.shape[-1], "time")

    radius = (max_derivative_order + 1) // 2
    if any(size <= 2 * radius for size in u.shape[:-1]) or u.shape[-1] < 3:
        raise ValueError(
            f"Need at least {2 * radius + 1} points on each spatial axis and 3 time points."
        )
    interior = (slice(radius, -radius if radius else None),) * spatial_dim + (slice(1, -1),)

    time_derivative = _finite_difference(u, time_step, axis=spatial_dim, order=1)
    y = time_derivative[interior].ravel()
    X = np.empty((len(y), len(terms)))
    powers = [u**power for power in range(min_polynomial_degree, max_polynomial_degree + 1)]

    for column, (derivative, power) in enumerate(terms):
        feature = powers[power - min_polynomial_degree]
        for axis, order in enumerate(derivative):
            if order > 0:
                feature = _finite_difference(feature, spatial_steps[axis], axis, order)
        X[:, column] = feature[interior].ravel()

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Powers or finite differences overflowed; check the data scale.")
    return X, y


def _lasso(X, y, rho_1: float, max_iter: int, tol: float) -> np.ndarray:
    """Minimize mean squared residual + rho_1 * ||beta||_1.

    Cyclic coordinate descent uses the original columns, so the L1 penalty
    is on coefficients in their original units. The Gram matrix avoids
    looping over the full space-time dataset on each iteration.
    """
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")

    sample_count = len(y)
    gram = X.T @ X / sample_count
    cross_product = X.T @ y / sample_count
    diagonal = np.diag(gram)
    beta = np.zeros(X.shape[1])

    for _ in range(max_iter):
        for j in range(len(beta)):
            if diagonal[j] == 0:
                continue  # A zero column has optimal coefficient zero.
            partial_correlation = cross_product[j] - gram[j] @ beta
            partial_correlation += diagonal[j] * beta[j]
            # MSE has no factor 1/2, hence soft-threshold at rho_1 / 2.
            magnitude = max(abs(partial_correlation) - rho_1 / 2.0, 0.0)
            beta[j] = np.sign(partial_correlation) * magnitude / diagonal[j]

        # KKT conditions: gradient = -rho_1*sign(beta) for nonzero entries,
        # and |gradient| <= rho_1 for zero entries.
        gradient = 2.0 * (gram @ beta - cross_product)
        violation = np.maximum(np.abs(gradient) - rho_1, 0.0)
        nonzero = beta != 0
        violation[nonzero] = np.abs(gradient[nonzero] + rho_1 * np.sign(beta[nonzero]))
        if np.max(violation) <= tol * max(1.0, rho_1):
            return beta

    raise RuntimeError(
        f"LASSO did not converge in {max_iter} iterations "
        f"(maximum KKT violation {np.max(violation):.3g}); increase max_iter."
    )


def _least_squares(X, y) -> np.ndarray:
    """Solve identifiable least squares, undoing numerical column scaling."""
    if X.shape[0] < X.shape[1]:
        raise np.linalg.LinAlgError(
            f"Only {X.shape[0]} samples for {X.shape[1]} coefficients."
        )

    column_norms = np.linalg.norm(X, axis=0)
    if np.any(column_norms == 0):
        raise np.linalg.LinAlgError("The sampled library contains zero columns.")
    scaled_X = X / column_norms
    scaled_beta, _, rank, _ = np.linalg.lstsq(scaled_X, y, rcond=None)
    if rank < X.shape[1]:
        raise np.linalg.LinAlgError(
            f"The sampled library has rank {rank} for {X.shape[1]} coefficients."
        )
    return scaled_beta / column_norms


def _threshold_grid(thresholds):
    if thresholds is None:
        thresholds = np.logspace(-4, 0, 50)
    thresholds = np.asarray(thresholds, dtype=float)
    if (
        thresholds.ndim != 1 or thresholds.size == 0
        or not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0)
    ):
        raise ValueError("thresholds must be a nonempty 1D sequence of finite positive values.")
    return np.unique(thresholds)


def _mstls(X, y, thresholds, max_iter: int) -> np.ndarray:
    """Messenger & Bortz (2021), equations (4.4)-(4.7), in original units.

    For each lambda, start from full OLS, threshold, and refit OLS on the
    retained columns until the support stops changing. Retain coefficient j
    only if lambda*max(1, ||y||/||X_j||) <= |beta_j| <=
    min(1, ||y||/||X_j||)/lambda. No L1 shrinkage or ridge penalty is applied.

    Select the smallest lambda minimizing
      ||X(beta_lambda-beta_OLS)|| / ||X beta_OLS|| + nonzero(beta_lambda)/P.
    The default 50 thresholds are log-spaced from 1e-4 to 1 (paper Section 5.2).
    Supply a one-element sequence for a fixed threshold. Empty supports give
    the zero model, as in the paper's equations. Full OLS must be identifiable;
    a zero target returns zero directly, avoiding an undefined relative loss.
    """
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    thresholds = _threshold_grid(thresholds)  # Ascending order also resolves loss ties.

    if not np.any(y):
        return np.zeros(X.shape[1])
    beta_ols = _least_squares(X, y)
    projection_norm = np.linalg.norm(X @ beta_ols)
    if projection_norm == 0:
        return np.zeros(X.shape[1])
    norm_ratios = np.linalg.norm(y) / np.linalg.norm(X, axis=0)
    best_loss = np.inf
    best_beta = None

    for threshold in thresholds:
        lower = threshold * np.maximum(1.0, norm_ratios)
        upper = np.minimum(1.0, norm_ratios) / threshold
        beta = beta_ols.copy()
        active = np.ones(X.shape[1], dtype=bool)

        # max_iter counts refits; the last pass checks the resulting support.
        for iteration in range(max_iter + 1):
            retained = active & (np.abs(beta) >= lower) & (np.abs(beta) <= upper)
            if np.array_equal(retained, active):
                break
            if iteration == max_iter:
                raise RuntimeError(
                    f"MSTLS did not converge in {max_iter} refits "
                    f"at threshold {threshold:.6g}; increase max_iter."
                )
            beta = np.zeros(X.shape[1])
            if not np.any(retained):
                break
            beta[retained] = _least_squares(X[:, retained], y)
            active = retained

        loss = np.linalg.norm(X @ (beta - beta_ols)) / projection_norm
        loss += np.count_nonzero(beta) / X.shape[1]
        if loss < best_loss:
            best_loss = loss
            best_beta = beta.copy()
    return best_beta


def _fit_coefficients(
    X, y, rho_1: float, max_iter: int, tol: float,
    regression: str = "lasso", thresholds=None,
) -> np.ndarray:
    """Dispatch regression without mixing L1 penalties and hard thresholds."""
    if regression not in ("lasso", "mstls"):
        raise ValueError("regression must be 'lasso' or 'mstls'.")
    if regression == "mstls":
        if rho_1 != 0:
            raise ValueError("rho_1 must be 0 when regression='mstls'; use thresholds instead.")
        return _mstls(X, y, thresholds, max_iter)
    if thresholds is not None:
        raise ValueError("thresholds applies only to regression='mstls'.")
    if rho_1 > 0:
        return _lasso(X, y, rho_1, max_iter, tol)
    return _least_squares(X, y)


def sindy(
    u: np.ndarray,
    spatial_grid: tuple[np.ndarray, ...],
    time: np.ndarray,
    *,
    rho_1: float = 0.0,
    regression: str = "lasso",
    thresholds: tuple[float, ...] | np.ndarray | None = None,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    min_derivative_order: int = 1,
    min_polynomial_degree: int = 1,
    max_iter: int = 10000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Estimate beta from strong-form data using LASSO/OLS or MSTLS.

    regression='lasso' (default) minimizes
      ||y - X beta||_2^2 / N + rho_1 * ||beta||_1,
    where N is the number of retained interior grid points. rho_1=0 uses OLS;
    rho_1>0 uses LASSO coordinate descent. There is no intercept.

    regression='mstls' uses the WSINDy paper's modified sequential-thresholding
    algorithm, including its relative-fit-plus-support-size criterion for
    choosing a threshold. thresholds=None searches 50 log-spaced values in
    [1e-4, 1]; thresholds=(0.01,) fixes lambda=0.01. rho_1 must be zero.
    Thresholding and refitting iterate until the support stabilizes, with at
    most max_iter refits per threshold, otherwise raising RuntimeError.

    Pass observed values and coordinates only; no true coefficients or clean
    solution are used. In 2D the defaults return 100 coefficients, ordered by
    polynomial_library_terms(2). The penalty and returned coefficients always
    use the original units. Column normalization is used internally for OLS
    solves and undone before thresholding; it does not change MSTLS bounds.

    OLS (including MSTLS initialization) raises LinAlgError for a rank-deficient
    sampled library. MSTLS returns zero directly when the target is all zero.
    LASSO also supports rank-deficient and underdetermined systems; its
    coefficient minimizer need not be unique. For LASSO, max_iter and tol
    require maximum KKT violation <= tol * max(1, rho_1), otherwise raise
    RuntimeError after max_iter sweeps. tol is unused by MSTLS. No solver guarantees
    accurate coefficient recovery from noisy or nonsmooth data.
    """
    if not np.isfinite(rho_1) or rho_1 < 0:
        raise ValueError("rho_1 must be finite and nonnegative.")

    X, y = build_sindy_system(
        u, spatial_grid, time,
        max_derivative_order=max_derivative_order,
        max_polynomial_degree=max_polynomial_degree,
        min_derivative_order=min_derivative_order,
        min_polynomial_degree=min_polynomial_degree,
    )
    return _fit_coefficients(X, y, rho_1, max_iter, tol, regression, thresholds)


def _test_function_weights(half_width, spacing, degree, max_order, test_function="polynomial"):
    """Sample derivatives of b(r)=(1-r^2)^degree, including quadrature weights.

    r=(z-center)/(half_width*spacing). The test function has peak one and
    is zero outside |r|<1. degree>max_order makes all sampled derivatives
    vanish at the endpoints, so the trapezoidal endpoint weights are zero.
    """
    r = np.arange(-half_width, half_width + 1, dtype=float) / half_width
    radius = half_width * spacing
    if test_function == "paper":
        derivatives = _paper_bump_derivatives_at_points(
            r * radius, 0.0, radius, _paper_bump_factors(max_order)
        )
        return [spacing * derivatives[:, order] for order in range(max_order + 1)]
    base = Polynomial([1.0, 0.0, -1.0])
    factor = Polynomial([1.0])
    weights = []
    for order in range(max_order + 1):
        # b^(order)(r) = (1-r^2)^(degree-order) * factor(r).
        # Keeping this factorization avoids expanding a high-degree bump.
        derivative = (1.0 - r**2)**(degree - order) * factor(r)
        weights.append(spacing * derivative / radius**order)
        factor = base * factor.deriv() - Polynomial([0.0, 2.0 * (degree - order)]) * factor
    return weights


def _weak_integrals(values, weights, strides):
    """Integrate over translated tensor-product supports, one axis at a time.

    Only supports fully inside the observation grid are used. A row's center
    on axis d has index half_width[d] + k*strides[d]. No periodic wrapping
    or padding is applied. The final rows follow C order (time varies fastest).
    """
    for axis, (kernel, stride) in enumerate(zip(weights, strides)):
        count = (values.shape[axis] - len(kernel)) // stride + 1
        shape = list(values.shape)
        shape[axis] = count
        integrated = np.zeros(shape)
        selection = [slice(None)] * values.ndim
        for offset, weight in enumerate(kernel):
            selection[axis] = slice(offset, offset + count * stride, stride)
            integrated += weight * values[tuple(selection)]
        values = integrated
    return values.ravel()


def _prepare_weak_tests(
    u: np.ndarray,
    spatial_grid: tuple[np.ndarray, ...],
    time: np.ndarray,
    *,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    min_derivative_order: int = 1,
    min_polynomial_degree: int = 1,
    half_widths: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
    test_degrees: tuple[int, ...] | None = None,
    test_function: str = "polynomial",
):
    """Validate weak-form inputs and construct the shared test-function weights."""
    u = np.asarray(u, dtype=float)
    spatial_dim = len(spatial_grid)
    terms = polynomial_library_terms(
        spatial_dim, max_derivative_order, max_polynomial_degree,
        min_derivative_order=min_derivative_order,
        min_polynomial_degree=min_polynomial_degree,
    )
    if u.ndim != spatial_dim + 1:
        raise ValueError("u must have one axis per spatial coordinate, then time.")
    if not np.all(np.isfinite(u)):
        raise ValueError("u must contain only finite observations.")

    steps = []
    for axis, grid in enumerate(spatial_grid):
        steps.append(_grid_spacing(grid, u.shape[axis], f"spatial_grid[{axis}]"))
    steps.append(_grid_spacing(time, u.shape[-1], "time"))
    max_orders = (max_derivative_order,) * spatial_dim + (1,)
    if test_function not in ("polynomial", "paper"):
        raise ValueError("test_function must be 'polynomial' or 'paper'.")

    if half_widths is None:
        half_widths = tuple(max(2, size // 4) for size in u.shape)
    if len(half_widths) != u.ndim:
        raise ValueError("half_widths must have one entry per spatial axis and time.")
    for size, m in zip(u.shape, half_widths):
        if not isinstance(m, (int, np.integer)) or m < 2 or 2 * m + 1 > size:
            raise ValueError("Each half_width must be an integer >=2 with 2*m+1 <= axis size.")

    if strides is None:
        strides = tuple(max(1, m // 4) for m in half_widths)
    if len(strides) != u.ndim or any(
        not isinstance(s, (int, np.integer)) or s < 1 for s in strides
    ):
        raise ValueError("strides must contain one positive integer per spatial axis and time.")

    if test_function == "paper":
        if test_degrees is not None:
            raise ValueError("test_degrees must be omitted for the paper bump.")
        test_degrees = tuple(0 for _ in u.shape)
    elif test_degrees is None:
        degrees = []
        for m, order in zip(half_widths, max_orders):
            first_interior_value = (2.0 * m - 1.0) / m**2
            decay_degree = int(np.ceil(np.log(1e-10) / np.log(first_interior_value)))
            degrees.append(max(order + 1, decay_degree))
        test_degrees = tuple(degrees)
    if test_function == "polynomial" and (
        len(test_degrees) != u.ndim or any(
            not isinstance(p, (int, np.integer)) or p <= order
            for p, order in zip(test_degrees, max_orders)
        )
    ):
        raise ValueError("Each test_degree must be an integer greater than its axis derivative order.")

    # weights[axis][order] includes that coordinate's quadrature spacing.
    weights = []
    for m, step, p, order in zip(half_widths, steps, test_degrees, max_orders):
        weights.append(_test_function_weights(m, step, p, order, test_function))
    return u, terms, weights, strides


def build_wsindy_system(
    u: np.ndarray,
    spatial_grid: tuple[np.ndarray, ...],
    time: np.ndarray,
    *,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    min_derivative_order: int = 1,
    min_polynomial_degree: int = 1,
    half_widths: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
    test_degrees: tuple[int, ...] | None = None,
    test_function: str = "polynomial",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the weak system X, y from the draft's Section 2, equation (4).

    y[k] = -integral(d_t(phi_k) * u)
    X[k, (alpha, j)] = (-1)^|alpha| * integral(d^alpha(phi_k) * u**j)

    Integrals use tensor-product trapezoidal quadrature on the given uniform
    grid. Only test functions are differentiated. Columns have exactly the
    same order as build_sindy_system() and polynomial_library_terms().

    Each phi_k is a translate of a product of compact polynomial bumps
    (1-r_d^2)^p_d, |r_d|<1, with peak one. Parameters have one entry per
    spatial axis followed by time:
      half_widths: support radii in grid cells (2*m+1 points), each m>=2.
        Default max(2, axis_size//4); every support must fit in the data.
      strides: center spacings in grid cells; default max(1, m//4).
      test_degrees: exponents p_d greater than the largest derivative on
        that axis (max_derivative_order in space, 1 in time). By default,
        use the smallest such integer with (1-(1-1/m)^2)^p_d <= 1e-10.

    This is the bump family and degree rule in Messenger & Bortz (2021),
    equations (4.2)-(4.3). Support defaults are fixed grid-size heuristics,
    not their Fourier-based support selection. Narrow, poorly resolved bumps
    can give large quadrature error; support selection is an experiment parameter.
    """
    u, terms, weights, strides = _prepare_weak_tests(
        u, spatial_grid, time,
        max_derivative_order=max_derivative_order,
        max_polynomial_degree=max_polynomial_degree,
        min_derivative_order=min_derivative_order,
        min_polynomial_degree=min_polynomial_degree,
        half_widths=half_widths, strides=strides, test_degrees=test_degrees,
        test_function=test_function,
    )

    time_weights = [axis_weights[0] for axis_weights in weights[:-1]] + [weights[-1][1]]
    y = -_weak_integrals(u, time_weights, strides)
    X = np.empty((len(y), len(terms)))
    powers = [u**power for power in range(min_polynomial_degree, max_polynomial_degree + 1)]
    for column, (derivative, power) in enumerate(terms):
        spatial_weights = [weights[axis][order] for axis, order in enumerate(derivative)]
        feature_weights = spatial_weights + [weights[-1][0]]
        integrals = _weak_integrals(powers[power - min_polynomial_degree], feature_weights, strides)
        X[:, column] = (-1)**sum(derivative) * integrals

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Powers or weak integrals overflowed; check the data and test function scales.")
    return X, y


def wsindy(
    u: np.ndarray,
    spatial_grid: tuple[np.ndarray, ...],
    time: np.ndarray,
    *,
    rho_1: float = 0.0,
    regression: str = "lasso",
    thresholds: tuple[float, ...] | np.ndarray | None = None,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    min_derivative_order: int = 1,
    min_polynomial_degree: int = 1,
    half_widths: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
    test_degrees: tuple[int, ...] | None = None,
    test_function: str = "polynomial",
    max_iter: int = 10000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Estimate beta from weak-form data using LASSO/OLS or MSTLS.

    X, y come from build_wsindy_system(); K is the number of test functions.
    regression='lasso' (default) minimizes
      ||y - X beta||_2^2 / K + rho_1 * ||beta||_1.
    rho_1=0 uses OLS and rho_1>0 uses LASSO. regression='mstls' uses the original
    WSINDy paper's modified sequential-thresholding algorithm (Section 4.2).
    thresholds=None searches the paper's 50 log-spaced candidates in [1e-4, 1];
    pass a one-element sequence for a fixed threshold. rho_1 must be zero in
    this mode. The solvers, coefficient ordering, rank checks and convergence
    controls are shared with sindy(). No sparsity level or truth is supplied.

    Test functions have peak one, without row normalization. Consequently
    changing their support or amplitude changes the scale of the loss and
    the effective LASSO penalty. Equal rho_1 values in SINDy and WSINDy need
    not impose comparable regularization. WSINDy does not correct the bias
    in powers of noisy observations; smoothing and WENDy are separate methods.
    """
    if not np.isfinite(rho_1) or rho_1 < 0:
        raise ValueError("rho_1 must be finite and nonnegative.")
    X, y = build_wsindy_system(
        u, spatial_grid, time,
        max_derivative_order=max_derivative_order,
        max_polynomial_degree=max_polynomial_degree,
        min_derivative_order=min_derivative_order,
        min_polynomial_degree=min_polynomial_degree,
        half_widths=half_widths,
        strides=strides,
        test_degrees=test_degrees,
        test_function=test_function,
    )
    return _fit_coefficients(X, y, rho_1, max_iter, tol, regression, thresholds)


def _bump_derivative_factors(degree, max_order):
    """Polynomial factors in derivatives of (1-r^2)^degree."""
    base = Polynomial([1.0, 0.0, -1.0])
    factor = Polynomial([1.0])
    factors = []
    for order in range(max_order + 1):
        factors.append(factor)
        factor = base * factor.deriv() - Polynomial([0.0, 2.0 * (degree - order)]) * factor
    return factors


def _bump_derivatives_at_points(coordinates, center, radius, degree, factors):
    """Evaluate a compact bump and its derivatives at irregular coordinates."""
    scaled = (coordinates - center) / radius
    base = np.maximum(1.0 - scaled**2, 0.0)
    inside = np.abs(scaled) < 1.0
    return np.column_stack([
        np.where(inside, base**(degree - order) * factor(scaled) / radius**order, 0.0)
        for order, factor in enumerate(factors)
    ])


def _paper_bump_factors(max_order):
    """Numerators of derivatives of exp(-9/(1-r^2)) on |r|<1."""
    q = Polynomial([1.0, 0.0, -1.0])
    r = Polynomial([0.0, 1.0])
    factor = Polynomial([1.0])
    factors = []
    for order in range(max_order + 1):
        factors.append(factor)
        factor = q**2 * factor.deriv() + 4 * order * r * q * factor - 18 * r * factor
    return factors


def _paper_bump_derivatives_at_points(coordinates, center, radius, factors):
    """Evaluate the paper's C-infinity bump and its spatial derivatives."""
    scaled = (np.asarray(coordinates) - center) / radius
    values = np.zeros((len(scaled), len(factors)))
    inside = np.abs(scaled) < 1.0
    r = scaled[inside]
    q = 1.0 - r**2
    bump = np.exp(-9.0 / q)
    for order, factor in enumerate(factors):
        values[inside, order] = bump * factor(r) / q**(2 * order) / radius**order
    return values


def _sampled_weak_inputs(points, observations, domain_bounds, test_centers,
                         test_half_widths, test_degrees, max_derivative_order,
                         test_function):
    """Validate the shared Monte Carlo observations and test geometry."""
    points = np.asarray(points, dtype=float)
    observations = np.asarray(observations, dtype=float)
    bounds = np.asarray(domain_bounds, dtype=float)
    centers = np.asarray(test_centers, dtype=float)
    widths = np.asarray(test_half_widths, dtype=float)
    if test_function not in ("polynomial", "paper"):
        raise ValueError("test_function must be 'polynomial' or 'paper'.")
    degrees = np.asarray(test_degrees) if test_degrees is not None else None
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("Point coordinates must have spatial axes followed by time.")
    dimension = points.shape[1]
    if (
        len(points) == 0 or observations.shape != (len(points),)
        or bounds.shape != (dimension, 2) or centers.ndim != 2
        or centers.shape[1] != dimension or len(centers) == 0
        or widths.shape != (dimension,)
        or (test_function == "polynomial" and (degrees is None or degrees.shape != (dimension,)))
    ):
        raise ValueError("Data, bounds, centers, widths, and degrees have incompatible shapes.")
    if not all(np.all(np.isfinite(a)) for a in (
        points, observations, bounds, centers, widths,
    )):
        raise ValueError("All observations and test-function settings must be finite.")
    if np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(widths <= 0):
        raise ValueError("Domain bounds and test-function widths must be positive.")
    if np.any(points < bounds[:, 0]) or np.any(points > bounds[:, 1]):
        raise ValueError("Evaluation coordinates must lie inside domain_bounds.")
    boundary_tolerance = 1e-12 * np.maximum(1.0, bounds[:, 1] - bounds[:, 0])
    if (
        np.any(centers - widths < bounds[:, 0] - boundary_tolerance)
        or np.any(centers + widths > bounds[:, 1] + boundary_tolerance)
    ):
        raise ValueError("Every test-function support must lie inside domain_bounds.")
    max_orders = [max_derivative_order] * (dimension - 1) + [1]
    if test_function == "polynomial":
        if not np.all(np.isfinite(degrees)):
            raise ValueError("test_degrees must be finite.")
        if np.any(degrees != np.floor(degrees)) or np.any(degrees <= max_orders):
            raise ValueError("Each test degree must be an integer above its derivative order.")
    else:
        if degrees is not None:
            raise ValueError("test_degrees must be omitted for the paper bump.")
        degrees = np.zeros(dimension, dtype=int)
    return points, observations, bounds, centers, widths, degrees


def _integrate_sampled_weak_system(points, observations, powers, bounds,
                                   centers, widths, degrees,
                                   max_derivative_order, max_polynomial_degree,
                                   min_derivative_order, min_polynomial_degree,
                                   test_function):
    """Integrate raw or corrected powers against the same compact tests."""
    dimension = points.shape[1]
    spatial_dim = dimension - 1
    terms = polynomial_library_terms(
        spatial_dim, max_derivative_order, max_polynomial_degree,
        min_derivative_order=min_derivative_order,
        min_polynomial_degree=min_polynomial_degree,
    )
    derivatives = list(dict.fromkeys(derivative for derivative, _ in terms))
    derivative_indices = {derivative: index for index, derivative in enumerate(derivatives)}
    term_derivatives = np.array([derivative_indices[derivative] for derivative, _ in terms])
    term_powers = np.array([power - min_polynomial_degree for _, power in terms])
    orders = np.asarray(derivatives)
    signs = (-1.0)**np.sum(orders, axis=1)
    max_orders = [max_derivative_order] * spatial_dim + [1]
    if test_function == "paper":
        factors = [_paper_bump_factors(order) for order in max_orders]
    else:
        factors = [
            _bump_derivative_factors(int(degree), order)
            for degree, order in zip(degrees, max_orders)
        ]

    x = np.empty((len(centers), len(terms)))
    y = np.empty(len(centers))
    weight = np.prod(bounds[:, 1] - bounds[:, 0]) / len(points)
    tree = cKDTree(points / widths)
    for row, center in enumerate(centers):
        indices = tree.query_ball_point(center / widths, r=1.0, p=np.inf)
        if not indices:
            raise ValueError("A test function has no evaluation observations in its support.")
        local_points = points[indices]
        bumps = []
        for axis in range(dimension):
            if test_function == "paper":
                bumps.append(_paper_bump_derivatives_at_points(
                    local_points[:, axis], center[axis], widths[axis], factors[axis]
                ))
            else:
                bumps.append(_bump_derivatives_at_points(
                    local_points[:, axis], center[axis], widths[axis],
                    int(degrees[axis]), factors[axis]
                ))
        spatial_bump = np.ones(len(indices))
        spatial_derivatives = np.ones((len(indices), len(derivatives)))
        for axis in range(spatial_dim):
            spatial_bump *= bumps[axis][:, 0]
            spatial_derivatives *= bumps[axis][:, orders[:, axis]]
        y[row] = -weight * np.dot(spatial_bump * bumps[-1][:, 1], observations[indices])
        weighted_powers = powers[indices] * bumps[-1][:, 0, None]
        integrals = spatial_derivatives.T @ weighted_powers
        x[row] = weight * signs[term_derivatives] * integrals[term_derivatives, term_powers]

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Monte Carlo weak integrals are non-finite.")
    return x, y


def build_debiased_wsindy_system(
    training_points, training_values, evaluation_points, evaluation_values,
    u_estimator, *, domain_bounds, test_centers, test_half_widths,
    test_degrees=None, max_derivative_order=5, max_polynomial_degree=5,
    min_derivative_order=1, min_polynomial_degree=1,
    test_function="polynomial",
):
    """Build the corrected Monte Carlo weak system (X_db, Y_db).

    Training and evaluation are independent samples of space-time coordinates.
    Evaluation coordinates must be uniform over domain_bounds. The estimator
    is fitted only on training data; its predictions are evaluated on the
    independent sample. Spatial derivatives act on the compact test functions.
    """
    training_points = np.asarray(training_points, dtype=float)
    training_values = np.asarray(training_values, dtype=float)
    evaluation_points, evaluation_values, bounds, centers, widths, degrees = (
        _sampled_weak_inputs(
            evaluation_points, evaluation_values, domain_bounds, test_centers,
            test_half_widths, test_degrees, max_derivative_order, test_function,
        )
    )
    if (
        training_points.ndim != 2
        or training_points.shape[1] != evaluation_points.shape[1]
        or len(training_points) == 0
        or training_values.shape != (len(training_points),)
    ):
        raise ValueError("Training points and values have incompatible shapes.")
    if not np.all(np.isfinite(training_points)) or not np.all(np.isfinite(training_values)):
        raise ValueError("Training observations must be finite.")

    fitted_u = u_estimator.fit(training_points, training_values)
    values = np.asarray(fitted_u.predict(evaluation_points), dtype=float)
    if values.shape != evaluation_values.shape or not np.all(np.isfinite(values)):
        raise ValueError("u_estimator must predict one finite value per evaluation point.")

    # The exact first-order correction to v^j is v^j - j*v^(j-1)*(v-U).
    corrected_powers = np.column_stack([
        np.ones_like(values) if power == 0 else
        values**power - power * values**(power - 1) * (values - evaluation_values)
        for power in range(min_polynomial_degree, max_polynomial_degree + 1)
    ])

    return _integrate_sampled_weak_system(
        evaluation_points, evaluation_values, corrected_powers, bounds,
        centers, widths, degrees, max_derivative_order, max_polynomial_degree,
        min_derivative_order, min_polynomial_degree, test_function,
    )


def build_sampled_wsindy_system(
    points, observations, *, domain_bounds, test_centers, test_half_widths,
    test_degrees=None, max_derivative_order=5, max_polynomial_degree=5,
    min_derivative_order=1, min_polynomial_degree=1,
    test_function="polynomial",
):
    """Build ordinary WSINDy weak equations from uniform iid point samples."""
    points, observations, bounds, centers, widths, degrees = _sampled_weak_inputs(
        points, observations, domain_bounds, test_centers, test_half_widths,
        test_degrees, max_derivative_order, test_function,
    )
    powers = np.column_stack([
        observations**power for power in range(min_polynomial_degree, max_polynomial_degree + 1)
    ])
    return _integrate_sampled_weak_system(
        points, observations, powers, bounds, centers, widths, degrees,
        max_derivative_order, max_polynomial_degree,
        min_derivative_order, min_polynomial_degree, test_function,
    )


def sampled_wsindy(
    points, observations, *, domain_bounds, test_centers, test_half_widths,
    test_degrees=None, lambda_=0.0, regression="lasso", thresholds=None,
    max_derivative_order=5, max_polynomial_degree=5,
    min_derivative_order=1, min_polynomial_degree=1,
    test_function="polynomial",
    max_iter=10000, tol=1e-8,
):
    """Fit ordinary WSINDy on iid points using Monte Carlo weak integrals."""
    x, y = build_sampled_wsindy_system(
        points, observations, domain_bounds=domain_bounds, test_centers=test_centers,
        test_half_widths=test_half_widths, test_degrees=test_degrees,
        max_derivative_order=max_derivative_order,
        max_polynomial_degree=max_polynomial_degree,
        min_derivative_order=min_derivative_order,
        min_polynomial_degree=min_polynomial_degree,
        test_function=test_function,
    )
    return fit_sampled_wsindy_system(
        x, y, lambda_=lambda_, regression=regression, thresholds=thresholds,
        max_iter=max_iter, tol=tol,
    )


def fit_sampled_wsindy_system(
    x, y, *, lambda_=0.0, regression="lasso", thresholds=None,
    max_iter=10000, tol=1e-8,
):
    """Fit a precomputed Monte Carlo weak system, reusable across regressions."""
    if not np.isscalar(lambda_) or not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and nonnegative.")
    return _fit_coefficients(
        x, y, 2.0 * lambda_ / len(y), max_iter, tol, regression, thresholds
    )


def debiased_wsindy(
    training_points, training_values, evaluation_points, evaluation_values,
    u_estimator, *, domain_bounds, test_centers, test_half_widths,
    test_degrees=None, lambda_=0.0, regression="lasso", thresholds=None,
    max_derivative_order=5,
    max_polynomial_degree=5, min_derivative_order=1, min_polynomial_degree=1,
    test_function="polynomial",
    max_iter=10000, tol=1e-8,
) -> np.ndarray:
    """Fit the corrected weak system by OLS/LASSO or MSTLS.

    The default minimizes ||Y_db-X_db beta||^2/2 + lambda_*||beta||_1;
    lambda_=0 uses OLS. regression='mstls' applies the WSINDy threshold-and-
    refit rule to the same corrected system, with lambda_=0.
    ``tol`` controls KKT error on the equivalent mean-squared-loss scale,
    which is 2/K times the gradient scale of the displayed LASSO objective.
    """
    if not np.isscalar(lambda_) or not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and nonnegative.")
    x_debiased, y_debiased = build_debiased_wsindy_system(
        training_points, training_values, evaluation_points, evaluation_values,
        u_estimator, domain_bounds=domain_bounds, test_centers=test_centers,
        test_half_widths=test_half_widths, test_degrees=test_degrees,
        max_derivative_order=max_derivative_order,
        max_polynomial_degree=max_polynomial_degree,
        min_derivative_order=min_derivative_order,
        min_polynomial_degree=min_polynomial_degree,
        test_function=test_function,
    )
    rho_1 = 2.0 * lambda_ / len(y_debiased)
    return _fit_coefficients(
        x_debiased, y_debiased, rho_1, max_iter, tol, regression, thresholds
    )


def _tensor_weights(weights):
    product = weights[0]
    for axis_weights in weights[1:]:
        product = np.multiply.outer(product, axis_weights)
    return product.ravel()


class _WeakResidual:
    """Shared weak residual and its Jacobian with respect to observations.

    Store translated support indices and one kernel per spatial derivative,
    rather than a dense (coefficient, test, observation) Jacobian tensor.
    A(beta) is sparse; its full K-by-K covariance retains correlations between
    overlapping tests. This is costlier than WSINDy as the number of tests grows.
    """

    def __init__(self, u, spatial_grid, time, **test_settings):
        self.X, self.y = build_wsindy_system(u, spatial_grid, time, **test_settings)
        u, terms, weights, strides = _prepare_weak_tests(u, spatial_grid, time, **test_settings)
        self.u = u.ravel()
        self.terms = terms
        self.degree = max(power for _, power in terms)
        derivatives = list(dict.fromkeys(alpha for alpha, _ in terms))
        derivative_indices = {derivative: index for index, derivative in enumerate(derivatives)}
        self.term_derivative_indices = np.array([
            derivative_indices[derivative] for derivative, _ in terms
        ])
        self.term_powers = np.array([power for _, power in terms])
        self.columns_by_derivative = [
            np.flatnonzero(self.term_derivative_indices == index)
            for index in range(len(derivatives))
        ]
        self.columns_by_power = [
            np.flatnonzero(self.term_powers == power)
            for power in range(1, self.degree + 1)
        ]

        widths = [len(axis_weights[0]) // 2 for axis_weights in weights]
        starts = np.meshgrid(
            *[np.arange(0, size - 2 * m, stride) for size, m, stride in zip(u.shape, widths, strides)],
            indexing="ij",
        )
        offsets = np.meshgrid(*[np.arange(1, 2 * m) for m in widths], indexing="ij")
        starts = np.ravel_multi_index(tuple(starts), u.shape).ravel()
        offsets = np.ravel_multi_index(tuple(offsets), u.shape).ravel()
        self.indices = starts[:, None] + offsets[None, :]
        self.indptr = np.arange(len(starts) + 1) * len(offsets)
        self.local_u = self.u[self.indices]

        # Endpoints vanish for every derivative used, so omit them from A.
        self.time_kernel = -_tensor_weights(
            [axis_weights[0][1:-1] for axis_weights in weights[:-1]] + [weights[-1][1][1:-1]]
        )
        self.spatial_kernels = np.array([
            (-1)**sum(alpha) * _tensor_weights(
                [weights[axis][order][1:-1] for axis, order in enumerate(alpha)]
                + [weights[-1][0][1:-1]]
            )
            for alpha in derivatives
        ])

    def jacobian(self, beta):
        """A = d(y-X beta)/dU = W_t - sum beta[alpha,j] W_alpha diag(j U^(j-1))."""
        values = np.broadcast_to(self.time_kernel, self.indices.shape).copy()
        for kernel, columns in zip(self.spatial_kernels, self.columns_by_derivative):
            slope_coefficients = np.zeros(self.degree)
            for column in columns:
                power = self.term_powers[column]
                if power > 0:
                    slope_coefficients[power - 1] = power * beta[column]
            if np.any(slope_coefficients):
                slope = np.polynomial.polynomial.polyval(self.u, slope_coefficients)
                values -= kernel[None, :] * slope[self.indices]
        return sparse.csr_matrix(
            (values.ravel(), self.indices.ravel(), self.indptr),
            shape=(len(self.y), len(self.u)),
        )

    def covariance(self, beta, alpha):
        """C=(1-alpha) A A.T + alpha I, without a noise-variance factor."""
        if alpha == 1:
            return np.eye(len(self.y))
        A = self.jacobian(beta)
        covariance = (1.0 - alpha) * (A @ A.T).toarray()
        covariance.flat[::len(self.y) + 1] += alpha
        return covariance

    def whiten(self, beta, alpha):
        factor = linalg.cholesky(self.covariance(beta, alpha), lower=True)
        X = linalg.solve_triangular(factor, self.X, lower=True)
        y = linalg.solve_triangular(factor, self.y, lower=True)
        return X, y

    def likelihood(self, beta, noise_std, alpha):
        """Mean Gaussian weak-residual negative log likelihood and exact gradient.

        f = [logdet(C) + r.T C^-1 r / noise_std^2] / (2 K), r=y-X beta.
        Terms independent of beta are omitted. Both the log determinant and
        the beta-dependence of C in the quadratic term contribute derivatives.
        """
        count = len(self.y)
        variance = noise_std**2
        residual = self.y - self.X @ beta
        if alpha == 1:
            value = residual @ residual / (2.0 * count * variance)
            gradient = -self.X.T @ residual / (count * variance)
            return value, gradient

        A = self.jacobian(beta)
        covariance = (1.0 - alpha) * (A @ A.T).toarray()
        covariance.flat[::count + 1] += alpha
        factor = linalg.cholesky(covariance, lower=True)
        weighted_residual = linalg.cho_solve((factor, True), residual)
        logdet = 2.0 * np.sum(np.log(np.diag(factor)))
        value = (logdet + residual @ weighted_residual / variance) / (2.0 * count)
        gradient = -self.X.T @ weighted_residual / (count * variance)

        # dC = (1-alpha) (dA A.T + A dA.T).
        inverse = linalg.cho_solve((factor, True), np.eye(count))
        metric = inverse - np.outer(weighted_residual, weighted_residual) / variance
        covariance_gradient = np.zeros(len(self.terms))
        for start in range(0, count, 32):
            stop = min(start + 32, count)
            # Only materialize 32 rows of metric @ A, not the full K-by-N array.
            block = (A.T @ metric[start:stop].T).T
            local = block[np.arange(stop - start)[:, None], self.indices[start:stop]].copy()
            for power, matching in enumerate(self.columns_by_power, start=1):
                moments = power * np.sum(local, axis=0)
                if len(matching):
                    derivatives = self.spatial_kernels @ moments
                    covariance_gradient[matching] -= derivatives[
                        self.term_derivative_indices[matching]
                    ]
                local *= self.local_u[start:stop]
        gradient += (1.0 - alpha) * covariance_gradient / count
        return float(value), gradient


def _validate_wendy_controls(rho_1, regression, thresholds, alpha, max_iter, tol):
    if not np.isfinite(rho_1) or rho_1 < 0:
        raise ValueError("rho_1 must be finite and nonnegative.")
    if regression not in ("lasso", "mstls"):
        raise ValueError("regression must be 'lasso' or 'mstls'.")
    if regression == "mstls" and rho_1 != 0:
        raise ValueError("rho_1 must be 0 when regression='mstls'; use thresholds instead.")
    if regression == "lasso" and thresholds is not None:
        raise ValueError("thresholds applies only to regression='mstls'.")
    if regression == "mstls":
        _threshold_grid(thresholds)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1].")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")


def _initial_coefficients(problem, initial_beta, rho_1, regression, thresholds, max_iter, tol):
    if initial_beta is None:
        return _fit_coefficients(problem.X, problem.y, rho_1, max_iter, tol, regression, thresholds)
    beta = np.asarray(initial_beta, dtype=float)
    if beta.shape != (problem.X.shape[1],) or not np.all(np.isfinite(beta)):
        raise ValueError("initial_beta must be a finite vector with one entry per library term.")
    return beta.copy()


def wendy(
    u: np.ndarray,
    spatial_grid: tuple[np.ndarray, ...],
    time: np.ndarray,
    *,
    rho_1: float = 0.0,
    regression: str = "lasso",
    thresholds: tuple[float, ...] | np.ndarray | None = None,
    alpha: float = 1e-10,
    max_reweights: int = 100,
    reweight_tol: float = 1e-6,
    normality_tol: float | None = 1e-4,
    normality_start: int = 10,
    initial_beta: np.ndarray | None = None,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    min_derivative_order: int = 1,
    min_polynomial_degree: int = 1,
    half_widths: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
    test_degrees: tuple[int, ...] | None = None,
    test_function: str = "polynomial",
    max_iter: int = 10000,
    tol: float = 1e-8,
) -> np.ndarray:
    """WENDy-IRLS with LASSO or MSTLS in each covariance-weighted regression.

    Use the same weak tests/library as WSINDy. At each reweighting step freeze
    C=(1-alpha) A(beta,U) A(beta,U).T + alpha I and fit the whitened X and y.
    LASSO minimizes r.T C^-1 r / K + rho_1*||beta||_1; rho_1=0 is GLS.
    MSTLS applies its bounds and selection loss to the whitened system. These
    sparsity additions extend the draft's nonsparse WENDy iteration.

    Initial coefficients come from the corresponding WSINDy regression unless
    initial_beta is supplied. Stop at relative coefficient change <= reweight_tol
    (denominator max(||beta_old||,1e-12)); hitting max_reweights raises RuntimeError.
    As in Bortz et al. (2023), Eq. (18), optionally stop when Shapiro-Wilk p-value
    falls below normality_tol after normality_start reweights, emitting a warning
    that this is not fixed-point convergence. Set normality_tol=None to disable.
    max_iter/tol control inner regression. alpha=1 reduces to WSINDy.

    No noise standard deviation or clean solution is used. In particular, the
    L1 penalty here applies to the stated whitened loss without a sigma factor.
    The full covariance includes correlations between overlapping tests; choose
    strides to control its K-by-K size when working with large grids.
    """
    _validate_wendy_controls(rho_1, regression, thresholds, alpha, max_iter, tol)
    if not isinstance(max_reweights, (int, np.integer)) or max_reweights < 1:
        raise ValueError("max_reweights must be a positive integer.")
    if not np.isfinite(reweight_tol) or reweight_tol <= 0:
        raise ValueError("reweight_tol must be finite and positive.")
    if normality_tol is not None and (not np.isfinite(normality_tol) or not 0 < normality_tol < 1):
        raise ValueError("normality_tol must be None or in (0, 1).")
    if not isinstance(normality_start, (int, np.integer)) or normality_start < 1:
        raise ValueError("normality_start must be a positive integer.")
    problem = _WeakResidual(
        u, spatial_grid, time,
        max_derivative_order=max_derivative_order, max_polynomial_degree=max_polynomial_degree,
        min_derivative_order=min_derivative_order, min_polynomial_degree=min_polynomial_degree,
        half_widths=half_widths, strides=strides, test_degrees=test_degrees,
        test_function=test_function,
    )
    beta = _initial_coefficients(problem, initial_beta, rho_1, regression, thresholds, max_iter, tol)
    for iteration in range(max_reweights):
        X, y = problem.whiten(beta, alpha)
        updated = _fit_coefficients(X, y, rho_1, max_iter, tol, regression, thresholds)
        relative_change = np.linalg.norm(updated - beta) / max(np.linalg.norm(beta), 1e-12)
        beta = updated
        if relative_change <= reweight_tol:
            return beta
        if normality_tol is not None and iteration + 1 > normality_start and len(y) >= 3:
            p_value = stats.shapiro(y - X @ beta).pvalue
            if p_value < normality_tol:
                warnings.warn(
                    f"WENDy stopped on the normality test after {iteration + 1} reweights "
                    f"(p={p_value:.3g}); the fixed-point tolerance was not met.",
                    RuntimeWarning, stacklevel=2,
                )
                return beta
    raise RuntimeError(
        f"WENDy did not converge in {max_reweights} reweights; "
        "check alpha, initialization, sparsity settings, or the iteration budget."
    )


def _fit_weak_likelihood(problem, initial_beta, active, noise_std, alpha, rho_1, max_iter, tol):
    """Optimize the actual beta-dependent likelihood on a specified support.

    L1 uses beta=scale*(positive-negative) with nonnegative optimization
    variables. This keeps the objective differentiable in the optimization
    variables while preserving exactly rho_1*||beta||_1 at the optimum.
    """
    beta = np.zeros(problem.X.shape[1])
    if not np.any(active):
        return beta
    X = problem.X[:, active]
    if alpha == 1:
        # The likelihood is then exactly a scaled least-squares objective.
        beta[active] = _fit_coefficients(
            X, problem.y, 2.0 * noise_std**2 * rho_1, max_iter, tol
        )
        return beta

    column_norms = np.linalg.norm(X, axis=0)
    target_norm = max(np.linalg.norm(problem.y), 1e-12)
    scales = np.divide(target_norm, column_norms, out=np.ones_like(column_norms), where=column_norms > 0)
    start = initial_beta[active] / scales
    size = len(start)

    def objective(parameters):
        if rho_1 > 0:
            values = parameters[:size] - parameters[size:]
        else:
            values = parameters
        coefficients = np.zeros_like(beta)
        coefficients[active] = scales * values
        value, gradient = problem.likelihood(coefficients, noise_std, alpha)
        gradient = scales * gradient[active]
        if rho_1 > 0:
            value += rho_1 * np.sum(scales * (parameters[:size] + parameters[size:]))
            gradient = np.concatenate([gradient + rho_1 * scales, -gradient + rho_1 * scales])
        return value, gradient

    if rho_1 > 0:
        start = np.concatenate([np.maximum(start, 0.0), np.maximum(-start, 0.0)])
        bounds = [(0.0, None)] * len(start)
        algorithm = "L-BFGS-B"
        options = {"maxiter": max_iter, "gtol": tol, "ftol": 0.0, "maxls": 50, "maxcor": 20}
    else:
        bounds = None
        algorithm = "BFGS"
        options = {"maxiter": max_iter, "gtol": tol}
    solution = optimize.minimize(
        objective, start, jac=True, method=algorithm, bounds=bounds, options=options,
    )
    parameters = solution.x[:size] - solution.x[size:] if rho_1 > 0 else solution.x
    beta[active] = scales * parameters
    value, gradient = problem.likelihood(beta, noise_std, alpha)
    gradient = gradient[active]
    violation = np.maximum(np.abs(gradient) - rho_1, 0.0)
    nonzero = beta[active] != 0
    violation[nonzero] = np.abs(gradient[nonzero] + rho_1 * np.sign(beta[active][nonzero]))
    optimality = np.max(scales * violation)
    if not np.isfinite(value) or not np.all(np.isfinite(beta)) or not np.isfinite(optimality) or optimality > tol:
        raise RuntimeError(
            f"WENDy-MLE did not reach its stationarity tolerance "
            f"(scaled KKT violation {optimality:.3g}, target {tol:g}): {solution.message}"
        )
    return beta


def _threshold_weak_likelihood(problem, initial_beta, noise_std, alpha, thresholds, max_iter, tol):
    """MSTLS extension: threshold and refit the nonlinear weak likelihood.

    Freeze the *selection* geometry at the dense MLE covariance: threshold
    bounds and prediction-distance scores use this common whitened X and y.
    Every refit still updates covariance inside the likelihood. This is an
    explicitly defined extension, not the linear-OLS MSTLS algorithm itself.
    """
    thresholds = _threshold_grid(thresholds)
    full_support = np.ones(problem.X.shape[1], dtype=bool)
    reference = _fit_weak_likelihood(
        problem, initial_beta, full_support, noise_std, alpha, 0.0, max_iter, tol
    )
    X, y = problem.whiten(reference, alpha)
    reference_norm = np.linalg.norm(X @ reference)
    if reference_norm == 0:
        return np.zeros_like(reference)
    column_norms = np.linalg.norm(X, axis=0)
    if np.any(column_norms == 0):
        raise np.linalg.LinAlgError("MSTLS selection requires nonzero library columns.")
    ratios = np.linalg.norm(y) / column_norms
    best_loss = np.inf
    best_beta = None
    for threshold in thresholds:
        lower = threshold * np.maximum(1.0, ratios)
        upper = np.minimum(1.0, ratios) / threshold
        beta = reference.copy()
        active = full_support.copy()
        # A term can only be removed, so at most P support reductions occur.
        for _ in range(len(beta) + 1):
            retained = active & (np.abs(beta) >= lower) & (np.abs(beta) <= upper)
            if np.array_equal(retained, active):
                break
            beta = _fit_weak_likelihood(
                problem, beta, retained, noise_std, alpha, 0.0, max_iter, tol
            )
            active = retained
        loss = np.linalg.norm(X @ (beta - reference)) / reference_norm
        loss += np.count_nonzero(beta) / len(beta)
        if loss < best_loss:
            best_loss = loss
            best_beta = beta.copy()
    return best_beta


def wendy_mle(
    u: np.ndarray,
    spatial_grid: tuple[np.ndarray, ...],
    time: np.ndarray,
    *,
    noise_std: float,
    rho_1: float = 0.0,
    regression: str = "lasso",
    thresholds: tuple[float, ...] | np.ndarray | None = None,
    alpha: float = 0.0,
    initial_beta: np.ndarray | None = None,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    min_derivative_order: int = 1,
    min_polynomial_degree: int = 1,
    half_widths: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
    test_degrees: tuple[int, ...] | None = None,
    test_function: str = "polynomial",
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Minimize the draft's Gaussian weak-residual likelihood, optionally sparse.

    With r=y-X beta and C=(1-alpha) A(beta,U) A(beta,U).T + alpha I, LASSO minimizes
      [logdet(C) + r.T C^-1 r / noise_std^2] / (2 K) + rho_1*||beta||_1.
    Dividing the draft's loss by K leaves unpenalized fits unchanged and fixes
    the penalty convention. alpha=0 is the draft likelihood; positive alpha
    explicitly regularizes it. noise_std must be supplied and strictly positive:
    the sigma=0 Gaussian likelihood is undefined. It is never replaced by an
    undisclosed variance floor or estimated from a clean solution.

    L1 uses nonnegative positive/negative parameter parts and L-BFGS-B;
    unpenalized fits use BFGS. Both use analytic derivatives of both likelihood
    terms, including covariance.
    rho_1=0 is unpenalized MLE. max_iter bounds nonlinear iterations per fit;
    tol bounds KKT violations after scaling beta_j by ||y||/||X_j|| (with
    floors for zero norms). Failure raises RuntimeError. This nonconvex solve
    finds a stationary point, not a guaranteed global optimum. alpha=1 is
    solved exactly as a scaled OLS/LASSO problem with its usual KKT tolerance.

    MSTLS is an extension: compute a dense MLE, freeze its whitening for the
    usual threshold bounds/selection score, then threshold and refit the full
    nonlinear likelihood on each retained support. thresholds=None uses the
    same 50 candidates as WSINDy; rho_1 must be zero in this mode. No LASSO or
    fixed-covariance least-squares solve is substituted for the MLE refits.

    Initial coefficients come from the corresponding WSINDy regression, or
    from initial_beta; max_iter/tol also control any LASSO initialization.
    The weak tests and returned coefficient order match
    WSINDy. This is an approximate residual likelihood, not the exact
    observation-data likelihood discussed separately in the draft.
    """
    _validate_wendy_controls(rho_1, regression, thresholds, alpha, max_iter, tol)
    variance = float(noise_std) * float(noise_std)
    if not np.isfinite(noise_std) or noise_std <= 0 or not np.isfinite(variance) or variance <= 0:
        raise ValueError("noise_std must have a finite, strictly positive variance; sigma=0 has no Gaussian MLE.")
    problem = _WeakResidual(
        u, spatial_grid, time,
        max_derivative_order=max_derivative_order, max_polynomial_degree=max_polynomial_degree,
        min_derivative_order=min_derivative_order, min_polynomial_degree=min_polynomial_degree,
        half_widths=half_widths, strides=strides, test_degrees=test_degrees,
        test_function=test_function,
    )
    beta = _initial_coefficients(problem, initial_beta, rho_1, regression, thresholds, max_iter, tol)
    if regression == "mstls":
        return _threshold_weak_likelihood(problem, beta, noise_std, alpha, thresholds, max_iter, tol)
    return _fit_weak_likelihood(
        problem, beta, np.ones(len(beta), dtype=bool), noise_std, alpha, rho_1, max_iter, tol
    )
