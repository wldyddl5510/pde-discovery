"""Estimate PDE coefficients from observations on a uniform space-time grid.

Arrays have spatial axes first and time last, as in simulation_generation.py.
SINDy differentiates the observations; WSINDy differentiates test functions.
Both offer least squares with an optional LASSO penalty, or the WSINDy
paper's modified sequential-thresholding least squares (MSTLS).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import factorial

import numpy as np
from numpy.polynomial import Polynomial


def polynomial_library_terms(
    spatial_dim: int,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
) -> list[tuple[tuple[int, ...], int]]:
    """List (spatial_derivative, polynomial_power) in coefficient-vector order.

    Include every mixed derivative with total order 1 through the maximum,
    and every power 1 through the maximum. There is no zeroth derivative or
    constant feature. In 2D, derivative order starts (1, 0), (0, 1), (2, 0),
    (1, 1), (0, 2), ... . All powers of one derivative are listed together,
    matching the draft's coefficient index (s - 1) * J + (j - 1) in Python.
    """
    for name, value in (
        ("spatial_dim", spatial_dim),
        ("max_derivative_order", max_derivative_order),
        ("max_polynomial_degree", max_polynomial_degree),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    terms = []
    for total_order in range(1, max_derivative_order + 1):
        for axes in combinations_with_replacement(range(spatial_dim), total_order):
            derivative = tuple(axes.count(axis) for axis in range(spatial_dim))
            for power in range(1, max_polynomial_degree + 1):
                terms.append((derivative, power))
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
        spatial_dim, max_derivative_order, max_polynomial_degree
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
    interior = (slice(radius, -radius),) * spatial_dim + (slice(1, -1),)

    time_derivative = _finite_difference(u, time_step, axis=spatial_dim, order=1)
    y = time_derivative[interior].ravel()
    X = np.empty((len(y), len(terms)))
    powers = [u**power for power in range(1, max_polynomial_degree + 1)]

    for column, (derivative, power) in enumerate(terms):
        feature = powers[power - 1]
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
    if thresholds is None:
        thresholds = np.logspace(-4, 0, 50)
    thresholds = np.asarray(thresholds, dtype=float)
    if (
        thresholds.ndim != 1 or thresholds.size == 0
        or not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0)
    ):
        raise ValueError("thresholds must be a nonempty 1D sequence of finite positive values.")
    thresholds = np.unique(thresholds)  # Ascending order also resolves loss ties.

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
    )
    return _fit_coefficients(X, y, rho_1, max_iter, tol, regression, thresholds)


def _test_function_weights(half_width, spacing, degree, max_order):
    """Sample derivatives of b(r)=(1-r^2)^degree, including quadrature weights.

    r=(z-center)/(half_width*spacing). The test function has peak one and
    is zero outside |r|<1. degree>max_order makes all sampled derivatives
    vanish at the endpoints, so the trapezoidal endpoint weights are zero.
    """
    r = np.arange(-half_width, half_width + 1, dtype=float) / half_width
    radius = half_width * spacing
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


def build_wsindy_system(
    u: np.ndarray,
    spatial_grid: tuple[np.ndarray, ...],
    time: np.ndarray,
    *,
    max_derivative_order: int = 5,
    max_polynomial_degree: int = 5,
    half_widths: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
    test_degrees: tuple[int, ...] | None = None,
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
    not their Fourier-based support selection. Direct separable sums keep
    the implementation simple. Narrow, poorly resolved bumps can give large
    quadrature error; support selection is an experiment parameter.
    """
    u = np.asarray(u, dtype=float)
    spatial_dim = len(spatial_grid)
    terms = polynomial_library_terms(
        spatial_dim, max_derivative_order, max_polynomial_degree
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

    if test_degrees is None:
        degrees = []
        for m, order in zip(half_widths, max_orders):
            first_interior_value = (2.0 * m - 1.0) / m**2
            decay_degree = int(np.ceil(np.log(1e-10) / np.log(first_interior_value)))
            degrees.append(max(order + 1, decay_degree))
        test_degrees = tuple(degrees)
    if len(test_degrees) != u.ndim or any(
        not isinstance(p, (int, np.integer)) or p <= order
        for p, order in zip(test_degrees, max_orders)
    ):
        raise ValueError("Each test_degree must be an integer greater than its axis derivative order.")

    # weights[axis][order] includes that coordinate's quadrature spacing.
    weights = []
    for m, step, p, order in zip(half_widths, steps, test_degrees, max_orders):
        weights.append(_test_function_weights(m, step, p, order))

    time_weights = [axis_weights[0] for axis_weights in weights[:-1]] + [weights[-1][1]]
    y = -_weak_integrals(u, time_weights, strides)
    X = np.empty((len(y), len(terms)))
    powers = [u**power for power in range(1, max_polynomial_degree + 1)]
    for column, (derivative, power) in enumerate(terms):
        spatial_weights = [weights[axis][order] for axis, order in enumerate(derivative)]
        feature_weights = spatial_weights + [weights[-1][0]]
        integrals = _weak_integrals(powers[power - 1], feature_weights, strides)
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
    half_widths: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
    test_degrees: tuple[int, ...] | None = None,
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
        half_widths=half_widths,
        strides=strides,
        test_degrees=test_degrees,
    )
    return _fit_coefficients(X, y, rho_1, max_iter, tol, regression, thresholds)
