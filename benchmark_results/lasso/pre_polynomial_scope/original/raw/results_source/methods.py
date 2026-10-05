"""Convolutional WSINDy with author-code and printed-algorithm profiles.

Space axes precede time. Library terms are D^alpha f(U). Both profiles solve
the scaled weak system; the author profile sparsifies physical coefficients.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import comb, factorial
import numpy as np
from scipy import fft, linalg, optimize


@dataclass(frozen=True)
class LibraryTerm:
    powers: tuple[int, ...]
    derivative: tuple[int, ...]  # includes time
    kind: str = "poly"
    component: int = 0
    frequency: int = 1

    def label(self, names):
        if self.kind == "poly":
            f = "*".join(n if p == 1 else f"{n}^{p}" for n, p in zip(names, self.powers) if p) or "1"
        else:
            f = f"{self.kind}({self.frequency}*{names[self.component]})"
        return f if not any(self.derivative) else f"D{self.derivative}({f})"


@dataclass
class WeakSystem:
    G: np.ndarray
    b: np.ndarray
    coefficient_scales: np.ndarray
    terms: tuple[LibraryTerm, ...]
    state_scales: np.ndarray
    coordinate_scales: np.ndarray
    centers: tuple[np.ndarray, ...]
    half_widths: tuple[int, ...]
    degrees: tuple[int, ...]
    test_function: str = "polynomial"


@dataclass
class MSTLSResult:
    coefficients: np.ndarray
    scaled_coefficients: np.ndarray
    threshold: float
    thresholds: np.ndarray
    losses: np.ndarray
    least_squares: np.ndarray
    rank: int


@dataclass
class LassoResult(MSTLSResult):
    alpha_max: np.ndarray
    normalized_coefficients: np.ndarray
    column_rms: np.ndarray
    response_rms: np.ndarray
    kkt_errors: np.ndarray
    solver_warnings: tuple[str, ...]


def polynomial_library(components, spatial_dim, max_degree, max_derivative, *,
                       trigonometric_frequencies=()):
    """Table 3: total-degree monomials and pure spatial derivatives.

    Include one constant; omit its derivatives and spatial cross derivatives.
    """
    for value in (components, spatial_dim, max_degree, max_derivative):
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("Library dimensions and orders must be nonnegative integers.")
    if components < 1 or spatial_dim < 1:
        raise ValueError("Need at least one component and one spatial dimension.")
    zero = (0,) * (spatial_dim + 1)
    derivatives = [zero] + [tuple(q if d == a else 0 for d in range(spatial_dim+1))
                            for a in range(spatial_dim) for q in range(1, max_derivative+1)]
    terms = [LibraryTerm(p, d) for p in product(range(max_degree+1), repeat=components)
             if sum(p) <= max_degree for d in (derivatives if any(p) else [zero])]
    for frequency in trigonometric_frequencies:
        if not isinstance(frequency, int) or frequency <= 0:
            raise ValueError("Trigonometric frequencies must be positive integers.")
        for c in range(components):
            for kind in ("sin", "cos"):
                terms.extend(LibraryTerm((0,)*components, d, kind, c, frequency) for d in derivatives)
    return tuple(terms)


def test_function_weights(m, max_order, *, degree=None, tau=1e-10):
    """Algorithm 4.1: analytic derivatives of (1-r^2)^p on [-1, 1]."""
    if not isinstance(m, (int, np.integer)) or m < 2:
        raise ValueError("Half-width must be an integer >= 2.")
    if not isinstance(max_order, (int, np.integer)) or max_order < 0:
        raise ValueError("Derivative order must be a nonnegative integer.")
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must lie strictly between zero and one.")
    if degree is None:
        degree = max(max_order+1, int(np.ceil(np.log(tau)/np.log((2*m-1)/m**2))))
    if not isinstance(degree, (int, np.integer)) or degree <= max_order:
        raise ValueError("Test degree must exceed the largest derivative order.")
    r = np.linspace(-1, 1, 2*m+1)
    weights = np.zeros((max_order+1, len(r)))
    falling = [factorial(degree)/factorial(degree-k) for k in range(max_order+1)]
    # Leibniz form avoids expanding a high-degree polynomial in the monomial basis.
    for q in range(max_order+1):
        for left in range(q+1):
            right = q-left
            weights[q] += (comb(q, left)*falling[left]*falling[right]*(-1)**right
                           * (1+r)**(degree-left)*(1-r)**(degree-right))
    weights[:, (0, -1)] = 0
    return weights, int(degree)


def bump_function_weights(m, max_order):
    """Analytic derivatives of exp(9/(r^2-1)), consistency paper Section 5.1.

    f^(q) = f P_q/(1-r^2)^(2q). The polynomial recurrence avoids numerical
    differentiation, including at the eighth spatial derivative.
    """
    if not isinstance(m, (int, np.integer)) or m < 2:
        raise ValueError("Half-width must be an integer >= 2.")
    if not isinstance(max_order, (int, np.integer)) or max_order < 0:
        raise ValueError("Derivative order must be a nonnegative integer.")
    from numpy.polynomial import Polynomial
    r = np.linspace(-1, 1, 2*m+1)[1:-1]
    denominator = 1-r*r
    phi = np.exp(-9/denominator)
    weights = np.zeros((max_order+1, 2*m+1))
    polynomial, coordinate = Polynomial([1.]), Polynomial([0., 1.])
    one_minus_square = Polynomial([1., 0., -1.])
    for q in range(max_order+1):
        weights[q, 1:-1] = phi*polynomial(r)/denominator**(2*q)
        polynomial = (one_minus_square**2*polynomial.deriv()
                      +(4*q*coordinate*one_minus_square-18*coordinate)*polynomial)
    return weights


def _coordinate_scale(m, spacing, degree, order):
    if order == 0:
        return 1.0
    # Eq. 4.7 is stated for even orders. The authors' factorial convention
    # extends it to odd orders (the first-derivative amplitude factor is 1).
    amplitude = (factorial(degree)/factorial(degree-order//2)
                 * factorial(order)/factorial((order+1)//2))
    return amplitude**(1/order)/(m*spacing)


def _observations(u, axes):
    values = (u,) if isinstance(u, np.ndarray) else tuple(u)
    values = tuple(np.asarray(v, dtype=float) for v in values)
    if not values or len(axes) < 2:
        raise ValueError("Need observations and spatial/time grids.")
    shape = values[0].shape
    if len(shape) != len(axes) or any(v.shape != shape or not np.all(np.isfinite(v)) for v in values):
        raise ValueError("Components must be finite arrays on the same spatial/time grid.")
    grids, spacing = [], []
    for axis, size in zip(axes, shape):
        axis = np.asarray(axis, dtype=float)
        if axis.ndim != 1 or len(axis) != size or size < 5 or not np.all(np.isfinite(axis)):
            raise ValueError("Each grid must match its data axis and have >= 5 finite points.")
        steps = np.diff(axis)
        if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-7, atol=0):
            raise ValueError("Grids must be increasing and uniformly spaced.")
        grids.append(axis)
        spacing.append(float(steps[0]))
    return values, tuple(grids), np.asarray(spacing)


def _validate_term(term, components, dimensions):
    if len(term.powers) != components or len(term.derivative) != dimensions:
        raise ValueError("Term dimensions must match the observations.")
    if any(not isinstance(q, (int, np.integer)) or q < 0 for q in (*term.powers, *term.derivative)):
        raise ValueError("Powers and derivative orders must be nonnegative integers.")
    if term.kind not in ("poly", "sin", "cos"):
        raise ValueError("Only polynomial, sine, and cosine trial functions are supported.")
    if term.kind == "poly" and not any(term.powers) and any(term.derivative):
        raise ValueError("Derivatives of constants must be omitted.")
    if term.kind != "poly" and (any(term.powers) or not 0 <= term.component < components
            or not isinstance(term.frequency, int) or term.frequency <= 0):
        raise ValueError("Invalid trigonometric term.")


def _weak_columns(values, derivatives, weights, radii, spacing, strides):
    """Eq. 3.12, reusing FFTs and subsampling intermediate axes."""
    def visit(array, axis, suffixes, prefix):
        length = array.shape[axis]
        kernel_length = weights[axis].shape[1]
        # Circular convolution agrees with linear convolution on the valid
        # indices kernel_length-1:length; wrapped tails only affect discarded
        # boundary indices. This is the paper's N-point FFT construction.
        nfft = fft.next_fast_len(length)
        transformed = fft.rfft(array, n=nfft, axis=axis)
        for order in sorted({s[0] for s in suffixes}):
            kernel = weights[axis][order]*spacing[axis]/radii[axis]**order
            shape = [1]*array.ndim
            shape[axis] = -1
            convolved = fft.irfft(transformed*fft.rfft(kernel, nfft).reshape(shape), n=nfft, axis=axis)
            selection = [slice(None)]*array.ndim
            selection[axis] = slice(kernel_length-1, length, strides[axis])
            reduced = convolved[tuple(selection)].copy()
            next_prefix = prefix+(order,)
            if axis == array.ndim-1:
                yield next_prefix, reduced.ravel()
            else:
                yield from visit(reduced, axis+1, [s[1:] for s in suffixes if s[0] == order], next_prefix)
    yield from visit(values, 0, list(derivatives), ())


def build_wsindy_system(u, spatial_grid, time, *, library_terms, lhs_components=(0,),
                        lhs_time_order=1, half_widths, strides, test_degrees=None,
                        tau=1e-10, rescale=True, state_scale_rule="printed",
                        test_function="polynomial"):
    """Algorithm 4.2, with coupled components and arbitrary LHS time order.

    Select the printed 1/beta_max or author-code 1/(beta_max-1) exponent.
    Restore physical units
    using the coordinate identity; see README for the printed sign discrepancy.
    Trigonometric functions are evaluated on the original observations.
    """
    values, grids, spacing = _observations(u, (*spatial_grid, time))
    if state_scale_rule not in ("printed", "authors"):
        raise ValueError("state_scale_rule must be printed or authors.")
    dimensions, count = len(grids), len(values)
    terms = tuple(library_terms)
    if not terms or len(set(terms)) != len(terms):
        raise ValueError("A library must contain unique terms.")
    for term in terms:
        _validate_term(term, count, dimensions)
        if term.derivative[-1]:
            raise ValueError("Paper RHS libraries contain only spatial derivatives.")
    if not isinstance(lhs_time_order, int) or lhs_time_order < 1:
        raise ValueError("lhs_time_order must be a positive integer.")
    lhs_components = tuple(lhs_components)
    if not lhs_components or len(set(lhs_components)) != len(lhs_components) or any(
        not isinstance(c, int) or not 0 <= c < count for c in lhs_components):
        raise ValueError("lhs_components must select distinct observed components.")
    lhs = [LibraryTerm(tuple(int(i == c) for i in range(count)),
                       (0,)*(dimensions-1)+(lhs_time_order,)) for c in lhs_components]
    all_terms = (*terms, *lhs)
    orders = tuple(max(t.derivative[d] for t in all_terms) for d in range(dimensions))
    widths, steps = tuple(half_widths), tuple(strides)
    if len(widths) != dimensions or any(not isinstance(m, (int, np.integer)) or m < 2
            or 2*m+1 > len(g) for m, g in zip(widths, grids)):
        raise ValueError("Each integer half-width must be >= 2 and fit its grid.")
    if len(steps) != dimensions or any(not isinstance(s, (int, np.integer)) or s < 1 for s in steps):
        raise ValueError("Each axis needs a positive integer stride.")
    degrees = (None,)*dimensions if test_degrees is None else tuple(test_degrees)
    if len(degrees) != dimensions:
        raise ValueError("Each axis needs one test degree.")
    if test_function == "polynomial":
        weights, degrees = zip(*(test_function_weights(m, q, degree=p, tau=tau)
                                for m, q, p in zip(widths, orders, degrees)))
    elif test_function == "bump":
        if rescale:
            raise ValueError("The consistency bump profile uses physical coordinates without rescaling.")
        weights = tuple(bump_function_weights(m, q) for m, q in zip(widths, orders))
        degrees = (0,)*dimensions  # not polynomial test functions
    else:
        raise ValueError("test_function must be polynomial or bump.")
    coordinate_scales, state_scales = np.ones(dimensions), np.ones(count)
    if rescale:
        coordinate_scales = np.array([_coordinate_scale(m, dx, p, q)
                                    for m, dx, p, q in zip(widths, spacing, degrees, orders)])
        power = max((sum(t.powers) for t in terms if t.kind == "poly"), default=1)
        if power > 1:
            for c, observations in enumerate(values):
                amplitude = float(np.max(np.abs(observations)))
                if amplitude:
                    normalized = observations/amplitude
                    log_ratio = ((1-power)*np.log(amplitude)+np.log(linalg.norm(normalized.ravel()))
                                 -np.log(linalg.norm((normalized**power).ravel())))
                    state_scales[c] = np.exp(log_ratio/(power if state_scale_rule == "printed" else power-1))
    scaled_spacing = spacing*coordinate_scales
    radii = scaled_spacing*np.asarray(widths)
    centers = tuple(g[m:len(g)-m:s] for g, m, s in zip(grids, widths, steps))
    rows = int(np.prod([len(c) for c in centers]))
    matrix = np.empty((rows, len(all_terms)))
    grouped = {}
    for index, term in enumerate(all_terms):
        key = (term.powers, term.kind, term.component, term.frequency)
        grouped.setdefault(key, []).append((index, term.derivative))
    for (powers, kind, component, frequency), columns in grouped.items():
        if kind == "poly":
            trial = np.ones(values[0].shape)
            for observations, gamma, power in zip(values, state_scales, powers):
                if power:
                    trial *= (gamma*observations)**power
        else:
            trial = (np.sin if kind == "sin" else np.cos)(frequency*values[component])
        weak = dict(_weak_columns(trial, {d for _, d in columns}, weights, radii, scaled_spacing, steps))
        for index, derivative in columns:
            matrix[:, index] = weak[derivative]
    factors = np.array([(np.prod(state_scales**np.array(t.powers)) if t.kind == "poly" else 1.0)
                        * np.prod(coordinate_scales**(-np.array(t.derivative))) for t in all_terms])
    scales = factors[:len(terms), None]/factors[len(terms):][None, :]
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(scales)):
        raise ValueError("Nonfinite weak system or coefficient scales.")
    return WeakSystem(matrix[:, :len(terms)], matrix[:, len(terms):], scales, terms,
                      state_scales, coordinate_scales, centers, widths, degrees, test_function)


def mstls(G, b, thresholds=None, *, coefficient_scales=None,
          threshold_units="scaled", allow_empty=True, selection_rule="relative"):
    """Eqs. 4.4--4.6; choose the smallest loss minimizer.

    For systems, use matrix 2-norms and the fraction of selected entries in
    the joint loss; refit each equation separately at each common threshold.
    Physical thresholding converts coefficients and bounds before selection,
    while all least-squares refits and projection losses use the scaled system.
    allow_empty=False reproduces the author's return-before-empty rule.
    """
    G, b = np.asarray(G, dtype=float), np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b[:, None]
    if G.ndim != 2 or b.ndim != 2 or len(G) != len(b) or not G.size or not b.shape[1]:
        raise ValueError("G and b must be nonempty matrices with the same row count.")
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(b)):
        raise ValueError("G and b must be finite.")
    if threshold_units not in ("physical", "scaled"):
        raise ValueError("threshold_units must be physical or scaled.")
    if selection_rule not in ("relative", "absolute"):
        raise ValueError("selection_rule must be relative or absolute.")
    scales = np.ones((G.shape[1], b.shape[1])) if coefficient_scales is None else np.asarray(coefficient_scales, dtype=float)
    if scales.shape != (G.shape[1], b.shape[1]) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Coefficient scales must be a positive finite library-by-equation matrix.")
    units = scales if threshold_units == "physical" else np.ones_like(scales)
    thresholds = np.logspace(-4, 0, 50) if thresholds is None else np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0):
        raise ValueError("Thresholds must be a nonempty sequence of positive finite values.")
    thresholds = np.unique(thresholds)
    least_squares, _, rank, _ = linalg.lstsq(G, b, lapack_driver="gelsd")
    projected_norm = linalg.norm(G@least_squares, ord=2)
    norms = linalg.norm(G, axis=0)
    bounds = np.divide(linalg.norm(b, axis=0)[None, :], norms[:, None],
                       out=np.full((G.shape[1], b.shape[1]), np.inf), where=norms[:, None] > 0)
    bounds *= units
    losses, candidates = [], []
    refits = [dict() for _ in range(b.shape[1])]
    for threshold in thresholds:
        if selection_rule == "absolute":
            lower, upper = np.full_like(bounds, threshold), np.full_like(bounds, np.inf)
        else:
            lower, upper = threshold*np.maximum(1, bounds), np.minimum(1, bounds)/threshold
        coefficients = least_squares.copy()
        for equation in range(b.shape[1]):
            active = norms > 0
            for _ in range(G.shape[1]+1):
                magnitudes = np.abs(coefficients[:, equation]*units[:, equation])
                selected = active & (magnitudes >= lower[:, equation]) & (magnitudes <= upper[:, equation])
                if np.array_equal(selected, active) or (not allow_empty and not np.any(selected)):
                    break
                active = selected
                coefficients[:, equation] = 0
                if np.any(active):
                    key = active.tobytes()
                    if key not in refits[equation]:
                        refits[equation][key] = linalg.lstsq(
                            G[:, active], b[:, equation], lapack_driver="gelsd")[0]
                    coefficients[active, equation] = refits[equation][key]
        mismatch = linalg.norm(G@(coefficients-least_squares), ord=2)
        losses.append(float((mismatch/projected_norm if projected_norm else 0)+np.count_nonzero(coefficients)/coefficients.size))
        candidates.append(coefficients)
    best = int(np.argmin(losses))
    selected = candidates[best]
    return MSTLSResult(selected*scales, selected, float(thresholds[best]), thresholds,
                       np.asarray(losses), least_squares, int(rank))


def _polish_lasso(gram, xy, alpha, theta, *, max_steps=2000):
    """Primal active-set corrections for an approximate convex LASSO solution."""
    theta = theta.copy()
    signs = np.sign(theta)
    tolerance = 1e-9
    for _ in range(max_steps):
        active = np.flatnonzero(signs)
        proposal = np.zeros_like(theta)
        if len(active):
            proposal[active] = linalg.lstsq(gram[np.ix_(active, active)],
                xy[active]-alpha*signs[active], cond=1e-13, lapack_driver="gelsd")[0]
            crosses = active[proposal[active]*signs[active] <= 0]
            if len(crosses):
                steps = theta[crosses]/(theta[crosses]-proposal[crosses])
                first = int(np.argmin(steps))
                theta += max(0., steps[first])*(proposal-theta)
                hit = crosses[first]
                theta[hit], signs[hit] = 0., 0.
                continue
        theta = proposal
        gradient = gram@theta-xy
        inactive = signs == 0
        violations = np.where(inactive, np.maximum(np.abs(gradient)-alpha, 0),
                              np.abs(gradient+alpha*signs))
        if np.max(violations) <= tolerance:
            return theta
        if not np.any(inactive) or np.max(violations[inactive]) <= tolerance:
            break
        index = int(np.argmax(np.where(inactive, violations, -np.inf)))
        signs[index] = -np.sign(gradient[index])
    raise RuntimeError("LASSO active-set polishing did not converge")


def lasso(G, b, alpha_ratios=None, *, coefficient_scales=None):
    """L1 weak regression with data-only WSINDy projection/support selection.

    Each equation minimizes ||Z theta - y||_2^2/(2*K) + alpha*||theta||_1,
    where columns of Z and y have unit RMS, without centering or an intercept.
    alpha = ratio*max(abs(Z.T@y))/K. Physical coefficients are restored after
    fitting. Returned coefficients retain LASSO shrinkage; no support refit.
    """
    import warnings
    from sklearn.linear_model import lars_path_gram, lasso_path
    G, b = np.asarray(G, dtype=float), np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b[:, None]
    if G.ndim != 2 or b.ndim != 2 or len(G) != len(b) or not G.size or not b.shape[1]:
        raise ValueError("G and b must be nonempty matrices with the same row count.")
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(b)):
        raise ValueError("G and b must be finite.")
    scales = np.ones((G.shape[1], b.shape[1])) if coefficient_scales is None else np.asarray(coefficient_scales, dtype=float)
    if scales.shape != (G.shape[1], b.shape[1]) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Coefficient scales must be a positive finite library-by-equation matrix.")
    ratios = np.logspace(-4, 0, 100) if alpha_ratios is None else np.asarray(alpha_ratios, dtype=float)
    if ratios.ndim != 1 or not len(ratios) or not np.all(np.isfinite(ratios)) or np.any(ratios <= 0) or np.any(ratios > 1):
        raise ValueError("LASSO alpha ratios must lie in (0, 1].")
    ratios = np.unique(ratios)
    rows, columns = G.shape
    column_rms = linalg.norm(G, axis=0)/np.sqrt(rows)
    response_rms = linalg.norm(b, axis=0)/np.sqrt(rows)
    active = column_rms > 0
    Z = np.asfortranarray(G[:, active]/column_rms[active])
    gram = Z.T@Z
    candidates = np.zeros((len(ratios), columns, b.shape[1]))
    normalized = np.zeros_like(candidates)
    alpha_max = np.zeros(b.shape[1])
    kkt_errors = np.zeros((len(ratios), b.shape[1]))
    messages = []
    for equation in range(b.shape[1]):
        if not response_rms[equation] or not np.any(active):
            continue
        y = b[:, equation]/response_rms[equation]
        xy = Z.T@y
        alpha_max[equation] = np.max(np.abs(xy))/rows
        if not alpha_max[equation]:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            alphas, _, path = lars_path_gram(xy, gram, n_samples=rows, method="lasso",
                alpha_min=ratios[0]*alpha_max[equation], max_iter=5000)
        messages.extend(str(w.message) for w in caught)
        theta = np.column_stack([np.interp(ratios*alpha_max[equation], alphas[::-1], p[::-1]) for p in path]).T
        theta[:, ratios == 1] = 0
        gradient = (gram@theta-xy[:, None])/rows
        penalties = ratios*alpha_max[equation]
        violations = np.where(theta != 0, np.abs(gradient+penalties*np.sign(theta)),
                              np.maximum(np.abs(gradient)-penalties, 0))
        kkt_errors[:, equation] = np.max(violations, axis=0)/alpha_max[equation]
        initial_bad = np.flatnonzero(kkt_errors[:, equation] > 1e-7)
        if len(initial_bad):
            corrected = 0
            for candidate in initial_bad:
                try:
                    theta[:, candidate] = _polish_lasso(gram/rows/alpha_max[equation],
                        xy/rows/alpha_max[equation], ratios[candidate], theta[:, candidate], max_steps=200)
                    corrected += 1
                except RuntimeError:
                    pass  # The checked coordinate-descent fallback follows.
            messages.append(f"Direct active-set polishing corrected {corrected}/{len(initial_bad)} candidates")
            gradient = (gram@theta-xy[:, None])/rows
            violations = np.where(theta != 0, np.abs(gradient+penalties*np.sign(theta)),
                                  np.maximum(np.abs(gradient)-penalties, 0))
            kkt_errors[:, equation] = np.max(violations, axis=0)/alpha_max[equation]
        if np.max(kkt_errors[:, equation]) > 1e-7:
            messages.append("LARS KKT check triggered coordinate-descent fallback")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, cd_path, _ = lasso_path(Z, y, alphas=penalties[::-1], precompute=gram,
                    Xy=xy, tol=1e-9, max_iter=200000)
            messages.extend(str(w.message) for w in caught)
            theta = cd_path[:, ::-1]
            theta[:, ratios == 1] = 0
            gradient = (gram@theta-xy[:, None])/rows
            violations = np.where(theta != 0, np.abs(gradient+penalties*np.sign(theta)),
                                  np.maximum(np.abs(gradient)-penalties, 0))
            kkt_errors[:, equation] = np.max(violations, axis=0)/alpha_max[equation]
        bad = np.flatnonzero(kkt_errors[:, equation] > 1e-7)
        if len(bad):
            messages.append(f"Active-set polishing applied to {len(bad)} candidates")
            for candidate in bad:
                theta[:, candidate] = _polish_lasso(gram/rows/alpha_max[equation],
                    xy/rows/alpha_max[equation], ratios[candidate], theta[:, candidate])
            gradient = (gram@theta-xy[:, None])/rows
            violations = np.where(theta != 0, np.abs(gradient+penalties*np.sign(theta)),
                                  np.maximum(np.abs(gradient)-penalties, 0))
            kkt_errors[:, equation] = np.max(violations, axis=0)/alpha_max[equation]
        if np.max(kkt_errors[:, equation]) > 1e-7:
            raise RuntimeError(f"LASSO path failed KKT check: {np.max(kkt_errors[:, equation]):.3g}")
        normalized[:, active, equation] = theta.T
        candidates[:, active, equation] = theta.T*response_rms[equation]/column_rms[active]
    least_squares, _, rank, _ = linalg.lstsq(G, b, lapack_driver="gelsd")
    projected = G@least_squares
    projected_norm = linalg.norm(projected, ord=2)
    losses = np.array([(linalg.norm(G@candidate-projected, ord=2)/projected_norm if projected_norm else 0)
                       +np.count_nonzero(candidate)/candidate.size for candidate in candidates])
    best = int(np.argmin(losses))
    return LassoResult(candidates[best]*scales, candidates[best], float(ratios[best]), ratios,
        losses, least_squares, int(rank), alpha_max, normalized[best], column_rms,
        response_rms, kkt_errors, tuple(messages))


def wsindy(u, spatial_grid, time, **settings):
    """Author-code baseline by default; profile='printed' selects the ablation."""
    profile = settings.pop("profile", "authors")
    regression = settings.pop("regression", "mstls")
    if regression not in ("mstls", "lasso"):
        raise ValueError("regression must be mstls or lasso.")
    if profile not in ("authors", "printed", "consistency"):
        raise ValueError("profile must be authors, printed or consistency.")
    thresholds = settings.pop("thresholds", None)
    if profile == "consistency":
        settings.setdefault("rescale", False)
        settings.setdefault("test_function", "bump")
        thresholds = np.logspace(-4, 0, 100) if thresholds is None else thresholds
    else:
        settings.setdefault("state_scale_rule", profile)
    system = build_wsindy_system(u, spatial_grid, time, **settings)
    if regression == "lasso":
        result = lasso(system.G, system.b, coefficient_scales=system.coefficient_scales)
        return result, system
    result = mstls(system.G, system.b, thresholds, coefficient_scales=system.coefficient_scales,
                   threshold_units="physical" if profile in ("authors", "consistency") else "scaled",
                   allow_empty=profile != "authors",
                   selection_rule="absolute" if profile == "consistency" else "relative")
    return result, system


def select_test_supports(u, spatial_grid, time, *, tau=1e-10, tau_hat=3.0):
    """Appendix A: weighted spectral corner fits and the decay root equation.

    This fits continuous two-line models to cumulative negative-frequency
    spectra. The fixed Table 3 supports are used by the reproduction runner.
    """
    values, grids, _ = _observations(u, (*spatial_grid, time))
    if not 0 < tau < 1 or not np.isfinite(tau_hat) or tau_hat <= 0:
        raise ValueError("Need 0 < tau < 1 and positive tau_hat.")
    all_widths, all_corners = [], []
    for observations in values:
        widths, corners = [], []
        for axis, grid in enumerate(grids):
            size = len(grid)
            other = tuple(d for d in range(observations.ndim) if d != axis)
            spectrum = np.mean(np.abs(fft.fftshift(fft.fft(observations, axis=axis), axes=axis)), axis=other)
            if not np.any(spectrum):
                raise ValueError("Spectral support selection needs a nonzero observed component.")
            cumulative = np.cumsum(spectrum)[:size//2+1]
            x = np.arange(len(cumulative), dtype=float)
            floor = max(float(cumulative[-1])*np.finfo(float).eps, np.finfo(float).tiny)
            weight = 1/np.maximum(cumulative, floor)
            errors = []
            for corner in range(1, len(x)-1):
                design = np.column_stack((np.ones(len(x)), x, np.maximum(x-corner, 0)))
                fit = linalg.lstsq(design*weight[:, None], cumulative*weight)[0]
                errors.append(linalg.norm((design@fit-cumulative)*weight))
            wave = max(size//2-(1+int(np.argmin(errors))), 1)
            def decay(m):
                return np.log((2*m-1)/m**2)*((2*np.pi*wave*m)**2-3*(size*tau_hat)**2)-2*(size*tau_hat)**2*np.log(tau)
            maximum = (size-1)//2
            if decay(2) <= 0:
                width = 2
            elif decay(maximum) >= 0:
                width = maximum
            else:
                width = int(np.ceil(optimize.brentq(decay, 2, maximum)))
            widths.append(max(2, min(width, maximum)))
            corners.append(wave)
        all_widths.append(widths)
        all_corners.append(tuple(corners))
    means = np.mean(all_widths, axis=0)
    spatial = min(int(np.ceil(np.mean(means[:-1]))), min((len(g)-1)//2 for g in grids[:-1]))
    temporal = min(int(np.ceil(means[-1])), (len(grids[-1])-1)//2)
    return (spatial,)*(len(grids)-1)+(temporal,), tuple(all_corners)
