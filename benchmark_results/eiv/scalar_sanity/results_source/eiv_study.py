"""Section 6 scalar errors-in-variables conic estimator and plug-in tolerances.

All arrays, coefficients and residual tolerances retain physical units.
There is no state/coordinate/RMS scaling, pilot, split, path search or refit.
The observed envelopes and estimated variance define an empirical sanity
protocol, not a proved simultaneous confidence bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial
import numpy as np
from scipy import linalg, sparse
from numpy.polynomial import Polynomial, Chebyshev

from methods import (build_wsindy_system, test_function_weights,
                     bump_function_weights, _observations)


@dataclass
class EIVResult:
    coefficients: np.ndarray
    t: float
    objective: float
    status: str
    success: bool
    support: np.ndarray
    diagnostics: dict


def _nonnegative(value, name, *, positive=False):
    if (not np.isscalar(value) or not np.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} scalar.")
    return float(value)


def conic_eiv(X, y, *, mu, tau, lam, noise_bias=None,
              solver_tolerance=1e-8, support_tolerance=1e-7,
              max_iter=200, equilibrate=False):
    """Solve min ||beta||1+lam*t, ||y-(X-C)beta||inf<=tau+mu*t, ||beta||2<=t.

    X is already corrected if noise_bias is None. Coefficients are NEVER
    thresholded or refitted; support is a separate diagnostic mask using an
    absolute physical-unit tolerance. Clarabel equilibration defaults off.
    Solver failures return success=False and retain diagnostics.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if (X.ndim != 2 or y.ndim != 1 or not X.size or X.shape[0] != len(y)
            or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y))):
        raise ValueError("X and y must be finite nonempty scalar-equation arrays with matching rows.")
    mu, tau, lam = (_nonnegative(mu, "mu"), _nonnegative(tau, "tau"),
                    _nonnegative(lam, "lam", positive=True))
    tol = _nonnegative(solver_tolerance, "solver_tolerance", positive=True)
    support_tolerance = _nonnegative(support_tolerance, "support_tolerance")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(equilibrate, (bool, np.bool_)):
        raise ValueError("equilibrate must be Boolean.")
    if noise_bias is not None:
        noise_bias = np.asarray(noise_bias, dtype=float)
        if noise_bias.shape != X.shape or not np.all(np.isfinite(noise_bias)):
            raise ValueError("noise_bias must be finite and have the same shape as X.")
        X = X-noise_bias
        if not np.all(np.isfinite(X)):
            raise ValueError("Corrected design must be finite.")
    rows, columns = X.shape
    y_inf = float(linalg.norm(y, ord=np.inf))
    common = dict(mu=mu, tau=tau, lam=lam, y_inf=y_inf,
                  support_tolerance=support_tolerance, solver_tolerance=tol,
                  equilibrate=bool(equilibrate), rescale=False,
                  zero_with_t_zero_feasible=tau >= y_inf)
    # This is an exact optimizer certificate, not a numerical zero cutoff.
    # All objective terms are nonnegative and beta=t=0 attains objective zero.
    if tau >= y_inf:
        return EIVResult(np.zeros(columns), 0., 0., "AnalyticZero", True,
            np.zeros(columns, dtype=bool), dict(common, solver="analytic",
            residual_inf=y_inf, residual_violation=0., cone_violation=0.,
            objective_dual=0., duality_gap=0., iterations=0,
            certificate="tau >= ||y||_inf: beta=0,t=0 is feasible with globally minimal objective 0"))
    try:
        import clarabel
    except ImportError as exc:
        raise ImportError("Section 6 requires clarabel>=0.11,<0.12; install requirements.txt.") from exc
    # Variables are [beta (p), t (1), z (p)], with z >= |beta|.
    eye = sparse.eye(columns, format="csc")
    zero_p = sparse.csc_matrix((columns, 1))
    zero_rp = sparse.csc_matrix((rows, columns))
    neg_mu = sparse.csc_matrix(np.full((rows, 1), -mu))
    matrix = sparse.vstack([
        sparse.hstack([eye, zero_p, -eye]),
        sparse.hstack([-eye, zero_p, -eye]),
        sparse.hstack([-sparse.csc_matrix(X), neg_mu, zero_rp]),
        sparse.hstack([sparse.csc_matrix(X), neg_mu, zero_rp]),
        sparse.csc_matrix(([ -1. ], ([0], [columns])), shape=(1, 2*columns+1)),
        sparse.hstack([-eye, zero_p, sparse.csc_matrix((columns, columns))]),
    ], format="csc")
    rhs = np.concatenate([np.zeros(2*columns), tau-y, tau+y, np.zeros(columns+1)])
    objective = np.concatenate([np.zeros(columns), [lam], np.ones(columns)])
    cones = [clarabel.NonnegativeConeT(2*columns+2*rows),
             clarabel.SecondOrderConeT(columns+1)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = int(max_iter)
    settings.max_threads = 1
    settings.equilibrate_enable = bool(equilibrate)
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = tol
    settings.tol_infeas_abs = settings.tol_infeas_rel = tol
    solution = clarabel.DefaultSolver(sparse.csc_matrix((len(objective), len(objective))),
        objective, matrix, rhs, cones, settings).solve()
    beta = np.asarray(solution.x[:columns], dtype=float)
    t = float(solution.x[columns])
    status = str(solution.status)
    residual = float(linalg.norm(y-X@beta, ord=np.inf))
    beta_norm = float(linalg.norm(beta))
    residual_violation = max(0., residual-tau-mu*t)
    cone_violation = max(0., beta_norm-t)
    value = float(linalg.norm(beta, ord=1)+lam*t)
    primal_scale = max(1., y_inf, residual, abs(tau+mu*t))
    cone_scale = max(1., beta_norm, abs(t))
    gap = abs(float(solution.obj_val)-float(solution.obj_val_dual))
    valid = (np.all(np.isfinite(beta)) and np.isfinite(t) and status == "Solved"
             and residual_violation <= 10*tol*primal_scale
             and cone_violation <= 10*tol*cone_scale
             and gap <= 10*tol*max(1., abs(float(solution.obj_val)), abs(float(solution.obj_val_dual))))
    diagnostics = dict(common, solver="clarabel", solver_version=clarabel.__version__,
        residual_inf=residual, residual_violation=residual_violation,
        cone_violation=cone_violation, objective_epigraph=float(solution.obj_val),
        objective_dual=float(solution.obj_val_dual), duality_gap=gap,
        solver_primal_residual=float(solution.r_prim), solver_dual_residual=float(solution.r_dual),
        iterations=int(solution.iterations), solve_time=float(solution.solve_time))
    return EIVResult(beta, t, value, status, bool(valid), np.abs(beta)>support_tolerance, diagnostics)


@lru_cache(maxsize=128)
def _derivative_suprema(test_function, degree, max_order):
    """1D analytic-derivative extrema on [-1,1], including one-sided limits.

    Polynomial root finding is numerical. These extrema estimates are used
    by the sanity calibration, not certified interval-arithmetic bounds.
    """
    maxima = []
    if test_function == "polynomial":
        base = Chebyshev([.5, 0., -.5])**degree
        for order in range(max_order+1):
            derivative = base.deriv(order)
            roots = derivative.deriv().roots()
            candidates = [-1., 1., *[float(r.real) for r in roots
                if abs(r.imag) < 1e-8 and -1 < r.real < 1]]
            maxima.append(float(np.max(np.abs(derivative(candidates)))))
    elif test_function == "bump":
        r = Polynomial([0., 1.]); denominator = 1-r*r
        polynomials = [Polynomial([1.])]
        for q in range(max_order+1):
            current = polynomials[-1]
            polynomials.append(denominator**2*current.deriv()
                +(4*q*r*denominator-18*r)*current)
        for order in range(max_order+1):
            roots = polynomials[order+1].roots()
            candidates = np.array([0., *[float(r.real) for r in roots
                if abs(r.imag) < 1e-8 and -1 < r.real < 1]])
            denom = 1-candidates*candidates
            amplitude = np.exp(-9/denom)*np.abs(polynomials[order](candidates))/denom**(2*order)
            maxima.append(float(np.max(amplitude)))
    else:
        raise ValueError("Unsupported test function.")
    if not np.all(np.isfinite(maxima)):
        raise ValueError("Test-function derivative extrema are nonfinite.")
    return tuple(maxima)


def _tensor_kernel(weights, radii, spacing, derivative):
    value = np.array(1.)
    for w, radius, dx, order in zip(weights, radii, spacing, derivative):
        value = np.multiply.outer(value, w[order]*dx/radius**order)
    return value.ravel()


def eiv_tolerances(u, spatial_grid, time, *, system, delta=.05, sigma2=None, M=None, M1=None):
    """User-specified Section 6 tolerances, using g2=max_k||F_k||op.

    Valid translated weak kernels have identical norms, so one compact
    support represents every k, with zeros outside that support. D is the
    union of full uniform quadrature cells (volume n*Delta_z); h is their
    Euclidean diameter. M/M1 can supply true bounds; absent bounds are raw
    data proxies. Finite differences include space AND time, use second-order
    central interiors and second-order one-sided edges, without smoothing.
    """
    values, grids, spacing = _observations(u, (*spatial_grid, time))
    if len(values) != 1 or any(t.kind != "poly" or len(t.powers) != 1 for t in system.terms):
        raise ValueError("Section 6 sanity calibration currently supports scalar polynomial dictionaries only.")
    if not np.all(system.state_scales == 1) or not np.all(system.coordinate_scales == 1):
        raise ValueError("Section 6 requires rescale=False.")
    delta = _nonnegative(delta, "delta", positive=True)
    if delta >= 1:
        raise ValueError("delta must lie strictly between zero and one.")
    estimated = sigma2 is None
    if estimated:
        from experiments import estimate_noise_std
        sigma2 = estimate_noise_std(values[0])**2
    sigma2 = _nonnegative(sigma2, "sigma2")
    observed = values[0]
    observed_M = float(np.max(np.abs(observed)))
    gradients = np.gradient(observed, *spacing, edge_order=2)
    observed_M1 = float(np.max(np.sqrt(sum(g*g for g in gradients))))
    M_source, M1_source = ("observed_proxy" if M is None else "supplied_bound",
                           "raw_space_time_finite_difference_proxy" if M1 is None else "supplied_bound")
    M = observed_M if M is None else _nonnegative(M, "M")
    M1 = observed_M1 if M1 is None else _nonnegative(M1, "M1")
    terms, widths, degrees = system.terms, system.half_widths, system.degrees
    dimensions = len(spacing)
    lhs_order = system.lhs_time_order
    lhs_derivative = (0,)*(dimensions-1)+(lhs_order,)
    derivatives = sorted({term.derivative for term in terms if term.powers[0] > 0})
    J = max(term.powers[0] for term in terms)
    if not derivatives or J < 1:
        raise ValueError("Calibration needs at least one nonconstant polynomial term.")
    S = len(derivatives)
    orders = tuple(max([lhs_derivative[d], *[term.derivative[d] for term in terms]]) for d in range(dimensions))
    radii = spacing*np.asarray(widths)
    if system.test_function == "polynomial":
        weights = tuple(test_function_weights(m, q, degree=p)[0] for m, q, p in zip(widths, orders, degrees))
    else:
        weights = tuple(bump_function_weights(m, q) for m, q in zip(widths, orders))
    F = np.column_stack([_tensor_kernel(weights, radii, spacing, d) for d in derivatives])
    g2 = float(linalg.svdvals(F)[0])
    g_inf = float(np.max(linalg.norm(F, axis=1)))
    g0 = float(linalg.norm(_tensor_kernel(weights, radii, spacing, lhs_derivative)))
    one_d = [np.asarray(_derivative_suprema(system.test_function, p, q+1))/radius**np.arange(q+2)
             for p, q, radius in zip(degrees, orders, radii)]

    def norms(derivative):
        value = float(np.prod([bound[q] for bound, q in zip(one_d, derivative)]))
        partials = [float(np.prod([bound[q+int(d == axis)]
            for d, (bound, q) in enumerate(zip(one_d, derivative))])) for axis in range(dimensions)]
        return value, float(linalg.norm(partials))

    # All cells use the SAME Delta_z as the weak convolution. This declares
    # the rectangular-cell geometry rather than mixing endpoint conventions.
    delta_z = float(np.prod(spacing))
    volume = float(observed.size*delta_z)
    h = float(linalg.norm(spacing))
    lhs_sup, lhs_grad = norms(lhs_derivative)
    b_Y = volume*h*(M*lhs_grad+M1*lhs_sup)
    bounds, term_norms = [], []
    for term in terms:
        j = term.powers[0]
        phi_sup, phi_grad = norms(term.derivative)
        derivative_power = j*M**(j-1)*M1 if j else 0.
        bounds.append(volume*h*(M**j*phi_grad+derivative_power*phi_sup))
        term_norms.append(dict(phi_sup=phi_sup, gradient_sup_bound=phi_grad))
    b_x2 = float(linalg.norm(bounds))
    L = float(max(2., np.log(8*system.G.shape[0]*(S*J+1)/delta)))
    d_r = [float(np.sqrt(sum(comb(j, r)**2*M**(2*(j-r)) for j in range(r, J+1))))
           for r in range(1, J+1)]
    Gamma = float(np.sqrt(sum(factorial(r)*sigma2**r*d_r[r-1]**2 for r in range(1, J+1))))
    mu_parts = dict(quadrature=b_x2, variance=Gamma*g2*np.sqrt(L),
                    higher_order=g_inf*sum(sigma2**(r/2)*d_r[r-1]*L**(r/2) for r in range(2, J+1)))
    tau_parts = dict(quadrature=b_Y, gaussian=np.sqrt(sigma2)*g0*np.sqrt(2*L))
    mu, tau = float(sum(mu_parts.values())), float(sum(tau_parts.values()))
    if not np.all(np.isfinite([mu, tau, Gamma, M, M1])):
        raise ValueError("Nonfinite physical-unit tolerance; no automatic rescaling is applied.")
    return dict(mu=mu, tau=tau, delta=delta, sigma2=float(sigma2),
        sigma2_source="estimated_sixth_difference" if estimated else "supplied",
        M=M, M1=M1, M_source=M_source, M1_source=M1_source,
        observed_M=observed_M, observed_M1=observed_M1,
        L_delta=L, J=J, S=S, K=system.G.shape[0], b_Y=float(b_Y), b_x2=b_x2,
        b_columns=[float(v) for v in bounds], g0=g0, g2=g2, g_inf=g_inf,
        Gamma=Gamma, d_r=d_r, mu_parts={k:float(v) for k,v in mu_parts.items()},
        tau_parts={k:float(v) for k,v in tau_parts.items()},
        lhs_norms=dict(phi_sup=lhs_sup, gradient_sup_bound=lhs_grad), term_norms=term_norms,
        geometry=dict(spacing=spacing.tolist(), delta_z=delta_z, domain_volume=volume, h=h,
            cell_convention="full cells centered at observations; |D|=n*Delta_z",
            gradient="space-time Euclidean; np.gradient edge_order=2; no smoothing",
            derivative_suprema="analytic 1D derivative extrema via numerical polynomial roots; gradient bound from component suprema"),
        g2_definition="max_k ||F_k||_op", confidence_claim=False)


def fit_eiv(u, spatial_grid, time, *, library_terms, half_widths, strides,
            test_degrees=None, test_function="polynomial", lhs_time_order=1,
            delta=.05, lam=1., sigma2=None, mu=None, tau=None, M=None, M1=None,
            solver_tolerance=1e-8, support_tolerance=1e-7, equilibrate=False):
    """Scalar Section 6 fit. Missing mu/tau use the declared delta formula.

    sigma2=None estimates variance from raw observations. Inputs M and M1
    can replace empirical proxies with externally supplied true envelopes.
    No clean observations, true coefficients or noise metadata are consumed.
    """
    values, _, _ = _observations(u, (*spatial_grid, time))
    if len(values) != 1:
        raise ValueError("Only scalar PDEs are enabled for the Section 6 sanity study.")
    estimated = sigma2 is None
    if estimated:
        from experiments import estimate_noise_std
        sigma2 = estimate_noise_std(values[0])**2
    sigma2 = _nonnegative(sigma2, "sigma2")
    system = build_wsindy_system(values[0], spatial_grid, time,
        library_terms=library_terms, half_widths=half_widths, strides=strides,
        test_degrees=test_degrees, test_function=test_function,
        lhs_time_order=lhs_time_order, rescale=False, gaussian_variance=sigma2)
    calibration = eiv_tolerances(values[0], spatial_grid, time, system=system,
        sigma2=sigma2, delta=delta, M=M, M1=M1)
    calibration["sigma2_source"] = "estimated_sixth_difference" if estimated else "supplied"
    calibration["mu_formula"], calibration["tau_formula"] = calibration["mu"], calibration["tau"]
    calibration["mu_source"] = "formula" if mu is None else "supplied"
    calibration["tau_source"] = "formula" if tau is None else "supplied"
    if mu is not None:
        calibration["mu"] = _nonnegative(mu, "mu")
    if tau is not None:
        calibration["tau"] = _nonnegative(tau, "tau")
    result = conic_eiv(system.G, system.b[:, 0], mu=calibration["mu"], tau=calibration["tau"],
        lam=lam, solver_tolerance=solver_tolerance, support_tolerance=support_tolerance,
        equilibrate=equilibrate)
    return result, system, calibration
