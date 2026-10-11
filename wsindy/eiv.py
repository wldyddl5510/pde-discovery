"""Section 6 errors-in-variables estimator (scalar polynomial dictionaries, physical units).

min ||beta||_1 + lam t  s.t.  ||y - X beta||_inf <= tau + mu t,  ||beta||_2 <= t,
where X holds Gaussian-unbiased (Hermite) monomials of the X-side state, and mu, tau
follow the declared delta formula with data proxies for the envelopes M, M1.
These tolerances carry no proved confidence guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy import linalg, sparse
from .regression import Fit


def _clarabel(objective, A, rhs, linear, columns, tolerance):
    import clarabel
    settings = clarabel.DefaultSettings()
    settings.verbose, settings.max_iter, settings.max_threads, settings.equilibrate_enable = False, 200, 1, False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = tolerance
    settings.tol_infeas_abs = settings.tol_infeas_rel = tolerance
    solution = clarabel.DefaultSolver(sparse.csc_matrix((len(objective),)*2), objective, A, rhs,
        [clarabel.NonnegativeConeT(linear), clarabel.SecondOrderConeT(columns+1)], settings).solve()
    gap = abs(float(solution.obj_val)-float(solution.obj_val_dual))
    gap_ok = gap <= 10*tolerance*max(1., abs(float(solution.obj_val)), abs(float(solution.obj_val_dual)))
    return np.asarray(solution.x), str(solution.status) == "Solved" and gap_ok, dict(
        status=str(solution.status), duality_gap=gap, iterations=int(solution.iterations))


_GUROBI_ENV = None


def _gurobi(objective, A, rhs, linear, columns, tolerance):
    """Linear rows as A v <= rhs; the cone block as ||beta||_2^2 <= t^2 with t >= 0 (Gurobi SOC)."""
    global _GUROBI_ENV
    import gurobipy as gp
    if _GUROBI_ENV is None:   # one license check per process, without the banner
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam("OutputFlag", 0)
        _GUROBI_ENV.start()
    with gp.Model(env=_GUROBI_ENV) as model:
        model.Params.Threads = 1
        model.Params.FeasibilityTol = model.Params.OptimalityTol = max(tolerance, 1e-9)
        model.Params.BarQCPConvTol = tolerance
        lower = np.full(2*columns+1, -gp.GRB.INFINITY)
        lower[columns] = 0.
        v = model.addMVar(2*columns+1, lb=lower)
        model.addConstr(A[:linear]@v <= rhs[:linear])
        model.addConstr(v[:columns]@v[:columns] <= v[columns:columns+1]@v[columns:columns+1])
        model.setObjective(objective@v, gp.GRB.MINIMIZE)
        model.optimize()
        solved = model.Status == gp.GRB.OPTIMAL
        return (np.asarray(v.X) if model.SolCount else np.full(2*columns+1, np.nan), solved,
                dict(status="Solved" if solved else f"GurobiStatus{model.Status}", iterations=int(model.BarIterCount)))


def conic(X, y, mu, tau, lam, tolerance=1e-8, solver="clarabel"):
    """Returns (beta, t, status, success, diagnostics); beta = 0 is certified when tau >= ||y||_inf."""
    rows, columns = X.shape
    y_inf = float(linalg.norm(y, ord=np.inf))
    if tau >= y_inf:
        return np.zeros(columns), 0., "AnalyticZero", True, dict(residual_inf=y_inf)
    eye, zero_p = sparse.eye(columns, format="csc"), sparse.csc_matrix((columns, 1))
    zero_rp, neg_mu = sparse.csc_matrix((rows, columns)), sparse.csc_matrix(np.full((rows, 1), -mu))
    # Variables [beta, t, z] with z >= |beta|.
    A = sparse.vstack([sparse.hstack([eye, zero_p, -eye]), sparse.hstack([-eye, zero_p, -eye]),
                       sparse.hstack([-sparse.csc_matrix(X), neg_mu, zero_rp]),
                       sparse.hstack([sparse.csc_matrix(X), neg_mu, zero_rp]),
                       sparse.csc_matrix(([-1.], ([0], [columns])), shape=(1, 2*columns+1)),
                       sparse.hstack([-eye, zero_p, sparse.csc_matrix((columns, columns))])], format="csc")
    rhs = np.concatenate([np.zeros(2*columns), tau-y, tau+y, np.zeros(columns+1)])
    objective = np.concatenate([np.zeros(columns), [lam], np.ones(columns)])
    solve = {"clarabel": _clarabel, "gurobi": _gurobi}[solver]
    x, solved, diagnostics = solve(objective, A, rhs, 2*columns+2*rows, columns, tolerance)
    beta, t = x[:columns].astype(float), float(x[columns])
    residual = float(np.max(np.abs(y-X@beta)))   # NaN when the solver failed
    residual_violation = max(0., residual-tau-mu*t)
    cone_violation = max(0., float(np.linalg.norm(beta))-t)
    success = (solved and np.all(np.isfinite(beta)) and np.isfinite(t)
               and residual_violation <= 10*tolerance*max(1., y_inf, residual, abs(tau+mu*t))
               and cone_violation <= 10*tolerance*max(1., float(linalg.norm(beta)), abs(t)))
    status = diagnostics.pop("status")
    return beta, t, status, bool(success), dict(diagnostics, solver=solver, residual_inf=residual,
        residual_violation=residual_violation, cone_violation=cone_violation)


@lru_cache(maxsize=128)
def _derivative_suprema(shape, degree, max_order):
    """max |phi^(q)| on [-1, 1] for q = 0..max_order (numerical polynomial roots)."""
    maxima = []
    if shape == "poly":
        base = Chebyshev([.5, 0., -.5])**degree
        for order in range(max_order+1):
            derivative = base.deriv(order)
            roots = derivative.deriv().roots()
            candidates = [-1., 1., *[float(r.real) for r in roots if abs(r.imag) < 1e-8 and -1 < r.real < 1]]
            maxima.append(float(np.max(np.abs(derivative(candidates)))))
    else:
        r = Polynomial([0., 1.])
        polynomials = [Polynomial([1.])]
        for q in range(max_order+1):
            polynomials.append((1-r*r)**2*polynomials[-1].deriv()+(4*q*r*(1-r*r)-18*r)*polynomials[-1])
        for order in range(max_order+1):
            roots = polynomials[order+1].roots()
            candidates = np.array([0., *[float(x.real) for x in roots if abs(x.imag) < 1e-8 and -1 < x.real < 1]])
            denominator = 1-candidates*candidates
            maxima.append(float(np.max(np.exp(-9/denominator)*np.abs(polynomials[order](candidates))/denominator**(2*order))))
    return tuple(maxima)


def _kernel(weights, radii, spacing, alpha):
    value = np.array(1.)
    for w, radius, h, order in zip(weights, radii, spacing, alpha):
        value = np.multiply.outer(value, w[order]*h/radius**order)
    return value.ravel()


def tolerances(system, u, rows, variance, delta, M=None, M1=None):
    """Declared mu, tau (physical units); g2 = max_k ||F_k||_op over translated test kernels."""
    spacing, tests, terms = system.spacing, system.tests, system.terms
    M = float(np.max(np.abs(u))) if M is None else M
    M1 = float(np.max(np.sqrt(sum(g*g for g in np.gradient(u, *spacing, edge_order=2))))) if M1 is None else M1
    lhs_alpha = system.lhs[0].alpha
    derivatives = sorted({t.alpha for t in terms if t.powers[0] > 0})
    J, S = max(t.powers[0] for t in terms), len(derivatives)
    radii = spacing*np.asarray(tests.m)
    F = np.column_stack([_kernel(system.weights, radii, spacing, d) for d in derivatives])
    g2, g_inf = float(linalg.svdvals(F)[0]), float(np.max(linalg.norm(F, axis=1)))
    g0 = float(linalg.norm(_kernel(system.weights, radii, spacing, lhs_alpha)))
    degrees = tests.p or (0,)*len(spacing)
    one_d = [np.asarray(_derivative_suprema(tests.shape, p, q+1))/radius**np.arange(q+2)
             for p, q, radius in zip(degrees, system.orders, radii)]

    def norms(alpha):
        value = float(np.prod([bound[q] for bound, q in zip(one_d, alpha)]))
        partials = [float(np.prod([bound[q+int(d == axis)] for d, (bound, q) in enumerate(zip(one_d, alpha))]))
                    for axis in range(len(spacing))]
        return value, float(linalg.norm(partials))

    volume, h = float(u.size*np.prod(spacing)), float(linalg.norm(spacing))
    lhs_sup, lhs_grad = norms(lhs_alpha)
    b_Y = volume*h*(M*lhs_grad+M1*lhs_sup)
    bounds = []
    for t in terms:
        j = t.powers[0]
        phi_sup, phi_grad = norms(t.alpha)
        bounds.append(volume*h*(M**j*phi_grad+(j*M**(j-1)*M1 if j else 0.)*phi_sup))
    L = float(max(2., np.log(8*rows*(S*J+1)/delta)))
    d_r = [float(np.sqrt(sum(comb(j, r)**2*M**(2*(j-r)) for j in range(r, J+1)))) for r in range(1, J+1)]
    Gamma = float(np.sqrt(sum(factorial(r)*variance**r*d_r[r-1]**2 for r in range(1, J+1))))
    mu = float(linalg.norm(bounds)+Gamma*g2*np.sqrt(L)
               + g_inf*sum(variance**(r/2)*d_r[r-1]*L**(r/2) for r in range(2, J+1)))
    tau = float(b_Y+np.sqrt(variance)*g0*np.sqrt(2*L))
    return dict(mu=mu, tau=tau, M=M, M1=M1, L_delta=L, g0=g0, g2=g2, g_inf=g_inf, Gamma=Gamma)


@dataclass
class EIV:
    delta: float = .05
    lam: float = 1.
    mu: float | None = None          # None: declared formula
    tau: float | None = None
    variance: float | None = None    # None: sixth-difference estimate from the X-side state
    M: float | None = None
    M1: float | None = None
    tolerance: float = 1e-8
    solver: str = "clarabel"         # or "gurobi" (gurobipy with a license)
    support_tolerance: float = 1e-7  # diagnostic support mask; coefficients are not thresholded

    def regress(self, system):
        from .estimators import noise_std
        if system.U.shape[-1] != 1 or any(t.powers is None for t in system.terms):
            raise ValueError("EIV supports scalar polynomial dictionaries.")
        u = system.X_state[..., 0]
        variance = noise_std(u)**2 if self.variance is None else self.variance
        X = system.columns(system.terms, system.X_state, variance=variance, rescale=False)
        y = system.columns(system.lhs, system.Y_state, rescale=False)[:, 0]
        calibration = tolerances(system, u, X.shape[0], variance, self.delta, self.M, self.M1)
        mu = calibration["mu"] if self.mu is None else self.mu
        tau = calibration["tau"] if self.tau is None else self.tau
        beta, t, status, success, diagnostics = conic(X, y, mu, tau, self.lam, self.tolerance, self.solver)
        return Fit(beta[None, :], dict(calibration, mu=mu, tau=tau, variance=variance, t=t, status=status,
                                       success=success, support=[(np.abs(beta) > self.support_tolerance).tolist()],
                                       **diagnostics))
