"""Section 6 errors-in-variables estimator (scalar polynomial dictionaries, physical units).

min ||beta||_1 + lam t  s.t.  ||Y(U) - (X(U) - E[eta_noise]) beta||_inf <= tau + mu t,  ||beta||_2 <= t
(Belloni, Rosenbaum & Tsybakov 2017, Eq. 3). X(U) - E[eta_noise] is X built from Gaussian-unbiased
(Hermite) monomials of the X-side state. The derivation of tau and mu is pending; for sanity checks,
oracle=True sets them from the true beta* and noise variance, and lam is chosen on a grid.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .regression import Fit, gurobi_env


def conic(X, y, mu, tau, lam, tolerance=1e-8):
    """Second-order cone program solved by Gurobi; beta = 0 is certified when tau >= ||y||_inf.
    Returns (beta, t, status, success, diagnostics)."""
    import gurobipy as gp
    columns = X.shape[1]
    y_inf = float(np.max(np.abs(y)))
    if tau >= y_inf:
        return np.zeros(columns), 0., "AnalyticZero", True, dict(residual_inf=y_inf)
    with gp.Model(env=gurobi_env()) as model:
        model.Params.Threads = 1
        model.Params.FeasibilityTol = model.Params.OptimalityTol = max(tolerance, 1e-9)
        model.Params.BarQCPConvTol = tolerance
        beta, z, t = model.addMVar(columns, lb=-gp.GRB.INFINITY), model.addMVar(columns), model.addVar()
        model.addConstr(beta <= z)
        model.addConstr(-beta <= z)                       # z >= |beta|
        model.addConstr(y-X@beta <= tau+mu*t)
        model.addConstr(X@beta-y <= tau+mu*t)             # ||y - X beta||_inf <= tau + mu t
        model.addConstr(beta@beta <= t*t)                 # ||beta||_2 <= t (second-order cone)
        model.setObjective(z.sum()+lam*t, gp.GRB.MINIMIZE)
        model.optimize()
        solved = model.Status == gp.GRB.OPTIMAL
        beta, t = (beta.X, t.X) if model.SolCount else (np.full(columns, np.nan), np.nan)
        status, iterations = "Solved" if solved else f"GurobiStatus{model.Status}", int(model.BarIterCount)
    residual = float(np.max(np.abs(y-X@beta)))   # NaN when the solver failed
    residual_violation = max(0., residual-tau-mu*t)
    cone_violation = max(0., float(np.linalg.norm(beta))-t)
    success = (solved and np.all(np.isfinite(beta)) and np.isfinite(t)
               and residual_violation <= 10*tolerance*max(1., y_inf, residual, abs(tau+mu*t))
               and cone_violation <= 10*tolerance*max(1., float(np.linalg.norm(beta)), abs(t)))
    return beta, float(t), status, bool(success), dict(iterations=iterations, residual_inf=residual,
        residual_violation=residual_violation, cone_violation=cone_violation)


@dataclass
class EIV:
    lam: object = field(default_factory=lambda: np.logspace(-3, 3, 13))   # value or grid
    tau: float | None = None         # required unless oracle
    mu: float | None = None
    oracle: bool = False             # tau, mu (and the grid choice of lam) from the true beta* and sigma^2
    tau_share: float = 1.            # oracle: tau = share*r*, mu = (1-share)*r*/||beta*||_2, r* = ||y - X beta*||_inf
    variance: float | None = None    # None: true sigma^2 if oracle, else sixth-difference estimate
    normalize: bool = False          # solve on unit-norm columns, so ||beta||_1, ||beta||_2 are scale-free
    row_normalize: bool = False      # oracle: divide each weak equation by its noise std (Monte Carlo around u*)
    draws: int = 50
    tolerance: float = 1e-8
    support_tolerance: float = 1e-3  # diagnostic support |beta_j| > tolerance*max|beta|; no thresholding

    def regress(self, system):
        from .estimators import noise_std
        if system.U.shape[-1] != 1 or any(t.powers is None for t in system.terms):
            raise ValueError("EIV supports scalar polynomial dictionaries.")
        if self.oracle:
            truth = system.pde.truth_matrix(system.terms)[0]
            variance = system.noise_std[0]**2 if self.variance is None else self.variance
        else:
            if self.tau is None or self.mu is None:
                raise ValueError("Give tau and mu, or use oracle=True.")
            variance = noise_std(system.X_state[..., 0])**2 if self.variance is None else self.variance
        X = system.columns(system.terms, system.X_state, variance=variance, rescale=False)
        y = system.columns(system.lhs, system.Y_state, rescale=False)[:, 0]
        if self.row_normalize:
            if not self.oracle:
                raise ValueError("Row normalization uses the oracle noise model; set oracle=True.")
            std = self._row_std(system, truth, variance)
            weights = 1/std if np.all(std > 0) else np.ones_like(std)   # noise-free: no row scale
            X, y = X*weights[:, None], y*weights
        scale = np.linalg.norm(X, axis=0) if self.normalize else np.ones(X.shape[1])
        X = X/scale
        if self.oracle:
            residual = float(np.max(np.abs(y-X@(truth*scale))))
            tau, mu = self.tau_share*residual, (1-self.tau_share)*residual/np.linalg.norm(truth*scale)
        else:
            tau, mu = self.tau, self.mu
        lams = np.atleast_1d(self.lam).astype(float)
        if len(lams) > 1 and not self.oracle:
            raise ValueError("A lam grid needs oracle=True (selection by coefficient error).")
        fits = [conic(X, y, mu, tau, lam, self.tolerance) for lam in lams]
        errors = [float(np.linalg.norm(f[0]/scale-truth)/np.linalg.norm(truth)) if self.oracle else np.nan for f in fits]
        best = int(np.nanargmin(errors)) if len(lams) > 1 else 0
        beta, t, status, success, diagnostics = fits[best]
        support = np.abs(beta) > self.support_tolerance*np.max(np.abs(beta)) if np.any(beta) else np.zeros(len(beta), bool)
        return Fit((beta/scale)[None, :], dict(diagnostics, lam=float(lams[best]), lams=lams.tolist(),
                   oracle_e2_path=errors if self.oracle else None, tau=tau, mu=mu, variance=variance, t=t,
                   status=status, success=success, support=[support.tolist()]))

    def _row_std(self, system, truth, variance):
        """Standard deviation of each weak residual Y_k(U) - X_k(U) beta* over fresh noise around u*."""
        rng, active = np.random.default_rng(0), truth != 0
        terms = [t for t, a in zip(system.terms, active) if a]
        residuals = []
        for _ in range(self.draws):
            V = system.clean+np.sqrt(variance)*rng.standard_normal(system.clean.shape)
            residuals.append(system.columns(system.lhs, V, rescale=False)[:, 0]
                             - system.columns(terms, V, variance=variance, rescale=False)@truth[active])
        return np.std(residuals, axis=0)
