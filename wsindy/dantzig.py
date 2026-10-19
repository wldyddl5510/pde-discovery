"""Discrete Dantzig selector (Eqs. 9-10) on the weak score b - H beta of ||Y - X beta||^2/(2K) (Eq. 8).

loss="debiased" adds the score's derivative along U - uhat at the pilot uhat (the X-side state):
H = (X'X + X'dX + dX'X)/K, b = (X'Y(uhat) + dX'Y(uhat) + X'dY)/K, dX = D X(uhat)[U - uhat], dY = Y(U) - Y(uhat).
The constraint is on the standardized score D^-1 (b - H beta), D = diag(column RMS of X): in physical units
a degree-6 score spans ~30 orders of magnitude, so a common lambda would constrain only its top rows.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .regression import Fit, gurobi_env


def score(system, loss="debiased"):
    """(H, b, w) on the scaled system: score b - H beta of shape (n_terms, n_eq), w = column RMS of X."""
    X = system.X
    if loss == "debiased":
        Y = system.columns(system.lhs, system.X_state)
        dX = system.columns(system.terms, system.U, pilot=system.X_state)-X
        dY = system.columns(system.lhs, system.U)-Y   # exact: the left-hand side is linear
        H, b = X.T@X+X.T@dX+dX.T@X, X.T@Y+dX.T@Y+X.T@dY
    else:
        H, b = X.T@X, X.T@system.Y
    return H/len(X), b/len(X), np.sqrt(np.diag(X.T@X)/len(X))


def min_residual(A, y, k, start=None, time_limit=20.):
    """min ||y - A beta||_inf s.t. ||beta||_0 <= k; SOS-1 pairs (beta_j, 1 - a_j) as in Eq. 10.
    Returns (beta, optimal); on timeout, the incumbent."""
    import gurobipy as gp
    p = A.shape[1]
    with gp.Model(env=gurobi_env()) as model:
        model.Params.Threads, model.Params.TimeLimit = 1, time_limit
        model.Params.FeasibilityTol = model.Params.IntFeasTol = 1e-9
        beta, zero, radius = model.addMVar(p, lb=-gp.GRB.INFINITY), model.addMVar(p, vtype=gp.GRB.BINARY), model.addVar()
        for pair in zip(beta.tolist(), zero.tolist()):
            model.addSOS(gp.GRB.SOS_TYPE1, list(pair))
        model.addConstr(A@beta-y <= radius)
        model.addConstr(y-A@beta <= radius)
        model.addConstr(zero.sum() >= p-k)
        model.setObjective(radius)
        if start is not None:
            beta.Start, zero.Start = start, (start == 0).astype(float)
        model.optimize()
        return np.where(zero.X < .5, beta.X, 0.), model.Status == gp.GRB.OPTIMAL


@dataclass
class Dantzig:
    """min ||beta||_0 s.t. ||D^-1 (b - H beta)||_inf <= lam ||D^-1 b||_inf, without refit.

    The lambda path is exact: beta_k minimizes r_k = ||D^-1 (b - H beta)||_inf over ||beta||_0 <= k (one MIP
    each), and lambda selects beta_k for the smallest k with r_k <= lambda. Sanity checks with the truth
    beta* (lambda_n is not derived yet): lam="theory" is lambda* = ||D^-1 (b - H beta*)||_inf, the smallest
    lambda at which beta* is feasible; lam="oracle" picks the beta_k, k <= ||beta*||_0 + 2, closest to beta*.
    """
    lam: float | str = "oracle"
    loss: str = "debiased"
    time_limit: float = 20.   # seconds per MIP; on timeout the incumbent is used and reported

    def regress(self, system):
        H, b, w = score(system, self.loss)
        units = system.scales   # physical = units * scaled coefficient
        active = w > 0
        A = H[np.ix_(active, active)]/np.outer(w[active], w[active])
        columns = np.max(np.abs(A), axis=0)
        A = A/columns
        coef = np.zeros((b.shape[1], len(w)))
        info = dict(lam=[], k=[], radii=[], optimal=[], e2_path=[])
        for e in range(b.shape[1]):
            y = b[active, e]/w[active]
            scale = np.max(np.abs(y)) or 1.
            y = y/scale

            def physical(theta):   # back from the standardized coordinates in which y - A theta is the score
                beta = np.zeros(len(w))
                beta[active] = theta*scale/columns/w[active]
                return beta*units[:, e]

            lam, k_max = self.lam, len(y)
            if isinstance(lam, str):
                truth = system.pde.truth_matrix(system.terms)[e]
                star = truth[active]/units[active, e]*w[active]*columns/scale
                lam = float(np.max(np.abs(y-A@star)))
                k_max = len(y) if self.lam == "theory" else min(len(y), np.count_nonzero(truth)+2)
            thetas, radii, optimal = [np.zeros(len(y))], [1.], [True]
            while len(thetas) <= k_max and (self.lam == "oracle" or radii[-1] > lam+1e-9):
                theta, solved = min_residual(A, y, len(thetas), thetas[-1], self.time_limit)
                thetas.append(theta)
                radii.append(float(np.max(np.abs(y-A@theta))))
                optimal.append(solved)
            errors = [float(np.linalg.norm(physical(t)-truth)/np.linalg.norm(truth)) for t in thetas] \
                if isinstance(self.lam, str) else []
            k = int(np.argmin(errors)) if self.lam == "oracle" else len(thetas)-1
            coef[e] = physical(thetas[k])
            for key, value in zip(info, (lam, k, radii, optimal, errors)):
                info[key].append(value)
        return Fit(coef, info)
