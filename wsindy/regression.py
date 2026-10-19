"""Sparse regressions on a WeakSystem. Every regression returns physical coefficients (n_eq, n_terms).

A regression is a name ("mstls", "lasso", "eiv"), one of our objects (anything with
`regress(system)`), or any PySINDy/scikit-learn optimizer (anything with `fit` and `coef_`).
MSTLS and LassoPath take a loss that supplies the design and the least-squares solve:
"ls" uses X and Y as constructed; "debiased" linearizes the monomials at the X-side state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import warnings
import numpy as np
from scipy import linalg
from sklearn.base import clone


@dataclass
class Fit:
    coef: np.ndarray                         # (n_eq, n_terms), physical units
    info: dict = field(default_factory=dict) # selection details for the trial record


class LeastSquares:
    def matrices(self, system):
        return system.X, system.Y

    def solve(self, X, y):
        solution, _, rank, _ = linalg.lstsq(X, y, lapack_driver="gelsd")
        return solution, rank


class Debiased(LeastSquares):
    """Section 5: f(pilot) + grad f(pilot).(U - pilot) for monomials of degree >= 2, pilot = X-side state."""
    def matrices(self, system):
        return system.columns(system.terms, system.U, pilot=system.X_state), system.Y


LOSSES = {"ls": LeastSquares, "debiased": Debiased}


def _loss(loss):
    return LOSSES[loss]() if isinstance(loss, str) else loss


@dataclass
class MSTLS:
    """Modified STLS over a threshold path; bounds act on physical coefficients (authors' code).

    Unset options follow the PDE profile: 50 relative thresholds without empty supports
    (original suite), or 100 absolute thresholds allowing empty supports (consistency suite).
    """
    thresholds: np.ndarray | None = None
    rule: str | None = None             # "relative": lambda*max(1,|b|/|G_j|) bounds; "absolute"
    allow_empty: bool | None = None
    loss: object = "ls"

    def regress(self, system):
        profile, loss = system.pde.profile, _loss(self.loss)
        thresholds = np.unique(profile.thresholds if self.thresholds is None else self.thresholds)
        rule = self.rule or profile.rule
        allow_empty = profile.allow_empty if self.allow_empty is None else self.allow_empty
        G, b = loss.matrices(system)
        units = system.scales
        least_squares, rank = loss.solve(G, b)
        projected_norm = linalg.norm(G@least_squares, ord=2)
        norms = linalg.norm(G, axis=0)
        bounds = np.divide(linalg.norm(b, axis=0)[None, :], norms[:, None],
                           out=np.full(units.shape, np.inf), where=norms[:, None] > 0)*units
        losses, candidates = [], []
        refits = [dict() for _ in range(b.shape[1])]
        for threshold in thresholds:
            if rule == "absolute":
                lower, upper = np.full_like(bounds, threshold), np.full_like(bounds, np.inf)
            else:
                lower, upper = threshold*np.maximum(1, bounds), np.minimum(1, bounds)/threshold
            coefficients = least_squares.copy()
            for e in range(b.shape[1]):
                active = norms > 0
                for _ in range(G.shape[1]+1):
                    magnitudes = np.abs(coefficients[:, e]*units[:, e])
                    selected = active & (magnitudes >= lower[:, e]) & (magnitudes <= upper[:, e])
                    if np.array_equal(selected, active) or (not allow_empty and not np.any(selected)):
                        break
                    active = selected
                    coefficients[:, e] = 0
                    if np.any(active):
                        key = active.tobytes()
                        if key not in refits[e]:
                            refits[e][key] = loss.solve(G[:, active], b[:, e])[0]
                        coefficients[active, e] = refits[e][key]
            # Eq. 4.4 with matrix 2-norms: relative projection error + fraction of nonzero entries.
            mismatch = linalg.norm(G@(coefficients-least_squares), ord=2)
            losses.append(float((mismatch/projected_norm if projected_norm else 0)
                                + np.count_nonzero(coefficients)/coefficients.size))
            candidates.append(coefficients)
        best = int(np.argmin(losses))
        return Fit((candidates[best]*units).T, dict(threshold=float(thresholds[best]),
                   thresholds=thresholds.tolist(), losses=losses, loss=losses[best], rank=int(rank)))


def _polish(gram, xy, alpha, theta, max_steps=2000):
    """Primal active-set corrections for an approximate LASSO solution."""
    theta, signs, tolerance = theta.copy(), np.sign(theta), 1e-9
    for _ in range(max_steps):
        active = np.flatnonzero(signs)
        proposal = np.zeros_like(theta)
        if len(active):
            proposal[active] = linalg.lstsq(gram[np.ix_(active, active)], xy[active]-alpha*signs[active],
                                            cond=1e-13, lapack_driver="gelsd")[0]
            crosses = active[proposal[active]*signs[active] <= 0]
            if len(crosses):
                steps = theta[crosses]/(theta[crosses]-proposal[crosses])
                first = int(np.argmin(steps))
                theta += max(0., steps[first])*(proposal-theta)
                theta[crosses[first]], signs[crosses[first]] = 0., 0.
                continue
        theta = proposal
        gradient = gram@theta-xy
        inactive = signs == 0
        violations = np.where(inactive, np.maximum(np.abs(gradient)-alpha, 0), np.abs(gradient+alpha*signs))
        if np.max(violations) <= tolerance:
            return theta
        if not np.any(inactive) or np.max(violations[inactive]) <= tolerance:
            break
        index = int(np.argmax(np.where(inactive, violations, -np.inf)))
        signs[index] = -np.sign(gradient[index])
    raise RuntimeError("LASSO active-set polishing did not converge")


@dataclass
class LassoPath:
    """||Z theta - y||^2/(2K) + alpha ||theta||_1 on RMS-normalized columns/responses, alpha = ratio*alpha_max.

    The ratio is chosen by the MSTLS selection loss; penalized coefficients are kept (no refit).
    Every candidate must satisfy the KKT conditions to 1e-7*alpha_max.
    """
    ratios: np.ndarray = field(default_factory=lambda: np.logspace(-4, 0, 100))
    loss: object = "ls"

    def _path(self, gram, xy, rows, ratios, alpha_max, messages):
        from sklearn.linear_model import lars_path_gram
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            alphas, _, path = lars_path_gram(xy, gram, n_samples=rows, method="lasso",
                                             alpha_min=ratios[0]*alpha_max, max_iter=5000)
        messages.extend(str(w.message) for w in caught)
        theta = np.column_stack([np.interp(ratios*alpha_max, alphas[::-1], p[::-1]) for p in path]).T
        theta[:, ratios == 1] = 0
        return theta

    def regress(self, system):
        from sklearn.linear_model import lasso_path
        G, b = _loss(self.loss).matrices(system)
        ratios = np.unique(self.ratios)
        rows, columns = G.shape
        column_rms, response_rms = linalg.norm(G, axis=0)/np.sqrt(rows), linalg.norm(b, axis=0)/np.sqrt(rows)
        active = column_rms > 0
        Z = np.asfortranarray(G[:, active]/column_rms[active])
        gram = Z.T@Z
        candidates = np.zeros((len(ratios), columns, b.shape[1]))
        normalized = np.zeros_like(candidates)
        alpha_max, kkt = np.zeros(b.shape[1]), np.zeros((len(ratios), b.shape[1]))
        messages = []
        for e in range(b.shape[1]):
            if not response_rms[e] or not np.any(active):
                continue
            y = b[:, e]/response_rms[e]
            xy = Z.T@y
            alpha_max[e] = np.max(np.abs(xy))/rows
            if not alpha_max[e]:
                continue
            penalties = ratios*alpha_max[e]

            def violations(theta):
                gradient = (gram@theta-xy[:, None])/rows
                return np.max(np.where(theta != 0, np.abs(gradient+penalties*np.sign(theta)),
                                       np.maximum(np.abs(gradient)-penalties, 0)), axis=0)/alpha_max[e]

            def polish(theta, bad, max_steps):
                fixed = 0
                for k in bad:
                    try:
                        theta[:, k] = _polish(gram/rows/alpha_max[e], xy/rows/alpha_max[e], ratios[k], theta[:, k], max_steps)
                        fixed += 1
                    except RuntimeError:
                        if max_steps > 200:
                            raise
                return fixed

            theta = self._path(gram, xy, rows, ratios, alpha_max[e], messages)
            kkt[:, e] = violations(theta)
            bad = np.flatnonzero(kkt[:, e] > 1e-7)
            if len(bad):
                messages.append(f"Direct active-set polishing corrected {polish(theta, bad, 200)}/{len(bad)} candidates")
                kkt[:, e] = violations(theta)
            if np.max(kkt[:, e]) > 1e-7:
                messages.append("LARS KKT check triggered coordinate-descent fallback")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    _, path, _ = lasso_path(Z, y, alphas=penalties[::-1], precompute=gram, Xy=xy, tol=1e-9, max_iter=200000)
                messages.extend(str(w.message) for w in caught)
                theta = path[:, ::-1]
                theta[:, ratios == 1] = 0
                kkt[:, e] = violations(theta)
            bad = np.flatnonzero(kkt[:, e] > 1e-7)
            if len(bad):
                messages.append(f"Active-set polishing applied to {len(bad)} candidates")
                polish(theta, bad, 2000)
                kkt[:, e] = violations(theta)
            if np.max(kkt[:, e]) > 1e-7:
                raise RuntimeError(f"LASSO path failed KKT check: {np.max(kkt[:, e]):.3g}")
            normalized[:, active, e] = theta.T
            candidates[:, active, e] = theta.T*response_rms[e]/column_rms[active]
        least_squares, rank = _loss(self.loss).solve(G, b)
        projected = G@least_squares
        projected_norm = linalg.norm(projected, ord=2)
        losses = [float((linalg.norm(G@c-projected, ord=2)/projected_norm if projected_norm else 0)
                        + np.count_nonzero(c)/c.size) for c in candidates]
        best = int(np.argmin(losses))
        return Fit((candidates[best]*system.scales).T, dict(
            threshold=float(ratios[best]), thresholds=ratios.tolist(), losses=losses, loss=losses[best],
            rank=int(rank), alpha_max=alpha_max.tolist(), selected_alpha=(ratios[best]*alpha_max).tolist(),
            column_rms=column_rms.tolist(), response_rms=response_rms.tolist(),
            normalized_coefficients=normalized[best].tolist(), kkt_errors=kkt.tolist(), solver_warnings=messages))


@lru_cache(maxsize=None)
def gurobi_env():
    """One Gurobi license check per process, without the banner."""
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


@dataclass
class BestSubset:
    """Best subset selection by mixed-integer optimization (Bertsimas, King & Mazumder, arXiv:1507.03133;
    MIOSR, Bertsimas & Gurnee, arXiv:2206.00176): min ||Y - X W||_F^2 with at most r nonzero
    coefficients in total (int r) or r[e] in equation e (tuple). Supports are SOS-1 encoded as in
    MIOSR; the warm start is the least-squares fit restricted to its r largest entries; the chosen
    support is refitted by the loss.

    r may also map PDE names to bounds. r=None: for each equation, r = 1, 2, ... until the
    selection loss of Eq. 4.4 (as in MSTLS) increases; the previous r is kept.

    threshold=lambda adds MSTLS's relative magnitude prior on physical coefficients: a selected
    term needs lambda*max(1, B_j) <= |c_j| <= min(1, B_j)/lambda, B_j = |b|/|G_j| in physical units.
    The constrained minimizer is returned without the least-squares refit.
    """
    r: int | tuple[int, ...] | dict | None = None
    loss: object = "ls"
    time_limit: float = 30.   # seconds per MIO; on timeout the incumbent is used and its gap reported
    mip_gap: float = 1e-4
    threshold: float | None = None

    def _bounds(self, G, b, units):
        """MSTLS's relative bounds on |coefficients| of the scaled system; inf for zero columns."""
        if self.threshold is None:
            return None
        norms = linalg.norm(G, axis=0)
        B = np.divide(linalg.norm(b, axis=0)[None, :], norms[:, None], out=np.full(units.shape, np.inf),
                      where=norms[:, None] > 0)*units
        return self.threshold*np.maximum(1, B)/units, np.minimum(1, B)/self.threshold/units

    @staticmethod
    def _refit(loss, G, b, support):
        coefficients = np.zeros(support.shape)
        for e in range(b.shape[1]):
            if support[:, e].any():
                coefficients[support[:, e], e] = loss.solve(G[:, support[:, e]], b[:, e])[0]
        return coefficients

    def _solve(self, loss, G, b, r, bounds=None):
        import gurobipy as gp
        J, E = G.shape[1], b.shape[1]
        if not np.isscalar(r) and len(r) != E:
            raise ValueError("Give one sparsity bound per equation.")
        largest = np.abs(loss.solve(G, b)[0])
        keep = np.zeros(largest.shape, dtype=bool)
        if np.isscalar(r):
            keep.flat[np.argsort(largest, axis=None)[::-1][:r]] = True
        else:
            for e, r_e in enumerate(r):
                keep[np.argsort(largest[:, e])[::-1][:r_e], e] = True
        start = self._refit(loss, G, b, keep)
        # ||b - G W||^2 = ||Q^T b - R W||^2 + const for the reduced G = QR (R has min(K, J) rows;
        # exact when K <= J): conditioning of G rather than G^T G.
        Q, R = np.linalg.qr(G)
        with gp.Model(env=gurobi_env()) as model:
            model.Params.Threads, model.Params.TimeLimit, model.Params.MIPGap = 1, self.time_limit, self.mip_gap
            W = model.addMVar((J, E), lb=-gp.GRB.INFINITY)
            zero = model.addMVar((J, E), vtype=gp.GRB.BINARY)   # 1 forces the coefficient to zero
            residual = model.addMVar((R.shape[0], E), lb=-gp.GRB.INFINITY)
            for w, z in zip(W.tolist(), zero.tolist()):
                for pair in zip(w, z):
                    model.addSOS(gp.GRB.SOS_TYPE1, list(pair))
            model.addConstr(residual == Q.T@b-R@W)
            if bounds is not None:   # sign binaries: lower <= |W| <= upper on the support, W = 0 off it
                lower, upper = bounds
                usable = np.isfinite(lower)
                lower, upper = np.where(usable, lower, 0), np.where(usable, upper, 0)
                positive, negative = (model.addMVar((J, E), vtype=gp.GRB.BINARY) for _ in range(2))
                model.addConstr(positive+negative == 1-zero)
                model.addConstr(W >= lower*positive-upper*negative)
                model.addConstr(W <= upper*positive-lower*negative)
                zero.lb = (~usable).astype(float)
            if np.isscalar(r):
                model.addConstr(zero.sum() >= J*E-r)
            else:
                for e, r_e in enumerate(r):
                    model.addConstr(zero[:, e].sum() >= J-r_e)
            model.setObjective((residual*residual).sum(), gp.GRB.MINIMIZE)
            W.Start, zero.Start = start, (start == 0).astype(float)
            model.optimize()
            if not model.SolCount:
                raise RuntimeError(f"Gurobi found no solution (status {model.Status}).")
            support, gap = zero.X < .5, float(model.MIPGap)
            info = dict(optimal=model.Status == gp.GRB.OPTIMAL, mip_gap=gap if np.isfinite(gap) else None,
                        solve_seconds=float(model.Runtime))
            constrained = np.where(support, W.X, 0.)
        return (self._refit(loss, G, b, support) if bounds is None else constrained), info

    def regress(self, system):
        loss = _loss(self.loss)
        G, b = loss.matrices(system)
        r = self.r[system.pde.name] if isinstance(self.r, dict) else self.r
        bounds = self._bounds(G, b, system.scales)
        if r is not None:
            W, info = self._solve(loss, G, b, r, bounds)
            return Fit((W*system.scales).T, dict(info, r=r if np.isscalar(r) else list(r)))
        W = np.zeros((G.shape[1], b.shape[1]))
        info = dict(r=[], losses=[], optimal=[], mip_gap=[], solve_seconds=0.)
        for e in range(b.shape[1]):   # least squares separates by equation
            y = b[:, [e]]
            least_squares = loss.solve(G, y)[0]
            projected = linalg.norm(G@least_squares)
            losses, chosen = [], None
            for r in range(1, G.shape[1]+1):
                w, step = self._solve(loss, G, y, r, bounds and tuple(x[:, [e]] for x in bounds))
                info["solve_seconds"] += step["solve_seconds"]
                losses.append(float((linalg.norm(G@(w-least_squares))/projected if projected else 0)
                                    + np.count_nonzero(w)/w.size))
                if len(losses) > 1 and losses[-1] > losses[-2]:
                    break
                chosen = (r, w, step)
            r, W[:, [e]], step = chosen
            info["r"].append(r); info["losses"].append(losses)
            info["optimal"].append(step["optimal"]); info["mip_gap"].append(step["mip_gap"])
        return Fit((W*system.scales).T, info)


@dataclass
class FromAPI:
    """Any PySINDy optimizer or scikit-learn regressor, fitted on the scaled system."""
    optimizer: object

    def regress(self, system):
        optimizer = clone(self.optimizer).fit(system.X, system.Y)
        coef = np.reshape(optimizer.coef_, (system.Y.shape[1], system.X.shape[1]))
        return Fit(coef*system.scales.T, dict(optimizer=repr(self.optimizer)))


def as_regression(regression):
    if isinstance(regression, str):
        from .eiv import EIV
        return {"mstls": MSTLS, "lasso": LassoPath, "eiv": EIV}[regression]()
    if hasattr(regression, "regress"):
        return regression
    if hasattr(regression, "fit"):
        return FromAPI(regression)
    raise TypeError(f"Not a regression: {regression!r}")


def fit(system, regression="mstls"):
    return as_regression(regression).regress(system)
