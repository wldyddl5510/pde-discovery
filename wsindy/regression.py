"""Sparse regressions on a WeakSystem. Every regression returns physical coefficients (n_eq, n_terms).

A regression is a name ("mstls", "lasso", "eiv"), one of our objects (anything with
`regress(system)`), or any PySINDy/scikit-learn optimizer (anything with `fit` and `coef_`).
MSTLS and LassoPath take a loss that supplies the design and the least-squares solve:
"ls" uses X and Y as constructed; "debiased" linearizes the monomials at the X-side state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
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
