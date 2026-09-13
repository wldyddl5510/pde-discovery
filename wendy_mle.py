"""WENDy-MLE: minimize [logdet C(w)+R(w)^T C(w)^(-1) R(w)]/(2K)."""
from time import perf_counter

import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize

from wsindy import FitResult, Iterate, expand, prepare_fit, fit as weak_fit
from wendy import ResidualCovariance, hard_threshold, l1_minimize, penalty_reference, fit as irls_fit


class WeakLikelihood:
    """Cached likelihood with analytic gradient and Hessian."""
    def __init__(self, X_hat, Y_hat, covariance):
        self.X_hat, self.Y_hat, self.covariance = X_hat, Y_hat, covariance
        self._w = None

    def evaluate(self, w, with_hessian=True):
        if self._w is not None and np.array_equal(w, self._w) and (not with_hessian or self._hessian is not None):
            return self._value, self._gradient, self._hessian
        K, q = self.X_hat.shape
        R = self.Y_hat-self.X_hat@w
        chol = cholesky(self.covariance.matrix(w), lower=True)
        inverse = cho_solve((chol, True), np.eye(K))
        v = cho_solve((chol, True), R)
        dC = self.covariance.gradient(w)
        Q = inverse-np.outer(v, v)
        value = (np.log(np.diag(chol)).sum()+.5*R@v)/K
        gradient = (-self.X_hat.T@v+.5*np.einsum("ij,aij->a", Q, dC))/K
        hessian = np.empty((q, q)) if with_hessian else None
        for j in range(q if with_hessian else 0):
            dv = inverse@(-self.X_hat[:, j]-dC[j]@v)
            dQ = -inverse@dC[j]@inverse-np.outer(dv, v)-np.outer(v, dv)
            hessian[:, j] = -self.X_hat.T@dv+.5*np.einsum("ij,aij->a", dQ, dC)
            for i in range(q):
                hessian[i, j] += .5*np.sum(Q*self.covariance.hessian(i, j))
        if with_hessian:
            hessian = (hessian+hessian.T)/(2*K)
        self._w = np.array(w, copy=True)
        self._value, self._gradient, self._hessian = float(value), gradient, hessian
        return self._value, self._gradient, self._hessian

    def value(self, w):
        return self.evaluate(w, False)[0]

    def gradient(self, w):
        return self.evaluate(w, False)[1]

    def hessian(self, w):
        return self.evaluate(w)[2]


def fit(system, *, sigma, sparsity=None, l1=0., support=None, initial=None,
        maxiter=100, tol=1e-8, ridge=1e-10):
    if not np.isfinite(sigma) or sigma < 0 or maxiter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Require sigma>=0, maxiter>=1, tol>0")
    if not np.isfinite(l1) or l1 < 0 or (l1 and sparsity is not None):
        raise ValueError("Use either sparsity or a nonnegative relative L1 penalty")
    if sigma == 0:
        return irls_fit(system, sigma=0, sparsity=sparsity, l1=l1, support=support,
                        initial=initial, maxiter=maxiter, tol=tol, ridge=ridge)
    start = perf_counter()
    support, X_hat, w = prepare_fit(system, support, initial)
    if len(support) == 0:
        return weak_fit(system, support=support)
    history = [Iterate(0, perf_counter()-start, expand(system, support, w))]
    covariance = ResidualCovariance(system, support, sigma, ridge)
    objective = WeakLikelihood(X_hat, system.Y_hat, covariance)
    def callback(w):
        history.append(Iterate(len(history), perf_counter()-start, expand(system, support, w)))
    if l1:
        lambda_max, scale, normalizer = penalty_reference(X_hat, system.Y_hat, covariance)
        result = l1_minimize(lambda v: objective.evaluate(v, False)[:2], w, l1*lambda_max,
                             scale, normalizer, maxiter, tol, callback)
    else:
        result = minimize(objective.value, w, jac=objective.gradient, hess=objective.hessian,
                          method="trust-exact", callback=callback,
                          options={"gtol": tol, "maxiter": maxiter})
    full = expand(system, support, hard_threshold(result.x, sparsity))
    elapsed = perf_counter()-start
    if not np.array_equal(full, history[-1].w):
        history.append(Iterate(int(result.nit)+(sparsity is not None), elapsed, full.copy()))
    return FitResult(full, history, elapsed, int(result.nit), bool(result.success), str(result.message))
