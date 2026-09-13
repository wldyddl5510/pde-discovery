"""WENDy IRLS: C(w)=sigma^2 L(w)L(w)^T+ridge I; L=dR/dU."""
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from wsindy import FitResult, Iterate, expand, prepare_fit, fit as weak_fit, least_squares


def hard_threshold(w, sparsity):
    """Keep the largest raw |w_j|; no support refit."""
    if sparsity is None:
        return w.copy()
    if not isinstance(sparsity, (int, np.integer)) or not 1 <= sparsity <= len(w):
        raise ValueError("sparsity must be an integer between 1 and the library size")
    keep = np.argsort(-np.abs(w), kind="stable")[:sparsity]
    result = np.zeros_like(w)
    result[keep] = w[keep]
    return result


def l1_minimize(value_gradient, w, penalty, scale, normalizer, maxiter, tol, callback=None):
    """Smooth bound-constrained formulation w=scale*(positive-negative)."""
    q = len(w)
    def objective(z):
        value, gradient = value_gradient(scale*(z[:q]-z[q:]))
        value += penalty*np.dot(scale, z[:q]+z[q:])
        gradient = np.r_[scale*(gradient+penalty), scale*(-gradient+penalty)]
        return value/normalizer, gradient/normalizer
    result = minimize(objective, np.r_[np.maximum(w/scale, 0), np.maximum(-w/scale, 0)],
                      jac=True, bounds=[(0, None)]*(2*q), method="L-BFGS-B",
                      callback=(lambda z: callback(scale*(z[:q]-z[q:]))) if callback else None,
                      options={"maxiter": maxiter, "gtol": tol, "ftol": 1e-14, "maxls": 50,
                               "maxcor": 30})
    result.x = scale*(result.x[:q]-result.x[q:])
    return result


def penalty_reference(X, y, covariance):
    """Fixed lambda_max=||X_0.T y_0/K||_inf, whitened with C(0)."""
    if covariance is not None:
        chol = cholesky(covariance.matrix(np.zeros(X.shape[1])), lower=True)
        X, y = solve_triangular(chol, X, lower=True), solve_triangular(chol, y, lower=True)
    norm = max(np.linalg.norm(y), np.finfo(float).tiny)
    return np.max(np.abs(X.T@y))/len(y), norm/np.linalg.norm(X, axis=0), norm**2/len(y)


class ResidualCovariance:
    """Cache Gram blocks of L(w)=A_0-sum w_sj A_s diag(f_j'(U))."""
    def __init__(self, system, support, sigma, ridge=1e-10):
        if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("Covariance requires positive finite sigma and ridge")
        terms = system.data_jacobian_terms(support)
        m, K = len(terms), system.K
        self.H = np.empty((m, m, K, K))
        for a in range(m):
            for b in range(a, m):
                block = sigma**2*(terms[a]@terms[b].T).toarray()
                self.H[a, b], self.H[b, a] = block, block.T
        self.ridge = ridge*max(np.trace(self.H[0, 0])/K, np.finfo(float).tiny)

    def matrix(self, w):
        z = np.r_[1., w]
        C = np.einsum("a,b,abij->ij", z, z, self.H, optimize=True)
        C = (C+C.T)/2
        C.flat[::len(C)+1] += self.ridge
        return C

    def gradient(self, w):
        one = np.einsum("b,abij->aij", np.r_[1., w], self.H[1:], optimize=True)
        return one+one.transpose(0, 2, 1)

    def hessian(self, i, j):
        block = self.H[i+1, j+1]
        return block+block.T


def fit(system, *, sigma, sparsity=None, l1=0., support=None, initial=None,
        maxiter=100, tol=1e-8, ridge=1e-10):
    if not np.isfinite(sigma) or sigma < 0 or maxiter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Require sigma>=0, maxiter>=1, tol>0")
    if not np.isfinite(l1) or l1 < 0 or (l1 and sparsity is not None):
        raise ValueError("Use either sparsity or a nonnegative relative L1 penalty")
    start = perf_counter()
    support, X_hat, w = prepare_fit(system, support, initial)
    if len(support) == 0:
        return weak_fit(system, support=support)
    history = [Iterate(0, perf_counter()-start, expand(system, support, w))]
    covariance = ResidualCovariance(system, support, sigma, ridge) if sigma else None
    lambda_max, scale, normalizer = penalty_reference(X_hat, system.Y_hat, covariance)
    success, message = False, "IRLS iteration limit"
    for iteration in range(1, maxiter+1):
        X, y = X_hat, system.Y_hat
        if covariance is not None:
            chol = cholesky(covariance.matrix(w), lower=True)
            X, y = solve_triangular(chol, X, lower=True), solve_triangular(chol, y, lower=True)
        if l1:
            gram, rhs = X.T@X/system.K, X.T@y/system.K
            def objective(v):
                return .5*v@gram@v-rhs@v, gram@v-rhs
            inner = l1_minimize(objective, w, l1*lambda_max, scale, normalizer, 2000, tol)
            new = inner.x
            if not inner.success:
                message = "L1 subproblem: "+str(inner.message)
                w = new
                history.append(Iterate(iteration, perf_counter()-start, expand(system, support, w)))
                break
        else:
            new = least_squares(X, y)
        history.append(Iterate(iteration, perf_counter()-start, expand(system, support, new)))
        step = np.linalg.norm(new-w)
        success = step <= tol*(1+np.linalg.norm(w))
        w = new
        if success:
            message = "IRLS step tolerance"
            break
    if sparsity is not None:
        w = hard_threshold(w, sparsity)
        history.append(Iterate(iteration+1, perf_counter()-start, expand(system, support, w)))
    return FitResult(expand(system, support, w), history, perf_counter()-start,
                     iteration, bool(success), message)
