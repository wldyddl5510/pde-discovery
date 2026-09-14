"""WENDy IRLS: C(w)=sigma^2 L(w)L(w)^T+ridge I; L=dR/dU."""
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
from copy import copy

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize, OptimizeResult
from scipy import sparse

from wsindy import ConvolutionSystem, FitResult, Iterate, expand, prepare_fit, fit as weak_fit, least_squares


def hard_threshold(w, sparsity):
    """Keep the largest input |w_j|; no support refit."""
    if sparsity is None:
        return w.copy()
    if not isinstance(sparsity, (int, np.integer)) or not 1 <= sparsity <= len(w):
        raise ValueError("sparsity must be an integer between 1 and the library size")
    keep = np.argsort(-np.abs(w), kind="stable")[:sparsity]
    result = np.zeros_like(w)
    result[keep] = w[keep]
    return result


def l1_minimize(value_gradient, w, penalty, scale, maxiter, tol, callback=None):
    """Smooth bound-constrained formulation w=scale*(positive-negative)."""
    q = len(w)
    reference = max(np.max(np.abs(value_gradient(np.zeros(q))[1])), penalty)
    normalizer = max(reference*np.max(scale), np.finfo(float).tiny)
    tolerance = max(10*np.finfo(float).eps*reference, min(tol*reference, .01*penalty))
    def objective(z):
        value, gradient = value_gradient(scale*(z[:q]-z[q:]))
        value += penalty*np.dot(scale, z[:q]+z[q:])
        gradient = np.r_[scale*(gradient+penalty), scale*(-gradient+penalty)]
        return value/normalizer, gradient/normalizer
    result = minimize(objective, np.r_[np.maximum(w/scale, 0), np.maximum(-w/scale, 0)],
                      jac=True, bounds=[(0, None)]*(2*q), method="L-BFGS-B",
                      callback=(lambda z: callback(scale*(z[:q]-z[q:]))) if callback else None,
                      options={"maxiter": maxiter, "gtol": tolerance*np.min(scale)/normalizer,
                               "ftol": 0., "maxls": 50,
                               "maxcor": 30})
    result.x = scale*(result.x[:q]-result.x[q:])
    gradient = value_gradient(result.x)[1]
    violation = np.where(result.x != 0, np.abs(gradient+penalty*np.sign(result.x)),
                         np.maximum(np.abs(gradient)-penalty, 0))
    result.success = np.max(violation) <= tolerance
    result.message = ("L1 KKT tolerance" if result.success else
                      f"L1 KKT residual {np.max(violation)/reference:.3g}: {result.message}")
    return result


def orthonormalize(system, condition=1e4, information=.95, max_rows=200):
    """SVD of separable WSINDy tests; apply the same row map to every weak operator."""
    if not isinstance(system, ConvolutionSystem) or hasattr(system, "projection"):
        return system
    result = copy(system)
    bases, derivatives, singular = [], [], []
    for n, kernels, stride in zip(system.shape, (system.wt, system.wx), system.strides):
        starts = np.arange(0, n-len(kernels[0])+1, stride)
        operators = []
        for kernel in kernels:
            matrix = np.zeros((len(starts), n))
            for row, start in zip(matrix, starts):
                row[start:start+len(kernel)] = kernel[::-1]
            operators.append(matrix)
        left, values, _ = np.linalg.svd(operators[0], full_matrices=False)
        basis = left.T/values[:, None]
        bases.append(basis)
        derivatives.append([basis@matrix for matrix in operators])
        singular.append(values)
    values = np.outer(*singular).ravel()
    order = np.argsort(-values)
    values = values[order]
    count = min(max_rows, np.count_nonzero(values >= values[0]/condition),
                np.searchsorted(np.cumsum(values), information*np.sum(values))+1)
    it, ix = np.unravel_index(order[:count], (len(singular[0]), len(singular[1])))
    result.projection = np.einsum("ki,kj->kij", bases[0][it], bases[1][ix]).reshape(count, system.K)
    result.X_hat, result.Y_hat = result.projection@system.X_hat, result.projection@system.Y_hat
    result.K = count
    result.test_information = np.sum(values[:count])/np.sum(values)
    result.test_derivatives = [(derivatives[0][dt][it], derivatives[1][dx][ix]) for dx, dt in system.alpha]
    return result


def penalty_reference(X, y, covariance):
    """Fixed lambda_max=||X_0.T y_0/K||_inf, whitened with C(0)."""
    if covariance is not None:
        chol = cholesky(covariance.matrix(np.zeros(X.shape[1])), lower=True)
        X, y = solve_triangular(chol, X, lower=True), solve_triangular(chol, y, lower=True)
    norm = max(np.linalg.norm(y), np.finfo(float).tiny)
    return np.max(np.abs(X.T@y))/len(y), norm/np.linalg.norm(X, axis=0)


def lasso(X, y, penalty, tol=1e-8, maxiter=2000):
    """Active-set LASSO; SVD solves avoid squaring the design condition number."""
    scale = np.sqrt(len(y))/np.linalg.norm(X, axis=0)
    A, b, lam = X*scale/np.sqrt(len(y)), y/np.sqrt(len(y)), penalty*scale
    w, signs = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    reference = max(np.max(np.abs(X.T@y/len(y))), penalty)
    tolerance = max(10*np.finfo(float).eps*reference, min(tol*reference, .01*penalty))
    for iteration in range(maxiter):
        gradient = A.T@(A@w-b)
        violation = np.where(w != 0, np.abs(gradient+lam*np.sign(w)),
                             np.maximum(np.abs(gradient)-lam, 0))/scale
        if np.max(violation) <= tolerance:
            break
        inactive = signs == 0
        if inactive.any():
            index = np.argmax(np.where(inactive, np.abs(gradient)-lam, -np.inf))
            if abs(gradient[index]) > lam[index]:
                signs[index] = -np.sign(gradient[index])
        for _ in range(len(w)+1):
            ids = np.flatnonzero(signs)
            left, singular, right = np.linalg.svd(A[:, ids], full_matrices=False)
            candidate = right.T@((left.T@b)/singular-(right@(lam[ids]*signs[ids]))/singular**2)
            crossing = candidate*signs[ids] <= 0
            if not crossing.any():
                w[ids] = candidate
                break
            crossed = ids[crossing]
            difference = w[crossed]-candidate[crossing]
            ratio = np.divide(w[crossed], difference, out=np.zeros(len(crossed)), where=difference != 0)
            first = np.argmin(ratio)
            w[ids] += ratio[first]*(candidate-w[ids])
            w[crossed[first]], signs[crossed[first]] = 0., 0.
    gradient = X.T@(X@(w*scale)-y)/len(y)
    violation = np.where(w != 0, np.abs(gradient+penalty*np.sign(w)),
                         np.maximum(np.abs(gradient)-penalty, 0))
    success = np.max(violation) <= tolerance
    return OptimizeResult(x=w*scale, nit=iteration+1, success=success,
                          message="LASSO KKT tolerance" if success else
                          f"LASSO KKT residual {np.max(violation)/reference:.3g}; iteration limit")


class ResidualCovariance:
    """C(w) via cached Gram blocks or shared convolution stencils."""
    def __init__(self, system, support, sigma, ridge=1e-10):
        if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("Covariance requires positive finite sigma and ridge")
        self.direct = isinstance(system, ConvolutionSystem)
        if self.direct:
            self.system, self.support, self.sigma = system, support, sigma
            self.projected = hasattr(system, "projection")
            self.powers = np.array([j*system.U**(j-1) for j in range(1, system.J)])
            self._w = None
            if self.projected:
                self.operators = [np.einsum("kt,kx->ktx", t, x).reshape(system.K, system.n)
                                  for t, x in system.test_derivatives]
                self.ridge = ridge*sigma**2*np.sum(self.operators[0]**2)/system.K
                return
            self.indices, self.kernels = system.stencil
            ptr = np.arange(system.K+1, dtype=np.int32)*self.indices.shape[1]
            self.L = sparse.csr_matrix((np.empty(self.indices.size), self.indices.ravel(), ptr),
                                       shape=(system.K, system.n))
            self.ridge = ridge*sigma**2*np.sum(self.kernels[0]**2)
            return
        terms = system.data_jacobian_terms(support)
        m, K = len(terms), system.K
        self.H = np.empty((m, m, K, K))
        for a in range(m):
            for b in range(a, m):
                block = sigma**2*(terms[a]@terms[b].T).toarray()
                self.H[a, b], self.H[b, a] = block, block.T
        self.ridge = ridge*max(np.trace(self.H[0, 0])/K, np.finfo(float).tiny)

    def matrix(self, w):
        if self.direct:
            if self._w is not None and np.array_equal(w, self._w):
                return self._C
            s = self.system
            self.L = self.factor(w)
            if self.projected:
                C = self.sigma**2*(self.L@self.L.T)
            else:
                self.LT = self.L.T.tocsr()
                C = self.sigma**2*self.blocks(lambda a, b: (self.L[a:b]@self.LT).toarray())
            C.flat[::s.K+1] += self.ridge
            self._w, self._C = w.copy(), C
            return C
        z = np.r_[1., w]
        C = np.einsum("a,b,abij->ij", z, z, self.H, optimize=True)
        C = (C+C.T)/2
        C.flat[::len(C)+1] += self.ridge
        return C

    def factor(self, w, direction=False):
        s = self.system
        full = expand(s, self.support, w).reshape(s.S, s.J)
        if self.projected:
            values = np.zeros_like(self.operators[0]) if direction else self.operators[0].copy()
        else:
            values = np.zeros(self.indices.shape) if direction else np.broadcast_to(self.kernels[0], self.indices.shape).copy()
        for d in range(s.S):
            derivative = full[d, 1:]@self.powers
            values -= (derivative*self.operators[d+1] if self.projected else
                       derivative[self.indices]*self.kernels[d+1])
        if self.projected:
            return values
        return sparse.csr_matrix((values.ravel(), self.L.indices, self.L.indptr), shape=self.L.shape)

    def blocks(self, function):
        edges = np.linspace(0, self.system.K, self.system.workers+1, dtype=int)
        if self.system.workers == 1:
            return function(0, self.system.K)
        with ThreadPoolExecutor(self.system.workers) as pool:
            return np.concatenate(list(pool.map(lambda ab: function(*ab), zip(edges[:-1], edges[1:]))))

    def contract_gradient(self, w, Q, factor=None):
        """Exact trace(Q dC/dw), without storing 43 dense derivative matrices."""
        if not self.direct:
            return np.einsum("ij,aij->a", Q, self.gradient(w))
        self.matrix(w)
        s = self.system
        if self.projected:
            weighted = Q@(self.L if factor is None else factor)
            gradient = np.zeros((s.S, s.J))
            for d in range(s.S):
                h = np.einsum("kn,kn->n", weighted, self.operators[d+1])
                gradient[d, 1:] = -2*self.sigma**2*(self.powers@h)
            return gradient.ravel()[self.support]
        LT = self.LT if factor is None else factor.T.tocsr()
        weighted = self.blocks(lambda a, b: (LT@Q[:, a:b])[self.indices[a:b], np.arange(b-a)[:, None]])
        gradient = np.zeros(s.S*s.J)
        for d in range(s.S):
            h = np.bincount(self.indices.ravel(), weights=(weighted*self.kernels[d+1]).ravel(), minlength=s.n)
            gradient[d*s.J+1:(d+1)*s.J] = -2*self.sigma**2*(self.powers@h)
        return gradient[self.support]

    def gradient(self, w):
        one = np.einsum("b,abij->aij", np.r_[1., w], self.H[1:], optimize=True)
        return one+one.transpose(0, 2, 1)

    def hessian(self, i, j):
        block = self.H[i+1, j+1]
        return block+block.T


def fit(system, *, sigma, sparsity=None, l1=0., support=None, initial=None,
        maxiter=100, tol=1e-8, ridge=1e-10, time_limit=np.inf):
    if not np.isfinite(sigma) or sigma < 0 or maxiter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Require sigma>=0, maxiter>=1, tol>0")
    if not np.isfinite(l1) or l1 < 0 or (l1 and sparsity is not None):
        raise ValueError("Use either sparsity or a nonnegative relative L1 penalty")
    if np.isnan(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive")
    start = perf_counter()
    system = orthonormalize(system)
    support, X_hat, w = prepare_fit(system, support, initial)
    if l1 and initial is None:
        w[:] = 0.
    if len(support) == 0:
        return weak_fit(system, support=support)
    history = [Iterate(0, perf_counter()-start, expand(system, support, w))]
    covariance = ResidualCovariance(system, support, sigma, ridge) if sigma else None
    if l1:
        lambda_max, _ = penalty_reference(X_hat, system.Y_hat, covariance)
    success, message = False, "IRLS iteration limit"
    for iteration in range(1, maxiter+1):
        if perf_counter()-start >= time_limit:
            message = "Time limit"
            break
        X, y = X_hat, system.Y_hat
        if covariance is not None:
            chol = cholesky(covariance.matrix(w), lower=True)
            X, y = solve_triangular(chol, X, lower=True), solve_triangular(chol, y, lower=True)
        if l1:
            inner = lasso(X, y, l1*lambda_max, tol)
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
    iteration = len(history)-1
    if sparsity is not None:
        w = hard_threshold(w, sparsity)
        history.append(Iterate(iteration+1, perf_counter()-start, expand(system, support, w)))
    return FitResult(expand(system, support, w), history, perf_counter()-start,
                     iteration, bool(success), message)
