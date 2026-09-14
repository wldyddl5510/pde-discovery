"""PDE weak form: Y_hat=A_0 U, X_hat[:,(s-1)J+j]=A_s f_j(U), R=Y_hat-X_hat w."""
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable
from functools import cached_property
from math import comb, factorial

import numpy as np
from numpy.polynomial import Polynomial
from scipy import sparse
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class Feature:
    name: str
    f: Callable
    df: Callable
    constant: bool = False


def polynomial_dictionary(degree=3):
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    return tuple(Feature("1" if p == 0 else f"u^{p}",
                         lambda U, p=p: U**p,
                         lambda U, p=p: p*U**(p-1) if p else np.zeros_like(U),
                         p == 0) for p in range(degree+1))


@dataclass
class WeakSystem:
    """Section 2: full S*J layout, operator s first; Python j is zero-based."""
    U: np.ndarray
    A: tuple
    features: tuple
    alpha: tuple
    X_hat: np.ndarray = field(init=False)
    Y_hat: np.ndarray = field(init=False)
    admissible: np.ndarray = field(init=False)

    def __post_init__(self):
        self.U = np.asarray(self.U, dtype=float).reshape(-1).copy()
        self.A = tuple(sparse.csr_matrix(a) for a in self.A)
        if len(self.A) < 2 or len(self.alpha) != len(self.A) or not self.features:
            raise ValueError("Provide A_0, at least one A_s, matching alpha, and features")
        shape = self.A[0].shape
        if shape[1] != self.U.size or any(a.shape != shape for a in self.A):
            raise ValueError("Every A_s must have shape (K,n), with U of length n")
        if not np.isfinite(self.U).all() or any(not np.isfinite(a.data).all() for a in self.A):
            raise ValueError("Observations and operators must be finite")
        self.S, self.J = len(self.A)-1, len(self.features)
        self.K, self.n = shape
        self.Y_hat = self.A[0] @ self.U
        columns, allowed = [], []
        for s in range(1, self.S+1):
            for feature in self.features:
                # D^alpha(constant)=0 exactly. Do not learn quadrature roundoff.
                structural_zero = feature.constant and sum(self.alpha[s]) > 0
                column = np.zeros(self.K) if structural_zero else self.A[s] @ feature.f(self.U)
                columns.append(column)
                allowed.append(not structural_zero and np.linalg.norm(column) > 1e-14)
        self.X_hat = np.column_stack(columns)
        self.admissible = np.flatnonzero(allowed)

    @property
    def labels(self):
        return [f"D^{self.alpha[s]}({feature.name})"
                for s in range(1, self.S+1) for feature in self.features]

    def index(self, s, j):
        if not 1 <= s <= self.S or not 0 <= j < self.J:
            raise ValueError("Invalid (s,j)")
        return (s-1)*self.J+j

    def R(self, w):
        return self.Y_hat - self.X_hat @ np.asarray(w)

    def data_jacobian_terms(self, support):
        terms = [self.A[0]]
        for index in support:
            s0, j = divmod(int(index), self.J)
            terms.append(-self.A[s0+1].multiply(self.features[j].df(self.U)).tocsr())
        return terms


class ConvolutionSystem(WeakSystem):
    """WSINDy Tables 3--4; FFT weak form in rescaled coordinates."""
    def __init__(self, U, x, t, windows, strides, workers=1):
        U = np.asarray(U, dtype=float)
        self.workers = workers
        self.shape, self.strides = U.shape, strides
        mx, mt = windows
        dx, dt = x[1]-x[0], t[1]-t[0]
        px = max(7, int(np.ceil(np.log(1e-10)/np.log((2*mx-1)/mx**2))))
        pt = max(2, int(np.ceil(np.log(1e-10)/np.log((2*mt-1)/mt**2))))
        # Author implementation get_scales.m / wsindy_pde_fun.m.
        au = (np.linalg.norm(U**6)/np.linalg.norm(U))**(1/5)
        gx, gt = (comb(px, 3)*factorial(6))**(1/6)/(mx*dx), 1/(mt*dt)
        self.noise_scale = 1/au
        self.U, self.n = (U/au).ravel(), U.size
        self.features, self.S, self.J = polynomial_dictionary(6), 7, 7
        self.alpha = ((0, 1),)+tuple((d, 0) for d in range(7))
        self.coefficient_scale = np.array([au**(1-j)*gx**(-d)*gt for d in range(7) for j in range(7)])
        def weights(m, p, d, h, scale):
            z = np.linspace(-1, 1, 2*m+1)
            v = (Polynomial([1, 0, -1])**p).deriv(d)(z)/(m*h*scale)**d/(2*m+1)
            v[[0, -1]] = 0.
            return v
        self.wx = [weights(mx, px, d, dx, gx) for d in range(7)]
        self.wt = [weights(mt, pt, d, dt, gt) for d in range(2)]
        def apply(v, d, order_t):
            a = fftconvolve(v, self.wx[d][None, :], mode="valid", axes=1)[:, ::strides[1]]
            return fftconvolve(a, self.wt[order_t][:, None], mode="valid", axes=0)[::strides[0]].ravel()
        self.Y_hat = apply(U/au, 0, 1)
        self.K = len(self.Y_hat)
        self.X_hat = np.column_stack([np.zeros(self.K) if j == 0 and d else
                                     apply((U/au)**j, d, 0)
                                     for d in range(7) for j in range(7)])
        self.admissible = np.array([d*7+j for d in range(7) for j in range(7) if not (j == 0 and d)])

    @cached_property
    def stencil(self):
        """One shared CSR index pattern; translated kernels need no dense A_s."""
        nt, nx = self.shape
        st, sx = self.strides
        mt, mx = (len(self.wt[0])-1)//2, (len(self.wx[0])-1)//2
        starts = (np.arange(0, nt-2*mt, st)[:, None]*nx+np.arange(0, nx-2*mx, sx)).ravel()
        local = (np.arange(2*mt+1)[:, None]*nx+np.arange(2*mx+1)).ravel()
        indices = (starts[:, None]+local).astype(np.int32)
        kernels = np.array([np.outer(self.wt[dt][::-1], self.wx[dx][::-1]).ravel() for dx, dt in self.alpha])
        return indices, kernels

    def data_jacobian_terms(self, support):
        indices, kernels = self.stencil
        ptr = np.arange(self.K+1, dtype=np.int32)*indices.shape[1]
        def operator(i):
            return sparse.csr_matrix((np.tile(kernels[i], self.K), indices.ravel(), ptr), shape=(self.K, self.n))
        return [operator(0)]+[-operator(i//self.J+1).multiply(self.features[i%self.J].df(self.U)) for i in support]


def _bump(grid, center, radius, derivative=0, degree=7):
    z = (grid-center)/radius
    inside = np.abs(z) < 1
    values = np.zeros_like(z)
    poly = (Polynomial([1, 0, -1])**degree).deriv(derivative)
    values[inside] = poly(z[inside])/radius**derivative
    return values


def build_test_matrices(x, t, alpha, half_widths, centers=11, degree=7):
    """A_s includes integration-by-parts signs and trapezoid weights; U is (time, space)."""
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    hx, ht = half_widths
    if min(len(x), len(t)) < 3 or centers < 2 or min(hx, ht) <= 0:
        raise ValueError("Need increasing grids, positive windows, and centers>=2")
    if any(len(a) != 2 or any(v < 0 or int(v) != v for v in a) for a in alpha):
        raise ValueError("alpha must contain nonnegative integer (x,t) derivative orders")
    if degree <= max(max(a) for a in alpha):
        raise ValueError("Test function degree must exceed derivative orders")
    if 2*hx >= np.ptp(x) or 2*ht >= np.ptp(t):
        raise ValueError("Test windows must fit inside the space-time domain")
    def weights(grid):
        delta = np.diff(grid)
        if np.any(delta <= 0):
            raise ValueError("Grids must be strictly increasing")
        return np.r_[delta[0]/2, (delta[:-1]+delta[1:])/2, delta[-1]/2]
    qx, qt = weights(x), weights(t)
    rows = [[] for _ in alpha]
    for ct in np.linspace(t[0]+ht, t[-1]-ht, centers):
        for cx in np.linspace(x[0]+hx, x[-1]-hx, centers):
            for s, (dx, dt) in enumerate(alpha):
                vx = _bump(x, cx, hx, dx, degree)*qx
                vt = _bump(t, ct, ht, dt, degree)*qt
                values = (-1)**(dx+dt)*np.outer(vt, vx)
                rows[s].append(sparse.csr_matrix(values.reshape(1, -1)))
    return tuple(sparse.vstack(r, format="csr") for r in rows)


@dataclass
class Iterate:
    iteration: int
    seconds: float
    w: np.ndarray


@dataclass
class FitResult:
    w: np.ndarray
    history: list[Iterate]
    seconds: float
    iterations: int
    success: bool
    message: str


def least_squares(X_hat, Y_hat):
    if X_hat.shape[1] == 0:
        return np.empty(0)
    scales = np.maximum(np.linalg.norm(X_hat, axis=0), np.finfo(float).tiny)
    return np.linalg.lstsq(X_hat/scales, Y_hat, rcond=1e-12)[0]/scales


def expand(system, support, values):
    w = np.zeros(system.S*system.J)
    w[support] = values
    return w


def prepare_fit(system, support=None, initial=None):
    if support is None:
        support = system.admissible.copy()
    else:
        raw = np.asarray(support)
        if raw.size and (raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer)):
            raise ValueError("support must contain integer flat column indices")
        support = raw.astype(int)
        if len(np.unique(support)) != len(support) or not np.isin(support, system.admissible).all():
            raise ValueError("support must contain unique admissible column indices")
    X_hat = system.X_hat[:, support]
    if initial is None:
        w = least_squares(X_hat, system.Y_hat)
    else:
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (system.S*system.J,) or not np.isfinite(initial).all():
            raise ValueError("initial must be a finite full S*J coefficient vector")
        w = initial[support].copy()
    return support, X_hat, w


def fit(system, *, support=None, thresholds=None, threshold_scale=None):
    """MSTLS (WSINDy Eqs. 4.4--4.6); explicit support requests weak OLS."""
    start = perf_counter()
    indices, X_hat, w_ls = prepare_fit(system, support)
    if support is not None or len(indices) == 0:
        w = expand(system, indices, w_ls)
        elapsed = perf_counter()-start
        return FitResult(w, [Iterate(0, elapsed, w.copy())], elapsed, 0, True, "Fixed-support weak OLS")
    thresholds = np.logspace(-5, -.05, 60) if thresholds is None else np.asarray(thresholds)
    if len(thresholds) == 0 or not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError("thresholds must be finite and positive")
    scale = np.ones(len(indices)) if threshold_scale is None else np.asarray(threshold_scale)[indices]
    ratio = np.linalg.norm(system.Y_hat)/np.maximum(np.linalg.norm(X_hat, axis=0), 1e-30)*scale
    def branch(lam, record=False):
        w = w_ls.copy()
        history = [Iterate(0, perf_counter()-start, expand(system, indices, w))] if record else []
        previous = np.ones(len(w), dtype=bool)
        for iteration in range(1, len(w)+2):
            active = (np.abs(w*scale) >= lam*np.maximum(1, ratio)) & (np.abs(w*scale) <= np.minimum(1, ratio)/lam)
            w = np.zeros_like(w)
            w[active] = least_squares(X_hat[:, active], system.Y_hat)
            if record:
                history.append(Iterate(iteration, perf_counter()-start, expand(system, indices, w)))
            if np.array_equal(active, previous):
                break
            previous = active
        return w, history
    best_loss, best_threshold = np.inf, None
    for lam in sorted(thresholds):
        w, _ = branch(lam)
        loss = np.linalg.norm(X_hat@(w-w_ls))/max(np.linalg.norm(X_hat@w_ls), 1e-30)+np.count_nonzero(w)/len(w)
        if loss < best_loss:
            best_loss, best_threshold = loss, float(lam)
    w, history = branch(best_threshold, record=True)
    return FitResult(expand(system, indices, w), history, perf_counter()-start,
                     history[-1].iteration, True, "MSTLS support stable")
