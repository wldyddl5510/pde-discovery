"""Convolutional weak system Y ~ X W (Messenger & Bortz, arXiv:2007.02848v3, Algorithm 4.2).

Arrays follow PySINDy: U has shape (*space, time, n_components). The states used
on the two sides are chosen at construction: WeakSystem(pde, U, grid, X=uhat, Y=U).
"""
from __future__ import annotations
from functools import cached_property
import numpy as np
from scipy import fft


def _weak_columns(values, derivatives, weights, radii, spacing, strides):
    """Valid FFT convolutions with the test-function derivatives, subsampled at the query points."""
    def visit(array, axis, suffixes, prefix):
        length = array.shape[axis]
        kernel_length = weights[axis].shape[1]
        # The circular convolution equals the linear one on indices kernel_length-1:length.
        nfft = fft.next_fast_len(length)
        transformed = fft.rfft(array, n=nfft, axis=axis)
        for order in sorted({s[0] for s in suffixes}):
            kernel = weights[axis][order]*spacing[axis]/radii[axis]**order
            shape = [1]*array.ndim
            shape[axis] = -1
            convolved = fft.irfft(transformed*fft.rfft(kernel, nfft).reshape(shape), n=nfft, axis=axis)
            selection = [slice(None)]*array.ndim
            selection[axis] = slice(kernel_length-1, length, strides[axis])
            reduced = convolved[tuple(selection)].copy()
            if axis == array.ndim-1:
                yield prefix+(order,), reduced.ravel()
            else:
                yield from visit(reduced, axis+1, [s[1:] for s in suffixes if s[0] == order], prefix+(order,))
    yield from visit(values, 0, list(derivatives), ())


def monomial(V, gamma, powers, pilot=None):
    """prod_c (gamma_c V_c)^p_c, or its first-order expansion at `pilot` (exact for degree <= 1)."""
    scaled = [g*V[..., c] for c, g in enumerate(gamma)]
    if pilot is None or sum(powers) < 2:
        trial = np.ones(V.shape[:-1])
        for v, p in zip(scaled, powers):
            if p:
                trial *= v**p
        return trial
    centered = [g*pilot[..., c] for c, g in enumerate(gamma)]
    base = np.ones(V.shape[:-1])
    for v, p in zip(centered, powers):
        if p:
            base *= v**p
    trial = (1-sum(powers))*base
    for c, p in enumerate(powers):
        if p:
            partial = p*scaled[c]*centered[c]**(p-1)
            for other, q in enumerate(powers):
                if q and other != c:
                    partial *= centered[other]**q
            trial += partial
    return trial


def hermite(v, degree, variance):
    """Gaussian-unbiased monomial: E[H_j(u+e; s^2)] = u^j for e ~ N(0, s^2)."""
    previous = np.ones_like(v)
    if degree == 0:
        return previous
    current = v.copy()
    for j in range(1, degree):
        previous, current = current, v*current-j*variance*previous
    return current


class WeakSystem:
    def __init__(self, pde, U, grid, X=None, Y=None, *, scale_state=None):
        self.pde, self.U = pde, np.asarray(U, dtype=float)
        self.grid = tuple(np.asarray(g, dtype=float) for g in grid)
        self.X_state = self.U if X is None else np.asarray(X, dtype=float)
        self.Y_state = self.U if Y is None else np.asarray(Y, dtype=float)
        self.libraries, self.terms = pde.dictionary_terms()
        self.lhs = pde.lhs_terms()
        dims = len(self.grid)
        self.spacing = np.array([g[1]-g[0] for g in self.grid])
        self.orders = tuple(max(t.alpha[d] for t in (*self.terms, *self.lhs)) for d in range(dims))
        self.tests = pde.tests.resolve(self.U, self.grid, self.orders)
        self.weights = tuple(self.tests.weights(d, q) for d, q in enumerate(self.orders))
        self.centers = tuple(g[m:len(g)-m:s] for g, m, s in zip(self.grid, self.tests.m, self.tests.s))
        self.coordinate_scales, self.state_scales = np.ones(dims), np.ones(self.U.shape[-1])
        if pde.profile.rescale:
            self.coordinate_scales = np.array([self.tests.coordinate_scale(d, h, q)
                                               for d, (h, q) in enumerate(zip(self.spacing, self.orders))])
            power = max((sum(t.powers) for t in self.terms if t.powers is not None), default=1)
            if power > 1:   # gamma_u = (||U||/||U^P||)^(1/(P-1)) per component (toggle_scale=2)
                for c in range(self.U.shape[-1]):
                    v = (self.Y_state if scale_state is None else np.asarray(scale_state))[..., c]
                    amplitude = float(np.max(np.abs(v)))
                    if amplitude:
                        normalized = v/amplitude
                        log_ratio = ((1-power)*np.log(amplitude)+np.log(np.linalg.norm(normalized.ravel()))
                                     -np.log(np.linalg.norm((normalized**power).ravel())))
                        self.state_scales[c] = np.exp(log_ratio/(power-1))

    def _quadrature_values(self, values):
        return values

    def columns(self, terms, state, *, pilot=None, variance=None, rescale=True):
        """Weak columns of D^alpha f(state); monomials can be linearized at `pilot` or Hermite-corrected."""
        gamma = self.state_scales if rescale else np.ones_like(self.state_scales)
        spacing = self.spacing*(self.coordinate_scales if rescale else 1)
        radii = spacing*np.asarray(self.tests.m)
        groups, transformed = {}, {}
        for index, term in enumerate(terms):
            groups.setdefault(("feature", term.source) if term.source else ("state", term.powers), []).append((index, term))
        matrix = np.empty((int(np.prod([len(c) for c in self.centers])), len(terms)))
        for members in groups.values():
            term = members[0][1]
            if term.powers is None:
                b, c = term.source
                if b not in transformed:
                    transformed[b] = np.asarray(self.libraries[b].transform(state))
                trial = transformed[b][..., c]
            elif variance is not None:   # scalar states only
                trial = hermite(state[..., 0], term.powers[0], variance)
            else:
                trial = monomial(state, gamma, term.powers, pilot)
            weak = dict(_weak_columns(self._quadrature_values(trial), {t.alpha for _, t in members},
                                      self.weights, radii, spacing, self.tests.s))
            for index, t in members:
                matrix[:, index] = weak[t.alpha]
        return matrix

    @cached_property
    def X(self):
        return self.columns(self.terms, self.X_state)

    @cached_property
    def Y(self):
        return self.columns(self.lhs, self.Y_state)

    @cached_property
    def scales(self):
        """Physical coefficient = scales * coefficient of the scaled system, shape (n_terms, n_eq)."""
        def factor(t):
            state = np.prod(self.state_scales**np.array(t.powers)) if t.powers is not None else 1.0
            return state*np.prod(self.coordinate_scales**(-np.array(t.alpha)))
        return np.array([[factor(t)/factor(l) for l in self.lhs] for t in self.terms])


class SplitSystem(WeakSystem):
    """Weak system on the regression times of a split: the times at which the estimate X is defined.

    X comes from a split estimator such as SplitPilot and is NaN at the times it was estimated from.
    Those training times only set the state scaling; no integral uses them. The regression times are
    integrated with trapezoid weights on their own grid (2*dt each for ratio .5), while test-function
    supports, derivatives and centers stay those of the full grid.
    """
    def __init__(self, pde, U, grid, X, Y=None):
        X, U = np.asarray(X, dtype=float), np.asarray(U, dtype=float)
        others = tuple(a for a in range(X.ndim) if a != X.ndim-2)
        training = np.isnan(X).all(axis=others)
        if not training.any() or np.isnan(X).any(axis=others)[~training].any() or np.sum(~training) < 2:
            raise ValueError("SplitSystem needs X that is NaN on whole training time slices and finite at >= 2 others")
        if pde.tests.m is None:
            raise ValueError("SplitSystem requires fixed Tests.m")
        super().__init__(pde, U, grid, X=X, Y=Y, scale_state=U[..., training, :])
        kept = np.flatnonzero(~training)
        middle = (kept[1:]+kept[:-1])/2
        weights = np.zeros(len(training))
        weights[kept] = np.diff(np.r_[2*kept[0]-middle[0], middle, 2*kept[-1]-middle[-1]])
        self.time_weights = weights.reshape([1]*(U.ndim-2)+[-1])
        self.sampling_info = dict(training_times=int(training.sum()), regression_times=len(kept))

    def _quadrature_values(self, values):
        return np.where(self.time_weights > 0, self.time_weights*values, 0.)


class StrongSystem:
    """Pointwise (PDE-FIND) system: X = D^alpha f(X_state), Y = D_t^k Y_state at every s-th grid point.

    Derivatives use a PySINDy differentiation object, rebuilt with each term's order `d` and `axis`;
    the default is second-order finite differences. Coefficients are fitted in physical units.
    """
    def __init__(self, pde, U, grid, X=None, Y=None, differentiation=None):
        import pysindy as ps
        self.pde, self.U = pde, np.asarray(U, dtype=float)
        self.grid = tuple(np.asarray(g, dtype=float) for g in grid)
        self.X_state = self.U if X is None else np.asarray(X, dtype=float)
        self.Y_state = self.U if Y is None else np.asarray(Y, dtype=float)
        self.libraries, self.terms = pde.dictionary_terms()
        self.lhs = pde.lhs_terms()
        self.differentiation = differentiation or ps.FiniteDifference()
        self.points = tuple(slice(None, None, s) for s in (pde.tests.s or (1,)*len(self.grid)))
        self.scales = np.ones((len(self.terms), len(self.lhs)))

    def columns(self, terms, state):
        matrix, transformed = [], {}
        for term in terms:
            if term.powers is None:
                b, c = term.source
                if b not in transformed:
                    transformed[b] = np.asarray(self.libraries[b].transform(state))
                values = transformed[b][..., c]
            else:
                values = monomial(state, np.ones(state.shape[-1]), term.powers)
            for axis, order in enumerate(term.alpha):
                if order:   # rebuilt: FiniteDifference sizes its stencil in __init__
                    method = type(self.differentiation)(**{**self.differentiation.get_params(), "d": order, "axis": axis})
                    values = np.asarray(method(values[..., None], t=self.grid[axis]))[..., 0]
            matrix.append(values[self.points].ravel())
        return np.column_stack(matrix)

    @cached_property
    def X(self):
        return self.columns(self.terms, self.X_state)

    @cached_property
    def Y(self):
        return self.columns(self.lhs, self.Y_state)
