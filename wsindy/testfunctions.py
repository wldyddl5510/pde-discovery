"""Separable test functions, their supports and query strides.

Unset fields are chosen from data by `Tests.resolve`, as in the authors' code:
supports by the spectral changepoint rule (dm973/WSINDy_PDE findcorners.m),
polynomial degrees by Algorithm 4.1, strides by the paper's N/50 (1D) or N/25 rule.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import comb, factorial
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq


def poly_weights(m, order, p):
    """Derivatives 0..order of (1-r^2)^p at 2m+1 equispaced points of [-1, 1] (Leibniz form)."""
    r = np.linspace(-1, 1, 2*m+1)
    weights = np.zeros((order+1, len(r)))
    falling = [factorial(p)/factorial(p-k) for k in range(order+1)]
    for q in range(order+1):
        for left in range(q+1):
            right = q-left
            weights[q] += (comb(q, left)*falling[left]*falling[right]*(-1)**right
                           * (1+r)**(p-left)*(1-r)**(p-right))
    weights[:, (0, -1)] = 0
    return weights


def bump_weights(m, order):
    """Derivatives of exp(9/(r^2-1)) via f^(q) = f P_q/(1-r^2)^(2q) (consistency paper, Sec. 5.1)."""
    r = np.linspace(-1, 1, 2*m+1)[1:-1]
    denominator = 1-r*r
    phi = np.exp(-9/denominator)
    weights = np.zeros((order+1, 2*m+1))
    polynomial, x, one_minus_square = Polynomial([1.]), Polynomial([0., 1.]), Polynomial([1., 0., -1.])
    for q in range(order+1):
        weights[q, 1:-1] = phi*polynomial(r)/denominator**(2*q)
        polynomial = one_minus_square**2*polynomial.deriv()+(4*q*x*one_minus_square-18*x)*polynomial
    return weights


def _corner(u, axis, x):
    """Spectral corner wavenumber along one axis (findcornerpts in findcorners.m)."""
    size = len(x)
    waves = (np.arange(size)-size//2)*2*np.pi/np.ptp(x)
    xx = waves[:-(-size//2)]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(u, axis=axis), axes=axis))
    cumulative = np.cumsum(np.moveaxis(spectrum, axis, 0).reshape(size, -1).mean(axis=1))[:len(xx)]
    errors = []
    for k in range(2, len(xx)):
        a, b, xa, xb = cumulative[:k], cumulative[k-1:], xx[:k], xx[k-1:]
        line_a = a.min()+np.ptp(a)/np.ptp(xa)*(xa-xa[0])
        line_b = b.max()+np.ptp(b)/np.ptp(xb)*(xb-xb[-1])
        errors.append(np.sqrt(np.sum(((line_a-a)/a)**2)+np.sum(((line_b-b)/b)**2)))
    return max(len(xx)-(int(np.argmin(errors))+1)-3, 1)


def changepoint_supports(U, grid, tau, tau_hat):
    """Half-widths from the decay condition at each component's spectral corner (findcorners.m)."""
    def decay(m, k, n):
        return np.log((2*m-1)/m**2)*(4*np.pi**2*k**2*m**2-3*n**2*tau_hat**2)-2*n**2*tau_hat**2*np.log(tau)
    spatial, temporal, shortest, steps = [], [], len(grid[0]), len(grid[-1])
    for c in range(U.shape[-1]):
        corners = [_corner(U[..., c], d, g) for d, g in enumerate(grid)]
        for n, k in zip((len(g) for g in grid[:-1]), corners):
            low, high = np.sqrt(3)/np.pi*n/2/k*tau_hat, tau_hat*n/2/k*np.sqrt(3-8*np.log(tau))/np.pi
            spatial.append(min(brentq(decay, low, high, args=(k, n)), n/2-1))
            shortest = min(shortest, n)
        temporal.append(min(brentq(decay, 1, 2/np.sqrt(tau), args=(corners[-1], steps)), steps/2-1))
    m_x = min((shortest-1)//2, int(np.ceil(np.mean(spatial))))
    m_t = min((steps-1)//2, int(np.ceil(np.mean(temporal))))
    return (m_x,)*(len(grid)-1)+(m_t,)


@dataclass(frozen=True)
class Tests:
    """Test functions phi on 2m+1 points per axis, centered every s points.

    shape "poly" is (1-r^2)^p; "bump" is exp(9/(r^2-1)) and needs m.
    """
    m: tuple[int, ...] | None = None
    p: tuple[int, ...] | None = None
    s: tuple[int, ...] | None = None
    shape: str = "poly"
    tau: float = 1e-10
    tau_hat: float = 2.

    def resolve(self, U, grid, orders):
        """Copy with every unset field chosen from the observations U on `grid`."""
        m = self.m
        if m is None:
            if self.shape != "poly":
                raise ValueError("The changepoint rule is defined for polynomial test functions.")
            m = changepoint_supports(U, grid, self.tau, self.tau_hat)
        p = self.p
        if p is None and self.shape == "poly":
            p = tuple(max(q+1, int(np.ceil(np.log(self.tau)/np.log((2*mm-1)/mm**2)))) for mm, q in zip(m, orders))
        s = self.s
        if s is None:
            k = 50 if len(grid) == 2 else 25
            s = (max(1, len(grid[0])//k),)*(len(grid)-1)+(max(1, len(grid[-1])//k),)
        if p is not None and any(pp <= q for pp, q in zip(p, orders)):
            raise ValueError("Test degrees must exceed the derivative orders.")
        return replace(self, m=tuple(m), p=p, s=tuple(s))

    def weights(self, axis, order):
        if self.shape == "poly":
            return poly_weights(self.m[axis], order, self.p[axis])
        return bump_weights(self.m[axis], order)

    def coordinate_scale(self, axis, spacing, order):
        """Eq. 4.7 with the factorial convention of get_scales.m for odd orders."""
        if order == 0:
            return 1.0
        p = self.p[axis]
        amplitude = factorial(p)/factorial(p-order//2)*factorial(order)/factorial((order+1)//2)
        return amplitude**(1/order)/(self.m[axis]*spacing)
