"""State estimators uhat(U): ours, PySINDy smoothers, or any function.

An estimator is called as estimator(U, grid, pde) -> (uhat, info). `estimate` also
accepts a name, a PySINDy differentiation object, or a plain function f(U, grid).
"""
from __future__ import annotations
from dataclasses import dataclass
from math import comb
from typing import Callable
import numpy as np
import pysindy as ps
from scipy.ndimage import uniform_filter, uniform_filter1d
from copy import deepcopy
from .testfunctions import changepoint_supports


def noise_std(v):
    """RMS of unit-L2 sixth differences along time (last axis of one component)."""
    differences = np.diff(v, n=6, axis=-1)
    return float(np.linalg.norm(differences.ravel())/np.sqrt(differences.size*comb(12, 6)))


def _components(U):
    return [U[..., c] for c in range(U.shape[-1])]


def _half_widths(U, grid, pde):
    tests = pde.tests
    return tests.m or changepoint_supports(U, grid, tests.tau, tests.tau_hat)


class Estimator:
    def __call__(self, U, grid, pde):
        raise NotImplementedError


class Identity(Estimator):
    def __call__(self, U, grid, pde):
        return U, {}


@dataclass
class MovingAverage(Estimator):
    """Box average with reflection; window side 2*(binom(p_max,2) sigma^2/prior)^(1/D) (consistency paper,
    Sec. 5.4), capped per axis by (2m+1)/2, or by the test-function volume when rule="volume"."""
    prior: float = .01
    rule: str | None = None   # None: the PDE profile's rule

    def __call__(self, U, grid, pde):
        components = _components(U)
        estimates = [noise_std(v) for v in components]
        dims, m = U.ndim-1, _half_widths(U, grid, pde)
        target = 2*(comb(pde.degree, 2)*max(estimates)**2/self.prior)**(1/dims)
        if (self.rule or pde.profile.filter_rule) == "volume":
            cap = np.prod([2*w+1 for w in m])**(1/dims)/2
            widths = (max(1, int(np.floor(min(target, cap)))),)*dims
        else:
            widths = tuple(max(1, int(np.floor(min(target, (2*w+1)/2)))) for w in m)
        filtered = np.stack([uniform_filter(v, size=widths, mode="reflect") for v in components], axis=-1)
        return filtered, dict(filter_widths=widths, estimated_noise_std=estimates, filter_prior=self.prior,
                              filter_points=int(np.prod(widths)))


@dataclass
class CrossFitPilot(Estimator):
    """Each time parity is averaged over the other parity only; its noise estimate and window
    (volume rule, odd, >= 3) also use only the other parity. Space wraps for periodic PDEs."""
    prior: float = .01

    def __call__(self, U, grid, pde):
        components, dims, m = _components(U), U.ndim-1, _half_widths(U, grid, pde)
        modes = ["wrap" if pde.periodic else "reflect"]*(dims-1)+["reflect"]
        pilot, folds = np.empty_like(U), []
        for parity in (0, 1):
            estimates = [noise_std(v[..., 1-parity::2]) for v in components]
            target = 2*(comb(pde.degree, 2)*max(estimates)**2/self.prior)**(1/dims)
            side = max(1, int(np.floor(min(target, np.prod([2*w+1 for w in m])**(1/dims)/2))))
            side = side if side % 2 else side+1
            window = tuple(min(max(3, side), 2*w+1) for w in m)
            mask = (np.arange(U.shape[-2]) % 2 != parity).astype(float)
            weight = uniform_filter1d(mask, window[-1], mode="reflect")[parity::2]
            for c, v in enumerate(components):
                averaged = uniform_filter(v*mask, size=window, mode=modes)
                pilot[..., parity::2, c] = averaged[..., parity::2]/weight
            folds.append(dict(evaluation_parity=parity, estimated_noise_std=estimates, rule_size=side,
                              window=window, minimum_window_fraction=float(weight.min())))
        return pilot, dict(split="time2", prior=self.prior, boundary=modes, folds=folds)


@dataclass
class FromPySINDy(Estimator):
    """uhat = smoothed_x_ of a PySINDy differentiation method, applied along each axis in turn."""
    method: object
    axes: tuple[int, ...] = (-2,)   # time axis of (*space, time, n)

    def __call__(self, U, grid, pde):
        for axis in self.axes:
            method = deepcopy(self.method)
            method.axis = axis   # as pysindy's WeakSINDy sets it
            if "axis" in getattr(method, "smoother_kws", {}):   # SmoothedFiniteDifference smooths along this axis
                method.smoother_kws = dict(method.smoother_kws, axis=axis)
            method(U, t=grid[axis % U.ndim])
            U = np.asarray(method.smoothed_x_)
        return U, {}


@dataclass
class FromCallable(Estimator):
    function: Callable   # function(U, grid) -> uhat

    def __call__(self, U, grid, pde):
        return np.asarray(self.function(U, grid), dtype=float), {}


def as_estimator(estimator):
    if estimator is None:
        return Identity()
    if isinstance(estimator, str):
        return {"none": Identity, "movavg": MovingAverage, "crossfit": CrossFitPilot}[estimator]()
    if isinstance(estimator, Estimator):
        return estimator
    if isinstance(estimator, ps.BaseDifferentiation):
        return FromPySINDy(estimator)
    if callable(estimator):
        return FromCallable(estimator)
    raise TypeError(f"Not a state estimator: {estimator!r}")


def estimate(U, grid, pde, estimator=None):
    return as_estimator(estimator)(U, grid, pde)
