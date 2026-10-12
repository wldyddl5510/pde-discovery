"""State estimators uhat(U): ours, PySINDy smoothers, or any function.

An estimator is called as estimator(U, grid, pde) -> (uhat, info). `estimate` also
accepts a name, a PySINDy differentiation object, or a plain function f(U, grid).
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Callable
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pysindy as ps
from scipy.ndimage import uniform_filter, uniform_filter1d
from copy import deepcopy
from .testfunctions import changepoint_supports


def noise_std(v, times=None):
    """RMS of unit-L2 sixth differences along time (last axis of one component). Samples at unequally
    spaced `times` use sixth divided differences, which still annihilate quintics."""
    if times is None or len(set(np.diff(times))) == 1:
        differences = np.diff(v, n=6, axis=-1)
        return float(np.linalg.norm(differences.ravel())/np.sqrt(differences.size*comb(12, 6)))
    nodes = sliding_window_view(np.asarray(times, dtype=float), 7)
    weights = 1/np.prod(nodes[:, :, None]-nodes[:, None, :]+np.eye(7), axis=-1)
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)
    differences = np.einsum("...ij,ij->...i", sliding_window_view(v, 7, axis=-1), weights)
    return float(np.sqrt(np.mean(differences**2)))


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
class SplitPilot(Estimator):
    """Estimate from a fraction `ratio` of the time points only, at the remaining (regression) times.

    Training times are spread evenly (a centered Bresenham pattern for ratio = a/b: time k trains iff
    ((k+1)a + (b-1)//2) mod b < a), so ratio=.5 trains on the odd times. Their noise level sets a box
    window (volume rule, odd, >= 3, and long enough in time to reach a training time), which averages
    training times only. Supports must be fixed (Tests.m), so that no choice depends on the regression
    times. The estimate is NaN at the training times; SplitSystem integrates exactly the other times.
    Space wraps for periodic PDEs.
    """
    ratio: float = .5
    prior: float = .01

    def __call__(self, U, grid, pde):
        if not 0 < self.ratio < 1 or self.prior <= 0:
            raise ValueError("SplitPilot needs 0 < ratio < 1 and prior > 0")
        if pde.tests.m is None:
            raise ValueError("SplitPilot requires fixed Tests.m; data-dependent supports would use the regression times")
        components, dims, m = _components(U), U.ndim-1, pde.tests.m
        a, b = Fraction(self.ratio).limit_denominator(1000).as_integer_ratio()
        steps = np.arange(U.shape[-2])
        training = ((steps+1)*a+(b-1)//2) % b < a
        times = steps[training]
        if len(times) < 7 or len(times) > len(steps)-2:
            raise ValueError(f"ratio {self.ratio} leaves too few training or regression times")
        reach = int(np.abs(steps[:, None]-times[None, :]).min(axis=1).max())   # farthest time from training
        estimates = [noise_std(v[..., training], times) for v in components]
        target = 2*(comb(pde.degree, 2)*max(estimates)**2/self.prior)**(1/dims)
        side = max(1, int(np.floor(min(target, np.prod([2*w+1 for w in m])**(1/dims)/2))))
        side = side if side % 2 else side+1
        low = (3,)*(dims-1)+(max(3, 2*reach+1),)
        window = tuple(min(max(l, side), 2*w+1) for l, w in zip(low, m))
        if window[-1] < 2*reach+1:
            raise ValueError(f"ratio {self.ratio} leaves regression times {reach} steps from training, beyond the support")
        modes = ["wrap" if pde.periodic else "reflect"]*(dims-1)+["reflect"]
        mask = training.astype(float)
        weight = uniform_filter1d(mask, window[-1], mode="reflect")
        pilot = np.stack([uniform_filter(v*mask, size=window, mode=modes)/weight for v in components], axis=-1)
        pilot[..., training, :] = np.nan
        return pilot, dict(ratio=self.ratio, training_times=len(times), prior=self.prior,
                           estimated_noise_std=estimates, rule_size=side, window=window, boundary=modes,
                           minimum_window_fraction=float(weight[~training].min()))


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
        return {"none": Identity, "movavg": MovingAverage, "split": SplitPilot}[estimator]()
    if isinstance(estimator, Estimator):
        return estimator
    if isinstance(estimator, ps.BaseDifferentiation):
        return FromPySINDy(estimator)
    if callable(estimator):
        return FromCallable(estimator)
    raise TypeError(f"Not a state estimator: {estimator!r}")


def estimate(U, grid, pde, estimator=None):
    return as_estimator(estimator)(U, grid, pde)
