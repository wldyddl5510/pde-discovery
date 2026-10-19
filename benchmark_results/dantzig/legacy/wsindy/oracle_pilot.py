"""Clean-state oracle pilots for checking the nuisance-error assumption.

These are diagnostics, not estimators available from noisy observations.  The
clean state is used only to contract an existing training-only split pilot;
neither PDE coefficients nor the regression observations are inputs.
"""
from __future__ import annotations

import numpy as np


def oracle_pilot(clean, base, grid, *, mode="rate", exponent=.5, cap=1.):
    """Return ``(pilot, info)`` with the same split mask as ``base``.

    Arrays have shape ``(*space, time, components)``.  ``base`` must be NaN on
    entire training-time slices and finite elsewhere.  ``grid`` is the uniform
    physical grid, before any weak-system rescaling.

    In ``rate`` mode, the output on regression nodes is
    ``clean + alpha*(base-clean)``, with the largest contraction factor at most
    one satisfying the target, up to a floating-point safety margin.  Its
    split-quadrature weighted squared error is at most
    ``cap * n_regression**(-exponent)``.  Here ``n_regression`` counts spacetime
    nodes, not scalar component values, and the vector squared norm includes
    every component.  ``exponent > 1/4`` gives the requested little-o rate.

    Split time weights match ``SplitSystem``: the first/last weight is the
    distance to the next/previous regression time and interior weights are
    half the distance between the adjacent times, in full-grid time steps.
    They are all at least one, so this bound also controls the unweighted
    ``delta_z * sum_i ||pilot_i-clean_i||**2``.  For the usual half split the
    weights are exactly two.  Training-node errors are excluded from both sums.

    ``exact`` mode returns the clean state at regression nodes and has zero
    error.  Both guarantees are deterministic for every realization; they do
    not estimate or invoke a probability parameter delta.
    """
    if mode not in ("rate", "exact"):
        raise ValueError("mode must be 'rate' or 'exact'")
    if not np.isfinite(exponent) or exponent <= .25:
        raise ValueError("exponent must be finite and greater than 1/4")
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be finite and positive")
    clean, base = np.asarray(clean, dtype=float), np.asarray(base, dtype=float)
    if clean.shape != base.shape or clean.ndim < 2 or clean.shape[-1] < 1:
        raise ValueError("clean and base must have equal (*space, time, components) shapes")
    if not np.isfinite(clean).all():
        raise ValueError("clean must be finite at every grid node")
    grid = tuple(np.asarray(axis, dtype=float) for axis in grid)
    if len(grid) != clean.ndim-1:
        raise ValueError("grid must have one coordinate vector per spacetime axis")
    spacing = []
    for axis, size in zip(grid, clean.shape[:-1]):
        if axis.ndim != 1 or len(axis) != size or len(axis) < 2 or not np.isfinite(axis).all():
            raise ValueError("grid coordinate vectors must be finite, match the state, and have >= 2 points")
        steps = np.diff(axis)
        if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=0):
            raise ValueError("oracle_pilot requires an increasing uniform physical grid")
        spacing.append(float(steps[0]))
    delta_z = float(np.prod(spacing))
    if not np.isfinite(delta_z) or delta_z <= 0:
        raise ValueError("physical grid cell volume must be finite and positive")

    others = tuple(axis for axis in range(base.ndim) if axis != base.ndim-2)
    training = np.isnan(base).all(axis=others)
    regression = ~training
    if not training.any() or np.sum(regression) < 2 or not np.isfinite(base[..., regression, :]).all():
        raise ValueError("base must be NaN on whole training time slices and finite at >= 2 others")
    kept = np.flatnonzero(regression)
    middle = (kept[1:]+kept[:-1])/2
    time_weights = np.diff(np.r_[2*kept[0]-middle[0], middle, 2*kept[-1]-middle[-1]])
    n_regression = int(np.prod(clean.shape[:-2], dtype=np.int64))*len(kept)
    n_total = int(np.prod(clean.shape[:-1], dtype=np.int64))
    target = float(cap*n_regression**(-exponent))
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target energy must be representable as a positive finite float")
    true = clean[..., regression, :]
    difference = base[..., regression, :]-true

    def energies(values):
        squared = np.sum(values*values, axis=-1)
        return float(delta_z*np.sum(squared*time_weights)), float(delta_z*np.sum(squared))

    with np.errstate(over="ignore", invalid="ignore"):
        base_energy, base_unweighted = energies(difference)
    if not np.isfinite(base_energy) or not np.isfinite(base_unweighted):
        raise ValueError("base error energy must be finite")
    alpha = 0. if mode == "exact" else min(1., np.sqrt(target/base_energy)) if base_energy else 1.
    # Leave a small margin when a contraction is necessary.  Rechecking the
    # constructed array also covers cancellation/rounding near a large clean u.
    if 0 < alpha < 1:
        alpha *= 1-64*np.finfo(float).eps
    if alpha == 1:
        values = base[..., regression, :].copy()
    else:
        values = true+alpha*difference
    energy, unweighted = energies(values-true)
    for _ in range(8):
        if energy <= target:
            break
        # Adding a small correction to a large clean value can exceed the
        # cap by a few ulps even though alpha's exact-arithmetic error fits.
        # Correct the measured excess gradually, retaining essentially the
        # requested error radius when it is numerically representable.
        alpha *= np.sqrt(target/energy)*(1-1e-6)
        values = true+alpha*difference
        energy, unweighted = energies(values-true)
    # A discrete floating-point lattice can prevent an arbitrarily precise
    # radius, so retain a guaranteed terminating fallback for extreme scales.
    while energy > target:
        alpha *= .5
        values = true+alpha*difference
        energy, unweighted = energies(values-true)
    pilot = np.full_like(clean, np.nan)
    pilot[..., regression, :] = values
    info = dict(mode=mode, alpha=float(alpha), exponent=float(exponent), cap=float(cap),
                n_regression=n_regression, n_total=n_total, components=int(clean.shape[-1]),
                regression_times=len(kept), training_times=int(training.sum()),
                delta_z=delta_z, target_energy=target, base_energy=base_energy,
                base_energy_unweighted=base_unweighted, energy=energy,
                energy_unweighted=unweighted,
                error_over_n_quarter=float(energy*n_regression**.25),
                bound_over_n_quarter=float(cap*n_regression**(.25-exponent)),
                time_weight_min=float(time_weights.min()), time_weight_max=float(time_weights.max()))
    return pilot, info
