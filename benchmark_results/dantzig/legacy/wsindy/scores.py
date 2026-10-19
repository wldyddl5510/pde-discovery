"""Physical-unit weak scores for the discrete Dantzig selector.

The debiased score is the first-order expansion of ``X.T @ (Y-X beta)/K``
at the pilot state.  In particular, it does not contain the quadratic products
``dX.T @ dY`` and ``dX.T @ dX`` that arise from ordinary least squares on the
first-order corrected weak design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreMatrices:
    """A score ``b-H beta`` in physical coefficient units, one column per PDE.

    ``X`` and ``Y`` are weak matrices at the expansion state; ``dX`` and ``dY``
    are directional increments.  For ``plugin`` and ``raw``, the increments
    are zero.  The weak matrices remain available for support-refit diagnostics.
    """

    H: np.ndarray
    b: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    dX: np.ndarray
    dY: np.ndarray
    variant: str

    @property
    def n_rows(self):
        return self.X.shape[0]

    @property
    def refit_X(self):
        """First-order corrected physical design (ordinary design for raw/plugin)."""
        return self.X+self.dX

    @property
    def refit_Y(self):
        """Observed physical response for debiased/raw; pilot response for plugin."""
        return self.Y+self.dY


def score_matrices(system, *, variant="debiased"):
    """Construct debiased, pilot plug-in, or observed-data weak scores.

    ``debiased`` evaluates the weak design and linear left-hand side at
    ``system.X_state`` and differentiates along ``system.U-system.X_state``.
    ``plugin`` uses the same pilot and quadrature without the correction.
    ``raw`` uses observations on both sides, retaining the system's quadrature.
    Thus a SplitSystem provides a same-split ablation; a WeakSystem provides a
    full-data raw score.  ``system.Y_state`` does not select the score's response.

    All columns use ``rescale=False``: neither the original WSINDy state and
    coordinate scaling nor column normalization changes the common tolerance.
    SplitSystem masks training time slices in its own quadrature, so the pilot's
    NaNs on those slices never enter weak integrals.  No full state-size
    directional arrays are allocated here.
    """
    if variant not in {"debiased", "plugin", "raw"}:
        raise ValueError("variant must be 'debiased', 'plugin', or 'raw'")
    if variant == "debiased":
        if any(term.powers is None for term in system.terms):
            raise ValueError("Debiased scores require polynomial dictionary terms")
        if any(term.powers is None or sum(term.powers) != 1 for term in system.lhs):
            raise ValueError("Debiased scores require a linear left-hand side")

    state = system.U if variant == "raw" else system.X_state
    X = system.columns(system.terms, state, rescale=False)
    Y = system.columns(system.lhs, state, rescale=False)
    if X.shape[0] == 0 or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Score construction needs nonempty, finite weak matrices")

    if variant == "debiased":
        # columns(..., pilot=...) returns X(pilot)+D X(pilot)[U-pilot].
        # The linear LHS admits the exact increment Y(U)-Y(pilot).
        dX = system.columns(system.terms, system.U, pilot=state, rescale=False)-X
        dY = system.columns(system.lhs, system.U, rescale=False)-Y
        if not np.isfinite(dX).all() or not np.isfinite(dY).all():
            raise ValueError("Score construction needs finite directional increments")
    else:
        dX, dY = np.zeros_like(X), np.zeros_like(Y)

    K = X.shape[0]
    cross = X.T@dX
    H = (X.T@X+(cross+cross.T))/K
    b = (X.T@Y+dX.T@Y+X.T@dY)/K
    return ScoreMatrices(H=H, b=b, X=X, Y=Y, dX=dX, dY=dY, variant=variant)
