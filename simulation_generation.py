"""Synthetic PDE data, with a separate generator for each experiment.

Each generator returns SimulationData. Spatial axes come first and time comes
last, so a two-dimensional experiment has arrays of shape (nx, ny, nt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationData:
    """Clean data, noisy observations, and the known PDE used to generate them.

    A true_coefficients key is (spatial_derivative, polynomial_power).
    For example, ((1, 1), 2) denotes d_x d_y (u**2). Unlisted coefficients
    are zero. These coefficients describe the right-hand side of u_t.
    """

    name: str
    spatial_grid: tuple[np.ndarray, ...]
    time: np.ndarray
    u_true: np.ndarray
    u_observed: np.ndarray
    noise_std: float
    true_coefficients: dict[tuple[tuple[int, ...], int], float]


# The mixed derivative in div(D grad(u**2)) has coefficient 2 * D[0, 1].
_POROUS_MEDIUM_DIFFUSION = ((0.3, -0.4), (-0.4, 1.0))


def anisotropic_porous_medium_solution(x, y, t) -> np.ndarray:
    """Evaluate the mass-one Barenblatt solution at broadcastable coordinates.

    This is the exact weak solution used in Messenger and Bortz (2021),
    Appendix B.5: https://pmc.ncbi.nlm.nih.gov/articles/PMC8570254/

        u_t = 0.3 * d_xx(u**2) - 0.8 * d_xy(u**2) + d_yy(u**2)
        u(x, y, t) = t**(-1/2) * max(C - (x, y) D^{-1} (x, y)^T
                                               / (16 * sqrt(t)), 0)

    The solution is defined on R^2 for t > 0. Its gradient jumps at the
    moving support boundary, where the PDE holds in the weak sense.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("The Barenblatt solution requires finite times t > 0.")

    diffusion = np.array(_POROUS_MEDIUM_DIFFUSION)
    inverse_diffusion = np.linalg.inv(diffusion)
    determinant = np.linalg.det(diffusion)
    normalization = 1.0 / np.sqrt(8.0 * np.pi * np.sqrt(determinant))

    quadratic_form = (
        inverse_diffusion[0, 0] * x**2
        + 2.0 * inverse_diffusion[0, 1] * x * y
        + inverse_diffusion[1, 1] * y**2
    )
    sqrt_time = np.sqrt(t)
    profile = normalization - quadratic_form / (16.0 * sqrt_time)
    return np.maximum(profile, 0.0) / sqrt_time


def add_gaussian_noise(
    u_true: np.ndarray, noise_ratio: float = 0.0, seed: int | None = None
) -> tuple[np.ndarray, float]:
    """Add iid Gaussian noise with std = noise_ratio * RMS(u_true).

    The clean input is not modified. The local random generator leaves the
    global NumPy random state unchanged. Negative observations are retained.
    """
    if not np.isfinite(noise_ratio) or noise_ratio < 0:
        raise ValueError("noise_ratio must be finite and nonnegative.")

    signal_rms = np.sqrt(np.mean(u_true**2))
    noise_std = float(noise_ratio * signal_rms)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=u_true.shape)
    return u_true + noise, noise_std


def _uniform_grid(bounds, size: int, name: str) -> np.ndarray:
    """Create an endpoint-inclusive observation grid."""
    if not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError(f"{name} grid size must be an integer of at least 2.")
    if len(bounds) != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError(f"{name} bounds must contain two finite endpoints.")
    lower, upper = bounds
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be smaller than its upper bound.")
    return np.linspace(lower, upper, size)


def generate_anisotropic_porous_medium(
    *,
    nx: int = 64,
    ny: int = 64,
    nt: int = 32,
    x_bounds: tuple[float, float] = (-5.0, 5.0),
    y_bounds: tuple[float, float] = (-5.0, 5.0),
    time_bounds: tuple[float, float] = (0.5, 2.5),
    noise_ratio: float = 0.0,
    seed: int | None = None,
) -> SimulationData:
    """Generate the 2D anisotropic porous medium benchmark.

    Defaults use the paper's physical domain and times with a smaller grid.
    Set nx=200, ny=200, nt=128 for the paper's observation grid.

    u_true is evaluated from the exact weak solution; no time stepping is
    needed. The default domain contains its support throughout the time
    interval, so u is zero on the spatial boundary. Custom bounds only
    change the observation window: they do not rescale coordinates, alter
    PDE coefficients, or impose new boundary conditions.
    """
    x = _uniform_grid(x_bounds, nx, "x")
    y = _uniform_grid(y_bounds, ny, "y")
    time = _uniform_grid(time_bounds, nt, "time")

    # Broadcasting constructs (nx, ny, nt) data without dense coordinate grids.
    u_true = anisotropic_porous_medium_solution(
        x[:, None, None], y[None, :, None], time[None, None, :]
    )
    u_observed, noise_std = add_gaussian_noise(u_true, noise_ratio, seed)

    diffusion = _POROUS_MEDIUM_DIFFUSION
    true_coefficients = {
        ((2, 0), 2): diffusion[0][0],
        ((1, 1), 2): 2.0 * diffusion[0][1],
        ((0, 2), 2): diffusion[1][1],
    }
    return SimulationData(
        name="anisotropic_porous_medium",
        spatial_grid=(x, y),
        time=time,
        u_true=u_true,
        u_observed=u_observed,
        noise_std=noise_std,
        true_coefficients=true_coefficients,
    )
