"""Synthetic PDE data, with a separate generator for each experiment.

Grid generators return SimulationData, with spatial axes first and time last.
The point-sample generator returns separate training and evaluation samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


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


@dataclass
class PointSampleData:
    """Independent training and evaluation observations on random coordinates."""

    name: str
    training_points: np.ndarray
    training_values: np.ndarray
    evaluation_points: np.ndarray
    evaluation_values: np.ndarray
    evaluation_true: np.ndarray
    noise_std: float
    true_coefficients: dict[tuple[tuple[int, ...], int], float]


BURGERS_CRITICAL_NOISE_STD = np.sqrt(0.01 / 3.0)


def burgers_true_coefficients() -> dict[tuple[tuple[int, ...], int], float]:
    """Coefficients of the nonlinear viscous Burgers equation in the paper."""
    return {
        ((2,), 1): 0.01,
        ((1,), 2): -0.5,
        ((0,), 3): -1.0,
        ((0,), 2): 2.0,
        ((0,), 0): 1.0,
    }


def generate_nonlinear_viscous_burgers(
    *, nx: int = 256, nt: int = 257, noise_std: float = 0.0,
    seed: int = 0,
) -> SimulationData:
    """Numerically solve the paper's Burgers PDE on a periodic 1D domain.

    The paper does not specify its initial/boundary data. This reproducible
    instance uses x in [-1, 1), t in [0, 1.5], periodic boundaries, and
    u(x,0)=0.5+0.7 sin(pi*x)+0.25 sin(2*pi*x+0.3). A sparse BDF solve uses
    centered spatial differences. These choices make this an adapted PDE
    experiment, not an exact reproduction of the paper's solution dataset.
    """
    if not isinstance(nx, (int, np.integer)) or nx < 16:
        raise ValueError("nx must be an integer of at least 16.")
    if not isinstance(nt, (int, np.integer)) or nt < 3:
        raise ValueError("nt must be an integer of at least 3.")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative.")
    x = np.linspace(-1.0, 1.0, nx, endpoint=False)
    time = np.linspace(0.0, 1.5, nt)
    dx = x[1] - x[0]
    indices = np.arange(nx)
    d1 = sparse.csr_matrix(
        (np.tile([-0.5, 0.5], nx),
         (np.repeat(indices, 2), np.column_stack(((indices - 1) % nx, (indices + 1) % nx)).ravel())),
        shape=(nx, nx),
    ) / dx
    d2 = sparse.csr_matrix(
        (np.tile([1.0, -2.0, 1.0], nx),
         (np.repeat(indices, 3), np.column_stack(((indices - 1) % nx, indices,
                                                  (indices + 1) % nx)).ravel())),
        shape=(nx, nx),
    ) / dx**2

    def rhs(_, values):
        return 0.01 * (d2 @ values) - 0.5 * (d1 @ (values**2)) - values**3 + 2 * values**2 + 1

    def jacobian(_, values):
        reaction_slope = -3 * values**2 + 4 * values
        return 0.01 * d2 - d1 @ sparse.diags(values) + sparse.diags(reaction_slope)

    initial = 0.5 + 0.7 * np.sin(np.pi * x) + 0.25 * np.sin(2 * np.pi * x + 0.3)
    solution = solve_ivp(
        rhs, (0.0, 1.5), initial, method="BDF", t_eval=time,
        jac=jacobian, rtol=1e-9, atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Burgers PDE solve failed: {solution.message}")
    clean = solution.y
    rng = np.random.default_rng(seed)
    observed = clean + rng.normal(0.0, noise_std, size=clean.shape)
    return SimulationData(
        name="nonlinear_viscous_burgers", spatial_grid=(x,), time=time,
        u_true=clean, u_observed=observed, noise_std=float(noise_std),
        true_coefficients=burgers_true_coefficients(),
    )


def sample_nonlinear_viscous_burgers(
    data: SimulationData, n_observations: int, *, seed: int = 0,
) -> PointSampleData:
    """Draw independent pilot and evaluation locations from a clean grid solution."""
    if data.name != "nonlinear_viscous_burgers":
        raise ValueError("data must be a nonlinear viscous Burgers simulation.")
    if not isinstance(n_observations, (int, np.integer)) or n_observations < 2:
        raise ValueError("n_observations must be an integer of at least two.")
    x, = data.spatial_grid
    rng = np.random.default_rng(seed)
    points = rng.uniform((-1.0, 0.0), (1.0, 1.5), size=(n_observations, 2))
    # Extend one periodic cell so interpolation remains continuous at x=1.
    extended_x = np.r_[x, 1.0]
    extended_u = np.vstack((data.u_true, data.u_true[:1]))
    clean = RegularGridInterpolator((extended_x, data.time), extended_u)(points)
    observed = clean + rng.normal(0.0, data.noise_std, size=n_observations)
    split = n_observations // 2
    return PointSampleData(
        name=data.name,
        training_points=points[:split], training_values=observed[:split],
        evaluation_points=points[split:], evaluation_values=observed[split:],
        evaluation_true=clean[split:], noise_std=data.noise_std,
        true_coefficients=data.true_coefficients.copy(),
    )


# The mixed derivative in div(D grad(u**2)) has coefficient 2 * D[0, 1].
_POROUS_MEDIUM_DIFFUSION = ((0.3, -0.4), (-0.4, 1.0))
_POROUS_MEDIUM_DIFFUSION_3D = (
    (0.3, -0.1, 0.05),
    (-0.1, 0.7, -0.08),
    (0.05, -0.08, 1.0),
)


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


def anisotropic_porous_medium_solution_3d(x, y, z, t) -> np.ndarray:
    """Evaluate the mass-one 3D Barenblatt weak solution for t > 0.

    For u_t = div(D grad(u**2)), the 3D similarity exponents are 3/5 and
    1/5. With q = (x,y,z) D^{-1} (x,y,z)^T, the solution is

        u = t**(-3/5) * max(C - q / (20 * t**(2/5)), 0).

    Integrating the ellipsoidal profile gives mass
    sqrt(det(D)) * (8*pi/15) * 20**(3/2) * C**(5/2), which fixes C.
    The moving front is nonsmooth; the PDE holds there in the weak sense.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("The Barenblatt solution requires finite times t > 0.")

    diffusion = np.array(_POROUS_MEDIUM_DIFFUSION_3D)
    inverse_diffusion = np.linalg.inv(diffusion)
    determinant = np.linalg.det(diffusion)
    normalization = (15.0 / (8.0 * np.pi * 20.0**1.5 * np.sqrt(determinant)))**0.4
    quadratic_form = (
        inverse_diffusion[0, 0] * x**2
        + inverse_diffusion[1, 1] * y**2
        + inverse_diffusion[2, 2] * z**2
        + 2.0 * inverse_diffusion[0, 1] * x * y
        + 2.0 * inverse_diffusion[0, 2] * x * z
        + 2.0 * inverse_diffusion[1, 2] * y * z
    )
    profile = normalization - quadratic_form / (20.0 * t**0.4)
    return np.maximum(profile, 0.0) / t**0.6


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


def generate_anisotropic_porous_medium_3d(
    *,
    nx: int = 32,
    ny: int = 32,
    nz: int = 32,
    nt: int = 16,
    x_bounds: tuple[float, float] = (-5.0, 5.0),
    y_bounds: tuple[float, float] = (-5.0, 5.0),
    z_bounds: tuple[float, float] = (-5.0, 5.0),
    time_bounds: tuple[float, float] = (0.5, 2.5),
    noise_ratio: float = 0.0,
    seed: int | None = None,
) -> SimulationData:
    """Generate a separate 3D anisotropic porous-medium experiment.

    D is fixed, symmetric positive definite, and has three nonzero mixed
    entries. The default domain contains the exact solution's support at
    every observation time. Custom bounds change only the observation
    window, as in the 2D generator. No numerical time stepping is used.
    """
    x = _uniform_grid(x_bounds, nx, "x")
    y = _uniform_grid(y_bounds, ny, "y")
    z = _uniform_grid(z_bounds, nz, "z")
    time = _uniform_grid(time_bounds, nt, "time")
    u_true = anisotropic_porous_medium_solution_3d(
        x[:, None, None, None], y[None, :, None, None],
        z[None, None, :, None], time[None, None, None, :],
    )
    u_observed, noise_std = add_gaussian_noise(u_true, noise_ratio, seed)

    diffusion = _POROUS_MEDIUM_DIFFUSION_3D
    true_coefficients = {
        ((2, 0, 0), 2): diffusion[0][0],
        ((1, 1, 0), 2): 2.0 * diffusion[0][1],
        ((1, 0, 1), 2): 2.0 * diffusion[0][2],
        ((0, 2, 0), 2): diffusion[1][1],
        ((0, 1, 1), 2): 2.0 * diffusion[1][2],
        ((0, 0, 2), 2): diffusion[2][2],
    }
    return SimulationData(
        name="anisotropic_porous_medium_3d",
        spatial_grid=(x, y, z),
        time=time,
        u_true=u_true,
        u_observed=u_observed,
        noise_std=noise_std,
        true_coefficients=true_coefficients,
    )


def generate_anisotropic_porous_medium_3d_samples(
    *,
    n_observations: int = 524288,
    bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.5, 2.5)),
    noise_reference_shape: tuple[int, int, int, int] = (32, 32, 32, 16),
    noise_ratio: float = 0.0,
    seed: int = 0,
) -> PointSampleData:
    """Sample the existing 3D PDE solution at iid uniform space-time points.

    The total number of observations is split equally between an independent
    pilot-function sample and an evaluation sample for Monte Carlo integration.
    Noise uses the exact solution's RMS on a fixed reference grid. Its scale
    therefore does not depend on either random sample, preserving the split.
    """
    if not isinstance(n_observations, (int, np.integer)) or n_observations < 2:
        raise ValueError("n_observations must be an integer of at least two.")
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.shape != (4, 2) or not np.all(np.isfinite(bounds))
        or np.any(bounds[:, 1] <= bounds[:, 0]) or bounds[3, 0] <= 0
    ):
        raise ValueError("bounds must specify three spatial intervals and positive times.")
    if not np.isfinite(noise_ratio) or noise_ratio < 0:
        raise ValueError("noise_ratio must be finite and nonnegative.")
    if len(noise_reference_shape) != 4 or any(
        not isinstance(size, (int, np.integer)) or size < 2
        for size in noise_reference_shape
    ):
        raise ValueError("noise_reference_shape must contain four grid sizes of at least two.")

    rng = np.random.default_rng(seed)
    points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_observations, 4))
    clean = anisotropic_porous_medium_solution_3d(
        points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    )
    if noise_ratio == 0:
        noise_std = 0.0
    else:
        reference_axes = [
            np.linspace(lower, upper, size)
            for (lower, upper), size in zip(bounds, noise_reference_shape)
        ]
        reference_solution = anisotropic_porous_medium_solution_3d(
            reference_axes[0][:, None, None, None],
            reference_axes[1][None, :, None, None],
            reference_axes[2][None, None, :, None],
            reference_axes[3][None, None, None, :],
        )
        noise_std = float(noise_ratio * np.sqrt(np.mean(reference_solution**2)))
    observed = clean + rng.normal(0.0, noise_std, size=n_observations)
    n_training = n_observations // 2
    diffusion = _POROUS_MEDIUM_DIFFUSION_3D
    true_coefficients = {
        ((2, 0, 0), 2): diffusion[0][0],
        ((1, 1, 0), 2): 2.0 * diffusion[0][1],
        ((1, 0, 1), 2): 2.0 * diffusion[0][2],
        ((0, 2, 0), 2): diffusion[1][1],
        ((0, 1, 1), 2): 2.0 * diffusion[1][2],
        ((0, 0, 2), 2): diffusion[2][2],
    }
    return PointSampleData(
        name="anisotropic_porous_medium_3d",
        training_points=points[:n_training],
        training_values=observed[:n_training],
        evaluation_points=points[n_training:],
        evaluation_values=observed[n_training:],
        evaluation_true=clean[n_training:],
        noise_std=noise_std,
        true_coefficients=true_coefficients,
    )
