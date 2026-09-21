"""Physical and statistical checks for the synthetic reference solution."""

import unittest

import numpy as np

from simulation_generation import (
    anisotropic_porous_medium_solution,
    generate_anisotropic_porous_medium,
)


def trapezoid_weights(grid):
    weights = np.full(len(grid), grid[1] - grid[0])
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def bump_derivatives(coordinate, radius):
    """A compact test function and its first two derivatives."""
    base = np.maximum(1.0 - (coordinate / radius) ** 2, 0.0)
    value = base**4
    first = -8.0 * coordinate / radius**2 * base**3
    second = -8.0 / radius**2 * base**3
    second += 48.0 * coordinate**2 / radius**4 * base**2
    return value, first, second


class PorousMediumTests(unittest.TestCase):
    def test_grid_axes_boundary_and_true_coefficients(self):
        data = generate_anisotropic_porous_medium(nx=45, ny=51, nt=7)
        x, y = data.spatial_grid
        self.assertEqual(data.u_true.shape, (45, 51, 7))
        np.testing.assert_array_equal(x[[0, -1]], [-5.0, 5.0])
        np.testing.assert_array_equal(y[[0, -1]], [-5.0, 5.0])
        np.testing.assert_array_equal(data.time[[0, -1]], [0.5, 2.5])
        np.testing.assert_array_equal(data.u_observed, data.u_true)
        self.assertEqual(data.noise_std, 0.0)
        self.assertTrue(np.all(data.u_true >= 0.0))
        self.assertTrue(np.any(data.u_true > 0.0))
        self.assertTrue(np.all(data.u_true[[0, -1], :, :] == 0.0))
        self.assertTrue(np.all(data.u_true[:, [0, -1], :] == 0.0))
        self.assertEqual(
            data.true_coefficients,
            {((2, 0), 2): 0.3, ((1, 1), 2): -0.8, ((0, 2), 2): 1.0},
        )
        expected = anisotropic_porous_medium_solution(x[23], y[26], data.time[2])
        self.assertEqual(data.u_true[23, 26, 2], expected)

    def test_unit_mass_at_multiple_times_under_grid_refinement(self):
        errors = []
        for size in (65, 129):
            data = generate_anisotropic_porous_medium(nx=size, ny=size, nt=3)
            x, y = data.spatial_grid
            wx = trapezoid_weights(x)
            wy = trapezoid_weights(y)
            weighted_solution = data.u_true * wx[:, None, None] * wy[None, :, None]
            mass = np.sum(weighted_solution, axis=(0, 1))
            errors.append(np.max(np.abs(mass - 1.0)))

        self.assertLess(errors[1], 1e-4)
        self.assertLess(errors[1], errors[0] / 5.0)

    def test_pointwise_pde_inside_the_positive_support(self):
        # Avoid the moving front, where only the weak PDE is appropriate.
        u = anisotropic_porous_medium_solution
        x = np.array([0.1, 0.5, -0.6, 0.3])
        y = np.array([0.2, -0.1, 0.1, -0.3])
        t = np.array([0.6, 1.0, 1.7, 2.0])
        errors = []

        for h in (0.02, 0.01):
            dt = h**2
            time_derivative = (u(x, y, t + dt) - u(x, y, t - dt)) / (2.0 * dt)
            squared_solution = u(x, y, t) ** 2
            dxx = (u(x + h, y, t)**2 - 2*squared_solution + u(x - h, y, t)**2) / h**2
            dyy = (u(x, y + h, t)**2 - 2*squared_solution + u(x, y - h, t)**2) / h**2
            dxy = (
                u(x + h, y + h, t)**2 - u(x + h, y - h, t)**2
                - u(x - h, y + h, t)**2 + u(x - h, y - h, t)**2
            ) / (4.0 * h**2)
            rhs = 0.3 * dxx - 0.8 * dxy + dyy
            errors.append(np.max(np.abs(time_derivative - rhs)))

        self.assertLess(errors[1], 6e-5)
        self.assertLess(errors[1], 0.3 * errors[0])

    def test_weak_pde_across_the_moving_support_boundary(self):
        # This integral includes the nonsmooth front, unlike the pointwise check.
        errors = []
        for size, nt in ((65, 33), (129, 65)):
            data = generate_anisotropic_porous_medium(nx=size, ny=size, nt=nt)
            x, y = data.spatial_grid
            t = data.time
            bx, dx, dxx = bump_derivatives(x[:, None, None], radius=4.0)
            by, dy, dyy = bump_derivatives(y[None, :, None], radius=4.0)
            bt, dt, _ = bump_derivatives(t[None, None, :] - 1.5, radius=1.0)

            wx = trapezoid_weights(x)
            wy = trapezoid_weights(y)
            wt = trapezoid_weights(t)
            weights = wx[:, None, None] * wy[None, :, None] * wt[None, None, :]

            lhs = -np.sum(bx * by * dt * data.u_true * weights)
            rhs_test_function = (0.3 * dxx * by - 0.8 * dx * dy + bx * dyy) * bt
            rhs = np.sum(rhs_test_function * data.u_true**2 * weights)
            errors.append(abs(lhs - rhs) / abs(rhs))

        self.assertLess(errors[1], 2e-5)
        self.assertLess(errors[1], errors[0] / 5.0)

    def test_gaussian_noise_scaling_and_reproducibility(self):
        first = generate_anisotropic_porous_medium(noise_ratio=0.1, seed=17)
        repeated = generate_anisotropic_porous_medium(noise_ratio=0.1, seed=17)
        other = generate_anisotropic_porous_medium(noise_ratio=0.1, seed=18)
        np.testing.assert_array_equal(first.u_observed, repeated.u_observed)
        np.testing.assert_array_equal(first.u_true, other.u_true)
        self.assertFalse(np.array_equal(first.u_observed, other.u_observed))
        self.assertFalse(np.shares_memory(first.u_true, first.u_observed))

        expected_std = 0.1 * np.sqrt(np.mean(first.u_true**2))
        self.assertAlmostEqual(first.noise_std, expected_std)
        standardized_noise = (first.u_observed - first.u_true) / first.noise_std
        self.assertLess(abs(standardized_noise.mean()), 5 / np.sqrt(standardized_noise.size))
        self.assertLess(abs(standardized_noise.std() - 1.0), 0.01)
        self.assertTrue(np.any(first.u_observed < 0.0))

    def test_invalid_time_grid_and_noise(self):
        for settings in (
            {"time_bounds": (0.0, 2.5)},
            {"nx": 1},
            {"x_bounds": (5.0, -5.0)},
            {"noise_ratio": -0.1},
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    generate_anisotropic_porous_medium(**settings)


if __name__ == "__main__":
    unittest.main()
