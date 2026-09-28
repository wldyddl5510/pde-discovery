"""Checks for Monte Carlo weak equations and the Section 5 correction."""

import unittest
from types import SimpleNamespace

import numpy as np
from numpy.polynomial import Polynomial

from methods import (
    _bump_derivative_factors,
    _bump_derivatives_at_points,
    _mstls,
    build_debiased_wsindy_system,
    build_sampled_wsindy_system,
    debiased_wsindy,
    polynomial_library_terms,
    sampled_wsindy,
)
from experiments import _debiased_test_geometry
from simulation_generation import (
    generate_anisotropic_porous_medium_3d,
    generate_anisotropic_porous_medium_3d_samples,
)
from utils import MovingAverageEstimator


class DebiasedWSINDyTests(unittest.TestCase):
    def setUp(self):
        # Independent noisy values may be observed at the same test coordinates.
        self.points = np.array([
            [-0.2, -0.3], [0.4, -0.1], [0.1, 0.3], [0.5, 0.2]
        ])
        self.training_values = np.array([2.0, 3.0, 4.0, 5.0])
        self.evaluation_values = np.array([1.0, 4.0, 3.0, 6.0])
        self.centers = np.array([[0.0, 0.0], [0.2, 0.1]])
        self.bounds = ((-2.0, 2.0), (-2.0, 2.0))

    def settings(self):
        return {
            "domain_bounds": self.bounds,
            "test_centers": self.centers,
            "test_half_widths": (1.0, 1.0),
            "test_degrees": (3, 3),
            "max_derivative_order": 1,
            "max_polynomial_degree": 2,
        }

    def estimator(self):
        return MovingAverageEstimator(bandwidths=(0.01, 0.01))

    def data(self):
        return self.points, self.training_values, self.points, self.evaluation_values

    def test_monte_carlo_system_matches_direct_first_order_formula(self):
        x, y = build_debiased_wsindy_system(*self.data(), self.estimator(), **self.settings())
        expected_x = []
        expected_y = []
        v = self.training_values
        u = self.evaluation_values
        weight = 16.0 / len(u)
        for center in self.centers:
            rx = self.points[:, 0] - center[0]
            rt = self.points[:, 1] - center[1]
            bx = (1 - rx**2)**3
            bt = (1 - rt**2)**3
            dx = -6 * rx * (1 - rx**2)**2
            dt = -6 * rt * (1 - rt**2)**2
            expected_y.append(-weight * np.sum(bx * dt * u))
            q2 = 2 * v * u - v**2
            expected_x.append([
                -weight * np.sum(dx * bt * u),
                -weight * np.sum(dx * bt * q2),
            ])
        np.testing.assert_allclose(y, expected_y, rtol=0, atol=1e-12)
        np.testing.assert_allclose(x, expected_x, rtol=0, atol=1e-12)

    def test_sampled_wsindy_uses_raw_powers_and_the_same_weak_target(self):
        corrected_x, corrected_y = build_debiased_wsindy_system(
            *self.data(), self.estimator(), **self.settings()
        )
        x, y = build_sampled_wsindy_system(
            self.points, self.evaluation_values, **self.settings()
        )
        np.testing.assert_allclose(y, corrected_y, rtol=0, atol=1e-12)
        np.testing.assert_allclose(x[:, 0], corrected_x[:, 0], rtol=0, atol=1e-12)
        self.assertFalse(np.allclose(x[:, 1], corrected_x[:, 1]))
        expected_square = []
        weight = 16.0 / len(self.points)
        for center in self.centers:
            rx = self.points[:, 0] - center[0]
            rt = self.points[:, 1] - center[1]
            dx = -6 * rx * (1 - rx**2)**2
            bt = (1 - rt**2)**3
            expected_square.append(-weight * np.sum(dx * bt * self.evaluation_values**2))
        np.testing.assert_allclose(x[:, 1], expected_square, rtol=0, atol=1e-12)

    def test_sampled_wsindy_mstls_refits_its_raw_system(self):
        x, y = build_sampled_wsindy_system(
            self.points, self.evaluation_values, **self.settings()
        )
        expected = _mstls(x, y, thresholds=(0.01,), max_iter=100)
        actual = sampled_wsindy(
            self.points, self.evaluation_values, regression="mstls",
            thresholds=(0.01,), max_iter=100, **self.settings()
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_lasso_fit_satisfies_the_draft_objective_kkt_conditions(self):
        penalty = 0.1
        beta = debiased_wsindy(
            *self.data(), self.estimator(), lambda_=penalty, tol=1e-11, **self.settings()
        )
        x, y = build_debiased_wsindy_system(*self.data(), self.estimator(), **self.settings())
        gradient = x.T @ (x @ beta - y)
        active = beta != 0
        np.testing.assert_allclose(
            gradient[active], -penalty * np.sign(beta[active]), rtol=0, atol=1e-9
        )
        self.assertTrue(np.all(np.abs(gradient[~active]) <= penalty + 1e-9))

    def test_zero_penalty_is_ordinary_least_squares(self):
        x, y = build_debiased_wsindy_system(
            *self.data(), self.estimator(), **self.settings()
        )
        expected = np.linalg.lstsq(x, y, rcond=None)[0]
        actual = debiased_wsindy(
            *self.data(), self.estimator(), lambda_=0.0, **self.settings()
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_mstls_uses_the_corrected_weak_system(self):
        x, y = build_debiased_wsindy_system(
            *self.data(), self.estimator(), **self.settings()
        )
        expected = _mstls(x, y, thresholds=(0.01,), max_iter=100)
        actual = debiased_wsindy(
            *self.data(), self.estimator(), regression="mstls",
            thresholds=(0.01,), max_iter=100, **self.settings()
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_mstls_rejects_lasso_penalty(self):
        with self.assertRaisesRegex(ValueError, "rho_1 must be 0"):
            debiased_wsindy(
                *self.data(), self.estimator(), regression="mstls",
                lambda_=0.1, **self.settings()
            )

    def test_3d_mixed_derivative_and_time_sign(self):
        class ConstantPilot:
            def fit(self, points, observations):
                return self

            def predict(self, points):
                return np.full(len(points), 0.7)

        evaluation = np.array([
            [0.2, 0.3, -0.1, 1.1], [-0.4, 0.2, 0.3, 0.8],
            [0.1, -0.5, 0.2, 1.3],
        ])
        observations = np.array([0.4, 0.8, 0.6])
        x, y = build_debiased_wsindy_system(
            evaluation, observations, evaluation, observations, ConstantPilot(),
            domain_bounds=((-2, 2),) * 3 + ((0, 2),),
            test_centers=np.array([[0.0, 0.0, 0.0, 1.0]]),
            test_half_widths=(1.0, 1.0, 1.0, 0.8),
            test_degrees=(4, 4, 4, 3),
            max_derivative_order=2, max_polynomial_degree=2,
        )
        rx, ry, rz = evaluation[:, :3].T
        rt = (evaluation[:, 3] - 1.0) / 0.8
        bx, by, bz = ((1 - r**2)**4 for r in (rx, ry, rz))
        bt = (1 - rt**2)**3
        dx = -8 * rx * (1 - rx**2)**3
        dy = -8 * ry * (1 - ry**2)**3
        dt = -6 * rt * (1 - rt**2)**2 / 0.8
        weight = 128 / len(evaluation)
        expected_y = -weight * np.sum(bx * by * bz * dt * observations)
        corrected_square = 2 * 0.7 * observations - 0.7**2
        expected_mixed = weight * np.sum(dx * dy * bz * bt * corrected_square)
        column = polynomial_library_terms(3, 2, 2).index(((1, 1, 0), 2))
        self.assertAlmostEqual(y[0], expected_y)
        self.assertAlmostEqual(x[0, column], expected_mixed)

    def test_fifth_order_bump_derivatives_match_polynomial_reference(self):
        degree = 16
        half_width = 2.5
        positions = np.array([-3.0, -1.4, 0.2, 1.1, 3.0])
        actual = _bump_derivatives_at_points(
            positions, 0.0, half_width, degree,
            _bump_derivative_factors(degree, 5),
        )
        polynomial = Polynomial([1.0, 0.0, -1.0])**degree
        reference = np.zeros_like(actual)
        inside = np.abs(positions) < half_width
        for order in range(6):
            reference[inside, order] = (
                polynomial.deriv(order)(positions[inside] / half_width)
                / half_width**order
            )
        np.testing.assert_allclose(actual, reference, rtol=1e-8, atol=1e-9)

    def test_random_sample_generator_preserves_total_n_and_truth(self):
        first = generate_anisotropic_porous_medium_3d_samples(
            n_observations=200, noise_reference_shape=(8, 8, 8, 8),
            noise_ratio=1, seed=4,
        )
        repeated = generate_anisotropic_porous_medium_3d_samples(
            n_observations=200, noise_reference_shape=(8, 8, 8, 8),
            noise_ratio=1, seed=4,
        )
        self.assertEqual(first.training_points.shape, (100, 4))
        self.assertEqual(first.evaluation_points.shape, (100, 4))
        np.testing.assert_array_equal(first.evaluation_values, repeated.evaluation_values)
        self.assertEqual(first.true_coefficients[((1, 1, 0), 2)], -0.2)
        grid = generate_anisotropic_porous_medium_3d(
            nx=8, ny=8, nz=8, nt=8, noise_ratio=1, seed=4,
        )
        self.assertAlmostEqual(first.noise_std, grid.noise_std)
        other_sample_size = generate_anisotropic_porous_medium_3d_samples(
            n_observations=202, noise_reference_shape=(8, 8, 8, 8),
            noise_ratio=1, seed=4,
        )
        self.assertEqual(first.noise_std, other_sample_size.noise_std)

    def test_default_3d_test_geometry_matches_the_grid_experiment(self):
        settings = SimpleNamespace(
            nx=32, ny=32, nz=32, nt=16,
            half_widths=None, strides=None, test_degrees=None,
        )
        _, centers, widths, degrees, cells, strides = _debiased_test_geometry(settings)
        self.assertEqual(centers.shape, (4096, 4))
        self.assertEqual(cells, (8, 8, 8, 4))
        self.assertEqual(strides, (2, 2, 2, 1))
        self.assertEqual(degrees, (16, 16, 16, 28))
        np.testing.assert_allclose(widths, (80 / 31, 80 / 31, 80 / 31, 8 / 15))


if __name__ == "__main__":
    unittest.main()
