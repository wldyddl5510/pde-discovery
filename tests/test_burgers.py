"""Check the adapted Burgers data and the paper's weak-form settings."""

import unittest

import numpy as np

from methods import build_wsindy_system, polynomial_library_terms
from simulation_generation import (
    BURGERS_CRITICAL_NOISE_STD,
    generate_nonlinear_viscous_burgers,
    sample_nonlinear_viscous_burgers,
)
from utils import estimate_grid_noise_std, filter_grid_moving_average, paper_filter_width


class BurgersTests(unittest.TestCase):
    def test_data_are_reproducible_and_have_the_stated_pde(self):
        first = generate_nonlinear_viscous_burgers(nx=128, nt=129, noise_std=0.1, seed=4)
        second = generate_nonlinear_viscous_burgers(nx=128, nt=129, noise_std=0.1, seed=4)
        x, = first.spatial_grid
        expected_initial = 0.5 + 0.7 * np.sin(np.pi * x) + 0.25 * np.sin(2 * np.pi * x + 0.3)
        np.testing.assert_allclose(first.u_true[:, 0], expected_initial)
        np.testing.assert_array_equal(first.u_observed, second.u_observed)
        self.assertEqual(first.true_coefficients[((0,), 0)], 1.0)
        self.assertEqual(first.true_coefficients[((2,), 1)], 0.01)
        samples = sample_nonlinear_viscous_burgers(first, 2000, seed=5)
        self.assertEqual(samples.training_points.shape, (1000, 2))
        self.assertEqual(samples.evaluation_points.shape, (1000, 2))
        self.assertFalse(np.array_equal(samples.training_points, samples.evaluation_points))

    def test_clean_weak_pde_and_sixth_order_paper_bump(self):
        data = generate_nonlinear_viscous_burgers(nx=256, nt=257)
        library = dict(
            min_derivative_order=0, max_derivative_order=6,
            min_polynomial_degree=0, max_polynomial_degree=6,
        )
        X, y = build_wsindy_system(
            data.u_true, data.spatial_grid, data.time,
            half_widths=(32, 32), strides=(8, 8),
            test_function="paper", **library,
        )
        terms = polynomial_library_terms(1, **library)
        beta = np.array([data.true_coefficients.get(term, 0.0) for term in terms])
        self.assertEqual(X.shape, (600, 43))
        self.assertLess(np.linalg.norm(y - X @ beta) / np.linalg.norm(y), 0.002)

    def test_paper_noise_estimate_and_filter_width(self):
        rng = np.random.default_rng(2)
        observations = rng.normal(0, BURGERS_CRITICAL_NOISE_STD, size=(80, 200))
        estimate = estimate_grid_noise_std(observations)
        self.assertAlmostEqual(estimate / BURGERS_CRITICAL_NOISE_STD, 1.0, delta=0.03)
        self.assertEqual(paper_filter_width(0.0, 65 * 65), 1)
        self.assertEqual(paper_filter_width(BURGERS_CRITICAL_NOISE_STD, 65 * 65), 5)
        filtered = filter_grid_moving_average(observations, 5)
        self.assertLess(np.std(filtered), np.std(observations) / 3)


if __name__ == "__main__":
    unittest.main()
