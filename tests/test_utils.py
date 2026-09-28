"""Checks for pointwise estimators of the latent solution."""

import unittest

import numpy as np

from utils import MovingAverageEstimator


class MovingAverageEstimatorTests(unittest.TestCase):
    def test_box_average_uses_only_training_observations(self):
        training_points = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 0.0]])
        training_values = np.array([1.0, 3.0, 10.0])
        held_out_points = np.array([[0.1, 0.0], [0.9, 0.0]])

        estimator = MovingAverageEstimator(bandwidths=(0.25, 0.5))
        fitted = estimator.fit(training_points, training_values)
        self.assertIs(fitted, estimator)
        np.testing.assert_array_equal(estimator.predict(held_out_points), [2.0, 10.0])

        # Changing the caller's array after fit cannot leak new values into it.
        training_values[:] = -100.0
        np.testing.assert_array_equal(estimator.predict(held_out_points), [2.0, 10.0])

    def test_noise_reduction_at_independent_four_dimensional_points(self):
        rng = np.random.default_rng(11)
        training_points = rng.uniform(size=(5000, 4))
        held_out_points = rng.uniform(size=(300, 4))

        def true_solution(points):
            return points[:, 0] + 0.5 * points[:, 3]

        training_values = true_solution(training_points) + rng.normal(0, 0.5, len(training_points))
        held_out_values = true_solution(held_out_points) + rng.normal(0, 0.5, len(held_out_points))
        estimator = MovingAverageEstimator(bandwidths=(0.15, 0.15, 0.15, 0.15))
        estimates = estimator.fit(training_points, training_values).predict(held_out_points)

        estimate_error = np.mean((estimates - true_solution(held_out_points))**2)
        observation_error = np.mean((held_out_values - true_solution(held_out_points))**2)
        self.assertLess(estimate_error, observation_error)

    def test_rejects_missing_neighbors_and_invalid_inputs(self):
        with self.assertRaises(ValueError):
            MovingAverageEstimator(bandwidths=(1.0, 0.0))
        estimator = MovingAverageEstimator(bandwidths=(0.1, 0.1))
        with self.assertRaises(RuntimeError):
            estimator.predict(np.array([[0.0, 0.0]]))
        with self.assertRaises(ValueError):
            estimator.fit(np.array([[0.0, 0.0]]), np.array([np.nan]))

        estimator.fit(np.array([[0.0, 0.0]]), np.array([1.0]))
        with self.assertRaises(ValueError):
            estimator.predict(np.array([[1.0, 1.0]]))

