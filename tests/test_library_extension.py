"""Check constant, reaction, and sixth-order library terms across methods."""

import unittest

import numpy as np

from methods import (
    _WeakResidual,
    build_debiased_wsindy_system,
    build_sampled_wsindy_system,
    build_sindy_system,
    build_wsindy_system,
    polynomial_library_terms,
)


LIBRARY = dict(
    max_derivative_order=6,
    max_polynomial_degree=6,
    min_derivative_order=0,
    min_polynomial_degree=0,
)


class LibraryExtensionTests(unittest.TestCase):
    def test_burgers_library_omits_identically_zero_columns(self):
        terms = polynomial_library_terms(1, **LIBRARY)
        self.assertEqual(len(terms), 43)
        self.assertEqual(terms[:7], [((0,), power) for power in range(7)])
        self.assertEqual(terms[7:13], [((1,), power) for power in range(1, 7)])
        self.assertEqual(terms[-6:], [((6,), power) for power in range(1, 7)])
        self.assertEqual(sum(power == 0 for _, power in terms), 1)
        terms_3d = polynomial_library_terms(3, **LIBRARY)
        self.assertEqual(len(terms_3d), 505)
        self.assertIn(((2, 2, 2), 6), terms_3d)

    def test_strong_system_constant_reaction_and_sixth_derivative(self):
        x = np.linspace(-1.0, 1.0, 21)
        time = np.linspace(0.0, 0.2, 5)
        u = x[:, None]**6 + time[None, :]
        X, target = build_sindy_system(u, (x,), time, **LIBRARY)
        terms = polynomial_library_terms(1, **LIBRARY)
        self.assertEqual(X.shape, (15 * 3, 43))
        np.testing.assert_allclose(target, 1.0, atol=1e-12)
        np.testing.assert_array_equal(X[:, terms.index(((0,), 0))], np.ones(len(target)))
        np.testing.assert_allclose(X[:, terms.index(((0,), 1))], u[3:-3, 1:-1].ravel())
        np.testing.assert_allclose(X[:, terms.index(((6,), 1))], 720.0, atol=1e-6)

    def test_weak_system_sixth_derivative_of_exponential(self):
        x = np.linspace(-1.0, 1.0, 301)
        time = np.linspace(0.0, 0.5, 31)
        rate = 0.8
        u = np.exp(rate * x[:, None] + 0.3 * time[None, :])
        X, _ = build_wsindy_system(
            u, (x,), time, half_widths=(100, 8), strides=(40, 4),
            test_degrees=(14, 5), **LIBRARY,
        )
        terms = polynomial_library_terms(1, **LIBRARY)
        reaction = X[:, terms.index(((0,), 1))]
        derivative = X[:, terms.index(((6,), 1))]
        np.testing.assert_allclose(derivative, rate**6 * reaction, rtol=1e-7, atol=1e-9)
        self.assertTrue(np.all(X[:, terms.index(((0,), 0))] > 0))

    def test_sampled_and_debiased_constant_and_reaction_terms(self):
        class ZeroPilot:
            def fit(self, points, values):
                return self

            def predict(self, points):
                return np.zeros(len(points))

        points = np.array([[-0.2, -0.1], [0.0, 0.1], [0.2, 0.0]])
        values = np.array([0.4, 0.7, 0.9])
        settings = dict(
            domain_bounds=((-1.0, 1.0), (-1.0, 1.0)),
            test_centers=np.array([[0.0, 0.0]]),
            test_half_widths=(0.5, 0.5), test_degrees=(8, 3),
            **LIBRARY,
        )
        raw, y_raw = build_sampled_wsindy_system(points, values, **settings)
        corrected, y_corrected = build_debiased_wsindy_system(
            points, values, points, values, ZeroPilot(), **settings,
        )
        terms = polynomial_library_terms(1, **LIBRARY)
        constant = terms.index(((0,), 0))
        reaction = terms.index(((0,), 1))
        np.testing.assert_allclose(corrected[:, constant], raw[:, constant])
        np.testing.assert_allclose(corrected[:, reaction], raw[:, reaction])
        np.testing.assert_allclose(y_corrected, y_raw)
        self.assertTrue(np.all(np.isfinite(corrected)))

    def test_wendy_jacobian_and_likelihood_gradient_include_new_terms(self):
        x = np.linspace(-1.0, 1.0, 21)
        time = np.linspace(0.0, 0.4, 13)
        u = np.exp(0.4 * x[:, None] + 0.3 * time[None, :])
        settings = dict(
            max_derivative_order=2, max_polynomial_degree=2,
            min_derivative_order=0, min_polynomial_degree=0,
            half_widths=(5, 3), strides=(5, 3), test_degrees=(5, 4),
        )
        problem = _WeakResidual(u, (x,), time, **settings)
        beta = np.array([0.2, -0.1, 0.05, 0.1, -0.04, 0.03, 0.02])
        constant = polynomial_library_terms(1, 2, 2,
            min_derivative_order=0, min_polynomial_degree=0).index(((0,), 0))
        direction = np.zeros_like(beta)
        direction[constant] = 1.0
        np.testing.assert_allclose(problem.jacobian(beta + direction).toarray(),
                                   problem.jacobian(beta).toarray())
        _, gradient = problem.likelihood(beta, noise_std=0.1, alpha=0.2)
        epsilon = 1e-6
        for column in range(len(beta)):
            direction = np.eye(len(beta))[column]
            plus = problem.likelihood(beta + epsilon * direction, 0.1, 0.2)[0]
            minus = problem.likelihood(beta - epsilon * direction, 0.1, 0.2)[0]
            self.assertAlmostEqual(gradient[column], (plus - minus) / (2 * epsilon), places=5)


if __name__ == "__main__":
    unittest.main()
