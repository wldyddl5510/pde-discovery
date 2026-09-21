"""Check the library, finite differences, and OLS/LASSO SINDy regression."""

import unittest

import numpy as np

from methods import _lasso, build_sindy_system, polynomial_library_terms, sindy
from simulation_generation import generate_anisotropic_porous_medium


class SindyTests(unittest.TestCase):
    def test_all_mixed_derivatives_and_coefficient_order(self):
        for dim, expected_count in ((1, 25), (2, 100), (3, 275)):
            terms = polynomial_library_terms(dim)
            self.assertEqual(len(terms), expected_count)
            self.assertEqual(len(set(terms)), expected_count)
            self.assertTrue(all(1 <= sum(alpha) <= 5 for alpha, power in terms))

        terms = polynomial_library_terms(2)
        self.assertEqual(terms[:5], [((1, 0), j) for j in range(1, 6)])
        self.assertEqual(terms[5:10], [((0, 1), j) for j in range(1, 6)])
        self.assertEqual(terms[15:20], [((1, 1), j) for j in range(1, 6)])
        self.assertEqual(terms[-1], ((0, 5), 5))

    def test_derivatives_of_powers_and_space_time_row_alignment(self):
        x = np.arange(13) * 0.5 - 3.0
        y = np.arange(15) * 0.25 - 1.75
        time = np.arange(7) * 0.1
        u = x[:, None, None] + 2.0 * y[None, :, None] + time[None, None, :]
        original = u.copy()
        X, target = build_sindy_system(u, (x, y), time)
        terms = polynomial_library_terms(2)

        self.assertEqual(X.shape, (7 * 9 * 5, 100))
        np.testing.assert_array_equal(u, original)
        np.testing.assert_allclose(target, 1.0, atol=1e-12)
        interior_u = u[3:-3, 3:-3, 1:-1].ravel()
        np.testing.assert_allclose(X[:, terms.index(((1, 0), 2))], 2 * interior_u, atol=1e-12)
        np.testing.assert_allclose(X[:, terms.index(((0, 1), 2))], 4 * interior_u, atol=1e-12)

        # d_x^a d_y^b (x + 2y + t)^5 = 5! * 2^b when a + b = 5.
        for y_order in range(6):
            derivative = (5 - y_order, y_order)
            column = terms.index((derivative, 5))
            np.testing.assert_allclose(X[:, column], 120 * 2**y_order, rtol=0, atol=1e-7)

    def test_dense_nonlinear_coefficients_in_one_dimension(self):
        # Smooth solution of u_t = 0.4*u_x - 0.5*d_x(u^2); both coefficients
        # are nonzero and the two library columns are independent.
        x = np.linspace(-1.0, 1.0, 41)
        time = np.linspace(0.0, 0.5, 81)
        slope, intercept = 0.3, 0.8
        advection, nonlinear = 0.4, -0.5
        numerator = slope * x[:, None] + intercept + slope * advection * time[None, :]
        denominator = 1.0 - 2.0 * slope * nonlinear * time[None, :]
        u = numerator / denominator

        beta = sindy(u, (x,), time, max_derivative_order=1, max_polynomial_degree=2)
        np.testing.assert_allclose(beta, [advection, nonlinear], rtol=1e-4, atol=1e-5)

    def test_anisotropic_diffusion_coefficients_converge_on_smooth_data(self):
        # Several Fourier modes identify the two pure and one mixed second
        # derivative separately. First-derivative columns remain in the fit.
        modes = [
            (1, 0, 1.0, 0.1),
            (0, 1, 0.7, -0.2),
            (1, 1, 0.8, 0.4),
            (1, -1, 0.6, -0.7),
            (2, 1, 0.4, 0.9),
        ]
        errors = []
        for size, nt in ((33, 21), (65, 41)):
            x = np.linspace(0.0, 2.0 * np.pi, size)
            y = np.linspace(0.0, 2.0 * np.pi, size + 2)
            time = np.linspace(0.0, 0.5, nt)
            u = np.zeros((len(x), len(y), nt))
            for kx, ky, amplitude, phase in modes:
                decay_rate = 0.3 * kx**2 - 0.8 * kx * ky + ky**2
                angle = kx * x[:, None, None] + ky * y[None, :, None] + phase
                u += amplitude * np.cos(angle) * np.exp(-decay_rate * time[None, None, :])

            beta = sindy(u, (x, y), time, max_derivative_order=2, max_polynomial_degree=1)
            expected = np.array([0.0, 0.0, 0.3, -0.8, 1.0])
            errors.append(np.linalg.norm(beta - expected))

        self.assertLess(errors[1], 0.004)
        self.assertLess(errors[1], 0.3 * errors[0])

    def test_porous_medium_pipeline_solves_the_full_least_squares_problem(self):
        # Check the intended objective, without assuming pointwise derivatives
        # accurately recover coefficients at this solution's nonsmooth front.
        for noise_ratio in (0.0, 0.05):
            with self.subTest(noise_ratio=noise_ratio):
                data = generate_anisotropic_porous_medium(
                    nx=36, ny=40, nt=20, noise_ratio=noise_ratio, seed=0
                )
                beta = sindy(data.u_observed, data.spatial_grid, data.time)
                zero_penalty_beta = sindy(
                    data.u_observed, data.spatial_grid, data.time, rho_1=0.0
                )
                np.testing.assert_array_equal(beta, zero_penalty_beta)
                X, y = build_sindy_system(data.u_observed, data.spatial_grid, data.time)
                self.assertEqual(beta.shape, (100,))
                self.assertTrue(np.all(np.isfinite(beta)))
                residual = X @ beta - y
                scaled_X = X / np.linalg.norm(X, axis=0)
                gradient = scaled_X.T @ residual
                self.assertLess(np.linalg.norm(gradient) / np.linalg.norm(y), 1e-10)

    def test_lasso_penalizes_original_coefficients_with_the_correct_factor(self):
        # X.T @ X / N = diag(1, 100, 0.01), so the closed-form shrinkage is
        # rho_1 / (2 * diagonal), not a common amount after column scaling.
        X = np.sqrt(3.0) * np.diag([1.0, 10.0, 0.1])
        y = X @ np.array([2.0, -3.0, 1.0])
        beta = _lasso(X, y, rho_1=1.0, max_iter=100, tol=1e-12)
        np.testing.assert_allclose(beta, [1.5, -2.995, 0.0], atol=1e-12)

        # Repeating observations must not change the mean-loss objective.
        repeated = _lasso(
            np.tile(X, (4, 1)), np.tile(y, 4), rho_1=1.0, max_iter=100, tol=1e-12
        )
        np.testing.assert_allclose(repeated, beta, atol=1e-12)

    def test_large_lasso_penalty_returns_zero(self):
        X = np.array([[1.0, 2.0], [-1.0, 3.0], [2.0, -1.0]])
        y = np.array([2.0, -1.0, 0.5])
        rho_zero = 2.0 * np.max(np.abs(X.T @ y)) / len(y)
        beta = _lasso(X, y, rho_1=1.01 * rho_zero, max_iter=100, tol=1e-12)
        np.testing.assert_array_equal(beta, np.zeros(2))

    def test_lasso_accepts_underdetermined_and_zero_column_systems(self):
        X = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        y = np.array([2.0, -1.0])
        beta = _lasso(X, y, rho_1=0.2, max_iter=100, tol=1e-12)
        np.testing.assert_allclose(X @ beta, [1.8, -0.8], atol=1e-12)
        self.assertEqual(beta[-1], 0.0)

        # The public positive-penalty path must bypass the OLS rank checks.
        x = np.linspace(0.0, 1.0, 9)
        time = np.linspace(0.0, 1.0, 5)
        beta = sindy(np.zeros((9, 9, 5)), (x, x), time, rho_1=0.1)
        np.testing.assert_array_equal(beta, np.zeros(100))

    def test_lasso_kkt_conditions_on_porous_medium_data(self):
        data = generate_anisotropic_porous_medium(
            nx=36, ny=40, nt=20, noise_ratio=0.05, seed=0
        )
        rho_1 = 1e-4
        beta = sindy(data.u_observed, data.spatial_grid, data.time, rho_1=rho_1)
        X, y = build_sindy_system(data.u_observed, data.spatial_grid, data.time)
        # Recompute the gradient from the residual, independently of the
        # Gram-matrix calculation used inside coordinate descent.
        gradient = 2.0 * X.T @ (X @ beta - y) / len(y)
        nonzero = beta != 0
        self.assertTrue(np.any(nonzero))
        self.assertTrue(np.any(~nonzero))
        np.testing.assert_allclose(
            gradient[nonzero], -rho_1 * np.sign(beta[nonzero]), atol=1.1e-8, rtol=0
        )
        self.assertLessEqual(np.max(np.abs(gradient[~nonzero])), rho_1 + 1.1e-8)

    def test_invalid_penalty_and_lasso_controls(self):
        x = np.linspace(0.0, 1.0, 9)
        time = np.linspace(0.0, 1.0, 5)
        u = np.ones((9, 5))
        for rho_1 in (-0.1, np.nan, np.inf):
            with self.subTest(rho_1=rho_1):
                with self.assertRaises(ValueError):
                    sindy(u, (x,), time, rho_1=rho_1)
        for controls in ({"max_iter": 0}, {"tol": 0}, {"tol": np.nan}):
            with self.subTest(controls=controls):
                with self.assertRaises(ValueError):
                    sindy(u, (x,), time, rho_1=0.1, **controls)

    def test_lasso_nonconvergence_is_reported(self):
        X = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y = np.array([2.0, 1.0, 1.0])
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            _lasso(X, y, rho_1=0.1, max_iter=1, tol=1e-12)

    def test_unidentifiable_data_are_reported(self):
        x = np.linspace(0.0, 1.0, 9)
        time = np.linspace(0.0, 1.0, 5)
        with self.assertRaises(np.linalg.LinAlgError):
            sindy(np.zeros((9, 5)), (x,), time, max_derivative_order=1, max_polynomial_degree=1)
        with self.assertRaises(np.linalg.LinAlgError):
            sindy(np.ones((9, 9, 5)), (x, x), time)

    def test_nonuniform_or_mismatched_coordinates_are_rejected(self):
        x = np.linspace(0.0, 1.0, 9)
        time = np.linspace(0.0, 1.0, 5)
        u = np.ones((9, 5))
        irregular_x = x.copy()
        irregular_x[4] += 0.01
        for space, times in (((irregular_x,), time), ((x[:-1],), time), ((x,), time[:-1])):
            with self.subTest(space=space, times=times):
                with self.assertRaises(ValueError):
                    sindy(u, space, times)


if __name__ == "__main__":
    unittest.main()
