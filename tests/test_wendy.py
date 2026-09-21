"""Check the observation Jacobian, full covariance, IRLS, and weak likelihood."""

import unittest

import numpy as np
from methods import _WeakResidual, build_wsindy_system, wendy, wendy_mle, wsindy


def advection_data():
    x = np.linspace(-1.0, 1.0, 65)
    time = np.linspace(0.0, 0.5, 41)
    phase = x[:, None] + 0.4 * time[None, :]
    u = np.sin(phase) + 0.2 * np.cos(2.0 * phase)
    u += np.random.default_rng(2).normal(scale=0.01, size=u.shape)
    settings = dict(max_derivative_order=2, max_polynomial_degree=1, strides=(8, 5))
    return u, (x,), time, settings


class WendyTests(unittest.TestCase):
    def test_full_library_observation_jacobian_in_two_dimensions(self):
        x = np.linspace(-1.0, 1.0, 17)
        y = np.linspace(-0.5, 0.5, 19)
        time = np.linspace(0.0, 0.8, 13)
        u = 0.8 + 0.1 * x[:, None, None] + 0.2 * y[None, :, None] + time[None, None, :]**2
        original = u.copy()
        settings = dict(strides=(3, 4, 3))
        problem = _WeakResidual(u, (x, y), time, **settings)
        rng = np.random.default_rng(3)
        beta = rng.normal(scale=0.001, size=100)
        direction = rng.normal(size=u.shape)
        step = 1e-6
        X_plus, y_plus = build_wsindy_system(u + step * direction, (x, y), time, **settings)
        X_minus, y_minus = build_wsindy_system(u - step * direction, (x, y), time, **settings)
        finite_difference = ((y_plus - X_plus @ beta) - (y_minus - X_minus @ beta)) / (2 * step)
        derivative = problem.jacobian(beta) @ direction.ravel()
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7, atol=1e-7)
        np.testing.assert_array_equal(u, original)

    def test_covariance_and_likelihood_against_dense_reference(self):
        u, space, time, settings = advection_data()
        problem = _WeakResidual(u, space, time, **settings)
        beta = np.array([0.3, 0.02])
        A = problem.jacobian(beta).toarray()
        alpha, sigma = 0.02, 0.01
        C = (1.0 - alpha) * A @ A.T + alpha * np.eye(len(problem.y))
        np.testing.assert_allclose(problem.covariance(beta, alpha), C, atol=1e-14)
        off_diagonal = C - np.diag(np.diag(C))
        self.assertGreater(np.linalg.norm(off_diagonal), 1e-4)
        residual = problem.y - problem.X @ beta
        expected = (np.linalg.slogdet(C)[1] + residual @ np.linalg.solve(C, residual) / sigma**2)
        expected /= 2.0 * len(residual)
        value, _ = problem.likelihood(beta, sigma, alpha)
        self.assertAlmostEqual(value, expected, places=10)
        X_white, y_white = problem.whiten(beta, alpha)
        np.testing.assert_allclose(
            np.sum((y_white - X_white @ beta)**2), residual @ np.linalg.solve(C, residual),
            rtol=1e-12,
        )

    def test_likelihood_gradient_including_nonlinear_covariance(self):
        x = np.linspace(-1.0, 1.0, 25)
        time = np.linspace(0.0, 0.5, 21)
        phase = x[:, None] + 0.4 * time[None, :]
        u = np.sin(phase) + 0.2 * np.cos(2.0 * phase)
        problem = _WeakResidual(
            u, (x,), time, max_derivative_order=2, max_polynomial_degree=2, strides=(5, 4)
        )
        beta = np.array([0.3, -0.1, 0.02, 0.01])
        step = 1e-6
        for alpha in (0.0, 0.03, 1.0):
            _, gradient = problem.likelihood(beta, 0.02, alpha)
            finite_difference = np.zeros_like(beta)
            for index in range(len(beta)):
                direction = np.eye(len(beta))[index] * step
                plus = problem.likelihood(beta + direction, 0.02, alpha)[0]
                minus = problem.likelihood(beta - direction, 0.02, alpha)[0]
                finite_difference[index] = (plus - minus) / (2.0 * step)
            np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-6)

    def test_identity_covariance_reduces_to_existing_solvers(self):
        u, space, time, settings = advection_data()
        for regression, controls in (
            ("lasso", {"rho_1": 0.0}),
            ("lasso", {"rho_1": 1e-5}),
            ("mstls", {"thresholds": (0.01, 0.05, 0.1)}),
        ):
            expected = wsindy(u, space, time, regression=regression, **controls, **settings)
            irls = wendy(u, space, time, alpha=1.0, regression=regression, **controls, **settings)
            np.testing.assert_allclose(irls, expected, atol=1e-12)
            mle_controls = dict(controls)
            if regression == "lasso":
                # MLE's residual has factor 1/(2*sigma^2), unlike the WSINDy MSE.
                mle_controls["rho_1"] /= 2.0 * 0.2**2
            mle = wendy_mle(
                u, space, time, alpha=1.0, noise_std=0.2, regression=regression,
                tol=1e-8, **mle_controls, **settings,
            )
            np.testing.assert_allclose(mle, expected, atol=1e-10)

    def test_irls_lasso_fixed_point_and_weighted_kkt(self):
        u, space, time, settings = advection_data()
        rho_1 = 1e-3
        beta = wendy(
            u, space, time, rho_1=rho_1, normality_tol=None,
            reweight_tol=1e-9, tol=1e-10, **settings,
        )
        problem = _WeakResidual(u, space, time, **settings)
        X, y = problem.whiten(beta, 1e-10)
        gradient = 2.0 * X.T @ (X @ beta - y) / len(y)
        active = beta != 0
        self.assertTrue(np.any(active))
        self.assertTrue(np.any(~active))
        np.testing.assert_allclose(gradient[active], -rho_1 * np.sign(beta[active]), atol=2e-9)
        self.assertLessEqual(np.max(np.abs(gradient[~active])), rho_1 + 2e-9)

    def test_mle_lasso_stationarity_and_exact_zeros(self):
        u, space, time, settings = advection_data()
        rho_1 = 10.0
        beta = wendy_mle(u, space, time, noise_std=0.01, rho_1=rho_1, **settings)
        problem = _WeakResidual(u, space, time, **settings)
        _, gradient = problem.likelihood(beta, 0.01, 0.0)
        scales = np.linalg.norm(problem.y) / np.linalg.norm(problem.X, axis=0)
        active = beta != 0
        self.assertTrue(np.any(active))
        self.assertTrue(np.any(~active))
        violation = np.maximum(np.abs(gradient) - rho_1, 0.0)
        violation[active] = np.abs(gradient[active] + rho_1 * np.sign(beta[active]))
        self.assertLessEqual(np.max(scales * violation), 1e-6)

    def test_two_dimensional_diffusion_with_both_sparsity_options(self):
        x = np.linspace(0.0, 2.0 * np.pi, 33)
        y = np.linspace(0.0, 2.0 * np.pi, 35)
        time = np.linspace(0.0, 0.5, 25)
        u = np.zeros((len(x), len(y), len(time)))
        modes = [(1, 0, 1.0, 0.1), (0, 1, 0.7, -0.2), (1, 1, 0.8, 0.4),
                 (1, -1, 0.6, -0.7), (2, 1, 0.4, 0.9)]
        for kx, ky, amplitude, phase in modes:
            rate = 0.3 * kx**2 - 0.8 * kx * ky + ky**2
            angle = kx * x[:, None, None] + ky * y[None, :, None] + phase
            u += amplitude * np.cos(angle) * np.exp(-rate * time[None, None, :])
        u += np.random.default_rng(1).normal(scale=0.01, size=u.shape)
        settings = dict(max_derivative_order=2, max_polynomial_degree=1, strides=(8, 8, 5))
        expected = [0.0, 0.0, 0.3, -0.8, 1.0]
        for method, extra in ((wendy, {"normality_tol": None}), (wendy_mle, {"noise_std": 0.01})):
            for regression, controls in (
                ("lasso", {"rho_1": 1e-3}),
                ("mstls", {"thresholds": (0.01, 0.05, 0.1)}),
            ):
                with self.subTest(method=method.__name__, regression=regression):
                    beta = method(u, (x, y), time, regression=regression, **extra, **controls, **settings)
                    self.assertLess(np.linalg.norm(beta - expected), 0.15 if regression == "lasso" else 0.025)
                    if regression == "mstls":
                        np.testing.assert_array_equal(beta[:2], [0.0, 0.0])
                    if method is wendy_mle and regression == "mstls":
                        problem = _WeakResidual(u, (x, y), time, **settings)
                        _, gradient = problem.likelihood(beta, 0.01, 0.0)
                        scales = np.linalg.norm(problem.y) / np.linalg.norm(problem.X, axis=0)
                        self.assertLessEqual(np.max(np.abs(scales[2:] * gradient[2:])), 1e-6)

    def test_nonconvergence_is_reported(self):
        u, space, time, settings = advection_data()
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            wendy(u, space, time, max_reweights=1, normality_tol=None, **settings)
        with self.assertRaisesRegex(RuntimeError, "stationarity tolerance"):
            wendy_mle(u, space, time, noise_std=0.01, max_iter=1, **settings)

    def test_normality_stop_is_reported_separately(self):
        u, space, time, settings = advection_data()
        with self.assertWarnsRegex(RuntimeWarning, "normality test"):
            beta = wendy(
                u, space, time, normality_tol=0.999, normality_start=1,
                reweight_tol=1e-14, **settings,
            )
        self.assertEqual(beta.shape, (2,))
        self.assertTrue(np.all(np.isfinite(beta)))

    def test_invalid_controls_and_degenerate_noise(self):
        u, space, time, settings = advection_data()
        for sigma in (0.0, -0.1, np.inf, np.nan, 1e-300, 1e200):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                wendy_mle(u, space, time, noise_std=sigma, **settings)
        invalid = [
            {"alpha": -0.1}, {"alpha": 1.1}, {"alpha": np.nan},
            {"rho_1": -0.1}, {"regression": "unknown"},
            {"regression": "mstls", "rho_1": 0.1}, {"thresholds": (0.1,)},
            {"regression": "mstls", "thresholds": (0.0,)},
            {"initial_beta": [0.0]}, {"initial_beta": [0.0, np.nan]},
            {"max_iter": 0}, {"tol": 0.0},
        ]
        for method, extra in ((wendy, {}), (wendy_mle, {"noise_std": 0.01})):
            for controls in invalid:
                with self.subTest(method=method.__name__, controls=controls), self.assertRaises(ValueError):
                    method(u, space, time, **extra, **controls, **settings)
        for controls in ({"max_reweights": 0}, {"reweight_tol": 0}, {"normality_tol": 1.0}, {"normality_start": 0}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                wendy(u, space, time, **controls, **settings)


if __name__ == "__main__":
    unittest.main()
