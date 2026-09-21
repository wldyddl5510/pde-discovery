"""Check weak integrals independently of the regression and of data derivatives."""

import unittest

import numpy as np

from methods import build_wsindy_system, polynomial_library_terms, wsindy
from simulation_generation import generate_anisotropic_porous_medium, generate_anisotropic_porous_medium_3d


class WsindyTests(unittest.TestCase):
    def test_3d_porous_medium_residual_under_grid_refinement(self):
        errors = []
        for refinement in (1, 2):
            size = 31 * refinement + 1
            nt = 15 * refinement + 1
            data = generate_anisotropic_porous_medium_3d(nx=size, ny=size, nz=size, nt=nt)
            # Same physical tests and centers as the 32^3 x 16 benchmark.
            # Only the integration grid is refined. The smaller diagnostic
            # library includes every nonzero truth term; this is not a fit.
            X, target = build_wsindy_system(
                data.u_true, data.spatial_grid, data.time,
                max_derivative_order=2, max_polynomial_degree=2,
                half_widths=tuple(refinement * m for m in (8, 8, 8, 4)),
                strides=tuple(refinement * s for s in (4, 4, 4, 1)),
                test_degrees=(16, 16, 16, 28),
            )
            truth = np.array([
                data.true_coefficients.get(term, 0.0)
                for term in polynomial_library_terms(3, 2, 2)
            ])
            self.assertEqual(X.shape, (512, 18))
            errors.append(np.linalg.norm(X @ truth - target) / np.linalg.norm(target))
        # The front is nonsmooth: check second-order convergence rather than
        # treating either finite observation grid as an exact integral.
        self.assertLess(errors[1], errors[0] / 4)

    def test_all_weak_columns_against_continuous_integrals(self):
        # For exp(a*x+b*y+c*t), every strong derivative is known exactly.
        # Independently integrate phi times that derivative using Gauss
        # quadrature. This checks all mixed derivatives through order five,
        # powers, signs, physical spacings, and the order of translated tests.
        x = np.linspace(-1.0, 1.0, 81)
        y = np.linspace(-0.8, 1.2, 85)
        time = np.linspace(0.0, 0.6, 61)
        rates = (0.8, -0.6, 0.7)
        u = np.exp(
            rates[0] * x[:, None, None]
            + rates[1] * y[None, :, None]
            + rates[2] * time[None, None, :]
        )
        original = u.copy()
        half_widths, strides, degrees = (35, 36, 25), (8, 9, 8), (16, 16, 8)
        X, target = build_wsindy_system(
            u, (x, y), time,
            half_widths=half_widths, strides=strides, test_degrees=degrees,
        )

        nodes, weights = np.polynomial.legendre.leggauss(48)
        integrals = {}
        for power in range(1, 6):
            axis_integrals = []
            for grid, rate, m, stride, degree in zip(
                (x, y, time), rates, half_widths, strides, degrees
            ):
                centers = grid[m:len(grid) - m:stride]
                radius = m * (grid[1] - grid[0])
                locations = centers[:, None] + radius * nodes[None, :]
                integrand = (1.0 - nodes**2)**degree * np.exp(power * rate * locations)
                axis_integrals.append(radius * (integrand @ weights))
            integral = (
                axis_integrals[0][:, None, None]
                * axis_integrals[1][None, :, None]
                * axis_integrals[2][None, None, :]
            )
            integrals[power] = integral.ravel()

        expected = np.column_stack([
            (power * rates[0])**alpha[0] * (power * rates[1])**alpha[1] * integrals[power]
            for alpha, power in polynomial_library_terms(2)
        ])
        self.assertEqual(X.shape, (8, 100))
        np.testing.assert_array_equal(u, original)
        np.testing.assert_allclose(X, expected, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(target, rates[2] * integrals[1], rtol=1e-8, atol=1e-11)

    def test_dense_nonlinear_coefficients(self):
        # Exact smooth solution of u_t = 0.4*u_x - 0.5*d_x(u^2).
        x = np.linspace(-1.0, 1.0, 81)
        time = np.linspace(0.0, 0.5, 81)
        u = (0.3 * x[:, None] + 0.8 + 0.12 * time[None, :]) / (1.0 + 0.3 * time[None, :])
        beta = wsindy(u, (x,), time, max_derivative_order=1, max_polynomial_degree=2)
        np.testing.assert_allclose(beta, [0.4, -0.5], rtol=1e-7, atol=1e-8)

    def test_anisotropic_diffusion_coefficients_on_smooth_data(self):
        errors = []
        for size, nt in ((33, 25), (65, 49)):
            x = np.linspace(0.0, 2.0 * np.pi, size)
            y = np.linspace(0.0, 2.0 * np.pi, size + 2)
            time = np.linspace(0.0, 0.5, nt)
            u = np.zeros((len(x), len(y), nt))
            modes = [
                (1, 0, 1.0, 0.1), (0, 1, 0.7, -0.2), (1, 1, 0.8, 0.4),
                (1, -1, 0.6, -0.7), (2, 1, 0.4, 0.9),
            ]
            for kx, ky, amplitude, phase in modes:
                decay_rate = 0.3 * kx**2 - 0.8 * kx * ky + ky**2
                angle = kx * x[:, None, None] + ky * y[None, :, None] + phase
                u += amplitude * np.cos(angle) * np.exp(-decay_rate * time[None, None, :])
            beta = wsindy(u, (x, y), time, max_derivative_order=2, max_polynomial_degree=1)
            errors.append(np.linalg.norm(beta - [0.0, 0.0, 0.3, -0.8, 1.0]))

        self.assertLess(errors[1], 1e-8)
        self.assertLess(errors[1], 0.1 * errors[0])

    def test_weak_porous_medium_residual_converges_across_the_front(self):
        errors = []
        for size in (65, 129):
            data = generate_anisotropic_porous_medium(nx=size, ny=size, nt=(size + 1) // 2)
            m, mt = (size - 1) // 4, (size - 1) // 8
            # Fix the physical test functions and centers under refinement.
            X, target = build_wsindy_system(
                data.u_observed, data.spatial_grid, data.time,
                max_derivative_order=2, max_polynomial_degree=2,
                half_widths=(m, m, mt), strides=(m // 2, m // 2, mt // 2),
                test_degrees=(12, 12, 12),
            )
            truth = np.array([
                data.true_coefficients.get(term, 0.0)
                for term in polynomial_library_terms(2, 2, 2)
            ])
            self.assertEqual(X.shape, (125, 10))
            errors.append(np.linalg.norm(X @ truth - target) / np.linalg.norm(target))
        self.assertLess(errors[1], 0.002)
        self.assertLess(errors[1], 0.15 * errors[0])

    def test_full_porous_medium_ols_and_lasso_objectives(self):
        for noise_ratio in (0.0, 0.05):
            with self.subTest(noise_ratio=noise_ratio):
                data = generate_anisotropic_porous_medium(
                    nx=36, ny=40, nt=20, noise_ratio=noise_ratio, seed=0
                )
                inputs = (data.u_observed, data.spatial_grid, data.time)
                X, target = build_wsindy_system(*inputs)
                beta = wsindy(*inputs)
                np.testing.assert_array_equal(beta, wsindy(*inputs, rho_1=0.0))
                self.assertEqual(beta.shape, (100,))
                self.assertTrue(np.all(np.isfinite(beta)))
                gradient = (X / np.linalg.norm(X, axis=0)).T @ (X @ beta - target)
                self.assertLess(np.linalg.norm(gradient) / np.linalg.norm(target), 1e-10)

                rho_1 = 1e-6
                beta = wsindy(*inputs, rho_1=rho_1, tol=1e-10)
                gradient = 2.0 * X.T @ (X @ beta - target) / len(target)
                nonzero = beta != 0
                self.assertTrue(np.any(nonzero))
                self.assertTrue(np.any(~nonzero))
                np.testing.assert_allclose(
                    gradient[nonzero], -rho_1 * np.sign(beta[nonzero]), rtol=0, atol=1.1e-10
                )
                self.assertLessEqual(np.max(np.abs(gradient[~nonzero])), rho_1 + 1.1e-10)

    def test_rank_checks_and_positive_penalty_on_zero_data(self):
        x = np.linspace(0.0, 1.0, 17)
        time = np.linspace(0.0, 1.0, 9)
        u = np.zeros((17, 17, 9))
        with self.assertRaises(np.linalg.LinAlgError):
            wsindy(u, (x, x), time)
        with self.assertRaisesRegex(np.linalg.LinAlgError, "Only 1 samples"):
            wsindy(u, (x, x), time, strides=(100, 100, 100))
        beta = wsindy(u, (x, x), time, rho_1=0.1, strides=(100, 100, 100))
        np.testing.assert_array_equal(beta, np.zeros(100))

    def test_invalid_test_functions_and_solver_controls(self):
        x = np.linspace(0.0, 1.0, 17)
        time = np.linspace(0.0, 1.0, 9)
        u = np.ones((17, 9))
        invalid = [
            {"half_widths": (4,)}, {"half_widths": (1, 2)},
            {"half_widths": (9, 2)}, {"half_widths": (3.5, 2)},
            {"strides": (1,)}, {"strides": (1, 0)}, {"strides": (1, 1.5)},
            {"test_degrees": (8,)}, {"test_degrees": (5, 4)},
            {"test_degrees": (8, 1)}, {"test_degrees": (8.5, 4)},
            {"rho_1": -1.0}, {"rho_1": np.nan}, {"rho_1": np.inf},
            {"rho_1": 0.1, "max_iter": 0}, {"rho_1": 0.1, "tol": 0.0},
        ]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    wsindy(u, (x,), time, **parameters)

    def test_invalid_observations_and_grids(self):
        x = np.linspace(0.0, 1.0, 17)
        time = np.linspace(0.0, 1.0, 9)
        u = np.ones((17, 9))
        irregular = x.copy()
        irregular[4] += 0.01
        invalid = [
            (u, (irregular,), time), (u, (x[:-1],), time), (u, (x,), time[:-1]),
            (u[..., None], (x,), time), (u * np.nan, (x,), time),
            (u[:, :4], (x,), time[:4]),
        ]
        for inputs in invalid:
            with self.assertRaises(ValueError):
                build_wsindy_system(*inputs)


if __name__ == "__main__":
    unittest.main()
