"""Check MSTLS bounds, repeated refits, threshold selection, and public routing."""

import unittest

import numpy as np

from methods import _mstls, build_wsindy_system, sindy, wsindy
from simulation_generation import generate_anisotropic_porous_medium


class RegressionTests(unittest.TestCase):
    def test_mstls_uses_coefficient_and_contribution_bounds(self):
        # The first term fails the coefficient lower bound, the second the
        # coefficient upper bound, and the fourth the contribution lower bound.
        # Retained orthogonal coefficients are refitted without shrinkage.
        X = np.diag([100.0, 0.001, 1.0, 0.001, 1.0])
        y = X @ np.array([0.0002, 200.0, 1.0, 0.5, 0.02])
        beta = _mstls(X, y, thresholds=(0.01,), max_iter=10)
        np.testing.assert_allclose(beta, [0.0, 0.0, 1.0, 0.0, 0.02], atol=1e-14)

        # Scaling every equation equally must preserve the relative bounds.
        scaled = _mstls(17.0 * X, 17.0 * y, thresholds=(0.01,), max_iter=10)
        np.testing.assert_allclose(scaled, beta, atol=1e-14)

    def test_mstls_rejects_large_cancelling_contributions(self):
        # Both coefficients pass the absolute bounds [0.1, 10], but their
        # individual contributions greatly exceed the small target norm.
        X = np.array([[1.0, 1.0], [0.0, 0.001]])
        y = X @ np.array([0.5, -0.49])
        beta = _mstls(X, y, thresholds=(0.1,), max_iter=10)
        np.testing.assert_array_equal(beta, np.zeros(2))

    def test_mstls_refits_after_thresholding(self):
        X = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        y = X @ np.array([1.0, 0.05])
        beta = _mstls(X, y, thresholds=(0.1,), max_iter=1)
        # Refitting shifts the retained coefficient by 0.05*(X_1.T X_2)/2.
        np.testing.assert_allclose(beta, [1.025, 0.0], atol=1e-14)

    def test_mstls_repeats_thresholding_until_support_stabilizes(self):
        X = np.array([[1.0, -0.9, 0.0], [0.0, np.sqrt(0.19), 0.0], [0.0, 0.0, 1.0]])
        y = X @ np.array([0.15, 0.09, 1.0])
        # Removing the second coefficient moves the first from 0.15 to 0.069,
        # so another threshold-and-refit step must remove it as well.
        with self.assertRaisesRegex(RuntimeError, "MSTLS did not converge"):
            _mstls(X, y, thresholds=(0.1,), max_iter=1)
        beta = _mstls(X, y, thresholds=(0.1,), max_iter=2)
        np.testing.assert_allclose(beta, [0.0, 0.0, 1.0], atol=1e-14)

    def test_threshold_selection_uses_the_ols_projection_loss(self):
        X = np.vstack([np.eye(4), np.zeros(4)])
        y = np.array([1.0, 0.1, 0.01, 0.001, 100.0])
        # The last observation is orthogonal to the entire library. Eq. (4.4)
        # compares fitted projections, not residuals against this observation.
        # Candidate scores are approximately 1, 0.51, 0.35, 1; keep one term.
        beta = _mstls(X, y, thresholds=(0.0002, 0.002, 2.0, 1e-6), max_iter=10)
        np.testing.assert_array_equal(beta, [1.0, 0.0, 0.0, 0.0])

    def test_threshold_loss_ties_choose_the_smallest_threshold(self):
        # Full and zero models both have loss exactly one. Input order must
        # not let the larger threshold's zero model win this tie.
        beta = _mstls(np.eye(2), np.ones(2), thresholds=(2.0, 0.01), max_iter=10)
        np.testing.assert_array_equal(beta, np.ones(2))

    def test_public_regression_choices_on_advection(self):
        x = np.linspace(-1.0, 1.0, 81)
        time = np.linspace(0.0, 0.5, 61)
        phase = x[:, None] + 0.4 * time[None, :]
        u = np.sin(phase) + 0.2 * np.cos(2.0 * phase)
        controls = {"max_derivative_order": 2, "max_polynomial_degree": 1}
        for method, atol in ((sindy, 6e-5), (wsindy, 1e-8)):
            with self.subTest(method=method.__name__):
                inputs = (u, (x,), time)
                for thresholds in (None, (0.05,)):
                    beta = method(*inputs, regression="mstls", thresholds=thresholds, **controls)
                    np.testing.assert_allclose(beta, [0.4, 0.0], rtol=0, atol=atol)
                    self.assertEqual(beta[1], 0.0)
                for rho_1 in (0.0, 1e-6):
                    default = method(*inputs, rho_1=rho_1, **controls)
                    explicit = method(*inputs, regression="lasso", rho_1=rho_1, **controls)
                    np.testing.assert_array_equal(default, explicit)

    def test_full_weak_pipeline_refits_selected_terms(self):
        for noise_ratio in (0.0, 0.05):
            data = generate_anisotropic_porous_medium(
                nx=36, ny=40, nt=20, noise_ratio=noise_ratio, seed=0
            )
            inputs = (data.u_observed, data.spatial_grid, data.time)
            X, target = build_wsindy_system(*inputs)
            beta = wsindy(*inputs, regression="mstls")
            self.assertEqual(beta.shape, (100,))
            self.assertTrue(np.all(np.isfinite(beta)))
            active = beta != 0
            self.assertTrue(np.any(active))
            self.assertTrue(np.any(~active))
            retained_X = X[:, active] / np.linalg.norm(X[:, active], axis=0)
            gradient = retained_X.T @ (X @ beta - target)
            self.assertLess(np.linalg.norm(gradient) / np.linalg.norm(target), 1e-10)

    def test_zero_target_and_rank_deficiency(self):
        beta = _mstls(np.zeros((2, 4)), np.zeros(2), thresholds=None, max_iter=10)
        np.testing.assert_array_equal(beta, np.zeros(4))
        with self.assertRaises(np.linalg.LinAlgError):
            _mstls(np.ones((3, 2)), np.ones(3), thresholds=None, max_iter=10)

    def test_invalid_or_conflicting_regression_controls(self):
        x = np.linspace(0.0, 1.0, 17)
        time = np.linspace(0.0, 0.5, 9)
        u = x[:, None] + time[None, :]
        invalid = [
            {"regression": "unknown"},
            {"regression": "mstls", "rho_1": 0.1},
            {"regression": "lasso", "thresholds": (0.1,)},
            {"regression": "mstls", "max_iter": 0},
            {"regression": "mstls", "max_iter": 1.5},
        ]
        for thresholds in ((), (0.0,), (-0.1,), (np.nan,), (np.inf,), 0.1, ((0.1,),)):
            invalid.append({"regression": "mstls", "thresholds": thresholds})
        for method in (sindy, wsindy):
            for parameters in invalid:
                with self.subTest(method=method.__name__, parameters=parameters):
                    with self.assertRaises(ValueError):
                        method(u, (x,), time, **parameters)


if __name__ == "__main__":
    unittest.main()
