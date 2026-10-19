"""Small exhaustive-support checks for the SOS1 Dantzig optimizer."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from wsindy.dantzig import solve_one, solve_path, validate_score


class Dantzig(unittest.TestCase):
    def test_physical_validation(self):
        H = np.array([[2., 1.], [0., -3.]])
        b = np.array([1., 2.])
        self.assertTrue(validate_score(H, b, np.linalg.solve(H, b), 0.)["validated"])
        self.assertFalse(validate_score(H, b, np.zeros(2), .1)["validated"])

    def test_analytic_zero(self):
        result = solve_one(np.eye(3), np.array([1., -2., .5]), 2.)
        self.assertTrue(result["optimal"])
        self.assertTrue(result["validated"])
        self.assertEqual(result["cardinality"], 0)

    def test_oracle_excludes_invalid_candidate_and_reports_incumbent(self):
        def fake(H, b, lam, **options):
            invalid = lam < .5
            return dict(coefficients=[1.] if invalid else [.8] if lam < 1 else [0.],
                        support=[True], lambda_=lam, validated=not invalid,
                        optimal=lam == 1, status="Mock", runtime=0.)
        with patch("wsindy.dantzig.solve_one", side_effect=fake):
            result = solve_path(np.eye(1), np.ones(1), np.ones(1),
                                ratios=[.1, .5, 1.], refinements=0)
        self.assertEqual(result["coefficients"], [0.])
        self.assertTrue(result["selected"]["optimal"])
        self.assertEqual(result["incumbent_oracle_selected"]["coefficients"], [.8])
        self.assertEqual(result["certified_only_selected"]["coefficients"], [0.])

    def test_nonfinite_residual_is_not_validated(self):
        with np.errstate(over="ignore", invalid="ignore"):
            result = validate_score(np.array([[1e308, -1e308]]), np.ones(1), np.array([1e308, 1e308]), 1.)
        self.assertFalse(result["validated"])

    def test_lower_boundary_search_expands(self):
        def fake(H, b, lam, **options):
            return dict(coefficients=[1-lam], support=[True], lambda_=lam,
                        validated=True, optimal=True, status="Mock", runtime=0.)
        with patch("wsindy.dantzig.solve_one", side_effect=fake):
            result = solve_path(np.eye(1), np.ones(1), np.ones(1),
                                ratios=[.01, .1, 1.], refinements=1)
        self.assertAlmostEqual(result["selected"]["lambda_ratio"], .0001)
        self.assertTrue(result["boundary_best"])

    def test_matches_exhaustive_lexicographic_support_search(self):
        rng = np.random.default_rng(19)
        H = rng.normal(size=(6, 6))
        H = H+H.T  # indefinite: this must not silently turn into least squares
        b, lam = rng.normal(size=6), .35
        best = None
        for cardinality in range(1, 7):
            residuals = []
            for support in itertools.combinations(range(6), cardinality):
                A = H[:, support]
                inequalities = np.vstack([np.column_stack([A, -np.ones(6)]),
                                          np.column_stack([-A, -np.ones(6)])])
                fit = linprog(np.r_[np.zeros(cardinality), 1.], A_ub=inequalities,
                              b_ub=np.r_[b, -b], bounds=[(None, None)]*cardinality+[(0., lam)], method="highs")
                if fit.success:
                    residuals.append(fit.fun)
            if residuals:
                best = cardinality, min(residuals)
                break
        try:
            result = solve_one(H, b, lam, time_limit=10.)
        except Exception as error:
            if "license" in str(error).lower() or "token.gurobi" in str(error).lower():
                self.skipTest(str(error))
            raise
        self.assertTrue(result["optimal"], result)
        self.assertTrue(result["validated"], result)
        self.assertEqual(result["cardinality"], best[0])
        self.assertAlmostEqual(result["residual_inf"], best[1], places=6)

    def test_recovers_noiseless_sparse_score(self):
        rng = np.random.default_rng(24)
        X = rng.normal(size=(30, 5))
        H = X.T@X/len(X)
        truth = np.array([0., -.5, 0., 1., 0.])
        try:
            result = solve_path(H, H@truth, truth, ratios=[1e-5, .01, 1.], refinements=0)
        except Exception as error:
            if "license" in str(error).lower() or "token.gurobi" in str(error).lower():
                self.skipTest(str(error))
            raise
        self.assertTrue(result["selected"]["optimal"])
        np.testing.assert_allclose(result["coefficients"], truth, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
