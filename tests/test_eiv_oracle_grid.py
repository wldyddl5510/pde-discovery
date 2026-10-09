"""Oracle selection safeguards, without loading any benchmark dataset."""
import unittest

import numpy as np

from benchmark_results.eiv import run_oracle_grid as grid
from eiv_study import EIVResult, conic_eiv


def synthetic_result(*, beta=(0.,), t=1., objective=1., dual=1.,
                     residual_violation=0., cone_violation=0., success=True):
    coefficients = np.asarray(beta, dtype=float)
    return EIVResult(coefficients, t, objective, "Solved", success,
                     np.abs(coefficients) > grid.SUPPORT_TOLERANCE,
                     dict(objective_dual=dual,
                          residual_violation=residual_violation,
                          cone_violation=cone_violation))


class CandidateValidationTests(unittest.TestCase):
    def test_tiny_response_requires_relative_residual_accuracy_despite_solved(self):
        # A 1e-12 violation is small in absolute units but is 1e-4 of this
        # response. A solver status alone must not make it an oracle candidate.
        fit = synthetic_result(residual_violation=1e-12)
        self.assertTrue(fit.success)
        self.assertEqual(fit.status, "Solved")
        checked = grid.candidate_validation(fit, y_inf=1e-8, mu=0., tau=0.)
        self.assertFalse(checked['eligible'])
        self.assertIn('physical_residual_violation', checked['rejection_reasons'])
        self.assertAlmostEqual(checked['residual_limit'], 1e-14, delta=1e-27)

        fit.diagnostics['residual_violation'] = 1e-15
        self.assertTrue(grid.candidate_validation(fit, 1e-8, 0., 0.)['eligible'])

    def test_returned_coefficient_objective_must_match_dual(self):
        checked = grid.candidate_validation(
            synthetic_result(objective=2., dual=1.), 1., 0., 0.)
        self.assertFalse(checked['eligible'])
        self.assertEqual(checked['returned_objective_dual_gap'], 1.)
        self.assertIn('returned_objective_dual_gap', checked['rejection_reasons'])

    def test_candidate_cannot_cost_more_than_feasible_zero_beta(self):
        # With lambda=1, beta=0,t=(1-.25)/1 is feasible with objective .75.
        checked = grid.candidate_validation(
            synthetic_result(t=2., objective=2., dual=2.), 1., 1., .25)
        self.assertFalse(checked['eligible'])
        self.assertEqual(checked['feasible_zero_beta_objective'], .75)
        self.assertIn('worse_than_feasible_zero_beta', checked['rejection_reasons'])

        checked = grid.candidate_validation(
            synthetic_result(t=.75, objective=.75, dual=.75), 1., 1., .25)
        self.assertTrue(checked['eligible'])

    def test_mu_zero_has_no_finite_zero_beta_upper_bound_below_y(self):
        checked = grid.candidate_validation(synthetic_result(), 1., 0., .25)
        self.assertTrue(checked['eligible'])
        self.assertIsNone(checked['feasible_zero_beta_objective'])

    def test_failed_solver_and_cone_violation_are_rejected(self):
        cases = (
            (synthetic_result(success=False), 'solver_not_successful'),
            (synthetic_result(cone_violation=.01), 'cone_violation'),
        )
        for fit, reason in cases:
            with self.subTest(reason=reason):
                checked = grid.candidate_validation(fit, 1., 0., 0.)
                self.assertFalse(checked['eligible'])
                self.assertIn(reason, checked['rejection_reasons'])

    def test_nonfinite_outputs_never_become_eligible(self):
        for overrides in ({'beta': (np.nan,)}, {'t': np.inf},
                          {'objective': np.nan}, {'dual': np.inf},
                          {'residual_violation': np.nan},
                          {'cone_violation': np.inf}):
            with self.subTest(overrides=overrides):
                checked = grid.candidate_validation(synthetic_result(**overrides), 1., 0., 0.)
                self.assertFalse(checked['eligible'])
                self.assertIn('nonfinite_output', checked['rejection_reasons'])


class OracleSelectionTests(unittest.TestCase):
    def test_chooses_coefficient_e2_not_objective_or_support(self):
        best_e2 = dict(candidate_id=2, eligible=True, e2=.1,
                       objective=100., exact_support=False)
        attractive_but_worse = dict(candidate_id=1, eligible=True, e2=.2,
                                   objective=.001, exact_support=True)
        self.assertIs(grid.select_best([attractive_but_worse, best_e2]), best_e2)

    def test_excludes_failures_and_nonfinite_errors_even_when_scores_look_best(self):
        good = dict(candidate_id=8, eligible=True, e2=.5)
        rejected = [dict(candidate_id=0, eligible=False, e2=0.),
                    dict(candidate_id=1, e2=0.)]
        rejected += [dict(candidate_id=i+2, eligible=True, e2=error)
                     for i, error in enumerate((None, np.nan, np.inf, -np.inf))]
        self.assertIs(grid.select_best([*rejected, good]), good)
        self.assertIsNone(grid.select_best(rejected))
        self.assertIsNone(grid.select_best([]))

    def test_equal_e2_tie_break_is_deterministic_candidate_id(self):
        earlier = dict(candidate_id=1, eligible=True, e2=.2)
        later = dict(candidate_id=3, eligible=True, e2=.2)
        self.assertIs(grid.select_best([later, earlier]), earlier)
        self.assertIs(grid.select_best([earlier, later]), earlier)


class GridRefinementTests(unittest.TestCase):
    def test_coarse_and_refined_domain_contains_zero_and_excludes_trivial_tau(self):
        self.assertEqual(grid.MU_RATIOS[0], 0.)
        self.assertEqual(grid.TAU_RATIOS[0], 0.)
        self.assertTrue(all(0 <= value < 1 for value in grid.TAU_RATIOS))
        for coarse in (grid.MU_RATIOS, grid.TAU_RATIOS):
            for index, value in enumerate(coarse):
                with self.subTest(coarse=coarse, center=value):
                    refined = grid.neighborhood(value, coarse)
                    low = coarse[max(0, index-1)]
                    high = coarse[min(len(coarse)-1, index+1)]
                    self.assertEqual(refined, sorted(set(refined)))
                    self.assertTrue(all(low <= point <= high for point in refined))
                    self.assertGreater(len(refined), 1)
                    if low == 0:
                        self.assertIn(0., refined)
            if coarse is grid.TAU_RATIOS:
                self.assertLess(max(grid.neighborhood(coarse[-1], coarse)), 1.)

    def test_interior_refinement_contains_both_neighboring_endpoints(self):
        refined = grid.neighborhood(.01, [0., .001, .01, .1, .5])
        self.assertEqual(len(refined), 7)
        self.assertEqual(refined[0], .001)
        self.assertEqual(refined[-1], .1)
        np.testing.assert_allclose(np.diff(np.log10(refined)), 1/3,
                                   rtol=1e-13, atol=1e-13)

    def test_real_scalar_conic_path_selects_smallest_coefficient_error(self):
        # mu=0 gives min 2*|beta| subject to |1-beta|<=tau, so beta=1-tau.
        # Its conic objective decreases as its coefficient error increases.
        candidates = []
        for index, tau in enumerate((0., .25, .75)):
            fit = conic_eiv(np.array([[1.]]), np.array([1.]), mu=0., tau=tau,
                            lam=1., solver_tolerance=grid.SOLVER_TOLERANCE,
                            support_tolerance=grid.SUPPORT_TOLERANCE, equilibrate=False)
            checked = grid.candidate_validation(fit, 1., 0., tau)
            self.assertTrue(checked['eligible'], (fit.status, checked))
            np.testing.assert_allclose(fit.coefficients, [1-tau], rtol=1e-8, atol=1e-8)
            candidates.append(dict(candidate_id=index, objective=fit.objective,
                                   e2=float(abs(fit.coefficients[0]-1)), **checked))
        self.assertEqual(grid.select_best(candidates)['candidate_id'], 0)
        self.assertGreater(candidates[0]['objective'], candidates[-1]['objective'])


if __name__ == '__main__':
    unittest.main()
