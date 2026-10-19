"""The debiased weak score (Eq. 8) and the discrete Dantzig selector on it (Eqs. 9-10)."""
import itertools
import unittest
import numpy as np
from scipy.optimize import linprog
from pdes import metrics
from wsindy import Dantzig, SplitSystem, WeakSystem, fit
from wsindy.dantzig import min_residual, score
from test_wsindy import ADVECTION, advection


def split_system(pilot_error=.05, noise=.1):
    """Advection observed with noise; the pilot is the clean state plus a smooth error at the even times."""
    clean, grid = advection()
    rng = np.random.default_rng(0)
    U = clean+noise*rng.normal(size=clean.shape)
    pilot = clean+pilot_error*np.cos(grid[0])[:, None, None]
    pilot[:, 1::2] = np.nan
    return SplitSystem(ADVECTION, U, grid, X=pilot), clean


def gurobi(test):
    try:
        return test()
    except Exception as error:
        if "license" in str(error).lower():
            raise unittest.SkipTest(str(error))
        raise


class Score(unittest.TestCase):
    def test_debiased_score_is_the_first_order_expansion_along_U_minus_uhat(self):
        system, _ = split_system()
        H, b, _ = score(system)
        direction, step = system.U-system.X_state, 1e-5
        expansions = []
        for state in (system.X_state+step*direction, system.X_state-step*direction):
            X, Y = system.columns(system.terms, state), system.columns(system.lhs, state)
            expansions.append((X.T@X/len(X), X.T@Y/len(X)))
        X, Y = system.X, system.columns(system.lhs, system.X_state)   # the score at uhat, plus its derivative
        np.testing.assert_allclose(H, X.T@X/len(X)+(expansions[0][0]-expansions[1][0])/(2*step), rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(b, X.T@Y/len(X)+(expansions[0][1]-expansions[1][1])/(2*step), rtol=1e-6, atol=1e-10)

    def test_exact_pilot_gives_the_observed_score(self):
        system, _ = split_system()
        system.X_state = np.where(np.isnan(system.X_state), np.nan, system.U)
        H, b, _ = score(system)
        observed = SplitSystem(ADVECTION, system.U, system.grid, X=system.X_state, Y=system.U)
        H0, b0, _ = score(observed, "ls")
        np.testing.assert_allclose(H, H0, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(b, b0, rtol=1e-12, atol=1e-14)


class Selector(unittest.TestCase):
    def test_min_residual_matches_exhaustive_supports(self):
        rng = np.random.default_rng(19)
        A = rng.normal(size=(6, 6))
        A, y = A+A.T, rng.normal(size=6)   # indefinite, like a debiased H
        for k in (1, 2):
            best = min(linprog(np.r_[np.zeros(k), 1.], A_ub=np.block([[A[:, s], -np.ones((6, 1))], [-A[:, s], -np.ones((6, 1))]]),
                               b_ub=np.r_[y, -y], bounds=[(None, None)]*k+[(0, None)], method="highs").fun
                       for s in map(list, itertools.combinations(range(6), k)))
            beta, optimal = gurobi(lambda: min_residual(A, y, k))
            self.assertTrue(optimal)
            self.assertLessEqual(np.count_nonzero(beta), k)
            self.assertAlmostEqual(np.max(np.abs(y-A@beta)), best, places=7)

    def test_recovers_advection(self):
        U, grid = advection()
        system = WeakSystem(ADVECTION, U, grid)
        truth = ADVECTION.truth_matrix(system.terms)
        for lam in ("theory", "oracle"):
            coef = gurobi(lambda: fit(system, Dantzig(lam=lam, loss="ls")).coef)
            self.assertTrue(metrics(coef, truth)["exact_support"], lam)
            np.testing.assert_allclose(coef[truth != 0], -1, rtol=1e-6)

    def test_debiased_recovers_advection_from_a_split_pilot(self):
        system, _ = split_system()
        result = gurobi(lambda: fit(system, Dantzig(lam="theory")))
        truth = ADVECTION.truth_matrix(system.terms)
        self.assertTrue(metrics(result.coef, truth)["exact_support"])
        self.assertEqual(result.info["k"], [1])
        self.assertLessEqual(result.info["radii"][0][-1], result.info["lam"][0]+1e-9)


if __name__ == "__main__":
    unittest.main()
