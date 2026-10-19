import unittest
from dataclasses import replace
import numpy as np
import pysindy as ps
from scipy.optimize import approx_fprime
from pdes import AUTHORS, PDE, PDES, Profile, metrics, observe, trial_seed
from wsindy import (BestSubset, Block, EIV, FromAPI, FromCallable, FromPySINDy, MSTLS, StrongSystem, Tests, WeakSystem,
                    estimate, fit, spatial)
from wsindy.eiv import conic
from wsindy.testfunctions import changepoint_supports, poly_weights


def advection(n=128, steps=101):
    """u = sin(x - t) + 0.5 cos(2(x - t)) on a periodic grid: u_t = -u_x."""
    x, t = np.linspace(0, 2*np.pi, n, endpoint=False), np.linspace(0, 2, steps)
    xi = x[:, None]-t[None, :]
    return (np.sin(xi)+.5*np.cos(2*xi))[..., None], (x, t)


ADVECTION = PDE("advection", "", "", ("u",), (Block(ps.PolynomialLibrary(degree=2), spatial(2, 1)),), ("u_t",),
                ({"u_x": -1.},), Tests(m=(16, 12), s=(4, 4)))


class Dictionary(unittest.TestCase):
    def test_paper_dictionaries(self):
        sizes = {"IB": 43, "KdV": 43, "KS": 43, "NLS": 190, "RD": 181, "NS": 50, "HKS": 73, "VBG": 43}
        for name, size in sizes.items():
            terms = PDES[name].dictionary_terms()[1]
            self.assertEqual(len(terms), size, name)
            PDES[name].truth_matrix(terms)   # every true term is in the dictionary
            self.assertFalse(any(t.powers == (0,)*len(t.powers) and any(t.alpha) for t in terms))

    def test_general_features(self):
        block = Block(ps.PolynomialLibrary(degree=1)+ps.FourierLibrary(n_frequencies=1), spatial(1, 1))
        pde = replace(ADVECTION, dictionary=(block,), _cache={})
        terms = pde.dictionary_terms()[1]
        self.assertIn("(sin(1 u))_x", [t.label for t in terms])
        U, grid = advection()
        system = WeakSystem(pde, U, grid)
        self.assertTrue(np.all(np.isfinite(system.X)))


class TestFunctions(unittest.TestCase):
    def test_weights_are_derivatives(self):
        m, p = 20, 8
        r = np.linspace(-1, 1, 2*m+1)
        weights = poly_weights(m, 2, p)
        phi = lambda x: (1-x[0]**2)**p
        numeric = [approx_fprime([x], phi, 1e-7)[0] for x in r[5:-5]]
        np.testing.assert_allclose(weights[1, 5:-5], numeric, rtol=1e-5, atol=1e-6)

    def test_algorithm_4_1_reproduces_table_degrees(self):
        for name in ("IB", "KdV", "KS", "NLS", "RD", "NS"):
            pde = PDES[name]
            terms = (*pde.dictionary_terms()[1], *pde.lhs_terms())
            orders = tuple(max(t.alpha[d] for t in terms) for d in range(len(pde.tests.m)))
            grid = tuple(np.arange(400.) for _ in orders)
            self.assertEqual(replace(pde.tests, p=None).resolve(None, grid, orders).p, pde.tests.p, name)

    def test_changepoint_supports(self):
        U, grid = advection()
        m = changepoint_supports(U, grid, 1e-10, 2.)
        self.assertEqual(len(m), 2)
        self.assertTrue(all(2 <= w <= (len(g)-1)//2 for w, g in zip(m, grid)))
        resolved = Tests().resolve(U, grid, (2, 1))
        self.assertEqual(resolved.m, m)
        self.assertEqual(resolved.s, (128//50, 101//50))


class Regression(unittest.TestCase):
    def test_recovers_advection(self):
        U, grid = advection()
        system = WeakSystem(ADVECTION, U, grid)
        for regression in ("mstls", "lasso", MSTLS(loss="debiased"), FromAPI(ps.STLSQ(threshold=.1, alpha=0.))):
            coef = fit(system, regression).coef
            truth = ADVECTION.truth_matrix(system.terms)
            self.assertTrue(metrics(coef, truth)["exact_support"], regression)
            if regression != "lasso":   # LASSO keeps its shrinkage
                np.testing.assert_allclose(coef[truth != 0], -1, rtol=1e-6)

    def test_strong_form_recovers_advection(self):
        U, grid = advection(steps=401)
        system = StrongSystem(ADVECTION, U, grid)   # spectral derivatives would assume periodic time
        for regression in ("mstls", FromAPI(ps.STLSQ(threshold=.1, alpha=0.))):
            coef = fit(system, regression).coef
            self.assertTrue(metrics(coef, ADVECTION.truth_matrix(system.terms))["exact_support"])
            np.testing.assert_allclose(coef[0, [t.label for t in system.terms].index("u_x")], -1, rtol=1e-3)

    def test_scaling_restores_physical_least_squares(self):
        U, grid = advection()
        def least_squares(profile):
            system = WeakSystem(replace(ADVECTION, profile=profile, tests=replace(ADVECTION.tests, p=(9, 9))), U, grid)
            return np.linalg.lstsq(system.X, system.Y, rcond=None)[0]*system.scales
        raw = Profile(False, AUTHORS.thresholds, "relative", False, "rms", "per_axis")
        np.testing.assert_allclose(least_squares(AUTHORS), least_squares(raw), rtol=1e-6, atol=1e-9)

    def test_debiased_loss_with_exact_pilot_is_least_squares(self):
        U, grid = advection()
        system = WeakSystem(ADVECTION, U, grid, X=U)
        np.testing.assert_allclose(fit(system, MSTLS(loss="debiased")).coef, fit(system, "mstls").coef, rtol=1e-10)

    def test_empty_support_rule(self):
        rng = np.random.default_rng(0)
        U, grid = rng.normal(size=(64, 60, 1)), (np.arange(64.), np.arange(60.))
        pde = replace(ADVECTION, tests=Tests(m=(8, 8), s=(2, 2)))
        system = WeakSystem(pde, U, grid)
        empty = MSTLS(thresholds=[1.], allow_empty=True).regress(system)
        kept = MSTLS(thresholds=[1.], allow_empty=False).regress(system)
        self.assertFalse(np.any(empty.coef))
        self.assertTrue(np.any(kept.coef))

    def test_eiv(self):
        require_gurobi(self)
        X = np.eye(3)
        beta, t, status, success, _ = conic(X, np.array([1e-3, 0, 0]), mu=0., tau=1., lam=1.)
        self.assertEqual(status, "AnalyticZero")
        beta, t, status, success, _ = conic(X, np.array([2., 0, 0]), mu=0., tau=.5, lam=1.)
        self.assertTrue(success)
        np.testing.assert_allclose(beta, [1.5, 0, 0], atol=1e-6)
        U, grid = advection()
        system = WeakSystem(replace(ADVECTION, profile=Profile(False, AUTHORS.thresholds, "relative", False,
                                                               "rms", "per_axis")), U, grid)
        system.noise_std = [0.]
        for normalize in (False, True):   # oracle tau keeps beta* feasible; lam is chosen on the grid
            result = fit(system, EIV(oracle=True, normalize=normalize))
            self.assertEqual(result.coef.shape, (1, 7))
            self.assertIn(result.info["lam"], result.info["lams"])
        with self.assertRaises(ValueError):
            fit(system, EIV())   # tau and mu are required without the oracle


    def test_best_subset_is_exact(self):
        require_gurobi(self)
        from itertools import combinations
        from types import SimpleNamespace
        rng = np.random.default_rng(3)
        X = rng.normal(size=(80, 9))
        X[:, 1] = X[:, 0]+.1*rng.normal(size=80)   # correlated columns defeat greedy selection
        Y = np.column_stack([X@np.r_[1., -1., 0, 0, .5, 0, 0, 0, 0], X@np.r_[0, 0, 2., 0, 0, 0, 0, -1., 0]])
        Y += .3*rng.normal(size=Y.shape)
        system = SimpleNamespace(X=X, Y=Y, scales=np.ones((9, 2)))
        def best(y, r):   # exhaustive search
            return min(np.linalg.norm(y-X[:, list(c)]@np.linalg.lstsq(X[:, list(c)], y, rcond=None)[0])**2
                       for c in combinations(range(9), r))
        for r in ((3, 2), (2, 2)):
            W = BestSubset(r=r).regress(system).coef
            self.assertEqual([int(np.count_nonzero(w)) for w in W], list(r))
            for e in range(2):
                self.assertAlmostEqual(np.linalg.norm(Y[:, e]-X@W[e])**2, best(Y[:, e], r[e]), places=8)
        independent = rng.normal(size=(80, 9))   # r = 1, 2, ... per equation until the loss increases
        signal = np.column_stack([independent@np.r_[1., -1., 0, 0, .5, 0, 0, 0, 0],
                                  independent@np.r_[0, 0, 2., 0, 0, 0, 0, -1., 0]])
        greedy = BestSubset().regress(SimpleNamespace(X=independent, Y=signal+.01*rng.normal(size=(80, 2)),
                                                      scales=np.ones((9, 2))))
        self.assertEqual(greedy.info["r"], [3, 2])
        self.assertTrue(all(path[-1] > path[-2] for path in greedy.info["losses"]))
        fit_total = BestSubset(r=5).regress(system)
        self.assertTrue(fit_total.info["optimal"])
        self.assertEqual(np.count_nonzero(fit_total.coef), 5)
        total = min(best(Y[:, 0], a)+best(Y[:, 1], 5-a) for a in range(1, 5))
        self.assertAlmostEqual(np.linalg.norm(Y-X@fit_total.coef.T)**2, total, places=8)

    def test_best_subset_with_fewer_rows_than_terms(self):
        require_gurobi(self)
        from itertools import combinations
        from types import SimpleNamespace
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 20))   # K = 10 weak rows < J = 20 terms
        Y = X[:, [2, 5, 7]]@np.array([[1.], [-2.], [.5]])+.01*rng.normal(size=(10, 1))
        system = SimpleNamespace(X=X, Y=Y, scales=np.ones((20, 1)))
        W = BestSubset(r=3).regress(system).coef[0]
        best = min(np.linalg.norm(Y[:, 0]-X[:, list(c)]@np.linalg.lstsq(X[:, list(c)], Y[:, 0], rcond=None)[0])**2
                   for c in combinations(range(20), 3))
        self.assertEqual(np.count_nonzero(W), 3)
        self.assertAlmostEqual(np.linalg.norm(Y[:, 0]-X@W)**2, best, places=8)
        self.assertEqual(BestSubset().regress(system).coef.shape, (1, 20))

    def test_best_subset_magnitude_prior(self):
        require_gurobi(self)
        from types import SimpleNamespace
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 6))
        X[:, 1] = X[:, 0]/1e3+1e-5*rng.normal(size=60)   # a rescaled copy, fitted with a coefficient ~1e3
        y = X[:, 0]+.3*rng.normal(size=60)
        system = SimpleNamespace(X=X, Y=y[:, None], scales=np.full((6, 1), 2.))
        self.assertEqual(np.flatnonzero(BestSubset(r=1).regress(system).coef[0]).tolist(), [1])
        regression = BestSubset(r=1, threshold=.05)
        lower, upper = regression._bounds(X, y[:, None], system.scales)
        def clipped(j):   # best |w| in [lower, upper] with the least-squares sign
            w = X[:, j]@y/(X[:, j]@X[:, j])
            return np.sign(w)*np.clip(abs(w), lower[j, 0], upper[j, 0])
        objective = [np.sum((y-X[:, j]*clipped(j))**2) for j in range(6)]
        c = regression.regress(system).coef[0]
        (j,) = np.flatnonzero(c)
        self.assertEqual(j, int(np.argmin(objective)))
        self.assertEqual(j, 0)
        self.assertAlmostEqual(c[j], 2*clipped(j), places=6)

    def test_best_subset_recovers_advection(self):
        require_gurobi(self)
        U, grid = advection()
        system = WeakSystem(ADVECTION, U, grid)
        for regression in (BestSubset(r=1), BestSubset()):
            coef = fit(system, regression).coef
            self.assertTrue(metrics(coef, ADVECTION.truth_matrix(system.terms))["exact_support"])
            np.testing.assert_allclose(coef[coef != 0], -1, rtol=1e-6)


def require_gurobi(test):
    try:
        from wsindy.regression import gurobi_env
        gurobi_env()
    except Exception:
        test.skipTest("gurobipy or a Gurobi license is unavailable")


class Estimators(unittest.TestCase):
    def test_external_estimators(self):
        U, grid = advection()
        noisy = U+.05*np.random.default_rng(2).normal(size=U.shape)
        for estimator in (ps.SmoothedFiniteDifference(), FromPySINDy(ps.SmoothedFiniteDifference(), axes=(0, -2)),
                          FromCallable(lambda V, g: V), "movavg"):
            uhat, _ = estimate(noisy, grid, ADVECTION, estimator)
            self.assertEqual(uhat.shape, U.shape)
        smoothed = estimate(noisy, grid, ADVECTION, ps.SmoothedFiniteDifference())[0]
        self.assertLess(np.linalg.norm(smoothed-U), np.linalg.norm(noisy-U))


class Observations(unittest.TestCase):
    def test_noise_and_seeds(self):
        U = np.ones((50, 40, 2))*np.array([1., 3.])
        noisy, stds = observe(PDES["NLS"], U, .5, trial_seed("NLS", .5, 0))
        np.testing.assert_allclose(stds, [.5, 1.5])
        self.assertEqual(trial_seed("KS", .5, 3), trial_seed("KS", .5, 3))
        self.assertNotEqual(trial_seed("KS", .5, 3), trial_seed("KS", .5, 4))
        np.testing.assert_allclose(noisy.std(axis=(0, 1)), stds, rtol=.05)


if __name__ == "__main__":
    unittest.main()
