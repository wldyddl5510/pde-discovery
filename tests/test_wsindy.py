import unittest
from dataclasses import replace
import numpy as np
import pysindy as ps
from scipy.optimize import approx_fprime
from pdes import AUTHORS, PDE, PDES, Profile, metrics, observe, trial_seed
from wsindy import (Block, CrossFitPilot, EIV, FromAPI, FromCallable, FromPySINDy, MSTLS, StrongSystem, Tests, WeakSystem,
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
        X = np.eye(3)
        beta, t, status, success, _ = conic(X, np.array([1e-3, 0, 0]), mu=0., tau=1., lam=1.)
        self.assertEqual(status, "AnalyticZero")
        beta, t, status, success, _ = conic(X, np.array([2., 0, 0]), mu=0., tau=.5, lam=1.)
        self.assertTrue(success)
        np.testing.assert_allclose(beta, [1.5, 0, 0], atol=1e-6)
        U, grid = advection()
        result = fit(WeakSystem(replace(ADVECTION, profile=Profile(False, AUTHORS.thresholds, "relative", False,
                                                                     "rms", "per_axis")), U, grid), EIV(lam=1.))
        self.assertEqual(result.coef.shape, (1, 7))


    def test_eiv_gurobi_matches_clarabel(self):
        try:
            import gurobipy
            gurobipy.Model().dispose()
        except Exception:
            self.skipTest("gurobipy or a Gurobi license is unavailable")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 10))
        y = X@np.where(rng.random(10) < .3, 1., 0.)+.05*rng.normal(size=200)
        a, b = (conic(X, y, .05, .1*np.abs(y).max(), 1., solver=s) for s in ("clarabel", "gurobi"))
        self.assertTrue(a[3] and b[3])
        np.testing.assert_allclose(np.abs(a[0]).sum()+a[1], np.abs(b[0]).sum()+b[1], rtol=1e-6)


class Estimators(unittest.TestCase):
    def test_crossfit_pilot_ignores_its_own_fold(self):
        rng = np.random.default_rng(1)
        U, grid = advection()
        U = U+.05*rng.normal(size=U.shape)
        pilot = estimate(U, grid, ADVECTION, CrossFitPilot())[0]
        perturbed = U.copy()
        perturbed[:, ::2] += 10.
        self.assertTrue(np.allclose(estimate(perturbed, grid, ADVECTION, CrossFitPilot())[0][:, ::2], pilot[:, ::2]))

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
