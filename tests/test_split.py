"""Debiased methods split the time points: uhat from a fraction `ratio`, the regression on the rest."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pysindy as ps
from pdes import AUTHORS, PDE
from wsindy import Block, Debiased, SplitPilot, SplitSystem, Tests, estimate, noise_std, spatial


PDE_TEST = PDE("split-test", "", "", ("u",),
               (Block(ps.PolynomialLibrary(degree=2), spatial(1, 1)),), ("u_t",),
               ({"u_x": -1.},), Tests(m=(3, 3), p=(3, 3), s=(3, 2)),
               replace(AUTHORS, rescale=False))
RATIOS = (.5, .3, 2/3)


def observations(n=25, steps=41):
    x, t = np.linspace(-1, 1, n), np.linspace(0, 2, steps)
    u = np.sin(x[:, None]-t[None, :])+.2*x[:, None]**2+.1*np.cos(2*t[None, :])
    return u[..., None], (x, t)


def regression_times(uhat):
    return np.flatnonzero(np.isfinite(uhat[0, :, 0]))


class Split(unittest.TestCase):
    def test_training_times(self):
        U, grid = observations(steps=60)
        for ratio in RATIOS:
            training = np.isnan(estimate(U, grid, PDE_TEST, SplitPilot(ratio))[0][0, :, 0])
            self.assertLessEqual(abs(training.sum()-ratio*len(training)), 1)
            self.assertLessEqual(np.ptp(np.diff(np.flatnonzero(training))), 1)   # spread evenly
        np.testing.assert_array_equal(regression_times(estimate(U, grid, PDE_TEST, SplitPilot())[0]), np.arange(0, 60, 2))

    def test_estimate_and_scaling_ignore_regression_observations(self):
        U, grid = observations()
        U += .05*np.random.default_rng(2).normal(size=U.shape)
        pde = replace(PDE_TEST, profile=AUTHORS)
        for ratio in RATIOS:
            uhat, info = estimate(U, grid, pde, SplitPilot(ratio))
            changed = U.copy()
            changed[:, regression_times(uhat)] += np.random.default_rng(7).normal(0, 100, (U.shape[0], len(regression_times(uhat)), 1))
            new_uhat, new_info = estimate(changed, grid, pde, SplitPilot(ratio))
            np.testing.assert_array_equal(uhat, new_uhat)
            self.assertEqual(info, new_info)
            np.testing.assert_array_equal(SplitSystem(pde, U, grid, X=uhat).state_scales,
                                          SplitSystem(pde, changed, grid, X=new_uhat).state_scales)

    def test_training_values_do_not_enter_integrals(self):
        U, grid = observations()
        for ratio in RATIOS:
            uhat = estimate(U, grid, PDE_TEST, SplitPilot(ratio))[0]
            changed = U+10*np.isnan(uhat)
            old = Debiased().matrices(SplitSystem(PDE_TEST, U, grid, X=uhat))
            new = Debiased().matrices(SplitSystem(PDE_TEST, changed, grid, X=uhat))
            for a, b in zip(old, new):
                np.testing.assert_array_equal(a, b)

    def test_quadrature_matches_direct_physical_sums(self):
        U, grid = observations(n=19, steps=31)
        x, t = grid
        dx = x[1]-x[0]
        mx, mt = PDE_TEST.tests.m
        for ratio in RATIOS:
            uhat = estimate(U, grid, PDE_TEST, SplitPilot(ratio))[0]
            system = SplitSystem(PDE_TEST, U, grid, X=uhat)
            G, Y = Debiased().matrices(system)
            kept = regression_times(uhat)
            tk = t[kept]
            dt = np.r_[tk[1]-tk[0], (tk[2:]-tk[:-2])/2, tk[-1]-tk[-2]]   # trapezoid on the regression times
            column = [term.label for term in system.terms].index("(u^2)_x")
            corrected = 2*uhat[:, kept, 0]*U[:, kept, 0]-uhat[:, kept, 0]**2
            expected_x, expected_y = [], []
            for xc in system.centers[0]:
                rx = (x-xc)/(mx*dx)
                phi_x = np.maximum(1-rx**2, 0.)**3
                dphi_x = np.where(np.abs(rx) < 1, -6*rx*np.maximum(1-rx**2, 0.)**2/(mx*dx), 0.)
                for tc in system.centers[1]:
                    rt = (tk-tc)/(mt*(t[1]-t[0]))
                    phi_t = np.maximum(1-rt**2, 0.)**3
                    dphi_t = np.where(np.abs(rt) < 1, -6*rt*np.maximum(1-rt**2, 0.)**2/(mt*(t[1]-t[0])), 0.)
                    expected_x.append(-dx*np.sum(dphi_x[:, None]*(dt*phi_t)[None, :]*corrected))
                    expected_y.append(-dx*np.sum(phi_x[:, None]*(dt*dphi_t)[None, :]*U[:, kept, 0]))
            np.testing.assert_allclose(G[:, column], expected_x, rtol=1e-11, atol=2e-14)
            np.testing.assert_allclose(Y[:, 0], expected_y, rtol=1e-11, atol=2e-14)

    def test_matrices_are_affine_in_regression_noise_given_training_data(self):
        U, grid = observations()
        pde = replace(PDE_TEST, profile=AUTHORS)
        rng = np.random.default_rng(42)
        for ratio in RATIOS:
            uhat = estimate(U, grid, pde, SplitPilot(ratio))[0]
            deltas = [np.isfinite(uhat)*rng.normal(0, .1, U.shape) for _ in range(2)]
            def matrices(values):
                return Debiased().matrices(SplitSystem(pde, values, grid, X=uhat))
            base, a, b, both = [matrices(v) for v in (U, U+deltas[0], U+deltas[1], U+sum(deltas))]
            for j in (0, 1):
                np.testing.assert_allclose(both[j]-base[j], a[j]+b[j]-2*base[j], rtol=1e-10, atol=1e-11)

    def test_noise_level_on_unequal_times(self):
        times = np.flatnonzero((np.arange(400)+1)*3 % 10 < 3)   # ratio .3: gaps of 3 and 4
        quintic = np.polyval([1e-9, -2e-7, 3e-5, -1e-3, .02, 1.], times.astype(float))
        self.assertLess(noise_std(quintic, times), 1e-9*noise_std(quintic))
        noise = np.random.default_rng(0).normal(0, .3, (50, len(times)))
        self.assertAlmostEqual(noise_std(noise, times), .3, delta=.01)
        nearly_uniform = np.arange(1, 201, 2.)
        nearly_uniform[-1] += 1e-6   # divided differences, which reduce to the binomial ones
        np.testing.assert_allclose(noise_std(noise[:, :100], nearly_uniform), noise_std(noise[:, :100]), rtol=1e-6)

    def test_invalid_splits_are_rejected(self):
        U, grid = observations()
        with self.assertRaisesRegex(ValueError, "fixed Tests.m"):
            estimate(U, grid, replace(PDE_TEST, tests=replace(PDE_TEST.tests, m=None)), SplitPilot())
        with self.assertRaisesRegex(ValueError, "beyond the support"):
            estimate(*observations(steps=200), PDE_TEST, SplitPilot(.12))
        with self.assertRaisesRegex(ValueError, "NaN on whole training time slices"):
            SplitSystem(PDE_TEST, U, grid, X=U)
        with self.assertRaisesRegex(ValueError, "fixed Tests.m"):
            uhat = estimate(U, grid, PDE_TEST, SplitPilot())[0]
            SplitSystem(replace(PDE_TEST, tests=replace(PDE_TEST.tests, m=None)), U, grid, X=uhat)


class ComparisonReports(unittest.TestCase):
    def test_isolated_comparison_is_read_only_when_selected(self):
        from report import read
        record = dict(name="IB", method="wsindy", noise_ratio=0., trial=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main, comparison = root/"main", root/"comparison"
            main.mkdir(); comparison.mkdir()
            for folder in (main, comparison):
                (folder/"IB.jsonl").write_text(json.dumps(record)+"\n")
            (comparison/"report.scope.json").write_text(json.dumps(dict(include_when_scanning_parent=False)))
            self.assertEqual(len(read([root])[("IB", "wsindy")]), 1)
            self.assertEqual(len(read([comparison])[("IB", "wsindy")]), 1)
            with self.assertRaisesRegex(ValueError, "Duplicate trial"):
                read([root, comparison])

    def test_pooled_cross_fit_records_keep_their_own_name(self):
        from report import method_of
        self.assertEqual(method_of(dict(method="debiased", estimator=dict(split="time2"))), "crossfit-debiased")
        self.assertEqual(method_of(dict(method="debiased-wsindy", regression="lasso")), "crossfit-debiased-lasso")
        self.assertEqual(method_of(dict(method="split-debiased", estimator=dict(split="time2_single"))), "debiased")
        self.assertEqual(method_of(dict(method="debiased", estimator=dict(ratio=.5))), "debiased")


if __name__ == "__main__":
    unittest.main()
