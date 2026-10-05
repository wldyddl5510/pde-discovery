"""Checks for the moving-average pilot, the debiased weak system, and the study driver."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from scipy.ndimage import uniform_filter
from methods import LibraryTerm, build_wsindy_system, moving_average_pilot, polynomial_library
import debiasing_study
from simulation_generation import DEFAULT_DATA_DIR


class PilotTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.u = rng.normal(size=(12, 9))

    def test_no_split_matches_uniform_filter_with_per_axis_boundaries(self):
        pilot, info = moving_average_pilot(self.u, (5, 3), boundary=("wrap", "reflect"))
        np.testing.assert_allclose(pilot, uniform_filter(self.u, size=(5, 3), mode=["wrap", "reflect"]), atol=1e-14)
        self.assertEqual((info["window_points"], info["folds"], info["split"]), (15, 1, None))
        pair, _ = moving_average_pilot((self.u, 2*self.u), (5, 3), boundary="wrap")
        np.testing.assert_allclose(pair[1], 2*pair[0], atol=1e-14)

    def test_time_split_averages_the_other_parity_only(self):
        pilot, info = moving_average_pilot(self.u, (3, 5), boundary="wrap", split=("axis", -1, 2))
        i, j = 6, 4
        rows, columns = np.arange(i-1, i+2), np.arange(j-2, j+3)
        other = columns[columns % 2 != j % 2]
        np.testing.assert_allclose(pilot[i, j], self.u[np.ix_(rows, other)].mean(), atol=1e-14)
        self.assertEqual(info["split"], ("axis", 1, 2))
        self.assertAlmostEqual(info["minimum_window_fraction"], 2/5)
        # Reflection (d c b a | a b c d) at the time boundary: mirrored copies
        # keep their fold, so own-parity copies stay excluded and other-parity
        # copies are counted again.
        pilot, _ = moving_average_pilot(self.u, (1, 5), boundary=("wrap", "reflect"), split=("axis", 1, 2))
        np.testing.assert_allclose(pilot[i, 1], (2*self.u[i, 0]+self.u[i, 2])/3, atol=1e-14)
        np.testing.assert_allclose(pilot[i, 0], self.u[i, 1], atol=1e-14)

    def test_checkerboard_split_and_invalid_settings(self):
        pilot, info = moving_average_pilot(self.u, (3, 3), boundary="wrap", split="checkerboard")
        i, j = 5, 4
        block = self.u[i-1:i+2, j-1:j+2]
        parity = (np.add.outer(np.arange(i-1, i+2), np.arange(j-1, j+2)) % 2) != (i+j) % 2
        np.testing.assert_allclose(pilot[i, j], block[parity].mean(), atol=1e-14)
        self.assertEqual(info["folds"], 2)
        for settings in (dict(sizes=(4, 3)), dict(sizes=(3,)), dict(sizes=(13, 3)), dict(sizes=(3, 3), boundary="nearest"),
                         dict(sizes=(3, 1), split=("axis", 1, 2)), dict(sizes=(3, 3), split="rows"),
                         dict(sizes=(3, 3), split=("axis", 1, 1))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                moving_average_pilot(self.u, **settings)


class DebiasedSystemTests(unittest.TestCase):
    def setUp(self):
        self.x, self.t = np.linspace(-1, 1, 81), np.linspace(0, 0.5, 61)
        self.u = np.exp(0.6*self.x[:, None]+0.4*self.t)
        self.delta = 0.1*np.sin(3*self.x[:, None])*np.cos(2*self.t)
        self.options = dict(half_widths=(25, 20), strides=(6, 5), test_degrees=(12, 12), rescale=False)

    def test_pilot_equal_to_data_reproduces_the_raw_system(self):
        terms = polynomial_library(1, 1, 4, 3)
        raw = build_wsindy_system(self.u, (self.x,), self.t, library_terms=terms, **self.options)
        same = build_wsindy_system(self.u, (self.x,), self.t, library_terms=terms, pilot=self.u, **self.options)
        np.testing.assert_allclose(same.G, raw.G, rtol=1e-13, atol=1e-15)
        np.testing.assert_array_equal(same.b, raw.b)
        self.assertTrue(same.pilot and not raw.pilot)

    def test_columns_are_first_order_expansions_at_the_pilot(self):
        terms = polynomial_library(1, 1, 4, 3)
        pilot = self.u+self.delta
        raw = build_wsindy_system(self.u, (self.x,), self.t, library_terms=terms, **self.options)
        debiased = build_wsindy_system(self.u, (self.x,), self.t, library_terms=terms, pilot=pilot, **self.options)
        np.testing.assert_array_equal(debiased.b, raw.b)
        for index, term in enumerate(terms):
            p = term.powers[0]
            if p < 2:
                np.testing.assert_array_equal(debiased.G[:, index], raw.G[:, index])
                continue
            linearized = (1-p)*pilot**p+p*pilot**(p-1)*self.u
            reference = build_wsindy_system(linearized, (self.x,), self.t,
                library_terms=(LibraryTerm((1,), term.derivative),), **self.options)
            np.testing.assert_allclose(debiased.G[:, index], reference.G[:, 0], rtol=1e-12, atol=1e-15)
        # Second-order identity for squares: X_db - X(u_*) = -X(delta^2) when U = u_*.
        square = terms.index(LibraryTerm((2,), (1, 0)))
        remainder = build_wsindy_system(self.delta**2, (self.x,), self.t,
            library_terms=(LibraryTerm((1,), (1, 0)),), **self.options)
        np.testing.assert_allclose(debiased.G[:, square]-raw.G[:, square], -remainder.G[:, 0], rtol=1e-10, atol=1e-14)

    def test_coupled_monomials_and_rescaling_follow_the_same_expansion(self):
        v = np.exp(-0.3*self.x[:, None]+0.2*self.t)
        pilots = (self.u+self.delta, v-0.5*self.delta)
        terms = (LibraryTerm((2, 1), (1, 0)), LibraryTerm((1, 3), (0, 0)), LibraryTerm((0, 1), (1, 0)))
        options = dict(self.options, lhs_components=(0, 1))
        debiased = build_wsindy_system((self.u, v), (self.x,), self.t, library_terms=terms, pilot=pilots, **options)
        for index, term in enumerate(terms):
            a, b = term.powers
            pu, pv = pilots
            linearized = (1-a-b)*pu**a*pv**b+a*pu**(a-1)*pv**b*self.u+b*pu**a*pv**(b-1)*v if a+b > 1 else self.u**a*v**b
            reference = build_wsindy_system((linearized, v), (self.x,), self.t,
                library_terms=(LibraryTerm((1, 0), term.derivative),), **options)
            np.testing.assert_allclose(debiased.G[:, index], reference.G[:, 0], rtol=1e-12, atol=1e-15)
        # Rescaling commutes with the expansion: the usual scaling identity holds.
        scaled = build_wsindy_system((self.u, v), (self.x,), self.t, library_terms=terms, pilot=pilots,
                                     **dict(options, rescale=True, lhs_components=(0,)))
        unscaled = build_wsindy_system((self.u, v), (self.x,), self.t, library_terms=terms, pilot=pilots,
                                       **dict(options, lhs_components=(0,)))
        volume = np.prod(scaled.coordinate_scales)
        lhs_factor = scaled.state_scales[0]/scaled.coordinate_scales[-1]
        np.testing.assert_allclose(scaled.b, unscaled.b*volume*lhs_factor, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(scaled.G, unscaled.G*volume*lhs_factor*scaled.coefficient_scales[:, 0], rtol=1e-10, atol=1e-13)
        with self.assertRaisesRegex(ValueError, "polynomial"):
            build_wsindy_system(self.u, (self.x,), self.t, pilot=self.u+self.delta,
                library_terms=(LibraryTerm((0,), (1, 0), "sin"),), **self.options)
        with self.assertRaisesRegex(ValueError, "pilot"):
            build_wsindy_system(self.u, (self.x,), self.t, pilot=(self.u, v), library_terms=terms[:1], **self.options)


class StudyTests(unittest.TestCase):
    def test_time2_window_selection_excludes_evaluation_fold(self):
        rng = np.random.default_rng(4)
        values = (rng.normal(scale=.01, size=(20, 31)), rng.normal(scale=.02, size=(20, 31)))
        spec = SimpleNamespace(shape=(20, 31), half_widths=(6, 7), max_degree=3)
        pilot, info = debiasing_study.time2_pilot(values, spec)
        for parity in (0, 1):
            changed = tuple(v.copy() for v in values)
            for v in changed:
                v[..., parity::2] += rng.normal(scale=1000, size=v[..., parity::2].shape)
            new_pilot, new_info = debiasing_study.time2_pilot(changed, spec)
            self.assertEqual(info["folds"][parity], new_info["folds"][parity])
            for p, new_p in zip(pilot, new_pilot):
                np.testing.assert_array_equal(p[..., parity::2], new_p[..., parity::2])
            reference, _ = moving_average_pilot(values, info["folds"][parity]["window"],
                boundary=("wrap", "reflect"), split=("axis", -1, 2))
            for p, ref in zip(pilot, reference):
                np.testing.assert_allclose(p[..., parity::2], ref[..., parity::2], atol=1e-14)

    def test_coupled_sparse_fits_preserve_both_equations_and_units(self):
        system = SimpleNamespace(G=np.diag([2., 3.]), b=np.array([[8., 0.], [0., -6.]]),
                                 coefficient_scales=np.array([[.25, 4.], [2., .5]]))
        truth = np.array([[1., 0.], [0., -1.]])
        select = dict(thresholds=[.1], threshold_units="physical", selection_rule="absolute", allow_empty=False)
        fits = debiasing_study.fit_system(system, slice(None), [], truth, select, ["mstls", "lasso"])
        np.testing.assert_allclose(fits["mstls"]["coefficients"], truth)
        np.testing.assert_allclose(fits["lasso"]["coefficients"], .9999*truth, atol=1e-12)
        self.assertEqual(len(fits["lasso"]["selected_alpha"]), 2)

    def test_fits_restore_physical_coefficients_of_rescaled_system(self):
        system = SimpleNamespace(G=np.diag([2., 3.]), b=np.array([[8.], [0.]]),
                                 coefficient_scales=np.array([[.25], [2.]]))
        truth = np.array([1., 0.])
        select = dict(thresholds=[.1], threshold_units="physical",
                      selection_rule="absolute", allow_empty=False)
        fits = debiasing_study.fit_system(system, slice(None), [0], truth,
                                         select, ["ols", "mstls", "lasso"])
        np.testing.assert_allclose(fits["ols"]["coefficients"], truth)
        np.testing.assert_allclose(fits["mstls"]["coefficients"], truth)
        np.testing.assert_allclose(fits["lasso"]["coefficients"], [.9999, 0.], atol=1e-12)
        self.assertAlmostEqual(fits["lasso"]["selected_alpha"][0], 1e-4)
        for fit in fits.values():
            self.assertGreaterEqual(fit["regression_runtime"], 0)

    def test_noise_generated_terms_and_rule_size(self):
        spec = debiasing_study.BENCHMARKS["VBG"]
        terms = spec.library()
        generated = debiasing_study.noise_generated_terms(spec, terms)
        # -u^3 adds 3 sigma^2 u to its column, so restricted OLS fits +3 sigma^2 u;
        # 2u^2 adds 2 sigma^2 to the constant column, fitted as -2 sigma^2.
        self.assertEqual(generated[LibraryTerm((1,), (0, 0))], [(2, 3.)])
        self.assertEqual(generated[LibraryTerm((0,), (0, 0))], [(2, -2.)])
        self.assertNotIn(LibraryTerm((0,), (1, 0)), generated)
        self.assertEqual(debiasing_study.rule_size(0., spec, .01), 1)
        self.assertEqual(debiasing_study.rule_size(10., spec, .01) % 2, 1)
        rows = debiasing_study.time_boundary_rows(spec, 32)
        self.assertEqual(len(rows), 32*31)
        self.assertLess(np.count_nonzero(rows), len(rows))

    def test_bad_arguments_are_rejected(self):
        for flags in (["--pilot-sizes", "4"], ["--trials", "0"], ["--sigma-multipliers", "1", "1"],
                      ["--benchmark", "NLS"], ["--benchmark", "KS"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                debiasing_study.parse_arguments(flags)

    @unittest.skipUnless((DEFAULT_DATA_DIR/"viscous_burgers_growth.npz").exists(), "VBG data are not cached")
    def test_small_study_runs_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"study.md"
            args = debiasing_study.parse_arguments(["--trials", "1", "--sigma-multipliers", "2", "--pilot-sizes", "5",
                "--no-rule-size", "--splits", "none", "time2", "--regressions", "ols", "mstls", "--offline",
                "--output", str(output)])
            with contextlib.redirect_stderr(io.StringIO()):
                records = debiasing_study.run(args)
            self.assertEqual([(r["method"], r["split"]) for r in records],
                             [("raw", None), ("filtered", "none"), ("debiased", "none"), ("filtered", "time2"), ("debiased", "time2")])
            terms = debiasing_study.BENCHMARKS["VBG"].library()
            linear = [i for i, t in enumerate(terms) if sum(t.powers) <= 1]
            raw, debiased = records[0]["fits"]["ols"]["coefficients"], records[2]["fits"]["ols"]["coefficients"]
            self.assertNotEqual(raw, debiased)
            report = debiasing_study.format_report(args, records)
            self.assertIn("## Restricted OLS", report)
            self.assertIn("β[u] − 0 (raw pred. +0.04)", report)
            self.assertIn("(smallest true)", report)
            self.assertIn("## Full-library MSTLS", report)
            self.assertIn("| debiased | time2 | 5 |", report)


if __name__ == "__main__":
    unittest.main()
