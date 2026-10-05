"""Independent moment, weak-integral, and conic checks for scalar Gaussian EIV."""
import unittest

import numpy as np

from methods import LibraryTerm, build_wsindy_system, gaussian_polynomial
from eiv_study import conic_eiv, eiv_tolerances


class GaussianPolynomialTests(unittest.TestCase):
    def test_gaussian_quadrature_recovers_latent_monomial_moments(self):
        # Gauss-Hermite quadrature integrates these polynomial moments exactly;
        # the expected answer is the latent power, without sampling error.
        nodes, weights = np.polynomial.hermite_e.hermegauss(12)
        means = np.array([-1.3, -.2, 0., .4, 1.1])
        for variance in (.03, .7, 2.):
            observations = means[:, None] + np.sqrt(variance)*nodes
            for degree in range(9):
                with self.subTest(variance=variance, degree=degree):
                    corrected = gaussian_polynomial(observations, degree, variance)
                    expectation = corrected @ weights / np.sqrt(2*np.pi)
                    np.testing.assert_allclose(expectation, means**degree,
                                               rtol=2e-10, atol=2e-10)

    def test_zero_variance_returns_original_powers_without_mutating_data(self):
        values = np.array([[-2., -.3, 0.], [.2, .7, 1.5]])
        original = values.copy()
        for degree in range(9):
            with self.subTest(degree=degree):
                np.testing.assert_allclose(gaussian_polynomial(values, degree, 0.),
                                           values**degree, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(values, original)

    def test_invalid_degree_variance_and_observations_are_rejected(self):
        for degree in (-1, 1.5):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                gaussian_polynomial(np.array([1.]), degree, .1)
        for variance in (-.1, np.nan, np.inf):
            with self.subTest(variance=variance), self.assertRaises(ValueError):
                gaussian_polynomial(np.array([1.]), 2, variance)
        for value in (np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gaussian_polynomial(np.array([value]), 2, .1)


class GaussianWeakSystemTests(unittest.TestCase):
    def setUp(self):
        self.x, self.time = np.linspace(-1, 1, 65), np.linspace(0, .8, 49)
        self.values = 2.5*np.exp(.3*self.x[:, None] + .2*self.time)
        self.options = dict(half_widths=(22, 16), strides=(5, 4),
                            test_degrees=(12, 12), rescale=False)

    def test_correction_matches_lower_power_weak_integrals_in_physical_units(self):
        variance = .36
        terms = tuple(LibraryTerm((degree,), (0, 0)) for degree in range(7)) + (
            LibraryTerm((1,), (1, 0)), LibraryTerm((3,), (1, 0)),
            LibraryTerm((5,), (1, 0)), LibraryTerm((1,), (2, 0)),
            LibraryTerm((3,), (2, 0)), LibraryTerm((5,), (2, 0)))
        raw = build_wsindy_system(self.values, (self.x,), self.time,
                                  library_terms=terms, **self.options)
        corrected = build_wsindy_system(self.values, (self.x,), self.time,
                                        library_terms=terms,
                                        gaussian_variance=variance, **self.options)
        # Closed forms supply an independent reference for the recurrence and
        # check that correction precedes spatial weak differentiation.
        expected = raw.G.copy()
        expected[:, 2] = raw.G[:, 2] - variance*raw.G[:, 0]
        expected[:, 3] = raw.G[:, 3] - 3*variance*raw.G[:, 1]
        expected[:, 4] = raw.G[:, 4] - 6*variance*raw.G[:, 2] + 3*variance**2*raw.G[:, 0]
        expected[:, 5] = raw.G[:, 5] - 10*variance*raw.G[:, 3] + 15*variance**2*raw.G[:, 1]
        expected[:, 6] = (raw.G[:, 6] - 15*variance*raw.G[:, 4]
                          + 45*variance**2*raw.G[:, 2] - 15*variance**3*raw.G[:, 0])
        for first, third, fifth in ((7, 8, 9), (10, 11, 12)):
            expected[:, third] = raw.G[:, third] - 3*variance*raw.G[:, first]
            expected[:, fifth] = (raw.G[:, fifth] - 10*variance*raw.G[:, third]
                                  + 15*variance**2*raw.G[:, first])
        np.testing.assert_allclose(corrected.G, expected, rtol=2e-11, atol=2e-12)
        np.testing.assert_array_equal(corrected.b, raw.b)
        np.testing.assert_array_equal(corrected.state_scales, 1.)
        np.testing.assert_array_equal(corrected.coordinate_scales, 1.)
        np.testing.assert_array_equal(corrected.coefficient_scales, 1.)
        self.assertEqual(corrected.gaussian_variance, variance)
        self.assertIsNone(raw.gaussian_variance)
        self.assertFalse(corrected.pilot)

    def test_zero_variance_weak_system_matches_raw_system(self):
        terms = (LibraryTerm((0,), (0, 0)), LibraryTerm((3,), (2, 0)))
        raw = build_wsindy_system(self.values, (self.x,), self.time,
                                  library_terms=terms, **self.options)
        corrected = build_wsindy_system(self.values, (self.x,), self.time,
                                        library_terms=terms, gaussian_variance=0.,
                                        **self.options)
        np.testing.assert_allclose(corrected.G, raw.G, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(corrected.b, raw.b)
        self.assertEqual(corrected.gaussian_variance, 0.)

    def test_unsupported_corrections_are_rejected(self):
        base = dict(self.options, library_terms=(LibraryTerm((2,), (0, 0)),),
                    gaussian_variance=.1)
        for change in ({"rescale": True}, {"pilot": self.values},
                       {"gaussian_variance": -.1}, {"gaussian_variance": np.nan},
                       {"gaussian_variance": np.inf},
                       {"library_terms": (LibraryTerm((0,), (0, 0), "sin"),)}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                build_wsindy_system(self.values, (self.x,), self.time, **(base | change))
        with self.assertRaises(ValueError):
            build_wsindy_system((self.values, self.values), (self.x,), self.time,
                                **(base | {"library_terms": (LibraryTerm((1, 1), (0, 0)),)}))


class EIVToleranceTests(unittest.TestCase):
    def setUp(self):
        self.x, self.time = np.linspace(-2, 2, 17), np.linspace(0, 2, 17)
        self.values = 5+3*self.x[:, None]+4*self.time
        self.terms = (LibraryTerm((1,), (0, 0)), LibraryTerm((1,), (1, 0)),
                      LibraryTerm((2,), (1, 0)))
        self.system = build_wsindy_system(self.values, (self.x,), self.time,
            library_terms=self.terms, half_widths=(4, 4), strides=(2, 2),
            test_degrees=(2, 2), rescale=False)

    def test_g2_is_operator_norm_of_physical_quadrature_kernels(self):
        result = eiv_tolerances(self.values, (self.x,), self.time,
                                system=self.system, sigma2=.1)
        r = np.linspace(-1., 1., 9)
        phi, dphi = (1-r*r)**2, -4*r*(1-r*r)
        dx, dt = .25, .125
        radius_x, radius_t = 4*dx, 4*dt
        constant_kernel = dx*dt*np.outer(phi, phi)
        space_kernel = dx*dt/radius_x*np.outer(dphi, phi)
        time_kernel = dx*dt/radius_t*np.outer(phi, dphi)
        # Even/odd spatial parity makes the two feature kernels orthogonal,
        # so their singular values are their separate Euclidean norms.
        feature_norms = np.array([np.linalg.norm(constant_kernel),
                                  np.linalg.norm(space_kernel)])
        self.assertAlmostEqual(result["g2"], feature_norms.max(), places=14)
        self.assertLess(result["g2"], np.linalg.norm(feature_norms))
        self.assertAlmostEqual(result["g0"], np.linalg.norm(time_kernel), places=14)
        self.assertEqual(result["S"], 2)  # Repeated derivative does not add a kernel.
        self.assertFalse(result["confidence_claim"])

    def test_gradient_proxy_includes_time_and_extrema_include_boundary_limits(self):
        result = eiv_tolerances(self.values, (self.x,), self.time,
                                system=self.system, sigma2=.1)
        self.assertAlmostEqual(result["observed_M1"], 5., places=13)
        self.assertAlmostEqual(result["M1"], 5., places=13)
        # For phi=(1-r^2)^2, max|phi'|=8/(3*sqrt(3)) and
        # max|phi''|=8 occurs at the one-sided endpoint limits. Replacing
        # endpoint derivatives by the sampled weak weights' zeros is wrong.
        first = 8/(3*np.sqrt(3))
        expected_gradient = np.sqrt(8**2 + (first**2/.5)**2)
        self.assertAlmostEqual(result["term_norms"][1]["gradient_sup_bound"],
                               expected_gradient, places=12)
        supplied = eiv_tolerances(self.values, (self.x,), self.time,
                                  system=self.system, sigma2=.1, M=20., M1=8.)
        self.assertEqual((supplied["M"], supplied["M1"]), (20., 8.))
        self.assertEqual((supplied["M_source"], supplied["M1_source"]),
                         ("supplied_bound", "supplied_bound"))


class ConicEIVTests(unittest.TestCase):
    def assert_feasible(self, result, X, y, *, mu, tau, noise_bias=None, atol=2e-7):
        self.assertTrue(result.success, result.diagnostics)
        self.assertTrue(result.status)
        self.assertEqual(result.coefficients.shape, (X.shape[1],))
        self.assertEqual(result.support.shape, result.coefficients.shape)
        self.assertEqual(result.support.dtype, np.dtype(bool))
        corrected = X if noise_bias is None else X-noise_bias
        residual = np.linalg.norm(y-corrected@result.coefficients, ord=np.inf)
        self.assertLessEqual(residual, tau+mu*result.t+atol)
        self.assertLessEqual(np.linalg.norm(result.coefficients), result.t+atol)

    def test_duplicate_columns_choose_minimum_norm_split(self):
        X, y = np.array([[1., 1.]]), np.array([1.])
        result = conic_eiv(X, y, mu=0., tau=0., lam=1.)
        self.assert_feasible(result, X, y, mu=0., tau=0.)
        np.testing.assert_allclose(result.coefficients, [.5, .5], atol=2e-7)
        self.assertAlmostEqual(result.t, 1/np.sqrt(2), delta=2e-7)
        self.assertAlmostEqual(result.objective, 1+1/np.sqrt(2), delta=4e-7)
        np.testing.assert_array_equal(result.support, [True, True])

    def test_residual_uncertainty_can_require_t_strictly_above_coefficient_norm(self):
        X, y = np.zeros((1, 2)), np.array([2.])
        result = conic_eiv(X, y, mu=1., tau=0., lam=1.)
        self.assert_feasible(result, X, y, mu=1., tau=0.)
        np.testing.assert_allclose(result.coefficients, 0., atol=2e-7)
        self.assertAlmostEqual(result.t, 2., delta=2e-7)
        self.assertAlmostEqual(result.objective, 2., delta=4e-7)
        np.testing.assert_array_equal(result.support, [False, False])

    def test_noise_bias_is_subtracted_and_input_arrays_remain_unchanged(self):
        X, bias, y = np.array([[3.], [6.]]), np.array([[1.], [2.]]), np.array([4., 8.])
        originals = [array.copy() for array in (X, bias, y)]
        result = conic_eiv(X, y, noise_bias=bias, mu=0., tau=0., lam=.5)
        self.assert_feasible(result, X, y, noise_bias=bias, mu=0., tau=0.)
        np.testing.assert_allclose(result.coefficients, [2.], atol=2e-7)
        self.assertAlmostEqual(result.objective, 3., delta=4e-7)
        for array, original in zip((X, bias, y), originals):
            np.testing.assert_array_equal(array, original)

    def test_solver_preserves_physical_units_and_support_does_not_threshold_coefficients(self):
        X, y = np.diag([1., 4.]), np.array([1., .04])
        for equilibrate in (False, True):
            with self.subTest(equilibrate=equilibrate):
                result = conic_eiv(X, y, mu=0., tau=0., lam=.7,
                                  support_tolerance=.1, equilibrate=equilibrate)
                self.assert_feasible(result, X, y, mu=0., tau=0.)
                np.testing.assert_allclose(result.coefficients, [1., .01], atol=2e-7)
                np.testing.assert_array_equal(result.support, [True, False])
                expected = 1.01+.7*np.sqrt(1+.01**2)
                self.assertAlmostEqual(result.objective, expected, delta=4e-7)

    def test_feasible_origin_is_exact_global_zero_optimum(self):
        X, y = np.array([[3., 4.], [-1., 2.]]), np.array([.2, -.5])
        result = conic_eiv(X, y, mu=.8, tau=.5, lam=.2)
        self.assert_feasible(result, X, y, mu=.8, tau=.5)
        np.testing.assert_array_equal(result.coefficients, [0., 0.])
        np.testing.assert_array_equal(result.support, [False, False])
        self.assertEqual(result.t, 0.)
        self.assertEqual(result.objective, 0.)

    def test_infeasible_problem_is_reported_as_failure(self):
        result = conic_eiv(np.zeros((1, 1)), np.array([1.]), mu=0., tau=0., lam=1.)
        self.assertFalse(result.success)
        self.assertTrue(result.status)
        self.assertTrue(result.diagnostics)

    def test_invalid_arrays_and_parameters_are_rejected(self):
        X, y = np.eye(2), np.array([1., 2.])
        options = dict(mu=.1, tau=.2, lam=1.)
        for bad_X, bad_y in ((np.array([1., 2.]), y), (np.zeros((0, 2)), np.zeros(0)),
                             (np.zeros((2, 0)), y), (X, np.ones(3)),
                             (X, np.ones((2, 2))), (np.full((2, 2), np.nan), y),
                             (X, np.array([1., np.inf]))):
            with self.subTest(X_shape=bad_X.shape, y_shape=bad_y.shape), self.assertRaises(ValueError):
                conic_eiv(bad_X, bad_y, **options)
        for change in ({"mu": -.1}, {"tau": -.1}, {"lam": 0.}, {"lam": -1.},
                       {"mu": np.nan}, {"tau": np.inf}, {"lam": np.nan},
                       {"solver_tolerance": 0.}, {"support_tolerance": -.1},
                       {"max_iter": 0}, {"max_iter": 1.5},
                       {"noise_bias": np.zeros((1, 2))},
                       {"noise_bias": np.full((2, 2), np.inf)}):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                conic_eiv(X, y, **(options | change))


if __name__ == "__main__":
    unittest.main()
