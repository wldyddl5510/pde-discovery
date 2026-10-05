"""Independent checks of the paper's weak integrals and sparse selection."""
import unittest
import numpy as np
from methods import (LibraryTerm, build_wsindy_system, lasso, mstls, polynomial_library,
                     bump_function_weights, select_test_supports, test_function_weights, wsindy)


class WeakIntegralTests(unittest.TestCase):
    def test_bump_derivatives_and_eighth_order_integral_against_quadrature(self):
        table = bump_function_weights(200, 8)
        r = np.linspace(-1, 1, 401)[1:-1]
        d, phi = 1-r*r, np.exp(-9/(1-r*r))
        np.testing.assert_allclose(table[1, 1:-1], -18*r*phi/d**2, atol=1e-15)
        np.testing.assert_allclose(table[2, 1:-1], phi*((-18*r/d**2)**2-18/d**2-72*r*r/d**3), atol=1e-14)
        self.assertTrue(np.all(table[:, (0, -1)] == 0))
        x, t = np.linspace(-1, 1, 513), np.linspace(0, 1, 129)
        u = np.exp(1.3*x[:, None]+.7*t)
        terms = tuple(LibraryTerm((1,), (q, 0)) for q in range(9))
        system = build_wsindy_system(u, (x,), t, library_terms=terms,
            half_widths=(200, 50), strides=(20, 10), rescale=False, test_function="bump")
        nodes, weights = np.polynomial.legendre.leggauss(128)
        integrals = []
        for grid, rate, m, centers in zip((x, t), (1.3, .7), (200, 50), system.centers):
            radius = m*(grid[1]-grid[0])
            integrals.append(radius*(np.exp(-9/(1-nodes**2))*np.exp(rate*(centers[:, None]+radius*nodes)))@weights)
        reference = (integrals[0][:, None]*integrals[1]).ravel()
        np.testing.assert_allclose(system.G, np.column_stack([1.3**q*reference for q in range(9)]), rtol=2e-5, atol=1e-13)
        np.testing.assert_allclose(system.b[:, 0], .7*reference, rtol=1e-10, atol=1e-15)

    def test_all_columns_against_continuous_quadrature(self):
        x, t = np.linspace(-1, 1, 161), np.linspace(0, 0.6, 121)
        a, b, c, d = 0.8, -0.6, 0.7, -0.4
        u, v = np.exp(a*x[:, None]+c*t), np.exp(b*x[:, None]+d*t)
        terms = polynomial_library(2, 1, 3, 6)
        m, s, p = (70, 50), (8, 8), (16, 8)
        system = build_wsindy_system((u, v), (x,), t, library_terms=terms,
            lhs_components=(0, 1), half_widths=m, strides=s, test_degrees=p, rescale=False)
        nodes, weights = np.polynomial.legendre.leggauss(64)
        def integral(xrate, trate):
            values = []
            for grid, rate, radius_points, stride, degree in zip((x, t), (xrate, trate), m, s, p):
                radius = radius_points*(grid[1]-grid[0])
                centers = grid[radius_points:len(grid)-radius_points:stride]
                locations = centers[:, None]+radius*nodes
                values.append(radius*((1-nodes**2)**degree*np.exp(rate*locations))@weights)
            return (values[0][:, None]*values[1]).ravel()
        expected = []
        for term in terms:
            xrate = term.powers[0]*a+term.powers[1]*b
            trate = term.powers[0]*c+term.powers[1]*d
            expected.append(xrate**term.derivative[0]*integral(xrate, trate))
        np.testing.assert_allclose(system.G, np.column_stack(expected), rtol=1e-6, atol=2e-10)
        np.testing.assert_allclose(system.b, np.column_stack((c*integral(a, c), d*integral(b, d))), rtol=1e-8, atol=1e-11)

    def test_scaling_identity_for_monomials_trigonometry_and_second_time_derivative(self):
        x, t = np.linspace(-1, 1, 65), np.linspace(0, 1, 49)
        u = 20*np.exp(0.2*x[:, None]+0.1*t)
        terms = polynomial_library(1, 1, 4, 2, trigonometric_frequencies=(1, 2))
        options = dict(library_terms=terms, lhs_time_order=2, half_widths=(24, 18),
                       strides=(5, 4), test_degrees=(14, 12))
        original = build_wsindy_system(u, (x,), t, rescale=False, **options)
        scaled = build_wsindy_system(u, (x,), t, rescale=True, **options)
        volume = np.prod(scaled.coordinate_scales)
        lhs_factor = scaled.state_scales[0]/scaled.coordinate_scales[-1]**2
        np.testing.assert_allclose(scaled.b, original.b*volume*lhs_factor, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(scaled.G, original.G*volume*lhs_factor*scaled.coefficient_scales[:, 0], rtol=2e-8, atol=2e-9)
        gamma = (np.linalg.norm(u)/np.linalg.norm(u**4))**0.25
        self.assertAlmostEqual(scaled.state_scales[0], gamma, places=14)
        authors = build_wsindy_system(u, (x,), t, state_scale_rule="authors", **options)
        self.assertAlmostEqual(authors.state_scales[0], gamma**(4/3), places=14)

    def test_second_time_derivative_and_mixed_spatial_term(self):
        x, y, t = np.linspace(-1, 1, 49), np.linspace(-2, 2, 53), np.linspace(0, 1, 81)
        u = np.exp(0.3*x[:, None, None]-0.4*y[None, :, None]+0.7*t)
        terms = (LibraryTerm((1,), (0, 0, 0)), LibraryTerm((1,), (1, 1, 0)))
        system = build_wsindy_system(u, (x, y), t, library_terms=terms, lhs_time_order=2,
            half_widths=(18, 20, 30), strides=(6, 6, 10), test_degrees=(14, 14, 12), rescale=False)
        np.testing.assert_allclose(system.G[:, 1], -0.12*system.G[:, 0], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(system.b[:, 0], 0.49*system.G[:, 0], rtol=1e-8, atol=1e-10)

    def test_analytic_test_derivatives_and_decay(self):
        table, p = test_function_weights(8, 4)
        r = np.linspace(-1, 1, 17)
        self.assertLessEqual((1-(1-1/8)**2)**p, 1e-10)
        self.assertGreater((1-(1-1/8)**2)**(p-1), 1e-10)
        np.testing.assert_allclose(table[1], -2*p*r*(1-r**2)**(p-1), atol=1e-13)
        np.testing.assert_allclose(table[2], -2*p*(1-r**2)**(p-1)+4*p*(p-1)*r**2*(1-r**2)**(p-2), atol=1e-12)
        self.assertTrue(np.all(table[:, (0, -1)] == 0))

    def test_invalid_grids_terms_and_test_support(self):
        x, t = np.linspace(0, 1, 17), np.linspace(0, 1, 9)
        u = np.ones((17, 9))
        base = dict(library_terms=polynomial_library(1, 1, 1, 2), half_widths=(4, 2), strides=(1, 1))
        for change in ({"half_widths": (9, 2)}, {"strides": (0, 1)}, {"test_degrees": (2, 4)},
                       {"library_terms": (LibraryTerm((1.5,), (1, 0)),)}, {"lhs_components": (1,)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_wsindy_system(u, (x,), t, **(base | change))
        irregular = x.copy()
        irregular[3] += 0.01
        with self.assertRaises(ValueError):
            build_wsindy_system(u, (irregular,), t, **base)

    def test_spectral_support_selection_is_deterministic_and_fits_grid(self):
        x, t = np.linspace(0, 2*np.pi, 64), np.linspace(0, 1, 64)
        # An exactly piecewise-constant Fourier amplitude gives a known
        # cumulative-spectrum corner, independent of a PDE or random noise.
        waves = np.fft.fftfreq(64)*64
        amplitudes = np.where(np.abs(waves) <= 8, 0.5, 0.01)
        profile = np.fft.ifft(amplitudes).real
        u = profile[:, None]*profile[None, :]
        widths, corners = select_test_supports(u, (x,), t)
        self.assertEqual(corners, ((9, 9),))
        self.assertEqual((widths, corners), select_test_supports(u, (x,), t))
        for m, size in zip(widths, u.shape):
            self.assertGreaterEqual(m, 2)
            self.assertLessEqual(2*m+1, size)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            select_test_supports(np.zeros_like(u), (x,), t)


class SparseSelectionTests(unittest.TestCase):
    def test_lasso_matches_independent_soft_threshold_and_retains_shrinkage(self):
        G = np.diag([1., 4., 20.])
        truth = np.array([2., -.075, .0005])
        result = lasso(G, G@truth, [.1])
        # Orthogonal columns: shrink physical coefficients by alpha-scaled
        # prediction amplitude / original column amplitude.
        np.testing.assert_allclose(result.coefficients[:, 0], [1.8, -.025, 0], atol=1e-14)
        self.assertLess(result.kkt_errors.max(), 1e-12)
        self.assertEqual(result.coefficients[0, 0], result.scaled_coefficients[0, 0])

    def test_lasso_is_invariant_to_diagonal_preconditioning_and_handles_zero_response(self):
        rng = np.random.default_rng(21)
        G = rng.normal(size=(70, 8)); G[:, -1] = 0
        b = np.column_stack((G[:, 0]+.2*G[:, 2], np.zeros(70)))
        scale = np.logspace(-4, 4, 8)
        original = lasso(G, b)
        changed = lasso(G*scale, b, coefficient_scales=np.broadcast_to(scale[:, None], (8, 2)))
        np.testing.assert_allclose(original.coefficients, changed.coefficients, atol=1e-10)
        np.testing.assert_allclose(original.losses, changed.losses, atol=1e-10)
        np.testing.assert_array_equal(original.coefficients[:, 1], 0)
        self.assertEqual(original.coefficients[-1, 0], 0)

    def test_lasso_correlated_and_duplicate_columns_satisfy_kkt(self):
        rng = np.random.default_rng(12)
        G = rng.normal(size=(100, 12))
        G[:, 1] = G[:, 0]; G[:, 3] = G[:, 2]+1e-5*rng.normal(size=100)
        b = G[:, 0]-.2*G[:, 2]+.01*rng.normal(size=100)
        result = lasso(G, b)
        self.assertLessEqual(result.kkt_errors.max(), 1e-7)
        theta = result.normalized_coefficients[:, 0]
        Z = G/result.column_rms; y = b/result.response_rms[0]
        alpha = result.threshold*result.alpha_max[0]
        gradient = Z.T@(Z@theta-y)/len(y)
        violation = np.where(theta != 0, np.abs(gradient+alpha*np.sign(theta)),
            np.maximum(np.abs(gradient)-alpha, 0))
        self.assertLess(violation.max()/result.alpha_max[0], 1e-7)

    def test_consistency_absolute_threshold_ignores_relative_column_bounds(self):
        G = np.diag([1., 1e6])
        truth = np.array([.2, .03])
        result = mstls(G, G@truth, [.01], selection_rule="absolute")
        np.testing.assert_allclose(result.coefficients[:, 0], truth, atol=1e-14)
        relative = mstls(G, G@truth, [.01])
        self.assertEqual(relative.coefficients[0, 0], 0)
        np.testing.assert_array_equal(mstls(G, G@truth, [1], selection_rule="absolute").coefficients, np.zeros((2, 1)))

    def test_physical_selection_is_invariant_to_diagonal_preconditioning(self):
        G, b = np.diag([1., 10.]), np.array([0.2, 0.3])
        scales = np.array([[10.], [0.01]])
        for threshold in (0.01, 0.1, 0.5):
            original = mstls(G, b, [threshold], threshold_units="physical", allow_empty=False)
            scaled = mstls(G*scales[:, 0], b, [threshold], coefficient_scales=scales,
                           threshold_units="physical", allow_empty=False)
            np.testing.assert_allclose(original.coefficients, scaled.coefficients, atol=1e-14)
            np.testing.assert_allclose(original.losses, scaled.losses, atol=1e-14)

    def test_author_empty_guard_returns_previous_model(self):
        G, b = np.eye(2), np.array([0.2, 0.03])
        guarded = mstls(G, b, [1], allow_empty=False)
        np.testing.assert_allclose(guarded.coefficients[:, 0], b)
        np.testing.assert_array_equal(mstls(G, b, [1], allow_empty=True).coefficients, np.zeros((2, 1)))

    def test_bounds_apply_to_scaled_coefficients_and_allow_empty_support(self):
        result = mstls(np.eye(2), np.array([0.2, 0.03]), [0.1])
        np.testing.assert_allclose(result.scaled_coefficients[:, 0], [0.2, 0])
        np.testing.assert_array_equal(mstls(np.eye(2), [0.2, 0.03], [1]).coefficients, np.zeros((2, 1)))

    def test_smallest_threshold_wins_exact_loss_tie(self):
        result = mstls(np.eye(2), [0.2, 0.03], [0.15, 0.1])
        self.assertEqual(result.threshold, 0.1)
        self.assertEqual(result.losses[0], result.losses[1])

    def test_joint_loss_uses_the_matrix_two_norm(self):
        result = mstls(np.eye(2), np.diag([0.2, 0.03]), [0.1])
        # Removed projection norm: .03; full projection norm: .2;
        # one selected entry among four candidate coefficients.
        self.assertAlmostEqual(result.losses[0], 0.03/0.2 + 1/4)

    def test_rank_deficiency_and_zero_columns_are_supported(self):
        G = np.array([[1., 1., 0.], [0., 0., 0.]])
        result = mstls(G, [0.4, 0], [0.01])
        self.assertEqual(result.rank, 1)
        np.testing.assert_allclose(G@result.coefficients[:, 0], [0.4, 0])
        np.testing.assert_array_equal(mstls(G, [0, 0]).coefficients, np.zeros((3, 1)))

    def test_nonlinear_transport_recovers_physical_coefficients(self):
        x, t = np.linspace(-1, 1, 81), np.linspace(0, 0.5, 81)
        u = (0.3*x[:, None]+0.8+0.12*t)/(1+0.3*t)
        terms = (LibraryTerm((1,), (1, 0)), LibraryTerm((2,), (1, 0)))
        for rescale in (False, True):
            result, _ = wsindy(u, (x,), t, library_terms=terms, half_widths=(25, 25),
                strides=(5, 5), test_degrees=(12, 12), thresholds=(0.01,), rescale=rescale)
            np.testing.assert_allclose(result.coefficients[:, 0], [0.4, -0.5], rtol=1e-7, atol=1e-8)

    def test_coupled_linear_schrodinger_equations(self):
        x, t = np.linspace(-np.pi, np.pi, 81), np.linspace(0, 3, 81)
        u, v = np.sin(x[:, None])*np.cos(0.5*t), np.sin(x[:, None])*np.sin(0.5*t)
        terms = (LibraryTerm((1, 0), (2, 0)), LibraryTerm((0, 1), (2, 0)))
        result, _ = wsindy((u, v), (x,), t, library_terms=terms, lhs_components=(0, 1),
            half_widths=(25, 25), strides=(5, 5), test_degrees=(12, 12), thresholds=(0.01,))
        np.testing.assert_allclose(result.coefficients, [[0, -0.5], [0.5, 0]], rtol=1e-7, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
