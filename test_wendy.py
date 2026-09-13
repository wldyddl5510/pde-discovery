"""Numerical tests for Section 2 notation and all three PDE estimators."""
import unittest

import numpy as np

import experiment
import wsindy
import wendy
import wendy_mle


def make_system(name, grid=64, centers=5, noise=0.):
    problem = experiment.PROBLEMS[name]
    x = np.linspace(*problem.xlim, grid)
    t = np.linspace(*problem.tlim, grid)
    u = problem.solution(x, t)
    U = u+noise*np.random.default_rng(4).normal(size=u.shape)
    A = wsindy.build_test_matrices(x, t, problem.alpha, problem.half_widths, centers)
    return wsindy.WeakSystem(U, A, wsindy.polynomial_dictionary(), problem.alpha), problem.w_star()


class NumericalChecks(unittest.TestCase):
    def test_support_metrics(self):
        score = experiment.support_metrics(np.array([7., 0., 9.]), np.array([1., 1., 0.]))
        self.assertEqual((score['tp'], score['fp'], score['fn']), (1, 1, 1))
        self.assertEqual((score['precision'], score['recall'], score['f1']), (.5, .5, .5))
        self.assertFalse(score['support_exact'])
        self.assertTrue(experiment.support_metrics(np.array([100., 0.]), np.array([1., 0.]))['support_exact'])
        self.assertEqual(experiment.support_metrics(np.zeros(2), np.array([1., 0.]))['f1'], 0.)
        self.assertFalse(experiment.support_metrics(np.array([np.nan]), np.ones(1))['support_valid'])

    def test_section2_layout_and_residual(self):
        system, target = make_system('burgers')
        self.assertEqual(system.X_hat.shape, (25, 12))
        self.assertEqual(system.index(2, 2), 6)
        self.assertEqual(target[6], -.5)  # PDF w_7
        np.testing.assert_allclose(system.Y_hat, system.A[0]@system.U)
        for s in range(1, system.S+1):
            for j, feature in enumerate(system.features):
                index = system.index(s, j)
                if j == 0 and sum(system.alpha[s]):
                    np.testing.assert_array_equal(system.X_hat[:, index], 0)
                    self.assertNotIn(index, system.admissible)
                else:
                    np.testing.assert_allclose(system.X_hat[:, index], system.A[s]@feature.f(system.U))
        w = np.arange(12)/10
        expected = system.A[0]@system.U
        for s in range(1, system.S+1):
            for j in range(1, 4):
                expected -= w[(s-1)*4+j]*(system.A[s]@system.features[j].f(system.U))
        np.testing.assert_allclose(system.R(w), expected, atol=1e-10)

    def test_entropy_solution_at_shock_time(self):
        values = experiment.burgers(np.array([-3., -1., .25, 1.]), np.array([0., 2., 3.]))
        np.testing.assert_allclose(values, [[1, .5, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])
        self.assertTrue(np.isfinite(values).all())

    def test_kdv_solution_satisfies_strong_pde(self):
        # Independent finite-difference validation of the exact data generator.
        x, t = np.array([-2.1, -.3, 1.7, 4.2]), np.array([.2, 1.3, 2.7])
        h, f = .002, experiment.kdv
        U = f(x, t)
        ut = (f(x, t+h)-f(x, t-h))/(2*h)
        ux = (f(x+h, t)-f(x-h, t))/(2*h)
        uxxx = (f(x+2*h, t)-2*f(x+h, t)+2*f(x-h, t)-f(x-2*h, t))/(2*h**3)
        self.assertLess(np.linalg.norm(ut+U*ux+uxxx)/np.linalg.norm(ut), 3e-5)

    def test_data_jacobian_and_covariance(self):
        system, _ = make_system('kdv')
        support, w = np.array([1, 6, 13]), np.array([.02, -.4, -.9])
        direction = np.random.default_rng(7).normal(size=system.n)
        def residual(U):
            model = wsindy.WeakSystem(U, system.A, system.features, system.alpha)
            return model.R(wsindy.expand(system, support, w))
        finite_difference = (residual(system.U+1e-6*direction)-residual(system.U-1e-6*direction))/2e-6
        L = system.A[0].copy()
        for index, coefficient in zip(support, w):
            s0, j = divmod(index, system.J)
            L -= coefficient*system.A[s0+1].multiply(system.features[j].df(system.U))
        np.testing.assert_allclose(L@direction, finite_difference, rtol=1e-5, atol=1e-8)
        covariance = wendy.ResidualCovariance(system, support, .03)
        expected = .03**2*(L@L.T).toarray()+covariance.ridge*np.eye(system.K)
        np.testing.assert_allclose(covariance.matrix(w), expected, rtol=1e-9, atol=1e-13)
        self.assertGreater(np.linalg.norm(expected-np.diag(np.diag(expected))), 0)

    def test_likelihood_gradient_and_hessian(self):
        system, _ = make_system('kdv')
        support, w = np.array([1, 6, 13]), np.array([.02, -.4, -.9])
        cov = wendy.ResidualCovariance(system, support, .03)
        objective = wendy_mle.WeakLikelihood(system.X_hat[:, support], system.Y_hat, cov)
        _, gradient, hessian = objective.evaluate(w)
        gradient_numeric, hessian_numeric = np.empty(3), np.empty((3, 3))
        for j in range(3):
            step = np.eye(3)[j]*1e-6
            gradient_numeric[j] = (objective.value(w+step)-objective.value(w-step))/2e-6
            hessian_numeric[:, j] = (objective.gradient(w+step)-objective.gradient(w-step))/2e-6
        np.testing.assert_allclose(gradient, gradient_numeric, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(hessian, hessian_numeric, rtol=1e-5, atol=1e-5)

    def test_noiseless_sparse_pde_recovery(self):
        for name in ('burgers', 'kdv'):
            with self.subTest(problem=name):
                system, target = make_system(name, 128, 11)
                selected = wsindy.fit(system)
                np.testing.assert_array_equal(np.flatnonzero(selected.w), np.flatnonzero(target))
                self.assertLess(np.linalg.norm(selected.w-target)/np.linalg.norm(target), .001)
                for method in (wendy, wendy_mle):
                    fitted = method.fit(system, sigma=0., sparsity=np.count_nonzero(target), initial=np.zeros_like(target))
                    dense = wsindy.expand(system, system.admissible, wsindy.least_squares(
                        system.X_hat[:, system.admissible], system.Y_hat))
                    np.testing.assert_allclose(fitted.w, wendy.hard_threshold(dense, np.count_nonzero(target)), atol=1e-12)
                    np.testing.assert_array_equal(np.flatnonzero(fitted.w), np.flatnonzero(target))

    def test_noisy_pde_recovery_and_fit_histories(self):
        system, target = make_system('kdv', 128, 11, noise=.03)
        for module in (wendy, wendy_mle):
            fitted = module.fit(system, sigma=.03, sparsity=2)
            self.assertEqual(np.count_nonzero(fitted.w), 2)
            np.testing.assert_array_equal(np.flatnonzero(fitted.w), np.flatnonzero(target))
            self.assertLess(np.linalg.norm(fitted.w-target)/np.linalg.norm(target), .1)
            self.assertGreaterEqual(fitted.seconds, fitted.history[-1].seconds)
            self.assertTrue(all(b.seconds >= a.seconds for a, b in zip(fitted.history, fitted.history[1:])))
            np.testing.assert_allclose(fitted.w, fitted.history[-1].w)

    def test_l1_solution_and_stationarity(self):
        # Independent exact LASSO solution for an orthogonal design.
        b, penalty = np.array([3., -.7, .1]), .2
        result = wendy.l1_minimize(lambda w: (.5*np.sum((w-b)**2), w-b),
                                   np.zeros(3), penalty, np.array([.5, 2., 4.]), 7., 1000, 1e-10)
        expected = np.sign(b)*np.maximum(np.abs(b)-penalty, 0.)
        np.testing.assert_allclose(result.x, expected, atol=1e-8)
        self.assertTrue(result.success)
        system, _ = make_system('kdv', 128, 11)
        fitted = wendy.fit(system, sigma=0., l1=1e-6)
        X, y, w = system.X_hat[:, system.admissible], system.Y_hat, fitted.w[system.admissible]
        penalty = 1e-6*wendy.penalty_reference(X, y, None)[0]
        gradient = X.T@(X@w-y)/len(y)
        active = w != 0
        np.testing.assert_allclose(gradient[active]+penalty*np.sign(w[active]), 0., atol=1e-5)
        self.assertTrue(np.all(np.abs(gradient[~active]) <= penalty+1e-5))

    def test_weak_quadrature_refinement(self):
        errors = []
        for grid in (64, 128):
            system, target = make_system('burgers', grid)
            fitted = wsindy.fit(system, support=np.flatnonzero(target))
            errors.append(np.linalg.norm(fitted.w-target))
        self.assertLess(errors[1], errors[0])

    def test_invalid_support_rejected(self):
        system, _ = make_system('burgers')
        for support in ([4], [-1], [6, 6], [6.5]):
            with self.assertRaises(ValueError):
                wsindy.fit(system, support=support)


if __name__ == '__main__':
    unittest.main()
