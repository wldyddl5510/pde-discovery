"""Numerical tests for Section 2 notation and all three PDE estimators."""
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d
from scipy.integrate import simpson

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
    def test_strong_weak_pde_generator(self):
        x, t = np.linspace(-np.pi, np.pi, 129), np.linspace(0, .5, 129)
        w = experiment.SIGNAL.w_star(J=7)
        u = experiment.strong_weak(x, t)
        refined = experiment.strong_weak(np.linspace(-np.pi, np.pi, 257), t)[:, ::2]
        self.assertEqual(np.count_nonzero(np.abs(w) == 1), 3)
        self.assertEqual(np.count_nonzero(w == .001), 16)
        np.testing.assert_array_equal(u[:, 0], u[:, -1])
        self.assertLess(np.linalg.norm(u-refined)/np.linalg.norm(u), 2e-7)
        dominant = np.zeros_like(w)
        dominant[experiment.STRONG] = w[experiment.STRONG]
        self.assertGreater(np.linalg.norm(u-experiment.strong_weak(x, t, dominant))/np.linalg.norm(u), 1e-4)
        # Periodic flux and diffusion integrate to zero; only reaction changes the mean.
        mean = u[:, :-1].mean(axis=1)
        reaction = np.polynomial.polynomial.polyval(u[:, :-1], w[:7]).mean(axis=1)
        self.assertLess(abs(mean[-1]-mean[0]-simpson(reaction, x=t)), 1e-10)

    def test_l1_objective_scaling_and_tiny_penalty(self):
        b = np.array([3., -.7, .1])
        for penalty in (.2, 1e-10):
            for normalizer in (1., 1e12):
                result = wendy.l1_minimize(lambda w: (.5*np.sum((w-b)**2)/normalizer, (w-b)/normalizer),
                                           b.copy(), penalty/normalizer, np.ones(3), 1000, 1e-8)
                expected = np.sign(b)*np.maximum(np.abs(b)-penalty, 0.)
                np.testing.assert_allclose(result.x, expected, atol=penalty*.01, rtol=0.)
                self.assertTrue(result.success, result.message)
                self.assertGreater(result.nit, 0)

    def test_lasso_correlated_design(self):
        X = np.random.default_rng(3).normal(size=(40, 5))
        X[:, 1] += .9*X[:, 0]
        target, penalty = np.array([1., 0., -.5, 0., 0.]), .02
        y = X@target+X@np.linalg.solve(X.T@X, len(X)*penalty*np.sign(target))
        result = wendy.lasso(X, y, penalty)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, target, atol=1e-12)

    def test_time_limit_is_not_convergence(self):
        system, target = make_system('kdv', noise=.02)
        for method in (wendy, wendy_mle):
            result = method.fit(system, sigma=.02, sparsity=2, time_limit=1e-9)
            self.assertFalse(result.success)
            self.assertEqual(result.message, 'Time limit')
            self.assertEqual(result.iterations, 0)
            np.testing.assert_array_equal(result.w, result.history[-1].w)

    @unittest.skipUnless(Path('tmp/WSINDy_PDE/datasets/burgers.mat').exists(), 'Original paper data not cached')
    def test_noiseless_paper_lasso_kkt(self):
        x, t, u, _ = experiment.paper_data('burgers')
        raw = experiment.paper_system('burgers', u, x, t)
        system = wendy.orthonormalize(raw)
        result = wendy.fit(raw, sigma=0., l1=1e-10)
        X, y = system.X_hat[:, system.admissible], system.Y_hat
        w = result.w[system.admissible]
        reference = np.max(np.abs(X.T@y/len(y)))
        penalty = 1e-10*reference
        gradient = X.T@(X@w-y)/len(y)
        violation = np.where(w != 0, np.abs(gradient+penalty*np.sign(w)), np.maximum(np.abs(gradient)-penalty, 0))
        self.assertLess(np.max(violation), .01*penalty)
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.iterations, 0)

    def test_projected_pde_operators_and_hessian_product(self):
        x, t = np.linspace(-4, 4, 64), np.linspace(0, 3, 64)
        U = 2+np.cos(x)[None, :]*np.exp(-t[:, None])
        raw = wsindy.ConvolutionSystem(U, x, t, (12, 12), (6, 6))
        system = wendy.orthonormalize(raw, max_rows=12)
        support, w = np.array([0, 9, 22, 6, 48]), np.array([.1, -.5, -.1, .02, .02])
        cov = wendy.ResidualCovariance(system, support, .03)
        np.testing.assert_allclose(cov.operators[1]@cov.operators[1].T, np.eye(system.K), atol=1e-12)
        terms = raw.data_jacobian_terms(support)
        for raw_term, projected_term in zip(terms, system.data_jacobian_terms(support)):
            np.testing.assert_allclose(projected_term.toarray(), system.projection@raw_term.toarray(), atol=1e-12)
        factor = system.projection@(terms[0]+sum(a*A for a, A in zip(w, terms[1:]))).toarray()
        np.testing.assert_allclose(cov.matrix(w), .03**2*factor@factor.T+cov.ridge*np.eye(system.K), atol=1e-12)
        np.testing.assert_allclose(system.Y_hat, cov.operators[0]@system.U, atol=1e-12)
        for j in support:
            np.testing.assert_allclose(system.X_hat[:, j], cov.operators[j//7+1]@system.U**(j%7), atol=1e-10)
        objective = wendy_mle.WeakLikelihood(system.X_hat[:, support], system.Y_hat, cov)
        direction = np.array([.2, -.1, .3, .05, -.02])
        analytic = objective.hessp(w, direction)
        numeric = (objective.gradient(w+1e-6*direction)-objective.gradient(w-1e-6*direction))/2e-6
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=1e-5)

    def test_convolution_data_jacobian(self):
        # Perturb rescaled observations with fixed kernels/scales, independently of CSR assembly.
        x, t = np.linspace(-4, 4, 64), np.linspace(0, 3, 64)
        U = 2+np.cos(x)[None, :]*np.exp(-t[:, None])
        system = wsindy.ConvolutionSystem(U, x, t, (12, 12), (12, 12))
        support, w = np.array([0, 9, 22, 6, 48]), np.array([.1, -.5, -.1, .02, .02])
        def residual(v):
            v = v.reshape(system.shape)
            def apply(values, dx, dt):
                kernel = np.outer(system.wt[dt], system.wx[dx])
                return convolve2d(values, kernel, mode='valid')[::12, ::12].ravel()
            return apply(v, 0, 1)-sum(a*apply(v**(i%7), i//7, 0) for i, a in zip(support, w))
        direction = np.random.default_rng(7).normal(size=system.n)
        numeric = (residual(system.U+1e-6*direction)-residual(system.U-1e-6*direction))/2e-6
        terms = system.data_jacobian_terms(support)
        analytic = (terms[0]+sum(a*A for a, A in zip(w, terms[1:])))@direction
        np.testing.assert_allclose(numeric, analytic, rtol=1e-7, atol=1e-9)

    def test_irls_step_matches_gls(self):
        system, _ = make_system('kdv', noise=.02)
        support, initial = np.array([1, 6, 13]), np.zeros(system.S*system.J)
        X, y = system.X_hat[:, support], system.Y_hat
        covariance = wendy.ResidualCovariance(system, support, .02).matrix(initial[support])
        inverse_X = np.linalg.solve(covariance, X)
        expected = np.linalg.solve(X.T@inverse_X, inverse_X.T@y)
        actual = wendy.fit(system, sigma=.02, support=support, initial=initial, maxiter=1)
        np.testing.assert_allclose(actual.w[support], expected, rtol=1e-9)

    def test_convolution_covariance_and_likelihood(self):
        x, t = np.linspace(-4, 4, 80), np.linspace(0, 3, 80)
        U = 2+np.cos(x)[None, :]*np.exp(-t[:, None])
        system = wsindy.ConvolutionSystem(U, x, t, (12, 12), (12, 12))
        support, w = np.array([0, 9, 22, 6, 48]), np.array([.1, -.5, -.1, .02, .02])
        covariance = wendy.ResidualCovariance(system, support, .03)
        terms = system.data_jacobian_terms(support)
        L = terms[0]+sum(value*A for value, A in zip(w, terms[1:]))
        np.testing.assert_allclose(system.Y_hat, terms[0]@system.U, atol=1e-13)
        np.testing.assert_allclose(system.X_hat[:, 9], -terms[2]@system.U/2, atol=1e-13)
        np.testing.assert_allclose(system.X_hat[:, 22], -terms[3]@system.U, atol=1e-13)
        np.testing.assert_allclose(covariance.matrix(w), .03**2*(L@L.T).toarray()+
                                   covariance.ridge*np.eye(system.K), rtol=1e-9, atol=1e-13)
        objective = wendy_mle.WeakLikelihood(system.X_hat[:, support], system.Y_hat, covariance)
        gradient = objective.gradient(w)
        numeric = [(objective.value(w+np.eye(len(w))[j]*1e-6)-objective.value(w-np.eye(len(w))[j]*1e-6))/2e-6
                   for j in range(len(w))]
        np.testing.assert_allclose(gradient, numeric, rtol=1e-4, atol=1e-4)
        system.workers = 2
        parallel = wendy.ResidualCovariance(system, support, .03)
        parallel_objective = wendy_mle.WeakLikelihood(system.X_hat[:, support], system.Y_hat, parallel)
        np.testing.assert_allclose(parallel_objective.gradient(w), gradient, rtol=1e-12)

    @unittest.skipUnless(Path('tmp/WSINDy_PDE/datasets/KdV.mat').exists(), 'Original paper data not cached')
    def test_paper_table5(self):
        for name, K, reference in (('burgers', 784, 4.3e-5), ('kdv', 1443, 3.1e-7)):
            x, t, u, target = experiment.paper_data(name)
            system = experiment.paper_system(name, u, x, t)
            self.assertEqual((system.K, len(system.admissible)), (K, 43))
            result = wsindy.fit(system, thresholds=np.logspace(-4, 0, 50), threshold_scale=system.coefficient_scale)
            estimate = system.coefficient_scale*result.w
            np.testing.assert_array_equal(np.flatnonzero(estimate), np.flatnonzero(target))
            error = np.max(np.abs((estimate-target)[target != 0]/target[target != 0]))
            self.assertAlmostEqual(error/reference, 1., delta=.02)

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
                                   np.zeros(3), penalty, np.array([.5, 2., 4.]), 1000, 1e-10)
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
