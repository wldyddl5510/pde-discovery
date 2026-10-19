"""The discrete Dantzig score differentiates the score, not only its design."""
from dataclasses import replace
import unittest

import numpy as np
import pysindy as ps

from pdes import AUTHORS, PDE
from wsindy import Block, SplitSystem, Tests, WeakSystem, spatial
from wsindy.scores import score_matrices


def example(components=1, rescale=True, split=True):
    x, t = np.linspace(-1, 1, 19), np.linspace(0, 2, 25)
    u = 1.7*np.sin(x[:, None]-.6*t[None, :])+.2*x[:, None]**2+.1
    clean = [u]
    if components == 2:
        clean.append(.9*np.cos(1.4*x[:, None]+t[None, :])+.3)
    pilot = np.stack(clean, axis=-1)
    observed = pilot+.12*np.random.default_rng(4).normal(size=pilot.shape)
    names = ("u",) if components == 1 else ("u", "v")
    pde = PDE("score-test", "", "", names,
              (Block(ps.PolynomialLibrary(degree=4), spatial(1, 1)),),
              tuple(f"{name}_t" for name in names), tuple({} for _ in names),
              Tests(m=(3, 3), p=(4, 4), s=(3, 2)), replace(AUTHORS, rescale=rescale))
    if split:
        pilot[:, 1::2] = np.nan
        system = SplitSystem(pde, observed, (x, t), X=pilot)
    else:
        system = WeakSystem(pde, observed, (x, t), X=pilot)
    return system


class PhysicalScores(unittest.TestCase):
    def test_directional_derivatives_agree_with_finite_differences(self):
        # Mixed monomials u^a v^b test the full multicomponent derivative.
        for components in (1, 2):
            with self.subTest(components=components):
                system = example(components)
                score = score_matrices(system)
                direction = system.U-system.X_state
                epsilon = 2e-5
                plus = system.X_state+epsilon*direction
                minus = system.X_state-epsilon*direction
                dx = (system.columns(system.terms, plus, rescale=False)
                      -system.columns(system.terms, minus, rescale=False))/(2*epsilon)
                dy = (system.columns(system.lhs, plus, rescale=False)
                      -system.columns(system.lhs, minus, rescale=False))/(2*epsilon)
                np.testing.assert_allclose(score.dX, dx, rtol=2e-7, atol=1e-11)
                np.testing.assert_allclose(score.dY, dy, rtol=2e-7, atol=1e-11)

    def test_first_order_score_omits_quadratic_increment_products(self):
        system = example(2)
        score = score_matrices(system)
        K = score.n_rows
        # Independent score differentiation, rather than differentiating columns.
        direction = system.U-system.X_state
        epsilon = 2e-5
        derivatives = []
        for state in (system.X_state+epsilon*direction, system.X_state-epsilon*direction):
            X = system.columns(system.terms, state, rescale=False)
            Y = system.columns(system.lhs, state, rescale=False)
            derivatives.append((X.T@X/K, X.T@Y/K))
        expected_H = score.X.T@score.X/K+(derivatives[0][0]-derivatives[1][0])/(2*epsilon)
        expected_b = score.X.T@score.Y/K+(derivatives[0][1]-derivatives[1][1])/(2*epsilon)
        np.testing.assert_allclose(score.H, expected_H, rtol=1e-8, atol=1e-11)
        np.testing.assert_allclose(score.b, expected_b, rtol=1e-8, atol=1e-11)
        corrected_H = score.refit_X.T@score.refit_X/K
        corrected_b = score.refit_X.T@score.refit_Y/K
        np.testing.assert_allclose(corrected_H-score.H, score.dX.T@score.dX/K, rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(corrected_b-score.b, score.dX.T@score.dY/K, rtol=1e-9, atol=1e-15)
        self.assertGreater(np.linalg.norm(corrected_H-score.H), 1e-6)
        self.assertGreater(np.linalg.norm(corrected_b-score.b), 1e-7)
        np.testing.assert_array_equal(score.H, score.H.T)

    def test_exact_pilot_reduces_to_observed_score(self):
        for split in (False, True):
            system = example(2, split=split)
            exact = system.U.copy()
            exact[np.isnan(system.X_state)] = np.nan
            system.X_state = exact
            scores = [score_matrices(system, variant=variant) for variant in ("debiased", "plugin", "raw")]
            for score in scores:
                np.testing.assert_allclose(score.H, scores[0].H, rtol=2e-14, atol=1e-15)
                np.testing.assert_allclose(score.b, scores[0].b, rtol=2e-14, atol=1e-15)
                np.testing.assert_allclose(score.dX, 0., atol=1e-15)
                np.testing.assert_allclose(score.dY, 0., atol=1e-15)

    def test_scores_are_invariant_to_wsindy_scaling(self):
        scaled, unscaled = example(2, rescale=True), example(2, rescale=False)
        self.assertGreater(np.linalg.norm(scaled.state_scales-1), .1)
        self.assertGreater(np.linalg.norm(scaled.coordinate_scales-1), .1)
        for variant in ("debiased", "plugin", "raw"):
            a, b = score_matrices(scaled, variant=variant), score_matrices(unscaled, variant=variant)
            for name in ("X", "Y", "dX", "dY", "H", "b"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_split_training_observations_do_not_enter_any_score(self):
        system = example(2)
        changed = system.U.copy()
        changed[:, 1::2] += 1e3
        other = SplitSystem(system.pde, changed, system.grid, X=system.X_state)
        for variant in ("debiased", "plugin", "raw"):
            a, b = score_matrices(system, variant=variant), score_matrices(other, variant=variant)
            for name in ("X", "Y", "dX", "dY", "H", "b"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
                self.assertTrue(np.isfinite(getattr(a, name)).all())
            self.assertEqual(a.b.shape, (len(system.terms), 2))

    def test_plugin_and_raw_use_declared_states_and_same_quadrature(self):
        system = example(2)
        # Score response is Y(pilot), even if WeakSystem's ordinary Y_state is U.
        for variant, state in (("plugin", system.X_state), ("raw", system.U)):
            score = score_matrices(system, variant=variant)
            X = system.columns(system.terms, state, rescale=False)
            Y = system.columns(system.lhs, state, rescale=False)
            np.testing.assert_array_equal(score.H, X.T@X/len(X))
            np.testing.assert_array_equal(score.b, X.T@Y/len(X))
            np.testing.assert_array_equal(score.refit_X, X)
            np.testing.assert_array_equal(score.refit_Y, Y)
        plugin, raw = score_matrices(system, variant="plugin"), score_matrices(system, variant="raw")
        self.assertGreater(np.linalg.norm(plugin.H-raw.H), 1e-5)

    def test_unsupported_feature_derivatives_fail_explicitly(self):
        system = example(split=False)
        pde = replace(system.pde, dictionary=(Block(ps.FourierLibrary(), spatial(1, 1)),), _cache={})
        general = WeakSystem(pde, system.U, system.grid, X=system.X_state)
        with self.assertRaisesRegex(ValueError, "polynomial dictionary"):
            score_matrices(general)
        self.assertTrue(np.isfinite(score_matrices(general, variant="raw").H).all())
        with self.assertRaisesRegex(ValueError, "variant"):
            score_matrices(system, variant="unknown")


if __name__ == "__main__":
    unittest.main()
