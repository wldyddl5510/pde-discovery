"""The clean-state oracle enforces the nuisance assumption in physical units."""
import unittest

import numpy as np

from wsindy.oracle_pilot import oracle_pilot


def example(nx=7, nt=13, components=1):
    x, t = np.linspace(-1, 2, nx), np.linspace(0, .6, nt)
    clean = np.sin(x[:, None]-t[None, :])[..., None]
    clean = np.concatenate([clean+c for c in range(components)], axis=-1)
    base = clean+3*np.random.default_rng(11).normal(size=clean.shape)
    base[:, 1::2, :] = np.nan
    return clean, base, (x, t)


class OraclePilot(unittest.TestCase):
    def test_physical_weighted_bound_and_formula(self):
        clean, base, grid = example(components=2)
        pilot, info = oracle_pilot(clean, base, grid)
        kept = np.arange(0, clean.shape[-2], 2)
        dz = (grid[0][1]-grid[0][0])*(grid[1][1]-grid[1][0])
        n = len(grid[0])*len(kept)
        measured = dz*2*np.sum((pilot[:, kept]-clean[:, kept])**2)
        raw = dz*np.sum((pilot[:, kept]-clean[:, kept])**2)
        expected_alpha = np.sqrt(n**(-.5)/(dz*2*np.sum((base[:, kept]-clean[:, kept])**2)))
        self.assertLessEqual(measured, n**(-.5))
        self.assertAlmostEqual(measured, info["energy"], places=14)
        self.assertAlmostEqual(raw, info["energy_unweighted"], places=14)
        self.assertAlmostEqual(info["alpha"], expected_alpha, places=13)
        self.assertEqual(info["n_regression"], n)
        self.assertEqual(info["n_total"], len(grid[0])*len(grid[1]))
        self.assertEqual(info["components"], 2)
        np.testing.assert_allclose(pilot[:, kept]-clean[:, kept], info["alpha"]*(base[:, kept]-clean[:, kept]), atol=1e-15)

    def test_exact_preserves_split_mask_and_inputs(self):
        clean, base, grid = example(components=3)
        clean_original, base_original = clean.copy(), base.copy()
        exact, info = oracle_pilot(clean, base, grid, mode="exact")
        np.testing.assert_array_equal(np.isnan(exact), np.isnan(base))
        np.testing.assert_array_equal(exact[np.isfinite(exact)], clean[np.isfinite(exact)])
        np.testing.assert_array_equal(clean, clean_original)
        np.testing.assert_array_equal(base, base_original)
        self.assertEqual(info["energy"], 0.)
        self.assertEqual(info["alpha"], 0.)

    def test_already_accurate_base_is_unchanged_and_deterministic(self):
        clean, base, grid = example()
        base[np.isfinite(base)] = clean[np.isfinite(base)]+1e-6
        first, first_info = oracle_pilot(clean, base, grid)
        second, second_info = oracle_pilot(clean, base, grid)
        np.testing.assert_array_equal(first, base)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_info, second_info)
        self.assertEqual(first_info["alpha"], 1.)
        base[np.isfinite(base)] = clean[np.isfinite(base)]
        exact, info = oracle_pilot(clean, base, grid)
        np.testing.assert_array_equal(exact, base)
        self.assertEqual(info["base_energy"], 0.)
        self.assertEqual(info["energy"], 0.)

    def test_general_split_weights_match_direct_trapezoid(self):
        clean, _, grid = example(nt=17)
        kept = np.array([0, 1, 4, 6, 7, 10, 13, 14, 16])
        base = np.full_like(clean, np.nan)
        base[:, kept] = clean[:, kept]+.05
        _, info = oracle_pilot(clean, base, grid)
        tk = grid[1][kept]
        dt = np.r_[tk[1]-tk[0], (tk[2:]-tk[:-2])/2, tk[-1]-tk[-2]]
        measured = (grid[0][1]-grid[0][0])*np.sum((base[:, kept]-clean[:, kept])**2*dt[None, :, None])
        self.assertAlmostEqual(info["base_energy"], measured, places=14)
        self.assertGreaterEqual(info["base_energy"], info["base_energy_unweighted"])

    def test_multidimensional_grid_and_component_count(self):
        x, y, t = np.arange(4)*.2, np.arange(5)*.3, np.arange(8)*.1
        clean = np.zeros((4, 5, 8, 3))
        base = np.ones_like(clean)
        base[:, :, 1::2] = np.nan
        _, info = oracle_pilot(clean, base, (x, y, t))
        self.assertEqual(info["n_regression"], 4*5*4)
        self.assertEqual(info["n_total"], 4*5*8)
        self.assertAlmostEqual(info["delta_z"], .2*.3*.1)
        self.assertAlmostEqual(info["base_energy"], .2*.3*.1*2*(4*5*4)*3)

    def test_rate_implies_little_o_bound_and_accepts_custom_target(self):
        for nx in (7, 17, 37):
            clean, base, grid = example(nx=nx)
            _, info = oracle_pilot(clean, base, grid, exponent=.75, cap=.4)
            n = info["n_regression"]
            self.assertLessEqual(info["energy"], .4*n**(-.75))
            self.assertLessEqual(info["error_over_n_quarter"], .4*n**(-.5))
            self.assertAlmostEqual(info["bound_over_n_quarter"], .4*n**(-.5))

    def test_scalar_time_only_and_floating_cancellation_stay_feasible(self):
        t = np.arange(8)*.1
        clean = np.full((8, 1), 1e8)
        base = clean+3
        base[1::2] = np.nan
        pilot, info = oracle_pilot(clean, base, (t,), cap=1e-14)
        self.assertEqual(info["n_regression"], 4)
        self.assertLessEqual(info["energy"], info["target_energy"])
        self.assertTrue(np.isfinite(pilot[::2]).all())

    def test_roundoff_excess_preserves_near_target_radius(self):
        # The desired residual .025 is representable with small relative
        # error, but adding it to 1e8 rounds upwards.  Halving alpha after
        # that tiny excess would unnecessarily reduce energy by a factor 4.
        t = np.arange(8)*.1
        clean = np.full((8, 1), 1e8)
        base = clean+3.
        base[1::2] = np.nan
        _, info = oracle_pilot(clean, base, (t,), cap=.001)
        self.assertLessEqual(info["energy"], info["target_energy"])
        self.assertGreater(info["energy"], info["target_energy"]*(1-1e-4))

    def test_invalid_inputs_fail_explicitly(self):
        clean, base, grid = example(components=2)
        for kwargs in (dict(mode="unavailable"), dict(exponent=.25), dict(exponent=np.inf),
                       dict(cap=0), dict(cap=np.nan)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                oracle_pilot(clean, base, grid, **kwargs)
        cases = [
            (clean[..., 0], base, grid),
            (clean, clean, grid),
            (clean, np.full_like(base, np.nan), grid),
            (clean, base, grid[:1]),
            (clean, base, (grid[0][::-1], grid[1])),
            (clean, base, (grid[0], grid[1]**2)),
        ]
        partial_nan = base.copy(); partial_nan[0, 0, 0] = np.nan
        infinite = base.copy(); infinite[0, 0, 0] = np.inf
        invalid_clean = clean.copy(); invalid_clean[0, 0, 0] = np.nan
        cases.extend([(clean, partial_nan, grid), (clean, infinite, grid), (invalid_clean, base, grid)])
        for a, b, g in cases:
            with self.subTest(shape=a.shape, grid=g), self.assertRaises(ValueError):
                oracle_pilot(a, b, g)


if __name__ == "__main__":
    unittest.main()
