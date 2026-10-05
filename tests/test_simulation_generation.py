"""Original data conventions, libraries, and reproducible observation noise."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from simulation_generation import (BENCHMARKS, add_gaussian_noise, file_checksum,
    _etdrk4_solution, _spectral_nonlinearity, load_clean_benchmark, observation_instance)


class BenchmarkTests(unittest.TestCase):
    def test_consistency_pde_libraries_and_centered_noise(self):
        self.assertEqual(len(BENCHMARKS["HKS"].library()), 73)
        self.assertEqual(len(BENCHMARKS["VBG"].library()), 43)
        self.assertEqual(np.count_nonzero(BENCHMARKS["HKS"].truth()), 4)
        self.assertEqual(np.count_nonzero(BENCHMARKS["VBG"].truth()), 5)
        data = np.linspace(-1, 1, 10000).reshape(100, 100)+10
        first, sigma = add_gaussian_noise(data, .5, 3, scale_rule="centered_std")
        second, shifted_sigma = add_gaussian_noise(data+100, .5, 3, scale_rule="centered_std")
        self.assertAlmostEqual(sigma, .5*np.std(data, ddof=1), places=14)
        self.assertAlmostEqual(shifted_sigma, sigma, places=14)
        np.testing.assert_allclose(second-first, 100, atol=1e-13)

    def test_padded_spectral_products_match_direct_pde_rhs(self):
        n = 32
        x = np.arange(n)*2*np.pi/n
        wave = 2*np.pi*np.fft.fftfreq(n, d=2*np.pi/n)
        u = 1+.5*np.sin(2*x)+.2*np.cos(3*x)
        v, square = np.fft.fft(u), np.fft.fft(u*u)
        hyper = (-.5j*wave+.1*(1j*wave)**3)*square
        burgers = -.5j*wave*square+np.fft.fft(-u**3+2*u**2+1)
        np.testing.assert_allclose(_spectral_nonlinearity(v, wave, "HKS"), hyper, atol=5e-12)
        np.testing.assert_allclose(_spectral_nonlinearity(v, wave, "VBG"), burgers, atol=5e-12)

    def test_etdrk4_actual_burgers_trajectory_converges_under_time_refinement(self):
        x, t, coarse = _etdrk4_solution("VBG", 64, 101, 4)
        _, _, medium = _etdrk4_solution("VBG", 64, 101, 8)
        _, _, fine = _etdrk4_solution("VBG", 64, 101, 16)
        self.assertEqual(coarse.shape, (64, 101))
        np.testing.assert_allclose(coarse[:, 0], 2*np.sin(np.pi*x), atol=1e-14)
        self.assertEqual(t[-1], 1.5)
        self.assertLess(np.linalg.norm(medium-fine)/np.linalg.norm(fine), 1e-6)
        self.assertGreater(np.linalg.norm(coarse-medium)/np.linalg.norm(medium-fine), 10)

    def test_published_libraries_with_archived_rd_definition(self):
        expected = {"IB": 43, "KdV": 43, "KS": 43, "NLS": 190, "RD": 181, "NS": 50}
        for name, count in expected.items():
            with self.subTest(name=name):
                spec = BENCHMARKS[name]
                terms = spec.library()
                self.assertEqual(len(terms), count)
                self.assertEqual(len(set(terms)), count)
                self.assertEqual(spec.truth().shape, (count, len(spec.lhs_components)))
                self.assertTrue(all(t.derivative[-1] == 0 for t in terms))
                self.assertEqual(sum(t.kind == "poly" and not any(t.powers) for t in terms), 1)
        self.assertEqual(BENCHMARKS["NS"].lhs_components, (0,))
        self.assertNotIn("PM", BENCHMARKS)  # PM is in the journal version, not arXiv v3.

    def test_active_benchmarks_have_only_polynomial_dictionary_and_truth(self):
        self.assertEqual(set(BENCHMARKS), {"IB", "KdV", "KS", "NLS", "RD", "NS", "HKS", "VBG"})
        for spec in BENCHMARKS.values():
            self.assertTrue(all(term.kind == "poly" for term in spec.library()))
            self.assertTrue(all(term.kind == "poly" for equation in spec.true_coefficients for term in equation))

    def test_navier_stokes_special_library(self):
        for term in BENCHMARKS["NS"].library():
            if any(term.derivative):
                self.assertGreater(term.powers[0], 0)
                self.assertLessEqual(sum(term.powers), 3)
            else:
                self.assertLessEqual(sum(term.powers), 2)

    def test_cell_arrays_keep_component_and_axis_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"NLS.mat"
            components = np.empty(2, dtype=object)
            components[0], components[1] = np.arange(99).reshape(9, 11), -np.arange(99).reshape(9, 11)
            axes = np.empty(2, dtype=object)
            axes[0], axes[1] = np.linspace(-2, 2, 9), np.linspace(0, 1, 11)
            savemat(path, {"U_exact": components, "xs": axes})
            spec = replace(BENCHMARKS["NLS"], shape=(9, 11), checksum=file_checksum(path))
            with patch.dict(BENCHMARKS, {"NLS": spec}):
                data = load_clean_benchmark("NLS", data_dir=directory, download=False)
            np.testing.assert_array_equal(data.u_true[0], components[0])
            np.testing.assert_array_equal(data.u_true[1], components[1])
            np.testing.assert_array_equal(data.time, axes[1])
            noisy = observation_instance(data, 0.1, 3)
            np.testing.assert_array_equal(data.u_true[0], components[0])
            self.assertFalse(np.array_equal(noisy.u_observed[0], data.u_true[0]))

    def test_navier_stokes_archive_order_is_mapped_to_vorticity_first(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"Nav_Stokes.mat"
            components = np.empty(3, dtype=object)
            for i in range(3):
                components[i] = np.full((9, 11, 13), i+1.0)
            axes = np.empty(3, dtype=object)
            for i, size in enumerate((9, 11, 13)):
                axes[i] = np.linspace(0, 1, size)
            savemat(path, {"U_exact": components, "xs": axes})
            spec = replace(BENCHMARKS["NS"], shape=(9, 11, 13), checksum=file_checksum(path))
            with patch.dict(BENCHMARKS, {"NS": spec}):
                data = load_clean_benchmark("NS", data_dir=directory, download=False)
            self.assertEqual([v[0, 0, 0] for v in data.u_true], [3, 1, 2])

    def test_missing_and_corrupt_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_clean_benchmark("IB", data_dir=directory, download=False)
            (Path(directory)/"burgers.mat").write_bytes(b"not the authors data")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                load_clean_benchmark("IB", data_dir=directory, download=False)

    def test_noise_per_component_rms_seed_and_input_preservation(self):
        clean = (np.ones((200, 200)), 10*np.ones((200, 200)))
        before = tuple(v.copy() for v in clean)
        first, sigma = add_gaussian_noise(clean, 0.2, 7)
        second, _ = add_gaussian_noise(clean, 0.2, 7)
        self.assertEqual(sigma, (0.2, 2.0))
        for i in range(2):
            np.testing.assert_array_equal(clean[i], before[i])
            np.testing.assert_array_equal(first[i], second[i])
            self.assertAlmostEqual(np.std(first[i]-clean[i])/sigma[i], 1, delta=0.02)
        self.assertLess(abs(np.corrcoef((first[0]-clean[0]).ravel(), (first[1]-clean[1]).ravel())[0, 1]), 0.02)
        np.random.seed(12)
        expected = np.random.rand()
        np.random.seed(12)
        add_gaussian_noise(clean, 0.2, 7)
        self.assertEqual(np.random.rand(), expected)

    def test_invalid_noise_and_clean_values(self):
        for ratio in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                add_gaussian_noise(np.ones((3, 3)), ratio)
        with self.assertRaises(ValueError):
            add_gaussian_noise(np.full((3, 3), np.nan))


if __name__ == "__main__":
    unittest.main()
