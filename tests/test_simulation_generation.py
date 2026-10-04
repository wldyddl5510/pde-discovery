"""Original data conventions, libraries, and reproducible observation noise."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from simulation_generation import (BENCHMARKS, add_gaussian_noise, file_checksum,
    load_clean_benchmark, observation_instance)


class BenchmarkTests(unittest.TestCase):
    def test_published_libraries_with_archived_rd_definition(self):
        expected = {"IB": 43, "KdV": 43, "KS": 43, "NLS": 190, "SG": 73, "RD": 181, "NS": 50}
        for name, count in expected.items():
            with self.subTest(name=name):
                spec = BENCHMARKS[name]
                terms = spec.library()
                self.assertEqual(len(terms), count)
                self.assertEqual(len(set(terms)), count)
                self.assertEqual(spec.truth().shape, (count, len(spec.lhs_components)))
                self.assertTrue(all(t.derivative[-1] == 0 for t in terms))
                self.assertEqual(sum(t.kind == "poly" and not any(t.powers) for t in terms), 1)
        self.assertEqual(BENCHMARKS["SG"].lhs_time_order, 2)
        self.assertEqual(BENCHMARKS["NS"].lhs_components, (0,))
        self.assertNotIn("PM", BENCHMARKS)  # PM is in the journal version, not arXiv v3.

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
