"""Paper schedules, coefficient metrics, and the actual execution path."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments import (BENCHMARK_NOISE_RATIOS, PAPER_NOISE_RATIOS, coefficient_metrics, format_report,
                         parse_arguments, read_records, run, summary_rows, trial_seed)
from simulation_generation import DEFAULT_DATA_DIR


class ExperimentTests(unittest.TestCase):
    def test_defaults_are_the_reduced_comparison_schedule(self):
        args = parse_arguments([])
        self.assertEqual(args.trials, 50)
        self.assertEqual(tuple(args.noise_ratios), BENCHMARK_NOISE_RATIOS)
        self.assertEqual(len(args.noise_ratios), 10)
        self.assertTrue(set(args.noise_ratios).issubset(PAPER_NOISE_RATIOS))
        self.assertEqual(args.benchmarks, ["IB", "KdV", "KS", "NLS", "SG", "RD", "NS"])
        self.assertEqual(args.profile, "authors")

    def test_seed_is_independent_of_request_order_and_varies_by_trial(self):
        seed = trial_seed(0, "KS", 0.2, 3)
        self.assertEqual(seed, trial_seed(0, "KS", 0.2, 3))
        for other in (trial_seed(0, "KS", 0.2, 4), trial_seed(0, "KS", 0.3, 3), trial_seed(0, "NLS", 0.2, 3)):
            self.assertNotEqual(seed, other)

    def test_metrics_count_false_positive_and_missing_terms(self):
        truth = np.array([[2., 0.], [-1., 1.], [0., 0.]])
        fitted = np.array([[1., 0.], [0., 1.], [0.5, 0.]])
        result = coefficient_metrics(fitted, truth)
        self.assertEqual((result["tp"], result["fn"], result["fp"]), (2, 1, 1))
        self.assertEqual(result["tpr"], 0.5)
        self.assertEqual(result["e_inf"], 1)
        self.assertFalse(result["exact_support"])
        self.assertAlmostEqual(result["e2"], np.sqrt(2.25/6))

    def test_bad_cli_arguments_are_rejected(self):
        for flags in (["--trials", "0"], ["--seed", "-1"], ["--noise-ratios", "1.1"],
                      ["--noise-ratios", "0", "0"], ["--benchmarks", "IB", "IB"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(flags)

    @unittest.skipUnless((DEFAULT_DATA_DIR/"burgers.mat").exists(), "Original IB data are not cached")
    def test_original_burgers_runs_without_mocks(self):
        args = parse_arguments(["--benchmarks", "IB", "--noise-ratios", "0", "--trials", "1", "--offline"])
        with contextlib.redirect_stderr(io.StringIO()):
            rows = run(args)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["G_shape"], (784, 43))
        self.assertEqual(rows[0]["tpr"], 1)
        self.assertLess(rows[0]["e2"], 1e-3)
        report = format_report(args, rows)
        self.assertIn("Subset of the paper", report)
        self.assertIn("1/(beta_max-1)", report)
        self.assertIn("physical-unit MSTLS", report)
        self.assertNotIn("LASSO", report)

    def test_resume_repairs_only_an_incomplete_final_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trials.jsonl"
            valid = json.dumps({"trial": 0})+"\n"
            path.write_text(valid+'{"trial":')
            self.assertEqual(read_records(path), [{"trial": 0}])
            self.assertEqual(read_records(path, repair=True), [{"trial": 0}])
            self.assertEqual(path.read_text(), valid)
            path.write_text(valid+"broken\n"+valid)
            with self.assertRaisesRegex(ValueError, "Invalid record inside"):
                read_records(path, repair=True)

    def test_summary_counts_exact_recovery_and_includes_failures(self):
        args = parse_arguments(["--benchmarks", "KS", "--noise-ratios", "0.2", "--trials", "2"])
        rows = [dict(name="KS", noise_ratio=.2, exact_support=i == 0, tpr=1-i*.5,
                     e_inf=i, e2=i, runtime=2+i, threshold=.1) for i in range(2)]
        summary = summary_rows(args, rows)[0]
        self.assertEqual(summary["exact_rate"], .5)
        self.assertEqual(summary["e2_mean"], .5)
        self.assertLess(summary["exact_ci95_low"], .5)
        self.assertGreater(summary["exact_ci95_high"], .5)


if __name__ == "__main__":
    unittest.main()
