"""Paper schedules, coefficient metrics, and the actual execution path."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments import (BENCHMARK_NOISE_RATIOS, PAPER_NOISE_RATIOS, aggregate_comparisons, coefficient_metrics, format_report,
                         estimate_noise_std, format_comparison_report, moving_average_observations,
                         paired_records, parse_arguments, read_records, run, summary_rows, trial_seed)
from simulation_generation import DEFAULT_DATA_DIR


class ExperimentTests(unittest.TestCase):
    def test_lasso_dispatch_report_and_observation_pairing(self):
        args = parse_arguments(["--benchmarks", "IB", "--noise-ratios", ".2", "--trials", "1", "--offline", "--regression", "lasso"])
        with contextlib.redirect_stderr(io.StringIO()):
            record = run(args)[0]
        self.assertEqual(record["regression"], "lasso")
        self.assertEqual(record["seed"], trial_seed(0, "IB", .2, 0))
        self.assertEqual(len(record["thresholds"]), 100)
        self.assertLessEqual(np.max(record["kkt_errors"]), 1e-7)
        report = format_report(args, [record])
        self.assertIn("LASSO", report)
        self.assertIn("no OLS support refit", report)
        self.assertNotIn("physical-unit MSTLS bounds", report)

    def test_four_method_report_uses_common_trials_and_checks_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for regression in ("mstls", "lasso"):
                for method in ("wsindy", "filtered-wsindy"):
                    root = Path(directory)/(method+regression); root.mkdir()
                    path = root/"results.md"; paths.append(path)
                    raw = root/"results_trials"; raw.mkdir()
                    manifest = dict(protocol_id=method+regression, benchmarks={"KS": {}},
                        root_seed=0, noise_ratios=[.2], trials=2, rng="PCG64", noise="Gaussian",
                        dtype="float64", profile="authors", state_scale_rule="authors", method=method, regression=regression)
                    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
                    rows = [dict(name="KS", noise_ratio=.2, trial=i, seed=[i], noise_std=[.1],
                        G_shape=[4, 2], b_shape=[4, 1], protocol_id=manifest["protocol_id"],
                        exact_support=True, tpr=1, fp=0, e2=0, e_inf=0, runtime=1, threshold=.1)
                        for i in range(1 if regression == "lasso" else 2)]
                    (raw/"KS.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            output = Path(directory)/"comparison.md"
            self.assertFalse(aggregate_comparisons(paths, output))
            self.assertEqual(output.read_text().count("| 1 | 100.0% |"), 4)
            bad = paths[-1].parent/"results_trials"/"KS.jsonl"
            row = json.loads(bad.read_text()); row["seed"] = [99]
            bad.write_text(json.dumps(row)+"\n")
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                aggregate_comparisons(paths, output)

    def test_defaults_are_the_reduced_comparison_schedule(self):
        args = parse_arguments([])
        self.assertEqual(args.trials, 100)
        self.assertEqual(tuple(args.noise_ratios), BENCHMARK_NOISE_RATIOS)
        self.assertEqual(tuple(args.noise_ratios), (0., 0.2, 0.5, 0.75, 1.))
        self.assertTrue(set(args.noise_ratios).issubset(PAPER_NOISE_RATIOS))
        self.assertEqual(args.benchmarks, ["IB", "KdV", "KS", "NLS", "RD", "NS"])
        self.assertEqual(args.profile, "authors")
        self.assertEqual(args.method, "wsindy")

    def test_seed_is_independent_of_request_order_and_varies_by_trial(self):
        seed = trial_seed(0, "KS", 0.2, 3)
        self.assertEqual(seed, trial_seed(0, "KS", 0.2, 3))
        for other in (trial_seed(0, "KS", 0.2, 4), trial_seed(0, "KS", 0.3, 3), trial_seed(0, "NLS", 0.2, 3)):
            self.assertNotEqual(seed, other)

    def test_polynomial_scope_rejects_sg_and_keeps_archived_seed_indices(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(["--benchmarks", "SG"])
        for name, index in (("RD", 5), ("NS", 6), ("HKS", 7), ("VBG", 8)):
            ratio, trial = .2, 3
            bits = int(np.float64(ratio).view(np.uint64))
            old_seed = np.random.SeedSequence([0, index, bits & 0xffffffff, bits >> 32, trial]).generate_state(4).tolist()
            self.assertEqual(trial_seed(0, name, ratio, trial), old_seed)

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
                      ["--noise-ratios", "0", "0"], ["--benchmarks", "IB", "IB"],
                      ["--filter-prior", "0"], ["--compare-to", "baseline.md"]):
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

    def test_noise_estimate_removes_local_polynomial_signal(self):
        t = np.linspace(-1, 1, 50000)
        signal = 4+t+2*t**3+t**5
        self.assertLess(estimate_noise_std(signal), 1e-12)
        observed = signal+np.random.default_rng(123).normal(0, .4, signal.shape)
        self.assertAlmostEqual(estimate_noise_std(observed), .4, delta=.01)

    def test_moving_average_matches_explicit_reflected_tensor_window(self):
        observed = np.random.default_rng(7).normal(0, .4, (9, 11))
        filtered, info = moving_average_observations((observed, observed+2), (4, 5), 6)
        self.assertEqual(info["filter_widths"], (4, 5))
        padding = tuple((w//2, (w-1)//2) for w in info["filter_widths"])
        padded = np.pad(observed, padding, mode="symmetric")
        expected = np.zeros_like(observed)
        for x in range(4):
            for t in range(5):
                expected += padded[x:x+9, t:t+11]/20
        np.testing.assert_allclose(filtered[0], expected, atol=1e-14)
        np.testing.assert_allclose(filtered[1]-filtered[0], 2, atol=1e-14)

    def test_comparison_uses_paired_subset_and_rejects_mismatched_seeds(self):
        args = parse_arguments(["--benchmarks", "KS", "--noise-ratios", ".2", "--trials", "2",
            "--method", "filtered-wsindy", "--compare-to", "baseline.md", "--output", "filtered.md"])
        baseline = [dict(name="KS", noise_ratio=.2, trial=i, seed=[i], exact_support=i == 0,
            tpr=1-i, e_inf=i, e2=i, runtime=1, threshold=.1) for i in range(2)]
        filtered = [dict(baseline[1], method="filtered-wsindy", tpr=.5)]
        original, smooth = paired_records(args, baseline, filtered)
        self.assertEqual(original, [baseline[1]])
        self.assertEqual(smooth, filtered)
        report = format_comparison_report(args, baseline, filtered)
        self.assertIn("| 0.2 | WSINDy | 1 | 0.0% | 0 | 1 | 1 |", report)
        self.assertIn("| 0.2 | Filtered WSINDy | 1 | 0.0% | 0.5 |", report)
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            paired_records(args, baseline, [dict(filtered[0], seed=[99])])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_records(args, baseline, filtered*2)

    @unittest.skipUnless((DEFAULT_DATA_DIR/"burgers.mat").exists(), "Original IB data are not cached")
    def test_filtered_execution_uses_the_same_observation_instance(self):
        args = parse_arguments(["--benchmarks", "IB", "--noise-ratios", ".2", "--trials", "1", "--offline"])
        with contextlib.redirect_stderr(io.StringIO()):
            original = run(args)[0]
            args.method = "filtered-wsindy"
            filtered = run(args)[0]
        self.assertEqual(original["seed"], filtered["seed"])
        self.assertEqual(original["noise_std"], filtered["noise_std"])
        self.assertEqual(filtered["method"], "filtered-wsindy")
        self.assertEqual(filtered["G_shape"], original["G_shape"])
        self.assertGreater(filtered["preprocessing_runtime"], 0)
        self.assertTrue(all(1 <= w <= m for w, m in zip(filtered["filter_widths"], (60, 60))))


if __name__ == "__main__":
    unittest.main()
