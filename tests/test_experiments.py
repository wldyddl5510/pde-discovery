"""Check that unavailable or failed fits remain explicit in experiment reports."""

import contextlib
import io
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

import experiments


def arguments(*options):
    with patch("sys.argv", ["experiments.py", "--nx", "16", "--ny", "16", "--nt", "12",
                            "--repeats", "1", *options]):
        return experiments.parse_arguments()


class ExperimentTests(unittest.TestCase):
    def test_noiseless_mle_is_skipped_and_positive_noise_is_fitted(self):
        args = arguments("--methods", "wendy-mle-lasso")
        with patch.object(experiments, "wendy_mle", return_value=np.zeros(100)) as estimator:
            with contextlib.redirect_stdout(io.StringIO()):
                results = experiments.run_experiments(args)
        self.assertEqual(results[0]["wendy-mle-lasso"]["status"], "not_applicable")
        self.assertEqual(results[1]["wendy-mle-lasso"]["status"], "ok")
        self.assertEqual(estimator.call_count, 2)  # One warm-up and one timed noisy fit.
        self.assertGreater(estimator.call_args.kwargs["noise_std"], 0)
        self.assertEqual(estimator.call_args.kwargs["max_iter"], args.mle_max_iter)

    def test_failed_fit_does_not_prevent_other_measurements(self):
        args = arguments("--methods", "wsindy-lasso", "wendy-lasso", "--noise-ratios", "1")
        with patch.object(experiments, "wsindy", side_effect=RuntimeError("did not converge")):
            with patch.object(experiments, "wendy", return_value=np.zeros(100)):
                with contextlib.redirect_stdout(io.StringIO()):
                    results = experiments.run_experiments(args)
        failed = results[1]["wsindy-lasso"]
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["squared_error"])
        self.assertIsNone(failed["runtime_seconds"])
        self.assertEqual(failed["failed_call"], "warm-up")
        self.assertEqual(results[1]["wendy-lasso"]["status"], "ok")

    def test_warning_and_undefined_mle_are_visible_in_report(self):
        def normality_stop(*inputs, **settings):
            warnings.warn("normality stop; fixed-point tolerance was not met", RuntimeWarning)
            return np.zeros(100)

        args = arguments("--methods", "wendy-lasso", "wendy-mle-lasso", "--noise-ratios", "0")
        with patch.object(experiments, "wendy", side_effect=normality_stop):
            with contextlib.redirect_stdout(io.StringIO()):
                results = experiments.run_experiments(args)
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "results.md"
            experiments.write_report(args, results)
            report = args.output.read_text()
        self.assertIn("1.730000e+00 * | N/A", report)
        self.assertIn("fixed-point tolerance was not met", report)
        self.assertEqual(len(results[0]["wendy-lasso"]["warnings"]), 1)

    @unittest.skipUnless(hasattr(signal, "setitimer"), "Unix interval timer required")
    def test_timeout_is_propagated_and_timer_is_cleared(self):
        previous_handler = signal.getsignal(signal.SIGALRM)

        def exceed_deadline():
            signal.raise_signal(signal.SIGALRM)

        with self.assertRaises(TimeoutError):
            experiments.call_estimator(exceed_deadline, (), {}, 10)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous_handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
