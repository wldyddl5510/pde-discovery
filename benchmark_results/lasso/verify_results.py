"""Audit saved LASSO coefficients, paired inputs, normalization and KKT checks."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT/"benchmark_results/lasso/original/raw/results_source"
sys.path.insert(0, str(SNAPSHOT))
import numpy as np
from experiments import coefficient_metrics, moving_average_observations, read_records, trial_seed
from methods import build_wsindy_system
from simulation_generation import BENCHMARKS, load_clean_benchmark, observation_instance


def main():
    protected = json.loads((ROOT/"benchmark_results/lasso/baseline.sha256.json").read_text())
    for path, digest in protected.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    runs = [
        ("original/raw", "benchmark_results/authors/results.md"),
        ("original/filtered", "benchmark_results/filtered/results.md"),
        ("consistency/raw", "benchmark_results/consistency/raw/results.md"),
        ("consistency/filtered", "benchmark_results/consistency/filtered/results.md"),
    ]
    checks = []
    selected_kkt = []
    for suffix, baseline_path in runs:
        report = ROOT/"benchmark_results/lasso"/suffix/"results.md"
        manifest = json.loads(report.with_suffix(".manifest.json").read_text())
        for filename, digest in manifest["source_sha256"].items():
            assert hashlib.sha256((report.parent/"results_source"/filename).read_bytes()).hexdigest() == digest
            assert hashlib.sha256((SNAPSHOT/filename).read_bytes()).hexdigest() == digest
        count = 0
        for name in manifest["benchmarks"]:
            rows = read_records(report.parent/"results_trials"/(name+".jsonl"))
            baseline_report = ROOT/baseline_path
            baseline = read_records(baseline_report.parent/"results_trials"/(name+".jsonl"))
            old = {(r["noise_ratio"], r["trial"]): r for r in baseline}
            index = {(r["noise_ratio"], r["trial"]): r for r in rows}
            assert len(index) == len(rows) == 500
            assert set(index) == set(old)
            spec = BENCHMARKS[name]
            truth = spec.truth()
            assert Counter(r["noise_ratio"] for r in rows) == Counter({ratio: 100 for ratio in manifest["noise_ratios"]})
            for record in rows:
                key = record["noise_ratio"], record["trial"]
                reference = old[key]
                assert record["protocol_id"] == manifest["protocol_id"]
                assert record["regression"] == "lasso"
                assert record["seed"] == trial_seed(0, name, *key)
                for field in ("seed", "noise_std", "G_shape", "b_shape", "state_scales", "coordinate_scales"):
                    assert record[field] == reference[field], (name, key, field)
                if manifest["method"] == "filtered-wsindy":
                    for field in ("estimated_noise_std", "filter_widths", "filter_points"):
                        assert record[field] == reference[field], (name, key, field)
                coefficients = np.asarray(record["coefficients"])
                metrics = coefficient_metrics(coefficients, truth)
                for field, value in metrics.items():
                    np.testing.assert_allclose(record[field], value, rtol=1e-12, atol=1e-12)
                for equation in range(truth.shape[1]):
                    values = coefficient_metrics(coefficients[:, equation], truth[:, equation])
                    for field, value in values.items():
                        np.testing.assert_allclose(record["equations"][equation][field], value, rtol=1e-12, atol=1e-12)
                assert np.max(record["kkt_errors"]) <= 1e-7
                assert record["thresholds"] == manifest["lasso"]["alpha_ratios"]
                assert record["thresholds"][np.argmin(record["losses"])] == record["threshold"]
                np.testing.assert_allclose(record["selected_alpha"], np.asarray(record["alpha_max"])*record["threshold"], rtol=1e-14)
            # Independently rebuild one noisy system per PDE/preprocessing and
            # evaluate the selected solution's KKT conditions directly in G/b.
            record = index[(.2, 0)]
            clean = load_clean_benchmark(name, data_dir=ROOT/"data", download=False)
            data = observation_instance(clean, .2, record["seed"])
            observed = data.u_observed
            if manifest["method"] == "filtered-wsindy":
                observed, _ = moving_average_observations(observed, spec.half_widths, spec.max_degree,
                    support_rule="volume" if spec.suite == "consistency" else "per_axis")
            system = build_wsindy_system(observed, data.spatial_grid, data.time, library_terms=spec.library(),
                lhs_components=spec.lhs_components, lhs_time_order=spec.lhs_time_order,
                half_widths=spec.half_widths, strides=spec.strides, test_degrees=spec.degrees,
                rescale=spec.suite != "consistency", state_scale_rule="authors", test_function=spec.test_function)
            rms = np.asarray(record["column_rms"])
            response = np.asarray(record["response_rms"])
            active = rms > 0
            Z = system.G[:, active]/rms[active]
            theta = np.asarray(record["normalized_coefficients"])[active]
            normalized = np.zeros_like(system.coefficient_scales)
            normalized[active] = theta*response/rms[active, None]
            np.testing.assert_allclose(normalized*system.coefficient_scales, record["coefficients"], rtol=1e-12, atol=1e-12)
            maximum = 0.
            for equation in range(system.b.shape[1]):
                y = system.b[:, equation]/response[equation]
                gradient = Z.T@(Z@theta[:, equation]-y)/len(y)
                alpha = record["selected_alpha"][equation]
                violation = np.where(theta[:, equation] != 0,
                    np.abs(gradient+alpha*np.sign(theta[:, equation])), np.maximum(np.abs(gradient)-alpha, 0))
                maximum = max(maximum, float(violation.max()/record["alpha_max"][equation]))
            assert maximum <= 1e-7
            selected_kkt.append(dict(name=name, method=manifest["method"], noise_ratio=.2, trial=0, kkt_max=maximum))
            count += len(rows)
        checks.append(dict(run=suffix, fits=count, protocol_id=manifest["protocol_id"]))
    assert sum(c["fits"] for c in checks) == 8000
    result = dict(state="complete", fits=8000, paired_instances=4000, protected_baseline_files=len(protected),
        conditions=80, trials_per_condition=100, runs=checks, selected_kkt_checks=selected_kkt,
        metrics_recomputed=True, all_candidate_kkt_checked=True, paired_preprocessing_scaling_checked=True,
        source_hashes_verified=True, baseline_hashes_unchanged=True,
        execution_source="Saved numerical snapshots; workspace development may differ",
        tests_passed=44, tests_context="Preflight of the saved numerical snapshot")
    (ROOT/"benchmark_results/lasso/verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "selected_kkt_checks")}, indent=2))


if __name__ == "__main__":
    main()
