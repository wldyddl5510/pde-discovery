"""Run the Section 5 debiasing method on the paired IB benchmark, always time2.

The pilot is chosen by debiasing_study.rule_size (prior 0.01), without a
truth-based window sweep. Existing MSTLS/LASSO results are read, never rerun.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
FROZEN = (HERE/"methods.py").exists()
ROOT = Path.cwd() if FROZEN else HERE.parents[1]
SOURCE = HERE if FROZEN else ROOT
sys.path.insert(0, str(SOURCE))
for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import numpy as np
from debiasing_study import SPLITS, fit_system, profile_settings, rule_size
from experiments import (BENCHMARK_NOISE_RATIOS, coefficient_metrics, estimate_noise_std,
                         parse_arguments, protocol_manifest, read_records, summary_rows,
                         trial_seed, true_equation_lines)
from methods import build_wsindy_system, moving_average_pilot
from simulation_generation import load_clean_benchmark, observation_instance

REGRESSIONS = ("mstls", "lasso")
DESCRIPTION = ("Section 5 first-order polynomial expansion `(1-p)*pilot^p + p*pilot^(p-1)*U`; "
    "the LHS and linear columns use raw U. Pilot: other time-index parity only (`time2`), "
    "box moving average, periodic spatial extension and reflected time extension. "
    "Window: debiasing_study.py data-driven volume rule, prior 0.01, rounded up to odd "
    "and at least 3; no window selection using truth. IB selects 61 x 61 at every tested noise level. "
    "Author-code weak supports, state/coordinate scaling and physical coefficient restoration "
    "are retained. Both regressions use the same pilot/system and the existing penalty selection. "
    "All 100 trials at noise ratios 0, 0.2, 0.5, 0.75, 1 use the original benchmark's observation seeds.")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    from benchmark_results.lasso.penalty_levels import atomic_text
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False)+"\n")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"benchmark_results/debiasing/IB")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    return args


def manifest(args, regression):
    settings = parse_arguments(["--benchmarks", "IB", "--regression", regression, "--offline",
                                "--workers", str(args.workers), "--data-dir", str(args.data_dir)])
    settings.method = "debiased-wsindy"
    result = protocol_manifest(settings)
    result["source_sha256"]["debiasing_study.py"] = digest(SOURCE/"debiasing_study.py")
    result["source_sha256"]["run_ib.py"] = digest(Path(__file__))
    result["preprocessing"] = dict(description=DESCRIPTION, split="time2", folds=2,
        boundary=["wrap", "reflect"], prior=.01, window_rule="debiasing_study.rule_size; odd >= 3",
        pilot_selection="Observed noise estimate only; no oracle windows", time_boundary_rows="all")
    content = {key: value for key, value in result.items()
               if key not in ("protocol_id", "workers", "thread_limits")}
    result["protocol_id"] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return result


def prepare(args):
    paths, manifests = {}, {}
    for regression in REGRESSIONS:
        folder = args.output_dir/regression
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/"results.md"
        current = manifest(args, regression)
        manifest_path = path.with_suffix(".manifest.json")
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["protocol_id"] != current["protocol_id"]:
                raise ValueError(f"Saved protocol differs: {manifest_path}; use its frozen source.")
        else:
            write_json(manifest_path, current)
            snapshot = folder/"results_source"
            snapshot.mkdir()
            for filename in current["source_sha256"]:
                source = Path(__file__) if filename == "run_ib.py" else SOURCE/filename
                shutil.copyfile(source, snapshot/filename)
        (folder/"results_trials").mkdir(exist_ok=True)
        paths[regression], manifests[regression] = path, current
    return paths, manifests


_CLEAN = None


def clean_data(data_dir):
    global _CLEAN
    if _CLEAN is None:
        _CLEAN = load_clean_benchmark("IB", data_dir=data_dir, download=False)
    return _CLEAN


def make_system(data_dir, ratio, trial):
    clean = clean_data(data_dir)
    spec = clean.benchmark
    seed = trial_seed(0, "IB", ratio, trial)
    data = observation_instance(clean, ratio, seed)
    start = perf_counter()
    estimate = estimate_noise_std(data.u_observed[0])
    requested_size = rule_size(estimate, spec, .01)
    size = max(3, requested_size)
    window = tuple(min(size, 2*m+1) for m in spec.half_widths)
    pilot, info = moving_average_pilot(data.u_observed, window,
                                      boundary=("wrap", "reflect"), split=SPLITS["time2"])
    pilot_runtime = perf_counter()-start
    build, select = profile_settings(spec)
    start = perf_counter()
    system = build_wsindy_system(data.u_observed, data.spatial_grid, data.time,
        library_terms=spec.library(), pilot=pilot, **build)
    weak_runtime = perf_counter()-start
    details = dict(name="IB", noise_ratio=ratio, trial=trial, seed=seed, method="debiased-wsindy",
        split="time2", pilot_size=size, rule_size=requested_size, filter_widths=window,
        estimated_noise_std=[estimate], filter_prior=.01, boundary=["wrap", "reflect"],
        minimum_window_fraction=info["minimum_window_fraction"], window_points=info["window_points"],
        observation_sha256=array_digest(data.u_observed[0]), pilot_sha256=array_digest(pilot[0]),
        G_sha256=array_digest(system.G), b_sha256=array_digest(system.b),
        pilot_mse=float(np.mean((pilot[0]-clean.u_true[0])**2)),
        noise_std=data.noise_std, profile="authors", noise_rule="rms", test_function=system.test_function,
        shape=spec.shape, G_shape=system.G.shape, b_shape=system.b.shape,
        coefficient_scales=system.coefficient_scales.tolist(),
        state_scales=system.state_scales.tolist(), coordinate_scales=system.coordinate_scales.tolist(),
        preprocessing_runtime=pilot_runtime, weak_runtime=weak_runtime)
    return clean, data, pilot, system, select, details


def job(task):
    data_dir, ratio, trial = task
    clean, _, _, system, select, details = make_system(data_dir, ratio, trial)
    truth = clean.benchmark.truth()[:, 0]
    # Full-library sparse fits: the restricted-support OLS sanity check is not a benchmark method.
    fits = fit_system(system, slice(None), [], truth, select, REGRESSIONS)
    records = []
    for regression, fit in fits.items():
        fit = dict(fit)
        coefficients = np.asarray(fit.pop("coefficients"))[:, None]
        records.append(dict(details, regression=regression, **fit, coefficients=coefficients.tolist(),
            equations=[coefficient_metrics(coefficients, truth[:, None])],
            runtime=details["preprocessing_runtime"]+details["weak_runtime"]+fit["regression_runtime"]))
    return records


def preflight(args):
    checks = []
    baselines = {}
    for regression, folder in (("mstls", "authors"), ("lasso", "lasso/original/raw")):
        records = read_records(ROOT/"benchmark_results"/folder/"results_trials/IB.jsonl")
        baselines[regression] = {(r["noise_ratio"], r["trial"]): r for r in records}
    for ratio in BENCHMARK_NOISE_RATIOS:
        clean, data, pilot, debiased, select, details = make_system(args.data_dir, ratio, 0)
        build, _ = profile_settings(clean.benchmark)
        raw = build_wsindy_system(data.u_observed, data.spatial_grid, data.time,
                                 library_terms=clean.benchmark.library(), **build)
        np.testing.assert_array_equal(raw.b, debiased.b)
        linear = [i for i, term in enumerate(raw.terms) if sum(term.powers) < 2]
        np.testing.assert_array_equal(raw.G[:, linear], debiased.G[:, linear])
        np.testing.assert_array_equal(raw.coefficient_scales, debiased.coefficient_scales)
        fits = fit_system(raw, slice(None), [], clean.benchmark.truth()[:, 0], select, REGRESSIONS)
        for regression, fit in fits.items():
            reference = baselines[regression][(ratio, 0)]
            assert details["seed"] == reference["seed"]
            assert list(data.noise_std) == reference["noise_std"]
            np.testing.assert_array_equal(raw.state_scales, reference["state_scales"])
            np.testing.assert_allclose(np.asarray(fit["coefficients"])[:, None], reference["coefficients"],
                                       rtol=1e-10, atol=1e-12)
            assert fit["exact_support"] == reference["exact_support"]
        width = details["pilot_size"]//2
        indices = np.arange(128-width, 128+width+1)
        other = indices[indices % 2 != 128 % 2]
        np.testing.assert_allclose(pilot[0][128, 128], data.u_observed[0][np.ix_(indices, other)].mean(),
                                   rtol=1e-12, atol=1e-10)
        perturbed = data.u_observed[0].copy()
        perturbed[:, ::2] += 12345
        changed, _ = moving_average_pilot(perturbed, details["filter_widths"],
                                          boundary=("wrap", "reflect"), split=SPLITS["time2"])
        np.testing.assert_array_equal(changed[:, ::2], pilot[0][:, ::2])
        checks.append(dict(noise_ratio=ratio, pilot_size=details["pilot_size"],
            raw_mstls_and_lasso_match_saved=True, raw_lhs_and_linear_columns_retained=True,
            other_parity_average_verified=True, own_fold_perturbation_has_no_effect=True))
    write_json(args.output_dir/"preflight.json", dict(state="complete", checks=checks))


def status(path, state, completed):
    write_json(path.with_suffix(".status.json"), dict(state=state, completed=completed, requested=500,
        updated_at=datetime.now(timezone.utc).isoformat()))


def refresh(paths, manifests, records):
    from benchmark_results.lasso.penalty_levels import main as refresh_comparison
    for regression, path in paths.items():
        settings = argparse.Namespace(benchmarks=["IB"], noise_ratios=list(BENCHMARK_NOISE_RATIOS),
                                      method="debiased-wsindy")
        summaries = [dict(row, regression=regression) for row in summary_rows(settings, records[regression])]
        write_json(path.with_suffix(".summary.json"), summaries)
        if not path.exists():
            text = [f"# Debiased WSINDy (time2) + {regression.upper()}: IB", "",
                    "Status: complete; 500/500 fits. Noise ratios: 0, 0.2, 0.5, 0.75, 1; 100 trials each.", "",
                    DESCRIPTION, "", "Timings include noise estimation, pilot, weak system and the selected "
                    "regression's entire path; data/noise generation and diagnostics are excluded. The weak "
                    "system is built once per observation and its measured time is attributed to each fit.", "",
                    "## IB", "", *true_equation_lines(clean_data(ROOT/"data").benchmark), "",
                    "Observation shape `(256, 256)`; library 43 terms; weak half-widths `(60, 60)`, "
                    "strides `(5, 5)`; pilot window `(61, 61)`; split `time2`.", ""]
            path.write_text("\n".join(text)+"\n")
    refresh_comparison()


def main():
    args = arguments()
    paths, manifests = prepare(args)
    records, keys = {}, {}
    for regression, path in paths.items():
        records[regression] = read_records(path.parent/"results_trials/IB.jsonl", repair=True)
        if any(r["protocol_id"] != manifests[regression]["protocol_id"] for r in records[regression]):
            raise ValueError("Cannot mix source snapshots or protocols")
        keys[regression] = {(r["noise_ratio"], r["trial"]) for r in records[regression]}
        if len(keys[regression]) != len(records[regression]):
            raise ValueError("Duplicate saved trials")
    if args.report_only:
        if any(len(values) != 500 for values in records.values()):
            raise ValueError("The report requires all 500 trials per regression")
        refresh(paths, manifests, records)
        return
    preflight(args)
    pending = [(str(args.data_dir), ratio, trial) for ratio in BENCHMARK_NOISE_RATIOS for trial in range(100)
               if any((ratio, trial) not in keys[regression] for regression in REGRESSIONS)]
    streams = {regression: (path.parent/"results_trials/IB.jsonl").open("a")
               for regression, path in paths.items()}
    try:
        for regression, path in paths.items():
            status(path, "running", len(records[regression]))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, batch in enumerate(pool.map(job, pending), start=1):
                for record in batch:
                    regression = record["regression"]
                    key = (record["noise_ratio"], record["trial"])
                    if key in keys[regression]:
                        continue
                    record.update(protocol_id=manifests[regression]["protocol_id"], requested_workers=args.workers)
                    streams[regression].write(json.dumps(record, allow_nan=False)+"\n")
                    streams[regression].flush()
                    keys[regression].add(key)
                    records[regression].append(record)
                if done == 1 or done % 25 == 0 or done == len(pending):
                    for regression, path in paths.items():
                        status(path, "running", len(records[regression]))
                    print(f"IB time2: {done}/{len(pending)} paired observations; "
                          f"{sum(map(len, records.values()))}/1000 sparse fits", flush=True)
        for regression, path in paths.items():
            assert len(records[regression]) == 500
            status(path, "complete", 500)
        refresh(paths, manifests, records)
        print("IB time2 debiasing complete: 1000 fits and the shared tables updated.", flush=True)
    except BaseException:
        for regression, path in paths.items():
            status(path, "failed", len(records[regression]))
        raise
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == "__main__":
    main()
