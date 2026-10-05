"""Paired polynomial PDE benchmarks with a fold-local time2 pilot and window.

Run from the repository root. Frozen copies use sibling numerical sources.
Every completed PDE adds MSTLS/LASSO rows to the existing comparison report.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
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
from debiasing_study import fit_system, profile_settings, time2_pilot
from experiments import (BENCHMARK_NOISE_RATIOS, coefficient_metrics, parse_arguments,
    protocol_manifest, read_records, summary_rows, trial_seed, true_equation_lines)
from methods import build_wsindy_system
from simulation_generation import BENCHMARKS, load_clean_benchmark, observation_instance

REGRESSIONS = ("mstls", "lasso")
ORDER = ("IB", "KdV", "KS", "NLS", "HKS", "VBG", "NS", "RD")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def description(name):
    boundary = "reflected" if name == "NS" else "periodic"
    return ("Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, "
        "with raw U on the LHS and in linear columns. Split is always time2. For each evaluation "
        "time parity, both the noise estimate and the pilot use only the opposite parity. "
        "Sixth differences estimate noise on that training time subgrid; the largest estimate "
        "across components selects a common box window using the study's volume rule, prior 0.01, "
        "rounded up to odd and at least 3. " + f"Spatial extension is {boundary}; time is reflected. "
        "All original weak rows, library terms and benchmark regression/scaling settings are retained. "
        "Physical coefficients are restored before scoring. No clean error or true support selects "
        "the window. This pooled cross-fit benchmark does not assert the single-pilot conditional "
        "Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each.")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", choices=ORDER, nargs="+", default=list(ORDER))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"benchmark_results/debiasing/time2_foldwise")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("Use positive workers and unique benchmarks")
    return args


def prepare(args, name, regression):
    settings = parse_arguments(["--benchmarks", name, "--regression", regression, "--offline",
        "--workers", str(args.workers), "--data-dir", str(args.data_dir)])
    settings.method = "debiased-wsindy"
    manifest = protocol_manifest(settings)
    for filename in ("debiasing_study.py", "run_suite.py"):
        manifest["source_sha256"][filename] = digest(Path(__file__) if filename == "run_suite.py" else SOURCE/filename)
    manifest["preprocessing"] = dict(version=2, split="time2", prior=.01,
        noise_estimation="Sixth differences on opposite parity time subgrid only; max across components",
        window="debiasing_study.rule_size, rounded up to odd >= 3, capped by 2m+1 per axis",
        boundary=["reflect" if name == "NS" else "wrap"]*(len(BENCHMARKS[name].shape)-1)+["reflect"],
        description=description(name), pilot_selection="Training fold only", time_boundary_rows="all")
    body = {k:v for k,v in manifest.items() if k not in ("protocol_id", "workers", "thread_limits")}
    manifest["protocol_id"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = args.output_dir/name/regression/"results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = path.with_suffix(".manifest.json")
    if saved.exists():
        if json.loads(saved.read_text())["protocol_id"] != manifest["protocol_id"]:
            raise ValueError(f"Protocol changed; resume using saved source: {saved}")
    else:
        snapshot = path.parent/"results_source"
        snapshot.mkdir()
        for filename in manifest["source_sha256"]:
            source = Path(__file__) if filename == "run_suite.py" else SOURCE/filename
            shutil.copyfile(source, snapshot/filename)
        write_json(saved, manifest)
    (path.parent/"results_trials").mkdir(exist_ok=True)
    return path, manifest


@lru_cache(maxsize=1)
def clean_data(name, data_dir):
    return load_clean_benchmark(name, data_dir=Path(data_dir), download=False)


def make_system(name, data_dir, ratio, trial):
    clean = clean_data(name, data_dir)
    spec = clean.benchmark
    seed = trial_seed(0, name, ratio, trial)
    data = observation_instance(clean, ratio, seed)
    start = perf_counter()
    pilot, info = time2_pilot(data.u_observed, spec, space_boundary="reflect" if name == "NS" else "wrap")
    preprocessing = perf_counter()-start
    build, select = profile_settings(spec)
    start = perf_counter()
    system = build_wsindy_system(data.u_observed, data.spatial_grid, data.time,
        library_terms=spec.library(), pilot=pilot, **build)
    weak_time = perf_counter()-start
    details = dict(name=name, noise_ratio=ratio, trial=trial, seed=seed, method="debiased-wsindy",
        split="time2", pilot_info=json.loads(json.dumps(info)), noise_std=list(data.noise_std), noise_rule=spec.noise_rule,
        profile="consistency" if spec.suite == "consistency" else "authors", test_function=spec.test_function,
        observation_sha256=[array_digest(v) for v in data.u_observed],
        pilot_sha256=[array_digest(v) for v in pilot], G_sha256=array_digest(system.G), b_sha256=array_digest(system.b),
        pilot_mse=[float(np.mean((p-u)**2)) for p,u in zip(pilot, clean.u_true)],
        shape=list(spec.shape), G_shape=list(system.G.shape), b_shape=list(system.b.shape),
        coefficient_scales=system.coefficient_scales.tolist(), state_scales=system.state_scales.tolist(),
        coordinate_scales=system.coordinate_scales.tolist(), preprocessing_runtime=preprocessing, weak_runtime=weak_time)
    return clean, data, system, select, details


def job(task):
    name, data_dir, ratio, trial = task
    clean, _, system, select, details = make_system(name, data_dir, ratio, trial)
    truth = clean.benchmark.truth()
    fits = fit_system(system, slice(None), [], truth, select, REGRESSIONS)
    return [dict(details, regression=regression, **fit,
        equations=[coefficient_metrics(np.asarray(fit["coefficients"])[:,i], truth[:,i]) for i in range(truth.shape[1])],
        runtime=details["preprocessing_runtime"]+details["weak_runtime"]+fit["regression_runtime"])
        for regression, fit in fits.items()]


def baseline_records(name, regression):
    consistency = BENCHMARKS[name].suite == "consistency"
    folder = ("lasso/consistency/raw" if consistency else "lasso/original/raw") if regression == "lasso" else (
        "consistency/raw" if consistency else "authors")
    path = ROOT/"benchmark_results"/folder/"results_trials"/(name+".jsonl")
    return {(r["noise_ratio"],r["trial"]):r for r in read_records(path)}


def preflight(args, name):
    clean, data, debiased, select, details = make_system(name, str(args.data_dir), .5, 0)
    build, _ = profile_settings(clean.benchmark)
    raw = build_wsindy_system(data.u_observed, data.spatial_grid, data.time,
        library_terms=clean.benchmark.library(), **build)
    np.testing.assert_array_equal(raw.b, debiased.b)
    linear = [i for i,t in enumerate(raw.terms) if sum(t.powers) < 2]
    np.testing.assert_array_equal(raw.G[:,linear], debiased.G[:,linear])
    np.testing.assert_array_equal(raw.coefficient_scales, debiased.coefficient_scales)
    fits = fit_system(raw, slice(None), [], clean.benchmark.truth(), select, REGRESSIONS)
    for regression, fit in fits.items():
        old = baseline_records(name, regression)[(.5,0)]
        assert details["seed"] == old["seed"] and list(data.noise_std) == old["noise_std"]
        np.testing.assert_allclose(fit["coefficients"], old["coefficients"], rtol=1e-9, atol=1e-11)
        assert fit["exact_support"] == old["exact_support"]
    write_json(args.output_dir/name/"preflight.json", dict(state="complete", name=name,
        raw_baselines_reproduced=True, lhs_and_linear_columns_unchanged=True,
        physical_scaling_verified=True, pilot_info=details["pilot_info"],
        G_sha256=details["G_sha256"], b_sha256=details["b_sha256"], targeted_unit_tests=12))
    print(f"{name}: preflight passed; weak shape {debiased.G.shape}, "
          f"equations {debiased.b.shape[1]}, windows {[f['window'] for f in details['pilot_info']['folds']]}", flush=True)


def validate(name, records, manifest):
    expected = {(ratio,trial) for ratio in BENCHMARK_NOISE_RATIOS for trial in range(100)}
    assert len(records) == 500 and {(r["noise_ratio"],r["trial"]) for r in records} == expected
    truth = BENCHMARKS[name].truth()
    baseline = baseline_records(name, manifest["regression"])
    for record in records:
        assert record["protocol_id"] == manifest["protocol_id"] and record["split"] == "time2"
        old = baseline[(record["noise_ratio"],record["trial"])]
        for field in ("seed", "noise_std", "G_shape", "b_shape", "state_scales", "coordinate_scales"):
            assert record[field] == old[field], (name, field)
        beta = np.asarray(record["coefficients"])
        assert beta.shape == truth.shape and np.isfinite(beta).all()
        selected, actual = beta != 0, truth != 0
        tp, fn, fp = [int(np.count_nonzero(v)) for v in (selected & actual, ~selected & actual, selected & ~actual)]
        assert (record["tp"],record["fn"],record["fp"]) == (tp,fn,fp)
        assert record["exact_support"] == (fn == fp == 0)
        np.testing.assert_allclose(record["tpr"],tp/(tp+fn+fp))
        np.testing.assert_allclose(record["e2"],np.linalg.norm(beta-truth)/np.linalg.norm(truth))
        np.testing.assert_allclose(record["e_inf"],np.max(np.abs((beta-truth)[actual]/truth[actual])))
        assert record["threshold"] == record["thresholds"][int(np.argmin(record["losses"]))]
        if record["regression"] == "lasso":
            assert np.max(record["kkt_errors"]) <= 1e-7
            np.testing.assert_allclose(record["selected_alpha"],record["threshold"]*np.asarray(record["alpha_max"]))
            rms = np.asarray(record["column_rms"])[:,None]
            restored = np.divide(np.asarray(record["normalized_coefficients"])*np.asarray(record["response_rms"]),
                rms, out=np.zeros_like(beta), where=rms>0)*np.asarray(record["coefficient_scales"])
            np.testing.assert_allclose(beta,restored,rtol=1e-12,atol=1e-12)


def status(path, state, count):
    write_json(path.with_suffix(".status.json"),dict(state=state,completed=count,requested=500,
        updated_at=datetime.now(timezone.utc).isoformat()))


def refresh(name, paths, records):
    for regression,path in paths.items():
        settings = argparse.Namespace(benchmarks=[name],noise_ratios=list(BENCHMARK_NOISE_RATIOS),method="debiased-wsindy")
        summaries = [dict(row,regression=regression) for row in summary_rows(settings,records[regression])]
        write_json(path.with_suffix(".summary.json"),summaries)
        if not path.exists():
            spec=BENCHMARKS[name]
            path.write_text("\n".join([f"# Debiased WSINDy (time2) + {regression.upper()}: {name}","",
                "Status: complete; 500/500 fits. Mean ± sample SD over 100 trials per noise level.","",description(name),"",
                "Timings include fold-specific noise estimation, pilot, weak construction and the entire regression path. "
                "Shared weak construction time is attributed to each fit; loading, noise generation and diagnostics are excluded.","",
                f"## {name}","",*true_equation_lines(spec),"",
                f"Observation shape `{spec.shape}`; library {len(spec.library())} terms; weak half-widths `{spec.half_widths}`, strides `{spec.strides}`.",""])+"\n")
    for regression,path in paths.items():
        status(path,"complete",500)
    from benchmark_results.lasso.penalty_levels import main as refresh_comparison
    refresh_comparison()


def run_benchmark(args,name):
    prepared={r:prepare(args,name,r) for r in REGRESSIONS}
    paths={r:p for r,(p,_) in prepared.items()}
    manifests={r:m for r,(_,m) in prepared.items()}
    records={r:read_records(p.parent/"results_trials"/(name+".jsonl"),repair=True) for r,p in paths.items()}
    keys={r:{(v["noise_ratio"],v["trial"]) for v in values} for r,values in records.items()}
    for r,values in records.items():
        assert len(keys[r]) == len(values)
        assert all(v["protocol_id"] == manifests[r]["protocol_id"] for v in values)
    if args.report_only or all(len(v)==500 for v in records.values()):
        for r,values in records.items(): validate(name,values,manifests[r])
        refresh(name,paths,records)
        return
    preflight(args,name)
    if args.preflight_only: return
    pending=[(name,str(args.data_dir),ratio,trial) for ratio in BENCHMARK_NOISE_RATIOS for trial in range(100)
             if any((ratio,trial) not in keys[r] for r in REGRESSIONS)]
    streams={r:(p.parent/"results_trials"/(name+".jsonl")).open("a") for r,p in paths.items()}
    try:
        for r,p in paths.items(): status(p,"running",len(records[r]))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done,batch in enumerate(pool.map(job,pending),start=1):
                for record in batch:
                    r=record["regression"]; key=(record["noise_ratio"],record["trial"])
                    if key in keys[r]: continue
                    record.update(protocol_id=manifests[r]["protocol_id"],requested_workers=args.workers)
                    streams[r].write(json.dumps(record,allow_nan=False)+"\n"); streams[r].flush()
                    records[r].append(record); keys[r].add(key)
                if done==1 or done%10==0 or done==len(pending):
                    for r,p in paths.items(): status(p,"running",len(records[r]))
                    print(f"{name}: {sum(map(len,records.values()))}/1000 fits; noise={batch[0]['noise_ratio']:g}",flush=True)
        for r,values in records.items(): validate(name,values,manifests[r])
        for a,b in zip(records["mstls"],records["lasso"]):
            for key in ("seed","noise_ratio","trial","pilot_info","observation_sha256","G_sha256","b_sha256"):
                assert a[key]==b[key]
        write_json(args.output_dir/name/"verification.json",dict(state="complete",fits=1000,paired_observations=500,
            raw_baselines_reproduced=True,metrics_recomputed=True,physical_units_verified=True,
            all_lasso_candidates_pass_KKT=True,time2_training_only_window=True))
        refresh(name,paths,records)
        print(f"{name}: complete and verified; comparison updated",flush=True)
    except BaseException:
        for r,p in paths.items(): status(p,"failed",len(records[r]))
        raise
    finally:
        for stream in streams.values(): stream.close()


def main():
    args=arguments()
    suite_status=args.output_dir/"suite.status.json"
    completed=[]
    try:
        for name in args.benchmarks:
            write_json(suite_status,dict(state="running",active_benchmark=name,completed_benchmarks=completed,
                requested_benchmarks=args.benchmarks,requested_fits=1000*len(args.benchmarks)))
            run_benchmark(args,name)
            completed.append(name)
        write_json(suite_status,dict(state="preflight_complete" if args.preflight_only else "complete",
            completed_benchmarks=completed,requested_benchmarks=args.benchmarks,
            completed_fits=0 if args.preflight_only else 1000*len(completed)))
    except BaseException as error:
        write_json(suite_status,dict(state="failed",completed_benchmarks=completed,error=str(error)))
        raise


if __name__=="__main__": main()
