"""Small scalar Section 6 EIV sanity study; never changes baseline reports.

Defaults to one paired observation per noise level and scalar PDE. All fitting
uses observed data and an estimated variance, including zero injected noise.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
FROZEN = (HERE / "eiv_study.py").exists()
ROOT = Path.cwd() if FROZEN else HERE.parents[1]
SOURCE = HERE if FROZEN else ROOT
sys.path.insert(0, str(SOURCE))
for _variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

from experiments import trial_seed
from simulation_generation import BENCHMARKS, load_clean_benchmark, observation_instance

SCALAR_BENCHMARKS = ("IB", "KdV", "KS", "HKS", "VBG")
NOISE_RATIOS = (0., .2, .5, .75, 1.)
DESCRIPTION = (
    "Empirical plug-in sanity study of the scalar Section 6 conic estimator. "
    "The variance is estimated from each observed array, including at zero injected noise. "
    "The Gaussian polynomial correction and derived tolerances use that estimate. "
    "No clean state, injected variance, true coefficient or support selects the fit. "
    "The known-variance conditional calculations do not establish 1-delta coverage for "
    "this data-dependent plug-in procedure. The specified g2 uses the operator 2-norm. "
    "Physical state and coordinates, the full polynomial library, and the benchmark's "
    "weak tests are retained; there is no silent state, column or response rescaling. "
    "Coefficient error uses the returned coefficients without thresholding or refitting. "
    "Support alone uses the solver's declared support tolerance."
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=SCALAR_BENCHMARKS,
                        default=list(SCALAR_BENCHMARKS))
    parser.add_argument("--noise-ratios", nargs="+", type=float, default=list(NOISE_RATIOS))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--support-tolerance", type=float, default=1e-7)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmark_results/eiv/scalar_sanity")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an identical protocol without rerunning saved records.")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild only this study's reports from its saved records.")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be positive.")
    if not np.isfinite(args.lam) or args.lam <= 0:
        parser.error("--lambda must be finite and positive.")
    if not np.isfinite(args.delta) or not 0 < args.delta < 1:
        parser.error("--delta must lie in (0, 1).")
    if any(not np.isfinite(x) or x < 0 or x > 1 for x in args.noise_ratios):
        parser.error("Noise ratios must be finite and lie in [0, 1].")
    if len(set(args.benchmarks)) != len(args.benchmarks) or len(set(args.noise_ratios)) != len(args.noise_ratios):
        parser.error("Benchmark names and noise ratios must be unique.")
    if any(not np.isfinite(x) or x <= 0 for x in (args.solver_tolerance, args.support_tolerance)):
        parser.error("Solver and support tolerances must be finite and positive.")
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.shape, str(values.dtype))).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serializable(value), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def protocol(args):
    sources = {name: SOURCE / name for name in
               ("methods.py", "eiv_study.py", "experiments.py", "simulation_generation.py", "requirements.txt")}
    sources["run_sanity.py"] = Path(__file__)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "clarabel")}
    benchmarks = {}
    for name in args.benchmarks:
        spec = BENCHMARKS[name]
        data_file = args.data_dir / spec.filename
        benchmarks[name] = dict(
            suite=spec.suite, filename=spec.filename,
            data_sha256=sha256(data_file) if data_file.exists() else None,
            shape=spec.shape, noise_rule=spec.noise_rule,
            lhs_components=spec.lhs_components, lhs_time_order=spec.lhs_time_order,
            half_widths=spec.half_widths, strides=spec.strides,
            test_degrees=spec.degrees if spec.test_function == "polynomial" else None,
            test_function=spec.test_function, library=[asdict(t) for t in spec.library()],
        )
    body = serializable(dict(
        schema_version=1, method="scalar Section 6 Gaussian-corrected conic EIV",
        description=DESCRIPTION, benchmarks=benchmarks, benchmark_order=args.benchmarks,
        noise_ratios=args.noise_ratios, trials=args.trials, seed=0,
        seed_rule="experiments.trial_seed(0, name, noise_ratio, trial)",
        lam=args.lam, delta=args.delta, sigma2=None, mu=None, tau=None, M=None, M1=None,
        solver_tolerance=args.solver_tolerance, support_tolerance=args.support_tolerance,
        equilibrate=False, download=False, variance_source="observations only",
        truth_use="Only after fitting: coefficient/support scoring and weak-residual diagnostics",
        zero_optimality="If tau >= ||y||_inf, (w,t)=(0,0) is feasible and attains the nonnegative objective's minimum zero.",
        numerical_versions=versions, python=platform.python_version(),
        source_sha256={name: sha256(path) for name, path in sources.items()},
    ))
    body["protocol_id"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body, sources


def prepare(args):
    manifest, sources = protocol(args)
    path = args.output_dir / "results.manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["protocol_id"] != manifest["protocol_id"]:
            raise ValueError(f"Protocol or source changed. Use a fresh --output-dir: {path}")
        if not args.resume:
            raise ValueError(f"Study already exists; use --resume or a fresh --output-dir: {path}")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(f"Refusing to mix an existing directory with a new study: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = args.output_dir / "results_source"
        snapshot.mkdir()
        for name, source in sources.items():
            shutil.copyfile(source, snapshot / name)
        write_json(path, dict(manifest, created_at=datetime.now(timezone.utc).isoformat()))
    (args.output_dir / "results_trials").mkdir(exist_ok=True)
    return manifest


def score(coefficients, support, truth):
    """Score coefficients verbatim; score support using the separately returned mask."""
    coefficients, support, truth = np.asarray(coefficients), np.asarray(support, dtype=bool), np.asarray(truth)
    if coefficients.shape != truth.shape or support.shape != truth.shape:
        raise ValueError("Scalar coefficient, support and truth shapes must agree.")
    if not np.isfinite(coefficients).all():
        raise ValueError("Cannot score nonfinite coefficients.")
    actual = truth != 0
    tp, fn, fp = (int(np.count_nonzero(mask)) for mask in
                  (support & actual, ~support & actual, support & ~actual))
    return dict(e2=float(np.linalg.norm(coefficients - truth) / np.linalg.norm(truth)),
                e_inf=float(np.max(np.abs((coefficients - truth)[actual] / truth[actual]))),
                tp=tp, fn=fn, fp=fp, tpr=tp / (tp + fn + fp),
                exact_support=fn == fp == 0, selected_terms=int(np.count_nonzero(support)))


def run_trial(args, name, ratio, trial, clean=None):
    from eiv_study import fit_eiv

    started = perf_counter()
    seed = trial_seed(0, name, ratio, trial)
    record = dict(name=name, noise_ratio=ratio, trial=trial, seed=seed,
                  status="failed", success=False, support_tolerance=args.support_tolerance,
                  solver_tolerance=args.solver_tolerance, lam=args.lam, delta=args.delta)
    phase = "load observations"
    try:
        clean = clean if clean is not None else load_clean_benchmark(name, data_dir=args.data_dir, download=False)
        spec = clean.benchmark
        if len(spec.component_names) != 1 or len(spec.lhs_components) != 1:
            raise ValueError("This study supports scalar PDEs only.")
        data = observation_instance(clean, ratio, seed)
        terms = spec.library()
        record.update(noise_rule=spec.noise_rule,
                      observation_sha256=[array_digest(v) for v in data.u_observed],
                      test_function=spec.test_function, shape=spec.shape)
        phase = "fit"
        fit_started = perf_counter()
        result, system, tolerance = fit_eiv(
            data.u_observed[spec.lhs_components[0]], data.spatial_grid, data.time,
            library_terms=terms, half_widths=spec.half_widths, strides=spec.strides,
            test_degrees=spec.degrees if spec.test_function == "polynomial" else None,
            test_function=spec.test_function, lhs_time_order=spec.lhs_time_order,
            delta=args.delta, lam=args.lam, sigma2=None, mu=None, tau=None, M=None, M1=None,
            solver_tolerance=args.solver_tolerance, support_tolerance=args.support_tolerance,
            equilibrate=False,
        )
        record["fit_runtime"] = perf_counter() - fit_started
        y = np.asarray(system.b).reshape(-1)
        y_inf = float(np.linalg.norm(y, ord=np.inf))
        tau, mu = float(tolerance["tau"]), float(tolerance["mu"])
        record.update(
            status=result.status, solver_status=result.status, success=bool(result.success),
            coefficients=result.coefficients, support=result.support,
            t=result.t, objective=result.objective, diagnostics=result.diagnostics,
            tolerance=tolerance, sigma2=tolerance["sigma2"], mu=mu, tau=tau,
            y_inf=y_inf, tau_over_y_inf=tau / y_inf if y_inf else None,
            zero_analytically_optimal=bool(tau >= y_inf),
            G_shape=system.G.shape, b_shape=system.b.shape,
            G_sha256=array_digest(system.G), b_sha256=array_digest(system.b),
            coefficient_scales=system.coefficient_scales,
            state_scales=system.state_scales, coordinate_scales=system.coordinate_scales,
        )
        # Truth and the injected variance enter only after the estimator returns.
        phase = "post-fit diagnostics"
        truth = spec.truth(terms)[:, 0]
        true_norm = float(np.linalg.norm(truth))
        true_residual = float(np.linalg.norm(system.G @ truth - y, ord=np.inf))
        true_bound = mu * true_norm + tau
        record.update(
            **score(result.coefficients, result.support, truth),
            injected_noise_std=data.noise_std,
            truth_coefficients=truth,
            truth_residual_inf=true_residual,
            truth_minimal_t=true_norm,
            truth_bound_at_minimal_t=true_bound,
            truth_feasible_at_minimal_t=bool(true_residual <= true_bound),
            truth_margin_at_minimal_t=true_bound - true_residual,
        )
    except Exception as error:
        record.update(status="failed", success=False, failed_phase=phase,
                      error_type=type(error).__name__, error=str(error))
    record["runtime"] = perf_counter() - started
    return serializable(record)


def read_records(folder, manifest):
    records, keys = [], set()
    for name in manifest["benchmark_order"]:
        path = folder / "results_trials" / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record["name"], record["noise_ratio"], record["trial"])
            if record["protocol_id"] != manifest["protocol_id"] or key in keys:
                raise ValueError(f"Mismatched or duplicate saved record: {path}")
            if key[0] != name or key[1] not in manifest["noise_ratios"] or not 0 <= key[2] < manifest["trials"]:
                raise ValueError(f"Record is outside the saved protocol: {path}")
            keys.add(key)
            records.append(record)
    return records


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.4g}" if isinstance(value, (int, float)) else str(value).replace("|", "\\|")


def refresh(folder, manifest, records, state):
    summaries = []
    for name in manifest["benchmark_order"]:
        for ratio in manifest["noise_ratios"]:
            group = [r for r in records if r["name"] == name and r["noise_ratio"] == ratio]
            good = [r for r in group if r["success"]]
            summaries.append(dict(name=name, noise_ratio=ratio, completed=len(group), successful=len(good),
                failures=len(group) - len(good), exact_support_count=sum(r["exact_support"] for r in good),
                mean_e2_successful=mean([r.get("e2") for r in good]),
                mean_e_inf_successful=mean([r.get("e_inf") for r in good]),
                mean_sigma2=mean([r.get("sigma2") for r in group]),
                mean_mu=mean([r.get("mu") for r in group]), mean_tau=mean([r.get("tau") for r in group]),
                mean_y_inf=mean([r.get("y_inf") for r in group]),
                mean_tau_over_y_inf=mean([r.get("tau_over_y_inf") for r in group]),
                zero_analytically_optimal_count=sum(r.get("zero_analytically_optimal", False) for r in group)))
    requested = len(manifest["benchmark_order"]) * len(manifest["noise_ratios"]) * manifest["trials"]
    write_json(folder / "results.summary.json", summaries)
    write_json(folder / "results.status.json", dict(state=state, completed=len(records), requested=requested,
               successful=sum(r["success"] for r in records), updated_at=datetime.now(timezone.utc).isoformat()))
    lines = ["# Scalar Section 6 EIV sanity study", "",
             f"Status: {state}; {len(records)}/{requested} attempted fits. "
             f"Trials per PDE/noise level: {manifest['trials']}.", "", DESCRIPTION, "",
             f"λ = {manifest['lam']:g}, δ = {manifest['delta']:g}; "
             f"solver tolerance = {manifest['solver_tolerance']:g}; "
             f"support tolerance = {manifest['support_tolerance']:g}.", "",
             "When τ ≥ ‖y‖∞, (w,t)=(0,0) is feasible and has objective zero. "
             "Because ‖w‖₁ + λt is nonnegative, the zero solution is globally optimal. "
             "A zero estimate in this regime is a consequence of these tolerances, not a solver failure.", "",
             "`Exact support` uses the returned support mask. E2 is ‖w−w*‖₂/‖w*‖₂ "
             "from the unmodified conic estimate. `Truth feasible` tests the residual bound at "
             "t=‖w*‖₂ after fitting and is diagnostic only. Failed fits remain in the trial records; "
             "summary errors average successful fits only. Nonfinite values are saved as JSON null.", "",
             "| PDE | Noise | Trial | Status | σ̂² | μ | τ | ‖y‖∞ | τ/‖y‖∞ | Zero optimal | "
             f"Exact support (tol={manifest['support_tolerance']:g}) | E2 | Truth feasible |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|"]
    order = {name: i for i, name in enumerate(manifest["benchmark_order"])}
    for record in sorted(records, key=lambda r: (order[r["name"]], r["noise_ratio"], r["trial"])):
        fields = ("name", "noise_ratio", "trial", "status", "sigma2", "mu", "tau", "y_inf",
                  "tau_over_y_inf", "zero_analytically_optimal", "exact_support", "e2", "truth_feasible_at_minimal_t")
        lines.append("| " + " | ".join(fmt(record.get(field)) for field in fields) + " |")
    errors = [r for r in records if r.get("error")]
    if errors:
        lines.extend(["", "Recorded failures:", ""])
        lines.extend(f"- {r['name']}, noise {r['noise_ratio']:g}, trial {r['trial']}: "
                     f"{r['error_type']} during {r['failed_phase']}: {r['error']}" for r in errors)
    lines.extend(["", "Reproducibility: [manifest](results.manifest.json), "
                  "[summary](results.summary.json), per-PDE JSONL in `results_trials/`, "
                  "and frozen numerical sources in `results_source/`. "
                  "The manifest records source and dataset SHA-256 hashes. "
                  "Original and consistency datasets retain their existing noise conventions and seeds.", ""])
    (folder / "results.md").write_text("\n".join(lines))


def main(argv=None):
    args = arguments(argv)
    if args.report_only:
        manifest = json.loads((args.output_dir / "results.manifest.json").read_text())
        records = read_records(args.output_dir, manifest)
        requested = len(manifest["benchmark_order"]) * len(manifest["noise_ratios"]) * manifest["trials"]
        refresh(args.output_dir, manifest, records,
                "complete" if len(records) == requested else "partial")
        return 0
    manifest = prepare(args)
    records = read_records(args.output_dir, manifest)
    keys = {(r["name"], r["noise_ratio"], r["trial"]) for r in records}
    refresh(args.output_dir, manifest, records, "running")
    for name in args.benchmarks:
        clean = None
        try:
            clean = load_clean_benchmark(name, data_dir=args.data_dir, download=False)
        except Exception:
            pass  # Each requested trial records the load error below.
        for ratio in args.noise_ratios:
            for trial in range(args.trials):
                if (name, ratio, trial) in keys:
                    continue
                record = run_trial(args, name, ratio, trial, clean)
                record["protocol_id"] = manifest["protocol_id"]
                with (args.output_dir / "results_trials" / f"{name}.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                records.append(record)
                refresh(args.output_dir, manifest, records, "running")
                print(f"{name} noise={ratio:g} trial={trial}: {record['status']}; "
                      f"E2={fmt(record.get('e2'))}, tau/||y||inf={fmt(record.get('tau_over_y_inf'))}", flush=True)
    refresh(args.output_dir, manifest, records, "complete")
    return 0 if all(r["success"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
