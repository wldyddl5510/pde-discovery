"""Section 5 debiasing sanity check: raw, filtered and debiased WSINDy on one PDE.

Each trial draws one Gaussian observation set and, for every pilot window
size and split rule, fits a box moving-average pilot and builds three weak
systems from the same data: raw (U), filtered (pilot plug-in) and debiased
(first-order expansion of U^p at the pilot). Each system is fitted by
restricted OLS on the true support plus the noise-generated terms, and by
MSTLS and LASSO on the full library. Records stream to JSONL next to the
Markdown report.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from math import comb, sqrt
import os
from pathlib import Path
import sys
from time import perf_counter
for _thread_variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")
import numpy as np
from scipy import linalg
from scipy.ndimage import uniform_filter, uniform_filter1d
from experiments import coefficient_metrics, estimate_noise_std, trial_seed
from methods import build_wsindy_system, lasso, moving_average_pilot, mstls
from simulation_generation import BENCHMARKS, DEFAULT_DATA_DIR, load_clean_benchmark

CRITICAL_NOISE = {"VBG": sqrt(0.01/3)}  # consistency paper, Section 5.4.2
DEFAULT_SIZES = (3, 5, 9, 17, 33, 65)
DEFAULT_MULTIPLIERS = (0.5, 1., 2., 5., 10.)
SPLITS = {"none": None, "time2": ("axis", -1, 2), "checkerboard": "checkerboard"}
METHODS = ("raw", "filtered", "debiased")
REGRESSIONS = ("ols", "mstls", "lasso")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), default="VBG")
    parser.add_argument("--sigma-c", type=float, help="Reference noise std; defaults to the paper's critical noise.")
    parser.add_argument("--sigma-multipliers", type=float, nargs="+", default=list(DEFAULT_MULTIPLIERS))
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot-sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="Odd window lengths used on every axis, capped by the test support 2m+1.")
    parser.add_argument("--no-rule-size", action="store_true",
                        help="Do not add the consistency paper's data-driven window (rounded up to odd).")
    parser.add_argument("--filter-prior", type=float, default=.01)
    parser.add_argument("--splits", choices=tuple(SPLITS), nargs="+", default=["none", "time2"])
    parser.add_argument("--space-boundary", choices=("wrap", "reflect"), default="wrap",
                        help="Pilot boundary extension on spatial axes; time always uses reflect.")
    parser.add_argument("--exclude-time-boundary", action="store_true",
                        help="Drop test functions whose time support meets the largest pilot boundary layer.")
    parser.add_argument("--regressions", choices=REGRESSIONS, nargs="+", default=list(REGRESSIONS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--offline", action="store_true", help="Require cached data; do not generate or download.")
    parser.add_argument("--output", type=Path, default=Path("results_debiasing.md"))
    args = parser.parse_args(argv)
    spec = BENCHMARKS[args.benchmark]
    if len(spec.component_names) != 1 or spec.lhs_time_order != 1:
        parser.error("The study supports scalar first-order-in-time polynomial benchmarks.")
    if args.sigma_c is None:
        args.sigma_c = CRITICAL_NOISE.get(args.benchmark)
    if args.sigma_c is None or not np.isfinite(args.sigma_c) or args.sigma_c <= 0:
        parser.error("Give a positive --sigma-c for this benchmark.")
    if args.trials < 1 or args.seed < 0 or args.workers < 1:
        parser.error("--trials and --workers must be positive and --seed nonnegative.")
    if any(not np.isfinite(m) or m < 0 for m in args.sigma_multipliers) or len(set(args.sigma_multipliers)) != len(args.sigma_multipliers):
        parser.error("Noise multipliers must be unique, finite and nonnegative.")
    if any(w < 3 or w % 2 == 0 for w in args.pilot_sizes) or len(set(args.pilot_sizes)) != len(args.pilot_sizes):
        parser.error("Pilot sizes must be unique odd integers >= 3.")
    if not 0 < args.filter_prior <= 1:
        parser.error("--filter-prior must lie in (0, 1].")
    if len(set(args.splits)) != len(args.splits) or len(set(args.regressions)) != len(args.regressions):
        parser.error("Splits and regressions must be unique.")
    if "lasso" in args.regressions:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            parser.error("LASSO needs scikit-learn (requirements.txt); drop it from --regressions.")
    return args


def noise_generated_terms(spec, terms):
    """Library terms that Gaussian noise generates from the true monomials.

    E[(u+e)^p] = sum_k binom(p,2k)(2k-1)!! sigma^(2k) u^(p-2k): the raw column
    of a true term w D^alpha u^p contains binom(p,2k)(2k-1)!! sigma^(2k) times
    the column of D^alpha u^(p-2k), which a restricted OLS fit compensates
    with coefficient -w binom(p,2k)(2k-1)!! sigma^(2k) on that generated
    term. Return {term: [(2k, predicted coefficient per sigma^(2k)), ...]}.
    """
    generated = {}
    for equation in spec.true_coefficients:
        for term, weight in equation.items():
            p = term.powers[0]
            for k in range(1, p//2+1):
                candidate = type(term)((p-2*k,), term.derivative)
                if candidate in terms:
                    factor = -weight*comb(p, 2*k)*np.prod(np.arange(1, 2*k, 2))
                    generated.setdefault(candidate, []).append((2*k, float(factor)))
    return generated


def profile_settings(spec):
    """Weak-system and sparse-selection settings of the benchmark's suite."""
    build = dict(lhs_components=spec.lhs_components, lhs_time_order=spec.lhs_time_order,
                 half_widths=spec.half_widths, strides=spec.strides, test_function=spec.test_function)
    if spec.suite == "consistency":
        build.update(rescale=False)
        select = dict(thresholds=np.logspace(-4, 0, 100), threshold_units="physical", selection_rule="absolute")
    else:
        build.update(rescale=True, state_scale_rule="authors", test_degrees=spec.degrees)
        select = dict(thresholds=None, threshold_units="physical", selection_rule="relative", allow_empty=False)
    return build, select


def rule_size(sigma_estimate, spec, prior):
    """Consistency paper Section 5.4 window (volume rule), rounded up to odd."""
    dimensions = len(spec.shape)
    target = 2*(comb(spec.max_degree, 2)*sigma_estimate**2/prior)**(1/dimensions)
    cap = np.prod([2*m+1 for m in spec.half_widths])**(1/dimensions)/2
    side = max(1, int(np.floor(min(target, cap))))
    return side if side % 2 else side+1


def _window(size, spec):
    return tuple(min(size, 2*m+1) for m in spec.half_widths)


def _boundary(spec, space_boundary):
    return (space_boundary,)*(len(spec.shape)-1)+("reflect",)


def time2_pilot(observed, spec, prior=.01, *, space_boundary="wrap"):
    """Cross-fit both the box window and the pilot using the other time parity.

    Noise is estimated on the training parity's time subgrid. A common window
    across components preserves mixed-monomial alignment. Window selection
    and smoothing for one evaluation fold never read that fold's values.
    """
    values = tuple(np.asarray(v, dtype=float) for v in observed)
    if not values or any(v.shape != spec.shape or not np.isfinite(v).all() for v in values):
        raise ValueError("Finite observed components must match the benchmark grid.")
    if not 0 < prior <= 1 or space_boundary not in ("wrap", "reflect"):
        raise ValueError("Invalid pilot prior or boundary mode.")
    pilot = tuple(np.empty_like(v) for v in values)
    modes = _boundary(spec, space_boundary)
    fold_info = []
    for parity in (0, 1):
        estimates = [estimate_noise_std(v[..., 1-parity::2]) for v in values]
        requested = rule_size(max(estimates), spec, prior)
        window = _window(max(3, requested), spec)
        mask = (np.arange(spec.shape[-1]) % 2 != parity).astype(float)
        weight = uniform_filter1d(mask, window[-1], mode="reflect")[parity::2]
        if np.any(weight <= 0):
            raise ValueError("The time2 pilot window has no training points.")
        for target, values_c in zip(pilot, values):
            averaged = uniform_filter(values_c*mask, size=window, mode=list(modes))
            target[..., parity::2] = averaged[..., parity::2]/weight
        fold_info.append(dict(evaluation_parity=parity, training_parity=1-parity,
            estimated_noise_std=estimates, rule_size=requested, window=window,
            window_points=int(np.prod(window)), minimum_window_fraction=float(weight.min())))
    return pilot, dict(split="time2", prior=prior, boundary=modes, folds=fold_info,
        window_selection="Sixth differences on the opposite time-parity subgrid only")


def time_boundary_rows(spec, half_width):
    """Rows whose time support stays clear of the pilot boundary layer."""
    size, m, stride = spec.shape[-1], spec.half_widths[-1], spec.strides[-1]
    time_centers = np.arange(m, size-m, stride)
    keep_time = (time_centers >= m+half_width) & (time_centers <= size-1-m-half_width)
    space_rows = int(np.prod([len(range(mm, n-mm, s)) for n, mm, s in zip(spec.shape[:-1], spec.half_widths[:-1], spec.strides[:-1])]))
    return np.tile(keep_time, space_rows)


_CLEAN = {}


def _clean(name, data_dir, generate):
    key = (name, str(data_dir))
    if key not in _CLEAN:
        _CLEAN[key] = load_clean_benchmark(name, data_dir=data_dir, download=generate)
    return _CLEAN[key]


def fit_system(system, rows, restricted, truth, select, regressions):
    G, b = system.G[rows], system.b[rows]
    scales = system.coefficient_scales
    truth = np.asarray(truth, dtype=float)
    scalar = truth.ndim == 1
    truth_matrix = truth[:, None] if scalar else truth
    if truth_matrix.shape != (G.shape[1], b.shape[1]):
        raise ValueError("Truth must have one column per fitted equation.")
    def coefficients_for_report(result):
        return (result.coefficients[:, 0] if scalar else result.coefficients).tolist()
    fits = {}
    if "ols" in regressions:
        if not scalar or b.shape[1] != 1:
            raise ValueError("Restricted OLS remains a scalar sanity check.")
        start = perf_counter()
        coefficients = np.zeros(G.shape[1])
        coefficients[restricted] = (linalg.lstsq(G[:, restricted], b[:, 0], lapack_driver="gelsd")[0]
                                    * scales[restricted, 0])
        fits["ols"] = dict(coefficients=coefficients.tolist(),
                           e_inf=float(np.max(np.abs((coefficients-truth)[truth != 0]/truth[truth != 0]))),
                           regression_runtime=perf_counter()-start)
    if "mstls" in regressions:
        start = perf_counter()
        result = mstls(G, b, coefficient_scales=scales, **select)
        fits["mstls"] = dict(coefficients=coefficients_for_report(result), threshold=result.threshold,
                             loss=float(np.min(result.losses)), rank=result.rank,
                             thresholds=result.thresholds.tolist(), losses=result.losses.tolist(),
                             regression_runtime=perf_counter()-start,
                             **coefficient_metrics(result.coefficients, truth_matrix))
    if "lasso" in regressions:
        start = perf_counter()
        result = lasso(G, b, coefficient_scales=scales)
        fits["lasso"] = dict(coefficients=coefficients_for_report(result), threshold=result.threshold,
                             loss=float(np.min(result.losses)), rank=result.rank,
                             thresholds=result.thresholds.tolist(), losses=result.losses.tolist(),
                             selected_alpha=(result.threshold*result.alpha_max).tolist(),
                             alpha_max=result.alpha_max.tolist(), column_rms=result.column_rms.tolist(),
                             response_rms=result.response_rms.tolist(),
                             normalized_coefficients=result.normalized_coefficients.tolist(),
                             kkt_errors=result.kkt_errors.tolist(), solver_warnings=result.solver_warnings,
                             regression_runtime=perf_counter()-start,
                             **coefficient_metrics(result.coefficients, truth_matrix))
    return fits


def run_trial(config, multiplier, trial):
    clean = _clean(config["benchmark"], config["data_dir"], not config["offline"])
    spec = clean.benchmark
    terms = spec.library()
    truth = spec.truth(terms)[:, 0]
    generated = noise_generated_terms(spec, terms)
    restricted = sorted({i for i, t in enumerate(terms) if truth[i] or t in generated})
    build, select = profile_settings(spec)
    sigma = multiplier*config["sigma_c"]
    seed = trial_seed(config["seed"], spec.name, sigma, trial)
    rng = np.random.default_rng(seed)
    observed = tuple(v+sigma*rng.standard_normal(v.shape) for v in clean.u_true)
    sigma_estimate = estimate_noise_std(observed[0])
    sizes = sorted(set(config["sizes"])|({rule_size(sigma_estimate, spec, config["prior"])} if config["rule_size"] else set()))
    rows = slice(None)
    if config["exclude_time_boundary"]:
        rows = time_boundary_rows(spec, (_window(max(sizes), spec)[-1]-1)//2)
    boundary = _boundary(spec, config["space_boundary"])
    base = dict(name=spec.name, multiplier=multiplier, sigma=sigma, trial=trial, seed=seed,
                sigma_estimate=sigma_estimate, rule_size=rule_size(sigma_estimate, spec, config["prior"]),
                rows=int(np.count_nonzero(rows)) if config["exclude_time_boundary"] else None)
    grids = (clean.spatial_grid, clean.time)
    records = []
    start = perf_counter()
    raw = build_wsindy_system(observed, *grids, library_terms=terms, **build)
    records.append(dict(base, method="raw", split=None, size=None, window=None,
                        fits=fit_system(raw, rows, restricted, truth, select, config["regressions"]),
                        runtime=perf_counter()-start))
    for split_name in config["splits"]:
        for size in sizes:
            start = perf_counter()
            window = _window(size, spec)
            pilot, info = moving_average_pilot(observed, window, boundary=boundary, split=SPLITS[split_name])
            smoothed = moving_average_pilot(clean.u_true, window, boundary=boundary)[0]
            diagnostics = dict(pilot_mse=float(np.mean((pilot[0]-clean.u_true[0])**2)),
                               smoothing_bias_sq=float(np.mean((smoothed[0]-clean.u_true[0])**2)),
                               window_points=info["window_points"], minimum_window_fraction=info["minimum_window_fraction"])
            pilot_time = perf_counter()-start
            for method, system in (("filtered", build_wsindy_system(pilot, *grids, library_terms=terms, **build)),
                                   ("debiased", build_wsindy_system(observed, *grids, library_terms=terms, pilot=pilot, **build))):
                records.append(dict(base, method=method, split=split_name, size=size, window=window, **diagnostics,
                                    fits=fit_system(system, rows, restricted, truth, select, config["regressions"]),
                                    runtime=pilot_time+perf_counter()-start))
    return records


def _job(arguments):
    return run_trial(*arguments)


def run(args, on_record=None):
    config = dict(benchmark=args.benchmark, data_dir=args.data_dir, offline=args.offline, sigma_c=args.sigma_c,
                  seed=args.seed, sizes=tuple(args.pilot_sizes), rule_size=not args.no_rule_size, prior=args.filter_prior,
                  splits=tuple(args.splits), space_boundary=args.space_boundary,
                  exclude_time_boundary=args.exclude_time_boundary, regressions=tuple(args.regressions))
    _clean(args.benchmark, args.data_dir, not args.offline)  # generate or verify once, before any worker starts
    jobs = [(config, multiplier, trial) for multiplier in args.sigma_multipliers for trial in range(args.trials)]
    records = []
    def collect(batch, done):
        records.extend(batch)
        if on_record:
            for record in batch:
                on_record(record)
        if done == 1 or done % 10 == 0 or done == len(jobs):
            print(f"{done}/{len(jobs)} trials", file=sys.stderr, flush=True)
    if args.workers == 1:
        for done, job in enumerate(jobs, start=1):
            collect(_job(job), done)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, batch in enumerate(pool.map(_job, jobs), start=1):
                collect(batch, done)
    return records


def _mean_sd(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return "—"
    return f"{np.mean(values):.3g} ± {np.std(values, ddof=1):.2g}" if len(values) > 1 else f"{values[0]:.3g}"


def _label(record):
    if record["method"] == "raw":
        return "raw", "—", "—"
    return record["method"], record["split"], str(record["size"])


def format_report(args, records):
    spec = BENCHMARKS[args.benchmark]
    terms = spec.library()
    truth = spec.truth(terms)[:, 0]
    labels = [t.label(spec.component_names) for t in terms]
    generated = noise_generated_terms(spec, terms)
    build, select = profile_settings(spec)
    settings = sorted({_label(r) for r in records}, key=lambda s: (METHODS.index(s[0]), s[1], int(s[2]) if s[2].isdigit() else 0))
    equation = " + ".join(f"({w:g}) {labels[i]}" for i, w in enumerate(truth) if w)
    lines = ["# Debiased WSINDy sanity check", "",
        f"Benchmark `{spec.name}`: `u_t = {equation}` on grid `{spec.shape}`; reference noise "
        f"`sigma_c = {args.sigma_c:g}`; multipliers `{args.sigma_multipliers}`; `{args.trials}` trials per level "
        f"(root seed {args.seed}). Library: `{len(terms)}` terms; `m = {spec.half_widths}`, `s = {spec.strides}`, "
        f"`{spec.test_function}` test functions, rescale `{build['rescale']}`; sparse selection "
        f"`{select['selection_rule']}` thresholds in `{select['threshold_units']}` units.", "",
        "Methods on identical observations and pilots: **raw** integrates `U^p`; **filtered** integrates "
        "`pilot^p`; **debiased** integrates `(1-p) pilot^p + p pilot^(p-1) U` (Section 5 with `u_*` replaced by "
        "`U`), so its LHS and linear columns equal the raw ones. The pilot is a box moving average with odd "
        f"window length `size` on every axis (capped by `2m+1`), boundary `{_boundary(spec, args.space_boundary)}`; "
        "split `time2` averages each point over the other time-index parity only (2-fold cross-fitting), "
        "`checkerboard` over the other index-sum parity, `none` over all points. The data-driven window of the "
        "consistency paper (volume rule, prior `tau*`, rounded up to odd) is "
        f"{'excluded' if args.no_rule_size else 'included per trial'}.", "",
        "Restricted OLS uses the true support plus the terms Gaussian noise generates from it "
        f"(`{', '.join(labels[i] for i, t in enumerate(terms) if t in generated and not truth[i])}` here). "
        "Columns show fitted coefficient minus truth; each generated term's header gives the coefficient "
        "predicted for raw OLS from the Gaussian moments of the true monomials, and the smallest true "
        "coefficient is shown because it dominates `E_inf`, the maximum relative error over the true "
        "coefficients. Entries are mean ± sample standard deviation over trials.",
        "" if not args.exclude_time_boundary else
        "Test functions whose time support meets the largest pilot boundary layer are dropped for every method.", ""]
    if "ols" in args.regressions:
        lines.append("## Restricted OLS")
        for multiplier in args.sigma_multipliers:
            sigma = multiplier*args.sigma_c
            headers = []
            for term, parts in generated.items():
                index = terms.index(term)
                predicted = sum(factor*sigma**power for power, factor in parts)
                headers.append((index, f"β[{labels[index]}] − {truth[index]:g} (raw pred. {predicted:+.3g})"))
            smallest = int(np.argmin(np.where(truth != 0, np.abs(truth), np.inf)))
            if smallest not in [index for index, _ in headers]:
                headers.append((smallest, f"β[{labels[smallest]}] − {truth[smallest]:g} (smallest true)"))
            lines += ["", f"### sigma = {multiplier:g} sigma_c = {sigma:.4g}", "",
                      "| method | split | size | " + " | ".join(h for _, h in headers) + " | E_inf |",
                      "| --- | --- | ---: | " + " | ".join("---:" for _ in headers) + " | ---: |"]
            for setting in settings:
                group = [r for r in records if r["multiplier"] == multiplier and _label(r) == setting and "ols" in r["fits"]]
                cells = [_mean_sd([r["fits"]["ols"]["coefficients"][index]-truth[index] for r in group]) for index, _ in headers]
                lines.append(f"| {setting[0]} | {setting[1]} | {setting[2]} | " + " | ".join(cells)
                             + f" | {_mean_sd([r['fits']['ols']['e_inf'] for r in group])} |")
    for regression in ("mstls", "lasso"):
        if regression not in args.regressions:
            continue
        lines += ["", f"## Full-library {regression.upper()}: exact-support rate / mean E2", "",
                  "| method | split | size | " + " | ".join(f"{m:g} sigma_c" for m in args.sigma_multipliers) + " |",
                  "| --- | --- | ---: | " + " | ".join("---:" for _ in args.sigma_multipliers) + " |"]
        for setting in settings:
            cells = []
            for multiplier in args.sigma_multipliers:
                group = [r["fits"][regression] for r in records if r["multiplier"] == multiplier and _label(r) == setting]
                cells.append(f"{100*np.mean([g['exact_support'] for g in group]):.0f}% / {np.mean([g['e2'] for g in group]):.3g}" if group else "—")
            lines.append(f"| {setting[0]} | {setting[1]} | {setting[2]} | " + " | ".join(cells) + " |")
    lines += ["", "## Pilot diagnostics", "",
              "`r_n^2` is the mean squared pilot error on the grid, split into the squared smoothing bias "
              "(the unsplit window applied to clean data) and the remainder, which is the pilot variance for "
              "`none` and only approximately so for split pilots, whose deterministic part differs slightly; "
              "`coverage` is the smallest fraction of a window that holds other-fold points.", "",
              "| split | size | sigma mult. | r_n^2 | bias^2 | variance | coverage | rule size (trials) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for setting in settings:
        if setting[0] != "filtered":
            continue
        for multiplier in args.sigma_multipliers:
            group = [r for r in records if r["multiplier"] == multiplier and _label(r) == setting]
            if not group:
                continue
            mse = np.mean([r["pilot_mse"] for r in group]); bias = np.mean([r["smoothing_bias_sq"] for r in group])
            rule = sum(r["rule_size"] == r["size"] for r in group)
            lines.append(f"| {setting[1]} | {setting[2]} | {multiplier:g} | {mse:.3g} | {bias:.3g} | {mse-bias:.3g} | "
                         f"{min(r['minimum_window_fraction'] for r in group):.2f} | {rule} |")
    return "\n".join(lines)+"\n"


def main():
    args = parse_arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".jsonl")
    with raw_path.open("w") as raw:
        def save(record):
            raw.write(json.dumps(record, allow_nan=False)+"\n")
            raw.flush()
        records = run(args, save)
    args.output.write_text(format_report(args, records))
    print(f"Saved {args.output} and {raw_path}")


if __name__ == "__main__":
    main()
