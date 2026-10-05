"""Run the original WSINDy PDE suite or the two consistency-paper PDE benchmarks."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from functools import lru_cache
import json
from math import comb
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
# Avoid oversubscribing BLAS when benchmark processes run concurrently.
for _thread_variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")
import numpy as np
import scipy
from scipy.ndimage import uniform_filter
from methods import wsindy
from simulation_generation import (BENCHMARKS, BENCHMARK_SEED_ORDER, ORIGINAL_BENCHMARKS, DEFAULT_DATA_DIR,
    file_checksum, generated_config, load_clean_benchmark, observation_instance)

PAPER_NOISE_RATIOS = tuple(float(k/40) for k in range(41))
PAPER_TRIALS = 200
BENCHMARK_NOISE_RATIOS = (0., 0.2, 0.5, 0.75, 1.)
BENCHMARK_TRIALS = 100
PROTOCOL_VERSION = 2
METHOD_LABELS = {"wsindy": "WSINDy", "filtered-wsindy": "Filtered WSINDy"}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", choices=tuple(BENCHMARKS), nargs="+", default=list(ORIGINAL_BENCHMARKS))
    parser.add_argument("--noise-ratios", type=float, nargs="+", default=list(BENCHMARK_NOISE_RATIOS))
    parser.add_argument("--trials", type=int, default=BENCHMARK_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=("authors", "printed", "consistency"))
    parser.add_argument("--method", choices=tuple(METHOD_LABELS), default="wsindy")
    parser.add_argument("--regression", choices=("mstls", "lasso"), default="mstls")
    parser.add_argument("--comparison-reports", type=Path, nargs="+",
                        help="Aggregate saved MSTLS/LASSO reports into shared tables without fitting.")
    parser.add_argument("--filter-prior", type=float, default=.01,
                        help="Coefficient-ratio prior for the adaptive moving-average window.")
    parser.add_argument("--compare-to", type=Path,
                        help="Original WSINDy report whose saved trials are paired with filtered results.")
    parser.add_argument("--comparison-output", type=Path, default=Path("results.md"))
    parser.add_argument("--include-report", type=Path, nargs="*", default=[],
                        help="Completed comparison reports to include before this comparison.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent benchmark processes.")
    parser.add_argument("--resume", action="store_true", help="Resume an identical saved protocol.")
    parser.add_argument("--report-only", action="store_true", help="Refresh reports from completed trial records.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--offline", action="store_true", help="Require existing verified data; do not download.")
    parser.add_argument("--download-only", action="store_true", help="Verify/cache original data and exit.")
    parser.add_argument("--output", type=Path, default=Path("results.md"))
    args = parser.parse_args(argv)
    suites = {BENCHMARKS[name].suite for name in args.benchmarks}
    if len(suites) != 1:
        parser.error("Run the original and consistency suites separately; their numerical protocols differ.")
    consistency = suites == {"consistency"}
    args.profile = args.profile or ("consistency" if consistency else "authors")
    if consistency != (args.profile == "consistency"):
        parser.error("Consistency PDEs require --profile consistency; original PDEs use authors or printed.")
    if args.trials < 1 or args.seed < 0 or args.workers < 1:
        parser.error("--trials and --workers must be positive and --seed must be nonnegative.")
    if not np.isfinite(args.filter_prior) or not 0 < args.filter_prior <= 1:
        parser.error("--filter-prior must lie in (0, 1].")
    if args.compare_to and args.method != "filtered-wsindy":
        parser.error("--compare-to requires --method filtered-wsindy.")
    if args.compare_to and args.regression != "mstls":
        parser.error("Use --comparison-reports to compare saved LASSO and MSTLS runs.")
    if args.compare_to and args.comparison_output.resolve() in (args.output.resolve(), args.compare_to.resolve()):
        parser.error("The comparison output must be separate from both individual reports.")
    if any(not np.isfinite(r) or r < 0 or r > 1 for r in args.noise_ratios):
        parser.error("Noise ratios must be finite and lie in [0, 1].")
    if len(set(args.benchmarks)) != len(args.benchmarks) or len(set(args.noise_ratios)) != len(args.noise_ratios):
        parser.error("Benchmark names and noise levels must be unique.")
    return args


def trial_seed(base_seed, name, noise_ratio, trial):
    """Stable across benchmark ordering, selected noise subsets, and reruns."""
    bits = int(np.float64(noise_ratio).view(np.uint64))
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown active benchmark: {name}")
    sequence = np.random.SeedSequence([base_seed, BENCHMARK_SEED_ORDER.index(name), bits & 0xffffffff, bits >> 32, trial])
    return sequence.generate_state(4).tolist()


def coefficient_metrics(coefficients, truth):
    """Paper Eqs. 5.3--5.5, with pooled support for coupled systems."""
    coefficients, truth = np.asarray(coefficients), np.asarray(truth)
    if coefficients.shape != truth.shape or not np.any(truth):
        raise ValueError("Coefficients must match a nonzero ground-truth matrix.")
    selected, actual = coefficients != 0, truth != 0
    tp = int(np.count_nonzero(selected & actual))
    fn = int(np.count_nonzero(~selected & actual))
    fp = int(np.count_nonzero(selected & ~actual))
    return {"tpr": tp/(tp+fn+fp), "e2": float(np.linalg.norm(coefficients-truth)/np.linalg.norm(truth)),
            "e_inf": float(np.max(np.abs((coefficients-truth)[actual]/truth[actual]))),
            "tp": tp, "fn": fn, "fp": fp, "exact_support": fn == 0 and fp == 0}


def estimate_noise_std(values):
    """Appendix G: RMS of unit-L2 sixth differences along the time axis."""
    values = np.asarray(values, dtype=float)
    if values.shape[-1] < 7:
        raise ValueError("Noise estimation requires at least seven timepoints.")
    differences = np.diff(values, n=6, axis=-1)
    return float(np.linalg.norm(differences.ravel())/np.sqrt(differences.size*comb(12, 6)))


def moving_average_observations(values, half_widths, max_degree, prior=.01, *, support_rule="per_axis"):
    """Data-only adaptation of the consistency paper's Section 5.4 rule.

    Generalize 1500 to binom(p_max, 2)/prior and cap each side independently
    for anisotropic test supports. No clean data, noise labels, or truth enter.
    """
    values = tuple(np.asarray(v, dtype=float) for v in values)
    if not values or len(half_widths) != values[0].ndim:
        raise ValueError("Filter supports must match the observation dimensions.")
    if not np.isfinite(prior) or not 0 < prior <= 1 or max_degree < 2:
        raise ValueError("Need a positive prior <= 1 and polynomial degree >= 2.")
    if any(v.shape != values[0].shape or not np.all(np.isfinite(v)) for v in values):
        raise ValueError("Components must be finite and share a grid.")
    if any(not isinstance(m, (int, np.integer)) or m < 1 for m in half_widths):
        raise ValueError("Test half-widths must be positive integers.")
    estimates = tuple(estimate_noise_std(v) for v in values)
    sigma = max(estimates)
    target_side = 2*(comb(max_degree, 2)*sigma**2/prior)**(1/values[0].ndim)
    if support_rule == "volume":
        cap = np.prod([2*m+1 for m in half_widths])**(1/values[0].ndim)/2
        side = max(1, int(np.floor(min(target_side, cap))))
        widths = (side,)*values[0].ndim
    elif support_rule == "per_axis":
        widths = tuple(max(1, int(np.floor(min(target_side, (2*m+1)/2)))) for m in half_widths)
    else:
        raise ValueError("support_rule must be per_axis or volume.")
    # One shared window preserves component alignment, including NS velocities.
    filtered = tuple(uniform_filter(v, size=widths, mode="reflect") for v in values)
    return filtered, dict(filter_widths=widths, estimated_noise_std=estimates,
                         filter_prior=prior, filter_points=int(np.prod(widths)))


def consistency_description():
    return ("Target: [Messenger & Bortz, arXiv:2211.16000v1, Sections 5.3.2/5.4.2]"
        "(https://arxiv.org/html/2211.16000v1#S5). This is a **declared resimulation benchmark**, "
        "not an exact replication of the authors' trajectories or resolution sweeps. "
        "The published equations, domains, libraries and bump test function are used. "
        "Initial conditions, periodic boundaries, fixed observation grids, query strides and solver "
        "step sizes are explicitly chosen here because the original data/generation settings were "
        "not found in the checked public sources. The high-resolution spectral ETDRK4 trajectories "
        "are subsampled by four on both axes. Timestep refinement is checked and recorded in the manifest. "
        "No model-selection results are used to choose these settings. Noise uses the centered clean "
        "sample standard deviation (ddof=1), unlike the original suite's RMS convention. "
        "Both methods use physical coordinates without state/coordinate rescaling, the C-infinity "
        "bump exp(9/(r^2-1)), absolute hard-threshold STLS until stable, and 100 thresholds "
        "logspace(-4,0,100), with the projection-plus-sparsity loss of Eqs. 2.6–2.9.")


def consistency_filter_description(prior):
    return (f"Filter prior tau_star={prior:g}. Noise is estimated with unit-L2 sixth differences "
        "along time. For D space-time axes and m=prod(2*half_width+1), use the common side "
        "w=max(1,floor(min(2*(binom(p_max,2)*sigma_est^2/tau_star)^(1/D),m^(1/D)/2))). "
        "Average space and time with symmetric reflection before evaluating the nonlinear library; "
        "both weak LHS and RHS use filtered states. The VBG degree-6 factor is 1500. "
        "For HKS the same rule is extended to degree 8 (factor 2800); the paper does not report "
        "filtered HKS results. At zero noise the estimator still uses observed data only. "
        "Smoothing bias can worsen identification.")


def filter_protocol(prior):
    return dict(kind="separable simple moving average", prior=prior, boundary="reflect", origin=0,
        source="https://arxiv.org/pdf/2211.16000", reference="Sections 4.2, 5.4 and Appendix G",
        noise_estimator="RMS of L2-normalized sixth differences along the time axis; stencil [1,-6,15,-20,15,-6,1]/sqrt(924)",
        sigma="Maximum estimated component noise standard deviation; one shared window for every component",
        width="w_d = max(1, floor(min(2*(binom(p_max,2)*sigma_est^2/prior)^(1/D), (2*m_d+1)/2)))",
        D="Number of space-time axes", p_max="Benchmark library polynomial degree",
        m_d="Published weak-test half-width along axis d",
        adaptation="Generalize the degree-6 factor 1500 and cap anisotropic axes separately",
        scope="Moving-average preprocessing on the original six polynomial PDEs; retain the existing WSINDy test functions, library, scaling and regression",
        inputs="Observed arrays only; no clean data, injected noise standard deviation, support labels or coefficient truth")


def filter_description(prior):
    return ("Filtered WSINDy estimates component noise with unit-L2 sixth differences along time "
        "([1,-6,15,-20,15,-6,1]/sqrt(924)), then applies a separable space-time moving average "
        "with symmetric reflection (`scipy.ndimage.uniform_filter`, origin 0). "
        f"The coefficient-ratio prior is {prior:g}; sigma_est is the maximum component estimate. "
        "For D space-time axes, library degree p_max, and weak-test half-width m_d, "
        "w_d = max(1, floor(min(2*(binom(p_max,2)*sigma_est^2/prior)^(1/D), (2*m_d+1)/2))). "
        "All components share the same window. Both LHS and RHS use filtered states, with library "
        "nonlinearities evaluated after averaging. These are adaptations of "
        "[the consistency paper, Sections 4.2/5.4 and Appendix G](https://arxiv.org/pdf/2211.16000): "
        "the degree-6 factor 1500 is generalized, and each anisotropic axis is capped separately. "
        "The existing polynomial tests, 50 thresholds, scaling, and MSTLS are retained. "
        "Window selection uses observed data only, including at zero noise. This applies the "
        "preprocessing comparison to the original six-PDE polynomial benchmark protocol. "
        "Smoothing bias and signal contamination of the noise estimate can reduce accuracy.")


def lasso_protocol():
    return dict(objective="||Z theta-y||_2^2/(2*K) + alpha*||theta||_1 per equation",
        normalization="Z=G/column_RMS, y=b/response_RMS; no centering, no extra intercept",
        alpha="ratio*max(abs(Z.T@y))/K per equation",
        alpha_ratios=np.logspace(-4, 0, 100).tolist(),
        selection="matrix-2-norm OLS projection difference / OLS projection norm + nonzero fraction; smallest ratio breaks ties",
        coefficients="Penalized coefficients, restored to physical units; no OLS support refit",
        support="Exact solver zeros; no coefficient cutoff",
        solver="scikit-learn LARS Gram path; primal active-set polishing first, then coordinate descent fallback and final polishing if KKT error exceeds 1e-7*alpha_max",
        active_set=dict(initial_max_steps=200, final_max_steps=2000, kkt_tolerance=1e-9),
        validation="Every candidate must satisfy max coordinate KKT violation / alpha_max <= 1e-7",
        coordinate_descent=dict(tol=1e-9, max_iter=200000),
        inputs="Weak G and b only; no true coefficients, support or clean observations")


def lasso_description():
    return ("LASSO replaces sequential hard thresholding. For K weak rows, normalize each nonzero "
        "column Z_j=G_j/RMS(G_j) and each response y_e=b_e/RMS(b_e), without centering or an extra "
        "intercept (the constant library term is penalized). Solve "
        "`||Z theta-y_e||_2^2/(2*K)+alpha_e*||theta||_1`, with "
        "`alpha_e=ratio*max(abs(Z.T@y_e))/K` at 100 ratios `logspace(-4,0,100)`. "
        "Coupled equations use separate alpha_max values and one selected ratio. Select using the "
        "existing WSINDy loss: matrix-2-norm prediction difference from full OLS / OLS prediction "
        "norm + fraction of nonzero coefficients; smallest ratio breaks ties. "
        "Retain the penalized coefficients, including shrinkage, and restore physical units; "
        "there is no OLS support refit or coefficient cutoff. This is a column-normalized L1 penalty, "
        "equivalently a weighted penalty in physical coefficient units. Empty supports are allowed. "
        "The scikit-learn LARS Gram path is checked against KKT conditions at every candidate; "
        "primal active-set polishing is tried first, followed by coordinate descent and final polishing "
        "if needed; violation / alpha_max "
        "above 1e-7 rejects the fit. "
        "Lambda selection uses G and b only. These are additional comparisons, not the papers' "
        "published sparse-regression algorithms.")


def lasso_report_header(args, records):
    requested = len(args.benchmarks)*len(args.noise_ratios)*args.trials
    context = ("Original six polynomial PDE datasets and published weak supports/degrees/strides. "
        f"Weak-system construction uses the `{args.profile}` state/coordinate scaling profile. "
        "Noise ratio is relative to clean-component RMS. RD uses the archived 181-term library.")
    if args.profile == "consistency":
        context = ("HKS/VBG are declared resimulations using the consistency-paper equations; "
            "IC/BC/grids/strides are declared here. Weak systems use the C-infinity bump with physical "
            "coordinates and no rescaling. Noise ratio is relative to centered clean sample std.")
    lines = [f"# {METHOD_LABELS[args.method]} + LASSO benchmark", "",
        f"Status: {'complete' if len(records) == requested else 'in progress'}; "
        f"{len(records):,}/{requested:,} completed trials.", "",
        f"Schedule: {args.trials} trials per noise level at "
        f"`{', '.join(f'{r:g}' for r in args.noise_ratios)}`; root seed {args.seed}. "
        "One fixed equation and trajectory per PDE; only observation noise varies. "
        "Zero-noise repeats are identical clean checks.", "", context, "", lasso_description(), "",
        "TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E_inf is the maximum relative error "
        "over true coefficients; E2 is relative coefficient L2 error. Mean ± sample SD includes "
        "failed identifications. Runtime includes weak construction, normalization and the full "
        "LASSO path; loading, noise generation and reports are excluded.", ""]
    if args.method == "filtered-wsindy":
        description = consistency_filter_description(args.filter_prior) if args.profile == "consistency" else filter_description(args.filter_prior)
        description = description.replace("The existing polynomial tests, 50 thresholds, scaling, and MSTLS are retained.",
            "The existing polynomial tests and state/coordinate scaling are retained; regression uses LASSO.")
        lines.extend([description, "", "Runtime includes filtering and noise estimation.", ""])
    return lines


@lru_cache(maxsize=1)
def _cached_clean(name, data_dir, download):
    return load_clean_benchmark(name, data_dir=data_dir, download=download)


def run(args, on_result=None, *, completed=frozenset(), keep_records=True):
    records = []
    for name in args.benchmarks:
        clean = _cached_clean(name, args.data_dir, not args.offline)
        spec = clean.benchmark
        if args.download_only:
            print(f"{name}: verified {spec.filename}, {len(clean.u_true)} component(s), {spec.shape}", file=sys.stderr)
            continue
        terms = spec.library()
        truth = spec.truth(terms)
        # Produce 20 paired trials at diagnostic noise levels first, then fill
        # the complete schedule. Execution order does not change any seed.
        priority = [r for r in (0., .2, .5, 1.) if r in args.noise_ratios]
        initial = min(20, args.trials)
        schedule = [(r, 0, initial) for r in priority] + [
            (r, initial if r in priority else 0, args.trials) for r in args.noise_ratios]
        for ratio, first_trial, last_trial in schedule:
            for trial in range(first_trial, last_trial):
                if (name, ratio, trial) in completed:
                    continue
                seed = trial_seed(args.seed, name, ratio, trial)
                data = observation_instance(clean, ratio, seed)
                start = perf_counter()
                observed = data.u_observed
                filter_info = {}
                if args.method == "filtered-wsindy":
                    observed, filter_info = moving_average_observations(
                        observed, spec.half_widths, spec.max_degree, args.filter_prior,
                        support_rule="volume" if spec.suite == "consistency" else "per_axis")
                preprocessing_runtime = perf_counter()-start
                result, system = wsindy(observed, data.spatial_grid, data.time,
                    library_terms=terms, lhs_components=spec.lhs_components, lhs_time_order=spec.lhs_time_order,
                    half_widths=spec.half_widths, strides=spec.strides, test_degrees=spec.degrees,
                    profile=args.profile, test_function=spec.test_function, regression=args.regression)
                record = {"name": name, "noise_ratio": ratio, "trial": trial, "seed": seed,
                    "method": args.method, "regression": args.regression, **filter_info,
                    **coefficient_metrics(result.coefficients, truth),
                    "equations": [coefficient_metrics(result.coefficients[:, i], truth[:, i])
                                  for i in range(truth.shape[1])],
                    "runtime": perf_counter()-start, "threshold": result.threshold,
                    "preprocessing_runtime": preprocessing_runtime,
                    "profile": args.profile, "noise_std": data.noise_std,
                    "test_function": system.test_function, "noise_rule": spec.noise_rule,
                    "requested_workers": args.workers,
                    "loss": float(np.min(result.losses)), "rank": result.rank,
                    "shape": spec.shape, "G_shape": system.G.shape, "b_shape": system.b.shape,
                    "state_scales": system.state_scales.tolist(), "coordinate_scales": system.coordinate_scales.tolist(),
                    "coefficients": result.coefficients.tolist(),
                    "losses": result.losses.tolist(), "thresholds": result.thresholds.tolist()}
                if args.regression == "lasso":
                    record.update(alpha_max=result.alpha_max.tolist(),
                        selected_alpha=(result.threshold*result.alpha_max).tolist(),
                        column_rms=result.column_rms.tolist(), response_rms=result.response_rms.tolist(),
                        normalized_coefficients=result.normalized_coefficients.tolist(),
                        kkt_errors=result.kkt_errors.tolist(), solver_warnings=result.solver_warnings)
                if keep_records:
                    records.append(record)
                if on_result:
                    on_result(record)
                if trial == 0 or (trial+1) % 20 == 0 or trial+1 == args.trials:
                    print(f"{name} noise={ratio:g} trial={trial+1}/{args.trials}: "
                          f"TPR={record['tpr']:.4g} E2={record['e2']:.4g} "
                          f"lambda={result.threshold:.4g} seconds={record['runtime']:.3g}", file=sys.stderr, flush=True)
    return records


def true_equation_lines(spec):
    """Render the physical equations from the same truth used for scoring."""
    names = tuple(r"\omega" if name == "omega" else name for name in spec.component_names)
    axes = "xyz"[:len(spec.shape)-1]+"t"
    equations = []
    for component, truth in zip(spec.lhs_components, spec.true_coefficients):
        rhs = ""
        for term, coefficient in truth.items():
            if not coefficient:
                continue
            if term.kind == "poly":
                factors = [name if power == 1 else f"{name}^{{{power}}}"
                           for name, power in zip(names, term.powers) if power]
                expression = r"\,".join(factors) or "1"
            else:
                argument = names[term.component]
                if term.frequency != 1:
                    argument = f"{term.frequency}\\,{argument}"
                expression = f"\\{term.kind}\\left({argument}\\right)"
            suffix = "".join(axis*order for axis, order in zip(axes, term.derivative))
            if suffix:
                if term.kind != "poly" or sum(term.powers) != 1:
                    expression = r"\left("+expression+r"\right)"
                expression += "_{"+suffix+"}"
            magnitude = abs(coefficient)
            if magnitude != 1:
                expression = f"{magnitude:g}\\,{expression}"
            sign = (" - " if coefficient < 0 else " + ") if rhs else ("-" if coefficient < 0 else "")
            rhs += sign+expression
        lhs = names[component]+"_{"+"t"*spec.lhs_time_order+"}"
        equations.append(lhs+r" &= "+rhs)
    lines = ["**True equation (physical units):**", "", "$$", r"\begin{aligned}",
             " \\\\\n".join(equations), r"\end{aligned}", "$$", ""]
    if spec.name == "NS":
        lines.extend([r"Here $\omega$ is vorticity and $(u,v)$ is the observed incompressible velocity "
                      r"($u_x+v_y=0$). Only the vorticity equation is identified; "
                      r"$-(\omega u)_x-(\omega v)_y=-u\omega_x-v\omega_y$. "
                      r"The viscosity is $\nu=0.01$.", ""])
    return lines


def format_report(args, records):
    requested = len(args.benchmarks)*len(args.noise_ratios)*args.trials
    complete = len(records) == requested
    reused = sum("reused_from_protocol_id" in r for r in records)
    paper_schedule = args.trials == PAPER_TRIALS and tuple(args.noise_ratios) == PAPER_NOISE_RATIOS
    lines = [f"# {METHOD_LABELS[args.method]} reproduction", "",
        "Target: [Messenger & Bortz, arXiv:2007.02848v3](https://arxiv.org/html/2007.02848v3). "
        "Original `U_exact` arrays from the [authors' archive](https://zenodo.org/records/20787783), with MD5 verification. "
        "No resimulation, interpolation, or coarsening.", "",
        f"Status: {'complete' if complete else 'in progress'}; {len(records):,}/{requested:,} completed trials. "
        "KS, NLS, RD are primary benchmarks; IB, KdV, NS are supplementary.", "",
        f"Schedule: {args.trials} trials per noise level; root seed {args.seed}. "
        f"Noise ratios: {', '.join(f'{r:g}' for r in args.noise_ratios)}. "
        f"{'Full paper identification schedule for the selected PDEs.' if paper_schedule else 'Subset of the paper identification schedule.'}", "",
        ("Author-code baseline: least squares on the scaled system, physical-unit MSTLS bounds, "
         "return before an empty support, and state scale exponent `1/(beta_max-1)`. "
         if args.profile == "authors" else
         "Printed-algorithm ablation: scaled-unit MSTLS, empty support admissible, "
         "and state scale exponent `1/beta_max`. ") +
        "Both use published supports/degrees/strides and 50 thresholds `logspace(-4,0,50)`. "
        "SVD least squares uses SciPy GELSD, not MATLAB backslash. "
        "RD uses the archived degree-5/order-4 library (181 columns, 4860 rows); "
        "the paper's contradictory RD row count is not reproduced. See README for source discrepancies.", "",
        "TPR is TP/(TP+FN+FP), E2 is relative coefficient L2 error, and E_inf is the maximum "
        "relative error over true nonzero coefficients. The table reports mean ± sample standard deviation "
        "when there are at least two trials, and the observed value for a single trial. "
        "Exact is the percentage of trials with precisely the true support. "
        "For coupled systems metrics summarize the full coefficient matrix; per-equation values are in the JSONL files. "
        "Errors include failed identifications. Runtime covers weak-system construction, scaling, and all threshold refits, "
        "excluding disk loading, noise generation, and reporting; concurrent timings are not serial MATLAB timings.", "",
        "These results measure identification and coefficient accuracy as in arXiv v3. "
        "Solution-prediction metrics added in the later journal article are not evaluated. "
        "Original MATLAB random draws and runtime measurements are not reproduced.", ""]
    if reused:
        lines.extend([f"Reused {reused:,} previously completed trials with unchanged numerical method and seeds. "
            "Selection is by requested noise level and trial index only. Original protocol IDs are retained "
            "in raw records; see `results.reuse.json` for provenance.", ""])
    if args.profile == "consistency":
        lines = [f"# {METHOD_LABELS[args.method]} consistency PDE benchmark", "",
            consistency_description(), "",
            f"Status: {'complete' if complete else 'in progress'}; {len(records):,}/{requested:,} completed trials.", "",
            f"Schedule: {args.trials} trials per noise ratio at "
            f"`{', '.join(f'{r:g}' for r in args.noise_ratios)}`; root seed {args.seed}. "
            "One fixed trajectory per PDE; only Gaussian observation noise varies. Zero-noise repeats are identical.", "",
            "TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E_inf is maximum relative error "
            "over true coefficients; E2 is relative coefficient L2 error. Mean ± sample SD includes "
            "failed identifications. Runtime excludes generation/loading/noise and includes preprocessing. "
            "Subset-support probabilities and Wilson exact-recovery intervals are saved in summaries.", ""]
    if args.method == "filtered-wsindy":
        lines.extend([consistency_filter_description(args.filter_prior) if args.profile == "consistency"
                      else filter_description(args.filter_prior), "",
            "Runtime includes noise estimation and moving-average preprocessing.", ""])
    if args.regression == "lasso":
        lines = lasso_report_header(args, records)
    for name in args.benchmarks:
        spec = BENCHMARKS[name]
        first = next((record for record in records if record["name"] == name), None)
        lines.extend([f"## {name}", ""])
        lines.extend(true_equation_lines(spec))
        if spec.suite == "consistency":
            config = generated_config(name)
            lines.extend([f"Declared IC: `{config['initial_condition']}`; boundary: periodic; "
                f"x in `{config['x_interval']}`, t in `{config['time_interval']}`. "
                f"Fine simulation: `{config['fine_shape']}`; observation subsampling: `{config['subsample']}`.", ""])
        if args.method == "filtered-wsindy":
            widths = [r["filter_widths"] for r in records if r["name"] == name]
            if widths:
                limits = [(min(w[d] for w in widths), max(w[d] for w in widths))
                          for d in range(len(spec.shape))]
                lines.extend([f"Observed filter-side ranges (grid points, space then time): `{limits}`.", ""])
        lines.extend([f"Data shape: `{spec.shape}`, components `{spec.component_names}`; "
            f"G shape: `{tuple(first['G_shape']) if first else 'pending'}`. "
            f"`m={spec.half_widths}`, `s={spec.strides}`, `p={spec.degrees}`.", "",
            "| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for ratio in args.noise_ratios:
            group = [r for r in records if r["name"] == name and r["noise_ratio"] == ratio]
            if not group:
                continue
            cells = []
            for field in ("tpr", "e_inf", "e2"):
                values = np.array([r[field] for r in group])
                cells.append(f"{np.mean(values):.6g} ± {np.std(values, ddof=1):.3g}"
                             if len(values) > 1 else f"{values[0]:.6g}")
            exact = 100*np.mean([r["exact_support"] for r in group])
            runtime = np.median([r["runtime"] for r in group])
            lines.append(f"| {ratio:g} | {len(group)} | {exact:.1f}% | " + " | ".join(cells) + f" | {runtime:.3g} |")
        lines.append("")
    return "\n".join(lines)


def protocol_manifest(args):
    """Freeze the data, library, method, environment, and paired noise protocol."""
    root = Path(__file__).parent
    benchmark_specs = {}
    for name in args.benchmarks:
        s = BENCHMARKS[name]
        benchmark_specs[name] = dict(filename=s.filename, md5=s.checksum, shape=s.shape,
            component_names=s.component_names, archive_component_order=s.archive_component_order,
            lhs_components=s.lhs_components, lhs_time_order=s.lhs_time_order,
            half_widths=s.half_widths, strides=s.strides, test_degrees=s.degrees,
            terms=[asdict(t) for t in s.library()], term_labels=[t.label(s.component_names) for t in s.library()],
            true_coefficients=s.truth().tolist())
        if s.suite == "consistency":
            load_clean_benchmark(name, data_dir=args.data_dir, download=not args.offline)
            data_path = args.data_dir/s.filename
            with np.load(data_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            benchmark_specs[name].update(md5=file_checksum(data_path), suite=s.suite,
                noise_rule=s.noise_rule, test_function=s.test_function, generation=metadata)
    manifest = dict(version=PROTOCOL_VERSION, profile=args.profile, root_seed=args.seed,
        noise_ratios=args.noise_ratios, trials=args.trials, benchmarks=benchmark_specs,
        rng="numpy.random.default_rng / PCG64; trial_seed as stored in experiments.py",
        noise="Independent Gaussian noise per point/component, std = ratio * clean component RMS",
        dtype="float64", thresholds=np.logspace(-4, 0, 50).tolist(), ls_solver="SciPy GELSD",
        state_scale_rule=args.profile, threshold_units="physical" if args.profile == "authors" else "scaled",
        allow_empty=args.profile == "printed", authors_commit="d9296be4c17c5e0b4df14472f4cd8276a8ae4eed",
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
        source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest()
                       for f in ("methods.py", "simulation_generation.py", "experiments.py")})
    manifest["method"] = args.method
    manifest["regression"] = args.regression
    if args.profile == "consistency":
        manifest.update(noise="Independent Gaussian per point; std = ratio * centered clean sample std (ddof=1)",
            thresholds=np.logspace(-4, 0, 100).tolist(), state_scale_rule="none", threshold_units="physical",
            allow_empty=True, selection_rule="absolute coefficient hard threshold", test_function="C-infinity bump",
            source_paper="arXiv:2211.16000v1", provenance="Declared resimulation; IC/BC/grids/strides selected here")
        manifest.pop("authors_commit")
    if args.method == "filtered-wsindy":
        manifest["preprocessing"] = filter_protocol(args.filter_prior)
        if args.profile == "consistency":
            manifest["preprocessing"].update(width="Common side w=max(1,floor(min(2*(binom(p_max,2)*sigma_est^2/prior)^(1/D),prod(2*m_d+1)^(1/D)/2)))",
                adaptation="Volume cap as in Section 5.4; extend degree-6 rule to the degree-8 HKS library",
                scope="Consistency PDE benchmark; raw/filtered use identical bump tests and absolute STLS")
    if args.regression == "lasso":
        import sklearn
        manifest.update(lasso=lasso_protocol(), sklearn=sklearn.__version__,
            thresholds=np.logspace(-4, 0, 100).tolist(), threshold_units="ratio of alpha_max",
            allow_empty=True, selection_rule="LASSO penalized path / WSINDy projection-support loss")
        if "preprocessing" in manifest:
            manifest["preprocessing"]["scope"] = "Identical moving-average preprocessing; LASSO regression replaces MSTLS"
    if args.compare_to:
        baseline = json.loads(args.compare_to.with_suffix(".manifest.json").read_text())
        if baseline.get("method", "wsindy") != "wsindy":
            raise ValueError("--compare-to must reference an original WSINDy run.")
        for key in ("profile", "root_seed", "noise_ratios", "trials", "benchmarks", "rng", "noise", "dtype",
                    "thresholds", "ls_solver", "state_scale_rule", "threshold_units", "allow_empty", "numpy", "scipy"):
            if json.loads(json.dumps(manifest[key])) != baseline[key]:
                raise ValueError(f"Paired baseline differs in {key}.")
        for filename in ("methods.py", "simulation_generation.py"):
            if manifest["source_sha256"][filename] != baseline["source_sha256"][filename]:
                raise ValueError(f"Paired baseline numerical source differs: {filename}.")
        manifest["paired_baseline_protocol_id"] = baseline["protocol_id"]
    manifest["protocol_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["workers"] = args.workers
    manifest["thread_limits"] = {key: os.environ.get(key) for key in
                                ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    return manifest


def read_records(path, *, repair=False, compact=False):
    """An incomplete final append is ignored, or repaired only when resuming."""
    if not path.exists():
        return []
    rows = []
    with path.open("r+b" if repair else "rb") as source:
        while True:
            start = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if source.read(1):
                    raise ValueError(f"Invalid record inside {path}")
                if repair:
                    source.truncate(start)
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid record inside {path}") from error
            if compact:
                row = {k: v for k, v in row.items() if k not in
                       ("coefficients", "losses", "thresholds", "equations", "state_scales", "coordinate_scales",
                        "normalized_coefficients", "column_rms", "response_rms", "alpha_max", "selected_alpha",
                        "kkt_errors", "solver_warnings")}
            rows.append(row)
    return rows


def _saved_benchmark(args, name, protocol_id, path):
    existing = read_records(path, repair=args.resume, compact=True)
    if any(r.get("protocol_id") != protocol_id for r in existing):
        raise ValueError(f"Cannot mix protocols in {path}")
    if any(r.get("method", "wsindy") != args.method for r in existing):
        raise ValueError(f"Cannot mix methods in {path}")
    completed = {(r["name"], r["noise_ratio"], r["trial"]) for r in existing}
    if len(completed) != len(existing):
        raise ValueError(f"Duplicate trial keys in {path}")
    if len(completed) == len(args.noise_ratios)*args.trials:
        return name
    local = argparse.Namespace(**vars(args))
    local.benchmarks = [name]
    with path.open("a") as raw:
        def save(record):
            record["protocol_id"] = protocol_id
            raw.write(json.dumps(record, allow_nan=False)+"\n")
            raw.flush()
        run(local, on_result=save, completed=completed, keep_records=False)
    return name


def summary_rows(args, records):
    summaries = []
    for name in args.benchmarks:
        for ratio in args.noise_ratios:
            group = [r for r in records if r["name"] == name and r["noise_ratio"] == ratio]
            if not group:
                continue
            n = len(group)
            exact = sum(r["exact_support"] for r in group)
            p, z = exact/n, 1.959963984540054
            center = (p+z*z/(2*n))/(1+z*z/n)
            half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            row = dict(name=name, noise_ratio=ratio, trials=n, exact_count=exact, exact_rate=p,
                       exact_ci95_low=center-half, exact_ci95_high=center+half)
            row["subset_support_rate"] = sum(r["fp"] == 0 for r in group)/n if all("fp" in r for r in group) else None
            row["method"] = getattr(args, "method", "wsindy")
            for field in ("tpr", "e_inf", "e2", "runtime", "threshold"):
                values = np.array([r[field] for r in group])
                row.update({field+"_mean": float(values.mean()),
                    field+"_sd": float(values.std(ddof=1)) if n > 1 else None,
                    field+"_median": float(np.median(values)), field+"_p90": float(np.quantile(values, .9))})
            summaries.append(row)
    return summaries


def _atomic_text(path, content):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def paired_records(args, baseline, filtered):
    """Use exactly the same available trial keys and noise seeds in both rows."""
    def index(records):
        result = {}
        for record in records:
            key = (record["name"], record["noise_ratio"], record["trial"])
            if key in result:
                raise ValueError(f"Duplicate comparison trial: {key}")
            if (key[0] not in args.benchmarks or key[1] not in args.noise_ratios
                    or not 0 <= key[2] < args.trials):
                raise ValueError(f"Comparison trial outside requested schedule: {key}")
            result[key] = record
        return result
    original, smooth = index(baseline), index(filtered)
    if not smooth.keys() <= original.keys():
        raise ValueError("Filtered results contain trials absent from the original baseline.")
    keys = sorted(smooth)
    for key in keys:
        if original[key]["seed"] != smooth[key]["seed"]:
            raise ValueError(f"Comparison noise seed mismatch: {key}")
        if original[key].get("noise_std") != smooth[key].get("noise_std"):
            raise ValueError(f"Comparison injected noise mismatch: {key}")
    return [original[k] for k in keys], [smooth[k] for k in keys]


def format_comparison_report(args, baseline, filtered):
    original, smooth = paired_records(args, baseline, filtered)
    requested = len(args.benchmarks)*len(args.noise_ratios)*args.trials
    def link(path):
        return Path(os.path.relpath(path.resolve(), args.comparison_output.parent.resolve())).as_posix()
    lines = ["# WSINDy benchmark comparison", "",
        f"Original WSINDy: {len(baseline):,}/{requested:,} trials; "
        f"Filtered WSINDy: {len(filtered):,}/{requested:,} trials; {len(smooth):,} completed pairs. "
        f"Status: {'complete' if len(smooth) == requested else 'in progress'}.", "",
        f"Schedule: {args.trials} trials per method at noise ratios "
        f"`{', '.join(f'{r:g}' for r in args.noise_ratios)}`; root seed {args.seed}. "
        "KS, NLS, RD are primary benchmarks; IB, KdV, NS are supplementary. "
        "Each PDE uses one fixed true equation, coefficients, initial condition and clean trajectory. "
        "Only observation noise changes. Zero-noise repeats use identical observations.", "",
        "The two rows at each noise level use exactly the same completed trial indices and input seeds. "
        "While filtering is running, both rows summarize only those paired trials; the full original "
        "100-trial results remain available in the individual report. Pending metrics are shown as —.", "",
        f"[Original results]({link(args.compare_to)}) · "
        f"[Filtered results]({link(args.output)}) · "
        f"[Filtered protocol]({link(args.output.with_suffix('.manifest.json'))}) · "
        f"[Execution status]({link(args.output.with_suffix('.status.json'))}) · "
        f"[Paired CSV summary]({link(args.output.with_suffix('.paired.summary.csv'))})", "",
        filter_description(args.filter_prior), "",
        f"Both methods use the `{args.profile}` regression profile, identical original grids, libraries, "
        "test supports/degrees/strides, and 50 thresholds. Noise ratios use clean-component RMS as in "
        "the original PDE paper. Raw baseline records and its original protocol are preserved.", "",
        "TPR = TP/(TP+FN+FP); Exact is the percentage of exact support recoveries. E_inf is maximum "
        "relative error over true nonzero coefficients; E2 is relative coefficient L2 error. "
        "Metrics pool coupled equations and include failed identifications. Values are mean ± sample SD. "
        "Filtered runtime includes noise estimation and averaging; loading, noise generation and reports "
        "are excluded. Timings come from separate runs with recorded concurrency and are not paired "
        "serial speed measurements.", ""]
    if args.profile == "consistency":
        lines = ["# Consistency paper: two additional PDE benchmarks", "",
            consistency_description(), "", consistency_filter_description(args.filter_prior), "",
            f"WSINDy: {len(baseline):,}/{requested:,}; Filtered WSINDy: {len(filtered):,}/{requested:,}; "
            f"{len(smooth):,} completed pairs. Status: {'complete' if len(smooth) == requested else 'in progress'}.", "",
            f"Schedule: {args.trials} trials per method at noise ratios "
            f"`{', '.join(f'{r:g}' for r in args.noise_ratios)}`; root seed {args.seed}. "
            "Each pair uses the same noise seed and injected standard deviation. Rows summarize only "
            "completed matching trials. Each PDE has a fixed true equation and clean trajectory; "
            "zero-noise repeats are identical.", "",
            "TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E_inf is maximum relative error over "
            "true coefficients; E2 is relative coefficient L2 error. Mean ± sample SD includes failures. "
            "Median seconds includes filtering, excludes loading/noise/generation, and is measured under "
            "the recorded concurrency. Subset-support recovery is included in CSV/JSON summaries.", "",
            f"[Raw results]({link(args.compare_to)}) · [Filtered results]({link(args.output)}) · "
            f"[Protocol]({link(args.output.with_suffix('.manifest.json'))}) · "
            f"[Execution status]({link(args.output.with_suffix('.status.json'))})", ""]
    for name in args.benchmarks:
        spec = BENCHMARKS[name]
        lines.extend([f"## {name}", "", *true_equation_lines(spec)])
        if spec.suite == "consistency":
            config = generated_config(name)
            lines.extend([f"Declared IC: `{config['initial_condition']}`; periodic boundary; "
                f"x in `{config['x_interval']}`, t in `{config['time_interval']}`. "
                f"Fine simulation: `{config['fine_shape']}`; subsampling: `{config['subsample']}`.", ""])
        lines.extend([
            f"Data shape: `{spec.shape}`; components `{spec.component_names}`; "
            f"`m={spec.half_widths}`, `s={spec.strides}`, `p={spec.degrees}`.", "",
            "| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |",
            "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for ratio in args.noise_ratios:
            for method, records in (("wsindy", original), ("filtered-wsindy", smooth)):
                group = [r for r in records if r["name"] == name and r["noise_ratio"] == ratio]
                if not group:
                    lines.append(f"| {ratio:g} | {METHOD_LABELS[method]} | 0 | — | — | — | — | — |")
                    continue
                cells = []
                for field in ("tpr", "e_inf", "e2"):
                    values = np.array([r[field] for r in group])
                    cells.append(f"{values.mean():.6g} ± {values.std(ddof=1):.3g}"
                                 if len(values) > 1 else f"{values[0]:.6g}")
                exact = 100*np.mean([r["exact_support"] for r in group])
                runtime = np.median([r["runtime"] for r in group])
                lines.append(f"| {ratio:g} | {METHOD_LABELS[method]} | {len(group)} | {exact:.1f}% | "
                             + " | ".join(cells)+f" | {runtime:.3g} |")
        lines.append("")
    import re
    prefixes = []
    for path in getattr(args, "include_report", []):
        if path.resolve() == args.comparison_output.resolve():
            raise ValueError("An included report must differ from the comparison output.")
        def rebase(match):
            target = match.group(2)
            if target.startswith(("https:", "http:", "#", "/")):
                return match.group(0)
            relative = os.path.relpath((path.parent/target).resolve(), args.comparison_output.parent.resolve())
            return match.group(1)+Path(relative).as_posix()+")"
        prefixes.append(re.sub(r"(\[[^\]]*\]\()([^\s)]+)\)", rebase, path.read_text()).rstrip())
    return "\n\n---\n\n".join([*prefixes, "\n".join(lines)])


def refresh_comparison(args, filtered):
    manifest = json.loads(args.compare_to.with_suffix(".manifest.json").read_text())
    raw_dir = args.compare_to.parent/(args.compare_to.stem+"_trials")
    baseline = [r for name in args.benchmarks for r in read_records(raw_dir/(name+".jsonl"), compact=True)]
    if any(r.get("protocol_id") != manifest["protocol_id"] for r in baseline):
        raise ValueError("Original comparison records do not match their protocol.")
    original, smooth = paired_records(args, baseline, filtered)
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(args.comparison_output, format_comparison_report(args, baseline, filtered))
    local = argparse.Namespace(**vars(args))
    summaries = []
    for method, records in (("wsindy", original), ("filtered-wsindy", smooth)):
        local.method = method
        summaries.extend(summary_rows(local, records))
    summaries.sort(key=lambda r: (args.benchmarks.index(r["name"]), r["noise_ratio"],
                                 0 if r["method"] == "wsindy" else 1))
    _atomic_text(args.output.with_suffix(".paired.summary.json"), json.dumps(summaries, indent=2, allow_nan=False)+"\n")
    if summaries:
        _write_csv(args.output.with_suffix(".paired.summary.csv"), summaries)


def _write_csv(path, rows):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def refresh_reports(args, raw_dir):
    records = [row for name in args.benchmarks for row in read_records(raw_dir/(name+".jsonl"), compact=True)]
    summaries = summary_rows(args, records)
    _atomic_text(args.output, format_report(args, records))
    _atomic_text(args.output.with_suffix(".summary.json"), json.dumps(summaries, indent=2, allow_nan=False)+"\n")
    if summaries:
        _write_csv(args.output.with_suffix(".summary.csv"), summaries)
    if args.compare_to:
        refresh_comparison(args, records)
    return len(records)


def aggregate_comparisons(paths, output):
    """Merge saved methods, checking the observational protocol and every seed."""
    runs = []
    for path in paths:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        raw_dir = path.parent/(path.stem+"_trials")
        # Historical manifests can contain SG. Their immutable files remain
        # readable, but only active polynomial benchmarks enter this report.
        records = [r for name in manifest["benchmarks"] if name in BENCHMARKS
                   for r in read_records(raw_dir/(name+".jsonl"), compact=True)]
        if any(r.get("protocol_id") != manifest["protocol_id"] for r in records):
            raise ValueError(f"Comparison records differ from manifest: {path}")
        index = {(r["name"], r["noise_ratio"], r["trial"]): r for r in records}
        if len(index) != len(records):
            raise ValueError(f"Duplicate records: {path}")
        for r in records:
            if (r["name"] not in manifest["benchmarks"] or r["noise_ratio"] not in manifest["noise_ratios"]
                    or not 0 <= r["trial"] < manifest["trials"]):
                raise ValueError(f"Record outside protocol: {path}")
        runs.append((path, manifest, index))
    names = [name for name in BENCHMARKS if any(name in m["benchmarks"] for _, m, _ in runs)]
    lines = ["# WSINDy benchmark: MSTLS and LASSO", "",
        "Raw and filtered observations are each fitted with the existing MSTLS/STLS profile and LASSO. "
        "For each PDE/noise condition, all rows use the intersection of completed trial keys across "
        "the four methods. Seeds and injected noise are checked per trial. Each PDE has one fixed "
        "equation and trajectory; only observation noise varies. Zero-noise repeats are identical.", "",
        lasso_description(), "",
        "The original six polynomial PDEs use the existing author-code WSINDy construction and component RMS "
        "noise convention. HKS/VBG use the existing declared resimulations, physical bump tests and "
        "centered sample-standard-deviation noise convention. Their MSTLS-labelled baseline uses "
        "the consistency paper's absolute STLS profile. Libraries, weak supports and preprocessing "
        "are fixed within each suite; the two suites have different protocols.", "",
        "TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E2 is relative coefficient L2 error; "
        "E_inf is maximum relative error over true coefficients. Mean ± sample SD includes failed "
        "identifications. LASSO errors include shrinkage. Timings include preprocessing/weak systems/"
        "regression and exclude loading/noise/reporting; recorded concurrency differs between runs.", "",
        "## Saved runs", ""]
    all_complete = True
    summaries = []
    for path, manifest, index in runs:
        relative = Path(os.path.relpath(path.resolve(), output.parent.resolve())).as_posix()
        active_names = [name for name in manifest["benchmarks"] if name in BENCHMARKS]
        requested = len(active_names)*len(manifest["noise_ratios"])*manifest["trials"]
        label = METHOD_LABELS[manifest.get("method", "wsindy")]+" + "+manifest.get("regression", "mstls").upper()
        lines.append(f"- [{label}: {', '.join(active_names)}]({relative}): {len(index):,}/{requested:,} active polynomial fits.")
        all_complete &= len(index) == requested
    lines.extend(["", f"Overall status: {'complete' if all_complete else 'in progress'}.", ""])
    for name in names:
        selected = [(path, m, ix) for path, m, ix in runs if name in m["benchmarks"]]
        selected.sort(key=lambda run: (run[1].get("regression", "mstls") == "lasso",
                                      run[1].get("method", "wsindy") == "filtered-wsindy"))
        labels = [(m.get("method", "wsindy"), m.get("regression", "mstls")) for _, m, _ in selected]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate method/regression for {name}")
        reference = selected[0][1]
        for path, m, _ in selected[1:]:
            for key in ("root_seed", "noise_ratios", "trials", "rng", "noise", "dtype", "profile", "state_scale_rule"):
                if m[key] != reference[key]:
                    raise ValueError(f"Comparison input differs in {name}/{key}: {path}")
            if m["benchmarks"][name] != reference["benchmarks"][name]:
                raise ValueError(f"Comparison dataset/library differs: {name}/{path}")
            # Compare preprocessing only between runs with the same method.
            peers = [other for _, other, _ in selected if other.get("method", "wsindy") == m.get("method", "wsindy")]
            for other in peers:
                left, right = dict(m.get("preprocessing", {})), dict(other.get("preprocessing", {}))
                left.pop("scope", None); right.pop("scope", None)
                if left != right:
                    raise ValueError(f"Comparison filter differs: {name}/{path}")
        keys = set.intersection(*(set(ix) for _, _, ix in selected))
        keys = {key for key in keys if key[0] == name}
        for key in keys:
            source = selected[0][2][key]
            for path, _, ix in selected[1:]:
                for field in ("seed", "noise_std", "G_shape", "b_shape"):
                    if ix[key].get(field) != source.get(field):
                        raise ValueError(f"Comparison {field} mismatch: {key}/{path}")
        spec = BENCHMARKS[name]
        lines.extend([f"## {name}", "", *true_equation_lines(spec)])
        if spec.suite == "consistency":
            config = generated_config(name)
            lines.extend([f"Declared resimulation: IC `{config['initial_condition']}`; periodic boundary; "
                f"x in `{config['x_interval']}`, t in `{config['time_interval']}`.", ""])
        lines.extend([f"Observation shape `{spec.shape}`; library {len(spec.library())} terms; "
            f"weak half-widths `{spec.half_widths}`, strides `{spec.strides}`.", "",
            "| Noise ratio | Method | Paired trials | Exact | TPR | E_inf | E2 | Median seconds |",
            "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for ratio in reference["noise_ratios"]:
            for _, manifest, index in selected:
                method, regression = manifest.get("method", "wsindy"), manifest.get("regression", "mstls")
                group = [index[key] for key in sorted(keys) if key[1] == ratio]
                label = METHOD_LABELS[method]+" + "+regression.upper()
                if not group:
                    lines.append(f"| {ratio:g} | {label} | 0 | — | — | — | — | — |")
                    continue
                cells = []
                for field in ("tpr", "e_inf", "e2"):
                    values = np.array([r[field] for r in group])
                    cells.append(f"{values.mean():.6g} ± {values.std(ddof=1):.3g}" if len(values) > 1 else f"{values[0]:.6g}")
                lines.append(f"| {ratio:g} | {label} | {len(group)} | "
                    f"{100*np.mean([r['exact_support'] for r in group]):.1f}% | "+" | ".join(cells)
                    +f" | {np.median([r['runtime'] for r in group]):.3g} |")
                local = argparse.Namespace(benchmarks=[name], noise_ratios=[ratio], method=method)
                summaries.extend(dict(row, regression=regression) for row in summary_rows(local, group))
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(output, "\n".join(lines))
    _atomic_text(output.with_suffix(".summary.json"), json.dumps(summaries, indent=2)+"\n")
    if summaries:
        _write_csv(output.with_suffix(".summary.csv"), summaries)
    return all_complete


def _trial_chunk(args, name, keys):
    local = argparse.Namespace(**vars(args))
    local.benchmarks = [name]
    completed = {(name, ratio, trial) for ratio in args.noise_ratios for trial in range(args.trials)}-set(keys)
    return run(local, completed=completed)


def run_lasso_chunks(args, raw_dir, protocol_id, update):
    """Parallelize trial chunks so the largest PDE does not serialize a suite."""
    tasks = []
    for name in args.benchmarks:
        existing = read_records(raw_dir/(name+".jsonl"), repair=args.resume, compact=True)
        if any(r.get("protocol_id") != protocol_id for r in existing):
            raise ValueError(f"Cannot mix protocols for {name}")
        done = {(r["name"], r["noise_ratio"], r["trial"]) for r in existing}
        if len(done) != len(existing):
            raise ValueError(f"Duplicate trial keys for {name}")
        for ratio in args.noise_ratios:
            remaining = [(name, ratio, trial) for trial in range(args.trials) if (name, ratio, trial) not in done]
            tasks.extend((name, remaining[start:start+10]) for start in range(0, len(remaining), 10))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        remaining_tasks = iter(tasks)
        pending = set()
        def submit_next():
            task = next(remaining_tasks, None)
            if task is not None:
                pending.add(pool.submit(_trial_chunk, args, *task))
        for _ in range(2*args.workers):
            submit_next()
        last_update = perf_counter()
        while pending:
            finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                records = future.result()
                if records:
                    with (raw_dir/(records[0]["name"]+".jsonl")).open("a") as target:
                        for record in records:
                            record["protocol_id"] = protocol_id
                            target.write(json.dumps(record, allow_nan=False)+"\n")
                submit_next()
            if not pending or perf_counter()-last_update >= 30:
                update()
                last_update = perf_counter()


def plot_results(args):
    """Static scientific plots from saved summaries; no fitting or recomputation."""
    os.environ["MPLCONFIGDIR"] = str(args.output.parent.resolve()/".mplcache")
    os.environ["XDG_CACHE_HOME"] = str(args.output.parent.resolve()/".mplcache"/"xdg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries = json.loads(args.output.with_suffix(".summary.json").read_text())
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name in args.benchmarks:
        rows = sorted((r for r in summaries if r["name"] == name), key=lambda r: r["noise_ratio"])
        if not rows:
            continue
        x = [r["noise_ratio"] for r in rows]
        for axis, field in zip(axes.ravel(), ("tpr_mean", "exact_rate", "e2_mean", "e_inf_mean")):
            axis.plot(x, [max(r[field], 1e-12) if field.startswith("e") and field != "exact_rate" else r[field]
                          for r in rows], label=name, marker=".", markersize=3)
    for axis, label in zip(axes.ravel(), ("Mean support TPR", "Exact support recovery probability",
                                          "Mean relative coefficient error E2", "Mean true-coefficient error E_inf")):
        axis.set(xlabel="Noise / clean standard deviation" if args.profile == "consistency" else "Noise / clean RMS", ylabel=label)
        axis.grid(alpha=.25)
    axes[0, 0].set_ylim(-.03, 1.03)
    axes[0, 1].set_ylim(-.03, 1.03)
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(ncol=3)
    complete = all(r["trials"] == args.trials for r in summaries) and len(summaries) == len(args.benchmarks)*len(args.noise_ratios)
    sizes = [r["trials"] for r in summaries]
    sample_label = f"n={min(sizes)}–{max(sizes)}" if sizes and min(sizes) != max(sizes) else f"n={sizes[0] if sizes else 0}"
    figure.suptitle(f"{METHOD_LABELS[args.method]} {args.profile} — "
                   f"{'complete' if complete else 'partial results'}; {sample_label} per noise level")
    figure.savefig(args.output.with_suffix(".png"), dpi=180)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    args = parse_arguments()
    if args.comparison_reports:
        if not args.report_only:
            raise ValueError("--comparison-reports requires --report-only")
        complete = aggregate_comparisons(args.comparison_reports, args.comparison_output)
        print(f"Comparison {'complete' if complete else 'in progress'}: {args.comparison_output}")
        return
    if args.download_only:
        run(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.parent/(args.output.stem+"_trials")
    raw_dir.mkdir(exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = protocol_manifest(args)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["protocol_id"] != manifest["protocol_id"]:
            raise ValueError("Saved protocol differs in source, settings, or environment; choose a new output path.")
        if not args.resume and not args.report_only:
            raise FileExistsError("This run already exists; use --resume or choose a new --output.")
    else:
        if args.report_only or any(raw_dir.glob("*.jsonl")):
            raise ValueError("Trial records require their original protocol manifest.")
        _atomic_text(manifest_path, json.dumps(manifest, indent=2)+"\n")
        snapshot = args.output.parent/(args.output.stem+"_source")
        snapshot.mkdir(exist_ok=True)
        for filename in (*manifest["source_sha256"], "requirements.txt", "README.md"):
            shutil.copyfile(Path(__file__).parent/filename, snapshot/filename)
    def status(state, count, error=None):
        _atomic_text(args.output.with_suffix(".status.json"), json.dumps(dict(state=state,
            completed=count, requested=len(args.benchmarks)*len(args.noise_ratios)*args.trials,
            pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), error=error), indent=2)+"\n")
    count = refresh_reports(args, raw_dir)
    try:
        if not args.report_only:
            status("running", count)
            last_plot = perf_counter()
            if args.regression == "lasso":
                def update():
                    nonlocal count
                    count = refresh_reports(args, raw_dir)
                    status("running", count)
                run_lasso_chunks(args, raw_dir, manifest["protocol_id"], update)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    pending = {pool.submit(_saved_benchmark, args, name, manifest["protocol_id"], raw_dir/(name+".jsonl"))
                               for name in args.benchmarks}
                    while pending:
                        finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                        for future in finished:
                            print(f"Completed {future.result()}", flush=True)
                        count = refresh_reports(args, raw_dir)
                        status("running", count)
                        if count and (finished or perf_counter()-last_plot > 600):
                            plot_results(args)
                            last_plot = perf_counter()
        count = refresh_reports(args, raw_dir)
        plot_results(args)
        status("complete" if count == len(args.benchmarks)*len(args.noise_ratios)*args.trials else "incomplete", count)
    except BaseException as error:
        status("failed", count, f"{type(error).__name__}: {error}")
        raise
    print(f"Saved {count} trials, {args.output}, summaries, manifest, PNG/PDF plots, and {raw_dir}")


if __name__ == "__main__":
    main()
