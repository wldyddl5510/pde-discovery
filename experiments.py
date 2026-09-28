"""Run small PDE-discovery comparisons and write error/runtime tables."""

import argparse
import os
import platform
import shlex
import signal
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from methods import (
    build_debiased_wsindy_system, build_sampled_wsindy_system,
    build_sindy_system, build_wsindy_system,
    debiased_wsindy, fit_sampled_wsindy_system,
    polynomial_library_terms,
    sindy, wsindy, wendy, wendy_mle,
)
from simulation_generation import (
    BURGERS_CRITICAL_NOISE_STD,
    generate_anisotropic_porous_medium,
    generate_anisotropic_porous_medium_3d,
    generate_anisotropic_porous_medium_3d_samples,
    generate_nonlinear_viscous_burgers,
    sample_nonlinear_viscous_burgers,
)
from utils import (
    MovingAverageEstimator, estimate_grid_noise_std,
    filter_grid_moving_average, paper_filter_width,
)


METHOD_LABELS = {
    "sindy-ols": "SINDy (OLS)",
    "wsindy-ols": "WSINDy (OLS)",
    "sindy-lasso": "SINDy (LASSO)",
    "wsindy-lasso": "WSINDy (LASSO)",
    "wsindy-mstls": "WSINDy (MSTLS)",
    "wendy-lasso": "WENDy (LASSO)",
    "wendy-ols": "WENDy (OLS)",
    "wendy-mle-lasso": "WENDy-MLE (LASSO)",
    "wendy-mle": "WENDy-MLE (unpenalized)",
    "debiased-wsindy": "Debiased WSINDy",
    "sampled-wsindy-ols": "Sampled WSINDy (OLS)",
    "sampled-wsindy-lasso": "Sampled WSINDy (LASSO)",
    "sampled-wsindy-mstls": "Sampled WSINDy (MSTLS)",
    "sindy-mstls": "SINDy (MSTLS)",
    "paper-filtered-wsindy": "Paper filtered WSINDy (MSTLS)",
    "sampled-plug-in-wsindy": "Sampled plug-in WSINDy (MSTLS)",
    "sampled-debiased-wsindy": "Sampled debiased WSINDy (MSTLS)",
}
SAMPLED_METHODS = ("sampled-wsindy-ols", "sampled-wsindy-lasso", "sampled-wsindy-mstls")
DEFAULT_METHODS = ["sindy-ols", "wsindy-ols", "sindy-lasso", "wsindy-lasso", "wsindy-mstls"]
GENERATORS = {
    "anisotropic_porous_medium": generate_anisotropic_porous_medium,
    "anisotropic_porous_medium_3d": generate_anisotropic_porous_medium_3d,
    "nonlinear_viscous_burgers": generate_nonlinear_viscous_burgers,
}
BURGERS_METHODS = [
    "sindy-ols", "sindy-mstls", "wsindy-ols", "wsindy-mstls",
    "paper-filtered-wsindy", "sampled-wsindy-mstls",
    "sampled-plug-in-wsindy", "sampled-debiased-wsindy",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=GENERATORS, default="anisotropic_porous_medium")
    parser.add_argument("--methods", nargs="+", choices=METHOD_LABELS, default=DEFAULT_METHODS)
    parser.add_argument("--noise-ratios", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--noise-multipliers", nargs="+", type=float, default=[0.0, 1.0, 2.0],
                        help="Burgers only: sigma / sqrt(0.01/3).")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Burgers only: independent noise and sample realizations.")
    parser.add_argument("--nx", type=int, default=None, help="Default: 64 in 2D, 32 in 3D.")
    parser.add_argument("--ny", type=int, default=None, help="Default: 64 in 2D, 32 in 3D.")
    parser.add_argument("--nz", type=int, default=None, help="3D only; default: 32.")
    parser.add_argument("--nt", type=int, default=None, help="Default: 32 in 2D, 16 in 3D.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sindy-rho-1", type=float, default=1e-4)
    parser.add_argument("--wsindy-rho-1", type=float, default=1e-6)
    parser.add_argument("--wendy-rho-1", type=float, default=1e-4)
    parser.add_argument("--wendy-mle-rho-1", type=float, default=1e-3)
    parser.add_argument("--debiased-lambda", type=float, default=0.0,
                        help="Penalty in 0.5*||Y_db-X_db beta||^2 + lambda*||beta||_1.")
    parser.add_argument("--debiased-regression", choices=("lasso", "mstls"),
                        default="lasso", help="Regression rule for the corrected weak system.")
    parser.add_argument("--sampled-lambda", type=float, default=0.2,
                        help="L1 penalty for ordinary WSINDy on iid point samples.")
    parser.add_argument("--pilot-bandwidths", nargs="+", type=float, default=None,
                        help="Physical box-kernel bandwidths in x, y, z, t order.")
    parser.add_argument("--n-observations", type=int, default=None,
                        help="Total iid observations, split equally for the debiased method.")
    parser.add_argument("--wendy-alpha", type=float, default=1e-10)
    parser.add_argument("--wendy-mle-alpha", type=float, default=0.0)
    parser.add_argument("--mle-noise-std", type=float, default=None,
                        help="Working MLE noise std; default uses the generator's known noise std.")
    parser.add_argument("--mle-max-iter", type=int, default=1000)
    parser.add_argument("--mle-tol", type=float, default=1e-6)
    parser.add_argument("--max-reweights", type=int, default=100)
    parser.add_argument("--reweight-tol", type=float, default=1e-6)
    parser.add_argument("--disable-normality-stop", action="store_true")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--half-widths", nargs="+", type=int, default=None)
    parser.add_argument("--strides", nargs="+", type=int, default=None)
    parser.add_argument("--test-degrees", nargs="+", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=None,
                        help="Optional wall-time limit per estimator call (Unix).")
    parser.add_argument("--output", type=Path, default=Path("results.md"))
    parser.add_argument("--append", action="store_true",
                        help="Append this experiment's report, preserving previous results.")
    args = parser.parse_args()
    if args.instance == "nonlinear_viscous_burgers":
        if args.ny is not None or args.nz is not None:
            parser.error("The Burgers instance has one spatial dimension; omit --ny and --nz.")
        args.nx = 256 if args.nx is None else args.nx
        args.nt = 257 if args.nt is None else args.nt
        args.n_observations = 524288 if args.n_observations is None else args.n_observations
        if args.nx < 32 or args.nt < 17 or args.n_observations < 2:
            parser.error("Burgers needs nx>=32, nt>=17, and n-observations>=2.")
        if (args.nx % 8 != 0 or (args.nt - 1) % 8 != 0):
            parser.error("Burgers needs nx divisible by 8 and nt-1 divisible by 8.")
        if args.test_degrees is not None or args.half_widths is not None or args.strides is not None:
            parser.error("The Burgers comparison fixes the paper bump and physical test supports.")
        if args.methods == DEFAULT_METHODS:
            args.methods = BURGERS_METHODS.copy()
        if any(method not in BURGERS_METHODS for method in args.methods):
            parser.error("The Burgers instance supports only its eight comparison methods.")
        if args.replicates < 1 or args.repeats < 1:
            parser.error("--replicates and --repeats must be positive.")
        if any(not np.isfinite(level) or level < 0 for level in args.noise_multipliers):
            parser.error("--noise-multipliers must contain finite nonnegative values.")
        return args
    spatial_dim = 3 if args.instance == "anisotropic_porous_medium_3d" else 2
    if spatial_dim == 2 and args.nz is not None:
        parser.error("--nz is only used by a 3D instance.")
    if args.nx is None:
        args.nx = 32 if spatial_dim == 3 else 64
    if args.ny is None:
        args.ny = 32 if spatial_dim == 3 else 64
    if spatial_dim == 3 and args.nz is None:
        args.nz = 32
    if args.nt is None:
        args.nt = 16 if spatial_dim == 3 else 32
    for flag in ("half_widths", "strides", "test_degrees"):
        values = getattr(args, flag)
        if values is not None and len(values) != spatial_dim + 1:
            parser.error(f"--{flag.replace('_', '-')} needs {spatial_dim + 1} values (space, then time).")
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
    if "debiased-wsindy" in args.methods:
        if args.methods != ["debiased-wsindy"] or spatial_dim != 3:
            parser.error("Run debiased-wsindy alone on the 3D instance.")
        if not np.isfinite(args.debiased_lambda) or args.debiased_lambda < 0:
            parser.error("--debiased-lambda must be finite and nonnegative.")
        if args.debiased_regression == "mstls" and args.debiased_lambda != 0:
            parser.error("MSTLS uses --thresholds; set --debiased-lambda to zero.")
        if args.debiased_regression == "lasso" and args.thresholds is not None:
            parser.error("--thresholds applies only to MSTLS.")
        if args.n_observations is not None and args.n_observations < 2:
            parser.error("--n-observations must be at least two.")
        if args.pilot_bandwidths is not None and (
            len(args.pilot_bandwidths) != 4
            or not np.all(np.isfinite(args.pilot_bandwidths))
            or any(width <= 0 for width in args.pilot_bandwidths)
        ):
            parser.error("--pilot-bandwidths needs four finite positive values.")
    if any(method in SAMPLED_METHODS for method in args.methods):
        if spatial_dim != 3 or any(method not in SAMPLED_METHODS for method in args.methods):
            parser.error("Run sampled-wsindy methods together on the 3D instance.")
        if not np.isfinite(args.sampled_lambda) or args.sampled_lambda < 0:
            parser.error("--sampled-lambda must be finite and nonnegative.")
        if "sampled-wsindy-lasso" in args.methods and args.sampled_lambda == 0:
            parser.error("sampled-wsindy-lasso requires a positive --sampled-lambda.")
        if args.n_observations is not None and args.n_observations < 2:
            parser.error("--n-observations must be at least two.")
        if args.thresholds is not None and "sampled-wsindy-mstls" not in args.methods:
            parser.error("--thresholds applies only to MSTLS.")
    if args.max_iter < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--max-iter and --tol must be positive, with finite --tol.")
    if args.mle_max_iter < 1 or not np.isfinite(args.mle_tol) or args.mle_tol <= 0:
        parser.error("--mle-max-iter and --mle-tol must be positive, with finite --mle-tol.")
    if args.timeout_seconds is not None:
        if not np.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            parser.error("--timeout-seconds must be finite and positive.")
        if not hasattr(signal, "setitimer"):
            parser.error("--timeout-seconds requires Unix interval timers.")
    for penalty in (args.sindy_rho_1, args.wsindy_rho_1, args.wendy_rho_1, args.wendy_mle_rho_1):
        if not np.isfinite(penalty) or penalty <= 0:
            parser.error("LASSO penalties must be finite and strictly positive.")
    if any(name.startswith("wendy-mle") for name in args.methods):
        if args.mle_noise_std is not None and (not np.isfinite(args.mle_noise_std) or args.mle_noise_std <= 0):
            parser.error("--mle-noise-std must be finite and positive.")
    return args


def call_estimator(estimator, inputs, settings, timeout_seconds):
    """Apply an optional deadline; the caller records failures and warnings."""
    if timeout_seconds is None:
        return estimator(*inputs, **settings)

    def time_limit_reached(signum, frame):
        raise TimeoutError(f"Estimator exceeded the {timeout_seconds:g} s wall-time limit.")

    previous_handler = signal.signal(signal.SIGALRM, time_limit_reached)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return estimator(*inputs, **settings)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _burgers_settings(data):
    """Fixed physical support ratio 1/16 and the paper's polynomial library."""
    x, = data.spatial_grid
    nx, nt = data.u_true.shape
    half_widths = (nx // 8, (nt - 1) // 8)
    strides = (max(1, half_widths[0] // 4), max(1, half_widths[1] // 4))
    centers = np.column_stack([
        axis.ravel() for axis in np.meshgrid(
            x[half_widths[0]:nx - half_widths[0]:strides[0]],
            data.time[half_widths[1]:nt - half_widths[1]:strides[1]],
            indexing="ij",
        )
    ])
    library = dict(
        min_derivative_order=0, max_derivative_order=6,
        min_polynomial_degree=0, max_polynomial_degree=6,
    )
    grid_tests = dict(
        half_widths=half_widths, strides=strides, test_function="paper"
    )
    sampled_tests = dict(
        domain_bounds=((-1.0, 1.0), (0.0, 1.5)), test_centers=centers,
        test_half_widths=(half_widths[0] * (x[1] - x[0]),
                          half_widths[1] * (data.time[1] - data.time[0])),
        test_function="paper",
    )
    return library, grid_tests, sampled_tests


def _burgers_pilot_bandwidths(data, sample_count, filter_width):
    """Map the paper's grid width to iid boxes, with 64 expected training neighbors."""
    x, = data.spatial_grid
    training_count = sample_count // 2
    density_fraction = np.sqrt(64.0 / training_count)
    grid_radii = ((filter_width - 1) * (x[1] - x[0]) / 2.0,
                  (filter_width - 1) * (data.time[1] - data.time[0]) / 2.0)
    density_radii = (density_fraction, 0.75 * density_fraction)
    return tuple(max(grid, floor) for grid, floor in zip(grid_radii, density_radii))


def _fit_burgers_method(method, data, samples, library, grid_tests,
                        sampled_tests, bandwidths, filter_width, thresholds):
    """Build one system and fit it; return the inputs for a truth-residual check."""
    if method.startswith("sindy-"):
        X, y = build_sindy_system(data.u_observed, data.spatial_grid, data.time, **library)
    elif method in ("wsindy-ols", "wsindy-mstls", "paper-filtered-wsindy"):
        observations = data.u_observed
        if method == "paper-filtered-wsindy":
            observations = filter_grid_moving_average(observations, filter_width)
        X, y = build_wsindy_system(
            observations, data.spatial_grid, data.time, **library, **grid_tests
        )
    elif method == "sampled-wsindy-mstls":
        X, y = build_sampled_wsindy_system(
            samples.evaluation_points, samples.evaluation_values,
            **library, **sampled_tests,
        )
    elif method == "sampled-plug-in-wsindy":
        pilot = MovingAverageEstimator(bandwidths).fit(
            samples.training_points, samples.training_values
        )
        predictions = pilot.predict(samples.evaluation_points)
        X, y = build_sampled_wsindy_system(
            samples.evaluation_points, predictions, **library, **sampled_tests,
        )
    elif method == "sampled-debiased-wsindy":
        X, y = build_debiased_wsindy_system(
            samples.training_points, samples.training_values,
            samples.evaluation_points, samples.evaluation_values,
            MovingAverageEstimator(bandwidths), **library, **sampled_tests,
        )
    else:
        raise ValueError(f"Unknown Burgers method: {method}")

    regression = "lasso" if method.endswith("-ols") else "mstls"
    beta = fit_sampled_wsindy_system(
        X, y, regression=regression,
        thresholds=thresholds if regression == "mstls" else None,
    )
    return beta, X, y


def run_burgers_experiments(args):
    """Compare grid and independent-sample methods on one Burgers solution."""
    thresholds = np.logspace(-4, 0, 100)
    results = []
    metadata = []
    for multiplier in args.noise_multipliers:
        noise_std = multiplier * BURGERS_CRITICAL_NOISE_STD
        for replicate in range(args.replicates):
            seed = args.seed + replicate
            data = generate_nonlinear_viscous_burgers(
                nx=args.nx, nt=args.nt, noise_std=noise_std, seed=seed
            )
            samples = sample_nonlinear_viscous_burgers(
                data, args.n_observations, seed=seed + 10000
            )
            library, grid_tests, sampled_tests = _burgers_settings(data)
            terms = polynomial_library_terms(1, **library)
            beta_true = np.array([data.true_coefficients.get(term, 0.0) for term in terms])
            support_points = np.prod([2 * width + 1 for width in grid_tests["half_widths"]])
            sigma_est = estimate_grid_noise_std(data.u_observed)
            filter_width = paper_filter_width(sigma_est, int(support_points))
            bandwidths = _burgers_pilot_bandwidths(data, args.n_observations, filter_width)
            pilot = MovingAverageEstimator(bandwidths).fit(
                samples.training_points, samples.training_values
            )
            prediction = pilot.predict(samples.evaluation_points[:10000])
            pilot_mse = float(np.mean((prediction - samples.evaluation_true[:10000])**2))
            metadata.append(dict(
                multiplier=multiplier, replicate=replicate, sigma_est=sigma_est,
                filter_width=filter_width, bandwidths=bandwidths,
                pilot_mse=pilot_mse, K=len(sampled_tests["test_centers"]),
                grid_n=data.u_true.size, iid_n=args.n_observations,
            ))
            for method in args.methods:
                start = perf_counter()
                try:
                    beta, X, y = _fit_burgers_method(
                        method, data, samples, library, grid_tests,
                        sampled_tests, bandwidths, filter_width, thresholds,
                    )
                    result = dict(
                        multiplier=multiplier, replicate=replicate, method=method,
                        status="ok", runtime=perf_counter() - start,
                        squared_error=float(np.sum((beta - beta_true)**2)),
                        truth_residual=float(np.linalg.norm(y - X @ beta_true) / np.linalg.norm(y)),
                        support_recovered=bool(np.array_equal(beta != 0, beta_true != 0)),
                        nonzero=int(np.count_nonzero(beta)),
                    )
                    print(
                        f"sigma/sigma_c={multiplier:g} seed={seed} {METHOD_LABELS[method]}: "
                        f"error={result['squared_error']:.6g}, residual={result['truth_residual']:.3g}, "
                        f"runtime={result['runtime']:.2f}s", flush=True,
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    result = dict(
                        multiplier=multiplier, replicate=replicate, method=method,
                        status="failed", runtime=perf_counter() - start,
                        reason=str(error),
                    )
                    print(f"sigma/sigma_c={multiplier:g} seed={seed} {method}: failed: {error}", flush=True)
                results.append(result)
    return results, metadata


def write_burgers_report(args, results, metadata):
    """Keep grid and iid estimates separate while reporting the same metrics."""
    lines = [
        "## Nonlinear viscous Burgers: paper-filter comparison", "",
        "PDE: `u_t = 0.01 u_xx - 0.5 d_x(u^2) - u^3 + 2u^2 + 1`.",
        "This is an adapted instance: periodic `x in [-1,1)`, `t in [0,1.5]`, and",
        "`u(x,0)=0.5+0.7 sin(pi*x)+0.25 sin(2*pi*x+0.3)`. The",
        "[Messenger–Bortz paper](https://arxiv.org/pdf/2211.16000) does not",
        "specify these initial/boundary data. A centered-difference, sparse-BDF",
        "solution is the numerical truth; the grid and iid observations share it.", "",
        f"- Grid: `{args.nx} x {args.nt}` (`n={metadata[0]['grid_n']}`); iid observations: "
        f"`n={args.n_observations}`, split equally into independent pilot and evaluation samples.",
        f"- `K={metadata[0]['K']}` tests, support half-widths `(0.25,0.1875)` in `(x,t)`, "
        "support volume ratio `1/16`; translated paper bump "
        "`exp(9/(r^2-1))` on `|r|<1`.",
        "- Library: spatial orders `0..6`, powers `u^0..u^6`; 43 nonzero columns "
        "after omitting positive derivatives of the constant. All sparse fits use "
        "MSTLS with `logspace(-4,0,100)` thresholds.",
        "- Paper filter: sixth-difference estimate of `sigma`, then the paper's "
        "equation (5.6) per-axis box width (periodic space, reflected time). For iid pilots, the "
        "grid width is mapped to physical units with a floor giving 64 expected "
        "training neighbors in an interior box. This is an adaptation to iid data.",
        f"- `sigma_c=sqrt(0.01/3)={BURGERS_CRITICAL_NOISE_STD:.6g}`. "
        f"Each level has {args.replicates} independent seeds starting at `{args.seed}`. "
        "Entries are mean squared coefficient error, support recovery count, "
        "mean relative system residual at the true coefficients, and median "
        "complete fit runtime (seconds). Data generation and width selection "
        "are excluded from runtime. OLS has no sparse support score.", "",
        "| sigma/sigma_c | estimated sigma | filter width | iid pilot bandwidths (x,t) | pilot MSE |",
        "| ---: | ---: | ---: | --- | ---: |",
    ]
    for multiplier in args.noise_multipliers:
        entries = [item for item in metadata if item["multiplier"] == multiplier]
        widths = sorted({item["filter_width"] for item in entries})
        bandwidths = entries[0]["bandwidths"]
        lines.append(
            f"| {multiplier:g} | {np.mean([item['sigma_est'] for item in entries]):.4g} | "
            f"{','.join(map(str, widths))} | ({bandwidths[0]:.5g}, {bandwidths[1]:.5g}) | "
            f"{np.mean([item['pilot_mse'] for item in entries]):.4g} |"
        )
    for title, methods in (
        ("Grid methods", [method for method in args.methods if not method.startswith("sampled-")]),
        ("Independent-point Monte Carlo methods", [method for method in args.methods if method.startswith("sampled-")]),
    ):
        if not methods:
            continue
        lines.extend(["", f"### {title}", "",
                      "| sigma/sigma_c | Method | Squared error | Support | True-system residual | Runtime (s) |",
                      "| ---: | --- | ---: | ---: | ---: | ---: |"])
        for multiplier in args.noise_multipliers:
            for method in methods:
                entries = [item for item in results if item["multiplier"] == multiplier
                           and item["method"] == method]
                successes = [item for item in entries if item["status"] == "ok"]
                if not successes:
                    lines.append(f"| {multiplier:g} | {METHOD_LABELS[method]} | failed | — | — | — |")
                    continue
                errors = [item["squared_error"] for item in successes]
                residuals = [item["truth_residual"] for item in successes]
                support = "—" if method.endswith("-ols") else (
                    f"{sum(item['support_recovered'] for item in successes)}/{len(entries)}"
                )
                lines.append(
                    f"| {multiplier:g} | {METHOD_LABELS[method]} | "
                    f"{np.mean(errors):.6g} | {support} | {np.mean(residuals):.4g} | "
                    f"{np.median([item['runtime'] for item in successes]):.3f} |"
                )
    lines.extend(["", "Grid and iid errors are separate comparisons because their observation "
                  "locations and quadrature differ. The paper reports 200 noise realizations; "
                  "these runs are a smaller numerical check, not a reproduction of its figure.",
                  "", "Reproduce with:", "",
                  "```sh",
                  "OPENBLAS_NUM_THREADS=1 python experiments.py --instance nonlinear_viscous_burgers "
                  f"--noise-multipliers {' '.join(f'{level:g}' for level in args.noise_multipliers)} "
                  f"--replicates {args.replicates} --n-observations {args.n_observations} "
                  "--append",
                  "```", ""])
    report = "\n".join(lines)
    if args.append and args.output.exists() and args.output.stat().st_size:
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write("\n\n" + report)
    else:
        args.output.write_text(report, encoding="utf-8")


def run_experiments(args):
    """Generate once per noise level, then time complete estimator calls."""
    methods = {
        "sindy-ols": (sindy, {"regression": "lasso", "rho_1": 0.0}),
        "wsindy-ols": (wsindy, {"regression": "lasso", "rho_1": 0.0}),
        "sindy-lasso": (sindy, {"regression": "lasso", "rho_1": args.sindy_rho_1}),
        "wsindy-lasso": (wsindy, {"regression": "lasso", "rho_1": args.wsindy_rho_1}),
        "wsindy-mstls": (wsindy, {"regression": "mstls"}),
        "wendy-lasso": (wendy, {"regression": "lasso", "rho_1": args.wendy_rho_1}),
        "wendy-ols": (wendy, {"regression": "lasso", "rho_1": 0.0}),
        "wendy-mle-lasso": (wendy_mle, {"regression": "lasso", "rho_1": args.wendy_mle_rho_1}),
        "wendy-mle": (wendy_mle, {"regression": "lasso", "rho_1": 0.0}),
    }
    results = {}
    for noise_ratio in args.noise_ratios:
        spatial_sizes = {"nx": args.nx, "ny": args.ny}
        if args.instance == "anisotropic_porous_medium_3d":
            spatial_sizes["nz"] = args.nz
        data = GENERATORS[args.instance](
            **spatial_sizes, nt=args.nt,
            noise_ratio=noise_ratio, seed=args.seed,
        )
        terms = polynomial_library_terms(len(data.spatial_grid))
        if not set(data.true_coefficients).issubset(terms):
            raise ValueError("The candidate library omits a true PDE term.")
        beta_true = np.array([data.true_coefficients.get(term, 0.0) for term in terms])
        results[noise_ratio] = {}

        for method_name in args.methods:
            estimator, settings = methods[method_name]
            settings = {**settings, "max_iter": args.max_iter, "tol": args.tol}
            if estimator is not sindy:
                settings.update(half_widths=args.half_widths, strides=args.strides, test_degrees=args.test_degrees)
            if settings["regression"] == "mstls":
                settings["thresholds"] = args.thresholds
            if estimator is wendy:
                settings.update(
                    alpha=args.wendy_alpha, max_reweights=args.max_reweights,
                    reweight_tol=args.reweight_tol,
                    normality_tol=None if args.disable_normality_stop else 1e-4,
                )
            if estimator is wendy_mle:
                settings.update(alpha=args.wendy_mle_alpha, max_iter=args.mle_max_iter, tol=args.mle_tol)
                settings["noise_std"] = data.noise_std if args.mle_noise_std is None else args.mle_noise_std
                if settings["noise_std"] == 0:
                    results[noise_ratio][method_name] = {
                        "status": "not_applicable", "squared_error": None, "runtime_seconds": None,
                        "reason": "The sigma=0 Gaussian likelihood is undefined; no working variance was supplied.",
                    }
                    print(f"noise={noise_ratio:g} {METHOD_LABELS[method_name]}: N/A (sigma=0)", flush=True)
                    continue
            inputs = (data.u_observed, data.spatial_grid, data.time)
            print(f"noise={noise_ratio:g} {METHOD_LABELS[method_name]}: starting", flush=True)
            durations = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    # Call zero is the warm-up; include only later calls in the median.
                    for call_index in range(args.repeats + 1):
                        start = perf_counter()
                        beta = call_estimator(estimator, inputs, settings, args.timeout_seconds)
                        elapsed = perf_counter() - start
                        if call_index > 0:
                            durations.append(elapsed)
                        print(f"  {'warm-up' if call_index == 0 else f'run {call_index}'}: {elapsed:.3f} s", flush=True)
            except (RuntimeError, np.linalg.LinAlgError, TimeoutError) as error:
                elapsed = perf_counter() - start
                results[noise_ratio][method_name] = {
                    "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                    "squared_error": None, "runtime_seconds": None,
                    "reason": str(error), "failed_attempt_seconds": elapsed,
                    "failed_call": "warm-up" if call_index == 0 else f"timed run {call_index}",
                    "durations": durations,
                    "warnings": list(dict.fromkeys(str(item.message) for item in caught)),
                }
                print(f"  {results[noise_ratio][method_name]['status']}: {error} ({elapsed:.3f} s)", flush=True)
                continue

            if beta.shape != beta_true.shape or not np.all(np.isfinite(beta)):
                raise ValueError(f"{method_name} returned an invalid coefficient vector.")
            # Compare all coefficients, including the true zeros, in original units.
            squared_error = float(np.sum((beta - beta_true)**2))
            runtime = float(np.median(durations))
            results[noise_ratio][method_name] = {
                "status": "warning" if caught else "ok",
                "squared_error": squared_error,
                "runtime_seconds": runtime,
                "durations": durations,
                "nonzero_coefficients": int(np.count_nonzero(beta)),
                "warnings": list(dict.fromkeys(str(item.message) for item in caught)),
            }
            print(
                f"noise={noise_ratio:g} {METHOD_LABELS[method_name]}: "
                f"squared error={squared_error:.9e}, median runtime={runtime:.6f} s, "
                f"runs={[round(value, 6) for value in durations]}",
                flush=True,
            )
    return results


def _debiased_test_geometry(args):
    """Translate the existing 3D grid's test centers to physical coordinates."""
    sizes = (args.nx, args.ny, args.nz, args.nt)
    bounds = np.array(((-5.0, 5.0),) * 3 + ((0.5, 2.5),))
    widths_in_cells = tuple(args.half_widths) if args.half_widths else tuple(
        max(2, size // 4) for size in sizes
    )
    strides = tuple(args.strides) if args.strides else tuple(
        max(1, width // 4) for width in widths_in_cells
    )
    degrees = tuple(args.test_degrees) if args.test_degrees else tuple(
        max(order + 1, int(np.ceil(np.log(1e-10) / np.log((2 * width - 1) / width**2))))
        for width, order in zip(widths_in_cells, (5, 5, 5, 1))
    )
    grids = [np.linspace(lower, upper, size) for (lower, upper), size in zip(bounds, sizes)]
    if any(2 * width + 1 > size for width, size in zip(widths_in_cells, sizes)):
        raise ValueError("Each test-function support must fit inside its observation axis.")
    center_axes = [
        grid[np.arange(width, size - width, stride)]
        for grid, size, width, stride in zip(grids, sizes, widths_in_cells, strides)
    ]
    centers = np.column_stack([
        axis.ravel() for axis in np.meshgrid(*center_axes, indexing="ij")
    ])
    widths = tuple(
        width * (grid[1] - grid[0]) for width, grid in zip(widths_in_cells, grids)
    )
    return bounds, centers, widths, degrees, widths_in_cells, strides


def run_debiased_experiments(args):
    """Measure the new method on independent samples from the same 3D PDE."""
    bounds, centers, widths, degrees, cell_widths, strides = _debiased_test_geometry(args)
    n_observations = args.n_observations or args.nx * args.ny * args.nz * args.nt
    pilot_bandwidths = tuple(args.pilot_bandwidths) if args.pilot_bandwidths else (0.8, 0.8, 0.8, 0.35)
    metadata = {
        "bounds": bounds, "centers": centers, "widths": widths,
        "degrees": degrees, "cell_widths": cell_widths, "strides": strides,
        "n_observations": n_observations, "pilot_bandwidths": pilot_bandwidths,
    }
    results = {}
    for noise_ratio in args.noise_ratios:
        data = generate_anisotropic_porous_medium_3d_samples(
            n_observations=n_observations, bounds=bounds,
            noise_reference_shape=(args.nx, args.ny, args.nz, args.nt),
            noise_ratio=noise_ratio, seed=args.seed,
        )
        terms = polynomial_library_terms(3)
        beta_true = np.array([data.true_coefficients.get(term, 0.0) for term in terms])
        inputs = (
            data.training_points, data.training_values,
            data.evaluation_points, data.evaluation_values,
        )
        settings = {
            "domain_bounds": bounds, "test_centers": centers,
            "test_half_widths": widths, "test_degrees": degrees,
            "lambda_": args.debiased_lambda, "max_iter": args.max_iter, "tol": args.tol,
            "regression": args.debiased_regression, "thresholds": args.thresholds,
        }
        durations = []
        print(f"noise={noise_ratio:g} debiased WSINDy: starting", flush=True)
        try:
            for call_index in range(args.repeats + 1):
                estimator = MovingAverageEstimator(pilot_bandwidths)
                start = perf_counter()
                beta = call_estimator(
                    debiased_wsindy, inputs + (estimator,), settings, args.timeout_seconds
                )
                elapsed = perf_counter() - start
                if call_index:
                    durations.append(elapsed)
                print(f"  {'warm-up' if call_index == 0 else f'run {call_index}'}: {elapsed:.3f} s", flush=True)
        except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
            results[noise_ratio] = {
                "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                "squared_error": None, "runtime_seconds": None,
                "reason": str(error), "failed_attempt_seconds": perf_counter() - start,
                "durations": durations, "noise_std": data.noise_std,
            }
            print(f"  {results[noise_ratio]['status']}: {error}", flush=True)
            continue
        if beta.shape != beta_true.shape or not np.all(np.isfinite(beta)):
            raise ValueError("Debiased estimator returned an invalid coefficient vector.")
        results[noise_ratio] = {
            "status": "ok", "squared_error": float(np.sum((beta - beta_true)**2)),
            "runtime_seconds": float(np.median(durations)), "durations": durations,
            "noise_std": data.noise_std,
            "nonzero_coefficients": int(np.count_nonzero(beta)),
            "beta": beta.tolist(),
        }
        print(
            f"noise={noise_ratio:g} debiased WSINDy: "
            f"squared error={results[noise_ratio]['squared_error']:.9e}, "
            f"median runtime={results[noise_ratio]['runtime_seconds']:.6f} s",
            flush=True,
        )
    return results, metadata


def run_sampled_experiments(args):
    """Fit ordinary WSINDy on the debiased method's evaluation sample."""
    bounds, centers, widths, degrees, cell_widths, strides = _debiased_test_geometry(args)
    n_observations = args.n_observations or args.nx * args.ny * args.nz * args.nt
    metadata = {
        "n_observations": n_observations, "centers": centers,
        "cell_widths": cell_widths, "strides": strides, "degrees": degrees,
    }
    results = {}
    for noise_ratio in args.noise_ratios:
        data = generate_anisotropic_porous_medium_3d_samples(
            n_observations=n_observations, bounds=bounds,
            noise_reference_shape=(args.nx, args.ny, args.nz, args.nt),
            noise_ratio=noise_ratio, seed=args.seed,
        )
        terms = polynomial_library_terms(3)
        beta_true = np.array([data.true_coefficients.get(term, 0.0) for term in terms])
        durations = {method: [] for method in args.methods}
        estimates = {}
        failures = {}
        for call_index in range(args.repeats + 1):
            start = perf_counter()
            try:
                x, y = call_estimator(
                    build_sampled_wsindy_system,
                    (data.evaluation_points, data.evaluation_values),
                    {
                        "domain_bounds": bounds, "test_centers": centers,
                        "test_half_widths": widths, "test_degrees": degrees,
                    },
                    args.timeout_seconds,
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
                for method in args.methods:
                    failures[method] = error
                break
            construction_seconds = perf_counter() - start
            print(
                f"noise={noise_ratio:g} weak-system "
                f"{'warm-up' if call_index == 0 else f'run {call_index}'}: "
                f"{construction_seconds:.3f} s", flush=True,
            )
            for method in args.methods:
                if method in failures:
                    continue
                regression = "mstls" if method.endswith("mstls") else "lasso"
                penalty = args.sampled_lambda if method.endswith("lasso") else 0.0
                start = perf_counter()
                try:
                    beta = call_estimator(
                        fit_sampled_wsindy_system, (x, y),
                        {
                            "lambda_": penalty, "regression": regression,
                            "thresholds": args.thresholds if regression == "mstls" else None,
                            "max_iter": args.max_iter, "tol": args.tol,
                        },
                        args.timeout_seconds,
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
                    failures[method] = error
                    print(f"noise={noise_ratio:g} {METHOD_LABELS[method]}: failed: {error}", flush=True)
                    continue
                total_seconds = construction_seconds + perf_counter() - start
                estimates[method] = beta
                if call_index:
                    durations[method].append(total_seconds)
                print(
                    f"  {METHOD_LABELS[method]}: {total_seconds:.3f} s "
                    "including construction", flush=True,
                )
        for method in args.methods:
            if method in failures:
                error = failures[method]
                results[(noise_ratio, method)] = {
                    "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                    "reason": str(error), "runtime_seconds": None, "squared_error": None,
                }
                continue
            beta = estimates[method]
            penalty = args.sampled_lambda if method.endswith("lasso") else 0.0
            results[(noise_ratio, method)] = {
                "status": "ok", "runtime_seconds": float(np.median(durations[method])),
                "squared_error": float(np.sum((beta - beta_true)**2)),
                "nonzero_coefficients": int(np.count_nonzero(beta)),
                "penalty": penalty,
            }
            print(
                f"noise={noise_ratio:g} {METHOD_LABELS[method]}: "
                f"squared error={results[(noise_ratio, method)]['squared_error']:.9e}, "
                f"median runtime={results[(noise_ratio, method)]['runtime_seconds']:.6f} s",
                flush=True,
            )
    return results, metadata


def write_sampled_report(args, results, metadata):
    """Write ordinary WSINDy results from the matched iid evaluation sample."""
    n_evaluation = metadata["n_observations"] - metadata["n_observations"] // 2
    lines = [
        "# 3D ordinary WSINDy on iid point samples", "",
        f"- Seed={args.seed}; total n={metadata['n_observations']}; "
        f"the final {n_evaluation} evaluation observations are used for each fit.",
        f"- K={len(metadata['centers'])}; half-widths in cells={metadata['cell_widths']}; "
        f"strides={metadata['strides']}; bump degrees={metadata['degrees']}.",
        "- Same evaluation sample, test functions, and Monte Carlo integration "
        "as debiased WSINDy; only the latter fits a pilot on the training half.",
        f"- Runtime: median of {args.repeats} construction-plus-fit sums after one warm-up. "
        "Construction is shared across methods within each repeat; data generation excluded.",
        "", "| Noise | Method | lambda | Squared error | Runtime (s) | Nonzero | Status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for (noise_ratio, method), result in results.items():
        if result["status"] == "ok":
            lines.append(
                f"| {noise_ratio:g} | {METHOD_LABELS[method]} | "
                f"{result['penalty']:g} | {result['squared_error']:.9e} | "
                f"{result['runtime_seconds']:.6f} | "
                f"{result['nonzero_coefficients']} | ok |"
            )
        else:
            lines.append(
                f"| {noise_ratio:g} | {METHOD_LABELS[method]} | — | — | — | — | "
                f"{result['status']} |"
            )
            lines.append(f"\n{result['reason']}\n")
    with args.output.open("a" if args.append else "w", encoding="utf-8") as report:
        if args.append and args.output.stat().st_size:
            report.write("\n---\n\n")
        report.write("\n".join(lines) + "\n")


def write_debiased_report(args, results, metadata):
    """Write the random-design comparison without relabeling old grid results."""
    if args.debiased_regression == "mstls":
        threshold_description = (
            "50 log-spaced thresholds from 1e-4 to 1"
            if args.thresholds is None else f"thresholds={args.thresholds}"
        )
        regression_description = (
            "- Regression: MSTLS on the corrected weak system; "
            f"{threshold_description}. This is a threshold-and-refit rule, "
            "not L1 minimization."
        )
    else:
        regression_description = (
            f"- Regression: {'OLS' if args.debiased_lambda == 0 else 'LASSO'}; "
            f"lambda={args.debiased_lambda:g}; objective is "
            "||Y_db-X_db beta||^2/2 + lambda*||beta||_1."
        )
    lines = [
        "# 3D porous medium: debiased WSINDy on iid point samples", "",
        "- Same exact 3D PDE, domain, seed, S=55, J=5, and polynomial coefficient order as the grid experiment.",
        f"- n={metadata['n_observations']} total observations; "
        f"{metadata['n_observations']//2} train and "
        f"{metadata['n_observations']-metadata['n_observations']//2} evaluation observations.",
        "- Observation coordinates are iid uniform on [-5,5]^3 x [0.5,2.5]; "
        "the older results use a fixed grid, so their errors and runtimes are not controlled comparisons.",
        f"- K={len(metadata['centers'])} translated bump tests; cell half-widths={metadata['cell_widths']}, "
        f"strides={metadata['strides']}, degrees={metadata['degrees']}.",
        "- Test reference: b_p(r)=(1-r^2)^p for |r|<1, zero outside; "
        "the 4D test is a product of these translated bumps.",
        "- Physical test half-widths=("
        + ", ".join(f"{width:.6f}" for width in metadata["widths"])
        + f"); pilot moving-average bandwidths={metadata['pilot_bandwidths']}.",
        regression_description,
        "- Monte Carlo weight is domain volume / evaluation sample size. "
        "The same evaluation observations enter Y and the first-order correction to X.",
        f"- Seed={args.seed}; noise std = noise ratio * RMS(clean solution on the fixed "
        f"{(args.nx, args.ny, args.nz, args.nt)} reference grid).",
        f"- Runtime: median of {args.repeats} complete calls after one warm-up, "
        "including pilot fitting, weak-system construction, and regression; data generation excluded.",
        "",
        "| Noise ratio | Squared coefficient error | Runtime (seconds) | Nonzero coefficients | Noise std | Status |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for noise_ratio, result in results.items():
        error = f"{result['squared_error']:.9e}" if result['squared_error'] is not None else "N/A"
        runtime = f"{result['runtime_seconds']:.6f}" if result['runtime_seconds'] is not None else "N/A"
        nonzero = str(result.get("nonzero_coefficients", "N/A"))
        lines.append(
            f"| {noise_ratio:g} | {error} | {runtime} | {nonzero} | "
            f"{result['noise_std']:.9e} | {result['status']} |"
        )
        if result['status'] != "ok":
            lines.append(f"\nNoise {noise_ratio:g} failure: {result['reason']}\n")
    lines.extend(["", "## Reproduce", "", "```sh",
        "OPENBLAS_NUM_THREADS=1 conda run -n pde_discovery python experiments.py \\",
        "  --instance anisotropic_porous_medium_3d --methods debiased-wsindy \\",
        f"  --nx {args.nx} --ny {args.ny} --nz {args.nz} --nt {args.nt} \\",
        f"  --n-observations {metadata['n_observations']} --seed {args.seed} --repeats {args.repeats} \\",
        "  --noise-ratios " + " ".join(f"{ratio:g}" for ratio in args.noise_ratios) + " \\",
        "  --half-widths " + " ".join(map(str, metadata["cell_widths"])) + " \\",
        "  --strides " + " ".join(map(str, metadata["strides"])) + " \\",
        "  --test-degrees " + " ".join(map(str, metadata["degrees"])) + " \\",
        "  --pilot-bandwidths " + " ".join(map(str, metadata["pilot_bandwidths"])) + " \\",
        f"  --debiased-regression {args.debiased_regression} "
        f"--debiased-lambda {args.debiased_lambda:g} "
        f"--max-iter {args.max_iter} --tol {args.tol:g} \\",
    ])
    if args.thresholds is not None:
        lines.append("  --thresholds " + " ".join(map(str, args.thresholds)) + " \\")
    lines.extend(["  --output /tmp/pde-debiased-results.md", "```", ""])
    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as report:
        if args.append and args.output.stat().st_size:
            report.write("\n---\n\n")
        report.write("\n".join(lines))


def write_report(args, results):
    """Write one table per metric and noise level, with methods as columns."""
    spatial_sizes = (args.nx, args.ny)
    spatial_axes = ("x", "y")
    pde = "u_t = 0.3 d_xx(u^2) - 0.8 d_xy(u^2) + d_yy(u^2)"
    truth_squared_norm = 1.73
    if args.instance == "anisotropic_porous_medium_3d":
        spatial_sizes += (args.nz,)
        spatial_axes += ("z",)
        pde = "u_t = 0.3 d_xx(u^2) - 0.2 d_xy(u^2) + 0.1 d_xz(u^2) + 0.7 d_yy(u^2) - 0.16 d_yz(u^2) + d_zz(u^2)"
        truth_squared_norm = 1.6556
    spatial_dim = len(spatial_sizes)
    axes = spatial_axes + ("t",)
    shape = spatial_sizes + (args.nt,)
    axis_labels = "(" + ", ".join(axes) + ")"
    instance_label = f"{spatial_dim}D anisotropic porous medium"
    widths = tuple(args.half_widths) if args.half_widths else tuple(max(2, size // 4) for size in shape)
    strides = tuple(args.strides) if args.strides else tuple(max(1, width // 4) for width in widths)
    degrees = tuple(args.test_degrees) if args.test_degrees else tuple(
        max(order + 1, int(np.ceil(np.log(1e-10) / np.log((2.0 * width - 1) / width**2))))
        for width, order in zip(widths, (5,) * spatial_dim + (1,))
    )
    terms = polynomial_library_terms(spatial_dim=spatial_dim)
    spatial_derivatives = {derivative for derivative, power in terms}
    max_derivative_order = max(sum(derivative) for derivative in spatial_derivatives)
    polynomial_degree = max(power for derivative, power in terms)
    radius = (max_derivative_order + 1) // 2
    sindy_shape = tuple(size - 2 * radius for size in spatial_sizes) + (args.nt - 2,)
    centers_per_axis = tuple(
        len(range(width, size - width, stride))
        for size, width, stride in zip(shape, widths, strides)
    )
    center_ranges = ", ".join(
        f"{axis}: range({width}, {size - width}, {stride})"
        for axis, size, width, stride in zip(axes, shape, widths, strides)
    )
    spacings = tuple(10.0 / (size - 1) for size in spatial_sizes) + (2.0 / (args.nt - 1),)
    physical_widths = tuple(width * spacing for width, spacing in zip(widths, spacings))
    n = int(np.prod(shape))
    K = int(np.prod(centers_per_axis))
    thread_settings = ", ".join(
        f"{name}={os.environ.get(name, 'unset')}"
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    )
    thread_command = " ".join(
        f"{name}={shlex.quote(os.environ[name])}"
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        if name in os.environ
    )
    lines = [
        f"# {instance_label}: PDE-discovery sanity check",
        "",
        "## Experiment settings",
        "",
        f"- Instance: `{args.instance}`; exact {spatial_dim}D anisotropic porous-medium weak solution.",
        f"- PDE: `{pde}`.",
        f"- Grid: `{shape}` on `[-5, 5]^{spatial_dim}`, with time in `[0.5, 2.5]`; endpoints included.",
        f"- **n = {n:,}** observed space-time points: `{' * '.join(map(str, shape))}`;",
        "  this counts the full input grid for every method, including spatial/time endpoints.",
        f"- **K = {K:,}** test functions / weak equations: `{' * '.join(map(str, centers_per_axis))}`",
        f"  centers along `{axis_labels}`. All selected weak methods use the same",
        "  tests at every noise ratio. SINDy uses no test functions, so K does not apply to it.",
        f"- **S = {len(spatial_derivatives)}** spatial derivative operators, using the draft's indexing:",
        f"  every {spatial_dim}-component multi-index `alpha` with nonnegative entries and `1 <= sum(alpha) <= {max_derivative_order}`.",
        f"  The maximum total spatial derivative order is **{max_derivative_order}**; S counts operators.",
        f"- **J = {polynomial_degree}** polynomial powers: `u, u^2, ..., u^{polynomial_degree}`.",
        "  J describes powers of u, separately from the test-function exponents below.",
        f"- Library: `D^alpha(u^j)`, giving `S * J = {len(terms)}` coefficients, in",
        f"  `polynomial_library_terms({spatial_dim})` order. No zeroth spatial derivative or constant power is included.",
        f"- SINDy retains `{' * '.join(map(str, sindy_shape))} = {int(np.prod(sindy_shape)):,}` regression rows",
        f"  after removing {radius} points at each spatial end and 1 at each time end.",
        f"  Its design matrix is `{int(np.prod(sindy_shape)):,} x {len(terms)}`; weak design matrices are `{K:,} x {len(terms)}`.",
        f"- One Gaussian-noise realization per noise ratio, seed `{args.seed}`;",
        "  `noise_std = noise_ratio * RMS(u_true)`. Each method receives the same observations.",
        f"- Error: `sum((beta_hat - beta_true)**2)` over all {len(terms)} coefficients, with no",
        f"  relative normalization. Truth is used only for evaluation; its squared norm is {truth_squared_norm:g}.",
        f"- LASSO: SINDy `rho_1={args.sindy_rho_1:g}`; WSINDy `rho_1={args.wsindy_rho_1:g}`;",
        f"  `max_iter={args.max_iter}`, `tol={args.tol:g}`. These are fixed example penalties, not tuned values.",
        "- OLS: `rho_1=0`. WSINDy MSTLS candidates: "
        + (f"`{tuple(args.thresholds)}`." if args.thresholds else "`np.logspace(-4, 0, 50)`."),
        f"- Runtime: median of {args.repeats} timed calls after one untimed warm-up per method",
        "  and noise level. Each call includes library/weak-system construction and regression;",
        "  data generation, imports, error calculation, and report writing are excluded.",
        f"- Environment: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__},",
        f"  {platform.platform()}; `{thread_settings}`.",
        "",
    ]
    if spatial_dim == 3:
        lines.extend([
            "## Exact reference solution",
            "",
            "```text",
            "D = [[0.3, -0.1, 0.05], [-0.1, 0.7, -0.08], [0.05, -0.08, 1.0]]",
            "q = (x,y,z) D^{-1} (x,y,z)^T",
            "C = (15 / (8*pi * 20^(3/2) * sqrt(det(D))))^(2/5)",
            "u_true(x,y,z,t) = t^(-3/5) * max(C - q/(20*t^(2/5)), 0)",
            "```",
            "",
            "D is symmetric positive definite. This exact Barenblatt weak solution has unit mass",
            "on R^3; the default observation domain contains its support throughout the time interval.",
            "The initial condition is the profile at t=0.5, and the spatial boundary stays zero.",
            "No numerical time stepping is used. Mixed PDE coefficients are twice the off-diagonal entries of D.",
            "",
        ])
    if args.timeout_seconds is not None:
        lines.extend([f"- Wall-time limit: {args.timeout_seconds:g} seconds per estimator call, including warm-up.", ""])
    if any(name.startswith("wendy") for name in args.methods):
        lines.extend([
            f"- WENDy: LASSO `rho_1={args.wendy_rho_1:g}`, OLS `rho_1=0`; `alpha={args.wendy_alpha:g}`,",
            f"  `max_reweights={args.max_reweights}`, `reweight_tol={args.reweight_tol:g}`,",
            f"  normality stopping {'disabled' if args.disable_normality_stop else 'enabled (p<1e-4 after 10 reweights)' }.",
            f"- WENDy-MLE: LASSO `rho_1={args.wendy_mle_rho_1:g}`, unpenalized `rho_1=0`; `alpha={args.wendy_mle_alpha:g}`;",
            f"  `max_iter={args.mle_max_iter}`, `tol={args.mle_tol:g}` (scaled KKT tolerance).",
            "  noise std: " + (f"explicit working value `{args.mle_noise_std:g}`."
                                 if args.mle_noise_std is not None else "known generator noise std for each noise level."),
            "- WENDy (OLS) fits unpenalized least squares on each whitened system (GLS/IRLS).",
            "  WENDy-MLE (unpenalized) minimizes the full Gaussian weak-residual likelihood,",
            "  including the log determinant, with no L1 penalty. Neither uses thresholding.",
            "  Both nonsparse fits start from full-library WSINDy (OLS).",
            "",
        ])
    lines.extend([
        "## Test functions",
        "",
        "Every weak method starts from the same reference function, a tensor product",
        "of compact polynomial bumps (the WSINDy family):",
        "",
        "```text",
        "b_p(r) = (1-r^2)^p for |r| < 1, and 0 otherwise.",
        "phi_ref(" + ",".join(f"r_{axis}" for axis in axes) + ") = "
        + " * ".join(f"b_{degree}(r_{axis})" for axis, degree in zip(axes, degrees)) + ".",
        "```",
        "",
        f"The reference function is centered at the origin, supported on `[-1,1]^{spatial_dim + 1}`,",
        "and satisfies `phi_ref(0)=1`. Construct each physical test function by",
        "scaling its coordinates and translating its center:",
        "",
        "```text",
        ", ".join(f"Delta_{axis} = 10/(n{axis}-1)" for axis in spatial_axes) + ", Delta_t = 2/(nt-1).",
        "h_axis = m_axis * grid_spacing_axis.",
        "c_k = (" + ", ".join(f"-5 + i_{axis}*Delta_{axis}" for axis in spatial_axes) + ", 0.5 + i_t*Delta_t).",
        "phi_k" + axis_labels + " = phi_ref("
        + ", ".join(f"({axis}-c_k{axis})/h_{axis}" for axis in axes) + ").",
        "```",
        "",
        "Take every Cartesian-product combination of the center indices `("
        + ",".join(f"i_{axis}" for axis in axes) + ")`",
        f"listed below, with time varying fastest. This gives `{' * '.join(map(str, centers_per_axis))} = {K}` functions.",
        "The reference function and support half-widths are fixed; only the center changes.",
        "Each translated function is supported on the product of intervals",
        "`[c_k_axis-h_axis, c_k_axis+h_axis]`, with peak value one.",
        "There is no volume factor `1/prod(h_axis)` or L2 normalization.",
        "",
        f"- Bump exponents in `{axis_labels}` order: `{degrees}` (one-dimensional polynomial degrees `{tuple(2 * p for p in degrees)}`).",
        f"- Support half-widths in grid cells, in `{axis_labels}` order: `{widths}`;",
        f"  each support spans `{tuple(2 * width + 1 for width in widths)}` grid points including endpoints.",
        f"  Physical half-widths in the same order are approximately `({', '.join(f'{value:.6f}' for value in physical_widths)})`.",
        f"- Center strides in grid cells: `{strides}`. Zero-based center indices are",
        f"  `{center_ranges}` (range stops excluded), giving `{centers_per_axis}` centers and `K={K}`.",
        "- Default support rule: `m_axis=max(2, axis_size//4)`; default stride: `max(1, m_axis//4)`.",
        "  These are fixed grid-size rules; the paper's Fourier-based support selection is not used.",
        "- Default exponent rule: the smallest integer p greater than the derivative order on that axis",
        f"  (`{max_derivative_order}` in space, `1` in time) with `(1-(1-1/m)^2)^p <= 1e-10`.",
        "  These polynomial bumps have finite smoothness, sufficient for the derivatives used here.",
        "- Derivatives act analytically on the test functions. Integrals use tensor-product trapezoidal",
        "  quadrature with the physical grid spacings; there is no normalization of individual equations.",
        "  Only supports fully inside the observed grid are used, with no padding or periodic wrapping.",
        "",
    ])
    for metric, title in (
        ("squared_error", "Squared coefficient error"),
        ("runtime_seconds", "Runtime (seconds)"),
    ):
        for noise_ratio in args.noise_ratios:
            lines.extend([
                f"## Noise ratio {noise_ratio:g}: {title}",
                "",
                "| Experiment instance | " + " | ".join(METHOD_LABELS[m] for m in args.methods) + " |",
                "| --- | " + " | ".join("---:" for _ in args.methods) + " |",
            ])
            cells = []
            for method in args.methods:
                result = results[noise_ratio][method]
                status = result["status"]
                if status in ("ok", "warning"):
                    value = result[metric]
                    cell = f"{value:.6e}" if metric == "squared_error" else f"{value:.6f}"
                    if status == "warning":
                        cell += " *"
                elif status == "not_applicable":
                    cell = "N/A"
                else:
                    cell = "TIMEOUT" if status == "timeout" else "FAIL"
                    if metric == "runtime_seconds":
                        cell += f" ({result['failed_attempt_seconds']:.3f} s)"
                cells.append(cell)
            lines.extend([f"| {instance_label} | " + " | ".join(cells) + " |", ""])

    lines.extend([
        "## Fit status",
        "",
        "`*` marks a returned estimate with a warning; see the stop reason below.",
        "`FAIL` and `TIMEOUT` mark incomplete benchmarks. Their parenthesized",
        "times measure the failed call, not a median successful-fit runtime.",
        "`N/A` means the sigma=0 MLE objective is undefined; no fit was attempted.",
        "",
    ])
    for noise_ratio in args.noise_ratios:
        for method in args.methods:
            result = results[noise_ratio][method]
            label = f"Noise ratio {noise_ratio:g}, {METHOD_LABELS[method]}"
            if "reason" in result:
                phase = f" ({result['failed_call']})" if "failed_call" in result else ""
                lines.append(f"- {label}{phase}: {result['reason']}")
            for warning in result.get("warnings", []):
                lines.append(f"- {label}: {warning}")
    lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "This is one fixed-grid sanity check with one seed, not a Monte Carlo comparison.",
        f"The zero coefficient vector has squared error {truth_squared_norm:g}, so errors above {truth_squared_norm:g} are",
        "worse than that reference on this coefficient metric. The solution has a",
        "nonsmooth moving front and the full library is highly correlated; solving the",
        "regression accurately does not by itself ensure accurate coefficient recovery.",
        "LASSO penalties act on differently scaled losses, so the chosen",
        "penalties do not represent equal regularization strength across methods.",
        "Runtime covers this implementation, including all configured MSTLS candidates.",
        "Repeated timings reuse the same data; they are not independent noise trials.",
        "A timeout describes the implementation under the stated compute budget; it",
        "does not establish nonconvergence or an accuracy ranking for that method.",
        "",
        "## Reproduce",
        "",
        "```sh",
        (thread_command + " \\") if thread_command else "# BLAS thread environment variables were unset.",
        f"python experiments.py --instance {args.instance} \\",
        "  " + " ".join(f"--n{axis} {size}" for axis, size in zip(spatial_axes, spatial_sizes))
        + f" --nt {args.nt} --seed {args.seed} --repeats {args.repeats} \\",
        "  --noise-ratios " + " ".join(f"{value:g}" for value in args.noise_ratios) + " \\",
        "  --methods " + " ".join(args.methods) + " \\",
        f"  --sindy-rho-1 {args.sindy_rho_1:g} --wsindy-rho-1 {args.wsindy_rho_1:g} \\",
        f"  --max-iter {args.max_iter} --tol {args.tol:g} --output {shlex.quote(str(args.output))}",
    ])
    extra_options = []
    if args.append:
        extra_options.append("--append")
    if args.timeout_seconds is not None:
        extra_options.append(f"--timeout-seconds {args.timeout_seconds:g}")
    for flag, values in (("half-widths", args.half_widths), ("strides", args.strides),
                         ("test-degrees", args.test_degrees), ("thresholds", args.thresholds)):
        if values is not None:
            extra_options.append(f"--{flag} " + " ".join(str(value) for value in values))
    if any(name.startswith("wendy") for name in args.methods):
        extra_options.extend([
            f"--wendy-rho-1 {args.wendy_rho_1:g} --wendy-mle-rho-1 {args.wendy_mle_rho_1:g}",
            f"--wendy-alpha {args.wendy_alpha:g} --wendy-mle-alpha {args.wendy_mle_alpha:g}",
            f"--max-reweights {args.max_reweights} --reweight-tol {args.reweight_tol:g}",
            f"--mle-max-iter {args.mle_max_iter} --mle-tol {args.mle_tol:g}",
        ])
        if args.mle_noise_std is not None:
            extra_options.append(f"--mle-noise-std {args.mle_noise_std:g}")
        if args.disable_normality_stop:
            extra_options.append("--disable-normality-stop")
    for option in extra_options:
        lines[-1] += " \\"
        lines.append("  " + option)
    lines.extend(["```", ""])
    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as report:
        if args.append and args.output.stat().st_size:
            report.write("\n---\n\n")
        report.write("\n".join(lines))


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.instance == "nonlinear_viscous_burgers":
        measurements, settings = run_burgers_experiments(arguments)
        write_burgers_report(arguments, measurements, settings)
    elif arguments.methods == ["debiased-wsindy"]:
        measurements, settings = run_debiased_experiments(arguments)
        write_debiased_report(arguments, measurements, settings)
    elif all(method in SAMPLED_METHODS for method in arguments.methods):
        measurements, settings = run_sampled_experiments(arguments)
        write_sampled_report(arguments, measurements, settings)
    else:
        measurements = run_experiments(arguments)
        write_report(arguments, measurements)
    print(f"Saved {arguments.output}")
