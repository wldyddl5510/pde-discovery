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

from methods import polynomial_library_terms, sindy, wsindy, wendy, wendy_mle
from simulation_generation import generate_anisotropic_porous_medium


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
}
DEFAULT_METHODS = ["sindy-ols", "wsindy-ols", "sindy-lasso", "wsindy-lasso", "wsindy-mstls"]
GENERATORS = {"anisotropic_porous_medium": generate_anisotropic_porous_medium}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=GENERATORS, default="anisotropic_porous_medium")
    parser.add_argument("--methods", nargs="+", choices=METHOD_LABELS, default=DEFAULT_METHODS)
    parser.add_argument("--noise-ratios", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sindy-rho-1", type=float, default=1e-4)
    parser.add_argument("--wsindy-rho-1", type=float, default=1e-6)
    parser.add_argument("--wendy-rho-1", type=float, default=1e-4)
    parser.add_argument("--wendy-mle-rho-1", type=float, default=1e-3)
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
    parser.add_argument("--half-widths", nargs=3, type=int, default=None)
    parser.add_argument("--strides", nargs=3, type=int, default=None)
    parser.add_argument("--test-degrees", nargs=3, type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=None,
                        help="Optional wall-time limit per estimator call (Unix).")
    parser.add_argument("--output", type=Path, default=Path("results.md"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
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
        data = GENERATORS[args.instance](
            nx=args.nx, ny=args.ny, nt=args.nt,
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


def write_report(args, results):
    """Write one table per metric and noise level, with methods as columns."""
    shape = (args.nx, args.ny, args.nt)
    widths = tuple(args.half_widths) if args.half_widths else tuple(max(2, size // 4) for size in shape)
    strides = tuple(args.strides) if args.strides else tuple(max(1, width // 4) for width in widths)
    degrees = tuple(args.test_degrees) if args.test_degrees else tuple(
        max(order + 1, int(np.ceil(np.log(1e-10) / np.log((2.0 * width - 1) / width**2))))
        for width, order in zip(widths, (5, 5, 1))
    )
    terms = polynomial_library_terms(spatial_dim=2)
    spatial_derivatives = {derivative for derivative, power in terms}
    max_derivative_order = max(sum(derivative) for derivative in spatial_derivatives)
    polynomial_degree = max(power for derivative, power in terms)
    radius = (max_derivative_order + 1) // 2
    sindy_shape = (args.nx - 2 * radius, args.ny - 2 * radius, args.nt - 2)
    centers_per_axis = tuple(
        len(range(width, size - width, stride))
        for size, width, stride in zip(shape, widths, strides)
    )
    center_ranges = ", ".join(
        f"{axis}: range({width}, {size - width}, {stride})"
        for axis, size, width, stride in zip(("x", "y", "t"), shape, widths, strides)
    )
    spacings = (10.0 / (args.nx - 1), 10.0 / (args.ny - 1), 2.0 / (args.nt - 1))
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
        "# PDE-discovery sanity check",
        "",
        "## Experiment settings",
        "",
        f"- Instance: `{args.instance}`; exact 2D anisotropic porous-medium weak solution.",
        "- PDE: `u_t = 0.3 d_xx(u^2) - 0.8 d_xy(u^2) + d_yy(u^2)`.",
        f"- Grid: `{shape}` on `[-5, 5]^2`, with time in `[0.5, 2.5]`; endpoints included.",
        f"- **n = {n:,}** observed space-time points: `{' * '.join(map(str, shape))}`;",
        "  this counts the full input grid for every method, including spatial/time endpoints.",
        f"- **K = {K:,}** test functions / weak equations: `{' * '.join(map(str, centers_per_axis))}`",
        "  centers along `(x, y, t)`. All WSINDy, WENDy, and WENDy-MLE variants use the same",
        "  tests at every noise ratio. SINDy uses no test functions, so K does not apply to it.",
        f"- **S = {len(spatial_derivatives)}** spatial derivative operators, using the draft's indexing:",
        f"  every multi-index `alpha=(a,b)` with `a,b >= 0` and `1 <= a+b <= {max_derivative_order}`.",
        f"  The maximum total spatial derivative order is **{max_derivative_order}**; S counts operators.",
        f"- **J = {polynomial_degree}** polynomial powers: `u, u^2, ..., u^{polynomial_degree}`.",
        "  J describes powers of u, separately from the test-function exponents below.",
        f"- Library: `D^alpha(u^j)`, giving `S * J = {len(terms)}` coefficients, in",
        "  `polynomial_library_terms(2)` order. No zeroth spatial derivative or constant power is included.",
        f"- SINDy retains `{' * '.join(map(str, sindy_shape))} = {int(np.prod(sindy_shape)):,}` regression rows",
        f"  after removing {radius} points at each spatial end and 1 at each time end.",
        f"  Its design matrix is `{int(np.prod(sindy_shape)):,} x {len(terms)}`; weak design matrices are `{K:,} x {len(terms)}`.",
        f"- One Gaussian-noise realization per noise ratio, seed `{args.seed}`;",
        "  `noise_std = noise_ratio * RMS(u_true)`. Each method receives the same observations.",
        "- Error: `sum((beta_hat - beta_true)**2)` over all 100 coefficients, with no",
        "  relative normalization. Truth is used only for evaluation; its squared norm is 1.73.",
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
        f"phi_ref(r_x,r_y,r_t) = b_{degrees[0]}(r_x) * b_{degrees[1]}(r_y) * b_{degrees[2]}(r_t).",
        "```",
        "",
        "The reference function is centered at `(0,0,0)`, supported on `[-1,1]^3`,",
        "and satisfies `phi_ref(0,0,0)=1`. Construct each physical test function by",
        "scaling its coordinates and translating its center:",
        "",
        "```text",
        "Delta_x = 10/(nx-1), Delta_y = 10/(ny-1), Delta_t = 2/(nt-1).",
        "h_axis = m_axis * grid_spacing_axis.",
        "c_k = (-5 + i_x*Delta_x, -5 + i_y*Delta_y, 0.5 + i_t*Delta_t).",
        "phi_k(x,y,t) = phi_ref((x-c_kx)/h_x, (y-c_ky)/h_y, (t-c_kt)/h_t).",
        "```",
        "",
        "Take every Cartesian-product combination of the center indices `(i_x,i_y,i_t)`",
        f"listed below, with time varying fastest. This gives `{' * '.join(map(str, centers_per_axis))} = {K}` functions.",
        "The reference function and support half-widths are fixed; only the center changes.",
        "Each translated function has support `[c_kx-h_x,c_kx+h_x]` times",
        "`[c_ky-h_y,c_ky+h_y]` times `[c_kt-h_t,c_kt+h_t]`, with peak value one.",
        "There is no volume factor `1/(h_x*h_y*h_t)` or L2 normalization.",
        "",
        f"- Bump exponents `(p_x, p_y, p_t) = {degrees}` (one-dimensional polynomial degrees `{tuple(2 * p for p in degrees)}`).",
        f"- Support half-widths in grid cells: `(m_x, m_y, m_t) = {widths}`;",
        f"  each support spans `{tuple(2 * width + 1 for width in widths)}` grid points including endpoints.",
        f"  Physical half-widths `(h_x, h_y, h_t)` are approximately `({', '.join(f'{value:.6f}' for value in physical_widths)})`.",
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
            lines.extend(["| 2D anisotropic porous medium | " + " | ".join(cells) + " |", ""])

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
        "The zero coefficient vector has squared error 1.73, so errors above 1.73 are",
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
        f"  --nx {args.nx} --ny {args.ny} --nt {args.nt} --seed {args.seed} --repeats {args.repeats} \\",
        "  --noise-ratios " + " ".join(f"{value:g}" for value in args.noise_ratios) + " \\",
        "  --methods " + " ".join(args.methods) + " \\",
        f"  --sindy-rho-1 {args.sindy_rho_1:g} --wsindy-rho-1 {args.wsindy_rho_1:g} \\",
        f"  --max-iter {args.max_iter} --tol {args.tol:g} --output {shlex.quote(str(args.output))}",
    ])
    extra_options = []
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
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    arguments = parse_arguments()
    measurements = run_experiments(arguments)
    write_report(arguments, measurements)
    print(f"Saved {arguments.output}")
