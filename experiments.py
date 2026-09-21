"""Run small PDE-discovery comparisons and write error/runtime tables."""

import argparse
import os
import platform
import shlex
from pathlib import Path
from time import perf_counter

import numpy as np

from methods import polynomial_library_terms, sindy, wsindy
from simulation_generation import generate_anisotropic_porous_medium


METHOD_LABELS = {
    "sindy-ols": "SINDy (OLS)",
    "wsindy-ols": "WSINDy (OLS)",
    "sindy-lasso": "SINDy (LASSO)",
    "wsindy-lasso": "WSINDy (LASSO)",
    "wsindy-mstls": "WSINDy (MSTLS)",
}
GENERATORS = {"anisotropic_porous_medium": generate_anisotropic_porous_medium}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=GENERATORS, default="anisotropic_porous_medium")
    parser.add_argument("--methods", nargs="+", choices=METHOD_LABELS, default=list(METHOD_LABELS))
    parser.add_argument("--noise-ratios", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sindy-rho-1", type=float, default=1e-4)
    parser.add_argument("--wsindy-rho-1", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results.md"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
    if args.max_iter < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--max-iter and --tol must be positive, with finite --tol.")
    for penalty in (args.sindy_rho_1, args.wsindy_rho_1):
        if not np.isfinite(penalty) or penalty <= 0:
            parser.error("LASSO penalties must be finite and strictly positive.")
    return args


def run_experiments(args):
    """Generate once per noise level, then time complete estimator calls."""
    methods = {
        "sindy-ols": (sindy, {"regression": "lasso", "rho_1": 0.0}),
        "wsindy-ols": (wsindy, {"regression": "lasso", "rho_1": 0.0}),
        "sindy-lasso": (sindy, {"regression": "lasso", "rho_1": args.sindy_rho_1}),
        "wsindy-lasso": (wsindy, {"regression": "lasso", "rho_1": args.wsindy_rho_1}),
        "wsindy-mstls": (wsindy, {"regression": "mstls"}),
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
            inputs = (data.u_observed, data.spatial_grid, data.time)
            estimator(*inputs, **settings)  # One untimed warm-up for this configuration.
            durations = []
            for _ in range(args.repeats):
                start = perf_counter()
                beta = estimator(*inputs, **settings)
                durations.append(perf_counter() - start)

            if beta.shape != beta_true.shape or not np.all(np.isfinite(beta)):
                raise ValueError(f"{method_name} returned an invalid coefficient vector.")
            # Compare all coefficients, including the true zeros, in original units.
            squared_error = float(np.sum((beta - beta_true)**2))
            runtime = float(np.median(durations))
            results[noise_ratio][method_name] = {
                "squared_error": squared_error,
                "runtime_seconds": runtime,
                "durations": durations,
                "nonzero_coefficients": int(np.count_nonzero(beta)),
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
    widths = tuple(max(2, size // 4) for size in shape)
    strides = tuple(max(1, width // 4) for width in widths)
    degrees = tuple(
        max(order + 1, int(np.ceil(np.log(1e-10) / np.log((2.0 * width - 1) / width**2))))
        for width, order in zip(widths, (5, 5, 1))
    )
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
        "- Library: all mixed spatial derivatives of total order 1 through 5, applied to",
        "  powers 1 through 5: 100 coefficients, in `polynomial_library_terms(2)` order.",
        f"- One Gaussian-noise realization per noise ratio, seed `{args.seed}`;",
        "  `noise_std = noise_ratio * RMS(u_true)`. Each method receives the same observations.",
        "- Error: `sum((beta_hat - beta_true)**2)` over all 100 coefficients, with no",
        "  relative normalization. Truth is used only for evaluation; its squared norm is 1.73.",
        f"- LASSO: SINDy `rho_1={args.sindy_rho_1:g}`; WSINDy `rho_1={args.wsindy_rho_1:g}`;",
        f"  `max_iter={args.max_iter}`, `tol={args.tol:g}`. These are fixed example penalties, not tuned values.",
        "- OLS: `rho_1=0`. MSTLS: automatic selection from `np.logspace(-4, 0, 50)`.",
        f"- WSINDy: half-widths `{widths}`, strides `{strides}`, bump exponents `{degrees}`;",
        "  peak-one test functions and physical quadrature weights, with no row normalization.",
        f"- Runtime: median of {args.repeats} timed calls after one untimed warm-up per method",
        "  and noise level. Each call includes library/weak-system construction and regression;",
        "  data generation, imports, error calculation, and report writing are excluded.",
        f"- Environment: Python {platform.python_version()}, NumPy {np.__version__},",
        f"  {platform.platform()}; `{thread_settings}`.",
        "",
    ]
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
            values = [results[noise_ratio][m][metric] for m in args.methods]
            cells = [f"{value:.6e}" if metric == "squared_error" else f"{value:.6f}" for value in values]
            lines.extend(["| 2D anisotropic porous medium | " + " | ".join(cells) + " |", ""])

    lines.extend([
        "## Interpretation",
        "",
        "This is one fixed-grid sanity check with one seed, not a Monte Carlo comparison.",
        "The zero coefficient vector has squared error 1.73, so errors above 1.73 are",
        "worse than that reference on this coefficient metric. The solution has a",
        "nonsmooth moving front and the full library is highly correlated; solving the",
        "regression accurately does not by itself ensure accurate coefficient recovery.",
        "LASSO penalties act on differently scaled strong and weak losses, so the chosen",
        "penalties do not represent equal regularization strength across the two methods.",
        "Runtime covers this implementation, including all 50 MSTLS candidates.",
        "Repeated timings reuse the same data; they are not independent noise trials.",
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
        "```",
        "",
    ])
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    arguments = parse_arguments()
    measurements = run_experiments(arguments)
    write_report(arguments, measurements)
    print(f"Saved {arguments.output}")
