"""Synthetic PDE coefficient recovery. Run: python experiment.py"""
import argparse
import csv
import json
import hashlib
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import numpy as np
from scipy.special import softmax
from scipy.io import loadmat
from scipy.fft import set_workers
from scipy import __version__ as scipy_version

import wsindy
import wendy
import wendy_mle


def burgers(x, t):
    """WSINDy Appendix B.1 entropy solution, with x and u divided by 1000."""
    X, T = np.meshgrid(x, t)
    u = np.zeros_like(X)
    ramp = (T < 2) & (X > T-2) & (X <= 0)
    np.divide(-.5*X, 1-.5*T, out=u, where=ramp)
    u[T >= np.maximum(X+2, 2*X+2)] = 1
    return u


def kdv(x, t):
    """Exact two-soliton KdV solution u=12 D_x^2 log(tau), evaluated as a variance."""
    X, T = np.meshgrid(x, t)
    k1, k2 = 1.5, 1.
    eta1, eta2 = k1*(X+3)-k1**3*T, k2*X-k2**3*T
    eta = np.stack([np.zeros_like(X), eta1, eta2,
                    eta1+eta2+2*np.log((k1-k2)/(k1+k2))])
    probabilities = softmax(eta, axis=0)
    slopes = np.array([0, k1, k2, k1+k2])[:, None, None]
    mean = np.sum(probabilities*slopes, axis=0)
    return 12*np.sum(probabilities*(slopes-mean)**2, axis=0)


@dataclass(frozen=True)
class Problem:
    xlim: tuple
    tlim: tuple
    half_widths: tuple
    alpha: tuple
    target: dict
    solution: object

    def w_star(self, J=4):
        w = np.zeros((len(self.alpha)-1)*J)
        for (s, j), value in self.target.items():
            w[(s-1)*J+j] = value
        return w


# alpha^0=(0,1); remaining alpha^s index RHS differential operators.
# j=0,1,2,3 corresponds to the PDF's j=1,2,3,4, i.e. 1,u,u^2,u^3.
PROBLEMS = {
    "burgers": Problem( (-4., 4.), (0., 4.), (1.2, .65),
                       ((0, 1), (0, 0), (1, 0), (2, 0)), {(2, 2): -.5}, burgers),
    "kdv": Problem( (-10., 12.), (0., 6.), (3.2, 1.2),
                   ((0, 1), (0, 0), (1, 0), (2, 0), (3, 0)),
                   {(2, 2): -.5, (4, 1): -1.}, kdv),
}

PAPER_SOURCE = "https://raw.githubusercontent.com/dm973/WSINDy_PDE/95686ccd9e32e3a9f62acfb3014fb77d5ef039ab"


def paper_data(name):
    filename = ("KdV" if name == "kdv" else name)+".mat"
    path = Path("tmp/WSINDy_PDE/datasets")/filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(PAPER_SOURCE+"/datasets/"+filename, path)
    data = loadmat(path, squeeze_me=True)
    target = np.zeros(49)
    target[9] = -.5
    if name == "kdv":
        target[22] = -1.
    return data["x"], data["t"], data["U_exact"].T, target


def paper_system(name, U, x, t, workers=1):
    windows, strides = ((60, 60), (5, 5)) if name == "burgers" else ((45, 80), (12, 8))
    return wsindy.ConvolutionSystem(U, x, t, windows, strides, workers)


def save_csv(path, records, append=False):
    if not records:
        return
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        if not append or new:
            writer.writeheader()
        writer.writerows(records)


def load_csv(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    text_fields = {"problem", "grid", "method", "status", "w", "selected_terms"}
    for row in rows:
        for key, value in row.items():
            if key not in text_fields:
                row[key] = value == "True" if value in ("True", "False") else float(value)
        if row["grid"].isdigit():
            row["grid"] = int(row["grid"])
    return rows


def support_metrics(w, w_star):
    """Support only: |w_hat|>1e-12; coefficient accuracy is irrelevant."""
    if not np.isfinite(w).all():
        return dict(support_valid=False, support_exact=False,
                    **{k: float("nan") for k in ("tp", "fp", "fn", "precision", "recall", "f1", "tpr")})
    selected, true = np.abs(w) > 1e-12, np.asarray(w_star) != 0
    tp, fp, fn = (int(np.count_nonzero(mask)) for mask in
                  (selected & true, selected & ~true, ~selected & true))
    return dict(support_valid=True, support_exact=fp+fn == 0, tp=tp, fp=fp, fn=fn,
                precision=tp/(tp+fp) if tp+fp else float(not true.any()),
                recall=tp/(tp+fn) if tp+fn else 1.,
                f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.,
                tpr=tp/(tp+fp+fn) if tp+fp+fn else 1.)


def comparison_tables(trials, paper=False):
    """One table per noise level, with methods as columns."""
    aggregate = np.mean if paper else np.median
    methods = sorted({r["method"] for r in trials}, key=lambda m:
                     ("MLE" in m, m != "WSINDy", "L1" in m, float(m.split("=")[-1]) if "=" in m else 0))
    labels = [m.replace("WENDy-MLE", "MLE").replace("WENDy", "W").replace(" L1=", ":").replace(" HT", "-HT")
              for m in methods]
    lines = []
    for noise in sorted({r["noise"] for r in trials}):
        lines += [f"## Noise {100*noise:g}%", "", "| PDE / metric | " + " | ".join(labels) + " |",
                  "|---|" + "---:|"*len(methods)]
        for problem, grid in sorted({(r["problem"], r["grid"]) for r in trials}):
            groups = [[r for r in trials if (r["problem"], r["grid"], r["noise"], r["method"]) ==
                       (problem, grid, noise, method)] for method in methods]
            title = "KdV" if problem == "kdv" else "Burgers"
            if len({r["grid"] for r in trials if r["problem"] == problem}) > 1:
                title += f" ({grid})"
            for metric, key in (("E∞", "coefficient_linf"), ("E₂", "coefficient_error"),
                                ("Support", "support_exact"), ("Time (s)", "total_seconds"),
                                ("Optimizer", "optimizer_success")):
                values = []
                for rows in groups:
                    if not rows or key not in rows[0]:
                        value = "—"
                    elif key in ("support_exact", "optimizer_success"):
                        value = f"{sum(r[key] for r in rows)}/{len(rows)}"
                    else:
                        average = np.median if key == "total_seconds" else aggregate
                        value = format(average([r[key] for r in rows]), ".4g" if key == "total_seconds" else ".3g")
                    values.append(value)
                lines.append(f"| {title} · {metric} | " + " | ".join(values) + " |")
        lines.append("")
    return lines


def write_tables(out, trials, paper=False):
    lines = ["# Results", "", "W = WENDy; MLE = WENDy-MLE; W:α / MLE:α use L1 penalty λ=αλ_ref.",
             "Support and optimizer completion are counts; time is median seconds.",
             f"E₂ and E∞ are relative errors ({'mean' if paper else 'median'} over seeds), not percentages.",
             "Failed fits are included; inf denotes a nonfinite error; — means unavailable.", ""]
    lines += comparison_tables(trials, paper)
    lines += ["E₂ = ‖ŵ−w★‖₂/‖w★‖₂; E∞ = max relative error on true nonzero terms.",
              "Noise = σ/RMS(clean u). Time includes weak-form assembly and fitting, excluding data generation and file writing.",
              "Settings: [config.json](config.json). Raw fits: [trials.csv](trials.csv).", ""]
    (out/"summary.md").write_text("\n".join(lines))


def run(args):
    out = Path(args.output)
    paper = args.setup == "paper"
    config = {k: v for k, v in vars(args).items() if k != "resume"}
    config["solver_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("wsindy.py", "wendy.py", "wendy_mle.py", "experiment.py")}
    config["wendy_test_basis"] = dict(condition_limit=1e4, information_target=.95, max_rows=200)
    config["environment"] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy_version,
                                 platform=platform.platform(), blas={name: os.environ.get(name) for name in
                                 ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    if paper:
        config["data_source"] = PAPER_SOURCE
        config["selection_units"] = "rescaled coefficients; errors reported in original units"
    if args.l1 and any(m != "WSINDy" for m in args.methods):
        config["l1_evaluation"] = "active-set SVD LASSO (IRLS); KKT-certified L-BFGS-B (MLE)"
    if args.resume:
        previous = json.loads((out/"config.json").read_text())
        changed = [k for k in config if k != "output" and config[k] != previous.get(k)
                   and not (k == "l1" and set(previous[k]) <= set(config[k]))]
        if changed:
            raise ValueError("Cannot resume changed settings: "+", ".join(changed))
        trials = load_csv(out/"trials.csv")
    else:
        out.mkdir(parents=True, exist_ok=False)
        trials = []
    (out/"config.json").write_text(json.dumps(config, indent=2))
    completed = {(r["problem"], r["grid"], r["noise"], r["seed"], r["method"]) for r in trials}
    for name in args.problems:
        problem = PROBLEMS[name]
        sparsity = args.sparsity if args.sparsity is not None else len(problem.target)
        methods = [("WSINDy", wsindy, {})] if "WSINDy" in args.methods else []
        for label, module in (("WENDy", wendy), ("WENDy-MLE", wendy_mle)):
            if label in args.methods:
                methods.append((label+" HT", module, dict(sparsity=sparsity)))
                methods.extend((f"{label} L1={alpha:g}", module, dict(l1=alpha)) for alpha in args.l1)
        for grid in [None] if paper else args.grids:
            if paper:
                x, t, u, w_star = paper_data(name)
                grid, operator_seconds = f"{len(x)}×{len(t)}", 0.
            else:
                x, t = np.linspace(*problem.xlim, grid), np.linspace(*problem.tlim, grid)
                u, w_star = problem.solution(x, t), problem.w_star()
                start = perf_counter()
                A = wsindy.build_test_matrices(x, t, problem.alpha, problem.half_widths, args.centers)
                operator_seconds = perf_counter()-start
            for noise in args.noise:
                for seed in range(args.seeds if noise else 1):
                    key = (name, grid, noise, seed)
                    if all((*key, method) in completed for method, _, _ in methods):
                        continue
                    sigma = noise*np.sqrt(np.mean(u**2))
                    U = u+sigma*np.random.default_rng(seed).normal(size=u.shape)
                    start = perf_counter()
                    system = (paper_system(name, U, x, t, args.workers) if paper else
                              wsindy.WeakSystem(U, A, wsindy.polynomial_dictionary(), problem.alpha))
                    common_seconds = operator_seconds+perf_counter()-start
                    coefficient_scale = system.coefficient_scale if paper else np.ones_like(w_star)
                    fit_sigma = sigma*system.noise_scale if paper else sigma
                    for method, module, selection in methods:
                        if (*key, method) in completed:
                            continue
                        start = perf_counter()
                        try:
                            if module is wsindy:
                                result = module.fit(system, thresholds=np.logspace(-4, 0, 50) if paper else None,
                                                    threshold_scale=coefficient_scale if paper else None)
                            else:
                                result = module.fit(system, sigma=fit_sigma, maxiter=args.maxiter, tol=args.tol,
                                                    time_limit=args.time_limit, **selection)
                        except (np.linalg.LinAlgError, FloatingPointError) as exc:
                            result = wsindy.FitResult(np.full_like(w_star, np.nan), [], perf_counter()-start, 0, False, str(exc))
                        estimate = coefficient_scale*result.w
                        error = np.linalg.norm(estimate-w_star)/np.linalg.norm(w_star)
                        linf = np.max(np.abs((estimate-w_star)[w_star != 0]/w_star[w_star != 0]))
                        row = dict(problem=name, grid=grid, noise=noise, sigma=float(sigma), seed=seed,
                                   method=method, coefficient_error=float(error) if np.isfinite(error) else float("inf"),
                                   coefficient_linf=float(linf) if np.isfinite(linf) else float("inf"),
                                   **support_metrics(result.w, w_star/coefficient_scale), optimizer_success=result.success,
                                   status=result.message, iterations=result.iterations, fit_seconds=result.seconds,
                                   total_seconds=common_seconds+result.seconds, w=json.dumps(estimate.tolist()),
                                   selected_terms=json.dumps(np.flatnonzero(np.abs(result.w) > 1e-12).tolist()))
                        trials.append(row)
                        save_csv(out/"trials.csv", [row], append=True)
                        completed.add((*key, method))
                        print(f"{name} noise={noise:g} seed={seed} {method}: E2={error:.3g}, "
                              f"{result.seconds:.1f}s, success={result.success}", flush=True)
    write_tables(out, trials, paper)
    print(f"Saved {len(trials)} fits and summary.md to {out}")
    return trials


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--methods", nargs="+", choices=["WSINDy", "WENDy", "WENDy-MLE"], default=["WSINDy", "WENDy", "WENDy-MLE"])
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--grids", nargs="+", type=int, default=[256])
    parser.add_argument("--noise", nargs="+", type=float, default=[0., .2])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="FFT and exact convolution-covariance threads")
    parser.add_argument("--sparsity", type=int, help="HT term count; default Burgers=1, KdV=2")
    parser.add_argument("--l1", nargs="*", type=float, default=[1e-10, 1e-8, 1e-6, 1e-4, .01], help="Fixed relative L1 strengths")
    parser.add_argument("--centers", type=int, default=11)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--time-limit", type=float, default=200., help="Seconds per fit; checked between numerical operations")
    parser.add_argument("--output", default="results/pde_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--resume", action="store_true", help="Continue an existing output with identical settings")
    args = parser.parse_args()
    if min(args.grids) < 32 or args.seeds < 1 or args.workers < 1 or not np.isfinite(args.noise).all() or min(args.noise) < 0 or args.centers < 4 or args.maxiter < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("Require grids>=32, seeds/workers>=1, finite noise>=0, centers>=4 and positive finite tolerances/maxiter")
    if not np.isfinite(args.l1).all() or any(v <= 0 for v in args.l1) or (args.sparsity is not None and not 1 <= args.sparsity <= 10):
        parser.error("Require finite L1 strengths>0 and 1<=sparsity<=10")
    if not np.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("Require a positive finite time limit")
    return args


if __name__ == "__main__":
    args = parse_args()
    with set_workers(args.workers):
        run(args)
