"""Synthetic PDE coefficient recovery. Run: python experiment.py"""
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import softmax

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


def save_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def support_metrics(w, w_star):
    """Support only: |w_hat|>1e-12; coefficient accuracy is irrelevant."""
    if not np.isfinite(w).all():
        return dict(support_valid=False, support_exact=False,
                    **{k: float("nan") for k in ("tp", "fp", "fn", "precision", "recall", "f1")})
    selected, true = np.abs(w) > 1e-12, np.asarray(w_star) != 0
    tp, fp, fn = (int(np.count_nonzero(mask)) for mask in
                  (selected & true, selected & ~true, ~selected & true))
    return dict(support_valid=True, support_exact=fp+fn == 0, tp=tp, fp=fp, fn=fn,
                precision=tp/(tp+fp) if tp+fp else float(not true.any()),
                recall=tp/(tp+fn) if tp+fn else 1.,
                f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.)


def write_tables(out, trials):
    descriptions = {
        "support": "정확한 support 복원율 (%). |w_hat|>1e-12로 판정하며, 높을수록 좋습니다.",
        "coefficient_accuracy": "전체 계수 벡터의 상대 L2 오차 중앙값 (%): 100 × ||w_hat-w_star||₂ / ||w_star||₂. 낮을수록 좋습니다.",
    }
    groups = {}
    for row in trials:
        groups.setdefault((row["problem"], row["grid"]), []).append(row)
    for name, description in descriptions.items():
        lines = [description, "", "열은 잡음 비율 σ/RMS(u)입니다. L1 값은 데이터 기반 λ_ref에 곱하는 상대 penalty입니다.", ""]
        for (problem, grid), rows in groups.items():
            noises = sorted(set(r["noise"] for r in rows))
            methods = list(dict.fromkeys(r["method"] for r in rows))
            grouped = {(m, n): [r for r in rows if r["method"] == m and r["noise"] == n]
                       for m in methods for n in noises}
            samples = [grouped[methods[0], n] for n in noises]
            title = "KdV" if problem == "kdv" else "Burgers"
            lines += [f"## {title} · {grid}×{grid}", "",
                      "실제 σ: " + ", ".join(f"{r[0]['sigma']:.4g}" for r in samples) + ".",
                      "반복 횟수: " + ", ".join(str(len(r)) for r in samples) + ".", "",
                      "| 방법 | " + " | ".join(f"{100*n:g}%" for n in noises) + " |",
                      "|---|" + "---:|"*len(noises)]
            for method in methods:
                values = []
                for noise in noises:
                    runs = grouped[method, noise]
                    value = (np.mean([r["support_exact"] for r in runs]) if name == "support"
                             else np.median([r["coefficient_error"] for r in runs]))
                    values.append(format(100*value, ".3g" if name == "support" else ".4g"))
                lines.append("| " + method + " | " + " | ".join(values) + " |")
            lines.append("")
        failed = sum(not r["optimizer_success"] for r in trials)
        lines += [f"전체 {len(trials)}회 중 optimizer 미수렴 {failed}회도 포함했습니다. 종료 상태와 추정 계수는 trials.csv에 있습니다.", ""]
        (out/f"{name}.md").write_text("\n".join(lines))


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/"config.json").write_text(json.dumps(vars(args), indent=2))
    trials = []
    for name in args.problems:
        problem = PROBLEMS[name]
        w_star = problem.w_star()
        for grid in args.grids:
            x, t = np.linspace(*problem.xlim, grid), np.linspace(*problem.tlim, grid)
            u = problem.solution(x, t)
            start = perf_counter()
            A = wsindy.build_test_matrices(x, t, problem.alpha, problem.half_widths, args.centers)
            operator_seconds = perf_counter()-start
            for noise in args.noise:
                for seed in range(args.seeds if noise else 1):
                    sigma = noise*np.sqrt(np.mean(u**2))
                    U = u+sigma*np.random.default_rng(seed).normal(size=u.shape)
                    start = perf_counter()
                    system = wsindy.WeakSystem(U, A, wsindy.polynomial_dictionary(), problem.alpha)
                    assembly_seconds = perf_counter()-start
                    common_seconds = operator_seconds+assembly_seconds
                    sparsity = args.sparsity if args.sparsity is not None else len(problem.target)
                    methods = [("WSINDy", wsindy, {})]
                    for label, module in (("WENDy", wendy), ("WENDy-MLE", wendy_mle)):
                        methods.append((label+" HT", module, dict(sparsity=sparsity)))
                        methods.extend((f"{label} L1={alpha:g}", module, dict(l1=alpha)) for alpha in args.l1)
                    for method, module, selection in methods:
                        start = perf_counter()
                        try:
                            result = (module.fit(system) if module is wsindy else
                                      module.fit(system, sigma=sigma, maxiter=args.maxiter, tol=args.tol, **selection))
                        except (np.linalg.LinAlgError, FloatingPointError) as exc:
                            result = wsindy.FitResult(np.full_like(w_star, np.nan), [], perf_counter()-start, 0, False, str(exc))
                        error = np.linalg.norm(result.w-w_star)/np.linalg.norm(w_star)
                        trials.append(dict(problem=name, grid=grid, noise=noise, sigma=float(sigma), seed=seed,
                                           method=method, coefficient_error=float(error) if np.isfinite(error) else float("inf"),
                                           **support_metrics(result.w, w_star), optimizer_success=result.success,
                                           status=result.message, iterations=result.iterations, fit_seconds=result.seconds,
                                           total_seconds=common_seconds+result.seconds, w=json.dumps(result.w.tolist())))
                    print(f"{name} grid={grid} noise={noise:g} seed={seed}: " + ", ".join(
                        f"{r['method']} error={r['coefficient_error']:.3g}" for r in trials[-len(methods):]), flush=True)
    save_csv(out/"trials.csv", trials)
    write_tables(out, trials)
    print(f"Saved {len(trials)} fits and two comparison tables to {out}")
    return trials


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--grids", nargs="+", type=int, default=[256])
    parser.add_argument("--noise", nargs="+", type=float, default=[0., .01, .1, .5, 1.])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sparsity", type=int, help="HT term count; default Burgers=1, KdV=2")
    parser.add_argument("--l1", nargs="+", type=float, default=[1e-6, 1e-4, .01], help="Fixed relative L1 strengths")
    parser.add_argument("--centers", type=int, default=11)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--output", default="results/pde_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if min(args.grids) < 32 or args.seeds < 1 or not np.isfinite(args.noise).all() or min(args.noise) < 0 or args.centers < 4 or args.maxiter < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("Require grids>=32, seeds>=1, finite noise>=0, centers>=4 and positive finite tolerances/maxiter")
    if not np.isfinite(args.l1).all() or min(args.l1) <= 0 or (args.sparsity is not None and not 1 <= args.sparsity <= 10):
        parser.error("Require finite L1 strengths>0 and 1<=sparsity<=10")
    return args


if __name__ == "__main__":
    run(parse_args())
