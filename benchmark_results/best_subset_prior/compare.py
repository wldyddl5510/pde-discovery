"""IB: best subset (true r) with MSTLS's magnitude prior vs MSTLS, on the same observations as run.py.

    python benchmark_results/best_subset_prior/compare.py --workers 9

Variants: no prior; fixed lambda; lambda = the threshold MSTLS selects on that trial (same
hypothesis space as MSTLS's final step). Writes IB.md and IB.csv next to this script.
"""
import argparse, csv, os, sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pdes import PDES, load, metrics, observe, trial_seed
from run import METHODS, TRUE_R
from wsindy import BestSubset, MSTLS, estimate, fit

NAME, FIXED = "IB", (1e-3, 1e-2, 1e-1)
FAMILIES = {"wsindy": ("wsindy", "best-subset"), "filtered": ("filtered", "filtered-best-subset"),
            "debiased": ("debiased", "debiased-best-subset")}

def trial(args):
    family, ratio, t = args
    pde, (mstls_name, bs_name) = PDES[NAME], FAMILIES[family]
    clean, grid = load(NAME)
    U, _ = observe(pde, clean, ratio, trial_seed(NAME, ratio, t))
    m = METHODS[mstls_name]
    uhat = estimate(U, grid, pde, m.estimator)[0] if "uhat" in (m.X, m.Y) else U
    states = dict(U=U, uhat=uhat)
    system = m.system(pde, U, grid, X=states[m.X], Y=states[m.Y])
    truth = pde.truth_matrix(system.terms)
    out = {}
    ms = fit(system, m.regression)
    out["MSTLS"] = ms.coef
    bs = METHODS[bs_name].regression
    out["BS"] = fit(system, bs).coef
    for lam in FIXED:
        out[f"BS+prior {lam:g}"] = fit(system, replace(bs, threshold=lam)).coef
    out["BS+prior MSTLS-lambda"] = fit(system, replace(bs, threshold=ms.info["threshold"])).coef
    return family, ratio, {k: metrics(c, truth) for k, c in out.items()}, ms.info["threshold"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    ratios = (0., .1, .2, .5, .75, 1., 2.)
    tasks = [(f, q, t) for f in FAMILIES for q in ratios for t in range(1 if q == 0 else args.trials)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(trial, tasks, chunksize=10))
    rows = []
    for f in FAMILIES:
        for q in ratios:
            group = [r for r in results if r[0] == f and r[1] == q]
            for variant in group[0][2]:
                ms = [r[2][variant] for r in group]
                rows.append(dict(family=f, noise_ratio=q, variant=variant, trials=len(group),
                                 exact=float(np.mean([m["exact_support"] for m in ms])),
                                 e_inf=float(np.mean([m["e_inf"] for m in ms])), e2=float(np.mean([m["e2"] for m in ms])),
                                 mstls_lambda_median=float(np.median([r[3] for r in group]))))
    here = Path(__file__).parent
    with (here/"IB.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    lines = ["# IB: best subset with MSTLS's magnitude prior", "",
             f"Exact-support rate (%) · mean E2; r = {TRUE_R[NAME]}; same seeds as run.py. "
             "Prior: lambda*max(1,B_j) <= |c_j| <= min(1,B_j)/lambda (MSTLS's relative bounds).", ""]
    for f in FAMILIES:
        lines += [f"## {f}", "", "| Variant | " + " | ".join(f"{q:g}" for q in ratios) + " |",
                  "| :--- |" + " ---: |"*len(ratios)]
        for v in variants:
            cells = {r["noise_ratio"]: r for r in rows if r["family"] == f and r["variant"] == v}
            lines.append(f"| {v} | " + " | ".join(f"{100*cells[q]['exact']:.0f} · {cells[q]['e2']:.2g}" for q in ratios) + " |")
        lam = {r["noise_ratio"]: r["mstls_lambda_median"] for r in rows if r["family"] == f}
        lines += ["", "Median MSTLS lambda: " + ", ".join(f"{q:g}: {lam[q]:.3g}" for q in ratios), ""]
    (here/"IB.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
