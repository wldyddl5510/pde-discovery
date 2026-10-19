"""IB, a few seeds: each method's objective ||b - G w||^2 (G, b from its own loss), selected PDE,
coefficient errors and Gurobi status, on the observations run.py uses. Writes a markdown section
that report.py can insert into results.md (--include).

    python benchmark_results/ib_seeds/table.py --noise 1 --trials 0 5 7 --methods wsindy best-subset \
        -o benchmark_results/ib_seeds/IB_raw_noise1.md
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pdes import PDES, load, metrics, observe, trial_seed
from run import METHODS
from wsindy import estimate, fit
from wsindy.regression import _loss

NAME = "IB"


def equation(coef, terms, contribution, dense=8):
    """Selected PDE; with more than `dense` nonzeros, their count and the three largest contributions."""
    nonzero = np.flatnonzero(coef)
    if len(nonzero) > dense:
        top = nonzero[np.argsort(contribution[nonzero])[::-1][:3]]
        return f"{len(nonzero)} nonzero terms; largest: " + " ".join(f"{coef[j]:+.4g} {terms[j].label}" for j in top)
    parts = [f"{coef[j]:+.4g} {terms[j].label}" for j in nonzero]
    return "u_t = " + (" ".join(parts) if parts else "0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=1.)
    parser.add_argument("--trials", type=int, nargs="+", default=[0, 5, 7])
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=["wsindy", "best-subset"])
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    pde = PDES[NAME]; clean, grid = load(NAME)
    lines = [f"## IB case study: {', '.join(args.methods)} on single seeds (noise ratio {args.noise:g})", "",
             "Trials of `run.py` (same observations as the tables). Objective = ||b - G w||^2 of the method's own "
             "regression (G, b from its loss) at the returned coefficients, in the scaled weak system; for LASSO and "
             "EIV, which minimize other objectives, only this least-squares part. `wsindy` is raw WSINDy with MSTLS; "
             "`best-subset` solves the r = 1 MIO with Gurobi. E2 and E_inf as in the tables. Truth: u_t = -0.5 (u^2)_x.",
             "", "| Trial | Method | Objective | Selected PDE | E2 | E_inf | Gurobi optimal |",
             "| ---: | :--- | ---: | :--- | ---: | ---: | :---: |"]
    for trial in args.trials:
        U, noise_std = observe(pde, clean, args.noise, trial_seed(NAME, args.noise, trial))
        for name in args.methods:
            m = METHODS[name]
            uhat = estimate(U, grid, pde, m.estimator)[0] if "uhat" in (m.X, m.Y) else U
            states = dict(U=U, uhat=uhat)
            system = m.system(pde, U, grid, X=states[m.X], Y=states[m.Y])
            system.clean, system.noise_std = clean, noise_std   # oracle-only, as in run.py (EIV)
            G, b = _loss(getattr(m.regression, "loss", "ls")).matrices(system)
            b, scales = b[:, 0], system.scales[:, 0]
            result = fit(system, m.regression)
            coef = result.coef[0]
            errors = metrics(result.coef, pde.truth_matrix(system.terms))
            optimal = result.info.get("optimal")
            lines.append(f"| {trial} | {name} | {np.sum((b-G@(coef/scales))**2):.4f} | "
                         f"{equation(coef, system.terms, np.abs(coef/scales)*np.linalg.norm(G, axis=0))} | "
                         f"{errors['e2']:.4g} | {errors['e_inf']:.4g} | {'—' if optimal is None else 'yes' if optimal else 'no'} |")
    lines.append("")
    output = args.output or Path(__file__).with_name(f"IB_noise{args.noise:g}.md")
    output.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
