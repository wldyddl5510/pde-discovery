"""Paired EIV diagnostic: forbid all terms outside the true PDE support.

Run from the repository root with the same Python environment as run.py:
    python benchmark_results/eiv/true_support/run_study.py --workers 4

Uses existing EIV records, observations and conic solver unchanged. Results stay
in this study directory. The OLS control is fit on the same Hermite-corrected
matrix and true support; it is not an additional benchmark method.
"""
from __future__ import annotations

import os
for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy import linalg

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pdes import PDES, load, metrics, observe, trial_seed
from wsindy import EIV, WeakSystem
from wsindy.eiv import conic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic(baseline):
    start = perf_counter()
    name, ratio = baseline["name"], baseline["noise_ratio"]
    pde = PDES[name]
    clean, grid = load(name, download=False)
    assert baseline["seed"] == trial_seed(name, ratio, baseline["trial"])
    U, std = observe(pde, clean, ratio, baseline["seed"])
    np.testing.assert_allclose(std, baseline["noise_std"], rtol=1e-13, atol=0.)
    np.testing.assert_allclose(std[0]**2, baseline["variance"], rtol=1e-13, atol=0.)
    system = WeakSystem(pde, U, grid)
    truth = pde.truth_matrix(system.terms)[0]
    active = truth != 0
    X = system.columns(system.terms, U, variance=baseline["variance"], rescale=False)
    y = system.columns(system.lhs, U, rescale=False)[:, 0]
    scales = np.linalg.norm(X, axis=0)
    assert np.all(np.isfinite(scales)) and np.all(scales > 0)
    Z = X/scales
    A, units, target = Z[:, active], scales[active], truth[active]
    true_theta = target*units
    truth_residual = float(np.max(np.abs(y-A@true_theta)))
    tau, mu = baseline["tau"], baseline["mu"]
    assert mu == 0., "This diagnostic expects the saved tau_share=1 benchmark."
    np.testing.assert_allclose(truth_residual, tau, rtol=1e-7, atol=1e-9)
    full_beta = np.asarray(baseline["coefficients"])[0]
    np.testing.assert_allclose(metrics(full_beta, truth)["e2"], baseline["e2"], rtol=1e-12)
    rebuilt_full_residual = float(np.max(np.abs(y-Z@(full_beta*scales))))
    np.testing.assert_allclose(rebuilt_full_residual, baseline["residual_inf"], rtol=1e-7, atol=1e-8)

    def describe(theta, lam=None, t=None, status=None, success=True, diagnostics=None):
        beta = theta/units
        support = np.abs(theta) > EIV().support_tolerance*np.max(np.abs(theta))
        residual = float(np.max(np.abs(y-A@theta)))
        result = dict(metrics(beta, target, support), coefficients=beta.tolist(),
                      normalized_coefficients=theta.tolist(), residual_inf=residual,
                      feasible=bool(residual <= tau+1e-7*max(1., np.max(np.abs(y)))),
                      coefficient_ratios=(beta/target).tolist(),
                      norm_ratio=float(np.linalg.norm(beta)/np.linalg.norm(target)),
                      lam=lam, status=status, success=bool(success))
        if lam is not None:
            objective = float(np.sum(np.abs(theta))+lam*t)
            truth_objective = float(np.sum(np.abs(true_theta))+lam*np.linalg.norm(true_theta))
            result.update(t=t, objective=objective, truth_objective=truth_objective,
                          objective_ratio=objective/truth_objective, diagnostics=diagnostics)
        return result

    path = []
    for lam in baseline["lams"]:
        theta, t, status, success, info = conic(A, y, mu, tau, lam, EIV().tolerance)
        path.append(describe(theta, lam, t, status, success, info))
    assert all(np.isfinite(r["e2"]) for r in path), "A conic candidate returned no finite coefficients."
    raw_best = int(np.nanargmin([r["e2"] for r in path]))  # original, unchecked oracle selection
    eligible = [i for i, r in enumerate(path) if r["success"] and r["feasible"]]
    assert eligible, f"No validated candidate: {name}, noise={ratio}, trial={baseline['trial']}"
    best = min(eligible, key=lambda i: path[i]["e2"])
    fixed = baseline["lams"].index(baseline["lam"])
    ols_theta, _, rank, singular = linalg.lstsq(A, y, lapack_driver="gelsd")
    ols = describe(ols_theta, status="LeastSquares")
    ols.update(rank=int(rank), condition_number=float(singular[0]/singular[-1]))

    full = {k: baseline[k] for k in ("e2", "e_inf", "exact_support", "tp", "fn", "fp", "lam",
                                    "coefficients", "status", "success", "residual_inf")}
    full["true_term_coefficient_ratios"] = (full_beta[active]/target).tolist()
    # The smallest c in [0,1] for which c beta* satisfies the same residual constraint.
    # If c<1, every positive lambda prefers that feasible vector to beta*.
    prediction = A@true_theta
    nz = prediction != 0
    endpoints = np.stack(((y[nz]-tau)/prediction[nz], (y[nz]+tau)/prediction[nz]))
    radial_min = float(np.clip(np.max(np.minimum(*endpoints)), 0., 1.))
    return dict(name=name, noise_ratio=ratio, trial=baseline["trial"], seed=baseline["seed"],
                tau=tau, mu=mu, variance=baseline["variance"], true_residual_inf=truth_residual,
                tau_relative_reconstruction_error=abs(truth_residual-tau)/max(1., tau),
                full_residual_relative_reconstruction_error=abs(rebuilt_full_residual-baseline["residual_inf"])
                    /max(1., baseline["residual_inf"]),
                active_terms=[t.label for t, keep in zip(system.terms, active) if keep],
                true_coefficients=target.tolist(), radial_min=radial_min, full=full,
                fixed=path[fixed], oracle=path[best], raw_oracle=path[raw_best], ols=ols, path=path,
                runtime=perf_counter()-start)


def chunk(rows):
    return [diagnostic(row) for row in rows]


def average(rows, method, key):
    values = np.array([r[method][key] for r in rows])
    return float(np.mean(values))


def report(directory):
    records = [json.loads(line) for line in (directory/"trials.jsonl").read_text().splitlines()]
    records.sort(key=lambda r: (list(PDES).index(r["name"]), r["noise_ratio"], r["trial"]))
    groups = {}
    for r in records:
        groups.setdefault((r["name"], r["noise_ratio"]), []).append(r)
    summaries = []
    for (name, noise), rows in groups.items():
        result = dict(name=name, noise_ratio=noise, trials=len(rows), terms=rows[0]["active_terms"],
                      true_coefficients=rows[0]["true_coefficients"])
        for method in ("full", "fixed", "oracle", "ols"):
            values = np.array([r[method]["e2"] for r in rows])
            result[method] = dict(mean_e2=float(values.mean()), median_e2=float(np.median(values)),
                mean_e_inf=average(rows, method, "e_inf"), exact_support=average(rows, method, "exact_support"),
                mean_coefficients=np.mean([np.array(r[method]["coefficients"])[0]
                    if method == "full" else r[method]["coefficients"] for r in rows], axis=0).tolist())
            if method != "full":
                result[method].update(mean_coefficient_ratios=np.mean([r[method]["coefficient_ratios"]
                    for r in rows], axis=0).tolist(), feasible_rate=average(rows, method, "feasible"))
        result.update(mean_radial_min=float(np.mean([r["radial_min"] for r in rows])),
                      mean_objective_ratio=average(rows, "oracle", "objective_ratio"),
                      objective_below_truth=sum(r["oracle"]["objective_ratio"] < 1-1e-5 for r in rows),
                      oracle_support_drop=sum(r["oracle"]["fn"] > 0 for r in rows),
                      fixed_failed=sum(not r["fixed"]["success"] for r in rows),
                      raw_oracle_failed=sum(not r["raw_oracle"]["success"] for r in rows),
                      failed_path_solves=sum(not p["success"] for r in rows for p in r["path"]),
                      max_tau_reconstruction_error=max(r["tau_relative_reconstruction_error"] for r in rows))
        summaries.append(result)
    (directory/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    text = ["# EIV with the true support supplied", "",
        "Paired diagnostic of the saved `benchmark_results/runs/eiv` results. Only IB, KdV and KS.", "",
        "## Protocol", "",
        "- Same observations/seeds, true noise variance, Hermite correction, weak equations, unit-norm columns, "
        "Gurobi conic solver and tolerance (1e-8) as the saved EIV benchmark.",
        "- Keep each saved tau exactly; mu = 0. Forbid all columns outside the true support. "
        "Allowed true terms may still receive zero coefficients; their signs and values are not supplied.",
        "- **Fixed lambda:** reuse the lambda selected in the full dictionary. "
        "**Oracle lambda:** repeat the same 13-value grid (1e-3 to 1e3) and select minimum physical coefficient E2.",
        "- Original-solver failures are retained in the path and fixed-lambda diagnostics. "
        "The oracle column selects only converged, constraint-validated candidates; "
        "raw_oracle separately records the original unchecked argmin rule. "
        "Fixed-lambda averages include returned finite coefficients even when convergence was not certified; see the audit table.",
        "- **OLS control:** unpenalized least squares on the same corrected matrix and true support. "
        "Its infinity-residual constraint is checked, not imposed.",
        "- No coefficient thresholding or refitting after the conic solve. "
        "Support diagnostics use the original relative 1e-3 cutoff in normalized coordinates.",
        "- E2 is relative L2 error over all coefficients. Forbidden coefficients are exactly zero. "
        "Exact support is not a useful standalone score here because the support was supplied.",
        "- Noise level 0 repeats identical clean observations 100 times; those repeats are not independent data.",
        "", "## Mean E2", "",
        "| Task | Noise | Trials | Full EIV | True support, fixed lambda | True support, oracle lambda | True-support OLS |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        text.append(f"| {s['name']} | {s['noise_ratio']:g} | {s['trials']} | "
            + " | ".join(f"{s[m]['mean_e2']:.6g}" for m in ("full", "fixed", "oracle", "ols"))+" |")
    text += ["", "## Median E2", "",
        "Large false physical coefficients can dominate the full-dictionary mean; paired medians are also shown.", "",
        "| Task | Noise | Full EIV | True support, fixed lambda | True support, oracle lambda | True-support OLS |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        text.append(f"| {s['name']} | {s['noise_ratio']:g} | "
            + " | ".join(f"{s[m]['median_e2']:.6g}" for m in ("full", "fixed", "oracle", "ols"))+" |")
    text += ["", "## Coefficients and feasibility", "",
        "The coefficient ratios are beta_hat / beta_true, averaged per term: 1 = correct, "
        "0 = dropped, values between 0 and 1 = attenuated magnitude.", ""]
    for name in dict.fromkeys(s["name"] for s in summaries):
        cases = [s for s in summaries if s["name"] == name]
        text += [f"### {name}", "", "Term order: "+", ".join(cases[0]["terms"])+".", "",
            "True coefficients: "+", ".join(str(v) for v in cases[0]["true_coefficients"])+".", "",
            "| Noise | Oracle EIV mean coefficient ratios | OLS mean coefficient ratios | EIV trials dropping a true term | OLS feasible | EIV objective below truth |",
            "| ---: | :--- | :--- | ---: | ---: | ---: |"]
        for s in cases:
            fmt = lambda values: ", ".join(f"{v:.4g}" for v in values)
            text.append(f"| {s['noise_ratio']:g} | {fmt(s['oracle']['mean_coefficient_ratios'])} | "
                f"{fmt(s['ols']['mean_coefficient_ratios'])} | {s['oracle_support_drop']}/{s['trials']} | "
                f"{s['ols']['feasible_rate']:.0%} | {s['objective_below_truth']}/{s['trials']} |")
    failures = sum(not p["success"] for r in records for p in r["path"])
    text += ["", "## Audit", "",
        f"- {len(records)} paired trials; {sum(len(r['path']) for r in records)} conic solves; {failures} failed path solves.",
        f"- Maximum relative tau reconstruction discrepancy: {max(r['tau_relative_reconstruction_error'] for r in records):.3g}.",
        f"- Maximum relative full-fit residual reconstruction discrepancy: {max(r['full_residual_relative_reconstruction_error'] for r in records):.3g}.",
        "- Every reconstructed true residual matches saved tau, and every saved full-fit E2 was independently recomputed.",
        "", "| Task | Noise | Failed fixed-lambda solves | Failed unchecked oracle choices | Failed path solves |",
        "| :--- | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        text.append(f"| {s['name']} | {s['noise_ratio']:g} | {s['fixed_failed']}/{s['trials']} | "
                    f"{s['raw_oracle_failed']}/{s['trials']} | {s['failed_path_solves']}/{13*s['trials']} |")
    text += [
        "", "## Reproduce", "", "```sh",
        "python benchmark_results/eiv/true_support/run_study.py --workers 4", "```", "",
        "Completed trials are skipped. Use --report-only to regenerate this report. "
        "Raw paired coefficients and every lambda candidate are in trials.jsonl; aggregate values are in summary.json.", ""]
    (directory/"results.md").write_text("\n".join(text))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--noise", type=float, nargs="+")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(directory)
        return
    baselines = []
    for name in ("IB", "KdV", "KS"):
        path = ROOT/"benchmark_results/runs/eiv"/f"{name}.jsonl"
        baselines.extend(r for r in map(json.loads, path.read_text().splitlines())
            if r["trial"] < args.trials and (args.noise is None or r["noise_ratio"] in args.noise))
    source_files = [Path(__file__).resolve(), ROOT/"wsindy/eiv.py", ROOT/"wsindy/system.py",
                    ROOT/"wsindy/regression.py", ROOT/"pdes.py", ROOT/"wsindy/testfunctions.py"]
    manifest = dict(protocol="same-eiv-true-support-v1", sources={str(p.relative_to(ROOT)): digest(p) for p in source_files},
        baseline_hashes={name: digest(ROOT/"benchmark_results/runs/eiv"/f"{name}.jsonl") for name in ("IB", "KdV", "KS")},
        baseline_config=json.loads((ROOT/"benchmark_results/runs/eiv/config.json").read_text()),
        trials=args.trials, noise=args.noise, selected_records=len(baselines))
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Study inputs changed; use a fresh output directory."
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    path = directory/"trials.jsonl"
    done = set()
    if path.exists():
        done = {(r["name"], r["noise_ratio"], r["trial"]) for r in map(json.loads, path.read_text().splitlines())}
    grouped = {}
    for row in baselines:
        if (row["name"], row["noise_ratio"], row["trial"]) not in done:
            grouped.setdefault((row["name"], row["noise_ratio"]), []).append(row)
    chunks = [rows[i:i+5] for rows in grouped.values() for i in range(0, len(rows), 5)]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(chunk, rows): rows for rows in chunks}
        for count, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            with path.open("a") as sink:
                sink.writelines(json.dumps(r, allow_nan=False)+"\n" for r in rows)
            if count % 10 == 0 or count == len(chunks):
                print(f"[{count}/{len(chunks)} chunks] {rows[0]['name']} noise={rows[0]['noise_ratio']:g}", flush=True)
    summaries = report(directory)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
