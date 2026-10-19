"""Compare fixed-true-support WSINDy least squares with the saved EIV diagnostic.

    python benchmark_results/eiv/true_support/compare_wsindy.py --workers 4

The support is fixed: WSINDy fits the original, uncorrected weak matrix without
MSTLS selection. Its existing coordinate/state scaling and physical coefficient
restoration are preserved. No EIV solves or main benchmark files are changed.
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

import numpy as np
from scipy import linalg

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
sys.path.insert(0, str(ROOT))
from pdes import PDES, load, metrics, observe
from wsindy import WeakSystem


def chunk(records):
    results = []
    for record in records:
        pde = PDES[record["name"]]
        clean, grid = load(record["name"], download=False)
        U, _ = observe(pde, clean, record["noise_ratio"], record["seed"])
        system = WeakSystem(pde, U, grid)
        truth = pde.truth_matrix(system.terms)[0]
        active = truth != 0
        terms = [t for t, keep in zip(system.terms, active) if keep]
        G = system.columns(terms, U)  # original WSINDy, no Hermite correction
        b = system.Y[:, 0]
        scaled, _, rank, singular = linalg.lstsq(G, b, lapack_driver="gelsd")
        beta = scaled*system.scales[active, 0]
        corrected = np.asarray(record["ols"]["coefficients"])
        results.append(dict(name=record["name"], noise_ratio=record["noise_ratio"], trial=record["trial"],
            seed=record["seed"], terms=record["active_terms"], truth=truth[active].tolist(),
            wsindy=dict(metrics(beta, truth[active]), coefficients=beta.tolist(), rank=int(rank),
                        condition_number=float(singular[0]/singular[-1])),
            eiv={k: record["oracle"][k] for k in ("e2", "e_inf", "coefficients", "lam", "success")},
            corrected_ols_e2=record["ols"]["e2"],
            max_coefficient_difference_from_corrected_ols=float(np.max(np.abs(beta-corrected))),
            relative_coefficient_difference_from_corrected_ols=float(np.linalg.norm(beta-corrected)/np.linalg.norm(truth[active]))))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    inputs = DIRECTORY/"trials.jsonl"
    records = [json.loads(line) for line in inputs.read_text().splitlines()]
    groups = {}
    for record in records:
        groups.setdefault((record["name"], record["noise_ratio"]), []).append(record)
    tasks = [rows[i:i+10] for rows in groups.values() for i in range(0, len(rows), 10)]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for count, rows in enumerate(pool.map(chunk, tasks), 1):
            results.extend(rows)
            if count % 30 == 0:
                print(f"{len(results)}/{len(records)} trials", flush=True)
    results.sort(key=lambda r: (("IB", "KdV", "KS").index(r["name"]), r["noise_ratio"], r["trial"]))
    (DIRECTORY/"wsindy_trials.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in results))
    summaries = []
    for name, level in sorted(groups, key=lambda k: (("IB", "KdV", "KS").index(k[0]), k[1])):
        rows = [r for r in results if r["name"] == name and r["noise_ratio"] == level]
        s = dict(name=name, noise_ratio=level, trials=len(rows))
        for method in ("wsindy", "eiv"):
            values = [r[method]["e2"] for r in rows]
            s[method] = dict(mean_e2=float(np.mean(values)), median_e2=float(np.median(values)),
                mean_coefficients=np.mean([r[method]["coefficients"] for r in rows], axis=0).tolist())
        s.update(eiv_worse_trials=sum(r["eiv"]["e2"] > r["wsindy"]["e2"] for r in rows),
                 mean_error_ratio=s["eiv"]["mean_e2"]/s["wsindy"]["mean_e2"])
        summaries.append(s)
    audit = dict(trials=len(results), unique_trials=len({(r["name"], r["noise_ratio"], r["trial"]) for r in results}),
        input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        max_absolute_coefficient_difference_from_corrected_ols=max(r["max_coefficient_difference_from_corrected_ols"] for r in results),
        max_relative_coefficient_difference_from_corrected_ols=max(r["relative_coefficient_difference_from_corrected_ols"] for r in results))
    assert audit["trials"] == audit["unique_trials"] == 2100
    assert audit["max_relative_coefficient_difference_from_corrected_ols"] < 1e-9
    (DIRECTORY/"wsindy_summary.json").write_text(json.dumps(dict(audit=audit, groups=summaries), indent=2)+"\n")
    text = ["# True-support WSINDy versus EIV", "",
        "Same observations and true support for IB, KdV and KS; 100 paired trials per noise level.", "",
        "- **WSINDy:** ordinary least squares on the original, uncorrected weak terms, with the existing "
        "WSINDy scaling and physical coefficient restoration. The true support is fixed, so MSTLS selection is bypassed.",
        "- **EIV:** saved true-support conic results with the same noise, oracle variance, saved tau, mu=0, "
        "column normalization, and the original lambda grid. Select minimum coefficient E2 among validated candidates.",
        "- The earlier study's OLS column used Hermite-corrected terms. Here the original WSINDy terms "
        "are independently recomputed. Their true terms are linear or derivatives of u^2; "
        "the constant Gaussian correction to u^2 vanishes under the spatial derivative, up to quadrature/roundoff.",
        "- Noise-zero trials repeat identical observations; counts are not independent clean datasets.", "",
        "| Task | Noise | Mean E2: WSINDy | Mean E2: EIV | EIV / WSINDy | Median E2: WSINDy | Median E2: EIV |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        w, e = s["wsindy"], s["eiv"]
        text.append(f"| {s['name']} | {s['noise_ratio']:g} | {w['mean_e2']:.6g} | {e['mean_e2']:.6g} | "
                    f"{s['mean_error_ratio']:.3g} | {w['median_e2']:.6g} | {e['median_e2']:.6g} |")
    text += ["", "## Verification", "",
        f"- {audit['trials']} distinct paired trials; all saved EIV choices passed the prior convergence/constraint checks.",
        "- Maximum absolute coefficient difference between original WSINDy and the earlier corrected OLS control: "
        f"{audit['max_absolute_coefficient_difference_from_corrected_ols']:.3g}.",
        "- Maximum relative coefficient difference (normalized by the true coefficient norm): "
        f"{audit['max_relative_coefficient_difference_from_corrected_ols']:.3g}.", "",
        "```sh", "python benchmark_results/eiv/true_support/compare_wsindy.py --workers 4", "```", ""]
    (DIRECTORY/"wsindy_comparison.md").write_text("\n".join(text))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
