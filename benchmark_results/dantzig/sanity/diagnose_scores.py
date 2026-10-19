"""Diagnose cached scores without state reconstruction or weak FFT recomputation.

The exhaustive one-column LP is an independent check of the score-residual
tie-break. True support appears only in labeled diagnostics and evaluation; it
is never supplied to the study's estimator. Conditioning standardizations below
are diagnostics only and do not change any estimation constraint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def one_column_paths(H, b, truth, terms):
    """Solve every cardinality-one infinity-residual LP using independent HiGHS.

    Divide ALL score rows by the same ||b||_inf and substitute a column-scaled
    coefficient. Both changes leave the physical common-tolerance optimization
    exactly unchanged. No coordinate-dependent score row scaling is performed.
    """
    scale = float(np.max(np.abs(b)))
    y = b/scale
    paths = []
    for j in range(H.shape[1]):
        col = H[:, j]/scale
        magnitude = float(np.max(np.abs(col)))
        a = col/magnitude if magnitude else col
        matrix = np.vstack((np.column_stack((a, -np.ones(len(b)))),
                            np.column_stack((-a, -np.ones(len(b))))))
        solution = linprog([0., 1.], A_ub=matrix, b_ub=np.r_[y, -y],
                           bounds=[(None, None), (0., None)], method="highs",
                           options={"dual_feasibility_tolerance": 1e-10,
                                    "primal_feasibility_tolerance": 1e-10})
        if not solution.success:
            paths.append(dict(index=j, term=terms[j], success=False,
                              status=int(solution.status), message=solution.message))
            continue
        coefficient = float(solution.x[0]/magnitude) if magnitude else 0.
        estimate = np.zeros_like(truth)
        estimate[j] = coefficient
        residual = float(np.max(np.abs(b-H@estimate))/scale)
        # Re-evaluate the cached-double inputs with a wider accumulator, too.
        extended = np.max(np.abs(b.astype(np.longdouble)
                                 -H[:, j].astype(np.longdouble)*np.longdouble(coefficient)))
        paths.append(dict(index=j, term=terms[j], success=True, coefficient=coefficient,
                          normalized_residual=residual,
                          extended_accumulator_normalized_residual=float(extended/np.longdouble(scale)),
                          lp_objective=float(solution.fun),
                          independently_validated=abs(residual-solution.fun) <= 1e-9,
                          coefficient_e2=float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth)),
                          matches_true_support=bool(np.array_equal(estimate != 0, truth != 0))))
    paths.sort(key=lambda r: r.get("normalized_residual", np.inf))
    successful = [r for r in paths if r["success"]]
    true = [r for r in successful if r["matches_true_support"]]
    best = successful[0]
    return dict(score_scale=scale, best_one_column=best, true_support_one_column=true,
                all_one_column_solutions=paths,
                interpretation=("For lambda/||b||_inf between the best one-column radius and 1, "
                                "minimum cardinality is one and the residual tie-break selects a "
                                "global best one-column fit. A strictly worse true-support radius "
                                "cannot win this tie-break at any such lambda."))


def inspect(path):
    metadata = json.loads(path.read_text())
    matrix_path = path.with_suffix(".npz")
    actual_hash = digest(matrix_path)
    if actual_hash != metadata["score_cache_sha256"]:
        raise ValueError(f"Matrix cache hash does not match metadata: {matrix_path}")
    with np.load(matrix_path) as archive:
        matrices = {key: archive[key] for key in archive.files}
    terms = metadata["terms"]
    truth = np.asarray(metadata["truth"])
    plugin_H = matrices["plugin_H"]
    # plugin_H = X(pilot).T X(pilot)/K, so these are physical weak column RMSes.
    rms = np.sqrt(np.maximum(np.diag(plugin_H), 0.))
    positive = rms[rms > 0]
    out = dict(name=metadata["name"], noise_ratio=metadata["noise_ratio"], trial=metadata["trial"],
               metadata_file=path.name, metadata_sha256=digest(path),
               matrix_file=matrix_path.name, matrix_sha256=actual_hash,
               preparation_hash=metadata["preparation_hash"], seed=metadata["seed"],
               pilot=metadata["pilot"],
               physical_weak_design=dict(column_rms_min=float(rms.min()), column_rms_max=float(rms.max()),
                                         positive_column_rms_ratio=float(positive.max()/positive.min()),
                                         smallest_column=terms[int(rms.argmin())],
                                         largest_column=terms[int(rms.argmax())],
                                         column_rms=dict(zip(terms, rms.tolist()))), scores={})
    for variant in ("plugin", "debiased"):
        H, B = matrices[variant+"_H"], matrices[variant+"_b"]
        score = dict(rank=metadata["score"][variant]["rank"],
                     physical_condition_number=metadata["score"][variant]["condition_number"],
                     equations=[])
        if metadata["noise_ratio"] == 0:
            # Congruence standardization is solely a conditioning diagnosis.
            diagonal = np.where(rms > 0, rms, 1.)
            standardized = H/diagonal[:, None]/diagonal[None, :]
            eig = np.linalg.eigvalsh(standardized)
            score["diagnostic_standardization"] = dict(
                definition="diag(plugin column RMS)^(-1) H diag(plugin column RMS)^(-1); diagnostic only",
                rank=int(np.linalg.matrix_rank(standardized)),
                condition_number=float(np.linalg.cond(standardized)),
                minimum_eigenvalue=float(eig.min()), maximum_eigenvalue=float(eig.max()),
                negative_eigenvalues_above_relative_1e_minus_10=int(np.sum(eig < -1e-10*np.max(np.abs(eig)))))
        for e in range(B.shape[1]):
            b, beta = B[:, e], truth[e]
            scale = float(np.max(np.abs(b)))
            residual = b-H@beta
            largest = np.argsort(np.abs(residual))[::-1][:5]
            true_indices = np.flatnonzero(beta)
            equation = dict(
                equation_index=e, score_scale=scale,
                truth_feasibility_ratio=float(np.max(np.abs(residual))/scale),
                largest_b_coordinate=terms[int(np.argmax(np.abs(b)))],
                largest_truth_residual_coordinates=[dict(term=terms[j], physical_residual=float(residual[j]),
                                                         normalized_residual=float(residual[j]/scale)) for j in largest],
                true_support_coordinate_diagnostics=[dict(term=terms[j], true_coefficient=float(beta[j]),
                                                          normalized_b=float(b[j]/scale),
                                                          normalized_truth_residual=float(residual[j]/scale))
                                                     for j in true_indices])
            if metadata["name"] == "IB" and metadata["noise_ratio"] == 0:
                equation["independent_exhaustive_support1_diagnosis"] = one_column_paths(H, b, beta, terms)
            score["equations"].append(equation)
        out["scores"][variant] = score
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path(__file__).resolve().parent/"prepared")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent/"diagnostics.json")
    args = parser.parse_args()
    paths = sorted(args.prepared.glob("*.json"))
    records = [inspect(path) for path in paths]
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["name"], record["noise_ratio"])].append(record)
    summaries = []
    for (name, noise), group in sorted(grouped.items()):
        summaries.append(dict(name=name, noise_ratio=noise, trials=len(group),
                              median_truth_feasibility_ratio={
                                  variant: float(np.median([e["truth_feasibility_ratio"] for r in group
                                                           for e in r["scores"][variant]["equations"]]))
                                  for variant in ("plugin", "debiased")}))
    payload = dict(
        schema_version=1,
        purpose="Read-only scientific and numerical diagnoses of prepared physical-unit scores",
        provenance=dict(prepared_directory=str(args.prepared.resolve()), cache_files=len(records),
                        numpy=version("numpy"), scipy=version("scipy"), script_sha256=digest(__file__)),
        interpretation=[
            "No states, pilots, or weak FFT columns were recomputed; all diagnoses use cached H, b, and metadata.",
            "True coefficients/support are used only for explicitly labeled evaluation diagnostics, never by the main estimator.",
            "The independent support1 LP uses only common score normalization and a coefficient substitution; it preserves the physical infinity-norm optimization.",
            "Standardized H spectra are diagnostic congruences only; no changed normalization is used in the main study.",
            "A tiny floating-point rank of the unscaled H does not prove algebraic rank deficiency: the tolerance depends on its largest singular value.",
            "The snapshot includes only cache JSON files available when this script ran; rerun after preparation to include later settings."],
        grouped_truth_feasibility=summaries, records=records)
    args.out.write_text(json.dumps(serializable(payload), indent=2)+"\n")
    print(f"Wrote {args.out}: {len(records)} prepared settings, independent IB support1 LP included")


if __name__ == "__main__":
    main()
