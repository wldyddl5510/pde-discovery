"""Independent exhaustive NLS two-column minimax LP diagnostic.

This does not change any saved Gurobi flag or the primary experiment.  Every
pair of dictionary columns is fitted without using true coefficients, using
the physical infinity norm.  Where possible, a three-row dual witness is
reconstructed with exact rational arithmetic on the saved binary64 H and b.
"""
from __future__ import annotations

import os
for _key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

warnings.filterwarnings("ignore", category=OptimizeWarning,
                        message="Unrecognized options detected:.*")
METHOD = "exact_plugin_dantzig"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational_witness(rows, j, k, H, b):
    """Return a feasible exact dual objective, or None.

    Rows index [H;-H].  Nonnegative weights with sum one and zero weighted
    column vector give ||b-H_pair beta||_infty >= -sum(weight*sign*b_row).
    """
    p = len(H)
    signed = [(row % p, 1 if row < p else -1) for row in rows]
    a = [H[row][j]*sign for row, sign in signed]
    c = [H[row][k]*sign for row, sign in signed]
    if len(rows) == 1:
        if a[0] or c[0]:
            return None
        weights = [Fraction(1)]
    elif len(rows) == 2:
        vector = a if a[0] != a[1] else c
        if vector[0] == vector[1]:
            return None
        weights = [-vector[1]/(vector[0]-vector[1]), vector[0]/(vector[0]-vector[1])]
        if sum(w*v for w, v in zip(weights, a)) or sum(w*v for w, v in zip(weights, c)):
            return None
    elif len(rows) == 3:
        cross = [a[1]*c[2]-a[2]*c[1], a[2]*c[0]-a[0]*c[2], a[0]*c[1]-a[1]*c[0]]
        total = sum(cross)
        if not total:
            return None
        weights = [v/total for v in cross]
    else:
        return None
    if any(w < 0 for w in weights):
        return None
    bound = -sum(w*sign*b[row] for w, (row, sign) in zip(weights, signed))
    return bound if bound >= 0 else None


def find_witness(result, j, k, H, b):
    marginal = np.asarray(result.ineqlin.marginals)
    support = np.flatnonzero(marginal < -1e-10).tolist()
    if len(support) <= 3:
        bound = rational_witness(support, j, k, H, b)
        if bound is not None:
            return bound, support
    # Rare near-degenerate solves can omit a numerically negligible multiplier.
    # A dual witness need not use active rows, so try nearby constraints too.
    candidates = list(dict.fromkeys(support+np.argsort(result.ineqlin.residual)[:10].tolist()))
    best = None
    for size in (1, 2, 3):
        for rows in combinations(candidates, size):
            bound = rational_witness(rows, j, k, H, b)
            if bound is not None and (best is None or bound > best[0]):
                best = bound, list(rows)
    return best


def run(directory, limit=None, threshold_ratio=.01):
    start = perf_counter()
    metadata_path = directory/"prepared/NLS_noise-0_trial-0.json"
    matrix_path = metadata_path.with_suffix(".npz")
    solution_path = directory/f"solved/NLS_noise-0_trial-0__{METHOD}.json"
    metadata = json.loads(metadata_path.read_text())
    solution = json.loads(solution_path.read_text())
    if digest(matrix_path) != metadata["score_cache_sha256"]:
        raise RuntimeError("Saved physical matrix cache hash mismatch")
    with np.load(matrix_path, allow_pickle=False) as archive:
        H, b = archive[METHOD+"_H"], archive[METHOD+"_b"]
    p, q = b.shape
    if H.shape != (p, p):
        raise ValueError("Expected a square physical score")
    b_inf = np.max(np.abs(b), axis=0)
    column_scale = np.max(np.abs(H), axis=0)
    column_scale[column_scale == 0] = 1.
    A, y = H/column_scale, b/b_inf
    Hf = [[Fraction.from_float(float(v)) for v in row] for row in H]
    bf = [[Fraction.from_float(float(v)) for v in b[:, eq]] for eq in range(q)]
    pairs = list(combinations(range(p), 2))
    planned = len(pairs)
    if limit is not None:
        pairs = pairs[:limit]
    n = len(pairs)
    lp_value = np.full((q, n), np.nan)
    primal_residual = np.full((q, n), np.nan)
    numerical_dual = np.full((q, n), np.nan)
    exact_dual = np.full((q, n), np.nan)
    exact_rows = np.full((q, n, 3), -1, dtype=np.int32)
    exact_excludes_threshold = np.zeros((q, n), dtype=bool)
    coefficients = np.full((q, n, 2), np.nan)
    minimum_exact_bound = [None]*q
    max_primal_violation = max_dual_stationarity = max_dual_mass_excess = 0.
    failures = []
    completed = 0
    for index, (j, k) in enumerate(pairs):
        B = A[:, [j, k]]
        constraints = np.vstack([np.column_stack([B, -np.ones(p)]),
                                 np.column_stack([-B, -np.ones(p)])])
        for eq in range(q):
            result = linprog([0., 0., 1.], A_ub=constraints,
                b_ub=np.r_[y[:, eq], -y[:, eq]], bounds=[(None, None), (None, None), (0, None)],
                method="highs", options=dict(threads=1, primal_feasibility_tolerance=1e-9,
                    dual_feasibility_tolerance=1e-9, small_matrix_value=1e-12,
                    simplex_scale_strategy=0))
            if not result.success:
                failures.append(dict(pair=[j, k], equation=eq, status=result.status, message=result.message))
                continue
            beta = result.x[:2]*b_inf[eq]/column_scale[[j, k]]
            residual = float(np.max(np.abs(b[:, eq]-H[:, [j, k]]@beta))/b_inf[eq])
            dual = np.asarray(result.ineqlin.marginals)
            lower = float(np.r_[y[:, eq], -y[:, eq]]@dual)
            stationarity = float(np.max(np.abs(constraints[:, :2].T@dual)))
            mass_excess = max(0., float(-np.sum(dual)-1))
            violation = max(0., residual-float(result.fun))
            max_primal_violation = max(max_primal_violation, violation)
            max_dual_stationarity = max(max_dual_stationarity, stationarity)
            max_dual_mass_excess = max(max_dual_mass_excess, mass_excess)
            if violation > 1e-7 or stationarity > 1e-7 or mass_excess > 1e-7:
                raise RuntimeError(f"LP physical or dual validation failed: pair {(j,k)}, eq {eq}")
            coefficients[eq, index] = beta
            lp_value[eq, index], primal_residual[eq, index], numerical_dual[eq, index] = result.fun, residual, lower
            witness = find_witness(result, j, k, Hf, bf[eq])
            if witness is not None:
                bound, rows = witness
                exact_dual[eq, index] = float(bound)/b_inf[eq]
                exact_rows[eq, index, :len(rows)] = rows
                physical_threshold = Fraction.from_float(float(threshold_ratio*b_inf[eq]))
                exact_excludes_threshold[eq, index] = bound > physical_threshold
                if minimum_exact_bound[eq] is None or bound < minimum_exact_bound[eq]:
                    minimum_exact_bound[eq] = bound
            completed += 1
        if (index+1) % 1000 == 0:
            print(f"[pairs {index+1}/{n}] solved={completed}/{q*n}, "
                  f"exact threshold exclusions={int(exact_excludes_threshold[:, :index+1].sum())}, "
                  f"elapsed={perf_counter()-start:.1f}s", flush=True)
    archive_path = directory/"nls_pairs.npz"
    np.savez_compressed(archive_path, pairs=np.asarray(pairs, dtype=np.int32),
        lp_value_ratio=lp_value, physical_residual_ratio=primal_residual,
        numerical_dual_ratio=numerical_dual, exact_dual_ratio=exact_dual,
        exact_dual_rows=exact_rows, exact_excludes_threshold=exact_excludes_threshold,
        coefficients=coefficients)
    equations = []
    truth = np.asarray(metadata["truth"])
    for eq in range(q):
        best = int(np.nanargmin(primal_residual[eq]))
        j, k = pairs[best]
        candidate_rows = []
        for candidate in solution["result"]["paths"][eq]["path"]:
            beta = np.asarray(candidate["coefficients"])
            support_size = int(np.count_nonzero(beta))
            if not candidate.get("validated") or support_size != 3:
                continue
            actual_residual = float(np.max(np.abs(b[:, eq]-H@beta)))
            support = np.flatnonzero(beta).tolist()
            beta_fraction = [Fraction.from_float(float(beta[j])) for j in support]
            exact_residual = max(abs(bf[eq][row]-sum(Hf[row][j]*v for j,v in zip(support,beta_fraction)))
                                 for row in range(p))
            exact_lambda = Fraction.from_float(float(candidate["lambda"]))
            physical_feasible = exact_residual <= exact_lambda
            all_witnesses = bool(np.isfinite(exact_dual[eq]).all())
            sparse_exclusion = (physical_feasible and n == planned and all_witnesses
                and minimum_exact_bound[eq] is not None and minimum_exact_bound[eq] > exact_lambda)
            candidate_rows.append(dict(lambda_ratio=candidate["lambda_ratio"], lam=candidate["lambda"],
                physical_residual=actual_residual, e2=candidate["e2"],
                coefficients=[dict(index=j,term=metadata["terms"][j],value=float(beta[j])) for j in support],
                exact_physical_residual=float(exact_residual),
                exact_physical_residual_fraction=str(exact_residual),
                physical_feasible=bool(physical_feasible),
                minimum_dual_bound_over_lambda=float(minimum_exact_bound[eq]/exact_lambda),
                cardinality_three_independently_supported=bool(sparse_exclusion),
                original_cardinality_optimal=candidate.get("cardinality_optimal", False)))
        equations.append(dict(equation=eq, label=metadata["equations"][eq],
            minimum_two_column_residual_ratio=float(primal_residual[eq, best]),
            minimum_two_column_residual=float(primal_residual[eq, best]*b_inf[eq]),
            best_pair=[j,k], best_pair_labels=[metadata["terms"][j],metadata["terms"][k]],
            best_pair_coefficients=coefficients[eq,best].tolist(),
            minimum_exact_dual_ratio=float(np.nanmin(exact_dual[eq])),
            minimum_exact_dual_physical=float(minimum_exact_bound[eq]),
            minimum_exact_dual_fraction=str(minimum_exact_bound[eq]),
            minimum_exact_dual_over_threshold=float(minimum_exact_bound[eq])/(threshold_ratio*b_inf[eq]),
            exact_witness_count=int(np.isfinite(exact_dual[eq]).sum()),
            exact_threshold_exclusions=int(exact_excludes_threshold[eq].sum()),
            all_pairs_exactly_excluded_at_threshold=bool(exact_excludes_threshold[eq].all() and n == planned),
            truth_feasibility_ratio=float(np.max(np.abs(b[:,eq]-H@truth[eq]))/b_inf[eq]),
            saved_three_term_candidates=candidate_rows))
    payload = dict(diagnostic="exhaustive-two-column-minimax-NLS-exact-plugin-noiseless",
        planned_pairs=planned, completed_pairs=n, equations=q, solved_LPs=completed,
        complete=bool(n == planned and not failures), method=METHOD,
        threshold_ratio=threshold_ratio, runtime_seconds=perf_counter()-start, failures=failures,
        max_normalized_primal_violation=max_primal_violation,
        max_normalized_dual_stationarity=max_dual_stationarity,
        max_dual_mass_excess=max_dual_mass_excess,
        provenance=dict(metadata_file=metadata_path.name, metadata_sha256=digest(metadata_path),
            score_file=matrix_path.name, score_sha256=digest(matrix_path),
            saved_solution=solution_path.name, saved_solution_sha256=digest(solution_path),
            diagnostic_source_sha256=digest(Path(__file__).resolve()),
            output_archive_sha256=digest(archive_path)),
        scaling="LP uses H[:,j]/max|H[:,j]|, b/||b||inf and transforms coefficients back; physical norm unchanged",
        exact_certificate="Nonnegative rational weights sum to 1 and annihilate both raw binary64 H columns exactly; "
            "-sum(weights*sign*b) is a dual lower bound. Row indices refer to [H;-H], -1 is padding.",
        interpretation="All supports of size <=2 are contained in a column pair. Excluding every pair below a lambda "
            "and exhibiting a feasible 3-term vector establishes minimum cardinality 3 for the saved floating score. "
            "This separate diagnostic never edits Gurobi flags and does not certify the residual tie-break.",
        results=equations)
    (directory/"nls_pairs.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    lines = ["# NLS exact-pilot two-column diagnostic", "",
        f"Scanned {n:,}/{planned:,} column pairs for each of {q} equations ({completed:,} LPs), "
        f"using one HiGHS thread; runtime {payload['runtime_seconds']:.1f} s.", "",
        "The LP fits every support without true coefficients. Column and response scaling only changes coordinates; "
        "residuals are checked again in the saved physical score. True coefficients enter only the final error comparison.", "",
        "| Equation | Best two-column residual / ‖b‖∞ | Smallest exact dual bound / ‖b‖∞ | Exact exclusions at λ / ‖b‖∞ = .01 |",
        "| :--- | ---: | ---: | ---: |"]
    for row in equations:
        lines.append(f"| {row['label']} | {row['minimum_two_column_residual_ratio']:.9g} | "
                     f"{row['minimum_exact_dual_ratio']:.9g} | {row['exact_threshold_exclusions']:,}/{planned:,} |")
    lines += ["", "For each row witness, exact rational nonnegative weights sum to one and annihilate both physical "
        "columns. Their signed response sum is a rigorous lower bound for the saved binary64 score. The NPZ stores "
        "the signed row IDs so every witness can be reconstructed from the hashed score cache.", ""]
    for row in equations:
        good = [c for c in row["saved_three_term_candidates"]
                if c["physical_feasible"] and c["cardinality_three_independently_supported"]]
        if good:
            best = min(good, key=lambda c:c["e2"])
            lines.append(f"- Equation {row['equation']}: a saved feasible three-term candidate has relative E₂ "
                f"{best['e2']:.9g}. All supports of size ≤2 are independently excluded for its λ "
                f"(dual lower bound / λ = {best['minimum_dual_bound_over_lambda']:.6g}). "
                f"Thus its cardinality is 3 for this floating score. The primal residual is also checked with "
                f"exact rational arithmetic: {best['exact_physical_residual']:.9g} ≤ λ={best['lam']:.9g}.")
            lines.append("  Coefficients: "+", ".join(f"{v['term']} = {v['value']:.12g}" for v in best["coefficients"])+".")
        else:
            lines.append(f"- Equation {row['equation']}: this scan does not establish a cardinality-3 certificate; "
                         "inspect the exact witness counts and candidate thresholds.")
    lines += ["", "This is an independent optimization diagnostic. The main experiment and its Gurobi certificate "
        "flags remain unchanged. Residual tie-break optimality is not established here; the result applies to the saved "
        "finite-grid candidates and score, rather than a continuous-λ oracle or a theorem for the population score.", ""]
    (directory/"nls_pairs.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold-ratio", type=float, default=.01)
    args = parser.parse_args()
    run(args.directory.resolve(), args.limit, args.threshold_ratio)
