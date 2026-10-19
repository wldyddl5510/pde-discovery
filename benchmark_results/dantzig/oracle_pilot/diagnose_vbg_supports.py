"""Independent VBG four-column minimax support scan and exact dual witnesses.

This diagnostic never changes the primary study or its solver flags.  Every
four-column support is fitted without true coefficients, in a physical-
equivalent infinity norm.  Integer cofactors recover exact rational dual
witnesses for the original saved binary64 score, using at most five signed rows.
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
from math import comb
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


def determinant(matrix):
    """Exact integer determinant by fraction-free Bareiss elimination."""
    n = len(matrix)
    if not n:
        return 1
    a, previous, sign = [list(row) for row in matrix], 1, 1
    for k in range(n-1):
        if a[k][k] == 0:
            swap = next((i for i in range(k+1, n) if a[i][k]), None)
            if swap is None:
                return 0
            a[k], a[swap], sign = a[swap], a[k], -sign
        pivot = a[k][k]
        for i in range(k+1, n):
            for j in range(k+1, n):
                numerator = a[i][j]*pivot-a[i][k]*a[k][j]
                value, remainder = divmod(numerator, previous)
                if remainder:
                    raise ArithmeticError("Bareiss exact division failed")
                a[i][j] = value
            a[i][k] = 0
        previous = pivot
    return sign*a[-1][-1]


def exact_witness(rows, support, H, b):
    """Return a feasible exact dual objective, or None.

    H and b store binary64 integer ratios.  Rows index [H;-H].  Clear each
    equation's power-of-two denominators, compute an integer null vector,
    require nonnegative weights summing to one, and bound the physical residual.
    """
    p, size = len(H), len(rows)
    if size < 1 or size > len(support)+1:
        return None
    signed = [(row % p, 1 if row < p else -1) for row in rows]
    equations = []
    for j in support:
        ratios = [(H[row][j][0]*sign, H[row][j][1]) for row, sign in signed]
        common = max(denominator for _, denominator in ratios)
        equations.append([numerator*(common//denominator) for numerator, denominator in ratios])
    for selected in combinations(range(len(support)), size-1):
        square = [equations[i] for i in selected]
        vector = [(-1 if column % 2 else 1)*determinant(
            [[entry for index, entry in enumerate(row) if index != column] for row in square])
            for column in range(size)]
        total = sum(vector)
        if not total:
            continue
        if total < 0:
            vector, total = [-value for value in vector], -total
        if any(value < 0 for value in vector):
            continue
        if any(sum(a*v for a,v in zip(equation, vector)) for equation in equations):
            continue
        common = max(b[row][1] for row,_ in signed)
        numerator = -sum(v*sign*b[row][0]*(common//b[row][1])
                         for v,(row,sign) in zip(vector,signed))
        bound = Fraction(numerator, total*common)
        if bound >= 0:
            return bound
    return None


def find_witness(result, support, H, b):
    marginal = np.asarray(result.ineqlin.marginals)
    rows = np.flatnonzero(marginal < -1e-10).tolist()
    bound = exact_witness(rows, support, H, b)
    if bound is not None:
        return bound, rows
    # Nearby active rows repair rare numerical degeneracy in the LP basis.
    candidates = list(dict.fromkeys(rows+np.argsort(result.ineqlin.residual)[:10].tolist()))
    best = None
    for signed_rows in combinations(candidates, len(support)+1):
        bound = exact_witness(signed_rows, support, H, b)
        if bound is not None and (best is None or bound > best[0]):
            best = bound, list(signed_rows)
    return best


def exact_primal_residual(H, b, support, beta):
    values = [Fraction.from_float(float(value)) for value in beta]
    return max(abs(Fraction(*response)-sum(Fraction(*row[j])*value for j,value in zip(support,values)))
               for row,response in zip(H,b))


def run(directory, limit=None, threshold_ratio=1e-6):
    start = perf_counter()
    metadata_path = directory/"prepared/VBG_noise-0_trial-0.json"
    matrix_path = metadata_path.with_suffix(".npz")
    solution_path = directory/f"solved/VBG_noise-0_trial-0__{METHOD}.json"
    metadata, solution = json.loads(metadata_path.read_text()), json.loads(solution_path.read_text())
    if digest(matrix_path) != metadata["score_cache_sha256"]:
        raise RuntimeError("Saved physical matrix cache hash mismatch")
    with np.load(matrix_path, allow_pickle=False) as archive:
        H, b = archive[METHOD+"_H"], archive[METHOD+"_b"][:,0]
    p = len(b)
    if H.shape != (p,p):
        raise ValueError("Expected a square scalar physical score")
    b_inf = float(np.max(np.abs(b)))
    column_scale = np.max(np.abs(H), axis=0)
    column_scale[column_scale == 0] = 1.
    A, y = H/column_scale, b/b_inf
    Hr = [[float(value).as_integer_ratio() for value in row] for row in H]
    br = [float(value).as_integer_ratio() for value in b]
    planned = comb(p,4)
    supports = list(combinations(range(p),4))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        supports = supports[:limit]
    target = Fraction.from_float(float(threshold_ratio*b_inf))
    lp_value, physical_residual, dual_ratio, witnesses, excludes, coefficient_rows = [],[],[],[],[],[]
    minimum_exact_bound, minimum_support = None,None
    failures, counterexample = [],None
    max_primal_violation = max_dual_stationarity = max_dual_mass_excess = 0.
    visited = []
    for index,support in enumerate(supports):
        B = A[:,support]
        constraints = np.vstack([np.column_stack([B,-np.ones(p)]),
                                  np.column_stack([-B,-np.ones(p)])])
        result = linprog([0.,0.,0.,0.,1.], A_ub=constraints, b_ub=np.r_[y,-y],
            bounds=[(None,None)]*4+[(0,None)], method="highs",
            options=dict(threads=1,primal_feasibility_tolerance=1e-9,
                         dual_feasibility_tolerance=1e-9,small_matrix_value=1e-12,simplex_scale_strategy=0))
        visited.append(support)
        if not result.success:
            failures.append(dict(support=list(support),status=result.status,message=result.message))
            lp_value.append(np.nan);physical_residual.append(np.nan);dual_ratio.append(np.nan)
            witnesses.append([-1]*5);excludes.append(False);coefficient_rows.append([np.nan]*4)
            continue
        beta = result.x[:4]*b_inf/column_scale[list(support)]
        residual = float(np.max(np.abs(b-H[:,support]@beta))/b_inf)
        dual = result.ineqlin.marginals
        max_primal_violation = max(max_primal_violation,max(0.,residual-float(result.fun)))
        max_dual_stationarity = max(max_dual_stationarity,float(np.max(np.abs(constraints[:,:4].T@dual))))
        max_dual_mass_excess = max(max_dual_mass_excess,max(0.,float(-np.sum(dual)-1)))
        witness = find_witness(result,support,Hr,br)
        bound,rows = witness if witness is not None else (None,[])
        lp_value.append(float(result.fun));physical_residual.append(residual)
        coefficient_rows.append(beta.tolist());dual_ratio.append(float(bound)/b_inf if bound is not None else np.nan)
        witnesses.append(list(rows)+[-1]*(5-len(rows)))
        excludes.append(bool(bound is not None and bound > target))
        if bound is not None and (minimum_exact_bound is None or bound < minimum_exact_bound):
            minimum_exact_bound,minimum_support = bound,support
        # A feasible four-term counterexample at the requested target is enough
        # to stop this proposed five-term proof; verify it exactly before stopping.
        if residual <= threshold_ratio:
            exact_residual = exact_primal_residual(Hr,br,support,beta)
            if exact_residual <= target:
                counterexample = dict(support=list(support),labels=[metadata["terms"][j] for j in support],
                    coefficients=beta.tolist(),residual_ratio=residual,
                    exact_physical_residual=str(exact_residual),threshold=str(target))
                print(f"Exact four-term counterexample found after {index+1} supports; stopping",flush=True)
                break
        if (index+1) % 10000 == 0:
            print(f"[supports {index+1}/{len(supports)}] exact exclusions={sum(excludes)}, "
                  f"min ratio={min(physical_residual):.8g}, elapsed={perf_counter()-start:.1f}s",flush=True)
    n = len(visited)
    complete = bool(n == planned and not failures and counterexample is None)
    all_exact = bool(complete and np.isfinite(dual_ratio).all())
    archive_path = directory/"vbg_supports.npz"
    np.savez_compressed(archive_path,supports=np.asarray(visited,dtype=np.int32),
        lp_value_ratio=lp_value,physical_residual_ratio=physical_residual,
        exact_dual_ratio=dual_ratio,exact_dual_signed_rows=np.asarray(witnesses,dtype=np.int32),
        exact_excludes_threshold=np.asarray(excludes,dtype=bool),coefficients=coefficient_rows)
    best_index = int(np.nanargmin(physical_residual))
    best_support = visited[best_index]
    saved_candidates = []
    truth = np.asarray(metadata["truth"])[0]
    for candidate in solution["result"]["paths"][0]["path"]:
        beta = np.asarray(candidate["coefficients"])
        support = np.flatnonzero(beta).tolist()
        if not candidate.get("validated") or len(support) != 5:
            continue
        exact_residual = exact_primal_residual(Hr,br,support,beta[support])
        lam = Fraction.from_float(float(candidate["lambda"]))
        feasible = exact_residual <= lam
        minimum_cardinality = 5 if all_exact and feasible and minimum_exact_bound > lam else None
        saved_candidates.append(dict(lambda_ratio=candidate["lambda_ratio"],lam=candidate["lambda"],
            e2=float(np.linalg.norm(beta-truth)/np.linalg.norm(truth)),
            coefficients=[dict(index=j,term=metadata["terms"][j],value=float(beta[j])) for j in support],
            exact_physical_residual=float(exact_residual),exact_physical_residual_fraction=str(exact_residual),
            physical_feasible=bool(feasible),independently_certified_minimum_cardinality=minimum_cardinality,
            minimum_dual_bound_over_lambda=float(minimum_exact_bound/lam) if minimum_exact_bound is not None else None,
            original_cardinality_optimal=candidate.get("cardinality_optimal",False)))
    payload = dict(diagnostic="exhaustive-four-column-minimax-VBG-exact-plugin-noiseless",
        planned_supports=planned,visited_supports=n,complete=complete,
        solved_LPs=n-len(failures),method=METHOD,target_ratio=threshold_ratio,
        runtime_seconds=perf_counter()-start,failures=failures,counterexample=counterexample,
        exact_witnesses=int(np.isfinite(dual_ratio).sum()),exact_threshold_exclusions=int(np.sum(excludes)),
        all_supports_exactly_excluded_at_target=bool(complete and all(excludes)),
        minimum_four_column_residual_ratio=float(physical_residual[best_index]),
        best_support=list(best_support),best_support_labels=[metadata["terms"][j] for j in best_support],
        best_support_coefficients=coefficient_rows[best_index],
        minimum_exact_dual_ratio=float(minimum_exact_bound)/b_inf if minimum_exact_bound is not None else None,
        minimum_exact_dual_physical=float(minimum_exact_bound) if minimum_exact_bound is not None else None,
        minimum_exact_dual_fraction=str(minimum_exact_bound),
        minimum_exact_dual_support=list(minimum_support) if minimum_support is not None else None,
        minimum_exact_dual_over_target=float(minimum_exact_bound/target) if minimum_exact_bound is not None else None,
        max_normalized_primal_violation=max_primal_violation,
        max_normalized_dual_stationarity=max_dual_stationarity,max_dual_mass_excess=max_dual_mass_excess,
        saved_five_term_candidates=saved_candidates,
        provenance=dict(metadata_file=metadata_path.name,metadata_sha256=digest(metadata_path),
            score_file=matrix_path.name,score_sha256=digest(matrix_path),solution_file=solution_path.name,
            solution_sha256=digest(solution_path),diagnostic_source_sha256=digest(Path(__file__).resolve()),
            output_archive_sha256=digest(archive_path)),
        exact_certificate="Binary64 H,b are interpreted as exact integer ratios. Clear equation denominators, "
            "use signed integer minors to find nonnegative dual weights summing to one that annihilate each "
            "original physical support column. Their signed response is an exact residual lower bound.",
        interpretation="Every support of size<=4 is contained in a four-column support. Excluding every such support "
            "below lambda and exhibiting an exactly feasible five-term vector certifies minimum cardinality5 "
            "for the saved floating score. This diagnostic never changes primary Gurobi flags or proves a residual tie-break.")
    (directory/"vbg_supports.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    lines=["# VBG exact-pilot four-column diagnostic","",
        f"Scanned {n:,}/{planned:,} supports in {payload['runtime_seconds']:.1f}s with one HiGHS thread. "
        f"Exact physical dual witnesses: {payload['exact_witnesses']:,}; exact exclusions at "
        f"λ/‖b‖∞={threshold_ratio:g}: {payload['exact_threshold_exclusions']:,}.","",
        f"Smallest numerical four-column residual / ‖b‖∞: {payload['minimum_four_column_residual_ratio']:.12g}. "
        f"Smallest exact dual bound / ‖b‖∞: {payload['minimum_exact_dual_ratio']:.12g}.","",
        "Every witness uses at most five signed rows of the original physical H,b. Integer cofactors produce "
        "nonnegative exact rational weights summing to one and annihilating all four columns. The NPZ stores "
        "signed row IDs for independent reconstruction. Every five-term primal candidate is also checked exactly.","",
        "| Saved λ / ‖b‖∞ | Relative E₂ | Exact dual lower bound / λ | Independently minimum cardinality 5 |",
        "| ---: | ---: | ---: | :--- |"]
    for row in saved_candidates:
        lines.append(f"| {row['lambda_ratio']:.9g} | {row['e2']:.9g} | "
            f"{row['minimum_dual_bound_over_lambda']:.9g} | {row['independently_certified_minimum_cardinality']==5} |")
    if counterexample:
        lines += ["","An exactly feasible four-term counterexample was found at the requested target; the exhaustive "
                  "scan stopped as intended. This does not certify a five-term minimum."]
    good=[r for r in saved_candidates if r["independently_certified_minimum_cardinality"]==5]
    if good:
        best=min(good,key=lambda row:row["e2"])
        lines += ["","An independently cardinality-optimal saved five-term vector has coefficients:",
            "",*["- "+v["term"]+f" = {v['value']:.14g}" for v in best["coefficients"]],"",
            f"Exact physical residual {best['exact_physical_residual']:.12g} ≤ λ={best['lam']:.12g}. "
            f"Relative coefficient error is {best['e2']:.12g}."]
    lines += ["","The scan uses no true coefficient or support information. True coefficients are used only after "
        "the scan to evaluate saved candidates. Primary study files and Gurobi flags remain unchanged. "
        "This proves cardinality only for the saved floating score, not residual tie-break optimality or a continuous-λ oracle.",""]
    (directory/"vbg_supports.md").write_text("\n".join(lines))
    print(json.dumps(payload,indent=2),flush=True)
    return payload


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--threshold-ratio",type=float,default=1e-6)
    args=parser.parse_args()
    run(args.directory.resolve(),args.limit,args.threshold_ratio)
