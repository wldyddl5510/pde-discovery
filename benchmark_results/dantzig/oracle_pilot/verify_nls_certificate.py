"""Verify saved NLS sparsity certificates with exact arithmetic and no LP solves.

Unlike the certificate generator's cross-product construction, this verifier
recovers dual weights with rational Gaussian elimination. Every stated bound
uses the physical saved binary64 H, b, beta, and lambda, viewed as exact rationals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational_weights(columns):
    """Solve [1; signed H_j; signed H_k] w = [1; 0; 0] by RREF."""
    width = len(columns)
    matrix = [[Fraction(1) for _ in columns]+[Fraction(1)],
              [column[0] for column in columns]+[Fraction(0)],
              [column[1] for column in columns]+[Fraction(0)]]
    pivot_row = 0
    pivots = []
    for column in range(width):
        pivot = next((row for row in range(pivot_row,3) if matrix[row][column]),None)
        if pivot is None:
            continue
        matrix[pivot_row],matrix[pivot] = matrix[pivot],matrix[pivot_row]
        divisor = matrix[pivot_row][column]
        matrix[pivot_row] = [value/divisor for value in matrix[pivot_row]]
        for row in range(3):
            if row != pivot_row and matrix[row][column]:
                multiple = matrix[row][column]
                matrix[row] = [value-multiple*base for value,base in zip(matrix[row],matrix[pivot_row])]
        pivots.append((pivot_row,column))
        pivot_row += 1
        if pivot_row == 3:
            break
    if any(not any(row[:width]) and row[width] for row in matrix):
        raise ValueError("Saved rows have no exact dual stationarity solution")
    weights = [Fraction(0) for _ in columns]
    for row,column in pivots:
        weights[column] = matrix[row][width]
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    directory = args.directory.resolve()
    start = perf_counter()
    diagnostic_path = directory/"nls_pairs.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    provenance = diagnostic["provenance"]
    metadata_path = directory/"prepared"/provenance["metadata_file"]
    matrix_path = directory/"prepared"/provenance["score_file"]
    solution_path = directory/"solved"/provenance["saved_solution"]
    archive_path = directory/"nls_pairs.npz"
    failures,counts = [],Counter()

    def check(condition,label,**context):
        counts["assertions"] += 1
        if not condition:
            failures.append(dict(check=label,**context))

    for path,sha in ((metadata_path,provenance["metadata_sha256"]),(matrix_path,provenance["score_sha256"]),
                     (solution_path,provenance["saved_solution_sha256"]),(archive_path,provenance["output_archive_sha256"]),
                     (directory/"diagnose_nls_pairs.py",provenance["diagnostic_source_sha256"])):
        check(digest(path)==sha,"certificate input provenance hash",path=path.name)
    metadata = json.loads(metadata_path.read_text())
    solution = json.loads(solution_path.read_text())
    method = diagnostic["method"]
    with np.load(matrix_path,allow_pickle=False) as archive:
        H,B = archive[method+"_H"],archive[method+"_b"]
    with np.load(archive_path,allow_pickle=False) as archive:
        pairs = archive["pairs"]
        rows_saved = archive["exact_dual_rows"]
        ratio_saved = archive["exact_dual_ratio"]
        excluded_saved = archive["exact_excludes_threshold"]
    p,q = B.shape
    check(H.shape==(p,p) and H.dtype==np.float64 and B.dtype==np.float64,"physical binary64 score dimensions")
    expected_pairs = np.asarray(list(combinations(range(p),2)),dtype=int)
    check(np.array_equal(pairs,expected_pairs),"all unordered pairs exactly once, covering every support of size <=2")
    check(rows_saved.shape==(q,len(expected_pairs),3),"one saved dual witness per pair per equation")
    check(diagnostic["complete"] and diagnostic["completed_pairs"]==len(expected_pairs),"exhaustive diagnostic complete flag")
    Hf = [[Fraction.from_float(float(value)) for value in row] for row in H]
    Bf = [[Fraction.from_float(float(value)) for value in B[:,equation]] for equation in range(q)]
    equations = []
    for equation in range(q):
        scale = float(np.max(np.abs(B[:,equation])))
        threshold_float = float(diagnostic["threshold_ratio"]*scale)
        threshold = Fraction.from_float(threshold_float)
        minimum = None
        minimum_pair = None
        for index,(j,k) in enumerate(pairs):
            context = dict(equation=equation,pair=[int(j),int(k)])
            stored = rows_saved[equation,index].tolist()
            identifiers = [row for row in stored if row >= 0]
            check(1<=len(identifiers)<=3 and len(set(identifiers))==len(identifiers)
                  and all(0<=row<2*p for row in identifiers),"valid signed witness row IDs",**context)
            signed = [(row%p,1 if row<p else -1) for row in identifiers]
            columns = [(sign*Hf[row][j],sign*Hf[row][k]) for row,sign in signed]
            weights = rational_weights(columns)
            check(all(weight>=0 for weight in weights) and sum(weights)==1,"exact nonnegative dual weights with unit mass",**context)
            check(sum(weight*column[0] for weight,column in zip(weights,columns))==0
                  and sum(weight*column[1] for weight,column in zip(weights,columns))==0,
                  "exact physical dual stationarity",**context)
            # Form the actual dual vector, including possible cancellation of
            # opposite signed copies of one score row. Its l1 norm is <=1.
            dual = {}
            for weight,(row,sign) in zip(weights,signed):
                dual[row] = dual.get(row,Fraction(0))-weight*sign
            check(sum(abs(value) for value in dual.values())<=1,"exact physical dual l1 norm <=1",**context)
            bound = sum(value*Bf[equation][row] for row,value in dual.items())
            check(bound>threshold,"exact pair exclusion above stored physical lambda",**context)
            check(bool(excluded_saved[equation,index]),"saved exclusion flag matches exact proof",**context)
            check(np.isclose(ratio_saved[equation,index],float(bound)/scale,rtol=2e-14,atol=0),"saved dual ratio matches exact reconstruction",**context)
            if minimum is None or bound<minimum:
                minimum,minimum_pair = bound,[int(j),int(k)]
            counts["exact_dual_witnesses_verified"] += 1
            if (index+1)%5000==0:
                print(f"[NLS exact duals] equation={equation},pairs={index+1}/{len(pairs)},elapsed={perf_counter()-start:.1f}s",flush=True)
        published = diagnostic["results"][equation]
        check(minimum==Fraction(published["minimum_exact_dual_fraction"]),"reported minimum exact lower bound",equation=equation)
        candidates = []
        for saved in published["saved_three_term_candidates"]:
            support = [item["index"] for item in saved["coefficients"]]
            values = [Fraction.from_float(float(item["value"])) for item in saved["coefficients"]]
            check(len(support)==len(set(support))==3 and all(value!=0 for value in values),"exact primal support cardinality three",equation=equation,lambda_ratio=saved["lambda_ratio"])
            coefficients = np.zeros(p)
            for item in saved["coefficients"]:
                coefficients[item["index"]] = item["value"]
                check(item["term"]==metadata["terms"][item["index"]],"saved primal dictionary label",equation=equation)
            source = [candidate for candidate in solution["result"]["paths"][equation]["path"]
                      if candidate["lambda"]==saved["lam"] and candidate["coefficients"]==coefficients.tolist()]
            check(len(source)==1,"primal coefficients and lambda are original saved candidate",equation=equation,lambda_ratio=saved["lambda_ratio"])
            residual = max(abs(Bf[equation][row]-sum(Hf[row][j]*value for j,value in zip(support,values))) for row in range(p))
            lam = Fraction.from_float(float(saved["lam"]))
            feasible = residual<=lam
            certified = feasible and minimum>lam
            check(feasible==saved["physical_feasible"],"exact primal feasibility metadata",equation=equation,lambda_ratio=saved["lambda_ratio"])
            check(residual==Fraction(saved["exact_physical_residual_fraction"]),"exact primal residual fraction",equation=equation,lambda_ratio=saved["lambda_ratio"])
            check(certified==saved["cardinality_three_independently_supported"],"primal plus exhaustive dual exclusion certifies cardinality",equation=equation,lambda_ratio=saved["lambda_ratio"])
            target = np.asarray(metadata["truth"])[equation]
            e2 = float(np.linalg.norm(coefficients-target)/np.linalg.norm(target))
            check(np.isclose(e2,saved["e2"],rtol=2e-13,atol=0),"post-proof coefficient error",equation=equation,lambda_ratio=saved["lambda_ratio"])
            candidates.append(dict(lambda_ratio=saved["lambda_ratio"],physical_lambda=str(lam),
                exact_residual=str(residual),physical_feasible=feasible,minimum_cardinality_three_proved=certified,
                coefficient_e2=e2,original_cardinality_flag=source[0]["cardinality_optimal"] if source else None,
                original_secondary_flag=source[0]["secondary_optimal"] if source else None))
            counts["exact_three_term_primals_verified"] += 1
            counts["minimum_cardinality_three_certificates"] += int(certified)
        threshold_candidates = [candidate for candidate in candidates if candidate["lambda_ratio"]==diagnostic["threshold_ratio"]]
        check(len(threshold_candidates)==1 and threshold_candidates[0]["minimum_cardinality_three_proved"],"cardinality three established at stated .01 score ratio",equation=equation)
        equations.append(dict(equation=equation,pairs_verified=len(pairs),physical_lambda_at_threshold=str(threshold),
            minimum_pair_dual_bound=str(minimum),minimum_pair=minimum_pair,
            minimum_dual_bound_over_threshold=float(minimum/threshold),all_supports_size_at_most_two_excluded=True,
            saved_three_term_candidates=candidates))
    report = dict(protocol="independent-exact-NLS-certificate-verification-v1",passed=not failures,
        created_at_utc=datetime.now(timezone.utc).isoformat(),runtime_seconds=perf_counter()-start,
        equations=q,dictionary_columns=p,pairs_per_equation=len(pairs),counts=dict(counts),failures=failures,results=equations,
        provenance=dict(verifier_source_sha256=digest(__file__),diagnostic_json_sha256=digest(diagnostic_path),
            certificate_archive_sha256=digest(archive_path),physical_score_archive_sha256=digest(matrix_path),
            original_saved_solution_sha256=digest(solution_path)),
        proof="All physical binary64 values are exact Fractions. Gaussian elimination reconstructs dual weights; exact stationarity and l1<=1 imply residual>=dual objective. Every pair's bound exceeds lambda; a feasible nonzero three-term primal proves minimum cardinality 3.",
        limitations=["Does not certify the residual tie-break, alter any main solver flag, or rerun a linear program.",
                     "The proof concerns the saved finite-grid binary64 score and physical lambda, rather than exact continuum weak integrals."])
    json_path = directory/"nls_certificate_verification.json"
    json_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    lines = ["# Independent NLS certificate verification","",
        f"Passed: **{report['passed']}**. Verified {counts['exact_dual_witnesses_verified']:,} exact-rational dual witnesses, covering every one of {len(pairs):,} pairs in each equation. No LP or MILP was rerun.","",
        "The verifier independently reconstructs weights with rational Gaussian elimination. It checks exact physical H stationarity, nonnegative unit mass, the actual dual vector's l1 norm ≤ 1, and an exact dual objective strictly above the stored physical lambda.","",
        "| Equation | Verified pair exclusions | Minimum bound / lambda at .01 | Saved feasible three-term E2 |",
        "| ---: | ---: | ---: | ---: |"]
    for row in equations:
        candidate = next(candidate for candidate in row["saved_three_term_candidates"] if candidate["lambda_ratio"]==diagnostic["threshold_ratio"])
        lines.append(f"| {row['equation']} | {row['pairs_verified']:,} | {row['minimum_dual_bound_over_threshold']:.9g} | {candidate['coefficient_e2']:.9g} |")
    lines += ["","Every support of size ≤2 is contained in a checked pair. The saved three-term coefficient vectors were checked against all physical constraints with exact rational arithmetic. Thus their minimum cardinality is three for the stated saved score and lambda.","",
        "This proof does not establish residual tie-break optimality. Main experiment flags and summaries remain unchanged. It concerns the saved binary64 score and finite lambda candidates, not continuum integral exactness.",""]
    (directory/"nls_certificate_verification.md").write_text("\n".join(lines))
    print(json.dumps(dict(passed=report["passed"],witnesses=counts["exact_dual_witnesses_verified"],primals=counts["exact_three_term_primals_verified"],failures=len(failures),runtime_seconds=report["runtime_seconds"],report=str(json_path))))
    if failures:
        for failure in failures[:20]:
            print(json.dumps(failure))
        raise SystemExit(1)


if __name__=="__main__":
    main()
