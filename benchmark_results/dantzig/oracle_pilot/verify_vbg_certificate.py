"""Independently verify VBG minimum-cardinality certificates without LP solves.

The generator uses four-by-four cofactors. This verifier instead solves the
dual linear system by fraction-free Gaussian elimination, then checks every
physical binary64 identity with exact integer ratios and rational objectives.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from math import lcm
from pathlib import Path
from time import perf_counter

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve_weights(signed,support,H):
    """Solve all exact stationarity equations and unit mass by elimination."""
    width = len(signed)
    matrix = [[1]*width+[1]]
    for column in support:
        ratios = [(sign*H[row][column][0],H[row][column][1]) for row,sign in signed]
        denominator = max(value[1] for value in ratios)
        matrix.append([numerator*(denominator//base) for numerator,base in ratios]+[0])
    rows,previous = len(matrix),1
    for column in range(width):
        selected = next((row for row in range(column,rows) if matrix[row][column]),None)
        if selected is None:
            raise ValueError("Stored witness lacks a uniquely determined dual solution")
        matrix[column],matrix[selected] = matrix[selected],matrix[column]
        pivot = matrix[column][column]
        for row in range(column+1,rows):
            multiplier = matrix[row][column]
            for later in range(column+1,width+1):
                numerator = pivot*matrix[row][later]-multiplier*matrix[column][later]
                value,remainder = divmod(numerator,previous)
                if remainder:
                    raise ValueError("Exact Gaussian elimination division failed")
                matrix[row][later] = value
            matrix[row][column] = 0
        previous = pivot
    if any(matrix[row][width] for row in range(width,rows)):
        raise ValueError("Dual linear system is inconsistent")
    weights = [Fraction(0)]*width
    for row in range(width-1,-1,-1):
        weights[row] = (Fraction(matrix[row][width])-sum(matrix[row][column]*weights[column] for column in range(row+1,width)))/matrix[row][row]
    denominator = lcm(*(weight.denominator for weight in weights))
    integers = [weight.numerator*(denominator//weight.denominator) for weight in weights]
    return integers,denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    directory = args.directory.resolve()
    start = perf_counter()
    diagnostic_path = directory/"vbg_supports.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    provenance = diagnostic["provenance"]
    metadata_path = directory/"prepared"/provenance["metadata_file"]
    matrix_path = directory/"prepared"/provenance["score_file"]
    solution_path = directory/"solved"/provenance["solution_file"]
    archive_path = directory/"vbg_supports.npz"
    counts,failures = Counter(),[]

    def check(condition,label,**context):
        counts["assertions"] += 1
        if not condition:
            failures.append(dict(check=label,**context))

    for path,sha in ((metadata_path,provenance["metadata_sha256"]),(matrix_path,provenance["score_sha256"]),
                     (solution_path,provenance["solution_sha256"]),(archive_path,provenance["output_archive_sha256"]),
                     (directory/"diagnose_vbg_supports.py",provenance["diagnostic_source_sha256"])):
        check(digest(path)==sha,"certificate input provenance hash",path=path.name)
    metadata = json.loads(metadata_path.read_text())
    solution = json.loads(solution_path.read_text())
    method = diagnostic["method"]
    with np.load(matrix_path,allow_pickle=False) as archive:
        H,b = archive[method+"_H"],archive[method+"_b"][:,0]
    with np.load(archive_path,allow_pickle=False) as archive:
        supports,rows_saved,ratios_saved,excluded_saved = (archive[key] for key in
            ("supports","exact_dual_signed_rows","exact_dual_ratio","exact_excludes_threshold"))
    p = len(b)
    expected = np.asarray(list(combinations(range(p),4)),dtype=int)
    check(H.shape==(p,p) and H.dtype==b.dtype==np.float64,"saved physical binary64 score")
    check(np.array_equal(supports,expected),"all unordered four-column supports exactly once")
    check(rows_saved.shape==(len(expected),5),"one signed dual witness per support")
    check(diagnostic["complete"] and diagnostic["counterexample"] is None and diagnostic["visited_supports"]==len(expected),"complete support exclusion diagnostic")
    if supports.shape!=expected.shape:
        raise ValueError("Wait for the complete VBG support archive before verifying")
    Hr = [[float(value).as_integer_ratio() for value in row] for row in H]
    br = [float(value).as_integer_ratio() for value in b]
    scale = float(np.max(np.abs(b)))
    target_float = float(diagnostic["target_ratio"]*scale)
    target = Fraction.from_float(target_float)
    minimum,minimum_support = None,None
    for index,support_array in enumerate(supports):
        support = support_array.tolist()
        context = dict(support=support)
        identifiers = [row for row in rows_saved[index].tolist() if row>=0]
        check(1<=len(identifiers)<=5 and len(set(identifiers))==len(identifiers)
              and all(row<2*p for row in identifiers),"signed witness row IDs",**context)
        signed = [(row%p,1 if row<p else -1) for row in identifiers]
        integers,total = solve_weights(signed,support,Hr)
        check(all(value>=0 for value in integers) and sum(integers)==total,"exact nonnegative unit dual mass",**context)
        for column in support:
            denominator = max(Hr[row][column][1] for row,sign in signed)
            stationarity = sum(weight*sign*Hr[row][column][0]*(denominator//Hr[row][column][1])
                               for weight,(row,sign) in zip(integers,signed))
            check(stationarity==0,"exact raw physical H dual stationarity",column=column,**context)
        dual = {}
        for weight,(row,sign) in zip(integers,signed):
            dual[row] = dual.get(row,0)-weight*sign
        check(sum(abs(value) for value in dual.values())<=total,"exact dual vector l1 norm <=1",**context)
        response_denominator = max(br[row][1] for row in dual)
        numerator = sum(value*br[row][0]*(response_denominator//br[row][1]) for row,value in dual.items())
        bound = Fraction(numerator,total*response_denominator)
        check(bound>target,"exact dual bound excludes stored physical lambda",**context)
        check(bool(excluded_saved[index]),"saved exact exclusion flag",**context)
        check(np.isclose(ratios_saved[index],float(bound)/scale,rtol=2e-14,atol=0),"saved dual objective ratio",**context)
        if minimum is None or bound<minimum:
            minimum,minimum_support = bound,support
        counts["exact_dual_witnesses_verified"] += 1
        if (index+1)%20000==0:
            print(f"[VBG exact duals] {index+1}/{len(supports)},elapsed={perf_counter()-start:.1f}s",flush=True)
    check(minimum==Fraction(diagnostic["minimum_exact_dual_fraction"]),"reported exact minimum support lower bound")
    candidates = []
    for candidate in diagnostic["saved_five_term_candidates"]:
        support = [coefficient["index"] for coefficient in candidate["coefficients"]]
        values = [Fraction.from_float(float(coefficient["value"])) for coefficient in candidate["coefficients"]]
        check(len(support)==len(set(support))==5 and all(values),"exact primal support size five",lambda_ratio=candidate["lambda_ratio"])
        beta = np.zeros(p)
        for coefficient in candidate["coefficients"]:
            beta[coefficient["index"]] = coefficient["value"]
            check(coefficient["term"]==metadata["terms"][coefficient["index"]],"five-term dictionary labels")
        matches = [row for row in solution["result"]["paths"][0]["path"]
                   if row["lambda"]==candidate["lam"] and row["coefficients"]==beta.tolist()]
        check(len(matches)==1,"primal beta/lambda match existing saved path",lambda_ratio=candidate["lambda_ratio"])
        residual = max(abs(Fraction(*response)-sum(Fraction(*row[column])*value for column,value in zip(support,values)))
                       for row,response in zip(Hr,br))
        lam = Fraction.from_float(float(candidate["lam"]))
        feasible = residual<=lam
        minimum_cardinality_five = feasible and minimum>lam
        check(candidate["physical_feasible"]==feasible,"exact primal feasibility",lambda_ratio=candidate["lambda_ratio"])
        check(Fraction(candidate["exact_physical_residual_fraction"])==residual,"exact primal residual fraction",lambda_ratio=candidate["lambda_ratio"])
        check((candidate["independently_certified_minimum_cardinality"]==5)==minimum_cardinality_five,
              "primal plus exhaustive dual certificate establishes minimum cardinality",lambda_ratio=candidate["lambda_ratio"])
        truth = np.asarray(metadata["truth"])[0]
        e2 = float(np.linalg.norm(beta-truth)/np.linalg.norm(truth))
        check(np.isclose(candidate["e2"],e2,rtol=2e-13,atol=0),"post-proof coefficient error",lambda_ratio=candidate["lambda_ratio"])
        candidates.append(dict(lambda_ratio=candidate["lambda_ratio"],physical_lambda=str(lam),exact_physical_residual=str(residual),
            physical_feasible=feasible,minimum_cardinality_five_proved=minimum_cardinality_five,coefficient_e2=e2,
            minimum_dual_bound_over_lambda=float(minimum/lam),
            original_cardinality_flag=matches[0]["cardinality_optimal"] if matches else None,
            original_secondary_flag=matches[0]["secondary_optimal"] if matches else None))
        counts["exact_five_term_primals_verified"] += 1
        counts["minimum_cardinality_five_certificates"] += int(minimum_cardinality_five)
    check(any(row["lambda_ratio"]==diagnostic["target_ratio"] and row["minimum_cardinality_five_proved"] for row in candidates),"minimum cardinality five established at stated target")
    report = dict(protocol="independent-exact-VBG-certificate-verification-v1",passed=not failures,
        created_at_utc=datetime.now(timezone.utc).isoformat(),runtime_seconds=perf_counter()-start,
        dictionary_columns=p,supports_verified=len(supports),counts=dict(counts),failures=failures,
        physical_lambda_at_target=str(target),minimum_support_dual_bound=str(minimum),minimum_support=minimum_support,
        minimum_dual_bound_over_target=float(minimum/target),all_supports_size_at_most_four_excluded=True,
        saved_five_term_candidates=candidates,
        provenance=dict(verifier_source_sha256=digest(__file__),diagnostic_json_sha256=digest(diagnostic_path),
            certificate_archive_sha256=digest(archive_path),physical_score_archive_sha256=digest(matrix_path),
            original_saved_solution_sha256=digest(solution_path)),
        proof="Physical binary64 values are exact integer ratios. Gaussian elimination solves mass/stationarity, then exact raw-H identities and l1<=1 yield supportwise lower bounds. Every four-support bound exceeds lambda and a feasible five-term primal establishes minimum cardinality5.",
        limitations=["Does not certify secondary residual tie-break optimality or alter main solver flags.",
            "Certification transfers only to identical saved physical H,b and existing proved beta/lambda candidates; lambda ratio1e-5 is not certified by this exclusion bound."])
    (directory/"vbg_certificate_verification.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    lines = ["# Independent VBG certificate verification","",
        f"Passed: **{report['passed']}**. Verified {len(supports):,} exact dual witnesses covering every four-column support, without rerunning any LP.","",
        "The verifier independently solves each witness by fraction-free Gaussian elimination and checks exact physical stationarity, nonnegative mass one, dual l1 norm ≤1, and a strictly larger lower bound than the stored target lambda.","",
        f"Smallest exact dual bound / target lambda: **{report['minimum_dual_bound_over_target']:.12g}**.","",
        "| Saved lambda / ‖b‖∞ | Relative E2 | Minimum dual bound / lambda | Minimum cardinality5 proved |",
        "| ---: | ---: | ---: | :--- |"]
    for row in candidates:
        lines.append(f"| {row['lambda_ratio']:.9g} | {row['coefficient_e2']:.12g} | {row['minimum_dual_bound_over_lambda']:.9g} | {row['minimum_cardinality_five_proved']} |")
    lines += ["","Every smaller support is contained in a checked four-column support. Exact feasibility of a nonzero five-term vector therefore establishes minimum cardinality five at each certified saved lambda.","",
        "The lambda ratio1e-5 candidate is outside this certificate's bound and is excluded from the augmented reporting pool. The proof does not establish residual tie-break optimality or continuum-score accuracy. Main flags remain unchanged.",""]
    (directory/"vbg_certificate_verification.md").write_text("\n".join(lines))
    print(json.dumps(dict(passed=report["passed"],witnesses=len(supports),certified_primals=counts["minimum_cardinality_five_certificates"],
        failures=len(failures),runtime_seconds=report["runtime_seconds"],report=str(directory/"vbg_certificate_verification.json"))))
    if failures:
        for failure in failures[:20]:
            print(json.dumps(failure))
        raise SystemExit(1)


if __name__=="__main__":
    main()
