"""Independent, solver-free audit of complete or incrementally saved sanity trials.

Uses saved physical score matrices, independently recomputed metrics and seeds,
and frozen source/data hashes.  No optimization or trial output is modified.
"""
from __future__ import annotations

import os
for _variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pdes import PDES


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def independent_metrics(coefficients, truth):
    coefficients, truth = np.asarray(coefficients), np.asarray(truth)
    selected, actual = coefficients != 0, truth != 0
    tp = int(np.sum(selected & actual))
    fn = int(np.sum(~selected & actual))
    fp = int(np.sum(selected & ~actual))
    return dict(e2=float(np.linalg.norm(coefficients-truth)/np.linalg.norm(truth)),
                e_inf=float(np.max(np.abs((coefficients-truth)[actual]/truth[actual]))),
                tp=tp, fn=fn, fp=fp, tpr=tp/(tp+fn+fp), exact_support=fn == 0 and fp == 0)


def independent_seed(name, noise, trial):
    # Explicit reconstruction, without calling the experiment's seed helper.
    order = ("IB", "KdV", "KS", "NLS", "SG", "RD", "NS", "HKS", "VBG")
    bits = int(np.float64(noise).view(np.uint64))
    sequence = np.random.SeedSequence([0, order.index(name), bits & 0xffffffff, bits >> 32, trial])
    return sequence.generate_state(4).tolist()


def cache_paths(directory, job):
    name, noise, trial = job
    key = json.dumps([name, noise, trial], separators=(",", ":"))
    suffix = hashlib.sha256(key.encode()).hexdigest()[:12]
    stem = f"{name}_noise-{noise:g}_trial-{trial}-{suffix}"
    return directory/f"{stem}.json", directory/f"{stem}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory/"manifest.json").read_text())
    preparation = json.loads((directory/"preparation_manifest.json").read_text())
    preparation_hash = hashlib.sha256(json.dumps(preparation, sort_keys=True).encode()).hexdigest()
    failures, counts = [], Counter()
    maxima = dict(score_residual_recompute_difference_relative_to_b=0., coefficient_e2_recompute_difference=0.,
                  selected_constraint_violation_relative_to_b=0., candidate_constraint_violation_relative_to_b=0.)

    def check(condition, label, **context):
        counts["assertions"] += 1
        if not condition:
            failures.append(dict(check=label, **context))

    def close(actual, expected, label, *, rtol=2e-10, atol=1e-13, **context):
        check(bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)), label, **context)

    def audit_metrics(result, truth, label):
        coefficients = np.asarray(result["coefficients"], dtype=float)
        check(coefficients.shape == truth.shape and np.all(np.isfinite(coefficients)), "finite metric coefficients", item=label)
        for key, value in independent_metrics(coefficients, truth).items():
            close(result[key], value, "recomputed metric", item=label, metric=key)
        if "equations" in result:
            check(len(result["equations"]) == len(truth), "equation metric count", item=label)
            for equation, (coefficient, target) in enumerate(zip(coefficients, truth)):
                for key, value in independent_metrics(coefficient, target).items():
                    close(result["equations"][equation][key], value, "recomputed equation metric", item=label, equation=equation, metric=key)

    check(manifest["preparation_hash"] == preparation_hash, "preparation manifest signature")
    for relative, expected_hash in manifest["sources"].items():
        check(digest(ROOT/relative) == expected_hash, "frozen source hash", path=relative)
    for relative, expected_hash in preparation["sources"].items():
        check(digest(ROOT/relative) == expected_hash, "preparation source hash", path=relative)
    for name, info in manifest["data"].items():
        check(digest(ROOT/"data"/info["file"]) == info["sha256"], "clean data hash", pde=name)
    # The optimization and score-construction functions have no truth input/name.
    solver_ast = ast.parse((ROOT/"wsindy/dantzig.py").read_text())
    one = next(node for node in solver_ast.body if isinstance(node, ast.FunctionDef) and node.name == "solve_one")
    check(not any(isinstance(node, ast.Name) and node.id == "truth" for node in ast.walk(one)), "truth absent from solve_one")
    check(not any(argument.arg == "truth" for argument in one.args.args+one.args.kwonlyargs), "truth absent from solve_one signature")
    score_ast = ast.parse((ROOT/"wsindy/scores.py").read_text())
    check(not any(isinstance(node, ast.Name) and node.id == "truth" for node in ast.walk(score_ast)), "truth absent from score construction")

    expected_jobs = {(name, noise, trial) for name in manifest["pdes"] for noise in manifest["noise"]
                     for trial in range(1 if noise == 0 else manifest["trials"])}
    snapshot = (directory/"trials.jsonl").read_bytes()
    partial_tail = bool(snapshot and not snapshot.endswith(b"\n"))
    lines = snapshot.splitlines()
    if partial_tail:
        lines = lines[:-1]
    records = [json.loads(line) for line in lines if line.strip()]
    seen_jobs = set()
    prepared_count = 0
    # Verify all atomically completed cache pairs, including trials not solved yet.
    for metadata_path in sorted((directory/"prepared").glob("*.json")):
        metadata = json.loads(metadata_path.read_text())
        job = (metadata["name"], metadata["noise_ratio"], metadata["trial"])
        matrix_path = metadata_path.with_suffix(".npz")
        check(job in expected_jobs, "prepared cache in protocol scope", job=job)
        check(metadata["preparation_hash"] == preparation_hash, "cache preparation signature", job=job)
        check(matrix_path.exists() and digest(matrix_path) == metadata["score_cache_sha256"], "score archive hash", job=job)
        check(metadata["score_cache_file"] == matrix_path.name, "score archive filename", job=job)
        check(metadata["seed"] == independent_seed(*job), "prepared deterministic noise seed", job=job)
        prepared_count += 1

    for record in records:
        job = (record["name"], record["noise_ratio"], record["trial"])
        check(job in expected_jobs and job not in seen_jobs, "unique protocol realization", job=job)
        seen_jobs.add(job)
        check(record["seed"] == independent_seed(*job), "paired deterministic noise seed", job=job)
        metadata_path, matrix_path = cache_paths(directory/"prepared", job)
        metadata = json.loads(metadata_path.read_text())
        check(record["score_cache_sha256"] == digest(matrix_path), "trial score archive hash", job=job)
        check(record["preparation_hash"] == preparation_hash, "trial preparation signature", job=job)
        for key in ("name", "noise_ratio", "trial", "seed", "noise_std", "truth", "terms", "pilot", "score", "score_cache_file"):
            check(record[key] == metadata[key], "trial matches prepared metadata", job=job, field=key)
        truth = np.asarray(record["truth"], dtype=float)
        pde = PDES[record["name"]]
        expected_truth = np.array([[equation.get(term, 0.) for term in record["terms"]] for equation in pde.truth])
        check(np.array_equal(truth, expected_truth), "physical dictionary truth", job=job)
        for method in ("wsindy", "debiased_mstls"):
            check(record["methods"][method] == metadata["methods"][method], "baseline unchanged from paired preparation", job=job, method=method)
        for method, result in record["methods"].items():
            audit_metrics(result, truth, f"{job}/{method}")
            if not method.endswith("dantzig"):
                continue
            audit_metrics(result["incumbent_diagnostic"], truth, f"{job}/{method}/incumbent")
            variant = method.removesuffix("_dantzig")
            with np.load(matrix_path, allow_pickle=False) as archive:
                H, B = archive[f"{variant}_H"], archive[f"{variant}_b"]
            check(np.all(np.isfinite(H)) and np.all(np.isfinite(B)), "finite physical score", job=job, method=method)
            check(np.array_equal(H, H.T), "symmetric score", job=job, method=method)
            check(len(result["paths"]) == len(truth), "one oracle path per equation", job=job, method=method)
            for equation, (path, target) in enumerate(zip(result["paths"], truth)):
                context = dict(job=job, method=method, equation=equation)
                b = B[:, equation]
                scale = float(np.max(np.abs(b)))
                close(path["score_scale"], scale, "physical score scale", **context)
                candidates = path["path"]
                valid_indices, certified_indices = [], []
                for index, candidate in enumerate(candidates):
                    ctx = dict(context, candidate=index)
                    counts["candidate_solutions"] += 1
                    lam = candidate["lambda"]
                    close(lam, candidate["lambda_ratio"]*scale, "physical lambda", **ctx)
                    if candidate["coefficients"] is None:
                        check(not candidate["validated"] and not candidate["optimal"], "no incumbent cannot be eligible", **ctx)
                        continue
                    coefficient = np.asarray(candidate["coefficients"], dtype=float)
                    check(coefficient.shape == target.shape and np.all(np.isfinite(coefficient)), "finite physical candidate", **ctx)
                    independent_e2 = float(np.linalg.norm(coefficient-target)/np.linalg.norm(target))
                    close(candidate["e2"], independent_e2, "candidate physical E2", **ctx)
                    maxima["coefficient_e2_recompute_difference"] = max(maxima["coefficient_e2_recompute_difference"], abs(independent_e2-candidate["e2"]))
                    residual = float(np.max(np.abs(b-H@coefficient)))
                    safe_scale = max(scale, np.finfo(float).tiny)
                    violation = max(0., residual-lam)
                    slack = min(2e-8*safe_scale, .01*lam)+50*np.finfo(float).eps*safe_scale
                    independently_valid = bool(np.all(np.isfinite(coefficient)) and np.isfinite(residual) and violation <= slack)
                    recompute_difference = abs(residual-candidate["residual_inf"])/safe_scale
                    maxima["score_residual_recompute_difference_relative_to_b"] = max(maxima["score_residual_recompute_difference_relative_to_b"], recompute_difference)
                    close(candidate["residual_inf"], residual, "candidate physical score residual", atol=2e-13*safe_scale, **ctx)
                    close(candidate["validation_slack"], slack, "candidate physical validation slack", atol=2e-15*safe_scale, **ctx)
                    check(candidate["validated"] == independently_valid, "independent score feasibility eligibility", **ctx)
                    support = np.asarray(candidate["support"], dtype=bool)
                    check(np.all(coefficient[~support] == 0.), "SOS1 excluded coefficients exactly zero", **ctx)
                    actual_cardinality, permitted_cardinality = int(np.count_nonzero(coefficient)), int(np.count_nonzero(support))
                    check(candidate["cardinality"] == actual_cardinality, "candidate cardinality metadata", **ctx)
                    if candidate["cardinality_optimal"]:
                        lower = candidate["cardinality_bound"]
                        check(lower is not None and np.isfinite(lower) and np.ceil(lower-1e-7) >= permitted_cardinality,
                              "integer lower bound certifies primary cardinality", **ctx)
                        check(lower is not None and lower <= permitted_cardinality+1e-6, "primary bound below incumbent cardinality", **ctx)
                    if candidate["optimal"]:
                        check(candidate["cardinality_optimal"] and candidate["secondary_optimal"], "lexicographic certification", **ctx)
                        check(candidate["secondary_status"] in (2, "AnalyticZero"), "secondary solved status", **ctx)
                    if candidate["validated"]:
                        valid_indices.append(index)
                        maxima["candidate_constraint_violation_relative_to_b"] = max(maxima["candidate_constraint_violation_relative_to_b"], violation/safe_scale)
                        if candidate["optimal"]:
                            certified_indices.append(index)
                            counts["certified_candidate_solutions"] += 1
                check(bool(certified_indices), "certified oracle candidate exists", **context)
                expected_index = min(certified_indices, key=lambda i: (candidates[i]["e2"], candidates[i]["lambda_ratio"]))
                check(path["selected_index"] == expected_index, "oracle argmin restricted to certified valid path", **context)
                selected = candidates[expected_index]
                check(path["selected"] == selected, "selected candidate matches path index", **context)
                check(path["coefficients"] == selected["coefficients"] == result["coefficients"][equation], "selected physical coefficients propagated", **context)
                check(result["solver"][equation]["selected"] == selected, "runner solver audit propagated", **context)
                incumbent_index = min(valid_indices, key=lambda i: (candidates[i]["e2"], candidates[i]["lambda_ratio"]))
                check(path["incumbent_oracle_selected"] == candidates[incumbent_index], "incumbent diagnostic oracle argmin", **context)
                check(result["incumbent_diagnostic"]["coefficients"][equation] == candidates[incumbent_index]["coefficients"], "incumbent diagnostic coefficients propagated", **context)
                ratios = [candidate["lambda_ratio"] for candidate in candidates]
                check(len(ratios) == len(set(ratios)), "unique lambda path values", **context)
                base = sorted(set(manifest["options"]["ratios"]) | {1.}, reverse=True)
                check(ratios[:len(base)] == base, "initial lambda grid matches protocol", **context)
                consumed = len(base)
                for _ in range(manifest["options"]["refinements"]):
                    past = candidates[:consumed]
                    eligible = [candidate for candidate in past if candidate["validated"] and np.isfinite(candidate["e2"])]
                    winner = min(eligible, key=lambda candidate: (candidate["e2"], candidate["lambda_ratio"]))
                    seen = {candidate["lambda_ratio"] for candidate in past}
                    ordered = sorted(seen)
                    position = ordered.index(winner["lambda_ratio"])
                    neighbors = []
                    if position:
                        neighbors.append(float(np.sqrt(ordered[position-1]*ordered[position])))
                    elif ordered[position] > 1e-12:
                        neighbors.extend([ordered[position]/10, ordered[position]/100])
                    if position+1 < len(ordered):
                        neighbors.append(float(np.sqrt(ordered[position]*ordered[position+1])))
                    new = []
                    for ratio in neighbors:
                        if ratio not in seen:
                            new.append(ratio)
                            seen.add(ratio)
                    check(ratios[consumed:consumed+len(new)] == new, "oracle-only lambda refinement schedule", **context)
                    consumed += len(new)
                check(consumed == len(candidates), "lambda path consists only of declared grid/refinement", **context)
                truth_residual = float(np.max(np.abs(b-H@target)))
                close(record["score"][variant]["truth_residual_inf"][equation], truth_residual, "prepared truth score diagnostic", atol=2e-13*max(scale, truth_residual), **context)
                close(result["solver"][equation]["truth_residual_inf"], truth_residual, "runner truth score diagnostic", atol=2e-13*max(scale, truth_residual), **context)
                selected_coefficient = np.asarray(selected["coefficients"])
                selected_violation = max(0., float(np.max(np.abs(b-H@selected_coefficient)))-selected["lambda"])
                maxima["selected_constraint_violation_relative_to_b"] = max(maxima["selected_constraint_violation_relative_to_b"], selected_violation/max(scale, np.finfo(float).tiny))
                counts["equation_oracles"] += 1
        counts["completed_realizations"] += 1
    if args.require_complete:
        check(seen_jobs == expected_jobs, "all protocol realizations complete")
    audit = dict(protocol="independent-solver-free-dantzig-audit-v1", passed=not failures,
        audited_at_utc=datetime.now(timezone.utc).isoformat(), executable=sys.executable, numpy=np.__version__,
        expected_realizations=len(expected_jobs), audited_realizations=len(records), completed_preparation_caches=prepared_count,
        complete=seen_jobs == expected_jobs, ignored_in_progress_jsonl_tail=partial_tail,
        trial_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), auditor_source_sha256=digest(__file__),
        counts=dict(counts), maxima=maxima, failures=failures,
        source_review="No truth input/name in solve_one or score construction. Reviewed truth uses in the frozen runner/path: physical E2 and support evaluation, score diagnostics, oracle lambda ranking/refinement. Warm starts select saved feasible coefficients by neighboring lambda, not coefficient truth.",
        limitations=["Does not rerun Gurobi or independently reproduce its branch-and-bound lower bounds; checks reported bounds and certification consistency.",
                     "Checks saved physical score caches, their signatures and all candidate constraints; does not rebuild full weak scores from raw observations."])
    output = directory/"audit.json"
    temporary = output.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, output)
    print(json.dumps(dict(passed=audit["passed"], complete=audit["complete"], audited_realizations=len(records),
                         expected_realizations=len(expected_jobs), candidates=counts["candidate_solutions"],
                         equation_oracles=counts["equation_oracles"], failures=len(failures), audit=str(output))))
    if failures:
        for failure in failures[:12]:
            print(json.dumps(failure))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
