"""Independently recompute metrics and physical score feasibility from saved candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import interpret
import run_study as study
from wsindy.dantzig import validate_score


def audit(directory, *, allow_partial=False):
    manifest = json.loads((directory/"manifest.json").read_text())
    preparation = json.loads((directory/"preparation_manifest.json").read_text())
    preparation_hash = hashlib.sha256(json.dumps(preparation, sort_keys=True).encode()).hexdigest()
    assert manifest["preparation_hash"] == preparation_hash
    for relative, target in manifest["sources"].items():
        assert study.digest(study.ROOT/relative) == target, f"Changed source: {relative}"
    for name, entry in manifest["data"].items():
        assert study.digest(study.DATA_DIR/entry["file"]) == entry["sha256"], f"Changed clean data: {name}"
    planned = {(name, noise, trial) for name in manifest["pdes"] for noise in manifest["noise"]
               for trial in range(1 if noise == 0 else manifest["trials"])}
    records = study.load_records(directory)
    keys = {(r["name"], r["noise_ratio"], r["trial"]) for r in records}
    assert len(records) == len(keys), "Duplicate realizations"
    assert keys <= planned
    if not allow_partial:
        assert keys == planned, f"Only {len(keys)}/{len(planned)} realizations are complete"
    candidates = valid = main_equations = primary_equations = zero = unused_indicator_candidates = 0
    max_relative_violation = 0.
    for record in records:
        name, noise, trial = record["name"], record["noise_ratio"], record["trial"]
        assert record["seed"] == study.trial_seed(name, noise, trial)
        _, matrices = study.read_prepared(directory/"prepared", name, noise, trial, preparation_hash)
        truth = np.asarray(record["truth"])
        for method in study.METHODS:
            result = record["methods"][method]
            recomputed = study.metrics(np.asarray(result["coefficients"]), truth)
            for metric in ("e2", "e_inf", "tp", "fp", "fn", "exact_support"):
                assert np.isclose(recomputed[metric], result[metric]), (name, noise, trial, method, metric)
            if not method.endswith("dantzig"):
                continue
            variant = method.removesuffix("_dantzig")
            H, b = matrices[f"{variant}_H"], matrices[f"{variant}_b"]
            zero += int(not np.any(result["coefficients"]))
            for e, path in enumerate(result["paths"]):
                selected = path["selected"]
                assert selected["validated"] and selected["optimal"]
                assert np.array_equal(selected["coefficients"], result["coefficients"][e])
                main_equations += 1
                for candidate in path["path"]:
                    candidates += 1
                    coefficient = candidate.get("coefficients")
                    if coefficient is None:
                        assert not candidate.get("validated", False)
                        continue
                    beta = np.asarray(coefficient)
                    expected_e2 = np.linalg.norm(beta-truth[e])/np.linalg.norm(truth[e])
                    assert np.isclose(expected_e2, candidate["e2"])
                    assert np.isclose(candidate["lambda"], candidate["lambda_ratio"]*np.max(np.abs(b[:, e])), rtol=1e-12, atol=0)
                    checked = validate_score(H, b[:, e], beta, candidate["lambda"])
                    assert checked["validated"] == candidate["validated"]
                    if candidate["validated"]:
                        valid += 1
                        max_relative_violation = max(max_relative_violation, checked["relative_constraint_violation"])
                        assert np.isfinite(beta).all()
                        support = np.asarray(candidate["support"], dtype=bool)
                        assert not np.any((beta != 0) & ~support)
                        # An unresolved incumbent may leave indicators active
                        # while assigning their coefficients exactly zero.
                        unused = np.any(support & (beta == 0))
                        unused_indicator_candidates += int(unused)
                        if candidate.get("cardinality_optimal", False):
                            assert not unused
            all_levels = {level: interpret.select(record, method, level) for level in interpret.LEVELS}
            errors = [all_levels[level]["e2"] for level in interpret.LEVELS]
            assert errors[2] <= errors[1]+1e-12 <= errors[0]+2e-12 <= 1+3e-12
            for selected in all_levels["cardinality"]["selected"]:
                assert selected["validated"] and selected["cardinality_optimal"]
                primary_equations += 1
    result = dict(completed_realizations=len(records), expected_realizations=len(planned),
        numerical_source_hashes_match=True, data_hashes_match=True, cache_hashes_match=True,
        no_duplicate_realizations=True, seeds_match=True, baseline_and_dantzig_metrics_recomputed=True,
        candidate_e2_recomputed=True, candidate_feasibility_recomputed=True,
        nested_oracle_errors_monotone=True, main_selected_equations=main_equations,
        cardinality_selected_equations=primary_equations, candidates=candidates,
        validated_candidates=valid, max_relative_constraint_violation=max_relative_violation,
        uncertified_candidates_with_unused_indicators=unused_indicator_candidates,
        main_all_zero_models=zero)
    filename = "results_audit.partial.json" if allow_partial else "results_audit.json"
    (directory/filename).write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.output.resolve(), allow_partial=args.allow_partial), indent=2))
