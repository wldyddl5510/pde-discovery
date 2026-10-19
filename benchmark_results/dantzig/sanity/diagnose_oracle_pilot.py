"""Separate oracle-pilot diagnostic for noiseless IB, outside all main methods.

This deliberately uses the clean state as the pilot at regression times. It is
not a lambda-only oracle result and must never be pooled with the main study.
It isolates whether pilot error can explain the observed support-selection
failure while preserving quadrature, dictionary, tests, and physical score units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pdes import DATA_DIR, PDES, load
from wsindy import SplitPilot, SplitSystem, estimate
from wsindy.scores import score_matrices
from diagnose_scores import digest, one_column_paths, serializable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path(__file__).resolve().parent/"prepared")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent/"diagnostics_oracle_pilot.json")
    args = parser.parse_args()
    metadata_path = next(args.prepared.glob("IB_noise-0_trial-0-*.json"))
    metadata = json.loads(metadata_path.read_text())
    original_path = metadata_path.with_suffix(".npz")
    if digest(original_path) != metadata["score_cache_sha256"]:
        raise ValueError("Current pilot cache SHA256 does not match metadata")

    clean, grid = load("IB", download=False)
    pde = PDES["IB"]
    current_pilot, pilot_info = estimate(clean, grid, pde, SplitPilot(ratio=.5, prior=.01))
    training = np.isnan(current_pilot).all(axis=(0, 2))
    oracle_pilot = clean.copy()
    oracle_pilot[:, training, :] = np.nan
    system = SplitSystem(pde, clean, grid, X=oracle_pilot, Y=clean)
    scores = score_matrices(system, variant="debiased")
    terms = [term.label for term in system.terms]
    truth = pde.truth_matrix(system.terms)[0]
    H, b = scores.H, scores.b[:, 0]
    oracle_lp = one_column_paths(H, b, truth, terms)
    oracle_rms = np.sqrt(np.maximum(np.diag(scores.X.T@scores.X/scores.n_rows), 0.))
    current_lp = {}
    with np.load(original_path) as current:
        for variant in ("plugin", "debiased"):
            current_lp[variant] = one_column_paths(current[variant+"_H"], current[variant+"_b"][:, 0], truth, terms)
        current_rms = np.sqrt(np.maximum(np.diag(current["plugin_H"]), 0.))
    weak_cache = args.out.with_suffix(".npz")
    np.savez_compressed(weak_cache, H=H, b=b, X=scores.X, Y=scores.Y)
    base_H = scores.X.T@scores.X/scores.n_rows
    payload = dict(
        schema_version=1,
        diagnostic_only=True,
        label="ORACLE PILOT: noiseless IB, excluded from the main lambda-only oracle study",
        question="Does exact pilot restore correct support under the same physical score and residual tie-break?",
        answer=dict(correct_one_term_support=oracle_lp["best_one_column"]["matches_true_support"],
                    best_one_column=oracle_lp["best_one_column"],
                    next_best_wrong_support=next(r for r in oracle_lp["all_one_column_solutions"]
                                                if r["success"] and not r["matches_true_support"])),
        provenance=dict(
            name="IB", noise_ratio=0., trial=0, seed=metadata["seed"],
            data_file=str((DATA_DIR/pde.file).resolve()), data_sha256=digest(DATA_DIR/pde.file),
            current_metadata_file=metadata_path.name, current_metadata_sha256=digest(metadata_path),
            current_matrix_file=original_path.name, current_matrix_sha256=digest(original_path),
            oracle_weak_cache_file=weak_cache.name, oracle_weak_cache_sha256=digest(weak_cache),
            sources={str(path.relative_to(ROOT)): digest(path)
                     for path in (Path(__file__), ROOT/"pdes.py", ROOT/"wsindy/system.py",
                                  ROOT/"wsindy/scores.py", ROOT/"wsindy/estimators.py")}),
        common_protocol=dict(
            test_half_widths=list(system.tests.m), test_degrees=list(system.tests.p),
            query_strides=list(system.tests.s), weak_rows=scores.n_rows,
            training_time_indices=np.flatnonzero(training).tolist(),
            regression_time_indices=np.flatnonzero(~training).tolist(),
            score_units="Physical columns (rescale=False), common infinity-norm tolerance",
            split_quadrature="Identical SplitSystem time weights, masks, and full-grid test functions",
            tie_break="Exhaustive independent one-column infinity-residual minimization with scipy.optimize.linprog/HiGHS"),
        state_diagnostics=dict(
            clean_min=float(clean.min()), clean_max=float(clean.max()), current_pilot_info=pilot_info,
            current_pilot_regression_relative_l2=float(np.linalg.norm(current_pilot[:, ~training]-clean[:, ~training])
                                                     /np.linalg.norm(clean[:, ~training])),
            oracle_pilot_regression_relative_l2=0.,
            exact_pilot_score_relative_difference_from_plain_gram=float(np.max(np.abs(H-base_H))/np.max(np.abs(base_H))),
            directional_increment_relative_norm=float(np.linalg.norm(scores.dX)/np.linalg.norm(scores.X))),
        physical_weak_design=dict(
            current_pilot_column_rms_min=float(current_rms.min()), current_pilot_column_rms_max=float(current_rms.max()),
            current_pilot_column_rms_ratio=float(current_rms.max()/current_rms.min()),
            oracle_pilot_column_rms_min=float(oracle_rms.min()), oracle_pilot_column_rms_max=float(oracle_rms.max()),
            oracle_pilot_column_rms_ratio=float(oracle_rms.max()/oracle_rms.min())),
        truth_feasibility_ratio=dict(
            current_pilot_debiased=metadata["score"]["debiased"]["truth_residual_inf"][0]
                                    /metadata["score"]["debiased"]["lambda_max"][0],
            oracle_pilot=float(np.max(np.abs(b-H@truth))/np.max(np.abs(b)))),
        current_pilot_support1_diagnoses=current_lp,
        oracle_pilot_support1_diagnosis=oracle_lp,
        interpretation=[
            "This uses a second oracle (the clean pilot), so it is diagnostic evidence only and is excluded from main method comparisons.",
            "The LP estimator never receives true coefficients/support; truth is used after each LP for labels and coefficient error.",
            "Correct support with an exact pilot shows the current IB failure is not unavoidable score non-identifiability for these masks/tests/physical units.",
            "Its remaining coefficient error includes weak quadrature/discretization effects and the score minimax tie-break; it need not match the full-data OLS error.",
            "Large physical feature dynamic range remains under the exact pilot; this diagnostic does not establish robustness to pilot or observation error."])
    args.out.write_text(json.dumps(serializable(payload), indent=2)+"\n")
    best = oracle_lp["best_one_column"]
    print(f"Wrote {args.out}; exact-pilot winner {best['term']}={best['coefficient']:.12g}, "
          f"E2={best['coefficient_e2']:.8g}, normalized radius={best['normalized_residual']:.8g}")


if __name__ == "__main__":
    main()
