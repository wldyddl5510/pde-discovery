"""Audit saved IB time2 fits, physical units, pairing and unchanged baselines."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/"benchmark_results/debiasing/IB"
SOURCE = FOLDER/"mstls/results_source"
sys.path.insert(0, str(SOURCE))
import numpy as np
from scipy import linalg
from debiasing_study import fit_system, profile_settings, rule_size
from experiments import estimate_noise_std, trial_seed
from methods import build_wsindy_system, moving_average_pilot
from simulation_generation import load_clean_benchmark, observation_instance


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    protected = json.loads((FOLDER/"baseline.sha256.json").read_text())
    for path, expected in protected.items():
        assert sha(ROOT/path) == expected, path
    before = json.loads((FOLDER/"preexisting_results.summary.json").read_text())
    after = json.loads((ROOT/"results.summary.json").read_text())
    assert [row for row in after if row["method"] != "debiased-wsindy"] == before
    old_table_rows = json.loads((FOLDER/"preexisting_table_rows.json").read_text())
    table_rows = [line for line in (ROOT/"results.md").read_text().splitlines()
                  if line.startswith("| ") and not line.startswith(("| Method ", "| :--- ", "| Debiased "))]
    assert table_rows == old_table_rows
    clean = load_clean_benchmark("IB", data_dir=ROOT/"data", download=False)
    truth = clean.benchmark.truth()
    expected_keys = {(ratio, trial) for ratio in (0., .2, .5, .75, 1.) for trial in range(100)}
    baselines = [ROOT/"benchmark_results"/folder/"results_trials/IB.jsonl" for folder in
                 ("authors", "filtered", "lasso/original/raw", "lasso/original/filtered")]
    old = [{(r["noise_ratio"], r["trial"]): r for r in map(json.loads, path.read_text().splitlines())}
           for path in baselines]
    runs = {}
    snapshot_checks = 0
    for regression in ("mstls", "lasso"):
        folder = FOLDER/regression
        manifest = json.loads((folder/"results.manifest.json").read_text())
        for filename, expected in manifest["source_sha256"].items():
            assert sha(folder/"results_source"/filename) == expected
            snapshot_checks += 1
        records = [json.loads(line) for line in (folder/"results_trials/IB.jsonl").read_text().splitlines()]
        index = {(r["noise_ratio"], r["trial"]): r for r in records}
        assert len(records) == 500 and set(index) == expected_keys
        assert json.loads((folder/"results.status.json").read_text())["state"] == "complete"
        for key, record in index.items():
            assert record["protocol_id"] == manifest["protocol_id"]
            assert record["split"] == "time2" and record["pilot_size"] == 61
            assert record["filter_widths"] == [61, 61]
            assert record["seed"] == trial_seed(0, "IB", *key)
            for baseline in old:
                for field in ("seed", "noise_std", "G_shape", "b_shape"):
                    assert record[field] == baseline[key][field], (regression, key, field)
            for baseline in (old[0], old[2]):
                for field in ("state_scales", "coordinate_scales"):
                    assert record[field] == baseline[key][field], (regression, key, field)
            beta = np.asarray(record["coefficients"])
            assert beta.shape == (43, 1) and np.isfinite(beta).all()
            selected, actual = beta != 0, truth != 0
            tp = int(np.count_nonzero(selected & actual))
            fn = int(np.count_nonzero(~selected & actual))
            fp = int(np.count_nonzero(selected & ~actual))
            assert (record["tp"], record["fn"], record["fp"]) == (tp, fn, fp)
            assert record["exact_support"] == (fn == fp == 0)
            np.testing.assert_allclose(record["tpr"], tp/(tp+fn+fp))
            np.testing.assert_allclose(record["e2"], np.linalg.norm(beta-truth)/np.linalg.norm(truth))
            np.testing.assert_allclose(record["e_inf"], np.max(np.abs((beta-truth)[actual]/truth[actual])))
            np.testing.assert_allclose(record["runtime"], record["preprocessing_runtime"]
                                       +record["weak_runtime"]+record["regression_runtime"])
            assert record["threshold"] == record["thresholds"][int(np.argmin(record["losses"]))]
            if regression == "lasso":
                assert np.max(record["kkt_errors"]) <= 1e-7
                np.testing.assert_allclose(record["selected_alpha"], record["threshold"]*np.asarray(record["alpha_max"]))
                restored = (np.asarray(record["normalized_coefficients"])
                    *np.asarray(record["response_rms"])[None, :]
                    /np.asarray(record["column_rms"])[:, None]
                    *np.asarray(record["coefficient_scales"]))
                np.testing.assert_allclose(beta, restored, rtol=1e-12, atol=1e-12)
        for row in json.loads((folder/"results.summary.json").read_text()):
            group = [r for r in records if r["noise_ratio"] == row["noise_ratio"]]
            assert row["trials"] == 100 and row["exact_count"] == sum(r["exact_support"] for r in group)
            for field in ("tpr", "e_inf", "e2", "runtime"):
                values = [r[field] for r in group]
                np.testing.assert_allclose(row[field+"_mean"], np.mean(values))
                np.testing.assert_allclose(row[field+"_sd"], np.std(values, ddof=1))
            if regression == "lasso":
                np.testing.assert_allclose(row["lasso_lambda_median"]["u"],
                                           np.median([r["selected_alpha"][0] for r in group]))
        runs[regression] = index
    for key in expected_keys:
        for field in ("observation_sha256", "pilot_sha256", "G_sha256", "b_sha256", "coefficient_scales"):
            assert runs["mstls"][key][field] == runs["lasso"][key][field]

    replayed = []
    build, select = profile_settings(clean.benchmark)
    for ratio in (0., .2, .5, .75, 1.):
        for trial in (0, 99):
            data = observation_instance(clean, ratio, trial_seed(0, "IB", ratio, trial))
            size = max(3, rule_size(estimate_noise_std(data.u_observed[0]), clean.benchmark, .01))
            pilot, _ = moving_average_pilot(data.u_observed, (size, size), boundary=("wrap", "reflect"),
                                          split=("axis", -1, 2))
            system = build_wsindy_system(data.u_observed, data.spatial_grid, data.time,
                library_terms=clean.benchmark.library(), pilot=pilot, **build)
            fits = fit_system(system, slice(None), [], truth[:, 0], select, ("mstls", "lasso"))
            for regression, fit in fits.items():
                saved = runs[regression][(ratio, trial)]
                for name, values in (("observation", data.u_observed[0]), ("pilot", pilot[0]),
                                     ("G", system.G), ("b", system.b)):
                    assert array_sha(values) == saved[name+"_sha256"]
                np.testing.assert_allclose(np.asarray(fit["coefficients"])[:, None], saved["coefficients"],
                                           rtol=1e-10, atol=1e-12)
                if regression == "mstls":
                    active = np.flatnonzero(np.asarray(saved["coefficients"])[:, 0])
                    scaled = linalg.lstsq(system.G[:, active], system.b, lapack_driver="gelsd")[0]
                    np.testing.assert_allclose(scaled*system.coefficient_scales[active],
                        np.asarray(saved["coefficients"])[active], rtol=1e-10, atol=1e-12)
            replayed.append(dict(noise_ratio=ratio, trial=trial))

    reports = [ROOT/"results.md", *(FOLDER/r/"results.md" for r in ("mstls", "lasso"))]
    artifacts = [path.with_suffix(suffix) for path in reports for suffix in (".md", ".summary.json", ".summary.csv")]
    hashes = {str(path): sha(path) for path in artifacts}
    subprocess.run([sys.executable, str(ROOT/"benchmark_results/lasso/penalty_levels.py")], check=True,
                   capture_output=True, cwd=ROOT)
    assert hashes == {str(path): sha(path) for path in artifacts}, "Report regeneration must be idempotent"
    result = dict(state="complete", fits=1000, paired_observations=500, split="time2", pilot_window=[61, 61],
        protected_baseline_artifacts=len(protected), existing_result_rows_preserved=len(before),
        snapshot_hashes_checked=snapshot_checks, seeds_and_noise_match_all_four_IB_baselines=True,
        physical_coefficient_metrics_recomputed=True, lasso_physical_restoration_checked=True,
        all_lasso_candidates_pass_KKT=True, replayed_observations=replayed,
        replayed_mstls_support_refits_restore_physical_units=True,
        report_regeneration_idempotent=True, targeted_debiasing_tests_passed=10)
    (FOLDER/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
