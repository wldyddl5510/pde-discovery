"""Verify completed fold-local time2 reports without rerunning the experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/"benchmark_results/debiasing/time2_foldwise"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    while True:
        status=json.loads((FOLDER/"suite.status.json").read_text())
        if status["state"]=="complete": break
        if status["state"]=="failed" or not args.wait:
            raise RuntimeError(f"Suite is not complete: {status}")
        time.sleep(30)
    protected=json.loads((FOLDER/"baseline.sha256.json").read_text())
    for filename,expected in protected.items():
        assert sha(ROOT/filename)==expected, filename
    before=json.loads((FOLDER/"preexisting_results.summary.json").read_text())
    old=[r for r in before if r["method"]!="debiased-wsindy"]
    merged=json.loads((ROOT/"results.summary.json").read_text())
    assert [r for r in merged if r["method"]!="debiased-wsindy"]==old
    assert len(merged)==240
    checked=0; source_hashes=0; windows={}; recovery={}
    for name in status["completed_benchmarks"]:
        preflight=json.loads((FOLDER/name/"preflight.json").read_text())
        assert preflight["state"]=="complete" and preflight["raw_baselines_reproduced"]
        audit=json.loads((FOLDER/name/"verification.json").read_text())
        assert audit["state"]=="complete" and audit["fits"]==1000
        pair={}
        for regression in ("mstls","lasso"):
            folder=FOLDER/name/regression
            manifest=json.loads((folder/"results.manifest.json").read_text())
            for filename,expected in manifest["source_sha256"].items():
                assert sha(folder/"results_source"/filename)==expected
                assert sha(FOLDER/"source"/filename)==expected
                source_hashes+=1
            spec=manifest["benchmarks"][name]
            rows=[json.loads(line) for line in (folder/"results_trials"/(name+".jsonl")).read_text().splitlines()]
            index={(r["noise_ratio"],r["trial"]):r for r in rows}
            assert len(rows)==len(index)==500
            assert set(index)=={(noise,i) for noise in (0.,.2,.5,.75,1.) for i in range(100)}
            assert all(r["protocol_id"]==manifest["protocol_id"] and r["split"]=="time2" for r in rows)
            anchor=index[(.5,0)]
            assert anchor["G_sha256"]==preflight["G_sha256"] and anchor["b_sha256"]==preflight["b_sha256"]
            summaries=json.loads((folder/"results.summary.json").read_text())
            assert len(summaries)==5
            for row in summaries:
                group=[r for r in rows if r["noise_ratio"]==row["noise_ratio"]]
                assert len(group)==row["trials"]==100
                assert row["exact_count"]==sum(r["exact_support"] for r in group)
                for field in ("tpr","e_inf","e2","runtime"):
                    values=np.array([r[field] for r in group])
                    np.testing.assert_allclose(row[field+"_mean"],values.mean())
                    np.testing.assert_allclose(row[field+"_sd"],values.std(ddof=1))
                    np.testing.assert_allclose(row[field+"_median"],np.median(values))
                if regression=="lasso":
                    for i,c in enumerate(spec["lhs_components"]):
                        label=spec["component_names"][c]
                        values=[r["selected_alpha"][i] for r in group]
                        for suffix,function in (("min",min),("median",np.median),("max",max)):
                            np.testing.assert_allclose(row["lasso_lambda_"+suffix][label],function(values))
                    assert max(float(np.max(r["kkt_errors"])) for r in group)<=1e-7
                counterpart=next(r for r in merged if r["name"]==name and r["noise_ratio"]==row["noise_ratio"]
                                 and r["method"]=="debiased-wsindy" and r["regression"]==regression)
                assert counterpart==row
            recovery[f"{name}/{regression}"]={str(r["noise_ratio"]):r["exact_count"] for r in summaries}
            checked+=len(rows); pair[regression]=index
        for key,a in pair["mstls"].items():
            b=pair["lasso"][key]
            for field in ("seed","noise_std","pilot_info","observation_sha256","pilot_sha256","G_sha256","b_sha256"):
                assert a[field]==b[field]
        windows[name]={str(noise):sorted({tuple(f["window"]) for r in pair["mstls"].values()
            if r["noise_ratio"]==noise for f in r["pilot_info"]["folds"]}) for noise in (0.,.2,.5,.75,1.)}
    assert checked==8000
    report_paths=[ROOT/"results.md",*FOLDER.glob("*/*/results.md")]
    artifacts=[p.with_suffix(s) for p in report_paths for s in (".md",".summary.json",".summary.csv")]
    hashes={str(p):sha(p) for p in artifacts}
    subprocess.run([sys.executable,str(ROOT/"benchmark_results/lasso/penalty_levels.py")],check=True,
                   cwd=ROOT,capture_output=True)
    assert hashes=={str(p):sha(p) for p in artifacts}, "Reports must regenerate identically"
    result=dict(state="complete",fits=checked,paired_observations=4000,benchmarks=status["completed_benchmarks"],
        protected_artifacts_unchanged=len(protected),original_result_rows_preserved=len(old),
        compared_result_rows=len(merged),source_hashes_verified=source_hashes,summary_statistics_verified=True,
        all_lasso_candidates_pass_KKT=True,lambda_statistics_verified=True,report_regeneration_idempotent=True,
        actual_windows=windows,exact_recovery_counts=recovery,completed_at=datetime.now(timezone.utc).isoformat())
    (FOLDER/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
