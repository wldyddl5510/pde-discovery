"""Finish saved benchmark artifacts once the resumed LASSO runs complete."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNS = [ROOT/"benchmark_results/lasso"/suffix/"results.status.json" for suffix in
        ("original/raw", "original/filtered", "consistency/raw", "consistency/filtered")]
STATUS = ROOT/"benchmark_results/lasso/suite.status.json"


def save(state, runs, error=None):
    data = dict(state=state, completed=sum(r["completed"] for r in runs),
                requested=sum(r["requested"] for r in runs),
                updated_at=datetime.now(timezone.utc).isoformat(), error=error,
                execution_source="Saved numerical snapshots")
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2)+"\n")
    temporary.replace(STATUS)


def main():
    while True:
        runs = [json.loads(path.read_text()) for path in RUNS]
        if all(r["state"] == "complete" and r["completed"] == r["requested"] for r in runs):
            break
        if any(r["state"] in ("failed", "paused", "incomplete") for r in runs):
            save("paused" if any(r["state"] == "paused" for r in runs) else "failed", runs,
                 "A numerical run was stopped or failed; no new fits are launched by this helper")
            return
        save("running", runs)
        time.sleep(30)
    save("verifying", runs)
    try:
        for filename in ("verify_results.py", "refresh_plots.py"):
            subprocess.run([sys.executable, str(ROOT/"benchmark_results/lasso"/filename)],
                           cwd=ROOT, check=True)
        # Refresh final tables using the same source as the saved numerical runs.
        sys.path.insert(0, str(ROOT/"benchmark_results/lasso/original/raw/results_source"))
        from experiments import aggregate_comparisons
        reports = [ROOT/p for p in ("benchmark_results/authors/results.md",
            "benchmark_results/filtered/results.md", "benchmark_results/consistency/raw/results.md",
            "benchmark_results/consistency/filtered/results.md")]
        reports.extend(path.with_name("results.md") for path in RUNS)
        assert aggregate_comparisons(reports, ROOT/"results.md")
        subprocess.run([sys.executable, str(ROOT/"benchmark_results/lasso/penalty_levels.py")],
                       cwd=ROOT, check=True)
        save("complete", runs)
        print("All 8,000 LASSO fits, paired tables, plots and saved-result checks complete.", flush=True)
    except BaseException as error:
        save("failed", runs, f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
