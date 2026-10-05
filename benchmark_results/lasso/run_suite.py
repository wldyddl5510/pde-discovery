"""Run/resume four LASSO protocols and refresh the combined benchmark report."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT/"benchmark_results/lasso/original/raw/results_source"
# Resume the saved numerical implementation even when workspace development
# has advanced independently of these experiments.
sys.path.insert(0, str(SNAPSHOT if SNAPSHOT.is_dir() else ROOT))
from experiments import aggregate_comparisons, parse_arguments, protocol_manifest, refresh_reports

RUNS = [
    ["--benchmarks", "RD", "NS", "NLS", "KS", "KdV", "IB", "--workers", "2",
     "--output", "benchmark_results/lasso/original/raw/results.md"],
    ["--benchmarks", "RD", "NS", "NLS", "KS", "KdV", "IB", "--workers", "2", "--method", "filtered-wsindy",
     "--output", "benchmark_results/lasso/original/filtered/results.md"],
    ["--benchmarks", "HKS", "VBG", "--workers", "2",
     "--output", "benchmark_results/lasso/consistency/raw/results.md"],
    ["--benchmarks", "HKS", "VBG", "--workers", "2", "--method", "filtered-wsindy",
     "--output", "benchmark_results/lasso/consistency/filtered/results.md"],
]
BASELINES = [ROOT/p for p in ["benchmark_results/authors/results.md", "benchmark_results/filtered/results.md",
    "benchmark_results/consistency/raw/results.md", "benchmark_results/consistency/filtered/results.md"]]


def prepare(flags):
    args = parse_arguments(["--regression", "lasso", "--offline", *flags])
    args.output = ROOT/args.output
    args.data_dir = ROOT/"data"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = protocol_manifest(args)
    path = args.output.with_suffix(".manifest.json")
    if path.exists():
        if json.loads(path.read_text())["protocol_id"] != manifest["protocol_id"]:
            raise ValueError(f"Protocol changed: {path}")
    else:
        path.write_text(json.dumps(manifest, indent=2)+"\n")
        snapshot = args.output.parent/"results_source"; snapshot.mkdir()
        for filename in (*manifest["source_sha256"], "requirements.txt", "README.md"):
            shutil.copyfile(ROOT/filename, snapshot/filename)
    raw = args.output.parent/"results_trials"; raw.mkdir(exist_ok=True)
    count = refresh_reports(args, raw)
    if not args.output.with_suffix(".status.json").exists():
        args.output.with_suffix(".status.json").write_text(json.dumps(dict(state="pending", completed=count,
            requested=len(args.benchmarks)*len(args.noise_ratios)*args.trials), indent=2)+"\n")
    return args


def main():
    os.chdir(ROOT)
    args = [prepare(flags) for flags in RUNS]
    reports = [*BASELINES, *(a.output for a in args)]
    aggregate_comparisons(reports, ROOT/"results.md")
    processes = []
    try:
        for phase in (range(4),):
            processes = []
            for i in phase:
                log = (args[i].output.parent/"run.log").open("a")
                source = args[i].output.parent/"results_source"/"experiments.py"
                command = [sys.executable, str(source), "--regression", "lasso", "--offline",
                    "--resume", "--data-dir", str(ROOT/"data"), *RUNS[i]]
                p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                (args[i].output.parent/"run.pid").write_text(str(p.pid)+"\n")
                processes.append((p, log))
            while any(p.poll() is None for p, _ in processes):
                for p, _ in processes:
                    if p.poll() not in (None, 0):
                        raise RuntimeError(f"LASSO process {p.pid} failed: {p.returncode}")
                aggregate_comparisons(reports, ROOT/"results.md")
                counts = [json.loads(a.output.with_suffix(".status.json").read_text())["completed"] for a in args]
                print(datetime.now(timezone.utc).isoformat(), "counts", counts, flush=True)
                time.sleep(30)
            for p, log in processes:
                log.close()
                if p.returncode:
                    raise RuntimeError(f"LASSO process {p.pid} failed: {p.returncode}")
        assert aggregate_comparisons(reports, ROOT/"results.md")
        subprocess.run([sys.executable, str(ROOT/"benchmark_results/lasso/penalty_levels.py")],
                       cwd=ROOT, check=True)
        print("All 8,000 LASSO fits and combined report complete.", flush=True)
    finally:
        for p, log in processes:
            if p.poll() is None:
                p.terminate()
            log.close()


if __name__ == "__main__":
    main()
