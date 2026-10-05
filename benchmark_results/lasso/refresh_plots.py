"""Label completed LASSO plots explicitly without changing frozen fit sources."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"benchmark_results/lasso/original/raw/results_source"))
from experiments import METHOD_LABELS, parse_arguments, plot_results


def main():
    METHOD_LABELS.update({"wsindy": "WSINDy + LASSO", "filtered-wsindy": "Filtered WSINDy + LASSO"})
    for report in (ROOT/"benchmark_results/lasso").glob("*/*/results.md"):
        status = report.with_suffix(".status.json")
        if not status.exists() or json.loads(status.read_text())["state"] != "complete":
            continue
        manifest = json.loads(report.with_suffix(".manifest.json").read_text())
        args = parse_arguments(["--benchmarks", *manifest["benchmarks"], "--method", manifest["method"],
            "--regression", "lasso", "--output", str(report)])
        plot_results(args)
        print("LASSO plot labels refreshed:", report, flush=True)


if __name__ == "__main__":
    main()
