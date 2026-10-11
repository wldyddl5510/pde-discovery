"""Comparison tables from saved trial records (no fitting).

    python report.py benchmark_results -o results.md --plot results.png

Every <PDE>.jsonl below the given paths is read, including runs saved by earlier
versions. Each row summarizes all trials of one method; trials shared by several
methods must have identical noise seeds.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from pdes import PDES

ORDER = ("wsindy", "filtered", "debiased", "lasso", "filtered-lasso", "debiased-lasso", "best-subset", "eiv")
LEGACY = {("wsindy", "lasso"): "lasso", ("filtered-wsindy", "mstls"): "filtered",
          ("filtered-wsindy", "lasso"): "filtered-lasso", ("debiased-wsindy", "mstls"): "debiased",
          ("debiased-wsindy", "lasso"): "debiased-lasso"}


def method_of(record):
    if "method" not in record and "mu" in record:   # earlier EIV sanity study
        return "eiv"
    method = record.get("method", "wsindy")
    return LEGACY.get((method, record.get("regression", "mstls")), method)


def read(paths):
    runs = defaultdict(dict)   # (pde, method) -> {(noise ratio, trial): record}
    for path in paths:
        for file in sorted(Path(path).rglob("*.jsonl")):
            if file.stem not in PDES:
                continue
            with file.open() as source:
                for record in map(json.loads, source):
                    run, key = runs[(record["name"], method_of(record))], (record["noise_ratio"], record["trial"])
                    if key in run:
                        raise ValueError(f"Duplicate trial {record['name']}/{method_of(record)}/{key} in {file}")
                    run[key] = record
    return runs


def _cell(values):
    return f"{np.mean(values):.6g} ± {np.std(values, ddof=1):.3g}" if len(values) > 1 else f"{values[0]:.6g}"


def _selection(pde, group):
    """Mean LASSO penalty, or best-subset sparsity r, per equation."""
    names = [lhs.rsplit("_", 1)[0] for lhs in pde.lhs]
    for key, symbol in (("selected_alpha", "lambda"), ("r", "r")):
        if key in group[0]:
            values = np.array([np.broadcast_to(r[key], len(names)) for r in group], dtype=float)
            return ", ".join(f"{symbol}_{n}={np.mean(values[:, e]):.4g}" for e, n in enumerate(names))
    return "—"


def tables(runs):
    lines, rows = [], []
    for name, pde in PDES.items():
        methods = sorted({m for n, m in runs if n == name}, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))
        if not methods:
            continue
        seeds = defaultdict(set)
        for m in methods:
            for key, record in runs[(name, m)].items():
                seeds[key].add(json.dumps(record["seed"]))
        if any(len(s) > 1 for s in seeds.values()):
            raise ValueError(f"Noise seeds differ across methods for {name}")
        terms = pde.dictionary_terms()[1]
        lines += [f"## {name}", "", *[f"`{eq}`  " for eq in pde.equations()], "",
                  f"{len(terms)} dictionary terms; test functions `{pde.tests}`.", ""]
        for ratio in sorted({r for r, _ in seeds}):
            lines += [f"### Noise ratio {ratio:g}", "",
                      "| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |",
                      "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
            for method in methods:
                group = [r for k, r in sorted(runs[(name, method)].items()) if k[0] == ratio]
                if not group:
                    continue
                exact = float(np.mean([r["exact_support"] for r in group]))
                runtime = float(np.mean([r["runtime"] for r in group]))
                lines.append(f"| {method} | {len(group)} | {100*exact:.1f}% | "
                             + " | ".join(_cell([r[f] for r in group]) for f in ("tpr", "e_inf", "e2"))
                             + f" | {runtime:.3g} | {_selection(pde, group)} |")
                rows.append(dict(name=name, noise_ratio=ratio, method=method, trials=len(group), exact_rate=exact,
                                 **{f"{f}_mean": float(np.mean([r[f] for r in group])) for f in ("tpr", "e_inf", "e2")},
                                 runtime_mean=runtime, selection=_selection(pde, group)))
            lines.append("")
    return lines, rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r["name"] for r in rows))
    figure, axes = plt.subplots(2, len(names), figsize=(3*len(names), 5.5), squeeze=False, constrained_layout=True)
    for column, name in enumerate(names):
        for method in dict.fromkeys(r["method"] for r in rows if r["name"] == name):
            group = [r for r in rows if r["name"] == name and r["method"] == method]
            x = [r["noise_ratio"] for r in group]
            axes[0, column].plot(x, [r["exact_rate"] for r in group], marker=".", label=method)
            axes[1, column].plot(x, [max(r["e2_mean"], 1e-12) for r in group], marker=".")
        axes[0, column].set(title=name, ylim=(-.03, 1.03))
        axes[1, column].set(yscale="log", xlabel="noise ratio")
    axes[0, 0].set_ylabel("exact support rate")
    axes[1, 0].set_ylabel("mean E2")
    axes[0, 0].legend(fontsize=7)
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("results.md"))
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--methods", nargs="+", help="Only these methods (default: all found)")
    args = parser.parse_args()
    runs = {key: run for key, run in read(args.paths).items() if not args.methods or key[1] in args.methods}
    lines, rows = tables(runs)
    header = ["# WSINDy benchmark", "",
              "Methods are defined in `run.py` (METHODS). TPR = TP/(TP+FN+FP); Exact = exact-support rate; "
              "E_inf = max relative error over true terms; E2 = relative coefficient L2 error; mean ± sample SD "
              "over trials, failures included. Seconds cover estimation, weak system and regression. "
              "Mean lambda / r: mean selected LASSO penalty, or best-subset sparsity bound r, per equation.", ""]
    args.output.write_text("\n".join(header+lines))
    with args.output.with_suffix(".summary.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        plot(rows, args.plot)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
