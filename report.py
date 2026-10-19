"""Comparison tables from saved trial records (no fitting).

    python report.py benchmark_results -o results.md --plot results.png --pdes IB KdV KS NLS RD NS \
        --include benchmark_results/ib_seeds/IB_raw_noise1.md

Every <PDE>.jsonl below the given paths is read, including runs saved by earlier
versions. Each row summarizes all trials of one method; trials shared by several
methods must have identical noise seeds.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import numpy as np
from pdes import PDES, load

ORDER = ("wsindy", "filtered", "debiased", "crossfit-debiased", "lasso", "filtered-lasso", "debiased-lasso",
         "crossfit-debiased-lasso", "best-subset", "filtered-best-subset", "debiased-best-subset", "oracle-debiased",
         "oracle-debiased-lasso", "oracle-debiased-best-subset", "eiv", "eiv-rownorm", "eiv-formula", "wsindy-dantzig",
         "dantzig", "oracle-plugin-dantzig", "oracle-dantzig")
# Earlier debiased runs pooled a time-parity cross-fit pilot with both parities in every integral.
LEGACY = {("wsindy", "lasso"): "lasso", ("filtered-wsindy", "mstls"): "filtered",
          ("filtered-wsindy", "lasso"): "filtered-lasso", ("debiased-wsindy", "mstls"): "crossfit-debiased",
          ("debiased-wsindy", "lasso"): "crossfit-debiased-lasso"}


# Saved runs left out of reports unless named with --methods: earlier pooled cross-fit debiasing,
# row-normalized EIV, the EIV sanity study with the removed tolerance formula, the oracle-pilot sanity check
# and the Dantzig selectors with the oracle lambda.
EXCLUDED = ("crossfit-debiased", "crossfit-debiased-lasso", "eiv-rownorm", "eiv-formula",
            "oracle-debiased", "oracle-debiased-lasso", "oracle-debiased-best-subset",
            "wsindy-dantzig", "dantzig", "oracle-plugin-dantzig", "oracle-dantzig")


def method_of(record):
    if "method" not in record and "mu" in record:   # earlier EIV sanity study (removed tolerance formula)
        return "eiv-formula"
    method = record.get("method", "wsindy")
    if record.get("estimator", {}).get("split") == "time2":   # pooled cross-fit, saved under the debiased names
        return "crossfit-"+method
    return LEGACY.get((method, record.get("regression", "mstls")), method.removeprefix("split-"))


def read(paths):
    runs = defaultdict(dict)   # (pde, method) -> {(noise ratio, trial): record}
    for path in paths:
        root = Path(path)
        isolated = [marker.parent for marker in root.rglob("report.scope.json")
                    if marker.parent != root and not json.loads(marker.read_text()).get("include_when_scanning_parent", True)]
        for file in sorted(root.rglob("*.jsonl")):
            if any(folder in file.parents for folder in isolated):
                continue
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


def _seeds(rows):
    """Number of noise seeds (trials) per noise level, for every method and PDE."""
    names = list(dict.fromkeys(r["name"] for r in rows))
    methods = sorted({r["method"] for r in rows}, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))
    ratios = sorted({r["noise_ratio"] for r in rows})
    def cell(method, name):
        counts = [r["trials"] for r in rows if r["method"] == method and r["name"] == name]
        if not counts:
            return "—"
        return f"{counts[0]}" if len(set(counts)) == 1 and len(counts) == len(ratios) else \
               "/".join(str(next((r["trials"] for r in rows if r["method"] == method and r["name"] == name
                                  and r["noise_ratio"] == q), 0)) for q in ratios)
    return ["## Seeds", "",
            f"Noise seeds (trials) per noise level; noise ratios {', '.join(f'{q:g}' for q in ratios)}. "
            "Trial i uses the same seed for every method (pdes.trial_seed, root seed 0), so methods are "
            "compared on identical observations. At noise ratio 0 all trials coincide.", "",
            "| Method | " + " | ".join(names) + " |", "| :--- |" + " ---: |"*len(names),
            *[f"| {m} | " + " | ".join(cell(m, n) for n in names) + " |" for m in methods], ""]


def _structure(pde, terms):
    """Shapes of the weak regression Y = X beta and the true sparsity, as a small table."""
    U, grid = load(pde.name, download=False)
    K = int(np.prod([len(range(m, len(g)-m, s)) for g, m, s in zip(grid, pde.tests.m, pde.tests.s)]))
    J, E = len(terms), len(pde.lhs)
    sparsity = [len(equation) for equation in pde.truth]
    per_equation = ", ".join(f"{lhs}: {r}" for lhs, r in zip(pde.lhs, sparsity))
    return ["| Data U | X (K x J) | Y (K x E) | beta (J x E) | True nonzeros |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| {' x '.join(map(str, U.shape[:-1]))} grid; components {', '.join(pde.components)} "
            f"| {K} x {J} | {K} x {E} | {J} x {E} | {sum(sparsity)} of {J*E} ({per_equation}) |", "",
            f"K = weak equations (test-function centers), J = dictionary terms, E = equations. "
            f"Test functions `{pde.tests}`."]


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
        lines += [f"## {name}", "", *[f"`{eq}`  " for eq in pde.equations()], "", *_structure(pde, terms), ""]
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


def summary(rows):
    """Exact-support rate and mean E_inf per PDE, method and noise ratio."""
    ratios = sorted({r["noise_ratio"] for r in rows})
    lines = ["## Summary", "", "Exact-support rate (%) · mean E_inf, per noise ratio.", "",
             "| PDE | Method | "+" | ".join(f"{x:g}" for x in ratios)+" |", "| :--- | :--- |"+" ---: |"*len(ratios)]
    cells = {(r["name"], r["method"], r["noise_ratio"]): f"{100*r['exact_rate']:.0f} · {r['e_inf_mean']:.2g}" for r in rows}
    for name, method in dict.fromkeys((r["name"], r["method"]) for r in rows):
        lines.append(f"| {name} | {method} | "+" | ".join(cells.get((name, method, x), "—") for x in ratios)+" |")
    return lines+[""]


def _style(method):
    """Color: state handling (raw, filtered, debiased, cross-fit); line: regression."""
    state = next((s for s in ("crossfit", "filtered", "debiased", "eiv", "oracle") if method.startswith(s)), method)
    color = {"crossfit": "C3", "filtered": "C1", "debiased": "C2", "eiv": "C4", "sindy": "C5", "oracle": "C6"}.get(state, "C0")
    line = ("--" if method.endswith("lasso") else ":" if method.endswith("best-subset")
            else "-." if method.endswith("rownorm") else "-")
    return dict(color=color, linestyle=line, marker=".", label=method)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r["name"] for r in rows))
    figure, axes = plt.subplots(3, len(names), figsize=(2.6*len(names), 8.4), squeeze=False, constrained_layout=True)
    for column, name in enumerate(names):
        for method in dict.fromkeys(r["method"] for r in rows if r["name"] == name):
            group = [r for r in rows if r["name"] == name and r["method"] == method]
            x = [r["noise_ratio"] for r in group]
            axes[0, column].plot(x, [r["exact_rate"] for r in group], **_style(method))
            axes[1, column].plot(x, [max(r["e2_mean"], 1e-12) for r in group], **_style(method))
            axes[2, column].plot(x, [max(r["e_inf_mean"], 1e-12) for r in group], **_style(method))
        axes[0, column].set(title=name, ylim=(-.03, 1.03))
        axes[1, column].set(yscale="log")
        axes[2, column].set(yscale="log", xlabel="noise ratio")
    axes[0, 0].set_ylabel("exact support rate")
    axes[1, 0].set_ylabel("mean E2")
    axes[2, 0].set_ylabel("mean E_inf")
    handles = {h.get_label(): h for ax in axes.flat for h in ax.get_lines()}
    figure.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=min(7, len(handles)), fontsize=8)
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("results.md"))
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--methods", nargs="+", help="Only these methods (default: all found)")
    parser.add_argument("--pdes", nargs="+", help="Only these PDEs in tables and plots (default: all found)")
    parser.add_argument("--plot-pdes", nargs="+", help="Only these PDEs in the plot (default: all in the tables)")
    parser.add_argument("--include", nargs="+", type=Path, default=[],
                        help="Markdown sections inserted after the summary (e.g. benchmark_results/ib_seeds/*.md)")
    args = parser.parse_args()
    runs = {key: run for key, run in read(args.paths).items()
            if (key[1] in args.methods if args.methods else key[1] not in EXCLUDED)
            and (not args.pdes or key[0] in args.pdes)}
    lines, rows = tables(runs)
    header = ["# WSINDy benchmark", "",
              "Methods are defined in `run.py` (METHODS). TPR = TP/(TP+FN+FP); Exact = exact-support rate; "
              "E_inf = max relative error over true terms; E2 = relative coefficient L2 error; mean ± sample SD "
              "over trials, failures included. Seconds cover estimation, weak system and regression. "
              "Mean lambda / r: mean selected LASSO penalty, or best-subset sparsity bound r, per equation.", ""]
    figure = [f"![Exact support rate and mean E2]({os.path.relpath(args.plot, args.output.parent)})", ""] if args.plot else []
    included = [path.read_text().rstrip("\n")+"\n" for path in args.include]
    args.output.write_text("\n".join(header+_seeds(rows)+figure+summary(rows)+included+lines))
    with args.output.with_suffix(".summary.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        plot([r for r in rows if not args.plot_pdes or r["name"] in args.plot_pdes], args.plot)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
