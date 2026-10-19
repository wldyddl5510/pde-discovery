"""Static scientific comparison of three debiased-Dantzig oracle candidate sets.

Only completed paired rows in trials.jsonl are used. Partial snapshots explicitly
show their completed count. The physical coefficient E2 values are plotted
without a lower clipping floor, including the very small noiseless baseline errors.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"pde-oracle-levels-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from interpret import LEVELS, select
from run_study import load_records


LABELS = {
    "wsindy": "WSINDy / native MSTLS tuning",
    "lexicographic": "Dantzig: full lexicographic certificate",
    "cardinality": "Dantzig: cardinality certificate",
    "incumbent": "Dantzig: any validated incumbent",
}
STYLE = {
    "wsindy": dict(color="#526779", marker="o", linestyle="-", linewidth=2., markersize=5.5),
    "lexicographic": dict(color="#008974", marker="o", linestyle="-", linewidth=2.4, markersize=8.),
    "cardinality": dict(color="#4665b5", marker="D", linestyle="--", linewidth=1.9, markersize=5.8),
    "incumbent": dict(color="#bd7423", marker="^", linestyle=":", linewidth=1.9, markersize=4.6),
}


def plot(directory, output):
    records = load_records(directory)
    if not records:
        raise RuntimeError("No completed trials in trials.jsonl")
    manifest = json.loads((directory/"manifest.json").read_text())
    names, noises = manifest["pdes"], manifest["noise"]
    expected_per_pde = sum(1 if noise == 0 else manifest["trials"] for noise in noises)
    expected = len(names)*expected_per_pde
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["name"], record["noise_ratio"])].append(record)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 14, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True)
    handles = {}
    minimum_plotted = np.inf
    for ax, name in zip(axes.flat, names):
        total = sum(len(grouped[(name, noise)]) for noise in noises)
        ax.set_title(f"{name}  ({total}/{expected_per_pde})", loc="left", fontweight="bold")
        all_values = []
        for method in ("wsindy", *LEVELS):
            x, y = [], []
            for noise in noises:
                rows = grouped[(name, noise)]
                if not rows:
                    continue
                values = ([row["methods"]["wsindy"]["e2"] for row in rows] if method == "wsindy"
                          else [select(row, "debiased_dantzig", method)["e2"] for row in rows])
                value = float(np.mean(values))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"Log plot requires finite positive E2: {name}, noise={noise}, {method}: {value}")
                x.append(noise)
                y.append(value)
            if x:
                line, = ax.plot(x, y, label=LABELS[method], markerfacecolor="white",
                                markeredgewidth=1.4, **STYLE[method])
                handles[method] = line
                all_values.extend(y)
        ax.set_yscale("log")
        ax.axhline(1., color="#999999", linestyle=(0, (2, 3)), linewidth=.8, zorder=0)
        if all_values:
            low, high = min(all_values), max(all_values)
            minimum_plotted = min(minimum_plotted, low)
            ax.set_ylim(low/2.2, high*2.2)
        else:
            ax.set_ylim(.1, 10)
            ax.text(.5, .5, "No completed realizations", ha="center", va="center", transform=ax.transAxes, color="#777777")
        ax.set_xlim(min(noises)-.035, max(noises)+.035)
        ax.set_xticks(noises)
        ax.set_xticklabels([f"{noise:g}" for noise in noises])
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", which="major", color="#dddddd", linewidth=.7)
        ax.grid(axis="x", color="#ededed", linewidth=.6)
    for ax in axes[-1]:
        ax.set_xlabel("Observation noise ratio")
    fig.supylabel("Mean relative coefficient $E_2$  (physical units)", x=.012, fontsize=14)
    complete = len(records) == expected
    fig.suptitle("Debiased Dantzig: oracle candidate sets and solver certificates", y=.975, fontsize=20, fontweight="bold")
    fig.text(.5, .927,
             f"{'Complete' if complete else 'Partial'} snapshot: {len(records)}/{expected} completed paired realizations  |  "
             "Oracle tolerance selected within each Dantzig candidate set", ha="center", fontsize=12, color="#444444")
    fig.legend([handles[k] for k in ("wsindy", *LEVELS) if k in handles],
               [LABELS[k] for k in ("wsindy", *LEVELS) if k in handles],
               loc="lower center", bbox_to_anchor=(.5, .064), ncol=2,
               frameon=False, columnspacing=3., handlelength=3.3, fontsize=11)
    fig.text(.5, .025,
             "Means use completed paired seeds only; panel titles give completed / planned. "
             "At completion: one noiseless run and three noise draws per nonzero level.\n"
             "Only the first Dantzig curve requires both optimization stages certified. "
             "Incumbent selection is a feasibility diagnostic. The zero model has $E_2=1$.",
             ha="center", va="center", fontsize=10, color="#555555", linespacing=1.6)
    fig.subplots_adjust(left=.070, right=.985, bottom=.195, top=.855, wspace=.30, hspace=.40)
    fig.savefig(output, dpi=170, facecolor="white", metadata={
        "Description": f"{len(records)}/{expected} completed realizations. Unclipped mean physical E2. "
                       "Debiased Dantzig oracle sets: lexicographic, cardinality, validated incumbent."})
    plt.close(fig)
    print(f"Wrote {output}: {len(records)}/{expected} realizations; minimum plotted mean E2={minimum_plotted:.12g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plot(args.directory.resolve(), args.out or args.directory/"oracle_levels.png")
