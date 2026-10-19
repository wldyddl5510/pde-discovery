"""Plot saved oracle-pilot results without fitting or changing numerical caches.

The main figure compares cardinality-certified finite-grid oracle choices.  A
second figure separates the three solver-certificate levels for each oracle
pilot.  Only completed paired realizations are plotted, and partial counts are
visible.  Physical coefficient errors are never clipped to a plotting floor.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"pde-oracle-pilot-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from run_study import load_records, select


METHODS = ("wsindy", "debiased_dantzig", "rate_plugin_dantzig", "rate_debiased_dantzig",
           "exact_plugin_dantzig", "exact_debiased_dantzig")
LABELS = {
    "wsindy": "Native WSINDy / MSTLS",
    "debiased_dantzig": "Original pilot: debiased Dantzig",
    "rate_plugin_dantzig": "Rate oracle: plugin Dantzig",
    "rate_debiased_dantzig": "Rate oracle: debiased Dantzig",
    "exact_plugin_dantzig": "Exact oracle: plugin Dantzig",
    "exact_debiased_dantzig": "Exact oracle: debiased Dantzig",
}
STYLES = {
    "wsindy": dict(color="#526779", marker="x", linestyle="-", linewidth=1.9, markersize=6.),
    "debiased_dantzig": dict(color="#bb7622", marker="v", linestyle=":", linewidth=1.7, markersize=6.),
    "rate_plugin_dantzig": dict(color="#45a99c", marker="^", linestyle="--", linewidth=1.7, markersize=6.5),
    "rate_debiased_dantzig": dict(color="#007e69", marker="o", linestyle="-", linewidth=2.1, markersize=5.),
    "exact_plugin_dantzig": dict(color="#ac7bc4", marker="D", linestyle="--", linewidth=1.7, markersize=6.5),
    "exact_debiased_dantzig": dict(color="#744298", marker="s", linestyle="-", linewidth=2.1, markersize=4.7),
}
LEVELS = ("lexicographic", "cardinality", "incumbent")
LEVEL_LABELS = {"wsindy": LABELS["wsindy"],
    "lexicographic": "Dantzig: cardinality + residual tie-break certified",
    "cardinality": "Dantzig: cardinality certified (Equation 9)",
    "incumbent": "Dantzig: any validated incumbent"}
LEVEL_STYLES = {"wsindy": STYLES["wsindy"],
    "lexicographic": dict(color="#007e69", marker="o", linestyle="-", linewidth=2.2, markersize=6.7),
    "cardinality": dict(color="#4665b5", marker="D", linestyle="--", linewidth=1.7, markersize=5.2),
    "incumbent": dict(color="#bb7622", marker="^", linestyle=":", linewidth=1.7, markersize=4.5)}


def _snapshot(directory):
    records = load_records(directory)
    if not records:
        raise RuntimeError("No completed paired realizations in trials.jsonl")
    manifest = json.loads((directory/"manifest.json").read_text())
    names, noises = manifest["pdes"], sorted(manifest["noise"])
    expected = manifest["expected_realizations"]
    expected_per_pde = sum(1 if noise == 0 else manifest["trials"] for noise in noises)
    grouped, values = defaultdict(list), {}
    keys = set()
    for record in records:
        key = (record["name"], record["noise_ratio"], record["trial"])
        if key in keys:
            raise ValueError(f"Duplicate paired realization: {key}")
        keys.add(key)
        grouped[key[:2]].append(record)
        for method in METHODS:
            levels = ("cardinality",) if method not in ("rate_debiased_dantzig", "exact_debiased_dantzig") else LEVELS
            for level in levels:
                error = float(select(record, method, level)["e2"])
                if not np.isfinite(error) or error < 0:
                    raise ValueError(f"Invalid physical coefficient E2: {key}, {method}, {level}: {error}")
                values[key+(method, level)] = error
    return records, names, noises, expected, expected_per_pde, grouped, values


def _series(name, noises, grouped, values, method, level):
    x, y = [], []
    for noise in noises:
        rows = grouped[(name, noise)]
        if not rows:
            continue
        x.append(noise)
        y.append(float(np.mean([values[(name, noise, r["trial"], method, level)] for r in rows])))
    return x, y


def _axis(ax, title, noises, all_values):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
    # Retain exact zero using symlog rather than changing it to a positive floor.
    positives = [v for v in all_values if v > 0]
    if all_values and min(all_values) == 0:
        threshold = min(positives)/10 if positives else 1e-12
        ax.set_yscale("symlog", linthresh=threshold)
        ax.set_ylim(-threshold/10, max(all_values+[threshold])*2.2)
        ax.text(.02, .02, "symlog: exact zero retained", transform=ax.transAxes,
                fontsize=7.5, color="#555555")
    else:
        ax.set_yscale("log")
        if all_values:
            ax.set_ylim(min(all_values)/2.2, max(all_values)*2.2)
        else:
            ax.set_ylim(.1, 10)
            ax.text(.5, .5, "No completed realizations", ha="center", va="center",
                    transform=ax.transAxes, color="#777777", fontsize=10)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1., color="#999999", linestyle=(0, (2, 3)), linewidth=.8, zorder=0)
    ax.set_xlim(min(noises)-.035, max(noises)+.035)
    ax.set_xticks(noises)
    ax.set_xticklabels([f"{noise:g}" for noise in noises])
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.7)
    ax.grid(axis="x", color="#ededed", linewidth=.6)


def _line(ax, x, y, label, style):
    return ax.plot(x, y, label=label, markerfacecolor="white", markeredgewidth=1.3, **style)[0]


def plot(directory):
    directory = Path(directory)
    records, names, noises, expected, per_pde, grouped, values = _snapshot(directory)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    status = "Complete" if len(records) == expected else "Partial"
    snapshot = f"{status} snapshot: {len(records)}/{expected} completed paired realizations"
    minimum = min(values.values())
    panel_rows = (len(names)+3)//4
    fig, axes = plt.subplots(panel_rows, 4, figsize=(19, 4.5*panel_rows+1.6), squeeze=False)
    handles = {}
    for ax, name in zip(axes.flat, names):
        count = sum(len(grouped[(name, noise)]) for noise in noises)
        all_values = []
        for method in METHODS:
            x, y = _series(name, noises, grouped, values, method, "cardinality")
            if x:
                handles[method] = _line(ax, x, y, LABELS[method], STYLES[method])
                all_values.extend(y)
        _axis(ax, f"{name}  ({count}/{per_pde})", noises, all_values)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Observation noise ratio")
    fig.supylabel("Mean relative coefficient $E_2$  (physical units)", x=.012, fontsize=14)
    fig.suptitle("Oracle-pilot Dantzig: coefficient recovery", y=.975, fontsize=20, fontweight="bold")
    fig.text(.5, .928, snapshot+"  |  All Dantzig curves use cardinality-certified oracle choices",
             ha="center", fontsize=12, color="#444444")
    ordered = [method for method in METHODS if method in handles]
    fig.legend([handles[m] for m in ordered], [LABELS[m] for m in ordered],
               loc="lower center", bbox_to_anchor=(.5, .063), ncol=3, frameon=False,
               columnspacing=2., handlelength=3., fontsize=10.5)
    fig.text(.5, .024,
        "Rate oracle: weighted squared pilot error $\\leq n_{reg}^{-1/2}$; exact oracle: zero pilot error. "
        "True coefficients choose $\\lambda$ within the saved finite candidate grid, equation by equation.\n"
        "Means use completed paired seeds only; at completion: one noiseless run and three noise draws per nonzero level. "
        "Residual tie-break may be unresolved. The zero coefficient model has $E_2=1$.",
        ha="center", va="center", fontsize=9.5, color="#555555", linespacing=1.6)
    fig.subplots_adjust(left=.067, right=.985, bottom=.195, top=.85, wspace=.31, hspace=.38)
    main_path = directory/"recovery.png"
    fig.savefig(main_path, dpi=170, facecolor="white", metadata={"Description":
        snapshot+"; cardinality-certified finite-grid oracle choices; unclipped physical coefficient E2."})
    plt.close(fig)

    fig, axes = plt.subplots(2*panel_rows, 4, figsize=(19, 3.6*2*panel_rows+2.1), squeeze=False)
    handles = {}
    for mode_index, mode in enumerate(("rate", "exact")):
        for index, name in enumerate(names):
            ax = axes[mode_index*panel_rows+index//4, index % 4]
            count = sum(len(grouped[(name, noise)]) for noise in noises)
            all_values = []
            for level in ("wsindy", *LEVELS):
                method = "wsindy" if level == "wsindy" else f"{mode}_debiased_dantzig"
                candidate_level = "cardinality" if level == "wsindy" else level
                x, y = _series(name, noises, grouped, values, method, candidate_level)
                if x:
                    handles[level] = _line(ax, x, y, LEVEL_LABELS[level], LEVEL_STYLES[level])
                    all_values.extend(y)
            _axis(ax, f"{name} — {mode} oracle  ({count}/{per_pde})", noises, all_values)
        for index in range(len(names), panel_rows*4):
            axes[mode_index*panel_rows+index//4, index % 4].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Observation noise ratio")
    fig.supylabel("Mean relative coefficient $E_2$  (physical units)", x=.012, fontsize=14)
    fig.suptitle("Debiased score: oracle-pilot recovery and solver certificates", y=.98,
                 fontsize=20, fontweight="bold")
    fig.text(.5, .947, snapshot+"  |  Rate oracle in the upper half; exact oracle in the lower half",
             ha="center", fontsize=12, color="#444444")
    ordered = [level for level in ("wsindy", *LEVELS) if level in handles]
    fig.legend([handles[level] for level in ordered], [LEVEL_LABELS[level] for level in ordered],
               loc="lower center", bbox_to_anchor=(.5, .035), ncol=2, frameon=False,
               columnspacing=2.8, handlelength=3.2, fontsize=11)
    fig.text(.5, .018,
        "Oracle $\\lambda$ is selected separately within each candidate set. Incumbents are physically feasible; "
        "they need not minimize cardinality and serve only as an optimization diagnostic.",
        ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.067, right=.985, bottom=.115, top=.91, wspace=.31, hspace=.40)
    solver_path = directory/"solver_levels.png"
    fig.savefig(solver_path, dpi=170, facecolor="white", metadata={"Description":
        snapshot+"; rate/exact debiased Dantzig solver-certificate comparison; unclipped physical E2."})
    plt.close(fig)
    print(f"Wrote {main_path} and {solver_path}: {len(records)}/{expected} paired realizations; "
          f"minimum individual physical E2={minimum:.12g}")
    return main_path, solver_path


def plot_with_proof(directory):
    """Plot the verified supplemental snapshot without changing raw eligibility.

    Every method uses the same saved supplemental realization set.  The only
    expanded pool is exact-plugin NLS/VBG, as specified and hash-checked by
    supplement.py; raw Gurobi figures and flags remain separate.
    """
    directory = Path(directory)
    payload = json.loads((directory/"supplement.json").read_text())
    if payload["protocol"] != "independent-proof-augmented-cardinality-oracle-report-v1":
        raise ValueError("Unknown independent-proof supplement protocol")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    provenance = payload["provenance"]
    if digest(directory/"manifest.json") != provenance["manifest_sha256"]:
        raise ValueError("Proof supplement manifest changed; regenerate supplement first")
    if digest(directory/"supplement.py") != provenance["report_source_sha256"]:
        raise ValueError("Proof supplement source changed; regenerate supplement first")
    for name, proof in provenance["independent_proofs"].items():
        path = directory/f"{name.lower()}_certificate_verification.json"
        verification = json.loads(path.read_text())
        if digest(path) != proof["verification_sha256"] or not verification["passed"]:
            raise ValueError(f"Independent {name} verification changed or did not pass")
    records = payload["selections"]
    if len(records) != payload["completed_realizations"] or not records:
        raise ValueError("Invalid supplemental realization count")
    manifest = json.loads((directory/"manifest.json").read_text())
    names, noises = manifest["pdes"], sorted(manifest["noise"])
    expected = payload["expected_realizations"]
    per_pde = sum(1 if noise == 0 else manifest["trials"] for noise in noises)
    grouped, values, keys = defaultdict(list), {}, set()
    for record in records:
        key = (record["name"], record["noise_ratio"], record["trial"])
        if key in keys or key not in {tuple(job) for job in manifest["scope"]}:
            raise ValueError(f"Duplicate or out-of-protocol supplemental realization: {key}")
        keys.add(key)
        grouped[key[:2]].append(record)
        for method in METHODS:
            policy = "with_independent_proof" if method == "exact_plugin_dantzig" else "gurobi_cardinality"
            value = float(record["methods"][method][policy]["e2"])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid supplemental E2: {key}, {method}: {value}")
            values[key+(method,"cardinality")] = value
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.labelsize":11,"xtick.labelsize":9,"ytick.labelsize":9,
        "axes.spines.top":False,"axes.spines.right":False})
    labels = dict(LABELS,exact_plugin_dantzig="Exact plugin: Gurobi + independent proofs")
    status = "Complete" if len(records) == expected else "Partial"
    rows = (len(names)+3)//4
    fig,axes = plt.subplots(rows,4,figsize=(19,4.5*rows+1.6),squeeze=False)
    handles = {}
    for ax,name in zip(axes.flat,names):
        count = sum(len(grouped[(name,noise)]) for noise in noises)
        all_values = []
        for method in METHODS:
            x,y = _series(name,noises,grouped,values,method,"cardinality")
            if x:
                handles[method] = _line(ax,x,y,labels[method],STYLES[method])
                all_values.extend(y)
        _axis(ax,f"{name}  ({count}/{per_pde})",noises,all_values)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Observation noise ratio")
    fig.supylabel("Mean relative coefficient $E_2$  (physical units)",x=.012,fontsize=14)
    fig.suptitle("Oracle-pilot recovery with independent cardinality proofs",y=.975,
                 fontsize=20,fontweight="bold")
    fig.text(.5,.928,f"{status} snapshot: {len(records)}/{expected} paired realizations  |  "
        "Exact plugin adds verified NLS/VBG proofs; other Dantzig curves use Gurobi cardinality certificates",
        ha="center",fontsize=11.5,color="#444444")
    ordered = [method for method in METHODS if method in handles]
    fig.legend([handles[method] for method in ordered],[labels[method] for method in ordered],
        loc="lower center",bbox_to_anchor=(.5,.063),ncol=3,frameon=False,
        columnspacing=2.,handlelength=3.,fontsize=10.5)
    fig.text(.5,.024,
        "Proofs certify Equation (9) minimum cardinality; the secondary residual tie-break is not established. "
        "True coefficients choose $\\lambda$ within each saved certified candidate pool.\n"
        "Rate pilot squared error $\\leq n_{reg}^{-1/2}$; exact pilot error is zero. "
        "One noiseless run and three noise draws per nonzero level at completion; no coefficient error clipping.",
        ha="center",va="center",fontsize=9.5,color="#555555",linespacing=1.6)
    fig.subplots_adjust(left=.067,right=.985,bottom=.195,top=.85,wspace=.31,hspace=.38)
    output = directory/"recovery_with_proof.png"
    fig.savefig(output,dpi=170,facecolor="white",metadata={"Description":
        f"{len(records)}/{expected} supplemental paired realizations; exact-plugin Gurobi plus independently "
        "verified NLS/VBG minimum-cardinality candidates. Bare Equation9, no residual tie-break proof."})
    plt.close(fig)
    print(f"Wrote {output}: {len(records)}/{expected} supplemental paired realizations; "
          f"minimum individual physical E2={min(values.values()):.12g}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", "--output", dest="directory", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--proof-only",action="store_true",
                        help="Only refresh recovery_with_proof.png from the verified supplement snapshot")
    args = parser.parse_args()
    if not args.proof_only:
        plot(args.directory.resolve())
    plot_with_proof(args.directory.resolve())
