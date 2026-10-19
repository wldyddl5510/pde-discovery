"""Additional descriptive diagnostics for the completed oracle sanity study."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

import run_study as study


LEVELS = ("lexicographic", "cardinality", "incumbent")


def baseline_summary(rows, method):
    values = [row["methods"][method] for row in rows]
    return dict(realizations=len(rows), mean_e2=float(np.mean([v["e2"] for v in values])),
        median_e2=float(np.median([v["e2"] for v in values])),
        exact_support=sum(v["exact_support"] for v in values),
        selected_all_zero=sum(not np.any(v["coefficients"]) for v in values),
        mean_fp=float(np.mean([v["fp"] for v in values])), mean_fn=float(np.mean([v["fn"] for v in values])),
        mean_runtime=float(np.mean([v["runtime"] for v in values])))


def select(record, method, level):
    value = record["methods"][method]
    selected = []
    for path in value["paths"]:
        if level == "lexicographic":
            candidate = path["selected"]
        else:
            eligible = [c for c in path["path"] if c.get("validated", False)
                        and (level == "incumbent" or c.get("cardinality_optimal", False))]
            if not eligible:
                raise RuntimeError("Analytic zero candidate should make every oracle level nonempty")
            candidate = min(eligible, key=lambda c: (c["e2"], c["lambda_ratio"]))
        selected.append(candidate)
    result = study.evaluate([c["coefficients"] for c in selected], np.asarray(record["truth"]))
    result.update(selected=selected,
        selected_secondary_unresolved=sum(not c.get("secondary_optimal", False) for c in selected),
        selected_cardinality_unresolved=sum(not c.get("cardinality_optimal", False) for c in selected))
    return result


def describe(rows, method, level="lexicographic"):
    chosen = [select(r, method, level) for r in rows]
    candidates = [c for row in rows for p in row["methods"][method]["paths"] for c in p["path"]]
    nonanalytic = [c for c in candidates if c.get("status") != "AnalyticZero"]
    zeros = [not np.any(np.asarray(value["coefficients"])) for value in chosen]
    wins = [value["e2"] < row["methods"]["wsindy"]["e2"] for row, value in zip(rows, chosen)]
    return dict(realizations=len(rows), mean_e2=float(np.mean([v["e2"] for v in chosen])),
        median_e2=float(np.median([v["e2"] for v in chosen])), selected_all_zero=sum(zeros),
        selected_zero_equations=sum(not np.any(equation) for v in chosen for equation in v["coefficients"]),
        selected_equations=sum(len(v["coefficients"]) for v in chosen),
        selected_secondary_unresolved=sum(v["selected_secondary_unresolved"] for v in chosen),
        selected_cardinality_unresolved=sum(v["selected_cardinality_unresolved"] for v in chosen),
        exact_support=sum(v["exact_support"] for v in chosen), e2_below_point1=sum(v["e2"] < .1 for v in chosen),
        mean_fp=float(np.mean([v["fp"] for v in chosen])), mean_fn=float(np.mean([v["fn"] for v in chosen])),
        wins_vs_native_wsindy=sum(wins), wins_with_all_zero_model=sum(w and z for w, z in zip(wins, zeros)),
        wins_with_nonzero_model=sum(w and not z for w, z in zip(wins, zeros)),
        validated_candidates=sum(c.get("validated", False) for c in candidates), candidates=len(candidates),
        certified_candidates=sum(c.get("validated", False) and c.get("optimal", False) for c in candidates),
        cardinality_certified_candidates=sum(c.get("validated", False) and c.get("cardinality_optimal", False) for c in candidates),
        nonanalytic_candidates=len(nonanalytic),
        certified_nonanalytic_candidates=sum(c.get("validated", False) and c.get("optimal", False) for c in nonanalytic),
        status_counts=dict(Counter(c.get("status", "unknown") for c in candidates)),
        primary_status_counts=dict(Counter(str(c.get("primary_status", "unknown")) for c in candidates)),
        secondary_status_counts=dict(Counter(str(c.get("secondary_status", "unknown")) for c in candidates)))


def interpret(directory):
    records = study.load_records(directory)
    noisy = [r for r in records if r["noise_ratio"] > 0]
    methods = ("plugin_dantzig", "debiased_dantzig")
    levels = {level: dict(all={method: describe(records, method, level) for method in methods},
        nonzero_noise={method: describe(noisy, method, level) for method in methods} if noisy else {},
        by_pde={name: {method: describe([r for r in noisy if r["name"] == name], method, level) for method in methods}
                for name in dict.fromkeys(r["name"] for r in noisy)}) for level in LEVELS}
    settings = []
    for name, noise in dict.fromkeys((r["name"], r["noise_ratio"]) for r in records):
        rows = [r for r in records if r["name"] == name and r["noise_ratio"] == noise]
        settings.append(dict(name=name, noise_ratio=noise,
            oracles={level: {method: describe(rows, method, level) for method in methods} for level in LEVELS}))
    output = dict(completed_realizations=len(records), **levels["lexicographic"],
        oracle_levels=levels, by_setting=settings,
        baselines=dict(all={method: baseline_summary(records, method) for method in ("wsindy", "debiased_mstls")},
            nonzero_noise={method: baseline_summary(noisy, method) for method in ("wsindy", "debiased_mstls")} if noisy else {}))
    (directory/"interpretation.json").write_text(json.dumps(output, indent=2)+"\n")
    csv_rows = [dict(name=s["name"], noise_ratio=s["noise_ratio"], oracle=level, score=method,
        **{k: v for k, v in s["oracles"][level][method].items() if not isinstance(v, dict)})
        for s in settings for level in LEVELS for method in methods]
    with (directory/"interpretation.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (directory/"cardinality_oracle.jsonl").open("w") as sink:
        for record in records:
            row = dict(name=record["name"], noise_ratio=record["noise_ratio"], trial=record["trial"],
                       methods={method: select(record, method, "cardinality") for method in methods})
            sink.write(json.dumps(study.serializable(row), allow_nan=False)+"\n")
    lines = ["# Interpretation checks for the oracle sanity study", "",
        f"{len(records)} completed realizations, including {len(noisy)} with nonzero observation noise.", "",
        "The analytic zero model is always on the candidate path and has relative coefficient E2=1. "
        "Consequently every main oracle result has E2<=1 by construction. A win against a baseline whose E2 exceeds1 "
        "can consist entirely of selecting the zero model; the table distinguishes these from nonzero model wins.", "",
        "Three oracle levels are kept separate: (a) main lexicographic certification requires both minimum cardinality "
        "and the secondary residual criterion proved optimal; (b) cardinality certification requires a validated minimum-cardinality "
        "solution but permits an unresolved secondary criterion; (c) the incumbent diagnostic requires feasibility only. "
        "Level(b) is still a certified minimizer of Equation(9). Its returned coefficients use the same truth-independent "
        "secondary solver, but the residual tie-break may not have reached its optimum. Truth only selects the tolerance.", "",
        "Certificate coverage measures how much of the finite tolerance path was actually resolved. "
        "Poor coefficient recovery with low coverage cannot by itself establish failure of the mathematical estimator.", "",
        "## Native baselines, nonzero-noise realizations", "",
        "| Method | N | Mean E2 | Median E2 | Exact support | All-zero selections |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for method, row in output["baselines"]["nonzero_noise"].items():
        lines.append(f"| {method} | {row['realizations']} | {row['mean_e2']:.5g} | {row['median_e2']:.5g} | "
            f"{row['exact_support']} | {row['selected_all_zero']} |")
    lines += ["", "## Dantzig oracle levels, nonzero-noise realizations", "",
        "| Oracle | Score | N | Mean E2 | Median E2 | All-zero selections | Exact support | Wins vs native WSINDy | Wins using zero | Wins using nonzero |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for level in LEVELS:
        for method, row in levels[level]["nonzero_noise"].items():
            lines.append(f"| {level} | {method} | {row['realizations']} | {row['mean_e2']:.5g} | {row['median_e2']:.5g} | "
                f"{row['selected_all_zero']} | {row['exact_support']} | {row['wins_vs_native_wsindy']} | "
                f"{row['wins_with_all_zero_model']} | {row['wins_with_nonzero_model']} |")
    lines += ["", "## Per-setting mean E2 by oracle level", "",
        "Each error cell lists lexicographic / cardinality / incumbent. Support columns use the cardinality-certified oracle.", "",
        "| PDE | Noise | Pilot E2 (three levels) | Debiased E2 (three levels) | Pilot exact / N; mean(FP,FN) | Debiased exact / N; mean(FP,FN) |",
        "| :--- | ---: | :--- | :--- | :--- | :--- |"]
    for setting in settings:
        values = setting["oracles"]
        cells = [" / ".join(f"{values[level][method]['mean_e2']:.5g}" for level in LEVELS) for method in methods]
        support = []
        for method in methods:
            row = values["cardinality"][method]
            support.append(f"{row['exact_support']}/{row['realizations']}; ({row['mean_fp']:.3g},{row['mean_fn']:.3g})")
        lines.append(f"| {setting['name']} | {setting['noise_ratio']:g} | "+" | ".join(cells+support)+" |")
    lines += ["", "## Finite-path certificate coverage (all realizations)", "",
        "| Score | Valid / total candidates | Lexicographic certified / total | Cardinality certified / total | Lexicographic nonanalytic / nonanalytic | Main all-zero / realizations |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for method, row in output["all"].items():
        lines.append(f"| {method} | {row['validated_candidates']}/{row['candidates']} | "
            f"{row['certified_candidates']}/{row['candidates']} | "
            f"{row['cardinality_certified_candidates']}/{row['candidates']} | "
            f"{row['certified_nonanalytic_candidates']}/{row['nonanalytic_candidates']} | "
            f"{row['selected_all_zero']}/{row['realizations']} |")
    lines += ["", "The source of these counts is trials.jsonl; interpretation.json contains status histograms and per-PDE details, "
        "interpretation.csv contains every PDE/noise/oracle summary, and cardinality_oracle.jsonl stores the additional oracle's selected candidates and coefficients. "
        "Paired wins are descriptive counts, not significance tests. The native MSTLS comparison uses its existing tuning rule, "
        "while Dantzig receives per-equation oracle tolerance selection.", ""]
    (directory/"interpretation.md").write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = interpret(args.output.resolve())
    print(json.dumps(result["nonzero_noise"], indent=2))
