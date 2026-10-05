"""Add selected LASSO penalty levels to reports using saved trials only."""
import csv
import io
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LASSO_REPORTS = [ROOT/"benchmark_results/lasso"/suffix/"results.md" for suffix in
                 ("original/raw", "original/filtered", "consistency/raw", "consistency/filtered")]
BASELINES = [ROOT/path for path in ("benchmark_results/authors/results.md",
    "benchmark_results/filtered/results.md", "benchmark_results/consistency/raw/results.md",
    "benchmark_results/consistency/filtered/results.md")]
METHOD_LABELS = {"wsindy": "WSINDy", "filtered-wsindy": "Filtered WSINDy",
                 "debiased-wsindy": "Debiased WSINDy (time2)"}
DEBIASED_REPORTS = [ROOT/"benchmark_results/debiasing/IB"/regression/"results.md"
                    for regression in ("mstls", "lasso")]
NOTE = """## LASSO penalty selection

For K weak rows, normalize each nonzero library column as `Z_j=G_j/RMS(G_j)` and each response as `y_e=b_e/RMS(b_e)`. Solve `||Z theta-y_e||_2^2/(2*K) + lambda_e*||theta||_1`, without centering or an extra intercept; the constant library term is penalized.

For each trial and equation, define `lambda_max,e = max(abs(Z.T @ y_e))/K`, the smallest penalty for which the zero model is optimal. Evaluate 100 logarithmically spaced candidates from `1e-4*lambda_max,e` to `lambda_max,e`: `lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99)`, for `k=0,...,99`. Coupled equations have separate lambda_max values and use the same candidate index k.

Choose the candidate that minimizes the existing WSINDy score: matrix 2-norm prediction difference from full OLS, divided by the full OLS prediction norm, plus the fraction of nonzero coefficients. Exact ties select the smallest lambda. Selection uses the observed weak system G/b; it uses neither the true equation nor cross-validation.

The `lambda median` column gives the median selected lambda over the 100 trials, separately for each LHS equation. Lambda multiplies the L1 penalty on RMS-normalized coefficients; physical coefficients are restored afterward. Retain the penalized coefficients, including shrinkage, with no OLS support refit or coefficient cutoff. Empty supports are allowed. CSV/JSON summaries retain the full penalty statistics; MSTLS rows have no LASSO penalty.

The scikit-learn LARS Gram path is checked against KKT conditions at every candidate. Primal active-set polishing is tried first, followed by coordinate descent and final polishing if needed; violation divided by lambda_max above `1e-7` rejects the fit. These LASSO runs are additional regression comparisons to the papers' published algorithms."""
FIELDS = ("lasso_ratio_min", "lasso_ratio_median", "lasso_ratio_max",
          "lasso_lambda_min", "lasso_lambda_median", "lasso_lambda_max")


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as target:
        target.write(content)
        temporary = target.name
    os.replace(temporary, path)


def read_run(report, names):
    manifest = json.loads(report.with_suffix(".manifest.json").read_text())
    index = {}
    for name in manifest["benchmarks"]:
        if name not in names:
            continue
        path = report.parent/"results_trials"/(name+".jsonl")
        if not path.exists():
            continue
        with path.open() as source:
            for line in source:
                if not line.endswith("\n"):
                    break
                record = json.loads(line)
                if record["protocol_id"] != manifest["protocol_id"]:
                    raise ValueError(f"Record protocol mismatch: {path}")
                key = (name, record["noise_ratio"], record["trial"])
                if key in index:
                    raise ValueError(f"Duplicate trial: {path}/{key}")
                index[key] = {k: record[k] for k in
                    ("threshold", "selected_alpha") if k in record}
    return manifest, index


def penalty_statistics(records, components):
    if not records:
        return {field: None for field in FIELDS}
    ratios = [r["threshold"] for r in records]
    alphas = [r["selected_alpha"] for r in records]
    if (any(len(values) != len(components) for values in alphas)
            or any(not math.isfinite(v) or v < 0 for values in alphas for v in values)
            or any(not 1e-4 <= r <= 1 for r in ratios)):
        raise ValueError("Invalid saved LASSO penalty levels")
    result = {}
    for label, function in (("min", min), ("median", median), ("max", max)):
        result["lasso_ratio_"+label] = function(ratios)
        result["lasso_lambda_"+label] = {name: function(values[e] for values in alphas)
                                        for e, name in enumerate(components)}
    return result


def enrich(report, runs, common):
    combined = report == ROOT/"results.md"
    own = None if combined else next(run for run in runs if run[0] == report)
    stats = {}
    summaries_path = report.with_suffix(".summary.json")
    summaries = json.loads(summaries_path.read_text())
    for row in summaries:
        name, ratio = row["name"], row["noise_ratio"]
        regression = row.get("regression", own[1]["regression"] if own else "mstls")
        method = row.get("method", "wsindy")
        key = (name, ratio, method, regression)
        result = {field: None for field in FIELDS}
        if regression == "lasso":
            source = next(run for run in runs if name in run[1]["benchmarks"]
                and run[1].get("method", "wsindy") == method
                and run[1].get("regression", "mstls") == regression)
            spec = source[1]["benchmarks"][name]
            components = [spec["component_names"][i] for i in spec["lhs_components"]]
            keys = common[name] if combined else source[2].keys()
            records = [source[2][k] for k in keys if k[0] == name and k[1] == ratio]
            if len(records) != row["trials"]:
                raise ValueError(f"Penalty/result trial counts differ: {report}/{key}")
            result = penalty_statistics(records, components)
        row.update(result)
        stats[key] = result
    text = report.read_text()
    results_heading = "## Results by noise level"
    equations_heading = "## True equations and experiment setup"
    if results_heading in text:
        introduction = text.split(results_heading, 1)[0]
        equations = text.split(equations_heading, 1)[1]
    else:
        first_equation = re.search(r"^## (?:"+"|".join(map(re.escape, common))+r")$", text, re.M)
        if first_equation is None:
            raise ValueError(f"No experiment definitions in {report}")
        introduction, equations = text[:first_equation.start()], text[first_equation.start():]
        equations = "\n".join(line for line in equations.splitlines() if not line.startswith("| "))
    introduction = introduction.replace(NOTE, "")
    introduction = "\n".join(line for line in introduction.splitlines()
        if not line.startswith(("LASSO penalty levels:",
                                "LASSO replaces sequential hard thresholding."))).rstrip()
    introduction = re.sub(r"\n{3,}", "\n\n", introduction)
    names = list(dict.fromkeys(row["name"] for row in summaries))
    headings = list(re.finditer(r"^#{2,3} ("+"|".join(map(re.escape, names))+r")$", equations, re.M))
    contexts = {}
    for i, heading in enumerate(headings):
        end = headings[i+1].start() if i+1 < len(headings) else len(equations)
        body = equations[heading.end():end]
        body = "\n".join(line for line in body.splitlines()
                         if not line.startswith(("| ", "### Noise ratio ")))
        contexts[heading.group(1)] = re.sub(r"\n{3,}", "\n\n", body).strip()
    title, _, introduction_body = introduction.partition("\n")
    lines = [title, "", NOTE, "", introduction_body.strip(), ""]
    for name in names:
        if name not in contexts:
            raise ValueError(f"Missing experiment definition: {report}/{name}")
        lines.extend([f"## {name}", "", contexts[name], ""])
        for ratio in sorted({row["noise_ratio"] for row in summaries if row["name"] == name}):
            lines.extend([f"### Noise ratio {ratio:g}", "",
                "| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"])
            for row in summaries:
                if row["name"] != name or row["noise_ratio"] != ratio:
                    continue
                regression = row.get("regression", own[1]["regression"] if own else "mstls")
                method = row.get("method", "wsindy")
                label = METHOD_LABELS[method]+" + "+regression.upper()
                metrics = []
                for field in ("tpr", "e_inf", "e2"):
                    mean, sd = row[field+"_mean"], row[field+"_sd"]
                    metrics.append(f"{mean:.6g} ± {sd:.3g}" if sd is not None else f"{mean:.6g}")
                values = stats[(name, ratio, method, regression)]
                lambda_cell = "—"
                if values["lasso_lambda_median"] is not None:
                    lambda_cell = "; ".join(f"{component}={value:.4g}" for component, value
                                           in values["lasso_lambda_median"].items())
                cells = [label, str(row["trials"]), f"{100*row['exact_rate']:.1f}%",
                         *metrics, f"{row['runtime_median']:.3g}", lambda_cell]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    atomic_text(report, "\n".join(lines)+"\n")
    atomic_text(summaries_path, json.dumps(summaries, indent=2, allow_nan=False)+"\n")
    if summaries:
        target = io.StringIO(newline="")
        writer = csv.DictWriter(target, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict)
                             else value for key, value in row.items()})
        atomic_text(report.with_suffix(".summary.csv"), target.getvalue())
    return len(summaries)


def main():
    names = set().union(*(json.loads(report.with_suffix(".manifest.json").read_text())["benchmarks"]
                          for report in LASSO_REPORTS))
    candidates = list(DEBIASED_REPORTS)
    suite_root = ROOT/"benchmark_results/debiasing/time2_foldwise"
    for name in ("IB", "KdV", "KS", "NLS", "RD", "NS", "HKS", "VBG"):
        updated = [suite_root/name/regression/"results.md" for regression in ("mstls", "lasso")]
        if all(p.with_suffix(".status.json").exists()
               and json.loads(p.with_suffix(".status.json").read_text())["state"] == "complete"
               and p.with_suffix(".summary.json").exists() for p in updated):
            if name == "IB":
                candidates = []
            candidates.extend(updated)
    extras = [report for report in candidates
              if report.with_suffix(".status.json").exists()
              and json.loads(report.with_suffix(".status.json").read_text())["state"] == "complete"]
    runs = [(report, *read_run(report, names)) for report in [*BASELINES, *LASSO_REPORTS, *extras]]
    common = {name: set.intersection(*(set(index) for _, manifest, index in runs
                                      if name in manifest["benchmarks"])) for name in names}
    if extras:
        summary_path = ROOT/"results.summary.json"
        summaries = json.loads(summary_path.read_text())
        summaries = [row for row in summaries if row.get("method") != "debiased-wsindy"]
        new_rows = [row for report in extras
                    for row in json.loads(report.with_suffix(".summary.json").read_text())]
        for new_row in new_rows:
            position = max(i for i, row in enumerate(summaries)
                           if row["name"] == new_row["name"] and row["noise_ratio"] == new_row["noise_ratio"])
            summaries.insert(position+1, new_row)
        atomic_text(summary_path, json.dumps(summaries, indent=2, allow_nan=False)+"\n")
        report_path = ROOT/"results.md"
        text = report_path.read_text().replace("across the four methods.", "across the methods listed for that PDE.")
        for name in names:
            selected = [report for report in extras if name in
                        json.loads(report.with_suffix(".manifest.json").read_text())["benchmarks"]]
            if not selected:
                continue
            description = json.loads(selected[0].with_suffix(".manifest.json").read_text())["preprocessing"]["description"]
            links = "; ".join(f"[{report.parent.name.upper()}]({report.relative_to(ROOT).as_posix()})" for report in selected)
            note = f"{name} debiasing: {description} Saved trial reports: {links}."
            text = "\n".join(line for line in text.splitlines() if not line.startswith(f"{name} debiasing:"))+"\n"
            text = text.replace(f"## {name}\n", f"## {name}\n\n"+note+"\n", 1)
        suite_status_path = suite_root/"suite.status.json"
        if suite_status_path.exists():
            suite_status = json.loads(suite_status_path.read_text())
            completed_fits = sum(row["trials"] for row in new_rows)
            requested_fits = suite_status.get("requested_fits", suite_status.get("completed_fits"))
            state = "complete" if completed_fits == requested_fits else "in progress"
            text = re.sub(r"^Overall status:.*$", "Baseline suites status: complete.", text, flags=re.M)
            text = "\n".join(line for line in text.splitlines()
                              if not line.startswith("Debiasing suite status:"))+"\n"
            text = text.replace("Baseline suites status: complete.",
                "Baseline suites status: complete.\n\n"
                f"Debiasing suite status: {state}; {completed_fits:,}/{requested_fits:,} "
                "fits from completed PDE reports are included below.", 1)
        atomic_text(report_path, text)
    for report in [ROOT/"results.md", *LASSO_REPORTS, *extras]:
        count = enrich(report, runs, common)
        print(f"Penalty levels added to {report.relative_to(ROOT)}: {count} summary rows", flush=True)


if __name__ == "__main__":
    main()
