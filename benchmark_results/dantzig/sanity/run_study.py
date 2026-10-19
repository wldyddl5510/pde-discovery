"""Paired small-seed oracle-lambda sanity study of the PDF's discrete Dantzig estimator.

Run from the repository root, e.g.:
    python benchmark_results/dantzig/sanity/run_study.py --workers 2

The script only reads local clean trajectories. Completed trials are appended and
skipped on restart; changes to numerical inputs require a fresh output directory.
"""
from __future__ import annotations

import os
for _variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pdes import DATA_DIR, PDES, load, metrics, observe, trial_seed
from wsindy import MSTLS, SplitPilot, SplitSystem, WeakSystem, estimate, fit

METHODS = ("wsindy", "debiased_mstls", "plugin_dantzig", "debiased_dantzig")
LABELS = dict(wsindy="WSINDy / MSTLS", debiased_mstls="Split debiased MSTLS",
              plugin_dantzig="Pilot score / oracle Dantzig", debiased_dantzig="Debiased score / oracle Dantzig")
COLORS = dict(wsindy="#526779", debiased_mstls="#c18427", plugin_dantzig="#7974b8", debiased_dantzig="#009c85")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def serializable(value):
    """JSON-safe numerical results, including explicit nulls for solver infinities."""
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def evaluate(coefficients, truth):
    coefficients = np.asarray(coefficients, dtype=float)
    return dict(coefficients=coefficients.tolist(), **metrics(coefficients, truth),
                equations=[metrics(c, target) for c, target in zip(coefficients, truth)])


def baseline(system, truth, regression):
    start = perf_counter()
    result = fit(system, regression)
    return dict(evaluate(result.coef, truth), runtime=perf_counter()-start, selection=result.info)


def selected_candidate(result):
    """Resolve the selected result without assuming a particular summary spelling."""
    for key in ("selected", "summary"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    for key in ("selected_index", "best_index", "index"):
        value = result.get(key)
        if isinstance(value, (int, np.integer)):
            return result["path"][int(value)]
    coefficients = np.asarray(result.get("coefficients"), dtype=float)
    for candidate in result.get("path", []):
        trial = np.asarray(candidate.get("coefficients"), dtype=float)
        if trial.shape == coefficients.shape and np.array_equal(trial, coefficients):
            return candidate
    return result


def dantzig(score, truth, options):
    from wsindy.dantzig import solve_path
    start = perf_counter()
    paths, coefficients, incumbent_coefficients, audits = [], [], [], []
    for e, target in enumerate(truth):
        result = solve_path(score.H, score.b[:, e], target, ratios=np.asarray(options["ratios"]),
                            time_limit=options["time_limit"], mip_gap=options["mip_gap"],
                            refinements=options["refinements"])
        chosen = selected_candidate(result)
        coefficient = np.asarray(result["coefficients"], dtype=float)
        if coefficient.shape != target.shape or not np.all(np.isfinite(coefficient)):
            raise RuntimeError(f"No finite validated candidate for equation {e}")
        if not chosen.get("validated", result.get("validated", False)):
            raise RuntimeError(f"Oracle chose an unvalidated candidate for equation {e}")
        if not chosen.get("optimal", result.get("optimal", False)):
            raise RuntimeError(f"Main oracle chose an uncertified candidate for equation {e}")
        incumbent = result.get("incumbent_oracle_selected")
        incumbent_coefficients.append(incumbent["coefficients"] if incumbent is not None else coefficient)
        candidates = result["path"]
        valid = [c for c in candidates if c.get("validated", False)]
        candidate_ratios = [float(c["lambda_ratio"]) for c in candidates]
        chosen_ratio = chosen.get("lambda_ratio", result.get("lambda_ratio"))
        audits.append(dict(equation=e, selected=chosen, candidate_count=len(candidates),
            validated_candidates=len(valid), optimal_candidates=sum(c.get("optimal", False) for c in valid),
            selected_optimal=bool(chosen.get("optimal", result.get("optimal", False))),
            lower_boundary=chosen_ratio is not None and np.isclose(chosen_ratio, min(candidate_ratios), rtol=1e-10, atol=0),
            upper_boundary=chosen_ratio is not None and np.isclose(chosen_ratio, max(candidate_ratios), rtol=1e-10, atol=0),
            truth_residual_inf=float(np.max(np.abs(score.b[:, e]-score.H@target))),
            lambda_max=float(np.max(np.abs(score.b[:, e])))))
        paths.append(result)
        coefficients.append(coefficient)
    return dict(evaluate(coefficients, truth), runtime=perf_counter()-start,
                incumbent_diagnostic=evaluate(incumbent_coefficients, truth),
                selection_unit="per_equation_per_realization", solver=audits, paths=paths)


def prepare_trial(name, noise, trial):
    """Build both paired baselines and physical scores without importing Gurobi."""
    from wsindy.scores import score_matrices
    start = perf_counter()
    pde = PDES[name]
    clean, grid = load(name, download=False)
    seed = trial_seed(name, noise, trial)
    U, noise_std = observe(pde, clean, noise, seed)
    full = WeakSystem(pde, U, grid)
    truth = pde.truth_matrix(full.terms)
    terms = [term.label for term in full.terms]
    results = {"wsindy": baseline(full, truth, MSTLS())}
    del full
    pilot_start = perf_counter()
    pilot, info = estimate(U, grid, pde, SplitPilot(ratio=.5))
    split = SplitSystem(pde, U, grid, X=pilot, Y=U)
    pilot_runtime = perf_counter()-pilot_start
    results["debiased_mstls"] = baseline(split, truth, MSTLS(loss="debiased"))
    score_info, matrices = {}, {}
    for variant, method in (("plugin", "plugin_dantzig"), ("debiased", "debiased_dantzig")):
        score_start = perf_counter()
        score = score_matrices(split, variant=variant)
        construction_runtime = perf_counter()-score_start
        score_info[variant] = dict(rows=score.n_rows, shape=list(score.H.shape),
            rank=int(np.linalg.matrix_rank(score.H)), condition_number=float(np.linalg.cond(score.H)),
            symmetry_error=float(np.max(np.abs(score.H-score.H.T))),
            smallest_eigenvalue=float(np.linalg.eigvalsh((score.H+score.H.T)/2)[0]),
            construction_runtime=construction_runtime)
        matrices[f"{variant}_H"], matrices[f"{variant}_b"] = score.H, score.b
        score_info[variant]["truth_residual_inf"] = np.max(np.abs(score.b-score.H@truth.T), axis=0).tolist()
        score_info[variant]["lambda_max"] = np.max(np.abs(score.b), axis=0).tolist()
    metadata = serializable(dict(name=name, noise_ratio=noise, trial=trial, seed=seed,
        noise_std=noise_std, truth=truth, terms=terms, equations=pde.equations(),
        pilot=dict(info, **split.sampling_info), pilot_runtime=pilot_runtime,
        score=score_info, methods=results, preparation_runtime=perf_counter()-start))
    return metadata, matrices


def cache_paths(directory, name, noise, trial):
    key = json.dumps([name, noise, trial], separators=(",", ":"))
    suffix = hashlib.sha256(key.encode()).hexdigest()[:12]
    stem = f"{name}_noise-{noise:g}_trial-{trial}-{suffix}"
    return directory/f"{stem}.json", directory/f"{stem}.npz"


def read_prepared(directory, name, noise, trial, preparation_hash, *, matrices=True):
    metadata_path, matrix_path = cache_paths(directory, name, noise, trial)
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("preparation_hash") != preparation_hash:
        raise RuntimeError(f"Preparation source/data signature changed for {metadata_path}")
    if not matrix_path.exists() or digest(matrix_path) != metadata["score_cache_sha256"]:
        raise RuntimeError(f"Physical score cache is missing or has changed: {matrix_path}")
    if not matrices:
        return metadata
    with np.load(matrix_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    p, equations = len(metadata["terms"]), len(metadata["truth"])
    for variant in ("plugin", "debiased"):
        if values[f"{variant}_H"].shape != (p, p) or values[f"{variant}_b"].shape != (p, equations):
            raise RuntimeError(f"Unexpected score matrix dimensions in {matrix_path}")
        if not all(np.all(np.isfinite(values[f"{variant}_{part}"])) for part in ("H", "b")):
            raise RuntimeError(f"Nonfinite physical scores in {matrix_path}")
    return metadata, values


def prepare_cached(name, noise, trial, directory, preparation_hash):
    directory = Path(directory)
    existing = read_prepared(directory, name, noise, trial, preparation_hash)
    if existing is not None:
        return existing
    metadata, matrices = prepare_trial(name, noise, trial)
    metadata_path, matrix_path = cache_paths(directory, name, noise, trial)
    temporary_matrix = matrix_path.with_suffix(f".npz.{os.getpid()}.tmp")
    with temporary_matrix.open("wb") as sink:
        np.savez_compressed(sink, **matrices)
    os.replace(temporary_matrix, matrix_path)
    metadata.update(preparation_hash=preparation_hash, score_cache_sha256=digest(matrix_path),
                    score_cache_file=matrix_path.name)
    temporary_metadata = metadata_path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary_metadata.write_text(json.dumps(metadata, allow_nan=False)+"\n")
    os.replace(temporary_metadata, metadata_path)
    return metadata, matrices


def prepare_job(name, noise, trial, directory, preparation_hash):
    """Return only small metadata to the parent process; matrices remain on disk."""
    return prepare_cached(name, noise, trial, directory, preparation_hash)[0]


def run_trial(name, noise, trial, options, cache_directory, preparation_hash):
    start = perf_counter()
    metadata, matrices = prepare_cached(name, noise, trial, cache_directory, preparation_hash)
    truth = np.asarray(metadata["truth"])
    for variant, method in (("plugin", "plugin_dantzig"), ("debiased", "debiased_dantzig")):
        score = SimpleNamespace(H=matrices[f"{variant}_H"], b=matrices[f"{variant}_b"],
                                n_rows=metadata["score"][variant]["rows"])
        metadata["methods"][method] = dantzig(score, truth, options)
        metadata["methods"][method]["score_construction_runtime"] = metadata["score"][variant]["construction_runtime"]
    metadata.update(solver_stage_runtime=perf_counter()-start,
                    runtime=metadata["preparation_runtime"]+perf_counter()-start)
    return serializable(metadata)


def load_records(directory):
    path = directory/"trials.jsonl"
    records = []
    if path.exists():
        with path.open() as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise RuntimeError(f"Incomplete JSONL at {path}:{line_number}; repair before resuming") from error
    return sorted(records, key=lambda r: (list(PDES).index(r["name"]), r["noise_ratio"], r["trial"]))


def aggregate(records):
    groups = {}
    for record in records:
        groups.setdefault((record["name"], record["noise_ratio"]), []).append(record)
    summaries = []
    for (name, noise), rows in groups.items():
        for method in METHODS:
            values = [r["methods"][method] for r in rows]
            errors = [v["e2"] for v in values]
            summary = dict(name=name, noise_ratio=noise, method=method, trials=len(rows),
                mean_e2=float(np.mean(errors)), median_e2=float(np.median(errors)), min_e2=min(errors), max_e2=max(errors),
                exact_support_rate=float(np.mean([v["exact_support"] for v in values])),
                mean_fp=float(np.mean([v["fp"] for v in values])), mean_fn=float(np.mean([v["fn"] for v in values])),
                mean_runtime=float(np.mean([v["runtime"] for v in values])))
            if method.endswith("dantzig"):
                audits = [a for v in values for a in v["solver"]]
                candidates = [c for v in values for p in v["paths"] for c in p["path"]]
                summary.update(candidate_count=len(candidates), validated_candidates=sum(c.get("validated", False) for c in candidates),
                    optimal_candidates=sum(c.get("validated", False) and c.get("optimal", False) for c in candidates),
                    selected_equations=len(audits), selected_optimal=sum(a["selected_optimal"] for a in audits),
                    selected_lower_boundary=sum(a["lower_boundary"] for a in audits),
                    selected_upper_boundary=sum(a["upper_boundary"] for a in audits),
                    incumbent_mean_e2=float(np.mean([v["incumbent_diagnostic"]["e2"] for v in values])),
                    incumbent_exact_support_rate=float(np.mean([v["incumbent_diagnostic"]["exact_support"] for v in values])))
            summaries.append(summary)
    return summaries


def plot(directory, summaries):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pde-dantzig-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(s["name"] for s in summaries))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True, sharex=True)
    for ax, name in zip(axes.flat, names):
        for method in METHODS:
            rows = sorted((s for s in summaries if s["name"] == name and s["method"] == method),
                          key=lambda s: s["noise_ratio"])
            noise = [s["noise_ratio"] for s in rows]
            error = np.maximum([s["mean_e2"] for s in rows], 1e-8)
            ax.plot(noise, error, "o-", ms=4, lw=1.7, color=COLORS[method], label=LABELS[method])
        ax.set(title=name, yscale="log", xlabel="Observation noise ratio", ylabel="Mean relative coefficient E2")
        ax.grid(alpha=.2)
        ax.set_xticks([0, .2, .5, .75, 1])
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    trials = json.loads((directory/"manifest.json").read_text())["trials"]
    fig.suptitle(f"Discrete Dantzig sanity check: oracle tolerance, {trials} noise seeds\n"
                 "Each equation tunes its own tolerance; native MSTLS tuning in baselines", fontsize=13)
    fig.savefig(directory/"results.png", dpi=170)
    plt.close(fig)


def report(directory):
    records = load_records(directory)
    if not records:
        raise RuntimeError(f"No trial records in {directory}")
    summaries = aggregate(records)
    (directory/"summary.json").write_text(json.dumps(serializable(summaries), indent=2)+"\n")
    fields = list(dict.fromkeys(k for row in summaries for k in row))
    with (directory/"summary.csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    manifest = json.loads((directory/"manifest.json").read_text())
    options = manifest["options"]
    expected = len(manifest["pdes"])*(1 if 0 in manifest["noise"] else 0)+len(manifest["pdes"])*sum(n != 0 for n in manifest["noise"])*manifest["trials"]
    lines = ["# Oracle discrete Dantzig: small-seed sanity check", "",
        f"Completed {len(records)}/{expected} paired realizations. This is a preliminary coefficient-recovery experiment, "
        "not a tuning rule or a statistical power study.", "",
        "## Protocol", "",
        "- Same local trajectories, dictionaries, weak test functions, observation noise definitions and deterministic paired seeds as the repository.",
        f"- PDEs: {', '.join(manifest['pdes'])}; noise ratios: {manifest['noise']}; {manifest['trials']} independent observation-noise seeds per nonzero noise level. Noise-free data are evaluated once.",
        "- A 50:50 split pilot is estimated from training time slices only. Both Dantzig methods use the same pilot, regression time slices and physical-unit quadrature.",
        "- Debiased score: H=(X'X+dX'X+X'dX)/K, b=(X'Y+dX'Y+X'dY)/K. The product dX'dX and dX'dY terms are excluded.",
        "- Pilot-score control: H=X'X/K, b=X'Y/K on the same pilot and split. The contrast isolates score correction within this setup.",
        "- Each Dantzig fit minimizes nonzero count subject to the physical score infinity-residual bound. No coefficient thresholding or OLS refit is used to improve its returned estimate.",
        "- Truth enters only oracle tolerance selection and evaluation, never support selection, tie-breaking, pilot estimation or coefficient fitting.",
        "- Coupled PDEs use a separate oracle tolerance for each equation. This is a particularly favorable oracle comparison; baselines use the repository's native common-threshold MSTLS selection.",
        f"- Initial tolerance ratios are {len(options['ratios'])} logarithmically spaced values from {min(options['ratios']):g} to {max(options['ratios']):g}, "
        f"lambda=ratio*||b||inf, with {options['refinements']} local refinement round(s). See each saved path for the actual grid and any lower-bound extension.",
        f"- Per mixed-integer primary solve time limit: {options['time_limit']:g} s; secondary phase gets max(.2,min(2,T/2)) seconds; relative MIP gap target: {options['mip_gap']:g}. "
        "The main oracle requires finite constraint-validated candidates with both cardinality and secondary optimality certified.",
        "- The main lambda oracle chooses the smallest physical coefficient E2 among certified candidates. "
        "A separate incumbent diagnostic also admits validated time-limited candidates. Neither is guaranteed to find the globally best tolerance; the grid is finite and solver limits can leave parts unresolved.",
        "- The l0 objective may admit multiple supports or coefficients. The solver uses a truth-independent secondary score-residual criterion; its exact tie-break and certification metadata are saved in every path.",
        "- Solver optimality certifies the cardinality problem and secondary score-residual criterion at a fixed tolerance, not recovery of the true PDE. "
        "In particular main E2=1 can arise because only the analytic zero model was certified; this alone does not establish estimator failure.",
        "- E2 is the relative Frobenius norm over all physical coefficients. Exact support requires no false positives or false negatives across every equation. FP/FN are counts across all equations.",
        "- Runtime columns exclude shared pilot construction; Dantzig runtime also excludes score construction. The raw per-trial records store these costs separately.",
        "", "## Mean coefficient E2", "",
        "| PDE | Noise | Seeds | WSINDy/MSTLS | Split debiased MSTLS | Pilot-score oracle | Debiased-score oracle |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    groups = list(dict.fromkeys((s["name"], s["noise_ratio"]) for s in summaries))
    for name, noise in groups:
        rows = {s["method"]: s for s in summaries if s["name"] == name and s["noise_ratio"] == noise}
        lines.append(f"| {name} | {noise:g} | {rows[METHODS[0]]['trials']} | "+" | ".join(f"{rows[m]['mean_e2']:.5g}" for m in METHODS)+" |")
    lines += ["", "## Support recovery", "",
        "Each cell shows exact-support rate, then mean (FP, FN).", "",
        "| PDE | Noise | WSINDy/MSTLS | Split debiased MSTLS | Pilot-score oracle | Debiased-score oracle |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for name, noise in groups:
        rows = {s["method"]: s for s in summaries if s["name"] == name and s["noise_ratio"] == noise}
        lines.append(f"| {name} | {noise:g} | "+" | ".join(f"{rows[m]['exact_support_rate']:.0%}; ({rows[m]['mean_fp']:.2g}, {rows[m]['mean_fn']:.2g})" for m in METHODS)+" |")
    lines += ["", "## Solver audit", "",
        "Counts are equation fits. Candidate certification refers to validated candidates with proven minimal cardinality; "
        "time-limited incumbents are eligible only for the separately reported incumbent diagnostic.", "",
        "| PDE | Noise | Score | Valid candidates / total | Certified / valid | Certified selections / equations | Lower-bound selections | Upper-bound selections |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        if s["method"].endswith("dantzig"):
            lines.append(f"| {s['name']} | {s['noise_ratio']:g} | {s['method'].removesuffix('_dantzig')} | "
                f"{s['validated_candidates']}/{s['candidate_count']} | {s['optimal_candidates']}/{s['validated_candidates']} | "
                f"{s['selected_optimal']}/{s['selected_equations']} | {s['selected_lower_boundary']} | {s['selected_upper_boundary']} |")
    lines += ["", "## Time-limited incumbent diagnostic", "",
        "This additional oracle admits every constraint-validated candidate, including unresolved cardinality or tie-break solves. "
        "It can look better than the exact estimator by retaining more terms than the minimum-cardinality solution. "
        "Use it to diagnose optimization limitations, not as the estimator's main performance.", "",
        "| PDE | Noise | Pilot certified E2 | Pilot incumbent E2 | Debiased certified E2 | Debiased incumbent E2 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for name, noise in groups:
        rows = {s["method"]: s for s in summaries if s["name"] == name and s["noise_ratio"] == noise}
        p, d = rows["plugin_dantzig"], rows["debiased_dantzig"]
        lines.append(f"| {name} | {noise:g} | {p['mean_e2']:.5g} | {p['incumbent_mean_e2']:.5g} | "
                     f"{d['mean_e2']:.5g} | {d['incumbent_mean_e2']:.5g} |")
    noisy = [r for r in records if r["noise_ratio"] > 0]
    if noisy:
        db_wins = sum(r["methods"]["debiased_dantzig"]["e2"] < r["methods"]["plugin_dantzig"]["e2"] for r in noisy)
        baseline_wins = sum(r["methods"]["debiased_dantzig"]["e2"] < r["methods"]["debiased_mstls"]["e2"] for r in noisy)
        lines += ["", "## Paired descriptive comparison", "",
            f"Among {len(noisy)} nonzero-noise realizations, debiased-score oracle E2 is strictly smaller than pilot-score oracle E2 in {db_wins}, "
            f"and smaller than split debiased MSTLS in {baseline_wins}. Ties count in neither direction. "
            "This count weights PDE/noise settings equally per seed and is not a significance test."]
    lines += ["", "## Reproduce", "", "```sh",
        "python benchmark_results/dantzig/sanity/run_study.py --workers 2", "",
        "# Regenerate figures and tables from saved trials only:",
        "python benchmark_results/dantzig/sanity/run_study.py --report-only", "```", "",
        "The manifest stores source/data SHA256 hashes and package versions. trials.jsonl stores every oracle candidate, "
        "coefficients, physical-score truth residuals, solver statuses and seeds. summary.csv/json contain grouped statistics; "
        "results.png plots mean E2. Results are exploratory: three noise draws on a single trajectory per PDE cannot establish robustness.", ""]
    (directory/"results.md").write_text("\n".join(lines))
    plot(directory, summaries)
    return summaries


def make_manifest(args, options):
    sources = [Path(__file__).resolve(), ROOT/"pdes.py", ROOT/"wsindy/dantzig.py", ROOT/"wsindy/scores.py",
               ROOT/"wsindy/system.py", ROOT/"wsindy/regression.py", ROOT/"wsindy/estimators.py",
               ROOT/"wsindy/dictionary.py", ROOT/"wsindy/testfunctions.py"]
    packages = {}
    for package in ("numpy", "scipy", "pysindy", "gurobipy", "scikit-learn", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return dict(protocol="physical-score-discrete-dantzig-oracle-v1", pdes=args.pdes, noise=args.noise,
        trials=args.trials, options=options, pilot="SplitPilot(ratio=.5, prior=.01)", oracle="per-equation minimum physical E2",
        baseline_tuning="native MSTLS", no_refit=True, commit=commit, packages=packages,
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        data={name: {"file": PDES[name].file, "sha256": digest(DATA_DIR/PDES[name].file)} for name in args.pdes})


def preparation_manifest(args):
    """The cache depends on preparation code/data, not solver code or its lambda grid."""
    sources = [Path(__file__).resolve(), ROOT/"pdes.py", ROOT/"wsindy/scores.py", ROOT/"wsindy/system.py",
               ROOT/"wsindy/regression.py", ROOT/"wsindy/estimators.py", ROOT/"wsindy/dictionary.py",
               ROOT/"wsindy/testfunctions.py"]
    return dict(protocol="paired-physical-score-preparation-v1", pdes=args.pdes, noise=args.noise,
        trials=args.trials, pilot="SplitPilot(ratio=.5, prior=.01)",
        baselines=["wsindy", "debiased_mstls"], score_variants=["plugin", "debiased"],
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        data={name: {"file": PDES[name].file, "sha256": digest(DATA_DIR/PDES[name].file)} for name in args.pdes})


def register_manifest(path, manifest):
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise SystemExit(f"Numerical study inputs changed: choose a fresh --output instead of mixing results in {path.parent}")
    else:
        path.write_text(json.dumps(manifest, indent=2)+"\n")


def preparation_report(directory, rows):
    rows.sort(key=lambda r: (list(PDES).index(r["name"]), r["noise_ratio"], r["trial"]))
    output = []
    for row in rows:
        output.append(dict(name=row["name"], noise_ratio=row["noise_ratio"], trial=row["trial"],
            preparation_runtime=row["preparation_runtime"],
            wsindy_e2=row["methods"]["wsindy"]["e2"],
            debiased_mstls_e2=row["methods"]["debiased_mstls"]["e2"],
            plugin_rank=row["score"]["plugin"]["rank"], debiased_rank=row["score"]["debiased"]["rank"],
            matrix_file=row["score_cache_file"], sha256=row["score_cache_sha256"]))
    if output:
        with (directory/"preparation_summary.csv").open("w", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    (directory/"preparation_summary.json").write_text(json.dumps(output, indent=2)+"\n")
    lines = ["# Prepared paired score systems", "",
        f"{len(rows)} realizations have paired MSTLS baselines and physical-unit plugin/debiased score matrices cached.", "",
        "The Dantzig estimator has not been evaluated by this preparation stage. No Gurobi license or true support is used.", "",
        "Each prepared/*.json file contains seeds, true coefficients for later evaluation, pilot metadata, baseline results, "
        "score diagnostics and the SHA256 hash of its matching prepared/*.npz matrix file. "
        "preparation_manifest.json records source/data hashes; mismatched caches are rejected on restart.", "",
        "```sh", "# Reuse these matrices to run the solver once its license is available:",
        "python benchmark_results/dantzig/sanity/run_study.py --workers 2", "```", ""]
    (directory/"prepared.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdes", nargs="+", choices=tuple(PDES), default=list(PDES))
    parser.add_argument("--noise", type=float, nargs="+", default=[0., .2, .5, .75, 1.])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=5.)
    parser.add_argument("--mip-gap", type=float, default=0.)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--ratios", type=float, nargs="+", default=np.logspace(-6, 0, 13).tolist())
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="cache paired MSTLS baselines and physical score matrices without a solver license")
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(directory)
        return
    if args.trials < 1 or args.workers < 1 or any(noise < 0 for noise in args.noise):
        parser.error("Require positive trials/workers and nonnegative noise ratios")
    preparation = preparation_manifest(args)
    register_manifest(directory/"preparation_manifest.json", preparation)
    preparation_hash = hashlib.sha256(json.dumps(preparation, sort_keys=True).encode()).hexdigest()
    cache_directory = directory/"prepared"
    cache_directory.mkdir(exist_ok=True)
    scope = [(name, noise, trial) for name in args.pdes for noise in args.noise
             for trial in range(1 if noise == 0 else args.trials)]
    if args.prepare_only:
        completed, jobs = [], []
        for job in scope:
            cached = read_prepared(cache_directory, *job, preparation_hash, matrices=False)
            if cached is None:
                jobs.append(job)
            else:
                completed.append(cached)
        print(f"Preparation: {len(completed)}/{len(scope)} cached, {len(jobs)} pending, {args.workers} workers", flush=True)
        start = perf_counter()
        errors = []
        with ProcessPoolExecutor(args.workers) as pool:
            futures = {pool.submit(prepare_job, *job, cache_directory, preparation_hash): job for job in jobs}
            for count, future in enumerate(as_completed(futures), 1):
                name, noise, trial = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    failure = dict(name=name, noise_ratio=noise, trial=trial, error=repr(error))
                    with (directory/"preparation_errors.jsonl").open("a") as sink:
                        sink.write(json.dumps(failure)+"\n")
                    errors.append(failure)
                    print(f"[prepare {count}/{len(jobs)}] FAILED {name} noise={noise:g} trial={trial}: {error!r}", flush=True)
                    continue
                completed.append(record)
                print(f"[prepare {count}/{len(jobs)}] {name} noise={noise:g} trial={trial}: "
                      f"baseline E2={record['methods']['wsindy']['e2']:.4g}/"
                      f"{record['methods']['debiased_mstls']['e2']:.4g}, "
                      f"{len(record['terms'])} columns, {record['score']['debiased']['rows']} rows; "
                      f"trial={record['preparation_runtime']:.1f}s, elapsed={perf_counter()-start:.1f}s", flush=True)
        preparation_report(directory, completed)
        if errors:
            raise SystemExit(f"{len(errors)} preparation(s) failed; inspect preparation_errors.jsonl and rerun to retry")
        return
    options = dict(ratios=sorted(set(args.ratios)), time_limit=args.time_limit, mip_gap=args.mip_gap, refinements=args.refinements)
    manifest = make_manifest(args, options)
    manifest["preparation_hash"] = preparation_hash
    manifest_path = directory/"manifest.json"
    register_manifest(manifest_path, manifest)
    records = load_records(directory)
    done = {(r["name"], r["noise_ratio"], r["trial"]) for r in records}
    jobs = [job for job in scope if job not in done]
    errors = []
    start = perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_trial, *job, options, cache_directory, preparation_hash): job for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            name, noise, trial = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failure = dict(name=name, noise_ratio=noise, trial=trial, error=repr(error))
                with (directory/"errors.jsonl").open("a") as sink:
                    sink.write(json.dumps(failure)+"\n")
                errors.append(failure)
                print(f"[{count}/{len(jobs)}] FAILED {name} noise={noise:g} trial={trial}: {error!r}", flush=True)
                continue
            with (directory/"trials.jsonl").open("a") as sink:
                sink.write(json.dumps(result, allow_nan=False)+"\n")
            db = result["methods"]["debiased_dantzig"]
            certified = sum(a["selected_optimal"] for a in db["solver"])
            print(f"[{count}/{len(jobs)}] {name} noise={noise:g} trial={trial}: "
                  f"db E2={db['e2']:.4g}, exact={db['exact_support']}, certified={certified}/{len(db['solver'])}; "
                  f"trial={result['runtime']:.1f}s, elapsed={perf_counter()-start:.1f}s", flush=True)
    if load_records(directory):
        report(directory)
    if errors:
        raise SystemExit(f"{len(errors)} trial(s) failed; inspect errors.jsonl and rerun to retry missing trials")


if __name__ == "__main__":
    main()
