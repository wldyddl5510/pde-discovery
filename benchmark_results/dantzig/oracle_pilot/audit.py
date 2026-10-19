"""Independent, solver-free audit of the paired oracle-pilot experiment.

Reconstructs every prepared observation and actual oracle state without calling
the experiment's observe, seed, oracle-pilot, score, selection, or metrics helpers.
Optional weak-score reconstruction uses the frozen convolution primitive but
independently composes the first-order score. Only audit.json is written.
"""
from __future__ import annotations

import os
for _variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pdes import DATA_DIR, PDES, load
from wsindy import SplitPilot, SplitSystem, estimate

OLD = ROOT/"benchmark_results/dantzig/sanity"
NEW = tuple(f"{mode}_{variant}_dantzig" for mode in ("rate", "exact") for variant in ("plugin", "debiased"))
BASELINES = ("wsindy", "debiased_mstls", "plugin_dantzig", "debiased_dantzig")
LEVELS = ("lexicographic", "cardinality", "incumbent")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    result = hashlib.sha256()
    result.update(str(array.dtype).encode())
    result.update(json.dumps(array.shape).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def independent_seed(name, noise, trial):
    order = ("IB", "KdV", "KS", "NLS", "SG", "RD", "NS", "HKS", "VBG")
    bits = int(np.float64(noise).view(np.uint64))
    return np.random.SeedSequence([0, order.index(name), bits & 0xffffffff, bits >> 32, trial]).generate_state(4).tolist()


def observation(pde, clean, noise, seed):
    rng = np.random.default_rng(seed)
    pooled = float(np.std(np.concatenate([clean[..., c].ravel() for c in range(clean.shape[-1])]), ddof=1))
    states, deviations = [], []
    for component in range(clean.shape[-1]):
        true = clean[..., component]
        deviation = float(noise*(pooled if pde.profile.noise == "std" else np.sqrt(np.mean(true**2))))
        states.append(true.copy() if deviation == 0 else true+rng.normal(0, deviation, true.shape))
        deviations.append(deviation)
    return np.stack(states, axis=-1), deviations


def metrics(coefficients, truth):
    coefficients, truth = np.asarray(coefficients), np.asarray(truth)
    selected, actual = coefficients != 0, truth != 0
    tp, fn, fp = (int(np.count_nonzero(mask)) for mask in (selected & actual, ~selected & actual, selected & ~actual))
    return dict(e2=float(np.linalg.norm(coefficients-truth)/np.linalg.norm(truth)),
                e_inf=float(np.max(np.abs((coefficients-truth)[actual]/truth[actual]))),
                tp=tp, fn=fn, fp=fp, tpr=tp/(tp+fn+fp), exact_support=fn == 0 and fp == 0)


def key_for(job):
    return f"{job[0]}_noise-{job[1]:g}_trial-{job[2]}"


def json_records(path):
    if not path.exists():
        return [], b"", False
    snapshot = path.read_bytes()
    partial = bool(snapshot and not snapshot.endswith(b"\n"))
    lines = snapshot.splitlines()
    if partial:
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()], snapshot, partial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--rebuild-scores-noiseless", action="store_true")
    parser.add_argument("--rebuild-score-jobs", nargs="*", default=[], help="Extra prepared key(s), e.g. IB_noise-0.2_trial-0")
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory/"manifest.json").read_text())
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    expected = {tuple(job) for job in manifest["scope"]}
    canonical = {name:next(tuple(job) for job in manifest["scope"] if job[0] == name) for name in manifest["pdes"]}
    failures, counts = [], Counter()
    maxima = Counter()
    rebuilt = []
    exact_noiseless_differences = {}
    # The writer creates immutable preparation/solution files before atomically
    # publishing joined records. Snapshot in reverse dependency order so a
    # concurrent partial audit never sees a joined record without its inputs.
    records, snapshot, partial_tail = json_records(directory/"trials.jsonl")
    solution_paths = sorted((directory/"solved").glob("*.json"))
    metadata_paths = sorted((directory/"prepared").glob("*.json"))

    def check(condition, label, **context):
        counts["assertions"] += 1
        if not condition:
            failures.append(dict(check=label, **context))

    def close(actual, expected_value, label, *, rtol=2e-10, atol=1e-13, **context):
        check(bool(np.allclose(actual, expected_value, rtol=rtol, atol=atol, equal_nan=False)), label, **context)

    def audit_metrics(result, truth, context):
        coefficients = np.asarray(result["coefficients"], dtype=float)
        check(coefficients.shape == truth.shape and np.isfinite(coefficients).all(), "finite physical coefficient dimensions", **context)
        for metric, value in metrics(coefficients, truth).items():
            close(result[metric], value, "independently recomputed recovery metric", metric=metric, **context)
        if "equations" in result:
            check(len(result["equations"]) == len(truth), "equation metric count", **context)
            for equation, (coefficient, target) in enumerate(zip(coefficients, truth)):
                for metric, value in metrics(coefficient, target).items():
                    close(result["equations"][equation][metric], value, "independently recomputed equation metric", metric=metric, equation=equation, **context)

    check(manifest["expected_realizations"] == len(expected), "declared realization count")
    fits_expected = {(job, method) for job in expected for method in NEW if method != "exact_plugin_dantzig"}
    fits_expected.update((job, "exact_plugin_dantzig") for job in canonical.values())
    check(manifest["expected_unique_score_fits"] == len(fits_expected), "declared unique score fit count")
    check(manifest["oracle_exponent"] > .25 and manifest["oracle_cap"] > 0, "oracle family little-o exponent and positive cap")
    for relative, sha in manifest["inputs"]["sources"].items():
        check(digest(ROOT/relative) == sha, "frozen numerical source hash", path=relative)
    for name, data in manifest["inputs"]["data"].items():
        check(digest(DATA_DIR/data["file"]) == data["sha256"], "clean trajectory file hash", pde=name)
    for filename, sha in manifest["inputs"]["old_study"].items():
        check(digest(OLD/filename) == sha, "original paired study hash", path=filename)
    for relative, sha in json.loads((OLD/"manifest.json").read_text())["sources"].items():
        check(digest(ROOT/relative) == sha, "original numerical source remains unchanged", path=relative)
    old_records, _, _ = json_records(OLD/"trials.jsonl")
    original = {(r["name"], r["noise_ratio"], r["trial"]):r for r in old_records}
    check(len(original) == len(old_records), "original realization uniqueness")
    solver_ast = ast.parse((ROOT/"wsindy/dantzig.py").read_text())
    solver = next(node for node in solver_ast.body if isinstance(node, ast.FunctionDef) and node.name == "solve_one")
    check(not any(isinstance(node, ast.Name) and node.id == "truth" for node in ast.walk(solver)), "truth absent from fixed-lambda optimization")
    helper_ast = ast.parse((ROOT/"wsindy/oracle_pilot.py").read_text())
    helper = next(node for node in helper_ast.body if isinstance(node, ast.FunctionDef) and node.name == "oracle_pilot")
    check([arg.arg for arg in helper.args.args] == ["clean", "base", "grid"], "oracle construction does not receive regression observations or true coefficients")

    metadata_by_key, scores_by_key, exact_scores = {}, {}, {}
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        job = (metadata["name"], metadata["noise_ratio"], metadata["trial"])
        key = key_for(job)
        context = dict(job=job)
        check(job in expected and key not in metadata_by_key, "unique prepared realization in protocol", **context)
        metadata_by_key[key] = metadata
        check(metadata_path.stem == key and metadata["signature"] == signature, "prepared filename and signature", **context)
        matrix_path = metadata_path.with_suffix(".npz")
        check(digest(matrix_path) == metadata["score_cache_sha256"] and matrix_path.name == metadata["score_cache_file"], "physical score archive hash", **context)
        with np.load(matrix_path, allow_pickle=False) as archive:
            matrices = {name:archive[name] for name in archive.files}
        scores_by_key[key] = matrices
        old = original[job]
        for field in ("name", "noise_ratio", "trial", "seed", "noise_std", "truth", "terms", "equations", "pilot"):
            check(metadata[field] == old[field], "paired original realization metadata", field=field, **context)
        for method in BASELINES:
            check(metadata["methods"][method] == old["methods"][method], "original baseline result copied unchanged", method=method, **context)
            audit_metrics(metadata["methods"][method], np.asarray(metadata["truth"]), dict(context, method=method))
        provenance = metadata["baseline_provenance"]
        check(provenance["trial_key"] == list(job) and provenance["study"] == str(OLD.relative_to(ROOT)), "baseline provenance source realization", **context)
        for prefix, child in (("metadata", "json"), ("matrix", "npz")):
            source = OLD/"prepared"/provenance[prefix+"_file"]
            check(source.suffix == "."+child and digest(source) == provenance[prefix+"_sha256"], "original paired cache provenance hash", part=prefix, **context)
        expected_truth = np.array([[equation.get(term, 0.) for term in metadata["terms"]] for equation in PDES[job[0]].truth])
        check(np.array_equal(expected_truth, metadata["truth"]), "physical dictionary truth matrix", **context)
        seed = independent_seed(*job)
        check(metadata["seed"] == seed, "independently reconstructed observation seed", **context)
        clean, grid = load(job[0], download=False)
        observed, deviations = observation(PDES[job[0]], clean, job[1], seed)
        check(metadata["noise_std"] == deviations, "independently reconstructed noise deviations", **context)
        check(array_hash(clean) == metadata["clean_sha256"], "actual clean state hash", **context)
        check(array_hash(observed) == metadata["observation_sha256"], "actual paired observed state hash", **context)
        base, pilot_info = estimate(observed, grid, PDES[job[0]], SplitPilot(ratio=.5, prior=.01))
        check(array_hash(base) == metadata["base_pilot_sha256"], "actual frozen training-only pilot hash", **context)
        for field, value in pilot_info.items():
            check(metadata["pilot"][field] == json.loads(json.dumps(value)), "frozen pilot metadata reconstructed", field=field, **context)
        training = np.arange(clean.shape[-2]) % 2 == 1
        regression = ~training
        check(metadata["training_time_indices"] == np.flatnonzero(training).tolist(), "training-only split indices", **context)
        check(metadata["regression_time_indices"] == np.flatnonzero(regression).tolist(), "regression split indices", **context)
        check(np.isnan(base[..., training, :]).all() and np.isfinite(base[..., regression, :]).all(), "actual pilot whole-time split mask", **context)
        delta_z = float(np.prod([axis[1]-axis[0] for axis in grid]))
        n_total = int(np.prod(clean.shape[:-1], dtype=np.int64))
        n_regression = int(np.prod(clean.shape[:-2], dtype=np.int64))*int(regression.sum())
        true, difference = clean[..., regression, :], base[..., regression, :]-clean[..., regression, :]
        # For the fixed odd/even split every retained node's time-cell weight is 2.
        def energies(error):
            square = np.sum(error*error, axis=-1)
            return float(delta_z*np.sum(2*square)), float(delta_z*np.sum(square))
        base_energy, base_raw = energies(difference)
        rebuild = (args.rebuild_scores_noiseless and job[1] == 0) or key in args.rebuild_score_jobs
        for mode in ("rate", "exact"):
            info = metadata["oracle"][mode]
            ctx = dict(context, mode=mode)
            check(info["mode"] == mode, "oracle mode label", **ctx)
            for field, value in dict(n_regression=n_regression, n_total=n_total, components=clean.shape[-1],
                                     regression_times=int(regression.sum()), training_times=int(training.sum())).items():
                check(info[field] == value, "oracle sample/component counts", field=field, **ctx)
            for field, value in dict(delta_z=delta_z, base_energy=base_energy, base_energy_unweighted=base_raw,
                                     exponent=manifest["oracle_exponent"], cap=manifest["oracle_cap"],
                                     time_weight_min=2., time_weight_max=2.).items():
                close(info[field], value, "oracle grid and base-error metadata", field=field, **ctx)
            target = float(manifest["oracle_cap"]*n_regression**(-manifest["oracle_exponent"]))
            close(info["target_energy"], target, "deterministic squared-error target", **ctx)
            alpha = info["alpha"]
            nominal = min(1., np.sqrt(target/base_energy)) if base_energy else 1.
            check(np.isfinite(alpha) and 0 <= alpha <= nominal*(1+1e-13), "oracle contraction bounded by derived rate", **ctx)
            if mode == "exact":
                check(alpha == 0, "exact oracle contraction is zero", **ctx)
            values = base[..., regression, :].copy() if alpha == 1 else true+alpha*difference
            pilot = np.full_like(clean, np.nan)
            pilot[..., regression, :] = values
            check(array_hash(pilot) == metadata["oracle_pilot_sha256"][mode], "actual oracle state follows clean-plus-contracted-error algebra", **ctx)
            actual_energy, actual_raw = energies(values-true)
            close(info["energy"], actual_energy, "directly integrated actual oracle squared error", atol=1e-14*target, **ctx)
            close(info["energy_unweighted"], actual_raw, "directly integrated actual raw oracle squared error", atol=1e-14*target, **ctx)
            check(actual_energy <= target and actual_raw <= target, "actual oracle satisfies requested physical error bound", **ctx)
            close(info["error_over_n_quarter"], actual_energy*n_regression**.25, "actual error over n^-1/4", **ctx)
            close(info["bound_over_n_quarter"], manifest["oracle_cap"]*n_regression**(.25-manifest["oracle_exponent"]), "vanishing deterministic ratio bound", **ctx)
            if mode == "exact":
                check(actual_energy == 0 and actual_raw == 0, "exact oracle actual zero error", **ctx)
            maxima["oracle_energy_over_target"] = max(maxima["oracle_energy_over_target"], actual_energy/target)
            maxima["rate_error_over_n_quarter"] = max(maxima["rate_error_over_n_quarter"], actual_energy*n_regression**.25)
            counts["actual_oracle_states_reconstructed"] += 1
            for variant in ("plugin", "debiased"):
                method = mode+"_"+variant+"_dantzig"
                score_info = metadata["score"][method]
                H, b = matrices[method+"_H"], matrices[method+"_b"]
                shape = (len(metadata["terms"]), len(metadata["terms"]))
                check(H.shape == shape and b.shape == (shape[0], len(expected_truth)), "physical score matrix dimensions", method=method, **context)
                check(np.isfinite(H).all() and np.isfinite(b).all() and np.array_equal(H, H.T), "finite symmetric physical score", method=method, **context)
                check(array_hash(H) == score_info["H_sha256"] and array_hash(b) == score_info["b_sha256"], "physical score array hashes", method=method, **context)
                scale = np.max(np.abs(b), axis=0)
                residual = np.max(np.abs(b-H@expected_truth.T), axis=0)
                close(score_info["lambda_max"], scale, "prepared score scale diagnostic", atol=1e-13*max(float(scale.max()),np.finfo(float).tiny), method=method, **context)
                close(score_info["truth_residual_inf"], residual, "prepared physical truth residual diagnostic", atol=1e-13*max(float(scale.max()),float(residual.max()),np.finfo(float).tiny), method=method, **context)
            if rebuild:
                system = SplitSystem(PDES[job[0]], observed, grid, X=pilot, Y=observed)
                X = system.columns(system.terms, pilot, rescale=False)
                Y = system.columns(system.lhs, pilot, rescale=False)
                dX = system.columns(system.terms, observed, pilot=pilot, rescale=False)-X
                dY = system.columns(system.lhs, observed, rescale=False)-Y
                K = len(X)
                for variant in ("plugin", "debiased"):
                    method = mode+"_"+variant+"_dantzig"
                    recomputed = {"H":X.T@X/K, "b":X.T@Y/K}
                    if variant == "debiased":
                        recomputed = {"H":(X.T@X+dX.T@X+X.T@dX)/K,
                                      "b":(X.T@Y+dX.T@Y+X.T@dY)/K}
                    for part, value in recomputed.items():
                        saved = matrices[method+"_"+part]
                        scale = max(float(np.max(np.abs(saved))),np.finfo(float).tiny)
                        error = float(np.max(np.abs(value-saved)))/scale
                        maxima["rebuilt_score_relative_difference"] = max(maxima["rebuilt_score_relative_difference"], error)
                        close(saved, value, "independently composed weak-score algebra", rtol=2e-11, atol=2e-13*scale, method=method, part=part, **context)
                        check(metadata["score"][method]["rows"] == K, "reconstructed weak-row count", method=method, **context)
                    counts["physical_score_pairs_rebuilt"] += 1
        if rebuild:
            rebuilt.append(key)
        exact_pair = tuple(array_hash(matrices["exact_plugin_dantzig_"+part]) for part in ("H", "b"))
        if job[0] in exact_scores:
            check(exact_pair == exact_scores[job[0]], "exact plugin physical scores identical across noises/seeds", **context)
        else:
            exact_scores[job[0]] = exact_pair
        if job[1] == 0:
            exact_noiseless_differences[job[0]] = {}
            for part in ("H", "b"):
                plugin, debiased = (matrices["exact_"+variant+"_dantzig_"+part] for variant in ("plugin", "debiased"))
                scale = max(float(np.max(np.abs(plugin))),np.finfo(float).tiny)
                error = float(np.max(np.abs(plugin-debiased)))/scale
                exact_noiseless_differences[job[0]][part] = error
                check(error < 1e-10, "noiseless exact plugin/debiased agree within polynomial roundoff", part=part, **context)
        counts["prepared_realizations"] += 1
        if counts["prepared_realizations"] % 8 == 0:
            print(f"[audit states] {counts['prepared_realizations']}/{len(metadata_paths)}", flush=True)

    def candidates_at_levels(result, truth, matrices, method, options, context):
        audit_metrics(result, truth, context)
        audit_metrics(result["incumbent_diagnostic"], truth, dict(context, diagnostic="incumbent"))
        check(len(result["paths"]) == len(truth), "one lambda path per equation", **context)
        H, B = matrices[method+"_H"], matrices[method+"_b"]
        winners = {level:[] for level in LEVELS}
        for equation, (path, target) in enumerate(zip(result["paths"], truth)):
            ctx = dict(context, equation=equation)
            b = B[:, equation]
            scale = float(np.max(np.abs(b)))
            safe_scale = max(scale, np.finfo(float).tiny)
            close(path["score_scale"], scale, "physical lambda path scale", atol=1e-13*safe_scale, **ctx)
            candidates = path["path"]
            eligible = {level:[] for level in LEVELS}
            for index, candidate in enumerate(candidates):
                item = dict(ctx, candidate=index)
                counts["candidate_solutions"] += 1
                lam = candidate["lambda"]
                close(lam, candidate["lambda_ratio"]*scale, "physical lambda value", atol=1e-13*safe_scale, **item)
                if candidate["coefficients"] is None:
                    check(not candidate["validated"] and not candidate["optimal"], "missing incumbent cannot be selected", **item)
                    continue
                coefficient = np.asarray(candidate["coefficients"], dtype=float)
                check(coefficient.shape == target.shape and np.isfinite(coefficient).all(), "finite physical candidate", **item)
                e2 = float(np.linalg.norm(coefficient-target)/np.linalg.norm(target))
                close(candidate["e2"], e2, "independent candidate coefficient error", **item)
                residual = float(np.max(np.abs(b-H@coefficient)))
                violation = max(0., residual-lam)
                slack = min(2e-8*safe_scale, .01*lam)+50*np.finfo(float).eps*safe_scale
                valid = bool(np.isfinite(coefficient).all() and np.isfinite(residual) and violation <= slack)
                close(candidate["residual_inf"], residual, "independent physical candidate score residual", atol=2e-13*safe_scale, **item)
                close(candidate["validation_slack"], slack, "physical score validation slack", atol=2e-15*safe_scale, **item)
                check(candidate["validated"] == valid, "independent physical feasibility eligibility", **item)
                support = np.asarray(candidate["support"], dtype=bool)
                check(support.shape == coefficient.shape and np.all(coefficient[~support] == 0), "SOS1 excluded physical coefficients exactly zero", **item)
                check(candidate["cardinality"] == int(np.count_nonzero(coefficient)), "candidate physical cardinality metadata", **item)
                permitted = int(support.sum())
                if candidate["cardinality_optimal"]:
                    lower = candidate["cardinality_bound"]
                    check(lower is not None and np.isfinite(lower) and np.ceil(lower-1e-7) >= permitted and lower <= permitted+1e-6,
                          "reported integer lower bound certifies sparsity", **item)
                if candidate["optimal"]:
                    check(candidate["cardinality_optimal"] and candidate["secondary_optimal"], "both optimization phases certified", **item)
                    check(candidate["secondary_status"] in (2,"AnalyticZero"), "certified secondary solver status", **item)
                if valid:
                    eligible["incumbent"].append(index)
                    if candidate["cardinality_optimal"]:
                        eligible["cardinality"].append(index)
                    if candidate["optimal"]:
                        eligible["lexicographic"].append(index)
                    maxima["candidate_constraint_violation_relative_to_b"] = max(maxima["candidate_constraint_violation_relative_to_b"], violation/safe_scale)
            for level in LEVELS:
                check(bool(eligible[level]), "eligible oracle candidate exists", level=level, **ctx)
                best = min(eligible[level], key=lambda index:(candidates[index]["e2"], candidates[index]["lambda_ratio"]))
                winners[level].append(candidates[best])
                if level == "lexicographic":
                    check(path["selected_index"] == best and path["selected"] == candidates[best], "certified oracle argmin and tie-break", **ctx)
                    check(path["coefficients"] == candidates[best]["coefficients"] == result["coefficients"][equation], "selected physical coefficients propagated without refit", **ctx)
                    check(result["solver"][equation]["selected"] == candidates[best], "solver audit selected candidate propagated", **ctx)
                if level == "incumbent":
                    check(path["incumbent_oracle_selected"] == candidates[best], "any-valid diagnostic oracle argmin", **ctx)
                    check(result["incumbent_diagnostic"]["coefficients"][equation] == candidates[best]["coefficients"], "incumbent diagnostic coefficients propagated", **ctx)
            ratios = [candidate["lambda_ratio"] for candidate in candidates]
            check(len(ratios) == len(set(ratios)), "unique lambda path", **ctx)
            base = sorted(set(options["ratios"]) | {1.}, reverse=True)
            check(ratios[:len(base)] == base, "declared initial lambda grid", **ctx)
            consumed = len(base)
            for _ in range(options["refinements"]):
                previous = candidates[:consumed]
                valid_previous = [c for c in previous if c["validated"] and np.isfinite(c["e2"])]
                best = min(valid_previous, key=lambda c:(c["e2"],c["lambda_ratio"]))
                seen = {c["lambda_ratio"] for c in previous}
                ordered = sorted(seen)
                position = ordered.index(best["lambda_ratio"])
                proposed = []
                if position:
                    proposed.append(float(np.sqrt(ordered[position-1]*ordered[position])))
                elif ordered[position] > 1e-12:
                    proposed.extend([ordered[position]/10,ordered[position]/100])
                if position+1 < len(ordered):
                    proposed.append(float(np.sqrt(ordered[position]*ordered[position+1])))
                new = []
                for ratio in proposed:
                    if ratio not in seen:
                        new.append(ratio);seen.add(ratio)
                check(ratios[consumed:consumed+len(new)] == new, "truth-only oracle refinement schedule", **ctx)
                consumed += len(new)
            check(consumed == len(candidates), "only declared grid and refinements are solved", **ctx)
            residual = float(np.max(np.abs(b-H@target)))
            close(result["solver"][equation]["truth_residual_inf"], residual, "runner physical truth score diagnostic", atol=2e-13*max(scale,residual), **ctx)
            counts["unique_equation_oracles"] += 1
        levels = {level:metrics([c["coefficients"] for c in selected],truth) for level,selected in winners.items()}
        check(levels["incumbent"]["e2"] <= levels["cardinality"]["e2"]+1e-12 <= levels["lexicographic"]["e2"]+2e-12,
              "nested oracle candidate sets improve weakly", **context)
        return winners

    solutions = {}
    for path in solution_paths:
        solution = json.loads(path.read_text())
        job, method = tuple(solution["job"]), solution["method"]
        context = dict(job=job, method=method)
        check((job,method) in fits_expected and (job,method) not in solutions, "unique solved system belongs to protocol", **context)
        solutions[job,method] = solution
        check(path.stem == key_for(job)+"__"+method and solution["signature"] == signature, "solver filename and signature", **context)
        metadata = metadata_by_key[key_for(job)]
        check(solution["score_cache_file"] == metadata["score_cache_file"] and solution["score_cache_sha256"] == metadata["score_cache_sha256"], "solver input cache provenance", **context)
        for part in ("H","b"):
            check(solution["input_score_sha256"][part] == metadata["score"][method][part+"_sha256"], "solver physical input score hash", part=part, **context)
        candidates_at_levels(solution["result"], np.asarray(metadata["truth"]), scores_by_key[key_for(job)], method, manifest["options"], context)
        counts["unique_score_fits"] += 1

    seen = set()
    for record in records:
        job = (record["name"],record["noise_ratio"],record["trial"])
        context = dict(job=job)
        check(job in expected and job not in seen, "unique joined protocol realization", **context)
        seen.add(job)
        metadata = metadata_by_key[key_for(job)]
        for field, value in metadata.items():
            if field != "methods":
                check(record[field] == value, "joined trial matches prepared metadata", field=field, **context)
        check(set(record["methods"]) == set(BASELINES+NEW), "all eight paired methods present", **context)
        for method in BASELINES:
            check(record["methods"][method] == original[job]["methods"][method], "joined original paired result unchanged", method=method, **context)
        for method in NEW:
            source_job = canonical[job[0]] if method == "exact_plugin_dantzig" else job
            solution = solutions[source_job,method]
            check(record["methods"][method] == solution["result"], "joined fit unchanged from solved result", method=method, **context)
            reuse = record["solver_reuse"][method]
            check(reuse == dict(source_job=list(source_job),source_method=method,reused=source_job != job,
                                solved_file=key_for(source_job)+"__"+method+".json"), "explicit solver reuse provenance", method=method, **context)
            for part in ("H","b"):
                check(metadata["score"][method][part+"_sha256"] == solution["input_score_sha256"][part], "reused fit has identical physical score", method=method, part=part, **context)
        counts["completed_realizations"] += 1
    counts["oracle_realization_pairs"] = 2*len(records)
    counts["new_method_realization_combinations"] = 4*len(records)
    if args.require_complete:
        check(seen == expected, "all requested 104 realizations completed")
        check(set(solutions) == fits_expected, "all unique physical score paths completed")
        check(len(metadata_by_key) == len(expected), "all requested state caches completed")
    for requested in args.rebuild_score_jobs:
        check(requested in rebuilt, "requested weak-score rebuild was completed", job_key=requested)
    summary_path = directory/"summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["completed_realizations"] == len(records):
            for scope, rows in dict(all=records,nonzero_noise=[r for r in records if r["noise_ratio"]>0],noiseless=[r for r in records if r["noise_ratio"]==0]).items():
                if not rows:
                    continue
                for level in LEVELS:
                    for method in BASELINES+NEW:
                        independent = []
                        for record in rows:
                            result = record["methods"][method]
                            if method in BASELINES[:2]:
                                coefficient = result["coefficients"]
                            else:
                                chosen = []
                                for path in result["paths"]:
                                    eligible = [c for c in path["path"] if c["validated"] and
                                        (level == "incumbent" or (c["optimal"] if level == "lexicographic" else c["cardinality_optimal"]))]
                                    chosen.append(min(eligible,key=lambda c:(c["e2"],c["lambda_ratio"]))["coefficients"])
                                coefficient = chosen
                            value = metrics(coefficient,np.asarray(record["truth"]))
                            value["all_zero"] = not np.any(coefficient)
                            independent.append(value)
                        expected_summary = dict(realizations=len(rows),mean_e2=float(np.mean([v["e2"] for v in independent])),
                            median_e2=float(np.median([v["e2"] for v in independent])),exact_support=sum(v["exact_support"] for v in independent),
                            all_zero=sum(v["all_zero"] for v in independent),mean_fp=float(np.mean([v["fp"] for v in independent])),mean_fn=float(np.mean([v["fn"] for v in independent])))
                        for field,value in expected_summary.items():
                            close(summary["aggregate"][scope][level][method][field],value,"independent reported aggregate",scope=scope,level=level,method=method,field=field)
            counts["report_summary_snapshots_checked"] += 1
        elif args.require_complete:
            check(False,"summary snapshot includes all final records")
    elif args.require_complete:
        check(False,"final summary exists")
    audit = dict(protocol="independent-oracle-pilot-audit-v1",passed=not failures,
        audited_at_utc=datetime.now(timezone.utc).isoformat(),executable=sys.executable,numpy=np.__version__,
        complete=seen==expected and set(solutions)==fits_expected,expected_realizations=len(expected),expected_unique_score_fits=len(fits_expected),
        prepared_realizations=len(metadata_by_key),completed_realizations=len(records),unique_score_fits=len(solutions),
        trial_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),ignored_in_progress_jsonl_tail=partial_tail,
        auditor_source_sha256=digest(__file__),counts=dict(counts),maxima=dict(maxima),
        weak_score_rebuilt_jobs=rebuilt,noiseless_exact_plugin_debiased_relative_difference=exact_noiseless_differences,
        failures=failures,limitations=[
            "Does not rerun Gurobi or independently reproduce branch-and-bound lower bounds; checks reported bound/certificate consistency.",
            "Reconstructs all prepared seeds, noisy observations, frozen base pilots, actual oracle states, energies and state hashes; only the explicitly listed jobs rebuild weak FFT score matrices.",
            "One grid size cannot empirically establish a little-o asymptotic rate or statistical confidence coverage; the oracle family enforces the displayed squared-error bound deterministically."])
    output = directory/"audit.json"
    temporary = output.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,output)
    print(json.dumps(dict(passed=audit["passed"],complete=audit["complete"],prepared=len(metadata_by_key),realizations=len(records),
                         score_fits=len(solutions),candidates=counts["candidate_solutions"],rebuilt_jobs=rebuilt,failures=len(failures),audit=str(output))))
    if failures:
        for failure in failures[:20]:
            print(json.dumps(failure))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
