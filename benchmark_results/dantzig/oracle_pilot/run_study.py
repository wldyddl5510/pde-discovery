"""Paired rate-controlled and exact-clean oracle-pilot Dantzig experiment.

Preparation reconstructs precisely the 104 observation realizations from the
completed sanity study.  Four score systems are constructed per realization;
exact-clean plugin systems are solved only once per PDE and explicitly reused.
All caches are atomic, source/data signed, and independently resumable.
"""
from __future__ import annotations

import os
for _variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OLD = ROOT/"benchmark_results/dantzig/sanity"
_spec = importlib.util.spec_from_file_location("prior_sanity_study", OLD/"run_study.py")
prior = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = prior
_spec.loader.exec_module(prior)
from pdes import DATA_DIR, PDES, load, observe, trial_seed
from wsindy import SplitPilot, SplitSystem, estimate
from wsindy.oracle_pilot import oracle_pilot
from wsindy.scores import score_matrices

digest, serializable, evaluate = prior.digest, prior.serializable, prior.evaluate
MODES = ("rate", "exact")
VARIANTS = ("plugin", "debiased")
NEW_METHODS = tuple(f"{mode}_{variant}_dantzig" for mode in MODES for variant in VARIANTS)
BASELINES = ("wsindy", "debiased_mstls", "plugin_dantzig", "debiased_dantzig")
METHODS = BASELINES+NEW_METHODS
LEVELS = ("lexicographic", "cardinality", "incumbent")


def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(serializable(value), allow_nan=False)+"\n")
    os.replace(temporary, path)


def scope_for(args):
    return [(name, noise, trial) for trial in range(args.trials) for name in args.pdes
            for noise in args.noise if noise != 0 or trial == 0]


def key_for(name, noise, trial):
    return f"{name}_noise-{noise:g}_trial-{trial}"


def source_inputs(args):
    sources = (Path(__file__).resolve(), ROOT/"wsindy/oracle_pilot.py", OLD/"run_study.py",
               ROOT/"pdes.py", ROOT/"wsindy/scores.py", ROOT/"wsindy/dantzig.py",
               ROOT/"wsindy/system.py", ROOT/"wsindy/regression.py", ROOT/"wsindy/estimators.py",
               ROOT/"wsindy/dictionary.py", ROOT/"wsindy/testfunctions.py")
    packages = {}
    for package in ("numpy","scipy","pysindy","gurobipy","scikit-learn","matplotlib"):
        try: packages[package] = version(package)
        except PackageNotFoundError: packages[package] = None
    return dict(sources={str(p.relative_to(ROOT)): digest(p) for p in sources}, packages=packages,
        data={name:dict(file=PDES[name].file, sha256=digest(DATA_DIR/PDES[name].file)) for name in args.pdes},
        old_study={filename:digest(OLD/filename) for filename in
                   ("manifest.json", "preparation_manifest.json", "trials.jsonl")})


def verify_old_study():
    manifest = json.loads((OLD/"manifest.json").read_text())
    for relative, expected in manifest["sources"].items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError(f"Old study numerical source changed: {relative}")
    for name, data in manifest["data"].items():
        if digest(DATA_DIR/data["file"]) != data["sha256"]:
            raise RuntimeError(f"Old study data changed: {name}")
    records = prior.load_records(OLD)
    old = {(r["name"], r["noise_ratio"], r["trial"]):r for r in records}
    if len(old) != len(records):
        raise RuntimeError("Duplicate old study realization")
    return old


def make_manifest(args):
    options = dict(ratios=sorted(set(args.ratios)), time_limit=args.time_limit,
                   mip_gap=0., refinements=args.refinements)
    scope = scope_for(args)
    return dict(protocol="paired-oracle-pilot-discrete-dantzig-v1", pdes=args.pdes,
        noise=args.noise, trials=args.trials, scope=scope, expected_realizations=len(scope),
        expected_unique_score_fits=3*len(scope)+len(args.pdes), modes=list(MODES),
        options=options, oracle_exponent=args.exponent, oracle_cap=args.cap,
        n_definition="regression spacetime nodes, excluding component multiplicity",
        energy="Delta z times SplitSystem time weights times squared vector error; raw sum also saved",
        rate_bound="energy <= cap*n_regression**(-exponent), exponent>1/4",
        pilot="SplitPilot(ratio=.5, prior=.01), same mask as original study",
        truth_use="clean state constructs oracle pilots; true beta only lambda choice/evaluation",
        no_refit=True, exact_plugin_reuse="one identical physical score per PDE",
        inputs=source_inputs(args))


def read_prepared(directory, key, signature):
    metadata_path = directory/"prepared"/(key+".json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    matrix_path = metadata_path.with_suffix(".npz")
    if metadata.get("signature") != signature:
        raise RuntimeError(f"Preparation signature changed: {metadata_path}")
    if digest(matrix_path) != metadata["score_cache_sha256"]:
        raise RuntimeError(f"Prepared score cache changed: {matrix_path}")
    with np.load(matrix_path, allow_pickle=False) as archive:
        matrices = {k:archive[k] for k in archive.files}
    p, q = len(metadata["terms"]), len(metadata["truth"])
    for method in NEW_METHODS:
        if matrices[method+"_H"].shape != (p,p) or matrices[method+"_b"].shape != (p,q):
            raise RuntimeError(f"Invalid score dimensions: {key}/{method}")
        if not all(np.all(np.isfinite(matrices[method+"_"+part])) for part in ("H", "b")):
            raise RuntimeError(f"Nonfinite physical scores: {key}/{method}")
    return metadata, matrices


def prepare_trial(job, directory, signature, exponent, cap):
    directory = Path(directory)
    name, noise, trial = job
    key = key_for(*job)
    cached = read_prepared(directory, key, signature)
    if cached is not None:
        return cached[0]
    start = perf_counter()
    original = verify_old_study()[tuple(job)]
    old_metadata_path, old_matrix_path = prior.cache_paths(OLD/"prepared", *job)
    old_metadata = json.loads(old_metadata_path.read_text())
    if digest(old_matrix_path) != old_metadata["score_cache_sha256"]:
        raise RuntimeError(f"Original physical score cache changed: {old_matrix_path}")
    clean, grid = load(name, download=False)
    pde = PDES[name]
    seed = trial_seed(name, noise, trial)
    U, noise_std = observe(pde, clean, noise, seed)
    if seed != original["seed"] or noise_std != original["noise_std"]:
        raise RuntimeError("Paired observation reconstruction does not match original seed/noise")
    base, base_info = estimate(U, grid, pde, SplitPilot(ratio=.5, prior=.01))
    others = tuple(axis for axis in range(base.ndim) if axis != base.ndim-2)
    training = np.isnan(base).all(axis=others)
    oracle_info, score_info, matrices, pilot_hashes = {}, {}, {}, {}
    for mode in MODES:
        pilot, info = oracle_pilot(clean, base, grid, mode=mode, exponent=exponent, cap=cap)
        system = SplitSystem(pde, U, grid, X=pilot, Y=U)
        terms = [term.label for term in system.terms]
        truth = pde.truth_matrix(system.terms)
        if terms != original["terms"] or not np.array_equal(truth, np.asarray(original["truth"])):
            raise RuntimeError("Paired dictionary/physical coefficients changed")
        if dict(system.sampling_info) != {key:original["pilot"][key] for key in system.sampling_info}:
            raise RuntimeError("Training/regression time split changed")
        if info["energy"] > info["target_energy"] or info["energy_unweighted"] > info["target_energy"]:
            raise RuntimeError("Oracle pilot violates requested squared-error bound")
        if mode == "exact" and info["energy"] != 0:
            raise RuntimeError("Exact oracle has nonzero error")
        oracle_info[mode] = info
        pilot_hashes[mode] = array_digest(pilot)
        for variant in VARIANTS:
            method = f"{mode}_{variant}_dantzig"
            score_start = perf_counter()
            score = score_matrices(system, variant=variant)
            matrices[method+"_H"], matrices[method+"_b"] = score.H, score.b
            score_info[method] = dict(rows=score.n_rows, shape=list(score.H.shape),
                rank=int(np.linalg.matrix_rank(score.H)), condition_number=float(np.linalg.cond(score.H)),
                symmetry_error=float(np.max(np.abs(score.H-score.H.T))),
                construction_runtime=perf_counter()-score_start,
                truth_residual_inf=np.max(np.abs(score.b-score.H@truth.T),axis=0).tolist(),
                lambda_max=np.max(np.abs(score.b),axis=0).tolist(),
                H_sha256=array_digest(score.H), b_sha256=array_digest(score.b))
    metadata = serializable(dict(name=name, noise_ratio=noise, trial=trial, seed=seed,
        noise_std=noise_std, truth=truth, terms=terms, equations=original["equations"],
        observation_sha256=array_digest(U), clean_sha256=array_digest(clean),
        base_pilot_sha256=array_digest(base), oracle_pilot_sha256=pilot_hashes,
        training_time_indices=np.flatnonzero(training).tolist(),
        regression_time_indices=np.flatnonzero(~training).tolist(),
        pilot=dict(base_info, **system.sampling_info), oracle=oracle_info, score=score_info,
        methods={m:original["methods"][m] for m in BASELINES},
        baseline_provenance=dict(study=str(OLD.relative_to(ROOT)), trial_key=list(job),
            metadata_file=old_metadata_path.name, metadata_sha256=digest(old_metadata_path),
            matrix_file=old_matrix_path.name, matrix_sha256=digest(old_matrix_path)),
        preparation_runtime=perf_counter()-start, signature=signature))
    matrix_path = directory/"prepared"/(key+".npz")
    temporary = matrix_path.with_suffix(f".npz.{os.getpid()}.tmp")
    with temporary.open("wb") as sink:
        np.savez_compressed(sink, **matrices)
    os.replace(temporary, matrix_path)
    metadata.update(score_cache_file=matrix_path.name, score_cache_sha256=digest(matrix_path))
    atomic_json(matrix_path.with_suffix(".json"), metadata)
    return metadata


def job_id(job, method, canonical):
    exact_reuse = method == "exact_plugin_dantzig"
    source_job = canonical[job[0]] if exact_reuse else job
    source_method = "exact_plugin_dantzig" if exact_reuse else method
    return key_for(*source_job)+"__"+source_method


def read_solution(directory, ident, signature):
    path = directory/"solved"/(ident+".json")
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("signature") != signature:
        raise RuntimeError(f"Solver signature changed: {path}")
    return value


def solve_job(job, method, directory, signature, options):
    directory = Path(directory)
    key = key_for(*job)
    ident = key+"__"+method
    existing = read_solution(directory, ident, signature)
    if existing is not None:
        return existing
    metadata, matrices = read_prepared(directory, key, signature)
    score = SimpleNamespace(H=matrices[method+"_H"], b=matrices[method+"_b"],
                            n_rows=metadata["score"][method]["rows"])
    result = prior.dantzig(score, np.asarray(metadata["truth"]), options)
    result["score_construction_runtime"] = metadata["score"][method]["construction_runtime"]
    value = dict(job=list(job), method=method, signature=signature, result=result,
        input_score_sha256=dict(H=metadata["score"][method]["H_sha256"],
                                b=metadata["score"][method]["b_sha256"]),
        score_cache_file=metadata["score_cache_file"], score_cache_sha256=metadata["score_cache_sha256"])
    atomic_json(directory/"solved"/(ident+".json"), value)
    return value


def assemble(directory, scope, signature, canonical):
    records = []
    for job in scope:
        key = key_for(*job)
        prepared = read_prepared(directory, key, signature)
        if prepared is None:
            continue
        metadata, _ = prepared
        solutions = {method:read_solution(directory, job_id(job,method,canonical), signature)
                     for method in NEW_METHODS}
        if not all(solutions.values()):
            continue
        metadata["solver_reuse"] = {}
        for method, solution in solutions.items():
            for part in ("H", "b"):
                if metadata["score"][method][part+"_sha256"] != solution["input_score_sha256"][part]:
                    raise RuntimeError(f"Reused score differs from solved source: {key}/{method}/{part}")
            metadata["methods"][method] = solution["result"]
            metadata["solver_reuse"][method] = dict(source_job=solution["job"],source_method=solution["method"],
                reused=solution["job"] != list(job) or solution["method"] != method,
                solved_file=job_id(job,method,canonical)+".json")
        records.append(metadata)
    path = directory/"trials.jsonl"
    temporary = path.with_suffix(f".jsonl.{os.getpid()}.tmp")
    with temporary.open("w") as sink:
        for record in records:
            sink.write(json.dumps(record,allow_nan=False)+"\n")
    os.replace(temporary,path)
    return records


def load_records(directory):
    path = directory/"trials.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def select(record, method, level):
    value = record["methods"][method]
    if method in BASELINES[:2]:
        return value
    selected = []
    for path in value["paths"]:
        if level == "lexicographic":
            candidate = path["selected"]
        else:
            eligible = [c for c in path["path"] if c.get("validated",False)
                and (level == "incumbent" or c.get("cardinality_optimal",False))]
            candidate = min(eligible,key=lambda c:(c["e2"],c["lambda_ratio"]))
        selected.append(candidate)
    output = evaluate([c["coefficients"] for c in selected],np.asarray(record["truth"]))
    output.update(selected=selected, runtime=value["runtime"],
        secondary_unresolved=sum(not c.get("secondary_optimal",False) for c in selected),
        cardinality_unresolved=sum(not c.get("cardinality_optimal",False) for c in selected))
    return output


def summarize(records, method, level):
    values = [select(record,method,level) for record in records]
    return dict(realizations=len(values),mean_e2=float(np.mean([v["e2"] for v in values])),
        median_e2=float(np.median([v["e2"] for v in values])),
        exact_support=sum(v["exact_support"] for v in values),
        all_zero=sum(not np.any(v["coefficients"]) for v in values),
        mean_fp=float(np.mean([v["fp"] for v in values])),mean_fn=float(np.mean([v["fn"] for v in values])),
        cardinality_unresolved=sum(v.get("cardinality_unresolved",0) for v in values),
        secondary_unresolved=sum(v.get("secondary_unresolved",0) for v in values))


def report(directory):
    records = load_records(directory)
    if not records:
        return
    manifest = json.loads((directory/"manifest.json").read_text())
    noisy = [r for r in records if r["noise_ratio"] > 0]
    clean = [r for r in records if r["noise_ratio"] == 0]
    scopes = dict(all=records,nonzero_noise=noisy,noiseless=clean)
    aggregate = {scope:{level:{method:summarize(rows,method,level) for method in METHODS}
        for level in LEVELS} for scope,rows in scopes.items() if rows}
    groups = list(dict.fromkeys((r["name"],r["noise_ratio"]) for r in records))
    settings = [dict(name=name,noise_ratio=noise,oracles={level:{method:summarize(
        [r for r in records if r["name"]==name and r["noise_ratio"]==noise],method,level)
        for method in METHODS} for level in LEVELS}) for name,noise in groups]
    payload = dict(completed_realizations=len(records),expected_realizations=manifest["expected_realizations"],
        expected_unique_score_fits=manifest["expected_unique_score_fits"],aggregate=aggregate,by_setting=settings)
    (directory/"summary.json").write_text(json.dumps(serializable(payload),indent=2)+"\n")
    csv_rows = [dict(name=s["name"],noise_ratio=s["noise_ratio"],level=level,method=method,**v)
        for s in settings for level in LEVELS for method,v in s["oracles"][level].items()]
    with (directory/"summary.csv").open("w",newline="") as sink:
        writer=csv.DictWriter(sink,fieldnames=list(csv_rows[0])); writer.writeheader();writer.writerows(csv_rows)
    lines=["# Oracle-pilot Dantzig sanity check","",
        f"Completed {len(records)}/{manifest['expected_realizations']} paired realizations; "
        f"{manifest['expected_unique_score_fits']} unique score paths in the planned full scope.","",
        "Two clean-state oracle pilots are evaluated with the same noisy observations, split masks, physical quadrature and dictionaries as the original sanity study. "
        "The rate oracle contracts the original pilot error direction until its squared error energy is at most n_reg^(-1/2); "
        "the exact oracle is clean u* at the regression nodes. Both use the clean state deliberately and are diagnostic oracles.","",
        "n_reg counts regression spacetime nodes, without counting vector components as separate nodes. "
        "Energy sums squared vector error with Delta z times SplitSystem time weights; the unweighted Delta z sum is also saved and bounded. "
        "Thus energy/n_reg^(-1/4) <= n_reg^(-1/4) tends to zero. This deterministic construction satisfies the probability requirement with probability one; "
        "a single grid is not an empirical asymptotic-rate validation.","",
        "The same finite lambda grid, truth-independent residual tie-break, time limits and no-refit rule are used. True beta only chooses lambda and evaluates coefficients. "
        "Exact plugin scores are identical within each PDE and are solved once then reused with score hashes checked. "
        "Original baseline and original-pilot results are copied byte-for-byte numerically from the completed study.","",
        "Oracle levels: lexicographic requires certified cardinality and residual tie-break; cardinality requires a certified Equation (9) minimizer; "
        "incumbent permits any physically feasible candidate, including unresolved cardinality. The last is only an optimization diagnostic.","",
        "## Aggregate coefficient recovery","",
        "| Scope | Oracle level | Method | N | Mean E2 | Median E2 | Exact support | All zero |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for scope,levels in aggregate.items():
        for level,methods in levels.items():
            for method,v in methods.items():
                if method in BASELINES[:2] and level != LEVELS[0]:continue
                lines.append(f"| {scope} | {level} | {method} | {v['realizations']} | {v['mean_e2']:.6g} | {v['median_e2']:.6g} | {v['exact_support']} | {v['all_zero']} |")
    lines += ["","## Per-setting cardinality-certified mean E2","",
        "| PDE | Noise | Native MSTLS | Original debiased Dantzig | Rate plugin | Rate debiased | Exact plugin | Exact debiased |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in settings:
        values=s["oracles"]["cardinality"]
        methods=("wsindy","debiased_dantzig")+NEW_METHODS
        lines.append(f"| {s['name']} | {s['noise_ratio']:g} | "+" | ".join(f"{values[m]['mean_e2']:.6g}" for m in methods)+" |")
    lines += ["","Full values, all three oracle levels, support counts and per-setting comparisons are in summary.json/csv. "
        "Every prepared JSON saves n, Delta z, energy, contraction alpha, target cap, pilot and observation hashes. "
        "Solver paths and certificates are saved in solved/*.json; trials.jsonl joins them with explicit reuse provenance.",""]
    (directory/"results.md").write_text("\n".join(lines))
    return payload


def run(args):
    directory=args.output.resolve();directory.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        report(directory);return
    if args.workers<1 or args.prepare_workers<1 or args.trials<1:
        raise ValueError("workers/trials must be positive")
    old=verify_old_study();scope=scope_for(args)
    if any(tuple(job) not in old for job in scope):raise ValueError("Requested realization missing from old paired study")
    manifest=make_manifest(args)
    prior.register_manifest(directory/"manifest.json",manifest)
    signature=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    for child in ("prepared","solved"): (directory/child).mkdir(exist_ok=True)
    canonical={name:next(job for job in scope if job[0]==name) for name in args.pdes}
    unique=[]
    for job in scope:
        for method in NEW_METHODS:
            if method!="exact_plugin_dantzig" or job==canonical[job[0]]:unique.append((job,method))
    prepared={key_for(*job) for job in scope if read_prepared(directory,key_for(*job),signature) is not None}
    done={job_id(job,method,canonical) for job,method in unique
          if read_solution(directory,job_id(job,method,canonical),signature) is not None}
    pending_prepare=[job for job in scope if key_for(*job) not in prepared]
    pending_solve=[(job,method) for job,method in unique if job_id(job,method,canonical) not in done]
    print(f"Start: prepared {len(prepared)}/{len(scope)}, solved {len(done)}/{len(unique)}, "
          f"prepare workers={args.prepare_workers}, solver workers={args.workers}",flush=True)
    start=perf_counter();errors=[]
    with ProcessPoolExecutor(args.prepare_workers) as prep_pool,ProcessPoolExecutor(args.workers) as solve_pool:
        prep_futures={};solve_futures={}
        while pending_prepare or prep_futures or pending_solve or solve_futures:
            while pending_prepare and len(prep_futures)<args.prepare_workers:
                job=pending_prepare.pop(0)
                future=prep_pool.submit(prepare_trial,job,directory,signature,args.exponent,args.cap)
                prep_futures[future]=job
            if not args.prepare_only:
                for job,method in list(pending_solve):
                    if len(solve_futures)>=args.workers:break
                    if key_for(*job) in prepared:
                        future=solve_pool.submit(solve_job,job,method,directory,signature,manifest["options"])
                        solve_futures[future]=(job,method);pending_solve.remove((job,method))
            elif not pending_prepare and not prep_futures:
                break
            active=set(prep_futures)|set(solve_futures)
            if not active:
                if pending_solve:raise RuntimeError("Pending solver jobs have no prepared inputs")
                break
            completed,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in completed:
                is_prep=future in prep_futures
                job=prep_futures.pop(future) if is_prep else solve_futures.pop(future)
                try:value=future.result()
                except Exception as error:
                    failure=dict(stage="prepare" if is_prep else "solve",job=job,error=repr(error))
                    with (directory/"errors.jsonl").open("a") as sink:sink.write(json.dumps(failure)+"\n")
                    errors.append(failure);print(f"FAILED {failure}",flush=True);continue
                if is_prep:
                    prepared.add(key_for(*job))
                    print(f"[prepare {len(prepared)}/{len(scope)}] {key_for(*job)} "
                        f"alpha={value['oracle']['rate']['alpha']:.3g},energy={value['oracle']['rate']['energy']:.3g}, "
                        f"time={value['preparation_runtime']:.1f}s elapsed={perf_counter()-start:.1f}s",flush=True)
                else:
                    source_job,method=job;done.add(job_id(source_job,method,canonical))
                    result=value["result"]
                    print(f"[solve {len(done)}/{len(unique)}] {key_for(*source_job)} {method} "
                        f"E2={result['e2']:.5g},exact={result['exact_support']},time={result['runtime']:.1f}s "
                        f"elapsed={perf_counter()-start:.1f}s",flush=True)
                    if len(done)%8==0:
                        joined=assemble(directory,scope,signature,canonical)
                        print(f"[checkpoint] joined {len(joined)}/{len(scope)} realizations",flush=True)
            if not completed:print(f"[heartbeat] prepared={len(prepared)}/{len(scope)},solved={len(done)}/{len(unique)}, "
                                  f"active={len(solve_futures)},elapsed={perf_counter()-start:.1f}s",flush=True)
    records=assemble(directory,scope,signature,canonical)
    if records:report(directory)
    if errors:raise RuntimeError(f"{len(errors)} jobs failed; inspect errors.jsonl then resume")
    print(f"Finished: prepared {len(prepared)}/{len(scope)}, solved {len(done)}/{len(unique)}, "
          f"joined {len(records)}/{len(scope)}, elapsed={perf_counter()-start:.1f}s",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdes",nargs="+",choices=tuple(PDES),default=list(PDES))
    parser.add_argument("--noise",type=float,nargs="+",default=[0.,.2,.5,.75,1.])
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--prepare-workers",type=int,default=1)
    parser.add_argument("--exponent",type=float,default=.5)
    parser.add_argument("--cap",type=float,default=1.)
    parser.add_argument("--time-limit",type=float,default=5.)
    parser.add_argument("--refinements",type=int,default=1)
    parser.add_argument("--ratios",type=float,nargs="+",default=np.logspace(-6,0,13).tolist())
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
