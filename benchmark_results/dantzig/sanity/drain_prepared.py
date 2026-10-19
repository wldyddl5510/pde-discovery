"""Consume completed score caches while the independent preparation run continues.

    GRB_LICENSE_FILE=/path/to/gurobi.lic python benchmark_results/dantzig/sanity/drain_prepared.py

This companion never builds a missing cache and never changes numerical sources.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, sleep
from types import SimpleNamespace

import numpy as np

import run_study as study


def solve_cached(job, options, cache_directory, preparation_hash):
    # The parent schedules only completed metadata/archive pairs. Validate once
    # more in the child before entering run_trial's cache-only fast path.
    if study.read_prepared(cache_directory, *job, preparation_hash, matrices=False) is None:
        raise RuntimeError(f"Cache disappeared before solver scheduling: {job}")
    return study.run_trial(*job, options, cache_directory, preparation_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--workers-during-preparation", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--time-limit", type=float, default=5.)
    parser.add_argument("--mip-gap", type=float, default=0.)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--ratios", type=float, nargs="+", default=np.logspace(-6, 0, 13).tolist())
    args = parser.parse_args()
    directory = args.output.resolve()
    (directory/"solver.pid").write_text(str(os.getpid())+"\n")
    preparation = json.loads((directory/"preparation_manifest.json").read_text())
    preparation_hash = hashlib.sha256(json.dumps(preparation, sort_keys=True).encode()).hexdigest()
    scope = {(name, noise, trial) for name in preparation["pdes"] for noise in preparation["noise"]
             for trial in range(1 if noise == 0 else preparation["trials"])}
    config_args = SimpleNamespace(pdes=preparation["pdes"], noise=preparation["noise"], trials=preparation["trials"])
    options = dict(ratios=sorted(set(args.ratios)), time_limit=args.time_limit,
                   mip_gap=args.mip_gap, refinements=args.refinements)
    manifest = study.make_manifest(config_args, options)
    manifest["preparation_hash"] = preparation_hash
    study.register_manifest(directory/"manifest.json", manifest)
    cache_directory = directory/"prepared"
    done = {(r["name"], r["noise_ratio"], r["trial"]) for r in study.load_records(directory)}
    failed, scheduled, futures = set(), set(), {}
    start = last_status = perf_counter()
    print(f"Cached-only solver: {len(done)}/{len(scope)} done, workers={args.workers}, "
          f"ratios={len(options['ratios'])}, primary time limit={args.time_limit:g}s, pid={os.getpid()}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        while len(done|failed) < len(scope):
            available = set()
            for path in cache_directory.glob("*.json"):
                metadata = json.loads(path.read_text())
                job = (metadata["name"], metadata["noise_ratio"], metadata["trial"])
                if job in scope:
                    available.add(job)
            ready = sorted(available-done-failed-scheduled,
                           key=lambda job: (job[2], job[1], list(study.PDES).index(job[0])))
            active_limit = args.workers if len(available) == len(scope) else min(args.workers, args.workers_during_preparation)
            for job in ready[:max(0, active_limit-len(futures))]:
                future = pool.submit(solve_cached, job, options, cache_directory, preparation_hash)
                futures[future] = job
                scheduled.add(job)
            if futures:
                finished, _ = wait(futures, timeout=2., return_when=FIRST_COMPLETED)
                for future in finished:
                    name, noise, trial = job = futures.pop(future)
                    scheduled.remove(job)
                    try:
                        record = future.result()
                    except Exception as error:
                        failed.add(job)
                        with (directory/"errors.jsonl").open("a") as sink:
                            sink.write(json.dumps(dict(name=name, noise_ratio=noise, trial=trial, error=repr(error)))+"\n")
                        print(f"FAILED {name} noise={noise:g} trial={trial}: {error!r}", flush=True)
                        continue
                    with (directory/"trials.jsonl").open("a") as sink:
                        sink.write(json.dumps(record, allow_nan=False)+"\n")
                    done.add(job)
                    methods = record["methods"]
                    db, plugin = methods["debiased_dantzig"], methods["plugin_dantzig"]
                    print(f"[solve {len(done)}/{len(scope)}; {len(available)} prepared] {name} noise={noise:g} trial={trial}: "
                          f"E2 plugin/db={plugin['e2']:.4g}/{db['e2']:.4g}; "
                          f"incumbent={plugin['incumbent_diagnostic']['e2']:.4g}/{db['incumbent_diagnostic']['e2']:.4g}; "
                          f"exact={db['exact_support']}; solver_stage={record['solver_stage_runtime']:.1f}s; "
                          f"elapsed={perf_counter()-start:.1f}s", flush=True)
                    last_status = perf_counter()
            else:
                if (directory/"preparation_summary.json").exists() and len(available) < len(scope):
                    raise SystemExit(f"Preparation ended with only {len(available)}/{len(scope)} caches; inspect preparation_errors.jsonl")
                sleep(2.)
            if perf_counter()-last_status >= 30:
                print(f"Solver progress: {len(done)}/{len(scope)} done, {len(futures)} running, "
                      f"{len(available)}/{len(scope)} prepared; elapsed={perf_counter()-start:.1f}s", flush=True)
                last_status = perf_counter()
    if done:
        study.report(directory)
    if failed:
        raise SystemExit(f"{len(failed)} cached solver jobs failed; inspect errors.jsonl and restart to retry")
    print(f"Completed {len(done)} cached solver realizations in {perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
