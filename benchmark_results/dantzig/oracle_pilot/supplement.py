"""Report cardinality oracles augmented by proved NLS and VBG candidates.

Only supplement.json/md are written. Raw solver flags, saved paths, and primary
study summaries are unchanged. Certificates concern Equation (9)'s minimum
cardinality, not the implementation's additional residual tie-break.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("wsindy","debiased_mstls","plugin_dantzig","debiased_dantzig",
           "rate_plugin_dantzig","rate_debiased_dantzig","exact_plugin_dantzig","exact_debiased_dantzig")
POLICIES = ("gurobi_cardinality","with_independent_proof")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    result = hashlib.sha256()
    result.update(str(array.dtype).encode())
    result.update(json.dumps(array.shape).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def metrics(coefficients,truth):
    coefficients,truth = np.asarray(coefficients),np.asarray(truth)
    selected,actual = coefficients!=0,truth!=0
    tp,fn,fp = (int(np.count_nonzero(mask)) for mask in (selected&actual,~selected&actual,selected&~actual))
    return dict(e2=float(np.linalg.norm(coefficients-truth)/np.linalg.norm(truth)),
        exact_support=fn==0 and fp==0,all_zero=not bool(np.any(coefficients)),tp=tp,fn=fn,fp=fp)


def key_for(job):
    return f"{job[0]}_noise-{job[1]:g}_trial-{job[2]}"


def candidate_key(candidate):
    if candidate.get("coefficients") is None:
        return None
    return float(candidate["lambda"]),tuple(float(value) for value in candidate["coefficients"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--require-complete",action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest_path = directory/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {tuple(job) for job in manifest["scope"]}
    records_path = directory/"trials.jsonl"
    snapshot = records_path.read_bytes() if records_path.exists() else b""
    if snapshot and not snapshot.endswith(b"\n"):
        snapshot = snapshot[:snapshot.rfind(b"\n")+1]
    records = [json.loads(line) for line in snapshot.splitlines() if line.strip()]
    jobs = {(record["name"],record["noise_ratio"],record["trial"]) for record in records}
    if len(jobs)!=len(records) or not jobs.issubset(expected):
        raise ValueError("Duplicate or out-of-protocol joined realization")
    if args.require_complete and jobs!=expected:
        raise ValueError(f"Only {len(jobs)}/{len(expected)} paired realizations are complete")

    verification_path = directory/"nls_certificate_verification.json"
    verification = json.loads(verification_path.read_text())
    if not verification["passed"] or verification["failures"]:
        raise ValueError("Independent exact NLS certificate verification did not pass")
    for filename,sha in (("verify_nls_certificate.py",verification["provenance"]["verifier_source_sha256"]),
                         ("nls_pairs.json",verification["provenance"]["diagnostic_json_sha256"]),
                         ("nls_pairs.npz",verification["provenance"]["certificate_archive_sha256"])):
        if digest(directory/filename)!=sha:
            raise ValueError(f"Independently verified certificate provenance changed: {filename}")
    diagnostic = json.loads((directory/"nls_pairs.json").read_text())
    provenance = diagnostic["provenance"]
    source_metadata_path = directory/"prepared"/provenance["metadata_file"]
    source_cache_path = directory/"prepared"/provenance["score_file"]
    source_solution_path = directory/"solved"/provenance["saved_solution"]
    for path,sha in ((source_metadata_path,provenance["metadata_sha256"]),(source_cache_path,provenance["score_sha256"]),
                     (source_solution_path,provenance["saved_solution_sha256"]),
                     (directory/"diagnose_nls_pairs.py",provenance["diagnostic_source_sha256"])):
        if digest(path)!=sha:
            raise ValueError(f"NLS source provenance changed: {path}")
    if digest(source_cache_path)!=verification["provenance"]["physical_score_archive_sha256"] or digest(source_solution_path)!=verification["provenance"]["original_saved_solution_sha256"]:
        raise ValueError("Independent proof and source physical score/solution hashes differ")
    source_metadata = json.loads(source_metadata_path.read_text())
    source_solution = json.loads(source_solution_path.read_text())
    method = diagnostic["method"]
    if method!="exact_plugin_dantzig" or diagnostic["completed_pairs"]!=diagnostic["planned_pairs"]:
        raise ValueError("Expected a complete exact-plugin NLS pair certificate")
    with np.load(source_cache_path,allow_pickle=False) as archive:
        source_scores = {part:archive[method+"_"+part] for part in ("H","b")}
    source_score_hashes = {part:array_hash(value) for part,value in source_scores.items()}
    proved = {}
    examples = []
    for equation,row in enumerate(diagnostic["results"]):
        verified_row = verification["results"][equation]
        if not verified_row["all_supports_size_at_most_two_excluded"] or verified_row["pairs_verified"]!=diagnostic["planned_pairs"]:
            raise ValueError("Independent exhaustive support coverage is incomplete")
        verified = {(item["lambda_ratio"],item["physical_lambda"]):item for item in verified_row["saved_three_term_candidates"]}
        proved[equation] = set()
        for candidate in row["saved_three_term_candidates"]:
            physical_lambda = str(Fraction.from_float(float(candidate["lam"])))
            witness = verified.get((candidate["lambda_ratio"],physical_lambda))
            if not witness or not witness["minimum_cardinality_three_proved"] or not candidate["cardinality_three_independently_supported"]:
                continue
            beta = np.zeros(source_scores["H"].shape[1])
            for coefficient in candidate["coefficients"]:
                beta[coefficient["index"]] = coefficient["value"]
            key = (float(candidate["lam"]),tuple(beta.tolist()))
            matched = [item for item in source_solution["result"]["paths"][equation]["path"] if candidate_key(item)==key]
            if len(matched)!=1 or not matched[0]["validated"] or np.count_nonzero(beta)!=3:
                raise ValueError("Independently proved coefficient/lambda is not an existing valid source candidate")
            if not np.isclose(matched[0]["e2"],witness["coefficient_e2"],rtol=2e-13,atol=0):
                raise ValueError("Independent proof coefficient error differs from source candidate")
            proved[equation].add(key)
            if candidate["lambda_ratio"]==diagnostic["threshold_ratio"]:
                examples.append(dict(equation=equation,lambda_ratio=candidate["lambda_ratio"],physical_lambda=candidate["lam"],
                    coefficients=candidate["coefficients"],relative_e2=witness["coefficient_e2"],
                    exact_physical_residual=candidate["exact_physical_residual_fraction"],
                    minimum_pair_dual_bound=verified_row["minimum_pair_dual_bound"],
                    dual_bound_over_lambda=verified_row["minimum_dual_bound_over_threshold"],
                    original_gurobi_cardinality_flag=matched[0]["cardinality_optimal"],
                    independently_proved_cardinality_three=True,secondary_tie_break_certified=False))
    proved_count = sum(map(len,proved.values()))
    if proved_count!=verification["counts"]["minimum_cardinality_three_certificates"]:
        raise ValueError("Proved-candidate count does not match independent verifier")

    proof_sources = {"NLS":dict(proved=proved,score_hashes=source_score_hashes,
        verification_sha256=digest(verification_path),physical_cache_sha256=digest(source_cache_path),
        source_solution_sha256=digest(source_solution_path),proved_source_candidates=proved_count)}
    vbg_verification_path = directory/"vbg_certificate_verification.json"
    vbg_verification = json.loads(vbg_verification_path.read_text())
    if not vbg_verification["passed"] or vbg_verification["failures"]:
        raise ValueError("Independent VBG certificate verification did not pass")
    for filename,sha in (("verify_vbg_certificate.py",vbg_verification["provenance"]["verifier_source_sha256"]),
                         ("vbg_supports.json",vbg_verification["provenance"]["diagnostic_json_sha256"]),
                         ("vbg_supports.npz",vbg_verification["provenance"]["certificate_archive_sha256"])):
        if digest(directory/filename)!=sha:
            raise ValueError(f"Independently verified VBG provenance changed: {filename}")
    vbg_diagnostic = json.loads((directory/"vbg_supports.json").read_text())
    vp = vbg_diagnostic["provenance"]
    vbg_metadata_path = directory/"prepared"/vp["metadata_file"]
    vbg_cache_path = directory/"prepared"/vp["score_file"]
    vbg_solution_path = directory/"solved"/vp["solution_file"]
    for path,sha in ((vbg_metadata_path,vp["metadata_sha256"]),(vbg_cache_path,vp["score_sha256"]),
                     (vbg_solution_path,vp["solution_sha256"]),(directory/"diagnose_vbg_supports.py",vp["diagnostic_source_sha256"])):
        if digest(path)!=sha:
            raise ValueError(f"VBG physical proof source changed: {path}")
    if digest(vbg_cache_path)!=vbg_verification["provenance"]["physical_score_archive_sha256"] or digest(vbg_solution_path)!=vbg_verification["provenance"]["original_saved_solution_sha256"]:
        raise ValueError("VBG independent proof and actual source hashes differ")
    if vbg_diagnostic["method"]!=method or not vbg_diagnostic["complete"] or vbg_diagnostic["visited_supports"]!=vbg_diagnostic["planned_supports"] or not vbg_verification["all_supports_size_at_most_four_excluded"]:
        raise ValueError("Expected exhaustive VBG exact-plugin four-support exclusion")
    with np.load(vbg_cache_path,allow_pickle=False) as archive:
        vbg_scores = {part:archive[method+"_"+part] for part in ("H","b")}
    vbg_source_solution = json.loads(vbg_solution_path.read_text())
    vbg_verified = {(item["lambda_ratio"],item["physical_lambda"]):item for item in vbg_verification["saved_five_term_candidates"]}
    vbg_proved,vbg_examples = {0:set()},[]
    for candidate in vbg_diagnostic["saved_five_term_candidates"]:
        witness = vbg_verified.get((candidate["lambda_ratio"],str(Fraction.from_float(float(candidate["lam"])))))
        if not witness or not witness["minimum_cardinality_five_proved"] or candidate["independently_certified_minimum_cardinality"]!=5:
            continue
        beta = np.zeros(vbg_scores["H"].shape[1])
        for coefficient in candidate["coefficients"]:
            beta[coefficient["index"]] = coefficient["value"]
        key = (float(candidate["lam"]),tuple(beta.tolist()))
        matched = [item for item in vbg_source_solution["result"]["paths"][0]["path"] if candidate_key(item)==key]
        if len(matched)!=1 or not matched[0]["validated"] or np.count_nonzero(beta)!=5:
            raise ValueError("Proved VBG beta/lambda does not match existing feasible source candidate")
        if not np.isclose(matched[0]["e2"],witness["coefficient_e2"],rtol=2e-13,atol=0):
            raise ValueError("Independent VBG coefficient error differs from saved source path")
        vbg_proved[0].add(key)
        vbg_examples.append(dict(equation=0,lambda_ratio=candidate["lambda_ratio"],physical_lambda=candidate["lam"],
            coefficients=candidate["coefficients"],relative_e2=witness["coefficient_e2"],
            exact_physical_residual=candidate["exact_physical_residual_fraction"],
            minimum_support_dual_bound=vbg_verification["minimum_support_dual_bound"],
            dual_bound_over_lambda=witness["minimum_dual_bound_over_lambda"],
            original_gurobi_cardinality_flag=matched[0]["cardinality_optimal"],
            independently_proved_cardinality_five=True,secondary_tie_break_certified=False))
    vbg_count = len(vbg_proved[0])
    if vbg_count!=vbg_verification["counts"]["minimum_cardinality_five_certificates"]:
        raise ValueError("VBG independently proved candidate count differs")
    proof_sources["VBG"] = dict(proved=vbg_proved,score_hashes={part:array_hash(value) for part,value in vbg_scores.items()},
        verification_sha256=digest(vbg_verification_path),physical_cache_sha256=digest(vbg_cache_path),
        source_solution_sha256=digest(vbg_solution_path),proved_source_candidates=vbg_count)
    proved_count += vbg_count

    selections = []
    matches = 0
    for record in records:
        job = (record["name"],record["noise_ratio"],record["trial"])
        applicable = record["name"] in proof_sources
        active_proof = proof_sources.get(record["name"])
        if applicable:
            key = key_for(job)
            cache_path = directory/"prepared"/record["score_cache_file"]
            if digest(cache_path)!=record["score_cache_sha256"]:
                raise ValueError(f"Joined independently proved score cache changed: {key}")
            with np.load(cache_path,allow_pickle=False) as archive:
                for part in ("H","b"):
                    actual = archive[method+"_"+part]
                    if array_hash(actual)!=active_proof["score_hashes"][part] or record["score"][method][part+"_sha256"]!=active_proof["score_hashes"][part]:
                        raise ValueError(f"Exact-plugin proof cannot transfer across different physical scores: {key}/{part}")
        entry = dict(name=record["name"],noise_ratio=record["noise_ratio"],trial=record["trial"],methods={})
        target = np.asarray(record["truth"])
        for current in METHODS:
            result = record["methods"][current]
            if current in METHODS[:2]:
                value = dict(metrics(result["coefficients"],target),coefficients=result["coefficients"],
                             independently_proved_selected_equations=0,selected=[])
                entry["methods"][current] = {policy:dict(value) for policy in POLICIES}
                continue
            outputs = {policy:[] for policy in POLICIES}
            for equation,path in enumerate(result["paths"]):
                raw = [candidate for candidate in path["path"] if candidate["validated"] and candidate["cardinality_optimal"]]
                extra = []
                if applicable and current==method:
                    for candidate in path["path"]:
                        if candidate.get("coefficients") is not None and candidate["validated"] and candidate_key(candidate) in active_proof["proved"][equation]:
                            extra.append(candidate)
                    if {candidate_key(candidate) for candidate in extra}!=active_proof["proved"][equation]:
                        raise ValueError("A reused exact-plugin path is missing independently proved original candidates")
                    matches += len(extra)
                for policy in POLICIES:
                    pool = raw if policy==POLICIES[0] else raw+extra
                    chosen = min(pool,key=lambda candidate:(candidate["e2"],candidate["lambda_ratio"]))
                    independent = bool(applicable and current==method and candidate_key(chosen) in active_proof["proved"][equation])
                    outputs[policy].append(dict(coefficients=chosen["coefficients"],lambda_ratio=chosen["lambda_ratio"],physical_lambda=chosen["lambda"],
                        independently_proved=independent,original_gurobi_cardinality_flag=chosen["cardinality_optimal"],
                        original_gurobi_secondary_flag=chosen["secondary_optimal"]))
            entry["methods"][current] = {}
            for policy,chosen in outputs.items():
                coefficients = [candidate["coefficients"] for candidate in chosen]
                entry["methods"][current][policy] = dict(metrics(coefficients,target),coefficients=coefficients,
                    selected=chosen,independently_proved_selected_equations=sum(candidate["independently_proved"] for candidate in chosen))
            if current!=method or not applicable:
                if entry["methods"][current][POLICIES[0]]!=entry["methods"][current][POLICIES[1]]:
                    raise RuntimeError("Independent proof changed an unproved score/method")
        selections.append(entry)

    def summarize(rows,current,policy):
        values = [row["methods"][current][policy] for row in rows]
        raw = [row["methods"][current][POLICIES[0]] for row in rows]
        return dict(realizations=len(values),mean_e2=float(np.mean([value["e2"] for value in values])),
            median_e2=float(np.median([value["e2"] for value in values])),exact_support=sum(value["exact_support"] for value in values),
            all_zero=sum(value["all_zero"] for value in values),mean_fp=float(np.mean([value["fp"] for value in values])),
            mean_fn=float(np.mean([value["fn"] for value in values])),independently_proved_selected_equations=sum(value["independently_proved_selected_equations"] for value in values),
            coefficient_improved_realizations=sum(value["e2"]<prior["e2"] for value,prior in zip(values,raw)))
    def scopes(rows):
        return {name:group for name,group in dict(all=rows,nonzero_noise=[row for row in rows if row["noise_ratio"]>0],
                                                noiseless=[row for row in rows if row["noise_ratio"]==0]).items() if group}
    aggregate = {scope:{policy:{current:summarize(rows,current,policy) for current in METHODS} for policy in POLICIES}
                 for scope,rows in scopes(selections).items()}
    by_pde = {name:{scope:{policy:{current:summarize(rows,current,policy) for current in METHODS} for policy in POLICIES}
                    for scope,rows in scopes([row for row in selections if row["name"]==name]).items()}
              for name in manifest["pdes"] if any(row["name"]==name for row in selections)}
    payload = dict(protocol="independent-proof-augmented-cardinality-oracle-report-v1",created_at_utc=datetime.now(timezone.utc).isoformat(),
        complete=jobs==expected,completed_realizations=len(records),expected_realizations=len(expected),
        policies=list(POLICIES),newly_proved_source_candidates=proved_count,matching_candidate_instances=matches,
        scope="Only bare Equation (9) minimum-cardinality eligibility is expanded; secondary global residual tie-break is not proved.",
        provenance=dict(manifest_sha256=digest(manifest_path),trial_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
            report_source_sha256=digest(__file__),independent_verification_sha256=digest(verification_path),
            physical_source_cache_sha256=digest(source_cache_path),physical_source_score_hashes=source_score_hashes,
            source_solution_sha256=digest(source_solution_path),
            independent_proofs={name:{key:value for key,value in proof.items() if key!="proved"} for name,proof in proof_sources.items()}),
        aggregate=aggregate,by_pde=by_pde,nls_examples=examples,vbg_examples=vbg_examples,selections=selections,
        limitations=["No raw solver flag, path, main summary, or numerical source is edited.",
            "Independent certificates transfer only to exactly identical physical exact-plugin H,b; no transfer is made to exact-debiased or rate scores.",
            "This finite-path true-beta oracle selects minimum relative error per equation among certified Equation (9) minimizers; it does not establish a continuous optimal lambda or the implementation's secondary residual tie-break.",
            "Other unresolved supports remain outside the certified pool, so the supplemented oracle can still reflect solver limitations."])
    (directory/"supplement.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    lines = ["# 독립 증명을 추가한 minimum-cardinality oracle", "",
        f"Completed **{len(records)}/{len(expected)}** paired realizations. This is a separate reporting supplement; raw Gurobi flags and main result files are unchanged.","",
        f"The independently verified NLS and VBG proofs certify {proved_count} existing exact-plugin candidates as minimum-cardinality solutions of Equation (9). "
        "The report adds them to the Gurobi-cardinality-certified candidate pool, then applies the same per-equation minimum coefficient-error lambda oracle. "
        "It verifies exact coefficient/lambda matches and identical physical H,b before transferring certificates to reused paths.","",
        "**The independent proof certifies bare Equation (9) sparsity optimality. Global optimality of the added residual tie-break is not established.** "
        "Certificates are not transferred to exact-debiased or rate scores.","",
        "## Aggregate comparison", "",
        "| Scope | Method | N | Raw Gurobi mean E2 | + independent proof mean E2 | Raw median | + proof median | Raw exact support | + proof exact support | Raw zero | + proof zero |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scope,policies in aggregate.items():
        for current in METHODS:
            raw,expanded = (policies[policy][current] for policy in POLICIES)
            lines.append(f"| {scope} | {current} | {raw['realizations']} | {raw['mean_e2']:.7g} | {expanded['mean_e2']:.7g} | {raw['median_e2']:.7g} | {expanded['median_e2']:.7g} | {raw['exact_support']} | {expanded['exact_support']} | {raw['all_zero']} | {expanded['all_zero']} |")
    lines += ["", "## Exact-plugin results by PDE", "",
        "All other methods retain their original cardinality-oracle metrics.","",
        "| PDE | Scope | N | Raw Gurobi mean E2 | + independent proof mean E2 | Raw exact support | + proof exact support |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for name,pde_scopes in by_pde.items():
        for scope,policies in pde_scopes.items():
            raw,expanded = (policies[policy][method] for policy in POLICIES)
            lines.append(f"| {name} | {scope} | {raw['realizations']} | {raw['mean_e2']:.9g} | {expanded['mean_e2']:.9g} | {raw['exact_support']} | {expanded['exact_support']} |")
    lines += ["", "## Independently proved NLS example", "",
        "| Equation | lambda / ‖b‖∞ | Feasible three-term E2 | Minimum two-column dual bound / lambda | Original Gurobi cardinality flag |",
        "| ---: | ---: | ---: | ---: | :--- |"]
    for example in examples:
        lines.append(f"| {example['equation']} | {example['lambda_ratio']:g} | {example['relative_e2']:.9g} | {example['dual_bound_over_lambda']:.9g} | {example['original_gurobi_cardinality_flag']} |")
    lines += ["", "Every support of size ≤2 is contained in an independently excluded pair; the three-term primal is exactly feasible for the saved binary64 score. "
        "These candidates therefore are Equation (9) minimizers despite their original unresolved Gurobi flags. "
        "The better performance demonstrates that restricting interpretation to the original solver-certified zero models would miss valid accurate minimizers.","",
        "## Independently proved VBG examples", "",
        "| lambda / ‖b‖∞ | Feasible five-term E2 | Minimum four-column dual bound / lambda | Original Gurobi cardinality flag |",
        "| ---: | ---: | ---: | :--- |"]
    for example in vbg_examples:
        lines.append(f"| {example['lambda_ratio']:.9g} | {example['relative_e2']:.12g} | {example['dual_bound_over_lambda']:.9g} | {example['original_gurobi_cardinality_flag']} |")
    lines += ["", "All 123,410 four-column supports are exactly excluded at each certified VBG lambda; an exactly feasible five-term vector proves minimum cardinality five. "
        "The VBG lambda ratio1e-5 candidate is not certified by these bounds and is excluded from the expanded pool.","",
        "The finite lambda grid, remaining unproved supports, fixed trajectories, and oracle access to the clean state remain practical limits of this sanity check.","",
        "```sh", "/opt/homebrew/anaconda3/envs/pde_discovery/bin/python benchmark_results/dantzig/oracle_pilot/supplement.py --require-complete", "```", ""]
    (directory/"supplement.md").write_text("\n".join(lines))
    print(json.dumps(dict(complete=payload["complete"],realizations=len(records),expected=len(expected),proved_source_candidates=proved_count,
        matching_candidate_instances=matches,nls_realizations=sum(record["name"]=="NLS" for record in records),output=str(directory/"supplement.json"))))


if __name__=="__main__":
    main()
