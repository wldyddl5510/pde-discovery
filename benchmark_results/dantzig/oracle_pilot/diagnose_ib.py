"""Independent exhaustive support-one LP diagnosis of IB oracle-pilot scores.

Uses SciPy/HiGHS, independently of the main MILP solver. Truth labels fits only
after each optimization; it is never an input to a support's LP objective or
constraints. Saved results cover every dictionary column and every available IB
realization, both oracle pilots and both physical scores.
"""
from __future__ import annotations

import os
for _variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LP_SOURCE = ROOT/"benchmark_results/dantzig/sanity/diagnose_scores.py"
_spec = importlib.util.spec_from_file_location("independent_original_score_diagnosis", LP_SOURCE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
one_column_paths, serializable = _module.one_column_paths, _module.serializable
METHODS = tuple(f"{mode}_{variant}_dantzig" for mode in ("rate", "exact") for variant in ("plugin", "debiased"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory/"manifest.json").read_text())
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    expected = {tuple(job) for job in manifest["scope"] if job[0] == "IB"}
    paths = sorted((directory/"prepared").glob("IB_noise-*_trial-*.json"))
    records, jobs = [], set()
    # A conservative absolute normalized-residual margin: the independent LP
    # helper checks objective/physical-residual agreement to this tolerance.
    ranking_tolerance = 1e-9
    counts = Counter()
    for metadata_path in paths:
        metadata = json.loads(metadata_path.read_text())
        matrix_path = metadata_path.with_suffix(".npz")
        job = (metadata["name"],metadata["noise_ratio"],metadata["trial"])
        if job not in expected or job in jobs or metadata["signature"] != signature:
            raise ValueError(f"Unexpected, duplicate, or stale IB metadata: {metadata_path}")
        jobs.add(job)
        if digest(matrix_path) != metadata["score_cache_sha256"]:
            raise ValueError(f"Score cache hash changed: {matrix_path}")
        truth = np.asarray(metadata["truth"],dtype=float)
        if truth.shape[0] != 1 or np.count_nonzero(truth) != 1:
            raise ValueError("IB diagnostic requires its one-equation, one-term ground truth")
        target = truth[0]
        true_index = int(np.flatnonzero(target)[0])
        record = dict(name="IB",noise_ratio=job[1],trial=job[2],seed=metadata["seed"],
            oracle=metadata["oracle"],
            provenance=dict(metadata_file=metadata_path.name,metadata_sha256=digest(metadata_path),
                score_cache_file=matrix_path.name,score_cache_sha256=digest(matrix_path),
                observation_sha256=metadata["observation_sha256"],clean_sha256=metadata["clean_sha256"],
                base_pilot_sha256=metadata["base_pilot_sha256"],oracle_pilot_sha256=metadata["oracle_pilot_sha256"],
                score_arrays={method:{part:metadata["score"][method][part+"_sha256"] for part in ("H","b")} for method in METHODS}),
            scores={})
        with np.load(matrix_path,allow_pickle=False) as archive:
            for method in METHODS:
                H, b = archive[method+"_H"],archive[method+"_b"][:,0]
                result = one_column_paths(H,b,target,metadata["terms"])
                solutions = result["all_one_column_solutions"]
                if len(solutions) != len(target) or not all(row["success"] and row["independently_validated"] for row in solutions):
                    raise RuntimeError(f"An exhaustive support-one LP failed physical validation: {job}/{method}")
                true_fit = next(row for row in solutions if row["index"] == true_index)
                wrong_fit = min((row for row in solutions if row["index"] != true_index),key=lambda row:row["normalized_residual"])
                best = result["best_one_column"]
                gap = true_fit["normalized_residual"]-wrong_fit["normalized_residual"]
                if gap > ranking_tolerance:
                    ranking = "wrong_support_strictly_better_at_recorded_tolerance"
                elif gap < -ranking_tolerance:
                    ranking = "true_support_strictly_better_at_recorded_tolerance"
                else:
                    ranking = "unresolved_residual_tie_at_recorded_tolerance"
                record["scores"][method] = dict(result,
                    true_term=metadata["terms"][true_index],true_coefficient=float(target[true_index]),
                    optimized_true_term_fit=true_fit,best_wrong_term_fit=wrong_fit,
                    truth_coefficient_feasibility_ratio=float(np.max(np.abs(b-H@target))/np.max(np.abs(b))),
                    true_minus_best_wrong_residual=gap,
                    ranking=ranking,
                    best_is_true_term=bool(best["index"]==true_index and best["coefficient"]!=0),
                    best_to_true_support_residual_ratio=float(best["normalized_residual"]/true_fit["normalized_residual"]) if true_fit["normalized_residual"] else None)
                counts["score_systems"] += 1
                counts["support_one_linear_programs"] += len(solutions)
                counts[ranking] += 1
        records.append(record)
        print(f"[IB LP] noise={job[1]:g},trial={job[2]}: "+", ".join(
            method.removesuffix("_dantzig")+"="+record["scores"][method]["best_one_column"]["term"] for method in METHODS),flush=True)
    if args.require_complete and jobs != expected:
        raise RuntimeError(f"Only {len(jobs)}/{len(expected)} IB realizations are available")
    grouped = {}
    for scope, rows in dict(all=records,nonzero_noise=[r for r in records if r["noise_ratio"]>0],noiseless=[r for r in records if r["noise_ratio"]==0]).items():
        if not rows:
            continue
        grouped[scope] = {method:dict(realizations=len(rows),best_is_true_term=sum(r["scores"][method]["best_is_true_term"] for r in rows),
            wrong_support_strictly_better=sum(r["scores"][method]["true_minus_best_wrong_residual"]>ranking_tolerance for r in rows),
            true_support_strictly_better=sum(r["scores"][method]["true_minus_best_wrong_residual"] < -ranking_tolerance for r in rows),
            unresolved_residual_ties=sum(abs(r["scores"][method]["true_minus_best_wrong_residual"])<=ranking_tolerance for r in rows),
            median_best_coefficient_e2=float(np.median([r["scores"][method]["best_one_column"]["coefficient_e2"] for r in rows])),
            median_optimized_true_coefficient_e2=float(np.median([r["scores"][method]["optimized_true_term_fit"]["coefficient_e2"] for r in rows]))) for method in METHODS}
    payload = dict(protocol="independent-oracle-pilot-IB-support-one-LP-v1",complete=jobs==expected,
        created_at_utc=datetime.now(timezone.utc).isoformat(),expected_realizations=len(expected),realizations=len(records),
        counts=dict(counts),ranking_tolerance=ranking_tolerance,
        provenance=dict(manifest_sha256=digest(directory/"manifest.json"),signature=signature,
                        script_sha256=digest(__file__),lp_helper=str(LP_SOURCE.relative_to(ROOT)),lp_helper_sha256=digest(LP_SOURCE),
                        numpy=version("numpy"),scipy=version("scipy")),grouped=grouped,records=records,
        interpretation=[
            "Every LP optimizes its coefficient and common infinity-residual radius without true coefficient/support input; truth is used afterward for labels and errors.",
            "The LP uses one common score normalization by ||b||_inf and a coefficient substitution, preserving the physical infinity-norm objective. No score-coordinate reweighting is applied.",
            "For lambda below ||b||_inf but at least the global best nonzero one-term radius, minimum cardinality is one. A true support with a strictly larger optimal radius cannot win the implemented residual tie-break in that interval.",
            "This is a statement about the implemented residual tie-break, not an impossibility for bare Equation (9): the true and wrong one-term supports can both be Equation (9) minimizers when lambda admits both.",
            "Below the best one-term radius a denser solution may be optimal. This diagnostic does not characterize those supports or their coefficient errors.",
            "The numerical LP ranking uses the stated normalized tolerance and an extended-precision residual accumulator. It is not an exact-rational certificate of all LP optimal values.",
            "This snapshot includes only available prepared IB caches; rerun when later seeds are prepared."])
    json_path = directory/"ib_diagnostics.json"
    json_path.write_text(json.dumps(serializable(payload),indent=2,allow_nan=False)+"\n")
    lines = ["# IB oracle-pilot support 진단", "",
        f"**{len(records)}/{len(expected)}개 IB realization**, 각 4 score에서 43개 one-term support를 모두 SciPy/HiGHS LP로 검사했다. 총 {counts['support_one_linear_programs']}개 LP가 physical constraint validation을 통과했다.","",
        "True coefficient/support는 각 LP를 푼 뒤의 라벨과 계수 오차 계산에만 사용했다. 공통 score infinity norm으로 나누고 계수 변수를 치환했으며, score coordinate별 재가중은 적용하지 않았다.","",
        "## Noise가 있는 조건", "",
        "| Oracle / score | N | 최선이 true term | Wrong residual이 더 작음 | True residual이 더 작음 | 경계 tie | 최선 계수 median E2 | True-term 계수 median E2 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, row in grouped.get("nonzero_noise",{}).items():
        lines.append(f"| {method} | {row['realizations']} | {row['best_is_true_term']} | {row['wrong_support_strictly_better']} | {row['true_support_strictly_better']} | {row['unresolved_residual_ties']} | {row['median_best_coefficient_e2']:.6g} | {row['median_optimized_true_coefficient_e2']:.6g} |")
    lines += ["", "## 개별 결과", "",
        "R은 최적화한 score residual을 ||b||∞로 나눈 값이다. ΔR = true-term R - 가장 좋은 wrong-term R이며, 양수면 wrong support가 더 작은 residual을 갖는다.","",
        "| Noise | Trial | Oracle / score | 최선 term | 계수 | 계수 E2 | 최선 R | True-term 계수 | True-term R | ΔR |",
        "| ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for record in sorted(records,key=lambda row:(row["noise_ratio"],row["trial"])):
        for method in METHODS:
            row = record["scores"][method]
            best,true = row["best_one_column"],row["optimized_true_term_fit"]
            lines.append(f"| {record['noise_ratio']:g} | {record['trial']} | {method} | {best['term']} | {best['coefficient']:.8g} | {best['coefficient_e2']:.6g} | {best['normalized_residual']:.7g} | {true['coefficient']:.8g} | {true['normalized_residual']:.7g} | {row['true_minus_best_wrong_residual']:.7g} |")
    lines += ["", "## 해석 범위", "",
        "λ가 최선 one-term residual 이상이고 ||b||∞ 미만이면 최적 sparsity는 1이다. 이 구간에서 true term의 최적 residual이 wrong term보다 크면, 현재 구현이 추가한 residual tie-break는 true support를 선택하지 않는다.","",
        "이는 **식 (9) 자체의 복원 불가능성 주장과 다르다**. λ가 두 support를 모두 허용하면 true support와 wrong support가 식 (9)의 minimizer로 함께 존재할 수 있다. 최선 one-term residual보다 작은 λ에서 선택되는 더 큰 support의 성능은 이 진단으로 결정되지 않는다.","",
        f"수치 ranking은 정규화 residual 차이 {ranking_tolerance:g}를 기준으로 구분했다. LP objective와 직접 계산한 physical residual을 비교하고 extended accumulator residual도 저장했다. Exact rational optimality certificate는 아니다.","",
        "모든 support별 coefficient/residual/LP status, score 및 pilot provenance, alpha/실제 pilot error는 ib_diagnostics.json에 저장되어 있다.","",
        "```sh", "/opt/homebrew/anaconda3/envs/pde_discovery/bin/python benchmark_results/dantzig/oracle_pilot/diagnose_ib.py --require-complete", "```", ""]
    (directory/"ib_diagnostics.md").write_text("\n".join(lines))
    print(json.dumps(dict(complete=payload["complete"],realizations=len(records),linear_programs=counts["support_one_linear_programs"],output=str(json_path))))


if __name__ == "__main__":
    main()
