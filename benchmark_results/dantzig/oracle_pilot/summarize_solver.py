"""Describe unique solver sources without counting reused exact-plugin paths twice."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

METHODS=("rate_plugin_dantzig","rate_debiased_dantzig","exact_plugin_dantzig","exact_debiased_dantzig")


def summarize(directory):
    manifest=json.loads((directory/"manifest.json").read_text())
    solutions=[json.loads(path.read_text()) for path in sorted((directory/"solved").glob("*.json"))]
    output=dict(completed_unique_score_paths=len(solutions),
        expected_unique_score_paths=manifest["expected_unique_score_fits"],
        definition="Unique source solves over every noise condition; reused exact-plugin paths count once per PDE.",
        methods={})
    for method in METHODS:
        values=[value for value in solutions if value["method"]==method]
        paths=[path for value in values for path in value["result"]["paths"]]
        candidates=[candidate for path in paths for candidate in path["path"]]
        nonanalytic=[c for c in candidates if c.get("status")!="AnalyticZero"]
        output["methods"][method]=dict(unique_score_paths=len(values),equation_paths=len(paths),
            candidates=len(candidates),validated_candidates=sum(c.get("validated",False) for c in candidates),
            cardinality_certified_candidates=sum(c.get("validated",False) and c.get("cardinality_optimal",False) for c in candidates),
            lexicographic_certified_candidates=sum(c.get("validated",False) and c.get("optimal",False) for c in candidates),
            nonanalytic_candidates=len(nonanalytic),
            cardinality_certified_nonanalytic=sum(c.get("validated",False) and c.get("cardinality_optimal",False) for c in nonanalytic),
            lexicographic_certified_nonanalytic=sum(c.get("validated",False) and c.get("optimal",False) for c in nonanalytic),
            status_counts=dict(Counter(c.get("status","unknown") for c in candidates)))
    (directory/"solver_coverage.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    print(json.dumps(summarize(parser.parse_args().output.resolve()),indent=2))
