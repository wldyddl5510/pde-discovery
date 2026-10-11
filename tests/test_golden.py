"""Saved benchmark trials are reproduced by the current code (skipped without data/results)."""
import json
from pathlib import Path
import unittest
import numpy as np
from pdes import DATA_DIR, PDES
from run import run_trial
from wsindy import Term

RESULTS = Path(__file__).parents[1]/"benchmark_results"
CASES = [("authors", "wsindy", "KS", .5, 3), ("authors", "wsindy", "NLS", .2, 1),
         ("filtered", "filtered", "KdV", .5, 2), ("lasso/original/raw", "lasso", "IB", .75, 0),
         ("debiasing/time2_foldwise/KS/mstls", "debiased", "KS", 1., 5), ("consistency/raw", "wsindy", "VBG", .5, 0)]


def saved(run, name, ratio, trial):
    base = RESULTS/run
    spec = json.loads((base/"results.manifest.json").read_text())["benchmarks"][name]
    labels = [Term(" ".join(n if p == 1 else f"{n}^{p}" for n, p in zip(spec["component_names"], t["powers"]) if p)
                   or "1", tuple(t["derivative"]), tuple(t["powers"])).label for t in spec["terms"]]
    with (base/"results_trials"/f"{name}.jsonl").open() as source:
        record = next(r for r in map(json.loads, source) if r["noise_ratio"] == ratio and r["trial"] == trial)
    return labels, np.array(record["coefficients"]).reshape(len(labels), -1).T


class Golden(unittest.TestCase):
    def test_saved_trials(self):
        for run, method, name, ratio, trial in CASES:
            if not (RESULTS/run).exists() or not (DATA_DIR/PDES[name].file).exists():
                self.skipTest("saved results or data missing")
            with self.subTest(run=run, pde=name):
                labels, old = saved(run, name, ratio, trial)
                new = np.array(run_trial(name, method, ratio, trial)["coefficients"])
                order = [t.label for t in PDES[name].dictionary_terms()[1]]
                new = new[:, [order.index(label) for label in labels]]
                np.testing.assert_array_equal(new != 0, old != 0)
                np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-12*np.max(np.abs(old)))


if __name__ == "__main__":
    unittest.main()
