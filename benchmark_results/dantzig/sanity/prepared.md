# Prepared paired score systems

104 realizations have paired MSTLS baselines and physical-unit plugin/debiased score matrices cached.

The Dantzig estimator has not been evaluated by this preparation stage. No Gurobi license or true support is used.

Each prepared/*.json file contains seeds, true coefficients for later evaluation, pilot metadata, baseline results, score diagnostics and the SHA256 hash of its matching prepared/*.npz matrix file. preparation_manifest.json records source/data hashes; mismatched caches are rejected on restart.

```sh
# Reuse these matrices to run the solver once its license is available:
python benchmark_results/dantzig/sanity/run_study.py --workers 2
```
