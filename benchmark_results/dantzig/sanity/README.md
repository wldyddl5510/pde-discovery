# Discrete Dantzig sanity check

This study evaluates equations (9)–(10) of `pde_discovery (20).pdf`, using the
first-order debiased score on page 9. It does not replace the existing benchmark
results or change their regression implementations.

## Run

From the repository root, in the `pde_discovery` Python environment:

```sh
python benchmark_results/dantzig/sanity/run_offline.py --workers 2
```

`run_offline.py` selects gurobipy's bundled size-limited license in this process
only. These linear mixed-integer models fit its size limits; no license-server
connection or user license-file edit is needed. Use `run_study.py` directly to
select a different license normally. Completed trials and prepared weak matrices
are reused after their provenance hashes are checked.

```sh
# Rebuild tables/plot from completed trial records:
python benchmark_results/dantzig/sanity/run_offline.py --report-only
```

## Fixed protocol

- All eight existing PDEs, original trajectories, dictionaries and test functions.
- Noise ratios 0, .2, .5, .75 and 1. One noiseless realization and three independent
  observation-noise seeds per nonzero ratio: 104 paired realizations in total.
- Baselines: original WSINDy/MSTLS and the current split, residual-debiased MSTLS.
- New estimators: pilot plug-in score Dantzig and first-order debiased-score
  Dantzig, using the same 50:50 SplitPilot and regression quadrature.
- All score matrices and reported coefficients are in physical units. There is
  no row weighting, dictionary normalization, coefficient magnitude prior, or
  support supplied from the true PDE. Numerical column substitutions in the
  solver preserve the original score constraint exactly in real arithmetic.
- `lambda = ratio * ||b||_inf`: 13 logarithmic ratios from 1e-6 to 1, followed by
  one refinement around the best validated candidate. A lower-boundary optimum
  adds smaller ratios; remaining boundary optima are reported.
- Oracle selection minimizes physical relative coefficient E2 separately for
  each equation and realization. Truth selects/refines lambda only. This is an
  optimistic diagnostic, not a data-driven tuning rule or a fair comparison of
  tuning rules against native MSTLS selection.
- Minimize cardinality with SOS1 constraints, then minimize the score infinity
  residual at that cardinality. No OLS refit follows. Solver seed 0, one thread.
- Primary time limit 5 seconds per lambda; secondary limit 2 seconds. Main results
  require cardinality and secondary optimality certificates plus physical-score
  feasibility. A separate diagnostic admits validated time-limited incumbents.
  Partial certification can leave parts of the lambda path unresolved; a zero
  selected model alone does not establish failure of the estimator.

## Files

- `results.md`, `summary.csv`, `results.png`: final results and solver audit tables.
- `interpretation.md`, `interpretation.csv`: separate fully lexicographic,
  cardinality-only certified, and all-validated incumbent oracle comparisons.
- `oracle_levels.png`: the same three candidate sets compared with native WSINDy.
- `trials.jsonl`: complete coefficient paths, oracle choices and solver statuses.
- `manifest.json`: numerical options, source/data hashes and package versions.
- `prepared/`: paired baselines and physical H/b matrices, with archive hashes.
- `prepared.md`: preparation counts and cache description.
- `diagnose_scores.py`: independent score-conditioning/support diagnostics.
- `diagnostics.md`: interpretation of those diagnostics and the separately labeled
  clean-pilot oracle experiment; the latter is excluded from main performance.
- `audit_results.py`: independent checks of saved observations, choices and metrics.
- `audit_recomputed.py`: additional metric/physical-feasibility recomputation.

`wsindy/scores.py` implements the score, and `wsindy/dantzig.py` implements the
SOS1 optimizer. The tests compare score derivatives against finite differences,
the optimizer against exhaustive support enumeration, and recovery on an exactly
sparse noiseless score problem.
