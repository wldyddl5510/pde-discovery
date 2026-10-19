# Oracle-pilot Dantzig sanity study

This follow-up changes the nuisance pilot while retaining the previous paired
observations, dictionaries, weak tests, split quadrature, physical score units,
and oracle-lambda protocol. Both oracle pilots deliberately use the clean state;
they are diagnostics, not procedures available from noisy observations.
Here `u_*` is the repository's stored clean benchmark trajectory.

## The nuisance assumption and the two oracles

The requested assumption is on the **squared** discrete L2 error:

\[
\Delta z\sum_{i=1}^{n}|\widehat u(z_i)-u_*(z_i)|^2=o(n^{-1/4}).
\]

For vector PDEs, the pointwise square is the sum over components. The component
count does not multiply the number of spacetime observations `n`.

Let `base` be the existing training-only 50:50 `SplitPilot`. On the regression
nodes, define

\[
E_{\rm base}=\Delta z\sum_{i\in I_{\rm reg}}w_i
\|\widetilde u(z_i)-u_*(z_i)\|_2^2,
\quad
a_n=\min\{1,\sqrt{n_{\rm reg}^{-1/2}/E_{\rm base}}\}.
\]

- **Rate oracle:** `u_rate = clean + a_n * (base-clean)`. If the base error is
  zero, retain the base. Floating-point construction is checked and can use a
  slightly stronger contraction to enforce the bound.
- **Exact oracle:** `u_exact = clean` on regression nodes.

For the rate oracle, every realization satisfies

\[
E_{\rm rate}\le n_{\rm reg}^{-1/2},\qquad
\frac{E_{\rm rate}}{n_{\rm reg}^{-1/4}}
\le n_{\rm reg}^{-1/4}\longrightarrow0.
\]

Thus the assumption holds pathwise, with probability 1, for every confidence
level `1-delta/2`. The exact oracle has error zero. This is an explicitly defined
rate-guaranteed oracle family; experiments on one grid size do not independently
verify an asymptotic rate. If a condition on the L2 **norm**, rather than its
square, is desired, this exponent must be revisited.

Here `n_reg` is the number of regression spacetime nodes, excluding components
and distinct from the number of weak equations `K`. `delta_z` is the product of
the original physical coordinate spacings. We use the same positive time weights
as `SplitSystem` (`w_i=2` for the 50:50 split). The weighted error bound is stronger
than the original unweighted `delta_z * sum(error**2)`, which is also recorded.
The full-grid and regression sample sizes differ by a fixed factor. A conceptual
full-grid oracle can be set to the clean state at training nodes, adding zero
error; the actual score input preserves the NaN training mask.

The shrinkage uses training-only pilot information and deterministic clean truth,
so it preserves independence from regression observation noise. True coefficients
and supports are not used to construct either pilot or solve a fixed-lambda fit.

## Paired experiment

- Eight PDEs: IB, KdV, KS, NLS, RD, NS, HKS, VBG.
- Observation noise ratios: 0, .2, .5, .75, 1. Noiseless once; each nonzero
  ratio uses the original three seeds: 104 paired observation realizations.
- Both oracles are evaluated at every realization: 208 oracle-realization pairs.
- Each oracle supplies a pilot plug-in score and a first-order debiased score.
- Native WSINDy/MSTLS and the previous split-debiased MSTLS results are reused
  from the original, provenance-checked study. Their pilots are not replaced.
- The score and optimizer remain `wsindy/scores.py` and `wsindy/dantzig.py`.
  There is no row reweighting, dictionary normalization, true-support input,
  coefficient-magnitude bound, or post-selection least-squares refit.
- Lambda ratios, refinement, per-equation coefficient-error oracle, solver
  tolerances, and certification levels match the previous study. The finite
  grid and time limits do not establish a continuous globally optimal lambda.
- Exact-pilot plug-in scores do not use noisy observations. They are computed
  and solved once per PDE and reused with explicit provenance. Debiased scores
  retain regression noise even with an exact pilot and require paired evaluation.

This assumption alone does not control polynomial derivative constants, physical
score conditioning, quadrature error, or unresolved mixed-integer optimization.
Those effects must remain separate from the pilot-error guarantee when
interpreting recovery.

## Implementation

`wsindy/oracle_pilot.py` constructs and checks the two oracle pilots.
`tests/test_oracle_pilot.py` checks physical weights, vector states, split masks,
deterministic bounds, and floating-point edge cases. All results from the prior
study remain in `../sanity/`.

From the repository root, using the `pde_discovery` Python environment:

```sh
python benchmark_results/dantzig/oracle_pilot/run_offline.py --workers 6 --prepare-workers 1
```

The wrapper selects the bundled size-limited Gurobi license for this process
only. Prepared matrices and completed solver paths resume after their input
signatures are checked. Six solvers each use one thread; preparation uses a
separate worker.

```sh
python benchmark_results/dantzig/oracle_pilot/run_offline.py --report-only
python benchmark_results/dantzig/oracle_pilot/pilot_summary.py
python benchmark_results/dantzig/oracle_pilot/audit.py --require-complete --rebuild-scores-noiseless --rebuild-score-jobs IB_noise-0.2_trial-0
python benchmark_results/dantzig/oracle_pilot/diagnose_ib.py --require-complete
python benchmark_results/dantzig/oracle_pilot/supplement.py --require-complete
python benchmark_results/dantzig/oracle_pilot/plot_results.py
```

- `manifest.json`: numerical options, source/data hashes, and paired-study inputs.
- `prepared/*.json`: oracle error budgets, actual errors, alpha, sample sizes,
  cell volumes, observation/state hashes, and copied baseline provenance.
- `prepared/*.npz`: four physical score systems for each paired realization.
- `solved/*.json`: 320 distinct score paths, with coefficients and solver statuses.
- `trials.jsonl`: 104 paired records joining both oracles with explicit score reuse.
- `results.md`, `summary.json`, `summary.csv`: coefficient and support recovery
  with all three certification criteria.
- `recovery.png`: cardinality-certified comparison for all eight PDEs.
- `recovery_with_proof.png`: the same comparison with independently proved
  NLS/VBG exact-plugin candidates included; generate the supplement first.
- `solver_levels.png`: rate/exact debiased scores under the three certificate
  criteria, separating unresolved optimization from coefficient recovery.
- `audit.json`: independent seed/noise/state/error-budget reconstruction,
  score/candidate/metric/provenance validation, and optional weak-score rebuilds.
- `pilot_summary.md/csv/json`: actual per-PDE sample sizes, physical cell volumes,
  contraction ranges, squared error energies, and their ratios to the rate bound.
- `ib_diagnostics.md/json`: independent exhaustive one-term support LPs, including
  the distinction between bare Equation (9) minimizers and the implementation's
  residual tie-break.
- `nls_pairs.md/json/npz`: noiseless exact-pilot NLS diagnostic. All 17,955
  two-column supports per equation are excluded at lambda/||b||_inf=.01 by exact
  rational dual witnesses for the saved physical H/b. Saved three-term candidates
  are exactly feasible and have E2 about 2.8e-5, establishing minimum cardinality
  three independently of the five-second solver flags. Their global secondary
  residual optimality is not asserted; main results are left unchanged.
- `nls_certificate_verification.md/json`: a second implementation independently
  verifies every saved rational dual witness and three-term primal candidate,
  without rerunning the pair LPs (`verify_nls_certificate.py`).
- `vbg_supports.md/json/npz` and `vbg_certificate_verification.md/json`: exhaustive
  four-column scans and an independent rational proof verification for VBG's
  exact-plugin score. All 123,410 supports are excluded below residual ratio
  9.64588e-6. Three saved five-term candidates at smaller lambdas are minimum
  cardinality solutions with E2=2.526e-10. The five-term candidate at ratio 1e-5
  is correctly excluded from this certificate pool.
- `supplement.md/json`: an additional bare-Equation-(9) cardinality oracle that
  admits the independently proved NLS/VBG exact-plugin candidates. Every added
  coefficient/lambda and physical score is matched to the saved solver paths and
  verified proof. It does not modify raw flags or the primary report, and does
  not assert global residual tie-break optimality.
