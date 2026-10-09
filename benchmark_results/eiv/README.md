# Scalar Section 6 EIV sanity study

Run from the repository root:

```sh
python benchmark_results/eiv/run_sanity.py
```

The default is **25 fits**: IB, KdV, KS, HKS and VBG; noise ratios 0, 0.2,
0.5, 0.75 and 1; one trial per setting. It uses only existing verified data
(`download=False`). The original benchmark noise instances are reproduced
with `experiments.trial_seed(0, name, noise_ratio, trial)` and
`simulation_generation.observation_instance`.

The scalar estimator always receives `sigma2=None`: variance is estimated
from the observed field even at zero injected noise. It derives μ and τ
without coefficient truth, clean data or the injected noise standard deviation.
The full library, physical coordinates, original weak supports and strides,
polynomial degrees for original datasets, and bump tests for HKS/VBG are used.
Solver equilibration is disabled and failures are recorded without silently
rescaling the problem.

This is an empirical plug-in sanity study. The known-variance conditional
calculations do **not** supply a 1−δ coverage guarantee when variance and other
bounds are estimated from the same observed data. The requested definition of
g₂ uses the operator 2-norm.

Coefficient errors use the returned conic coefficients verbatim. Support
scoring uses the separately returned support mask at the declared tolerance;
there is no thresholded coefficient error or least-squares refit. Truth
residuals and feasibility at t=‖w*‖₂ are recorded only after fitting.

If τ ≥ ‖y‖∞, the zero coefficient vector with t=0 is feasible and attains
objective zero. Since ‖w‖₁+λt is nonnegative for λ>0, this is a global optimum.
An all-zero fit in that regime is therefore explained by the tolerances.

Outputs are written under `benchmark_results/eiv/scalar_sanity/`:

- `results.md`: per-trial status, estimated variance, μ, τ, response norm,
  τ/‖y‖∞, analytic zero-optimality flag, support recovery and E2.
- `results_trials/*.jsonl`: coefficients, support, complete tolerance and
  solver diagnostics, post-fit truth diagnostics, hashes and failure records.
- `results.summary.json` and `results.status.json`: counts and aggregate errors
  over successful solves.
- `results.manifest.json` and `results_source/`: complete protocol, software
  versions, data/source hashes and frozen numerical sources.

The runner does not edit previous benchmark reports or baseline artifacts.
An existing run requires `--resume` and an identical protocol and source hash.
Use `--report-only` to regenerate its report from saved trial records. Use a
fresh output directory when changing settings or code, for example:

```sh
python benchmark_results/eiv/run_sanity.py --benchmarks IB KS --noise-ratios 0 0.5 \
  --lambda 1 --delta 0.05 --trials 1 \
  --output-dir benchmark_results/eiv/custom_sanity
```

The defaults deliberately do not run the 100-trial benchmark suite. Coupled
NLS/RD and the multicomponent NS library are outside this scalar runner.

## Oracle μ/τ grid sanity check (λ=1)

```sh
python benchmark_results/eiv/run_oracle_grid.py --jobs 3
```

This separate study uses the same five scalar PDEs, five noise levels, one
trial per setting, observed-data variance estimates and physical weak systems.
It fixes λ=1 and selects μ and τ by **minimum coefficient E2 against truth**.
This is an oracle diagnostic, not a data-driven tuning rule or a fair tuned
benchmark. Neither the root `results.md` nor the formula study is overwritten.

The coarse grid has 144 pairs: μ/‖y‖∞ is zero or 10^q for integer q=−8,…,2;
τ/‖y‖∞ is zero, 10^q for q=−6,…,−1, or 0.2, 0.4, 0.6, 0.8, 0.95. Up to 49
nearby points refine the best eligible coarse candidate. The ratios construct
input tolerances only; solver arrays and the physical coefficient penalty are
not rescaled. The range is finite and does not certify a global hyperparameter
optimum. There is no refinement when every coarse solve fails validation.

Clarabel uses its existing estimator tolerance 1e−8, no equilibration, and at
most 200 iterations. `--solver-tolerance` can change this numerical tolerance.
An additional residual check requires violation ≤ 1e−6‖y‖∞, without a unit-size
floor. This prevents accepting nominal convergence when the entire HKS/VBG
response is smaller than a default absolute tolerance. Cone feasibility,
returned objective versus dual, and a known feasible zero-coefficient objective
bound are also checked. Failures retain their raw status and rejection reasons
and never enter the oracle ranking. A positive μ always permits some feasible
solution by increasing t, so `PrimalInfeasible` at positive μ is a numerical
failure. At μ=0, infeasibility may be genuine.

Default-tolerance outputs are in
[`oracle_grid_default_tolerance/results.md`](oracle_grid_default_tolerance/results.md), with
case records, every candidate (including failures), protocol/data/source hashes,
frozen sources and checks that earlier artifacts were preserved. Use `--resume`
with identical sources/protocol to continue, or `--report-only` to rebuild the
separate report. Incomplete cases are rerun in full on resume. A custom output
folder must be a new folder under `benchmark_results/eiv/`, outside
`scalar_sanity/`.

The initial stricter 1e−12 run is retained in
[`oracle_grid/results.md`](oracle_grid/results.md). That tolerance rejects
accurate clean-KS candidates with `AlmostSolved` status. The default-tolerance
study therefore repeats the full grid at 1e−8, with the independent physical
residual and objective checks unchanged. The strict run is a numerical
sensitivity diagnostic, not the preferred result table.

The saved default-tolerance run contains 4,200 solves, including two pairs
that became identical physical inputs after floating-point rounding. Their
coefficients, status and E2 are identical, so selection is unchanged. The
current runner deduplicates physical inputs too. Use the saved
`results_source/run_oracle_grid.py` for an exact reproduction or `--resume`
of that archived protocol; use a fresh output directory for the current code.
`audit.json` records independently rebuilt systems, recomputed eligible
coefficients/constraints, paired-data checks and the harmless duplicates.

Physical-unit L1/L2 penalties depend on dictionary units: a very large column
can explain a response using a very small coefficient. A converged fit with
poor coefficient recovery can therefore reflect this objective's preferences,
in addition to any numerical conditioning issues. The study deliberately keeps
the requested `rescale=False` setting so these effects remain visible.
