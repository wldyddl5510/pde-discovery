# pde-discovery

Python implementation of **WSINDy for PDEs**, fixed to
[Messenger & Bortz, arXiv:2007.02848v3, 21 December 2020](https://arxiv.org/html/2007.02848v3).
The scope is the convolutional weak formulation, polynomial test functions,
scale invariance, MSTLS, and the polynomial-dictionary identification experiments in that version.
The default comparison baseline follows the authors' physical-unit sparsification
and state scaling. The printed-algorithm profile is retained as an ablation.
KS, NLS, and RD are primary comparisons; IB, KdV and NS are supplementary.
Two additional PDE benchmarks follow the equations and method settings in
[arXiv:2211.16000v1](https://arxiv.org/html/2211.16000v1): Hyper-KS (`HKS`) and
nonlinear viscous Burgers growth (`VBG`). These use declared resimulations;
the authors' original trajectories have not been obtained. No ODE experiments
are included.

The active experiment scope is **polynomial state dictionaries only**:
`IB, KdV, KS, NLS, RD, NS, HKS, VBG`. Sine–Gordon (`SG`) is excluded
because its true equation requires `sin(u)` and its library includes sin/cos
terms. NLS is polynomial in the real/imaginary components; NS is polynomial
in observed vorticity/velocity components. Trigonometric initial conditions
and the C-infinity weak test function in HKS/VBG do not change their state
dictionaries, which contain only polynomial terms. Historical result/source
snapshots remain available; the active CLI, defaults and combined tables
exclude SG. Removing SG preserves every other experiment's original noise
seed index.

The existing file layout is retained:

- `methods.py`: Algorithms 4.1–4.2, weak FFT integrals, scaling, MSTLS, LASSO, and Appendix A support selection.
- `simulation_generation.py`: verified original datasets, Table 2 ground truth,
  published settings with the archived 181-term RD library, and independent
  observation-noise instances.
- `experiments.py`: identification trials, moving-average preprocessing, paired comparisons, and reports.
- `eiv_study.py`: scalar Section 6 Gaussian polynomial correction, tolerance calibration, and conic regression.
- `tests/`: quadrature, scaling identities, sparse selection, coupled equations,
  data conventions, and experiment checks.
- `results.md`: index of generated benchmark reports.
- `benchmark_results/`: saved protocols, source snapshots, raw trials, summaries,
  plots, execution logs, and resumable run status.
- `data/`: ignored local cache of the authors' original `.mat` files.
- `third_party/PyWSINDy-for-PDEs/`: retained reference submodule; the implementation
  has no runtime dependency on it.

## Running

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Verify/download the six active original polynomial datasets (~400 MB total).
python experiments.py --download-only

# Short verification on the original grids.
python experiments.py --benchmarks IB KdV KS --noise-ratios 0 --trials 1 --offline --output benchmark_results/check/results.md

# All six original polynomial clean-data experiments.
python experiments.py --noise-ratios 0 --trials 1 --offline --output benchmark_results/clean/results.md

# Comparison schedule: 6 PDEs, 5 noise levels, 100 trials per level (3,000 total).
python experiments.py --offline --workers 3 --output benchmark_results/polynomial/authors/results.md

# Resume the same saved protocol after interruption.
python experiments.py --offline --workers 3 --output benchmark_results/polynomial/authors/results.md --resume

# Refresh summaries/plots from the saved trials without fitting.
python experiments.py --offline --output benchmark_results/polynomial/authors/results.md --report-only

# Filtered WSINDy on the same 3,000 active observation instances; update the shared tables.
python experiments.py --method filtered-wsindy --filter-prior 0.01 --compare-to benchmark_results/polynomial/authors/results.md --comparison-output results.md --offline --workers 3 --output benchmark_results/polynomial/filtered/results.md

# Resume that filtered run with its original protocol.
python experiments.py --method filtered-wsindy --filter-prior 0.01 --compare-to benchmark_results/polynomial/authors/results.md --comparison-output results.md --offline --workers 3 --output benchmark_results/polynomial/filtered/results.md --resume
```

The existing `pde_discovery` conda environment can be used by replacing `python`
with `conda run -n pde_discovery python`. The numerical method requires NumPy and
SciPy; reports also use Matplotlib.
The default is a subset of the paper schedule: 100 trials at noise ratios
`0, 0.2, 0.5, 0.75, 1.0`.
These shared levels are fixed for every method. This samples notable
parts of the curves rather than estimating each transition precisely.
The paper's 41-level, 200-trial schedule remains available by explicitly setting
`--trials 200 --noise-ratios` followed by `k/40` for `k=0,...,40`.
Use `--output /path/to/report.md` for a separate run.
On macOS Accelerate, prefix the command with `VECLIB_MAXIMUM_THREADS=1`
to avoid thread overhead in the many small least-squares refits; for OpenBLAS
the corresponding setting is `OPENBLAS_NUM_THREADS=1`.

Raw trials are streamed to one JSONL file per PDE in `results_trials/`, including
seeds, physical coefficients, threshold losses, ranks, scales, and per-equation
metrics. A stopped run retains completed records; `--resume` repairs an incomplete
final append and skips completed trial keys. Source/environment/protocol mismatch
is rejected. Reports and CSV/JSON summaries refresh every 30 seconds. Plots refresh
periodically and at completion. The report states the actual sample count at every
noise level; partial results are not labelled as complete.

MSTLS execution first collects up to 20 trials at noise ratios 0, 0.2, 0.5, and 1,
then fills the requested levels to the requested trial count. LASSO execution
uses concurrent chunks of 10 trials. Every trial keeps the same seed regardless of
execution order or requested subsets. All repeats, including noise zero, are
actually executed. Timings cover the method, excluding loading and noise generation,
and are measured under the recorded process/thread settings.

`results.manifest.json` fixes dataset checksums, library terms and order, truth,
scaling/selection rules, the threshold/penalty path, RNG, package versions, and source hashes.
`results_source/` preserves the numerical implementation used for the run. To
resume after later code changes, execute that snapshot with the original arguments
and an explicit `--data-dir` pointing to this repository's cache.

For a new method, regenerate exactly the same input from each baseline record:

```python
from simulation_generation import load_clean_benchmark, observation_instance
clean = load_clean_benchmark(record["name"], download=False)
data = observation_instance(clean, record["noise_ratio"], record["seed"])
# Pass data.u_observed, data.spatial_grid, data.time to the new method.
```

Use the saved term order and ground truth for physical coefficient comparison.
Summaries include TPR, exact-support recovery with Wilson 95% intervals, E_inf,
E2, runtime, threshold, sample SD, median, and 90th percentile. Failed model
identifications are included in error summaries.

The comparison run reuses completed trials with indices 0 through 99
from the completed 50-trial run and the interrupted 200-trial run. Selection uses only noise level and trial
index, never the recovery result. Each reused record retains its original
protocol ID in `reused_from_protocol_id`; `results.reuse.json` records the source
manifests and unchanged numerical source hashes. The previous runs are preserved
in `benchmark_results/authors_50_completed/` and `benchmark_results/authors_200_stopped/`. At zero noise all observations are
identical, so those repeated trials are clean-data checks, not independent
statistical samples.

## Filtered WSINDy comparison

`results.md` contains a shared table for each PDE, with adjacent WSINDy and
Filtered WSINDy rows at each noise level. The filtered run regenerates the
same original observations from the saved root seed and trial indices. Pairing
checks input seeds and injected component noise levels; protocol validation
also checks the datasets, libraries, truth, numerical sources, scaling and
regression settings. During execution, both rows use only completed matching
trials. The full 100-trial original baseline remains in its individual report.
Once complete, each method has 100 trials at each of the five noise levels.

The preprocessing follows the moving-average construction in
[Messenger & Bortz, arXiv:2211.16000, Sections 4.2/5.4 and Appendix G](https://arxiv.org/pdf/2211.16000).
For each observed component, estimate its noise standard deviation using the
RMS of unit-L2 sixth differences along time: stencil
`[1,-6,15,-20,15,-6,1]/sqrt(924)`. Use the maximum component estimate and a common
window for every component. With `D` space-time axes, library polynomial degree
`p_max`, test half-width `m_d` and prior `tau_star=0.01`, each side has

```text
w_d = max(1, floor(min(2*(binom(p_max,2)*sigma_est^2/tau_star)^(1/D), (2*m_d+1)/2)))
```

This generalizes the paper's degree-6 factor `1500` and caps anisotropic test
supports per axis. Apply `scipy.ndimage.uniform_filter` over space and time with
`mode="reflect"`, `origin=0`. Evaluate the nonlinear library **after** averaging,
and use the filtered states in both the weak LHS and RHS. The original grids,
libraries, polynomial tests, scaling, 50 thresholds and author MSTLS rules are
retained. Thus this evaluates the preprocessing on the six active polynomial PDEs;
the consistency paper uses different experiments and regression settings.

Window selection uses only observed arrays, including at zero noise. Signal
structure can contaminate the noise estimate (especially shocks), and averaging
introduces smoothing bias, so improvement is not assumed. These are fixed rules,
with no tuning on ground-truth support or coefficient errors. Filtered raw records
store the estimated noise, actual window sizes, number of averaged points and
preprocessing runtime. Total filtered runtime includes this preprocessing.

The original and filtered runs keep separate immutable protocol manifests and
source snapshots. The shared report is refreshed every 30 seconds, with paired
CSV/JSON summaries in `benchmark_results/filtered/results.paired.summary.*`.

## LASSO comparison

Use `--regression lasso` with either `--method wsindy` or
`--method filtered-wsindy`. The default regression remains `mstls`.
The comparison adds 8 PDEs x 5 noise ratios x 100 trials x 2 preprocessing
methods = 8,000 fits on the same observations as the saved baselines.

For each weak system with K rows, normalize each nonzero column by its RMS
and each equation's response by its RMS. No centering or extra intercept is
used; the existing constant library term is penalized. Solve

```text
min_theta ||Z theta - y_e||_2^2/(2*K) + lambda_e*||theta||_1
lambda_max,e = max(abs(Z.T@y_e))/K
lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99), k=0,...,99
```

Each trial and equation has its own `lambda_max`, the smallest penalty for
which the zero model is optimal. Search 100 logarithmically spaced candidates
from `1e-4*lambda_max` to `lambda_max`; coupled equations share one selected
candidate index and have separate lambda values.
Select using the existing WSINDy criterion: matrix 2-norm prediction difference
from full OLS / full OLS prediction norm + fraction of nonzero coefficients.
Smallest lambda breaks exact ties. Selection uses weak G/b only, without true
support or coefficients. This is a column-normalized L1 penalty, equivalent
to a weighted L1 penalty in physical units. Report the penalized coefficients
after restoring physical units; do not refit their support with OLS. Support
uses the solver's exact zeros, without a coefficient cutoff. Empty models are
admissible.

The [scikit-learn LARS Gram path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path_gram.html)
is checked against the LASSO KKT conditions at every candidate. Correlated
libraries first use primal active-set polishing (at most 200 iterations per
candidate). Remaining violations trigger coordinate descent (`tol=1e-9`,
`max_iter=200000`) and final polishing (at most 2000 iterations). Fits are
rejected if maximum coordinate KKT
violation divided by alpha_max exceeds `1e-7`. Saved trials include alpha_max,
selected alpha, RMS normalization, normalized coefficients, KKT errors and
solver diagnostics. These LASSO runs are additional regression comparisons;
they are not the papers' published sparse-regression algorithms.

```sh
python experiments.py --benchmarks RD NS NLS KS KdV IB --regression lasso --offline --workers 2 --output benchmark_results/lasso/original/raw/results.md
python experiments.py --benchmarks RD NS NLS KS KdV IB --regression lasso --method filtered-wsindy --offline --workers 2 --output benchmark_results/lasso/original/filtered/results.md
python experiments.py --benchmarks HKS VBG --regression lasso --offline --workers 2 --output benchmark_results/lasso/consistency/raw/results.md
python experiments.py --benchmarks HKS VBG --regression lasso --method filtered-wsindy --offline --workers 2 --output benchmark_results/lasso/consistency/filtered/results.md

# Rebuild all four rows per PDE/noise condition, without fitting.
python experiments.py --report-only --comparison-reports benchmark_results/authors/results.md benchmark_results/filtered/results.md benchmark_results/consistency/raw/results.md benchmark_results/consistency/filtered/results.md benchmark_results/lasso/original/raw/results.md benchmark_results/lasso/original/filtered/results.md benchmark_results/lasso/consistency/raw/results.md benchmark_results/lasso/consistency/filtered/results.md --comparison-output results.md
python benchmark_results/lasso/penalty_levels.py

# Audit the complete 8,000-fit run and refresh explicitly labelled LASSO plots.
python benchmark_results/lasso/verify_results.py
python benchmark_results/lasso/refresh_plots.py
```

Trial chunks are processed concurrently so RD does not serialize the whole
suite; a single parent appends completed chunks to JSONL files. `--resume`
keeps completed trial keys and reruns unfinished chunks. Each run retains its
own immutable manifest and source snapshot. After source edits, resume using
that snapshot and an explicit `--data-dir`, as with MSTLS runs. The aggregate
report checks dataset/library/protocol agreement and per-trial seed/noise/weak
matrix shapes. Each condition uses the intersection of completed trials across
its methods; full individual reports remain available. At completion every
row has 100 trials. `results.summary.csv/json` save all four methods' metrics.

LASSO result tables show the median selected `lambda` for each LHS equation.
The beginning of each report documents the candidate range and selection
criterion in terms of lambda; the relative penalty ratio is omitted from tables.
The combined report and individual LASSO reports group results by experiment.
Each experiment contains its true equation and settings, then one noise-level
subheading and comparison table for each of the five noise levels.
Lambda refers to the RMS-normalized objective above. The CSV/JSON summaries
also save minimum and maximum ratios and per-equation lambda values.
Regenerate these columns from saved trials with
`python benchmark_results/lasso/penalty_levels.py`; the suite completion helpers
also run this report step. Individual trial values remain in `results_trials/`.

### Section 5 debiasing: all polynomial PDEs

The comparison also includes the polynomial linearization implemented in
`debiasing_study.py`: `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw
observations on the LHS and in linear columns. For a scalar power this is
`(1-p)*pilot^p + p*pilot^(p-1)*U`. Coupled monomials use the full gradient;
NLS/RD retain both equations and NS uses all three observed components.
These full-library fits use MSTLS and LASSO separately;
restricted-support OLS from the sanity study is excluded from the benchmark.
The study's regression helper restores physical coefficient units for both
rescaled sparse fits and restricted OLS.

The split is always `time2`: even time indices use an average of odd time
indices, and odd indices use even indices. The pilot is a box moving average
with periodic spatial and reflected time boundaries; NS uses reflected
spatial boundaries because its cropped wake domain is not periodic.
Each evaluation parity's noise estimate and window selection use only the
opposite time parity. Sixth differences are evaluated on that training
subgrid, with time spacing twice the original. The largest estimate across
components selects a shared window using the study's volume rule at prior
`0.01`, rounded up to odd and at least 3. No true
coefficients, clean trajectory errors, or support recovery scores select the
window. This protocol retains smoothing bias, including at zero noise.
The split and adaptive window's independence are verified by perturbing an
entire evaluation fold and checking that its pilot and chosen window stay
identical. The pooled regression is a benchmark implementation; the
single-pilot conditional Gaussian expression in Section 5.1 is not claimed
as a theorem for the pooled cross-fit estimator.

```sh
# Eight PDEs, 500 paired observations each, MSTLS and LASSO: 8,000 fits.
python benchmark_results/debiasing/run_suite.py --workers 4

# Regenerate completed PDE additions in the shared tables from saved trials.
python benchmark_results/lasso/penalty_levels.py

# Audit completed records, source snapshots, lambda summaries and old baselines.
python benchmark_results/debiasing/verify_suite.py
```

The schedule, noise amplitudes and observation seeds match the saved
baseline: ratios `0, 0.2, 0.5, 0.75, 1`, 100 trials per ratio, root seed 0.
Trial records, immutable manifests and numerical source snapshots are saved
under `benchmark_results/debiasing/time2_foldwise/<PDE>/{mstls,lasso}/`.
The original IB pilot with window selection on the full observations remains
archived under `benchmark_results/debiasing/IB/`; the shared table uses the
replacement IB run once complete. The runner resumes only the same numerical
protocol. The saved suite can be resumed from the repository root with
`PYTHONPATH="$PWD:$PYTHONPATH" python benchmark_results/debiasing/time2_foldwise/source/run_suite.py --workers 4`.
Timings include noise
estimation, pilot construction, weak construction and the entire regression
path. The same measured pilot/weak construction cost is attributed to each
regression; loading, noise generation and clean-data diagnostics are excluded.

### Section 6: scalar conic sanity study

`eiv_study.py` implements `min ||beta||_1 + lam*t` subject to
`||y-(X-noise_bias) beta||_inf <= tau+mu*t` and `||beta||_2 <= t`.
`conic_eiv(X, y, mu=..., tau=..., lam=..., noise_bias=...)` accepts explicit
tolerances and an optional correction matrix. Its input X is already corrected
when `noise_bias=None`. The scalar data entry point is:

```python
from eiv_study import fit_eiv
result, system, calibration = fit_eiv(
    observed_u, spatial_grid, time,
    library_terms=terms, half_widths=half_widths, strides=strides,
    test_degrees=test_degrees, lam=1., delta=.05, sigma2=None,
)
```

This uses physical coordinates and coefficients (`rescale=False`), without
RMS normalization, a pilot, splitting, penalty search, or support refitting.
Clarabel solves the conic program; solver equilibration also defaults off.
`sigma2=None` estimates variance using the existing sixth-difference estimator.
The corrected powers obey `H_0=1`, `H_1=U`,
`H_(j+1)=U*H_j-j*sigma2*H_(j-1)`. Known true variance gives Gaussian-unbiased
polynomials; estimated variance gives a plug-in correction.

Missing `mu` and `tau` use the specified delta-dependent formula, with
`g2=max_k ||F_k||_op` and `g_inf=max_(k,i) ||F_k[i,:]||_2`.
Optional `M`, `M1` supply true envelopes; defaults use max absolute raw U and
max Euclidean raw space-time gradient from second-order finite differences
(one-sided at grid boundaries). The declared uniform-cell geometry uses
`Delta_z=prod(spacing)`, `|D|=number_of_samples*Delta_z`, and
`h=norm(spacing,2)`. Test derivative extrema are calculated numerically from
analytic one-dimensional derivatives; the gradient bound combines component
suprema. These data-derived tolerances carry no proved confidence guarantee.

The result retains the conic coefficients and t. An absolute physical-unit
support tolerance (default `1e-7`) produces a separate diagnostic mask; it does
not alter coefficients. If `tau >= ||y||_inf`, beta=t=0 is an analytically
certified global optimum, recorded as `AnalyticZero`.

Run the small study after installing `requirements.txt`:

```sh
python benchmark_results/eiv/run_sanity.py --lambda 1 --delta 0.05
```

It runs IB, KdV, KS, HKS and VBG at noise ratios 0, 0.2, 0.5, 0.75 and 1,
with one trial each and variance estimated from observations. NLS/RD/NS are
excluded. [Study instructions](benchmark_results/eiv/README.md) describe the
saved trials and frozen sources. [Sanity results](benchmark_results/eiv/scalar_sanity/results.md)
are separate from the existing 100-trial comparison tables.

## Method

### Additional consistency-paper PDEs

Run this suite separately because its noise definition and regression differ
from the six original polynomial PDEs. The default remains these six PDEs.

```sh
# Generate/cache the two spectral solutions, with timestep refinement checks.
python experiments.py --benchmarks HKS VBG --download-only

# 2 PDEs x 5 noise ratios x 100 trials = 1,000 raw WSINDy fits.
python experiments.py --benchmarks HKS VBG --offline --workers 2 --output benchmark_results/consistency/raw/results.md

# 1,000 filtered fits on exactly the same noise instances.
python experiments.py --benchmarks HKS VBG --method filtered-wsindy --compare-to benchmark_results/consistency/raw/results.md --comparison-output benchmark_results/consistency/results.md --offline --workers 2 --output benchmark_results/consistency/filtered/results.md
```

The `consistency` profile is selected automatically for these names. It uses
physical coordinates without rescaling, analytic derivatives of the published
compact bump `exp(9/(r^2-1))`, absolute coefficient hard thresholds, iterative
STLS until support stabilizes, and 100 thresholds `logspace(-4,0,100)`. Its loss
follows Eqs. 2.6–2.9. Original author/printed profiles retain their previous
numerical rules. Original saved runs can be resumed using their source snapshots
and an explicit `--data-dir` after these source changes.

| PDE | Declared initial condition | Periodic spatial domain | Time | Fine grid | Observation grid | Library |
| --- | --- | --- | --- | --- | --- | --- |
| HKS | `cos(x/16)*(1+sin(x/16))` | `[0,32*pi)` | `[0,82]` | `1024 x 1025` | `256 x 257` | degree 0–8, dx 0–8; 73 terms |
| VBG | `2*sin(pi*x)` | `[-1,1)` | `[0,1.5]` | `2048 x 1801` | `512 x 451` | degree 0–6, dx 0–6; 43 terms |

The equations, physical domains, fine-grid sizes, libraries, bump function and
regression are from the paper. The ICs, periodic BCs, integrator timesteps,
observation resolution and query strides are declared choices here, fixed
before the noise trials. The paper does not specify enough information to
recover these original trajectories from its text. Consequently, these tables
are a fixed-resolution benchmark using the paper's models and method, not an
exact replication of its figures. The specified noise ratio 1 also exceeds the
VBG paper's reported range, which ends near 0.5.

The generator uses Fourier spectral ETDRK4 and a twice-padded grid for quadratic
and cubic products. It retains the finer of two timestep runs and rejects a
relative trajectory difference above `2e-5`. Generated arrays have content
digests; manifests record the archive checksum, generation configuration and
refinement errors. Cached data/config mismatches are rejected. Grid points in
the periodic spatial interval exclude the duplicate right endpoint.

Noise is independent Gaussian with `sigma=ratio*std(clean.ravel(), ddof=1)`.
This uses centered sample standard deviation rather than the original suite's
component RMS. Seeds remain fixed across methods; appending the new names
preserves all previously saved benchmark seed indices; the former SG slot is reserved. Zero-noise repetitions
are identical clean checks.

Filtering uses the paper's common-side volume cap, with
`m=prod(2*half_width+1)` and
`w=max(1,floor(min(2*(binom(p_max,2)*sigma_est^2/tau_star)^(1/D),m^(1/D)/2)))`.
The prior is `0.01`; the noise estimator uses sixth differences in time.
VBG uses the paper's factor 1500; HKS generalizes it to 2800 for degree 8.
Filtered HKS is an additional comparison (the paper reports filtered results
only for its cubic ODE and VBG). Both LHS/RHS use averaged states, with a
reflecting boundary convention for the moving-average preprocessing; the
simulation boundary condition is periodic. The report records both conventions.

Use `--include-report PATH` when writing a comparison to include a previously
completed suite in the same report; relative links are rebased. Trials and
protocols remain separate, and summaries include subset-support probability
alongside the existing exact recovery, TPR, coefficient error and timing fields.

The active eight-PDE comparison is in `results.md`. The two new PDEs have
1,000 completed paired instances (2,000 fits); their independent report is
`benchmark_results/consistency/results.md`. Refresh the combined report without
refitting, using the saved historical comparison:

```sh
python experiments.py --report-only --comparison-reports benchmark_results/authors/results.md benchmark_results/filtered/results.md benchmark_results/consistency/raw/results.md benchmark_results/consistency/filtered/results.md --comparison-output results.md
```

`benchmark_results/consistency/data_validation.json` records the independent
spatial-refinement check. `verification.json` records paired-count, checksum,
source-snapshot and coefficient-metric checks. The old baseline files remain
unchanged; the two suites keep their own source snapshots.

### Original-paper numerical method

Observations are tuples of component arrays with shape `(space..., time)`. The
right-hand library contains `D^alpha f(U)` terms and no time derivatives. NLS and
RD have two equations; NS uses observed vorticity and two velocity components to
identify only the vorticity equation. Its archive cells `(u,v,omega)` are mapped
to the library order `(omega,u,v)` without changing their values.
The constant appears once, and its derivatives are excluded.

1. Sample analytic derivatives of the separable test function
   `phi(r)=(1-r^2)^p` on `[-1,1]`. Algorithm 4.1 selects the smallest integer
   `p > max_derivative` satisfying the penultimate-point tolerance `tau=1e-10`.
   The experiment runner fixes the published Table 4 degrees.
2. Build valid convolutions with FFTs, physical quadrature spacing, and the
   Table 3 subsampling strides. The row ordering is the tensor grid flattened
   in NumPy C order. Test centers are `grid[m:N-m:s]`.
3. Scale coordinates according to Eq. 4.7. For odd orders use the factorial
   convention in the authors' `get_scales.m`; order one has factor one.
   For each component, the author profile uses state scale
   `gamma_u=(||U||_2/||U^beta_max||_2)^(1/(beta_max-1))`; the printed profile uses
   exponent `1/beta_max`.
4. Apply MSTLS bounds to **physical** coefficients and physical bounds for the
   author profile, or scaled coefficients/bounds for the printed profile. Search
   the paper's 50 thresholds, `logspace(-4,0,50)`, using the projection-error
   plus support-size loss in Eq. 4.4. Return the smallest minimizer.
   The author profile returns the previous fit if an iteration would delete all
   terms. The printed profile allows empty support. Least squares uses an SVD, including on
   restricted and rank-deficient libraries; no ridge or column normalization
   is added. For coupled systems, each equation is refitted separately and a
   common threshold is selected using matrix 2-norms and the fraction of
   nonzero entries in the full coefficient matrix.
5. Restore physical coefficient units after sparse selection.

Appendix A support selection is exposed separately by `select_test_supports`.
It uses continuous weighted two-line fits of cumulative negative-frequency
spectra and solves the Appendix A decay equation. Experiments use Table 3's
fixed supports, so reproducing MATLAB's changepoint routine is not a dependency.

### Debiased WSINDy (draft Section 5)

`methods.build_wsindy_system(..., pilot=...)` builds the first-order debiased
weak system: every monomial of total degree `p >= 2` is replaced by its
expansion at a pilot `v` evaluated at the observations,
`(1-p) v^p + p v^(p-1) U` (coupled monomials expand in every component), so
the LHS and linear columns equal the raw ones and the draft's `u_*` is
replaced by `U`. `methods.moving_average_pilot` provides the pilot: a box
moving average with odd window lengths, `wrap`/`reflect` boundary extension
per axis, and optional lattice splits (`("axis", index, folds)` residue
classes or `checkerboard`) that average each point over the other folds only,
i.e. cross-fitting on one grid. Trigonometric terms are rejected.

`debiasing_study.py` compares raw, filtered (pilot plug-in) and debiased
systems built from identical observations and pilots:

```sh
# VBG, sigma/sigma_c in {0.5,1,2,5,10}, 50 trials, window sweep, no split and time-parity split.
python debiasing_study.py --workers 8 --offline
```

Restricted OLS on the true support plus the noise-generated terms measures
coefficient bias directly (raw OLS predicts `+3 sigma^2` on `u` and
`-2 sigma^2` on the constant for VBG); full-library MSTLS and LASSO measure
support recovery. `results_debiasing.md` and its `.jsonl` record the current
run. Omit `lasso` from `--regressions` to run without scikit-learn.

## Source discrepancies and reproducibility boundaries

These differences are recorded explicitly so they cannot be mistaken for exact
agreement with every published number:

- **Version:** the linked arXiv v3 contains seven PDEs. The later
  [journal article](https://doi.org/10.1016/j.jcp.2021.110525) additionally includes
  anisotropic porous medium (PM). PM is outside this repository's current target.
  Earlier descriptions calling PM unrelated to the journal paper were incorrect.
- **State scale:** the printed exponent is `1/beta_max`. The current MATLAB
  `toggle_scale=2` uses `1/(beta_max-1)` instead. `--profile authors` is the
  default baseline; `--profile printed` selects the printed equation and selection
  rules. Exact Table 4 scales/condition numbers depend on the noisy realization.
- **Coefficient restoration:** the printed formula for `M` has the reciprocal
  signs for the stated direction `w=M*w_tilde`. We use the coordinate identity
  itself, also used by the authors' implementation. For a polynomial term in
  equation k, the physical coefficient multiplier is
  `prod(gamma_u_i^(beta_i-delta_ik)) * prod(gamma_d^(alpha_lhs_d-alpha_term_d))`.
  Trigonometric arguments are evaluated on original observations, giving no
  polynomial state-scaling factor for those trial functions. Tests verify this
  identity against independently built unscaled weak systems, including general trigonometric terms and second time derivatives.
- **MSTLS reference:** the current author MATLAB script has both original-unit
  and scaled-unit sparsification modes, and its default chooses original units.
  Algorithm 4.2 explicitly calls MSTLS on the scaled system; it is retained as
  the printed ablation. The baseline uses the author's physical sparsification
  and return-before-empty rule, with SciPy SVD refits. It is a Python port of
  these rules, not a direct MATLAB execution. The current MATLAB script also uses 100 thresholds and
  subsamples the data by default, whereas the paper uses 50 thresholds and the
  grids in Table 3.
- **RD tables:** total-degree monomials through 4 and pure spatial derivatives
  through 5 in Table 3 give **155 columns**, and its grid/support/strides give
  **4860 rows**. Table 4 reports `11638 x 181`. The archived `rxn_diff.mat`
  stores `polys=0:5`, `max_dx=4`, and `use_cross_dx=0`, which instead give
  **181 columns**: `21 * 9 - 8`. This supports the interpretation that Table 3
  swapped the polynomial and derivative bounds, although it does not resolve
  the reported row count. The comparison protocol fixes the archived degree-5,
  derivative-order-4 library (181 columns) for every method, and retains the
  published grid/support/strides (4860 rows).
- **Queries:** Eq. 4.9 prints a floor, but Table 4 and the author's query indexing
  use the inclusive sequence `m, m+s, ... < N-m`, whose count is a ceiling.
  This sequence gives, for example, the reported 784 IB rows.
- **Noise:** Eq. 5.1 is implemented independently for each component. The authors'
  [Zenodo archive](https://zenodo.org/records/20787783) supplies clean data, not
  the original 200 noise draws. Results can reproduce the statistical experiment
  protocol, not the original MATLAB random stream or exact runtime.
- **Metrics:** report TPR, E_inf, and E2 from Eqs. 5.3–5.5, including per-equation
  records. These are the identification metrics in the target arXiv v3.
  The later journal article adds solution-prediction error and prediction horizon
  in its Eqs. 5.6–5.7; they are not included here. In arXiv v3, Eq. 5.6 instead
  describes a reduced reaction-diffusion model.

The authors' MATLAB code was inspected at
[MathBioCU/WSINDy_PDE, commit d9296be4c17c5e0b4df14472f4cd8276a8ae4eed](https://github.com/MathBioCU/WSINDy_PDE/tree/d9296be4c17c5e0b4df14472f4cd8276a8ae4eed).
Dataset checksums come from the authors' Zenodo archive. The Python numerical
implementation is local; it does not execute the reference submodule.
