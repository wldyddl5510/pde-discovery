# pde-discovery

Python implementation of **WSINDy for PDEs**, fixed to
[Messenger & Bortz, arXiv:2007.02848v3, 21 December 2020](https://arxiv.org/html/2007.02848v3).
The scope is the convolutional weak formulation, polynomial test functions,
scale invariance, MSTLS, and the seven identification experiments in that version.
The default comparison baseline follows the authors' physical-unit sparsification
and state scaling. The printed-algorithm profile is retained as an ablation.
KS, NLS, and RD are primary comparisons; IB, KdV, NS, and SG are supplementary.
Two additional PDE benchmarks follow the equations and method settings in
[arXiv:2211.16000v1](https://arxiv.org/html/2211.16000v1): Hyper-KS (`HKS`) and
nonlinear viscous Burgers growth (`VBG`). These use declared resimulations;
the authors' original trajectories have not been obtained. No ODE experiments
are included.

The existing file layout is retained:

- `methods.py`: Algorithms 4.1–4.2, weak FFT integrals, scaling, MSTLS, LASSO, and Appendix A support selection.
- `simulation_generation.py`: verified original datasets, Table 2 ground truth,
  published settings with the archived 181-term RD library, and independent
  observation-noise instances.
- `experiments.py`: identification trials, moving-average preprocessing, paired comparisons, and reports.
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

# Verify/download all seven original datasets (~485 MB total).
python experiments.py --download-only

# Short verification on the original grids.
python experiments.py --benchmarks IB KdV KS --noise-ratios 0 --trials 1 --offline --output benchmark_results/check/results.md

# All seven clean-data experiments.
python experiments.py --noise-ratios 0 --trials 1 --offline --output benchmark_results/clean/results.md

# Comparison schedule: 7 PDEs, 5 noise levels, 100 trials per level (3,500 total).
python experiments.py --offline --workers 3 --output benchmark_results/authors/results.md

# Resume the same saved protocol after interruption.
python experiments.py --offline --workers 3 --output benchmark_results/authors/results.md --resume

# Refresh summaries/plots from the saved trials without fitting.
python experiments.py --offline --output benchmark_results/authors/results.md --report-only

# Filtered WSINDy on the same 3,500 observation instances; update the shared tables.
python experiments.py --method filtered-wsindy --filter-prior 0.01 --compare-to benchmark_results/authors/results.md --comparison-output results.md --offline --workers 3 --output benchmark_results/filtered/results.md

# Resume that filtered run with its original protocol.
python experiments.py --method filtered-wsindy --filter-prior 0.01 --compare-to benchmark_results/authors/results.md --comparison-output results.md --offline --workers 3 --output benchmark_results/filtered/results.md --resume
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

The execution first collects up to 20 trials at noise ratios 0, 0.2, 0.5, and 1,
then fills the requested levels to the requested trial count. Every trial keeps the same seed regardless of
execution order or requested subsets. All repeats, including noise zero, are
actually executed. Timings cover the method, excluding loading and noise generation,
and are measured under the recorded process/thread settings.

`results.manifest.json` fixes dataset checksums, library terms and order, truth,
scaling/selection rules, 50 thresholds, RNG, package versions, and source hashes.
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
retained. Thus this evaluates the preprocessing on the existing seven PDEs;
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
The comparison adds 9 PDEs x 5 noise ratios x 100 trials x 2 preprocessing
methods = 9,000 fits on the same observations as the saved baselines.

For each weak system with K rows, normalize each nonzero column by its RMS
and each equation's response by its RMS. No centering or extra intercept is
used; the existing constant library term is penalized. Solve

```text
min_theta ||Z theta - y_e||_2^2/(2*K) + alpha_e*||theta||_1
alpha_e = ratio * max(abs(Z.T@y_e))/K
ratio in logspace(-4, 0, 100)
```

Each coupled equation has its own `alpha_max` and shares one selected ratio.
Select using the existing WSINDy criterion: matrix 2-norm prediction difference
from full OLS / full OLS prediction norm + fraction of nonzero coefficients.
Smallest ratio breaks exact ties. Selection uses weak G/b only, without true
support or coefficients. This is a column-normalized L1 penalty, equivalent
to a weighted L1 penalty in physical units. Report the penalized coefficients
after restoring physical units; do not refit their support with OLS. Support
uses the solver's exact zeros, without a coefficient cutoff. Empty models are
admissible.

The [scikit-learn LARS Gram path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path_gram.html)
is checked against the LASSO KKT conditions at every candidate. Correlated
libraries can require coordinate descent (`tol=1e-9`, `max_iter=200000`) and
primal active-set polishing. Fits are rejected if maximum coordinate KKT
violation divided by alpha_max exceeds `1e-7`. Saved trials include alpha_max,
selected alpha, RMS normalization, normalized coefficients, KKT errors and
solver diagnostics. These LASSO runs are additional regression comparisons;
they are not the papers' published sparse-regression algorithms.

```sh
python experiments.py --regression lasso --offline --workers 3 --output benchmark_results/lasso/original/raw/results.md
python experiments.py --regression lasso --method filtered-wsindy --offline --workers 3 --output benchmark_results/lasso/original/filtered/results.md
python experiments.py --benchmarks HKS VBG --regression lasso --offline --workers 2 --output benchmark_results/lasso/consistency/raw/results.md
python experiments.py --benchmarks HKS VBG --regression lasso --method filtered-wsindy --offline --workers 2 --output benchmark_results/lasso/consistency/filtered/results.md

# Rebuild all four rows per PDE/noise condition, without fitting.
python experiments.py --report-only --comparison-reports benchmark_results/authors/results.md benchmark_results/filtered/results.md benchmark_results/consistency/raw/results.md benchmark_results/consistency/filtered/results.md benchmark_results/lasso/original/raw/results.md benchmark_results/lasso/original/filtered/results.md benchmark_results/lasso/consistency/raw/results.md benchmark_results/lasso/consistency/filtered/results.md --comparison-output results.md
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

## Method

### Additional consistency-paper PDEs

Run this suite separately because its noise definition and regression differ
from the original seven. The default remains the original seven PDEs.

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
preserves the original seven benchmark seed indices. Zero-noise repetitions
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

The completed nine-PDE comparison is in `results.md`. The two new PDEs have
1,000 completed paired instances (2,000 fits); their independent report is
`benchmark_results/consistency/results.md`. Refresh the combined report without
refitting, using the saved seven-PDE comparison:

```sh
python experiments.py --benchmarks HKS VBG --method filtered-wsindy --compare-to benchmark_results/consistency/raw/results.md --comparison-output results.md --include-report benchmark_results/original_comparison/results.md --offline --workers 2 --output benchmark_results/consistency/filtered/results.md --report-only
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
to the library order `(omega,u,v)` without changing their values. SG has a second time derivative on its LHS.
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
  identity against independently built unscaled weak systems, including SG.
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
