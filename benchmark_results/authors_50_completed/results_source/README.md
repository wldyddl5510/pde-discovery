# pde-discovery

Python implementation of **WSINDy for PDEs**, fixed to
[Messenger & Bortz, arXiv:2007.02848v3, 21 December 2020](https://arxiv.org/html/2007.02848v3).
The scope is the convolutional weak formulation, polynomial test functions,
scale invariance, MSTLS, and the seven identification experiments in that version.
The default comparison baseline follows the authors' physical-unit sparsification
and state scaling. The printed-algorithm profile is retained as an ablation.
KS, NLS, and RD are primary comparisons; IB, KdV, NS, and SG are supplementary.

The existing file layout is retained:

- `methods.py`: Algorithms 4.1–4.2, weak FFT integrals, scaling, MSTLS, and Appendix A support selection.
- `simulation_generation.py`: verified original datasets, Table 2 ground truth,
  published settings with the archived 181-term RD library, and independent
  observation-noise instances.
- `experiments.py`: identification trials, coefficient metrics, and reports.
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

# Comparison schedule: 7 PDEs, 10 noise levels, 50 trials per level (3,500 total).
python experiments.py --offline --workers 3 --output benchmark_results/authors/results.md

# Resume the same saved protocol after interruption.
python experiments.py --offline --workers 3 --output benchmark_results/authors/results.md --resume

# Refresh summaries/plots from the saved trials without fitting.
python experiments.py --offline --output benchmark_results/authors/results.md --report-only
```

The existing `pde_discovery` conda environment can be used by replacing `python`
with `conda run -n pde_discovery python`. The numerical method requires NumPy and
SciPy; reports also use Matplotlib.
The default is a subset of the paper schedule: 50 trials at noise ratios
`0, 0.05, 0.1, 0.2, 0.225, 0.3, 0.4, 0.5, 0.75, 1.0`.
The low levels resolve sensitivity to modest noise; 0.2/0.225/0.3 cover the RD
transition region; 0.4/0.5 cover the NLS transition region; 0.75/1.0 test high
noise. These shared levels are fixed for every method. This samples notable
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

The reduced comparison run reuses completed trials with indices 0 through 49
from the interrupted 200-trial run. Selection uses only noise level and trial
index, never the recovery result. Each reused record retains its original
protocol ID in `reused_from_protocol_id`; `results.reuse.json` records the source
manifest and unchanged numerical source hashes. The original run is preserved
in `benchmark_results/authors_200_stopped/`. At zero noise all observations are
identical, so those repeated trials are clean-data checks, not independent
statistical samples.

## Method

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
