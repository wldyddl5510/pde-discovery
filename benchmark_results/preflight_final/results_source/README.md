# pde-discovery

Python implementation of **WSINDy for PDEs**, fixed to
[Messenger & Bortz, arXiv:2007.02848v3, 21 December 2020](https://arxiv.org/html/2007.02848v3).
The scope is the convolutional weak formulation, polynomial test functions,
scale invariance, MSTLS, and the seven identification experiments in that version.

The existing file layout is retained:

- `methods.py`: Algorithms 4.1–4.2, weak FFT integrals, scaling, MSTLS, and Appendix A support selection.
- `simulation_generation.py`: verified original datasets, Table 2 ground truth,
  Table 3 libraries/settings, and independent observation-noise instances.
- `experiments.py`: identification trials, coefficient metrics, and reports.
- `tests/`: quadrature, scaling identities, sparse selection, coupled equations,
  data conventions, and experiment checks.
- `results.md`: results actually obtained with this implementation.
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
python experiments.py --benchmarks IB KdV KS --noise-ratios 0 --trials 1 --offline

# All seven clean-data experiments.
python experiments.py --noise-ratios 0 --trials 1 --offline

# Full identification schedule: 7 PDEs, 41 noise levels, 200 draws per level.
python experiments.py --offline
```

The existing `pde_discovery` conda environment can be used by replacing `python`
with `conda run -n pde_discovery python`. Only NumPy and SciPy are required.
The full schedule is a substantial computation. A shorter run must explicitly
set `--trials` and `--noise-ratios`; the report labels it as a subset.
Use `--output /path/to/report.md` for a separate run.
On macOS Accelerate, prefix the command with `VECLIB_MAXIMUM_THREADS=1`
to avoid thread overhead in the many small least-squares refits; for OpenBLAS
the corresponding setting is `OPENBLAS_NUM_THREADS=1`.

Raw trials are streamed to the adjacent `.jsonl` file, including seeds, selected
coefficients, threshold losses, ranks, and scales. A stopped run retains completed
trial records. The Markdown report is written when the requested run finishes.
Noise seeds are stable across benchmark ordering and requested noise subsets.

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
   For each component, use the **printed Eq. 4.8** state scale
   `gamma_u=(||U||_2/||U^beta_max||_2)^(1/beta_max)`.
4. Apply Eq. 4.5 MSTLS bounds to coefficients of the **scaled** system. Search
   the paper's 50 thresholds, `logspace(-4,0,50)`, using the projection-error
   plus support-size loss in Eq. 4.4. Return the smallest minimizer.
   Empty support is admissible. Least squares uses an SVD, including on
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
  `toggle_scale=2` uses `1/(beta_max-1)` instead. This repository follows the
  printed equation and does not silently substitute the MATLAB default.
  Accordingly, Table 4's numerical scales/condition numbers are not acceptance
  criteria for this implementation.
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
  Algorithm 4.2 explicitly calls MSTLS on the scaled system. That algorithm is
  the target here. The current MATLAB script also uses 100 thresholds and
  subsamples the data by default, whereas the paper uses 50 thresholds and the
  grids in Table 3.
- **RD tables:** total-degree monomials through 4 and pure spatial derivatives
  through 5 in Table 3 give **155 columns**, and its grid/support/strides give
  **4860 rows**. Table 4 reports `11638 x 181`. The archived `rxn_diff.mat`
  stores `polys=0:5`, `max_dx=4`, and `use_cross_dx=0`, which instead give
  **181 columns**: `21 * 9 - 8`. This supports the interpretation that Table 3
  swapped the polynomial and derivative bounds, although it does not resolve
  the reported row count. The current implementation explicitly follows Table 3
  (155 columns); comparisons with the 181-column experiment must first fix the
  same library for every method.
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
