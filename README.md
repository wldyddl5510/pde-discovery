# Sparse PDE coefficient recovery

`wsindy.py`: weak form / MSTLS; `wendy.py`: IRLS / HT / L1;
`wendy_mle.py`: weak likelihood; `experiment.py`: synthetic data and three tables.
Notation follows Section 2 of `pde_discovery.pdf`.

```sh
conda activate pde_discovery
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiment.py \
  --setup paper --noise 0 .2 --seeds 3 --workers 4 \
  --output results/my_comparison
python -m unittest -v test_wendy
```

`--setup paper` is the default. Original author datasets are downloaded once
into ignored `tmp/WSINDy_PDE/datasets/`, pinned to commit
`95686ccd9e32e3a9f62acfb3014fb77d5ef039ab` of
[WSINDy_PDE](https://github.com/dm973/WSINDy_PDE).

Full-covariance fits at this resolution can take minutes per fit, so the
method comparison takes hours. To reproduce the faster WSINDy noise sweep:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiment.py \
  --methods WSINDy --noise 0 .1 .2 .5 1 --seeds 200 \
  --output results/my_wsindy_reference
```

| Paper Tables 3–4 | Burgers | KdV |
|---|---:|---:|
| Grid (space × time) | 256 × 256 | 400 × 601 |
| Polynomial powers / spatial derivatives | 0–6 / 0–6 | 0–6 / 0–6 |
| Test half-widths (mx, mt), grid steps | (60, 60) | (45, 80) |
| Query strides (sx, st) | (5, 5) | (8, 12) |
| Test polynomial degrees (px, pt) | (7, 7) | (8, 7) |
| Weak equations × fitted columns | 784 × 43 | 1443 × 43 |

The full vector has J=S=7 and 49 entries in operator-major order; six
constant-derivative columns are structural zeros. Burgers has Python
`w[9]=-.5`; KdV additionally has `w[22]=-1`, i.e.
`u_t=-.5 D_x(u²)-D_x³(u)`. These are PDF indices w_10 and w_23.

The weak form uses the author's separable FFT convolution and rescaling,
with fixed Table 3 windows, decay tolerance 1e-10 and 50 MSTLS thresholds
log-spaced from 1e-4 to 1. The rescaling follows the author code where its
formula differs from the printed paper; both Table 5 errors are reproduced.

## Selection and estimation

Every method starts with the full admissible library and observed weak OLS.
WENDy methods receive the known injected sigma. There is no oracle support
or WSINDy-based initialization. HT/L1 are our sparse PDE extensions of
WENDy and WENDy-MLE, not their original sparse discovery algorithms.

| Variant | Selection |
|---|---|
| WSINDy | MSTLS threshold search with support refitting |
| WENDy HT / WENDy-MLE HT | Full-library fit, then keep s largest absolute rescaled coefficients |
| WENDy L1 / WENDy-MLE L1 | Penalize sum of absolute rescaled coefficients; no HT |

HT knows s=1 for Burgers and s=2 for KdV. No HT/L1 post-selection refit is
performed. In paper mode, fitting and selection use rescaled coordinates;
estimates are converted back to physical units for coefficient errors.
Rescaling is computed from observed U and held fixed during each fit; sigma
is transformed to those same coordinates.
For `ConvolutionSystem`, direct fits take `sigma*system.noise_scale` and return
rescaled coefficients. The runner saves `system.coefficient_scale*result.w`.
The L1 GLS objective evaluates the residual directly, avoiding cancellation
from expanding the quadratic through X.T@X.
The penalty is alpha * lambda_ref with alpha in {1e-6, 1e-4, .01} and
lambda_ref=max(abs(X_0.T@y_0))/K, where X_0,y_0 are whitened by C(0).
The reference is fixed per fit; no alpha is chosen using ground truth.

```text
R(w) = Y_hat - X_hat w
L(w) = A_0 - sum_sj w_sj A_s diag(f'_j(U))
C(w) = sigma² L(w)L(w).T + ridge I
WENDy: minimize R(w).T inv(C(w_old)) R(w)/(2K) + lambda ||w||_1
MLE: minimize [logdet C(w) + R(w).T inv(C(w)) R(w)]/(2K) + lambda ||w||_1
```

The full covariance is retained. Shared sparse stencils avoid storing all
43² covariance blocks. `--workers` controls FFT and covariance threads
(default 1); splitting exact products into blocks does not approximate C.
MLE uses scaled BFGS for the large paper system and trust-exact for the small
legacy system. L1 uses L-BFGS-B with split nonnegative variables. At sigma=0,
both covariance methods fall back to ordinary weak LS with HT/L1.
The ridge is 1e-10 times mean(diag(sigma² A_0 A_0.T)).

## Experiments and tables

- [Paper comparison](results/paper_comparison.md): setup, published WSINDy
  results versus this implementation, and the method comparison.
- `results/wsindy_paper_reference/`: WSINDy, noise ratios 0, .1, .2, .5, 1;
  200 seeds per positive level and one noiseless run, one thread.
- `results/paper_method_comparison/`: all nine variants, noise ratios 0 and .2;
  three paired seeds at .2 and one noiseless run, four threads.
- `results/selection_comparison/`: retained historical small-library baseline.

Each run writes only `config.json`, `trials.csv`, `support.md`,
`coefficient_accuracy.md`, and `runtime.md`. Paper-mode tables use:

| Metric | Definition |
|---|---|
| Exact support recovery | Percentage with FP=FN=0 |
| Paper TPR | Mean TP/(TP+FP+FN), not recall or exact recovery |
| E2 | Mean ||w_hat-w_star||₂ / ||w_star||₂, in physical units |
| E∞ | Mean maximum relative error on true nonzero coefficients |
| Runtime | Median seconds including weak-form assembly and all fitting |

Data loading/generation and file writing are excluded from runtime. Support
uses abs(rescaled coefficient)>1e-12. Nonfinite fits count as failed recovery;
their TPR is scored as zero in tables and coefficient errors as infinity.
Optimizer failures remain in all summaries and are flagged in `trials.csv`.
Optimizer termination alone does not imply correct coefficients.
On a numerical exception, iteration count 0 means unavailable, not zero work.
Interrupted runs can continue with the same command plus `--resume`; completed
fits are skipped and changes to saved settings are rejected.

Noise is additive iid Gaussian with sigma=noise_ratio*RMS(clean u), following
paper Eq. 5.1. Actual sigma is also recorded. Defaults are noise ratios 0 and
.2 with three noisy seeds, not the paper's complete 41-level × 200-seed sweep.
`--maxiter 300` and `--tol 1e-8` control the iterative fits. An iteration limit
is reported as nonconvergence, not accepted as a converged solution.

`--setup legacy` reproduces the previous degree-3, 121-test-function setup
with scaled Burgers and a different KdV trajectory. Only legacy mode uses
`--grids` and `--centers`; its HT/L1 uses physical coefficients and its error
tables use medians. The saved historical baseline predates the direct-residual
L1 evaluation; the mathematical objective is unchanged. For example:

```sh
python experiment.py --setup legacy --grids 128 --noise 0 .1 --seeds 1
```

Sources: [WSINDy](https://arxiv.org/abs/2007.02848),
[WENDy](https://arxiv.org/abs/2302.13271),
[WENDy-MLE](https://arxiv.org/abs/2502.08881).
