# Sparse PDE recovery

**[Results: tables and figure](results/paper_comparison.md)**

[Implementation corrections](results/implementation_check.md): L1 convergence checks, SVD test conditioning, and second-order MLE optimization. All 23 numerical tests pass.

Compare WSINDy, WENDy, and WENDy-MLE on Burgers and KdV using the
WSINDy paper's data and candidate library. Notation follows Section 2 of `pde_discovery.pdf`.

| File | Purpose |
|---|---|
| `wsindy.py` | Weak form and MSTLS selection |
| `wendy.py` | IRLS with HT or L1 selection |
| `wendy_mle.py` | Weak likelihood with HT or L1 selection |
| `experiment.py` | Synthetic experiments and summary table |
| `dimension_experiment.py` | Exact linear PDEs in one and five spatial dimensions |
| `report.py` | Rebuild the paper comparison and figure from saved results |

```sh
conda activate pde_discovery
python report.py                  # Refresh presentation; no experiments rerun
python -m unittest test_wendy
```

Run the method comparison:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiment.py \
  --setup paper --noise 0 .2 --seeds 3 --workers 4 \
  --l1 1e-10 1e-8 1e-6 1e-4 .01 --time-limit 200 \
  --output results/my_comparison
```

For the WSINDy noise sweep, use `--methods WSINDy --noise 0 .1 .2 .5 1 --seeds 200 --workers 1`.
Each run saves `summary.md`, `trials.csv`, and `config.json`. Tables have methods as columns, one table per noise level.
Add `--resume` to continue or extend the L1 grid; completed fits are reused; settings and recorded solver hashes must match.

**[Strong signals with weak background: tables](results/strong_weak/summary.md)**

A separate quick experiment uses 19 coefficients: three of magnitude 1 and sixteen of magnitude 0.001.
The periodic PDE includes every weak term. One paired noise seed covers 0–100% noise.
Tables report recovery of the three dominant coordinates, strong/full coefficient error, and runtime.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiment.py \
  --setup signal --grids 129 --centers 8 --noise 0 .01 .05 .1 .2 .5 1 \
  --seeds 1 --workers 1 --l1 1e-4 --maxiter 1000 --time-limit 5 \
  --output results/my_strong_weak
```

**[Five spatial dimensions: tables](results/spatial_dimension/summary.md)**

A linear reaction–advection–diffusion PDE on [−π,π)⁵ has 21 coefficients: three of magnitude 1 and eighteen of magnitude 0.001.
An exact Fourier solution and Fourier weak tests keep the experiment quick. The 1D reference uses the same operator family;
both cases have 32,768 spatial observations × 129 times, one seed, and 0–100% noise.
This is a separate benchmark from the nonlinear strong/weak experiment above.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python dimension_experiment.py \
  --output results/my_spatial_dimension
```

Noise is iid Gaussian with σ = noise ratio × RMS(clean u). WENDy variants receive σ (adjusted for Fourier projection in the dimension experiment).
The following settings describe the paper comparison:
HT keeps the s largest rescaled coefficients (Burgers s=1, KdV s=2).
L1 uses λ=αλ_ref, α∈{1e-10, 1e-8, 1e-6, 1e-4, .01}, with λ_ref=max|X₀ᵀy₀|/K from whitening by C(0).
All methods start from the full library; HT/L1 have no post-selection refit.
WENDy orthonormalizes the WSINDy tests by SVD (52 Burgers / 200 KdV equations), retaining the full projected covariance.
MLE uses trust-region Newton-CG; MLE-L1 uses L-BFGS-B with a KKT check. WENDy uses IRLS with an active-set LASSO subsolver.
These are sparse PDE extensions; they do not reproduce the original ODE experiments.
Fits have a 200-second limit and at most 300 outer iterations; timeouts are failures. At zero noise, MLE uses the WENDy least-squares/LASSO fallback.
Support uses |rescaled coefficient|>1e-12; errors use physical coefficients.
Failed fits remain in summaries; optimizer completion does not imply correct recovery.

Author data are downloaded into ignored `tmp/`, pinned to
[WSINDy_PDE commit 95686cc](https://github.com/dm973/WSINDy_PDE/tree/95686ccd9e32e3a9f62acfb3014fb77d5ef039ab).
Exact settings and termination messages are in each result folder's JSON and CSV.
`--setup legacy` uses the previous smaller library; its error summaries use medians.

Sources: [WSINDy](https://arxiv.org/abs/2007.02848) ·
[WENDy](https://arxiv.org/abs/2302.13271) · [WENDy-MLE](https://arxiv.org/abs/2502.08881).
