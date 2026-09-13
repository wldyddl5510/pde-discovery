# PDE coefficient selection

Minimal PDE implementations using Section 2 of `pde_discovery.pdf`:
`wsindy.py` (weak form and MSTLS), `wendy.py` (IRLS and sparse helpers),
`wendy_mle.py` (likelihood and MLE), `experiment.py` (data and comparison tables).

```sh
conda activate pde_discovery
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiment.py
python -m unittest -v test_wendy
```

Default: Burgers and KdV, 256² grid, five paired seeds per nonzero noise level.
`--noise 0 .01 .1 .5 1` specifies sigma/RMS(u); Tables and raw results also show actual
sigma. Noiseless data run once. Example of a smaller run:

```sh
python experiment.py --grids 128 --noise 0 .1 --seeds 1 --output results/my_trial
```

## Selection and objectives

Every estimator starts from the full admissible library and observed weak OLS.
There are no oracle/discovery modes or WSINDy-selected inputs to WENDy.

| Variant | Coefficient selection |
|---|---|
| WSINDy | MSTLS, including its threshold search and support refit |
| WENDy HT | Full-library IRLS, then keep the largest s raw absolute coefficients |
| WENDy-MLE HT | Full-library MLE, then the same hard threshold |
| WENDy L1 | Add lambda times sum(abs(w)) to each frozen-covariance GLS subproblem |
| WENDy-MLE L1 | Add lambda times sum(abs(w)) to the weak negative log likelihood |

HT knows only the term count: Burgers s=1, KdV s=2 (`--sparsity` overrides it).
It does not refit after thresholding. L1 does not use s or the true support,
and there is no post-selection refit. Thus L1 shrinkage contributes to error.
These HT/L1 extensions are experimental PDE adaptations, not native sparse
selection algorithms from the original WENDy or WENDy-MLE papers.

```text
R(w) = Y_hat - X_hat w
C(w) = sigma² L(w)L(w).T + ridge I
L(w) = A_0 - sum_sj w_sj A_s diag(f'_j(U))
WENDy step: min_w R(w).T inv(C(w_old)) R(w)/(2K) + lambda ||w||_1
MLE: min_w [logdet C(w) + R(w).T inv(C(w)) R(w)]/(2K) + lambda ||w||_1
```

The default relative L1 strengths are `--l1 1e-6 1e-4 .01`. All are reported;
none is chosen using true coefficients or support. With X_0,y_0 whitened by
C(0), set lambda = alpha * max(abs(X_0.T @ y_0))/K, fixed throughout each fit.
This is the zero-solution threshold for the reference GLS LASSO, not a claim
about the nonconvex MLE optimum. Raw coefficients are penalized: feature units
and correlated polynomial columns affect selection. Internal variable scaling
only conditions the optimizer and does not change this penalty.
L1 uses w=positive-negative with L-BFGS-B bounds; HT MLE uses trust-exact.
At sigma=0 both covariance methods use ordinary weak LS, retaining HT or L1.

## Data and notation

The dictionary is {1,u,u²,u³}. Burgers has w_7=-0.5 in u_t=-0.5 D_x(u²).
KdV additionally has w_14=-1 multiplying D_x³(u). Other entries are zero.
Burgers uses the WSINDy paper's exact entropy solution with x,u divided by
1000; KdV uses a different exact two-soliton trajectory of the paper's PDE.

A_s[k,i]=(-1)^|alpha^s| D^alpha^s psi_k(z_i) Delta z_i;
Y_hat=A_0 U, X_hat[:,(s-1)*J+j]=A_s f_j(U). Coefficients retain all S*J entries
in operator-major order. Python j is zero-based; w retains the full operator-major vector.
Structural zero columns, e.g. D_x(1), remain zero and are excluded from fits.
U is flattened in (time,space) order, alpha=(dx,dt). The code follows Eq. (5),
resolving the PDF's inconsistent transpose and Taylor sign.

These experiments use fixed compact polynomial test functions, trapezoidal
quadrature, full residual covariance, and known injected sigma. They are not
full reproductions of the original software. The fixed ridge is 1e-10 times
mean(diag(sigma² A_0 A_0.T)); quadrature error at shocks remains.

## Outputs

Only the latest comparison is kept in `results/selection_comparison/`:

- [support.md](results/selection_comparison/support.md): exact support recovery
  percentage, with methods as rows and noise levels as columns.
- [coefficient_accuracy.md](results/selection_comparison/coefficient_accuracy.md):
  median relative coefficient error (%) in the same layout; lower is better.
- `trials.csv`: per-seed estimates (`w`, a JSON vector), errors, support metrics,
  timing and optimizer status. This is the single source for both tables.
- `config.json`: experiment arguments. The saved run uses ten paired seeds.

Support is abs(w)>1e-12. Nonfinite fits count as failed recovery and have infinite
coefficient error. Failed optimizer results remain included and are flagged;
optimizer termination does not imply recovery of the true coefficients.
Runtime includes weak-form setup, OLS initialization, covariance construction
and fitting. Estimators retain their in-memory iteration histories; the runner
exports only final estimates. No plots or separate trace/summary files are made.

Sources: [WSINDy](https://arxiv.org/abs/2007.02848),
[WENDy](https://arxiv.org/abs/2302.13271),
[WENDy-MLE](https://arxiv.org/abs/2502.08881).
