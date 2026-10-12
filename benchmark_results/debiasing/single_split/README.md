# One-way data splitting: IB

## Protocol

- Fixed true equation: `u_t = -0.5 (u^2)_x`, with the original archived `burgers.mat` trajectory.
- Noise ratios: 0, 0.2, 0.5, 0.75, 1; 100 trials per ratio. Only the noise changes across trials.
- All six methods receive the same noisy 256 x 256 observations and identical noise seeds.
- `split-debiased` and `split-debiased-lasso` estimate the whole function from the 128 odd time
  indices. Only the 128 even time indices enter correction, integration and regression.
- Noise estimation, averaging-window selection and state scales use the odd observations only.
  The two groups are not swapped or pooled for regression.
- Test functions remain `Tests(m=(60,60), p=(7,7), s=(5,5), shape="poly")`, with 784 weak rows
  and the original 43-term polynomial dictionary. Their physical support and centers are unchanged.
- Every retained observation has time quadrature weight `2*dt`; training observations have weight zero.
- The averaging rule uses `TimeSplitPilot(prior=0.01)`. No parameter tuning was performed.

The reference methods `debiased` and `debiased-lasso` use both time parities: each point is estimated
from the opposite parity and all corrected points are integrated. The strict split instead reserves
one group for estimation and the other for regression. This changes the number of observations
contributing to each integral and uses training-only scaling, so the performance comparison includes
those changes as well.

## Results

[Full tables](results.md), [summary CSV](results.summary.csv), [plot](results.png).

MSTLS exact support recovery:

| Noise ratio | Existing debiased | One-way split |
| ---: | ---: | ---: |
| 0 | 0% | 0% |
| 0.2 | 8% | 13% |
| 0.5 | 25% | 12% |
| 0.75 | 22% | 24% |
| 1 | 24% | 24% |

Both debiased LASSO variants have 0% exact support recovery at every tested noise ratio. Splitting
does not consistently improve recovery with this averaging rule. The selected averaging window is
61 x 61 in these runs. On clean data its relative estimation RMSE at regression points is 0.261,
so the function-estimation error is substantial even without observation noise. This is a diagnostic
of the retained averaging choice; it does not isolate the cause of every regression failure.

LASSO solves `||Z theta - y||^2/(2K) + lambda ||theta||_1` after RMS-normalizing columns and response.
The code searches its existing normalized penalty path, `lambda = ratio * lambda_max`, where
`lambda_max = max(abs(Z.T @ y))/K`, and selects the minimum of projection mismatch plus nonzero
fraction. Tables show the mean selected final lambda; each trial stores `selected_alpha`. Physical
PDE coefficients are restored after regression.

## Validation and theoretical scope

The split-specific tests verify that changing regression observations leaves the entire function
estimate and scaling unchanged; raw training values do not enter the integrals; the integrals agree
with direct physical sums; and the corrected weak matrices are affine in regression observations
conditional on the training data. Both parities and odd test-function half-widths are covered.

Thus the training-data conditioning argument gives Gaussian residuals under independent Gaussian
regression noise. This does not by itself establish coefficient-error or support-recovery guarantees
for MSTLS/LASSO. Function-estimation remainder, quadrature error, matrix conditioning and sparse
selection still need analysis.

[Verification](verification.json) records schedule completeness, seed matching and source hashes.
[Diagnostics](diagnostics.json) use clean truth only for evaluation. The known-support least-squares
fits there are diagnostic calculations and are not included as benchmark methods.

## Reproduce

```sh
python run.py IB --method wsindy debiased split-debiased lasso debiased-lasso split-debiased-lasso \
  --noise 0 0.2 0.5 0.75 1 --trials 100 --workers 2 \
  --out benchmark_results/debiasing/single_split/runs
python report.py benchmark_results/debiasing/single_split/runs \
  -o benchmark_results/debiasing/single_split/results.md \
  --plot benchmark_results/debiasing/single_split/results.png
```

`report.scope.json` keeps this comparison out of parent-directory report scans, preventing the
rerun reference trials from duplicating the main benchmark. Select this directory explicitly to report it.
