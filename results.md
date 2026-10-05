# WSINDy benchmark: MSTLS and LASSO

## LASSO penalty selection

For K weak rows, normalize each nonzero library column as `Z_j=G_j/RMS(G_j)` and each response as `y_e=b_e/RMS(b_e)`. Solve `||Z theta-y_e||_2^2/(2*K) + lambda_e*||theta||_1`, without centering or an extra intercept; the constant library term is penalized.

For each trial and equation, define `lambda_max,e = max(abs(Z.T @ y_e))/K`, the smallest penalty for which the zero model is optimal. Evaluate 100 logarithmically spaced candidates from `1e-4*lambda_max,e` to `lambda_max,e`: `lambda_e,k = lambda_max,e * 10**(-4 + 4*k/99)`, for `k=0,...,99`. Coupled equations have separate lambda_max values and use the same candidate index k.

Choose the candidate that minimizes the existing WSINDy score: matrix 2-norm prediction difference from full OLS, divided by the full OLS prediction norm, plus the fraction of nonzero coefficients. Exact ties select the smallest lambda. Selection uses the observed weak system G/b; it uses neither the true equation nor cross-validation.

The `lambda median` column gives the median selected lambda over the 100 trials, separately for each LHS equation. Lambda multiplies the L1 penalty on RMS-normalized coefficients; physical coefficients are restored afterward. Retain the penalized coefficients, including shrinkage, with no OLS support refit or coefficient cutoff. Empty supports are allowed. CSV/JSON summaries retain the full penalty statistics; MSTLS rows have no LASSO penalty.

The scikit-learn LARS Gram path is checked against KKT conditions at every candidate. Primal active-set polishing is tried first, followed by coordinate descent and final polishing if needed; violation divided by lambda_max above `1e-7` rejects the fit. These LASSO runs are additional regression comparisons to the papers' published algorithms.

Raw and filtered observations are each fitted with the existing MSTLS/STLS profile and LASSO. For each PDE/noise condition, all rows use the intersection of completed trial keys across the methods listed for that PDE. Seeds and injected noise are checked per trial. Each PDE has one fixed equation and trajectory; only observation noise varies. Zero-noise repeats are identical.

The original six polynomial PDEs use the existing author-code WSINDy construction and component RMS noise convention. HKS/VBG use the existing declared resimulations, physical bump tests and centered sample-standard-deviation noise convention. Their MSTLS-labelled baseline uses the consistency paper's absolute STLS profile. Libraries, weak supports and preprocessing are fixed within each suite; the two suites have different protocols.

TPR=TP/(TP+FN+FP); Exact is exact-support recovery; E2 is relative coefficient L2 error; E_inf is maximum relative error over true coefficients. Mean ± sample SD includes failed identifications. LASSO errors include shrinkage. Timings include preprocessing/weak systems/regression and exclude loading/noise/reporting; recorded concurrency differs between runs.

## Saved runs

- [WSINDy + MSTLS: KS, NLS, RD, IB, KdV, NS](benchmark_results/authors/results.md): 3,000/3,000 active polynomial fits.
- [Filtered WSINDy + MSTLS: KS, NLS, RD, IB, KdV, NS](benchmark_results/filtered/results.md): 3,000/3,000 active polynomial fits.
- [WSINDy + MSTLS: HKS, VBG](benchmark_results/consistency/raw/results.md): 1,000/1,000 active polynomial fits.
- [Filtered WSINDy + MSTLS: HKS, VBG](benchmark_results/consistency/filtered/results.md): 1,000/1,000 active polynomial fits.
- [WSINDy + LASSO: RD, NS, NLS, KS, KdV, IB](benchmark_results/lasso/original/raw/results.md): 3,000/3,000 active polynomial fits.
- [Filtered WSINDy + LASSO: RD, NS, NLS, KS, KdV, IB](benchmark_results/lasso/original/filtered/results.md): 3,000/3,000 active polynomial fits.
- [WSINDy + LASSO: HKS, VBG](benchmark_results/lasso/consistency/raw/results.md): 1,000/1,000 active polynomial fits.
- [Filtered WSINDy + LASSO: HKS, VBG](benchmark_results/lasso/consistency/filtered/results.md): 1,000/1,000 active polynomial fits.

Baseline suites status: complete.

Debiasing suite status: complete; 8,000/8,000 fits from completed PDE reports are included below.

## IB

IB debiasing: Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each. Saved trial reports: [MSTLS](benchmark_results/debiasing/time2_foldwise/IB/mstls/results.md); [LASSO](benchmark_results/debiasing/time2_foldwise/IB/lasso/results.md).

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x}
\end{aligned}
$$

Observation shape `(256, 256)`; library 43 terms; weak half-widths `(60, 60)`, strides `(5, 5)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 4.30846e-05 ± 0 | 4.30846e-05 ± 0 | 0.0253 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.2 ± 5.58e-17 | 0.258283 ± 5.58e-17 | 2.88961 ± 4.46e-16 | 0.0215 | — |
| WSINDy + LASSO | 100 | 100.0% | 1 ± 0 | 0.00183166 ± 0 | 0.00183166 ± 0 | 0.0335 | u=0.001789 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.890185 ± 1.12e-16 | 870.704 ± 0 | 0.0255 | u=0.01517 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.166667 ± 5.58e-17 | 0.0122654 ± 0 | 27.4227 ± 7.14e-15 | 0.0242 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.658965 ± 1.12e-16 | 638.54 ± 0 | 0.0241 | u=0.0166 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.00408295 ± 0.00318 | 0.00408295 ± 0.00318 | 0.0253 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.170988 ± 0.0325 | 0.315689 ± 0.209 | 128.567 ± 263 | 0.0213 | — |
| WSINDy + LASSO | 100 | 25.0% | 0.625 ± 0.218 | 0.0778663 ± 0.0679 | 34.6959 ± 44.8 | 0.0378 | u=0.009538 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.891891 ± 0.00943 | 870.693 ± 10.7 | 0.0239 | u=0.01663 |
| Debiased WSINDy (time2) + MSTLS | 100 | 8.0% | 0.334262 ± 0.212 | 0.0584391 ± 0.0972 | 26.0369 ± 100 | 0.0246 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.662164 ± 0.024 | 639.399 ± 22.5 | 0.025 | u=0.0182 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 99.0% | 0.991111 ± 0.0889 | 0.0109775 ± 0.0094 | 0.940651 ± 9.3 | 0.0241 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.168206 ± 0.0387 | 0.340184 ± 0.278 | 135.368 ± 268 | 0.0211 | — |
| WSINDy + LASSO | 100 | 21.0% | 0.561667 ± 0.261 | 0.262172 ± 0.278 | 40544 ± 4.03e+05 | 0.0388 | u=0.02094 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.491667 ± 0.0365 | 0.89946 ± 0.0208 | 871.311 ± 26.3 | 0.0237 | u=0.02098 |
| Debiased WSINDy (time2) + MSTLS | 100 | 25.0% | 0.477833 ± 0.312 | 0.0510718 ± 0.0322 | 15.5908 ± 25 | 0.0245 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.498333 ± 0.0167 | 0.684961 ± 0.0616 | 651.399 ± 55.5 | 0.0248 | u=0.02627 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0182255 ± 0.0129 | 0.0182255 ± 0.0129 | 0.0236 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.17306 ± 0.0415 | 0.38071 ± 0.337 | 114.619 ± 260 | 0.0212 | — |
| WSINDy + LASSO | 100 | 21.0% | 0.540833 ± 0.272 | 0.349902 ± 0.268 | 61863.4 ± 6.16e+05 | 0.0397 | u=0.03954 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.474167 ± 0.0623 | 0.898433 ± 0.033 | 232531 ± 2.32e+06 | 0.0247 | u=0.02407 |
| Debiased WSINDy (time2) + MSTLS | 100 | 22.0% | 0.470667 ± 0.297 | 0.0540189 ± 0.0661 | 21.8176 ± 65.5 | 0.0243 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.473333 ± 0.102 | 0.695807 ± 0.101 | 648.556 ± 92.1 | 0.0254 | u=0.0273 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 99.0% | 0.995 ± 0.05 | 0.0214272 ± 0.0151 | 0.0219393 ± 0.016 | 0.0243 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.18081 ± 0.0423 | 0.449489 ± 0.433 | 105.431 ± 237 | 0.0202 | — |
| WSINDy + LASSO | 100 | 5.0% | 0.416667 ± 0.203 | 0.526774 ± 0.286 | 77815.3 ± 7.72e+05 | 0.0411 | u=0.04464 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.446667 ± 0.108 | 0.905774 ± 0.0416 | 1.32148e+06 ± 5.85e+06 | 0.0281 | u=0.02392 |
| Debiased WSINDy (time2) + MSTLS | 100 | 24.0% | 0.4955 ± 0.297 | 0.0596323 ± 0.0743 | 22.9425 ± 68.3 | 0.0244 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.485 ± 0.0753 | 0.709272 ± 0.113 | 654.127 ± 113 | 0.0255 | u=0.03736 |

## KdV

KdV debiasing: Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each. Saved trial reports: [MSTLS](benchmark_results/debiasing/time2_foldwise/KdV/mstls/results.md); [LASSO](benchmark_results/debiasing/time2_foldwise/KdV/lasso/results.md).

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xxx}
\end{aligned}
$$

Observation shape `(400, 601)`; library 43 terms; weak half-widths `(45, 80)`, strides `(8, 12)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 3.13515e-07 ± 0 | 2.84321e-07 ± 1.06e-22 | 0.0691 | — |
| Filtered WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 3.13515e-07 ± 0 | 2.84321e-07 ± 1.06e-22 | 0.0593 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.0139033 ± 5.23e-18 | 0.0124488 ± 5.23e-18 | 0.109 | u=9.987e-05 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.0139033 ± 5.23e-18 | 0.0124488 ± 5.23e-18 | 0.0642 | u=9.987e-05 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.00132425 ± 0 | 0.00120636 ± 2.18e-19 | 0.0838 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.0141497 ± 0 | 0.0126812 ± 1.74e-18 | 0.0848 | u=9.987e-05 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 98.0% | 0.988333 ± 0.0829 | 0.0277983 ± 0.14 | 0.250095 ± 2.34 | 0.0749 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.457302 ± 0.0955 | 5.19599 ± 1.6 | 4.97227 ± 1.8 | 0.0683 | — |
| WSINDy + LASSO | 100 | 1.0% | 0.286667 ± 0.129 | 0.927385 ± 0.248 | 0.830413 ± 0.222 | 0.114 | u=0.007211 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 1 ± 0 | 385.212 ± 0.817 | 0.0681 | u=0.03004 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.5 ± 0 | 1 ± 0 | 0.94095 ± 0.000279 | 0.0862 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.662333 ± 0.0313 | 1.3462 ± 0.102 | 1.09786 ± 0.117 | 0.0854 | u=0.01077 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 95.0% | 0.971667 ± 0.125 | 0.0706185 ± 0.215 | 0.0631837 ± 0.192 | 0.0755 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.44746 ± 0.107 | 5.61902 ± 1.09 | 5.52692 ± 1.53 | 0.0701 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.275 ± 0.0995 | 0.958036 ± 0.168 | 0.857904 ± 0.15 | 0.115 | u=0.007208 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 1 ± 0 | 384.979 ± 2.34 | 0.0742 | u=0.03003 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.5 ± 0 | 1 ± 0 | 0.941011 ± 0.000713 | 0.0866 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.656667 ± 0.0398 | 1.34893 ± 0.0496 | 1.1166 ± 0.106 | 0.0855 | u=0.009823 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 94.0% | 0.971667 ± 0.114 | 0.0813984 ± 0.213 | 0.146024 ± 0.751 | 0.0755 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.417437 ± 0.114 | 5.75482 ± 0.741 | 7.00888 ± 11.1 | 0.0706 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.265833 ± 0.0498 | 0.995386 ± 0.0461 | 0.891908 ± 0.0414 | 0.117 | u=0.007904 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 1 ± 0 | 384.765 ± 3.65 | 0.0723 | u=0.03002 |
| Debiased WSINDy (time2) + MSTLS | 100 | 18.0% | 0.59 ± 0.193 | 0.961619 ± 0.0825 | 0.852086 ± 0.191 | 0.0873 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.632333 ± 0.0697 | 1.38459 ± 0.129 | 1.18492 ± 0.194 | 0.0866 | u=0.00981 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 99.0% | 0.995 ± 0.05 | 0.0520222 ± 0.103 | 0.0465661 ± 0.0917 | 0.0757 | — |
| Filtered WSINDy + MSTLS | 100 | 1.0% | 0.397802 ± 0.142 | 5.66193 ± 1.24 | 10.5294 ± 22 | 0.0705 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.2975 ± 0.0559 | 0.994591 ± 0.0541 | 0.892453 ± 0.0487 | 0.116 | u=0.02529 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 1 ± 0 | 383.727 ± 4.36 | 0.0743 | u=0.02999 |
| Debiased WSINDy (time2) + MSTLS | 100 | 58.0% | 0.789 ± 0.249 | 0.930331 ± 0.562 | 4.22309 ± 35.6 | 0.0874 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.618095 ± 0.0968 | 1.37355 ± 0.145 | 4.70232 ± 17.9 | 0.0865 | u=0.009795 |

## KS

KS debiasing: Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each. Saved trial reports: [MSTLS](benchmark_results/debiasing/time2_foldwise/KS/mstls/results.md); [LASSO](benchmark_results/debiasing/time2_foldwise/KS/lasso/results.md).

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= -0.5\,\left(u^{2}\right)_{x} - u_{xx} - u_{xxxx}
\end{aligned}
$$

Observation shape `(256, 301)`; library 43 terms; weak half-widths `(23, 22)`, strides `(5, 6)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 8.18316e-07 ± 0 | 5.54625e-07 ± 0 | 0.0176 | — |
| Filtered WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 8.18316e-07 ± 0 | 5.54625e-07 ± 0 | 0.0182 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.272727 ± 5.58e-17 | 0.00920226 ± 0 | 0.00759012 ± 2.62e-18 | 0.0516 | u=5.794e-05 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.272727 ± 5.58e-17 | 0.00920226 ± 0 | 0.00759012 ± 2.62e-18 | 0.0256 | u=5.794e-05 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.00204525 ± 0 | 0.00170512 ± 4.36e-19 | 0.0327 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.333333 ± 1.12e-16 | 0.0319087 ± 6.97e-18 | 0.0261405 ± 0 | 0.0272 | u=0.0001482 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0122972 ± 0.00551 | 0.0104377 ± 0.0056 | 0.0429 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.0723575 ± 0.0196 | 2.26388 ± 0.339 | 6.69491 ± 1.47 | 0.0376 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.285054 ± 0.0251 | 0.866745 ± 0.257 | 0.711172 ± 0.211 | 0.0537 | u=0.01267 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.391619 ± 0.0277 | 1 ± 0 | 0.977837 ± 0.00175 | 0.0299 | u=0.0306 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.142857 ± 5.58e-17 | 1.60645 ± 0.0324 | 2.56245 ± 0.0909 | 0.0472 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.437095 ± 0.0725 | 1 ± 0 | 0.883672 ± 0.00438 | 0.0289 | u=0.03323 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0722229 ± 0.0168 | 0.0670009 ± 0.0173 | 0.0459 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.0992914 ± 0.0272 | 2.10143 ± 0.407 | 3.50823 ± 2.15 | 0.0371 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.262917 ± 0.0333 | 0.903213 ± 0.203 | 0.737055 ± 0.166 | 0.0478 | u=0.01213 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.00666667 ± 0.0328 | 1 ± 0 | 0.999641 ± 0.00198 | 0.0294 | u=0.05897 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.0147469 ± 0.0455 | 1.10817 ± 0.31 | 1.68134 ± 0.99 | 0.0419 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.162381 ± 0.00919 | 1 ± 0 | 1.04377 ± 0.00578 | 0.0293 | u=0.05776 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 98.0% | 0.995 ± 0.0352 | 0.164424 ± 0.0274 | 0.15789 ± 0.0246 | 0.0504 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.0895441 ± 0.0297 | 2.20814 ± 0.423 | 4.02027 ± 2.05 | 0.0371 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.26944 ± 0.0447 | 0.956166 ± 0.16 | 0.780472 ± 0.126 | 0.0472 | u=0.01495 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.0317143 ± 0.0732 | 1.00071 ± 0.00502 | 0.998044 ± 0.00451 | 0.0288 | u=0.05766 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.0108683 ± 0.0388 | 1.0852 ± 0.273 | 1.56451 ± 0.887 | 0.0421 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.164369 ± 0.0214 | 1 ± 0 | 1.03918 ± 0.0145 | 0.0289 | u=0.05611 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 87.0% | 0.943357 ± 0.158 | 0.326388 ± 0.222 | 0.309675 ± 0.172 | 0.0501 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.090529 ± 0.0351 | 2.20801 ± 0.422 | 4.03428 ± 2.03 | 0.0377 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.289096 ± 0.0508 | 0.979625 ± 0.104 | 0.816684 ± 0.0799 | 0.0464 | u=0.02106 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.0248095 ± 0.0665 | 1.00147 ± 0.00871 | 0.998598 ± 0.00387 | 0.0289 | u=0.05508 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.00907634 ± 0.0277 | 1.09071 ± 0.277 | 1.52218 ± 0.844 | 0.043 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.171083 ± 0.0254 | 1 ± 0 | 1.0322 ± 0.00805 | 0.0292 | u=0.05518 |

## NLS

NLS debiasing: Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each. Saved trial reports: [MSTLS](benchmark_results/debiasing/time2_foldwise/NLS/mstls/results.md); [LASSO](benchmark_results/debiasing/time2_foldwise/NLS/lasso/results.md).

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.5\,v_{xx} + u^{2}\,v + v^{3} \\
v_{t} &= -0.5\,u_{xx} - u\,v^{2} - u^{3}
\end{aligned}
$$

Observation shape `(256, 251)`; library 190 terms; weak half-widths `(19, 25)`, strides `(5, 5)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 9.28616e-08 ± 0 | 5.60489e-08 ± 6.65e-24 | 0.0804 | — |
| Filtered WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 9.28616e-08 ± 0 | 5.60489e-08 ± 6.65e-24 | 0.0863 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.00084906 ± 0 | 0.000618916 ± 1.09e-19 | 0.185 | u=9.083e-05; v=9.087e-05 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 0.00084906 ± 0 | 0.000618916 ± 1.09e-19 | 0.107 | u=9.083e-05; v=9.087e-05 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.000184292 ± 0 | 0.000133458 ± 5.45e-20 | 0.111 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.6 ± 0 | 0.00103542 ± 0 | 0.000439056 ± 1.09e-19 | 0.123 | u=0.0001094; v=0.0001094 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.031011 ± 0.00358 | 0.0200444 ± 0.00267 | 0.586 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.662469 ± 0.0953 | 0.134777 ± 0.0371 | 0.149227 ± 0.0358 | 0.551 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.359777 ± 0.0326 | 0.126324 ± 0.0103 | 0.0593371 ± 0.00781 | 0.199 | u=0.006564; v=0.006566 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.274269 ± 0.0301 | 0.128724 ± 0.0133 | 0.105388 ± 0.00935 | 0.121 | u=0.007592; v=0.007592 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0318603 ± 0.00323 | 0.0120356 ± 0.00211 | 0.58 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.334231 ± 0.0355 | 0.126382 ± 0.0119 | 0.0545315 ± 0.0109 | 0.139 | u=0.006564; v=0.006564 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 74.0% | 0.940207 ± 0.124 | 0.192192 ± 0.166 | 0.13637 ± 0.0799 | 0.811 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.129448 ± 0.0584 | 2.41777 ± 0.798 | 5.35843 ± 1.9 | 0.958 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.245019 ± 0.0222 | 0.243522 ± 0.0198 | 0.154701 ± 0.0104 | 0.221 | u=0.007937; v=0.007939 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.120304 ± 0.0274 | 0.983704 ± 0.0607 | 1.05245 ± 0.0863 | 0.138 | u=0.01246; v=0.01247 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.4393 ± 0.193 | 0.548822 ± 0.348 | 0.732471 ± 0.361 | 1.01 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.174329 ± 0.0125 | 0.485489 ± 0.0895 | 0.377117 ± 0.0548 | 0.164 | u=0.007982; v=0.007991 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 3.0% | 0.636592 ± 0.163 | 0.326789 ± 0.143 | 0.393518 ± 0.124 | 0.887 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.131142 ± 0.0416 | 2.57529 ± 0.767 | 4.97586 ± 2.05 | 1.04 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.232413 ± 0.0186 | 0.378335 ± 0.0264 | 0.285532 ± 0.0192 | 0.231 | u=0.01147; v=0.0115 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.131838 ± 0.0366 | 0.947129 ± 0.0988 | 0.98706 ± 0.122 | 0.147 | u=0.01362; v=0.01363 |
| Debiased WSINDy (time2) + MSTLS | 100 | 2.0% | 0.35525 ± 0.214 | 0.821844 ± 0.333 | 1.33838 ± 0.667 | 1.12 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.17681 ± 0.0156 | 0.483523 ± 0.0589 | 0.318389 ± 0.0617 | 0.177 | u=0.01151; v=0.01153 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.367984 ± 0.19 | 0.587566 ± 0.228 | 1.01973 ± 0.51 | 0.862 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.123039 ± 0.0372 | 2.42934 ± 0.906 | 4.83771 ± 1.67 | 1.09 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.224861 ± 0.0213 | 0.538202 ± 0.0475 | 0.448255 ± 0.0328 | 0.24 | u=0.01389; v=0.01391 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.139004 ± 0.0349 | 0.918399 ± 0.127 | 0.950743 ± 0.125 | 0.141 | u=0.01361; v=0.0136 |
| Debiased WSINDy (time2) + MSTLS | 100 | 2.0% | 0.245133 ± 0.182 | 1.20906 ± 0.498 | 2.13849 ± 0.89 | 1.21 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.179367 ± 0.0243 | 0.595747 ± 0.0504 | 0.318989 ± 0.06 | 0.188 | u=0.01389; v=0.0139 |

## RD

RD debiasing: Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each. Saved trial reports: [MSTLS](benchmark_results/debiasing/time2_foldwise/RD/mstls/results.md); [LASSO](benchmark_results/debiasing/time2_foldwise/RD/lasso/results.md).

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.1\,u_{xx} + 0.1\,u_{yy} - u\,v^{2} - u^{3} + v^{3} + u^{2}\,v + u \\
v_{t} &= 0.1\,v_{xx} + 0.1\,v_{yy} - u\,v^{2} - u^{3} - v^{3} - u^{2}\,v + v
\end{aligned}
$$

Observation shape `(256, 256, 201)`; library 181 terms; weak half-widths `(13, 13, 14)`, strides `(13, 13, 12)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 4.45443e-10 ± 0 | 1.79099e-10 ± 2.6e-26 | 11.1 | — |
| Filtered WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 4.45443e-10 ± 0 | 1.79099e-10 ± 2.6e-26 | 11.8 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.35 ± 1.12e-16 | 0.0153923 ± 0 | 0.0134158 ± 1.74e-18 | 15.3 | u=8.461e-05; v=8.463e-05 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.35 ± 1.12e-16 | 0.0153923 ± 0 | 0.0134158 ± 1.74e-18 | 15.6 | u=8.461e-05; v=8.463e-05 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.000441219 ± 0 | 0.000251307 ± 0 | 23.2 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.368421 ± 5.58e-17 | 0.0381231 ± 6.97e-18 | 0.0329104 ± 1.39e-17 | 23.2 | u=0.0002145; v=0.0002146 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 97.0% | 0.998 ± 0.0114 | 0.0444576 ± 0.00371 | 0.0283019 ± 0.00161 | 14.1 | — |
| Filtered WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0135013 ± 0.00292 | 0.00771947 ± 0.00135 | 11.5 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.285994 ± 0.0149 | 1 ± 0 | 0.911146 ± 0.0143 | 15.8 | u=0.008092; v=0.008095 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.315114 ± 0.021 | 0.715256 ± 0.201 | 0.584312 ± 0.223 | 16.5 | u=0.004199; v=0.004201 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0202617 ± 0.004 | 0.0119097 ± 0.00217 | 24.5 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.299874 ± 0.0159 | 0.972419 ± 0.0776 | 0.890435 ± 0.112 | 22 | u=0.008829; v=0.008831 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.870882 ± 0.014 | 0.106342 ± 0.0221 | 0.0932207 ± 0.00825 | 10.8 | — |
| Filtered WSINDy + MSTLS | 100 | 74.0% | 0.959036 ± 0.0753 | 0.0999785 ± 0.0902 | 0.068204 ± 0.0731 | 10.4 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.17717 ± 0.0123 | 1 ± 0 | 0.950779 ± 0.024 | 16.3 | u=0.00882; v=0.008824 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.290288 ± 0.0188 | 0.932402 ± 0.132 | 0.779452 ± 0.177 | 17.2 | u=0.01007; v=0.01008 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0899071 ± 0.012 | 0.0583135 ± 0.00548 | 24.3 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.208079 ± 0.0128 | 1 ± 0 | 0.877272 ± 0.0456 | 21.1 | u=0.01042; v=0.01043 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.765429 ± 0.0469 | 0.304019 ± 0.119 | 0.260724 ± 0.0478 | 10.4 | — |
| Filtered WSINDy + MSTLS | 100 | 4.0% | 0.751797 ± 0.1 | 0.426226 ± 0.126 | 0.402213 ± 0.146 | 10.5 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.149377 ± 0.0173 | 1 ± 0 | 1.04107 ± 0.0289 | 18.5 | u=0.01035; v=0.01035 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.295891 ± 0.0211 | 0.766422 ± 0.181 | 0.548757 ± 0.199 | 19.1 | u=0.007221; v=0.007219 |
| Debiased WSINDy (time2) + MSTLS | 100 | 60.0% | 0.968667 ± 0.0413 | 0.201127 ± 0.038 | 0.148 ± 0.0446 | 24.4 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.196781 ± 0.0146 | 1 ± 0 | 0.845487 ± 0.0332 | 21.1 | u=0.01227; v=0.01227 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.662298 ± 0.051 | 0.55438 ± 0.188 | 0.47153 ± 0.0719 | 10.5 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.618625 ± 0.11 | 0.598599 ± 0.0527 | 0.65562 ± 0.151 | 10.8 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.0933204 ± 0.0377 | 1 ± 0 | 1.02229 ± 0.0233 | 17.3 | u=0.009949; v=0.009947 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.296331 ± 0.0231 | 0.70345 ± 0.209 | 0.481669 ± 0.191 | 17.3 | u=0.00754; v=0.007545 |
| Debiased WSINDy (time2) + MSTLS | 100 | 24.0% | 0.927128 ± 0.0671 | 0.383092 ± 0.143 | 0.319902 ± 0.115 | 24.6 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.187261 ± 0.0165 | 1 ± 0 | 0.810359 ± 0.034 | 21.1 | u=0.01423; v=0.01424 |

## NS

NS debiasing: Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is reflected; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each. Saved trial reports: [MSTLS](benchmark_results/debiasing/time2_foldwise/NS/mstls/results.md); [LASSO](benchmark_results/debiasing/time2_foldwise/NS/lasso/results.md).

**True equation (physical units):**

$$
\begin{aligned}
\omega_{t} &= -\left(\omega\,u\right)_{x} - \left(\omega\,v\right)_{y} + 0.01\,\omega_{xx} + 0.01\,\omega_{yy}
\end{aligned}
$$

Here $\omega$ is vorticity and $(u,v)$ is the observed incompressible velocity ($u_x+v_y=0$). Only the vorticity equation is identified; $-(\omega u)_x-(\omega v)_y=-u\omega_x-v\omega_y$. The viscosity is $\nu=0.01$.

Observation shape `(324, 149, 201)`; library 50 terms; weak half-widths `(31, 31, 14)`, strides `(12, 12, 8)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.00537952 ± 2.62e-18 | 0.000789496 ± 2.18e-19 | 3.11 | — |
| Filtered WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.00537952 ± 2.62e-18 | 0.000789496 ± 2.18e-19 | 3.65 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.571429 ± 2.23e-16 | 0.606835 ± 1.12e-16 | 0.0425387 ± 6.97e-18 | 4.31 | omega=0.00602 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.571429 ± 2.23e-16 | 0.606835 ± 1.12e-16 | 0.0425387 ± 6.97e-18 | 4.65 | omega=0.00602 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.00435014 ± 0 | 0.000599219 ± 1.09e-19 | 9.59 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.571429 ± 2.23e-16 | 0.097654 ± 0 | 0.00660374 ± 8.72e-19 | 9.58 | omega=0.0009366 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0128688 ± 0.00914 | 0.00131589 ± 0.000509 | 3.42 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.606333 ± 0.225 | 0.542068 ± 0.384 | 0.0704365 ± 0.0196 | 4.13 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.436786 ± 0.0692 | 0.624728 ± 0.41 | 0.0488924 ± 0.0291 | 4.28 | omega=0.009585 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.5 ± 0 | 1 ± 0 | 0.0901335 ± 0.00484 | 4.79 | omega=0.02008 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0121845 ± 0.00927 | 0.00114294 ± 0.000473 | 10.1 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.450893 ± 0.0727 | 0.561745 ± 0.409 | 0.0436035 ± 0.0292 | 10 | omega=0.002164 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.398667 ± 0.0276 | 1 ± 0 | 0.0697783 ± 0.00971 | 3.41 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.31381 ± 0.0235 | 1 ± 0 | 0.244773 ± 0.0726 | 3.96 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.407083 ± 0.0293 | 1 ± 0 | 0.131849 ± 0.0164 | 4.27 | omega=0.01675 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.375 ± 0 | 1 ± 0 | 0.130315 ± 0.011 | 4.78 | omega=0.01504 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.390667 ± 0.0657 | 0.982076 ± 0.126 | 0.0714498 ± 0.0187 | 11.1 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.428571 ± 5.58e-17 | 1 ± 0 | 0.0989538 ± 0.00994 | 11 | omega=0.0139 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.409 ± 0.0288 | 1 ± 0 | 0.0653139 ± 0.0101 | 3.43 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.309008 ± 0.102 | 1 ± 0 | 0.370325 ± 0.198 | 3.82 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.336433 ± 0.0241 | 0.997563 ± 0.0244 | 0.177631 ± 0.0151 | 4.27 | omega=0.007246 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.320348 ± 0.0279 | 1 ± 0 | 0.259175 ± 0.0206 | 5 | omega=0.03418 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.371667 ± 0.0356 | 1 ± 0 | 0.0860582 ± 0.0293 | 9.52 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.398155 ± 0.0274 | 1 ± 0 | 0.111612 ± 0.0139 | 9.43 | omega=0.01525 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.5 ± 0 | 1 ± 0 | 0.0468941 ± 0.00246 | 3.1 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.30867 ± 0.054 | 1 ± 0 | 0.322707 ± 0.124 | 3.8 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.33375 ± 0.00417 | 1 ± 0 | 0.298139 ± 0.00902 | 5.11 | omega=0.01265 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.231923 ± 0.00459 | 1 ± 0 | 0.404132 ± 0.0133 | 5.36 | omega=0.004805 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.37 ± 0.0333 | 1 ± 0 | 0.089269 ± 0.029 | 9.28 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.353988 ± 0.0233 | 1 ± 0 | 0.153071 ± 0.0202 | 9.19 | omega=0.01265 |

## HKS

HKS debiasing: Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each. Saved trial reports: [MSTLS](benchmark_results/debiasing/time2_foldwise/HKS/mstls/results.md); [LASSO](benchmark_results/debiasing/time2_foldwise/HKS/lasso/results.md).

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= u_{xxxx} + 0.75\,u_{xxxxxx} - 0.5\,\left(u^{2}\right)_{x} + 0.1\,\left(u^{2}\right)_{xxx}
\end{aligned}
$$

Declared resimulation: IC `cos(x/16)*(1+sin(x/16))`; periodic boundary; x in `[0.0, 100.53096491487338]`, t in `[0.0, 82.0]`.

Observation shape `(256, 257)`; library 73 terms; weak half-widths `(26, 26)`, strides `(7, 7)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 1.66438e-06 ± 0 | 5.10996e-07 ± 2.13e-22 | 0.0267 | — |
| Filtered WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 1.66438e-06 ± 0 | 5.10996e-07 ± 2.13e-22 | 0.0284 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.375 ± 0 | 1 ± 0 | 0.756139 ± 1.12e-16 | 0.0495 | u=0.006203 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.375 ± 0 | 1 ± 0 | 0.756139 ± 1.12e-16 | 0.0483 | u=0.006203 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.0187338 ± 3.49e-18 | 0.00419541 ± 8.72e-19 | 0.0482 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.375 ± 0 | 1 ± 0 | 0.759471 ± 1.12e-16 | 0.0353 | u=0.006815 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 75.0% | 0.933 ± 0.119 | 0.311273 ± 0.402 | 0.040731 ± 0.0349 | 0.111 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.0528484 ± 0.0162 | 103.489 ± 43.8 | 99.4648 ± 31.7 | 0.0855 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.345298 ± 0.0456 | 1 ± 0 | 0.869025 ± 0.0339 | 0.0488 | u=0.02716 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0 ± 0 | 1 ± 0 | 1.00024 ± 5.08e-05 | 0.0576 | u=0.05077 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.154898 ± 0.0615 | 47.559 ± 14.3 | 8.22241 ± 3.16 | 0.11 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.331714 ± 0.0196 | 7.99581 ± 0.131 | 1.2368 ± 0.00978 | 0.0426 | u=0.04299 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 4.0% | 0.73469 ± 0.116 | 0.986278 ± 0.177 | 0.168049 ± 0.179 | 0.122 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.0618768 ± 0.0164 | 64.4332 ± 16.1 | 93.6061 ± 36.2 | 0.0904 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.342877 ± 0.0493 | 1 ± 0 | 0.870665 ± 0.0232 | 0.0535 | u=0.03499 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.146607 ± 0.0286 | 1 ± 0 | 0.955193 ± 0.0123 | 0.0615 | u=0.04509 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.143486 ± 0.109 | 12.6754 ± 14.7 | 7.68424 ± 10.2 | 0.108 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.409969 ± 0.136 | 12.938 ± 4.03 | 1.51181 ± 0.325 | 0.0425 | u=0.03921 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.628053 ± 0.193 | 1.19907 ± 0.771 | 0.343221 ± 0.284 | 0.121 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.0622497 ± 0.0208 | 59.5208 ± 22 | 82.7057 ± 39.4 | 0.0965 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.32475 ± 0.0595 | 1 ± 0 | 0.86543 ± 0.0319 | 0.0596 | u=0.04305 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.132262 ± 0.06 | 1 ± 0 | 0.965435 ± 0.0197 | 0.0605 | u=0.04668 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.11901 ± 0.102 | 14.3208 ± 17.4 | 8.09673 ± 8.92 | 0.108 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.421672 ± 0.134 | 11.7296 ± 3.86 | 1.43433 ± 0.312 | 0.043 | u=0.04093 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.606274 ± 0.209 | 1.61323 ± 2.66 | 0.483157 ± 0.366 | 0.104 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.0603478 ± 0.0218 | 55.3038 ± 24.3 | 79.685 ± 38 | 0.0949 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.313516 ± 0.0651 | 1 ± 0 | 0.847748 ± 0.0341 | 0.0581 | u=0.04773 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.119405 ± 0.0662 | 1 ± 0 | 0.969243 ± 0.0206 | 0.0621 | u=0.04871 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.119115 ± 0.102 | 15.6073 ± 18.8 | 6.93294 ± 6.6 | 0.122 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.384824 ± 0.141 | 12.5892 ± 6.1 | 1.52108 ± 0.537 | 0.0464 | u=0.03918 |

## VBG

VBG debiasing: Section 5 polynomial linearization `f(pilot) + grad(f)(pilot) dot (U-pilot)`, with raw U on the LHS and in linear columns. Split is always time2. For each evaluation time parity, both the noise estimate and the pilot use only the opposite parity. Sixth differences estimate noise on that training time subgrid; the largest estimate across components selects a common box window using the study's volume rule, prior 0.01, rounded up to odd and at least 3. Spatial extension is periodic; time is reflected. All original weak rows, library terms and benchmark regression/scaling settings are retained. Physical coefficients are restored before scoring. No clean error or true support selects the window. This pooled cross-fit benchmark does not assert the single-pilot conditional Gaussian theorem for the pooled regression. Ratios 0, 0.2, 0.5, 0.75, 1; 100 paired trials each. Saved trial reports: [MSTLS](benchmark_results/debiasing/time2_foldwise/VBG/mstls/results.md); [LASSO](benchmark_results/debiasing/time2_foldwise/VBG/lasso/results.md).

**True equation (physical units):**

$$
\begin{aligned}
u_{t} &= 0.01\,u_{xx} - 0.5\,\left(u^{2}\right)_{x} - u^{3} + 2\,u^{2} + 1
\end{aligned}
$$

Declared resimulation: IC `2*sin(pi*x)`; periodic boundary; x in `[-1.0, 1.0]`, t in `[0.0, 1.5]`.

Observation shape `(512, 451)`; library 43 terms; weak half-widths `(64, 56)`, strides `(12, 11)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 2.18747e-11 ± 0 | 1.17192e-12 ± 2.03e-28 | 0.0541 | — |
| Filtered WSINDy + MSTLS | 100 | 100.0% | 1 ± 0 | 2.18747e-11 ± 0 | 1.17192e-12 ± 2.03e-28 | 0.0575 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.222222 ± 2.79e-17 | 1 ± 0 | 0.936933 ± 3.35e-16 | 0.102 | u=0.006524 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.222222 ± 2.79e-17 | 1 ± 0 | 0.936933 ± 3.35e-16 | 0.11 | u=0.006524 |
| Debiased WSINDy (time2) + MSTLS | 100 | 100.0% | 1 ± 0 | 0.00066011 ± 3.27e-19 | 1.49056e-05 ± 0 | 0.081 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.222222 ± 2.79e-17 | 1 ± 0 | 0.936931 ± 1.12e-16 | 0.0814 | u=0.006524 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.799333 ± 0.0314 | 0.935103 ± 0.238 | 0.0412677 ± 0.0531 | 0.0594 | — |
| Filtered WSINDy + MSTLS | 100 | 9.0% | 0.765746 ± 0.11 | 0.80834 ± 0.154 | 0.0920576 ± 0.102 | 0.0652 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.222222 ± 2.79e-17 | 1 ± 0 | 0.93563 ± 0.00198 | 0.106 | u=0.008614 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.329667 ± 0.0105 | 1 ± 0 | 0.928663 ± 0.00363 | 0.113 | u=0.01337 |
| Debiased WSINDy (time2) + MSTLS | 100 | 20.0% | 0.834238 ± 0.0901 | 0.773952 ± 0.383 | 0.0647664 ± 0.0608 | 0.0967 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.229111 ± 0.0315 | 1 ± 0 | 0.92576 ± 0.00641 | 0.0915 | u=0.01498 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.669655 ± 0.108 | 1.00289 ± 0.106 | 0.288737 ± 0.163 | 0.0611 | — |
| Filtered WSINDy + MSTLS | 100 | 1.0% | 0.654307 ± 0.133 | 2.47793 ± 0.473 | 0.233236 ± 0.22 | 0.0672 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.212495 ± 0.0131 | 1 ± 0 | 0.932625 ± 0.00891 | 0.115 | u=0.01643 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.309 ± 0.0149 | 1.27929 ± 0.11 | 0.927085 ± 0.00453 | 0.118 | u=0.02173 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.714774 ± 0.147 | 1.00062 ± 0.0579 | 0.328902 ± 0.301 | 0.0982 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.315495 ± 0.0249 | 1 ± 0 | 0.923946 ± 0.00729 | 0.0938 | u=0.02122 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.490853 ± 0.0615 | 1.00482 ± 0.0346 | 0.716361 ± 0.0989 | 0.0641 | — |
| Filtered WSINDy + MSTLS | 100 | 1.0% | 0.619222 ± 0.148 | 3.85489 ± 0.908 | 0.326748 ± 0.257 | 0.0672 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.206955 ± 0.0186 | 1 ± 0 | 0.933057 ± 0.0142 | 0.11 | u=0.02144 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.332667 ± 0.00469 | 2.65282 ± 0.121 | 0.924961 ± 0.00286 | 0.121 | u=0.02463 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.582466 ± 0.214 | 1.03907 ± 0.206 | 0.690778 ± 0.495 | 0.0945 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.295434 ± 0.0289 | 1 ± 0 | 0.933211 ± 0.015 | 0.0905 | u=0.02093 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Median seconds | lambda median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| WSINDy + MSTLS | 100 | 0.0% | 0.490536 ± 0.0788 | 1.04512 ± 0.139 | 1.05185 ± 0.318 | 0.062 | — |
| Filtered WSINDy + MSTLS | 100 | 0.0% | 0.593655 ± 0.138 | 4.02666 ± 1.14 | 0.413761 ± 0.302 | 0.0686 | — |
| WSINDy + LASSO | 100 | 0.0% | 0.225084 ± 0.0342 | 1 ± 0 | 0.925098 ± 0.0241 | 0.116 | u=0.03363 |
| Filtered WSINDy + LASSO | 100 | 0.0% | 0.332667 ± 0.00469 | 3.01326 ± 0.162 | 0.926116 ± 0.00395 | 0.117 | u=0.02642 |
| Debiased WSINDy (time2) + MSTLS | 100 | 0.0% | 0.50885 ± 0.2 | 1.08043 ± 0.211 | 0.822207 ± 0.437 | 0.0938 | — |
| Debiased WSINDy (time2) + LASSO | 100 | 0.0% | 0.299482 ± 0.0377 | 1 ± 0 | 0.928707 ± 0.0149 | 0.0894 | u=0.02878 |

