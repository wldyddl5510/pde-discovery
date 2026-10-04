# WSINDy reproduction

Target: [Messenger & Bortz, arXiv:2007.02848v3](https://arxiv.org/html/2007.02848v3). Original `U_exact` arrays from the [authors' archive](https://zenodo.org/records/20787783), with MD5 verification. No resimulation, interpolation, or coarsening.

Status: complete; 28/28 completed trials. KS, NLS, RD are primary benchmarks; IB, KdV, NS, SG are supplementary.

Schedule: 1 independent observation instances per noise level; root seed 0. Subset of the paper identification schedule.

Author-code baseline: least squares on the scaled system, physical-unit MSTLS bounds, return before an empty support, and state scale exponent `1/(beta_max-1)`. Both use published supports/degrees/strides and 50 thresholds `logspace(-4,0,50)`. SVD least squares uses SciPy GELSD, not MATLAB backslash. RD uses the archived degree-5/order-4 library (181 columns, 4860 rows); the paper's contradictory RD row count is not reproduced. See README for source discrepancies.

TPR is TP/(TP+FN+FP), E2 is relative coefficient L2 error, and E_inf is the maximum relative error over true nonzero coefficients. The table reports mean ± sample standard deviation when there are at least two trials, and the observed value for a single trial. Exact is the percentage of trials with precisely the true support. For coupled systems metrics summarize the full coefficient matrix; per-equation values are in the JSONL files. Errors include failed identifications. Runtime covers weak-system construction, scaling, and all threshold refits, excluding disk loading, noise generation, and reporting; concurrent timings are not serial MATLAB timings.

These results measure identification and coefficient accuracy as in arXiv v3. Solution-prediction metrics added in the later journal article are not evaluated. Original MATLAB random draws and runtime measurements are not reproduced.

## KS

Data shape: `(256, 301)`, components `('u',)`; G shape: `(1806, 43)`. `m=(23, 22)`, `s=(5, 6)`, `p=(10, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 8.18316e-07 | 5.54625e-07 | 0.0175 |
| 0.2 | 1 | 100.0% | 1 | 0.00400306 | 0.00342719 | 0.0356 |
| 0.5 | 1 | 100.0% | 1 | 0.0740804 | 0.0667874 | 0.0377 |
| 1 | 1 | 100.0% | 1 | 0.258303 | 0.257077 | 0.0399 |

## NLS

Data shape: `(256, 251)`, components `('u', 'v')`; G shape: `(1804, 190)`. `m=(19, 25)`, `s=(5, 5)`, `p=(11, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 9.28616e-08 | 5.60489e-08 | 0.0728 |
| 0.2 | 1 | 100.0% | 1 | 0.0290038 | 0.0183071 | 0.466 |
| 0.5 | 1 | 100.0% | 1 | 0.158112 | 0.109597 | 0.661 |
| 1 | 1 | 0.0% | 0.3 | 0.508347 | 1.1634 | 0.643 |

## RD

Data shape: `(256, 256, 201)`, components `('u', 'v')`; G shape: `(4860, 181)`. `m=(13, 13, 14)`, `s=(13, 13, 12)`, `p=(13, 13, 12)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 4.45443e-10 | 1.79099e-10 | 10.1 |
| 0.2 | 1 | 100.0% | 1 | 0.0387101 | 0.0268002 | 10.3 |
| 0.5 | 1 | 0.0% | 0.875 | 0.123843 | 0.0940302 | 10.1 |
| 1 | 1 | 0.0% | 0.666667 | 0.673452 | 0.534833 | 10.3 |

## IB

Data shape: `(256, 256)`, components `('u',)`; G shape: `(784, 43)`. `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 4.30846e-05 | 4.30846e-05 | 0.0178 |
| 0.2 | 1 | 100.0% | 1 | 0.0016787 | 0.0016787 | 0.0192 |
| 0.5 | 1 | 100.0% | 1 | 0.00386154 | 0.00386154 | 0.0192 |
| 1 | 1 | 100.0% | 1 | 0.0102051 | 0.0102051 | 0.0189 |

## KdV

Data shape: `(400, 601)`, components `('u',)`; G shape: `(1443, 43)`. `m=(45, 80)`, `s=(8, 12)`, `p=(8, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 3.13515e-07 | 2.84321e-07 | 0.052 |
| 0.2 | 1 | 100.0% | 1 | 0.00256303 | 0.00229297 | 0.0558 |
| 0.5 | 1 | 0.0% | 0.5 | 1 | 0.894714 | 0.0552 |
| 1 | 1 | 100.0% | 1 | 0.00346119 | 0.00313621 | 0.0556 |

## NS

Data shape: `(324, 149, 201)`, components `('omega', 'u', 'v')`; G shape: `(3872, 50)`. `m=(31, 31, 14)`, `s=(12, 12, 8)`, `p=(9, 9, 12)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 0.00537952 | 0.000789496 | 3.04 |
| 0.2 | 1 | 100.0% | 1 | 0.00376534 | 0.00210236 | 3.09 |
| 0.5 | 1 | 0.0% | 0.4 | 1 | 0.0689292 | 2.9 |
| 1 | 1 | 0.0% | 0.5 | 1 | 0.0467674 | 3 |

## SG

Data shape: `(129, 403, 205)`, components `('u',)`; G shape: `(13000, 73)`. `m=(40, 40, 25)`, `s=(5, 5, 8)`, `p=(8, 8, 10)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 100.0% | 1 | 4.36442e-05 | 2.74649e-05 | 3.26 |
| 0.2 | 1 | 100.0% | 1 | 0.0382644 | 0.0221037 | 3.59 |
| 0.5 | 1 | 100.0% | 1 | 0.211923 | 0.122974 | 3.78 |
| 1 | 1 | 100.0% | 1 | 1.02757 | 0.594598 | 3.47 |
