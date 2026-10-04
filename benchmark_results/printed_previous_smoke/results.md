# WSINDy reproduction

Target: [Messenger & Bortz, arXiv:2007.02848v3](https://arxiv.org/html/2007.02848v3). Original `U_exact` arrays from the [authors' archive](https://zenodo.org/records/20787783), with MD5 verification. No resimulation, interpolation, or coarsening.

Schedule: 1 independent observation instances per noise level; root seed 0. Subset of the paper identification schedule.

Algorithm 4.2: separable FFT weak integrals, polynomial test functions, scaled MSTLS, 50 thresholds `logspace(-4,0,50)`, and physical-unit restoration after model selection. State scaling uses the printed exponent `1/beta_max`. See README for source discrepancies.

TPR is TP/(TP+FN+FP), E2 is relative coefficient L2 error, and E_inf is the maximum relative error over true nonzero coefficients. The table reports mean ± sample standard deviation when there are at least two trials, and the observed value for a single trial. For coupled systems these summarize the full coefficient matrix; per-equation values are in the JSONL file.

These results measure identification and coefficient accuracy as in arXiv v3. Solution-prediction metrics added in the later journal article are not evaluated. Original MATLAB random draws and runtime measurements are not reproduced.

## IB

Data shape: `(256, 256)`, components `('u',)`; G shape: `(784, 43)`. `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

| Noise ratio | Trials | TPR | E_inf | E2 | Selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 4.30846e-05 | 4.30846e-05 | 0.00910298 |
| 0.2 | 1 | 1 | 0.0016787 | 0.0016787 | 0.0719686 |

## KdV

Data shape: `(400, 601)`, components `('u',)`; G shape: `(1443, 43)`. `m=(45, 80)`, `s=(8, 12)`, `p=(8, 7)`.

| Noise ratio | Trials | TPR | E_inf | E2 | Selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 3.13515e-07 | 2.84321e-07 | 0.0001 |
| 0.2 | 1 | 0.5 | 1 | 0.894723 | 0.0868511 |

## KS

Data shape: `(256, 301)`, components `('u',)`; G shape: `(1806, 43)`. `m=(23, 22)`, `s=(5, 6)`, `p=(10, 10)`.

| Noise ratio | Trials | TPR | E_inf | E2 | Selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 8.18316e-07 | 5.54625e-07 | 0.0001 |
| 0.2 | 1 | 1 | 0.00400306 | 0.00342719 | 0.0868511 |

## NLS

Data shape: `(256, 251)`, components `('u', 'v')`; G shape: `(1804, 190)`. `m=(19, 25)`, `s=(5, 5)`, `p=(11, 10)`.

| Noise ratio | Trials | TPR | E_inf | E2 | Selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 9.28616e-08 | 5.60489e-08 | 0.0001 |
| 0.2 | 1 | 0.6 | 0.0467139 | 0.0312971 | 0.0494171 |

## SG

Data shape: `(129, 403, 205)`, components `('u',)`; G shape: `(13000, 73)`. `m=(40, 40, 25)`, `s=(5, 5, 8)`, `p=(8, 8, 10)`.

| Noise ratio | Trials | TPR | E_inf | E2 | Selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 4.36442e-05 | 2.74649e-05 | 0.0001 |
| 0.2 | 1 | 1 | 0.0382644 | 0.0221037 | 0.00355648 |

## RD

Data shape: `(256, 256, 201)`, components `('u', 'v')`; G shape: `(4860, 155)`. `m=(13, 13, 14)`, `s=(13, 13, 12)`, `p=(13, 13, 12)`.

| Noise ratio | Trials | TPR | E_inf | E2 | Selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 4.45443e-10 | 1.79098e-10 | 0.0001 |
| 0.2 | 1 | 0.777778 | 0.060895 | 0.0260446 | 0.0232995 |

## NS

Data shape: `(324, 149, 201)`, components `('omega', 'u', 'v')`; G shape: `(3872, 50)`. `m=(31, 31, 14)`, `s=(12, 12, 8)`, `p=(9, 9, 12)`.

| Noise ratio | Trials | TPR | E_inf | E2 | Selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 0.00537952 | 0.000789496 | 0.000449843 |
| 0.2 | 1 | 1 | 0.00376534 | 0.00210236 | 0.00517947 |
