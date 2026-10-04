# WSINDy reproduction

Target: [Messenger & Bortz, arXiv:2007.02848v3](https://arxiv.org/html/2007.02848v3). Original `U_exact` arrays from the [authors' archive](https://zenodo.org/records/20787783), with MD5 verification. No resimulation, interpolation, or coarsening.

Status: complete; 2/2 completed trials. KS, NLS, RD are primary benchmarks; IB, KdV, NS, SG are supplementary.

Schedule: 2 independent observation instances per noise level; root seed 0. Subset of the paper identification schedule.

Author-code baseline: least squares on the scaled system, physical-unit MSTLS bounds, return before an empty support, and state scale exponent `1/(beta_max-1)`. Both use published supports/degrees/strides and 50 thresholds `logspace(-4,0,50)`. SVD least squares uses SciPy GELSD, not MATLAB backslash. RD uses the archived degree-5/order-4 library (181 columns, 4860 rows); the paper's contradictory RD row count is not reproduced. See README for source discrepancies.

TPR is TP/(TP+FN+FP), E2 is relative coefficient L2 error, and E_inf is the maximum relative error over true nonzero coefficients. The table reports mean ± sample standard deviation when there are at least two trials, and the observed value for a single trial. Exact is the percentage of trials with precisely the true support. For coupled systems metrics summarize the full coefficient matrix; per-equation values are in the JSONL files. Errors include failed identifications. Runtime covers weak-system construction, scaling, and all threshold refits, excluding disk loading, noise generation, and reporting; concurrent timings are not serial MATLAB timings.

These results measure identification and coefficient accuracy as in arXiv v3. Solution-prediction metrics added in the later journal article are not evaluated. Original MATLAB random draws and runtime measurements are not reproduced.

## IB

Data shape: `(256, 256)`, components `('u',)`; G shape: `(784, 43)`. `m=(60, 60)`, `s=(5, 5)`, `p=(7, 7)`.

| Noise ratio | Trials | Exact | TPR | E_inf | E2 | Median seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 2 | 100.0% | 1 ± 0 | 0.0027842 ± 0.00156 | 0.0027842 ± 0.00156 | 0.0175 |
