# PDE recovery

**Saved fits: 94/104. The smaller-λ sweep is incomplete; cells use saved fits only. — means unavailable.**

Burgers 256×256; KdV 400×601; 43 candidate terms. Planned repeats: 1 at 0% noise, 3 at 20% noise.
W = WENDy; MLE = WENDy-MLE. W:α / MLE:α use λ=αλ_ref; HT keeps the known number of terms.
Errors are mean relative errors, not %. Support / optimizer are counts; time is median seconds.

## Noise 0%

| PDE / metric | WSINDy | W-HT | W:1e-10 | W:1e-08 | W:1e-06 | W:0.0001 | W:0.01 | MLE-HT | MLE:1e-10 | MLE:1e-08 | MLE:1e-06 | MLE:0.0001 | MLE:0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Burgers · E∞ | 4.31e-05 | 1 | 44.9 | 42.4 | 1 | 20.4 | 0.128 | 1 | 44.9 | 42.4 | 1 | 20.4 | 0.128 |
| Burgers · E₂ | 4.31e-05 | 1 | 2.66e+13 | 1.53e+14 | 9.95e+13 | 8.26e+15 | 15.2 | 1 | 2.66e+13 | 1.53e+14 | 9.95e+13 | 8.26e+15 | 15.2 |
| Burgers · Support | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| Burgers · Time (s) | 0.03201 | 0.02594 | 0.02847 | 1.913 | 4.115 | 0.3912 | 0.1178 | 0.02602 | 0.02757 | 1.9 | 4.063 | 0.3965 | 0.1182 |
| Burgers · Optimizer | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/1 |
| KdV · E∞ | 3.14e-07 | 5.33e-07 | 5.33e-07 | 3.59e-06 | 0.000216 | 0.022 | 1 | 5.33e-07 | 5.33e-07 | 3.59e-06 | 0.000216 | 0.022 | 1 |
| KdV · E₂ | 2.84e-07 | 3.56e-07 | 4.43e-05 | 0.000502 | 0.000453 | 0.247 | 14.5 | 3.56e-07 | 4.43e-05 | 0.000502 | 0.000453 | 0.247 | 14.5 |
| KdV · Support | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| KdV · Time (s) | 0.1778 | 0.1729 | 0.1392 | 0.2015 | 0.1755 | 0.178 | 0.1772 | 0.1729 | 0.1391 | 0.1988 | 0.1752 | 0.1779 | 0.1772 |
| KdV · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 20%

| PDE / metric | WSINDy | W-HT | W:1e-10 | W:1e-08 | W:1e-06 | W:0.0001 | W:0.01 | MLE-HT | MLE:1e-10 | MLE:1e-08 | MLE:1e-06 | MLE:0.0001 | MLE:0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Burgers · E∞ | 0.00484 | 1 | 30.1 | 19.2 | 0.769 | 0.599 | 0.342 | 1 | 16.7 | 17.1 | 15.6 | inf | inf |
| Burgers · E₂ | 0.00484 | 1 | 8.15e+14 | 9.17e+14 | 9.28e+14 | 2.42e+14 | 2e+11 | 1 | 9.38e+14 | 1.05e+15 | 1.03e+15 | inf | inf |
| Burgers · Support | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Burgers · Time (s) | 0.05212 | 457.5 | 4.549 | 13.19 | 511.4 | 6.366 | 139.1 | 462.9 | 968.2 | 537.4 | 966.6 | 139.9 | 979.7 |
| Burgers · Optimizer | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 3/3 | 0/3 | 0/3 | 3/3 | 3/3 | 0/3 | 1/3 |
| KdV · E∞ | 0.338 | 1 | 1.94 | 1.93 | 0.249 | 0.358 | 1 | 1 | — | — | 1.14 | 1.56 | 1 |
| KdV · E₂ | 0.303 | 1 | 6.29e+03 | 6.41e+03 | 68.3 | 201 | 527 | 1 | — | — | 3.29e+03 | 4.07e+03 | 547 |
| KdV · Support | 2/3 | 0/3 | 0/1 | 0/1 | 0/3 | 0/3 | 0/3 | 0/3 | — | — | 0/3 | 0/3 | 0/3 |
| KdV · Time (s) | 0.1332 | 486.9 | 6.26 | 6.264 | 492.2 | 3.614 | 48.4 | 1415 | — | — | 1590 | 1970 | 1878 |
| KdV · Optimizer | 3/3 | 0/3 | 0/1 | 0/1 | 1/3 | 0/3 | 3/3 | 0/3 | — | — | 1/3 | 0/3 | 1/3 |

E₂ = ‖ŵ−w★‖₂/‖w★‖₂; E∞ = max relative error on true nonzero terms.
Failed fits remain included; inf = nonfinite error. Optimizer completion does not imply recovery.
[WSINDy Table 5](https://arxiv.org/html/2007.02848v3#S5.T5): noiseless E∞ = 4.3e-5 (Burgers), 3.1e-7 (KdV).

![WSINDy coefficient errors versus noise](coefficient_error.png)

WSINDy noise sweep: mean errors over 200 seeds per positive noise level, following Figure 6.
WENDy variants were tested only at 0% and 20%; the figure shows our WSINDy measurements.

Same PDE extensions and settings: known σ, maxiter=300, tol=1e-8; full covariance.
Runtime: Apple M4 / Python, FFT/covariance up to 4 threads, BLAS 1; includes failed fits, excludes data generation and file writing.
Rescaling follows author code (−1/5), differing from printed Eq. 4.8 (−1/6).

Data: [method comparison](paper_method_comparison/summary.md) ·
[WSINDy sweep](wsindy_paper_reference/summary.md) · [legacy baseline](selection_comparison/summary.md).
