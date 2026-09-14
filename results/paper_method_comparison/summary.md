# Results

Support and optimizer completion are counts; time is median seconds.
E₂ and E∞ are relative errors (mean over seeds), not percentages.
Failed fits are included; inf denotes a nonfinite error; — means unavailable. L1 labels give α in λ=αλ_ref.

## Burgers · 256×256 · 0% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 1/1 | 4.31e-05 | 4.31e-05 | 0.03201 | 1/1 |
| WENDy HT | 0/1 | 1 | 1 | 0.02594 | 1/1 |
| WENDy L1=1e-06 | 0/1 | 9.95e+13 | 1 | 4.115 | 0/1 |
| WENDy L1=0.0001 | 0/1 | 8.26e+15 | 20.4 | 0.3912 | 0/1 |
| WENDy L1=0.01 | 0/1 | 15.2 | 0.128 | 0.1178 | 1/1 |
| WENDy-MLE HT | 0/1 | 1 | 1 | 0.02602 | 1/1 |
| WENDy-MLE L1=1e-06 | 0/1 | 9.95e+13 | 1 | 4.063 | 0/1 |
| WENDy-MLE L1=0.0001 | 0/1 | 8.26e+15 | 20.4 | 0.3965 | 0/1 |
| WENDy-MLE L1=0.01 | 0/1 | 15.2 | 0.128 | 0.1182 | 1/1 |

## Burgers · 256×256 · 20% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 3/3 | 0.00484 | 0.00484 | 0.05212 | 3/3 |
| WENDy HT | 0/3 | 1 | 1 | 457.5 | 0/3 |
| WENDy L1=1e-06 | 0/3 | 9.28e+14 | 0.769 | 511.4 | 0/3 |
| WENDy L1=0.0001 | 0/3 | 2.42e+14 | 0.599 | 6.366 | 0/3 |
| WENDy L1=0.01 | 0/3 | 2e+11 | 0.342 | 139.1 | 3/3 |
| WENDy-MLE HT | 0/3 | 1 | 1 | 462.9 | 0/3 |
| WENDy-MLE L1=1e-06 | 0/3 | 1.03e+15 | 15.6 | 966.6 | 3/3 |
| WENDy-MLE L1=0.0001 | 0/3 | inf | inf | 139.9 | 0/3 |
| WENDy-MLE L1=0.01 | 0/3 | inf | inf | 979.7 | 1/3 |

## KdV · 400×601 · 0% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 1/1 | 2.84e-07 | 3.14e-07 | 0.1778 | 1/1 |
| WENDy HT | 1/1 | 3.56e-07 | 5.33e-07 | 0.1729 | 1/1 |
| WENDy L1=1e-06 | 0/1 | 0.000453 | 0.000216 | 0.1755 | 1/1 |
| WENDy L1=0.0001 | 0/1 | 0.247 | 0.022 | 0.178 | 1/1 |
| WENDy L1=0.01 | 0/1 | 14.5 | 1 | 0.1772 | 1/1 |
| WENDy-MLE HT | 1/1 | 3.56e-07 | 5.33e-07 | 0.1729 | 1/1 |
| WENDy-MLE L1=1e-06 | 0/1 | 0.000453 | 0.000216 | 0.1752 | 1/1 |
| WENDy-MLE L1=0.0001 | 0/1 | 0.247 | 0.022 | 0.1779 | 1/1 |
| WENDy-MLE L1=0.01 | 0/1 | 14.5 | 1 | 0.1772 | 1/1 |

## KdV · 400×601 · 20% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 2/3 | 0.303 | 0.338 | 0.1332 | 3/3 |
| WENDy HT | 0/3 | 1 | 1 | 486.9 | 0/3 |
| WENDy L1=1e-06 | 0/3 | 68.3 | 0.249 | 492.2 | 1/3 |
| WENDy L1=0.0001 | 0/3 | 201 | 0.358 | 3.614 | 0/3 |
| WENDy L1=0.01 | 0/3 | 527 | 1 | 48.4 | 3/3 |
| WENDy-MLE HT | 0/3 | 1 | 1 | 1415 | 0/3 |
| WENDy-MLE L1=1e-06 | 0/3 | 3.29e+03 | 1.14 | 1590 | 1/3 |
| WENDy-MLE L1=0.0001 | 0/3 | 4.07e+03 | 1.56 | 1970 | 0/3 |
| WENDy-MLE L1=0.01 | 0/3 | 547 | 1 | 1878 | 1/3 |

E₂ = ‖ŵ−w★‖₂/‖w★‖₂; E∞ = max relative error on true nonzero terms.
Noise = σ/RMS(clean u). Time includes weak-form assembly and fitting, excluding data generation and file writing.
Settings: [config.json](config.json). Raw fits: [trials.csv](trials.csv).
