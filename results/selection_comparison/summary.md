# Results

Support and optimizer completion are counts; time is median seconds.
E₂ and E∞ are relative errors (median over seeds), not percentages.
Failed fits are included; inf denotes a nonfinite error; — means unavailable. L1 labels give α in λ=αλ_ref.

## Burgers · 256 · 0% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 1/1 | 4.86e-05 | — | 0.1846 | 1/1 |
| WENDy HT | 1/1 | 0.00981 | — | 0.1817 | 1/1 |
| WENDy L1=1e-06 | 0/1 | 0.00862 | — | 0.2045 | 1/1 |
| WENDy L1=0.0001 | 0/1 | 0.000626 | — | 0.1837 | 1/1 |
| WENDy L1=0.01 | 0/1 | 0.0182 | — | 0.1824 | 1/1 |
| WENDy-MLE HT | 1/1 | 0.00981 | — | 0.1816 | 1/1 |
| WENDy-MLE L1=1e-06 | 0/1 | 0.00862 | — | 0.2039 | 1/1 |
| WENDy-MLE L1=0.0001 | 0/1 | 0.000626 | — | 0.1837 | 1/1 |
| WENDy-MLE L1=0.01 | 0/1 | 0.0182 | — | 0.1824 | 1/1 |

## Burgers · 256 · 1% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 10/10 | 0.000213 | — | 0.1823 | 10/10 |
| WENDy HT | 10/10 | 0.0128 | — | 1.128 | 10/10 |
| WENDy L1=1e-06 | 0/10 | 0.0127 | — | 1.165 | 9/10 |
| WENDy L1=0.0001 | 0/10 | 0.00275 | — | 1.136 | 10/10 |
| WENDy L1=0.01 | 0/10 | 0.0276 | — | 1.134 | 10/10 |
| WENDy-MLE HT | 10/10 | 0.0137 | — | 1.16 | 9/10 |
| WENDy-MLE L1=1e-06 | 0/10 | 0.232 | — | 1.314 | 10/10 |
| WENDy-MLE L1=0.0001 | 0/10 | 0.0632 | — | 1.257 | 10/10 |
| WENDy-MLE L1=0.01 | 0/10 | 0.0345 | — | 1.211 | 10/10 |

## Burgers · 256 · 10% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 10/10 | 0.00212 | — | 0.1826 | 10/10 |
| WENDy HT | 7/10 | 0.27 | — | 1.149 | 10/10 |
| WENDy L1=1e-06 | 0/10 | 0.219 | — | 1.22 | 9/10 |
| WENDy L1=0.0001 | 0/10 | 0.0395 | — | 1.175 | 10/10 |
| WENDy L1=0.01 | 0/10 | 0.502 | — | 1.157 | 9/10 |
| WENDy-MLE HT | 7/10 | 0.22 | — | 1.199 | 7/10 |
| WENDy-MLE L1=1e-06 | 0/10 | 0.64 | — | 1.377 | 10/10 |
| WENDy-MLE L1=0.0001 | 0/10 | 0.866 | — | 1.295 | 10/10 |
| WENDy-MLE L1=0.01 | 0/10 | 0.549 | — | 1.213 | 10/10 |

## Burgers · 256 · 50% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 10/10 | 0.00831 | — | 0.1823 | 10/10 |
| WENDy HT | 6/10 | 0.533 | — | 1.159 | 10/10 |
| WENDy L1=1e-06 | 0/10 | 0.98 | — | 1.221 | 10/10 |
| WENDy L1=0.0001 | 0/10 | 0.362 | — | 1.213 | 9/10 |
| WENDy L1=0.01 | 0/10 | 1.23 | — | 1.148 | 10/10 |
| WENDy-MLE HT | 8/10 | 0.238 | — | 1.198 | 6/10 |
| WENDy-MLE L1=1e-06 | 0/10 | 0.752 | — | 1.44 | 10/10 |
| WENDy-MLE L1=0.0001 | 0/10 | 1.12 | — | 1.355 | 10/10 |
| WENDy-MLE L1=0.01 | 0/10 | 1.23 | — | 1.267 | 10/10 |

## Burgers · 256 · 100% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 4/10 | 1.39 | — | 0.1828 | 10/10 |
| WENDy HT | 4/10 | 1.16 | — | 1.164 | 10/10 |
| WENDy L1=1e-06 | 0/10 | 0.914 | — | 1.201 | 10/10 |
| WENDy L1=0.0001 | 0/10 | 0.698 | — | 1.185 | 10/10 |
| WENDy L1=0.01 | 0/10 | 1.08 | — | 1.147 | 10/10 |
| WENDy-MLE HT | 3/10 | 1.14 | — | 1.21 | 7/10 |
| WENDy-MLE L1=1e-06 | 0/10 | 0.868 | — | 1.364 | 10/10 |
| WENDy-MLE L1=0.0001 | 0/10 | 0.907 | — | 1.422 | 10/10 |
| WENDy-MLE L1=0.01 | 0/10 | 1.07 | — | 1.254 | 10/10 |

## KdV · 256 · 0% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 1/1 | 1.18e-08 | — | 0.2606 | 1/1 |
| WENDy HT | 1/1 | 3.26e-07 | — | 0.2586 | 1/1 |
| WENDy L1=1e-06 | 0/1 | 0.00225 | — | 0.2611 | 1/1 |
| WENDy L1=0.0001 | 0/1 | 0.225 | — | 0.261 | 1/1 |
| WENDy L1=0.01 | 0/1 | 1 | — | 0.2602 | 1/1 |
| WENDy-MLE HT | 1/1 | 3.26e-07 | — | 0.2586 | 1/1 |
| WENDy-MLE L1=1e-06 | 0/1 | 0.00225 | — | 0.261 | 1/1 |
| WENDy-MLE L1=0.0001 | 0/1 | 0.225 | — | 0.2609 | 1/1 |
| WENDy-MLE L1=0.01 | 0/1 | 1 | — | 0.2601 | 1/1 |

## KdV · 256 · 1% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 10/10 | 0.000217 | — | 0.2587 | 10/10 |
| WENDy HT | 10/10 | 0.00194 | — | 2.282 | 10/10 |
| WENDy L1=1e-06 | 0/10 | 0.0035 | — | 2.278 | 10/10 |
| WENDy L1=0.0001 | 0/10 | 0.288 | — | 2.303 | 10/10 |
| WENDy L1=0.01 | 0/10 | 1 | — | 2.278 | 10/10 |
| WENDy-MLE HT | 10/10 | 0.00191 | — | 2.323 | 0/10 |
| WENDy-MLE L1=1e-06 | 0/10 | 0.00248 | — | 2.484 | 10/10 |
| WENDy-MLE L1=0.0001 | 0/10 | 0.276 | — | 2.683 | 10/10 |
| WENDy-MLE L1=0.01 | 0/10 | 1 | — | 2.511 | 10/10 |

## KdV · 256 · 10% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 10/10 | 0.00196 | — | 0.2584 | 10/10 |
| WENDy HT | 10/10 | 0.0148 | — | 2.151 | 10/10 |
| WENDy L1=1e-06 | 0/10 | 0.0193 | — | 2.196 | 10/10 |
| WENDy L1=0.0001 | 0/10 | 0.295 | — | 2.158 | 10/10 |
| WENDy L1=0.01 | 0/10 | 1 | — | 2.144 | 10/10 |
| WENDy-MLE HT | 10/10 | 0.0154 | — | 2.205 | 9/10 |
| WENDy-MLE L1=1e-06 | 0/10 | 0.0304 | — | 2.543 | 10/10 |
| WENDy-MLE L1=0.0001 | 0/10 | 0.277 | — | 2.645 | 10/10 |
| WENDy-MLE L1=0.01 | 0/10 | 1 | — | 2.377 | 10/10 |

## KdV · 256 · 50% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 10/10 | 0.0167 | — | 0.2586 | 10/10 |
| WENDy HT | 10/10 | 0.236 | — | 2.125 | 10/10 |
| WENDy L1=1e-06 | 0/10 | 0.284 | — | 2.189 | 10/10 |
| WENDy L1=0.0001 | 0/10 | 0.466 | — | 2.164 | 10/10 |
| WENDy L1=0.01 | 0/10 | 1 | — | 2.127 | 10/10 |
| WENDy-MLE HT | 10/10 | 0.253 | — | 2.193 | 3/10 |
| WENDy-MLE L1=1e-06 | 0/10 | 0.32 | — | 3.2 | 10/10 |
| WENDy-MLE L1=0.0001 | 0/10 | 0.418 | — | 2.635 | 10/10 |
| WENDy-MLE L1=0.01 | 0/10 | 1 | — | 2.326 | 10/10 |

## KdV · 256 · 100% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 10/10 | 0.07 | — | 0.2588 | 10/10 |
| WENDy HT | 8/10 | 0.408 | — | 2.183 | 10/10 |
| WENDy L1=1e-06 | 0/10 | 0.529 | — | 2.339 | 8/10 |
| WENDy L1=0.0001 | 0/10 | 0.754 | — | 2.249 | 10/10 |
| WENDy L1=0.01 | 0/10 | 1 | — | 2.194 | 10/10 |
| WENDy-MLE HT | 8/10 | 0.443 | — | 2.228 | 9/10 |
| WENDy-MLE L1=1e-06 | 0/10 | 0.525 | — | 3.168 | 10/10 |
| WENDy-MLE L1=0.0001 | 0/10 | 0.777 | — | 2.91 | 10/10 |
| WENDy-MLE L1=0.01 | 0/10 | 1 | — | 2.354 | 10/10 |

E₂ = ‖ŵ−w★‖₂/‖w★‖₂; E∞ = max relative error on true nonzero terms.
Noise = σ/RMS(clean u). Time includes weak-form assembly and fitting, excluding data generation and file writing.
Settings: [config.json](config.json). Raw fits: [trials.csv](trials.csv).
