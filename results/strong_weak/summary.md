# Strong signals with weak background coefficients

19 candidate terms: three coefficients of magnitude 1, sixteen of magnitude 0.001.
The leading PDE is u_t = −u − D_x(u²) + D_x²(u); all weak terms are included in the simulated PDE.
Library: D_x^d(u^j), d=0,1,2 and j=0,…,6, excluding spatial derivatives of constants.
Seeds: 1 (one at zero noise); HT keeps 3 terms. Budget: 1000 iterations / 5 seconds per fit.
W = WENDy; MLE = WENDy-MLE; W:α / MLE:α use L1 penalty λ=αλ_ref.
Top-3 hits counts correctly ranked dominant coordinates (out of three), not a probability over seeds.
E₂ is relative coefficient error, not a percentage. Optimizer counts fits meeting their stopping criterion; failed fits remain included.

## Noise 0%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| Strong/weak · Top-3 hits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Strong/weak · Strong E₂ | 0.000943 | 6.38e-06 | 0.00143 | 6.38e-06 | 0.00143 |
| Strong/weak · Full E₂ | 0.00249 | 0.00231 | 0.00264 | 0.00231 | 0.00264 |
| Strong/weak · Time (s) | 0.04299 | 0.04001 | 0.04044 | 0.04002 | 0.04028 |
| Strong/weak · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 1%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| Strong/weak · Top-3 hits | 3/3 | 0/3 | 3/3 | 0/3 | 3/3 |
| Strong/weak · Strong E₂ | 0.00567 | 1 | 0.00507 | 1 | 0.005 |
| Strong/weak · Full E₂ | 0.00612 | 2 | 0.00903 | 1.99 | 0.0091 |
| Strong/weak · Time (s) | 0.0435 | 0.1589 | 0.1634 | 0.1988 | 0.3221 |
| Strong/weak · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 5%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| Strong/weak · Top-3 hits | 3/3 | 0/3 | 3/3 | 0/3 | 3/3 |
| Strong/weak · Strong E₂ | 0.0232 | 1 | 0.00982 | 1 | 0.0103 |
| Strong/weak · Full E₂ | 0.0233 | 6.97 | 0.191 | 7.19 | 0.191 |
| Strong/weak · Time (s) | 0.04327 | 0.1583 | 0.1608 | 0.1962 | 0.4061 |
| Strong/weak · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 10%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| Strong/weak · Top-3 hits | 3/3 | 0/3 | 1/3 | 0/3 | 1/3 |
| Strong/weak · Strong E₂ | 0.0421 | 1 | 0.0462 | 1 | 0.0574 |
| Strong/weak · Full E₂ | 0.0422 | 13.4 | 1.94 | 14.6 | 1.97 |
| Strong/weak · Time (s) | 0.04343 | 0.1644 | 0.1654 | 0.1972 | 0.3969 |
| Strong/weak · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |

## Noise 20%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| Strong/weak · Top-3 hits | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Strong/weak · Strong E₂ | 0.0771 | 1 | 0.163 | 1 | 0.172 |
| Strong/weak · Full E₂ | 0.0771 | 36.5 | 6.21 | 36.4 | 6.76 |
| Strong/weak · Time (s) | 0.04331 | 0.1594 | 0.1714 | 0.2025 | 0.4167 |
| Strong/weak · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |

## Noise 50%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| Strong/weak · Top-3 hits | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Strong/weak · Strong E₂ | 0.271 | 1 | 1.66 | 1 | 0.933 |
| Strong/weak · Full E₂ | 0.271 | 62.1 | 28.9 | 33.4 | 21.1 |
| Strong/weak · Time (s) | 0.04336 | 0.1672 | 0.1702 | 0.2333 | 0.7476 |
| Strong/weak · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |

## Noise 100%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| Strong/weak · Top-3 hits | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Strong/weak · Strong E₂ | 0.826 | 1 | 4.22 | 1 | 1.49 |
| Strong/weak · Full E₂ | 3.25 | 87.9 | 37.1 | 42.4 | 33.7 |
| Strong/weak · Time (s) | 0.04348 | 0.1747 | 1.833 | 0.2365 | 1.072 |
| Strong/weak · Optimizer | 1/1 | 1/1 | 0/1 | 1/1 | 0/1 |

Perfect three-term truncation still gives Full E₂=0.00231 because the weak coefficients are nonzero.
At zero noise, MLE uses the WENDy least-squares/LASSO fallback.
Strong E₂ uses the three dominant coordinates; Full E₂ uses all 19 coefficients.
Noise = σ/RMS(clean u). Time includes weak-form assembly and fitting, excluding data generation and file writing.
Settings: [config.json](config.json). Raw fits: [trials.csv](trials.csv).
