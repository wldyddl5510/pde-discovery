# Spatial dimension: 1D and 5D

Linear PDE: u_t = c u + Σ_i b_i ∂_i u + Σ_{i≤j} a_ij ∂_i∂_j u, on periodic [−π,π)^d.
c=−1, b₁=−1, a₁₁=1; every other coefficient is +0.001. There are 3 coefficients in 1D and 21 in 5D.
The exact Fourier solution varies in every spatial direction. Both cases use 32,768 spatial samples × 129 times and one paired noise seed.
Both cases use the same linear operator family; the earlier nonlinear 1D results are a separate experiment.
The 1D library is fully active: WSINDy's sparsity criterion selects only one term. Dimension and candidate count change together here.
W = WENDy; MLE = WENDy-MLE; HT keeps three terms; :0.0001 uses λ=0.0001λ_ref.
Budget: 1000 iterations / 5 seconds per fit.

## Noise 0%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| 1D · Top-3 hits | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1D · Strong E₂ | 0.837 | 3.33e-13 | 0.00068 | 3.33e-13 | 0.00068 |
| 1D · Full E₂ | 0.837 | 3.33e-13 | 0.00068 | 3.33e-13 | 0.00068 |
| 1D · Time (s) | 0.005561 | 0.004398 | 0.004634 | 0.004429 | 0.004709 |
| 1D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| 5D · Top-3 hits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5D · Strong E₂ | 0.00131 | 2.33e-13 | 0.000362 | 2.33e-13 | 0.000362 |
| 5D · Full E₂ | 0.00278 | 0.00245 | 0.000921 | 0.00245 | 0.000921 |
| 5D · Time (s) | 0.03931 | 0.03716 | 0.03873 | 0.03715 | 0.03864 |
| 5D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 1%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| 1D · Top-3 hits | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1D · Strong E₂ | 0.837 | 0.000121 | 0.000631 | 0.000121 | 0.000631 |
| 1D · Full E₂ | 0.837 | 0.000121 | 0.000631 | 0.000121 | 0.000631 |
| 1D · Time (s) | 0.005492 | 0.005474 | 0.005513 | 0.00591 | 0.008023 |
| 1D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| 5D · Top-3 hits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5D · Strong E₂ | 0.00136 | 0.000106 | 0.000359 | 0.000106 | 0.000359 |
| 5D · Full E₂ | 0.0028 | 0.00245 | 0.00102 | 0.00245 | 0.00102 |
| 5D · Time (s) | 0.03952 | 0.05119 | 0.05417 | 0.07654 | 0.2499 |
| 5D · Optimizer | 1/1 | 1/1 | 1/1 | 0/1 | 1/1 |

## Noise 5%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| 1D · Top-3 hits | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1D · Strong E₂ | 0.837 | 0.000605 | 0.000317 | 0.000605 | 0.000317 |
| 1D · Full E₂ | 0.837 | 0.000605 | 0.000317 | 0.000605 | 0.000317 |
| 1D · Time (s) | 0.005502 | 0.005086 | 0.005433 | 0.005591 | 0.01164 |
| 1D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| 5D · Top-3 hits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5D · Strong E₂ | 0.00164 | 0.00053 | 0.000571 | 0.000531 | 0.000571 |
| 5D · Full E₂ | 0.00295 | 0.00251 | 0.00193 | 0.00251 | 0.00193 |
| 5D · Time (s) | 0.04002 | 0.05215 | 0.05435 | 0.07635 | 0.1686 |
| 5D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 10%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| 1D · Top-3 hits | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1D · Strong E₂ | 0.837 | 0.00121 | 0.000621 | 0.00121 | 0.000621 |
| 1D · Full E₂ | 0.837 | 0.00121 | 0.000621 | 0.00121 | 0.000621 |
| 1D · Time (s) | 0.005471 | 0.005122 | 0.005486 | 0.005582 | 0.008585 |
| 1D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| 5D · Top-3 hits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5D · Strong E₂ | 0.00211 | 0.00106 | 0.000968 | 0.00106 | 0.000968 |
| 5D · Full E₂ | 0.00323 | 0.00267 | 0.0032 | 0.00267 | 0.0032 |
| 5D · Time (s) | 0.04059 | 0.05221 | 0.05439 | 0.07662 | 0.1868 |
| 5D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 20%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| 1D · Top-3 hits | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1D · Strong E₂ | 0.837 | 0.00242 | 0.00178 | 0.00242 | 0.00178 |
| 1D · Full E₂ | 0.837 | 0.00242 | 0.00178 | 0.00242 | 0.00178 |
| 1D · Time (s) | 0.005536 | 0.0051 | 0.005459 | 0.005552 | 0.008404 |
| 1D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| 5D · Top-3 hits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5D · Strong E₂ | 0.00321 | 0.00212 | 0.00219 | 0.00212 | 0.00219 |
| 5D · Full E₂ | 0.00403 | 0.00324 | 0.00663 | 0.00324 | 0.00663 |
| 5D · Time (s) | 0.0404 | 0.05239 | 0.05572 | 0.0767 | 0.1934 |
| 5D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 50%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| 1D · Top-3 hits | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1D · Strong E₂ | 0.837 | 0.00606 | 0.00539 | 0.00606 | 0.0054 |
| 1D · Full E₂ | 0.837 | 0.00606 | 0.00539 | 0.00606 | 0.0054 |
| 1D · Time (s) | 0.005432 | 0.005117 | 0.005529 | 0.005518 | 0.007725 |
| 1D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| 5D · Top-3 hits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5D · Strong E₂ | 0.0068 | 0.00529 | 0.00544 | 0.00529 | 0.00545 |
| 5D · Full E₂ | 0.00723 | 0.00583 | 0.0191 | 0.00583 | 0.0191 |
| 5D · Time (s) | 0.04002 | 0.05256 | 0.05607 | 0.07679 | 0.2114 |
| 5D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Noise 100%

| PDE / metric | WSINDy | W-HT | W:0.0001 | MLE-HT | MLE:0.0001 |
|---|---:|---:|---:|---:|---:|
| 1D · Top-3 hits | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1D · Strong E₂ | 0.837 | 0.0121 | 0.0115 | 0.0122 | 0.0115 |
| 1D · Full E₂ | 0.837 | 0.0121 | 0.0115 | 0.0122 | 0.0115 |
| 1D · Time (s) | 0.005424 | 0.005085 | 0.005466 | 0.005502 | 0.009403 |
| 1D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| 5D · Top-3 hits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 5D · Strong E₂ | 0.013 | 0.0105 | 0.0107 | 0.0106 | 0.0107 |
| 5D · Full E₂ | 0.0132 | 0.0108 | 0.0405 | 0.0108 | 0.0405 |
| 5D · Time (s) | 0.04018 | 0.05263 | 0.05681 | 0.07711 | 0.2909 |
| 5D · Optimizer | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

Top-3 hits counts dominant coordinates, not a success probability. Strong/Full E₂ are relative coefficient errors, not percentages.
Noise = σ/RMS(clean u), before spatial projection. Time includes Fourier tests, projection, weak assembly, and fitting; excludes data generation.
Optimizer counts fits meeting their stopping criterion; failures remain included. At zero noise, MLE uses the WENDy least-squares/LASSO fallback.
Projection averages physical noise: coefficient noise has standard deviation σ/√32768. Fixed total observations control this averaging across dimensions.
These timings measure the Fourier weak implementation; dense-grid PDE simulation is not timed.
Settings and numerical checks: [config.json](config.json). Raw fits: [trials.csv](trials.csv).
