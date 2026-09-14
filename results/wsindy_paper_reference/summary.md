# Results

W = WENDy; MLE = WENDy-MLE; W:α / MLE:α use L1 penalty λ=αλ_ref.
Support and optimizer completion are counts; time is median seconds.
E₂ and E∞ are relative errors (mean over seeds), not percentages.
Failed fits are included; inf denotes a nonfinite error; — means unavailable.

## Noise 0%

| PDE / metric | WSINDy |
|---|---:|
| Burgers · E∞ | 4.31e-05 |
| Burgers · E₂ | 4.31e-05 |
| Burgers · Support | 1/1 |
| Burgers · Time (s) | 0.03524 |
| Burgers · Optimizer | 1/1 |
| KdV · E∞ | 3.14e-07 |
| KdV · E₂ | 2.84e-07 |
| KdV · Support | 1/1 |
| KdV · Time (s) | 0.1125 |
| KdV · Optimizer | 1/1 |

## Noise 10%

| PDE / metric | WSINDy |
|---|---:|
| Burgers · E∞ | 0.00227 |
| Burgers · E₂ | 0.173 |
| Burgers · Support | 199/200 |
| Burgers · Time (s) | 0.03903 |
| Burgers · Optimizer | 200/200 |
| KdV · E∞ | 0.014 |
| KdV · E₂ | 0.0125 |
| KdV · Support | 198/200 |
| KdV · Time (s) | 0.1322 |
| KdV · Optimizer | 200/200 |

## Noise 20%

| PDE / metric | WSINDy |
|---|---:|
| Burgers · E∞ | 0.00423 |
| Burgers · E₂ | 0.00423 |
| Burgers · Support | 200/200 |
| Burgers · Time (s) | 0.03955 |
| Burgers · Optimizer | 200/200 |
| KdV · E∞ | 0.018 |
| KdV · E₂ | 0.0161 |
| KdV · Support | 198/200 |
| KdV · Time (s) | 0.1329 |
| KdV · Optimizer | 200/200 |

## Noise 50%

| PDE / metric | WSINDy |
|---|---:|
| Burgers · E∞ | 0.0107 |
| Burgers · E₂ | 0.0107 |
| Burgers · Support | 200/200 |
| Burgers · Time (s) | 0.04022 |
| Burgers · Optimizer | 200/200 |
| KdV · E∞ | 0.0551 |
| KdV · E₂ | 0.19 |
| KdV · Support | 188/200 |
| KdV · Time (s) | 0.1338 |
| KdV · Optimizer | 200/200 |

## Noise 100%

| PDE / metric | WSINDy |
|---|---:|
| Burgers · E∞ | 0.0235 |
| Burgers · E₂ | 0.0565 |
| Burgers · Support | 197/200 |
| Burgers · Time (s) | 0.04053 |
| Burgers · Optimizer | 200/200 |
| KdV · E∞ | 0.0788 |
| KdV · E₂ | 0.107 |
| KdV · Support | 192/200 |
| KdV · Time (s) | 0.1344 |
| KdV · Optimizer | 200/200 |

E₂ = ‖ŵ−w★‖₂/‖w★‖₂; E∞ = max relative error on true nonzero terms.
Noise = σ/RMS(clean u). Time includes weak-form assembly and fitting, excluding data generation and file writing.
Settings: [config.json](config.json). Raw fits: [trials.csv](trials.csv).
