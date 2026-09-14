# Results

Support and optimizer completion are counts; time is median seconds.
E₂ and E∞ are relative errors (mean over seeds), not percentages.
Failed fits are included; inf denotes a nonfinite error; — means unavailable. L1 labels give α in λ=αλ_ref.

## Burgers · 256×256 · 0% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 1/1 | 4.31e-05 | 4.31e-05 | 0.03524 | 1/1 |

## Burgers · 256×256 · 10% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 199/200 | 0.173 | 0.00227 | 0.03903 | 200/200 |

## Burgers · 256×256 · 20% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 200/200 | 0.00423 | 0.00423 | 0.03955 | 200/200 |

## Burgers · 256×256 · 50% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 200/200 | 0.0107 | 0.0107 | 0.04022 | 200/200 |

## Burgers · 256×256 · 100% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 197/200 | 0.0565 | 0.0235 | 0.04053 | 200/200 |

## KdV · 400×601 · 0% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 1/1 | 2.84e-07 | 3.14e-07 | 0.1125 | 1/1 |

## KdV · 400×601 · 10% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 198/200 | 0.0125 | 0.014 | 0.1322 | 200/200 |

## KdV · 400×601 · 20% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 198/200 | 0.0161 | 0.018 | 0.1329 | 200/200 |

## KdV · 400×601 · 50% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 188/200 | 0.19 | 0.0551 | 0.1338 | 200/200 |

## KdV · 400×601 · 100% noise

| Method | Support | E₂ | E∞ | Time (s) | Optimizer |
|---|---:|---:|---:|---:|---:|
| WSINDy | 192/200 | 0.107 | 0.0788 | 0.1344 | 200/200 |

E₂ = ‖ŵ−w★‖₂/‖w★‖₂; E∞ = max relative error on true nonzero terms.
Noise = σ/RMS(clean u). Time includes weak-form assembly and fitting, excluding data generation and file writing.
Settings: [config.json](config.json). Raw fits: [trials.csv](trials.csv).
