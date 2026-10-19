# True-support WSINDy versus EIV

Same observations and true support for IB, KdV and KS; 100 paired trials per noise level.

- **WSINDy:** ordinary least squares on the original, uncorrected weak terms, with the existing WSINDy scaling and physical coefficient restoration. The true support is fixed, so MSTLS selection is bypassed.
- **EIV:** saved true-support conic results with the same noise, oracle variance, saved tau, mu=0, column normalization, and the original lambda grid. Select minimum coefficient E2 among validated candidates.
- The earlier study's OLS column used Hermite-corrected terms. Here the original WSINDy terms are independently recomputed. Their true terms are linear or derivatives of u^2; the constant Gaussian correction to u^2 vanishes under the spatial derivative, up to quadrature/roundoff.
- Noise-zero trials repeat identical observations; counts are not independent clean datasets.

| Task | Noise | Mean E2: WSINDy | Mean E2: EIV | EIV / WSINDy | Median E2: WSINDy | Median E2: EIV |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| IB | 0 | 4.30846e-05 | 0.000196035 | 4.55 | 4.30846e-05 | 0.000196035 |
| IB | 0.1 | 0.00237355 | 0.00606226 | 2.55 | 0.00219142 | 0.00331428 |
| IB | 0.2 | 0.00408295 | 0.0117743 | 2.88 | 0.00342493 | 0.00398292 |
| IB | 0.5 | 0.0106589 | 0.0381423 | 3.58 | 0.00879773 | 0.0282741 |
| IB | 0.75 | 0.0182255 | 0.0649556 | 3.56 | 0.0152297 | 0.0584428 |
| IB | 1 | 0.0217288 | 0.0848365 | 3.9 | 0.0180717 | 0.0729033 |
| IB | 2 | 0.0816916 | 0.240088 | 2.94 | 0.0692967 | 0.241026 |
| KdV | 0 | 2.84321e-07 | 1.74526e-06 | 6.14 | 2.84321e-07 | 1.74526e-06 |
| KdV | 0.1 | 0.00398894 | 0.0316287 | 7.93 | 0.0032587 | 0.0341884 |
| KdV | 0.2 | 0.00700317 | 0.0609816 | 8.71 | 0.00587159 | 0.0538271 |
| KdV | 0.5 | 0.0191497 | 0.140676 | 7.35 | 0.0143916 | 0.136714 |
| KdV | 0.75 | 0.0290362 | 0.211418 | 7.28 | 0.0240972 | 0.185302 |
| KdV | 1 | 0.0377484 | 0.286456 | 7.59 | 0.026666 | 0.27458 |
| KdV | 2 | 0.118143 | 0.615925 | 5.21 | 0.104439 | 0.650505 |
| KS | 0 | 5.54625e-07 | 8.68648e-06 | 15.7 | 5.54625e-07 | 8.68648e-06 |
| KS | 0.1 | 0.00406559 | 0.0474149 | 11.7 | 0.00323557 | 0.0474913 |
| KS | 0.2 | 0.0104377 | 0.0866547 | 8.3 | 0.00938375 | 0.074581 |
| KS | 0.5 | 0.0670009 | 0.264646 | 3.95 | 0.0675746 | 0.245188 |
| KS | 0.75 | 0.156629 | 0.459717 | 2.94 | 0.151947 | 0.478716 |
| KS | 1 | 0.272792 | 0.702803 | 2.58 | 0.269418 | 0.726643 |
| KS | 2 | 0.751704 | 0.999307 | 1.33 | 0.752952 | 1 |

## Verification

- 2100 distinct paired trials; all saved EIV choices passed the prior convergence/constraint checks.
- Maximum absolute coefficient difference between original WSINDy and the earlier corrected OLS control: 2.29e-14.
- Maximum relative coefficient difference (normalized by the true coefficient norm): 2.05e-14.

```sh
python benchmark_results/eiv/true_support/compare_wsindy.py --workers 4
```
