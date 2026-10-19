# WSINDy benchmark

Methods are defined in `run.py` (METHODS). TPR = TP/(TP+FN+FP); Exact = exact-support rate; E_inf = max relative error over true terms; E2 = relative coefficient L2 error; mean ± sample SD over trials, failures included. Seconds cover estimation, weak system and regression. Mean lambda / r: mean selected LASSO penalty, or best-subset sparsity bound r, per equation.

## Seeds

Noise seeds (trials) per noise level; noise ratios 0, 0.1, 0.2, 0.5, 0.75, 1, 2. Trial i uses the same seed for every method (pdes.trial_seed, root seed 0), so methods are compared on identical observations. At noise ratio 0 all trials coincide.

| Method | IB |
| :--- | ---: |
| wsindy | 100 |
| debiased | 100 |
| debiased-lasso | 100 |
| debiased-best-subset | 100 |
| oracle-debiased | 100 |
| oracle-debiased-lasso | 100 |
| oracle-debiased-best-subset | 100 |

## Summary

Exact-support rate (%) · mean E_inf, per noise ratio.

| PDE | Method | 0 | 0.1 | 0.2 | 0.5 | 0.75 | 1 | 2 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IB | wsindy | 100 · 4.3e-05 | 100 · 0.0024 | 100 · 0.0041 | 99 · 0.011 | 100 · 0.018 | 99 · 0.021 | 90 · 0.084 |
| IB | debiased | 0 · 0.051 | 19 · 0.041 | 13 · 0.047 | 12 · 0.05 | 24 · 0.055 | 24 · 0.061 | 36 · 0.29 |
| IB | debiased-lasso | 0 · 0.66 | 0 · 0.66 | 0 · 0.66 | 0 · 0.71 | 0 · 0.7 | 0 · 0.75 | 0 · 0.77 |
| IB | debiased-best-subset | 0 · 1 | 0 · 1 | 0 · 1 | 0 · 1 | 7 · 0.93 | 13 · 0.88 | 17 · 0.85 |
| IB | oracle-debiased | 100 · 0.0036 | 64 · 0.0083 | 34 · 0.016 | 17 · 0.066 | 17 · 0.13 | 11 · 0.25 | 10 · 1 |
| IB | oracle-debiased-lasso | 100 · 0.0091 | 36 · 0.045 | 20 · 0.12 | 15 · 0.27 | 13 · 0.39 | 8 · 0.52 | 6 · 0.67 |
| IB | oracle-debiased-best-subset | 100 · 0.0036 | 100 · 0.0047 | 99 · 0.016 | 93 · 0.084 | 75 · 0.27 | 55 · 0.46 | 35 · 0.67 |

## IB

`u_t = -0.5 (u^2)_x`  

| Data U | X (K x J) | Y (K x E) | beta (J x E) | True nonzeros |
| :--- | :--- | :--- | :--- | :--- |
| 256 x 256 grid; components u | 784 x 43 | 784 x 1 | 43 x 1 | 1 of 43 (u_t: 1) |

K = weak equations (test-function centers), J = dictionary terms, E = equations. Test functions `Tests(m=(60, 60), p=(7, 7), s=(5, 5), shape='poly', tau=1e-10, tau_hat=2.0)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 100.0% | 1 ± 0 | 4.30846e-05 ± 0 | 4.30846e-05 ± 0 | 0.0252 | — |
| debiased | 100 | 0.0% | 0.2 ± 5.58e-17 | 0.0511407 ± 1.39e-17 | 0.686371 ± 1.12e-16 | 0.0338 | — |
| debiased-lasso | 100 | 0.0% | 0.5 ± 0 | 0.656464 ± 1.12e-16 | 645.186 ± 0 | 0.0672 | lambda_u=0.01144 |
| debiased-best-subset | 100 | 0.0% | 0 ± 0 | 1 ± 0 | 999.372 ± 2.29e-13 | 0.0723 | r_u=1 |
| oracle-debiased | 100 | 100.0% | 1 ± 0 | 0.00360447 ± 0 | 0.00360447 ± 0 | 0.0384 | — |
| oracle-debiased-lasso | 100 | 100.0% | 1 ± 0 | 0.00905958 ± 0 | 0.00905958 ± 0 | 0.0342 | lambda_u=0.01262 |
| oracle-debiased-best-subset | 100 | 100.0% | 1 ± 0 | 0.00360447 ± 0 | 0.00360447 ± 0 | 0.0537 | r_u=1 |

### Noise ratio 0.1

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 100.0% | 1 ± 0 | 0.00237355 ± 0.00169 | 0.00237355 ± 0.00169 | 0.0383 | — |
| debiased | 100 | 19.0% | 0.429667 ± 0.289 | 0.0409509 ± 0.0213 | 10.6022 ± 13.8 | 0.131 | — |
| debiased-lasso | 100 | 0.0% | 0.5 ± 0 | 0.660958 ± 0.0167 | 644.703 ± 16.7 | 0.0852 | lambda_u=0.01668 |
| debiased-best-subset | 100 | 0.0% | 0 ± 0 | 1 ± 0 | 999.624 ± 3.91 | 0.115 | r_u=1 |
| oracle-debiased | 100 | 64.0% | 0.779667 ± 0.303 | 0.00827178 ± 0.0134 | 5.06575 ± 14.4 | 0.0367 | — |
| oracle-debiased-lasso | 100 | 36.0% | 0.68 ± 0.241 | 0.0452314 ± 0.0336 | 22.7976 ± 33.1 | 0.0369 | lambda_u=0.01298 |
| oracle-debiased-best-subset | 100 | 100.0% | 1 ± 0 | 0.00474566 ± 0.00319 | 0.00474566 ± 0.00319 | 0.0568 | r_u=1 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 100.0% | 1 ± 0 | 0.00408295 ± 0.00318 | 0.00408295 ± 0.00318 | 0.0258 | — |
| debiased | 100 | 13.0% | 0.376095 ± 0.253 | 0.0467351 ± 0.0256 | 13.2255 ± 13.7 | 0.0345 | — |
| debiased-lasso | 100 | 0.0% | 0.5 ± 0 | 0.663719 ± 0.0316 | 643.642 ± 31.1 | 0.0647 | lambda_u=0.02104 |
| debiased-best-subset | 100 | 0.0% | 0 ± 0 | 1 ± 0 | 1000.3 ± 6.87 | 0.0744 | r_u=1 |
| oracle-debiased | 100 | 34.0% | 0.580262 ± 0.317 | 0.0159992 ± 0.0388 | 12.7409 ± 40.7 | 0.0379 | — |
| oracle-debiased-lasso | 100 | 20.0% | 0.59 ± 0.218 | 0.119793 ± 0.152 | 93.9214 ± 329 | 0.0368 | lambda_u=0.01767 |
| oracle-debiased-best-subset | 100 | 99.0% | 0.99 ± 0.1 | 0.0164289 ± 0.0995 | 0.0164289 ± 0.0995 | 0.0568 | r_u=1 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 99.0% | 0.991111 ± 0.0889 | 0.0109775 ± 0.0094 | 0.940651 ± 9.3 | 0.025 | — |
| debiased | 100 | 12.0% | 0.381167 ± 0.247 | 0.0502907 ± 0.0319 | 17.1055 ± 18.6 | 0.0337 | — |
| debiased-lasso | 100 | 0.0% | 0.468333 ± 0.113 | 0.711842 ± 0.104 | 667.07 ± 82.4 | 0.0648 | lambda_u=0.03167 |
| debiased-best-subset | 100 | 0.0% | 0 ± 0 | 1 ± 0 | 999.908 ± 17.6 | 0.0767 | r_u=1 |
| oracle-debiased | 100 | 17.0% | 0.436 ± 0.281 | 0.0662688 ± 0.172 | 55.7589 ± 174 | 0.0389 | — |
| oracle-debiased-lasso | 100 | 15.0% | 0.55 ± 0.215 | 0.265414 ± 0.228 | 659.555 ± 3.29e+03 | 0.0379 | lambda_u=0.03372 |
| oracle-debiased-best-subset | 100 | 93.0% | 0.93 ± 0.256 | 0.0838987 ± 0.253 | 70.307 ± 258 | 0.0576 | r_u=1 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 100.0% | 1 ± 0 | 0.0182255 ± 0.0129 | 0.0182255 ± 0.0129 | 0.0237 | — |
| debiased | 100 | 24.0% | 0.491667 ± 0.302 | 0.0545264 ± 0.0391 | 12.6103 ± 21.5 | 0.0361 | — |
| debiased-lasso | 100 | 0.0% | 0.475833 ± 0.0733 | 0.700693 ± 0.122 | 654.879 ± 121 | 0.0629 | lambda_u=0.03895 |
| debiased-best-subset | 100 | 7.0% | 0.07 ± 0.256 | 0.932 ± 0.249 | 927.799 ± 257 | 0.0774 | r_u=1 |
| oracle-debiased | 100 | 17.0% | 0.433357 ± 0.286 | 0.130358 ± 0.38 | 88.3877 ± 232 | 0.0372 | — |
| oracle-debiased-lasso | 100 | 13.0% | 0.515 ± 0.226 | 0.390966 ± 0.276 | 292.574 ± 245 | 0.0384 | lambda_u=0.04484 |
| oracle-debiased-best-subset | 100 | 75.0% | 0.75 ± 0.435 | 0.265586 ± 0.426 | 229.721 ± 423 | 0.058 | r_u=1 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 99.0% | 0.995 ± 0.05 | 0.0214272 ± 0.0151 | 0.0219393 ± 0.016 | 0.0248 | — |
| debiased | 100 | 24.0% | 0.49 ± 0.304 | 0.0614507 ± 0.103 | 22.3852 ± 94.1 | 0.0346 | — |
| debiased-lasso | 100 | 0.0% | 0.415 ± 0.17 | 0.751486 ± 0.166 | 1565.95 ± 4.8e+03 | 0.0639 | lambda_u=0.04624 |
| debiased-best-subset | 100 | 13.0% | 0.13 ± 0.338 | 0.875027 ± 0.325 | 866.306 ± 338 | 0.0763 | r_u=1 |
| oracle-debiased | 100 | 11.0% | 0.3715 ± 0.248 | 0.249237 ± 0.789 | 108.665 ± 282 | 0.0356 | — |
| oracle-debiased-lasso | 100 | 8.0% | 0.448333 ± 0.238 | 0.517432 ± 0.309 | 1601.41 ± 6.57e+03 | 0.0393 | lambda_u=0.05281 |
| oracle-debiased-best-subset | 100 | 55.0% | 0.55 ± 0.5 | 0.463746 ± 0.488 | 429.132 ± 497 | 0.0564 | r_u=1 |

### Noise ratio 2

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 90.0% | 0.943429 ± 0.176 | 0.0837299 ± 0.0567 | 0.65399 ± 3.78 | 0.0407 | — |
| debiased | 100 | 36.0% | 0.571468 ± 0.34 | 0.290534 ± 0.816 | 52.2848 ± 341 | 0.114 | — |
| debiased-lasso | 100 | 0.0% | 0.256167 ± 0.223 | 0.772015 ± 0.252 | 1.38598e+07 ± 1.31e+08 | 0.0895 | lambda_u=0.07536 |
| debiased-best-subset | 100 | 17.0% | 0.17 ± 0.378 | 0.846417 ± 0.342 | 807.452 ± 374 | 0.111 | r_u=1 |
| oracle-debiased | 100 | 10.0% | 0.341138 ± 0.247 | 1.00226 ± 2.18 | 197.382 ± 541 | 0.0366 | — |
| oracle-debiased-lasso | 100 | 6.0% | 0.295262 ± 0.278 | 0.669103 ± 0.323 | 808212 ± 4.53e+06 | 0.0402 | lambda_u=0.0777 |
| oracle-debiased-best-subset | 100 | 35.0% | 0.35 ± 0.479 | 0.674717 ± 0.446 | 621.703 ± 472 | 0.0515 | r_u=1 |
