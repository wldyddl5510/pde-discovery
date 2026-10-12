# WSINDy benchmark

Methods are defined in `run.py` (METHODS). TPR = TP/(TP+FN+FP); Exact = exact-support rate; E_inf = max relative error over true terms; E2 = relative coefficient L2 error; mean ± sample SD over trials, failures included. Seconds cover estimation, weak system and regression. Mean lambda / r: mean selected LASSO penalty, or best-subset sparsity bound r, per equation.

## IB

`u_t = -0.5 (u^2)_x`  

43 dictionary terms; test functions `Tests(m=(60, 60), p=(7, 7), s=(5, 5), shape='poly', tau=1e-10, tau_hat=2.0)`.

### Noise ratio 0

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 100.0% | 1 ± 0 | 4.30846e-05 ± 0 | 4.30846e-05 ± 0 | 0.0627 | — |
| debiased | 100 | 0.0% | 0.166667 ± 5.58e-17 | 0.0122654 ± 0 | 27.4227 ± 3.57e-15 | 0.0634 | — |
| split-debiased | 100 | 0.0% | 0.2 ± 5.58e-17 | 0.0511407 ± 1.39e-17 | 0.686371 ± 1.12e-16 | 0.0707 | — |
| lasso | 100 | 100.0% | 1 ± 0 | 0.00183166 ± 0 | 0.00183166 ± 0 | 0.0674 | lambda_u=0.001789 |
| debiased-lasso | 100 | 0.0% | 0.5 ± 0 | 0.658965 ± 1.12e-16 | 638.54 ± 1.14e-13 | 0.0644 | lambda_u=0.0166 |
| split-debiased-lasso | 100 | 0.0% | 0.5 ± 0 | 0.656464 ± 1.12e-16 | 645.186 ± 0 | 0.0636 | lambda_u=0.01144 |

### Noise ratio 0.2

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 100.0% | 1 ± 0 | 0.00408295 ± 0.00318 | 0.00408295 ± 0.00318 | 0.0585 | — |
| debiased | 100 | 8.0% | 0.334262 ± 0.212 | 0.0584391 ± 0.0972 | 26.0369 ± 100 | 0.0616 | — |
| split-debiased | 100 | 13.0% | 0.376095 ± 0.253 | 0.0467351 ± 0.0256 | 13.2255 ± 13.7 | 0.073 | — |
| lasso | 100 | 25.0% | 0.625 ± 0.218 | 0.0778663 ± 0.0679 | 34.6959 ± 44.8 | 0.0691 | lambda_u=0.01324 |
| debiased-lasso | 100 | 0.0% | 0.5 ± 0 | 0.662164 ± 0.024 | 639.399 ± 22.5 | 0.0657 | lambda_u=0.01969 |
| split-debiased-lasso | 100 | 0.0% | 0.5 ± 0 | 0.663719 ± 0.0316 | 643.642 ± 31.1 | 0.0624 | lambda_u=0.02104 |

### Noise ratio 0.5

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 99.0% | 0.991111 ± 0.0889 | 0.0109775 ± 0.0094 | 0.940651 ± 9.3 | 0.0553 | — |
| debiased | 100 | 25.0% | 0.477833 ± 0.312 | 0.0510718 ± 0.0322 | 15.5908 ± 25 | 0.0615 | — |
| split-debiased | 100 | 12.0% | 0.381167 ± 0.247 | 0.0502907 ± 0.0319 | 17.1055 ± 18.6 | 0.0758 | — |
| lasso | 100 | 21.0% | 0.561667 ± 0.261 | 0.262172 ± 0.278 | 40544 ± 4.03e+05 | 0.0617 | lambda_u=0.02681 |
| debiased-lasso | 100 | 0.0% | 0.498333 ± 0.0167 | 0.684961 ± 0.0616 | 651.399 ± 55.5 | 0.0668 | lambda_u=0.0287 |
| split-debiased-lasso | 100 | 0.0% | 0.468333 ± 0.113 | 0.711842 ± 0.104 | 667.07 ± 82.4 | 0.0626 | lambda_u=0.03167 |

### Noise ratio 0.75

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 100.0% | 1 ± 0 | 0.0182255 ± 0.0129 | 0.0182255 ± 0.0129 | 0.0539 | — |
| debiased | 100 | 22.0% | 0.470667 ± 0.297 | 0.0540189 ± 0.0661 | 21.8176 ± 65.5 | 0.071 | — |
| split-debiased | 100 | 24.0% | 0.491667 ± 0.302 | 0.0545264 ± 0.0391 | 12.6103 ± 21.5 | 0.0711 | — |
| lasso | 100 | 21.0% | 0.540833 ± 0.272 | 0.349902 ± 0.268 | 61863.4 ± 6.16e+05 | 0.0568 | lambda_u=0.04173 |
| debiased-lasso | 100 | 0.0% | 0.473333 ± 0.102 | 0.695807 ± 0.101 | 648.556 ± 92.1 | 0.0662 | lambda_u=0.0326 |
| split-debiased-lasso | 100 | 0.0% | 0.475833 ± 0.0733 | 0.700693 ± 0.122 | 654.879 ± 121 | 0.066 | lambda_u=0.03895 |

### Noise ratio 1

| Method | Trials | Exact | TPR | E_inf | E2 | Mean seconds | Mean lambda / r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| wsindy | 100 | 99.0% | 0.995 ± 0.05 | 0.0214272 ± 0.0151 | 0.0219393 ± 0.016 | 0.0576 | — |
| debiased | 100 | 24.0% | 0.4955 ± 0.297 | 0.0596323 ± 0.0743 | 22.9425 ± 68.3 | 0.0678 | — |
| split-debiased | 100 | 24.0% | 0.49 ± 0.304 | 0.0614507 ± 0.103 | 22.3852 ± 94.1 | 0.0736 | — |
| lasso | 100 | 5.0% | 0.416667 ± 0.203 | 0.526774 ± 0.286 | 77815.3 ± 7.72e+05 | 0.0582 | lambda_u=0.04705 |
| debiased-lasso | 100 | 0.0% | 0.485 ± 0.0753 | 0.709272 ± 0.113 | 654.127 ± 113 | 0.0645 | lambda_u=0.0423 |
| split-debiased-lasso | 100 | 0.0% | 0.415 ± 0.17 | 0.751486 ± 0.166 | 1565.95 ± 4.8e+03 | 0.0721 | lambda_u=0.04624 |
