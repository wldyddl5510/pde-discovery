# Constructed oracle-pilot error budgets

Prepared 104/104 paired observations; 208 oracle pilots. These values summarize the actual floating-point states.

E is the squared vector error integrated with physical Delta z and the split time weights. The rate target is n_reg^(-1/2); E/n_reg^(-1/4) <= n_reg^(-1/4) tends to zero. The exact oracle has E=0. Raw Delta z sums are also recorded in CSV/JSON.

| PDE | Oracle | N | n_reg | Delta z | Target E | Alpha range | Maximum E | Max E/target | Max E/n_reg^(-1/4) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| IB | rate | 13 | 32768 | 0.492118 | 0.00552427 | 2.23607e-06 - 2.28145e-06 | 0.00552427 | 1 | 0.0743254 |
| IB | exact | 13 | 32768 | 0.492118 | 0.00552427 | 0 - 0 | 0 | 0 | 0 |
| KdV | rate | 13 | 120400 | 1.5708e-07 | 0.00288195 | 0.00128507 - 0.0485479 | 0.00288195 | 1 | 0.0536838 |
| KdV | exact | 13 | 120400 | 1.5708e-07 | 0.00288195 | 0 - 0 | 0 | 0 | 0 |
| KS | rate | 13 | 38656 | 0.19635 | 0.00508618 | 0.000555585 - 0.0105644 | 0.00508618 | 1 | 0.0713174 |
| KS | exact | 13 | 38656 | 0.19635 | 0.00508618 | 0 - 0 | 0 | 0 | 0 |
| NLS | rate | 13 | 32256 | 0.000490874 | 0.00556794 | 0.0729181 - 1 | 0.00556794 | 1 | 0.0746186 |
| NLS | exact | 13 | 32256 | 0.000490874 | 0.00556794 | 0 - 0 | 0 | 0 | 0 |
| RD | rate | 13 | 6619136 | 0.000305176 | 0.000388686 | 0.00266657 - 0.0273215 | 0.000388686 | 1 | 0.0197151 |
| RD | exact | 13 | 6619136 | 0.000305176 | 0.000388686 | 0 - 0 | 0 | 0 | 0 |
| NS | rate | 13 | 4875876 | 8e-05 | 0.00045287 | 0.00212745 - 0.0116635 | 0.00045287 | 1 | 0.0212807 |
| NS | exact | 13 | 4875876 | 8e-05 | 0.00045287 | 0 - 0 | 0 | 0 | 0 |
| HKS | rate | 13 | 33024 | 0.125786 | 0.00550282 | 0.000823588 - 0.0166132 | 0.00550282 | 1 | 0.074181 |
| HKS | exact | 13 | 33024 | 0.125786 | 0.00550282 | 0 - 0 | 0 | 0 | 0 |
| VBG | rate | 13 | 115712 | 1.30208e-05 | 0.00293975 | 0.186 - 1 | 0.00293975 | 1 | 0.0542195 |
| VBG | exact | 13 | 115712 | 1.30208e-05 | 0.00293975 | 0 - 0 | 0 | 0 | 0 |
