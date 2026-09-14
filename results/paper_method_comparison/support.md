열은 잡음 비율 σ/RMS(u)입니다. L1 값은 λ_ref에 곱하는 상대 penalty입니다.

## Burgers · 256×256

실제 σ: 0, 139.9.
반복 횟수: 1, 3.

**정확한 support 복원율 (%)**

| 방법 | 0% | 20% |
|---|---:|---:|
| WSINDy | 100 | 100 |
| WENDy HT | 0 | 0 |
| WENDy L1=1e-06 | 0 | 0 |
| WENDy L1=0.0001 | 0 | 0 |
| WENDy L1=0.01 | 0 | 0 |
| WENDy-MLE HT | 0 | 0 |
| WENDy-MLE L1=1e-06 | 0 | 0 |
| WENDy-MLE L1=0.0001 | 0 | 0 |
| WENDy-MLE L1=0.01 | 0 | 0 |

**논문 TPR 평균: TP/(TP+FP+FN)**

| 방법 | 0% | 20% |
|---|---:|---:|
| WSINDy | 1 | 1 |
| WENDy HT | 0 | 0 |
| WENDy L1=1e-06 | 0.02381 | 0.008547 |
| WENDy L1=0.0001 | 0.02326 | 0.03314 |
| WENDy L1=0.01 | 0.3333 | 0.1082 |
| WENDy-MLE HT | 0 | 0 |
| WENDy-MLE L1=1e-06 | 0.02381 | 0.02326 |
| WENDy-MLE L1=0.0001 | 0.02326 | 0 |
| WENDy-MLE L1=0.01 | 0.3333 | 0.05808 |

## KdV · 400×601

실제 σ: 0, 54.89.
반복 횟수: 1, 3.

**정확한 support 복원율 (%)**

| 방법 | 0% | 20% |
|---|---:|---:|
| WSINDy | 100 | 66.67 |
| WENDy HT | 100 | 0 |
| WENDy L1=1e-06 | 0 | 0 |
| WENDy L1=0.0001 | 0 | 0 |
| WENDy L1=0.01 | 0 | 0 |
| WENDy-MLE HT | 100 | 0 |
| WENDy-MLE L1=1e-06 | 0 | 0 |
| WENDy-MLE L1=0.0001 | 0 | 0 |
| WENDy-MLE L1=0.01 | 0 | 0 |

**논문 TPR 평균: TP/(TP+FP+FN)**

| 방법 | 0% | 20% |
|---|---:|---:|
| WSINDy | 1 | 0.8333 |
| WENDy HT | 1 | 0 |
| WENDy L1=1e-06 | 0.25 | 0.05817 |
| WENDy L1=0.0001 | 0.4 | 0.0523 |
| WENDy L1=0.01 | 0.2 | 0.1429 |
| WENDy-MLE HT | 1 | 0 |
| WENDy-MLE L1=1e-06 | 0.25 | 0.04651 |
| WENDy-MLE L1=0.0001 | 0.4 | 0.04651 |
| WENDy-MLE L1=0.01 | 0.2 | 0.1032 |

비유한 계수로 끝난 수치 실패는 support 복원 실패 및 TPR=0으로 집계합니다.

전체 72회 중 optimizer 미수렴 39회도 포함했습니다. 종료 상태와 추정 계수는 trials.csv에 있습니다.
