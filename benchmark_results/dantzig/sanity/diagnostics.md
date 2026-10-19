# Physical-score 진단: pilot 오차와 IB의 support 선택

이 문서는 주 실험의 성능 집계와 별도인 과학적 진단이다. 주 실험에서는 현재 `SplitPilot`을 사용하고, true coefficient는 최종 λ 선택과 평가에만 사용한다. 아래 **oracle-pilot 진단은 clean state까지 사용하는 추가 oracle**이며, 주 실험 결과에 포함하지 않는다.

## 독립 LP로 확인한 noiseless IB 결과

IB의 true equation은 (u_t=-0.5(u^2)_x)이다. 43개 후보 support 각각에 대해 계수 하나만 허용하고

\[
\min_c \frac{\|b-H_{\cdot,j}c\|_\infty}{\|b\|_\infty}
\]

를 SciPy/HiGHS의 연속 LP로 풀었다. 이는 Gurobi/MILP와 독립된 확인이다. 모든 score coordinate에 **동일한** 상수를 나누고 계수 변수만 치환했으며, estimator의 물리 단위나 공통 tolerance는 바꾸지 않았다. True support는 LP 이후의 평가에만 사용했다.

| Pilot / score | 전체 one-term support 중 최선 | 정규화 residual | True support만 허용한 최선 |
|---|---|---:|---|
| 현재 pilot / plug-in | \(-536.208151u_x\) | \(5.66768\times10^{-5}\) | \(-0.361966734(u^2)_x\), residual \(1.00185\times10^{-4}\) |
| 현재 pilot / debiased | \(-537.130871u_x\) | \(4.56211\times10^{-5}\) | \(-0.405998721(u^2)_x\), residual \(2.02541\times10^{-4}\) |
| **추가 oracle: exact pilot / debiased** | **\(-0.501730853(u^2)_x\)** | **\(1.25053\times10^{-7}\)** | 전체 최선과 동일; 상대 계수 \(E_2=0.00346171\) |

현재 debiased score에서는 잘못된 `u_x` support가 true support보다 작은 minimax residual을 달성한다. 따라서 **현재 pilot와 구현에서 추가한 residual tie-break를 사용하는 경우**, 최적 sparsity가 1인 λ 구간에서는 λ grid를 촘촘하게 해도 true support를 선택할 수 없다. 이 결론을 식 (9)의 ℓ₀ 목적 자체의 불가능성으로 일반화할 수는 없다. λ가 true support의 최소 residual 이상이고 sparsity 1이 최적인 경우, 식 (9)는 true support와 잘못된 support를 동시에 minimizer로 가질 수 있다. 위의 불리한 residual ranking은 독립 LP에서도 재현되므로 Gurobi의 수치 실패만으로 설명되지 않는다.

Exact pilot에서는 올바른 support가 복원된다. 다음으로 좋은 잘못된 support는 `(u^3)_x`이며 residual은 \(2.60385\times10^{-5}\), true support보다 약 208배 크다. 따라서 이 데이터·dictionary·split quadrature에서는 **pilot가 정확해도 피할 수 없는 support 비식별성**이 현재 실패의 원인은 아니다. 이 진단만으로 noisy-data 성능을 보장할 수는 없다.

## Pilot와 물리 단위의 영향

관측 noise가 0이어도 현재 IB pilot의 regression 시점 상대 \(L_2\) 오차는 **0.261244**이다. Clean state 범위는 0–1000이며, pilot의 noise estimate는 16.407, smoothing window는 61×61이다. 실제 잡음이 없는 경우에도 이 데이터에서는 현재 pilot가 상당한 오차를 남긴다.

동시에 물리 단위의 weak-column RMS 최대/최소 비율은 현재 pilot에서 \(6.25\times10^{31}\), exact pilot에서도 \(1.32\times10^{31}\)이다. 현재 score의 truth residual은 `u^6` coordinate가 지배한다. 그 최대값을 \(\|b\|_\infty\)로 나눈 비율은 0.231780인 반면, true term `(u^2)_x`의 score coordinate에서는 \(2.21\times10^{-17}\)이다. 공통 infinity-norm tolerance가 dictionary coordinate의 물리적 크기에 매우 민감한 상황이다.

Exact pilot는 truth feasibility ratio를 0.231780에서 0.00344964로 줄인다. 큰 feature dynamic range는 그대로이지만 support가 복원되므로, 이 진단은 **현재 pilot 오차와 score의 scale 민감도가 함께 작용한다**는 해석을 뒷받침한다. 다만 정규화 방식을 바꾸는 ablation은 수행하지 않았으므로, scale 효과의 인과적 기여를 pilot 오차와 분리하여 확인한 것은 아니다. Exact-pilot 계수의 남은 오차에는 weak quadrature/discretization과 residual minimax 기준의 영향이 포함될 수 있다.

## 전체 cache 확인 및 재현

`diagnostics.json`은 8개 PDE 각각 13개 realization, 총 **104개**를 포함한다. 모든 cache의 SHA256, 행렬 차원, 유한 값, (H)의 대칭성을 확인했다. Noiseless 데이터에서는 8개 PDE 모두 debiasing이 plug-in 대비 truth feasibility ratio를 줄였지만, 이것만으로 support recovery나 계수 오차 개선이 보장되지는 않는다.

전체 score 진단은 state/weak FFT를 다시 계산하지 않는다. Exact-pilot IB 진단만 동일한 test functions·training mask·regression quadrature로 weak matrices를 새로 구성했다. 모든 독립 one-column LP는 constraint validation을 통과했다.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/homebrew/anaconda3/envs/pde_discovery/bin/python benchmark_results/dantzig/sanity/diagnose_scores.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/homebrew/anaconda3/envs/pde_discovery/bin/python benchmark_results/dantzig/sanity/diagnose_oracle_pilot.py
```

근거: [전체 score 진단 JSON](diagnostics.json), [추가 oracle-pilot 진단 JSON](diagnostics_oracle_pilot.json), [oracle-pilot cached matrices](diagnostics_oracle_pilot.npz). JSON에는 데이터·source·cache hash와 모든 one-column LP 결과가 저장되어 있다.
