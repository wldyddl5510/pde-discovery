# WSINDy 논문 설정과의 비교

기존 두 문제인 Burgers와 KdV를 [WSINDy 논문](https://arxiv.org/html/2007.02848v3)의
Table 3 고정 설정과 Table 4의 시스템 크기에 맞췄다. 논문의 나머지 다섯 PDE는 이번 비교 범위에 포함하지 않았다.
원 저자의 [공개 데이터와 코드](https://github.com/dm973/WSINDy_PDE/tree/95686ccd9e32e3a9f62acfb3014fb77d5ef039ab)를 사용했다.

| 설정 | 기존 실험 | 이번 논문 설정 |
|---|---|---|
| Burgers 데이터 | x,u를 1000으로 나눈 entropy solution | 저자의 원 단위 데이터, 256×256 |
| KdV 데이터 | 다른 두 솔리톤 해, 256×256 | 저자의 Fourier/ETDRK4 데이터, 400×601 |
| 다항식 차수 | 0–3 | 0–6 |
| 공간 미분 차수 | Burgers 0–2, KdV 0–3 | 두 문제 모두 0–6 |
| 실제 후보 항 수 | 10 / 13 | 43 / 43 |
| Weak equation 수 | 121 / 121 | 784 / 1443 |
| Test half-width (mx,mt), 격자 단위 | 별도 물리적 길이 설정 | (60,60) / (45,80) |
| Query stride (sx,st) | 각 축 11개 중심점 | (5,5) / (8,12) |
| 약형 구성 | 희소 행렬, 사다리꼴 적분 | 저자의 separable FFT convolution |
| HT/L1 좌표 | 물리 단위 계수 | 저자의 rescaling을 적용한 계수 |
| Coefficient error | 상대 L2 오차의 중앙값 | 논문처럼 물리 단위 E2, E∞의 평균 |
| L1 GLS 목적함수 계산 | XᵀX를 이용한 전개 | 잔차 노름 직접 계산 (수식은 동일) |

Rescaling에는 재현상의 차이가 있다. 인쇄된 Eq. 4.8은
γu=(||U⁶||₂/||U||₂)^(-1/6)이지만, 공개 코드의 `toggle_scale=2`는 지수 -1/5를 쓴다.
이번에는 공개 코드를 따랐다. 20% 잡음 seed 0의 γu는 Burgers 9.03006e-4,
KdV 6.07074e-4로, Table 4의 4.5e-4와 5.7e-4를 그대로 재현한 값은 아니다.
따라서 모든 내부 숫자가 같은 재현이라고 할 수는 없다. 아래의 WSINDy 계수 오차와
support 통계를 별도로 검증했고, HT/L1도 이 공개 코드 기준의 rescaled 좌표에서 비교했다.

Burgers의 정답은 $u_t=-0.5\partial_x(u^2)$, KdV는 여기에
$-\partial_x^3u$를 더한다. Section 2의 operator-major 표기를 유지한다.
잡음은 논문 Eq. 5.1대로 iid Gaussian이며, σ=잡음비율×RMS(clean u)다.
실제 RMS는 Burgers 699.701718, KdV 274.457382이다.

## 원 논문 WSINDy 결과의 재현

무잡음 E∞는 실제 비영 계수들에 대한 최대 상대 오차다. 아래 값은 %가 아닌 비율이다.

| 문제 | 논문 Table 5 E∞ | 이번 E∞ | 이번 정확한 support |
|---|---:|---:|---:|
| Burgers | 4.3e-5 | 4.30846e-5 | 성공 |
| KdV | 3.1e-7 | 3.13515e-7 | 성공 |

WSINDy는 대표 잡음 0·10·20·50·100%에서 실행했다. 양의 잡음마다 200개 독립 seed,
무잡음은 1회다. 논문의 전체 41수준 sweep은 수행하지 않았다.

논문의 TPR은 **TP/(TP+FP+FN)**이며, support 전체를 정확히 맞출 확률과 다르다.
논문 §5.4/Figure 4는 두 문제에서 100% 잡음까지 평균 TPR>0.95를 보고한다.

| 문제 | σ/RMS(u) | 정확한 support 복원율 | 평균 TPR | 평균 E2 (%) | 평균 E∞ (%) |
|---|---:|---:|---:|---:|---:|
| Burgers | 0% | 100% | 1.0000 | 0.004308 | 0.004308 |
| Burgers | 10% | 99.5% | 0.9975 | 17.3164 | 0.227450 |
| Burgers | 20% | 100% | 1.0000 | 0.422519 | 0.422519 |
| Burgers | 50% | 100% | 1.0000 | 1.07388 | 1.07388 |
| Burgers | 100% | 98.5% | 0.9910 | 5.64981 | 2.34935 |
| KdV | 0% | 100% | 1.0000 | 0.00002843 | 0.00003135 |
| KdV | 10% | 99% | 0.9950 | 1.25196 | 1.39891 |
| KdV | 20% | 99% | 0.9950 | 1.60842 | 1.79669 |
| KdV | 50% | 94% | 0.9695 | 19.0306 | 5.50732 |
| KdV | 100% | 96% | 0.9800 | 10.7011 | 7.88316 |

Burgers 10%의 높은 평균 E2는 200회 중 한 번 큰 가짜 계수가 선택된 영향이다.
E∞는 가짜 항을 직접 평가하지 않으므로 E2와 차이가 난다. 논문 §5.5도 가짜 advection
항으로 인한 E2 outlier를 설명한다. Figure 6의 수치를 임의로 읽어 정확한 참조값처럼 쓰지 않았다.

| 20% 잡음 runtime | 논문 Table 4 | 이번 WSINDy 중앙값, 200회 |
|---|---:|---:|
| Burgers | 0.12 s | 0.03955 s |
| KdV | 0.39 s | 0.13286 s |

논문은 Intel i7-2670QM 2.2 GHz / 8 GB RAM의 serial MATLAB,
이번 측정은 Apple M4 / 16 GiB RAM의 Python 3.11.16, NumPy 2.4.6,
SciPy 1.17.1이다. 이 표의 Python WSINDy는 FFT/BLAS 모두 단일 스레드다.
하드웨어와 언어가 다르므로 위 시간 비를 알고리즘 자체의 speedup으로 해석할 수 없다.

## 기존 실험과의 차이

이전 CSV에서도 평균 E2를 다시 계산했다. 아래는 동일한 잡음 *비율*에서의 WSINDy 비교다.
기존 10회와 이번 200회는 표본 수, 데이터, 후보 항, test function이 달라 통제된 ablation이 아니다.

| 문제 · 잡음비율 | 기존 정확한 support | 이번 정확한 support | 기존 평균 E2 (%) | 이번 평균 E2 (%) |
|---|---:|---:|---:|---:|
| Burgers · 10% | 100% | 99.5% | 0.202760 | 17.3164 |
| Burgers · 100% | 40% | 98.5% | 86.2491 | 5.64981 |
| KdV · 10% | 100% | 99% | 0.207044 | 1.25196 |
| KdV · 100% | 100% | 96% | 7.03691 | 10.7011 |

## 같은 데이터에서의 방법 비교

두 PDE에서 아홉 변형의 총 72회 비교를 완료했다. 20% 잡음에서 WSINDy는 Burgers
3/3회, KdV 2/3회 정확한 support를 회복했다. 현재 WENDy/WENDy-MLE의 HT 및
세 L1 강도는 두 문제에서 각각 0/3회였다. L1은 일부 정답 항을 남겼지만 가짜 항도
선택해 정확한 support 회복으로 이어지지 않았다.

### Burgers · 20% 잡음 · 3 paired seeds

| 방법 | 정확한 support (%) | 평균 E2 (%) | Runtime 중앙값 (s) | Optimizer 정상 종료 |
|---|---:|---:|---:|---:|
| WSINDy | 100 | 0.484 | 0.05212 | 3/3 |
| WENDy HT | 0 | 100 | 457.5 | 0/3 |
| WENDy L1=1e-06 | 0 | 9.275e+16 | 511.4 | 0/3 |
| WENDy L1=0.0001 | 0 | 2.421e+16 | 6.366 | 0/3 |
| WENDy L1=0.01 | 0 | 1.997e+13 | 139.1 | 3/3 |
| WENDy-MLE HT | 0 | 100.1 | 462.9 | 0/3 |
| WENDy-MLE L1=1e-06 | 0 | 1.032e+17 | 966.6 | 3/3 |
| WENDy-MLE L1=0.0001 | 0 | inf | 139.9 | 0/3 |
| WENDy-MLE L1=0.01 | 0 | inf | 979.7 | 1/3 |

### KdV · 20% 잡음 · 3 paired seeds

| 방법 | 정확한 support (%) | 평균 E2 (%) | Runtime 중앙값 (s) | Optimizer 정상 종료 |
|---|---:|---:|---:|---:|
| WSINDy | 66.67 | 30.25 | 0.1332 | 3/3 |
| WENDy HT | 0 | 100 | 486.9 | 0/3 |
| WENDy L1=1e-06 | 0 | 6834 | 492.2 | 1/3 |
| WENDy L1=0.0001 | 0 | 2.009e+04 | 3.614 | 0/3 |
| WENDy L1=0.01 | 0 | 5.268e+04 | 48.4 | 3/3 |
| WENDy-MLE HT | 0 | 100 | 1415 | 0/3 |
| WENDy-MLE L1=1e-06 | 0 | 3.289e+05 | 1590 | 1/3 |
| WENDy-MLE L1=0.0001 | 0 | 4.073e+05 | 1970 | 0/3 |
| WENDy-MLE L1=0.01 | 0 | 5.467e+04 | 1878 | 1/3 |

`inf`는 비유한 계수로 종료한 수치 실패를 포함한 평균이다. 정상 종료와 정답 회복은 별개의 지표다.
전체 72회 중 optimizer 미수렴은 39회이며, 이 중 4회는 Cholesky 수치 실패다.
미수렴을 제외하지 않은 실행 시간이며, 정답에 수렴하는 데 걸린 시간으로 해석하지 않는다.

- [Support recovery 및 논문 TPR](paper_method_comparison/support.md)
- [Coefficient error E2 / E∞](paper_method_comparison/coefficient_accuracy.md)
- [Runtime 및 optimizer 정상 종료 비율](paper_method_comparison/runtime.md)

설정: 무잡음 1회, 20% 잡음 3개 paired seed; WSINDy와 WENDy/WENDy-MLE의
HT 및 세 L1 강도(1e-6, 1e-4, .01)를 모두 비교한다. 최대 300회 반복, tol=1e-8.
각 방법에 같은 noisy data와 43개 후보 항을 주며, WENDy 계열에는 주입한 σ를 제공한다.
HT는 정답의 항 수 s만 사용하고, L1 강도는 정답을 보고 고르지 않는다.

**작은 표본의 영향:** KdV의 seed 1에서 WSINDy는 분산항을 누락해 E2=0.894710을
기록했다. 같은 실패가 200회 reference에도 포함되어 있음을 확인했다. 따라서 WSINDy의
20% 잡음 KdV exact recovery는 reference에서 99%지만, paired seed 0–2에서는 2/3이다.
이 세 seed의 평균 E2는 약 30.25%로, 200회 평균 1.608%와 크게 다르다.
실패 seed를 교체하지 않았으며, 이 작은 paired 실험만으로 성공 확률을 추정하지 않는다.

무잡음에서도 문제별 차이가 있다. KdV에서는 두 HT 방법 모두 정답 support를 찾았고
E2는 3.56e-7이다. Burgers에서는 두 방법 모두 u⁵를 선택해 E2=1이었다.
무잡음 Burgers의 열 정규화 weak matrix 조건수는 약 2.34e8이다. 따라서 전체 43항을
먼저 적합한 뒤 큰 계수만 남기는 절차의 실패를 잡음 효과만으로 설명할 수는 없다.
L1 결과도 선택한 penalty 범위와 최적화 종료 상태에 한정해서 해석해야 한다.

이 비교는 FFT/정확한 공분산 곱에 최대 4개 스레드를 허용하며 BLAS는 1개다.
공분산의 대각 근사나 test equation 축소는 하지 않는다. 큰 문제의 MLE는 메모리를
절약하기 위해 analytic gradient와 BFGS를 사용한다. 따라서 이것은 WENDy 원 논문의
ODE 실험이나 원 MLE 최적화 소프트웨어를 그대로 재현한 결과가 아니다.
[원 WENDy](https://arxiv.org/html/2302.13271v2)의 orthonormal multiscale test function과
Shapiro–Wilk 종료 규칙 대신, 이번에는 WSINDy의 polynomial test function과 위의 공통
반복 기준을 사용했다. HT/L1을 포함한 현재 PDE 확장의 성능을 비교한 것이다.

이 구현은 매 반복에서 784×784 또는 1443×1443의 공분산을 다시 계산하므로,
작은 weak regression을 푸는 WSINDy보다 계산 비용이 크다. 메모리와 구현 방식에도
영향받는 측정값이며, 방법 자체의 보편적인 runtime 순위로 해석하지 않는다.

세 번의 반복으로 얻은 0/33.3/66.7/100%를 정밀한 성공 확률 추정으로 해석할 수 없다.
미수렴 결과도 error/support/runtime 집계에 포함하며 종료 상태를 함께 보고한다.
원시 계수, 선택한 항, 반복 수와 종료 메시지는 각 폴더의 `trials.csv`에 저장한다.
