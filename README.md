# Metropolis Simulation

## Overview

주기 경계를 적용한 2차원 정사각 Ising 격자를 단일 스핀 Metropolis 방법으로 시뮬레이션합니다. 기본 예제에서 격자와 에너지·자화를 확인하고, 개별 실험에서 초기조건이나 온도의 영향을 비교할 수 있습니다.

## Installation

저장소 폴더에서 처음 한 번 실행합니다.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

이미 활성화한 Python 가상환경이 있다면 `python -m pip install -r requirements.txt`로 설치하고, 아래 예제의 `uv run python`을 `python`으로 바꿔 실행하면 됩니다.

## Basic Usage

```bash
uv run python main.py
```

`main.py` 상단에서 격자 한 변 `L`, 온도 `T`, 초기 배열 `INITIALIZATION`, 시드 `SEED` 및 측정 간격을 바꿀 수 있습니다. 격자, 스핀당 에너지와 자화의 변화를 그래프로 보여줍니다. Monte Carlo sweep 한 번은 스핀 개수($L^2$)만큼 뒤집기를 시도한 단위입니다.

## Experiments

### Ordered vs Random Initial State

같은 온도에서 **Ordered**(모든 스핀이 위를 향하는 격자)와 **Random**(각 스핀을 위·아래로 무작위 배치한 격자)을 비교합니다. 시점 $t$마다 초기조건별로 스핀당 에너지($E/N$)를 평균하고, 두 평균 차이의 절댓값을 계산합니다.

$$
D_E(t) =
\left|
\left\langle E/N\right\rangle_{\mathrm{random}}(t) - \left\langle E/N\right\rangle_{\mathrm{ordered}}(t)
\right|
$$

**기본 실행**

```bash
uv run python -m experiments.compare_initial_states
```

**옵션을 바꿔 실행**

```bash
uv run python -m experiments.compare_initial_states \
  --temperatures 1.0 1.5 2.0 2.27 2.5 3.0 5.0 \
  --L 40 \
  --J 1 \
  --runs 8 \
  --max-sweeps 2000 \
  --sample-interval 1 \
  --max-workers 10 \
  --seed 42 \
  --tail-fraction 0.25 \
  --energy-tolerance 0.02 \
  --relaxation-window 10 \
  --output-dir results/initial_state_comparison \
  --no-show
```

| 옵션 | 바꾸는 값 |
| --- | --- |
| `--temperatures` | 실험할 온도 목록. 공백으로 구분합니다. |
| `--L` | 격자 한 변의 스핀 수. `--L 40`은 40×40 격자입니다. |
| `--J` | 이웃 스핀 사이의 결합 세기. |
| `--runs` | 온도와 초기조건 조합당 독립 실행 수. `8`이면 온도마다 총 16회입니다. |
| `--max-sweeps` | 독립 실행 한 번에 진행할 총 sweep 수. |
| `--sample-interval` | 에너지를 기록하는 간격(sweep). 시작값과 마지막 값도 기록합니다. |
| `--max-workers` | 동시에 실행할 프로세스의 최대 수. `1`이면 순차 실행합니다. |
| `--seed` | 결과를 재현할 때 사용하는 난수 시드. |
| `--tail-fraction` | 후반부 분포에 포함할 실행 구간의 비율. `0.25`는 마지막 25%입니다. |
| `--energy-tolerance` | 완화 판정에서 허용하는 스핀당 평균 에너지 차이. |
| `--relaxation-window` | 완화 판정에 사용하는 구간 하나의 표본 수. |
| `--output-dir` | PNG를 저장할 폴더. |
| `--no-show` | 그래프 창을 띄우지 않습니다. |

완화 sweep은 서로 붙어 있는 두 구간에서 각각 두 초기조건의 평균 에너지 차이가 허용값 이내에 들어오는 첫 구간의 시작입니다. 짧게 시험하려면 위 명령에서 `--L 16 --runs 3 --max-sweeps 200 --max-workers 2`에 해당하는 값만 바꿔 보세요.

**결과 파일**

| 파일 | 보는 법 |
| --- | --- |
| [`energy_relaxation.png`](docs/energy_relaxation.png) | 온도별 평균 에너지 곡선과 독립 실행 간 표준편차 띠. |
| [`energy_gap.png`](docs/energy_gap.png) | 온도별로 $D_E(t)$가 줄어드는 속도. |
| [`late_energy_distribution.png`](docs/late_energy_distribution.png) | 후반부 에너지의 경험적 누적분포(CDF). 두 초기조건의 분포가 얼마나 겹치는지 볼 수 있습니다. |

링크는 예시 그래프입니다. 새 실험 그래프는 `--output-dir`로 지정한 폴더에 저장됩니다.

**예시 결과**

40×40 격자에서 $J=1$, 시드 42로 초기조건마다 8회씩 2,000 sweep을 실행했습니다. Ordered의 시작 에너지는 $E/N=-2$, Random은 0 근처였습니다.

| 온도 | 완화 추정 (sweep) | 후반부 평균 $E/N$ 차이 |
| ---: | ---: | ---: |
| 1.0 | 기준 미충족 | 0.0470 |
| 1.5 | 1,009 | 0.0066 |
| 2.0 | 477 | 0.0030 |
| 2.27 | 422 | 0.0420 |
| 2.5 | 118 | 0.0074 |
| 3.0 | 24 | 0.0010 |
| 5.0 | 3 | 0.0024 |

후반부 차이는 sweep 1,500–2,000에서 초기조건별로 에너지를 평균한 뒤 차이의 절댓값을 낸 값입니다. $T=5$에서는 두 조건의 에너지가 금방 비슷해졌습니다. $T=2.27$에서는 422 sweep 부근에 평균이 잠시 가까워졌지만 후반부에도 차이가 남았습니다.

## Schroeder Problems

### Problem 8.29 — Correlation Function and Correlation Length

거리 $r$만큼 떨어진 두 스핀 $s_i, s_j\in\{-1,+1\}$의 곱을 평균한 값이 원시 상관 $C(r)$입니다. 여기서 평균 자화 $\bar{s}$의 제곱을 빼면 연결 상관 $c(r)$이 됩니다.

$$
C(r)=\overline{s_i s_j},\qquad
c(r)=\overline{s_i s_j}-\bar{s_i}^{\,2}
$$

#### (a) 한 lattice에서 거리별 correlation 측정

`analysis/correlation.py`의 `axis_pair_correlation()`은 하나의 lattice에서 모든 lattice point를 기준으로, 가로·세로 방향으로 거리 $r$만큼 떨어진 spin pair의 곱을 평균합니다. 경계에서는 periodic boundary condition을 적용하며, 기본적으로 $0\le r\le\lfloor L/2\rfloor$ 범위를 계산합니다.

`problem_8_29.py`에서는 simulation 중 여러 시점의 lattice에서 $c(r)$을 측정하며, `snapshot_correlations.png`에서 각 시점의 correlation을 비교할 수 있습니다.

#### (b) Temperature에 따른 correlation 비교

여러 temperature에서 simulation을 실행하여 $c(r)$이 어떻게 달라지는지 비교합니다. $J=1$인 2D square Ising model의 critical temperature는

$$
T_c\approx2.269
$$

이므로 $T<T_c$, $T\approx T_c$, $T>T_c$ 영역을 함께 계산합니다. `correlation_vs_r.png`에서는 여러 temperature의 $c(r)$을 같은 graph에서 비교할 수 있습니다. 높은 temperature에서는 correlation이 짧은 거리에서 빠르게 감소하고, critical temperature 근처에서는 더 먼 거리까지 correlation이 유지됩니다.

#### (c) 여러 lattice의 평균과 Correlation length

System을 충분히 equilibrate한 뒤 여러 lattice configuration에서 correlation을 반복해서 측정하고 평균합니다. 이렇게 얻은 connected correlation $c(r)$을 $c(0)$으로 normalize하여

$$
\frac{c(r)}{c(0)}
$$

를 계산합니다.

Problem 8.29에서 correlation length $\xi$는 correlation이 처음 값의 $1/e$까지 감소하는 characteristic distance로 두었습니다. 따라서 코드에서는

$$
\frac{c(\xi)}{c(0)}=\frac{1}{e}
$$

가 되는 거리를 simulation data에서 직접 찾습니다. Lattice에서는 측정 가능한 거리가 discrete하므로, $1/e$를 사이에 두는 두 data point 사이를 linear interpolation하여 empirical correlation length $\xi$를 구합니다.

`normalized_correlation_vs_r.png`에서는 각 temperature에 대한 $c(r)/c(0)$과 $1/e$ 기준선을 함께 표시하여 $\xi$가 결정되는 위치를 직접 확인할 수 있습니다. `correlation_length_vs_temperature.png`에서는 이렇게 구한 empirical correlation length를 temperature에 따라 비교합니다.

Critical temperature $T_c$ 근처에서는 correlation이 먼 거리까지 유지되므로 $\xi$가 크게 증가합니다. 이 계산은 특정한 decay function을 가정하지 않고 finite lattice에서 실제로 측정된 correlation으로부터 직접 $\xi$를 구한 것입니다. 만약 측정한 최대 거리까지도 $c(r)/c(0)>1/e$라면 정확한 crossing을 정할 수 없으므로, 해당 최대 거리를 correlation length의 lower bound로 기록합니다.

추가적인 비교를 위해

$$
c(r)\approx Ae^{-r/\xi_{\rm exp}}
$$

형태의 exponential fit도 수행할 수 있습니다. 이 경우

$$
\ln c(r)\approx \ln A-\frac{r}{\xi_{\rm exp}}
$$

를 이용하지만, 이는 empirical $\xi$를 구하기 위한 필수 과정이 아니라 correlation decay가 exponential 형태에 가까운 경우의 보조적인 분석입니다. 특히 critical point에서는 correlation이 단순한 exponential decay를 따르지 않으므로 exponential fit 결과보다 직접 측정한 empirical correlation length를 기본 결과로 사용합니다.

#### 실행 방법

**기본 실행**

```bash
uv run python -m problems.problem_8_29
```

기본값은 교재의 가로·세로 방향을 평균하는 `axis`입니다. 거리 0은 계산에 포함되며, (a)의 막대그래프는 거리 1부터 표시합니다. 결과는 `results/problem_8_29/`에 저장합니다.

**옵션을 바꿔 실행**

```bash
uv run python -m problems.problem_8_29 \
  --temperatures 1.5 2.0 2.15 2.25 2.269 2.35 2.5 3.0 \
  --L 40 \
  --J 1 \
  --runs 4 \
  --burn-in 3000 \
  --samples 500 \
  --sample-spacing 10 \
  --max-workers 10 \
  --seed 42 \
  --sector auto \
  --fit-r-min 2 \
  --geometry axis \
  --correlation both \
  --lattice-snapshots \
  --output-dir results/problem_8_29 \
  --no-show
```

| 옵션 | 바꾸는 값 |
| --- | --- |
| `--temperatures` | 실험할 온도 목록. 공백으로 구분합니다. |
| `--L` | 격자 한 변의 스핀 수. `--L 40`은 40×40 격자입니다. |
| `--J` | 이웃 스핀 사이의 결합 세기. 양수를 사용합니다. |
| `--runs` | 온도별 독립 실행 수. `4`이면 온도마다 네 번 실행합니다. |
| `--burn-in` | 측정을 시작하기 전 격자를 안정화하는 데 사용할 sweep 수. |
| `--samples` | 독립 실행 한 번에 측정할 격자 수. |
| `--sample-spacing` | 두 측정 사이의 sweep 수. |
| `--max-workers` | 동시에 실행할 프로세스의 최대 수. `1`이면 순차 실행합니다. |
| `--seed` | 결과를 재현할 때 사용하는 난수 시드. |
| `--sector` | 평균 자화에 사용할 값. `signed`는 부호를 유지하고, `aligned`는 절댓값을 사용합니다. 기본값 `auto`는 임계점 아래에서 `aligned`, 임계점부터 `signed`를 사용합니다. |
| `--fit-r-min` | 보조 지수 피팅에 사용할 최소 거리. 기본값은 `2`입니다. |
| `--fit-r-max` | 보조 지수 피팅에 사용할 최대 거리. 지정하지 않으면 격자 크기에 따라 `L//4`로 정해집니다. |
| `--geometry` | 거리별로 평균할 방향. `axis`는 가로·세로, `radial`은 모든 방향, `both`는 같은 격자에서 두 방식 모두 계산합니다. |
| `--correlation` | 거리별 그래프에 표시할 상관. `connected`는 $c(r)$, `raw`는 $C(r)$, `both`는 둘 다 저장합니다. |
| `--lattice-snapshots` | 대표 온도의 마지막 측정 격자도 그림으로 저장합니다. |
| `--output-dir` | 그래프와 데이터 파일을 저장할 폴더. |
| `--no-show` | 그래프 창을 띄우지 않고 파일을 저장합니다. |

짧게 시험하려면 위 명령에서 `--L 24 --runs 2 --burn-in 500 --samples 100 --max-workers 2`에 해당하는 값을 바꿔 보세요.

**결과 파일**

| 파일 | 보는 법 |
| --- | --- |
| `snapshot_correlations.png` | 측정 시점별로 한 격자의 연결 상관을 막대그래프로 비교합니다. |
| `correlation_vs_r.png` | 온도별 연결 상관 $c(r)$을 비교합니다. `--correlation raw`를 지정하면 원시 상관 $C(r)$을 표시합니다. |
| `raw_correlation_vs_r.png` | `--correlation both`에서 추가로 저장하는 원시 상관 그래프입니다. |
| `normalized_correlation_vs_r.png` | 정규화한 연결 상관과 `1/e` 기준선. 원은 교차점, 삼각형은 측정 범위의 끝까지 교차하지 않은 경우입니다. |
| `correlation_semilog.png` | 통과한 구간의 $\ln c(r)$ 측정점과 피팅 직선을 보여 줍니다. 통과한 피팅이 없으면 상태를 표시합니다. |
| `correlation_length_vs_temperature.png` | 온도별 경험적 상관 길이, 미교차 하한, 유효한 지수 피팅 결과를 함께 표시합니다. |
| `geometry_comparison.png` | `--geometry both`에서 같은 표본으로 측정한 axis와 radial을 비교합니다. |
| `raw_geometry_comparison.png` | `--geometry both --correlation both`에서 추가로 저장하는 원시 상관 비교 그래프입니다. |
| `lattice_snapshots.png` | `--lattice-snapshots`에서 저장하는 대표 온도의 격자 그림입니다. |
| `summary.csv` | 기본 결과 `xi_empirical`, 계산 상태와 하한 여부, 보조 결과 `xi_exp_*` 및 피팅 진단을 확인합니다. |
| `correlation_data.npz` | 거리별 상관과 정규화한 곡선, 경험적 길이와 하한 플래그, 독립 실행별 피팅 결과를 저장합니다. |

#### 확장 기능 — Axis와 radial 비교

`--geometry axis`가 기본값이며 교재의 가로·세로 평균을 사용합니다. `--geometry radial`은 대각선을 포함한 모든 주기 변위를 사용합니다. 각 성분에 minimum image를 적용한 후 정수 $r^2=d_x^2+d_y^2$가 같은 변위끼리 평균하고, 표시할 때 $r=\sqrt{r^2}$로 바꿉니다. radial 곡선은 격자의 대각선 방향까지 포함해 최대 $\sqrt{2}\lfloor L/2\rfloor$까지 나옵니다. 보조 지수 피팅의 기본 상한은 두 방식 모두 `L//4`입니다.

```bash
uv run python -m problems.problem_8_29 --geometry both --correlation both \
  --output-dir results/problem_8_29_both --no-show
```

`both`는 같은 lattice configuration에서 두 곡선을 계산합니다. axis 결과는 지정한 폴더에, radial 결과는 그 안의 `radial/`에 저장합니다. `geometry_comparison.png`에서 두 연결 상관을 비교하고, 위 예제처럼 `--correlation both`를 지정하면 `raw_geometry_comparison.png`도 생성됩니다.

### Problem 8.32 — Block-Spin Transformation

#### (a) 3×3 다수결 변환

평형에 가까워진 격자를 겹치지 않는 $3\times3$ 블록으로 나누고, 각 블록에서는 다수인 스핀 방향 하나를 남깁니다.

$$
S'_{IJ}=\operatorname{sign}\left(\sum_{(i,j)\in\mathrm{block}(I,J)}s_{ij}\right)
$$

기본 $90\times90$ 격자는 $30\times30$으로 줄어듭니다. `block_spin_snapshots.png`에서 같은 시점의 원본과 결과를 나란히 봅니다.

#### (b) 변환 온도 $T'$ 추정

각 기준 온도에서 독립적으로 시뮬레이션한 원본 격자에서, 변환 격자와 같은 크기의 패치를 무작위로 뽑습니다. 두 격자에서 모두 스핀당 절대 자화 $\lvert m\rvert$와 가로·세로 원시 상관 $C(1),\ldots,C(5)$를 측정합니다. 상관을 계산할 때는 패치의 양 끝을 서로 연결하지 않습니다.

각 특징의 차이를 전체 기준 표본에서 측정한 표준편차로 나눈 뒤, 변환 표본의 평균과 기준 온도별 평균 사이의 유클리드 거리가 가장 작은 온도를 $T'$로 고릅니다. 기준 표본에서 변하지 않는 특징은 거리 계산에서 제외합니다. 이 $T'$는 선택한 특징과 기준 온도 격자에 한정된 **경험적 best matching temperature**입니다. 정확한 재규격화 온도를 정하려면 변환 후의 상호작용을 별도로 분석해야 합니다.

#### (c) 온도 흐름

화살표는 측정한 $T\to T'$를 표시합니다. $T'<T$는 낮은 온도 쪽, $T'>T$는 높은 온도 쪽이며, $\lvert T'-T\rvert$가 허용값 이내인 점을 고정점 후보로 표시합니다. 교재에서 반복 변환의 흐름은 임계점 아래에서 $0$ 쪽, 위에서 $\infty$ 쪽이며 $T_c$가 두 흐름을 가르는 고정점입니다.

$$
\frac{T_c}{J}=\frac{2}{\ln(1+\sqrt{2})}
$$

정확한 $T_c$는 그래프의 참고선입니다. 수치상 교차점은 유한 격자, 표본 수, 기준 온도 간격에 따라 달라집니다.

#### 실행 방법

```bash
uv run python -m problems.problem_8_32 --no-show
```

옵션을 바꿔 실행하는 예입니다. 이 설정에서는 두 시점에서 $90\to30\to10$ 변환을 그리며, $T'$는 첫 번째 변환을 기준으로 계산합니다.

```bash
uv run python -m problems.problem_8_32 \
  --L 90 --J 1 --temperatures 1.8 2.0 2.15 2.269 2.4 2.6 3.0 \
  --reference-temperatures 1.0 1.3 1.6 1.9 2.1 2.2 2.25 2.269 2.32 2.4 2.5 2.65 2.8 3.0 3.4 4.0 5.0 8.0 12.0 \
  --burn-in 3000 --samples 80 --sample-spacing 10 \
  --runs 3 --max-workers 10 --seed 42 \
  --snapshots 2 --block-levels 2 --fixed-point-tolerance 0.05 \
  --output-dir results/problem_8_32 --no-show
```

| 옵션 | 바꾸는 값 |
| --- | --- |
| `--L`, `--J` | 원본 격자의 한 변과 양의 결합 세기. `L`은 6 이상이며 단계마다 3으로 나누어떨어져야 합니다. |
| `--temperatures`, `--reference-temperatures` | 변환할 원본의 온도 목록과 $T'$를 고를 후보 온도 목록. |
| `--burn-in`, `--samples`, `--sample-spacing` | 안정화에 사용할 sweep 수, 체인당 표본 수, 측정 간격. |
| `--runs`, `--max-workers`, `--seed` | 온도와 표본 집단당 독립 체인 수, 병렬 작업 수, 난수 시드. |
| `--snapshots`, `--block-levels` | 저장할 측정 시점 수와 시각화용 반복 변환 횟수. 기본값은 각각 1입니다. |
| `--fixed-point-tolerance` | 고정점 후보의 $\lvert\Delta T\rvert$ 상한. 기본값은 $0.05J$입니다. |
| `--output-dir`, `--no-show` | 저장 위치와 그래프 창 표시 여부. |

| 파일 | 보는 법 |
| --- | --- |
| `block_spin_snapshots.png` | 원본과 다수결 변환. 측정 시점을 늘리면 번호가 붙은 PNG도 저장됩니다. |
| `reference_patch_comparison.png` | 변환 격자와 추정한 $T'$의 원본 패치. |
| `transformed_temperature.png`, `delta_temperature.png` | $T'=T$ 기준과의 비교와 $\Delta T=T'-T$. |
| `rg_flow.png` | 온도 흐름 화살표, 후보 고정점, 정확한 $T_c$ 참고선. |
| `summary.csv`, `block_spin_data.npz` | $T$, $T'$, $\Delta T$, 매칭 점수와 기준 온도 범위 끝에 도달했는지 여부. NPZ에는 표본별 특징, 모든 후보의 점수와 예시 격자도 저장합니다. |

## Project Structure

```text
Metropolis-simulation/
├── README.md
├── .gitignore
├── requirements.txt
├── main.py
├── model.py
├── sampler.py
├── runner.py
├── convergence.py
├── plotting.py
├── analysis/
│   └── correlation.py
├── experiments/
│   └── compare_initial_states.py
├── problems/
│   └── problem_8_29.py
├── docs/
│   ├── energy_relaxation.png
│   ├── energy_gap.png
│   └── late_energy_distribution.png
└── tests/
    ├── test_ising.py
    └── test_correlation.py
```

루트의 Python 모듈은 격자·샘플링·실행·그래프 기능을 맡습니다. `analysis/`는 공통 분석 함수, `experiments/`는 실험 진입점, `problems/`는 교재 문제별 진입점입니다. `docs/`에는 문서용 예시, `tests/`에는 검증 코드가 있습니다. 새 실험 결과는 `results/`에 저장할 수 있습니다.

## Tests

```bash
uv run python -m pytest -q
```

스핀 뒤집기와 에너지 계산, 주기 경계, 상관함수 평균, 알려진 지수 감쇠에서의 상관 길이 복원을 검사합니다.
