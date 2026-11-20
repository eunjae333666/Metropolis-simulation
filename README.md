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

`main.py` 상단에서 격자 한 변 `L`, 온도 `T`, 초기 배열 `INITIALIZATION`, 시드 `SEED` 및 측정 간격을 바꿀 수 있습니다. 격자, 스핀당 에너지와 자화의 변화를 그래프로 보여줍니다. 1 Monte Carlo sweep은 스핀 개수 $L^2$만큼 뒤집기를 시도한 단위입니다.

## Experiments

### Ordered vs Random Initial State

같은 온도에서 **모든 스핀이 위를 향하는 격자(Ordered)** 와 **각 스핀을 위·아래로 무작위 배치한 격자(Random)** 를 비교합니다. 시점 $t$마다 각 초기조건의 독립 실행에서 스핀당 에너지 $E/N$를 평균하고, 두 평균 차이의 절댓값를 계산합니다.

$$
D_E(t) =
\left|
\left\langle E/N\right\rangle_{\mathrm{random}}(t)
-
\left\langle E/N\right\rangle_{\mathrm{ordered}}(t)
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

### **Problem 8.29** — Correlation Function and Correlation Length

거리 $r$만큼 떨어진 spin $s_i,s_j\in\{-1,+1\}$에 대해 correlation function을 다음과 같이 정의합니다.

$$
c(r)=\overline{s_i s_j}-\bar{s}^{\,2}
$$

여기서 $\overline{s_i s_j}$는 거리 $r$만큼 떨어진 spin pair들의 $s_i s_j$ 평균이고, $\bar{s}$는 lattice 전체 spin의 평균입니다. 코드에서는 $\overline{s_i s_j}$를 따로 $C(r)$로 두어

$$
C(r)=\overline{s_i s_j},\qquad c(r)=C(r)-\bar{s}^{\,2}
$$

로 계산합니다.

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

System을 충분히 equilibrate한 뒤 여러 lattice configuration에서 correlation을 반복 측정하여 평균합니다. 이렇게 얻은 correlation function이 대략

$$
c(r)\approx Ae^{-r/\xi}
$$

와 같이 감소한다고 보면, correlation length $\xi$는 correlation이 약 $1/e$배로 감소하는 characteristic distance입니다. 따라서

$$
\ln c(r)\approx\ln A-\frac{r}{\xi}
$$

이고, $\ln c(r)$을 $r$에 대해 fitting하여 $\xi$를 추정합니다.

`correlation_semilog.png`에서는 correlation의 decay와 fitting 결과를, `correlation_length_vs_temperature.png`에서는 temperature에 따른 $\xi$의 변화를 확인할 수 있습니다. Critical temperature 근처에서는 $\xi$가 증가하지만, finite lattice에서는 lattice size의 영향을 받고 정확히 critical point에서는 correlation이 단순한 exponential 형태로 감소하지 않으므로 fitting 결과는 주의해서 해석해야 합니다.

#### Axis와 radial 비교

`--geometry axis`가 기본값이며 교재의 가로·세로 평균을 사용합니다. `--geometry radial`은 대각선을 포함한 모든 주기 변위를 사용합니다. 각 성분에 minimum image를 적용한 후 정수 $r^2=d_x^2+d_y^2$가 같은 변위끼리 평균하고, 표시할 때 $r=\sqrt{r^2}$로 바꿉니다. radial 곡선은 격자의 대각선 방향까지 포함해 최대 $\sqrt{2}\lfloor L/2\rfloor$까지 나옵니다. 기본 피팅 상한은 두 방식 모두 `L//4`입니다.

```bash
uv run python -m problems.problem_8_29 --geometry both --correlation both \
  --output-dir results/problem_8_29_both --no-show
```

`both`는 같은 lattice configuration에서 두 곡선을 계산합니다. axis 결과는 지정한 폴더에, radial 결과는 그 안의 `radial/`에 저장합니다. `geometry_comparison.png`에서 두 연결 상관을 비교하고, 위 예제처럼 `--correlation both`를 지정하면 `raw_geometry_comparison.png`도 생성됩니다.

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