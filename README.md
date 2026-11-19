# 2차원 Ising 모형

모든 스핀이 위를 향한 격자(Ordered)와 각 스핀을 위·아래로 무작위 배치한 격자(Random)를 같은 온도에서 시작합니다. 스핀당 에너지 `E/N`가 얼마나 빨리 비슷해지는지 비교합니다.

## 실행하기

`uv`가 설치된 터미널에서 `ising_model` 폴더를 열고 아래 명령을 실행합니다. 처음 두 줄은 환경을 처음 만들 때만 실행하면 됩니다.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -m experiments.compare_initial_states --output-dir results --no-show
```

Python 가상환경을 이미 쓰고 있다면 `python -m pip install -r requirements.txt`로 설치하고 마지막 명령을 `python -m experiments.compare_initial_states --output-dir results --no-show`로 실행해도 됩니다. 그래프 창을 띄우려면 `--no-show`를 빼세요.

## 결과

기본 설정(`40×40` 격자, `J=1`, 시드 42)으로 온도마다 두 초기조건을 각각 8회씩 2,000 sweep 돌렸습니다. 시작 에너지 `E/N`는 Ordered가 −2, Random은 0 근처였습니다.

| 온도 | 완화 추정 (sweep) | 후반부 평균 `E/N` 차이 |
| ---: | ---: | ---: |
| 1.0 | 기준 미충족 | 0.0470 |
| 1.5 | 1,009 | 0.0066 |
| 2.0 | 477 | 0.0030 |
| 2.27 | 422 | 0.0420 |
| 2.5 | 118 | 0.0074 |
| 3.0 | 24 | 0.0010 |
| 5.0 | 3 | 0.0024 |

후반부 차이는 sweep 1,500–2,000에서 초기조건별로 에너지를 평균한 뒤 그 차이의 절댓값을 낸 값입니다. `T=5`에서는 두 조건의 에너지가 금방 비슷해졌습니다. `T=2.27`에서는 422 sweep 부근에 평균이 잠시 가까워졌지만 후반부에도 차이가 남았습니다.

- [에너지 곡선](docs/energy_relaxation.png)에서 두 조건의 평균과 표준편차 띠를 봅니다.
- [평균 에너지 차이](docs/energy_gap.png)에서 온도마다 차이가 줄어드는 속도를 비교합니다.
- [후반부 에너지 분포](docs/late_energy_distribution.png)에서 두 조건의 경험적 누적분포가 얼마나 겹치는지 봅니다.

## 더 해보기

- 먼저 가볍게 돌려보려면 실행 명령 끝에 `--L 16 --runs 3 --max-sweeps 200 --max-workers 2`를 붙이세요. 기본 실험 결과를 보존하려면 `--output-dir results`를 `--output-dir quick_results`로 바꾸세요.
- 단일 시뮬레이션은 `uv run python main.py`로 실행합니다. 격자 크기와 온도는 `main.py` 상단에서 바꿀 수 있습니다.
- 1 sweep은 스핀 개수만큼 뒤집기를 시도한 단위입니다. 완화 추정은 연속한 두 10-sweep 구간에서 각각 평균 에너지 차이가 0.02 이하인 첫 구간의 시작점입니다. 이 값은 초기조건의 영향이 줄어드는 정도를 비교하는 지표이며, 이후에도 두 분포가 비슷한지는 후반부 그래프로 확인합니다.
