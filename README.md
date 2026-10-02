# Markov-K-means

**Markov-K-means**는 K-means 변형이에요. 군집 주소가 아직 바뀌는 점이 극소수만 남으면 반복을 멈추고, 남은 점은 **각자의 라벨 이력으로 만든 Markov 연쇄의 장기 분포 π**로 편입해요.
* hard 라벨: 장기 확률이 가장 큰 군집
* soft 소속도: 분포 그 자체. 예를 들어 두 군집을 번갈아 오간 점은 `[0.5, 0.5]`

*Markov-K-means stops K-means once only a few points still change cluster and settles them with the long-run law of the Markov chain of their own label history.*

```python
from markov_kmeans import MarkovKMeans

mk = MarkovKMeans(n_clusters=50, unstable_tol=1e-3, random_state=0).fit(X)
mk.labels_              # 최종 hard 라벨
mk.cluster_centers_     # 대표점
mk.membership_          # (n, k) 희소 소속도: 안정점은 0/1, 비수렴 점은 Markov 장기 분포
mk.unstable_indices_    # Markov로 편입된 점들
mk.uncertainty_         # 각 비수렴 점의 1 − max π
```

## 설치

```bash
git clone https://github.com/busimm3318/Markov-K-means.git
cd Markov-K-means
pip install -e .            # 실험·테스트까지: pip install -e .[dev]
```

* 의존성: NumPy, SciPy, numba, scikit-learn.
* 패키지 이름은 `markov-kmeans`이고 `import markov_kmeans`로 불러요.

## 알고리즘

1. 기반 알고리즘을 돌리면서 모든 점의 최근 `window + 1`개 라벨을 링 버퍼(N × (W+1), uint8)에 저장한다.
   * `"hamerly"`(기본): Lloyd와 결과가 비트 단위로 같은 정확 가속법
   * `"lloyd"`
   * `"minibatch"`: Sculley 방식. 학습률은 1/count 또는 고정 η
2. **비수렴 집합 U**는 최근 `window`회 안에 라벨이 바뀐 점이다. 다음 중 하나가 되면 멈춘다.
   * \|U\|/N ≤ `unstable_tol`
   * 지정한 `t0`에 도달
   * 수렴 또는 `max_iter` 도달
3. U의 각 점에 대해, 방문한 군집 사이의 전이 횟수로 전이행렬을 만든다.
   * Dirichlet 평활: `alpha`
   * 전체 이동으로 만든 공유 사전분포: `pooled_prior`
4. 현재 라벨에서 출발한 연쇄의 **Cesàro 장기 분포 π**를 구한다. 게으른 연쇄 (I+P)/2를 반복 제곱해서 계산한다.
   * 흡수형 이력 `A A A B B` → `[0, 1]`
   * 진동 `A B A B` → `[0.5, 0.5]`
5. hard 라벨은 argmax π로 정한다. 동점이면 현재 라벨을 택한다. 대표점 갱신 방식은 `center_update`로 고른다: `"hard"` / `"soft"`(π 가중) / `"none"`.

Markov 단계만 따로 쓸 수도 있어요. 다른 반복형 군집 알고리즘의 라벨 이력에도 적용돼요.

```python
from markov_kmeans import assign_from_history

r = assign_from_history(history)   # history: (반복 수, 점 수), 오래된 것부터
r.labels, r.dense(n_clusters), r.uncertainty
```

전체 예제는 `examples/quickstart.py`에 있어요.

## 주요 매개변수와 속성

| 매개변수 | 기본값 | 의미 |
|---|---|---|
| `n_clusters` | 8 | 군집 수 k |
| `algorithm` | `"hamerly"` | `"hamerly"`, `"lloyd"`, `"minibatch"` |
| `init` | `"k-means++"` | `"k-means++"`, `"random"`, 또는 (k, d) 배열 |
| `t0` | `None` | 고정 정지 반복. `None`이면 `unstable_tol` 규칙 사용 |
| `unstable_tol` | `1e-3` | \|U\|/N이 이 값 이하가 되면 정지 |
| `min_iter`, `max_iter` | 10, 300 | 반복 하한·상한 |
| `window` | 10 | 라벨 창 길이 W |
| `alpha` | 0.0 | 전이 횟수 평활. 0이면 흡수 상태를 보존 |
| `pooled_prior` | 0.0 | 공유 사전분포 세기 |
| `center_update` | `"auto"` | `"hard"`, `"soft"`, `"none"`. auto는 Lloyd/Hamerly→hard, mini-batch→none |
| `batch_size`, `learning_rate`, `eta` | 4096, `"count"`, 2e-4 | mini-batch 설정 |

| 속성 | 의미 |
|---|---|
| `labels_`, `cluster_centers_`, `inertia_` | 결과 |
| `labels_at_stop_` | 정지 시점 기반 알고리즘의 라벨(Markov 편입 전) |
| `unstable_indices_`, `unstable_states_`, `unstable_probs_`, `uncertainty_` | U와 각 점의 방문 군집·장기 확률 |
| `membership_` | (n, k) SciPy 희소 행렬 |
| `n_iter_`, `stop_reason_`, `unstable_fraction_`, `n_dist_`, `fit_seconds_` | 진단: 정지 이유는 `converged` / `few_unstable` / `t0` / `max_iter`, `n_dist_`는 거리 계산 횟수 |

`predict`, `transform`, `fit_predict`, `score`는 scikit-learn과 같은 의미예요. 새 점에는 라벨 이력이 없으니 `predict`는 가장 가까운 대표점을 돌려줘요.

## 무엇이 검증되었나 (요약)

N=10⁷, 수렴까지 100회 이상 걸리는 사례에서 T₀=10, 20, …, 90마다 다른 방법들과 같은 상태에서 비교했어요. 자세한 내용은 [`docs/results.md`](docs/results.md)와 [`docs/report.html`](docs/report.html)에 있어요.

| 질문 | 결과 |
|---|---|
| 계산량 | 극소수만 남았을 때 멈추면 Lloyd를 끝까지 돌리는 것보다 거리 계산을 81–98% 줄인다. Hamerly 대비로는 시간 52–68%, 거리 16–38%를 줄인다. 이 절감은 **멈춤**에서 나오고, Markov 편입은 Hamerly 반복 약 1회 비용만 더한다. |
| 오차 | Markov 편입은 다수결(이미 옮긴 점을 되돌려 U의 43–53% 오류)의 함정을 피하지만, "현재 라벨 유지"보다 정확하지는 않다. 남은 오차의 90% 이상은 U 밖에서 **앞으로 움직일** 점에서 나온다. |
| 연속 해석 | π가 분수인 점은 U의 0.5–4%뿐이다. 일반 Lloyd와 1/count mini-batch에서는 그런 점도 결국 한 군집에 정착한다. 그래서 π는 "소속 비율"보다 "**어디로 갈지 아직 모름**"의 신호로 읽어야 한다. mini-batch 후반에는 틀릴 점을 가려내는 신호로 GMM 사후확률보다 낫다(AUROC 0.76 대 0.64). 고정 학습률(영구 진동) 체제 평가는 진행 중이다. |

**권장 사용법:**
* 극소수 비수렴 점 때문에 반복을 계속하는 비용을 줄이는, 원칙 있는 정지 규칙으로 쓴다.
* 그 점들의 불확실성 표시(`uncertainty_`, `membership_`)로 쓴다.
* 정확도를 높이는 방법으로 기대하면 안 된다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `markov_kmeans/` | 배포용 패키지: `MarkovKMeans`, `assign_from_history` |
| `kmeans_accel/` | 연구용 커널과 기준선: Lloyd, Hamerly, mini-batch, 꼬리 처리 전략, Markov 편입 핵심부 |
| `experiments/` | T₀ 스윕(`markov_tail.py`), 점별 연구(`oscillators.py`), 집계·그림(`analyze_markov_tail.py`), 실행 스크립트 |
| `docs/` | 연구 계획, 선행연구, 결과 보고서(Markdown·HTML), 그림·표 |
| `tests/` | Lloyd 동등성, Markov 편입, 패키지 API 테스트 |

```bash
pytest -q                                  # 테스트
./experiments/run_markov_tail.sh all       # 전체 실험 재현 (4 CPU 기준 수 시간)
python -m experiments.analyze_markov_tail  # 그림·표·보고서 갱신
```

## 라이선스

Apache-2.0 (`LICENSE`).
