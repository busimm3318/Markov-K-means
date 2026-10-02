# Markov-K-means

[English](#english) · [한국어](#한국어)

```bash
pip install markov-kmeans
```

```python
from markov_kmeans import MarkovKMeans

mk = MarkovKMeans(n_clusters=50, unstable_tol=1e-3, random_state=0).fit(X)
mk.labels_              # final hard labels
mk.cluster_centers_     # centers
mk.membership_          # (n, k) sparse memberships: one-hot for stable points,
                        # Markov long-run distribution for the few unstable ones
mk.unstable_indices_    # points settled by their Markov chains
mk.uncertainty_         # 1 - max long-run probability of each of them
```

---

## English

**Markov-K-means** is a K-means variant for the long tail of convergence. It stops iterating once only a few points still change cluster. Each remaining point is settled by the **long-run distribution π of the Markov chain of its own label history**: the hard label is the most likely cluster, and the soft membership is π itself. A point that alternated between two clusters gets `[0.5, 0.5]`; a point that moved once and stayed is absorbed in its new cluster.

### How it works

1. A base algorithm runs while the last `window + 1` labels of every point are kept in a ring buffer (n × (W+1), uint8). Choices:
   * `"hamerly"` (default): exact Lloyd iterates with Hamerly bounds
   * `"lloyd"`
   * `"minibatch"`: Sculley's mini-batch, with a 1/count or constant step
2. The **unstable set U** holds the points whose label changed within the last `window` iterations. The run stops at the first of:
   * |U|/n ≤ `unstable_tol`
   * iteration `t0`
   * convergence or `max_iter`
3. For every point of U, a transition matrix is built from its observed moves, optionally smoothed by `alpha` or shrunk towards pooled moves by `pooled_prior`. Its Cesàro long-run law, started from the current label, gives π.
4. The hard label is argmax π, with ties kept at the current label. Centers are updated by `center_update`: `"hard"`, `"soft"` (π-weighted) or `"none"`.

The Markov step alone also works on label histories from any iterative clustering:

```python
from markov_kmeans import assign_from_history

r = assign_from_history(history)        # history: (n_steps, m), oldest first
r.labels, r.dense(n_clusters), r.uncertainty
```

### What the evaluation found

Instances: n = 10⁷ points, each needing ≥ 100 iterations to converge. At every cut-off T₀ = 10, 20, …, 90, all methods started from the same state. Full report: [docs/results.md](https://github.com/busimm3318/Markov-K-means/blob/main/docs/results.md).

* **Compute.**
  * Stopping when few points move saves 81–98% of distance evaluations versus running Lloyd to convergence.
  * Against Hamerly-accelerated Lloyd it saves 52–68% of time and 16–38% of distances.
  * The saving comes from *stopping*. The Markov step costs about one Hamerly iteration.
* **Accuracy.**
  * The Markov step avoids the failure of majority voting, which pulls points that already moved back to their old cluster.
  * It is not more accurate than keeping the current labels.
  * Over 90% of the remaining error comes from points that look stable at T₀ and move later.
* **Soft reading.**
  * Fractional π occurs for 0.5–4% of U.
  * Under exact Lloyd and 1/count mini-batch, such points almost always end in one cluster.
  * So read π as "not yet decided" rather than "shared membership". As an uncertainty flag it beats a GMM posterior late in mini-batch runs (AUROC 0.76 vs 0.64).

Use Markov-K-means as a principled stopping rule plus an uncertainty flag for the last unstable points, not as an accuracy improvement.

### Main parameters

| parameter | default | meaning |
|---|---|---|
| `n_clusters` | 8 | number of clusters |
| `algorithm` | `"hamerly"` | `"hamerly"`, `"lloyd"`, `"minibatch"` |
| `init` | `"k-means++"` | `"k-means++"`, `"random"` or a (k, d) array |
| `t0` | `None` | fixed stopping iteration (`None`: adaptive rule) |
| `unstable_tol` | `1e-3` | stop once \|U\|/n ≤ this |
| `min_iter`, `max_iter` | 10, 300 | iteration bounds |
| `window` | 10 | label window W |
| `alpha` | 0.0 | transition-count smoothing (0 keeps absorbing states) |
| `pooled_prior` | 0.0 | strength of the pooled empirical-Bayes prior |
| `center_update` | `"auto"` | `"hard"`, `"soft"`, `"none"` (auto: hard for Lloyd/Hamerly, none for mini-batch) |
| `batch_size`, `learning_rate`, `eta` | 4096, `"count"`, 2e-4 | mini-batch settings |

Diagnostics: `n_iter_`, `stop_reason_` (`converged` / `few_unstable` / `t0` / `max_iter`), `unstable_fraction_`, `n_dist_` (point–center distance evaluations), `fit_seconds_`, `labels_at_stop_`. `predict`, `transform`, `fit_predict` and `score` follow scikit-learn; `predict` returns the nearest center, because new points have no label history.

---

## 한국어

**Markov-K-means**는 K-means의 긴 수렴 꼬리를 줄이는 변형이에요. 군집 주소가 아직 바뀌는 점이 극소수만 남으면 반복을 멈추고, 남은 점은 **각자의 라벨 이력으로 만든 Markov 연쇄의 장기 분포 π**로 편입해요.
* hard 라벨: 장기 확률이 가장 큰 군집
* soft 소속도: π 자체. 두 군집을 번갈아 오간 점은 `[0.5, 0.5]`, 한 번 옮겨 가서 머문 점은 새 군집에 1을 받아요.

### 설치

```bash
pip install markov-kmeans
# 개발판: pip install "git+https://github.com/busimm3318/Markov-K-means"
```

의존성은 NumPy, SciPy, numba, scikit-learn이에요.

### 알고리즘

1. 기반 알고리즘을 돌리면서 모든 점의 최근 `window + 1`개 라벨을 링 버퍼에 저장한다.
   * `"hamerly"`(기본): Lloyd와 결과가 같은 정확 가속법
   * `"lloyd"`
   * `"minibatch"`
2. **비수렴 집합 U**는 최근 `window`회 안에 라벨이 바뀐 점이다. 다음 중 하나가 되면 멈춘다.
   * \|U\|/N ≤ `unstable_tol`
   * `t0` 도달
   * 수렴 또는 `max_iter` 도달
3. U의 각 점에 대해 방문한 군집 사이의 전이 횟수로 전이행렬을 만든다. `alpha`로 평활하거나 `pooled_prior`로 공유 사전분포를 줄 수 있다.
4. 현재 라벨에서 출발한 Cesàro 장기 분포 π를 구한다. hard 라벨은 argmax π다. 대표점은 `center_update`(`"hard"` / `"soft"` / `"none"`)로 갱신한다.

`assign_from_history`로 Markov 편입만 따로 쓸 수 있어요. 다른 반복형 군집 알고리즘의 라벨 이력에도 적용돼요. 전체 예제는 [`examples/quickstart.py`](https://github.com/busimm3318/Markov-K-means/blob/main/examples/quickstart.py)에 있어요.

### 검증 결과 요약

N=10⁷, 수렴까지 100회 이상 걸리는 사례에서 T₀=10, 20, …, 90마다 같은 상태에서 비교했어요. 자세한 내용은 [docs/results.md](https://github.com/busimm3318/Markov-K-means/blob/main/docs/results.md)와 [HTML 보고서](https://github.com/busimm3318/Markov-K-means/blob/main/docs/report.html)에 있어요.

| 질문 | 결과 |
|---|---|
| 계산량 | Lloyd를 끝까지 돌리는 것보다 거리 계산을 81–98% 줄인다. Hamerly 대비로는 시간 52–68%, 거리 16–38%를 줄인다. 절감은 **멈춤**에서 나오고, Markov 편입은 Hamerly 반복 약 1회 비용만 더한다. |
| 오차 | Markov 편입은 다수결의 함정(이미 옮긴 점을 되돌림)을 피하지만, "현재 라벨 유지"보다 정확하지는 않다. 남은 오차의 90% 이상은 U 밖에서 앞으로 움직일 점에서 나온다. |
| 연속 해석 | π가 분수인 점은 U의 0.5–4%다. 그런 점도 대부분 결국 한 군집에 정착하므로, π는 "소속 비율"보다 "아직 모름"의 신호다. mini-batch 후반에는 틀릴 점 탐지에서 GMM 사후확률보다 낫다(AUROC 0.76 대 0.64). |

**권장 사용법:** 극소수 비수렴 점 때문에 반복을 계속하는 비용을 줄이는 정지 규칙, 그리고 그 점들의 불확실성 표시로 쓴다. 정확도를 높이는 방법으로 기대하면 안 된다.

### 저장소 구성

| 경로 | 내용 |
|---|---|
| `markov_kmeans/` | 배포 패키지(PyPI `markov-kmeans`) |
| `kmeans_accel/` | 연구용 커널과 기준선(저장소 전용): Lloyd, Hamerly, mini-batch, 꼬리 처리 전략 |
| `experiments/` | T₀ 스윕, 점별 연구, 집계·그림, 실행 스크립트 |
| `docs/` | 연구 계획, 선행연구, 결과 보고서, 그림·표 |
| `tests/` | Lloyd 동등성, Markov 편입, 패키지 API 테스트 |

```bash
pip install -e ".[dev]"
python -m pytest -q                        # 테스트
./experiments/run_markov_tail.sh all       # 전체 실험 재현 (4 CPU 기준 수 시간)
python -m experiments.analyze_markov_tail  # 그림·표·보고서 갱신
```

### 라이선스

Apache-2.0
