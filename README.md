# Markov-K-means

[English](#english) · [한국어](#한국어)

```bash
pip install markov-kmeans
```

```python
from markov_kmeans import MarkovKMeans

mk = MarkovKMeans(n_clusters=50, random_state=0).fit(X)
mk.labels_              # final hard labels
mk.cluster_centers_     # centers
mk.stop_reason_         # converged / few_unstable / stable_drift / oscillation / t0 / max_iter
mk.regime_              # "drift" or "oscillation": how the last moving points behaved
mk.membership_          # (n, k) sparse memberships: one-hot for stable points,
                        # a distribution over clusters for the few unstable ones
mk.unstable_indices_    # the points settled from their label histories
mk.uncertainty_         # 1 - max probability of each of them
mk.decision_trace_      # the state judged at every iteration
```

---

## English

**Markov-K-means** is a K-means variant for the long tail of convergence. It watches the state of the whole run and stops once only a few points still change cluster while the representatives and the partition are stable. Each remaining point is then settled from **its own label history**:
* a point that moved once and stayed is absorbed in its new cluster: the long-run law of the Markov chain of its labels;
* a point that keeps alternating between clusters gets its occupancy, e.g. `[0.5, 0.5]`.

The hard label is the most likely cluster; the soft membership is the whole distribution.

### How it works

1. A base algorithm runs while the last `window + 1` labels of every point are kept in a ring buffer (n × (W+1), uint8 or uint16). Choices:
   * `"hamerly"` (default): exact Lloyd iterates with Hamerly bounds
   * `"lloyd"`
   * `"minibatch"`: Sculley's mini-batch, with a 1/count or constant step (a constant step behaves like an EMA-trained VQ codebook)
2. After every iteration a state monitor judges the whole run (`stop_rule="auto"`):

   | group | signals | stable when |
   |---|---|---|
   | movement | share of points that changed within the last `window` iterations (the unstable set U); share of U that switched twice or more; changes still to come, extrapolated from the decay of the change counts | U ≤ `max_unstable` |
   | representatives | largest center move relative to half the distance to the nearest other center; `center_travel`, displacement over the window ÷ mean one-step move (≈1 for jitter around fixed points, ≈W for a drift) | moves ≤ `center_tol`, or at a noise floor ≤ `center_noise_tol` with `center_travel` ≤ `travel_ratio` |
   | partitions | relative change of cluster sizes; share of a cluster's members moving; `net_flow`, net ÷ gross transfers between cluster pairs (0 when moves balance out) | sizes steady and few members moving, or moves balanced |

   The Markov step intervenes when U is small and representatives and partitions are stable:
   * `oscillation`: the points of U keep switching and their number no longer falls
   * `stable_drift`: the changes still to come are negligible
   * `few_unstable`: |U|/n ≤ `unstable_tol`
   * the run also stops on convergence, at `t0`, or at `max_iter`
3. The points of U are settled according to the diagnosed regime (`assign_rule="auto"`):
   * drift: a point that switched once keeps its label, a point that oscillated gets its window occupancy
   * oscillation: every point gets its window occupancy

   `assign_rule="markov"` uses the long-run law of each point's label Markov chain for every point. It is optionally smoothed by `alpha`, or shrunk towards the pooled moves by `pooled_prior`.
4. Centers are updated by `center_update`: `"hard"`, `"soft"` (membership-weighted) or `"none"`.

The 0.1.0 behaviour is `stop_rule="unstable", assign_rule="markov"`.

The Markov step alone also works on label histories from any iterative clustering:

```python
from markov_kmeans import assign_from_history

r = assign_from_history(history)        # history: (n_steps, m), oldest first
r.labels, r.dense(n_clusters), r.uncertainty
```

### What the evaluation found

Full report: [docs/results.md](https://github.com/busimm3318/Markov-K-means/blob/main/docs/results.md).

**Synthetic instances:**
* n = 10⁷ (also d = 128, and n = 3×10⁷)
* each needs ≥ 100 iterations to converge
* at every cut-off T₀ = 10, 20, …, 90, all methods start from the same state

**Findings:**
* **Compute.** Stopping when few points move saves 81–98% of distance evaluations versus running Lloyd to convergence. Against Hamerly-accelerated Lloyd it saves 52–68% of time (73–84% at n = 3×10⁷). The saving comes from *stopping*; the Markov step costs about one Hamerly iteration.
* **Accuracy.**
  * After a drift (exact Lloyd, 1/count mini-batch), Markov settlement is as good as keeping the current labels. It avoids the failure of majority voting, which pulls points that already moved back to their old cluster (43–53% wrong).
  * Over 90% of the remaining error comes from points that look stable at T₀ and move later.
  * In a persistent oscillation (constant-step mini-batch), occupancy and Markov settlement beat keeping the current label (11–21% vs 23–30% wrong in U).
* **Soft reading.**
  * After a drift, fractional memberships are rare (0.2–9% of U) and mean "not yet decided": such points almost always end in one cluster. As a flag for points that will be wrong, they beat a GMM posterior in mini-batch runs (AUROC 0.71–0.98 vs 0.60–0.75).
  * In a persistent oscillation, 80% of these points really keep alternating, and π is calibrated to their long-run share (0.55 → 0.53, 0.85 → 0.83). Plain window frequency is as good.
* **Automatic rule (n = 10⁶, 10 instances).**
  * It uses 6–83% of the full run's time at an error of 0.05–0.62%.
  * With a constant step, scikit-learn's tolerance rule never stops; the automatic rule stops at 10–62% with an error close to the full run's.

* **Applications.**
  * Colour quantization of 100 photos (2–21 MP): image quality is within 0.01 dB of the converged result after about a third of the iterations. The automatic rule is safe there (≤ 0.002 dB lost) but conservative (79–84% of the time); fitting the palette on a sample is cheapest.
  * EMA VQ codebooks (MobileNetV2 latents, pixel patches; the image-tokenizer setting): 5–26% of the vectors keep switching codes and the codes jitter by up to 60% of their spacing. The rule correctly declines to intervene, and distance-based soft tokens beat history-based ones.

Use Markov-K-means as a principled stopping rule plus an uncertainty flag (or, under persistent oscillation, a soft membership) for the last unstable points. It is not an accuracy improvement over a converged run.

### Main parameters

| parameter | default | meaning |
|---|---|---|
| `n_clusters` | 8 | number of clusters |
| `algorithm` | `"hamerly"` | `"hamerly"`, `"lloyd"`, `"minibatch"` |
| `init` | `"k-means++"` | `"k-means++"`, `"random"` or a (k, d) array |
| `stop_rule` | `"auto"` | `"auto"` (judge the state) or `"unstable"` (only `unstable_tol`, as in 0.1.0) |
| `assign_rule` | `"auto"` | `"auto"` (by regime), `"markov"`, `"frequency"`, `"keep"` |
| `t0` | `None` | fixed stopping iteration (overrides `stop_rule`) |
| `unstable_tol`, `max_unstable` | 1e-3, 0.2 | stop once \|U\|/n ≤ `unstable_tol`; never intervene above `max_unstable` |
| `osc_share`, `plateau_ratio`, `change_tol` | 0.3, 0.8, 1e-3 | oscillation share, plateau test of the change counts, projected changes |
| `center_tol`, `center_noise_tol`, `travel_ratio` | 0.01, 0.25, 2 | representative stability |
| `cluster_tol`, `size_tol`, `flow_tol` | 0.1, 0.02, 0.1 | partition stability |
| `min_iter`, `max_iter` | 10, 300 | iteration bounds |
| `window` | 10 | label window W |
| `alpha`, `pooled_prior` | 0.0, 0.0 | Markov rule: transition smoothing, pooled empirical-Bayes prior |
| `center_update` | `"auto"` | `"hard"`, `"soft"`, `"none"` (auto: hard for Lloyd/Hamerly, none for mini-batch) |
| `batch_size`, `learning_rate`, `eta` | 4096, `"count"`, 2e-4 | mini-batch settings |

Diagnostics: `n_iter_`, `stop_reason_`, `regime_`, `decision_trace_`, `unstable_fraction_`, `oscillating_share_`, `change_counts_`, `center_drift_`, `n_dist_` (point–center distance evaluations), `fit_seconds_`, `labels_at_stop_`. `predict`, `transform`, `fit_predict` and `score` follow scikit-learn; `predict` returns the nearest center, because new points have no label history.

---

## 한국어

**Markov-K-means**는 K-means의 긴 수렴 꼬리를 줄이는 변형이에요. 전체 실행 상태를 지켜보다가, 군집 주소가 아직 바뀌는 점이 극소수만 남고 대표점과 분할이 안정되면 반복을 멈춰요. 남은 점은 **각자의 라벨 이력**으로 편입해요.
* 한 번 옮겨 가서 머문 점: 라벨 Markov 연쇄의 장기 분포에 따라 새 군집에 흡수
* 두 군집을 계속 오가는 점: 창 안 점유율, 예: `[0.5, 0.5]`

hard 라벨은 확률이 가장 큰 군집이고, soft 소속도는 그 분포 전체예요.

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
   * `"minibatch"`: 1/count 또는 상수 스텝. 상수 스텝은 EMA로 학습하는 VQ 코드북과 같은 구조다.
2. 매 반복 뒤 전체 상태를 판정한다(`stop_rule="auto"`).
   * **움직임:** 최근 `window`회 안에 라벨이 바뀐 점(비수렴 집합 U)의 비율, 그중 진동하는 점의 비율, 앞으로 남은 변경 수
   * **대표점:** 이동 거리 ÷ 가장 가까운 다른 대표점까지 거리의 절반. 그리고 `center_travel`(창 전체 변위 ÷ 평균 한 걸음)로 "제자리 흔들림"과 "표류"를 구분한다.
   * **분할:** 군집 크기 변화, 움직이는 구성원 비율, 그리고 `net_flow`(군집 쌍 사이 순 이동 ÷ 총 이동)로 "오가는 이동"과 "재편"을 구분한다.
3. U가 작고 대표점과 분할이 안정하면 Markov 단계가 개입한다.
   * `oscillation`: U의 점들이 계속 오가고 그 수가 더 줄지 않음
   * `stable_drift`: 남은 변경이 무시할 만함
   * `few_unstable`: \|U\|/N ≤ `unstable_tol`
   * 그 밖에 수렴, `t0`, `max_iter`에서도 멈춘다.
4. U의 점은 진단된 체제에 따라 편입한다(`assign_rule="auto"`).
   * 표류: 한 번만 바뀐 점은 현재 라벨 유지, 진동한 점은 점유율
   * 진동: 모든 점을 점유율로
5. 대표점은 `center_update`(`"hard"` / `"soft"` / `"none"`)로 갱신한다.

0.1.0 동작은 `stop_rule="unstable", assign_rule="markov"`로 쓸 수 있어요. `assign_from_history`로 Markov 편입만 따로 쓸 수도 있어요. 다른 반복형 군집 알고리즘의 라벨 이력에도 적용돼요. 전체 예제는 [`examples/quickstart.py`](https://github.com/busimm3318/Markov-K-means/blob/main/examples/quickstart.py)에 있어요.

### 검증 결과 요약

수렴까지 100회 이상 걸리는 합성 사례(N=10⁷, d=128, N=3×10⁷)에서 T₀=10, 20, …, 90마다 같은 상태에서 비교했어요. 자세한 내용은 [docs/results.md](https://github.com/busimm3318/Markov-K-means/blob/main/docs/results.md)와 [HTML 보고서](https://github.com/busimm3318/Markov-K-means/blob/main/docs/report.html)에 있어요.

| 질문 | 결과 |
|---|---|
| 계산량 | Lloyd를 끝까지 돌리는 것보다 거리 계산을 81–98% 줄인다. Hamerly 대비 시간 52–68%(N=3×10⁷에서 73–84%)를 줄인다. 절감은 **멈춤**에서 나오고, Markov 편입은 Hamerly 반복 약 1회 비용만 더한다. |
| 오차 | 표류 체제에서 Markov 편입은 "현재 라벨 유지"와 같은 수준이고, 다수결의 함정(이미 옮긴 점을 되돌림, 43–53% 오류)을 피한다. 남은 오차의 90% 이상은 U 밖에서 나온다. 영구 진동 체제에서는 점유율·Markov 편입이 유지보다 낫다(U 오류 11–21% 대 23–30%). |
| 연속 해석 | 표류 체제에서 분수형 π는 드물고(U의 0.2–9%) "아직 미결정"의 신호다. mini-batch에서는 틀릴 점 탐지에서 GMM 사후보다 낫다(AUROC 0.71–0.98 대 0.60–0.75). 영구 진동 체제에서는 그런 점의 80%가 실제로 계속 오가고, π가 장기 점유율에 맞게 보정돼 있다(0.55→0.53, 0.85→0.83). |
| 응용 | 색 양자화(사진 100장)에서는 반복의 약 1/3 시점에 화질이 이미 끝까지 돌린 결과와 0.01 dB 안으로 들어온다. 자동 규칙은 안전하지만(손실 0.002 dB 이하) 보수적이고(시간 79–84%), 표본 학습이 가장 싸다. EMA VQ 코드북(이미지 토크나이저)에서는 벡터의 5–26%가 계속 코드를 바꾸고 코드가 크게 흔들려 자동 규칙이 개입하지 않았다. soft 토큰은 거리 기반이 이력 기반보다 낫다. |
| 자동 판단 | N=10⁶ 10개 사례에서 끝까지 돌린 시간의 6–83%로 오류율 0.05–0.62%를 낸다. 상수 스텝에서는 sklearn식 규칙이 끝내 멈추지 못하지만, 자동 규칙은 10–62% 시점에서 멈추고 끝까지 돌린 결과와 비슷한 오류를 낸다. |

**권장 사용법:** 극소수 비수렴 점 때문에 반복을 계속하는 비용을 줄이는 정지 규칙, 그리고 그 점들의 불확실성 표시로 쓴다. 영구 진동 체제에서는 소속 비율로도 쓴다. 끝까지 수렴한 결과보다 정확해지는 방법으로 기대하면 안 된다.

### 저장소 구성

| 경로 | 내용 |
|---|---|
| `markov_kmeans/` | 배포 패키지(PyPI `markov-kmeans`) |
| `kmeans_accel/` | 연구용 커널과 기준선(저장소 전용): Lloyd, Hamerly, mini-batch, 꼬리 처리 전략 |
| `experiments/` | T₀ 스윕, 점별 연구, 자동 규칙 평가, 응용 실험(색 양자화, VQ 코드북), 집계·그림 |
| `docs/` | 연구 계획, 선행연구, 결과 보고서, 그림·표 |
| `tests/` | Lloyd 동등성, 판단 규칙, Markov 편입, 패키지 API 테스트 |

```bash
pip install -e ".[dev]"
python -m pytest -q                        # 테스트
./experiments/run_markov_tail.sh all       # T₀ 스윕 재현 (4 CPU 기준 수 시간)
python -m experiments.auto_rule            # 자동 판단 규칙 평가
./experiments/run_applications.sh          # 응용 실험
python -m experiments.analyze_markov_tail  # 그림·표·보고서 갱신
python -m experiments.analyze_applications
```

### 라이선스

Apache-2.0
