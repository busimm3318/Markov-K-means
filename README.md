# Markov-K-means

K-means that stops once the only work left is a handful of boundary points changing cluster, and settles those points from their own label history.

[English](#english) · [한국어](#한국어)

```bash
pip install markov-kmeans
```

```python
from markov_kmeans import MarkovKMeans

mk = MarkovKMeans(n_clusters=50, random_state=0).fit(X)
mk.labels_                    # hard labels
mk.cluster_centers_           # centers
mk.unstable_indices_          # points still undecided when the run stopped
mk.membership_                # sparse (n, k): one-hot for settled points, a distribution for undecided ones
mk.stop_reason_, mk.regime_   # why it stopped; "drift" or "oscillation"
```

---

## English

### What it does

In a long K-means run, most iterations are spent moving a few points near cluster boundaries, long after everything else has settled. Markov-K-means keeps a short label history for every point and watches three things during the run:

- how many points still change cluster;
- whether the centers are still travelling or only jittering in place;
- whether the clusters are still reorganising or only exchanging members back and forth.

Once all three are quiet, it stops and settles the points that were still changing:

- a point that moved once and stayed keeps its new cluster;
- a point that keeps alternating gets the share of time it spent in each cluster as its soft membership, and the cluster it occupies most as its hard label.

### Time versus error, compared with other stopping rules

![Time needed for a target error, per stopping rule](https://raw.githubusercontent.com/busimm3318/Markov-K-means/main/docs/figures/fig9_time_for_error.png)

Each line is one way of deciding when to stop:

- Markov-K-means with its automatic rule;
- Markov-K-means stopped at a fixed iteration, then settled;
- scikit-learn's tolerance on center movement;
- the changed-labels threshold of Pérez-Ortega et al.;
- a fixed number of iterations.

Each procedure's own setting is swept, and every setting is applied unchanged to all datasets of a panel, as a user would choose it without knowing the outcome. A point gives the resulting average error (share of labels that differ from the reference partition) and the average time relative to running to the end. Lower and further left is better. Circles mark the default settings.

**Constant step or EMA (right).** This is where Markov-K-means pays off most. The centers never stop jittering, so boundary points flip forever and running longer does not reduce the error. The other rules cannot recognise this state:

- the center-tolerance rule, at its default, waits for a stillness that never comes and runs to the end;
- the changed-labels rule stops either too early or too late;
- a fixed budget picks an arbitrary moment of the oscillation.

Markov-K-means recognises the oscillation as stationary, stops, and gives each oscillating point the cluster it occupies most. Its curves lie under all others, and only its settlement reaches errors lower than the full run's.

**1/count mini-batch (middle).** As the step shrinks, points drift and then settle, so all procedures follow nearly the same curve. The automatic rule must first observe a full window of epochs. For a rough answer, a short fixed run is therefore cheaper. For a precise answer, the automatic rule is among the fastest.

**Exact Lloyd or Hamerly (left).** Again all adaptive rules lie on one curve: fewer changed labels and smaller center moves amount to the same thing here. A threshold tuned for the data is slightly faster, because Markov-K-means pays for monitoring and for its observation window. In return, its default lands at a low error without tuning, while the defaults of the other rules are scattered along the curve. It also returns the list of points that were still undecided.

In short, each unit of error costs Markov-K-means the least time under oscillating training. Under drift it costs the same as the other rules, and on exact Lloyd slightly more than a well-tuned threshold.

### Where it applies

- **Built-in base algorithms:**
  - `algorithm="hamerly"`: exact Lloyd with Hamerly bounds (the default);
  - `algorithm="lloyd"`;
  - `algorithm="minibatch"` with `learning_rate="count"`: Sculley's 1/count step, as in scikit-learn's `MiniBatchKMeans`;
  - `algorithm="minibatch"` with `learning_rate="constant"`: a fixed step, the same update as an EMA-trained codebook.
- **Any other iterative hard-assignment procedure.** Record the labels at each step and pass the history to `assign_from_history`. This covers online or streaming k-means, VQ codebook training, and k-medoids or k-modes style reassignment.

```python
from markov_kmeans import assign_from_history

r = assign_from_history(history)        # history: (n_steps, n_points), oldest first
r.labels, r.dense(n_clusters), r.uncertainty
```

### When it helps

- **Large data with a long convergence tail.** Each pass over the data is expensive, and the run spends many passes moving a few boundary points. The automatic rule cuts the tail without a threshold to tune. The answer differs from the converged one only on a small set of boundary points.
- **Constant-step, EMA or online training.** No other stopping rule knows when such a run is done, and no amount of extra training fixes the boundary points. Markov-K-means stops at the right time. Its labels are more accurate than the run's last state, and its memberships reflect how much time each point spends in each cluster.
- **You need to know which points are uncertain.** The undecided points come out as a list with soft memberships. You can send them to human review, to a more expensive model, or to a downstream step that accepts soft assignments.
- **One setting must work across datasets.** The good threshold of other rules depends on the data. The defaults of Markov-K-means stay near the best trade-off whether the run drifts or oscillates.

### When it hurts

- **Small or easy data.** If K-means converges quickly anyway, the monitor and the observation window are pure overhead, and the run is slightly slower than plain K-means.
- **Exact Lloyd with an already-tuned threshold.** A stopping threshold tuned for this kind of data is slightly cheaper for the same error.
- **Only aggregate quality matters.** Examples are colour palettes, inertia and compression quality. The aggregate is stable long before individual boundary points settle, and the automatic rule waits for the points, so it is conservative here. Fitting on a sample, or a short fixed run, is cheaper.
- **Codebooks that jitter strongly.** Codes may move a large part of the distance between neighbouring codes, as is common for EMA-trained VQ codebooks in image tokenizers. The rule then correctly declines to intervene, so monitoring costs time and saves nothing. For soft tokens, distance-based assignments are better than history-based ones there.
- **You need the exact converged partition.** For bit-exact reproducibility or comparison against a reference run, do not stop early.
- **A rough answer from 1/count mini-batch.** A short fixed number of epochs is cheaper than waiting for the observation window.

### Limits

- The rule only sees what has happened so far. A point that looks settled when the run stops but would move later is not caught, and most of the error left after a drift comes from such points. A smaller `unstable_tol` stops later and leaves fewer of them.
- Under drift, the gain comes from stopping; settlement keeps the labels the run already has. Accuracy beyond the run's own end comes only under oscillation.
- A fractional membership means two different things. After a drift it means "still undecided"; under oscillation it means "really shared". `regime_` tells which.
- New points get the nearest center from `predict`, since they have no history.

### How it works

1. The base algorithm runs. The last `window + 1` labels of every point are kept in a ring buffer.
2. After every iteration the monitor judges the whole run. Cheap checks run first; expensive ones run only when needed.
   - **Movement:** the share of points that changed within the window, and whether they switch back and forth.
   - **Representatives:** whether the centers travel or only jitter in place.
   - **Partitions:** whether cluster sizes change, and whether members flow one way or are exchanged.
3. When all are quiet, the run stops and the regime is named:
   - `drift`: points moved and are settling;
   - `oscillation`: points keep switching and their number no longer falls.
4. The undecided points are settled according to the regime:
   - under drift, a point that moved once keeps its label and an oscillating point gets its occupancy;
   - under oscillation, every point gets its occupancy.

   `assign_rule="markov"` instead uses the long-run law of each point's label Markov chain.

### Main parameters

| parameter | default | meaning |
|---|---|---|
| `n_clusters` | 8 | number of clusters |
| `algorithm` | `"hamerly"` | `"hamerly"`, `"lloyd"`, `"minibatch"` |
| `learning_rate`, `eta`, `batch_size` | `"count"`, 2e-4, 4096 | mini-batch step: `"count"` (1/count) or `"constant"` (fixed `eta`) |
| `stop_rule` | `"auto"` | `"auto"` judges the state; `"unstable"` looks only at the share of changing points |
| `assign_rule` | `"auto"` | `"auto"` (by regime), `"markov"`, `"frequency"`, `"keep"` |
| `t0` | `None` | stop at this iteration instead, then settle |
| `unstable_tol` | 1e-3 | lower stops later and leaves fewer undecided points |
| `window` | 10 | iterations of label history kept per point |
| `min_iter`, `max_iter` | 10, 300 | iteration bounds |
| `center_update` | `"auto"` | final centers: `"hard"`, `"soft"` (membership-weighted), `"none"` |

The monitor's other thresholds are documented in the docstring; the evaluation used their defaults. Diagnostics include `n_iter_`, `stop_reason_`, `regime_`, `decision_trace_`, `unstable_fraction_`, `n_dist_`, `fit_seconds_` and `labels_at_stop_`. `predict`, `transform`, `fit_predict` and `score` follow scikit-learn.

### Further reading

- [Evaluation report](https://github.com/busimm3318/Markov-K-means/blob/main/docs/results.md): methods, every number behind the figure, colour quantization and VQ codebook studies
- [Prior work](https://github.com/busimm3318/Markov-K-means/blob/main/docs/literature_review.md)
- [Quick start](https://github.com/busimm3318/Markov-K-means/blob/main/examples/quickstart.py)

---

## 한국어

### 무엇을 하는가

K-means를 오래 돌리면 반복의 대부분이 군집 경계의 소수 점을 옮기는 데 쓰인다. 나머지 점들은 이미 오래전에 자리를 잡은 상태다. Markov-K-means는 모든 점의 최근 라벨 이력을 짧게 기록하면서, 실행 중 세 가지를 지켜본다.

- 아직 군집을 바꾸는 점이 얼마나 남았는가
- 대표점이 아직 이동 중인가, 아니면 제자리에서 흔들리기만 하는가
- 군집이 재편되는 중인가, 아니면 구성원을 서로 주고받기만 하는가

셋이 모두 잠잠해지면 멈추고, 아직 바뀌던 점들을 각자의 이력으로 정리한다.

- 한 번 옮겨 가서 머문 점은 새 군집에 둔다.
- 계속 오가는 점은 각 군집에 머문 시간의 비율을 soft 소속도로, 가장 오래 머문 군집을 hard 라벨로 받는다.

### 다른 정지 규칙과 비교한 시간–오차 교환

![정지 규칙별 목표 오차에 드는 시간](https://raw.githubusercontent.com/busimm3318/Markov-K-means/main/docs/figures/fig9_time_for_error.png)

선 하나가 "언제 멈출지 정하는 방법" 하나다.

- Markov-K-means 자동 규칙
- 정해진 반복에서 멈춘 뒤 이력으로 정리하는 Markov-K-means
- 대표점 이동량으로 멈추는 scikit-learn의 tol
- 라벨이 바뀐 점의 비율로 멈추는 Pérez-Ortega 등의 규칙
- 고정 반복 수

각 방법의 설정값을 바꿔 가며 돌렸고, 설정 하나는 한 패널의 모든 데이터에 그대로 적용했다. 결과를 모르는 사용자가 설정을 고르는 상황을 그대로 재현한 것이다. 점의 위치는 그 설정의 평균 오차(기준 분할과 라벨이 다른 점의 비율)와 평균 시간(끝까지 돌린 시간 대비)이다. 아래·왼쪽일수록 같은 오차를 더 빨리 얻는다. 동그라미는 각 방법의 기본 설정이다.

**고정 학습률 또는 EMA (오른쪽).** Markov-K-means가 가장 큰 차이를 내는 경우다. 대표점이 끝까지 흔들리므로 경계 점들은 영원히 오가고, 더 오래 돌려도 오차가 줄지 않는다. 다른 규칙들은 이 상태를 알아보지 못한다.

- 대표점 이동량 규칙은 기본값에서 오지 않을 정지 상태를 기다리다 끝까지 돈다.
- 바뀐 라벨 비율 규칙은 너무 이르거나 너무 늦게 멈춘다.
- 고정 반복 수는 진동 중 아무 순간이나 집어 온다.

Markov-K-means는 이 진동이 더 줄지 않는 정상 상태임을 알아보고 멈춘 뒤, 오가는 점마다 가장 오래 머문 군집을 준다. 곡선이 다른 모든 방법보다 아래에 있고, 끝까지 돌린 결과보다 낮은 오차에 닿는 것은 이 정리뿐이다.

**1/count mini-batch (가운데).** 보폭이 줄면서 점들이 옮겨 갔다가 자리를 잡으므로, 모든 방법이 거의 같은 곡선 위에 놓인다. 자동 규칙은 먼저 관찰 창 동안 지켜봐야 한다. 그래서 대략적인 답만 필요하면 짧은 고정 반복이 더 싸다. 정밀한 답이 필요하면 자동 규칙이 가장 빠른 축에 든다.

**exact Lloyd 또는 Hamerly (왼쪽).** 여기서도 적응형 규칙들이 한 곡선 위에 있다. 바뀐 라벨이 줄어드는 것과 대표점 이동이 작아지는 것이 사실상 같은 일이기 때문이다. Markov-K-means는 감시 비용과 관찰 창 비용을 치르므로, 데이터에 맞게 조정한 임계값 규칙이 조금 더 빠르다. 대신 Markov-K-means는 조정 없이 기본값만으로 낮은 오차 쪽에 자리 잡는다. 다른 규칙의 기본값은 곡선 여기저기에 흩어져 있다. 또 끝까지 미결정이던 점의 목록도 함께 준다.

정리하면, 오차 한 단위를 줄이는 데 드는 시간은 진동하는 학습에서 Markov-K-means가 가장 적다. 표류 후 정착하는 학습에서는 다른 규칙과 같고, exact Lloyd에서 잘 조정된 임계값과 비교하면 조금 더 든다.

### 적용 가능한 알고리즘

- **내장 기반 알고리즘**
  - `algorithm="hamerly"`: Hamerly 경계를 쓰는 exact Lloyd (기본값)
  - `algorithm="lloyd"`
  - `algorithm="minibatch"`, `learning_rate="count"`: Sculley의 1/count 보폭. scikit-learn `MiniBatchKMeans`와 같은 방식이다.
  - `algorithm="minibatch"`, `learning_rate="constant"`: 고정 보폭. EMA로 학습하는 코드북과 같은 갱신이다.
- **그 밖의 반복형 hard 배정 절차 전반.** 매 단계의 라벨을 기록해 `assign_from_history`에 넘기면 된다. 온라인·스트리밍 k-means, VQ 코드북 학습, k-medoids·k-modes처럼 재배정을 반복하는 방법이 여기에 해당한다.

```python
from markov_kmeans import assign_from_history

r = assign_from_history(history)        # history: (단계 수, 점 수), 오래된 단계부터
r.labels, r.dense(n_clusters), r.uncertainty
```

### 이럴 때 쓰면 좋다

- **수렴 꼬리가 긴 대용량 데이터.** 데이터를 한 번 훑는 비용이 크고, 경계의 소수 점을 옮기느라 반복이 길어지는 경우다. 자동 규칙이 임계값 조정 없이 꼬리를 끊어 준다. 결과는 수렴한 결과와 경계의 소수 점에서만 다르다.
- **고정 학습률, EMA, 온라인 학습.** 다른 정지 규칙은 이런 학습이 언제 끝난 것인지 알지 못하고, 학습을 더 해도 경계 점은 바로잡히지 않는다. Markov-K-means는 적절한 시점에 멈춘다. 라벨은 마지막 상태보다 정확하고, 소속도는 각 점이 군집별로 머문 시간의 비율을 나타낸다.
- **어떤 점이 불확실한지 알아야 할 때.** 미결정 점이 soft 소속도와 함께 목록으로 나온다. 이 점들을 사람 검토, 더 비싼 모델, soft 배정을 받는 후속 단계로 바로 넘길 수 있다.
- **설정 하나로 여러 데이터를 처리해야 할 때.** 다른 규칙은 좋은 임계값이 데이터마다 달라진다. Markov-K-means의 기본값은 표류든 진동이든 최적 교환선 가까이에 놓인다.

### 이럴 때는 오히려 독이 된다

- **작거나 쉬운 데이터.** 어차피 빨리 수렴한다면 감시와 관찰 창은 순수한 추가 비용이 되고, 그냥 K-means보다 약간 느려진다.
- **임계값을 이미 조정해 둔 exact Lloyd.** 같은 종류의 데이터에 맞춘 정지 임계값이 있다면 같은 오차를 그쪽이 조금 더 싸게 얻는다.
- **전체 품질만 중요할 때.** 색 팔레트, inertia, 압축 화질 같은 경우다. 이런 집계 지표는 경계 점들이 자리 잡기 훨씬 전에 이미 안정된다. 자동 규칙은 점 하나하나가 자리 잡기를 기다리므로 여기서는 보수적이다. 표본으로 학습하거나 짧게 고정 반복하는 편이 싸다.
- **크게 흔들리는 코드북.** 코드가 이웃 코드와의 거리 중 상당 부분만큼 움직일 수 있다. 이미지 토크나이저의 EMA VQ 코드북이 흔히 그렇다. 이때는 규칙이 개입하지 않는 것이 옳은 판단이므로, 감시 비용만 들고 절감은 없다. soft 토큰이 필요하면 이력 기반보다 거리 기반 배정이 낫다.
- **정확히 수렴한 분할이 필요할 때.** 비트 단위 재현이나 기준 실행과의 대조가 목적이면 일찍 멈추면 안 된다.
- **1/count mini-batch로 대략적인 답만 필요할 때.** 관찰 창을 기다리는 것보다 짧은 고정 epoch가 싸다.

### 한계

- 판단은 지금까지 관찰한 것만으로 내린다. 멈출 때 자리 잡은 듯 보였다가 나중에 옮겨 갈 점은 잡지 못하고, 표류 뒤 남는 오차는 대부분 이런 점에서 나온다. `unstable_tol`을 낮추면 더 늦게 멈추는 대신 이런 점이 줄어든다.
- 표류 체제에서는 이득이 멈춤에서 나오고, 정리 단계는 이미 가진 라벨을 지킨다. 실행을 끝까지 돌린 결과보다 정확해지는 것은 진동 체제에서다.
- 분수형 소속도는 두 가지 뜻이 있다. 표류 뒤에는 "아직 미결정"이고, 진동 중에는 "실제로 나눠 가짐"이다. `regime_`으로 구분한다.
- 새 점은 이력이 없으므로 `predict`가 가장 가까운 대표점을 준다.

### 작동 방식

1. 기반 알고리즘을 돌린다. 모든 점의 최근 `window + 1`개 라벨은 링 버퍼에 저장한다.
2. 매 반복 뒤 감시기가 실행 전체를 판정한다. 싼 검사부터 하고, 비싼 검사는 필요할 때만 한다.
   - **움직임:** 창 안에서 라벨이 바뀐 점의 비율, 그리고 그 점들이 왔다 갔다 하는지
   - **대표점:** 대표점이 이동 중인지, 제자리에서 흔들리기만 하는지
   - **분할:** 군집 크기가 변하는지, 구성원이 한쪽으로 흐르는지 아니면 서로 주고받는지
3. 모두 잠잠해지면 멈추고 체제를 판정한다.
   - `drift`: 점들이 옮겨 가 자리를 잡는 중
   - `oscillation`: 점들이 계속 바뀌고 그 수가 더 줄지 않음
4. 미결정 점을 체제에 맞게 정리한다.
   - 표류에서는 한 번 옮긴 점은 라벨을 유지하고, 오가는 점은 머문 비율을 받는다.
   - 진동에서는 모든 점이 머문 비율을 받는다.

   `assign_rule="markov"`를 쓰면 대신 각 점의 라벨 Markov 연쇄의 장기 분포를 쓴다.

### 주요 매개변수

| 매개변수 | 기본값 | 의미 |
|---|---|---|
| `n_clusters` | 8 | 군집 수 |
| `algorithm` | `"hamerly"` | `"hamerly"`, `"lloyd"`, `"minibatch"` |
| `learning_rate`, `eta`, `batch_size` | `"count"`, 2e-4, 4096 | mini-batch 보폭: `"count"`(1/count) 또는 `"constant"`(고정 `eta`) |
| `stop_rule` | `"auto"` | `"auto"`는 상태를 판정하고, `"unstable"`은 바뀌는 점의 비율만 본다 |
| `assign_rule` | `"auto"` | `"auto"`(체제별), `"markov"`, `"frequency"`, `"keep"` |
| `t0` | `None` | 이 반복에서 멈춘 뒤 정리한다 |
| `unstable_tol` | 1e-3 | 낮출수록 늦게 멈추고 미결정 점이 줄어든다 |
| `window` | 10 | 점마다 보관하는 라벨 이력의 반복 수 |
| `min_iter`, `max_iter` | 10, 300 | 반복 수 범위 |
| `center_update` | `"auto"` | 최종 대표점: `"hard"`, `"soft"`(소속도 가중), `"none"` |

감시기의 나머지 임계값은 docstring에 정리되어 있고, 평가는 모두 기본값으로 했다. 진단 속성으로 `n_iter_`, `stop_reason_`, `regime_`, `decision_trace_`, `unstable_fraction_`, `n_dist_`, `fit_seconds_`, `labels_at_stop_`이 있다. `predict`, `transform`, `fit_predict`, `score`는 scikit-learn과 같다.

### 더 보기

- [평가 보고서](https://github.com/busimm3318/Markov-K-means/blob/main/docs/results.md): 방법, 그림 뒤의 모든 수치, 색 양자화·VQ 코드북 실험
- [선행연구](https://github.com/busimm3318/Markov-K-means/blob/main/docs/literature_review.md)
- [빠른 시작 예제](https://github.com/busimm3318/Markov-K-means/blob/main/examples/quickstart.py)

---

Apache-2.0
