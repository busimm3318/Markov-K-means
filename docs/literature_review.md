# 선행연구 조사: 비수렴 벡터의 Markov 연쇄 기반 편입

연구계획서(`docs/research_plan.md`) 6절의 주제 A–H에 따른 주석 문헌 목록이다.

* **검증 방식:** 웹 검색으로 학회·학술지 페이지, Semantic Scholar, ResearchGate, PMLR, DBLP 계열 원문 페이지를 확인했다.
* **표기:** "검증"은 그렇게 확인한 서지 정보다. "미검증"은 기억에 의존한 정보이니 인용 전에 다시 확인해야 한다.
* **한계:** 이 환경에서는 arXiv 본문 접근(WebFetch)이 차단돼 있다. arXiv 논문은 검색 결과의 초록·서지로만 확인했다.

## 핵심 결론

| 구분 | 내용 |
|---|---|
| 가장 가까운 선행연구 | Pérez-Ortega 연구그룹의 "조기 종료 / 안정 객체 제외" 계열(H). 라벨이 바뀌는 점의 수가 임계값 아래로 떨어지면 멈추거나, 바뀌지 않는 점을 계산에서 뺀다. |
| 제안법과 공유하는 출발점 | 소수의 불안정 점만 남은 상태에서 전체 반복은 낭비다. |
| 제안법만의 차별점 | ① 멈춘 시점에 남은 불안정 점의 편입을 **점별 라벨 전이 이력의 Markov 연쇄 정상분포**로 정한다. ② 그 분포를 **연속적 소속도**로 해석한다. |
| 신규성 검색 결과 | 점별 라벨 궤적에서 전이행렬을 만들어 최종 편입·소속도를 정하는 K-means 연구는 찾지 못했다. 시도한 검색어는 label oscillation, assignment history, label trajectory Markov chain, iteration history membership 등이다. |
| 이론적 근거 | 할당 연쇄의 장기 점유율을 소속 확률로 읽는 해석은 베이지안 혼합모형의 Gibbs 샘플링(D)에서 표준이다. 다만 그곳의 연쇄는 사후분포를 목표로 설계된 확률적 연쇄다. 정확 Lloyd의 라벨 궤적은 결정적 동역학이라 같은 해석이 자동으로 성립하지 않는다(RQ3에서 검증). |

## A. Lloyd의 수렴과 반복 횟수 — "수렴하지 않는 점"이 가능한가

| 문헌 | 서지 | 요점 / 우리 연구와의 관계 | 상태 |
|---|---|---|---|
| Selim & Ismail | "K-Means-Type Algorithms: A Generalized Convergence Theorem and Characterization of Local Optimality", IEEE TPAMI 6(1):81–87, 1984 | K-means형 알고리즘의 **유한 수렴**을 엄밀히 증명했다. 정확 Lloyd에서 라벨이 영원히 진동하는 일은 없으므로, 문제는 "긴 꼬리"다(Exp 0과 일치). | 검증 |
| Arthur & Vassilvitskii | "How slow is the k-means method?", SoCG 2006 | 최악의 경우 반복 수가 2^Ω(√n)로 초다항이다. 꼬리가 길 수 있다는 이론적 근거다. | 검증 |
| Vattani | "k-means Requires Exponentially Many Iterations Even in the Plane", Discrete & Computational Geometry 45:596–616, 2011 | 평면에서도 2^Ω(n) 반복이 필요한 예가 있다. | 검증 |
| Arthur, Manthey & Röglin | "Smoothed Analysis of the k-Means Method", Journal of the ACM 58(5), 2011 | 평활 분석에서는 다항 반복이다. 실무에서 빠른 이유를 설명하지만, 상수와 꼬리 길이는 데이터에 따라 크다. | 검증 |
| Pokojovy, Jobe & Lacoste-Julien | "Lloyd's K-Means Clustering Algorithm Is Frank-Wolfe in Disguise", AISTATS 2026 | Lloyd = Frank-Wolfe의 특수형. 국소최소로 O(1/t) 비점근 수렴률을 보인다. 꼬리 동역학을 해석하는 최신 틀이다. | 검증(검색 초록) |

## B. 후반 반복 가속 — 제안법의 직접 경쟁자 (RQ1)

| 문헌 | 서지 | 요점 | 상태 |
|---|---|---|---|
| Hamerly | "Making k-means even faster", SIAM SDM 2010 | 점마다 상·하한 2개(O(n) 메모리)로 거리 계산을 건너뛴다. **본 실험의 M0′ 기준선.** 꼬리 반복은 이미 대부분의 점을 건너뛴다. | 검증 |
| Elkan | "Using the Triangle Inequality to Accelerate k-Means", ICML 2003 | 점×중심 하한 O(nk). n=10⁷에서는 메모리 때문에 비현실적이다. | 미검증(널리 알려진 서지) |
| Ding et al. | "Yinyang K-Means: A Drop-In Replacement of the Classic K-Means with Consistent Speedup", ICML 2015 | 중심 그룹 하한. Lloyd와 같은 결과를 보장한다. | 검증 |
| Xia et al. | "Ball k-means", arXiv 2005.00784 (TPAMI) | 공 구조로 "안정 영역" 점을 건너뛴다. 안정/불안정 점 구분의 기하학적 버전이다. | 검색 확인(학술지 서지 미검증) |

## C. 확률적·온라인 K-means — 지속 진동 체제

| 문헌 | 서지 | 요점 | 상태 |
|---|---|---|---|
| Sculley | "Web-Scale K-Means Clustering", WWW 2010 | mini-batch K-means(중심별 학습률 1/count). **본 실험의 mini-batch 체제.** | 검증 |
| Tang & Monteleoni | "Convergence Rate of Stochastic K-Means", AISTATS 2017 (PMLR 54:1495–1503) | 확률적 K-means가 국소최적으로 O(1/t)에 수렴한다. 라벨 진동 폭은 줄지만 0이 되는 시점은 보장되지 않는다. | 검증 |
| Schwartzman | "Mini-batch k-means terminates within O(d/ε) iterations", ICLR 2023 | 중심 이동 기준의 종료 보장(라벨 기준은 아님). | 검증(OpenReview) |

## D. Markov 연쇄 / MCMC와 군집화 — 연속 해석의 이론적 기반 (RQ3)

| 문헌 | 서지 | 요점 | 상태 |
|---|---|---|---|
| Diebolt & Robert | "Estimation of Finite Mixture Distributions Through Bayesian Sampling", JRSS B 56(2):363–375, 1994 | 혼합모형의 할당 z를 Gibbs 샘플링한다. 연쇄의 장기 점유율 = 사후 할당확률. **"정상분포를 소속도로 읽는다"의 표준 선례.** 단, 연쇄가 사후분포를 정상분포로 갖도록 설계돼 있다. | 검증 |
| Bachem, Lucic, Hassani & Krause | "Approximate K-Means++ in Sublinear Time", AAAI 2016 (K-MC²) | K-means와 Markov 연쇄의 기존 접점이다. MCMC는 **시딩**(D² 샘플링 근사)에 쓰이며, 꼬리 할당과는 무관하다. | 검증 |
| Stephens | "Dealing with label switching in mixture models", JRSS B 62(4), 2000 | MCMC 할당 표본의 라벨 교환 문제. 점별 연쇄를 군집 간에 비교할 때의 주의점이다. | 미검증 |
| van Dongen | Markov Cluster Algorithm (MCL), 박사논문 2000 | 이름만 비슷하고, 그래프 위 확률보행 기반의 전혀 다른 방법이다. | 미검증 |
| Anderson & Goodman | "Statistical inference about Markov chains", Ann. Math. Stat. 1957 | 전이행렬 최대우도 = 전이 횟수 비율. 짧은 이력의 추정 오차 근거다. | 미검증 |

## E. 연속적·모호 소속 — 연속 해석의 비교 대상

| 문헌 | 서지 | 요점 | 상태 |
|---|---|---|---|
| Lingras & West | "Interval set clustering of web users with rough k-means", J. Intelligent Information Systems 23:5–16, 2004 | 군집을 하·상 근사로 표현한다. 경계(fringe) 점을 여러 군집의 상근사에 동시에 둔다. **"비수렴 점을 한 군집에 억지로 넣지 않는다"는 발상의 가장 가까운 형태**지만, 근거가 이력이 아니라 거리 비율이다. | 검증 |
| Bezdek | *Pattern Recognition with Fuzzy Objective Function Algorithms*, Plenum 1981 | fuzzy c-means. 본 실험의 `fcm_soft` 비교군. | 미검증(표준 서지) |
| Masson & Denœux | "ECM: An evidential version of the fuzzy c-means algorithm", Pattern Recognition 41:1384–1397, 2008 | 신용(credal) 분할: 군집 집합에 질량을 부여한다. "0.5/0.5"를 명시적 모호성 질량으로 표현하는 틀이다. | 검증 |

## F. 군집 불확실성·안정성

| 문헌 | 서지 | 요점 | 상태 |
|---|---|---|---|
| Monti, Tamayo, Mesirov & Golub | "Consensus Clustering: A Resampling-Based Method for Class Discovery…", Machine Learning 52:91–118, 2003 | 재표본 반복 간 공동배정 빈도로 소속 안정성을 측정한다. 제안법은 **한 번의 실행 안에서 반복 간 빈도**를 쓴다는 점에서 저비용 변형으로 볼 수 있다. | 검증 |

## H. 신규성 확인 — 가장 가까운 선행연구 (Pérez-Ortega 계열)

| 문헌 | 서지 | 요점 | 상태 |
|---|---|---|---|
| Pérez O. et al. | "Improving the Efficiency and Efficacy of the K-means Clustering Algorithm Through a New Convergence Condition", ICCSA 2007, LNCS 4707 | "Early Stop K-means". 실행시간 약 2/100, 품질 손실 3% 미만이라고 보고했다. | 검증 |
| Mexicano et al. | "The early stop heuristic: A new convergence criterion for K-means", AIP Conf. Proc. 1738, 2016 (ICNAAM 2015) | 반복 간 개선이 작아지면 멈추는 휴리스틱. | 검증 |
| Mexicano et al. | "Identifying stable objects for accelerating the classification phase of k-means", 3PGCIC 2016 (Springer LNDECT) | 반복 사이에 같은 군집에 머무는 객체를 분류 단계 계산에서 제외한다. | 검증 |
| Pérez, Montes, Mexicano & Rodríguez | "Acceleration of the K-means algorithm by removing stable items", Int. J. Space-Based and Situated Computing 7(2):72–81, 2017 | 안정 항목 제거로 가속한다. | 검증 |
| Pérez-Ortega, Almanza-Ortega & Romero | "Balancing effort and benefit of K-means clustering algorithms in Big Data realms", PLoS ONE 13(9):e0201874, 2018 | **라벨을 바꾸는 객체 수가 임계값 미만이면 정지.** 임계값 약 0.03n에서 시간 약 4/100, 품질 손실 2% 미만. 본 연구의 "T₀에서 멈춤"과 같은 설정이다. 이 계열은 남은 불안정 점을 **그대로 둔다**(B1). | 검증 |

**시사점:** 본 연구의 B1(T₀에서 정지, 현재 라벨 유지)은 사실상 Pérez-Ortega 계열의 정지 규칙이다. 그래서 제안법(MK)의 기여는 다음 두 질문으로 판정된다.
1. 정지 시점의 불안정 점 처리를 Markov 연쇄로 바꿨을 때 B1보다 나은가?
2. 연속 소속도가 추가 정보를 주는가?

## 참고 링크 (검증에 사용)

- [Lloyd's K-Means Clustering Algorithm Is Frank-Wolfe in Disguise (arXiv)](https://arxiv.org/abs/2607.25190)
- [How slow is the k-means method? (Semantic Scholar)](https://www.semanticscholar.org/paper/How-slow-is-the-k-means-method-Arthur-Vassilvitskii/a68e042d017f075f085b3a72b22829bc2f15082b)
- [k-means Requires Exponentially Many Iterations Even in the Plane (Springer)](https://link.springer.com/article/10.1007/s00454-011-9340-1)
- [K-Means-Type Algorithms: A Generalized Convergence Theorem (Semantic Scholar)](https://www.semanticscholar.org/paper/K-Means-Type-Algorithms:-A-Generalized-Convergence-Selim-Ismail/c49ba2f8ed1ccda37f76f4f01624bce0768ef6ae)
- [Yinyang K-Means (ICML 2015 PDF)](https://research.csc.ncsu.edu/picture/publications/papers/icml15.pdf)
- [Ball k-means (arXiv)](https://arxiv.org/pdf/2005.00784)
- [Convergence Rate of Stochastic k-means (PMLR)](https://proceedings.mlr.press/v54/tang17b.html)
- [Mini-batch k-means terminates within O(d/ε) iterations (OpenReview)](https://openreview.net/pdf?id=jREF4bkfi_S)
- [Bayesian Modelling and Inference on Mixtures of Distributions (Marin, Mengersen & Robert)](https://www.ceremade.dauphine.fr/~xian/mixo.pdf)
- [Approximate K-Means++ in Sublinear Time (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/10259)
- [Rough clustering / rough k-means (Springer)](https://link.springer.com/chapter/10.1007/11908029_68)
- [ECM software page (Denœux)](https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/software/ecm)
- [Consensus Clustering (Semantic Scholar)](https://www.semanticscholar.org/paper/Consensus-Clustering:-A-Resampling-Based-Method-for-Monti-Tamayo/1f29553ecbaa388b6be3402bc7af28178f5e24ef)
- [Improving the Efficiency and Efficacy of the K-means … New Convergence Condition (Springer)](https://link.springer.com/chapter/10.1007/978-3-540-74484-9_58)
- [The early stop heuristic (AIP)](https://pubs.aip.org/aip/acp/article-abstract/1738/1/310003/884665/The-early-stop-heuristic-A-new-convergence?redirectedFrom=fulltext)
- [Identifying stable objects for accelerating the classification phase of k-means (Springer)](https://link.springer.com/chapter/10.1007/978-3-319-49109-7_88)
- [Acceleration of the K-means algorithm by removing stable items (ResearchGate)](https://www.researchgate.net/publication/320049070_Acceleration_of_the_K-means_algorithm_by_removing_stable_items)
- [Balancing effort and benefit of K-means clustering algorithms in Big Data realms (PLoS ONE)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0201874)
- [Hamerly 2010, SDM proceedings](https://dx.doi.org/10.1137/1.9781611972801.12)
- [Diebolt & Robert 1994 등 Gibbs 혼합모형 개관 (arXiv)](https://arxiv.org/pdf/0804.2413)
