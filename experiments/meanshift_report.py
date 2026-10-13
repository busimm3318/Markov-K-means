"""Tables and figure of application study 4 (mean-shift segmentation built on the bilateral filter).

    python -m experiments.meanshift_seg report
Inputs: experiments/results/meanshift_rules.csv, meanshift_curves.csv, meanshift_runs.csv
(written by ``python -m experiments.meanshift_seg run``). Output: docs/figures/fig14_meanshift_seg.png
and the <!-- TABLE:meanshift_* --> blocks of docs/results.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analyze_markov_tail import FIG, GRID, INK, INK2, RESULTS, SERIES, SURFACE, _style, fill_report, md, plt

KERNEL_KO = {"flat": "평탄 커널", "gauss": "가우스 커널"}
FORM_KO = {"ms": "평균이동 필터링", "ibf": "반복 bilateral 필터"}
SETTING_KO = {"fine": "(hs, hr) = (7, 6.5)", "coarse": "(hs, hr) = (16, 19)"}


def rule_ko(rule, form):
    if rule == "full":
        return "끝까지 (기준)"
    if rule == "edison":
        return "EDISON식 정지 (점별 이동 < 0.1, 최대 100회)"
    if rule.startswith("eps_"):
        e = rule.split("_")[1]
        base = f"점별 이동 < {e}" if form == "ms" else f"모든 화소 변화 < {e}·hr"
        return base + (" + AR(1) 외삽" if rule.endswith("_ar1") else "")
    if rule.startswith("T"):
        t = int(rule[1:])
        return f"고정 {t}회" + (" (OpenCV 기본값)" if t == 5 and form == "ms" else "")
    return {"mk_stop": "Markov-K-means 정지만 (감시 비용 포함)",
            "mk": "**Markov-K-means** (정지 + 의사 대표점 + 라벨 연쇄)",
            "mk_free": "Markov-K-means, 감시 비용 0으로 가정"}[rule]


def _order(rules, form):
    eps = sorted({r for r in rules if r.startswith("eps_") and not r.endswith("_ar1")}, key=lambda r: -float(r.split("_")[1]))
    if "edison" in rules:                  # the same rule as eps_0.1 with a cap of 100 iterations
        eps = [r for r in eps if r != "eps_0.1"]
    ar = [r + "_ar1" for r in eps if r + "_ar1" in rules]
    ts = sorted({r for r in rules if r.startswith("T")}, key=lambda r: int(r[1:]))
    head = ["edison"] if "edison" in rules else []
    return head + eps + ar + ts + ["mk_stop", "mk", "mk_free", "full"]


def oracle(curves, target):
    """Per run: share of time and of iterations when the error of stopping falls to <= target for good."""
    out = []
    for key, c in curves.groupby(["image", "form", "kernel", "setting"]):
        c = c.sort_values("t")
        err = c.error.values
        ok = np.maximum.accumulate(err[::-1])[::-1] <= target       # error stays <= target from here on
        i = int(np.argmax(ok))
        out.append(dict(zip(["image", "form", "kernel", "setting"], key), time_frac=c.time_frac.values[i],
                        passes_frac=c.passes_frac.values[i]))
    return pd.DataFrame(out)


def tables(rules, curves, runs):
    T = {}
    rows = {}
    for (form, setting, kernel), r in runs.groupby(["form", "setting", "kernel"], sort=False):
        mk = rules[(rules.form == form) & (rules.setting == setting) & (rules.kernel == kernel) & (rules.rule == "mk")]
        per_iter = (r.mk_monitor_s / (r.t + 1)).median() * 1e3
        stop = mk.set_index("image").loc[r.image.values]
        filt_to_stop = stop.seconds.values - r.mk_monitor_s.values - r.seg_s.values - r.seconds.values
        after = (r.full_s.values - filt_to_stop) / np.maximum(r["T"].values - r.t.values, 1)
        rows[f"{FORM_KO[form]} · {KERNEL_KO[kernel]} · {SETTING_KO[setting]}"] = {
            "영상 수": len(r), "수렴 반복 (중앙값)": r["T"].median(), "수렴 반복 (범위)": f"{r['T'].min()}–{r['T'].max()}",
            "상한 도달": int((~r.converged.astype(bool)).sum()),
            "끝까지 필터링 (초, 중앙값)": r.full_s.median(), "분할 수 (중앙값)": r.segments_ref.median(),
            "감시기 (ms/반복)": per_iter, "Markov 정지 반복 (중앙값)": mk.iterations.median(),
            "정지 뒤 필터링 (ms/반복)": float(np.median(after)) * 1e3}
    T["meanshift_setup"] = md(pd.DataFrame(rows).T, {
        "영상 수": "{:.0f}", "수렴 반복 (중앙값)": "{:.0f}", "수렴 반복 (범위)": "{}", "상한 도달": "{:.0f}",
        "끝까지 필터링 (초, 중앙값)": "{:.2f}", "분할 수 (중앙값)": "{:.0f}", "감시기 (ms/반복)": "{:.1f}",
        "Markov 정지 반복 (중앙값)": "{:.0f}", "정지 뒤 필터링 (ms/반복)": "{:.1f}"}, index_name="형태 · 커널 · 대역폭")

    for form, setting, key in (("ms", "fine", "meanshift_ms_fine"), ("ms", "coarse", "meanshift_ms_coarse"),
                               ("ibf", "fine", "meanshift_ibf")):
        d = rules[(rules.form == form) & (rules.setting == setting)]
        if d.empty:
            continue
        cols, fmt = {}, {}
        for kernel in ("flat", "gauss"):
            g = d[d.kernel == kernel].groupby("rule")
            if not len(g):
                continue
            k = "평탄" if kernel == "flat" else "가우스"
            cols[f"{k}: 시간"] = g.time_frac.median()
            cols[f"{k}: 반복"] = g.passes_frac.median()
            cols[f"{k}: 오차 (중앙값)"] = g.error.median()
            cols[f"{k}: 오차 (90%)"] = g.error.quantile(0.9)
            fmt.update({f"{k}: 시간": "{:.0%}", f"{k}: 반복": "{:.0%}", f"{k}: 오차 (중앙값)": "{:.2%}",
                        f"{k}: 오차 (90%)": "{:.2%}"})
        t = pd.DataFrame(cols)
        t = t.loc[[r for r in _order(set(t.index), form) if r in t.index]]
        t.index = [rule_ko(r, form) for r in t.index]
        T[key] = md(t, fmt, index_name="정지 규칙")

    rows = {}
    for (form, setting, kernel), r in runs.groupby(["form", "setting", "kernel"], sort=False):
        sel = (curves.form == form) & (curves.setting == setting) & (curves.kernel == kernel)
        o1, o01 = oracle(curves[sel], 0.01), oracle(curves[sel], 0.001)
        mk = rules[(rules.form == form) & (rules.setting == setting) & (rules.kernel == kernel)]
        free = mk[mk.rule == "mk_free"].set_index("image")
        rows[f"{FORM_KO[form]} · {KERNEL_KO[kernel]} · {SETTING_KO[setting]}"] = {
            "Markov 정지: 반복": free.passes_frac.median(),
            "Markov 정지: 이미 쓴 시간": free.time_frac.median(),
            "오차 ≤ 1% 유지: 반복": o1.passes_frac.median(), "오차 ≤ 1% 유지: 시간": o1.time_frac.median(),
            "오차 ≤ 0.1% 유지: 반복": o01.passes_frac.median(), "오차 ≤ 0.1% 유지: 시간": o01.time_frac.median()}
    T["meanshift_tail"] = md(pd.DataFrame(rows).T, "{:.0%}", index_name="형태 · 커널 · 대역폭")

    rows = {}
    for (form, setting, kernel), r in runs.groupby(["form", "setting", "kernel"], sort=False):
        reasons = r.reason.value_counts()
        rows[f"{FORM_KO[form]} · {KERNEL_KO[kernel]} · {SETTING_KO[setting]}"] = {
            "정지 사유": ", ".join(f"{k}×{v}" for k, v in reasons.items()),
            "정지 때 움직이는 점": r.moving_frac.median(), "미결정 U": r.unstable_frac.median(),
            "U 중 진동": r.oscillating_share.median(),
            "의사 대표점 채택": r.accepted_frac.mean(),
            "라벨 연쇄가 옮긴 화소 (합)": r.settled_moved.sum(),
            "감시 비용 / 끝까지 필터링": (r.mk_monitor_s / r.full_s).median(),
            "마무리 단계 / 끝까지 필터링": (r.seconds / r.full_s).median()}
    T["meanshift_mk"] = md(pd.DataFrame(rows).T, {
        "정지 사유": "{}", "정지 때 움직이는 점": "{:.2%}", "미결정 U": "{:.3%}", "U 중 진동": "{:.1%}",
        "의사 대표점 채택": "{:.0%}", "라벨 연쇄가 옮긴 화소 (합)": "{:,.0f}", "감시 비용 / 끝까지 필터링": "{:.0%}",
        "마무리 단계 / 끝까지 필터링": "{:.1%}"}, index_name="형태 · 커널 · 대역폭")
    return T


def _median_curve(c, grid):
    ys = [np.interp(grid, g.passes_frac.values, g.time_frac.values) for _, g in c.groupby("image")]
    return np.median(ys, 0)


def figure(rules, curves):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    styles = {"flat": ("-", "o"), "gauss": ("--", "s")}

    ax = axes[0]
    grid = np.linspace(0, 1, 201)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    ax.text(0.98, 0.52, "full passes: every pixel at every\niteration (iterated bilateral filter)", color=INK2,
            fontsize=8, ha="right", va="top")
    notes = []
    for kernel in ("flat", "gauss"):
        c = curves[(curves.form == "ms") & (curves.setting == "fine") & (curves.kernel == kernel)]
        if c.empty:
            continue
        y = _median_curve(c, grid)
        ls, mk = styles[kernel]
        ax.plot(grid, y, color=INK2, lw=2, ls=ls, label=f"mean-shift filtering, {kernel} kernel")
        f = rules[(rules.form == "ms") & (rules.setting == "fine") & (rules.kernel == kernel) & (rules.rule == "mk_free")]
        x0, y0 = f.passes_frac.median(), f.time_frac.median()     # medians of the actual stops, as in the tables
        ax.scatter([x0], [y0], color=SERIES[0], marker=mk, s=80, zorder=5, edgecolor=SURFACE, linewidth=1.5)
        notes.append(f"{kernel}: {x0:.0%} of the iterations, {y0:.0%} of the time")
    ax.scatter([], [], color=SERIES[0], marker="o", s=60, label="Markov-K-means stops (median)")
    ax.text(0.98, 0.14, "Markov-K-means stops after\n" + "\n".join(notes), color=INK, fontsize=8.5, ha="right",
            va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "Mean-shift filtering: where the time goes (median)", "share of the run's filtering time spent",
           "share of the run's iterations done")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0.22, 0.86))

    def plane(ax, form, families, title):
        d = rules[(rules.form == form) & (rules.setting == "fine")]
        for kernel in ("flat", "gauss"):
            k = d[d.kernel == kernel]
            if k.empty:
                continue
            g = k.groupby("rule").agg(t=("time_frac", "median"), e=("error", "median"))
            ls, mk = styles[kernel]
            for j, (name, members) in enumerate(families):
                members = [m for m in members if m in g.index]
                if not members:
                    continue
                pts = g.loc[members].sort_values("t")
                col = SERIES[j + 1]
                ax.plot(pts.t, pts.e.clip(lower=1e-5) * 100, color=col, lw=2, ls=ls, marker=mk, ms=6,
                        markeredgecolor=SURFACE, label=f"{name} ({kernel})")
            for rule, col, lab in (("mk", SERIES[0], "Markov-K-means"), ("mk_free", SERIES[0], None)):
                if rule in g.index:
                    face = col if rule == "mk" else SURFACE
                    ax.scatter([g.loc[rule, "t"]], [max(g.loc[rule, "e"], 1e-5) * 100], marker="D" if kernel == "gauss" else "o",
                               s=90, facecolor=face, edgecolor=col, linewidth=2, zorder=6,
                               label=f"{lab} ({kernel})" if lab else None)
        ax.axvline(1.0, color=GRID, lw=1.5)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        _style(ax, title, "pixels in another segment than at convergence (%, log; 0 drawn at 0.001)",
               "time / run to convergence (median; hollow: Markov-K-means with a free monitor)")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left", ncol=1)

    eps_ms = [r for r in ("edison", "eps_0.03", "eps_0.01", "eps_0.003")]
    t_ms = ["T5", "T10", "T20", "T50", "T100", "T200"]
    plane(axes[1], "ms", [("per-point shift < eps", eps_ms), ("fixed iterations", t_ms)],
          "Mean-shift filtering: time and error")
    eps_ibf = ["eps_0.1", "eps_0.03", "eps_0.01", "eps_0.003"]
    t_ibf = ["T5", "T10", "T20", "T50", "T100", "T200", "T500", "T1000"]
    plane(axes[2], "ibf", [("max colour change < eps·hr", eps_ibf), ("fixed iterations", t_ibf)],
          "Iterated bilateral filter: time and error")
    fig.tight_layout()
    fig.savefig(FIG / "fig14_meanshift_seg.png", dpi=150)
    plt.close(fig)


def report():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    rules = pd.read_csv(RESULTS / "meanshift_rules.csv", dtype={"image": str})
    curves = pd.read_csv(RESULTS / "meanshift_curves.csv", dtype={"image": str})
    runs = pd.read_csv(RESULTS / "meanshift_runs.csv", dtype={"image": str})
    figure(rules, curves)
    T = tables(rules, curves, runs)
    fill_report(T)
    for k, v in T.items():
        print(f"== {k}\n{v}\n")
    return T


if __name__ == "__main__":
    report()
