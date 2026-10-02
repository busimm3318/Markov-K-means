"""Tables and figures for the automatic rule and the two application studies (docs/results.md).

    python -m experiments.analyze_applications
Inputs: experiments/results/auto_rule.csv (+ auto_rule_before_travel.csv), color_quant.csv,
color_quant_curve.csv, vq_codebook.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analyze_markov_tail import (FIG, GRID, INK2, MARKERS, RESULTS, SERIES, SURFACE, _style, fill_report,
                                             md, plt)

RNAME = {"exact": "정확 Lloyd", "minibatch": "mini-batch 1/count", "minibatch_const": "mini-batch 상수 스텝"}


def _read(name):
    f = RESULTS / name
    return pd.read_csv(f) if f.exists() else None


# ----------------------------------------------------------------------------- automatic rule

def auto_rule_tables():
    new, old = _read("auto_rule.csv"), _read("auto_rule_before_travel.csv")
    if new is None:
        return {}
    new = new[new.n == 1_000_000].copy()
    if old is not None:
        old = old[old.n == 1_000_000].copy()
    key = ["regime", "sep", "seed", "eta"]
    for d in (new, old):
        if d is not None:
            d["eta"] = d["eta"].fillna(0.0)
    pick = lambda d, pol: d[d.policy == pol].set_index(key)
    a, f = pick(new, "auto_stop+auto"), pick(new, "full_run")
    sk, v1 = pick(new, "sklearn_tol(center_shift<=1e-4 var)"), pick(new, "v0.1.0(unstable_tol)+markov")
    out = pd.DataFrame({
        "끝까지(반복)": f.T_stop,
        "자동 정지": a.T_stop.astype(str) + " (" + a.stop_reason + ")",
        "시간 비율": a.time_frac, "오류율": a.disagree_rate, "끝까지 오류율": f.disagree_rate,
        "sklearn tol 시간": sk.time_frac, "sklearn tol 오류": sk.disagree_rate,
        "0.1.0 규칙 시간": v1.time_frac, "0.1.0 규칙 오류": v1.disagree_rate})
    if old is not None:
        o = pick(old, "auto_stop+auto")
        out["수정 전 정지"] = o.T_stop.reindex(out.index)
        out["수정 전 오류"] = o.disagree_rate.reindex(out.index)
    out = out.reset_index()
    out.index = [f"{RNAME[r]}, sep {s}, seed {sd}" + (f", η={e:g}" if r == "minibatch_const" else "")
                 for r, s, sd, e in zip(out.regime, out.sep, out.seed, out.eta)]
    out = out.drop(columns=key)
    pct = "{:.0%}"
    err = "{:.3%}"
    fmt = {"끝까지(반복)": "{:.0f}", "자동 정지": "{}", "시간 비율": pct, "오류율": err, "끝까지 오류율": err,
           "sklearn tol 시간": pct, "sklearn tol 오류": err, "0.1.0 규칙 시간": pct, "0.1.0 규칙 오류": err,
           "수정 전 정지": "{:.0f}", "수정 전 오류": err}
    T = {"auto_rule": md(out, fmt, index_name="사례")}
    v = new[new.policy.str.startswith("auto_stop+")].pivot_table(index=key, columns="policy", values="U_err")
    v = v.rename(columns=lambda c: {"auto_stop+auto": "auto(체제별)", "auto_stop+markov": "Markov",
                                    "auto_stop+frequency": "빈도", "auto_stop+keep": "현재 라벨 유지"}[c])
    v = v.reset_index()
    v.index = [f"{RNAME[r]}, sep {s}, seed {sd}" + (f", η={e:g}" if r == "minibatch_const" else "")
               for r, s, sd, e in zip(v.regime, v.sep, v.seed, v.eta)]
    v = v.drop(columns=key)[["auto(체제별)", "Markov", "빈도", "현재 라벨 유지"]]
    T["auto_assign"] = md(v, "{:.1%}", index_name="사례 (같은 정지 시점, U 내부 오류율)")
    return T


# ----------------------------------------------------------------------------- color quantization

POLICIES = ["auto", "v0.1.0", "sklearn_tol", "fixed_T25", "fixed_T50", "sample_faiss", "sample_100k"]
PNAME = {"auto": "자동 규칙", "v0.1.0": "0.1.0 규칙 (|U|/n ≤ 0.1%)", "sklearn_tol": "sklearn tol", "fixed_T25": "고정 25회 (faiss)",
         "fixed_T50": "고정 50회", "sample_faiss": "표본 256·k점 + 25회 (faiss식)", "sample_100k": "표본 10만 점, 수렴까지",
         "full": "끝까지 수렴"}


def color_tables():
    cq = _read("color_quant.csv")
    if cq is None or cq.empty:
        return {}
    T = {}
    rows = []
    for k, d in cq.groupby("k"):
        full = d[d.policy == "full"]
        rows.append(dict(k=k, images=d.image.nunique(), mpix=full.n.mean() / 1e6, T_full=full["T"].mean(),
                         T_full_max=full["T"].max(), psnr=full.psnr.mean(), sec=full.seconds.mean()))
    s = pd.DataFrame(rows).set_index("k")
    s.columns = ["사진 수", "평균 화소(백만)", "수렴 반복(평균)", "수렴 반복(최대)", "PSNR(끝까지)", "끝까지 시간(초)"]
    T["cq_setup"] = md(s, {"사진 수": "{:.0f}", "평균 화소(백만)": "{:.1f}", "수렴 반복(평균)": "{:.0f}",
                           "수렴 반복(최대)": "{:.0f}", "PSNR(끝까지)": "{:.2f}", "끝까지 시간(초)": "{:.1f}"}, index_name="k")
    for k, d in cq.groupby("k"):
        g = d[d.policy.isin(POLICIES)].groupby("policy")
        t = pd.DataFrame({"시간 비율(평균)": g.time_frac.mean(), "거리 비율(평균)": g.dist_frac.mean(),
                          "PSNR 손실 평균(dB)": -g.d_psnr.mean(), "PSNR 손실 최대(dB)": -g.d_psnr.min(),
                          "라벨 불일치": g.disagree_rate.mean(), "반복(평균)": g["T"].mean()})
        t = t.reindex([p for p in POLICIES if p in t.index])
        t.index = [PNAME[p] for p in t.index]
        T[f"cq_k{k}"] = md(t, {"시간 비율(평균)": "{:.0%}", "거리 비율(평균)": "{:.0%}", "PSNR 손실 평균(dB)": "{:.4f}",
                               "PSNR 손실 최대(dB)": "{:.4f}", "라벨 불일치": "{:.3%}", "반복(평균)": "{:.0f}"},
                           index_name=f"정책 (k={k})")
    a = cq[(cq.policy == "auto") & cq.n_U.gt(0)]
    if len(a):
        t = a.groupby("k").agg(stop_reason=("stop_reason", lambda x: ", ".join(f"{v}×{c}" for v, c in x.value_counts().items())),
                               frac_U=("n_U", "sum"), n=("n", "sum"), frac_fractional=("frac_fractional", "mean"),
                               d_blur=("psnr_blur_pi_dither", "mean"), blur=("psnr_blur", "mean"))
        t["frac_U"] = t["frac_U"] / t["n"]
        t["d_blur"] = t["d_blur"] - t["blur"]
        t = t.drop(columns=["n", "blur"])
        t.columns = ["정지 사유", "|U|/n", "U 중 분수형 π", "π 디더링의 흐린-PSNR 변화(dB)"]
        T["cq_soft"] = md(t, {"정지 사유": "{}", "|U|/n": "{:.3%}", "U 중 분수형 π": "{:.1%}",
                              "π 디더링의 흐린-PSNR 변화(dB)": "{:+.5f}"}, index_name="k")
    color_figure(cq)
    return T


PNAME_EN = {"auto": "auto rule", "sklearn_tol": "sklearn tol", "fixed_T25": "fixed 25 iterations (faiss)",
            "sample_faiss": "sample 256·k + 25 it. (faiss)", "sample_100k": "sample 100k, to convergence"}


def color_figure(cq):
    FIG.mkdir(parents=True, exist_ok=True)
    ks = sorted(cq.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(5.6 * len(ks), 4.6), squeeze=False)
    show = ["auto", "sklearn_tol", "fixed_T25", "sample_faiss", "sample_100k"]
    for ax, k in zip(axes[0], ks):
        d = cq[cq.k == k]
        for i, p in enumerate(show):
            q = d[d.policy == p]
            ax.scatter(q.time_frac, np.maximum(-q.d_psnr, 1e-5), s=28, color=SERIES[i], marker=MARKERS[i],
                       edgecolor=SURFACE, linewidth=1, label=PNAME_EN[p], alpha=0.85)
        ax.set_yscale("log")
        ax.set_xscale("log")
        _style(ax, f"k = {k}: PSNR lost vs time spent (one point per photo)", "PSNR loss vs convergence (dB, log)",
               "time / time to convergence (log)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_color_quant.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- VQ codebooks

def vq_tables():
    vq = _read("vq_codebook.csv")
    if vq is None or vq.empty:
        return {}
    T = {}
    for setting, d in vq.groupby("setting", sort=False):
        d = d.set_index("policy")
        rows = d[d.mse_U_hard.notna()] if "mse_U_hard" in d else d
        t = pd.DataFrame({"T": d["T"], "|U|/n": d.frac_U, "시간 비율": d.time_frac})
        for c, lab in (("U_err_current", "토큰 오류: 현재"), ("U_err_nearest", "최근접(C_T)"), ("U_err_markov", "Markov"),
                       ("U_err_frequency", "빈도"), ("U_err_auto", "auto")):
            t[lab] = d[c]
        rel = lambda c: 1 - rows[c] / rows["mse_U_hard"]
        t["MSE 감소: π soft"] = rel("mse_U_pi")
        t["점유율 soft"] = rel("mse_U_occupancy")
        t["거리 softmax"] = 1 - rows["mse_U_distsoft"] / rows["mse_U_hard_heldout"]
        t["상위2 최적 혼합(oracle)"] = rel("mse_U_top2_oracle")
        for c, lab in (("auroc_pi", "AUROC π"), ("auroc_occupancy", "AUROC 점유율"), ("auroc_distsoft", "AUROC 거리 softmax"),
                       ("auroc_margin", "AUROC 거리 차")):
            t[lab] = d[c]
        t["TV π"] = d.tv_pi
        t["TV 점유율"] = d.tv_occupancy
        if setting == "patch":
            t["전체 PSNR 변화(π soft, dB)"] = d.psnr_all_pi - d.psnr_all_hard
        fmt = {c: "{:.1%}" for c in t.columns}
        fmt.update({"T": "{:.0f}", "|U|/n": "{:.2%}", "TV π": "{:.3f}", "TV 점유율": "{:.3f}",
                    "전체 PSNR 변화(π soft, dB)": "{:+.4f}"})
        fmt.update({c: "{:.3f}" for c in t.columns if c.startswith("AUROC")})
        T[f"vq_{setting}"] = md(t, fmt, index_name="시점")
    vq_figure(vq)
    return T


def vq_figure(vq):
    FIG.mkdir(parents=True, exist_ok=True)
    settings = list(dict.fromkeys(vq.setting))
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 4.4), squeeze=False)
    series = [("mse_U_pi", "Markov π soft token"), ("mse_U_occupancy", "window-occupancy soft token"),
              ("mse_U_distsoft", "distance softmax (tuned τ)"), ("mse_U_top2_oracle", "best mix of 2 nearest codes")]
    for ax, s in zip(axes[0], settings):
        d = vq[(vq.setting == s) & vq.policy.str.startswith("T0=")].sort_values("T")
        for i, (c, lab) in enumerate(series):
            base = d["mse_U_hard_heldout"] if c == "mse_U_distsoft" else d["mse_U_hard"]
            ax.plot(d["T"], 100 * (1 - d[c] / base), color=SERIES[i], marker=MARKERS[i], lw=2, ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
        ax.axhline(0, color=GRID, lw=1.5, zorder=0)
        _style(ax, f"{s}: reconstruction error of unstable vectors", "MSE reduction vs hard nearest code (%)",
               "T0 (epochs of EMA codebook training)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_vq_soft_tokens.png", dpi=150)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    T = {}
    T.update(auto_rule_tables())
    T.update(color_tables())
    T.update(vq_tables())
    fill_report(T)
    for k, v in T.items():
        print(f"== {k}\n{v}\n")


if __name__ == "__main__":
    main()
