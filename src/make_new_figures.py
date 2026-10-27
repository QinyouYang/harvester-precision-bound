#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为重排后的论文生成三张【新主结果】成品图, 全部从已落盘的真实 JSON 读取数据:
  fig_fluctuation_control.png  (exp6): 非线性涨落调控设计律 —— 主菜
  fig_mechanism_decomposition.png (exp7): 二因子分解机理与判据
  fig_activity_kinetic.png     (exp5): 动力学活动极限(动理学 vs 热力学) —— 概念骨架

严格遵循投稿图规范: 无总标题/无分图标题, 仅 (a)(b) 角标 (transAxes 0.02,0.97, bold, 英文),
统一配色 蓝#2c6fbb / 红#d1495b / 绿#2e8b57, dpi=150 bbox_inches tight, 图内文字英文。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("PIDM_RESULTS") or os.path.join(HERE, "..", "results")   # JSON 来源
OUT = os.environ.get("PIDM_OUTDIR") or RES                                     # 图输出目录 (默认 results/)
os.makedirs(OUT, exist_ok=True)

BLUE, RED, GREEN = "#2c6fbb", "#d1495b", "#2e8b57"
plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9,
                     "xtick.direction": "out", "ytick.direction": "out"})


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def panel_label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")


# ============================ Fig (exp6): 设计律 ============================
def fig_design_law():
    e6 = load("exp6_metrics.json")
    sweep = e6["wave_hardening_sweep"]
    beta = np.array([r["beta"] for r in sweep])
    e2tB = np.array([r["eps2_t_Beff"] for r in sweep])
    eps2 = np.array([r["eps2_tobs"] for r in sweep])
    P = np.array([r["P_harv"] for r in sweep])
    bist = e6["bistable_check"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    # (a) eps^2 * t * B_eff vs hardening strength: 线性=2 基线, 硬化压到 ~1, 双稳 >2
    ax1.axhline(2.0, color="0.5", ls="--", lw=1.2, zorder=1)
    # 把注释锚到坐标轴右内侧 (x 用 axes 比例, y 用数据值), 右对齐 -> 永远不会越过右框线
    _tr = blended_transform_factory(ax1.transAxes, ax1.transData)
    ax1.text(0.98, 2.04, "linear-Gaussian limit (=2)", color="0.4", fontsize=9,
             ha="right", va="bottom", transform=_tr)
    ax1.plot(beta, e2tB, "o-", color=BLUE, lw=2, ms=6, zorder=3, label="hardening Duffing")
    ax1.plot([0.30], [bist["eps2_t_Beff"]], "s", color=RED, ms=8, zorder=4,
             label="bistable (k<0)")
    ax1.set_xlabel(r"hardening strength  $\beta$")
    ax1.set_ylabel(r"$\varepsilon^{2}\, t\, B_\mathrm{eff}$  (asymptotic)")
    ax1.set_ylim(0.8, 2.4)
    ax1.legend(frameon=False, fontsize=9, loc="center right")
    panel_label(ax1, "(a)")

    # (b) Pareto: 相对涨落 eps^2 vs 平均功率, 随 beta 变化 (右下=更稳更低功率)
    sc = ax2.scatter(P, eps2, c=beta, cmap="viridis", s=70, zorder=3, edgecolor="k", linewidth=0.4)
    ax2.plot(P, eps2, "-", color="0.6", lw=1, zorder=2)
    # 标注线性基线点
    ax2.annotate("linear (β=0)", (P[0], eps2[0]), textcoords="offset points",
                 xytext=(-15, -16), fontsize=9, color="0.3", ha="right",
                 arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8))
    ax2.set_xlabel(r"mean harvested power  $P=c_\mathrm{load}\langle v^2\rangle$")
    ax2.set_ylabel(r"relative fluctuation  $\varepsilon^{2}(t_\mathrm{obs})$")
    cb = fig.colorbar(sc, ax=ax2, pad=0.02)
    cb.set_label(r"$\beta$", fontsize=10)
    panel_label(ax2, "(b)")

    fig.tight_layout()
    p = os.path.join(OUT, "fig_fluctuation_control.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


# ====================== Fig (exp7): 二因子分解机理 ======================
def fig_mechanism():
    e7 = load("exp7_metrics.json")
    cases = e7["cases"]
    pred = np.array([c["pred_eps2_t_Beff"] for c in cases])
    meas = np.array([c["meas_eps2_t_Beff"] for c in cases])
    Ak = np.array([c["A_kappa"] for c in cases])
    Rt = np.array([c["R_tau"] for c in cases])
    # 分类着色: 线性(灰) / 硬化(蓝) / 双稳(红)
    cols = []
    for c in cases:
        lab = c["label"]
        if "双稳" in lab or "bistable" in lab.lower():
            cols.append(RED)
        elif "硬化" in lab or "harden" in lab.lower():
            cols.append(BLUE)
        else:
            cols.append("0.45")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    # (a) 预测 vs 实测 (验证分解), y=x 线
    lim = [0.7, 2.2]
    ax1.plot(lim, lim, "--", color="0.5", lw=1.2, zorder=1)
    ax1.scatter(pred, meas, c=cols, s=70, zorder=3, edgecolor="k", linewidth=0.4)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel(r"predicted  $(\kappa+2)(\tau_{v^2}/\tau_c)$")
    ax1.set_ylabel(r"measured  $\varepsilon^{2} t\, B_\mathrm{eff}$")
    ax1.text(0.96, 0.06, "agreement\n$\\leq$12%", transform=ax1.transAxes,
             fontsize=9, ha="right", color="0.3")
    panel_label(ax1, "(a)")

    # (b) 二因子平面: A_kappa (x) vs R_tau (y); 等值线 A*R=const; 判据线 A*R=1
    axb = ax2
    xg = np.linspace(0.55, 1.05, 100); yg = np.linspace(0.5, 1.6, 100)
    X, Y = np.meshgrid(xg, yg)
    cs = axb.contour(X, Y, X*Y, levels=[0.5, 0.75, 1.0, 1.25], colors="0.75", linewidths=0.8)
    axb.clabel(cs, fmt=lambda v: f"$A_\\kappa R_\\tau$={v:g}", fontsize=7, inline=True)
    # 判据分隔线 A*R=1 (=eps^2 tB_eff/2=1 -> eps^2 tB_eff=2)
    axb.plot(xg, 1.0/xg, color="k", lw=1.4, zorder=2)
    axb.scatter(Ak, Rt, c=cols, s=75, zorder=4, edgecolor="k", linewidth=0.4)
    # 文字注解区域
    axb.text(0.62, 0.62, "suppressed\n$\\varepsilon^2 tB_\\mathrm{eff}<2$", fontsize=8.5, color=GREEN)
    axb.text(0.86, 1.42, "enhanced\n$>2$", fontsize=8.5, color=RED)
    axb.set_xlim(0.55, 1.05); axb.set_ylim(0.5, 1.6)
    axb.set_xlabel(r"amplitude factor  $A_\kappa=(\kappa+2)/2$")
    axb.set_ylabel(r"correlation-time factor  $R_\tau=\tau_{v^2}/\tau_c$")
    # 图例(手工)
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor='0.45', markersize=8, label='linear'),
           Line2D([0],[0], marker='o', color='w', markerfacecolor=BLUE, markersize=8, label='hardening'),
           Line2D([0],[0], marker='o', color='w', markerfacecolor=RED, markersize=8, label='bistable')]
    axb.legend(handles=leg, frameon=False, fontsize=8.5, loc="center left")
    panel_label(axb, "(b)")

    fig.tight_layout()
    p = os.path.join(OUT, "fig_mechanism_decomposition.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


# ================== Fig (exp5): 动理学 vs 热力学 ==================
def fig_kinetic():
    e5 = load("exp5_metrics.json")
    sweep = e5["sweep"]
    sy2 = np.array([s["sigma_y2"] for s in sweep])
    e2tB = np.array([s["rows"][-1]["eps2_t_Beff"] for s in sweep])
    e2S = np.array([s["rows"][-1]["eps2_Sigma_t"] for s in sweep])
    sdot = np.array([s["sigma_dot"] for s in sweep])

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    # x 轴用驱动强度 sigma_y2 (含 0=平衡), 用 symlog 以含 0
    x = sy2.copy()
    ax.axhline(2.0, color="0.5", ls="--", lw=1.2)
    ax.text(x.max()*0.30, 2.15, "floor = 2", color="0.4", fontsize=9)
    ax.plot(x, e2tB, "o-", color=BLUE, lw=2, ms=6,
            label=r"$\varepsilon^{2} t\, B_\mathrm{eff}$  (kinetic: fixed at 2)")
    ax.plot(x, e2S, "s-", color=RED, lw=2, ms=6,
            label=r"$\varepsilon^{2}\Sigma_t$  (thermodynamic: no floor)")
    ax.set_xlabel(r"drive strength  $\sigma_y^{2}$  (0 = equilibrium)")
    ax.set_ylabel("uncertainty product")
    ax.set_yscale("symlog", linthresh=3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_activity_kinetic.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    fig_design_law()
    fig_mechanism()
    fig_kinetic()
    print("DONE: three new-main-result figures generated.")
