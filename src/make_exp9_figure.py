#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 exp9 的落盘 JSON 生成论文【补充图】: Torino 独立双稳模型的仿真交叉验证。
两个面板:
  (a) 收集功的 eps^2 t B_eff (实测) vs 跨阱率, 机械(∫v^2)与负载(∫e^2)两口径; 虚线标线性基线 2。
      -> 跨阱区两口径都远高于 2, 且随跨阱增强而上升。
  (b) 实测 vs 二因子预测 (kappa+2)(tau_v2/tau_c); 点落在对角线附近
      -> 同一二因子机理在独立模型上也成立 (负载口径 kappa<0 仍 >2, 因 R_tau>>1 主导)。
图内英文; 遵循图规范(无标题, 仅角标, 指定调色板, dpi=150)。诚实标注: 仿真, 非实验。
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "..", "results", "exp9_torino_bistable.json")
OUT  = os.path.join(os.environ.get("PIDM_OUTDIR") or os.path.join(HERE, "..", "results"),
                    "fig_independent_crosscheck.png")

BLUE, RED, GREEN = "#2c6fbb", "#d1495b", "#2e8b57"   # 既定调色板


def main():
    d = json.load(open(JSON, encoding="utf-8"))
    rows = d["rows"]
    # 机械口径 / 负载口径, 各按 sigma 排序
    mech = sorted([r for r in rows if "机械" in r["channel"]], key=lambda r: r["sigma"])
    load = sorted([r for r in rows if "负载" in r["channel"]], key=lambda r: r["sigma"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # ---- (a) eps^2 t B_eff vs 跨阱率 ----
    axA.plot([r["hop_rate"] for r in mech], [r["meas_eps2_t_Beff"] for r in mech],
             "o-", color=BLUE, lw=1.8, ms=7, label=r"mechanical  $\int v^2$")
    axA.plot([r["hop_rate"] for r in load], [r["meas_eps2_t_Beff"] for r in load],
             "s-", color=RED, lw=1.8, ms=7, label=r"load  $\int e^2$")
    axA.axhline(2.0, ls="--", color="#888888", lw=1.4)
    axA.text(0.97, 2.0 * 1.06, "linear baseline = 2", color="#555555",
             ha="right", va="bottom", fontsize=10, transform=axA.get_yaxis_transform())
    axA.set_yscale("log")
    axA.set_xlabel("inter-well hopping rate", fontsize=11)
    axA.set_ylabel(r"$\varepsilon^2\, t\, B_\mathrm{eff}$  (harvested work)", fontsize=11)
    axA.legend(frameon=False, fontsize=10, loc="center right")
    axA.grid(True, which="both", alpha=0.25)
    axA.text(0.02, 0.97, "(a)", transform=axA.transAxes, fontsize=13, fontweight="bold", va="top")

    # ---- (b) 实测 vs 二因子预测 ----
    allr = mech + load
    cols = [BLUE] * len(mech) + [RED] * len(load)
    preds = [r["pred_eps2_t_Beff"] for r in allr]
    meass = [r["meas_eps2_t_Beff"] for r in allr]
    lo = 0.8 * min(preds + meass)
    hi = 1.2 * max(preds + meass)
    axB.plot([lo, hi], [lo, hi], "-", color="#888888", lw=1.3, zorder=1)  # 对角线 y=x
    for p, mval, c, r in zip(preds, meass, cols, allr):
        mk = "o" if c == BLUE else "s"
        axB.scatter([p], [mval], c=c, marker=mk, s=60, zorder=3, edgecolors="white", linewidths=0.6)
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlim(lo, hi); axB.set_ylim(lo, hi)
    axB.set_xlabel(r"two-factor prediction  $(\kappa+2)\,(\tau_{v^2}/\tau_c)$", fontsize=11)
    axB.set_ylabel(r"measured  $\varepsilon^2\, t\, B_\mathrm{eff}$", fontsize=11)
    axB.grid(True, which="both", alpha=0.25)
    axB.text(0.02, 0.97, "(b)", transform=axB.transAxes, fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
