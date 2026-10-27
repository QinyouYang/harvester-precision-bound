#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp10 成品图: 实测压电器件 (Costanzo et al., Entropy 2021) 上的 ε²·t·B_eff 检验。
全部从 results/exp10_metrics.json 读取 (含原始时间序列重算的机理面板)。

三个面板:
  (a) 代表数据集 (a=1g, R=2200Ω): 实测 ε²tB_eff(t) vs t/τ_c, 零参数高斯精确曲线,
      平稳高斯代理系综 (落在曲线上), 全协方差恒等式曲线 (含慢调制), 虚线=2;
  (b) 全部 9 组: 实测长窗值 / 平稳代理 / 调制代理(实测慢超额协方差合成包络) /
      有限时高斯期望, 虚线=2;
  (c) 机理: v² 归一化自协方差 vs Isserlis 2R²/C(0) (对数滞后), 慢超额台阶清晰可见。

严格遵循投稿图规范: 无标题, 仅 (a)(b)(c) 角标 (transAxes 0.02,0.97, bold, 英文),
配色 蓝#2c6fbb / 红#d1495b / 绿#2e8b57, font 11, dpi=150, bbox_inches=tight。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.environ.get("PIDM_OUTDIR", RES)
DATADIR = os.environ.get("PIEZO_DATADIR", os.path.join(HERE, "..", "data", "piezo_costanzo2021"))
os.makedirs(OUT, exist_ok=True)

BLUE, RED, GREEN = "#2c6fbb", "#d1495b", "#2e8b57"
GRAY = "0.45"
plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9,
                     "xtick.direction": "out", "ytick.direction": "out"})


def panel_label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")


def autocorr_unbiased(x, kmax):
    n = x.size
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    F = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(np.abs(F) ** 2, n=nfft)[:kmax + 1]
    return ac / (n - np.arange(kmax + 1))


def _require_piezo_data(datadir):
    """实测压电数据不随仓库公开 (归 Costanzo 等所有); 缺失时给出清晰提示并退出。"""
    probe = os.path.join(datadir, "Acceleration_g1", "R2200_V_load.txt")
    if not os.path.exists(probe):
        raise SystemExit(
            "[make_exp10_figure] Measured piezo records not found in %s\n"
            "        These data belong to Costanzo et al. (Entropy 23, 677, 2021) and are NOT\n"
            "        redistributed with this repository; request them from the original authors\n"
            "        and place them as described in data/piezo_costanzo2021/README.md.\n"
            "        The computed results are provided in results/exp10_metrics.json." % datadir)


def main():
    _require_piezo_data(DATADIR)
    with open(os.path.join(RES, "exp10_metrics.json"), encoding="utf-8") as f:
        M = json.load(f)
    ds = M["datasets"]
    rep = next(d for d in ds if d["a_g"] == "1.0" and d["R_ohm"] == 2200)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 3.9))

    # ---------------- (a) 代表数据集曲线 ----------------
    tau_c = rep["tau_c_s"]
    t = np.array([w["t"] for w in rep["windows"]])
    y = np.array([w["eps2_t_Beff"] for w in rep["windows"]])
    ysd = np.array([w["eps2_t_Beff_sd"] for w in rep["windows"]])
    yx = np.array([w["eps2_t_Beff_exact"] for w in rep["windows"]])
    yfc = np.array(rep["eps2_t_Beff_fullC"])
    sm = np.array(rep["surrogate"]["curve_mean"])
    sse = np.array(rep["surrogate"]["curve_se"])
    xx = t / tau_c

    ax1.axhline(2.0, color="0.5", ls="--", lw=1.2, zorder=1)
    ax1.plot(xx, yx, "-", color=BLUE, lw=2, zorder=2,
             label="exact Gaussian (zero-parameter)")
    ax1.plot(xx, yfc, "-.", color=GREEN, lw=1.6, zorder=2,
             label=r"exact, full $C_{v^2}$ (incl. modulation)")
    ax1.errorbar(xx, sm, yerr=sse, fmt="^", color=GRAY, ms=4.5, lw=1,
                 capsize=2, zorder=3, label="stationary surrogates")
    ax1.errorbar(xx, y, yerr=ysd, fmt="o", color=RED, ms=5, lw=1.2,
                 capsize=2.5, zorder=4, label="measured device")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$t/\tau_c$")
    ax1.set_ylabel(r"$\varepsilon^{2}(t)\,t\,B_\mathrm{eff}$")
    ax1.set_ylim(0, 4.6)
    ax1.legend(frameon=False, fontsize=8, loc="upper left",
               bbox_to_anchor=(0.015, 0.90))
    panel_label(ax1, "(a)")

    # ---------------- (b) 9 组汇总 ----------------
    order = sorted(range(len(ds)), key=lambda i: (float(ds[i]["a_g"]), ds[i]["R_ohm"]))
    xpos = np.arange(len(order))
    raw = np.array([ds[i]["long_eps2tB"] for i in order])
    rsd = np.array([ds[i]["long_eps2tB_sd"] for i in order])
    sur = np.array([ds[i]["surrogate"]["long_eps2tB"] for i in order])
    sse2 = np.array([ds[i]["surrogate"]["long_eps2tB_se"] for i in order])
    mod = np.array([ds[i]["modulated_surrogate"]["long_eps2tB"] for i in order])
    mse = np.array([ds[i]["modulated_surrogate"]["long_eps2tB_se"] for i in order])
    gex = np.array([ds[i]["surrogate"]["gauss_expect_long"] for i in order])

    ax2.axhline(2.0, color="0.5", ls="--", lw=1.2, zorder=1)
    ax2.plot(xpos, gex, "_", color=BLUE, ms=13, mew=2.2, zorder=2,
             label="finite-$t$ Gaussian expectation")
    ax2.errorbar(xpos - 0.10, sur, yerr=sse2, fmt="^", color=GRAY, ms=5.5,
                 lw=1, capsize=2, zorder=3, label="stationary surrogates")
    ax2.errorbar(xpos + 0.10, mod, yerr=mse, fmt="s", color=GREEN, ms=5,
                 lw=1, capsize=2, zorder=3,
                 label="surrogate × slow envelope")
    ax2.errorbar(xpos, raw, yerr=rsd, fmt="o", color=RED, ms=6, lw=1.2,
                 capsize=2.5, zorder=4, label="measured device")
    labels = [f"{ds[i]['R_ohm']}" for i in order]
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_xlabel(r"load resistance $R$ ($\Omega$)")
    for k, a in enumerate(("0.8", "1.0", "1.2")):
        ax2.text(3 * k + 1, 0.30, rf"$a={a}\,g$", ha="center", fontsize=9,
                 color="0.3")
        if k:
            ax2.axvline(3 * k - 0.5, color="0.85", lw=0.8, zorder=0)
    ax2.set_ylabel(r"$\varepsilon^{2} t\,B_\mathrm{eff}$  ($t\geq 30\,\tau_c$)")
    ax2.set_ylim(0, 6.0)
    hh, ll = ax2.get_legend_handles_labels()
    order = [0, 2, 1, 3]  # 列优先: 长键名同列, 控制图例总宽
    ax2.legend([hh[i] for i in order], [ll[i] for i in order],
               frameon=False, fontsize=7.0, loc="upper left",
               bbox_to_anchor=(0.115, 1.005), ncol=2, columnspacing=0.9,
               handlelength=1.1, handletextpad=0.35, labelspacing=0.32,
               borderaxespad=0.1)
    panel_label(ax2, "(b)")

    # ---------------- (c) 机理: v² 自协方差的慢超额 ----------------
    path = os.path.join(DATADIR, "Acceleration_g1", "R2200_V_load.txt")
    d = np.loadtxt(path)
    vp = d[:, 1] - d[:, 1].mean()
    dt = float(np.mean(np.diff(d[:, 0])))
    var = float(np.var(vp))
    kmax = int(round(2.0 / dt))
    R = autocorr_unbiased(vp, kmax)
    C = autocorr_unbiased(vp ** 2 - var, kmax)
    tau = np.arange(kmax + 1) * dt
    sel = tau >= dt
    # 慢段平滑 (50 ms) 仅用于视觉引导
    w = int(round(0.05 / dt))
    Cs = np.convolve(C, np.ones(w) / w, mode="same")

    ax3.plot(tau[sel], (2 * R[sel] ** 2) / C[0], "-", color=BLUE, lw=2,
             label=r"Isserlis $2R^2/C(0)$ (Gaussian)")
    ax3.plot(tau[sel], C[sel] / C[0], "-", color=RED, lw=1.0, alpha=0.35)
    ax3.plot(tau[sel], Cs[sel] / C[0], "-", color=RED, lw=2,
             label=r"measured $C_{v^2}/C(0)$")
    ax3.fill_between(tau[sel], (2 * R[sel] ** 2) / C[0],
                     np.maximum(Cs[sel] / C[0], (2 * R[sel] ** 2) / C[0]),
                     color=RED, alpha=0.12, lw=0)
    ax3.annotate("slow excess (drive-intensity modulation)",
                 xy=(0.45, 0.030), xycoords="data",
                 xytext=(0.975, 0.845), textcoords="axes fraction",
                 ha="right", va="top", fontsize=8.2, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1,
                                 shrinkA=4, shrinkB=2))
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(2e-4, 2.0)
    ax3.set_ylim(3e-3, 3.0)
    ax3.set_xlabel(r"lag $\tau$ (s)")
    ax3.set_ylabel(r"$C_{v^2}(\tau)/C_{v^2}(0)$")
    ax3.legend(frameon=False, fontsize=8, loc="upper right")
    panel_label(ax3, "(c)")

    fig.tight_layout(w_pad=2.0)
    out = os.path.join(OUT, "fig_measured_device.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("written:", out)


if __name__ == "__main__":
    main()
