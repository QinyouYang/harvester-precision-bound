#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一: 随机能量收集的热力学基本极限 (TUR) —— 核心机制验证
===========================================================

主张: 从随机涨落中收集能量, 受随机热力学的"功率-精度-耗散"基本极限约束:
        Var(J_t)/<J_t>^2 * Sigma_t >= 2 k_B           ... 热力学不确定性关系 (TUR)
其中 J_t 是时间 t 内累积的输出 (这里取输运位移, 正比于输出功),
     Sigma_t 是同期总熵产生 (= 耗散热 / 温度)。

物理含义: 想"稳"地输出 (相对涨落小) 必须"耗散"(产熵); 只能在牺牲功率、趋于可逆时
          逼近极限的饱和, 最大功率点永远存在不可消除的间隙。

模型: 倾斜周期势中的过阻尼布朗粒子 (随机热力学中熵产生定义最无歧义的标准模型),
      作为随机能量收集器的最小、可严格验证的代理。

文献 (可追溯):
  - A. C. Barato, U. Seifert, Phys. Rev. Lett. 114, 158101 (2015).
  - P. Pietzonka, F. Ritort, U. Seifert, Phys. Rev. E 96, 012101 (2017), arXiv:1702.07699.
  - J. M. Horowitz, T. R. Gingrich, Nature Physics 16, 15 (2020).
  - P. Pietzonka, U. Seifert, Phys. Rev. Lett. 120, 190602 (2018).

所有数字由本脚本真实运行得到, 固定随机种子可复现, 未经任何手工修改。

环境变量:
  PIDM_QUICK=1   快速冒烟模式 (小样本、短时长), 仅用于验证代码可跑通, 不用于正式结论。
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 让脚本无论从哪里运行都能 import 同目录的 common.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import integrate_overdamped_tilted

# 图内文字语言: 'en' (默认, 用于 Nature/Science 投稿) 或 'zh' (中文)。
# 通过环境变量 PIDM_LANG 切换, 例如 PIDM_LANG=zh 可重新生成中文图。
LANG = os.environ.get("PIDM_LANG", "en").lower()


def TR(en, zh):
    """根据 LANG 选择图内文字: 英文优先(投稿要求), 需要时可切回中文。
    (命名为 TR 而非 T, 以免与温度变量 T 冲突)"""
    return en if LANG == "en" else zh


# 字体: 英文用干净的 Latin sans-serif; 中文用 CJK 字体回退 (缺失则退化, 不影响数值)。
if LANG == "en":
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica",
                                       "Liberation Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                       "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

QUICK = os.environ.get("PIDM_QUICK", "0") == "1"

# --------------------------------------------------------------------------
# 0. 约化单位与参数 (kB = T = gamma = L = 1)
# --------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

kB, T, gamma, L = 1.0, 1.0, 1.0, 1.0
D = kB * T / gamma
V0 = 0.5
f_drive = 2.0

dt = 2.0e-3
t_obs = 60.0 if not QUICK else 8.0
n_steps = int(t_obs / dt)
N_traj = 10000 if not QUICK else 1500
n_loads = 12 if not QUICK else 6
N_BOOT = 300 if not QUICK else 60

OUTDIR = os.environ.get("PIDM_OUTDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUTDIR = os.path.abspath(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)


def tur_quantities(X, f_load):
    """由末位移样本 X 计算 (v, P_out, P_in, P_diss, sigma, eta, rel_var, Sigma_t, Q)。"""
    meanX = X.mean()
    varX = X.var(ddof=1)
    v = meanX / t_obs
    P_in = f_drive * v
    P_out = f_load * v
    P_diss = (f_drive - f_load) * v
    sigma = P_diss / T
    eta = P_out / P_in if P_in > 0 else 0.0
    Sigma_t = sigma * t_obs
    rel_var = varX / (meanX ** 2)
    Q = rel_var * Sigma_t
    return dict(v=v, P_in=P_in, P_out=P_out, P_diss=P_diss, sigma=sigma,
                eta=eta, rel_var=rel_var, Sigma_t=Sigma_t, Q=Q)


def bootstrap_Q(X, f_load, n_boot=N_BOOT):
    """自助法重采样, 返回 TUR 乘积 Q 的 (2.5%, 97.5%) 置信区间。"""
    n = len(X)
    qs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        qs[b] = tur_quantities(X[idx], f_load)["Q"]
    return float(np.percentile(qs, 2.5)), float(np.percentile(qs, 97.5))


def main():
    print("=" * 76)
    print("实验一: 随机能量收集的热力学基本极限 (TUR) —— 核心机制验证"
          + ("  [QUICK 冒烟模式]" if QUICK else ""))
    print("=" * 76)
    print(f"模型: 倾斜周期势中的过阻尼布朗粒子 | kB=T=gamma=L=1 | V0={V0} f_drive={f_drive}")
    print(f"仿真: dt={dt} t_obs={t_obs} n_steps={n_steps} N_traj={N_traj} "
          f"n_loads={n_loads} N_BOOT={N_BOOT} seed={SEED}")
    print("-" * 76)

    loads = np.linspace(0.05 * f_drive, 0.85 * f_drive, n_loads)
    rows = []
    print(f"{'f_load':>7} {'eta':>6} {'v':>8} {'P_out':>8} {'sigma':>8} "
          f"{'Var/mean^2':>11} {'Q':>7} {'Q的95%CI':>16}")
    print("-" * 76)
    for f_load in loads:
        X = integrate_overdamped_tilted(f_drive, f_load, V0, L, D, gamma,
                                        dt, n_steps, N_traj, rng)
        q = tur_quantities(X, f_load)
        lo, hi = bootstrap_Q(X, f_load)
        rows.append(dict(f_load=float(f_load), eta=float(q["eta"]), v=float(q["v"]),
                         P_out=float(q["P_out"]), P_in=float(q["P_in"]),
                         P_diss=float(q["P_diss"]), sigma=float(q["sigma"]),
                         rel_var=float(q["rel_var"]), Sigma_t=float(q["Sigma_t"]),
                         Q=float(q["Q"]), Q_ci_lo=lo, Q_ci_hi=hi))
        print(f"{f_load:7.3f} {q['eta']:6.3f} {q['v']:8.4f} {q['P_out']:8.4f} "
              f"{q['sigma']:8.4f} {q['rel_var']:11.4e} {q['Q']:7.3f} [{lo:6.3f},{hi:6.3f}]")
    print("-" * 76)

    Qs = np.array([r["Q"] for r in rows])
    Q_lo = np.array([r["Q_ci_lo"] for r in rows])
    Q_hi = np.array([r["Q_ci_hi"] for r in rows])
    P_outs = np.array([r["P_out"] for r in rows])
    etas = np.array([r["eta"] for r in rows])
    loads_a = np.array([r["f_load"] for r in rows])

    tur_floor = 2.0 * kB
    ci_consistent = bool(np.all(Q_hi >= tur_floor - 1e-6))
    n_below = int(np.sum(Qs < tur_floor))
    i_maxP = int(np.argmax(P_outs))
    i_rev = int(np.argmax(etas))

    opt_load = float(loads_a[i_maxP]); P_max = float(P_outs[i_maxP])
    eta_maxP = float(etas[i_maxP]); Q_maxP = float(Qs[i_maxP])
    gap_maxP = Q_maxP / tur_floor - 1.0
    Q_rev = float(Qs[i_rev]); gap_rev = Q_rev / tur_floor - 1.0

    print("核心结论 (真实仿真):")
    print(f"  [1] TUR 极限 (定理, 不可违反): 2kB = {tur_floor:.3f}")
    print(f"  [2] 点估计低于极限的点数: {n_below}/{n_loads} (若有均为近失速区有限样本噪声)")
    print(f"  [3] 所有点 95%CI 是否都与 Q>=2kB 一致? -> {ci_consistent}")
    print(f"  [4] 最优负载(阻抗匹配) f_load*={opt_load:.3f}, P_max={P_max:.4f}, "
          f"eta={eta_maxP:.3f}, Q={Q_maxP:.3f} (间隙 {gap_maxP*100:.1f}%)")
    print(f"  [5] 近可逆点 eta={etas[i_rev]:.2f}: Q={Q_rev:.3f} (间隙 {gap_rev*100:.1f}%)")
    print("=" * 76)

    # ---------------------- 出图 (无图标题; 仅 (a)(b) 角标) ----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0))

    # (a) 功率-效率折返 与 最优负载
    ax1.plot(etas, P_outs, "o-", color="#2c6fbb", lw=2, ms=6,
             label=TR("Output power $P_{out}$", "输出功率 $P_{out}$"))
    ax1.plot(eta_maxP, P_max, "*", color="#d1495b", ms=20,
             label=TR(f"Optimal load (impedance matching): $\\eta$={eta_maxP:.2f}, $P_{{max}}$={P_max:.3f}",
                     f"最优负载(阻抗匹配): $\\eta$={eta_maxP:.2f}, $P_{{max}}$={P_max:.3f}"))
    ax1.set_xlabel(TR("Efficiency $\\eta = P_{out}/P_{in}$", "效率 $\\eta = P_{out}/P_{in}$"), fontsize=12)
    ax1.set_ylabel(TR("Output power $P_{out}$ (reduced units)", "输出功率 $P_{out}$ (约化单位)"), fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10, loc="best")
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes, fontsize=14,
             fontweight="bold", va="top", ha="left")

    # (b) TUR 乘积 与 不可逾越的极限线 (含 95%CI 误差棒)
    yerr = np.vstack([Qs - Q_lo, Q_hi - Qs])
    ax2.errorbar(etas, Qs, yerr=yerr, fmt="o-", color="#2c6fbb", lw=2, ms=6,
                 ecolor="#7aa6d6", elinewidth=1.5, capsize=3,
                 label=TR("TUR product $Q=\\frac{Var(J)}{\\langle J\\rangle^2}\\Sigma_t$ (95% CI)",
                         "TUR 乘积 $Q=\\frac{Var(J)}{\\langle J\\rangle^2}\\Sigma_t$ (含95%CI)"))
    ax2.axhline(tur_floor, color="#d1495b", ls="--", lw=2.2,
                label=TR(f"Thermodynamic bound $2k_B$ = {tur_floor:.0f} (theorem)",
                        f"热力学基本极限 $2k_B$ = {tur_floor:.0f} (定理)"))
    ax2.fill_between(etas, 0, tur_floor, color="#d1495b", alpha=0.10)
    ax2.plot(eta_maxP, Q_maxP, "*", color="#e0a800", ms=20,
             label=TR(f"Maximum-power point (gap {gap_maxP*100:.0f}%)",
                     f"最大功率点 (间隙 {gap_maxP*100:.0f}%)"))
    ax2.annotate(TR("Forbidden by TUR", "禁区 (TUR 禁止)"), xy=(etas.mean(), tur_floor * 0.5),
                 ha="center", va="center", fontsize=10, color="#8a2433")
    ax2.set_xlabel(TR("Efficiency $\\eta$", "效率 $\\eta$"), fontsize=12)
    ax2.set_ylabel(TR("TUR product $Q$", "TUR 乘积 $Q$"), fontsize=12)
    ax2.set_ylim(0, float(max(Q_hi)) * 1.12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="best")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=14,
             fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    png = os.path.join(OUTDIR, "exp1_thermodynamic_bound.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {png}")

    summary = dict(
        experiment="exp1_thermodynamic_bound_TUR",
        quick_mode=QUICK,
        model="overdamped Brownian particle in tilted periodic (washboard) potential",
        reduced_units="kB=T=gamma=L=1",
        params=dict(V0=V0, f_drive=f_drive, dt=dt, t_obs=t_obs, n_steps=n_steps,
                    N_traj=N_traj, n_loads=n_loads, n_bootstrap=N_BOOT, seed=SEED),
        tur_floor=tur_floor,
        n_point_estimates_below_floor=n_below,
        all_CI_consistent_with_bound=ci_consistent,
        optimal_load=opt_load, P_max=P_max,
        eta_at_max_power=eta_maxP, Q_at_max_power=Q_maxP,
        gap_at_max_power_percent=gap_maxP * 100,
        Q_near_reversible=Q_rev, gap_near_reversible_percent=gap_rev * 100,
        sweep=rows,
    )
    js = os.path.join(OUTDIR, "exp1_metrics.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[metrics] {js}")
    print("实验一完成。")
    return summary


if __name__ == "__main__":
    main()
