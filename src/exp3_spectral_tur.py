#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验三: 谱不确定性关系 (Spectral TUR) 的推导与数值验证
========================================================

核心智力贡献 (论文"缺失一": 把 TUR 推广到非热、有色噪声驱动的能量收集)
-----------------------------------------------------------------------
考虑线性电机械能量收集器在平稳有色高斯力 eta(t) 驱动下的运动:

        m x'' + (c_mech + c_load) x' + k x = g * eta(t)            (1)

其稳态速度 v(t) = x'(t) 是一个**平稳零均值高斯过程**, 其功率谱由
传递函数 H(w)=1/(k - m w^2 + i w c_total) 决定:

        S_v(w) = w^2 |H(w)|^2 g^2 S_eta(w),   c_total = c_mech + c_load   (2)

记速度自相关 R(tau) = <v(t) v(t+tau)> = (1/2π)∫ S_v(w) e^{i w tau} dw
(本代码采用单边谱、∫S dw 约定, 与 common.synthesize_wave 一致; tau_c 等
最终量纲不依赖该约定)。

收集功 (取出的能量, 正、单调、累积的平方型耗散量):

        W_t = c_load ∫_0^t v(s)^2 ds                                (3)

对零均值高斯 v, 由 **Isserlis / Wick 定理** <v_a v_b v_c v_d> =
R_ab R_cd + R_ac R_bd + R_ad R_bc, 立刻得到 v^2 的协方差
Cov(v(s)^2, v(s')^2) = 2 R(s-s')^2。代入 (3):

        <W_t> = c_load R(0) t
        Var(W_t) = 2 c_load^2 ∫_0^t∫_0^t R(s-s')^2 ds ds'
                 = 4 c_load^2 ∫_0^t (t-tau) R(tau)^2 dtau          (4)

定义相对涨落 (Fano/CV^2, 与公因子 c_load 无关):

        eps^2(t) ≡ Var(W_t)/<W_t>^2
                 = 4 ∫_0^t (t-tau) R(tau)^2 dtau / (R(0) t)^2       (5)  [精确, 无近似]

当观测时间 t >> tau_c (相关时间) 时:

        eps^2(t) -> 2 tau_c / t,   tau_c ≡ ∫ R(tau)^2 dtau / R(0)^2  (6)

定义有效带宽 B_eff ≡ 1/tau_c = R(0)^2 / ∫R^2dtau。于是得到可证的、
依赖"驱动谱 × 器件传递函数"的紧致极限:

        ┌─────────────────────────────────────────────┐
        │   eps^2(t) · t · B_eff  ->  2     (t>>tau_c) │   (7)
        └─────────────────────────────────────────────┘

这正是热力学不确定性关系 Var(J)/<J>^2 · Sigma >= 2 k_B 中 "2 k_B" 的
有色、非热谱驱动对应物: 把"熵产"换成"积累时间 × 有效带宽 (t·B_eff)",
不等式饱和为等式 (线性高斯收集器是 TUR 的饱和子)。
有限观测修正 (由 (5) 展开): eps^2·t = 2 tau_c - K/t + O(1/t^2),
K = 4∫_0^∞ tau R(tau)^2 dtau / R(0)^2 —— 使 eps^2·t 从**下方**趋于 2 tau_c。

新现象 (论文"缺失二": 只有从该极限视角才看得见)
-------------------------------------------------------
B_eff 与平均功率 P=c_load<v^2> 是 c_load 的**两个方向相反**的函数:
增大负载阻尼使输出更"稳"(B_eff 上升), 却把功率推离其匹配峰值。
于是存在**功率-精度 Pareto 前沿**: 最大功率工作点 c*_P 与最高精度
(最大 B_eff) 工作点并不重合。据此可把器件分两类——
"已撞精度天花板"(大阻尼、低功率端, eps^2·t·B_eff 已≈2, 提升空间小) 与
"远未撞天花板"(高功率端, 涨落大、B_eff 低, 有被忽视的稳定化余量)。

验证策略 (三块, 全部真实运行、可复现、无编造)
-----------------------------------------------
(A) OU 解析锚点: 取 v 为 Ornstein-Uhlenbeck 过程, R(tau)=sigma^2 e^{-|tau|/tau_r},
    则 tau_c=tau_r、精确 eps^2·t = 2 tau_r - tau_r^2/t 完全解析。用真实
    OU Langevin (Euler, 非周期) 积分, 数值点应落在该精确曲线上 —— 干净检验式(5)-(7)。
(B) 真实海浪谱: 用 JONSWAP 谱 (文献见下) 经 (2) 得 S_v, 数值算 R(tau)、
    tau_c、精确 eps^2_exact(t) (式5)。再用随机相位法 (irfft 实现, 给出 (1)
    的**精确稳态**速度) 合成 v(t), 测得 eps^2(t)。数值点应落在**零参数**
    解析曲线 eps^2_exact(t) 上, 并随 t 增大趋于 2 tau_c。
(C) 功率-精度权衡: 沿 c_load 扫描, 解析给出 P(c_load) 与 B_eff(c_load)、
    固定观测窗的 eps^2, 展示峰值错位 (新现象)。

文献 (可追溯, 与项目其余部分一致):
  - L. Isserlis, Biometrika 12, 134 (1918).  (高斯高阶矩 / Wick 定理)
  - T. Koyuk & U. Seifert, Phys. Rev. Lett. 125, 260604 (2020). (任意含时驱动 TUR)
  - A. C. Barato & U. Seifert, Phys. Rev. Lett. 114, 158101 (2015). (TUR 原型)
  - W. J. Pierson & L. Moskowitz, J. Geophys. Res. 69, 5181 (1964).
  - K. Hasselmann et al. (JONSWAP), Dtsch. Hydrogr. Z. A8, 12 (1973).
  - J. Falnes, "Ocean Waves and Oscillating Systems", Cambridge (2002).
  - G. E. Uhlenbeck & L. S. Ornstein, Phys. Rev. 36, 823 (1930).

所有数字由本脚本真实运行得到, 固定随机种子可复现, 未经任何手工修改。
本实验是"真实计算实验/真实标准谱", **不是**物理水槽实测; 不混淆两者。

环境变量:
  PIDM_QUICK=1   快速冒烟模式 (少实现、短时长、粗网格)。
  PIDM_LANG=zh   图内文字切换为中文 (默认 en, 用于投稿)。
  PIDM_OUTDIR    输出目录 (默认 <project>/results)。
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import jonswap_spectrum

# --------------------------------------------------------------------------
# 0. 运行配置 (语言 / 字体 / 输出目录 / 快速模式), 与 exp1、exp2 保持一致
# --------------------------------------------------------------------------
LANG = os.environ.get("PIDM_LANG", "en").lower()


def TR(en, zh):
    """图内文字: 英文优先(投稿), PIDM_LANG=zh 时切中文。"""
    return en if LANG == "en" else zh


if LANG == "en":
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica",
                                       "Liberation Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                       "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

QUICK = os.environ.get("PIDM_QUICK", "0") == "1"

_here = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("PIDM_OUTDIR") or os.path.join(_here, os.pardir, "results")
OUTDIR = os.path.abspath(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)

# 调色板 (与 exp2 一致)
C_BLUE, C_BLUE2, C_RED, C_GREY = "#2c6fbb", "#6c8ebf", "#d1495b", "#9aa0a6"
C_GREEN = "#2e8b57"

# --------------------------------------------------------------------------
# 收集器参数 (与 exp2 的点吸收式波浪能收集器完全一致, 保证可追溯)
# --------------------------------------------------------------------------
Hs, Tp = 2.0, 8.0
OMEGA_P = 2 * np.pi / Tp
GAMMA_J = 3.3
M_MASS, C_MECH, G_FORCE = 1.0, 0.05, 1.0
K_STIFF = M_MASS * OMEGA_P ** 2
C_LOAD_STAR = 0.118          # exp2 求得的最优 PTO 阻尼 (功率匹配点)


# ==========================================================================
# 1. 解析工具: 由谱得到 R(tau)、tau_c、K、以及精确 eps^2_exact(t)
# ==========================================================================
def velocity_spectrum(omega, c_load):
    """式(2): 速度功率谱 S_v(w) (单边, ∫S dw 约定)。"""
    c_total = C_MECH + c_load
    H = 1.0 / (K_STIFF - M_MASS * omega ** 2 + 1j * omega * c_total)
    S_eta = jonswap_spectrum(omega, Hs, OMEGA_P, GAMMA_J)
    return (omega ** 2) * (np.abs(H) ** 2) * (G_FORCE ** 2) * S_eta


def autocorr_from_spectrum(omega, S_v, tau_grid):
    """R(tau) = ∫ S_v(w) cos(w tau) dw (单边, 实偶)。分块防大矩阵。"""
    R = np.empty_like(tau_grid)
    CH = 400
    for s in range(0, len(tau_grid), CH):
        tt = tau_grid[s:s + CH]
        R[s:s + CH] = np.trapezoid(S_v[None, :] * np.cos(np.outer(tt, omega)),
                                   omega, axis=1)
    return R


def tauc_K_from_R(R, tau_grid):
    """tau_c=∫R^2/R(0)^2 (式6), K=4∫tau R^2/R(0)^2 (有限-t 修正系数)。"""
    R0sq = R[0] ** 2
    tau_c = 2.0 * np.trapezoid(R ** 2, tau_grid) / R0sq        # 2∫_0^∞ = ∫_{-∞}^∞
    K = 4.0 * np.trapezoid(tau_grid * R ** 2, tau_grid) / R0sq
    return tau_c, K


def eps2_exact(t, R, tau_grid):
    """式(5) 精确: eps^2(t)=4∫_0^t (t-tau)R^2 dtau /(R(0) t)^2。"""
    msk = tau_grid <= t
    tt, RR = tau_grid[msk], R[msk]
    var = 4.0 * np.trapezoid((t - tt) * RR ** 2, tt)
    return var / (R[0] * t) ** 2


# ==========================================================================
# 2. (A) OU 解析锚点: 真实 OU Langevin 积分 (非周期, 无 FFT 圆周假象)
# ==========================================================================
def run_ou_anchor():
    sigma2, tau_r = 1.0, 3.0
    sigma = np.sqrt(sigma2)
    if QUICK:
        dt, T, N_real, BATCH = 0.05, 300.0, 300, 150
        windows = [20, 40, 80, 160, 250]
    else:
        dt, T, N_real, BATCH = 0.02, 600.0, 4000, 500
        windows = [20, 40, 80, 160, 320, 500]
    n = int(round(T / dt))
    wsteps = [int(round(Tw / dt)) for Tw in windows]
    a = np.exp(-dt / tau_r)
    b = sigma * np.sqrt(1 - a ** 2)
    rng = np.random.default_rng(7)

    W_at = {Tw: np.empty(N_real) for Tw in windows}
    var_acc, var_cnt, done = 0.0, 0, 0
    while done < N_real:
        nb = min(BATCH, N_real - done)
        v = np.empty((n, nb))
        v[0] = rng.standard_normal(nb) * sigma
        for i in range(n - 1):
            v[i + 1] = a * v[i] + b * rng.standard_normal(nb)
        var_acc += float(v.var(axis=0).sum()); var_cnt += nb
        v2 = v * v
        cum = np.empty_like(v2); cum[0] = 0.0
        np.cumsum(0.5 * (v2[1:] + v2[:-1]) * dt, axis=0, out=cum[1:])
        for Tw, ns in zip(windows, wsteps):
            W_at[Tw][done:done + nb] = cum[ns]
        done += nb

    # 解析 R(tau)=sigma^2 e^{-|tau|/tau_r}: tau_c=tau_r, K=tau_r^2, eps^2·t=2tau_r - tau_r^2/t
    tau_c, K = tau_r, tau_r ** 2
    rng_b = np.random.default_rng(101)
    rows = []
    for Tw in windows:
        W = W_at[Tw]
        mW, vW = W.mean(), W.var(ddof=1)
        e2 = vW / mW ** 2
        # bootstrap 不确定性
        idx = rng_b.integers(0, len(W), size=(200, len(W)))
        bs = W[idx]
        e2_bs = bs.var(axis=1, ddof=1) / bs.mean(axis=1) ** 2
        rows.append(dict(t=Tw, mean_W=mW, var_W=vW, eps2=e2,
                         eps2_t=e2 * Tw, eps2_err=float(e2_bs.std()),
                         eps2_t_err=float(e2_bs.std() * Tw),
                         eps2_exact=(2 * tau_c / Tw - K / Tw ** 2),
                         eps2_t_exact=(2 * tau_c - K / Tw)))
    return dict(model="OU", sigma2=sigma2, tau_r=tau_r, tau_c=tau_c, K=K,
                B_eff=1.0 / tau_c, dt=dt, T=T, N_real=N_real,
                sigma2_emp=var_acc / var_cnt, rows=rows)


# ==========================================================================
# 3. (B) 波浪收集器: 真实 JONSWAP 谱 + irfft 随机相位精确稳态合成
# ==========================================================================
def synth_velocity_irfft(amp, n, nb, rng):
    """一次性批量合成 nb 条稳态速度 v(t): irfft 实现随机相位法。
    amp 已按 |C_k|=(n/2)√(2 S_v(w_k) dw) 设定, 使 Var(v)=∫S_v dw。"""
    phi = rng.uniform(0.0, 2 * np.pi, size=(amp.shape[0], nb))
    C = amp[:, None] * np.exp(1j * phi)
    return np.fft.irfft(C, n=n, axis=0)


def run_wave_harvester():
    c_load = C_LOAD_STAR
    # --- 解析 R(tau), tau_c, K, 精确 eps^2 (细网格) ---
    omega = np.linspace(1e-3, 6.0, 4000)
    S_v = velocity_spectrum(omega, c_load)
    tau_grid = np.arange(0.0, 400.0, 0.02)
    R = autocorr_from_spectrum(omega, S_v, tau_grid)
    R0 = R[0]                              # = sigma_v^2
    tau_c, K = tauc_K_from_R(R, tau_grid)
    B_eff = 1.0 / tau_c

    # --- irfft 合成参数: 长周期 T 使 t<<T, 避开圆周假象 ---
    if QUICK:
        dt, T, N_real, BATCH = 0.15, 3000.0, 300, 100
        windows = [40, 80, 120, 200, 300]
    else:
        dt, T, N_real, BATCH = 0.08, 6000.0, 4000, 150
        windows = [40, 80, 120, 160, 240, 320]
    n = int(round(T / dt))
    df = 2 * np.pi / (n * dt)
    kk = np.arange(n // 2 + 1)
    w_fft = kk * df
    Sv_f = velocity_spectrum(w_fft, c_load)
    amp = (n / 2.0) * np.sqrt(2.0 * np.maximum(Sv_f, 0.0) * df)
    amp[0] = 0.0
    wsteps = [int(round(Tw / dt)) for Tw in windows]
    rng = np.random.default_rng(2026)

    W_at = {Tw: np.empty(N_real) for Tw in windows}
    sig_acc, cnt, done = 0.0, 0, 0
    acf_emp = None
    while done < N_real:
        nb = min(BATCH, N_real - done)
        v = synth_velocity_irfft(amp, n, nb, rng)
        sig_acc += float(v.var(axis=0).sum())
        v2 = v * v
        cum = np.empty_like(v2); cum[0] = 0.0
        np.cumsum(0.5 * (v2[1:] + v2[:-1]) * dt, axis=0, out=cum[1:])
        for Tw, ns in zip(windows, wsteps):
            W_at[Tw][done:done + nb] = cum[ns]
        if acf_emp is None:                # 用首批 FFT 自相关交叉验证 tau_c (近乎免费)
            vc = v - v.mean(axis=0, keepdims=True)
            P = np.abs(np.fft.rfft(vc, axis=0)) ** 2
            ac = np.fft.irfft(P, n=n, axis=0)[: int(120 / dt)].mean(axis=1)
            acf_emp = ac
        cnt += nb; done += nb

    sigma_v2_emp = sig_acc / cnt
    lags = np.arange(len(acf_emp)) * dt
    tau_c_emp = 2.0 * np.trapezoid((acf_emp / acf_emp[0]) ** 2, dx=dt)

    rng_b = np.random.default_rng(202)
    rows = []
    for Tw in windows:
        W = W_at[Tw]
        mW, vW = W.mean(), W.var(ddof=1)
        e2 = vW / mW ** 2
        idx = rng_b.integers(0, len(W), size=(200, len(W)))
        bs = W[idx]
        e2_bs = bs.var(axis=1, ddof=1) / bs.mean(axis=1) ** 2
        rows.append(dict(t=Tw, mean_W=mW, var_W=vW, eps2=e2, eps2_t=e2 * Tw,
                         eps2_err=float(e2_bs.std()),
                         eps2_t_err=float(e2_bs.std() * Tw),
                         eps2_exact=eps2_exact(Tw, R, tau_grid),
                         eps2_t_exact=eps2_exact(Tw, R, tau_grid) * Tw))
    # 供画图的密集解析曲线
    t_dense = np.linspace(windows[0] * 0.6, windows[-1] * 1.05, 60)
    e2_curve = np.array([eps2_exact(tt, R, tau_grid) for tt in t_dense])
    return dict(model="wave_JONSWAP", c_load=c_load, Hs=Hs, Tp=Tp,
                sigma_v2=R0, sigma_v2_emp=sigma_v2_emp,
                tau_c=tau_c, tau_c_emp=tau_c_emp, K=K, B_eff=B_eff,
                dt=dt, T=T, N_real=N_real, rows=rows,
                t_dense=t_dense.tolist(), eps2_t_curve=(e2_curve * t_dense).tolist(),
                two_tau_c=2 * tau_c)


# ==========================================================================
# 4. (C) 功率-精度权衡: 沿 c_load 解析扫描 (新现象)
# ==========================================================================
def run_power_precision_tradeoff(t_obs=200.0):
    omega = np.linspace(1e-3, 6.0, 3000)
    tau_grid = np.arange(0.0, 300.0, 0.04)
    c_arr = np.linspace(0.02, 1.2, 40)
    P, B, e2 = [], [], []
    for c in c_arr:
        S_v = velocity_spectrum(omega, c)
        R = autocorr_from_spectrum(omega, S_v, tau_grid)
        sig2 = R[0]
        tau_c, _ = tauc_K_from_R(R, tau_grid)
        P.append(c * sig2)                 # 平均功率 P=c_load<v^2> (与 exp2 一致)
        B.append(1.0 / tau_c)
        e2.append(eps2_exact(t_obs, R, tau_grid))
    P = np.array(P); B = np.array(B); e2 = np.array(e2)
    i_pmax = int(np.argmax(P))
    i_bmax = int(np.argmax(B))             # 精度最高 = B_eff 最大
    return dict(t_obs=t_obs, c_arr=c_arr.tolist(), P=P.tolist(),
                B_eff=B.tolist(), eps2=e2.tolist(),
                c_at_Pmax=float(c_arr[i_pmax]), P_max=float(P[i_pmax]),
                c_at_Bmax=float(c_arr[i_bmax]), B_max=float(B[i_bmax]),
                B_eff_at_Pmax=float(B[i_pmax]), eps2_at_Pmax=float(e2[i_pmax]))


# ==========================================================================
# 5. 画图 (1×3): 无总标题/无分图标题, 仅 (a)(b)(c) 角标, 英文(投稿规范)
# ==========================================================================
def make_figure(ou, wave, trade):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.8, 5.0))

    # (a) OU 锚点: eps^2·t vs t, 数值点 + 精确曲线 + 渐近线
    t_o = np.array([r["t"] for r in ou["rows"]], float)
    y_o = np.array([r["eps2_t"] for r in ou["rows"]], float)
    ye_o = np.array([r["eps2_t_err"] for r in ou["rows"]], float)
    tt = np.linspace(t_o.min() * 0.6, t_o.max() * 1.05, 200)
    ax1.plot(tt, 2 * ou["tau_c"] - ou["K"] / tt, "-", color=C_BLUE, lw=2.2,
             label=TR(r"Exact: $2\tau_c-\tau_r^2/t$", r"精确: $2\tau_c-\tau_r^2/t$"))
    ax1.axhline(2 * ou["tau_c"], ls="--", color=C_GREY, lw=1.6,
                label=TR(r"$2\tau_c$ (asympt.)", r"$2\tau_c$ (渐近)"))
    ax1.errorbar(t_o, y_o, yerr=ye_o, fmt="o", color=C_RED, ms=7, capsize=3,
                 ecolor=C_RED, elinewidth=1.3,
                 label=TR("OU simulation", "OU 模拟"))
    ax1.set_xlabel(TR(r"Observation time $t$", r"观测时间 $t$"), fontsize=12)
    ax1.set_ylabel(TR(r"$\varepsilon^2(t)\cdot t$", r"$\varepsilon^2(t)\cdot t$"), fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes, fontsize=14,
             fontweight="bold", va="top")
    ax1.text(0.97, 0.30, TR(r"OU anchor, $\tau_r=%.0f$" % ou["tau_r"],
                            r"OU 锚点, $\tau_r=%.0f$" % ou["tau_r"]),
             transform=ax1.transAxes, fontsize=9.5, ha="right", va="bottom",
             color="#333333")

    # (b) 波浪: eps^2·t vs t, 数值点 + 零参数精确曲线 + 2tau_c 渐近线
    t_w = np.array([r["t"] for r in wave["rows"]], float)
    y_w = np.array([r["eps2_t"] for r in wave["rows"]], float)
    ye_w = np.array([r["eps2_t_err"] for r in wave["rows"]], float)
    ax2.plot(wave["t_dense"], wave["eps2_t_curve"], "-", color=C_BLUE, lw=2.2,
             label=TR(r"Exact (Eq. 5), zero-param", r"精确 (式5), 零参数"))
    ax2.axhline(wave["two_tau_c"], ls="--", color=C_GREY, lw=1.6,
                label=TR(r"$2\tau_c=%.1f$" % wave["two_tau_c"],
                         r"$2\tau_c=%.1f$" % wave["two_tau_c"]))
    ax2.errorbar(t_w, y_w, yerr=ye_w, fmt="s", color=C_RED, ms=7, capsize=3,
                 ecolor=C_RED, elinewidth=1.3,
                 label=TR("Spectral synthesis", "谱合成模拟"))
    ax2.set_xlabel(TR(r"Observation time $t$ (s)", r"观测时间 $t$ (s)"), fontsize=12)
    ax2.set_ylabel(TR(r"$\varepsilon^2(t)\cdot t$ (s)", r"$\varepsilon^2(t)\cdot t$ (s)"), fontsize=12)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=14,
             fontweight="bold", va="top")
    ax2.text(0.97, 0.30, TR(r"JONSWAP, $H_s$=%.0f m, $T_p$=%.0f s" % (Hs, Tp),
                            r"JONSWAP, $H_s$=%.0f m, $T_p$=%.0f s" % (Hs, Tp)),
             transform=ax2.transAxes, fontsize=9.5, ha="right", va="bottom",
             color="#333333")

    # (c) 功率-精度权衡: 双 y 轴, P(c_load) 与 B_eff(c_load), 标峰值错位
    c = np.array(trade["c_arr"], float)
    P = np.array(trade["P"], float)
    B = np.array(trade["B_eff"], float)
    ax3.plot(c, P, "-", color=C_BLUE, lw=2.4,
             label=TR(r"Power $P=c_{load}\langle v^2\rangle$",
                      r"功率 $P=c_{load}\langle v^2\rangle$"))
    ax3.axvline(trade["c_at_Pmax"], ls=":", color=C_BLUE, lw=1.5)
    ax3.plot([trade["c_at_Pmax"]], [trade["P_max"]], "*", color=C_BLUE, ms=18)
    ax3.set_xlabel(TR(r"Load damping $c_{load}$ (N$\cdot$s/m)",
                      r"负载阻尼 $c_{load}$ (N$\cdot$s/m)"), fontsize=12)
    ax3.set_ylabel(TR(r"Mean power $P$ (norm.)", r"平均功率 $P$ (归一化)"),
                   fontsize=12, color=C_BLUE)
    ax3.tick_params(axis="y", labelcolor=C_BLUE)
    ax3b = ax3.twinx()
    ax3b.plot(c, B, "-", color=C_GREEN, lw=2.4,
              label=TR(r"Bandwidth $B_{eff}=1/\tau_c$",
                       r"带宽 $B_{eff}=1/\tau_c$"))
    ax3b.set_ylabel(TR(r"Effective bandwidth $B_{eff}$ (rad/s)",
                       r"有效带宽 $B_{eff}$ (rad/s)"), fontsize=12, color=C_GREEN)
    ax3b.tick_params(axis="y", labelcolor=C_GREEN)
    ax3.text(0.02, 0.97, "(c)", transform=ax3.transAxes, fontsize=14,
             fontweight="bold", va="top")
    ax3.text(trade["c_at_Pmax"], trade["P_max"],
             TR("  power-optimal", "  功率最优"), fontsize=9.5,
             color=C_BLUE, va="bottom", ha="left")
    # 图例合并
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    # 置于两条曲线之间的右侧空带, 避免与上升的绿色带宽曲线相叠
    ax3.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center right",
               bbox_to_anchor=(0.985, 0.52))

    fig.tight_layout()
    png = os.path.join(OUTDIR, "exp3_spectral_tur.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


# ==========================================================================
# 6. main
# ==========================================================================
def main():
    t0 = time.time()
    print(f"[exp3] 谱 TUR 验证  (QUICK={QUICK}, LANG={LANG})", flush=True)
    print("  (A) OU 解析锚点: 真实 OU Langevin ...", flush=True)
    ou = run_ou_anchor()
    print(f"      tau_c={ou['tau_c']:.3f}  sigma^2(emp)={ou['sigma2_emp']:.4f}"
          f"  (解析 {ou['sigma2']})", flush=True)
    print("  (B) 波浪收集器: JONSWAP 谱 + irfft 精确稳态合成 ...", flush=True)
    wave = run_wave_harvester()
    print(f"      tau_c={wave['tau_c']:.4f} (emp {wave['tau_c_emp']:.4f})  "
          f"sigma_v^2={wave['sigma_v2']:.4f} (emp {wave['sigma_v2_emp']:.4f})  "
          f"2tau_c={wave['two_tau_c']:.3f}", flush=True)
    print("  (C) 功率-精度权衡: 沿 c_load 解析扫描 ...", flush=True)
    trade = run_power_precision_tradeoff()
    print(f"      功率最优 c*={trade['c_at_Pmax']:.3f} (P_max={trade['P_max']:.4f}); "
          f"该点 B_eff={trade['B_eff_at_Pmax']:.4f}; "
          f"带宽最大处 c={trade['c_at_Bmax']:.3f}", flush=True)

    png = make_figure(ou, wave, trade)

    # ---- 控制台核对表 ----
    print(f"\n[A] OU:   {'t':>5} {'eps2*t(num)':>12} {'+/-':>7} {'eps2*t(exact)':>14}")
    for r in ou["rows"]:
        print(f"          {r['t']:5.0f} {r['eps2_t']:12.4f} {r['eps2_t_err']:7.4f} "
              f"{r['eps2_t_exact']:14.4f}")
    print(f"\n[B] wave: {'t':>5} {'eps2*t(num)':>12} {'+/-':>7} {'eps2*t(exact)':>14} "
          f"{'eps2*t*Beff':>12}")
    for r in wave["rows"]:
        print(f"          {r['t']:5.0f} {r['eps2_t']:12.4f} {r['eps2_t_err']:7.4f} "
              f"{r['eps2_t_exact']:14.4f} {r['eps2_t']*wave['B_eff']:12.4f}")

    summary = dict(
        experiment="exp3_spectral_tur",
        description="Spectral TUR: derivation + numerical verification "
                    "(OU anchor, JONSWAP-driven harvester, power-precision tradeoff).",
        quick=QUICK, lang=LANG, runtime_sec=round(time.time() - t0, 2),
        collector=dict(m=M_MASS, c_mech=C_MECH, k=K_STIFF, g=G_FORCE,
                       Hs=Hs, Tp=Tp, omega_p=OMEGA_P, gamma_jonswap=GAMMA_J,
                       c_load_star=C_LOAD_STAR),
        ou_anchor=ou, wave_harvester=wave, power_precision=trade,
        figure=os.path.basename(png),
        references=[
            "Isserlis, Biometrika 12, 134 (1918)",
            "Koyuk & Seifert, PRL 125, 260604 (2020)",
            "Barato & Seifert, PRL 114, 158101 (2015)",
            "Pierson & Moskowitz, JGR 69, 5181 (1964)",
            "Hasselmann et al. (JONSWAP), DHZ A8, 12 (1973)",
            "Falnes, Ocean Waves and Oscillating Systems (2002)",
            "Uhlenbeck & Ornstein, Phys. Rev. 36, 823 (1930)",
        ],
    )
    js = os.path.join(OUTDIR, "exp3_metrics.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[exp3] 图: {png}")
    print(f"[exp3] 指标: {js}")
    print(f"[exp3] 用时 {summary['runtime_sec']} s")
    return summary


if __name__ == "__main__":
    main()
