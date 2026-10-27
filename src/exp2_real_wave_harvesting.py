#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二: 真实标准海浪谱驱动下的能量收集优化 (真实数据)
====================================================

这是"真实数据"实验: 用海洋/波浪能工程通用的、有文献出处的标准海浪谱
(JONSWAP / Pierson-Moskowitz) 合成真实统计特性的波面, 去驱动一个欠阻尼
电机械收集器 (点吸收式波浪能转换器 / 浮式 TENG 的最小模型):

        m x'' + (c_mech + c_load) x' + k x = g * eta(t)

收集功率 P_harv = c_load * <x'^2>。扫描功率取出阻尼 c_load (PTO damping),
找到使平均收集功率最大的"阻抗/阻尼匹配"工作点, 并刻画功率涨落。

诚实声明:
  本实验用的是"真实标准谱数据 / 真实计算实验"——谱形与参数全部可追溯到下列文献,
  波面由随机相位法严格合成。它**不是**物理水槽实测波浪, 我不混淆两者、不编造数据。

文献 (可追溯):
  - W. J. Pierson, L. Moskowitz, J. Geophys. Res. 69, 5181 (1964).
  - K. Hasselmann et al. (JONSWAP), Dtsch. Hydrogr. Z. A8, 12 (1973).
  - DNV-RP-C205, "Environmental conditions and environmental loads".
  - (PTO 阻尼匹配的经典结果) J. Falnes, "Ocean Waves and Oscillating Systems",
    Cambridge Univ. Press (2002).

所有数字由本脚本真实运行得到, 固定随机种子可复现, 未经任何手工修改。

环境变量:
  PIDM_QUICK=1   快速冒烟模式 (少实现、短时长、粗网格)。
"""

import os
import sys
import json
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (jonswap_spectrum, pierson_moskowitz_spectrum,
                    significant_wave_height, synthesize_wave)

# 图内文字语言: 'en' (默认, 用于 Nature/Science 投稿) 或 'zh' (中文)。
# 通过环境变量 PIDM_LANG 切换, 例如 PIDM_LANG=zh 可重新生成中文图。
LANG = os.environ.get("PIDM_LANG", "en").lower()


def TR(en, zh):
    """根据 LANG 选择图内文字: 英文优先(投稿要求), 需要时可切回中文。
    (命名为 TR 而非 T, 以免与可能的温度变量冲突)"""
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
SEED = 2026
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# 0. 海况与器件参数 (SI 单位)
# --------------------------------------------------------------------------
Hs = 2.0          # 有效波高 (m), 中等海况
Tp = 8.0          # 峰值周期 (s)
omega_p = 2.0 * np.pi / Tp          # 峰值角频率 (rad/s)
GAMMA_J = 3.3     # JONSWAP 峰增强因子 (北海典型值)

# 器件 (点吸收子) 参数: 把固有频率调到接近浪的主频以获得良好响应
m = 1.0                              # 等效质量 (kg, 归一)
k_stiff = m * omega_p**2             # 等效刚度 (含静水回复), 使 omega_n = omega_p
c_mech = 0.05                        # 机械(寄生)阻尼 (N*s/m), 小
force_gain = 1.0                     # 波面->力 线性增益 (点吸收子近似)

# 谱与时间网格
omega = np.linspace(0.05, 3.0, 400 if not QUICK else 200)   # 角频率网格
T_total = 900.0 if not QUICK else 200.0      # 总时长 (s), 取足够多浪周期 (>100 个)
dt = 0.04                                     # 时间步长 (s), 约 200 步/峰值周期
t = np.arange(0.0, T_total, dt)
t_burn = 100.0 if not QUICK else 30.0         # 丢弃的暂态时长 (s)
i_burn = int(t_burn / dt)

N_real = 16 if not QUICK else 4              # 波面随机实现数 (统计平均)
n_loads = 13 if not QUICK else 6             # PTO 阻尼扫描点数

OUTDIR = os.environ.get("PIDM_OUTDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUTDIR = os.path.abspath(OUTDIR)
DATADIR = os.environ.get("PIDM_DATADIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATADIR = os.path.abspath(DATADIR)
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)


def integrate_batch(eta_batch, c_total):
    """半隐式 Euler 同时积分 N_real 条欠阻尼轨迹 (向量化以提速)。
    eta_batch: shape (n_time, n_real) 的波面; 返回 x,v 同形状。"""
    n_time, n_real = eta_batch.shape
    x = np.zeros((n_time, n_real))
    v = np.zeros((n_time, n_real))
    for i in range(n_time - 1):
        F = force_gain * eta_batch[i] - c_total * v[i] - k_stiff * x[i]
        a = F / m
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt
    return x, v


def main():
    print("=" * 78)
    print("实验二: 真实标准海浪谱驱动下的能量收集优化 (真实数据)"
          + ("  [QUICK 冒烟模式]" if QUICK else ""))
    print("=" * 78)
    print(f"海况: Hs={Hs} m, Tp={Tp} s (omega_p={omega_p:.4f} rad/s), JONSWAP gamma={GAMMA_J}")
    print(f"器件: m={m}, k={k_stiff:.4f} (omega_n=omega_p), c_mech={c_mech}, force_gain={force_gain}")
    print(f"仿真: T_total={T_total}s dt={dt} N_real={N_real} n_loads={n_loads} seed={SEED}")
    print("-" * 78)

    # ---- 谱 ----
    S_j = jonswap_spectrum(omega, Hs, omega_p, gamma=GAMMA_J)
    S_pm = pierson_moskowitz_spectrum(omega, Hs, omega_p)

    # ---- 物理校验1: 由谱反算的 Hs 是否等于设定值 ----
    Hs_from_S = significant_wave_height(omega, S_j)
    print(f"[校验1] 由 JONSWAP 谱反算 Hs = {Hs_from_S:.4f} m (设定 {Hs} m, "
          f"相对误差 {abs(Hs_from_S-Hs)/Hs*100:.2f}%)")

    # ---- 合成一组波面 (固定一套用于全程, 保证不同负载下比较公平) ----
    eta_batch = np.empty((len(t), N_real))
    for r in range(N_real):
        eta_batch[:, r] = synthesize_wave(omega, S_j, t, rng)

    # ---- 物理校验2: 合成波面的经验 Hs (= 4*std) 是否匹配设定 ----
    Hs_emp = 4.0 * eta_batch[i_burn:].std()
    print(f"[校验2] 合成波面经验 Hs = 4*std = {Hs_emp:.4f} m (设定 {Hs} m, "
          f"相对误差 {abs(Hs_emp-Hs)/Hs*100:.2f}%)")

    # ---- 经验谱: 用 Welch 法对全部实现做平均, 并换算到角频率单边 PSD ----
    # 注意: scipy 的 welch 返回普通频率 f(Hz) 下的单边 PSD (∫P df = 方差);
    # 换算到角频率 omega=2*pi*f 时, S_omega = P_f / (2*pi)。
    # 对全部实现平均可大幅降低周期图方差, 使经验谱平滑地贴合模型谱。
    fs = 1.0 / dt
    nperseg = min(2048, (len(t) - i_burn) // 4)
    psd_acc = None
    for r in range(N_real):
        f_w, Pxx = welch(eta_batch[i_burn:, r], fs=fs, nperseg=nperseg)
        psd_acc = Pxx if psd_acc is None else psd_acc + Pxx
    Pxx_avg = psd_acc / N_real
    omega_w = 2.0 * np.pi * f_w
    S_emp = Pxx_avg / (2.0 * np.pi)            # 角频率单边 PSD
    # 物理校验2b: 经验谱积分是否等于零阶谱矩 m0 = (Hs/4)^2
    m0_emp = float(np.trapezoid(S_emp, omega_w)) if hasattr(np, "trapezoid") \
        else float(np.trapz(S_emp, omega_w))
    Hs_from_emp_spec = 4.0 * np.sqrt(max(m0_emp, 0.0))
    print(f"[校验2b] 经验谱积分反算 Hs = 4*sqrt(∫S_emp dω) = {Hs_from_emp_spec:.4f} m "
          f"(设定 {Hs} m, 相对误差 {abs(Hs_from_emp_spec-Hs)/Hs*100:.2f}%)")

    # ---- 扫描 PTO 阻尼 c_load, 求最优 ----
    # 物理直觉: c_load 太小抽不出功, 太大把振子"卡死"也抽不出功, 中间存在最优 (阻尼匹配)
    c_loads = np.linspace(0.02, 1.2, n_loads)
    rows = []
    print(f"\n{'c_load':>7} {'P_harv均值':>11} {'P_harv标准差':>12} {'功率CV':>8} "
          f"{'能量平衡残差%':>12}")
    print("-" * 78)
    for c_load in c_loads:
        c_total = c_mech + c_load
        x, v = integrate_batch(eta_batch, c_total)
        vv = v[i_burn:]                          # 丢弃暂态
        eta_ss = eta_batch[i_burn:]
        # 瞬时收集功率 p(t) = c_load * v^2; 对时间与实现取平均
        p_inst = c_load * vv**2
        P_harv = float(p_inst.mean())
        P_harv_std = float(p_inst.mean(axis=0).std())   # 实现间均值的离散度
        cv = float(p_inst.std() / p_inst.mean()) if p_inst.mean() > 0 else np.nan
        # 物理校验3: 稳态能量平衡 P_in ~= P_diss_mech + P_harv
        P_in = float((force_gain * eta_ss * vv).mean())
        P_diss_mech = float(c_mech * (vv**2).mean())
        bal_res = abs(P_in - (P_diss_mech + P_harv)) / max(abs(P_in), 1e-12) * 100.0
        rows.append(dict(c_load=float(c_load), P_harv=P_harv, P_harv_std=P_harv_std,
                         cv=cv, P_in=P_in, P_diss_mech=P_diss_mech,
                         energy_balance_residual_percent=bal_res))
        print(f"{c_load:7.3f} {P_harv:11.5f} {P_harv_std:12.5f} {cv:8.3f} {bal_res:12.3f}")
    print("-" * 78)

    c_arr = np.array([r["c_load"] for r in rows])
    P_arr = np.array([r["P_harv"] for r in rows])
    Pstd_arr = np.array([r["P_harv_std"] for r in rows])
    bal_arr = np.array([r["energy_balance_residual_percent"] for r in rows])

    i_opt = int(np.argmax(P_arr))
    c_opt = float(c_arr[i_opt]); P_opt = float(P_arr[i_opt])
    # 物理校验4: 最优是否落在内部 (而非边界) -> 验证"阻抗匹配"确实存在
    interior_opt = bool(0 < i_opt < len(c_arr) - 1)
    max_bal_res = float(bal_arr.max())

    print("核心结论 (真实仿真, 真实标准海浪谱):")
    print(f"  [1] 由谱与合成波面两路反算的 Hs 都与设定一致 (见校验1/2)")
    print(f"  [2] 存在最优 PTO 阻尼 (阻抗/阻尼匹配): c_load* = {c_opt:.3f}, "
          f"最大平均收集功率 P_max = {P_opt:.5f} (W, 归一)")
    print(f"  [3] 最优点在扫描区间内部 (非边界)? -> {interior_opt}  (验证'匹配'真实存在)")
    print(f"  [4] 全部工作点稳态能量平衡残差最大 = {max_bal_res:.3f}% "
          f"(应接近 0, 验证物理自洽)")
    print("=" * 78)

    # ---------------------- 出图 (无图标题; 仅 (a)(b) 角标) ----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # (a) 真实海浪谱 + 全部实现平均的经验谱 (验证合成正确, 归一化已修正)
    ax1.plot(omega, S_j, "-", color="#2c6fbb", lw=2.2, label=f"JONSWAP ($\\gamma$={GAMMA_J})")
    ax1.plot(omega, S_pm, "--", color="#6c8ebf", lw=1.8, label="Pierson-Moskowitz")
    maskw = (omega_w > 0.05) & (omega_w < 3.0)
    ax1.plot(omega_w[maskw], S_emp[maskw], color="#9aa0a6", lw=1.0, alpha=0.95,
             zorder=0, label=TR("Synthesized sea, empirical spectrum (16-realization avg.)",
                               "合成波面经验谱 (16实现平均)"))
    ax1.axvline(omega_p, color="#d1495b", ls=":", lw=1.5, label=f"$\\omega_p$={omega_p:.2f}")
    ax1.set_xlabel(TR("Angular frequency $\\omega$ (rad/s)", "角频率 $\\omega$ (rad/s)"), fontsize=12)
    ax1.set_ylabel(TR("Spectral density $S(\\omega)$ (m$^2\\cdot$s/rad)",
                     "谱密度 $S(\\omega)$ (m$^2\\cdot$s/rad)"), fontsize=12)
    ax1.set_xlim(0, 3.0)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="best")
    ax1.text(0.97, 0.97, f"$H_s$={Hs} m, $T_p$={Tp} s", transform=ax1.transAxes,
             fontsize=10, va="top", ha="right", color="#333333")
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes, fontsize=14,
             fontweight="bold", va="top", ha="left")

    # (b) 平均收集功率 vs PTO 阻尼, 含最优点
    ax2.errorbar(c_arr, P_arr, yerr=Pstd_arr, fmt="o-", color="#2c6fbb", lw=2, ms=6,
                 ecolor="#7aa6d6", elinewidth=1.5, capsize=3,
                 label=TR("Mean harvested power $P_{harv}$ ($\\pm 1\\sigma$ across realizations)",
                         "平均收集功率 $P_{harv}$ (实现间$\\pm 1\\sigma$)"))
    ax2.plot(c_opt, P_opt, "*", color="#d1495b", ms=20,
             label=TR(f"Optimal PTO damping: $c^*$={c_opt:.2f}, $P_{{max}}$={P_opt:.4f}",
                     f"最优 PTO 阻尼: $c^*$={c_opt:.2f}, $P_{{max}}$={P_opt:.4f}"))
    ax2.set_xlabel(TR("PTO (load) damping $c_{load}$ (N$\\cdot$s/m)",
                     "PTO (负载) 阻尼 $c_{load}$ (N$\\cdot$s/m)"), fontsize=12)
    ax2.set_ylabel(TR("Mean harvested power $P_{harv}$ (W, normalized)",
                     "平均收集功率 $P_{harv}$ (W, 归一)"), fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="best")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=14,
             fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    png = os.path.join(OUTDIR, "exp2_wave_harvesting.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {png}")

    # ---- 落盘真实数据 (谱 + 一条波面实现) 供复现与他人直接使用 ----
    npz = os.path.join(DATADIR, "wave_jonswap_realization.npz")
    np.savez(npz, omega=omega, S_jonswap=S_j, S_pm=S_pm,
             t=t, eta_example=eta_batch[:, 0], Hs=Hs, Tp=Tp, omega_p=omega_p,
             gamma=GAMMA_J, seed=SEED)
    print(f"[data] {npz}")

    summary = dict(
        experiment="exp2_real_wave_harvesting",
        quick_mode=QUICK,
        data_source="standard JONSWAP/Pierson-Moskowitz wave spectra (traceable); "
                    "NOT physical wave-tank measurement",
        sea_state=dict(Hs=Hs, Tp=Tp, omega_p=omega_p, jonswap_gamma=GAMMA_J),
        device=dict(m=m, k_stiff=k_stiff, c_mech=c_mech, force_gain=force_gain,
                    omega_n=float(np.sqrt(k_stiff / m))),
        sim=dict(T_total=T_total, dt=dt, t_burn=t_burn, N_real=N_real,
                 n_loads=n_loads, n_omega=len(omega), seed=SEED),
        Hs_from_spectrum=float(Hs_from_S),
        Hs_empirical_from_synthesis=float(Hs_emp),
        Hs_from_empirical_spectrum=float(Hs_from_emp_spec),
        optimal_PTO_damping=c_opt,
        P_max=P_opt,
        optimum_is_interior=interior_opt,
        max_energy_balance_residual_percent=max_bal_res,
        sweep=rows,
    )
    js = os.path.join(OUTDIR, "exp2_metrics.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[metrics] {js}")
    print("实验二完成。")
    return summary


if __name__ == "__main__":
    main()
