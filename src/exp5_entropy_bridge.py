#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验五 (exp5): 线性非平衡稳态 (NESS) 的【熵产生桥接】。

目标 (回应"谱关系到底是不是热力学的"这个最深的问题):
  我之前推出的关系是 eps^2(t)*t*B_eff -> 2 (线性高斯收集器精确饱和)。它在【形式】上像
  热力学不确定性关系 (TUR: eps^2 * Sigma_t >= 2k_B), 但我没有证明 t*B_eff 就是真实熵产生
  Sigma_t/k_B。本实验把这件事做严格:
    1. 把有色驱动嵌入为一个 Ornstein-Uhlenbeck 辅助模 y, 机械+负载阻尼 c 接一个温度 T 的
       热浴 (满足涨落耗散), 于是 (x, v, y) 是一个【线性高斯马尔可夫过程】, 熵产生有严格定义;
    2. 解 Lyapunov 方程 A Σ + Σ A^T + B B^T = 0 得稳态协方差 Σ;
    3. 用 Sekimoto 随机能量学算【熵产生率】 sigma_dot = (c/T)(<v^2> - T)
       (= 热浴吸热率 / T; 平衡态<v^2>=T 时严格为 0 — 用此自校验);
    4. 由速度自相关 R_v(s)=[exp(As)Σ]_vv 算 tau_c、B_eff=1/tau_c;
    5. 用我的精确公式算收集功 W_t=c_load∫v^2 的 eps^2(t), 然后【同时】检验两条关系:
         (热力学型)  eps^2 * Sigma_t  与 2k_B 的关系;
         (动力学型)  eps^2 * t * B_eff 与 2 的关系。

  关键科学发现 (本脚本将用数字展示, 见输出与下方解读):
    收集功 W_t = c_load∫v^2 在时间反演下是【偶】的 (v->-v, v^2 不变), 它是耗散型/时间对称
    可观测量, 【不是】TUR 所约束的那类 (奇) 流。因此 eps^2 * Sigma_t 【没有】普适下限
    (弱驱动时 Sigma_t->0 而 eps^2 有限, 乘积 ->0, 远低于 2k_B); 而 eps^2 * t * B_eff 始终 ->2。
    这说明: 这条谱关系的正确热力学归属是【动力学活动 / 动理学不确定性关系 (KUR)】这一族
    (约束时间对称量的涨落, 用"动力学活动"而非熵产生), 其中 t*B_eff = t/tau_c 扮演
    "观测窗内独立涨落的有效个数"(一种动力学活动量度) 的角色, 并被线性高斯动力学精确饱和。

  这是一个把结果【正确归位】的桥接, 而不是把它硬说成 TUR。所有数字来自本脚本真实计算。
"""

import os
import sys
import json
import time
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm

OUTDIR = os.environ.get("PIDM_OUTDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

KB = 1.0  # 约化单位 (玻尔兹曼常数), 与项目其余部分一致


def build_system(k, c, g, T, tau_d, sigma_y2):
    """构造 3 维线性 SDE  dz = A z dt + B dW 的 A 与 BB^T。
    状态 z=(x, v, y): x 位置, v 速度(动量, m=1), y 有色驱动(OU 模)。
      dx = v dt
      dv = (-k x - c v + g y) dt + sqrt(2 c T) dW_th      # 阻尼 c 接温度 T 的热浴(FDT)
      dy = -(1/tau_d) y dt + sqrt(2 sigma_y2 / tau_d) dW_y  # 有色驱动: 稳态方差 sigma_y2, 相关时间 tau_d
    返回 A, BBt。"""
    A = np.array([
        [0.0,      1.0,            0.0],
        [-k,      -c,              g],
        [0.0,      0.0,           -1.0 / tau_d],
    ])
    BBt = np.diag([0.0, 2.0 * c * T, 2.0 * sigma_y2 / tau_d])
    return A, BBt


def stationary_cov(A, BBt):
    """解 Lyapunov 方程 A Σ + Σ A^T + BB^T = 0 (即 A Σ + Σ A^T = -BB^T)。"""
    return solve_continuous_lyapunov(A, -BBt)


def velocity_autocorr(A, Sigma, s_grid):
    """速度自相关 R_v(s) = [exp(A s) Σ]_{vv} (平稳高斯过程的标准结果, s>=0)。
    用一次 expm(A*ds) 作传播子反复相乘, 高效得到整条 R_v(s)。"""
    ds = s_grid[1] - s_grid[0]
    P = expm(A * ds)                 # 单步传播子
    R = np.empty_like(s_grid)
    M = np.eye(A.shape[0])           # exp(A * 0) = I
    for n in range(len(s_grid)):
        R[n] = (M @ Sigma)[1, 1]     # 取 (v, v) 分量
        M = M @ P
    return R


def eps2_exact(R, s_grid, t):
    """我的精确公式: eps^2(t) = 4∫_0^t (t-τ) R_v(τ)^2 dτ / (R_v(0) t)^2 (线性高斯, Isserlis)。"""
    mask = s_grid <= t
    tau = s_grid[mask]
    integrand = (t - tau) * R[mask] ** 2
    num = 4.0 * np.trapezoid(integrand, tau)
    return num / (R[0] * t) ** 2


def analyze(label, k, c, c_load, g, T, tau_d, sigma_y2, s_grid, t_list):
    """对一组参数: 解 Σ, 算熵产生率、B_eff, 并对每个 t 同时给出
       热力学型乘积 eps^2*Sigma_t 与 动力学型乘积 eps^2*t*B_eff。"""
    A, BBt = build_system(k, c, g, T, tau_d, sigma_y2)
    Sigma = stationary_cov(A, BBt)
    v2 = Sigma[1, 1]                              # <v^2>

    # 熵产生率 (Sekimoto): 热浴吸热率 = c<v^2> - c T; 除以 T 即 EP 率。平衡态(无驱动)=0。
    sigma_dot = (c * v2 - c * T) / T
    P_harv = c_load * v2                          # 收集功率 (负载耗散)

    R = velocity_autocorr(A, Sigma, s_grid)
    R0 = R[0]
    # tau_c = ∫_{-∞}^∞ R^2 / R(0)^2 = 2∫_0^∞ R^2/R0^2;  B_eff = 1/tau_c
    tau_c = 2.0 * np.trapezoid(R ** 2, s_grid) / (R0 ** 2)
    B_eff = 1.0 / tau_c

    rows = []
    for t in t_list:
        e2 = eps2_exact(R, s_grid, t)
        Sigma_t = sigma_dot * t                   # 时间 t 内的总熵产生 (k_B 单位)
        rows.append({
            "t": t,
            "eps2": e2,
            "eps2_t_Beff": e2 * t * B_eff,          # 动力学型: 应 ->2
            "eps2_Sigma_t": e2 * Sigma_t,           # 热力学(TUR)型: 看是否 >=2k_B
        })
    return {
        "label": label, "sigma_y2": sigma_y2, "tau_d": tau_d,
        "v2": v2, "T": T, "sigma_dot": sigma_dot, "P_harv": P_harv,
        "tau_c": tau_c, "B_eff": B_eff,
        "sigma_dot_over_Beff": sigma_dot / B_eff if B_eff > 0 else float('nan'),
        "rows": rows,
    }


def main():
    t0 = time.time()
    # --- 物理参数 (与项目波浪收集器同量级) ---
    OMEGA_P = 2 * np.pi / 8.0
    k = OMEGA_P ** 2          # 刚度
    c_mech, c_load = 0.05, 0.118
    c = c_mech + c_load       # 总阻尼 = 接热浴的耗散通道
    g = 1.0
    T = 1.0                   # 冷浴温度 (约化)
    tau_d = 8.0               # 有色驱动相关时间 (~浪周期), 体现"有色"

    s_grid = np.arange(0.0, 200.0, 0.02)   # 自相关/积分网格 (足够长以含多个 tau_c)
    t_list = [20, 60, 120, 200]

    print("=" * 80)
    print("实验五 — 线性 NESS 熵产生桥接: 谱关系的正确热力学归属")
    print("=" * 80)
    print(f"参数: k={k:.4f} c={c} (c_mech={c_mech}+c_load={c_load}) g={g} T={T} tau_d={tau_d}")
    print("自校验: 无驱动 (sigma_y2=0) 时应得 <v^2>=T 且熵产生率 sigma_dot=0 (平衡态)。\n")

    # 驱动强度从 0(平衡) 扫到强非平衡
    sigma_y2_list = [0.0, 0.5, 2.0, 8.0, 32.0]
    results = []
    for sy2 in sigma_y2_list:
        r = analyze(f"sigma_y2={sy2}", k, c, c_load, g, T, tau_d, sy2, s_grid, t_list)
        results.append(r)

    # --- 自校验输出 ---
    eq = results[0]
    print(f"[自校验 平衡态 sigma_y2=0]  <v^2>={eq['v2']:.6f} (应=T={T})   "
          f"sigma_dot={eq['sigma_dot']:.3e} (应=0)")
    ok_equil = abs(eq["v2"] - T) < 1e-9 and abs(eq["sigma_dot"]) < 1e-9
    print(f"   平衡态校验: {'通过 ✓ (熵产生定义正确)' if ok_equil else '未通过 ✗'}\n")

    # --- 主表: 各驱动强度下的 EP 率、B_eff, 以及两条关系 ---
    print("各驱动强度下 (取最大观测窗 t={}):".format(t_list[-1]))
    print(f"{'sigma_y2':>9} {'<v^2>':>9} {'EP率σ̇':>10} {'B_eff':>9} {'σ̇/B_eff':>9} "
          f"{'ε²·t·B_eff':>11} {'ε²·Σ_t':>10}")
    for r in results:
        last = r["rows"][-1]
        print(f"{r['sigma_y2']:>9.2f} {r['v2']:>9.4f} {r['sigma_dot']:>10.4f} "
              f"{r['B_eff']:>9.4f} {r['sigma_dot_over_Beff']:>9.3f} "
              f"{last['eps2_t_Beff']:>11.4f} {last['eps2_Sigma_t']:>10.4f}")

    # --- 关键解读 (用一个中等驱动档展示 eps^2*t*B_eff 随 t ->2, 而 eps^2*Sigma_t 不收敛到 2) ---
    mid = results[2]   # sigma_y2=2.0
    print(f"\n[随观测时间 t 的行为, sigma_y2={mid['sigma_y2']}]  "
          f"(σ̇={mid['sigma_dot']:.4f}, B_eff={mid['B_eff']:.4f})")
    print(f"{'t':>6} {'ε²·t·B_eff(→2?)':>16} {'ε²·Σ_t(有普适下限?)':>20}")
    for row in mid["rows"]:
        print(f"{row['t']:>6.0f} {row['eps2_t_Beff']:>16.4f} {row['eps2_Sigma_t']:>20.4f}")

    print("\n" + "-" * 80)
    print("解读:")
    print(" · ε²·t·B_eff 在所有驱动强度下都 ->2 (线性高斯精确饱和) —— 这是【动力学/谱】事实。")
    print(" · ε²·Σ_t 【没有】普适下限: 它随驱动强度从近 0 (弱驱动, Σ_t->0) 升到很大;")
    print("   弱驱动时远低于 2k_B —— 因为收集功 W=c∫v² 是【时间对称(偶)】的耗散量,")
    print("   不是 TUR 约束的那类(奇)流。故这条关系的正确归属是【动理学不确定性关系/动力学活动】,")
    print("   其中 t·B_eff=t/τ_c 是观测窗内独立涨落的有效个数(一种活动量度), 被线性高斯精确饱和。")
    print(" · 因此严格的命题是: Σ̇ 与 B_eff 是两个独立的量 (前者热力学、后者动力学);")
    print("   收集精度的天花板由【动力学活动 t·B_eff】设定, 而非熵产生。")

    runtime = round(time.time() - t0, 2)
    summary = {
        "experiment": "exp5_entropy_bridge",
        "description": "Rigorous placement of the spectral relation within stochastic "
                       "thermodynamics. Linear NESS (colored OU drive + thermal bath), entropy "
                       "production via Sekimoto heat flow (validated: equilibrium -> 0). Finding: "
                       "harvested work is time-symmetric (even), so it is governed by a kinetic "
                       "uncertainty relation (dynamical activity t*B_eff), NOT the entropy-production "
                       "TUR. eps^2*t*B_eff -> 2 universally; eps^2*Sigma_t has no universal floor.",
        "runtime_sec": runtime,
        "params": {"k": k, "c_mech": c_mech, "c_load": c_load, "c_total": c,
                   "g": g, "T": T, "tau_d": tau_d, "kB": KB},
        "equilibrium_check_passed": bool(ok_equil),
        "sweep": results,
    }
    os.makedirs(OUTDIR, exist_ok=True)
    js = os.path.join(OUTDIR, "exp5_metrics.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[exp5] 指标: {js}")
    print(f"[exp5] 用时 {runtime} s")
    return summary


if __name__ == "__main__":
    main()
