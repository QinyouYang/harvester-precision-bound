#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验四 (exp4): 谱 TUR 干净关系 eps^2 * t * B_eff -> 2 的【适用域】数值检验。

动机 (诚实地回应"只对线性证明"这一边界):
  谱 TUR 的精确等式 eps^2(t)*t*B_eff -> 2 是用 Isserlis 定理对【高斯】速度推出的。
  线性收集器的稳态速度是高斯过程, 所以等式成立且被饱和。但真实器件常是【非线性】的
  (双稳、碰撞、硬化弹簧), 此时速度【不再是高斯过程】, 收集功 W_t=c_load∫v^2 的方差要用
  四阶【累积量】(而非仅二阶自相关 R^2), 因此用二阶谱量定义的 B_eff=1/tau_c 不再能保证
  eps^2*t*B_eff=2。本实验就用数值把这件事看清楚:
    - 线性 (beta=0): 必须复现 exp3 的波浪结果 (tau_c≈15.2, eps^2*t*B_eff->2)，
      这同时验证"时间积分路径"与 exp3"解析/irfft 路径"一致;
    - 非线性 (beta>0 / 双稳): 看 eps^2*t*B_eff 如何偏离 2。
  结论方向: 这条干净的【谱】形式是线性高斯类的专属性质; 对非线性器件, 需要的是
  真实【熵产生】版本的 TUR (见论文 Discussion 与后续工作), 而非这个二阶谱代理。

  这是一个【适用域刻画】实验, 不是"把不等式推广到非线性"的证明。我们如实报告偏离。

复用 exp3 完全相同的物理参数, 使 beta=0 档可直接与 exp3 对照。
所有数字来自本脚本真实运行, 固定随机种子, 无编造。
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import jonswap_spectrum  # noqa: E402

QUICK = os.environ.get("PIDM_QUICK", "0") == "1"
OUTDIR = os.environ.get("PIDM_OUTDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

# --- 与 exp3 波浪收集器完全一致的物理参数 ---
Hs, Tp = 2.0, 8.0
OMEGA_P = 2 * np.pi / Tp
GAMMA_J = 3.3
M_MASS = 1.0
K_STIFF = M_MASS * OMEGA_P ** 2     # 线性刚度, 固有频率=谱峰
C_MECH = 0.05
C_LOAD = 0.118                      # 与 exp3 阻抗匹配最优点一致
G_FORCE = 1.0


def synth_force(omega, S, n, dwsynth, rng):
    """用 irfft 随机相位法合成一条平稳高斯【力】时间序列 (与 exp3 速度合成同一约定)。
    amp_k=(n/2)*sqrt(2 S(omega_k) dw), amp[0]=0; 使 Var(eta)≈∫S dω。"""
    amp = (n / 2.0) * np.sqrt(2.0 * S * dwsynth)
    amp[0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, size=amp.shape)
    spec = amp * np.exp(1j * phase)
    return np.fft.irfft(spec, n)


def integrate_batch(eta, dt, k_lin, beta, c_total):
    """半隐式 Euler (Euler-Cromer) 矢量化积分一个 batch 的非线性收集器:
        v' = (g*eta - c_total*v - k_lin*x - beta*x^3)/m,  x' = v
    eta: (B, n) 力序列; 返回 v: (B, n) 速度序列。对振子+阻尼在小 dt 下稳定。"""
    B, n = eta.shape
    x = np.zeros(B)
    v = np.zeros(B)
    vout = np.empty((B, n), dtype=np.float64)
    for j in range(n):
        a = (G_FORCE * eta[:, j] - c_total * v - k_lin * x - beta * x ** 3) / M_MASS
        v = v + dt * a
        x = x + dt * v
        vout[:, j] = v
    return vout


def windowed_eps2(v_ss, dt, t_list):
    """对稳态速度 v_ss (B, m) 计算不同观测时长 t 下收集功 W_t=c_load∫v^2 的 eps^2。
    每条实现切成非重叠窗口, 汇总所有窗口求 Var(W)/mean(W)^2。"""
    B, m = v_ss.shape
    out = {}
    for t in t_list:
        L = int(round(t / dt))
        if L < 4 or L > m:
            continue
        nwin = m // L
        if nwin < 1:
            continue
        # 每条实现的前 nwin*L 点切成 nwin 段, ∫v^2 dt ≈ sum(v^2)*dt
        seg = v_ss[:, :nwin * L].reshape(B, nwin, L)
        W = C_LOAD * (seg ** 2).sum(axis=2) * dt     # W_t=c_load∫v^2 dt (对 v 平方!)
        W = W.reshape(-1)                            # 汇总所有窗口
        out[t] = (W.mean(), W.var(ddof=1), W.size)
    return out


def empirical_tau_c(v_ss, dt, tau_max):
    """由稳态速度的经验自相关 (Wiener-Khinchin: 周期图的 IFFT) 估计
    R(tau), 再 tau_c=2∫_0^{tau_max} R^2 dtau / R(0)^2 (单边×2=全轴)。"""
    B, m = v_ss.shape
    # 去均值
    v0 = v_ss - v_ss.mean(axis=1, keepdims=True)
    nfft = 1
    while nfft < 2 * m:
        nfft *= 2
    F = np.fft.rfft(v0, n=nfft, axis=1)
    psd = (np.abs(F) ** 2)
    ac = np.fft.irfft(psd, n=nfft, axis=1)[:, :m]    # 无偏前的原始自相关
    lags = np.arange(m)
    norm = (m - lags)                                # 无偏归一
    ac = ac / norm[None, :]
    R = ac.mean(axis=0)                              # 系综平均
    R0 = R[0]
    kmax = int(round(tau_max / dt))
    kmax = min(kmax, m - 1)
    tau = np.arange(kmax + 1) * dt
    tau_c = 2.0 * np.trapezoid(R[:kmax + 1] ** 2, tau) / (R0 ** 2)
    return tau_c, R0


def run_case(label, k_lin, beta, T, dt, B, n_batches, t_list, transient, tau_max, seed0):
    """跑一种器件设置 (线性/硬化/双稳), 返回该设置的统计。"""
    n = int(round(T / dt))
    # irfft 合成所用频率网格: omega_k = 2*pi*k/T
    freqs = np.fft.rfftfreq(n, d=dt)                 # Hz
    omega = 2 * np.pi * freqs                        # rad/s
    dwsynth = omega[1] - omega[0]
    S = jonswap_spectrum(np.maximum(omega, 1e-6), Hs, OMEGA_P, GAMMA_J)

    n_trans = int(round(transient / dt))
    # 用 Welford 跨 batch 汇总每个 t 的 W 一阶/二阶矩 + 速度自相关(末 batch 估计即可)
    agg = {t: [0.0, 0.0, 0] for t in t_list}         # [sumW, sumW2, count]
    var_eta_check = None
    tau_c = R0 = None
    for b in range(n_batches):
        rng = np.random.default_rng(seed0 + 17 * b)
        eta = np.empty((B, n))
        for i in range(B):
            eta[i] = synth_force(omega, S, n, dwsynth, rng)
        if var_eta_check is None:
            var_eta_check = float(eta.var())
        c_total = C_MECH + C_LOAD
        v = integrate_batch(eta, dt, k_lin, beta, c_total)
        v_ss = v[:, n_trans:]
        # 累积窗口统计
        wstats = windowed_eps2(v_ss, dt, t_list)
        for t, (mean, var, cnt) in wstats.items():
            # 用 mean,var,cnt 还原 sum, sum2 以便跨 batch 合并
            s = mean * cnt
            s2 = var * (cnt - 1) + s * mean           # sum2 = var*(n-1)+sum*mean
            agg[t][0] += s
            agg[t][1] += s2
            agg[t][2] += cnt
        # 自相关只用最后一个 batch 估计 (足够; 节省内存)
        if b == n_batches - 1:
            tau_c, R0 = empirical_tau_c(v_ss, dt, tau_max)

    B_eff = 1.0 / tau_c
    rows = []
    for t in t_list:
        s, s2, cnt = agg[t]
        if cnt < 2:
            continue
        mean = s / cnt
        var = (s2 - s * mean) / (cnt - 1)
        eps2 = var / mean ** 2
        rows.append({
            "t": t, "eps2": eps2, "eps2_t": eps2 * t,
            "eps2_t_Beff": eps2 * t * B_eff, "n_windows": cnt,
        })
    return {
        "label": label, "beta": beta, "k_lin": k_lin,
        "T": T, "dt": dt, "B_total": B * n_batches,
        "sigma_v2_emp": R0, "tau_c_emp": tau_c, "B_eff": B_eff,
        "var_eta_check": var_eta_check, "rows": rows,
    }


def main():
    t0 = time.time()
    if QUICK:
        T, dt, B, n_batches = 900.0, 0.08, 40, 1
        t_list = [40, 80]
        transient, tau_max = 150.0, 80.0
    else:
        T, dt, B, n_batches = 2200.0, 0.06, 200, 1
        t_list = [40, 80, 120, 160]
        transient, tau_max = 250.0, 140.0

    print("=" * 78)
    print("实验四 — 谱 TUR 关系 eps^2*t*B_eff->2 的【适用域】检验 (线性 vs 非线性)")
    print("=" * 78)
    print(f"模式={'QUICK' if QUICK else 'FULL'}  T={T}s dt={dt}s B_total={B*n_batches}  "
          f"窗口 t={t_list}s  瞬态弃={transient}s")
    print("提示: beta=0 应复现 exp3 波浪结果 (tau_c≈15.2, eps^2*t*B_eff->2), 作为基线与正确性校验。")

    cases = [
        ("线性 (beta=0, 高斯基线)",            K_STIFF,  0.0),
        ("硬化 Duffing (beta=0.05)",          K_STIFF,  0.05),
        ("强硬化 Duffing (beta=0.30)",        K_STIFF,  0.30),
        ("双稳 (k<0, beta=0.30)",            -K_STIFF,  0.30),
    ]
    results = []
    for i, (label, k_lin, beta) in enumerate(cases):
        r = run_case(label, k_lin, beta, T, dt, B, n_batches, t_list,
                     transient, tau_max, seed0=1000 + 100 * i)
        results.append(r)
        print(f"\n[{label}]  Var(eta)合成校验={r['var_eta_check']:.4f} (目标≈0.25)")
        print(f"  经验 sigma_v^2={r['sigma_v2_emp']:.4f}  tau_c={r['tau_c_emp']:.3f}s  "
              f"B_eff={r['B_eff']:.4f} rad/s")
        print(f"     t(s)    eps^2*t     eps^2*t*B_eff   窗口数")
        for row in r["rows"]:
            print(f"   {row['t']:6.0f}  {row['eps2_t']:10.4f}    "
                  f"{row['eps2_t_Beff']:10.4f}     {row['n_windows']}")

    # 汇总: 取最大窗口处的 eps^2*t*B_eff 作为"渐近代理", 比较各 case 偏离 2 的程度
    print("\n" + "-" * 78)
    print("汇总 (最大窗口处 eps^2*t*B_eff, 距离 2 的相对偏离):")
    for r in results:
        if r["rows"]:
            last = r["rows"][-1]["eps2_t_Beff"]
            dev = (last - 2.0) / 2.0 * 100
            print(f"  {r['label']:32s} -> {last:6.3f}  ({dev:+.1f}% vs 2)")

    runtime = round(time.time() - t0, 2)
    summary = {
        "experiment": "exp4_nonlinear_scope",
        "description": "Validity-domain test of the spectral relation eps^2*t*B_eff->2 "
                       "for linear (Gaussian, saturating) vs nonlinear (non-Gaussian) harvesters. "
                       "Honest scope delineation, not a generality proof.",
        "quick": QUICK, "runtime_sec": runtime,
        "shared_params": {"Hs": Hs, "Tp": Tp, "c_mech": C_MECH, "c_load": C_LOAD,
                          "g": G_FORCE, "k_linear": K_STIFF, "jonswap_gamma": GAMMA_J},
        "cases": results,
    }
    os.makedirs(OUTDIR, exist_ok=True)
    js = os.path.join(OUTDIR, "exp4_metrics.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[exp4] 指标: {js}")
    print(f"[exp4] 用时 {runtime} s")
    return summary


if __name__ == "__main__":
    main()
