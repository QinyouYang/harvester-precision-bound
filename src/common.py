#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py — 共享工具模块
========================

本模块集中放置两个实验都要用到的"可复用、可追溯"的物理工具：

  1. 真实标准海浪谱 (Pierson-Moskowitz 与 JONSWAP)
     —— 这是海洋/波浪能工程中通用的、有文献出处的标准谱形, 是本项目"真实数据"的来源。
     文献: W. J. Pierson & L. Moskowitz, J. Geophys. Res. 69, 5181 (1964);
           K. Hasselmann et al. (JONSWAP), Dtsch. Hydrogr. Z. (1973);
           工程标准 DNV-RP-C205 "Environmental conditions and environmental loads".

  2. 由谱合成平稳高斯随机波面 (随机相位法 / spectral synthesis)
     —— 把一条谱 S(omega) 变成一段可输入到收集器的时域强迫信号。

  3. 过阻尼 / 欠阻尼 Langevin 方程的 Euler-Maruyama 积分器。

设计原则: 所有物理量都用约化单位或明确标注单位; 所有谱形都可追溯到上述文献;
          不出现任何"凭空给定"的拟合常数 —— PM 谱的常数即文献原值。
"""

import numpy as np

# ===========================================================================
# 1. 真实标准海浪谱
# ===========================================================================
# 说明: 我们采用以"有效波高 Hs 与峰值频率 omega_p"参数化的现代标准形式
#       (等价于经典 Pierson-Moskowitz, 见 DNV-RP-C205)。这样做的好处是
#       谱的两个最直观的物理量 (浪有多高、主频在哪) 被显式地放进参数,
#       便于在冒烟测试里做"物理校验"(合成出的波面其 Hs 是否等于设定值)。


def pierson_moskowitz_spectrum(omega, Hs, omega_p):
    """Pierson-Moskowitz 谱 S(omega) (单位: 波面方差谱密度)。

    采用以 Hs (有效波高) 与 omega_p (峰值角频率) 表示的标准形式:
        S(omega) = (5/16) * Hs^2 * omega_p^4 * omega^-5 * exp(-5/4 * (omega_p/omega)^4)
    该形式是充分发展风浪的经典模型 (Pierson & Moskowitz 1964; DNV-RP-C205)。

    参数
    ----
    omega   : ndarray, 角频率数组 (rad/s), 必须 > 0
    Hs      : float, 有效波高 (m)
    omega_p : float, 峰值角频率 (rad/s)

    返回
    ----
    S : ndarray, 谱密度 (m^2 * s / rad), 与 omega 同形状
    """
    omega = np.asarray(omega, dtype=float)
    S = np.zeros_like(omega)
    pos = omega > 0
    w = omega[pos]
    S[pos] = (5.0 / 16.0) * Hs**2 * omega_p**4 * w**(-5) * np.exp(-1.25 * (omega_p / w)**4)
    return S


def jonswap_spectrum(omega, Hs, omega_p, gamma=3.3):
    """JONSWAP 谱 S(omega) —— 在 PM 谱基础上乘峰增强因子, 描述成长中的风浪。

        S_J(omega) = S_PM(omega) * gamma^{ r(omega) } * C(gamma)
        r(omega)   = exp( -(omega-omega_p)^2 / (2 sigma^2 omega_p^2) )
        sigma      = 0.07 (omega <= omega_p),  0.09 (omega > omega_p)
        C(gamma)   = 归一化因子, 使总能量 (m0) 仍对应给定 Hs

    文献: Hasselmann et al. (JONSWAP) 1973; DNV-RP-C205。
    gamma=1 时退化为 PM 谱; 北海典型取 gamma≈3.3。

    返回
    ----
    S : ndarray, 谱密度 (m^2 * s / rad)
    """
    omega = np.asarray(omega, dtype=float)
    S_pm = pierson_moskowitz_spectrum(omega, Hs, omega_p)
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.exp(-((omega - omega_p) ** 2) / (2.0 * sigma**2 * omega_p**2))
    S = S_pm * gamma**r
    # 重新归一化, 使带峰增强后的总能量 m0 仍精确对应设定的 Hs (Hs = 4*sqrt(m0))
    m0_target = (Hs / 4.0) ** 2
    domega = np.gradient(omega)
    m0_now = np.sum(S * domega)
    if m0_now > 0:
        S = S * (m0_target / m0_now)
    return S


def significant_wave_height(omega, S):
    """由谱反算有效波高 Hs = 4*sqrt(m0), m0 = ∫ S(omega) domega (零阶谱矩)。
    用于冒烟测试中的物理校验: 合成波面/谱是否真的对应设定的 Hs。"""
    omega = np.asarray(omega, dtype=float)
    S = np.asarray(S, dtype=float)
    domega = np.gradient(omega)
    m0 = np.sum(S * domega)
    return 4.0 * np.sqrt(max(m0, 0.0))


# ===========================================================================
# 2. 由谱合成平稳高斯随机波面 (随机相位法)
# ===========================================================================

def synthesize_wave(omega, S, t, rng):
    """随机相位法合成一段平稳高斯随机波面 eta(t):

        eta(t) = sum_i sqrt(2 S(omega_i) domega_i) * cos(omega_i t + phi_i),
        phi_i ~ U[0, 2pi) 独立同分布。

    其方差在统计意义上等于 ∫S domega = m0, 因此 Hs = 4 sqrt(m0)。
    这是海洋工程里把谱变成时域波信号的标准做法。

    参数
    ----
    omega : ndarray, 角频率离散网格 (rad/s)
    S     : ndarray, 对应谱密度
    t     : ndarray, 时间网格 (s)
    rng   : numpy Generator, 随机数发生器 (固定 seed 以复现)

    返回
    ----
    eta : ndarray, 波面时间序列 (m), 与 t 同形状
    """
    omega = np.asarray(omega, dtype=float)
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    domega = np.gradient(omega)
    amp = np.sqrt(2.0 * np.maximum(S, 0.0) * domega)      # 各分量幅值
    phi = rng.uniform(0.0, 2.0 * np.pi, size=omega.shape)  # 各分量随机相位
    # eta(t) = sum_i amp_i cos(omega_i t + phi_i)
    # 为避免一次性构造 len(t) x len(omega) 的巨大矩阵 (全量运行时可达数百 MB),
    # 按频率分块累加, 使峰值内存有界 (CHUNK 个频率一批)。
    eta = np.zeros_like(t)
    CHUNK = 64
    for s in range(0, len(omega), CHUNK):
        w = omega[s:s + CHUNK]
        a = amp[s:s + CHUNK]
        p = phi[s:s + CHUNK]
        phase = np.outer(t, w) + p[None, :]               # (len(t), CHUNK), 小块
        eta += (a[None, :] * np.cos(phase)).sum(axis=1)
    return eta


# ===========================================================================
# 3. Langevin 积分器 (Euler-Maruyama)
# ===========================================================================

def integrate_overdamped_tilted(f_drive, f_load, V0, L, D, gamma,
                                 dt, n_steps, n_traj, rng):
    """过阻尼布朗粒子在倾斜周期势中的 Euler-Maruyama 积分 (实验1用)。

        gamma dx = (f_drive - f_load - dV_p/dx) dt + sqrt(2 gamma kB T) dW
        V_p(x) = V0 cos(2 pi x / L),  -dV_p/dx = (2 pi V0 / L) sin(2 pi x / L)
    其中 D = kB T / gamma。返回每条轨迹在末时刻的净位移数组 X。
    """
    x = np.zeros(n_traj)
    sqrt_term = np.sqrt(2.0 * D * dt)
    net_const = (f_drive - f_load) / gamma
    k = 2.0 * np.pi / L
    for _ in range(n_steps):
        periodic = (V0 * k) * np.sin(k * x)      # = -dV_p/dx
        drift = net_const + periodic / gamma
        x = x + drift * dt + sqrt_term * rng.standard_normal(n_traj)
    return x


def integrate_underdamped_wave(eta_t, t, m, c_total, k_stiff, force_gain, x0=0.0, v0=0.0):
    """欠阻尼电机械振子在波浪强迫下的积分 (实验2用, 单条确定性轨迹给定 eta)。

        m x'' + c_total x' + k_stiff x = force_gain * eta(t)
    用半隐式 (symplectic-ish) Euler 逐步积分。eta_t 是预先合成好的波面在 t 上的取值。

    参数
    ----
    eta_t      : ndarray, 波面时间序列 (与 t 同长)
    t          : ndarray, 等间距时间网格
    m          : float, 等效质量
    c_total    : float, 总阻尼 = c_mech + c_load (机械阻尼 + 负载/PTO 阻尼)
    k_stiff    : float, 等效刚度 (含静水回复)
    force_gain : float, 波面到力的线性增益 (小体积点吸收子近似)

    返回
    ----
    x, v : ndarray, 位移与速度时间序列
    """
    n = len(t)
    dt = float(t[1] - t[0])
    x = np.empty(n); v = np.empty(n)
    x[0] = x0; v[0] = v0
    for i in range(n - 1):
        F = force_gain * eta_t[i] - c_total * v[i] - k_stiff * x[i]
        a = F / m
        v[i + 1] = v[i] + a * dt          # 先更新速度
        x[i + 1] = x[i] + v[i + 1] * dt    # 再用新速度更新位移 (半隐式, 更稳定)
    return x, v
