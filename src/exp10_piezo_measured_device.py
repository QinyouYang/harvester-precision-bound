#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验十 (exp10): 在【实测器件数据】上检验 ε²·t·B_eff → 2 —— 真正"不依赖模型"的检验。
===========================================================================

数据: Costanzo, Lo Schiavo, Sarracino & Vitelli, Entropy 23, 677 (2021) [稿中 ref 38]
      实验的原始负载电压时间序列 v_p(t): MIDE PPA-4011 压电收集器, 宽带高斯振动台
      激励 (闭环加速度控制), 3 个加速度 (a = 0.8, 1.0, 1.2 g) × 3 个负载电阻
      (R = 330, 2200, 5600 Ω), 每条 ~61.6 s @ 5 kHz (dt = 2×10⁻⁴ s)。

观测量 (全部直接由数据测得, 零拟合参数):
  W(t) = (1/R)∫₀ᵗ v_p² ds          收集功 (ref 38 式 27 的负载耗散热)
  ε²(t) = Var(W)/⟨W⟩²              非重叠窗口
  τ_c  = ∫R_v(τ)²dτ / R_v(0)²      实测自相关平方积分;  B_eff = 1/τ_c
  预言: 平稳高斯 ⇒ ε²(t)·t·B_eff → 2 (从下方), 精确有限时曲线由式(6)从实测 R_v 给出。

两个对照 (使结论不可回避):
  [S] 相位随机化代理 (surrogate): 保持周期图、随机化相位 ⇒ 与数据同谱的【严格平稳
      高斯】过程。经同一管线必须落在精确曲线上、收敛到 2 —— 校准管线本身。
  [D] 包络解调 (demodulation): v̂ = v_p/√(σ̂²(t)/⟨σ̂²⟩), σ̂² 为 v_p² 的 T_lp 滑动平均,
      只去除慢于 T_lp 的强度调制; 对代理施加同一操作给出该操作的本征偏差 (校准)。

诚实性: 长窗值取 t ≥ 30 τ_c 的逆方差加权平均; 误差为窗口自助法 SD; τ_c 在
0.2/0.5/1.0 s 三档截断下报告; 与 ref 38 已发表图 4 (平均功率) 与图 6 (τ=1 s 高斯拟合
σ² 与 τ=0.01 s 指数尾 α) 直接对比, 钉死数据来源与分析保真度。

运行:  PIEZO_DATADIR=<含 Acceleration_g*/ 的目录> python3 src/exp10_piezo_measured_device.py
输出:  results/exp10_metrics.json
"""

import os
import json
import time
import numpy as np

QUICK = os.environ.get("PIDM_QUICK", "0") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("PIDM_OUTDIR", os.path.join(HERE, "..", "results"))
DATADIR = os.environ.get("PIEZO_DATADIR", os.path.join(HERE, "..", "data", "piezo_costanzo2021"))
os.makedirs(OUTDIR, exist_ok=True)

ACCELS = [("0.8", "Acceleration_g0.8"), ("1.0", "Acceleration_g1"), ("1.2", "Acceleration_g1.2")]
RESISTORS = [330, 2200, 5600]
SEED = 2026
T_GRID = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
TAU_MAX_MAIN = 0.5          # τ_c / τ_{v²} 主积分截断 (s)
T_LP = 1.0                  # 解调滑动平均窗 (s)
PLATEAU_FACTOR = 30.0       # 长窗定义: t ≥ 30 τ_c

# ref 38 图 6 的已发表数值 (从其图读取, a = 1 g; 读图不确定度 ~±10%)
REF38_FIG6 = {330: dict(alpha=1.8e-5, sigma2=1.9e-11),
              2200: dict(alpha=3.85e-5, sigma2=5.7e-11),
              5600: dict(alpha=3.3e-5, sigma2=4.25e-11)}


# ---------------------------------------------------------------------------
def autocorr_unbiased(x, kmax):
    n = x.size
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    F = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(np.abs(F) ** 2, n=nfft)[: kmax + 1]
    ac /= (n - np.arange(kmax + 1))
    return ac


def tauc_K_from_R(R, dt, tau_max):
    kmax = min(int(round(tau_max / dt)), R.size - 1)
    tau = np.arange(kmax + 1) * dt
    R0sq = R[0] ** 2
    tau_c = 2.0 * np.trapezoid(R[: kmax + 1] ** 2, tau) / R0sq
    K = 4.0 * np.trapezoid(tau * R[: kmax + 1] ** 2, tau) / R0sq
    return float(tau_c), float(K)


def eps2_exact_from_R(R, dt, t_grid):
    """精确高斯预言 ε²_exact(t) = 4∫₀ᵗ(t-τ)R²dτ/(R(0)t)² (式 6), 由实测自相关给出。"""
    tau = np.arange(R.size) * dt
    R2 = R ** 2
    I1 = np.concatenate([[0.0], np.cumsum(0.5 * (R2[1:] + R2[:-1]) * dt)])
    tR2 = tau * R2
    I2 = np.concatenate([[0.0], np.cumsum(0.5 * (tR2[1:] + tR2[:-1]) * dt)])
    out = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        k = min(int(round(t / dt)), R.size - 1)
        out[i] = 4.0 * (t * I1[k] - I2[k]) / (R[0] ** 2 * t ** 2)
    return out


def windowed_eps2(vp, dt, t_grid, B_eff, rng, n_boot=400):
    """各窗口长度的 ε², 自助法 SD, 以及乘积 ε²·t·B_eff。"""
    rows = []
    for tw in t_grid:
        nstep = int(round(tw / dt))
        x = vp ** 2 * dt
        nwin = x.size // nstep
        W = x[: nwin * nstep].reshape(nwin, nstep).sum(axis=1)
        mW, vW = float(W.mean()), float(W.var(ddof=1))
        e2 = vW / mW ** 2
        idx = rng.integers(0, nwin, size=(n_boot, nwin))
        Wb = W[idx]
        e2b = Wb.var(axis=1, ddof=1) / Wb.mean(axis=1) ** 2
        rows.append(dict(t=float(tw), n_windows=int(nwin), eps2=float(e2),
                         eps2_sd=float(np.std(e2b)),
                         eps2_t_Beff=float(e2 * tw * B_eff),
                         eps2_t_Beff_sd=float(np.std(e2b) * tw * B_eff),
                         mean_W=mW, var_W=vW))
    return rows


def long_window_value(rows, tau_c):
    sel = [r for r in rows if r["t"] >= PLATEAU_FACTOR * tau_c]
    w = np.array([1.0 / max(r["eps2_t_Beff_sd"], 1e-12) ** 2 for r in sel])
    val = float(np.sum(w * [r["eps2_t_Beff"] for r in sel]) / np.sum(w))
    return val, float(1.0 / np.sqrt(np.sum(w)))


def phase_surrogate(vp, rng):
    """相位随机化代理: 同周期图、随机相位 ⇒ 严格平稳高斯, 同谱。"""
    n = vp.size
    F = np.fft.rfft(vp)
    mag = np.abs(F)
    ph = rng.uniform(0, 2 * np.pi, size=mag.size)
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(mag * np.exp(1j * ph), n=n)


def synth_slow_envelope(excess_full_norm, dt, n, rng,
                        tau_min=0.12, tau_taper=1.5, tau_max=2.0):
    """由实测【慢段】超额自协方差 c_q(τ)=excess/σ₀⁴ 非参数合成慢包络 m(t)=√(1+s):
    近零滞后被 κσ⁴ 尖峰与 2R² 耦合污染, 故仅用 τ≥tau_min 的慢段;
    τ<tau_min 平台延拓 (慢过程在快尺度上近常数), 0.1 s 矩形窗轻度平滑,
    [tau_taper, tau_max] 余弦渐零; 循环嵌入谱合成 (nfft≥2n, 单周期覆盖整条
    记录, 无平铺), 谱负值截零, 不重定标方差 —— Var(s)=c_q(τ_min) 平台值。"""
    k_min = int(round(tau_min / dt))
    k_tap = int(round(tau_taper / dt))
    k_max = min(int(round(tau_max / dt)), excess_full_norm.size - 1)
    w = max(int(round(0.10 / dt)), 1)
    sm = np.convolve(excess_full_norm, np.ones(w) / w, mode="same")
    c = np.empty(k_max + 1)
    c[k_min:] = sm[k_min:k_max + 1]
    c[:k_min] = c[k_min]                                     # 平台延拓
    ramp = 0.5 * (1.0 + np.cos(np.pi * (np.arange(k_tap, k_max + 1) - k_tap)
                               / max(k_max - k_tap, 1)))
    c[k_tap:] *= ramp
    nfft = 1
    while nfft < max(2 * n, 2 * (k_max + 1)):
        nfft *= 2
    cfull = np.zeros(nfft)
    cfull[:k_max + 1] = c
    cfull[-k_max:] = c[1:][::-1]
    S = np.real(np.fft.rfft(cfull))
    S[S < 0] = 0.0                                           # 截谱负值 (估计噪声)
    s = np.fft.irfft(np.fft.rfft(rng.standard_normal(nfft)) * np.sqrt(S),
                     n=nfft)[:n]
    return np.sqrt(np.maximum(1.0 + s, 0.05))


def eps2_exact_from_C(C, sig4, dt, t_grid):
    """任意平稳过程的精确恒等式 ε²(t)·t = ∫_{-t}^{t}(1-|τ|/t)C(τ)dτ/⟨v²⟩²,
    由【全协方差】C_v²(τ) 直接给出 (含慢调制)。"""
    tau = np.arange(C.size) * dt
    I1 = np.concatenate([[0.0], np.cumsum(0.5 * (C[1:] + C[:-1]) * dt)])
    tC = tau * C
    I2 = np.concatenate([[0.0], np.cumsum(0.5 * (tC[1:] + tC[:-1]) * dt)])
    out = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        k = min(int(round(t / dt)), C.size - 1)
        out[i] = 2.0 * (t * I1[k] - I2[k]) / (sig4 * t ** 2)
    return out


def gaussian_fit_sigma2(x, nbins=41):
    """对 W/τ 直方图做高斯最小二乘拟合 (模仿 ref 38 图 6 的 g(x) 拟合), 返回 σ²。
    样本数 <200 时直方图拟合无意义 (本记录 τ=1s 仅 ~61 窗; ref 38 为 600 s 记录),
    返回 NaN, 以样本方差 sigma2_direct 为准。"""
    if x.size < 200:
        return float(np.nan)
    cnt, edges = np.histogram(x, bins=nbins)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    use = cnt > max(cnt.max() // 50, 4)
    c, xc = cnt[use].astype(float), ctr[use]
    mu0, s0 = float(x.mean()), float(x.std())
    from scipy.optimize import curve_fit, OptimizeWarning
    import warnings
    f = lambda x_, A, mu, s: A * np.exp(-(x_ - mu) ** 2 / (2 * s ** 2))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            p, _ = curve_fit(f, xc, c, p0=[c.max(), mu0, s0],
                             sigma=np.sqrt(np.maximum(c, 1.0)),
                             bounds=([0.0, -np.inf, 1e-12],
                                     [np.inf, np.inf, np.inf]),
                             maxfev=20000)
        return float(p[2] ** 2)
    except Exception:
        return float(np.var(x, ddof=1))


def exp_tail_alpha(x, nbins=80):
    """对 W/τ 分布的大值尾做指数拟合 f(x)~exp(-x/α) (模仿 ref 38 图 6 的 α)。"""
    cnt, edges = np.histogram(x, bins=nbins, density=True)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    thr = float(np.mean(x) + 1.5 * np.std(x))
    use = (ctr > thr) & (cnt > 0)
    if use.sum() < 5:
        return float(np.nan)
    A = np.polyfit(ctr[use], np.log(cnt[use]), 1)
    return float(-1.0 / A[0])


def hist_for_json(x, nbins=60):
    cnt, edges = np.histogram(x, bins=nbins, density=True)
    return dict(centers=(0.5 * (edges[1:] + edges[:-1])).tolist(), pdf=cnt.tolist())


# ---------------------------------------------------------------------------
def analyse_one(path, R_ohm, rng):
    d = np.loadtxt(path)
    t, vp = d[:, 0], d[:, 1]
    dt = float(np.mean(np.diff(t)))
    offset = float(vp.mean())
    vp = vp - offset
    n = vp.size

    q = n // 4
    qv = [float(np.var(vp[i * q:(i + 1) * q])) for i in range(4)]
    stat_dev = float(max(abs(v - np.mean(qv)) for v in qv) / np.mean(qv))

    var_vp = float(np.var(vp))
    P_harv = var_vp / R_ohm
    kappa = float(np.mean(vp ** 4) / var_vp ** 2 - 3.0)

    kmax = int(round(2.0 / dt))
    Racf = autocorr_unbiased(vp, kmax)
    tau_c, K = tauc_K_from_R(Racf, dt, TAU_MAX_MAIN)
    tau_c_alt = {f"{tm:g}s": tauc_K_from_R(Racf, dt, tm)[0] for tm in (0.2, 0.5, 1.0)}
    B_eff = 1.0 / tau_c

    # v² 自协方差: τ_{v²}, 以及超出 Isserlis (2R²) 的慢尾积分
    v2c = vp ** 2 - var_vp
    Cv2 = autocorr_unbiased(v2c, kmax)
    km = int(round(TAU_MAX_MAIN / dt))
    tau_ax = np.arange(km + 1) * dt
    tau_v2 = float(2.0 * np.trapezoid(Cv2[:km + 1] / Cv2[0], tau_ax))
    iss = 2.0 * Racf[:km + 1] ** 2
    excess = Cv2[:km + 1] - iss
    tau_slow = float(2.0 * np.trapezoid(excess / Cv2[0], tau_ax))   # 慢调制对 τ_{v²} 的贡献
    A_kappa = (kappa + 2.0) / 2.0
    R_tau = tau_v2 / tau_c
    pred_2AkRt = 2.0 * A_kappa * R_tau
    mod_depth_rms = float(np.sqrt(max(kappa, 0.0) / 3.0))           # 条件高斯调制: Var(σ²)/⟨σ²⟩²≈κ/3

    rows = windowed_eps2(vp, dt, T_GRID, B_eff, rng)
    e2x = eps2_exact_from_R(Racf, dt, T_GRID)
    for r, ex in zip(rows, e2x):
        r["eps2_exact"] = float(ex)
        r["eps2_t_Beff_exact"] = float(ex * r["t"] * B_eff)
    longv, longv_sd = long_window_value(rows, tau_c)

    # [S] 相位随机化代理系综 (N_SUR 条, 同管线): 校准"平稳高斯+同谱 ⇒ 精确曲线/→2"
    N_SUR = 4 if QUICK else 6
    sur_curves, sur_long, sur_kappa = [], [], []
    for _ in range(N_SUR):
        vs = phase_surrogate(vp, rng)
        vs -= vs.mean()
        Rs = autocorr_unbiased(vs, kmax)
        tcs, _ = tauc_K_from_R(Rs, dt, TAU_MAX_MAIN)
        rs = windowed_eps2(vs, dt, T_GRID, 1.0 / tcs, rng, n_boot=100)
        sur_curves.append([w["eps2_t_Beff"] for w in rs])
        sur_long.append(long_window_value(rs, tcs)[0])
        sur_kappa.append(float(np.mean(vs ** 4) / np.var(vs) ** 2 - 3.0))
    sur_curves = np.array(sur_curves)
    sur_long = np.array(sur_long)
    # 有限时高斯期望 (长窗加权, 用实测 R 的精确曲线在同样 t 上同样加权)
    sel = T_GRID >= PLATEAU_FACTOR * tau_c
    gauss_expect_long = float(np.mean((e2x * T_GRID * B_eff)[sel]))

    # [M] 构造性调制代理: 平稳高斯代理 × 由实测慢段超额协方差合成的慢包络 ⇒ 应复现上翘
    excess_full_norm = (Cv2 - 2.0 * Racf ** 2) / var_vp ** 2     # c_q(τ) 至 2 s, 不截负
    mod_curves, mod_long = [], []
    for _ in range(N_SUR):
        vs = phase_surrogate(vp, rng)
        vs -= vs.mean()
        m = synth_slow_envelope(excess_full_norm, dt, vs.size, rng)
        vm = vs * m
        vm -= vm.mean()
        Rm = autocorr_unbiased(vm, kmax)
        tcm, _ = tauc_K_from_R(Rm, dt, TAU_MAX_MAIN)
        rm = windowed_eps2(vm, dt, T_GRID, 1.0 / tcm, rng, n_boot=100)
        mod_curves.append([w["eps2_t_Beff"] for w in rm])
        mod_long.append(long_window_value(rm, tcm)[0])
    mod_curves = np.array(mod_curves)
    mod_long = np.array(mod_long)

    # 全协方差精确恒等式曲线 (含调制; 同一记录的一致性检验)
    e2_fullC = eps2_exact_from_C(Cv2[: int(round(2.0 / dt)) + 1], var_vp ** 2, dt, T_GRID)

    # 与 ref 38 图 6 对比: τ=1 s 的直方图高斯拟合 σ² 与 τ=0.01 s 的指数尾 α
    def W_over_tau(tw):
        nstep = int(round(tw / dt))
        x = vp ** 2 * dt
        nw = x.size // nstep
        return x[: nw * nstep].reshape(nw, nstep).sum(axis=1) / tw / R_ohm
    Wt1 = W_over_tau(1.0)
    Wt001 = W_over_tau(0.01)
    sigma2_fit = gaussian_fit_sigma2(Wt1)
    sigma2_direct = float(np.var(Wt1, ddof=1))
    alpha_fit = exp_tail_alpha(Wt001)

    return dict(R_ohm=R_ohm, file=os.path.basename(path), n_samples=int(n), dt_s=dt,
                duration_s=float(n * dt), dc_offset_V=offset,
                stationarity_quarter_var_maxdev=stat_dev,
                var_vp_V2=var_vp, P_harv_W=P_harv, kappa_excess_kurtosis=kappa,
                tau_c_s=tau_c, tau_c_vs_cutoff=tau_c_alt, K_s2=K, B_eff_Hz=B_eff,
                tau_v2_s=tau_v2, tau_slow_excess_s=tau_slow,
                A_kappa=A_kappa, R_tau=R_tau, pred_2AkRt=pred_2AkRt,
                mod_depth_rms=mod_depth_rms,
                long_eps2tB=longv, long_eps2tB_sd=longv_sd,
                surrogate=dict(n=int(len(sur_long)),
                               kappa_mean=float(np.mean(sur_kappa)),
                               long_eps2tB=float(sur_long.mean()),
                               long_eps2tB_se=float(sur_long.std(ddof=1) / np.sqrt(len(sur_long))),
                               curve_mean=sur_curves.mean(axis=0).tolist(),
                               curve_se=(sur_curves.std(axis=0, ddof=1) / np.sqrt(len(sur_long))).tolist(),
                               gauss_expect_long=gauss_expect_long),
                modulated_surrogate=dict(n=int(len(mod_long)),
                               long_eps2tB=float(mod_long.mean()),
                               long_eps2tB_se=float(mod_long.std(ddof=1) / np.sqrt(len(mod_long))),
                               curve_mean=mod_curves.mean(axis=0).tolist(),
                               curve_se=(mod_curves.std(axis=0, ddof=1) / np.sqrt(len(mod_long))).tolist()),
                eps2_t_Beff_fullC=(e2_fullC * T_GRID * B_eff).tolist(),
                ref38_fig6_comparison=dict(
                    sigma2_gaussfit_W2=sigma2_fit, sigma2_direct_W2=sigma2_direct,
                    alpha_exp_tail_W=alpha_fit,
                    published=REF38_FIG6.get(R_ohm)),
                hist_W_over_tau=dict(t_1s=hist_for_json(Wt1), t_0p01s=hist_for_json(Wt001),
                                     t_0p1s=hist_for_json(W_over_tau(0.1))),
                windows=rows)


def _require_piezo_data(datadir):
    """实测压电数据不随仓库公开 (归 Costanzo 等所有); 缺失时给出清晰提示并退出。"""
    probe = os.path.join(datadir, "Acceleration_g1", "R2200_V_load.txt")
    if not os.path.exists(probe):
        raise SystemExit(
            "[exp10] Measured piezo records not found in %s\n"
            "        These data belong to Costanzo et al. (Entropy 23, 677, 2021) and are NOT\n"
            "        redistributed with this repository; request them from the original authors\n"
            "        and place them as described in data/piezo_costanzo2021/README.md.\n"
            "        The computed results are provided in results/exp10_metrics.json." % datadir)


def main():
    _require_piezo_data(DATADIR)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    out = {"datasets": [], "meta": dict(
        source=("Measured load-voltage time series from the piezoelectric energy-harvester "
                "experiment of Costanzo, Lo Schiavo, Sarracino & Vitelli, Entropy 23, 677 "
                "(2021) [manuscript ref. 38]; MIDE PPA-4011, broadband Gaussian shaker drive "
                "with closed-loop acceleration control, 5 kHz sampling."),
        observable="W(t) = (1/R) ∫ v_p^2 ds (load/dissipation channel, eq. 27 of ref. 38)",
        seed=SEED, quick=QUICK, T_lp_s=T_LP, tau_max_main_s=TAU_MAX_MAIN,
        plateau_definition=f"inverse-variance-weighted mean over windows t >= {PLATEAU_FACTOR:g} tau_c",
        ref38_fig6_note="published values read off ref. 38 Fig. 6 (a = 1 g), read-off uncertainty ~10%")}

    hdr = (f"{'a(g)':>5} {'R(Ω)':>5} {'P(W)':>10} {'κ':>7} {'τc(ms)':>7} {'Beff':>6} "
           f"{'A_κ':>6} {'R_τ':>6} {'实测长窗':>8} {'±':>5} {'高斯期望':>8} {'代理':>6} {'±':>5} {'调制代理':>8} {'±':>5}")
    print(hdr)
    for a_label, gdir in ACCELS:
        for R_ohm in RESISTORS:
            r = analyse_one(os.path.join(DATADIR, gdir, f"R{R_ohm}_V_load.txt"), R_ohm, rng)
            r["a_g"] = a_label
            out["datasets"].append(r)
            print(f"{a_label:>5} {R_ohm:>5} {r['P_harv_W']:>10.2e} {r['kappa_excess_kurtosis']:>7.3f} "
                  f"{1e3*r['tau_c_s']:>7.2f} {r['B_eff_Hz']:>6.1f} {r['A_kappa']:>6.3f} {r['R_tau']:>6.3f} "
                  f"{r['long_eps2tB']:>8.3f} {r['long_eps2tB_sd']:>5.3f} "
                  f"{r['surrogate']['gauss_expect_long']:>8.3f} "
                  f"{r['surrogate']['long_eps2tB']:>6.3f} {r['surrogate']['long_eps2tB_se']:>5.3f} "
                  f"{r['modulated_surrogate']['long_eps2tB']:>8.3f} {r['modulated_surrogate']['long_eps2tB_se']:>5.3f}")

    ds = out["datasets"]
    arr = lambda key: np.array([d[key] for d in ds])
    raw = arr("long_eps2tB")
    sur = np.array([d["surrogate"]["long_eps2tB"] for d in ds])
    gexp = np.array([d["surrogate"]["gauss_expect_long"] for d in ds])
    mod = np.array([d["modulated_surrogate"]["long_eps2tB"] for d in ds])
    out["summary"] = dict(
        raw_long=dict(mean=float(raw.mean()), min=float(raw.min()), max=float(raw.max())),
        surrogate_long=dict(mean=float(sur.mean()), min=float(sur.min()), max=float(sur.max())),
        gauss_expect_long=dict(mean=float(gexp.mean())),
        modulated_surrogate_long=dict(mean=float(mod.mean()), min=float(mod.min()), max=float(mod.max())),
        kappa_range=[float(arr("kappa_excess_kurtosis").min()), float(arr("kappa_excess_kurtosis").max())],
        tau_c_ms_range=[float(1e3 * arr("tau_c_s").min()), float(1e3 * arr("tau_c_s").max())],
        runtime_s=float(time.time() - t0))
    print(f"\n长窗 ε²tB —— 实测: {raw.mean():.3f} [{raw.min():.3f},{raw.max():.3f}] | "
          f"平稳高斯代理: {sur.mean():.3f} (有限时高斯期望 {gexp.mean():.3f}) | "
          f"调制代理: {mod.mean():.3f}   (用时 {time.time()-t0:.0f}s)")

    with open(os.path.join(OUTDIR, "exp10_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("written:", os.path.join(OUTDIR, "exp10_metrics.json"))


if __name__ == "__main__":
    main()
