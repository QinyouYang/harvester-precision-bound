#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验八 (exp8): 用【真实 NDBC 浮标实测谱】驱动线性收集器 —— 消除"合成谱"批评点。
Experiment 8: drive the linear harvester with a REAL measured NDBC buoy spectrum.

谱精度界 ε²·t·B_eff -> 2 的推导只依赖"线性 + 高斯", 与谱形无关。本实验把驱动从合成的
JONSWAP 换成美国国家数据浮标中心 (NDBC) 的实测谱密度, 由它当 S_η 重算 τ_c / B_eff。

数据来源 (可复现, 离线默认):
    NDBC station 46042 (Monterey, CA), spectral wave density, record 2026-06-03 07:40 UTC,
    from the NDBC monthly archive file
        https://www.ndbc.noaa.gov/data/swden/Jun/4604262026.txt.gz
    stored unmodified in  data/ndbc_46042_swden_2026_06.txt.gz .
    默认读取该本地存档文件, 因此结果与网络无关、可逐位复现。

可选: 设置 PIDM_NDBC_LIVE=1 时抓取 NDBC realtime2 的"最新"谱做演示; 其结果随时间变化,
写入单独的 exp8_ndbc_live_spectrum.json, 不会覆盖存档结果 exp8_ndbc_real_spectrum.json。

关键数值细节: 实测谱只有 ~47 个频点 (间隔约 0.03–0.06 rad/s), 若直接用粗网格, 低阻尼共振
(宽度 ~c) 只被 1–2 个频点覆盖 -> 欠采样 -> τ_c 被灌水。因此先把实测谱线性插值到细网格再积分。

环境变量: PIDM_OUTDIR (默认 <project>/results), PIDM_DATADIR (默认 <project>/data),
          PIDM_NDBC_LIVE=1 (可选, 联网抓最新谱)。
"""
import os, re, json, sys, gzip
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("PIDM_OUTDIR") or os.path.join(HERE, "..", "results")
DATADIR = os.environ.get("PIDM_DATADIR") or os.path.join(HERE, "..", "data")
LIVE = os.environ.get("PIDM_NDBC_LIVE", "0") == "1"

ARCHIVE_FILE = "ndbc_46042_swden_2026_06.txt.gz"
ARCHIVE_URL = "https://www.ndbc.noaa.gov/data/swden/Jun/4604262026.txt.gz"
ARCHIVE_STATION = "46042"
ARCHIVE_RECORD = "2026 06 03 07 40"          # YYYY MM DD hh mm (UTC)

STATIONS = ["46042", "46026", "51201", "46022", "41008"]   # live 模式依次尝试
M_MASS, G_FORCE = 1.0, 1.0
C_TOTAL = 0.10 + 0.018                                       # c_mech + c_load (沿用项目量级)


def load_archived_spectrum(path=None, record=ARCHIVE_RECORD):
    """读取 NDBC 月度存档 swden 文件中指定时刻的一条谱: 返回 (freq_Hz, E_m2_per_Hz, station, url, ts)。"""
    path = path or os.path.join(DATADIR, ARCHIVE_FILE)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    header = [l for l in lines if l.startswith("#")][0]
    f = np.array([float(x) for x in header.split()[5:]])
    for l in lines:
        if l.startswith(record):
            E = np.array([float(x) for x in l.split()[5:]])
            if E.size != f.size:
                raise ValueError("frequency/energy column mismatch in archived record")
            return f, E, ARCHIVE_STATION, ARCHIVE_URL, record
    raise ValueError(f"record '{record}' not found in {path}")


def fetch_latest_spectrum():
    """(可选) 抓第一个可用浮标的最新一条 realtime 谱: 返回 (freq_Hz, E, station, url, ts)。"""
    import urllib.request
    hdr = {"User-Agent": "Mozilla/5.0 (research; harvester-spectral-bound)"}
    for st in STATIONS:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{st}.data_spec"
        try:
            req = urllib.request.Request(url, headers=hdr)
            txt = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
        except Exception as e:
            print(f"  [{st}] 取数失败: {e}", file=sys.stderr); continue
        lines = [l for l in txt.splitlines() if l.strip() and not l.startswith("#")]
        if not lines:
            continue
        latest = lines[0]
        pairs = re.findall(r"([0-9.eE+-]+)\s*\(\s*([0-9.eE+-]+)\s*\)", latest)
        if len(pairs) < 10:
            continue
        E = np.array([float(e) for e, f in pairs])
        f = np.array([float(f) for e, f in pairs])
        o = np.argsort(f)
        return f[o], E[o], st, url, " ".join(latest.split()[:5])
    return None


def analyse(f, E):
    """由实测谱 E(f) 计算谐振调谐到谱峰的线性收集器的 σ_v², τ_c, B_eff。"""
    Hs = 4 * np.sqrt(np.trapezoid(E, f))
    w_meas = 2 * np.pi * f
    S_eta_meas = E / (2 * np.pi)                 # m^2/Hz -> m^2/(rad/s) (单边)
    wp = w_meas[np.argmax(E)]                     # 谱峰角频率
    k = M_MASS * wp ** 2                          # 共振调谐到谱峰
    zeta = C_TOTAL / (2 * np.sqrt(M_MASS * k))
    # 实测谱插值到细网格, 解析共振
    w = np.linspace(w_meas.min(), w_meas.max(), 4000)
    S_eta = np.interp(w, w_meas, S_eta_meas)
    H2 = 1.0 / ((k - M_MASS * w ** 2) ** 2 + (C_TOTAL * w) ** 2)
    S_v = (w ** 2) * H2 * (G_FORCE ** 2) * S_eta
    # τ_c, B_eff; 带收敛检查
    tau = np.linspace(0, 300, 12000)
    R = np.array([np.trapezoid(S_v * np.cos(w * t), w) for t in tau])
    R0 = R[0]
    integ = (R / R0) ** 2
    half = len(tau) // 2
    tau_c = 2 * np.trapezoid(integ, tau)
    tau_c_150 = 2 * np.trapezoid(integ[:half], tau[:half])
    return dict(Hs=Hs, wp=wp, zeta=zeta, n_grid=len(w), R0=R0,
                tau_c=tau_c, tau_c_150=tau_c_150, B_eff=1.0 / tau_c)


def main():
    if LIVE:
        got = fetch_latest_spectrum()
        if got is None:
            print("ALL_FETCH_FAILED: 无法抓取任何 NDBC 浮标谱; 去掉 PIDM_NDBC_LIVE 以使用存档谱。")
            return None
        source = "NDBC realtime2 (latest record; changes over time)"
        out_name = "exp8_ndbc_live_spectrum.json"
    else:
        got = load_archived_spectrum()
        source = "NDBC monthly archive (fixed record; reproducible)"
        out_name = "exp8_ndbc_real_spectrum.json"
    f, E, st, url, ts = got
    a = analyse(f, E)

    print("=" * 78)
    print(f"实验八 — 真实 NDBC 浮标谱驱动线性收集器 (station {st}, {ts} UTC)")
    print(f"来源: {source}")
    print("=" * 78)
    print(f"实测谱: {len(f)} 频点, {f.min():.4f}-{f.max():.4f} Hz, Hs={a['Hs']:.3f} m")
    print(f"谱峰 ω_p={a['wp']:.3f} rad/s (Tp={2*np.pi/a['wp']:.1f} s), 阻尼比 ζ={a['zeta']:.3f}")
    print(f"细网格 {a['n_grid']} 点 (共振宽度 ~c={C_TOTAL:.3f} 已充分解析)")
    print(f"σ_v^2={a['R0']:.4f}")
    print(f"τ_c={a['tau_c']:.2f} s  (积分到150s={a['tau_c_150']:.2f} s, 接近=>收敛)")
    print(f"B_eff=1/τ_c={a['B_eff']:.4f} rad/s")

    out = {"experiment": "exp8_real_ndbc_spectrum", "source": source,
           "station": st, "url": url, "timestamp_utc": ts,
           "n_spectral_points_measured": int(len(f)), "n_grid_interpolated": int(a["n_grid"]),
           "Hs_from_spectrum_m": float(a["Hs"]), "omega_peak_rad_s": float(a["wp"]),
           "Tp_s": float(2 * np.pi / a["wp"]), "damping_ratio_zeta": float(a["zeta"]),
           "sigma_v2": float(a["R0"]), "tau_c_s": float(a["tau_c"]),
           "tau_c_s_check_150": float(a["tau_c_150"]), "B_eff_rad_s": float(a["B_eff"]),
           "note": "Spectral precision bound eps^2*t*B_eff->2 holds for any linear-Gaussian harvester; "
                   "driven here by a REAL NDBC measured spectrum. The measured spectrum is linearly "
                   "interpolated onto a fine grid so the harvester resonance (width ~c) is resolved "
                   "(otherwise the coarse grid under-resolves it and inflates tau_c)."}
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, out_name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[exp8] 指标: {p}")
    return out


if __name__ == "__main__":
    main()
