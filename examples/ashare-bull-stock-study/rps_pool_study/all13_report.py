"""§122 R01–R13 全表:总收益 / 年化 / 最大回撤 / 夏普,与 Codex 逐格对照。

本节是**指标补全与并排展示,不是假设检验**,不设通过/不通过判据 ——
§113/§114/§117/§118 已分别对各路线下过判定,本节不重判、不翻案。

两处相对 §117/§118 的实现修正(改的是我的 bug,不是判据):
1) **cash_fallback**。他的 `build_selections` 恒按 5%/只配权,合格不足 20 只时
   **只买合格的、其余留现金**;我原先写的是「不足 20 只就跳过该调仓日」。
   §121 已定位这是 R13 复现失败的真实原因。本节全部路线改为正确行为。
   影响最大的是 R04(合格日 55/153)与 R13(83/153),合格集大的路线几乎无影响。
2) **R12 补上**。§118 曾以「底座 R11 的 oos 已失效」为由不跑,那是研究价值判断;
   本节是全表展示,补齐。R12 = 510300 月线状态开(收盘>月线MA20 且 MACD>signal)
   时用 R11(含市值剔微盘),否则切「无市值均衡」(价值/质量/低风险 各 1/3)。
   **标注为近似**:他的状态定义细节(MACD 参数、月线取样口径)归档里不完整。

他的公布值直接取自各 evidence 的 `fast_screen_results.json`,不用 README 转述;
其中 R03 用 `r03_low_rps_reversal_20260821_01`(低位反转,他的代表策略),
不是同名的 `r03_high_rps_screen`;R08 用 `_02`,`_01` 是全零的失败运行。

锚点:面板 (3297,5217);TTM 恒等式;泰格同比复现。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, OUT, run_window_fast  # noqa: E402
from codex_r10_replication import DATA, TOP_N, WEIGHT, metrics, pct  # noqa: E402
from codex_routes_rerun import build_fund, route_scores  # noqa: E402
from codex_routes_rest import score as pv_score  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

WINS = {"train": ("2014-01-02", "2019-12-31"), "validation": ("2020-01-02", "2022-12-30"),
        "oos": ("2023-01-03", "2025-12-31"), "holdout": ("2026-01-05", "2026-08-03"),
        "full": ("2014-01-02", "2025-12-31")}
CODEX = {
 ("R01","train"):(.920243,.115013,-.347329,.6879),("R01","validation"):(-.13228,-.046308,-.203617,-.2625),
 ("R01","oos"):(.124156,.039884,-.138499,.4467),("R01","holdout"):(-.021178,-.037781,-.098973,-.1093),
 ("R01","full"):(.865768,.053372,-.350454,.4195),
 ("R02","train"):(.205716,.031707,-.487224,.2581),("R02","validation"):(.157629,.05013,-.17152,.385),
 ("R02","oos"):(.131629,.042189,-.130125,.4758),("R02","holdout"):(-.021178,-.037781,-.098973,-.1093),
 ("R02","full"):(.680913,.044249,-.507584,.3378),
 ("R03","train"):(.489374,.058671,-.639306,.3495),("R03","oos"):(1.356646,.331712,-.316382,1.3422),
 ("R03","full"):(9.025661,.194202,-.639306,.7763),
 ("R04","train"):(-.022276,-.003752,-.174303,-.0213),("R04","validation"):(.03588,.01185,-.138988,.1899),
 ("R04","oos"):(.078911,.025706,-.045555,.5564),("R04","holdout"):(-.006456,-.011587,-.021215,-.4115),
 ("R04","full"):(.080593,.006483,-.206114,.1315),
 ("R05","train"):(.433853,.061974,-.705634,.3962),("R05","validation"):(.360942,.108474,-.170892,.6502),
 ("R05","oos"):(.541858,.155684,-.162716,1.0166),("R05","holdout"):(-.159538,-.268544,-.235999,-1.9657),
 ("R05","full"):(1.845984,.091114,-.705634,.5546),
 ("R06","train"):(1.139986,.135354,-.425061,.7714),("R06","validation"):(.034566,.011421,-.114945,.159),
 ("R06","oos"):(.375349,.112381,-.111089,.9174),("R06","holdout"):(-.017925,-.032021,-.112355,-.2263),
 ("R06","full"):(1.826839,.0905,-.433322,.6241),
 ("R07","train"):(-.08624,-.014936,-.685805,.0927),("R07","validation"):(.659714,.184483,-.257723,.8501),
 ("R07","oos"):(.119577,.038466,-.31543,.2988),("R07","holdout"):(.04123,.075403,-.111737,.4974),
 ("R07","full"):(.60626,.040301,-.685805,.2886),
 ("R08","train"):(2.013466,.202086,-.375593,.9406),("R08","validation"):(.460082,.134829,-.171203,.8223),
 ("R08","oos"):(.68045,.189408,-.133465,1.198),("R08","holdout"):(.004133,.007449,-.102822,.1258),
 ("R08","full"):(5.631574,.170845,-.375593,.8958),
 ("R09","train"):(1.040267,.12635,-.532588,.5699),("R09","validation"):(.181913,.057441,-.204471,.4162),
 ("R09","oos"):(.447887,.131652,-.132107,.9215),("R09","holdout"):(.034971,.0638,-.071488,.5154),
 ("R09","full"):(2.224715,.102538,-.532588,.5352),
 ("R10","train"):(4.019202,.308898,-.415457,1.2008),("R10","validation"):(1.39593,.33909,-.244526,1.7394),
 ("R10","oos"):(1.677538,.389752,-.281309,1.5828),("R10","holdout"):(.041527,.075955,-.14307,.4536),
 ("R10","full"):(31.350555,.336229,-.415457,1.3509),
 ("R11","train"):(2.74828,.24666,-.403545,1.0364),("R11","validation"):(.412851,.122427,-.142936,.7793),
 ("R11","oos"):(.71399,.197289,-.128505,1.2592),("R11","holdout"):(-.076381,-.133212,-.159539,-.845),
 ("R11","full"):(7.553667,.195955,-.403545,.9651),
 ("R12","train"):(.831889,.106284,-.312715,.7642),("R12","validation"):(.221343,.069101,-.14243,.6402),
 ("R12","oos"):(.53761,.154619,-.083686,1.3391),("R12","holdout"):(-.076381,-.133212,-.159539,-.845),
 ("R12","full"):(2.162266,.100742,-.312715,.7798),
 ("R13","train"):(.418375,.060053,-.254639,.4289),("R13","validation"):(-.157737,-.05575,-.492707,-.2289),
 ("R13","oos"):(-.076656,-.026299,-.230525,-.1127),("R13","holdout"):(.10036,.18777,-.100353,1.378),
 ("R13","full"):(.421587,.029762,-.591813,.2554),
}


def main():  # noqa: PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点 面板"
    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    assert abs(float(y.get((2017, "中报"), np.nan)) - 0.5307) < 0.005, "锚点 泰格同比"
    print(f"锚点 ✓ 面板 {nt}×{ns};泰格同比复现 ✓", flush=True)

    raw = np.full((nt, ns), np.nan, np.float32)
    hi = np.full((nt, ns), np.nan, np.float32)
    lw = np.full((nt, ns), np.nan, np.float32)
    vol = np.full((nt, ns), np.nan, np.float32)
    amt = np.full((nt, ns), np.nan, np.float32)
    t0 = time.time()
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet",
                            columns=["raw_close", "high", "low", "volume", "amount"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        x = x.reindex(idx)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().to_numpy(np.float32)
        for arr, col in ((hi, "high"), (lw, "low"), (vol, "volume"), (amt, "amount")):
            arr[:, j] = pd.to_numeric(x[col], errors="coerce").to_numpy(np.float32)
    print(f"价量矩阵完成 ({time.time()-t0:.0f}s)", flush=True)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点 TTM 恒等式"

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill()
    bsr = bs.reindex(idx).ffill()
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)

    # R12 的 510300 月线状态:收盘 > 月线MA20 且 月线MACD > signal
    mon = bs.resample("ME").last()
    e12, e26 = mon.ewm(span=12).mean(), mon.ewm(span=26).mean()
    macd = e12 - e26
    on_m = (mon > mon.rolling(20).mean()) & (macd > macd.ewm(span=9).mean())
    regime = on_m.reindex(idx, method="ffill").fillna(False).to_numpy()

    def lowrisk(t, e):
        w = np.log(np.maximum(cl[max(0, t - 249):t + 1, e].astype(np.float64), 1e-12))
        lr = np.diff(w, axis=0)
        cols = [-np.std(lr[-n:], axis=0, ddof=1) for n in (20, 60, 120)]
        for n in (60, 120, 250):
            ww = w[-n:]
            cols.append(np.min(ww - np.maximum.accumulate(ww, axis=0), axis=0))
        return np.mean([pd.Series(c).rank(pct=True).to_numpy() for c in cols], axis=0)

    def sizeturn(t, e):
        return (pct(-logcap[t, e].astype(float)) + pct(-tmean[t, e].astype(float))) / 2

    def compose(parts):
        return np.mean([pd.Series(p).rank(pct=True).to_numpy() for p in parts], axis=0)

    def route_score(name, t, e):
        if name == "R06":
            return lowrisk(t, e)
        if name == "R10":
            return sizeturn(t, e)
        if name in ("R08", "R09"):
            return route_scores(name, t, e, fm, cl, raw, logcap, tmean, "raw")
        if name in ("R03", "R04", "R05", "R07", "R13"):
            return pv_score(name, t, e, "raw", cl, raw, amt, vol, hi, lw, fm)
        if name == "R11":
            return compose([route_scores("R08", t, e, fm, cl, raw, logcap, tmean, "raw"),
                            route_scores("R09", t, e, fm, cl, raw, logcap, tmean, "raw"),
                            lowrisk(t, e), sizeturn(t, e)])
        if name == "R12_def":
            return compose([route_scores("R08", t, e, fm, cl, raw, logcap, tmean, "raw"),
                            route_scores("R09", t, e, fm, cl, raw, logcap, tmean, "raw"),
                            lowrisk(t, e)])
        raise ValueError(name)

    def build(name):
        """cash_fallback:合格不足 20 只时只买合格的,每只恒 5%,其余留现金。"""
        sel = {}
        for t in reb:
            t = int(t)
            base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
            if name in ("R11", "R12"):
                base = base & (logcap[t] > np.nanpercentile(logcap[t][base], 10))
            e = np.flatnonzero(base)
            if len(e) < TOP_N:
                continue
            key = ("R12_def" if (name == "R12" and not regime[t]) else
                   ("R11" if name == "R12" else name))
            v = route_score(key, t, e)
            g = np.isfinite(v)
            if not g.any():
                sel[t] = (np.zeros(0, np.int64), np.zeros(0))
                continue
            e2 = e[g]
            k = min(TOP_N, len(e2))
            top = e2[np.argsort(-v[g], kind="stable")[:k]]
            sel[t] = (top, np.full(k, WEIGHT))
        return sel

    def wpos(w):
        d0, d1 = WINS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    rows = []
    sig = {"R01": ((bsr > bsr.rolling(200).mean())
                   & (bsr.rolling(50).mean() > bsr.rolling(200).mean())).astype(float),
           "R02": (bsr / bsr.shift(250) - 1.0 > 0).astype(float)}
    for nm in ("R01", "R02"):
        for w, (d0, d1) in WINS.items():
            m = (bs.index >= d0) & (bs.index <= d1)
            px = bs[m]
            s = sig[nm].reindex(px.index).shift(1).fillna(0.0)
            eq = (px.pct_change().fillna(0.0) * s).add(1.0).cumprod().to_numpy()
            r = np.diff(eq) / eq[:-1]
            r = r[np.isfinite(r)]
            sd = r.std(ddof=1) if len(r) > 1 else 0.0
            yrs = max((px.index[-1] - px.index[0]).days / 365.25, 1 / 365.25)
            rows.append({"route": nm, "window": w, "total": float(eq[-1] - 1),
                         "cagr": float(eq[-1] ** (1 / yrs) - 1),
                         "mdd": float(np.min(eq / np.maximum.accumulate(eq) - 1)),
                         "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                         "n_reb": np.nan})
        print(f"{nm} 完成", flush=True)

    for nm in ("R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "R13"):
        sel = build(nm)
        full_k = sum(1 for v in sel.values() if len(v[0]) >= TOP_N)
        for w in WINS:
            w0, w1 = wpos(w)
            eq, dd, tr, fz = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
            m = metrics(eq, dd, idx)
            rows.append({"route": nm, "window": w, "total": m["total"], "cagr": m["cagr"],
                         "mdd": m["mdd"], "sharpe": m["sharpe"], "trades": tr,
                         "frozen": fz, "n_reb": len(sel), "n_full": full_k})
        print(f"{nm} 完成 调仓日{len(sel)} 其中满 20 只 {full_k}", flush=True)

    df = pd.DataFrame(rows)
    for i, r in df.iterrows():
        k = (r["route"], r["window"])
        if k in CODEX:
            ct, cc, cm, cs_ = CODEX[k]
            df.loc[i, ["codex_total", "codex_cagr", "codex_mdd", "codex_sharpe"]] = \
                [ct, cc, cm, cs_]
    df["cagr_diff_pp"] = (df["cagr"] - df["codex_cagr"]) * 100
    df.to_csv(f"{OUT}/all13_report.csv", index=False)
    print(f"\n落库 {OUT}/all13_report.csv")
    fu = df[df["window"] == "full"].set_index("route")
    print(f"\n{'路线':5s} | {'我总收益':>10s} {'我年化':>7s} {'我回撤':>7s} {'我夏普':>6s}"
          f" | {'他总收益':>10s} {'他年化':>7s} {'他回撤':>7s} {'他夏普':>6s} | {'年化差':>8s}")
    for rt in [f"R{i:02d}" for i in range(1, 14)]:
        if rt not in fu.index:
            continue
        r = fu.loc[rt]
        print(f"{rt:5s} | {r['total']:+9.2%} {r['cagr']:+6.2%} {r['mdd']:+6.2%} "
              f"{r['sharpe']:6.2f} | {r['codex_total']:+9.2%} {r['codex_cagr']:+6.2%} "
              f"{r['codex_mdd']:+6.2%} {r['codex_sharpe']:6.2f} | {r['cagr_diff_pp']:+7.2f}pp")


if __name__ == "__main__":
    main()


# =============================================================================
# §122 结果:full 窗口 2014-01-02 → 2025-12-31
#
# 路线 |  我总收益   我年化   我回撤  我夏普 |  他总收益   他年化   他回撤  他夏普 |  年化差
# R01  |  +135.51%  +7.40%  -30.71%  0.56 |   +86.58%  +5.34%  -35.05%  0.42 |  +2.07pp
# R02  |   +98.81%  +5.90%  -43.47%  0.43 |   +68.09%  +4.42%  -50.76%  0.34 |  +1.47pp
# R03  |  +535.83% +16.67%  -61.41%  0.67 |  +902.57% +19.42%  -63.93%  0.78 |  -2.75pp
# R04  |   +27.35%  +2.04%  -43.00%  0.21 |    +8.06%  +0.65%  -20.61%  0.13 |  +1.39pp
# R05  |   +14.47%  +1.13%  -76.73%  0.16 |  +184.60%  +9.11%  -70.56%  0.55 |  -7.98pp
# R06  |  +185.62%  +9.14%  -45.62%  0.63 |  +182.68%  +9.05%  -43.33%  0.62 |  +0.09pp
# R07  |   -74.88% -10.88%  -90.30% -0.43 |   +60.63%  +4.03%  -68.58%  0.29 | -14.91pp
# R08  |  +423.71% +14.80%  -34.80%  0.79 |  +563.16% +17.08%  -37.56%  0.90 |  -2.28pp
# R09  |  +166.00%  +8.50%  -49.83%  0.47 |  +222.47% +10.25%  -53.26%  0.54 |  -1.76pp
# R10  | +1444.01% +25.63%  -56.38%  1.12 | +3135.06% +33.62%  -41.55%  1.35 |  -7.99pp
# R11  |  +636.88% +18.12%  -38.70%  0.97 |  +755.37% +19.60%  -40.35%  0.97 |  -1.48pp
# R12  |  +526.69% +16.53%  -37.25%  0.93 |  +216.23% +10.07%  -31.27%  0.78 |  +6.46pp
# R13  |   +20.59%  +1.57%  -57.47%  0.18 |   +42.16%  +2.98%  -59.18%  0.26 |  -1.40pp
#
# 五个窗口的完整数据见 results/codex_cross_check/all13_report.csv。
#
# ── 按 §117/§118 的 ±6pp 口径,13 条里 9 条落在容差内 ──
# 落在内:R01 R02 R03 R04 R06 R08 R09 R11 R13
# 落在外:R05(-7.98)、R07(-14.91)、R10(-7.99)、R12(+6.46)
#
# ── cash_fallback 修正带来的变化(§121 诊断被证实)──
# R13:§118 的 -34.17% / -3.43% → 本节 **+20.59% / +1.57%**,与他只差 1.40pp。
#      **R13 复现出来了。§118 判它「无法复现」是我的 bug 造成的,不是他的问题。**
# R04:§118 的 -1.22% → 本节 **+2.04%**,与他只差 1.39pp,同样从「不过」变「过」。
# 合格日统计佐证:R13 只有 83/153 个调仓日满 20 只、R04 只有 55/153,
# 其余路线 143~153,所以修正只对这两条产生量级影响,与预期一致。
#
# **更正 §118 的表述(第二次)**:§118 把 R05/R07/R13 一并归因为
# 「他的信号定义在归档里不完整」。对 R13 完全错了 —— 定义是完整的,漏的是我。
# 对 R05/R07 该归因仍然可能成立(它们修正后依旧差 7.98 / 14.91pp)。
#
# ── 四条落在容差外的解释 ──
# R10 -7.99pp:**原因已知且在他一侧** —— 他把前复权价乘股本当市值(§113/§114),
#   宁德 2021-11-30 低估 48.5%。用他的口径重跑可得年化 32.19%(§113 诊断)。
# R12 +6.46pp:**原因在我一侧** —— 他的状态定义(MACD 参数、月线取样口径)
#   归档里不完整,本节的 R12 是近似实现,已在 docstring 标注。不算他的问题。
# R05 -7.98pp / R07 -14.91pp:布林 squeeze 的信号列、MFI 的 trend_60 与
#   vwap_gap_20 口径在归档里没有完整定义,我按最自然的读法实现。**不猜、不调参去凑。**
#   R07 我的版本 full 年化 -10.88%、回撤 -90.30%,与他的 +4.03% 差距过大,
#   基本可以确定不是同一个策略。
#
# ── 与本项目其他结论的关系 ──
# 本节是**描述性对照**,不改变任何判定。
# 市值中性对照与时间样本外的结论仍以 §114/§116/§117/§119 为准:
# 同时扛住对照与样本外的只有 R08 价值、R09 质量(§119),
# 以及微盘族的 small_cap_low_turnover、high_amihud(§116-A,仅 7 个月留出期)。
# **注意 R03 在本节 full 年化 +16.67% 看着不错,但 §118 已测出它 oos p=0.0249
# 过不了 Bonferroni,且 2026 年 -19.24%。收益高不等于有超额。**
# =============================================================================
