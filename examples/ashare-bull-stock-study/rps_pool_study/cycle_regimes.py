"""§126 牛熊周期切分:R01–R13 在每一轮牛市/熊市里谁最好?

用户问题
--------
「按沪深300 月线的牛熊周期来划分,每一轮牛市哪个组合最好,熊市哪个最好?」

**先说这个问题的陷阱。** 事后挑出「上一轮牛市谁最好」几乎必然是数据挖掘 ——
13 条路线 × 若干个周期,总有一条在某个周期里排第一,那多半是噪声。
**真正有信息量的不是排名,是排名的一致性**:
同一条路线是否在**多轮同类周期里稳定胜出**。所以本节的重点输出是一致性统计,
而不是「某轮冠军」。

周期切分(事前写死,不得事后调)
--------------------------------
对象:沪深300(510300)**月末收盘**,本面板口径(含分红,与组合口径一致)。
方法:ZigZag —— 从起点开始,记录running 极值;当回撤/反弹幅度超过阈值 TH 时确认拐点。
    TH = **20%**(主口径,事前定死)
    稳健性:同时跑 **15%** 与 **25%**,若三个阈值给出的周期数或冠军差异很大,
            则本节结论按「不稳健」记录,不下断言。
低点→高点 = 牛市段;高点→低点 = 熊市段。末段未确认拐点者标注「进行中」。

策略口径
--------
R01–R13 全部沿用 §122 `all13_report.py` 的构造(含 cash_fallback、真实不复权价、
R12 为近似实现),引擎与 §114 一致。**本节不改任何策略定义。**
每条路线跑**一条连续净值曲线**(2014-01-02 → 面板末),再按周期切片。

判据
----
**本节不设通过/不通过判据** —— 它是周期切片的描述性统计,不是假设检验。
§113–§125 已分别对各路线下过判定,本节不重判、不翻案。

锚点(不过则本节作废)
----------------------
X1 面板 (3297, 5217)。
X2 **曲线一致性锚点**:连续曲线在 2014-01-02→2025-12-31 上切出的
   总收益/年化/最大回撤/夏普,必须与 §122 `all13_report.csv` 的 full 行
   **逐条相符**(年化绝对差 < 0.05pp)。若本节的构造与 §122 有任何出入,此项必炸。
X3 周期切分自洽:各段首尾相接、无重叠、无缺口;牛熊交替出现。

必须与结果一起读的三条
----------------------
① **每轮冠军不可外推。** 除非同一条在多轮同类周期里稳定居前,否则那只是噪声。
② R01/R02 是 510300 单资产择时,熊市里空仓 → 天然占优;
   R03–R13 是股票组合,始终满仓 → 天然吃亏。**两类不可直接比排名**,分开列。
③ §125 已测出 R08 的持仓 75.8% 是银行;若它在某类周期里领先,
   **首先要怀疑的是那类周期恰好是银行的好时候**,而不是选股能力。
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
from codex_r10_replication import DATA, TOP_N, WEIGHT, pct  # noqa: E402
from codex_routes_rerun import build_fund, route_scores  # noqa: E402
from codex_routes_rest import score as pv_score  # noqa: E402

THS = (0.20, 0.15, 0.25)
TIMING = ("R01", "R02")
STOCKS = ("R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "R13")
CODEX_FULL_CAGR = {"R01": .0740, "R02": .0590, "R03": .1667, "R04": .0204,
                   "R05": .0113, "R06": .0914, "R07": -.1088, "R08": .1480,
                   "R09": .0850, "R10": .2563, "R11": .1812, "R12": .1653,
                   "R13": .0157}


def zigzag(s, th):
    """月末收盘的 ZigZag:确认拐点后切段。返回 [(起, 止, '牛'/'熊'), ...]。"""
    v = s.to_numpy(float)
    d = s.index
    piv = [0]
    direction = 0
    ext, ei = v[0], 0
    for i in range(1, len(v)):
        if direction >= 0 and v[i] > ext:
            ext, ei = v[i], i
        if direction <= 0 and v[i] < ext:
            ext, ei = v[i], i
        if direction >= 0 and v[i] <= ext * (1 - th) and ei != piv[-1]:
            piv.append(ei)
            direction, ext, ei = -1, v[i], i
        elif direction <= 0 and v[i] >= ext * (1 + th) and ei != piv[-1]:
            piv.append(ei)
            direction, ext, ei = 1, v[i], i
    piv.append(len(v) - 1)
    segs = []
    for a, b in zip(piv[:-1], piv[1:], strict=True):
        if b <= a:
            continue
        segs.append((d[a], d[b], "牛" if v[b] > v[a] else "熊"))
    return segs


def main():  # noqa: PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点X1"
    print(f"锚点X1 ✓ {nt}×{ns}", flush=True)

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
    print(f"价量矩阵 {time.time()-t0:.0f}s", flush=True)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点 TTM"

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill()
    bsr = bs.reindex(idx).ffill()
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)

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

    def comp(ps):
        return np.mean([pd.Series(p).rank(pct=True).to_numpy() for p in ps], axis=0)

    def rscore(name, t, e):
        if name == "R06":
            return lowrisk(t, e)
        if name == "R10":
            return sizeturn(t, e)
        if name in ("R08", "R09"):
            return route_scores(name, t, e, fm, cl, raw, logcap, tmean, "raw")
        if name in ("R03", "R04", "R05", "R07", "R13"):
            return pv_score(name, t, e, "raw", cl, raw, amt, vol, hi, lw, fm)
        if name == "R11":
            return comp([route_scores("R08", t, e, fm, cl, raw, logcap, tmean, "raw"),
                         route_scores("R09", t, e, fm, cl, raw, logcap, tmean, "raw"),
                         lowrisk(t, e), sizeturn(t, e)])
        if name == "R12_def":
            return comp([route_scores("R08", t, e, fm, cl, raw, logcap, tmean, "raw"),
                         route_scores("R09", t, e, fm, cl, raw, logcap, tmean, "raw"),
                         lowrisk(t, e)])
        raise ValueError(name)

    def build(name):
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
            v = rscore(key, t, e)
            g = np.isfinite(v)
            if not g.any():
                sel[t] = (np.zeros(0, np.int64), np.zeros(0))
                continue
            e2 = e[g]
            k = min(TOP_N, len(e2))
            sel[t] = (e2[np.argsort(-v[g], kind="stable")[:k]], np.full(k, WEIGHT))
        return sel

    # 一条连续净值曲线(2014-01-02 → 面板末)
    w0 = int(ipos.get_indexer([pd.Timestamp("2014-01-02")], method="bfill")[0])
    w1 = int(ipos.get_indexer([pd.Timestamp("2026-08-03")], method="ffill")[0])
    curves = {}
    sig = {"R01": ((bsr > bsr.rolling(200).mean())
                   & (bsr.rolling(50).mean() > bsr.rolling(200).mean())).astype(float),
           "R02": (bsr / bsr.shift(250) - 1.0 > 0).astype(float)}
    days = cal_pos[(cal_pos >= w0) & (cal_pos <= w1)]
    di = idx[days]
    for nm in TIMING:
        px = bs.reindex(di).ffill()
        s = sig[nm].reindex(di).shift(1).fillna(0.0)
        curves[nm] = pd.Series(
            (px.pct_change().fillna(0.0) * s).add(1.0).cumprod().to_numpy(), index=di)
    for nm in STOCKS:
        eq, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, build(nm), cal_pos, w0, w1)
        curves[nm] = pd.Series(eq / eq[0], index=idx[dd])
        print(f"{nm} 曲线完成", flush=True)
    curves["510300"] = (bs.reindex(di).ffill() / bs.reindex(di).ffill().iloc[0])

    # 锚点 X2:切出 full 窗口须与 §122 相符
    print("\n锚点X2 曲线一致性(vs §122 all13_report full 年化)", flush=True)
    bad = []
    for nm, tgt in CODEX_FULL_CAGR.items():
        c = curves[nm]
        c = c[(c.index >= "2014-01-02") & (c.index <= "2025-12-31")]
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        g = (c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1
        d = abs(g - tgt) * 100
        if d >= 0.05:
            bad.append((nm, g, tgt, d))
    print(f"  超差 {len(bad)} 条 {'✓' if not bad else '✗ ' + str(bad[:4])}")
    assert not bad, "锚点X2 不过"

    rows = []
    for th in THS:
        segs = zigzag(mon, th)
        print(f"\n=== ZigZag 阈值 {th:.0%}:{len(segs)} 段 ===", flush=True)
        for a, bb, kind in segs:
            a2 = max(a, di[0])
            if bb <= a2:
                continue
            res = {}
            for nm, c in curves.items():
                cc = c[(c.index >= a2) & (c.index <= bb)]
                if len(cc) < 5:
                    continue
                res[nm] = float(cc.iloc[-1] / cc.iloc[0] - 1)
            if "510300" not in res:
                continue
            ongoing = bb >= mon.index[-1]
            best_s = max(STOCKS, key=lambda n: res.get(n, -9))
            best_t = max(TIMING, key=lambda n: res.get(n, -9))
            rows.append({"th": th, "起": a2.date(), "止": bb.date(), "类型": kind,
                         "月数": round((bb - a2).days / 30.44), "进行中": ongoing,
                         "沪深300": res["510300"], "最优股票组合": best_s,
                         "最优收益": res[best_s], "最优择时": best_t,
                         "择时收益": res[best_t],
                         **{f"r_{n}": res.get(n) for n in TIMING + STOCKS}})
            print(f"  {a2.date()} → {bb.date()} {kind}{'(进行中)' if ongoing else ''} "
                  f"{round((bb-a2).days/30.44):3d}个月 300:{res['510300']:+8.2%} | "
                  f"股票最优 {best_s} {res[best_s]:+8.2%} | 择时最优 {best_t} "
                  f"{res[best_t]:+8.2%}", flush=True)
    df = pd.DataFrame(rows)
    m = df[df.th == 0.20]
    print("\n=== 一致性:主口径 20% 下各路线在牛/熊段的夺冠次数 ===")
    for kind in ("牛", "熊"):
        sub = m[m["类型"] == kind]
        vc = sub["最优股票组合"].value_counts()
        print(f"  {kind}市 {len(sub)} 段 → " +
              ", ".join(f"{k}×{v}" for k, v in vc.items()))
    print("\n=== 各路线在牛/熊段跑赢沪深300 的次数(主口径 20%)===")
    for n in STOCKS + TIMING:
        w = {k: int((m[m['类型'] == k][f"r_{n}"] > m[m['类型'] == k]["沪深300"]).sum())
             for k in ("牛", "熊")}
        tot = {k: len(m[m["类型"] == k]) for k in ("牛", "熊")}
        print(f"  {n:5s} 牛 {w['牛']}/{tot['牛']}  熊 {w['熊']}/{tot['熊']}")
    df.to_csv(f"{OUT}/cycle_regimes.csv", index=False)
    print(f"\n落库 {OUT}/cycle_regimes.csv")


if __name__ == "__main__":
    main()
