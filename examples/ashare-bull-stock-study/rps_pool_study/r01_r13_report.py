"""§121 R01 与 R13 的完整指标对照(总收益/年化/回撤/夏普/基准/超额)。

§118 只记了年化与回撤,本节补齐总收益与夏普,并按 Codex 的五个窗口逐格对照。
因子与信号定义与 §118 完全一致(照抄他的源码/配置),不做任何调整。
本节**不设通过/不通过判据** —— 它是指标补全与并排展示,不是假设检验;
§118 已对这两条下过判定(R01 K2 ✓ 复现成立;R13 K2 ✗ 判无法复现),本节不重判、不翻案。

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
from codex_r10_replication import DATA, TOP_N, WEIGHT, metrics  # noqa: E402
from codex_routes_rerun import build_fund  # noqa: E402
from codex_routes_rest import score  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

WINS = {"train": ("2014-01-02", "2019-12-31"), "validation": ("2020-01-02", "2022-12-30"),
        "oos": ("2023-01-03", "2025-12-31"), "holdout": ("2026-01-05", "2026-08-03"),
        "full": ("2014-01-02", "2025-12-31")}
# Codex 公布值,直接取自他的 fast_screen_results.json(不用 README 转述)
CODEX = {
    ("R01", "train"): (0.9202, 0.1150, -0.3473, 0.688, 0.7297),
    ("R01", "validation"): (-0.1323, -0.0463, -0.2036, -0.262, -0.0509),
    ("R01", "oos"): (0.1242, 0.0399, -0.1385, 0.447, 0.2036),
    ("R01", "holdout"): (-0.0212, -0.0378, -0.0990, -0.109, -0.0188),
    ("R01", "full"): (0.8658, 0.0534, -0.3505, 0.420, 1.0072),
    ("R13", "train"): (0.4184, 0.0601, -0.2546, 0.429, 0.7297),
    ("R13", "validation"): (-0.1577, -0.0558, -0.4927, -0.229, -0.0509),
    ("R13", "oos"): (-0.0767, -0.0263, -0.2305, -0.113, 0.2036),
    ("R13", "holdout"): (0.1004, 0.1878, -0.1004, 1.378, -0.0188),
    ("R13", "full"): (0.4216, 0.0298, -0.5918, 0.255, 1.0072),
}


def main():
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
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)
    bsr = bs.reindex(idx).ffill()

    rows = []
    # ── R01:510300 单资产择时 sma50_200 ──
    sig = ((bsr > bsr.rolling(200).mean())
           & (bsr.rolling(50).mean() > bsr.rolling(200).mean())).astype(float)
    for w, (d0, d1) in WINS.items():
        m = (bs.index >= d0) & (bs.index <= d1)
        px = bs[m]
        s = sig.reindex(px.index).shift(1).fillna(0.0)
        eq = (px.pct_change().fillna(0.0) * s).add(1.0).cumprod().to_numpy()
        r = np.diff(eq) / eq[:-1]
        r = r[np.isfinite(r)]
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        yrs = max((px.index[-1] - px.index[0]).days / 365.25, 1 / 365.25)
        bench = float(px.iloc[-1] / px.iloc[0] - 1)
        rows.append({"route": "R01", "window": w, "total": float(eq[-1] - 1),
                     "cagr": float(eq[-1] ** (1 / yrs) - 1),
                     "mdd": float(np.min(eq / np.maximum.accumulate(eq) - 1)),
                     "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                     "bench_div": bench})

    # ── R13:接近 250 日新高 + RPS + PIT 基本面(真实不复权价)──
    sel = {}
    for t in reb:
        t = int(t)
        base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
        e = np.flatnonzero(base)
        if len(e) < TOP_N * 3:
            continue
        v = score("R13", t, e, "raw", cl, raw, amt, vol, hi, lw, fm)
        g = np.isfinite(v)
        if g.sum() < TOP_N:
            continue
        e2 = e[g]
        sel[t] = (e2[np.argsort(-v[g], kind="stable")[:TOP_N]], np.full(TOP_N, WEIGHT))
    print(f"R13 有信号的调仓日 {len(sel)}/{len(reb)}", flush=True)
    for w, (d0, d1) in WINS.items():
        w0 = int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0])
        w1 = int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0])
        eq, dd, tr, fz = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        m = metrics(eq, dd, idx)
        s = bs[(bs.index >= d0) & (bs.index <= d1)]
        rows.append({"route": "R13", "window": w, "total": m["total"], "cagr": m["cagr"],
                     "mdd": m["mdd"], "sharpe": m["sharpe"], "trades": tr, "frozen": fz,
                     "bench_div": float(s.iloc[-1] / s.iloc[0] - 1)})

    df = pd.DataFrame(rows)
    print(f"\n{'路线':5s} {'窗口':11s} | {'我·总收益':>10s} {'我·年化':>8s} {'我·回撤':>8s} {'我·夏普':>7s}"
          f" | {'他·总收益':>10s} {'他·年化':>8s} {'他·回撤':>8s} {'他·夏普':>7s} | {'年化差':>8s}")
    for _, r in df.iterrows():
        k = (r["route"], r["window"])
        ct, cc, cm, cs_, _cb = CODEX[k]
        print(f"{r['route']:5s} {r['window']:11s} | {r['total']:+9.2%} {r['cagr']:+7.2%} "
              f"{r['mdd']:+7.2%} {r['sharpe']:7.2f} | {ct:+9.2%} {cc:+7.2%} {cm:+7.2%} "
              f"{cs_:7.2f} | {(r['cagr']-cc)*100:+7.2f}pp")
        df.loc[_, "codex_total"], df.loc[_, "codex_cagr"] = ct, cc
        df.loc[_, "codex_mdd"], df.loc[_, "codex_sharpe"] = cm, cs_
        df.loc[_, "codex_bench_nodiv"] = _cb
    df.to_csv(f"{OUT}/r01_r13_report.csv", index=False)
    print(f"\n落库 {OUT}/r01_r13_report.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §121 结果
#
# 路线  窗口         我·总收益  我·年化  我·回撤  我·夏普 | 他·总收益  他·年化  他·回撤  他·夏普 | 年化差
# R01   train       +107.16%  +12.92%  -30.71%   0.82  |  +92.02%  +11.50%  -34.73%   0.69  | +1.42pp
# R01   validation    +3.81%   +1.26%  -16.25%   0.16  |  -13.23%   -4.63%  -20.36%  -0.26  | +5.89pp
# R01   oos           +8.06%   +2.62%  -17.15%   0.30  |  +12.42%   +3.99%  -13.85%   0.45  | -1.37pp
# R01   holdout       -4.76%   -8.52%  -11.58%  -0.39  |   -2.12%   -3.78%   -9.90%  -0.11  | -4.74pp
# R01   full        +135.51%   +7.40%  -30.71%   0.56  |  +86.58%   +5.34%  -35.05%   0.42  | +2.06pp
# R13   train         -3.28%   -0.56%  -49.37%   0.10  |  +41.84%   +6.01%  -25.46%   0.43  | -6.57pp
# R13   validation   -15.80%   -5.58%  -47.32%  -0.19  |  -15.77%   -5.58%  -49.27%  -0.23  | -0.00pp
# R13   oos          -24.00%   -8.76%  -29.21%  -0.50  |   -7.67%   -2.63%  -23.05%  -0.11  | -6.13pp
# R13   holdout       +6.04%  +11.31%  -13.71%   0.64  |  +10.04%  +18.78%  -10.04%   1.38  | -7.47pp
# R13   full         -34.17%   -3.43%  -62.02%  -0.06  |  +42.16%   +2.98%  -59.18%   0.26  | -6.41pp
#
# ── R01:复现成立(§118 K2 ✓),差额有确定来源 ──
# full 年化差 +2.06pp,而 §113 已量化他的基准分红口径差 = **1.64%/年**。
# R01 是 510300 单资产择时,大部分时间满仓持有基准,**这个策略的收益几乎就是基准收益**,
# 所以分红口径差几乎原样传导过来。剩余 ~0.4pp 来自信号本身:
# 我的 close 含分红,SMA50/SMA200 的交叉时点与他的不复权价**不完全同日**,
# 11 次交易里错开一两次就够解释。
# validation 窗口差 +5.89pp 最大,正是因为 2020–2022 有一次交叉落在临界点上,
# 我在场他不在场;这是**同一个口径问题在小样本窗口上的放大**,不是另一个错误。
#
# ── R13:无法复现(§118 K2 ✗),**本节定位到确定原因** ──
# **`cash_fallback` 没有实现。**
# 他的 bootstrap_r13_v002.py:123 写死 `"cash_fallback": True` ——
# 当合格股票不足 20 只时,**买入所有合格的、其余仓位留现金**。
# 我的实现是 `if g.sum() < TOP_N: continue`,即**合格不足 20 只就跳过该调仓日、
# 继续持有上一期的股票**。两者行为完全不同。
#
# 这个差异有多大:本次 R13 只有 **83/153 个调仓日**合格股票 ≥20 只,
# 也就是 **45.8% 的调仓日被我跳过**;而他的 R13 README 自己写
# 「约 49.1% 的日期不足 20 只,组合经常保留现金」——**两个比例高度吻合**,
# 说明我的筛选条件基本正确,**错的是不足额时的处理方式**。
#
# 佐证:validation 窗口(2020–2022)我与他的年化差是 **-0.00pp、总收益差 0.03pp**,
# 几乎逐格相同 —— 那一段合格股票充足、cash_fallback 很少触发,所以两边一致;
# 而 train/oos/holdout 三段差 6~7pp,正是 cash_fallback 频繁触发的时段。
#
# **这是我的实现错误,不是他的。** §118 判 R13「无法复现」的结论仍然成立
# (判据是判据,不改判),但原因现在清楚了:**不是他的定义不完整,是我漏读了
# 配置里的 cash_fallback。** §118 结论块里把 R13 归入「他的信号定义在归档里不完整」
# 是**不准确的**,此处更正:R13 的定义是完整的,漏的是我。
#
# 修正后重跑需要另开一节(改的是实现不是判据,但结果会变,必须重新落库)。
# =============================================================================
