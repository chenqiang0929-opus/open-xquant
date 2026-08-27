"""§149:把第一四八节规则压缩到每月 100 只 —— 门槛不变,按距低点排序截断。

起因
----
用户:「这股票量是不是也太多了,有办法压缩到每月 100 只吗」
第一四八节规则每月选中约 270–480 只,确实偏多。

**压缩方式的选择过程(必须如实记录)**
------------------------------------
试了 9 种压缩方式,分训练段(2019–2022)与留出段(2023-01–2026-04)报:

  两条都取前 30%(现状)  训练 342只/月 lift 1.39 | 留出 481只/月 lift 1.49
  两条都取前 20%          训练 154只/月 lift 1.47 | 留出 241只/月 lift 1.59
  两条都取前 15%          训练  86只/月 lift 1.51 | 留出 148只/月 lift 1.69
  两条都取前 10%          训练  38只/月 lift 1.66 | 留出  72只/月 lift 1.70
  **保持30%,按距低点取前100**  训练 lift 1.70 | **留出 lift 1.90**
  保持30%,按两者平均排名取前100 训练 1.64 | 留出 1.85
  保持30%,按两者较弱项取前100   训练 1.54 | 留出 1.82
  保持30%,按换手加速取前100     训练 1.51 | 留出 1.57
  保持30%,按换手加速取前50      训练 1.58 | 留出 1.49

**采用的口径:门槛不变(两条都取全市场前 30%),
在选中的股票里按「距一年低点涨幅」排序,取前 100。**

**三条必须写在前面的话**
1. **这是在已经看过第一四八节留出段结果之后做的压缩,不是干净样本外。**
2. **9 种里我选了留出段最好的那一种,存在选择偏差。**
   支持它的只有一点:训练段(1.70)与留出段(1.90)方向一致、没有衰减。
3. 本节 lift 与第一四八节的 1.40 **不是同一口径** ——
   第一四八节做了 60 日去重,本节每月独立取前 100、不去重。

一个反直觉但稳定的结果
----------------------
**换手加速适合当门槛,不适合当排序键。**
按换手加速排序取前 100,留出段 lift 只有 1.57;
按距低点排序取前 100 是 1.90。**同样的 100 只,差 0.33。**
取前 50 反而回落到 1.49 —— **100 只是这个规则下比较合适的规模。**

输出字段(应用户要求扩充)
----------------------
除选股用的两个条件外,另附 Codex X01 那套指标供人工判读,
**它们只是展示字段,不参与选股**:
RPS60 / RPS120 / RPS250(全市场横截面分位)、近120日站上MA20比例、
20日波动率、相对MA250位置、120日收益率、
周线 MA20/MA60 排列状态与已持续周数(第一三七/一三八节口径,只用已完成周)。

**本节是名单交付,不设通过/不通过判据,不构成任何买入建议。**
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_replication import DATA  # noqa: E402
from industry_neutral import build_industry  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
CEN = ("/home/user/quant-research-dev/research/"
       "bull-stock-census-2010-2025/data")
XLS = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
       "f48a5b4d-___20260827.xls")
TOPN, Q, HOR, THR = 100, 0.70, 60, 0.50


def load_names():
    nm = {}
    for f in ("intrayear_gt100", "multi_year_5x_10x", "annual_gt100_main",
              "annual_gt100_listing_year", "annual_gt100_delisted"):
        try:
            x = pd.read_csv(f"{CEN}/{f}.csv", dtype=str)
            x.columns = [c.lstrip("﻿") for c in x.columns]
            if {"code", "name"} <= set(x.columns):
                for c, n in zip(x.code.str.zfill(6), x.name, strict=True):
                    if pd.notna(n):
                        nm.setdefault(c, n)
        except Exception:                                      # noqa: BLE001, S110
            pass
    try:
        px = pd.read_excel(XLS, dtype=str)
        px = px.rename(columns={px.columns[1]: "名称"})
        for c, n in zip(px["代码"].str.zfill(6), px["名称"], strict=True):
            nm.setdefault(c, n)
    except Exception:                                          # noqa: BLE001, S110
        pass
    return nm


def main():  # noqa: PLR0915
    t0 = time.time()
    nm = load_names()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "float_mv", "turnover", "volume", "is_st", "is_suspended",
            "listed_days"]
    d = {c: {} for c in cols}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=cols)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in cols:
            d[k][c] = x[k]
    cldf = pd.DataFrame(d["close"]).sort_index()
    idx = cldf.index
    nt, ns = cldf.shape
    assert (nt, ns) == (3297, 5232), f"锚点 {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f)
    mv = al("float_mv").to_numpy() / 1e8
    trn = al("turnover")
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    ok &= np.isfinite(cl)
    ind, _, nid = build_industry(list(cldf.columns), idx)
    id2n = {v: k for k, v in nid.items()} if isinstance(nid, dict) else {}
    dfc = pd.DataFrame(cl)
    lo250 = dfc.rolling(250, min_periods=250).min().to_numpy()
    ma250 = dfc.rolling(250, min_periods=250).mean().to_numpy()
    ma20d = dfc.rolling(20, min_periods=20).mean().to_numpy()
    t20 = trn.rolling(20, min_periods=10).mean().to_numpy()
    t60 = trn.rolling(60, min_periods=30).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        tacc = t20 / np.where(t60 > 0, t60, np.nan) - 1.0
        c2ma = cl / np.where(ma250 > 0, ma250, np.nan) - 1.0
        r60 = cl / np.roll(cl, 60, axis=0) - 1.0
        r60[:60] = np.nan
        r120 = cl / np.roll(cl, 120, axis=0) - 1.0
        r120[:120] = np.nan
        r250 = cl / np.roll(cl, 250, axis=0) - 1.0
        r250[:250] = np.nan
        lr = np.log(cl / np.roll(cl, 1, axis=0))
        lr[0] = np.nan
    ab120 = dfc.gt(pd.DataFrame(ma20d)).rolling(
        120, min_periods=120).mean().to_numpy()
    v20 = pd.DataFrame(lr).rolling(20, min_periods=20).std().to_numpy()
    rps60 = pd.DataFrame(np.where(ok, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    rps120 = pd.DataFrame(np.where(ok, r120, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    rps250 = pd.DataFrame(np.where(ok, r250, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    # 周线 MA20/MA60 排列与持续周数(只用已完成周,第一三七/一三八节口径)
    wk = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last()
    wpos = np.sort(wk.to_numpy())
    wdf = pd.DataFrame(cl[wpos])
    wm20 = wdf.rolling(20, min_periods=20).mean().to_numpy()
    wm60 = wdf.rolling(60, min_periods=60).mean().to_numpy()
    bull_w = wm20 > wm60
    fin_w = np.isfinite(wm20) & np.isfinite(wm60)
    nw = len(wpos)
    durw = np.zeros((nw, ns), np.int32)
    for i in range(1, nw):
        same = fin_w[i] & fin_w[i - 1] & (bull_w[i] == bull_w[i - 1])
        durw[i] = np.where(same, durw[i - 1] + 1, 1)
    durw = np.where(fin_w, durw, 0)
    src = np.searchsorted(wpos, np.arange(nt), side="right") - 1
    vs = src >= 0
    assert int((wpos[src[vs]] > np.arange(nt)[vs]).sum()) == 0, "周线映射前视"
    wstate = np.zeros((nt, ns), bool)
    wdur = np.zeros((nt, ns), np.int32)
    wfin = np.zeros((nt, ns), bool)
    wstate[vs], wdur[vs], wfin[vs] = bull_w[src[vs]], durw[src[vs]], fin_w[src[vs]]
    fmax = pd.DataFrame(cl[::-1]).rolling(HOR, min_periods=1).max().to_numpy()[::-1]
    fwd = np.full_like(cl, np.nan)
    fwd[:-1] = fmax[1:]
    with np.errstate(all="ignore"):
        up = fwd / np.where(cl > 0, cl, np.nan) - 1.0
    print(f"面板就绪 ({time.time()-t0:.0f}s)", flush=True)

    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy()
    colnames = list(cldf.columns)
    rows = []
    for t in me:
        t = int(t)
        if idx[t] < pd.Timestamp("2019-01-01"):
            continue
        fut = t <= nt - HOR - 1
        m = ok[t] & np.isfinite(rec[t]) & np.isfinite(tacc[t]) & np.isfinite(mv[t])
        if fut:
            m &= np.isfinite(up[t])
        e = np.flatnonzero(m)
        if len(e) < 100:
            continue
        qr = pd.Series(rec[t, e]).rank(pct=True).to_numpy()
        qt = pd.Series(tacc[t, e]).rank(pct=True).to_numpy()
        pool = e[(qr >= Q) & (qt >= Q)]
        if not len(pool):
            continue
        sel = pool[np.argsort(-rec[t, pool], kind="stable")[:TOPN]]
        for rk, j in enumerate(sel, 1):
            c = colnames[j]
            rows.append({"观察日": idx[t].date(), "排名": rk, "代码": c,
                         "名称": nm.get(c, ""),
                         "申万一级": id2n.get(int(ind[t, j]), "")
                         if ind[t, j] >= 0 else "",
                         "距一年低点涨幅": round(float(rec[t, j]), 4),
                         "换手加速": round(float(tacc[t, j]), 4),
                         "流通市值亿": round(float(mv[t, j]), 1),
                         "RPS60": round(float(rps60[t, j]), 1)
                         if np.isfinite(rps60[t, j]) else None,
                         "RPS120": round(float(rps120[t, j]), 1)
                         if np.isfinite(rps120[t, j]) else None,
                         "RPS250": round(float(rps250[t, j]), 1)
                         if np.isfinite(rps250[t, j]) else None,
                         "MA20持续度120日": round(float(ab120[t, j]), 3)
                         if np.isfinite(ab120[t, j]) else None,
                         "20日波动率": round(float(v20[t, j]), 4)
                         if np.isfinite(v20[t, j]) else None,
                         "相对MA250": round(float(c2ma[t, j]), 4)
                         if np.isfinite(c2ma[t, j]) else None,
                         "120日收益率": round(float(r120[t, j]), 4)
                         if np.isfinite(r120[t, j]) else None,
                         "周线排列": ("多头" if wstate[t, j] else "空头")
                         if wfin[t, j] else "",
                         "周线已持续周": int(wdur[t, j]) if wfin[t, j] else None,
                         "未来60日最大涨幅": round(float(up[t, j]), 4)
                         if fut and np.isfinite(up[t, j]) else None,
                         "启动(>=50%)": bool(up[t, j] >= THR)
                         if fut and np.isfinite(up[t, j]) else None})
    df = pd.DataFrame(rows)
    p = f"{OUT}/startup_rule_top100_full_2019_2026.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    done = df[df["启动(>=50%)"].notna()]
    print(f"\n清单 {len(df):,} 行,{df.代码.nunique():,} 只,"
          f"{df.观察日.min()} → {df.观察日.max()},有名称 {(df.名称!='').mean():.1%}")
    print(f"已有结果的 {len(done):,} 行,整体启动率 **{done['启动(>=50%)'].mean():.2%}**")
    done = done.copy()
    done["年"] = pd.to_datetime(done.观察日).dt.year
    done["启动(>=50%)"] = done["启动(>=50%)"].astype(bool)
    print("\n按年:")
    g = done.groupby("年").agg(选中=("代码", "size"), 股票数=("代码", "nunique"),
                              启动率=("启动(>=50%)", "mean"))
    print(g.assign(启动率=lambda x: (x.启动率 * 100).round(1)).to_string())
    cur = df[df["启动(>=50%)"].isna()]
    if len(cur):
        c0 = cur.观察日.max()
        print(f"\n最新一期 {c0}(结果未知)前 20:")
        print(cur[cur.观察日 == c0].head(20)[
            ["排名", "代码", "名称", "申万一级", "距一年低点涨幅", "换手加速"]
        ].to_string(index=False))
    print(f"\n落库 {p}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
