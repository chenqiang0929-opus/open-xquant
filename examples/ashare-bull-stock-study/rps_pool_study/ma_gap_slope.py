"""§140:空头排列内部再分层 —— 两线间距 gap 与 MA60 斜率(描述性,无判据)。

起因
----
用户看图指出:同样是空头排列,形态完全不同 ——
「海尔生物、贝泰妮,20周均线和60周均线一直横盘,起不来;
或者就像百润股份,有两次机会,但是就无法持续性的新高。」

这比「多头/空头」精确一层,**也正是第一三八/一三九节缺的那一维**:
第一三八节只用「排列方向 + 持续周数」,没有区分
「两线黏在一起横盘」与「两线张开持续下行」。

新增两个维度(观察日当日可算,无前视)
------------------------------------
- **gap** = MA20周 / MA60周 − 1(空头排列时为负;越接近 0 表示两线越黏合)
- **slope60** = MA60周(t) / MA60周(t−26周) − 1(MA60 过去半年的变化率)

在**空头排列**样本内做 3×3 分层:
  slope60:< −10%(明显下行) / −10%~0%(缓降) / > 0%(走平或上行)
  gap:    < −15%(张开) / −15%~−5%(中等) / > −5%(**黏合**)

**本节是描述性统计,没有通过/不通过判据。**
**第一三九节的教训必须照做:每一格都要报年份集中度**,
最大年份占比过高的格子不能当规律看。同时报同市值同行业对照。

锚点(不过则作废)
------------------
面板 (3297, 5232);周线映射无前视违例 0;行业违例 0。

不做的
------
不改第一三八/一三九节;不新增顶层目录;不 force push;
**不因为某一格好看就宣称找到规则** —— 那要另开一节重新事前登记;
不作任何可交易性声明。
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
from codex_r10_neutral import NBR, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from industry_neutral import build_industry  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NSEED, HOR = 500, 250
SL = [(-10.0, -0.10, "明显下行"), (-0.10, 0.0, "缓降"), (0.0, 10.0, "走平/上行")]
GP = [(-10.0, -0.15, "张开"), (-0.15, -0.05, "中等"), (-0.05, 0.0, "黏合")]
CASES = [("300957", "贝泰妮"), ("688139", "海尔生物"), ("002568", "百润股份")]


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "float_mv", "volume", "is_st", "is_suspended", "listed_days"]
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
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    ok &= np.isfinite(cl)
    ind, _, _ = build_industry(list(cldf.columns), idx)

    wk = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last()
    wpos = np.sort(wk.to_numpy())
    wdf = pd.DataFrame(cl[wpos])
    m20 = wdf.rolling(20, min_periods=20).mean().to_numpy()
    m60 = wdf.rolling(60, min_periods=60).mean().to_numpy()
    with np.errstate(all="ignore"):
        gapw = m20 / np.where(m60 > 0, m60, np.nan) - 1.0
        slw = m60 / np.where(np.roll(m60, 26, axis=0) > 0,
                             np.roll(m60, 26, axis=0), np.nan) - 1.0
    slw[:26] = np.nan
    finw = np.isfinite(gapw) & np.isfinite(slw)
    src = np.searchsorted(wpos, np.arange(nt), side="right") - 1
    vs = src >= 0
    bad = int((wpos[src[vs]] > np.arange(nt)[vs]).sum())
    print(f"锚点 周线映射无前视 违例 {bad} {'✓' if bad == 0 else '✗ 作废'}", flush=True)
    assert bad == 0
    g_d = np.full((nt, ns), np.nan)
    s_d = np.full((nt, ns), np.nan)
    f_d = np.zeros((nt, ns), bool)
    g_d[vs], s_d[vs], f_d[vs] = gapw[src[vs]], slw[src[vs]], finw[src[vs]]

    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy()
    rows = []
    for t in me:
        t = int(t)
        if t < 60 or t > nt - HOR - 1:
            continue
        e = np.flatnonzero(ok[t] & f_d[t] & (g_d[t] < 0) & np.isfinite(mv[t])
                           & (ind[t] >= 0))
        for j in e:
            rows.append((t, idx[t].year, int(j), g_d[t, j], s_d[t, j],
                         cl[t + HOR, j] / cl[t, j] - 1.0))
    p = pd.DataFrame(rows, columns=["t", "year", "j", "gap", "slope", "fr"])
    p = p[np.isfinite(p.fr)]
    print(f"空头排列样本 {len(p):,} ({time.time()-t0:.0f}s)", flush=True)

    tv, jv = p.t.to_numpy(), p.j.to_numpy()
    pre = {}
    for t in np.unique(tv):
        e = np.flatnonzero(ok[t] & np.isfinite(mv[t]) & (ind[t] >= 0))
        o = e[np.argsort(mv[t, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        pre[t] = (o, rk)
    ch, off, lens = [], np.zeros(len(p), np.int64), np.zeros(len(p), np.int64)
    pos, keep = 0, np.ones(len(p), bool)
    for k in range(len(p)):
        t, j = int(tv[k]), int(jv[k])
        o, rk = pre[t]
        p0, i0 = rk[j], ind[t, j]
        a_, b_ = max(0, p0 - NBR), min(len(o) - 1, p0 + NBR)
        cand = o[a_:b_ + 1]
        cand = cand[ind[t, cand] == i0]
        if len(cand) < 2:
            cand = o[ind[t, o] == i0]
        if len(cand) < 2:
            keep[k] = False
            continue
        off[k], lens[k] = pos, len(cand)
        pos += len(cand)
        ch.append(cand)
    flat = np.concatenate(ch).astype(np.int64)
    rng = np.random.default_rng(SEED)
    pk = np.full((NSEED, len(p)), -1, np.int64)
    kk = np.flatnonzero(keep)
    for s0 in range(0, NSEED, 50):
        r = rng.random((50, len(kk)))
        pk[s0:s0 + 50, kk] = flat[off[kk][None, :]
                                  + (r * lens[kk][None, :]).astype(np.int64)]
    v = int((ind[tv[kk], pk[:, kk]] != ind[tv[kk], jv[kk]][None, :]).sum())
    print(f"锚点 行业违例 {v} 次 {'✓' if v == 0 else '✗ 作废'}\n", flush=True)
    assert v == 0

    print("=" * 108)
    print("空头排列内部 3×3(行=MA60半年斜率,列=两线间距);每格:n / 胜率 / 中位 / 扣对照 / 最大年份占比")
    print("=" * 108)
    print(f"{'MA60斜率':<12}" + "".join(f"{g[2]:>30}" for g in GP))
    res = []
    for slo, shi, sname in SL:
        line = f"{sname:<12}"
        for glo, ghi, gname in GP:
            m = ((p.slope.to_numpy() >= slo) & (p.slope.to_numpy() < shi)
                 & (p.gap.to_numpy() >= glo) & (p.gap.to_numpy() < ghi))
            gi = np.flatnonzero(m)
            if len(gi) < 300:
                line += f"{'样本<300':>30}"
                continue
            fr = p.fr.to_numpy()[gi]
            wr = float((fr > 0).mean())
            q = pk[:, gi]
            gg = q >= 0
            tq = tv[gi][None, :]
            with np.errstate(all="ignore"):
                cfr = cl[np.minimum(tq + HOR, nt - 1), np.maximum(q, 0)] \
                    / cl[tq, np.maximum(q, 0)] - 1.0
            cw = np.where(gg & np.isfinite(cfr), cfr > 0, np.nan)
            cwr = float(np.nanmedian(np.nanmean(cw, axis=1)))
            yc = p.year.to_numpy()[gi]
            ymax = float(pd.Series(yc).value_counts(normalize=True).iloc[0])
            line += (f"{len(gi):>7,} {wr:>6.1%} {np.median(fr):>+7.1%} "
                     f"{(wr-cwr)*100:>+6.1f} {ymax:>5.0%}")
            res.append({"斜率": sname, "间距": gname, "n": len(gi), "胜率": wr,
                        "中位": float(np.median(fr)), "对照胜率": cwr,
                        "扣对照pp": (wr - cwr) * 100, "最大年份占比": ymax})
        print(line)
    print("\n注:「最大年份占比」= 该格样本里占比最高的那个买入年份的份额。"
          "第一三九节的教训 —— 占比过高的格子是市场时点,不是规律。")

    print("\n" + "=" * 108)
    print("三只案例股在最近可测观察点的落位")
    print("=" * 108)
    cp = {c: j for j, c in enumerate(cldf.columns)}
    for code, nm in CASES:
        j = cp.get(code)
        sub = p[(p.j == j)].tail(3)
        if not len(sub):
            print(f"  {nm} {code}:无空头排列可测点")
            continue
        for _, r in sub.iterrows():
            print(f"  {nm} {code} {idx[int(r.t)].date()}  gap {r.gap:+.1%}  "
                  f"MA60半年斜率 {r.slope:+.1%}  未来250日 {r.fr:+.1%}")
    pd.DataFrame(res).to_csv(f"{OUT}/ma_gap_slope.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n落库 {OUT}/ma_gap_slope.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
