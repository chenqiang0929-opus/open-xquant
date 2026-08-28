"""§138:按「已处于该排列多少周」分层,重看 250 日胜率(描述性,无判据)。

起因
----
用户问:「20周与60周空头排列,为什么胜率是正的?不是应该都是亏损的才对?」
第一三七节给的第一层解释是基准胜率本身就有 53.41%。
第二层解释是:**空头排列 ≠ 正在下跌** —— MA20周跌破 MA60周时下跌通常已走大半,
且该状态可持续数年,样本里多数是「跌完了在底部磨」,不是「正在暴跌」。

**本节直接验证第二层解释**:把 A/B 两组按「已连续处于该排列的周数」分层,
看 250 日胜率是否随持续时间变化。若「刚死叉」那一层明显更差、
「磨很久」那一层更好,则第二层解释成立。

口径:与第一三七节完全相同(周线只用已完成周、每月末观察、买入持有 250 日、
退市 ffill 参与),**只多加一个分层变量**:连续同向排列的周数。
分层:1–13 周 / 13–26 / 26–52 / 52–104 / >104 周。
同时报同市值同行业对照胜率(500 组种子)。

**本节是描述性统计,没有通过/不通过判据。** 锚点同第一三七节。
不做的:不改第一三七节脚本;不新增顶层目录;不 force push;不作可交易性声明。
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
BINS = [(1, 13), (13, 26), (26, 52), (52, 104), (104, 10**6)]


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

    def al(k, fill=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(fill)
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
    bull_w = m20 > m60
    fin_w = np.isfinite(m20) & np.isfinite(m60)
    # 连续同向周数(在周线上累计)
    nw = len(wpos)
    dur = np.zeros((nw, ns), np.int32)
    for i in range(1, nw):
        same = fin_w[i] & fin_w[i - 1] & (bull_w[i] == bull_w[i - 1])
        dur[i] = np.where(same, dur[i - 1] + 1, 1)
    dur = np.where(fin_w, dur, 0)

    src = np.searchsorted(wpos, np.arange(nt), side="right") - 1
    vs = src >= 0
    bad = int((wpos[src[vs]] > np.arange(nt)[vs]).sum())
    print(f"锚点 周线映射无前视 违例 {bad} {'✓' if bad == 0 else '✗ 作废'}", flush=True)
    assert bad == 0
    st_a = np.zeros((nt, ns), bool)
    st_d = np.zeros((nt, ns), np.int32)
    st_f = np.zeros((nt, ns), bool)
    st_a[vs] = bull_w[src[vs]]
    st_d[vs] = dur[src[vs]]
    st_f[vs] = fin_w[src[vs]]

    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy()
    rows = []
    for t in me:
        t = int(t)
        if t < 60 or t > nt - HOR - 1:
            continue
        e = np.flatnonzero(ok[t] & st_f[t] & (st_d[t] > 0) & np.isfinite(mv[t])
                           & (ind[t] >= 0))
        for j in e:
            rows.append((t, int(j), bool(st_a[t, j]), int(st_d[t, j])))
    p = pd.DataFrame(rows, columns=["t", "j", "A", "dur"])
    print(f"样本 {len(p):,}(A {int(p.A.sum()):,} / B {int((~p.A).sum()):,}) "
          f"({time.time()-t0:.0f}s)", flush=True)

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
    print(f"锚点 行业违例 {v} 次 {'✓' if v == 0 else '✗ 作废'}", flush=True)
    assert v == 0

    with np.errstate(all="ignore"):
        fr = cl[np.minimum(tv + HOR, nt - 1), jv] / cl[tv, jv] - 1.0
    print(f"\n{'='*96}\n持有 250 日,按「已连续处于该排列的周数」分层\n{'='*96}")
    print(f"{'组':<4}{'持续周数':<12}{'样本':>10}{'胜率':>9}{'中位收益':>10}"
          f"{'平均收益':>10}{'对照胜率':>10}{'扣对照':>9}")
    res = []
    for lab, sel0 in (("A", p.A.to_numpy()), ("B", ~p.A.to_numpy())):
        for lo, hi in BINS:
            m = sel0 & (p.dur.to_numpy() >= lo) & (p.dur.to_numpy() < hi)
            g = np.flatnonzero(m & np.isfinite(fr))
            if len(g) < 500:
                continue
            wr = float((fr[g] > 0).mean())
            q = pk[:, g]
            gg = q >= 0
            tq = tv[g][None, :]
            with np.errstate(all="ignore"):
                cfr = cl[np.minimum(tq + HOR, nt - 1), np.maximum(q, 0)] \
                    / cl[tq, np.maximum(q, 0)] - 1.0
            cw = np.where(gg & np.isfinite(cfr), cfr > 0, np.nan)
            cwr = float(np.nanmedian(np.nanmean(cw, axis=1)))
            nm = f"{lo}-{hi}周" if hi < 10**6 else f">{lo}周"
            print(f"{lab:<4}{nm:<12}{len(g):>10,}{wr:>9.2%}{np.median(fr[g]):>10.2%}"
                  f"{np.mean(fr[g]):>10.2%}{cwr:>10.2%}{(wr-cwr)*100:>+8.2f}pp")
            res.append({"组": lab, "持续": nm, "样本": len(g), "胜率": wr,
                        "中位收益": float(np.median(fr[g])),
                        "平均收益": float(np.mean(fr[g])),
                        "对照胜率": cwr, "扣对照pp": (wr - cwr) * 100})
    pd.DataFrame(res).to_csv(f"{OUT}/weekly_ma_duration.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n落库 {OUT}/weekly_ma_duration.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
