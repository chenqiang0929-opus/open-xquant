"""§137:周线 ma20d vs ma60d,多头排列(A)与空头排列(B)买入后的胜率。

用户问:「一只股票 20周均线与 60周均线多头排列(A)和空头排列(B),
你买入之后是 A 胜率大还是 B 胜率大?」

**本节是描述性统计,没有通过/不通过判据** —— 问的是「胜率是多少」,
不是「某个规则成不成立」,所以没有可以事后放宽的门槛。口径写在下面,跑完照报。

口径(跑之前写死)
------------------
- **周线**:按自然周重采样,取每周最后一个交易日的收盘价(前复权)。
  `ma20d周` / `ma60d周` 在周线序列上计算,**只用已完成的周**,
  再前向映射回日线(第 t 日用的是 t 之前最后一个已收周的均线值)——**逐点断言无前视**。
- **A 多头排列**:ma20d周 > ma60d周;**B 空头排列**:ma20d周 < ma60d周。
- **观察点**:每月最后一个交易日(避免同一状态被日频重复计数)。
- **买入并持有**:前瞻 **20 / 60 / 120 / 250** 个交易日的**持有期收益**
  (不是峰值,用户问的是买入之后)。
- **胜率**:P(持有期收益 > 0)。
- 合格样本:非 ST、非停牌、上市满 250 日、当日有成交。
- 退市股按最后有效价 ffill 参与,**绝不剔除**。

同时报两组数,缺一不可
----------------------
1. **绝对胜率** —— 用户直接问的那个;
2. **同市值同行业对照的胜率**(名次 ±25 + 同申万一级行业,500 组种子)。
   **理由**:绝对胜率会被市场整体涨跌主导(牛市里什么都涨),
   两组的样本又落在不同的市场时点上 —— A 多出现在上涨期、B 多出现在下跌期,
   **不做对照就是在比市场状态,不是在比排列。**

锚点(不过则本节作废)
----------------------
V1(a) 面板 (3297, 5232);
V1(b) **无前视**:周线均线映射回日线后,逐点断言所用周收盘日 ≤ 观察日;
V1(c) 行业恒等式:对照与被对照股同行业,违例 > 0 即作废;
V1(d) A/B 两组样本数之和 = 全部合格观察数(排列只有两种,不许有第三类)。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;
**不因为某个前瞻期结果好看就单独拎出来当结论** —— 四个前瞻期一起报。
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
NSEED = 500
HORS = (20, 60, 120, 250)


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
    assert (nt, ns) == (3297, 5232), f"锚点V1a {cldf.shape}"

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
    print(f"锚点V1a ✓ {cldf.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 周线 ma20d/ma60d,只用已完成周,再前向映射回日线 ----
    wk = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last()
    wpos = np.sort(wk.to_numpy())
    wcl = cl[wpos]
    wdf = pd.DataFrame(wcl)
    m20 = wdf.rolling(20, min_periods=20).mean().to_numpy()
    m60 = wdf.rolling(60, min_periods=60).mean().to_numpy()
    # 日线第 t 日用「最后一个收盘日 <= t」的那一周的均线;严格用上一完成周
    src = np.searchsorted(wpos, np.arange(nt), side="right") - 1
    valid_src = src >= 0
    ma20d = np.full((nt, ns), np.nan)
    ma60d = np.full((nt, ns), np.nan)
    ma20d[valid_src] = m20[src[valid_src]]
    ma60d[valid_src] = m60[src[valid_src]]
    # 锚点 V1b:所用周收盘日 <= 观察日
    bad = int((wpos[src[valid_src]] > np.arange(nt)[valid_src]).sum())
    print(f"锚点V1b 周线映射无前视 违例 {bad} {'✓' if bad == 0 else '✗ 作废'}",
          flush=True)
    assert bad == 0

    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy()
    rows = []
    for t in me:
        t = int(t)
        if t < 60 or t > nt - min(HORS) - 1:
            continue
        e = np.flatnonzero(ok[t] & np.isfinite(ma20d[t]) & np.isfinite(ma60d[t])
                           & np.isfinite(mv[t]) & (ind[t] >= 0))
        for j in e:
            rows.append((t, int(j), bool(ma20d[t, j] > ma60d[t, j])))
    p = pd.DataFrame(rows, columns=["t", "j", "A"])
    na, nb = int(p.A.sum()), int((~p.A).sum())
    assert na + nb == len(p), "锚点V1d"
    print(f"锚点V1d ✓ A {na:,} + B {nb:,} = {len(p):,}", flush=True)

    tv, jv = p.t.to_numpy(), p.j.to_numpy()
    pre = {}
    for t in np.unique(tv):
        e = np.flatnonzero(ok[t] & np.isfinite(mv[t]) & (ind[t] >= 0))
        o = e[np.argsort(mv[t, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        pre[t] = (o, rk)
    ch, off, lens, keep = [], np.zeros(len(p), np.int64), np.zeros(len(p), np.int64), \
        np.ones(len(p), bool)
    pos = 0
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
    print(f"锚点V1c 行业违例 {v} 次 {'✓' if v == 0 else '✗ 作废'}", flush=True)
    assert v == 0

    print(f"\n{'='*100}\n买入并持有的胜率(A=多头排列 ma20d周>ma60d周;B=空头排列)\n{'='*100}")
    print(f"{'前瞻':<8}{'组':<4}{'样本':>9}{'胜率':>9}{'中位收益':>10}{'平均收益':>10}"
          f"{'对照胜率':>10}{'胜率差':>9}")
    res = []
    for hor in HORS:
        okh = tv <= nt - hor - 1
        with np.errstate(all="ignore"):
            fr = cl[np.minimum(tv + hor, nt - 1), jv] / cl[tv, jv] - 1.0
        for lab, sel in (("A", p.A.to_numpy() & okh), ("B", (~p.A.to_numpy()) & okh)):
            g = np.flatnonzero(sel & np.isfinite(fr))
            wr = float((fr[g] > 0).mean())
            q = pk[:, g]
            gg = q >= 0
            tq = tv[g][None, :]
            with np.errstate(all="ignore"):
                cfr = cl[np.minimum(tq + hor, nt - 1), np.maximum(q, 0)] \
                    / cl[tq, np.maximum(q, 0)] - 1.0
            cw = np.where(gg & np.isfinite(cfr), cfr > 0, np.nan)
            cwr = float(np.nanmedian(np.nanmean(cw, axis=1)))
            print(f"{hor:<8}{lab:<4}{len(g):>9,}{wr:>9.2%}{np.median(fr[g]):>10.2%}"
                  f"{np.mean(fr[g]):>10.2%}{cwr:>10.2%}{(wr-cwr)*100:>+8.2f}pp")
            res.append({"前瞻": hor, "组": lab, "样本": len(g), "胜率": wr,
                        "中位收益": float(np.median(fr[g])),
                        "平均收益": float(np.mean(fr[g])),
                        "对照胜率": cwr, "胜率差pp": (wr - cwr) * 100})
    df = pd.DataFrame(res)
    print(f"\n{'='*100}\nA 减 B(同前瞻期直接相减)\n{'='*100}")
    for hor in HORS:
        a = df[(df.前瞻 == hor) & (df.组 == "A")].iloc[0]
        b = df[(df.前瞻 == hor) & (df.组 == "B")].iloc[0]
        print(f"  {hor:>3}日:绝对胜率 A−B {(a.胜率-b.胜率)*100:+6.2f}pp  |  "
              f"扣对照后 A−B {a['胜率差pp']-b['胜率差pp']:+6.2f}pp  |  "
              f"中位收益 A−B {(a.中位收益-b.中位收益)*100:+6.2f}pp")
    df.to_csv(f"{OUT}/weekly_ma_winrate.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/weekly_ma_winrate.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
