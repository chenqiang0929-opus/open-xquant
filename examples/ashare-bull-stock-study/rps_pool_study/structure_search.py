"""§151 事前登记 + 实现:换结构能不能抓住"穿窗口"的大牛股。

起因
----
用户:「换结构,能做到吗」
第一五〇节查明:现有规则的门槛(距低点前30%)与上限(≤100%)之间的窗口,
真正的大牛股常常一个月就穿过 —— 中际旭创 2023-02 换手加速分位 **0.96**
(量能已极强),但距低点分位仅 0.57 不达门槛;次月距低点 132% 又超上限。
**所以要换的是结构,不是阈值。**

候选结构(跑之前写死,不增不减)
--------------------------------
S1 **纯量能突变**:换手加速 ∈ 全市场前 **5%**,**不设任何位置门槛**
S2 **纯量能突变(宽)**:换手加速 ∈ 前 **10%**,不设位置门槛
S3 **量能极强 + 位置中位**:换手加速前 10% 且 距低点分位 ∈ [0.30, 0.70]
   —— **这正是中际旭创 2023-02 的状态**
S4 **成交额突变**:成交额加速 ∈ 前 5%,不设位置门槛
S5 **量价齐升**:换手加速前 20% 且 20 日收益 ∈ 全市场前 10%
S6 **现有规则**(距低点前30% & 换手加速前30% & 距低点≤100%,降序前100)—— 对照基线

判据(跑之前写死,跑完照判,不放宽)
----------------------------------
K1 锚点:面板 (3297, 5232);价格 ffill(第一五〇节修过的 bug);无前视。
K2 **两条同时满足才算「换结构成功」**:
   (a) **留出段(2023-01–2026-04)lift ≥ 1.40** —— 不低于现有规则的水平;
   (b) **案例召回 ≥ 50%** —— 定义:四只案例股(中际旭创 300308、胜宏科技 300476、
       双林股份 300100、浙江荣泰 603119)在其「未来 60 日涨 ≥50%」的那些月份里,
       有多少比例被该结构选中。**现有规则在这一项上接近 0,这正是要解决的问题。**
K3 每月只数须 ≤ 300(否则不具备"观察池"的实用性),描述项不参与判定。

**不做的**:不因为哪个结构好看就回头改判据;不新增顶层目录;不 force push;
**若 6 个结构全部不过,就如实写「换结构在本节候选内没做到」,不再试第 7 个。**
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

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
HOR, THR = 60, 0.50
CASES = {"300308": "中际旭创", "300476": "胜宏科技",
         "300100": "双林股份", "603119": "浙江荣泰"}


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "turnover", "amount", "volume", "is_st", "is_suspended",
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
    assert (nt, ns) == (3297, 5232), f"锚点K1 {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f)
    trn, amt = al("turnover"), al("amount")
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    ok &= np.isfinite(cl)
    dfc = pd.DataFrame(cl)
    lo250 = dfc.rolling(250, min_periods=250).min().to_numpy()
    t20 = trn.rolling(20, min_periods=10).mean().to_numpy()
    t60 = trn.rolling(60, min_periods=30).mean().to_numpy()
    a20 = amt.rolling(20, min_periods=10).mean().to_numpy()
    a60 = amt.rolling(60, min_periods=30).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        tacc = t20 / np.where(t60 > 0, t60, np.nan) - 1.0
        aacc = a20 / np.where(a60 > 0, a60, np.nan) - 1.0
        r20 = cl / np.roll(cl, 20, axis=0) - 1.0
        r20[:20] = np.nan
    fmax = pd.DataFrame(cl[::-1]).rolling(HOR, min_periods=1).max().to_numpy()[::-1]
    fwd = np.full_like(cl, np.nan)
    fwd[:-1] = fmax[1:]
    with np.errstate(all="ignore"):
        up = fwd / np.where(cl > 0, cl, np.nan) - 1.0
    print(f"锚点K1 ✓ {cldf.shape};ffill 已修 ({time.time()-t0:.0f}s)", flush=True)

    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy()
    cp = {c: j for j, c in enumerate(cldf.columns)}
    cj = {cp[c] for c in CASES if c in cp}
    snap = {}
    for t in me:
        t = int(t)
        if t > nt - HOR - 1 or idx[t] < pd.Timestamp("2019-01-01"):
            continue
        m = (ok[t] & np.isfinite(rec[t]) & np.isfinite(tacc[t])
             & np.isfinite(aacc[t]) & np.isfinite(r20[t]) & np.isfinite(up[t]))
        e = np.flatnonzero(m)
        if len(e) < 100:
            continue
        snap[t] = (e, pd.Series(rec[t, e]).rank(pct=True).to_numpy(),
                   pd.Series(tacc[t, e]).rank(pct=True).to_numpy(),
                   pd.Series(aacc[t, e]).rank(pct=True).to_numpy(),
                   pd.Series(r20[t, e]).rank(pct=True).to_numpy(),
                   rec[t, e], up[t, e] >= THR)
    print(f"月度截面 {len(snap)} 个", flush=True)

    def s1(qr, qt, qa, q2, rv):
        return qt >= 0.95

    def s2(qr, qt, qa, q2, rv):
        return qt >= 0.90

    def s3(qr, qt, qa, q2, rv):
        return (qt >= 0.90) & (qr >= 0.30) & (qr <= 0.70)

    def s4(qr, qt, qa, q2, rv):
        return qa >= 0.95

    def s5(qr, qt, qa, q2, rv):
        return (qt >= 0.80) & (q2 >= 0.90)

    def s6(qr, qt, qa, q2, rv):
        m = (qr >= 0.70) & (qt >= 0.70) & (rv <= 1.00)
        out = np.zeros(len(qr), bool)
        ii = np.flatnonzero(m)
        if len(ii):
            out[ii[np.argsort(-rv[ii], kind="stable")[:100]]] = True
        return out

    structs = {"S1 纯量能突变(换手加速前5%)": s1,
               "S2 纯量能突变(前10%)": s2,
               "S3 量能前10% + 位置中位[0.3,0.7]": s3,
               "S4 成交额加速前5%": s4,
               "S5 换手加速前20% + 20日收益前10%": s5,
               "S6 现有规则(对照基线)": s6}

    print(f"\n{'='*104}")
    print(f"{'结构':<34}{'留出/月':>8}{'启动率':>8}{'lift':>7}{'训练lift':>9}"
          f"{'案例召回':>9}{'K2':>5}")
    print("=" * 104)
    rows = []
    for nm, fn in structs.items():
        res = {}
        cse_hit = cse_tot = 0
        for seg, lo, hi in (("train", 2019, 2022), ("hold", 2023, 2026)):
            tot = hit = nmo = bn = bh = 0
            for t, (e, qr, qt, qa, q2, rv, y) in snap.items():
                if not (lo <= idx[t].year <= hi):
                    continue
                s = fn(qr, qt, qa, q2, rv)
                tot += int(s.sum())
                hit += int(y[s].sum())
                nmo += 1
                bn += len(e)
                bh += int(y.sum())
                if seg == "hold" or True:
                    for k, j in enumerate(e):
                        if j in cj and y[k]:
                            cse_tot += 1
                            cse_hit += int(s[k])
            b = bh / bn
            res[seg] = (tot / max(nmo, 1), hit / max(tot, 1),
                        (hit / max(tot, 1)) / b)
        rc = cse_hit / max(cse_tot, 1)
        h = res["hold"]
        k2 = (h[2] >= 1.40) and (rc >= 0.50)
        print(f"{nm:<34}{h[0]:>8.0f}{h[1]:>8.2%}{h[2]:>7.2f}"
              f"{res['train'][2]:>9.2f}{rc:>9.0%}{'✓' if k2 else '✗':>5}")
        rows.append({"结构": nm, "留出每月": h[0], "留出启动率": h[1],
                     "留出lift": h[2], "训练lift": res["train"][2],
                     "案例召回": rc, "案例命中": cse_hit, "案例机会": cse_tot,
                     "K2": k2})
    df = pd.DataFrame(rows)
    print(f"\n案例机会总数(四只股票未来60日涨>=50%的月份数)= {rows[0]['案例机会']}")
    npass = int(df.K2.sum())
    print(f"\n**K2 通过 {npass}/6**:"
          f"{', '.join(df.loc[df.K2, '结构']) if npass else '**无**'}")
    print("  判据:留出 lift ≥1.40 且 案例召回 ≥50%,两条同时满足")
    df.to_csv(f"{OUT}/structure_search.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/structure_search.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
