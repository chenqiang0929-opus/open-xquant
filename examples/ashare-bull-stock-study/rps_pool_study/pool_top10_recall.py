"""§146:观察池「锁定 10%」—— 按召回率而不是命中率评价筛选器。

用户的目标(与前几节完全不同,必须分清)
--------------------------------------
「我的目标是 663 只股票,如果我每年能锁定 10% 的股票池 50-100 只左右,
我重点观察不就行了」;并举例「陶博士 2003-2007 观察 1000 多只股票,
RPS250 大于 95 的股票,大概 50 只左右,持续跟踪」。

**这是「缩小观察范围」,不是「预测翻倍」。** 对应的指标是
**召回率 = 未来牛股落在名单里的比例**,不是命中率。
第一四三/一四五节测的是命中率和 lift,回答的是另一个问题。

评价方式
--------
每个筛选器都**卡到池内约 10%**(取池内分位前 10%),然后报:
  选中数、**召回率**(覆盖了池内多少比例的当年翻倍股)、
  **lift = 召回率 ÷ 选中占比**(随机选 10% 的期望召回率就是 10%,lift=1)。
基线:**随机抽同样只数,500 次**,报召回率中位与 95 分位;
筛选器必须超过随机的 95 分位才算有用。

候选筛选器
----------
RPS250(全市场分位) / RPS60(全市场分位) / 池内 120日收益 / 池内距一年低点 /
池内 MA20持续度 / 池内 20日波动率 / 池内流通市值最小 / 池内换手最低 /
以及用户举的**陶博士口径:RPS250 > 95(全市场)**,不卡 10%,报实际只数。

数据边界
--------
可验证时点只有 **2024 与 2025**(2023 时点池内合格为 0);
**2024 年池内仅 2 只翻倍,统计上没有意义,只列不判**;主看 2025。
**只有一年有效样本,任何结论都必须带这个前提。**

**本节不设通过/不通过判据。** 不新增顶层目录;不 force push;不作可交易性声明。
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
from startup_threshold_scan import load_labels  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
XLS = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
       "f48a5b4d-___20260827.xls")
NRAND = 500


def main():  # noqa: PLR0915
    t0 = time.time()
    px = pd.read_excel(XLS, dtype=str)
    px = px.rename(columns={px.columns[1]: "名称"})
    px["代码"] = px["代码"].str.zfill(6)
    pool = dict(zip(px.代码, px.名称, strict=True))
    indm = dict(zip(px.代码, px["一二级行业"], strict=True))

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
    turn = al("turnover").rolling(20, min_periods=10).mean().to_numpy()
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).to_numpy(np.float64)
    ok &= np.isfinite(cl)
    dfc = pd.DataFrame(cl)
    lo250 = dfc.rolling(250, min_periods=250).min().to_numpy()
    ma20 = dfc.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        r120 = cl / np.roll(cl, 120, axis=0) - 1.0
        r120[:120] = np.nan
        r60 = cl / np.roll(cl, 60, axis=0) - 1.0
        r60[:60] = np.nan
        r250 = cl / np.roll(cl, 250, axis=0) - 1.0
        r250[:250] = np.nan
        lr = np.log(cl / np.roll(cl, 1, axis=0))
        lr[0] = np.nan
    ab = pd.DataFrame(cl > ma20).rolling(120, min_periods=120).mean().to_numpy()
    v20 = pd.DataFrame(lr).rolling(20, min_periods=20).std().to_numpy()
    rps60 = pd.DataFrame(np.where(ok, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    rps250 = pd.DataFrame(np.where(ok, r250, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    print(f"面板就绪 ({time.time()-t0:.0f}s)", flush=True)

    scr = {"RPS250(全市场分位,高)": rps250, "RPS60(全市场分位,高)": rps60,
           "120日收益(高)": r120, "距一年低点(高)": rec,
           "MA20持续度(高)": ab, "20日波动率(高)": v20,
           "流通市值(小)": -mv, "20日均换手(低)": -turn}
    l1s, _ = load_labels()
    cp = {c: j for j, c in enumerate(cldf.columns)}
    ip = pd.Index(idx)
    rows = []
    for ty, ds in ((2024, "2023-12-29"), (2025, "2024-12-31"),
                   (None, "2026-08-03")):
        t = int(ip.get_indexer([pd.Timestamp(ds)], method="ffill")[0])
        elig = [c for c in pool if c in cp and ok[t, cp[c]]]
        js = np.array([cp[c] for c in elig])
        n10 = max(int(round(len(elig) * 0.10)), 1)
        print(f"\n{'='*100}")
        if ty:
            y = np.array([(ty, c) in l1s for c in elig])
            nb = int(y.sum())
            print(f"{ty} 年:池内合格 {len(elig)},翻倍 {nb} 只"
                  f"(池内基准 {y.mean():.2%});锁定 10% = {n10} 只")
            if nb < 10:
                print("  **翻倍股不足 10 只,召回率统计无意义,只列不判**")
        else:
            y = None
            print(f"2026-08-03(当前,结果未知):池内合格 {len(elig)};"
                  f"锁定 10% = {n10} 只")
        print("=" * 100)
        if ty and y.sum() >= 10:
            rg = np.random.default_rng(20260827)
            rr = [y[rg.choice(len(elig), n10, replace=False)].sum() / y.sum()
                  for _ in range(NRAND)]
            print(f"  随机抽 {n10} 只的召回率:中位 {np.median(rr):.1%}  "
                  f"**95分位 {np.percentile(rr,95):.1%}**")
            hi95 = float(np.percentile(rr, 95))
        else:
            hi95 = np.nan
        if ty:
            print(f"\n{'筛选器':<24}{'选中':>6}{'覆盖牛股':>9}{'召回率':>9}"
                  f"{'lift':>7}{'超随机95分位':>12}")
        for nm, mat in scr.items():
            v = mat[t, js]
            gd = np.isfinite(v)
            if gd.sum() < n10:
                continue
            order = np.argsort(-np.where(gd, v, -np.inf), kind="stable")[:n10]
            if ty:
                rc = y[order].sum() / max(y.sum(), 1)
                lf = rc / (n10 / len(elig))
                mark = "✓" if (np.isfinite(hi95) and rc > hi95) else ""
                print(f"{nm:<24}{n10:>6}{int(y[order].sum()):>9}{rc:>9.1%}"
                      f"{lf:>7.2f}{mark:>12}")
                rows.append({"年": ty, "筛选器": nm, "选中": n10,
                             "覆盖牛股": int(y[order].sum()), "召回率": float(rc),
                             "lift": float(lf), "随机95分位": hi95})
            else:
                sel = [elig[i] for i in order]
                rows += [{"年": "当前", "筛选器": nm, "代码": c, "名称": pool[c],
                          "行业": indm.get(c)} for c in sel]
        if ty:
            # 陶博士口径:RPS250 > 95(全市场),不卡 10%
            m = np.isfinite(rps250[t, js]) & (rps250[t, js] > 95)
            if m.sum() >= 3:
                rc = y[m].sum() / max(y.sum(), 1)
                print(f"{'【陶博士】RPS250>95':<24}{int(m.sum()):>6}"
                      f"{int(y[m].sum()):>9}{rc:>9.1%}"
                      f"{rc/(m.sum()/len(elig)):>7.2f}")
                rows.append({"年": ty, "筛选器": "【陶博士】RPS250>95",
                             "选中": int(m.sum()), "覆盖牛股": int(y[m].sum()),
                             "召回率": float(rc),
                             "lift": float(rc / (m.sum() / len(elig)))})
        else:
            t2 = t
            for nm, mat in (("RPS250(全市场分位,高)", rps250),):
                v = mat[t2, js]
                gd = np.isfinite(v)
                order = np.argsort(-np.where(gd, v, -np.inf), kind="stable")[:n10]
                print(f"\n  **当前 RPS250 前 {n10} 只**(陶博士口径最接近的一档):")
                for i in order[:n10]:
                    c = elig[i]
                    print(f"    {c} {pool[c]:<10}{str(indm.get(c,''))[:16]:<18}"
                          f"RPS250 {rps250[t2,cp[c]]:5.1f}  "
                          f"RPS60 {rps60[t2,cp[c]]:5.1f}  "
                          f"距低点 {rec[t2,cp[c]]:+6.0%}")
    pd.DataFrame(rows).to_csv(f"{OUT}/pool_top10_recall.csv", index=False,
                              encoding="utf-8-sig")
    print(f"\n落库 {OUT}/pool_top10_recall.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
