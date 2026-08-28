"""§143:把冻结的 X01 三档规则应用到用户给的次新股池,三个时点出名单。

用户需求
--------
「如果我给你股票池,最早上市时间是 2022 年 6 月,那么时间回到 2023 年 1 月
或者 2024 年 1 月或者 2025 年 1 月,你可以给我做一个股票池启动的信号吗」

数据边界(先说清楚,不能含糊)
------------------------------
池子 663 只(2026-08-27 导出,上市 2022-06-08 → 2026-08-25),
其中 **618 只在本面板**(缺 45 只:39 只 2026 年上市、面板末日 2026-08-03 前无足够数据)。
- **2023-01-01 时点做不了**:池内最早上市 2022-06-08,到 2023-01 仅 7 个月,
  **上市满 250 日的合格股票数为 0**。这一条不绕过、不放宽合格条件。
- 2024-01 时点合格 206 只;2025-01 时点合格 441 只。
- 另加 **2026-08-03(面板末日)** 一个当前时点 —— **只有它是真正前瞻的**。

规则:第一四二节冻结的 Codex 三档,**一个数没改**
------------------------------------------------
观察档 = 距一年低点 ≥ 40% 且 120 日收益 ≥ 10%
标准启动档 = 观察档 + 近120日站上MA20比例 ≥ 55% + RPS60 ≥ 80
强确认档 = 观察档 + 近120日站上MA20比例 ≥ 55% + RPS60 ≥ 90
**RPS60 用全市场 5,232 只横截面分位算,不是池内分位。**

必须同时报的两个基准(否则会误读)
----------------------------------
- **池内基准**:该时点池内全部合格股票的下一年翻倍率;
- **全市场基准**:同期全市场的翻倍率。
次新股池的基准与全市场不同,**不报池内基准就无法判断三档在这个池子里有没有增量**。

性质
----
**2024-01 与 2025-01 是事后验证(结果已知),不是预测。**
2026-08-03 那份是前瞻名单,**结果未知,不构成买入建议**。
本节不设通过/不通过判据,是名单交付 + 后验描述。

不做的
------
不改三档阈值;不因为某个时点效果差就换规则;不新增顶层目录;不 force push;
**不基于本节名单做任何可交易性声明**。
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


def main():  # noqa: PLR0915
    t0 = time.time()
    px = pd.read_excel(XLS, dtype=str)
    px = px.rename(columns={px.columns[1]: "名称"})
    px["代码"] = px["代码"].str.zfill(6)
    px["ld"] = pd.to_datetime(px["上市日期"], format="%Y%m%d", errors="coerce")
    pool = dict(zip(px.代码, px.名称, strict=True))
    ldmap = dict(zip(px.代码, px.ld, strict=True))
    indmap = dict(zip(px.代码, px["一二级行业"], strict=True))

    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "volume", "is_st", "is_suspended", "listed_days"]
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
    ab = pd.DataFrame(cl > ma20).rolling(120, min_periods=120).mean().to_numpy()
    rps60 = pd.DataFrame(np.where(ok, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    print(f"面板 {cldf.shape} 就绪 ({time.time()-t0:.0f}s)", flush=True)

    l1s, _ = load_labels()
    cp = {c: j for j, c in enumerate(cldf.columns)}
    ip = pd.Index(idx)
    rows = []
    for tag, ds, ty in (("2023-01", "2022-12-30", 2023),
                        ("2024-01", "2023-12-29", 2024),
                        ("2025-01", "2024-12-31", 2025),
                        ("2026-08-03(当前)", "2026-08-03", None)):
        t = int(ip.get_indexer([pd.Timestamp(ds)], method="ffill")[0])
        elig = [c for c in pool if c in cp and ok[t, cp[c]]
                and np.isfinite(rec[t, cp[c]]) and np.isfinite(r120[t, cp[c]])
                and np.isfinite(ab[t, cp[c]]) and np.isfinite(rps60[t, cp[c]])]
        print(f"\n{'='*96}\n时点 {tag}(观察日 {idx[t].date()}):"
              f"池内合格 {len(elig)} 只 / 池 {len(pool)} 只")
        if not elig:
            print("  **合格数为 0,该时点无法出信号**"
                  "(池内最早上市 2022-06-08,不满上市 250 日条件)")
            print("=" * 96)
            continue
        mkt = np.flatnonzero(ok[t] & np.isfinite(rec[t]) & np.isfinite(r120[t])
                             & np.isfinite(ab[t]) & np.isfinite(rps60[t]))
        if ty:
            pb = np.mean([(ty, c) in l1s for c in elig])
            mb = np.mean([(ty, cldf.columns[j]) in l1s for j in mkt])
            print(f"  {ty} 年翻倍率:**池内基准 {pb:.2%}**;全市场基准 {mb:.2%}"
                  f"(全市场合格 {len(mkt):,})")
        print("=" * 96)
        sel = {}
        for c in elig:
            j = cp[c]
            base = rec[t, j] >= 0.40 and r120[t, j] >= 0.10
            std = base and ab[t, j] >= 0.55 and rps60[t, j] >= 80
            strong = base and ab[t, j] >= 0.55 and rps60[t, j] >= 90
            if base:
                sel[c] = ("强确认档" if strong else
                          "标准启动档" if std else "观察档")
        for tier in ("观察档", "标准启动档", "强确认档"):
            g = [c for c in sel if (sel[c] == tier or
                                    (tier == "观察档") or
                                    (tier == "标准启动档" and sel[c] == "强确认档"))]
            if tier == "观察档":
                g = list(sel)
            elif tier == "标准启动档":
                g = [c for c in sel if sel[c] in ("标准启动档", "强确认档")]
            else:
                g = [c for c in sel if sel[c] == "强确认档"]
            if not g:
                print(f"  {tier}:0 只")
                continue
            line = f"  **{tier}** {len(g)} 只(占合格 {len(g)/len(elig):.1%})"
            if ty:
                hit = [c for c in g if (ty, c) in l1s]
                rets = []
                for c in g:
                    j = cp[c]
                    te = int(ip.get_indexer([pd.Timestamp(f"{ty}-12-31")],
                                            method="ffill")[0])
                    rets.append(cl[te, j] / cl[t, j] - 1.0)
                line += (f" | {ty} 翻倍 {len(hit)} 只 = **{len(hit)/len(g):.2%}**"
                         f" | 中位涨幅 {np.nanmedian(rets):+.1%}")
            print(line)
            if tier == "强确认档" or (tier == "观察档" and not ty):
                for c in sorted(g, key=lambda z: -rps60[t, cp[z]])[:15]:
                    j = cp[c]
                    ex = ""
                    if ty:
                        te = int(ip.get_indexer([pd.Timestamp(f"{ty}-12-31")],
                                                method="ffill")[0])
                        ex = (f"  {ty}实际 {cl[te,j]/cl[t,j]-1:+7.1%}"
                              f" {'**翻倍**' if (ty,c) in l1s else ''}")
                    print(f"      {c} {pool[c]:<10}{str(indmap.get(c,''))[:14]:<16}"
                          f"距低点{rec[t,j]:+6.0%} 120日{r120[t,j]:+6.0%} "
                          f"MA20持续{ab[t,j]:4.0%} RPS60 {rps60[t,j]:4.0f}{ex}")
            for c in g:
                j = cp[c]
                rows.append({"时点": tag, "观察日": idx[t].date(), "档": tier,
                             "代码": c, "名称": pool[c],
                             "行业": indmap.get(c), "上市日": ldmap.get(c),
                             "距低点涨幅": rec[t, j], "120日收益": r120[t, j],
                             "MA20持续度": ab[t, j], "RPS60": rps60[t, j],
                             "目标年": ty,
                             "翻倍": ((ty, c) in l1s) if ty else None})
    pd.DataFrame(rows).to_csv(f"{OUT}/pool_tier_signal.csv", index=False,
                              encoding="utf-8-sig")
    print(f"\n落库 {OUT}/pool_tier_signal.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
