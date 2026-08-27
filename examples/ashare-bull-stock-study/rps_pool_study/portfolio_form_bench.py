"""§152 附:组合口径的机器校验 —— 对照年化 +14.54% 是不是算错了。

起因
----
第一五二节的对照组在留出段跑出 **年化 +14.54%**、2025 年 **+53.81%**,
量级偏大。**在把「策略跑输对照」写进正文之前,必须先证明这套连乘机器本身没错。**

做法(不引入任何新判据,纯校验)
------------------------------
用**同一套**月末调仓、等权买入并持有、ffill 参与的机器,算两条参照线:
   (a) **全市场等权**:每月末买入当日全部合格股票(与选股同一合格池),持有到下月末;
   (b) **510300(沪深300 ETF)**:同样的月末到月末口径。
把 (a)(b) 与第一五二节的对照中位数并排逐年打印。

判据
----
**本脚本不设通过/不通过** —— 它只回答「机器对不对」:
若 (a) 的逐年数字与「A股等权/小盘的公认年景」大方向一致(2023 亏、2024 涨、2025 大涨),
且 (b) 与沪深300 的公认年度涨跌大方向一致,则第一五二节的连乘机器可信。
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_replication import DATA  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")


def main():
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
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    ok &= np.isfinite(cl)

    hs = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"]
    if getattr(hs.index, "tz", None) is not None:
        hs.index = hs.index.tz_localize(None)
    hs = hs.reindex(idx).ffill().to_numpy(np.float64)

    me = np.sort(pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy().astype(int))
    ent, few, fhs = [], [], []
    for a, b in zip(me[:-1], me[1:], strict=True):
        a, b = int(a), int(b)
        if idx[a] < pd.Timestamp("2019-01-01") or idx[a] > pd.Timestamp("2026-04-30"):
            continue
        e = np.flatnonzero(ok[a])
        if len(e) < 100:
            continue
        ent.append(idx[a])
        few.append(float(np.mean(cl[b, e] / cl[a, e])))
        fhs.append(float(hs[b] / hs[a]))
    ent, few, fhs = np.array(ent), np.array(few), np.array(fhs)

    prev = pd.read_csv(f"{OUT}/portfolio_form.csv")
    pm = {str(r["段"]): r for _, r in prev.iterrows()}
    w = 88
    print(f"{'='*w}\n机器校验:同一套月末到月末的连乘机器,三条参照线逐年\n{'='*w}")
    print("  注:'年'按**调仓日**标注,故 2023 实际覆盖 2023-01-31 → 2024-01-31,"
          "与自然年报表差一个月。")
    print(f"{'年':<7}{'区间':>25}{'全市场等权':>12}{'510300':>10}"
          f"{'§152对照中位':>14}{'§152组合':>11}")
    rows = []
    for y in range(2019, 2027):
        m = np.array([e.year == y for e in ent])
        if m.sum() < 6:
            continue
        a = float(np.prod(few[m]) - 1)
        h = float(np.prod(fhs[m]) - 1)
        r = pm.get(str(y))
        c = float(r["对照年化中位"]) if r is not None else np.nan
        s = float(r["零成本年化"]) if r is not None else np.nan
        k = np.flatnonzero(m)
        span = f"{ent[k[0]].date()}→{ent[k[-1]].date()}+1M"
        print(f"{y:<7}{span:>25}{a:>12.2%}{h:>10.2%}{c:>14.2%}{s:>11.2%}")
        rows.append({"年": y, "区间": span, "全市场等权": a, "510300": h,
                     "§152对照中位": c, "§152组合": s})
    print(f"\n{'='*w}\n分段年化(同一机器,零成本口径)\n{'='*w}")
    for tag, lo, hi in (("训练段 2019-2022", "2019-01-01", "2022-12-31"),
                        ("留出段 2023-01–2026-04", "2023-01-01", "2026-04-30")):
        m = (ent >= pd.Timestamp(lo)) & (ent <= pd.Timestamp(hi))
        nd = 966 if lo.startswith("2019") else 806
        aa = float(np.prod(few[m]) ** (250 / nd) - 1)
        hh = float(np.prod(fhs[m]) ** (250 / nd) - 1)
        key = [k for k in pm if tag[:3] in k]
        r = pm[key[0]] if key else None
        print(f"  {tag}:全市场等权 {aa:+.2%}   510300 {hh:+.2%}")
        rows.append({"年": tag, "区间": f"{lo}→{hi}", "全市场等权": aa,
                     "510300": hh,
                     "§152对照中位": float(r["对照年化中位"]) if r is not None else np.nan,
                     "§152组合": float(r["零成本年化"]) if r is not None else np.nan})
    pd.DataFrame(rows).to_csv(f"{OUT}/portfolio_form_bench.csv", index=False,
                              encoding="utf-8-sig")
    print(f"\n落库 {OUT}/portfolio_form_bench.csv")


if __name__ == "__main__":
    main()
