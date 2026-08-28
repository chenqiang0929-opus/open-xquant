"""§154 附:8% 止损为什么会把组合打崩 —— 用数字确认,不猜。

**本脚本不设通过/不通过判据**,只回答一个问题:
被选中的股票在买入后 21 个交易日内,**触及 −8% 回撤的概率有多大**,
其中**最终仍上涨**的又占多少 —— 即「被噪音扫出局」的比例。

口径与第一五四节一致(留出段 2023-01–2026-04,月末观察点,ffill 参与)。
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
STOP, FWD = 0.08, 21


def main():
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "turnover", "volume", "is_st", "is_suspended", "listed_days"]
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
    assert (nt, ns) == (3297, 5232)

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f)
    trn = al("turnover")
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0) & np.isfinite(cl))
    px = pd.DataFrame(cl)
    lo250 = px.rolling(250, min_periods=250).min().to_numpy()
    ma20 = px.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        mfrac = pd.DataFrame((cl > ma20).astype(np.float64)).where(
            np.isfinite(ma20)).rolling(120, min_periods=120).mean().to_numpy()
        r120 = px.pct_change(120).to_numpy()
        r60 = px.pct_change(60).to_numpy()
        dr = px.pct_change(1).to_numpy()
        vol20 = pd.DataFrame(dr).rolling(20, min_periods=20).std().to_numpy()
        tacc = (trn.rolling(20, min_periods=10).mean().to_numpy()
                / np.where(trn.rolling(60, min_periods=30).mean().to_numpy() > 0,
                           trn.rolling(60, min_periods=30).mean().to_numpy(),
                           np.nan) - 1.0)
    trad = (~al("is_suspended", True).astype(bool).to_numpy()
            & (al("volume", 0).to_numpy() > 0) & np.isfinite(r60))
    rps60 = pd.DataFrame(np.where(trad, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100.0
    me = np.sort(pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy().astype(int))

    rows = []
    for t in me:
        t = int(t)
        if not (pd.Timestamp("2023-01-01") <= idx[t] <= pd.Timestamp("2026-04-30")):
            continue
        if t + FWD >= nt:
            continue
        e = np.flatnonzero(ok[t])
        cs = e[(rec[t, e] >= 0.40) & (r120[t, e] >= 0.10) & (mfrac[t, e] >= 0.55)
               & (rps60[t, e] >= 90) & np.isfinite(rec[t, e])]
        v = np.isfinite(rec[t, e]) & np.isfinite(tacc[t, e])
        qr = pd.Series(np.where(v, rec[t, e], np.nan)).rank(pct=True).to_numpy()
        qt = pd.Series(np.where(v, tacc[t, e], np.nan)).rank(pct=True).to_numpy()
        r1 = e[v & (qr >= 0.70) & (qt >= 0.70)]
        top = cs[np.argsort(-rps60[t, cs], kind="stable")][:10]
        for nm, js in (("Codex强确认·全部", cs), ("Codex强确认·RPS60前10", top),
                       ("第一四八节规则·全部", r1), ("全市场合格(参照)", e)):
            if not len(js):
                continue
            path = cl[t + 1:t + 1 + FWD, js] / cl[t, js]
            hit = (path.min(axis=0) <= 1 - STOP)
            end = path[-1] - 1.0
            rows.append({"组": nm, "日期": idx[t].date(), "只数": len(js),
                         "触及-8%比例": float(hit.mean()),
                         "21日后仍上涨比例": float((end > 0).mean()),
                         "触发止损但21日后仍上涨": float(
                             (hit & (end > 0)).sum() / max(hit.sum(), 1)),
                         "20日波动率中位": float(np.nanmedian(vol20[t, js])),
                         "21日收益中位": float(np.median(end))})
    df = pd.DataFrame(rows)
    g = df.groupby("组").apply(
        lambda x: pd.Series({
            "月数": len(x), "月均只数": x["只数"].mean(),
            "触及-8%比例": np.average(x["触及-8%比例"], weights=x["只数"]),
            "其中21日后仍上涨": np.average(x["触发止损但21日后仍上涨"],
                                    weights=x["只数"]),
            "20日日波动率中位": x["20日波动率中位"].median(),
            "21日收益中位": np.average(x["21日收益中位"], weights=x["只数"])}),
        include_groups=False)
    w = 96
    print(f"{'='*w}\n买入后 21 个交易日内触及 −8% 的概率(留出段 2023-01–2026-04)\n{'='*w}")
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n注:「其中21日后仍上涨」= 触发了 8% 止损、但 21 日后价格仍高于买入价的比例,\n"
          "    即**被噪音扫出局**的比例。日波动率是 20 日日收益标准差。")
    g.to_csv(f"{OUT}/stop_whipsaw_diag.csv", encoding="utf-8-sig")
    print(f"\n落库 {OUT}/stop_whipsaw_diag.csv")


if __name__ == "__main__":
    main()
