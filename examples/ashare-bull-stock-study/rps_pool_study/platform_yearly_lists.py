"""§156 名单交付:平台筛选器近 10 年逐年买点清单 + 框架实际成交流水。

**本脚本不做假设检验,不设通过/不通过判据** —— 与第八十七节同规格:
只出名单与后验描述。判定早已做过,不在这里重判:

    第六十一节:「三条全中」组合年化 +10.37%,但 300 次随机对照 **p = 0.16**;
    第一五五节:接上买点+止损+择时后年化 +15.83%、回撤 16.7%,
                但同市值同行业随机对照 +18.29%,**超额 −2.46pp、p = 0.656**。
    → **本名单是状态标记,不是买点推荐,不构成投资建议。**

口径(与第一五五节逐字一致,一个字不改)
--------------------------------------
- 尺子:**绝对阈值(legacy)** 缩量比<0.80、收敛比<0.80、深度≤0.352,
  调整天数≥15,且要求 20 周线向上
- 买点:当日三条全中 **且** 收盘 > 平台内(强势日→前一日)最高收盘
- 止损:平台下沿(平台内→前一日最低收盘);距买入价超 15% 则上移到 −15%
- 大盘过滤:全市场等权净值 < 自身 MA200 → 清仓且不新开仓
- 仓位:10 等权槽位,突破日先到先得,同日按收敛比升序;持有上限 120 交易日
- 价格 ffill 参与,退市股绝不剔除

两张产出
--------
(A) **逐年买点清单** `platform_buypoints_YYYY.csv` / Excel 每年一张表
    每个买点带:形态字段、止损价与止损σ倍数、大盘状态、
    以及 20/60/120/250 日后验与「是否触及止损」。
(B) **框架实际成交流水** `platform_trades.csv`
    10 槽位真正买入的每一笔:买入日、卖出日、卖出原因、单笔收益。
    **(A) 有两万多行,(B) 才是这套框架实际会买的东西 —— 先看 (B)。**
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codex_r10_replication import DATA  # noqa: E402
from consolidation_screener import THR_ATR, THR_DEPTH, THR_SHRINK, load_panel  # noqa: E402
from industry_neutral import build_industry  # noqa: E402
from platform_pivot import HOLD_MAX, MA_MKT, MAXPOS, STOP_CAP, vec_screen  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
CENSUS = ("/home/user/quant-research-dev/research/"
          "bull-stock-census-2010-2025/data/*.csv")
Y0 = 2016


def name_map():
    m = {}
    for f in glob.glob(CENSUS):
        try:
            x = pd.read_csv(f, dtype=str)
        except Exception:                                      # noqa: BLE001
            continue
        x.columns = [c.strip("﻿") for c in x.columns]
        if "code" in x.columns and "name" in x.columns:
            for c, n in zip(x["code"], x["name"], strict=True):
                if isinstance(c, str) and isinstance(n, str):
                    m[c.zfill(6)] = n
    try:
        with open(f"{OUT}/code_name_map.json", encoding="utf-8") as fh:
            m.update({k.zfill(6): v for k, v in json.load(fh).items()})
    except OSError:
        pass
    return m


def main():  # noqa: PLR0915
    t0 = time.time()
    cl_df, frames, strong, ma100 = load_panel(DATA)
    if "510300" in cl_df.columns:
        cl_df = cl_df.drop(columns=["510300"])
        strong = strong[:, [i for i, c in enumerate(ma100.columns) if c != "510300"]]
    idx, codes = cl_df.index, list(cl_df.columns)
    nt, ns = cl_df.shape
    assert (nt, ns) == (3297, 5232), f"锚点 {cl_df.shape}"
    ts_a, adj_a, dep, shr, cnv, hi, lo = vec_screen(
        cl_df.to_numpy(float), frames, strong, ma100, idx, codes)
    del frames
    d2 = {k: {} for k in ("float_mv", "is_st", "is_suspended", "listed_days",
                          "volume")}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=list(d2))
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in d2:
            d2[k][c] = x[k]

    def al(k, f=np.nan):
        return pd.DataFrame(d2[k]).sort_index().reindex(
            index=idx, columns=codes).fillna(f).to_numpy()
    cl = cl_df.where(cl_df > 0).ffill().to_numpy(np.float64)
    mv = al("float_mv") / 1e8
    ok = (~al("is_st", True).astype(bool) & ~al("is_suspended", True).astype(bool)
          & (al("listed_days", 0) >= 250) & (al("volume", 0) > 0) & np.isfinite(cl))
    ind, ind_names, _ = build_industry(codes, idx)
    vol20 = pd.DataFrame(cl).pct_change(1).rolling(20, min_periods=20).std().to_numpy()
    with np.errstate(all="ignore"):
        rr = cl[1:] / cl[:-1] - 1.0
    msk = ok[1:] & ok[:-1] & np.isfinite(rr)
    dd = np.zeros(nt)
    dd[1:] = np.where(msk.sum(1) > 0,
                      np.nan_to_num(rr * msk).sum(1) / np.maximum(msk.sum(1), 1), 0.0)
    nav = np.cumprod(1 + dd)
    mmv = pd.Series(nav).rolling(MA_MKT, min_periods=MA_MKT).mean().to_numpy()
    mkt_on = ~(np.isfinite(mmv) & (nav < mmv))
    del rr, msk
    hit3 = (shr < THR_SHRINK) & (cnv < THR_ATR) & (dep <= THR_DEPTH) & (adj_a >= 0)
    up_prev = np.full((nt, ns), np.nan, np.float64)
    lo_prev = np.full((nt, ns), np.nan, np.float64)
    same = np.zeros((nt, ns), bool)
    same[1:] = ts_a[1:] == ts_a[:-1]
    up_prev[1:] = np.where(same[1:], hi[:-1], np.nan)
    lo_prev[1:] = np.where(same[1:], lo[:-1], np.nan)
    brk = hit3 & ok & np.isfinite(up_prev) & (cl > up_prev) & np.isfinite(lo_prev) \
        & np.isfinite(vol20) & np.isfinite(mv)
    nm = name_map()
    print(f"买点总数 {int(brk.sum()):,};{Y0} 年起 "
          f"{int(brk[idx.year >= Y0].sum()):,}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- (A) 逐年买点清单 ----
    rows = []
    bp = np.argwhere(brk)
    for t, j in bp:
        if idx[t].year < Y0:
            continue
        e = cl[t, j]
        sl = max(float(lo_prev[t, j]), e * (1 - STOP_CAP))
        pct = 1 - sl / e
        fw = {}
        for k in (20, 60, 120, 250):
            u = min(t + k, nt - 1)
            fw[k] = cl[u, j] / e - 1.0 if u > t else np.nan
        u60 = min(t + 60, nt - 1)
        path = cl[t + 1:u60 + 1, j]
        rows.append({
            "买点日": idx[t].date(), "年": idx[t].year, "代码": codes[j],
            "名称": nm.get(codes[j], ""),
            "申万一级": None if ind[t, j] < 0 else ind_names[ind[t, j]],
            "强势日": idx[int(ts_a[t, j])].date(), "调整天数": int(adj_a[t, j]),
            "深度": float(dep[t, j]), "缩量比": float(shr[t, j]),
            "收敛比": float(cnv[t, j]), "买入价": e,
            "平台下沿": float(lo_prev[t, j]), "止损价": sl, "止损距离": pct,
            "买入日日波动率": float(vol20[t, j]),
            "止损σ倍数": pct / max(float(vol20[t, j]), 1e-9),
            "大盘过滤开": bool(mkt_on[t]), "流通市值亿": float(mv[t, j]),
            "后20日": fw[20], "后60日": fw[60], "后120日": fw[120], "后250日": fw[250],
            "60日内峰值涨幅": float(np.nanmax(path) / e - 1) if path.size else np.nan,
            "60日内触及止损": bool(path.size and np.nanmin(path) <= sl)})
    bp_df = pd.DataFrame(rows).sort_values(["买点日", "收敛比"])
    bp_df.to_csv(f"{OUT}/platform_buypoints.csv", index=False, encoding="utf-8-sig")

    # ---- (B) 框架实际成交流水(10 槽位,单次模拟)----
    ta = int(np.searchsorted(idx, pd.Timestamp(f"{Y0}-01-01")))
    pj = np.full(MAXPOS, -1, np.int64)
    pe = np.zeros(MAXPOS, np.int64)
    ppx = np.zeros(MAXPOS)
    psl = np.zeros(MAXPOS)
    trades = []

    def close_pos(s, t, why):
        j = int(pj[s])
        trades.append({"代码": codes[j], "名称": nm.get(codes[j], ""),
                       "申万一级": None if ind[pe[s], j] < 0 else ind_names[ind[pe[s], j]],
                       "买入日": idx[pe[s]].date(), "卖出日": idx[t].date(),
                       "持有交易日": int(t - pe[s]), "买入价": ppx[s],
                       "卖出价": cl[t, j], "止损价": psl[s],
                       "收益": cl[t, j] / ppx[s] - 1.0, "卖出原因": why})
        pj[s] = -1

    for t in range(ta, nt):
        for s in range(MAXPOS):
            if pj[s] < 0:
                continue
            if not mkt_on[t]:
                close_pos(s, t, "大盘过滤清仓")
            elif cl[t, int(pj[s])] <= psl[s]:
                close_pos(s, t, "止损")
            elif t - pe[s] >= HOLD_MAX:
                close_pos(s, t, "持有到期120日")
        if not mkt_on[t]:
            continue
        e = np.flatnonzero(brk[t])
        if not len(e):
            continue
        e = e[np.argsort(cnv[t, e], kind="stable")]
        k = 0
        for s in range(MAXPOS):
            if pj[s] >= 0:
                continue
            while k < len(e):
                j = int(e[k])
                k += 1
                if j in pj:
                    continue
                pj[s], pe[s] = j, t
                ppx[s] = cl[t, j]
                psl[s] = max(float(lo_prev[t, j]), cl[t, j] * (1 - STOP_CAP))
                break
            else:
                break
    for s in range(MAXPOS):
        if pj[s] >= 0:
            close_pos(s, nt - 1, "期末仍持有")
    tr_df = pd.DataFrame(trades).sort_values("买入日")
    tr_df["年"] = pd.to_datetime(tr_df["买入日"]).dt.year
    tr_df.to_csv(f"{OUT}/platform_trades.csv", index=False, encoding="utf-8-sig")

    # ---- 汇总 ----
    w = 104
    print(f"\n{'='*w}\n逐年买点与实际成交(不设判据,仅描述)\n{'='*w}")
    print(f"{'年':<6}{'买点数':>8}{'涉及只数':>9}{'大盘开占比':>11}"
          f"{'后60日中位':>11}{'60日峰值≥50%':>13}{'触止损占比':>11}"
          f"{'│成交笔':>9}{'胜率':>8}{'单笔中位':>10}{'单笔均值':>10}")
    summ = []
    for y in sorted(bp_df["年"].unique()):
        a = bp_df[bp_df["年"] == y]
        b = tr_df[tr_df["年"] == y]
        r = {"年": int(y), "买点数": len(a), "涉及只数": a["代码"].nunique(),
             "大盘开占比": float(a["大盘过滤开"].mean()),
             "后60日中位": float(a["后60日"].median()),
             "60日峰值ge50": float((a["60日内峰值涨幅"] >= 0.5).mean()),
             "触止损占比": float(a["60日内触及止损"].mean()),
             "成交笔": len(b),
             "胜率": float((b["收益"] > 0).mean()) if len(b) else np.nan,
             "单笔中位": float(b["收益"].median()) if len(b) else np.nan,
             "单笔均值": float(b["收益"].mean()) if len(b) else np.nan}
        summ.append(r)
        print(f"{y:<6}{r['买点数']:>8,}{r['涉及只数']:>9,}{r['大盘开占比']:>11.1%}"
              f"{r['后60日中位']:>11.2%}{r['60日峰值ge50']:>13.1%}"
              f"{r['触止损占比']:>11.1%}{len(b):>9}"
              f"{r['胜率']:>8.1%}{r['单笔中位']:>10.2%}{r['单笔均值']:>10.2%}")
    sm_df = pd.DataFrame(summ)
    sm_df.to_csv(f"{OUT}/platform_yearly_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n合计:买点 {len(bp_df):,} 个、涉及 {bp_df['代码'].nunique():,} 只;"
          f"实际成交 {len(tr_df):,} 笔,胜率 {(tr_df['收益']>0).mean():.1%},"
          f"单笔中位 {tr_df['收益'].median():+.2%}、均值 {tr_df['收益'].mean():+.2%}")
    print(tr_df["卖出原因"].value_counts().to_string())
    print(f"\n落库 {OUT}/platform_buypoints.csv、platform_trades.csv、"
          f"platform_yearly_summary.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
