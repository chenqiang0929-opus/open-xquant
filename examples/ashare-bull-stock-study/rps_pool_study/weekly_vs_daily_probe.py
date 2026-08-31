"""平台筛选器能不能只用周线算?—— 用同一批面板直接量,不靠推理。

**描述性核查,不设通过/不通过判据。**

做法:把强势日 ts 与触线日 td **固定为日线口径的取值**(即先不追问"周线能不能定位
ts/td"),只问一件事 —— **同一个 [ts, t] 窗口,改用周线 OHLCV 算出来的三项,
与日线版差多少。** 这样能把两个问题拆开:
  (甲) 三项指标本身能不能从周线复现;
  (乙) ts/td 的日期分辨率损失(单独讨论,不在本脚本)。

周线口径:W-FRI 重采样,剔除整周无交易的周;
  周high=当周最高、周low=当周最低、周close=当周最后收盘、周volume=当周成交量之和;
  周TR = max(周high−周low, |周high−上周close|, |周low−上周close|)。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codex_r10_replication import DATA  # noqa: E402
from consolidation_screener import (  # noqa: E402
    THR_ATR,
    THR_DEPTH,
    THR_SHRINK,
    load_panel,
    series_of,
)
from platform_pivot import vec_screen  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NSAMP = 400          # 抽样股票数(全量太慢,抽样足以看出量级)


def main():  # noqa: PLR0915
    t0 = time.time()
    cl_df, frames, strong, ma100 = load_panel(DATA)
    if "510300" in cl_df.columns:
        cl_df = cl_df.drop(columns=["510300"])
        strong = strong[:, [i for i, c in enumerate(ma100.columns) if c != "510300"]]
    idx, codes = cl_df.index, list(cl_df.columns)
    nt, ns = cl_df.shape
    assert (nt, ns) == (3297, 5232), f"锚点 {cl_df.shape}"
    ts_a, adj_a, dep, shr, cnv, _, _ = vec_screen(
        cl_df.to_numpy(float), frames, strong, ma100, idx, codes)
    hit3 = (shr < THR_SHRINK) & (cnv < THR_ATR) & (dep <= THR_DEPTH) & (adj_a >= 0)
    print(f"日线口径就绪 ({time.time()-t0:.0f}s)", flush=True)

    # 周边界:W-FRI,剔除整周无交易
    wk = pd.Series(np.arange(nt), index=idx).resample("W-FRI").last().dropna()
    wend = wk.to_numpy().astype(int)                 # 每周最后一个交易日的下标
    wstart = np.concatenate([[0], wend[:-1] + 1])
    rng = np.random.default_rng(20260828)
    js = rng.choice(ns, NSAMP, replace=False)
    rows = []
    for j in js:
        h, low, c, v = series_of(frames, idx, codes[j])
        if not np.isfinite(c).any():
            continue
        # 周线 OHLCV
        wh = np.array([np.nanmax(h[a:b + 1]) if np.isfinite(h[a:b + 1]).any()
                       else np.nan for a, b in zip(wstart, wend, strict=True)])
        wl = np.array([np.nanmin(low[a:b + 1]) if np.isfinite(low[a:b + 1]).any()
                       else np.nan for a, b in zip(wstart, wend, strict=True)])
        wc = c[wend]
        wv = np.array([np.nansum(v[a:b + 1]) for a, b in zip(wstart, wend,
                                                             strict=True)])
        pc = np.roll(wc, 1)
        pc[0] = np.nan
        wtr = np.maximum(wh - wl, np.maximum(np.abs(wh - pc), np.abs(wl - pc)))
        # 日线 TR
        dpc = np.roll(c, 1)
        dpc[0] = np.nan
        dtr = np.maximum(h - low, np.maximum(np.abs(h - dpc), np.abs(low - dpc)))
        for wi_, t in enumerate(wend):              # 只在周末评估,口径可比
            if adj_a[t, j] < 0 or not np.isfinite(dep[t, j]):
                continue
            ts = int(ts_a[t, j])
            # 把 [ts, t] 与 [ts-60, ts-1] 映射到周下标
            w_ts = int(np.searchsorted(wend, ts, side="left"))
            w_pre0 = int(np.searchsorted(wend, max(ts - 60, 0), side="left"))
            if w_ts <= w_pre0 or wi_ <= w_ts:
                continue
            seg_h, seg_l = wh[w_ts:wi_ + 1], wl[w_ts:wi_ + 1]
            pre_v, pre_tr = wv[w_pre0:w_ts], wtr[w_pre0:w_ts]
            if not (np.isfinite(seg_h).any() and np.isfinite(seg_l).any()
                    and np.isfinite(pre_v).any() and np.isfinite(pre_tr).any()):
                continue
            wdep = 1 - np.nanmin(seg_l) / np.nanmax(seg_h)
            wshr = np.nanmean(wv[w_ts:wi_ + 1]) / np.nanmean(pre_v)
            wcnv = np.nanmean(wtr[w_ts:wi_ + 1]) / np.nanmean(pre_tr)
            rows.append({
                "j": int(j), "t": int(t), "日期": idx[t].date(),
                "日_深度": float(dep[t, j]), "周_深度": wdep,
                "日_缩量比": float(shr[t, j]), "周_缩量比": wshr,
                "日_收敛比": float(cnv[t, j]), "周_收敛比": wcnv,
                "日_三条全中": bool(hit3[t, j]),
                "周_三条全中": bool((wshr < THR_SHRINK) & (wcnv < THR_ATR)
                                & (wdep <= THR_DEPTH)),
                "日_TR均": float(np.nanmean(dtr[ts:t + 1])),
                "周_TR均": float(np.nanmean(wtr[w_ts:wi_ + 1]))})
    df = pd.DataFrame(rows)
    w = 96
    print(f"\n{'='*w}\n周线 vs 日线:同一 [ts, t] 窗口的三项指标\n{'='*w}")
    print(f"可比样本 {len(df):,} 个(股票 {df.j.nunique()} 只,只在周末评估)\n")
    print(f"{'指标':<10}{'相关':>9}{'中位|差|':>11}{'中位相对差':>12}"
          f"{'|差|<0.02占比':>14}{'周/日 中位比':>13}")
    for k in ("深度", "缩量比", "收敛比"):
        a, b = df[f"日_{k}"].to_numpy(), df[f"周_{k}"].to_numpy()
        g = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a[g] - b[g])
        rel = np.abs(a[g] - b[g]) / np.maximum(np.abs(a[g]), 1e-9)
        print(f"{k:<10}{np.corrcoef(a[g], b[g])[0,1]:>9.4f}{np.median(d):>11.4f}"
              f"{np.median(rel):>12.1%}{(d < 0.02).mean():>14.1%}"
              f"{np.median(b[g]/np.maximum(a[g],1e-9)):>13.3f}")
    print(f"\n{'='*w}\n判定翻转:三条全中\n{'='*w}")
    ct = pd.crosstab(df["日_三条全中"], df["周_三条全中"])
    print(ct.to_string())
    agree = float((df["日_三条全中"] == df["周_三条全中"]).mean())
    dt = int(((df["日_三条全中"]) & (~df["周_三条全中"])).sum())
    tw = int(((~df["日_三条全中"]) & (df["周_三条全中"])).sum())
    print(f"\n一致率 {agree:.1%};日线中周线不中 {dt};周线中日线不中 {tw}")
    print(f"\n周 TR / 日 TR 的中位倍数:"
          f"{np.median(df['周_TR均']/np.maximum(df['日_TR均'],1e-9)):.2f}×")
    df.to_csv(f"{OUT}/weekly_vs_daily_probe.csv", index=False,
              encoding="utf-8-sig")
    print(f"\n落库 {OUT}/weekly_vs_daily_probe.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
