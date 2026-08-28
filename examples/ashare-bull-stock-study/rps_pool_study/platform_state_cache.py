"""§160 前置:把平台状态算一次存下来,并核查「叠加到 R 路线」的可行性。

**本脚本不设通过/不通过判据** —— 两件事:
(1) 把 `vec_screen` 的输出缓存成 npz,后续叠加实验直接读,不再每次重算 2~3 分钟;
(2) 报出「三条全中」与「突破买点」的覆盖率,用来判断
    「把平台筛选器叠加到 R01–R13 上」在样本量上到底可不可行。

口径与第一五五/一五六/一五九节逐字一致(legacy 绝对阈值)。
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
from consolidation_screener import THR_ATR, THR_DEPTH, THR_SHRINK, load_panel  # noqa: E402
from platform_pivot import MA_MKT, vec_screen  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
CACHE = f"{OUT}/platform_state.npz"


def main():
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
    d2 = {k: {} for k in ("is_st", "is_suspended", "listed_days", "volume")}
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
    ok = (~al("is_st", True).astype(bool) & ~al("is_suspended", True).astype(bool)
          & (al("listed_days", 0) >= 250) & (al("volume", 0) > 0) & np.isfinite(cl))
    hit3 = (shr < THR_SHRINK) & (cnv < THR_ATR) & (dep <= THR_DEPTH) & (adj_a >= 0)
    up_prev = np.full((nt, ns), np.nan, np.float64)
    lo_prev = np.full((nt, ns), np.nan, np.float64)
    same = np.zeros((nt, ns), bool)
    same[1:] = ts_a[1:] == ts_a[:-1]
    up_prev[1:] = np.where(same[1:], hi[:-1], np.nan)
    lo_prev[1:] = np.where(same[1:], lo[:-1], np.nan)
    brk = hit3 & ok & np.isfinite(up_prev) & (cl > up_prev) & np.isfinite(lo_prev)
    with np.errstate(all="ignore"):
        rr = cl[1:] / cl[:-1] - 1.0
    msk = ok[1:] & ok[:-1] & np.isfinite(rr)
    dd = np.zeros(nt)
    dd[1:] = np.where(msk.sum(1) > 0,
                      np.nan_to_num(rr * msk).sum(1) / np.maximum(msk.sum(1), 1), 0.0)
    nav = np.cumprod(1 + dd)
    mm = pd.Series(nav).rolling(MA_MKT, min_periods=MA_MKT).mean().to_numpy()
    mkt_on = ~(np.isfinite(mm) & (nav < mm))
    np.savez_compressed(CACHE, hit3=hit3, brk=brk, ts_a=ts_a, lo_prev=lo_prev,
                        cnv=cnv.astype(np.float32), dep=dep.astype(np.float32),
                        shr=shr.astype(np.float32), ok=ok, mkt_on=mkt_on, nav=nav,
                        codes=np.array(codes), dates=idx.values)
    w = 92
    h3 = hit3 & ok
    print(f"\n{'='*w}\n叠加可行性:合格股票·日 中「三条全中」与「突破买点」的占比\n{'='*w}")
    print(f"{'年':<7}{'合格股·日':>12}{'三条全中':>11}{'占比':>9}"
          f"{'每日只数':>10}{'│突破买点':>11}{'占比':>9}{'每日只数':>10}")
    yrs = pd.Series(idx).dt.year.to_numpy()
    for y in range(2016, 2027):
        m = yrs == y
        n = int(ok[m].sum())
        a, b = int(h3[m].sum()), int(brk[m].sum())
        nd = int(m.sum())
        print(f"{y:<7}{n:>12,}{a:>11,}{a/max(n,1):>9.3%}{a/nd:>10.1f}"
              f"{b:>11,}{b/max(n,1):>9.4%}{b/nd:>10.1f}")
    n, a, b = int(ok.sum()), int(h3.sum()), int(brk.sum())
    print(f"\n全期:合格 {n:,} 股·日;三条全中 {a:,}({a/n:.3%});突破买点 {b:,}({b/n:.4%})")
    print(f"\n如果一个因子组合每期持 N 只,平均能落在平台状态里的只数 ≈ N × {a/n:.3%};")
    print(f"落在突破买点上的 ≈ N × {b/n:.4%} —— N=50 时分别约 "
          f"{50*a/n:.2f} 只与 {50*b/n:.3f} 只。")
    print(f"\n缓存 {CACHE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
