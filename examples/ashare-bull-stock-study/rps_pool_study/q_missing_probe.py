"""核查 2026-08-28 的 R09 质量分缺失率为何从 42% 抬到 57%(描述性,不设判据)。

做法:在**同一张扩展面板**上,对中报披露前后的若干个交易日各算一次 R09 核心质量分,
并拆出四个因子各自的缺失来源。若缺失率在中报披露日附近跳升,说明是财报周期效应;
若一直高,说明我的财务拼接有问题。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE  # noqa: E402
from codex_routes_rerun import build_fund, wrank  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR")
DATES = ["2026-06-30", "2026-07-31", "2026-08-14", "2026-08-21", "2026-08-25",
         "2026-08-28"]


def main():
    t0 = time.time()
    z = np.load(CACHE, allow_pickle=True)
    zc = list(z["codes"])
    import glob
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cl0 = pd.read_parquet(f"{DATA}/{codes[0]}.parquet", columns=["close"])
    idx = pd.DatetimeIndex(pd.to_datetime(cl0.index).tz_localize(None))
    for c in codes[:50]:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["close"])
        i2 = pd.DatetimeIndex(pd.to_datetime(x.index).tz_localize(None))
        idx = idx.union(i2)
    nt = len(idx)
    raw = np.full((nt, len(zc)), np.nan, np.float32)
    for j, c in enumerate(zc):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        x.index = pd.to_datetime(x.index).tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    fm, _ = build_fund(zc, idx)
    ok = z["OK"]
    ok = np.vstack([ok, np.repeat(ok[-1:], nt - ok.shape[0], 0)])[:nt]
    print(f"面板 {nt} 日 × {len(zc)} 只;因子面板就绪 ({time.time()-t0:.0f}s)\n",
          flush=True)
    print(f"{'日期':<13}{'合格':>7}{'净利率缺':>10}{'ROE缺':>9}{'ROE改善缺':>11}"
          f"{'现金转换缺':>11}{'四项复合缺':>12}")
    for d in DATES:
        t = int(idx.searchsorted(pd.Timestamp(d)))
        if t >= nt or idx[t] != pd.Timestamp(d):
            continue
        e = np.flatnonzero(ok[t])
        px = raw[t, e].astype(np.float64)
        ep = fm["eps_ttm"][t, e] / px
        cfp = fm["ocfps_ttm"][t, e] / px
        marg = fm["ni_ttm"][t, e] / np.where(fm["rev_ttm"][t, e] != 0,
                                             fm["rev_ttm"][t, e], np.nan)
        roe, roec = fm["roe_lvl"][t, e], fm["roe_chg"][t, e]
        conv = fm["ocfps_ttm"][t, e] / np.where(fm["eps_ttm"][t, e] != 0,
                                                fm["eps_ttm"][t, e], np.nan)
        prof, cash = ep > 0, cfp > 0
        r = [wrank(marg, marg > 0), wrank(roe, roe > 0), wrank(roec, prof),
             wrank(conv, prof & cash & (conv > 0))]
        miss = [float(np.isnan(x).mean()) for x in r]
        comp = float(np.any(np.isnan(np.vstack(r)), axis=0).mean())
        print(f"{d:<13}{len(e):>7,}{miss[0]:>10.1%}{miss[1]:>9.1%}{miss[2]:>11.1%}"
              f"{miss[3]:>11.1%}{comp:>12.1%}")
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
