"""检测器人工核对:10 个真实牛股案例,把基底的起止日、深度、pivot 打出来

检测器写错的话后面全是废数,所以先核对再跑全量。
案例取自 bull_feature_panel.parquet 里的牛股(年涨幅>100%),
只加载这些股票的 parquet,不加载全市场。

核对要点(逐条看输出):
  1. 基底起点/终点日期是否落在起涨点**之前**(不能有前视)
  2. 杯柄的深度是否在 12~33%、手柄 8~15%
  3. pivot 是否 ≤ 起涨点之后的最高价(pivot 高于全年最高说明算错了)
  4. 前期涨幅是否真的 ≥30%
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_pattern_detector import CUP, NEED, PRIOR, WIN, detect_base  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"

P = pd.read_parquet(f"{SP}/bull_feature_panel.parquet", columns=["code", "year", "bull", "t"])
# 需要完整窗口,且分散在不同年份
cand = P[P.bull & (P.t >= NEED)].sort_values(["year", "code"])
pick = cand.groupby("year").head(2).head(14)
print(f"候选牛股-年份 {len(cand):,} 个,抽 {len(pick)} 个核对\n")

# 用任一股票的索引作为全局交易日轴
_any = pd.read_parquet(f"{DATA}/{P.code.iloc[0]}.parquet", columns=["close"])
IDX = _any.index.tz_localize(None)
# 面板的索引是全市场并集,单只股票的 parquet 未必等长 —— 用并集重建
codes_needed = sorted(pick.code.unique())
frames = {}
for cd in codes_needed:
    x = pd.read_parquet(f"{DATA}/{cd}.parquet",
                        columns=["high", "low", "close", "volume"])
    x.index = x.index.tz_localize(None)
    frames[cd] = x
UNION = pd.read_parquet(f"{SP}/base_pattern_axis.parquet").index if os.path.exists(
    f"{SP}/base_pattern_axis.parquet") else None
if UNION is None:                      # 首次运行:用全市场并集重建交易日轴
    import glob
    ax = None
    for f in sorted(glob.glob(f"{DATA}/*.parquet")):
        if os.path.basename(f) == "510300.parquet":
            continue
        i = pd.read_parquet(f, columns=["close"]).index
        ax = i if ax is None else ax.union(i)
    UNION = ax.tz_localize(None)
    pd.DataFrame(index=UNION).to_parquet(f"{SP}/base_pattern_axis.parquet")
print(f"交易日轴 {len(UNION):,} 天  {UNION[0].date()} ~ {UNION[-1].date()}\n")

nfound = 0
for _, r in pick.iterrows():
    x = frames[r.code].reindex(UNION)
    c = pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0).to_numpy()
    h = pd.to_numeric(x["high"], errors="coerce").where(lambda s: s > 0).to_numpy()
    lo = pd.to_numeric(x["low"], errors="coerce").where(lambda s: s > 0).to_numpy()
    v = pd.to_numeric(x["volume"], errors="coerce").to_numpy()
    pmin = pd.Series(c).rolling(PRIOR, min_periods=60).min().shift(1).to_numpy()
    t = int(r.t)
    s0 = t - WIN
    b = detect_base(c[s0:t], h[s0:t], lo[s0:t], v[s0:t], pmin[s0:t])
    tag = [k for k in ("cup", "flat", "dbl") if b[k]]
    if not tag:
        print(f"{r.code} {r.year}  起涨点 {UNION[t].date()}  —— 无基底")
        continue
    nfound += 1
    yr_hi = np.nanmax(h[t:t + 250]) if t + 250 <= len(h) else np.nanmax(h[t:])
    print(f"{r.code} {r.year}  起涨点 {UNION[t].date()}  收盘 {c[t]:.2f}   形态 {tag}")
    for k, nm in (("cup", "杯柄"), ("flat", "平底"), ("dbl", "双底")):
        if not b[k]:
            continue
        st = s0 + b[f"{k}_start"]
        pv = b[f"{k}_pivot"]
        pre = c[st] / pmin[st] - 1 if np.isfinite(pmin[st]) and pmin[st] > 0 else np.nan
        extra = f"  手柄深 {b['cup_handle']:.1%}" if k == "cup" else ""
        print(f"    {nm}: {UNION[st].date()} → {UNION[t-1].date()}"
              f"  ({t-st} 交易日)  深度 {b[f'{k}_depth']:.1%}{extra}")
        print(f"         前期涨幅 {pre:+.1%}   pivot {pv:.2f}"
              f"   起涨点价/pivot {c[t]/pv-1:+.1%}   之后250日最高 {yr_hi:.2f}")
        # 硬检查
        assert st < t, "基底起点不在起涨点之前"
        assert pv <= np.nanmax(h[st:t]) + 1e-9, "pivot 超出基底期内最高价 —— 有前视"
print(f"\n{nfound}/{len(pick)} 个案例检出基底")
