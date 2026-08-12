"""检测器核对(第二版):在**真实突破日**上检测,而不是在 t\\* 上

═══ 第一版发现的结构性问题(必须记下来) ═══
在 t\\*(该年最大涨幅的**起点**)上检测,14 个牛股案例**一个杯柄都没有**,
只有平底。查下来不是 bug,是定义决定的:
  t\\* 是低点,而杯柄的手柄要求价格已**回到左沿的 95%**、
  且手柄低点在基底**上半部** —— 一个刚到低点的股票不可能满足。
**结论:归因(A 部分)在 t\\* 上只能测出平底,杯柄/双底必须在突破日上测。**
这不是把检验换个地方做,而是这两种形态本来就定义在突破点。

本脚本在 `oneil_prelaunch_events_fixed.csv` 的突破日上跑检测器,
既验证杯柄能被检出,也给 B 部分探底命中率。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_pattern_detector import NEED, PRIOR, WIN, detect_base  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"

UNION = pd.read_parquet(f"{SP}/base_pattern_axis.parquet").index
pos = {d: i for i, d in enumerate(UNION)}

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv",
                 usecols=["code", "D", "fwd_gain", "winner"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
ev["dp"] = ev["D"].map(pos)
ev = ev.dropna(subset=["dp"])
ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp >= NEED]
print(f"突破事件 {len(ev):,}(其中 dp≥{NEED} 的)")

rng = np.random.default_rng(20260812)
smp = ev.iloc[rng.choice(len(ev), 4000, replace=False)].sort_values("code")
cnt = {"cup": 0, "flat": 0, "dbl": 0, "any": 0}
shown = 0
for cd, g in smp.groupby("code"):
    try:
        x = pd.read_parquet(f"{DATA}/{cd}.parquet",
                            columns=["high", "low", "close", "volume"])
    except Exception:
        continue
    x.index = x.index.tz_localize(None)
    x = x.reindex(UNION)
    c = pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0).to_numpy()
    h = pd.to_numeric(x["high"], errors="coerce").where(lambda s: s > 0).to_numpy()
    lo = pd.to_numeric(x["low"], errors="coerce").where(lambda s: s > 0).to_numpy()
    v = pd.to_numeric(x["volume"], errors="coerce").to_numpy()
    pmin = pd.Series(c).rolling(PRIOR, min_periods=60).min().shift(1).to_numpy()
    for _, r in g.iterrows():
        t = int(r.dp)
        s0 = t - WIN
        b = detect_base(c[s0:t], h[s0:t], lo[s0:t], v[s0:t], pmin[s0:t])
        for k in ("cup", "flat", "dbl"):
            cnt[k] += bool(b[k])
        cnt["any"] += bool(b["cup"] or b["flat"] or b["dbl"])
        if b["cup"] and shown < 8:
            shown += 1
            st = s0 + b["cup_start"]
            pv = b["cup_pivot"]
            assert st < t and pv <= np.nanmax(h[st:t]) + 1e-9
            print(f"  杯柄 {cd} 突破日 {UNION[t].date()}  基底 {UNION[st].date()}"
                  f"→{UNION[t-1].date()} ({t-st}日)  深 {b['cup_depth']:.1%}"
                  f"  手柄 {b['cup_handle']:.1%}  pivot {pv:.2f}"
                  f"  突破日收盘/pivot {c[t]/pv-1:+.1%}  后续 {r.fwd_gain:+.1%}")

n = len(smp)
print(f"\n{n:,} 个突破日抽样的命中率:")
for k, nm in (("cup", "杯柄"), ("flat", "平底"), ("dbl", "双底"), ("any", "任一")):
    print(f"  {nm:<4} {cnt[k]:>6,}  {cnt[k]/n:>7.1%}")
