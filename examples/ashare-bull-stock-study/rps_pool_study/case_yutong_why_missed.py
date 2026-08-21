"""诊断:宇通 2013-2014 那段平台,筛选器为什么没认出来(不是检验,不设判据)

用户看图指出:2013→2014 是平台整理,2015 上涨成功;2016-17 也是平台,只是失败了。
§93 实测筛选器在宇通身上只亮过 4 段(2015-12/2016-11/2017-12/2023-11),
**2013-2014 那段不在其中。** 本脚本逐月末打印宇通的全部中间量,
看是哪一个条件把它挡住了 —— **是筛选器漏了,还是那段本来就不满足定义。**

检测一律调 `score_one`(公开入口),中间量另算仅用于解释,不参与判定。
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from consolidation_screener import (  # noqa: E402
    MIN_ADJ_DAYS_LEGACY,
    STRONG_LOOKBACK,
    THR_ATR,
    THR_DEPTH,
    THR_SHRINK,
    load_panel,
    score_one,
    series_of,
)

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
CODE = "600066"

CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    k = list(CL.columns).index("510300")
    STRONG = np.delete(STRONG, k, axis=1)
    CL = CL.drop(columns=["510300"])
    MA100 = MA100.drop(columns=["510300"])
    frames.pop("510300", None)
idx = CL.index
J = list(CL.columns).index(CODE)
h, lo, c, v = series_of(frames, idx, CODE)
ma = MA100[CODE].to_numpy(float)
sdall = np.flatnonzero(STRONG[:, J])
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}")
print(f"宇通全期强势日(RPS60>90)共 {sdall.size} 天;"
      f"最早 {idx[sdall[0]].date() if sdall.size else '无'}")
print(f"  2013-2015 年内的强势日:"
      f" {[str(idx[x].date()) for x in sdall if idx[x].year <= 2015][:40]}")

ym = idx.to_period("M")
print(f"\n{'月末':<9}{'强势日ts':<12}{'调整天数':>7}{'触线':>5}{'深度':>8}"
      f"{'缩量比':>8}{'收敛比':>8}   legacy 判定 / 被谁挡住")
for p in [x for x in ym.unique() if 2013 <= x.year <= 2016]:
    t = int(np.flatnonzero(ym == p)[-1])
    if not np.isfinite(c[t]):
        continue
    sd = sdall[sdall <= t]
    cand = sd[sd >= t - STRONG_LOOKBACK]
    if cand.size == 0:
        print(f"{p!s:<9}{'—':<12}{'':>7}{'':>5}{'':>8}{'':>8}{'':>8}"
              f"   ✗ 250 日内没有强势日(RPS60>90)")
        continue
    ts = int(cand[-1])
    # 触线日:仅用于解释,判定仍以 score_one 为准
    td = -1
    for kk in range(ts + 1, t + 1):
        ok = (np.isfinite(ma[kk]) and np.isfinite(lo[kk]) and lo[kk] <= ma[kk] * 1.03
              and kk >= 20 and np.isfinite(ma[kk - 20]) and ma[kk] > ma[kk - 20])
        if ok:
            td = kk
            break
    s = score_one(h, lo, c, v, ma, sdall, t, legacy=True)
    tag = f"{idx[ts].date()}"
    if s is None:
        why = (f"调整天数 {t - ts} < {MIN_ADJ_DAYS_LEGACY}"
               if t - ts < MIN_ADJ_DAYS_LEGACY
               else "**没有触线日**(未回踩20周线,或20周线未向上)")
        print(f"{p!s:<9}{tag:<12}{t-ts:>7}{'否' if td<0 else '是':>5}"
              f"{'':>8}{'':>8}{'':>8}   ✗ {why}")
        continue
    bad = []
    if not s["缩量比"] < THR_SHRINK:
        bad.append(f"缩量比 {s['缩量比']:.2f}≥{THR_SHRINK}")
    if not s["收敛比"] < THR_ATR:
        bad.append(f"收敛比 {s['收敛比']:.2f}≥{THR_ATR}")
    if not s["深度"] <= THR_DEPTH:
        bad.append(f"深度 {s['深度']:.1%}>{THR_DEPTH:.1%}")
    print(f"{p!s:<9}{tag:<12}{s['调整天数']:>7}{'是':>5}{s['深度']:>8.1%}"
          f"{s['缩量比']:>8.2f}{s['收敛比']:>8.2f}"
          f"   {'✓ 三条全中' if not bad else '✗ ' + '、'.join(bad)}")

print(f"\n{'='*100}\n参考:宇通 2013-2015 的实际走势(月末收盘)\n{'='*100}")
fa = CL[CODE].where(CL[CODE] > 0).ffill().to_numpy(float)
row = []
for p in [x for x in ym.unique() if 2013 <= x.year <= 2015]:
    t = int(np.flatnonzero(ym == p)[-1])
    row.append(f"{p!s} {fa[t]:.2f}")
for i in range(0, len(row), 6):
    print("   " + "   ".join(row[i:i + 6]))
