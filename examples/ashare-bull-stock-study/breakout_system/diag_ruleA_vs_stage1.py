"""诊断:规则A(+4.61%/笔)与阶段1 同参数配置(+4.00%/笔)为何不同

两者参数相同(-10%固定止损、无止盈、无MA50、252日上限),
但笔数 70,124 vs 69,517、净期望 +4.61% vs +4.00%。
**在把任一数字写进文档前必须定位差异,不能猜。**

新脚本相对阶段1 有三处改动,逐一开关以隔离:
  (1) 非正价格清洗  OP/HI/LO/CL = X.where(X > 0)
  (2) 跳空穿越止损线 → 以开盘价成交(阶段1 一律按止损价成交)
  (3) 到期日收盘为 NaN 时按最后有效价平仓(阶段1 直接丢弃该笔)
"""
import glob
import os

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
STOP, MAX_HOLD, COST = 0.10, 252, 0.003

o, h, l, c = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "high", "low", "close"])
    if x.empty:
        continue
    o[k] = pd.to_numeric(x["open"], errors="coerce"); h[k] = pd.to_numeric(x["high"], errors="coerce")
    l[k] = pd.to_numeric(x["low"], errors="coerce"); c[k] = pd.to_numeric(x["close"], errors="coerce")
OPr = pd.DataFrame(o).sort_index(); OPr.index = OPr.index.tz_localize(None)
HIr = pd.DataFrame(h).set_axis(OPr.index); LOr = pd.DataFrame(l).set_axis(OPr.index)
CLr = pd.DataFrame(c).set_axis(OPr.index)
idx = OPr.index; NT = len(idx); pos = {d: i for i, d in enumerate(idx)}

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv", usecols=["code", "D"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
ev = ev[ev.code.isin(OPr.columns)].copy()
ev["dp"] = ev["D"].map(pos); ev = ev.dropna(subset=["dp"]); ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp + 1 < NT - 5]
print(f"事件 {len(ev):,}")


def sim(clean, gap_fill, keep_nan_end):
    if clean:
        OP = OPr.where(OPr > 0); HI = HIr.where(HIr > 0)
        LO = LOr.where(LOr > 0); CL = CLr.where(CLr > 0)
    else:
        OP, HI, LO, CL = OPr, HIr, LOr, CLr
    OPa, LOa, CLa = OP.to_numpy(), LO.to_numpy(), CL.to_numpy()
    col = {cd: i for i, cd in enumerate(OP.columns)}
    rets, dropped, gapped = [], 0, 0
    for code, grp in ev.groupby("code", sort=False):
        ci = col[code]
        op, lo, cl = OPa[:, ci], LOa[:, ci], CLa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            spx = entry * (1 - STOP)
            end = min(e + MAX_HOLD, NT - 1)
            exit_px, last = None, entry
            for t in range(e, end + 1):
                if not np.isfinite(cl[t]):
                    continue
                last = cl[t]
                if np.isfinite(lo[t]) and lo[t] <= spx:
                    if gap_fill and np.isfinite(op[t]) and op[t] < spx:
                        exit_px = op[t]; gapped += 1
                    else:
                        exit_px = spx
                    break
            if exit_px is None:
                exit_px = cl[end]
                if not np.isfinite(exit_px):
                    if keep_nan_end:
                        exit_px = last
                    else:
                        dropped += 1
                        continue
            if not np.isfinite(exit_px) or exit_px <= 0:
                dropped += 1
                continue
            rets.append(exit_px / entry - 1)
    r = np.array(rets)
    return r, dropped, gapped


print(f"\n{'清洗非正价':<12}{'跳空按开盘':<12}{'保留NaN到期':<13}{'笔数':>9}{'胜率':>8}{'毛期望':>9}{'净期望':>9}{'丢弃':>7}")
cfgs = [
    (False, False, False, "阶段1 原始配置"),
    (True,  False, False, "只加①清洗"),
    (True,  True,  False, "①+②跳空"),
    (True,  True,  True,  "①+②+③ = 规则A"),
    (False, False, True,  "只加③保留NaN到期"),
]
base = None
for clean, gap, keep, label in cfgs:
    r, dr, gp = sim(clean, gap, keep)
    net = r.mean() - COST
    if base is None:
        base = net
    print(f"{str(clean):<12}{str(gap):<12}{str(keep):<13}{len(r):>9,}{(r>0).mean():>8.1%}"
          f"{r.mean():>+9.2%}{net:>+9.2%}{dr:>7,}   {label}  (Δ净期望 {net-base:+.2f}pp)")

# 单独看被阶段1 丢弃的那批交易到底是什么
r_keep, _, _ = sim(True, True, True)
r_drop, nd, _ = sim(True, True, False)
print(f"\n阶段1 丢弃的 {nd:,} 笔(到期日收盘为NaN,即退市/长停):")
n_all, n_kept = len(r_keep), len(r_drop)
print(f"  含它们 {n_all:,} 笔 均值 {r_keep.mean():+.2%};不含 {n_kept:,} 笔 均值 {r_drop.mean():+.2%}")
implied = (r_keep.sum() - r_drop.sum()) / (n_all - n_kept) if n_all > n_kept else float("nan")
print(f"  → 这批交易的隐含平均收益 **{implied:+.1%}**(按最后有效成交价计)")
