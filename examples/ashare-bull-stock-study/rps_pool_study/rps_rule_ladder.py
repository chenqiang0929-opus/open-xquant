"""按用户描述的规则逐级收紧,量化"未知的那层筛选"到底值多少

═══ 起因 ═══
用户澄清:筛选规则是「RPS50/120/250 至少一个 > 90」。
实测确认:B 池 7,199 行中 100.0% 满足该条件,只有 1 行例外。

**但规模对不上**:
  我按"至少一个>90"重建 → 每期 **856 只**
  我按"三个全>90"重建   → 每期 **178 只**
  用户实际池子           → 每期 **90 只**   ← 比"三个全>90"还窄一倍

我的 RPS 计算没问题:与用户值的相关 0.961/0.978/0.990,中位数对到 0.1。
所以**用户的筛选里还有一层他没提到、我也复现不了的收窄**。

═══ 本脚本要回答的 ═══
把 RPS 规则从松到紧排成阶梯,看收益随之如何变化:
  L1 至少一个>90        (856只) ← 用户口述的规则
  L2 至少两个>90
  L3 三个全>90          (178只)
  L4 三个全>95
  L5 用户实际池子        ( 90只)
每级都再叠加双增长,与全市场等权基准并列。

**这不是参数搜索**:五级全部由用户自己描述的规则加"更严"的自然延伸构成,
方向事前锁定(越严越接近用户池子),不挑最优、不回头调。
目的是**归因**——把"可复述规则的贡献"与"未知收窄的贡献"分开。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST, TURN = 0.003, 0.32          # 与主检验同口径

t0 = time.time()
op, cl, niy, rev = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "close", "ni_yoy_252", "revenue"])
    if x.empty:
        continue
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    niy[k] = pd.to_numeric(x["ni_yoy_252"], errors="coerce")
    rev[k] = pd.to_numeric(x["revenue"], errors="coerce")
OP = pd.DataFrame(op).sort_index(); OP.index = OP.index.tz_localize(None)
CL = pd.DataFrame(cl).set_axis(OP.index)
NIY = pd.DataFrame(niy).set_axis(OP.index); REV = pd.DataFrame(rev).set_axis(OP.index)
OP = OP.where(OP > 0); CL = CL.where(CL > 0)
idx = OP.index
OPa, CLa = OP.to_numpy(), CL.to_numpy()
col_of = {c: i for i, c in enumerate(OP.columns)}
R = {n: (CL.pct_change(n).rank(axis=1, pct=True) * 100) for n in (50, 120, 250)}
REVY = (REV / REV.shift(252) - 1).replace([np.inf, -np.inf], np.nan)
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del op, cl, niy, rev


def rets(codes, e, x):
    out = []
    for c in codes:
        ci = col_of.get(c)
        if ci is None:
            continue
        a, b = OPa[e, ci], OPa[x, ci]
        if not np.isfinite(a) or a <= 0:
            continue
        if not np.isfinite(b) or b <= 0:
            seg = CLa[e:x + 1, ci]; seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                continue
            b = seg[-1]
        out.append(b / a - 1)
    return np.array(out)


def comp(r, d):
    r = np.asarray(r, float); d = np.asarray(d, float)
    ok = np.isfinite(r) & np.isfinite(d)
    if ok.sum() == 0:
        return np.nan
    t = np.prod(1 + r[ok]); y = d[ok].sum() / 252
    return t ** (1 / y) - 1 if t > 0 and y > 0 else -1.0


LEVELS = [("L1 至少一个>90(用户口述的规则)", 1, 90),
          ("L2 至少两个>90", 2, 90),
          ("L3 三个全>90", 3, 90),
          ("L4 三个全>95", 3, 95)]

for tag in ("A", "B"):
    pool = pd.read_parquet(f"{SP}/rps_pool_{tag}.parquet")
    snaps = sorted(pool.snap.unique())
    print(f"\n{'='*114}\n股池 {tag}  RPS 规则阶梯归因\n{'='*114}")
    acc = {k: {"r": [], "n": []} for k, _, _ in LEVELS}
    acc.update({f"{k}+双增长": {"r": [], "n": []} for k, _, _ in LEVELS})
    acc["L5 用户实际池子"] = {"r": [], "n": []}
    acc["L5 用户池·双增长"] = {"r": [], "n": []}
    acc["全市场等权"] = {"r": [], "n": []}
    days = []

    for i in range(len(snaps) - 1):
        s, s2 = snaps[i], snaps[i + 1]
        e = idx.searchsorted(pd.Timestamp(s), side="right")
        x = idx.searchsorted(pd.Timestamp(s2), side="right")
        if e >= len(idx) or x >= len(idx) or x <= e or e < 251:
            continue
        p = e - 1                                   # 快照日当天可得
        alive = np.isfinite(OP.iloc[e]) & np.isfinite(OP.iloc[x])
        cnt90 = sum((R[n].iloc[p] > 90).fillna(False).astype(int) for n in (50, 120, 250))
        cnt95 = sum((R[n].iloc[p] > 95).fillna(False).astype(int) for n in (50, 120, 250))
        dual = (NIY.iloc[p] > 0).fillna(False) & (REVY.iloc[p] > 0).fillna(False)
        g = pool[pool.snap == s]
        days.append(x - e)

        for name, k, th in LEVELS:
            m = ((cnt95 if th == 95 else cnt90) >= k) & alive
            codes = OP.columns[m]
            r = rets(codes, e, x)
            acc[name]["r"].append(r.mean() if len(r) else np.nan)
            acc[name]["n"].append(len(codes))
            codes2 = OP.columns[m & dual]
            r2 = rets(codes2, e, x)
            acc[f"{name}+双增长"]["r"].append(r2.mean() if len(r2) else np.nan)
            acc[f"{name}+双增长"]["n"].append(len(codes2))

        r = rets(g.code.to_numpy(), e, x)
        acc["L5 用户实际池子"]["r"].append(r.mean() if len(r) else np.nan)
        acc["L5 用户实际池子"]["n"].append(len(g))
        gd = g[g.dual]
        r = rets(gd.code.to_numpy(), e, x)
        acc["L5 用户池·双增长"]["r"].append(r.mean() if len(r) else np.nan)
        acc["L5 用户池·双增长"]["n"].append(len(gd))
        av = OP.columns[alive]
        acc["全市场等权"]["r"].append((OP.iloc[x][av] / OP.iloc[e][av] - 1).mean())
        acc["全市场等权"]["n"].append(len(av))

    days = np.array(days)
    print(f"{'口径':<32}{'每期只数':>10}{'年化':>11}{'逐期均值':>11}{'相对等权':>12}")
    base = comp(np.array(acc["全市场等权"]["r"]) - 2 * COST * 1.0, days)
    order = [k for k, _, _ in LEVELS] + [f"{k}+双增长" for k, _, _ in LEVELS] \
        + ["L5 用户实际池子", "L5 用户池·双增长", "全市场等权"]
    for name in order:
        r = np.array(acc[name]["r"], float)
        c = 2 * COST * (1.0 if name == "全市场等权" else TURN)
        a = comp(r - c, days)
        ex = "" if name == "全市场等权" else f"{(a-base)*100:>+11.1f}pp"
        print(f"{name:<32}{np.median(acc[name]['n']):>10.0f}{a:>+11.2%}"
              f"{np.nanmean(r-c):>+11.3%}{ex:>12}")
    print(f"  ({time.time()-t0:.0f}s)")

print(f"\n耗时 {time.time()-t0:.0f}s")
