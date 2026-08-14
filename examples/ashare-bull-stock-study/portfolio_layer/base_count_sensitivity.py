"""基地计数的参数敏感性:孤峰还是高原(第七十二节)

═══ 这不是检验 ═══
§71 的 C 规则(基地计数)超额 +3.0%、p=0.010,四个参数事前锁定、未做搜索。
但那只是九宫格里的**一格**。本节把网格摊开,**目的不是找更好的参数**,
而是回答一个只有两种答案的问题:

  **孤峰** —— 只有 (3, 15%) 那一格好,周围全塌  → +3.0% 大概率是运气
  **高原** —— 周围多数格子同向且显著            → 是结构,不是运气

**本节不产生新结论,不用于支持任何主张。**
若结果是高原,§71 的结论**维持原样**(不因此增强);
若结果是孤峰,§71 的 C 部分**降级为待复核**。

═══ 事前写死「怎么读」(免得事后合理化) ═══
  高原  九格中 **≥7 格超额为正** 且 **≥5 格 p<0.05**
  孤峰  九格中 <5 格超额为正,或仅 (3,15%) 一格 p<0.05
  中间态 两者都不满足 → 记为「不确定」,不向任何一边解释

═══ 网格(只动两个参数,另两个保持 §71 原值) ═══
  N_BASE   ∈ {2, 3, 4}        完成几个基地之后才允许卖
  FAIL_DD  ∈ {10%, 15%, 20%}  之后自最高点回撤多少即卖
  BASE_MIN = 25 交易日、BASE_MAXDD = 35% —— **保持 §71 原值不动**

═══ 顺带补上 §71 的实现疏漏 ═══
§71 限定⑤ 写了:「没有记录各规则的实际持有天数,『C 是慢规则』靠峰值倒推」。
**本节直接记录持有天数**,把那个推断变成测量。

═══ 其余设定与 §70/§71 完全一致(不重调) ═══
  红轴入场、次月首日开盘、50 只等权、200 次自助、单边 0.3%、
  最长 750 日、同市值五分位对照

═══ 锚点 ═══
  面板 3,297 × 5,232
  (3, 15%) 这一格必须复现 §71 的 C:实收 +25.3%/笔、超额 +3.0%、p=0.010
  **对不上则本节实现有错,整表作废。**
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
NPICK, NBOOT, SEED, COST = 50, 200, 20260814, 0.003
MAXHOLD = 750
Y_LO, Y_HI, NQ = 365, 1095, 5
BASE_MIN, BASE_MAXDD = 25, 0.35                 # §71 原值,不动
NB_GRID, DD_GRID = (2, 3, 4), (0.10, 0.15, 0.20)
CFG = [(n, d) for n in NB_GRID for d in DD_GRID]
NC = len(CFG)

t0 = time.time()
op, cl, ld, mv = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "close", "listed_days", "float_mv"])
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
OP = pd.DataFrame(op).sort_index()
OP.index = OP.index.tz_localize(None)
CL = pd.DataFrame(cl).set_axis(OP.index)
LD = pd.DataFrame(ld).set_axis(OP.index)
MV = pd.DataFrame(mv).set_axis(OP.index)
OP, CL = OP.where(OP > 0), CL.where(CL > 0)
idx = OP.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

OPa, CLa, LDa, MVa = (OP.to_numpy(float), CL.to_numpy(float),
                      LD.to_numpy(float), MV.to_numpy(float))
ALIVE = np.isfinite(CLa) & (CLa > 0)
CLf = pd.DataFrame(CLa).ffill().to_numpy(float)
OPf = pd.DataFrame(OPa).ffill().to_numpy(float)

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
first_td = {p: int(np.flatnonzero(ym == p)[0]) for p in ym.unique()}
allm = sorted(last_td)
mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
mkc = mk["close"].reindex(idx).ffill()
mm = mkc.resample("ME").last()
d = mm.ewm(span=12, adjust=False).mean() - mm.ewm(span=26, adjust=False).mean()
ih = (d - d.ewm(span=9, adjust=False).mean())
ih.index = ih.index.to_period("M")
reg = {p: int(v > 0) for p, v in ih.items()}
red = [p for p in allm if reg.get(p, 0) == 1 and last_td[p] + 250 < NT]
print(f"红轴月 {len(red)} 个  ({time.time()-t0:.0f}s)")


def run_all(j, e):
    """一次遍历同时结算九组参数。返回 (实收[9], 峰值[9], 持有天数[9])。"""
    entry = OPf[e, j]
    out = np.full((NC, 3), np.nan)
    if not np.isfinite(entry) or entry <= 0:
        return out
    end = min(e + MAXHOLD, NT - 1)
    peak, nbase, below = entry, 0, 0
    live = np.ones(NC, dtype=bool)
    pk_at = np.full(NC, entry)
    for t in range(e, end + 1):
        c = CLf[t, j]
        if not np.isfinite(c):
            continue
        if c > peak:
            if below >= BASE_MIN:
                nbase += 1
            below = 0
        else:
            below += 1
        peak = max(peak, c)
        pk_at[live] = peak
        dd = c / peak - 1
        if t > e:
            for k, (n, ddx) in enumerate(CFG):
                if live[k] and (dd <= -BASE_MAXDD or (nbase >= n and dd <= -ddx)):
                    t1 = min(t + 1, NT - 1)
                    px = OPf[t1, j] if np.isfinite(OPf[t1, j]) else c
                    out[k] = (px / entry - 1, pk_at[k] / entry - 1, t - e)
                    live[k] = False
            if not live.any():
                return out
    for k in np.flatnonzero(live):
        out[k] = (CLf[end, j] / entry - 1, pk_at[k] / entry - 1, end - e)
    return out


mon = []
for p in red:
    i = allm.index(p)
    e = first_td[allm[i + 1]] if i + 1 < len(allm) else None
    if e is None or e >= NT - 5:
        continue
    t = last_td[p]
    base = ALIVE[t] & np.isfinite(OPa[e]) & (OPa[e] > 0)
    pool = np.flatnonzero(base & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI))
    if len(pool) < NPICK:
        continue
    m = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(m[base], np.linspace(0, 1, NQ + 1)[1:-1])
    ctrl, rng0 = [], np.random.default_rng(SEED + hash(str(p)) % 9999)
    for i2 in range(NQ):
        lo = -np.inf if i2 == 0 else q[i2 - 1]
        hi = np.inf if i2 == NQ - 1 else q[i2]
        band = np.flatnonzero(base & (m > lo) & (m <= hi))
        npool = int(np.sum((m[pool] > lo) & (m[pool] <= hi)))
        if npool and len(band) >= npool:
            ctrl.append(rng0.choice(band, npool, replace=False))
    ctrl = np.concatenate(ctrl) if ctrl else pool
    mon.append((np.array([run_all(j, e) for j in pool]),
                np.array([run_all(j, e) for j in ctrl])))
    if len(mon) % 15 == 0:
        print(f"  {p}  池 {len(pool)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"完成 {len(mon)} 个红轴月  ({time.time()-t0:.0f}s)")

rng = np.random.default_rng(SEED)
print(f"\n{'='*104}\n基地计数敏感性(**这不是检验**;红轴入场,50 只等权,{NBOOT} 次自助)\n{'='*104}")
print(f"{'基地数':<8}{'失败回撤':<10}{'实收/笔':>10}{'峰值/笔':>10}{'兑现率':>8}"
      f"{'持有天':>8}{'组合':>9}{'对照':>9}{'超额':>9}{'p':>8}")
rows = []
for k, (n, dd) in enumerate(CFG):
    pr = [a[:, k, 0][np.isfinite(a[:, k, 0])] for a, _ in mon]
    pk = [a[:, k, 1][np.isfinite(a[:, k, 1])] for a, _ in mon]
    hd = [a[:, k, 2][np.isfinite(a[:, k, 2])] for a, _ in mon]
    cr = [b[:, k, 0][np.isfinite(b[:, k, 0])] for _, b in mon]
    bp = [np.mean([np.mean(rng.choice(a, NPICK)) - COST for a in pr if len(a) >= 10])
          for _ in range(NBOOT)]
    bc = [np.mean([np.mean(rng.choice(b, NPICK)) - COST for b in cr if len(b) >= 10])
          for _ in range(NBOOT)]
    ap, ac = float(np.median(bp)), float(np.median(bc))
    pv = float((np.array(bc) >= ap).mean())
    real = float(np.mean(np.concatenate(pr)))
    peak = float(np.mean(np.concatenate(pk)))
    days = float(np.mean(np.concatenate(hd)))
    rate = real / peak if peak > 0 else np.nan
    rows.append({"基地数": n, "失败回撤": dd, "实收每笔": real, "峰值每笔": peak,
                 "兑现率": rate, "持有天": days, "组合": ap, "对照": ac,
                 "超额": ap - ac, "p": pv})
    print(f"{n:<8}{f'{dd:.0%}':<10}{real:>+10.1%}{peak:>+10.1%}{rate:>8.0%}"
          f"{days:>8.0f}{ap:>+9.1%}{ac:>+9.1%}{ap-ac:>+9.1%}{pv:>8.3f}")

D = pd.DataFrame(rows)
c = D[(D["基地数"] == 3) & (D["失败回撤"] == 0.15)].iloc[0]
ok_anchor = abs(c["实收每笔"] - 0.253) < 0.01 and abs(c["超额"] - 0.030) < 0.01
npos = int((D["超额"] > 0).sum())
nsig = int((D["p"] < 0.05).sum())
only_center = nsig == 1 and c["p"] < 0.05

print(f"\n{'='*104}\n事前写死的读法(不放宽)\n{'='*104}")
print(f"  锚点:(3, 15%) 复现 §71 的 C(实收 +25.3%、超额 +3.0%)      "
      f"{c['实收每笔']:+.1%} / {c['超额']:+.1%}   {'✓' if ok_anchor else '✗'}")
if not ok_anchor:
    print("\n  **锚点对不上,本表作废。**")
else:
    print(f"  九格中超额为正:{npos}/9      p<0.05:{nsig}/9")
    if npos >= 7 and nsig >= 5:
        v = "高原 —— 是结构,不是运气。§71 的结论**维持原样,不因此增强**。"
    elif npos < 5 or only_center:
        v = "孤峰 —— §71 的 C 部分**降级为待复核**。"
    else:
        v = "不确定 —— 两个读法都不满足,**不向任何一边解释**。"
    print(f"\n  **判读:{v}**")

print(f"\n  **持有天数(补 §71 限定⑤ 的疏漏):** "
      f"{D['持有天'].min():.0f} ~ {D['持有天'].max():.0f} 天,"
      f"(3,15%) 为 {c['持有天']:.0f} 天")
D.to_csv(f"{SP}/base_count_sensitivity.csv", index=False)
print(f"\n→ {SP}/base_count_sensitivity.csv   ({time.time()-t0:.0f}s)")
