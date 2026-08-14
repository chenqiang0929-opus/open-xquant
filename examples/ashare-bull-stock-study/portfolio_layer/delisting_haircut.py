"""退市股 ffill 对资金曲线的高估有多大(第七十四节)

═══ 起因:这是我自己交付里的缺陷 ═══
§73 给出了年化 +10.81% / 最大回撤 −53.6%,并写进了 `TRADING_SYSTEM.md` v0.2。
**然后才发现那个数偏高。**

全研究一贯的处理是「退市股按最后有效价前向填充,**不剔除**」。
在**密度**检验(§67-69)里这是**对的** —— 剔除退市股就是幸存者偏差,
那正是这个研究一直在批判的东西。

**但在资金曲线里,它等于「退市股仓位冻结,不亏钱」。**
现实中退市要么进三板打三折、要么接近归零。
**§73 的 S1/S2/S3 年化因此全部偏高,幅度未量化。**

本节量它。

═══ 三种口径(同一份持仓、同一份换仓日历,只改退市后的估值) ═══
  T0  ffill 冻结          §73 现状 —— 退市后价格不动
  T2  三板折价 30%        退市后按最后有效价 × 0.30 计
  T1  归零                退市后按 0 计 ——**最保守**

═══ 退市的判定(事前写死) ═══
  某标的最后一个有效收盘在 **面板末日往前 20 个交易日之前**,
  且此后再无有效价 → 判定为在该日**退市**。
  20 日的缓冲是为了不把「面板末尾的停牌」误判成退市。
  **报出判定到的退市只数与占比,供核对。**

═══ 事前判据(跑之前写死,不放宽) ═══
  ① **最保守口径下仍跑赢指数**:T1(归零)下 S2 年化 > S0 的 +5.80%
  ② **§73 主结论稳健**:三种口径下「择时贡献 = S2 − S1」**符号一致**(都为负)
  ③ **§69 效应稳健**:三种口径下「池子贡献 = S1 − S3」**都为正**

  ①②③ 全过 → §73 的结论方向不变,只需把年化数字下调
  ① 不过   → **系统在保守口径下不如买指数,TRADING_SYSTEM.md 应标记不可上线**
  ③ 不过   → **次新池的优势有一部分是退市股冻结造成的假象** —— 那是严重问题

**事前预测(写下以便被证伪)**:我预计 ①②③ 全过,且
**T1 对 S3(全市场)的打击大于对 S1(次新池)** ——
因为退市通常发生在上市多年的问题公司身上,而次新池按定义只有 1-3 年。
**若果真如此,「池子贡献 +2.59pp」在保守口径下会**变大**而不是变小,
即 §73 低估了池子效应。若相反,说明我完全想错了方向。**

═══ 锚点 ═══
  T0 口径必须逐位复现 §73:S1 +11.51% / S2 +10.81% / S3 +8.93%
  **对不上则本节实现有错,整表作废。**
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
NPICK, NSEED, SEED, COST = 50, 200, 20260814, 0.003
Y_LO, Y_HI, TAIL_BUF = 365, 1095, 20
HAIRCUT = {"T0 ffill 冻结(§73 现状)": None, "T2 三板折价 30%": 0.30, "T1 归零": 0.0}

t0 = time.time()
op, cl, ld = {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "close", "listed_days"])
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
OP = pd.DataFrame(op).sort_index()
OP.index = OP.index.tz_localize(None)
CL = pd.DataFrame(cl).set_axis(OP.index)
LD = pd.DataFrame(ld).set_axis(OP.index)
OP, CL = OP.where(OP > 0), CL.where(CL > 0)
idx = OP.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

CLa, LDa = CL.to_numpy(float), LD.to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
CLf = pd.DataFrame(CLa).ffill().to_numpy(float)
OPf = pd.DataFrame(OP.to_numpy(float)).ffill().to_numpy(float)

# ── 退市判定 ──
has = ALIVE.any(axis=0)
lastv = np.where(has, ALIVE.shape[0] - 1 - np.argmax(ALIVE[::-1], axis=0), -1)
delisted = has & (lastv < NT - 1 - TAIL_BUF)
DEL_T = np.where(delisted, lastv + 1, NT + 1)      # 退市生效日
print(f"判定退市 {delisted.sum()} 只 / {has.sum()} 只有数据的标的 "
      f"({delisted.sum()/max(has.sum(),1):.1%})  ({time.time()-t0:.0f}s)")
yr = pd.Series(idx[np.clip(lastv[delisted], 0, NT - 1)]).dt.year.value_counts().sort_index()
print("  退市年份分布:", ", ".join(f"{y}:{n}" for y, n in yr.items()))

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
first_td = {p: int(np.flatnonzero(ym == p)[0]) for p in ym.unique()}
allm = sorted(last_td)
mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
mkc = mk["close"].reindex(idx).ffill()
IDX = mkc.to_numpy(float)
mm = mkc.resample("ME").last()
d = mm.ewm(span=12, adjust=False).mean() - mm.ewm(span=26, adjust=False).mean()
ih = d - d.ewm(span=9, adjust=False).mean()
ih.index = ih.index.to_period("M")
reg = {p: int(v > 0) for p, v in ih.items() if p in last_td}

segs, cur, start = [], None, None
for i, p in enumerate(allm[:-1]):
    if p not in reg:
        continue
    e, r = first_td[allm[i + 1]], reg[p]
    if cur is None:
        cur, start = r, e
    elif r != cur:
        segs.append((start, e, cur))
        cur, start = r, e
if start is not None and start < NT - 1:
    segs.append((start, NT - 1, cur))
print(f"分段 {len(segs)} 段(红 {sum(r for _,_,r in segs)})  ({time.time()-t0:.0f}s)")


def seg_path(picks, s, e, hc):
    if len(picks) == 0:
        return np.ones(e - s + 1)
    ent = OPf[s, picks]
    ok = np.isfinite(ent) & (ent > 0)
    picks, ent = picks[ok], ent[ok]
    if len(picks) == 0:
        return np.ones(e - s + 1)
    px = CLf[s:e + 1][:, picks].copy()
    if hc is not None:
        dt = DEL_T[picks]
        days = np.arange(s, e + 1)[:, None]
        px = np.where(days >= dt[None, :], px * hc, px)
    px = px / ent
    px = np.where(np.isfinite(px), px, 1.0)
    path = px.mean(axis=1) * (1 - COST)
    path[-1] *= (1 - COST)
    return path


def pool_new(s):
    t = s - 1
    return np.flatnonzero(ALIVE[t] & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI)
                          & np.isfinite(OPf[s]) & (OPf[s] > 0))


def pool_all(s):
    t = s - 1
    return np.flatnonzero(ALIVE[t] & np.isfinite(OPf[s]) & (OPf[s] > 0))


def build(pool_fn, timing, hc, rng):
    eq, v = np.ones(NT), 1.0
    for s, e, r in segs:
        if timing and not r:
            eq[s:e + 1] = v
            continue
        cand = pool_fn(s)
        picks = (rng.choice(cand, min(NPICK, len(cand)), replace=False)
                 if len(cand) else np.array([], dtype=int))
        eq[s:e + 1] = v * seg_path(picks, s, e, hc)
        v = eq[e]
    eq[:segs[0][0]] = 1.0
    return eq


def mdd(e):
    return float((e / np.maximum.accumulate(e) - 1).min())


yrs = (idx[-1] - idx[segs[0][0]]).days / 365.25
STRAT = (("S1 次新池不择时", pool_new, False),
         ("S2 次新池+红轴择时", pool_new, True),
         ("S3 全市场不择时", pool_all, False))
a0 = float((IDX[-1] / IDX[segs[0][0]]) ** (1 / yrs) - 1)

RES = {}
for hn, hc in HAIRCUT.items():
    rng = np.random.default_rng(SEED)         # 每种口径用同一组种子 → 持仓完全相同
    for nm, fn, tm in STRAT:
        E = np.array([build(fn, tm, hc, rng) for _ in range(NSEED)])
        ann = E[:, -1] ** (1 / yrs) - 1
        RES[(hn, nm)] = (float(np.median(ann)),
                         float(np.median([mdd(e) for e in E])))
    print(f"  {hn} 完成  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n{'='*104}\n三种退市口径下的四路分解({NSEED} 种子中位,{yrs:.1f} 年)\n{'='*104}")
print(f"{'退市口径':<24}{'S1 不择时':>14}{'S2 本系统':>14}{'S3 全市场':>14}"
      f"{'择时贡献':>12}{'池子贡献':>12}")
for hn in HAIRCUT:
    a1, a2, a3 = (RES[(hn, "S1 次新池不择时")][0], RES[(hn, "S2 次新池+红轴择时")][0],
                  RES[(hn, "S3 全市场不择时")][0])
    print(f"{hn:<24}{a1:>+14.2%}{a2:>+14.2%}{a3:>+14.2%}"
          f"{a2-a1:>+12.2f}{a1-a3:>+12.2f}".replace("+0.0", "+0.0"))
print(f"\n{'退市口径':<24}{'S1 回撤':>14}{'S2 回撤':>14}{'S3 回撤':>14}")
for hn in HAIRCUT:
    print(f"{hn:<24}" + "".join(
        f"{RES[(hn, nm)][1]:>14.1%}" for nm, _, _ in STRAT))
print(f"\n  沪深300 全程持有 年化 {a0:+.2%}(不受退市口径影响)")

print(f"\n{'='*104}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*104}")
t0n = "T0 ffill 冻结(§73 现状)"
anc = RES[(t0n, "S1 次新池不择时")][0], RES[(t0n, "S2 次新池+红轴择时")][0], \
      RES[(t0n, "S3 全市场不择时")][0]
ok_anc = (abs(anc[0] - 0.1151) < 0.005 and abs(anc[1] - 0.1081) < 0.005
          and abs(anc[2] - 0.0893) < 0.005)
print(f"  锚点 T0 复现 §73(+11.51/+10.81/+8.93)   "
      f"{anc[0]:+.2%}/{anc[1]:+.2%}/{anc[2]:+.2%}   {'✓' if ok_anc else '✗'}")
if not ok_anc:
    print("\n  **锚点对不上,本表作废。**")
else:
    a2z = RES[("T1 归零", "S2 次新池+红轴择时")][0]
    c1 = a2z > a0
    tim = [RES[(h, "S2 次新池+红轴择时")][0] - RES[(h, "S1 次新池不择时")][0]
           for h in HAIRCUT]
    poo = [RES[(h, "S1 次新池不择时")][0] - RES[(h, "S3 全市场不择时")][0]
           for h in HAIRCUT]
    c2 = all(x < 0 for x in tim) or all(x > 0 for x in tim)
    c3 = all(x > 0 for x in poo)
    print(f"  ① 归零口径下 S2 > 沪深300           {a2z:+.2%} vs {a0:+.2%}   "
          f"{'✓' if c1 else '✗'}")
    print(f"  ② 择时贡献三口径符号一致            "
          + " / ".join(f"{x:+.2%}" for x in tim) + f"   {'✓' if c2 else '✗'}")
    print(f"  ③ 池子贡献三口径都为正              "
          + " / ".join(f"{x:+.2%}" for x in poo) + f"   {'✓' if c3 else '✗'}")
    print(f"\n  **§73 的年化高估幅度:** S2 {anc[1]:+.2%} → 归零口径 {a2z:+.2%} "
          f"= **{(a2z-anc[1])*100:+.2f}pp**")
    if c1 and c2 and c3:
        print("\n  **结论:§73 的方向全部不变,只需把年化数字按上表下调。**")
    elif not c1:
        print("\n  **结论:① 不过 —— 保守口径下系统不如买指数,"
              "TRADING_SYSTEM.md 应标记不可上线。**")
    else:
        print("\n  **结论:有判据不过,见上。**")
    if poo[2] > poo[0]:
        print(f"  **注意:池子贡献在归零口径下**变大**({poo[0]:+.2%} → {poo[2]:+.2%})"
              f" —— 退市集中在老公司,§73 低估了次新池的优势。**")

pd.DataFrame([{"退市口径": h, "策略": n, "年化": RES[(h, n)][0],
               "最大回撤": RES[(h, n)][1]}
              for h in HAIRCUT for n, _, _ in STRAT]).to_csv(
    f"{SP}/delisting_haircut.csv", index=False)
print(f"\n→ {SP}/delisting_haircut.csv   ({time.time()-t0:.0f}s)")
