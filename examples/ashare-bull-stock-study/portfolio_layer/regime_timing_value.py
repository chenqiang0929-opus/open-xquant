"""红轴择时到底值不值钱:收益归因的四路分解(第七十三节)

═══ 起因:这是整套系统最大的空洞 ═══
`TRADING_SYSTEM.md` 第 5 章列的**第一条未验证假设**:

  「红轴择时本身是否优于买入持有 —— 整套系统一半以上的收益压在这上面,
    而它从未被单独检验过。红轴期间市场本来就在涨,
    『红轴 +30.1%』里有多少只是牛市 beta?」

§71 测的是「同一段红轴内,不同离场规则的**相对**优劣」,对照组也在红轴内,
**所以 beta 被约掉了,择时本身的价值一次也没量过。**
在这个数出来之前,整套系统的收益归因不成立。

═══ 四路分解(唯一变量法) ═══
所有策略共用**同一份换仓日历** —— 每次红/绿轴切换确认后的次月首日。
**唯一的区别是那一段持什么。**

  S0  沪深300 全程持有                    ← 最朴素的基准
  S1  次新池 50 只,**全程持有不择时**      ← 红轴绿轴都在场
  S2  次新池 50 只,**红轴持有绿轴空仓**    ← 本系统
  S3  全市场随机 50 只,全程持有            ← 剥离「次新」这个因素

分解式:
  **择时贡献 = S2 − S1**   (同一个池子,只差绿轴要不要空仓)
  **池子贡献 = S1 − S3**   (同样不择时,只差池子是次新还是全市场)
  **全系统 vs 买指数 = S2 − S0**

═══ 与 §70-72 的关键区别 ═══
本节构建**连续资金曲线**,不是「每一批的平均结果」——
这同时补上 `TRADING_SYSTEM.md` 第 5 章限定 3(资金曲线未构建)。
可以报出**最大回撤**,那是前面所有节都给不出来的数。

═══ 事前锁定(不搜索、不调参) ═══
  闸门    510300 月线 MACD(12,26,9) 柱正负,**月末确认、次月首日执行**
          (§71 修正口径,不用 §70 那个含前视的版本)
  池子    次新 = 段起点当月末 listed_days ∈ [365,1095) 且在市
  持仓    段起点随机 50 只等权,段内不动;**200 个种子**
  成本    每次换仓单边 0.3%(买卖各一次)
  区间    2013-01-04 ~ 2026-08-03(面板全程)
  绿轴    S2 持现金,**收益记 0**(不算利息,保守)

═══ 事前判据(跑之前写死,不放宽) ═══
  ① **择时有正价值**:S2 年化 > S1 年化
  ② **跑赢简单买指数**:S2 年化 > S0 年化
  ③ **择时改善回撤**:S2 最大回撤 < S1 最大回撤
  ④ **池子有正价值**:S1 年化 > S3 年化

  ①③ 都过 → 红轴择时的价值成立,系统的收益归因站得住
  ① 不过   → **「红轴 +30.1%」基本是 beta,择时不创造收益**
              (③ 若仍过,则择时的价值只在回撤,不在收益 —— 那也是有用的结论)
  ② 不过   → 整套系统不如直接买沪深300,**不应上线**

**事前预测(写下以便被证伪)**:
我预计 **① 勉强通过或不通过、③ 明显通过、② 通过、④ 通过**。
理由:红轴占 49.7% 的月份且覆盖了全部牛市,空仓避开的主要是下跌,
**这在回撤上一定有效,在年化上则要看绿轴期间踏空了多少反弹。**
**若 ① 明显通过(S2 比 S1 高 5pp 以上),说明择时本身就是主要收益来源,我低估了它。**

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
  红轴月 82 个(§71 报 68 个是因为它要求每月后有完整 250 日前瞻窗口,
  砍掉了最后约 14 个月;本节不需要前瞻窗口,用全部 82 个)
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
NPICK, NSEED, SEED, COST = 50, 200, 20260814, 0.003
Y_LO, Y_HI = 365, 1095

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

# ── 换仓日历:月末确认、次月首日执行(无前视) ──
segs = []           # (起日, 止日, 该段是否红轴)
cur, start = None, None
for i, p in enumerate(allm[:-1]):
    if p not in reg:
        continue
    nxt = allm[i + 1]
    e = first_td[nxt]
    r = reg[p]
    if cur is None:
        cur, start = r, e
    elif r != cur:
        segs.append((start, e, cur))
        cur, start = r, e
if start is not None and start < NT - 1:
    segs.append((start, NT - 1, cur))
nred = sum(1 for _, _, r in segs if r)
print(f"分段 {len(segs)} 段(红 {nred} / 绿 {len(segs)-nred}),"
      f"红轴月 {sum(reg[p] for p in allm if p in reg)} 个  ({time.time()-t0:.0f}s)")
for s, e, r in segs:
    print(f"    {'红' if r else '绿'}  {idx[s].date()} ~ {idx[e].date()}"
          f"  ({e-s} 个交易日)")

rng = np.random.default_rng(SEED)


def seg_path(picks, s, e):
    """一段内 50 只等权的日度净值路径(含两端各 0.3% 成本)。"""
    if len(picks) == 0:
        return np.ones(e - s + 1)
    ent = OPf[s, picks]
    ok = np.isfinite(ent) & (ent > 0)
    picks, ent = picks[ok], ent[ok]
    if len(picks) == 0:
        return np.ones(e - s + 1)
    px = CLf[s:e + 1][:, picks] / ent
    px = np.where(np.isfinite(px), px, 1.0)
    path = px.mean(axis=1) * (1 - COST)
    path[-1] *= (1 - COST)
    return path


def build(pool_fn, timing):
    """pool_fn(段起点)->候选下标;timing=True 表示绿轴空仓。返回 (日度净值[NT], 年化)"""
    eq = np.ones(NT)
    v = 1.0
    for s, e, r in segs:
        if timing and not r:
            eq[s:e + 1] = v
            continue
        cand = pool_fn(s)
        picks = (rng.choice(cand, min(NPICK, len(cand)), replace=False)
                 if len(cand) else np.array([], dtype=int))
        path = seg_path(picks, s, e)
        eq[s:e + 1] = v * path
        v = eq[e]
    eq[:segs[0][0]] = 1.0
    return eq


def pool_new(s):
    t = s - 1
    return np.flatnonzero(ALIVE[t] & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI)
                          & np.isfinite(OPf[s]) & (OPf[s] > 0))


def pool_all(s):
    t = s - 1
    return np.flatnonzero(ALIVE[t] & np.isfinite(OPf[s]) & (OPf[s] > 0))


yrs = (idx[-1] - idx[segs[0][0]]).days / 365.25
print(f"\n计算区间 {idx[segs[0][0]].date()} ~ {idx[-1].date()}  {yrs:.2f} 年")

CURVES = {}
for nm, fn, tm in (("S1 次新池不择时", pool_new, False),
                   ("S2 次新池+红轴择时", pool_new, True),
                   ("S3 全市场不择时", pool_all, False)):
    E = np.array([build(fn, tm) for _ in range(NSEED)])
    CURVES[nm] = E
    print(f"  {nm} 完成 {NSEED} 种子  ({time.time()-t0:.0f}s)", flush=True)

s0 = IDX / IDX[segs[0][0]]
s0[:segs[0][0]] = 1.0
CURVES["S0 沪深300持有"] = s0[None, :]


def mdd(e):
    return float((e / np.maximum.accumulate(e) - 1).min())


print(f"\n{'='*100}\n四路分解:连续资金曲线({NSEED} 种子中位,{yrs:.1f} 年)\n{'='*100}")
print(f"{'策略':<22}{'期末净值':>10}{'年化':>9}{'最大回撤':>10}"
      f"{'年化区间(5%~95%)':>22}")
R = {}
for nm in ("S0 沪深300持有", "S1 次新池不择时", "S2 次新池+红轴择时", "S3 全市场不择时"):
    E = CURVES[nm]
    fin = E[:, -1]
    ann = fin ** (1 / yrs) - 1
    md = np.array([mdd(e) for e in E])
    R[nm] = (float(np.median(fin)), float(np.median(ann)), float(np.median(md)))
    lo, hi = np.percentile(ann, [5, 95])
    print(f"{nm:<22}{np.median(fin):>10.2f}{np.median(ann):>+9.2%}"
          f"{np.median(md):>10.1%}{f'[{lo:+.1%}, {hi:+.1%}]':>22}")

a0, a1, a2, a3 = (R["S0 沪深300持有"][1], R["S1 次新池不择时"][1],
                  R["S2 次新池+红轴择时"][1], R["S3 全市场不择时"][1])
m1, m2 = R["S1 次新池不择时"][2], R["S2 次新池+红轴择时"][2]

print(f"\n{'='*100}\n收益归因分解\n{'='*100}")
print(f"  **择时贡献** = S2 − S1 = {a2:+.2%} − {a1:+.2%} = **{a2-a1:+.2%}**")
print(f"  **池子贡献** = S1 − S3 = {a1:+.2%} − {a3:+.2%} = **{a1-a3:+.2%}**")
print(f"  **全系统 vs 买指数** = S2 − S0 = {a2:+.2%} − {a0:+.2%} = **{a2-a0:+.2%}**")
print(f"  **回撤改善** = S1 → S2 = {m1:.1%} → {m2:.1%} = **{(m2-m1)*100:+.1f}pp**")
print(f"  **Calmar(年化/最大回撤)**  S1 {a1/abs(m1):.3f}   S2 {a2/abs(m2):.3f}"
      f"   S3 {a3/abs(R['S3 全市场不择时'][2]):.3f}"
      f"   S0 {a0/abs(R['S0 沪深300持有'][2]):.3f}")
print("\n  **逐层归因(从买指数走到本系统):**")
print(f"    沪深300 持有                {a0:+.2%}")
print(f"    → 全市场等权随机 50 只       {a3:+.2%}   (+{(a3-a0)*100:.2f}pp  等权/小市值溢价)")
print(f"    → 换成次新池                {a1:+.2%}   (+{(a1-a3)*100:.2f}pp  §69 年龄效应)")
print(f"    → 加红轴择时                {a2:+.2%}   ({(a2-a1)*100:+.2f}pp  择时)")

print(f"\n{'='*100}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*100}")
c1, c2, c3, c4 = a2 > a1, a2 > a0, m2 > m1, a1 > a3
print(f"  ① 择时有正价值   S2 > S1        {a2:+.2%} vs {a1:+.2%}   {'✓' if c1 else '✗'}")
print(f"  ② 跑赢买指数     S2 > S0        {a2:+.2%} vs {a0:+.2%}   {'✓' if c2 else '✗'}")
print(f"  ③ 择时改善回撤   |S2| < |S1|    {m2:.1%} vs {m1:.1%}   {'✓' if c3 else '✗'}")
print(f"  ④ 池子有正价值   S1 > S3        {a1:+.2%} vs {a3:+.2%}   {'✓' if c4 else '✗'}")
if c1 and c3:
    v = "红轴择时的价值成立,系统的收益归因站得住"
elif c3:
    v = "**择时的价值只在回撤,不在收益** —— 「红轴 +30.1%」基本是 beta"
else:
    v = "**红轴择时不创造价值**"
print(f"\n  **结论:{v}**")
if not c2:
    print("  **② 不过:整套系统不如直接买沪深300,不应上线。**")

out = pd.DataFrame({nm: np.median(CURVES[nm], axis=0) for nm in CURVES},
                   index=idx)
out.to_csv(f"{SP}/regime_timing_value.csv")
print(f"\n→ {SP}/regime_timing_value.csv   ({time.time()-t0:.0f}s)")
