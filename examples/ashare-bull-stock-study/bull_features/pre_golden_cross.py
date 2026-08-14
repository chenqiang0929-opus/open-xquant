"""「金叉前夜」形态:长期下跌后的第一次放量突破(第七十五节)

═══ 起因:我先前把两个相反的形态当成了一个 ═══
我在 §67-74 里反复说「均线多头排列是负 alpha,§57/62/63 三次确认」。
**但 §57/62/63 测的是「已经多头排列」(MA100 > MA300)。**

用户拿生益电子(688183)@2024-05 做例子,面板核对下来:

    RPS50           **99.7**(涨幅 +64.8%,全市场前 0.3%)   ✓
    量比 20/60      **1.50**                              ✓ 放量
    收盘 vs 20周线  14.49 vs 9.69 = **+49.5%**            ✓ 已站上且大幅偏离
    20周线 vs 60周线 9.69 vs 10.74 = **−9.8%**            ← **仍是空头排列!**
    距 250 日新高   **−8.6%**                             ✗ 并未创新高
    上市年数        3.26 年                                (刚超出 [1,3) 窗口)
    后续            2024-05-31 收 14.49 → 峰值 141.07 = **+873.6%**

**这是「金叉前夜」——均线还没穿,但价格已经放量拉起。**
**它是转折点形态,不是趋势确立形态。两者方向相反,而我把它们当成了一个东西。**
**本节测的是我没测过的那一个。**

═══ 事件定义(事前锁定,不搜索、不调参) ═══
在交易日 t,同时满足下列全部条件即触发:

  ① RPS50(t) ≥ 90              50 日涨幅的全市场百分位(欧奈尔口径)
  ② close(t) ≥ MA100(t) × 1.20 已站上 20 周线且偏离 ≥20%
  ③ MA100(t) < MA300(t)        **仍是空头排列 —— 本节的核心条件**
     且 MA100/MA300 在最近 20 日**上升**(差距在收敛 = 正在逼近金叉)
  ④ V20/V60 ≥ 1.2              放量
  ⑤ (加层,单独报)净利润同比 > 0 且较上一期加速

  去重:同一标的 60 个交易日内只取第一次触发。

**参数取值全部取自生益电子案例的邻近整数,一次性锁定,不做网格搜索。**
敏感性表若要做,另跑另报,并标注「那不是检验」(与 §72 同规矩)。

═══ 为什么这可能和以前测的不一样 ═══
  §55-61  RPS 池 / 口袋支点 / 基底形态  —— 都是**趋势中继**形态
  §57/62/63 多头排列                    —— **趋势已确立**
  **本节 ③ 要求 MA100 < MA300** —— 标的仍在长期下降通道里,
  这是**第一次**放量拉起,不是中继。**样本几乎不与前面重叠。**

═══ 事前判据(跑之前写死,不放宽) ═══
  ① **编码正确性锚点**:688183 在 2024-04-01~2024-06-30 内被检出。
     **检不出则本节编码错了,结论作废,不得事后调参数去凑。**
  ② **交易级**:事件后 250 日实收均值 > 同市值档随机(同日、同五分位)
  ③ **组合级**:每月从当月事件中随机 50 只等权,200 次自助,**p < 0.05**
  ④ **分段同向**:2013-2019 与 2020-2026 两段的组合超额**都为正**

  ②③④ 全过 → 「金叉前夜」是本研究第一个成立的**买点**形态
  ③ 不过   → 又一次「交易级成立、组合级不成立」(本研究已出现 10 次)

**事前预测(写下以便被证伪)**:我预计 **① 过、② 过、③ 不过**。
理由:§62「所有过滤器都在削右尾」,且「交易级 ≠ 组合级」在本研究出现过 10 次。
**若 ③ 通过,这是 75 节里第一个通过组合级检验的买点信号,用户的形态是对的,我错了。**

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
  688183 @ 2024-05-31:RPS50 99.7 / MA100 9.69 / MA300 10.74 / 量比 1.50
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
RPS_N, RPS_MIN = 50, 90.0
DEV_MIN, VOL_MIN, CONV_N = 1.20, 1.2, 20
COOLDOWN, H = 60, 250
NPICK, NBOOT, SEED, COST = 50, 200, 20260814, 0.003
NQ = 5
CASE, CASE_LO, CASE_HI = "688183", "2024-04-01", "2024-06-30"

t0 = time.time()
op, cl, vo, mv, ni = {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    cols = ["open", "close", "volume", "float_mv"]
    x = pd.read_parquet(f)
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    vo[k] = pd.to_numeric(x["volume"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    ni[k] = (pd.to_numeric(x["net_income"], errors="coerce")
             if "net_income" in x.columns else pd.Series(np.nan, index=x.index))
OP = pd.DataFrame(op).sort_index()
OP.index = OP.index.tz_localize(None)
CL = pd.DataFrame(cl).set_axis(OP.index)
VO = pd.DataFrame(vo).set_axis(OP.index)
MV = pd.DataFrame(mv).set_axis(OP.index)
NI = pd.DataFrame(ni).set_axis(OP.index)
OP, CL = OP.where(OP > 0), CL.where(CL > 0)
idx = OP.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"
fund_cov = float(NI.notna().to_numpy().mean())
print(f"基本面(net_income)覆盖 {fund_cov:.1%}"
      f"{'  —— 为 0 则第 ⑤ 层跳过' if fund_cov < 0.01 else ''}")

CLf = CL.ffill()
CLa, OPa = CLf.to_numpy(float), OP.ffill().to_numpy(float)
ALIVE = (CL.notna()).to_numpy()
MA100 = CLf.rolling(100, min_periods=100).mean().to_numpy(float)
MA300 = CLf.rolling(300, min_periods=300).mean().to_numpy(float)
V20 = VO.rolling(20, min_periods=10).mean().to_numpy(float)
V60 = VO.rolling(60, min_periods=30).mean().to_numpy(float)
MVa = MV.to_numpy(float)
print(f"均线/量能完成  ({time.time()-t0:.0f}s)")

# RPS50:逐日全市场百分位
RET = CLf / CLf.shift(RPS_N) - 1
RPS = RET.where(pd.DataFrame(ALIVE, index=idx, columns=CL.columns)).rank(
    axis=1, pct=True).to_numpy(float) * 100
print(f"RPS{RPS_N} 完成  ({time.time()-t0:.0f}s)")

ratio = MA100 / MA300
conv = np.full_like(ratio, False, dtype=bool)
conv[CONV_N:] = ratio[CONV_N:] > ratio[:-CONV_N]

TRIG = (ALIVE
        & (RPS >= RPS_MIN)
        & (CLa >= MA100 * DEV_MIN)
        & (MA100 < MA300)                      # ③ 仍是空头排列
        & conv                                 # ③ 差距在收敛
        & (V20 >= V60 * VOL_MIN))
TRIG = np.where(np.isfinite(MA300) & np.isfinite(V60), TRIG, False)

# 去重:同标的 COOLDOWN 内只取第一次
EV = np.zeros_like(TRIG)
lastfire = np.full(NS, -10**9)
for t in range(NT):
    hit = np.flatnonzero(TRIG[t] & (t - lastfire >= COOLDOWN))
    EV[t, hit] = True
    lastfire[hit] = t
n_ev = int(EV.sum())
print(f"事件 {n_ev} 笔(去重前 {int(TRIG.sum())})  ({time.time()-t0:.0f}s)")

# ── 判据 ①:案例锚点 ──
col = {c: i for i, c in enumerate(CL.columns)}
jc = col.get(CASE, -1)
win = (idx >= CASE_LO) & (idx <= CASE_HI)
case_hit = bool(EV[win, jc].any()) if jc >= 0 else False
case_days = [str(d.date()) for d in idx[win][EV[win, jc]]] if case_hit else []
print(f"\n判据① 案例锚点 {CASE} 在 {CASE_LO}~{CASE_HI} "
      f"{'✓ 检出 ' + ', '.join(case_days) if case_hit else '✗ 未检出'}")
if not case_hit and jc >= 0:
    tt = np.flatnonzero(win)
    print("  诊断(该窗口内逐条件通过情况,取最接近的一天):")
    for t in tt[::5]:
        print(f"    {idx[t].date()}  RPS {RPS[t,jc]:5.1f}  "
              f"偏离 {CLa[t,jc]/MA100[t,jc]-1:+6.1%}  "
              f"MA100<MA300 {MA100[t,jc]<MA300[t,jc]}  "
              f"收敛 {bool(conv[t,jc])}  量比 {V20[t,jc]/V60[t,jc]:.2f}")

# ── 前瞻:固定持有 250 日,实收与峰值 ──
FMAX = pd.DataFrame(CLa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]


def fwd(t, j):
    e = min(t + 1, NT - 1)
    ent = OPa[e, j]
    if not np.isfinite(ent) or ent <= 0:
        return np.nan, np.nan
    x = min(e + H, NT - 1)
    return CLa[x, j] / ent - 1, FMAX[e, j] / ent - 1


rng = np.random.default_rng(SEED)
ym = idx.to_period("M")
months = sorted(set(ym))
rows, cohorts = [], []
for p in months:
    tt = np.flatnonzero((ym == p))
    if not len(tt) or tt[-1] + H >= NT:
        continue
    ev = [(t, j) for t in tt for j in np.flatnonzero(EV[t])]
    if len(ev) < 10:
        continue
    t0m = tt[-1]
    base = ALIVE[t0m]
    m = np.where(base, MVa[t0m], np.nan)
    q = np.nanquantile(m[base], np.linspace(0, 1, NQ + 1)[1:-1])
    er, cr = [], []
    rng0 = np.random.default_rng(SEED + hash(str(p)) % 9999)
    for t, j in ev:
        r, pk = fwd(t, j)
        if not np.isfinite(r):
            continue
        er.append(r)
        mvj = MVa[t, j]
        i = int(np.searchsorted(q, mvj)) if np.isfinite(mvj) else 0
        lo = -np.inf if i == 0 else q[i - 1]
        hi = np.inf if i >= NQ - 1 else q[i]
        band = np.flatnonzero(base & (m > lo) & (m <= hi))
        if len(band):
            rc, _ = fwd(t, int(rng0.choice(band)))
            if np.isfinite(rc):
                cr.append(rc)
    if len(er) >= 10 and len(cr) >= 10:
        cohorts.append((str(p), np.array(er), np.array(cr)))
        rows.append({"月": str(p), "事件数": len(er), "事件均值": float(np.mean(er)),
                     "对照均值": float(np.mean(cr))})
print(f"有效月 {len(cohorts)} 个  ({time.time()-t0:.0f}s)")

allr = np.concatenate([c[1] for c in cohorts])
allc = np.concatenate([c[2] for c in cohorts])
print(f"\n{'='*96}\n交易级:事件 vs 同市值档随机(固定持有 {H} 日,实收)\n{'='*96}")
print(f"  事件 {len(allr):5d} 笔   均值 {np.mean(allr):+.2%}   中位 {np.median(allr):+.2%}"
      f"   ≥100% 占比 {np.mean(allr>=1.0):.1%}")
print(f"  对照 {len(allc):5d} 笔   均值 {np.mean(allc):+.2%}   中位 {np.median(allc):+.2%}"
      f"   ≥100% 占比 {np.mean(allc>=1.0):.1%}")


def boot(cs):
    ev = [np.mean(rng.choice(a, NPICK)) - COST for _, a, _ in cs]
    ct = [np.mean(rng.choice(b, NPICK)) - COST for _, _, b in cs]
    return float(np.mean(ev)), float(np.mean(ct))


bp = np.array([boot(cohorts) for _ in range(NBOOT)])
pe, pc = float(np.median(bp[:, 0])), float(np.median(bp[:, 1]))
pval = float((bp[:, 1] >= pe).mean())
print(f"\n{'='*96}\n组合级:每月随机 {NPICK} 只等权,{NBOOT} 次自助,单边成本 {COST:.1%}\n{'='*96}")
print(f"  事件组合 {pe:+.2%}   同市值档随机 {pc:+.2%}   超额 {pe-pc:+.2%}   p={pval:.4f}")

seg = {"13-19": [c for c in cohorts if c[0] < "2020-01"],
       "20-26": [c for c in cohorts if c[0] >= "2020-01"]}
segres = {}
for nm, cs in seg.items():
    if len(cs) < 6:
        segres[nm] = np.nan
        continue
    b = np.array([boot(cs) for _ in range(NBOOT)])
    segres[nm] = float(np.median(b[:, 0]) - np.median(b[:, 1]))
    print(f"  {nm}  {len(cs):3d} 个月  超额 {segres[nm]:+.2%}")

print(f"\n{'='*96}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*96}")
c1 = case_hit
c2 = np.mean(allr) > np.mean(allc)
c3 = pval < 0.05
c4 = all(np.isfinite(v) and v > 0 for v in segres.values())
print(f"  ① 编码锚点 {CASE} 被检出                 {'✓' if c1 else '✗'}")
print(f"  ② 交易级 事件 > 同市值档随机   "
      f"{np.mean(allr):+.2%} vs {np.mean(allc):+.2%}   {'✓' if c2 else '✗'}")
print(f"  ③ 组合级 p<0.05                          p={pval:.4f}   {'✓' if c3 else '✗'}")
print(f"  ④ 两段超额都为正              "
      + " / ".join(f"{nm} {v:+.2%}" for nm, v in segres.items())
      + f"   {'✓' if c4 else '✗'}")
if not c1:
    print("\n  **① 不过:编码没抓住用户描述的形态,本节结论作废。**")
    print("  **不得事后调参数去凑 —— 那是曲线拟合,不是检验。**")
elif c2 and c3 and c4:
    print("\n  **结论:「金叉前夜」是本研究第一个通过组合级检验的买点形态。**")
else:
    print(f"\n  **结论:不成立"
          f"{'(又一次交易级成立、组合级不成立)' if c2 and not c3 else ''}**")

pd.DataFrame(rows).to_csv(f"{SP}/pre_golden_cross.csv", index=False)
print(f"\n→ {SP}/pre_golden_cross.csv   ({time.time()-t0:.0f}s)")
