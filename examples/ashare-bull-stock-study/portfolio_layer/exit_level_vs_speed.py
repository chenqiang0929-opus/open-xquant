"""离场规则的 2×3 分解:看个股还是看市场,看得快还是看得慢(第七十一节)

═══ 起因:§70 把两个轴混在一起了 ═══
§70 的六条规则里,R1-R4 全是「**个股 + 快**」,R5 是「**市场 + 慢**」。
四条主动规则全灭、R5 显著 —— 但**无法区分**到底是
「看个股不行」还是「看得太快不行」。**本节把这两个轴拆开。**

用户提出的两个方向正好卡在这里:
  ①「5浪理论 / 一波一基地」 → 个股 + 中速
  ②「月线指标」            → 慢速(§70 R5 已属此类)
若决定因素是**速度**,①有希望;若是**层级**,①注定没戏。

**关于 5 浪:浪型事后可数、事前不可数**(同一段行情不同人数出不同的浪),
无法机械化因而无法回测,不构成可复制的菜谱。
**但「基地计数」可以机械化** —— 本节测的是后者,不是浪。

═══ 必须先说的一件事:§70 R5 有前视偏差 ═══
§70 的 R5 写的是 `green_start[p] = first_td[第一个绿轴月]` ——
**在绿轴月的首个交易日卖出**,而月线 MACD 柱是否转负
要到**该月收盘**才知道。等于提前一个月知道结果。

本节把 B3 实现为**正确口径**(绿轴月确认后、**次月首日**卖),
并同时跑一条 B3′(§70 原口径)**专门用来量这个偏差有多大**。
§70 的 R5 结论在本节结果出来之前不能采信。

═══ 设计:同一条规则形式,分别作用于个股与指数 ═══
                     个股(A)              市场 510300(B)
  快  破 MA100        A1(=§70 R1)         B1
  中  破 MA200        A2(=§70 R2)         B2
  慢  月线 MACD 转负   A3                  B3(§70 R5 的修正版)

**同形式、只换作用对象** —— 这是层级轴唯一干净的比法。
A1/A2 与 §70 的 R1/R2 应当逐位复现,**作为本节的正确性锚点**。

外加用户的想法本身:
  C   基地计数:第 3 个基地完成后,回撤 15% 才卖(个股 + 中速)

═══ 基地计数的机械定义(事前写死,不搜索) ═══
  入场后跟踪持仓期最高价 H。
  价格低于 H 即进入「筑基」;若在 H 之下连续 ≥ BASE_MIN=25 个交易日
  之后**再创新高**,记为完成一个基地(+1)。
  筑基期间自 H 回撤超过 BASE_MAXDD=35% → 结构破坏,直接卖出。
  完成 N_BASE=3 个基地之后,自 H 回撤 FAIL_DD=15% 即卖出;
  **不满 3 个基地之前,不因回撤卖出**(只受 35% 结构破坏约束)。
  四个参数全部事前锁定,**不做网格搜索**。

═══ 其余设定与 §70 完全一致(不重调) ═══
  池子 红轴月末 listed_days ∈ [365,1095) 且在市;入场 次月首日开盘
  持仓 随机 50 只等权、200 次自助;成本 单边 0.3%;最长 750 日
  对照 同市值五分位内随机同样多只(市值中性化)

═══ 事前判据(跑之前写死,不放宽) ═══
  ① **层级轴**:三对(A1/B1、A2/B2、A3/B3)中,市场版超额 ≥ 个股版超额
     的对数 **≥ 2/3**  → 决定因素是「看谁」
  ② **速度轴**:慢档(A3,B3)平均超额 > 快档(A1,B1)平均超额
     → 决定因素是「多快」
  ③ **基地计数**:C 的超额 **p < 0.05** → 用户的「一波一基地」在实收口径下成立
  ④ **复现锚点**:A1 实收/笔 与 §70 R1(+2.4%)差 < 0.5pp;
     A2 与 §70 R2(+5.2%)差 < 0.5pp。对不上则本节实现有错,结论作废。

  ①②同时成立 → 两个轴都有效,慢的市场级规则是唯一方向
  ①成立②不成立 → 只是层级问题,个股价格无论快慢都不能用
  ③成立 → 基地计数是例外,值得单独展开

**事前预测(写下以便被证伪)**:① 通过、② 通过、③ **不通过**。
理由:§62「所有过滤器都在削右尾」、§70 四条主动规则全灭。
**若 ③ 通过,说明个股结构性信息确实存在,我错了。**
另外预测:**B3 修正后的超额会明显小于 §70 R5 的 +4.6pp**,
因为那里面含一个月的前视。

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
  §70 复现:R1 +2.4%/笔、R2 +5.2%/笔
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
BASE_MIN, BASE_MAXDD, N_BASE, FAIL_DD = 25, 0.35, 3, 0.15

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
MA100 = pd.DataFrame(CLa).rolling(100, min_periods=100).mean().to_numpy(float)
MA200 = pd.DataFrame(CLa).rolling(200, min_periods=200).mean().to_numpy(float)
print(f"个股均线完成  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
first_td = {p: int(np.flatnonzero(ym == p)[0]) for p in ym.unique()}
allm = sorted(last_td)


def monthly_hist(df):
    """月线 MACD(12,26,9) 柱,列为标的。返回 (月 Period 索引, 柱 DataFrame)。"""
    m = df.resample("ME").last()
    d = m.ewm(span=12, adjust=False).mean() - m.ewm(span=26, adjust=False).mean()
    h = d - d.ewm(span=9, adjust=False).mean()
    h.index = h.index.to_period("M")
    return h


# ── 个股月线 MACD:第 m 月末确认转负 → 第 m+1 月首日卖(无前视) ──
HS = monthly_hist(CL)
SELL_A3 = np.zeros((NT, NS), dtype=bool)
hs_months = list(HS.index)
pos_m = {p: i for i, p in enumerate(hs_months)}
HSa = HS.to_numpy(float)
for i, p in enumerate(hs_months[:-1]):
    nxt = hs_months[i + 1]
    if nxt in first_td:
        SELL_A3[first_td[nxt]] = np.isfinite(HSa[i]) & (HSa[i] < 0)
print(f"个股月线 MACD 完成  ({time.time()-t0:.0f}s)")

# ── 指数:日线 MA100 / MA200 / 月线 MACD ──
mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
mkc = mk["close"].reindex(idx).ffill()
IDX = mkc.to_numpy(float)
IMA100 = mkc.rolling(100, min_periods=100).mean().to_numpy(float)
IMA200 = mkc.rolling(200, min_periods=200).mean().to_numpy(float)
SELL_B1 = np.isfinite(IMA100) & (IDX < IMA100)
SELL_B2 = np.isfinite(IMA200) & (IDX < IMA200)

ih = monthly_hist(mkc.to_frame("i"))["i"]
reg = {p: int(v > 0) for p, v in ih.items()}
im = list(ih.index)
SELL_B3 = np.zeros(NT, dtype=bool)          # 正确口径:确认后次月首日
for i, p in enumerate(im[:-1]):
    nxt = im[i + 1]
    if nxt in first_td and ih.iloc[i] < 0:
        SELL_B3[first_td[nxt]] = True
SELL_B3P = np.zeros(NT, dtype=bool)         # §70 原口径:绿轴月首日(含前视)
for p in im:
    if p in first_td and reg.get(p, 1) == 0:
        SELL_B3P[first_td[p]] = True
red = [p for p in allm if reg.get(p, 0) == 1 and last_td[p] + 250 < NT]
print(f"指数信号完成,红轴月 {len(red)} 个  ({time.time()-t0:.0f}s)")

RULES = ["A1 个股破20周线", "A2 个股破10月线", "A3 个股月线MACD",
         "B1 指数破20周线", "B2 指数破10月线", "B3 指数月线MACD",
         "B3′ §70原口径(含前视)", "C  基地计数(第3基)"]
NR = len(RULES)


def run_trade(j, e, rule):
    """从 e(买入日)起按 rule 离场,返回 (实收, 期间峰值涨幅)。"""
    entry = OPf[e, j]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, np.nan
    end = min(e + MAXHOLD, NT - 1)
    peak = entry
    nbase, below = 0, 0
    for t in range(e, end + 1):
        c = CLf[t, j]
        if not np.isfinite(c):
            continue
        if rule == 7:                                   # 基地计数
            if c > peak:
                if below >= BASE_MIN:
                    nbase += 1
                below = 0
            else:
                below += 1
            dd = c / peak - 1
            hit = dd <= -BASE_MAXDD or (nbase >= N_BASE and dd <= -FAIL_DD)
        elif rule == 0:
            hit = np.isfinite(MA100[t, j]) and c < MA100[t, j]
        elif rule == 1:
            hit = np.isfinite(MA200[t, j]) and c < MA200[t, j]
        elif rule == 2:
            hit = bool(SELL_A3[t, j])
        elif rule == 3:
            hit = bool(SELL_B1[t])
        elif rule == 4:
            hit = bool(SELL_B2[t])
        elif rule == 5:
            hit = bool(SELL_B3[t])
        else:
            hit = bool(SELL_B3P[t])
        peak = max(peak, c)
        if hit and t > e:
            t1 = min(t + 1, NT - 1)
            px = OPf[t1, j] if np.isfinite(OPf[t1, j]) else c
            return px / entry - 1, peak / entry - 1
    return CLf[end, j] / entry - 1, peak / entry - 1


mon = []
for p in red:
    i = allm.index(p)
    e = first_td[allm[i + 1]] if i + 1 < len(allm) else None
    if e is None or e >= NT - 5:
        continue
    base = ALIVE[last_td[p]] & np.isfinite(OPa[e]) & (OPa[e] > 0)
    t = last_td[p]
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
    res = {r: (np.array([run_trade(j, e, r) for j in pool]),
               np.array([run_trade(j, e, r) for j in ctrl])) for r in range(NR)}
    mon.append((p, res))
    if len(mon) % 15 == 0:
        print(f"  {p}  池 {len(pool)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"完成 {len(mon)} 个红轴月  ({time.time()-t0:.0f}s)")

rng = np.random.default_rng(SEED)
print(f"\n{'='*104}\n离场规则 2×3 分解 + 基地计数(红轴入场,50 只等权,{NBOOT} 次自助)\n{'='*104}")
print(f"{'规则':<24}{'实收/笔':>10}{'峰值/笔':>10}{'兑现率':>8}"
      f"{'组合':>9}{'对照':>9}{'超额':>9}{'p':>8}")
S = {}
for r in range(NR):
    pr = [res[r][0][:, 0][np.isfinite(res[r][0][:, 0])] for _, res in mon]
    pk = [res[r][0][:, 1][np.isfinite(res[r][0][:, 1])] for _, res in mon]
    cr = [res[r][1][:, 0][np.isfinite(res[r][1][:, 0])] for _, res in mon]
    bp = [np.mean([np.mean(rng.choice(a, NPICK)) - COST for a in pr if len(a) >= 10])
          for _ in range(NBOOT)]
    bc = [np.mean([np.mean(rng.choice(b, NPICK)) - COST for b in cr if len(b) >= 10])
          for _ in range(NBOOT)]
    ap, ac = float(np.median(bp)), float(np.median(bc))
    pv = float((np.array(bc) >= ap).mean())
    real, peak = float(np.mean(np.concatenate(pr))), float(np.mean(np.concatenate(pk)))
    S[r] = (real, peak, real / peak if peak > 0 else np.nan, ap, ac, ap - ac, pv)
    print(f"{RULES[r]:<24}{real:>+10.1%}{peak:>+10.1%}{S[r][2]:>8.0%}"
          f"{ap:>+9.1%}{ac:>+9.1%}{ap-ac:>+9.1%}{pv:>8.3f}")

print(f"\n{'='*104}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*104}")
pairs = [(0, 3, "破20周线"), (1, 4, "破10月线"), (2, 5, "月线MACD")]
win = sum(S[b][5] >= S[a][5] for a, b, _ in pairs)
c1 = win >= 2
slow, fast = (S[2][5] + S[5][5]) / 2, (S[0][5] + S[3][5]) / 2
c2 = slow > fast
c3 = S[7][6] < 0.05
c4 = abs(S[0][0] - 0.024) < 0.005 and abs(S[1][0] - 0.052) < 0.005
print(f"  ① 层级轴:市场版超额 ≥ 个股版 的对数 ≥ 2/3     {win}/3   {'✓' if c1 else '✗'}")
for a, b, nm in pairs:
    print(f"       {nm:<10} 个股 {S[a][5]:+.1%}  vs  市场 {S[b][5]:+.1%}"
          f"   {'市场胜' if S[b][5] >= S[a][5] else '个股胜'}")
print(f"  ② 速度轴:慢档平均超额 > 快档                {slow:+.2%} vs {fast:+.2%}"
      f"   {'✓' if c2 else '✗'}")
print(f"  ③ 基地计数超额 p < 0.05                      p={S[7][6]:.3f}"
      f"   {'✓' if c3 else '✗'}")
print(f"  ④ 复现 §70 锚点(R1 +2.4% / R2 +5.2%)        "
      f"A1 {S[0][0]:+.1%} / A2 {S[1][0]:+.1%}   {'✓' if c4 else '✗'}")
print(f"\n  **§70 R5 前视偏差的量级:** 修正 B3 超额 {S[5][5]:+.2%}(p={S[5][6]:.3f})"
      f"  vs  原口径 B3′ {S[6][5]:+.2%}(p={S[6][6]:.3f})"
      f"  → 前视贡献 {S[6][5]-S[5][5]:+.2%}")
if not c4:
    print("\n  **④ 不过:本节实现与 §70 对不上,以上结论全部作废。**")
else:
    print(f"\n  **结论:{'层级与速度两个轴都成立' if (c1 and c2) else ('只有层级轴成立' if c1 else ('只有速度轴成立' if c2 else '两个轴都不成立'))}"
          f";基地计数{'成立' if c3 else '不成立'}**")

pd.DataFrame([{"规则": RULES[r], "实收每笔": S[r][0], "峰值每笔": S[r][1],
               "兑现率": S[r][2], "组合": S[r][3], "对照": S[r][4],
               "超额": S[r][5], "p": S[r][6]} for r in range(NR)]).to_csv(
    f"{SP}/exit_level_vs_speed.csv", index=False)
print(f"\n→ {SP}/exit_level_vs_speed.csv   ({time.time()-t0:.0f}s)")
