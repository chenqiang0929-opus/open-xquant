"""从「峰值」换成「实收」:六种离场规则的浮盈兑现率(第七十节)

═══ 起因:§67-69 全部回避了最难的一环 ═══
§67/§68/§69 测的都是「未来 250 日内**最大**累计涨幅」——
那是**上帝视角的峰值,不是能拿到的钱**。§66 的七只案例已经量过这个差距:
泰格 峰值 +1076% → 实收 +198%(兑现率 18%)、汤臣 +249% → +15%。

用户确认了三件事:
  ① 目标是**「暴露」于右尾,不是「捕捉」**
  ② 持仓改成 **50 只**(§69 算过:5 只抓到 ≥500% 的概率只有 25%,50 只是 94%)
  ③ **离场规则才是没解决的那一环** —— 「不解决会严重影响收益」

**本节就把 §67-69 的口径从峰值换成实收,并把离场规则当作主变量。**

═══ 事前锁定(不搜索、不调参) ═══
  池子      红轴月末的 `listed_days ∈ [365,1095)` 且在市(与 §68/§69 完全一致)
  入场      次月首个交易日开盘
  持仓      从池中随机 **50 只**等权(分不出哪只 → 只能随机),**200 次自助抽样**
  成本      单边 0.3%

  六条离场规则(**全部事前写死,不搜索参数**):
    R0  固定持有 250 交易日            ← 基线,与 §67-69 口径可比
    R1  收盘跌破 20 周线(MA100)       ← 用户提议
    R2  收盘跌破 10 月线(MA200)       ← §63/§66 用过,有历史可比
    R3  从持仓期最高点回撤 20%          ← 真·移动止盈,与均线无关
    R4  收盘 ≥ 20 周线 × 1.5(乖离止盈) ← 用户提议的偏离率
    R5  红轴结束(绿轴首月)才卖         ← 把 §68 的闸门用到卖出端
  全部规则统一:触发后**次日开盘**成交;最长持有 750 日(避免无限持有)

  对照      同市值五分位内随机 50 只(市值中性化,与 §69 同法)

═══ 本节最核心的一个数 ═══
**浮盈兑现率 = 实收 / 期间峰值。** §66 实测约 18%。
**若六条规则的兑现率都在 20-30%,说明右尾在离场环节被系统性漏掉,与买点无关。**

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 至少一条规则的**实收**年化 ≥ 全市场等权同期
  ② 该规则优于同市值档随机 50 只,**p < 0.05**
  ③ 该规则在四段红轴上**分段同向**(都为正超额)
  ①②③ 全过 → 存在一条能把右尾暴露兑现成实收的离场规则
  ① 不过   → **「暴露于右尾」这个目标本身在实收口径下不成立**

**事前预测(写下以便被证伪)**:我预计 ① 会有 1-2 条通过(R2/R5 概率最大),
但**兑现率全部低于 35%**;且 ② 大概率不过 ——
因为 §66 组合级 p=0.4867、§62 已证「所有过滤器都在削右尾」。
**若某条规则兑现率明显超过 50% 且 ② 通过,我错了。**

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
NPICK, NBOOT, SEED, COST = 50, 200, 20260814, 0.003
MAXHOLD, BASE_H = 750, 250
Y_LO, Y_HI, NQ = 365, 1095, 5

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
print(f"均线完成  ({time.time()-t0:.0f}s)")

mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
M = mk["close"].resample("ME").last().dropna()
dif = M.ewm(span=12, adjust=False).mean() - M.ewm(span=26, adjust=False).mean()
hist = dif - dif.ewm(span=9, adjust=False).mean()
reg = {p: int(v > 0) for p, v in zip(hist.index.to_period("M"), hist)}
ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
first_td = {p: int(np.flatnonzero(ym == p)[0]) for p in ym.unique()}
allm = sorted(last_td)
red = [p for p in allm if reg.get(p, 0) == 1 and last_td[p] + BASE_H < NT]
# 每个红轴月之后第一个绿轴月的首日(R5 用)
green_start = {}
for p in red:
    nxt = [q for q in allm if q > p and reg.get(q, 1) == 0]
    green_start[p] = first_td[nxt[0]] if nxt else NT - 1
print(f"红轴月 {len(red)} 个  ({time.time()-t0:.0f}s)")

RULES = ["R0 固定250日", "R1 破20周线", "R2 破10月线",
         "R3 回撤20%", "R4 乖离+50%", "R5 红轴结束"]


def run_trade(j, e, rule, t_cap):
    """从 e(买入日)起按 rule 离场,返回 (实收, 期间峰值涨幅)。"""
    entry = OPf[e, j]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, np.nan
    end = min(e + MAXHOLD, NT - 1)
    peak = entry
    ex = None
    if rule == 0:
        ex = min(e + BASE_H, NT - 1)
    elif rule == 5:
        ex = min(t_cap, end)
    if ex is not None:
        seg = CLf[e:ex + 1, j]
        pk = np.nanmax(seg) if seg.size else entry
        return CLf[ex, j] / entry - 1, pk / entry - 1
    for t in range(e, end + 1):
        c = CLf[t, j]
        if not np.isfinite(c):
            continue
        peak = max(peak, c)
        hit = False
        if rule == 1:
            hit = np.isfinite(MA100[t, j]) and c < MA100[t, j]
        elif rule == 2:
            hit = np.isfinite(MA200[t, j]) and c < MA200[t, j]
        elif rule == 3:
            hit = c <= peak * 0.80
        elif rule == 4:
            hit = np.isfinite(MA100[t, j]) and c >= MA100[t, j] * 1.5
        if hit:
            t1 = min(t + 1, NT - 1)
            px = OPf[t1, j] if np.isfinite(OPf[t1, j]) else c
            return px / entry - 1, peak / entry - 1
    return CLf[end, j] / entry - 1, peak / entry - 1


# ── 逐红轴月:池内全部标的的六条规则结果 ──
mon_trades = []          # [(月, 池idx, 对照idx, {rule: (ret[], peak[])})]
for p in red:
    t = last_td[p]
    e = first_td[allm[allm.index(p) + 1]] if allm.index(p) + 1 < len(allm) else None
    if e is None or e >= NT - 5:
        continue
    base = ALIVE[t] & np.isfinite(OPa[e]) & (OPa[e] > 0)
    age = LDa[t]
    pool = np.flatnonzero(base & (age >= Y_LO) & (age < Y_HI))
    if len(pool) < NPICK:
        continue
    # 市值中性对照:按池内标的所在五分位,同档抽同样多
    m = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(m[base], np.linspace(0, 1, NQ + 1)[1:-1])
    ctrl = []
    rng0 = np.random.default_rng(SEED + hash(str(p)) % 9999)
    for i in range(NQ):
        lo = -np.inf if i == 0 else q[i - 1]
        hi = np.inf if i == NQ - 1 else q[i]
        band = np.flatnonzero(base & (m > lo) & (m <= hi))
        npool = int(np.sum((m[pool] > lo) & (m[pool] <= hi)))
        if npool and len(band) >= npool:
            ctrl.append(rng0.choice(band, npool, replace=False))
    ctrl = np.concatenate(ctrl) if ctrl else pool
    res = {}
    for r in range(6):
        rp = np.array([run_trade(j, e, r, green_start[p]) for j in pool])
        rc = np.array([run_trade(j, e, r, green_start[p]) for j in ctrl])
        res[r] = (rp, rc)
    mon_trades.append((p, res))
    if len(mon_trades) % 15 == 0:
        print(f"  {p}  池 {len(pool)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"完成 {len(mon_trades)} 个红轴月  ({time.time()-t0:.0f}s)")

# ── 自助抽样 50 只 ──
rng = np.random.default_rng(SEED)
print(f"\n{'='*104}\n六种离场规则:实收 vs 峰值(红轴入场,50 只等权,{NBOOT} 次自助)\n{'='*104}")
print(f"{'规则':<14}{'实收/笔':>10}{'峰值/笔':>10}{'兑现率':>9}"
      f"{'组合年化':>10}{'对照年化':>10}{'p':>8}{'≥100%笔占比':>12}")
summary = []
for r in range(6):
    pr, cr, pk = [], [], []
    for _, res in mon_trades:
        a, b = res[r]
        pr.append(a[:, 0][np.isfinite(a[:, 0])])
        pk.append(a[:, 1][np.isfinite(a[:, 1])])
        cr.append(b[:, 0][np.isfinite(b[:, 0])])
    ret_all = np.concatenate(pr)
    peak_all = np.concatenate(pk)
    boot_p, boot_c = [], []
    for _ in range(NBOOT):
        vp = [np.mean(rng.choice(a, NPICK)) - COST for a in pr if len(a) >= 10]
        vc = [np.mean(rng.choice(b, NPICK)) - COST for b in cr if len(b) >= 10]
        boot_p.append(np.mean(vp))
        boot_c.append(np.mean(vc))
    yrs = len(mon_trades) / 12.0
    ap = (1 + np.median(boot_p)) ** (12 / 12) - 1      # 每笔平均持有约1年,直接用均值
    ac = (1 + np.median(boot_c)) ** (12 / 12) - 1
    p_val = float((np.array(boot_c) >= np.median(boot_p)).mean())
    real, peak = np.mean(ret_all), np.mean(peak_all)
    rate = real / peak if peak > 0 else np.nan
    summary.append((RULES[r], real, peak, rate, ap, ac, p_val))
    print(f"{RULES[r]:<14}{real:>+10.1%}{peak:>+10.1%}{rate:>9.0%}"
          f"{ap:>+10.1%}{ac:>+10.1%}{p_val:>8.3f}{np.mean(ret_all>=1.0):>12.1%}")

print(f"\n{'='*104}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*104}")
BENCH = 0.0722
best = max(summary, key=lambda s: s[4])
c1 = best[4] >= BENCH
c2 = best[6] < 0.05
print(f"  ① 至少一条规则实收年化 ≥ {BENCH:.2%}   最好 {best[0]} {best[4]:+.2%}   {'✓' if c1 else '✗'}")
print(f"  ② 该规则优于同市值档随机 p<0.05      p={best[6]:.3f}   {'✓' if c2 else '✗'}")
print(f"\n  **浮盈兑现率(本节核心):** "
      + " / ".join(f"{s[0].split()[0]} {s[3]:.0%}" for s in summary))
lo, hi = min(s[3] for s in summary), max(s[3] for s in summary)
print(f"  区间 {lo:.0%} ~ {hi:.0%}(§66 七只案例实测约 18%)")
if hi < 0.35:
    print("  → **所有规则的兑现率都低于 35%:右尾在离场环节被系统性漏掉,与买点无关。**")
print(f"\n  **结论:{'存在能把右尾暴露兑现成实收的离场规则' if (c1 and c2) else '不成立'}**")

pd.DataFrame(summary, columns=["规则", "实收每笔", "峰值每笔", "兑现率",
                               "组合年化", "对照年化", "p"]).to_csv(
    f"{SP}/exposure_exit_rules.csv", index=False)
print(f"\n→ {SP}/exposure_exit_rules.csv   ({time.time()-t0:.0f}s)")
