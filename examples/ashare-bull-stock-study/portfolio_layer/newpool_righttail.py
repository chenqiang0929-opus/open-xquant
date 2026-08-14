"""地基检验:次新池的「右尾密度」是不是真的更高(第六十七节)

═══ 为什么先测这一条 ═══
用户提出转向:不再问「信号能不能打败市场」,改为按五模块框架
(核心逻辑/信号/标的池/组合构建/风险管理)搭一套可执行系统。
四个模块已给出,**核心逻辑空缺**。

建议的核心逻辑分三层,而**第 ① 层是整套系统的地基**:

  ① 供给层:中国仍在产业升级,每隔几年有一个行业从 0 到 1,
     这些公司几乎都在**上市后 2-5 年内**兑现
     → **次新池的十倍股密度应高于全市场**
  ② 不可知层:无法事前分辨是哪几家(右尾特征 lift 1.05/0.95/0.95、ML AUC 0.57)
  ③ 验证层:市场先于财报确认,表现为周线趋势;
     **趋势只作「已确认」的验证信号,不作选股因子**(后者是负 alpha,三次独立确认)

**如果 ① 不成立,后面的信号/组合/风控全部没有意义。**
而这一条恰好有互相矛盾的现有证据:

  我们 §66      次新池交易级 +3.28pp(全市场同日随机 +1.26% → 次新同日随机 +4.54%)
  但            组合级 p=0.4867 不显著
  DeepSeek      次新效应 13-19 年 83.9% → 20-25 年 48.5%,**掉 35pp**

**若 DeepSeek 对,则「结构性供给、不衰减」不成立,核心逻辑要重写。**
这一条 30 分钟能测完,搭完整系统要几天 —— 先测便宜的那个。

═══ 事前锁定(不搜索、不调参) ═══
  池子      `listed_days ∈ [365, 1825]`(上市 1~5 年)**滚动窗口**
            → 用户原方案「2022-08 之后」是固定日期,池子会老化,
              且只有 4 年、几乎无样本外;滚动窗口覆盖 2013-2026 且可分段
            → 上限 5 年(非 §66 的 3 年):用户案例里宁德/迈瑞/华虹
              的启动都在上市后 3-5 年。**只测这一个窗口。**
  右尾密度  自某月末起,未来 H 个交易日内**最大累计涨幅** ≥ G 的比例
            H ∈ {250, 500, 750}   G ∈ {100%, 200%, 500%}
            **九格全报,不挑**
  对照      同月全市场(在市且可交易)的同一指标
  分段      2013-2019 / 2020-2026(直接检验 DeepSeek 说的衰减)
  成本      本检验只算涨幅密度,不涉及交易,无成本假设

═══ 必须堵住的陷阱 ═══
**检验自身不能有幸存者偏差。** 退市股按最后有效价前向填充参与,
**绝不剔除** —— 否则这个检验就变成了它要批判的那个东西。
脚本会报出「期末已无有效价」的样本占比。

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 全区间 次新池翻倍率 / 全市场翻倍率 ≥ **1.3**
  ② 两段同向:2013-2019 与 2020-2026 **都 ≥ 1.0**
  ③ 后段不显著衰减:后段比值 ≥ 前段比值 × **0.7**
  ④ 三个门槛(100/200/500%)方向一致:都 ≥ 1.0
  四条全过 → 核心逻辑第 ① 层成立,可以往下搭系统
  ②或③ 不过 → DeepSeek 是对的,「结构性不衰减」被证伪,核心逻辑要重写

**事前预测(写下以便被证伪)**:① 大概率过;**③ 大概率不过** ——
因 DeepSeek 的分段证据,以及「日期决定一切」在本 session 已出现五次。
若果真如此,正确的核心逻辑不是「次新池永远好」,
而是「次新池在 IPO 稀缺 / 产业周期上行时好」——**条件性**而非结构性。

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

AGE_LO, AGE_HI = 365, 1825          # 上市 1~5 年(自然日),事前锁定
HORIZONS = (250, 500, 750)
GAINS = (1.0, 2.0, 5.0)
SPLIT = "2020-01-01"

t0 = time.time()
cl, ld = {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "listed_days"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
LD = pd.DataFrame(ld).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

CLa = CL.to_numpy(float)
LDa = LD.to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
# 前向填充:承接停牌与退市(退市按最后有效价,与全研究引擎一致,**不剔除**)
CLf = pd.DataFrame(CLa).ffill().to_numpy(float)
IS_NEW = (LDa >= AGE_LO) & (LDa <= AGE_HI) & ALIVE

# 未来 H 日内的最大价(用 ffill 后的序列,反向滚动 max)
FMAX = {}
for H in HORIZONS:
    rev = CLf[::-1]
    m = pd.DataFrame(rev).rolling(H, min_periods=1).max().to_numpy(float)[::-1]
    FMAX[H] = m
print(f"前瞻最大价矩阵完成  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = sorted(last_td)

rows = []
n_dead = n_tot = 0
for p in months:
    t = last_td[p]
    if t + min(HORIZONS) >= NT:
        continue
    base = ALIVE[t] & np.isfinite(CLa[t]) & (CLa[t] > 0)
    newp = base & IS_NEW[t]
    if newp.sum() < 20:
        continue
    n_tot += int(base.sum())
    n_dead += int((base & ~np.isfinite(CLa[min(t + 250, NT - 1)])).sum())
    rec = {"月": str(p), "全市场数": int(base.sum()), "次新数": int(newp.sum())}
    for H in HORIZONS:
        e = min(t + H, NT - 1)
        ratio = np.where(base, FMAX[H][e] / CLa[t] - 1, np.nan)
        for G in GAINS:
            hit = np.isfinite(ratio) & (ratio >= G)
            rec[f"全_H{H}_G{int(G*100)}"] = float(np.mean(hit[base]))
            rec[f"新_H{H}_G{int(G*100)}"] = float(np.mean(hit[newp]))
    rows.append(rec)

R = pd.DataFrame(rows)
print(f"逐月样本 {len(R)} 个月  ({time.time()-t0:.0f}s)")
print(f"次新池平均规模 {R['次新数'].mean():.0f} 只 / 全市场 {R['全市场数'].mean():.0f} 只 "
      f"({R['次新数'].mean()/R['全市场数'].mean():.1%})")
print(f"退市/停牌自检:250 日后已无有效价的样本占 {n_dead/max(n_tot,1):.2%}"
      f"(这些**未被剔除**,按最后有效价参与)")

sp = str(pd.Period(SPLIT, "M"))
seg = {"全区间": R, "13-19": R[R["月"] < sp], "20-26": R[R["月"] >= sp]}

print(f"\n{'='*104}\n右尾密度:次新池 vs 全市场(九格全报)\n{'='*104}")
print(f"{'窗口':<8}{'门槛':<8}{'区间':<10}{'全市场':>10}{'次新池':>10}{'比值':>9}{'月数':>7}")
tab = {}
for H in HORIZONS:
    for G in GAINS:
        for nm, d in seg.items():
            if len(d) < 12:
                continue
            a = d[f"全_H{H}_G{int(G*100)}"].mean()
            b = d[f"新_H{H}_G{int(G*100)}"].mean()
            r = b / a if a > 0 else np.nan
            tab[(H, G, nm)] = r
            print(f"{H:<8}{'≥'+str(int(G*100))+'%':<8}{nm:<10}{a:>10.2%}{b:>10.2%}"
                  f"{r:>9.2f}{len(d):>7}")
        print()

print(f"{'='*104}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*104}")
base_key = (250, 1.0)          # 主口径:250 日内翻倍
c1 = tab.get((*base_key, "全区间"), np.nan) >= 1.3
r1, r2 = tab.get((*base_key, "13-19"), np.nan), tab.get((*base_key, "20-26"), np.nan)
c2 = bool(np.isfinite(r1) and np.isfinite(r2) and r1 >= 1.0 and r2 >= 1.0)
c3 = bool(np.isfinite(r1) and np.isfinite(r2) and r2 >= r1 * 0.7)
c4 = all(tab.get((250, g, "全区间"), 0) >= 1.0 for g in GAINS)
print(f"  ① 全区间比值 ≥ 1.3        {tab.get((*base_key,'全区间'), float('nan')):.2f}"
      f"          {'✓' if c1 else '✗'}")
print(f"  ② 两段都 ≥ 1.0            {r1:.2f} / {r2:.2f}      {'✓' if c2 else '✗'}")
print(f"  ③ 后段 ≥ 前段×0.7         {r2:.2f} vs {r1*0.7:.2f}   {'✓' if c3 else '✗'}")
print(f"  ④ 三门槛方向一致          "
      + " / ".join(f"{tab.get((250,g,'全区间'), float('nan')):.2f}" for g in GAINS)
      + f"   {'✓' if c4 else '✗'}")
ok = c1 and c2 and c3 and c4
print(f"\n  **结论:{'地基成立,可以往下搭系统' if ok else '地基不成立'}**")
if not ok and np.isfinite(r1) and np.isfinite(r2) and r2 < r1 * 0.7:
    print("  → ②/③ 不过 = DeepSeek 是对的:次新效应在衰减。")
    print("    正确的核心逻辑不是「次新池永远好」,而是「次新池在特定条件下好」——")
    print("    **条件性**而非结构性,整套系统的地基要重写。")

print(f"\n{'='*104}\n案例回归:用户点名的十倍股,启动月是否落在池内\n{'='*104}")
CASES = [("300750", "宁德时代", "2019-06"), ("300760", "迈瑞医疗", "2019-06"),
         ("688347", "华虹公司", "2025-06"), ("688313", "仕佳光子", "2024-10"),
         ("688498", "源杰科技", "2024-10"), ("301377", "鼎泰高科", "2024-10")]
col_of = {c: i for i, c in enumerate(CL.columns)}
for code, nm, mth in CASES:
    if code not in col_of:
        print(f"  {code} {nm:<8} 不在面板")
        continue
    j = col_of[code]
    p = pd.Period(mth, "M")
    if p not in last_td:
        print(f"  {code} {nm:<8} {mth} 超出面板")
        continue
    t = last_td[p]
    v = LDa[t, j]
    inp = bool(IS_NEW[t, j])
    print(f"  {code} {nm:<8} {mth}  listed_days={v if np.isfinite(v) else '—':>6}"
          f"  {'✅ 在池内' if inp else '❌ 不在池内'}")

R.to_csv(f"{SP}/newpool_righttail.csv", index=False)
print(f"\n→ {SP}/newpool_righttail.csv   ({time.time()-t0:.0f}s)")
