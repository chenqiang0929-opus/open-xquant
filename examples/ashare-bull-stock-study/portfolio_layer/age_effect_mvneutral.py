"""市值中性化后的年龄效应还剩多少(第六十九节)

═══ 起因:§68 留下的唯一悬案 ═══
§68 按事前判据「算发现」(红轴期间 1-3年/>10年 = 1.40 ≥ 1.3,四段牛市全过),
**但通过后自查加严,市值中性化把它打到 1.21**,低于事前门槛。

§68 明确写了:「剩下的 1.21 是不是真的年龄效应,本节没有定论 ——
需要做**市值中性化的随机对照**才能判,那是下一步。」**本节就是那一步。**

═══ 对照怎么设才算对(§68 踩过的坑) ═══
§68 的疏漏是**把「>10年」当对照**,而该档系统性差于市场 ——
同规模随机对照的中位就有 **1.16** 而不是 1.00,等于把 1.3 的门槛设松了。

**本节的零假设写成:在同一个市值分档内,「年轻」不比「同档随机抽同样多只」更好。**

  对每个红轴月 t、每个市值五分位 b:
    young_b = 该档内 listed_days ∈ [365,1095) 的股票
    rand_b  = 该档内**任意年龄**随机抽 |young_b| 只(200 种子)
    两者都除以同一个 old_b 做归一,**分母完全相同,只有分子的年龄条件不同**

这样市值(按档)与样本量(同规模)都被控住,**只剩年龄这一个变量**。

═══ 事前锁定(不搜索、不调参) ═══
  牛熊      510300 月线 MACD(12,26,9) 柱正负(与 §68 完全一致,不重调)
  年龄      年轻 = listed_days ∈ [365, 1095);年老 = ≥ 3650
  市值分档  当月 float_mv 五分位
  指标      自月末起未来 250 日内最大累计涨幅 ≥ G,G ∈ {100%, 200%, 500%}
  种子      200
  红轴/绿轴 分别报(绿轴用于判断是否牛市专属)
  退市      前向填充参与,**不剔除**

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 红轴、≥100% 门槛:观测比值 > 随机比值,**p < 0.05**
  ② 三个门槛(100/200/500%)方向一致(观测 ≥ 随机中位)
  ③ 绿轴同口径一并报出 —— 若绿轴也显著,则**不是牛市专属**

  ①② 都过 → 年龄效应在市值中性化后仍然存在,§68 的 1.21 是真的
  ① 不过   → 1.21 与「同档随机」不可区分,**年龄效应基本可以归给市值**

**事前预测(写下以便被证伪)**:我预计 ① **勉强通过或不通过**。
理由:§68 加严后 1.43→1.21,而未中性化时随机基准已达 1.16 ——
中性化后观测与随机的差距会被进一步压缩。
**若 p 明显小于 0.05 且三门槛一致,说明年龄确实有独立于市值的信息,我错了。**

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
  §68 复现:红轴 1-3年/>10年 = 1.40(未中性化)
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
H, NSEED, SEED = 250, 200, 20260814
GAINS = (1.0, 2.0, 5.0)
Y_LO, Y_HI, O_LO = 365, 1095, 3650
NQ = 5

t0 = time.time()
cl, ld, mv = {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "listed_days", "float_mv"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
LD = pd.DataFrame(ld).set_axis(CL.index)
MV = pd.DataFrame(mv).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

CLa, LDa, MVa = CL.to_numpy(float), LD.to_numpy(float), MV.to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
CLf = pd.DataFrame(CLa).ffill().to_numpy(float)
FMAX = pd.DataFrame(CLf[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]

mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
M = mk["close"].resample("ME").last().dropna()
dif = M.ewm(span=12, adjust=False).mean() - M.ewm(span=26, adjust=False).mean()
hist = dif - dif.ewm(span=9, adjust=False).mean()
reg = {p: int(v > 0) for p, v in zip(hist.index.to_period("M"), hist)}
ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = [p for p in sorted(last_td) if last_td[p] + H < NT and p in reg]
print(f"可用月 {len(months)}(红轴 {sum(reg[p] for p in months)} / "
      f"绿轴 {sum(1-reg[p] for p in months)})")

rng = np.random.default_rng(SEED)
OUT = {}
for want, lab in ((1, "红轴"), (0, "绿轴")):
    mos = [p for p in months if reg[p] == want]
    for G in GAINS:
        oy, oo, rr = [], [], []
        for p in mos:
            t = last_td[p]
            base = ALIVE[t]
            hit = np.where(base, FMAX[min(t + H, NT - 1)] / CLa[t] - 1, np.nan) >= G
            age, m = LDa[t], np.where(base, MVa[t], np.nan)
            young = base & (age >= Y_LO) & (age < Y_HI)
            old = base & (age >= O_LO)
            q = np.nanquantile(m[base], np.linspace(0, 1, NQ + 1)[1:-1])
            ys, os_, rs = [], [], np.zeros(NSEED)
            nb = 0
            for i in range(NQ):
                lo = -np.inf if i == 0 else q[i - 1]
                hi = np.inf if i == NQ - 1 else q[i]
                band = base & (m > lo) & (m <= hi)
                by, bo = band & young, band & old
                if by.sum() < 10 or bo.sum() < 10:
                    continue
                ys.append(np.mean(hit[by]))
                os_.append(np.mean(hit[bo]))
                pool = np.flatnonzero(band)          # 同档、任意年龄
                n = int(by.sum())
                rs += np.array([np.mean(hit[rng.choice(pool, n, replace=False)])
                                for _ in range(NSEED)])
                nb += 1
            if nb:
                oy.append(np.mean(ys))
                oo.append(np.mean(os_))
                rr.append(rs / nb)
        oy, oo, rr = np.mean(oy), np.mean(oo), np.array(rr).mean(axis=0)
        obs = oy / oo
        rnd = rr / oo
        OUT[(lab, G)] = (obs, np.median(rnd), rnd.min(), rnd.max(),
                         float((rnd >= obs).mean()))
        print(f"  {lab} ≥{int(G*100)}% 完成  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n{'='*100}\n市值五分位内部:年轻 vs 同档任意年龄随机(分母都是同档年老)\n{'='*100}")
print(f"{'区间':<8}{'门槛':<10}{'观测':>9}{'随机中位':>10}{'随机区间':>20}{'p':>10}")
for lab in ("红轴", "绿轴"):
    for G in GAINS:
        o, md, lo, hi, p = OUT[(lab, G)]
        print(f"{lab:<8}{'≥'+str(int(G*100))+'%':<10}{o:>9.2f}{md:>10.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>20}{p:>10.4f}")
    print()

print(f"{'='*100}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*100}")
o1, _, _, _, p1 = OUT[("红轴", 1.0)]
c1 = p1 < 0.05
c2 = all(OUT[("红轴", G)][0] >= OUT[("红轴", G)][1] for G in GAINS)
p_bear = OUT[("绿轴", 1.0)][4]
print(f"  ① 红轴 ≥100%:p < 0.05            观测 {o1:.2f}, p={p1:.4f}   {'✓' if c1 else '✗'}")
print(f"  ② 三门槛方向一致                  "
      + " / ".join(f"{OUT[('红轴',G)][0]:.2f}vs{OUT[('红轴',G)][1]:.2f}" for G in GAINS)
      + f"   {'✓' if c2 else '✗'}")
print(f"  ③ 绿轴同口径(诊断)               p={p_bear:.4f}"
      f"  {'也显著 → 不是牛市专属' if p_bear < 0.05 else '不显著 → 牛市专属'}")
ok = c1 and c2
print(f"\n  **结论:{'年龄效应在市值中性化后仍然存在' if ok else '年龄效应基本可归给市值'}**")
if not ok:
    print("  → §68 的 1.21 与「同市值档随机」不可区分,")
    print("    「牛市买次新」的超额可以基本归给「次新股市值小」。")

pd.DataFrame([{"区间": k[0], "门槛": f"≥{int(k[1]*100)}%", "观测": v[0],
               "随机中位": v[1], "随机下限": v[2], "随机上限": v[3], "p": v[4]}
              for k, v in OUT.items()]).to_csv(f"{SP}/age_effect_mvneutral.csv", index=False)
print(f"\n→ {SP}/age_effect_mvneutral.csv   ({time.time()-t0:.0f}s)")
