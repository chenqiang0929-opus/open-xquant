"""同一时刻、同样条件下,其他股票后来涨了多少(第七十六节)

═══ 用户的问题 ═══
「我很想知道,在这 3 个时间点,**满足同样的条件下,其他的股票后 250 日峰值是多少**。」

**这个问题问得对,而且比任何回测都直接** —— 它就是基础概率。
用户记住的是三只成功的;这个问题问的是「同一天、同样条件的那一批,整体怎么样」。

═══ 用户给的条件(原样实现,不改) ═══
  ① RPS50 ≥ 95              近 50 日涨幅全市场百分位
  ② 收盘 > MA100            站上 20 周线
  ③ 收盘 ≥ 250日最高 × 0.90 距新高 ≤ 10%
  ④ MA100 ≥ MA300 × 0.90    多头排列,接受 10% 差距
  ⑤ (附加层,单独报)量比 V20/V60 ≥ 1.0

**⚠️ 参数是拟合出来的,必须写在结论旁边:**
条件 ④ 的 10% 容差正好卡在生益电子头上 —— 它 2024-05-31 的
`MA100/MA300 = 9.69/10.74 = 0.9022`,**差 0.3% 就被排除**。
这是按一个案例定出来的容差。三个时点的结果**不能当作对信号的检验**,
只能当作「这三天的横截面长什么样」。真正的检验要看全样本(下一节)。

═══ 本节不设通过/不通过判据 ═══
这是一次**描述性测量**,不是假设检验。目的是把基础概率摆出来。

**唯一的硬要求(编码正确性锚点):**
三只点名股票必须出现在各自日期的满足条件集合里 ——
宁德时代 300750 @2019-12-31、生益电子 688183 @2024-05-31、宇通客车 600066 @2024-01-31。
**不出现说明条件编码错了,不得调容差去凑。**

═══ 三只在面板里的实测画像(§75 已核对) ═══
                 闸门   上市年  RPS250  RPS50  MA100>MA300  距新高    后250日峰值
  宁德 2019-12   红轴    1.56    74.4   98.5     True       −3.3%    +288.9%
  生益 2024-05   红轴    3.26    93.1   99.7     **False**  −8.6%    +199.7%
  宇通 2024-01   绿轴   26.75    99.4   99.0     True       −4.9%    +106.6%

**四项里三项对不上,唯一的共同项是 RPS50 ∈ [98.5, 99.7]。**

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
H, NQ, SEED = 250, 5, 20260814
RPS_MIN, NEAR_HIGH, MA_TOL, VOL_MIN = 95.0, 0.90, 0.90, 1.0
CASES = [("300750", "宁德时代", "2019-12-31"),
         ("688183", "生益电子", "2024-05-31"),
         ("600066", "宇通客车", "2024-01-31")]

t0 = time.time()
cl, vo, mv, ld = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "volume", "float_mv", "listed_days"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    vo[k] = pd.to_numeric(x["volume"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
VO = pd.DataFrame(vo).set_axis(CL.index)
MV = pd.DataFrame(mv).set_axis(CL.index)
LD = pd.DataFrame(ld).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

F = CL.ffill()
Fa = F.to_numpy(float)
ALIVE = CL.notna().to_numpy()
MA1 = F.rolling(100, min_periods=100).mean().to_numpy(float)
MA3 = F.rolling(300, min_periods=300).mean().to_numpy(float)
HI = F.rolling(250, min_periods=100).max().to_numpy(float)
V20 = VO.rolling(20, min_periods=10).mean().to_numpy(float)
V60 = VO.rolling(60, min_periods=30).mean().to_numpy(float)
MVa, LDa = MV.to_numpy(float), LD.to_numpy(float)
# 前瞻 250 日最大价(反向滚动 max,与 §67 同法);自 t+1 起,不含当日
FMAX = pd.DataFrame(Fa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]
print(f"指标完成  ({time.time()-t0:.0f}s)")

cols = list(CL.columns)
ci = {c: i for i, c in enumerate(cols)}
rng = np.random.default_rng(SEED)
BUCKETS = [(0.5, "≥50%"), (1.0, "≥100%"), (2.0, "≥200%"), (5.0, "≥500%")]


def peak(t, j):
    e = min(t + 1, NT - 1)
    if not (np.isfinite(Fa[t, j]) and Fa[t, j] > 0):
        return np.nan
    return FMAX[e, j] / Fa[t, j] - 1


def describe(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(n=len(v), med=np.median(v), mean=np.mean(v),
                **{lab: float((v >= g).mean()) for g, lab in BUCKETS})


def line(tag, d):
    if d is None:
        return f"  {tag:<14} —"
    return (f"  {tag:<14} n={d['n']:<5} 中位 {d['med']:+7.1%}  均值 {d['mean']:+7.1%}   "
            + "  ".join(f"{lab} {d[lab]:5.1%}" for _, lab in BUCKETS))


rows = []
for code, nm, ds in CASES:
    t = idx.get_indexer([pd.Timestamp(ds)], method="ffill")[0]
    base = ALIVE[t] & np.isfinite(HI[t]) & np.isfinite(MA3[t])
    last = Fa[t]
    r = pd.Series(np.where(base, last / Fa[t - 50] - 1, np.nan)).rank(pct=True) * 100
    rps = r.to_numpy(float)
    cond = (base & (rps >= RPS_MIN) & (last > MA1[t])
            & (last >= HI[t] * NEAR_HIGH) & (MA1[t] >= MA3[t] * MA_TOL))
    condv = cond & (V20[t] >= V60[t] * VOL_MIN)
    sel = np.flatnonzero(cond)
    jj = ci[code]

    print(f"\n{'='*104}\n{ds}   {nm}({code})\n{'='*104}")
    if not cond[jj]:
        print(f"  ✗ **{nm} 未被条件检出 —— 编码错了,不得调容差去凑**")
        print(f"     RPS50 {rps[jj]:.1f}  站上MA100 {last[jj]>MA1[t,jj]}  "
              f"距新高 {last[jj]/HI[t,jj]-1:+.1%}  "
              f"MA100/MA300 {MA1[t,jj]/MA3[t,jj]:.4f}")
        continue

    pk = np.array([peak(t, j) for j in sel])
    d_all = describe(pk)
    pkv = np.array([peak(t, j) for j in np.flatnonzero(condv)])
    d_vol = describe(pkv)

    # 同市值五分位随机对照:对每只入选标的,在其所在档内随机换一只
    m = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(m[base], np.linspace(0, 1, NQ + 1)[1:-1])
    ctrl = []
    for j in sel:
        b = int(np.searchsorted(q, m[j])) if np.isfinite(m[j]) else 0
        lo = -np.inf if b == 0 else q[b - 1]
        hi = np.inf if b >= NQ - 1 else q[b]
        band = np.flatnonzero(base & (m > lo) & (m <= hi))
        if len(band):
            ctrl.append(peak(t, int(rng.choice(band))))
    d_ctl = describe(np.array(ctrl))

    my = peak(t, jj)
    rank = int(np.nansum(pk > my)) + 1
    print(f"  ✓ {nm} 在检出集内   后 250 日峰值 **{my:+.1%}**   "
          f"排名 **{rank}/{d_all['n']}**(前 {rank/d_all['n']:.1%})")
    print()
    print(line("符合条件", d_all))
    print(line("+ 量比≥1.0", d_vol))
    print(line("同市值档随机", d_ctl))

    order = np.argsort(-np.where(np.isfinite(pk), pk, -np.inf))
    print(f"\n  该批峰值最高 5 只 / 最低 5 只:")
    for tag, ks in (("最高", order[:5]), ("最低", order[-5:][::-1])):
        s = "  ".join(f"{cols[sel[k]]} {pk[k]:+.0%}" for k in ks)
        print(f"    {tag}  {s}")

    rows.append({"日期": ds, "案例": nm, "案例峰值": my, "案例排名": rank,
                 "符合条件数": d_all["n"], "条件_中位": d_all["med"],
                 "条件_均值": d_all["mean"],
                 **{f"条件_{lab}": d_all[lab] for _, lab in BUCKETS},
                 "随机_中位": d_ctl["med"], "随机_均值": d_ctl["mean"],
                 **{f"随机_{lab}": d_ctl[lab] for _, lab in BUCKETS}})

if rows:
    R = pd.DataFrame(rows)
    print(f"\n{'='*104}\n汇总:三个时点的基础概率\n{'='*104}")
    print(f"{'日期':<13}{'案例':<10}{'案例峰值':>10}{'排名':>12}"
          f"{'该批中位':>10}{'随机中位':>10}{'该批≥100%':>11}{'随机≥100%':>11}")
    for _, r in R.iterrows():
        print(f"{r['日期']:<13}{r['案例']:<10}{r['案例峰值']:>+10.1%}"
              f"{f'{r.案例排名}/{r.符合条件数}':>12}"
              f"{r['条件_中位']:>+10.1%}{r['随机_中位']:>+10.1%}"
              f"{r['条件_≥100%']:>11.1%}{r['随机_≥100%']:>11.1%}")
    print(f"\n{'='*104}\n必须一起说的话\n{'='*104}")
    print("  · **参数是拟合出来的**:条件④ 的 10% 容差正好卡在生益电子头上")
    print("    (MA100/MA300 = 0.9022,差 0.3% 就被排除)。")
    print("  · **三个时点是用户挑出来的日期**,存在选择偏差 ——")
    print("    这三张表只描述「这三天长什么样」,**不构成对信号的检验**。")
    print("  · **峰值是上帝视角**:§70 实测浮盈兑现率只有 13%~55%,")
    print("    §66 七只案例约 18%(泰格 峰值 +1076% → 实收 +198%)。")
    print("    **右尾密度提高 ≠ 钱变多。**")
    R.to_csv(f"{SP}/case_date_baserate.csv", index=False)
    print(f"\n→ {SP}/case_date_baserate.csv   ({time.time()-t0:.0f}s)")
