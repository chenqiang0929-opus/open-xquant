"""第八十六节:宇通客车案例解剖 —— 用一个具体问题回答「能不能预测」

═══ 用户的提问 ═══
> **2023 年的宇通客车,我有没有办法在 2024-01 出现买入信号,
>   预测到未来 6 个月涨幅达到 100%?**

**这是整个项目里最好的一个提问。** 它不依赖任何统计设计 ——
不需要 lift、不需要 GPD、不需要秩相关,只要把那一天那一批股票摆出来。
在 §83/§85 连续两节因**判据设计**作废之后,这条路径的价值更高:
**它几乎没有可以被我写坏的判据。**

═══ 本节要回答三件事 ═══
① **前提成不成立**:宇通客车 6 个月内到底有没有涨到 100%?
② **信号那天亮了吗、那批股票整体怎么样**?
③ **宇通在那批里是什么位置**?

═══ 与 §76 的关系 ═══
§76 已经报过这三只在 **250 日**口径下的表格
(宇通 +106.6%、排 3/105、该批 ≥100% 3.8% vs 随机 18.1%)。
**本节补的是 6 个月口径**,并把 §76 的随机对照从**单次抽签**改成**多种子**。

**一处必须先讲清的实现细节**:§76 的对照是
「对每只入选标的,在其所在市值档内**随机换一只**」—— **只抽一次**。
n=105、比率 ~15% 时单次抽样标准差约 **3.5pp**,
所以 §76 报的 15.4% / 18.4% / 18.1% 这几个随机基准**本身带 ±3~4pp 噪音**。
本节用 **NSEED=200** 把它钉死,并把单次版一并报出用于对账。

═══ 口径(与 §76 逐行一致,不重调) ═══
  条件   RPS50≥95 且 收盘>MA100 且 收盘≥250日高×0.90 且 MA100≥MA300×0.90
  base   ALIVE & isfinite(HI250) & isfinite(MA300),**HI 用 min_periods=100**(§76 原样)
  峰值   自 t+1 起、未来 H 日内最大累计涨幅(反向滚动 max,§67/§76 同法)
  对照   同市值五分位内随机换一只 × **NSEED 次**
  H      60 / **120(6个月)** / 180 / 250

═══ 本节不设通过/不通过判据 ═══
与 §76 同规格(§76 原文:「本节不设通过/不通过判据……目的是把基础概率摆出来」)。
**但硬性正确性锚点必须有,且不得事后调参数去凑。**

═══ 锚点(不过则本节结论作废) ═══
  ① 面板 (3297, 5232)
  ② **三只案例必须被四条件检出**(§76/§77 的编码锚点)
  ③ **§76 复现**:250 日口径下,该批 ≥100% 比例 = **3.8%**、宇通排 **3/105**
     (容差 ±0.5pp / ±1 名)。对不上说明实现与 §76 有别,须查清再落库。

═══ 事前预测(写下以便被证伪) ═══
- **6 个月口径下三只的峰值都不到 +100%**(宇通预跑已知 +94.8%)
- 多种子对照后,250 日的随机 ≥100% 会落在 **11%~18%** 之间
  —— 若落在这个区间,说明 §76 的 18.1% 与我预跑的 11.4% 都只是单次抽样的两端
"""
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
np.seterr(invalid="ignore", divide="ignore")

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
NQ, SEED, NSEED = 5, 20260814, 200
RPS_MIN, NEAR_HIGH, MA_TOL = 95.0, 0.90, 0.90
HORIZONS = [(60, "3个月"), (120, "6个月"), (180, "9个月"), (250, "12个月/250日")]
CASES = [("300750", "宁德时代", "2019-12-31"),
         ("688183", "生益电子", "2024-05-31"),
         ("600066", "宇通客车", "2024-01-31")]

t0 = time.time()
cl, mv = {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "float_mv"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
MV = pd.DataFrame(mv).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

F = CL.ffill()
Fa = F.to_numpy(float)
ALIVE = CL.notna().to_numpy()
MA1 = F.rolling(100, min_periods=100).mean().to_numpy(float)
MA3 = F.rolling(300, min_periods=300).mean().to_numpy(float)
HI = F.rolling(250, min_periods=100).max().to_numpy(float)     # §76 原样
MVa = MV.to_numpy(float)
FMX = {h: pd.DataFrame(Fa[::-1]).rolling(h, min_periods=1).max().to_numpy(float)[::-1]
       for h, _ in HORIZONS}
ci = {c: i for i, c in enumerate(CL.columns)}
print(f"指标完成  ({time.time()-t0:.0f}s)")
rows = []


def peak(t, j, h):
    if not (np.isfinite(Fa[t, j]) and Fa[t, j] > 0):
        return np.nan
    return FMX[h][min(t + 1, NT - 1), j] / Fa[t, j] - 1


# ══ A 部分:宇通客车的真实路径 ═════════════════════════════════════════
print(f"\n{'='*100}\nA 部分:宇通客车 600066 从 2024-01-31 起的真实路径\n{'='*100}")
jy = ci["600066"]
ty = idx.get_indexer([pd.Timestamp("2024-01-31")], method="ffill")[0]
p0 = Fa[ty, jy]
ser = pd.Series(Fa[:, jy], index=idx)
print(f"  基准日 {idx[ty].date()}  收盘 {p0:.2f}")
print(f"\n{'口径':<14}{'期末':>10}{'期间峰值':>10}{'兑现率':>9}{'峰值日期':>14}")
for h, lab in HORIZONS:
    seg = ser.iloc[ty + 1:ty + 1 + h]
    pk, end = seg.max() / p0 - 1, seg.iloc[-1] / p0 - 1
    rr = end / pk if pk > 0 else np.nan
    print(f"{lab:<14}{end:>+10.1%}{pk:>+10.1%}{rr:>9.0%}{str(seg.idxmax().date()):>14}")
    rows.append(dict(部分="A", 口径=lab, 期末=float(end), 峰值=float(pk),
                     兑现率=float(rr)))
seg250 = ser.iloc[ty + 1:ty + 1 + 250]
hit = seg250[seg250 / p0 - 1 >= 1.0]
first = (list(seg250.index).index(hit.index[0]) + 1) if len(hit) else None
print(f"\n  **首次触及 +100%:{hit.index[0].date() if len(hit) else '250 日内从未'}"
      + (f" = 第 {first} 个交易日**" if first else "**"))
rows.append(dict(部分="A", 口径="首次触及100%",
                 期末=float(first) if first else np.nan))

# ══ B/C 部分:三只案例各自的同批分布 ═══════════════════════════════════
rng = np.random.default_rng(SEED)
anchor_ok = {"检出": True, "复现": None}
for code, nm, ds in CASES:
    t = idx.get_indexer([pd.Timestamp(ds)], method="ffill")[0]
    j = ci[code]
    base = ALIVE[t] & np.isfinite(HI[t]) & np.isfinite(MA3[t])
    last = Fa[t]
    rps = (pd.Series(np.where(base, last / Fa[t - 50] - 1, np.nan))
           .rank(pct=True).to_numpy(float) * 100)
    cond = (base & (rps >= RPS_MIN) & (last > MA1[t])
            & (last >= HI[t] * NEAR_HIGH) & (MA1[t] >= MA3[t] * MA_TOL))
    sel = np.flatnonzero(cond)
    print(f"\n{'='*100}\n{ds}   {nm}({code})   条件检出 {len(sel)} 只\n{'='*100}")
    if not cond[j]:
        print(f"  ✗ **{nm} 未被检出 —— 编码错了,不得调容差去凑**")
        anchor_ok["检出"] = False
        continue
    m = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(m[base], np.linspace(0, 1, NQ + 1)[1:-1])
    bands = {}
    for jj in sel:
        b = int(np.searchsorted(q, m[jj])) if np.isfinite(m[jj]) else 0
        lo = -np.inf if b == 0 else q[b - 1]
        hi = np.inf if b >= NQ - 1 else q[b]
        bands[jj] = np.flatnonzero(base & (m > lo) & (m <= hi))
    print(f"{'口径':<14}{'该批≥100%':>11}{'随机中位':>10}{'随机5%~95%':>16}"
          f"{'该批中位':>10}{'随机中位涨幅':>13}{nm+'排名':>12}")
    for h, lab in HORIZONS:
        pk = np.array([peak(t, x, h) for x in sel])
        pk = pk[np.isfinite(pk)]
        my = peak(t, j, h)
        rank = int(np.nansum(pk > my)) + 1
        hits, meds = [], []
        for _ in range(NSEED):
            c = np.array([peak(t, int(rng.choice(bands[jj])), h) for jj in sel])
            c = c[np.isfinite(c)]
            hits.append(float((c >= 1.0).mean()))
            meds.append(float(np.median(c)))
        hits = np.array(hits)
        lo5, hi95 = np.percentile(hits, [5, 95])
        print(f"{lab:<14}{(pk >= 1.0).mean():>11.1%}{np.median(hits):>10.1%}"
              f"{f'[{lo5:.1%}, {hi95:.1%}]':>16}{np.median(pk):>+10.1%}"
              f"{np.median(meds):>+13.1%}{f'{rank}/{len(pk)}':>12}")
        rows.append(dict(部分="B", 案例=nm, 日期=ds, 口径=lab, n=len(pk),
                         该批ge100=float((pk >= 1.0).mean()),
                         随机ge100中位=float(np.median(hits)),
                         随机ge100_5=float(lo5), 随机ge100_95=float(hi95),
                         该批中位=float(np.median(pk)),
                         随机中位=float(np.median(meds)),
                         案例峰值=float(my), 排名=rank))
        if code == "600066" and h == 250:
            anchor_ok["复现"] = (float((pk >= 1.0).mean()), rank, len(pk))

# ══ 锚点核对 ═══════════════════════════════════════════════════════════
print(f"\n{'='*100}\n锚点核对(不过则本节结论作废)\n{'='*100}")
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if anchor_ok['复现'] and anchor_ok['检出'] else '✗'} "
      f"锚点② 三只案例全部被四条件检出")
rep = anchor_ok["复现"]
if rep is None:
    print("  ✗ 锚点③ 算不出 —— 不通过")
    ok3 = False
else:
    r_pct, r_rank, r_n = rep
    ok3 = abs(r_pct - 0.038) <= 0.005 and abs(r_rank - 3) <= 1
    print(f"  {'✓' if ok3 else '✗'} 锚点③ 复现 §76:250日该批 ≥100% "
          f"{r_pct:.1%}(期望 3.8%)、宇通排 {r_rank}/{r_n}(期望 3/105)")
print()
if not (anchor_ok["检出"] and ok3):
    print("  **锚点不过:本节结论作废。**")
else:
    print("  **锚点全部通过。**")

pd.DataFrame(rows).to_csv(f"{SP}/case_yutong_6m.csv", index=False)
print(f"\n→ {SP}/case_yutong_6m.csv   ({time.time()-t0:.0f}s)")
