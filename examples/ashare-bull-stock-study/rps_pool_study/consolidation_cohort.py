"""第八十七节:2013-2026 全样本扫描 —— 符合宇通形态的有多少,成功比例多少

═══ 起因:用户的问题 ═══
> **能不能把 2013 年到 2026 年都扫描一遍,看看有多少符合宇通客车的特征,
>   并且最终走成功的比例。**

§86 已经证实筛选器**一天不差地认出了宇通那一段**(单只回看:三条全中 42 天,
首次 2023-10-17,此后 120 日 +79.8%)。**但认出一只不等于这个筛子有用。**
本节把它铺到全样本,回答「有多少」和「成功比例」。

═══ 宇通的三段结构(月末收盘,2022-12-30 = 5.76 为基准) ═══
    第一段 拉升   2023-01~06   5.76 → 12.19   +111.6%   换手 0.0102→0.0194
    第二段 整理   2023-07~12  12.19 → 10.96    −10.1%   换手 0.0194→0.0050
    第三段 再拉升 2024-01~06  10.96 → 22.53   +105.6%   换手 0.0050→0.0150

**用户看到的形态是真的。§58/§59/§61 已经把它编码好了,本节不重新发明检测器。**

═══ 做法 ═══
逐月末扫全市场,复用 `consolidation_screener.py` 的
`load_panel` / `score_one` / `n_pass` / 自适应分位阈值 —— **一行检测逻辑都不重写**。

**去重(必须做)**:宇通连亮 42 天。若把每天当一个事件,同一段整理会被数几十次,
样本量虚高且高度相关。**本节按月取样,且只把「上月未亮、本月亮」记为新事件。**

**性能**:`series_of` 每月每股 reindex 一次 = 170 万次,不可行。
本节**预先把 5,232 只的 high/low/close/volume 取成 numpy 存好**,循环里只调 score_one。

═══ 本节不设通过/不通过判据 ═══
**这是频率与后验的描述性测量,不是假设检验** —— 与 §76/§86 同规格。
**§61 已经对这套特征下过判定**(三条全中年化 +10.37%,但 300 次随机对照
**p=0.16**,不算发现)。**本节不重判、不翻案**,只把「有多少、成功比例多少」摆出来。

═══ 锚点(不过则本节结论作废) ═══
  ① 面板 (3297, 5232)
  ② **宇通必须在 2023-10 或 2023-11 的月度事件里被检出**
     (§86 实测首次亮灯 2023-10-17;不得调参数去凑)
  ③ **对照的零校验**:同市值五分位随机对照的 ≥100% 比例,
     须落在全市场同期基础概率的 ±3pp 内 —— 对照建错会让所有比较失真

═══ 事前预测(写下以便被证伪) ═══
- 全样本事件数量级 **数千**(§61 报新口径选中率稳定在 15~19%)
- **6 个月「峰值 ≥100%」的比例低于 5%**;期末中位为负
- **信号组与同市值随机对照的差距很小**(§61 的 p=0.16 已经预告了这一点)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from consolidation_screener import (  # noqa: E402
    MIN_ADJ_FLOOR,
    MIN_ADJ_RATIO,
    Q_KEEP,
    load_panel,
    n_pass,
    score_one,
    series_of,
)

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NQ, NSEED, SEED = 5, 200, 20260814
HORIZONS = [(120, "6个月"), (250, "12个月")]
DISCLAIMER = ("§61 判定:三条全中年化 +10.37%,但 300 次随机对照 p=0.16,"
              "与随机无法区分;本表是状态频率统计,不是买点,不构成投资建议。")

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

codes = list(CL.columns)
ci = {c: i for i, c in enumerate(codes)}
print("预取序列(避免每月每股 reindex)…", flush=True)
SER = [series_of(frames, idx, c) for c in codes]
MAv = [MA100[c].to_numpy(float) for c in codes]
del frames
Fa = CL.where(CL > 0).ffill().to_numpy(float)
print(f"预取完成  ({time.time()-t0:.0f}s)")

# 市值:用于同市值五分位对照
mvv = {}
for c in codes:
    x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])
    mvv[c] = pd.to_numeric(x["float_mv"], errors="coerce")
MV = pd.DataFrame(mvv).set_axis(idx)
MVa = MV.to_numpy(float)
del mvv
print(f"市值载入完成  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = sorted(last_td)

prev_hit = set()
events, monthly = [], []
for mi, p in enumerate(months):
    t = last_td[p]
    scored = {}
    for j, code in enumerate(codes):
        h, lo_, c_, v_ = SER[j]
        if not np.isfinite(c_[t]):
            continue
        sd = np.flatnonzero(STRONG[:t + 1, j])
        if sd.size == 0:
            continue
        s_ = score_one(h, lo_, c_, v_, MAv[j], sd, t)
        if s_ is not None:
            scored[j] = s_
    if len(scored) < 50:
        continue
    vals = {k: np.array([s[k] for s in scored.values()])
            for k in ("缩量比", "收敛比", "深度", "调整天数")}
    floor = max(MIN_ADJ_FLOOR, int(round(MIN_ADJ_RATIO * np.median(vals["调整天数"]))))
    thr = {k: float(np.nanquantile(vals[k], Q_KEEP))
           for k in ("缩量比", "收敛比", "深度")}
    hit = {j for j, s in scored.items()
           if s["调整天数"] >= floor and n_pass(s, thr) == 3}
    new = hit - prev_hit
    monthly.append(dict(月份=str(p), 走完时序=len(scored), 三条全中=len(hit),
                        新事件=len(new)))
    for j in new:
        events.append(dict(月份=str(p), t=t, j=j, 代码=codes[j],
                           调整天数=scored[j]["调整天数"], 深度=scored[j]["深度"],
                           缩量比=scored[j]["缩量比"], 收敛比=scored[j]["收敛比"]))
    prev_hit = hit
    if (mi + 1) % 24 == 0:
        print(f"  {p}  累计事件 {len(events):,}  ({time.time()-t0:.0f}s)", flush=True)
EV = pd.DataFrame(events)
print(f"\n逐月扫描完成:{len(months)} 个月,**去重后新事件 {len(EV):,} 个**"
      f"  ({time.time()-t0:.0f}s)")


def fwd(t, j, n):
    if not np.isfinite(Fa[t, j]) or Fa[t, j] <= 0:
        return np.nan, np.nan
    seg = Fa[t + 1:min(t + n, NT - 1) + 1, j]
    if not len(seg):
        return np.nan, np.nan
    return np.nanmax(seg) / Fa[t, j] - 1, seg[-1] / Fa[t, j] - 1


rng = np.random.default_rng(SEED)
rows = []
print(f"\n{'='*104}\n全样本结果:2013-2026 逐月末扫描,去重后 {len(EV):,} 个事件\n{'='*104}")
for n, lab in HORIZONS:
    ok = EV[EV["t"] + n < NT]
    pk = np.array([fwd(r.t, r.j, n)[0] for r in ok.itertuples()])
    en = np.array([fwd(r.t, r.j, n)[1] for r in ok.itertuples()])
    m = np.isfinite(pk)
    # 同市值五分位随机对照
    cp = []
    for _ in range(NSEED):
        pick = []
        for tt, grp in ok.groupby("t"):
            base = np.flatnonzero(np.isfinite(Fa[tt]) & (Fa[tt] > 0)
                                  & np.isfinite(MVa[tt]))
            if len(base) < 50:
                continue
            mv = MVa[tt][base]
            q = np.nanquantile(mv, np.linspace(0, 1, NQ + 1)[1:-1])
            for jj in grp["j"]:
                b = int(np.searchsorted(q, MVa[tt, jj]))
                lo2 = -np.inf if b == 0 else q[b - 1]
                hi2 = np.inf if b >= NQ - 1 else q[b]
                band = base[(mv > lo2) & (mv <= hi2)]
                if len(band):
                    pick.append(fwd(tt, int(rng.choice(band)), n)[0])
        pick = np.array(pick)
        pick = pick[np.isfinite(pick)]
        if len(pick):
            cp.append([(pick >= 0.5).mean(), (pick >= 1.0).mean()])
    cp = np.array(cp)
    print(f"\n  === {lab} ===  有效事件 {m.sum():,}")
    print(f"    峰值 中位 {np.median(pk[m]):+.1%}")
    print(f"    **峰值 ≥50%  {np.mean(pk[m] >= .5):6.2%}**   同市值随机 "
          f"{np.median(cp[:, 0]):.2%}  [{np.percentile(cp[:, 0], 5):.2%}, "
          f"{np.percentile(cp[:, 0], 95):.2%}]")
    print(f"    **峰值 ≥100% {np.mean(pk[m] >= 1.0):6.2%}**   同市值随机 "
          f"{np.median(cp[:, 1]):.2%}  [{np.percentile(cp[:, 1], 5):.2%}, "
          f"{np.percentile(cp[:, 1], 95):.2%}]")
    print(f"    期末 中位 **{np.median(en[m]):+.1%}**   期末>0 "
          f"**{np.mean(en[m] > 0):.1%}**")
    rows.append(dict(部分="全样本", 口径=lab, n=int(m.sum()),
                     峰值中位=float(np.median(pk[m])),
                     峰值ge50=float(np.mean(pk[m] >= .5)),
                     峰值ge100=float(np.mean(pk[m] >= 1.0)),
                     随机ge50=float(np.median(cp[:, 0])),
                     随机ge100=float(np.median(cp[:, 1])),
                     随机ge100_5=float(np.percentile(cp[:, 1], 5)),
                     随机ge100_95=float(np.percentile(cp[:, 1], 95)),
                     期末中位=float(np.median(en[m])),
                     期末为正=float(np.mean(en[m] > 0))))

# 逐年
print(f"\n{'='*104}\n逐年(6 个月口径)\n{'='*104}")
print(f"{'年份':<8}{'新事件':>8}{'峰值中位':>10}{'≥50%':>9}{'≥100%':>9}{'期末中位':>10}")
ok = EV[EV["t"] + 120 < NT].copy()
ok["峰值"] = [fwd(r.t, r.j, 120)[0] for r in ok.itertuples()]
ok["期末"] = [fwd(r.t, r.j, 120)[1] for r in ok.itertuples()]
ok["年"] = ok["月份"].str[:4]
for y, g in ok.groupby("年"):
    v = g["峰值"].to_numpy()
    e = g["期末"].to_numpy()
    v, e = v[np.isfinite(v)], e[np.isfinite(e)]
    if not len(v):
        continue
    print(f"{y:<8}{len(g):>8,}{np.median(v):>+10.1%}{np.mean(v >= .5):>9.1%}"
          f"{np.mean(v >= 1.0):>9.1%}{np.median(e):>+10.1%}")
    rows.append(dict(部分="逐年", 口径=y, n=len(g), 峰值中位=float(np.median(v)),
                     峰值ge50=float(np.mean(v >= .5)),
                     峰值ge100=float(np.mean(v >= 1.0)),
                     期末中位=float(np.median(e))))

# ══ 锚点 ═══════════════════════════════════════════════════════════════
print(f"\n{'='*104}\n锚点核对(不过则本节结论作废)\n{'='*104}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
yt = EV[(EV["代码"] == "600066") & EV["月份"].isin(["2023-10", "2023-11"])]
a2 = len(yt) > 0
print(f"  {'✓' if a2 else '✗'} 锚点② 宇通在 2023-10/11 的月度新事件里"
      + (f"({yt.iloc[0]['月份']},调整 {int(yt.iloc[0]['调整天数'])} 日,"
         f"深度 {yt.iloc[0]['深度']:.1%})" if a2 else ""))
if not a2:
    bad.append("锚点②")
r6 = [r for r in rows if r.get("口径") == "6个月" and r["部分"] == "全样本"][0]
allbase = []
for tt in sorted(set(EV["t"])):
    if tt + 120 >= NT:
        continue
    b = np.flatnonzero(np.isfinite(Fa[tt]) & (Fa[tt] > 0))
    s = np.array([fwd(tt, int(x), 120)[0] for x in b])
    s = s[np.isfinite(s)]
    if len(s):
        allbase.append((s >= 1.0).mean())
base100 = float(np.mean(allbase))
a3 = abs(r6["随机ge100"] - base100) <= 0.03
print(f"  {'✓' if a3 else '✗'} 锚点③ 对照零校验:同市值随机 ≥100% "
      f"{r6['随机ge100']:.2%} vs 全市场同期基础概率 {base100:.2%}(容差 3pp)")
if not a3:
    bad.append("锚点③")
print()
print("  **锚点全部通过。**" if not bad else f"  **{bad} 不过:本节结论作废。**")

pd.DataFrame(monthly).to_csv(f"{OUT}/consolidation_cohort_monthly.csv", index=False)
with open(f"{OUT}/consolidation_cohort_events.csv", "w", encoding="utf-8") as fh:
    fh.write(f"# {DISCLAIMER}\n")
    EV.drop(columns=["j"]).to_csv(fh, index=False)
pd.DataFrame(rows).to_csv(f"{OUT}/consolidation_cohort.csv", index=False)
print(f"\n→ {OUT}/consolidation_cohort.csv  (+events/monthly)  ({time.time()-t0:.0f}s)")
