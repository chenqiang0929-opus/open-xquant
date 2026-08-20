"""第九十九节:财报公告后还有没有肉 —— PEAD 在右尾口径下成不成立(事前登记)

═══ 起因:三个案例说明业绩占一半,但它是滞后信息 ═══
`case_yutong_why.py` / 本轮 301004、301061 的分解(描述性,已落库):

    嘉益 301004  ×7.88 = 估值 ×2.95 × 每股净资产 ×2.68   → 基本面 **48%** / 估值 52%
    匠心 301061  ×7.71,净利 3.35亿→8.57亿 ×2.56        → 基本面 **46%** / 估值 54%

**两只毫不相关的股票都接近 50/50 —— 业绩解释了约一半涨幅。**
**但它是滞后的:**

    嘉益  股价高点 2025-01-17;首次「净利同比转负」公告 2025-04-30,已跌 **−19.5%**
    匠心  股价高点 2025-08-20;首次转负公告 2026-04-30,已跌 **−47.5%**
    嘉益  净利同比 +394% 的公告(2023-08-16)时股价距高点 −69%;
          +225.7% 的公告(2024-10-31)时距高点仅 **−5.2%** —— 最漂亮的财报出来,行情已走完

**§98 已证「突破点上的盈利方向」分不出右尾(+0.25pp, p=0.0975)。
本节换一个更直接的问法:把财报公告日本身当事件,公告之后还有没有肉?
—— 即盈余公告后漂移(PEAD)在 A 股 ≥100% 右尾口径下成不成立。**

═══ 口径(事前锁定)═══
  事件    **净利润字段发生变更的交易日 = 财报公告日**
          变更判定两侧都有效(`isfinite(prev) & isfinite(cur) & (cur != prev)`)
          —— §97 用 `np.diff != 0` 把 NaN 段逐日计成变更,算出 15.91% 的垃圾
  变量    **净利同比 = ni[t] / |ni[t−250]| − 1**(250 个交易日前落在上年同一报告期)
  分档    **按公告所在年月做横截面五分位**(公告集中在 4/8/10 月,按月分档保证样本量)
  前瞻    **公告日收盘入场**,6 个月(120 日)峰值 ≥100% = 判据口径;12 个月仅描述
  对照A   同日同市值五分位随机 × 200 组
  对照B   同日同市值 **且当日也创 250 日新高** —— 隔离动量后的基准
  退市股按最后有效价 ffill 参与,绝不剔除

═══ 锚点(不过则全节作废;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **公告日校验(无前视)**:全市场 roe 变更落在 6 月或 12 月的比例 **< 2%**
     —— 修正算法后**已预验证实测 0.27%**(178,452 次变更,6-30/12-31 仅 0.017%)
  ③ **单只恒等复现**:嘉益 301004 在 **2024-10-31** 的净利同比 = **+225.7%**
     (单只已实测,全样本管线必须复现,容差 ±0.5pp)
  ④ **恒等零校验**:各档对照的中位命中率 vs 同档总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽)═══
  **前置条件**:某档事件数 **< 300** 不判;逐年某年 **< 100** 不计入判据③
  ① **对照A**:最高档(净利同比 Q5)6 个月 ≥100% 的 **lift ≥ 1.3 且 p < 0.05/4 = 0.0125**
  ② **对照B**(隔离动量):同档 **lift ≥ 1.3 且 p < 0.0125**
  ③ **逐年一致**(§91 立的规矩):逐年 Q5 对对照A 的 **lift > 1.0 的年份占比 ≥ 80%**

**①②③ 全过 = PEAD 在右尾口径下成立且独立于动量 —— 本项目第一个站住的非价格因子。
① ③ 过而 ② 不过 = 财报后的漂移就是动量本身,与 §89/§92/§94/§98 同构。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问:什么会让它通过而不回答问题?**
→ 高增长公告的股票本来就在涨(动量)→ **堵法:判据② 用对照B**。
→ Q5 集中在牛市年份 → **堵法:对照按同日抽 + 判据③ 逐年**。
→ 前视 → **堵法:锚点② 公告日恒等校验,已预验证 0.27%**。
→ 5 档搜索出假阳性 → **堵法:判据只压在 Q5(事前指定,不是挑出来的)+ Bonferroni 0.05/4**。

**反问:什么会让它不通过而与问题无关?**
→ 分档样本不足 → **堵法:按月横截面分档 + 前置 n≥300**。
→ 净利同比在亏损转盈时爆表(分母趋零)→ **用 |ni[t−250]| 做分母;
  且分档是「排序」不是「阈值」,极端值不影响档位边界**。
→ 锚点误杀正确实现 → **堵法:四个锚点全是恒等式,②③ 均已单独预验证**。

═══ 事前预测(写下以便被证伪)═══
**① 通过、② 不通过、③ 不通过。**
理由:PEAD 是文献里最稳健的异象之一,对同市值随机(对照A)应当看得到;
**但盈余动量与价格动量高度相关,一旦对照换成「当日也在创 250 日新高的同市值股」,
我预计增量归零 —— 这将是同一个答案第五次出现**(§89 0.97、§92 p=0.42~0.75、
§94 1.01、§98 1.02)。③ 我预测不通过,因为本项目每个逐年检验都翻过号。
**若 ② 通过,那 PEAD 就是这个项目找到的第一个独立于动量的右尾来源,我错了。**
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
from consolidation_screener import load_panel  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NQ, NSEED, SEED = 5, 200, 20260814
MIN_N, MIN_N_YEAR, NCELL = 300, 100, 4
ALPHA = 0.05 / NCELL
LIFT_MIN, YR_FRAC = 1.3, 0.80
HOR = [(120, "6个月"), (250, "12个月")]

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    CL = CL.drop(columns=["510300"])
del frames, STRONG, MA100
idx = CL.index
NT, NS = CL.shape
codes = list(CL.columns)
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

Fa = CL.where(CL > 0).ffill().to_numpy(float)
HI250 = pd.DataFrame(Fa).rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(HI250) & (Fa >= HI250 * 0.9999)
del HI250

FLD = ["net_income", "roe", "float_mv"]
raw = {c: pd.read_parquet(f"{DATA}/{c}.parquet", columns=FLD) for c in codes}
M = {}
for f in FLD:
    df = pd.DataFrame({c: pd.to_numeric(v[f], errors="coerce") for c, v in raw.items()})
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    M[f] = df.reindex(idx.tz_localize(None)).ffill().to_numpy(float)
del raw
print(f"基本面矩阵完成  ({time.time()-t0:.0f}s)", flush=True)

ROE = M["roe"]
_p, _c = ROE[:-1], ROE[1:]
chg_roe = np.isfinite(_p) & np.isfinite(_c) & (_c != _p)
mon = np.array([q.month for q in idx[1:]])
n_all, n612 = int(chg_roe.sum()), int(chg_roe[np.isin(mon, [6, 12])].sum())
a2 = n_all > 0 and n612 / n_all < 0.02
print(f"  {'✓' if a2 else '✗'} 锚点② 公告日校验:roe 变更 {n_all:,} 次,"
      f"落 6/12 月 {n612:,} = {n612/max(n_all,1):.2%} < 2%")

NI = M["net_income"]
YOY = np.full((NT, NS), np.nan)
den = np.abs(NI[:-250])
YOY[250:] = np.where(den > 0, NI[250:] / np.where(den > 0, den, 1) - 1, np.nan)
_p, _c = NI[:-1], NI[1:]
ANN = np.zeros((NT, NS), bool)
ANN[1:] = np.isfinite(_p) & np.isfinite(_c) & (_c != _p)

J = codes.index("301004")
tj = int(np.flatnonzero(idx == pd.Timestamp("2024-10-31", tz=idx.tz))[0])
a3 = abs(YOY[tj, J] - 2.257) <= 0.005
print(f"  {'✓' if a3 else '✗'} 锚点③ 嘉益 2024-10-31 净利同比 {YOY[tj, J]:+.1%}(期望 +225.7%)")

MV = M["float_mv"]
QUINT = np.full((NT, NS), -1, dtype=np.int8)
for t in range(NT):
    ok = np.isfinite(MV[t]) & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if ok.sum() < 50:
        continue
    QUINT[t, ok] = np.searchsorted(np.nanquantile(MV[t][ok], [.2, .4, .6, .8]),
                                   MV[t][ok], side="right")
del M, MV, NI, ROE


def fwd_peak(n):
    m = pd.DataFrame(Fa[::-1]).rolling(n, min_periods=1).max().to_numpy(float)[::-1]
    out = np.full((NT, NS), np.nan)
    out[:-1] = m[1:]
    out = (out / Fa - 1.0).astype(np.float32)
    out[NT - n:] = np.nan
    return out


PK = {n: fwd_peak(n) for n, _ in HOR}
ym = idx.to_period("M")
EV = [(int(t), int(j)) for t, j in zip(*np.where(ANN), strict=True)
      if np.isfinite(YOY[t, j])]
print(f"\n财报公告事件 **{len(EV):,}** 个(净利同比可算)  ({time.time()-t0:.0f}s)", flush=True)

bym = {}
for t, j in EV:
    bym.setdefault(ym[t], []).append((t, j))
BUCK = {}
for m, evs in bym.items():
    v = np.array([YOY[t, j] for t, j in evs])
    if len(v) < 25:
        continue
    e = np.nanquantile(v, [.2, .4, .6, .8])
    for (t, j), q in zip(evs, np.searchsorted(e, v, side="right"), strict=True):
        BUCK.setdefault(int(q), []).append((t, j))
print("  按月横截面五分位:" + "  ".join(f"Q{q+1}={len(BUCK.get(q,[])):,}" for q in range(5)))

rng = np.random.default_rng(SEED)


def control(sub, pm, newhi):
    cnt = {}
    for t, j in sub:
        q = int(QUINT[t, j])
        if q >= 0:
            cnt[(t, q)] = cnt.get((t, q), 0) + 1
    hit, tot, th, tn = np.zeros(NSEED), 0, 0.0, 0
    for (t, q), k in cnt.items():
        pool = np.flatnonzero((QUINT[t] == q) & np.isfinite(pm[t]))
        if newhi:
            pool = pool[NEWHI[t, pool]]
        if pool.size == 0:
            continue
        v = pm[t, pool] >= 1.0
        th += float(v.mean()) * k
        tn += k
        hit += v[rng.integers(0, pool.size, size=(NSEED, k))].sum(axis=1)
        tot += k
    if tot == 0:
        return np.array([]), np.nan
    return hit / tot, th / tn


W = 106
rows, MAIN = [], {}
for n, hname in HOR:
    pm = PK[n]
    print(f"\n{'='*W}\n{hname}峰值 ≥100%(公告日收盘入场)"
          f"{'  【判据口径】' if n == 120 else '  (描述)'}\n{'='*W}")
    print(f"{'档':<5}{'净利同比中位':>13}{'事件数':>9}{'≥100%':>9}{'对照A':>9}{'liftA':>7}"
          f"{'pA':>8}{'对照B':>9}{'liftB':>7}{'pB':>8}{'零校验':>8}")
    for q in range(5):
        sub = [(t, j) for t, j in BUCK.get(q, []) if np.isfinite(pm[t, j])]
        if not sub:
            continue
        v = np.array([pm[t, j] for t, j in sub]) >= 1.0
        obs = float(v.mean())
        med = float(np.nanmedian([YOY[t, j] for t, j in sub]))
        ca, tha = control(sub, pm, False)
        cb, _ = control(sub, pm, True)
        ra = float(np.median(ca)) if ca.size else np.nan
        rb = float(np.median(cb)) if cb.size else np.nan
        r = dict(前瞻=hname, 档=f"Q{q+1}", 同比中位=med, 事件数=len(sub), ge100=obs,
                 对照A=ra, liftA=obs / ra if ra > 0 else np.nan,
                 pA=float((ca >= obs).mean()) if ca.size else np.nan,
                 对照B=rb, liftB=obs / rb if rb > 0 else np.nan,
                 pB=float((cb >= obs).mean()) if cb.size else np.nan,
                 零校验=abs(ra - tha))
        rows.append(r)
        print(f"Q{q+1:<4}{med:>13.1%}{len(sub):>9,}{obs:>9.2%}{ra:>9.2%}"
              f"{r['liftA']:>7.2f}{r['pA']:>8.4f}{rb:>9.2%}{r['liftB']:>7.2f}"
              f"{r['pB']:>8.4f}{r['零校验']:>8.2%}")
        if n == 120 and q == 4:
            MAIN = dict(r=r, sub=sub)
R = pd.DataFrame(rows)

print(f"\n{'='*W}\n逐年:Q5(净利同比最高档)对对照A\n{'='*W}")
pm = PK[120]
yr = []
for y in sorted({idx[t].year for t, _ in MAIN["sub"]}):
    ev = [(t, j) for t, j in MAIN["sub"] if idx[t].year == y]
    if len(ev) < MIN_N_YEAR:
        continue
    o = float((np.array([pm[t, j] for t, j in ev]) >= 1.0).mean())
    c, _ = control(ev, pm, False)
    md = float(np.median(c)) if c.size else np.nan
    lf = o / md if md > 0 else np.nan
    yr.append(dict(年=y, n=len(ev), ge100=o, 对照A=md, lift=lf))
    print(f"  {y}  n={len(ev):>5,}  ≥100% {o:>6.2%}  对照A {md:>6.2%}  "
          f"lift {lf:>5.2f}  {'✓' if lf > 1.0 else '✗'}")
Y = pd.DataFrame(yr)
yfrac = float((Y["lift"] > 1.0).mean()) if len(Y) else np.nan

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② 公告日校验(无前视)")
print(f"  {'✓' if a3 else '✗'} 锚点③ 嘉益 2024-10-31 净利同比恒等复现")
z = R[R["前瞻"] == "6个月"]["零校验"]
a4 = bool(z.notna().all() and (z <= 0.03).all())
print(f"  {'✓' if a4 else '✗'} 锚点④ 恒等零校验 最大差 "
      f"{z.max():.2%} ≤ 3pp" if z.notna().all() else "  ✗ 锚点④ 算不出 = 不通过")
for ok, nm in ((a2, "锚点②"), (a3, "锚点③"), (a4, "锚点④")):
    if not ok:
        bad.append(nm)

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*W}")
r = MAIN["r"]
print(f"  前置条件:Q5 事件数 {r['事件数']:,} ≥ {MIN_N};逐年合格 {len(Y)} 年")
c1 = bool(r["事件数"] >= MIN_N and r["liftA"] >= LIFT_MIN and r["pA"] < ALPHA)
c2 = bool(r["事件数"] >= MIN_N and r["liftB"] >= LIFT_MIN and r["pB"] < ALPHA)
c3 = bool(np.isfinite(yfrac) and yfrac >= YR_FRAC)
print(f"  {'✓' if c1 else '✗'} 判据① 对照A  lift {r['liftA']:.2f} ≥ {LIFT_MIN} "
      f"且 p {r['pA']:.4f} < {ALPHA}")
print(f"  {'✓' if c2 else '✗'} 判据② 对照B  lift {r['liftB']:.2f} ≥ {LIFT_MIN} "
      f"且 p {r['pB']:.4f} < {ALPHA}")
print(f"  {'✓' if c3 else '✗'} 判据③ 逐年 lift>1.0 占比 {yfrac:.1%} ≥ {YR_FRAC:.0%}")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:PEAD 在右尾口径下成立且独立于动量 —— 本项目第一个站住的非价格因子。**")
    print("  **事前预测被证伪 —— 我错了。**")
elif c1:
    print("  **结论:财报后的漂移对同市值随机有效,但对隔离动量的对照归零 ——")
    print("     它就是动量本身。同一个答案第五次出现。事前预测命中。**")
else:
    print("  **结论:财报公告后在右尾口径下没有漂移。**")

R.to_csv(f"{OUT}/pead_righttail.csv", index=False)
Y.to_csv(f"{OUT}/pead_righttail_yearly.csv", index=False)
print(f"\n→ {OUT}/pead_righttail.csv + _yearly.csv   ({time.time()-t0:.0f}s)")
