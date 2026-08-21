"""第九十一节:0.70 的真相 —— 「当日创250日新高」本身是不是右尾因子(事前登记)

═══ 起因:我把 §89 的 0.70 说反了一半,先更正 ═══
§89 的状态格对**对照B**(同日同市值**且当日也创250日新高**)lift = **0.70**,
我在对话里说成「正在整理的股票有右尾折价」。**这个说法是错的。**
同一张表里,状态格对**对照A**(同日同市值随机)的 lift 是 **1.04 / 0.95** ——
**整理股恰好等于平均水平,没有任何折价。**

真正发生的事是另一件:**对照B 那一组自己高**。用 §89 四格的对照数字算:

    legacy|状态    新高组 7.74%   全体 5.23%   **新高溢价 1.48x**
    adaptive|状态  新高组 8.07%   全体 5.92%   **1.36x**
    legacy|突破    新高组 7.64%   全体 5.62%   **1.36x**
    adaptive|突破  新高组 8.10%   全体 6.19%   **1.31x**

**0.70 = 1 ÷ 1.43,是「新高组有溢价而整理股没有」,不是「整理股被折价」。**

**而这个溢价恰好就是用户第一段定义里的那一条:「突破 250 新高」。**
它在 §89 里只是对照组的副产品,从来没被当成信号测过。本节把它扶正来测。

═══ 口径(事前锁定) ═══
  信号    **月末创 250 日新高**(收盘 ≥ 过去 250 日最高 × 0.9999,min_periods=100)
          取样点用月末,与 §89 状态格同频,避免同一段行情被日频重复计数
  对照A   同月末、**同市值五分位**随机 × 200 组
  对照C   同月末、同市值五分位、**且同 RPS60 五分位**随机 × 200 组
          —— **隔离「已经涨了」之后,「创新高」还剩多少信息**
  RPS60   `CL.pct_change(60).rank(axis=1, pct=True) * 100`,逐字取自
          `consolidation_screener.load_panel` 源码;锚点② 用恒等式证明没抄错。
          **必须在删掉 510300 之前算** —— 横截面分位的分母是全表列数,
          先删后算会让分母从 5233 变 5232,锚点② 必挂(合成数据实测 33 格不一致)
  前瞻    6 个月(120 日)峰值 ≥100% = 判据口径;12 个月仅描述

═══ 锚点(不过则全节作废) ═══
  ① 面板 (3297, 5232)
  ② **RPS60 恒等校验**:自算的 `(RPS60 > 90)` 与 `load_panel` 返回的 STRONG
     矩阵**逐格相同**(证明重算的分位定义与筛选器一致,没自拼错)
  ③ **无前视校验**:把面板截断到 2020-12-31 重算 NEWHI,
     与全样本版本在 `t ≤ 2020-12-31` 上**逐格相同**(证明新高判定不看未来)
  ④ **恒等零校验**:对照A / 对照C 的中位命中率 vs 各自同格总体命中率,差 ≤ 3pp

**三个锚点都是恒等式 —— 实现正确必过、写错必抓。**
§85/§87/§88 三次作废都栽在「复现一个精确数字」型锚点上(§89 已立规矩)。

═══ 事前判据(跑之前写死,不放宽) ═══
  **前置条件**:事件数 **< 300** 不判;分年检验中某年 **< 100** 的年份不计入
  ① 新高组对**对照A**:6 个月 ≥100% 的 **lift ≥ 1.3 且 p < 0.05/4 = 0.0125**
  ② 新高组对**对照C**(隔离 RPS60 分位):**同样 lift ≥ 1.3 且 p < 0.0125**
  ③ **稳定性**:逐年对对照A 的 lift **> 1.0 的年份占比 ≥ 80%**

**①③ 过而 ② 不过 = 新高溢价是真的,但它只是动量的代理(与 §89 第三段同构)。
①②③ 全过 = 「创新高」携带超出单纯涨幅的信息,那是一个独立因子。**

═══ 判据自查(§79 正问 + §83 反问) ═══
**正问:什么会让它通过而不回答我的问题?**
→ 创新高 = 已经涨了,lift 可能纯粹是 60 日涨幅的代理 → **堵法:判据② 用对照C**。
→ 新高股集中在牛市年份 → **堵法:对照按同月末抽,月份构成天然配平;判据③ 逐年**。
→ 新高股偏小盘/偏大盘 → **堵法:对照A/C 都在同市值五分位内抽**。
→ 4 次比较搜出假阳性 → **堵法:Bonferroni 0.05/4**。

**反问:什么会让它不通过而与问题无关?**
→ 对照C 的格子被切得太碎、池子太小 → **堵法:池子 < 20 只的格子跳过,并打印跳过占比**。
→ 分年样本不足 → **堵法:每年 n≥100 才计入判据③**。
→ 锚点写成「复现精确数字」而误杀正确实现(§85/§87/§88 病根)→
  **堵法:本节三个锚点全部是恒等式**。

═══ 事前预测(写下以便被证伪) ═══
**① 通过、③ 通过、② 不通过。**
理由:§89 已经在四个不同事件集上一致看到 1.31~1.48x 的新高溢价(① 会过);
但 §62 层一实测「把股票推上 RPS60>90 的只有换手率和涨停,启动前形态 lift 0.95」,
§89 又证明突破格的全部优势对隔离动量的对照归零 —— **「创新高」很可能只是
「60 日涨幅高」的另一种写法,一旦按 RPS60 分位配平就没了(② 不过)。**
**若 ② 也通过,说明创新高携带超出涨幅的信息,我错了 ——
那会是本项目第一个独立于动量的右尾因子。**
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
MIN_N, MIN_N_YEAR, MIN_POOL, NCELL = 300, 100, 20, 4
ALPHA = 0.05 / NCELL
LIFT_MIN, YEAR_FRAC = 1.3, 0.80
HOR = [(120, "6个月"), (250, "12个月")]
CUT = "2020-12-31"

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
# 逐字取自 consolidation_screener.load_panel。**必须在删 510300 之前算** ——
# 横截面分位的分母是全表列数,先删后算会让 5233 变 5232,锚点② 必挂(已实测 33 格不一致)
RPS60 = (CL.pct_change(60).rank(axis=1, pct=True) * 100).to_numpy(float)
if "510300" in CL.columns:
    k = list(CL.columns).index("510300")
    STRONG = np.delete(STRONG, k, axis=1)
    RPS60 = np.delete(RPS60, k, axis=1)
    CL = CL.drop(columns=["510300"])
del frames, MA100
idx = CL.index
NT, NS = CL.shape
codes = list(CL.columns)
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

a2 = bool(np.array_equal(RPS60 > 90, STRONG))
print(f"  {'✓' if a2 else '✗'} 锚点② RPS60 恒等校验:(RPS60>90) == STRONG")

Fa = CL.where(CL > 0).ffill().to_numpy(float)


def newhi(px):
    hi = pd.DataFrame(px).rolling(250, min_periods=100).max().to_numpy(float)
    return np.isfinite(hi) & (px >= hi * 0.9999)


NEWHI = newhi(Fa)
kc = int(np.searchsorted(idx, pd.Timestamp(CUT, tz=idx.tz), side="right"))
a3 = bool(np.array_equal(newhi(Fa[:kc]), NEWHI[:kc]))
print(f"  {'✓' if a3 else '✗'} 锚点③ 无前视校验:截断到 {CUT}(前 {kc} 行)重算 NEWHI 一致")

mvv = {c: pd.to_numeric(pd.read_parquet(f"{DATA}/{c}.parquet",
                                        columns=["float_mv"])["float_mv"],
                        errors="coerce") for c in codes}
MVa = pd.DataFrame(mvv).set_axis(idx).to_numpy(float)
del mvv


def quints(vals, valid):
    qq = np.full((NT, NS), -1, dtype=np.int8)
    for t in range(NT):
        ok = np.isfinite(vals[t]) & valid[t]
        if ok.sum() < 50:
            continue
        qq[t, ok] = np.searchsorted(np.nanquantile(vals[t][ok], [.2, .4, .6, .8]),
                                    vals[t][ok], side="right")
    return qq


VALID = np.isfinite(Fa) & (Fa > 0)
QMV = quints(MVa, VALID)
QRP = quints(RPS60, VALID)
del MVa


def fwd_peak(n):
    m = pd.DataFrame(Fa[::-1]).rolling(n, min_periods=1).max().to_numpy(float)[::-1]
    out = np.full((NT, NS), np.nan)
    out[:-1] = m[1:]
    out = (out / Fa - 1.0).astype(np.float32)
    out[NT - n:] = np.nan
    return out


PK = {n: fwd_peak(n) for n, _ in HOR}
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)

ym = idx.to_period("M")
mend = sorted(int(np.flatnonzero(ym == p)[-1]) for p in ym.unique())
rng = np.random.default_rng(SEED)


def control(sub, pk_mat, keys):
    """按 keys 指定的维度(市值,或市值+RPS60)同格随机抽,200 组。
    返回 (200 组命中率, 同格总体命中率, 跳过占比)。"""
    cnt = {}
    for t, j in sub:
        key = (t,) + tuple(int(kk[t, j]) for kk in keys)
        if min(key[1:]) >= 0:
            cnt[key] = cnt.get(key, 0) + 1
    hit, tot, th, tn, skip = np.zeros(NSEED), 0, 0.0, 0, 0
    for key, k in cnt.items():
        t = key[0]
        m = np.isfinite(pk_mat[t])
        for kk, q in zip(keys, key[1:], strict=True):
            m &= kk[t] == q
        pool = np.flatnonzero(m)
        if pool.size < MIN_POOL:
            skip += k
            continue
        v = pk_mat[t, pool] >= 1.0
        th += float(v.mean()) * k
        tn += k
        hit += v[rng.integers(0, pool.size, size=(NSEED, k))].sum(axis=1)
        tot += k
    if tot == 0:
        return np.array([]), np.nan, 1.0
    return hit / tot, th / tn, skip / (skip + tot)


rows = []
SUB = {}
for n, hname in HOR:
    pm = PK[n]
    sub = [(t, j) for t in mend for j in np.flatnonzero(NEWHI[t] & np.isfinite(pm[t]))]
    SUB[n] = sub
    v = np.array([pm[t, j] for t, j in sub]) >= 1.0
    obs = float(v.mean())
    r = dict(前瞻=hname, 事件数=len(sub), ge100=obs)
    for nm, keys in (("A", (QMV,)), ("C", (QMV, QRP))):
        c, th, sk = control(sub, pm, keys)
        med = float(np.median(c)) if c.size else np.nan
        r |= {f"对照{nm}": med, f"lift{nm}": obs / med if med > 0 else np.nan,
              f"p{nm}": float((c >= obs).mean()) if c.size else np.nan,
              f"零校验{nm}": abs(med - th), f"跳过{nm}": sk}
    rows.append(r)
    print(f"\n{'='*104}\n{hname}峰值 ≥100%{'  【判据口径】' if n == 120 else '  (仅描述)'}"
          f"\n{'='*104}")
    print(f"  事件数 {len(sub):,}   新高组 ≥100% = {obs:.2%}")
    for nm, lbl in (("A", "对照A 同市值随机          "),
                    ("C", "对照C 同市值+同RPS60分位  ")):
        print(f"  {lbl} {r['对照'+nm]:.2%}   lift {r['lift'+nm]:.2f}   "
              f"p {r['p'+nm]:.4f}   零校验 {r['零校验'+nm]:.2%}   "
              f"格子跳过 {r['跳过'+nm]:.1%}")
R = pd.DataFrame(rows)
M6 = R[R["前瞻"] == "6个月"].iloc[0]

print(f"\n{'='*104}\n逐年稳定性(对照A,6 个月口径)\n{'='*104}")
yr_rows = []
pm = PK[120]
for y in sorted({idx[t].year for t, _ in SUB[120]}):
    ev = [(t, j) for t, j in SUB[120] if idx[t].year == y]
    if len(ev) < MIN_N_YEAR:
        print(f"  {y}  n={len(ev):>5,}  < {MIN_N_YEAR},不计入判据③")
        continue
    v = np.array([pm[t, j] for t, j in ev]) >= 1.0
    o = float(v.mean())
    c, _, _ = control(ev, pm, (QMV,))
    md = float(np.median(c)) if c.size else np.nan
    lf = o / md if md > 0 else np.nan
    yr_rows.append(dict(年=y, n=len(ev), ge100=o, 对照A=md, lift=lf))
    print(f"  {y}  n={len(ev):>5,}  ≥100% {o:>6.2%}  对照A {md:>6.2%}  "
          f"lift {lf:>5.2f}  {'✓' if lf > 1.0 else '✗'}")
Y = pd.DataFrame(yr_rows)
frac = float((Y["lift"] > 1.0).mean()) if len(Y) else np.nan

print(f"\n{'='*104}\n锚点核对(不过则全节作废)\n{'='*104}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② RPS60 恒等校验")
print(f"  {'✓' if a3 else '✗'} 锚点③ 无前视校验")
if not a2:
    bad.append("锚点②")
if not a3:
    bad.append("锚点③")
zs = [R[f"零校验{n}"].max() for n in ("A", "C")]
a4 = bool(np.isfinite(zs).all() and max(zs) <= 0.03)
print(f"  {'✓' if a4 else '✗'} 锚点④ 恒等零校验 最大差 "
      f"{max(zs):.2%} ≤ 3pp" if np.isfinite(zs).all() else "  ✗ 锚点④ 算不出 = 不通过")
if not a4:
    bad.append("锚点④")

print(f"\n{'='*104}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*104}")
print(f"  前置条件:事件数 {M6['事件数']:,} ≥ {MIN_N};逐年合格 {len(Y)} 年")
c1 = bool(M6["liftA"] >= LIFT_MIN and M6["pA"] < ALPHA)
c2 = bool(M6["liftC"] >= LIFT_MIN and M6["pC"] < ALPHA)
c3 = bool(np.isfinite(frac) and frac >= YEAR_FRAC)
print(f"  {'✓' if c1 else '✗'} 判据① 对照A  lift {M6['liftA']:.2f} ≥ {LIFT_MIN} "
      f"且 p {M6['pA']:.4f} < {ALPHA}")
print(f"  {'✓' if c2 else '✗'} 判据② 对照C  lift {M6['liftC']:.2f} ≥ {LIFT_MIN} "
      f"且 p {M6['pC']:.4f} < {ALPHA}")
print(f"  {'✓' if c3 else '✗'} 判据③ 逐年 lift>1.0 占比 {frac:.1%} ≥ {YEAR_FRAC:.0%}")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:「创250日新高」携带超出单纯涨幅的信息,是独立于动量的右尾因子。**")
    print("  **事前预测被证伪 —— 我错了。**")
elif c1 and c3:
    print("  **结论:新高溢价是真的且逐年稳定,但按 RPS60 分位配平后消失 ——")
    print("     它只是动量的另一种写法,不是独立因子。事前预测命中。**")
else:
    print("  **结论:新高溢价在右尾口径下不成立。**")

R.to_csv(f"{OUT}/newhigh_premium.csv", index=False)
Y.to_csv(f"{OUT}/newhigh_premium_yearly.csv", index=False)
print(f"\n→ {OUT}/newhigh_premium.csv + _yearly.csv   ({time.time()-t0:.0f}s)")
