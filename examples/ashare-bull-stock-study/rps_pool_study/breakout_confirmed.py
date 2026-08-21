"""第八十九节:第三段——突破按日判定,状态变事件(事前登记)

═══ 起因:§88 其实没测到用户的第三段 ═══
用户的假设:
> **「第三段必然需要突破箱体 —— 如果一直不突破,那就是横盘或下跌趋势。」**

§88 想测它,但突破格事件数只有 166/83/15/14,全部低于 n≥300 前置条件,**没判成**。
**根因是我的取样频率错了**:突破是**瞬时事件**,而 §88 按**月末快照**判定
「上月亮、本月末已站上区间高」—— 月中突破然后继续走的,到月末往往已不满足前提,
被大量漏掉。

§88 还犯了第二个错:用 `score_one(legacy=False)` **自己拼** legacy 尺子,
漏掉了藏在函数内部、由 `legacy` 参数控制的「20周线向上」条件。

**本节把这两件事都修掉。**

═══ 修法 ═══
① **状态用月度检测**(整理是慢变量),**突破用日频判定**(瞬时事件):
   月末三条全中 → 记下**区间高** → **逐日**往后找第一次 `收盘 > 区间高` → 那才是突破日。
   一个整理段只产生**一个**突破事件(去重),**同一只股票允许多个整理段**。
② **直接调 `score_one(..., legacy=True)`**,不再自己拼等价实现。

═══ 锚点已先行验证(本节新增的纪律) ═══
**§85/§87/§88 三次作废都栽在锚点,而 §81 是唯一没作废的 ——
区别在于 §81 把锚点先在已知答案上验证过才锁定。本节照 §81 做。**

预先验证结果(legacy=True,宇通 600066):

    首次三条全中月末  **2023-11**   区间高 **11.87**
    按日判定突破日    **2024-01-08**  收盘 12.14(距月末 26 个交易日)
    突破后 6 个月     峰值 **+101.1%**  期末 +78.8%
    突破后 12 个月    峰值 +108.6%     期末 +103.8%

**注意:从突破日入场,6 个月峰值确实到了 +101.1%;
而 §86 从月末 2024-01-31 起算只有 +94.8%。差别就是这 26 个交易日的入场点。**

═══ 事前登记之前改掉的三处(§7 要求主动说,不许悄悄改)═══
写完初稿后自查,在**提交事前登记之前**改了三处。**结果一个数都还没跑。**
**一、对照根本没做市值中性化。** 初稿算了五分位边界 `q` 却**从没用过**,
    直接 `rng.choice(base)` 从全市场抽 —— 违反本项目第 4 条纪律。现改为
    **同日同五分位内**抽样(对照B 再叠加「当日也创 250 日新高」)。
**二、每只股票一辈子只算一个突破。** 初稿 `if j not in pending[r]` 让 2015 年的
    箱体永久占住坑位,后面 13 年的整理段全被吞掉 —— 既压死样本量,
    **也会让锚点② 直接不过**(宇通若在 2015 年先亮过灯,2024-01-08 根本不会被记录)。
    现改为按整理段逐段推进,同一只股票可以有多段。
**三、锚点③ 原设计会误杀。** 原文是「对照A 的 ≥100% 概率 vs **全市场**基础概率 ±3pp」,
    可对照A 一旦正确地做了市值中性化,本来就该偏离全市场均值(事件偏小盘)——
    **正确的实现反而不过**,正是 §85/§87/§88 的病根。现改为**恒等零校验**:
    对照A 的中位命中率 vs **同一批 (日期,五分位) 格子的总体命中率**(事件数加权)。
    抽样正确则必然相等(200 组差异只剩抽样噪声),抽错则必然偏离。**四格逐格核对。**

═══ 口径(事前锁定) ═══
  尺子    legacy(`score_one(legacy=True)`,函数内部含「20周线向上」+ 写死 15 日下限)
          adaptive(`score_one(legacy=False)` + 当期 40% 分位 + 自适应下限)
  状态格  月末三条全中且上月未亮(=§87/§88 的 nobrk 口径)
  突破格  该状态之后**逐日**首次 收盘 > 区间高;**入场在突破日收盘**
          **上限 250 个交易日**:一年内没突破,该整理段作废,不产生事件
          (否则 2015 年的箱体在 2023 年被「突破」,那不是用户说的第三段)
  前瞻    6 个月(120 日)峰值 = **判据口径**;12 个月(250 日)仅作描述
  对照A   同日**同市值五分位**随机(与事件同格同数量,200 组)
  对照B   同日同五分位 **且当日也创 250 日新高** 的随机股 —— **隔离动量后的正确基准**

**为什么必须有对照B**:突破本身就是「已经涨了」,对照A 的 lift 含动量成分。
§88 实测对照B 一致高于对照A(7.07%~14.29% vs 5.17%~7.23%),**这个偏差是真的**。

═══ 锚点(不过则全节作废) ═══
  ① 面板 (3297, 5232)
  ② **宇通在 legacy 突破事件里,突破日 = 2024-01-08**(已预先验证可达)
  ③ **恒等零校验**:四格的对照A 中位命中率与「同格总体命中率」之差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽) ═══
  **前置条件**:某格事件数 **< 300** 则该格不参与判据(不判,而非判负)
  ① 两个突破格中至少一格,「6 个月 峰值≥100%」对**对照A** 的
     **lift ≥ 1.3 且 Bonferroni 校正后 p < 0.05/4 = 0.0125**
  ② 同一格对**对照B**(隔离动量)也须 **lift ≥ 1.3 且 p < 0.0125**

**①② 都过才算「突破确认有独立价值」;只过① 说明那只是动量。**

═══ 判据自查(§79 正问 + §83 反问) ═══
**正问:什么会让它通过而不回答问题?**
→ 突破=已涨,对照A 的 lift 含动量 → **堵法:判据② 用对照B**。
→ 4 格搜索出假阳性 → **堵法:Bonferroni 0.05/4**。
→ 事件偏小盘而对照没中性化 → **堵法:同五分位抽样 + 锚点③ 恒等零校验**。

**反问:什么会让它不通过而与问题无关?**
→ 样本量不足 → **堵法:前置条件 n≥300,不足则不判**。
→ 锚点设计不当(§85/§87/§88 的病根)→ **堵法:锚点② 已预先验证可达;
   锚点③ 改成恒等式,正确实现必过**。
→ 尺子拼错(§88 的病根)→ **堵法:直接调 score_one(legacy=True)**。

═══ 事前预测(写下以便被证伪) ═══
**① 可能通过**(突破格对对照A 会明显更好,因为含动量);
**② 不通过** —— 对隔离动量后的对照B,突破确认没有独立价值。
理由:§88 实测四个突破格对对照B 无一更好(9.04 vs 7.83、4.82 vs 8.43、
13.33 vs 13.33、7.14 vs 14.29);§62「所有提高胜率的过滤器都在削右尾」。
**若②通过,说明「箱体突破」相对「一般新高」确有增量,我错了 ——
那会是本项目第一个在右尾口径下站住的形态结论。**
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
    THR_ATR,
    THR_DEPTH,
    THR_SHRINK,
    load_panel,
    score_one,
    series_of,
)

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NQ, NSEED, SEED = 5, 200, 20260814
MIN_N, NCELL, BRK_CAP = 300, 4, 250      # 前置样本量 / 格数 / 突破搜索上限(交易日)
ALPHA = 0.05 / NCELL
HOR = [(120, "6个月"), (250, "12个月")]

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:                # ETF 不属于本项目股票池(§87 栽过)
    k = list(CL.columns).index("510300")
    STRONG = np.delete(STRONG, k, axis=1)
    CL = CL.drop(columns=["510300"])
    MA100 = MA100.drop(columns=["510300"])
    frames.pop("510300", None)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

codes = list(CL.columns)
SER = [series_of(frames, idx, c) for c in codes]
MAv = [MA100[c].to_numpy(float) for c in codes]
del frames
Fa = CL.where(CL > 0).ffill().to_numpy(float)   # 退市股 ffill 参与,绝不剔除

# ── 当日是否创 250 日新高(对照B 用)──
HI250 = pd.DataFrame(Fa).rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(HI250) & (Fa >= HI250 * 0.9999)
del HI250

# ── 市值五分位矩阵(对照A/B 的中性化维度)──
mvv = {c: pd.to_numeric(pd.read_parquet(f"{DATA}/{c}.parquet",
                                        columns=["float_mv"])["float_mv"],
                        errors="coerce") for c in codes}
MVa = pd.DataFrame(mvv).set_axis(idx).to_numpy(float)
del mvv
QUINT = np.full((NT, NS), -1, dtype=np.int8)
for t in range(NT):
    ok = np.isfinite(MVa[t]) & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if ok.sum() < 50:
        continue
    QUINT[t, ok] = np.searchsorted(np.nanquantile(MVa[t][ok], [.2, .4, .6, .8]),
                                   MVa[t][ok], side="right")
del MVa


def fwd_peak(n):
    """PK[t, j] = 未来 n 日最高价 / 今日收盘 - 1;不足 n 日前瞻的置 NaN。"""
    m = pd.DataFrame(Fa[::-1]).rolling(n, min_periods=1).max().to_numpy(float)[::-1]
    out = np.full((NT, NS), np.nan)
    out[:-1] = m[1:]                        # max(Fa[t+1 .. t+n])
    out = (out / Fa - 1.0).astype(np.float32)
    out[NT - n:] = np.nan
    return out


PK = {n: fwd_peak(n) for n, _ in HOR}
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = sorted(last_td)

ST = {"legacy": [], "adaptive": []}       # 状态事件 (t, j)
SEG = {"legacy": {}, "adaptive": {}}      # j -> [(t, 区间高), ...]
prev = {"legacy": set(), "adaptive": set()}

for mi, p in enumerate(months):
    t = last_td[p]
    sc_l, sc_a = {}, {}
    for j in range(NS):
        h, lo_, c_, v_ = SER[j]
        if not np.isfinite(c_[t]):
            continue
        sd = np.flatnonzero(STRONG[:t + 1, j])
        if sd.size == 0:
            continue
        s_l = score_one(h, lo_, c_, v_, MAv[j], sd, t, legacy=True)   # 直接调,不自拼
        if s_l is not None:
            sc_l[j] = s_l
        s_a = score_one(h, lo_, c_, v_, MAv[j], sd, t, legacy=False)
        if s_a is not None:
            sc_a[j] = s_a
    if len(sc_a) < 50:
        continue
    adj = np.array([s["调整天数"] for s in sc_a.values()])
    floor = max(MIN_ADJ_FLOOR, int(round(MIN_ADJ_RATIO * np.median(adj))))
    thr = {k: float(np.nanquantile([s[k] for s in sc_a.values()], Q_KEEP))
           for k in ("缩量比", "收敛比", "深度")}
    hits = {
        # legacy 的 15 日下限由 score_one 内部执行,这里只加三条阈值
        "legacy": {j: s for j, s in sc_l.items()
                   if s["缩量比"] < THR_SHRINK and s["收敛比"] < THR_ATR
                   and s["深度"] <= THR_DEPTH},
        "adaptive": {j: s for j, s in sc_a.items()
                     if s["调整天数"] >= floor and s["缩量比"] <= thr["缩量比"]
                     and s["收敛比"] <= thr["收敛比"] and s["深度"] <= thr["深度"]},
    }
    for r in ("legacy", "adaptive"):
        for j, s in hits[r].items():
            if j in prev[r]:                       # 上月已亮 = 同一段,不重复计事件
                continue
            ST[r].append((t, j))
            if np.isfinite(s["距区间高"]) and s["现价"] > 0:
                pk = s["现价"] / (1 + s["距区间高"])
                if np.isfinite(pk) and pk > 0:
                    SEG[r].setdefault(j, []).append((t, pk))
        prev[r] = set(hits[r])
    if (mi + 1) % 40 == 0:
        print(f"  {p}  状态 L{len(ST['legacy']):,}/A{len(ST['adaptive']):,}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

# ── 突破:逐日判定;一个整理段只取一次,同一只股票可有多段;超 250 日作废 ──
BK = {"legacy": [], "adaptive": []}
for r in ("legacy", "adaptive"):
    for j, segs in SEG[r].items():
        col, cur = Fa[:, j], -1
        for t, pk in sorted(segs):
            if t <= cur:                           # 还在上一段的突破之前 = 同一段
                continue
            w = np.flatnonzero(col[t + 1:min(t + 1 + BRK_CAP, NT)] > pk)
            if w.size:
                cur = t + 1 + int(w[0])
                BK[r].append((cur, j))
print(f"\n事件:状态 legacy {len(ST['legacy']):,} / adaptive {len(ST['adaptive']):,}"
      f"   **突破 legacy {len(BK['legacy']):,} / adaptive {len(BK['adaptive']):,}**"
      f"  ({time.time()-t0:.0f}s)", flush=True)

rng = np.random.default_rng(SEED)


def control(sub, pk_mat, newhi):
    """同日同五分位(newhi 再叠加当日创新高)抽同样多只,200 组。
    返回 (200 组命中率, 同格总体命中率) —— 后者是锚点③ 的恒等式目标。"""
    cnt = {}
    for t, j in sub:
        q = int(QUINT[t, j])
        if q >= 0:
            cnt[(t, q)] = cnt.get((t, q), 0) + 1
    hit, tot, th, tn = np.zeros(NSEED), 0, 0.0, 0
    for (t, q), k in cnt.items():
        pool = np.flatnonzero((QUINT[t] == q) & np.isfinite(pk_mat[t]))
        if newhi:
            pool = pool[NEWHI[t, pool]]
        if pool.size == 0:
            continue
        v = pk_mat[t, pool] >= 1.0
        th += float(v.mean()) * k
        tn += k
        hit += v[rng.integers(0, pool.size, size=(NSEED, k))].sum(axis=1)
        tot += k
    if tot == 0:
        return np.array([]), np.nan
    return hit / tot, th / tn


rows = []
for n, hname in HOR:
    pm = PK[n]
    tag = "  【判据口径】" if n == 120 else "  (仅描述)"
    print(f"\n{'='*116}\n{hname}峰值 ≥100%{tag}\n{'='*116}")
    print(f"{'格':<20}{'事件数':>8}{'≥100%':>9}{'对照A':>9}{'liftA':>7}{'pA':>8}"
          f"{'对照B':>9}{'liftB':>7}{'pB':>8}{'零校验':>9}")
    for r in ("legacy", "adaptive"):
        for kind, lst in (("状态", ST[r]), ("突破", BK[r])):
            sub = [(t, j) for t, j in lst if np.isfinite(pm[t, j])]
            if not sub:
                continue
            v = np.array([pm[t, j] for t, j in sub])
            obs = float((v >= 1.0).mean())
            ca, ra_th = control(sub, pm, False)
            cb, _ = control(sub, pm, True)
            ra = float(np.median(ca)) if ca.size else np.nan
            rb = float(np.median(cb)) if cb.size else np.nan
            pa_ = float((ca >= obs).mean()) if ca.size else np.nan
            pb_ = float((cb >= obs).mean()) if cb.size else np.nan
            la = obs / ra if ra and ra > 0 else np.nan
            lb = obs / rb if rb and rb > 0 else np.nan
            gap = abs(ra - ra_th)
            print(f"{r+'|'+kind:<20}{len(sub):>8,}{obs:>9.2%}{ra:>9.2%}{la:>7.2f}"
                  f"{pa_:>8.4f}{rb:>9.2%}{lb:>7.2f}{pb_:>8.4f}{gap:>9.2%}")
            rows.append(dict(前瞻=hname, 格=f"{r}|{kind}", 事件数=len(sub), ge100=obs,
                             对照A=ra, liftA=la, pA=pa_, 对照B=rb, liftB=lb, pB=pb_,
                             对照A总体=ra_th, 零校验差=gap))
R = pd.DataFrame(rows)
M = R[R["前瞻"] == "6个月"]

print(f"\n{'='*116}\n锚点核对(不过则全节作废)\n{'='*116}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
jy = codes.index("600066")
yb = [idx[t].strftime("%Y-%m-%d") for t, j in BK["legacy"] if j == jy]
a2 = "2024-01-08" in yb
print(f"  {'✓' if a2 else '✗'} 锚点② 宇通 legacy 突破日含 2024-01-08(已预先验证)"
      f"   实测 {yb if yb else '无'}")
if not a2:
    bad.append("锚点②")
if M["零校验差"].notna().all():                    # 算不出 = 不通过(§79/§85 栽过)
    a3 = bool((M["零校验差"] <= 0.03).all())
    print(f"  {'✓' if a3 else '✗'} 锚点③ 恒等零校验 四格最大差 "
          f"{M['零校验差'].max():.2%} ≤ 3pp")
else:
    a3 = False
    print("  ✗ 锚点③ 算不出 = 不通过")
if not a3:
    bad.append("锚点③")

print(f"\n{'='*116}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*116}")
brk = M[M["格"].str.endswith("突破")]
elig = brk[brk["事件数"] >= MIN_N]
print(f"  前置条件:突破格事件数 ≥{MIN_N} 的 {len(elig)}/{len(brk)}")
w1 = elig[(elig["liftA"] >= 1.3) & (elig["pA"] < ALPHA)]
w2 = elig[(elig["liftB"] >= 1.3) & (elig["pB"] < ALPHA)]
c1, c2 = len(w1) > 0, len(w2) > 0
print(f"  {'✓' if c1 else '✗'} 判据① 对照A:lift≥1.3 且 p<{ALPHA}   {len(w1)} 格")
print(f"  {'✓' if c2 else '✗'} 判据② 对照B(隔离动量):lift≥1.3 且 p<{ALPHA}   {len(w2)} 格")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif not len(elig):
    print("  **突破格样本量仍不足 300,判据不判(不是判负)。**")
elif c1 and c2:
    print("  **结论:突破确认有独立价值(隔离动量后仍成立)。事前预测被证伪 —— 我错了。**")
elif c1:
    print("  **结论:突破格优于同市值随机,但对「同日也创新高」的对照无优势 ——")
    print("     那只是动量,不是箱体突破的功劳。事前预测命中。**")
else:
    print("  **结论:突破确认在右尾口径下不成立。**")

R.to_csv(f"{OUT}/breakout_confirmed.csv", index=False)
print(f"\n→ {OUT}/breakout_confirmed.csv   ({time.time()-t0:.0f}s)")
