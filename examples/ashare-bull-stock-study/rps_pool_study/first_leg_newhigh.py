"""第九十节:第一段的「250日新高」条件到底加不加得动(事前登记)

═══ 起因:三段里唯一没测成的一段 ═══
用户的原话把第一段定义成**两个条件**:
> 「宇通第一段上涨,肯定伴随着 **RPS 大于 90+**,**且突破 250 新高**(是不是存在共性)」

但筛选器只实现了前一半:

    RPS60 = CL.pct_change(60).rank(axis=1, pct=True) * 100
    return CL, frames, (RPS60 > 90).to_numpy(), ...      # 没有「突破250新高」

§88 加过后一半(`STRONG_B = STRONG_A & NEWHI`),**但 §88 因锚点作废**;
§89 为了先把第三段测干净,把 A/B 这个维度整个砍掉了。
**所以「第一段必须同时满足 RPS>90 和 250日新高」至今没有一个未作废的结论。本节补上。**

═══ 与 §88 的关键差别:不重跑两套 STRONG,而是就地劈开 ═══
§88 用两套 STRONG 矩阵各跑一遍。**那样不可比** —— STRONG 变了,
`score_one` 里 `ts = cand[-1]`(最近一次强势日)就变,整理段的起点、
调整天数、深度、缩量比全部跟着漂移,两组根本不是同一批整理段。

**本节改成:只跑一套(A = RPS60>90),把每个事件按其锚定强势日 `ts`
当天是否也创 250 日新高,就地劈成两半。**
同一条管线、同一天、同一个整理段 —— **配对干净,差异只来自那一个条件。**

═══ 锚点已先行验证(§89 立的纪律)═══
本节的事件检测逻辑与 §89 逐字相同(只多带出 `s["_ts"]`),
所以中间计数必须**恒等复现**。预先验证:跑前 40 个月,得到

    2016-04  状态 L1,577/A1,279        与 §89 日志逐字一致 ✓

═══ 口径(事前锁定) ═══
  尺子    legacy / adaptive,与 §89 完全一致(直接调 score_one,不自拼)
  状态格  月末三条全中且上月未亮;突破格 该状态后逐日首次 收盘>区间高(上限250日)
  劈分    事件的锚定强势日 ts = score_one 返回的 `_ts`;
          **有新高 = NEWHI[ts, j]**(ts 当天收盘 ≥ 过去250日最高 × 0.9999)
  前瞻    6 个月(120 日)峰值 ≥100% = 判据口径;12 个月仅描述
  对照    同日同市值五分位随机 × 200 组(两半各自对各自的对照)
  检验    **月内标签置换 × 2000**(不是分块 bootstrap,见下)——
          250 日前瞻窗口按月重叠,观测不独立;置换保留月内相关性与月份构成

═══ 锚点(不过则全节作废) ═══
  ① 面板 (3297, 5232)
  ② **事件数恒等复现 §89**:legacy 状态 19,704 / 突破 12,161,
     adaptive 状态 10,861 / 突破 6,676(同一管线,必须一个不差)
  ③ **恒等零校验**:各半格对照的中位命中率 vs 同格总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽) ═══
  **前置条件**:某半格事件数 **< 300** 则该格不参与判据(不判,而非判负)
  ① 四格中至少一格,「有新高」半边的 6 个月 ≥100% 率
     **减去「无新高」半边 ≥ 1.5pp**,且月内置换检验 **p < 0.05/4 = 0.0125**
  ② 同一格,两半各自对**同日同市值对照**的 **lift 之差 ≥ 0.15**
     (堵住月份/市值构成混杂:万一「有新高」的事件恰好集中在右尾更肥的年份)

**①② 都过,才算用户的「且突破250新高」这个条件加得动。**

═══ 判据自查(§79 正问 + §83 反问) ═══
**正问:什么会让它通过而不回答问题?**
→ 「有新高」的事件可能集中在牛市年份/小市值,右尾天然更肥 →
  **堵法:判据② 要求 lift 之差,不只是原始率之差**。
→ 4 格搜索出假阳性 → **堵法:Bonferroni 0.05/4**。
→ 观测不独立导致 p 值虚低 → **堵法:月内标签置换,保留月内相关性**。
→ **检验本身在 H0 下就会显著**(下一条,已实测)→ **堵法:换掉 bootstrap**。

**反问:什么会让它不通过而与问题无关?**
→ 某半格样本太小 → **堵法:前置 n≥300,不足则不判**。
→ 两套 STRONG 让段边界漂移、两组不可比(§88 的病根)→ **堵法:就地劈开,不重跑**。
→ 锚点设成「复现精确数字」而实现细节一变就挂(§85/§87/§88 的病根)→
  **堵法:本节锚点② 是恒等式**(同一管线必然同一计数),且已在前 40 个月预验证。

═══ 事前登记之前改掉的一处(第 7 条纪律:主动说,不许悄悄改)═══
初稿的显著性检验写的是**月度分块 bootstrap**(重抽月份,看率差是否稳定为正)。
**在合成数据上一验就废了:两半完全同分布(真 H0)时它给出 p = 0.028。**
原因是它根本不是零假设检验 —— 重抽月份得到的是「观测差值的置信区间」,
观测差只要偏离 0 一点点、标准误又小,它就会报出小 p。**照这个写下去,
判据① 可能被噪声点亮 —— 正是 §85「挑错门槛」的同一种病。**

改成**月内标签置换检验**:H0 下「有新高」这个标签在同月内可交换,
月内把 k 个标签随机分配,落在命中事件上的个数服从超几何分布。
它保留月份构成(堵月份混杂)与月内事件相关性(重排标签、不动结果)。实测标定:

    H0 重复 40 次   p 中位 **0.477**(应≈0.5)   p<0.05 占 7.5%   p<0.0125 占 0.0%
    有新高高 1.5pp  p = 0.0000        有新高高 4pp   p = 0.0000
    有新高反而低    p = 1.0000
    超几何版与逐个重排版的置换分布一致(均值差 0.00002、标准差差 0.00002),快 33 倍

═══ 事前预测(写下以便被证伪) ═══
**① 不通过;② 不通过。**
理由:§62 层一实测「启动前距 250 日高」启动股 −25.8% vs 对照股 −27.3%,
**lift 0.95 —— 启动前的形态根本不是驱动因子**;真正把股票推上 RPS60>90 的
只有换手率(lift 1.27)和涨停。§89 又证明所有优势都来自**当下**的动量,
不是**历史**的形态。
**另外预测描述量:「有新高」的占比落在 30%~60% 之间** ——
RPS60>90 只要求 60 日涨幅排前 10%,并不要求创新高,所以远不是「肯定伴随」。
**若占比 >85%,说明用户的「肯定伴随」在描述层面成立,我这条预测错。**
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
NQ, NSEED, SEED, NPERM = 5, 200, 20260814, 2000
MIN_N, NCELL, BRK_CAP = 300, 4, 250
ALPHA = 0.05 / NCELL
GAP_MIN, LIFT_GAP_MIN = 0.015, 0.15
HOR = [(120, "6个月"), (250, "12个月")]
EXP_EV = {("legacy", "状态"): 19704, ("legacy", "突破"): 12161,
          ("adaptive", "状态"): 10861, ("adaptive", "突破"): 6676}

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
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
Fa = CL.where(CL > 0).ffill().to_numpy(float)
HI250 = pd.DataFrame(Fa).rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(HI250) & (Fa >= HI250 * 0.9999)
del HI250
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
    m = pd.DataFrame(Fa[::-1]).rolling(n, min_periods=1).max().to_numpy(float)[::-1]
    out = np.full((NT, NS), np.nan)
    out[:-1] = m[1:]
    out = (out / Fa - 1.0).astype(np.float32)
    out[NT - n:] = np.nan
    return out


PK = {n: fwd_peak(n) for n, _ in HOR}
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = sorted(last_td)

ST = {"legacy": [], "adaptive": []}       # (t, j, ts)
SEG = {"legacy": {}, "adaptive": {}}      # j -> [(t, 区间高, ts)]
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
        s_l = score_one(h, lo_, c_, v_, MAv[j], sd, t, legacy=True)
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
        "legacy": {j: s for j, s in sc_l.items()
                   if s["缩量比"] < THR_SHRINK and s["收敛比"] < THR_ATR
                   and s["深度"] <= THR_DEPTH},
        "adaptive": {j: s for j, s in sc_a.items()
                     if s["调整天数"] >= floor and s["缩量比"] <= thr["缩量比"]
                     and s["收敛比"] <= thr["收敛比"] and s["深度"] <= thr["深度"]},
    }
    for r in ("legacy", "adaptive"):
        for j, s in hits[r].items():
            if j in prev[r]:
                continue
            ST[r].append((t, j, int(s["_ts"])))
            if np.isfinite(s["距区间高"]) and s["现价"] > 0:
                pk = s["现价"] / (1 + s["距区间高"])
                if np.isfinite(pk) and pk > 0:
                    SEG[r].setdefault(j, []).append((t, pk, int(s["_ts"])))
        prev[r] = set(hits[r])
    if (mi + 1) % 40 == 0:
        print(f"  {p}  状态 L{len(ST['legacy']):,}/A{len(ST['adaptive']):,}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

BK = {"legacy": [], "adaptive": []}
for r in ("legacy", "adaptive"):
    for j, segs in SEG[r].items():
        col, cur = Fa[:, j], -1
        for t, pk, ts in sorted(segs):
            if t <= cur:
                continue
            w = np.flatnonzero(col[t + 1:min(t + 1 + BRK_CAP, NT)] > pk)
            if w.size:
                cur = t + 1 + int(w[0])
                BK[r].append((cur, j, ts))
CELL = {(r, k): v for r in ("legacy", "adaptive")
        for k, v in (("状态", ST[r]), ("突破", BK[r]))}
print("\n事件数(锚点② 恒等复现 §89):", flush=True)
a2 = True
for key, want in EXP_EV.items():
    got = len(CELL[key])
    ok = got == want
    a2 &= ok
    print(f"  {'✓' if ok else '✗'} {key[0]}|{key[1]:<4} {got:>7,}  (§89 {want:,})")

rng = np.random.default_rng(SEED)


def control(sub, pk_mat):
    cnt = {}
    for t, j, _ in sub:
        q = int(QUINT[t, j])
        if q >= 0:
            cnt[(t, q)] = cnt.get((t, q), 0) + 1
    hit, tot, th, tn = np.zeros(NSEED), 0, 0.0, 0
    for (t, q), k in cnt.items():
        pool = np.flatnonzero((QUINT[t] == q) & np.isfinite(pk_mat[t]))
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


def perm_p(mon_hi, mon_lo):
    """月内标签置换检验。H0:「有新高」这个标签在同一个月内可交换。
    保留月份构成(堵住月份混杂)与月内事件相关性(重排标签,不动结果)。
    月内落到命中事件上的标签数 ~ 超几何,精确抽样。返回 (观测率差, p)。"""
    ms = sorted(set(mon_hi) | set(mon_lo))
    pool, nh = [], []
    for m in ms:
        a, b = mon_hi.get(m, []), mon_lo.get(m, [])
        if len(a) + len(b) == 0:
            continue
        pool.append(np.asarray(list(a) + list(b), float))
        nh.append(len(a))
    n_hi = sum(nh)
    n_lo = sum(len(x) - k for x, k in zip(pool, nh, strict=True))
    if len(pool) < 12 or n_hi == 0 or n_lo == 0:
        return np.nan, np.nan
    obs = (sum(x[:k].sum() for x, k in zip(pool, nh, strict=True)) / n_hi
           - sum(x[k:].sum() for x, k in zip(pool, nh, strict=True)) / n_lo)
    hs, ls = np.zeros(NPERM), np.zeros(NPERM)
    for x, k in zip(pool, nh, strict=True):
        tot, n = int(x.sum()), len(x)
        if k == 0:
            ls += tot
            continue
        if k == n:
            hs += tot
            continue
        s = (rng.hypergeometric(tot, n - tot, k, size=NPERM) if 0 < tot < n
             else np.full(NPERM, tot * k / n, float))
        hs += s
        ls += tot - s
    d = hs / n_hi - ls / n_lo
    return float(obs), float((d >= obs).mean())


rows = []
for n, hname in HOR:
    pm = PK[n]
    print(f"\n{'='*118}\n{hname}峰值 ≥100%{'  【判据口径】' if n == 120 else '  (仅描述)'}"
          f"\n{'='*118}")
    print(f"{'格':<16}{'半边':<8}{'事件数':>8}{'占比':>7}{'≥100%':>9}{'对照':>9}"
          f"{'lift':>7}{'率差':>8}{'lift差':>8}{'p(置换)':>9}{'零校验':>8}")
    for r in ("legacy", "adaptive"):
        for kind in ("状态", "突破"):
            sub = [(t, j, ts) for t, j, ts in CELL[(r, kind)] if np.isfinite(pm[t, j])]
            if not sub:
                continue
            half = {"有新高": [e for e in sub if NEWHI[e[2], e[1]]],
                    "无新高": [e for e in sub if not NEWHI[e[2], e[1]]]}
            st = {}
            for hn, ev in half.items():
                if not ev:
                    continue
                v = np.array([pm[t, j] for t, j, _ in ev]) >= 1.0
                ca, th = control(ev, pm)
                ra = float(np.median(ca)) if ca.size else np.nan
                mp = {}
                for (t, _, _), x in zip(ev, v, strict=True):
                    mp.setdefault(ym[t], []).append(int(x))
                st[hn] = dict(n=len(ev), obs=float(v.mean()), ctl=ra,
                              lift=float(v.mean()) / ra if ra > 0 else np.nan,
                              gap0=abs(ra - th), mon=mp)
            if len(st) < 2:
                continue
            g, pv = perm_p(st["有新高"]["mon"], st["无新高"]["mon"])
            lg = st["有新高"]["lift"] - st["无新高"]["lift"]
            tot = st["有新高"]["n"] + st["无新高"]["n"]
            for hn in ("有新高", "无新高"):
                d = st[hn]
                sh = "—" if hn == "无新高" else f"{g:+.2%}"
                sl = "—" if hn == "无新高" else f"{lg:+.2f}"
                sp = "—" if hn == "无新高" else f"{pv:.4f}"
                print(f"{r+'|'+kind:<16}{hn:<8}{d['n']:>8,}{d['n']/tot:>7.1%}"
                      f"{d['obs']:>9.2%}{d['ctl']:>9.2%}{d['lift']:>7.2f}"
                      f"{sh:>8}{sl:>8}{sp:>9}{d['gap0']:>8.2%}")
                rows.append(dict(前瞻=hname, 格=f"{r}|{kind}", 半边=hn, 事件数=d["n"],
                                 占比=d["n"] / tot, ge100=d["obs"], 对照=d["ctl"],
                                 lift=d["lift"], 率差=g if hn == "有新高" else np.nan,
                                 lift差=lg if hn == "有新高" else np.nan,
                                 p置换=pv if hn == "有新高" else np.nan,
                                 零校验差=d["gap0"]))
R = pd.DataFrame(rows)
M = R[R["前瞻"] == "6个月"]

print(f"\n{'='*118}\n锚点核对(不过则全节作废)\n{'='*118}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② 四格事件数恒等复现 §89")
if not a2:
    bad.append("锚点②")
if M["零校验差"].notna().all():
    a3 = bool((M["零校验差"] <= 0.03).all())
    print(f"  {'✓' if a3 else '✗'} 锚点③ 恒等零校验 八半格最大差 "
          f"{M['零校验差'].max():.2%} ≤ 3pp")
else:
    a3 = False
    print("  ✗ 锚点③ 算不出 = 不通过")
if not a3:
    bad.append("锚点③")

print(f"\n{'='*118}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*118}")
H = M[M["半边"] == "有新高"].copy()
L = M[M["半边"] == "无新高"].set_index("格")["事件数"]
H["n_lo"] = H["格"].map(L)
elig = H[(H["事件数"] >= MIN_N) & (H["n_lo"] >= MIN_N)]
print(f"  前置条件:两半都 ≥{MIN_N} 的 {len(elig)}/{len(H)} 格")
w1 = elig[(elig["率差"] >= GAP_MIN) & (elig["p置换"] < ALPHA)]
w2 = elig[elig["lift差"] >= LIFT_GAP_MIN]
c1, c2 = len(w1) > 0, len(w2) > 0
print(f"  {'✓' if c1 else '✗'} 判据① 率差≥{GAP_MIN:.1%} 且 分块 p<{ALPHA}   {len(w1)} 格")
print(f"  {'✓' if c2 else '✗'} 判据② lift 差≥{LIFT_GAP_MIN}   {len(w2)} 格")
sh = float(H["事件数"].sum() / (H["事件数"].sum() + H["n_lo"].sum()))
print(f"\n  描述量:「有新高」占比 **{sh:.1%}**"
      f"(事前预测 30%~60%;>85% 则「肯定伴随」在描述层面成立)")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2:
    print("  **结论:第一段加上「突破250新高」确有增量。事前预测被证伪 —— 我错了。**")
elif c1 or c2:
    print("  **结论:两条只过一条,不足以认定该条件加得动(判据要求 ①② 都过)。**")
else:
    print("  **结论:第一段加不加「突破250新高」没有区别。事前预测命中。**")

R.to_csv(f"{OUT}/first_leg_newhigh.csv", index=False)
print(f"\n→ {OUT}/first_leg_newhigh.csv   ({time.time()-t0:.0f}s)")
