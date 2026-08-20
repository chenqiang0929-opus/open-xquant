"""第九十二节:换用户原话的口径重测第一段 —— 窗口内创过新高,不是最后一天(事前登记)

═══ 起因:§90 测的不是用户问的那个问题 ═══
§90 三个锚点全过、两条判据都不过,**但那一节测的口径是我挑错的时点**(第 7 次失误)。

§90 用 `score_one` 返回的 `_ts`(最近一次 RPS60>90 的交易日)当天是否创 250 日新高
来劈分,结果「有新高」只占 **2.2%**(我事前预测 30%~60%,错得离谱)。

**诊断(本节 A 部分给出实测证据,不靠推理):**
`RPS60>90` 是「**过去 60 日**涨幅排全市场前 10%」。股票见顶回落之后,
那个 60 日窗口里仍然装着前面那波大涨,RPS60 会继续 >90 好几周 ——
**所以 `cand[-1]`(最后一个强势日)系统性地落在第一段的顶之后。**
本节 A 部分实测 `ts` 距最近一次新高有多少个交易日、`ts` 当天距 250 日高多少个百分点,
**把这条诊断做成数字,不是我的说辞。**

**用户的原话是:**
> 「宇通第一段**上涨**,肯定伴随着 RPS 大于 90+,**且突破 250 新高**」

指的是**这一段行情期间**创过新高,不是「最后一个强势日当天」。本节按这个口径重测。

═══ 口径(事前锁定) ═══
  管线    与 §89/§90 逐字相同(直接调 score_one,不自拼);锚点② 恒等复现四格事件数
  **主口径劈分**  **窗口 [ts−60, ts] 内任一天 NEWHI** ——
          即 RPS60 自己度量的那 60 个交易日,与信号定义同源,**没有自由参数**
  副口径  [强势日连续段起点 − 60, ts] 内任一天 NEWHI,**仅描述,不参与判据**
  前瞻    6 个月(120 日)峰值 ≥100% = 判据口径;12 个月仅描述
  对照    同日同市值五分位随机 × 200 组(两半各自对各自的对照)
  检验    月内标签置换 × 2000(超几何精确抽样),标定见 §90 事前登记

═══ 锚点(不过则全节作废;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **四格事件数恒等复现 §89**:legacy 状态 19,704 / 突破 12,161,
     adaptive 状态 10,861 / 突破 6,676(同一管线,必须一个不差 —— §90 已验证可达)
  ③ **恒等零校验**:各半格对照的中位命中率 vs 同格总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽;与 §90 完全同形,可直接对比)═══
  **前置条件**:某半格事件数 **< 300** 则该格不参与判据(不判,而非判负)
  ① 四格中至少一格,「窗口内有新高」半边的 6 个月 ≥100% 率
     **减去「无新高」半边 ≥ 1.5pp**,且月内置换 **p < 0.05/4 = 0.0125**
  ② 同一格,两半各自对同日同市值对照的 **lift 之差 ≥ 0.15**

**①② 都过,才算用户的「且突破250新高」这个条件加得动。**

═══ 判据自查(§79 正问 + §83 反问) ═══
**正问:什么会让它通过而不回答问题?**
→ 原始率差被月份/市值构成撑起来(**§90 实测 +4.59pp 折算后只剩 lift 差 +0.12**)
  → **堵法:判据② 要求 lift 之差**。
→ 4 格搜索出假阳性 → **堵法:Bonferroni 0.05/4**。
→ 观测不独立 → **堵法:月内标签置换**。

**反问:什么会让它不通过而与问题无关?**
→ **窗口又选错**(§90 的病根,第 7 次)→ **堵法:窗口 = RPS60 自己度量的 60 日,
  与信号定义同源、没有自由参数;并同时报副口径,看结论稳不稳**。
→ 某半格样本太小(§90 四格里三格没够着前置条件)→ **堵法:前置 n≥300;
  且 A 部分先打印占比,占比过低就等于口径又选窄了,必须在正文里说**。
→ 锚点误杀正确实现 → **堵法:三个锚点全是恒等式,② 已在 §90 实证可达**。

═══ 事前预测(写下以便被证伪) ═══
**描述量:「窗口内有新高」占比 40%~75%**(§90 的 2.2% 是时点口径的机械后果;
换成窗口口径应当大幅上升。**若仍 <10%,说明我的诊断也错了,要在正文里说**)。
**判据① 不通过、② 不通过。**
理由:§89 已证突破格的全部优势对隔离动量的对照归零;§62 层一实测
「启动前距 250 日高」启动股 −25.8% vs 对照股 −27.3%,**lift 0.95**;
§90 换个劈分点也只有 lift 差 +0.12/+0.03。
**若①② 都过,说明第一段确实需要「期间创过新高」这个条件,我错了。**
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
MIN_N, NCELL, BRK_CAP, WIN = 300, 4, 250, 60
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
GAP250 = (Fa / HI250 - 1.0).astype(np.float32)      # ts 当天距 250 日高多少
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

ST = {"legacy": [], "adaptive": []}       # (t, j, ts, run_start)
SEG = {"legacy": {}, "adaptive": {}}
prev = {"legacy": set(), "adaptive": set()}


def run_start(sd, ts):
    """强势日连续段的起点:从 ts 往回,允许 5 日以内的空档算同一段。"""
    a = sd[sd <= ts]
    if a.size == 0:
        return ts
    b = int(a[-1])
    for x in a[::-1][1:]:
        if b - int(x) > 5:
            break
        b = int(x)
    return b


for mi, p in enumerate(months):
    t = last_td[p]
    sc_l, sc_a, sdc = {}, {}, {}
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
        if s_l is not None or s_a is not None:
            sdc[j] = sd
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
            ts = int(s["_ts"])
            rs = run_start(sdc[j], ts)
            ST[r].append((t, j, ts, rs))
            if np.isfinite(s["距区间高"]) and s["现价"] > 0:
                pk = s["现价"] / (1 + s["距区间高"])
                if np.isfinite(pk) and pk > 0:
                    SEG[r].setdefault(j, []).append((t, pk, ts, rs))
        prev[r] = set(hits[r])
    if (mi + 1) % 40 == 0:
        print(f"  {p}  状态 L{len(ST['legacy']):,}/A{len(ST['adaptive']):,}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

BK = {"legacy": [], "adaptive": []}
for r in ("legacy", "adaptive"):
    for j, segs in SEG[r].items():
        col, cur = Fa[:, j], -1
        for t, pk, ts, rs in sorted(segs):
            if t <= cur:
                continue
            w = np.flatnonzero(col[t + 1:min(t + 1 + BRK_CAP, NT)] > pk)
            if w.size:
                cur = t + 1 + int(w[0])
                BK[r].append((cur, j, ts, rs))
CELL = {(r, k): v for r in ("legacy", "adaptive")
        for k, v in (("状态", ST[r]), ("突破", BK[r]))}
print("\n事件数(锚点② 恒等复现 §89):", flush=True)
a2 = True
for key, want in EXP_EV.items():
    got = len(CELL[key])
    ok = got == want
    a2 &= ok
    print(f"  {'✓' if ok else '✗'} {key[0]}|{key[1]:<4} {got:>7,}  (§89 {want:,})")

# ── A 部分:诊断 —— ts 到底落在第一段的什么位置(给 §90 的 2.2% 一个实测解释)──
allev = [e for v in CELL.values() for e in v]
d_lag, d_gap = [], []
for _, j, ts, _ in allev:
    nh = np.flatnonzero(NEWHI[:ts + 1, j])
    d_lag.append(ts - int(nh[-1]) if nh.size else np.nan)
    d_gap.append(float(GAP250[ts, j]))
d_lag = np.array(d_lag, float)
d_gap = np.array(d_gap, float)
d_lag = d_lag[np.isfinite(d_lag)]
d_gap = d_gap[np.isfinite(d_gap)]
print(f"\n{'='*110}\nA 诊断:最后一个强势日 ts 落在第一段的什么位置(n={len(allev):,})\n{'='*110}")
print(f"  ts 距最近一次 250 日新高(交易日)  中位 **{np.median(d_lag):.0f}**   "
      f"四分位 [{np.percentile(d_lag,25):.0f}, {np.percentile(d_lag,75):.0f}]   "
      f"ts 当天就是新高的占比 **{(d_lag == 0).mean():.1%}**")
print(f"  ts 当天距 250 日高                 中位 **{np.median(d_gap):.1%}**   "
      f"四分位 [{np.percentile(d_gap,25):.1%}, {np.percentile(d_gap,75):.1%}]")

rng = np.random.default_rng(SEED)


def control(sub, pk_mat):
    cnt = {}
    for e in sub:
        t, j = e[0], e[1]
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
    """月内标签置换检验(与 §90 同一实现,标定见 §90 事前登记)。"""
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


def has_hi(j, lo, hi):
    lo = max(int(lo), 0)
    return bool(NEWHI[lo:int(hi) + 1, j].any())


rows = []
for n, hname in HOR:
    pm = PK[n]
    print(f"\n{'='*118}\n{hname}峰值 ≥100%{'  【判据口径】' if n == 120 else '  (仅描述)'}"
          f"   主口径 = 窗口 [ts−{WIN}, ts] 内任一天创 250 日新高\n{'='*118}")
    print(f"{'格':<16}{'半边':<8}{'事件数':>8}{'占比':>7}{'≥100%':>9}{'对照':>9}"
          f"{'lift':>7}{'率差':>8}{'lift差':>8}{'p(置换)':>9}{'零校验':>8}{'副口径占比':>10}")
    for r in ("legacy", "adaptive"):
        for kind in ("状态", "突破"):
            sub = [e for e in CELL[(r, kind)] if np.isfinite(pm[e[0], e[1]])]
            if not sub:
                continue
            half = {"有新高": [], "无新高": []}
            n_alt = 0
            for e in sub:
                t, j, ts, rs = e
                half["有新高" if has_hi(j, ts - WIN, ts) else "无新高"].append(e)
                n_alt += has_hi(j, rs - WIN, ts)
            st = {}
            for hn, ev in half.items():
                if not ev:
                    continue
                v = np.array([pm[e[0], e[1]] for e in ev]) >= 1.0
                ca, th = control(ev, pm)
                ra = float(np.median(ca)) if ca.size else np.nan
                mp = {}
                for e, x in zip(ev, v, strict=True):
                    mp.setdefault(ym[e[0]], []).append(int(x))
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
                sa = f"{n_alt/tot:>10.1%}" if hn == "有新高" else f"{'—':>10}"
                print(f"{r+'|'+kind:<16}{hn:<8}{d['n']:>8,}{d['n']/tot:>7.1%}"
                      f"{d['obs']:>9.2%}{d['ctl']:>9.2%}{d['lift']:>7.2f}"
                      f"{sh:>8}{sl:>8}{sp:>9}{d['gap0']:>8.2%}{sa}")
                rows.append(dict(前瞻=hname, 格=f"{r}|{kind}", 半边=hn, 事件数=d["n"],
                                 占比=d["n"] / tot, ge100=d["obs"], 对照=d["ctl"],
                                 lift=d["lift"], 率差=g if hn == "有新高" else np.nan,
                                 lift差=lg if hn == "有新高" else np.nan,
                                 p置换=pv if hn == "有新高" else np.nan,
                                 零校验差=d["gap0"],
                                 副口径占比=n_alt / tot if hn == "有新高" else np.nan))
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
print(f"  前置条件:两半都 ≥{MIN_N} 的 {len(elig)}/{len(H)} 格(§90 只有 1/4)")
w1 = elig[(elig["率差"] >= GAP_MIN) & (elig["p置换"] < ALPHA)]
w2 = elig[elig["lift差"] >= LIFT_GAP_MIN]
c1, c2 = len(w1) > 0, len(w2) > 0
print(f"  {'✓' if c1 else '✗'} 判据① 率差≥{GAP_MIN:.1%} 且置换 p<{ALPHA}   {len(w1)} 格")
print(f"  {'✓' if c2 else '✗'} 判据② lift 差≥{LIFT_GAP_MIN}   {len(w2)} 格")
sh = float(H["事件数"].sum() / (H["事件数"].sum() + H["n_lo"].sum()))
print(f"\n  描述量:「窗口内有新高」占比 **{sh:.1%}**(§90 时点口径 2.2%;"
      f"事前预测 40%~75%,若仍 <10% 则我的诊断也错)")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif not len(elig):
    print("  **两半都够 300 的格子一个都没有,判据不判(不是判负)。**")
elif c1 and c2:
    print("  **结论:第一段确实需要「期间创过新高」。事前预测被证伪 —— 我错了。**")
elif c1 or c2:
    print("  **结论:两条只过一条,不足以认定(判据要求 ①② 都过)。**")
else:
    print("  **结论:换成用户原话的窗口口径,第一段加不加「突破250新高」仍然没有区别。**")

R.to_csv(f"{OUT}/first_leg_window.csv", index=False)
print(f"\n→ {OUT}/first_leg_window.csv   ({time.time()-t0:.0f}s)")
