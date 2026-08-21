"""第一〇六节:RPS250 分档重跑 —— 锚点值改对,判据逐字不变(事前登记)

═══ 证据等级声明:本节不是盲测 ═══
**§105 B 部分因锚点④ 不过而作废,根因是我拿错了样本** ——
锚点值 (RPS250 中位 71.2、>90 占比 12.1%) 是在**全部 21,876 个事件**上测的,
而 B 部分只能用 **15,413 个可比事件**(t0+500 需在面板内),实测 **72.6 / 12.6%**。

**那次作废的运行已经把结果打印出来了,我看过。所以本节不是盲测:**

    <50 档 20.99% / 50–70 档 10.69% / 70–85 档 7.90% / >85 档 8.84%
    Spearman **−0.80**、档间 **−12.16pp**、四档 liftA 全在 1.03~1.10

**因此本节的判据结果证据等级低于其他各节** —— 与 §91(漏做事前登记)同等处理。
**判据 B①②③ 逐字不变、不因为已知方向而改写**(改了就是事后编故事);
只把锚点④ 的期望值换成 15,413 事件上的正确值。

**新增部分是真正没看过的**:加入**对照B(同日同市值 + 当日也创 250 日新高)**,
**仅作描述、不设判据** —— 用来回答「低 RPS 档那个 20.99% 是不是也只是时点效应」。

═══ 口径(与 §105 B 部分完全相同,只改锚点值)═══
  事件 首次创 250 日新高(此前 120 日未创),取 t0+500 在面板内的 **15,413** 个
  分档 按首次新高**当日 RPS250**:**<50 / 50–70 / 70–85 / >85**
  入场 t0 收盘;前瞻 24 个月(500 日)峰值 **≥200%**
  对照A 同日同市值五分位随机 × 200 组
  **对照B 同日同市值 且当日也创 250 日新高 × 200 组(新增,仅描述)**

═══ 锚点(不过则全节作废;四个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② §103 事件数恒等复现:首次新高事件 **21,876** 个
  ③ 四只案例恒等复现:宇通 **9** / 匠心 **1** / 嘉益 **2** / 泰格 **3**
  ④ **RPS250 分布(改对的值)**:15,413 个可比事件上,
     中位 **72.6**、>90 占比 **12.6%**(容差 ±0.5 / ±0.5pp)

═══ 事前判据(与 §105 B 逐字相同;Bonferroni **0.05/8 = 0.00625**)═══
  **前置**:某档事件数 **< 300** 不判
  B① **最高档(RPS250 >85)对对照A**:24 个月 ≥200% 的 **lift ≥ 1.3 且 p < 0.00625**
  B② **单调性**:四档命中率对档位序号的 **Spearman ≥ +0.8**
  B③ **档间**:最高档减最低档(<50)**≥ 5pp** 且月内置换 **p < 0.00625**

═══ 事前预测 ═══
**B①②③ 全不过 —— 但这不算预测,我已经看过数字了(见上)。**
**真正还没看过的是对照B。我预测:低 RPS 档(<50)的对照B 会明显高于其对照A,
即那个 20.99% 主要是「所在时点与市值档本来右尾就肥」,不是低 RPS 本身的功劳。**
理由:§89~§104 十余次一致显示,任何变量的绝对优势换成同日同类对照后归零;
而 §105 已看到四档 liftA 全在 1.03~1.10,说明相对同市值随机几乎没有差别。
**若对照B 反而低于对照A,说明低 RPS 档确有超出「时点+市值」的东西,我错了。**
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
GAP, MA, H24, H6 = 120, 100, 500, 120
MIN_N, ALPHA, TAIL_R, TREND_MAX = 1000, 0.05 / 8, 0.90, 0.20
MIN_NB, LIFT_MIN, GAP_MIN, RHO_MIN = 300, 1.3, 0.05, 0.80
RB = [(-1, 50, "<50"), (50, 70, "50–70"), (70, 85, "70–85"), (85, 101, ">85")]
NSEED, SEED, NPERM = 200, 20260814, 2000
EXP_EV, EXP4 = 21876, {"600066": 9, "301061": 1, "301004": 2, "300347": 3}
CUT = "2020-12-31"

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    CL = CL.drop(columns=["510300"])
del frames, STRONG, MA100
idx = CL.index
NT, NS = CL.shape
codes = list(CL.columns)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"
Fa = CL.where(CL > 0).ffill().to_numpy(float)


def newhi(px):
    hi = pd.DataFrame(px).rolling(250, min_periods=100).max().to_numpy(float)
    return np.isfinite(hi) & (px >= hi * 0.9999)


def ma(px, n):
    return pd.DataFrame(px).rolling(n, min_periods=n).mean().to_numpy(float)


NH, M50 = newhi(Fa), ma(Fa, MA)
RPS250 = (CL.pct_change(250).rank(axis=1, pct=True) * 100).to_numpy(float)


def events(nh, lim):
    out = []
    for j in range(nh.shape[1]):
        col = nh[:, j]
        for t in np.flatnonzero(col[:lim]):
            if t < 250 or col[max(t - GAP, 0):t].any():
                continue
            out.append((int(t), j))
    return out


EV = events(NH, NT)
cnt = {}
for t, j in EV:
    cnt[codes[j]] = cnt.get(codes[j], 0) + 1
a2 = len(EV) == EXP_EV
a3 = all(cnt.get(c, 0) == v for c, v in EXP4.items())
print(f"首次新高事件 {len(EV):,}  {'✓' if a2 else '✗'} 锚点②(期望 {EXP_EV:,})")
print(f"  {'✓' if a3 else '✗'} 锚点③ 四只:" + " ".join(
    f"{c}{cnt.get(c,0)}/{v}" for c, v in EXP4.items()))


def exit_day(t, j, lim=None):
    """规则A:收盘首次跌破 MA50 的那一天(含);未破返回 None。"""
    hi = NT if lim is None else lim
    col, m = Fa[t + 1:hi, j], M50[t + 1:hi, j]
    w = np.flatnonzero(np.isfinite(m) & (col < m))
    return t + 1 + int(w[0]) if w.size else None


kc = int(np.searchsorted(idx, pd.Timestamp(CUT, tz=idx.tz), side="right"))
M50c = ma(Fa[:kc], MA)
same = True
for t, j in EV[:200000]:
    if t >= kc - 600:
        continue
    col, m = Fa[t + 1:kc, j], M50c[t + 1:kc, j]
    w = np.flatnonzero(np.isfinite(m) & (col < m))
    e1 = t + 1 + int(w[0]) if w.size else None
    e2 = exit_day(t, j, kc)
    if e1 != e2:
        same = False
        break
a4 = same
print(f"  {'✓' if a4 else '✗'} 锚点④ 无前视:截断到 {CUT} 重算卖出日一致")
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)

rows = []
for t, j in EV:
    if t + H24 >= NT:
        continue
    p0 = Fa[t, j]
    if not np.isfinite(p0) or p0 <= 0:
        continue
    e = exit_day(t, j)
    held = e is None or e > t + H24
    ee = min(e, NT - 1) if e is not None else NT - 1
    r_a = Fa[ee, j] / p0 - 1
    pk_a = np.nanmax(Fa[t + 1:ee + 1, j]) / p0 - 1 if ee > t else np.nan
    r_b = Fa[t + H24, j] / p0 - 1
    pk_b = np.nanmax(Fa[t + 1:t + H24 + 1, j]) / p0 - 1
    r_c = Fa[t + H6, j] / p0 - 1
    # 跌破之后 250 日内是否再创 250 日新高
    again = np.nan
    if e is not None and e + 250 < NT:
        again = bool(NH[e + 1:e + 251, j].any())
    rows.append(dict(t=t, j=j, 年=idx[t].year, 未破=held, 持有日=(ee - t),
                     A实收=r_a, A峰值=pk_a, B实收=r_b, B峰值=pk_b, C实收=r_c, 再创新高=again))
D = pd.DataFrame(rows)
print(f"可比事件(t0+{H24} 在面板内){len(D):,}   "
      f"24 个月内未跌破 10 周线的占比 **{D['未破'].mean():.1%}**")

W = 96
print(f"\n{'='*W}\n规则对比(入场 = 首次新高日收盘)\n{'='*W}")
print(f"{'规则':<26}{'中位实收':>11}{'均值实收':>11}{'实收≥200%':>12}"
      f"{'实收>0':>10}{'中位持有日':>11}")
for nm, col, hold in (("A 10周线不破就持有(用户)", "A实收", D["持有日"].median()),
                      ("B 固定持有 500 日(24月)", "B实收", H24),
                      ("C 固定持有 120 日(6月)", "C实收", H6)):
    v = D[col].dropna()
    print(f"{nm:<26}{v.median():>11.1%}{v.mean():>11.1%}{(v>=2).mean():>12.2%}"
          f"{(v>0).mean():>10.1%}{hold:>11.0f}")
print(f"\n  峰值(未实现)口径:A 中位 {D['A峰值'].median():+.1%} / ≥200% "
      f"{(D['A峰值']>=2).mean():.2%}   B 中位 {D['B峰值'].median():+.1%} / ≥200% "
      f"{(D['B峰值']>=2).mean():.2%}")

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
for ok, nm in ((a2, "锚点② 事件数"), (a3, "锚点③ 四只案例"), (a4, "锚点④ 无前视")):
    print(f"  {'✓' if ok else '✗'} {nm}")
    if not ok:
        bad.append(nm)

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,Bonferroni {ALPHA:.5f})\n{'='*W}")
m_a, m_b = D["A实收"].median(), D["B实收"].median()
t_a, t_b = float((D["A实收"] >= 2).mean()), float((D["B实收"] >= 2).mean())
ag = D["再创新高"].dropna()
fr = float(ag.mean()) if len(ag) else np.nan
c1 = bool(m_a >= m_b)
c2 = bool(t_b > 0 and t_a >= TAIL_R * t_b)
c3 = bool(np.isfinite(fr) and fr < TREND_MAX)
print(f"  前置:可比事件 {len(D):,} ≥ {MIN_N}")
print(f"  {'✓' if c1 else '✗'} ① 实收不亏:A 中位 {m_a:+.2%} ≥ B 中位 {m_b:+.2%}")
print(f"  {'✓' if c2 else '✗'} ② 不削右尾:A 实收≥200% {t_a:.2%} ≥ "
      f"{TAIL_R:.0%}×B {t_b:.2%} = {TAIL_R*t_b:.2%}")
print(f"  {'✓' if c3 else '✗'} ③ 跌破=趋势结束:跌破后 250 日内再创 250 日新高的比例 "
      f"**{fr:.1%}** < {TREND_MAX:.0%}   (n={len(ag):,})")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:用户的规则成立 —— 10 周线既保住右尾,跌破也确实标志趋势结束。**")
    print("  **事前预测被证伪 —— 我错了。**")
elif c1 and not c2:
    print("  **结论:10 周线离场提高了实收中位,但削掉了右尾 —— §62 的又一次验证。**")
else:
    print("  **结论:用户的规则不成立,详见上表。**")

# ══════════════ B 部分:RPS250 分档 ══════════════
MV = pd.DataFrame({c: pd.to_numeric(
    pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])["float_mv"],
    errors="coerce") for c in codes})
if getattr(MV.index, "tz", None) is not None:
    MV.index = MV.index.tz_localize(None)
MV = MV.reindex(idx.tz_localize(None)).ffill().to_numpy(float)
QUINT = np.full((NT, NS), -1, dtype=np.int8)
for tt in range(NT):
    ok = np.isfinite(MV[tt]) & np.isfinite(Fa[tt]) & (Fa[tt] > 0)
    if ok.sum() < 50:
        continue
    QUINT[tt, ok] = np.searchsorted(np.nanquantile(MV[tt][ok], [.2, .4, .6, .8]),
                                    MV[tt][ok], side="right")
del MV
m = pd.DataFrame(Fa[::-1]).rolling(H24, min_periods=1).max().to_numpy(float)[::-1]
PKm = np.full((NT, NS), np.nan)
PKm[:-1] = m[1:]
PKm = (PKm / Fa - 1.0).astype(np.float32)
PKm[NT - H24:] = np.nan
rv = np.array([RPS250[t, j] for t, j in zip(D["t"], D["j"], strict=True)])
print(f"\n{'='*W}\nB 部分:按首次新高当日 RPS250 分档(24 个月峰值 ≥200%)\n{'='*W}")
print(f"  RPS250 分布:中位 **{np.nanmedian(rv):.1f}**  >90 占比 "
      f"**{np.nanmean(rv > 90):.1%}**  (期望 72.6 / 12.6%)")
a4b = abs(np.nanmedian(rv) - 72.6) <= 0.5 and abs(np.nanmean(rv > 90) - .126) <= .005
print(f"  {'✓' if a4b else '✗'} 锚点④ RPS250 分布恒等复现")
rng = np.random.default_rng(SEED)


def ctl_a(sub, newhi=False):
    cn = {}
    for t, j in sub:
        q = int(QUINT[t, j])
        if q >= 0:
            cn[(t, q)] = cn.get((t, q), 0) + 1
    hit, tot, th, tn = np.zeros(NSEED), 0, 0.0, 0
    for (t, q), k in cn.items():
        pool = np.flatnonzero((QUINT[t] == q) & np.isfinite(PKm[t]))
        if newhi:
            pool = pool[NH[t, pool]]
        if pool.size == 0:
            continue
        v = PKm[t, pool] >= 2.0
        th += float(v.mean()) * k
        tn += k
        hit += v[rng.integers(0, pool.size, size=(NSEED, k))].sum(axis=1)
        tot += k
    return (hit / tot, th / tn) if tot else (np.array([]), np.nan)


def permp(mh, ml):
    ms = sorted(set(mh) | set(ml))
    pool, nh_ = [], []
    for mm in ms:
        a, b = mh.get(mm, []), ml.get(mm, [])
        if len(a) + len(b) == 0:
            continue
        pool.append(np.asarray(list(a) + list(b), float))
        nh_.append(len(a))
    n1 = sum(nh_)
    n0 = sum(len(v) - k for v, k in zip(pool, nh_, strict=True))
    if len(pool) < 12 or n1 == 0 or n0 == 0:
        return np.nan, np.nan
    obs = (sum(v[:k].sum() for v, k in zip(pool, nh_, strict=True)) / n1
           - sum(v[k:].sum() for v, k in zip(pool, nh_, strict=True)) / n0)
    hs, ls = np.zeros(NPERM), np.zeros(NPERM)
    for v, k in zip(pool, nh_, strict=True):
        tt2, n = int(v.sum()), len(v)
        if k == 0:
            ls += tt2
            continue
        if k == n:
            hs += tt2
            continue
        sm = (rng.hypergeometric(tt2, n - tt2, k, size=NPERM) if 0 < tt2 < n
              else np.full(NPERM, tt2 * k / n, float))
        hs += sm
        ls += tt2 - sm
    dd = hs / n1 - ls / n0
    return float(obs), float((dd >= obs).mean())


ymm = idx.to_period("M")
print(f"\n{'档':<8}{'事件数':>9}{'RPS中位':>9}{'≥200%':>9}{'对照A':>9}{'liftA':>7}"
      f"{'pA':>8}{'对照B':>9}{'liftB':>7}{'零校验':>8}")
hits, MB, MONB = [], {}, {}
for lo, hi_, nm in RB:
    sel = [(int(t), int(j)) for t, j, r in zip(D["t"], D["j"], rv, strict=True)
           if np.isfinite(r) and lo < r <= hi_ and np.isfinite(PKm[t, j])]
    if not sel:
        continue
    v = np.array([PKm[t, j] for t, j in sel]) >= 2.0
    obs = float(v.mean())
    ca, tha = ctl_a(sel)
    cb, _ = ctl_a(sel, True)
    ra = float(np.median(ca)) if ca.size else np.nan
    rb = float(np.median(cb)) if cb.size else np.nan
    hits.append(obs)
    mp = {}
    for (t, _), xx in zip(sel, v, strict=True):
        mp.setdefault(ymm[t], []).append(int(xx))
    MONB[nm] = mp
    r = dict(档=nm, n=len(sel), rps=float(np.nanmedian([RPS250[t, j] for t, j in sel])),
             obs=obs, ctl_a=ra, liftA=obs / ra if ra > 0 else np.nan,
             pA=float((ca >= obs).mean()) if ca.size else np.nan, gap0=abs(ra - tha),
             ctlB=rb, liftB=obs / rb if rb > 0 else np.nan)
    print(f"{nm:<8}{len(sel):>9,}{r['rps']:>9.1f}{obs:>9.2%}{ra:>9.2%}"
          f"{r['liftA']:>7.2f}{r['pA']:>8.4f}{rb:>9.2%}{r['liftB']:>7.2f}"
          f"{r['gap0']:>8.2%}")
    if nm == ">85":
        MB = r
g_b, p_b = permp(MONB.get(">85", {}), MONB.get("<50", {}))
rho = (float(np.corrcoef(pd.Series(hits).rank(), np.arange(1, len(hits) + 1))[0, 1])
       if len(hits) == 4 else np.nan)
b1 = bool(MB.get("n", 0) >= MIN_NB and MB.get("liftA", 0) >= LIFT_MIN
          and MB.get("pA", 1) < ALPHA)
b2 = bool(np.isfinite(rho) and rho >= RHO_MIN)
b3 = bool(np.isfinite(g_b) and g_b >= GAP_MIN and np.isfinite(p_b) and p_b < ALPHA)
print(f"\n  {'✓' if b1 else '✗'} B① >85 档 liftA {MB.get('liftA',float('nan')):.2f} ≥1.3 "
      f"且 p {MB.get('pA',float('nan')):.4f} < {ALPHA:.5f}")
print(f"  {'✓' if b2 else '✗'} B② 单调性 Spearman {rho:+.2f} ≥ +{RHO_MIN}"
      f"   四档 " + " ".join(f"{h:.2%}" for h in hits))
print(f"  {'✓' if b3 else '✗'} B③ 档间 >85 减 <50 = **{g_b:+.2%}** ≥5pp 且置换 p {p_b:.4f}")
if not a4b:
    print("\n  **锚点④ 不过:B 部分结论作废。**")

D.drop(columns=["t", "j"]).to_csv(f"{OUT}/rps_bucket_rerun.csv", index=False)
print(f"\n→ {OUT}/newhigh_ma100_rps.csv   ({time.time()-t0:.0f}s)")
