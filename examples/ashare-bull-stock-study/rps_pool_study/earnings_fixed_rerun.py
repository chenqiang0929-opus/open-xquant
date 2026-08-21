"""第一〇〇节:用修好的同比口径重跑 §98 与 §99(事前登记)

═══ 为什么重跑 ═══
§98(盈利方向)与 §99(PEAD 五档)用的「净利同比」是 `ni[t]/|ni[t−250]|−1`,
在财报公布日不均匀的节奏下**系统性落早一个报告期**。泰格 300347 逐行核对:

    2017-08-23 中报 vs **2016一季报**  我 +197.5%  真值 **+53.07%**
    2017-10-31 三季 vs **2016中报**    我 +156.4%  真值 **+101.03%**
    2018-04-20 年报 vs **2016三季**    我 +202.0%  真值 **+114.01%**
    2018-05-02 一季 vs **2016三季**    我  −3.8%   真值 **+121.07%**

中报/三季/年报被**高估**(分母偏小),一季报被压成**假负数**。
全市场抽样:4/5 月同比中位 **−67.7%** vs 其余月份 **+54.2%**,差 **122pp** —— 纯日历效应。

**修复模块 `fundamental_yoy.py` 已通过三个锚点**(泰格四行复现真值误差 <0.004%;
报告期标签自洽;日历效应 122pp → **4.1pp**,消除 96.6%)。本节用它重跑。

**§98/§99 的旧结果不追认、不复用**(§78 的处理方式)。

═══ 口径(事前锁定)═══
  同比    `fundamental_yoy.yoy_series`:本期累计 ÷ |上年同一报告期累计| − 1
  A 部分  **复用 §94 的三段突破事件**(锚点② 恒等复现 9,598),按事件日**最近一次公告**
          的净利同比劈成「改善(>0)/恶化(≤0)」—— 与 §98 同形,只换同比口径
  B 部分  **财报公告日本身为事件**,按公告所在年月做横截面五分位 —— 与 §99 同形
  前瞻    6 个月(120 日)峰值 ≥100% = 判据口径;12 个月仅描述
  对照A   同日同市值五分位随机 × 200 组
  对照B   同日同市值 **且当日也创 250 日新高** —— 隔离动量后的基准
  退市股 ffill 参与,绝不剔除

═══ 锚点(不过则全节作废;四个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **§94 事件数恒等复现**:三段突破 6 个月口径 **9,598** 个
  ③ **泰格 300347 四行复现雪球真值**(±0.5pp):中报2017 +53.07% / 三季2017 +101.03%
     / 年报2017 +114.01% / 一季2018 +121.07% —— 证明管线里用的确是修好的口径
  ④ **恒等零校验**:各格对照的中位命中率 vs 同格总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽;**Bonferroni 加严到 0.05/8 = 0.00625**)═══
本节一次做 A、B 两组共 8 个比较,故门槛比 §98/§99 的 0.0125 **更严**,不是放宽。

  **A 部分**(前置:某半格 <300 不判;逐年 <100 不计入 A③)
   A① 「改善」半边 6 个月 ≥100% 率 **减去「恶化」半边 ≥ 1.5pp**,月内置换 **p < 0.00625**
   A② 「改善」半边对**对照B** **lift ≥ 1.3 且 p < 0.00625**
   A③ 逐年「改善 > 恶化」的年份占比 **≥ 80%**

  **B 部分**(前置:某档 <300 不判;逐年 <100 不计入 B③)
   B① Q5 对**对照A** **lift ≥ 1.3 且 p < 0.00625**
   B② Q5 对**对照B** **lift ≥ 1.3 且 p < 0.00625**
   B③ 逐年 Q5 对对照A 的 **lift > 1.0 的年份占比 ≥ 80%**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问**:高增长股本来就在涨 → **堵法:A②/B② 用对照B**;
事件集中在牛市年份 → **堵法:对照按同日抽 + A③/B③ 逐年**;
8 个比较搜出假阳性 → **堵法:Bonferroni 0.05/8**。
**反问**:样本不足 → **前置 n≥300**;同比口径又错 → **锚点③ 用雪球真值恒等核对**;
锚点误杀正确实现(§85/§87/§88/§100-模块 四次病根)→ **四个锚点全是恒等式,
②③ 均已单独预验证可达**。

═══ 事前预测(写下以便被证伪)═══
**A①②③ 全不过;B①②③ 全不过。**
**并且预测:B 部分的「U 形」会消失。**
§99 原表 Q1(同比中位 −167%)命中率 **7.11%** 最高、Q3(+31.7%)**4.69%** 最低,
呈 U 形。**我诊断那是口径产物**(Q1 是被压负的一季报、Q5 是被高估的中报/三季报)。
**若修好口径后 U 形仍在,说明我的诊断错了,必须在正文里明说。**
理由:§89/§92/§94/§98/§99 五次一致显示,任何变量对「当日也在创250日新高的
同市值股」都归零;修口径改的是变量质量,不改这个结构性事实。
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consolidation_screener import THR_DEPTH, load_panel  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NQ, NSEED, SEED, NPERM = 5, 200, 20260814, 2000
TH, UP_MIN, PLAT_MIN, CAP = 0.10, 0.30, 60, 250
BAND = THR_DEPTH
MIN_N, MIN_N_YEAR = 300, 100
ALPHA = 0.05 / 8
GAP_MIN, LIFT_MIN, YR_FRAC = 0.015, 1.3, 0.80
EXP_EV6 = 9598
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
FIRST = np.argmax(np.isfinite(Fa), axis=0)
H2 = pd.DataFrame(Fa).rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(H2) & (Fa >= H2 * 0.9999)
del H2

idxn = idx.tz_localize(None)
pos = {d: i for i, d in enumerate(idxn)}
YOY = np.full((NT, NS), np.nan, dtype=np.float32)
ANN = np.zeros((NT, NS), bool)
anchor3 = {}
for j, c in enumerate(codes):
    try:
        d = yoy_series(c)
    except Exception:
        continue
    if d.empty:
        continue
    for _, r in d.iterrows():
        k = pos.get(r["公告日"])
        if k is None:
            continue
        ANN[k, j] = True
        YOY[k, j] = r["同比"]
        if c == "300347" and (str(r["报告年"]), r["报告期"]) in (
                ("2017", "中报"), ("2017", "三季报"), ("2017", "年报"), ("2018", "一季报")):
            anchor3[(str(r["报告年"]), r["报告期"])] = r["同比"]
    if (j + 1) % 1500 == 0:
        print(f"  同比矩阵 {j+1:,}/{NS:,}  ({time.time()-t0:.0f}s)", flush=True)
YOY = pd.DataFrame(YOY).ffill().to_numpy(np.float32)   # 事件日取最近一次公告
TRUE = {("2017", "中报"): .5307, ("2017", "三季报"): 1.0103,
        ("2017", "年报"): 1.1401, ("2018", "一季报"): 1.2107}
a3 = len(anchor3) == 4 and all(abs(anchor3[k] - v) <= .005 for k, v in TRUE.items())
print(f"  {'✓' if a3 else '✗'} 锚点③ 泰格四行:" +
      " ".join(f"{k[1]} {anchor3.get(k, float('nan')):.2%}" for k in TRUE))

MV = pd.DataFrame({c: pd.to_numeric(
    pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])["float_mv"],
    errors="coerce") for c in codes})
if getattr(MV.index, "tz", None) is not None:
    MV.index = MV.index.tz_localize(None)
MV = MV.reindex(idxn).ffill().to_numpy(float)
QUINT = np.full((NT, NS), -1, dtype=np.int8)
for t in range(NT):
    ok = np.isfinite(MV[t]) & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if ok.sum() < 50:
        continue
    QUINT[t, ok] = np.searchsorted(np.nanquantile(MV[t][ok], [.2, .4, .6, .8]),
                                   MV[t][ok], side="right")
del MV


def fwd_peak(n):
    m = pd.DataFrame(Fa[::-1]).rolling(n, min_periods=1).max().to_numpy(float)[::-1]
    o = np.full((NT, NS), np.nan)
    o[:-1] = m[1:]
    o = (o / Fa - 1.0).astype(np.float32)
    o[NT - n:] = np.nan
    return o


PK = {n: fwd_peak(n) for n, _ in HOR}
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)


def zigzag(px, s0):
    piv = [(s0, "L")]
    ext, ei, up = px[s0], s0, True
    for i in range(s0 + 1, len(px)):
        if up:
            if px[i] > ext:
                ext, ei = px[i], i
            elif px[i] <= ext * (1 - TH):
                piv.append((ei, "H"))
                ext, ei, up = px[i], i, False
        else:
            if px[i] < ext:
                ext, ei = px[i], i
            elif px[i] >= ext * (1 + TH):
                piv.append((ei, "L"))
                ext, ei, up = px[i], i, True
    piv.append((ei, "H" if up else "L"))
    return piv


EV = []
for j in range(NS):
    s0 = int(FIRST[j])
    px = Fa[:, j]
    if not np.isfinite(px[s0]) or NT - s0 < 300:
        continue
    piv = zigzag(px, s0)
    seen = set()
    for a in range(len(piv) - 1):
        i0, k0 = piv[a]
        i1, k1 = piv[a + 1]
        if not (k0 == "L" and k1 == "H") or px[i0] <= 0 or px[i1] / px[i0] - 1 < UP_MIN:
            continue
        b, hi, lo = a + 1, px[i1], px[i1]
        while b + 1 < len(piv):
            q = piv[b + 1][0]
            nh, nl = max(hi, px[q]), min(lo, px[q])
            if nl <= 0 or nh / nl - 1 > BAND:
                break
            hi, lo, b = nh, nl, b + 1
        end = piv[b][0]
        if end - i1 < PLAT_MIN:
            continue
        shi = float(np.nanmax(px[i1:end + 1]))
        w = np.flatnonzero(px[end + 1:min(end + 1 + CAP, NT)] > shi)
        if not w.size:
            continue
        bk = end + 1 + int(w[0])
        if bk in seen:
            continue
        seen.add(bk)
        EV.append((bk, j))
n6 = sum(1 for t, j in EV if np.isfinite(PK[120][t, j]))
a2 = n6 == EXP_EV6
print(f"\n三段突破事件 {len(EV):,};6 个月口径 {n6:,}  "
      f"{'✓' if a2 else '✗'} 锚点②(期望 {EXP_EV6:,})  ({time.time()-t0:.0f}s)", flush=True)

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


def perm_p(mh, ml):
    ms = sorted(set(mh) | set(ml))
    pool, nh = [], []
    for m in ms:
        a, b = mh.get(m, []), ml.get(m, [])
        if len(a) + len(b) == 0:
            continue
        pool.append(np.asarray(list(a) + list(b), float))
        nh.append(len(a))
    nhi = sum(nh)
    nlo = sum(len(v) - k for v, k in zip(pool, nh, strict=True))
    if len(pool) < 12 or nhi == 0 or nlo == 0:
        return np.nan, np.nan
    obs = (sum(v[:k].sum() for v, k in zip(pool, nh, strict=True)) / nhi
           - sum(v[k:].sum() for v, k in zip(pool, nh, strict=True)) / nlo)
    hs, ls = np.zeros(NPERM), np.zeros(NPERM)
    for v, k in zip(pool, nh, strict=True):
        tot, n = int(v.sum()), len(v)
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
    d = hs / nhi - ls / nlo
    return float(obs), float((d >= obs).mean())


ym = idx.to_period("M")
W = 108
ROWS, A, B = [], {}, {}
pm = PK[120]

print(f"\n{'='*W}\nA 部分:三段突破事件按净利同比方向劈分(6 个月峰值 ≥100%)\n{'='*W}")
sub = [(t, j) for t, j in EV if np.isfinite(pm[t, j])]
half = {"改善": [], "恶化": [], "缺失": []}
for t, j in sub:
    v = YOY[t, j]
    half["缺失" if not np.isfinite(v) else ("改善" if v > 0 else "恶化")].append((t, j))
print(f"{'半边':<6}{'事件数':>8}{'≥100%':>9}{'对照A':>9}{'liftA':>7}{'对照B':>9}"
      f"{'liftB':>7}{'pB':>8}{'率差':>9}{'p(置换)':>9}{'零校验':>8}   缺失 {len(half['缺失']):,}")
st = {}
for hn in ("改善", "恶化"):
    ev = half[hn]
    v = np.array([pm[t, j] for t, j in ev]) >= 1.0
    ca, tha = control(ev, pm, False)
    cb, _ = control(ev, pm, True)
    ra = float(np.median(ca)) if ca.size else np.nan
    rb = float(np.median(cb)) if cb.size else np.nan
    mp = {}
    for (t, _), xx in zip(ev, v, strict=True):
        mp.setdefault(ym[t], []).append(int(xx))
    st[hn] = dict(n=len(ev), obs=float(v.mean()), ctlA=ra, ctlB=rb,
                  liftA=float(v.mean()) / ra if ra > 0 else np.nan,
                  liftB=float(v.mean()) / rb if rb > 0 else np.nan,
                  pB=float((cb >= v.mean()).mean()) if cb.size else np.nan,
                  gap0=abs(ra - tha), mon=mp)
g, pv = perm_p(st["改善"]["mon"], st["恶化"]["mon"])
for hn in ("改善", "恶化"):
    d = st[hn]
    print(f"{hn:<6}{d['n']:>8,}{d['obs']:>9.2%}{d['ctlA']:>9.2%}{d['liftA']:>7.2f}"
          f"{d['ctlB']:>9.2%}{d['liftB']:>7.2f}{d['pB']:>8.4f}"
          f"{(f'{g:+.2%}' if hn=='改善' else '—'):>9}"
          f"{(f'{pv:.4f}' if hn=='改善' else '—'):>9}{d['gap0']:>8.2%}")
    ROWS.append(dict(部分="A", 格=hn, **{k: v for k, v in d.items() if k != "mon"},
                     率差=g, p置换=pv))
A = dict(st=st, g=g, pv=pv, half=half)

print(f"\n{'='*W}\nB 部分:财报公告日按净利同比五分位(6 个月峰值 ≥100%)\n{'='*W}")
EVB = [(int(t), int(j)) for t, j in zip(*np.where(ANN), strict=True)]
EVB = [(t, j) for t, j in EVB if np.isfinite(YOY[t, j]) and np.isfinite(pm[t, j])]
bym = {}
for t, j in EVB:
    bym.setdefault(ym[t], []).append((t, j))
BUCK = {}
for m, evs in bym.items():
    v = np.array([YOY[t, j] for t, j in evs])
    if len(v) < 25:
        continue
    e = np.nanquantile(v, [.2, .4, .6, .8])
    for (t, j), q in zip(evs, np.searchsorted(e, v, side="right"), strict=True):
        BUCK.setdefault(int(q), []).append((t, j))
print(f"  公告事件 {len(EVB):,}")
print(f"{'档':<5}{'同比中位':>11}{'事件数':>9}{'≥100%':>9}{'对照A':>9}{'liftA':>7}{'pA':>8}"
      f"{'对照B':>9}{'liftB':>7}{'pB':>8}{'零校验':>8}")
for q in range(5):
    ev = BUCK.get(q, [])
    if not ev:
        continue
    v = np.array([pm[t, j] for t, j in ev]) >= 1.0
    obs = float(v.mean())
    ca, tha = control(ev, pm, False)
    cb, _ = control(ev, pm, True)
    ra = float(np.median(ca)) if ca.size else np.nan
    rb = float(np.median(cb)) if cb.size else np.nan
    r = dict(部分="B", 格=f"Q{q+1}", 同比中位=float(np.nanmedian([YOY[t, j] for t, j in ev])),
             n=len(ev), obs=obs, ctlA=ra, liftA=obs / ra if ra > 0 else np.nan,
             pA=float((ca >= obs).mean()) if ca.size else np.nan,
             ctlB=rb, liftB=obs / rb if rb > 0 else np.nan,
             pB=float((cb >= obs).mean()) if cb.size else np.nan, gap0=abs(ra - tha))
    ROWS.append(r)
    print(f"Q{q+1:<4}{r['同比中位']:>11.1%}{r['n']:>9,}{obs:>9.2%}{ra:>9.2%}"
          f"{r['liftA']:>7.2f}{r['pA']:>8.4f}{rb:>9.2%}{r['liftB']:>7.2f}"
          f"{r['pB']:>8.4f}{r['gap0']:>8.2%}")
    if q == 4:
        B = dict(r=r, ev=ev)
R = pd.DataFrame(ROWS)

ya, yb = [], []
for y in sorted({idx[t].year for t, _ in sub}):
    aa = [(t, j) for t, j in A["half"]["改善"] if idx[t].year == y]
    bb = [(t, j) for t, j in A["half"]["恶化"] if idx[t].year == y]
    if len(aa) >= MIN_N_YEAR and len(bb) >= MIN_N_YEAR:
        ra_ = float((np.array([pm[t, j] for t, j in aa]) >= 1.0).mean())
        rb_ = float((np.array([pm[t, j] for t, j in bb]) >= 1.0).mean())
        ya.append(ra_ > rb_)
for y in sorted({idx[t].year for t, _ in B["ev"]}):
    ev = [(t, j) for t, j in B["ev"] if idx[t].year == y]
    if len(ev) < MIN_N_YEAR:
        continue
    o = float((np.array([pm[t, j] for t, j in ev]) >= 1.0).mean())
    c, _ = control(ev, pm, False)
    md = float(np.median(c)) if c.size else np.nan
    yb.append(md > 0 and o / md > 1.0)
fa_ = float(np.mean(ya)) if ya else np.nan
fb_ = float(np.mean(yb)) if yb else np.nan

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② §94 事件数恒等复现")
print(f"  {'✓' if a3 else '✗'} 锚点③ 泰格四行复现雪球真值")
z = R["gap0"]
a4 = bool(z.notna().all() and (z <= 0.03).all())
print(f"  {'✓' if a4 else '✗'} 锚点④ 恒等零校验 最大差 {z.max():.2%} ≤ 3pp")
for ok, nm in ((a2, "锚点②"), (a3, "锚点③"), (a4, "锚点④")):
    if not ok:
        bad.append(nm)

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,Bonferroni 0.05/8={ALPHA})\n{'='*W}")
S = A["st"]
ea = S["改善"]["n"] >= MIN_N and S["恶化"]["n"] >= MIN_N
a1 = bool(ea and A["g"] >= GAP_MIN and A["pv"] < ALPHA)
a2c = bool(ea and S["改善"]["liftB"] >= LIFT_MIN and S["改善"]["pB"] < ALPHA)
a3c = bool(np.isfinite(fa_) and fa_ >= YR_FRAC)
rb_ = B["r"]
b1 = bool(rb_["n"] >= MIN_N and rb_["liftA"] >= LIFT_MIN and rb_["pA"] < ALPHA)
b2 = bool(rb_["n"] >= MIN_N and rb_["liftB"] >= LIFT_MIN and rb_["pB"] < ALPHA)
b3 = bool(np.isfinite(fb_) and fb_ >= YR_FRAC)
print(f"  A① 率差 {A['g']:+.2%} ≥1.5% 且 p {A['pv']:.4f} < {ALPHA:.5f}   {'✓' if a1 else '✗'}")
print(f"  A② 改善 liftB {S['改善']['liftB']:.2f} ≥1.3 且 p {S['改善']['pB']:.4f}   {'✓' if a2c else '✗'}")
print(f"  A③ 逐年占比 {fa_:.1%} ≥80%   {'✓' if a3c else '✗'}")
print(f"  B① Q5 liftA {rb_['liftA']:.2f} ≥1.3 且 p {rb_['pA']:.4f}   {'✓' if b1 else '✗'}")
print(f"  B② Q5 liftB {rb_['liftB']:.2f} ≥1.3 且 p {rb_['pB']:.4f}   {'✓' if b2 else '✗'}")
print(f"  B③ 逐年占比 {fb_:.1%} ≥80%   {'✓' if b3 else '✗'}")
u = R[R["部分"] == "B"].set_index("格")["obs"]
uform = bool(len(u) == 5 and u["Q1"] > u["Q3"] and u["Q5"] > u["Q3"])
print(f"\n  **U 形是否仍在**:Q1 {u.get('Q1', np.nan):.2%} / Q3 {u.get('Q3', np.nan):.2%}"
      f" / Q5 {u.get('Q5', np.nan):.2%}  ->  {'**仍是 U 形 —— 我的诊断错了**' if uform else '**U 形消失,诊断成立**'}")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif not any([a1, a2c, a3c, b1, b2, b3]):
    print("  **结论:修好口径后,盈利方向与盈利分档在右尾口径下仍然分不出东西。**")
    print("  **事前预测命中(A①②③、B①②③ 全不过)。**")
else:
    print("  **结论:有判据通过,详见上表 —— 事前预测被证伪的部分需在正文明说。**")

R.to_csv(f"{OUT}/earnings_fixed_rerun.csv", index=False)
print(f"\n→ {OUT}/earnings_fixed_rerun.csv   ({time.time()-t0:.0f}s)")
