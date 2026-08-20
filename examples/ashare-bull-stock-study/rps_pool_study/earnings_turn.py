"""第九十七节:业绩反转 vs 价格形态 —— 形态是不是只是盈利的影子(事前登记)

═══ 起因:三次独立尝试,纯价格路线全部停在同一格 ═══
§89 第三段 liftB **0.97**、§92 第一段置换 p **0.42~0.75**、§94 新分段 liftB **1.01**。
**三套不同的检测逻辑、三种不同的分段方式,只要把对照换成
「当日也在创 250 日新高的同市值股」,增量一律归零。**

而宇通身上唯一把成败分开的东西**不是形态,是盈利方向**
(`case_yutong_why.py`,4 成 5 败):

    净利润同比  成功组中位 **+118.1%**  vs  失败组 **−16.5%**
    ROE 同比    **+5.62**              vs  **−0.60**
    2021-07 那次失败,净利润是 **−1.11 亿(亏损)**

**若业绩反转是真因,则 §77 以来所有纯技术检验测不出东西就有了统一解释 ——
不是形态不管用,是形态只是盈利的影子。本节测它。**

═══ 前视风险已排查(必须先说)═══
财报字段在本数据里**按公告日打戳,不是报告期末**。实测宇通 eps 变更日:
2013-08-20 / 2013-10-31 / 2014-03-25 / 2015-04-30 …,
月份分布 {3:8, 4:12, 5:6, 8:12, 9:1, 10:10, 11:3} ——
**6 月和 12 月为 0**。若按报告期末打戳,半年报必落 6 月、年报必落 12 月。
**故无需额外滞后;锚点③ 在全市场上把这条做成恒等校验。**

═══ 口径(事前锁定)═══
  事件    **复用 §94 的三段突破事件**(ZigZag θ=10%,同一套代码,
          锚点② 要求事件数恒等复现 **9,598**)—— 这样「形态」这一维完全固定,
          唯一变化的是按盈利方向劈分
  劈分    **ROE 同比 = roe[t] − roe[t−250]**(250 个交易日前落在上年同一报告期)
          **改善 = >0;恶化 = ≤0**;缺失(未披露)的事件**单列不参与判据**
  前瞻    6 个月(120 日)峰值 ≥100% = 判据口径;12 个月仅描述
  对照A   同日同市值五分位随机 × 200 组
  对照B   同日同市值 **且当日也创 250 日新高** —— 隔离动量后的基准
  检验    月内标签置换 × 2000(超几何精确抽样),标定见 §90 事前登记
  净利同比 作为稳健性口径一并报出,**判据只压在 ROE 同比上**

═══ 锚点(不过则全节作废;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **§94 事件数恒等复现**:三段突破事件在 6 个月口径下 **9,598** 个
  ③ **公告日校验**:全市场 roe 变更日落在 **6 月或 12 月**的比例 **< 2%**
     (若按报告期末打戳,半年报/年报必然大量落在这两个月)
  ④ **恒等零校验**:各半格对照的中位命中率 vs 同格总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽)═══
  **前置条件**:某半格事件数 **< 300** 不判;逐年某年 **< 100** 不计入判据③
  ① **增量**:「ROE 同比改善」半边的 6 个月 ≥100% 率**减去「恶化」半边 ≥ 1.5pp**,
     且月内置换 **p < 0.05/4 = 0.0125**
  ② **不只是动量**:「改善」半边对**对照B** 的 **lift ≥ 1.3 且 p < 0.0125**
  ③ **逐年一致**(§91 立的规矩):逐年「改善半边率 > 恶化半边率」的年份占比 **≥ 80%**

**①②③ 全过 = 盈利方向携带形态之外、动量之外的信息 ——
那将是本项目第一个站住的非价格因子,也解释了 §89/§92/§94 为何一律归零。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问:什么会让它通过而不回答问题?**
→ 盈利改善的股票恰好也在涨(动量)→ **堵法:判据② 用对照B**。
→ 改善组集中在牛市年份 → **堵法:置换保留月份构成 + 判据③ 逐年**。
→ 前视(财报未公布就用)→ **堵法:锚点③ 公告日恒等校验**。
→ 幅度大但不显著,或显著但幅度小 → **堵法:①② 都是「幅度 + 显著性」双要求**
  (§92 的教训:判据只有幅度门槛会和显著性打架)。

**反问:什么会让它不通过而与问题无关?**
→ 某半格样本不足 → **堵法:前置 n≥300**。
→ ROE 同比对财报期口径敏感 → **堵法:净利同比一并报出;
  若两个口径结论相反,必须在正文里说,不得只报有利的那个**。
→ 锚点误杀正确实现 → **堵法:四个锚点全是恒等式,② 已在 §94 实测**。

═══ 事前预测(写下以便被证伪)═══
**① 通过、② 通过、③ 不通过。**
**这是本项目我第一次预测某条「会通过」。**
理由:盈利是全新维度,不在 §77-§94 那条纯价格路线上;
文献里的盈余动量(PEAD)有稳定证据;宇通 4 成 5 败上盈利方向是唯一分得开的量。
③ 我预测不通过,因为本项目**每一个逐年检验都翻过号**(§91 的「创新高」6/14)。
**若 ① 或 ② 不通过,说明连基本面方向也解释不了右尾,
那么「A 股 6 个月 +100% 基本不可事前预测」这个结论就要写死了 —— 我会明说我错了。**
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
from consolidation_screener import THR_DEPTH, load_panel  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NQ, NSEED, SEED, NPERM = 5, 200, 20260814, 2000
TH, UP_MIN, PLAT_MIN, CAP = 0.10, 0.30, 60, 250
BAND = THR_DEPTH
MIN_N, MIN_N_YEAR, NCELL = 300, 100, 4
ALPHA = 0.05 / NCELL
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
HI250 = pd.DataFrame(Fa).rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(HI250) & (Fa >= HI250 * 0.9999)
del HI250

FUND = ["roe", "net_income", "float_mv"]
raw = {c: pd.read_parquet(f"{DATA}/{c}.parquet", columns=FUND) for c in codes}
MAT = {}
for f in FUND:
    df = pd.DataFrame({c: pd.to_numeric(v[f], errors="coerce") for c, v in raw.items()})
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    MAT[f] = df.reindex(idx.tz_localize(None)).ffill().to_numpy(float)
del raw
print(f"基本面矩阵完成  ({time.time()-t0:.0f}s)", flush=True)

# 锚点③ 公告日校验
ROE = MAT["roe"]
chg = np.diff(ROE, axis=0) != 0
mon = np.array([q.month for q in idx[1:]])
n_all = int(chg.sum())
n_612 = int(chg[np.isin(mon, [6, 12])].sum())
a3 = n_all > 0 and n_612 / n_all < 0.02
print(f"  {'✓' if a3 else '✗'} 锚点③ 公告日校验:roe 变更 {n_all:,} 次,"
      f"落在 6/12 月的 {n_612:,} 次 = {n_612/max(n_all,1):.2%} < 2%")

MVa = MAT["float_mv"]
QUINT = np.full((NT, NS), -1, dtype=np.int8)
for t in range(NT):
    ok = np.isfinite(MVa[t]) & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if ok.sum() < 50:
        continue
    QUINT[t, ok] = np.searchsorted(np.nanquantile(MVa[t][ok], [.2, .4, .6, .8]),
                                   MVa[t][ok], side="right")
DROE = np.full((NT, NS), np.nan)
DROE[250:] = ROE[250:] - ROE[:-250]
DNI = np.full((NT, NS), np.nan)
NI = MAT["net_income"]
DNI[250:] = np.where(np.abs(NI[:-250]) > 0, NI[250:] / np.abs(NI[:-250]) - 1, np.nan)
del MAT, MVa


def fwd_peak(n):
    m = pd.DataFrame(Fa[::-1]).rolling(n, min_periods=1).max().to_numpy(float)[::-1]
    out = np.full((NT, NS), np.nan)
    out[:-1] = m[1:]
    out = (out / Fa - 1.0).astype(np.float32)
    out[NT - n:] = np.nan
    return out


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
print(f"\n三段突破事件 {len(EV):,} 个;6 个月口径 {n6:,}")
print(f"  {'✓' if a2 else '✗'} 锚点② §94 事件数恒等复现(期望 {EXP_EV6:,})"
      f"  ({time.time()-t0:.0f}s)", flush=True)

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


def perm_p(mon_hi, mon_lo):
    ms = sorted(set(mon_hi) | set(mon_lo))
    pool, nh = [], []
    for m in ms:
        a, b = mon_hi.get(m, []), mon_lo.get(m, [])
        if len(a) + len(b) == 0:
            continue
        pool.append(np.asarray(list(a) + list(b), float))
        nh.append(len(a))
    n_hi = sum(nh)
    n_lo = sum(len(v) - k for v, k in zip(pool, nh, strict=True))
    if len(pool) < 12 or n_hi == 0 or n_lo == 0:
        return np.nan, np.nan
    obs = (sum(v[:k].sum() for v, k in zip(pool, nh, strict=True)) / n_hi
           - sum(v[k:].sum() for v, k in zip(pool, nh, strict=True)) / n_lo)
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
    d = hs / n_hi - ls / n_lo
    return float(obs), float((d >= obs).mean())


ym = idx.to_period("M")
W = 112
OUTR, MAIN = [], {}
for split, SPM in (("ROE同比", DROE), ("净利同比", DNI)):
    for n, hname in HOR:
        pm = PK[n]
        sub = [(t, j) for t, j in EV if np.isfinite(pm[t, j])]
        half = {"改善": [], "恶化": [], "缺失": []}
        for t, j in sub:
            v = SPM[t, j]
            half["缺失" if not np.isfinite(v) else ("改善" if v > 0 else "恶化")].append((t, j))
        st = {}
        for hn in ("改善", "恶化"):
            ev = half[hn]
            if not ev:
                continue
            vv = np.array([pm[t, j] for t, j in ev]) >= 1.0
            ca, tha = control(ev, pm, False)
            cb, _ = control(ev, pm, True)
            ra = float(np.median(ca)) if ca.size else np.nan
            rb = float(np.median(cb)) if cb.size else np.nan
            mp = {}
            for (t, _), xx in zip(ev, vv, strict=True):
                mp.setdefault(ym[t], []).append(int(xx))
            st[hn] = dict(n=len(ev), obs=float(vv.mean()), ctlA=ra, ctlB=rb,
                          liftA=float(vv.mean()) / ra if ra > 0 else np.nan,
                          liftB=float(vv.mean()) / rb if rb > 0 else np.nan,
                          pB=float((cb >= vv.mean()).mean()) if cb.size else np.nan,
                          gap0=abs(ra - tha), mon=mp)
        if len(st) < 2:
            continue
        g, pv = perm_p(st["改善"]["mon"], st["恶化"]["mon"])
        print(f"\n{'='*W}\n{split} × {hname}峰值 ≥100%"
              f"{'  【判据口径】' if (n == 120 and split == 'ROE同比') else '  (描述)'}"
              f"   缺失 {len(half['缺失']):,} 个不参与\n{'='*W}")
        print(f"{'半边':<6}{'事件数':>8}{'≥100%':>9}{'对照A':>9}{'liftA':>7}"
              f"{'对照B':>9}{'liftB':>7}{'pB':>8}{'率差':>9}{'p(置换)':>9}{'零校验':>8}")
        for hn in ("改善", "恶化"):
            dd = st[hn]
            sh = f"{g:+.2%}" if hn == "改善" else "—"
            sp = f"{pv:.4f}" if hn == "改善" else "—"
            print(f"{hn:<6}{dd['n']:>8,}{dd['obs']:>9.2%}{dd['ctlA']:>9.2%}"
                  f"{dd['liftA']:>7.2f}{dd['ctlB']:>9.2%}{dd['liftB']:>7.2f}"
                  f"{dd['pB']:>8.4f}{sh:>9}{sp:>9}{dd['gap0']:>8.2%}")
            OUTR.append(dict(劈分=split, 前瞻=hname, 半边=hn, **{
                k: v for k, v in dd.items() if k != "mon"}, 率差=g, p置换=pv))
        if split == "ROE同比" and n == 120:
            MAIN = dict(st=st, g=g, pv=pv, half=half)

print(f"\n{'='*W}\n逐年:ROE 同比改善半边 vs 恶化半边(6 个月 ≥100% 率)\n{'='*W}")
pm = PK[120]
yr = []
for y in sorted({idx[t].year for t, _ in EV}):
    a = [(t, j) for t, j in MAIN["half"]["改善"] if idx[t].year == y]
    b = [(t, j) for t, j in MAIN["half"]["恶化"] if idx[t].year == y]
    if len(a) < MIN_N_YEAR or len(b) < MIN_N_YEAR:
        continue
    ra = float((np.array([pm[t, j] for t, j in a]) >= 1.0).mean())
    rb = float((np.array([pm[t, j] for t, j in b]) >= 1.0).mean())
    yr.append(dict(年=y, n改善=len(a), n恶化=len(b), 率改善=ra, 率恶化=rb, 差=ra - rb))
    print(f"  {y}  改善 n={len(a):>5,} {ra:>6.2%}   恶化 n={len(b):>5,} {rb:>6.2%}"
          f"   差 {ra-rb:>+7.2%}  {'✓' if ra > rb else '✗'}")
Y = pd.DataFrame(yr)
yfrac = float((Y["差"] > 0).mean()) if len(Y) else np.nan

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② §94 事件数恒等复现")
print(f"  {'✓' if a3 else '✗'} 锚点③ 公告日校验(无前视)")
R = pd.DataFrame(OUTR)
z = R[(R["劈分"] == "ROE同比") & (R["前瞻"] == "6个月")]["gap0"]
a4 = bool(z.notna().all() and (z <= 0.03).all())
print(f"  {'✓' if a4 else '✗'} 锚点④ 恒等零校验 最大差 "
      f"{z.max():.2%} ≤ 3pp" if z.notna().all() else "  ✗ 锚点④ 算不出 = 不通过")
for ok, nm in ((a2, "锚点②"), (a3, "锚点③"), (a4, "锚点④")):
    if not ok:
        bad.append(nm)

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*W}")
S = MAIN["st"]
elig = S["改善"]["n"] >= MIN_N and S["恶化"]["n"] >= MIN_N
print(f"  前置条件:改善 {S['改善']['n']:,} / 恶化 {S['恶化']['n']:,},"
      f"两半都 ≥{MIN_N}:{'是' if elig else '否'};逐年合格 {len(Y)} 年")
c1 = bool(elig and MAIN["g"] >= GAP_MIN and MAIN["pv"] < ALPHA)
c2 = bool(elig and S["改善"]["liftB"] >= LIFT_MIN and S["改善"]["pB"] < ALPHA)
c3 = bool(np.isfinite(yfrac) and yfrac >= YR_FRAC)
print(f"  {'✓' if c1 else '✗'} 判据① 率差 {MAIN['g']:+.2%} ≥ {GAP_MIN:.1%} "
      f"且置换 p {MAIN['pv']:.4f} < {ALPHA}")
print(f"  {'✓' if c2 else '✗'} 判据② 改善半边 liftB {S['改善']['liftB']:.2f} ≥ {LIFT_MIN} "
      f"且 p {S['改善']['pB']:.4f} < {ALPHA}")
print(f"  {'✓' if c3 else '✗'} 判据③ 逐年改善>恶化 占比 {yfrac:.1%} ≥ {YR_FRAC:.0%}")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:盈利方向携带形态之外、动量之外的信息 —— 本项目第一个站住的非价格因子。**")
elif c1 and c2:
    print("  **结论:盈利方向有独立增量,但逐年不稳定(§91 同款),不足以称为因子。**")
elif c1:
    print("  **结论:盈利改善半边确实更好,但对隔离动量的对照归零 —— 又一次落回动量。**")
else:
    print("  **结论:连基本面方向也解释不了右尾。**")
    print("  **事前预测被证伪 —— 我预测①②会通过,错了。**")
    print("  **「A 股 6 个月 +100% 基本不可事前预测」这个结论可以写死了。**")

R.to_csv(f"{OUT}/earnings_turn.csv", index=False)
Y.to_csv(f"{OUT}/earnings_turn_yearly.csv", index=False)
print(f"\n→ {OUT}/earnings_turn.csv + _yearly.csv   ({time.time()-t0:.0f}s)")
