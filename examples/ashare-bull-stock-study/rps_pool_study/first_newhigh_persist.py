"""第一〇三节:首次新高 + 后续持续性 —— 把对照从「结果」换成「同类事件」(事前登记)

═══ 起因:用户指出我的对照组是结果不是归因 ═══
> **「当下正在创 250 日新高,看起来是结果不是归因吧。
>   最好是首次 250 日新高之后,关注它的持续性。」**

**这个批评成立。** §89~§102 的对照B 一直是 `当日创250日新高`,那是**同期状态**,
几乎与「股价在涨」同义;拿它当基准,等于用正在涨的股票比正在涨的股票。
**更糟的是它把两种东西搅在一起**:刚刚**首次**突破的(趋势起点)、
和已经连创几十次新高的(趋势中段)。**这可能是 §77 以来最大的一个设计问题。**

**§64 测过「新高密度」但方向相反** —— 那测的是**买点之前**的回望密度
(N1_启动前、N2_买点前,中位数 0,首轮退化作废、修复后 0/4)。
**本节测的是首次新高之后的前瞻确认,没测过。**

**同时吸收 §102 暴露的第二个问题**:本项目主口径「6 个月 ≥100%」来自用户最初的
提问,但四只案例股同一批信号 6 个月 **0/11**、24 个月 **4/9** ——
**大涨是 2~3 年的事**。故本节判据口径改为 **24 个月峰值 ≥200%**。

═══ 口径(事前锁定)═══
  **事件**  **首次创 250 日新高**:当日 `收盘 ≥ 过去250日最高×0.9999`,
            且**此前 120 个交易日内未创过 250 日新高**(GAP=120)
  **分档**  首次新高日之后 **60 个交易日内**再创新高的次数:**0 / 1–5 / 6–15 / 16+**
  **入场**  **首次新高后第 60 个交易日收盘**(t0+60)—— 分档只用 [t0+1, t0+60] 的
            信息,入场在观察窗末端,**无前视**(锚点③ 用截断面板证明)
  前瞻    **24 个月(500 日)峰值 ≥200% = 判据口径**;6/12 个月 ≥100% 仅描述
  对照A   同日同市值五分位随机 × 200 组
  **档间对照**  最高档 vs 最低档 —— **两者都是首次新高事件,只是后续持续性不同,
            这是本项目第一次用「同类事件」做对照,而不是用「结果」**
  退市股 ffill 参与,绝不剔除

═══ 锚点(不过则全节作废;四个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **四只案例恒等复现首次新高事件数**(单只已预验证):
     宇通 600066 **9** 次、匠心 301061 **1** 次、嘉益 301004 **2** 次、泰格 300347 **3** 次
  ③ **无前视校验**:把面板截断到 2020-12-31 重算事件与分档,
     与全样本版本在该日之前**逐个相同**
  ④ **恒等零校验**:各档对照A 的中位命中率 vs 同档总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽;Bonferroni **0.05/8 = 0.00625**)═══
  **前置条件**:某档事件数 **< 300** 不判
  ① **最高档(16+)对对照A**:24 个月 ≥200% 的 **lift ≥ 1.3 且 p < 0.00625**
  ② **档间(核心)**:最高档 ≥200% 率**减去最低档(0 次)≥ 5pp**,
     且月内标签置换 **p < 0.00625** —— **同为首次新高事件,只按后续持续性劈分**
  ③ **单调性**:四档 ≥200% 命中率对档位序号的 **Spearman ≥ +0.8**

**①②③ 全过 = 「首次新高之后的持续性」是一个独立的、单调的右尾来源,
且对照是同类事件不是结果 —— 那将是本项目第一个真正站住的信号。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问**:持续创新高 = 已经涨了 → **堵法:判据② 的对照是同类事件(同为首次新高),
不是「没涨的股票」**;高档集中在牛市 → **堵法:对照按同日抽 + 月内置换**;
4 档搜索出假阳性 → **堵法:判据只压最高档(事前指定)+ Bonferroni 0.05/8**;
**分档用了未来信息** → **堵法:入场在 t0+60,分档窗口完全在入场之前**。

**反问**:分档样本不足 → **前置 n≥300**;
500 日前瞻砍掉 2024-08 后的事件 → **逐档同等受影响**;
6 个月口径太苛刻(§102 的教训)→ **判据口径已改为 24 个月 ≥200%**;
锚点误杀正确实现(已四次病根)→ **四个锚点全是恒等式,② 已单只预验证可达**。

═══ 事前预测(写下以便被证伪)═══
**①②③ 全通过。**
**这是本项目我第二次预测「会通过」(第一次是 §98,错了)。**
理由:分档用的是**入场前**的信息,且分档本身就是**趋势确认** ——
「首次突破后 60 日内再创 16 次以上新高」= 资金持续推动。
§62 的核心结论是「资金推动的强势 = 胜率低 + 右尾肥」,
而本项目 25 节里唯一反复站住的东西就是动量。**这一节测的正是动量本身,
而且第一次用同类事件做对照,不再被「结果当归因」污染。**
**若 ② 不过,说明连「持续创新高」也不带来同类事件之内的增量,
那么「A 股右尾不可事前预测」这个结论就可以彻底写死了 —— 我会明说我又错了。**
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
NQ, NSEED, SEED, NPERM = 5, 200, 20260814, 2000
GAP, WIN = 120, 60
MIN_N = 300
ALPHA = 0.05 / 8
LIFT_MIN, GAP_MIN, RHO_MIN = 1.3, 0.05, 0.80
BUCK = [(0, 0, "0"), (1, 5, "1–5"), (6, 15, "6–15"), (16, 9999, "16+")]
HOR = [(120, "6个月", 1.0), (250, "12个月", 1.0), (500, "24个月", 2.0)]
EXP = {"600066": 9, "301061": 1, "301004": 2, "300347": 3}
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


NH = newhi(Fa)


def find_events(nh, upto=None):
    """首次新高事件 (t0, j)。此前 GAP 日内未创新高。"""
    lim = nh.shape[0] if upto is None else upto
    out = []
    for j in range(nh.shape[1]):
        col = nh[:, j]
        for t in np.flatnonzero(col[:lim]):
            if t < 250 or col[max(t - GAP, 0):t].any():
                continue
            out.append((int(t), j))
    return out


EV0 = find_events(NH)
cnt = {}
for t, j in EV0:
    cnt[codes[j]] = cnt.get(codes[j], 0) + 1
a2 = all(cnt.get(c, 0) == v for c, v in EXP.items())
print(f"首次新高事件 **{len(EV0):,}** 个")
print(f"  {'✓' if a2 else '✗'} 锚点② 四只案例:" +
      "  ".join(f"{c} {cnt.get(c,0)}(期望 {v})" for c, v in EXP.items()))

kc = int(np.searchsorted(idx, pd.Timestamp(CUT, tz=idx.tz), side="right"))
EVc = find_events(newhi(Fa[:kc]), upto=kc)
a3 = set(EVc) == {(t, j) for t, j in EV0 if t < kc}
print(f"  {'✓' if a3 else '✗'} 锚点③ 无前视:截断到 {CUT}(前 {kc} 行)重算,"
      f"事件 {len(EVc):,} vs {sum(1 for t,_ in EV0 if t<kc):,}")

MV = pd.DataFrame({c: pd.to_numeric(
    pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])["float_mv"],
    errors="coerce") for c in codes})
if getattr(MV.index, "tz", None) is not None:
    MV.index = MV.index.tz_localize(None)
MV = MV.reindex(idx.tz_localize(None)).ffill().to_numpy(float)
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


PK = {n: fwd_peak(n) for n, _, _ in HOR}
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)

# 入场 t0+60,分档用 [t0+1, t0+60]
EV = []
for t, j in EV0:
    k = t + WIN
    if k >= NT:
        continue
    EV.append((k, j, int(NH[t + 1:k + 1, j].sum())))
print(f"可入场事件(t0+{WIN} 在面板内){len(EV):,}")

rng = np.random.default_rng(SEED)
ym = idx.to_period("M")


def control(sub, pm, thr):
    cn = {}
    for t, j, _ in sub:
        q = int(QUINT[t, j])
        if q >= 0:
            cn[(t, q)] = cn.get((t, q), 0) + 1
    hit, tot, th, tn = np.zeros(NSEED), 0, 0.0, 0
    for (t, q), k in cn.items():
        pool = np.flatnonzero((QUINT[t] == q) & np.isfinite(pm[t]))
        if pool.size == 0:
            continue
        v = pm[t, pool] >= thr
        th += float(v.mean()) * k
        tn += k
        hit += v[rng.integers(0, pool.size, size=(NSEED, k))].sum(axis=1)
        tot += k
    if tot == 0:
        return np.array([]), np.nan
    return hit / tot, th / tn


def perm_p(mh, ml):
    ms = sorted(set(mh) | set(ml))
    pool, nh_ = [], []
    for m in ms:
        a, b = mh.get(m, []), ml.get(m, [])
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
        tt, n = int(v.sum()), len(v)
        if k == 0:
            ls += tt
            continue
        if k == n:
            hs += tt
            continue
        s = (rng.hypergeometric(tt, n - tt, k, size=NPERM) if 0 < tt < n
             else np.full(NPERM, tt * k / n, float))
        hs += s
        ls += tt - s
    d = hs / n1 - ls / n0
    return float(obs), float((d >= obs).mean())


W = 100
rows, MAIN, MONS = [], {}, {}
for n, hname, thr in HOR:
    pm = PK[n]
    tag = "  【判据口径】" if n == 500 else "  (描述)"
    print(f"\n{'='*W}\n{hname}峰值 ≥{thr:.0%}{tag}   入场 = 首次新高后第 {WIN} 日\n{'='*W}")
    print(f"{'后60日新高次数':<14}{'事件数':>9}{f'≥{thr:.0%}':>9}{'对照A':>9}"
          f"{'liftA':>7}{'pA':>8}{'零校验':>8}")
    hits = []
    for lo, hi, nm in BUCK:
        ev = [(t, j, c) for t, j, c in EV if lo <= c <= hi and np.isfinite(pm[t, j])]
        if not ev:
            continue
        v = np.array([pm[t, j] for t, j, _ in ev]) >= thr
        obs = float(v.mean())
        ca, tha = control(ev, pm, thr)
        ra = float(np.median(ca)) if ca.size else np.nan
        r = dict(前瞻=hname, 档=nm, n=len(ev), obs=obs, ctlA=ra,
                 liftA=obs / ra if ra > 0 else np.nan,
                 pA=float((ca >= obs).mean()) if ca.size else np.nan,
                 gap0=abs(ra - tha))
        rows.append(r)
        hits.append(obs)
        print(f"{nm:<14}{len(ev):>9,}{obs:>9.2%}{ra:>9.2%}{r['liftA']:>7.2f}"
              f"{r['pA']:>8.4f}{r['gap0']:>8.2%}")
        if n == 500:
            mp = {}
            for (t, _, _), xx in zip(ev, v, strict=True):
                mp.setdefault(ym[t], []).append(int(xx))
            MONS[nm] = mp
            if nm == "16+":
                MAIN = r
    if n == 500 and len(hits) == 4:
        MAIN["hits"] = hits
        MAIN["rho"] = float(np.corrcoef(pd.Series(hits).rank(),
                                        np.arange(1, 5))[0, 1])
R = pd.DataFrame(rows)

g, pv = perm_p(MONS.get("16+", {}), MONS.get("0", {}))
print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② 四只案例事件数恒等复现")
print(f"  {'✓' if a3 else '✗'} 锚点③ 无前视截断校验")
z = R[R["前瞻"] == "24个月"]["gap0"]
a4 = bool(z.notna().all() and (z <= 0.03).all())
print(f"  {'✓' if a4 else '✗'} 锚点④ 恒等零校验 最大差 {z.max():.2%} ≤ 3pp")
for ok, nm in ((a2, "锚点②"), (a3, "锚点③"), (a4, "锚点④")):
    if not ok:
        bad.append(nm)

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,Bonferroni {ALPHA})\n{'='*W}")
c1 = bool(MAIN.get("n", 0) >= MIN_N and MAIN.get("liftA", 0) >= LIFT_MIN
          and MAIN.get("pA", 1) < ALPHA)
c2 = bool(np.isfinite(g) and g >= GAP_MIN and np.isfinite(pv) and pv < ALPHA)
rho = MAIN.get("rho", np.nan)
c3 = bool(np.isfinite(rho) and rho >= RHO_MIN)
print(f"  前置:16+ 档事件数 {MAIN.get('n',0):,} ≥ {MIN_N}")
print(f"  {'✓' if c1 else '✗'} ① 16+ 对照A lift {MAIN.get('liftA',float('nan')):.2f} ≥1.3 "
      f"且 p {MAIN.get('pA',float('nan')):.4f} < {ALPHA}")
print(f"  {'✓' if c2 else '✗'} ② 档间(同类事件)16+ 减 0 档 = **{g:+.2%}** ≥5pp "
      f"且置换 p {pv:.4f} < {ALPHA}")
print(f"  {'✓' if c3 else '✗'} ③ 单调性 Spearman {rho:+.2f} ≥ +{RHO_MIN}"
      f"   四档命中率 " + " ".join(f"{h:.2%}" for h in MAIN.get("hits", [])))
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:首次新高之后的持续性是独立、单调的右尾来源,且对照是同类事件。**")
    print("  **本项目第一个真正站住的信号。事前预测命中。**")
elif c2:
    print("  **结论:档间差成立(持续性有增量),但最高档对同市值随机未达 lift 1.3。**")
else:
    print("  **结论:连「首次新高后的持续性」也不带来同类事件之内的增量。**")
    print("  **事前预测被证伪 —— 我第二次押「会通过」,又错了。**")

R.to_csv(f"{OUT}/first_newhigh_persist.csv", index=False)
print(f"\n→ {OUT}/first_newhigh_persist.csv   ({time.time()-t0:.0f}s)")
