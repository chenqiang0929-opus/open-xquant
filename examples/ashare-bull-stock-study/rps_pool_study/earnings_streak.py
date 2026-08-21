"""第一〇一节:业绩持续性 —— 连增越久,涨幅是否可持续更大(事前登记)

═══ 起因:用户的假设,以及一个到现在才能测的量 ═══
用户看泰格医药 300347 提出:
> **「业绩存在持续性的股票,股价涨幅是否可持续更大?」**

**这个量到 §100 修好同比口径之后才第一次可算。** 旧口径下泰格「最长连续正增长」
只有 **3 期**(被一季报假负数机械截断);修好后是 **21 期**(2017-05 一季报 → 2022-05
一季报,约 5.25 年),与雪球年报口径「2017-2021 连续 5 年高增长」完全吻合。

**顺带一个已实测的观察(不作判据,仅记录):泰格 streak 在 2022-05-05 才见顶 21 期,
而股价高点是 2021-07-01(197.53)—— 持续性指标比股价晚了 10 个月见顶。**

§98/§99/§100 测的都是**单期**盈利(方向、同比大小),**从未测过持续性**。本节补上。

═══ 口径(事前锁定)═══
  同比    `fundamental_yoy.yoy_series`(§100 已用,泰格四行复现雪球真值)
  streak  截至该次公告,**连续多少期净利同比 > 0**(一期 = 一次财报,四期约一年)
  事件    **财报公告日**(与 §100 B 部分同一事件集)
  分档    **streak = 0 / 1–2 / 3–4 / 5–7 / 8+**(8 期 ≈ 连续两年)
  **判据口径**  **24 个月(500 交易日)峰值 ≥200%** —— 直接对应「涨幅可持续更大」
  描述    6 个月(120 日)/ 12 个月(250 日)峰值 ≥100% 一并报出
  对照A   同日同市值五分位随机 × 200 组
  对照B   同日同市值 **且当日也创 250 日新高** —— 隔离动量后的基准
  退市股 ffill 参与,绝不剔除

═══ 锚点(不过则全节作废;四个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **泰格四行复现雪球真值**(±0.5pp):中报2017 +53.07% / 三季2017 +101.03%
     / 年报2017 +114.01% / 一季2018 +121.07%
  ③ **泰格 streak 恒等复现**:最长连增 **21** 期,出现在 **2022-05-05**(已预验证)
  ④ **恒等零校验**:各档对照的中位命中率 vs 同档总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽;Bonferroni **0.05/10 = 0.005**)═══
本节报 5 档 × 2 对照 = 10 个比较,故门槛 0.005。
  **前置条件**:某档事件数 **< 300** 不判。
  ① **最高档(streak ≥8)对对照A**:24 个月 ≥200% 的 **lift ≥ 1.3 且 p < 0.005**
  ② **同档对对照B**(隔离动量):**lift ≥ 1.3 且 p < 0.005**
  ③ **单调性**:五档 ≥200% 命中率对档位序号的 **Spearman 相关 ≥ +0.8**
     —— 这一条才是直接回答「持续性越强,涨幅越大」

**①②③ 全过 = 业绩持续性是一个独立于动量、且单调的右尾来源 ——
本项目第一个站住的非价格因子。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问**:连增久的股票本来就一直在涨(动量)→ **堵法:判据② 用对照B**;
高 streak 集中在牛市 → **堵法:对照按同日抽**;
5 档搜索出假阳性 → **堵法:判据只压最高档(事前指定)+ Bonferroni 0.05/10**;
**「涨幅更大」被 6 个月口径低估** → **堵法:判据口径改用 24 个月 ≥200%**。

**反问**:分档样本不足 → **前置 n≥300**;
500 日前瞻砍掉 2024-08 之后的事件 → **样本仍在十万量级,且逐档同等受影响**;
同比口径又错 → **锚点② 用雪球真值恒等核对**;
锚点误杀正确实现(已四次病根)→ **四个锚点全是恒等式,②③ 均已预验证可达**。

═══ 事前预测(写下以便被证伪)═══
**①②③ 全不过。**
理由:§100 刚测出**盈利恶化那半边右尾反而更肥**(11.40% vs 9.40%,率差 −2.00%),
且五档呈 **U 形**(两端肥、中间薄)—— **右尾来自波动,而持续稳定高增长的公司
波动恰恰低**。若 streak 也呈 U 形(streak=0 里装着正在崩的公司),判据③ 必不过。
再加上泰格实测 streak 比股价晚 10 个月见顶,持续性是滞后确认量。
**但这是用户提出的假设,且文献里 earnings persistence/quality 有支持。
若 ①②③ 全过,那是本项目第一个站住的非价格因子,我错了 —— 会在正文里明说。**
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
from consolidation_screener import load_panel  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NQ, NSEED, SEED = 5, 200, 20260814
MIN_N = 300
ALPHA = 0.05 / 10
LIFT_MIN, RHO_MIN = 1.3, 0.80
BUCK = [(0, 0, "0"), (1, 2, "1–2"), (3, 4, "3–4"), (5, 7, "5–7"), (8, 99, "8+")]
HOR = [(120, "6个月", 1.0), (250, "12个月", 1.0), (500, "24个月", 2.0)]

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
H2 = pd.DataFrame(Fa).rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(H2) & (Fa >= H2 * 0.9999)
del H2

idxn = idx.tz_localize(None)
pos = {d: i for i, d in enumerate(idxn)}
STREAK = np.full((NT, NS), -1, dtype=np.int16)
ANN = np.zeros((NT, NS), bool)
anc2, anc3 = {}, (0, None)
for j, c in enumerate(codes):
    try:
        d = yoy_series(c)
    except Exception:
        continue
    if d.empty:
        continue
    s = 0
    for _, r in d.iterrows():
        if not np.isfinite(r["同比"]):
            continue
        s = s + 1 if r["同比"] > 0 else 0
        k = pos.get(r["公告日"])
        if k is not None:
            ANN[k, j] = True
            STREAK[k, j] = s
        if c == "300347":
            if (str(r["报告年"]), r["报告期"]) in (
                    ("2017", "中报"), ("2017", "三季报"),
                    ("2017", "年报"), ("2018", "一季报")):
                anc2[(str(r["报告年"]), r["报告期"])] = r["同比"]
            if s > anc3[0]:
                anc3 = (s, r["公告日"])
    if (j + 1) % 1500 == 0:
        print(f"  streak 矩阵 {j+1:,}/{NS:,}  ({time.time()-t0:.0f}s)", flush=True)
TRUE = {("2017", "中报"): .5307, ("2017", "三季报"): 1.0103,
        ("2017", "年报"): 1.1401, ("2018", "一季报"): 1.2107}
a2 = len(anc2) == 4 and all(abs(anc2[k] - v) <= .005 for k, v in TRUE.items())
a3 = anc3[0] == 21 and str(anc3[1].date()) == "2022-05-05"
print(f"  {'✓' if a2 else '✗'} 锚点② 泰格四行复现雪球真值")
print(f"  {'✓' if a3 else '✗'} 锚点③ 泰格最长连增 {anc3[0]} 期 @ {anc3[1].date()}"
      f"(期望 21 @ 2022-05-05)")

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


PK = {n: fwd_peak(n) for n, _, _ in HOR}
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)
EV = [(int(t), int(j)) for t, j in zip(*np.where(ANN), strict=True)]
print(f"财报公告事件 {len(EV):,}")

rng = np.random.default_rng(SEED)


def control(sub, pm, thr, newhi):
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
        v = pm[t, pool] >= thr
        th += float(v.mean()) * k
        tn += k
        hit += v[rng.integers(0, pool.size, size=(NSEED, k))].sum(axis=1)
        tot += k
    if tot == 0:
        return np.array([]), np.nan
    return hit / tot, th / tn


W = 106
rows, MAIN = [], {}
for n, hname, thr in HOR:
    pm = PK[n]
    tag = "  【判据口径】" if n == 500 else "  (描述)"
    print(f"\n{'='*W}\n{hname}峰值 ≥{thr:.0%}{tag}\n{'='*W}")
    print(f"{'连增期数':<8}{'事件数':>9}{'streak中位':>11}{f'≥{thr:.0%}':>9}{'对照A':>9}"
          f"{'liftA':>7}{'pA':>8}{'对照B':>9}{'liftB':>7}{'pB':>8}{'零校验':>8}")
    hits = []
    for lo, hi, nm in BUCK:
        ev = [(t, j) for t, j in EV
              if lo <= STREAK[t, j] <= hi and np.isfinite(pm[t, j])]
        if not ev:
            continue
        v = np.array([pm[t, j] for t, j in ev]) >= thr
        obs = float(v.mean())
        ca, tha = control(ev, pm, thr, False)
        cb, _ = control(ev, pm, thr, True)
        ra = float(np.median(ca)) if ca.size else np.nan
        rb = float(np.median(cb)) if cb.size else np.nan
        r = dict(前瞻=hname, 档=nm, n=len(ev),
                 sk=float(np.median([STREAK[t, j] for t, j in ev])), obs=obs,
                 ctlA=ra, liftA=obs / ra if ra > 0 else np.nan,
                 pA=float((ca >= obs).mean()) if ca.size else np.nan,
                 ctlB=rb, liftB=obs / rb if rb > 0 else np.nan,
                 pB=float((cb >= obs).mean()) if cb.size else np.nan,
                 gap0=abs(ra - tha))
        rows.append(r)
        hits.append(obs)
        print(f"{nm:<8}{len(ev):>9,}{r['sk']:>11.0f}{obs:>9.2%}{ra:>9.2%}"
              f"{r['liftA']:>7.2f}{r['pA']:>8.4f}{rb:>9.2%}{r['liftB']:>7.2f}"
              f"{r['pB']:>8.4f}{r['gap0']:>8.2%}")
        if n == 500 and nm == "8+":
            MAIN = r
    if n == 500 and len(hits) == 5:
        rk = pd.Series(hits).rank()
        rho = float(np.corrcoef(rk, np.arange(1, 6))[0, 1])
        MAIN["rho"] = rho
        MAIN["hits"] = hits
R = pd.DataFrame(rows)

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② 泰格四行")
print(f"  {'✓' if a3 else '✗'} 锚点③ 泰格 streak")
z = R[R["前瞻"] == "24个月"]["gap0"]
a4 = bool(z.notna().all() and (z <= 0.03).all())
print(f"  {'✓' if a4 else '✗'} 锚点④ 恒等零校验 最大差 {z.max():.2%} ≤ 3pp")
for ok, nm in ((a2, "锚点②"), (a3, "锚点③"), (a4, "锚点④")):
    if not ok:
        bad.append(nm)

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,Bonferroni 0.05/10={ALPHA})\n{'='*W}")
c1 = bool(MAIN.get("n", 0) >= MIN_N and MAIN.get("liftA", 0) >= LIFT_MIN
          and MAIN.get("pA", 1) < ALPHA)
c2 = bool(MAIN.get("n", 0) >= MIN_N and MAIN.get("liftB", 0) >= LIFT_MIN
          and MAIN.get("pB", 1) < ALPHA)
rho = MAIN.get("rho", np.nan)
c3 = bool(np.isfinite(rho) and rho >= RHO_MIN)
print(f"  前置:8+ 档事件数 {MAIN.get('n',0):,} ≥ {MIN_N}")
print(f"  {'✓' if c1 else '✗'} ① 8+ 档对照A  lift {MAIN.get('liftA',float('nan')):.2f} ≥1.3 "
      f"且 p {MAIN.get('pA',float('nan')):.4f} < {ALPHA}")
print(f"  {'✓' if c2 else '✗'} ② 8+ 档对照B  lift {MAIN.get('liftB',float('nan')):.2f} ≥1.3 "
      f"且 p {MAIN.get('pB',float('nan')):.4f} < {ALPHA}")
print(f"  {'✓' if c3 else '✗'} ③ 单调性 Spearman {rho:+.2f} ≥ +{RHO_MIN}"
      f"   五档命中率 " + " ".join(f"{h:.2%}" for h in MAIN.get("hits", [])))
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:业绩持续性是独立于动量且单调的右尾来源 —— 本项目第一个站住的非价格因子。**")
    print("  **事前预测被证伪 —— 我错了。**")
elif c3:
    print("  **结论:持续性与右尾单调相关,但最高档对隔离动量的对照不成立。**")
else:
    print("  **结论:业绩持续性不构成「涨幅可持续更大」的来源。事前预测命中。**")

R.to_csv(f"{OUT}/earnings_streak.csv", index=False)
print(f"\n→ {OUT}/earnings_streak.csv   ({time.time()-t0:.0f}s)")
