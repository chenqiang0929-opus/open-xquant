"""第八十二节:换手率是怎么作用于右尾的 —— 用 EVT 拆开(事前登记)

═══ 起因:§62 留下一个只能由右尾解释的组合 ═══
§62 实测(突破池内,三条全中的 1,606 笔再切两半):

    A  三条全中 且 **非**涨停/高换手   670 笔   胜率 **25.37%**   年化 **−0.30%**
    对照 三条全中 且 **是**涨停/高换手  936 笔   胜率 17.20%      年化 **+8.72%**

**高换手那批胜率更低、钱却更多。低胜率 + 高收益,只能由右尾解释。**
§62 同时测出「把股票推上 RPS60>90 的,就是换手率和涨停 —— 没有别的东西」
(换手分位 lift 1.27),以及右尾 top-5% 的换手分位 0.725 vs 其余 0.609。

**但从没人拆开看它是怎么作用于右尾的。** §81 刚把 EVT 验通(GPD 外推
九格相对误差 0.4%~8.0%),正好用来做这件事。

═══ 要回答的问题 ═══
换手率抬高右尾,是靠
  **更容易起涨**(超阈率高 → 但那会同时抬高胜率,与 §62 矛盾),还是
  **起涨后走得更远**(ξ / σ 更大 → 胜率不变甚至更低,而远端概率更高)?

**执行含义完全不同**:前者是入场信号,后者是「进去之后别急着卖」。

═══ 口径(事前锁定,不搜索) ═══
  换手指标  **过去 20 日平均 turnover** 的当月末横截面分位
  分档      **在每个市值五分位内部**再分 5 档,然后把跨市值档的同一换手档合并
            —— 小盘天然高换手,**不做这层控制就是在测市值**(§77/§79/§80 一贯要求)
  结果变量  未来 250 日内最大累计涨幅(与 §77/§79/§80/§81 完全一致)
  POT 阈值  u = **+50%**(与 §81 同,不重调)
  掩码      isfinite(MA300)&isfinite(HI250) 统一掩码(§79/§80/§81 同)
  重抽样    **按月分块 bootstrap 200 次**(250 日窗口逐月重叠,不能按笔重抽)

═══ 锚点(任一不过则全节作废) ═══
  ① 合成数据:GPD 估计器还原已知 ξ,误差 < 0.05(与 §81 同一套,不过则不许跑真数据)
  ② 面板 (3297, 5232)
  ③ 外推一致性:各档 GPD 外推 P(≥500%) 对直接数出来的相对误差 **< 25%**

═══ 事前判据(跑之前写死,不放宽) ═══
  ① **主判据**:最高换手档 Q5 − 最低档 Q1 的 **Δξ**,
     按月分块 bootstrap **95% CI 不含 0 且中位为正**
  ② **诊断(非判据,但必须报出)**:超阈率随换手档的方向、
     Δσ、以及各档外推的 P(≥500%)

═══ 判据自查(§79 固化的规则) ═══
**「什么东西会让它通过,而不回答我的问题?」**
→ 换手率与市值强相关,若不控市值,Q5 会挤满小盘股,
  测到的是市值效应不是换手效应。
→ **堵法:分档在市值五分位内部做**,Q1~Q5 的市值分布按构造平衡。
→ 另一条:ξ 的估计对最右端少数几笔敏感,单档样本若太少会虚高。
  **堵法:报出每档的 n超阈,少于 500 的档不参与判据。**

═══ 事前预测(写下以便被证伪) ═══
**① 通过。ξ 随换手档递增,Q5 显著高于 Q1。**
**这是本研究 24 次事前检验里,我第一次预测「通过」。**
理由不是直觉,是 §62 已经把答案逼到墙角:
高换手组**胜率更低而年化更高**,这个组合在数学上只能由右尾更厚解释;
而 §81 已证 GPD 在这条尾上成立,若差异真实存在,ξ 就应该测得出来。

**同时预测 ②:超阈率随换手档递减**(与 §62 的低胜率一致)。

**若 ① 不通过,说明 §62 那个「低胜率高收益」的组合另有解释,
不是尾更厚 —— 那我这次的高调预测就被证伪,必须在正文明说。**
"""
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
np.seterr(invalid="ignore", divide="ignore")

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
H, NQ, NTQ, NBOOT, SEED = 250, 5, 5, 200, 20260814
U, TO_WIN = 0.50, 20
GAINS = [1.0, 2.0, 5.0]
MIN_EXC, MIN_EXC_CRIT = 30, 500

t0 = time.time()


def gpd_pwm(y):
    """PWM 估计 GPD(ξ, σ)。α_r=σ/((r+1)(r+1-ξ)) → ξ=(R-4)/(R-2), σ=α0(1-ξ)。"""
    y = np.sort(np.asarray(y, float))
    n = len(y)
    if n < MIN_EXC:
        return np.nan, np.nan
    p = (np.arange(1, n + 1) - 0.35) / n
    a0 = float(y.mean())
    a1 = float((y * (1 - p)).mean())
    if a1 <= 0 or abs(a0 - 2 * a1) < 1e-12:
        return np.nan, np.nan
    r = a0 / a1
    if abs(r - 2) < 1e-9:
        return np.nan, np.nan
    xi = (r - 4) / (r - 2)
    return float(xi), float(a0 * (1 - xi))


def gpd_sf(x, xi, sig, rate):
    if not np.isfinite(xi) or not np.isfinite(sig) or sig <= 0:
        return np.nan
    z = 1 + xi * (x - U) / sig
    return 0.0 if z <= 0 else float(rate * z ** (-1 / xi))


print("=" * 96)
print("锚点① 合成数据:GPD 估计器能否还原已知参数(不过则不许跑真数据)")
print("=" * 96)
rng0 = np.random.default_rng(0)
worst = 0.0
for xi_t in (0.1, 0.3, 0.5):
    u_ = rng0.random(5000)
    xi_h, _ = gpd_pwm(1.0 / xi_t * ((1 - u_) ** (-xi_t) - 1))
    d = abs(xi_h - xi_t)
    worst = max(worst, d)
    print(f"  {'✓' if d < 0.05 else '✗'} 真 ξ={xi_t:.2f} → 估 {xi_h:.4f}  |Δξ|={d:.4f}")
assert worst < 0.05, f"锚点① 不过({worst:.4f}),禁止继续"
print(f"  → 最大误差 {worst:.4f} < 0.05,**通过**\n")

cl, mvv, tov = {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "float_mv", "turnover"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mvv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    tov[k] = pd.to_numeric(x["turnover"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
MV = pd.DataFrame(mvv).set_axis(CL.index)
TO = pd.DataFrame(tov).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点② 对不上 {(NT, NS)}"

CLa, MVa = CL.to_numpy(float), MV.to_numpy(float)
TOa = TO.rolling(TO_WIN, min_periods=TO_WIN).mean().to_numpy(float)   # 过去20日均换手
ALIVE = np.isfinite(CLa) & (CLa > 0)
F = pd.DataFrame(CLa).ffill()
Fa = F.to_numpy(float)
HI250 = F.rolling(250, min_periods=250).max().to_numpy(float)
MA300 = F.rolling(300, min_periods=300).mean().to_numpy(float)
FMAX = pd.DataFrame(Fa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]
print(f"换手/前瞻完成  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = [p for p in sorted(last_td) if last_td[p] + H < NT]

TAGS = [f"Q{i+1}" for i in range(NTQ)]
per_month = []
for p in months:
    t = last_td[p]
    base = ALIVE[t] & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if base.sum() < 200:
        continue
    ok = base & np.isfinite(MA300[t]) & np.isfinite(HI250[t]) & np.isfinite(TOa[t])
    if ok.sum() < 200:
        continue
    ratio = np.where(base, FMAX[min(t + 1, NT - 1)] / Fa[t] - 1, np.nan)
    mvt = np.where(ok, MVa[t], np.nan)
    qm = np.nanquantile(mvt[ok], np.linspace(0, 1, NQ + 1)[1:-1])
    buckets = {g: [] for g in TAGS}
    for i in range(NQ):                       # 市值五分位内部再按换手分 5 档
        lo = -np.inf if i == 0 else qm[i - 1]
        hi = np.inf if i >= NQ - 1 else qm[i]
        band = np.flatnonzero(ok & (mvt > lo) & (mvt <= hi))
        if len(band) < NTQ * 5:
            continue
        tv = TOa[t][band]
        qt = np.nanquantile(tv, np.linspace(0, 1, NTQ + 1)[1:-1])
        for k in range(NTQ):
            a = -np.inf if k == 0 else qt[k - 1]
            b = np.inf if k >= NTQ - 1 else qt[k]
            sel = band[(tv > a) & (tv <= b)]
            if len(sel):
                buckets[TAGS[k]].append(ratio[sel][np.isfinite(ratio[sel])])
    d = {g: (np.concatenate(v) if v else np.array([])) for g, v in buckets.items()}
    if min(len(v) for v in d.values()) < 20:
        continue
    per_month.append((p, d))
print(f"逐月分档完成 {len(per_month)} 月  ({time.time()-t0:.0f}s)")


def fit(ms, g):
    v = np.concatenate([d[g] for _, d in ms]) if ms else np.array([])
    if len(v) < 100:
        return np.nan, np.nan, np.nan, 0, v
    exc = v[v > U] - U
    if len(exc) < MIN_EXC:
        return np.nan, np.nan, np.nan, len(exc), v
    xi, sig = gpd_pwm(exc)
    return xi, sig, len(exc) / len(v), len(exc), v


print(f"\n{'='*96}\n点估计:换手档(市值五分位内部分档,u=+{U:.0%})\n{'='*96}")
print(f"{'换手档':<8}{'n':>10}{'超阈率':>9}{'n超阈':>9}{'ξ 形状':>10}{'σ 尺度':>10}"
      f"{'EVT P(≥500%)':>14}{'直接数':>10}{'相对误差':>10}")
pt, rows, anc3 = {}, [], []
for g in TAGS:
    xi, sig, rate, ne, v = fit(per_month, g)
    ev = gpd_sf(5.0, xi, sig, rate)
    emp = float((v >= 5.0).mean()) if len(v) else np.nan
    rel = abs(ev - emp) / emp if emp and emp > 0 else np.nan
    pt[g] = (xi, sig, rate, ne)
    anc3.append(rel)
    print(f"{g:<8}{len(v):>10,}{rate:>9.2%}{ne:>9,}{xi:>10.4f}{sig:>10.4f}"
          f"{ev:>14.3%}{emp:>10.3%}{rel:>10.1%}")
    rows.append(dict(部分="点估计", 换手档=g, n=len(v), 超阈率=rate, n超阈=ne,
                     xi=xi, sigma=sig, EVT_P500=ev, 直接数_P500=emp, 相对误差=rel))

print(f"\n{'='*96}\n判据① 按月分块 bootstrap({NBOOT} 次):Q5 − Q1\n{'='*96}")
rngb = np.random.default_rng(SEED)
nm = len(per_month)
dx, ds, dr = [], [], []
for _ in range(NBOOT):
    pick = [per_month[i] for i in rngb.integers(0, nm, nm)]
    x5, s5, r5, _, _ = fit(pick, "Q5")
    x1, s1, r1, _, _ = fit(pick, "Q1")
    if np.isfinite(x5) and np.isfinite(x1):
        dx.append(x5 - x1)
        ds.append(s5 - s1)
        dr.append(r5 - r1)
dx, ds, dr = np.array(dx), np.array(ds), np.array(dr)
lox, hix = np.percentile(dx, [2.5, 97.5])
los, his = np.percentile(ds, [2.5, 97.5])
lor, hir = np.percentile(dr, [2.5, 97.5])
print(f"  Δξ    中位 {np.median(dx):+.4f}   95% CI [{lox:+.4f}, {hix:+.4f}]")
print(f"  Δσ    中位 {np.median(ds):+.4f}   95% CI [{los:+.4f}, {his:+.4f}]")
print(f"  Δ超阈率 中位 {np.median(dr):+.4f}   95% CI [{lor:+.4f}, {hir:+.4f}]")
for nmx, a, lo, hi in (("Δξ", dx, lox, hix), ("Δσ", ds, los, his),
                       ("Δ超阈率", dr, lor, hir)):
    rows.append(dict(部分="bootstrap", 换手档=nmx, 中位=float(np.median(a)),
                     CI下界=float(lo), CI上界=float(hi), n次=len(a)))

print(f"\n{'='*96}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*96}")
a3ok = all(np.isfinite(r) and r < 0.25 for r in anc3)
nok = all(pt[g][3] >= MIN_EXC_CRIT for g in ("Q1", "Q5"))
c1 = bool(not (lox <= 0 <= hix) and np.median(dx) > 0)
print(f"  ✓ 锚点① 合成数据还原 ξ,误差 {worst:.4f} < 0.05")
print("  ✓ 锚点② 面板 (3297, 5232)")
print(f"  {'✓' if a3ok else '✗'} 锚点③ 各档 P(≥500%) 外推相对误差 <25%   "
      f"最大 {np.nanmax(anc3):.1%}")
print(f"  {'✓' if nok else '✗'} 样本量 Q1/Q5 的 n超阈 ≥{MIN_EXC_CRIT}   "
      f"{pt['Q1'][3]:,} / {pt['Q5'][3]:,}")
print(f"  {'✓' if c1 else '✗'} 判据① Δξ(Q5−Q1)95% CI 不含 0 且中位为正   "
      f"[{lox:+.4f}, {hix:+.4f}]")
print()
if not (a3ok and nok):
    print("  **锚点/样本量不过:本节结论作废。**")
elif c1:
    print("  **结论:换手率通过加厚右尾起作用,不是通过提高起涨概率。**")
    print("  **事前预测命中(本研究第一次预测「通过」并命中)。**")
else:
    print("  **结论:ξ 无显著差异 —— 换手率不是靠加厚尾起作用的。**")
    print("  **我第一次高调预测「通过」,被证伪。**")

pd.DataFrame(rows).to_csv(f"{SP}/turnover_tail_evt.csv", index=False)
print(f"\n→ {SP}/turnover_tail_evt.csv   ({time.time()-t0:.0f}s)")
