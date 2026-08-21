"""第八十一节:用极值理论重测右尾 —— 是尾更厚,还是整体更高?(事前登记)

═══ 起因:§67-§80 只用过一种右尾工具 ═══
整个项目从 §67 到 §80 都在研究右尾,却一直用「**数有多少笔超过阈值**」这一种方法。
而 ≥500% 的发生率只有 **0.33%/月** —— 每月每个市值档只有个位数样本,
`lift` 的噪音极大(§80 里 POST_UNLOCK 的 lift 区间宽到 [0.45, 3.17])。

**数阈值只用了「超没超」这 1 bit,把「超了多少」全扔了。**
EVT 的 Peaks-Over-Threshold 用**全部超额幅度**拟合广义帕累托分布(GPD),
同样样本量能榨出多得多的信息。

═══ 它能回答一个 lift 回答不了的问题 ═══
§77 的 AGE_YOUNG lift 1.58,到底是
  **尾更厚**(形状 ξ 更大 → 极端事件结构性地更容易发生),还是
  **整体平移**(尺度 σ 更大 → 只是波动大一点,尾的形状没变)?
**两者的执行含义完全不同,而 lift 分不出来。**

═══ GPD 估计器:自己写,不引入 scipy ═══
README 写明「脚本只依赖 numpy / pandas / pyarrow」,scipy 也确实没装。
用**概率加权矩(PWM)**闭式解,对小样本比 MLE 稳,无需数值优化:

    α_r = E[Y(1-F)^r] = σ / ((r+1)(r+1-ξ))
    R = α0/α1 = 2(2-ξ)/(1-ξ)   →   **ξ = (R-4)/(R-2)**,  σ = α0(1-ξ)

**公式不凭记忆采信** —— 锚点① 用合成数据反推验证,过不了不许碰真数据。

═══ 口径(事前锁定,不搜索) ═══
  结果变量  未来 250 日内最大累计涨幅(与 §77/§79/§80 完全一致)
  POT 阈值  u = **+50%**(在 §77 三门槛 100/200/500% 之下,便于外推到三者)
  取样      逐月末;isfinite(MA300)&isfinite(HI250) 统一掩码(§79/§80 同)
  对照      同月同市值五分位内随机抽同样多只(§77 同一台机器)
  重抽样    **按月分块 bootstrap 200 次** —— 250 日窗口逐月重叠,
            观测不独立,**不能按笔重抽**(§69-§71 反复强调的重叠窗口问题)

═══ 锚点(任一不过则全节作废) ═══
  ① **合成数据**:已知 ξ ∈ {0.1,0.3,0.5},还原误差 < 0.05
     —— 编码正确性锚点,**不过则不许跑真数据**
  ② 面板 (3297, 5232)
  ③ **外推一致性**:u=+50% 拟合的 GPD 外推全市场 P(≥500%),
     与 §77/§80 直接数出来的 **0.33%** 相对误差 < 25%
     —— 对不上说明 GPD 在这条尾上不适用,结论作废

═══ 事前判据(跑之前写死,不放宽) ═══
  ① AGE_YOUNG 的 **ξ − 对照 ξ**,按月分块 bootstrap **95% CI 不含 0**
  ② 若 ① 通过,方向须为**正**(ξ 更大 = 尾更厚)

**判据自查(§79 固化的规则):什么东西会让它通过而不回答我的问题?**
→ 次新股整体波动更大,会同时抬高 σ 与 ξ 的估计。
→ **堵法:σ 与 ξ 必须一起报**;① **只认 ξ,不认 σ** ——
  「整体更高」不算「尾更厚」。

═══ 事前预测(写下以便被证伪) ═══
**① 不通过。** 我预测差异主要在 **σ(尺度)**,不在 **ξ(形状)** ——
次新股是「整体波动更大」,不是「尾结构不同」。
理由:§74 实测次新池退市率仅 0.196%(结构上不出极端左尾);
§79 显示个股状态约 25% 方差来自市场。两条都指向「幅度大」而非「形状异」。

**若 ① 通过且 ξ 显著更大,说明次新股的右尾是结构性的,我错了 ——
那会比 §77 的 lift 1.58 更硬,因为形状参数不受阈值选择影响。**
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
H, NQ, NBOOT, SEED = 250, 5, 200, 20260814
U = 0.50                       # POT 阈值:+50%
GAINS = [1.0, 2.0, 5.0]
Y_LO, Y_HI = 365, 1095
MIN_EXC = 30                   # 单次拟合最少超额样本数

t0 = time.time()


def gpd_pwm(y):
    """PWM 估计 GPD(ξ, σ);y 为超过阈值的超额部分(>0)。"""
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
    """外推 P(X ≥ x) = rate · (1+ξ(x-u)/σ)^(-1/ξ),x > u。"""
    if not np.isfinite(xi) or not np.isfinite(sig) or sig <= 0:
        return np.nan
    z = 1 + xi * (x - U) / sig
    if z <= 0:
        return 0.0
    return float(rate * z ** (-1 / xi))


# ── 锚点①:合成数据(不过则退出) ──────────────────────────────────────────
print("=" * 96)
print("锚点① 合成数据:GPD 估计器能否还原已知参数(不过则不许跑真数据)")
print("=" * 96)
rng0 = np.random.default_rng(0)
worst = 0.0
for xi_t in (0.1, 0.3, 0.5):
    u_ = rng0.random(5000)
    y_ = 1.0 / xi_t * ((1 - u_) ** (-xi_t) - 1)
    xi_h, sig_h = gpd_pwm(y_)
    d = abs(xi_h - xi_t)
    worst = max(worst, d)
    print(f"  {'✓' if d < 0.05 else '✗'} 真 ξ={xi_t:.2f} → 估 {xi_h:.4f}"
          f"  (σ 估 {sig_h:.4f},|Δξ|={d:.4f})")
assert worst < 0.05, f"锚点① 不过(最大误差 {worst:.4f}),禁止继续"
print(f"  → 最大误差 {worst:.4f} < 0.05,**通过**\n")

# ── 载入面板 ─────────────────────────────────────────────────────────────
cl, mvv, ld = {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "float_mv", "listed_days"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mvv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
MV = pd.DataFrame(mvv).set_axis(CL.index)
LD = pd.DataFrame(ld).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点② 对不上 {(NT, NS)}"

CLa, MVa, LDa = CL.to_numpy(float), MV.to_numpy(float), LD.to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
F = pd.DataFrame(CLa).ffill()          # 退市股 ffill,绝不剔除
Fa = F.to_numpy(float)
HI250 = F.rolling(250, min_periods=250).max().to_numpy(float)
MA300 = F.rolling(300, min_periods=300).mean().to_numpy(float)
FMAX = pd.DataFrame(Fa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]
print(f"前瞻最大值完成  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = [p for p in sorted(last_td) if last_td[p] + H < NT]

# ── 逐月收集:信号组 / 对照组 / 全市场 的 250 日峰值 ──────────────────────
rng = np.random.default_rng(SEED)
per_month = []                          # (月, {组: 峰值数组})
for p in months:
    t = last_td[p]
    base = ALIVE[t] & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if base.sum() < 200:
        continue
    ok = base & np.isfinite(MA300[t]) & np.isfinite(HI250[t])
    ratio = np.where(base, FMAX[min(t + 1, NT - 1)] / Fa[t] - 1, np.nan)
    mvt = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(mvt[base], np.linspace(0, 1, NQ + 1)[1:-1])
    bands = []
    for i in range(NQ):
        lo = -np.inf if i == 0 else q[i - 1]
        hi = np.inf if i >= NQ - 1 else q[i]
        bands.append(np.flatnonzero(base & (mvt > lo) & (mvt <= hi)))
    young = np.flatnonzero(ok & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI))
    if len(young) < 20:
        continue
    # 同市值五分位内,抽与信号组同样多只
    ctrl = []
    for b in bands:
        nb = int(np.isin(b, young).sum())
        if nb and len(b) > nb:
            ctrl.append(rng.choice(b, nb, replace=False))
    if not ctrl:
        continue
    ctrl = np.concatenate(ctrl)
    allj = np.flatnonzero(ok)
    per_month.append((p, {
        "AGE_YOUNG 上市[1,3)年": ratio[young][np.isfinite(ratio[young])],
        "同市值随机对照": ratio[ctrl][np.isfinite(ratio[ctrl])],
        "全市场": ratio[allj][np.isfinite(ratio[allj])],
    }))
print(f"逐月收集完成 {len(per_month)} 月  ({time.time()-t0:.0f}s)")

GROUPS = ["AGE_YOUNG 上市[1,3)年", "同市值随机对照", "全市场"]


def fit(months_sub, g):
    """把若干月的峰值合并后拟合 GPD;返回 (ξ, σ, 超阈率, n_exc)。"""
    v = np.concatenate([d[g] for _, d in months_sub]) if months_sub else np.array([])
    if len(v) < 100:
        return np.nan, np.nan, np.nan, 0
    exc = v[v > U] - U
    if len(exc) < MIN_EXC:
        return np.nan, np.nan, np.nan, len(exc)
    xi, sig = gpd_pwm(exc)
    return xi, sig, len(exc) / len(v), len(exc)


print(f"\n{'='*96}\n点估计(全样本合并拟合,POT 阈值 u=+{U:.0%})\n{'='*96}")
print(f"{'组':<24}{'n':>10}{'超阈率':>9}{'n超阈':>9}{'ξ 形状':>10}{'σ 尺度':>10}")
pt = {}
rows = []
for g in GROUPS:
    xi, sig, rate, ne = fit(per_month, g)
    n = sum(len(d[g]) for _, d in per_month)
    pt[g] = (xi, sig, rate)
    print(f"{g:<24}{n:>10,}{rate:>9.2%}{ne:>9,}{xi:>10.4f}{sig:>10.4f}")
    rows.append(dict(部分="点估计", 组=g, n=n, 超阈率=rate, n超阈=ne, xi=xi, sigma=sig))

# ── 锚点③:外推一致性 ────────────────────────────────────────────────────
print(f"\n{'='*96}\n锚点③ 外推一致性:GPD 外推的 P(≥G) vs 直接数出来的\n{'='*96}")
print(f"{'组':<24}{'门槛':>8}{'EVT外推':>10}{'直接数':>10}{'相对误差':>10}")
anc3 = None
for g in GROUPS:
    xi, sig, rate = pt[g]
    v = np.concatenate([d[g] for _, d in per_month])
    for G in GAINS:
        ev = gpd_sf(G, xi, sig, rate)
        emp = float((v >= G).mean())
        rel = abs(ev - emp) / emp if emp > 0 else np.nan
        print(f"{g:<24}{f'≥{G:.0%}':>8}{ev:>10.3%}{emp:>10.3%}{rel:>10.1%}")
        rows.append(dict(部分="外推一致性", 组=g, 门槛=f"≥{G:.0%}",
                         EVT外推=ev, 直接数=emp, 相对误差=rel))
        if g == "全市场" and G == 5.0:
            anc3 = rel

# ── 判据①②:按月分块 bootstrap ──────────────────────────────────────────
print(f"\n{'='*96}\n判据①② 按月分块 bootstrap({NBOOT} 次;250日窗口逐月重叠,不能按笔重抽)\n{'='*96}")
rngb = np.random.default_rng(SEED)
nm = len(per_month)
d_xi, d_sig = [], []
for _ in range(NBOOT):
    pick = [per_month[i] for i in rngb.integers(0, nm, nm)]
    xa, sa, _, _ = fit(pick, "AGE_YOUNG 上市[1,3)年")
    xc, sc, _, _ = fit(pick, "同市值随机对照")
    if np.isfinite(xa) and np.isfinite(xc):
        d_xi.append(xa - xc)
        d_sig.append(sa - sc)
d_xi, d_sig = np.array(d_xi), np.array(d_sig)
lo_x, hi_x = np.percentile(d_xi, [2.5, 97.5])
lo_s, hi_s = np.percentile(d_sig, [2.5, 97.5])
print(f"  Δξ  (次新 − 对照)  中位 {np.median(d_xi):+.4f}   95% CI [{lo_x:+.4f}, {hi_x:+.4f}]")
print(f"  Δσ  (次新 − 对照)  中位 {np.median(d_sig):+.4f}   95% CI [{lo_s:+.4f}, {hi_s:+.4f}]")
rows.append(dict(部分="bootstrap", 组="Δξ", 中位=float(np.median(d_xi)),
                 CI下界=float(lo_x), CI上界=float(hi_x), n次=len(d_xi)))
rows.append(dict(部分="bootstrap", 组="Δσ", 中位=float(np.median(d_sig)),
                 CI下界=float(lo_s), CI上界=float(hi_s), n次=len(d_sig)))

print(f"\n{'='*96}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*96}")
a1 = worst < 0.05
a2 = (NT, NS) == (3297, 5232)
a3 = bool(anc3 is not None and np.isfinite(anc3) and anc3 < 0.25)
c1 = bool(not (lo_x <= 0 <= hi_x))
c2 = bool(c1 and np.median(d_xi) > 0)
print(f"  {'✓' if a1 else '✗'} 锚点① 合成数据还原 ξ,误差 {worst:.4f} < 0.05")
print(f"  {'✓' if a2 else '✗'} 锚点② 面板 (3297, 5232)")
print(f"  {'✓' if a3 else '✗'} 锚点③ 全市场 P(≥500%) 外推相对误差 "
      f"{anc3:.1%} < 25%" if anc3 is not None else "  ✗ 锚点③ 算不出")
print(f"  {'✓' if c1 else '✗'} 判据① Δξ 的 95% CI 不含 0    "
      f"[{lo_x:+.4f}, {hi_x:+.4f}]")
print(f"  {'✓' if c2 else '✗'} 判据② 方向为正(ξ 更大=尾更厚)  中位 {np.median(d_xi):+.4f}")
print()
if not (a1 and a2 and a3):
    print("  **锚点不过:本节结论全部作废。**")
elif c1 and c2:
    print("  **结论:次新股的右尾是结构性的(尾更厚),不只是波动更大。**")
    print("  **事前预测被证伪 —— 我错了。**")
elif c1:
    print("  **结论:ξ 有显著差异但方向为负 —— 尾反而更薄,与 §77 的 lift 需要重新解释。**")
else:
    print("  **结论:ξ 无显著差异 —— 次新股是「整体更高」,不是「尾更厚」。**")
    print("  **事前预测命中。**")

pd.DataFrame(rows).to_csv(f"{SP}/evt_tail_index.csv", index=False)
print(f"\n→ {SP}/evt_tail_index.csv   ({time.time()-t0:.0f}s)")
