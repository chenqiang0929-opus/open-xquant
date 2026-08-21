"""第八十三节:提高中位数是不是必然削右尾 —— 系统检验本项目的中心断言(事前登记)

═══ 起因:一条贯穿十几节的断言,从没被系统检验过 ═══
§62 写的是:**「所有提高胜率的过滤器都在削右尾」**。
这句话是 §62-§78 整个解释框架的地基 —— §70 用它解释四条主动规则全灭,
§77 用它解释 B 部分 24 格全灭,§78 用它解释离场规则为什么削掉右尾。

**但它是从一个例子归纳出来的**:跨段组合把胜率提到全研究最高的 25.37%,
组合年化却从 +10.37% 变成 −0.30%,因为它把 80 个最大赢家里的 60 个筛掉了。
**一个例子,推出一条普遍规律,然后当地基用了十几节。**

═══ 用分位曲线直接测 ═══
对每个信号同时测两件事:它把**中位数**推了多少、把 **99% 分位**推了多少。
**若 §62 成立,这两个效应应当系统性反号。**

**命名要老实**:本节的预测变量全是二值的,所以做的是
**分组条件分位比较**,不是带连续自变量的分位数回归。
**不套用听起来更高级的名字。**

═══ 口径(全部沿用 §77-§82,不重调) ═══
  结果变量  未来 250 日内最大累计涨幅
  配对      同月**同市值五分位内**,信号组 vs 同档随机同样多只
  分位      τ ∈ {0.25, 0.50, 0.75, 0.90, 0.95, 0.99}
  掩码      isfinite(MA300)&isfinite(HI250) 统一掩码
  重抽样    **按月分块 bootstrap 200 次**(250 日窗口逐月重叠,不能按笔重抽)

═══ 锚点(不过则全节作废) ═══
  ① 面板 (3297, 5232)
  ② 零锚点 SMALL_MV 在六个 τ 上 **|Δ| 均 < 0.03**
     —— 对照本就按市值档抽,最小档在自己档内即全体,Δ 必然 ≈0

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 跨信号,**Δτ=0.50 与 Δτ=0.99 的 Spearman 秩相关 < 0**,
     且按月分块 bootstrap 的 **95% CI 不含 0**

  **前置条件(写死,不是事后加的)**:Δτ=0.50 的跨信号极差必须 **> 0.05**。
  若所有信号对中位数的影响都差不多,秩相关由噪音决定,**判据①不予采信**。

═══ 判据自查(§79 固化的规则) ═══
**「什么东西会让它通过,而不回答我的问题?」**
→ 若信号全都同向(都正或都负),秩相关就是在给噪音排序。
→ **堵法**:上面那条前置条件;并**同时报出「Δ中位>0 且 Δ99分位<0」的信号个数**
  —— 这个计数才是 §62 断言的字面含义,秩相关只是它的连续版本。

═══ 事前预测(写下以便被证伪) ═══
**① 不通过,秩相关接近 0。**
理由:§82 刚测出换手率**同时**抬高超阈率与 σ(不是 trade-off);
§81 测出次新股超阈率、ξ、σ 三个参数同向。
**「提高胜率必削右尾」很可能是从跨段组合那一个例子过度归纳的。**

**若 ① 通过,§62 的框架得到系统支持,我错了 ——
那反而是好消息:这个项目十几节的解释地基就被坐实了。**
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
TAUS = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
Y_LO, Y_HI = 365, 1095
MIN_SIG, MIN_BAND = 10, 3
SPREAD_MIN = 0.05

t0 = time.time()
cl, mvv, tov, ld = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "float_mv", "turnover", "listed_days"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mvv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    tov[k] = pd.to_numeric(x["turnover"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
MV = pd.DataFrame(mvv).set_axis(CL.index)
TO = pd.DataFrame(tov).set_axis(CL.index)
LD = pd.DataFrame(ld).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

CLa, MVa, LDa = CL.to_numpy(float), MV.to_numpy(float), LD.to_numpy(float)
TOa = TO.rolling(20, min_periods=20).mean().to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
F = pd.DataFrame(CLa).ffill()
Fa = F.to_numpy(float)
HI250 = F.rolling(250, min_periods=250).max().to_numpy(float)
MA100 = F.rolling(100, min_periods=100).mean().to_numpy(float)
MA300 = F.rolling(300, min_periods=300).mean().to_numpy(float)
RET50 = Fa / F.shift(50).to_numpy(float) - 1
RET250 = Fa / F.shift(250).to_numpy(float) - 1
FMAX = pd.DataFrame(Fa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]
print(f"派生量完成  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = [p for p in sorted(last_td) if last_td[p] + H < NT]


def pct(arr, t, base):
    v = np.where(base, arr[t], np.nan)
    return pd.Series(v).rank(pct=True).to_numpy(float) * 100


NAMES = ["AGE_YOUNG 上市[1,3)年", "AGE_OLD 上市>10年", "SMALL_MV 市值最小档(零锚点)",
         "RPS250>=90", "RPS50>=95", "RPS50<=30 弱势",
         "MA_BULL 已多头排列", "MA_BEAR 空头排列", "NEAR_HIGH 距新高<=10%",
         "FAR_HIGH 距新高>30%", "TO_Q5 换手最高档", "TO_Q1 换手最低档",
         "ABOVE_MA100 站上20周线"]

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
    # band 从 base 建、信号用 ok(带 MA300/HI250 掩码)—— 与 §79/§80 一致。
    # 两边都用 ok 会让 SMALL_MV 恰等于 band 0,五个档被 len(b)<=len(si) 全跳过,
    # 零锚点算不出来。§79 首轮栽过一次,这里我又写回来了一次。
    mvt = np.where(base, MVa[t], np.nan)
    qm = np.nanquantile(mvt[base], np.linspace(0, 1, NQ + 1)[1:-1])
    bands = []
    for i in range(NQ):
        lo = -np.inf if i == 0 else qm[i - 1]
        hi = np.inf if i >= NQ - 1 else qm[i]
        bands.append(np.flatnonzero(base & (mvt > lo) & (mvt <= hi)))
    r50, r250 = pct(RET50, t, ok), pct(RET250, t, ok)
    tq = pct(TOa, t, ok)
    q20mv = np.nanquantile(mvt[base], 0.2)
    S = {
        "AGE_YOUNG 上市[1,3)年": ok & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI),
        "AGE_OLD 上市>10年": ok & (LDa[t] >= 3650),
        "SMALL_MV 市值最小档(零锚点)": ok & (mvt <= q20mv),
        "RPS250>=90": ok & (r250 >= 90),
        "RPS50>=95": ok & (r50 >= 95),
        "RPS50<=30 弱势": ok & (r50 <= 30),
        "MA_BULL 已多头排列": ok & (MA100[t] > MA300[t]),
        "MA_BEAR 空头排列": ok & (MA100[t] <= MA300[t]),
        "NEAR_HIGH 距新高<=10%": ok & (Fa[t] >= HI250[t] * 0.90),
        "FAR_HIGH 距新高>30%": ok & (Fa[t] < HI250[t] * 0.70),
        "TO_Q5 换手最高档": ok & (tq >= 80),
        "TO_Q1 换手最低档": ok & (tq <= 20),
        "ABOVE_MA100 站上20周线": ok & (Fa[t] > MA100[t]),
    }
    rng0 = np.random.default_rng(SEED + hash(str(p)) % 99991)
    d = {}
    for nm in NAMES:
        sel = S[nm]
        if sel.sum() < MIN_SIG:
            continue
        sig, ctl = [], []
        for b in bands:
            si = b[sel[b]]
            if len(si) < MIN_BAND or len(b) <= len(si):
                continue
            sig.append(ratio[si])
            ctl.append(ratio[rng0.choice(b, len(si), replace=False)])
        if not sig:
            continue
        a = np.concatenate(sig)
        c = np.concatenate(ctl)
        a, c = a[np.isfinite(a)], c[np.isfinite(c)]
        if len(a) >= 20 and len(c) >= 20:
            d[nm] = (a, c)
    if d:
        per_month.append((p, d))
print(f"逐月完成 {len(per_month)} 月  ({time.time()-t0:.0f}s)")


def delta(ms, nm):
    """返回各 τ 上「信号分位 − 对照分位」。"""
    a = [x[nm][0] for _, x in ms if nm in x]
    c = [x[nm][1] for _, x in ms if nm in x]
    if len(a) < 12:
        return None
    va, vc = np.concatenate(a), np.concatenate(c)
    return np.array([np.quantile(va, q) - np.quantile(vc, q) for q in TAUS])


print(f"\n{'='*112}\n分位曲线:信号组 − 同市值档随机对照(250 日峰值涨幅)\n{'='*112}")
print(f"{'信号':<28}" + "".join(f"{f'τ={q:.2f}':>12}" for q in TAUS))
pt, rows = {}, []
for nm in NAMES:
    dv = delta(per_month, nm)
    if dv is None:
        continue
    pt[nm] = dv
    print(f"{nm:<28}" + "".join(f"{v:>+12.1%}" for v in dv))
    rows.append(dict(部分="分位曲线", 信号=nm,
                     **{f"tau_{q:.2f}": float(v) for q, v in zip(TAUS, dv)}))

i50, i99 = TAUS.index(0.50), TAUS.index(0.99)
core = [n for n in pt if "零锚点" not in n]


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


rho = spearman([pt[n][i50] for n in core], [pt[n][i99] for n in core])
cnt = sum(pt[n][i50] > 0 and pt[n][i99] < 0 for n in core)
spread = max(pt[n][i50] for n in core) - min(pt[n][i50] for n in core)

print(f"\n{'='*112}\n判据① 按月分块 bootstrap({NBOOT} 次)\n{'='*112}")
rngb = np.random.default_rng(SEED)
nm_ = len(per_month)
rhos = []
for _ in range(NBOOT):
    pick = [per_month[i] for i in rngb.integers(0, nm_, nm_)]
    dd = {n: delta(pick, n) for n in core}
    dd = {n: v for n, v in dd.items() if v is not None}
    if len(dd) >= 6:
        rhos.append(spearman([v[i50] for v in dd.values()],
                             [v[i99] for v in dd.values()]))
rhos = np.array(rhos)
lo, hi = np.percentile(rhos, [2.5, 97.5])
print(f"  Spearman ρ(Δτ=0.50, Δτ=0.99)  点估计 **{rho:+.4f}**")
print(f"  bootstrap 中位 {np.median(rhos):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  「Δ中位>0 且 Δ99分位<0」的信号数:**{cnt}/{len(core)}**")
print(f"  Δτ=0.50 跨信号极差:{spread:.3f}(前置条件门槛 >{SPREAD_MIN})")
rows.append(dict(部分="判据", 信号="Spearman", rho=rho, 中位=float(np.median(rhos)),
                 CI下界=float(lo), CI上界=float(hi), 反号信号数=cnt,
                 信号总数=len(core), 中位极差=float(spread)))

print(f"\n{'='*112}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*112}")
zero = pt.get("SMALL_MV 市值最小档(零锚点)")
a2 = zero is not None and bool(np.all(np.abs(zero) < 0.03))
pre = spread > SPREAD_MIN
c1 = bool(rho < 0 and not (lo <= 0 <= hi))
print("  ✓ 锚点① 面板 (3297, 5232)")
if zero is None:
    print("  ✗ 锚点② 零锚点算不出 —— 不通过")
else:
    print(f"  {'✓' if a2 else '✗'} 锚点② 零锚点六个 τ 上 |Δ| 均 <0.03   "
          f"最大 {np.abs(zero).max():.4f}")
print(f"  {'✓' if pre else '✗'} 前置条件 Δτ=0.50 跨信号极差 >{SPREAD_MIN}   {spread:.3f}")
print(f"  {'✓' if c1 else '✗'} 判据① ρ<0 且 95% CI 不含 0   "
      f"ρ={rho:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
print()
if not a2:
    print("  **锚点②不过:本节结论作废。**")
elif not pre:
    print("  **前置条件不过:判据①不予采信(事前写死,非事后加)。**")
elif c1:
    print("  **结论:§62「提高胜率必削右尾」得到系统支持。事前预测被证伪 —— 我错了。**")
else:
    print("  **结论:中位数效应与右尾效应之间没有系统性反号关系。**")
    print("  **§62 那条断言是从一个例子过度归纳的,不能当普遍规律用。事前预测命中。**")

pd.DataFrame(rows).to_csv(f"{SP}/quantile_curve.csv", index=False)
print(f"\n→ {SP}/quantile_curve.csv   ({time.time()-t0:.0f}s)")
