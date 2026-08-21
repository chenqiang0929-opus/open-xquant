"""第八十五节:重做分位曲线 —— 用设计正确的零锚点验本项目的中心断言(事前登记)

═══ 这是新的事前登记,不是对 §83 的放宽 ═══
**§83 在记录里保持作废,其脚本与正文一字不改。**

§83 要验的是 §62 的中心断言「**所有提高胜率的过滤器都在削右尾**」——
这条当了 §62~§78 十几节的解释地基,却从没被系统验过。
**但 §83 的零锚点被我写坏了,整节作废(第 25 次判据检验,未放宽)。**

两个设计错误:
① **绝对阈值套在尺度差 10~50 倍的量上** ——
   τ=0.25 处收益量级 0~20%,τ=0.99 处 300~800%,固定 3pp 门槛在两端
   不是同一个严格程度。
② **更根本:SMALL_MV 在「收益差」空间里本来就不是零。**
   §77 测它用 lift(比值),1.05/1.03/0.91 ≈ 1 所以叫零锚点;
   但在绝对差值空间,最小市值档**内部**仍有真实规模梯度(实测 +1.5~+4.8pp)。
   **我把比值空间的零锚点搬到了差值空间。**

═══ 修法:零锚点换成置换零假设 ═══
**零锚点从 SMALL_MV 换成 RANDOM_LABEL** —— 在每个市值五分位内部随机贴上
与真信号**同样多**的「假信号」标签,再算同一条 Δ 分位曲线。
**它按构造期望为零**,不像 SMALL_MV 自带规模梯度。

**这同时解决尺度问题**:每个 τ 由置换分布自己给出噪音带,
不需要跨 τ 通用的绝对阈值。

**主判据据此标准化**:每个信号在每个 τ 上的效应换算成
**相对该 τ 置换零分布的 z 值**,秩相关在 z 值上算 ——
原始百分点在 τ=0.99 处天然比 τ=0.50 大一个量级,直接算相关是**拿尺度当信号**。

═══ 口径(其余沿用 §77-§84,不重调) ═══
  结果变量 未来 250 日内最大累计涨幅
  配对     同月同市值五分位内;**band 从 base 建、信号用 ok**(§79/§80 的正确写法)
  τ        {0.25, 0.50, 0.75, 0.90, 0.95, 0.99}
  置换     每月每信号 NPERM 次随机贴标签,给出每 τ 的零分布
  重抽样   按月分块 bootstrap(250 日窗口逐月重叠,不能按笔重抽)

═══ 锚点(不过则全节作废) ═══
  ① 面板 (3297, 5232)
  ② **RANDOM_LABEL 的 Δ 在六个 τ 上 |z| 均 < 2** —— 真零锚点应落在自己的零分布里
  ③ **AGE_YOUNG 在 τ=0.99 上 z > 2** —— **正向锚点,设为作废条件**。
     §77/§80 两次测出它 ≥500% lift 1.58/1.60、p=0.0000;
     若分位口径连这个已知效应都感知不到,本节的「测不出」就没有意义。
     > 这是 §77「正向锚点设计失败」那次的补课 —— 那次正向锚点只作诊断打印。

═══ 事前判据(跑之前写死,不放宽) ═══
  ① **连续版**:跨信号,z(Δτ=0.50) 与 z(Δτ=0.99) 的 **Spearman 秩相关 < 0**,
     按月分块 bootstrap **95% CI 不含 0**
  ② **字面版**:「z(Δτ=0.50) > 2 且 z(Δτ=0.99) < −2」的信号数 **≥ 3**

**①② 是 §62 断言的连续版与字面版,分开判、都要报。**

═══ 判据自查(§79 规则 + §83 补上的对称一问) ═══
**正问:什么会让它「通过」而不回答我的问题?**
→ 若信号全同向,秩相关只是给噪音排序。**堵法**:②的计数版独立于秩相关。

**反问(§83 正是栽在这里,本次补上):什么会让它「不通过」而与问题无关?**
→ 尺度不可比        → **全部在 z 值空间比较**
→ 零锚点自带真实效应 → **换成置换零假设**
→ 口径感知不到已知效应 → **锚点③ 正向锚点设为作废条件**

═══ 一处必须事前声明的近似 ═══
按月分块 bootstrap **只重算 Δ,置换零分布的 mu/sd 沿用全样本那一版** ——
每个 bootstrap 轮内重跑 200 次置换等于 4 万倍工作量,不可行。
**这假定零分布的位置与尺度对月集重抽样不敏感。** 写在这里,不在结果出来后补。

═══ 事前预测(写下以便被证伪) ═══
**①② 都不通过。** 秩相关接近 0,计数版 ≤2。
理由:§81 测出次新股超阈率/ξ/σ 三参数同向;§82 测出换手率同时抬高
超阈率与 σ(不是 trade-off);§83 作废前的观测里 ρ=+0.08、计数 2/12。
**「提高胜率必削右尾」很可能是从跨段组合那一个例子过度归纳的。**

**若 ①② 通过,§62 的框架被系统坐实,我错了 —— 那对这个项目是好消息,
因为十几节的解释地基就不再只靠一个例子撑着。**
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
H, NQ, NPERM, NBOOT, SEED = 250, 5, 200, 200, 20260814
TAUS = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
Y_LO, Y_HI = 365, 1095
MIN_SIG, MIN_BAND = 10, 3

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


def pctl(arr, t, base):
    return pd.Series(np.where(base, arr[t], np.nan)).rank(pct=True).to_numpy(float) * 100


NAMES = ["AGE_YOUNG 上市[1,3)年", "AGE_OLD 上市>10年", "RPS250>=90", "RPS50>=95",
         "RPS50<=30 弱势", "MA_BULL 已多头排列", "MA_BEAR 空头排列",
         "NEAR_HIGH 距新高<=10%", "FAR_HIGH 距新高>30%",
         "TO_Q5 换手最高档", "TO_Q1 换手最低档", "ABOVE_MA100 站上20周线"]
ZERO = "RANDOM_LABEL 随机贴标签(零锚点)"

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
    # band 从 base 建、信号用 ok —— §79/§80 的正确写法(§83 在这里写反过)
    mvt = np.where(base, MVa[t], np.nan)
    qm = np.nanquantile(mvt[base], np.linspace(0, 1, NQ + 1)[1:-1])
    bands = []
    for i in range(NQ):
        lo = -np.inf if i == 0 else qm[i - 1]
        hi = np.inf if i >= NQ - 1 else qm[i]
        bands.append(np.flatnonzero(base & (mvt > lo) & (mvt <= hi)))
    r50, r250, tq = pctl(RET50, t, ok), pctl(RET250, t, ok), pctl(TOa, t, ok)
    S = {
        "AGE_YOUNG 上市[1,3)年": ok & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI),
        "AGE_OLD 上市>10年": ok & (LDa[t] >= 3650),
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
    # 零锚点:随机贴标签,只数与 AGE_YOUNG 同 —— 按构造期望为零
    rngz = np.random.default_rng(SEED + hash(str(p)) % 99991)
    zsel = np.zeros(NS, bool)
    for b in bands:
        nb = int(S["AGE_YOUNG 上市[1,3)年"][b].sum())
        if nb and len(b) > nb:
            zsel[rngz.choice(b, nb, replace=False)] = True
    S[ZERO] = zsel
    d = {}
    for nm, sel in S.items():
        if sel.sum() < MIN_SIG:
            continue
        sig, pool = [], []
        for b in bands:
            si = b[sel[b]]
            if len(si) < MIN_BAND or len(b) <= len(si):
                continue
            sig.append(ratio[si])
            pool.append((b, len(si)))
        if not sig:
            continue
        va = np.concatenate(sig)
        va = va[np.isfinite(va)]
        if len(va) >= 20:
            d[nm] = (va, pool)
    if d:
        per_month.append((p, d, ratio))
print(f"逐月完成 {len(per_month)} 月  ({time.time()-t0:.0f}s)")

ALL = NAMES + [ZERO]


def curves(ms):
    """返回 {信号: (Δ实测[6], z[6])};Δ 与置换零分布均在同一批月上算。"""
    out = {}
    rngp = np.random.default_rng(SEED)
    for nm in ALL:
        sub = [(d[nm], ratio) for _, d, ratio in ms if nm in d]
        if len(sub) < 12:
            continue
        obs = np.concatenate([s[0][0] for s in sub])
        q_obs = np.array([np.quantile(obs, q) for q in TAUS])
        perm = np.zeros((NPERM, len(TAUS)))
        for k in range(NPERM):
            vals = []
            for (va_pool, ratio) in [(s[0][1], s[1]) for s in sub]:
                for b, n in va_pool:
                    v = ratio[rngp.choice(b, n, replace=False)]
                    vals.append(v[np.isfinite(v)])
            pv = np.concatenate(vals)
            perm[k] = [np.quantile(pv, q) for q in TAUS]
        mu, sd = perm.mean(axis=0), perm.std(axis=0)
        delta = q_obs - mu
        z = np.where(sd > 0, delta / sd, np.nan)
        out[nm] = (delta, z, mu, sd)
    return out


res = curves(per_month)
print(f"置换零分布完成({NPERM} 次/信号)  ({time.time()-t0:.0f}s)")

print(f"\n{'='*120}\nΔ 分位曲线(信号 − 同市值档置换零均值),括号内为 z 值\n{'='*120}")
print(f"{'信号':<28}" + "".join(f"{f'τ={q:.2f}':>15}" for q in TAUS))
rows = []
for nm in ALL:
    if nm not in res:
        continue
    dl, z, _, _ = res[nm]
    print(f"{nm:<28}" + "".join(f"{f'{a:+.1%}({b:+.1f})':>15}" for a, b in zip(dl, z)))
    rows.append(dict(信号=nm, **{f"delta_{q:.2f}": float(a) for q, a in zip(TAUS, dl)},
                     **{f"z_{q:.2f}": float(b) for q, b in zip(TAUS, z)}))

i50, i99 = TAUS.index(0.50), TAUS.index(0.99)
core = [n for n in NAMES if n in res]


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


rho = spearman([res[n][1][i50] for n in core], [res[n][1][i99] for n in core])
cnt = sum(res[n][1][i50] > 2 and res[n][1][i99] < -2 for n in core)

print(f"\n{'='*120}\n判据 按月分块 bootstrap({NBOOT} 次)\n{'='*120}")
# 重抽样只重算 Δ,**置换零分布的 mu/sd 沿用全样本那一版**。
# 每个 bootstrap 轮内重跑 200 次置换 = 4 万倍工作量,不可行;
# 这是一处**近似**,必须写进正文限定:它假定零分布的位置与尺度对月集重抽样不敏感。
rngb = np.random.default_rng(SEED)
nmn = len(per_month)
rhos = []
for _ in range(NBOOT):
    pick = [per_month[i] for i in rngb.integers(0, nmn, nmn)]
    zz = {}
    for n in core:
        vs = [d[n][0] for _, d, _ in pick if n in d]
        if len(vs) < 12:
            continue
        v = np.concatenate(vs)
        _, _, mu, sd = res[n]
        q = np.array([np.quantile(v, t) for t in TAUS])
        zz[n] = np.where(sd > 0, (q - mu) / sd, np.nan)
    if len(zz) < 6:
        continue
    rhos.append(spearman([v[i50] for v in zz.values()],
                         [v[i99] for v in zz.values()]))
rhos = np.array(rhos)
lo, hi = np.percentile(rhos, [2.5, 97.5])
print(f"  Spearman ρ(z@τ=0.50, z@τ=0.99)  点估计 **{rho:+.4f}**")
print(f"  bootstrap 中位 {np.median(rhos):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  「z@0.50 >2 且 z@0.99 <−2」的信号数:**{cnt}/{len(core)}**(门槛 ≥3)")
rows.append(dict(信号="判据", rho=rho, CI下界=float(lo), CI上界=float(hi),
                 反号信号数=cnt, 信号总数=len(core)))

print(f"\n{'='*120}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*120}")
zr = res.get(ZERO)
ay = res.get("AGE_YOUNG 上市[1,3)年")
a2 = zr is not None and bool(np.all(np.abs(zr[1]) < 2))
a3 = ay is not None and bool(ay[1][i99] > 2)
c1 = bool(rho < 0 and not (lo <= 0 <= hi))
c2 = bool(cnt >= 3)
print("  ✓ 锚点① 面板 (3297, 5232)")
if zr is None:
    print("  ✗ 锚点② 零锚点算不出 —— 不通过")
else:
    print(f"  {'✓' if a2 else '✗'} 锚点② RANDOM_LABEL 六个 τ 上 |z|<2   "
          f"最大 |z| = {np.abs(zr[1]).max():.2f}")
if ay is None:
    print("  ✗ 锚点③ AGE_YOUNG 算不出 —— 不通过")
else:
    print(f"  {'✓' if a3 else '✗'} 锚点③ AGE_YOUNG 在 τ=0.99 上 z>2(正向锚点)   "
          f"z = {ay[1][i99]:+.2f}")
print(f"  {'✓' if c1 else '✗'} 判据① ρ<0 且 95% CI 不含 0   "
      f"ρ={rho:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  {'✓' if c2 else '✗'} 判据② 反号信号数 ≥3   {cnt}/{len(core)}")
print()
if not (a2 and a3):
    print("  **锚点不过:本节结论作废。**")
elif c1 and c2:
    print("  **结论:§62「提高胜率必削右尾」得到系统支持。事前预测被证伪 —— 我错了。**")
elif c1 or c2:
    print("  **结论:两版判据只过一版,§62 的断言部分成立,不能当普遍规律用。**")
else:
    print("  **结论:中位数效应与右尾效应之间没有系统性反号关系。**")
    print("  **§62 那条是从一个例子过度归纳的,不能当普遍规律。事前预测命中。**")

pd.DataFrame(rows).to_csv(f"{SP}/quantile_curve_v2.csv", index=False)
print(f"\n→ {SP}/quantile_curve_v2.csv   ({time.time()-t0:.0f}s)")
