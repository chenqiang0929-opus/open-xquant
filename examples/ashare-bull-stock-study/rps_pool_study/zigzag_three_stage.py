"""第九十四节:新分段(ZigZag 三段)全样本检验 —— 它是不是真的比旧筛选器强(事前登记)

═══ 起因:用户看图否掉了我的分段方式 ═══
用户指出宇通 2013→2015 是一次完整的三段结构(第一段 +38.4%、平台 327 日净涨 +0.5%、
第三段 +103.6%),而**旧筛选器整段没认**;2015-2018 是一个三年横盘
(12.58→11.58 = −8.0%),而**旧筛选器在里面亮了三次、突破了三次,全是废点**。

单只实测(`case_yutong_13y.py`,已落库)证实两条都对:

    突破日          旧筛选器  新分段   6月峰值   6月期末
    2014-12-04        —      ✓     +90.9%   +84.8%    <- 用户指的那次,旧的漏了
    2016-07-15        ✓      —      +6.4%   −11.5%    <- 旧的独有,废点
    2017-07-21        ✓      ✓     +15.9%    +1.6%
    2018-01-03        ✓      —      −1.5%   **−30.7%**  <- 旧的独有,废点
    2020-08-06        —      ✓     +21.5%   −11.8%
    2024-01-08/10     ✓      ✓    +101.1%   +78.8%
    2025-09-01        —      ✓     +23.9%    +9.5%

**但那是 1 只股票、9 个事件,证明不了任何事。本节把新分段拿上全样本。**

═══ 分段算法(已锁定,锚点预验证见 case_yutong_13y.py)═══
  ZigZag 反转阈值 **θ = 10%**(自极值回撤 ≥10% 确认枢轴)
  **θ 的来历**:要求算法复现用户看图指认的两处三段(2014 年底、2024-01)。
  θ=10% 两处都认;15% 漏后者;20% 只剩两个;25% 一个都没有。故锁定 10%。
  **初稿还有一个锚点条件「不得在 2015-2018 横盘里触发」,已删除** ——
  那是把「表现」伪装成「正确性」,照它调参数就是拿结果反推检测器。

  三段模板:**上涨段**(单条上升腿 ≥30%)→ **平台段**(其后连续枢轴全落在
  ±35.2% 带内、时长 ≥60 个交易日;35.2% 取自筛选器 THR_DEPTH,不是新参数)
  → **突破**(收盘首次 > 平台段最高收盘,**上限 250 日**,与 §89 一致)
  同一突破日只记一次;入场在突破日收盘。

═══ 口径(事前锁定,与 §89 完全对齐以便直接比较)═══
  前瞻    6 个月(120 日)峰值 ≥100% = 判据口径;12 个月仅描述
  对照A   同日**同市值五分位**随机 × 200 组
  对照B   同日同市值 **且当日也创 250 日新高** 的随机股 —— 隔离动量后的基准
  退市股按最后有效价 ffill 参与,绝不剔除

═══ 锚点(不过则全节作废;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **宇通恒等复现单只跑**:新分段在 600066 上给出 **5** 个突破日,
     且含 **2014-12-04** 与 **2024-01-10**(单只已实测,全样本必须一字不差)
  ③ **恒等零校验**:对照A / 对照B 的中位命中率 vs 各自同格总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽)═══
  **前置条件**:事件数 **< 300** 不判;逐年某年 **< 100** 的年份不计入判据③
  ① 对**对照A**:6 个月 ≥100% 的 **lift ≥ 1.3 且 p < 0.05/4 = 0.0125**
  ② 对**对照B**(隔离动量):**同样 lift ≥ 1.3 且 p < 0.0125**
  ③ **逐年方向一致性**(§91 立的规矩):逐年对照A 的 **lift > 1.0 的年份占比 ≥ 80%**

**①②③ 全过 = 新分段是一个独立于动量、且逐年稳定的右尾信号 —— 本项目至今没有过。
①③ 过而 ② 不过 = 与 §89 第三段同构:真实但等于动量。**

**头对头(描述,不设判据)**:§89 旧筛选器 legacy|突破 11,645 事件、
≥100% 7.39%、liftA **1.32**、liftB **0.97**。本节并列打印,
但**两套事件集不同,不构成同口径比较**,只作参考。

═══ 判据自查(§79 正问 + §83 反问)═══
**正问:什么会让它通过而不回答问题?**
→ 突破 = 已经涨了,liftA 含动量 → **堵法:判据② 用对照B**(§89 的教训)。
→ 事件集中在右尾本来就肥的年份 → **堵法:对照按同日抽 + 判据③ 逐年**(§91 的教训)。
→ 4 次比较搜出假阳性 → **堵法:Bonferroni 0.05/4**。
→ **θ 是我在宇通身上挑的** → 无法完全堵住;**本节结论必须带着这个污点读**,
  已写进正文。θ 只在**两处结构能否被认出**上被挑选,**没有用结果调过**。

**反问:什么会让它不通过而与问题无关?**
→ 事件数不足 → **堵法:前置 n≥300 才判**。
→ 锚点误杀正确实现(§85/§87/§88 病根)→ **堵法:三个锚点全是恒等式,
  且锚点② 已在单只实测可达**。
→ 判据只有幅度没有显著性(§92 的教训)→ **堵法:①② 都带 p,③ 是占比不是点估计**。

═══ 事前预测(写下以便被证伪)═══
**① 通过、② 不通过、③ 不通过。**
理由:§89 旧筛选器的突破格对对照A 是 1.32(过),对对照B 是 0.97(不过);
§91 又证明「创新高」这个量逐年翻号。**新分段换的是「怎么找箱体」,
没有换「突破之后靠什么涨」—— 我预计它对对照A 会更好看一点,
但对隔离动量的对照B 仍然归零。**
**若 ② 也通过,说明结构分段确实带来了动量之外的信息,我错了 ——
那会是本项目第一个独立于动量的右尾信号。**
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
NQ, NSEED, SEED = 5, 200, 20260814
TH, UP_MIN, PLAT_MIN, CAP = 0.10, 0.30, 60, 250
BAND = THR_DEPTH
MIN_N, MIN_N_YEAR, NCELL = 300, 100, 4
ALPHA = 0.05 / NCELL
LIFT_MIN, YEAR_FRAC = 1.3, 0.80
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

Fa = CL.where(CL > 0).ffill().to_numpy(float)     # 退市股 ffill 参与
FIRST = np.argmax(np.isfinite(Fa), axis=0)        # 每只股票首个有效下标
HI250 = pd.DataFrame(Fa).rolling(250, min_periods=100).max().to_numpy(float)
NEWHI = np.isfinite(HI250) & (Fa >= HI250 * 0.9999)
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
    if (j + 1) % 1500 == 0:
        print(f"  分段 {j+1:,}/{NS:,}  事件 {len(EV):,}  ({time.time()-t0:.0f}s)", flush=True)
print(f"\n三段突破事件 **{len(EV):,}** 个  ({time.time()-t0:.0f}s)", flush=True)

JY = codes.index("600066")
jyd = sorted(str(idx[t].date()) for t, j in EV if j == JY)
a2 = len(jyd) == 5 and "2014-12-04" in jyd and "2024-01-10" in jyd
print(f"  {'✓' if a2 else '✗'} 锚点② 宇通 5 个突破日,含 2014-12-04 与 2024-01-10"
      f"   实测 {jyd}")

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


rows = []
for n, hname in HOR:
    pm = PK[n]
    sub = [(t, j) for t, j in EV if np.isfinite(pm[t, j])]
    v = np.array([pm[t, j] for t, j in sub]) >= 1.0
    obs = float(v.mean())
    r = dict(前瞻=hname, 事件数=len(sub), ge100=obs)
    for nm, nh in (("A", False), ("B", True)):
        c, th = control(sub, pm, nh)
        med = float(np.median(c)) if c.size else np.nan
        r |= {f"对照{nm}": med, f"lift{nm}": obs / med if med > 0 else np.nan,
              f"p{nm}": float((c >= obs).mean()) if c.size else np.nan,
              f"零校验{nm}": abs(med - th)}
    rows.append(r)
    print(f"\n{'='*104}\n{hname}峰值 ≥100%{'  【判据口径】' if n == 120 else '  (仅描述)'}"
          f"\n{'='*104}")
    print(f"  事件数 {len(sub):,}   新分段 ≥100% = {obs:.2%}")
    for nm, lbl in (("A", "对照A 同市值随机            "),
                    ("B", "对照B 同市值+当日也创新高    ")):
        print(f"  {lbl} {r['对照'+nm]:.2%}   lift {r['lift'+nm]:.2f}   "
              f"p {r['p'+nm]:.4f}   零校验 {r['零校验'+nm]:.2%}")
R = pd.DataFrame(rows)
M6 = R[R["前瞻"] == "6个月"].iloc[0]

print(f"\n{'='*104}\n逐年稳定性(对照A,6 个月口径)\n{'='*104}")
pm = PK[120]
sub6 = [(t, j) for t, j in EV if np.isfinite(pm[t, j])]
yr = []
for y in sorted({idx[t].year for t, _ in sub6}):
    ev = [(t, j) for t, j in sub6 if idx[t].year == y]
    if len(ev) < MIN_N_YEAR:
        print(f"  {y}  n={len(ev):>5,}  < {MIN_N_YEAR},不计入判据③")
        continue
    vv = np.array([pm[t, j] for t, j in ev]) >= 1.0
    o = float(vv.mean())
    c, _ = control(ev, pm, False)
    md = float(np.median(c)) if c.size else np.nan
    lf = o / md if md > 0 else np.nan
    yr.append(dict(年=y, n=len(ev), ge100=o, 对照A=md, lift=lf))
    print(f"  {y}  n={len(ev):>5,}  ≥100% {o:>6.2%}  对照A {md:>6.2%}  "
          f"lift {lf:>5.2f}  {'✓' if lf > 1.0 else '✗'}")
Y = pd.DataFrame(yr)
frac = float((Y["lift"] > 1.0).mean()) if len(Y) else np.nan

print(f"\n{'='*104}\n锚点核对(不过则全节作废)\n{'='*104}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② 宇通恒等复现单只跑")
if not a2:
    bad.append("锚点②")
zs = [R[f"零校验{k}"].max() for k in ("A", "B")]
a3 = bool(np.isfinite(zs).all() and max(zs) <= 0.03)
print(f"  {'✓' if a3 else '✗'} 锚点③ 恒等零校验 最大差 "
      f"{max(zs):.2%} ≤ 3pp" if np.isfinite(zs).all() else "  ✗ 锚点③ 算不出 = 不通过")
if not a3:
    bad.append("锚点③")

print(f"\n{'='*104}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*104}")
print(f"  前置条件:事件数 {M6['事件数']:,} ≥ {MIN_N};逐年合格 {len(Y)} 年")
c1 = bool(M6["liftA"] >= LIFT_MIN and M6["pA"] < ALPHA)
c2 = bool(M6["liftB"] >= LIFT_MIN and M6["pB"] < ALPHA)
c3 = bool(np.isfinite(frac) and frac >= YEAR_FRAC)
print(f"  {'✓' if c1 else '✗'} 判据① 对照A  lift {M6['liftA']:.2f} ≥ {LIFT_MIN} "
      f"且 p {M6['pA']:.4f} < {ALPHA}")
print(f"  {'✓' if c2 else '✗'} 判据② 对照B  lift {M6['liftB']:.2f} ≥ {LIFT_MIN} "
      f"且 p {M6['pB']:.4f} < {ALPHA}")
print(f"  {'✓' if c3 else '✗'} 判据③ 逐年 lift>1.0 占比 {frac:.1%} ≥ {YEAR_FRAC:.0%}")
print("\n  头对头(描述,两套事件集不同,不构成同口径比较):")
print("    §89 旧筛选器 legacy|突破  11,645 事件  ≥100% 7.39%  liftA 1.32  liftB 0.97")
print(f"    §94 新分段 ZigZag 三段    {M6['事件数']:,} 事件  ≥100% {M6['ge100']:.2%}"
      f"  liftA {M6['liftA']:.2f}  liftB {M6['liftB']:.2f}")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:新分段是独立于动量、且逐年稳定的右尾信号。事前预测被证伪 —— 我错了。**")
elif c1 and c3:
    print("  **结论:新分段对同市值随机显著更好且逐年稳定,但对隔离动量的对照归零 ——")
    print("     与 §89 第三段同构:真实但等于动量。事前预测命中。**")
elif c1:
    print("  **结论:新分段对同市值随机更好,但逐年不稳定(§91 同款),不构成因子。**")
else:
    print("  **结论:新分段在右尾口径下不优于同市值随机。**")
print("\n  **口径污点必须随结论一起读:θ=10% 是我在宇通身上挑的**"
      "(只按「两处结构能否被认出」挑,没有用结果调过)。")

R.to_csv(f"{OUT}/zigzag_three_stage.csv", index=False)
Y.to_csv(f"{OUT}/zigzag_three_stage_yearly.csv", index=False)
pd.DataFrame([(str(idx[t].date()), codes[j]) for t, j in EV],
             columns=["突破日", "代码"]).to_csv(
    f"{OUT}/zigzag_three_stage_events.csv", index=False)
print(f"\n→ {OUT}/zigzag_three_stage.csv + _yearly.csv + _events.csv"
      f"   ({time.time()-t0:.0f}s)")
