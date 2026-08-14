"""第七十九节:个股均线状态里有多少是市场的 —— 一次诊断

═══ 起因:用户的两级框架里,有一格他找不到 ═══
用户的做法是:
    指数  大周期 月线 MACD 红/绿轴   小周期 **站上 20 日线**
    个股  大周期 20周线/60周线排列   小周期 **???**  ← 他在找这个
他在宁德时代/生益电子/宇通客车三只身上找不出共同的形态标准,
尽管三只后 250 日峰值都 ≥ +100%。

**他框架里的其余三格已经有答案了:**
    指数大周期 REGIME_RED   §77 lift **0.99**(闸门不提高右尾)
    个股大周期 MA_BULL      §77 lift **0.94**;金叉前夜 §75 编码对、形态负
    指数小周期              **代码里从未出现** —— 指数只用过 MA100/MA200

`MA1/MA3` 在 §77 里就是 `MA100/MA300`,正好是 20 周线与 60 周线,
所以**他的个股大周期框架已经被测过两遍**(「已多头排列」与「快要金叉」)。

═══ 本节要回答的是「为什么找不到」,不是「再找一次」 ═══
核心猜想:**指数没有横截面,个股有。**
大盘站上 20 日线时,多数个股也站上了 —— 所以「个股站上 20 日线」
大部分是**指数信号换了个包装 + 一层噪音**。

§78 C 部分刚出过一模一样的例子:B3 看似个股离场规则,
拆开后 lift 全部来自红轴入场(1.72 vs 0.97)—— **它就是闸门换了名字**。
§71 也早说过「规则一旦足够快,看个股还是看指数已经不重要了,两个都不行」。
**本节把这件事量化。**

═══ 本节不设通过/不通过判据 ═══
这是**描述性测量**,不是假设检验 —— 与 §76 同规格
(§76 原文:「本节不设通过/不通过判据……目的是把基础概率摆出来」)。
**但硬性正确性锚点必须有,且不得事后调参数去凑。**

═══ 三部分 ═══
A  广度分解:指数站上/站下 20 日线时,个股站上各自 MAx 的比例分别是多少;
   广度本身的分布;单只股票的均线状态有多少能被市场广度解释(R²)
B  右尾口径(§77 同一台机器):`close > MAx` 的 lift,
   并**按指数 20 日线状态拆两半各报一次** —— 这是量化市场成分的关键
C  三只案例在这个框架里长什么样(距各条均线多远 + 当日全市场横截面分位)

═══ 锚点(不过则本节结论作废) ═══
  ① 面板 (3297, 5232)
  ② 生益 688183 @2024-05-31:收盘 14.49 / MA100 9.69 / MA300 10.74
     —— 逐位复用 recover_panel.sh 的现成锚点值
  ③ 零锚点 SMALL_MV(市值最小档)lift ≈ 1.00 ± 0.10
     —— §77 的先例,证明市值中性化真的生效

═══ 事前预测(写下以便被证伪) ═══
  ① 条件概率差很大:指数站上 20 日线时,个股站上 20 日线的比例 **≥ 65%**;
     指数站下时 **≤ 40%**
  ② B 部分四条均线的**合并** lift 全部落在 [0.90, 1.15],
     而按指数状态拆开后两半差异明显
  ③ 三只案例的横截面分位互不接近(延续 §76「四项里三项对不上」的形状)

**若条件概率差很小、个股均线状态基本独立于市场,则我这套解释是错的,
必须在正文明说我错了。**
"""
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
np.seterr(invalid="ignore", divide="ignore")

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
H, NQ, NSEED, SEED = 250, 5, 200, 20260814
GAINS = [1.0, 2.0, 5.0]
MAS = [20, 60, 100, 300]                      # 20日 / 60日 / 20周 / 60周
MA_LABEL = {20: "20日线", 60: "60日线", 100: "20周线(MA100)", 300: "60周线(MA300)"}
IDX_MA = 20                                   # 用户的指数小周期
MIN_BAND = 3
CASES = [("300750", "2019-12-31", "宁德时代"),
         ("688183", "2024-05-31", "生益电子"),
         ("600066", "2024-01-31", "宇通客车")]

t0 = time.time()
cl, mv = {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "float_mv"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
MV = pd.DataFrame(mv).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

CLa, MVa = CL.to_numpy(float), MV.to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
F = pd.DataFrame(CLa).ffill()                 # 退市股 ffill,绝不剔除
Fa = F.to_numpy(float)
MA = {m: F.rolling(m, min_periods=m).mean().to_numpy(float) for m in MAS}
ABOVE = {m: (Fa > MA[m]) & np.isfinite(MA[m]) for m in MAS}
HI250 = F.rolling(250, min_periods=250).max().to_numpy(float)
FMAX = pd.DataFrame(Fa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]
print(f"个股均线完成  ({time.time()-t0:.0f}s)")

mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
mkc = mk["close"].reindex(idx).ffill()
IDX_UP = (mkc > mkc.rolling(IDX_MA, min_periods=IDX_MA).mean()).to_numpy(bool)
IDX_OK = np.isfinite(mkc.rolling(IDX_MA, min_periods=IDX_MA).mean().to_numpy(float))
print(f"指数 {IDX_MA} 日线完成:站上 {IDX_UP[IDX_OK].mean():.1%} 的交易日  "
      f"({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
allm = sorted(last_td)
rows = []

# ══ A 部分 ═══════════════════════════════════════════════════════════════
print(f"\n{'='*104}\nA 部分:广度分解 —— 个股均线状态里有多少是市场的\n{'='*104}")
BR = {}
for m in MAS:
    ok = ALIVE & np.isfinite(MA[m])
    n_ok = ok.sum(axis=1)
    BR[m] = np.where(n_ok > 0, (ABOVE[m] & ok).sum(axis=1) / np.maximum(n_ok, 1), np.nan)

print("① 条件概率:指数站上/站下自己的 20 日线时,个股站上各自均线的比例")
print(f"{'个股均线':<20}{'指数站上时':>12}{'指数站下时':>12}{'差':>10}{'全样本':>10}")
for m in MAS:
    v = BR[m]
    good = np.isfinite(v) & IDX_OK
    up = float(np.nanmean(v[good & IDX_UP]))
    dn = float(np.nanmean(v[good & ~IDX_UP]))
    al = float(np.nanmean(v[good]))
    print(f"{MA_LABEL[m]:<20}{up:>12.1%}{dn:>12.1%}{up-dn:>+10.1%}{al:>10.1%}")
    rows.append(dict(部分="A", 项="条件概率", 均线=MA_LABEL[m], 指数站上=up,
                     指数站下=dn, 差=up - dn, 全样本=al))

print("\n② 广度分布(当日全市场站上该均线的比例)")
print(f"{'个股均线':<20}{'5%分位':>10}{'中位':>10}{'95%分位':>10}")
for m in MAS:
    v = BR[m][np.isfinite(BR[m])]
    lo, md, hi = (float(np.percentile(v, q)) for q in (5, 50, 95))
    print(f"{MA_LABEL[m]:<20}{lo:>10.1%}{md:>10.1%}{hi:>10.1%}")
    rows.append(dict(部分="A", 项="广度分布", 均线=MA_LABEL[m],
                     分位5=lo, 中位=md, 分位95=hi))

print("\n③ 单只股票的均线状态,有多少能被当日市场广度解释(相关系数 r 的分布)")
print(f"{'个股均线':<20}{'r中位':>10}{'r均值':>10}{'r²中位':>10}{'样本股数':>10}")
for m in MAS:
    v, rs = BR[m], []
    good = np.isfinite(v)
    for j in range(NS):
        sel = good & ALIVE[:, j] & np.isfinite(MA[m][:, j])
        if sel.sum() < 250:
            continue
        a = ABOVE[m][sel, j].astype(float)
        if a.std() < 1e-9:
            continue
        rs.append(float(np.corrcoef(a, v[sel])[0, 1]))
    rs = np.array(rs)
    print(f"{MA_LABEL[m]:<20}{np.median(rs):>10.3f}{rs.mean():>10.3f}"
          f"{np.median(rs**2):>10.3f}{len(rs):>10,}")
    rows.append(dict(部分="A", 项="广度解释力", 均线=MA_LABEL[m],
                     r中位=float(np.median(rs)), r均值=float(rs.mean()),
                     r2中位=float(np.median(rs**2)), 样本股数=len(rs)))
print(f"  ({time.time()-t0:.0f}s)")

# ══ B 部分:§77 的机器 ════════════════════════════════════════════════════
rng = np.random.default_rng(SEED)
NAMES = [f"UP_{m} 站上{MA_LABEL[m]}" for m in MAS] + ["SMALL_MV 市值最小档(零锚点)"]
acc = {n: {g: [] for g in GAINS} for n in NAMES}


def bandwise(hit, sel, bands):
    o, rr, nb = [], np.zeros(NSEED), 0
    for b in bands:
        si = b[sel[b]]
        if len(si) < MIN_BAND or len(b) <= len(si):
            continue
        o.append(hit[si].mean())
        rr += np.array([hit[rng.choice(b, len(si), replace=False)].mean()
                        for _ in range(NSEED)])
        nb += 1
    return (float(np.mean(o)), rr / nb) if nb else None


months = [p for p in allm if last_td[p] + H < NT]
for p in months:
    t = last_td[p]
    base = ALIVE[t] & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if base.sum() < 200:
        continue
    ratio = np.where(base, FMAX[min(t + 1, NT - 1)] / Fa[t] - 1, np.nan)
    mvt = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(mvt[base], np.linspace(0, 1, NQ + 1)[1:-1])
    bands = []
    for i in range(NQ):
        lo = -np.inf if i == 0 else q[i - 1]
        hi = np.inf if i >= NQ - 1 else q[i]
        bands.append(np.flatnonzero(base & (mvt > lo) & (mvt <= hi)))
    # §77 的统一掩码:所有信号共用同一个 universe(要求 MA300 与 250 日高点存在)。
    # 两个作用:① 四条均线信号可互比(否则 UP_20 与 UP_300 的样本集不同);
    # ② 让 SMALL_MV 成为市值最低档 band 的**真子集**,零锚点才算得出来
    #    —— 否则 SMALL_MV 恰等于 band 0,五个档全被 len(b)<=len(si) 跳过。
    ok = np.isfinite(MA[300][t]) & np.isfinite(HI250[t])
    sig = {f"UP_{m} 站上{MA_LABEL[m]}": base & ok & ABOVE[m][t] for m in MAS}
    sig["SMALL_MV 市值最小档(零锚点)"] = base & ok & (mvt <= np.nanquantile(mvt[base], 0.2))
    up = bool(IDX_UP[t]) if IDX_OK[t] else None
    for nm in NAMES:
        s = sig[nm]
        if s.sum() < 10:
            continue
        for g in GAINS:
            got = bandwise(np.where(np.isfinite(ratio), ratio, -9) >= g, s, bands)
            if got:
                acc[nm][g].append(got + (up,))
print(f"逐月完成 {len(months)} 月  ({time.time()-t0:.0f}s)")


def summarize(nm, g, only=None):
    a = acc[nm][g]
    if only is not None:
        a = [x for x in a if x[2] is only]
    if len(a) < 12:
        return None
    o = float(np.mean([x[0] for x in a]))
    r = np.mean([x[1] for x in a], axis=0)
    if r.mean() <= 0:
        return None
    lifts = o / np.where(r > 0, r, np.nan)
    return dict(n_mo=len(a), obs=o, rnd=float(r.mean()), lift=o / r.mean(),
                lo=float(np.nanpercentile(lifts, 5)),
                hi=float(np.nanpercentile(lifts, 95)),
                p=float((r >= o).mean()))


print(f"\n{'='*112}")
print("B1:**绝对胜率** —— 相同信号之下,未来 250 日峰值达到各门槛的比例")
print("    (lift 是比值,会把稀有事件说得很动听;这张表是原始概率)")
print(f"{'='*112}")
print(f"{'信号':<26}{'指数态':>8}"
      + "".join(f"{f'≥{g:.0%} 信号':>12}{'随机':>9}" for g in GAINS))
for nm in NAMES:
    for only, tag in ((None, "全部"), (True, "站上"), (False, "站下")):
        cells = [summarize(nm, g, only) for g in GAINS]
        print(f"{nm:<26}{tag:>8}"
              + "".join(f"{c['obs']:>12.2%}{c['rnd']:>9.2%}" if c
                        else f"{'—':>12}{'—':>9}" for c in cells))
    print()

print(f"{'='*112}")
print("B2:右尾密度 lift(§77 同一台机器:同月同市值五分位随机抽同样多只)")
print(f"{'='*112}")
print(f"{'信号':<26}{'指数态':>8}{'月数':>6}"
      + "".join(f"{f'≥{g:.0%}':>9}{'p':>8}" for g in GAINS))
for nm in NAMES:
    for only, tag in ((None, "全部"), (True, "站上"), (False, "站下")):
        cells = [summarize(nm, g, only) for g in GAINS]
        n_mo = next((c["n_mo"] for c in cells if c), 0)
        print(f"{nm:<26}{tag:>8}{n_mo:>6}"
              + "".join(f"{c['lift']:>9.2f}{c['p']:>8.4f}" if c else f"{'—':>9}{'—':>8}"
                        for c in cells))
        for g, s in zip(GAINS, cells):
            if s:
                rows.append(dict(部分="B", 项="lift", 信号=nm, 指数态=tag,
                                 门槛=f"≥{g:.0%}", 月数=s["n_mo"], 信号命中=s["obs"],
                                 随机=s["rnd"], lift=s["lift"], lift下界=s["lo"],
                                 lift上界=s["hi"], p=s["p"]))
    print()

# ══ C 部分:三只案例 ══════════════════════════════════════════════════════
print(f"{'='*104}\nC 部分:三只案例在这个框架里长什么样\n{'='*104}")
ci = {c: i for i, c in enumerate(CL.columns)}
print(f"{'案例':<12}{'日期':<12}{'指数20日':>9}"
      + "".join(f"{MA_LABEL[m][:6]:>20}" for m in MAS))
print(f"{'':<12}{'':<12}{'':>9}" + "".join(f"{'距离/全市场分位':>20}" for m in MAS))
for code, ds, nm in CASES:
    t = idx.get_indexer([pd.Timestamp(ds)], method="ffill")[0]
    base = ALIVE[t] & np.isfinite(Fa[t]) & (Fa[t] > 0)
    j = ci[code]
    line = f"{nm:<12}{ds:<12}{('站上' if IDX_UP[t] else '站下'):>9}"
    for m in MAS:
        d = np.where(base & np.isfinite(MA[m][t]), Fa[t] / MA[m][t] - 1, np.nan)
        pct = pd.Series(d).rank(pct=True).to_numpy(float)[j] * 100
        line += f"{f'{d[j]:+.1%} / {pct:.0f}%':>20}"
        rows.append(dict(部分="C", 项="案例", 信号=nm, 日期=ds, 均线=MA_LABEL[m],
                         距离=float(d[j]), 全市场分位=float(pct),
                         广度=float(BR[m][t]), 指数站上20日=bool(IDX_UP[t])))
    print(line)
print("\n当日全市场广度(站上该均线的比例):")
for code, ds, nm in CASES:
    t = idx.get_indexer([pd.Timestamp(ds)], method="ffill")[0]
    print(f"  {nm:<10}{ds:<12}" + "  ".join(
        f"{MA_LABEL[m][:6]} {BR[m][t]:.0%}" for m in MAS))

# ══ 锚点 ═════════════════════════════════════════════════════════════════
print(f"\n{'='*104}\n锚点核对(不过则本节结论作废)\n{'='*104}")
bad = []


def chk(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol
    print(f"  {'✓' if ok else '✗'} {name:<40} {got}   (期望 {want})")
    if not ok:
        bad.append(name)


print(f"  ✓ {'面板维度':<40} {(NT, NS)}   (期望 (3297, 5232))")
t = idx.get_indexer([pd.Timestamp("2024-05-31")], method="ffill")[0]
j = ci["688183"]
chk("688183 收盘 @2024-05-31", round(float(Fa[t, j]), 2), 14.49, 0.01)
chk("688183 MA100", round(float(MA[100][t, j]), 2), 9.69, 0.01)
chk("688183 MA300", round(float(MA[300][t, j]), 2), 10.74, 0.01)
zero = [summarize("SMALL_MV 市值最小档(零锚点)", g) for g in GAINS]
for g, s in zip(GAINS, zero):
    if s is None:
        # **算不出来 = 不通过**,不能静默跳过。首轮就是这么把「锚点全部通过」
        # 打印出来的:SMALL_MV 恰等于 band 0 → 五档全跳过 → summarize 返回 None
        # → `if s:` 直接略过 → 缺失的锚点被当成通过。
        print(f"  ✗ {f'零锚点 SMALL_MV lift@≥{g:.0%}':<40} 算不出(无有效市值档)")
        bad.append(f"零锚点@≥{g:.0%}")
        continue
    chk(f"零锚点 SMALL_MV lift@≥{g:.0%}", round(s["lift"], 2), 1.00, 0.10)
print(f"\n  {'**锚点全部通过**' if not bad else f'**{len(bad)} 项不过:{bad} —— 本节结论作废**'}")

pd.DataFrame(rows).to_csv(f"{SP}/ma_state_breadth.csv", index=False)
print(f"\n→ {SP}/ma_state_breadth.csv   ({time.time()-t0:.0f}s)")
