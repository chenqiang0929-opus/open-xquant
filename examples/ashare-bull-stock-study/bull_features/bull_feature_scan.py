"""牛股特征系统扫描 —— 还有哪些与非牛股不同的因子

═══ 起因 ═══
用户看完第五十一节(基底形态/均线/CANSLIM)后问:
"除了肉眼可识别的形态、均线,还有没有没挖掘出来的、与非牛完全不同的因子?"

本脚本扫描面板里**有数据但从没在牛股归因里用过**的字段。

═══ 沿用第五十一节的口径(否则不可比) ═══
对每只股票每一年,用同一方法定位「该年最大涨幅的起点」t*:
    t* = argmax over t of ( max(close[t..年末]) ÷ close[t] − 1 )
牛股与非牛股在 t* 上测同样的特征,**全部在 t* 之前测量,无前视**。
每个特征报 P(特征|牛股)、P(特征|非牛)、**P(牛股|特征)**、lift。

═══ 扫描的七类(用户选定) ═══
  1 涨停/跌停次数    is_limit_up/down  —— A股独有的题材股指纹,美股方法论没有
  2 换手率           turnover          —— 此前只用过"量比",没用过换手率本身
  3 次新股           listed_days
  5 盈利加速度       当季同比的二阶导   —— 欧奈尔强调,有干净单季数据后第一次能算
  6 现金流质量       OCF / net_income
  8 估值极端         bp_correct        —— **U形**:两端可能都是高发区,单调因子测不出
  9 逆势强度         大盘跌时的相对表现
参考项(用户表示不打算用,仅供观察,不进主判据):
  4 ST 摘帽          is_st
  7 股本扩张         outstanding_share —— 用户指出 2018 后送转已不流行

═══ 扫 20+ 个特征必然出假阳性,两条纪律 ═══
**A. 噪音上界(最重要)**:每年内打乱牛股标签(保留各年基准率)200 次,
   每次记录**所有特征里的最高 lift**,得到"纯噪音下 best-of-N 能到多少"。
   三十一节实测该值约 2.09;第五十一节全场最高 lift 才 2.68。
   **真实特征的 lift 必须显著超过这条线才算发现。**
**B. FDR 校正**:对每个特征的 Fisher 精确检验 p 值做 Benjamini-Hochberg 校正。
"""
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
Y0, Y1 = 2013, 2025
N_PERM = 200
SEED = 20260811

t0 = time.time()
COLS = ["close", "high", "low", "volume", "turnover", "float_mv", "bp_correct",
        "is_limit_up", "is_limit_down", "is_st", "listed_days",
        "outstanding_share", "operating_cash_flow", "net_income"]
d = {c: {} for c in COLS}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    try:
        x = pd.read_parquet(f, columns=COLS)
    except Exception:
        continue
    if x.empty:
        continue
    for c in COLS:
        d[c][k] = pd.to_numeric(x[c], errors="coerce")
CL = pd.DataFrame(d["close"]).sort_index(); CL.index = CL.index.tz_localize(None)


def al(k):
    f = pd.DataFrame(d[k]).sort_index(); f.index = f.index.tz_localize(None)
    return f.reindex(index=CL.index, columns=CL.columns)


TURN, MV, BP = al("turnover"), al("float_mv"), al("bp_correct")
LU, LD, ST = al("is_limit_up"), al("is_limit_down"), al("is_st")
LDAYS, OSH = al("listed_days"), al("outstanding_share")
OCF, NI = al("operating_cash_flow"), al("net_income")
CL = CL.where(CL > 0)
idx = CL.index
A = CL.to_numpy()
NT, NC = A.shape
codes = list(CL.columns)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
del d

CQ = pd.read_parquet(f"{SP}/clean_growth_c_qyoy.parquet").reindex(
    index=idx, columns=CL.columns)
assert CQ.notna().mean().mean() > 0.01, "clean_growth C 字段几乎全空"

# 预计算(全部只用 t* 及之前的信息)
LU250 = LU.rolling(250, min_periods=120).sum().to_numpy()
LU60 = LU.rolling(60, min_periods=30).sum().to_numpy()
LD250 = LD.rolling(250, min_periods=120).sum().to_numpy()
TURN20 = TURN.rolling(20, min_periods=10).mean()
TURN_PCT = TURN20.rank(axis=1, pct=True).to_numpy()
BP_PCT = BP.rank(axis=1, pct=True).to_numpy()
# 盈利加速度:当季同比 − 上一报告期(约63交易日前)的当季同比
ACCEL = (CQ - CQ.shift(63)).to_numpy()
CQa = CQ.to_numpy()
# 现金流质量(同为 YTD 口径,比值有意义)
OCF_NI = (OCF / NI.where(NI > 0)).to_numpy()
LDAYSa, OSHa, STa, TURNa = LDAYS.to_numpy(), OSH.to_numpy(), ST.to_numpy(), TURN.to_numpy()
# 逆势强度:大盘下跌日的个股相对表现(过去60日)
_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mret = mkt.pct_change().to_numpy()
sret = CL.pct_change().to_numpy()
down = mret < 0
exc = np.where(down[:, None], sret - mret[:, None], np.nan)
CONTRA = pd.DataFrame(exc).rolling(60, min_periods=20).mean().to_numpy()
print(f"因子就绪  ({time.time()-t0:.0f}s)")

year = idx.year.to_numpy()
rows = []
for j, cd in enumerate(codes):
    a = A[:, j]
    fin = np.isfinite(a) & (a > 0)
    if fin.sum() < 300:
        continue
    for y in range(Y0, Y1 + 1):
        cur = np.flatnonzero((year == y) & fin)
        if cur.size < 100:
            continue
        prev = np.flatnonzero((year == y - 1) & fin)
        if prev.size == 0:
            continue
        yr_ret = a[cur[-1]] / a[prev[-1]] - 1
        fwd_max = np.maximum.accumulate(a[cur][::-1])[::-1]
        t = int(cur[int(np.argmax(fwd_max / a[cur] - 1))])
        if t < 310:
            continue
        rows.append({
            "code": cd, "year": y, "bull": yr_ret > 1.0, "t": t, "j": j,
            "lu250": LU250[t, j], "lu60": LU60[t, j], "ld250": LD250[t, j],
            "turn_pct": TURN_PCT[t, j], "turn_raw": TURNa[t, j],
            "listed": LDAYSa[t, j],
            "accel": ACCEL[t, j], "cq": CQa[t, j],
            "ocf_ni": OCF_NI[t, j],
            "bp_pct": BP_PCT[t, j],
            "contra": CONTRA[t, j],
            "is_st": STa[t, j],
            "osh_chg": (OSHa[t, j] / OSHa[t - 250, j] - 1) if t >= 250 and
                       np.isfinite(OSHa[t - 250, j]) and OSHa[t - 250, j] > 0 else np.nan,
        })
P = pd.DataFrame(rows)
BASE = P.bull.mean()
print(f"\n样本 {len(P):,},牛股 {int(P.bull.sum()):,}(基准率 **{BASE:.2%}**)"
      f"  ({time.time()-t0:.0f}s)")

# ---------------- 特征定义(全部二元,阈值取常规值,不调参) ----------------
FEATS = {
    # 1 涨停/跌停
    "涨停次数(250日)≥5": P.lu250 >= 5,
    "涨停次数(250日)≥10": P.lu250 >= 10,
    "涨停次数(60日)≥3": P.lu60 >= 3,
    "跌停次数(250日)≥3": P.ld250 >= 3,
    "涨停多且跌停少(≥5 且 ≤1)": (P.lu250 >= 5) & (P.ld250 <= 1),
    # 2 换手率
    "换手率分位 最高30%": P.turn_pct >= 0.70,
    "换手率分位 最低30%": P.turn_pct <= 0.30,
    "换手率 >10%": P.turn_raw > 10,
    # 3 次新股
    "次新股 上市<750日(3年)": P.listed < 750,
    "次新股 上市<500日": P.listed < 500,
    "老股 上市>2500日(10年)": P.listed > 2500,
    # 5 盈利加速度
    "盈利加速(当季同比环比上升)": P.accel > 0,
    "盈利加速 且 当季同比>25%": (P.accel > 0) & (P.cq > 0.25),
    "盈利大幅加速(提升>20pp)": P.accel > 0.20,
    # 6 现金流质量
    "现金流质量 OCF/NI>0.8": P.ocf_ni > 0.8,
    "现金流差 OCF/NI<0": P.ocf_ni < 0,
    # 8 估值极端(U形:两端 vs 中间)
    "估值极端 BP最低10%(高估值)": P.bp_pct <= 0.10,
    "估值极端 BP最高10%(破净)": P.bp_pct >= 0.90,
    "估值居中 BP 40~60%": (P.bp_pct >= 0.40) & (P.bp_pct <= 0.60),
    # 9 逆势强度
    "逆势强 大盘跌时超额>0": P.contra > 0,
    "逆势很强 大盘跌时超额>0.3%": P.contra > 0.003,
}
REF_FEATS = {   # 参考项,不进主判据
    "【参考】ST 股": P.is_st > 0,
    "【参考】股本扩张>20%(送转)": P.osh_chg > 0.20,
}


def stat(mask):
    m = mask.fillna(False).astype(bool).to_numpy()
    n1 = int(m.sum())
    if n1 < 30:
        return None
    b = P.bull.to_numpy()
    p_bull_given_f = b[m].mean()
    lift = p_bull_given_f / BASE
    tab = [[int((b & m).sum()), int((~b & m).sum())],
           [int((b & ~m).sum()), int((~b & ~m).sum())]]
    _, pv = stats.fisher_exact(tab)
    return {"P(特征|牛股)": m[b].mean(), "P(特征|非牛)": m[~b].mean(),
            "P(牛股|特征)": p_bull_given_f, "lift": lift, "命中数": n1, "p": pv}


print(f"\n{'#'*112}\n基准牛股率 {BASE:.2%};lift = P(牛股|特征) ÷ 基准\n{'#'*112}")
print(f"{'特征':<32}{'P(特征|牛股)':>12}{'P(特征|非牛)':>12}{'**P(牛股|特征)**':>14}"
      f"{'lift':>8}{'命中数':>10}{'Fisher p':>11}")
res = {}
for nm, mk in FEATS.items():
    s = stat(mk)
    if s is None:
        print(f"{nm:<32}{'样本<30':>12}")
        continue
    res[nm] = s
    print(f"{nm:<32}{s['P(特征|牛股)']:>12.1%}{s['P(特征|非牛)']:>12.1%}"
          f"{s['P(牛股|特征)']:>14.2%}{s['lift']:>8.2f}{s['命中数']:>10,}{s['p']:>11.2e}")
print("\n参考项(用户表示不打算用):")
for nm, mk in REF_FEATS.items():
    s = stat(mk)
    if s is None:
        print(f"{nm:<32}{'样本<30':>12}")
        continue
    print(f"{nm:<32}{s['P(特征|牛股)']:>12.1%}{s['P(特征|非牛)']:>12.1%}"
          f"{s['P(牛股|特征)']:>14.2%}{s['lift']:>8.2f}{s['命中数']:>10,}{s['p']:>11.2e}")

# ---------------- 纪律 A:噪音上界(best-of-N) ----------------
print(f"\n{'='*112}\n纪律A 噪音上界:年内打乱牛股标签 {N_PERM} 次,每次取所有特征的最高 lift\n{'='*112}")
rng = np.random.default_rng(SEED)
masks = {nm: FEATS[nm].fillna(False).to_numpy() for nm in res}
yr = P.year.to_numpy()
best_null = []
for _ in range(N_PERM):
    bb = P.bull.to_numpy().copy()
    for yv in np.unique(yr):
        s = yr == yv
        bb[s] = rng.permutation(bb[s])          # 年内打乱,保留各年基准率
    lifts = [bb[m].mean() / BASE for m in masks.values() if m.sum() >= 30]
    best_null.append(max(lifts))
best_null = np.array(best_null)
real_best = max(v["lift"] for v in res.values())
real_best_nm = max(res, key=lambda k: res[k]["lift"])
print(f"  纯噪音 best-of-{len(masks)} lift:中位 **{np.median(best_null):.2f}**   "
      f"95%分位 **{np.quantile(best_null,.95):.2f}**   最大 {best_null.max():.2f}")
print(f"  实际最高 lift:**{real_best:.2f}**({real_best_nm})")
print(f"  → {'**超出噪音上界**' if real_best > np.quantile(best_null,.95) else '**未超出噪音上界,不能算发现**'}")
print(f"  (对照:三十一节实测纯噪音 best-of-N lift 中位数 2.09;"
      f"第五十一节全场最高 lift 2.68)")

# ---------------- 纪律 B:FDR 校正 ----------------
print(f"\n{'='*112}\n纪律B Benjamini-Hochberg FDR 校正(α=0.05)\n{'='*112}")
names = list(res); pv = np.array([res[n]["p"] for n in names])
order = np.argsort(pv); m = len(pv)
crit = (np.arange(1, m + 1) / m) * 0.05
passed = pv[order] <= crit
kmax = np.max(np.flatnonzero(passed)) if passed.any() else -1
sig = {names[order[i]] for i in range(kmax + 1)} if kmax >= 0 else set()
print(f"{'特征':<32}{'lift':>8}{'p':>11}{'FDR':>8}{'超噪音上界':>12}")
for i in order:
    n = names[i]
    ok_fdr = "✓" if n in sig else "✗"
    ok_noise = "✓" if res[n]["lift"] > np.quantile(best_null, .95) else "✗"
    print(f"{n:<32}{res[n]['lift']:>8.2f}{res[n]['p']:>11.2e}{ok_fdr:>8}{ok_noise:>12}")

winners = [n for n in names if n in sig and res[n]["lift"] > np.quantile(best_null, .95)]
print(f"\n{'='*112}\n判定\n{'='*112}")
print(f"  同时通过 FDR 与噪音上界的特征:**{len(winners)} 个**")
for n in winners:
    s = res[n]
    print(f"    {n:<30} lift {s['lift']:.2f}   P(牛股|特征) {s['P(牛股|特征)']:.2%}   "
          f"覆盖牛股 {s['P(特征|牛股)']:.1%}   命中 {s['命中数']:,}")
if winners:
    bestw = max(winners, key=lambda n: res[n]["lift"])
    print(f"\n  **注意:归因 ≠ 可交易性。** 最强的 {bestw} 也只有 "
          f"P(牛股|特征) = {res[bestw]['P(牛股|特征)']:.1%},")
    print(f"  意味着按此买入 {1-res[bestw]['P(牛股|特征)']:.0%} 的时候买到的不是牛股。")

pd.DataFrame(res).T.to_csv(f"{SP}/bull_feature_scan.csv")
P.to_parquet(f"{SP}/bull_feature_panel.parquet")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: bull_feature_scan.csv, bull_feature_panel.parquet")
