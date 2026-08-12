"""用户的三个假设:牛股是否有基底形态、均线多头排列、CANSLIM 特征

═══ 三个假设 ═══
一、所有大牛股启动前(或启动后)都有平台调整/基底形态?能否特征化?
二、所有主升浪都是 10周线/20周线/60周线 均线多头排列?
三、13年牛股清单能否用 CANSLIM 指标特征化?

═══ 两个必须先说清的设计陷阱 ═══

**陷阱一:归因 ≠ 可交易性。**
本session已三次踩此坑(rmdd20 归因说深回撤出牛股,按此交易 OOS 亏 6.15%)。
所以每个特征都报两个数:
  **P(特征|牛股)** —— 用户的假设问的是这个
  **P(牛股|特征)** 与 lift —— 决定能不能用的是这个
若 95% 的牛股有基底、但 90% 的普通股也有,该特征没有信息量。

**陷阱二:假设二有同义反复风险。**
一只翻倍的股票在上涨途中 MA50 必然上穿 MA100/MA300 ——
"主升浪都是多头排列"≈"上涨的股票在上涨"。
所以**分别报「起涨点当天」与「主升浪期间」**,把差别显示出来。
只有前者有预测意义。

═══ 对照组设计(关键) ═══
不能拿牛股的起涨点去比非牛股的随机日。
做法:**对每只股票每一年,都用同一方法找"该年最大涨幅的起点" t\\*** ——
  t\\* = argmax over t of ( max(close[t..年末]) / close[t] − 1 )
牛股与非牛股在 t\\* 上测同样的特征,口径完全对称。

═══ 牛股定义 ═══
年涨幅 = 该年最后有效收盘 ÷ **上一年最后有效收盘** − 1 > 100%
(此口径与 DeepSeek 清单相关 1.0000,见三十三节)
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
Y0, Y1 = 2013, 2025

t0 = time.time()
d = {c: {} for c in ["close", "high", "low", "volume", "float_mv", "roe"]}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    try:
        x = pd.read_parquet(f, columns=list(d))
    except Exception:
        continue
    if x.empty:
        continue
    for c in d:
        d[c][k] = pd.to_numeric(x[c], errors="coerce")
CL = pd.DataFrame(d["close"]).sort_index(); CL.index = CL.index.tz_localize(None)


def al(k):
    f = pd.DataFrame(d[k]).sort_index(); f.index = f.index.tz_localize(None)
    return f.reindex(index=CL.index, columns=CL.columns)


HI, LO, VOL, MV, ROE = al("high"), al("low"), al("volume"), al("float_mv"), al("roe")
CL = CL.where(CL > 0)
idx = CL.index
A, Ha, La, Va, Mva, Ra = (CL.to_numpy(), HI.to_numpy(), LO.to_numpy(),
                          VOL.to_numpy(), MV.to_numpy(), ROE.to_numpy())
NT, NC = A.shape
codes = list(CL.columns)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
del d

# 干净的成长字段(build_clean_growth.py 产出)
try:
    CQ = pd.read_parquet(f"{SP}/clean_growth_c_qyoy.parquet").reindex(
        index=idx, columns=CL.columns).to_numpy()
    NT_TTM = pd.read_parquet(f"{SP}/clean_growth_ni_ttm_yoy.parquet").reindex(
        index=idx, columns=CL.columns).to_numpy()
    RV_TTM = pd.read_parquet(f"{SP}/clean_growth_rev_ttm_yoy.parquet").reindex(
        index=idx, columns=CL.columns).to_numpy()
    HAS_GROWTH = True
    print("已载入修正后的成长字段(当季同比 / TTM同比)")
except Exception as e:
    HAS_GROWTH = False
    print(f"**未找到修正后的成长字段,假设三的 C/A 分量将跳过**  ({e})")

MA50 = CL.rolling(50, min_periods=50).mean().to_numpy()      # 10周线
MA100 = CL.rolling(100, min_periods=100).mean().to_numpy()   # 20周线
MA300 = CL.rolling(300, min_periods=300).mean().to_numpy()   # 60周线
RPS250 = (CL.pct_change(250).rank(axis=1, pct=True) * 100).to_numpy()
TR = np.maximum(Ha - La, np.maximum(np.abs(Ha - np.roll(A, 1, 0)),
                                    np.abs(La - np.roll(A, 1, 0))))
ATR20 = pd.DataFrame(TR).rolling(20, min_periods=10).mean().to_numpy()
ATR60 = pd.DataFrame(TR).rolling(60, min_periods=30).mean().to_numpy()
VOL50 = pd.DataFrame(Va).rolling(50, min_periods=30).mean().to_numpy()
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
        p0, p1 = prev[-1], cur[-1]
        yr_ret = a[p1] / a[p0] - 1
        # 该年最大涨幅的起点 t*(牛股与非牛股同一口径)
        seg = cur
        fwd_max = np.maximum.accumulate(a[seg][::-1])[::-1]
        gains = fwd_max / a[seg] - 1
        k = int(np.argmax(gains))
        t = int(seg[k])
        if t < 310:
            continue
        w250, w120, w60, w20 = a[t - 250:t + 1], a[t - 120:t + 1], a[t - 60:t + 1], a[t - 20:t + 1]
        w250 = w250[np.isfinite(w250)]; w120 = w120[np.isfinite(w120)]
        w60 = w60[np.isfinite(w60)]; w20 = w20[np.isfinite(w20)]
        if w250.size < 150 or w60.size < 30:
            continue
        rec = {
            "code": cd, "year": y, "yr_ret": yr_ret, "bull": yr_ret > 1.0,
            "max_gain": gains[k], "t": t,
            # ---- 假设一:基底形态(全部在 t* 之前测量,无前视)----
            "base_range_60": (w60.max() - w60.min()) / w60.min(),
            "base_range_120": (w120.max() - w120.min()) / w120.min() if w120.size > 60 else np.nan,
            "vol_contract": ATR20[t, j] / ATR60[t, j] if np.isfinite(ATR60[t, j]) and ATR60[t, j] > 0 else np.nan,
            "dist_high_250": a[t] / w250.max() - 1,
            "pullback_depth": 1 - w120.min() / w250.max() if w120.size > 60 else np.nan,
            # ---- 假设二:均线多头排列(**起涨点当天**)----
            "ma_bull_before": bool(np.isfinite(MA300[t, j]) and MA50[t, j] > MA100[t, j]
                                   > MA300[t, j] and a[t] > MA50[t, j]),
            "above_ma50_before": bool(np.isfinite(MA50[t, j]) and a[t] > MA50[t, j]),
            # ---- 假设三:CANSLIM ----
            "L_rps250": RPS250[t, j],
            "N_dist_high": a[t] / w250.max() - 1,
            "S_mv_pct": np.nan, "S_volr": Va[t, j] / VOL50[t, j] if np.isfinite(VOL50[t, j]) and VOL50[t, j] > 0 else np.nan,
            "A_roe": Ra[t, j],
        }
        if HAS_GROWTH:
            rec["C_qyoy"] = CQ[t, j]
            rec["A_ni_ttm"] = NT_TTM[t, j]
            rec["A_rev_ttm"] = RV_TTM[t, j]
        # 主升浪期间的多头排列比例(用于展示同义反复)
        end = int(seg[-1])
        span = slice(t, end + 1)
        m = (np.isfinite(MA300[span, j]) & (MA50[span, j] > MA100[span, j])
             & (MA100[span, j] > MA300[span, j]) & (a[span] > MA50[span, j]))
        rec["ma_bull_during"] = float(m.mean()) if m.size else np.nan
        rows.append(rec)
    if (j + 1) % 1000 == 0:
        print(f"  已处理 {j+1:,} 只  ({time.time()-t0:.0f}s)")

P = pd.DataFrame(rows)
mvp = pd.DataFrame(Mva).rank(axis=1, pct=True).to_numpy()
_ci = {c: i for i, c in enumerate(codes)}          # codes.index() 是 O(n),6万行会很慢
P["S_mv_pct"] = mvp[P.t.to_numpy(int), P.code.map(_ci).to_numpy(int)]
print(f"\n样本 {len(P):,} 个(股票,年份),其中牛股 **{int(P.bull.sum()):,}** 只"
      f"({P.bull.mean():.2%})  ({time.time()-t0:.0f}s)")
print(f"逐年牛股数:{P[P.bull].groupby('year').size().to_dict()}")

BASE = P.bull.mean()


def report(name, mask_series, higher_is_feature=True):
    """对一个二元特征,同时报 P(特征|牛股) 与 P(牛股|特征)。"""
    m = mask_series.fillna(False).astype(bool)
    p_f_given_bull = m[P.bull].mean()
    p_f_given_non = m[~P.bull].mean()
    p_bull_given_f = P.bull[m].mean() if m.sum() > 0 else np.nan
    lift = p_bull_given_f / BASE if BASE > 0 else np.nan
    print(f"{name:<34}{p_f_given_bull:>12.1%}{p_f_given_non:>12.1%}"
          f"{p_bull_given_f:>14.2%}{lift:>8.2f}{int(m.sum()):>10,}")
    return {"特征": name, "P(特征|牛股)": p_f_given_bull, "P(特征|非牛股)": p_f_given_non,
            "P(牛股|特征)": p_bull_given_f, "lift": lift, "命中数": int(m.sum())}


print(f"\n{'#'*104}")
print(f"基准牛股率 = **{BASE:.2%}**   lift = P(牛股|特征) ÷ 基准")
print(f"{'#'*104}")
print(f"\n{'='*104}\n假设一:启动前是否有平台/基底形态\n{'='*104}")
print(f"{'特征':<34}{'P(特征|牛股)':>12}{'P(特征|非牛)':>12}{'**P(牛股|特征)**':>14}{'lift':>8}{'命中数':>10}")
out = []
out.append(report("60日振幅 < 30%(窄幅整理)", P.base_range_60 < 0.30))
out.append(report("60日振幅 < 50%", P.base_range_60 < 0.50))
out.append(report("120日振幅 < 40%", P.base_range_120 < 0.40))
out.append(report("波动收缩 ATR20/ATR60 < 0.8", P.vol_contract < 0.8))
out.append(report("距250日高点 > -15%(贴近前高)", P.dist_high_250 > -0.15))
out.append(report("前期回调深度 15~35%", (P.pullback_depth > 0.15) & (P.pullback_depth < 0.35)))
out.append(report("**窄幅+贴近前高(组合)**",
                  (P.base_range_60 < 0.30) & (P.dist_high_250 > -0.15)))

print(f"\n{'='*104}\n假设二:均线多头排列(10周/20周/60周 = MA50/MA100/MA300)\n{'='*104}")
print(f"{'特征':<34}{'P(特征|牛股)':>12}{'P(特征|非牛)':>12}{'**P(牛股|特征)**':>14}{'lift':>8}{'命中数':>10}")
out.append(report("**起涨点当天** 多头排列", P.ma_bull_before))
out.append(report("起涨点当天 站上MA50", P.above_ma50_before))
print(f"\n  【同义反复对照】主升浪**期间**多头排列的天数占比:")
print(f"    牛股 中位 **{P.ma_bull_during[P.bull].median():.1%}**   "
      f"非牛股 中位 {P.ma_bull_during[~P.bull].median():.1%}")
print(f"    → 期间的差异主要是『上涨的股票在上涨』,**没有预测意义**;"
      f"有意义的是上面那行『起涨点当天』")

print(f"\n{'='*104}\n假设三:CANSLIM 各分量\n{'='*104}")
print(f"{'特征':<34}{'P(特征|牛股)':>12}{'P(特征|非牛)':>12}{'**P(牛股|特征)**':>14}{'lift':>8}{'命中数':>10}")
if HAS_GROWTH:
    out.append(report("C 当季净利同比 ≥ 25%", P.C_qyoy >= 0.25))
    out.append(report("A TTM净利同比 ≥ 25%", P.A_ni_ttm >= 0.25))
    out.append(report("A ROE ≥ 17%", P.A_roe >= 17))
out.append(report("N 距250日高点 > -10%", P.N_dist_high > -0.10))
out.append(report("S 流通市值最小30%", P.S_mv_pct <= 0.30))
out.append(report("S 起涨日量比 ≥ 1.5", P.S_volr >= 1.5))
out.append(report("L RPS250 > 90", P.L_rps250 > 90))
if HAS_GROWTH:
    combo = ((P.C_qyoy >= 0.25) & (P.A_roe >= 17) & (P.L_rps250 > 90)
             & (P.N_dist_high > -0.10) & (P.S_volr >= 1.5))
    out.append(report("**CANSLIM 五项全中**", combo))

R = pd.DataFrame(out)
R.to_csv(f"{SP}/bull_pattern_hypotheses.csv", index=False)
P.to_parquet(f"{SP}/bull_pattern_panel.parquet")

print(f"\n{'='*104}\n判读\n{'='*104}")
best = R.loc[R["lift"].idxmax()]
print(f"  lift 最高的特征:**{best['特征']}**  lift {best['lift']:.2f}  "
      f"P(牛股|特征) {best['P(牛股|特征)']:.2%}  命中 {int(best['命中数']):,}")
print(f"  基准牛股率 {BASE:.2%}")
print(f"\n  **注意:lift 是归因指标,不是可交易性。**")
print(f"  即使 lift = 3,P(牛股|特征) 也只有 {BASE*3:.1%} —— "
      f"意味着按此买入,{1-BASE*3:.0%} 的时候买到的不是牛股。")
print(f"  本session已三次证明:归因强 ≠ 回测赚钱(rmdd20 归因显著,交易亏 6.15%)。")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: bull_pattern_hypotheses.csv, bull_pattern_panel.parquet")
