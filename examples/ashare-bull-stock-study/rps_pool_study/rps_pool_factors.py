"""RPS250>90 动量池内叠加二级因子 —— 池子里还能再挑吗?

═══ 为什么在这个池子里做 ═══
用户的原始发现(第四十三节)是「RPS>90 池内筛双增长,收益不低」,
而第四十六/五十二节证明:**同一个双增长过滤在全市场、小市值、高BP 三个池子里
全部失效或反向,只在动量池里有效**。DeepSeek 的 W 章用完全独立的数据/代码
得到同一结论(成长因子单独 +1.30%,放进趋势系统 +9.1pp/年)。

**所以「池内叠加」是有先验支持的方向,不是随便试。** 本脚本把这个先验推广:
除了双增长,池内还能叠加什么?

═══ 池子的定义 ═══
RPS250 = 250日收益率的横截面百分位 × 100,取 **> 90**(用户口径)。
第四十四节已对账:我重建的 RPS 与用户快照**相关 0.990、中位绝对差 0.07**,
所以用我自己的面板重建,覆盖 2014-2026 全程(用户快照只有 2024-2026)。

═══ 二级因子(事前锁定 8 个,方向事前定,不做方向搜索) ═══
  1 双增长        净利TTM同比>0 且 收入TTM同比>0   —— 用户的原始发现
  2 盈利加速+     当季同比环比上升 且 当季>25%      —— §53 lift 1.38(池外)
  3 高估值        BP 最低 30%                     —— §53 破净 lift 0.51、高估值 1.41
  4 破净          BP 最高 30%                     —— 反向对照,预期更差
  5 小市值        流通市值最小 50%                 —— §48 再平衡收益都在这
  6 抗跌 rmdd20   20日最大回撤最浅 50%            —— DeepSeek 近5年最强因子
  7 换手不冷      换手率分位 > 30%                —— §53「最低30%」lift 0.74
  8 现金流不差    OCF/净利 ≥ 0                    —— §53 lift 0.85(反向)
**每个因子单独叠加,不做多因子组合搜索** —— 组合搜索在 20 个因子上已经
被第五十三节证明过必然出假阳性。

═══ 组合构建(用第五十六节验过的结论) ═══
月度再平衡,持仓 20 只,**等额 与 inv_vol 两种都跑**
(§56 实测 inv_vol 8/8 胜出、回撤全部改善)。

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 年化 ≥ **池基线**(RPS>90 池不加任何筛选,同样 20 只等额)
  ② 年化 ≥ **+12.25%**(全市场等权月度,§48 口径)—— 打不过就没有存在意义
  ③ **300 次**同数量随机抽取对照,p < 0.05/8 = **0.00625**(Bonferroni)
     (§56 教训:20 次抽样定不了 p,同一个东西能跑出 0.150 和 0.000)
三条缺一不可。另**必须分 2015-05 前后两段各报一次** ——
§55 已证明七种买点在后一段全部失效,不分段等于自欺。

═══ 锚点(不过就停) ═══
全市场等权·月度(§48 口径,期末无价者剔除)= **+12.25%**。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST, SLOTS, SEED, N_RAND = 0.003, 20, 20260812, 300
START, CUT_DATE = "2014-06-30", "2015-05-22"

t0 = time.time()
d = {c: {} for c in ["open", "close", "float_mv", "is_st", "listed_days",
                     "volume", "turnover", "bp_correct", "operating_cash_flow",
                     "net_income"]}
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


OP, MV, ST, LD = al("open"), al("float_mv"), al("is_st"), al("listed_days")
VO, TURN, BP = al("volume"), al("turnover"), al("bp_correct")
OCF, NI = al("operating_cash_flow"), al("net_income")
CL = CL.where(CL > 0); OP = OP.where(OP > 0)
idx = CL.index
NT, NC = CL.shape
A, OPa = CL.to_numpy(float), OP.to_numpy(float)
MVa, STa, LDa = MV.to_numpy(float), ST.to_numpy(float), LD.to_numpy(float)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
del d

last_valid = np.full(NC, -1)
for j in range(NC):
    fv = np.flatnonzero(np.isfinite(A[:, j]))
    if fv.size:
        last_valid[j] = fv[-1]

# ── 因子 ──
RPS250 = (CL.pct_change(250).rank(axis=1, pct=True) * 100).to_numpy(float)
BP_PCT = BP.rank(axis=1, pct=True).to_numpy(float)
MV_PCT = MV.rank(axis=1, pct=True).to_numpy(float)
TURN_PCT = TURN.rolling(20, min_periods=10).mean().rank(axis=1, pct=True).to_numpy(float)
_r = CL.pct_change()
_cum = (1 + _r).rolling(20, min_periods=10).apply(lambda x: np.prod(x), raw=True)
RMDD20 = (CL / CL.rolling(20, min_periods=10).max() - 1).to_numpy(float)
RMDD_PCT = pd.DataFrame(RMDD20).rank(axis=1, pct=True).to_numpy(float)   # 越大=回撤越浅
OCF_NI = (OCF / NI.where(NI > 0)).to_numpy(float)

# 干净成长字段(§52 修正后)
NI_TTM = pd.read_parquet(f"{SP}/clean_growth_ni_ttm_yoy.parquet").reindex(
    index=idx, columns=CL.columns).to_numpy(float)
RV_TTM = pd.read_parquet(f"{SP}/clean_growth_rev_ttm_yoy.parquet").reindex(
    index=idx, columns=CL.columns).to_numpy(float)
CQ = pd.read_parquet(f"{SP}/clean_growth_c_qyoy.parquet").reindex(
    index=idx, columns=CL.columns).to_numpy(float)
ACCEL = CQ - np.roll(CQ, 63, axis=0); ACCEL[:63] = np.nan
assert np.isfinite(NI_TTM).mean() > 0.01, "成长字段几乎全空"
VOL20 = _r.rolling(20, min_periods=10).std().to_numpy(float)
print(f"因子就绪  ({time.time()-t0:.0f}s)")

s0 = idx.searchsorted(pd.Timestamp(START))
eN = NT - 1
CUT = idx.searchsorted(pd.Timestamp(CUT_DATE))
YRS = (idx[eN] - idx[s0]).days / 365.25
_ds = [x for x in CL.resample("ME").last().index if idx[s0] <= x <= idx[eN]]
RB = np.array(sorted({idx.searchsorted(x, side="right") - 1 for x in _ds} | {s0, eN}))
print(f"窗口 {idx[s0].date()} ~ {idx[eN].date()}  {len(RB)-1} 个调仓期")


def seg_ret(a, b, cols):
    p0, p1 = A[a, cols], A[b, cols].copy()
    for k in np.flatnonzero(~np.isfinite(p1)):
        lv = last_valid[cols[k]]
        p1[k] = A[lv, cols[k]] if lv >= a else np.nan
    r = p1 / p0 - 1
    return np.where(np.isfinite(r), r, 0.0)


# ══════════ 锚点 ══════════
eq = 1.0
for a, b in zip(RB[:-1], RB[1:]):
    ci = np.flatnonzero(np.isfinite(A[a]))
    ci = ci[np.isfinite(A[b, ci])]
    if ci.size >= 5:
        eq *= 1 + float(np.nanmean(A[b, ci] / A[a, ci] - 1))
EW = eq ** (1 / YRS) - 1
print(f"\n锚点:全市场等权·月度【§48口径】 **{EW:+.2%}**  (应 +12.25%)")
assert abs(EW - 0.1225) < 0.003, f"锚点对不上:{EW:+.4%}"
print("锚点通过")


def pool_at(t):
    """RPS250>90 且可投资(非ST、上市>250日、有价)。"""
    ok = (np.isfinite(A[t]) & (LDa[t] >= 250) & (STa[t] != 1)
          & np.isfinite(RPS250[t]) & (RPS250[t] > 90))
    return np.flatnonzero(ok)


FACTORS = {
    "【池基线】不筛": lambda t, c: np.ones(c.size, bool),
    "1 双增长(净利+收入 TTM同比>0)": lambda t, c: (NI_TTM[t, c] > 0) & (RV_TTM[t, c] > 0),
    "2 盈利加速 且 当季>25%": lambda t, c: (ACCEL[t, c] > 0) & (CQ[t, c] > 0.25),
    "3 高估值 BP最低30%": lambda t, c: BP_PCT[t, c] <= 0.30,
    "4 破净 BP最高30%": lambda t, c: BP_PCT[t, c] >= 0.70,
    "5 小市值 最小50%": lambda t, c: MV_PCT[t, c] <= 0.50,
    "6 抗跌 rmdd20最浅50%": lambda t, c: RMDD_PCT[t, c] >= 0.50,
    "7 换手不冷 分位>30%": lambda t, c: TURN_PCT[t, c] > 0.30,
    "8 现金流不差 OCF/NI≥0": lambda t, c: OCF_NI[t, c] >= 0,
}


def run(sel_fn, weight="eq", rng=None, n_pick=None, seg=None):
    """月度再平衡;返回 (年化, 每期选中数中位, 逐期净收益)。"""
    lo, hi = (s0, eN) if seg is None else seg
    rb = RB[(RB >= lo) & (RB <= hi)]
    eqv, cnts, rets = 1.0, [], []
    for a, b in zip(rb[:-1], rb[1:]):
        c = pool_at(a)
        if c.size < 5:
            continue
        m = sel_fn(a, c)
        m = np.where(np.isfinite(m.astype(float)), m, False).astype(bool)
        sel = c[m]
        if rng is not None:                       # 随机对照:池内随机抽同样多只
            k = n_pick if n_pick is not None else sel.size
            k = min(max(k, 1), c.size)
            sel = rng.choice(c, k, replace=False)
        if sel.size == 0:
            continue
        cnts.append(sel.size)
        take = sel[:SLOTS] if sel.size <= SLOTS else \
            sel[np.argsort(MVa[a, sel])[:SLOTS]]   # 超过仓位数时按小市值优先(与前几节一致)
        if weight == "inv_vol":
            v = VOL20[a, take].copy()
            good = np.isfinite(v) & (v > 0)
            v[~good] = np.median(v[good]) if good.any() else 0.02
            w = (1 / v) / (1 / v).sum()
        else:
            w = np.full(take.size, 1 / take.size)
        r = float(np.sum(w * seg_ret(a, b, take)))
        rets.append(r - 2 * COST)                  # 每期全换仓,双边成本
        eqv *= 1 + rets[-1]
    yrs = (idx[min(hi, eN)] - idx[lo]).days / 365.25
    return (eqv ** (1 / yrs) - 1 if eqv > 0 and yrs > 0 else -1.0,
            float(np.median(cnts)) if cnts else 0.0, np.array(rets))


print(f"\n{'='*118}\nRPS250>90 池内叠加二级因子(月度再平衡,{SLOTS}只)\n{'='*118}")
print(f"{'因子':<32}{'池内命中中位':>14}{'等额 年化':>12}{'inv_vol 年化':>14}"
      f"{'2015-05前':>12}{'2015-05后':>12}")
rows = []
for nm, fn in FACTORS.items():
    a_eq, cnt, _ = run(fn)
    a_iv, _, _ = run(fn, weight="inv_vol")
    a_pre, _, _ = run(fn, seg=(s0, CUT))
    a_post, _, _ = run(fn, seg=(CUT, eN))
    rows.append({"因子": nm, "命中中位": cnt, "等额": a_eq, "inv_vol": a_iv,
                 "前段": a_pre, "后段": a_post})
    print(f"{nm:<32}{cnt:>14.0f}{a_eq:>+12.2%}{a_iv:>+14.2%}"
          f"{a_pre:>+12.2%}{a_post:>+12.2%}   ({time.time()-t0:.0f}s)")

R = pd.DataFrame(rows)
BASE_ANN = float(R[R.因子.str.startswith("【池基线】")]["等额"].iloc[0])
BASE_POST = float(R[R.因子.str.startswith("【池基线】")]["后段"].iloc[0])
print(f"\n  池基线(不筛):全期 **{BASE_ANN:+.2%}**、后段 **{BASE_POST:+.2%}**")
print(f"  全市场等权基准:**{EW:+.2%}**")

# ══════════ 判据③:池内同数量随机 × 300 ══════════
print(f"\n{'='*118}\n判据③ 池内随机对照 × {N_RAND}(§56 教训:20 次定不了 p)\n{'='*118}")
ALPHA = 0.05 / (len(FACTORS) - 1)
print(f"  Bonferroni:8 个因子 → 需 **p < {ALPHA:.5f}**\n")
print(f"{'因子':<32}{'实际(全期)':>12}{'随机中位':>11}{'95%区间':>24}{'p':>9}"
      f"{'实际(后段)':>12}{'p(后段)':>10}")
for _, r in R.iterrows():
    if r["因子"].startswith("【池基线】"):
        continue
    k = int(round(r["命中中位"]))
    rng = np.random.default_rng(SEED)
    full = np.array([run(FACTORS[r["因子"]], rng=rng, n_pick=k)[0] for _ in range(N_RAND)])
    rng2 = np.random.default_rng(SEED + 1)
    post = np.array([run(FACTORS[r["因子"]], rng=rng2, n_pick=k, seg=(CUT, eN))[0]
                     for _ in range(N_RAND)])
    p1 = float((full >= r["等额"]).mean())
    p2 = float((post >= r["后段"]).mean())
    R.loc[R.因子 == r["因子"], "p_全期"] = p1
    R.loc[R.因子 == r["因子"], "p_后段"] = p2
    R.loc[R.因子 == r["因子"], "随机中位"] = float(np.median(full))
    q = np.quantile(full, [0.025, 0.975])
    print(f"{r['因子']:<32}{r['等额']:>+12.2%}{np.median(full):>+11.2%}"
          f"   [{q[0]:+.2%}, {q[1]:+.2%}]{p1:>9.4f}{r['后段']:>+12.2%}{p2:>10.4f}"
          f"   ({time.time()-t0:.0f}s)")

print(f"\n{'='*118}\n判定(三条判据,未放宽)\n{'='*118}")
for _, r in R.iterrows():
    if r["因子"].startswith("【池基线】"):
        continue
    c1 = r["等额"] >= BASE_ANN
    c2 = r["等额"] >= EW
    c3 = r.get("p_全期", 1) < ALPHA
    ok = c1 and c2 and c3
    print(f"  {r['因子']:<32} ①≥池基线 {'✓' if c1 else '✗'}  "
          f"②≥{EW:.2%} {'✓' if c2 else '✗'}  ③p<{ALPHA:.5f} {'✓' if c3 else '✗'}"
          f"   **{'算发现' if ok else '不算发现'}**")

R.to_csv(f"{SP}/rps_pool_factors.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: rps_pool_factors.csv")
