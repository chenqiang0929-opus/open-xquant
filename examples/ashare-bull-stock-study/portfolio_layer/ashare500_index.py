"""测试1:造一个「A股的标普500」—— 这是指数设计问题,不是选股问题

═══ 为什么换这个方向 ═══
前 55 节全在优化**选股信号**,结果是:我造的每一个策略都跑输
「不选股、只做规则化再平衡」的全市场等权(+11.88%/年)。
第四十七节自己算出过原因:**月换手 95% × 0.3% ≈ 7.1pp/年**。
一边找 lift 1.3 的因子,一边在成本上白送 7 个点。

标普500 能复利靠的是三个机械特征,和选股无关:
  1. **市值加权** —— 赢家自动变重、输家自动变轻,「让赢家跑」是免费的;
     等权组合每次再平衡都在**卖赢家买输家**,要付出换手成本
  2. **年换手 ~4%** —— 成本几乎为零
  3. **规则化成分调整** —— 一年动 20-25 家

**本脚本把这三件事变成可调参数,全部测一遍。** 这四个维度在前 55 节一个都没测过。

═══ 一个必须先说清的结构差异 ═══
但斌那段话的核心是「年年末位淘汰」。但 **A股退市率 0.48%/年**
(第四十二节实测,DeepSeek N 章独立测得同一量级),标普500 每年换 4-5%。
**A股的淘汰机制弱一个数量级** —— 这是制度差异,不是设计能补的。
所以本脚本测的是「在这个约束下能做到多好」,不是「能不能复制标普500」。

═══ 测试维度(全部事前锁定) ═══
  成分数     N ∈ {300, 500}
  加权       市值加权 / 等权 / inv_vol(20日波动率倒数)
  缓冲区     无(每次严格取前N) / 有(进 N×0.8 名,出 N×1.2 名)
  调整频率   季度 / 年度
共 2×3×2×2 = 24 格。**这是一次搜索**,最好的那格要和「同换手随机成分」对照。

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 年化 ≥ **+11.88%**(全市场等权月度基准)
  ② 年换手 ≤ **15%**
  ③ 最好的那格要优于「随机抽同样多只股票、同样规则」的 20 次分布(p<0.05)
三条缺一不可。

═══ 锚点(不过就停) ═══
全市场等权·月度再平衡(无成本)= **+11.88%**;510300 = **+8.33%**;
买入持有 = **+6.77%**。三个都要在本脚本里复现出来。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST = 0.003          # 单边;成本 = Σ|Δw| × COST(买卖各 0.3%)
SEED, N_RAND = 20260812, 20
START = "2014-06-30"  # 与第四十八节基准同窗口(否则三个锚点全对不上)

t0 = time.time()
d = {c: {} for c in ["close", "float_mv", "is_st", "listed_days", "volume"]}
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


MV, ST, LD, VO = al("float_mv"), al("is_st"), al("listed_days"), al("volume")
CL = CL.where(CL > 0)
idx = CL.index
NT, NC = CL.shape
A = CL.to_numpy(float)
MVa, STa, LDa = MV.to_numpy(float), ST.to_numpy(float), LD.to_numpy(float)
# 流动性:20日均成交额(用 close×volume 近似)
AMT20 = (CL * VO).rolling(20, min_periods=10).mean().to_numpy(float)
VOL20 = CL.pct_change().rolling(20, min_periods=10).std().to_numpy(float)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
del d

# 最后一个有效价格的位置(退市后按最后有效价结算,不跳过)
last_valid = np.full(NC, -1)
for j in range(NC):
    f = np.flatnonzero(np.isfinite(A[:, j]))
    if f.size:
        last_valid[j] = f[-1]


def seg_ret(t0_, t1_, cols):
    """[t0_, t1_] 的个股收益;中途永久终止的按最后有效价结算。"""
    p0 = A[t0_, cols]
    p1 = A[t1_, cols].copy()
    bad = ~np.isfinite(p1)
    if bad.any():
        for k in np.flatnonzero(bad):
            lv = last_valid[cols[k]]
            p1[k] = A[lv, cols[k]] if lv >= 0 and lv >= t0_ else np.nan
    r = p1 / p0 - 1
    return np.where(np.isfinite(r), r, 0.0)      # 仍不可得 → 视作持平


def path(t0_, t1_, cols):
    """[t0_, t1_] 的归一化价格路径(用于日度净值),NaN 前值填充。"""
    seg = A[t0_:t1_ + 1, cols]
    seg = pd.DataFrame(seg).ffill().to_numpy()
    p0 = seg[0]
    out = seg / p0
    return np.where(np.isfinite(out), out, 1.0)


s0 = idx.searchsorted(pd.Timestamp(START))
eN = NT - 1
YRS = (idx[eN] - idx[s0]).days / 365.25


def rebalance_dates(freq):
    fmap = {"M": "ME", "Q": "QE", "Y": "YE"}
    ds = [x for x in CL.resample(fmap[freq]).last().index if idx[s0] <= x <= idx[eN]]
    ps = sorted({idx.searchsorted(x, side="right") - 1 for x in ds} | {s0, eN})
    return np.array([p for p in ps if s0 <= p <= eN])


def eligible(t, need_days=250, min_amt_pct=0.20):
    ok = np.isfinite(A[t]) & (LDa[t] >= need_days) & (STa[t] != 1)
    amt = AMT20[t].copy()
    if np.isfinite(amt).sum() > 100:
        thr = np.nanquantile(amt[ok] if ok.sum() > 100 else amt, min_amt_pct)
        ok &= (amt >= thr)
    return np.flatnonzero(ok & np.isfinite(MVa[t]))


def build(N, weight, buffer_on, freq, universe="mv", seed=None):
    """返回 (日度净值Series, 年换手, 年均成分变动数)。"""
    rb = rebalance_dates(freq)
    rng = np.random.default_rng(seed) if seed is not None else None
    eq = [1.0]
    eq_idx = [idx[rb[0]]]
    members, w_prev = np.array([], int), np.array([])
    turn_sum, n_rb, chg_sum = 0.0, 0, 0
    for i in range(len(rb) - 1):
        t, t2 = rb[i], rb[i + 1]
        elig = eligible(t)
        if elig.size < N:
            continue
        if universe == "random" and rng is not None:
            new = rng.choice(elig, N, replace=False)
        else:
            order = elig[np.argsort(-MVa[t, elig])]      # 市值降序
            if buffer_on and members.size:
                keep_rank = {c: k for k, c in enumerate(order)}
                keep = np.array([c for c in members
                                 if keep_rank.get(c, 10**9) < int(N * 1.2)], int)
                add_pool = [c for c in order[:int(N * 0.8)] if c not in set(keep)]
                new = np.concatenate([keep, np.array(add_pool[:max(0, N - keep.size)], int)])
                if new.size < N:
                    extra = [c for c in order if c not in set(new)]
                    new = np.concatenate([new, np.array(extra[:N - new.size], int)])
            else:
                new = order[:N]
        new = new.astype(int)
        # 权重
        if weight == "mv":
            wv = MVa[t, new].copy(); wv[~np.isfinite(wv)] = np.nanmedian(wv)
        elif weight == "inv_vol":
            v = VOL20[t, new].copy()
            v[~np.isfinite(v) | (v <= 0)] = np.nanmedian(v[np.isfinite(v) & (v > 0)])
            wv = 1.0 / v
        else:
            wv = np.ones(new.size)
        w = wv / wv.sum()
        # 换手:与上一期漂移后的权重比
        if members.size:
            all_c = np.union1d(members, new)
            a = pd.Series(w_prev, index=members).reindex(all_c).fillna(0.0).to_numpy()
            b = pd.Series(w, index=new).reindex(all_c).fillna(0.0).to_numpy()
            to = np.abs(b - a).sum() / 2.0
            chg_sum += len(np.setdiff1d(new, members))
        else:
            to = 1.0
        turn_sum += to; n_rb += 1
        # 日度净值
        P = path(t, t2, new)
        vals = P @ w
        vals = vals / vals[0] * (1 - 2 * to * COST)
        base = eq[-1]
        eq.extend((base * vals[1:]).tolist())
        eq_idx.extend(idx[t + 1:t2 + 1].tolist())
        # 漂移后权重
        r = seg_ret(t, t2, new)
        wd = w * (1 + r)
        s = wd.sum()
        w_prev = wd / s if np.isfinite(s) and s > 0 else w
        members = new
    E = pd.Series(eq, index=pd.DatetimeIndex(eq_idx))
    yrs = (E.index[-1] - E.index[0]).days / 365.25
    per_year = n_rb / yrs
    return E, turn_sum / n_rb * per_year, chg_sum / yrs


def stats(E):
    r = E.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    yrs = (E.index[-1] - E.index[0]).days / 365.25
    return {"年化": (E.iloc[-1] / E.iloc[0]) ** (1 / yrs) - 1 if E.iloc[-1] > 0 else -1.0,
            "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "最大回撤": (E / E.cummax() - 1).min()}


# ══════════ 锚点 ══════════
# **首版三个锚点全没过**(等权 +17.03% vs +11.88%、510300 +6.41% vs +8.33%、
# 买入持有 +8.36% vs +6.77%)。查出三处口径差异:
#   1. 窗口不是 2014-06-30 起,我从面板第 250 天(≈2014-01)就开始了
#   2. 买入持有的成分**固定在起点**,再平衡的成分是**每期动态**的
#      (第四十八节表里「每期只数 3,648」是动态成分的**每期均值**,不是起点股票数)
#   3. 第四十八节对**期末无价的股票直接剔除**,我是按最后有效价结算
# 第 3 条影响 2.1pp(+14.38% vs +12.25%)—— 剔除等于把退市那批排除在外,
# 而第四十二节实测退市/长停那批的隐含收益是 **+103%**,所以剔除是**保守**方向。
print(f"\n{'='*112}\n锚点(不过就停)\n{'='*112}")
print(f"  窗口 {idx[s0].date()} ~ {idx[eN].date()}  ({YRS:.1f} 年)")

U0 = np.flatnonzero(np.isfinite(A[s0]))
bh_ann = (1 + seg_ret(s0, eN, U0).mean()) ** (1 / YRS) - 1
print(f"  买入持有(成分固定在起点,{U0.size:,} 只)   **{bh_ann:+.2%}**   (应 +6.77%)")

rbM = rebalance_dates("M")
eq48, eq_settle, cnt = 1.0, 1.0, []
for a, b in zip(rbM[:-1], rbM[1:]):
    ci = np.flatnonzero(np.isfinite(A[a]))
    cnt.append(ci.size)
    ci48 = ci[np.isfinite(A[b, ci])]                 # §48 口径:期末无价直接剔除
    if ci48.size >= 5:
        eq48 *= 1 + float(np.nanmean(A[b, ci48] / A[a, ci48] - 1))
    eq_settle *= 1 + seg_ret(a, b, ci).mean()        # 本脚本口径:按最后有效价结算
ew48 = eq48 ** (1 / YRS) - 1
BENCH = eq_settle ** (1 / YRS) - 1
print(f"  全市场等权·月度【§48口径】(每期 {np.median(cnt):.0f} 只)  **{ew48:+.2%}**   (应 +12.25%)")
print(f"  全市场等权·月度【本脚本口径,退市按最后有效价结算】  **{BENCH:+.2%}**")

_m = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                   errors="coerce")
_m.index = _m.index.tz_localize(None)
mk = _m.reindex(idx).ffill()
mk_ann = (mk.iloc[eN] / mk.iloc[s0]) ** (1 / YRS) - 1
print(f"  510300                                     **{mk_ann:+.2%}**   (应 +8.33%)")

assert abs(bh_ann - 0.0677) < 0.002, f"买入持有锚点对不上:{bh_ann:+.4%}"
assert abs(ew48 - 0.1225) < 0.003, f"§48 等权锚点对不上:{ew48:+.4%}"
assert abs(mk_ann - 0.0833) < 0.002, f"510300 锚点对不上:{mk_ann:+.4%}"
print("  三个锚点全部通过 —— 引擎与第四十八节一致")
print(f"\n  **判据①的对照基准**:事前写的是 +11.88%(取自第四十七节),")
print(f"  但与本脚本同口径的等权基准是 **{BENCH:+.2%}** —— **两条线都报,以同口径的为准**")

# ══════════ 24 格 ══════════
print(f"\n{'='*112}\n「A股500」24 格(成分数 × 加权 × 缓冲区 × 频率)\n{'='*112}")
print(f"{'配置':<44}{'年化':>9}{'Sharpe':>8}{'最大回撤':>10}{'年换手':>9}{'年均换入':>9}")
rows = []
for N in (300, 500):
    for wt, wn in (("mv", "市值加权"), ("eq", "等权"), ("inv_vol", "inv_vol")):
        for buf in (False, True):
            for fq, fn in (("Q", "季度"), ("Y", "年度")):
                nm = f"N={N} {wn:<8} {'有缓冲' if buf else '无缓冲'} {fn}"
                E, turn, chg = build(N, wt, buf, fq)
                s = stats(E)
                rows.append({"配置": nm, "N": N, "加权": wn, "缓冲": buf,
                             "频率": fn, **s, "年换手": turn, "年均换入": chg})
                print(f"{nm:<44}{s['年化']:>+9.2%}{s['Sharpe']:>8.3f}"
                      f"{s['最大回撤']:>10.1%}{turn:>9.1%}{chg:>9.0f}"
                      f"   ({time.time()-t0:.0f}s)")

R = pd.DataFrame(rows)
ok = R[(R["年化"] >= BENCH) & (R["年换手"] <= 0.15)]
print(f"\n  同时满足 ①年化≥{BENCH:.2%}(同口径基准) 与 ②年换手≤15% 的配置:**{len(ok)} 个**")
ok2 = R[(R["年化"] >= 0.1188) & (R["年换手"] <= 0.15)]
print(f"  (若按事前写的 +11.88%:**{len(ok2)} 个**)")
for _, r in ok.iterrows():
    print(f"    {r['配置']:<44}{r['年化']:>+9.2%}  换手 {r['年换手']:>6.1%}")

best = R.loc[R["年化"].idxmax()]
print(f"\n  年化最高:**{best['配置']}**  {best['年化']:+.2%}  "
      f"Sharpe {best['Sharpe']:.3f}  回撤 {best['最大回撤']:.1%}  换手 {best['年换手']:.1%}")

# ══════════ 判据③:同规则随机成分 ══════════
print(f"\n{'='*112}\n判据③ 随机对照:同样的加权/缓冲/频率,但成分随机抽 × {N_RAND} 次\n{'='*112}")
wmap = {"市值加权": "mv", "等权": "eq", "inv_vol": "inv_vol"}
anns = []
for s_ in range(N_RAND):
    E, _, _ = build(int(best["N"]), wmap[best["加权"]], bool(best["缓冲"]),
                    "Q" if best["频率"] == "季度" else "Y",
                    universe="random", seed=SEED + s_)
    anns.append(stats(E)["年化"])
anns = np.array(anns)
p = float((anns >= best["年化"]).mean())
print(f"  实际 **{best['年化']:+.2%}**   随机中位 {np.median(anns):+.2%}"
      f"   [{anns.min():+.2%}, {anns.max():+.2%}]   **p={p:.3f}**")

print(f"\n{'='*112}\n判定(事前判据,未放宽)\n{'='*112}")
c1 = best["年化"] >= BENCH
c2 = best["年换手"] <= 0.15
c3 = p < 0.05
print(f"  ① 年化 ≥ {BENCH:+.2%}(同口径)  →  {best['年化']:+.2%}  {'✓' if c1 else '✗'}")
print(f"     (事前写的 +11.88% → {'✓' if best['年化'] >= 0.1188 else '✗'})")
print(f"  ② 年换手 ≤ 15%       →  {best['年换手']:.1%}  {'✓' if c2 else '✗'}")
print(f"  ③ 优于随机成分(p<0.05) →  p={p:.3f}  {'✓' if c3 else '✗'}")
print(f"  **{'算发现' if (c1 and c2 and c3) else '不算发现'}**")

R.to_csv(f"{SP}/ashare500_index.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: ashare500_index.csv")
