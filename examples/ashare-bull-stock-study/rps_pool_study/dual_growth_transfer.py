"""双增长过滤能不能迁移?—— 把它叠加到四种不同的选股逻辑上

═══ 为什么问这个 ═══
到四十五节为止,整轮研究里唯一在多个独立数据源、多个独立时段都成立的发现是:
**在动量股池内部,用"净利润同比>0 且 收入同比>0"过滤能稳定提升收益。**
证据:用户A池 +15.9pp、用户B池 +91.3pp、RPS阶梯 8/8 档位全部改善、
2017-2023 独立样本外 +4.7pp,置换 p=0.025 / 0.000。

**但它改善的是一个本身亏钱的池子**(样本外 -9.17% → -4.48%,
仍跑输等权 7.2pp)。所以真正该问的是:
**这个过滤器是动量池专属的,还是一个可迁移的通用过滤器?**

═══ 两个必须同时排除的替代解释 ═══

**替代解释一:它其实只是"别买亏损公司"。**
"净利润同比>0"隐含要求上年也有可比基数;而一个更简单的条件
"净利润为正"可能做了同样的事。所以必须**把 `净利润>0` 作为独立一档
并列测**。若它captures了大部分效果,结论就该改写成"避开亏损公司",
而不是"增长"。

**替代解释二:它只是规模/估值 β 的伪装。**
三十五、三十六节里那些"发现"最后都还原成了小市值 β。
所以要在**四种不同的选股逻辑**上分别测,而不是只在一种上测。

═══ 设计 ═══
四种基础选股(互不相同的逻辑) × 四种过滤:

  基础:  ALL   全市场(无选股)—— 最干净的检验
         SIZE  流通市值最小 20%(三十五节说它是唯一稳定的锚)
         VALUE BP 最高 20%(四十节说中性化后它最强,t=+6.64)
         MOM   RPS250>90(已知结论,作对照锚)

  过滤:  基线(不过滤)
         双增长   净利润同比>0 且 收入同比>0
         **净利润>0**  ← 替代解释一的对照
         非双增长  补集,关键对照

月频调仓、等权、次月开盘进出、单边0.3%按实际换手扣。
基准:全市场等权,按**自身再平衡换手**扣成本(四十三节订正过的口径)。

═══ 判据(事前写死) ═══
"可迁移"需同时满足:
  ① 「双增长 − 非双增长」的月度差 t值 > 2,在 **4 种基础逻辑中至少 3 种**成立
  ② 方向一致(全部为正)
  ③ 置换检验 p < 0.05
  ④ **且效果不能被「净利润>0」完全解释** ——
     若「净利润>0」的增量 ≥ 双增长增量的 80%,判为"避开亏损公司"而非"增长"

只在 MOM 上成立 → 动量池专属,不可迁移。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST = 0.003
N_PERM = 200
SEED = 20260811
START = "2014-06-01"          # 需 252 日同比 + 250 日 RPS

t0 = time.time()
COLS = ["open", "close", "float_mv", "ni_yoy_252", "revenue", "net_income", "bp_correct"]
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
OP = pd.DataFrame(d["open"]).sort_index(); OP.index = OP.index.tz_localize(None)


def _align(key):
    """先把各自的索引统一成 tz-naive 再按标签对齐。

    **踩过的坑**:直接 `pd.DataFrame(d[key]).reindex_like(OP)` 会全部变 NaN ——
    OP 的索引已 tz_localize(None),而源数据仍是 tz-aware,按标签一条都匹配不上。
    症状是除"只用OP的全市场基线"外所有池子都为空,且不报错。
    """
    f = pd.DataFrame(d[key]).sort_index()
    f.index = f.index.tz_localize(None)
    return f.reindex(index=OP.index, columns=OP.columns)


CL = _align("close")
MV = _align("float_mv")
NIY = _align("ni_yoy_252")
REV = _align("revenue")
NI = _align("net_income")
BP = _align("bp_correct")
OP = OP.where(OP > 0); CL = CL.where(CL > 0)
idx = OP.index
OPa, CLa = OP.to_numpy(), CL.to_numpy()
print(f"面板 {OP.shape}  {idx.min().date()} ~ {idx.max().date()}  ({time.time()-t0:.0f}s)")
for _n, _f in (("close", CL), ("float_mv", MV), ("ni_yoy_252", NIY),
               ("revenue", REV), ("net_income", NI), ("bp_correct", BP)):
    _r = _f.notna().mean().mean()
    assert _r > 0.01, f"{_n} 几乎全为 NaN(非空率 {_r:.4%})——对齐出错"
    print(f"  {_n:<14} 非空率 {_r:>6.1%}")
del d


# ═══ 成长字段:改用修正后的口径(见 build_clean_growth.py 与第五十二节) ═══
# 原 `ni_yoy_252` = net_income/net_income.shift(252)-1,而 net_income 是 YTD 累计,
# 252 交易日 ≈ 1.04 年会跨报告期 —— 茅台 2023-05-04 得 -60.4%(单季比全年)。
# 污染量化:横截面「>0比例」按月份极差 25.1pp。
# 修正后(去累计 + 报告期对齐,已对官方财报核验 9 个单季全部吻合)极差 2.5~2.8pp。
def _load_clean_growth(index, columns):
    ni = pd.read_parquet(f"{SP}/clean_growth_ni_ttm_yoy.parquet").reindex(
        index=index, columns=columns)
    rv = pd.read_parquet(f"{SP}/clean_growth_rev_ttm_yoy.parquet").reindex(
        index=index, columns=columns)
    assert ni.notna().mean().mean() > 0.01, "clean_growth 净利字段几乎全空"
    assert rv.notna().mean().mean() > 0.01, "clean_growth 收入字段几乎全空"
    print(f"  成长字段 = **修正后 TTM 同比**(净利非空 {ni.notna().mean().mean():.1%}、"
          f"收入非空 {rv.notna().mean().mean():.1%})")
    return ni, rv

NIY, REVY = _load_clean_growth(OP.index, OP.columns)
RPS250 = CL.pct_change(250).rank(axis=1, pct=True) * 100
MVpct = MV.rank(axis=1, pct=True)
BPpct = BP.rank(axis=1, pct=True)
print(f"因子就绪  ({time.time()-t0:.0f}s)")

month_ends = [d_ for d_ in OP.resample("ME").last().index if d_ >= pd.Timestamp(START)]
month_ends = [d_ for d_ in month_ends if idx[0] <= d_ <= idx[-1]]
print(f"月度调仓 {len(month_ends)} 期  {month_ends[0].date()} ~ {month_ends[-1].date()}")

BASES = ["ALL 全市场", "SIZE 小市值20%", "VALUE 高BP20%", "MOM RPS250>90"]
FILTS = ["基线", "双增长", "净利润>0", "非双增长"]


def mean_ret(mask_arr, e, x):
    """等权组合的期间收益(次日开盘进出;期间停牌退到最后有效收盘)。"""
    ci = np.flatnonzero(mask_arr)
    if ci.size == 0:
        return np.nan, 0
    a = OPa[e, ci]; b = OPa[x, ci]
    good = np.isfinite(a) & (a > 0)
    ci, a, b = ci[good], a[good], b[good]
    bad = ~(np.isfinite(b) & (b > 0))
    for j in np.flatnonzero(bad):
        seg = CLa[e:x + 1, ci[j]]; seg = seg[np.isfinite(seg)]
        b[j] = seg[-1] if seg.size else np.nan
    r = b / a - 1
    r = r[np.isfinite(r)]
    return (r.mean() if r.size else np.nan), r.size


rows = []
prev = {}
CACHE = []
for i in range(len(month_ends) - 1):
    e = idx.searchsorted(month_ends[i], side="right")
    x = idx.searchsorted(month_ends[i + 1], side="right")
    if e >= len(idx) or x >= len(idx) or x <= e:
        continue
    p = e - 1
    alive = (np.isfinite(OP.iloc[e]) & np.isfinite(OP.iloc[x])).to_numpy()
    if alive.sum() < 500:
        continue
    niy = (NIY.iloc[p] > 0).fillna(False).to_numpy()
    revy = (REVY.iloc[p] > 0).fillna(False).to_numpy()
    dual = niy & revy
    nipos = (NI.iloc[p] > 0).fillna(False).to_numpy()
    base_masks = {
        "ALL 全市场": alive,
        "SIZE 小市值20%": alive & (MVpct.iloc[p] <= 0.2).fillna(False).to_numpy(),
        "VALUE 高BP20%": alive & (BPpct.iloc[p] >= 0.8).fillna(False).to_numpy(),
        "MOM RPS250>90": alive & (RPS250.iloc[p] > 90).fillna(False).to_numpy(),
    }
    rec = {"month": month_ends[i], "e": e, "x": x, "days": x - e}
    # 缓存基础掩码,供置换检验复用(否则每次置换都要重建,慢一个数量级)
    CACHE.append({"e": e, "x": x, "masks": {k: v.copy() for k, v in base_masks.items()},
                  "dual": dual.copy()})
    # 基准:全市场等权,按自身再平衡换手扣成本
    br, _ = mean_ret(alive, e, x)
    ci = np.flatnonzero(alive)
    rr = OPa[x, ci] / OPa[e, ci] - 1
    rr = rr[np.isfinite(rr)]
    w = (1 + rr) / (1 + rr).sum()
    rec["bench"] = br - 2 * COST * (0.5 * np.abs(w - 1 / len(w)).sum())

    for bn, bm in base_masks.items():
        for fn in FILTS:
            m = bm.copy()
            if fn == "双增长":
                m &= dual
            elif fn == "净利润>0":
                m &= nipos
            elif fn == "非双增长":
                m &= ~dual
            r, n = mean_ret(m, e, x)
            key = f"{bn}|{fn}"
            codes = set(np.flatnonzero(m))
            pv = prev.get(key, set())
            to = 1.0 if not pv else 1 - len(pv & codes) / max(len(codes), 1)
            prev[key] = codes
            rec[f"ret|{key}"] = r
            rec[f"n|{key}"] = n
            rec[f"to|{key}"] = to
        # 当期收益(检验机械相关):上月末→本月末
        pe = idx.searchsorted(month_ends[i - 1], side="right") if i > 0 else None
        if pe is not None and pe < e:
            rec[f"cur|{bn}"] = mean_ret(bm & dual, pe, e)[0]
    rows.append(rec)

P = pd.DataFrame(rows)
print(f"有效月份 {len(P)}  ({time.time()-t0:.0f}s)")


def comp(r, dd):
    r = np.asarray(r, float); dd = np.asarray(dd, float)
    ok = np.isfinite(r) & np.isfinite(dd)
    if ok.sum() == 0:
        return np.nan
    t = np.prod(1 + r[ok]); y = dd[ok].sum() / 252
    return t ** (1 / y) - 1 if t > 0 and y > 0 else -1.0


bench_ann = comp(P["bench"], P["days"])
print(f"\n{'#'*118}")
print(f"全市场等权基准(按自身再平衡换手扣成本):年化 **{bench_ann:+.2%}**"
      f"   {P.month.min().date()} ~ {P.month.max().date()}")
print(f"{'#'*118}")

print(f"\n{'='*118}\n主表:四种选股逻辑 × 四种过滤(年化,已扣成本)\n{'='*118}")
print(f"{'选股逻辑':<18}{'过滤':<12}{'每期只数':>9}{'换手':>8}{'年化':>11}"
      f"{'相对基准':>11}{'相对本逻辑基线':>14}")
ann = {}
for bn in BASES:
    b0 = None
    for fn in FILTS:
        key = f"{bn}|{fn}"
        net = P[f"ret|{key}"] - 2 * COST * P[f"to|{key}"]
        a = comp(net, P["days"])
        ann[key] = a
        if fn == "基线":
            b0 = a
        inc = "" if fn == "基线" else f"{(a-b0)*100:>+13.1f}pp"
        print(f"{bn:<18}{fn:<12}{P[f'n|{key}'].median():>9.0f}"
              f"{P[f'to|{key}'].median():>8.1%}{a:>+11.2%}{(a-bench_ann)*100:>+10.1f}pp{inc:>14}")
    print()

print(f"{'='*118}\n判据①②:「双增长 − 非双增长」的月度差(逐月不重叠)\n{'='*118}")
print(f"{'选股逻辑':<18}{'月均差':>10}{'月数':>7}{'t值':>9}{'胜率':>9}{'当期差(机械相关对照)':>22}")
tstats = {}
for bn in BASES:
    dd = (P[f"ret|{bn}|双增长"] - P[f"ret|{bn}|非双增长"]).dropna()
    t = dd.mean() / dd.std() * np.sqrt(len(dd)) if dd.std() > 0 else np.nan
    tstats[bn] = t
    cur = P.get(f"cur|{bn}")
    cs = f"{cur.mean():+.3%}" if cur is not None and cur.notna().any() else "—"
    print(f"{bn:<18}{dd.mean():>+10.3%}{len(dd):>7}{t:>+9.2f}{(dd>0).mean():>9.1%}{cs:>22}"
          f"{'  **' if abs(t) > 2 else ''}")

print(f"\n{'='*118}\n判据④:效果能否被「净利润>0」解释\n{'='*118}")
print(f"{'选股逻辑':<18}{'双增长增量':>13}{'净利润>0增量':>15}{'后者占比':>11}{'判定':>22}")
for bn in BASES:
    g_dual = ann[f"{bn}|双增长"] - ann[f"{bn}|基线"]
    g_ni = ann[f"{bn}|净利润>0"] - ann[f"{bn}|基线"]
    share = g_ni / g_dual if abs(g_dual) > 1e-9 else np.nan
    verdict = ("**被'不亏损'解释**" if np.isfinite(share) and share >= 0.8
               else ("增长本身有额外贡献" if g_dual > 0 else "双增长无正增量"))
    print(f"{bn:<18}{g_dual*100:>+12.1f}pp{g_ni*100:>+14.1f}pp"
          f"{share:>11.0%}{verdict:>22}" if np.isfinite(share) else
          f"{bn:<18}{g_dual*100:>+12.1f}pp{g_ni*100:>+14.1f}pp{'—':>11}{verdict:>22}")

print(f"\n{'='*118}\n判据③:置换检验(每月在该逻辑池内打乱双增长标签 {N_PERM} 次)\n{'='*118}")
rng = np.random.default_rng(SEED)
pvals = {}
for bn in BASES:
    real = (P[f"ret|{bn}|双增长"] - P[f"ret|{bn}|非双增长"]).dropna().mean()
    # 预先取出该逻辑每月的 (e, x, 池内下标, 双增长只数),置换时只打乱标签
    slots = []
    for c in CACHE:
        ci = np.flatnonzero(c["masks"][bn])
        if ci.size < 20:
            continue
        k = int((c["masks"][bn] & c["dual"]).sum())
        if k < 1 or k >= ci.size:
            continue
        slots.append((c["e"], c["x"], ci, k))
    null = []
    for _ in range(N_PERM):
        diffs = []
        for e, x, ci, k in slots:
            perm = rng.permutation(ci.size)
            m1 = np.zeros(len(OP.columns), bool); m1[ci[perm[:k]]] = True
            m2 = np.zeros(len(OP.columns), bool); m2[ci[perm[k:]]] = True
            a1 = mean_ret(m1, e, x)[0]; a2 = mean_ret(m2, e, x)[0]
            if np.isfinite(a1) and np.isfinite(a2):
                diffs.append(a1 - a2)
        null.append(np.mean(diffs) if diffs else np.nan)
    null = np.array([v for v in null if np.isfinite(v)])
    p = float((np.abs(null) >= abs(real)).mean())
    pvals[bn] = p
    print(f"  {bn:<18} 真实月均差 {real:+.3%}   纯噪音 2.5%~97.5% "
          f"[{np.quantile(null,.025):+.3%}, {np.quantile(null,.975):+.3%}]   "
          f"**p={p:.3f}**  {'显著' if p < 0.05 else '与噪音不可区分'}   ({time.time()-t0:.0f}s)")

print(f"\n{'='*118}\n分段(按调仓月份)\n{'='*118}")
print(f"{'选股逻辑':<18}{'2014-2017':>14}{'2018-2021':>14}{'2022-2026':>14}   (双增长−非双增长 月均差)")
for bn in BASES:
    out = []
    for y0, y1 in ((2014, 2017), (2018, 2021), (2022, 2026)):
        g = P[(P.month.dt.year >= y0) & (P.month.dt.year <= y1)]
        dd = (g[f"ret|{bn}|双增长"] - g[f"ret|{bn}|非双增长"]).dropna()
        out.append(f"{dd.mean():+.3%}" if len(dd) > 6 else "—")
    print(f"{bn:<18}{out[0]:>14}{out[1]:>14}{out[2]:>14}")

print(f"\n{'='*118}\n最终判定(事前写死)\n{'='*118}")
ok1 = sum(1 for bn in BASES if tstats[bn] > 2)
ok2 = all(tstats[bn] > 0 for bn in BASES)
ok3 = sum(1 for bn in BASES if pvals[bn] < 0.05)
print(f"  ① t>2 的逻辑数:{ok1}/4  {'✓' if ok1 >= 3 else '✗(需≥3)'}")
print(f"  ② 方向全部为正:{'✓' if ok2 else '✗'}")
print(f"  ③ 置换 p<0.05 的逻辑数:{ok3}/4")
print(f"\n  **{'可迁移的通用过滤器' if (ok1 >= 3 and ok2 and ok3 >= 3) else '不满足可迁移判据'}**")

P.to_csv(f"{SP}/dual_growth_transfer.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: dual_growth_transfer.csv")
