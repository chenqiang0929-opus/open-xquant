"""成长是"排序因子"还是"过滤器"—— 调和本session与 DeepSeek 报告的表面矛盾

═══ 分歧 ═══
DeepSeek(H 章):`growth_comp`(净利+营收**增速**)近5年年化 **1.30%**、
  Sharpe 0.183、回撤 -54.2%,"加入任何组合都是拖累"。
本session(四十六节):双增长(净利同比 **>0** 且 收入同比 **>0**)
  作为**二元过滤**,动量池 +4.7pp(t=+3.72, p=0.000)、高BP池 +3.3pp(p=0.010)。

**两者测的可能根本不是一回事**:他们买"增速最高的",我们只是"剔除负增长的"。
四十六节判据④已给旁证:单纯"净利润>0"只解释 19~25% 的效果,
说明价值可能集中在分布**低端**(排除收缩的公司),而不是**高端**(买最快的)。

═══ 用五分档一次讲清 ═══
成长综合分 = zscore(净利润同比) + zscore(收入同比),月频等权。
在**全市场**与**高BP20%**两个基础池上分别做,避免只在一种池子上下结论。

═══ 事前判据 ═══
  Q1 显著差于 Q2~Q5(≥3pp),且 Q2~Q5 极差 ≤2pp
     → 价值在"排除负增长"不在"买高增长" → **两份报告可完全调和**
  Q5 > Q4 > Q3 > Q2 单调
     → 是排序因子,DeepSeek 口径没问题,本session"过滤"的表述需修正
  无规律 → 两者都不成立,四十六节结论需降级
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST = 0.003
START = "2014-06-01"

t0 = time.time()
COLS = ["open", "close", "ni_yoy_252", "revenue", "bp_correct"]
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
    """索引统一成 tz-naive 再对齐 —— 四十六节踩过的坑:直接 reindex_like 会全 NaN 且不报错。"""
    f = pd.DataFrame(d[key]).sort_index()
    f.index = f.index.tz_localize(None)
    return f.reindex(index=OP.index, columns=OP.columns)


CL = _align("close"); NIY = _align("ni_yoy_252")
REV = _align("revenue"); BP = _align("bp_correct")
OP = OP.where(OP > 0); CL = CL.where(CL > 0)
idx = OP.index
OPa, CLa = OP.to_numpy(), CL.to_numpy()
print(f"面板 {OP.shape}  {idx.min().date()} ~ {idx.max().date()}  ({time.time()-t0:.0f}s)")
for _n, _f in (("close", CL), ("ni_yoy_252", NIY), ("revenue", REV), ("bp_correct", BP)):
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


def zs(df_row):
    """横截面 z-score,先按分位再转 z,避免极端增速主导。"""
    r = df_row.rank(pct=True)
    return (r - 0.5) * 3.4643        # 均匀分位 → 近似标准正态尺度


BPpct = BP.rank(axis=1, pct=True)

month_ends = [x for x in OP.resample("ME").last().index
              if x >= pd.Timestamp(START) and idx[0] <= x <= idx[-1]]
print(f"月度调仓 {len(month_ends)} 期")


def mean_ret(mask, e, x):
    ci = np.flatnonzero(mask)
    if ci.size == 0:
        return np.nan, 0
    a = OPa[e, ci]; b = OPa[x, ci]
    good = np.isfinite(a) & (a > 0)
    ci, a, b = ci[good], a[good], b[good]
    for j in np.flatnonzero(~(np.isfinite(b) & (b > 0))):
        seg = CLa[e:x + 1, ci[j]]; seg = seg[np.isfinite(seg)]
        b[j] = seg[-1] if seg.size else np.nan
    r = b / a - 1
    r = r[np.isfinite(r)]
    return (r.mean() if r.size else np.nan), r.size


BASES = ["ALL 全市场", "VALUE 高BP20%"]
BUCKETS = ["Q1 增速最低", "Q2", "Q3", "Q4", "Q5 增速最高"]
rows, prev = [], {}
for i in range(len(month_ends) - 1):
    e = idx.searchsorted(month_ends[i], side="right")
    x = idx.searchsorted(month_ends[i + 1], side="right")
    if e >= len(idx) or x >= len(idx) or x <= e:
        continue
    p = e - 1
    alive = (np.isfinite(OP.iloc[e]) & np.isfinite(OP.iloc[x])).to_numpy()
    if alive.sum() < 500:
        continue
    ny, ry = NIY.iloc[p], REVY.iloc[p]
    g = zs(ny) + zs(ry)                       # 成长综合分(DeepSeek 的 growth_comp 口径)
    g = g.where(ny.notna() & ry.notna())
    dual = ((ny > 0) & (ry > 0)).fillna(False).to_numpy()
    base_masks = {"ALL 全市场": alive,
                  "VALUE 高BP20%": alive & (BPpct.iloc[p] >= 0.8).fillna(False).to_numpy()}
    rec = {"month": month_ends[i], "days": x - e}
    ci = np.flatnonzero(alive)
    rr = OPa[x, ci] / OPa[e, ci] - 1
    rr = rr[np.isfinite(rr)]
    w = (1 + rr) / (1 + rr).sum()
    rec["bench"] = np.nanmean(rr) - 2 * COST * (0.5 * np.abs(w - 1 / len(w)).sum())

    for bn, bm in base_masks.items():
        # **基线/双增长先记,再判断分档是否有足够样本** ——
        # 首版把 `if ok.sum() < 100: continue` 放在前面,成长字段覆盖率一变,
        # 被跳过的月份连基线一起丢掉(实测丢了 9 个月,7个在2014、2个在2015 牛市),
        # 基线年化从 +12.01% 假摔到 +6.21%,新旧口径不可比。锚点自检抓到。
        for fn, m in (("双增长过滤", bm & dual), ("基线", bm)):
            r, n = mean_ret(m, e, x)
            key = f"{bn}|{fn}"
            cs = set(np.flatnonzero(m)); pv = prev.get(key, set())
            rec[f"ret|{key}"] = r; rec[f"n|{key}"] = n
            rec[f"to|{key}"] = 1.0 if not pv else 1 - len(pv & cs) / max(len(cs), 1)
            prev[key] = cs
        gv = g.where(pd.Series(bm, index=g.index))
        ok = gv.notna()
        if ok.sum() < 100:
            continue
        q = pd.qcut(gv[ok].rank(method="first"), 5, labels=False)
        for qi, qn in enumerate(BUCKETS):
            m = np.zeros(len(bm), bool)
            m[[g.index.get_loc(c) for c in q.index[q == qi]]] = True
            r, n = mean_ret(m, e, x)
            key = f"{bn}|{qn}"
            cs = set(np.flatnonzero(m)); pv = prev.get(key, set())
            rec[f"ret|{key}"] = r; rec[f"n|{key}"] = n
            rec[f"to|{key}"] = 1.0 if not pv else 1 - len(pv & cs) / max(len(cs), 1)
            prev[key] = cs
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


bench = comp(P["bench"], P["days"])
print(f"\n{'#'*104}\n全市场等权基准:年化 **{bench:+.2%}**   "
      f"{P.month.min().date()} ~ {P.month.max().date()}\n{'#'*104}")

res = {}
for bn in BASES:
    print(f"\n{'='*104}\n{bn}:成长五分档 + 双增长过滤\n{'='*104}")
    print(f"{'档位':<14}{'每期只数':>9}{'换手':>8}{'年化':>11}{'相对基准':>11}{'相对本池基线':>14}")
    b0 = comp(P[f"ret|{bn}|基线"] - 2 * COST * P[f"to|{bn}|基线"], P["days"])
    for nm in BUCKETS + ["双增长过滤", "基线"]:
        key = f"{bn}|{nm}"
        if f"ret|{key}" not in P.columns:
            continue
        a = comp(P[f"ret|{key}"] - 2 * COST * P[f"to|{key}"], P["days"])
        res[key] = a
        inc = "" if nm == "基线" else f"{(a-b0)*100:>+13.1f}pp"
        print(f"{nm:<14}{P[f'n|{key}'].median():>9.0f}{P[f'to|{key}'].median():>8.1%}"
              f"{a:>+11.2%}{(a-bench)*100:>+10.1f}pp{inc:>14}")

print(f"\n{'='*104}\n判据判定\n{'='*104}")
for bn in BASES:
    q = [res.get(f"{bn}|{n}") for n in BUCKETS]
    if any(v is None or not np.isfinite(v) for v in q):
        print(f"  {bn}: 样本不足")
        continue
    q1, rest = q[0], q[1:]
    gap = min(rest) - q1
    spread = max(rest) - min(rest)
    mono = all(q[i] < q[i + 1] for i in range(1, 4))
    print(f"  {bn}")
    print(f"    Q1 {q1:+.2%}  |  Q2~Q5 {min(rest):+.2%} ~ {max(rest):+.2%}"
          f"  (极差 {spread*100:.1f}pp)")
    print(f"    Q1 与 Q2~Q5 最差档的差距 **{gap*100:+.1f}pp**   Q2→Q5 是否单调递增:"
          f"{'是' if mono else '否'}")
    if gap >= 0.03 and spread <= 0.02:
        v = "**低端效应 → 价值在『排除负增长』,两份报告可调和**"
    elif mono and (q[4] - q[1]) >= 0.03:
        v = "**排序因子 → DeepSeek 口径成立,本session表述需修正**"
    else:
        v = "两种解释都不干净,需降级表述"
    print(f"    → {v}")

P.to_csv(f"{SP}/growth_rank_vs_filter.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: growth_rank_vs_filter.csv")
