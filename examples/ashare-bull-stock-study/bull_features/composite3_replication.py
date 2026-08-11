"""在含退市股的数据上复现 DeepSeek 的推荐策略 composite3

═══ 为什么必须重测 ═══
DeepSeek 报告 5.1 节推荐 `composite3 = zscore(ep) + zscore(net_profit_margin)
+ zscore(rmdd20)`,称近5年年化 12.51%、Sharpe 0.830。

但它 7.2 节自陈:**股票池为"当前在市"快照(剔除退市股)**,
并把幸存者偏差修正量估为 **-0.4~-0.6pp/年**。

**那是市场平均值,不是因子特定值。** 本session二十九节已实测:
`rmdd20` 单因子在含退市股数据上 OOS 年化 **+7.85%**,而它声称 +18.3%,
差距里**幸存者偏差约占 10.5pp** —— 因为 rmdd20 偏向弱势股,正是退市高发区。
而 composite3 的三个成分之一就是 rmdd20。

═══ 三档并列,把幸存者偏差单独隔离 ═══
  ① 含退市股(本session数据,真实口径)
  ② 剔除退市股(模拟 DeepSeek 口径:只保留样本期末仍在交易的股票)
  ③ ①②之差 = **该策略特定的幸存者偏差**,与它估的 -0.4~-0.6pp 对照

═══ 方向定义(二十九节踩过坑,必须打印出来核对) ═══
`rmdd20` 取**抗跌方向**:20日窗口内的最大回撤(负数),**数值越接近0越抗跌**,
所以 zscore 越高越好。二十九节第一次实现时读反成"选深回撤",
结论从 +7.85% 翻成 -6.15% —— 一处方向读反就足以让整个复现翻个个儿。

═══ 事前判据 ═══
composite3 在**含退市股**数据上需:
  ① 跑赢同期全市场等权基准 ≥ 2pp
  ② 且 2014-2017 / 2018-2021 / 2022-2026 三段都不为负
才算复现成功。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST = 0.003
TOPN = 20
START = "2014-06-01"

t0 = time.time()
COLS = ["open", "close", "eps", "net_income", "revenue"]
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
    f = pd.DataFrame(d[key]).sort_index()
    f.index = f.index.tz_localize(None)
    return f.reindex(index=OP.index, columns=OP.columns)


CL = _align("close"); EPS = _align("eps")
NI = _align("net_income"); REV = _align("revenue")
OP = OP.where(OP > 0); CL = CL.where(CL > 0)
idx = OP.index
OPa, CLa = OP.to_numpy(), CL.to_numpy()
NT, NC = OP.shape
print(f"面板 {OP.shape}  {idx.min().date()} ~ {idx.max().date()}  ({time.time()-t0:.0f}s)")
for _n, _f in (("close", CL), ("eps", EPS), ("net_income", NI), ("revenue", REV)):
    _r = _f.notna().mean().mean()
    assert _r > 0.01, f"{_n} 几乎全为 NaN(非空率 {_r:.4%})"
    print(f"  {_n:<12} 非空率 {_r:>6.1%}")
del d

# 退市判定:样本期末仍在交易 = 最后 20 个交易日内有过有效收盘
fin = np.isfinite(CLa)
last_valid = np.where(fin.any(axis=0), NT - 1 - np.argmax(fin[::-1], axis=0), -1)
still_listed = last_valid >= NT - 20
print(f"\n样本期末仍在交易 {still_listed.sum():,} 只 / 共 {NC:,} 只"
      f"  → 已退市/长停 **{(~still_listed).sum():,} 只({(~still_listed).mean():.1%})**")

LOGP = np.log(CLa)
EPa, NIa, REVa = EPS.to_numpy(), NI.to_numpy(), REV.to_numpy()

month_ends = [x for x in OP.resample("ME").last().index
              if x >= pd.Timestamp(START) and idx[0] <= x <= idx[-1]]
print(f"月度调仓 {len(month_ends)} 期")

print("\n因子方向(核对用):")
print("  ep      = eps / 收盘价              越高越好(便宜)")
print("  npm     = net_income / revenue      越高越好(盈利质量)")
print("  rmdd20  = 20日窗口内最大回撤(负数)  **越接近0越好(抗跌)** ← 二十九节订正后的方向")


def zrank(v):
    """横截面分位 z(对极端值稳健)。"""
    s = pd.Series(v)
    r = s.rank(pct=True)
    return ((r - 0.5) * 3.4643).to_numpy()


def mean_ret(ci, e, x):
    if ci.size == 0:
        return np.nan, 0
    a = OPa[e, ci].copy(); b = OPa[x, ci].copy()
    good = np.isfinite(a) & (a > 0)
    ci2, a, b = ci[good], a[good], b[good]
    for j in np.flatnonzero(~(np.isfinite(b) & (b > 0))):
        seg = CLa[e:x + 1, ci2[j]]; seg = seg[np.isfinite(seg)]
        b[j] = seg[-1] if seg.size else np.nan
    r = b / a - 1
    r = r[np.isfinite(r)]
    return (r.mean() if r.size else np.nan), r.size


UNIVERSES = [("① 含退市股(真实口径)", None),
             ("② 剔除退市股(模拟DeepSeek口径)", still_listed)]

rows, prev = [], {}
for i in range(len(month_ends) - 1):
    e = idx.searchsorted(month_ends[i], side="right")
    x = idx.searchsorted(month_ends[i + 1], side="right")
    if e >= len(idx) or x >= len(idx) or x <= e or e < 25:
        continue
    p = e - 1
    alive = np.isfinite(OPa[e]) & np.isfinite(OPa[x])
    if alive.sum() < 500:
        continue
    # rmdd20:只在调仓日算,窗口内 logp − 窗口内累计最高,取最小值
    w = LOGP[p - 19:p + 1]
    rmdd = np.nanmin(w - np.maximum.accumulate(np.nan_to_num(w, nan=-np.inf), axis=0), axis=0)
    rmdd = np.where(np.isfinite(w).sum(axis=0) >= 15, rmdd, np.nan)
    ep = EPa[p] / np.where(CLa[p] > 0, CLa[p], np.nan)
    npm = NIa[p] / np.where(np.abs(REVa[p]) > 0, REVa[p], np.nan)
    rec = {"month": month_ends[i], "days": x - e}

    for un, umask in UNIVERSES:
        base = alive if umask is None else (alive & umask)
        valid = base & np.isfinite(ep) & np.isfinite(npm) & np.isfinite(rmdd)
        vi = np.flatnonzero(valid)
        if vi.size < 100:
            continue
        score = zrank(ep[vi]) + zrank(npm[vi]) + zrank(rmdd[vi])
        top = vi[np.argsort(score)[::-1][:TOPN]]
        r, n = mean_ret(top, e, x)
        key = f"c3|{un}"
        cs = set(top.tolist()); pv = prev.get(key, set())
        rec[f"ret|{key}"] = r; rec[f"n|{key}"] = n
        rec[f"to|{key}"] = 1.0 if not pv else 1 - len(pv & cs) / max(len(cs), 1)
        prev[key] = cs
        # 同口径基准
        bi = np.flatnonzero(base)
        br, _ = mean_ret(bi, e, x)
        rr = OPa[x, bi] / OPa[e, bi] - 1
        rr = rr[np.isfinite(rr)]
        ww = (1 + rr) / (1 + rr).sum()
        rec[f"bench|{un}"] = br - 2 * COST * (0.5 * np.abs(ww - 1 / len(ww)).sum())
    rows.append(rec)

P = pd.DataFrame(rows)
print(f"\n有效月份 {len(P)}  ({time.time()-t0:.0f}s)")


def comp(r, dd):
    r = np.asarray(r, float); dd = np.asarray(dd, float)
    ok = np.isfinite(r) & np.isfinite(dd)
    if ok.sum() == 0:
        return np.nan
    t = np.prod(1 + r[ok]); y = dd[ok].sum() / 252
    return t ** (1 / y) - 1 if t > 0 and y > 0 else -1.0


def sharpe(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan


def mdd(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    eq = np.cumprod(1 + r)
    return (eq / np.maximum.accumulate(eq) - 1).min() if eq.size else np.nan


SEGS = [("全区间 2014-2026", 2014, 2026), ("2014-2017", 2014, 2017),
        ("2018-2021", 2018, 2021), ("2022-2026", 2022, 2026),
        ("近5年 2021-2026", 2021, 2026)]

print(f"\n{'='*112}\ncomposite3 复现结果(Top-20 等权,月频,次日开盘,单边0.3%按换手扣)\n{'='*112}")
print(f"{'口径':<30}{'期间':<18}{'年化':>10}{'Sharpe':>9}{'最大回撤':>10}"
      f"{'同口径基准':>12}{'超额':>11}")
out = {}
for un, _ in UNIVERSES:
    k = f"ret|c3|{un}"
    if k not in P.columns:
        continue
    net = P[k] - 2 * COST * P[f"to|c3|{un}"]
    for sn, y0, y1 in SEGS:
        g = (P.month.dt.year >= y0) & (P.month.dt.year <= y1)
        a = comp(net[g], P.days[g]); b = comp(P[f"bench|{un}"][g], P.days[g])
        out[(un, sn)] = (a, b)
        print(f"{un:<30}{sn:<18}{a:>+10.2%}{sharpe(net[g]):>9.3f}{mdd(net[g]):>10.2%}"
              f"{b:>+12.2%}{(a-b)*100:>+10.1f}pp")
    print()

print(f"{'='*112}\n幸存者偏差的隔离(①含退市 − ②剔除退市)\n{'='*112}")
u1, u2 = UNIVERSES[0][0], UNIVERSES[1][0]
print(f"{'期间':<18}{'含退市':>11}{'剔除退市':>11}{'差 = 幸存者偏差':>18}{'DeepSeek 自估':>15}")
for sn, _, _ in SEGS:
    if (u1, sn) in out and (u2, sn) in out:
        a1 = out[(u1, sn)][0]; a2 = out[(u2, sn)][0]
        print(f"{sn:<18}{a1:>+11.2%}{a2:>+11.2%}{(a1-a2)*100:>+17.1f}pp"
              f"{'-0.4~-0.6pp':>15}")

print(f"\n{'='*112}\n与 DeepSeek 报告数字对照\n{'='*112}")
print(f"  DeepSeek 5.2 节声称:全区间年化 13.06%、近5年 12.31%、近5年 Sharpe 0.878、回撤 -14.8%")
print(f"                     (其口径 = 剔除退市股 + inv_vol 加权;本脚本为等权,不含 inv_vol)")
if (u2, "近5年 2021-2026") in out:
    a2 = out[(u2, "近5年 2021-2026")][0]
    print(f"  本脚本 ②剔除退市股 近5年:{a2:+.2%}   与其 12.31% 相差 {(a2-0.1231)*100:+.1f}pp")
    print(f"    → 若差距很大,说明是因子实现差异而非幸存者偏差,须先查清再下结论")

print(f"\n{'='*112}\n判据判定(①跑赢同口径基准≥2pp ②三段都不为负)\n{'='*112}")
seg3 = ["2014-2017", "2018-2021", "2022-2026"]
a, b = out.get((u1, "全区间 2014-2026"), (np.nan, np.nan))
c1 = np.isfinite(a) and (a - b) >= 0.02
c2 = all(out.get((u1, s), (np.nan,))[0] > 0 for s in seg3)
print(f"  ① 含退市股全区间超额 {(a-b)*100:+.1f}pp  {'✓' if c1 else '✗(需≥+2pp)'}")
for s in seg3:
    v = out.get((u1, s), (np.nan, np.nan))
    print(f"     {s} 年化 {v[0]:+.2%}(基准 {v[1]:+.2%})  {'✓' if v[0] > 0 else '✗'}")
print(f"\n  **{'复现成功' if (c1 and c2) else '复现失败 —— 在含退市股的数据上不成立'}**")

P.to_csv(f"{SP}/composite3_replication.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: composite3_replication.csv")
