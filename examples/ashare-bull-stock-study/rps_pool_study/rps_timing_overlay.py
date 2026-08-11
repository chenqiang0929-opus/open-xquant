"""把择时开关加到用户股池上,分段检验

═══ 为什么试这个 ═══
四十一节已量化:「510300 跌破 MA200 不开新仓」把突破系统年化从
**-13.73% 变成 +6.34%**,是本session测过效果最大的单一开关。
四十三节暴露的问题正是它要解决的 —— 同一套股池三段超额:
  A 期(2023-10~2024-12)  **-18.2pp**
  2025 全年               **-18.2pp**
  2026 H1                **+241.3pp**

═══ 判据(事前写死,而且必须分段看) ═══
**2026 H1 那段本来就在 MA200 之上,择时碰不到它**,
所以"总数改善"几乎必然为正 —— **看总数会自欺**。
真正的判据只有一条:
  **A 期与 2025 年这两段的超额,能否从 -18.2pp 收敛到 -5pp 以内。**
若两段改善但 2026 H1 大幅受损 → 记为"用回撤换收益",不算改进。
若只有总数改善、两段仍是 -18pp → 择时对这套股池无效,如实记。

═══ 口径:直接复用四十三节已算好的逐期收益 ═══
读 `rps_growth_periods_{A,B}.csv`(由 rps_growth_test.py 产出),
在其上叠加择时掩码。**这样基线必然精确等于四十三节的
-10.75% / +82.24%**,不存在新脚本口径漂移的可能。

═══ 基准口径已订正,不要再犯 ═══
等权全市场基准每周只需把权重拉回等权,**实测再平衡换手约 2%/期**,
不是组合的 32%,更不是 100%。第一版和阶梯脚本各错一次,
正确基准为 A 期 **+7.48%** / B 期 **+28.68%**。

═══ 择时信号无前视 ═══
用**快照日当天**(入场前一交易日)的 510300 收盘价与其 MA200 比较。
名单在快照日收盘后拿到,次日开盘入场 —— 信号在决策时点已知。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST = 0.003          # 单边

t0 = time.time()
op = {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    op[k] = pd.to_numeric(pd.read_parquet(f, columns=["open"])["open"], errors="coerce")
OP = pd.DataFrame(op).sort_index()
OP.index = OP.index.tz_localize(None)
OP = OP.where(OP > 0)
idx = OP.index
del op

mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["open", "close"])
mk.index = mk.index.tz_localize(None)
MKO = pd.to_numeric(mk["open"], errors="coerce").reindex(idx).ffill()
MKC = pd.to_numeric(mk["close"], errors="coerce").reindex(idx).ffill()
MA200 = MKC.rolling(200, min_periods=200).mean()
print(f"面板 {OP.shape}   510300 在 MA200 之上的总体比例 "
      f"{(MKC > MA200).mean():.1%}  ({time.time()-t0:.0f}s)")


def comp(r, d):
    r = np.asarray(r, float); d = np.asarray(d, float)
    ok = np.isfinite(r) & np.isfinite(d)
    if ok.sum() == 0:
        return np.nan
    t = np.prod(1 + r[ok]); y = d[ok].sum() / 252
    return t ** (1 / y) - 1 if t > 0 and y > 0 else -1.0


def maxdd(r):
    r = np.asarray(r, float)[np.isfinite(np.asarray(r, float))]
    eq = np.cumprod(1 + r)
    return (eq / np.maximum.accumulate(eq) - 1).min() if len(eq) else np.nan


SEGS = [("A 期 2023-10~2024-12", "A", None, None),
        ("B·2025 全年", "B", "2025-01-01", "2025-12-31"),
        ("B·2026 H1", "B", "2026-01-01", "2026-12-31"),
        ("B 全期", "B", None, None)]

frames = {}
for tag in ("A", "B"):
    P = pd.read_csv(f"{SP}/rps_growth_periods_{tag}.csv", parse_dates=["snap"])
    sig, bench = [], []
    for _, row in P.iterrows():
        e, x = int(row["e"]), int(row["x"])
        p = e - 1                                    # 快照日当天,信号已知
        above = bool(np.isfinite(MA200.iat[p]) and MKC.iat[p] > MA200.iat[p])
        sig.append(above)
        al = OP.columns[np.isfinite(OP.iloc[e]) & np.isfinite(OP.iloc[x])]
        r = OP.iloc[x][al] / OP.iloc[e][al] - 1
        w = (1 + r) / (1 + r).sum()
        to = 0.5 * np.abs(w - 1 / len(w)).sum()      # 等权基准自己的再平衡换手
        bench.append(r.mean() - 2 * COST * to)
    P["above_ma200"] = sig
    P["bench"] = bench
    P["hs300"] = [MKO.iat[int(r.x)] / MKO.iat[int(r.e)] - 1 for _, r in P.iterrows()]
    P["net_dual"] = P["ret_dual"] - P["cost"]
    P["net_all"] = P["ret_all"] - P["cost"]
    # 择时:MA200 之下则空仓(收益0、无成本);半仓变体
    m = P["above_ma200"].to_numpy()
    P["net_dual_timed"] = np.where(m, P["net_dual"], 0.0)
    P["net_dual_half"] = np.where(m, P["net_dual"], 0.5 * P["net_dual"])
    P["net_all_timed"] = np.where(m, P["net_all"], 0.0)
    frames[tag] = P

print(f"\n{'#'*112}")
print("验证1:基线须精确复现四十三节 —— 双增长子集 A -10.75% / B +82.24%")
for tag in ("A", "B"):
    P = frames[tag]
    print(f"  {tag} 池基线 {comp(P.net_dual, P.days):+.2%}   "
          f"(四十三节记录值 {'-10.75%' if tag == 'A' else '+82.24%'})")
print(f"{'#'*112}")

print(f"\n{'='*112}\n验证3:各段 510300 在 MA200 之上的期数占比\n{'='*112}")
for label, tag, d0, d1 in SEGS:
    P = frames[tag]
    g = P if d0 is None else P[(P.snap >= d0) & (P.snap <= d1)]
    print(f"  {label:<22} {len(g):>3} 期   在MA200之上 **{g.above_ma200.mean():>6.1%}**"
          f"   {'← 几乎全程在上方,择时本就不起作用' if g.above_ma200.mean() > 0.9 else ''}")

print(f"\n{'='*112}")
print("择时效果(双增长子集;等权基准按真实再平衡换手~2%扣成本)")
print(f"{'='*112}")
print(f"{'期间':<22}{'基线':>11}{'+择时':>11}{'+半仓择时':>11}{'等权基准':>11}"
      f"{'基线超额':>11}{'择时后超额':>12}")
rows = []
for label, tag, d0, d1 in SEGS:
    P = frames[tag]
    g = P if d0 is None else P[(P.snap >= d0) & (P.snap <= d1)]
    if len(g) < 3:
        continue
    b = comp(g.bench, g.days)
    a0 = comp(g.net_dual, g.days)
    a1 = comp(g.net_dual_timed, g.days)
    a2 = comp(g.net_dual_half, g.days)
    rows.append({"期间": label, "基线": a0, "择时": a1, "半仓择时": a2, "基准": b,
                 "基线超额": a0 - b, "择时超额": a1 - b, "期数": len(g),
                 "基线回撤": maxdd(g.net_dual), "择时回撤": maxdd(g.net_dual_timed)})
    print(f"{label:<22}{a0:>+11.2%}{a1:>+11.2%}{a2:>+11.2%}{b:>+11.2%}"
          f"{(a0-b)*100:>+10.1f}pp{(a1-b)*100:>+11.1f}pp")

print(f"\n{'='*112}\n回撤对比(逐期净值)\n{'='*112}")
for r in rows:
    print(f"  {r['期间']:<22} 基线 {r['基线回撤']:>8.2%}   择时后 {r['择时回撤']:>8.2%}"
          f"   改善 {(r['择时回撤']-r['基线回撤'])*100:>+6.1f}pp")

print(f"\n{'='*112}\n判据判定(事前写死:A期 与 2025 两段的超额须从 -18.2pp 收敛到 -5pp 以内)\n{'='*112}")
key = [r for r in rows if r["期间"] in ("A 期 2023-10~2024-12", "B·2025 全年")]
ok = all(r["择时超额"] >= -0.05 for r in key)
for r in key:
    print(f"  {r['期间']:<22} 基线超额 {r['基线超额']*100:>+7.1f}pp  →  "
          f"择时后 {r['择时超额']*100:>+7.1f}pp   "
          f"{'**达标**' if r['择时超额'] >= -0.05 else '未达标'}")
h1 = [r for r in rows if r["期间"] == "B·2026 H1"]
if h1:
    r = h1[0]
    print(f"  {'B·2026 H1(参照)':<22} 基线 {r['基线']:+.2%}  →  择时后 {r['择时']:+.2%}"
          f"   {'(未受损)' if r['择时'] >= r['基线']*0.95 else '**受损**'}")
print(f"\n  **{'两段均达标 → 择时有效' if ok else '未同时达标 → 择时对这套股池无效或不充分'}**")

pd.DataFrame(rows).to_csv(f"{SP}/rps_timing_overlay.csv", index=False)
for tag in ("A", "B"):
    frames[tag].to_csv(f"{SP}/rps_timing_periods_{tag}.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: rps_timing_overlay.csv, rps_timing_periods_A/B.csv")
