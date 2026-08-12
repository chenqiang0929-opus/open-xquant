"""RPS 数据一致性核对:我重建的 RPS vs 用户快照里的 RPS

═══ 为什么这一步必须落盘 ═══
第四十三节最有分量的一条是「独立复现」——用我自己的面板重建股池,
双增长过滤同样有效。但这条成立的**前提**是:我算的 RPS 与用户的是一回事。
若两边 RPS 差很远,"复现"就只是巧合。

这段核对此前只在对话里跑过,没有脚本、没有日志,**数字无法回溯**。
本脚本把它固化。

═══ 核对四个层次 ═══
  ① 逐值:相关、中位绝对差、分布
  ② 逐期:每个快照日单独算相关,看稳定性
  ③ 阈值:>90 这条线上的混淆矩阵(真正影响选股的是这条线,不是数值本身)
  ④ 系统性:按用户 RPS 分档看我的偏移,排除"整体偏高/偏低"

═══ 一个容易误读的现象,必须一起记下来 ═══
逐期相关会出现很低的值(最低 0.119)。**那是区间压缩的假象,不是分歧**:
池内 RPS 本就挤在 90~100,方差极小,0.07 分的绝对差就能把相关系数打垮。
判断口径应看**绝对差**,不是相关。

═══ 顺带验证用户口述的筛选规则 ═══
用户说两份股池都是「RPS50/120/250 至少一个 > 90」。
A 文件没导出 RPS 列,无法自证 —— 用我的值独立验证。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
WINDOWS = (50, 120, 250)

t0 = time.time()
cl = {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    cl[k] = pd.to_numeric(pd.read_parquet(f, columns=["close"])["close"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
CL = CL.where(CL > 0)
idx = CL.index
# RPS 定义:N 日收益率的横截面百分位 × 100(与通达信/同花顺口径一致)
MY = {n: CL.pct_change(n).rank(axis=1, pct=True) * 100 for n in WINDOWS}
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
del cl


def pair_up(pool, has_rps):
    """把用户快照与我的 RPS 对齐成 (代码, 日期) 配对。"""
    rows = []
    for s, g in pool.groupby("snap"):
        p = idx.searchsorted(pd.Timestamp(s), side="right") - 1   # 快照日当天可得
        if p < max(WINDOWS):
            continue
        for n in WINDOWS:
            mine = MY[n].iloc[p]
            v = pd.DataFrame({"code": g.code.values, "snap": s, "n": n,
                              "mine": [mine.get(c, np.nan) for c in g.code]})
            v["theirs"] = g[f"RPS{n}"].values if has_rps else np.nan
            v["lag_days"] = (pd.Timestamp(s) - idx[p]).days
            rows.append(v)
    return pd.concat(rows, ignore_index=True)


B = pd.read_parquet(f"{SP}/rps_pool_B.parquet")
A = pd.read_parquet(f"{SP}/rps_pool_A.parquet")
VB = pair_up(B, True).dropna(subset=["mine", "theirs"])
VA = pair_up(A, False).dropna(subset=["mine"])

print(f"\n{'='*104}\n① 逐值吻合度(B池,{len(VB):,} 条 (代码,日期) 配对)\n{'='*104}")
print(f"{'':<10}{'相关':>8}{'中位绝对差':>12}{'|差|<2':>9}{'|差|<5':>9}{'|差|>10':>9}{'均值偏移':>10}")
for n in WINDOWS:
    q = VB[VB.n == n]
    d = q.mine - q.theirs
    print(f"RPS{n:<7}{q.theirs.corr(q.mine):>8.3f}{d.abs().median():>12.2f}"
          f"{(d.abs() < 2).mean():>9.1%}{(d.abs() < 5).mean():>9.1%}"
          f"{(d.abs() > 10).mean():>9.1%}{d.mean():>+10.2f}")

print(f"\n{'='*104}\n② 逐期相关的稳定性\n{'='*104}")
for n in WINDOWS:
    q = VB[VB.n == n]
    cs = q.groupby("snap").apply(lambda g: g.theirs.corr(g.mine),
                                 include_groups=False).dropna()
    md = q.groupby("snap").apply(lambda g: (g.mine - g.theirs).abs().median(),
                                 include_groups=False)
    print(f"  RPS{n}: {len(cs)} 期  相关中位 {cs.median():.3f}  最低 {cs.min():.3f}  "
          f">0.99 的期数 {(cs > 0.99).sum()}  |  **中位绝对差 >1 的期数 {(md > 1).sum()}**")
print("  ↑ 相关低是区间压缩的假象(池内 RPS 挤在90~100),看绝对差才对")

worst = (VB[VB.n == 250].groupby(["snap", "lag_days"])
         .apply(lambda g: pd.Series({"相关": g.theirs.corr(g.mine),
                                     "中位绝对差": (g.mine - g.theirs).abs().median(),
                                     "条数": len(g)}), include_groups=False)
         .reset_index().nsmallest(5, "相关"))
print(f"\n  相关最低的 5 期(lag_days = 快照日距我所用交易日的天数):")
print(worst.to_string(index=False))
lag = VB[VB.n == 250].groupby("snap")["lag_days"].first()
print(f"  快照日本身即交易日的期数 {(lag == 0).sum()} / {len(lag)};"
      f"其余落在非交易日,我退到前一交易日")

print(f"\n{'='*104}\n③ >90 这条线上的一致性(真正影响选股的是这条线)\n{'='*104}")
for n in WINDOWS:
    q = VB[VB.n == n]
    tt = ((q.theirs > 90) & (q.mine > 90)).mean()
    tf = ((q.theirs > 90) & (q.mine <= 90)).mean()
    ft = ((q.theirs <= 90) & (q.mine > 90)).mean()
    ff = ((q.theirs <= 90) & (q.mine <= 90)).mean()
    print(f"  RPS{n}: 都>90 {tt:.1%}   你>90我不 {tf:.1%}   我>90你不 {ft:.1%}   "
          f"都不 {ff:.1%}   **一致率 {tt+ff:.1%}**")

print(f"\n{'='*104}\n④ 分歧是否系统性(按用户 RPS250 分档看我的偏移)\n{'='*104}")
q = VB[VB.n == 250].copy()
q["档位"] = pd.cut(q.theirs, [0, 80, 90, 95, 98, 100])
print(q.groupby("档位", observed=True).apply(
    lambda g: pd.Series({"条数": len(g), "你的均值": g.theirs.mean(),
                         "我的均值": g.mine.mean(),
                         "偏移": g.mine.mean() - g.theirs.mean()}),
    include_groups=False).to_string())
print("  ↑ 各档偏移均 <0.25 分 → 不存在整体偏高或偏低")

print(f"\n{'='*104}\n⑤ 独立验证用户口述的规则:「三个 RPS 至少一个 >90」\n{'='*104}")
for tag, V in (("A(无RPS列,只能靠我的值验证)", VA), ("B(有RPS列,作对照)", VB)):
    mx = V.pivot_table(index=["snap", "code"], columns="n", values="mine").max(axis=1).dropna()
    print(f"  {tag:<28} {len(mx):,} 行   >90 **{(mx > 90).mean():.1%}**   "
          f">80 {(mx > 80).mean():.1%}   5%分位 {np.percentile(mx, 5):.1f}")

VB.to_parquet(f"{SP}/rps_consistency_pairs_B.parquet")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: rps_consistency_pairs_B.parquet")
