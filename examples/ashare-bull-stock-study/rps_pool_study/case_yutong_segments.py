"""诊断:量化用户给的两段划分,并在宇通的 5 个整理段上试算候选判别量

**这是探索性诊断,不是检验,不设判据,不得据此下任何结论。**
样本只有 5 段 —— **5 个点上任何度量都能被凑得「分得开」**。
本脚本的唯一用途是:把候选判别量算出来看一眼,然后**挑一个**去做事前登记 + 全样本检验。

═══ 用户的划分(原话,本脚本按此划分,不自行改动)═══
> 「2013 年到 2015 年,宇通客车这段也是 3 段经典走势;
>   2015 年到 2018 年属于一直横盘的阶段。」

A 部分  按用户划分,量化这两段
B 部分  在宇通 5 个整理段上算候选判别量:
        4 个筛选器亮过的(2015-12 / 2016-11 / 2017-12 / 2023-11)
        + 1 个被缩量比否掉的(2013-07,即用户说的那段平台)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from consolidation_screener import load_panel, series_of  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
CODE = "600066"

CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    CL = CL.drop(columns=["510300"])
    frames.pop("510300", None)
idx = CL.index
h, lo, c, v = series_of(frames, idx, CODE)
fa = CL[CODE].where(CL[CODE] > 0).ffill().to_numpy(float)
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}\n")


def ix(d):
    return int(np.searchsorted(idx, pd.Timestamp(d, tz=idx.tz)))


def seg(d0, d1, name):
    a, b = ix(d0), min(ix(d1), len(idx) - 1)
    s = fa[a:b + 1]
    s = s[np.isfinite(s)]
    if not len(s):
        return
    hi, lom = float(np.nanmax(s)), float(np.nanmin(s))
    print(f"  {name:<26}{idx[a].date()} ~ {idx[b].date()}  ({b-a+1:>4} 日)"
          f"  {fa[a]:>6.2f} → {fa[b]:<6.2f} {fa[b]/fa[a]-1:>+8.1%}"
          f"   区间 [{lom:.2f}, {hi:.2f}] 振幅 {hi/lom-1:>+7.1%}")


W = 108
print("=" * W)
print("A-1  用户划分的「2013-2015 三段经典走势」")
print("=" * W)
seg("2013-01-04", "2013-06-28", "第一段 上涨")
seg("2013-07-01", "2014-10-31", "第二段 平台整理")
seg("2014-11-03", "2015-06-12", "第三段 再上涨")
a, b = ix("2014-10-31"), ix("2015-06-12")
pk = float(np.nanmax(fa[a:b + 1]))
print(f"\n  **第三段从 2014-10-31 收盘 {fa[a]:.2f} 到区间峰值 {pk:.2f} = "
      f"{pk/fa[a]-1:+.1%}**   峰值日 {idx[a+int(np.nanargmax(fa[a:b+1]))].date()}")
print("  **这是一个完整的、成功的三段结构,而筛选器整段没认**(缩量比 1.07~1.27 挡住)")

print("\n" + "=" * W)
print("A-2  用户划分的「2015-2018 一直横盘」")
print("=" * W)
seg("2015-06-15", "2018-12-31", "整段")
for y0, y1 in [("2015-06-15", "2016-06-30"), ("2016-07-01", "2017-06-30"),
               ("2017-07-03", "2018-06-29"), ("2018-07-02", "2018-12-31")]:
    seg(y0, y1, "  其中")
print("\n  筛选器在这段里的三次突破(§93 实测):")
for d, r6, e6 in [("2016-07-15", "+6.4%", "-11.5%"), ("2017-07-21", "+15.9%", "+1.6%"),
                  ("2018-01-03", "-1.5%", "-30.7%")]:
    t = ix(d)
    a3 = max(t - 750, 0)
    w = fa[a3:t + 1]
    w = w[np.isfinite(w)]
    print(f"    {d}  收盘 {fa[t]:>6.2f}   6 个月峰值 {r6:>7}  期末 {e6:>7}"
          f"   过去 3 年区间 [{np.nanmin(w):.2f}, {np.nanmax(w):.2f}]"
          f"  位置 {(fa[t]-np.nanmin(w))/(np.nanmax(w)-np.nanmin(w)):.2f}")

print("\n" + "=" * W)
print("B  5 个整理段的候选判别量(探索性,5 个点证明不了任何事)")
print("=" * W)
# (标签, ts 日, 突破日, 缩量比, 6个月峰值)  ts/缩量比/峰值 取自 §93 与本轮诊断实测
SEGS = [
    ("2013-07 段(被否)", "2013-07-02", "2014-11-03", 1.21, None),
    ("2015-12 段", "2015-10-28", "2016-07-15", 0.66, +0.064),
    ("2016-11 段", "2016-08-01", "2017-07-21", 0.76, +0.159),
    ("2017-12 段", "2017-12-06", "2018-01-03", 0.58, -0.015),
    ("2023-11 段", "2023-07-20", "2024-01-08", 0.67, +1.011),
]
print(f"{'段':<20}{'ts':<12}{'突破/起涨日':<13}{'缩量比':>7}{'6月峰值':>9}"
      f"{'第一段幅度':>11}{'清掉一段顶':>11}{'3年位置':>9}{'3年净涨':>9}")
rows = []
for name, tsd, bkd, shr, r6 in SEGS:
    ts, bk = ix(tsd), ix(bkd)
    w1 = fa[max(ts - 250, 0):ts + 1]
    w1 = w1[np.isfinite(w1)]
    leg1 = float(np.nanmax(w1) / np.nanmin(w1) - 1) if len(w1) else np.nan
    top1 = float(np.nanmax(w1)) if len(w1) else np.nan
    w3 = fa[max(bk - 750, 0):bk + 1]
    w3 = w3[np.isfinite(w3)]
    pos3 = float((fa[bk] - np.nanmin(w3)) / (np.nanmax(w3) - np.nanmin(w3)))
    ret3 = float(fa[bk] / fa[max(bk - 750, 0)] - 1)
    clr = fa[bk] > top1
    rows.append(dict(段=name, ts=tsd, 突破日=bkd, 缩量比=shr,
                     六月峰值=r6, 第一段幅度=leg1, 一段顶=top1,
                     突破收盘=float(fa[bk]), 清掉一段顶=bool(clr),
                     三年位置=pos3, 三年净涨=ret3))
    print(f"{name:<20}{tsd:<12}{bkd:<13}{shr:>7.2f}"
          f"{('—' if r6 is None else f'{r6:+.1%}'):>9}"
          f"{leg1:>11.1%}{('是' if clr else '否'):>11}{pos3:>9.2f}{ret3:>9.1%}")
print("\n  注:2013-07 段没有筛选器突破日(整段被否),这里用用户划分的第三段起点 2014-11-03;")
print("     其 6 个月峰值一栏留空,因为它不是同一种口径下的事件,不可与另外四个直接比。")
print("     该段从 2014-10-31 起算的实际峰值见 A-1。")

pd.DataFrame(rows).to_csv(
    os.environ.get("OXQ_OUT_DIR", SP) + "/case_yutong_segments.csv", index=False)
print("\n**再说一次:5 个点上任何度量都能凑得分得开。**")
print("**上面没有一行是结论。要下结论必须挑一个判别量,事前登记,上全样本。**")
