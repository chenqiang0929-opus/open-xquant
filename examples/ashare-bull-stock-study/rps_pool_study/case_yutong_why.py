"""宇通:为什么 2015/2024 成了,2018-07 / 2021-07 败了 —— 用已有字段查,不编故事

**描述性,单只股票,3 成 2 败。任何「原因」都是假设,不是结论。**
本轮此前从未碰过基本面字段(eps/roe/net_income/revenue/book_value_per_share),
本脚本把它们和价格状态放在同一张表上比。

口径说明:eps/net_income/revenue 是**累计值**(一季报/半年报/三季报/年报口径不同),
故一律用**同比**(与 250 个交易日前的同字段比,落在上年同一报告期)。
PB = 收盘 / 每股净资产,为时点值,可直接横向比。
"""
import os

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
OUT = os.environ.get("OXQ_OUT_DIR", SP)
COLS = ["close", "eps", "book_value_per_share", "roe", "revenue",
        "net_income", "turnover"]
x = pd.read_parquet(f"{SP}/oxq_stock_market_fixed/600066.parquet", columns=COLS)
x.index = x.index.tz_localize(None)
x = x[(x.index >= "2013-01-04") & (x.index <= "2026-08-03")]
d = x.index
p = x["close"].ffill().to_numpy(float)
N = len(p)
G = {c: x[c].ffill().to_numpy(float) for c in COLS}

KEY = [
    ("2014-03-20", "成", "第三段起涨 → +125.0%"),
    ("2014-12-04", "成", "三段信号 → 6月 +90.9%"),
    ("2022-10-10", "成", "底部起涨 → +133.0%"),
    ("2024-01-10", "成", "三段信号 → 6月 +91.0%"),
    ("2017-07-21", "败", "三段信号 → 6月 +15.9%/12月 −25.6%"),
    ("2018-03-09", "败", "顶部 → 156 日 −54.9%"),
    ("2018-07-02", "败", "20 月线止损点"),
    ("2021-03-08", "败", "顶部 → 99 日 −30.3%"),
    ("2021-07-01", "败", "跌势中"),
    ("2025-09-01", "?", "三段信号 → 6月 +23.9%"),
]


def at(s):
    return int(np.searchsorted(d, pd.Timestamp(s)))


def yoy(arr, t):
    a, b = arr[t], arr[max(t - 250, 0)]
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return np.nan
    return a / abs(b) - 1


W = 118
print(f"宇通客车 600066  {d[0].date()} ~ {d[-1].date()}\n")
print(f"{'日期':<12}{'成败':<5}{'收盘':>7}{'PB':>6}{'ROE':>7}{'ROE同比':>9}"
      f"{'净利同比':>10}{'营收同比':>10}{'换手率':>8}{'距250高':>9}{'过去3年':>9}   说明")
rows = []
for s, tag, note in KEY:
    t = at(s)
    pb = p[t] / G["book_value_per_share"][t]
    w2 = p[max(t - 250, 0):t + 1]
    w7 = p[max(t - 750, 0):t + 1]
    r = dict(日期=s, 成败=tag, 收盘=p[t], PB=pb, ROE=G["roe"][t],
             ROE同比=G["roe"][t] - G["roe"][max(t - 250, 0)],
             净利同比=yoy(G["net_income"], t), 营收同比=yoy(G["revenue"], t),
             换手率=G["turnover"][t],
             距250高=p[t] / np.nanmax(w2) - 1,
             过去3年=p[t] / p[max(t - 750, 0)] - 1, 说明=note)
    rows.append(r)
    print(f"{s:<12}{tag:<5}{p[t]:>7.2f}{pb:>6.2f}{r['ROE']:>7.2f}"
          f"{r['ROE同比']:>+9.2f}{r['净利同比']:>+10.1%}{r['营收同比']:>+10.1%}"
          f"{r['换手率']:>8.2%}{r['距250高']:>+9.1%}{r['过去3年']:>+9.1%}   {note}")
R = pd.DataFrame(rows)

print(f"\n{'='*W}\n成 vs 败:每个指标的中位数对比(成 4 个 / 败 5 个)\n{'='*W}")
S, F = R[R["成败"] == "成"], R[R["成败"] == "败"]
print(f"{'指标':<12}{'成功组中位':>12}{'失败组中位':>12}{'成功组范围':>22}"
      f"{'失败组范围':>22}   是否完全不重叠")
for c, f in [("PB", "{:.2f}"), ("ROE", "{:.2f}"), ("ROE同比", "{:+.2f}"),
             ("净利同比", "{:+.1%}"), ("营收同比", "{:+.1%}"),
             ("换手率", "{:.2%}"), ("距250高", "{:+.1%}"), ("过去3年", "{:+.1%}")]:
    a, b = S[c].dropna(), F[c].dropna()
    if a.empty or b.empty:
        continue
    sep = "**完全不重叠**" if (a.min() > b.max() or a.max() < b.min()) else "—"
    print(f"{c:<12}{f.format(a.median()):>12}{f.format(b.median()):>12}"
          f"{'[' + f.format(a.min()) + ', ' + f.format(a.max()) + ']':>22}"
          f"{'[' + f.format(b.min()) + ', ' + f.format(b.max()) + ']':>22}   {sep}")

print(f"\n{'='*W}\n盈利轨迹:关键年份的 ROE 与净利润(每年 6 月末快照)\n{'='*W}")
print(f"{'年':<6}{'收盘':>8}{'PB':>7}{'ROE':>8}{'净利润(亿)':>12}{'营收(亿)':>11}{'当年涨跌':>10}")
for y in range(2013, 2027):
    m = np.flatnonzero(d.year == y)
    if len(m) < 5:
        continue
    t = m[min(len(m) // 2, len(m) - 1)]
    print(f"{y:<6}{p[t]:>8.2f}{p[t]/G['book_value_per_share'][t]:>7.2f}"
          f"{G['roe'][t]:>8.2f}{G['net_income'][t]/1e8:>12.2f}"
          f"{G['revenue'][t]/1e8:>11.1f}{p[m[-1]]/p[m[0]]-1:>+10.1%}")

R.to_csv(f"{OUT}/case_yutong_why.csv", index=False)
print(f"\n→ {OUT}/case_yutong_why.csv")
print("\n**4 成 5 败、一只股票。上面任何「完全不重叠」都可能是巧合,**")
print("**且这些指标彼此高度相关(盈利差 → ROE 低 → 股价跌 → 过去 3 年为负),**")
print("**不能当成四条独立证据。要变成结论只有一条路:上全样本。**")
