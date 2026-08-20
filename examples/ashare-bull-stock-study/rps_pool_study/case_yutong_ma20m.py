"""宇通 600066:20 月线止损 —— 用户指出 2018 年那根是必须走的

**描述性,单只股票,不设判据。**
起因:上一轮我拿「买入持有 13 年 +625.6%」当基准,和五次信号的 6 个月期末
中位 +9.5% 比。**用户指出这个基准不成立** —— 2018 年 -49.6%(年内回撤 57.0%),
真实持有者在跌破 20 月线时是会走的。本脚本把这个基准补上。

口径:20 月线 = **最近 20 个月末收盘的均线**(需 20 个月历史,故 2014-09 起可用)。
规则:月末收盘 > MA20 则下一个月满仓,否则空仓;换仓按下月首日开盘所在交易日的收盘计。
**不含交易成本**(A 股双边约 0.1%~0.2%;13 年换手次数少,影响见正文)。

同时报 MA10 / MA12 / MA24 作为对参数的稳健性参考 —— **不是择优,是看结论稳不稳**。
"""
import os

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
OUT = os.environ.get("OXQ_OUT_DIR", SP)
CODE = "600066"

x = pd.read_parquet(f"{SP}/oxq_stock_market_fixed/{CODE}.parquet", columns=["close"])
if getattr(x.index, "tz", None) is not None:
    x.index = x.index.tz_localize(None)
x = x[(x.index >= "2013-01-04") & (x.index <= "2026-08-03")]
px = x["close"].ffill()
d = px.index
p = px.to_numpy(float)
N = len(p)

ym = d.to_period("M")
mend = np.array([int(np.flatnonzero(ym == q)[-1]) for q in ym.unique()])
mc = p[mend]
mp = ym.unique()


def mdd(eq):
    """最大回撤 = max over t of (运行峰值 − 净值)/运行峰值。
    **2026-08 修正**:原写法是「最大绝对跌幅 / 全局峰值」,早期回撤被后期高净值
    稀释,系统性低估 —— 由 §95 的全样本锚点抓出。合成校验 eq=[1,2,1.2,5,4]:
    旧 20.0%,正确 40.0%。"""
    pk = np.maximum.accumulate(eq)
    return float(np.max((pk - eq) / pk))


def run(win):
    """月末收盘 > MA(win) 则下月持有。返回 (净值曲线, 交易次数, 持仓月占比)。"""
    ma = pd.Series(mc).rolling(win).mean().to_numpy(float)
    hold = np.zeros(len(mc), bool)
    hold[1:] = (mc[:-1] > ma[:-1]) & np.isfinite(ma[:-1])   # 用上月末信号,无前视
    eq, cur, trades = [1.0], 1.0, 0
    for i in range(1, len(mc)):
        if hold[i]:
            cur *= mc[i] / mc[i - 1]
        if hold[i] != hold[i - 1]:
            trades += 1
        eq.append(cur)
    return np.array(eq), trades, float(hold.mean())


W = 96
print(f"宇通客车 600066   {N} 个交易日   {d[0].date()} ~ {d[-1].date()}   "
      f"{len(mc)} 个月")
bh = mc / mc[0]
print(f"\n{'='*W}\n一、20 月线择时 vs 买入持有(全期 {mp[0]} ~ {mp[-1]})\n{'='*W}")
print(f"{'策略':<16}{'总收益':>11}{'年化':>9}{'最大回撤':>10}{'2018 年':>10}"
      f"{'交易次数':>9}{'持仓月占比':>11}")
yrs = len(mc) / 12
rows = []
i18 = [i for i, q in enumerate(mp) if q.year == 2018]


def y18(eq):
    return float(eq[i18[-1]] / eq[i18[0] - 1] - 1)


print(f"{'买入持有':<16}{bh[-1]-1:>+11.1%}{bh[-1]**(1/yrs)-1:>+9.1%}"
      f"{mdd(bh):>10.1%}{y18(bh):>+10.1%}{1:>9}{'100.0%':>11}")
for w in (10, 12, 20, 24):
    eq, tr, ho = run(w)
    tag = f"MA{w} 月线" + ("  ←用户提的" if w == 20 else "")
    print(f"{tag:<16}{eq[-1]-1:>+11.1%}{eq[-1]**(1/yrs)-1:>+9.1%}"
          f"{mdd(eq):>10.1%}{y18(eq):>+10.1%}{tr:>9}{ho:>11.1%}")
    rows.append(dict(策略=f"MA{w}", 总收益=eq[-1] - 1, 年化=eq[-1] ** (1 / yrs) - 1,
                     最大回撤=mdd(eq), 年2018=y18(eq), 交易次数=tr, 持仓月占比=ho))

eq20, _, _ = run(20)
ma20 = pd.Series(mc).rolling(20).mean().to_numpy(float)
print(f"\n{'='*W}\n二、20 月线的进出场明细(信号在月末,次月执行)\n{'='*W}")
hold = np.zeros(len(mc), bool)
hold[1:] = (mc[:-1] > ma20[:-1]) & np.isfinite(ma20[:-1])
print(f"{'月份':<10}{'动作':<8}{'月末收盘':>10}{'MA20':>9}{'区间收益':>11}")
prev, anchor = False, None
for i in range(1, len(mc)):
    if hold[i] and not prev:
        anchor = mc[i - 1]
        print(f"{mp[i]!s:<10}{'▲ 买入':<8}{mc[i-1]:>10.2f}{ma20[i-1]:>9.2f}{'':>11}")
    elif prev and not hold[i]:
        print(f"{mp[i]!s:<10}{'▼ 卖出':<8}{mc[i-1]:>10.2f}{ma20[i-1]:>9.2f}"
              f"{mc[i-1]/anchor-1:>+11.1%}")
    prev = hold[i]
if prev:
    print(f"{mp[-1]!s:<10}{'持有中':<8}{mc[-1]:>10.2f}{ma20[-1]:>9.2f}"
          f"{mc[-1]/anchor-1:>+11.1%}")

print(f"\n{'='*W}\n三、五次三段信号 + 20 月线止损(买入后持有至跌破 20 月线)\n{'='*W}")
SIG = ["2014-12-04", "2017-07-21", "2020-08-06", "2024-01-10", "2025-09-01"]
print(f"{'信号日':<13}{'买入价':>9}{'离场':<10}{'离场价':>9}{'持有月':>8}"
      f"{'收益':>10}{'原 6 月期末':>12}")
res = []
for s in SIG:
    t = int(np.searchsorted(d, pd.Timestamp(s)))
    mi = int(np.searchsorted(mend, t))
    ex = None
    for i in range(mi + 1, len(mc)):
        if np.isfinite(ma20[i]) and mc[i] < ma20[i]:
            ex = i
            break
    e6 = (p[t + 120] / p[t] - 1) if t + 120 < N else np.nan
    if ex is None:
        r, lbl, xp, nm = mc[-1] / p[t] - 1, f"至今 {mp[-1]}", mc[-1], len(mc) - 1 - mi
    else:
        r, lbl, xp, nm = mc[ex] / p[t] - 1, str(mp[ex]), mc[ex], ex - mi
    res.append(r)
    print(f"{s:<13}{p[t]:>9.2f}{lbl:<10}{xp:>9.2f}{nm:>8}{r:>+10.1%}"
          f"{(f'{e6:+.1%}' if np.isfinite(e6) else '—'):>12}")
print(f"\n  五次「信号买入 + 20 月线止损」收益:"
      f"中位 **{np.median(res):+.1%}**、平均 {np.mean(res):+.1%}、"
      f"最差 {min(res):+.1%}、最好 {max(res):+.1%}")
print(f"  对照:五次的 6 个月期末中位 +9.5%;买入持有全期 {bh[-1]-1:+.1%}"
      f"(回撤 {mdd(bh):.1%});20 月线择时全期 {eq20[-1]-1:+.1%}(回撤 {mdd(eq20):.1%})")

pd.DataFrame(rows).to_csv(f"{OUT}/case_yutong_ma20m.csv", index=False)
print(f"\n→ {OUT}/case_yutong_ma20m.csv")
print("\n**单只股票。20 月线在宇通身上避开了 2018,但那是 1 次事件,"
      "不能据此认为它在别处也有效。**")
