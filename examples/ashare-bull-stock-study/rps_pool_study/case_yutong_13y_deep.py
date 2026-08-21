"""宇通客车 600066:13 年深挖 —— 3 条大涨腿 vs 另外 72 条,差在哪

**描述性,单只股票,3 个正样本。不设判据,不得据此对全市场下任何结论。**
分段沿用已锁定的 ZigZag θ=10%(锚点预验证见 case_yutong_13y.py)。

问的是本项目最核心的那个问题在单只股票上的版本:
**13 年里所有的钱都出在 3 条腿上 —— 这 3 条腿起涨之前,有没有和另外 72 条不一样的地方?**
"""
import os

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
OUT = os.environ.get("OXQ_OUT_DIR", SP)
CODE, TH, BIG = "600066", 0.10, 0.80

x = pd.read_parquet(f"{SP}/oxq_stock_market_fixed/{CODE}.parquet",
                    columns=["close", "volume", "float_mv"])
if getattr(x.index, "tz", None) is not None:
    x.index = x.index.tz_localize(None)
x = x[(x.index >= "2013-01-04") & (x.index <= "2026-08-03")]
p = x["close"].ffill().to_numpy(float)
vol = pd.to_numeric(x["volume"], errors="coerce").to_numpy(float)
mv = pd.to_numeric(x["float_mv"], errors="coerce").to_numpy(float)
d = x.index
N = len(p)
tor = vol * p / mv                       # 换手率 = 成交量 / 流通股数
W = 100


def zigzag(px, th):
    piv = [(0, "L")]
    ext, ei, up = px[0], 0, True
    for i in range(1, len(px)):
        if up:
            if px[i] > ext:
                ext, ei = px[i], i
            elif px[i] <= ext * (1 - th):
                piv.append((ei, "H"))
                ext, ei, up = px[i], i, False
        else:
            if px[i] < ext:
                ext, ei = px[i], i
            elif px[i] >= ext * (1 + th):
                piv.append((ei, "L"))
                ext, ei, up = px[i], i, True
    piv.append((ei, "H" if up else "L"))
    return piv


piv = zigzag(p, TH)
ups = []
for i in range(len(piv) - 1):
    a, b = piv[i][0], piv[i + 1][0]
    if p[b] <= p[a] or b - a < 5:
        continue
    lo = max(a - 60, 0)
    w7 = p[max(a - 750, 0):a + 1]
    w2 = p[max(a - 250, 0):a + 1]
    ups.append(dict(
        起=str(d[a].date()), 止=str(d[b].date()), 幅度=p[b] / p[a] - 1, 天数=b - a,
        前60日收益=p[a] / p[lo] - 1,
        前60日换手=float(np.nanmean(tor[lo:a + 1])),
        换手历史分位=float((tor[:a + 1] < np.nanmean(tor[lo:a + 1])).mean()),
        前60日量比=float(np.nanmean(vol[lo:a + 1]) / np.nanmean(vol[max(a - 250, 0):a + 1])),
        距250日高=p[a] / np.nanmax(w2) - 1,
        距750日高=p[a] / np.nanmax(w7) - 1,
        过去750日收益=p[a] / p[max(a - 750, 0)] - 1,
        三年区间位置=float((p[a] - np.nanmin(w7)) / (np.nanmax(w7) - np.nanmin(w7))),
        流通市值亿=float(mv[a]) / 1e8))
U = pd.DataFrame(ups)
U["大涨"] = U["幅度"] >= BIG

print(f"宇通客车 600066   {N} 个交易日   {d[0].date()} ~ {d[-1].date()}")
print(f"  买入持有 13 年:{p[0]:.2f} → {p[-1]:.2f} = **{p[-1]/p[0]-1:+.1%}**"
      f"(年化 {(p[-1]/p[0])**(250/N)-1:+.1%})")
print(f"  ZigZag θ={TH:.0%}:上升腿 {len(U)} 条,其中 ≥+{BIG:.0%} 的 **{U['大涨'].sum()}** 条")

print(f"\n{'='*W}\n一、分年拆解:13 年的钱是哪几年赚的\n{'='*W}")
print(f"{'年':<6}{'年初':>8}{'年末':>8}{'年收益':>10}{'年内最大回撤':>13}"
      f"{'日均换手':>10}{'≥+80%腿':>9}")
for y in range(2013, 2027):
    m = (d.year == y)
    if m.sum() < 5:
        continue
    q = p[m]
    dd = float((np.maximum.accumulate(q) - q).max() / np.maximum.accumulate(q).max())
    nb = sum(1 for _, r in U.iterrows() if r["大涨"] and r["起"][:4] == str(y))
    print(f"{y:<6}{q[0]:>8.2f}{q[-1]:>8.2f}{q[-1]/q[0]-1:>+10.1%}{dd:>13.1%}"
          f"{np.nanmean(tor[m]):>10.2%}{nb:>9}")

print(f"\n{'='*W}\n二、3 条大涨腿 vs 另外 {len(U)-3} 条上升腿:起涨**之前**的特征\n{'='*W}")
COLS = ["前60日收益", "前60日换手", "换手历史分位", "前60日量比",
        "距250日高", "距750日高", "过去750日收益", "三年区间位置", "流通市值亿"]
fmt = {"前60日换手": "{:.2%}", "换手历史分位": "{:.2f}", "前60日量比": "{:.2f}",
       "三年区间位置": "{:.2f}", "流通市值亿": "{:.0f}"}
print(f"{'特征':<14}{'大涨腿 1':>11}{'大涨腿 2':>11}{'大涨腿 3':>11}"
       f"{'其余中位':>11}{'其余四分位':>18}   3 条是否都在其余的同一侧")
bigs = U[U["大涨"]].reset_index(drop=True)
rest = U[~U["大涨"]]
for c in COLS:
    f = fmt.get(c, "{:+.1%}")
    med = rest[c].median()
    q1, q3 = rest[c].quantile(.25), rest[c].quantile(.75)
    side = ("全部 < 其余中位" if (bigs[c] < med).all()
            else "全部 > 其余中位" if (bigs[c] > med).all() else "—")
    star = "  **" if side != "—" else ""
    print(f"{c:<14}" + "".join(f"{f.format(bigs[c][i]):>11}" for i in range(3))
          + f"{f.format(med):>11}" + f"{'[' + f.format(q1) + ', ' + f.format(q3) + ']':>18}"
          + f"   {side}{star}")

print(f"\n{'='*W}\n三、3 条大涨腿的明细\n{'='*W}")
for i, r in bigs.iterrows():
    print(f"  {i+1}. {r['起']} → {r['止']}   {r['幅度']:+.1%}   {r['天数']} 个交易日"
          f"   流通市值 {r['流通市值亿']:.0f} 亿")
    print(f"     起涨前 60 日:收益 {r['前60日收益']:+.1%}  换手 {r['前60日换手']:.2%}"
          f"(历史分位 {r['换手历史分位']:.2f})  量比 {r['前60日量比']:.2f}")
    print(f"     起涨点位置:  距 250 日高 {r['距250日高']:+.1%}  "
          f"距 750 日高 {r['距750日高']:+.1%}  过去 750 日 {r['过去750日收益']:+.1%}")

print(f"\n{'='*W}\n四、五次三段信号 vs 买入持有(同期)\n{'='*W}")
SIG = ["2014-12-04", "2017-07-21", "2020-08-06", "2024-01-10", "2025-09-01"]
print(f"{'信号日':<13}{'6个月峰值':>11}{'6个月期末':>11}{'12个月期末':>12}"
      f"{'同期持有到今天':>15}")
tot6 = []
for s in SIG:
    t = int(np.searchsorted(d, pd.Timestamp(s)))
    r6 = (np.nanmax(p[t+1:t+121]) / p[t] - 1) if t + 120 < N else np.nan
    e6 = (p[t+120] / p[t] - 1) if t + 120 < N else np.nan
    e12 = (p[t+250] / p[t] - 1) if t + 250 < N else np.nan
    tot6.append(e6)
    print(f"{s:<13}{r6:>11.1%}{e6:>11.1%}"
          f"{(f'{e12:.1%}' if np.isfinite(e12) else '—'):>12}"
          f"{p[-1]/p[t]-1:>15.1%}")
tv = [v for v in tot6 if np.isfinite(v)]
print(f"\n  五次信号 6 个月期末:中位 **{np.median(tv):+.1%}**、平均 {np.mean(tv):+.1%}、"
      f"最差 {min(tv):+.1%}、最好 {max(tv):+.1%}")
print(f"  同期买入持有 13 年:**{p[-1]/p[0]-1:+.1%}**,期间最大回撤 "
      f"**{float((np.maximum.accumulate(p) - p).max() / np.maximum.accumulate(p).max()):.1%}**")

U.to_csv(f"{OUT}/case_yutong_13y_uplegs.csv", index=False)
print(f"\n→ {OUT}/case_yutong_13y_uplegs.csv")
print("\n**3 个正样本。上面任何「3 条都在同一侧」都可能是巧合"
      "(单个特征纯随机出现的概率约 2×(1/2)³ = 25%)。**")
