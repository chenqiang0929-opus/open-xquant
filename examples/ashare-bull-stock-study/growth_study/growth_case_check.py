"""成长股方向的案例核对:嘉益股份 / 匠心家居 / 东鹏饮料

═══ 要回答的四个问题 ═══
1 **财报能不能事前看出来**:第一次从财报看出高增长(净利同比 >50%)是哪一天?
  那天股价在哪、距最终高点还剩多少涨幅?
2 **失速信号来不来得及**:第一次「连续两期同比下滑」是哪一天?
  距股价高点多久、期间已经跌了多少?
3 **止损会不会被打掉**:从「首次看出高增长」持有到高点,
  中途最大回撤多少、有多少次跌破 -10%?
4 **252 天上限会不会切断**:从首次信号到高点用了多少个交易日?

═══ 口径(全部只用公布后的信息,无前视) ═══
财报公布日 = `net_income`(年初至今累计)发生变化的那一天 ——
在该日之前,市场看不到这个数,所以用这一天做买入基准是干净的。
同比 = 本期 YTD ÷ 四期之前的 YTD − 1(四期 = 同一报告期上一年)。

⚠️ 这是**三只票的案例核对,不是回测**。三只都是事后挑出来的赢家,
   任何统计量都带幸存者偏差。它只用来看「时间轴对不对得上」,
   不能用来估计收益。
"""
import os

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
CASES = [("301004", "嘉益股份"), ("301061", "匠心家居"), ("605499", "东鹏饮料")]
HIGH_GROWTH = 0.50      # 事前锁定:净利同比 >50% 算「看出高增长」
LAG = 4                 # 四期之前 = 同一报告期上一年

rows = []
for code, name in CASES:
    x = pd.read_parquet(f"{DATA}/{code}.parquet",
                        columns=["close", "low", "net_income", "revenue"])
    x.index = x.index.tz_localize(None)
    cl = pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0)
    lo = pd.to_numeric(x["low"], errors="coerce").where(lambda s: s > 0)
    ni = pd.to_numeric(x["net_income"], errors="coerce")
    rv = pd.to_numeric(x["revenue"], errors="coerce")

    # 财报公布日 = net_income 变化日
    chg = ni.diff().ne(0) & ni.notna()
    rd = list(ni.index[chg])
    print(f"\n{'='*100}\n{code} {name}   数据 {x.index[0].date()} ~ {x.index[-1].date()}"
          f"   财报公布日 {len(rd)} 个\n{'='*100}")

    rep = pd.DataFrame({"日期": rd,
                        "净利YTD": [ni.loc[d] for d in rd],
                        "营收YTD": [rv.loc[d] for d in rd],
                        "收盘": [cl.loc[d] for d in rd]})
    rep["净利同比"] = rep.净利YTD / rep.净利YTD.shift(LAG) - 1
    rep["营收同比"] = rep.营收YTD / rep.营收YTD.shift(LAG) - 1
    print(rep.assign(
        净利YTD=lambda d: (d.净利YTD / 1e8).round(2),
        营收YTD=lambda d: (d.营收YTD / 1e8).round(2),
        净利同比=lambda d: d.净利同比.map(lambda v: f"{v:+.1%}" if np.isfinite(v) else "—"),
        营收同比=lambda d: d.营收同比.map(lambda v: f"{v:+.1%}" if np.isfinite(v) else "—"),
        收盘=lambda d: d.收盘.round(2),
        日期=lambda d: d.日期.dt.date).to_string(index=False))

    # ── 问题1:第一次看出高增长 ──
    hi_mask = rep.净利同比 > HIGH_GROWTH
    if not hi_mask.any():
        print(f"  {name}:数据内从未出现净利同比 >{HIGH_GROWTH:.0%}")
        continue
    i0 = int(np.flatnonzero(hi_mask)[0])
    d0, p0 = rep.日期[i0], rep.收盘[i0]

    # ── 股价高点 ──
    seg = cl.loc[d0:]
    d_hi = seg.idxmax()
    p_hi = seg.max()
    p_now = cl.dropna().iloc[-1]
    d_now = cl.dropna().index[-1]

    # ── 问题2:第一次连续两期同比下滑 ──
    dec = (rep.净利同比 < 0)
    d_sig = p_sig = None
    for k in range(1, len(rep)):
        if bool(dec.iloc[k]) and bool(dec.iloc[k - 1]):
            d_sig, p_sig = rep.日期[k], rep.收盘[k]
            break

    # ── 问题3/4:持有期间的回撤与时长 ──
    hold = cl.loc[d0:d_hi]
    hold_lo = lo.loc[d0:d_hi]
    run = hold.cummax()
    dd = (hold / run - 1)
    n_10 = int((dd <= -0.10).sum())
    # 「跌破 -10% 止损」按持仓最高价回撤口径数「首次触发」次数
    trig, armed = 0, True
    for v in (hold_lo / run - 1).to_numpy():
        if armed and np.isfinite(v) and v <= -0.10:
            trig += 1
            armed = False
        elif not armed and np.isfinite(v) and v > -0.05:
            armed = True

    print(f"\n  ① 首次净利同比 >{HIGH_GROWTH:.0%}:**{d0}**  收盘 {p0:.2f}")
    print(f"     此后最高 **{p_hi:.2f}**({d_hi.date()})  "
          f"→ 若那天买入,到高点 **{p_hi/p0-1:+.1%}**")
    print(f"     用时 **{len(hold)} 个交易日**(约 {len(hold)/252:.1f} 年)"
          f"  ← 252 日上限{'**会**' if len(hold) > 252 else '不会'}切断")
    print(f"  ③ 持有期最大回撤 **{dd.min():.1%}**;"
          f"回撤≤-10% 的交易日 {n_10} 天,**-10% 止损会被触发约 {trig} 次**")
    if d_sig is not None:
        i_sig = int(cl.index.searchsorted(pd.Timestamp(d_sig)))
        print(f"  ② 首次「连续两期净利同比为负」:**{d_sig}**  收盘 {p_sig:.2f}")
        print(f"     距高点 {(pd.Timestamp(d_sig)-d_hi).days} 天,"
              f"此时已从高点跌 **{p_sig/p_hi-1:+.1%}**")
        print(f"     从买入算 **{p_sig/p0-1:+.1%}**(vs 高点 {p_hi/p0-1:+.1%})")
    else:
        print("  ② 数据内尚未出现「连续两期同比为负」")
    print(f"  现价 {p_now:.2f}({d_now.date()})  从高点 **{p_now/p_hi-1:+.1%}**"
          f"  从买入 **{p_now/p0-1:+.1%}**")

    rows.append({"代码": code, "名称": name, "首次高增长日": d0, "买入价": p0,
                 "高点日": d_hi.date(), "高点价": p_hi, "买到高点": p_hi / p0 - 1,
                 "持有交易日": len(hold), "期间最大回撤": dd.min(),
                 "止损触发次数": trig, "失速信号日": d_sig,
                 "失速时距高点跌幅": (p_sig / p_hi - 1) if d_sig is not None else np.nan,
                 "失速时累计收益": (p_sig / p0 - 1) if d_sig is not None else np.nan,
                 "现价": p_now, "现在累计收益": p_now / p0 - 1})

R = pd.DataFrame(rows)
print(f"\n{'='*100}\n汇总\n{'='*100}")
print(R.to_string(index=False))
R.to_csv(f"{SP}/growth_case_check.csv", index=False)
print(f"\n⚠️ 三只都是事后挑出的赢家,带幸存者偏差 —— 只看时间轴,不估收益。")
print(f"→ {SP}/growth_case_check.csv")
