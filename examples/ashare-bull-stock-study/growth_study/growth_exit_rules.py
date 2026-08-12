"""成长股方向的离场规则对比:10月均线 vs 其它四种

═══ 为什么问这个 ═══
案例核对(growth_case_check.py)测出三件事:
  ① 财报信号买入端成立 —— 三只都是先有 >50% 同比、后有大涨
  ② 现有的「252日上限 + -10%止损」结构性拿不到 —— 嘉益触发 13 次止损、持有 661 日
  ③ **卖出端不成立**:东鹏营收/净利至今 +20.7%/+15.9% 没失速,股价从 253 跌到 130(-48.8%)
     → **基本面离场信号在 A 股会失效**,离场只能用价格

用户提议:止损改用 **10 月均线**。它是价格信号,且周期够长,不会被中途回撤打掉。

═══ 五种离场规则(全部无前视) ═══
  A 不止损        持有到数据末端
  B -10% 固定止损  现有 62 节口径(买入价的 90%)
  C **10月均线**   月末收盘 < 10个月简单均线 → **次月首个交易日开盘**卖出
  D 日线 MA200     收盘 < MA200 → **次日开盘**卖出
  E 高点回撤 -25%  收盘从持仓期最高收盘回撤 25% → 次日开盘卖出

买入 = 首次净利同比 >50% 的财报日的**次日开盘**(财报当日收盘已含该信息,
不能用当日收盘买 —— 与第 41 节「入场用次日开盘」同一条纪律)。

⚠️ **三只事后挑出的赢家,带幸存者偏差。**
   这个对比只能用来**否定**一条规则(若在三只赢家上都失败,它就死了),
   **不能用来肯定** —— 通过了也只是候选,必须再上全市场检验。
"""
import os

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
CASES = [("301004", "嘉益股份"), ("301061", "匠心家居"), ("605499", "东鹏饮料"),
         ("688082", "盛美上海"),    # 用户反例:业绩连续五年增长,股价横盘两年半
         # 用户补充:第二段调整撞上**全市场熊市**的三只 ——
         # 泰格 2017 月线反转涨一倍 → 2018 熊市调整 → 2019 重新上涨;
         # 中际旭创 / 新易盛 2023-2025 是同一形态
         ("300347", "泰格医药"), ("300308", "中际旭创"), ("300502", "新易盛")]
HIGH_GROWTH, LAG = 0.50, 4
COST = 0.003

# 大盘状态(510300 在 MA200 之上 = 牛市),供 G 变体使用
_mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
_mk.index = _mk.index.tz_localize(None)
_mkc = pd.to_numeric(_mk["close"], errors="coerce")
MKT_OK = (_mkc > _mkc.rolling(200, min_periods=200).mean())

rows = []
for code, name in CASES:
    x = pd.read_parquet(f"{DATA}/{code}.parquet",
                        columns=["open", "high", "low", "close", "net_income"])
    x.index = x.index.tz_localize(None)
    op = pd.to_numeric(x["open"], errors="coerce").where(lambda s: s > 0)
    lo = pd.to_numeric(x["low"], errors="coerce").where(lambda s: s > 0)
    cl = pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0)
    ni = pd.to_numeric(x["net_income"], errors="coerce")
    idx = cl.index

    # ── 买入日:首次净利同比 >50% 的财报日,次日开盘成交 ──
    rd = list(ni.index[ni.diff().ne(0) & ni.notna()])
    vals = pd.Series([ni.loc[d] for d in rd], index=rd)
    yoy = vals / vals.shift(LAG) - 1
    sig = yoy[yoy > HIGH_GROWTH]
    if sig.empty:
        continue
    d_sig = sig.index[0]
    i_buy = int(idx.searchsorted(d_sig)) + 1
    while i_buy < len(idx) and not np.isfinite(op.iloc[i_buy]):
        i_buy += 1
    entry = float(op.iloc[i_buy])
    d_buy = idx[i_buy]

    # ── 10月均线(月末收盘的 10 期均线),次月首日开盘执行 ──
    m_close = cl.resample("ME").last().dropna()
    ma10m = m_close.rolling(10, min_periods=10).mean()
    below = (m_close < ma10m) & ma10m.notna()
    # 每个月末对应的「下一个交易日」
    next_td = {}
    for md in m_close.index:
        nxt = idx[idx > md]
        if len(nxt):
            next_td[md] = nxt[0]

    seg = cl.iloc[i_buy:]
    d_hi, p_hi = seg.idxmax(), seg.max()
    p_end, d_end = cl.dropna().iloc[-1], cl.dropna().index[-1]

    def result(nm, i_exit, px_exit):
        held = (i_exit - i_buy) if i_exit is not None else (len(idx) - 1 - i_buy)
        px = px_exit if px_exit is not None else p_end
        d_ex = idx[i_exit] if i_exit is not None else d_end
        ret = px / entry - 1 - 2 * COST
        return {"代码": code, "名称": name, "规则": nm, "买入日": d_buy.date(),
                "买入价": entry, "卖出日": d_ex.date(), "卖出价": px,
                "净收益": ret, "持有交易日": held,
                "吃到高点比例": (px / entry - 1) / (p_hi / entry - 1)
                if p_hi > entry else np.nan}

    # A 不止损
    rows.append(result("A 不止损", None, None))

    # B -10% 固定止损(买入价的 90%,盘中触发)
    stop = entry * 0.90
    hit = None
    for t in range(i_buy, len(idx)):
        if np.isfinite(lo.iloc[t]) and lo.iloc[t] <= stop:
            hit = t
            break
    rows.append(result("B -10%固定止损", hit,
                       (op.iloc[hit] if np.isfinite(op.iloc[hit]) and op.iloc[hit] < stop
                        else stop) if hit is not None else None))

    # C 10月均线
    hit = None
    for md in m_close.index:
        if md <= d_buy or md not in next_td:
            continue
        if bool(below.get(md, False)):
            t = int(idx.searchsorted(next_td[md]))
            if t > i_buy and np.isfinite(op.iloc[t]):
                hit = t
                break
    rows.append(result("**C 10月均线**", hit,
                       float(op.iloc[hit]) if hit is not None else None))

    # D 日线 MA200
    ma200 = cl.rolling(200, min_periods=200).mean()
    hit = None
    for t in range(i_buy + 1, len(idx) - 1):
        if np.isfinite(cl.iloc[t]) and np.isfinite(ma200.iloc[t]) and cl.iloc[t] < ma200.iloc[t]:
            if np.isfinite(op.iloc[t + 1]):
                hit = t + 1
                break
    rows.append(result("D 日线MA200", hit,
                       float(op.iloc[hit]) if hit is not None else None))

    # F 10月均线,但**浮盈 >100% 才启动**(用户提议:底部涨 N 倍之后才止损)
    hit = None
    armed = False
    for md in m_close.index:
        if md <= d_buy or md not in next_td:
            continue
        if not armed and np.isfinite(m_close[md]) and m_close[md] / entry - 1 >= 1.0:
            armed = True
        if armed and bool(below.get(md, False)):
            t = int(idx.searchsorted(next_td[md]))
            if t > i_buy and np.isfinite(op.iloc[t]):
                hit = t
                break
    rows.append(result("F 10月均线(浮盈>100%才启动)", hit,
                       float(op.iloc[hit]) if hit is not None else None))

    # G 10月均线,但**大盘在 MA200 之上时不止损**(用户提议:牛市不能止损)
    mk = MKT_OK.reindex(idx).ffill()
    hit = None
    for md in m_close.index:
        if md <= d_buy or md not in next_td:
            continue
        if bool(below.get(md, False)) and not bool(mk.get(md, False)):
            t = int(idx.searchsorted(next_td[md]))
            if t > i_buy and np.isfinite(op.iloc[t]):
                hit = t
                break
    rows.append(result("G 10月均线(牛市不止损)", hit,
                       float(op.iloc[hit]) if hit is not None else None))

    # H 用户的三段论:买入后先经历一次回撤 ≥15%、再创出持仓期新高
    #   = 「走完第二段调整、进入第三段」,此时才启动 10月均线止损。
    #   事前锁定 15%,只测这一个值,不搜索。
    hit = None
    armed, dipped, peak = False, False, entry
    for md in m_close.index:
        if md <= d_buy or md not in next_td:
            continue
        px = m_close[md]
        if not np.isfinite(px):
            continue
        if not armed:
            if px / peak - 1 <= -0.15:
                dipped = True
            if dipped and px > peak:
                armed = True          # 回撤过、又创新高 → 第三段开始
            peak = max(peak, px)
        if armed and bool(below.get(md, False)):
            t = int(idx.searchsorted(next_td[md]))
            if t > i_buy and np.isfinite(op.iloc[t]):
                hit = t
                break
    rows.append(result("H 10月均线(走完调整创新高才启动)", hit,
                       float(op.iloc[hit]) if hit is not None else None))

    # E 从最高收盘回撤 -25%
    hit, peak = None, entry
    for t in range(i_buy, len(idx) - 1):
        if not np.isfinite(cl.iloc[t]):
            continue
        peak = max(peak, cl.iloc[t])
        if cl.iloc[t] / peak - 1 <= -0.25 and np.isfinite(op.iloc[t + 1]):
            hit = t + 1
            break
    rows.append(result("E 高点回撤-25%", hit,
                       float(op.iloc[hit]) if hit is not None else None))

R = pd.DataFrame(rows)
print(f"{'='*118}\n成长股离场规则对比(买入=首次净利同比>50%的财报日次日开盘,含双边成本 0.3%)\n{'='*118}")
for code, name in CASES:
    s = R[R.代码 == code]
    if s.empty:
        continue
    hi = s.iloc[0]
    print(f"\n{code} {name}   买入 {hi.买入日} @ {hi.买入价:.2f}")
    print(f"{'规则':<18}{'卖出日':>12}{'卖出价':>9}{'净收益':>11}"
          f"{'持有交易日':>11}{'吃到高点':>10}")
    for _, r in s.iterrows():
        print(f"{r.规则:<18}{str(r.卖出日):>12}{r.卖出价:>9.2f}{r.净收益:>11.1%}"
              f"{r.持有交易日:>11}{r.吃到高点比例:>10.0%}")

print(f"\n{'='*118}\n汇总:每条规则的三只中位数\n{'='*118}")
g = R.groupby("规则").agg(中位净收益=("净收益", "median"),
                        最差=("净收益", "min"),
                        最好=("净收益", "max"),
                        中位持有日=("持有交易日", "median"),
                        中位吃到高点=("吃到高点比例", "median"))
print(g.sort_values("中位净收益", ascending=False).to_string())
R.to_csv(f"{SP}/growth_exit_rules.csv", index=False)
print(f"\n⚠️ 三只事后挑出的赢家。这个对比只能**否定**规则,不能**肯定** ——")
print("   通过了也只是候选,必须再上全市场检验。")
print(f"→ {SP}/growth_exit_rules.csv")
