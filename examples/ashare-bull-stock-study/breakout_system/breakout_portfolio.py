"""方向D 阶段2+3+4:把突破交易规则做成可执行组合

═══ 阶段1 的结论与其限制 ═══
最优配置(10%止损、无移动止损、不止盈)净期望 +4.00%/笔,胜率 18%,盈亏比 6.98。
修正两处后约 +3.2%/笔:
  - 跳空穿越止损线 12.1%,实际成交比理想差 0.30pp → 整体 -0.24pp
  - 退市/长停丢失 1.13% 样本(几乎全亏)→ 约 -0.6pp

**但 +3.2%/笔 不能直接变成年化**:12年 70,310 笔、中位持仓37天,
资金和持仓数都吃不下。阶段1 那句"年化+30.6%"按100%连续周转算,是误导。

═══ 本脚本要回答的 ═══
限制同时持仓 N 只之后,真实年化是多少?能不能跑赢等权基准
(修复后 OOS 年化 7.22% / Sharpe 0.423)?

═══ 三个必须有的对照 ═══
1. **随机选** —— 突破机会多于空位时随机挑。若随机也行,说明 alpha 来自
   **交易规则**而非选股,这是个重要区分
2. **大盘择时(欧内尔的M)** —— 指数跌破200日均线则不开新仓。
   本session所有回测都是满仓,从未测过这一层
3. **成本三档** —— 0.1%/0.3%/0.5% 双边。突破日放量大涨,滑点应高于平均

═══ 已修正的两个乐观假设 ═══
- 跳空穿越止损线时,**以当日开盘价成交**而非止损价
- 价格序列中断(退市/长停)时,**按最后有效价平仓**而非跳过该笔
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"

STOP = 0.10
MAX_HOLD = 252
SLOTS = [5, 10, 20]
COSTS = [0.001, 0.003, 0.005]
SEED = 20260810

t0 = time.time()

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv",
                 usecols=["code", "D"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)

o, h, l, c, mv = {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "high", "low", "close", "float_mv"])
    if x.empty:
        continue
    o[k] = pd.to_numeric(x["open"], errors="coerce")
    h[k] = pd.to_numeric(x["high"], errors="coerce")
    l[k] = pd.to_numeric(x["low"], errors="coerce")
    c[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
OP = pd.DataFrame(o).sort_index()
OP.index = OP.index.tz_localize(None)
# 源数据含负价格(200418 有136行、000418 有3行,全在2013年,源表 hfq_close 亦为NaN
# 说明数据商自己标记了异常但未清理 close 列)。非正价格会让 pct_change 产生 inf,
# 进而把横截面平均污染成 -inf —— 这正是首次运行时市场择时恒为False、0笔交易的原因。
for _df in (OP,):
    pass
HI = pd.DataFrame(h).set_axis(OP.index)
LO = pd.DataFrame(l).set_axis(OP.index)
CL = pd.DataFrame(c).set_axis(OP.index)
MV = pd.DataFrame(mv).set_axis(OP.index)
# 统一清洗:所有非正价格置为 NaN
OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
idx = OP.index
pos = {d: i for i, d in enumerate(idx)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")

ev = ev[ev.code.isin(OP.columns)].copy()
ev["dp"] = ev["D"].map(pos)
ev = ev.dropna(subset=["dp"])
ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp + 1 < len(idx) - 5]
by_day = {d: g["code"].tolist() for d, g in ev.groupby("dp")}
print(f"可用突破事件 {len(ev):,},覆盖 {len(by_day):,} 个交易日")

# 市场择时:全市场等权指数的200日均线
# 择时基准改用 510300(真实指数,此前已验证0个异常日)。
# **订正**:首版用"横截面日收益中位数逐日累乘"构造指数,得到年化 -37%、
# 仅2.8%的天数在MA200之上 —— 那不是数据问题(实测全市场5,078只各自年化
# 中位数 +3.41%,600519 +19.2%,300750 +45.5%),而是**该构造方法本身无效**:
# 每日参与计算的股票集合在变,且"收益的中位数"≠"中位数股票的收益"。
_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ma200 = mkt.rolling(200, min_periods=200).mean()
mkt_ok = (mkt > mkt_ma200).to_numpy()
print(f"择时基准 510300: {mkt.dropna().iloc[0]:.2f} → {mkt.dropna().iloc[-1]:.2f}, "
      f"在MA200之上的比例 {np.nanmean(mkt_ok):.1%}  (合理值应在40-60%)")
assert 0.25 < np.nanmean(mkt_ok) < 0.80, "择时基准比例异常,请检查指数数据"

OPa, HIa, LOa, CLa, MVa = OP.to_numpy(), HI.to_numpy(), LO.to_numpy(), CL.to_numpy(), MV.to_numpy()
col_of = {cd: i for i, cd in enumerate(OP.columns)}


def run(n_slots, cost, pick, use_timing, seed=SEED):
    """逐日推进的组合模拟。

    pick: 'small'=选流通市值最小的 / 'random'=随机 / 'first'=按代码序(中性对照)
    每只等额分配 1/n_slots 资金;平仓后资金释放,可用于新开仓。
    """
    rng = np.random.default_rng(seed)
    cash, holds = 1.0, {}          # code -> dict(entry, shares, stop, t_in)
    equity = np.zeros(len(idx))
    n_trades = 0
    trade_rets = []
    start = 200                     # 等 MA200 就绪
    for t in range(start, len(idx)):
        # --- 先处理平仓 ---
        for code in list(holds):
            hd = holds[code]
            ci = col_of[code]
            lo_t, cl_t, op_t = LOa[t, ci], CLa[t, ci], OPa[t, ci]
            exit_px = None
            if not np.isfinite(cl_t):
                # 价格中断(退市/长停):按最后有效价平仓,不跳过
                exit_px = hd["last"]
            elif np.isfinite(lo_t) and lo_t <= hd["stop"]:
                # 跳空穿越止损线 → 以开盘价成交(已修正的乐观假设)
                exit_px = op_t if (np.isfinite(op_t) and op_t < hd["stop"]) else hd["stop"]
            elif t - hd["t_in"] >= MAX_HOLD:
                exit_px = cl_t
            if np.isfinite(cl_t):
                hd["last"] = cl_t
            if exit_px is not None:
                cash += hd["shares"] * exit_px * (1 - cost)
                trade_rets.append(exit_px / hd["entry"] - 1)
                del holds[code]
                n_trades += 1

        # --- 再处理开仓(用昨日的突破事件,今日开盘入场) ---
        cands = by_day.get(t - 1, [])
        free = n_slots - len(holds)
        if cands and free > 0 and (not use_timing or mkt_ok[t]):
            cands = [cd for cd in cands if cd not in holds
                     and np.isfinite(OPa[t, col_of[cd]]) and OPa[t, col_of[cd]] > 0]
            if cands:
                if pick == "small":
                    cands.sort(key=lambda cd: MVa[t, col_of[cd]]
                               if np.isfinite(MVa[t, col_of[cd]]) else np.inf)
                elif pick == "random":
                    rng.shuffle(cands)
                for cd in cands[:free]:
                    alloc = cash / (n_slots - len(holds)) if n_slots > len(holds) else 0
                    if alloc <= 0:
                        break
                    px = OPa[t, col_of[cd]]
                    sh = alloc * (1 - cost) / px
                    cash -= alloc
                    holds[cd] = {"entry": px, "shares": sh, "stop": px * (1 - STOP),
                                 "t_in": t, "last": px}

        mtm = sum(hd["shares"] * (CLa[t, col_of[cd]] if np.isfinite(CLa[t, col_of[cd]])
                                  else hd["last"]) for cd, hd in holds.items())
        equity[t] = cash + mtm
    eq = pd.Series(equity[start:], index=idx[start:])
    return eq, n_trades, np.array(trade_rets)


def stats(eq, label, n_trades, tr=None):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    d = {"配置": label, "年化": ann, "Sharpe": sharpe, "最大回撤": mdd,
         "交易笔数": n_trades, "年均笔数": n_trades / yrs}
    if tr is not None and len(tr):
        d["笔均收益"] = tr.mean(); d["胜率"] = (tr > 0).mean()
    return d


print(f"\n{'='*112}\n阶段2:限制同时持仓数(止损{STOP:.0%}、无止盈、成本0.3%双边)\n{'='*112}")
rows = []
for n in SLOTS:
    for pick in ("small", "random"):
        eq, nt, tr = run(n, 0.003, pick, False)
        rows.append(stats(eq, f"{n}只 / {'小市值优先' if pick=='small' else '随机选'}", nt, tr))
        print(f"  {rows[-1]['配置']:<22} 年化 {rows[-1]['年化']:+7.2%}  "
              f"Sharpe {rows[-1]['Sharpe']:+6.3f}  回撤 {rows[-1]['最大回撤']:7.2%}  "
              f"年均 {rows[-1]['年均笔数']:.0f} 笔  笔均 {rows[-1].get('笔均收益',float('nan')):+.2%}"
              f"  胜率 {rows[-1].get('胜率',float('nan')):.1%}  ({time.time()-t0:.0f}s)")

print(f"\n{'='*112}\n阶段3:叠加大盘择时(等权指数跌破200日均线则不开新仓)\n{'='*112}")
for n in SLOTS:
    eq, nt, tr = run(n, 0.003, "small", True)
    rows.append(stats(eq, f"{n}只 / 小市值 + 择时", nt, tr))
    print(f"  {rows[-1]['配置']:<22} 年化 {rows[-1]['年化']:+7.2%}  "
          f"Sharpe {rows[-1]['Sharpe']:+6.3f}  回撤 {rows[-1]['最大回撤']:7.2%}  "
          f"年均 {rows[-1]['年均笔数']:.0f} 笔")

print(f"\n{'='*112}\n阶段4:成本敏感性(10只 / 小市值 + 择时)\n{'='*112}")
for cost in COSTS:
    eq, nt, tr = run(10, cost, "small", True)
    s = stats(eq, f"成本 {cost:.1%} 双边", nt, tr)
    rows.append(s)
    print(f"  {s['配置']:<22} 年化 {s['年化']:+7.2%}  Sharpe {s['Sharpe']:+6.3f}  "
          f"回撤 {s['最大回撤']:7.2%}  年均 {s['年均笔数']:.0f} 笔")

pd.DataFrame(rows).to_csv(f"{SP}/breakout_portfolio_results.csv", index=False)
print(f"\n{'='*112}")
print("对照:全市场等权基准(修复后) OOS 年化 7.22% / Sharpe 0.423")
print("      BP+3闸门(旧'最佳配置')     OOS 年化 6.64% / Sharpe 0.420")
print(f"{'='*112}")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: breakout_portfolio_results.csv")
