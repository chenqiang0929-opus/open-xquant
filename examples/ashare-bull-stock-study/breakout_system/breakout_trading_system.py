"""方向D 阶段1:突破+止损交易系统的逐日路径模拟

═══ 为什么这个测试和本session前40节都不同 ═══
前面所有回测都是:**满仓、等权、无止损、持有到调仓日**。
而欧内尔/陶博士方法的核心恰恰是**止损 + 让利润奔跑 + 经常空仓**。
我们一直在测"能不能预测牛股"(需要 IC 0.15+),
他们做的是"突破后跟随"(只需要盈亏比)。

═══ 为什么"最大涨幅分布"不能当结论 ═══
读-only 检查显示突破后252日内最大涨幅中位数 +32.9%、64.7% 达到过 +20%。
**但这是路径无关统计**:一只股票可能先跌 -8%(止损出局)再涨 +50%,
最大涨幅记 +50%,实际一分没赚。**只有逐日路径回放能给出真实答案。**

═══ 三个防前视的细节(容易做错) ═══
1. **入场用突破日次日开盘价**,不用突破日收盘 —— 突破日往往放量大涨,
   用当日收盘等于假设你能在盘中识别突破并成交
2. **止损判断用当日最低价**,不用收盘价 —— 用收盘会系统性低估止损触发率
3. **止盈判断用当日最高价**(移动止损同理)

═══ 事前写死的判据 ═══
0.3% 双边成本后 期望/笔 ≤ 0 → 证伪,不进入组合层面。
止损档位只测 3 个固定值(-7%/-8%/-10%),**不做参数优化**。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"

STOPS = [0.07, 0.08, 0.10]
MAX_HOLD = 252
TRAIL = [None, 0.15, 0.20]      # None=不用移动止损
COST_ROUNDTRIP = 0.003          # 双边 0.3%

t0 = time.time()

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv",
                 usecols=["code", "D", "year"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
print(f"突破事件 {len(ev):,},{ev.D.min().date()} ~ {ev.D.max().date()}")

# ---------------- OHLC 面板 ----------------
o, h, l, c = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    code = os.path.basename(f)[:-8]
    if code == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "high", "low", "close"])
    if x.empty:
        continue
    o[code] = pd.to_numeric(x["open"], errors="coerce")
    h[code] = pd.to_numeric(x["high"], errors="coerce")
    l[code] = pd.to_numeric(x["low"], errors="coerce")
    c[code] = pd.to_numeric(x["close"], errors="coerce")
OP = pd.DataFrame(o).sort_index()
OP.index = OP.index.tz_localize(None)
HI = pd.DataFrame(h).set_axis(OP.index)
LO = pd.DataFrame(l).set_axis(OP.index)
CL = pd.DataFrame(c).set_axis(OP.index)
MA50 = CL.rolling(50, min_periods=50).mean()
idx = OP.index
pos_of = {d: i for i, d in enumerate(idx)}
print(f"OHLC 面板 {OP.shape}  ({time.time()-t0:.0f}s)")

# 只保留有价格数据的事件
ev = ev[ev["code"].isin(OP.columns)].copy()
ev["dpos"] = ev["D"].map(pos_of)
ev = ev.dropna(subset=["dpos"])
ev["dpos"] = ev["dpos"].astype(int)
ev = ev[ev["dpos"] + 1 < len(idx) - 5]
print(f"可模拟事件 {len(ev):,}  ({time.time()-t0:.0f}s)")


def simulate(stop, trail, use_ma50):
    """逐日路径回放。返回每笔交易的毛收益(未扣成本)与持仓天数。

    入场:突破日次日开盘价
    止损:当日**最低价** <= 入场价×(1-stop) → 以止损价成交
    移动止损:当日**最高价**创新高后,回撤 trail → 以回撤价成交
    MA50:收盘跌破50日均线 → 次日开盘离场
    到期:持有 MAX_HOLD 日后按收盘离场
    """
    rets, days = [], []
    for code, dp in zip(ev["code"].to_numpy(), ev["dpos"].to_numpy()):
        e = dp + 1
        entry = OP[code].iloc[e]
        if not np.isfinite(entry) or entry <= 0:
            continue
        stop_px = entry * (1 - stop)
        peak = entry
        end = min(e + MAX_HOLD, len(idx) - 1)
        exit_px, held = None, 0
        hs = HI[code].to_numpy()
        ls = LO[code].to_numpy()
        cs = CL[code].to_numpy()
        ms = MA50[code].to_numpy()
        for t in range(e, end + 1):
            held = t - e + 1
            lo_t, hi_t, cl_t = ls[t], hs[t], cs[t]
            if not np.isfinite(cl_t):
                continue
            # 1) 固定止损(用最低价判断,不用收盘)
            if np.isfinite(lo_t) and lo_t <= stop_px:
                exit_px = stop_px
                break
            # 2) 移动止损(用最高价更新峰值)
            if trail is not None:
                if np.isfinite(hi_t) and hi_t > peak:
                    peak = hi_t
                tp = peak * (1 - trail)
                if np.isfinite(lo_t) and lo_t <= tp and tp > stop_px:
                    exit_px = tp
                    break
            # 3) 跌破50日均线
            if use_ma50 and np.isfinite(ms[t]) and cl_t < ms[t]:
                nxt = t + 1
                if nxt <= len(idx) - 1 and np.isfinite(OP[code].iloc[nxt]):
                    exit_px = OP[code].iloc[nxt]
                    held += 1
                else:
                    exit_px = cl_t
                break
        if exit_px is None:
            exit_px = cs[end]
            if not np.isfinite(exit_px):
                continue
        rets.append(exit_px / entry - 1)
        days.append(held)
    return np.array(rets), np.array(days)


print(f"\n{'='*104}")
print("阶段1:逐日路径模拟(入场=突破次日开盘;止损用最低价;止盈用最高价)")
print(f"{'='*104}")
print(f"{'止损':>6}{'移动止损':>10}{'MA50':>7}{'笔数':>8}{'胜率':>8}"
      f"{'均盈':>9}{'均亏':>9}{'盈亏比':>8}{'毛期望':>9}{'净期望':>9}{'中位天数':>9}")

results = []
for stop in STOPS:
    for trail in TRAIL:
        for use_ma in (False, True):
            if trail is None and not use_ma:
                pass    # 纯固定止损 + 到期
            r, d = simulate(stop, trail, use_ma)
            if len(r) < 100:
                continue
            win = r > 0
            avg_w = r[win].mean() if win.any() else 0
            avg_l = r[~win].mean() if (~win).any() else 0
            pl = abs(avg_w / avg_l) if avg_l != 0 else np.nan
            gross = r.mean()
            net = gross - COST_ROUNDTRIP
            results.append({"stop": stop, "trail": trail, "ma50": use_ma,
                            "n": len(r), "winrate": win.mean(), "avg_win": avg_w,
                            "avg_loss": avg_l, "pl_ratio": pl,
                            "gross": gross, "net": net, "med_days": np.median(d)})
            print(f"{stop:>6.0%}{(f'{trail:.0%}' if trail else '—'):>10}"
                  f"{('是' if use_ma else '否'):>7}{len(r):>8,}{win.mean():>8.1%}"
                  f"{avg_w:>+9.1%}{avg_l:>+9.1%}{pl:>8.2f}{gross:>+9.2%}{net:>+9.2%}"
                  f"{np.median(d):>9.0f}")

R = pd.DataFrame(results)
R.to_csv(f"{SP}/breakout_system_stage1.csv", index=False)

best = R.loc[R["net"].idxmax()]
print(f"\n{'='*104}")
print(f"净期望最高的配置:止损 {best['stop']:.0%}、"
      f"移动止损 {best['trail'] if best['trail'] else '无'}、MA50 {'是' if best['ma50'] else '否'}")
print(f"  笔数 {int(best['n']):,}  胜率 {best['winrate']:.1%}  "
      f"盈亏比 {best['pl_ratio']:.2f}  **净期望/笔 {best['net']:+.2%}**  "
      f"中位持仓 {best['med_days']:.0f} 天")
print(f"{'='*104}")
if best["net"] <= 0:
    print("\n**判据:0.3%成本后净期望 <= 0 → 证伪,不进入组合层面。**")
else:
    print(f"\n净期望为正,通过阶段1判据。年化粗估(按中位持仓 "
          f"{best['med_days']:.0f} 天、资金连续周转):"
          f" {(1+best['net'])**(252/max(best['med_days'],1))-1:+.1%}")
    print("  注:该粗估假设资金100%连续周转,实际受可用突破事件数限制,"
          "真实年化见阶段2。")

# ---------------- 抽查:人工确认路径逻辑 ----------------
print(f"\n{'='*104}\n抽查:随机5笔交易逐日核对(止损/止盈触发是否正确)\n{'='*104}")
rng = np.random.default_rng(20260810)
stop, trail, use_ma = best["stop"], best["trail"], bool(best["ma50"])
smp = ev.sample(5, random_state=11)
for _, row in smp.iterrows():
    code, dp = row["code"], int(row["dpos"])
    e = dp + 1
    entry = OP[code].iloc[e]
    if not np.isfinite(entry):
        continue
    sp = entry * (1 - stop)
    peak, exit_px, exit_t, why = entry, None, None, "到期"
    for t in range(e, min(e + MAX_HOLD, len(idx) - 1) + 1):
        lo_t, hi_t, cl_t = LO[code].iloc[t], HI[code].iloc[t], CL[code].iloc[t]
        if not np.isfinite(cl_t):
            continue
        if np.isfinite(lo_t) and lo_t <= sp:
            exit_px, exit_t, why = sp, t, "固定止损"
            break
        if trail is not None and trail == trail:
            if np.isfinite(hi_t) and hi_t > peak:
                peak = hi_t
            tp = peak * (1 - trail)
            if np.isfinite(lo_t) and lo_t <= tp and tp > sp:
                exit_px, exit_t, why = tp, t, "移动止损"
                break
        if use_ma:
            m = MA50[code].iloc[t]
            if np.isfinite(m) and cl_t < m:
                exit_px, exit_t, why = cl_t, t, "跌破MA50"
                break
    if exit_px is None:
        exit_t = min(e + MAX_HOLD, len(idx) - 1)
        exit_px = CL[code].iloc[exit_t]
    print(f"  [{code}] 突破 {idx[dp].date()} → 入场 {idx[e].date()} @{entry:.2f}  "
          f"止损线 {sp:.2f}  峰值 {peak:.2f}")
    print(f"          离场 {idx[exit_t].date()} @{exit_px:.2f} ({why})  "
          f"毛收益 {exit_px/entry-1:+.1%}  持有 {exit_t-e+1} 天")

print(f"\n耗时 {time.time()-t0:.0f}s   Saved: breakout_system_stage1.csv")
