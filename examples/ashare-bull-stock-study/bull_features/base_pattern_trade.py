"""B 部分:用严格基底过滤突破事件,看能不能真的赚到钱

═══ 为什么这一步不能省 ═══
A 部分给出「任一严格基底 lift 1.31」(粗定义只有 0.95),方向是对的。
但本 session 已经**四次**出现「归因显著、交易亏钱」。
lift 1.31 对应 P(牛股|形态) 仅 5.47% —— 按此买入 94.5% 的时候买到的不是牛股。
**只有交易检验能定性。**

═══ 引擎不改一行 ═══
规则 A(-10%固定止损、无止盈、最长252日)的 step/new_pos/run_trade_level/
run_portfolio 全部从 `breakout_exit_rules.py` 原样复制:
入场=突破次日开盘、止损用最低价、跳空按开盘成交、临时停牌持有穿越、
退市按最后有效价平仓、择时用 510300 的 MA200。
**唯一改变的是喂给引擎的事件集合。**

═══ 事前判据(跑之前写死,不放宽) ═══
基底过滤要算「有用」,需**同时**满足:
  1. 交易级净期望 ≥ **+6.0%/笔**(全量基线 +4.61%)
  2. 组合级年化 ≥ **+7.22%**(等权基准)
  3. 显著优于**同选中率的随机过滤**(20 个种子,p<0.05)
     —— 第四十五节的教训:少开仓本身就会改变结果,不控制这一条等于自欺

═══ 两个必须区分的基线 ═══
  全量 70,310 笔 → 锚点,必须复现 **+4.61%/笔**、组合 **+6.34%**
  dp≥575 的 60,650 笔 → **过滤的真正对照**(检测器需要 575 天历史,
  不能拿它和含早期事件的全量比)
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_pattern_detector import NEED, PRIOR, WIN, detect_base  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, SEED, N_RAND = 10, 20260810, 20
INF = float("inf")
RULE_A = dict(name="基线:-10%固定止损,无止盈,252日",
              stop=0.10, ma_mode="none", trail=None, arm=None, max_hold=252)

t0 = time.time()
o, h, l, c, mv, vo = {}, {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "high", "low", "close", "float_mv", "volume"])
    if x.empty:
        continue
    o[k] = pd.to_numeric(x["open"], errors="coerce")
    h[k] = pd.to_numeric(x["high"], errors="coerce")
    l[k] = pd.to_numeric(x["low"], errors="coerce")
    c[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    vo[k] = pd.to_numeric(x["volume"], errors="coerce")
OP = pd.DataFrame(o).sort_index(); OP.index = OP.index.tz_localize(None)
HI = pd.DataFrame(h).set_axis(OP.index); LO = pd.DataFrame(l).set_axis(OP.index)
CL = pd.DataFrame(c).set_axis(OP.index); MV = pd.DataFrame(mv).set_axis(OP.index)
VO = pd.DataFrame(vo).set_axis(OP.index)
OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
MA50 = CL.rolling(50, min_periods=50).mean()
PMIN = CL.rolling(PRIOR, min_periods=60).min().shift(1)
idx = OP.index
pos = {d: i for i, d in enumerate(idx)}
NT = len(idx)
OPa, HIa, LOa = OP.to_numpy(float), HI.to_numpy(float), LO.to_numpy(float)
CLa, MVa, MAa = CL.to_numpy(float), MV.to_numpy(float), MA50.to_numpy(float)
VOa, PMINa = VO.to_numpy(float), PMIN.to_numpy(float)
col_of = {cd: i for i, cd in enumerate(OP.columns)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv, vo

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv",
                 usecols=["code", "D"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
ev = ev[ev.code.isin(OP.columns)].copy()
ev["dp"] = ev["D"].map(pos)
ev = ev.dropna(subset=["dp"])
ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp + 1 < NT - 5].reset_index(drop=True)
print(f"可用突破事件 {len(ev):,}  ({time.time()-t0:.0f}s)")

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()

# ══════════ 引擎(自 breakout_exit_rules.py 原样复制,未改) ══════════
def step(rc, hd, t, op_t, hi_t, lo_t, cl_t, ma_t):
    stop_f, ma_mode, trail, arm = rc["stop"], rc["ma_mode"], rc["trail"], rc["arm"]
    taken_over = ma_mode in ("takeover", "arm100") and hd["armed_ma"]
    if stop_f is not None and not taken_over:
        if np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
            px = op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
            return px, "固定止损"
    if np.isfinite(hi_t) and hi_t > hd["peak"]:
        hd["peak"] = hi_t
    if arm is not None and hd["peak"] >= hd["entry"] * (1 + arm):
        if trail is not None:
            hd["armed"] = True
        if ma_mode == "arm100":
            hd["armed_ma"] = True
    if trail is not None and hd["armed"]:
        tp = hd["peak"] * (1 - trail)
        if np.isfinite(lo_t) and lo_t <= tp:
            px = op_t if (np.isfinite(op_t) and op_t < tp) else tp
            return px, "移动止盈"
    if ma_mode != "none" and np.isfinite(ma_t) and np.isfinite(cl_t):
        if ma_mode == "takeover" and not hd["armed_ma"] and cl_t > ma_t:
            hd["armed_ma"] = True
        elif hd["armed_ma"] and cl_t < ma_t:
            hd["pending"] = True
    return None, None


def new_pos(rc, entry, t):
    return {"entry": entry, "peak": entry, "t_in": t, "last": entry,
            "stop_px": entry * (1 - rc["stop"]) if rc["stop"] is not None else -INF,
            "armed": False, "armed_ma": (rc["ma_mode"] == "pure"), "pending": False}


def run_trade_level(rc, evs):
    max_hold = rc["max_hold"]
    drets, ddays = [], []
    for code, grp in evs.groupby("code", sort=False):
        ci = col_of[code]
        op, hi, lo, cl, ma = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci], MAa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            hd = new_pos(rc, entry, e)
            end = min(e + max_hold, NT - 1)
            exit_px, texit = None, None
            for t in range(e, end + 1):
                if hd["pending"]:
                    px = op[t] if np.isfinite(op[t]) else cl[t]
                    if np.isfinite(px):
                        exit_px, texit = px, t
                        break
                    hd["pending"] = False
                if not np.isfinite(cl[t]):
                    continue
                hd["last"] = cl[t]
                px, _ = step(rc, hd, t, op[t], hi[t], lo[t], cl[t], ma[t])
                if px is not None:
                    exit_px, texit = px, t
                    break
            if exit_px is None:
                texit = end
                exit_px = cl[end] if np.isfinite(cl[end]) else hd["last"]
            if not np.isfinite(exit_px) or exit_px <= 0:
                continue
            drets.append(exit_px / entry - 1)
            ddays.append(texit - e + 1)
    return np.array(drets), np.array(ddays)


def run_portfolio(rc, evs, n_slots=SLOTS, cost=COST_PF, pick="small",
                  use_timing=True, seed=SEED):
    by_day = {d: g["code"].tolist() for d, g in evs.groupby("dp")}
    rng2 = np.random.default_rng(seed)
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    n_trades, trs = 0, []
    max_hold, start = rc["max_hold"], 200
    for t in range(start, NT):
        for code in list(holds):
            hd = holds[code]
            ci = hd["ci"]
            op_t, hi_t, lo_t, cl_t, ma_t = (OPa[t, ci], HIa[t, ci], LOa[t, ci],
                                            CLa[t, ci], MAa[t, ci])
            exit_px = None
            if hd["pending"]:
                exit_px = op_t if np.isfinite(op_t) else (cl_t if np.isfinite(cl_t) else hd["last"])
            elif not np.isfinite(cl_t):
                exit_px = hd["last"]
            else:
                hd["last"] = cl_t
                exit_px, _ = step(rc, hd, t, op_t, hi_t, lo_t, cl_t, ma_t)
                if exit_px is None and t - hd["t_in"] >= max_hold:
                    exit_px = cl_t
            if exit_px is not None and np.isfinite(exit_px) and exit_px > 0:
                cash += hd["shares"] * exit_px * (1 - cost)
                trs.append(exit_px / hd["entry"] - 1)
                del holds[code]
                n_trades += 1
        cands = by_day.get(t - 1, [])
        free = n_slots - len(holds)
        if cands and free > 0 and (not use_timing or mkt_ok[t]):
            cands = [cd for cd in cands if cd not in holds
                     and np.isfinite(OPa[t, col_of[cd]]) and OPa[t, col_of[cd]] > 0]
            if cands:
                if pick == "small":
                    cands.sort(key=lambda cd: MVa[t, col_of[cd]]
                               if np.isfinite(MVa[t, col_of[cd]]) else np.inf)
                else:
                    rng2.shuffle(cands)
                for cd in cands[:free]:
                    alloc = cash / (n_slots - len(holds)) if n_slots > len(holds) else 0
                    if alloc <= 0:
                        break
                    px = OPa[t, col_of[cd]]
                    hd = new_pos(rc, px, t)
                    hd["ci"] = col_of[cd]
                    hd["shares"] = alloc * (1 - cost) / px
                    cash -= alloc
                    holds[cd] = hd
        equity[t] = cash + sum(
            hd["shares"] * (CLa[t, hd["ci"]] if np.isfinite(CLa[t, hd["ci"]]) else hd["last"])
            for hd in holds.values())
    eq = pd.Series(equity[start:], index=idx[start:])
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    return {"年化": ann, "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "最大回撤": (eq / eq.cummax() - 1).min(), "年均笔数": n_trades / yrs}


# ══════════ 锚点:全量事件必须复现 +4.61%/笔、组合 +6.34% ══════════
r_all, d_all = run_trade_level(RULE_A, ev)
net_all = r_all.mean() - COST_TRADE
pf_all = run_portfolio(RULE_A, ev)
print(f"\n锚点 全量 {len(ev):,} 笔:交易级净期望 **{net_all:+.2%}**(应 +4.61%)、"
      f"组合年化 **{pf_all['年化']:+.2%}**(应 +6.34%)  ({time.time()-t0:.0f}s)")
assert abs(net_all - 0.0461) < 0.0015, f"交易级锚点对不上:{net_all:+.4%}"
assert abs(pf_all["年化"] - 0.0634) < 0.002, f"组合级锚点对不上:{pf_all['年化']:+.4%}"
print("  锚点通过 —— 引擎与前一轮完全一致,后面只换事件集合")

# ══════════ 逐事件检测基底 ══════════
E = ev[ev.dp >= NEED].reset_index(drop=True)
flags = {k: np.zeros(len(E), bool) for k in ("cup", "flat", "dbl")}
above = {k: np.zeros(len(E), bool) for k in ("cup", "flat", "dbl")}
row_of = {}
for i, (cd, dp) in enumerate(zip(E.code.to_numpy(), E.dp.to_numpy())):
    row_of.setdefault(cd, []).append((i, int(dp)))
for cd, lst in row_of.items():
    ci = col_of[cd]
    for i, dp in lst:
        s0 = dp - WIN
        b = detect_base(CLa[s0:dp, ci], HIa[s0:dp, ci], LOa[s0:dp, ci],
                        VOa[s0:dp, ci], PMINa[s0:dp, ci])
        for k in ("cup", "flat", "dbl"):
            if b[k]:
                flags[k][i] = True
                pv = b[f"{k}_pivot"]
                # 真正的欧奈尔买点:突破日收盘站上 pivot
                above[k][i] = bool(np.isfinite(pv) and pv > 0 and CLa[dp, ci] >= pv)
print(f"\n{len(E):,} 个事件(dp≥{NEED})的基底检出率:  ({time.time()-t0:.0f}s)")
for k, nm in (("cup", "杯柄"), ("flat", "平底"), ("dbl", "双底")):
    print(f"  {nm:<4} 有基底 {flags[k].sum():>7,} ({flags[k].mean():>6.2%})   "
          f"其中突破日站上 pivot {above[k].sum():>7,} ({above[k].mean():>6.2%})")
anyb = flags["cup"] | flags["flat"] | flags["dbl"]
anyp = above["cup"] | above["flat"] | above["dbl"]
print(f"  任一   有基底 {anyb.sum():>7,} ({anyb.mean():>6.2%})   "
      f"其中突破日站上 pivot {anyp.sum():>7,} ({anyp.mean():>6.2%})")

SUBS = {"【对照】dp≥575 全部": np.ones(len(E), bool),
        "杯柄": flags["cup"], "平底": flags["flat"], "双底": flags["dbl"],
        "任一基底": anyb, "任一基底 且 站上pivot": anyp}

print(f"\n{'='*118}\n交易级 + 组合级(规则 A,引擎未改)\n{'='*118}")
print(f"{'事件子集':<26}{'事件数':>9}{'选中率':>8}{'胜率':>8}{'毛期望':>9}"
      f"{'净期望':>9}{'年化':>9}{'Sharpe':>8}{'最大回撤':>10}{'年均笔数':>9}")
rows = []
base_net = None
for nm, m in SUBS.items():
    sub = E[m]
    if len(sub) < 50:
        print(f"{nm:<26}{len(sub):>9,}   样本不足,跳过")
        continue
    r, _ = run_trade_level(RULE_A, sub)
    pf = run_portfolio(RULE_A, sub)
    net = r.mean() - COST_TRADE
    if base_net is None:
        base_net = net
    print(f"{nm:<26}{len(sub):>9,}{m.mean():>8.1%}{(r>0).mean():>8.1%}"
          f"{r.mean():>+9.2%}{net:>+9.2%}{pf['年化']:>+9.2%}{pf['Sharpe']:>8.3f}"
          f"{pf['最大回撤']:>10.1%}{pf['年均笔数']:>9.0f}   ({time.time()-t0:.0f}s)")
    rows.append({"子集": nm, "事件数": len(sub), "选中率": m.mean(), "胜率": (r > 0).mean(),
                 "毛期望": r.mean(), "净期望": net, **pf})

# ══════════ 同选中率的随机对照(第四十五节的教训) ══════════
print(f"\n{'='*118}\n随机对照:同选中率随机抽事件 × {N_RAND} 个种子\n{'='*118}")
rng = np.random.default_rng(SEED)
for nm, m in (("任一基底", anyb), ("任一基底 且 站上pivot", anyp)):
    k = int(m.sum())
    if k < 50:
        continue
    real_r, _ = run_trade_level(RULE_A, E[m])
    real_net = real_r.mean() - COST_TRADE
    real_pf = run_portfolio(RULE_A, E[m])["年化"]
    nets, anns = [], []
    for s in range(N_RAND):
        pick = np.zeros(len(E), bool)
        pick[rng.choice(len(E), k, replace=False)] = True
        rr, _ = run_trade_level(RULE_A, E[pick])
        nets.append(rr.mean() - COST_TRADE)
        anns.append(run_portfolio(RULE_A, E[pick], seed=SEED + s)["年化"])
    nets, anns = np.array(nets), np.array(anns)
    p_net = float((nets >= real_net).mean())
    p_ann = float((anns >= real_pf).mean())
    print(f"\n  {nm}(选中 {k:,} 笔,选中率 {m.mean():.1%})")
    print(f"    交易级净期望  实际 **{real_net:+.2%}**   随机 {N_RAND} 次:"
          f"中位 {np.median(nets):+.2%}  区间 [{nets.min():+.2%}, {nets.max():+.2%}]"
          f"   **p={p_net:.3f}**")
    print(f"    组合级年化    实际 **{real_pf:+.2%}**   随机 {N_RAND} 次:"
          f"中位 {np.median(anns):+.2%}  区间 [{anns.min():+.2%}, {anns.max():+.2%}]"
          f"   **p={p_ann:.3f}**")
    rows.append({"子集": f"{nm}·随机对照中位", "事件数": k, "净期望": float(np.median(nets)),
                 "年化": float(np.median(anns)), "p_净期望": p_net, "p_年化": p_ann})

print(f"\n{'='*118}\n判定(事前判据,未放宽)\n{'='*118}")
R = pd.DataFrame(rows)
for nm in ("任一基底", "任一基底 且 站上pivot"):
    row = R[R["子集"] == nm]
    ctl = R[R["子集"] == f"{nm}·随机对照中位"]
    if row.empty:
        continue
    net, ann = float(row.净期望.iloc[0]), float(row.年化.iloc[0])
    p1 = float(ctl.p_净期望.iloc[0]) if not ctl.empty else np.nan
    p2 = float(ctl.p_年化.iloc[0]) if not ctl.empty else np.nan
    c1, c2, c3 = net >= 0.060, ann >= 0.0722, (p1 < 0.05 and p2 < 0.05)
    print(f"  {nm}:")
    print(f"    ① 交易级净期望 ≥+6.0%  →  {net:+.2%}  {'✓' if c1 else '✗'}")
    print(f"    ② 组合级年化 ≥+7.22%   →  {ann:+.2%}  {'✓' if c2 else '✗'}")
    print(f"    ③ 优于同选中率随机(p<0.05 两项)→  p_笔={p1:.3f} p_年化={p2:.3f}"
          f"  {'✓' if c3 else '✗'}")
    print(f"    **{'算发现' if (c1 and c2 and c3) else '不算发现'}**")

R.to_csv(f"{SP}/base_pattern_trade.csv", index=False)
pd.DataFrame({"code": E.code, "dp": E.dp, **{k: flags[k] for k in flags},
              **{f"{k}_above": above[k] for k in above}}).to_parquet(
    f"{SP}/base_pattern_events.parquet")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: base_pattern_trade.csv, base_pattern_events.parquet")
