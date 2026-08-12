"""口袋支点(Pocket Pivot)买点 —— 两边都没测过的那一个

═══ 为什么值得单独做 ═══
DeepSeek 的 W 章自己写着(第 563 行):
「陶博士体系的基本面筛选 / **口袋支点买点** / 题材催化**未量化**」。
我这边到第五十四节为止测的全是「突破新高」类买点。
**口袋支点是双方都没碰过的空白。**

═══ 定义(Gil Morales / Chris Kacher《Trade Like an O'Neil Disciple》) ═══
口袋支点不是突破新高,而是**基底内部或上升趋势中的提前买点** ——
在别人还看不见的时候,机构已经在买。判据:

  1. **当日上涨**(close > 昨收)
  2. **当日成交量 > 过去10个交易日里「任何一个下跌日」的最大成交量**
     ← 这是全部定义的核心:买盘第一次压过最近所有卖盘
  3. 收盘位于当日振幅的**上半部**(close 在 (high+low)/2 之上)
  4. 股价在 **MA50 之上**,且 **MA50 向上**(处在建设性形态里,不是下跌途中)
  5. **不追高**:收盘距 MA50 不超过 +10%(口袋支点定义在基底里,不在拉升途中)

与突破买点的关键差别:**第 2 条只比「近期卖压」,不要求创新高。**
所以它必然更早、更频繁,也必然混进更多假信号 —— 这正是要量化的。

═══ 口径与前面完全一致(否则不可比) ═══
入场 = 信号日**次日开盘**;规则 A(-10%固定止损、无止盈、最长252日);
组合级 10 仓位、小市值优先、510300 的 MA200 择时、0.3% 成本。
**锚点**:60日新高买点必须复现 70,310 笔 / +4.61%/笔 / 组合 +6.34%。

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 交易级净期望 ≥ **+6.0%/笔**(基线 +4.61%)
  ② 组合级年化 ≥ **+7.22%**(等权基准)
  ③ 显著优于**同数量随机抽取**的对照(20 个种子,两项 p<0.05)
三条缺一不可。另测两个最小间隔(10日 / 60日),因为口袋支点本来就成簇出现,
**这两格算一次搜索,一并纳入判据③**。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, SEED, N_RAND = 10, 20260810, 20
INF = float("inf")

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
idx = OP.index
NT = len(idx)
OPa, HIa, LOa = OP.to_numpy(float), HI.to_numpy(float), LO.to_numpy(float)
CLa, MVa, MA50a = CL.to_numpy(float), MV.to_numpy(float), MA50.to_numpy(float)
col_of = {cd: i for i, cd in enumerate(OP.columns)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv, vo

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()

# ══════════ 口袋支点信号 ══════════
prev_c = CL.shift(1)
up = CL > prev_c
down = CL < prev_c
# 下跌日的成交量(非下跌日置 0),再取过去10日最大 —— **不含当日**
dn_vol = VO.where(down, 0.0)
max_dn10 = dn_vol.rolling(10, min_periods=5).max().shift(1)
cond_vol = VO > max_dn10
cond_pos = CL > (HI + LO) / 2                       # 收在振幅上半部
cond_ma = (CL > MA50) & (MA50 > MA50.shift(10))     # MA50 之上且 MA50 向上
cond_ext = (CL / MA50 - 1) <= 0.10                  # 不追高
PP = (up & cond_vol & cond_pos & cond_ma & cond_ext).to_numpy()
# 各条件单独的命中率(用于判断是哪一条在起作用)
for nm, m in (("当日上涨", up), ("量>10日内最大下跌日量", cond_vol),
              ("收在上半部", cond_pos), ("MA50上方且MA50向上", cond_ma),
              ("距MA50 ≤+10%", cond_ext)):
    print(f"  条件命中率 {nm:<24}{np.nanmean(m.to_numpy()):>7.2%}")
print(f"  五条全中(逐日逐股)          {PP.mean():>7.2%}")

BASE_MAX_RANGE, MIN_GAP, FWD_WIN = 0.50, 60, 252
# 与原事件脚本严格对齐(锚点没过时查出来的):rolling 用 min_periods=60,
# 且丢弃最后 252 个交易日的事件(原脚本对 fwd_gain 为 NaN 的事件 continue)。
# 口袋支点事件也用同一个截断,否则两组事件的可交易区间不同,没法比。
_rmax60 = CL.rolling(60, min_periods=60).max()
_rmin60 = CL.rolling(60, min_periods=60).min()
base60 = ((_rmax60 - _rmin60) / _rmin60.replace(0, np.nan)).shift(1)
BASE_OK = (base60 < BASE_MAX_RANGE).to_numpy()
BRK60 = (CLa > _rmax60.shift(1).to_numpy()) & BASE_OK
LAST_OK = NT - 1 - FWD_WIN


def to_events(hit, gap):
    codes, dps = [], []
    for j, cd in enumerate(OP.columns):
        last = -10**9
        for q in np.flatnonzero(hit[:, j]):
            if q - last < gap or q == 0 or q > LAST_OK:
                continue
            last = q
            codes.append(cd); dps.append(int(q))
    return pd.DataFrame({"code": codes, "dp": dps})


# ══════════ 引擎(原样) ══════════
def step(rc, hd, t, op_t, hi_t, lo_t, cl_t):
    if rc["stop"] is not None and np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
        return op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
    if np.isfinite(hi_t) and hi_t > hd["peak"]:
        hd["peak"] = hi_t
    return None


def new_pos(rc, entry, t):
    return {"entry": entry, "peak": entry, "t_in": t, "last": entry,
            "stop_px": entry * (1 - rc["stop"]) if rc["stop"] is not None else -INF}


def run_trade(rc, evs):
    out = []
    for code, grp in evs.groupby("code", sort=False):
        ci = col_of[code]
        op, hi, lo, cl = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            hd = new_pos(rc, entry, e)
            end = min(e + rc["max_hold"], NT - 1)
            ex = None
            for t in range(e, end + 1):
                if not np.isfinite(cl[t]):
                    continue
                hd["last"] = cl[t]
                ex = step(rc, hd, t, op[t], hi[t], lo[t], cl[t])
                if ex is not None:
                    break
            if ex is None:
                ex = cl[end] if np.isfinite(cl[end]) else hd["last"]
            if np.isfinite(ex) and ex > 0:
                out.append(ex / entry - 1)
    return np.array(out)


def run_pf(rc, evs, seed=SEED):
    by_day = {d: g["code"].tolist() for d, g in evs.groupby("dp")}
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    n_tr, start = 0, 200
    for t in range(start, NT):
        for code in list(holds):
            hd = holds[code]
            ci = hd["ci"]
            op_t, hi_t, lo_t, cl_t = OPa[t, ci], HIa[t, ci], LOa[t, ci], CLa[t, ci]
            ex = None
            if not np.isfinite(cl_t):
                ex = hd["last"]
            else:
                hd["last"] = cl_t
                ex = step(rc, hd, t, op_t, hi_t, lo_t, cl_t)
                if ex is None and t - hd["t_in"] >= rc["max_hold"]:
                    ex = cl_t
            if ex is not None and np.isfinite(ex) and ex > 0:
                cash += hd["shares"] * ex * (1 - COST_PF)
                del holds[code]
                n_tr += 1
        cands = by_day.get(t - 1, [])
        free = SLOTS - len(holds)
        if cands and free > 0 and mkt_ok[t]:
            cands = [cd for cd in cands if cd not in holds
                     and np.isfinite(OPa[t, col_of[cd]]) and OPa[t, col_of[cd]] > 0]
            cands.sort(key=lambda cd: MVa[t, col_of[cd]]
                       if np.isfinite(MVa[t, col_of[cd]]) else np.inf)
            for cd in cands[:free]:
                alloc = cash / (SLOTS - len(holds)) if SLOTS > len(holds) else 0
                if alloc <= 0:
                    break
                px = OPa[t, col_of[cd]]
                hd = new_pos(rc, px, t)
                hd["ci"] = col_of[cd]
                hd["shares"] = alloc * (1 - COST_PF) / px
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
            "最大回撤": (eq / eq.cummax() - 1).min(), "年均笔数": n_tr / yrs}


RULE_A = dict(stop=0.10, max_hold=252)
SETS = {"【锚点】60日新高突破": to_events(BRK60, 60),
        "口袋支点(最小间隔60日)": to_events(PP, 60),
        "口袋支点(最小间隔10日)": to_events(PP, 10),
        "口袋支点 ∩ 60日新高": to_events(PP & BRK60, 60)}

print(f"\n{'='*118}\n交易级 + 组合级(规则 A,与前几节完全同口径)\n{'='*118}")
print(f"{'买点':<30}{'事件数':>9}{'胜率':>8}{'毛期望':>9}{'净期望':>9}"
      f"{'年化':>9}{'Sharpe':>8}{'最大回撤':>10}{'年均笔数':>9}")
rows = {}
for nm, e in SETS.items():
    if len(e) < 50:
        print(f"{nm:<30}{len(e):>9,}   样本不足")
        continue
    r = run_trade(RULE_A, e)
    pf = run_pf(RULE_A, e)
    net = r.mean() - COST_TRADE
    rows[nm] = {"事件数": len(e), "胜率": (r > 0).mean(), "毛期望": r.mean(),
                "净期望": net, **pf}
    print(f"{nm:<30}{len(e):>9,}{(r>0).mean():>8.1%}{r.mean():>+9.2%}{net:>+9.2%}"
          f"{pf['年化']:>+9.2%}{pf['Sharpe']:>8.3f}{pf['最大回撤']:>10.1%}"
          f"{pf['年均笔数']:>9.0f}   ({time.time()-t0:.0f}s)")
    if nm.startswith("【锚点】"):
        print(f"    锚点核对:{len(e):,} 笔(应 70,310)、{net:+.2%}(应 +4.61%)、"
              f"{pf['年化']:+.2%}(应 +6.34%)")
        assert abs(len(e) - 70310) <= 50, f"事件数不符:{len(e)}"
        assert abs(net - 0.0461) < 0.0015, f"交易级锚点对不上:{net:+.4%}"
        assert abs(pf["年化"] - 0.0634) < 0.002, f"组合级锚点对不上:{pf['年化']:+.4%}"
        print("    锚点通过")

# ══════════ 随机对照 ══════════
print(f"\n{'='*118}\n随机对照:从 60日新高事件里随机抽同样多的笔数 × {N_RAND} 次\n{'='*118}")
base_ev = SETS["【锚点】60日新高突破"]
rng = np.random.default_rng(SEED)
best_nm = max([n for n in rows if not n.startswith("【锚点】")],
              key=lambda n: rows[n]["年化"])
for nm in [n for n in rows if not n.startswith("【锚点】")]:
    k = min(rows[nm]["事件数"], len(base_ev))
    nets, anns = [], []
    for s in range(N_RAND):
        sub = base_ev.iloc[rng.choice(len(base_ev), k, replace=False)]
        nets.append(run_trade(RULE_A, sub).mean() - COST_TRADE)
        anns.append(run_pf(RULE_A, sub, seed=SEED + s)["年化"])
    nets, anns = np.array(nets), np.array(anns)
    p1 = float((nets >= rows[nm]["净期望"]).mean())
    p2 = float((anns >= rows[nm]["年化"]).mean())
    rows[nm]["p_净期望"], rows[nm]["p_年化"] = p1, p2
    print(f"  {nm}(抽 {k:,} 笔)")
    print(f"    净期望 实际 **{rows[nm]['净期望']:+.2%}**  随机中位 {np.median(nets):+.2%}"
          f"  [{nets.min():+.2%}, {nets.max():+.2%}]  **p={p1:.3f}**")
    print(f"    年化   实际 **{rows[nm]['年化']:+.2%}**  随机中位 {np.median(anns):+.2%}"
          f"  [{anns.min():+.2%}, {anns.max():+.2%}]  **p={p2:.3f}**")

print(f"\n{'='*118}\n判定(事前判据,未放宽)\n{'='*118}")
for nm in [n for n in rows if not n.startswith("【锚点】")]:
    v = rows[nm]
    c1, c2 = v["净期望"] >= 0.060, v["年化"] >= 0.0722
    c3 = v.get("p_净期望", 1) < 0.05 and v.get("p_年化", 1) < 0.05
    print(f"  {nm}:")
    print(f"    ① 净期望 ≥+6.0%   → {v['净期望']:+.2%}  {'✓' if c1 else '✗'}")
    print(f"    ② 年化 ≥+7.22%    → {v['年化']:+.2%}  {'✓' if c2 else '✗'}")
    print(f"    ③ 优于同数量随机  → p_笔={v.get('p_净期望', np.nan):.3f} "
          f"p_年化={v.get('p_年化', np.nan):.3f}  {'✓' if c3 else '✗'}")
    print(f"    **{'算发现' if (c1 and c2 and c3) else '不算发现'}**")

# 分段:口袋支点在 2015-05 之后是否也塌
print(f"\n{'='*118}\n分段(第五十四节发现突破系统的钱几乎全在 2015-05 之前)\n{'='*118}")
for nm, e in SETS.items():
    for tag, sub in (("2015-05前", e[e.dp < 575]), ("2015-05后", e[e.dp >= 575])):
        if len(sub) < 50:
            continue
        r = run_trade(RULE_A, sub)
        print(f"  {nm:<30}{tag:<12}{len(sub):>8,} 笔   净期望 {r.mean()-COST_TRADE:>+8.2%}")

pd.DataFrame(rows).T.to_csv(f"{SP}/pocket_pivot.csv")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: pocket_pivot.csv")
