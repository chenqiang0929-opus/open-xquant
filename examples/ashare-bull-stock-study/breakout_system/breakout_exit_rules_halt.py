"""停牌/退市处理对结论的影响 —— 三种处理方式对比

═══ 为什么必须单独查这一项 ═══
诊断脚本 diag_ruleA_vs_stage1.py 实测:交易级 607 笔(0.87%)"到期日收盘为 NaN"
的持仓,按最后有效成交价计**隐含平均收益 +103.0%**,把基线净期望从 +3.75%
抬到 +4.61%(+0.86pp)。组合级用 50% 折价做压力测试时,**所有规则(含基线)
全线转负**(基线 +4.69% → -6.35%)。0.87% 的样本决定整个方向的正负,
说明这一处的假设比任何离场规则都重要。

═══ 但 50% 折价那版压力测试本身有缺陷 ═══
它对**任何**价格中断都强制平仓并打折,包括只停几天的临时停牌 —— 显然不对。
A股临时停牌极常见(重大事项、股东大会),复牌后继续交易,不该按退市处理。

═══ 本脚本的三种处理 ═══
  (a) 现行:任何 NaN → 按最后有效价平仓(= 前面主表的口径)
  (b) 修正:**临时停牌持有穿越**(不平仓、不判止损);
           **永久终止**(该股价格序列此后再无有效值)→ 按最后有效价平仓
  (c) 修正 + 退市折价:同 (b),但永久终止按 最后有效价 × 50% 平仓

(b) 是三者中最接近真实的:停牌期间既卖不掉也不该被止损扫出;
真正退市才是不可逆的损失。(c) 给退市损失一个保守下界。

═══ 判据 ═══
若 (b)/(c) 下基线与规则D 的排序与主表一致,主表结论成立;
若排序翻转,则**主表结论依赖于停牌处理假设,必须降级表述**。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
N_SEED, SLOTS, COST = 20, 10, 0.003
COST_TRADE = 0.003
INF = float("inf")

RULES = {
    "A": dict(name="基线:-10%固定,无止盈,252日", stop=0.10, ma_mode="none",
              trail=None, arm=None, max_hold=252),
    "B": dict(name="纯10周线", stop=None, ma_mode="pure", trail=None, arm=None, max_hold=252),
    "C": dict(name="固定+10周线接管", stop=0.10, ma_mode="takeover",
              trail=None, arm=None, max_hold=252),
    "D": dict(name="条件移动止盈(涨100%后)", stop=0.10, ma_mode="none",
              trail=0.20, arm=1.00, max_hold=252),
    "E": dict(name="条件+10周线(涨100%后)", stop=0.10, ma_mode="arm100",
              trail=None, arm=1.00, max_hold=252),
    "F": dict(name="基线+504日", stop=0.10, ma_mode="none", trail=None, arm=None, max_hold=504),
}
# (标签, 临时停牌是否持有穿越, 永久终止折价)
MODES = [("(a) 现行:任何中断按最后价平仓", False, 1.0),
         ("(b) 修正:停牌穿越 + 退市按最后价", True, 1.0),
         ("(c) 修正 + 退市折价50%", True, 0.5)]

t0 = time.time()
o, h, l, c, mv = {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "high", "low", "close", "float_mv"])
    if x.empty:
        continue
    o[k] = pd.to_numeric(x["open"], errors="coerce"); h[k] = pd.to_numeric(x["high"], errors="coerce")
    l[k] = pd.to_numeric(x["low"], errors="coerce"); c[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
OP = pd.DataFrame(o).sort_index(); OP.index = OP.index.tz_localize(None)
HI = pd.DataFrame(h).set_axis(OP.index); LO = pd.DataFrame(l).set_axis(OP.index)
CL = pd.DataFrame(c).set_axis(OP.index); MV = pd.DataFrame(mv).set_axis(OP.index)
OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
MA50 = CL.rolling(50, min_periods=50).mean()
idx = OP.index; NT = len(idx); pos = {d: i for i, d in enumerate(idx)}
OPa, HIa, LOa = OP.to_numpy(), HI.to_numpy(), LO.to_numpy()
CLa, MVa, MAa = CL.to_numpy(), MV.to_numpy(), MA50.to_numpy()
col_of = {cd: i for i, cd in enumerate(OP.columns)}
del o, h, l, c, mv

# 每只股票最后一个有效收盘的位置 —— 用于区分"临时停牌"与"永久终止"
fin = np.isfinite(CLa)
last_valid = np.where(fin.any(axis=0), NT - 1 - np.argmax(fin[::-1], axis=0), -1)
n_delist = int((last_valid < NT - 1).sum())
print(f"面板 {OP.shape},其中 {n_delist:,} 只({n_delist/OP.shape[1]:.1%})"
      f"在样本结束前已终止交易  ({time.time()-t0:.0f}s)")

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv", usecols=["code", "D"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
ev = ev[ev.code.isin(OP.columns)].copy()
ev["dp"] = ev["D"].map(pos); ev = ev.dropna(subset=["dp"]); ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp + 1 < NT - 5]
by_day = {d: g["code"].tolist() for d, g in ev.groupby("dp")}

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()


def step(rc, hd, op_t, hi_t, lo_t, cl_t, ma_t):
    stop_f, ma_mode, trail, arm = rc["stop"], rc["ma_mode"], rc["trail"], rc["arm"]
    taken = ma_mode in ("takeover", "arm100") and hd["armed_ma"]
    if stop_f is not None and not taken and np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
        return op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
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
            return op_t if (np.isfinite(op_t) and op_t < tp) else tp
    if ma_mode != "none" and np.isfinite(ma_t) and np.isfinite(cl_t):
        if ma_mode == "takeover" and not hd["armed_ma"] and cl_t > ma_t:
            hd["armed_ma"] = True
        elif hd["armed_ma"] and cl_t < ma_t:
            hd["pending"] = True
    return None


def new_pos(rc, px, t, ci):
    return {"entry": px, "peak": px, "t_in": t, "last": px, "ci": ci,
            "stop_px": px * (1 - rc["stop"]) if rc["stop"] is not None else -INF,
            "armed": False, "armed_ma": rc["ma_mode"] == "pure", "pending": False}


# ══════════ 交易级 ══════════
def trade_level(rc, hold_through, haircut):
    mh = rc["max_hold"]
    out = []
    for code, grp in ev.groupby("code", sort=False):
        ci = col_of[code]
        op, hi, lo, cl, ma = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci], MAa[:, ci]
        lv = last_valid[ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            hd = new_pos(rc, entry, e, ci)
            end = min(e + mh, NT - 1)
            exit_px = None
            t = e
            while t <= NT - 1:
                if not np.isfinite(cl[t]):
                    # 无价格的一天。**必须先判永久终止再判到期**:
                    # 初版把"t > end"的判断放在前面,导致"到期日恰逢已退市"的持仓
                    # 直接 break 到循环外的兜底分支,**绕过了退市折价** ——
                    # 而那正是折价要作用的那批(占0.87%、隐含+103%)。
                    if t > lv:                       # 永久终止:此后再无有效价
                        exit_px = hd["last"] * haircut
                        break
                    if not hold_through:             # 现行处理:任何中断即平仓
                        exit_px = hd["last"]
                        break
                    t += 1                           # 临时停牌:持有穿越,不判止损
                    continue
                hd["last"] = cl[t]
                if hd["pending"]:
                    exit_px = op[t] if np.isfinite(op[t]) else cl[t]
                    break
                if t > end:                          # 停牌顺延后的复牌首日:立即离场
                    exit_px = cl[t]
                    break
                px = step(rc, hd, op[t], hi[t], lo[t], cl[t], ma[t])
                if px is not None:
                    exit_px = px
                    break
                if t == end:                         # 到期且有价:按当日收盘
                    exit_px = cl[t]
                    break
                t += 1
            if exit_px is None:                      # 跑到面板末尾仍在交易
                exit_px = hd["last"]
            if np.isfinite(exit_px) and exit_px > 0:
                out.append(exit_px / entry - 1)
    return np.array(out)


# ══════════ 组合级 ══════════
def portfolio(rc, pick, seed, hold_through, haircut, keep=1.0):
    rng = np.random.default_rng(seed)
    pool = by_day
    if keep < 1.0:
        r2 = np.random.default_rng(seed + 777)
        pool = {d: [cd for cd in v if r2.random() < keep] for d, v in by_day.items()}
    cash, holds, equity = 1.0, {}, np.zeros(NT)
    mh, start = rc["max_hold"], 200
    for t in range(start, NT):
        for code in list(holds):
            hd = holds[code]; ci = hd["ci"]
            op_t, hi_t, lo_t, cl_t, ma_t = OPa[t, ci], HIa[t, ci], LOa[t, ci], CLa[t, ci], MAa[t, ci]
            ex = None
            if not np.isfinite(cl_t):
                if t > last_valid[ci]:
                    ex = hd["last"] * haircut          # 永久终止
                elif not hold_through:
                    ex = hd["last"]                    # 现行处理
                # else: 临时停牌 → 持有穿越,不判止损
            elif hd["pending"]:
                ex = op_t if np.isfinite(op_t) else cl_t
            else:
                hd["last"] = cl_t
                ex = step(rc, hd, op_t, hi_t, lo_t, cl_t, ma_t)
                if ex is None and t - hd["t_in"] >= mh:
                    ex = cl_t
            if ex is not None and np.isfinite(ex) and ex > 0:
                cash += hd["shares"] * ex * (1 - COST)
                del holds[code]
        cands = pool.get(t - 1, [])
        free = SLOTS - len(holds)
        if cands and free > 0 and mkt_ok[t]:
            cands = [cd for cd in cands if cd not in holds
                     and np.isfinite(OPa[t, col_of[cd]]) and OPa[t, col_of[cd]] > 0]
            if cands:
                if pick == "small":
                    cands.sort(key=lambda cd: MVa[t, col_of[cd]]
                               if np.isfinite(MVa[t, col_of[cd]]) else np.inf)
                else:
                    rng.shuffle(cands)
                for cd in cands[:free]:
                    alloc = cash / (SLOTS - len(holds)) if SLOTS > len(holds) else 0
                    if alloc <= 0:
                        break
                    px = OPa[t, col_of[cd]]
                    hd = new_pos(rc, px, t, col_of[cd])
                    hd["shares"] = alloc * (1 - COST) / px
                    holds[cd] = hd
                    cash -= alloc
        equity[t] = cash + sum(hd["shares"] * (CLa[t, hd["ci"]] if np.isfinite(CLa[t, hd["ci"]])
                                               else hd["last"]) for hd in holds.values())
    eq = pd.Series(equity[start:], index=idx[start:])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return ann, (r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan), \
        (eq / eq.cummax() - 1).min()


print(f"\n{'='*112}\n交易级:三种停牌/退市处理下的净期望/笔(仅 A/B/D 三条,回答用户的两个提议)\n{'='*112}")
print(f"{'处理方式':<34}{'A 基线':>12}{'B 纯10周线':>13}{'D 条件止盈':>13}")
for label, ht, hc in MODES:
    vals = []
    for k in ("A", "B", "D"):
        r = trade_level(RULES[k], ht, hc)
        vals.append(f"{r.mean()-COST_TRADE:+.2%}({len(r):,})")
    print(f"{label:<34}{vals[0]:>12}{vals[1]:>13}{vals[2]:>13}   ({time.time()-t0:.0f}s)")

if "--trade-only" in sys.argv:
    print(f"\n(--trade-only:跳过组合级)\n耗时 {time.time()-t0:.0f}s")
    raise SystemExit(0)

rows = []
for label, ht, hc in MODES:
    for pick, keep in (("random", 1.0), ("small", 0.9)):
        pname = "随机选" if pick == "random" else "小市值优先(90%重抽样)"
        print(f"\n{'='*112}\n组合级 {label} / {pname}:{N_SEED}次的分布\n{'='*112}")
        print(f"{'':<3}{'规则':<28}{'中位年化':>10}{'25%':>9}{'75%':>9}"
              f"{'>0比例':>8}{'中位Sharpe':>11}{'中位回撤':>10}")
        for key, rc in RULES.items():
            res = [portfolio(rc, pick, 20260810 + s, ht, hc, keep) for s in range(N_SEED)]
            a = np.array([x[0] for x in res]); sh = np.array([x[1] for x in res])
            dd = np.array([x[2] for x in res])
            rows.append({"处理": label, "选股": pname, "规则": key, "说明": rc["name"],
                         "中位年化": np.median(a), "q25": np.quantile(a, .25),
                         "q75": np.quantile(a, .75), "正收益比例": (a > 0).mean(),
                         "中位Sharpe": np.median(sh), "中位回撤": np.median(dd)})
            r = rows[-1]
            print(f"{key:<3}{rc['name']:<28}{r['中位年化']:>+10.2%}{r['q25']:>+9.2%}"
                  f"{r['q75']:>+9.2%}{r['正收益比例']:>8.0%}{r['中位Sharpe']:>+11.3f}"
                  f"{r['中位回撤']:>10.2%}   ({time.time()-t0:.0f}s)")

H = pd.DataFrame(rows)
H.to_csv(f"{SP}/breakout_exit_rules_halt.csv", index=False)

print(f"\n{'='*112}\n结论:D 相对 A 的优势在三种处理下是否稳定\n{'='*112}")
for pname in H["选股"].unique():
    print(f"\n[{pname}]")
    for label, _, _ in MODES:
        s = H[(H.处理 == label) & (H.选股 == pname)].set_index("规则")
        dA, dD, dB = s.loc["A"], s.loc["D"], s.loc["B"]
        print(f"  {label:<34} A {dA['中位年化']:+7.2%} | D {dD['中位年化']:+7.2%} "
              f"(Δ {dD['中位年化']-dA['中位年化']:+.2f}pp) | B {dB['中位年化']:+7.2%} "
              f"(Δ {dB['中位年化']-dA['中位年化']:+.2f}pp)")
print("\n对照:全市场等权基准 OOS 年化 +7.22% / Sharpe 0.423 / 回撤 -32.77%")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: breakout_exit_rules_halt.csv")
