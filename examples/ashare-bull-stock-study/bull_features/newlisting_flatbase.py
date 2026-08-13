"""次新股(上市 1-3 年)+ 箱体突破:第六十六节主检验

本脚本的价格锚点沿用 base_pattern_trade.py 的 assert(+4.61% / +6.34%),
对不上会直接停住。2026-08-13 面板已完整恢复并逐位复现 §54 整张表。

═══ 假说与机制(用户提出,2026-08-13)═══
从**上市 1-3 年的次新股**里找**箱体(平底)突破**,成功率是否更高?
案例:嘉益股份(2021-09上市)、寒武纪(2020-07)、汤臣倍健(2010-12)、
匠心家居(2021-08)四例吻合;大华股份(2008-05)是反例。

**这是本研究第一个有「机制 + 已知日历」而不只是「形状」的假说:**
  1. 无套牢盘 —— 上市 1-3 年,上方每个持有人都是近期买入的
  2. 限售解禁时间表已知 —— 首发 1 年 / 控股股东 3 年,恰好覆盖该窗口。
     **箱体不是巧合,是被供给压力压出来的;突破发生在压力出清之后**
  3. 「上市日期」100% 事前可观测、零前视
  4. 与已测过的一切正交 —— 64 节里从未按上市年龄切过

═══ 事前锁定(不搜索、不换切法) ═══
  上市年龄窗口   [250, 750] 交易日(≈1~3年)。**不测其他窗口**
  箱体定义       `base_pattern_detector.FLAT`,一个参数不改
  事件源         `oneil_prelaunch_events_fixed.csv`(与第五十四节同一份)
  离场           RULE_A(-10%固定止损、无止盈、最长252日),引擎不改一行
  仓位/成本      SLOTS=10、单边 0.3%、pick="small"、择时 510300 MA200

═══ 陷阱(原本三个,现在剩两个) ═══
① **~~面板起点会伪造出一批假次新股~~ —— 这个陷阱已经消失。**
   原计划用「首个有效交易日」推上市日。那个代理必然把面板起点之前上市的
   老股票全部误判成起点当天上市的次新股;而第五十四节已证整套突破系统的收益
   几乎全部来自 2013-04~2015-05(那段 fwd_gain 均值 +117.09%,之后只有 +48.22%)。
   **两个错误叠加会产出一个极其漂亮的假阳性。**
   2026-08-13 恢复数据时发现源表自带 **`listed_days`(真实上市天数)**,
   直接用它,代理与守卫全部删除。**不再需要任何近似。**

② **必须做双层同日随机对照,只做一层几乎必然测到池子而不是形态。**
   这是回答「成功率是否更高」里「**比谁高**」的关键:
     第一层  同日 × **全市场**随机替换      → 次新+箱体 **合起来** vs 市场
     第二层  同日 × **同为次新**的股票随机   → **箱体本身**的贡献(池子已控住)
   **两者之差 = 分解出「次新」与「箱体」各自的贡献。**

③ **IPO 是成批停发的,次新池自带年份聚集。**
   暂停记录:2012-11~2014-01(停 14 个月)、2015-07、2023-08 收紧。
   2021-2022 注册制那波 IPO 对应的正是 2023-2025 的突破。
   → 必须报出 13-19 / 20-25 两段,同向才算数。

═══ 事前判据(写死,不放宽;这是第 15 次事前判据检验) ═══
  ① 样本量        事件数 ≥ 300
  ② 交易级净期望  ≥ +6.0%   (与第五十四节同门槛)
  ③ 组合级年化    ≥ +7.22%  (与第五十四节同门槛)
  ④ 双层随机对照  **两层都要 p < 0.05,且组合级也必须过**
  ⑤ 两段同向      13-19 与 20-25 方向一致
  **任一不过 = 不算发现,原样写入第六十六节,不改判据。**

═══ 事前预测(写下来以便被证伪) ═══
按 14 连败的历史,**最可能是第 ④ 条的第二层过不去** ——
即「次新」有贡献而「箱体」没有。**这个结果本身就是有价值的答案**:
它意味着正确做法是「在次新股池里广撒网」而不是「在次新股里找箱体」,
两者的执行成本差一个数量级。

═══ 锚点(对不上就停,不往下看结论) ═══
  面板 5,232 × 3,297、2013-01-04 ~ 2026-08-03
  全量事件 70,310 笔 / 交易级净期望 +4.61% / 组合年化 +6.34%
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
SLOTS, SEED, N_RAND = 10, 20260813, 300
INF = float("inf")
RULE_A = dict(name="基线:-10%固定止损,无止盈,252日",
              stop=0.10, ma_mode="none", trail=None, arm=None, max_hold=252)

# ── 事前锁定的两个新参数,不搜索 ──
AGE_LO, AGE_HI = 365, 1095     # 上市年龄窗口(**自然日**,listed_days 是自然日)= 1~3 年
SPLIT = "2020-01-01"           # 陷阱③:两段分界

t0 = time.time()
o, h, l, c, mv, vo, ld = {}, {}, {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "high", "low", "close", "float_mv",
                                    "volume", "listed_days"])
    if x.empty:
        continue
    o[k] = pd.to_numeric(x["open"], errors="coerce")
    h[k] = pd.to_numeric(x["high"], errors="coerce")
    l[k] = pd.to_numeric(x["low"], errors="coerce")
    c[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    vo[k] = pd.to_numeric(x["volume"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
OP = pd.DataFrame(o).sort_index(); OP.index = OP.index.tz_localize(None)
HI = pd.DataFrame(h).set_axis(OP.index); LO = pd.DataFrame(l).set_axis(OP.index)
CL = pd.DataFrame(c).set_axis(OP.index); MV = pd.DataFrame(mv).set_axis(OP.index)
VO = pd.DataFrame(vo).set_axis(OP.index)
LDF = pd.DataFrame(ld).set_axis(OP.index)
OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
MA50 = CL.rolling(50, min_periods=50).mean()
PMIN = CL.rolling(PRIOR, min_periods=60).min().shift(1)
idx = OP.index
pos = {d: i for i, d in enumerate(idx)}
NT = len(idx)
OPa, HIa, LOa = OP.to_numpy(float), HI.to_numpy(float), LO.to_numpy(float)
CLa, MVa, MAa = CL.to_numpy(float), MV.to_numpy(float), MA50.to_numpy(float)
VOa, PMINa = VO.to_numpy(float), PMIN.to_numpy(float)
F = {"listed_days": LDF}
col_of = {cd: i for i, cd in enumerate(OP.columns)}
print(f"面板 {OP.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv, vo, ld

# ══════════ 上市年龄(用面板自带的真实 listed_days) ══════════
# 原计划用「首个有效交易日」当上市日代理,并硬性排除面板起点附近的标的 ——
# 那个代理必然把面板起点之前上市的老股全部误判成 2013 年初的次新股,
# 而突破系统的收益几乎全部来自 2013-04~2015-05,两个错误会叠成假阳性。
# 数据恢复后发现源表自带 `listed_days`(真实上市天数),**陷阱① 直接消失**。
LD = F["listed_days"].to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
IS_NEW = (LD >= AGE_LO) & (LD <= AGE_HI) & ALIVE       # 次新 = 上市 1~3 年且在市
print(f"\n上市年龄用面板自带 listed_days(自然日),窗口 [{AGE_LO}, {AGE_HI}]")
print(f"  有 listed_days 的股票日占比: {np.isfinite(LD).mean():.2%}")
print(f"  次新股票日占比: {IS_NEW.sum() / max(ALIVE.sum(), 1):.2%}  ({time.time()-t0:.0f}s)")

# ══════════ 事件源(与第五十四节同一份) ══════════
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


# ══════════ 引擎(自 base_pattern_trade.py 原样复制,未改一行) ══════════
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
    n_trades = 0
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


# ══════════ 锚点:对不上就停 ══════════
r_all, _ = run_trade_level(RULE_A, ev)
net_all = r_all.mean() - COST_TRADE
pf_all = run_portfolio(RULE_A, ev)
print(f"\n锚点 全量 {len(ev):,} 笔:交易级净期望 **{net_all:+.2%}**(应 +4.61%)、"
      f"组合年化 **{pf_all['年化']:+.2%}**(应 +6.34%)  ({time.time()-t0:.0f}s)")
assert abs(net_all - 0.0461) < 0.0015, f"交易级锚点对不上:{net_all:+.4%}"
assert abs(pf_all["年化"] - 0.0634) < 0.002, f"组合级锚点对不上:{pf_all['年化']:+.4%}"
print("  锚点通过 —— 引擎与第五十四节完全一致,后面只换事件集合")

# ══════════ 逐事件:平底检测 + 次新判定 ══════════
E = ev[ev.dp >= NEED].reset_index(drop=True)
flat = np.zeros(len(E), bool)
new_ = np.zeros(len(E), bool)
row_of = {}
for i, (cd, dp) in enumerate(zip(E.code.to_numpy(), E.dp.to_numpy())):
    row_of.setdefault(cd, []).append((i, int(dp)))
for cd, lst in row_of.items():
    ci = col_of[cd]
    for i, dp in lst:
        new_[i] = bool(IS_NEW[dp, ci])
        s0 = dp - WIN
        b = detect_base(CLa[s0:dp, ci], HIa[s0:dp, ci], LOa[s0:dp, ci],
                        VOa[s0:dp, ci], PMINa[s0:dp, ci])
        flat[i] = bool(b["flat"])
main = new_ & flat
print(f"\n{len(E):,} 个事件(dp≥{NEED})的切分:  ({time.time()-t0:.0f}s)")
print(f"  次新(上市 {AGE_LO}~{AGE_HI} 自然日) {new_.sum():>7,} ({new_.mean():>6.2%})")
print(f"  平底(箱体)                    {flat.sum():>7,} ({flat.mean():>6.2%})")
print(f"  **次新 × 平底(主检验)**        {main.sum():>7,} ({main.mean():>6.2%})")

# 判据① 先判,不够就直接停 —— 样本不足时后面的数都不该看
if main.sum() < 300:
    print(f"\n**判据① 未通过:主检验事件仅 {main.sum():,} 笔 < 300。**")
    print("  样本不足,后续统计不可采信。按纪律:不算发现,不放宽门槛,不换窗口。")

# ══════════ 主表 ══════════
SUBS = {"【对照】dp≥NEED 全部": np.ones(len(E), bool),
        "仅次新": new_,
        "仅平底": flat,
        "**次新 × 平底**": main}
print(f"\n{'='*118}\n交易级 + 组合级(RULE_A,引擎未改)\n{'='*118}")
print(f"{'事件子集':<26}{'事件数':>9}{'选中率':>8}{'胜率':>8}{'毛期望':>9}"
      f"{'净期望':>9}{'年化':>9}{'Sharpe':>8}{'最大回撤':>10}{'年均笔数':>9}")
rows = []
for nm, m in SUBS.items():
    sub = E[m]
    if len(sub) < 50:
        print(f"{nm:<26}{len(sub):>9,}   样本不足,跳过")
        continue
    r, _ = run_trade_level(RULE_A, sub)
    pf = run_portfolio(RULE_A, sub)
    net = r.mean() - COST_TRADE
    print(f"{nm:<26}{len(sub):>9,}{m.mean():>8.1%}{(r>0).mean():>8.1%}"
          f"{r.mean():>+9.2%}{net:>+9.2%}{pf['年化']:>+9.2%}{pf['Sharpe']:>8.3f}"
          f"{pf['最大回撤']:>10.1%}{pf['年均笔数']:>9.0f}   ({time.time()-t0:.0f}s)", flush=True)
    rows.append({"子集": nm, "事件数": len(sub), "选中率": m.mean(), "胜率": (r > 0).mean(),
                 "毛期望": r.mean(), "净期望": net, **pf})
R = pd.DataFrame(rows)

# ══════════ 陷阱③:两段必须同向 ══════════
print(f"\n{'='*118}\n两段分开(分界 {SPLIT};IPO 成批停发会让次新池自带年份聚集)\n{'='*118}")
seg_ok = {}
sp = pos.get(pd.Timestamp(SPLIT), None)
if sp is None:
    sp = int(np.searchsorted(idx.values, np.datetime64(SPLIT)))
print(f"{'子集':<26}{'13-19 笔数':>12}{'13-19 净期望':>14}{'20-25 笔数':>12}{'20-25 净期望':>14}{'同向':>7}")
for nm, m in SUBS.items():
    sub = E[m]
    if len(sub) < 50:
        continue
    a, b = sub[sub.dp < sp], sub[sub.dp >= sp]
    na = (run_trade_level(RULE_A, a)[0].mean() - COST_TRADE) if len(a) >= 30 else np.nan
    nb = (run_trade_level(RULE_A, b)[0].mean() - COST_TRADE) if len(b) >= 30 else np.nan
    same = bool(np.isfinite(na) and np.isfinite(nb) and (na > 0) == (nb > 0))
    seg_ok[nm] = same
    print(f"{nm:<26}{len(a):>12,}{na:>+14.2%}{len(b):>12,}{nb:>+14.2%}"
          f"{('✓' if same else '✗'):>7}", flush=True)

# ══════════ 陷阱②:双层同日随机对照 ══════════
# 第一层  同日 × 全市场随机  → 次新+箱体 **合起来** vs 市场
# 第二层  同日 × 同为次新随机 → **箱体本身**的贡献(池子已控住)
# 两者之差 = 分解出「次新」与「箱体」各自的贡献。只做第一层不可采信。
print(f"\n{'='*118}\n{N_RAND} 次同日随机替换对照(换股票、留日期、留离场规则)\n{'='*118}")
sub_main = E[main]
if len(sub_main) < 50:
    print("  主检验样本不足,跳过随机对照。")
    p_tier = {"第一层 同日×全市场": np.nan, "第二层 同日×同为次新": np.nan}
else:
    obs_net = run_trade_level(RULE_A, sub_main)[0].mean() - COST_TRADE
    obs_ann = run_portfolio(RULE_A, sub_main)["年化"]
    days = sub_main["dp"].to_numpy()
    codes_of_col = np.array(OP.columns)
    rng = np.random.default_rng(SEED)
    p_tier = {}
    for tier, tier_mask in (("第一层 同日×全市场", ALIVE),
                            ("第二层 同日×同为次新", IS_NEW)):
        # 候选池摊平成一个大数组 + 偏移量,抽样变成一次向量化索引
        # (逐事件调 rng.choice 慢约一千倍 —— 第六十三节踩过)
        pool_flat, pool_off, pool_sz = [], {}, {}
        for t in np.unique(days):
            t = int(t)
            p_ = np.flatnonzero(tier_mask[t] & np.isfinite(OPa[t + 1]) & (OPa[t + 1] > 0))
            pool_off[t] = len(pool_flat)
            pool_sz[t] = len(p_)
            pool_flat.extend(p_.tolist())
        pool_flat = np.asarray(pool_flat, dtype=np.int32)
        off_e = np.array([pool_off[int(t)] for t in days], dtype=np.int64)
        sz_e = np.array([pool_sz[int(t)] for t in days], dtype=np.int64)
        ok_e = sz_e > 0
        print(f"\n  {tier}:候选池 {len(pool_off):,} 个日期 / {len(pool_flat):,} 个槽位"
              f"(空池日 {int((~ok_e).sum()):,})  ({time.time()-t0:.0f}s)", flush=True)
        nets, anns = np.empty(N_RAND), np.empty(N_RAND)
        for k in range(N_RAND):
            pick = off_e + (rng.random(len(days)) * np.maximum(sz_e, 1)).astype(np.int64)
            rj = pool_flat[np.where(ok_e, pick, off_e)]
            fake = pd.DataFrame({"code": codes_of_col[rj], "dp": days})
            nets[k] = run_trade_level(RULE_A, fake)[0].mean() - COST_TRADE
            anns[k] = run_portfolio(RULE_A, fake, seed=SEED + k)["年化"]
        p_net = float((nets >= obs_net).mean())
        p_ann = float((anns >= obs_ann).mean())
        p_tier[tier] = (p_net, p_ann)
        print(f"    交易级净期望  观测 **{obs_net:+.2%}**   随机中位 {np.median(nets):+.2%}"
              f"  区间 [{nets.min():+.2%}, {nets.max():+.2%}]   **p={p_net:.4f}**")
        print(f"    组合级年化    观测 **{obs_ann:+.2%}**   随机中位 {np.median(anns):+.2%}"
              f"  区间 [{anns.min():+.2%}, {anns.max():+.2%}]   **p={p_ann:.4f}**", flush=True)
        R = pd.concat([R, pd.DataFrame([{
            "子集": f"{tier}·随机中位", "事件数": len(sub_main),
            "净期望": float(np.median(nets)), "年化": float(np.median(anns)),
            "p_净期望": p_net, "p_年化": p_ann}])], ignore_index=True)

# ══════════ 判定 ══════════
print(f"\n{'='*118}\n事前判据 vs 实际(判据在跑之前写死,未放宽)\n{'='*118}")
row = R[R["子集"] == "**次新 × 平底**"]
if row.empty:
    print("  主检验样本不足,五条判据全部无法判定 → **不算发现**")
else:
    n_ev = int(row.事件数.iloc[0])
    net = float(row.净期望.iloc[0])
    ann = float(row.年化.iloc[0])
    t1 = p_tier.get("第一层 同日×全市场", (np.nan, np.nan))
    t2 = p_tier.get("第二层 同日×同为次新", (np.nan, np.nan))
    c1 = n_ev >= 300
    c2 = net >= 0.060
    c3 = ann >= 0.0722
    c4 = all(np.isfinite(x) and x < 0.05 for x in (*t1, *t2))
    c5 = bool(seg_ok.get("**次新 × 平底**", False))
    print(f"  ① 样本量 ≥ 300              {n_ev:,}            {'✓' if c1 else '✗'}")
    print(f"  ② 交易级净期望 ≥ +6.0%      {net:+.2%}          {'✓' if c2 else '✗'}")
    print(f"  ③ 组合级年化 ≥ +7.22%       {ann:+.2%}          {'✓' if c3 else '✗'}")
    print(f"  ④ 双层随机对照 四个 p<0.05  第一层 {t1[0]:.4f}/{t1[1]:.4f}  "
          f"第二层 {t2[0]:.4f}/{t2[1]:.4f}   {'✓' if c4 else '✗'}")
    print(f"  ⑤ 两段同向                  {'同向' if c5 else '反向'}          {'✓' if c5 else '✗'}")
    ok = c1 and c2 and c3 and c4 and c5
    print(f"\n  **结论:{'算发现' if ok else '不算发现'}**"
          f"{'' if ok else ' —— 事前锁定全部参数,不回头搜索、不放宽门槛'}")
    if np.isfinite(t1[1]) and np.isfinite(t2[1]):
        print("\n  贡献分解(第 ② 个陷阱的意义所在):")
        print(f"    第一层 p_年化 {t1[1]:.4f} —— 次新+箱体 **合起来** 相对市场")
        print(f"    第二层 p_年化 {t2[1]:.4f} —— **箱体本身**(次新池已控住)")
        if t1[1] < 0.05 <= t2[1]:
            print("    → 第一层过、第二层不过:**是「次新」在起作用,不是「箱体」。**")
            print("      执行含义:在次新股池里广撒网,不要在次新股里找箱体。")

R.to_csv(f"{SP}/newlisting_flatbase.csv", index=False)
print(f"\n→ {SP}/newlisting_flatbase.csv   ({time.time()-t0:.0f}s)")
