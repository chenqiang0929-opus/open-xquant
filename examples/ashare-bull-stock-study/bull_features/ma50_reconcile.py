"""MA50 破位卖出的符号之争 —— 与 DeepSeek W 章对账

═══ 分歧 ═══
DeepSeek W 章:MA50 破位卖出是**体系里最大的正贡献**
  全区间 +8.4pp/年、近5年 +7.7pp(full -6.29% vs no_ma50 -14.71%)
本session:
  阶段1  10%止损 净期望 **+4.00%/笔** → 叠加 MA50 后 **+1.36%/笔**
  四十二节 纯10周线 组合级 -8.14%(小市值)/-1.17%(随机),回撤 -87~-89%

**同一个比较(在 -10% 止损之上加不加 MA50),符号相反。**

═══ 假设:差别在"系统里有没有别的离场" ═══
- 他们的持有规则是「长持有,不强制月度轮动」——去掉 MA50 后几乎无时间约束,
  反转的股票可以一直拿着(no_ma50 中位持仓 25.1 bar vs full 15.0 bar,
  年化却从 -6.29% 掉到 -14.71%)
- 我们的系统有 `MAX_HOLD = 252` —— **时间上限本身已经提供了离场**,
  再叠加 MA50 只剩「砍掉大赢家」的副作用
  (盈利>100% 的笔数 2,771 → 748)

若成立,真正的结论是:**系统必须有某种离场;MA50 与时间上限是互相替代的
两种方案,叠加反而有害。** 两份研究就都对了,只是基线不同。

═══ 语义差别(必须显式区分,否则两次实验根本不可比) ═══
  「叠加」additive:固定止损与 MA50 破位**同时生效**,任一触发即离场
                    ← DeepSeek 的口径,也是本脚本的口径
  「接管」takeover:股价站上 MA50 后,MA50 **取代**固定止损
                    ← 四十二节 B/C 的口径

═══ 2×2 消融(全部在 -10% 固定止损之上) ═══
  A′  252日上限,不叠加MA50   ← 锚点,须复现四十二节 rule A 的 +4.61%/笔
  I   252日上限,叠加MA50     ← 锚点,须接近阶段1 的 +1.36%/笔
  G   无时间上限,不叠加MA50
  H   无时间上限,叠加MA50    ← ≈ DeepSeek 的 full 口径

MA50 边际贡献 = (叠加 − 不叠加),在两个时间上限下分别算。

═══ 事前写死的判据 ═══
  边际贡献在「无上限」下为正、「252日」下为负 → 假设成立,分歧解开
  两个上限下都为负 → 他们的 +8.4pp 来自别处,**如实记为未解决的分歧**
  两个都为正 → 四十二节的「10周线证伪」结论需要修正
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE = 0.003          # 交易级:双边
COST_PF = 0.003             # 组合级:单边
SLOTS = 10
N_SEED = 20
SEED = 20260811
INF = float("inf")

NO_CAP = 10_000             # "无时间上限"用一个大于面板长度的数
RULES = {
    "A′ 252日上限 / 不叠加MA50": dict(stop=0.10, ma=False, max_hold=252),
    "I  252日上限 / **叠加**MA50": dict(stop=0.10, ma=True, max_hold=252),
    "G  无时间上限 / 不叠加MA50": dict(stop=0.10, ma=False, max_hold=NO_CAP),
    "H  无时间上限 / **叠加**MA50": dict(stop=0.10, ma=True, max_hold=NO_CAP),
}

t0 = time.time()
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
OP = pd.DataFrame(o).sort_index(); OP.index = OP.index.tz_localize(None)
HI = pd.DataFrame(h).set_axis(OP.index); LO = pd.DataFrame(l).set_axis(OP.index)
CL = pd.DataFrame(c).set_axis(OP.index); MV = pd.DataFrame(mv).set_axis(OP.index)
OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
MA50 = CL.rolling(50, min_periods=50).mean()
idx = OP.index; NT = len(idx)
pos = {d: i for i, d in enumerate(idx)}
OPa, HIa, LOa = OP.to_numpy(), HI.to_numpy(), LO.to_numpy()
CLa, MVa, MAa = CL.to_numpy(), MV.to_numpy(), MA50.to_numpy()
col_of = {cd: i for i, cd in enumerate(OP.columns)}
del o, h, l, c, mv
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv",
                 usecols=["code", "D"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
ev = ev[ev.code.isin(OP.columns)].copy()
ev["dp"] = ev["D"].map(pos); ev = ev.dropna(subset=["dp"]); ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp + 1 < NT - 5]
by_day = {d: g["code"].tolist() for d, g in ev.groupby("dp")}
print(f"突破事件 {len(ev):,}")

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()

print("\nMA50 语义(核对用):**叠加** —— 固定止损与 MA50 破位同时生效,任一触发即离场")
print("                    (四十二节 B/C 用的是『接管』:站上MA50后取代固定止损)")


def new_pos(rc, px, t, ci):
    return {"entry": px, "t_in": t, "last": px, "ci": ci, "pending": False,
            "stop_px": px * (1 - rc["stop"])}


def step(rc, hd, op_t, hi_t, lo_t, cl_t, ma_t):
    """推进一天。固定止损与 MA50 **同时**生效(叠加语义)。"""
    if np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
        return op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
    if rc["ma"] and np.isfinite(ma_t) and np.isfinite(cl_t) and cl_t < ma_t:
        hd["pending"] = True          # 收盘跌破 → 次日开盘离场
    return None


def trade_level(rc):
    mh = rc["max_hold"]
    rets, days, why = [], [], []
    for code, grp in ev.groupby("code", sort=False):
        ci = col_of[code]
        op, hi, lo, cl, ma = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci], MAa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            hd = new_pos(rc, entry, e, ci)
            end = min(e + mh, NT - 1)
            exit_px, reason, texit = None, "到期", None
            for t in range(e, end + 1):
                if hd["pending"]:
                    px = op[t] if np.isfinite(op[t]) else cl[t]
                    if np.isfinite(px):
                        exit_px, reason, texit = px, "跌破MA50", t
                        break
                    hd["pending"] = False
                if not np.isfinite(cl[t]):
                    continue
                hd["last"] = cl[t]
                px = step(rc, hd, op[t], hi[t], lo[t], cl[t], ma[t])
                if px is not None:
                    exit_px, reason, texit = px, "固定止损", t
                    break
            if exit_px is None:
                texit = end
                exit_px = cl[end] if np.isfinite(cl[end]) else hd["last"]
            if np.isfinite(exit_px) and exit_px > 0:
                rets.append(exit_px / entry - 1)
                days.append(texit - e + 1)
                why.append(reason)
    return np.array(rets), np.array(days), np.array(why)


def run_portfolio(rc, pick, seed):
    rng = np.random.default_rng(seed)
    cash, holds, equity = 1.0, {}, np.zeros(NT)
    mh, start = rc["max_hold"], 200
    for t in range(start, NT):
        for code in list(holds):
            hd = holds[code]; ci = hd["ci"]
            op_t, hi_t, lo_t = OPa[t, ci], HIa[t, ci], LOa[t, ci]
            cl_t, ma_t = CLa[t, ci], MAa[t, ci]
            ex = None
            if hd["pending"]:
                ex = op_t if np.isfinite(op_t) else (cl_t if np.isfinite(cl_t) else hd["last"])
            elif not np.isfinite(cl_t):
                ex = hd["last"]
            else:
                hd["last"] = cl_t
                ex = step(rc, hd, op_t, hi_t, lo_t, cl_t, ma_t)
                if ex is None and t - hd["t_in"] >= mh:
                    ex = cl_t
            if ex is not None and np.isfinite(ex) and ex > 0:
                cash += hd["shares"] * ex * (1 - COST_PF)
                del holds[code]
        cands = by_day.get(t - 1, [])
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
                    hd["shares"] = alloc * (1 - COST_PF) / px
                    holds[cd] = hd
                    cash -= alloc
        equity[t] = cash + sum(hd["shares"] * (CLa[t, hd["ci"]] if np.isfinite(CLa[t, hd["ci"]])
                                               else hd["last"]) for hd in holds.values())
    eq = pd.Series(equity[start:], index=idx[start:])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return ann, (eq / eq.cummax() - 1).min()


# ══════════════ 交易级 ══════════════
print(f"\n{'='*118}\n交易级(70,124 笔突破事件,0.3% 双边成本)\n{'='*118}")
print(f"{'变体':<30}{'笔数':>8}{'胜率':>8}{'均盈':>9}{'均亏':>8}{'盈亏比':>7}"
      f"{'净期望/笔':>11}{'中位天数':>9}{'>100%笔数':>10}")
res = {}
for name, rc in RULES.items():
    r, dsy, why = trade_level(rc)
    win = r > 0
    aw = r[win].mean() if win.any() else 0.0
    al = r[~win].mean() if (~win).any() else 0.0
    net = r.mean() - COST_TRADE
    res[name] = {"net": net, "n": len(r), "days": np.median(dsy),
                 "big": int((r > 1).sum()), "why": pd.Series(why).value_counts(normalize=True)}
    print(f"{name:<30}{len(r):>8,}{win.mean():>8.1%}{aw:>+9.1%}{al:>+8.1%}"
          f"{abs(aw/al) if al else np.nan:>7.2f}{net:>+11.2%}{np.median(dsy):>9.0f}"
          f"{int((r>1).sum()):>10,}   ({time.time()-t0:.0f}s)")

print(f"\n{'#'*118}\n验证1 锚点\n{'#'*118}")
a1 = res["A′ 252日上限 / 不叠加MA50"]["net"]
i1 = res["I  252日上限 / **叠加**MA50"]["net"]
print(f"  A′ = {a1:+.2%}   (四十二节 rule A 记录值 +4.61%)   "
      f"{'✓' if abs(a1-0.0461) < 0.005 else '✗ 口径漂移'}")
print(f"  I  = {i1:+.2%}   (阶段1『10%止损+MA50』记录值 +1.36%)   "
      f"{'✓ 接近' if abs(i1-0.0136) < 0.01 else '⚠ 差距较大,注意阶段1未做跳空/停牌修正'}")

print(f"\n{'='*118}\n离场原因分布\n{'='*118}")
for name in RULES:
    print(f"  {name:<30} " + "  ".join(f"{k} {v:.1%}" for k, v in res[name]["why"].items()))

print(f"\n{'='*118}\n**MA50 的边际贡献(交易级净期望/笔)**\n{'='*118}")
g_252 = i1 - a1
g_nocap = res["H  无时间上限 / **叠加**MA50"]["net"] - res["G  无时间上限 / 不叠加MA50"]["net"]
print(f"  252日上限下:{i1:+.2%} − {a1:+.2%} = **{g_252*100:+.2f}pp**")
print(f"  无时间上限下:{res['H  无时间上限 / **叠加**MA50']['net']:+.2%} − "
      f"{res['G  无时间上限 / 不叠加MA50']['net']:+.2%} = **{g_nocap*100:+.2f}pp**")

# ══════════════ 组合级 ══════════════
print(f"\n{'='*118}")
print(f"组合级({SLOTS}只、510300择时、单边{COST_PF:.1%};随机选跑 {N_SEED} 个种子)")
print(f"{'='*118}")
print(f"{'变体':<30}{'小市值优先':>12}{'回撤':>10}{'随机选中位':>12}{'25~75%':>18}{'中位回撤':>10}")
pf = {}
for name, rc in RULES.items():
    a_s, dd_s = run_portfolio(rc, "small", SEED)
    rs = [run_portfolio(rc, "random", SEED + k) for k in range(N_SEED)]
    ann = np.array([x[0] for x in rs]); dds = np.array([x[1] for x in rs])
    pf[name] = {"small": a_s, "rand_med": np.median(ann)}
    print(f"{name:<30}{a_s:>+12.2%}{dd_s:>10.2%}{np.median(ann):>+12.2%}"
          f"{f'{np.quantile(ann,.25):+.2%}~{np.quantile(ann,.75):+.2%}':>18}"
          f"{np.median(dds):>10.2%}   ({time.time()-t0:.0f}s)")

print(f"\n{'='*118}\n**MA50 的边际贡献(组合级年化)**\n{'='*118}")
for lbl, k0, k1 in (("252日上限", "A′ 252日上限 / 不叠加MA50", "I  252日上限 / **叠加**MA50"),
                    ("无时间上限", "G  无时间上限 / 不叠加MA50", "H  无时间上限 / **叠加**MA50")):
    print(f"  {lbl}:小市值 {(pf[k1]['small']-pf[k0]['small'])*100:+.1f}pp   "
          f"随机选中位 {(pf[k1]['rand_med']-pf[k0]['rand_med'])*100:+.1f}pp")

print(f"\n{'='*118}\n判据判定(事前写死)\n{'='*118}")
print(f"  交易级 MA50 边际贡献:252日 {g_252*100:+.2f}pp   无上限 {g_nocap*100:+.2f}pp")
if g_nocap > 0 and g_252 < 0:
    v = "**假设成立 → 分歧解开:离场机制不可叠加,MA50 与时间上限互为替代**"
elif g_nocap < 0 and g_252 < 0:
    v = "**两个上限下都为负 → DeepSeek 的 +8.4pp 来自别处,本检验无法解释,记为未解决的分歧**"
elif g_nocap > 0 and g_252 > 0:
    v = "**两个都为正 → 四十二节『10周线证伪』的结论需要修正**"
else:
    v = "**符号组合与三种预设都不符,须单独分析**"
print(f"\n  {v}")

pd.DataFrame({k: {"净期望": v["net"], "笔数": v["n"], "中位天数": v["days"],
                  ">100%笔数": v["big"]} for k, v in res.items()}).T.to_csv(
    f"{SP}/ma50_reconcile_trade.csv")
pd.DataFrame(pf).T.to_csv(f"{SP}/ma50_reconcile_portfolio.csv")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: ma50_reconcile_trade.csv / _portfolio.csv")
