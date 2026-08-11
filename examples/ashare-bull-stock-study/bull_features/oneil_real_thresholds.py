"""欧奈尔的真实入场门槛 —— 我们此前把它们全漏了

═══ 起因 ═══
用户问:"为什么那么多投资者用陶博士/欧奈尔方法就成功了,我们跑出来这么差?"
检查后发现:**我们的入场规则是纯价格的**(60日新高 + 前60日振幅<50%),
而面板里 volume / eps / net_income / roe 全都有,**一个都没用在入场上**。

而欧奈尔方法的核心恰恰包含:
  - 突破必须**放量**(量能高于均量 40-50%),缩量突破是假突破
  - **C**:当季 EPS 同比 ≥ 25%
  - **ROE ≥ 17%**(机构级盈利能力)
我们此前只用过"双增长 > 0",这是低得多的门槛。

═══ 设计:只加入场过滤,不改离场 ═══
离场沿用四十二节 rule A:-10% 固定止损、无止盈、252日上限、突破次日开盘入场。
阈值**全部取自欧奈尔原著,不做网格搜索**——它们不是拟合出来的。

  基线 / +放量 / +C / +ROE / **三个全加** = 5 组

═══ 必须一起报的副作用 ═══
三个过滤叠加后**每年还剩多少入场机会**。若从每年几千笔掉到几十笔,
那么**样本量本身**就解释了为什么个人体验方差巨大
(与 oneil_sampling_variance.py 互相印证)。

═══ 事前写死的判据 ═══
三个全加后需**同时**满足:
  交易级净期望 ≥ +6.0%/笔(基线 +4.61% 的 1.3 倍)
  组合级(随机选中位)年化 ≥ +7.22%(等权基准)
→ 才算"我们确实漏掉了关键成分"。
只满足前者 → "提高了单笔质量但机会太少,组合层吃不下"。
都不满足 → 欧奈尔的入场门槛**不能**解释差距,如实记录。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, N_SEED, SEED = 10, 20, 20260811
STOP, MAX_HOLD = 0.10, 252

# 阈值全部取自欧奈尔原著,不调参
VOL_MULT = 1.5          # 突破日量 ≥ 前50日均量 × 1.5(原著"高于均量40-50%")
C_THRESH = 0.25         # 当季/年度 净利润同比 ≥ 25%
ROE_THRESH = 0.17       # ROE ≥ 17%

t0 = time.time()
COLS = ["open", "high", "low", "close", "volume", "float_mv", "ni_yoy_252", "roe"]
d = {c: {} for c in COLS}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    try:
        x = pd.read_parquet(f, columns=COLS)
    except Exception:
        continue
    if x.empty:
        continue
    for c in COLS:
        d[c][k] = pd.to_numeric(x[c], errors="coerce")
OP = pd.DataFrame(d["open"]).sort_index(); OP.index = OP.index.tz_localize(None)


def _align(key):
    """索引统一成 tz-naive 再对齐(四十六节踩过:reindex_like 会全 NaN 且不报错)。"""
    f = pd.DataFrame(d[key]).sort_index()
    f.index = f.index.tz_localize(None)
    return f.reindex(index=OP.index, columns=OP.columns)


HI, LO, CL = _align("high"), _align("low"), _align("close")
VOL, MV = _align("volume"), _align("float_mv")
ROE = _align("roe")
# C 改用**当季**净利同比(欧奈尔真正的口径),原 ni_yoy_252 跨报告期已证实污染
NIY = pd.read_parquet(f"{SP}/clean_growth_c_qyoy.parquet").reindex(
    index=OP.index, columns=OP.columns)
assert NIY.notna().mean().mean() > 0.01, "clean_growth C 字段几乎全空"
OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
idx = OP.index; NT = len(idx)
pos = {dt: i for i, dt in enumerate(idx)}
OPa, HIa, LOa, CLa = OP.to_numpy(), HI.to_numpy(), LO.to_numpy(), CL.to_numpy()
MVa, VOLa, NIYa, ROEa = MV.to_numpy(), VOL.to_numpy(), NIY.to_numpy(), ROE.to_numpy()
col_of = {cd: i for i, cd in enumerate(OP.columns)}
for _n, _f in (("volume", VOL), ("ni_yoy_252", NIY), ("roe", ROE)):
    _r = _f.notna().mean().mean()
    assert _r > 0.01, f"{_n} 几乎全为 NaN(非空率 {_r:.4%})"
    print(f"  {_n:<12} 非空率 {_r:>6.1%}")
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del d

VOL50 = VOL.rolling(50, min_periods=30).mean().to_numpy()

ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv",
                 usecols=["code", "D"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
ev = ev[ev.code.isin(OP.columns)].copy()
ev["dp"] = ev["D"].map(pos); ev = ev.dropna(subset=["dp"]); ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp + 1 < NT - 5]
print(f"突破事件 {len(ev):,}")

# ---------------- 三个过滤在突破日的取值 ----------------
dps = ev["dp"].to_numpy()
cis = np.array([col_of[c] for c in ev["code"]])
volr = VOLa[dps, cis] / np.where(VOL50[dps, cis] > 0, VOL50[dps, cis], np.nan)
niy = NIYa[dps, cis]
roe = ROEa[dps, cis]
ev["pass_vol"] = volr >= VOL_MULT
ev["pass_c"] = niy >= C_THRESH
ev["pass_roe"] = roe >= ROE_THRESH
ev["year"] = idx[dps].year

print(f"\n{'='*112}\n三个过滤的触发率(阈值取自欧奈尔原著,不调参)\n{'='*112}")
for nm, col, thr, raw in (("放量突破", "pass_vol", f"量 ≥ 均量×{VOL_MULT}", volr),
                          ("C **当季**净利同比", "pass_c", f"≥ {C_THRESH:.0%}", niy),
                          ("ROE", "pass_roe", f"≥ {ROE_THRESH:.0%}", roe)):
    ok = np.isfinite(raw)
    print(f"  {nm:<12} {thr:<18} 可计算 {ok.mean():>6.1%}   "
          f"**通过率 {ev[col].mean():>6.1%}**   中位值 {np.nanmedian(raw):>8.3f}")
allp = ev.pass_vol & ev.pass_c & ev.pass_roe
print(f"  {'三个全通过':<12} {'':<18} {'':>13}   **通过率 {allp.mean():>6.1%}**"
      f"   剩 {int(allp.sum()):,} 笔")
print(f"\n  每年入场机会:基线 {len(ev)/13:.0f} 笔/年  →  三个全加 **{allp.sum()/13:.0f} 笔/年**")

CONFIGS = {
    "基线(纯价格突破)": None,
    "+放量突破": ["pass_vol"],
    "+C 净利同比≥25%": ["pass_c"],
    "+ROE≥17%": ["pass_roe"],
    "**三个全加**": ["pass_vol", "pass_c", "pass_roe"],
}


def new_pos(px, t, ci):
    return {"entry": px, "t_in": t, "last": px, "ci": ci, "stop_px": px * (1 - STOP)}


def step(hd, op_t, lo_t):
    if np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
        return op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
    return None


def trade_level(sub):
    rets, days = [], []
    for code, grp in sub.groupby("code", sort=False):
        ci = col_of[code]
        op, lo, cl = OPa[:, ci], LOa[:, ci], CLa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            hd = new_pos(entry, e, ci)
            end = min(e + MAX_HOLD, NT - 1)
            exit_px, texit = None, None
            for t in range(e, end + 1):
                if not np.isfinite(cl[t]):
                    continue
                hd["last"] = cl[t]
                px = step(hd, op[t], lo[t])
                if px is not None:
                    exit_px, texit = px, t
                    break
            if exit_px is None:
                texit = end
                exit_px = cl[end] if np.isfinite(cl[end]) else hd["last"]
            if np.isfinite(exit_px) and exit_px > 0:
                rets.append(exit_px / entry - 1)
                days.append(texit - e + 1)
    return np.array(rets), np.array(days)


_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()


def run_portfolio(by_day, pick, seed):
    rng = np.random.default_rng(seed)
    cash, holds, equity = 1.0, {}, np.zeros(NT)
    start = 200
    for t in range(start, NT):
        for code in list(holds):
            hd = holds[code]; ci = hd["ci"]
            op_t, lo_t, cl_t = OPa[t, ci], LOa[t, ci], CLa[t, ci]
            ex = None
            if not np.isfinite(cl_t):
                ex = hd["last"]
            else:
                hd["last"] = cl_t
                ex = step(hd, op_t, lo_t)
                if ex is None and t - hd["t_in"] >= MAX_HOLD:
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
                    hd = new_pos(px, t, col_of[cd])
                    hd["shares"] = alloc * (1 - COST_PF) / px
                    holds[cd] = hd
                    cash -= alloc
        equity[t] = cash + sum(hd["shares"] * (CLa[t, hd["ci"]] if np.isfinite(CLa[t, hd["ci"]])
                                               else hd["last"]) for hd in holds.values())
    eq = pd.Series(equity[start:], index=idx[start:])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    return ann, (eq / eq.cummax() - 1).min()


print(f"\n{'='*112}\n交易级(离场沿用四十二节 rule A:-10%止损、252日;只改入场)\n{'='*112}")
print(f"{'配置':<22}{'笔数':>9}{'每年':>7}{'胜率':>8}{'均盈':>9}{'盈亏比':>7}"
      f"{'净期望/笔':>11}{'中位天数':>9}{'>100%笔数':>10}")
res, subs = {}, {}
for name, flags in CONFIGS.items():
    sub = ev if flags is None else ev[np.logical_and.reduce([ev[f] for f in flags])]
    subs[name] = sub
    r, dsy = trade_level(sub)
    if len(r) < 50:
        print(f"{name:<22}{len(r):>9,}  样本过少")
        res[name] = {"net": np.nan, "n": len(r)}
        continue
    win = r > 0
    aw = r[win].mean() if win.any() else 0.0
    al = r[~win].mean() if (~win).any() else 0.0
    net = r.mean() - COST_TRADE
    res[name] = {"net": net, "n": len(r), "rets": r}
    print(f"{name:<22}{len(r):>9,}{len(r)/13:>7.0f}{win.mean():>8.1%}{aw:>+9.1%}"
          f"{abs(aw/al) if al else np.nan:>7.2f}{net:>+11.2%}{np.median(dsy):>9.0f}"
          f"{int((r>1).sum()):>10,}   ({time.time()-t0:.0f}s)")

print(f"\n{'#'*112}\n验证1 锚点:基线须复现四十二节 rule A 的 +4.61%/笔\n{'#'*112}")
b = res["基线(纯价格突破)"]["net"]
print(f"  基线 = {b:+.2%}   {'✓' if abs(b-0.0461) < 0.005 else '✗ 口径漂移,先查再继续'}")

print(f"\n{'='*112}\n组合级(10只、510300择时、单边0.3%;随机选 {N_SEED} 种子)\n{'='*112}")
print(f"{'配置':<22}{'小市值优先':>12}{'回撤':>10}{'随机选中位':>12}{'25~75%':>18}")
pf = {}
for name, sub in subs.items():
    if len(sub) < 50:
        continue
    by_day = {dd: g["code"].tolist() for dd, g in sub.groupby("dp")}
    a_s, dd_s = run_portfolio(by_day, "small", SEED)
    rs = [run_portfolio(by_day, "random", SEED + k)[0] for k in range(N_SEED)]
    rs = np.array(rs)
    pf[name] = {"small": a_s, "rand_med": np.median(rs)}
    print(f"{name:<22}{a_s:>+12.2%}{dd_s:>10.2%}{np.median(rs):>+12.2%}"
          f"{f'{np.quantile(rs,.25):+.2%}~{np.quantile(rs,.75):+.2%}':>18}"
          f"   ({time.time()-t0:.0f}s)")

print(f"\n{'='*112}\n判据判定(事前写死)\n{'='*112}")
k = "**三个全加**"
t_ok = np.isfinite(res[k]["net"]) and res[k]["net"] >= 0.06
p_ok = k in pf and pf[k]["rand_med"] >= 0.0722
print(f"  交易级净期望 {res[k]['net']:+.2%}  (需 ≥ +6.00%)  {'✓' if t_ok else '✗'}")
if k in pf:
    print(f"  组合级随机选中位 {pf[k]['rand_med']:+.2%}  (需 ≥ +7.22% 等权基准)  "
          f"{'✓' if p_ok else '✗'}")
if t_ok and p_ok:
    v = "**我们此前确实漏掉了关键成分 —— 欧奈尔的入场门槛能解释差距**"
elif t_ok:
    v = "**提高了单笔质量,但机会太少,组合层吃不下**"
else:
    v = "**欧奈尔的入场门槛不能解释差距**"
print(f"\n  {v}")
print(f"\n  副作用:三个全加后每年只剩 **{subs[k].shape[0]/13:.0f} 笔**入场机会"
      f"(基线 {len(ev)/13:.0f} 笔/年)")

pd.DataFrame({n: {"净期望": v["net"], "笔数": v["n"]} for n, v in res.items()}).T.to_csv(
    f"{SP}/oneil_thresholds_trade.csv")
pd.DataFrame(pf).T.to_csv(f"{SP}/oneil_thresholds_portfolio.csv")
np.save(f"{SP}/oneil_baseline_trade_rets.npy", res["基线(纯价格突破)"]["rets"])
ev[["code", "dp", "year", "pass_vol", "pass_c", "pass_roe"]].to_csv(
    f"{SP}/oneil_event_flags.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: oneil_thresholds_*.csv, "
      f"oneil_baseline_trade_rets.npy, oneil_event_flags.csv")
