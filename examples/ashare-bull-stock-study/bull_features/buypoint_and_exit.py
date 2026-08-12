"""买点升级(52周新高+量能确认) + 20日线止盈

═══ 起因 ═══
DeepSeek 的 W 章把「放量突破买点」列为全篇边际贡献最大的组件(**近5年 +17.1pp/年**)。
对照我自己的事件定义(`oneil_prelaunch_attribution.py`),我的买点弱两级:

               我              他们
  新高窗口     **60日**        **52周(250日)**
  量能确认     **无**          **≥1.5 × 20日均量**

第五十四节「基底形态组合级与随机无区别」「2015后十年白干」两个结论,
**都建立在这个弱买点上**。所以先升级买点,再谈别的。

═══ 第一部分:买点 2×2(只动一个维度,其余全部不变) ═══
新高窗口 {60, 250} × 量能确认 {无, ≥1.5×}
基底约束(前60日振幅<50%)、最小间隔 60 日、入场=次日开盘 —— 三者在四格里完全相同。
**锚点**:60日/无量能 这一格必须复现 **70,310 笔、+4.61%/笔、组合 +6.34%**。

═══ 第二部分:20日线止盈(用户提的,替代固定止盈) ═══
用户明确否决固定百分比止盈:「尤其是主升浪的时候」会卖飞。
改成**跌破20日线离场** —— 自适应、不封上限。
但「什么时候把控制权交给 MA20」是关键,所以测四个交接点:

  H1 全程        —— 一入场就归 MA20 管
  H2 有浮盈才启用 —— 亏损段仍由 -10% 固定止损管
  H3 涨25%后启用  —— 对应 DeepSeek 的 +25% 止盈,但改成移动的
  H4 涨100%后启用 —— 与第四十一节 arm100 同思路,只是把 MA50 换成 MA20

对照:第四十一节实测 **MA50 全程接管明显差于固定止损**。MA20 更快,
可能更差(更容易被震出)也可能更好(更快锁定主升浪利润)—— 这正是要测的。

═══ 事前判据(跑之前写死,不放宽) ═══
交易级净期望 ≥ **+6.0%/笔**(基线 +4.61%)**且** 组合级年化 ≥ **+7.22%**(等权基准)。
本脚本一共 14 个格子,是一次小规模搜索 ——
**最好的那个必须同时超过「同选中率随机」的 20 次分布**,否则只算噪音里的最高点。
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
BASE_MAX_RANGE, MIN_GAP = 0.50, 60      # 与原事件定义完全一致

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
MA20 = CL.rolling(20, min_periods=20).mean()
idx = OP.index
NT = len(idx)
OPa, HIa, LOa = OP.to_numpy(float), HI.to_numpy(float), LO.to_numpy(float)
CLa, MVa = CL.to_numpy(float), MV.to_numpy(float)
MA50a, MA20a = MA50.to_numpy(float), MA20.to_numpy(float)
col_of = {cd: i for i, cd in enumerate(OP.columns)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv, vo

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()

# ══════════ 事件生成:2×2 ══════════
# **锚点没过时查出来的三处不一致**(首版重建得到 83,482 笔,原文件 70,310):
#   1. 原脚本 `rolling(60, min_periods=60)`,我写成了 min_periods=30
#      → 上市不足 60 天就能触发突破,多出一批
#   2. 原脚本对 `fwd_gain` 为 NaN 的事件 `continue` —— 等于**丢掉最后 252 个
#      交易日的事件**(没有完整前瞻窗口就算不出标签)
#   3. 原脚本用原始 close,没有 `where(>0)`
# 前两条各自都会改变事件数,必须同时对齐才可比。
FWD_WIN = 252
_rmax60 = CL.rolling(60, min_periods=60).max()
_rmin60 = CL.rolling(60, min_periods=60).min()
base60 = ((_rmax60 - _rmin60) / _rmin60.replace(0, np.nan)).shift(1)
volr = (VO / VO.rolling(20, min_periods=10).mean()).to_numpy()
BASE_OK = (base60 < BASE_MAX_RANGE).to_numpy()


def make_events(win, vol_mult):
    """与原定义唯一的差别:新高窗口 win、量能门槛 vol_mult(None=不要求)。"""
    prev_max = CL.rolling(win, min_periods=win).max().shift(1).to_numpy()
    hit = (CLa > prev_max) & BASE_OK
    if vol_mult is not None:
        hit &= (volr >= vol_mult)
    last_ok = NT - 1 - FWD_WIN          # 与原脚本一致:无完整前瞻窗口的事件丢弃
    codes, dps = [], []
    for j, cd in enumerate(OP.columns):
        p = np.flatnonzero(hit[:, j])
        last = -10**9
        for q in p:
            if q - last < MIN_GAP or q == 0 or q > last_ok:
                continue
            last = q
            codes.append(cd); dps.append(int(q))
    return pd.DataFrame({"code": codes, "dp": dps})


# ══════════ 引擎(自 breakout_exit_rules.py 复制;新增 ma_len 与 ma_arm) ══════════
def step(rc, hd, t, op_t, hi_t, lo_t, cl_t, ma_t):
    stop_f, ma_mode, arm = rc["stop"], rc["ma_mode"], rc.get("ma_arm")
    taken = ma_mode == "arm" and hd["armed_ma"]
    if stop_f is not None and not taken:
        if np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
            px = op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
            return px, "固定止损"
    if np.isfinite(hi_t) and hi_t > hd["peak"]:
        hd["peak"] = hi_t
    # 交接:arm=None 全程;arm="profit" 收盘首次高于入场价;arm=0.25/1.0 涨幅达标
    # **首版这里是退化的**:arm=0.0 写成 peak >= entry*(1+0),而 peak 初始化就等于
    # entry,条件入场即成立 —— H2 与 H1 跑出完全相同的数,等于白测一格。
    # 改成用**收盘价**判断浮盈,才是「有浮盈才交给 MA20」的本意。
    if ma_mode == "arm" and not hd["armed_ma"]:
        if arm is None:
            hd["armed_ma"] = True
        elif arm == "profit":
            if np.isfinite(cl_t) and cl_t > hd["entry"]:
                hd["armed_ma"] = True
        elif hd["peak"] >= hd["entry"] * (1 + arm):
            hd["armed_ma"] = True
    if ma_mode == "arm" and hd["armed_ma"] and np.isfinite(ma_t) and np.isfinite(cl_t):
        if cl_t < ma_t:
            hd["pending"] = True
    return None, None


def new_pos(rc, entry):
    return {"entry": entry, "peak": entry, "t_in": 0, "last": entry,
            "stop_px": entry * (1 - rc["stop"]) if rc["stop"] is not None else -INF,
            "armed_ma": False, "pending": False}


def _ma(rc):
    return MA20a if rc.get("ma_len") == 20 else MA50a


def run_trade(rc, evs):
    MAref = _ma(rc)
    max_hold = rc["max_hold"]
    out = []
    for code, grp in evs.groupby("code", sort=False):
        ci = col_of[code]
        op, hi, lo, cl, ma = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci], MAref[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            hd = new_pos(rc, entry)
            hd["t_in"] = e
            end = min(e + max_hold, NT - 1)
            exit_px = None
            for t in range(e, end + 1):
                if hd["pending"]:
                    px = op[t] if np.isfinite(op[t]) else cl[t]
                    if np.isfinite(px):
                        exit_px = px
                        break
                    hd["pending"] = False
                if not np.isfinite(cl[t]):
                    continue
                hd["last"] = cl[t]
                px, _ = step(rc, hd, t, op[t], hi[t], lo[t], cl[t], ma[t])
                if px is not None:
                    exit_px = px
                    break
            if exit_px is None:
                exit_px = cl[end] if np.isfinite(cl[end]) else hd["last"]
            if np.isfinite(exit_px) and exit_px > 0:
                out.append(exit_px / entry - 1)
    return np.array(out)


def run_pf(rc, evs, seed=SEED):
    MAref = _ma(rc)
    by_day = {d: g["code"].tolist() for d, g in evs.groupby("dp")}
    rng2 = np.random.default_rng(seed)
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    n_tr, max_hold, start = 0, rc["max_hold"], 200
    for t in range(start, NT):
        for code in list(holds):
            hd = holds[code]
            ci = hd["ci"]
            op_t, hi_t, lo_t, cl_t = OPa[t, ci], HIa[t, ci], LOa[t, ci], CLa[t, ci]
            ma_t = MAref[t, ci]
            ex = None
            if hd["pending"]:
                ex = op_t if np.isfinite(op_t) else (cl_t if np.isfinite(cl_t) else hd["last"])
            elif not np.isfinite(cl_t):
                ex = hd["last"]
            else:
                hd["last"] = cl_t
                ex, _ = step(rc, hd, t, op_t, hi_t, lo_t, cl_t, ma_t)
                if ex is None and t - hd["t_in"] >= max_hold:
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
                hd = new_pos(rc, px)
                hd["t_in"] = t
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


RULE_A = dict(stop=0.10, ma_mode="none", max_hold=252)
EXITS = {
    "A 基线 -10%止损,无止盈": dict(stop=0.10, ma_mode="none", max_hold=252),
    "H1 MA20 全程接管": dict(stop=0.10, ma_mode="arm", ma_len=20, ma_arm=None, max_hold=252),
    "H2 MA20 收盘转正后启用": dict(stop=0.10, ma_mode="arm", ma_len=20, ma_arm="profit", max_hold=252),
    "H3 MA20 涨25%后启用": dict(stop=0.10, ma_mode="arm", ma_len=20, ma_arm=0.25, max_hold=252),
    "H4 MA20 涨100%后启用": dict(stop=0.10, ma_mode="arm", ma_len=20, ma_arm=1.00, max_hold=252),
}

# ══════════ 第一部分:买点 2×2 ══════════
print(f"\n{'='*118}\n第一部分 买点 2×2(规则 A 不变,只换事件)\n{'='*118}")
print(f"{'买点':<34}{'事件数':>9}{'胜率':>8}{'毛期望':>9}{'净期望':>9}"
      f"{'年化':>9}{'Sharpe':>8}{'最大回撤':>10}{'年均笔数':>9}")
EV, rows = {}, []
for win in (60, 250):
    for vm in (None, 1.5):
        nm = f"{'60日' if win==60 else '52周(250日)'}新高" + ("" if vm is None else f" + 量能≥{vm}×")
        e = make_events(win, vm)
        EV[nm] = e
        r = run_trade(RULE_A, e)
        pf = run_pf(RULE_A, e)
        net = r.mean() - COST_TRADE
        print(f"{nm:<34}{len(e):>9,}{(r>0).mean():>8.1%}{r.mean():>+9.2%}{net:>+9.2%}"
              f"{pf['年化']:>+9.2%}{pf['Sharpe']:>8.3f}{pf['最大回撤']:>10.1%}"
              f"{pf['年均笔数']:>9.0f}   ({time.time()-t0:.0f}s)")
        rows.append({"部分": "买点", "配置": nm, "事件数": len(e), "胜率": (r > 0).mean(),
                     "净期望": net, **pf})
        if win == 60 and vm is None:      # ── 锚点 ──
            print(f"    锚点核对:事件 {len(e):,}(应 70,310)、净期望 {net:+.2%}(应 +4.61%)、"
                  f"年化 {pf['年化']:+.2%}(应 +6.34%)")
            assert abs(len(e) - 70310) <= 50, f"事件数与原定义不符:{len(e)}"
            assert abs(net - 0.0461) < 0.0015, f"交易级锚点对不上:{net:+.4%}"
            assert abs(pf["年化"] - 0.0634) < 0.002, f"组合级锚点对不上:{pf['年化']:+.4%}"
            print("    锚点通过 —— 事件重建与原文件一致,后面的差异只来自买点定义")

best_bp = max([r for r in rows if r["部分"] == "买点"], key=lambda r: r["年化"])["配置"]
print(f"\n  组合级年化最高的买点:**{best_bp}**")

# ══════════ 第二部分:20日线止盈 ══════════
print(f"\n{'='*118}\n第二部分 20日线止盈(用户提议,替代固定止盈)\n{'='*118}")
for bp in ("60日新高", best_bp) if best_bp != "60日新高" else ("60日新高",):
    print(f"\n  【买点:{bp}】  事件 {len(EV[bp]):,}")
    print(f"  {'离场规则':<30}{'胜率':>8}{'毛期望':>9}{'净期望':>9}{'年化':>9}"
          f"{'Sharpe':>8}{'最大回撤':>10}{'年均笔数':>9}")
    for nm, rc in EXITS.items():
        r = run_trade(rc, EV[bp])
        pf = run_pf(rc, EV[bp])
        net = r.mean() - COST_TRADE
        print(f"  {nm:<30}{(r>0).mean():>8.1%}{r.mean():>+9.2%}{net:>+9.2%}"
              f"{pf['年化']:>+9.2%}{pf['Sharpe']:>8.3f}{pf['最大回撤']:>10.1%}"
              f"{pf['年均笔数']:>9.0f}   ({time.time()-t0:.0f}s)")
        rows.append({"部分": f"离场@{bp}", "配置": nm, "事件数": len(EV[bp]),
                     "胜率": (r > 0).mean(), "净期望": net, **pf})

# ══════════ 判定 + 随机对照 ══════════
R = pd.DataFrame(rows)
best = R.loc[R["年化"].idxmax()]
print(f"\n{'='*118}\n判定(事前判据,未放宽)\n{'='*118}")
print(f"  全场最好:**{best['部分']} / {best['配置']}**  "
      f"净期望 {best['净期望']:+.2%}  年化 {best['年化']:+.2%}")
c1, c2 = best["净期望"] >= 0.060, best["年化"] >= 0.0722
print(f"    ① 交易级净期望 ≥+6.0%  →  {best['净期望']:+.2%}  {'✓' if c1 else '✗'}")
print(f"    ② 组合级年化 ≥+7.22%   →  {best['年化']:+.2%}  {'✓' if c2 else '✗'}")

# 14 个格子是一次搜索:最好的那个要和「同选中率随机选事件」比
bp_best = best["配置"] if best["部分"] == "买点" else best["部分"].split("@")[1]
if bp_best in EV and bp_best != "60日新高":
    k = len(EV[bp_best])
    base_ev = EV["60日新高"]
    rng = np.random.default_rng(SEED)
    anns, nets = [], []
    for s in range(N_RAND):
        sub = base_ev.iloc[rng.choice(len(base_ev), min(k, len(base_ev)), replace=False)]
        nets.append(run_trade(RULE_A, sub).mean() - COST_TRADE)
        anns.append(run_pf(RULE_A, sub, seed=SEED + s)["年化"])
    nets, anns = np.array(nets), np.array(anns)
    rr = R[(R["部分"] == "买点") & (R["配置"] == bp_best)].iloc[0]
    p_net = float((nets >= rr["净期望"]).mean())
    p_ann = float((anns >= rr["年化"]).mean())
    print(f"\n  随机对照(从 60日新高事件里随机抽 {k:,} 笔 × {N_RAND} 次):")
    print(f"    净期望 实际 {rr['净期望']:+.2%}  随机中位 {np.median(nets):+.2%}"
          f"  [{nets.min():+.2%}, {nets.max():+.2%}]  **p={p_net:.3f}**")
    print(f"    年化   实际 {rr['年化']:+.2%}  随机中位 {np.median(anns):+.2%}"
          f"  [{anns.min():+.2%}, {anns.max():+.2%}]  **p={p_ann:.3f}**")
    print(f"    ③ 优于同数量随机(两项 p<0.05)→ "
          f"{'✓' if (p_net < 0.05 and p_ann < 0.05) else '✗'}")
    print(f"\n  **{'算发现' if (c1 and c2 and p_net<0.05 and p_ann<0.05) else '不算发现'}**")
else:
    print(f"\n  **{'算发现' if (c1 and c2) else '不算发现'}**(最好的就是基线买点,无需随机对照)")

# 分段:验证「2015后十年白干」是否因买点太弱
print(f"\n{'='*118}\n分段:升级买点能否救回 2015-05 之后的十年\n{'='*118}")
CUT = 575
for nm, e in EV.items():
    for tag, sub in (("2015-05前", e[e.dp < CUT]), ("2015-05后", e[e.dp >= CUT])):
        if len(sub) < 50:
            continue
        r = run_trade(RULE_A, sub)
        print(f"  {nm:<34}{tag:<12}{len(sub):>8,} 笔   净期望 {r.mean()-COST_TRADE:>+8.2%}")

R.to_csv(f"{SP}/buypoint_and_exit.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: buypoint_and_exit.csv")
