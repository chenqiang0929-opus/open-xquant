"""方向D 续:条件式止盈 + 10周线止损 —— 6 组离场规则的交易级 + 组合级评估

═══ 为什么测这个 ═══
当前最优(10只/小市值/择时,-10%固定止损、无止盈、最长252日):
  年化 +6.34% / Sharpe 0.377 / **回撤 -62.33%**
等权基准(修复后 OOS):年化 7.22% / Sharpe 0.423 / 回撤 -32.77%
**输在回撤**,不在收益。

阶段1 已显示:移动止盈把胜率 18%→36%,但均盈 +69%→+19%,**净期望反而降**
——过早保护利润会杀死大赢家,而大赢家是这套方法的全部收益来源。
用户提的"涨幅≥100%后再启动移动止盈"正好绕开该矛盾(前半程让它跑)。
用户提的"10周线止损"与我此前测法不同:此前 MA50 是**叠加**在固定止损之上的
额外离场条件,用户说的是把它当**止损本身**(止损线随股价上移)。

═══ 6 组固定规则,不做网格搜索 ═══
移动止盈回撤档只用 20%(阶段1 最好的一档),启动阈值只用 +100%(用户提的值)。
6 组本身构成一次小规模搜索 —— 结果需对照本session实测的纯噪音 best-of-N
lift 中位数 2.09 来判断组间差异是否超出噪音。

═══ 必须保留的正确性要素(前一轮已建立) ═══
1. 入场=突破次日开盘价;止损判断用当日**最低价**;止盈用当日**最高价**
2. 跳空穿越止损线 → 以**开盘价**成交(实测 12.1% 的止损属此类)
3. 价格中断(退市/长停)→ 按最后有效价平仓,**不跳过该笔**
4. 择时基准用 **510300**
5. "跌破MA50" → **次日开盘**离场(不是当日收盘成交),两个层次口径一致

═══ 判据(事前写死) ═══
基线 +6.34% / -62.33%。某规则需做到 **年化≥6% 且 回撤≤-45%** 才算实质改进。
"回撤改善但年化大跌"不算成功。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"

COST_TRADE = 0.003      # 交易级:双边 0.3%
COST_PF = 0.003         # 组合级:单边 0.3%(与阶段2-4 口径一致)
SLOTS = 10
SEED = 20260810
INF = float("inf")

# stop=固定止损比例(None=无固定止损)
# ma_mode: none / pure(全程用MA50) / takeover(站上MA50后接管) / arm100(涨100%后接管)
# trail=移动止盈回撤幅度; arm=移动止盈启动阈值(涨幅)
RULES = {
    "A": dict(name="基线:-10%固定止损,无止盈,252日",
              stop=0.10, ma_mode="none", trail=None, arm=None, max_hold=252),
    "B": dict(name="纯10周线:无固定止损,跌破MA50离场",
              stop=None, ma_mode="pure", trail=None, arm=None, max_hold=252),
    "C": dict(name="固定+10周线接管:站上MA50后改用MA50",
              stop=0.10, ma_mode="takeover", trail=None, arm=None, max_hold=252),
    "D": dict(name="条件移动止盈:涨≥100%后启动20%回撤止盈",
              stop=0.10, ma_mode="none", trail=0.20, arm=1.00, max_hold=252),
    "E": dict(name="条件+10周线:涨≥100%后改用MA50离场",
              stop=0.10, ma_mode="arm100", trail=None, arm=1.00, max_hold=252),
    "F": dict(name="基线+放开持有期:504日",
              stop=0.10, ma_mode="none", trail=None, arm=None, max_hold=504),
}

t0 = time.time()

# ---------------- 面板 ----------------
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
HI = pd.DataFrame(h).set_axis(OP.index)
LO = pd.DataFrame(l).set_axis(OP.index)
CL = pd.DataFrame(c).set_axis(OP.index)
MV = pd.DataFrame(mv).set_axis(OP.index)
# 源数据含负价格(200418 136行、000418 3行,全在2013年)。非正价会让收益出现 inf。
OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
MA50 = CL.rolling(50, min_periods=50).mean()      # 10周线,与阶段1 口径一致
idx = OP.index
pos = {d: i for i, d in enumerate(idx)}
NT = len(idx)
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")

OPa = OP.to_numpy(np.float64); HIa = HI.to_numpy(np.float64)
LOa = LO.to_numpy(np.float64); CLa = CL.to_numpy(np.float64)
MVa = MV.to_numpy(np.float64); MAa = MA50.to_numpy(np.float64)
col_of = {cd: i for i, cd in enumerate(OP.columns)}
del o, h, l, c, mv

# ---------------- 事件 ----------------
ev = pd.read_csv(f"{SP}/oneil_prelaunch_events_fixed.csv",
                 usecols=["code", "D"], dtype={"code": str})
ev["code"] = ev["code"].str.zfill(6)
ev["D"] = pd.to_datetime(ev["D"]).dt.tz_localize(None)
ev = ev[ev.code.isin(OP.columns)].copy()
ev["dp"] = ev["D"].map(pos)
ev = ev.dropna(subset=["dp"])
ev["dp"] = ev["dp"].astype(int)
ev = ev[ev.dp + 1 < NT - 5]
by_day = {d: g["code"].tolist() for d, g in ev.groupby("dp")}
print(f"可用突破事件 {len(ev):,},覆盖 {len(by_day):,} 个交易日  ({time.time()-t0:.0f}s)")

# ---------------- 择时基准 ----------------
_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()
assert 0.25 < np.nanmean(mkt_ok) < 0.80, "择时基准比例异常"
print(f"择时基准 510300 在MA200之上比例 {np.nanmean(mkt_ok):.1%}")


# ══════════════════ 共用的离场判断 ══════════════════
def step(rc, hd, t, op_t, hi_t, lo_t, cl_t, ma_t):
    """推进一天,返回 (exit_px|None, reason)。hd 就地更新 peak/armed/armed_ma/pending。

    调用方必须先处理 hd['pending'](次日开盘离场),再调用本函数。
    """
    stop_f, ma_mode, trail, arm = rc["stop"], rc["ma_mode"], rc["trail"], rc["arm"]

    # 1) 固定止损 —— MA50 接管后即失效(用户要的是"止损线随股价上移")
    taken_over = ma_mode in ("takeover", "arm100") and hd["armed_ma"]
    if stop_f is not None and not taken_over:
        if np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
            # 跳空穿越 → 以开盘价成交
            px = op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
            return px, "固定止损"

    # 2) 峰值更新(用最高价)
    if np.isfinite(hi_t) and hi_t > hd["peak"]:
        hd["peak"] = hi_t

    # 3) 启动阈值:涨幅达 arm 后才激活移动止盈 / MA50 接管
    if arm is not None and hd["peak"] >= hd["entry"] * (1 + arm):
        if trail is not None:
            hd["armed"] = True
        if ma_mode == "arm100":
            hd["armed_ma"] = True

    # 4) 移动止盈(仅在激活后)
    if trail is not None and hd["armed"]:
        tp = hd["peak"] * (1 - trail)
        if np.isfinite(lo_t) and lo_t <= tp:
            px = op_t if (np.isfinite(op_t) and op_t < tp) else tp
            return px, "移动止盈"

    # 5) 10周线:收盘跌破 → 次日开盘离场
    if ma_mode != "none" and np.isfinite(ma_t) and np.isfinite(cl_t):
        if ma_mode == "takeover" and not hd["armed_ma"] and cl_t > ma_t:
            hd["armed_ma"] = True          # 首次站上MA50,接管
        elif hd["armed_ma"] and cl_t < ma_t:
            hd["pending"] = True
    return None, None


def new_pos(rc, entry, t):
    return {"entry": entry, "peak": entry, "t_in": t, "last": entry,
            "stop_px": entry * (1 - rc["stop"]) if rc["stop"] is not None else -INF,
            "armed": False, "armed_ma": (rc["ma_mode"] == "pure"), "pending": False}


# ══════════════════ 层次1:交易级 ══════════════════
def run_trade_level(rc):
    """逐笔路径回放。返回 DataFrame(每笔的毛收益、天数、离场原因)。"""
    max_hold = rc["max_hold"]
    codes, drets, ddays, dreason, dyear, ddp = [], [], [], [], [], []
    for code, grp in ev.groupby("code", sort=False):
        ci = col_of[code]
        op, hi, lo, cl, ma = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci], MAa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            hd = new_pos(rc, entry, e)
            end = min(e + max_hold, NT - 1)
            exit_px, reason, texit = None, "到期", None
            for t in range(e, end + 1):
                if hd["pending"]:
                    px = op[t] if np.isfinite(op[t]) else cl[t]
                    if np.isfinite(px):
                        exit_px, reason, texit = px, "跌破MA50", t
                        break
                    hd["pending"] = False          # 无价可成交,继续持有
                if not np.isfinite(cl[t]):
                    continue
                hd["last"] = cl[t]
                px, why = step(rc, hd, t, op[t], hi[t], lo[t], cl[t], ma[t])
                if px is not None:
                    exit_px, reason, texit = px, why, t
                    break
            if exit_px is None:
                texit = end
                exit_px = cl[end] if np.isfinite(cl[end]) else hd["last"]
            if not np.isfinite(exit_px) or exit_px <= 0:
                continue
            codes.append(code)
            drets.append(exit_px / entry - 1)
            ddays.append(texit - e + 1)
            dreason.append(reason)
            dyear.append(idx[e].year)
            ddp.append(dp)
    return pd.DataFrame({"code": codes, "ret": drets, "days": ddays,
                         "reason": dreason, "year": dyear, "dp": ddp})


def trace(rc, code, dp):
    """单笔逐日追踪,返回可打印的诊断信息(用于人工核对规则是否按设计触发)。"""
    ci = col_of[code]
    op, hi, lo, cl, ma = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci], MAa[:, ci]
    e = dp + 1
    entry = op[e]
    hd = new_pos(rc, entry, e)
    end = min(e + rc["max_hold"], NT - 1)
    exit_px, reason, texit, t_arm = None, "到期", None, None
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
        was = hd["armed"] or hd["armed_ma"]
        px, why = step(rc, hd, t, op[t], hi[t], lo[t], cl[t], ma[t])
        if t_arm is None and not was and (hd["armed"] or hd["armed_ma"]):
            t_arm = t
        if px is not None:
            exit_px, reason, texit = px, why, t
            break
    if exit_px is None:
        texit = end
        exit_px = cl[end] if np.isfinite(cl[end]) else hd["last"]
    arm_s = (f"启动于{idx[t_arm].date()}(峰值{hd['peak']/entry-1:+.0%})"
             if t_arm is not None else "未启动")
    return (f"入场{idx[e].date()}@{entry:.2f} 峰值{hd['peak']:.2f}({hd['peak']/entry-1:+.0%}) "
            f"{arm_s} → 离场{idx[texit].date()}@{exit_px:.2f} [{reason}] "
            f"{exit_px/entry-1:+.1%} {texit-e+1}天")


def trade_stats(df):
    r = df["ret"].to_numpy()
    win = r > 0
    aw = r[win].mean() if win.any() else 0.0
    al = r[~win].mean() if (~win).any() else 0.0
    return {
        "笔数": len(r), "胜率": win.mean(), "均盈": aw, "均亏": al,
        "盈亏比": abs(aw / al) if al != 0 else np.nan,
        "毛期望": r.mean(), "净期望": r.mean() - COST_TRADE,
        "中位天数": np.median(df["days"]), "最大单笔": r.max(),
        ">100%笔数": int((r > 1.0).sum()), ">100%占比": (r > 1.0).mean(),
    }


# 规则B(纯10周线)在 MA50 缺失时**完全无保护** —— 先量化这个盲区有多大
_e = ev["dp"].to_numpy() + 1
_ci = np.array([col_of[cd] for cd in ev["code"]])
_ma_na = ~np.isfinite(MAa[_e, _ci])
print(f"入场日 MA50 不可用的事件占比 {_ma_na.mean():.2%} "
      f"(规则B 在这些笔上全程无止损,只受 max_hold 约束)")
_above = np.isfinite(MAa[_e, _ci]) & (CLa[_e, _ci] > MAa[_e, _ci])
print(f"入场日收盘已在 MA50 之上的占比 {_above.sum()/np.isfinite(MAa[_e,_ci]).sum():.1%} "
      f"(规则C 会立即接管,故 C 预期接近 B)")

print(f"\n{'='*118}")
print("层次1 交易级(入场=突破次日开盘;止损用最低价;止盈用最高价;跳空按开盘价成交)")
print(f"{'='*118}")
print(f"{'':<3}{'规则':<38}{'笔数':>8}{'胜率':>7}{'均盈':>9}{'均亏':>8}"
      f"{'盈亏比':>7}{'毛期望':>8}{'净期望':>8}{'中位天':>7}{'最大单笔':>10}{'>100%':>8}")

trade_res, trade_df = {}, {}
for key, rc in RULES.items():
    df = run_trade_level(rc)
    trade_df[key] = df
    s = trade_stats(df)
    trade_res[key] = s
    print(f"{key:<3}{rc['name']:<38}{s['笔数']:>8,}{s['胜率']:>7.1%}{s['均盈']:>+9.1%}"
          f"{s['均亏']:>+8.1%}{s['盈亏比']:>7.2f}{s['毛期望']:>+8.2%}{s['净期望']:>+8.2%}"
          f"{s['中位天数']:>7.0f}{s['最大单笔']:>+10.0%}{s['>100%笔数']:>8,}"
          f"   ({time.time()-t0:.0f}s)")

TR = pd.DataFrame(trade_res).T
TR.insert(0, "规则", [RULES[k]["name"] for k in TR.index])
TR.to_csv(f"{SP}/breakout_exit_rules_trade.csv")

# ---- 离场原因分布 ----
print(f"\n{'='*118}\n离场原因分布(检验各规则是否按设计触发)\n{'='*118}")
for key in RULES:
    vc = trade_df[key]["reason"].value_counts(normalize=True)
    print(f"  {key}  " + "  ".join(f"{k} {v:.1%}" for k, v in vc.items()))

# ---- 大赢家留存检查 ----
print(f"\n{'='*118}\n验证2 大赢家留存:条件式止盈是否真的保住了大赢家\n{'='*118}")
print(f"{'':<3}{'>100%笔数':>11}{'>100%占比':>11}{'>200%笔数':>11}{'最大单笔':>11}"
      f"{'前1%笔均收益':>14}{'前1%贡献毛期望':>16}")
for key in RULES:
    r = np.sort(trade_df[key]["ret"].to_numpy())[::-1]
    k1 = max(1, len(r) // 100)
    print(f"{key:<3}{int((r>1).sum()):>11,}{(r>1).mean():>11.2%}{int((r>2).sum()):>11,}"
          f"{r.max():>+11.0%}{r[:k1].mean():>+14.1%}{r[:k1].sum()/len(r):>+16.2%}")

# ---- 抽查:同一批交易在 6 组规则下的表现(可直接横向对比) ----
# 用规则F(最长持有、无止盈)挑样本:2笔大赢家 + 1笔亏损,
# 大赢家用于验证 D/E 的"涨幅≥100%才启动"是否真的在达标后才生效。
print(f"\n{'='*118}\n验证1 抽查:同一批交易在6组规则下逐日核对(2笔大赢家 + 1笔亏损)\n{'='*118}")
_f = trade_df["F"]
rng = np.random.default_rng(SEED)
big = _f[_f.ret > 2.0]
bad = _f[_f.ret < -0.05]
picks = []
if len(big):
    picks += [(r.code, int(r.dp)) for _, r in big.sample(min(2, len(big)), random_state=SEED).iterrows()]
if len(bad):
    picks += [(r.code, int(r.dp)) for _, r in bad.sample(1, random_state=SEED).iterrows()]
for cd, dp in picks:
    print(f"\n  【{cd}】突破日 {idx[dp].date()}")
    for key, rc in RULES.items():
        print(f"     {key}  {trace(rc, cd, dp)}")

# ---- 分段 ----
print(f"\n{'='*118}\n验证4 分段:牛市年(2015/2020/2025)vs 熊市年(2018/2022),按入场年份\n{'='*118}")
print(f"{'':<3}{'牛市年笔数':>11}{'牛市净期望':>12}{'牛市胜率':>10}"
      f"{'熊市年笔数':>12}{'熊市净期望':>12}{'熊市胜率':>10}")
BULL, BEAR = (2015, 2020, 2025), (2018, 2022)
for key in RULES:
    d = trade_df[key]
    b = d[d.year.isin(BULL)]["ret"].to_numpy()
    x = d[d.year.isin(BEAR)]["ret"].to_numpy()
    print(f"{key:<3}{len(b):>11,}{b.mean()-COST_TRADE:>+12.2%}{(b>0).mean():>10.1%}"
          f"{len(x):>12,}{x.mean()-COST_TRADE:>+12.2%}{(x>0).mean():>10.1%}")


# ══════════════════ 层次2:组合级 ══════════════════
def run_portfolio(rc, n_slots=SLOTS, cost=COST_PF, pick="small", use_timing=True, seed=SEED):
    rng2 = np.random.default_rng(seed)
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    n_trades, trs = 0, []
    max_hold = rc["max_hold"]
    start = 200
    for t in range(start, NT):
        # --- 平仓 ---
        for code in list(holds):
            hd = holds[code]
            ci = hd["ci"]
            op_t, hi_t, lo_t, cl_t, ma_t = OPa[t, ci], HIa[t, ci], LOa[t, ci], CLa[t, ci], MAa[t, ci]
            exit_px = None
            if hd["pending"]:
                # 昨日收盘跌破MA50 → 今日开盘离场
                exit_px = op_t if np.isfinite(op_t) else (cl_t if np.isfinite(cl_t) else hd["last"])
            elif not np.isfinite(cl_t):
                exit_px = hd["last"]        # 退市/长停:按最后有效价平仓,不跳过
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

        # --- 开仓(昨日突破,今日开盘入场) ---
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
    return eq, n_trades, np.array(trs)


def pf_stats(eq, n_trades, tr):
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    return {"年化": ann,
            "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "最大回撤": (eq / eq.cummax() - 1).min(),
            "年均笔数": n_trades / yrs,
            "笔均收益": tr.mean() if len(tr) else np.nan,
            "胜率": (tr > 0).mean() if len(tr) else np.nan}


print(f"\n{'='*118}")
print(f"层次2 组合级({SLOTS}只持仓、510300择时、{COST_PF:.1%}成本;沿用阶段2-4口径以保证可比)")
print(f"{'='*118}")
print(f"{'':<3}{'规则':<38}{'选股':<10}{'年化':>9}{'Sharpe':>9}{'最大回撤':>10}"
      f"{'年均笔数':>10}{'笔均':>9}{'胜率':>8}")
pf_rows = []
for key, rc in RULES.items():
    for pick in ("small", "random"):
        eq, nt, tr = run_portfolio(rc, pick=pick)
        s = pf_stats(eq, nt, tr)
        s.update({"规则": key, "说明": rc["name"],
                  "选股": "小市值优先" if pick == "small" else "随机选"})
        pf_rows.append(s)
        print(f"{key:<3}{rc['name']:<38}{s['选股']:<10}{s['年化']:>+9.2%}{s['Sharpe']:>+9.3f}"
              f"{s['最大回撤']:>10.2%}{s['年均笔数']:>10.0f}{s['笔均收益']:>+9.2%}"
              f"{s['胜率']:>8.1%}   ({time.time()-t0:.0f}s)")

PF = pd.DataFrame(pf_rows)[["规则", "说明", "选股", "年化", "Sharpe", "最大回撤",
                            "年均笔数", "笔均收益", "胜率"]]
PF.to_csv(f"{SP}/breakout_exit_rules_portfolio.csv", index=False)

# ---------------- 判据 ----------------
print(f"\n{'='*118}\n判据(事前写死):年化≥6% 且 最大回撤≤-45% 才算实质改进\n{'='*118}")
print("对照  基线A(前一轮实测)  年化 +6.34% / Sharpe 0.377 / 回撤 -62.33%")
print("对照  全市场等权基准 OOS   年化 +7.22% / Sharpe 0.423 / 回撤 -32.77%")
sm = PF[PF["选股"] == "小市值优先"]
ok = sm[(sm["年化"] >= 0.06) & (sm["最大回撤"] >= -0.45)]
if len(ok):
    print("\n**通过判据的规则:**")
    for _, r in ok.iterrows():
        print(f"  {r['规则']} {r['说明']}: 年化 {r['年化']:+.2%} / 回撤 {r['最大回撤']:.2%}")
else:
    print("\n**没有规则同时满足 年化≥6% 且 回撤≤-45% → 两条改进均未达到实质改进标准。**")
print(f"\n组间离散度(小市值优先):年化 {sm['年化'].min():+.2%} ~ {sm['年化'].max():+.2%},"
      f"回撤 {sm['最大回撤'].min():.2%} ~ {sm['最大回撤'].max():.2%}")
print("  注:6 组本身构成一次小规模搜索。本session实测纯噪音 best-of-N lift 中位数 2.09,")
print("      若最优组仅比基线高出该量级,不能当作真实改进。")

print(f"\n耗时 {time.time()-t0:.0f}s")
print("Saved: breakout_exit_rules_trade.csv, breakout_exit_rules_portfolio.csv")
