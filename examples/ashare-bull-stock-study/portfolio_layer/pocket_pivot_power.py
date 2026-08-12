"""测试3:口袋支点扩样本 —— 判据③败在样本量还是败在表现?

═══ 第五十五节留下的问题 ═══
「口袋支点 ∩ 60日新高」是唯一同时满足①②的配置:
  净期望 **+7.62%**(判据 +6.0% ✓)、组合年化 **+8.40%**(判据 +7.22% ✓)、
  回撤 -47.4%(比基线好 15pp)
**但判据③没过:组合级 p=0.150。**
原因是事件只有 24,094 笔,随机对照区间宽到 **[-3.27%, +10.06%]** ——
+8.40% 掉在里面。**是样本量不够,不是表现不行 —— 这两件事必须分开。**

═══ 本脚本怎么分开 ═══
**先做功效分析(power analysis),再做检验。**
随机对照的区间宽度只取决于**事件数**,与真实信号无关。
所以可以先测:事件数 k 取 5k/10k/24k/40k/64k 时,随机对照的年化区间有多宽?
→ 得到「要检出 +8.40% 与随机中位数的差距,至少需要多少笔事件」。
**如果 24,094 笔本来就没有功效,那 p=0.150 什么也没否定;
如果 24,094 笔功效足够,那 p=0.150 就是真的否定了它。**

═══ 然后才是扩样本 ═══
把 ∩ 条件放宽三档,看事件数上去后 p 是否随之下降:
  ∩60日新高(原始,24k)→ ∩30日新高 → ∩「距60日高点 -5% 内」→ 口袋支点全集(64k)
**放宽条件会稀释信号**,所以这不是「换个条件试到过为止」——
如果 p 随事件数增加而下降,说明原来确实是功效问题;
如果 p 不降反升,说明信号本来就弱,扩样本救不了。**两种结果都要报。**

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 交易级净期望 ≥ +6.0%/笔
  ② 组合级年化 ≥ +7.22%
  ③ 组合级 p_年化 < 0.05
另加一条**事前声明**:本脚本测 4 个 ∩ 变体,是一次搜索。
**最好的那个的 p 值要按 4 次比较做 Bonferroni 校正(即需 p < 0.0125)。**

**锚点**:60日新高 = 70,318 笔 / +4.61%/笔 / 组合 +6.34%。
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
RULE_A = dict(stop=0.10, max_hold=252)

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

FWD_WIN, BASE_MAX_RANGE = 252, 0.50
_rmax60 = CL.rolling(60, min_periods=60).max()
_rmin60 = CL.rolling(60, min_periods=60).min()
BASE_OK = (((_rmax60 - _rmin60) / _rmin60.replace(0, np.nan)).shift(1) < BASE_MAX_RANGE).to_numpy()
BRK60 = (CLa > _rmax60.shift(1).to_numpy()) & BASE_OK
_rmax30 = CL.rolling(30, min_periods=30).max()
BRK30 = (CLa > _rmax30.shift(1).to_numpy()) & BASE_OK
NEAR60 = (CLa >= _rmax60.shift(1).to_numpy() * 0.95) & BASE_OK   # 距60日高点 5% 内
prev_c = CL.shift(1)
dn_vol = VO.where(CL < prev_c, 0.0)
PP = ((CL > prev_c) & (VO > dn_vol.rolling(10, min_periods=5).max().shift(1))
      & (CL > (HI + LO) / 2) & (CL > MA50) & (MA50 > MA50.shift(10))
      & ((CL / MA50 - 1) <= 0.10)).to_numpy()
LAST_OK = NT - 1 - FWD_WIN


def to_events(hit, gap=60):
    codes, dps = [], []
    for j, cd in enumerate(OP.columns):
        last = -10**9
        for q in np.flatnonzero(hit[:, j]):
            if q - last < gap or q == 0 or q > LAST_OK:
                continue
            last = q
            codes.append(cd); dps.append(int(q))
    return pd.DataFrame({"code": codes, "dp": dps})


def run_trade(evs):
    out = []
    for code, grp in evs.groupby("code", sort=False):
        ci = col_of[code]
        op, hi, lo, cl = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            stop_px, last = entry * (1 - RULE_A["stop"]), entry
            end = min(e + RULE_A["max_hold"], NT - 1)
            ex = None
            for t in range(e, end + 1):
                if not np.isfinite(cl[t]):
                    continue
                last = cl[t]
                if np.isfinite(lo[t]) and lo[t] <= stop_px:
                    ex = op[t] if (np.isfinite(op[t]) and op[t] < stop_px) else stop_px
                    break
            if ex is None:
                ex = cl[end] if np.isfinite(cl[end]) else last
            if np.isfinite(ex) and ex > 0:
                out.append(ex / entry - 1)
    return np.array(out)


def run_pf(evs, seed=SEED):
    by_day = {d: g["code"].tolist() for d, g in evs.groupby("dp")}
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    start = 200
    for t in range(start, NT):
        for code in list(holds):
            hd = holds[code]
            ci = hd["ci"]
            op_t, lo_t, cl_t = OPa[t, ci], LOa[t, ci], CLa[t, ci]
            ex = None
            if not np.isfinite(cl_t):
                ex = hd["last"]
            else:
                hd["last"] = cl_t
                if np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
                    ex = op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
                elif t - hd["t_in"] >= RULE_A["max_hold"]:
                    ex = cl_t
            if ex is not None and np.isfinite(ex) and ex > 0:
                cash += hd["shares"] * ex * (1 - COST_PF)
                del holds[code]
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
                holds[cd] = {"entry": px, "t_in": t, "last": px, "ci": col_of[cd],
                             "stop_px": px * (1 - RULE_A["stop"]),
                             "shares": alloc * (1 - COST_PF) / px}
                cash -= alloc
        equity[t] = cash + sum(
            hd["shares"] * (CLa[t, hd["ci"]] if np.isfinite(CLa[t, hd["ci"]]) else hd["last"])
            for hd in holds.values())
    eq = pd.Series(equity[start:], index=idx[start:])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0


BASE = to_events(BRK60)
print(f"\n锚点:60日新高 {len(BASE):,} 笔(应 70,310)")
_r = run_trade(BASE)
_a = run_pf(BASE)
print(f"  净期望 {_r.mean()-COST_TRADE:+.2%}(应 +4.61%)   组合年化 {_a:+.2%}(应 +6.34%)")
assert abs(len(BASE) - 70310) <= 50 and abs(_r.mean() - COST_TRADE - 0.0461) < 0.0015
assert abs(_a - 0.0634) < 0.004
print("  锚点通过")

# ══════════ 功效分析:随机对照的区间宽度只取决于事件数 ══════════
print(f"\n{'='*112}")
print("【功效分析】先问「24,094 笔到底有没有能力检出差异」,再看 p 值")
print(f"{'='*112}")
print(f"{'事件数 k':>10}{'随机年化 中位':>16}{'2.5%分位':>12}{'97.5%分位':>12}"
      f"{'区间宽度':>12}{'能否检出 +8.40%':>18}")
rng = np.random.default_rng(SEED)
power = {}
for k in (5000, 10000, 24094, 40000, 64641):
    anns = []
    for s in range(N_RAND):
        sub = BASE.iloc[rng.choice(len(BASE), min(k, len(BASE)), replace=False)]
        anns.append(run_pf(sub, seed=SEED + s))
    anns = np.array(anns)
    q = np.quantile(anns, [0.025, 0.975])
    power[k] = anns
    can = "✓ 能" if 0.0840 > q[1] else "**✗ 不能**"
    print(f"{k:>10,}{np.median(anns):>+16.2%}{q[0]:>+12.2%}{q[1]:>+12.2%}"
          f"{(q[1]-q[0])*100:>11.1f}pp{can:>18}   ({time.time()-t0:.0f}s)")

# ══════════ 扩样本:四个 ∩ 变体 ══════════
print(f"\n{'='*112}\n扩样本:放宽 ∩ 条件,看 p 是否随事件数下降\n{'='*112}")
VARIANTS = {
    "口袋支点 ∩ 60日新高(原始)": to_events(PP & BRK60),
    "口袋支点 ∩ 30日新高": to_events(PP & BRK30),
    "口袋支点 ∩ 距60日高点5%内": to_events(PP & NEAR60),
    "口袋支点 全集": to_events(PP),
}
print(f"{'变体':<30}{'事件数':>9}{'净期望':>10}{'年化':>10}"
      f"{'随机中位':>11}{'随机区间':>22}{'p_年化':>9}")
rows = []
for nm, ev in VARIANTS.items():
    if len(ev) < 50:
        continue
    r = run_trade(ev)
    a = run_pf(ev)
    k = len(ev)
    anns = []
    for s in range(N_RAND):
        sub = BASE.iloc[rng.choice(len(BASE), min(k, len(BASE)), replace=False)]
        anns.append(run_pf(sub, seed=SEED + s))
    anns = np.array(anns)
    p = float((anns >= a).mean())
    rows.append({"变体": nm, "事件数": k, "净期望": r.mean() - COST_TRADE, "年化": a,
                 "随机中位": float(np.median(anns)), "p_年化": p})
    print(f"{nm:<30}{k:>9,}{r.mean()-COST_TRADE:>+10.2%}{a:>+10.2%}"
          f"{np.median(anns):>+11.2%}   [{anns.min():+.2%}, {anns.max():+.2%}]"
          f"{p:>9.3f}   ({time.time()-t0:.0f}s)")

R = pd.DataFrame(rows)
R.to_csv(f"{SP}/pocket_pivot_power.csv", index=False)

print(f"\n{'='*112}\n判定(事前判据 + Bonferroni 校正,未放宽)\n{'='*112}")
ALPHA = 0.05 / len(R)
print(f"  4 个变体是一次搜索 → Bonferroni 校正后需 **p < {ALPHA:.4f}**")
for _, r in R.iterrows():
    c1, c2 = r["净期望"] >= 0.060, r["年化"] >= 0.0722
    c3 = r["p_年化"] < ALPHA
    print(f"  {r['变体']}:")
    print(f"    ① 净期望≥+6.0% → {r['净期望']:+.2%} {'✓' if c1 else '✗'}   "
          f"② 年化≥+7.22% → {r['年化']:+.2%} {'✓' if c2 else '✗'}   "
          f"③ p<{ALPHA:.4f} → {r['p_年化']:.3f} {'✓' if c3 else '✗'}")
    print(f"    **{'算发现' if (c1 and c2 and c3) else '不算发现'}**")

print(f"\n耗时 {time.time()-t0:.0f}s   Saved: pocket_pivot_power.csv")
