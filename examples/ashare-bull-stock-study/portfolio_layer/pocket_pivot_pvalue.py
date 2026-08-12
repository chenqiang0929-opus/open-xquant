"""口袋支点判据③的 p 值到底是多少 —— 20 个种子定不了,用 300 个

═══ 必须做这一步的理由 ═══
**同一个变体、同样 24,094 笔事件、同样 20 个种子的随机对照,两次跑出相反结论:**

  §55  `pocket_pivot.py`        随机中位 +3.38%  区间 [-3.27%, +10.06%]  **p=0.150 → 不算发现**
  测试3 `pocket_pivot_power.py` 随机中位 +4.52%  区间 [-1.33%,  +7.98%]  **p=0.000 → 算发现**

两次唯一的差别是**抽到了不同的 20 个随机子集**(第二个脚本先做功效分析,
消耗了 RNG,后面的抽签就不一样了)。
而功效分析自己给出的答案是:**k=24,094 时随机分布的 95% 区间宽 10.1pp** ——
在这么宽的分布上用 20 个点估 p,标准误约 ±0.1,**0.000 和 0.150 本来就分不开**。

**结论不能取我喜欢的那个,得把 p 估准。** 本脚本对两个候选变体各跑 **300 次**随机对照。
300 次时 p 的标准误 ≈ sqrt(p(1-p)/300),p≈0.05 时约 ±0.013 —— 够用了。

═══ 事前判据(与前两节完全相同,不放宽) ═══
  ① 净期望 ≥ +6.0%   ② 年化 ≥ +7.22%   ③ p_年化 < 0.0125(4 变体 Bonferroni)
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, SEED, N_RAND = 10, 20260810, 300
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
OPa, LOa, CLa, MVa = OP.to_numpy(float), LO.to_numpy(float), CL.to_numpy(float), MV.to_numpy(float)
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
NEAR60 = (CLa >= _rmax60.shift(1).to_numpy() * 0.95) & BASE_OK
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
a0 = run_pf(BASE)
print(f"锚点:60日新高 {len(BASE):,} 笔  组合年化 {a0:+.2%}(应 +6.34%)")
assert abs(len(BASE) - 70310) <= 50 and abs(a0 - 0.0634) < 0.004
print("锚点通过\n")

CAND = {"口袋支点 ∩ 60日新高": (to_events(PP & BRK60), 0.0840),
        "口袋支点 ∩ 距60日高点5%内": (to_events(PP & NEAR60), 0.1052)}

print(f"{'='*112}")
print(f"每个变体跑 **{N_RAND} 次**随机对照(§55 与测试3 各只跑了 20 次,结论相反)")
print(f"{'='*112}")
rows = []
for nm, (ev, _prev) in CAND.items():
    real = run_pf(ev)
    k = len(ev)
    rng = np.random.default_rng(SEED)
    anns = np.empty(N_RAND)
    for s in range(N_RAND):
        sub = BASE.iloc[rng.choice(len(BASE), k, replace=False)]
        anns[s] = run_pf(sub)
        if (s + 1) % 100 == 0:
            p_now = float((anns[:s + 1] >= real).mean())
            print(f"  {nm}  已跑 {s+1:>3} 次   当前 p={p_now:.4f}   ({time.time()-t0:.0f}s)")
    p = float((anns >= real).mean())
    se = float(np.sqrt(max(p * (1 - p), 1e-9) / N_RAND))
    q = np.quantile(anns, [0.025, 0.5, 0.975])
    rows.append({"变体": nm, "事件数": k, "年化": real, "随机中位": q[1],
                 "随机2.5%": q[0], "随机97.5%": q[2], "p": p, "p标准误": se})
    print(f"\n  **{nm}**(事件 {k:,})")
    print(f"    实际年化 **{real:+.2%}**")
    print(f"    {N_RAND} 次随机:中位 {q[1]:+.2%}  95%区间 [{q[0]:+.2%}, {q[2]:+.2%}]"
          f"  最大 {anns.max():+.2%}")
    print(f"    **p = {p:.4f}**  (标准误 ±{se:.4f})\n")

R = pd.DataFrame(rows)
R.to_csv(f"{SP}/pocket_pivot_pvalue.csv", index=False)

print(f"{'='*112}\n判定(判据不放宽:①≥+6.0% ②≥+7.22% ③p<0.0125)\n{'='*112}")
NET = {"口袋支点 ∩ 60日新高": 0.0762, "口袋支点 ∩ 距60日高点5%内": 0.0657}
for _, r in R.iterrows():
    c1, c2, c3 = NET[r["变体"]] >= 0.060, r["年化"] >= 0.0722, r["p"] < 0.0125
    print(f"  {r['变体']}:")
    print(f"    ① 净期望 {NET[r['变体']]:+.2%} {'✓' if c1 else '✗'}   "
          f"② 年化 {r['年化']:+.2%} {'✓' if c2 else '✗'}   "
          f"③ p={r['p']:.4f} {'✓' if c3 else '✗'}")
    print(f"    **{'算发现' if (c1 and c2 and c3) else '不算发现'}**")

print(f"\n耗时 {time.time()-t0:.0f}s   Saved: pocket_pivot_pvalue.csv")
