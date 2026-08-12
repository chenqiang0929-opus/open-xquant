"""测试2 补跑:随机排序必须多种子 —— 首版只跑了一个种子,不能采信

═══ 为什么必须补这一步 ═══
`portfolio_layer_fix.py` 的「随机排序」列**每格只跑了一个种子**,
于是出现这种东西:

  60日新高 / 20仓 / inv_vol / 随机  = **+12.27%**  ← 32 格里唯一通过判据的
  60日新高 / 10仓 / inv_vol / 随机  = +3.94%
  60日新高 / 30仓 / inv_vol / 随机  = +5.20%

**同一条曲线上 10→20→30 仓是 +3.94% → +12.27% → +5.20%**,毫无单调性。
第四十一节早就写死过一条纪律:「**单次回测不可采信**:随机选必须跑多种子」。
首版违反了它,那个 +12.27% 极可能只是一个走运的抽签。

本脚本把每个「随机排序」格子跑 **20 个种子**,报中位数与区间。
确定性排序(小市值优先)只有一个值,直接并列对比。

═══ 事前判据(与首版相同,不放宽) ═══
  ① 组合年化 ≥ +11.88%   ② 最大回撤优于 -47.4%
**随机排序的格子按中位数判定,不按最好的种子判定。**
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_PF, SEED, N_SEED = 0.003, 20260810, 20
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
VOL20 = CL.pct_change().rolling(20, min_periods=10).std()
idx = OP.index
NT = len(idx)
OPa, LOa = OP.to_numpy(float), LO.to_numpy(float)
CLa, MVa = CL.to_numpy(float), MV.to_numpy(float)
VOL20a = VOL20.to_numpy(float)
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


EVENTS = {"60日新高": to_events(BRK60), "口袋支点": to_events(PP)}


def run_pf(evs, slots, weight, pick, seed):
    by_day = {d: g["code"].tolist() for d, g in evs.groupby("dp")}
    rng = np.random.default_rng(seed)
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
        free = slots - len(holds)
        if cands and free > 0 and mkt_ok[t]:
            cands = [cd for cd in cands if cd not in holds
                     and np.isfinite(OPa[t, col_of[cd]]) and OPa[t, col_of[cd]] > 0]
            if pick == "small":
                cands.sort(key=lambda cd: MVa[t, col_of[cd]]
                           if np.isfinite(MVa[t, col_of[cd]]) else np.inf)
            else:
                rng.shuffle(cands)
            take = cands[:free]
            budget = cash * len(take) / free if take else 0.0
            if weight == "inv_vol" and take:
                v = np.array([VOL20a[t, col_of[cd]] for cd in take], float)
                good = np.isfinite(v) & (v > 0)
                med = np.median(v[good]) if good.any() else 0.02
                v[~good] = med
                share = (1.0 / v) / (1.0 / v).sum()
            else:
                share = np.full(len(take), 1.0 / max(len(take), 1))
            for cd, sh in zip(take, share):
                alloc = budget * sh
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
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    return ann, (eq / eq.cummax() - 1).min()


a0, _ = run_pf(EVENTS["60日新高"], 10, "eq", "small", SEED)
print(f"锚点:60日新高/10仓/等额/小市值 = {a0:+.2%}(应 +6.34%)")
assert abs(a0 - 0.0634) < 0.004
print("锚点通过\n")

print(f"{'='*118}")
print(f"随机排序:每格 {N_SEED} 个种子(首版每格只有 1 个,那个 +12.27% 就是这么来的)")
print(f"{'='*118}")
print(f"{'买点':<10}{'仓位':>6}{'权重':>10}{'小市值(确定)':>14}"
      f"{'随机 中位':>12}{'随机 区间':>22}{'首版单种子':>12}{'单种子分位':>12}")
rows = []
for ev_nm, ev in EVENTS.items():
    for slots in (10, 20, 30, 50):
        for weight, wn in (("eq", "等额"), ("inv_vol", "inv_vol")):
            det, det_dd = run_pf(ev, slots, weight, "small", SEED)
            anns, dds = [], []
            for s in range(N_SEED):
                a, dd = run_pf(ev, slots, weight, "random", SEED + s)
                anns.append(a); dds.append(dd)
            anns = np.array(anns)
            one = anns[0]                       # 首版用的就是 seed=SEED 这一个
            pct = float((anns <= one).mean())
            rows.append({"买点": ev_nm, "仓位": slots, "权重": wn,
                         "小市值年化": det, "小市值回撤": det_dd,
                         "随机中位": float(np.median(anns)),
                         "随机min": float(anns.min()), "随机max": float(anns.max()),
                         "随机回撤中位": float(np.median(dds)),
                         "首版单种子": one, "单种子分位": pct})
            print(f"{ev_nm:<10}{slots:>6}{wn:>10}{det:>14.2%}"
                  f"{np.median(anns):>12.2%}   [{anns.min():+.2%}, {anns.max():+.2%}]"
                  f"{one:>12.2%}{pct:>12.0%}   ({time.time()-t0:.0f}s)")

R = pd.DataFrame(rows)
R.to_csv(f"{SP}/portfolio_layer_seeds.csv", index=False)

print(f"\n{'='*118}\n判定(随机格按中位数,不按最好的种子)\n{'='*118}")
ok_d = R[(R["小市值年化"] >= 0.1188) & (R["小市值回撤"] >= -0.474)]
ok_r = R[(R["随机中位"] >= 0.1188) & (R["随机回撤中位"] >= -0.474)]
print(f"  小市值排序 满足①②的格子:**{len(ok_d)} 个**")
print(f"  随机排序(按中位数) 满足①②的格子:**{len(ok_r)} 个**")
r = R[(R["买点"] == "60日新高") & (R["仓位"] == 20) & (R["权重"] == "inv_vol")].iloc[0]
print(f"\n  首版那个唯一「通过」的格子(60日新高/20仓/inv_vol/随机 +12.27%):")
print(f"    20 个种子:中位 **{r['随机中位']:+.2%}**   区间 [{r['随机min']:+.2%}, {r['随机max']:+.2%}]")
print(f"    首版那个种子排在第 **{r['单种子分位']:.0%}** 分位 → "
      f"{'**是抽签抽出来的,不是配置好**' if r['随机中位'] < 0.1188 else '中位数也过线,站得住'}")

print(f"\n  ── 仓位数有没有系统性改善(这是本测试的原假设)──")
for ev_nm in EVENTS:
    for weight in ("等额", "inv_vol"):
        vals = [R[(R["买点"] == ev_nm) & (R["仓位"] == s) & (R["权重"] == weight)]
                ["小市值年化"].iloc[0] for s in (10, 20, 30, 50)]
        mono = "单调上升" if all(vals[i] < vals[i + 1] for i in range(3)) else "**非单调**"
        print(f"    {ev_nm:<10}{weight:<8}小市值: " +
              "  ".join(f"{s}仓 {v:+.2%}" for s, v in zip((10, 20, 30, 50), vals)) +
              f"   {mono}")

print(f"\n  ── inv_vol 相对等额(4×2=8 个对照,看是否一致)──")
win = 0
for ev_nm in EVENTS:
    for slots in (10, 20, 30, 50):
        a = R[(R["买点"] == ev_nm) & (R["仓位"] == slots) & (R["权重"] == "等额")].iloc[0]
        b = R[(R["买点"] == ev_nm) & (R["仓位"] == slots) & (R["权重"] == "inv_vol")].iloc[0]
        d = b["小市值年化"] - a["小市值年化"]
        win += d > 0
        print(f"    {ev_nm:<10}{slots:>3}仓  等额 {a['小市值年化']:>+8.2%}  "
              f"inv_vol {b['小市值年化']:>+8.2%}   差 {d*100:>+6.2f}pp"
              f"   回撤 {a['小市值回撤']:>7.1%} → {b['小市值回撤']:>7.1%}")
print(f"    **inv_vol 胜出 {win}/8**")

print(f"\n耗时 {time.time()-t0:.0f}s   Saved: portfolio_layer_seeds.csv")
