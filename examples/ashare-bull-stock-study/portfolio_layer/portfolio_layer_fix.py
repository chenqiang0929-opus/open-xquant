"""测试2:组合层修复 —— 「六次交易级赚、组合级不赚」到底是不是 10 仓位造成的

═══ 症状 ═══
本 session 已经**六次**出现同一件事:
  §41 条件止盈 交易级 +4.61%→+6.12%/笔,组合级只多 0.81pp
  §54 基底过滤 交易级 +1.02%→+2.97%(p=0.000),组合级 p=0.250
  §55 H4 MA20  交易级 +4.61%→+5.52%,组合级 +6.34%→+2.45%
  ……
六次同一个症状,不是巧合。

═══ 我在第五十四节写过原因,但一直没去解决 ═══
> 组合只有 **10 个仓位**,而候选事件有 17,826 笔。
> 真正决定收益的是**哪 10 个被挑中**,不是候选池整体质量提高了多少。

**信号质量提升 3 倍,被 10 个仓位的抽样噪音全部吃掉。**
而组合层的三个参数(仓位数、权重、候选排序)从第四十一节起**一次都没动过**:
永远是 10 仓位、等额、小市值优先。

═══ 外部证据 ═══
DeepSeek 的 P 章顺手测了一下组合层就发现 **inv_vol 加权是免费午餐**:
Sharpe 0.830→0.878、回撤 -21.8%→**-14.8%**、年化只掉 0.2pp。
他们没做仓位数敏感性。

═══ 测试维度 ═══
事件用**口袋支点(间隔60日)**(§55 里唯一两个层次都 p=0.000 击败随机的买点)
和 60日新高基线做对照。
  仓位数  {10, 20, 30, 50}
  权重    {等额, inv_vol}
  候选排序 {小市值优先, 随机}
离场规则一律用规则 A(-10%固定止损、无止盈、252日),**不动**。
共 2买点 × 4仓位 × 2权重 × 2排序 = 32 格。

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 组合年化 ≥ **+11.88%**(全市场等权月度基准)
  ② 最大回撤优于 **-47.4%**(§55 里最好的那个)
  ③ 随机排序那一列必须一起报 —— 若「小市值优先」的优势在仓位数变大后消失,
     说明前 55 节的组合级结果有一部分是选股排序的运气

**锚点**:60日新高 / 10仓位 / 等额 / 小市值优先 = 组合年化 **+6.34%**。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_PF, SEED = 0.003, 20260810
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
VOL20 = CL.pct_change().rolling(20, min_periods=10).std()
idx = OP.index
NT = len(idx)
OPa, HIa, LOa = OP.to_numpy(float), HI.to_numpy(float), LO.to_numpy(float)
CLa, MVa = CL.to_numpy(float), MV.to_numpy(float)
MA50a, VOL20a = MA50.to_numpy(float), VOL20.to_numpy(float)
col_of = {cd: i for i, cd in enumerate(OP.columns)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv, vo

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()

# ══════════ 两组事件(与 §55 完全同口径) ══════════
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
for k, v in EVENTS.items():
    print(f"  {k}: {len(v):,} 笔")


# ══════════ 组合引擎(离场逻辑与 §55 完全一致,只把仓位/权重/排序参数化) ══════════
def run_pf(evs, slots, weight, pick, seed=SEED):
    by_day = {d: g["code"].tolist() for d, g in evs.groupby("dp")}
    rng = np.random.default_rng(seed)
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    n_tr, trs, start = 0, [], 200
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
                if np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
                    ex = op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
                elif t - hd["t_in"] >= RULE_A["max_hold"]:
                    ex = cl_t
            if ex is not None and np.isfinite(ex) and ex > 0:
                cash += hd["shares"] * ex * (1 - COST_PF)
                trs.append(ex / hd["entry"] - 1)
                del holds[code]
                n_tr += 1
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
            # 原引擎是**顺序**分配 `cash/(slots-len(holds))`,推导下来每笔恰好 = cash/free,
            # 所以一批 k 个候选的总预算 = cash × k/free。等额时与原引擎逐分逐厘一致
            # (锚点靠这个成立),inv_vol 只改这笔预算在批内的分法。
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
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    return {"年化": ann, "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "最大回撤": (eq / eq.cummax() - 1).min(), "年均笔数": n_tr / yrs,
            "笔均收益": float(np.mean(trs)) if trs else np.nan}


print(f"\n{'='*118}\n32 格:买点 × 仓位数 × 权重 × 候选排序(离场规则一律规则 A,不动)\n{'='*118}")
print(f"{'买点':<10}{'仓位':>6}{'权重':>10}{'排序':>10}{'年化':>10}{'Sharpe':>9}"
      f"{'最大回撤':>10}{'年均笔数':>10}{'笔均收益':>10}")
rows = []
for ev_nm, ev in EVENTS.items():
    for slots in (10, 20, 30, 50):
        for weight, wn in (("eq", "等额"), ("inv_vol", "inv_vol")):
            for pick, pn in (("small", "小市值"), ("random", "随机")):
                s = run_pf(ev, slots, weight, pick)
                rows.append({"买点": ev_nm, "仓位": slots, "权重": wn, "排序": pn, **s})
                print(f"{ev_nm:<10}{slots:>6}{wn:>10}{pn:>10}{s['年化']:>+10.2%}"
                      f"{s['Sharpe']:>9.3f}{s['最大回撤']:>10.1%}{s['年均笔数']:>10.0f}"
                      f"{s['笔均收益']:>+10.2%}   ({time.time()-t0:.0f}s)")
                if ev_nm == "60日新高" and slots == 10 and weight == "eq" and pick == "small":
                    print(f"    锚点核对:年化 {s['年化']:+.2%}(应 +6.34%)")
                    assert abs(s["年化"] - 0.0634) < 0.004, f"锚点对不上:{s['年化']:+.4%}"
                    print("    锚点通过 —— 组合引擎与 §55 一致,后面只动组合层参数")

R = pd.DataFrame(rows)
R.to_csv(f"{SP}/portfolio_layer_fix.csv", index=False)

print(f"\n{'='*118}\n判定(事前判据,未放宽)\n{'='*118}")
ok = R[(R["年化"] >= 0.1188) & (R["最大回撤"] >= -0.474)]
print(f"  同时满足 ①年化≥+11.88% 与 ②回撤优于 -47.4% 的配置:**{len(ok)} 个**")
for _, r in ok.iterrows():
    print(f"    {r['买点']} / {r['仓位']}仓 / {r['权重']} / {r['排序']}"
          f"   年化 {r['年化']:+.2%}  回撤 {r['最大回撤']:.1%}  Sharpe {r['Sharpe']:.3f}")
best = R.loc[R["年化"].idxmax()]
print(f"\n  年化最高:{best['买点']} / {best['仓位']}仓 / {best['权重']} / {best['排序']}"
      f"  **{best['年化']:+.2%}**  回撤 {best['最大回撤']:.1%}")

print(f"\n  ── 仓位数的边际效应(口袋支点/等额/小市值)──")
sub = R[(R["买点"] == "口袋支点") & (R["权重"] == "等额") & (R["排序"] == "小市值")]
for _, r in sub.sort_values("仓位").iterrows():
    print(f"    {r['仓位']:>3} 仓   年化 {r['年化']:>+8.2%}   Sharpe {r['Sharpe']:>6.3f}"
          f"   回撤 {r['最大回撤']:>7.1%}")

print(f"\n  ── 「小市值优先」相对「随机排序」的优势,随仓位数如何变化 ──")
for ev_nm in EVENTS:
    for slots in (10, 20, 30, 50):
        a = R[(R["买点"] == ev_nm) & (R["仓位"] == slots) & (R["权重"] == "等额")
              & (R["排序"] == "小市值")]["年化"].iloc[0]
        b = R[(R["买点"] == ev_nm) & (R["仓位"] == slots) & (R["权重"] == "等额")
              & (R["排序"] == "随机")]["年化"].iloc[0]
        print(f"    {ev_nm:<10}{slots:>3} 仓   小市值 {a:>+8.2%}   随机 {b:>+8.2%}"
              f"   差 {a-b:>+7.2f}pp".replace("pp", "pp") if False else
              f"    {ev_nm:<10}{slots:>3} 仓   小市值 {a:>+8.2%}   随机 {b:>+8.2%}"
              f"   差 {(a-b)*100:>+6.2f}pp")

print(f"\n  ── inv_vol 是不是「免费午餐」(对照 DeepSeek P 章)──")
for ev_nm in EVENTS:
    for slots in (10, 30):
        a = R[(R["买点"] == ev_nm) & (R["仓位"] == slots) & (R["权重"] == "等额")
              & (R["排序"] == "小市值")].iloc[0]
        b = R[(R["买点"] == ev_nm) & (R["仓位"] == slots) & (R["权重"] == "inv_vol")
              & (R["排序"] == "小市值")].iloc[0]
        print(f"    {ev_nm:<10}{slots:>3} 仓   等额 {a['年化']:>+8.2%}/{a['Sharpe']:.3f}"
              f"/{a['最大回撤']:>7.1%}   inv_vol {b['年化']:>+8.2%}/{b['Sharpe']:.3f}"
              f"/{b['最大回撤']:>7.1%}")

print(f"\n耗时 {time.time()-t0:.0f}s   Saved: portfolio_layer_fix.csv")
