"""把三个调整期特征搬到事件基数大得多的池子上

═══ 为什么做这个 ═══
第五十九节:三个调整期特征(缩量、波动收敛、浅回调)在 OOS 上把胜率
从 15.85% 提到 **22.88%**、净期望 +1.73% → **+6.68%**、回撤 -63.7% → **-26.1%**。
**但组合级年化只有 +5.08%(p=0.31),没过判据。**

原因诊断得很清楚:**「三条全中」只剩 389 笔 / 6.6 年 ≈ 59 笔/年**,
而组合有 10 个仓位、持有期最长 252 天 —— **大部分时间空着仓,年化被稀释**。
不是过滤器无效,是**过滤器与组合容量不匹配**。

→ 把同样三个过滤器搬到事件基数大 5~10 倍的池子上,让过滤后仍能填满仓位。

  口袋支点        64,641 笔(§56:后段 +3.44%,p=0.0000,唯一在2015后仍显著的信号)
  60日新高突破    70,318 笔(锚点池)

═══ 一个必须交代的定义调整 ═══
原三个特征以「强势日」为调整期起点,但这两个池子里**没有强势日这个锚**。
改用通用定义:**调整期起点 = 事件日之前 250 日内的最高价那一根**
(与第五十四节基底检测器的「左沿」同一思路):

  深度     = 1 − min(low[peak..t-1]) / high[peak]
  缩量比   = 均量(peak..t-1) ÷ 均量(peak前60日)
  波动收缩 = 均TR(最近20日) ÷ 均TR(peak前60日)

**这是必要的适配,不是为了让结果好看** —— 三个量的量纲与原定义一致,
所以 0.8 这个阈值原样沿用,不重新拟合。深度阈值仍取**选择集中位数**。

═══ 事前判据(与第五十九节相同,不放宽) ═══
选择集 = 2020-01-01 之前(只用来定深度中位数);验证集 = 2020-01-01 之后。
  ① OOS 组合年化 ≥ **+7.22%**
  ② **300次**同选中率随机对照(从同池同期事件抽),p < 0.05/4 = **0.0125**
两条都要过。**验证集上不调任何参数。**

**锚点**:60日新高池 = 70,310 笔 / 组合全期 +6.34%。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, SEED, N_RAND = 10, 20260812, 300
SPLIT = "2020-01-01"
PEAK_WIN = 250

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
OPa, HIa, LOa, CLa = (OP.to_numpy(float), HI.to_numpy(float),
                      LO.to_numpy(float), CL.to_numpy(float))
MVa, VOa, MA50a = MV.to_numpy(float), VO.to_numpy(float), MA50.to_numpy(float)
TRa = np.maximum(HIa - LOa, np.maximum(np.abs(HIa - np.roll(CLa, 1, 0)),
                                       np.abs(LOa - np.roll(CLa, 1, 0))))
codes = list(OP.columns)
col_of = {cd: i for i, cd in enumerate(codes)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv, vo

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()

FWD_WIN = 252
LAST_OK = NT - 1 - FWD_WIN
_rmax60 = CL.rolling(60, min_periods=60).max()
_rmin60 = CL.rolling(60, min_periods=60).min()
BASE_OK = (((_rmax60 - _rmin60) / _rmin60.replace(0, np.nan)).shift(1) < 0.50).to_numpy()
BRK60 = (CLa > _rmax60.shift(1).to_numpy()) & BASE_OK
prev_c = CL.shift(1)
dn_vol = VO.where(CL < prev_c, 0.0)
PP = ((CL > prev_c) & (VO > dn_vol.rolling(10, min_periods=5).max().shift(1))
      & (CL > (HI + LO) / 2) & (CL > MA50) & (MA50 > MA50.shift(10))
      & ((CL / MA50 - 1) <= 0.10)).to_numpy()


def to_events(hit, gap=60):
    cs, ds = [], []
    for j, cd in enumerate(codes):
        last = -10**9
        for q in np.flatnonzero(hit[:, j]):
            if q - last < gap or q == 0 or q > LAST_OK:
                continue
            last = q
            cs.append(cd); ds.append(int(q))
    return pd.DataFrame({"code": cs, "dp": ds})


def add_metrics(ev):
    """调整期起点 = 事件日之前 250 日内最高价那一根(通用定义,不需要强势日锚)。"""
    dep, shr, atr = [], [], []
    for cd, dp in zip(ev.code.to_numpy(), ev.dp.to_numpy()):
        j = col_of[cd]
        t = int(dp)
        lo_i = max(t - PEAK_WIN, 0)
        seg_h = HIa[lo_i:t, j]
        if seg_h.size < 40 or np.all(~np.isfinite(seg_h)):
            dep.append(np.nan); shr.append(np.nan); atr.append(np.nan)
            continue
        pk = lo_i + int(np.nanargmax(seg_h))
        hi_pk = HIa[pk, j]
        lows = LOa[pk:t, j]
        lows = lows[np.isfinite(lows)]
        d = 1 - lows.min() / hi_pk if lows.size and np.isfinite(hi_pk) and hi_pk > 0 else np.nan
        v_adj = VOa[pk:t, j]; v_adj = v_adj[np.isfinite(v_adj)]
        v_pre = VOa[max(pk - 60, 0):pk, j]; v_pre = v_pre[np.isfinite(v_pre)]
        s = (v_adj.mean() / v_pre.mean()) if (v_adj.size and v_pre.size and v_pre.mean() > 0) else np.nan
        tr_now = TRa[max(t - 20, 0):t, j]; tr_now = tr_now[np.isfinite(tr_now)]
        tr_pre = TRa[max(pk - 60, 0):pk, j]; tr_pre = tr_pre[np.isfinite(tr_pre)]
        a = (tr_now.mean() / tr_pre.mean()) if (tr_now.size and tr_pre.size and tr_pre.mean() > 0) else np.nan
        dep.append(d); shr.append(s); atr.append(a)
    ev = ev.copy()
    ev["深度"], ev["缩量比"], ev["波动收缩"] = dep, shr, atr
    ev["date"] = idx[ev.dp.to_numpy()]
    return ev


def run_pf(evs, lo, hi):
    by_day = {}
    for cd, dp in zip(evs.code.to_numpy(), evs.dp.to_numpy()):
        by_day.setdefault(int(dp), []).append(cd)
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    for t in range(lo, hi + 1):
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
                elif t - hd["t_in"] >= 252:
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
                             "stop_px": px * 0.90, "shares": alloc * (1 - COST_PF) / px}
                cash -= alloc
        equity[t] = cash + sum(
            hd["shares"] * (CLa[t, hd["ci"]] if np.isfinite(CLa[t, hd["ci"]]) else hd["last"])
            for hd in holds.values())
    eq = pd.Series(equity[lo:hi + 1], index=idx[lo:hi + 1])
    eq = eq[eq > 0]
    if len(eq) < 100:
        return np.nan, np.nan
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
            float((eq / eq.cummax() - 1).min()))


def trade_ret(evs):
    out = []
    for code, grp in evs.groupby("code", sort=False):
        ci = col_of[code]
        op, ll, cc = OPa[:, ci], LOa[:, ci], CLa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = int(dp) + 1
            entry = op[e] if e < NT else np.nan
            if not np.isfinite(entry) or entry <= 0:
                continue
            stop_px, last, ex = entry * 0.90, entry, None
            end = min(e + 252, NT - 1)
            for t in range(e, end + 1):
                if not np.isfinite(cc[t]):
                    continue
                last = cc[t]
                if np.isfinite(ll[t]) and ll[t] <= stop_px:
                    ex = op[t] if (np.isfinite(op[t]) and op[t] < stop_px) else stop_px
                    break
            if ex is None:
                ex = cc[end] if np.isfinite(cc[end]) else last
            if np.isfinite(ex) and ex > 0:
                out.append(ex / entry - 1)
    return np.array(out)


POOLS = {"60日新高突破": to_events(BRK60), "口袋支点": to_events(PP)}
_a, _ = run_pf(POOLS["60日新高突破"], 200, NT - 1)
print(f"\n锚点:60日新高 {len(POOLS['60日新高突破']):,} 笔、组合全期 {_a:+.2%}(应 +6.34%)")
assert abs(len(POOLS["60日新高突破"]) - 70310) <= 50 and abs(_a - 0.0634) < 0.004
print("锚点通过")

S0 = idx.searchsorted(pd.Timestamp(SPLIT))
ALPHA = 0.05 / 4
all_rows = []
for pname, ev in POOLS.items():
    ev = add_metrics(ev)
    IN = ev[ev.date < SPLIT]
    OUT = ev[ev.date >= SPLIT].reset_index(drop=True)
    THR_D = float(np.nanmedian(IN["深度"]))
    print(f"\n{'='*118}\n池:{pname}   全池 {len(ev):,} 笔   OOS {len(OUT):,} 笔"
          f"   深度阈值(选择集中位) {THR_D:.3f}\n{'='*118}")
    F = {
        "【基线】不筛": np.ones(len(OUT), bool),
        "缩量 <0.8×": (OUT["缩量比"] < 0.8).fillna(False).to_numpy(),
        "波动收敛 <0.8×": (OUT["波动收缩"] < 0.8).fillna(False).to_numpy(),
        "浅回调 ≤选择集中位": (OUT["深度"] <= THR_D).fillna(False).to_numpy(),
        "**三条全中**": ((OUT["缩量比"] < 0.8) & (OUT["波动收缩"] < 0.8)
                     & (OUT["深度"] <= THR_D)).fillna(False).to_numpy(),
    }
    print(f"{'配置':<24}{'事件数':>9}{'选中率':>8}{'年均笔数':>10}{'胜率':>9}"
          f"{'净期望':>10}{'年化':>10}{'最大回撤':>10}")
    res = {}
    for nm, m in F.items():
        sub = OUT[m]
        if len(sub) < 30:
            print(f"{nm:<24}{len(sub):>9,}   样本不足")
            continue
        r = trade_ret(sub)
        a, dd = run_pf(sub, S0, NT - 1)
        yrs = (idx[NT - 1] - idx[S0]).days / 365.25
        res[nm] = {"池": pname, "配置": nm, "事件数": len(sub), "选中率": m.mean(),
                   "年均笔数": len(sub) / yrs, "胜率": (r > 0).mean(),
                   "净期望": r.mean() - COST_TRADE, "年化": a, "回撤": dd}
        v = res[nm]
        print(f"{nm:<24}{len(sub):>9,}{m.mean():>8.1%}{v['年均笔数']:>10.0f}{v['胜率']:>9.2%}"
              f"{v['净期望']:>+10.2%}{a:>+10.2%}{dd:>10.1%}   ({time.time()-t0:.0f}s)")
    for nm, m in F.items():
        if nm.startswith("【基线】") or nm not in res:
            continue
        k = res[nm]["事件数"]
        rng = np.random.default_rng(SEED)
        anns = np.array([run_pf(OUT.iloc[rng.choice(len(OUT), k, replace=False)], S0, NT - 1)[0]
                         for _ in range(N_RAND)])
        anns = anns[np.isfinite(anns)]
        p = float((anns >= res[nm]["年化"]).mean())
        res[nm]["p"] = p
        q = np.quantile(anns, [0.025, 0.975])
        print(f"  随机对照 {nm}(抽 {k:,}) 实际 {res[nm]['年化']:+.2%}  中位 {np.median(anns):+.2%}"
              f"  [{q[0]:+.2%}, {q[1]:+.2%}]  **p={p:.4f}**   ({time.time()-t0:.0f}s)")
    print(f"\n  判定(①年化≥+7.22% ②p<{ALPHA:.4f},验证集未调参):")
    for nm, v in res.items():
        if nm.startswith("【基线】"):
            continue
        c1, c2 = v["年化"] >= 0.0722, v.get("p", 1) < ALPHA
        print(f"    {nm:<24} 年化 {v['年化']:+.2%} {'✓' if c1 else '✗'}   "
              f"p={v.get('p', np.nan):.4f} {'✓' if c2 else '✗'}"
              f"   **{'算发现' if (c1 and c2) else '不算发现'}**")
    all_rows.extend(res.values())

R = pd.DataFrame(all_rows)
R.to_csv(f"{SP}/consolidation_transfer.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: consolidation_transfer.csv")
