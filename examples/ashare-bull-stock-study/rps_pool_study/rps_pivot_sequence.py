"""强势股 → 充分调整 → 口袋支点买入:欧奈尔的完整时序,第一次拼起来

═══ 用户的问题 ═══
「RPS250>90 的股票池,去等待充分调整之后,出现口袋支点再买入,收益率是否更高?」

这正是欧奈尔体系的**完整时序**,而前面 56 节从来没把三段拼起来测过:
  ① **强势**:先证明这只股票有人要(RPS250>90)
  ② **调整**:等它把浮筹洗掉(回撤 / 走出基底形态)
  ③ **买点**:等机构第一次重新出手(口袋支点)

三块积木都是现成的、而且各自验过:
  - RPS250 重建:§44 与用户快照相关 **0.990**、中位绝对差 0.07
  - 基底检测器:§54,8 个案例人工核对全部形态正确
  - 口袋支点:§56 唯一通过三条判据、且**唯一在 2015 之后仍显著**的信号

═══ 「充分调整」的三个定义(事前锁定,不做网格) ═══
  A 不等待           —— 对照组:入池即可买
  B 回撤 ≥15%        —— 距 60 日最高收盘回撤 15% 以上
  C 回撤 ≥25%        —— 更深的洗盘
  D 走出严格基底      —— §54 检测器判定为 杯柄/平底/双底 之一
入池判定用 **RPS250 在过去 60 日内曾 >90**(不是当天>90)——
因为深度调整之后 RPS 本身会掉下来,要求当天>90 等于自相矛盾。
**这一条是定义上的必然,不是为了让结果好看,写在这里备查。**

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 交易级净期望 ≥ **+6.0%/笔**(突破基线 +4.61%)
  ② 组合级年化 ≥ **+7.22%**(等权基准的保守口径)
  ③ **300 次**同数量随机对照,p < 0.05/8 = **0.00625**(8 个格子 Bonferroni)
  ④ **2015-05 之后单独成立**(p < 0.00625)—— §55 证明七种买点在后段全失效,
     只在前段成立的一律不算
四条缺一不可。**判据④是本轮新加的,比前面几节更严。**

═══ 锚点(不过就停) ═══
60日新高突破 = 70,310 笔 / +4.61%/笔 / 组合 +6.34%。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_pattern_detector import PRIOR, WIN, detect_base  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, SEED, N_RAND = 10, 20260812, 300
CUT = 575          # ≈2015-05-22,与 §54/§55/§56 同一条分界线

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
CLa, MVa, VOa = CL.to_numpy(float), MV.to_numpy(float), VO.to_numpy(float)
PMINa = CL.rolling(PRIOR, min_periods=60).min().shift(1).to_numpy(float)
col_of = {cd: i for i, cd in enumerate(OP.columns)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv, vo

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()

# ── 三段积木 ──
RPS250 = (CL.pct_change(250).rank(axis=1, pct=True) * 100)
STRONG60 = (RPS250 > 90).rolling(60, min_periods=1).max().shift(1).to_numpy() > 0   # 过去60日曾>90
DD60 = (CL / CL.rolling(60, min_periods=30).max() - 1).to_numpy(float)              # 距60日高点
prev_c = CL.shift(1)
dn_vol = VO.where(CL < prev_c, 0.0)
PP = ((CL > prev_c) & (VO > dn_vol.rolling(10, min_periods=5).max().shift(1))
      & (CL > (HI + LO) / 2) & (CL > MA50) & (MA50 > MA50.shift(10))
      & ((CL / MA50 - 1) <= 0.10)).to_numpy()
_rmax60 = CL.rolling(60, min_periods=60).max()
_rmin60 = CL.rolling(60, min_periods=60).min()
BASE_OK = (((_rmax60 - _rmin60) / _rmin60.replace(0, np.nan)).shift(1) < 0.50).to_numpy()
BRK60 = (CLa > _rmax60.shift(1).to_numpy()) & BASE_OK
FWD_WIN = 252
LAST_OK = NT - 1 - FWD_WIN
NEED = WIN + PRIOR
print(f"积木就绪:强势日占比 {np.nanmean(STRONG60):.1%}、口袋支点日占比 {PP.mean():.2%}"
      f"  ({time.time()-t0:.0f}s)")


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


# ── 「走出严格基底」:只在口袋支点日上跑检测器(全面板跑太慢且无意义) ──
def base_flag(hit):
    out = np.zeros_like(hit, bool)
    for j in range(hit.shape[1]):
        qs = np.flatnonzero(hit[:, j])
        qs = qs[qs >= NEED]
        for q in qs:
            s = q - WIN
            b = detect_base(CLa[s:q, j], HIa[s:q, j], LOa[s:q, j], VOa[s:q, j], PMINa[s:q, j])
            if b["cup"] or b["flat"] or b["dbl"]:
                out[q, j] = True
    return out


PP_STRONG = PP & STRONG60
HAS_BASE = base_flag(PP_STRONG)
print(f"基底检测完成  ({time.time()-t0:.0f}s)")

SETS = {
    "【锚点】60日新高突破": to_events(BRK60),
    "口袋支点(不加RPS)": to_events(PP),
    "A 强势 + 口袋支点(不等待)": to_events(PP_STRONG),
    "B 强势 + 回撤≥15% + 口袋支点": to_events(PP_STRONG & (DD60 <= -0.15)),
    "C 强势 + 回撤≥25% + 口袋支点": to_events(PP_STRONG & (DD60 <= -0.25)),
    "D 强势 + 严格基底 + 口袋支点": to_events(HAS_BASE),
}


def run_trade(evs, lo=0, hi=None):
    hi = NT - 1 if hi is None else hi
    out = []
    e2 = evs[(evs.dp >= lo) & (evs.dp <= hi)]
    for code, grp in e2.groupby("code", sort=False):
        ci = col_of[code]
        op, hh, ll, cc = OPa[:, ci], HIa[:, ci], LOa[:, ci], CLa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e]
            if not np.isfinite(entry) or entry <= 0:
                continue
            stop_px, last = entry * 0.90, entry
            end = min(e + 252, NT - 1)
            ex = None
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


def run_pf(evs, lo=200, hi=None):
    hi = NT - 1 if hi is None else hi
    by_day = {d: g["code"].tolist() for d, g in evs.groupby("dp")}
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


BASE_EV = SETS["【锚点】60日新高突破"]
_r = run_trade(BASE_EV)
_a, _dd = run_pf(BASE_EV)
print(f"\n锚点:{len(BASE_EV):,} 笔(应 70,310)、净期望 {_r.mean()-COST_TRADE:+.2%}(应 +4.61%)、"
      f"组合 {_a:+.2%}(应 +6.34%)")
assert abs(len(BASE_EV) - 70310) <= 50 and abs(_r.mean() - COST_TRADE - 0.0461) < 0.0015
assert abs(_a - 0.0634) < 0.004
print("锚点通过")

print(f"\n{'='*118}\n强势 → 调整 → 口袋支点(规则A:-10%止损、无止盈、252日)\n{'='*118}")
print(f"{'配置':<32}{'事件数':>9}{'胜率':>8}{'净期望':>10}{'年化':>10}{'最大回撤':>10}"
      f"{'前段净期望':>12}{'后段净期望':>12}{'后段年化':>11}")
rows = {}
for nm, ev in SETS.items():
    if len(ev) < 50:
        print(f"{nm:<32}{len(ev):>9,}   样本不足")
        continue
    r = run_trade(ev)
    a, dd = run_pf(ev)
    r_pre = run_trade(ev, 0, CUT)
    r_post = run_trade(ev, CUT, NT - 1)
    a_post, _ = run_pf(ev, CUT, NT - 1)
    rows[nm] = {"事件数": len(ev), "胜率": (r > 0).mean(), "净期望": r.mean() - COST_TRADE,
                "年化": a, "最大回撤": dd,
                "前段净期望": r_pre.mean() - COST_TRADE if r_pre.size else np.nan,
                "后段净期望": r_post.mean() - COST_TRADE if r_post.size else np.nan,
                "后段年化": a_post}
    v = rows[nm]
    print(f"{nm:<32}{len(ev):>9,}{v['胜率']:>8.1%}{v['净期望']:>+10.2%}{a:>+10.2%}"
          f"{dd:>10.1%}{v['前段净期望']:>+12.2%}{v['后段净期望']:>+12.2%}"
          f"{a_post:>+11.2%}   ({time.time()-t0:.0f}s)")

# ══════════ 判据③④:300 次随机对照,全期 + 后段 ══════════
print(f"\n{'='*118}\n随机对照 × {N_RAND}(从 60日新高事件里抽同样多的笔数)\n{'='*118}")
cands = [n for n in rows if not n.startswith("【锚点】")]
ALPHA = 0.05 / max(len(cands), 1)
print(f"  Bonferroni:{len(cands)} 个配置 → 需 **p < {ALPHA:.5f}**\n")
POST_BASE = BASE_EV[BASE_EV.dp >= CUT]
for nm in cands:
    k = rows[nm]["事件数"]
    rng = np.random.default_rng(SEED)
    full = np.array([run_pf(BASE_EV.iloc[rng.choice(len(BASE_EV), min(k, len(BASE_EV)),
                                                    replace=False)])[0]
                     for _ in range(N_RAND)])
    kp = min(len(POST_BASE), max(int(k * len(POST_BASE) / max(len(BASE_EV), 1)), 50))
    rng2 = np.random.default_rng(SEED + 1)
    post = np.array([run_pf(POST_BASE.iloc[rng2.choice(len(POST_BASE), kp, replace=False)],
                            CUT, NT - 1)[0] for _ in range(N_RAND)])
    full, post = full[np.isfinite(full)], post[np.isfinite(post)]
    p1 = float((full >= rows[nm]["年化"]).mean())
    p2 = float((post >= rows[nm]["后段年化"]).mean())
    rows[nm]["p_全期"], rows[nm]["p_后段"] = p1, p2
    q1, q2 = np.quantile(full, [0.025, 0.975]), np.quantile(post, [0.025, 0.975])
    print(f"  {nm}(抽 {k:,} 笔)")
    print(f"    全期 实际 **{rows[nm]['年化']:+.2%}**  随机中位 {np.median(full):+.2%}"
          f"  [{q1[0]:+.2%}, {q1[1]:+.2%}]  **p={p1:.4f}**")
    print(f"    后段 实际 **{rows[nm]['后段年化']:+.2%}**  随机中位 {np.median(post):+.2%}"
          f"  [{q2[0]:+.2%}, {q2[1]:+.2%}]  **p={p2:.4f}**   ({time.time()-t0:.0f}s)")

print(f"\n{'='*118}\n判定(四条判据,未放宽)\n{'='*118}")
for nm in cands:
    v = rows[nm]
    c1, c2 = v["净期望"] >= 0.060, v["年化"] >= 0.0722
    c3, c4 = v.get("p_全期", 1) < ALPHA, v.get("p_后段", 1) < ALPHA
    print(f"  {nm}:")
    print(f"    ① 净期望 {v['净期望']:+.2%} {'✓' if c1 else '✗'}   "
          f"② 年化 {v['年化']:+.2%} {'✓' if c2 else '✗'}   "
          f"③ p_全期 {v.get('p_全期', np.nan):.4f} {'✓' if c3 else '✗'}   "
          f"④ p_后段 {v.get('p_后段', np.nan):.4f} {'✓' if c4 else '✗'}")
    print(f"    **{'算发现' if (c1 and c2 and c3 and c4) else '不算发现'}**")

pd.DataFrame(rows).T.to_csv(f"{SP}/rps_pivot_sequence.csv")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: rps_pivot_sequence.csv")
