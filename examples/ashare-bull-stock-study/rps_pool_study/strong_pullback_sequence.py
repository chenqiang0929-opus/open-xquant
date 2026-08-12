"""用户的时序,这次按他说的实现:强势 → 深调到20周线 → RPS250抬升 → 买点

═══ 上一节(五十七)的实现是错的,先说清错在哪 ═══
第五十七节问题②的结论是「等待充分调整有害」。**那个结论作废,因为检验写错了。**

用户点名三只股票做验收:宁德时代(2019-2020)、胜宏科技、生益电子(2024-2025)。
诊断结果:

  股票        触及20周线天数   **其中收盘在MA50之上的比例**   旧检验两条件同时成立
  宁德时代          121              **40.5%**                7 天
  胜宏科技           64              **23.4%**               24 天
  生益电子          143              **18.9%**               32 天

**「深调到20周线(=日线MA100)」时,价格 60~81% 的时间在 MA50 之下。**
而口袋支点的定义里硬性要求 `收盘 > MA50`(不追高约束)。
两个条件**在结构上互斥** —— 我把它们相交,剩下的 2,250 个样本
是「刚破位又勉强爬回MA50的弱势股」,胜率 8.5%。
**那测的不是用户说的形态,是它的反面。**

═══ 这次按用户的原话实现 ═══
  ① **先强势**:RPS60 > 90                        (短期动量领先)
  ② **再深调**:最低价触及 20周线(日线MA100×1.03)   (回踩到位,此时通常在MA50之下)
  ③ **RPS250 抬升到 80+**                        (长期动量在整理中补上来)
  ④ **后买点**:三种定义各测一次,**不做网格**
     P1 重新站上 MA50(10周线)
     P2 口袋支点式量能确认(当日量 > 过去10日任一下跌日的最大量)+ 站上MA50
     P3 突破整理期高点

时序约束(事前锁定):② 必须在 ① 之后 **250 日内**;④ 必须在 ② 之后 **120 日内**。
入场 = 信号日**次日开盘**。去重 60 日。

═══ 验收测试(先过这一关,不过就不看回测数字) ═══
**检测器必须在用户点名的三只股票上、在他指出的时间窗内触发。**
检不出就是实现还有问题,回测数字一律不采信。

═══ 事前判据(与第五十七节相同,不放宽) ═══
  ① 交易级净期望 ≥ +6.0%/笔   ② 组合级年化 ≥ +7.22%
  ③ 300次随机对照 p_全期 < 0.05/3   ④ **2015-05 之后单独成立**(同样阈值)
四条缺一不可。

**锚点**:60日新高 = 70,310 笔 / +4.61%/笔 / 组合 +6.34%。
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
CUT = 575
GAP_STRONG_TO_DIP, GAP_DIP_TO_BUY, MIN_GAP = 250, 120, 60

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
MA50 = CL.rolling(50, min_periods=50).mean()       # 10周线
MA100 = CL.rolling(100, min_periods=100).mean()    # **20周线**
idx = OP.index
NT = len(idx)
OPa, HIa, LOa, CLa = (OP.to_numpy(float), HI.to_numpy(float),
                      LO.to_numpy(float), CL.to_numpy(float))
MVa, VOa = MV.to_numpy(float), VO.to_numpy(float)
MA50a, MA100a = MA50.to_numpy(float), MA100.to_numpy(float)
codes = list(OP.columns)
col_of = {cd: i for i, cd in enumerate(codes)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv, vo

_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()

RPS60 = (CL.pct_change(60).rank(axis=1, pct=True) * 100).to_numpy(float)
RPS250 = (CL.pct_change(250).rank(axis=1, pct=True) * 100).to_numpy(float)
prev_c = CL.shift(1)
DNVOL10 = VO.where(CL < prev_c, 0.0).rolling(10, min_periods=5).max().shift(1).to_numpy(float)
UP = (CLa > np.roll(CLa, 1, axis=0))
UP[0] = False
FWD_WIN = 252
LAST_OK = NT - 1 - FWD_WIN
print(f"因子就绪  ({time.time()-t0:.0f}s)")


def build_sequence(buy_variant):
    """扫描 强势→深调→抬升→买点 的完整时序。返回事件 DataFrame。"""
    rows = []
    for j, cd in enumerate(codes):
        cl, lo, hi = CLa[:, j], LOa[:, j], HIa[:, j]
        m50, m100 = MA50a[:, j], MA100a[:, j]
        strong = np.flatnonzero(np.isfinite(RPS60[:, j]) & (RPS60[:, j] > 90))
        if strong.size == 0:
            continue
        last_ev = -10**9
        i = 0
        while i < strong.size:
            t_s = int(strong[i])
            # ② 深调:t_s 之后 250 日内,最低价触及 20周线
            hi_lim = min(t_s + GAP_STRONG_TO_DIP, NT - 1)
            seg = np.arange(t_s + 1, hi_lim + 1)
            if seg.size == 0:
                i += 1
                continue
            dip_mask = (np.isfinite(m100[seg]) & np.isfinite(lo[seg])
                        & (lo[seg] <= m100[seg] * 1.03)
                        & (m100[seg] > np.roll(m100, 20)[seg]))   # 20周线本身仍向上
            if not dip_mask.any():
                i += 1
                continue
            t_d = int(seg[np.argmax(dip_mask)])
            # ④ 买点:t_d 之后 120 日内
            hi2 = min(t_d + GAP_DIP_TO_BUY, NT - 1)
            seg2 = np.arange(t_d + 1, hi2 + 1)
            if seg2.size == 0:
                i += 1
                continue
            back50 = (np.isfinite(m50[seg2]) & np.isfinite(cl[seg2]) & (cl[seg2] > m50[seg2])
                      & (cl[seg2 - 1] <= m50[seg2 - 1]))          # 重新站上 MA50 那一天
            if buy_variant == "P1":
                buy_mask = back50
            elif buy_variant == "P2":
                volok = (np.isfinite(DNVOL10[seg2, j]) & (VOa[seg2, j] > DNVOL10[seg2, j])
                         & UP[seg2, j])
                buy_mask = (np.isfinite(m50[seg2]) & (cl[seg2] > m50[seg2]) & volok)
            else:                                                  # P3 突破整理期高点
                buy_mask = np.zeros(seg2.size, bool)
                base_hi = np.nanmax(hi[t_s:t_d + 1]) if t_d > t_s else np.nan
                if np.isfinite(base_hi):
                    buy_mask = np.isfinite(cl[seg2]) & (cl[seg2] > base_hi)
            # ③ RPS250 抬升到 80+
            buy_mask &= np.isfinite(RPS250[seg2, j]) & (RPS250[seg2, j] >= 80)
            if not buy_mask.any():
                i += 1
                continue
            t_b = int(seg2[np.argmax(buy_mask)])
            if t_b - last_ev >= MIN_GAP and t_b <= LAST_OK:
                last_ev = t_b
                rows.append({"code": cd, "dp": t_b, "t_strong": t_s, "t_dip": t_d})
            i = int(np.searchsorted(strong, t_b, side="right"))     # 跳过已用掉的强势日
    return pd.DataFrame(rows)


VARIANTS = {"P1 重新站上MA50": "P1", "P2 站上MA50+量能确认": "P2", "P3 突破整理期高点": "P3"}
EV = {nm: build_sequence(v) for nm, v in VARIANTS.items()}
for nm, e in EV.items():
    print(f"  {nm}: {len(e):,} 笔  ({time.time()-t0:.0f}s)")

# ══════════ 验收测试:必须在用户点名的三只股票上触发 ══════════
print(f"\n{'='*118}\n验收测试:检测器必须在用户点名的三只股票上触发(不过就不看回测数字)\n{'='*118}")
CASES = {"300750": ("宁德时代", "2019-06", "2021-01"),
         "300476": ("胜宏科技", "2025-01", "2026-07"),
         "688183": ("生益电子", "2024-06", "2026-07")}
ok_all = True
for code, (nm2, d0, d1) in CASES.items():
    print(f"\n  【{code} {nm2}】用户指出的窗口 {d0} ~ {d1}")
    for vn, e in EV.items():
        g = e[(e.code == code)]
        g = g[(idx[g.dp] >= pd.Timestamp(d0)) & (idx[g.dp] <= pd.Timestamp(d1))] if len(g) else g
        if len(g) == 0:
            print(f"    {vn:<22} **未触发**")
            ok_all = False
            continue
        for _, r in g.iterrows():
            fwd = CLa[min(int(r.dp) + 252, NT - 1), col_of[code]] / CLa[int(r.dp), col_of[code]] - 1
            print(f"    {vn:<22} 强势 {idx[int(r.t_strong)].date()} → "
                  f"触20周线 {idx[int(r.t_dip)].date()} → **买点 {idx[int(r.dp)].date()}**"
                  f"   之后252日 {fwd:+.1%}")
print(f"\n  验收 {'**通过**' if ok_all else '**未全部通过 —— 下面的回测数字仅供参考**'}")


# ══════════ 回测 ══════════
def run_trade(evs, lo=0, hi=None):
    hi = NT - 1 if hi is None else hi
    out = []
    e2 = evs[(evs.dp >= lo) & (evs.dp <= hi)]
    for code, grp in e2.groupby("code", sort=False):
        ci = col_of[code]
        op, ll, cc = OPa[:, ci], LOa[:, ci], CLa[:, ci]
        for dp in grp["dp"].to_numpy():
            e = dp + 1
            entry = op[e] if e < NT else np.nan
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


_rmax60 = CL.rolling(60, min_periods=60).max()
_rmin60 = CL.rolling(60, min_periods=60).min()
BASE_OK = (((_rmax60 - _rmin60) / _rmin60.replace(0, np.nan)).shift(1) < 0.50).to_numpy()
BRK = (CLa > _rmax60.shift(1).to_numpy()) & BASE_OK
bc, bd = [], []
for j, cd in enumerate(codes):
    last = -10**9
    for q in np.flatnonzero(BRK[:, j]):
        if q - last < 60 or q == 0 or q > LAST_OK:
            continue
        last = q
        bc.append(cd); bd.append(int(q))
BASE_EV = pd.DataFrame({"code": bc, "dp": bd})
_r = run_trade(BASE_EV)
_a, _ = run_pf(BASE_EV)
print(f"\n锚点:{len(BASE_EV):,} 笔(应 70,310)、净期望 {_r.mean()-COST_TRADE:+.2%}(应 +4.61%)、"
      f"组合 {_a:+.2%}(应 +6.34%)")
assert abs(len(BASE_EV) - 70310) <= 50 and abs(_r.mean() - COST_TRADE - 0.0461) < 0.0015
assert abs(_a - 0.0634) < 0.004
print("锚点通过")

print(f"\n{'='*118}\n强势 → 深调20周线 → RPS250≥80 → 买点\n{'='*118}")
print(f"{'配置':<26}{'事件数':>9}{'胜率':>8}{'净期望':>10}{'年化':>10}{'最大回撤':>10}"
      f"{'后段净期望':>12}{'后段年化':>11}")
rows = {}
for nm, e in list(EV.items()) + [("【对照】60日新高", BASE_EV)]:
    if len(e) < 30:
        print(f"{nm:<26}{len(e):>9,}   样本不足")
        continue
    r = run_trade(e)
    a, dd = run_pf(e)
    r_post = run_trade(e, CUT, NT - 1)
    a_post, _ = run_pf(e, CUT, NT - 1)
    rows[nm] = {"事件数": len(e), "胜率": (r > 0).mean(), "净期望": r.mean() - COST_TRADE,
                "年化": a, "最大回撤": dd,
                "后段净期望": r_post.mean() - COST_TRADE if r_post.size else np.nan,
                "后段年化": a_post}
    v = rows[nm]
    print(f"{nm:<26}{len(e):>9,}{v['胜率']:>8.1%}{v['净期望']:>+10.2%}{a:>+10.2%}{dd:>10.1%}"
          f"{v['后段净期望']:>+12.2%}{a_post:>+11.2%}   ({time.time()-t0:.0f}s)")

print(f"\n{'='*118}\n随机对照 × {N_RAND}\n{'='*118}")
cands = [n for n in rows if not n.startswith("【对照】")]
ALPHA = 0.05 / max(len(cands), 1)
print(f"  Bonferroni:{len(cands)} 个变体 → 需 **p < {ALPHA:.5f}**\n")
POST_BASE = BASE_EV[BASE_EV.dp >= CUT]
for nm in cands:
    k = rows[nm]["事件数"]
    rng = np.random.default_rng(SEED)
    full = np.array([run_pf(BASE_EV.iloc[rng.choice(len(BASE_EV), min(k, len(BASE_EV)),
                                                    replace=False)])[0] for _ in range(N_RAND)])
    kp = min(len(POST_BASE), max(int(k * 0.8), 30))
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
    print(f"  {nm}: ① {v['净期望']:+.2%} {'✓' if c1 else '✗'}  ② {v['年化']:+.2%} "
          f"{'✓' if c2 else '✗'}  ③ p={v.get('p_全期', np.nan):.4f} {'✓' if c3 else '✗'}  "
          f"④ p后={v.get('p_后段', np.nan):.4f} {'✓' if c4 else '✗'}"
          f"   **{'算发现' if (c1 and c2 and c3 and c4) else '不算发现'}**")

pd.DataFrame(rows).T.to_csv(f"{SP}/strong_pullback_sequence.csv")
for nm, e in EV.items():
    e.assign(买点日=idx[e.dp], 强势日=idx[e.t_strong], 触线日=idx[e.t_dip]).to_csv(
        f"{SP}/seq_events_{VARIANTS[nm]}.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: strong_pullback_sequence.csv")
