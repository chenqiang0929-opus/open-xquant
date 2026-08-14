"""第六十四节:N(新高密度)与 L(同期群内龙头度)

═══ 为什么只测这两条 ═══
用户把 CANSLIM 各要素对应到三段论上。拆成可检验条目后,
**十条里有八条本 session 已经测过,全部不显著**:
  C 当季同比 lift 1.07 p=0.194 / A 双增长 1.14、盈利加速 1.13(天花板 1.18,没过)
  S 强势期放量 lift **0.85** p=0.006(负向) / I 换手分位 lift **0.84**(负向)
  M 大盘 事件级 lift 0.92 p=0.288 / RPS 提升 = 事件定义本身
  均线多头排列 §51 不过 / 第二段横盘 §59-61 OOS +10.37% 但 p=0.16
  **板块效应 §62 板块共振 lift 1.00 —— 与同日对照股一模一样**

**只有两条真正没测过,而且恰好是用户最强调的:**

  **N** —— §55 测过「买点当天是否 52周新高」(**单点**);
  用户说的是「**能不能一直新高**」= **新高密度**,是**路径属性**。
  前 63 节所有特征都是单点快照,这是本 session 第一个路径特征。

  **L** —— §62 测的板块共振是「同时启动的有多少只」(**广度**,lift 1.00 无效);
  用户说的龙头是「**在同一批启动的股票里它排第几**」= **群内排名**,完全不同的量。

用户这段话最有价值的不是提出新因子,而是指出前面一直在**错误的层面**上测量。

═══ 三个特征(事前锁定,不搜索) ═══
  N1 启动前新高密度:t_strong 前 250 日里,收盘创 250 日新高的天数占比
  N2 买点前新高密度:买点日前 60 日里,收盘创 250 日新高的天数占比
  L1 同期群内龙头度:同期群 = [t_strong-20, t_strong+20] 内 RPS60 上穿 90 的股票;
                    龙头度 = 该股按「t_strong 前 250 日涨幅」在群内的分位
  第四格 = N1 ∩ N2 ∩ L1 三条全中

二值化阈值一律取**选择集(2014-2019)中位数**,验证集不重算。
锁定:回看 250 日、买点前窗口 60 日、群窗口 ±20 日、二值化取中位数。**一个都不调。**

═══ 事前判据(与 §59/§61/§62/§63 相同,不放宽) ═══
  第一关 三条纪律(选择集):自身零分布 500 次 p<0.05 / lift > 公平天花板(命中≥300)
                          / 2015-05 前后同向
  第二关 策略级(验证集 2020-2026,一个参数不动):
         组合级年化 ≥ +7.22%,且 300 次同日随机对照 p < 0.05/4 = 0.0125

**必须报出随机对照中位数** —— 第六十三节刚测出:同日随机买一只在市股票、
同样的离场规则,年化中位就有 **5.31%~6.46%**。任何新特征必须显著超过这条线。

═══ 先验(事前写下,免得事后自我安慰) ═══
本 session 已有 13 个连续的事前判据检验全部未通过,N/L 大概率也过不了。
值得跑的理由是结构性的:第一个路径特征、第一个同期群相对位置。
**事前声明:N/L 方向只此一轮,不过就写「不算发现」,不回头调窗口、不换二值化。**

═══ 锚点 ═══
  突破池 70,318 笔 / 净期望 +4.61%
  §61 三条全中 OOS 1,606 笔 / 胜率 20.61% / 年化 +10.37%
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST, SLOTS, SEED = 0.003, 10, 20260812
N_PERM, N_RAND = 500, 300
SPLIT = "2020-01-01"
HH_LOOK, N2_WIN, GRP_WIN = 250, 60, 20      # 事前锁定
ALPHA = 0.05 / 4

t0 = time.time()
NEW = pd.read_parquet(f"{SP}/adaptive_events_new.parquet")
print(f"事件 {len(NEW):,}(第六十一节口径)", flush=True)

cols = ["open", "high", "low", "close", "float_mv"]
acc = {c: {} for c in cols}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=cols)
    if x.empty:
        continue
    for c in cols:
        acc[c][k] = pd.to_numeric(x[c], errors="coerce")
OP = pd.DataFrame(acc["open"]).sort_index()
OP.index = OP.index.tz_localize(None)
F = {c: pd.DataFrame(acc[c]).set_axis(OP.index) for c in cols}
for c in ("open", "high", "low", "close"):
    F[c] = F[c].where(F[c] > 0)
idx = F["close"].index
NT = len(idx)
OPa, LOa, CLa = F["open"].to_numpy(float), F["low"].to_numpy(float), F["close"].to_numpy(float)
MVa = F["float_mv"].to_numpy(float)
codes = list(F["close"].columns)
NC = len(codes)
col_of = {cd: i for i, cd in enumerate(codes)}
_m = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                   errors="coerce")
_m.index = _m.index.tz_localize(None)
mkt = _m.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()
ALIVE = np.isfinite(CLa)
print(f"面板 {F['close'].shape}  ({time.time()-t0:.0f}s)", flush=True)
del acc

# ══════════ 锚点1:交易级突破池 ══════════
CL = F["close"]
_rmax = CL.rolling(60, min_periods=60).max()
_rmin = CL.rolling(60, min_periods=60).min()
BRK = (CLa > _rmax.shift(1).to_numpy()) & \
      (((_rmax - _rmin) / _rmin.replace(0, np.nan)).shift(1) < 0.50).to_numpy()
bc, bd = [], []
for j in range(NC):
    last = -10**9
    for q in np.flatnonzero(BRK[:, j]):
        if q - last < 60 or q == 0 or q > NT - 1 - 252:
            continue
        last = q
        bc.append(j); bd.append(int(q))


def anchor_trade(j, tb):
    e = tb + 1
    if e >= NT or not np.isfinite(OPa[e, j]) or OPa[e, j] <= 0:
        return np.nan
    entry, stop, last, ex = OPa[e, j], OPa[e, j] * 0.9, OPa[e, j], None
    end = min(e + 252, NT - 1)
    for t in range(e, end + 1):
        if not np.isfinite(CLa[t, j]):
            continue
        last = CLa[t, j]
        if np.isfinite(LOa[t, j]) and LOa[t, j] <= stop:
            ex = OPa[t, j] if (np.isfinite(OPa[t, j]) and OPa[t, j] < stop) else stop
            break
    return (ex if ex is not None else (CLa[end, j] if np.isfinite(CLa[end, j]) else last)) / entry - 1


_a = np.array([anchor_trade(j, d) for j, d in zip(bc, bd)])
print(f"\n锚点1 突破池 {len(bc):,} 笔(应 70,318)、净期望 {np.nanmean(_a)-COST:+.2%}(应 +4.61%)")
assert abs(len(bc) - 70318) <= 50 and abs(np.nanmean(_a) - COST - 0.0461) < 0.0015
print("锚点1 通过", flush=True)

# ══════════ N:新高密度(向量化) ══════════
# 「今天收盘 = 过去 250 日最高收盘」→ 是否创 250 日新高
HH = (CL >= CL.rolling(HH_LOOK, min_periods=HH_LOOK).max()) & CL.notna()
HHa = np.array(HH.to_numpy(float), copy=True)   # to_numpy 可能返回只读视图
HHa[~np.isfinite(CLa)] = np.nan
# 累积和用于任意窗口密度(shift(1) 保证只用当日及之前)
_cs = np.nancumsum(np.nan_to_num(HHa, nan=0.0), axis=0)
_cn = np.nancumsum(np.isfinite(CLa).astype(float), axis=0)


def hh_density(j: int, t: int, win: int) -> float:
    """[t-win+1, t] 窗口内创 250 日新高的天数 ÷ 该窗口内的有效交易日数。"""
    a = max(0, t - win + 1)
    hi = _cs[t, j] - (_cs[a - 1, j] if a > 0 else 0.0)
    n = _cn[t, j] - (_cn[a - 1, j] if a > 0 else 0.0)
    return hi / n if n > 0 else np.nan


# ══════════ L:同期群内龙头度 ══════════
RPS60 = (CL.pct_change(60).rank(axis=1, pct=True) * 100).to_numpy(float)
CROSS = (RPS60 > 90) & (np.roll(RPS60, 1, axis=0) <= 90)
CROSS[0] = False
RET250 = (CL / CL.shift(HH_LOOK) - 1).to_numpy(float)

n1, n2, l1, gsz = [], [], [], []
for cd, ts, tb in zip(NEW.code.to_numpy(), NEW.t_strong.to_numpy(), NEW.dp.to_numpy()):
    j, ts, tb = col_of[cd], int(ts), int(tb)
    n1.append(hh_density(j, ts, HH_LOOK))
    n2.append(hh_density(j, tb, N2_WIN))
    a, b = max(0, ts - GRP_WIN), min(NT - 1, ts + GRP_WIN)
    peers = np.unique(np.nonzero(CROSS[a:b + 1])[1])
    v = RET250[ts, peers]
    ok = np.isfinite(v)
    if ok.sum() >= 2 and np.isfinite(RET250[ts, j]):
        l1.append(float((v[ok] < RET250[ts, j]).mean()))
        gsz.append(int(ok.sum()))
    else:
        l1.append(np.nan); gsz.append(0)
D = NEW.copy()
D["N1_启动前新高密度"], D["N2_买点前新高密度"] = n1, n2
D["L1_群内龙头度"], D["群规模"] = l1, gsz
print(f"\nN/L 特征就绪  ({time.time()-t0:.0f}s)")
for c in ("N1_启动前新高密度", "N2_买点前新高密度", "L1_群内龙头度"):
    v = D[c].dropna()
    assert v.between(0, 1).all(), f"{c} 越界:[{v.min()}, {v.max()}]"
    print(f"  {c:<22} 有效 {len(v):>6,}  中位 {v.median():.3f}  "
          f"[{v.min():.3f}, {v.max():.3f}]  ✓ 取值域自检通过")
print(f"  群规模 中位 {D.群规模.median():.0f}  (≥2 才计入)")

# ══════════ 第一关:三条纪律(选择集 2014-2019) ══════════
IN = D[D.date < SPLIT].reset_index(drop=True)
OUT = D[D.date >= SPLIT].reset_index(drop=True)
b = (IN.trade > 0).to_numpy()
BASE = b.mean()
# ── 修复:首轮 N1/N2 用「≥ 选择集中位数」,而两者的选择集中位数都是 **0** ──
# 新高密度恒 ≥ 0,所以那个掩码选中了 **100% 的事件**,lift 必然 =1.00 ——
# 那是构造出来的,不是测出来的,首轮检验作废(不是「不算发现」)。
# 修复只改二值化:计数型特征用**零/非零**这个唯一的自然切点(不是搜出来的分位),
# 也正好贴合用户原话「能不能创新高」。L1 的中位数切法本来就正常(选中 41%),不动。
L1_THR = float(IN["L1_群内龙头度"].median())
print(f"\n{'='*104}\n第一关 三条纪律(选择集 {len(IN):,} 笔,基准交易胜率 {BASE:.2%})\n{'='*104}")
print(f"  二值化(修复后):N1 > 0、N2 > 0(创过至少一次250日新高);"
      f"L1 ≥ {L1_THR:.3f}(选择集中位数,未改)")


def masks(df):
    m1 = (df.N1_启动前新高密度 > 0).to_numpy()
    m2 = (df.N2_买点前新高密度 > 0).to_numpy()
    m3 = (df.L1_群内龙头度 >= L1_THR).to_numpy()
    return {"N1 启动前创过新高": m1, "N2 买点前创过新高": m2,
            "L1 群内龙头度 高50%": m3, "**N1∩N2∩L1 三条全中**": m1 & m2 & m3}


# 退化自检:任一掩码选中率 >90% 或 <5% 则该特征无法二值化,事前声明不再换第三种切法
_deg = {nm: m.mean() for nm, m in masks(IN).items()}
print("  选中率自检:" + "  ".join(f"{nm.split()[0]}={v:.1%}" for nm, v in _deg.items()))
for nm, v in _deg.items():
    if v > 0.90 or v < 0.05:
        print(f"  ⚠️ **{nm} 选中率 {v:.1%} 仍然退化 —— 该特征在本事件集上无法二值化**")


yr = IN.year.to_numpy()
rng = np.random.default_rng(SEED)
perms = np.empty((N_PERM, len(b)), bool)
for k in range(N_PERM):
    bb = b.copy()
    for yv in np.unique(yr):
        s = yr == yv
        bb[s] = rng.permutation(bb[s])
    perms[k] = bb
early = (IN.date < "2019-01-01").to_numpy()
print(f"\n{'特征':<26}{'命中':>8}{'P(赚钱|特征)':>13}{'lift':>8}{'p':>9}"
      f"{'早':>7}{'晚':>7}{'同向':>6}")
nulls, res = {}, {}
for nm, m in masks(IN).items():
    m = m & np.isfinite(IN.trade.to_numpy())
    if m.sum() < 100:
        continue
    lf = b[m].mean() / BASE
    nl = perms[:, m].mean(axis=1) / BASE
    nulls[nm] = nl
    p = float((np.abs(nl - 1) >= abs(lf - 1)).mean())
    e_ = b[m & early].mean() / b[early].mean() if (m & early).sum() >= 30 else np.nan
    l_ = b[m & ~early].mean() / b[~early].mean() if (m & ~early).sum() >= 30 else np.nan
    same = np.isfinite(e_) and np.isfinite(l_) and (e_ - 1) * (l_ - 1) > 0
    res[nm] = {"命中": int(m.sum()), "lift": lf, "p": p, "同向": same}
    print(f"{nm:<26}{int(m.sum()):>8,}{b[m].mean():>13.2%}{lf:>8.2f}{p:>9.4f}"
          f"{e_:>7.2f}{l_:>7.2f}{'✓' if same else '✗':>6}")
big = [n for n in res if res[n]["命中"] >= 300]
q95 = (float(np.quantile(np.vstack([nulls[n] for n in big]).max(axis=0), 0.95))
       if len(big) >= 2 else np.nan)
print(f"\n  公平 best-of-{len(big)} 噪音上界 **{q95:.2f}**")
n_disc = 0
for nm, v in res.items():
    ok = v["p"] < 0.05 and v["同向"] and np.isfinite(q95) and v["lift"] > q95
    n_disc += ok
    print(f"    {nm:<26} 三条纪律 {'**✓ 全过**' if ok else '✗'}")
print(f"  第一关通过的:**{n_disc} 个**", flush=True)


# ══════════ 组合回测 ══════════
def run_pf(ev_j, ev_t, lo, hi):
    by_day = {}
    for j, t in zip(ev_j, ev_t):
        by_day.setdefault(int(t), []).append(int(j))
    cash, holds = 1.0, {}
    eq = np.zeros(NT)
    for t in range(lo, hi + 1):
        for j in list(holds):
            hd = holds[j]
            op_t, lo_t, cl_t = OPa[t, j], LOa[t, j], CLa[t, j]
            ex = None
            if not np.isfinite(cl_t):
                ex = hd["last"]
            else:
                hd["last"] = cl_t
                if np.isfinite(lo_t) and lo_t <= hd["stop"]:
                    ex = op_t if (np.isfinite(op_t) and op_t < hd["stop"]) else hd["stop"]
                elif t - hd["t_in"] >= 252:
                    ex = cl_t
            if ex is not None and np.isfinite(ex) and ex > 0:
                cash += hd["shares"] * ex * (1 - COST)
                del holds[j]
        cands = [j for j in by_day.get(t - 1, [])
                 if j not in holds and np.isfinite(OPa[t, j]) and OPa[t, j] > 0]
        if cands and len(holds) < SLOTS and mkt_ok[t]:
            cands.sort(key=lambda j: MVa[t, j] if np.isfinite(MVa[t, j]) else np.inf)
            for j in cands[:SLOTS - len(holds)]:
                alloc = cash / max(1, SLOTS - len(holds))
                if alloc <= 0:
                    break
                px = OPa[t, j]
                holds[j] = {"t_in": t, "last": px, "stop": px * 0.90,
                            "shares": alloc * (1 - COST) / px}
                cash -= alloc
        eq[t] = cash + sum(hd["shares"] * (CLa[t, j] if np.isfinite(CLa[t, j]) else hd["last"])
                           for j, hd in holds.items())
    e = pd.Series(eq[lo:hi + 1], index=idx[lo:hi + 1])
    e = e[e > 0]
    if len(e) < 100:
        return np.nan, np.nan
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1, float((e / e.cummax() - 1).min())


S0 = int(idx.searchsorted(pd.Timestamp(SPLIT)))
OJ = np.array([col_of[c] for c in OUT.code])
OT = OUT.dp.to_numpy()

# 锚点2:§61 三条全中必须复现
tri = (OUT.满足条数 == 3).to_numpy()
a2, _ = run_pf(OJ[tri], OT[tri], S0, NT - 1)
print(f"\n锚点2 §61 三条全中:{int(tri.sum()):,} 笔(应 1,606)、"
      f"胜率 {(OUT.trade[tri] > 0).mean():.2%}(应 20.61%)、年化 {a2:.2%}(应 +10.37%)")
assert abs(tri.sum() - 1606) <= 5 and abs((OUT.trade[tri] > 0).mean() - 0.2061) < 0.005 \
    and abs(a2 - 0.1037) < 0.01, "锚点2 对不上"
print("锚点2 通过", flush=True)

# ══════════ 第二关:OOS + 300 次同日随机对照 ══════════
print(f"\n{'='*112}\n第二关 OOS 验证(2020-2026,判据:年化 ≥ +7.22% 且 p < {ALPHA})\n{'='*112}")
pool_flat, pool_off, pool_sz = [], {}, {}
for t in np.unique(OT):
    t = int(t)
    p_ = np.flatnonzero(ALIVE[t] & np.isfinite(OPa[t]) & (OPa[t] > 0))
    pool_off[t] = len(pool_flat); pool_sz[t] = len(p_)
    pool_flat.extend(p_.tolist())
pool_flat = np.asarray(pool_flat, dtype=np.int32)

base_a, base_dd = run_pf(OJ, OT, S0, NT - 1)
print(f"{'配置':<28}{'事件':>7}{'选中率':>8}{'胜率':>9}{'净期望':>9}"
      f"{'年化':>9}{'回撤':>9}{'随机中位':>10}{'p':>9}")
print(f"{'【基线】全部OOS事件':<28}{len(OUT):>7,}{1.0:>8.1%}"
      f"{(OUT.trade > 0).mean():>9.2%}{OUT.trade.mean()-COST:>9.2%}"
      f"{base_a:>9.2%}{base_dd:>9.1%}{'—':>10}{'—':>9}")
rows = []
for nm, m in masks(OUT).items():
    if m.sum() < 30:
        continue
    sub = OUT[m]
    a, dd = run_pf(OJ[m], OT[m], S0, NT - 1)
    et = OT[m]
    off_e = np.array([pool_off[int(t)] for t in et], dtype=np.int64)
    sz_e = np.array([pool_sz[int(t)] for t in et], dtype=np.int64)
    ok_e = sz_e > 0
    draws = np.empty(N_RAND)
    for k in range(N_RAND):
        pick = off_e + (rng.random(len(et)) * np.maximum(sz_e, 1)).astype(np.int64)
        draws[k], _ = run_pf(pool_flat[np.where(ok_e, pick, off_e)], et, S0, NT - 1)
    p = float((draws >= a).mean())
    med = float(np.nanmedian(draws))
    rows.append({"配置": nm, "事件": len(sub), "选中率": m.mean(),
                 "胜率": (sub.trade > 0).mean(), "净期望": sub.trade.mean() - COST,
                 "年化": a, "回撤": dd, "随机中位": med, "p": p})
    print(f"{nm:<28}{len(sub):>7,}{m.mean():>8.1%}{(sub.trade>0).mean():>9.2%}"
          f"{sub.trade.mean()-COST:>9.2%}{a:>9.2%}{dd:>9.1%}{med:>10.2%}{p:>9.4f}"
          f"  {'✓' if p < ALPHA else '✗'}", flush=True)
R = pd.DataFrame(rows)
R["过年化"] = R.年化 >= 0.0722
R["过p"] = R.p < ALPHA
R["算发现"] = R.过年化 & R.过p

print(f"\n  **随机对照中位数 {R.随机中位.min():.2%} ~ {R.随机中位.max():.2%}** "
      f"(第六十三节同一口径是 5.31%~6.46%)")
print(f"\n{'='*104}\n事前判据 vs 实际\n{'='*104}")
print(R[["配置", "事件", "年化", "p", "过年化", "过p", "算发现"]].to_string(index=False))
n_ok = int(R.算发现.sum())
print(f"\n  第一关三条纪律通过:**{n_disc} 个**")
print(f"  第二关两条判据同时通过:**{n_ok} 格**")
print(f"  **结论:{'有发现' if n_ok else '不算发现'}**"
      f"{'' if n_ok else ' —— 事前声明 N/L 只此一轮,不回头调窗口、不换二值化'}")
R.to_csv(f"{SP}/leader_path_features.csv", index=False)
D[["code", "date", "year", "N1_启动前新高密度", "N2_买点前新高密度",
   "L1_群内龙头度", "群规模", "trade", "raw252"]].to_csv(
    f"{SP}/leader_path_raw.csv", index=False)
print(f"\n→ leader_path_features.csv / leader_path_raw.csv  ({time.time()-t0:.0f}s)")
