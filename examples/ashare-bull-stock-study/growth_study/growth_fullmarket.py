"""成长股方向的全市场检验:三路信号 × 三种离场(第六十三节)

═══ 为什么这一节和前面 62 节都不同 ═══
前 62 节测的全是「252 天上限 + -10% 固定止损」的交易系统,
**结构上就不是复利工具** —— 案例核对已证明:
  嘉益股份持有 661 日、-10% 止损被触发 13 次;中际旭创全部收益在最后三年。
这一节**去掉 252 天上限、去掉固定止损**,离场只由长周期价格规则决定。

═══ 三路信号(用户指定,全部事前锁定) ═══
  **A 基本面**:净利同比 > 50%(YTD ÷ 四期前 YTD − 1),财报日次日开盘买入
  **B 月线巨量长阳**:当月涨幅 **且** 当月累计换手 **同时**进入全市场前 10%,
      次月首个交易日开盘买入
  **C = A ∪ B**,谁先来用谁

**为什么必须有 B**:用户的两个案例证明基本面信号会严重滞后 ——
新易盛 2023 年财报同比 -18.6% / -37.5% / -43.7% **全年为负**,
股价却在 2023-03 放量涨 51.9%、之后再涨 2148%;
中际旭创 2023-03 大涨时最近一期财报只有 +39.6%,不到门槛。
**只测 A 等于只测「基本面有多滞后」,不是测成长股方向行不行。**

而 B 恰好对应第六十二节 B 层一的实测结论:把股票推上强势的
**只有换手率(lift 1.27)和涨停**,板块共振 1.00、财报邻近 1.20 都无效。

═══ 三种离场(事前锁定) ═══
  ① 不止损:持有到数据末端(或退市)
  ② 10月均线:月末收盘 < 10个月均线 → 次月首日开盘卖
  ③ 三段论启动:先回撤 ≥15% 再创持仓期新高(= 走完第二段进入第三段),
     之后才启动 10月均线止损。用户提出的判据,案例核对里与「浮盈>100%」等价。

═══ 事前判据(多重比较必须校正,不放宽) ═══
  主检验 **3 信号 × 3 离场 = 9 格**
  ① 组合级年化 ≥ **+7.22%**(全市场等权基准,沿用前 62 节)
  ② **300 次**同日随机对照 p < **0.05/9 = 0.0056**(严格 Bonferroni)
  两条同时满足才算发现。**未校正的 p 也一并报出,但不作为判据。**

  Stage2 过滤、择时 ON/OFF 只作**诊断分解**报数字,不做显著性判定 ——
  它们是买入过滤/环境开关,不是策略本身,纳入主检验会把格子数翻倍。

═══ 随机对照的构造(关键) ═══
对照 = 在**每个信号发生的同一天**,随机换一只当时在市的股票,
**时间分布与真实信号完全相同**,再用**同一套离场规则**跑。
这样才能分离「信号有没有用」与「这套长持有期口径本身有没有用」。
(与第六十二节 B 层一的同日对照股同法。)

═══ 全部事前锁定,不搜索 ═══
  净利同比门槛 50%;月涨幅/月换手分位 10%;三段论回撤门槛 15%;
  10 个月均线;同股两次信号间隔 ≥ 250 交易日;10 个仓位;双边成本 0.3%
  **一个都不调。不过就写「不算发现」。**

═══ 锚点 ═══
  突破池 70,318 笔 / 净期望 +4.61% —— 不过就停。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST, SLOTS, SEED, N_RAND = 0.003, 10, 20260812, 300
NI_THR, Q_TOP, DIP_THR, MA_M = 0.50, 0.10, 0.15, 10
MIN_GAP, DEAD_DAYS = 250, 60
ALPHA = 0.05 / 9

t0 = time.time()
cols = ["open", "high", "low", "close", "turnover", "float_mv", "net_income"]
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

# ══════════ 锚点 ══════════
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
print(f"\n锚点 突破池 {len(bc):,} 笔(应 70,318)、净期望 {np.nanmean(_a)-COST:+.2%}(应 +4.61%)")
assert abs(len(bc) - 70318) <= 50 and abs(np.nanmean(_a) - COST - 0.0461) < 0.0015
print("锚点通过\n", flush=True)

# ══════════ 月度量:10月均线、月涨幅、月换手 ══════════
mk = pd.Series(idx.year * 100 + idx.month, index=idx)
MC = CL.groupby(mk).last()                     # 月末收盘
MT = F["turnover"].groupby(mk).sum()           # 月累计换手
MR = MC / MC.shift(1) - 1                      # 月涨幅(事前锁定:月末/上月末)
MA10 = MC.rolling(MA_M, min_periods=MA_M).mean()
BELOW = (MC < MA10) & MA10.notna()             # 月末跌破 10月均线
# 信号B:月涨幅 与 月换手 同时进入当月全市场前 10%
RTOP = MR.rank(axis=1, pct=True, ascending=False) <= Q_TOP
TTOP = MT.rank(axis=1, pct=True, ascending=False) <= Q_TOP
BIG = (RTOP & TTOP & MR.notna() & MT.notna()).to_numpy()
months = list(MC.index)
mrow = {m: i for i, m in enumerate(months)}
# 每个月对应「次月首个交易日」的下标
first_td = {}
for m in months:
    w = np.flatnonzero(mk.to_numpy() > m)
    if w.size:
        first_td[m] = int(w[0])
mk_arr = mk.to_numpy()
print(f"月度量就绪:{len(months)} 个月  ({time.time()-t0:.0f}s)", flush=True)

# ══════════ 信号 A:净利同比 > 50% ══════════
NI = F["net_income"]
sigA = [[] for _ in range(NC)]
for j, cd in enumerate(codes):
    v = NI[cd]
    rd = np.flatnonzero((v.diff() != 0).to_numpy() & np.isfinite(v.to_numpy(float)))
    if rd.size < 5:
        continue
    vals = v.to_numpy(float)[rd]
    yoy = np.full(rd.size, np.nan)
    yoy[4:] = np.where(vals[:-4] > 0, vals[4:] / vals[:-4] - 1, np.nan)
    for k in np.flatnonzero(yoy > NI_THR):
        sigA[j].append(int(rd[k]) + 1)          # 次日开盘买
# ══════════ 信号 B:月线巨量长阳 ══════════
sigB = [[] for _ in range(NC)]
for mi, m in enumerate(months):
    if m not in first_td:
        continue
    t = first_td[m]
    for j in np.flatnonzero(BIG[mi]):
        sigB[j].append(t)


def dedup(lsts):
    """同股两次信号间隔 ≥ MIN_GAP(事前锁定),并落到有效开盘日。"""
    ev_j, ev_t = [], []
    for j in range(NC):
        last = -10**9
        for t in sorted(lsts[j]):
            while t < NT and not (np.isfinite(OPa[t, j]) and OPa[t, j] > 0):
                t += 1
            if t >= NT or t - last < MIN_GAP:
                continue
            last = t
            ev_j.append(j); ev_t.append(t)
    return np.array(ev_j), np.array(ev_t)


AJ, AT = dedup(sigA)
BJ, BT = dedup(sigB)
CJ, CT = dedup([sorted(set(sigA[j]) | set(sigB[j])) for j in range(NC)])
SIGS = {"A 基本面(净利同比>50%)": (AJ, AT),
        "B 月线巨量长阳(涨幅+换手同进前10%)": (BJ, BT),
        "C = A ∪ B": (CJ, CT)}
for nm, (j_, t_) in SIGS.items():
    print(f"  {nm:<34} {len(j_):>7,} 笔   "
          f"年均 {len(j_)/((idx[-1]-idx[0]).days/365.25):>6.0f}")
print(f"信号就绪  ({time.time()-t0:.0f}s)", flush=True)


# ══════════ 离场规则(全部在**月度网格**上算 + 记忆化,否则 2700 次回测跑不完) ══════════
# 定义锁定:三段论的 peak 从**买入当月的月末收盘**起算(不是买入日收盘) ——
# 这样离场只依赖 (股票, 买入月),可以记忆化。这是为可算性做的定义选择,事前写死。
MCa = MC.to_numpy(float)
BELOWa = BELOW.to_numpy()
NM = len(months)
m_of_t = np.array([mrow.get(m, -1) for m in mk_arr])          # 每个交易日 → 月下标
ft = np.array([first_td.get(m, -1) for m in months])           # 每月 → 次月首个交易日
last_valid = np.array([int(np.flatnonzero(np.isfinite(CLa[:, j]))[-1])
                       if np.isfinite(CLa[:, j]).any() else 0 for j in range(NC)])
# next_below[mi, j] = 大于 mi 的最小月份下标且当月跌破 10月均线;没有则 -1
next_below = np.full((NM, NC), -1, dtype=np.int32)
nxt = np.full(NC, -1, dtype=np.int32)
for mi in range(NM - 1, -1, -1):
    next_below[mi] = nxt
    nxt = np.where(BELOWa[mi], mi, nxt)

_cache = {}


def exit_of(j: int, t_in: int, rule: str):
    """返回 (卖出下标, 卖出价)。全部无前视。"""
    mi = m_of_t[t_in]
    key = (j, mi, rule)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    out = None
    if rule != "none" and mi >= 0:
        if rule == "ma10m":
            k = next_below[mi, j]
        else:                                   # staged:先回撤 ≥15% 再创新高才启动
            k, peak, dipped, armed = -1, MCa[mi, j], False, False
            for u in range(mi + 1, NM):
                c = MCa[u, j]
                if not np.isfinite(c):
                    continue
                if not armed:
                    if np.isfinite(peak) and peak > 0 and c / peak - 1 <= -DIP_THR:
                        dipped = True
                    if dipped and (not np.isfinite(peak) or c > peak):
                        armed = True
                    peak = c if not np.isfinite(peak) else max(peak, c)
                if armed and BELOWa[u, j]:
                    k = u
                    break
        if k >= 0 and ft[k] > t_in and np.isfinite(OPa[ft[k], j]) and OPa[ft[k], j] > 0:
            out = (int(ft[k]), float(OPa[ft[k], j]))
    if out is None:                             # 不止损 / 未触发 → 持有到最后有效价
        u = max(last_valid[j], t_in)
        out = (int(u), float(CLa[u, j]) if np.isfinite(CLa[u, j]) else float(CLa[t_in, j]))
    _cache[key] = out
    return out


def run_pf(ev_j, ev_t, rule, timing=True):
    by_day = {}
    for j, t in zip(ev_j, ev_t):
        by_day.setdefault(int(t), []).append(int(j))
    cash, holds = 1.0, {}
    eq = np.zeros(NT)
    for t in range(200, NT):
        for j in list(holds):
            hd = holds[j]
            if t >= hd["t_out"]:
                cash += hd["shares"] * hd["px_out"] * (1 - COST)
                del holds[j]
        for j in by_day.get(t, []):
            if j in holds or len(holds) >= SLOTS:
                continue
            if timing and not mkt_ok[t]:
                continue
            if not (np.isfinite(OPa[t, j]) and OPa[t, j] > 0):
                continue
            alloc = cash / max(1, SLOTS - len(holds))
            if alloc <= 0:
                continue
            to, po = exit_of(j, t, rule)
            if to <= t or not np.isfinite(po) or po <= 0:
                continue
            holds[j] = {"t_out": to, "px_out": po, "shares": alloc * (1 - COST) / OPa[t, j]}
            cash -= alloc
        eq[t] = cash + sum(hd["shares"] * (CLa[t, j] if np.isfinite(CLa[t, j]) else hd["px_out"])
                           for j, hd in holds.items())
    e = pd.Series(eq[200:], index=idx[200:])
    e = e[e > 0]
    if len(e) < 100:
        return np.nan, np.nan
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1, float((e / e.cummax() - 1).min())


RULES = {"① 不止损": "none", "② 10月均线": "ma10m", "③ 三段论启动+10月均线": "staged"}
print(f"\n{'='*112}\n主检验 3 信号 × 3 离场(判据:年化 ≥ +7.22% 且 p < {ALPHA:.4f})\n{'='*112}")
print(f"{'信号':<34}{'离场':<22}{'事件':>7}{'年化':>9}{'回撤':>9}{'年化(择时OFF)':>14}")
rows = []
for snm, (ej, et) in SIGS.items():
    for rnm, rk in RULES.items():
        a_on, dd = run_pf(ej, et, rk, True)
        a_off, _ = run_pf(ej, et, rk, False)
        rows.append({"信号": snm, "离场": rnm, "事件": len(ej), "年化": a_on,
                     "回撤": dd, "年化_择时OFF": a_off})
        print(f"{snm:<34}{rnm:<22}{len(ej):>7,}{a_on:>9.2%}{dd:>9.1%}{a_off:>14.2%}",
              flush=True)
R = pd.DataFrame(rows)

# ══════════ 300 次同日随机对照 ══════════
print(f"\n{'='*112}\n300 次同日随机对照(同一天换一只在市股票,时间分布完全相同)\n{'='*112}")
rng = np.random.default_rng(SEED)
# 把每个日期的「在市且可开盘」候选池摊平成一个大数组 + 偏移量,
# 每次抽样变成一次向量化索引,而不是逐事件调 rng.choice(慢约一千倍)。
pool_flat, pool_off, pool_sz = [], {}, {}
for t in np.unique(np.concatenate([et for _, et in SIGS.values()])):
    t = int(t)
    p_ = np.flatnonzero(ALIVE[t] & np.isfinite(OPa[t]) & (OPa[t] > 0))
    pool_off[t] = len(pool_flat)
    pool_sz[t] = len(p_)
    pool_flat.extend(p_.tolist())
pool_flat = np.asarray(pool_flat, dtype=np.int32)
print(f"  候选池摊平:{len(pool_off):,} 个日期 / {len(pool_flat):,} 个槽位  "
      f"({time.time()-t0:.0f}s)", flush=True)

pv = []
for snm, (ej, et) in SIGS.items():
    off_e = np.array([pool_off[int(t)] for t in et], dtype=np.int64)
    sz_e = np.array([pool_sz[int(t)] for t in et], dtype=np.int64)
    ok_e = sz_e > 0
    for rnm, rk in RULES.items():
        obs = float(R[(R.信号 == snm) & (R.离场 == rnm)].年化.iloc[0])
        draws = np.empty(N_RAND)
        for k in range(N_RAND):
            pick = off_e + (rng.random(len(et)) * np.maximum(sz_e, 1)).astype(np.int64)
            rj = pool_flat[np.where(ok_e, pick, off_e)]
            draws[k], _ = run_pf(rj, et, rk, True)
        p = float((draws >= obs).mean())
        pv.append(p)
        print(f"{snm:<34}{rnm:<22}观测 {obs:>8.2%}  随机中位 {np.nanmedian(draws):>8.2%}  "
              f"**p={p:.4f}**  {'✓' if p < ALPHA else '✗'}  ({time.time()-t0:.0f}s)", flush=True)
R["p"] = pv
R["过年化"] = R.年化 >= 0.0722
R["过p"] = R.p < ALPHA
R["算发现"] = R.过年化 & R.过p

print(f"\n{'='*112}\n事前判据 vs 实际\n{'='*112}")
print(R[["信号", "离场", "事件", "年化", "p", "过年化", "过p", "算发现"]].to_string(index=False))
n_ok = int(R.算发现.sum())
print(f"\n  9 格里同时过两条判据的:**{n_ok} 格**")
print(f"  **结论:{'有发现' if n_ok else '不算发现'}**"
      f"{'' if n_ok else ' —— 事前锁定全部参数,不回头搜索'}")
R.to_csv(f"{SP}/growth_fullmarket.csv", index=False)
print(f"\n→ {SP}/growth_fullmarket.csv   ({time.time()-t0:.0f}s)")
