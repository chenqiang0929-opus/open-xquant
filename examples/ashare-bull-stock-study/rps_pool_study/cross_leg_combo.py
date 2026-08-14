"""A 跨段组合 + C 择时分解(第六十二节)

═══ 为什么做 A ═══
第五十九节把三段特征逐个测了,结论是:
  ① 第一段:靠涨停/放量/高换手冲上去的,后续交易胜率**显著更低**
     (涨停≥3次 lift 0.77 p=0.002;换手分位高50% lift 0.84 p=0.000)
  ② 第二段:缩量/波动收敛/浅回调三条**正向**,是唯一有区分度的一段
  ③ 第三段:买点日几乎没有信息,最强的一条还是负的(量比≥1.5 lift 0.85)
**但从没有人把 ① 的负向和 ② 的正向叠起来测。** 这是最后一个没试过的结构性想法。

═══ 为什么 C 要改成「分解」而不是「叠加」═══
组合回测函数 `run_pf` 里本来就有 `mkt_ok[t]` 这个闸门 ——
**第五十九节的 +5.08%、第六十一节的 +10.37% 全部已经含大盘 MA200 择时。**
所以「叠加择时」不是新测试。改成:同一组事件跑择时 ON / OFF,
看那 +10.37% 里有多少来自特征、多少来自大盘闸门。
(我最初把 C 说成「叠加」,是没看清 run_pf 就下的结论,这里更正。)

═══ 事前判据(与第五十九/六十/六十一节完全相同,不放宽) ═══
组合规则**在选择集 2014-2019 上定死**,验证集 2020-2026 一个参数不动:

    A 规则 = 「② 三条全中」 AND NOT(「① 强势期涨停 ≥3次」 OR 「① 换手分位 高50%」)

  第一关 回归:① 两个特征必须复现第五十九节的 lift(0.77 / 0.84,容差 ±0.05),
              对不上说明特征算错了,后面不看
  第二关 OOS:组合级年化 ≥ **+7.22%**
  第三关 OOS:**300 次**同选中率随机对照 p < **0.0125**(0.05/4,沿用)

**事前声明:跨段组合只测这一个规则,不搜索组合方式、不试别的 ① 特征、
不调阈值。不过就写「不算发现」。**

═══ 锚点 ═══
  交易级:60日新高突破池 70,310 笔 / 净期望 +4.61%
  第六十一节 OOS:三条全中 1,606 笔 / 胜率 20.61% / 年化 +10.37%
两个都必须复现,不过就停。
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
ALPHA = 0.0125

t0 = time.time()
NEW = pd.read_parquet(f"{SP}/adaptive_events_new.parquet")
print(f"事件 {len(NEW):,}(第六十一节自适应口径)")

# ══════════ 面板 ══════════
cols = ["open", "high", "low", "close", "volume", "turnover", "float_mv", "is_limit_up"]
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
OPa, HIa, LOa, CLa = (F["open"].to_numpy(float), F["high"].to_numpy(float),
                      F["low"].to_numpy(float), F["close"].to_numpy(float))
VOa, MVa, LUa = (F["volume"].to_numpy(float), F["float_mv"].to_numpy(float),
                 F["is_limit_up"].to_numpy(float))
# 与第五十九节逐字一致:20日均换手的全市场横截面分位
TURN_PCT = (F["turnover"].rolling(20, min_periods=10).mean()
            .rank(axis=1, pct=True).to_numpy(float))
col_of = {cd: i for i, cd in enumerate(F["close"].columns)}
_m = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                   errors="coerce")
_m.index = _m.index.tz_localize(None)
mkt = _m.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()
print(f"面板 {F['close'].shape}  ({time.time()-t0:.0f}s)")
del acc

# ══════════ 锚点1:交易级突破池 ══════════
CL = F["close"]
_rmax60 = CL.rolling(60, min_periods=60).max()
_rmin60 = CL.rolling(60, min_periods=60).min()
BASE_OK = (((_rmax60 - _rmin60) / _rmin60.replace(0, np.nan)).shift(1) < 0.50).to_numpy()
BRK = (CLa > _rmax60.shift(1).to_numpy()) & BASE_OK
LAST_OK = NT - 1 - 252
bc, bd = [], []
for j in range(len(col_of)):
    last = -10**9
    for q in np.flatnonzero(BRK[:, j]):
        if q - last < 60 or q == 0 or q > LAST_OK:
            continue
        last = q
        bc.append(j); bd.append(int(q))


def trade_ret(j: int, tb: int) -> float:
    """规则A:次日开盘进,-10% 固定止损,最长 252 日。"""
    e = tb + 1
    if e >= NT:
        return np.nan
    entry = OPa[e, j]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan
    stop, last, ex = entry * 0.90, entry, None
    end = min(e + 252, NT - 1)
    for t in range(e, end + 1):
        if not np.isfinite(CLa[t, j]):
            continue
        last = CLa[t, j]
        if np.isfinite(LOa[t, j]) and LOa[t, j] <= stop:
            ex = OPa[t, j] if (np.isfinite(OPa[t, j]) and OPa[t, j] < stop) else stop
            break
    if ex is None:
        ex = CLa[end, j] if np.isfinite(CLa[end, j]) else last
    return ex / entry - 1


_bt = np.array([trade_ret(j, d) for j, d in zip(bc, bd)])
print(f"\n锚点1 突破池 {len(bc):,} 笔(应 70,310)、"
      f"净期望 {np.nanmean(_bt)-COST_TRADE:+.2%}(应 +4.61%)")
assert abs(len(bc) - 70310) <= 50, f"事件数不符:{len(bc)}"
assert abs(np.nanmean(_bt) - COST_TRADE - 0.0461) < 0.0015, "交易级锚点对不上"
print("锚点1 通过")

# ══════════ 第一段特征(逐字复用第五十九节的定义) ══════════
lu, tp = [], []
for cd, ts in zip(NEW.code.to_numpy(), NEW.t_strong.to_numpy()):
    j, ts = col_of[cd], int(ts)
    lu.append(np.nansum(LUa[max(ts - 60, 0):ts + 1, j]))
    tp.append(TURN_PCT[ts, j])
NEW = NEW.copy()
NEW["S_涨停次数"], NEW["S_换手分位"] = lu, tp
print(f"第一段特征就绪  ({time.time()-t0:.0f}s)")

IN = NEW[NEW.date < SPLIT].reset_index(drop=True)
OUT = NEW[NEW.date >= SPLIT].reset_index(drop=True)
b_in = (IN.trade > 0).to_numpy()
BASE_IN = b_in.mean()

# ══════════ 第一关:① 两个特征必须复现第五十九节的 lift ══════════
print(f"\n{'='*104}\n第一关 回归:① 特征复现第五十九节(选择集,基准胜率 {BASE_IN:.2%})\n{'='*104}")
F1 = {"① 强势期涨停 ≥3次": (IN.S_涨停次数 >= 3).to_numpy(),
      "① 换手分位 高50%": (IN.S_换手分位 >= IN.S_换手分位.median()).to_numpy()}
EXPECT = {"① 强势期涨停 ≥3次": 0.77, "① 换手分位 高50%": 0.84}
print(f"{'特征':<22}{'命中':>8}{'P(赚钱|特征)':>13}{'lift':>8}{'§59 lift':>10}{'差':>8}")
ok_all = True
for nm, m in F1.items():
    lf = b_in[m].mean() / BASE_IN
    d = abs(lf - EXPECT[nm])
    ok_all &= d <= 0.05
    print(f"{nm:<22}{int(m.sum()):>8,}{b_in[m].mean():>13.2%}{lf:>8.2f}"
          f"{EXPECT[nm]:>10.2f}{d:>8.2f}{'  ✓' if d <= 0.05 else '  **✗**'}")
assert ok_all, "第一关不过:① 特征算得和第五十九节不一致,后面的数字不可信"
print("第一关 通过 —— ① 特征与第五十九节一致")

# ══════════ 事前锁定的组合规则 ══════════
TURN_MED = float(IN.S_换手分位.median())      # 在选择集上定死,验证集不重算


def combo_mask(df):
    """A 规则:② 三条全中 且 不是靠涨停/高换手冲上来的。阈值全部来自选择集。"""
    hot = (df.S_涨停次数 >= 3) | (df.S_换手分位 >= TURN_MED)
    return ((df.满足条数 == 3) & ~hot).to_numpy()
print(f"\n组合规则锁定:满足条数==3 且 NOT(涨停≥3 或 换手分位≥{TURN_MED:.3f})")

# ══════════ 组合回测 ══════════
def run_pf(evs_code, evs_dp, lo, hi, timing: bool):
    by_day = {}
    for cd, dp in zip(evs_code, evs_dp):
        by_day.setdefault(int(dp), []).append(col_of[cd])
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    for t in range(lo, hi + 1):
        for ci in list(holds):
            hd = holds[ci]
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
                del holds[ci]
        cands = by_day.get(t - 1, [])
        free = SLOTS - len(holds)
        if cands and free > 0 and (mkt_ok[t] or not timing):
            cands = [ci for ci in cands if ci not in holds
                     and np.isfinite(OPa[t, ci]) and OPa[t, ci] > 0]
            cands.sort(key=lambda ci: MVa[t, ci] if np.isfinite(MVa[t, ci]) else np.inf)
            for ci in cands[:free]:
                alloc = cash / (SLOTS - len(holds)) if SLOTS > len(holds) else 0
                if alloc <= 0:
                    break
                px = OPa[t, ci]
                holds[ci] = {"entry": px, "t_in": t, "last": px,
                             "stop_px": px * 0.90, "shares": alloc * (1 - COST_PF) / px}
                cash -= alloc
        equity[t] = cash + sum(
            hd["shares"] * (CLa[t, ci_] if np.isfinite(CLa[t, ci_]) else hd["last"])
            for ci_, hd in holds.items())
    eq = pd.Series(equity[lo:hi + 1], index=idx[lo:hi + 1])
    eq = eq[eq > 0]
    if len(eq) < 100:
        return np.nan, np.nan
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
            float((eq / eq.cummax() - 1).min()))


S0 = int(idx.searchsorted(pd.Timestamp(SPLIT)))
CELLS = {
    "【基线】全部OOS事件": np.ones(len(OUT), bool),
    "② 三条全中(第六十一节)": (OUT.满足条数 == 3).to_numpy(),
    "**A 三条全中 且 非涨停/高换手驱动**": combo_mask(OUT),
    "【对照】三条全中 且 **是** 涨停/高换手驱动":
        ((OUT.满足条数 == 3).to_numpy() & ~combo_mask(OUT)),
}
print(f"\n{'='*112}\nOOS 验证(2020-2026,规则在 2019 年底前定死)\n{'='*112}")
print(f"{'配置':<40}{'事件数':>8}{'选中率':>8}{'胜率':>9}{'净期望':>10}"
      f"{'年化(择时ON)':>13}{'年化(OFF)':>11}{'回撤':>9}")
rows = []
for nm, m in CELLS.items():
    sub = OUT[m]
    if len(sub) < 30:
        continue
    win = (sub.trade > 0).mean()
    exp_ = sub.trade.mean() - COST_TRADE
    a_on, dd = run_pf(sub.code.to_numpy(), sub.dp.to_numpy(), S0, NT - 1, True)
    a_off, _ = run_pf(sub.code.to_numpy(), sub.dp.to_numpy(), S0, NT - 1, False)
    rows.append({"配置": nm, "事件数": len(sub), "选中率": len(sub) / len(OUT),
                 "胜率": win, "净期望": exp_, "年化_择时ON": a_on,
                 "年化_择时OFF": a_off, "回撤": dd})
    print(f"{nm:<40}{len(sub):>8,}{len(sub)/len(OUT):>8.1%}{win:>9.2%}"
          f"{exp_:>10.2%}{a_on:>13.2%}{a_off:>11.2%}{dd:>9.1%}")
R = pd.DataFrame(rows)

# ══════════ 锚点2:第六十一节的三条全中必须复现 ══════════
r61 = R[R.配置 == "② 三条全中(第六十一节)"].iloc[0]
print(f"\n锚点2 三条全中:{int(r61.事件数):,} 笔(应 1,606)、"
      f"胜率 {r61.胜率:.2%}(应 20.61%)、年化 {r61.年化_择时ON:.2%}(应 +10.37%)")
assert abs(r61.事件数 - 1606) <= 5 and abs(r61.胜率 - 0.2061) < 0.005 \
    and abs(r61.年化_择时ON - 0.1037) < 0.01, "锚点2 对不上,管道与第六十一节不一致"
print("锚点2 通过")

# ══════════ 300 次同选中率随机对照 ══════════
print(f"\n{'='*104}\n300 次同选中率随机对照(判据 p < {ALPHA})\n{'='*104}")
rng = np.random.default_rng(SEED)
pv = {}
for nm in ("**A 三条全中 且 非涨停/高换手驱动**", "② 三条全中(第六十一节)"):
    obs = float(R.loc[R.配置 == nm, "年化_择时ON"].iloc[0])
    n = int(R.loc[R.配置 == nm, "事件数"].iloc[0])
    draws = []
    for k in range(N_RAND):
        pick = rng.choice(len(OUT), size=n, replace=False)
        sub = OUT.iloc[pick]
        a, _ = run_pf(sub.code.to_numpy(), sub.dp.to_numpy(), S0, NT - 1, True)
        draws.append(a)
        if (k + 1) % 60 == 0:
            print(f"  {nm[:20]}… {k+1}/{N_RAND}  ({time.time()-t0:.0f}s)", flush=True)
    draws = np.array(draws)
    p = float((draws >= obs).mean())
    pv[nm] = p
    print(f"  {nm:<40} 观测 {obs:+.2%}  随机中位 {np.nanmedian(draws):+.2%}  "
          f"**p={p:.4f}**  {'✓' if p < ALPHA else '✗'}")

# ══════════ C:择时分解 ══════════
print(f"\n{'='*104}\nC 择时分解:那 +10.37% 里有多少是特征、多少是大盘闸门\n{'='*104}")
print(f"{'配置':<40}{'年化 择时ON':>12}{'年化 择时OFF':>13}{'择时贡献':>10}")
for _, r in R.iterrows():
    gap = (r.年化_择时ON - r.年化_择时OFF) * 100
    print(f"{r.配置:<40}{r.年化_择时ON:>12.2%}{r.年化_择时OFF:>13.2%}{gap:>8.2f}pp")

# ══════════ 判据对照 ══════════
ra = R[R.配置 == "**A 三条全中 且 非涨停/高换手驱动**"].iloc[0]
pa = pv["**A 三条全中 且 非涨停/高换手驱动**"]
print(f"\n{'='*104}\n事前判据 vs 实际\n{'='*104}")
print(f"  第一关 ① 特征复现 §59            **通过**")
print(f"  第二关 组合级年化 ≥ +7.22%       {ra.年化_择时ON:+.2%}  "
      f"{'✓' if ra.年化_择时ON >= 0.0722 else '**✗**'}")
print(f"  第三关 300次随机对照 p < {ALPHA}   {pa:.4f}  {'✓' if pa < ALPHA else '**✗**'}")
verdict = (ra.年化_择时ON >= 0.0722) and (pa < ALPHA)
print(f"\n  **结论:{'算发现' if verdict else '不算发现'}**"
      f"{'' if verdict else ' —— 事前声明只测这一个组合规则,不回头搜索'}")

R["p"] = R.配置.map(pv)
R.to_csv(f"{SP}/cross_leg_combo.csv", index=False)
print(f"\n→ {SP}/cross_leg_combo.csv   ({time.time()-t0:.0f}s)")
