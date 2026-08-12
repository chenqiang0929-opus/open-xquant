"""右尾解剖:OOS 的钱到底是哪几笔赚的(第六十二节第三部分)

═══ 起因 ═══
同一节的 A 检验撞出一个反直觉结果:

  ② 三条全中              1,606 笔  胜率 20.61%  单笔净期望 +3.50%  年化 +10.37%
  A 三条全中且非涨停/高换手    670 笔  胜率 **25.37%**  +3.33%        年化 **-0.30%**
  对照 三条全中且是涨停/高换手  936 笔  胜率 17.20%  **+3.61%**       年化 +8.72%

**胜率提到全研究最高,组合年化却转负。** 被筛掉的那批胜率更低、单笔净期望更高。
「胜率」与「组合收益」在这里负相关 —— 这只可能来自右尾。
但那是**推断**,右尾从没被拆开看过。本脚本把它拆开。

═══ 必须堵住的陷阱 ═══
**「删掉最赚钱的几笔,收益当然变差」是废话,不构成证据。**
所以删右尾必须和**同数量随机剔除**对照(20 个种子,给误差棒)。
只有「删右尾的跌幅显著超出随机剔除的分布」,才说明收益确实集中在右尾。
单种子不可采信 —— 第四十一节栽过一次。

⚠️ **事后按收益剔除是有前视的。这是诊断,不是策略。**
   「剔除后的年化」不构成任何可执行结论。

═══ 事前锁定 ═══
  档位只有 1% / 5% / 10% 三档,不搜索、不加档
  随机剔除固定 20 个种子
  特征全部沿用已定死的定义,不新增、不调阈值
  **四张表必须全部报出,不得只报好看的**

═══ 锚点(任一对不上就停) ═══
  突破池 70,318 笔 / 净期望 +4.61%
  三条全中 OOS 1,606 笔 / 胜率 20.61% / 年化 +10.37%
  A 组合 OOS 670 笔 / 胜率 25.37% / 年化 -0.30%
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, SEED, N_SEED = 10, 20260812, 20
SPLIT = "2020-01-01"
TOPK = (0.01, 0.05, 0.10)          # 事前锁定,不搜索

t0 = time.time()
NEW = pd.read_parquet(f"{SP}/adaptive_events_new.parquet")
print(f"事件 {len(NEW):,}(第六十一节自适应口径)")

# ══════════ 面板(与 cross_leg_combo.py 逐字一致) ══════════
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


def trade_ret(j: int, tb: int) -> float:
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


# ══════════ 锚点1 ══════════
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
_bt = np.array([trade_ret(j, d) for j, d in zip(bc, bd)])
print(f"\n锚点1 突破池 {len(bc):,} 笔(应 70,318)、"
      f"净期望 {np.nanmean(_bt)-COST_TRADE:+.2%}(应 +4.61%)")
assert abs(len(bc) - 70318) <= 50 and abs(np.nanmean(_bt) - COST_TRADE - 0.0461) < 0.0015
print("锚点1 通过")

# ══════════ ① 特征 + 池子定义 ══════════
lu, tp = [], []
for cd, ts in zip(NEW.code.to_numpy(), NEW.t_strong.to_numpy()):
    j, ts = col_of[cd], int(ts)
    lu.append(np.nansum(LUa[max(ts - 60, 0):ts + 1, j]))
    tp.append(TURN_PCT[ts, j])
NEW = NEW.copy()
NEW["S_涨停次数"], NEW["S_换手分位"] = lu, tp
IN = NEW[NEW.date < SPLIT]
OUT = NEW[NEW.date >= SPLIT].reset_index(drop=True)
TURN_MED = float(IN.S_换手分位.median())
hot = ((OUT.S_涨停次数 >= 3) | (OUT.S_换手分位 >= TURN_MED)).to_numpy()
tri = (OUT.满足条数 == 3).to_numpy()
POOLS = {"【基线】全部OOS事件": np.ones(len(OUT), bool),
         "② 三条全中": tri,
         "A 三条全中且非涨停/高换手": tri & ~hot}
print(f"OOS {len(OUT):,} 笔;三条全中 {tri.sum():,};A 组合 {(tri & ~hot).sum():,}")


# ══════════ 组合回测(加两个容量指标,不改判定逻辑) ══════════
def run_pf(sub, lo, hi, timing=True, drop=None):
    if drop is not None:
        sub = sub[~drop]
    by_day = {}
    for cd, dp in zip(sub.code.to_numpy(), sub.dp.to_numpy()):
        by_day.setdefault(int(dp), []).append(col_of[cd])
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    n_hold, cash_frac, n_open = [], [], 0
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
                n_open += 1
        eq = cash + sum(hd["shares"] * (CLa[t, ci_] if np.isfinite(CLa[t, ci_]) else hd["last"])
                        for ci_, hd in holds.items())
        equity[t] = eq
        n_hold.append(len(holds))
        cash_frac.append(cash / eq if eq > 0 else 1.0)
    e = pd.Series(equity[lo:hi + 1], index=idx[lo:hi + 1])
    e = e[e > 0]
    if len(e) < 100:
        return dict(年化=np.nan, 回撤=np.nan, 平均持仓=np.nan, 资金利用率=np.nan, 开仓笔数=n_open)
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    return dict(年化=(e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1,
                回撤=float((e / e.cummax() - 1).min()),
                平均持仓=float(np.mean(n_hold)),
                资金利用率=float(1 - np.mean(cash_frac)),
                开仓笔数=n_open, 年数=yrs)


S0 = int(idx.searchsorted(pd.Timestamp(SPLIT)))

# ══════════ 锚点2/3 + 表4(容量) ══════════
print(f"\n{'='*104}\n表4 容量:是不是仓位没填满?\n{'='*104}")
print(f"{'池子':<28}{'事件数':>8}{'胜率':>9}{'年化':>9}{'年均开仓':>10}"
      f"{'平均持仓':>10}{'资金利用率':>11}")
base = {}
for nm, m in POOLS.items():
    sub = OUT[m]
    r = run_pf(sub, S0, NT - 1)
    r["胜率"] = (sub.trade > 0).mean()
    r["净期望"] = sub.trade.mean() - COST_TRADE
    r["事件数"] = len(sub)
    base[nm] = r
    print(f"{nm:<28}{len(sub):>8,}{r['胜率']:>9.2%}{r['年化']:>9.2%}"
          f"{r['开仓笔数']/r['年数']:>10.1f}{r['平均持仓']:>10.2f}{r['资金利用率']:>11.1%}")

a2, a3 = base["② 三条全中"], base["A 三条全中且非涨停/高换手"]
print(f"\n锚点2 三条全中:{a2['事件数']:,} 笔(应 1,606)、胜率 {a2['胜率']:.2%}(应 20.61%)、"
      f"年化 {a2['年化']:.2%}(应 +10.37%)")
print(f"锚点3 A 组合:{a3['事件数']:,} 笔(应 670)、胜率 {a3['胜率']:.2%}(应 25.37%)、"
      f"年化 {a3['年化']:.2%}(应 -0.30%)")
assert abs(a2["事件数"] - 1606) <= 5 and abs(a2["胜率"] - 0.2061) < 0.005 \
    and abs(a2["年化"] - 0.1037) < 0.01, "锚点2 对不上"
assert abs(a3["事件数"] - 670) <= 5 and abs(a3["胜率"] - 0.2537) < 0.005 \
    and abs(a3["年化"] + 0.0030) < 0.01, "锚点3 对不上"
print("锚点2/3 通过")

# ══════════ 表1 集中度(交易级) ══════════
print(f"\n{'='*104}\n表1 集中度:前 k% 的单笔贡献了多少总收益(交易级)\n{'='*104}")
print(f"{'池子':<28}{'有效笔数':>9}{'top-1%':>9}{'top-5%':>9}{'top-10%':>10}"
      f"{'其余90%':>10}{'最大单笔':>10}")
conc = []
for nm, m in POOLS.items():
    v = OUT.loc[m, "trade"].dropna().to_numpy()
    tot = v.sum()
    srt = np.sort(v)[::-1]
    row = {"池子": nm, "有效笔数": len(v), "总收益和": tot, "最大单笔": srt[0]}
    for k in TOPK:
        n = max(1, int(round(k * len(v))))
        row[f"top{int(k*100)}%占比"] = srt[:n].sum() / tot if tot != 0 else np.nan
    n10 = max(1, int(round(0.10 * len(v))))
    row["其余90%占比"] = srt[n10:].sum() / tot if tot != 0 else np.nan
    conc.append(row)
    print(f"{nm:<28}{len(v):>9,}{row['top1%占比']:>9.1%}{row['top5%占比']:>9.1%}"
          f"{row['top10%占比']:>10.1%}{row['其余90%占比']:>10.1%}{srt[0]:>10.1%}")
    chk = row["top10%占比"] + row["其余90%占比"]
    assert abs(chk - 1.0) < 1e-6, f"自洽检查失败:{chk}"
print("\n  自洽检查:top-10% 占比 + 其余90% 占比 = 100%  ✓")

# ══════════ 表2 组合级敏感性:删右尾 vs 删随机(20 种子) ══════════
print(f"\n{'='*104}\n表2 组合级:剔除 top-k% 后年化怎么变 —— 并列同数量随机剔除(20 种子)"
      f"\n{'='*104}")
print("⚠️ 事后按收益剔除有前视,这是诊断不是策略。")
print(f"{'池子':<28}{'档':>6}{'原年化':>9}{'删右尾':>9}{'删随机中位':>11}"
      f"{'随机5-95%':>18}{'判定':>8}")
sens = []
for nm, m in POOLS.items():
    sub = OUT[m].reset_index(drop=True)
    v = sub.trade.to_numpy()
    a0 = base[nm]["年化"]
    order = np.argsort(np.where(np.isfinite(v), v, -np.inf))[::-1]
    for k in TOPK:
        n = max(1, int(round(k * np.isfinite(v).sum())))
        drop_top = np.zeros(len(sub), bool)
        drop_top[order[:n]] = True
        a_top = run_pf(sub, S0, NT - 1, drop=drop_top)["年化"]
        rr = np.random.default_rng(SEED)
        rnd = []
        for _ in range(N_SEED):
            d = np.zeros(len(sub), bool)
            d[rr.choice(len(sub), size=n, replace=False)] = True
            rnd.append(run_pf(sub, S0, NT - 1, drop=d)["年化"])
        rnd = np.array(rnd)
        lo_, hi_ = np.nanpercentile(rnd, [5, 95])
        out_of = a_top < lo_
        sens.append({"池子": nm, "档": k, "原年化": a0, "删右尾年化": a_top,
                     "删随机中位": np.nanmedian(rnd), "随机5%": lo_, "随机95%": hi_,
                     "超出随机区间": bool(out_of)})
        print(f"{nm:<28}{int(k*100):>5}%{a0:>9.2%}{a_top:>9.2%}{np.nanmedian(rnd):>11.2%}"
              f"{f'[{lo_:+.2%}, {hi_:+.2%}]':>18}{'**是**' if out_of else '否':>8}")
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

# ══════════ 表3 右尾画像 ══════════
print(f"\n{'='*104}\n表3 右尾画像:top-5% 那批长什么样(在「三条全中」池内)\n{'='*104}")
sub = OUT[tri].reset_index(drop=True)
v = sub.trade.to_numpy()
n5 = max(1, int(round(0.05 * np.isfinite(v).sum())))
order = np.argsort(np.where(np.isfinite(v), v, -np.inf))[::-1]
is_top = np.zeros(len(sub), bool)
is_top[order[:n5]] = True
sub["_hot"] = ((sub.S_涨停次数 >= 3) | (sub.S_换手分位 >= TURN_MED))
FEATS = ["S_涨停次数", "S_换手分位", "深度", "缩量比", "收敛比", "调整天数", "trade", "raw252"]
print(f"{'指标':<14}{'top5% 中位':>13}{'其余 中位':>12}{'倍数':>8}")
port = []
for c in FEATS:
    a, b = sub.loc[is_top, c].median(), sub.loc[~is_top, c].median()
    port.append({"指标": c, "top5%中位": a, "其余中位": b})
    print(f"{c:<14}{a:>13.3f}{b:>12.3f}{(a/b if b else np.nan):>8.2f}")
p_hot_top = sub.loc[is_top, "_hot"].mean()
p_hot_rest = sub.loc[~is_top, "_hot"].mean()
print(f"\n  **涨停/高换手驱动占比**:top-5% **{p_hot_top:.1%}**   其余 {p_hot_rest:.1%}"
      f"   (A 组合把这批全剔掉了)")
print(f"  top-5% 共 {n5} 笔,其中被 A 规则剔除的 **{int(sub.loc[is_top,'_hot'].sum())} 笔**")

pd.DataFrame(conc).to_csv(f"{SP}/right_tail_concentration.csv", index=False)
pd.DataFrame(sens).to_csv(f"{SP}/right_tail_anatomy.csv", index=False)
pd.DataFrame(port).assign(top5_hot率=p_hot_top, 其余_hot率=p_hot_rest) \
    .to_csv(f"{SP}/right_tail_portrait.csv", index=False)
print(f"\n→ right_tail_anatomy.csv / _concentration.csv / _portrait.csv"
      f"   ({time.time()-t0:.0f}s)")
