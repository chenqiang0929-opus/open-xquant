"""陶博士「年度金股」规则的全市场检验(第六十六节第三部分)

═══ 规则来自作者原文,不是我重建的 ═══
《深刻反思过去十三年的年度金股(20171223)》原话:
  「这是一种一年 365 天满仓的策略,**不择时**的策略,
    **每年 1 月 1 日集体更换一次股票**,每只股票是**等权重**配置的。
    截至 2017 年 12 月 9 日,十三年前的一元钱,增长到 **18.49 元**,
    年均复合增长 **25.16%**。」
  「以后的年度金股必须坚持 RPS 大于 90 的基本原则,
    **120 日和 250 日的 RPS 至少要有一个是大于 90 的**。」

═══ 只能测机械的那一半,这一点必须说清楚 ═══
他的规则里**可机械化的只有「RPS>90 的候选池」**;
从池子里挑哪 10 只是他的**基本面判断**,无法复制。
所以本检验测的是:**这个池子本身值不值钱**,
而不是「陶博士选股水平如何」。**两者不能混为一谈。**

从池中取 10 只用**随机 200 种子**,报中位与区间 ——
这恰好也给出「一个没有选股能力的人照着这条规则做,会得到什么」。

═══ 事前锁定(不搜索、不调参) ═══
  换股日     每年第一个交易日,用**前一交易日收盘**算 RPS(无前视)
  RPS 定义   N 日收益在当日全市场的百分位 ×100(仅在市且有完整 N 日历史者参与)
  入池条件   RPS120 > 90 **或** RPS250 > 90
  持仓       10 只等权,持有到下一年第一个交易日,**期间不调仓、不止损、不择时**
  成本       每次换股单边 0.3%
  退市/停牌  前向填充最后有效价(与全研究引擎一致)
  种子       200
  区间       2014-01 ~ 2026-08(需要 250 日前置历史,故 2013 年不能作为建仓年)

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 全区间年化 ≥ **25.16%**(他公开声明的十三年成绩)
  ② 逐年 vs 全市场等权:**胜 ≥ 2/3 的年份**
  ③ RPS>90 池随机 10 只 vs **全市场随机 10 只**(同样 200 种子):p < 0.05
  ④ 复现性核对:2016、2017 两年是否如他所说「严重拖了后腿」
  任一不过 = 该条不成立,原样写出。

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
COST = 0.003
NPICK, NSEED, SEED = 10, 200, 20260813
TARGET_ANN = 0.2516

t0 = time.time()
cl = {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    cl[k] = pd.to_numeric(pd.read_parquet(f, columns=["close"])["close"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

CLa = CL.to_numpy(float)
CLf = pd.DataFrame(CLa).ffill().to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)

# 每年第一个交易日
years = sorted(set(idx.year))
first_td = {}
for y in years:
    w = np.flatnonzero(idx.year == y)
    if len(w):
        first_td[y] = int(w[0])

REB = [y for y in years if y >= 2014 and first_td[y] - 250 >= 0]
print(f"建仓年份 {REB[0]} ~ {REB[-1]}  共 {len(REB)} 次换股")


def rps_pool(t_sel):
    """用 t_sel-1 收盘计算 RPS,返回入池布尔数组(无前视)。"""
    tp = t_sel - 1
    out = np.zeros(NS, bool)
    for N in (120, 250):
        if tp - N < 0:
            continue
        a, b = CLa[tp - N], CLa[tp]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & ALIVE[tp]
        r = np.where(ok, b / np.where(a > 0, a, np.nan) - 1, np.nan)
        v = r[ok]
        if v.size < 50:
            continue
        thr = np.nanquantile(v, 0.90)          # RPS>90 = 收益率前 10%
        out |= ok & (r > thr)
    return out


rng = np.random.default_rng(SEED)
seg = []          # 每年:(年, 池大小, 池等权收益, 池随机10只 200种子, 全市场随机10只 200种子, 全市场等权)
for k, y in enumerate(REB):
    t_in = first_td[y]
    t_out = first_td[REB[k + 1]] if k + 1 < len(REB) else NT - 1
    pool = rps_pool(t_in)
    tradable = ALIVE[t_in]
    p_idx = np.flatnonzero(pool & tradable)
    a_idx = np.flatnonzero(tradable)
    ret = np.where(np.isfinite(CLa[t_in]) & (CLa[t_in] > 0),
                   CLf[t_out] / CLa[t_in] - 1, np.nan)

    def draw(pool_idx):
        if len(pool_idx) < NPICK:
            return np.full(NSEED, np.nan)
        out = np.empty(NSEED)
        for s in range(NSEED):
            sel = rng.choice(pool_idx, NPICK, replace=False)
            v = ret[sel]
            out[s] = np.nanmean(v) - COST if np.isfinite(v).any() else np.nan
        return out

    seg.append({
        "年": y, "池大小": len(p_idx),
        "池等权": np.nanmean(ret[p_idx]) - COST if len(p_idx) else np.nan,
        "池随机10": draw(p_idx), "全市场随机10": draw(a_idx),
        "全市场等权": np.nanmean(ret[a_idx]) - COST,
        "天数": t_out - t_in})
    s = seg[-1]
    print(f"  {y}  池 {s['池大小']:>4}  池等权 {s['池等权']:>+7.1%}  "
          f"池随机10中位 {np.nanmedian(s['池随机10']):>+7.1%}  "
          f"全市场等权 {s['全市场等权']:>+7.1%}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n{'='*104}\n逐年对照\n{'='*104}")
print(f"{'年':<6}{'池大小':>7}{'池等权':>10}{'池随机10中位':>14}{'全市场随机10中位':>18}"
      f"{'全市场等权':>12}{'池胜全市场':>12}")
win = 0
for s in seg:
    w = np.nanmedian(s["池随机10"]) > s["全市场等权"]
    win += bool(w)
    print(f"{s['年']:<6}{s['池大小']:>7}{s['池等权']:>+10.1%}"
          f"{np.nanmedian(s['池随机10']):>+14.1%}{np.nanmedian(s['全市场随机10']):>+18.1%}"
          f"{s['全市场等权']:>+12.1%}{('✓' if w else '✗'):>12}")

# 复利
yrs = sum(s["天数"] for s in seg) / 243.0


def ann(path):
    v = 1.0
    for x in path:
        v *= (1 + (0 if not np.isfinite(x) else x))
    return v ** (1 / yrs) - 1, v


a_pool_eq, v1 = ann([s["池等权"] for s in seg])
a_mkt_eq, v2 = ann([s["全市场等权"] for s in seg])
paths_pool = np.array([[np.nan if not np.isfinite(x) else x for x in s["池随机10"]] for s in seg])
paths_mkt = np.array([[np.nan if not np.isfinite(x) else x for x in s["全市场随机10"]] for s in seg])
ann_pool = np.array([ann(paths_pool[:, i])[0] for i in range(NSEED)])
ann_mkt = np.array([ann(paths_mkt[:, i])[0] for i in range(NSEED)])

print(f"\n{'='*104}\n全区间复利({yrs:.1f} 年)\n{'='*104}")
print(f"  RPS>90 池 等权全持     年化 {a_pool_eq:>+7.2%}   累计 {v1:>6.2f} 倍")
print(f"  RPS>90 池 随机10只     年化中位 {np.median(ann_pool):>+7.2%}   "
      f"区间 [{ann_pool.min():+.2%}, {ann_pool.max():+.2%}]")
print(f"  全市场   随机10只     年化中位 {np.median(ann_mkt):>+7.2%}   "
      f"区间 [{ann_mkt.min():+.2%}, {ann_mkt.max():+.2%}]")
print(f"  全市场   等权全持     年化 {a_mkt_eq:>+7.2%}   累计 {v2:>6.2f} 倍")
p = float((ann_mkt >= np.median(ann_pool)).mean())
print(f"\n  RPS>90池随机10 vs 全市场随机10:**p = {p:.4f}**")

print(f"\n{'='*104}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*104}")
c1 = np.median(ann_pool) >= TARGET_ANN
c2 = win >= (2 * len(seg) / 3)
c3 = p < 0.05
print(f"  ① 年化 ≥ 25.16%(他公开声明)   {np.median(ann_pool):+.2%}   {'✓' if c1 else '✗'}")
print(f"  ② 逐年胜全市场等权 ≥ 2/3        {win}/{len(seg)}   {'✓' if c2 else '✗'}")
print(f"  ③ 优于全市场随机10只 p<0.05     p={p:.4f}   {'✓' if c3 else '✗'}")
print(f"\n  **结论:{'算发现' if (c1 and c2 and c3) else '不算发现'}**")

print("\n  ④ 复现性核对(他说 2016、2017『严重拖了后腿』):")
for s in seg:
    if s["年"] in (2016, 2017):
        print(f"     {s['年']}  池随机10中位 {np.nanmedian(s['池随机10']):+.1%}   "
              f"全市场等权 {s['全市场等权']:+.1%}   "
              f"{'落后' if np.nanmedian(s['池随机10']) < s['全市场等权'] else '领先'}")

pd.DataFrame([{"年": s["年"], "池大小": s["池大小"], "池等权": s["池等权"],
               "池随机10中位": float(np.nanmedian(s["池随机10"])),
               "全市场随机10中位": float(np.nanmedian(s["全市场随机10"])),
               "全市场等权": s["全市场等权"]} for s in seg]).to_csv(
    f"{SP}/tao_golden10.csv", index=False)
print(f"\n→ {SP}/tao_golden10.csv   ({time.time()-t0:.0f}s)")
