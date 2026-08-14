"""幸存者偏差量化:随机重仓持有 10 年,有多少比例达到 100 倍?

═══ 要回答的问题 ═══
「世上存在若干个 10 年 100 倍的投资者」这个事实,
需不需要用「技能」来解释?

做法:在已验证的面板上,随机抽 1~3 只股票、等权、**不止损、持有 10 年**,
重复 10 万次,数有多少次达到 100 倍。再乘以「有多少人这样做过」。

═══ 三个必须堵住的陷阱 ═══
① **模拟自身不能有幸存者偏差(最致命)。**
   候选池 = 「入场当天还活着且可交易」的股票,**绝不要求它活到 10 年后**。
   退市/长期停牌按面板惯例用最后有效价结算(与全研究的引擎一致),
   不悄悄剔除。否则整个检验会自己变成它要批判的那个东西。
② **风格必须单独控住。**
   风生水起只买小盘股,不是全市场瞎抽。所以除全市场池外,
   另跑一组「入场日 float_mv 后 30%」的池子。
   两组之差 = 风格贡献;剩下的才轮得到技能。
③ **入场日期必须随机,不能固定。**
   面板 2013-01-04 ~ 2026-08-03 共 3,297 日,持有 2,430 日(≈10年),
   可用入场日只有前 867 个(2013-01 ~ 2016-07)——
   **这段窗口套着 2015 泡沫与 2015-16 崩盘,结果对它高度敏感,必须报出来。**

═══ 事前锁定(不搜索、不调参) ═══
  仓位数      1 / 2 / 3(用户指定),等权
  持有        2,430 个交易日(10 年),**不止损、不调仓、不再平衡**
  成本        入场单边 0.3%(与全研究一致);全程只买一次
  重复        每格 100,000 次
  入场价      入场日收盘价
  出场价      入场后第 2,430 个交易日的价格(前向填充,承接停牌/退市)
  随机种子    20260813

═══ 事前判据(跑之前写死) ═══
把 P(≥100倍) 乘以「实际这样做过的人数 N」:
  乘出来 ≥ 100 人  → 「存在若干个百倍投资者」**不需要技能解释**
  乘出来 < 1 人    → 技能是**必需**的
  1 ~ 100 人之间   → 运气能解释一部分,但不够
**N 取值必须事前说明理由,不能事后挑一个让结论好看的数。**

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
HOLD = 2430          # 交易日 ≈ 10 年
N_SIM = 100_000
KS = (1, 2, 3)
COST = 0.003
SEED = 20260813

t0 = time.time()
cl, mv = {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "float_mv"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
MV = pd.DataFrame(mv).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上: {(NT, NS)}"

CLa = CL.to_numpy(float)
MVa = MV.to_numpy(float)
# 前向填充:承接停牌与退市(退市按最后有效价,与全研究引擎一致)
CLf = pd.DataFrame(CLa).ffill().to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)

t_max = NT - 1 - HOLD
assert t_max > 0
print(f"可用入场日 {t_max+1:,} 个:{idx[0].date()} ~ {idx[t_max].date()}")
print(f"持有 {HOLD} 个交易日 ≈ {HOLD/243:.1f} 年")

# ── 候选池摊平(与第六十三节同法,逐次 rng.choice 慢约一千倍) ──
def build_pool(mask_fn):
    flat, off, sz = [], np.zeros(t_max + 1, np.int64), np.zeros(t_max + 1, np.int64)
    for t in range(t_max + 1):
        p = np.flatnonzero(mask_fn(t))
        off[t] = len(flat)
        sz[t] = len(p)
        flat.extend(p.tolist())
    return np.asarray(flat, np.int32), off, sz


def small_mask(t):
    a = ALIVE[t]
    m = np.where(a, MVa[t], np.nan)
    if np.all(~np.isfinite(m)):
        return a
    thr = np.nanquantile(m, 0.30)
    return a & (MVa[t] <= thr)


POOLS = {}
POOLS["全市场"] = build_pool(lambda t: ALIVE[t])
print(f"  全市场池摊平完成  ({time.time()-t0:.0f}s)", flush=True)
POOLS["小市值bottom30%"] = build_pool(small_mask)
print(f"  小市值池摊平完成  ({time.time()-t0:.0f}s)", flush=True)

rng = np.random.default_rng(SEED)
rows = []
print(f"\n{'='*104}\n随机重仓持有 10 年 · 每格 {N_SIM:,} 次\n{'='*104}")
print(f"{'池子':<18}{'仓位':>5}{'中位倍数':>10}{'均值倍数':>10}{'≥2倍':>8}{'≥5倍':>8}"
      f"{'≥10倍':>8}{'≥100倍':>9}{'亏损':>8}{'≤0.1倍':>9}")
for pname, (flat, off, sz) in POOLS.items():
    for k in KS:
        t0s = rng.integers(0, t_max + 1, N_SIM)
        okd = sz[t0s] > 0
        mult = np.zeros(N_SIM)
        for _ in range(k):
            pick = off[t0s] + (rng.random(N_SIM) * np.maximum(sz[t0s], 1)).astype(np.int64)
            j = flat[np.where(okd, pick, off[t0s])]
            p_in = CLa[t0s, j]
            p_out = CLf[t0s + HOLD, j]
            mult += np.where(np.isfinite(p_in) & (p_in > 0) & np.isfinite(p_out),
                             p_out / p_in, 1.0) / k
        mult *= (1 - COST)
        rows.append({
            "池子": pname, "仓位": k, "中位倍数": np.median(mult), "均值倍数": mult.mean(),
            "P_2x": (mult >= 2).mean(), "P_5x": (mult >= 5).mean(),
            "P_10x": (mult >= 10).mean(), "P_100x": (mult >= 100).mean(),
            "P_亏损": (mult < 1).mean(), "P_剩0.1": (mult <= 0.1).mean(),
            "最大倍数": mult.max()})
        r = rows[-1]
        print(f"{pname:<18}{k:>5}{r['中位倍数']:>10.2f}{r['均值倍数']:>10.2f}"
              f"{r['P_2x']:>8.1%}{r['P_5x']:>8.1%}{r['P_10x']:>8.2%}"
              f"{r['P_100x']:>9.4%}{r['P_亏损']:>8.1%}{r['P_剩0.1']:>9.2%}", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{SP}/survivorship_100x.csv", index=False)

# ── 入场年份敏感性(陷阱③:窗口套着 2015 泡沫) ──
print(f"\n{'='*104}\n按入场年份拆开(仓位=3,全市场池) —— 陷阱③:结果对入场窗口有多敏感\n{'='*104}")
flat, off, sz = POOLS["全市场"]
print(f"{'入场年':<10}{'可用日':>8}{'中位倍数':>10}{'≥10倍':>9}{'≥100倍':>10}{'亏损':>8}")
for yr in sorted(set(idx[:t_max + 1].year)):
    days = np.flatnonzero(idx[:t_max + 1].year == yr)
    if len(days) == 0:
        continue
    t0s = days[rng.integers(0, len(days), N_SIM)]
    okd = sz[t0s] > 0
    mult = np.zeros(N_SIM)
    for _ in range(3):
        pick = off[t0s] + (rng.random(N_SIM) * np.maximum(sz[t0s], 1)).astype(np.int64)
        j = flat[np.where(okd, pick, off[t0s])]
        p_in, p_out = CLa[t0s, j], CLf[t0s + HOLD, j]
        mult += np.where(np.isfinite(p_in) & (p_in > 0) & np.isfinite(p_out),
                         p_out / p_in, 1.0) / 3
    mult *= (1 - COST)
    print(f"{yr:<10}{len(days):>8}{np.median(mult):>10.2f}{(mult>=10).mean():>9.2%}"
          f"{(mult>=100).mean():>10.4%}{(mult<1).mean():>8.1%}", flush=True)

print(f"\n→ {SP}/survivorship_100x.csv   ({time.time()-t0:.0f}s)")
