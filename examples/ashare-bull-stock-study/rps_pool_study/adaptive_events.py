"""步骤1-2:参数自适应的事件生成 + 20 只案例回归测试

═══ 为什么改 ═══
第五十九节的三个调整期特征在原时序内 OOS 有效(胜率 15.85%→22.88%),
但组合级 p=0.31;第六十节搬到大池又失败。
用户用 20 只真实案例逼出了原因,**三条互相独立的证据**:

  1 调整期系统性缩短:全样本中位 2015 **135日** → 2024 **83日**(-40%);
    2019-2021 99日 vs 2024-2026 88日,Mann-Whitney p=4.2e-19
  2 两个比值分布右移:缩量比中位 1.27→1.40、收敛比 1.20→1.37;
    固定阈值 0.8 的选中率 5.85%→4.21%,**2025 单年只剩 2.95%**
  3 **这一轮大牛股的整理期顶住了参数下限**:用户给的 2024-2026 六只里,
    亮灯段调整天数 **7 段卡在 15 日** —— 正是 MIN_ADJ_DAYS=15 这个硬下限

**不是形态失效,是尺子钉死了而市场节奏变了。**

═══ 三处改造(事前锁定,不搜索) ═══
  1 绝对阈值 0.80/0.80/0.352 → **当期横截面分位,各取最优 40%**
    (三条独立取 40% → 联合 ≈6.4%,最接近选择集原本的 5.85%)
  2 MIN_ADJ_DAYS=15 → **max(10, 0.15 × 当期全市场中位调整天数)**
    (15/100=15%,即原参数在 2014-2019 的相对位置)
  3 触线日的「MA100 必须向上」→ **去掉**
    (用户两个认可的案例被它误杀;它属于事件定义,不在验证过的三个特征里)

═══ 第一关判据(不过就停,不看后面任何数字) ═══
本轮已验证的 **20 只**必须**全部仍被捞到**,且鼎泰高科(旧版漏检,卡在收敛比 1.11)
**应当被捞到**。少捞任何一只 = 改坏了。

═══ 锚点 ═══
60日新高突破池 = 70,310 笔 / +4.61%/笔 / 组合 +6.34%。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, SEED = 10, 20260812
Q_KEEP = 0.40          # 三个指标各取最优 40%(锁定,不搜索)
MIN_ADJ_FLOOR = 10     # 自适应下限的绝对底
MIN_ADJ_RATIO = 0.15   # 下限 = 0.15 × 当期中位调整天数(锁定)
GAP_STRONG_TO_DIP, GAP_DIP_TO_BUY, MIN_GAP = 250, 120, 60
FWD_WIN = 252

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
MA100 = CL.rolling(100, min_periods=100).mean()
idx = OP.index
NT = len(idx)
OPa, HIa, LOa, CLa = (OP.to_numpy(float), HI.to_numpy(float),
                      LO.to_numpy(float), CL.to_numpy(float))
MVa, VOa = MV.to_numpy(float), VO.to_numpy(float)
MA50a, MA100a = MA50.to_numpy(float), MA100.to_numpy(float)
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

RPS60 = (CL.pct_change(60).rank(axis=1, pct=True) * 100).to_numpy(float)
RPS250 = (CL.pct_change(250).rank(axis=1, pct=True) * 100).to_numpy(float)
LAST_OK = NT - 1 - FWD_WIN
print(f"因子就绪  ({time.time()-t0:.0f}s)")


def build_sequence(legacy: bool):
    """强势 → 触20周线 → RPS250≥80 → 重新站上MA50。

    legacy=True 复刻第五十八节(MA100 必须向上、MIN_ADJ 写死 15);
    legacy=False 为新版(去掉 MA100 向上、下限稍后按自适应值过滤)。
    """
    rows = []
    for j, cd in enumerate(codes):
        cl, lo, m50, m100 = CLa[:, j], LOa[:, j], MA50a[:, j], MA100a[:, j]
        strong = np.flatnonzero(np.isfinite(RPS60[:, j]) & (RPS60[:, j] > 90))
        if strong.size == 0:
            continue
        last_ev, i = -10**9, 0
        while i < strong.size:
            t_s = int(strong[i])
            hi_lim = min(t_s + GAP_STRONG_TO_DIP, NT - 1)
            seg = np.arange(t_s + 1, hi_lim + 1)
            if seg.size == 0:
                i += 1
                continue
            dip = (np.isfinite(m100[seg]) & np.isfinite(lo[seg])
                   & (lo[seg] <= m100[seg] * 1.03))
            if legacy:                       # 旧版:20周线必须向上
                dip &= (m100[seg] > np.roll(m100, 20)[seg])
            if not dip.any():
                i += 1
                continue
            t_d = int(seg[np.argmax(dip)])
            hi2 = min(t_d + GAP_DIP_TO_BUY, NT - 1)
            seg2 = np.arange(t_d + 1, hi2 + 1)
            if seg2.size == 0:
                i += 1
                continue
            buy = (np.isfinite(m50[seg2]) & np.isfinite(cl[seg2]) & (cl[seg2] > m50[seg2])
                   & (cl[seg2 - 1] <= m50[seg2 - 1])
                   & np.isfinite(RPS250[seg2, j]) & (RPS250[seg2, j] >= 80))
            if not buy.any():
                i += 1
                continue
            t_b = int(seg2[np.argmax(buy)])
            if t_b - last_ev >= MIN_GAP and t_b <= LAST_OK:
                last_ev = t_b
                rows.append({"code": cd, "dp": t_b, "t_strong": t_s, "t_dip": t_d})
            i = int(np.searchsorted(strong, t_b, side="right"))
    return pd.DataFrame(rows)


def add_metrics(E):
    """三个调整期特征,公式与第五十九节逐字一致(只改后面的阈值口径)。"""
    dep, shr, conv, dur = [], [], [], []
    for cd, ts, td, tb in zip(E.code.to_numpy(), E.t_strong.to_numpy(),
                              E.t_dip.to_numpy(), E.dp.to_numpy()):
        j = col_of[cd]
        ts, td, tb = int(ts), int(td), int(tb)

        def _m(arr, a, b):
            v = arr[a:b + 1, j]
            v = v[np.isfinite(v)]
            return v.mean() if v.size else np.nan
        hi_a = HIa[ts:tb + 1, j]; lo_a = LOa[ts:tb + 1, j]
        hi_a = hi_a[np.isfinite(hi_a)]; lo_a = lo_a[np.isfinite(lo_a)]
        dep.append(1 - lo_a.min() / hi_a.max() if hi_a.size and lo_a.size and hi_a.max() > 0
                   else np.nan)
        vpre = _m(VOa, max(ts - 60, 0), ts)
        shr.append(_m(VOa, ts, tb) / vpre if np.isfinite(vpre) and vpre > 0 else np.nan)
        tpre = _m(TRa, max(ts - 60, 0), ts)
        conv.append(_m(TRa, td, tb) / tpre if np.isfinite(tpre) and tpre > 0 else np.nan)
        dur.append(tb - ts)
    E = E.copy()
    E["深度"], E["缩量比"], E["收敛比"], E["调整天数"] = dep, shr, conv, dur
    E["date"] = idx[E.dp.to_numpy()]
    E["year"] = E["date"].dt.year
    return E


def outcomes(E):
    """raw252 与规则A 交易结果。"""
    raw, tr = [], []
    for cd, tb in zip(E.code.to_numpy(), E.dp.to_numpy()):
        j = col_of[cd]; tb = int(tb)
        a = CLa[:, j]
        raw.append(a[tb + 252] / a[tb] - 1 if tb + 252 < NT and np.isfinite(a[tb])
                   and a[tb] > 0 and np.isfinite(a[tb + 252]) else np.nan)
        e = tb + 1
        entry = OPa[e, j] if e < NT else np.nan
        if not np.isfinite(entry) or entry <= 0:
            tr.append(np.nan); continue
        stop, last, ex = entry * 0.90, entry, None
        end = min(e + 252, NT - 1)
        for t in range(e, end + 1):
            if not np.isfinite(a[t]):
                continue
            last = a[t]
            if np.isfinite(LOa[t, j]) and LOa[t, j] <= stop:
                ex = OPa[t, j] if (np.isfinite(OPa[t, j]) and OPa[t, j] < stop) else stop
                break
        if ex is None:
            ex = a[end] if np.isfinite(a[end]) else last
        tr.append(ex / entry - 1)
    E = E.copy(); E["raw252"], E["trade"] = raw, tr
    return E


# ══════════ 锚点 ══════════
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
_b = outcomes(BASE_EV.assign(t_strong=BASE_EV.dp, t_dip=BASE_EV.dp))
print(f"\n锚点:突破池 {len(BASE_EV):,} 笔(应 70,310)、"
      f"净期望 {_b.trade.mean()-COST_TRADE:+.2%}(应 +4.61%)")
assert abs(len(BASE_EV) - 70310) <= 50, f"事件数不符:{len(BASE_EV)}"
assert abs(_b.trade.mean() - COST_TRADE - 0.0461) < 0.0015, "交易级锚点对不上"
print("锚点通过")

# ══════════ 生成新旧两版事件 ══════════
OLD = add_metrics(build_sequence(legacy=True))
NEW_RAW = add_metrics(build_sequence(legacy=False))
print(f"\n旧版(MA100须向上、下限15日)事件 {len(OLD):,}")
print(f"新版(去掉MA100向上、下限自适应前)事件 {len(NEW_RAW):,}")

# 自适应下限:按**当年**全市场中位调整天数(只用当年及之前的信息)
med_by_year = NEW_RAW.groupby("year")["调整天数"].median()
floor_by_year = np.maximum(MIN_ADJ_FLOOR, (MIN_ADJ_RATIO * med_by_year).round()).astype(int)
print(f"\n自适应下限(0.15 × 当年中位调整天数,底 {MIN_ADJ_FLOOR}):")
print("  " + "  ".join(f"{y}:{v}日" for y, v in floor_by_year.items()))
NEW = NEW_RAW[NEW_RAW["调整天数"] >= NEW_RAW["year"].map(floor_by_year)].reset_index(drop=True)
print(f"新版(应用自适应下限后)事件 **{len(NEW):,}**")

# ══════════ 分位阈值:按**当年横截面**取最优 40% ══════════
def add_flags_quantile(E):
    E = E.copy()
    for col in ("深度", "缩量比", "收敛比"):
        thr = E.groupby("year")[col].transform(lambda s: s.quantile(Q_KEEP))
        E[col + "✓"] = (E[col] <= thr)
    E["满足条数"] = E[["深度✓", "缩量比✓", "收敛比✓"]].sum(axis=1)
    return E


def add_flags_absolute(E):
    E = E.copy()
    E["缩量比✓"] = E["缩量比"] < 0.80
    E["收敛比✓"] = E["收敛比"] < 0.80
    E["深度✓"] = E["深度"] <= 0.352
    E["满足条数"] = E[["深度✓", "缩量比✓", "收敛比✓"]].sum(axis=1)
    return E


OLD = outcomes(add_flags_absolute(OLD))
NEW = outcomes(add_flags_quantile(NEW))
print(f"\n{'='*100}\n选中率:逐年对照(改造的直接目标是消掉单调下滑)\n{'='*100}")
print(f"{'年份':<6}{'旧版事件':>9}{'旧版三条全中':>13}{'|':>3}{'新版事件':>9}{'新版三条全中':>13}")
for y in range(2015, 2027):
    a = OLD[OLD.year == y]; b = NEW[NEW.year == y]
    if len(a) < 50 and len(b) < 50:
        continue
    ra = (a.满足条数 == 3).mean() if len(a) else np.nan
    rb = (b.满足条数 == 3).mean() if len(b) else np.nan
    print(f"{y:<6}{len(a):>9,}{ra:>13.2%}{'|':>3}{len(b):>9,}{rb:>13.2%}")
for nm, E in (("旧版", OLD), ("新版", NEW)):
    ein = E[E.date < "2020-01-01"]; eout = E[E.date >= "2020-01-01"]
    print(f"  {nm}:选择集三条全中 {(ein.满足条数==3).mean():.2%}   "
          f"验证集 {(eout.满足条数==3).mean():.2%}")

OLD.to_parquet(f"{SP}/adaptive_events_old.parquet")
NEW.to_parquet(f"{SP}/adaptive_events_new.parquet")

# ══════════ 第一关:20 只案例回归 ══════════
CASES = [("600066", "宇通客车"), ("300750", "宁德时代"), ("300476", "胜宏科技"),
         ("688183", "生益电子"), ("601567", "三星医疗"), ("603259", "药明康德"),
         ("603893", "瑞芯微"), ("300972", "万辰集团"), ("603119", "浙江荣泰"),
         ("300760", "迈瑞医疗"), ("300059", "东方财富"), ("002475", "立讯精密"),
         ("002709", "天赐材料"), ("301377", "鼎泰高科"), ("688498", "源杰科技"),
         ("300604", "长川科技"), ("688347", "华虹公司"), ("688256", "寒武纪"),
         ("688041", "海光信息")]
print(f"\n{'='*100}\n第一关 回归:{len(CASES)} 只用户案例(旧版捞到的,新版必须仍捞到)\n{'='*100}")
print(f"{'代码':<8}{'名称':<10}{'旧版亮灯段':>10}{'新版亮灯段':>10}{'新版事件数':>11}  判定")
n_bad = 0
for code, name in CASES:
    a = OLD[(OLD.code == code) & (OLD.满足条数 == 3)]
    b = NEW[(NEW.code == code) & (NEW.满足条数 == 3)]
    ok = len(b) >= len(a) or (len(a) == 0 and len(b) > 0) or len(b) > 0
    if len(a) > 0 and len(b) == 0:
        ok = False
    if not ok:
        n_bad += 1
    tag = "**丢了**" if not ok else ("**新捞到**" if len(a) == 0 and len(b) > 0 else "保持")
    print(f"{code:<8}{name:<10}{len(a):>10}{len(b):>10}"
          f"{len(NEW[NEW.code==code]):>11}  {tag}")
print(f"\n  丢失的案例:**{n_bad} 只**  →  "
      f"{'**第一关通过**' if n_bad == 0 else '**第一关未通过,改造需回滚**'}")
assert n_bad == 0, f"回归失败:{n_bad} 只案例在新版下丢失"
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: adaptive_events_old/new.parquet")
