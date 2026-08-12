"""筛选器层面的案例覆盖复核(第六十一节落库前的自查)

═══ 为什么要跑这个 ═══
第六十一节的第一关回归是在**事件管线**(强势→触线→买点,adaptive_events.py)
上过的:19 只案例 0 丢失、新捞 6 只。但用户手里的工具是
`consolidation_screener.py`,它算的是**逐日状态**(今天这只处在不处在
干净整理里),和事件管线不是同一个对象。

落库前冒烟测试发现:宇通客车 2023-06~12 在 `--legacy` 下三条全中 36 天,
在**新默认口径下一天都没有**。

我先后给过两个错误解释,都被这个脚本推翻,记在这里:
  错解一「去掉 MA100 向上 → 触线日前移 → 收敛比被拉高」:
    单点核对 2023-10-17,新口径收敛比 **0.721**、旧口径 0.557,
    两个都远低于 0.80。td 前移确实抬高了收敛比,但**没抬到卡住的程度**。
  错解二(更早):把 2023-09-11 那天的收敛比 1.52 当成整段的值 ——
    那是整理段刚开始的读数,不是整段。

本脚本用 A/B/C 三档把两处改动拆开,直接测,不再猜:
  A = 旧版(MA100 向上 + 写死 15 日 + 绝对阈值 0.80/0.80/0.352)
  B = 新触线日规则 + 自适应下限 + **仍用绝对阈值**
  C = 新版默认(新触线日规则 + 自适应下限 + **当期横截面 40% 分位**)
A→B 隔离「事件定义」的改动,B→C 隔离「阈值口径」的改动。

═══ 本脚本做什么 ═══
把 19 只案例在筛选器口径下逐日重算,新/旧两版并列,只报数不调参。
(事前声明「第六十一节是最后一轮参数改造」,这里不改任何参数。)
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"

# 与 consolidation_screener.py 逐字一致
THR_SHRINK, THR_ATR, THR_DEPTH = 0.80, 0.80, 0.352
MIN_ADJ_DAYS_LEGACY = 15
Q_KEEP, MIN_ADJ_RATIO, MIN_ADJ_FLOOR = 0.40, 0.15, 10
STRONG_LOOKBACK, PRE_WIN = 250, 60

CASES = [("600066", "宇通客车"), ("300750", "宁德时代"), ("300476", "胜宏科技"),
         ("688183", "生益电子"), ("601567", "三星医疗"), ("603259", "药明康德"),
         ("603893", "瑞芯微"), ("300972", "万辰集团"), ("603119", "浙江荣泰"),
         ("300760", "迈瑞医疗"), ("300059", "东方财富"), ("002475", "立讯精密"),
         ("002709", "天赐材料"), ("301377", "鼎泰高科"), ("688498", "源杰科技"),
         ("300604", "长川科技"), ("688347", "华虹公司"), ("688256", "寒武纪"),
         ("688041", "海光信息")]

t0 = time.time()
hi, lo, cl, vo = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    x = pd.read_parquet(f, columns=["high", "low", "close", "volume"])
    if x.empty:
        continue
    hi[k] = pd.to_numeric(x["high"], errors="coerce")
    lo[k] = pd.to_numeric(x["low"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    vo[k] = pd.to_numeric(x["volume"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None) if getattr(CL.index, "tz", None) else CL.index
idx = CL.index
HI = pd.DataFrame(hi).set_axis(idx).where(lambda d: d > 0)
LO = pd.DataFrame(lo).set_axis(idx).where(lambda d: d > 0)
VO = pd.DataFrame(vo).set_axis(idx)
CL = CL.where(CL > 0)
MA100 = CL.rolling(100, min_periods=100).mean()
STRONG = ((CL.pct_change(60).rank(axis=1, pct=True) * 100) > 90).to_numpy()
Ha, La, Ca, Va = (HI.to_numpy(float), LO.to_numpy(float),
                  CL.to_numpy(float), VO.to_numpy(float))
Ma = MA100.to_numpy(float)
TRa = np.maximum(Ha - La, np.maximum(np.abs(Ha - np.roll(Ca, 1, 0)),
                                     np.abs(La - np.roll(Ca, 1, 0))))
TRa[0] = np.nan
codes = list(CL.columns)
NT = len(idx)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)", flush=True)


def _nanmean(a):
    a = a[np.isfinite(a)]
    return a.mean() if a.size else np.nan


def score(j: int, t: int, legacy: bool):
    """与 consolidation_screener.score_one 同口径。"""
    sd = np.flatnonzero(STRONG[max(0, t - STRONG_LOOKBACK):t + 1, j])
    if sd.size == 0:
        return None
    ts = int(sd[-1] + max(0, t - STRONG_LOOKBACK))
    if t - ts < (MIN_ADJ_DAYS_LEGACY if legacy else MIN_ADJ_FLOOR):
        return None
    td = -1
    for k in range(ts + 1, t + 1):
        touch = np.isfinite(Ma[k, j]) and np.isfinite(La[k, j]) and La[k, j] <= Ma[k, j] * 1.03
        if legacy:
            touch = touch and (k >= 20 and np.isfinite(Ma[k - 20, j])
                               and Ma[k, j] > Ma[k - 20, j])
        if touch:
            td = k
            break
    if td < 0:
        return None
    hs, ls = Ha[ts:t + 1, j], La[ts:t + 1, j]
    hs, ls = hs[np.isfinite(hs)], ls[np.isfinite(ls)]
    if not (hs.size and ls.size and hs.max() > 0):
        return None
    vpre = _nanmean(Va[max(ts - PRE_WIN, 0):ts, j])
    tpre = _nanmean(TRa[max(ts - PRE_WIN, 0):ts, j])
    return {"调整天数": t - ts, "深度": 1 - ls.min() / hs.max(),
            "缩量比": _nanmean(Va[ts:t + 1, j]) / vpre if vpre and vpre > 0 else np.nan,
            "收敛比": _nanmean(TRa[td:t + 1, j]) / tpre if tpre and tpre > 0 else np.nan,
            "_td": td, "_ts": ts}


def n_pass(s, thr):
    """注意:必须逐项 int(),不能直接相加 —— numpy 的 bool_ 相加是逻辑或,
    True+True+True 会得到 1 而不是 3(本脚本第一版就栽在这里)。
    筛选器本体里这些值是 float() 转过的 Python 标量,不受影响。"""
    if thr is None:
        return (int(s["缩量比"] < THR_SHRINK) + int(s["收敛比"] < THR_ATR)
                + int(s["深度"] <= THR_DEPTH))
    return (int(s["缩量比"] <= thr["缩量比"]) + int(s["收敛比"] <= thr["收敛比"])
            + int(s["深度"] <= thr["深度"]))


# ── 每月一次的横截面阈值(与筛选器同法:月初重算、月内沿用) ──
month_key = pd.Series(idx.year * 100 + idx.month, index=range(NT))
first_of_month = month_key.groupby(month_key).head(1).index.to_numpy()
THR = {}
for n_done, t in enumerate(first_of_month):
    if t < 300:
        continue
    vals = {"缩量比": [], "收敛比": [], "深度": [], "调整天数": []}
    for j in range(len(codes)):
        if not np.isfinite(Ca[t, j]):
            continue
        s = score(j, int(t), False)
        if s is None:
            continue
        for k in vals:
            vals[k].append(s[k])
    if len(vals["深度"]) < 50:
        continue
    THR[int(month_key[t])] = (
        {k: float(np.nanquantile(vals[k], Q_KEEP)) for k in ("缩量比", "收敛比", "深度")},
        max(MIN_ADJ_FLOOR, int(round(MIN_ADJ_RATIO * np.median(vals["调整天数"])))),
        len(vals["深度"]))
    if n_done % 20 == 0:
        print(f"  阈值 {int(month_key[t])}  n={len(vals['深度'])}  "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"月度阈值 {len(THR)} 个月  ({time.time()-t0:.0f}s)", flush=True)

rows = []
for code, name in CASES:
    if code not in codes:
        rows.append({"代码": code, "名称": name, "旧版亮灯": -1, "新版亮灯": -1})
        continue
    j = codes.index(code)
    # A=旧版(旧td规则+写死15日+绝对阈值) B=新td/下限+绝对阈值 C=新版默认(新td/下限+分位)
    # A→B 隔离「去掉MA100向上 + 自适应下限」的影响,B→C 隔离「绝对→分位」的影响
    nA = nB = nC = 0
    fA = fB = fC = None
    for t in range(300, NT):
        if not np.isfinite(Ca[t, j]):
            continue
        s_o = score(j, t, True)
        if s_o is not None and n_pass(s_o, None) == 3:
            nA += 1
            fA = fA or idx[t].date()
        mk = int(month_key[t])
        if mk not in THR:
            continue
        thr, floor, _ = THR[mk]
        s_n = score(j, t, False)
        if s_n is None or s_n["调整天数"] < floor:
            continue
        if n_pass(s_n, None) == 3:
            nB += 1
            fB = fB or idx[t].date()
        if n_pass(s_n, thr) == 3:
            nC += 1
            fC = fC or idx[t].date()
    rows.append({"代码": code, "名称": name, "A旧版": nA, "B新td绝对阈值": nB,
                 "C新版默认": nC, "A首次": fA, "B首次": fB, "C首次": fC})
    print(f"  {code} {name}  A旧 {nA:>4}  B新td+绝对 {nB:>4}  C新默认 {nC:>4}  "
          f"({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(rows)
R["判定"] = np.where((R.A旧版 > 0) & (R.C新版默认 == 0), "**新版丢了**",
                    np.where((R.A旧版 == 0) & (R.C新版默认 > 0), "新版新捞到",
                             np.where(R.C新版默认 >= R.A旧版, "新版更多", "新版更少")))
print(f"\n{'='*104}\n筛选器口径:19 只案例逐日亮灯天数(全历史)\n{'='*104}")
print(R.to_string(index=False))
print(f"\nA 旧版亮过灯: {(R.A旧版>0).sum()} 只   "
      f"B 新td+绝对阈值: {(R.B新td绝对阈值>0).sum()} 只   "
      f"C 新版默认: {(R.C新版默认>0).sum()} 只")
print(f"新版丢失: **{(R.判定=='**新版丢了**').sum()} 只**   "
      f"新版新捞: {(R.判定=='新版新捞到').sum()} 只")
print("\n分解:A→B 是「去掉MA100向上 + 自适应下限」;B→C 是「绝对阈值→当期分位」")
ex = sorted(THR)[::30]
print("\n抽样月度分位阈值(缩量比/收敛比/深度,下限,样本数):")
for m in ex:
    th, fl, n = THR[m]
    print(f"  {m}  {th['缩量比']:.2f} / {th['收敛比']:.2f} / {th['深度']:.1%}"
          f"   下限 {fl:>3} 日   n={n}")
R.to_csv(f"{SP}/screener_case_coverage.csv", index=False)
print(f"\n→ {SP}/screener_case_coverage.csv   ({time.time()-t0:.0f}s)")
