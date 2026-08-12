#!/usr/bin/env python3
"""整理形态筛选器:缩量 + 波动收敛 + 浅回调

═══ 这个脚本量化的是什么 ═══
第五十九节在 13,402 个「强势 → 深调20周线 → 买点」事件上做归因,20 个特征里
**只有三个通过全部纪律,而且全部来自调整期**:缩量、波动收敛、浅回调。

第六十一节把三个特征的**阈值口径**从「钉死的绝对值」改成「当期横截面分位」,
并让调整天数下限自适应。改的理由是三条实测证据:
  1 调整期系统性缩短:全样本中位 2015 **135日** → 2024 **83日**(-40%)
  2 两个比值的分布右移:缩量比中位 1.27→1.40、收敛比 1.20→1.37,
    使固定阈值 0.8 的选中率 5.85%→4.21%,**2025 单年只剩 2.95%**
  3 2024-2026 的大牛股整理期短到 **顶住了原来写死的 15 日下限**
**不是形态失效,是尺子钉死了而市场节奏变了。**

**新口径样本外结果(2020-2026,口径在 2019 年底前定死,验证集未调参):**

    配置            胜率           单笔净期望     组合年化      最大回撤
    不筛            15.43%        +1.61%       +1.50%      -62.0%
    三条全中        **20.61%**    +3.50%       +10.37%     -35.3%
    深度最浅40%     19.87%        +3.16%       **+14.14%** -32.4%

改造修好了旧口径的两个毛病:
  - 逐年选中率不再单调下滑(旧 5.85%→4.21%→2.95%;新稳定在 15~19%)
  - 第六十节「搬到大池就净期望转负」不再出现
    (60日新高 -0.02%→**+2.29%**、口袋支点 -0.43%→**+2.96%**)
  - 用户 19 只案例回归全过,并**新捞到 6 只**(宇通/宁德/胜宏/药明/迈瑞/立讯)

═══ 三条必须先读的限制 ═══

**一、主判据仍然没过。**「三条全中」OOS 组合年化 +10.37%,
但 300 次同数量随机对照 **p=0.16** —— 与随机抽同样多的事件无法区分。

**二、唯一过随机对照的是「深度最浅40%」单条**(年化 +14.14%、p=0.0100,
Bonferroni 阈值 0.0125)。但它是 4 选 1、p 值贴边,
且在两个大池的迁移测试里 **p=0.19 / 0.61 不显著**。**不足以称为发现。**

**三、它只在「强势 → 深调20周线 → 买点」这个时序里有意义。**

═══ 正确用法 ═══
  ✓ 对**已经走完「强势 → 深调20周线」**的股票做体检与排序
  ✓ 当纪律用:形态不干净的,即使故事再好也降权
  ✗ **不要**当作可以直接照买的信号系统
  ✗ **不要**拿它去筛没有前置强势段的普通股票 —— 那个场景实测无效
⚠️ 仅供研究参考,不构成投资建议。

═══ 三段口径(与第五十九节逐字一致,全部只用当日及之前的数据) ═══
  强势日 ts = 最近一次「60日涨幅进入全市场前10%」的交易日(回看 250 日内)
  触线日 td = ts 之后最早一天,最低价 ≤ 20周线(MA100)×1.03
              (`--legacy` 还要求 MA100 向上 —— 该条误杀了用户认可的两个案例)
  今天   t  = 待评估的买点

  1) 浅回调    深度   = 1 − min(low[ts..t]) ÷ max(high[ts..t])
  2) 缩量      缩量比 = 均量(ts..t) ÷ 均量(ts 之前 60 日)
  3) 波动收敛  收敛比 = 均真实波幅(td..t) ÷ 均真实波幅(ts 之前 60 日)

判定:**默认**用当期横截面最优 40% 分位、下限 = max(10, 0.15 × 当期中位调整天数);
`--legacy` 切回第五十九节的绝对阈值(0.80/0.80/0.352、下限写死 15 日、要求20周线向上)。

═══ 怎么用 ═══
    # 全市场:今天有哪些股票处在干净的整理里
    python consolidation_screener.py --data /path/to/parquet_dir
    python consolidation_screener.py --data DIR --date 2026-07-31 --top 40
    python consolidation_screener.py --data DIR --all --out 全部.csv

    # 单只回看:某只股票历史上什么时候亮过灯(用来验证你自己的案例)
    python consolidation_screener.py --data DIR --code 600066 --from 2023-01-01 --to 2024-03-31
    python consolidation_screener.py --data DIR --code 300750 --daily   # 逐日打印

数据要求:一个目录,每只股票一个 parquet,文件名即代码,
至少含 high / low / close / volume 四列,DatetimeIndex。
**「强势日」是全市场横截面分位,所以目录必须是全市场,只放几只算不出来。**

⚠️ 数据层面的已知偏差:若你的 parquet 目录只含**当前仍在市**的股票,
则历史横截面分位会偏乐观(退市股不在分母里)。本研究实测该偏差约
-0.4~-0.6pp/年(第四十二节),不影响形态判定本身,但看历史统计时要记得。
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

# ── 阈值:第五十九节在 2014-2019 上定死的,样本外未重算 ──
THR_SHRINK = 0.80      # 缩量比上限
THR_ATR = 0.80         # 波动收敛比上限
THR_DEPTH = 0.352      # 回调深度上限(选择集中位)
MIN_ADJ_DAYS_LEGACY = 15   # 旧口径:调整期下限写死
# ── 新口径(第六十一节,当期横截面分位 + 自适应下限;默认) ──
Q_KEEP = 0.40          # 三个指标各取当期最优 40%
MIN_ADJ_RATIO = 0.15   # 下限 = 0.15 × 当期中位调整天数
MIN_ADJ_FLOOR = 10     # 自适应下限的绝对底
STRONG_LOOKBACK = 250  # 往回找「强势日」的窗口
PRE_WIN = 60           # 强势日之前用于对比的基期长度
MIN_ADJ_DAYS = 15      # 调整期至少这么长


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = np.roll(c, 1)
    pc[0] = np.nan
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def score_one(h, l, c, v, ma100, strong_days, t: int, legacy: bool = False) -> dict | None:
    """按第五十九节口径,算截至下标 t(含)的三个整理指标。不成立返回 None。"""
    cand = strong_days[(strong_days <= t) & (strong_days >= t - STRONG_LOOKBACK)]
    if cand.size == 0:
        return None
    ts = int(cand[-1])                      # 最近一次强势日
    # 旧口径写死 15 日;新口径先用绝对底 10,真正的自适应下限由调用方按当期中位数再过滤
    # (第六十一节:2024-2026 大牛股整理期短到顶住了写死的 15 日)
    if t - ts < (MIN_ADJ_DAYS_LEGACY if legacy else MIN_ADJ_FLOOR):
        return None
    # 触线日:ts 之后最早一次触及 20周线,且 20周线仍向上
    td = -1
    for k in range(ts + 1, t + 1):
        touch = (np.isfinite(ma100[k]) and np.isfinite(l[k]) and l[k] <= ma100[k] * 1.03)
        if legacy:   # 旧口径还要求「20周线向上」—— 该条误杀了用户认可的两个案例
            touch = touch and (k >= 20 and np.isfinite(ma100[k - 20])
                               and ma100[k] > ma100[k - 20])
        if touch:
            td = k
            break
    if td < 0:
        return None                          # 还没深调到 20周线,时序不完整

    hi_seg = h[ts:t + 1]; lo_seg = l[ts:t + 1]
    hi_seg = hi_seg[np.isfinite(hi_seg)]; lo_seg = lo_seg[np.isfinite(lo_seg)]
    if not (hi_seg.size and lo_seg.size and hi_seg.max() > 0):
        return None
    depth = 1 - lo_seg.min() / hi_seg.max()

    v_adj = v[ts:t + 1]; v_adj = v_adj[np.isfinite(v_adj)]
    v_pre = v[max(ts - PRE_WIN, 0):ts]; v_pre = v_pre[np.isfinite(v_pre)]
    shrink = (v_adj.mean() / v_pre.mean()
              if v_adj.size and v_pre.size and v_pre.mean() > 0 else np.nan)

    tr = true_range(h, l, c)
    tr_now = tr[td:t + 1]; tr_now = tr_now[np.isfinite(tr_now)]
    tr_pre = tr[max(ts - PRE_WIN, 0):ts]; tr_pre = tr_pre[np.isfinite(tr_pre)]
    conv = (tr_now.mean() / tr_pre.mean()
            if tr_now.size and tr_pre.size and tr_pre.mean() > 0 else np.nan)

    cur, pk = c[t], hi_seg.max()
    return {"_ts": ts, "_td": td, "调整天数": t - ts, "触线后天数": t - td,
            "现价": float(cur) if np.isfinite(cur) else np.nan,
            "距区间高": float(cur / pk - 1) if np.isfinite(cur) and pk > 0 else np.nan,
            "深度": float(depth),
            "缩量比": float(shrink) if np.isfinite(shrink) else np.nan,
            "收敛比": float(conv) if np.isfinite(conv) else np.nan}


def load_panel(data_dir: str):
    """读全市场面板,并算出强势日矩阵与 20周线。两种模式共用。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"{data_dir} 下没有 parquet 文件")
    print(f"读取 {len(files)} 个文件…")
    closes, frames = {}, {}
    for f in files:
        code = os.path.basename(f)[:-8]
        try:
            x = pd.read_parquet(f, columns=["high", "low", "close", "volume"])
        except Exception:
            continue
        if x.empty:
            continue
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        frames[code] = x
        closes[code] = pd.to_numeric(x["close"], errors="coerce")
    CL = pd.DataFrame(closes).sort_index()
    CL = CL.where(CL > 0)
    # 强势日:60日涨幅的全市场横截面前 10%(必须用全市场,单只算不出横截面分位)
    RPS60 = CL.pct_change(60).rank(axis=1, pct=True) * 100
    return CL, frames, (RPS60 > 90).to_numpy(), CL.rolling(100, min_periods=100).mean()


def series_of(frames, idx, code):
    x = frames[code].reindex(idx)
    return (pd.to_numeric(x["high"], errors="coerce").where(lambda s: s > 0).to_numpy(float),
            pd.to_numeric(x["low"], errors="coerce").where(lambda s: s > 0).to_numpy(float),
            pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0).to_numpy(float),
            pd.to_numeric(x["volume"], errors="coerce").to_numpy(float))


def n_pass(s: dict, thr: dict | None = None) -> int:
    """thr=None 用旧口径绝对阈值;传入 {缩量比,收敛比,深度} 则用当期分位阈值。"""
    # 逐项 int() 后再相加:score_one 里这三个值已 float() 成 Python 标量,
    # 但若哪天传进来的是 numpy 标量,`np.bool_ + np.bool_` 是**逻辑或**,
    # True+True+True 会得到 1 而不是 3。写成这样才与实现细节无关。
    if thr is None:
        return (int(s["缩量比"] < THR_SHRINK) + int(s["收敛比"] < THR_ATR)
                + int(s["深度"] <= THR_DEPTH))
    return (int(s["缩量比"] <= thr["缩量比"]) + int(s["收敛比"] <= thr["收敛比"])
            + int(s["深度"] <= thr["深度"]))


def cross_section_thresholds(CL, frames, STRONG, MA100, t_pos):
    """t_pos 当天全市场处于整理中的股票的分位阈值 + 自适应下限。

    单只回看必须和全市场模式**用同一把尺子**,否则同一只股票两个模式判定不一致。
    逐日重算太慢(每天要扫全市场),所以调用方按**月**重算一次、月内沿用 ——
    阈值是慢变量,月内漂移可忽略。
    """
    idx = CL.index
    vals = {"缩量比": [], "收敛比": [], "深度": [], "调整天数": []}
    for ci, code in enumerate(CL.columns):
        h, l, c, v = series_of(frames, idx, code)
        if not np.isfinite(c[t_pos]):
            continue
        sd = np.flatnonzero(STRONG[:t_pos + 1, ci])
        if sd.size == 0:
            continue
        s_ = score_one(h, l, c, v, MA100[code].to_numpy(float), sd, t_pos)
        if s_ is None:
            continue
        for k in vals:
            vals[k].append(s_[k])
    if len(vals["深度"]) < 50:
        return None, MIN_ADJ_FLOOR
    floor = max(MIN_ADJ_FLOOR, int(round(MIN_ADJ_RATIO * np.median(vals["调整天数"]))))
    return ({k: float(np.nanquantile(vals[k], Q_KEEP))
             for k in ("缩量比", "收敛比", "深度")}, floor)


def run_single(a) -> None:
    """单只回看:逐日算三个指标,标出三条全中的日子。"""
    CL, frames, STRONG, MA100 = load_panel(a.data)
    idx = CL.index
    code = a.code
    if code not in frames:
        raise SystemExit(f"{a.data} 里没有 {code}.parquet")
    ci = list(CL.columns).index(code)
    h, l, c, v = series_of(frames, idx, code)
    m100 = MA100[code].to_numpy(float)
    sd_all = np.flatnonzero(STRONG[:, ci])
    d0 = pd.Timestamp(a.start) if a.start else idx[0]
    d1 = pd.Timestamp(a.end) if a.end else idx[-1]
    print(f"\n{'='*104}")
    print(f"{code}   区间 {d0.date()} ~ {d1.date()}   全期强势日(RPS60>90) {sd_all.size} 天")
    print("阈值:" + (f"绝对 {THR_SHRINK}/{THR_ATR}/{THR_DEPTH}(--legacy)"
                   if getattr(a, "legacy", False) else
                   f"当期最优 {Q_KEEP:.0%} 分位(逐月重算)+ 自适应调整天数下限"))
    print(f"{'='*104}")
    print(f"{'日期':<12}{'强势日':<12}{'触20周线':<12}{'调整天':>7}{'现价':>8}{'深度':>8}"
          f"{'缩量比':>8}{'收敛比':>8}{'条数':>5}")
    legacy = getattr(a, "legacy", False)
    hits, prev_key, shown = [], None, 0
    thr, floor, thr_month = None, MIN_ADJ_FLOOR, None
    for t in range(len(idx)):
        if not (d0 <= idx[t] <= d1) or not np.isfinite(c[t]):
            continue
        sd = sd_all[sd_all <= t]
        if sd.size == 0:
            continue
        if not legacy and (idx[t].year, idx[t].month) != thr_month:
            thr_month = (idx[t].year, idx[t].month)
            thr, floor = cross_section_thresholds(CL, frames, STRONG, MA100, t)
        s = score_one(h, l, c, v, m100, sd, t, legacy=legacy)
        if s is None or (not legacy and s["调整天数"] < floor):
            continue
        n = n_pass(s, None if legacy else thr)
        if n == 3:
            hits.append((idx[t], c[t]))
        key = (n, s["_ts"], s["_td"])          # 状态:条数 / 强势日 / 触线日
        if a.daily or key != prev_key or n == 3:
            print(f"{str(idx[t].date()):<12}{str(idx[s['_ts']].date()):<12}"
                  f"{str(idx[s['_td']].date()):<12}{s['调整天数']:>7}{c[t]:>8.2f}"
                  f"{s['深度']:>8.1%}{s['缩量比']:>8.2f}{s['收敛比']:>8.2f}{n:>5}"
                  f"{'  ← 三条全中' if n == 3 else ''}")
            shown += 1
        prev_key = key
    if shown == 0:
        print("(区间内没有走完「强势 → 深调20周线」这个时序)")

    print(f"\n{'-'*104}")
    if not hits:
        print("区间内 **没有** 三条全中的日子。")
        return
    d_first, p_first = hits[0]
    print(f"三条全中共 **{len(hits)} 天**   首次 **{d_first.date()}**(收盘 {p_first:.2f})"
          f"   最后 {hits[-1][0].date()}")
    t0i = int(idx.searchsorted(d_first))
    for nd in (60, 120, 252):
        if t0i + nd < len(c) and np.isfinite(c[t0i + nd]) and p_first > 0:
            print(f"  首次亮灯后 {nd:>3} 个交易日({idx[t0i+nd].date()}):"
                  f"{c[t0i+nd]:>8.2f}   {c[t0i+nd]/p_first-1:+.1%}")
    print(f"\n**亮灯持续 {len(hits)} 天 —— 这是状态标记,不是买点。**")
    print("第六十一节实测:同一时序内三条全中的组合级年化 +10.37%,"
          "但 300 次随机对照 p=0.16 —— 与随机无法区分。")
    print("⚠️ 仅供研究参考,不构成投资建议。")


def main() -> None:
    ap = argparse.ArgumentParser(description="整理形态筛选:缩量 + 波动收敛 + 浅回调")
    ap.add_argument("--data", required=True, help="逐股 parquet 所在目录")
    ap.add_argument("--date", default=None, help="评估日 YYYY-MM-DD,默认用数据最后一天")
    ap.add_argument("--top", type=int, default=30, help="输出前 N 只")
    ap.add_argument("--out", default=None, help="结果写到这个 CSV")
    ap.add_argument("--all", action="store_true", help="输出所有走完时序的股票,不只三条全中")
    ap.add_argument("--code", default=None, help="单只回看模式:只看这一只的历史")
    ap.add_argument("--from", dest="start", default=None, help="单只模式起始日")
    ap.add_argument("--to", dest="end", default=None, help="单只模式结束日")
    ap.add_argument("--daily", action="store_true",
                    help="单只模式逐日打印(默认只打印状态变化的日子)")
    ap.add_argument("--legacy", action="store_true",
                    help="用第五十九节旧口径(绝对阈值 0.8/0.8/0.352、下限写死15日、"
                         "要求20周线向上)。默认用第六十一节的自适应口径")
    a = ap.parse_args()

    if a.code:
        run_single(a)
        return

    CL, frames, STRONG, MA100 = load_panel(a.data)
    idx = CL.index
    asof = idx[-1] if a.date is None else pd.Timestamp(a.date)
    t_pos = int(idx.searchsorted(asof, side="right")) - 1
    if t_pos < STRONG_LOOKBACK + PRE_WIN:
        raise SystemExit(f"评估日 {asof.date()} 之前历史不足 {STRONG_LOOKBACK + PRE_WIN} 天")
    print(f"评估日 {idx[t_pos].date()}(共 {len(idx)} 个交易日)")

    rows, keep = [], {}
    for ci, code in enumerate(CL.columns):
        h, l, c, v = series_of(frames, idx, code)
        if not np.isfinite(c[t_pos]):
            continue
        sd = np.flatnonzero(STRONG[:t_pos + 1, ci])
        if sd.size == 0:
            continue
        m100 = MA100[code].to_numpy(float)
        s = score_one(h, l, c, v, m100, sd, t_pos, legacy=a.legacy)
        if s is None:
            continue
        s["代码"] = code
        s["强势日"] = idx[s.pop("_ts")].date()
        s["触20周线"] = idx[s.pop("_td")].date()
        rows.append(s)
        keep[code] = (h, l, c, v, m100, sd)

    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit("今天没有股票走完「强势 → 深调20周线」这个时序")
    if a.legacy:
        R["缩量✓"] = R["缩量比"] < THR_SHRINK
        R["收敛✓"] = R["收敛比"] < THR_ATR
        R["浅调✓"] = R["深度"] <= THR_DEPTH
        thr_txt = f"缩量比 < {THR_SHRINK}   收敛比 < {THR_ATR}   深度 ≤ {THR_DEPTH}(旧口径)"
    else:
        floor = max(MIN_ADJ_FLOOR, int(round(MIN_ADJ_RATIO * R["调整天数"].median())))
        R = R[R["调整天数"] >= floor].reset_index(drop=True)
        if R.empty:
            raise SystemExit("应用自适应下限后没有股票入选")
        qs = {c_: R[c_].quantile(Q_KEEP) for c_ in ("缩量比", "收敛比", "深度")}
        R["缩量✓"] = R["缩量比"] <= qs["缩量比"]
        R["收敛✓"] = R["收敛比"] <= qs["收敛比"]
        R["浅调✓"] = R["深度"] <= qs["深度"]
        thr_txt = (f"当期最优 {Q_KEEP:.0%} 分位 → 缩量比 ≤ {qs['缩量比']:.2f}   "
                   f"收敛比 ≤ {qs['收敛比']:.2f}   深度 ≤ {qs['深度']:.1%};"
                   f"自适应下限 {floor} 日")
    R["满足条数"] = R[["缩量✓", "收敛✓", "浅调✓"]].sum(axis=1)
    # 已亮灯天数:往回数连续三条全中的天数。宇通的案例显示这个信号能持续 42 天,
    # 所以「刚亮」和「亮了很久」是两回事,必须让用户一眼看见。只对当前三条全中的算。
    streak = {}
    for code in R.loc[R["满足条数"] == 3, "代码"]:
        h, l, c, v, m100, sd = keep[code]
        n = 0
        for t in range(t_pos, max(t_pos - 250, 0), -1):
            if not np.isfinite(c[t]):
                continue
            s2 = score_one(h, l, c, v, m100, sd[sd <= t], t)
            if s2 is None or n_pass(s2) < 3:
                break
            n += 1
        streak[code] = n
    R["已亮灯天数"] = R["代码"].map(streak).fillna(0).astype(int)
    R["排序分"] = (R["满足条数"] * 10
                 - R["缩量比"].fillna(9) - R["收敛比"].fillna(9) - R["深度"].fillna(9) * 2)
    cols = ["代码", "强势日", "触20周线", "调整天数", "触线后天数", "现价", "距区间高",
            "深度", "缩量比", "收敛比", "缩量✓", "收敛✓", "浅调✓", "满足条数", "已亮灯天数"]
    R = R.sort_values("排序分", ascending=False)
    sel = R if a.all else R[R["满足条数"] == 3]

    print(f"\n{'='*112}")
    print(f"阈值:{thr_txt}")
    print(f"走完「强势 → 深调20周线」的 **{len(R):,} 只**;"
          f"其中三条全中 **{int((R['满足条数']==3).sum()):,} 只**"
          f"({(R['满足条数']==3).mean():.1%})")
    print(f"{'='*112}")
    show = sel.head(a.top)
    if show.empty:
        print("今天没有股票三条全中。")
    else:
        disp = show[cols].copy()
        disp["距区间高"] = disp["距区间高"].map(lambda x: f"{x:+.1%}")
        disp["深度"] = disp["深度"].map(lambda x: f"{x:.1%}")
        for c_ in ("缩量比", "收敛比"):
            disp[c_] = disp[c_].map(lambda x: f"{x:.2f}")
        disp["现价"] = disp["现价"].map(lambda x: f"{x:.2f}")
        print(disp.to_string(index=False))

    print(f"\n{'='*112}")
    print("怎么读这张表(第六十一节,样本外 2020-2026,同一时序内):")
    print("  三条全中:胜率 **20.61%**(不筛 15.43%)、单笔净期望 +3.50%(vs +1.61%)、")
    print("  组合年化 +10.37%(vs +1.50%)、最大回撤 -35.3%(vs -62.0%)。")
    print("  **但 300 次同数量随机对照 p=0.16 —— 与随机抽同样多的事件无法区分。**")
    print("  唯一过随机对照的是「深度最浅40%」单条(年化 +14.14%、p=0.0100),但它是")
    print("  4 选 1、p 值贴边,且在两个大池的迁移测试里 p=0.19/0.61 不显著。")
    print("  → **当排序/体检工具用,只用在走完这个时序的股票上,不要当直接照买的信号。**")
    print("⚠️ 仅供研究参考,不构成投资建议。")

    if a.out:
        R[cols + ["排序分"]].to_csv(a.out, index=False)
        print(f"\n完整结果已写入 {a.out}({len(R):,} 行)")


if __name__ == "__main__":
    main()
