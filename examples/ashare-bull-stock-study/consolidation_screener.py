#!/usr/bin/env python3
"""整理形态筛选器:缩量 + 波动收敛 + 浅回调

═══ 这个脚本量化的是什么 ═══
第五十九节在 13,402 个「强势 → 深调20周线 → 买点」事件上做归因,
20 个特征里**只有三个通过全部纪律,而且全部来自调整期**:

    特征              P(交易赚钱|特征)   lift    2015前/后同向
    波动收缩 <0.8×        24.77%       1.37     1.48 / 1.27  ✓
    调整期缩量 <0.8×      24.06%       1.33     1.49 / 1.15  ✓
    调整深度 浅于中位      21.95%       1.21     1.19 / 1.32  ✓
    (基准 18.08%)

**样本外验证(阈值在 2019 年底前定死,2020-2026 上一个参数没动):**

    配置          胜率              单笔净期望         最大回撤
    不筛          15.85%           +1.73%          -63.7%
    三条全中      **22.88%**       **+6.68%**      **-26.1%**

═══ 两条必须先读的限制 ═══

**一、它没有通过可交易性判据。**
同一份 OOS 检验里组合级年化只有 +5.08%,与「同数量随机抽取」无法区分(p=0.31)。
原因已诊断:三条全中只留下 4.2% 的事件(约 59 笔/年),10 个仓位大部分时间空着。
**不是过滤器无效,是过滤器与组合容量不匹配。**

**二、它只在「强势 → 深调20周线 → 买点」这个时序里成立,搬不出去(第六十节)。**
把同样三个特征改用通用定义(调整期起点取近250日最高点)搬到两个大池:

    池                  基线胜率   三条全中胜率   三条全中净期望   随机对照 p
    60日新高(42,609)     16.09%     17.32%      **-0.02%**    0.28
    口袋支点(38,858)     17.87%     18.84%      **-0.43%**    0.83

胜率只提 1pp(原场景是 7pp),**净期望反而转负**,「缩量」这一条在两个池里方向都反了。
→ **所以本脚本坚持用第五十九节的原定义(以「强势日」为调整期起点),
不用那个搬不动的通用定义。** 用法上也必须限定在同一个时序里。

═══ 正确用法 ═══
  ✓ 对**已经走完「强势 → 深调20周线」**的股票做体检与排序
  ✓ 当纪律用:形态不干净的,即使故事再好也降权
  ✗ **不要**当作可以直接照买的信号系统
  ✗ **不要**拿它去筛没有前置强势段的普通股票 —— 那个场景实测无效
⚠️ 仅供研究参考,不构成投资建议。

═══ 三段口径(与第五十九节逐字一致,全部只用当日及之前的数据) ═══
  强势日 ts = 最近一次「60日涨幅进入全市场前10%」的交易日(回看 250 日内)
  触线日 td = ts 之后最早一天,最低价 ≤ 20周线(MA100)×1.03,且 MA100 仍向上
  今天   t  = 待评估的买点

  1) 浅回调    深度   = 1 − min(low[ts..t]) ÷ max(high[ts..t])
  2) 缩量      缩量比 = 均量(ts..t) ÷ 均量(ts 之前 60 日)
  3) 波动收敛  收敛比 = 均真实波幅(td..t) ÷ 均真实波幅(ts 之前 60 日)

判定(阈值取自第五十九节的选择集,样本外未重算):
  缩量比 < 0.80、收敛比 < 0.80、深度 ≤ 0.352

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
STRONG_LOOKBACK = 250  # 往回找「强势日」的窗口
PRE_WIN = 60           # 强势日之前用于对比的基期长度
MIN_ADJ_DAYS = 15      # 调整期至少这么长


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = np.roll(c, 1)
    pc[0] = np.nan
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def score_one(h, l, c, v, ma100, strong_days, t: int) -> dict | None:
    """按第五十九节口径,算截至下标 t(含)的三个整理指标。不成立返回 None。"""
    cand = strong_days[(strong_days <= t) & (strong_days >= t - STRONG_LOOKBACK)]
    if cand.size == 0:
        return None
    ts = int(cand[-1])                      # 最近一次强势日
    if t - ts < MIN_ADJ_DAYS:
        return None
    # 触线日:ts 之后最早一次触及 20周线,且 20周线仍向上
    td = -1
    for k in range(ts + 1, t + 1):
        if (np.isfinite(ma100[k]) and np.isfinite(l[k]) and l[k] <= ma100[k] * 1.03
                and k >= 20 and np.isfinite(ma100[k - 20]) and ma100[k] > ma100[k - 20]):
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


def n_pass(s: dict) -> int:
    return int((s["缩量比"] < THR_SHRINK) + (s["收敛比"] < THR_ATR)
               + (s["深度"] <= THR_DEPTH))


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
    print(f"阈值:缩量比 < {THR_SHRINK}   收敛比 < {THR_ATR}   深度 ≤ {THR_DEPTH}")
    print(f"{'='*104}")
    print(f"{'日期':<12}{'强势日':<12}{'触20周线':<12}{'调整天':>7}{'现价':>8}{'深度':>8}"
          f"{'缩量比':>8}{'收敛比':>8}{'条数':>5}")
    hits, prev_key, shown = [], None, 0
    for t in range(len(idx)):
        if not (d0 <= idx[t] <= d1) or not np.isfinite(c[t]):
            continue
        sd = sd_all[sd_all <= t]
        if sd.size == 0:
            continue
        s = score_one(h, l, c, v, m100, sd, t)
        if s is None:
            continue
        n = n_pass(s)
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
    print("第五十九节实测:同一时序内三条全中的组合级年化与随机无法区分(p=0.31)。")
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
        s = score_one(h, l, c, v, m100, sd, t_pos)
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
    R["缩量✓"] = R["缩量比"] < THR_SHRINK
    R["收敛✓"] = R["收敛比"] < THR_ATR
    R["浅调✓"] = R["深度"] <= THR_DEPTH
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
    print(f"阈值:缩量比 < {THR_SHRINK}   收敛比 < {THR_ATR}   深度 ≤ {THR_DEPTH}")
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
    print("怎么读这张表(第五十九节实测,样本外 2020-2026,同一时序内):")
    print("  三条全中的历史交易胜率 **22.88%**(不筛 15.85%)、单笔净期望 +6.68%(vs +1.73%)、")
    print("  最大回撤 -26.1%(vs -63.7%)。**但组合级年化与随机无法区分(p=0.31)。**")
    print("  第六十节另证:换到没有前置强势段的池子里,同样三条**净期望转负**、迁移失败。")
    print("  → **当排序/体检工具用,只用在走完这个时序的股票上,不要当直接照买的信号。**")
    print("⚠️ 仅供研究参考,不构成投资建议。")

    if a.out:
        R[cols + ["排序分"]].to_csv(a.out, index=False)
        print(f"\n完整结果已写入 {a.out}({len(R):,} 行)")


if __name__ == "__main__":
    main()
