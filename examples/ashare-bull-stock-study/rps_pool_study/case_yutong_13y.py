"""宇通客车 600066:13 年(2013-01-04 ~ 2026-08-03)完整结构量化

**描述性,单只股票,不设判据,不得据此对全市场下任何结论。**
用途:把宇通这一只股票的全部结构摊开,看能问出什么问题,再决定拿什么上全样本。

═══ 分段算法(锚点预验证后锁定,§89 立的规矩)═══
  ZigZag 反转阈值 **θ = 10%**(自极值回撤 ≥10% 确认一个枢轴)
  **θ 是这样定下来的**:要求算法必须复现用户看图指认的两处三段结构 ——
  ① 2013-14 平台 → 2014 年底突破(其后 +100%)
  ② 2023 平台 → 2024-01 突破(其后 +101%)
  实测 θ=10% 两处都认(2014-12-04 / 2024-01-10);θ=15% 漏掉 ②;
  θ=20% 只剩两个;θ=25% 一个都没有。**故锁定 10%。**

  **收回过一个锚点条件**:初稿还要求「不得在 2015-2018 那段横盘里触发」。
  那是**把「表现」伪装成「正确性」** —— 照它调参数就是拿结果反推检测器。
  该条已删除。检测器在哪里触发是**结果**,不是锚点。

  三段模板:上涨段(单条上升腿 ≥30%) → 平台段(后续枢轴全落在 ±35.2% 带内、
  时长 ≥60 日;35.2% 取自筛选器 THR_DEPTH,不是新参数) →
  收盘首次突破平台段最高收盘(**上限 250 日**,与 §89 一致)
"""
import os
import sys

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
OUT = os.environ.get("OXQ_OUT_DIR", SP)
CODE, TH, UP_MIN, BAND, PLAT_MIN, CAP = "600066", 0.10, 0.30, 0.352, 60, 250

x = pd.read_parquet(f"{SP}/oxq_stock_market_fixed/{CODE}.parquet",
                    columns=["close", "volume", "float_mv"])
if getattr(x.index, "tz", None) is not None:
    x.index = x.index.tz_localize(None)
x = x[(x.index >= "2013-01-04") & (x.index <= "2026-08-03")]
p = x["close"].ffill().to_numpy(float)
vol = pd.to_numeric(x["volume"], errors="coerce").to_numpy(float)
mv = pd.to_numeric(x["float_mv"], errors="coerce").to_numpy(float)
d = x.index
N = len(p)
if not (np.isfinite(p).all() and N > 3000):
    sys.exit(f"数据异常 N={N}")
print(f"宇通客车 600066   {N} 个交易日   {d[0].date()} ~ {d[-1].date()}")
print(f"  收盘 {p[0]:.2f} → {p[-1]:.2f}  ({p[-1]/p[0]-1:+.1%})"
      f"   全期最低 {p.min():.2f}({d[p.argmin()].date()})"
      f"   最高 {p.max():.2f}({d[p.argmax()].date()})")


def zigzag(px, th):
    piv = [(0, "L")]
    ext, ei, up = px[0], 0, True
    for i in range(1, len(px)):
        if up:
            if px[i] > ext:
                ext, ei = px[i], i
            elif px[i] <= ext * (1 - th):
                piv.append((ei, "H"))
                ext, ei, up = px[i], i, False
        else:
            if px[i] < ext:
                ext, ei = px[i], i
            elif px[i] >= ext * (1 + th):
                piv.append((ei, "L"))
                ext, ei, up = px[i], i, True
    piv.append((ei, "H" if up else "L"))
    return piv


piv = zigzag(p, TH)
W = 104
print(f"\n{'='*W}\n一、ZigZag(θ={TH:.0%})枢轴与波段 —— 13 年共 {len(piv)-1} 条腿\n{'='*W}")
print(f"{'#':>3} {'起':<12}{'止':<12}{'方向':<5}{'起价':>7}{'止价':>7}"
      f"{'幅度':>9}{'交易日':>7}{'日均量(万手)':>13}")
legs = []
for i in range(len(piv) - 1):
    a, b = piv[i][0], piv[i + 1][0]
    r = p[b] / p[a] - 1
    vv = np.nanmean(vol[a:b + 1]) / 1e6 if b > a else np.nan
    legs.append(dict(i=i, a=a, b=b, r=r, n=b - a, v=vv,
                     d0=str(d[a].date()), d1=str(d[b].date())))
    print(f"{i:>3} {d[a].date()!s:<12}{d[b].date()!s:<12}"
          f"{'↑ 上涨' if r > 0 else '↓ 下跌':<5}{p[a]:>7.2f}{p[b]:>7.2f}"
          f"{r:>+9.1%}{b-a:>7}{vv:>13.1f}")

print(f"\n{'='*W}\n二、13 年里全部 ≥+80% 的上升腿,以及它们**之前**是什么\n{'='*W}")
big = [x_ for x_ in legs if x_["r"] >= 0.80]
print(f"  共 {len(big)} 条(阈值 +80%,放宽一点以免只剩一条)")
for g in big:
    a = g["a"]
    prev = [q for q in legs if q["b"] <= a][-3:]
    r3 = p[a] / p[max(a - 750, 0)] - 1
    r1 = p[a] / p[max(a - 250, 0)] - 1
    lowpos = (p[a] - p[max(a - 750, 0):a + 1].min()) / \
             (p[max(a - 750, 0):a + 1].max() - p[max(a - 750, 0):a + 1].min())
    print(f"\n  **{g['d0']} → {g['d1']}   {p[a]:.2f} → {p[g['b']]:.2f}   "
          f"{g['r']:+.1%}   {g['n']} 个交易日**")
    print(f"     起涨点位置:过去 1 年 {r1:+.1%}、过去 3 年 {r3:+.1%}、"
          f"在 3 年区间中的位置 {lowpos:.2f}")
    print("     之前三条腿:  " + " | ".join(
        f"{q['d0']}→{q['d1']} {q['r']:+.0%} ({q['n']}日)" for q in prev))


def segments(px, dt):
    out, seen = [], set()
    for a in range(len(piv) - 1):
        i0, k0 = piv[a]
        i1, k1 = piv[a + 1]
        if not (k0 == "L" and k1 == "H") or px[i1] / px[i0] - 1 < UP_MIN:
            continue
        b, hi, lo = a + 1, px[i1], px[i1]
        while b + 1 < len(piv):
            j = piv[b + 1][0]
            nh, nl = max(hi, px[j]), min(lo, px[j])
            if nh / nl - 1 > BAND:
                break
            hi, lo, b = nh, nl, b + 1
        end = piv[b][0]
        if end - i1 < PLAT_MIN:
            continue
        shi = float(np.nanmax(px[i1:end + 1]))
        w = np.flatnonzero(px[end + 1:min(end + 1 + CAP, len(px))] > shi)
        if not w.size:
            continue
        bk = end + 1 + int(w[0])
        if str(dt[bk].date()) in seen:
            continue
        seen.add(str(dt[bk].date()))
        v1 = np.nanmean(vol[i0:i1 + 1])
        v2 = np.nanmean(vol[i1:end + 1])
        out.append(dict(
            一段起=str(dt[i0].date()), 一段顶=str(dt[i1].date()),
            一段幅度=px[i1] / px[i0] - 1, 一段天数=i1 - i0,
            平台天数=end - i1, 平台最深=1 - float(np.nanmin(px[i1:end + 1])) / shi,
            平台缩量比=v2 / v1 if v1 > 0 else np.nan,
            突破日=str(dt[bk].date()), 突破收盘=float(px[bk]), bk=bk,
            起涨前3年=px[i0] / px[max(i0 - 750, 0)] - 1,
            突破前3年=px[bk] / px[max(bk - 750, 0)] - 1,
            流通市值亿=float(mv[bk]) / 1e8 if np.isfinite(mv[bk]) else np.nan))
    return out


def fwd(t, n):
    if t + n >= N:
        return np.nan, np.nan
    seg = p[t + 1:t + n + 1]
    return float(np.nanmax(seg) / p[t] - 1), float(p[t + n] / p[t] - 1)


segs = segments(p, d)
print(f"\n{'='*W}\n三、三段结构实例({len(segs)} 次)与结果\n{'='*W}")
rows = []
for s in segs:
    pk6, en6 = fwd(s["bk"], 120)
    pk12, en12 = fwd(s["bk"], 250)
    s |= dict(峰值6=pk6, 期末6=en6, 峰值12=pk12, 期末12=en12)
    rows.append({k: v for k, v in s.items() if k != "bk"})
    print(f"\n  **突破 {s['突破日']}  收盘 {s['突破收盘']:.2f}**"
          f"   流通市值 {s['流通市值亿']:.0f} 亿")
    print(f"     第一段 {s['一段起']}→{s['一段顶']}  {s['一段幅度']:+.0%}"
          f"  {s['一段天数']} 日   起涨前 3 年 {s['起涨前3年']:+.1%}")
    print(f"     平台   {s['平台天数']} 日   最深 {s['平台最深']:.1%}"
          f"   缩量比 {s['平台缩量比']:.2f}")
    print(f"     突破前 3 年 **{s['突破前3年']:+.1%}**")
    print(f"     6 个月 峰值 {pk6:+.1%} 期末 {en6:+.1%}    "
          f"12 个月 峰值 {pk12:+.1%} 期末 {en12:+.1%}")

print(f"\n{'='*W}\n四、与旧筛选器(§93,legacy 口径)四次突破的对照\n{'='*W}")
OLD = {"2016-07-15": (+0.064, -0.115), "2017-07-21": (+0.159, +0.016),
       "2018-01-03": (-0.015, -0.307), "2024-01-08": (+1.011, +0.788)}
new = {s["突破日"]: s for s in segs}
print(f"{'突破日':<13}{'旧筛选器':<10}{'新分段':<10}{'6个月峰值':>10}{'6个月期末':>10}"
      f"{'突破前3年':>11}")
for dd in sorted(set(OLD) | set(new)):
    o = "✓" if dd in OLD else "—"
    nn = "✓" if dd in new else "—"
    if dd in new:
        s = new[dd]
        print(f"{dd:<13}{o:<10}{nn:<10}{s['峰值6']:>10.1%}{s['期末6']:>10.1%}"
              f"{s['突破前3年']:>11.1%}")
    else:
        pk, en = OLD[dd]
        print(f"{dd:<13}{o:<10}{nn:<10}{pk:>10.1%}{en:>10.1%}{'—':>11}")
print("\n  注:新旧两套算法的突破日不完全对齐(平台高的定义不同),"
      "2024-01-08 与 2024-01-10 是同一次突破。")

pd.DataFrame(rows).to_csv(f"{OUT}/case_yutong_13y_segments.csv", index=False)
pd.DataFrame(legs).to_csv(f"{OUT}/case_yutong_13y_legs.csv", index=False)
print(f"\n→ {OUT}/case_yutong_13y_segments.csv + _legs.csv")
