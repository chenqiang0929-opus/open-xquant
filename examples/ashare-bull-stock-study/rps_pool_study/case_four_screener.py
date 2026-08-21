"""四只案例股 vs 筛选器:宇通/匠心/嘉益/泰格 能不能被找到

**描述性,不设判据。**用 §94 已锁定的 ZigZag 三段检测器(θ=10%)直接跑这四只,
看它在每只身上发出过什么信号、信号之后发生了什么。
全样本基准取自 §94/§101(已落库):三段突破 6 个月 ≥100% = **10.23%**;
24 个月 ≥200% 各档 4.80%~9.19%。
"""
import os

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
TH, UP_MIN, BAND, PLAT_MIN, CAP = 0.10, 0.30, 0.352, 60, 250
CASES = [("600066", "宇通客车"), ("301061", "匠心家居"),
         ("301004", "嘉益股份"), ("300347", "泰格医药")]


def zigzag(px, s0):
    piv = [(s0, "L")]
    ext, ei, up = px[s0], s0, True
    for i in range(s0 + 1, len(px)):
        if up:
            if px[i] > ext:
                ext, ei = px[i], i
            elif px[i] <= ext * (1 - TH):
                piv.append((ei, "H"))
                ext, ei, up = px[i], i, False
        else:
            if px[i] < ext:
                ext, ei = px[i], i
            elif px[i] >= ext * (1 + TH):
                piv.append((ei, "L"))
                ext, ei, up = px[i], i, True
    piv.append((ei, "H" if up else "L"))
    return piv


W = 104
rows = []
for code, name in CASES:
    x = pd.read_parquet(f"{DATA}/{code}.parquet", columns=["close"])
    if getattr(x.index, "tz", None) is not None:
        x.index = x.index.tz_localize(None)
    x = x[x.index <= "2026-08-03"]
    p = x["close"].ffill().to_numpy(float)
    d = x.index
    n = len(p)
    piv = zigzag(p, 0)
    ev, seen = [], set()
    for a in range(len(piv) - 1):
        i0, k0 = piv[a]
        i1, k1 = piv[a + 1]
        if not (k0 == "L" and k1 == "H") or p[i1] / p[i0] - 1 < UP_MIN:
            continue
        b, hi, lo = a + 1, p[i1], p[i1]
        while b + 1 < len(piv):
            q = piv[b + 1][0]
            nh, nl = max(hi, p[q]), min(lo, p[q])
            if nh / nl - 1 > BAND:
                break
            hi, lo, b = nh, nl, b + 1
        end = piv[b][0]
        if end - i1 < PLAT_MIN:
            continue
        shi = float(np.nanmax(p[i1:end + 1]))
        w = np.flatnonzero(p[end + 1:min(end + 1 + CAP, n)] > shi)
        if not w.size:
            continue
        bk = end + 1 + int(w[0])
        if bk in seen:
            continue
        seen.add(bk)
        ev.append((bk, i0, i1))
    # 全期最大上升波段
    best = max(((p[piv[i+1][0]] / p[piv[i][0]] - 1, piv[i][0], piv[i+1][0])
                for i in range(len(piv) - 1) if p[piv[i+1][0]] > p[piv[i][0]]),
               default=(np.nan, 0, 0))
    print(f"\n{'='*W}\n{name} {code}   {d[0].date()} ~ {d[-1].date()}   "
          f"最低 {p.min():.2f} → 最高 {p.max():.2f}  ×{p.max()/p.min():.1f}\n{'='*W}")
    print(f"  全期最大上升波段:{d[best[1]].date()} → {d[best[2]].date()}  **{best[0]:+.0%}**")
    print(f"  **筛选器三段突破信号:{len(ev)} 次**")
    if ev:
        print(f"    {'突破日':<12}{'收盘':>8}{'6月峰值':>10}{'12月峰值':>10}"
              f"{'24月峰值':>10}{'≥100%(6月)':>12}{'≥200%(24月)':>13}")
    hit6 = hit24 = 0
    for bk, i0, i1 in ev:
        r = {}
        for h, k in ((120, "6"), (250, "12"), (500, "24")):
            r[k] = (np.nanmax(p[bk+1:bk+h+1]) / p[bk] - 1) if bk + h < n else np.nan
        h6 = np.isfinite(r["6"]) and r["6"] >= 1.0
        h24 = np.isfinite(r["24"]) and r["24"] >= 2.0
        hit6 += h6
        hit24 += h24
        print(f"    {str(d[bk].date()):<12}{p[bk]:>8.2f}"
              + "".join(f"{(f'{r[k]:+.1%}' if np.isfinite(r[k]) else '—'):>10}"
                        for k in ("6", "12", "24"))
              + f"{('✓' if h6 else '✗'):>12}{('✓' if h24 else '✗'):>13}")
        rows.append(dict(股票=name, 代码=code, 突破日=str(d[bk].date()), 收盘=p[bk],
                         峰值6=r["6"], 峰值12=r["12"], 峰值24=r["24"]))
    # 信号是否覆盖了最大波段
    cov = [str(d[bk].date()) for bk, _, _ in ev if best[1] <= bk <= best[2]]
    print(f"    命中:6 个月 ≥100% **{hit6}/{len(ev)}**   24 个月 ≥200% **{hit24}/{len(ev)}**")
    print(f"    最大波段区间内是否有信号:**{('有 → ' + ', '.join(cov)) if cov else '无'}**")

R = pd.DataFrame(rows)
print(f"\n{'='*W}\n汇总:四只共 {len(R)} 次信号\n{'='*W}")
v6 = R["峰值6"].dropna()
v24 = R["峰值24"].dropna()
print(f"  6 个月 ≥100%:{int((v6>=1).sum())}/{len(v6)} = {(v6>=1).mean():.1%}"
      f"   (§94 全样本 9,598 事件 = **10.23%**)")
print(f"  24 个月 ≥200%:{int((v24>=2).sum())}/{len(v24)} = {(v24>=2).mean():.1%}"
      f"   (§101 全样本各档 4.80%~9.19%)")
print(f"  6 个月峰值中位 {v6.median():+.1%}   24 个月峰值中位 {v24.median():+.1%}")
R.to_csv(f"{OUT}/case_four_screener.csv", index=False)
print(f"\n→ {OUT}/case_four_screener.csv")
