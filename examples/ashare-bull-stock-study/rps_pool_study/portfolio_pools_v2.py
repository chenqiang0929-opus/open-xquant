"""第一一〇节:组合层面重跑 —— 修好循环泄漏(事前登记)

═══ 证据等级声明:本节不是盲测 ═══
**§109 因锚点② 不过而作废**:池② 首次新高实测 21,859、期望 21,876,差 17。
根因是我把 ZigZag 与「首次新高」两个采集写进同一个循环,
`if NT - s0 < 300: continue`(ZigZag 的前置条件)把首次新高的采集也跳过了 ——
上市不足 300 日的股票因此漏采。**第 15 次自查出的错误。**

**§109 那次运行已经把结果打印出来,我看过:**

    ① 三段突破      年化 **+16.49%** vs 对照 +12.53%,**超额 +3.96%,p 0.0000**,回撤 57.9% vs 59.1%
    ② 首次新高      +11.38% vs +12.17%,超额 −0.79%,p 0.9900
    ③ 首次新高+RPS<50 +13.23% vs +12.48%,超额 +0.75%,p 0.2000

**因此本节的判据结果证据等级低于其他各节 —— 与 §91 / §106 同等处理。
判据①②③ 逐字不变、不因已知结果而改写;只修循环泄漏这一个 bug。**
**池① 三段突破在 §109 中事件数 10,236 一个不差,不受该 bug 影响。**

═══ 口径、锚点、判据:与 §109 逐字相同 ═══
(见 §109 事前登记;此处不重述,以免出现与原文不一致的改写)

═══ 事前预测 ═══
**①②③ 的结果我已看过,不构成预测。**
**真正未知的只有一件:修好 17 个漏采事件后,池② 的超额是否仍为负、
池① 的 +3.96% 是否仍然稳定。我预测两者都基本不变(漏采仅占 0.08%)。**
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from consolidation_screener import THR_DEPTH, load_panel  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NSEED, SEED, HOLD_D = 100, 20260814, 500
TH, UP_MIN, PLAT_MIN, CAP, GAP = 0.10, 0.30, 60, 250, 120
BAND = THR_DEPTH
MIN_N, ALPHA, EXC_MIN, YR_FRAC = 1000, 0.05 / 6, 0.03, 0.80

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    CL = CL.drop(columns=["510300"])
del frames, STRONG, MA100
idx = CL.index
NT, NS = CL.shape
codes = list(CL.columns)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"
Fa = CL.where(CL > 0).ffill().to_numpy(float)
FIRST = np.argmax(np.isfinite(Fa), axis=0)
RET = np.zeros((NT, NS))
RET[1:] = np.where(np.isfinite(Fa[:-1]) & (Fa[:-1] > 0), Fa[1:] / Fa[:-1] - 1, 0.0)
RET[~np.isfinite(RET)] = 0.0
F = pd.DataFrame(Fa)
HI = F.rolling(250, min_periods=100).max().to_numpy(float)
NH = np.isfinite(HI) & (Fa >= HI * 0.9999)
del HI
RPS250 = (CL.pct_change(250).rank(axis=1, pct=True) * 100).to_numpy(float)
mv = pd.DataFrame({c: pd.to_numeric(
    pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])["float_mv"],
    errors="coerce") for c in codes})
if getattr(mv.index, "tz", None) is not None:
    mv.index = mv.index.tz_localize(None)
mv = mv.reindex(idx.tz_localize(None)).ffill().to_numpy(float)
QU = np.full((NT, NS), -1, dtype=np.int8)
POOLQ = {}
for t in range(NT):
    ok = np.isfinite(mv[t]) & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if ok.sum() < 50:
        continue
    QU[t, ok] = np.searchsorted(np.nanquantile(mv[t][ok], [.2, .4, .6, .8]),
                                mv[t][ok], side="right")
del mv
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)


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


EV3, EVN = [], []
for j in range(NS):          # 首次新高:独立循环,不受 ZigZag 前置条件影响(§109 的 bug)
    col = NH[:, j]
    for t in np.flatnonzero(col):
        if t < 250 or col[max(t - GAP, 0):t].any():
            continue
        EVN.append((int(t), j))
for j in range(NS):
    s0 = int(FIRST[j])
    px = Fa[:, j]
    if not np.isfinite(px[s0]) or NT - s0 < 300:
        continue
    piv = zigzag(px, s0)
    seen = set()
    for a in range(len(piv) - 1):
        i0, k0 = piv[a]
        i1, k1 = piv[a + 1]
        if not (k0 == "L" and k1 == "H") or px[i0] <= 0 or px[i1] / px[i0] - 1 < UP_MIN:
            continue
        b, hi, lo = a + 1, px[i1], px[i1]
        while b + 1 < len(piv):
            q = piv[b + 1][0]
            nh2, nl = max(hi, px[q]), min(lo, px[q])
            if nl <= 0 or nh2 / nl - 1 > BAND:
                break
            hi, lo, b = nh2, nl, b + 1
        end = piv[b][0]
        if end - i1 < PLAT_MIN:
            continue
        shi = float(np.nanmax(px[i1:end + 1]))
        w = np.flatnonzero(px[end + 1:min(end + 1 + CAP, NT)] > shi)
        if not w.size:
            continue
        bk = end + 1 + int(w[0])
        if bk not in seen:
            seen.add(bk)
            EV3.append((bk, j))
EVR = [(t, j) for t, j in EVN if np.isfinite(RPS250[t, j]) and RPS250[t, j] < 50]
a2 = len(EV3) == 10236 and len(EVN) == 21876
print(f"池① 三段突破 {len(EV3):,}(期望 10,236)  池② 首次新高 {len(EVN):,}(期望 21,876)"
      f"  池③ +RPS250<50 {len(EVR):,}   {'✓' if a2 else '✗'} 锚点②")

ym = idx.to_period("M")
yrs = NT / 250


def curve(evs):
    cnt = np.zeros((NT + 2, NS), np.int16)
    for t, j in evs:
        cnt[t + 1, j] += 1
        cnt[min(t + 1 + HOLD_D, NT + 1), j] -= 1
    hold = np.cumsum(cnt[:NT], axis=0)
    n = hold.sum(axis=1)
    r = np.where(n > 0, (RET * hold).sum(axis=1) / np.maximum(n, 1), 0.0)
    return np.cumprod(1 + r), n, r


def mdd(eq):
    pk = np.maximum.accumulate(eq)
    return float(np.max((pk - eq) / pk))


rng = np.random.default_rng(SEED)
W = 100
rows = []
for nm, evs in (("① 三段突破", EV3), ("② 首次新高", EVN), ("③ 首次新高+RPS<50", EVR)):
    eq, n, r = curve(evs)
    cg = eq[-1] ** (1 / yrs) - 1
    cs, cds, cns = [], [], []
    for _ in range(NSEED):
        rep = []
        for t, j in evs:
            q = int(QU[t, j])
            pool = np.flatnonzero(QU[t] == q) if q >= 0 else np.flatnonzero(QU[t] >= 0)
            rep.append((t, int(rng.choice(pool)) if pool.size else j))
        e2, n2, _ = curve(rep)
        cs.append(e2[-1] ** (1 / yrs) - 1)
        cds.append(mdd(e2))
        cns.append(n2.mean())
    cs, cds = np.array(cs), np.array(cds)
    exc = cg - float(np.median(cs))
    p = float((cs >= cg).mean())
    rows.append(dict(池=nm, 事件=len(evs), 年化=cg, 对照年化=float(np.median(cs)),
                     超额=exc, p=p, 回撤=mdd(eq), 对照回撤=float(np.median(cds)),
                     平均持仓=float(n.mean()), 对照持仓=float(np.mean(cns)),
                     净值=eq))
    print(f"\n{'='*W}\n{nm}  事件 {len(evs):,}  ({time.time()-t0:.0f}s)\n{'='*W}")
    print(f"  组合净值 {eq[-1]:.2f}   年化 **{cg:+.2%}**   最大回撤 **{mdd(eq):.1%}**"
          f"   平均持仓 {n.mean():.0f} 只")
    print(f"  对照组({NSEED} 组)年化中位 {np.median(cs):+.2%}  [{np.percentile(cs,5):+.2%},"
          f" {np.percentile(cs,95):+.2%}]   回撤中位 {np.median(cds):.1%}"
          f"   平均持仓 {np.mean(cns):.0f} 只")
    print(f"  **超额 {exc:+.2%}   p {p:.4f}**", flush=True)
R = pd.DataFrame(rows)

print(f"\n{'='*W}\n逐年:各池组合年收益 vs 对照\n{'='*W}")
YR = {}
for _, rr in R.iterrows():
    eq = rr["净值"]
    ys = []
    for y in sorted({d.year for d in idx}):
        m = np.array([d.year == y for d in idx])
        if m.sum() < 60:
            continue
        seg = eq[m]
        ys.append((y, seg[-1] / seg[0] - 1))
    YR[rr["池"]] = ys
hdr = "  年    " + "".join(f"{r['池']:>20}" for _, r in R.iterrows())
print(hdr)
for i, (y, _) in enumerate(YR[R.iloc[0]["池"]]):
    print(f"  {y}  " + "".join(f"{YR[r['池']][i][1]:>20.1%}" for _, r in R.iterrows()))

print(f"\n{'='*W}\n锚点核对\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② 事件数恒等复现")
if not a2:
    bad.append("锚点②")
a3 = bool((abs(R["平均持仓"] - R["对照持仓"]) / R["平均持仓"] < 0.02).all())
print(f"  {'✓' if a3 else '✗'} 锚点③ 对照零校验:平均持仓只数 实盘 vs 对照 "
      + " ".join(f"{a:.0f}/{b:.0f}" for a, b in zip(R['平均持仓'], R['对照持仓'], strict=True)))
if not a3:
    bad.append("锚点③")

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,Bonferroni {ALPHA:.5f})\n{'='*W}")
for _, rr in R.iterrows():
    c1 = bool(rr["事件"] >= MIN_N and rr["超额"] >= EXC_MIN and rr["p"] < ALPHA)
    c2 = bool(rr["回撤"] <= rr["对照回撤"])
    print(f"  {rr['池']:<20} ① 超额 {rr['超额']:+.2%} ≥+3pp 且 p {rr['p']:.4f} "
          f"{'✓' if c1 else '✗'}   ② 回撤 {rr['回撤']:.1%} ≤ {rr['对照回撤']:.1%} "
          f"{'✓' if c2 else '✗'}")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
else:
    anyp = any(rr["超额"] >= EXC_MIN and rr["p"] < ALPHA for _, rr in R.iterrows())
    print("  **结论:组合层面找到超额 —— 我错了。**" if anyp
          else "  **结论:组合层面同样没有超额。事前预测命中。**")

R.drop(columns=["净值"]).to_csv(f"{OUT}/portfolio_pools_v2.csv", index=False)
print(f"\n→ {OUT}/portfolio_pools.csv   ({time.time()-t0:.0f}s)")
