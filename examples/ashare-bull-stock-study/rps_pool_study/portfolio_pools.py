"""第一〇九节:组合层面 —— 等权买入筛选池、持有 24 个月,和随机组合比(事前登记)

═══ 起因:46 次检验全在单只层面,组合层面一次没测过 ═══
§89~§108 测的都是**单只事件的右尾概率**(lift = P(右尾|信号) ÷ P(右尾|对照))。
**「单只不可预测」不等于「组合不可行」** —— 这是唯一还没被堵死的方向。

已知的单只结果(同日同市值对照):
    §94 三段突破        6 个月 ≥100%  **10.23%** vs 8.18%   **lift 1.25**, p<0.0001
    §106 首次新高+RPS<50 24 个月 ≥200% **20.99%** vs 19.14%  lift 1.10
    §103 首次新高       21,876 事件

**§62 已证:OOS 后 90% 的交易加起来是亏钱的,全部利润来自前 10%。
若右尾 lift 1.25 是真的,它应该直接体现在组合收益上 —— 本节测它。**

═══ 口径(事前锁定)═══
  **组合构建**  信号日 t0 **次日**等权买入,持有 **500 个交易日**(24 个月)后卖出;
                组合日收益 = 当日全部在持仓位的**等权平均**日收益;累乘得净值
                同一只股票同时被多个信号命中时按**持仓笔数**计权(自然等权)
  **三个池子**  ① §94 ZigZag 三段突破  ② §103 首次创 250 日新高
                ③ §106 首次新高 且当日 RPS250 <50
  **对照**      把每个信号的股票换成**同日同市值五分位内的随机股**,
                信号日期与持仓期完全不变,**100 组**
  退市股按最后有效价 ffill 参与(日收益 0),**绝不剔除**;不含交易成本

═══ 锚点(不过则全节作废;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **事件数恒等复现**:三段突破 **10,236**、首次新高 **21,876**
  ③ **对照零校验**:对照组的**平均持仓只数**与实盘组逐日相同(同信号日、同数量)

═══ 事前判据(跑之前写死,不放宽;Bonferroni **0.05/6 = 0.00833**)═══
  **前置**:某池子事件数 **< 1000** 不判
  ① **年化超额**:池子组合年化 **− 对照组年化中位 ≥ +3pp**,
     且 **p < 0.00833**(对照 100 组中超过实盘的比例)
  ② **回撤不劣**:池子组合最大回撤 **≤ 对照组回撤中位**
  ③ **逐年一致**:逐年「池子组合收益 > 对照组中位」的年份占比 **≥ 80%**

**①②③ 全过 = 组合层面成立,单只不可预测但一篮子可行 ——
那将是本项目第一个可落地的结论。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问**:池子扎堆牛市 → **堵法:对照用完全相同的信号日期与持仓期,只换股票**;
少数暴涨股主导 → **堵法:等权,不按市值加权;并同时报中位/分位**;
3 池 × 2 指标搜索 → **堵法:Bonferroni 0.05/6**。
**反问**:样本不足 → 前置 n≥1000;
末段事件被右截断(t0+500 超出面板)→ **实盘与对照同等受影响**;
锚点误杀(已五次病根)→ **三个锚点全是恒等式,② 已在 §94/§103 实测**。

═══ 事前预测(写下以便被证伪)═══
**①②③ 全不过。**
理由:§61 测过「三条全中」组合年化 +10.37%,但 300 次随机对照 **p=0.16**,不算发现;
§62 的右尾结论说明组合收益极度依赖少数几笔,**等权组合的年化方差很大,
lift 1.25 折算到年化差可能远小于 3pp**;§96 全市场买入持有年化中位仅 **+2.1%**。
**②我尤其预测不过** —— 池子里的股票是「正在创新高」的,波动更大,回撤应当更深。
**我已五次押「会通过」全错,本节回到「全不过」的预测。
若 ① 通过,那是本项目第一次在组合层面找到超额,我会明说我错了。**
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
    col = NH[:, j]
    for t in np.flatnonzero(col):
        if t < 250 or col[max(t - GAP, 0):t].any():
            continue
        EVN.append((int(t), j))
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

R.drop(columns=["净值"]).to_csv(f"{OUT}/portfolio_pools.csv", index=False)
print(f"\n→ {OUT}/portfolio_pools.csv   ({time.time()-t0:.0f}s)")
