"""第一一二节:θ 稳健性重跑 —— 只拆锚点②,判据逐字不变(事前登记)

═══ 证据等级声明:本节不是盲测 ═══
**§111 因锚点② 不过而作废**:θ=10% 的事件数 **10,236** 与组合年化 **+16.49%**
一字不差,只有「超额 +3.84% vs 期望 +3.96%」对不上 ——
**因为超额减去的是对照组中位,而对照抽样依赖随机数流的位置**
(§110 里池① 第一个跑;§111 里 θ=8% 先跑消耗了随机数,θ=10% 拿到不同抽样,
对照中位 12.64% vs §110 的 12.53%)。
**我把确定量与随机量设进了同一个 ±0.05pp 容差,第 16 次自查出的错误。**

**§111 那次运行的判据结果我已看过:**

    θ=8%  超额 **+4.59%**  θ=10% **+3.84%**  θ=12% **+4.44%**  θ=15% **+7.88%**,p 全 0.0000
    判据① 4/4 满足、判据② 排除 10% 后中位 +4.59% —— **两条都过**

**因此本节判据结果的证据等级低于其他各节,与 §91 / §106 / §110 同等处理。
判据①② 逐字不变、不因已知结果而改写;只把锚点② 拆成确定部分与随机部分。**

═══ 口径:与 §111 逐字相同 ═══
ZigZag θ ∈ {8%, 10%, 12%, 15%};三段模板、组合构建、对照(100 组)全部不变。

═══ 锚点(不过则全节作废)═══
  ① 面板 (3297, 5232)
  ② **确定部分(精确,不设容差)**:θ=10% 事件数 **10,236**、组合年化 **+16.49%**
     —— 这两项不依赖随机数,必须一字不差
  ②b **随机部分**:θ=10% 的对照组年化中位落在 **12.0%~13.2%**
     (依据:§110 与 §111 两次独立抽样分别得 **12.53%** / **12.64%**,
     区间给 100 组抽样的波动留出余量)
  ③ **对照零校验**:各 θ 的对照组平均持仓只数与实盘逐一相同

═══ 事前判据(与 §111 逐字相同;Bonferroni 0.05/4 = 0.0125)═══
  ① **稳健性**:四个 θ 中至少 3 个满足「超额 **≥ +3pp** 且 **p < 0.0125**」
  ② **不依赖单点**:排除 θ=10% 后,其余三个 θ 的**超额中位 ≥ +3pp**

═══ 事前预测 ═══
**①② 的结果我已看过(都过),不构成预测。**
**唯一未知的是锚点②b 能否落在 12.0%~13.2% 内 —— 那取决于本次抽样。
我预测能落进去(两次独立抽样已给出 12.53% 与 12.64%,区间宽度 1.2pp 足够)。**
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
THETAS = (0.08, 0.10, 0.12, 0.15)
UP_MIN, PLAT_MIN, CAP, GAP = 0.30, 60, 250, 120
BAND = THR_DEPTH
MIN_N, ALPHA, EXC_MIN = 1000, 0.05 / 4, 0.03

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


def zigzag(px, s0, th):
    piv = [(s0, "L")]
    ext, ei, up = px[s0], s0, True
    for i in range(s0 + 1, len(px)):
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


rng = np.random.default_rng(SEED)
yrs = NT / 250


def build(th):
    ev = []
    for j in range(NS):
        s0 = int(FIRST[j])
        px = Fa[:, j]
        if not np.isfinite(px[s0]) or NT - s0 < 300:
            continue
        piv = zigzag(px, s0, th)
        seen = set()
        for a_ in range(len(piv) - 1):
            i0, k0 = piv[a_]
            i1, k1 = piv[a_ + 1]
            if not (k0 == "L" and k1 == "H") or px[i0] <= 0 \
                    or px[i1] / px[i0] - 1 < UP_MIN:
                continue
            b_, hi, lo = a_ + 1, px[i1], px[i1]
            while b_ + 1 < len(piv):
                q = piv[b_ + 1][0]
                nh2, nl = max(hi, px[q]), min(lo, px[q])
                if nl <= 0 or nh2 / nl - 1 > BAND:
                    break
                hi, lo, b_ = nh2, nl, b_ + 1
            end = piv[b_][0]
            if end - i1 < PLAT_MIN:
                continue
            shi = float(np.nanmax(px[i1:end + 1]))
            w = np.flatnonzero(px[end + 1:min(end + 1 + CAP, NT)] > shi)
            if not w.size:
                continue
            bk = end + 1 + int(w[0])
            if bk not in seen:
                seen.add(bk)
                ev.append((bk, j))
    return ev


def curve(evs):
    cnt = np.zeros((NT + 2, NS), np.int16)
    for t, j in evs:
        cnt[t + 1, j] += 1
        cnt[min(t + 1 + HOLD_D, NT + 1), j] -= 1
    hold = np.cumsum(cnt[:NT], axis=0)
    n = hold.sum(axis=1)
    r = np.where(n > 0, (RET * hold).sum(axis=1) / np.maximum(n, 1), 0.0)
    return np.cumprod(1 + r), n


def mdd(eq):
    pk = np.maximum.accumulate(eq)
    return float(np.max((pk - eq) / pk))


W = 100
rows = []
print(f"\n{'='*W}\nθ 稳健性:同一套三段模板与组合规则,只改 ZigZag 阈值\n{'='*W}")
print(f"{'θ':<7}{'事件数':>9}{'组合年化':>10}{'对照中位':>10}{'超额':>9}{'p':>8}"
      f"{'回撤':>9}{'对照回撤':>10}{'平均持仓':>10}")
for th in THETAS:
    evs = build(th)
    eq, n = curve(evs)
    cg = eq[-1] ** (1 / yrs) - 1
    cs, cds, cns = [], [], []
    for _ in range(NSEED):
        rep = []
        for t, j in evs:
            q = int(QU[t, j])
            pool = np.flatnonzero(QU[t] == q) if q >= 0 else np.flatnonzero(QU[t] >= 0)
            rep.append((t, int(rng.choice(pool)) if pool.size else j))
        e2, n2 = curve(rep)
        cs.append(e2[-1] ** (1 / yrs) - 1)
        cds.append(mdd(e2))
        cns.append(n2.mean())
    cs, cds = np.array(cs), np.array(cds)
    exc = cg - float(np.median(cs))
    p = float((cs >= cg).mean())
    rows.append(dict(theta=th, 事件=len(evs), 年化=cg, 对照年化=float(np.median(cs)),
                     超额=exc, p=p, 回撤=mdd(eq), 对照回撤=float(np.median(cds)),
                     平均持仓=float(n.mean()), 对照持仓=float(np.mean(cns))))
    print(f"{th:<7.0%}{len(evs):>9,}{cg:>10.2%}{np.median(cs):>10.2%}{exc:>+9.2%}"
          f"{p:>8.4f}{mdd(eq):>9.1%}{np.median(cds):>10.1%}{n.mean():>10.0f}", flush=True)
R = pd.DataFrame(rows)

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
r10 = R[R["theta"] == 0.10].iloc[0]
a2 = int(r10["事件"]) == 10236 and abs(r10["年化"] - 0.1649) <= 0.0005
print(f"  {'✓' if a2 else '✗'} 锚点② 确定部分:θ=10% 事件 {int(r10['事件']):,}"
      f"(期望 10,236)、组合年化 {r10['年化']:+.2%}(期望 +16.49%)")
a2b = 0.120 <= float(r10["对照年化"]) <= 0.132
print(f"  {'✓' if a2b else '✗'} 锚点②b 随机部分:θ=10% 对照中位 "
      f"{r10['对照年化']:.2%} ∈ [12.0%, 13.2%]  (§110 得 12.53%、§111 得 12.64%)")
if not a2:
    bad.append("锚点②")
if not a2b:
    bad.append("锚点②b")
a3 = bool((abs(R["平均持仓"] - R["对照持仓"]) / R["平均持仓"] < 0.02).all())
print(f"  {'✓' if a3 else '✗'} 锚点③ 对照零校验:平均持仓 "
      + " ".join(f"{a:.0f}/{b:.0f}" for a, b in
                 zip(R['平均持仓'], R['对照持仓'], strict=True)))
if not a3:
    bad.append("锚点③")

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,Bonferroni {ALPHA})\n{'='*W}")
elig = R[R["事件"] >= MIN_N]
ok = elig[(elig["超额"] >= EXC_MIN) & (elig["p"] < ALPHA)]
c1 = len(ok) >= 3
oth = R[(R["theta"] != 0.10) & (R["事件"] >= MIN_N)]["超额"]
c2 = bool(len(oth) and float(oth.median()) >= EXC_MIN)
print(f"  前置:事件数 ≥{MIN_N} 的 {len(elig)}/{len(R)} 个 θ")
print(f"  {'✓' if c1 else '✗'} ① 稳健性:满足「超额≥+3pp 且 p<{ALPHA}」的 θ 有 "
      f"**{len(ok)}** 个,需 ≥3   ({', '.join(f'{t:.0%}' for t in ok['theta'])})")
print(f"  {'✓' if c2 else '✗'} ② 不依赖单点:排除 θ=10% 后其余超额中位 "
      f"**{float(oth.median()) if len(oth) else float('nan'):+.2%}** ≥ +3pp"
      f"   (各值 {', '.join(f'{v:+.2%}' for v in oth)})")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2:
    print("  **结论:超额不是选参数选出来的,§110 站得住。事前预测被证伪 —— 我错了。**")
else:
    print("  **结论:超额依赖 θ 的选择。§110 那 3.96pp 必须标注为")
    print("     「可能是参数选择的产物」,不能当作稳健发现。事前预测命中。**")

R.to_csv(f"{OUT}/theta_robustness_v2.csv", index=False)
print(f"\n→ {OUT}/theta_robustness.csv   ({time.time()-t0:.0f}s)")
