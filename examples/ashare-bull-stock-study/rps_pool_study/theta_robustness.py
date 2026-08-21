"""第一一一节:θ 的样本外验证 —— 组合超额是不是选参数选出来的(事前登记)

═══ 起因:唯一的正面结论建立在一个我挑的参数上 ═══
§110 是本项目 48 次事前判据检验里**唯一通过的一个**:
按 ZigZag 三段结构圈池、等权买入持有 24 个月,
**13 年年化 +16.49% vs 同日同市值随机 +12.53%,超额 +3.96pp,p=0.0000。**

**但 ZigZag 阈值 θ=10% 是我在宇通身上挑的**(§94 已注明):
挑选依据是「能否认出用户看图指认的两处三段结构(2014 年底、2024-01)」——
θ=10% 两处都认、15% 漏后者、20% 只剩两个、25% 一个都没有。
**挑的理由不是收益,但仍然是在样本内挑的。这一节就是来验它。**

**θ=10% 的结果我已看过,不是盲测;θ ∈ {8%, 12%, 15%} 一次都没跑过,是真盲测。**

═══ 口径(事前锁定)═══
  **与 §110 池① 逐字相同,只改 θ**:ZigZag θ ∈ **{8%, 10%, 12%, 15%}**
  三段模板(上涨腿 ≥30% → 平台 ±35.2% 带且 ≥60 日 → 突破,上限 250 日)、
  组合构建(信号次日等权买入、持有 500 日、日频等权平均)、
  对照(同日同市值五分位随机、100 组)**全部不变**

═══ 锚点(不过则全节作废;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **θ=10% 恒等复现 §110**:事件 **10,236**、组合年化 **+16.49%**、
     超额 **+3.96%**(容差 ±0.05pp)
  ③ **对照零校验**:各 θ 的对照组平均持仓只数与实盘逐一相同

═══ 事前判据(跑之前写死,不放宽;Bonferroni **0.05/4 = 0.0125**)═══
  **前置**:某 θ 的事件数 **< 1000** 则该 θ 不参与判据
  ① **稳健性**:四个 θ 中**至少 3 个**满足「超额 **≥ +3pp** 且 **p < 0.0125**」
     (门槛与 §110 逐字相同,不放宽)
  ② **不依赖单点**:**排除 θ=10% 后**,其余三个 θ 的**超额中位 ≥ +3pp**

**①② 都过 = 超额不是选参数选出来的,§110 的结论站得住。
只要有一条不过,§110 那 3.96pp 就必须标注为「可能是参数选择的产物」。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问**:四个 θ 里挑一个好的就宣称稳健 → **堵法:判据① 要求 ≥3/4,
判据② 直接把我挑的那个剔掉**;
θ 变小事件变多、组合更接近全市场,超额自然趋零 → **这正是要看的,不堵**。
**反问**:θ=15% 事件太少 → **堵法:前置 n≥1000,不足则不参与判据(不判非判负)**;
锚点误杀(已五次病根)→ **三个锚点全是恒等式,② 是对 §110 的恒等复现**。

═══ 事前预测(写下以便被证伪)═══
**①不通过、②不通过。**
**这是本项目我最没把握的一次预测。**
支持「会通过」的理由:θ 是按「能否认出结构」挑的,不是按收益挑的;
若三段结构本身是真的,换个阈值应当仍然有效。
支持「不通过」的理由:48 次检验只通过 1 次,基础概率极低;
θ 变化会显著改变「什么算平台」,8% 会把事件数推高到接近全市场(超额趋零),
15% 会让样本量掉到不够判。**我押不通过,但认为大约四六开。**
**若①② 都过,§110 就从「一个可能过拟合的数字」变成「一个稳健的发现」,
那是这个项目最有价值的产出 —— 我会明说我错了。**
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

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
r10 = R[R["theta"] == 0.10].iloc[0]
a2 = (int(r10["事件"]) == 10236 and abs(r10["年化"] - 0.1649) <= 0.0005
      and abs(r10["超额"] - 0.0396) <= 0.0005)
print(f"  {'✓' if a2 else '✗'} 锚点② θ=10% 恒等复现 §110:事件 {int(r10['事件']):,}"
      f"(期望 10,236)、年化 {r10['年化']:+.2%}(期望 +16.49%)、"
      f"超额 {r10['超额']:+.2%}(期望 +3.96%)")
if not a2:
    bad.append("锚点②")
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

R.to_csv(f"{OUT}/theta_robustness.csv", index=False)
print(f"\n→ {OUT}/theta_robustness.csv   ({time.time()-t0:.0f}s)")
