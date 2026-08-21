"""第一〇七节:陶博士「120/200/250 多头排列」—— 必要性 vs 充分性,分开测(事前登记)

═══ 起因:用户问「为什么陶博士说 N 倍股特征可以量化」═══
《230416 N倍股的共同技术特征》(陶博士2006)原文:

> 「N倍股的长期走势,都有一个共同的技术特征,在 N 倍的涨升途中,中长期均线必然是
>  多头排列。**提醒注意,反过来并不成立的,即中长期均线是多头排列的股票,
>  不一定会涨 N 倍的。**」
> 「长期均线 120日线、200日线和250日线的多头排列特征,是所有 N 倍股的一个共同的
>  技术特征。」

**他讲的是必要条件,并且自己明确否掉了充分性。本项目 45 次检验测的一直是充分性
(lift = 有信号的右尾概率 ÷ 对照)。两者不矛盾 —— 本节把它们分开量化。**

**文章给了可验证的具体断言,已预先验证(锚点②)**:
宁德时代 300750 「2020 年初…终于形成了多头排列」→ 实测首次形成日 **2020-02-24**;
「2020 年 4 月 14 日的口袋支点具有标志性意义」→ 实测该日 120=**56.9** > 200=**49.1**
> 250=**46.9**,多头排列成立,**其后 24 个月峰值 +446%**。**描述完全准确。**

**但同一只股票的多头排列首次形成共 5 次:2019-06-20(+713%)、2020-02-24(+346%)、
2024-10-11(+99%)、2025-07-31(+77%)、2025-09-03(+51%) —— 2 次 N 倍级别,3 次不是。**

═══ 口径(事前锁定)═══
  **多头排列** = MA120 > MA200 > MA250(日线收盘均线,均需满窗口)
  **A 必要性**  「N 倍涨升段」= 某日 t 起 500 日内峰值 ≥200%,且 t 为该段起点
                (前 60 日不满足此条件);计算 **[t, 峰值日] 区间内多头排列天数占比**
  **B 充分性**  事件 = 多头排列**首次形成日**(此前 60 日未处于多头排列);
                入场 = 该日收盘;前瞻 24 个月(500 日)峰值 **≥200%**
                对照A 同日同市值五分位随机;对照B 同日同市值 **且当日也处于多头排列**
  退市股 ffill 参与,绝不剔除

═══ 锚点(不过则全节作废;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **宁德时代 300750 恒等复现**(已预验证):多头排列首次形成日**含 2020-02-24**;
     2020-04-14 处于多头排列且其后 24 个月峰值 **+446%**(±1pp)
  ③ **恒等零校验**:对照A/B 的中位命中率 vs 同格总体命中率,差 ≤ 3pp

═══ 事前判据(跑之前写死,不放宽;Bonferroni **0.05/6 = 0.00833**)═══
  **前置**:A 段数 <300 不判;B 事件数 <300 不判
  A① **必要性(陶博士的主张)**:N 倍涨升段内多头排列天数占比的**中位 ≥ 80%**
  B① **充分性**:首次形成后 24 个月 ≥200% 对**对照A** 的 **lift ≥ 1.3 且 p < 0.00833**
  B② **充分性隔离动量**:同上对**对照B** 的 **lift ≥ 1.3 且 p < 0.00833**

**A① 过 = 陶博士的必要条件成立(所有 N 倍股确实都有这个特征);
B①② 不过 = 他自己说的「反过来不成立」被量化证实。两者可以同时为真。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问**:A① 可能恒真 —— 长期上涨本就会推高短均线 → **堵法:这正是要量化的
「有多必然」,80% 是明确门槛,且同时给出全市场基础占比作参照**;
B 事件扎堆牛市 → **堵法:对照按同日同市值抽 + 对照B 用同状态股**。
**反问**:样本不足 → 前置 n≥300;
锚点误杀(已五次病根)→ **三个锚点全是恒等式,② 已单只预验证可达**。

═══ 事前预测(写下以便被证伪)═══
**A① 通过、B① 不通过、B② 不通过。**
理由:A① 是必要条件,且宁德实测印证,**逻辑上「涨了 2 倍还没形成多头排列」很难**;
B①② 不过是因为 §89~§106 十余次一致显示任何状态量对同日同类对照都归零,
**而且陶博士本人就写着「反过来并不成立」**。
**若 A① 不过,说明连「N 倍股都多头排列」这个必要条件都不成立,陶博士错了,我也错了;
若 B①② 通过,说明多头排列有真实预测力,陶博士过于谦虚,我错了。**
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
from consolidation_screener import load_panel  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
NSEED, SEED, H = 200, 20260814, 500
MIN_N, ALPHA, LIFT_MIN, NEC_MIN, GAPD = 300, 0.05 / 6, 1.3, 0.80, 60

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
F = pd.DataFrame(Fa)
M120, M200, M250 = (F.rolling(n, min_periods=n).mean().to_numpy(float)
                    for n in (120, 200, 250))
BULL = np.isfinite(M250) & (M120 > M200) & (M200 > M250)
print(f"全市场任一日处于多头排列的比例 **{np.nanmean(BULL[np.isfinite(Fa)]):.1%}**")

# 未来 500 日峰值
m = F[::-1].rolling(H, min_periods=1).max().to_numpy(float)[::-1]
PK = np.full((NT, NS), np.nan)
PK[:-1] = m[1:]
PK = (PK / Fa - 1.0).astype(np.float32)
PK[NT - H:] = np.nan
ARG = np.full((NT, NS), -1, dtype=np.int32)
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)

# ── A 必要性 ──
segs = []
for j in range(NS):
    ok = np.isfinite(PK[:, j]) & (PK[:, j] >= 2.0)
    prev = False
    for t in np.flatnonzero(ok):
        if prev and t - last <= GAPD:      # noqa: F821
            last = t
            continue
        seg = Fa[t + 1:t + H + 1, j]
        pkd = t + 1 + int(np.nanargmax(seg))
        b = BULL[t:pkd + 1, j]
        segs.append((t, j, pkd - t, float(np.nanmean(b)) if b.size else np.nan))
        prev, last = True, t
A = pd.DataFrame(segs, columns=["t", "j", "长度", "多头占比"])
W = 96
print(f"\n{'='*W}\nA 必要性:N 倍涨升段(500 日内峰值 ≥200%)内的多头排列天数占比\n{'='*W}")
print(f"  段数 **{len(A):,}**   中位长度 {A['长度'].median():.0f} 交易日")
v = A["多头占比"].dropna()
print(f"  多头排列天数占比:中位 **{v.median():.1%}**  均值 {v.mean():.1%}  "
      f"25分位 {v.quantile(.25):.1%}  75分位 {v.quantile(.75):.1%}")
print(f"  占比 ≥80% 的段 **{(v>=0.8).mean():.1%}**   ≥50% 的段 {(v>=0.5).mean():.1%}"
      f"   =0% 的段 **{(v==0).mean():.1%}**")

# ── B 充分性 ──
ev = []
for j in range(NS):
    col = BULL[:, j]
    for t in np.flatnonzero(col):
        if col[max(t - GAPD, 0):t].any():
            continue
        ev.append((int(t), j))
JD = codes.index("300750")
jf = [str(idx[t].date()) for t, j in ev if j == JD]
t414 = int(np.flatnonzero(idx == pd.Timestamp("2020-04-14", tz=idx.tz))[0])
a2 = ("2020-02-24" in jf and bool(BULL[t414, JD])
      and abs(float(PK[t414, JD]) - 4.46) <= 0.01)
print(f"\n  {'✓' if a2 else '✗'} 锚点② 宁德:首次形成含 2020-02-24 {jf[:3]};"
      f" 2020-04-14 多头={BULL[t414,JD]} 24月峰值 {float(PK[t414,JD]):+.1%}")

mv = pd.DataFrame({c: pd.to_numeric(
    pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])["float_mv"],
    errors="coerce") for c in codes})
if getattr(mv.index, "tz", None) is not None:
    mv.index = mv.index.tz_localize(None)
mv = mv.reindex(idx.tz_localize(None)).ffill().to_numpy(float)
QU = np.full((NT, NS), -1, dtype=np.int8)
for t in range(NT):
    ok2 = np.isfinite(mv[t]) & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if ok2.sum() < 50:
        continue
    QU[t, ok2] = np.searchsorted(np.nanquantile(mv[t][ok2], [.2, .4, .6, .8]),
                                 mv[t][ok2], side="right")
del mv
rng = np.random.default_rng(SEED)


def ctl(sub, bullonly):
    cn = {}
    for t, j in sub:
        q = int(QU[t, j])
        if q >= 0:
            cn[(t, q)] = cn.get((t, q), 0) + 1
    hit, tot, th, tn = np.zeros(NSEED), 0, 0.0, 0
    for (t, q), k in cn.items():
        pool = np.flatnonzero((QU[t] == q) & np.isfinite(PK[t]))
        if bullonly:
            pool = pool[BULL[t, pool]]
        if pool.size == 0:
            continue
        x = PK[t, pool] >= 2.0
        th += float(x.mean()) * k
        tn += k
        hit += x[rng.integers(0, pool.size, size=(NSEED, k))].sum(axis=1)
        tot += k
    return (hit / tot, th / tn) if tot else (np.array([]), np.nan)


sub = [(t, j) for t, j in ev if np.isfinite(PK[t, j])]
vv = np.array([PK[t, j] for t, j in sub]) >= 2.0
obs = float(vv.mean())
ca, tha = ctl(sub, False)
cb, thb = ctl(sub, True)
ra = float(np.median(ca)) if ca.size else np.nan
rb = float(np.median(cb)) if cb.size else np.nan
la, lb = obs / ra if ra > 0 else np.nan, obs / rb if rb > 0 else np.nan
pa = float((ca >= obs).mean()) if ca.size else np.nan
pb = float((cb >= obs).mean()) if cb.size else np.nan
print(f"\n{'='*W}\nB 充分性:多头排列首次形成日入场,24 个月峰值 ≥200%\n{'='*W}")
print(f"  事件 **{len(sub):,}**   命中 **{obs:.2%}**")
print(f"  对照A 同市值随机          {ra:.2%}   lift **{la:.2f}**   p {pa:.4f}   "
      f"零校验 {abs(ra-tha):.2%}")
print(f"  对照B 同市值+当日也多头排列 {rb:.2%}   lift **{lb:.2f}**   p {pb:.4f}   "
      f"零校验 {abs(rb-thb):.2%}")

print(f"\n{'='*W}\n锚点核对\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
print(f"  {'✓' if a2 else '✗'} 锚点② 宁德时代恒等复现")
a3 = abs(ra - tha) <= 0.03 and abs(rb - thb) <= 0.03
print(f"  {'✓' if a3 else '✗'} 锚点③ 恒等零校验")
for ok3, nm in ((a2, "锚点②"), (a3, "锚点③")):
    if not ok3:
        bad.append(nm)

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,Bonferroni {ALPHA:.5f})\n{'='*W}")
c1 = bool(len(A) >= MIN_N and v.median() >= NEC_MIN)
c2 = bool(len(sub) >= MIN_N and la >= LIFT_MIN and pa < ALPHA)
c3 = bool(len(sub) >= MIN_N and lb >= LIFT_MIN and pb < ALPHA)
print(f"  {'✓' if c1 else '✗'} A① 必要性:N 倍段内多头占比中位 **{v.median():.1%}** ≥ 80%")
print(f"  {'✓' if c2 else '✗'} B① 充分性 对照A lift {la:.2f} ≥1.3 且 p {pa:.4f}")
print(f"  {'✓' if c3 else '✗'} B② 充分性 对照B lift {lb:.2f} ≥1.3 且 p {pb:.4f}")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and not (c2 or c3):
    print("  **结论:陶博士的必要条件成立,而他自己说的「反过来不成立」也被量化证实。**")
    print("  **两者同时为真 —— 这正是「共同特征」与「预测能力」的分界。事前预测命中。**")
elif c2 or c3:
    print("  **结论:多头排列有真实预测力 —— 陶博士过于谦虚,我错了。**")
else:
    print("  **结论:连必要条件都不成立。**")

A.drop(columns=["t", "j"]).to_csv(f"{OUT}/tao_bull_alignment.csv", index=False)
print(f"\n→ {OUT}/tao_bull_alignment.csv   ({time.time()-t0:.0f}s)")
