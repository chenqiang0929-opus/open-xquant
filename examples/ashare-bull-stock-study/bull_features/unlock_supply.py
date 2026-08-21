"""第八十节:限售解禁与右尾 —— 验证次新股效应背后的机制(事前登记)

═══ 起因:唯一稳固的发现,机制从没被直接验过 ═══
§77 给出全研究唯一稳固的信号:**上市[1,3)年 lift 1.58、p=0.0000**。
§66 与《检验规格》第 ⑤ 项给它写的机制是:

> **无套牢盘 + 限售解禁时间表已知(首发 1 年 / 控股股东 3 年),
>   恰好覆盖 1-3 年窗口。箱体不是巧合,是被供给压力压出来的;
>   突破发生在压力出清之后。**

**但整个研究从头到尾用「上市天数」当代理,从没碰过解禁本身。**
直接验机制,比再找十个新信号都值。

═══ 数据:解禁日历不存在,从流通股本跳变反推 ═══
`etf-netflow-dev/mktdata_enriched/others/` 里**没有解禁表**
(只有 financials / lhb / margin / hsgt)。
但面板自带 `outstanding_share`,**解禁会表现为流通股本的跳变**。

预检实例(宁德时代 300750,2018-06-11 上市):

    2019-06-11  +451.0%   ← 首发限售 1 年,落在上市周年日
    2021-06-11   +49.6%   ← 控股股东 3 年,同样落在周年日
    2023-04-26   +80.0%   ← 需扣除:送转不是解禁

**送转/转增也会让股本跳变,必须扣掉** —— 用 `corporate_actions.parquet` 里的
`stock_dividend` 与 `capitalization` 除权日排除(±5 个交易日窗口)。

═══ 一个绕不过去的限制:只能测「解禁之后」 ═══
从已发生的股本跳变反推,**天然只能看到过去的解禁,看不到未来的**。
「距下一次解禁 N 天」是**前视的**,本节不测。
所幸机制假说说的正是「突破发生在压力**出清之后**」—— 方向对得上。
**但这意味着本节无法检验「解禁前的压制」那一半假说。**

═══ 信号定义(事前锁定,不搜索、不调参) ═══
  UNLOCK 事件:`outstanding_share` 单日增幅 ≥ **10%**,
              且不在任何 stock_dividend / capitalization 除权日 ±5 交易日内
  POST_UNLOCK:距最近一次 UNLOCK ∈ (0, **120**] 交易日
  (120 ≈ 半年。宁德买点 2019-12-31 距其 2019-06-11 解禁约 138 交易日,
   **略在窗口外 —— 事前写下,不因此调窗口**)

═══ 事前判据(跑之前写死,不放宽) ═══
  ① **锚点**:面板 (3297,5232);且零锚点 SMALL_MV lift 必须
     **逐位复现 §77/§79 的 1.05 / 1.03 / 0.91**(±0.02)。
     对不上说明机器跑偏,**全节作废**。
  ② **主判据**:POST_UNLOCK 在 ≥500% 上 **lift ≥ 1.3 且 p < 0.05**
     (门槛沿用 §77 判据②,不新造)
  ③ **增量判据(本节的关键)**:在 **AGE_YOUNG(上市[1,3)年)基线内部**,
     POST_UNLOCK 仍需 **lift ≥ 1.2 且 p < 0.05**
     (门槛沿用 §77 判据③)

  ②过③不过 → **解禁只是年龄效应的重复,不是独立机制**
  ②③都过   → 解禁是年龄效应之上的独立增量,机制得到直接支持
  ②不过     → 机制假说在右尾口径下不成立

═══ 判据自查(按 §79 落库的那条规则) ═══
**「什么东西会让它通过,而不回答我的问题?」**
→ 解禁事件天然集中在上市 1-3 年,所以 ② 很可能纯粹是 AGE_YOUNG 的重复,
  **通过了也不说明解禁本身有信息**。
→ **这正是 ③ 要堵的漏**。③ 把基线换成 AGE_YOUNG 内部,
  只有解禁带来**增量**时才通得过。判据设计通过自查。

═══ 事前预测(写下以便被证伪) ═══
  ② **可能通过**(因为它与 AGE_YOUNG 高度相关);
  ③ **不通过**。
理由:§77 B 部分 8 信号 × 3 门槛 = 24 格,**≥500% 上一个 p<0.05 都没有**,
结论是「年龄把该抬的右尾抬完了,进了次新池之后再叠加任何信号统计上加不了分」。
**解禁若只是年龄的另一种说法,③ 必然不过。**
**若 ③ 通过,说明机制确实独立于年龄,我错了 —— 那会是 §77 之后第一个新增量。**

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
  零锚点 SMALL_MV lift = 1.05 / 1.03 / 0.91(§77 与 §79 两次一致)
"""
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
np.seterr(invalid="ignore", divide="ignore")

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OTH = f"{SP}/mktdata_enriched_others"
H, NQ, NSEED, SEED = 250, 5, 200, 20260814
GAINS = [1.0, 2.0, 5.0]
JUMP, POST_WIN, CA_PAD = 0.10, 120, 5
Y_LO, Y_HI = 365, 1095
MIN_BAND = 3

t0 = time.time()
cl, mvv, osh, ld = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "float_mv", "outstanding_share",
                                    "listed_days"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mvv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    osh[k] = pd.to_numeric(x["outstanding_share"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
MV = pd.DataFrame(mvv).set_axis(CL.index)
OS = pd.DataFrame(osh).set_axis(CL.index)
LD = pd.DataFrame(ld).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

CLa, MVa, LDa = CL.to_numpy(float), MV.to_numpy(float), LD.to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
F = pd.DataFrame(CLa).ffill()
Fa = F.to_numpy(float)
HI250 = F.rolling(250, min_periods=250).max().to_numpy(float)
MA300 = F.rolling(300, min_periods=300).mean().to_numpy(float)
FMAX = pd.DataFrame(Fa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]

# ── 解禁事件:股本跳变,扣掉送转 ────────────────────────────────────────────
OSa = OS.to_numpy(float)
grow = OSa[1:] / OSa[:-1] - 1
JUMPM = np.zeros((NT, NS), dtype=bool)
JUMPM[1:] = np.isfinite(grow) & (grow >= JUMP)
print(f"股本跳变 ≥{JUMP:.0%}:{JUMPM.sum():,} 个  ({time.time()-t0:.0f}s)")

ca = pd.read_parquet(f"{OTH}/corporate_actions.parquet")
ca["ex_date"] = pd.to_datetime(ca["ex_date"]).dt.tz_localize(None)
ca = ca[ca.action_type.isin(["stock_dividend", "capitalization"])]
ci = {c: i for i, c in enumerate(CL.columns)}
pos = {d: i for i, d in enumerate(idx)}
CAM = np.zeros((NT, NS), dtype=bool)
n_ca = 0
for code, g in ca.groupby("code", sort=False):
    j = ci.get(str(code))
    if j is None:
        continue
    for d in g["ex_date"]:
        i = idx.searchsorted(d)
        if 0 <= i < NT:
            CAM[max(0, i - CA_PAD):min(NT, i + CA_PAD + 1), j] = True
            n_ca += 1
UNLOCK = JUMPM & ~CAM
print(f"送转除权日 {n_ca:,} 个 → 扣除后解禁事件 **{UNLOCK.sum():,}** 个  "
      f"({time.time()-t0:.0f}s)")

# 距最近一次解禁的交易日数(只回看,无前视)
SINCE = np.full((NT, NS), 10**6, dtype=np.int32)
last = np.full(NS, -10**6, dtype=np.int64)
for t in range(NT):
    last = np.where(UNLOCK[t], t, last)
    SINCE[t] = np.minimum(t - last, 10**6)
POST = (SINCE > 0) & (SINCE <= POST_WIN)
print(f"POST_UNLOCK(0,{POST_WIN}] 覆盖 {POST.mean():.2%} 的(股票,日)  "
      f"({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = [p for p in sorted(last_td) if last_td[p] + H < NT]

rng = np.random.default_rng(SEED)
NAMES = ["POST_UNLOCK 解禁后120日", "AGE_YOUNG 上市[1,3)年",
         "SMALL_MV 市值最小档(零锚点)"]
acc = {"A": {n: {g: [] for g in GAINS} for n in NAMES},
       "B": {n: {g: [] for g in GAINS} for n in NAMES}}


def bandwise(hit, sel, bands):
    o, rr, nb = [], np.zeros(NSEED), 0
    for b in bands:
        si = b[sel[b]]
        if len(si) < MIN_BAND or len(b) <= len(si):
            continue
        o.append(hit[si].mean())
        rr += np.array([hit[rng.choice(b, len(si), replace=False)].mean()
                        for _ in range(NSEED)])
        nb += 1
    return (float(np.mean(o)), rr / nb) if nb else None


for p in months:
    t = last_td[p]
    base = ALIVE[t] & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if base.sum() < 200:
        continue
    ok = base & np.isfinite(MA300[t]) & np.isfinite(HI250[t])   # §77 统一掩码
    ratio = np.where(base, FMAX[min(t + 1, NT - 1)] / Fa[t] - 1, np.nan)
    mvt = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(mvt[base], np.linspace(0, 1, NQ + 1)[1:-1])
    bands = []
    for i in range(NQ):
        lo = -np.inf if i == 0 else q[i - 1]
        hi = np.inf if i >= NQ - 1 else q[i]
        bands.append(np.flatnonzero(base & (mvt > lo) & (mvt <= hi)))
    young = ok & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI)
    sig = {"POST_UNLOCK 解禁后120日": ok & POST[t],
           "AGE_YOUNG 上市[1,3)年": young,
           "SMALL_MV 市值最小档(零锚点)": ok & (mvt <= np.nanquantile(mvt[base], 0.2))}
    hits = {g: np.where(np.isfinite(ratio), ratio, -9) >= g for g in GAINS}
    for part, univ in (("A", None), ("B", young)):
        bb = bands if univ is None else [b[univ[b]] for b in bands]
        if univ is not None and univ.sum() < 50:
            continue
        for nm in NAMES:
            s = sig[nm] if univ is None else (sig[nm] & univ)
            if s.sum() < 10:
                continue
            for g in GAINS:
                got = bandwise(hits[g], s, bb)
                if got:
                    acc[part][nm][g].append(got)
print(f"逐月完成 {len(months)} 月  ({time.time()-t0:.0f}s)")


def summarize(part, nm, g):
    a = acc[part][nm][g]
    if len(a) < 12:
        return None
    o = float(np.mean([x[0] for x in a]))
    r = np.mean([x[1] for x in a], axis=0)
    if r.mean() <= 0:
        return None
    lifts = o / np.where(r > 0, r, np.nan)
    return dict(n_mo=len(a), obs=o, rnd=float(r.mean()), lift=o / r.mean(),
                lo=float(np.nanpercentile(lifts, 5)),
                hi=float(np.nanpercentile(lifts, 95)),
                p=float((r >= o).mean()))


rows = []
for part, title in (("A", "A 部分:全市场(同月同市值五分位随机对照)"),
                    ("B", "B 部分:增量检验(AGE_YOUNG 上市[1,3)年 基线内部)")):
    print(f"\n{'='*104}\n{title}\n{'='*104}")
    print(f"{'信号':<28}{'月数':>6}"
          + "".join(f"{f'≥{g:.0%}信号':>11}{'随机':>9}{'lift':>7}{'p':>8}" for g in GAINS))
    for nm in NAMES:
        cells = [summarize(part, nm, g) for g in GAINS]
        n_mo = next((c["n_mo"] for c in cells if c), 0)
        print(f"{nm:<28}{n_mo:>6}" + "".join(
            f"{c['obs']:>11.2%}{c['rnd']:>9.2%}{c['lift']:>7.2f}{c['p']:>8.4f}"
            if c else f"{'—':>11}{'—':>9}{'—':>7}{'—':>8}" for c in cells))
        for g, c in zip(GAINS, cells):
            if c:
                rows.append(dict(部分=part, 信号=nm, 门槛=f"≥{g:.0%}", 月数=c["n_mo"],
                                 信号命中=c["obs"], 随机=c["rnd"], lift=c["lift"],
                                 lift下界=c["lo"], lift上界=c["hi"], p=c["p"]))

print(f"\n{'='*104}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*104}")
bad = []
zero = [summarize("A", "SMALL_MV 市值最小档(零锚点)", g) for g in GAINS]
want = [1.05, 1.03, 0.91]
for g, s, w in zip(GAINS, zero, want):
    if s is None:
        print(f"  ✗ 零锚点 lift@≥{g:.0%}  算不出 —— 不通过")
        bad.append(f"零锚点@{g}")
        continue
    okz = abs(round(s["lift"], 2) - w) <= 0.02
    print(f"  {'✓' if okz else '✗'} 零锚点 SMALL_MV lift@≥{g:.0%}  "
          f"{s['lift']:.2f}   (§77/§79 = {w})")
    if not okz:
        bad.append(f"零锚点@{g}")
c1 = not bad
a5 = summarize("A", "POST_UNLOCK 解禁后120日", 5.0)
b5 = summarize("B", "POST_UNLOCK 解禁后120日", 5.0)
c2 = bool(a5 and a5["lift"] >= 1.3 and a5["p"] < 0.05)
c3 = bool(b5 and b5["lift"] >= 1.2 and b5["p"] < 0.05)
print(f"  {'✓' if c1 else '✗'} ① 锚点(面板 + 零锚点逐位复现 §77/§79)")
print(f"  {'✓' if c2 else '✗'} ② A 部分 lift@500% ≥1.3 且 p<0.05      "
      + (f"{a5['lift']:.2f}  p={a5['p']:.4f}" if a5 else "算不出"))
print(f"  {'✓' if c3 else '✗'} ③ B 部分 lift@500% ≥1.2 且 p<0.05      "
      + (f"{b5['lift']:.2f}  p={b5['p']:.4f}" if b5 else "算不出"))
print()
if not c1:
    print("  **① 不过:机器跑偏,本节结论全部作废。**")
elif c2 and c3:
    print("  **结论:解禁是年龄效应之上的独立增量,机制得到直接支持。**")
    print("  **事前预测(③不通过)被证伪 —— 我错了。**")
elif c2:
    print("  **结论:解禁只是年龄效应的重复,不是独立机制(②过③不过)。**")
    print("  **事前预测命中。**")
else:
    print("  **结论:机制假说在右尾口径下不成立(②就没过)。**")

pd.DataFrame(rows).to_csv(f"{SP}/unlock_supply.csv", index=False)
print(f"\n→ {SP}/unlock_supply.csv   ({time.time()-t0:.0f}s)")
