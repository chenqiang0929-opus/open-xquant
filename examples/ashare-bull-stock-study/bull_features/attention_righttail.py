"""第八十四节:关注度与右尾 —— 数据只够做部分,做能做的那部分(事前登记)

═══ 数据清点(先说做不了的) ═══
| 源 | 状态 |
|---|---|
| `lhb_detail.parquet` 龙虎榜 | ✅ 265,723 行 / 2005-2026 / 含个股 |
| `margin_detail_*` 两融      | ✅ 2010-2026 / 含个股(覆盖偏差严重) |
| `hsgt_hist.parquet` 沪深港通 | ❌ 仅 2,726 行**市场级日度汇总,无个股持股** |
| 分析师覆盖 / 研报数 / 搜索指数 / 新闻量 | ❌ **源仓库里不存在** |

**所以「关注度」只能用两个代理变量做部分检验,真正的关注度数据没有。**
这一条必须写在结论旁边,不能让读者以为测的是「关注度」本身。

═══ 两个必须堵的坑 ═══
**坑一:龙虎榜表含 `上榜后1/2/5/10日` 四列未来收益。**
正文 2075 行已把它标为**标签泄露源**。
本脚本**只读 `代码/上榜日`,那四列一列都不碰**。

**坑二:上榜条件本身就是「日涨幅偏离 7%」或「换手率 20%」。**
这是 conditioning on a big move ——
**不控当日涨幅,测到的是动量,不是关注度。**
→ 堵法:LHB 的对照除同市值档外,**再在当月 RPS50 五分位内配对**。

═══ 已测过、本节不重做 ═══
§39/§40 用 **IC / 下期月收益(均值口径)** 测过机构龙虎榜买入(t=−5.83)
与两融五特征(融资余额21日变化 t=−4.80、融券余量 t=−4.86,两段稳定)。
**空白是:这两个源从没在「右尾密度」口径下测过。**

═══ 信号(事前锁定) ═══
  LHB_RECENT   过去 60 交易日内上过龙虎榜(逐月末判定,只回看,无前视)
  MARGIN_HI    融资余额/流通市值 的当月末横截面分位 ≥80%
               **仅在两融覆盖股内部比**(§40 的教训:「是否两融标的」
               本身是显著负因子,月均 −0.696%,不控就是在测市值)

═══ 口径(沿用 §77-§83,不重调) ═══
  结果变量 未来 250 日内最大累计涨幅;对照 同月同市值五分位随机同样多只;
  200 种子;掩码 isfinite(MA300)&isfinite(HI250);退市股 ffill 不剔除。

═══ 锚点(不过则全节作废) ═══
  ① 面板 (3297, 5232)
  ② 零锚点 SMALL_MV lift **逐位复现 §77/§79/§80 的 1.05 / 1.03 / 0.91**(±0.02)
  ③ 两融覆盖率逐年表须与 §40 对得上(2013≈31.7% / 2019≈47.2% / 2022≈66.0%)

═══ 事前判据(跑之前写死,不放宽) ═══
  ① LHB_RECENT 在 ≥500% 上 lift ≥1.3 且 p<0.05(**控 RPS50 之后**)
  ② MARGIN_HI  在 ≥500% 上 lift ≥1.3 且 p<0.05
  ③ **分段同向**:两融 2013-2018 与 2019-2026 分开各报,方向一致才算数
     (§40:2015 年 539 只牛股仅 16.0% 在两融标的内,前段基本不可用)

═══ 判据自查(§79 固化的规则) ═══
**「什么东西会让它通过,而不回答我的问题?」**
→ LHB:上榜=刚大涨过,若不控动量,①通过只说明「涨过的还会涨」。
  **堵法:同市值 × 同 RPS50 双重配对。**
→ MARGIN:两融标的偏大市值,若拿全市场当对照,②通过只反映市值。
  **堵法:对照只从两融覆盖股里抽。**

═══ 事前预测(写下以便被证伪) ═══
**①② 都不通过。** 理由:§39/§40 在均值口径下测出的都是**反向**;
§77 B 部分 24 格全灭;§82 刚测出换手率(同为关注度代理)对极端右尾无净影响。
**若任一通过,说明右尾口径能看到均值口径看不见的东西,我错了。**
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
SRC = "/workspace/etf-netflow-dev/mktdata_enriched/others"
H, NQ, NSEED, SEED = 250, 5, 200, 20260814
GAINS = [1.0, 2.0, 5.0]
LHB_WIN, MIN_BAND, MIN_SIG = 60, 3, 10

t0 = time.time()
cl, mvv = {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "float_mv"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mvv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
MV = pd.DataFrame(mvv).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

CLa, MVa = CL.to_numpy(float), MV.to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
F = pd.DataFrame(CLa).ffill()
Fa = F.to_numpy(float)
HI250 = F.rolling(250, min_periods=250).max().to_numpy(float)
MA300 = F.rolling(300, min_periods=300).mean().to_numpy(float)
RET50 = Fa / F.shift(50).to_numpy(float) - 1
FMAX = pd.DataFrame(Fa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]
ci = {c: i for i, c in enumerate(CL.columns)}

# ── 龙虎榜:只读代码与上榜日,四列未来收益一列都不碰 ─────────────────────
lhb = pd.read_parquet(f"{SRC}/lhb_detail.parquet", columns=["代码", "上榜日"])
lhb["上榜日"] = pd.to_datetime(lhb["上榜日"], errors="coerce")
lhb = lhb.dropna()
LHB = np.zeros((NT, NS), dtype=bool)
n_hit = 0
for code, g in lhb.groupby("代码", sort=False):
    j = ci.get(str(code).zfill(6))
    if j is None:
        continue
    pos = idx.searchsorted(g["上榜日"].values)
    pos = pos[(pos >= 0) & (pos < NT)]
    if len(pos):
        LHB[pos, j] = True
        n_hit += len(pos)
LHB_RECENT = pd.DataFrame(LHB).rolling(LHB_WIN, min_periods=1).max().to_numpy(bool)
print(f"龙虎榜 {len(lhb):,} 行 → 落到面板 {n_hit:,} 个(标的,日);"
      f"过去{LHB_WIN}日覆盖 {LHB_RECENT.mean():.2%}  ({time.time()-t0:.0f}s)")

# ── 两融:沪深合并,只取融资余额;剔 ETF(按代码前缀) ───────────────────
mar = []
for f in sorted(glob.glob(f"{SRC}/margin_detail_*.parquet")):
    d = pd.read_parquet(f)
    cc = [c for c in d.columns if "证券代码" in c][0]
    dc = [c for c in d.columns if "日期" in c][0]
    bc = [c for c in d.columns if c == "融资余额"][0]
    t_ = pd.DataFrame({"code": d[cc].astype(str).str.zfill(6),
                       "date": pd.to_datetime(d[dc].astype(str), errors="coerce"),
                       "bal": pd.to_numeric(d[bc], errors="coerce")}).dropna()
    mar.append(t_[t_["code"].str.match(r"^(0|3|6|8)")])
mar = pd.concat(mar, ignore_index=True)
mar = mar[mar["code"].isin(ci)]
BAL = np.full((NT, NS), np.nan)
jj = mar["code"].map(ci).to_numpy()
tt = idx.searchsorted(mar["date"].values)
m_ok = (tt >= 0) & (tt < NT)
BAL[tt[m_ok], jj[m_ok]] = mar["bal"].to_numpy()[m_ok]
BAL = pd.DataFrame(BAL).ffill(limit=5).to_numpy(float)
MARGIN_R = BAL / np.where(MVa > 0, MVa, np.nan)
print(f"两融 {len(mar):,} 行 → 覆盖 {np.isfinite(BAL).mean():.2%} 的(标的,日)"
      f"  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
months = [p for p in sorted(last_td) if last_td[p] + H < NT]

print(f"\n{'='*96}\n锚点③ 两融覆盖率逐年(对照 §40:2013≈31.7% / 2019≈47.2% / 2022≈66.0%)\n{'='*96}")
cov = {}
for y in range(2013, 2027):
    ts = [last_td[p] for p in months if p.year == y]
    if not ts:
        continue
    t = ts[-1]
    alive = int(ALIVE[t].sum())
    have = int((ALIVE[t] & np.isfinite(BAL[t])).sum())
    cov[y] = have / alive if alive else np.nan
    print(f"  {y}  两融标的 {have:>5,} / 在市 {alive:>5,}  = {cov[y]:>6.1%}")


def pct(arr, t, base):
    return pd.Series(np.where(base, arr[t], np.nan)).rank(pct=True).to_numpy(float) * 100


rng = np.random.default_rng(SEED)
NAMES = ["LHB_RECENT 近60日上榜", "MARGIN_HI 融资余额占比前20%",
         "SMALL_MV 市值最小档(零锚点)"]
SEGS = ["全样本", "2013-2018", "2019-2026"]
acc = {s: {n: {g: [] for g in GAINS} for n in NAMES} for s in SEGS}


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
    ok = base & np.isfinite(MA300[t]) & np.isfinite(HI250[t])
    ratio = np.where(base, FMAX[min(t + 1, NT - 1)] / Fa[t] - 1, np.nan)
    mvt = np.where(ok, MVa[t], np.nan)
    qm = np.nanquantile(mvt[ok], np.linspace(0, 1, NQ + 1)[1:-1])
    r50 = pct(RET50, t, ok)
    qr = np.nanquantile(r50[ok], np.linspace(0, 1, NQ + 1)[1:-1])
    mrg_ok = ok & np.isfinite(MARGIN_R[t])
    mr = np.where(mrg_ok, MARGIN_R[t], np.nan)
    mrp = pd.Series(mr).rank(pct=True).to_numpy(float) * 100

    def mk_bands(univ, extra=None):
        bs = []
        for i in range(NQ):
            lo = -np.inf if i == 0 else qm[i - 1]
            hi = np.inf if i >= NQ - 1 else qm[i]
            sub = univ & (mvt > lo) & (mvt <= hi)
            if extra is None:
                bs.append(np.flatnonzero(sub))
            else:                      # LHB:再按 RPS50 五分位切,控住动量
                for k in range(NQ):
                    a = -np.inf if k == 0 else qr[k - 1]
                    b = np.inf if k >= NQ - 1 else qr[k]
                    bs.append(np.flatnonzero(sub & (r50 > a) & (r50 <= b)))
        return [b for b in bs if len(b) >= 2 * MIN_BAND]

    jobs = [
        ("LHB_RECENT 近60日上榜", ok & LHB_RECENT[t], mk_bands(ok, extra=True)),
        ("MARGIN_HI 融资余额占比前20%", mrg_ok & (mrp >= 80), mk_bands(mrg_ok)),
        ("SMALL_MV 市值最小档(零锚点)",
         ok & (mvt <= np.nanquantile(mvt[ok], 0.2)), mk_bands(ok)),
    ]
    segs = ["全样本", "2013-2018" if p.year <= 2018 else "2019-2026"]
    for nm, sel, bands in jobs:
        if sel.sum() < MIN_SIG or not bands:
            continue
        for g in GAINS:
            got = bandwise(np.where(np.isfinite(ratio), ratio, -9) >= g, sel, bands)
            if got:
                for s in segs:
                    acc[s][nm][g].append(got)
print(f"逐月完成 {len(months)} 月  ({time.time()-t0:.0f}s)")


def summarize(seg, nm, g):
    a = acc[seg][nm][g]
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
for seg in SEGS:
    print(f"\n{'='*104}\n{seg}\n{'='*104}")
    print(f"{'信号':<30}{'月数':>6}"
          + "".join(f"{f'≥{g:.0%}信号':>11}{'随机':>9}{'lift':>7}{'p':>8}" for g in GAINS))
    for nm in NAMES:
        cells = [summarize(seg, nm, g) for g in GAINS]
        n_mo = next((c["n_mo"] for c in cells if c), 0)
        print(f"{nm:<30}{n_mo:>6}" + "".join(
            f"{c['obs']:>11.2%}{c['rnd']:>9.2%}{c['lift']:>7.2f}{c['p']:>8.4f}"
            if c else f"{'—':>11}{'—':>9}{'—':>7}{'—':>8}" for c in cells))
        for g, c in zip(GAINS, cells):
            if c:
                rows.append(dict(分段=seg, 信号=nm, 门槛=f"≥{g:.0%}", 月数=c["n_mo"],
                                 信号命中=c["obs"], 随机=c["rnd"], lift=c["lift"],
                                 lift下界=c["lo"], lift上界=c["hi"], p=c["p"]))
for y, v in cov.items():
    rows.append(dict(分段="两融覆盖率", 信号=str(y), lift=v))

print(f"\n{'='*104}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*104}")
bad = []
zero = [summarize("全样本", "SMALL_MV 市值最小档(零锚点)", g) for g in GAINS]
want = [1.05, 1.03, 0.91]
for g, s, w in zip(GAINS, zero, want):
    if s is None:
        print(f"  ✗ 零锚点 lift@≥{g:.0%}  算不出 —— 不通过")
        bad.append(f"零{g}")
        continue
    okz = abs(round(s["lift"], 2) - w) <= 0.02
    print(f"  {'✓' if okz else '✗'} 零锚点 SMALL_MV lift@≥{g:.0%}  {s['lift']:.2f}"
          f"   (§77/§79/§80 = {w})")
    if not okz:
        bad.append(f"零{g}")
a3 = all(np.isfinite(v) for v in cov.values())
l5 = summarize("全样本", "LHB_RECENT 近60日上榜", 5.0)
m5 = summarize("全样本", "MARGIN_HI 融资余额占比前20%", 5.0)
m5a = summarize("2013-2018", "MARGIN_HI 融资余额占比前20%", 5.0)
m5b = summarize("2019-2026", "MARGIN_HI 融资余额占比前20%", 5.0)
c1 = bool(l5 and l5["lift"] >= 1.3 and l5["p"] < 0.05)
c2 = bool(m5 and m5["lift"] >= 1.3 and m5["p"] < 0.05)
c3 = bool(m5a and m5b and np.sign(m5a["lift"] - 1) == np.sign(m5b["lift"] - 1))
print(f"  {'✓' if not bad else '✗'} 锚点② 零锚点逐位复现")
print(f"  {'✓' if a3 else '✗'} 锚点③ 两融覆盖率逐年可算")
print(f"  {'✓' if c1 else '✗'} 判据① LHB_RECENT lift@500% ≥1.3 且 p<0.05(已控 RPS50)  "
      + (f"{l5['lift']:.2f}  p={l5['p']:.4f}" if l5 else "算不出"))
print(f"  {'✓' if c2 else '✗'} 判据② MARGIN_HI lift@500% ≥1.3 且 p<0.05  "
      + (f"{m5['lift']:.2f}  p={m5['p']:.4f}" if m5 else "算不出"))
print(f"  {'✓' if c3 else '✗'} 判据③ 两融两段同向  "
      + (f"{m5a['lift']:.2f} / {m5b['lift']:.2f}" if (m5a and m5b) else "算不出"))
print()
if bad:
    print("  **锚点②不过:本节结论作废。**")
elif c1 or c2:
    print("  **结论:关注度代理在右尾口径下有信号 —— 事前预测被证伪,我错了。**")
else:
    print("  **结论:两个关注度代理在右尾口径下都不成立。事前预测命中。**")
    print("  **注意:真正的关注度数据(分析师覆盖/研报/搜索指数)源仓库里没有,")
    print("    本节只是两个代理的部分检验,不能当作「关注度无用」的结论。**")

pd.DataFrame(rows).to_csv(f"{SP}/attention_righttail.csv", index=False)
print(f"\n→ {SP}/attention_righttail.csv   ({time.time()-t0:.0f}s)")
