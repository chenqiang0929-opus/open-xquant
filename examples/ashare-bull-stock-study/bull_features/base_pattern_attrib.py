"""A 部分:严格基底形态的归因检验(对照第五十一节的粗定义)

═══ 本节要回答的 ═══
第五十一节说「基底形态无信息量」(lift 0.99),但那测的是
「60日振幅<30%」—— 不是基底形态。换成欧奈尔的严格定义后,还是 0.99 吗?

═══ 口径完全沿用第五十一/五十三节 ═══
每只股票每一年定位「该年最大涨幅的起点」t\\*,牛股与非牛股用同一方法,
特征全部在 t\\* **之前**测量。

**锚点(必须先过)**:样本 41,557、牛股 2,232、基准率 5.37%
—— 与 bull_feature_scan.log 一致。对不上说明 t\\* 定位被我改动了,先修再跑。

═══ 一个必须交代的样本限制 ═══
检测器需要 t 之前 **325(基底)+250(前期涨幅)= 575** 天历史,
而锚点样本只要求 t≥310。**不改锚点**(改了就没法和前两节对比),
改为在完整窗口子集上做归因,并**明示子集的样本量与基准率**。

═══ 两条纪律(与第五十三节相同,不放宽) ═══
A. 每个形态自己的零分布(年内打乱 500 次)双侧 p < 0.05
B. lift > 公平 best-of-N 天花板(只让命中≥500 的形态参与)
C. 2013-2019 与 2020-2025 两段方向一致
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_pattern_detector import NEED, PRIOR, WIN, detect_base  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
Y0, Y1 = 2013, 2025
N_PERM = 500
MIN_HITS = 500
SEED = 20260812

t0 = time.time()
d = {c: {} for c in ["open", "high", "low", "close", "volume"]}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    try:
        x = pd.read_parquet(f, columns=list(d))
    except Exception:
        continue
    if x.empty:
        continue
    for c in d:
        d[c][k] = pd.to_numeric(x[c], errors="coerce")
CL = pd.DataFrame(d["close"]).sort_index(); CL.index = CL.index.tz_localize(None)


def al(k):
    f = pd.DataFrame(d[k]).sort_index(); f.index = f.index.tz_localize(None)
    return f.reindex(index=CL.index, columns=CL.columns)


HI, LO, VO = al("high"), al("low"), al("volume")
CL = CL.where(CL > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0)
idx = CL.index
A, Ha, La, Va = CL.to_numpy(), HI.to_numpy(), LO.to_numpy(), VO.to_numpy()
# 前期涨幅用的滚动最低收盘(shift 1:不含当天,避免自己比自己)
PMIN = CL.rolling(PRIOR, min_periods=60).min().shift(1).to_numpy()
codes = list(CL.columns)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
del d

year = idx.year.to_numpy()
rows = []
for j, cd in enumerate(codes):
    a = A[:, j]
    fin = np.isfinite(a) & (a > 0)
    if fin.sum() < 300:
        continue
    for y in range(Y0, Y1 + 1):
        cur = np.flatnonzero((year == y) & fin)
        if cur.size < 100:
            continue
        prev = np.flatnonzero((year == y - 1) & fin)
        if prev.size == 0:
            continue
        yr_ret = a[cur[-1]] / a[prev[-1]] - 1
        fwd_max = np.maximum.accumulate(a[cur][::-1])[::-1]
        t = int(cur[int(np.argmax(fwd_max / a[cur] - 1))])
        if t < 310:
            continue
        rec = {"code": cd, "year": y, "bull": yr_ret > 1.0, "t": t, "j": j,
               "full_win": t >= NEED}
        # 第五十一节的粗定义,原样保留做对照
        w60 = a[t - 60:t]; w60 = w60[np.isfinite(w60)]
        rec["coarse_range60"] = ((w60.max() - w60.min()) / w60.min()
                                 if w60.size > 30 and w60.min() > 0 else np.nan)
        if rec["full_win"]:
            s0 = t - WIN                      # 窗口 = [t-WIN, t-1],**不含 t**
            assert s0 >= PRIOR, "窗口起点早于前期涨幅可回看的范围"
            b = detect_base(a[s0:t], Ha[s0:t, j], La[s0:t, j], Va[s0:t, j],
                            PMIN[s0:t, j])
            rec.update(b)
            # pivot 相对 t 当天价格的位置(供 B 部分参考,不进归因)
            for p in ("cup", "flat", "dbl"):
                pv = b[f"{p}_pivot"]
                rec[f"{p}_gap"] = (a[t] / pv - 1) if np.isfinite(pv) and pv > 0 else np.nan
        rows.append(rec)
    if (j + 1) % 1000 == 0:
        print(f"  已处理 {j+1:,} 只  ({time.time()-t0:.0f}s)")

P = pd.DataFrame(rows)
for c in ("cup", "flat", "dbl"):
    P[c] = P[c].fillna(False).astype(bool)
P["any_base"] = P.cup | P.flat | P.dbl

# ═══ 锚点自检:不过就停 ═══
print(f"\n锚点自检:样本 {len(P):,}(应 41,557)、牛股 {int(P.bull.sum()):,}(应 2,232)、"
      f"基准率 {P.bull.mean():.2%}(应 5.37%)")
assert len(P) == 41557, f"样本数与前两节不一致:{len(P)}"
assert int(P.bull.sum()) == 2232, f"牛股数与前两节不一致:{int(P.bull.sum())}"

Q = P[P.full_win].copy()
BASE = Q.bull.mean()
print(f"\n完整窗口子集(t ≥ {NEED}):样本 **{len(Q):,}**、牛股 **{int(Q.bull.sum()):,}**、"
      f"基准率 **{BASE:.2%}**")
print(f"  被排除的 {len(P)-len(Q):,} 个是历史不足 {NEED} 天的(集中在 2013-2015),"
      f"其牛股率 {P[~P.full_win].bull.mean():.2%}")

FEATS = {
    "杯柄(cup with handle)": Q.cup,
    "平底(flat base)": Q.flat,
    "双底(double bottom)": Q.dbl,
    "任一严格基底": Q.any_base,
    "【第51节粗定义】60日振幅<30%": Q.coarse_range60 < 0.30,
    "【第51节粗定义】60日振幅<50%": Q.coarse_range60 < 0.50,
}

print(f"\n{'#'*112}\n基准牛股率 {BASE:.2%};lift = P(牛股|形态) ÷ 基准\n{'#'*112}")
print(f"{'形态':<34}{'P(形态|牛股)':>13}{'P(形态|非牛)':>13}"
      f"{'**P(牛股|形态)**':>15}{'lift':>8}{'命中数':>10}")
b = Q.bull.to_numpy()
masks, res = {}, {}
for nm, mk in FEATS.items():
    m = mk.fillna(False).to_numpy().astype(bool)
    if m.sum() < 30:
        print(f"{nm:<34}{'样本<30':>13}")
        continue
    masks[nm] = m
    pb = b[m].mean()
    res[nm] = {"P(形态|牛股)": m[b].mean(), "P(形态|非牛)": m[~b].mean(),
               "P(牛股|形态)": pb, "lift": pb / BASE, "命中数": int(m.sum())}
    print(f"{nm:<34}{m[b].mean():>13.1%}{m[~b].mean():>13.1%}"
          f"{pb:>15.2%}{pb/BASE:>8.2f}{int(m.sum()):>10,}")

# 命中率合理性(事前写的检查:>60% 或 <1% 说明阈值实现有误)
hr = Q.any_base.mean()
print(f"\n  合理性检查:任一严格基底的命中率 **{hr:.1%}**"
      f"  ({'正常' if 0.01 <= hr <= 0.60 else '**异常,需先查实现**'})")

# ═══ 纪律 A/B/C ═══
rng = np.random.default_rng(SEED)
yr = Q.year.to_numpy()
perms = np.empty((N_PERM, len(b)), bool)
for k in range(N_PERM):
    bb = b.copy()
    for yv in np.unique(yr):
        s = yr == yv
        bb[s] = rng.permutation(bb[s])
    perms[k] = bb

print(f"\n{'='*112}\n三条纪律(与第五十三节相同,未放宽)\n{'='*112}")
print(f"{'形态':<34}{'lift':>8}{'p(自身零分布)':>15}{'13-19':>9}{'20-25':>9}{'两段同向':>10}")
early = yr <= 2019
out = []
for nm, m in masks.items():
    lf = b[m].mean() / BASE
    nulls = perms[:, m].mean(axis=1) / BASE
    p_two = float((np.abs(nulls - 1.0) >= abs(lf - 1.0)).mean())

    def _lf(sel):
        mm = m & sel
        return b[mm].mean() / b[sel].mean() if mm.sum() >= 30 else np.nan
    e, la = _lf(early), _lf(~early)
    same = np.isfinite(e) and np.isfinite(la) and (e - 1) * (la - 1) > 0
    print(f"{nm:<34}{lf:>8.2f}{p_two:>15.3f}{e:>9.2f}{la:>9.2f}"
          f"{'✓' if same else '✗':>10}{' **' if p_two < 0.05 else ''}")
    out.append({**res[nm], "形态": nm, "p": p_two, "lift_13_19": e,
                "lift_20_25": la, "两段同向": same, "null": nulls})

big = [o for o in out if o["命中数"] >= MIN_HITS]
if len(big) >= 2:
    stack = np.vstack([o["null"] for o in big])
    q95 = float(np.quantile(stack.max(axis=0), 0.95))
    print(f"\n  公平 best-of-{len(big)} 噪音上界(命中≥{MIN_HITS}):"
          f"中位 {np.median(stack.max(axis=0)):.2f}   **95%分位 {q95:.2f}**")
else:
    q95 = np.nan
    print(f"\n  命中≥{MIN_HITS} 的形态不足 2 个,无法构造 best-of-N 天花板")

print(f"\n{'='*112}\n判定\n{'='*112}")
win = [o for o in out if o["p"] < 0.05 and o["两段同向"]
       and np.isfinite(q95) and o["lift"] > q95 and not o["形态"].startswith("【")]
print(f"  三条纪律全过的严格形态:**{len(win)} 个** {[o['形态'] for o in win]}")
for o in win:
    print(f"    {o['形态']:<30} lift {o['lift']:.2f}   "
          f"P(牛股|形态) {o['P(牛股|形态)']:.2%}   覆盖牛股 {o['P(形态|牛股)']:.1%}")

print("\n  严格定义 vs 粗定义:")
for nm in ("任一严格基底", "【第51节粗定义】60日振幅<30%"):
    if nm in res:
        print(f"    {nm:<32} lift {res[nm]['lift']:.2f}   "
              f"命中 {res[nm]['命中数']:,}   覆盖牛股 {res[nm]['P(形态|牛股)']:.1%}")

pd.DataFrame([{k: v for k, v in o.items() if k != "null"} for o in out]).to_csv(
    f"{SP}/base_pattern_attrib.csv", index=False)
P.drop(columns=[c for c in P.columns if c.endswith("_start")]).to_parquet(
    f"{SP}/base_pattern_panel.parquet")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: base_pattern_attrib.csv, base_pattern_panel.parquet")
