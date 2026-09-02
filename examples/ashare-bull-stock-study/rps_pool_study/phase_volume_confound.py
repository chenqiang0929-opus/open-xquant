"""第一七六节 C 部分 事前登记:对我自己 B 部分那条「缩量优于放量」的证伪检验。

B 部分跑出来最强的一条是 S3(突破周量比五分位),但**方向和我登记的相反**:
留出段 ρ = −0.90,Q1(最缩量)年化超额 +2.50pp、Q5(最放量)−16.89pp,
训练段同向(+11.25pp → −12.76pp)。按登记判据 L4 是**不通过**(我登记的是 ρ ≥ +0.60)。

在把它当成「平台突破的阶段判别式」写进正文之前,必须先排掉一个明显的混淆:
**A 股本来就有「高换手 / 放量 → 后续收益低」的横截面效应。**
B 部分的对照只匹配了同日、同市值名次、同申万一级行业,**没有匹配量比**,
所以那条梯度完全可能与「突破」无关,只是市场普遍效应在突破样本上的一个切片。

做法
----
从**非突破**的样本里抽同一套量比五分位,走完全相同的对照与统计流程:
  抽样池 = 每周最后一个交易日 × 每只可交易股票,**排除 B 部分的 110,770 个突破事件**;
  量比 = 该周成交量 ÷ 前 52 周周成交量均值(与 B 部分逐字同源);
  **五分位边界直接沿用 B 部分训练段的四个数(1.105 / 1.579 / 2.136 / 3.055)**,
  不重新估计 —— 这样两张表才是同一把尺子。
  随机抽 120,000 个点(种子固定),规模与 B 部分的 110,770 相当。

口径:面板 (3316, 5232) 末日 2026-08-28;持有 60 交易日;
对照同日、同市值名次 ±25、同申万一级,200 组种子;
训练段 2013→2021 只报数,留出段 2022-01-01 起看结论。
退市股按最后有效价 ffill 参与,绝不剔除(用户规则 5)。

判据(跑之前写死,跑完照判,不放宽)
------------------------------------
M1 锚点(不过则本部分作废)
   (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) 抽样点与 B 部分突破事件的交集 = **0**;
   (c) 对照抽样市值名次偏离 > 25 的违例 = 0。

M2 **判定(这是对我自己的证伪检验,不是找超额)**
   **若非突破样本在留出段也满足 ρ ≤ −0.60 且 (Q1 年化超额 − Q5 年化超额) ≥ +10.00pp,
   则判定「缩量优于放量」是全市场普遍效应,不是平台突破特有的** ——
   B 部分 S3 必须在正文里降级为「市场普遍效应的一个切片」,不得当成形态判别式。
   反之才说明这条梯度与突破形态有关。

M3 描述:并排给出突破 / 非突破两张五分位表(留出段、60 日),含各组样本数。

**本文件不构成任何投资建议。**
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codex_r10_neutral import NBR, SEED  # noqa: E402
from industry_neutral import build_industry  # noqa: E402
from panel_cache import cached  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR",
                      "/home/user/oxq-panel-0828/oxq_stock_market_fixed")
OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NSEED, JUDGE_H, NQ, NSAMP = 200, 60, 5, 120_000
QCUT = np.array([1.105, 1.579, 2.136, 3.055])   # B 部分训练段边界,原样沿用


def ann(r, h):
    return (1.0 + r) ** (250.0 / h) - 1.0 if r > -1 else np.nan


def spearman5(v):
    return float(pd.Series(np.asarray(v, float)).corr(
        pd.Series(np.arange(len(v)) + 1.0), method="spearman"))


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]

    def _build_panel():
        raise AssertionError("锚点:panel 缓存必须已存在(由 B 部分建立)")
    p = cached("panel", DATA, _build_panel)
    idx = pd.DatetimeIndex(p["idx"])
    cl, okm, vol = p["cl"], p["okm"], p["vol"]
    nt, ns = cl.shape
    assert (nt, ns) == (3316, 5232), f"锚点M1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点M1a 末日 {idx[-1].date()}"
    print(f"锚点M1a ✓ {(nt, ns)} 末日 {idx[-1].date()}", flush=True)
    mvm = cached("mv", DATA, lambda: (_ for _ in ()).throw(
        AssertionError("锚点:mv 缓存必须已存在")))["mv"]
    ind, _, _ = build_industry(codes, idx)

    wk = pd.Series(np.arange(nt), index=idx).resample("W-FRI").last().dropna()
    wsel = wk.to_numpy().astype(int)
    wc = cl[wsel]
    starts = np.concatenate([[0], wsel[:-1] + 1])
    vcs = np.vstack([np.zeros((1, ns)), np.cumsum(np.nan_to_num(vol), axis=0)])
    wv = vcs[wsel + 1] - vcs[starts]
    wdf = pd.DataFrame(wc)
    ma20w = wdf.rolling(20).mean().to_numpy()
    up20 = wdf.shift(1).rolling(20).max().to_numpy()
    hi52 = wdf.shift(1).rolling(52).max().to_numpy()
    vbase = pd.DataFrame(wv).shift(1).rolling(52).mean().to_numpy()

    cross = wc > up20
    prev = np.vstack([np.zeros((1, ns), bool), cross[:-1]])
    base_ok = (np.isfinite(up20) & np.isfinite(ma20w) & np.isfinite(hi52)
               & np.isfinite(vbase) & (vbase > 0) & okm[wsel])
    evm = cross & ~prev & (wc >= ma20w) & base_ok           # B 部分的突破事件
    pool = base_ok & ~evm                                    # 非突破抽样池
    tgrid = np.repeat(wsel[:, None], ns, 1)
    pool &= tgrid < nt - JUDGE_H
    ww, jj = np.nonzero(pool)
    print(f"非突破抽样池 {len(ww):,} 个点;B 部分突破事件 {int(evm.sum()):,} 个",
          flush=True)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(ww), size=min(NSAMP, len(ww)), replace=False)
    ww, jj = ww[pick], jj[pick]
    tt = wsel[ww]
    inter = int(evm[ww, jj].sum())
    print(f"锚点M1b 抽样 {len(tt):,} 个,与突破事件交集 {inter} "
          f"{'✓' if inter == 0 else '✗ 作废'}", flush=True)
    if inter:
        return
    vr = wv[ww, jj] / vbase[ww, jj]
    g = np.digitize(vr, QCUT)

    def controls(ct):
        r2 = np.random.default_rng(SEED)
        out = np.full((NSEED, len(ct)), -1, np.int32)
        viol, cache = 0, {}
        for k, (t, j) in enumerate(zip(ct, jj, strict=True)):
            if t not in cache:
                el = np.flatnonzero(okm[t] & np.isfinite(mvm[t]) & (ind[t] >= 0))
                if not len(el):
                    cache[t] = None
                else:
                    o = el[np.argsort(mvm[t, el], kind="stable")]
                    rk = np.full(ns, -1, np.int32)
                    rk[o] = np.arange(len(o), dtype=np.int32)
                    cache[t] = (o, rk)
            if cache[t] is None:
                continue
            o, rk = cache[t]
            p_, i0 = rk[j], ind[t, j]
            if p_ < 0:
                continue
            lo, hi = max(0, p_ - NBR), min(len(o), p_ + NBR + 1)
            cand = o[lo:hi]
            cand = cand[(ind[t, cand] == i0) & (cand != j)]
            if not len(cand):
                continue
            out[:, k] = r2.choice(cand, NSEED, replace=True)
            viol += int(np.any(np.abs(rk[out[:, k]] - p_) > NBR))
        return out, viol

    cs, viol = controls(tt)
    print(f"锚点M1c 抽样违例 {viol} 个 {'✓' if viol == 0 else '✗ 作废'} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if viol:
        return
    has = cs[0] >= 0
    split = int(np.searchsorted(idx.values, np.datetime64("2022-01-01")))
    segs = (("训练段13-21", tt < split), ("留出段22-26", tt >= split))

    p0 = cl[tt, jj]
    p1 = cl[np.clip(tt + JUDGE_H, 0, nt - 1), jj]
    with np.errstate(all="ignore"):
        r = p1 / np.where(p0 > 0, p0, np.nan) - 1.0

    rows, w = [], 100
    print(f"\n{'='*w}\nM3 非突破样本:量比五分位(边界沿用 B 部分训练段)"
          f",{JUDGE_H} 日持有\n{'='*w}")
    print(f"{'段':<12}{'组':<12}{'样本':>9}{'样本收益':>10}{'对照中位':>10}"
          f"{'超额pp':>9}{'年化超额pp':>12}{'p':>8}")
    for sn, sm in segs:
        for gi in range(NQ):
            m = sm & (g == gi) & has & np.isfinite(r)
            if m.sum() < 30:
                continue
            a = float(np.nanmean(r[m]))
            cm = np.empty(NSEED)
            for s in range(NSEED):
                ci = cs[s][m]
                cp0, cp1 = cl[tt[m], ci], cl[np.clip(tt[m] + JUDGE_H, 0, nt - 1), ci]
                with np.errstate(all="ignore"):
                    cm[s] = np.nanmean(cp1 / np.where(cp0 > 0, cp0, np.nan) - 1.0)
            med = float(np.nanmedian(cm))
            rec = {"段": sn, "组": f"Q{gi+1} 量比", "n": int(m.sum()),
                   "样本收益": a, "对照中位": med, "超额pp": (a - med) * 100,
                   "年化超额pp": (ann(a, JUDGE_H) - ann(med, JUDGE_H)) * 100,
                   "p": float((np.sum(cm >= a) + 1) / (NSEED + 1))}
            rows.append(rec)
            print(f"{sn:<12}{rec['组']:<12}{rec['n']:>9,}{a:>+10.2%}"
                  f"{med:>+10.2%}{rec['超额pp']:>+9.2f}"
                  f"{rec['年化超额pp']:>+12.2f}{rec['p']:>8.4f}")

    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/phase_volume_confound.csv", index=False, encoding="utf-8-sig")
    z = d[d["段"] == "留出段22-26"].sort_values("组")
    print(f"\n{'='*w}\nM2 判定\n{'='*w}")
    if len(z) < NQ:
        print("五组不全,不判定")
        return
    ex = z["年化超额pp"].to_numpy(float)
    rho, gap = spearman5(ex), float(ex[0] - ex[-1])
    same = bool(rho <= -0.60 and gap >= 10.0)
    print(f"非突破样本 留出段:ρ={rho:+.2f} Q1−Q5={gap:+.2f}pp "
          f"五分位 [{' '.join(f'{x:+.2f}' for x in ex)}]")
    print("对照 B 部分突破样本 留出段:ρ=-0.90 Q1−Q5=+19.39pp "
          "五分位 [+2.50 +4.11 -0.41 -5.96 -16.89]")
    print(f"\n判定:{'✓ 触发降级 —— 「缩量优于放量」是全市场普遍效应,'
                    '不是平台突破特有的' if same else
                    '✗ 未触发降级 —— 非突破样本没有复制出同样的梯度'}")
    print(f"\n落库 {OUT}/phase_volume_confound.csv ({time.time()-t0:.0f}s)")
    print("本表是状态记录,不是买点,不构成任何投资建议。")


if __name__ == "__main__":
    main()
