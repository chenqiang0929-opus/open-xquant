"""第一七七节 事前登记:把「概率」拆成三个,各自量一遍(描述,不判定)。

用户问:「市场强,个股突破的概率就非常大,是吗?」
第一七六节量的是**收益幅度**,不是概率,而且「概率」这个词在这里至少有三个意思。
本节把三个分开量,**只报数,不下判定** —— 与第七六/八六节同规格。

三个「概率」
------------
P1 **发生率**:市场强的时候,突破这件事**出现得更多**吗?
   = 突破事件数 ÷ 具备条件的(周 × 股票)格子数,按突破当日的市场广度五分位分组。
P2 **绝对胜率**:突破之后 60 个交易日**上涨**的比例 P(r60 > 0)。
P3 **相对胜率**:突破之后 60 日**跑赢同日同市值同行业对照**的比例
   P(r_事件 > r_对照),200 组种子全算,取全体事件 × 全体种子的比例。
   **同时报出对照自己的 P(r60 > 0) 作为基准** —— P2 高不高要和它比才有意义。

口径(与第一七六节 B 部分逐字同源,不改一个字)
------------------------------------------------
面板 (3316, 5232) 末日 2026-08-28;事件 = 周收盘首次上穿前 20 周最高周收盘
且收盘 ≥ 20 周线且当日可交易;市场广度 = 收盘 ≥ 自身 MA100 的可交易股票占比;
五分位边界**沿用第一七六节训练段的 0.3991 / 0.5729 / 0.7390 / 0.9548**,不重估;
对照同日、同市值名次 ±25、同申万一级,200 组种子;
训练段 2013→2021 / 留出段 2022-01 起分开报;
退市股按最后有效价 ffill 参与,绝不剔除(用户规则 5)。

锚点(不过则本节作废)
----------------------
N1 (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) 可跑满 250 日的事件数 = **110,770**(与第一七六节 B 部分逐字一致);
   (c) 训练/留出事件数 = **66,767 / 44,003**;
   (d) 对照抽样市值名次偏离 > 25 的违例 = 0。

N2 **本节不设通过/不通过判据** —— 是描述交付,不是假设检验。
   第一七六节已对这三条轴下过判定(0/3 不过),**本节不重判、不翻案**。

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
NSEED, H, NQ = 200, 60, 5
QCUT = np.array([0.3991, 0.5729, 0.7390, 0.9548])   # 第一七六节训练段边界


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    p = cached("panel", DATA, lambda: (_ for _ in ()).throw(
        AssertionError("锚点:panel 缓存必须已存在")))
    idx = pd.DatetimeIndex(p["idx"])
    cl, okm, vol = p["cl"], p["okm"], p["vol"]
    nt, ns = cl.shape
    assert (nt, ns) == (3316, 5232), f"锚点N1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点N1a 末日 {idx[-1].date()}"
    print(f"锚点N1a ✓ {(nt, ns)} 末日 {idx[-1].date()}", flush=True)
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
    evm = cross & ~prev & (wc >= ma20w) & base_ok

    ma100 = pd.DataFrame(cl).rolling(100).mean().to_numpy()
    with np.errstate(all="ignore"):
        breadth = ((cl >= ma100) & okm & np.isfinite(ma100)).sum(1) \
            / np.maximum(okm.sum(1), 1)
    wb = breadth[wsel]                        # 每周的市场广度
    gw = np.digitize(wb, QCUT)                # 每周的广度分位

    split = int(np.searchsorted(idx.values, np.datetime64("2022-01-01")))
    wtr = wsel < split

    # ---------- P1 发生率 ----------
    print(f"\n{'='*96}\nP1 发生率:市场强的时候,突破出现得更多吗\n{'='*96}")
    print(f"{'段':<12}{'广度分位':<12}{'具备条件格子':>14}{'突破事件':>10}{'发生率':>10}")
    p1rows = []
    for sn, sm in (("训练段13-21", wtr), ("留出段22-26", ~wtr)):
        for gi in range(NQ):
            rw = sm & (gw == gi)
            den, num = int(base_ok[rw].sum()), int(evm[rw].sum())
            if not den:
                continue
            p1rows.append({"段": sn, "广度分位": f"Q{gi+1}", "具备条件格子": den,
                           "突破事件": num, "发生率": num / den})
            print(f"{sn:<12}{'Q'+str(gi+1):<12}{den:>14,}{num:>10,}"
                  f"{num/den:>10.2%}")

    # ---------- 事件与对照 ----------
    ww, jj = np.nonzero(evm)
    tt = wsel[ww]
    keep = tt < nt - 250
    ww, jj, tt = ww[keep], jj[keep], tt[keep]
    assert len(tt) == 110_770, f"锚点N1b 事件 {len(tt)}"
    ntr, nho = int((tt < split).sum()), int((tt >= split).sum())
    assert (ntr, nho) == (66_767, 44_003), f"锚点N1c {(ntr, nho)}"
    print(f"\n锚点N1b ✓ 事件 {len(tt):,};锚点N1c ✓ 训练 {ntr:,} / 留出 {nho:,}",
          flush=True)
    g = np.digitize(breadth[tt], QCUT)

    rng = np.random.default_rng(SEED)
    cs = np.full((NSEED, len(tt)), -1, np.int32)
    viol, cache = 0, {}
    for k, (t, j) in enumerate(zip(tt, jj, strict=True)):
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
        cs[:, k] = rng.choice(cand, NSEED, replace=True)
        viol += int(np.any(np.abs(rk[cs[:, k]] - p_) > NBR))
    print(f"锚点N1d 抽样违例 {viol} 个 {'✓' if viol == 0 else '✗ 作废'} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if viol:
        return
    has = cs[0] >= 0
    with np.errstate(all="ignore"):
        r = cl[np.clip(tt + H, 0, nt - 1), jj] / cl[tt, jj] - 1.0

    # ---------- P2 / P3 ----------
    print(f"\n{'='*96}\nP2 绝对胜率 / P3 相对胜率({H} 日持有)\n{'='*96}")
    print(f"{'段':<12}{'广度分位':<12}{'事件':>9}{'P2 事件涨':>11}"
          f"{'对照涨(基准)':>14}{'P3 跑赢对照':>13}{'事件均值':>10}{'对照均值':>10}")
    p23 = []
    for sn, sm in (("训练段13-21", tt < split), ("留出段22-26", tt >= split)):
        for gi in list(range(NQ)) + [-1]:
            m = sm & has & np.isfinite(r) & ((g == gi) if gi >= 0 else True)
            if m.sum() < 30:
                continue
            idxm = np.flatnonzero(m)
            ci = cs[:, idxm]
            with np.errstate(all="ignore"):
                rc = cl[np.clip(tt[idxm] + H, 0, nt - 1)[None, :], ci] \
                    / cl[tt[idxm][None, :], ci] - 1.0
            fin = np.isfinite(rc)
            p2 = float((r[m] > 0).mean())
            pc = float((rc[fin] > 0).mean())
            p3 = float((rc[fin] < np.broadcast_to(r[m], rc.shape)[fin]).mean())
            lab = f"Q{gi+1}" if gi >= 0 else "全段"
            p23.append({"段": sn, "广度分位": lab, "事件": int(m.sum()),
                        "P2事件涨": p2, "对照涨": pc, "P3跑赢对照": p3,
                        "事件均值": float(np.nanmean(r[m])),
                        "对照均值": float(np.nanmean(rc[fin]))})
            print(f"{sn:<12}{lab:<12}{int(m.sum()):>9,}{p2:>11.1%}{pc:>14.1%}"
                  f"{p3:>13.1%}{np.nanmean(r[m]):>+10.2%}"
                  f"{np.nanmean(rc[fin]):>+10.2%}")

    d1, d2 = pd.DataFrame(p1rows), pd.DataFrame(p23)
    d1.to_csv(f"{OUT}/phase_hitrate_p1.csv", index=False, encoding="utf-8-sig")
    d2.to_csv(f"{OUT}/phase_hitrate_p23.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/phase_hitrate_p1.csv、{OUT}/phase_hitrate_p23.csv "
          f"({time.time()-t0:.0f}s)")
    print("本表是状态记录,不是买点,不构成任何投资建议。")


if __name__ == "__main__":
    main()
