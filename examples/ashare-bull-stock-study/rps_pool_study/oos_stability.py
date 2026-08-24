"""§116-A 第一批 4 个幸存者:2026 干净留出期(G1)+ 分期稳定性(G2)。

判据见 factor_sweep_fund.py 的事前登记(commit 3b50918),此处只实现。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, NBR, OUT, SEED, run_window_fast  # noqa: E402
from codex_r10_replication import DATA, TOP_N, WEIGHT, metrics, pct  # noqa: E402
from factor_sweep_pv import build_factors, draw_fast  # noqa: E402

NSEED = 500
SURV = ("small_cap", "low_turnover", "small_cap_low_turnover", "high_amihud")
SEGS = {
    "2026留出(G1)": ("2026-01-05", "2026-07-27"),
    "2014-2017": ("2014-01-02", "2017-12-29"),
    "2018-2021": ("2018-01-02", "2021-12-31"),
    "2022-2025": ("2022-01-04", "2025-12-31"),
}


def main():
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean, amih = z["LOGCAP"], z["TMEAN"], z["AMIH"]
    assert (len(idx), len(codes)) == (3297, 5217), "锚点 面板"
    print(f"锚点 ✓ 面板 {len(idx)}×{len(codes)}")

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)
    fac, _ = build_factors(cl, logcap, tmean, amih, reb)

    rows, viol = [], 0
    for name in SURV:
        sel, elig, selrank = {}, {}, {}
        for t in reb:
            t = int(t)
            base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
            if name == "small_cap_low_turnover":
                e = np.flatnonzero(base)
                if len(e) < TOP_N * 3:
                    continue
                s = (pct(-logcap[t, e].astype(float)) + pct(-tmean[t, e].astype(float))) / 2
            else:
                v = fac[name][t]
                e = np.flatnonzero(base & np.isfinite(v))
                if len(e) < TOP_N * 3:
                    continue
                s = v[e]
            top = e[np.argsort(-s, kind="stable")[:TOP_N]]
            sel[t] = (top, np.full(TOP_N, WEIGHT))
            order = e[np.argsort(logcap[t, e], kind="stable")]
            rk = {int(c): i for i, c in enumerate(order)}
            elig[t] = order
            selrank[t] = np.array([rk[int(c)] for c in top])

        row = {"factor": name}
        t1 = time.time()
        for seg, (d0, d1) in SEGS.items():
            w0 = int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0])
            w1 = int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0])
            eq, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
            m = metrics(eq, dd, idx)
            cg = []
            for sd in range(NSEED):
                rng = np.random.default_rng(SEED + sd)
                csel = {}
                for t in sel:
                    order, ranks = elig[t], selrank[t]
                    ps = draw_fast(rng, ranks, len(order))
                    viol += int(np.sum(np.abs(ps - ranks) > NBR))
                    csel[t] = (order[ps], np.full(TOP_N, WEIGHT))
                e2, d2, _, _ = run_window_fast(op, cl, susp, lu, ld, csel, cal_pos, w0, w1)
                cg.append(metrics(e2, d2, idx)["cagr"])
            a = np.array(cg)
            p = (1 + int(np.sum(a >= m["cagr"]))) / (NSEED + 1)
            row |= {f"{seg}_cagr": m["cagr"], f"{seg}_ctrl_med": float(np.median(a)),
                    f"{seg}_p": p}
        rows.append(row)
        seg3 = [k for k in SEGS if k != "2026留出(G1)"]
        row["G1"] = bool(row["2026留出(G1)_p"] < 0.05)
        row["G2"] = int(sum(row[f"{k}_p"] < 0.05 for k in seg3)) >= 2
        print(f"{name:24s} " + " | ".join(
            f"{k.split('留出')[0]}:{row[f'{k}_cagr']:+7.2%}(p={row[f'{k}_p']:.4f})"
            for k in SEGS) + f"  G1={'✓' if row['G1'] else '✗'} "
            f"G2={'✓' if row['G2'] else '✗'}  ({time.time()-t1:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n锚点 抽样越界 {viol} 次 {'✓' if viol == 0 else '✗ 作废'}")
    assert viol == 0
    print(f"G1(2026 干净留出期 p<0.05)通过 {int(df['G1'].sum())}/4:"
          f" {', '.join(df.loc[df['G1'], 'factor']) or '无'}")
    print(f"G2(三段至少两段 p<0.05)通过 {int(df['G2'].sum())}/4:"
          f" {', '.join(df.loc[df['G2'], 'factor']) or '无'}")
    df.to_csv(f"{OUT}/oos_stability.csv", index=False)
    print(f"落库 {OUT}/oos_stability.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §116-A 结果:4 个幸存者里只有 2 个同时过 G1 与 G2。锚点 抽样越界 0 次 ✓
#
# 因子                    2026留出(G1)        2014-2017        2018-2021        2022-2025      G1 G2
# small_cap              -18.16% p=0.1517   +44.77% p=.0958  +32.04% p=.0140  +19.59% p=.0020  ✗  ✓
# low_turnover            -1.32% p=0.0958   +19.75% p=.0579   +3.01% p=.2814  +16.81% p=.0020  ✗  ✗
# small_cap_low_turnover  +8.90% p=0.0020   +32.58% p=.0200  +10.98% p=.1697  +29.82% p=.0020  ✓  ✓
# high_amihud             +6.70% p=0.0020   +35.04% p=.0060  +12.89% p=.1417  +43.27% p=.0020  ✓  ✓
#
# G1 通过 2/4:small_cap_low_turnover、high_amihud
# G2 通过 3/4:small_cap、small_cap_low_turnover、high_amihud
# 两项都过:**small_cap_low_turnover、high_amihud**
#
# 事前预测:
# **Q1 错了。** 我预测「4 个里至少 3 个在 2026 干净留出期 p<0.05」,实际只有 2 个。
#   而且 `small_cap` 在 2026 是 **-18.16%**,是全表最差的一格 ——
#   §115-B 里它 full 年化 +33.14%、p=0.0020(500 种子下限,无一组对照跑赢),
#   在唯一没被看过的窗口上却大幅为负。**这正是「全在样本内」的代价。**
# **Q2 错了。** 我预测「4 个全部通过 G2」,`low_turnover` 只过 1/3 段
#   (2014-2017 p=.0579、2018-2021 p=.2814、2022-2025 p=.0020),
#   它的全期显著性主要来自 2022-2025 一段。
#
# 注意 G1 的证据力:2026 留出期只有约 7 个月、7 个调仓日,
# **不足以定论**,只能作为弱证据。但它是整段历史里唯一没被 §115-B 看过的窗口,
# 而它给出的信号是负面的 —— 这一条必须写在正文里,不能因为量小就略过。
#
# 一个必须承认的结构性缺陷:第一批的 16 个因子是在 full(2014-2025)上筛的,
# 事后再补留出期,**筛选偏差无法被这样补救**。真正干净的做法是
# 先定训练期、在训练期筛、留出期只看一次 —— §116-B 的财务因子从一开始就这么做。
# =============================================================================
