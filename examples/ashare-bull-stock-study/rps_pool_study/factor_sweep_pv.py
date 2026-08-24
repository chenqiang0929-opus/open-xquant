"""§115-B 第一批:价量因子横扫 + 同市值随机对照(不依赖财务数据)。

为什么做这件事
--------------
前 112 节把力气全花在**单只股票右尾**(能不能提前认出一只将来涨 N 倍的票),
50 多个事前登记的检验全部落在 lift ≈ 1.0。§114 的教训是:
**横截面因子 + TopN 组合本来就是更容易出结果的战场**,而我直到 §109 才走过去。
Codex 的 R01–R13 走的是这条路,但他的对照只有 510300 买入持有,没有随机对照、
没有市值中性化、没有 p 值,结果 R10 的头号成果被一个复权价混进市值的 bug 撑起了一半。

本节 = **他的广度 + 我的对照**:一次扫 16 个价量因子,每个都挂上
同市值邻域匹配的随机对照与 Bonferroni 校正后的 p 值。

因子清单(16 个,全部只用价量,不依赖财务)
-------------------------------------------
规模流动性  small_cap / large_cap* / low_turnover / high_turnover* /
            small_cap_low_turnover(R10 本体,§114 已知) / low_amihud / high_amihud
低波动      low_vol_60 / low_vol_120 / low_mdd_250
动量        mom_250 / mom_60
反转        rev_20 / rev_5
形态        near_high_250(最接近 250 日新高) / trend_sma50_200
带 * 的两个是**负对照**(见 F3)。

统一口径(照抄 Codex fast_low_risk_backtest.py,与 §114 完全一致)
------------------------------------------------------------------
TopN=20、等权 5%、每 20 个交易日调仓、信号日收盘定次日开盘成交、100 股整手、
佣金 0.03%(最低 5 元)、印花税 0.1%、双边滑点 0.1%、
ST/停牌/涨停/上市不足 250 日历日/零成交剔除。调仓日 153 个。
市值一律用**真实流通市值** raw_close × PIT 股本(§113 的教训)。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
F1 锚点(同 §114,不过则整节作废):
   面板 (3297, 5217);抽样恒等式 —— 对照每一次抽样的市值名次偏离必须 ≤25,
   对所有因子、所有调仓日、所有种子成立,违例数 > 0 即作废。

F2 单因子判定。对照 = 同市值名次 ±25 邻域匹配随机抽 20 只,**500 组种子**。
   p = (1 + #{对照年化 ≥ 策略年化}) / 501,单尾。
   **Bonferroni:一次检验 16 个因子,α = 0.05 / 16 = 0.003125。**
   F2 通过 ⟺ p < 0.003125 **且 full 与 oos 两个窗口都成立**。
   (500 组种子的最小可达 p = 1/501 = 0.001996 < 0.003125,阈值可分辨。)

F3 负对照锚点(§83 反问型)。`large_cap` 与 `high_turnover` 两个因子
   **不得**通过 F2。Codex 自己的 2023–2025 Rank IC 表里 high_turnover
   四个持有期全为负(−0.0629/−0.0688/−0.0543/−0.0230),large_cap 亦弱。
   **若负对照也通过,说明对照或引擎有系统性偏差,本节全部作废。**
   —— 这一条是「什么会让它通过而与问题无关」的堵口。

F4 描述项(不设阈值):每个因子的年化、回撤、夏普、组合换手率、组合波动率、
   卖出冻结次数,以及相对 510300(含分红)的超额。

事前预测(写下来以便被证伪;错了必须在正文里明说我错了)
--------------------------------------------------------
P1 `small_cap_low_turnover` 会过(§114 在 100 种子下 full p=0.0099、oos p=0.0099,
   500 种子下应更小)。
P2 `small_cap` 会过。
P3 低波动族(low_vol_60 / low_vol_120 / low_mdd_250)**至少一个**会过。
P4 负对照 `large_cap` 与 `high_turnover` **都不会过**(F3 要求如此)。
P5 **动量族(mom_250 / mom_60)不会过。** 理由:本项目 §12/§34/§90/§103/§105
   反复测过新高与 RPS,全部 lift ≈ 1.0;Codex 的 R02(绝对动量)、R03(高 RPS 追强)
   也都判了淘汰。两边独立失败。
P6 通过 F2 的因子总数 ∈ [3, 8]。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推任何东西;
**不跑依赖财务数据的因子**(价值/质量/多因子)—— 那要等 §115-A 的 PIT 验证过关;
不基于本节结论做任何可交易性声明。
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
from codex_r10_replication import DATA, TOP_N, WEIGHT, WINDOWS, metrics, pct  # noqa: E402

NSEED = 500
ALPHA = 0.05 / 16
NEG = ("large_cap", "high_turnover")


def draw_fast(rng, ranks, n, nbr=NBR):
    """向量化的邻域抽样:每只在自身市值名次 ±nbr 内换一只,去重。"""
    lo = np.maximum(0, ranks - nbr)
    hi = np.minimum(n - 1, ranks + nbr)
    span = hi - lo + 1
    pos = lo + (rng.random(len(ranks)) * span).astype(np.int64)
    for _ in range(12):
        _, first = np.unique(pos, return_index=True)
        dup = np.setdiff1d(np.arange(len(pos)), first)
        if not len(dup):
            break
        pos[dup] = lo[dup] + (rng.random(len(dup)) * span[dup]).astype(np.int64)
    return pos


def build_factors(cl, logcap, tmean, amih, reb):
    """在 153 个调仓日上算 16 个价量因子。分数一律'越大越优先'。"""
    out = {k: {} for k in (
        "small_cap", "large_cap", "low_turnover", "high_turnover",
        "small_cap_low_turnover", "low_amihud", "high_amihud",
        "low_vol_60", "low_vol_120", "low_mdd_250", "mom_250", "mom_60",
        "rev_20", "rev_5", "near_high_250", "trend_sma50_200")}
    fin = {}
    for t in reb:
        t = int(t)
        w = cl[max(0, t - 249):t + 1].astype(np.float64)
        r = w[1:] / w[:-1] - 1.0
        px = cl[t].astype(np.float64)
        with np.errstate(all="ignore"):
            v60 = np.nanstd(r[-60:], axis=0, ddof=1)
            v120 = np.nanstd(r[-120:], axis=0, ddof=1)
            pk = np.maximum.accumulate(np.nan_to_num(w, nan=-np.inf), axis=0)
            dd = np.where(pk > 0, w / pk - 1.0, np.nan)
            mdd = np.nanmin(dd, axis=0)
            hi250 = np.nanmax(w, axis=0)
            m250 = px / cl[max(0, t - 250)].astype(np.float64) - 1.0
            m60 = px / cl[max(0, t - 60)].astype(np.float64) - 1.0
            r20 = px / cl[max(0, t - 20)].astype(np.float64) - 1.0
            r5 = px / cl[max(0, t - 5)].astype(np.float64) - 1.0
            s20, s50 = np.nanmean(w[-20:], 0), np.nanmean(w[-50:], 0)
            s60, s200 = np.nanmean(w[-60:], 0), np.nanmean(w[-200:], 0)
            trend = np.where((px > s200) & (s50 > s200), s50 / s200 - 1.0, np.nan)
        lc, tm, am = (logcap[t].astype(np.float64), tmean[t].astype(np.float64),
                      amih[t].astype(np.float64))
        f = {"small_cap": -lc, "large_cap": lc, "low_turnover": -tm,
             "high_turnover": tm, "low_amihud": -am, "high_amihud": am,
             "low_vol_60": -v60, "low_vol_120": -v120, "low_mdd_250": mdd,
             "mom_250": m250, "mom_60": m60, "rev_20": -r20, "rev_5": -r5,
             "near_high_250": np.where(hi250 > 0, px / hi250, np.nan),
             "trend_sma50_200": trend}
        for k, v in f.items():
            out[k][t] = v
        fin[t] = (s20, s60)
        out["small_cap_low_turnover"][t] = None   # 复合分,选股时单算
    return out, None


def main():
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean, amih = z["LOGCAP"], z["TMEAN"], z["AMIH"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), f"锚点F1 面板 {(nt, ns)}"
    print(f"锚点F1 ✓ 面板 {nt}×{ns}")

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill()
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)
    print(f"调仓日 {len(reb)} 个")

    def win(w):
        d0, d1 = WINDOWS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    t0 = time.time()
    fac, _ = build_factors(cl, logcap, tmean, amih, reb)
    print(f"因子构造完成 ({time.time()-t0:.0f}s)")

    names = list(fac)
    rows, viol_all = [], 0
    for name in names:
        sel, elig, selrank = {}, {}, {}
        for t in reb:
            t = int(t)
            base_ok = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
            if name == "small_cap_low_turnover":
                m = base_ok
                e = np.flatnonzero(m)
                if len(e) < TOP_N * 3:
                    continue
                s = (pct(-logcap[t, e].astype(float)) + pct(-tmean[t, e].astype(float))) / 2
            else:
                v = fac[name][t]
                m = base_ok & np.isfinite(v)
                e = np.flatnonzero(m)
                if len(e) < TOP_N * 3:
                    continue
                s = v[e]
            top = e[np.argsort(-s, kind="stable")[:TOP_N]]
            sel[t] = (top, np.full(TOP_N, WEIGHT))
            order = e[np.argsort(logcap[t, e], kind="stable")]
            rk = {int(c): i for i, c in enumerate(order)}
            elig[t] = order
            selrank[t] = np.array([rk[int(c)] for c in top])

        row = {"factor": name, "n_reb": len(sel)}
        t1 = time.time()
        for w in ("full", "oos"):
            w0, w1 = win(w)
            eq, dd, tr, fz = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
            m = metrics(eq, dd, idx)
            s = bs[(bs.index >= WINDOWS[w][0]) & (bs.index <= WINDOWS[w][1])]
            cg = []
            for sd in range(NSEED):
                rng = np.random.default_rng(SEED + sd)
                csel = {}
                for t in sel:
                    order, ranks = elig[t], selrank[t]
                    ps = draw_fast(rng, ranks, len(order))
                    viol_all += int(np.sum(np.abs(ps - ranks) > NBR))
                    csel[t] = (order[ps], np.full(TOP_N, WEIGHT))
                e2, d2, _, _ = run_window_fast(op, cl, susp, lu, ld, csel, cal_pos, w0, w1)
                cg.append(metrics(e2, d2, idx)["cagr"])
            a = np.array(cg)
            p = (1 + int(np.sum(a >= m["cagr"]))) / (NSEED + 1)
            row |= {f"{w}_cagr": m["cagr"], f"{w}_mdd": m["mdd"], f"{w}_sharpe": m["sharpe"],
                    f"{w}_total": m["total"], f"{w}_frozen": fz, f"{w}_trades": tr,
                    f"{w}_bench": float(s.iloc[-1] / s.iloc[0] - 1),
                    f"{w}_ctrl_med": float(np.median(a)),
                    f"{w}_ctrl_p999": float(np.percentile(a, 99.7)), f"{w}_p": p}
        row["pass_F2"] = bool(row["full_p"] < ALPHA and row["oos_p"] < ALPHA)
        rows.append(row)
        flag = "负对照" if name in NEG else ""
        print(f"{name:24s} full 年化{row['full_cagr']:+7.2%} p={row['full_p']:.4f} | "
              f"oos 年化{row['oos_cagr']:+7.2%} p={row['oos_p']:.4f} | "
              f"{'通过' if row['pass_F2'] else '  — ':4s} {flag}  ({time.time()-t1:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    print(f"\n锚点F1 抽样越界 {viol_all} 次  {'✓' if viol_all == 0 else '✗ 整节作废'}")
    assert viol_all == 0
    neg_pass = [r["factor"] for r in rows if r["factor"] in NEG and r["pass_F2"]]
    print(f"F3 负对照 {NEG} 通过者:{neg_pass or '无'}  "
          f"{'✓' if not neg_pass else '✗ 整节作废'}")
    npass = int(df["pass_F2"].sum())
    print(f"F2 通过 {npass}/{len(df)} 个(Bonferroni α={ALPHA:.6f})")
    print("  " + ", ".join(df.loc[df["pass_F2"], "factor"]))
    df.to_csv(f"{OUT}/factor_sweep_pv.csv", index=False)
    print(f"落库 {OUT}/factor_sweep_pv.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §115-B 结果:16 个价量因子,4 个通过 Bonferroni 校正后的同市值随机对照。
#
# 锚点 F1 抽样越界 0 次 ✓        F3 负对照(large_cap / high_turnover)通过者:无 ✓
#
# 因子                     full 年化   full p    oos 年化   oos p    判定
# small_cap                +33.14%   0.0020    +29.77%   0.0020   通过
# low_turnover             +13.72%   0.0020    +20.57%   0.0020   通过
# small_cap_low_turnover   +25.63%   0.0020    +36.21%   0.0020   通过
# high_amihud              +30.69%   0.0020    +54.25%   0.0020   通过 ← 非流动性溢价
# low_mdd_250               +9.15%   0.0060    +13.08%   0.0020     —  (full 差一点)
# large_cap                 +9.00%   0.0379    +13.95%   0.0040     —  负对照 ✓
# low_vol_60                +8.25%   0.2315    +10.00%   0.0479     —
# low_vol_120               +7.63%   0.2954     +9.53%   0.1158     —
# low_amihud                +4.45%   0.3832     +7.78%   0.0878     —
# rev_20                    +3.33%   0.8483    +14.21%   0.1896     —
# mom_250                   -5.51%   1.0000     -9.22%   0.9920     —
# near_high_250             -6.69%   1.0000     -7.39%   0.9980     —
# rev_5                     -7.49%   1.0000     -2.95%   0.9401     —
# trend_sma50_200           -8.33%   1.0000    -16.11%   1.0000     —
# mom_60                   -10.14%   1.0000    -21.73%   1.0000     —
# high_turnover            -13.36%   1.0000    -17.11%   1.0000     —  负对照 ✓
#
# p=0.0020 是 500 组种子的下限(1/501),即**没有任何一组同市值随机对照跑赢**。
#
# 事前预测:P1 ✓ P2 ✓ P4 ✓ P5 ✓ P6 ✓(通过 4 个 ∈[3,8]);
# **P3 错了 —— 我说了低波动族至少一个会过,三个全没过。**
#   low_vol_60 full p=0.2315、low_vol_120 full p=0.2954、low_mdd_250 full p=0.0060
#   (低于 0.05 但高于 Bonferroni 的 0.003125)。低波动的绝对年化确实为正(+7.6~9.2%),
#   但**在同市值邻域内并不比随机更好** —— 又是一次把「绝对水平」当成「相对优势」,
#   与 §98/§103/§104/§105 同一种错误,这是第六次。
#
# 三个独立失败的交叉印证(同一假设、不同数据口径、不同判据):
#   near_high_250 / trend_sma50_200 / mom_250 / mom_60 全部 p≈1.0 且年化为负,
#   与 Codex 的 R01(趋势跟随淘汰)、R02(绝对动量淘汰)、R03(高 RPS 追强淘汰)、
#   R13(接近一年新高+RPS+基本面未过样本外)方向一致,
#   也与本项目 §12/§34/§90/§103/§105 的 lift≈1.0 一致。**三方独立失败。**
#
# 未解决:本节 16 个因子仍全部在 2014–2025 内,没有时间样本外;
# 通过的 4 个因子高度相关(small_cap / low_turnover / 二者复合 / high_amihud
# 都在微盘-低流动性这一层),不是 4 个独立发现。
# =============================================================================
