"""§114-D4(e) 拆解:D2/D3 通过的机制是"选股"还是"低成本低波动的构造"?

§79 正问 —— 什么会让 D2/D3 通过而不回答我的问题?
三个候选:(a) 对照的组合换手率高于策略,多付交易成本;
(b) 对照的组合波动率高于策略,复利拖累更大;(c) 整手/现金拖累。
三者都与"选出更好的股票"无关。

本脚本把成本与整手全部关掉(等权、可拆分、零佣金零印花零滑点),
再跑同一组策略与同两组对照。

  若零成本下策略相对对照的优势**大幅缩水**
      → D2/D3 通过是构造效应,不是选股能力。
  若零成本下优势**基本不变**
      → 低换手确实带来了可检出的选股增量。

同时报告双方的组合年化波动率与每次调仓的名单更换率,把机制说清楚。
本节为描述性拆解,不改 D2/D3 的判定(那两条已按事前判据判为通过)。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import (  # noqa: E402
    CACHE,
    NBR,
    OUT,
    SEED,
    build_sel,
    draw_decile1,
    draw_neighbour,
)
from codex_r10_replication import DATA, TOP_N, WEIGHT, WINDOWS  # noqa: E402

NFREE = 40   # 零成本对照的种子数(比 100 少,只为看优势是否缩水)


def run_free(cl, ok, sel, cal_pos, w0, w1):
    """零成本、等权、可拆分:名单固定期间按日等权收益链乘。"""
    days = cal_pos[(cal_pos >= w0) & (cal_pos <= w1)]
    eq, held = 1.0, None
    curve = np.empty(len(days))
    turn = []
    prev = None
    for k, t in enumerate(days):
        if held is not None and len(held):
            r = cl[t, held] / cl[t - 1, held] - 1.0
            r = r[np.isfinite(r)]
            eq *= 1.0 + (float(r.mean()) if len(r) else 0.0)
        curve[k] = eq
        if int(t) in sel:
            cols = sel[int(t)][0]
            cols = cols[ok[t, cols]] if len(cols) else cols
            if prev is not None:
                turn.append(1.0 - len(np.intersect1d(prev, cols)) / max(len(cols), 1))
            prev = cols
            held = cols
    return curve, days, float(np.mean(turn)) if turn else np.nan


def stats(curve, days, idx):
    yrs = (idx[days[-1]] - idx[days[0]]).days / 365.25
    r = np.diff(curve) / curve[:-1]
    r = r[np.isfinite(r)]
    return (curve[-1] ** (1 / yrs) - 1, float(r.std(ddof=1) * np.sqrt(252)))


def main():
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    cl, ok = z["CL"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    assert (len(idx), len(codes)) == (3297, 5217)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)
    sel, elig, selrank = build_sel(reb, ok, logcap, tmean)

    def win(w):
        d0, d1 = WINDOWS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    rows = []
    print(f"{'窗口':6s} {'口径':22s} {'策略年化':>9s} {'对照中位':>9s} {'优势':>9s} "
          f"{'策略vol':>8s} {'对照vol':>8s} {'策略换手':>9s} {'对照换手':>9s}")
    for w in ("full", "oos"):
        w0, w1 = win(w)
        curve, days, sturn = run_free(cl, ok, sel, cal_pos, w0, w1)
        scagr, svol = stats(curve, days, idx)
        for kind, name in (("nbr", "零成本 · 对照A邻域"), ("dec", "零成本 · 对照B最小十分位")):
            cg, cv, ct = [], [], []
            for s in range(NFREE):
                rng = np.random.default_rng(SEED + s)
                csel = {}
                for t in sel:
                    order, ranks = elig[t], selrank[t]
                    ps = (draw_neighbour(rng, order, ranks, NBR) if kind == "nbr"
                          else draw_decile1(rng, order))
                    csel[t] = (order[ps], np.full(TOP_N, WEIGHT))
                c2, d2, tn = run_free(cl, ok, csel, cal_pos, w0, w1)
                a, v = stats(c2, d2, idx)
                cg.append(a)
                cv.append(v)
                ct.append(tn)
            med = float(np.median(cg))
            print(f"{w:6s} {name:22s} {scagr:+8.2%} {med:+9.2%} "
                  f"{(scagr-med)*100:+8.2f}pp {svol:7.1%} {np.median(cv):7.1%} "
                  f"{sturn:8.1%} {np.median(ct):8.1%}")
            rows.append({"window": w, "control": kind, "strat_cagr_free": scagr,
                         "ctrl_med_cagr_free": med, "edge_pp_free": (scagr - med) * 100,
                         "strat_vol": svol, "ctrl_vol": float(np.median(cv)),
                         "strat_turnover": sturn, "ctrl_turnover": float(np.median(ct))})
    pd.DataFrame(rows).to_csv(f"{OUT}/codex_r10_decompose.csv", index=False)
    print(f"\n落库 {OUT}/codex_r10_decompose.csv")
    print("\n对照:含成本口径的优势(§114 D2/D3 已跑)")
    print("  D2 full 策略+25.63% 对照中位+14.59% → +11.04pp   oos +36.21%/+13.38% → +22.83pp")
    print("  D3 full 策略+25.63% 对照中位+16.31% → +9.32pp    oos +36.21%/+12.29% → +23.92pp")


if __name__ == "__main__":
    main()

# =============================================================================
# §114 结论
#
# D1 引擎恒等式 ✓(最大相对误差 0.000e+00,成交数与冻结数逐项相等)
# 锚点 A1 ✓ (3297,5217);A2 抽样越界 0 次 ✓;A3 宁德 2021-11-30 = 13,815.5 亿 ✓
#
# D2 通过。full 策略 +25.63% vs 对照A中位 +14.59%(95分位 +18.85%)p=0.0099;
#          oos  +36.21% vs +13.38%(95分位 +22.66%)p=0.0099。
# D3 通过。full 策略 +25.63% vs 对照B中位 +16.31%(95分位 +20.81%)p=0.0198;
#          oos  +36.21% vs +12.29%(95分位 +20.68%)p=0.0099。
#
# **事前预测 P2、P3 均被推翻,我错了。** 我预判"低换手在同市值邻域内没有
# 可检出增量"以及"策略打不过最小十分位随机",两条都不成立。
# 而且拆掉成本之后仍然不成立(见上表):零成本下 full 仍有 +7.33~10.34pp、
# oos +11.46~15.50pp。
#
# 机制拆解(full 窗口 D3 的 +9.32pp):
#   成本差 1.99pp(策略每次调仓换 51.4% 名单,对照 93.9%)
#   波动拖累差 ~2.0pp(策略年化波动 26.2%,对照 32.9%)
#   剩余 ~5.3pp 为真实横截面增量
#
# D4(a) 市值十档等权(不含成本)阶梯**严格单调**:
#   full D1 +26.4% → D10 +8.1%;oos D1 +32.7% → D10 +6.7%
#   十档单调是"这不是噪声"的强证据(不是单点显著,是一条斜坡)。
#   策略(付全部成本)+25.63% ≈ 最小档等权(不付成本)+26.40%,差 -0.77pp。
#
# D4(b) 卖出被冻结(停牌/退市挂单卖不掉):full 153 次、train 149 次、oos 2 次。
#   **事前预测 P4 命中**(预测 >50)。冻结几乎全部集中在 2014–2019,
#   与 2015 年大面积停牌一致。这是回测相对现实偏乐观的一处。
#
# D4(c) 剔除最小市值后:剔 5% → full 年化 +26.23%;剔 10% → +28.54%。
#   **不降反升** —— R10 的收益不依赖极端微盘,容量顾虑比 Codex 自己担心的小。
#
# D4(d) 分红口径(见 codex_r10_dividend.py):
#   Codex 混口径(组合含分红/基准不含)full 超额 +3034.34pp
#   自洽A 两边都含分红                  full 超额 +1299.97pp
#   自洽B 两边都不含分红                full 超额  +955.28pp
#   本组合自身股息率 2.44%/年,高于 510300 的 1.64%/年 —— 与我此前的判断相反,
#   我原以为微盘股不分红,写在正文里:**我错了**。
#
# 与 §113 合并的完整修正:
#   Codex 公布  full +3135.06%  年化 33.62%  回撤 -41.55%  超额 +3034.34pp
#   修正市值后  full +1444.01%  年化 25.63%  回撤 -56.38%
#   再自洽分红  full 超额 +1299.97pp(都含)/ +955.28pp(都不含)
#   → 收益腰斩、回撤恶化 15pp,**但方向性结论仍然成立且通过了市值中性对照**。
#
# 未解决的问题(不在本节判据内,如实记录):
#   ① R10 是他 130 个策略变体里挑出的最好一条,本节的对照**不能**校正
#      这个挑选偏差;2023–2025"样本外"也在他的搜索范围内。
#   ② 零成本版本用每日等权再平衡,含成本版本用权重漂移,两者的水平不可直接比,
#      只有"优势"可比。
#   ③ 退市/停牌持仓被冻结在最后有效价(照抄 Codex 引擎),现实中退市整理期
#      通常接近全损,故 full 窗口偏乐观。
# =============================================================================
