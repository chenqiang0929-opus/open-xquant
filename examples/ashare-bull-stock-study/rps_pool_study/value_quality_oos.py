"""§119 给 R08 价值 / R09 质量一个正规的时间样本外。

为什么要重做
------------
§117 里 R08/R09 在 full(2014–2025)与 oos(2023–2025)两个窗口都通过了
市值中性对照,p 全为 200 组种子的下限。但**这两个窗口都在 Codex 的搜索范围内**。
§118 补的 2026 干净留出期方向是对的(年化 +1.61% / −1.60%,对照中位
−13.86% / −15.32%,高出约 15pp),但**只有 7 个月、7 个调仓日**,
200 组对照里有 14 组跑赢,p 卡在 0.0697,过不了 0.05。

本节改用正规切法:**训练期 2014-01-02 → 2021-12-31(8 年,约 98 个调仓日),
留出期 2022-01-04 → 面板末(约 4.6 年,约 55 个调仓日)**。
留出期样本量是 2026 的 8 倍,足以给出确定答案。

**留出期只看一次。无论结果如何,不回头改因子定义、不改阈值、不换窗口。**

因子定义与 §117 完全一致(照抄他的源码),价格一律用真实不复权价:
  R08 value_composite      = ep_ttm / bp / cfp_ttm 三项 where(>0) 后 rank 均值
  R09 core_quality_composite = margin(>0)/roe(>0)/roe_chg(ep>0)/
                               cash_conv(ep>0&cfp>0&自身>0) 四项 winsor_rank 均值
引擎与口径与 §114/§117 一致;TTM 复用 `fundamental_yoy.label_periods`。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
L1 锚点(不过则整节作废):面板 (3297,5217);抽样越界 0 次;
   TTM 恒等式(年报日 TTM = 当期累计);泰格同比复现雪球真值。

L2 训练期确认。R08 与 R09 在训练期 2014–2021 上须通过同市值邻域匹配随机对照
   (**500 组种子**,p 下限 1/501=0.001996)。
   L2 通过 ⟺ p < 0.05/2 = 0.025(两条路线的 Bonferroni)。
   训练期不过的那条,其留出期结果不作数(说明它连样本内都不成立)。

L3 **留出期判定。核心判据,只看一次。**
   在 2022-01-04 → 面板末 上跑同一套对照,500 组种子。
   L3 通过 ⟺ p < 0.05/2 = 0.025。

L4 描述项(不设阈值):留出期的年化、最大回撤、夏普、组合换手率、卖出冻结次数,
   以及相对 510300(含分红)的超额;并给出训练期与留出期的年化差,
   用于判断衰减幅度。

事前预测(写下来以便被证伪;错了必须在正文里明说我错了)
--------------------------------------------------------
U1 L2 训练期:R08 与 R09 **都会过**。理由:§117 里两条在包含 2014–2021 的
   full 窗口上 p 都是种子下限。
U2 **L3 留出期:R08 会过,R09 不会过。**
   理由:§118 的 2026 留出期上 R08 是 +1.61%(正)、R09 是 −1.60%(负),
   虽然两条的 p 相同,但 R08 的绝对方向更好;且 R09 的四项因子里
   roe_change_yoy 与 cash_conversion 依赖财报的边际变化,更容易衰减。
U3 留出期年化相对训练期 **下降幅度 > 5pp**(两条都是)。
   理由:2022–2026 覆盖了 2022 熊市与 2026 的风格切换,
   而训练期含 2015、2019–2021 三轮牛市。

**本条预测 U2 是我第九次做「样本内强→样本外」类的外推,前八次全错。
这次我明确写下:如果 U2 又错,说明我在这类外推上没有可用的先验,
今后不应再基于样本内强度预测样本外结果。**

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推;
**不因为留出期结果不好就回头改训练期的因子定义或窗口**;
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
from codex_r10_replication import DATA, TOP_N, WEIGHT, metrics  # noqa: E402
from codex_routes_rerun import build_fund, route_scores  # noqa: E402
from factor_sweep_pv import draw_fast  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

NSEED, ALPHA = 500, 0.05 / 2
WINS = {"train": ("2014-01-02", "2021-12-31"), "holdout": ("2022-01-04", "2026-08-03")}


def main():
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点L1a"
    print(f"锚点L1a ✓ {nt}×{ns}", flush=True)

    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    truth = {(2017, "中报"): .5307, (2017, "三季报"): 1.0103,
             (2017, "年报"): 1.1401, (2018, "一季报"): 1.2107}
    bad = [k for k, v in truth.items() if abs(float(y.get(k, np.nan)) - v) > 0.005]
    print(f"锚点L1d 泰格同比复现:违例 {len(bad)} 项 {'✓' if not bad else '✗'}", flush=True)
    assert not bad

    raw = np.full((nt, ns), np.nan, np.float32)
    t0 = time.time()
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    print(f"不复权价矩阵完成 ({time.time()-t0:.0f}s)", flush=True)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点L1c TTM 恒等式不过"

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill()
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)

    def wpos(w):
        d0, d1 = WINS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    viol, rows = 0, []
    for name in ("R08", "R09"):
        sel, elig, srk = {}, {}, {}
        for t in reb:
            t = int(t)
            base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
            e = np.flatnonzero(base)
            if len(e) < TOP_N * 3:
                continue
            v = route_scores(name, t, e, fm, cl, raw, logcap, tmean, "raw")
            g = np.isfinite(v)
            if g.sum() < TOP_N:
                continue
            e2 = e[g]
            top = e2[np.argsort(-v[g], kind="stable")[:TOP_N]]
            sel[t] = (top, np.full(TOP_N, WEIGHT))
            order = e[np.argsort(logcap[t, e], kind="stable")]
            rk = {int(c): i for i, c in enumerate(order)}
            elig[t] = order
            srk[t] = np.array([rk[int(c)] for c in top])
        r = {"route": name}
        for w in ("train", "holdout"):
            w0, w1 = wpos(w)
            nre = sum(1 for t in sel if w0 <= t <= w1)
            eq, dd, tr, fz = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
            m = metrics(eq, dd, idx)
            s = bs[(bs.index >= WINS[w][0]) & (bs.index <= WINS[w][1])]
            cg = []
            for sd in range(NSEED):
                rng = np.random.default_rng(SEED + sd)
                cs = {}
                for t in sel:
                    o, rk = elig[t], srk[t]
                    ps = draw_fast(rng, rk, len(o))
                    viol += int(np.sum(np.abs(ps - rk) > NBR))
                    cs[t] = (o[ps], np.full(TOP_N, WEIGHT))
                e3, d3, _, _ = run_window_fast(op, cl, susp, lu, ld, cs, cal_pos, w0, w1)
                cg.append(metrics(e3, d3, idx)["cagr"])
            a = np.array(cg)
            p = (1 + int(np.sum(a >= m["cagr"]))) / (NSEED + 1)
            r |= {f"{w}_cagr": m["cagr"], f"{w}_mdd": m["mdd"], f"{w}_sharpe": m["sharpe"],
                  f"{w}_total": m["total"], f"{w}_nreb": nre, f"{w}_frozen": fz,
                  f"{w}_bench": float(s.iloc[-1] / s.iloc[0] - 1),
                  f"{w}_ctrl_med": float(np.median(a)), f"{w}_p": p}
            print(f"  {name} {w:8s} 调仓{nre:3d} 年化{m['cagr']:+7.2%} 回撤{m['mdd']:+7.2%} "
                  f"夏普{m['sharpe']:5.2f} | 对照中位{np.median(a):+7.2%} p={p:.4f}",
                  flush=True)
        r["L2"] = bool(r["train_p"] < ALPHA)
        r["L3"] = bool(r["holdout_p"] < ALPHA)
        r["decay_pp"] = (r["holdout_cagr"] - r["train_cagr"]) * 100
        rows.append(r)
        print(f"{name}  L2 训练期 {'✓' if r['L2'] else '✗'}  "
              f"L3 留出期 {'✓' if r['L3'] else '✗'}  衰减 {r['decay_pp']:+.2f}pp\n", flush=True)

    df = pd.DataFrame(rows)
    print(f"锚点L1b 抽样越界 {viol} 次 {'✓' if viol == 0 else '✗ 作废'}")
    assert viol == 0
    print(f"L2 训练期通过 {int(df['L2'].sum())}/2;L3 留出期通过 {int(df['L3'].sum())}/2"
          f"(α={ALPHA}):{', '.join(df.loc[df['L3'], 'route']) or '无'}")
    df.to_csv(f"{OUT}/value_quality_oos.csv", index=False)
    print(f"落库 {OUT}/value_quality_oos.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §119 结果:R08 与 R09 **都通过了正规的时间样本外**。
#
# 锚点 L1a ✓ 3297×5217  L1b ✓ 抽样越界 0 次
#      L1c ✓ TTM 恒等式 46,274 个年报点违例 0   L1d ✓ 泰格同比复现违例 0
#
# 路线  窗口      调仓日  年化      回撤      夏普   对照中位   p       判定
# R08   train 14-21  96  +14.76%  -34.77%  0.76   +6.51%   0.0040  L2 ✓
# R08   hold  22-26  55   +9.98%  -22.67%  0.67   -1.58%   0.0020  L3 ✓
# R09   train 14-21  91   +8.75%  -49.83%  0.46   +2.13%   0.0180  L2 ✓
# R09   hold  22-26  55   +6.63%  -22.22%  0.49   -2.51%   0.0060  L3 ✓
#
# **L2 训练期 2/2;L3 留出期 2/2(α=0.025)。**
#
# 留出期的对照中位数是**负的**(-1.58% / -2.51%),而策略是 +9.98% / +6.63% ——
# 相对同市值随机对照的超额约 **11.6pp / 9.1pp**,而且这一段是
# **参数与因子定义完全没看过的 4.6 年、55 个调仓日**。
# 衰减也小:R08 -4.79pp、R09 -2.12pp。
# 留出期最大回撤 -22.67% / -22.22%,远好于修正版 R10 的 -56.38%。
#
# ── 事前预测:U1 命中,U2 与 U3 都错 ──
# U1 ✓ 训练期两条都过。
# U2 **错。** 我预测「R08 会过、R09 不会过」,理由是 §118 的 2026 留出期上
#    R08 是 +1.61%(正)而 R09 是 -1.60%(负)。**R09 也过了,而且 p=0.0060。**
#    我拿 7 个月、7 个调仓日的方向去外推 4.6 年、55 个调仓日的结论 ——
#    §118 我自己刚写过「2026 对照方差极大」,转头又用它做预测。
# U3 **错。** 我预测两条的留出期年化相对训练期「下降 > 5pp」,
#    实际 -4.79pp 与 -2.12pp,**两条都不到 5pp**。
#
# ── 兑现我在事前登记里写下的那句话 ──
# 登记原文:「U2 是我第九次做『样本内强→样本外』类外推,前八次全错。
# 这次我明确写下:如果 U2 又错,说明我在这类外推上没有可用的先验,
# 今后不应再基于样本内强度预测样本外结果。」
# **U2 又错了,而且这次错在悲观方向**(前八次多为乐观方向)——
# 两个方向都错,说明确实没有信号。
# **自本节起,不再对『样本外会不会过』下事前预测,只登记判据。**
#
# ── 必须写清的边界 ──
# ① 本节的留出期对**我**是干净的(我没在 2022–2026 上选过因子或调过参),
#    但对 **Codex 不干净** —— 他把 R08/R09 定为 WATCHLIST 时用的是 2014–2025,
#    包含本节的整个留出期。**因子的挑选偏差没有被本节校正。**
#    要彻底干净,需要一组**在 2022 年之前就已公开写死**的因子定义。
# ② 夏普不高(0.67 / 0.49),训练期回撤 R09 达 -49.83%。
# ③ 2026 单独一段仍然弱(§118:+1.61% / -1.60%,p=0.0697)——
#    留出期的成绩主要来自 2022–2025。
# =============================================================================
