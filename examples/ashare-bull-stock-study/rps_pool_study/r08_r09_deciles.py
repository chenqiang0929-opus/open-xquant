"""第一六九节 事前登记:R08 / R09 的分数要到多少才有用 —— 十分位后验(结果未跑)。

起因
----
用户问:「Codex 模板增加了 R08 和 R09,但我不知道这个 R08 和 R09 要如何使用,
在多少分值的情况下,后续收益率可能更高?」

**必须先说清楚一件事。** R08 与 R09 在第一一七节通过检验的形式是
**横截面 TopN=20 等权组合、20 日调仓** —— 通过的是「**每期买全市场分数最高的 20 只**」
这件事,**不是「某只股票分数高就该买」**。分数是**当日全市场的百分位**(0–1),
0.9 的意思是「今天排在全市场前 10%」,**不是一个跨时间可比的绝对值**。
本节要量的正是:**这个百分位与后续收益的关系是什么形状**,
以及**能不能拿它当阈值用**。

做法
----
调仓日 = 与第一一七节完全相同的 `cal_pos[::20]`(2014-01-02 → 2026-08-20)。
每个调仓日在合格集上算 R08 / R09,按分数升序切 **10 个等份**,
每份等权持有到下一个调仓日,链式相乘。**价格口径用「真实口径」(不复权价)** ——
第一一七节已证 R08 的估值三比值在前复权价下被污染(全窗 −1.92pp),R09 不含价格。

**两套口径都跑,并把差报出来:**
  - **引擎口径**(与第一一七节同一套 `run_window_fast`:次日开盘、整手、
    停牌/涨跌停不可成交)—— 只用于 TopN=20 锚点,十分位不用(持仓 480 只太慢);
  - **收盘对收盘口径**(调仓日收盘 → 下一调仓日收盘,等权平均)—— 十分位用这套。
**必报桥接项**:TopN=20 在两套口径下的年化差,读者据此知道方法学缺口有多大。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
D1 锚点(不过则本节作废)
   (a) 面板 (3297, 5217);
   (b) TTM 恒等式违例 = 0;(c) 泰格 300347 同比复现违例 = 0;
   (d) **TopN=20 引擎年化必须复现第一一七节记录**:
       R08 前复权 full **+16.38%**、真实口径 full **+14.46%**;
       R09 两种口径 full 均 **+8.50%**。**容差 ±1.0pp,超出即作废。**

D2 **主判据:单调性**(在 oos 窗口 2023-01-03→2025-12-31 判,full 只报数)
   十分位年化对分位序号 1..10 的 **Spearman ρ**:
   **ρ ≥ 0.60 → 判「分数可以当阈值用」(高分位确实系统性更好);
     ρ < 0.60 → 判「只能当排序用,不可设阈值」。**
   R08 与 R09 **各判各的**。

D3 描述(不参与判定):每分位年化、与「全部合格股等权」基准之差、
   D10−D1 价差、逐年、各分位平均持股数、分数缺失率。

**判据写法自律**:绝对阈值,不写比值判据(第一五四节 A3 的教训)。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。**只登记判据。**

不做的
------
不改 `src/oxq/`;不调 TOP_N / 调仓间隔 / 合格口径;**跑完不许回头改分位数再跑**;
不新增顶层目录;不 force push;**不往 quant-research-dev / etf-netflow-dev 推**;
**不作任何可交易性声明** —— 本节是后验描述,不是买入建议。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codex_r10_neutral import CACHE, run_window_fast  # noqa: E402
from codex_r10_replication import DATA, TOP_N, WEIGHT, metrics  # noqa: E402
from codex_routes_rerun import WINS, build_fund, route_scores  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NBUCKET = 10
ANCHOR = {("R08", "qfq"): 0.1638, ("R08", "raw"): 0.1446,
          ("R09", "qfq"): 0.0850, ("R09", "raw"): 0.0850}
TOL = 0.010


def ann(eqv, ndays):
    """把链式净值换成年化。ndays = 期间的交易日数。"""
    if eqv <= 0 or ndays <= 0:
        return np.nan
    return eqv ** (250.0 / ndays) - 1.0


def main():  # noqa: PLR0915
    t0 = time.time()
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), f"锚点D1a {(nt, ns)}"
    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    truth = {(2017, "中报"): .5307, (2017, "三季报"): 1.0103,
             (2017, "年报"): 1.1401, (2018, "一季报"): 1.2107}
    bad = [k for k, v in truth.items() if abs(float(y.get(k, np.nan)) - v) > 0.005]
    assert not bad, f"锚点D1c 不过 {bad}"
    print(f"锚点D1a ✓ {nt}×{ns};D1c ✓ 泰格违例 0", flush=True)

    raw = np.full((nt, ns), np.nan, np.float32)
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点D1b TTM 恒等式不过"
    print(f"锚点D1b ✓ TTM 违例 0;不复权价矩阵完成 ({time.time()-t0:.0f}s)", flush=True)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = [int(t) for t in cal_pos[::20]]
    ipos = pd.Index(idx)

    # ---- 每个调仓日的分数与合格集(算一次,两个用途共用)----
    sc = {}
    for name in ("R08", "R09"):
        for mode in ("qfq", "raw"):
            d = {}
            for t in reb:
                base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
                e = np.flatnonzero(base)
                if len(e) < TOP_N * 3:
                    continue
                v = route_scores(name, t, e, fm, cl, raw, logcap, tmean, mode)
                d[t] = (e, v)
            sc[(name, mode)] = d
    print(f"分数矩阵完成({len(reb)} 个调仓日)({time.time()-t0:.0f}s)", flush=True)

    # ---- D1(d) TopN=20 引擎锚点 ----
    rows_anchor = []
    for (name, mode), d in sc.items():
        sel = {}
        for t, (e, v) in d.items():
            g = np.isfinite(v)
            if g.sum() < TOP_N:
                continue
            e2, v2 = e[g], v[g]
            sel[t] = (e2[np.argsort(-v2, kind="stable")[:TOP_N]],
                      np.full(TOP_N, WEIGHT))
        d0, d1 = WINS["full"]
        w0 = int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0])
        w1 = int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0])
        eq, dd, tr, fz = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        m = metrics(eq, dd, idx)
        exp = ANCHOR[(name, mode)]
        ok_a = abs(m["cagr"] - exp) <= TOL
        rows_anchor.append({"路线": name, "价格口径": mode, "引擎年化": m["cagr"],
                            "第117节记录": exp, "差pp": (m["cagr"] - exp) * 100,
                            "过": ok_a})
        print(f"锚点D1d {name}/{mode}:引擎 full 年化 {m['cagr']:+.2%} "
              f"(记录 {exp:+.2%},差 {(m['cagr']-exp)*100:+.2f}pp) "
              f"{'✓' if ok_a else '✗ 本节作废'}", flush=True)
    if not all(r["过"] for r in rows_anchor):
        pd.DataFrame(rows_anchor).to_csv(f"{OUT}/r08_r09_deciles_anchor.csv",
                                         index=False, encoding="utf-8-sig")
        print("锚点D1d 不过,按登记作废,不出十分位结果。")
        return

    # ---- 收盘对收盘口径:十分位 + TopN20 桥接 + 全市场等权基准 ----
    def c2c(members_of):
        """members_of(t, e, v) -> 下标数组;返回 (链式净值, 交易日数, 平均持股)。"""
        eqv, nd, held = 1.0, 0, []
        for k, t in enumerate(reb[:-1]):
            if t not in sc[("R08", "raw")]:
                continue
            t2 = reb[k + 1]
            if not (w0 <= t < w1):
                continue
            e, v = cur[t]
            mem = members_of(t, e, v)
            if mem is None or len(mem) == 0:
                continue
            p0 = cl[t, mem].astype(np.float64)
            p1 = cl[min(t2, nt - 1), mem].astype(np.float64)
            g = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0)
            if g.sum() == 0:
                continue
            eqv *= 1.0 + float(np.mean(p1[g] / p0[g] - 1.0))
            nd += t2 - t
            held.append(int(g.sum()))
        return eqv, nd, (float(np.mean(held)) if held else np.nan)

    out, mono = [], []
    for name in ("R08", "R09"):
        cur = sc[(name, "raw")]
        for wname in ("full", "oos"):
            d0, d1 = WINS[wname]
            w0 = int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0])
            w1 = int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0])
            base_eq, base_nd, base_h = c2c(lambda t, e, v: e)
            base_a = ann(base_eq, base_nd)
            top_eq, top_nd, _ = c2c(
                lambda t, e, v: e[np.isfinite(v)][
                    np.argsort(-v[np.isfinite(v)], kind="stable")[:TOP_N]]
                if np.isfinite(v).sum() >= TOP_N else None)
            aa = []
            for k in range(NBUCKET):
                def mk(t, e, v, k=k):
                    g = np.isfinite(v)
                    if g.sum() < NBUCKET * 5:
                        return None
                    e2, v2 = e[g], v[g]
                    o = np.argsort(v2, kind="stable")          # 升序:D1 最低分
                    lo = int(round(k * len(o) / NBUCKET))
                    hi = int(round((k + 1) * len(o) / NBUCKET))
                    return e2[o[lo:hi]]
                eqv, nd, h = c2c(mk)
                a = ann(eqv, nd)
                aa.append(a)
                out.append({"路线": name, "窗口": wname, "分位": f"D{k+1}",
                            "分位含义": f"分数第 {k*10}–{(k+1)*10} 百分位",
                            "年化": a, "相对全市场等权pp": (a - base_a) * 100,
                            "平均持股": h})
            r = pd.Series(aa).corr(pd.Series(range(1, NBUCKET + 1)), method="spearman")
            spread = (aa[-1] - aa[0]) * 100
            judge = ("可以当阈值用" if (wname == "oos" and r >= 0.60)
                     else ("只能当排序用,不可设阈值" if wname == "oos" else "—"))
            mono.append({"路线": name, "窗口": wname, "Spearman_ρ": r,
                         "D10−D1_pp": spread, "全市场等权年化": base_a,
                         "TopN20_收盘口径年化": ann(top_eq, top_nd),
                         "判定(D2)": judge})
            print(f"\n[{name} / {wname}] 全市场等权 {base_a:+.2%};"
                  f"TopN20(收盘口径) {ann(top_eq, top_nd):+.2%}")
            print("  " + "  ".join(f"D{i+1} {a:+.1%}" for i, a in enumerate(aa)))
            print(f"  Spearman ρ = {r:.3f};D10−D1 = {spread:+.2f}pp;"
                  f"D2 判定:{judge}", flush=True)

    ad = pd.DataFrame(rows_anchor)
    md = pd.DataFrame(mono)
    dd_ = pd.DataFrame(out)
    ad.to_csv(f"{OUT}/r08_r09_deciles_anchor.csv", index=False, encoding="utf-8-sig")
    md.to_csv(f"{OUT}/r08_r09_deciles_mono.csv", index=False, encoding="utf-8-sig")
    dd_.to_csv(f"{OUT}/r08_r09_deciles.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 90)
    print(md.to_string(index=False))
    print("\n桥接项(TopN=20 两套口径的差,用来标明方法学缺口):")
    for _, r in ad[ad["价格口径"] == "raw"].iterrows():
        tt = md[(md["路线"] == r["路线"]) & (md["窗口"] == "full")]
        if len(tt):
            print(f"  {r['路线']}:引擎 {r['引擎年化']:+.2%} vs 收盘口径 "
                  f"{tt.iloc[0]['TopN20_收盘口径年化']:+.2%};"
                  f"差 {(r['引擎年化']-tt.iloc[0]['TopN20_收盘口径年化'])*100:+.2f}pp")
    print(f"\n完成 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
