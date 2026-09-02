"""第一七二节 事前登记:信号确认后遇到深调,还回得来吗(结果未跑)。

起因
----
用户看 688347 华虹公司的图形后问:「这段图形再仔细研究下,看看是否适用于全市场」。

688347 那一段的事实(第一七一节之后逐日核出来的):
  2025-02-19 强确认,收盘 60.28(RPS50 91.9)
  → 一路跌到 2025-04-07 收盘 40.76,**回撤 −32.4%**
  → 信号在 3 月、4 月**仍在给标准确认**(共 11 天),到 5 月才熄灭,而底部是 4 月 7 日
  → 平台筛选器在 3–7 月**104 个交易日一天信号都没有**
     (段内深度从 0.1891 走到 0.3896,一次性打穿 0.352 上限且再也回不来)
  → 直到 2025-08-04 收盘 63.10 才重新超过 2 月高点,**隔了 113 个交易日**
  → 此后到 2026-06-30 涨到 336.30

**一只股票的后视不是证据。** 本节把这个形状拆成可测的三问,在全市场全历史上量。

三个问题
--------
Q1 **深调有多普遍**:统一信号首次触发后 250 日内,发生 ≥30% 回撤的比例是多少?
Q2 **深调是不是致命的**:深调组与非深调组,触发后 250/500 日收益差多少?
   深调组里最终能重新站上触发价的比例是多少?
Q3 **平台筛选器在深调期间是不是普遍静默**(688347 的机制是否可推广)?

口径
----
- 面板 (3316, 5232),末日 2026-08-28;合格 = 非 ST、非停牌、上市满 250 日、有成交;
- **事件 = 统一信号由 0 变 1 的那一天(段首)**,X01 分档用 RPS50(第一六六节口径);
- 为让 500 日前瞻有完整数据,**只取末日之前至少 500 个交易日的事件**;
- 回撤 = `min(close[t+1 : t+250]) / close[t] − 1`,**深调 = ≤ −30%**;
- 「回得来」= 触发后 500 日内**收盘价重新 ≥ 触发日收盘**;
- 退市股按最后有效价 ffill 参与,**绝不剔除**(用户规则5)。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
G1 锚点(不过则本节作废)
   (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) **688347 复现**:2025-02-19 必须是一个事件日,且其 250 日内最大回撤
       落在 **−32.4% ± 1.0pp**,「回得来」的首日必须是 **2025-08-04**。
       —— 这是把全市场口径钉在已核过的个案上,算不出或对不上即作废。

G2 **主判据**(全部事件,市值五分位内做中性化)
   **深调组触发后 500 日收益中位数 ≥ 非深调组的 50%** → 判「深调不致命」;
   **< 50%** → 判「深调是致命的」。
   (写成绝对比例而非差值,是因为两组基数可能一正一负;
    **若非深调组中位数 ≤ 0,则本判据无意义,直接判定作废并如实说明** ——
    第一五四节 A3 就是栽在「比值判据在负区间含义翻转」上,这里先把出口写好。)

G3 **平台静默的可推广性**(描述转判定)
   深调组在「触发日 → 最低点」这段窗口里,**平台信号非空的天数占比中位数 < 5%**
   → 判「688347 的平台静默是普遍现象」;**≥ 5%** → 判「不普遍,是个案」。

G4 描述(不参与判定):Q1 的深调比例、两组的 250/500 日收益分布、
   「回得来」的比例与所需天数分布、按年份拆分、深调组触发时的 RPS50 分布。

**判据写法自律**:G2 已按第一五四节 A3 的教训写了负区间出口;其余为绝对阈值。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。**只登记判据。**

不做的
------
不改 `src/oxq/`;不调信号定义 / 深调阈值 / 前瞻窗口;**跑完不许回头改阈值再跑**;
不新增顶层目录;不 force push;**不往 quant-research-dev / etf-netflow-dev 推**;
**不作任何可交易性声明** —— 本节是后验描述,不是持仓规则。
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
from consolidation_screener import THR_ATR, THR_DEPTH, THR_SHRINK, load_panel  # noqa: E402
from panel_cache import cached  # noqa: E402
from platform_pivot import vec_screen  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR",
                      "/home/user/oxq-panel-0828/oxq_stock_market_fixed")
OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
STRONG_N = int(os.environ.get("OXQ_STRONG_N", "50"))
FWD, DEEP, NQ = 250, -0.30, 5
LONG = 500


def main():  # noqa: PLR0915
    t0 = time.time()
    import glob
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]

    def _build_panel():
        cols = ["close", "volume", "is_st", "is_suspended", "listed_days"]
        d = {c: {} for c in cols}
        for c in codes:
            x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=cols)
            if getattr(x.index, "tz", None) is not None:
                x.index = x.index.tz_localize(None)
            for k in cols:
                d[k][c] = x[k]
        cldf_ = pd.DataFrame(d["close"]).sort_index()
        idx_ = cldf_.index

        def al_(k, f=np.nan):
            return pd.DataFrame(d[k]).sort_index().reindex(
                index=idx_, columns=cldf_.columns).fillna(f).to_numpy()
        cl_ = cldf_.where(cldf_ > 0).ffill().to_numpy(np.float64)
        ldf_ = pd.DataFrame(al_("listed_days", 0)).replace(0, np.nan).ffill(
        ).fillna(0).to_numpy()
        okm_ = (~al_("is_st", True).astype(bool)
                & ~al_("is_suspended", True).astype(bool)
                & (ldf_ >= 250) & (al_("volume", 0) > 0) & np.isfinite(cl_))
        return {"idx": idx_.values.astype("datetime64[ns]"), "cl": cl_, "okm": okm_}
    p = cached("panel", DATA, _build_panel)
    idx = pd.DatetimeIndex(p["idx"])
    cl, okm = p["cl"], p["okm"]
    nt, ns = cl.shape
    assert (nt, ns) == (3316, 5232), f"锚点G1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点G1a 末日 {idx[-1].date()}"
    print(f"锚点G1a ✓ {(nt, ns)} 末日 {idx[-1].date()} ({time.time()-t0:.0f}s)",
          flush=True)

    def _build_plat():
        pcl, pframes, pstrong, pma100 = load_panel(DATA)
        if STRONG_N != 60:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                clw = pcl.where(pcl > 0)
                rp = (clw.pct_change(STRONG_N, fill_method="pad")
                      .rank(axis=1, pct=True) * 100)
                r60 = clw.pct_change(60, fill_method="pad").rank(axis=1, pct=True) * 100
            assert np.array_equal((r60 > 90).to_numpy(), pstrong), "强势日恒等断言不过"
            pstrong = (rp >= 90).to_numpy()
        if "510300" in pcl.columns:
            keep = [i for i, c in enumerate(pma100.columns) if c != "510300"]
            pcl = pcl.drop(columns=["510300"])
            pstrong = pstrong[:, keep]
        a, b, c_, d_, e_, f_, g_ = vec_screen(
            pcl.to_numpy(float), pframes, pstrong, pma100, idx, codes)
        del pframes
        return {"ts_a": a, "adj_a": b, "dep": c_, "shr": d_, "cnv": e_,
                "phi": f_, "plo": g_}
    q = cached("platform", DATA, _build_plat, extra=f"rps{STRONG_N}")
    hit3 = ((q["shr"] < THR_SHRINK) & (q["cnv"] < THR_ATR)
            & (q["dep"] <= THR_DEPTH) & (q["adj_a"] >= 0))
    print(f"平台就绪 ({time.time()-t0:.0f}s)", flush=True)

    # ---- X01 分档(RPS50)----
    px = pd.DataFrame(cl)
    lo250 = px.rolling(250, min_periods=250).min().to_numpy()
    ma20 = px.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        r120 = cl / np.where(np.roll(cl, 120, 0) > 0, np.roll(cl, 120, 0), np.nan) - 1.0
        r120[:120] = np.nan
        rps50 = (pd.DataFrame(cl).pct_change(50, fill_method=None)
                 .rank(axis=1, pct=True).to_numpy() * 100)
    mfr = pd.DataFrame((cl > ma20).astype(float)).rolling(
        120, min_periods=120).mean().to_numpy()
    base = (rec >= .40) & (r120 >= .10) & (mfr >= .55)
    uni = base & np.isfinite(rps50) & (rps50 >= 80) & okm
    print(f"分档就绪:统一信号=1 共 {int(uni.sum()):,} 个股票-日 "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 事件:段首,且留足 LONG 日前瞻 ----
    ev = np.zeros_like(uni)
    ev[1:] = uni[1:] & ~uni[:-1]
    ev[0] = uni[0]
    ev[nt - LONG:] = False
    ti, ji = np.nonzero(ev)
    print(f"事件(统一信号段首,且留足 {LONG} 日前瞻)共 {len(ti):,} 个", flush=True)

    rows = []
    for t, j in zip(ti, ji, strict=True):
        p0 = cl[t, j]
        seg = cl[t + 1:t + 1 + FWD, j]
        lng = cl[t + 1:t + 1 + LONG, j]
        if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(seg).any():
            continue
        mn = np.nanmin(seg)
        dd = mn / p0 - 1.0
        tmin = t + 1 + int(np.nanargmin(seg))
        back = np.flatnonzero(np.isfinite(lng) & (lng >= p0)) if np.isfinite(lng).any() \
            else np.array([], int)
        back = back[back > (tmin - t - 1)] if len(back) else back
        # 深调期间平台是否静默
        w0, w1 = t, tmin
        sil = float(hit3[w0:w1 + 1, j].mean()) if w1 > w0 else np.nan
        rows.append({
            "t": int(t), "j": int(j), "日期": idx[t].date(), "代码": codes[j],
            "触发价": p0, "RPS50": rps50[t, j],
            "回撤250": dd, "最低日": idx[tmin].date(),
            "见底用时": int(tmin - t),
            "收益250": cl[min(t + FWD, nt - 1), j] / p0 - 1.0,
            "收益500": cl[min(t + LONG, nt - 1), j] / p0 - 1.0,
            "回得来": bool(len(back)),
            "回来用时": int(back[0] + 1) if len(back) else -1,
            "深调期平台占比": sil,
            "深调": bool(dd <= DEEP)})
    e = pd.DataFrame(rows)
    print(f"有效事件 {len(e):,} 个 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 锚点 G1b:688347 ----
    a = e[(e["代码"] == "688347") & (e["日期"].astype(str) == "2025-02-19")]
    ok_b = False
    if len(a):
        r = a.iloc[0]
        ok_b = (abs(r["回撤250"] - (-0.324)) <= 0.010
                and str(r["日期"]) == "2025-02-19")
        bk = idx[int(r["t"]) + r["回来用时"]].date() if r["回得来"] else None
        ok_b = ok_b and str(bk) == "2025-08-04"
        print(f"锚点G1b 688347 2025-02-19:回撤 {r['回撤250']:+.1%}(期望 −32.4%±1.0pp)、"
              f"回得来首日 {bk}(期望 2025-08-04) {'✓' if ok_b else '✗ 本节作废'}",
              flush=True)
    else:
        print("锚点G1b ✗ 688347 2025-02-19 不是事件日,本节作废", flush=True)
    if not ok_b:
        e.to_csv(f"{OUT}/deepdraw_events.csv", index=False, encoding="utf-8-sig")
        return

    # ---- Q1/Q2/Q3 ----
    w = 96
    print(f"\n{'='*w}\nQ1 深调有多普遍\n{'='*w}")
    print(f"  事件 {len(e):,} 个;回撤 ≤ −30% 的 {int(e['深调'].sum()):,} 个 "
          f"= {e['深调'].mean():.1%}")
    print("  回撤分位:" + "  ".join(
        f"{p}% {np.percentile(e['回撤250'], p):+.1%}" for p in (10, 25, 50, 75, 90)))
    d1, d0 = e[e["深调"]], e[~e["深调"]]
    print(f"\n{'='*w}\nQ2 深调是不是致命的\n{'='*w}")
    for lab, g in (("深调组", d1), ("非深调组", d0)):
        print(f"  {lab} n={len(g):,}  250日中位 {g['收益250'].median():+.1%}  "
              f"500日中位 {g['收益500'].median():+.1%}  "
              f"回得来 {g['回得来'].mean():.1%}")
    m1, m0 = d1["收益500"].median(), d0["收益500"].median()
    if m0 <= 0:
        verdict = "判据作废:非深调组 500 日收益中位 ≤ 0,比值判据在此失去含义"
    else:
        verdict = ("深调不致命" if m1 >= 0.5 * m0 else "深调是致命的")
    print(f"  **G2:深调组 {m1:+.2%} vs 非深调组 {m0:+.2%};"
          f"比值 {m1/m0 if m0 else float('nan'):.3f} → {verdict}**")
    print(f"\n{'='*w}\nQ3 平台在深调期间是否静默\n{'='*w}")
    s = d1["深调期平台占比"].dropna()
    g3 = "688347 的平台静默是普遍现象" if s.median() < 0.05 else "不普遍,是个案"
    print(f"  深调组 n={len(s):,};「触发日→最低点」窗口内平台信号非空占比:"
          f"中位 {s.median():.2%}、均值 {s.mean():.2%}、"
          f"完全为 0 的占 {(s == 0).mean():.1%}")
    print(f"  **G3 判定:{g3}**")
    print(f"\n{'='*w}\nG4 描述\n{'='*w}")
    b = d1[d1["回得来"]]
    if len(b):
        print(f"  深调组里回得来的 {len(b):,} 个,用时(交易日)分位:" + "  ".join(
            f"{p}% {np.percentile(b['回来用时'], p):.0f}" for p in (25, 50, 75, 90)))
    e["年"] = pd.to_datetime(e["日期"]).dt.year
    t2 = e.groupby("年").agg(事件=("深调", "size"), 深调比例=("深调", "mean"),
                             收益500中位=("收益500", "median"),
                             回得来比例=("回得来", "mean"))
    print(t2.round(3).to_string())
    e.to_csv(f"{OUT}/deepdraw_events.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"G2判定": verdict, "深调组500日中位": m1, "非深调组500日中位": m0,
                   "深调比例": float(e["深调"].mean()), "事件数": len(e),
                   "G3判定": g3, "深调期平台占比中位": float(s.median())}]).to_csv(
        f"{OUT}/deepdraw_verdict.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/deepdraw_events.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
