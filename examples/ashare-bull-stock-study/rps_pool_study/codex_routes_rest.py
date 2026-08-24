"""§118 跑完 Codex 剩余路线:R01/R02/R03/R04/R05/R07/R13,并给 R08/R09 补 2026 留出期。

范围与理由
----------
§113–§114 做完 R10,§117 做完 R06/R08/R09/R11。本节补齐其余,
使 R01–R13 全部有「他的数字 vs 我的数字」并排的一行。
**R12 不跑**:它是叠加在 R11 之上的状态开关,而 §117 已测出 R11 的 oos p=0.3333
(合成后反而不如组件),在一个已失效的底座上测开关没有信息量;
这一条在正文里明说,不是遗漏。

照抄的定义(逐条来自他的源码/配置,不是我的解读)
------------------------------------------------
R01 `sma50_200`   510300 单标的择时:close>SMA200 且 SMA50>SMA200 时持有,否则现金
R02 `momentum_250` 510300 单标的择时:250 日收益>0 时持有,否则现金
    —— 这两条是**单资产择时**,不是选股,**不适用市值中性对照**,只作描述性比较。
R03 `120`  fast_rps_backtest.py:134 `nsmallest(TOP_N)`,selection-mode="low"
    rps_120 = 120 日收益的当日横截面百分位(0–100),取**最低**的 20 只(低位反转)
R04 `rsi_21_threshold_20`  fast_mean_reversion_backtest.py:29 阈值 20.0
    合格 = RSI21 < 20,取 RSI21 最低的 20 只
    **假设标注**:他的 score 列未在归档里出现,本节取「最超卖优先」,
    若他实际用的是别的排序,本条复现可能有偏差 —— 结果里如实标注。
R05 `squeeze_low_width`  fast_breakout_trend_backtest.py:39,360
    信号 = 收盘突破 boll_upper_20_2;score = −boll_width_20_pct
    boll_width_20_pct = (upper20 − lower20) / close  (precompute_breakout_cache.py:93)
R07 `mfi_confirmed_trend`  fast_volume_price_backtest.py:34,41-43
    合格 = (trend_60>0) & (vwap_gap_20>0) & mfi_14 ∈ [50,80];score = pct(mfi_14)
R13 `r13_C_rps120_250`  bootstrap_r13_v002.py:122-123
    规则 = distance_to_high_250 ≥ −0.10 且 rps_120 ≥ 80 且 rps_250 ≥ 80 且
           ep_ttm_pit > 0 且 cfp_ttm_pit > 0 且 roe_level ≥ 8 且 cash_conversion_ttm ≥ 0.8
    score = 0.4·rps_120 + 0.3·rps_250 + 0.2·near_high + 0.1·fundamental_quality(分位)

引擎与口径与 §114/§117 完全一致;估值因子的价格一律用**真实不复权价**
(§113/§117 的教训),并对 R13 同时跑他的前复权口径以量化差额。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
K1 锚点(不过则整节作废):面板 (3297,5217);抽样越界 0 次;
   TTM 恒等式(年报日 TTM = 当期累计,相对误差 <1e-6);泰格同比复现雪球真值。
K2 复现判据。用他的口径重跑,full(2014-01-02→2025-12-31)年化须落在他公布值 ±6pp 内。
   他公布:R01 +5.34%、R02 +4.42%、R03 +21.19%、R04 +0.65%、R05 +9.11%、R07 +4.03%。
   R13 他没给全区间年化(只给总收益 +42.16%、年化 +2.98%),用 +2.98%。
   各判各的,不过的那条判「无法复现」,其对照结论一并作废。
K3 市值中性对照(只对 5 条选股路线 R03/R04/R05/R07/R13,R01/R02 是单资产择时不适用)。
   同市值名次 ±25 邻域匹配随机 20 只,**200 组种子**(p 下限 1/201=0.00498)。
   **Bonferroni:5 条,α = 0.05/5 = 0.01。**
   K3 通过 ⟺ full 与 oos 两个窗口的 p 都 < 0.01。
K4 R08/R09 的 2026 干净留出期(§117 欠下的)。用 §116-A 同一套做法,
   holdout 2026-01-05 → 面板末,200 组种子,**p < 0.05**(确认性检验,不做 Bonferroni)。
   证据力弱(约 7 个月),只作弱证据记录。

事前预测(写下来以便被证伪;错了必须在正文里明说我错了)
--------------------------------------------------------
T1 K2 复现:R01/R02(单资产择时,逻辑最简单)会过;R04 会过(阈值明确)。
T2 **R03/R04/R05/R07/R13 五条,通过 K3 的数量 ≤ 1。**
   理由:他自己把这五条全判了淘汰或未过门槛;§115-B 里对应的
   mom/rev/near_high/trend 全部 p≈1.0 且年化为负。
T3 **R13 不会通过 K3。** 它是「新高 + RPS + 基本面」,与本项目 §90/§103/§105
   同一个假设,那三节全部 lift≈1.0;他自己也判「未通过 2023–2025 样本外门槛」。
T4 **R08 与 R09 至少一条通过 K4 的 2026 留出期。** 理由:§117 里两条的
   full 与 oos 的 p 都是 200 种子下限,信号很强;而 §116-A 里同样强度的
   small_cap_low_turnover 与 high_amihud 都过了 2026。
T5 R03 低 RPS 反转的复现会过 K2 但不过 K3 —— 他公布 +21.19% 排他表里第 2,
   我预测那是小盘 beta(低 RPS 的票偏小盘),市值中性之后消失。
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

NSEED, ALPHA = 200, 0.05 / 5
WINS = {"full": ("2014-01-02", "2025-12-31"), "oos": ("2023-01-03", "2025-12-31"),
        "hold": ("2026-01-05", "2026-08-03")}
CODEX = {"R01": .0534, "R02": .0442, "R03": .2119, "R04": .0065,
         "R05": .0911, "R07": .0403, "R13": .0298}
SELECT = ("R03", "R04", "R05", "R07", "R13")



def rsi(w, n=21):
    d = np.diff(w, axis=0)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au, ad = np.mean(up[-n:], axis=0), np.mean(dn[-n:], axis=0)
    return np.where(au + ad > 0, 100.0 * au / (au + ad), 50.0)


def mfi(h, lw, c, v, n=14):
    tp = (h + lw + c) / 3.0
    rmf = tp * v
    d = np.diff(tp, axis=0)
    pos = np.nansum(np.where(d > 0, rmf[1:], 0.0)[-n:], axis=0)
    neg = np.nansum(np.where(d < 0, rmf[1:], 0.0)[-n:], axis=0)
    return np.where(pos + neg > 0, 100.0 * pos / (pos + neg), 50.0)


def score(name, t, e, mode, cl, raw, amt, vol, hi, lo, fm):
    """五条选股路线在调仓日 t、合格集 e 上的分数(越大越优先),NaN = 不合格。"""
    px = (raw if mode == "raw" else cl)[t, e].astype(np.float64)
    w = cl[max(0, t - 260):t + 1, e].astype(np.float64)
    if name == "R03":                      # 低 RPS120 反转:取分位最低的 20 只
        r120 = cl[t, e] / cl[max(0, t - 120), e] - 1.0
        return -pd.Series(r120).rank(pct=True).to_numpy()
    if name == "R04":                      # RSI21 < 20,最超卖优先
        rs = rsi(w, 21)
        return np.where(rs < 20.0, -rs, np.nan)
    if name == "R05":                      # 突破布林上轨,带宽最窄优先
        m20 = np.mean(w[-20:], axis=0)
        s20 = np.std(w[-20:], axis=0, ddof=1)
        width = (4 * s20) / np.where(w[-1] > 0, w[-1], np.nan)
        return np.where(w[-1] > m20 + 2 * s20, -width, np.nan)
    if name == "R07":                      # 趋势为正 + 站上 VWAP + MFI∈[50,80]
        tr60 = cl[t, e] / cl[max(0, t - 60), e] - 1.0
        a20 = np.nansum(amt[max(0, t - 20):t + 1, e], axis=0)
        v20 = np.nansum(vol[max(0, t - 20):t + 1, e], axis=0)
        gap = cl[t, e] / np.where(v20 > 0, a20 / np.where(v20 > 0, v20, np.nan),
                                  np.nan) - 1.0
        s = slice(max(0, t - 60), t + 1)
        mf = mfi(hi[s, e], lo[s, e], cl[s, e], vol[s, e], 14)
        elig = (tr60 > 0) & (gap > 0) & (mf >= 50) & (mf <= 80)
        return np.where(elig, pd.Series(mf).rank(pct=True).to_numpy(), np.nan)
    if name == "R13":                      # 近 250 日新高 + RPS + PIT 基本面
        h250 = np.nanmax(cl[max(0, t - 249):t + 1, e].astype(np.float64), axis=0)
        near = cl[t, e] / h250 - 1.0
        p120 = pd.Series(cl[t, e] / cl[max(0, t - 120), e] - 1).rank(pct=True).to_numpy() * 100
        p250 = pd.Series(cl[t, e] / cl[max(0, t - 250), e] - 1).rank(pct=True).to_numpy() * 100
        ep, cfp = fm["eps_ttm"][t, e] / px, fm["ocfps_ttm"][t, e] / px
        roe = fm["roe_lvl"][t, e]
        conv = fm["ocfps_ttm"][t, e] / np.where(fm["eps_ttm"][t, e] != 0,
                                                fm["eps_ttm"][t, e], np.nan)
        elig = ((near >= -0.10) & (p120 >= 80) & (p250 >= 80) & (ep > 0)
                & (cfp > 0) & (roe >= 8.0) & (conv >= 0.8))
        sc = (0.4 * p120 / 100 + 0.3 * p250 / 100
              + 0.2 * pd.Series(near).rank(pct=True).to_numpy()
              + 0.1 * pd.Series(roe).rank(pct=True).to_numpy())
        return np.where(elig, sc, np.nan)
    raise ValueError(name)


def main():
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点K1a"
    print(f"锚点K1a ✓ {nt}×{ns}", flush=True)

    raw = np.full((nt, ns), np.nan, np.float32)
    hi = np.full((nt, ns), np.nan, np.float32)
    lo = np.full((nt, ns), np.nan, np.float32)
    vol = np.full((nt, ns), np.nan, np.float32)
    amt = np.full((nt, ns), np.nan, np.float32)
    t0 = time.time()
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet",
                            columns=["raw_close", "high", "low", "volume", "amount"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        x = x.reindex(idx)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().to_numpy(np.float32)
        for arr, col in ((hi, "high"), (lo, "low"), (vol, "volume"), (amt, "amount")):
            arr[:, j] = pd.to_numeric(x[col], errors="coerce").to_numpy(np.float32)
        if (j + 1) % 2000 == 0:
            print(f"  价量 {j+1}/{ns} ({time.time()-t0:.0f}s)", flush=True)
    print(f"价量矩阵完成 ({time.time()-t0:.0f}s)", flush=True)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点K1c TTM 恒等式不过"

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

    rows = []
    sig01 = ((bs > bs.rolling(200).mean())
             & (bs.rolling(50).mean() > bs.rolling(200).mean())).astype(float)
    sig02 = (bs / bs.shift(250) - 1.0 > 0).astype(float)
    for nm, sg in (("R01", sig01), ("R02", sig02)):
        r = {"route": nm, "codex_cagr": CODEX[nm], "kind": "择时"}
        for w in ("full", "oos", "hold"):
            d0, d1 = WINS[w]
            m = (bs.index >= d0) & (bs.index <= d1)
            px = bs[m]
            s = sg.shift(1).reindex(px.index).fillna(0.0)
            eq = (px.pct_change().fillna(0.0) * s).add(1.0).cumprod()
            yrs = (px.index[-1] - px.index[0]).days / 365.25
            r[f"{w}_cagr"] = float(eq.iloc[-1] ** (1 / yrs) - 1)
            r[f"{w}_bench"] = float(px.iloc[-1] / px.iloc[0] - 1)
            r[f"{w}_mdd"] = float((eq / eq.cummax() - 1).min())
        r["K2"] = bool(abs(r["full_cagr"] - r["codex_cagr"]) <= 0.06)
        rows.append(r)
        print(f"{nm} 他{r['codex_cagr']:+6.2%} | 我 full{r['full_cagr']:+7.2%}"
              f"(K2 {'✓' if r['K2'] else '✗'}) oos{r['oos_cagr']:+7.2%} "
              f"2026{r['hold_cagr']:+7.2%} 回撤{r['full_mdd']:+7.2%}", flush=True)

    def pick(fn, mode):
        sel, elig, srk = {}, {}, {}
        for t in reb:
            t = int(t)
            base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
            e = np.flatnonzero(base)
            if len(e) < TOP_N * 3:
                continue
            v = fn(t, e, mode)
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
        return sel, elig, srk

    viol = 0

    def ctrl_p(sel, elig, srk, w, target):
        nonlocal viol
        w0, w1 = wpos(w)
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
        return float(np.median(a)), (1 + int(np.sum(a >= target))) / (NSEED + 1)

    for name in SELECT:
        r = {"route": name, "codex_cagr": CODEX[name], "kind": "选股"}
        for mode in ("qfq", "raw"):
            sel, elig, srk = pick(
                lambda t, e, m, n=name: score(n, t, e, m, cl, raw, amt, vol, hi, lo, fm),
                mode)
            r[f"{mode}_nreb"] = len(sel)
            for w in ("full", "oos", "hold"):
                if not sel:
                    r[f"{mode}_{w}_cagr"] = np.nan
                    continue
                w0, w1 = wpos(w)
                eq, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, sel,
                                               cal_pos, w0, w1)
                mm = metrics(eq, dd, idx)
                r[f"{mode}_{w}_cagr"] = mm["cagr"]
                r[f"{mode}_{w}_mdd"] = mm["mdd"]
            if mode == "raw" and sel:
                for w in ("full", "oos"):
                    med, p = ctrl_p(sel, elig, srk, w, r[f"raw_{w}_cagr"])
                    r[f"ctrl_{w}_med"], r[f"p_{w}"] = med, p
        r["K2"] = bool(abs(r.get("qfq_full_cagr", np.nan) - r["codex_cagr"]) <= 0.06)
        r["K3"] = bool(r.get("p_full", 1) < ALPHA and r.get("p_oos", 1) < ALPHA)
        rows.append(r)
        print(f"{name} 他{r['codex_cagr']:+6.2%} | 他口径{r.get('qfq_full_cagr', np.nan):+7.2%}"
              f"(K2 {'✓' if r['K2'] else '✗'}) | 真实{r.get('raw_full_cagr', np.nan):+7.2%} "
              f"oos{r.get('raw_oos_cagr', np.nan):+7.2%} 2026{r.get('raw_hold_cagr', np.nan):+7.2%}"
              f" | p_full={r.get('p_full', np.nan):.4f} p_oos={r.get('p_oos', np.nan):.4f}"
              f" K3 {'✓' if r['K3'] else '✗'} 调仓日{r.get('raw_nreb', 0)}", flush=True)

    print("\nK4 R08/R09 的 2026 干净留出期", flush=True)
    for name in ("R08", "R09"):
        sel, elig, srk = pick(
            lambda t, e, m, n=name: route_scores(n, t, e, fm, cl, raw, logcap,
                                                 tmean, "raw"), "raw")
        w0, w1 = wpos("hold")
        eq, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        sc = metrics(eq, dd, idx)["cagr"]
        med, p = ctrl_p(sel, elig, srk, "hold", sc)
        rows.append({"route": name + "_2026hold", "kind": "留出期",
                     "raw_hold_cagr": sc, "ctrl_hold_med": med, "p_hold": p,
                     "K4": bool(p < 0.05)})
        print(f"  {name} 2026 年化{sc:+7.2%} 对照中位{med:+7.2%} p={p:.4f} "
              f"K4 {'✓' if p < 0.05 else '✗'}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n锚点K1b 抽样越界 {viol} 次 {'✓' if viol == 0 else '✗ 作废'}")
    assert viol == 0
    k2 = df[df["kind"].isin(["择时", "选股"])]["K2"]
    sk = df[df["kind"] == "选股"]
    print(f"K2 复现通过 {int(k2.sum())}/{len(k2)};"
          f"K3 对照通过 {int(sk['K3'].sum())}/5(α={ALPHA}):"
          f"{', '.join(sk.loc[sk['K3'], 'route']) or '无'}")
    df.to_csv(f"{OUT}/codex_routes_rest.csv", index=False)
    print(f"落库 {OUT}/codex_routes_rest.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §118 结果:R01–R13 跑齐。**没有一条剩余路线扛住市值中性对照。**
#
# 锚点 K1a ✓ 3297×5217  K1b ✓ 抽样越界 0 次  K1c ✓ TTM 46,274 个年报点违例 0
#
# 路线  他公布   我(他的口径)  K2   真实口径   oos      2026     p_full  p_oos   K3
# R01   +5.34%   +7.40%       ✓   —(择时)  +2.62%   -8.52%   —       —      不适用
# R02   +4.42%   +5.90%       ✓   —(择时)  +6.76%   -0.71%   —       —      不适用
# R03  +21.19%  +16.67%       ✓   +16.67%  +21.20%  -19.24%  0.0050  0.0249  ✗
# R04   +0.65%   -1.22%       ✓    -1.22%   +4.78%  -10.11%  0.9900  0.8458  ✗
# R05   +9.11%   +0.95%       ✗    +0.95%   +3.88%  -26.39%  0.9005  0.1095  作废
# R07   +4.03%   -8.57%       ✗    -8.57%  -13.12%   -6.94%  1.0000  1.0000  作废
# R13   +2.98%   -3.43%       ✗    -3.43%   -8.76%  +11.31%  0.9950  1.0000  作废
#
# K2 复现 4/7;K3 对照通过 **0/5**。
# R05/R07/R13 判「无法复现」(差 8.16 / 12.60 / 6.41pp,超出 ±6pp),
# **按事前登记,这三条的对照结论一并作废** —— 不是「测出来不显著」,是「没测成」。
# 原因是他这三条的信号定义在归档里不完整(布林 squeeze 的信号列、
# MFI 的 trend_60/vwap_gap_20 口径、R13 的 fundamental_quality 合成方式),
# 我按最自然的读法实现,与他不一致。**不猜、不调参去凑他的数字。**
#
# 真正有效的判定只有 R03 与 R04 两条,**都不通过**:
#   R03 低 RPS 反转:他排名第 2 的路线,full p=0.0050 但 oos p=0.0249 > α=0.01,
#       且 **2026 年 -19.24%**。
#   R04 超卖反转:full p=0.9900,几乎所有同市值随机对照都跑赢它。
#
# ── K4 R08/R09 的 2026 干净留出期 ──
#   R08 年化 +1.61%,对照中位 -13.86%,p=0.0697  ✗
#   R09 年化 -1.60%,对照中位 -15.32%,p=0.0697  ✗
#   **方向对但没过线。** 两条都比同市值随机对照高出约 15pp,可 200 组对照里
#   有 14 组跑赢,p 卡在 0.0697。2026 只有 7 个月、7 个调仓日,对照方差极大。
#
# ── 事前预测:5 个错 1 个 ──
# T1 ✓ R01(+7.40 vs +5.34)、R02(+5.90 vs +4.42)、R04(-1.22 vs +0.65)复现全过。
# T2 ✓ 五条里通过 K3 的数量 0 ≤ 1。
# T3 ✓ R13 不通过(p_full=0.9950、p_oos=1.0000),与本项目 §90/§103/§105
#      以及他自己的「未通过 2023–2025 样本外门槛」三方独立一致。
# T4 **错。** 我预测「R08 与 R09 至少一条通过 K4」,理由是它们在 §117 的 p
#      都打到了种子下限、信号很强。实际**两条都没过**(p=0.0697)。
#      我又一次把「样本内信号强」外推成「样本外会过」——
#      §116-A 的 small_cap 已经用 -18.16% 教过我同一件事,这是**第八次**。
# T5 ✓ R03 复现得了(K2 ✓)但市值中性之后不显著(K3 ✗),与我预判的
#      「低 RPS 的票偏小盘,那是小盘 beta」方向一致。
#
# ── R01–R13 全表结论 ──
# 通过市值中性对照且在 2026 干净留出期也站得住的:**只有 R10 家族的
# small_cap_low_turnover 与 high_amihud(§116-A)**。
# R08 价值、R09 质量在 full/oos 上通过对照(§117)但**未过 2026 留出期**。
# 其余全部不通过或无法复现。
# =============================================================================
