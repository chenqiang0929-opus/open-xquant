"""§117 照抄重跑 Codex 的 WATCHLIST 路线:R06 / R08 / R09 / R11(R10 见 §113–§114)。

为什么改做法
------------
§115-B 我是「重新推导」—— 自己列 16 个因子去覆盖他的假设。结果除 R10 外
拿不出「他的数字 vs 我的数字」并排的那一行(我用 near_high_250 代替布林收缩突破、
用简单反转代替 RSI 阈值,根本不是同一个策略)。本节改为**照抄他的因子定义**。

范围:他 13 条路线里标 WATCHLIST 的 6 条,除 R10(§113–§114 已完成)与
R12(状态叠加层,不是选股因子)外的 4 条。被他自己判 REJECT 的 R01–R05/R07/R13
不重跑 —— §115-B 已在同方向独立失败(mom/near_high/trend 全部 p≈1.0 且年化为负)。

照抄的定义(逐条来自他的源码,不是我的解读)
--------------------------------------------
R06 `defensive_composite`  precompute_low_risk_cache.py + fast_low_risk_backtest.py
    vol_N   = 对数收益的滚动样本标准差(ddof=1),N ∈ {20,60,120}
    mdd_N   = 对数价格的滚动最大回撤,N ∈ {60,120,250}
    分数    = 六项各自 rank(pct=True, ascending=True) 后取均值,
              其中 vol 三项先乘 −1(低波动优先)
R08 `value_composite`      evaluate_value_factors.py:109-115
    ep_ttm_pit  = eps_ttm / close
    bp_pit      = book_value_per_share / close
    cfp_ttm_pit = ocfps_ttm / close
    分数 = 三项**先 where(>0)** 再 rank(pct=True) 后取均值(skipna=False)
R09 `core_quality_composite`  evaluate_quality_factors.py:30-66
    winsor_rank = 先按 eligible 掩码,再 clip 到 [1%, 99%] 分位,再 rank(pct=True)
    margin(>0) / roe_level(>0) / roe_change_yoy(ep>0) /
    cash_conversion(ep>0 & cfp>0 & 自身>0),四项均值 skipna=False
R11 `balanced_with_size_ex_micro10`  R11 文档第 2 节
    价值 / 质量 / 低风险 / 小市值低换手 各 25%,先各自转当日分位再合成,
    并剔除流通市值最小 10% 的股票
TTM:`single.rolling(4).sum()` 的等价式 —— 本年累计 + 上年年报 − 上年同期累计;
    报告期标签**复用** `fundamental_yoy.label_periods`,不重拼(§89 第一条规矩)。

引擎与口径:与 §114 完全一致(TopN=20、等权、20 日调仓、次日开盘、整手、
佣金/印花/滑点、ST/停牌/涨停/上市不足 365 日历日/零成交剔除),调仓日 153 个。

**两套价格口径都跑**(§113 的教训):
  「他的口径」= 前复权 close 做分母 / 做市值 —— 用来**复现**他的数字
  「真实口径」= 不复权 raw_close      —— 用来给出**修正值**
新发现:他的 `precompute_fundamental_pit_cache.py:165` 读的是
`market_dir = data/market`(Windows 路径 data\\market)的 `close` 列,而该目录的 close 已被他自己确认是
前复权价(宁德 2021-11-30 = 350.12)。**所以 ep/bp/cfp 三个估值因子的分母
与 R10 的市值是同一个 bug,R08 与 R11 都受影响;R09 四项都不含价格,是干净的。**

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
J1 锚点(不过则整节作废)
   (a) 面板 (3297, 5217);
   (b) 抽样恒等式:对照每次抽样的市值名次偏离 ≤25,违例 > 0 即作废;
   (c) TTM 恒等式:随机 200 只股票在其**年报**公告日当天,TTM 净利必须等于
       当期累计净利(年报累计即全年),相对误差 < 1e-6;
   (d) 报告期对齐锚点:复用 `yoy_series("300347")`,泰格 2017 中报/三季报/年报/
       2018 一季报同比须复现 0.5307/1.0103/1.1401/1.2107(±0.5pp)。

J2 复现判据。用**他的口径**重跑,full(2014-01-02→2025-12-31)年化须落在
   他公布值 ±6pp 内。他公布:R06 +9.05%、R08 +17.08%、R09 +10.25%、R11 +19.60%。
   四条各判各的;不过的那条判「无法复现」,其修正值与对照结论一并作废。
   (§113 的教训:复现要对着**他的口径**判,不是对着正确口径判。)

J3 价格口径的量化(描述项,不设阈值)。真实口径 vs 他的口径的年化差额,
   即该路线受前复权价污染的幅度。R09 预期为 0(不含价格)。

J4 市值中性对照。对**修正版**,用同市值名次 ±25 邻域匹配随机 20 只,**200 组种子**
   (p 下限 1/201 = 0.00498)。**Bonferroni:4 条路线,α = 0.05/4 = 0.0125。**
   J4 通过 ⟺ full 与 oos 两个窗口的 p 都 < 0.0125。

事前预测(写下来以便被证伪;错了必须在正文里明说我错了)
--------------------------------------------------------
S1 R06 与 R09 的 J2 复现会过,且**误差小于 R08/R11**,因为这两条不含价格,
   不受复权口径影响。
S2 R08 与 R11 用他的口径能复现(J2 过),换成真实价格后年化**明显下降**
   (下降幅度 > 3pp)。
S3 R09 `core_quality_composite` 会通过 J4。
S4 **R08 修正后不会通过 J4** —— 价值因子在同市值邻域内没有增量。
S5 四条路线里通过 J4 的数量 ∈ [1, 3]。
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
from factor_sweep_pv import draw_fast  # noqa: E402
from fundamental_yoy import label_periods, yoy_series  # noqa: E402

NSEED, ALPHA = 200, 0.05 / 4
WINS = {"full": ("2014-01-02", "2025-12-31"), "oos": ("2023-01-03", "2025-12-31")}
CODEX_CAGR = {"R06": 0.0905, "R08": 0.1708, "R09": 0.1025, "R11": 0.1960}
FLOW = ["eps", "revenue", "net_income", "operating_cash_flow"]


def wrank(v, elig=None, invert=False):
    """照抄 evaluate_quality_factors.winsor_rank。"""
    s = pd.Series(v, dtype=float)
    if elig is not None:
        s = s.where(elig)
    val = s.dropna()
    if val.empty:
        return s.to_numpy()
    lo, hi = val.quantile([0.01, 0.99])
    s = s.clip(lower=lo, upper=hi)
    if invert:
        s = -s
    return s.rank(pct=True, ascending=True).to_numpy()


def build_fund(codes, idx):
    """逐股拼 TTM 与同比,返回 (nt,ns) 矩阵字典。报告期标签复用 label_periods。"""
    nt, ns = len(idx), len(codes)
    keys = ["eps_ttm", "rev_ttm", "ni_ttm", "ocfps_ttm", "bps", "roe_lvl", "roe_chg"]
    fm = {k: np.full((nt, ns), np.nan, np.float32) for k in keys}
    anchor_bad, anchor_n = 0, 0
    t0 = time.time()
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet",
                            columns=[*FLOW, "book_value_per_share", "roe"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        ni = pd.to_numeric(x["net_income"], errors="coerce").ffill()
        ch = ni[ni.diff().fillna(0) != 0].index
        ch = ch[np.isfinite(ni[ch].to_numpy(float))]
        if len(ch) < 8:
            continue
        lab = label_periods(ch)
        cum = {k: {} for k in [*FLOW, "roe"]}
        rows, dates = [], []
        for t, (ry, rp) in zip(ch, lab, strict=True):
            if ry is None:
                continue
            r = {}
            for c2, key in zip([*FLOW, "roe"], [*FLOW, "roe"], strict=True):
                v = float(pd.to_numeric(x[c2], errors="coerce").ffill().get(t, np.nan))
                cum[key][(ry, rp)] = v
            for src, dst in (("eps", "eps_ttm"), ("revenue", "rev_ttm"),
                             ("net_income", "ni_ttm"), ("operating_cash_flow", "ocfps_ttm")):
                v = cum[src][(ry, rp)]
                fy, same = cum[src].get((ry - 1, 4)), cum[src].get((ry - 1, rp))
                r[dst] = (v if rp == 4 else
                          (v + fy - same if fy is not None and same is not None else np.nan))
                if dst == "ni_ttm" and rp == 4 and np.isfinite(v):
                    anchor_n += 1
                    anchor_bad += int(abs(r[dst] - v) > 1e-6 * max(abs(v), 1.0))
            r["roe_lvl"] = cum["roe"][(ry, rp)]
            prev = cum["roe"].get((ry - 1, rp))
            r["roe_chg"] = (r["roe_lvl"] - prev) if prev is not None else np.nan
            rows.append(r)
            dates.append(t)
        if not rows:
            continue
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates)).reindex(idx).ffill()
        for k in keys[:-3]:
            fm[k][:, j] = df[k].to_numpy(np.float32)
        fm["roe_lvl"][:, j] = df["roe_lvl"].to_numpy(np.float32)
        fm["roe_chg"][:, j] = df["roe_chg"].to_numpy(np.float32)
        # 【修正】原写法是 .ffill().reindex(idx) —— 先补后重排,若 idx 比文件索引长
        # (扩展面板),多出来的日期一律是 NaN,整条 R08 会静默变成全缺失。
        # 同函数其他字段走的是 line 158 的 .reindex(idx).ffill()(先重排后补)。
        # 索引相同时两者恒等,所以本改动对第一一七节是 no-op。
        fm["bps"][:, j] = pd.to_numeric(x["book_value_per_share"], errors="coerce"
                                        ).reindex(idx).ffill().to_numpy(np.float32)
        if (j + 1) % 1500 == 0:
            print(f"  财务 {j+1}/{ns} ({time.time()-t0:.0f}s)", flush=True)
    print(f"财务面板完成 ({time.time()-t0:.0f}s);TTM 恒等式:{anchor_n} 个年报点,"
          f"违例 {anchor_bad} 个 {'✓' if anchor_bad == 0 else '✗'}")
    return fm, anchor_bad


def route_scores(name, t, e, fm, cl, raw, logcap, tmean, price_mode):
    """四条路线在调仓日 t、合格集 e 上的分数(越大越优先)。price_mode: raw|qfq"""
    px = (raw if price_mode == "raw" else cl)[t, e].astype(np.float64)
    if name == "R06":
        w = np.log(np.maximum(cl[max(0, t - 249):t + 1, e].astype(np.float64), 1e-12))
        lr = np.diff(w, axis=0)
        cols = []
        for n in (20, 60, 120):
            cols.append(-np.std(lr[-n:], axis=0, ddof=1))
        for n in (60, 120, 250):
            ww = w[-n:]
            pk = np.maximum.accumulate(ww, axis=0)
            cols.append(np.min(ww - pk, axis=0))
        r = [pd.Series(c).rank(pct=True, ascending=True).to_numpy() for c in cols]
        return np.mean(r, axis=0)
    if name in ("R08", "R11_value"):
        ep, bp = fm["eps_ttm"][t, e] / px, fm["bps"][t, e] / px
        cfp = fm["ocfps_ttm"][t, e] / px
        r = []
        for v in (ep, bp, cfp):
            s = pd.Series(np.where(v > 0, v, np.nan))
            r.append(s.rank(pct=True, ascending=True).to_numpy())
        return np.mean(r, axis=0)
    if name in ("R09", "R11_qual"):
        ep = fm["eps_ttm"][t, e] / px
        cfp = fm["ocfps_ttm"][t, e] / px
        marg = fm["ni_ttm"][t, e] / np.where(fm["rev_ttm"][t, e] != 0,
                                            fm["rev_ttm"][t, e], np.nan)
        roe, roec = fm["roe_lvl"][t, e], fm["roe_chg"][t, e]
        conv = fm["ocfps_ttm"][t, e] / np.where(fm["eps_ttm"][t, e] != 0,
                                               fm["eps_ttm"][t, e], np.nan)
        prof, cash = ep > 0, cfp > 0
        r = [wrank(marg, marg > 0), wrank(roe, roe > 0), wrank(roec, prof),
             wrank(conv, prof & cash & (conv > 0))]
        return np.nanmean(np.where(np.isnan(r), np.nan, r), axis=0) * np.where(
            np.any(np.isnan(r), axis=0), np.nan, 1.0)
    raise ValueError(name)


def main():
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点J1a"
    print(f"锚点J1a ✓ 面板 {nt}×{ns}")

    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    truth = {(2017, "中报"): .5307, (2017, "三季报"): 1.0103,
             (2017, "年报"): 1.1401, (2018, "一季报"): 1.2107}
    bad = [k for k, v in truth.items() if abs(float(y.get(k, np.nan)) - v) > 0.005]
    print(f"锚点J1d 泰格同比复现:违例 {len(bad)} 项 {'✓' if not bad else '✗ ' + str(bad)}")
    assert not bad, "锚点J1d 不过"

    raw = np.full((nt, ns), np.nan, np.float32)
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    print("不复权价矩阵完成")

    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点J1c TTM 恒等式不过"

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)

    def build(name, mode):
        sel, elig, srk = {}, {}, {}
        for t in reb:
            t = int(t)
            base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
            if name == "R11":
                thr = np.nanpercentile(logcap[t][base], 10)
                base = base & (logcap[t] > thr)
            e = np.flatnonzero(base)
            if len(e) < TOP_N * 3:
                continue
            if name == "R11":
                v = np.nanmean(np.vstack([
                    pd.Series(route_scores("R11_value", t, e, fm, cl, raw, logcap,
                                           tmean, mode)).rank(pct=True).to_numpy(),
                    pd.Series(route_scores("R11_qual", t, e, fm, cl, raw, logcap,
                                           tmean, mode)).rank(pct=True).to_numpy(),
                    pd.Series(route_scores("R06", t, e, fm, cl, raw, logcap,
                                           tmean, mode)).rank(pct=True).to_numpy(),
                    pd.Series((pd.Series(-logcap[t, e]).rank(pct=True)
                               + pd.Series(-tmean[t, e]).rank(pct=True)) / 2
                              ).rank(pct=True).to_numpy()]), axis=0)
            else:
                v = route_scores(name, t, e, fm, cl, raw, logcap, tmean, mode)
            g = np.isfinite(v)
            if g.sum() < TOP_N:
                continue
            e2, v2 = e[g], v[g]
            top = e2[np.argsort(-v2, kind="stable")[:TOP_N]]
            sel[t] = (top, np.full(TOP_N, WEIGHT))
            order = e[np.argsort(logcap[t, e], kind="stable")]
            rk = {int(c): i for i, c in enumerate(order)}
            elig[t] = order
            srk[t] = np.array([rk[int(c)] for c in top])
        return sel, elig, srk

    def run(sel, w):
        d0, d1 = WINS[w]
        w0 = int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0])
        w1 = int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0])
        eq, dd, tr, fz = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        return metrics(eq, dd, idx), (w0, w1)

    rows, viol = [], 0
    for name in ("R06", "R08", "R09", "R11"):
        r = {"route": name, "codex_cagr": CODEX_CAGR[name]}
        for mode in ("qfq", "raw"):
            sel, elig, srk = build(name, mode)
            for w in WINS:
                m, _ = run(sel, w)
                r[f"{mode}_{w}_cagr"] = m["cagr"]
                r[f"{mode}_{w}_total"] = m["total"]
                r[f"{mode}_{w}_mdd"] = m["mdd"]
            if mode == "raw":
                for w in WINS:
                    d0, d1 = WINS[w]
                    w0 = int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0])
                    w1 = int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0])
                    cg = []
                    for sd in range(NSEED):
                        rng = np.random.default_rng(SEED + sd)
                        cs = {}
                        for t in sel:
                            o, rk = elig[t], srk[t]
                            ps = draw_fast(rng, rk, len(o))
                            viol += int(np.sum(np.abs(ps - rk) > NBR))
                            cs[t] = (o[ps], np.full(TOP_N, WEIGHT))
                        e2, d2, _, _ = run_window_fast(op, cl, susp, lu, ld, cs,
                                                       cal_pos, w0, w1)
                        cg.append(metrics(e2, d2, idx)["cagr"])
                    a = np.array(cg)
                    r[f"ctrl_{w}_med"] = float(np.median(a))
                    r[f"p_{w}"] = (1 + int(np.sum(a >= r[f"raw_{w}_cagr"]))) / (NSEED + 1)
        r["J2"] = abs(r["qfq_full_cagr"] - r["codex_cagr"]) <= 0.06
        r["J3_pp"] = (r["raw_full_cagr"] - r["qfq_full_cagr"]) * 100
        r["J4"] = bool(r["p_full"] < ALPHA and r["p_oos"] < ALPHA)
        rows.append(r)
        print(f"{name}  他{r['codex_cagr']:+6.2%} | 他口径{r['qfq_full_cagr']:+7.2%}"
              f"(J2 {'✓' if r['J2'] else '✗'}) | 真实{r['raw_full_cagr']:+7.2%}"
              f"(差{r['J3_pp']:+5.2f}pp) | oos{r['raw_oos_cagr']:+7.2%} | "
              f"p_full={r['p_full']:.4f} p_oos={r['p_oos']:.4f} "
              f"J4 {'✓' if r['J4'] else '✗'}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n锚点J1b 抽样越界 {viol} 次 {'✓' if viol == 0 else '✗ 作废'}")
    assert viol == 0
    print(f"J2 复现通过 {int(df['J2'].sum())}/4;J4 对照通过 {int(df['J4'].sum())}/4"
          f"(α={ALPHA}):{', '.join(df.loc[df['J4'], 'route']) or '无'}")
    df.to_csv(f"{OUT}/codex_routes_rerun.csv", index=False)
    print(f"落库 {OUT}/codex_routes_rerun.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §117 结果:四条路线**全部复现**,但只有两条扛住市值中性对照。
#
# 锚点 J1a ✓ 面板 3297×5217   J1b ✓ 抽样越界 0 次
#      J1c ✓ TTM 恒等式 46,274 个年报点,违例 0 个
#      J1d ✓ 泰格 300347 同比复现雪球真值,违例 0 项
#
# 路线  他公布   他的口径(J2)      真实口径(J3)        oos      p_full   p_oos    J4
# R06   +9.05%   +9.14% ✓(+0.09)  +9.14%(+0.00pp)  +11.64%  0.0299   0.0199   ✗
# R08  +17.08%  +16.38% ✓(-0.70)  +14.46%(-1.92pp) +17.26%  0.0050   0.0050   ✓
# R09  +10.25%   +8.50% ✓(-1.75)   +8.50%(+0.00pp) +12.45%  0.0050   0.0100   ✓
# R11  +19.60%  +22.76% ✓(+3.16)  +21.86%(-0.90pp) +10.89%  0.0050   0.3333   ✗
#
# J2 复现 **4/4 通过**;J4 对照 2/4 通过(Bonferroni α=0.0125):**R08、R09**。
#
# ── 四条发现 ──
# ① **R10 是特例,不是通例。** R06/R08/R09/R11 都在 ±6pp 内复现了。
#    前复权价污染对 R10 是致命的(年化 −6.56pp),因为**市值本身就是那个因子**;
#    对 R08/R11 只动了估值比率的分母,幅度 −1.92pp / −0.90pp,小一个量级;
#    对 R06/R09 是 **+0.00pp** —— 这两条四项因子都不含价格,完全干净。
#    (S1 的前半说对了,后半说错了,见下。)
# ② **价值与质量是微盘-低流动性之外的另外两族。** §115-B/§116-A 通过的两个因子
#    (small_cap_low_turnover、high_amihud)全挤在微盘层;R08 value_composite 与
#    R09 core_quality_composite 通过了同一套市值中性对照,且不在那一族。
#    **这是本项目第一次拿到两族互相独立的幸存者。**
# ③ **R11 多因子在样本外崩了。** 他排名第 3 的路线,full p=0.0050 但 oos p=0.3333,
#    合成之后**反而不如它的组件**(R08 oos p=0.0050、R09 oos p=0.0100)。
#    合成用的是「各类先转当日分位再等权」,四类里有两类(低风险、小市值低换手)
#    在 §115-B/§116-A 已被证明弱或不稳,把它们塞进复合分是在稀释。
# ④ **R06 低波动过不了 Bonferroni**(p=0.0299/0.0199,低于 0.05 但高于 0.0125),
#    与 §115-B 里 low_vol_60/low_vol_120/low_mdd_250 三个全部落选**独立一致**。
#
# ── 事前预测:5 个里错了 3 个 ──
# S1 **部分错。** 我说「R06 与 R09 的复现误差小于 R08/R11」。R06 误差 0.09pp
#    确实最小,但 **R09 误差 1.75pp 大于 R08 的 0.70pp** —— 我把「不含价格」
#    等同于「误差更小」,而 R09 的误差来自 winsor_rank 的 eligible 掩码与
#    TTM 拼法的细节,与价格无关。
# S2 **后半错。** J2 过了 ✓,但我预测真实价格会让 R08/R11 年化「下降 > 3pp」,
#    实际只有 −1.92pp 与 −0.90pp。我按 R10 的 −6.56pp 外推,忘了 R10 的市值是
#    **因子本身**、而 R08 的价格只是**比率的分母**,后者被横截面 rank 大幅吸收。
# S3 命中。R09 通过 J4。
# S4 **错。** 我预测「R08 修正后不会通过 J4」,理由是「价值因子在同市值邻域内
#    没有增量」。实际 R08 修正后 full 与 oos 的 p 都是 0.0050(200 种子下限),
#    **没有任何一组同市值随机对照跑赢它**。这是第七次把先验当结论。
# S5 命中。通过 J4 的数量 2 ∈ [1,3]。
#
# ── 未解决 ──
# 本节两个窗口(full 2014–2025、oos 2023–2025)都在 Codex 的搜索范围内,
# **R08/R09 还没有过时间样本外**;§116-A 对第一批做的 2026 干净留出期
# 尚未对 R08/R09 做。这是下一步该补的第一件事。
# =============================================================================
