"""§123 深入 R08/R09 第一步:换成文献标准因子,排除挑选偏差 + 两个稳健性检验。

为什么必须换定义
----------------
§119 里 R08 价值 / R09 质量通过了正规时间样本外(训练 2014-2021 / 留出 2022-2026,
55 个调仓日,留出期对照中位为负而策略为正,p=0.0020 / 0.0060)。
**但那个留出期对我干净、对 Codex 不干净** —— 他把这两条定为 WATCHLIST 时
用的是 2014–2025,包含我整个留出期,而且是从 130 个策略变体里挑出来的。
**因子的挑选偏差没有被 §119 校正。**

排除办法:换成**学术文献里 2022 年前就已公开写死的定义**,
谁都没在这份 A 股数据上挑过它们。若这些定义在同一套检验下仍然成立,
挑选偏差这个洞才算堵上。

因子清单(6 个;逐条注明出处与是否精确实现)
--------------------------------------------
V1 B/M   账面市值比      Fama & French (1993)        ✓ 精确 = BPS / raw_close
V2 E/P   盈利收益率      Basu (1977)                 ✓ 精确 = EPS_ttm / raw_close
V3 CF/P  现金流价格比    Lakonishok-Shleifer-Vishny (1994)
                         ✓ 用**经营现金流**每股值 / raw_close。
                         原文的 cash flow = 净利 + 折旧;经营现金流口径更严格,标注。
V4 S/P   销售价格比      Barbee-Mukherji-Raines (1996) ✓ 精确 = 营收_ttm / 真实市值
Q1 ROE   盈利能力        Haugen & Baker (1996)       ✓ 精确 = 面板 roe 列(百分数)
Q2 应计                  Sloan (1996)                ⚠ **变体**:原文按平均总资产
                         标准化,本面板无总资产,只能用每股口径
                         (EPS_ttm − OCFPS_ttm) / |EPS_ttm|,低者优。**不是原定义。**

**跑不了的**:Novy-Marx (2013) 毛利率 GP/A —— 需要 COGS 与总资产,
本面板两者都没有。**如实记录,不用近似替代去凑。**

价格一律用**真实不复权价** raw_close(§113 的教训);TTM 复用
`fundamental_yoy.label_periods`(§97–§101 的教训);
引擎与 §114/§119 完全一致,含 cash_fallback(§121–§122 的教训)。
训练 2014-01-02→2021-12-31 / 留出 2022-01-04→面板末,**留出期只看一次**。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
N1 锚点(不过则整节作废)
   (a) 面板 (3297, 5217);
   (b) 抽样恒等式:对照每次抽样的市值名次偏离 ≤25,违例 > 0 即作废;
   (c) TTM 恒等式:年报公告日当天 TTM 净利 = 当期累计,相对误差 < 1e-6;
   (d) 泰格 300347 同比复现雪球真值 0.5307/1.0103/1.1401/1.2107(±0.5pp);
   (e) **成本引擎恒等式**:本节新写的带成本倍率的引擎,在倍率 = 1.0 时
       必须与 §114 已验证的 `run_window_fast` **逐日相等**(最大相对误差 < 1e-9)。

N2 文献单因子(**核心判据**)。每个因子单独跑,不做任何复合、不调阈值。
   对照 = 同市值名次 ±25 邻域匹配随机 20 只,**500 组种子**(p 下限 1/501=0.001996)。
   **Bonferroni:6 个因子,α = 0.05/6 = 0.008333。**
   N2 通过 ⟺ 训练期与留出期的 p **都** < 0.008333。

N3 退市清算稳健性。已知偏差:引擎把停牌/退市持仓**冻结在最后有效价**,
   而现实中退市整理期通常接近全损(R08 全期冻结 31 次、R09 68 次)。
   本项把已退市股票(最后有效收盘早于面板末 60 个交易日)在其最后有效日之后的
   价格**一律乘 0.10**,即按 −90% 计。
   N3 通过 ⟺ 通过 N2 的因子(以及 R08、R09)在此口径下,
   **留出期 p 仍 < 0.05**(确认性检验,不再做 Bonferroni)。

N4 双倍成本压力。佣金 / 印花税 / 滑点全部 ×2(Codex 自己的 R13 门槛里就有
   「双倍成本下超额仍 >0」这一条)。
   N4 通过 ⟺ 同上范围的因子在双倍成本下,**留出期 p 仍 < 0.05**。

事前预测
--------
**本节不下预测。**
§119 的事前登记里我写过:「U2 是我第九次做『样本内强→样本外』类外推,前八次全错。
如果 U2 又错,说明我在这类外推上没有可用的先验,今后不应再基于样本内强度
预测样本外结果。」U2 又错了,而且错在悲观方向 —— 两个方向都错。
**自 §119 起已停止此类预测,本节兑现该承诺,只登记判据。**

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推;
**不做因子复合**(§117 已测出 R11 朴素合成后 oos p=0.3333,反而不如组件);
**不因留出期结果不好回头改因子定义或窗口**;
不用近似定义替代跑不了的 Novy-Marx;
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
from codex_r10_replication import DATA, INITIAL_CASH, TOP_N, WEIGHT, metrics  # noqa: E402
from codex_routes_rerun import build_fund, route_scores, wrank  # noqa: E402
from factor_sweep_pv import draw_fast  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

NSEED, ALPHA = 500, 0.05 / 6
WINS = {"train": ("2014-01-02", "2021-12-31"), "holdout": ("2022-01-04", "2026-08-03")}
LIT = ("BM_FF1993", "EP_Basu1977", "CFP_LSV1994", "SP_BMR1996",
       "ROE_HB1996", "ACCRUAL_Sloan1996v")


def run_cost(op, cl, susp, lu, ld, sel, cal_pos, w0, w1, mult=1.0):  # noqa: PLR0913
    """§114 run_window_fast 的等价实现,外加成本倍率。mult=1.0 时必须逐日相等。"""
    buy, sell, tax, slip, minf = (3e-4 * mult, 3e-4 * mult, 1e-3 * mult,
                                  1e-3 * mult, 5.0 * mult)
    days = cal_pos[(cal_pos >= w0) & (cal_pos <= w1)]
    n = len(days)
    is_sel = np.array([int(t) in sel for t in days])
    ex = np.flatnonzero(is_sel[:-1]) + 1 if n > 1 else np.zeros(0, np.int64)
    eq = np.empty(n)
    pos_i, pos_s = np.zeros(0, np.int64), np.zeros(0, np.float64)
    cash = INITIAL_CASH
    bounds = list(ex) + [n]
    if len(ex) == 0 or ex[0] > 0:
        eq[: (ex[0] if len(ex) else n)] = cash
    for b, k in enumerate(ex):
        cols, wts = sel[int(days[k - 1])]
        t = days[k]
        if len(pos_i):
            po = op[t, pos_i].astype(np.float64)
            bad = ~np.isfinite(po) | (po <= 0)
            po[bad] = cl[t, pos_i][bad].astype(np.float64)
            po[~np.isfinite(po)] = 0.0
            open_val = cash + float(np.sum(pos_s * po))
        else:
            open_val = cash
        px = op[t, cols].astype(np.float64)
        tgt = {int(c): float(int(open_val * w // (p * (1 + slip) * 100)) * 100)
               for c, w, p in zip(cols, wts, px, strict=True)
               if np.isfinite(p) and p > 0}
        keep = {}
        for c_, s_ in zip(pos_i, pos_s, strict=True):
            c_ = int(c_)
            q = max(0.0, s_ - tgt.get(c_, 0.0))
            p_ = op[t, c_]
            if q <= 0 or susp[t, c_] or ld[t, c_] or not np.isfinite(p_) or p_ <= 0:
                keep[c_] = s_
                continue
            amt = q * float(p_) * (1 - slip)
            cash += amt - max(minf, amt * sell) - amt * tax
            if s_ - q > 0:
                keep[c_] = s_ - q
        for c_ in cols:
            c_ = int(c_)
            q = max(0.0, tgt.get(c_, 0.0) - keep.get(c_, 0.0))
            p_ = op[t, c_]
            if q <= 0 or susp[t, c_] or lu[t, c_] or not np.isfinite(p_) or p_ <= 0:
                continue
            fill = float(p_) * (1 + slip)
            q = min(q, float(int(max(0.0, cash - minf) // (fill * 100)) * 100))
            while q > 0 and q * fill + max(minf, q * fill * buy) > cash + 1e-8:
                q -= 100
            if q <= 0:
                continue
            amt = q * fill
            cash -= amt + max(minf, amt * buy)
            keep[c_] = keep.get(c_, 0.0) + q
        pos_i = np.array(sorted(keep), np.int64)
        pos_s = np.array([keep[i] for i in pos_i], np.float64)
        a, zz = k, bounds[b + 1]
        if len(pos_i):
            seg = np.nan_to_num(cl[np.ix_(days[a:zz], pos_i)].astype(np.float64))
            eq[a:zz] = cash + seg @ pos_s
        else:
            eq[a:zz] = cash
    return eq, days


def lit_score(name, t, e, fm, raw, logcap):
    """六个文献因子。分数一律'越大越优先',不合格返回 NaN。"""
    px = raw[t, e].astype(np.float64)
    if name == "BM_FF1993":
        v = fm["bps"][t, e] / px
    elif name == "EP_Basu1977":
        v = fm["eps_ttm"][t, e] / px
    elif name == "CFP_LSV1994":
        v = fm["ocfps_ttm"][t, e] / px
    elif name == "SP_BMR1996":
        v = fm["rev_ttm"][t, e] / np.exp(logcap[t, e].astype(np.float64))
    elif name == "ROE_HB1996":
        v = fm["roe_lvl"][t, e]
        return wrank(v, v > 0)
    elif name == "ACCRUAL_Sloan1996v":
        ep = fm["eps_ttm"][t, e]
        acc = (ep - fm["ocfps_ttm"][t, e]) / np.where(ep != 0, np.abs(ep), np.nan)
        return wrank(acc, ep > 0, invert=True)
    else:
        raise ValueError(name)
    return wrank(v, v > 0)


def main():  # noqa: PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点N1a"
    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    assert abs(float(y.get((2017, "中报"), np.nan)) - 0.5307) < 0.005, "锚点N1d"
    print(f"锚点N1a ✓ {nt}×{ns};N1d 泰格同比 ✓", flush=True)

    raw = np.full((nt, ns), np.nan, np.float32)
    lastv = np.zeros(ns, np.int64)
    t0 = time.time()
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        s = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda v: v > 0).reindex(idx)
        g = np.flatnonzero(np.isfinite(s.to_numpy()))
        lastv[j] = g[-1] if len(g) else -1
        raw[:, j] = s.ffill().to_numpy(np.float32)
    print(f"不复权价矩阵完成 ({time.time()-t0:.0f}s)", flush=True)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点N1c TTM 恒等式"
    print("锚点N1c ✓ TTM 恒等式", flush=True)

    # 退市清算口径:最后有效收盘早于面板末 60 个交易日者,其后价格一律 ×0.10
    dead = (lastv >= 0) & (lastv < nt - 60)
    cl_liq = cl.copy()
    for j in np.flatnonzero(dead):
        cl_liq[lastv[j] + 1:, j] *= 0.10
    print(f"退市清算口径:判定已退市 {int(dead.sum())} 只,其后价格 ×0.10", flush=True)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)

    def wpos(w):
        d0, d1 = WINS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    def build(fn):
        sel, elig, srk = {}, {}, {}
        for t in reb:
            t = int(t)
            base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
            e = np.flatnonzero(base)
            if len(e) < TOP_N:
                continue
            v = np.asarray(fn(t, e), dtype=float)
            g = np.isfinite(v)
            if not g.any():
                continue
            e2 = e[g]
            k = min(TOP_N, len(e2))
            top = e2[np.argsort(-v[g], kind="stable")[:k]]
            sel[t] = (top, np.full(k, WEIGHT))
            order = e[np.argsort(logcap[t, e], kind="stable")]
            rk = {int(c): i for i, c in enumerate(order)}
            elig[t] = order
            srk[t] = np.array([rk[int(c)] for c in top])
        return sel, elig, srk

    viol = 0

    def evaluate(sel, elig, srk, w, clm, mult, nseed=NSEED):
        nonlocal viol
        w0, w1 = wpos(w)
        eq, days = run_cost(op, clm, susp, lu, ld, sel, cal_pos, w0, w1, mult)
        m = metrics(eq, days, idx)
        cg = []
        for sd in range(nseed):
            rng = np.random.default_rng(SEED + sd)
            cs = {}
            for t in sel:
                o, rk = elig[t], srk[t]
                ps = draw_fast(rng, rk, len(o))
                viol += int(np.sum(np.abs(ps - rk) > NBR))
                cs[t] = (o[ps], np.full(len(ps), WEIGHT))
            e2, d2 = run_cost(op, clm, susp, lu, ld, cs, cal_pos, w0, w1, mult)
            cg.append(metrics(e2, d2, idx)["cagr"])
        a = np.array(cg)
        return m, float(np.median(a)), (1 + int(np.sum(a >= m["cagr"]))) / (nseed + 1)

    # ---- 锚点 N1e:成本引擎恒等式(mult=1.0 时与 §114 引擎逐日相等)----
    probe, _, _ = build(lambda t, e: lit_score("BM_FF1993", t, e, fm, raw, logcap))
    okid = True
    for w in WINS:
        w0, w1 = wpos(w)
        e1, _, _, _ = run_window_fast(op, cl, susp, lu, ld, probe, cal_pos, w0, w1)
        e2, _ = run_cost(op, cl, susp, lu, ld, probe, cal_pos, w0, w1, 1.0)
        err = float(np.max(np.abs(e2 - e1) / np.maximum(np.abs(e1), 1e-9)))
        okid &= err < 1e-9
        print(f"锚点N1e 成本引擎恒等式 {w:8s} 最大相对误差 {err:.3e} "
              f"{'✓' if err < 1e-9 else '✗'}", flush=True)
    assert okid, "锚点N1e 不过"

    rows = []
    cand = [(n, (lambda t, e, nn=n: lit_score(nn, t, e, fm, raw, logcap))) for n in LIT]
    cand += [(n, (lambda t, e, nn=n: route_scores(nn, t, e, fm, cl, raw, logcap,
                                                  tmean, "raw"))) for n in ("R08", "R09")]
    keep = {}
    print("\n=== N2 文献单因子(基准口径)===", flush=True)
    for name, fn in cand:
        sel, elig, srk = build(fn)
        r = {"factor": name, "n_reb": len(sel)}
        for w in WINS:
            m, med, p = evaluate(sel, elig, srk, w, cl, 1.0)
            r |= {f"{w}_cagr": m["cagr"], f"{w}_mdd": m["mdd"],
                  f"{w}_sharpe": m["sharpe"], f"{w}_total": m["total"],
                  f"{w}_ctrl": med, f"{w}_p": p}
        r["N2"] = bool(name not in LIT or (r["train_p"] < ALPHA and r["holdout_p"] < ALPHA))
        rows.append(r)
        keep[name] = (sel, elig, srk)
        tag = "N2 " + ("✓" if r["N2"] else "✗") if name in LIT else "(参照)"
        print(f"{name:22s} 训练 年化{r['train_cagr']:+7.2%} p={r['train_p']:.4f} | "
              f"留出 年化{r['holdout_cagr']:+7.2%} 回撤{r['holdout_mdd']:+7.2%} "
              f"夏普{r['holdout_sharpe']:5.2f} 对照中位{r['holdout_ctrl']:+7.2%} "
              f"p={r['holdout_p']:.4f} | {tag}", flush=True)

    sub = [n for n in LIT if next(r for r in rows if r["factor"] == n)["N2"]] + ["R08", "R09"]
    print(f"\n=== N3 退市按 -90% 清算 / N4 双倍成本(仅对 {len(sub)} 条)===", flush=True)
    for name in sub:
        sel, elig, srk = keep[name]
        m3, med3, p3 = evaluate(sel, elig, srk, "holdout", cl_liq, 1.0, 200)
        m4, med4, p4 = evaluate(sel, elig, srk, "holdout", cl, 2.0, 200)
        for r in rows:
            if r["factor"] == name:
                r |= {"liq_cagr": m3["cagr"], "liq_ctrl": med3, "liq_p": p3,
                      "N3": bool(p3 < 0.05), "x2_cagr": m4["cagr"], "x2_ctrl": med4,
                      "x2_p": p4, "N4": bool(p4 < 0.05)}
        print(f"{name:22s} 退市清算 年化{m3['cagr']:+7.2%} 对照{med3:+7.2%} "
              f"p={p3:.4f} N3 {'✓' if p3 < 0.05 else '✗'} | 双倍成本 年化{m4['cagr']:+7.2%} "
              f"对照{med4:+7.2%} p={p4:.4f} N4 {'✓' if p4 < 0.05 else '✗'}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n锚点N1b 抽样越界 {viol} 次 {'✓' if viol == 0 else '✗ 作废'}")
    assert viol == 0
    n2 = df[df["factor"].isin(LIT)]["N2"]
    print(f"N2 文献因子通过 {int(n2.sum())}/6(α={ALPHA:.6f}):"
          f"{', '.join(df.loc[df['N2'] & df['factor'].isin(LIT), 'factor']) or '无'}")
    for k in ("N3", "N4"):
        if k in df.columns:
            print(f"{k} 通过:{', '.join(df.loc[df[k].fillna(False), 'factor']) or '无'}")
    df.to_csv(f"{OUT}/literature_factors.csv", index=False)
    print(f"落库 {OUT}/literature_factors.csv")


if __name__ == "__main__":
    main()
