"""§124 拆开 R08 与 R09:哪一项在贡献,哪一项在拖累?

缘起
----
§123 测出两件事:CFP(LSV 1994)与 SP(BMR 1996)两个三十年前写死的文献定义
在 2022–2026 留出期通过了同市值随机对照(p=.0020 / .0060),
**但 B/M(Fama-French 1993 的正宗价值因子)没过**(留出期 p=0.3792、年化 −1.16%),
**ROE(Haugen-Baker 1996)也没过**(p=0.0978)。

而 `bp` 正是 R08 三个分量之一,`roe_level` 正是 R09 四项之一。
本节把两条复合分**拆开**,看各自贡献 —— 这是一次拆解,**不是新的因子搜索**。

拆法(全部沿用他的原始构造,只改参与的分量)
--------------------------------------------
R08 `value_composite` = ep_ttm_pit / bp_pit / cfp_ttm_pit 三项
    先 `where(>0)` 再 `rank(pct=True)`,取均值 `skipna=False`
    → 3 个单项 + 3 个留一法(去掉 ep / 去掉 bp / 去掉 cfp)
R09 `core_quality_composite` = margin(>0) / roe(>0) / roe_chg(ep>0) /
    cash_conv(ep>0 & cfp>0 & 自身>0) 四项 winsor_rank 均值 skipna=False
    → 4 个单项 + 4 个留一法
共 14 个子集。价格一律真实不复权价;TTM 复用 label_periods;含 cash_fallback。
训练 2014-01-02→2021-12-31 / 留出 2022-01-04→面板末,**留出期只看一次**。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
P1 锚点(不过则整节作废):面板 (3297,5217);抽样越界 0 次;TTM 恒等式;
   泰格同比复现;**留一法恒等式** —— 「三项全留」的 R08 子集必须与 §119/§123
   的 R08 **逐日相等**(最大相对误差 < 1e-9),R09 同理。写错必抓。

P2 子集显著性。对照 = 同市值名次 ±25 邻域匹配随机,**400 组种子**
   (p 下限 1/401 = 0.002494)。**Bonferroni:14 个子集,α = 0.05/14 = 0.003571。**
   P2 通过 ⟺ 训练期与留出期的 p **都** < 0.003571。

P3 「去掉某项更好」的判定。**核心判据。**
   对某分量 X,「去掉 X 更好」成立 ⟺ 同时满足:
     (a) 留一子集(去掉 X)的**留出期年化严格高于**完整复合分;
     (b) 该留一子集通过 P2;
     (c) 完整复合分与该留一子集的留出期年化之差 **≥ 1.0pp**
         (低于 1pp 视为噪声,不下结论)。
   三条都满足才判「X 是拖累」;否则判「无法认定」——**不做「大概是」这种表述**。

事前预测
--------
**本节不下预测**(兑现 §119 登记里的承诺:已停止「样本外会不会过」类外推)。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推;
**不基于本节结果去搜索新的权重组合** —— 那就变成了在留出期上调参,
本节只做拆解,任何权重优化必须另开一节、换新的留出期;
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
from codex_routes_rerun import build_fund, route_scores, wrank  # noqa: E402
from factor_sweep_pv import draw_fast  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

NSEED, ALPHA, MINGAP = 400, 0.05 / 14, 1.0
WINS = {"train": ("2014-01-02", "2021-12-31"), "holdout": ("2022-01-04", "2026-08-03")}
V = ("ep", "bp", "cfp")
Q = ("margin", "roe", "roe_chg", "cash_conv")


def parts_r08(t, e, fm, raw):
    """R08 三分量:where(>0) 后 rank(pct=True)。与他的 factor_scores 一致。"""
    px = raw[t, e].astype(np.float64)
    out = {}
    for k, v in (("ep", fm["eps_ttm"][t, e] / px), ("bp", fm["bps"][t, e] / px),
                 ("cfp", fm["ocfps_ttm"][t, e] / px)):
        out[k] = pd.Series(np.where(v > 0, v, np.nan)).rank(pct=True).to_numpy()
    return out


def parts_r09(t, e, fm, raw):
    """R09 四分量:winsor_rank + 各自的 eligible 掩码。与他的 factor_scores 一致。"""
    px = raw[t, e].astype(np.float64)
    ep = fm["eps_ttm"][t, e] / px
    cfp = fm["ocfps_ttm"][t, e] / px
    rev = fm["rev_ttm"][t, e]
    marg = fm["ni_ttm"][t, e] / np.where(rev != 0, rev, np.nan)
    roe, roec = fm["roe_lvl"][t, e], fm["roe_chg"][t, e]
    conv = fm["ocfps_ttm"][t, e] / np.where(fm["eps_ttm"][t, e] != 0,
                                            fm["eps_ttm"][t, e], np.nan)
    prof, cash = ep > 0, cfp > 0
    return {"margin": wrank(marg, marg > 0), "roe": wrank(roe, roe > 0),
            "roe_chg": wrank(roec, prof),
            "cash_conv": wrank(conv, prof & cash & (conv > 0))}


def main():  # noqa: PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点P1a"
    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    assert abs(float(y.get((2017, "中报"), np.nan)) - 0.5307) < 0.005, "锚点P1"
    print(f"锚点P1a ✓ {nt}×{ns};泰格同比 ✓", flush=True)

    raw = np.full((nt, ns), np.nan, np.float32)
    t0 = time.time()
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    print(f"价格矩阵完成 ({time.time()-t0:.0f}s)", flush=True)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点P1c TTM"
    print("锚点P1c ✓ TTM 恒等式", flush=True)

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

    def subset_score(fam, keys):
        def fn(t, e):
            p = parts_r08(t, e, fm, raw) if fam == "V" else parts_r09(t, e, fm, raw)
            m = np.vstack([p[k] for k in keys])
            bad = np.any(np.isnan(m), axis=0)       # skipna=False
            return np.where(bad, np.nan, np.nanmean(m, axis=0))
        return fn

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

    def ev(sel, elig, srk, w):
        nonlocal viol
        w0, w1 = wpos(w)
        eq, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        m = metrics(eq, dd, idx)
        cg = []
        for sd in range(NSEED):
            rng = np.random.default_rng(SEED + sd)
            cs = {}
            for t in sel:
                o, rk = elig[t], srk[t]
                ps = draw_fast(rng, rk, len(o))
                viol += int(np.sum(np.abs(ps - rk) > NBR))
                cs[t] = (o[ps], np.full(len(ps), WEIGHT))
            e2, d2, _, _ = run_window_fast(op, cl, susp, lu, ld, cs, cal_pos, w0, w1)
            cg.append(metrics(e2, d2, idx)["cagr"])
        a = np.array(cg)
        return m, float(np.median(a)), (1 + int(np.sum(a >= m["cagr"]))) / (NSEED + 1)

    # 锚点 P1e:全留子集必须与 §119/§123 的 R08/R09 逐日相等
    okid = True
    for fam, keys, nm in (("V", V, "R08"), ("Q", Q, "R09")):
        s1, _, _ = build(subset_score(fam, list(keys)))
        s2, _, _ = build(lambda t, e, n=nm: route_scores(n, t, e, fm, cl, raw,
                                                         logcap, tmean, "raw"))
        w0, w1 = wpos("holdout")
        e1, _, _, _ = run_window_fast(op, cl, susp, lu, ld, s1, cal_pos, w0, w1)
        e2, _, _, _ = run_window_fast(op, cl, susp, lu, ld, s2, cal_pos, w0, w1)
        err = float(np.max(np.abs(e2 - e1) / np.maximum(np.abs(e1), 1e-9)))
        okid &= err < 1e-9
        print(f"锚点P1e 留一法恒等式 {nm} 最大相对误差 {err:.3e} "
              f"{'✓' if err < 1e-9 else '✗'}", flush=True)
    assert okid, "锚点P1e 不过"

    jobs = []
    for fam, keys in (("V", V), ("Q", Q)):
        for k in keys:
            jobs.append((fam, f"{fam}·单-{k}", [k]))
        for k in keys:
            jobs.append((fam, f"{fam}·去掉-{k}", [x for x in keys if x != k]))
        jobs.append((fam, f"{fam}·全留({'R08' if fam == 'V' else 'R09'})", list(keys)))

    rows = []
    print("", flush=True)
    for fam, label, keys in jobs:
        sel, elig, srk = build(subset_score(fam, keys))
        r = {"family": fam, "subset": label, "keys": "+".join(keys), "n_reb": len(sel)}
        for w in WINS:
            m, med, p = ev(sel, elig, srk, w)
            r |= {f"{w}_cagr": m["cagr"], f"{w}_mdd": m["mdd"],
                  f"{w}_sharpe": m["sharpe"], f"{w}_ctrl": med, f"{w}_p": p}
        r["P2"] = bool(r["train_p"] < ALPHA and r["holdout_p"] < ALPHA)
        rows.append(r)
        print(f"{label:22s} 训练{r['train_cagr']:+7.2%} p={r['train_p']:.4f} | "
              f"留出{r['holdout_cagr']:+7.2%} 回撤{r['holdout_mdd']:+7.2%} "
              f"夏普{r['holdout_sharpe']:5.2f} 对照{r['holdout_ctrl']:+7.2%} "
              f"p={r['holdout_p']:.4f} | P2 {'✓' if r['P2'] else '✗'}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n锚点P1b 抽样越界 {viol} 次 {'✓' if viol == 0 else '✗ 作废'}")
    assert viol == 0
    print(f"P2 通过 {int(df['P2'].sum())}/{len(df)}(α={ALPHA:.6f})\n")
    print("=== P3 「去掉某项更好」判定 ===")
    for fam, keys, nm in (("V", V, "R08"), ("Q", Q, "R09")):
        full = df[df.subset == f"{fam}·全留({nm})"].iloc[0]
        for k in keys:
            sub = df[df.subset == f"{fam}·去掉-{k}"].iloc[0]
            gap = (sub["holdout_cagr"] - full["holdout_cagr"]) * 100
            a, bb, c = gap > 0, bool(sub["P2"]), gap >= MINGAP
            v = a and bb and c
            df.loc[df.subset == f"{fam}·去掉-{k}", "P3_gap_pp"] = gap
            df.loc[df.subset == f"{fam}·去掉-{k}", "P3"] = v
            print(f"  {nm} 去掉 {k:9s} 留出年化 {sub['holdout_cagr']:+7.2%} vs 全留 "
                  f"{full['holdout_cagr']:+7.2%} 差 {gap:+6.2f}pp | "
                  f"(a)更高 {'✓' if a else '✗'} (b)过P2 {'✓' if bb else '✗'} "
                  f"(c)≥{MINGAP}pp {'✓' if c else '✗'} → "
                  f"{'**判定 ' + k + ' 是拖累**' if v else '无法认定'}")
    df.to_csv(f"{OUT}/component_decompose.csv", index=False)
    print(f"\n落库 {OUT}/component_decompose.csv")


if __name__ == "__main__":
    main()
