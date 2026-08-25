"""§114 修正版 R10:真实市值基线 + 市值中性对照。

背景
----
§113 复现失败并定位到原因:Codex 的 R10 把**前复权价**乘历史股本当市值
(宁德时代 2021-11-30:他的 close=350.12 × 20.33 亿股 = 7116.74 亿,
而当日真实不复权收盘 679.68,真实流通市值 13815.5 亿,低估 48.5%)。
横截面市值排序因此被系统性扭曲。§113 已照判作废,不改判。

本节以**真实市值**(raw_close × PIT outstanding_share,即本面板 float_mv,
宁德 2021-11-30 = 13815.5 亿,已核对)重建 R10,再补 Codex 完全没有做的
市值中性随机对照,回答唯一重要的问题:
**修正之后,「小市值 + 低换手」还剩下选股能力吗,还是纯粹的小盘风格 beta?**

已知事实(§113 诊断得到,**非事前登记,证据等级低**,此处只作陈述不作判据)
--------------------------------------------------------------------------
真实市值基线 full(2014-01-02→2025-12-31)= +1444.01%,年化 25.63%,
回撤 -56.38%,夏普 1.12;oos(2023-2025)= +152.11%,年化 36.21%。
用他的错误口径(前复权市值)重跑 = +2743.16%,年化 32.19%,
与他公布的 +3135.06%/33.62% 差 1.43pp/年(调仓网格、整手、tie-break)。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
D1 引擎自洽锚点(恒等式)。本节为提速新写了分段向量化引擎
   `run_window_fast`。它与 §113 已验证的逐日引擎 `run_window`
   在**同一组选股、同一窗口**上必须给出**逐日相等**的净值曲线
   (最大相对误差 < 1e-9)。
   → 这是恒等式:写对必过,写错必抓。不过则本节全部作废。

D2 低换手的市值中性增量。**核心判据。**
   对照A(邻域匹配):每个调仓日,把当日合格股票按真实市值升序排成一列;
   策略选中的 20 只各自落在名次 r_i;对照从名次 [r_i-25, r_i+25] 区间内
   随机抽一只替换(已抽中的排除),凑满 20 只。同引擎、同成本、同调仓日。
   100 组种子。→ 市值暴露与策略逐日对齐,被打掉的只有"低换手"这一维。
   D2 通过 ⟺ 策略年化 **严格高于** 100 组对照年化的 95 分位数
            (单尾 p < 0.05),且 full 与 oos **两个窗口都成立**。
   任一窗口不成立 → D2 不通过 → 低换手在同市值邻域内没有可检出的增量。

D3 是否只是"买最小一档"。
   对照B:每个调仓日,把合格股票按真实市值升序分 10 档,从最小档(D1档)
   内随机抽 20 只。同引擎、同成本。100 组种子。
   D3 通过 ⟺ 策略年化 **严格高于** 100 组对照年化的 95 分位数,
            且 full 与 oos 两个窗口都成立。
   不通过 → "小市值+低换手"并不比"在最小十分位里闭眼抽 20 只"更好,
            即 R10 的收益是小盘风格 beta。

D4 描述项(不设通过阈值):
   (a) 十档等权风格曲线(不含成本、不含整手)的市值阶梯是否单调;
   (b) 卖出被冻结次数(停牌/退市导致挂单卖不掉)及其占比;
   (c) 剔除最小 5% / 10% 市值后策略年化的变化(容量与可交易性);
   (d) 含分红 / 不含分红两种自洽基准口径下的超额收益对照。

锚点
----
A1 面板 (3297, 5217);universe 与 Codex strategy_spec.yaml 集合相等。
A2 恒等式:对照A 每一次抽样,|名次(对照_i) - 名次(策略_i)| ≤ 25 必须
   对**所有** i、**所有**调仓日、**所有**种子成立;违例数 > 0 即作废。
   对照B:所有被抽中者的市值名次必须 < ceil(合格数/10);违例数 > 0 即作废。
A3 真实市值口径核对:宁德时代 300750 在 2021-11-30 的
   float_mv 必须 ∈ [13800亿, 13830亿](真实 13815.5 亿)。
   若面板市值被复权污染,此项必炸。

事前预测(写下来以便被证伪;错了必须在正文里明说我错了)
--------------------------------------------------------
P1 D1 会过(引擎重写只是提速,不改逻辑)。
P2 **D2 不会过。** 理由:Codex 自己的 2023–2025 Rank IC 表里 small_cap
   单独 250 日 IC = 0.1259、small_cap_low_turnover = 0.1363,增量仅 0.0104;
   邻域匹配已吃掉全部小盘 beta。预测策略相对对照中位数年化超额 < 5pp
   且 p > 0.05。
P3 **D3 不会过**,而且**对照B 的中位数会高于策略年化**。理由:A 股
   2014–2025 微盘等权是极强的风格,策略把市值极端化到最小 20 只反而
   踩进流动性最差、退市率最高的一层。
P4 D4(b):卖出冻结次数在 full 窗口 > 50 次。理由:策略持有的是全市场
   最小的 20 只,停牌与退市集中在这一层。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;
**不往 quant-research-dev 推任何东西(只读)**;
不对 R10 之外的路线做复现;不基于本节结论做任何可交易性声明。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_replication import (  # noqa: E402
    DATA,
    INITIAL_CASH,
    TOP_N,
    WEIGHT,
    WINDOWS,
    metrics,
    pct,
    run_window,
)

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
OUT = os.environ.get("OXQ_OUT_DIR", SP)
CACHE = f"{SP}/codex_r10_matrices.npz"
NSEED, SEED, NBR = 100, 20260824, 25


def run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1):  # noqa: PLR0913
    """分段向量化:调仓之间持仓不变,净值一次矩阵乘算完。与 run_window 等价。"""
    days = cal_pos[(cal_pos >= w0) & (cal_pos <= w1)]
    n = len(days)
    is_sel = np.array([int(t) in sel for t in days])
    ex = np.flatnonzero(is_sel[:-1]) + 1 if n > 1 else np.zeros(0, np.int64)
    eq = np.empty(n)
    pos_i = np.zeros(0, np.int64)
    pos_s = np.zeros(0, np.float64)
    cash = INITIAL_CASH
    trades = frozen = 0
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
        tgt = {int(c): float(int(open_val * w // (p * (1 + 0.001) * 100)) * 100)
               for c, w, p in zip(cols, wts, px, strict=True)
               if np.isfinite(p) and p > 0}
        keep = {}
        for c_, s_ in zip(pos_i, pos_s, strict=True):
            c_ = int(c_)
            q = max(0.0, s_ - tgt.get(c_, 0.0))
            p_ = op[t, c_]
            blocked = susp[t, c_] or ld[t, c_] or not np.isfinite(p_) or p_ <= 0
            if q <= 0 or blocked:
                if q > 0 and (susp[t, c_] or not np.isfinite(p_) or p_ <= 0):
                    frozen += 1
                keep[c_] = s_
                continue
            amt = q * float(p_) * 0.999
            cash += amt - max(5.0, amt * 0.0003) - amt * 0.001
            trades += 1
            if s_ - q > 0:
                keep[c_] = s_ - q
        for c_ in cols:
            c_ = int(c_)
            q = max(0.0, tgt.get(c_, 0.0) - keep.get(c_, 0.0))
            p_ = op[t, c_]
            if q <= 0 or susp[t, c_] or lu[t, c_] or not np.isfinite(p_) or p_ <= 0:
                continue
            fill = float(p_) * 1.001
            q = min(q, float(int(max(0.0, cash - 5.0) // (fill * 100)) * 100))
            while q > 0 and q * fill + max(5.0, q * fill * 0.0003) > cash + 1e-8:
                q -= 100
            if q <= 0:
                continue
            amt = q * fill
            cash -= amt + max(5.0, amt * 0.0003)
            keep[c_] = keep.get(c_, 0.0) + q
            trades += 1
        pos_i = np.array(sorted(keep), np.int64)
        pos_s = np.array([keep[i] for i in pos_i], np.float64)
        a, z = k, bounds[b + 1]
        if len(pos_i):
            seg = np.nan_to_num(cl[np.ix_(days[a:z], pos_i)].astype(np.float64))
            eq[a:z] = cash + seg @ pos_s
        else:
            eq[a:z] = cash
    return eq, days, trades, frozen


def build_sel(reb, ok, logcap, tmean):
    """R10 small_cap_low_turnover:真实市值分位 + 低换手分位,取前 20。"""
    sel, elig, selrank = {}, {}, {}
    for t in reb:
        m = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
        e = np.flatnonzero(m)
        if len(e) < TOP_N * 3:
            continue
        s = (pct(-logcap[t, e].astype(float)) + pct(-tmean[t, e].astype(float))) / 2
        top = e[np.argsort(-s, kind="stable")[:TOP_N]]
        sel[int(t)] = (top, np.full(TOP_N, WEIGHT))
        order = e[np.argsort(logcap[t, e], kind="stable")]   # 市值升序
        elig[int(t)] = order
        rk = {int(c): i for i, c in enumerate(order)}
        selrank[int(t)] = np.array([rk[int(c)] for c in top])
    return sel, elig, selrank


def draw_neighbour(rng, order, ranks, nbr=NBR):
    """对照A:每只在自身市值名次 ±nbr 的邻域内换一只。"""
    n = len(order)
    taken, out = set(), []
    for r in ranks:
        lo, hi = max(0, r - nbr), min(n - 1, r + nbr)
        for _ in range(60):
            p = int(rng.integers(lo, hi + 1))
            if p not in taken:
                break
        taken.add(p)
        out.append(p)
    return np.array(out)


def draw_decile1(rng, order, k=TOP_N):
    """对照B:最小市值十分之一内随机抽 k 只。"""
    n = len(order)
    top = max(k, int(np.ceil(n / 10)))
    return rng.choice(top, size=k, replace=False)


def main():
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), f"锚点A1 {(nt, ns)}"

    # 锚点 A3:真实市值口径核对(宁德时代 2021-11-30 = 13815.5 亿)
    j = codes.index("300750")
    t = int(np.searchsorted(idx, pd.Timestamp("2021-11-30")))
    mv = float(np.exp(logcap[t, j])) / 1e8
    a3 = 13800 <= mv <= 13830
    print(f"锚点A3 宁德时代 {idx[t].date()} 流通市值 {mv:,.1f} 亿 "
          f"∈[13800,13830] ? {'✓' if a3 else '✗'}")
    assert a3, "锚点A3 不过 —— 面板市值被复权污染"

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill()
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)

    sel, elig, selrank = build_sel(reb, ok, logcap, tmean)
    print(f"调仓日 {len(sel)} 个")

    def win(w):
        d0, d1 = WINDOWS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    # ---- 判据 D1:引擎恒等式 ----
    print("\n" + "=" * 74 + "\nD1 引擎自洽锚点\n" + "=" * 74)
    d1 = True
    for w in ("oos", "full"):
        w0, w1 = win(w)
        e1, dd, tr1, fz1 = run_window(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        t0 = time.time()
        e2, _, tr2, fz2 = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        err = float(np.max(np.abs(e2 - e1) / np.maximum(np.abs(e1), 1e-9)))
        good = err < 1e-9 and tr1 == tr2 and fz1 == fz2
        d1 &= good
        print(f"  {w:5s} 最大相对误差 {err:.3e}  成交数 {tr1}/{tr2}  冻结 {fz1}/{fz2}  "
              f"快引擎 {time.time()-t0:.1f}s  {'✓' if good else '✗'}")
    print(f"D1 {'通过' if d1 else '不通过 —— 本节作废'}")
    assert d1, "D1 不通过"

    # ---- 策略基线 ----
    print("\n" + "=" * 74 + "\n修正版 R10(真实市值)基线\n" + "=" * 74)
    base = {}
    for w in WINDOWS:
        w0, w1 = win(w)
        eq, dd, tr, fz = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        m = metrics(eq, dd, idx)
        s = bs[(bs.index >= WINDOWS[w][0]) & (bs.index <= WINDOWS[w][1])]
        m["bench_div"] = float(s.iloc[-1] / s.iloc[0] - 1)
        m["trades"], m["frozen"] = tr, fz
        base[w] = m
        print(f"  {w:11s} 总 {m['total']:+10.2%} 年化 {m['cagr']:+7.2%} "
              f"回撤 {m['mdd']:+7.2%} 夏普 {m['sharpe']:5.2f} 成交 {tr:5d} 冻结 {fz:4d}")
    # ---- 判据 D2 / D3:两组市值中性随机对照 ----
    res = {}
    for kind, name, crit in (("nbr", "对照A 名次±25 邻域匹配", "D2"),
                             ("dec", "对照B 最小十分位内随机", "D3")):
        print("\n" + "=" * 74 + f"\n{crit} {name}({NSEED} 组种子)\n" + "=" * 74)
        t0 = time.time()
        viol = 0
        cg = {w: [] for w in ("full", "oos")}
        for sd in range(NSEED):
            rng = np.random.default_rng(SEED + sd)
            csel = {}
            for t, (_, wts) in sel.items():
                order, ranks = elig[t], selrank[t]
                if kind == "nbr":
                    ps = draw_neighbour(rng, order, ranks)
                    viol += int(np.sum(np.abs(ps - ranks) > NBR))
                else:
                    ps = draw_decile1(rng, order)
                    viol += int(np.sum(ps >= max(TOP_N, int(np.ceil(len(order) / 10)))))
                csel[t] = (order[ps], wts)
            for w in ("full", "oos"):
                w0, w1 = win(w)
                eq, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, csel, cal_pos, w0, w1)
                cg[w].append(metrics(eq, dd, idx)["cagr"])
        print(f"  锚点A2 抽样越界 {viol} 次  {'✓' if viol == 0 else '✗ 作废'}  ({time.time()-t0:.0f}s)")
        assert viol == 0, f"锚点A2 不过({name})"
        ok_all = True
        for w in ("full", "oos"):
            a = np.array(cg[w])
            q95 = float(np.percentile(a, 95))
            st = base[w]["cagr"]
            pv = (1 + int(np.sum(a >= st))) / (NSEED + 1)
            good = st > q95
            ok_all &= good
            print(f"  {w:5s} 策略 {st:+7.2%} | 对照 中位 {np.median(a):+7.2%} "
                  f"95分位 {q95:+7.2%} 最好 {a.max():+7.2%} | 超额(对中位) "
                  f"{(st-np.median(a))*100:+6.2f}pp  p={pv:.4f}  {'✓' if good else '✗'}")
        print(f"  {crit} {'通过' if ok_all else '不通过'}")
        res[crit] = {"pass": bool(ok_all),
                     **{w: {"ctrl": cg[w], "strat": base[w]["cagr"]} for w in ("full", "oos")}}

    # ---- D4(a) 十档等权风格曲线(不含成本、不含整手)----
    print("\n" + "=" * 74 + "\nD4(a) 市值十档等权(不含成本,纯风格)\n" + "=" * 74)
    ret = np.zeros_like(cl)
    ret[1:] = cl[1:] / cl[:-1] - 1.0
    ladder = {}
    for w in ("full", "oos"):
        w0, w1 = win(w)
        days = cal_pos[(cal_pos >= w0) & (cal_pos <= w1)]
        rebs = [int(t) for t in sel if w0 <= t <= w1]
        rebs.sort()
        out = []
        for d in range(10):
            eq, held = 1.0, None
            curve = []
            for k, t in enumerate(days):
                if held is not None and len(held):
                    r = ret[t, held]
                    r = r[np.isfinite(r)]
                    eq *= (1 + (r.mean() if len(r) else 0.0))
                curve.append(eq)
                if int(t) in sel:
                    o = elig[int(t)]
                    n = len(o)
                    a0, a1 = int(d * n / 10), int((d + 1) * n / 10)
                    held = o[a0:a1]
            yrs = (idx[days[-1]] - idx[days[0]]).days / 365.25
            out.append(eq ** (1 / yrs) - 1)
        ladder[w] = out
        print(f"  {w:5s} " + " ".join(f"D{i+1}:{v:+6.1%}" for i, v in enumerate(out)))
        print(f"        策略 {base[w]['cagr']:+.2%}  vs 最小档D1 {out[0]:+.2%}  "
              f"差 {(base[w]['cagr']-out[0])*100:+.2f}pp")

    # ---- D4(c) 剔除最小 5% / 10% 市值 ----
    print("\n" + "=" * 74 + "\nD4(c) 剔除最小市值后的策略\n" + "=" * 74)
    excl = {}
    for cut in (0.0, 0.05, 0.10):
        s2 = {}
        for t in sel:
            o = elig[t]
            n = len(o)
            keep = o[int(np.ceil(n * cut)):]
            m = np.isin(np.arange(ns), keep)
            e = np.flatnonzero(m & ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t]))
            if len(e) < TOP_N * 3:
                continue
            sc = (pct(-logcap[t, e].astype(float)) + pct(-tmean[t, e].astype(float))) / 2
            s2[t] = (e[np.argsort(-sc, kind="stable")[:TOP_N]], np.full(TOP_N, WEIGHT))
        line = []
        for w in ("full", "oos"):
            w0, w1 = win(w)
            eq, dd, _, fz = run_window_fast(op, cl, susp, lu, ld, s2, cal_pos, w0, w1)
            mm = metrics(eq, dd, idx)
            line.append(f"{w}:年化{mm['cagr']:+7.2%} 回撤{mm['mdd']:+7.2%} 冻结{fz:4d}")
            excl[(cut, w)] = mm["cagr"]
        print(f"  剔除最小 {cut:4.0%}  " + "  ".join(line))

    rows = []
    for w, m in base.items():
        rows.append({"window": w, **{k: v for k, v in m.items() if k != "years"}})
    pd.DataFrame(rows).to_csv(f"{OUT}/codex_r10_neutral_base.csv", index=False)
    cd = []
    for crit in res:
        for w in ("full", "oos"):
            for v in res[crit][w]["ctrl"]:
                cd.append({"criterion": crit, "window": w, "ctrl_cagr": v,
                           "strat_cagr": res[crit][w]["strat"]})
    pd.DataFrame(cd).to_csv(f"{OUT}/codex_r10_neutral_controls.csv", index=False)
    pd.DataFrame(ladder).to_csv(f"{OUT}/codex_r10_neutral_ladder.csv", index=False)
    print(f"\n落库 {OUT}/codex_r10_neutral_*.csv")
    return base, res, ladder, excl


if __name__ == "__main__":
    main()
