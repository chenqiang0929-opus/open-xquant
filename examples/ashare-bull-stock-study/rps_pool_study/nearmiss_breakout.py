"""第一八一节 事前登记:「近阈值突破候选」层 —— 差一条的那些,值不值得看(结果未跑)。

起因
----
Codex 2026-09-02 来信第 6 节提议建一个「近阈值突破候选」研究层:
已经发生价格突破、但被某一项入口条件挡住的股票,记下来看看。
我第一七八节 C1 登记过这条,并写死了硬边界:
**不许用宁德时代 / 仕佳光子 / 胜宏科技(或任何已知牛股)去定阈值。**

**本节不复刻他的入口规则** —— 第一七三节我已经栽过一次(把他 §3.2 的
close 口径套到 §3.5 的 high 上)。改用**我自己的平台筛选器**(第一五五节遗留口径),
三条阈值是我自己定的、我完全清楚,不存在反向工程的误读风险。

事件与分组(**零新增阈值**)
----------------------------
平台筛选器在每个交易日给出:平台上沿 `phi`、缩量比 `shr`、收敛比 `cnv`、
深度 `dep`、调整段起点 `adj_a`(≥0 表示存在合法平台段)。

    突破日 t ⟺ 收盘 首次上穿**前一日的**平台上沿(`cl[t] > phi[t−1]`,
               且前一日不是突破日),且 `adj_a[t] ≥ 0`(确实存在平台段)、当日可交易。

**口径说明(首跑前修正,尚未看到任何结果)**:缓存里的 `phi` 是**含当日**的
平台上沿 —— 688347 在 2025-02-17 的 `phi` 就等于当天收盘 51.72,
所以 `cl[t] > phi[t]` 恒不成立(首版这么写,全市场 0 个事件)。
正确写法是与**前一日**的上沿比:`phi[2025-02-14] = 50.62`,收盘 51.72 > 50.62,
是突破。**这是口径错误的修正,不是判据放宽;修正时 P2 的任何数字都还没跑出来。**

在突破日,三条阈值(**§155 原值,一个字不改**)各自成立与否:

    缩量比 shr < 0.80      收敛比 cnv < 0.80      深度 dep ≤ 0.352

按**失败条数**分组,**不设任何"接近"的容差** —— 这正是第一七八节 C1
登记的第一步「先把被哪一条挡住记成分类字段,不改任何阈值」:

    G0 正式突破      三条全过(= 第一五五节的平台突破)
    G1 仅缩量比不过  只有 shr 这一条失败
    G2 仅收敛比不过  只有 cnv 这一条失败
    G3 仅深度不过    只有 dep 这一条失败
    G4 差两条及以上  远离阈值,作为对照

口径
----
面板 (3316, 5232) 末日 2026-08-28;平台缓存 `platform/rps50`(第一六八节口径,
强势日 RPS50 ≥ 90);持有 5/20/60/120/250 交易日,**判据横期 60 日**;
对照同日、同市值名次 ±25、同申万一级,200 组种子(p 下限 1/201);
训练段 2013→2021 只报数,**留出段 2022-01-01 起判据**;
退市股按最后有效价 ffill 参与,绝不剔除(用户规则 5)。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
P1 锚点(不过则本节作废)
   (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) **688347 的 2025-02-17 必须落在 G1「仅缩量比不过」组** ——
       这一天我在第一六六节逐日核过:收盘 51.72 > 上沿 49.69,
       但缩量比 0.8049 > 0.80,是**唯一**不过的一条。
       把口径钉在已核过的个案上,对不上即作废;
   (c) 对照抽样市值名次偏离 > 25 的违例 = 0。

P2 **主判据**(留出段、60 日持有)
   **通过 ⟺ G1/G2/G3 三个「恰好差一条」组中,至少一个的年化超额 ≥ +3.00pp
   且单尾 p < 0.05。** 门槛与第一五二/一五五/一六八/一七三/一七四/一七六节一致,
   一个字不放宽。**三组全部报告;若只有一个过,按 Bonferroni 用
   α = 0.05/3 = 0.0167 复判,两个结论都写。**

P3 描述(必报,不参与判定)
   G0(正式突破)与 G4(差两条及以上)的同表数字 ——
   用来看候选层相对正式层是更好还是更差。

事前预测
--------
自第一一九节起不对「样本外会不会过」下预测,**只登记判据**。
但登记一条边界:**第一五五节已判定平台筛选器整体不通过(选股 −5.27pp / p 0.810),
所以本节即使某一组过了,也只能说"这一条阈值挡错了人",
不能说"平台筛选器成立"。**

**本文件不构成任何投资建议。**
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codex_r10_neutral import NBR, SEED  # noqa: E402
from industry_neutral import build_industry  # noqa: E402
from panel_cache import cached  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR",
                      "/home/user/oxq-panel-0828/oxq_stock_market_fixed")
OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
STRONG_N = int(os.environ.get("OXQ_STRONG_N", "50"))
HOLD = (5, 20, 60, 120, 250)
NSEED, JUDGE_H = 200, 60
THR_SHRINK, THR_ATR, THR_DEPTH = 0.80, 0.80, 0.352      # §155 原值,不改


def ann(r, h):
    return (1.0 + r) ** (250.0 / h) - 1.0 if r > -1 else np.nan


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    p = cached("panel", DATA, lambda: (_ for _ in ()).throw(
        AssertionError("锚点:panel 缓存必须已存在")))
    idx = pd.DatetimeIndex(p["idx"])
    cl, okm = p["cl"], p["okm"]
    nt, ns = cl.shape
    assert (nt, ns) == (3316, 5232), f"锚点P1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点P1a 末日 {idx[-1].date()}"
    print(f"锚点P1a ✓ {(nt, ns)} 末日 {idx[-1].date()}", flush=True)
    mvm = cached("mv", DATA, lambda: (_ for _ in ()).throw(
        AssertionError("锚点:mv 缓存必须已存在")))["mv"]
    ind, _, _ = build_industry(codes, idx)
    q = cached("platform", DATA, lambda: (_ for _ in ()).throw(
        AssertionError("锚点:platform 缓存必须已存在")), extra=f"rps{STRONG_N}")
    phi, shr, cnv, dep, adj = q["phi"], q["shr"], q["cnv"], q["dep"], q["adj_a"]
    print(f"平台缓存就绪 ({time.time()-t0:.0f}s)", flush=True)

    phip = np.vstack([np.full((1, ns), np.nan), phi[:-1]])   # 前一日的平台上沿
    with np.errstate(all="ignore"):
        above = np.isfinite(phip) & (cl > phip)
    prev = np.zeros_like(above)
    prev[1:] = above[:-1]                                    # 连续突破不重复计数
    brk = above & ~prev & (adj >= 0) & okm
    brk[:, :] &= np.isfinite(shr) & np.isfinite(cnv) & np.isfinite(dep)
    tgrid = np.arange(nt)[:, None]
    brk &= tgrid < nt - max(HOLD)
    tt, jj = np.nonzero(brk)
    nfail = ((shr[tt, jj] >= THR_SHRINK).astype(int)
             + (cnv[tt, jj] >= THR_ATR).astype(int)
             + (dep[tt, jj] > THR_DEPTH).astype(int))
    only = np.full(len(tt), 4, np.int8)                   # 4 = 差两条及以上
    only[nfail == 0] = 0
    only[(nfail == 1) & (shr[tt, jj] >= THR_SHRINK)] = 1
    only[(nfail == 1) & (cnv[tt, jj] >= THR_ATR)] = 2
    only[(nfail == 1) & (dep[tt, jj] > THR_DEPTH)] = 3
    labs = {0: "G0 正式突破(三条全过)", 1: "G1 仅缩量比不过",
            2: "G2 仅收敛比不过", 3: "G3 仅深度不过", 4: "G4 差两条及以上"}
    print(f"突破日合计 {len(tt):,} 个;" + "、".join(
        f"{labs[k]} {int((only == k).sum()):,}" for k in range(5)), flush=True)

    # 锚点 P1b:688347 2025-02-17 必须在 G1
    j0 = codes.index("688347")
    t0i = int(np.searchsorted(idx.values, np.datetime64("2025-02-17")))
    hit = np.flatnonzero((tt == t0i) & (jj == j0))
    ok_b = bool(len(hit)) and int(only[hit[0]]) == 1
    print("锚点P1b 688347 2025-02-17:"
          + (f"在 {labs[int(only[hit[0]])]}(缩量比 {shr[t0i, j0]:.4f}、"
             f"收敛比 {cnv[t0i, j0]:.4f}、深度 {dep[t0i, j0]:.4f}、"
             f"前日上沿 {phi[t0i - 1, j0]:.2f}、收盘 {cl[t0i, j0]:.2f})"
             if len(hit) else "不是突破日")
          + f" {'✓' if ok_b else '✗ 本节作废'}", flush=True)
    if not ok_b:
        return

    def controls(cts):
        rng = np.random.default_rng(SEED)
        out = np.full((NSEED, len(cts)), -1, np.int32)
        viol, cache = 0, {}
        for k, (t, j) in enumerate(zip(cts, jj, strict=True)):
            if t not in cache:
                el = np.flatnonzero(okm[t] & np.isfinite(mvm[t]) & (ind[t] >= 0))
                if not len(el):
                    cache[t] = None
                else:
                    o = el[np.argsort(mvm[t, el], kind="stable")]
                    rk = np.full(ns, -1, np.int32)
                    rk[o] = np.arange(len(o), dtype=np.int32)
                    cache[t] = (o, rk)
            if cache[t] is None:
                continue
            o, rk = cache[t]
            p_, i0 = rk[j], ind[t, j]
            if p_ < 0:
                continue
            lo, hi = max(0, p_ - NBR), min(len(o), p_ + NBR + 1)
            cand = o[lo:hi]
            cand = cand[(ind[t, cand] == i0) & (cand != j)]
            if not len(cand):
                continue
            out[:, k] = rng.choice(cand, NSEED, replace=True)
            viol += int(np.any(np.abs(rk[out[:, k]] - p_) > NBR))
        return out, viol

    cs, viol = controls(tt)
    print(f"锚点P1c 抽样违例 {viol} 个 {'✓' if viol == 0 else '✗ 作废'} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if viol:
        return
    has = cs[0] >= 0
    split = int(np.searchsorted(idx.values, np.datetime64("2022-01-01")))
    segs = (("训练段13-21", tt < split), ("留出段22-26", tt >= split))

    rows, w = [], 106
    print(f"\n{'='*w}\n按「差哪一条」分组的后验\n{'='*w}")
    print(f"{'段':<12}{'组':<22}{'事件':>8}{'持有':>5}{'事件收益':>10}"
          f"{'对照中位':>10}{'超额pp':>9}{'年化超额pp':>12}{'p':>8}")
    for sn, sm in segs:
        for k in range(5):
            for h in HOLD:
                m = sm & (only == k) & has
                p0 = cl[tt, jj]
                p1 = cl[np.clip(tt + h, 0, nt - 1), jj]
                with np.errstate(all="ignore"):
                    r = p1 / np.where(p0 > 0, p0, np.nan) - 1.0
                m = m & np.isfinite(r)
                if m.sum() < 30:
                    continue
                a = float(np.nanmean(r[m]))
                cm = np.empty(NSEED)
                for s in range(NSEED):
                    ci = cs[s][m]
                    cp0 = cl[tt[m], ci]
                    cp1 = cl[np.clip(tt[m] + h, 0, nt - 1), ci]
                    with np.errstate(all="ignore"):
                        cm[s] = np.nanmean(cp1 / np.where(cp0 > 0, cp0, np.nan) - 1)
                med = float(np.nanmedian(cm))
                rec = {"段": sn, "组": labs[k], "组号": k, "n": int(m.sum()),
                       "持有": h, "事件收益": a, "对照中位": med,
                       "超额pp": (a - med) * 100,
                       "年化超额pp": (ann(a, h) - ann(med, h)) * 100,
                       "p": float((np.sum(cm >= a) + 1) / (NSEED + 1))}
                rows.append(rec)
                print(f"{sn:<12}{labs[k]:<22}{rec['n']:>8,}{h:>5}{a:>+10.2%}"
                      f"{med:>+10.2%}{rec['超额pp']:>+9.2f}"
                      f"{rec['年化超额pp']:>+12.2f}{rec['p']:>8.4f}")

    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/nearmiss_breakout.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'='*w}\nP2 判定(留出段、{JUDGE_H} 日;门槛 +3.00pp 且 p<0.05)\n{'='*w}")
    npass, ps = 0, {}
    for k in (1, 2, 3):
        z = d[(d["段"] == "留出段22-26") & (d["组号"] == k) & (d["持有"] == JUDGE_H)]
        if not len(z):
            print(f"  {labs[k]}:样本不足,不通过")
            continue
        y = z.iloc[0]
        ok = bool(y["年化超额pp"] >= 3.0 and y["p"] < 0.05)
        npass += ok
        ps[labs[k]] = float(y["p"])
        print(f"  {labs[k]}:{y['年化超额pp']:+.2f}pp、p={y['p']:.4f}、"
              f"n={y['n']:,} → {'✓ 通过' if ok else '✗ 不通过'}")
    print(f"\nP2 三组通过 {npass} 组。", end="")
    if npass == 1:
        kk = [a for a, b in ps.items() if b < 0.05][0]
        print(f"唯一通过的是 {kk},这是 3 选 1 的 best-of-N;"
              f"Bonferroni α=0.0167 复判:p={ps[kk]:.4f} "
              f"{'仍过' if ps[kk] < 0.0167 else '不过'}。")
    else:
        print("无需 Bonferroni 复判。")
    print("\nP3 描述:G0 正式突破 / G4 差两条及以上(不参与判定)")
    for k in (0, 4):
        for sn in ("训练段13-21", "留出段22-26"):
            z = d[(d["段"] == sn) & (d["组号"] == k) & (d["持有"] == JUDGE_H)]
            if len(z):
                y = z.iloc[0]
                print(f"  {sn} {labs[k]}:{y['年化超额pp']:+.2f}pp、"
                      f"p={y['p']:.4f}、n={y['n']:,}")
    print("\n边界:第一五五节已判定平台筛选器整体不通过(选股 −5.27pp / p 0.810);"
          "本节即使某组通过,也只说明这一条阈值挡错了人,不说明筛选器成立。")
    print(f"落库 {OUT}/nearmiss_breakout.csv ({time.time()-t0:.0f}s)")
    print("本表是状态记录,不是买点,不构成任何投资建议。")


if __name__ == "__main__":
    main()
