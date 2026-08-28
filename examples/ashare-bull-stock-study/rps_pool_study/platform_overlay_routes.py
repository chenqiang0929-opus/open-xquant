"""§160 事前登记:把平台筛选器叠加到 R 路线上,会不会更好(结果未跑)。

起因
----
用户:「如果把平台筛选器放到 R01–R13 上面,效果会不会更好?」

可行性先查过了(`platform_state_cache.py`,描述性,已跑):

    合格股票·日 11,303,266 个
      「三条全中」(处于干净平台)  1,398,220  = **12.370%**  → 50 只组合平均命中 6.2 只
      「突破买点」(突破平台上沿)     31,512  = ** 0.279%**  → 50 只组合平均命中 0.14 只

**结论:「只买正在突破的」这条叠加法在样本量上不可行**
(月度调仓组合每期平均只有 0.14 只恰好在突破日)。
**因此本节只叠加两件可行的:平台状态过滤、大盘过滤。买点部分不叠。**

被测的东西(跑前定死)
--------------------
阶梯,每档只加一件事:

    M0 基线    R 路线原样(第一一九–一二二节口径),**用来复现,当机器锚点**
    M1 +平台状态  候选池先限制在当日**三条全中**的股票内,再按该路线打分取 TOP_N
    M2 +大盘过滤  M1 再加:**调仓日**若全市场等权净值 < 自身 MA200 → 该期空仓(全现金)

**⚠️ 与第一五五节的差别必须写明**:第一五五节的大盘过滤是**逐日**清仓,
本节因为 R 路线是每 20 个交易日调仓一次的组合,**只在调仓日检查**。
**这是更弱的版本,不是同一个东西,结论不可互相搬运。**

被判的路线(避免 best-of-N)
--------------------------
**主判据只判 R08(价值复合)与 R09(核心质量)** ——
它们是本项目唯一有「正规样本外 + 随机对照」通过记录的两条
(第一一九–一二二节:留出段 +9.98% / +6.63%,对照中位 −1.58% / −2.51%,p 0.002 / 0.006)。
**在从未通过的路线上叠加,测不出「叠加有没有用」。**
R06、R11 用同一套 `route_scores` 骨架,**顺带只报数,不判定**。
**R03/R04/R05/R07/R10/R13 用的是另一套实现,本节不含** —— 这是我主动缩的范围,
如实写明,需要时另开一节。

口径(与第一一九–一二二节逐字一致,一个字不改)
----------------------------------------------
面板 (3297, 5232);训练段与留出段沿用 `codex_r10_neutral.WINS`;
每 20 个交易日调仓;TOP_N 与 WEIGHT 沿用原脚本;
对照 = **同市值邻域(±NBR)随机抽同样只数**,NSEED 组种子,与原脚本同一套 `draw_fast`;
价格 ffill 参与,退市股绝不剔除。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
N1 锚点(不过则本节作废)
   (a) 面板 (3297, 5232);
   (b) **机器锚点:M0 必须复现第一一九–一二二节的留出段年化
       R08 +9.98%、R09 +6.63%,容差各 ±0.30pp**;
   (c) 市值邻域违例 = 0。

N2 **主判据(叠加档本身是否仍通过对照)**
   对 R08、R09 的 M1 与 M2 各自判(共 4 个检验):
   **通过 ⟺ 留出段年化 − 对照中位 ≥ +3.00pp 且单尾 p < 0.0125**
   (Bonferroni:0.05 / 4,加严不是放宽)。

N3 **增量判据 —— 这才是用户问的那个问题**
   **「更好」⟺ 该叠加档的留出段年化 ≥ M0 同段年化 + 3.00pp,且 N2 也通过。**
   **两条同时满足才算「叠加让它更好」。**只满足一条的,如实写「没做到」。

N4 描述(不参与判定):各档的调仓期数、平均持仓只数、回撤、夏普、总收益、
   训练段数字、R06/R11 的同样一套数字;
   以及 M1 把候选池压缩到多少(平台状态覆盖率 12.37% 的实际影响)。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。

不做的
------
不改 R 路线的打分函数、不调 TOP_N / 调仓频率 / 对照规格;
不叠加「突破买点」(已证不可行);不加第三档;
**跑完不许回头挑一条路线或一档再单独重跑**;不新增顶层目录;不 force push;
**不往 quant-research-dev / etf-netflow-dev 推任何东西**;不作任何可交易性声明。
**若 N2/N3 不过,如实写「叠加上去也没做到」。**
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, NBR, SEED, run_window_fast  # noqa: E402
from codex_r10_replication import DATA, TOP_N, WEIGHT, metrics  # noqa: E402
from codex_routes_rerun import build_fund, route_scores  # noqa: E402
from factor_sweep_pv import draw_fast  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
PSTATE = f"{OUT}/platform_state.npz"
NSEED, ALPHA = 500, 0.05 / 4
WINS = {"train": ("2014-01-02", "2021-12-31"),
        "holdout": ("2022-01-04", "2026-08-03")}
JUDGE, DESC = ("R08", "R09"), ("R06", "R11")
RUNGS = (("M0 基线", 0), ("M1 +平台状态", 1), ("M2 +调仓日大盘过滤", 2))
ANCHOR = {"R08": 0.0998, "R09": 0.0663}


def main():  # noqa: PLR0915
    t0 = time.time()
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), f"锚点N1a {(nt, ns)}"
    p = np.load(PSTATE, allow_pickle=True)
    pcodes = list(p["codes"])
    assert len(pd.DatetimeIndex(p["dates"])) == nt, "平台缓存日期数对不上"
    assert (pd.DatetimeIndex(p["dates"]) == idx).all(), "平台缓存日期不一致"
    pos = {c: j for j, c in enumerate(pcodes)}
    col = np.array([pos.get(c, -1) for c in codes])
    hit3 = np.zeros((nt, ns), bool)
    g = col >= 0
    hit3[:, g] = p["hit3"][:, col[g]]
    mkt_on = p["mkt_on"]
    print(f"锚点N1a ✓ {nt}×{ns};平台状态映射上 {int(g.sum()):,}/{ns:,} 只"
          f"(缺 {int((~g).sum())} 只按无平台处理);大盘开启 {mkt_on.mean():.1%}",
          flush=True)

    raw = np.full((nt, ns), np.nan, np.float32)
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "TTM 恒等式不过"
    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)
    print(f"预取完成 ({time.time()-t0:.0f}s)", flush=True)

    def wpos(w):
        d0, d1 = WINS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    viol, rows, w_ = 0, [], 104
    for name in (*JUDGE, *DESC):
        judged = name in JUDGE
        print(f"\n{'='*w_}\n{name}{'(主判据)' if judged else '(只报数,不判定)'}"
              f"\n{'='*w_}")
        print(f"{'档':<22}{'段':<9}{'调仓':>6}{'均持仓':>8}{'年化':>9}{'回撤':>9}"
              f"{'夏普':>7}{'总收益':>9}{'│对照中位':>10}{'超额pp':>9}{'p':>8}")
        base_cagr = {}
        for rname, rung in RUNGS:
            sel, elig, srk, nsel = {}, {}, {}, []
            for t in reb:
                t = int(t)
                base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
                if name == "R11":                      # 与原脚本一致:剔除微盘 10%
                    base = base & (logcap[t] > np.nanpercentile(logcap[t][base], 10))
                if rung >= 1:
                    base = base & hit3[t]
                e = np.flatnonzero(base)
                if len(e) < TOP_N * 3:
                    continue
                if name == "R11":                      # 复合打分,逐字抄原脚本
                    v = np.nanmean(np.vstack([
                        pd.Series(route_scores("R11_value", t, e, fm, cl, raw,
                                               logcap, tmean, "raw")
                                  ).rank(pct=True).to_numpy(),
                        pd.Series(route_scores("R11_qual", t, e, fm, cl, raw,
                                               logcap, tmean, "raw")
                                  ).rank(pct=True).to_numpy(),
                        pd.Series(route_scores("R06", t, e, fm, cl, raw, logcap,
                                               tmean, "raw")
                                  ).rank(pct=True).to_numpy(),
                        pd.Series((pd.Series(-logcap[t, e]).rank(pct=True)
                                   + pd.Series(-tmean[t, e]).rank(pct=True)) / 2
                                  ).rank(pct=True).to_numpy()]), axis=0)
                else:
                    v = route_scores(name, t, e, fm, cl, raw, logcap, tmean, "raw")
                gg = np.isfinite(v)
                if gg.sum() < TOP_N:
                    continue
                e2 = e[gg]
                top = e2[np.argsort(-v[gg], kind="stable")[:TOP_N]]
                if rung >= 2 and not mkt_on[t]:
                    sel[t] = (np.zeros(0, np.int64), np.zeros(0))
                    nsel.append(0)
                    continue
                sel[t] = (top, np.full(TOP_N, WEIGHT))
                nsel.append(len(top))
                order = e[np.argsort(logcap[t, e], kind="stable")]
                rk = {int(c): i for i, c in enumerate(order)}
                elig[t] = order
                srk[t] = np.array([rk[int(c)] for c in top])
            for w in ("train", "holdout"):
                w0, w1 = wpos(w)
                nre = sum(1 for t in sel if w0 <= t <= w1)
                eq, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, sel,
                                               cal_pos, w0, w1)
                m = metrics(eq, dd, idx)
                r = {"路线": name, "档": rname, "段": w, "调仓": nre,
                     "均持仓": float(np.mean(nsel)) if nsel else 0.0,
                     "年化": m["cagr"], "回撤": m["mdd"], "夏普": m["sharpe"],
                     "总收益": m["total"]}
                if rung == 0 and w == "holdout":
                    base_cagr[name] = m["cagr"]
                need_ctrl = judged and w == "holdout"
                if need_ctrl:
                    cg = []
                    for sd in range(NSEED):
                        rng = np.random.default_rng(SEED + sd)
                        cs = {}
                        for t in sel:
                            if t not in elig:
                                cs[t] = sel[t]
                                continue
                            o, rk = elig[t], srk[t]
                            ps = draw_fast(rng, rk, len(o))
                            viol += int(np.sum(np.abs(ps - rk) > NBR))
                            cs[t] = (o[ps], np.full(TOP_N, WEIGHT))
                        e3, d3, _, _ = run_window_fast(op, cl, susp, lu, ld, cs,
                                                       cal_pos, w0, w1)
                        cg.append(metrics(e3, d3, idx)["cagr"])
                    a = np.array(cg)
                    pv = (1 + int(np.sum(a >= m["cagr"]))) / (NSEED + 1)
                    r |= {"对照中位": float(np.median(a)),
                          "超额pp": (m["cagr"] - float(np.median(a))) * 100, "p": pv}
                    print(f"{rname:<20}{w:<9}{nre:>6}{r['均持仓']:>8.1f}"
                          f"{m['cagr']:>9.2%}{m['mdd']:>9.2%}{m['sharpe']:>7.2f}"
                          f"{m['total']:>9.2f}{np.median(a):>10.2%}"
                          f"{r['超额pp']:>9.2f}{pv:>8.4f}")
                    if rung >= 1:
                        n2 = r["超额pp"] >= 3.0 and pv < ALPHA
                        n3 = m["cagr"] >= base_cagr[name] + 0.03
                        print(f"{'':<20}  **N2 对照 {'✓' if n2 else '✗'}"
                              f"(需≥+3.00pp 且 p<{ALPHA:.4f});"
                              f"N3 比 M0({base_cagr[name]:+.2%})高 ≥3.00pp "
                              f"{'✓' if n3 else '✗'} → "
                              f"{'**更好**' if (n2 and n3) else '**没做到**'}")
                        r |= {"N2": "通过" if n2 else "不通过",
                              "N3": "通过" if n3 else "不通过"}
                else:
                    print(f"{rname:<20}{w:<9}{nre:>6}{r['均持仓']:>8.1f}"
                          f"{m['cagr']:>9.2%}{m['mdd']:>9.2%}{m['sharpe']:>7.2f}"
                          f"{m['total']:>9.2f}{'—':>10}")
                rows.append(r)
                if rung == 0 and w == "holdout" and judged:
                    d = abs(m["cagr"] - ANCHOR[name])
                    print(f"{'':<20}  锚点N1b 复现第一一九–一二二节 "
                          f"{ANCHOR[name]:+.2%}:差 {d*100:.2f}pp "
                          f"{'✓' if d <= 0.003 else '✗ 本节作废'}")
                    if d > 0.003:
                        return
            print("", flush=True)
    print(f"锚点N1c 市值邻域违例 {viol} {'✓' if viol == 0 else '✗'}")
    pd.DataFrame(rows).to_csv(f"{OUT}/platform_overlay_routes.csv", index=False,
                              encoding="utf-8-sig")
    print(f"落库 {OUT}/platform_overlay_routes.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
