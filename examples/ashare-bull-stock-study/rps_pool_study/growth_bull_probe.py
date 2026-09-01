"""第一七一节 事前登记:营收/净利增速高,找到牛股的概率是不是更大(结果未跑)。

起因
----
用户问两件事:
  (1) R08/R09 与营收增速、净利增速是否有关系;
  (2) **增速越高,找到牛股的概率是不是更大?**

第 (1) 已用 Codex 662 池当日截面量过(他自己的四列):
  R08 × 营收增速 ρ = −0.230、× 净利增速 ρ = −0.013
  R09 × 营收增速 ρ = +0.318、**× 净利增速 ρ = +0.734**
  R08 × R09     ρ = +0.095
但那是**一个日期、662 只次新股**的截面,不能当结论。本节做全市场、全历史的版本。

**为什么必须控市值**
--------------------
第一六九节刚查出一个规模效应:R08/R09 的超额相对同市值对照 +11.6pp,
相对全市场等权只剩 1.45pp —— **差 21pp**。高增速股天然偏小盘,
而 2023–25 小盘大涨。**不控市值,量出来的「增速有用」很可能只是「小盘有用」。**
所以本节每个指标都出两版:**原始十分位** 与 **市值五分位内的十分位**(组内排名)。

口径
----
- 面板 (3297, 5217),调仓日与第一一七/一六九节相同(`cal_pos[::20]`,153 个);
- 合格:非 ST、非停牌、上市满 250 日、当日有成交(复用缓存 OK);
- **增速定义**:`g = X_ttm[t] / X_ttm[t-250] - 1`,**仅在 `X_ttm[t-250] > 0` 时有效**
  —— 负基数上的增长率没有意义,按缺失处理(与 R08/R09 的资格门同一思路);
  X 取 `rev_ttm`(营收)与 `ni_ttm`(净利),均来自 `build_fund` 的 PIT 序列;
- **牛股定义(写死,不改)**:自调仓日起 **未来 250 个交易日内最大涨幅 ≥ +100%**
  (一年内翻倍)。同时报一个较松的:**未来 120 日内最大涨幅 ≥ +50%**。
  价格用前复权(涨幅比值与复权基准无关)。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
F1 锚点(不过则本节作废)
   (a) 面板 (3297, 5217);(b) TTM 恒等式违例 = 0;(c) 泰格同比违例 = 0;
   (d) **机器复现**:用同一套机器算 R08 的留出段十分位年化,D1 必须复现
       第一六九节的 **+6.12%**、D10 必须复现 **+19.33%**(各 ±0.30pp)。
       —— 证明本节的分档与收益计算路径与第一六九节是同一套。

F2 **主判据:市值中性后的单调性**(留出段 2023-01→2025-12 判,full 只报数)
   **市值五分位内**的十分位牛股命中率对分位序号 1..10 的 Spearman ρ:
   **ρ ≥ 0.60 → 判「该增速指标确实提高找到牛股的概率」;
     ρ < 0.60 → 判「不能仅凭增速提高牛股概率」。**
   营收增速与净利增速**各判各的**,两个牛股定义**各判各的**。

F3 描述(不参与判定):原始十分位(不控市值)的命中率与年化、
   每个十分位的市值中位数(用来显示规模混杂有多大)、
   D10 与 D1 的命中率之比、全市场基准命中率。

**判据写法自律**:绝对阈值,不写比值判据(第一五四节 A3 的教训)。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。**只登记判据。**

不做的
------
不改 `src/oxq/`;不调牛股定义 / 调仓间隔 / 合格口径;**跑完不许回头改阈值再跑**;
不新增顶层目录;不 force push;**不往 quant-research-dev / etf-netflow-dev 推**;
**不作任何可交易性声明** —— 本节是后验描述,不是选股规则。
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
from codex_r10_neutral import CACHE  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from codex_routes_rerun import WINS, build_fund, route_scores  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NB, NQ, LAG = 10, 5, 250          # 十分位、市值五分位、增速回看
BULL = ((250, 1.00), (120, 0.50))  # (未来交易日数, 最大涨幅门槛)


def ann(eqv, nd):
    return eqv ** (250.0 / nd) - 1.0 if (eqv > 0 and nd > 0) else np.nan


def main():  # noqa: PLR0915
    t0 = time.time()
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    cl, ok = z["CL"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), f"锚点F1a {(nt, ns)}"
    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    truth = {(2017, "中报"): .5307, (2017, "三季报"): 1.0103,
             (2017, "年报"): 1.1401, (2018, "一季报"): 1.2107}
    bad = [k for k, v in truth.items() if abs(float(y.get(k, np.nan)) - v) > 0.005]
    assert not bad, f"锚点F1c 不过 {bad}"
    print(f"锚点F1a ✓ {nt}×{ns};F1c ✓ 泰格违例 0", flush=True)

    raw = np.full((nt, ns), np.nan, np.float32)
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点F1b TTM 恒等式不过"
    print(f"锚点F1b ✓ TTM 违例 0 ({time.time()-t0:.0f}s)", flush=True)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = [int(t) for t in cal_pos[::20]]
    ipos = pd.Index(idx)
    clf = cl.astype(np.float64)

    def fwd_max(t, e, h):
        """自 t 起未来 h 个交易日的最大涨幅(用前复权收盘)。"""
        t2 = min(t + h, nt - 1)
        if t2 <= t:
            return np.full(len(e), np.nan)
        p0 = clf[t, e]
        seg = clf[t + 1:t2 + 1, e]
        with np.errstate(all="ignore"):
            return np.nanmax(seg, axis=0) / np.where(p0 > 0, p0, np.nan) - 1.0

    def growth(t, e, key):
        cur, pre = fm[key][t, e].astype(np.float64), fm[key][t - LAG, e].astype(np.float64)
        with np.errstate(all="ignore"):
            g = np.where(pre > 0, cur / np.where(pre > 0, pre, np.nan) - 1.0, np.nan)
        return g

    def buckets(v, k, within=None):
        """把 v 切成 k 份,返回每只的档号(0..k-1),NaN 处为 -1。
        within 非空时在 within 的每个组内分别切(市值中性)。"""
        out = np.full(len(v), -1, np.int64)
        groups = [np.arange(len(v))] if within is None else [
            np.flatnonzero(within == q) for q in np.unique(within[within >= 0])]
        for gidx in groups:
            g = gidx[np.isfinite(v[gidx])]
            if len(g) < k * 5:
                continue
            o = g[np.argsort(v[g], kind="stable")]
            for i in range(k):
                lo, hi = int(round(i * len(o) / k)), int(round((i + 1) * len(o) / k))
                out[o[lo:hi]] = i
        return out

    rows, mono, anch = [], [], []
    for wname in ("full", "oos"):
        d0, d1 = WINS[wname]
        w0 = int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0])
        w1 = int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0])
        days = [t for t in reb if w0 <= t < w1]
        # 累计容器
        acc = {}
        base_hit = {h: [0, 0] for h, _ in BULL}
        r08_eq = {i: [1.0, 0] for i in range(NB)}
        for t in days:
            e = np.flatnonzero(ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t]))
            if len(e) < NB * 20:
                continue
            capq = buckets(logcap[t, e].astype(np.float64), NQ)
            fw = {h: fwd_max(t, e, h) for h, _ in BULL}
            for h, thr in BULL:
                g = np.isfinite(fw[h])
                base_hit[h][0] += int((fw[h][g] >= thr).sum())
                base_hit[h][1] += int(g.sum())
            gv = {"营收增速": growth(t, e, "rev_ttm"), "净利增速": growth(t, e, "ni_ttm")}
            # F1(d) 机器复现:同一套分档 + 收盘对收盘,算 R08 十分位年化
            v8 = np.full(len(e), np.nan)
            v8[:] = route_scores("R08", t, e, fm, cl, raw, logcap, tmean, "raw")
            bk8 = buckets(v8, NB)
            t2 = reb[reb.index(t) + 1] if reb.index(t) + 1 < len(reb) else None
            if t2 is not None:
                p0, p1 = clf[t, e], clf[min(t2, nt - 1), e]
                with np.errstate(all="ignore"):
                    ret = p1 / np.where(p0 > 0, p0, np.nan) - 1.0
                for i in range(NB):
                    s = np.flatnonzero(bk8 == i)
                    s = s[np.isfinite(ret[s])]
                    if len(s):
                        r08_eq[i][0] *= 1.0 + float(np.mean(ret[s]))
                        r08_eq[i][1] += t2 - t
            for gname, v in gv.items():
                for mode, wi in (("原始", None), ("市值中性", capq)):
                    bk = buckets(v, NB, within=wi)
                    for h, thr in BULL:
                        for i in range(NB):
                            s = np.flatnonzero(bk == i)
                            s = s[np.isfinite(fw[h][s])]
                            if not len(s):
                                continue
                            k = (gname, mode, h, i)
                            a = acc.setdefault(k, [0, 0, []])
                            a[0] += int((fw[h][s] >= thr).sum())
                            a[1] += len(s)
                            a[2].append(float(np.median(logcap[t, e][s])))
        if wname == "oos":
            for i in (0, NB - 1):
                a = ann(r08_eq[i][0], r08_eq[i][1])
                exp = 0.0612 if i == 0 else 0.1933
                good = abs(a - exp) <= 0.0030
                anch.append({"项": f"R08 留出段 D{i+1}", "本节": a, "第169节": exp,
                             "差pp": (a - exp) * 100, "过": good})
                print(f"锚点F1d R08 留出段 D{i+1}:{a:+.2%}(第一六九节 {exp:+.2%},"
                      f"差 {(a-exp)*100:+.3f}pp) {'✓' if good else '✗ 本节作废'}", flush=True)
        for h, thr in BULL:
            bh = base_hit[h][0] / max(base_hit[h][1], 1)
            print(f"\n[{wname}] 牛股基准:未来{h}日最大涨幅≥{thr:+.0%} "
                  f"全市场命中率 {bh:.2%}({base_hit[h][1]:,} 个股票-调仓日)")
            for gname in ("营收增速", "净利增速"):
                for mode in ("原始", "市值中性"):
                    hr = []
                    for i in range(NB):
                        a = acc.get((gname, mode, h, i))
                        hr.append(a[0] / a[1] if a and a[1] else np.nan)
                        if a:
                            rows.append({"窗口": wname, "指标": gname, "分档方式": mode,
                                         "牛股定义": f"{h}日≥{thr:+.0%}", "分位": f"D{i+1}",
                                         "命中率": hr[-1], "样本": a[1],
                                         "相对基准倍数": (hr[-1] / bh) if bh else np.nan,
                                         "logcap中位": float(np.mean(a[2]))})
                    r = pd.Series(hr).corr(pd.Series(range(1, NB + 1)), method="spearman")
                    judge = "—"
                    if wname == "oos" and mode == "市值中性":
                        judge = ("确实提高牛股概率" if r >= 0.60
                                 else "不能仅凭增速提高牛股概率")
                    mono.append({"窗口": wname, "指标": gname, "分档方式": mode,
                                 "牛股定义": f"{h}日≥{thr:+.0%}", "Spearman_ρ": r,
                                 "D1命中": hr[0], "D10命中": hr[-1],
                                 "基准命中": bh, "判定(F2)": judge})
                    print(f"  {gname}/{mode:<5} " +
                          " ".join(f"{x:.1%}" for x in hr) +
                          f"  ρ={r:+.3f}" + (f"  → {judge}" if judge != "—" else ""))
    pd.DataFrame(anch).to_csv(f"{OUT}/growth_bull_anchor.csv", index=False,
                              encoding="utf-8-sig")
    pd.DataFrame(mono).to_csv(f"{OUT}/growth_bull_mono.csv", index=False,
                              encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(f"{OUT}/growth_bull.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print(pd.DataFrame(mono).to_string(index=False))
    print(f"\n完成 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
