"""第一七四节 事前登记:长期平台突破加一条「回踩确认」,超额能不能回来(结果未跑)。

【第一七五节补充登记 —— 时间样本外】
--------------------------------
第一七四节首跑的弱点是**把 2013-2026 全部数据一起用了**,而本项目自第一五二节起的
标准一直是「训练段只报数,留出段判据在那里」。R08/R09 之所以站得住,正是过了这一关。
所以 +2.86pp 那个数含金量低于表面。本次按时间切开重判,**判据门槛一个字不改**:

  **训练段 2013-07 → 2021-12:只报数,不判。**
  **留出段 2022-01-01 → 2026-08-28:判据在这里。**

J2 主判据(留出段、60 日持有、纯结构不加 RPS 门槛):
  **通过 ⟺ 四个 (N,k) 组合中至少一个的年化超额 ≥ +3.00pp 且单尾 p < 0.05。**
  四个组合全部报告;**若只有一个过,按 Bonferroni 用 α = 0.05/4 = 0.0125 复判**,
  两个结论都写。

**切分点是按项目惯例定的(与第一五二/一五五/一六八节的留出段起点一致),
不是看过结果后挑的。** 若留出段不过而全样本过,**以留出段为准**,
并在正文写明「+2.86pp 是全样本挖出来的」。


起因
----
第一七三节把 Codex 的长期平台突破(v2 34,737 / v3 36,297 事件)做了收益后验:
  主判据(他建议的 RPS50≥80、60 日)**不通过**:年化超额 −1.02pp、p 0.9502;
  RPS 门槛越严超额越差,单调:无门槛 +0.54pp → ≥75 −0.01pp → ≥90 −1.15pp;
  他为覆盖平安/寒武纪把量比从 0.8 一路放到 1.25,新增事件 60 日收益 3.51%,
  全样本 5.82% —— **每一次放宽都在消耗那点本就很薄的超额**。

随后查了三只目标股在设定特征上的位置,结果很硬:
  它们**结果极端好**(60 日收益百分位 91% / 100% / 99%),
  但**设定特征平平甚至在错的一侧** ——
  寒武纪 `recent_volume_ratio` 第 96 百分位、平安 `recent_atr_ratio` 第 95 百分位,
  而这两个特征与收益都是**负相关**(Q1 8.0%→Q5 4.5%、Q1 8.4%→Q5 3.7%)。
  **用它们标定阈值,等于把「好结果」倒灌进「设定特征」,这就是过拟合。**

**所以本节不再动设定特征。** 改试 Codex 自己在 §七 提过的一条:
**突破之后的回踩确认。** 它用的是突破日**之后**的信息,不是设定特征,
因此不受「用结果标定设定」的污染;实盘也可观察(突破后等几天再决定)。

规则(跑之前写死)
----------------
对每个突破事件(突破日 t、平台上沿 U = `upper_before_breakout`):
  **确认成立 ⟺ 突破后 N 个交易日内,最低收盘 ≥ U × (1 − k)**
  —— 即回踩没有跌破平台上沿(k 为容差)。
  N ∈ {5, 10};k ∈ {0, 0.03}。**四个组合全部登记、全部报告,不许只说好看的那个。**
**入场改在「确认日」= t + N**,收益从 `close[t+N]` 起算,不是从突破日起算。
**代价先写明:晚 N 天入场,吃不到突破当天那一段。**

**上沿的口径**:用 Codex CSV 里的 `upper_before_breakout` 与 `breakout_close` 的**比值**
换算到我的面板(`U_mine = close[t] / (breakout_close / upper_before_breakout)`)——
两边前复权锚定日不同,绝对价不可比、比值可比(第一七三节已验:三只样本
比值复现到 −0.016% / −0.016% / +0.000%)。

口径
----
- 事件源:Codex v3 `long_platform_breakouts.csv`(36,297 行),**不改他的事件**;
- 面板 (3316, 5232),末日 2026-08-28;退市股 ffill 参与(用户规则5);
- 持有期 5 / 20 / 60 / 120 / 250 个交易日,**从确认日起算**;
- 对照:**同一确认日**,同市值名次 ±25 且同申万一级行业随机抽 1 只,
  走完全相同的持有期,**200 组种子**(p 下限 1/201 = 0.00498)。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
I1 锚点(不过则本节作废)
   (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) 事件映射率 ≥ 90%;
   (c) 对照抽样市值名次偏离 > 25 的违例 = 0。

I2 **主判据**(60 日持有,**纯结构、不加 RPS 门槛**,零成本)
   **通过 ⟺ 四个 (N, k) 组合中至少一个的年化超额 ≥ +3.00pp 且单尾 p < 0.05。**
   **门槛与第一五二/一五三/一五四/一五五/一六八/一七三节完全一致,一个字不放宽。**
   选「纯结构、不加 RPS」,是因为第一七三节已测出 RPS 门槛让超额单调变差,
   再叠 RPS 只会更差 —— **这是加严,不是放宽**。
   **四个组合全部报告;若只有一个过,须在正文写明这是 4 选 1 的 best-of-N,
   并按 Bonferroni 用 α = 0.05/4 = 0.0125 复判一次,两个结论都写。**

I3 描述(必报,不参与判定)
   (a) 每个组合的确认率(多少比例的突破通过了回踩确认);
   (b) 未通过确认的那批事件的收益 —— 确认这条规则是不是真的在分离好坏;
   (c) 三只目标股(601318 / 688256 / 688347)是否通过确认
       —— **只作描述,不作判据**:它们已被用来标定过参数,不能再当证据;
   (d) 与第一七三节「不加确认」的基线并排。

**判据写法自律**:绝对阈值,不写比值判据(第一五四节 A3 的教训)。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。**只登记判据。**

**必须先声明的一条**:即使 I2 过了,**这仍然是在同一批数据上找出来的**,
真正的检验要等新的时间段。本节结论一律只能写成「在这批数据上成立」。

不做的
------
不改 Codex 的事件与设定参数;不改 `src/oxq/`;不调持有期 / 对照口径 / 判据门槛;
**跑完不许回头改 N 或 k 再跑**;不新增顶层目录;不 force push;
**不往 quant-research-dev / etf-netflow-dev 推**;**不作任何可交易性声明**。
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
EV = os.environ.get("OXQ_EV", "/root/.claude/uploads/"
                    "e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
                    "443008ba-long_platform_breakouts1.csv")
HOLD = (5, 20, 60, 120, 250)
COMBOS = ((5, 0.00), (5, 0.03), (10, 0.00), (10, 0.03))
NSEED, JUDGE_H = 200, 60
TARGETS = (("601318", "2017-04-26"), ("688256", "2023-02-07"),
           ("688347", "2025-07-24"))


def ann(r, h):
    return (1.0 + r) ** (250.0 / h) - 1.0 if r > -1 else np.nan


def main():  # noqa: PLR0915
    t0 = time.time()
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
    assert (nt, ns) == (3316, 5232), f"锚点I1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点I1a 末日 {idx[-1].date()}"
    print(f"锚点I1a ✓ {(nt, ns)} 末日 {idx[-1].date()}", flush=True)

    mv = {}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        mv[c] = x["float_mv"]
    mvm = pd.DataFrame(mv).sort_index().reindex(index=idx, columns=codes).to_numpy()
    ind, _, _ = build_industry(codes, idx)

    e = pd.read_csv(EV, encoding="utf-8-sig", dtype={"code": str})
    e["code"] = e["code"].str.zfill(6)
    e["breakout_date"] = pd.to_datetime(e["breakout_date"])
    pos = {c: j for j, c in enumerate(codes)}
    ipos = pd.Index(idx)
    e["j"] = e["code"].map(pos).fillna(-1).astype(int)
    e["t"] = ipos.get_indexer(e["breakout_date"])
    mapped = (e["j"] >= 0) & (e["t"] >= 0)
    rate = float(mapped.mean())
    print(f"锚点I1b 事件映射率 {rate:.1%} {'✓' if rate >= .90 else '✗ 作废'}", flush=True)
    if rate < 0.90:
        return
    # 需要:确认窗口 max(N) + 最长持有 250
    e = e[mapped & (e["t"] < nt - max(HOLD) - max(n for n, _ in COMBOS))].copy()
    tt, jj = e["t"].to_numpy(), e["j"].to_numpy()
    # 上沿换算到我的面板:用他的「上沿/收盘」比值,与复权基准无关
    ratio = (e["upper_before_breakout"].to_numpy(float)
             / e["breakout_close"].to_numpy(float))
    upper = cl[tt, jj] * ratio
    print(f"可用事件 {len(e):,} 个 ({time.time()-t0:.0f}s)", flush=True)

    def controls(ct):
        rng = np.random.default_rng(SEED)
        out = np.full((NSEED, len(ct)), -1, np.int64)
        viol = 0
        cache = {}
        for k, (t, j) in enumerate(zip(ct, jj, strict=True)):
            if t < 0 or t >= nt:
                continue
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

    # 时间切分:训练段只报数,留出段判据(第一七五节补充登记)
    split = int(np.searchsorted(idx.values, np.datetime64("2022-01-01")))
    segs = (("训练段13-21", tt < split), ("留出段22-26", tt >= split))
    rows, w = [], 104
    print(f"\n{'='*w}\nI2/I3 回踩确认(纯结构,不加 RPS 门槛)\n{'='*w}")
    print(f"时间切分点 {idx[split].date()};训练段 {int((tt < split).sum()):,} 事件、"
          f"留出段 {int((tt >= split).sum()):,} 事件")
    print(f"{'段':<12}{'确认':<12}{'事件':>8}{'持有':>5}{'事件收益':>10}"
          f"{'对照中位':>10}{'超额pp':>9}{'年化超额pp':>12}{'p':>8}")
    for n, k in COMBOS:
        # 确认:突破后 n 日内最低收盘 ≥ 上沿×(1−k)
        lo_n = np.full(len(e), np.nan)
        for i in range(len(e)):
            t, j = tt[i], jj[i]
            seg = cl[t + 1:t + 1 + n, j]
            lo_n[i] = np.nanmin(seg) if np.isfinite(seg).any() else np.nan
        okc = np.isfinite(lo_n) & (lo_n >= upper * (1.0 - k))
        ct = tt + n                       # 确认日 = 突破日 + n
        good = okc & (ct < nt - max(HOLD))
        cr = float(okc.mean())
        cs, viol = controls(np.where(good, ct, -1))
        if n == COMBOS[0][0] and k == COMBOS[0][1]:
            print(f"  锚点I1c 抽样违例 {viol} 个 {'✓' if viol == 0 else '✗'}")
        for segname, segm in segs:
          for h in HOLD:
            m = good & (cs[0] >= 0) & segm
            p0 = cl[np.clip(ct, 0, nt - 1), jj]
            p1 = cl[np.clip(ct + h, 0, nt - 1), jj]
            with np.errstate(all="ignore"):
                r = p1 / np.where(p0 > 0, p0, np.nan) - 1.0
            m = m & np.isfinite(r)
            if m.sum() < 30:
                continue
            a = float(np.nanmean(r[m]))
            cm = np.full(NSEED, np.nan)
            for s in range(NSEED):
                ci = cs[s][m]
                cp0, cp1 = cl[ct[m], ci], cl[np.clip(ct[m] + h, 0, nt - 1), ci]
                with np.errstate(all="ignore"):
                    cm[s] = np.nanmean(cp1 / np.where(cp0 > 0, cp0, np.nan) - 1.0)
            med = float(np.nanmedian(cm))
            pv = float((np.sum(cm >= a) + 1) / (NSEED + 1))
            rec = {"段": segname, "确认": f"N={n} k={k:.0%}", "确认率": cr,
                   "n": int(m.sum()), "持有": h, "事件收益": a, "对照中位": med,
                   "超额pp": (a - med) * 100,
                   "年化超额pp": (ann(a, h) - ann(med, h)) * 100, "p": pv}
            rows.append(rec)
            print(f"{segname:<12}{rec['确认']:<12}{rec['n']:>8,}{h:>5}"
                  f"{a:>+10.2%}{med:>+10.2%}{rec['超额pp']:>+9.2f}"
                  f"{rec['年化超额pp']:>+12.2f}{pv:>8.4f}")
        # I3(b) 未通过确认的那批
        bad = (~okc) & (tt + n < nt - JUDGE_H)
        if bad.sum() >= 30:
            p0 = cl[np.clip(tt + n, 0, nt - 1), jj]
            p1 = cl[np.clip(tt + n + JUDGE_H, 0, nt - 1), jj]
            with np.errstate(all="ignore"):
                rb = p1 / np.where(p0 > 0, p0, np.nan) - 1.0
            print(f"    I3(b) 未通过确认的 {int(bad.sum()):,} 个,"
                  f"{JUDGE_H} 日收益 {np.nanmean(rb[bad]):+.2%}")
        # I3(c) 三只目标股(只作描述)
        tg = []
        for c_, d_ in TARGETS:
            i = np.flatnonzero((e["code"].to_numpy() == c_)
                               & (e["breakout_date"].astype(str).to_numpy() == d_))
            tg.append(f"{c_}{'✓' if len(i) and okc[i[0]] else '✗'}")
        print(f"    I3(c) 三只目标股(描述,不作判据):{' '.join(tg)}")

    d = pd.DataFrame(rows)
    j = d[(d["持有"] == JUDGE_H) & (d["段"] == "留出段22-26")]
    print(f"\n{'='*w}\nJ2 主判据(**留出段 2022-2026**、{JUDGE_H} 日持有,四个组合)\n{'='*w}")
    npass = 0
    for _, r in j.iterrows():
        c1, c2 = r["年化超额pp"] >= 3.00, r["p"] < 0.05
        npass += int(c1 and c2)
        print(f"  {r['确认']:<12} 年化超额 {r['年化超额pp']:+7.2f}pp "
              f"(≥+3.00 {'✓' if c1 else '✗'});p {r['p']:.4f} "
              f"(<0.05 {'✓' if c2 else '✗'}) → {'通过' if c1 and c2 else '不通过'}")
    print(f"\n  **J2(留出段):四个组合中通过 {npass} 个 → "
          f"{'通过' if npass else '不通过'}**")
    if npass:
        print("  **Bonferroni 复判(α = 0.05/4 = 0.0125,4 选 1 的 best-of-N):**")
        for _, r in j.iterrows():
            if r["年化超额pp"] >= 3.00:
                print(f"    {r['确认']:<12} p {r['p']:.4f} "
                      f"{'仍通过' if r['p'] < 0.0125 else '**不通过**'}")
    d.to_csv(f"{OUT}/longplat_pullback_oos.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/longplat_pullback.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
