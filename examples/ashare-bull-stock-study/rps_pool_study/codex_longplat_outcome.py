"""第一七三节 事前登记:给 Codex 的「长期平台突破」补上收益后验(结果未跑)。

起因
----
Codex 2026-09-01 交来《长期平台突破:全市场历史筛选研究》:
放弃 RPS90 硬锚点、深度改用收盘价且放宽到 40%、量比/波幅比上限放到 1.20、
平台窗口在 60/90/120/150/180/220/250 日里择长,停牌冻结、复牌续比。
全市场扫出 **34,737 个事件 / 4,843 只**(2013-07-15 → 2026-08-20),
RPS50≥80 分层后 **17,204 个 / 4,552 只**。

**他自己在「研究限制」里点明了两条要害,我认同并照抄:**
  #1「只统计历史形态出现次数,没有计算突破后的收益率、胜率、最大回撤和持有期表现」;
  #5「1.2 的成交量比、波幅比上限是为了覆盖中国平安样本后的研究参数,
      必须通过样本外测试,不能直接视为最优参数」。
他在第九节第 6 条明确请我设计「点时无未来数据的收益率回测」。

**本节只做一件事:拿他的事件清单,在我的面板上量前向收益与对照。**
分工:**他出事件,我出后验。** 本节不改他的规则、不重算他的事件、不调他的参数。

必须先声明的一件事(best-of-N)
------------------------------
他的两处阈值(深度 40%、量比/波幅比 1.20、RPS50≥80)都是**在看过中国平安与华虹
两个案例之后定的**,他自己写明了。所以本节属于典型的 best-of-N 情形,
**判据只能加严不能放宽,且无论结果如何照实写**。

口径
----
- 事件源:`long_platform_breakouts.csv`(34,737 行),取 `code` + `breakout_date`;
- 面板 (3316, 5232),末日 2026-08-28;退市股按最后有效价 ffill 参与(用户规则5);
- **收益一律用比值**(`close[t+h]/close[t] − 1`),与前复权基准无关 ——
  他的缓存是 2026-08-20 锚定的前复权,我的是 2026-08-28,绝对价不可比、比值可比;
- 持有期 **5 / 20 / 60 / 120 / 250** 个交易日(他第九节第 6 条点名的四个,加 250);
- 只保留 `breakout_date` 在我面板内、且往后至少还有 250 个交易日的事件。

对照(用户规则4,不可省)
------------------------
每个事件在**同一天**,从**同市值名次 ±25 且同申万一级行业**的合格股里随机抽 1 只,
走**完全相同的持有期**;**200 组种子**(p 下限 1/201 = 0.00498)。
比的是「事件组各持有期的等权平均收益」与「对照组同一统计量」的分布。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
H1 锚点(不过则本节作废)
   (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) **两个样本的比值复现**:他给的
       601318 / 2017-04-26:突破收盘 25.9427、上沿 25.7986 → 比值 **1.00559**;
       688347 / 2025-07-24:突破收盘 58.0110、上沿 56.4311 → 比值 **1.02800**。
       **比值与复权基准无关**,我用自己的面板算这两天的
       `收盘 / 该事件 [anchor_date, breakout_date) 窗口内的最高价`,须落在 **±0.5%** 内。
       —— 绝对价不设锚点,因为两边前复权锚定日不同,对不上是应该的。

       **【更正】首跑我把上沿写成了「最高收盘」,两个样本分别差 +0.53% / +1.58%,不过。**
       **不过的原因是我读错了他的规则,不是数据对不上**:他 §3.2 的深度用收盘价
       (避免单日影线放大),而 §3.5 的上沿用**最高价** —— 两处是分开的,我把前者
       套到了后者上。改用最高价后 601318 = 1.00543(差 −0.02%)、
       688347 = 1.02800(差 −0.00%),两个精确复现。
       **目标值 1.00559 / 1.02800 与容差 ±0.5% 一个字未动,改的只是我对他规则的误读。**
   (c) 事件映射率 ≥ 90%:他的 34,737 个事件里能落到我面板上的比例,
       低于 90% 说明两边股票池差太多,结论不可比,作废。

H2 **主判据**(60 日持有期,RPS50≥80 分层,零成本)
   **通过 ⟺ 相对同市值同行业对照的超额 ≥ +3.00pp(年化口径)且单尾 p < 0.05。**
   **与第一五二/一五三/一五四/一五五/一六八节同一道门槛,一个字不放宽。**
   (选 60 日与 RPS50≥80,是因为这正是他建议写进模板的那一档。)

H3 **分层描述(必报,不参与判定)**
   5/20/60/120/250 五个持有期 × 四个 RPS 分层(无门槛 / ≥50 / ≥80 / ≥90)
   的事件数、等权平均收益、对照中位、超额、p 值;
   另按平台窗口(60~250 日)与年份拆分。

H4 **回答他第九节点名的问题**(描述项)
   (1) 收盘深度 vs 极值深度:两种深度在同一批事件上的分布差多少;
   (2) 量比/波幅比 1.20 是否过宽:把门槛降到 1.0 / 0.9 / 0.8,事件数与超额怎么变;
   (3) 事件去重:同一只股票相邻事件的间隔分布;
   (4) 生存者偏差:他的事件股票里有多少只在我面板上是已退市的。

**判据写法自律**:绝对阈值,不写比值判据(第一五四节 A3 的教训)。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。**只登记判据。**

不做的
------
不改他的规则、不重算他的事件、不调他的参数;不改 `src/oxq/`;
不调持有期 / 对照口径 / 判据门槛;**跑完不许回头改阈值再跑**;
不新增顶层目录;不 force push;**不往 quant-research-dev / etf-netflow-dev 推**;
**不作任何可交易性声明。若 H2 不过,如实写「他这套长期平台突破也没做到」。**
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
EV = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
      "999d097e-long_platform_breakouts.csv")
HOLD = (5, 20, 60, 120, 250)
NSEED, JUDGE_H, JUDGE_RPS = 200, 60, 80


def ann(r, h):
    """把持有期收益折成年化(250 交易日)。"""
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
    assert (nt, ns) == (3316, 5232), f"锚点H1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点H1a 末日 {idx[-1].date()}"
    print(f"锚点H1a ✓ {(nt, ns)} 末日 {idx[-1].date()}", flush=True)

    mv = {}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        mv[c] = x["float_mv"]
    mvm = pd.DataFrame(mv).sort_index().reindex(index=idx, columns=codes).to_numpy()
    ind, _, _ = build_industry(codes, idx)
    print(f"市值/行业就绪 ({time.time()-t0:.0f}s)", flush=True)

    e = pd.read_csv(EV, encoding="utf-8-sig", dtype={"code": str})
    e["code"] = e["code"].str.zfill(6)
    e["breakout_date"] = pd.to_datetime(e["breakout_date"])
    print(f"Codex 事件 {len(e):,} 个,{e['code'].nunique():,} 只", flush=True)

    pos = {c: j for j, c in enumerate(codes)}
    ipos = pd.Index(idx)
    e["j"] = e["code"].map(pos).fillna(-1).astype(int)
    e["t"] = ipos.get_indexer(e["breakout_date"])
    mapped = (e["j"] >= 0) & (e["t"] >= 0)
    rate = float(mapped.mean())
    print(f"锚点H1c 事件映射率 {rate:.1%}(要求 ≥90%)"
          f" {'✓' if rate >= 0.90 else '✗ 本节作废'}", flush=True)
    if rate < 0.90:
        return
    e = e[mapped & (e["t"] < nt - max(HOLD))].copy()
    print(f"  留足 {max(HOLD)} 日前瞻后 {len(e):,} 个事件", flush=True)

    # ---- 锚点 H1b:两个样本的比值 ----
    ok_b, det = True, []
    for code, dt, exp in (("601318", "2017-04-26", 1.00559),
                          ("688347", "2025-07-24", 1.02800)):
        row = e[(e["code"] == code) & (e["breakout_date"] == dt)]
        if not len(row):
            det.append(f"{code}/{dt} 不在可用事件里")
            ok_b = False
            continue
        r = row.iloc[0]
        t, j = int(r["t"]), int(r["j"])
        # 上沿 = [anchor_date, breakout_date) 窗口内的**最高价**(他 §3.5),
        # 不是最高收盘(那是他 §3.2 深度的口径)。只有这两只需要读 high。
        hx = pd.read_parquet(f"{DATA}/{code}.parquet", columns=["high", "close"])
        hx.index = pd.to_datetime(hx.index).tz_localize(None)
        hx = hx[hx["close"] > 0]
        seg = hx.loc[str(r["anchor_date"])[:10]:dt].iloc[:-1]
        up = float(seg["high"].max())
        got = float(hx.loc[dt, "close"]) / up
        good = abs(got / exp - 1.0) <= 0.005
        ok_b &= good
        det.append(f"{code}/{dt}:我 {got:.5f} vs 他 {exp:.5f} "
                   f"({'✓' if good else '✗'})")
    print("锚点H1b " + ";".join(det) + (" ✓" if ok_b else " ✗ 本节作废"), flush=True)
    if not ok_b:
        return

    # ---- 前向收益 ----
    tt, jj = e["t"].to_numpy(), e["j"].to_numpy()
    p0 = cl[tt, jj]
    fwd = {}
    for h in HOLD:
        fwd[h] = cl[np.minimum(tt + h, nt - 1), jj] / np.where(p0 > 0, p0, np.nan) - 1.0
        e[f"ret{h}"] = fwd[h]
    print(f"前向收益就绪 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 对照:同市值名次 ±25 且同申万一级 ----
    rng = np.random.default_rng(SEED)
    days = np.unique(tt)
    pick = {}
    for day in days:
        el = np.flatnonzero(okm[day] & np.isfinite(mvm[day]) & (ind[day] >= 0))
        if not len(el):
            continue
        o = el[np.argsort(mvm[day, el], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        pick[day] = (o, rk)
    ctrl = np.full((NSEED, len(e)), -1, np.int64)
    viol = 0
    for k, (t, j) in enumerate(zip(tt, jj, strict=True)):
        if t not in pick:
            continue
        o, rk = pick[t]
        p_, i0 = rk[j], ind[t, j]
        if p_ < 0:
            continue
        lo, hi = max(0, p_ - NBR), min(len(o), p_ + NBR + 1)
        cand = o[lo:hi]
        cand = cand[(ind[t, cand] == i0) & (cand != j)]
        if not len(cand):
            continue
        ctrl[:, k] = rng.choice(cand, NSEED, replace=True)
        viol += int(np.any(np.abs(rk[ctrl[:, k]] - p_) > NBR))
    print(f"锚点H1d 抽样市值名次偏离 >{NBR} 的违例 {viol} 个 "
          f"{'✓' if viol == 0 else '✗'};有对照的事件 "
          f"{int((ctrl[0] >= 0).sum()):,}/{len(e):,} ({time.time()-t0:.0f}s)", flush=True)

    def stats(mask, h):
        m = mask & np.isfinite(e[f"ret{h}"].to_numpy()) & (ctrl[0] >= 0)
        if m.sum() < 30:
            return None
        a = float(np.nanmean(e[f"ret{h}"].to_numpy()[m]))
        cm = np.full(NSEED, np.nan)
        ti_, ci_ = tt[m], ctrl[:, m]
        for s in range(NSEED):
            cp0 = cl[ti_, ci_[s]]
            cp1 = cl[np.minimum(ti_ + h, nt - 1), ci_[s]]
            with np.errstate(all="ignore"):
                cm[s] = np.nanmean(cp1 / np.where(cp0 > 0, cp0, np.nan) - 1.0)
        med = float(np.nanmedian(cm))
        pv = float((np.sum(cm >= a) + 1) / (NSEED + 1))
        return {"n": int(m.sum()), "事件收益": a, "对照中位": med,
                "超额pp": (a - med) * 100, "年化超额pp": (ann(a, h) - ann(med, h)) * 100,
                "p": pv}

    rows = []
    rp = e["breakout_rps50"].to_numpy()
    layers = [("无门槛", np.ones(len(e), bool)), ("RPS50≥50", rp >= 50),
              ("RPS50≥80", rp >= 80), ("RPS50≥90", rp >= 90)]
    w = 100
    print(f"\n{'='*w}\nH3 分层 × 持有期\n{'='*w}")
    print(f"{'分层':<10}{'持有':>5}{'事件':>8}{'事件收益':>10}{'对照中位':>10}"
          f"{'超额pp':>9}{'年化超额pp':>12}{'p':>8}")
    for lab, msk in layers:
        for h in HOLD:
            r = stats(msk, h)
            if r is None:
                continue
            rows.append({"分层": lab, "持有期": h, **r})
            print(f"{lab:<10}{h:>5}{r['n']:>8,}{r['事件收益']:>+10.2%}"
                  f"{r['对照中位']:>+10.2%}{r['超额pp']:>+9.2f}"
                  f"{r['年化超额pp']:>+12.2f}{r['p']:>8.4f}")
    j2 = [r for r in rows if r["分层"] == f"RPS50≥{JUDGE_RPS}"
          and r["持有期"] == JUDGE_H]
    print(f"\n{'='*w}\nH2 主判据(RPS50≥{JUDGE_RPS}、{JUDGE_H} 日持有)\n{'='*w}")
    if j2:
        r = j2[0]
        c1, c2 = r["年化超额pp"] >= 3.00, r["p"] < 0.05
        print(f"  年化超额 {r['年化超额pp']:+.2f}pp(≥+3.00 {'✓' if c1 else '✗'});"
              f"单尾 p {r['p']:.4f}(<0.05 {'✓' if c2 else '✗'})"
              f" → **{'通过' if c1 and c2 else '不通过'}**")
    pd.DataFrame(rows).to_csv(f"{OUT}/codex_longplat_outcome.csv", index=False,
                              encoding="utf-8-sig")
    e.drop(columns=["j"]).to_csv(f"{OUT}/codex_longplat_events.csv", index=False,
                                 encoding="utf-8-sig")
    print(f"\n落库 {OUT}/codex_longplat_outcome.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
