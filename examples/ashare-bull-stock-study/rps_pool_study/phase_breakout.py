"""第一七六节 B 部分 事前登记:平台突破为什么在不同阶段成 / 败(结果未跑)。

起因
----
用户给了两只票的周线图:金发科技 600143、卧龙电驱 600580。
2025 年 6-8 月两只都在 20 周线附近盘出平台后向上突破,**26 周 +67% / +80%**;
2025 年 12 月两只又在 20 周线附近盘了一次、也往上弹了,**但都没走出来**。
问题:同一个形态,为什么阶段不同结局相反?怎么量化?

A 部分(`phase_cases.py`)已经把事实摆出来了,并且**推翻了我原本的想法**:
12 月那两次根本**没有构成「突破」** —— 两只票在 2025-11-01 → 2026-02-28 窗口内
**一周都没有创过 20 周周收盘新高**,离 52 周高点始终差 2%~9%。
6 月那两次则是**突破当周就同时刷新 52 周高点**(距 52 周高 −4.1% / +0.0%),
且**突破周成交量是 52 周均量的 4.45 倍 / 2.51 倍**;
12 月那两次的周量比只有 0.88~1.09 倍。

所以本节要检验的不是「玄学阶段」,而是**三条能在突破当周就读出来的量化差别**。

事件定义(跑之前写死,零可调阈值)
----------------------------------
在**周线**上(日线按 W-FRI 聚合,取每周最后一个交易日的收盘;停牌周按最后有效价
ffill 参与,**绝不剔除**,用户规则 5):

    事件 ⟺ 本周周收盘 > 前 20 周周收盘最高值(首次上穿,连续周不重复计数)
           且 本周周收盘 ≥ 20 周线
           且 该股在突破日(= 该周最后一个交易日)可交易(非 ST、非停牌、
              上市 ≥ 250 日、有成交)

**没有量比阈值、没有深度阈值、没有 RPS 门槛** —— 「前 20 周没创过新高」本身
就已经是平台的定义。这是为了避开第一七四节的教训(**用结果去标定设定特征**)。

三条分组轴(全部只用突破日及之前的信息)
----------------------------------------
S1 **个股结构**:突破当周是否**同时**刷新 52 周周收盘高点(二分)。
S2 **市场阶段**:突破日全市场广度 = 可交易股票中「收盘 ≥ 自身 MA100」的占比,五分位。
S3 **量能**:突破周成交量 ÷ 前 52 周周成交量均值,五分位。

**S2/S3 的分位边界只用训练段(2013→2021)估计,再套到留出段** —— 这是加严,
不是放宽:留出段不许看自己的分布。

口径
----
- 面板 `/home/user/oxq-panel-0828/oxq_stock_market_fixed`,(3316, 5232),末日 2026-08-28;
- 持有期 5 / 20 / 60 / 120 / 250 **交易日**,从突破日起算;**判据横期 60 日**
  (与第一五二/一五五/一六八/一七三/一七四节一致,不换标尺);
- 对照:**同一突破日**、同市值名次 ±25、同申万一级行业随机抽 1 只,
  200 组种子(p 下限 1/201 = 0.00498)—— 用户规则 4;
- 时间切分:**训练段 2013→2021-12 只报数,留出段 2022-01-01 起判据**(第一七五节起的标准)。

**注意一个必须先讲明的性质**:对照是**同日**抽的,所以「市场好 → 大家都涨」
这一层会被差掉。S2 轴上的**超额**回答的是更硬的问题 ——
**平台突破相对同日同类股票的优势,是不是也随市场阶段变**。
绝对收益另表报告(L5a),两张表要一起读。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
L1 锚点(不过则本节作废)
   (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) 600143 的 2025-07-25、600580 的 2025-08-08 **必须在事件集中**;
   (c) 两只票在 2025-11-01 → 2026-02-28 **必须无事件**(与 A 部分一致);
   (d) 对照抽样市值名次偏离 > 25 的违例 = 0。

L2 **主判据 S1**(留出段,60 日持有)
   通过 ⟺ 「突破同时创 52 周新高」组年化超额 ≥ **+3.00pp** 且单尾 p < 0.05,
          **且** 该组年化超额 − 「未创 52 周新高」组年化超额 ≥ **+3.00pp**。
   两条都要满足才算过。

L3 **S2 市场广度**(留出段,60 日)
   通过 ⟺ 五分位年化超额的 Spearman ρ ≥ **+0.60**,且 Q5 年化超额 ≥ **+3.00pp**
          且 Q5 单尾 p < 0.05。

L4 **S3 突破周量比**(留出段,60 日)
   通过 ⟺ 判据同 L3(ρ ≥ +0.60、Q5 ≥ +3.00pp、p < 0.05)。

L5 描述(必报,**不参与判定**)
   (a) 各广度分位的**绝对**收益 —— 量一量「市场阶段决定绝对收益」到底有多大;
   (b) 训练段同表,只报数;
   (c) 两只样本股的 4 个事件在三条轴上的取值与百分位 ——
       **防止用结果标定设定**(第一七四节的教训),它们只是描述,不参与任何阈值。

L6 多重比较:S1/S2/S3 三条并列。**若只有一条过,按 Bonferroni 用 α = 0.05/3 =
   0.0167 复判一次,两个结论都写。**

事前预测
--------
自第一一九节起本项目不再对「样本外会不会过」下事前预测,只登记判据。
但有一条**可被证伪的机制预测**必须登记:
   **S1 会过、S2 不会过。** 理由:同日对照会把市场层面的涨跌差掉,
   而「是否同时创 52 周新高」是个股层面的结构差异,差不掉。
   **如果反过来(S2 过而 S1 不过),我在正文里明说我错了。**

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
HOLD = (5, 20, 60, 120, 250)
NSEED, JUDGE_H, NQ = 200, 60, 5
CASES = {("600143", "2025-07-25"), ("600580", "2025-08-08")}


def ann(r, h):
    return (1.0 + r) ** (250.0 / h) - 1.0 if r > -1 else np.nan


def spearman5(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).all():
        return np.nan
    return float(pd.Series(v).corr(pd.Series(np.arange(len(v)) + 1.0),
                                   method="spearman"))


def main():  # noqa: PLR0912, PLR0915
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
        return {"idx": idx_.values.astype("datetime64[ns]"), "cl": cl_,
                "okm": okm_, "vol": al_("volume", 0.0)}
    p = cached("panel", DATA, _build_panel)
    idx = pd.DatetimeIndex(p["idx"])
    cl, okm, vol = p["cl"], p["okm"], p["vol"]
    nt, ns = cl.shape
    assert (nt, ns) == (3316, 5232), f"锚点L1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点L1a 末日 {idx[-1].date()}"
    print(f"锚点L1a ✓ {(nt, ns)} 末日 {idx[-1].date()}", flush=True)

    def _build_mv():
        mv = {}
        for c in codes:
            x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["float_mv"])
            if getattr(x.index, "tz", None) is not None:
                x.index = x.index.tz_localize(None)
            mv[c] = x["float_mv"]
        return {"mv": pd.DataFrame(mv).sort_index().reindex(
            index=idx, columns=codes).to_numpy()}
    mvm = cached("mv", DATA, _build_mv)["mv"]
    ind, _, _ = build_industry(codes, idx)

    # ---------- 周线 ----------
    wk = pd.Series(np.arange(nt), index=idx).resample("W-FRI").last().dropna()
    wsel = wk.to_numpy().astype(int)
    wdates = idx[wsel]
    nw = len(wsel)
    wc = cl[wsel]                                  # (nw, ns) 周收盘
    starts = np.concatenate([[0], wsel[:-1] + 1])
    vcs = np.vstack([np.zeros((1, ns)), np.cumsum(np.nan_to_num(vol), axis=0)])
    wv = vcs[wsel + 1] - vcs[starts]               # 周成交量合计
    wdf = pd.DataFrame(wc)
    ma20w = wdf.rolling(20).mean().to_numpy()
    up20 = wdf.shift(1).rolling(20).max().to_numpy()   # 前 20 周(不含本周)最高周收
    hi52 = wdf.shift(1).rolling(52).max().to_numpy()   # 前 52 周(不含本周)最高周收
    vbase = pd.DataFrame(wv).shift(1).rolling(52).mean().to_numpy()
    print(f"周线 {nw} 周 {wdates[0].date()} → {wdates[-1].date()} "
          f"({time.time()-t0:.0f}s)", flush=True)

    cross = wc > up20
    prev = np.vstack([np.zeros((1, ns), bool), cross[:-1]])
    evm = (cross & ~prev & (wc >= ma20w)
           & np.isfinite(up20) & np.isfinite(ma20w) & np.isfinite(hi52)
           & np.isfinite(vbase) & (vbase > 0) & okm[wsel])
    ww, jj = np.nonzero(evm)
    tt = wsel[ww]
    keep = tt < nt - max(HOLD)
    ww, jj, tt = ww[keep], jj[keep], tt[keep]
    ne = len(tt)
    print(f"事件 {ne:,} 个(可跑满 {max(HOLD)} 日)", flush=True)

    ecodes = np.array(codes, object)[jj]
    edates = np.array([str(d.date()) for d in idx[tt]], object)
    s1 = wc[ww, jj] >= hi52[ww, jj]                       # 同时创 52 周新高
    s3 = wv[ww, jj] / vbase[ww, jj]                       # 突破周量比
    d52 = wc[ww, jj] / hi52[ww, jj] - 1.0                 # 距 52 周高(描述)

    # 市场广度:收盘 ≥ 自身 MA100 的可交易股票占比
    ma100 = pd.DataFrame(cl).rolling(100).mean().to_numpy()
    with np.errstate(all="ignore"):
        above = (cl >= ma100) & okm & np.isfinite(ma100)
    breadth = above.sum(1) / np.maximum(okm.sum(1), 1)
    s2 = breadth[tt]

    # 锚点 L1b / L1c
    hit = {(c, d) for c, d in zip(ecodes, edates, strict=True)} & CASES
    print(f"锚点L1b 两只样本的 6 月事件在集中:{sorted(hit)} "
          f"{'✓' if hit == CASES else '✗ 作废'}", flush=True)
    if hit != CASES:
        return
    m12 = np.array([(c in ("600143", "600580")) and ("2025-11-01" <= d <= "2026-02-28")
                    for c, d in zip(ecodes, edates, strict=True)])
    print(f"锚点L1c 两只票 2025-11→2026-02 事件数 {int(m12.sum())} "
          f"{'✓' if m12.sum() == 0 else '✗ 作废'}", flush=True)
    if m12.sum():
        return

    # ---------- 对照 ----------
    def controls(ct):
        rng = np.random.default_rng(SEED)
        out = np.full((NSEED, len(ct)), -1, np.int32)
        viol, cache = 0, {}
        for k, (t, j) in enumerate(zip(ct, jj, strict=True)):
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
    print(f"锚点L1d 抽样违例 {viol} 个 {'✓' if viol == 0 else '✗ 作废'} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if viol:
        return
    has = cs[0] >= 0

    split = int(np.searchsorted(idx.values, np.datetime64("2022-01-01")))
    trm, hom = tt < split, tt >= split
    segs = (("训练段13-21", trm), ("留出段22-26", hom))
    print(f"时间切分 {idx[split].date()};训练 {int(trm.sum()):,} / "
          f"留出 {int(hom.sum()):,}", flush=True)

    # 分位边界只用训练段估计,套到留出段(加严)
    def qbin(v, name):
        ref = v[trm & np.isfinite(v)]
        q = np.quantile(ref, np.linspace(0, 1, NQ + 1)[1:-1])
        print(f"  {name} 训练段分位边界 " + " ".join(f"{x:.4g}" for x in q))
        return np.digitize(v, q)

    print("\n分位边界(只用训练段):")
    g2, g3 = qbin(s2, "S2 市场广度"), qbin(s3, "S3 突破周量比")

    def stat(mask, h):
        m = mask & has & (tt < nt - h)
        p0, p1 = cl[tt, jj], cl[np.clip(tt + h, 0, nt - 1), jj]
        with np.errstate(all="ignore"):
            r = p1 / np.where(p0 > 0, p0, np.nan) - 1.0
        m = m & np.isfinite(r)
        if m.sum() < 30:
            return None
        a = float(np.nanmean(r[m]))
        cm = np.empty(NSEED)
        for s in range(NSEED):
            ci = cs[s][m]
            cp0, cp1 = cl[tt[m], ci], cl[np.clip(tt[m] + h, 0, nt - 1), ci]
            with np.errstate(all="ignore"):
                cm[s] = np.nanmean(cp1 / np.where(cp0 > 0, cp0, np.nan) - 1.0)
        med = float(np.nanmedian(cm))
        return {"n": int(m.sum()), "事件收益": a, "对照中位": med,
                "超额pp": (a - med) * 100,
                "年化超额pp": (ann(a, h) - ann(med, h)) * 100,
                "p": float((np.sum(cm >= a) + 1) / (NSEED + 1))}

    rows, wdt = [], 108
    def block(axis, groups, labels):
        print(f"\n{'='*wdt}\n{axis}\n{'='*wdt}")
        print(f"{'段':<12}{'组':<22}{'事件':>8}{'持有':>5}{'事件收益':>10}"
              f"{'对照中位':>10}{'超额pp':>9}{'年化超额pp':>12}{'p':>8}")
        for sn, sm in segs:
            for gi, gl in enumerate(labels):
                gm = sm & (groups == gi)
                for h in HOLD:
                    st = stat(gm, h)
                    if st is None:
                        continue
                    st |= {"轴": axis, "段": sn, "组": gl, "持有": h, "组序": gi}
                    rows.append(st)
                    print(f"{sn:<12}{gl:<22}{st['n']:>8,}{h:>5}"
                          f"{st['事件收益']:>+10.2%}{st['对照中位']:>+10.2%}"
                          f"{st['超额pp']:>+9.2f}{st['年化超额pp']:>+12.2f}"
                          f"{st['p']:>8.4f}")

    block("S1 是否同时创52周新高", s1.astype(int),
          ["未创52周新高", "同时创52周新高"])
    block("S2 市场广度五分位", g2, [f"Q{i+1} 广度" for i in range(NQ)])
    block("S3 突破周量比五分位", g3, [f"Q{i+1} 量比" for i in range(NQ)])

    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/phase_breakout.csv", index=False, encoding="utf-8-sig")

    def pick(axis, seg, gi, h=JUDGE_H):
        z = d[(d["轴"] == axis) & (d["段"] == seg) & (d["组序"] == gi)
              & (d["持有"] == h)]
        return z.iloc[0] if len(z) else None

    ho = "留出段22-26"
    print(f"\n{'='*wdt}\n判定(留出段、{JUDGE_H} 日持有,门槛与前六节完全一致)\n{'='*wdt}")
    verdicts = {}

    a, b = pick("S1 是否同时创52周新高", ho, 1), pick("S1 是否同时创52周新高", ho, 0)
    if a is None or b is None:
        print("L2 S1:样本不足,不通过")
        verdicts["L2 S1"] = (False, np.nan)
    else:
        gap = a["年化超额pp"] - b["年化超额pp"]
        ok = bool(a["年化超额pp"] >= 3.0 and a["p"] < 0.05 and gap >= 3.0)
        print(f"L2 S1 主判据:创新高组 {a['年化超额pp']:+.2f}pp p={a['p']:.4f};"
              f"未创新高组 {b['年化超额pp']:+.2f}pp;差 {gap:+.2f}pp "
              f"→ {'✓ 通过' if ok else '✗ 不通过'}")
        verdicts["L2 S1"] = (ok, a["p"])

    for tag, axis in (("L3 S2", "S2 市场广度五分位"), ("L4 S3", "S3 突破周量比五分位")):
        v = [pick(axis, ho, i) for i in range(NQ)]
        if any(x is None for x in v):
            print(f"{tag}:样本不足,不通过")
            verdicts[tag] = (False, np.nan)
            continue
        ex = [float(x["年化超额pp"]) for x in v]
        rho, q5, pq5 = spearman5(ex), ex[-1], float(v[-1]["p"])
        ok = bool(rho >= 0.60 and q5 >= 3.0 and pq5 < 0.05)
        print(f"{tag} {axis}:ρ={rho:+.2f} Q5={q5:+.2f}pp p={pq5:.4f} "
              f"五分位 [{' '.join(f'{x:+.2f}' for x in ex)}] "
              f"→ {'✓ 通过' if ok else '✗ 不通过'}")
        verdicts[tag] = (ok, pq5)

    npass = sum(1 for ok, _ in verdicts.values() if ok)
    print(f"\nL6 三条轴通过 {npass} 条。", end="")
    if npass == 1:
        k = [x for x, (ok, _) in verdicts.items() if ok][0]
        pv = verdicts[k][1]
        print(f"唯一通过的是 {k},这是 3 选 1 的 best-of-N;"
              f"Bonferroni α=0.0167 复判:p={pv:.4f} "
              f"{'仍过' if pv < 0.0167 else '不过'}。")
    else:
        print("无需 Bonferroni 复判。" if npass != 1 else "")

    # ---------- L5 描述 ----------
    print(f"\n{'='*wdt}\nL5(a) 绝对收益 vs 市场广度五分位(留出段,{JUDGE_H} 日)"
          f" —— 对照是同日抽的,这一层在超额里被差掉了\n{'='*wdt}")
    p0 = cl[tt, jj]
    p1 = cl[np.clip(tt + JUDGE_H, 0, nt - 1), jj]
    with np.errstate(all="ignore"):
        r60 = p1 / np.where(p0 > 0, p0, np.nan) - 1.0
    for sn, sm in segs:
        line = []
        for gi in range(NQ):
            m = sm & (g2 == gi) & np.isfinite(r60) & (tt < nt - JUDGE_H)
            line.append(f"Q{gi+1} {np.nanmean(r60[m]):+.2%}({int(m.sum()):,})"
                        if m.sum() >= 30 else f"Q{gi+1} n/a")
        print(f"  {sn}  " + "  ".join(line))
        mm = sm & np.isfinite(r60) & (tt < nt - JUDGE_H)
        print(f"    该段广度中位 {np.median(s2[sm]):.1%};"
              f"全段事件 {JUDGE_H} 日均值 {np.nanmean(r60[mm]):+.2%}")

    print(f"\n{'='*wdt}\nL5(c) 两只样本股的事件位置(**只作描述,不参与任何阈值**)\n{'='*wdt}")
    cw = []
    for c_, d_ in sorted(CASES):
        i = int(np.flatnonzero((ecodes == c_) & (edates == d_))[0])
        cw.append({"code": c_, "突破日": d_,
                   "创52周新高": bool(s1[i]), "距52周高": round(float(d52[i]), 4),
                   "周量比": round(float(s3[i]), 2),
                   "量比百分位": round(float((s3 < s3[i]).mean()), 3),
                   "市场广度": round(float(s2[i]), 3),
                   "广度百分位": round(float((s2 < s2[i]).mean()), 3),
                   f"+{JUDGE_H}日": round(float(r60[i]), 4)})
    print(pd.DataFrame(cw).to_string(index=False))
    pd.DataFrame(cw).to_csv(f"{OUT}/phase_breakout_cases.csv", index=False,
                            encoding="utf-8-sig")
    print(f"\n落库 {OUT}/phase_breakout.csv、{OUT}/phase_breakout_cases.csv"
          f" ({time.time()-t0:.0f}s)")
    print("本表是状态记录,不是买点,不构成任何投资建议。")


if __name__ == "__main__":
    main()
