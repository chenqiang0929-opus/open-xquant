"""§154 事前登记:把欧奈尔/陶博士的第二三段加回去 —— 阶梯归因(结果未跑)。

起因
----
用户问:「为什么欧奈尔和陶博士的选股方法和买入方法会成功,到了你这边就全部失效了」

第一五三节末尾我给出的回答是:**我根本没测他们的方法。**
他们的方法有三段 ——(1)筛选 (2)买点 (3)卖出与仓位 ——
而我在第一五二/一五三节测的是「把第一段的输出全部等权买下来、持有一个月、
不止损、不择时、不空仓」,**这是他们两人都明确反对的用法**。
而且第一段我其实测**通过**了(Codex 强确认 lift 1.50、我的 1.39)。

**本节把第二三段加回去,做阶梯归因。**

阶梯(每一档只加一件事,跑前定死)
--------------------------------
    L0 基线    第一五二/一五三节原样:月末买入全部选中股,等权,持有到下月末,
               无止损、无择时。**用来复现第一五二节,当机器锚点。**
    L1 +集中度 只买信号里 RPS60 最高的 **10 只**(槽位制),持有上限 **120 个交易日**。
               ⚠️ **这一档同时改了两件事**(集中度 + 持有上限 21→120 日),
               因为「砍掉亏的、让赚的跑」在他们的方法里是一件事,拆不开。
               **归因时必须承认这一点,不许事后假装只改了集中度。**
    L2 +止损   L1 加**买入价 −8% 收盘触发即卖出**(欧奈尔 7–8% 铁律,取严的那端)。
    L3 +大盘过滤 L2 加:全市场等权净值跌破自身 MA200 → **当日清仓且不新开仓**
               (欧奈尔 M 条件的硬开关版)。
    L4 +突破日买入 L3 改为:月末选出的股票在**接下来那个月里首次创 60 日新高**
               的当日买入(整月未创新高则该月不买),而不是月末直接买。

**判据在 L4** —— 四条全加上的那一档。L0–L3 只报数,用于归因,**不参与判定**
(否则就是 best-of-N)。

关键设计:对照组走完全相同的风控
--------------------------------
对照 = 每次策略实际开仓的**那一天、那一个槽位**,换成**同市值名次 ±25、
同申万一级行业**的随机股,然后**走完全一样的止损、到期、清仓逻辑**,500 组种子。

**这样才能分清两件事:**
- 若加上风控后**超额转正** → 边在「选股 × 风控」的配合上;
- 若策略与对照**一起抬升、超额仍≈0** → **风控是普适的,对任何一篮子股票都一样有用,
  不构成选股的边** —— 那么欧奈尔/陶博士赚的是风控的钱,不是选股的钱。

口径(与第一五二/一五三节一致,一个字不改)
------------------------------------------
- 面板 (3297, 5232);合格:非 ST、非停牌、上市满 250 日、当日有成交
- 退市股按最后有效价 ffill 参与,**绝不剔除**
- 观察点 = 每月最后一个交易日;训练段 2019-01–2022-12 只报数;
  **留出段 2023-01–2026-04 判据在这里**
- 组合日收益 = 持仓个股日收益之和 ÷ **10**(空槽记 0,即现金,不赚不亏)
- 成本:判定用**零成本**口径;**双边合计 0.2%/次往返**只作描述

参数(跑前定死,跑完不调)
------------------------
    MAXPOS = 10       集中度(陶博士量级)
    STOP   = 8%       相对买入价,收盘触发
    HOLD_MAX = 120    交易日
    排序键 = RPS60 降序        大盘过滤 = 全市场等权净值 vs 自身 MA200
    突破 = 收盘价 > 前 60 个交易日最高收盘价
    NSEED = 500       NBR = ±25 市值名次

被判的规则(只判一条,避免 best-of-N)
------------------------------------
**主判据只判「Codex 强确认」(反弹≥40%、120日收益≥10%、MA20持续度≥55%、RPS60≥90)。**
理由:它是最像欧奈尔/陶博士的一条(RPS 主导 + 趋势 + 反弹),
而且它在第一五三节的组合口径上是**最差的一条(超额 −4.95pp)**。
**用最差的那条当主判据是加严,不是放宽** —— 若风控能把它救回来,证据最强。
**第一四八节规则全档只报数,不判定。**

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
A1 锚点(不过则本节作废)
   (a) 面板 (3297, 5232);
   (b) 价格 ffill 后首个有效价之后无空洞;
   (c) 行业恒等式违例 = 0;
   (d) **无前视**:止损用 (t, ...] 的收盘、大盘 MA200 只用 ≤t、
       突破日判定只用 ≤t、开仓价 = 开仓日收盘;逐点断言;
   (e) **机器锚点:L0 必须复现第一五二节留出段的零成本年化 +11.12%
       (第一四八节规则),容差 ±0.30pp。算不出或对不上 = 本节作废。**

A2 **主判据**(Codex 强确认,L4 档,留出段 2023-01–2026-04)
   **通过 ⟺ 零成本年化超额 ≥ +3.00pp(对照 500 组中位数)且单尾 p < 0.05。**
   两条同时满足。**与第一五二/一五三节同一道门槛,不因仓位少、方差大而放宽。**

A3 **归因判定**(描述性,同样跑前写死)
   记 Δ策略 = L4策略年化 − L0策略年化;Δ对照 = L4对照中位 − L0对照中位。
   **若 Δ对照 ≥ 0.5 × Δ策略,则如实写「风控是普适的,不构成选股的边」。**

A4 描述(不参与判定):各档年化/超额/p/最大回撤/平均持仓数/换手/止损触发次数/
   空仓天数占比;双边 0.2% 成本口径;逐年超额。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。

不做的
------
不调 MAXPOS / STOP / HOLD_MAX / 排序键 —— **跑完不许回头改参数再跑**;
不改被判规则;不加第五档;不新增顶层目录;不 force push;
**不往 quant-research-dev / etf-netflow-dev 推任何东西**;
不作任何可交易性声明。**若 A2 不过,如实写「加上风控也没做到」。**
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
from codex_r10_neutral import NBR, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from industry_neutral import build_industry  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
MAXPOS, STOP, HOLD_MAX, NSEED, COST = 10, 0.08, 120, 500, 0.002
NCAND, MA_MKT, NH = 40, 200, 60
Q148 = 0.70
TRAIN, HOLD = ("2019-01-01", "2022-12-31"), ("2023-01-01", "2026-04-30")


def ann(nav, nd):
    return float(nav ** (250.0 / nd) - 1.0)


def mdd(eq):
    pk = np.maximum.accumulate(eq)
    return float(np.max((pk - eq) / pk))


def sim(cand, nrep, ta, tb, cl, mkt_on, use_stop, use_mkt):
    """槽位制日频模拟。cand: {day: (nrep, ncand) int 数组};空槽记 0 收益(现金)。"""
    pj = np.full((nrep, MAXPOS), -1, np.int64)
    pe = np.zeros((nrep, MAXPOS), np.int64)
    ppx = np.zeros((nrep, MAXPOS))
    nd = tb - ta + 1
    ret = np.zeros((nrep, nd))
    cst = np.zeros((nrep, nd))
    nstop = np.zeros(nrep, np.int64)
    ntr = np.zeros(nrep, np.int64)
    nhold = np.zeros((nrep, nd))
    for i, t in enumerate(range(ta, tb + 1)):
        m = pj >= 0
        z = np.where(m, pj, 0)
        if m.any():
            r = np.where(m, cl[t, z] / cl[t - 1, z] - 1.0, 0.0)
            ret[:, i] = np.nan_to_num(r).sum(axis=1) / MAXPOS
        nhold[:, i] = m.sum(axis=1)
        ex = m & ((t - pe) >= HOLD_MAX)
        if use_stop:
            s = m & (cl[t, z] <= ppx * (1 - STOP))
            nstop += (s & ~ex).sum(axis=1)
            ex |= s
        if use_mkt and not mkt_on[t]:
            ex |= m
        pj = np.where(ex, -1, pj)
        if use_mkt and not mkt_on[t]:
            continue
        c = cand.get(t)
        if c is None:
            continue
        for s_ in range(nrep):
            free = np.flatnonzero(pj[s_] < 0)
            if not len(free):
                continue
            k, row = 0, c[s_]
            for slot in free:
                placed = False
                while k < row.shape[0]:
                    j = int(row[k])
                    k += 1
                    if j < 0 or j in pj[s_]:
                        continue
                    pj[s_, slot] = j
                    pe[s_, slot] = t
                    ppx[s_, slot] = cl[t, j]
                    ntr[s_] += 1
                    cst[s_, i] += COST / MAXPOS
                    placed = True
                    break
                if not placed:
                    break
    return ret, cst, nstop, ntr, nhold


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "float_mv", "turnover", "volume", "is_st", "is_suspended",
            "listed_days"]
    d = {c: {} for c in cols}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=cols)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in cols:
            d[k][c] = x[k]
    cldf = pd.DataFrame(d["close"]).sort_index()
    idx = cldf.index
    nt, ns = cldf.shape
    assert (nt, ns) == (3297, 5232), f"锚点A1a {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f)
    mv = al("float_mv").to_numpy() / 1e8
    trn = al("turnover")
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0) & np.isfinite(cl))
    fin = np.isfinite(cl)
    fst = np.argmax(fin, axis=0)
    gapn = int(sum((~fin[fst[j]:, j]).sum() for j in range(ns) if fin[:, j].any()))
    ind, _, _ = build_industry(list(cldf.columns), idx)
    px = pd.DataFrame(cl)
    with np.errstate(all="ignore"):
        rec = cl / np.where(px.rolling(250, min_periods=250).min().to_numpy() > 0,
                            px.rolling(250, min_periods=250).min().to_numpy(),
                            np.nan) - 1.0
        ma20 = px.rolling(20, min_periods=20).mean().to_numpy()
        mfrac = pd.DataFrame((cl > ma20).astype(np.float64)).where(
            np.isfinite(ma20)).rolling(120, min_periods=120).mean().to_numpy()
        r120 = px.pct_change(120).to_numpy()
        r60 = px.pct_change(60).to_numpy()
        tacc = (trn.rolling(20, min_periods=10).mean().to_numpy()
                / np.where(trn.rolling(60, min_periods=30).mean().to_numpy() > 0,
                           trn.rolling(60, min_periods=30).mean().to_numpy(),
                           np.nan) - 1.0)
    sus = al("is_suspended", True).astype(bool).to_numpy()
    trad = ~sus & (al("volume", 0).to_numpy() > 0) & np.isfinite(r60)
    rps60 = pd.DataFrame(np.where(trad, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100.0
    del ma20, trad
    nh60 = np.zeros((nt, ns), bool)
    pm = px.rolling(NH, min_periods=NH).max().shift(1).to_numpy()
    nh60[np.isfinite(pm) & (cl > pm)] = True
    del pm
    # ---- 大盘过滤:全市场等权净值 vs 自身 MA200(只用 <=t)----
    with np.errstate(all="ignore"):
        rr = cl[1:] / cl[:-1] - 1.0
    msk = ok[1:] & ok[:-1] & np.isfinite(rr)
    dr = np.zeros(nt)
    dr[1:] = np.where(msk.sum(1) > 0,
                      np.nan_to_num(rr * msk).sum(1) / np.maximum(msk.sum(1), 1), 0.0)
    nav = np.cumprod(1 + dr)
    mm = pd.Series(nav).rolling(MA_MKT, min_periods=MA_MKT).mean().to_numpy()
    mkt_on = ~(np.isfinite(mm) & (nav < mm))
    del rr, msk
    # ---- A1(d) 无前视断言 ----
    rs = np.random.default_rng(13)
    nchk = 0
    for _ in range(2000):
        t = int(rs.integers(300, nt))
        j = int(rs.integers(0, ns))
        if np.isfinite(rec[t, j]):
            assert abs(cl[t, j] / np.nanmin(cl[t - 249:t + 1, j]) - 1
                       - rec[t, j]) < 1e-9, "A1d 距低点"
            nchk += 1
        if nh60[t, j]:
            assert cl[t, j] > np.nanmax(cl[t - NH:t, j]), "A1d 突破只用 <t"
        assert abs(np.mean(nav[t - MA_MKT + 1:t + 1]) - mm[t]) < 1e-6 \
            or not np.isfinite(mm[t]), "A1d 大盘MA200 只用 <=t"
    print(f"A1a ✓ {cldf.shape};A1b ffill 空洞 {gapn} {'✓' if gapn == 0 else '✗'};"
          f"A1d 无前视 {nchk} 点 ✓;大盘过滤开启天数占比 "
          f"{mkt_on.mean():.1%}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- 每月选股 ----
    me = np.sort(pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy().astype(int))
    rules = {"Codex强确认": None, "第一四八节规则": None}
    sel = {k: {} for k in rules}
    months = []
    for a, b in zip(me[:-1], me[1:], strict=True):
        a, b = int(a), int(b)
        if idx[a] < pd.Timestamp(TRAIN[0]) or idx[a] > pd.Timestamp(HOLD[1]):
            continue
        e = np.flatnonzero(ok[a] & np.isfinite(mv[a]))
        if len(e) < 100:
            continue
        months.append((a, b))
        cs = e[(rec[a, e] >= 0.40) & (r120[a, e] >= 0.10) & (mfrac[a, e] >= 0.55)
               & (rps60[a, e] >= 90) & np.isfinite(rec[a, e])]
        v = np.isfinite(rec[a, e]) & np.isfinite(tacc[a, e])
        qr = pd.Series(np.where(v, rec[a, e], np.nan)).rank(pct=True).to_numpy()
        qt = pd.Series(np.where(v, tacc[a, e], np.nan)).rank(pct=True).to_numpy()
        r1 = e[v & (qr >= Q148) & (qt >= Q148)]
        for k, arr in (("Codex强确认", cs), ("第一四八节规则", r1)):
            sel[k][a] = arr[np.argsort(-rps60[a, arr], kind="stable")]
    print(f"调仓月 {len(months)};Codex强确认每月中位 "
          f"{int(np.median([len(sel['Codex强确认'][a]) for a, _ in months]))} 只,"
          f"第一四八节 {int(np.median([len(sel['第一四八节规则'][a]) for a, _ in months]))} 只",
          flush=True)

    rng = np.random.default_rng(SEED)
    viol = [0]

    def subs(day, js, nrep):
        """同日、同市值名次±25、同申万一级行业的随机替换;返回 (nrep, len(js))。"""
        e = np.flatnonzero(ok[day] & np.isfinite(mv[day]) & (ind[day] >= 0))
        o = e[np.argsort(mv[day, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        out = np.full((nrep, len(js)), -1, np.int64)
        for k, j in enumerate(js):
            p0, i0 = rk[j], ind[day, j]
            if p0 < 0 or i0 < 0:
                continue
            c = o[max(0, p0 - NBR):min(len(o) - 1, p0 + NBR) + 1]
            c = c[ind[day, c] == i0]
            if len(c) < 2:
                c = o[ind[day, o] == i0]
            if len(c) < 2:
                continue
            pick = c[rng.integers(0, len(c), nrep)]
            viol[0] += int((ind[day, pick] != i0).sum())
            out[:, k] = pick
        return out

    # ---- L0 基线:第一五二节原样(月末全部等权买入,持有到下月末)----
    def l0(rule, lo, hi):
        m = [(a, b) for (a, b) in months
             if pd.Timestamp(lo) <= idx[a] <= pd.Timestamp(hi)]
        f, cf = [], []
        for a, b in m:
            s = sel[rule][a]
            if len(s) < 3:
                f.append(1.0)
                cf.append(np.ones(NSEED))
                continue
            f.append(float(np.mean(cl[b, s] / cl[a, s])))
            sb = subs(a, s, NSEED)
            v = sb >= 0
            rr2 = np.where(v, (cl[b] / cl[a])[np.where(v, sb, 0)], np.nan)
            cf.append(np.nanmean(rr2, axis=1))
        nd = m[-1][1] - m[0][0]
        cf = np.array(cf).T
        return (ann(float(np.prod(f)), nd),
                np.array([ann(float(np.prod(cf[k])), nd) for k in range(NSEED)]), nd)

    # ---- L1–L4 槽位制 ----
    def ladder(rule, lo, hi, rung):
        m = [(a, b) for (a, b) in months
             if pd.Timestamp(lo) <= idx[a] <= pd.Timestamp(hi)]
        ta, tb = m[0][0], m[-1][1]
        real, ctrl = {}, {}
        for a, b in m:
            s = sel[rule][a][:NCAND]
            if not len(s):
                continue
            if rung < 4:
                real.setdefault(a, []).extend(s.tolist())
            else:
                for j in s:                       # 首次创 60 日新高的那天买
                    w = np.flatnonzero(nh60[a + 1:b + 1, j])
                    if w.size:
                        real.setdefault(a + 1 + int(w[0]), []).append(int(j))
        for day in sorted(real):
            js = np.asarray(real[day], np.int64)
            real[day] = js.reshape(1, -1)
            ctrl[day] = subs(day, js, NSEED)
        us, um = rung >= 2, rung >= 3
        r1, c1, ns1, nt1, nh1 = sim(real, 1, ta, tb, cl, mkt_on, us, um)
        r2, c2, _, _, nh2 = sim(ctrl, NSEED, ta, tb, cl, mkt_on, us, um)
        nd = tb - ta
        g = ann(float(np.prod(1 + r1[0])), nd)
        gc_ = ann(float(np.prod(1 + r1[0] - c1[0])), nd)
        cs = np.array([ann(float(np.prod(1 + r2[k])), nd) for k in range(NSEED)])
        return {"年化": g, "成本后": gc_, "对照": cs, "nd": nd,
                "回撤": mdd(np.cumprod(1 + r1[0])),
                "持仓": float(nh1[0].mean()), "对照持仓": float(nh2.mean()),
                "止损次数": int(ns1[0]), "交易次数": int(nt1[0]),
                "净值": np.cumprod(1 + r1[0])}

    res, w = [], 100
    for rule in ("Codex强确认", "第一四八节规则"):
        judge_rule = rule == "Codex强确认"
        print(f"\n{'='*w}\n{rule}"
              f"{'(主判据)' if judge_rule else '(只报数,不判定)'}\n{'='*w}")
        keep = {}
        for lo, hi, tag, jseg in ((TRAIN[0], TRAIN[1], "训练段", False),
                                  (HOLD[0], HOLD[1], "留出段", True)):
            print(f"\n  ── {tag} ──")
            print(f"  {'档':<26}{'年化':>9}{'成本后':>9}{'对照中位':>10}"
                  f"{'超额pp':>9}{'p':>8}{'回撤':>8}{'持仓':>7}{'止损':>7}")
            g0, cs0, nd0 = l0(rule, lo, hi)
            e0, p0 = g0 - float(np.median(cs0)), float((cs0 >= g0).mean())
            print(f"  {'L0 基线(第一五二节原样)':<22}{g0:>9.2%}{'—':>9}"
                  f"{np.median(cs0):>10.2%}{e0*100:>9.2f}{p0:>8.4f}{'—':>8}"
                  f"{'全部':>7}{'—':>7}")
            res.append({"规则": rule, "段": tag, "档": "L0 基线", "年化": g0,
                        "对照中位": float(np.median(cs0)), "超额pp": e0 * 100,
                        "p": p0})
            keep[(tag, "L0")] = (g0, float(np.median(cs0)))
            for rung, nm in ((1, "L1 +集中度10只/上限120日"), (2, "L2 +止损8%"),
                             (3, "L3 +大盘MA200过滤"), (4, "L4 +突破日买入")):
                r = ladder(rule, lo, hi, rung)
                cm = float(np.median(r["对照"]))
                ex = r["年化"] - cm
                pv = float((r["对照"] >= r["年化"]).mean())
                print(f"  {nm:<24}{r['年化']:>9.2%}{r['成本后']:>9.2%}{cm:>10.2%}"
                      f"{ex*100:>9.2f}{pv:>8.4f}{r['回撤']:>8.1%}"
                      f"{r['持仓']:>7.1f}{r['止损次数']:>7}")
                res.append({"规则": rule, "段": tag, "档": nm, "年化": r["年化"],
                            "成本后": r["成本后"], "对照中位": cm, "超额pp": ex * 100,
                            "p": pv, "回撤": r["回撤"], "平均持仓": r["持仓"],
                            "对照持仓": r["对照持仓"], "止损次数": r["止损次数"],
                            "交易次数": r["交易次数"]})
                keep[(tag, f"L{rung}")] = (r["年化"], cm)
                if jseg and rung == 4:
                    a1, a2 = ex >= 0.03, pv < 0.05
                    print(f"\n  **A2 主判据(L4,留出段)**:超额≥+3.00pp "
                          f"{'✓' if a1 else '✗'}({ex*100:+.2f}pp);p<0.05 "
                          f"{'✓' if a2 else '✗'}({pv:.4f}) → "
                          f"**{'通过' if (a1 and a2) else '不通过'}**"
                          if judge_rule else "")
        ds = keep[("留出段", "L4")][0] - keep[("留出段", "L0")][0]
        dc = keep[("留出段", "L4")][1] - keep[("留出段", "L0")][1]
        print(f"\n  **A3 归因(留出段)**:Δ策略 {ds*100:+.2f}pp、"
              f"Δ对照 {dc*100:+.2f}pp;Δ对照/Δ策略 = "
              f"{dc/ds if ds != 0 else float('nan'):.2f}")
        print(f"  → **{'风控是普适的,不构成选股的边' if (ds != 0 and dc >= 0.5*ds) else '风控的提升未被对照同等吃到'}**")
        res.append({"规则": rule, "段": "留出段", "档": "A3归因",
                    "Δ策略pp": ds * 100, "Δ对照pp": dc * 100,
                    "Δ对照/Δ策略": dc / ds if ds != 0 else np.nan})
    print(f"\n锚点A1c 行业恒等式违例 {viol[0]} {'✓' if viol[0] == 0 else '✗'}")
    r148h = [x for x in res if x["规则"] == "第一四八节规则"
             and x["段"] == "留出段" and x["档"] == "L0 基线"][0]["年化"]
    ok_a1e = abs(r148h - 0.1112) <= 0.0030
    print(f"锚点A1e L0 复现第一五二节:{r148h:+.2%} vs +11.12%,"
          f"差 {abs(r148h-0.1112)*100:.2f}pp {'✓' if ok_a1e else '✗ 本节作废'}")
    pd.DataFrame(res).to_csv(f"{OUT}/oneil_tao_ladder.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n落库 {OUT}/oneil_tao_ladder.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
