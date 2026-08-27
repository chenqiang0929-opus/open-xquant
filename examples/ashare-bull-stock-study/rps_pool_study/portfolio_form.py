"""§152 事前登记:换评价形式 —— 把启动规则做成组合净值,对同数量随机对照(结果未跑)。

起因
----
用户在三条路里选了第 3 条:**换评价形式**。
第一五一节六个结构全部不过,说明「单股概率排序」这条路走不通;
而第一一〇/一一二节的三段突破是本项目**唯一**在
「组合层面 + 同数量随机对照」下通过的形态信号(+3.96pp,p=0.0000)。
**本节把第一四八节的规则从「概率排序」改成「组合净值」来评价。**

两者的区别(这是本节的全部意义)
--------------------------------
- 概率排序问:**这只股票未来 60 日涨 50% 的概率是否更高** → 被中位数主导;
- 组合净值问:**按规则持有一篮子,是否跑赢同样数量的随机一篮子** →
  **右尾的大涨会直接进入净值**,不会被"命中率"稀释。
第一五一节已证明 lift 与「抓住极端右尾」反向,**组合口径正是为这一点设计的。**

规则(不改,沿用第一四八节)
--------------------------
距一年低点涨幅 ∈ 全市场当日前 30% 且 换手加速 ∈ 全市场当日前 30%。
**不加上限、不加排序截断** —— 第一五〇节已证明上限会把主升段剔除。

组合构建
--------
- **每月最后一个交易日调仓**,等权买入当期全部选中股票,持有到下一个月末;
- 停牌/退市按最后有效价 ffill 参与(用户规则5),**绝不剔除**;
- 两个成本口径:**零成本** 与 **双边 0.2%**(每月全额换手的上界估计)。

对照(与第一一〇节同规格)
------------------------
每月末**同日、同市值名次 ±25、同申万一级行业**随机抽**同样只数**,
等权持有同样时间,**500 组种子**。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
L1 锚点(不过则本节作废)
   (a) 面板 (3297, 5232);价格 ffill;
   (b) **行业恒等式**:对照与被对照股同申万一级行业,违例 > 0 即作废;
   (c) **无前视**:选股只用 ≤t 的信息,收益区间严格 (t, t+1]。

L2 **核心判据(与第一一〇节同规格,两条同时满足)**
   **训练段 2019-2022 只报数,不判定**(规则已在别处挑过)。
   **留出段 2023-01–2026-04 判定:**
   (a) **零成本口径年化超额 ≥ +3.00pp**(对照 500 组的中位数);
   (b) **单尾 p < 0.05**(策略年化 > 多少组对照)。
   **两条同时满足才算通过。**

L3 描述项(不参与判定):最大回撤 vs 对照;双边 0.2% 成本口径的超额;逐年超额。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。

不做的
------
不改规则、不加条件、不调阈值;不新增顶层目录;不 force push;
**若 L2 不过,如实写「组合口径也没做到」,不回头改规则再跑**;
不作任何可交易性声明。
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
Q = 0.70                     # 第一四八节的分位阈值,不改
NSEED = 500                  # 对照种子数
COST = 0.002                 # 双边合计 0.2%/月(每月全额换手的上界估计)
TRAIN = ("2019-01-01", "2022-12-31")
HOLD = ("2023-01-01", "2026-04-30")


def ann(fac, nd):
    """把逐月毛因子连乘成年化(nd = 该段的交易日数,250 日 = 1 年)。"""
    return float(np.prod(fac) ** (250.0 / nd) - 1.0)


def mdd(eq):
    pk = np.maximum.accumulate(eq)
    return float(np.max((pk - eq) / pk))


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
    assert (nt, ns) == (3297, 5232), f"锚点L1a {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f)
    mv = al("float_mv").to_numpy() / 1e8
    trn = al("turnover")
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)   # 用户规则5:ffill 参与
    ok &= np.isfinite(cl)
    ind, _, _ = build_industry(list(cldf.columns), idx)
    lo250 = pd.DataFrame(cl).rolling(250, min_periods=250).min().to_numpy()
    t20 = trn.rolling(20, min_periods=10).mean().to_numpy()
    t60 = trn.rolling(60, min_periods=30).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        tacc = t20 / np.where(t60 > 0, t60, np.nan) - 1.0

    # ---- L1(a) 价格 ffill:首个有效价之后不得再出现 NaN ----
    fin = np.isfinite(cl)
    first = np.argmax(fin, axis=0)
    gap = int(sum((~fin[first[j]:, j]).sum() for j in range(ns) if fin[:, j].any()))
    # ---- L1(c) 无前视:rec / tacc 逐点重算 ----
    trnv = trn.to_numpy()
    rs = np.random.default_rng(13)
    n1 = n2 = 0
    for _ in range(3000):
        t = int(rs.integers(260, nt))
        j = int(rs.integers(0, ns))
        if np.isfinite(rec[t, j]):
            assert abs(cl[t, j] / np.nanmin(cl[t - 249:t + 1, j]) - 1.0
                       - rec[t, j]) < 1e-9, "L1c rec 用到了 >t 的数据"
            n1 += 1
        if np.isfinite(tacc[t, j]):
            a = trnv[t - 19:t + 1, j]
            b = trnv[t - 59:t + 1, j]
            if np.isfinite(a).sum() >= 10 and np.isfinite(b).sum() >= 30:
                assert abs(np.nanmean(a) / np.nanmean(b) - 1.0 - tacc[t, j]) < 1e-9, \
                    "L1c tacc 用到了 >t 的数据"
                n2 += 1
    print(f"锚点L1a ✓ 面板 {cldf.shape};ffill 后首价之后的空洞 {gap} "
          f"{'✓' if gap == 0 else '✗'}", flush=True)
    print(f"锚点L1c ✓ 无前视:rec 重算 {n1} 点、换手加速重算 {n2} 点,"
          f"全部一致({time.time()-t0:.0f}s)", flush=True)

    # ---- 每月最后一个交易日 ----
    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy().astype(int)
    me = np.sort(me)

    months, sels, univ = [], [], []
    for a, b in zip(me[:-1], me[1:], strict=True):
        a, b = int(a), int(b)
        if idx[a] < pd.Timestamp(TRAIN[0]) or idx[a] > pd.Timestamp(HOLD[1]):
            continue
        assert b > a, "L1c 收益区间必须严格 (t, t+1]"
        m = ok[a] & np.isfinite(rec[a]) & np.isfinite(tacc[a]) & np.isfinite(mv[a])
        e = np.flatnonzero(m)
        if len(e) < 100:
            continue
        qr = pd.Series(rec[a, e]).rank(pct=True).to_numpy()
        qt = pd.Series(tacc[a, e]).rank(pct=True).to_numpy()
        s = e[(qr >= Q) & (qt >= Q)]
        if len(s) < 10:
            continue
        months.append((a, b))
        sels.append(s)
        univ.append(e)
    print(f"调仓月 {len(months)} 个({idx[months[0][0]].date()} → "
          f"{idx[months[-1][1]].date()});每月选中中位 "
          f"{int(np.median([len(s) for s in sels]))} 只,"
          f"覆盖率中位 {np.median([len(s)/len(u) for s, u in zip(sels, univ, strict=True)]):.1%}",
          flush=True)

    # ---- 逐月毛因子:等权买入、持有到下月末(买入并持有,不做月内再平衡)----
    fac = np.array([float(np.mean(cl[b, s] / cl[a, s]))
                    for (a, b), s in zip(months, sels, strict=True)])

    # ---- 对照:同日、同市值名次±25、同申万一级行业,随机抽同样只数,500 组 ----
    rng = np.random.default_rng(SEED)
    cfac = np.ones((NSEED, len(months)))
    viol = kept = dropped = 0
    for i, ((a, b), s, e) in enumerate(zip(months, sels, univ, strict=True)):
        pool = e[(ind[a, e] >= 0)]
        o = pool[np.argsort(mv[a, pool], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        flat, off, lens, use = [], [], [], []
        pos = 0
        for j in s:
            p0, i0 = rk[j], ind[a, j]
            if p0 < 0 or i0 < 0:
                dropped += 1
                continue
            lo, hi = max(0, p0 - NBR), min(len(o) - 1, p0 + NBR)
            cand = o[lo:hi + 1]
            cand = cand[ind[a, cand] == i0]
            if len(cand) < 2:
                cand = o[ind[a, o] == i0]
            if len(cand) < 2:
                dropped += 1
                continue
            flat.append(cand)
            off.append(pos)
            lens.append(len(cand))
            use.append(j)
            pos += len(cand)
        if not flat:
            cfac[:, i] = fac[i]
            continue
        flat = np.concatenate(flat).astype(np.int64)
        off = np.asarray(off, np.int64)
        lens = np.asarray(lens, np.int64)
        use = np.asarray(use, np.int64)
        kept += len(use)
        rr = cl[b] / cl[a]
        r = rng.random((NSEED, len(use)))
        pick = flat[off[None, :] + (r * lens[None, :]).astype(np.int64)]
        viol += int((ind[a, pick] != ind[a, use][None, :]).sum())
        cfac[:, i] = rr[pick].mean(axis=1)
    print(f"锚点L1b 行业恒等式违例 {viol} {'✓' if viol == 0 else '✗'};"
          f"对照可配 {kept:,} 只次、无法配对丢弃 {dropped} 只次 "
          f"({time.time()-t0:.0f}s)", flush=True)
    if gap != 0 or viol != 0:
        print("**L1 不过 → 本节结论作废**")
        return

    # ---- 分段判定 ----
    ent = np.array([idx[a] for a, _ in months])
    res, curves = [], {}
    for tag, (lo, hi), judge in (("训练段 2019-2022(只报数,不判定)", TRAIN, False),
                                 ("**留出段 2023-01–2026-04(判据在这里)**",
                                  HOLD, True)):
        m = (ent >= pd.Timestamp(lo)) & (ent <= pd.Timestamp(hi))
        f, cf = fac[m], cfac[:, m]
        nd = int(months[int(np.flatnonzero(m)[-1])][1]
                 - months[int(np.flatnonzero(m)[0])][0])
        g = ann(f, nd)
        gc = ann(f * (1 - COST), nd)
        cs = np.array([ann(cf[k], nd) for k in range(NSEED)])
        cmed = float(np.median(cs))
        exc = g - cmed
        p = float((cs >= g).mean())
        eq = np.cumprod(f)
        ceq = np.cumprod(cf, axis=1)
        cdd = np.array([mdd(ceq[k]) for k in range(NSEED)])
        w = 92
        print(f"\n{'='*w}\n{tag}(调仓月 {int(m.sum())},交易日 {nd})\n{'='*w}")
        print(f"  组合净值 {eq[-1]:.3f}   **零成本年化 {g:+.2%}**   "
              f"双边0.2%/月年化 {gc:+.2%}   最大回撤(月度)**{mdd(eq):.1%}**")
        print(f"  对照({NSEED} 组)年化中位 {cmed:+.2%}  "
              f"[{np.percentile(cs,5):+.2%}, {np.percentile(cs,95):+.2%}]   "
              f"回撤中位 {np.median(cdd):.1%}")
        print(f"  **超额 {exc:+.2%}(= {exc*100:+.2f}pp)   单尾 p {p:.4f}**")
        if judge:
            c1, c2 = exc >= 0.03, p < 0.05
            print(f"  **L2 判定**:超额 ≥ +3.00pp {'✓' if c1 else '✗'}"
                  f"({exc*100:+.2f}pp);p < 0.05 {'✓' if c2 else '✗'}({p:.4f})"
                  f" → **{'通过' if (c1 and c2) else '不通过'}**")
        curves[tag] = (m, eq, ceq)
        res.append({"段": tag, "调仓月": int(m.sum()), "交易日": nd,
                    "零成本年化": g, "双边0.2%年化": gc, "对照年化中位": cmed,
                    "超额pp": exc * 100, "p": p, "回撤": mdd(eq),
                    "对照回撤中位": float(np.median(cdd))})

    # ---- L3 逐年(描述)----
    w = 92
    print(f"\n{'='*w}\nL3 逐年超额(描述,不参与判定)\n{'='*w}")
    print(f"{'年':<7}{'月数':>6}{'组合':>10}{'对照中位':>11}{'超额pp':>10}"
          f"{'对照>组合的比例':>16}")
    for y in range(2019, 2027):
        m = np.array([e.year == y for e in ent])
        if m.sum() < 6:
            continue
        f, cf = fac[m], cfac[:, m]
        pf = float(np.prod(f) - 1.0)
        pc = np.prod(cf, axis=1) - 1.0
        print(f"{y:<7}{int(m.sum()):>6}{pf:>10.2%}{np.median(pc):>11.2%}"
              f"{(pf-np.median(pc))*100:>10.2f}{(pc >= pf).mean():>16.1%}")
        res.append({"段": str(y), "调仓月": int(m.sum()), "零成本年化": pf,
                    "对照年化中位": float(np.median(pc)),
                    "超额pp": (pf - float(np.median(pc))) * 100,
                    "p": float((pc >= pf).mean())})

    pd.DataFrame(res).to_csv(f"{OUT}/portfolio_form.csv", index=False,
                             encoding="utf-8-sig")
    pd.DataFrame({"调仓日": [idx[a].date() for a, _ in months],
                  "到期日": [idx[b].date() for _, b in months],
                  "只数": [len(s) for s in sels],
                  "合格池": [len(u) for u in univ],
                  "组合月收益": fac - 1.0,
                  "对照月收益中位": np.median(cfac, axis=0) - 1.0}).to_csv(
        f"{OUT}/portfolio_form_monthly.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/portfolio_form.csv、portfolio_form_monthly.csv  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
