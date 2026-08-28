"""§133 事前登记:检验安信「一年三倍股择股模型」,并做逐条消融(结果未跑)。

模型原文(安信证券 2023-07-20 报告第 39 页,逐字摘录)
------------------------------------------------------
① 先定投资理念:高增长细分 >50% → 景气投资;30–50% → 核心资产;<30% → 产业主题
② 再选主线板块,四个基本点:1.板块经历长时间大幅度下跌 2.估值处于历史底部
   (近五年 20% 分位数) 3.行业关注度低,换手率为近五年 20% 分位数
   4.基本面正在发生积极变化
③ 最后选个股:市值 50 亿以下、PE<40 倍,且股价长期震荡(两年以上)
   或近半年下跌 20%;再按类型加条件(产业趋势型净利增速>100%;
   价格驱动型资产价格涨幅(预期)>300%;政策驱动型加仓在重要会议前一个月;
   技术驱动型把握技术异变点)
安信公布的回测:2010 年来年平均收益率 63.76%;**选出一年三倍股概率 9.61%**;
选出一年两倍股概率 27.13%;涨幅前五中出现三倍股概率 70%。

**明确排除、不做代理的条款(这是本节最重要的一段)**
------------------------------------------------------
以下条款**无法量化,本节一律不实现,也不编造代理指标**:
- ②-4「基本面正在发生积极变化」—— 主观判断,无对应字段;
- ①「高增长细分占比」—— 依赖安信自己的产业分类体系,本项目没有;
- ③「价格驱动型资产价格涨幅(**预期**)>300%」—— 是预期,**用实际值就是前视**;
- ③「政策驱动型加仓在重要会议前一个月」—— 需要会议日历,本项目没有;
- ③「技术驱动型把握技术异变点」—— 主观判断。
**因此本节检验的是安信模型的『可量化骨架』,不是它的全部。**
**骨架跑不过对照,不等于完整模型跑不过;骨架跑得过,也不等于完整模型的功劳。**
这句话必须原样写进结论,不许省略。

可量化骨架(逐条写死,跑完不改)
--------------------------------
板块层(申万一级,PIT,31 个行业):
  B1 长时间大幅下跌:行业等权指数近 **500 个交易日**收益 ≤ **−20%**
  B2 估值历史底部:行业中位 PE_TTM 在**近五年(1250 日)**分位 ≤ **20%**
  B3 关注度低:行业中位 20 日均换手率在近五年分位 ≤ **20%**
个股层:
  S1 流通市值 ≤ **50 亿**
  S2 PE_TTM ∈ (0, **40**)
  S3 形态二选一:(a) 近 **125 个交易日**收益 ≤ **−20%**,或
                 (b) 长期震荡:近 **500 个交易日** (max−min)/min < **50%**
**完整骨架 = B1&B2&B3&S1&S2&S3。**

评估口径
--------
- 调仓/评估时点:**每月最后一个交易日**(与 R01–R13 引擎同频)。
- 结局变量:**未来 250 个交易日峰值涨幅 ≥ +200%**(一年三倍),
  与第一二九节 Part C 的 AX 口径完全一致,可比。
- 同一只股票 **250 日内不重复计事件**。
- 区间:**2018-01 起**(近五年分位需要 1250 日历史,面板 2013-01 起)。
  **近 5 年 = 2021-01 起**,单独再判一次。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
K1 锚点(不过则本节作废)
   (a) 面板 (3297, 5232);TTM 恒等式违例 0;
   (b) 行业恒等式:对照与被对照股同属一个申万一级行业,违例 > 0 即作废;
   (c) 无前视:所有条件只用 ≤t 的信息,前瞻窗口起点严格 > t,逐点断言。

K2 骨架有效性。**核心判据。**
   统计量 = P(未来 250 日峰值 ≥ +200% | 选中)。
   对照:**同市值名次 ±25 且同申万一级行业**,同日抽取,**500 组种子**。
   两个区间各判各的,**Bonferroni:2 个区间,α = 0.05/2 = 0.025**。
   **K2 通过 ⟺ 单尾 p < 0.025。**

K3 逐条消融(**描述,不据此挑最优组合**——挑就是事后调参)
   分别去掉 B1/B2/B3/S1/S2/S3 各跑一次,报命中率与事件数,
   看每一条的边际贡献。**不得因为某个子集更好就宣称找到了改进版模型**;
   要变成规则必须另开一节重新事前登记。

K4 与安信公布值对照(描述)
   把 P(三倍股|选中) 与安信的 **9.61%** 并列。
   **口径差异必须同时列出**:安信区间 2010 起、本节 2018 起;
   安信"一年三倍股"定义未明,本节用未来 250 日峰值 ≥+200%;
   安信是年度评估,本节是月度事件。**不得直接说"高于/低于安信"就完事。**

事前预测
--------
**本节不下预测**(第一一九节起的约定)。只登记判据。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推;
**不给排除条款编代理指标**;**不因骨架结果就断言完整模型的好坏**;
**不基于本节结论做任何可交易性声明**。
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import NBR, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from codex_routes_rerun import build_fund  # noqa: E402
from industry_neutral import build_industry  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NSEED, ALPHA, GAP, HOR, THR = 500, 0.05 / 2, 250, 250, 2.00
W_DOWN, W_BOX, W_5Y = 500, 500, 1250


def fwd_hit(cl, hor, thr):
    nt, ns = cl.shape
    hit = np.zeros((nt, ns), bool)
    val = np.zeros((nt, ns), bool)
    for s in range(0, ns, 600):
        a = cl[:, s:s + 600].astype(np.float64)
        rm = pd.DataFrame(a[::-1]).rolling(hor, min_periods=1).max().to_numpy()[::-1]
        f = np.full_like(a, np.nan)
        f[:-1] = rm[1:]
        with np.errstate(all="ignore"):
            r = f / np.where(a > 0, a, np.nan) - 1.0
        v = np.isfinite(r)
        val[:, s:s + 600] = v
        hit[:, s:s + 600] = v & (r >= thr)
    return hit, val


def main():  # noqa: PLR0915
    t0 = time.time()
    files = sorted(glob.glob(f"{DATA}/*.parquet"))
    codes = [os.path.basename(f)[:-8] for f in files
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "raw_close", "float_mv", "turnover", "volume",
            "is_st", "is_suspended", "is_limit_up", "listed_days"]
    d = {c: {} for c in cols}
    for c in codes:
        have = pq.ParquetFile(f"{DATA}/{c}.parquet").schema.names
        x = pd.read_parquet(f"{DATA}/{c}.parquet",
                            columns=[w for w in cols if w in have])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in cols:
            d[k][c] = x[k] if k in x.columns else np.nan
    cldf = pd.DataFrame(d["close"]).sort_index()
    idx = cldf.index
    nt, ns = cldf.shape
    assert (nt, ns) == (3297, 5232), f"锚点K1a {cldf.shape}"

    def al(k, fill=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(fill)
    rawdf, mvdf = al("raw_close"), al("float_mv")
    turndf, voldf = al("turnover"), al("volume")
    stdf, susdf = al("is_st", True).astype(bool), al("is_suspended", True).astype(bool)
    ludf, lddf = al("is_limit_up", True).astype(bool), al("listed_days", 0)
    cl = cldf.where(cldf > 0).to_numpy(np.float64)
    print(f"面板 {cldf.shape} ({time.time()-t0:.0f}s)", flush=True)

    fm, abad = build_fund(list(cldf.columns), idx)
    assert abad == 0, "锚点K1a TTM"
    ind, _, _ = build_industry(list(cldf.columns), idx)
    ok = (~stdf.to_numpy() & ~susdf.to_numpy() & ~ludf.to_numpy()
          & (lddf.to_numpy() >= 365) & (voldf.to_numpy() > 0))
    print(f"锚点K1a ✓ {nt}×{ns};TTM ✓;行业覆盖 {(ind>=0).mean():.1%}", flush=True)

    with np.errstate(all="ignore"):
        eps = fm["eps_ttm"]
        pe = rawdf.to_numpy() / np.where(eps > 0, eps, np.nan)
        mv_yi = mvdf.to_numpy() / 1e8
        ret1 = cl / np.roll(cl, 1, axis=0) - 1.0
        ret1[0] = np.nan
        r500 = cl / np.roll(cl, W_DOWN, axis=0) - 1.0
        r500[:W_DOWN] = np.nan
        r125 = cl / np.roll(cl, 125, axis=0) - 1.0
        r125[:125] = np.nan
    dfc = pd.DataFrame(cl)
    bmax = dfc.rolling(W_BOX, min_periods=W_BOX).max().to_numpy()
    bmin = dfc.rolling(W_BOX, min_periods=W_BOX).min().to_numpy()
    with np.errstate(all="ignore"):
        box = bmax / np.where(bmin > 0, bmin, np.nan) - 1.0
    turn20 = turndf.rolling(20, min_periods=10).mean().to_numpy()

    # ---- 板块层:行业等权指数 / 中位 PE / 中位换手率 ----
    nind = int(ind.max()) + 1
    iret = np.full((nt, nind), np.nan)
    ipe = np.full((nt, nind), np.nan)
    itn = np.full((nt, nind), np.nan)
    for i in range(nind):
        m_ = (ind == i) & ok
        cnt = m_.sum(1)
        with np.errstate(all="ignore"):
            iret[:, i] = np.where(cnt > 0, np.nansum(np.where(m_, ret1, 0), 1)
                                  / np.maximum(cnt, 1), np.nan)
        pe_m = np.where(m_ & np.isfinite(pe), pe, np.nan)
        tn_m = np.where(m_ & np.isfinite(turn20), turn20, np.nan)
        with np.errstate(all="ignore"):
            ipe[:, i] = np.nanmedian(pe_m, 1)
            itn[:, i] = np.nanmedian(tn_m, 1)
    icum = np.nancumprod(1.0 + np.nan_to_num(iret), axis=0)
    with np.errstate(all="ignore"):
        i500 = icum / np.roll(icum, W_DOWN, axis=0) - 1.0
    i500[:W_DOWN] = np.nan
    qpe = pd.DataFrame(ipe).rolling(W_5Y, min_periods=W_5Y).rank(pct=True).to_numpy()
    qtn = pd.DataFrame(itn).rolling(W_5Y, min_periods=W_5Y).rank(pct=True).to_numpy()
    print(f"板块层完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 条件矩阵(个股级)----
    def bcast(a):
        out = np.full((nt, ns), np.nan)
        good = ind >= 0
        rows, cs = np.nonzero(good)
        out[rows, cs] = a[rows, ind[rows, cs]]
        return out
    b1 = bcast(i500) <= -0.20
    b2 = bcast(qpe) <= 0.20
    b3 = bcast(qtn) <= 0.20
    s1 = mv_yi <= 50
    s2 = (pe > 0) & (pe < 40)
    s3 = (r125 <= -0.20) | (box < 0.50)
    conds = {"b1": b1, "b2": b2, "b3": b3, "s1": s1, "s2": s2, "s3": s3}

    # 锚点 K1c 无前视(条件矩阵逐点重算)
    rs = np.random.default_rng(11)
    nchk = 0
    for _ in range(3000):
        t = int(rs.integers(W_5Y + 10, nt))
        j = int(rs.integers(0, ns))
        if np.isfinite(r125[t, j]):
            ref = cl[t, j] / cl[t - 125, j] - 1.0
            assert abs(ref - r125[t, j]) < 1e-9, "K1c r125"
            nchk += 1
        if np.isfinite(box[t, j]):
            w = cl[t - W_BOX + 1:t + 1, j]
            assert abs(np.nanmax(w) / np.nanmin(w) - 1.0 - box[t, j]) < 1e-6, "K1c box"
    print(f"锚点K1c 条件矩阵因果性 {nchk} 点一致 ✓", flush=True)

    hit, val = fwd_hit(cl, HOR, THR)
    nchk = 0
    for _ in range(2000):
        t = int(rs.integers(W_5Y, nt - HOR - 1))
        j = int(rs.integers(0, ns))
        if not val[t, j]:
            continue
        pk = np.nanmax(cl[t + 1:t + 1 + HOR, j]) / cl[t, j] - 1.0
        assert bool(pk >= THR) == bool(hit[t, j]), "K1c 前瞻窗口"
        nchk += 1
    print(f"锚点K1c 前瞻窗口起点 > t  {nchk} 点一致 ✓", flush=True)

    # 月末评估点
    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy()
    is_me = np.zeros(nt, bool)
    is_me[me] = True

    order_c, rank_c = {}, {}

    def prep(t):
        if t not in order_c:
            e = np.flatnonzero(ok[t] & np.isfinite(mv_yi[t]) & (ind[t] >= 0))
            o = e[np.argsort(mv_yi[t, e], kind="stable")]
            rk = np.full(ns, -1, np.int32)
            rk[o] = np.arange(len(o), dtype=np.int32)
            order_c[t], rank_c[t] = o, rk
        return order_c[t], rank_c[t]

    def run(mask, t_lo, tag):
        mk = mask & val & ok & is_me[:, None]
        tmax = nt - HOR - 1
        ts, js = [], []
        for j in range(ns):
            h = np.flatnonzero(mk[t_lo:tmax + 1, j])
            if not h.size:
                continue
            h += t_lo
            last = -10**9
            for t in h:
                if t - last >= GAP:
                    ts.append(int(t))
                    js.append(j)
                    last = t
        if len(ts) < 30:
            print(f"{tag:26s} 事件 {len(ts)} —— 少于 30,不判", flush=True)
            return {"tag": tag, "n_ev": len(ts), "p_hit": np.nan, "p": np.nan,
                    "K2": False, "viol": 0}
        ts = np.asarray(ts)
        js = np.asarray(js)
        ph = float(hit[ts, js].mean())
        chunks = []
        off = np.zeros(len(ts), np.int64)
        lens = np.zeros(len(ts), np.int64)
        keep = np.ones(len(ts), bool)
        pos_f = 0
        for k, (t, j) in enumerate(zip(ts, js, strict=True)):
            o, rk = prep(int(t))
            p0, i0 = rk[j], ind[t, j]
            if p0 < 0 or i0 < 0:
                keep[k] = False
                continue
            lo, hi = max(0, p0 - NBR), min(len(o) - 1, p0 + NBR)
            cand = o[lo:hi + 1]
            cand = cand[ind[t, cand] == i0]
            if len(cand) < 2:
                cand = o[ind[t, o] == i0]
            if len(cand) < 2:
                keep[k] = False
                continue
            off[k] = pos_f
            lens[k] = len(cand)
            pos_f += len(cand)
            chunks.append(cand)
        flat = np.concatenate(chunks).astype(np.int64)
        tk, jk, ofk, lnk = ts[keep], js[keep], off[keep], lens[keep]
        rng = np.random.default_rng(SEED)
        cp, viol = [], 0
        for _ in range(0, NSEED, 50):
            r = rng.random((50, len(tk)))
            pick = flat[ofk[None, :] + (r * lnk[None, :]).astype(np.int64)]
            viol += int((ind[tk, pick] != ind[tk, jk][None, :]).sum())
            v = val[tk, pick]
            h2 = hit[tk, pick] & v
            nv = v.sum(1)
            cp.extend(np.where(nv > 0, h2.sum(1) / np.maximum(nv, 1), np.nan))
        cp = np.asarray(cp, float)
        p = (1 + int(np.sum(cp >= ph))) / (NSEED + 1)
        print(f"{tag:26s} 事件{len(ts):6d} P(≥+200%)={ph:6.2%} | 对照中位{np.nanmedian(cp):6.2%}"
              f" 95分位{np.nanpercentile(cp,95):6.2%} p={p:.4f} "
              f"{'✓' if p < ALPHA else '✗'} | 行业违例{viol}", flush=True)
        return {"tag": tag, "n_ev": len(ts), "p_hit": ph,
                "ctrl_med": float(np.nanmedian(cp)), "p": p,
                "K2": bool(p < ALPHA), "viol": viol}

    full = b1 & b2 & b3 & s1 & s2 & s3
    t18 = int(np.searchsorted(idx, pd.Timestamp("2018-01-01")))
    t21 = int(np.searchsorted(idx, pd.Timestamp("2021-01-01")))
    print(f"\n=== K2 骨架有效性(α={ALPHA})===", flush=True)
    rows = [run(full, t18, "完整骨架 2018起"), run(full, t21, "完整骨架 2021起(近5年)")]
    print("\n=== K3 逐条消融(描述,不据此挑最优)===", flush=True)
    for k in conds:
        m = np.ones((nt, ns), bool)
        for k2, v2 in conds.items():
            if k2 != k:
                m &= v2
        rows.append(run(m, t18, f"去掉 {k} 2018起"))
    for k, v in conds.items():
        rows.append(run(v, t18, f"仅 {k} 2018起"))
    df = pd.DataFrame(rows)
    v = int(df["viol"].sum())
    print(f"\n锚点K1b 行业违例 {v} 次 {'✓' if v == 0 else '✗ 作废'}")
    assert v == 0
    core = df.head(2)
    print(f"K2 通过 {int(core['K2'].sum())}/2")
    print("\n=== K4 与安信公布值对照 ===\n"
          "  安信公布 P(一年三倍股|选中) = 9.61%(2010 起,年度评估,定义未明)")
    for _, r in core.iterrows():
        print(f"  本节 {r['tag']}:{r['p_hit']:.2%}(2018/2021 起,月末事件,未来250日峰值≥+200%)")
    df.to_csv(f"{OUT}/anxin_model_test.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/anxin_model_test.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
