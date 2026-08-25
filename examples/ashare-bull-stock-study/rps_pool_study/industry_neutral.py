"""§125 行业中性对照:R08 的超额是选股,还是银行板块 beta?

为什么必须做
------------
§117–§124 做了八轮检验,每一轮都做市值中性,**一次都没做行业中性**。
而 R08 的持仓(申万一级,PIT 分类,覆盖率 100%)是:

    银行 75.8% / 房地产 10.3% / 建筑装饰 3.5% / 钢铁 3.2% / 交通运输 2.5% / ...
    最新一期 20/20 全是银行;每次调仓的银行占比中位 90%,67 次里 40 次 ≥80%

**拿一个押注单一板块的组合,去比一个从全行业随机抽的同市值对照,
测出的「超额」可能就是银行板块 beta。** 月度数据三年方向全部吻合银行行情:
2022 R08 +0.79% / 沪深300 −20.35%(熊市银行防守);
2024 +39.92%,9 月单月 +15.85%(中特估行情);
2025 +9.14% / 沪深300 +20.77%(成长主导,银行跑输)。

按 §79 正问「什么会让它通过而不回答我的问题」,**行业集中就是那个东西,我漏检了八轮。**
本节补上,并接受结论可能推翻 §119/§123/§124 对 R08 的判定。

行业数据
--------
`quant-research-dev/mktdata/others/classification_enriched.parquet`(Git LFS,137KB)
申万一级 `industry_l1_name`,带 `effective_from` / `effective_to`,**逐期生效**,
不是「今日行业套全历史」。12,885 条记录。

对照设计
--------
**对照 I(行业+市值双中性)**:每个调仓日,把当日合格股票**按行业分组**;
策略选中的第 i 只属于行业 I_i、在**该行业内**的市值升序名次为 r_i;
对照从**同一行业内**名次 [r_i−NBR, r_i+NBR] 抽一只替换(已抽中的排除)。
→ 行业暴露与市值暴露**同时**与策略逐日对齐,被打掉的只有「行业内的选股」这一维。
若某行业内合格数不足,取该行业全部(记录发生次数)。

**基准 B(闭眼买银行)**:每个调仓日等权持有当日**全部合格银行股**(不含成本、
不含整手,纯板块曲线),用于回答「R08 比闭眼买银行更好吗」。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
W1 锚点(不过则整节作废)
   (a) 面板 (3297, 5217);
   (b) **行业恒等式**:对照 I 的每一次抽样,替换股与被替换股**必须同属一个
       申万一级行业**,对所有 i、所有调仓日、所有种子成立;违例 > 0 即作废;
   (c) 行业覆盖率:合格股票能查到 PIT 行业的比例 ≥ 95%(查不到的剔出对照池并记录);
   (d) TTM 恒等式;泰格同比复现。

W2 行业中性后的显著性。**核心判据。**
   对 R08、R09、CFP 单因子三条,用对照 I,**500 组种子**(p 下限 1/501=0.001996)。
   **Bonferroni:3 条,α = 0.05/3 = 0.016667。**
   W2 通过 ⟺ 训练期(2014-01-02→2021-12-31)与留出期(2022-01-04→面板末)
   的 p **都** < 0.016667。
   **不通过 → 判定该条的超额来自行业暴露,不是行业内的选股能力**,
   §119/§123/§124 对它的结论相应作废。

W3 对「闭眼买银行」的比较(描述项,不设通过阈值)。
   报告 R08 与基准 B 在训练期、留出期的年化、回撤、夏普,以及 R08 的银行占比。
   基准 B 不含成本而 R08 含成本,**这个比较对 R08 不利,如实标注**。

事前预测
--------
**本节不下预测**(§119 起已停止「样本外会不会过」类外推)。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;**不往 quant-research-dev 推任何东西**;
不因结果不利就改对照设计或换窗口;
不基于本节结论做任何可交易性声明。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, NBR, OUT, SEED, run_window_fast  # noqa: E402
from codex_r10_replication import DATA, TOP_N, WEIGHT, metrics  # noqa: E402
from codex_routes_rerun import build_fund, route_scores, wrank  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402

NSEED, ALPHA = 500, 0.05 / 3
WINS = {"train": ("2014-01-02", "2021-12-31"), "holdout": ("2022-01-04", "2026-08-03")}
CLS = ("/home/user/quant-research-dev/mktdata/others/"
       "classification_enriched.parquet")


def build_industry(codes, idx):
    """PIT 申万一级行业 -> (nt, ns) 的 int16 行业 id 矩阵,-1 = 未知。"""
    c = pd.read_parquet(CLS, columns=["code", "industry_l1_name",
                                      "effective_from", "effective_to"])
    c["code"] = c["code"].astype(str).str.zfill(6)
    c["effective_from"] = pd.to_datetime(c["effective_from"])
    c["effective_to"] = pd.to_datetime(c["effective_to"]).fillna(
        pd.Timestamp("2099-01-01"))
    names = sorted(c["industry_l1_name"].dropna().unique())
    nid = {n: i for i, n in enumerate(names)}
    pos = {code: j for j, code in enumerate(codes)}
    m = np.full((len(idx), len(codes)), -1, np.int16)
    arr = idx.to_numpy()
    for code, g in c.groupby("code", sort=False):
        j = pos.get(code)
        if j is None:
            continue
        for _, r in g.iterrows():
            if pd.isna(r["industry_l1_name"]):
                continue
            a = np.searchsorted(arr, r["effective_from"].to_datetime64(), "left")
            b = np.searchsorted(arr, r["effective_to"].to_datetime64(), "left")
            if b > a:
                m[a:b, j] = nid[r["industry_l1_name"]]
    return m, names, nid


def draw_industry(rng, order_in_ind, pos_in_ind, nbr=NBR):
    """对照 I:在**同一行业内**的市值名次邻域抽样。order_in_ind 为该行业内按市值升序的列号。"""
    n = len(order_in_ind)
    lo = max(0, pos_in_ind - nbr)
    hi = min(n - 1, pos_in_ind + nbr)
    return int(rng.integers(lo, hi + 1))


def main():  # noqa: PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点W1a"
    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    assert abs(float(y.get((2017, "中报"), np.nan)) - 0.5307) < 0.005, "锚点W1"
    print(f"锚点W1a ✓ {nt}×{ns};泰格同比 ✓", flush=True)

    t0 = time.time()
    ind, names, nid = build_industry(codes, idx)
    bank = nid.get("银行", -99)
    print(f"行业矩阵完成 ({time.time()-t0:.0f}s),{len(names)} 个申万一级行业,"
          f"银行 id={bank}", flush=True)

    raw = np.full((nt, ns), np.nan, np.float32)
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点W1d TTM"
    print("锚点W1d ✓ TTM 恒等式", flush=True)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)

    def wpos(w):
        d0, d1 = WINS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    def cfp_score(t, e):
        v = fm["ocfps_ttm"][t, e] / raw[t, e].astype(np.float64)
        return wrank(v, v > 0)

    def build(fn):
        """返回 sel 与每期的行业内定位信息(供对照 I 用)。"""
        sel, meta = {}, {}
        cov_n = cov_d = 0
        for t in reb:
            t = int(t)
            base = ok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t])
            e = np.flatnonzero(base)
            if len(e) < TOP_N:
                continue
            cov_d += len(e)
            cov_n += int((ind[t, e] >= 0).sum())
            e = e[ind[t, e] >= 0]                     # 查不到行业的剔出
            if len(e) < TOP_N:
                continue
            v = np.asarray(fn(t, e), dtype=float)
            g = np.isfinite(v)
            if not g.any():
                continue
            e2 = e[g]
            k = min(TOP_N, len(e2))
            top = e2[np.argsort(-v[g], kind="stable")[:k]]
            sel[t] = (top, np.full(k, WEIGHT))
            groups = {}
            for i2 in np.unique(ind[t, e]):
                gi = e[ind[t, e] == i2]
                groups[int(i2)] = gi[np.argsort(logcap[t, gi], kind="stable")]
            posmap = {}
            for c_ in top:
                i2 = int(ind[t, c_])
                posmap[int(c_)] = (i2, int(np.flatnonzero(groups[i2] == c_)[0]))
            meta[t] = (groups, posmap)
        return sel, meta, cov_n / max(cov_d, 1)

    viol = 0

    def ev(sel, meta, w, nseed=NSEED):
        nonlocal viol
        w0, w1 = wpos(w)
        eq, dd, _, _ = run_window_fast(op, cl, susp, lu, ld, sel, cal_pos, w0, w1)
        m = metrics(eq, dd, idx)
        cg = []
        for sd in range(nseed):
            rng = np.random.default_rng(SEED + sd)
            cs = {}
            for t, (cols, _) in sel.items():
                groups, posmap = meta[t]
                out, taken = [], set()
                for c_ in cols:
                    i2, p2 = posmap[int(c_)]
                    g2 = groups[i2]
                    for _ in range(40):
                        q = draw_industry(rng, g2, p2)
                        if (i2, q) not in taken:
                            break
                    taken.add((i2, q))
                    pick = int(g2[q])
                    viol += int(ind[t, pick] != i2)      # 行业恒等式
                    out.append(pick)
                cs[t] = (np.array(out, np.int64), np.full(len(out), WEIGHT))
            e2, d2, _, _ = run_window_fast(op, cl, susp, lu, ld, cs, cal_pos, w0, w1)
            cg.append(metrics(e2, d2, idx)["cagr"])
        a = np.array(cg)
        return m, float(np.median(a)), (1 + int(np.sum(a >= m["cagr"]))) / (nseed + 1)

    rows = []
    for name, fn in (("R08", lambda t, e: route_scores("R08", t, e, fm, cl, raw,
                                                       logcap, tmean, "raw")),
                     ("R09", lambda t, e: route_scores("R09", t, e, fm, cl, raw,
                                                       logcap, tmean, "raw")),
                     ("CFP_single", cfp_score)):
        sel, meta, cov = build(fn)
        bk = np.mean([float(np.mean(ind[t, v[0]] == bank)) for t, v in sel.items()])
        r = {"strategy": name, "n_reb": len(sel), "ind_cov": cov, "bank_share": bk}
        for w in WINS:
            m, med, p = ev(sel, meta, w)
            r |= {f"{w}_cagr": m["cagr"], f"{w}_mdd": m["mdd"],
                  f"{w}_sharpe": m["sharpe"], f"{w}_ctrl": med, f"{w}_p": p}
        r["W2"] = bool(r["train_p"] < ALPHA and r["holdout_p"] < ALPHA)
        rows.append(r)
        print(f"{name:11s} 银行占比{bk:5.1%} | 训练{r['train_cagr']:+7.2%} "
              f"对照{r['train_ctrl']:+7.2%} p={r['train_p']:.4f} | "
              f"留出{r['holdout_cagr']:+7.2%} 回撤{r['holdout_mdd']:+7.2%} "
              f"夏普{r['holdout_sharpe']:5.2f} 对照{r['holdout_ctrl']:+7.2%} "
              f"p={r['holdout_p']:.4f} | W2 {'✓' if r['W2'] else '✗'}", flush=True)

    # ---- W3 基准 B:等权持有当日全部合格银行股(不含成本)----
    ret = np.zeros_like(cl)
    ret[1:] = cl[1:] / cl[:-1] - 1.0
    print("\nW3 基准 B(等权全部合格银行股,不含成本、不含整手)", flush=True)
    for w in WINS:
        w0, w1 = wpos(w)
        days = cal_pos[(cal_pos >= w0) & (cal_pos <= w1)]
        eqv, held = 1.0, None
        curve = []
        for t in days:
            if held is not None and len(held):
                rr = ret[t, held]
                rr = rr[np.isfinite(rr)]
                eqv *= 1 + (float(rr.mean()) if len(rr) else 0.0)
            curve.append(eqv)
            if int(t) in reb:
                held = np.flatnonzero(ok[t] & (ind[t] == bank))
        cv = np.array(curve)
        yrs = (idx[days[-1]] - idx[days[0]]).days / 365.25
        rr = np.diff(cv) / cv[:-1]
        rr = rr[np.isfinite(rr)]
        sd = rr.std(ddof=1)
        print(f"  {w:8s} 年化{cv[-1]**(1/yrs)-1:+7.2%} 回撤"
              f"{np.min(cv/np.maximum.accumulate(cv)-1):+7.2%} "
              f"夏普{rr.mean()/sd*np.sqrt(252) if sd > 0 else 0:5.2f}", flush=True)
        rows.append({"strategy": f"BANK_EW_{w}", f"{w}_cagr": float(cv[-1]**(1/yrs)-1),
                     f"{w}_mdd": float(np.min(cv/np.maximum.accumulate(cv)-1))})

    df = pd.DataFrame(rows)
    print(f"\n锚点W1b 行业恒等式违例 {viol} 次 {'✓' if viol == 0 else '✗ 作废'}")
    assert viol == 0
    print(f"锚点W1c 行业覆盖率 {df['ind_cov'].dropna().min():.1%} "
          f"{'✓' if df['ind_cov'].dropna().min() >= 0.95 else '✗'}")
    print(f"W2 通过 {int(df['W2'].fillna(False).sum())}/3(α={ALPHA:.6f}):"
          f"{', '.join(df.loc[df['W2'].fillna(False), 'strategy']) or '无'}")
    df.to_csv(f"{OUT}/industry_neutral.csv", index=False)
    print(f"落库 {OUT}/industry_neutral.csv")


if __name__ == "__main__":
    main()
