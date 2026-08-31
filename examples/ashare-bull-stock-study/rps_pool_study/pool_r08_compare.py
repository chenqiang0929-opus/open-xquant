"""第一七〇节 事前登记:把 R08 也算出来,与 Codex 2026-08-28 的新模板逐只比(结果未跑)。

起因
----
用户上传 Codex 新版工作簿(662 只次新股池,RPS50 分档),他新增了
**R08 价值分**、**R09 核心质量分**、营收/净利增长率、行业、平台调整天数等列,
并要求「用 RPS50 重跑一下数据,比对下 Codex 数据,是否存在差异性」。

信号类字段(信号类型/平台信号/周线五态/RPS)第一六六节已比过 ——
**他有信号的 86 只 86/86 = 100%**,强确认 13、标准确认 4 完全一致。
**本节只比新增的两个分数:R08 与 R09。**

已查清、必须写在前面的一件事(与本节口径直接相关)
------------------------------------------------
他的「数据锚点」页写 `r08.price_basis = "observation-day unadjusted close"`
(观察日不复权收盘),**但实测不是**:

- `R08计算明细` 的 `value_price` 与 `全部清单` 的收盘价 **552/552 完全相等**;
- 该收盘价与源数据 `mktdata_enriched` 的 `close` 列 **100% 吻合**(|差| < 0.005);
- 而 `close` 是**前复权**:`adj_factor = hfq_close/close` 在整个 2026 年**逐股恒定**
  (茅台 8.882483–8.882545),且 `close` 的最大单日跳动与 `hfq_close` 完全相同
  (600519 于 2026-01-29 同为 8.61%)—— **没有除权跳空被 close 单独承担**。

**所以他的 R08 分母是前复权价,不是不复权价,锚点页那一行的措辞与实现不符。**
影响多大要量,不能拍脑袋 —— 本节因此把两种价格口径都算,并把差报出来。
(本地面板实测:`raw_close/close` 在面板末日中位 **1.0008**、区间 0.948–1.030,
即观察日两者接近但不相等;而在面板首日中位 1.68、最大 20.6。
所以**单期截面上的失真远小于第一一七节那种历史回测里的失真**。)

口径
----
面板 `/home/user/oxq-panel-0828`(3316×5232,末日 2026-08-28);
合格 = 非 ST、非停牌、上市满 250 日(含 `listed_days=0` 源缺陷修正)、有成交、价格有效;
R08/R09 定义**逐字复用** `codex_routes_rerun.route_scores`,不重写;
横截面 = 我的全市场合格集(**不是 662 池内排名**)—— 与他的做法一致。

判据(跑之前写死,跑完照判,不放宽)
----------------------------------
E1 锚点(不过则本节作废)
   (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) TTM 恒等式违例 = 0;
   (c) **本节算出的 R09 必须与 `pool_20260828.csv` 已落库的「R09核心质量分」
       逐只相等(容差 1e-9)** —— 证明我走的是同一条路径,不是另起炉灶。

E2 描述项(不设通过/不通过,只登记必须报什么)
   (a) R09:我 vs 他,可比只数、中位 |差|、|差| < 0.05 的占比、Spearman ρ;
   (b) R08:**分别用真实口径(不复权)与他的口径(前复权)各算一遍**,
       各自与他比,并给出「换价格口径本身造成多大差」;
   (c) 两个分数各自的缺失只数与缺失原因分布;
   (d) 他 13 只强确认在两个分数上的取值。

**本节不下预测**(第一一九节起的约定)。
**本节是口径核对,不构成任何买入建议。**
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
from codex_r10_neutral import CACHE  # noqa: E402
from codex_routes_rerun import build_fund, route_scores  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR",
                      "/home/user/oxq-panel-0828/oxq_stock_market_fixed")
OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
POOLCSV = ("/home/user/open-xquant/examples/ashare-bull-stock-study/results/"
           "codex_cross_check/pool_20260828.csv")
CODEXX = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
          "3bcdce21-____20260831_Claude_____RPS501.xlsx")


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    d = {k: {} for k in ("close", "raw_close", "is_st", "is_suspended",
                         "listed_days", "volume")}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=list(d))
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in d:
            d[k][c] = x[k]
    cldf = pd.DataFrame(d["close"]).sort_index()
    idx, ns = cldf.index, len(codes)
    nt = len(idx)
    assert (nt, ns) == (3316, 5232), f"锚点E1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点E1a 末日 {idx[-1].date()}"
    print(f"锚点E1a ✓ {(nt, ns)} 末日 {idx[-1].date()} ({time.time()-t0:.0f}s)",
          flush=True)

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=codes).fillna(f).to_numpy()
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)     # 用户规则5
    ld = al("listed_days", 0)
    ldf = pd.DataFrame(ld).replace(0, np.nan).ffill().fillna(0).to_numpy()
    okm = (~al("is_st", True).astype(bool) & ~al("is_suspended", True).astype(bool)
           & (ldf >= 250) & (al("volume", 0) > 0) & np.isfinite(cl))
    rawm = pd.DataFrame(d["raw_close"]).sort_index().reindex(
        index=idx, columns=codes)
    raw = rawm.where(rawm > 0).ffill().to_numpy(np.float32)
    print(f"末日合格 {int(okm[nt-1].sum()):,} 只 ({time.time()-t0:.0f}s)", flush=True)

    z = np.load(CACHE, allow_pickle=True)
    zc = list(z["codes"])
    okc = z["OK"]
    ssum = okc.sum(1)
    med = float(np.median(ssum[-250:]))
    clean = int(np.max(np.flatnonzero(ssum >= 0.9 * med)))

    def _pad(a, src):
        if a.shape[0] >= nt:
            return a[:nt]
        return np.vstack([a, np.repeat(a[src:src + 1], nt - a.shape[0], axis=0)])
    logcap, tmean = _pad(z["LOGCAP"], clean), _pad(z["TMEAN"], clean)
    zpos = {c: j for j, c in enumerate(zc)}
    cpos = {c: j for j, c in enumerate(codes)}
    zback = np.array([cpos.get(c, -1) for c in zc])
    gb = zback >= 0
    zok = np.zeros((nt, len(zc)), bool)
    zok[:, gb] = okm[:, zback[gb]]
    zcl = np.full((nt, len(zc)), np.nan, np.float64)
    zraw = np.full((nt, len(zc)), np.nan, np.float32)
    zcl[:, gb] = cl[:, zback[gb]]
    zraw[:, gb] = raw[:, zback[gb]]
    fm, abad = build_fund(zc, idx)
    assert abad == 0, "锚点E1b TTM 恒等式不过"
    print(f"锚点E1b ✓ TTM 违例 0 ({time.time()-t0:.0f}s)", flush=True)

    tl = nt - 1
    e = np.flatnonzero(zok[tl] & np.isfinite(logcap[tl]) & np.isfinite(tmean[tl]))
    res = {}
    for name, mode in (("R09", "raw"), ("R08", "raw"), ("R08", "qfq")):
        v = np.full(len(zc), np.nan)
        v[e] = route_scores(name, tl, e, fm, zcl, zraw, logcap, tmean, mode)
        res[(name, mode)] = v
        print(f"  {name}/{mode}:横截面 {len(e):,} 只,有值 "
              f"{int(np.isfinite(v).sum()):,} 只", flush=True)

    pool = pd.read_csv(POOLCSV, dtype={"股票代码": str}, encoding="utf-8-sig")
    pool["股票代码"] = pool["股票代码"].str.zfill(6)
    his = pd.read_excel(CODEXX, sheet_name="全部清单", header=1)
    his["股票代码"] = his["股票代码"].astype(str).str.zfill(6)
    his = his[["股票代码", "R08价值分", "R09核心质量分", "信号类型"]].rename(
        columns={"R08价值分": "他R08", "R09核心质量分": "他R09", "信号类型": "他信号"})

    rows = []
    for c in pool["股票代码"]:
        j = zpos.get(c, -1)
        rows.append({"股票代码": c,
                     "我R09": res[("R09", "raw")][j] if j >= 0 else np.nan,
                     "我R08_真实口径": res[("R08", "raw")][j] if j >= 0 else np.nan,
                     "我R08_前复权口径": res[("R08", "qfq")][j] if j >= 0 else np.nan})
    m = pool[["股票代码", "股票名称", "信号类型", "R09核心质量分"]].merge(
        pd.DataFrame(rows), on="股票代码").merge(his, on="股票代码", how="left")

    a, b = m["R09核心质量分"].to_numpy(float), m["我R09"].to_numpy(float)
    g = np.isfinite(a) & np.isfinite(b)
    dmax = float(np.max(np.abs(a[g] - b[g]))) if g.sum() else 0.0
    ok_c = dmax < 1e-9
    print(f"\n锚点E1c R09 与已落库 pool_20260828.csv:可比 {int(g.sum())} 只,"
          f"最大绝对差 {dmax:.3e} {'✓' if ok_c else '✗ 本节作废'}", flush=True)
    if not ok_c:
        return

    def cmp_(col_me, col_his, tag):
        x, y = m[col_me].to_numpy(float), m[col_his].to_numpy(float)
        gg = np.isfinite(x) & np.isfinite(y)
        if gg.sum() < 3:
            print(f"  [{tag}] 可比不足 {int(gg.sum())} 只")
            return {"对比": tag, "可比只数": int(gg.sum())}
        dv = np.abs(x[gg] - y[gg])
        rho = pd.Series(x[gg]).corr(pd.Series(y[gg]), method="spearman")
        r = {"对比": tag, "可比只数": int(gg.sum()), "中位绝对差": float(np.median(dv)),
             "绝对差<0.05占比": float((dv < 0.05).mean()),
             "绝对差<0.10占比": float((dv < 0.10).mean()),
             "最大绝对差": float(dv.max()), "Spearman_ρ": float(rho),
             "我有值": int(np.isfinite(x).sum()), "他有值": int(np.isfinite(y).sum())}
        print(f"  [{tag}] 可比 {r['可比只数']} 只;中位|差| {r['中位绝对差']:.4f};"
              f"|差|<0.05 {r['绝对差<0.05占比']:.1%};ρ {rho:.4f};"
              f"最大|差| {r['最大绝对差']:.4f}")
        return r

    print("\n" + "=" * 90 + "\nE2 逐只比对\n" + "=" * 90)
    out = [cmp_("我R09", "他R09", "R09 核心质量分"),
           cmp_("我R08_真实口径", "他R08", "R08(我用真实不复权价)vs 他"),
           cmp_("我R08_前复权口径", "他R08", "R08(我改用前复权价)vs 他")]
    x = m["我R08_真实口径"].to_numpy(float)
    y = m["我R08_前复权口径"].to_numpy(float)
    gg = np.isfinite(x) & np.isfinite(y)
    print(f"\n  [价格口径本身的影响] 我自己两套口径:可比 {int(gg.sum())} 只;"
          f"中位|差| {np.median(np.abs(x[gg]-y[gg])):.4f};"
          f"ρ {pd.Series(x[gg]).corr(pd.Series(y[gg]), method='spearman'):.4f}")
    out.append({"对比": "价格口径影响(我真实 vs 我前复权)", "可比只数": int(gg.sum()),
                "中位绝对差": float(np.median(np.abs(x[gg] - y[gg]))),
                "Spearman_ρ": float(pd.Series(x[gg]).corr(pd.Series(y[gg]),
                                                          method="spearman"))})
    print("\n缺失情况(662 池内):")
    for c in ("我R09", "他R09", "我R08_真实口径", "他R08"):
        print(f"  {c:<16} 有值 {int(m[c].notna().sum()):>3} / {len(m)}")
    q = m[m["信号类型"] == "强确认"].sort_values("我R09", ascending=False)
    print("\n13 只强确认在两个分数上的取值:")
    print(q[["股票代码", "股票名称", "我R09", "他R09", "我R08_真实口径",
             "我R08_前复权口径", "他R08"]].to_string(index=False, na_rep="—"))
    m.to_csv(f"{OUT}/pool_r08_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(out).to_csv(f"{OUT}/pool_r08_compare_summary.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n完成 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
