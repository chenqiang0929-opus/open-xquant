"""§131 事前登记:两份研报公布的数字,拿普查清单逐条对账(结果未跑)。

用户问的四件事
--------------
① 两份研报的核心结论能否用来找牛股?② 研究结果是否正确?③ 能否复用?
④ **研报里提到的牛股涨幅,是否匹配股票数据清单?**

①②③ 第一二九节已经答过,本节不重答:
  Part C 充分性 C2 通过 0/2;Part D 广发四条 lift 全 < 1;
  Part D2 安信两条同时的 lift 2.80 在换成不带后见之明的时点后塌回 1.03。
**本节只做 ④** —— 这是第一二九节答不了的,因为当时没有 2010 起的普查清单。

要分清的两件事(本节的全部意义)
--------------------------------
**「研报的事实陈述是否复现」与「研报的推论是否成立」是两回事。**
第一二九节判的是后者(不成立)。本节判前者:
**研报公布的那些数字,在独立数据上算得出来吗?**
两者可以同时成立 —— 事实对、推论错,这正是这类研报最常见的形态。

数据
----
`quant-research-dev/research/bull-stock-census-2010-2025/`(只读,commit de07686)
+ 本项目面板(市值、前 20 日涨幅、净利同比)。第一三〇节已验:
跨源自然年收益中位绝对差 0.0537%,两边说的是同一件事。

口径差异,先说清楚
------------------
- 安信说的是「上涨前**市值**」,未注明总市值还是流通市值;
  **本节用流通市值**(面板 PIT 口径 float_mv),**若研报用的是总市值,占比会系统性偏移**,
  这一条在结论里必须写出来,不能当成研报错了。
- 安信的「一年三倍股」统计区间研报未明确;
  **本节用普查的年内低点→高点涨幅 ≥ 200%(即 3 倍),区间 2010–2025**,
  与研报区间不一定相同,同样标注。
- 「业绩增速」本面板只有净利同比(TTM 代理),**不是研报口径的当期营收/净利增速**,
  标注,不得当成同一个东西。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
H1 锚点(不过则本节作废)
   (a) 本项目面板 (3297, 5232);
   (b) 普查三张年度表 (code, year) 两两无交集,并集 = 3420;
   (c) **无前视**:所有「上涨前」的特征一律在 intra_trough_date **当日或之前**测,
       逐点断言取数位置 <= trough 位置。

H2 广发表 4 对账(逐只列出,**不设通过/不通过**)
   22 只里 20 只 A 股(颐海国际/澳优是港股),逐只查:
   (a) 是否出现在普查跨年 10 倍榜 `multi_year_5x_10x`(is_10x);
   (b) 普查该股的 trough→peak 区间与研报的「加速起点→市值最高日」区间
       **重叠天数 / 研报区间天数**;
   (c) 普查 max_multiple 与研报公布倍数的比值。
   **「对得上」定义(写死):区间重叠 ≥ 50% 且倍数比值 ∈ [0.5, 2]。**
   伊利股份加速起点 2008-10-27 早于普查起点 2010-01-01,**剔除并列名**。

H3 安信三条事实陈述的复现。**核心判据。** 三条各判各的。
   样本:普查 `intrayear_gt100` 中 intra_max_multiple >= 3.0 的记录(一年三倍股)。
   (a) **上涨前流通市值 ∈ [10, 50] 亿的占比**,研报称 63%。
       **复现 ⟺ 复算值 ∈ [53%, 73%](±10pp)。**
   (b) **净利同比(代理)> 300% 的占比**,研报称「不足 15%」。
       **复现 ⟺ 复算值 < 15%。**
   (c) **上涨前 20 个交易日涨幅 > −1% 组的平均涨幅最高**,
       研报称该组平均 427.74% 且显著高于其他组别。
       分组:(−∞,−20%]、(−20%,−10%]、(−10%,−1%]、(−1%,+∞)。
       **复现 ⟺ 「> −1%」组的平均 intra_max_return 是四组里最高的。**
   **H3 通过数按三条分别记,不合并成一个「研报对不对」。**

H4 一句必须回答的话(描述,不设阈值)
   把 H3 的样本按「上涨前流通市值 10–50 亿」切开,
   报**该条件下的一年三倍股占同期同市值区间全部股票的比例** ——
   即 P(一年三倍 | 市值 10–50 亿)。
   **这是把研报的 P(市值|三倍股) 翻成 P(三倍股|市值),两个数放在一起看。**

事前预测
--------
**本节不下预测**(第一一九节起的约定)。只登记判据。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;**不往 quant-research-dev 推任何东西**;
**不因为事实复现了就说研报能用来选股** —— 那是第一二九节已经判过的另一件事;
**不因为事实没复现就说研报错了** —— 先查是不是口径差异,查不出再说。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_replication import DATA  # noqa: E402
from davis_double_click import GF_TABLE  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
CENSUS = ("/home/user/quant-research-dev/research/"
          "bull-stock-census-2010-2025/data")
OUT = SP


def rd(name):
    x = pd.read_csv(f"{CENSUS}/{name}.csv", dtype={"code": str})
    x.columns = [c.lstrip("﻿") for c in x.columns]
    x["code"] = x["code"].str.zfill(6)
    return x


def main():  # noqa: PLR0915
    import glob

    import pyarrow.parquet as pq
    files = sorted(glob.glob(f"{DATA}/*.parquet"))
    codes = [os.path.basename(f)[:-8] for f in files
             if os.path.basename(f)[:-8] != "510300"]
    cl, mv = {}, {}
    for c in codes:
        cols = pq.ParquetFile(f"{DATA}/{c}.parquet").schema.names
        want = [w for w in ("close", "float_mv") if w in cols]
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=want)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        cl[c] = pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0)
        mv[c] = pd.to_numeric(x["float_mv"], errors="coerce").where(lambda s: s > 0)
    q = pd.DataFrame(cl).sort_index()
    m = pd.DataFrame(mv).sort_index().reindex(index=q.index, columns=q.columns)
    assert q.shape == (3297, 5232), f"锚点H1a {q.shape}"
    print(f"锚点H1a ✓ 面板 {q.shape}", flush=True)

    mm, ll, dd = rd("annual_gt100_main"), rd("annual_gt100_listing_year"), \
        rd("annual_gt100_delisted")
    kk = [set(zip(x.year, x.code, strict=True)) for x in (mm, ll, dd)]
    inter = (kk[0] & kk[1]) | (kk[0] & kk[2]) | (kk[1] & kk[2])
    tot = len(kk[0] | kk[1] | kk[2])
    assert not inter and tot == 3420, "锚点H1b"
    print(f"锚点H1b ✓ 三表两两交集 0,并集 {tot}", flush=True)

    idx = q.index
    ipos = pd.Index(idx)
    npos = len(idx)

    # ---------------- H2 广发表 4 对账 ----------------
    my_ = rd("multi_year_5x_10x")
    my_["trough_date"] = pd.to_datetime(my_["trough_date"])
    my_["peak_date"] = pd.to_datetime(my_["peak_date"])
    print("\n=== H2 广发表 4 vs 普查跨年倍数榜(对得上 = 区间重叠≥50% 且倍数比 ∈[0.5,2])===")
    print(f"{'名称':<9}{'代码':<8}{'在10倍榜':>8}{'研报倍数':>9}{'普查倍数':>10}"
          f"{'倍数比':>8}{'区间重叠':>9}  对得上")
    rows2, nok, ntest = [], 0, 0
    for nm, code, d0, d1, rep in GF_TABLE:
        if code is None:
            print(f"{nm:<9}{'—':<8}  港股,不在普查 A 股清单")
            rows2.append({"name": nm, "code": None, "status": "港股"})
            continue
        if pd.Timestamp(d0) < pd.Timestamp("2010-01-01"):
            print(f"{nm:<9}{code:<8}  加速起点 {d0} 早于普查起点 2010-01-01,剔除")
            rows2.append({"name": nm, "code": code, "status": "起点早于普查"})
            continue
        ntest += 1
        r = my_[my_.code == code]
        if not len(r):
            print(f"{nm:<9}{code:<8}{'不在榜':>8}")
            rows2.append({"name": nm, "code": code, "status": "不在跨年榜"})
            continue
        r = r.iloc[0]
        a0, a1 = pd.Timestamp(d0), pd.Timestamp(d1)
        ov = (min(a1, r.peak_date) - max(a0, r.trough_date)).days
        frac = max(ov, 0) / max((a1 - a0).days, 1)
        ratio = float(r.max_multiple) / (rep + 1.0)
        good = frac >= 0.50 and 0.5 <= ratio <= 2.0
        nok += int(good)
        print(f"{nm:<9}{code:<8}{'是' if r.is_10x else '否':>8}"
              f"{rep+1:>9.2f}{float(r.max_multiple):>10.2f}{ratio:>8.2f}"
              f"{frac:>9.0%}  {'✓' if good else '✗'}")
        rows2.append({"name": nm, "code": code, "status": "ok",
                      "is_10x": bool(r.is_10x), "研报倍数": rep + 1,
                      "普查倍数": float(r.max_multiple), "倍数比": ratio,
                      "区间重叠": frac, "对得上": good,
                      "普查trough": r.trough_date.date(),
                      "普查peak": r.peak_date.date()})
    print(f"H2 对得上 {nok}/{ntest}(逐只已列,本条不设通过/不通过)", flush=True)

    # ---------------- H3 安信三条 ----------------
    iy = rd("intrayear_gt100")
    iy["intra_trough_date"] = pd.to_datetime(iy["intra_trough_date"])
    tri = iy[iy.intra_max_multiple >= 3.0].copy()
    tri = tri[tri.code.isin(q.columns)]
    tp = ipos.get_indexer(tri.intra_trough_date, method="ffill")
    tri["tpos"] = tp
    tri = tri[(tri.tpos >= 20) & (tri.tpos < npos)]
    # 锚点 H1c:取数位置必须 <= trough 日
    bad = int((idx[tri.tpos.to_numpy()] > tri.intra_trough_date).sum())
    assert bad == 0, f"锚点H1c 前视 {bad} 条"
    print(f"\n锚点H1c ✓ 取数位置逐条 <= 低点日({len(tri):,} 条)", flush=True)

    cpos = {c: i for i, c in enumerate(q.columns)}
    ja = tri.code.map(cpos).to_numpy()
    ta = tri.tpos.to_numpy()
    qa, ma = q.to_numpy(), m.to_numpy()
    tri["mv_yi"] = ma[ta, ja] / 1e8
    tri["r20"] = qa[ta, ja] / qa[ta - 20, ja] - 1.0

    print(f"\n=== H3 安信三条事实陈述(样本:年内 ≥3 倍,{len(tri):,} 条)===")
    sub = tri[tri.mv_yi.notna()]
    pa = float(((sub.mv_yi >= 10) & (sub.mv_yi <= 50)).mean())
    h3a = 0.53 <= pa <= 0.73
    print(f"(a) 上涨前流通市值 10–50 亿占比 **{pa:.1%}**(研报 63%,判据 [53%,73%])"
          f" {'✓ 复现' if h3a else '✗ 未复现'}   n={len(sub):,}")

    from codex_routes_rerun import build_fund
    fm, abad = build_fund(list(q.columns), idx)
    assert abad == 0, "锚点H1a TTM"
    ni = fm["ni_ttm"]
    nip = np.roll(ni, 250, axis=0)
    with np.errstate(all="ignore"):
        yoy = ni / np.where(nip != 0, np.abs(nip), np.nan) - 1.0
    yoy[:250] = np.nan
    tri["yoy"] = yoy[ta, ja]
    sb = tri[tri.yoy.notna()]
    pb = float((sb.yoy > 3.0).mean())
    h3b = pb < 0.15
    print(f"(b) 净利同比(代理)>300% 占比 **{pb:.1%}**(研报「不足 15%」)"
          f" {'✓ 复现' if h3b else '✗ 未复现'}   n={len(sb):,}")

    sc = tri[tri.r20.notna()].copy()
    bins = [-np.inf, -0.20, -0.10, -0.01, np.inf]
    labs = ["≤−20%", "(−20%,−10%]", "(−10%,−1%]", "**>−1%**"]
    sc["grp"] = pd.cut(sc.r20, bins, labels=labs)
    g = sc.groupby("grp", observed=True).intra_max_return.agg(["mean", "median", "count"])
    top = g["mean"].idxmax()
    h3c = top == "**>−1%**"
    print("(c) 按上涨前 20 日涨幅分组的最终涨幅(研报称 >−1% 组平均 427.74% 且最高):")
    for k, r in g.iterrows():
        print(f"      {str(k):<14} 平均 {r['mean']:>8.1%}  中位 {r['median']:>8.1%}"
              f"  n={int(r['count']):>5,}")
    print(f"    最高组 = {top}  {'✓ 复现' if h3c else '✗ 未复现'}")
    print(f"\n**H3 复现 {sum([h3a,h3b,h3c])}/3**", flush=True)

    # ---------------- H4 反过来问 ----------------
    print("\n=== H4 把 P(市值|三倍股) 翻成 P(三倍股|市值)===")
    yrs = sorted(tri.year.unique())
    hit = tot_ = 0
    for y in yrs:
        sel = (idx.year == y)
        if not sel.any():
            continue
        t = int(np.flatnonzero(sel)[0])
        band = (ma[t] / 1e8 >= 10) & (ma[t] / 1e8 <= 50)
        tot_ += int(band.sum())
        cs = set(tri[(tri.year == y)].code)
        hit += int(sum(band[cpos[c]] for c in cs if c in cpos))
    print(f"  研报的方向 P(上涨前市值 10–50 亿 | 一年三倍股) = **{pa:.1%}**")
    print(f"  选股要的方向 P(一年三倍股 | 年初市值 10–50 亿) = **{hit/max(tot_,1):.2%}**"
          f"  ({hit:,} / {tot_:,} 股票-年)")

    pd.DataFrame(rows2).to_csv(f"{OUT}/report_vs_census_gf.csv", index=False,
                               encoding="utf-8-sig")
    tri.to_csv(f"{OUT}/report_vs_census_ax.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/report_vs_census_gf.csv, report_vs_census_ax.csv")


if __name__ == "__main__":
    main()
