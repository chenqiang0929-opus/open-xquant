"""§130 事前登记 + 实现:与 quant-research-dev 的 A股牛股普查(2010–2025)交叉核对。

数据来源(只读,绝不回推)
------------------------
`quant-research-dev/research/bull-stock-census-2010-2025/`(commit de07686)
口径见其 README:自然年收益 = 本年最后交易日**后复权**收盘 / 上年最后交易日后复权收盘 − 1;
缺上年收盘时用本年首个交易日。主样本 = 剔除上市首年与退市股。

为什么先做这一步
----------------
这份普查覆盖 5,285 个行情文件、2010–2025;本项目面板是 5,232 只、2013-01-04 起。
**在用别人的清单之前,必须先确认它和我的面板在重叠区间上说的是同一件事** ——
与第一一三节核对 Codex 归档同一规格。核对不过就不能拿它当基础。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
G1 锚点(不过则本节作废)
   (a) 本项目面板 (3297, 5232);
   (b) **【本条作废 —— 我把锚点写错了,第 17 次锚点设计错误】**
       原文写的是「前复权(close)与后复权(raw_close_ca)算出的自然年收益
       逐只逐年相等,最大绝对差 < 1e-6」。跑出来 **5.239,远远不过**。
       但**问题在我这条锚点,不在数据核对本身**:`raw_close_ca` 根本不是后复权序列,
       它按定义就与 close 不同,**这条恒等式无论数据好坏都不可能成立**。
       跑完顺手查了这个列到底是什么(见下方「顺带查出的问题」),留档不删。
       **作废的锚点不许拿来当通过项,也不许悄悄删掉。**
       本节的价格口径改由下面 (d) 来把关。
       另记:有 **6 只股票的文件没有 raw_close_ca 列**
       (000522/000602/000990/301512/603400/688663),已剔除并列名,
       不用 close 顶替(顶替会让恒等式自动成立);
   (d) **跨源价格一致性**(跑出任何结果之前追加的一条,属加严):
       本面板算出的自然年收益 与 普查自己公布的 annual_return,在双方都有的
       (code, year) 上比较,要求 **中位绝对差 < 0.5% 且 99 分位 < 5%**。
       这条比 (b) 更硬 —— (b) 只比本面板自己的两列,(d) 比的是两个独立数据源;
   (c) 普查表内部自洽:main / listing_year / delisted 三张表的
       (code, year) 两两无交集,且并集条数 = summary.json 的 annual_gt100_events。

G2 重叠区间一致性。**核心判据。** 区间 2013–2025,只比**双方都能算**的
   (code, year):即该代码在本面板中存在、且上年末与本年末收盘都有值。
   记普查集合 C、本面板集合 M。
   **G2 通过 ⟺ Jaccard(C, M) ≥ 0.95 且 双向差集里「非边界样本」为 0。**
   「边界样本」定义为 |年收益 − 100%| < 1%(四舍五入与末日选取都可能翻边),
   **这条在跑之前写死,跑完不许改。**

G3 差异归因(描述,不设阈值):把「仅普查」与「仅本面板」逐条打上原因标签
   (不在本面板 / 上市首年 / 已退市 / 边界 / 未归类),**「未归类」必须逐条列出**。

事前预测
--------
**本节不下预测**(第一一九节起已停止此类外推)。只登记判据。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;**不往 quant-research-dev 推任何东西**;
不因为普查更大就默认它更对 —— 差异要逐条归因,不能一句「口径不同」带过。
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
CENSUS = ("/home/user/quant-research-dev/research/"
          "bull-stock-census-2010-2025/data")
Y0, Y1 = 2013, 2025
BORDER = 0.01


def main():  # noqa: PLR0915
    t0 = time.time()
    files = sorted(glob.glob(f"{DATA}/*.parquet"))
    codes = [os.path.basename(f)[:-8] for f in files if
             os.path.basename(f)[:-8] != "510300"]
    qfq, hfq, noca = {}, {}, []
    for c in codes:
        cols = pq.ParquetFile(f"{DATA}/{c}.parquet").schema.names
        want = ["close"] + (["raw_close_ca"] if "raw_close_ca" in cols else [])
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=want)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        qfq[c] = pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0)
        if "raw_close_ca" in want:
            hfq[c] = pd.to_numeric(x["raw_close_ca"], errors="coerce").where(
                lambda s: s > 0)
        else:
            noca.append(c)
    q = pd.DataFrame(qfq).sort_index()
    h = pd.DataFrame(hfq).sort_index().reindex(index=q.index)
    assert q.shape == (3297, 5232), f"锚点G1a {q.shape}"
    print(f"锚点G1a ✓ 面板 {q.shape}  ({time.time()-t0:.0f}s)", flush=True)

    # 年末收盘(每年最后一个有值的交易日)
    yr = q.index.year
    ye_q = q.groupby(yr).apply(lambda g: g.ffill().iloc[-1])
    ye_h = h.groupby(yr).apply(lambda g: g.ffill().iloc[-1])
    # 只在「当年确有报价」的位置保留(ffill 只是为了跨停牌取末值)
    live = q.groupby(yr).apply(lambda g: g.notna().any())
    ye_q = ye_q.where(live)
    ye_h = ye_h.where(live)

    r_q = ye_q / ye_q.shift(1) - 1.0
    r_h = ye_h / ye_h.shift(1) - 1.0
    both = r_q.notna() & r_h.notna()
    dmax = float((r_q[h.columns] - r_h).abs().where(both).max().max())
    print(f"锚点G1b 复权恒等式 最大绝对差 {dmax:.3e} "
          f"{'✓' if dmax < 1e-6 else '✗ 作废'};"
          f"无 raw_close_ca 已剔除 {len(noca)} 只:{','.join(noca)}", flush=True)
    print("  ↑ 本条已作废(锚点写错:raw_close_ca 不是后复权序列),"
          "不作为通过/不通过依据,见 docstring", flush=True)

    mm = pd.read_csv(f"{CENSUS}/annual_gt100_main.csv", dtype={"code": str})
    ll = pd.read_csv(f"{CENSUS}/annual_gt100_listing_year.csv", dtype={"code": str})
    dd = pd.read_csv(f"{CENSUS}/annual_gt100_delisted.csv", dtype={"code": str})
    for x in (mm, ll, dd):
        x.columns = [c.lstrip("﻿") for c in x.columns]
        x["code"] = x["code"].str.zfill(6)
    key = lambda x: set(zip(x.year, x.code, strict=True))  # noqa: E731
    km, kl, kd = key(mm), key(ll), key(dd)
    inter = (km & kl) | (km & kd) | (kl & kd)
    tot = len(km | kl | kd)
    print(f"锚点G1c 三表两两交集 {len(inter)} 条 {'✓' if not inter else '✗'};"
          f"并集 {tot} 条 vs summary 3420 "
          f"{'✓' if tot == 3420 else '✗ 作废'}", flush=True)
    assert not inter and tot == 3420, "锚点G1c"

    allc = pd.concat([mm, ll, dd], ignore_index=True)
    allc = allc[(allc.year >= Y0) & (allc.year <= Y1)]
    rl = r_q.stack().rename("mine").reset_index()
    rl.columns = ["year", "code", "mine"]
    cmp_ = allc.merge(rl, on=["year", "code"], how="inner")
    ad = (cmp_["mine"] - cmp_["annual_return"]).abs()
    med, p99 = float(ad.median()), float(ad.quantile(0.99))
    g1d = med < 0.005 and p99 < 0.05
    print(f"锚点G1d 跨源价格一致性 n={len(cmp_):,} 中位绝对差 {med:.4%} "
          f"99分位 {p99:.4%} {'✓' if g1d else '✗ 作废'}", flush=True)
    assert g1d, "锚点G1d"

    # ---- G2:只比双方都能算的 (code, year) ----
    panel = set(q.columns)
    cal = {(y, c) for (y, c) in (km | kl | kd) if Y0 <= y <= Y1}
    rows = []
    for y in range(Y0, Y1 + 1):
        rr = r_q.loc[y]
        ok = rr.notna()
        for c in rr.index[ok]:
            rows.append((y, c, float(rr[c])))
    mine = pd.DataFrame(rows, columns=["year", "code", "ret"])
    comparable = set(zip(mine.year, mine.code, strict=True))
    sm = {(y, c) for y, c, r in rows if r > 1.0}
    sc = {k for k in cal if k in comparable}
    both_s, only_c, only_m = sm & sc, sc - sm, sm - sc
    jac = len(both_s) / max(len(sm | sc), 1)
    rmap = {(y, c): r for y, c, r in rows}
    lab = {}
    for k in only_c | only_m:
        r = rmap.get(k, np.nan)
        if k[1] not in panel:
            lab[k] = "不在本面板"
        elif abs(r - 1.0) < BORDER:
            lab[k] = "边界(|收益−100%|<1%)"
        elif k in kl:
            lab[k] = "普查标为上市首年"
        elif k in kd:
            lab[k] = "普查标为已退市"
        else:
            lab[k] = "未归类"
    nun = sum(v == "未归类" for v in lab.values())
    g2 = jac >= 0.95 and nun == 0
    print("\n=== G2 重叠区间一致性(2013–2025,双方都能算的样本)===")
    print(f"  可比 (code,year) {len(comparable):,};普查在可比集合内 {len(sc):,};"
          f"本面板 {len(sm):,}")
    print(f"  交集 {len(both_s):,}  仅普查 {len(only_c):,}  仅本面板 {len(only_m):,}")
    print(f"  **Jaccard {jac:.4f}**(判据 ≥0.95){'✓' if jac >= 0.95 else '✗'}"
          f"  未归类 {nun} 条{'✓' if nun == 0 else '✗'}")
    print(f"  **G2 {'通过' if g2 else '不通过'}**", flush=True)

    print("\n=== G3 差异归因 ===")
    ser = pd.Series(lab)
    if len(ser):
        for tag, n in ser.value_counts().items():
            side = pd.Series({k: ("仅普查" if k in only_c else "仅本面板")
                              for k in ser.index[ser == tag]}).value_counts()
            print(f"  {tag:24s} {n:5d}  ({', '.join(f'{a} {b}' for a, b in side.items())})")
    if nun:
        print("\n  未归类逐条:")
        for k in [k for k, v in lab.items() if v == "未归类"][:40]:
            print(f"    {k[0]} {k[1]}  本面板年收益 {rmap.get(k, float('nan')):+.2%}"
                  f"  {'仅普查' if k in only_c else '仅本面板'}")

    out = pd.DataFrame([{"year": k[0], "code": k[1], "side":
                         "仅普查" if k in only_c else "仅本面板",
                         "本面板年收益": rmap.get(k, np.nan), "原因": v}
                        for k, v in lab.items()]).sort_values(["原因", "year", "code"])
    out.to_csv(f"{SP}/census_crosscheck_diff.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n落库 {SP}/census_crosscheck_diff.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
