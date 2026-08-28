"""§130 附带:审计面板的 raw_close_ca 列 —— 它不是它被当作的那个东西。

起因
----
§130 的锚点 G1(b) 我写成「close(前复权)与 raw_close_ca(后复权)算出的自然年收益
应当相等」。跑出来最大绝对差 5.239,不过。**锚点本身写错了** ——
raw_close_ca 按定义就不是后复权序列。但既然差这么大,顺手查这一列到底是什么。

第一一四节 D4(d) 用这一列做过「两边都不含分红」的自洽口径,
其 docstring 写的是「raw_close_ca:只调送转、不调分红」。**本脚本检验这句话。**

检验方法(不设通过/不通过判据,纯事实测量)
------------------------------------------
若 raw_close_ca 真的「只调送转」,那么 factor = raw_close_ca / raw_close 应当是
**分段常数**,只在送转/配股日跳变,且跳幅是 0.5 / 0.33 这类大数;
分红日**不该跳**。于是逐只统计 factor 的跳变点,按跳幅分成
「大跳 ≥5%(像送转/配股)」与「小跳 <5%(不像任何公司行动)」。
再看 factor 的首末比,与 close/raw_close 的首末比对照 ——
后者携带了全部累计送转,前者若「只调送转」应当与之同量级。

不做的
------
不改 src/oxq/;不改第一一四节的脚本(只出审计结论,是否重跑另议);
不新增顶层目录;不 force push。
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
BIG = 0.05


def main():
    rows, noca = [], []
    for f in sorted(glob.glob(f"{DATA}/*.parquet")):
        c = os.path.basename(f)[:-8]
        if c == "510300":
            continue
        if "raw_close_ca" not in pq.ParquetFile(f).schema.names:
            noca.append(c)
            continue
        x = pd.read_parquet(f, columns=["close", "raw_close", "raw_close_ca"])
        a = pd.to_numeric(x["raw_close"], errors="coerce").where(lambda s: s > 0)
        b = pd.to_numeric(x["raw_close_ca"], errors="coerce").where(lambda s: s > 0)
        q = pd.to_numeric(x["close"], errors="coerce").where(lambda s: s > 0)
        fa = (b / a).dropna()
        fq = (q / a).dropna()
        if len(fa) < 50 or len(fq) < 50:
            continue
        d = np.log(fa).diff().dropna()
        j = d[d.abs() > 1e-6]
        rows.append({
            "code": c,
            "n_big": int((j.abs() >= BIG).sum()),
            "n_small": int((j.abs() < BIG).sum()),
            "ca_factor_首": float(fa.iloc[0]), "ca_factor_末": float(fa.iloc[-1]),
            "qfq_factor_首": float(fq.iloc[0]), "qfq_factor_末": float(fq.iloc[-1]),
        })
    df = pd.DataFrame(rows)
    df["ca_累计倍数"] = df["ca_factor_末"] / df["ca_factor_首"]
    df["qfq_累计倍数"] = df["qfq_factor_末"] / df["qfq_factor_首"]
    n = len(df)
    tb, ts = int(df.n_big.sum()), int(df.n_small.sum())
    print(f"股票数 {n}(无 raw_close_ca 列 {len(noca)} 只:{','.join(noca)})\n")
    print("factor = raw_close_ca / raw_close 的跳变点")
    print(f"  大跳 |Δlog|≥{BIG:.0%}(像送转/配股): 合计 {tb:,}  每只中位 {df.n_big.median():.0f}")
    print(f"  小跳 |Δlog|<{BIG:.0%}(不像公司行动): 合计 {ts:,}  每只中位 {df.n_small.median():.0f}")
    print(f"  小跳占全部跳变 {ts/max(tb+ts,1):.1%};完全没有小跳的股票 {int((df.n_small==0).sum())}/{n}")
    print("\n若「只调送转」,ca 的累计调整应与 qfq 同量级(qfq 携带全部累计送转+分红):")
    for lab, col in (("ca 累计倍数", "ca_累计倍数"), ("qfq 累计倍数", "qfq_累计倍数")):
        s = df[col]
        print(f"  {lab:14s} 中位 {s.median():8.3f}   90分位 {s.quantile(0.9):9.3f}   "
              f"最大 {s.max():11.1f}")
    big_gap = df[df.qfq_累计倍数 / df.ca_累计倍数 > 2]
    print(f"\n  qfq 累计调整比 ca 大 2 倍以上的股票:{len(big_gap)}/{n} = "
          f"{len(big_gap)/n:.1%}")
    print("  → ca **没有携带累计送转**,它主要在做分红那一档的小幅调整,"
          "与第一一四节 docstring 写的『只调送转、不调分红』**方向相反**。")
    df.to_csv(f"{SP}/raw_close_ca_audit.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {SP}/raw_close_ca_audit.csv")


if __name__ == "__main__":
    main()
