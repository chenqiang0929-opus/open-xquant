"""§157 描述性分析:被选中的「图形好坏」到底和结果有没有关系。

**本脚本不设通过/不通过判据** —— 它回答用户的一个观察:
「浙江东日、德福科技、新金路走势都是非常经典的平台,但越往后的股票图形越来越差」。

问题的形式是:清单按**实现收益**降序排,那么排在后面的,
究竟是**形态本身更差**,还是只是**结果更差**?

做法:把第一五六节的 545 笔实际成交按实现收益分成五档,
逐个比较形态字段(深度/缩量比/收敛比/调整天数/买入日波动率),
并算各字段与实现收益的 Spearman 相关。**只描述,不判定。**
"""

from __future__ import annotations

import os

import pandas as pd

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
FORM = ["深度", "缩量比", "收敛比", "调整天数", "止损距离", "止损σ倍数",
        "买入日日波动率", "流通市值亿"]


def main():
    b = pd.read_csv(f"{OUT}/platform_trades.csv", dtype={"代码": str})
    a = pd.read_csv(f"{OUT}/platform_buypoints.csv", dtype={"代码": str})
    b["代码"] = b["代码"].str.zfill(6)
    a["代码"] = a["代码"].str.zfill(6)
    m = b.merge(a.rename(columns={"买点日": "买入日"})[["买入日", "代码", *FORM]],
                on=["买入日", "代码"], how="left")
    assert m["收敛比"].notna().all(), "形态字段未能全部匹配"
    m["分档"] = pd.qcut(m["收益"].rank(method="first"), 5,
                       labels=["Q1最差", "Q2", "Q3", "Q4", "Q5最好"])
    g = m.groupby("分档", observed=True).agg(
        笔数=("收益", "size"), 收益中位=("收益", "median"),
        深度=("深度", "median"), 缩量比=("缩量比", "median"),
        收敛比=("收敛比", "median"), 调整天数=("调整天数", "median"),
        持有日=("持有交易日", "median"), 买入日波动率=("买入日日波动率", "median"),
        市值亿=("流通市值亿", "median"))
    w = 96
    print(f"{'='*w}\n545 笔按实现收益分五档 —— 形态字段有没有差别\n{'='*w}")
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n{'='*w}\nSpearman 相关(形态字段 vs 实现收益)\n{'='*w}")
    cor = {c: float(m[c].corr(m["收益"], method="spearman")) for c in FORM}
    for c, r in sorted(cor.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {c:<14}{r:+.4f}")
    print(f"\n{'='*w}\n各档的卖出原因构成\n{'='*w}")
    ct = pd.crosstab(m["分档"], m["卖出原因"], normalize="index")
    print(ct.to_string(float_format=lambda v: f"{v:.1%}"))
    g.to_csv(f"{OUT}/platform_form_vs_outcome.csv", encoding="utf-8-sig")
    pd.Series(cor, name="spearman").to_csv(
        f"{OUT}/platform_form_corr.csv", encoding="utf-8-sig")
    print(f"\n落库 {OUT}/platform_form_vs_outcome.csv、platform_form_corr.csv")


if __name__ == "__main__":
    main()
