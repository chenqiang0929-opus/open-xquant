"""平台「强势日」用 RPS50 还是 RPS60:直接量差在哪,不设通过/不通过判据。

来由
----
Codex 交来的 `weekly_signals/weekly_engine.py` 里 `platform_state()` 用的是
`ranks[50].ge(90)` 作强势日,而本地 `consolidation_screener.load_panel()` 用的是
`RPS60 > 90`。在他那 662 只次新股池上,同一天(2026-08-28)他出「平台观察」2 只、
我出 19 只。我在 `codex_note_20260831.md` 里把原因列成两个候选而没有分辨:
  (a) 强势日尺子不同;(b) 他面板从 2021-08-30 起、早期窗口被截断。
本脚本只解 (a):**在同一张面板、同一套 legacy 阈值下,只把强势日矩阵换掉。**

口径(除强势日外一律不动)
------------------------
- 三项阈值仍是 legacy 绝对值:缩量比 < 0.80、收敛比 < 0.80、深度 ≤ 0.352;
- 调整天数 ≥ 15、强势日回看 ≤ 250、触线日要求 MA100 向上,全部复用 `vec_screen`;
- 两条腿唯一的差别:`strong = (RPS60 > 90)` vs `strong = (RPS50 >= 90)`。
  (>90 与 >=90 的差别一并保留 —— 那是两边源码里各自写着的样子,不做人为对齐。)

锚点(不过则本次结果作废)
------------------------
A. 面板 (3316, 5232);
B. 末日 = 2026-08-28;
C. RPS60 腿在 662 池内的「平台观察」必须复现 `pool_20260828.csv` 里的 **19 只**,
   且 RPS60 腿的宇通 600066 锚点必须复现 **42 天 / 首次 2023-10-17 / 最后 2024-01-09**
   —— 复现不了说明我这次的调用与出表时的调用不是同一条路径,结论作废。
D. 自算的 `RPS60 > 90` 强势日矩阵必须与 `load_panel` 返回的**逐点相等** ——
   这是为了保证两条腿唯一的差别是窗口长度(50 vs 60),不是缺失处理。
   **这条断言连抓了我自己的两个错,两版结果都已作废:**
   **(1)** 第一版 RPS50 腿写了 `fill_method=None`,而 RPS60 腿走的是 pandas 默认的
   `'pad'` —— 两条腿的缺失处理不同,比较根本不成立;
   **(2)** 第二版补了 `fill_method` 却把 RPS 挪到**剔除 510300 之后**才算,
   截面少一列,自算 RPS60 与 `load_panel` 差 358 点(1,217,240 vs 1,217,598)。
   RPS 是 `axis=1` 的横截面分位,**必须在剔除 510300 之前算**。

**不设通过/不通过判据**:本节是把一个已登记的候选原因量出来,不是假设检验。
**不构成任何买入建议。**
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from consolidation_screener import (  # noqa: E402
    THR_ATR,
    THR_DEPTH,
    THR_SHRINK,
    load_panel,
)
from platform_pivot import vec_screen  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR", "/home/user/oxq-panel-0828/oxq_stock_market_fixed")
OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
POOLCSV = ("/home/user/open-xquant/examples/ashare-bull-stock-study/results/"
           "codex_cross_check/pool_20260828.csv")


def hit_of(cl, frames, strong, ma100, idx, codes, okm, tag):
    t0 = time.time()
    ts_a, adj_a, dep, shr, cnv, phi, plo = vec_screen(cl, frames, strong, ma100, idx, codes)
    hit3 = ((shr < THR_SHRINK) & (cnv < THR_ATR) & (dep <= THR_DEPTH) & (adj_a >= 0))
    same = np.zeros(hit3.shape, bool)
    same[1:] = ts_a[1:] == ts_a[:-1]
    up_prev = np.full(hit3.shape, np.nan, np.float64)
    up_prev[1:] = np.where(same[1:], phi[:-1], np.nan)
    brk = hit3 & okm & np.isfinite(up_prev) & (cl > up_prev)
    print(f"  [{tag}] 强势日 {int(strong.sum()):,} 点;三条全中 {int((hit3 & okm).sum()):,};"
          f"突破 {int(brk.sum()):,} ({time.time()-t0:.0f}s)", flush=True)
    return hit3, brk, ts_a, adj_a


def main():
    t0 = time.time()
    cldf, frames, strong60, ma100 = load_panel(DATA)
    # 【顺序要紧】RPS 是 axis=1 的横截面分位,所以两把尺子都必须在**剔除 510300 之前**
    # 算,和 load_panel 的截面完全一样(5,233 列);算完再一起把该列删掉。
    # 第二版脚本先删列再算 RPS50,截面少了一列,锚点D 立刻不过(1,217,240 vs 1,217,598)。
    with warnings.catch_warnings():
        # pandas 2.3 的 pct_change 默认 fill_method='pad'(仅弃用警告,行为仍是 pad),
        # load_panel 用的就是默认值,这里显式写出来以免将来 pandas 改默认值时静默分叉。
        warnings.simplefilter("ignore", FutureWarning)
        clw = cldf.where(cldf > 0)
        rps60 = clw.pct_change(60, fill_method="pad").rank(axis=1, pct=True) * 100
        rps50 = clw.pct_change(50, fill_method="pad").rank(axis=1, pct=True) * 100
    chk = (rps60 > 90).to_numpy()
    assert np.array_equal(chk, strong60), (
        f"锚点D:自算 RPS60 强势日与 load_panel 不一致 "
        f"{int(chk.sum())} vs {int(strong60.sum())}")
    strong50 = (rps50 >= 90).to_numpy()
    if "510300" in cldf.columns:
        keep = [i for i, c in enumerate(cldf.columns) if c != "510300"]
        cldf = cldf.drop(columns=["510300"])
        strong60 = strong60[:, keep]
        strong50 = strong50[:, keep]
        ma100 = ma100.drop(columns=["510300"])
    idx = cldf.index
    codes = list(cldf.columns)
    nt, ns = cldf.shape
    assert (nt, ns) == (3316, 5232), f"锚点A {cldf.shape}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点B {idx[-1].date()}"
    print(f"面板 {cldf.shape};末日 {idx[-1].date()} ({time.time()-t0:.0f}s)", flush=True)

    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)   # 用户规则5:退市 ffill
    # 合格掩码:与出表脚本同一条件的可用子集(此处只需成交与价格有效)
    okm = np.isfinite(cl)

    print(f"强势日点数:RPS60>90 {int(strong60.sum()):,}  RPS50>=90 {int(strong50.sum()):,}"
          f"(同截面 5,233 列、同 fill_method='pad';锚点D 已过)", flush=True)

    res = {}
    for tag, sm in (("RPS60>90(我)", strong60), ("RPS50>=90(他)", strong50)):
        res[tag] = hit_of(cl, frames, sm, ma100, idx, codes, okm, tag)

    with open(POOLCSV, encoding="utf-8-sig") as f:
        pool = [r["股票代码"] for r in csv.DictReader(f)]
    pos = {c: j for j, c in enumerate(codes)}
    pj = [pos[c] for c in pool if c in pos]
    tl = nt - 1
    print(f"\n池 {len(pool)} 只,面板内 {len(pj)} 只;观察日 {idx[tl].date()}")
    rows = []
    for tag, (hit3, brk, ts_a, adj_a) in res.items():
        w = [c for c in pool if c in pos and hit3[tl, pos[c]] and okm[tl, pos[c]]]
        b = [c for c in pool if c in pos and brk[tl, pos[c]]]
        print(f"  [{tag}] 池内平台观察 {len(w)} 只、平台突破 {len(b)} 只")
        rows.append({"尺子": tag, "全市场三条全中": int((hit3[tl] & okm[tl]).sum()),
                     "池内平台观察": len(w), "池内平台突破": len(b),
                     "池内清单": "、".join(w)})
    # 逐只:RPS60 腿命中的 19 只,在 RPS50 腿上卡在哪一步
    h60, _, ts60, adj60 = res["RPS60>90(我)"]
    h50, _, ts50, adj50 = res["RPS50>=90(他)"]
    det = []
    for c in pool:
        if c not in pos:
            continue
        j = pos[c]
        if not (h60[tl, j] and okm[tl, j]):
            continue
        why = "同样命中" if h50[tl, j] else (
            "RPS50 腿近250日无强势日" if ts50[tl, j] < 0 else
            f"RPS50 腿最近强势日更近(调整 {int(adj50[tl, j])} 日<15)"
            if adj50[tl, j] < 15 else "三项阈值不过")
        det.append({"代码": c, "RPS60腿_强势日": str(idx[int(ts60[tl, j])].date()),
                    "RPS60腿_调整天数": int(adj60[tl, j]),
                    "RPS50腿_强势日": (str(idx[int(ts50[tl, j])].date())
                                       if ts50[tl, j] >= 0 else ""),
                    "RPS50腿_调整天数": int(adj50[tl, j]), "RPS50腿结果": why})
    # 宇通锚点:第八七/一五五节的 42 天 / 首次 2023-10-17 / 最后 2024-01-09 是在
    # RPS60 尺子下量的。这里把同一把尺子和 RPS50 尺子并排量一遍 ——
    # 换尺子后这三个数变成什么,是「要不要统一到 RPS50」的决定性依据。
    if "600066" in pos:
        j = pos["600066"]
        a = int(np.searchsorted(idx, pd.Timestamp("2023-01-01")))
        b = int(np.searchsorted(idx, pd.Timestamp("2024-12-31")))
        for tag, (hit3, _, _, _) in res.items():
            d = np.flatnonzero(hit3[a:b, j] & okm[a:b, j]) + a
            rows.append({"尺子": tag + " / 宇通600066锚点",
                         "全市场三条全中": len(d), "池内平台观察": "", "池内平台突破": "",
                         "池内清单": (f"三条全中 {len(d)} 天;首次 {idx[d[0]].date()};"
                                      f"最后 {idx[d[-1]].date()}") if len(d)
                                     else "区间内无三条全中"})
            print(f"  [{tag}] 宇通 600066(2023-01→2024-12):三条全中 {len(d)} 天"
                  + (f";首次 {idx[d[0]].date()};最后 {idx[d[-1]].date()}" if len(d) else ""))
    pd.DataFrame(rows).to_csv(f"{OUT}/strongday_rps50_vs_rps60.csv", index=False,
                              encoding="utf-8-sig")
    d = pd.DataFrame(det)
    d.to_csv(f"{OUT}/strongday_rps50_vs_rps60_detail.csv", index=False, encoding="utf-8-sig")
    print("\n逐只:RPS60 腿命中的这些股票在 RPS50 腿上的去向")
    print(d.to_string(index=False))
    print("\n去向汇总:", dict(d["RPS50腿结果"].value_counts()) if len(d) else {})
    print(f"\n完成 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
