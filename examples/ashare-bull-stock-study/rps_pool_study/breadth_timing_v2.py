"""第一八二节 事前登记:市场广度当择时变量,值不值(结果未跑)。

起因
----
第一七六节量出:市场广度对平台突破的**绝对**收益跨度 7.66pp(留出段 60 日),
但同日市值/行业中性化后基本被差掉(Q5 仅 +1.60pp、p=0.0647,不过判据)。
第一七七节量出:广度与突破**发生率**是 8.6 倍的关系(留出段 2.51% → 21.60%)。

**这两条合起来指向的是「择时」而不是「选股」。** 第一七八节 C2 登记了这一条。

规则(跑之前写死,零可调阈值)
------------------------------
- 市场日收益 `m[t]` = 当日全体可交易股票的**等权**日收益
  (退市股按最后有效价 ffill 参与,绝不剔除 —— 用户规则 5);
- 广度 `b[t]` = 可交易股票中「收盘 ≥ 自身 MA100」的占比
  (与第一七六/一七七节**逐字同源**);
- 择时规则 Rk(k = 1..4):**若 `b[t−1] ≥ 门槛k` 则第 t 日满仓,否则空仓(收益 0)**
  —— 用**前一日**的广度,不含当日信息;
- 门槛直接沿用第一七六节训练段的四个分位边界,**不重新估计**:
  **0.3991 / 0.5729 / 0.7390 / 0.9548**;
- 基准 = **恒定满仓**(每天都拿 `m[t]`);
- **不含任何交易成本** —— 这对择时规则是有利的假设,结论若不过则更硬;
  若过了,必须在正文写明「未计成本」这个前提。

零假设与 p
----------
**随机择时**:在同一段里随机抽**同样多**的在场日,走同样的复利,**500 次**;
p = (随机中 ≥ 策略的次数 + 1) / 501,**p 下限 1/501 = 0.002**。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
Q1 锚点(不过则本节作废)
   (a) 面板 (3316, 5232),末日 2026-08-28;
   (b) 广度序列自第 100 行起全部有限且落在 [0, 1];
   (c) **恒等断言**:基准年化用「逐日复利」与「等权收益序列复利」两条路算,
       两者之差 < 1e-10(第一六七节靠恒等断言抓到过我自己两个 bug)。

Q2 **主判据**(留出段 2022-01-01 起)
   **通过 ⟺ 四条规则中至少一条的年化收益 − 基准年化 ≥ +3.00pp 且单尾 p < 0.05。**
   门槛与第一五二/一五五/一六八/一七三/一七四/一七六/一八一节完全一致。
   **四条全部报告;若只有一条过,按 Bonferroni 用 α = 0.05/4 = 0.0125 复判,
   两个结论都写。**

Q3 描述(必报,不参与判定):训练段同表(只报数)、各规则的在场天数占比、
   最大回撤、以及**两段的参数排序是否一致**(第一七五节用过这个诊断)。

事前提醒(登记在此以便被证伪)
------------------------------
第一七七节 P1 已经量出:**广度与突破发生率在很大程度上是同义反复**
(广度量「多少股票站上 MA100」,突破量「谁在创新高」)。
所以这条大概率是在测**市场择时本身**,不是在测形态。
**若判据通过,必须能与第一二〇节已有的择时结论对得上才算数;
对不上就说明是我这次的实现有问题,不是发现了新东西。**

**本文件不构成任何投资建议。**
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
from codex_r10_neutral import SEED  # noqa: E402
from panel_cache import cached  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR",
                      "/home/user/oxq-panel-0828/oxq_stock_market_fixed")
OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
QCUT = (0.3991, 0.5729, 0.7390, 0.9548)      # §176 训练段边界,原样沿用
NDRAW = 500


def ann_from(x, n):
    return x ** (250.0 / n) - 1.0 if n > 0 and x > 0 else np.nan


def mdd(cum):
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum / peak) - 1.0)


def main():  # noqa: PLR0915
    t0 = time.time()
    [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))]
    p = cached("panel", DATA, lambda: (_ for _ in ()).throw(
        AssertionError("锚点:panel 缓存必须已存在")))
    idx = pd.DatetimeIndex(p["idx"])
    cl, okm = p["cl"], p["okm"]
    nt, ns = cl.shape
    assert (nt, ns) == (3316, 5232), f"锚点Q1a {(nt, ns)}"
    assert str(idx[-1].date()) == "2026-08-28", f"锚点Q1a 末日 {idx[-1].date()}"
    print(f"锚点Q1a ✓ {(nt, ns)} 末日 {idx[-1].date()}", flush=True)

    ma100 = pd.DataFrame(cl).rolling(100).mean().to_numpy()
    with np.errstate(all="ignore"):
        b = (((cl >= ma100) & okm & np.isfinite(ma100)).sum(1)
             / np.maximum(okm.sum(1), 1))
    okb = np.isfinite(b[99:]).all() and (b[99:] >= 0).all() and (b[99:] <= 1).all()
    print(f"锚点Q1b 广度自第 100 行起有限且在 [0,1] {'✓' if okb else '✗ 作废'};"
          f"全段 min {b[99:].min():.3f} / 中位 {np.median(b[99:]):.3f} / "
          f"max {b[99:].max():.3f}", flush=True)
    if not okb:
        return

    with np.errstate(all="ignore"):
        r = cl[1:] / np.where(cl[:-1] > 0, cl[:-1], np.nan) - 1.0
    trad = okm[1:] & okm[:-1] & np.isfinite(r)
    cnt = trad.sum(1)
    m = np.full(nt, np.nan)
    m[1:] = np.where(cnt > 0, np.nansum(np.where(trad, r, 0.0), 1)
                     / np.maximum(cnt, 1), np.nan)
    del r, trad
    # 锚点Q1c 恒等:两条路算基准
    v = np.where(np.isfinite(m), m, 0.0)
    a1 = float(np.prod(1.0 + v[100:]))
    a2 = float(np.exp(np.sum(np.log1p(v[100:]))))
    print(f"锚点Q1c 恒等断言 |累乘 − exp∑log1p| = {abs(a1 - a2):.3e} "
          f"{'✓' if abs(a1 - a2) < 1e-10 * max(1, abs(a1)) else '✗ 作废'}",
          flush=True)

    split = int(np.searchsorted(idx.values, np.datetime64("2022-01-01")))
    segs = (("训练段13-21", slice(100, split)), ("留出段22-26", slice(split, nt)))
    rng = np.random.default_rng(SEED)
    rows, w = [], 100
    print(f"\n{'='*w}\n广度择时(用前一日广度;不含交易成本)\n{'='*w}")
    print(f"{'段':<12}{'规则':<22}{'在场占比':>9}{'年化':>9}{'基准年化':>10}"
          f"{'超额pp':>9}{'最大回撤':>10}{'基准回撤':>10}{'p':>8}")
    for sn, sl in segs:
        vv = v[sl]
        n = len(vv)
        base_cum = np.cumprod(1.0 + vv)
        base_ann = ann_from(float(base_cum[-1]), n)
        for k, thr in enumerate(QCUT, 1):
            pos = np.zeros(nt, bool)
            pos[1:] = b[:-1] >= thr
            pk = pos[sl]
            nin = int(pk.sum())
            if nin < 30:
                continue
            cum = np.cumprod(1.0 + np.where(pk, vv, 0.0))
            a = ann_from(float(cum[-1]), n)
            nul = np.empty(NDRAW)
            for d in range(NDRAW):
                sel = np.zeros(n, bool)
                sel[rng.choice(n, nin, replace=False)] = True
                nul[d] = ann_from(float(np.prod(1.0 + np.where(sel, vv, 0.0))), n)
            pv = float((np.sum(nul >= a) + 1) / (NDRAW + 1))
            rec = {"段": sn, "规则": f"R{k} 广度≥{thr:.4f}", "k": k,
                   "在场占比": nin / n, "年化": a, "基准年化": base_ann,
                   "超额pp": (a - base_ann) * 100, "最大回撤": mdd(cum),
                   "基准回撤": mdd(base_cum), "p": pv}
            rows.append(rec)
            print(f"{sn:<12}{rec['规则']:<22}{nin/n:>9.1%}{a:>+9.2%}"
                  f"{base_ann:>+10.2%}{rec['超额pp']:>+9.2f}"
                  f"{rec['最大回撤']:>+10.1%}{rec['基准回撤']:>+10.1%}{pv:>8.4f}")

    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/breadth_timing_v2.csv", index=False, encoding="utf-8-sig")
    ho = d[d["段"] == "留出段22-26"]
    print(f"\n{'='*w}\nQ2 判定(留出段;门槛 +3.00pp 且 p<0.05)\n{'='*w}")
    npass, win = 0, None
    for _, y in ho.iterrows():
        ok = bool(y["超额pp"] >= 3.0 and y["p"] < 0.05)
        npass += ok
        if ok:
            win = y
        print(f"  {y['规则']}:{y['超额pp']:+.2f}pp、p={y['p']:.4f} "
              f"→ {'✓ 通过' if ok else '✗ 不通过'}")
    print(f"\nQ2 四条通过 {npass} 条。", end="")
    if npass == 1:
        print(f"唯一通过的是 {win['规则']},这是 4 选 1 的 best-of-N;"
              f"Bonferroni α=0.0125 复判:p={win['p']:.4f} "
              f"{'仍过' if win['p'] < 0.0125 else '不过'}。")
    else:
        print("无需 Bonferroni 复判。")
    tr = d[d["段"] == "训练段13-21"].sort_values("超额pp", ascending=False)["k"].tolist()
    hh = ho.sort_values("超额pp", ascending=False)["k"].tolist()
    print(f"\nQ3 参数排序:训练段 {tr} / 留出段 {hh} "
          f"→ {'一致' if tr == hh else '不一致'}")
    print("\n提醒(事前登记):第一七七节 P1 已量出广度与突破发生率在很大程度上"
          "是同义反复,本节大概率测的是市场择时本身,不是形态;"
          "若通过须能与第一二〇节的择时结论对上才算数。")
    print(f"落库 {OUT}/breadth_timing_v2.csv ({time.time()-t0:.0f}s)")
    print("本表是状态记录,不是买点,不构成任何投资建议。")


if __name__ == "__main__":
    main()
