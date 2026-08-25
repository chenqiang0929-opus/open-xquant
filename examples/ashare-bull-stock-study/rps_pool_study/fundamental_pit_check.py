"""§115-A 财务面板的 PIT 对齐验证 —— 跑第二批因子之前的门。

为什么必须先验
--------------
Codex 的 R08/R09/R11/R12/R13 全部依赖财务数据;本项目 §97–§101 也在财务口径上
栽过一次(累计口径的同比必须对齐同一报告期,错了一轮才发现)。
财务数据最容易藏的错是**前视泄漏**:把 2024Q1 的净利润打戳在 2024-03-31,
而它实际要到 4 月底才公开 —— 回测就提前 30 天知道了业绩。

决定性的测法
------------
A 股法定披露窗口:年报 & 一季报 4/30 前、中报 8/31 前、三季报 10/31 前。
  · 真 PIT(按公告日前向填充)→ 变更日压在 4 / 8 / 10 月,且集中在下旬
  · 报告期末打戳(泄漏)      → 变更日压在 3/31、6/30、9/30、12/31
两种分布不可能混淆。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
E1 月份分布。统计全市场所有股票、所有财务列的「值发生变化」的交易日。
   E1 通过 ⟺ 落在 {4月, 8月, 10月} 的比例 ≥ 60%
            **且** 落在 {3月, 6月, 9月, 12月} 的比例 ≤ 10%。
   两条都满足才算过。

E2 月内位置。E1 的三个披露月里,变更日的**中位日**必须 ≥ 该月 20 日
   (4月 ≥ 4/20、8月 ≥ 8/20、10月 ≥ 10/20)。三个月都要满足。
   理由:法定截止日在月末,真实公告高度集中在截止日前一两周;
   若中位日落在月初,说明打戳时点比公告早。

E3 无早于报告期末的变更。取 net_income,统计变更日落在
   {1/1–3/30, 4/1–6/29, 7/1–9/29, 10/1–12/30} 之外**且早于对应报告期末**的比例。
   操作化:统计 3 月、6 月、9 月、12 月内变更占比(见 E1 第二条,此处不重复设阈)。

E1 与 E2 任一不过 → **判定财务面板不是 PIT 对齐,第二批因子(价值/质量/多因子)
不跑**,并在正文里明说跑不了,而不是跑一个好看的错数字。

锚点
----
A1 面板 (3297, 5217)。
A2 恒等式:随机财务列在**同一报告期内**不得出现两次以上变更。
   操作化:每只股票每年的变更次数中位数必须 ∈ [3, 5](一年四次披露,
   允许 ±1 的修正/追溯)。若面板是逐日插值或随机噪声,此项必炸。

事前预测
--------
P1 E1 会过。理由:§97–§101 用过这批列做同比,当时 `fundamental_yoy.py` 的
   报告期识别规则是「7~9月→中报;10~11月→三季报;1~5月内当年第1次→上年年报,
   第2次→本年一季报」—— 那套规则能跑通,说明变更日确实落在公告窗口而非报告期末。
P2 E2 会过,但 **4 月的中位日会明显晚于 8 月和 10 月**,因为 4 月要同时消化
   年报与一季报,且大量公司卡在 4/29–4/30 披露。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_replication import DATA  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
OUT = os.environ.get("OXQ_OUT_DIR", SP)
COLS = ["eps", "revenue", "net_income", "book_value_per_share", "roe",
        "operating_cash_flow"]
DISCLOSE, PERIOD_END = {4, 8, 10}, {3, 6, 9, 12}


def main():
    z = np.load(f"{SP}/codex_r10_matrices.npz", allow_pickle=True)
    codes = list(z["codes"])
    idx = pd.DatetimeIndex(z["idx"])
    assert (len(idx), len(codes)) == (3297, 5217), "锚点A1"
    print(f"锚点A1 ✓ 面板 {len(idx)}×{len(codes)}")

    t0 = time.time()
    chg_dates = {c: [] for c in COLS}
    per_year = []
    for n, s in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{s}.parquet", columns=COLS)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for c in COLS:
            v = pd.to_numeric(x[c], errors="coerce").to_numpy(float)
            prev, cur = v[:-1], v[1:]
            # NaN != 0 在 numpy 里是 True —— §98 栽过一次,这里显式排除
            m = np.isfinite(prev) & np.isfinite(cur) & (cur != prev)
            d = x.index[1:][m]
            chg_dates[c].append(d)
            if c == "net_income" and len(d):
                per_year.append(pd.Series(1, index=d).groupby(d.year).sum())
        if (n + 1) % 1500 == 0:
            print(f"  {n+1}/{len(codes)}  ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    print(f"\n{'列':24s} {'变更数':>9s} {'4/8/10月':>9s} {'3/6/9/12月':>11s} "
          f"{'4月中位日':>9s} {'8月中位日':>9s} {'10月中位日':>10s}")
    for c in COLS:
        d = pd.DatetimeIndex(np.concatenate([a.values for a in chg_dates[c] if len(a)]))
        mo = d.month
        f_dis = float(np.isin(mo, list(DISCLOSE)).mean())
        f_pe = float(np.isin(mo, list(PERIOD_END)).mean())
        med = {m: float(np.median(d[mo == m].day)) if (mo == m).any() else np.nan
               for m in (4, 8, 10)}
        rows.append({"col": c, "n_change": len(d), "frac_disclose": f_dis,
                     "frac_period_end": f_pe, **{f"med_day_{m}": med[m] for m in (4, 8, 10)}})
        print(f"{c:24s} {len(d):9,d} {f_dis:8.1%} {f_pe:10.1%} "
              f"{med[4]:9.0f} {med[8]:9.0f} {med[10]:10.0f}")

    df = pd.DataFrame(rows)
    e1 = bool((df["frac_disclose"] >= 0.60).all() and (df["frac_period_end"] <= 0.10).all())
    e2 = bool((df[["med_day_4", "med_day_8", "med_day_10"]] >= 20).all().all())
    py = pd.concat(per_year)
    a2_med = float(py.median())
    a2 = 3 <= a2_med <= 5

    print(f"\n锚点A2 每股每年 net_income 变更次数中位 {a2_med:.1f} ∈[3,5] ? "
          f"{'✓' if a2 else '✗'}")
    print(f"E1 披露月占比全部 ≥60% 且 报告期末月占比全部 ≤10% ? {'✓' if e1 else '✗'}")
    print(f"E2 4/8/10 月的变更中位日全部 ≥ 20 日 ? {'✓' if e2 else '✗'}")
    verdict = e1 and e2 and a2
    print(f"\n判定:财务面板{'**是** PIT 对齐,第二批因子可以跑' if verdict else '**不是** PIT 对齐,第二批因子不跑'}")
    df.to_csv(f"{OUT}/fundamental_pit_check.csv", index=False)
    print(f"落库 {OUT}/fundamental_pit_check.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §115-A 结论:财务面板**是** PIT 对齐,第二批因子可以跑。
#
#   列(六列结果完全一致)          变更数    4/8/10月   3/6/9/12月  4月中位  8月中位  10月中位
#   eps                          177,940    75.3%       7.9%        29       26        31
#   revenue                      179,890    75.3%       7.9%        29       26        31
#   net_income                   179,959    75.3%       7.9%        29       26        31
#   book_value_per_share         179,208    75.3%       7.9%        29       26        31
#   roe                          178,452    75.3%       7.9%        29       26        31
#   operating_cash_flow          179,193    75.3%       7.9%        29       26        31
#
#   E1 ✓(披露月 75.3% ≥60%,报告期末月 7.9% ≤10%)
#   E2 ✓(4/8/10 月的变更中位日 = 29 / 26 / 31,全部 ≥20)
#   A2 ✓(每股每年 net_income 变更次数中位 4.0 ∈ [3,5])
#
# 三个披露月的变更中位日正好压在法定截止日(4/30、8/31、10/31)上,
# 报告期末四个月只占 7.9%。**没有前视泄漏。**
#
# 事前预测:P1 命中。
# **P2 只对了一半 —— 我错了一半。** 我预测「4 月的中位日会明显晚于 8 月和 10 月」,
# 理由是 4 月要同时消化年报与一季报、大量公司卡在 4/29–4/30。
# 4月(29) > 8月(26) ✓,但 **10月(31) 比 4 月还晚 ✗**。
# 三季报没有年报那种提前披露的压力,反而更集中在截止日 —— 这一层我没想到。
#
# 注意:本节只验了「值的变更时点是否 PIT」,**没有验数值本身是否正确**,
# 也没有验累计口径的同比对齐 —— 后者必须复用 `fundamental_yoy.py` 的
# `label_periods` / `yoy_series`(§97–§101 在这上面栽过一轮),
# 不得在第二批里重新拼一套等价实现(§89 落下的第一条规矩)。
# =============================================================================
