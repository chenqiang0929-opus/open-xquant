"""§116 时间样本外规则 + 第二批财务因子横扫。

两件事一次做完,顺序不能反:时间样本外的规则必须在第二批开跑**之前**锁死,
否则第二批会重蹈第一批「16 个因子全在 2014–2025 内评估」的覆辙。

═══ A 部分:时间样本外规则(对第一批的 4 个幸存者)═══

诚实交代:第一批已经看过 full(2014–2025)与 oos(2023–2025)两个窗口,
**这两段已经被我看过,不能再当留出期**。整段历史里唯一没被 §115-B 看过的
是 **2026-01-05 → 面板末**(§115-B 只跑了 full 与 oos,没跑 holdout)。
它只有约 7 个月,证据力弱,但它是干净的。

判据 G1(2026 干净留出期)。对 §115-B 通过的 4 个因子
(small_cap / low_turnover / small_cap_low_turnover / high_amihud):
   在 holdout(2026-01-05 → 2026-07-27)上跑同一引擎、同一套同市值邻域随机对照,
   500 组种子。
   G1 通过 ⟺ 该因子 p < 0.05(**此处不做 Bonferroni**:这是对 4 个已选定因子
   的确认性检验,不是搜索;但通过与否只作为**弱证据**记录,7 个月不足以定论)。

判据 G2(分期稳定性,不是样本外,是稳健性)。把 2014–2025 切成三段独立子期:
   2014-01-02→2017-12-29 / 2018-01-02→2021-12-31 / 2022-01-04→2025-12-31,
   每段单独跑 500 组同市值随机对照。
   G2 通过 ⟺ 该因子在**三段中至少两段** p < 0.05。
   不通过 → 该因子的全期显著性来自单一时期,不可用。

═══ B 部分:第二批财务因子(12 个)═══

前置门 §115-A 已通过(财务列按公告日前向填充,无前视泄漏)。

**累计口径的处理(§97–§101 栽过的地方)**:财务字段是累计值,
同一天不同股票可能停在不同报告期(中报 vs 三季报),直接横向比较是错的。
本节一律先转 **TTM**:TTM = 本年累计 + 上年年报 − 上年同期累计。
报告期标签**必须复用** `fundamental_yoy.label_periods`,不得重拼等价实现
(§89 落下的第一条规矩)。同比同样按同一报告期对齐。

因子清单(12 个)
  价值   ep_ttm / bp / sp_ttm / cfp_ttm
  质量   roe_ttm / margin_ttm / accrual(低者优)
  成长   ni_yoy / rev_yoy
  复合   value_composite(ep,bp,sp,cfp 分位均值)
         quality_composite(roe,margin,−accrual 分位均值)
  负对照 expensive(= −ep_ttm,最贵的 20 只)

**时间样本外(本批从一开始就有)**
  训练期 2014-01-02 → 2021-12-31(8 年)—— 判据在这里判
  留出期 2022-01-04 → 面板末          —— **只看一次,不回头调参**

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
H1 锚点(不过则本节作废):
   (a) 面板 (3297, 5217);
   (b) 抽样恒等式:对照每次抽样的市值名次偏离必须 ≤25,违例 > 0 即作废;
   (c) **TTM 恒等式**:随机抽 200 只股票,在其年报公告日当天,
       TTM 净利必须等于当期累计净利(年报的累计就是全年,TTM 与之相等)。
       允许 1e-6 相对误差。若 TTM 拼错,此项必炸。
   (d) **报告期对齐锚点**:复用 `yoy_series("300347")`,泰格 2017 中报/三季报/
       年报/2018 一季报的同比必须复现雪球真值 0.5307/1.0103/1.1401/1.2107(±0.5pp)。

H2 单因子判定(**只在训练期判**)。对照 = 同市值名次 ±25 邻域匹配随机 20 只,
   500 组种子。p = (1 + #{对照 ≥ 策略}) / 501,单尾。
   **Bonferroni:12 个因子,α = 0.05 / 12 = 0.004167。**
   H2 通过 ⟺ 训练期 p < 0.004167。

H3 留出期确认(**只看一次**)。H2 通过的因子在 2022→面板末 上报告 p。
   H3 通过 ⟺ 留出期 p < 0.05(不再做 Bonferroni,因为这是确认而非搜索)。
   **无论 H3 结果如何,都不回头改 H2 的因子定义或阈值。**

H4 负对照锚点(§83 反问型)。`expensive` 不得通过 H2。若通过,本节作废。

事前预测(写下来以便被证伪;错了必须在正文里明说我错了)
--------------------------------------------------------
Q1 G1:4 个因子里**至少 3 个**在 2026 干净留出期上 p < 0.05。
Q2 G2:4 个因子**全部**通过三段中至少两段。
Q3 H2:通过的财务因子数 ∈ [2, 6]。
Q4 **`bp`(账面市值比)会通过 H2** —— A 股价值因子里 BP 最经典,
   且 Codex 的 R08 价值复合在他口径下全区间 +563.16%。
Q5 **成长族(ni_yoy / rev_yoy)不会通过 H2。** 理由:本项目 §97–§101、§103
   反复测过业绩反转与业绩持续性,全部落在 lift≈1.0;Codex 的 R13 叠加基本面
   也未过样本外。
Q6 H3:通过 H2 的因子里,**至少一半**能过 H3 留出期。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推;
**不因为留出期结果不好就回头改训练期的因子定义**;
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
from codex_r10_replication import DATA, TOP_N, WEIGHT, metrics, pct  # noqa: E402
from factor_sweep_pv import draw_fast  # noqa: E402
from fundamental_yoy import label_periods, yoy_series  # noqa: E402

NSEED = 500
ALPHA = 0.05 / 12
TRAIN = ("2014-01-02", "2021-12-31")
HOLD = ("2022-01-04", "2026-07-27")
FLOW = ["eps", "revenue", "net_income", "operating_cash_flow"]


def ttm_and_yoy(code, idx):
    """把累计口径的流量字段转 TTM,并算同报告期同比。

    TTM = 本年累计 + 上年年报 − 上年同期累计;报告期标签复用 label_periods。
    返回 (ttm_df, yoy_df, bps),三者都已 reindex 到 idx 并按公告日前向填充。
    """
    x = pd.read_parquet(f"{DATA}/{code}.parquet",
                        columns=[*FLOW, "book_value_per_share"])
    if getattr(x.index, "tz", None) is not None:
        x.index = x.index.tz_localize(None)
    ni = x["net_income"].ffill()
    ch = ni[ni.diff().fillna(0) != 0].index
    ch = ch[np.isfinite(ni[ch].to_numpy(float))]
    if len(ch) < 8:
        e = pd.DataFrame(np.nan, index=idx, columns=FLOW)
        return e, e.copy(), pd.Series(np.nan, index=idx)
    lab = label_periods(ch)
    cum = {c: {} for c in FLOW}
    ttm_rows, yoy_rows, dates = [], [], []
    for t, (ry, rp) in zip(ch, lab, strict=True):
        if ry is None:
            continue
        tv, yv = {}, {}
        for c in FLOW:
            v = float(pd.to_numeric(x[c], errors="coerce").ffill().get(t, np.nan))
            cum[c][(ry, rp)] = v
            fy = cum[c].get((ry - 1, 4))          # 上年年报(全年累计)
            same = cum[c].get((ry - 1, rp))       # 上年同期累计
            tv[c] = (v + fy - same) if (rp < 4 and fy is not None
                                        and same is not None) else (v if rp == 4 else np.nan)
            yv[c] = (v / abs(same) - 1) if same not in (None, 0) else np.nan
        ttm_rows.append(tv)
        yoy_rows.append(yv)
        dates.append(t)
    di = pd.DatetimeIndex(dates)
    ttm = pd.DataFrame(ttm_rows, index=di).reindex(idx).ffill()
    yoy = pd.DataFrame(yoy_rows, index=di).reindex(idx).ffill()
    bps = pd.to_numeric(x["book_value_per_share"], errors="coerce").ffill().reindex(idx)
    return ttm, yoy, bps
