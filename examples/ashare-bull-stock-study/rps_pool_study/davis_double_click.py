"""§129 两篇研报的「牛股启动信号」——补上它们都没做的充分性检验。

两篇研报
--------
① 广发证券 2021-09-10《复盘食品饮料行业十年十倍股之戴维斯双击篇》
   链条:股价大跌 30%-50% → PE 跌到 20 倍以下筑底 → 业绩超预期 → 估值提升 → 双击
   表 4:22 只十倍股各自的**加速上涨开始时点**,平均 2.75 年涨 882%;
   表 5:加速期 PE 从 17.12 倍升至 62.58 倍,估值贡献 265.50% > 业绩贡献 168.73%。
② 安信证券 2023-07-20《A 股一年三倍股研究及十五大启示》
   一年三倍股上涨前市值 **10-50 亿占 63%**;业绩增速 >300% 的不足 15%
   → **上涨多数仍为估值提升**;上涨前一个月涨幅 > -1% 的最终平均涨幅 427.74%,
   显著高于其他组别。

两篇的共同问题(与 §108 陶博士完全同型)
--------------------------------------
**它们证明的都是必要性,不是充分性。**
广发:「十倍股里普遍存在双击」;安信:「一年三倍股里,上涨前不跌的涨得更多」。
安信那份更进一步 —— 它的分组分析是**在已选定的赢家样本内部**做的,
**在结果上做条件、且没有对照组**,这类结论天然不能用来选股。
「启动信号」要回答的是:**满足条件之后,大涨的概率相对同类股票是否更高。**
**两篇都没有对照组,本节补上。**

数据边界(如实记录)
--------------------
广发 22 只中:颐海国际(1579.HK)、澳优(1717.HK)**是港股,不在本 A 股面板**;
伊利股份加速起点 2008-10-27 **早于本面板起点 2013-01-04**。→ 可测 19 只。
2013 年之前的起点一律不做外推。
安信报告的市值口径为「上涨前市值」,本节用**流通市值**(面板口径),标注差异。

Part A 广发 22 只的代码验证(锚点性质)
--------------------------------------
按已知常识给出证券简称→6 位代码的映射,**逐只用面板重算**表 4 的
「加速上涨开始时点 → 市值最高时点」区间涨幅,与研报公布值比较。
**A1 锚点:相对误差 < 25% 才认为代码对上**(允许复权与市值/股价口径差异);
对不上的**剔除并列出**,不猜代码、不换日期去凑。

Part B 起点当日的三条件实测(描述性)
------------------------------------
对通过 A1 的每只,在其加速起点当日测:(a) 距 250 日最高收盘的回撤;
(b) PE_TTM = raw_close / eps_ttm;(c) 最近一期净利同比(报告期对齐)。
**「超预期」本面板没有一致预期数据,只能用同比增速代理,明确标注,
不得当成「超预期」本身。**

Part C 充分性检验(**两篇都没做的部分,本节核心**)
--------------------------------------------------
事件定义(同一只股票 250 个交易日内不重复计事件):
  **GF 口径**(广发):距 250 日最高收盘回撤 ≤ -30% 且 PE_TTM ∈ (0,20]
                     且 最近一期净利同比 > 0
  **AX 口径**(安信):流通市值 ∈ [10亿, 50亿] 且 过去 20 个交易日涨幅 > -1%
前瞻与门槛:
  GF → 事件后 **690 个交易日**(研报平均加速期 2.75 年)的峰值涨幅 ≥ **+100%**
  AX → 事件后 **250 个交易日**的峰值涨幅 ≥ **+200%**(一年三倍)
对照:**同市值名次 ±25 + 同申万一级行业**(§125 的双中性口径),
每个事件抽一只同日对照股,**500 组种子**。
  注:AX 口径本身就是市值条件,同市值对照会把「小市值」这一维中性掉 ——
  **这正是要问的:除掉小市值本身,安信说的其他特征还有没有增量。**
  「小市值」本身是否有效,§115-B/§116-A 已单独测过(small_cap 通过),此处不重复。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
C1 锚点(不过则整节作废)
   (a) 面板 (3297, 5217);TTM 恒等式;泰格 300347 同比复现雪球真值;
   (b) **行业恒等式**:对照与被对照股必须同属一个申万一级行业,违例 > 0 即作废;
   (c) **无前视**:事件条件只用 ≤t 的信息,前瞻窗口起点严格 > t,逐点断言。

C2 充分性判定。**核心判据。** 两个口径各判各的。
   统计量 = P(峰值涨幅 ≥ 门槛)。
   **Bonferroni:2 个口径,α = 0.05/2 = 0.025。**
   C2 通过 ⟺ 事件组的该概率**严格高于** 500 组对照的 (1-α) 分位数,
   即单尾 p < 0.025。
   **不通过 → 该研报的「启动信号」不具备可检出的充分性**:
   它描述了赢家的共同经历,但不能用来选股。

C3 分层描述(不设阈值):GF 口径的回撤门槛 -30%/-40%/-50% × PE 门槛 15/20/25;
   AX 口径的市值区间 [10,50]/[10,100]/[5,30] 亿。
   **仅描述,不据此挑最优组合** —— 挑就是事后调参。

C4 食品饮料子样本(描述项,不参与判定):GF 口径只在申万一级=食品饮料内重跑,
   **样本量必然很小,只作参考**。

事前预测
--------
**本节不下预测**(§119 起已停止此类外推)。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推;
**不把「同比增速」说成「超预期」**;
**不因某个门槛组合好看就把它当规则**(要变规则须另开一节重新事前登记);
不对 2013 年之前的起点做外推;
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
from codex_r10_neutral import CACHE, OUT  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from codex_routes_rerun import build_fund  # noqa: E402
from fundamental_yoy import yoy_series  # noqa: E402
from industry_neutral import CLS, build_industry  # noqa: E402

NSEED, ALPHA = 500, 0.05 / 2
# 广发表 4:证券简称 -> (代码, 加速起点, 市值最高日, 研报公布区间涨幅)
GF_TABLE = [
    ("颐海国际", None, "2017-03-14", "2020-09-02", 43.4667),
    ("酒鬼酒", "000799", "2019-01-03", "2021-06-22", 16.4827),
    ("山西汾酒", "600809", "2018-10-29", "2021-07-15", 15.7589),
    ("百润股份", "002568", "2019-01-31", "2021-02-10", 14.5957),
    ("澳优", None, "2016-07-07", "2019-07-03", 8.4048),
    ("舍得酒业", "600702", "2020-09-25", "2021-07-21", 8.0665),
    ("安井食品", "603345", "2019-03-13", "2021-02-09", 7.9740),
    ("重庆啤酒", "600132", "2018-03-23", "2021-02-10", 7.0597),
    ("贵州茅台", "600519", "2014-01-10", "2018-01-12", 6.9986),
    ("伊利股份", "600887", "2008-10-27", "2010-11-30", 5.9351),
    ("水井坊", "600779", "2015-09-30", "2018-06-26", 5.8667),
    ("安琪酵母", "600298", "2014-06-20", "2018-06-07", 5.8453),
    ("五粮液", "000858", "2014-01-08", "2018-01-15", 5.3407),
    ("顺鑫农业", "000860", "2018-02-09", "2020-08-28", 4.8061),
    ("古井贡酒", "000596", "2014-06-19", "2018-07-16", 4.7775),
    ("泸州老窖", "000568", "2014-06-04", "2018-01-16", 3.8971),
    ("老白干酒", "600559", "2014-07-10", "2015-06-24", 3.8119),
    ("涪陵榨菜", "002507", "2016-03-16", "2018-07-31", 3.6662),
    ("海天味业", "603288", "2018-02-07", "2020-09-02", 3.6260),
    ("恒顺醋业", "600305", "2018-03-23", "2020-08-24", 3.3745),
    ("中炬高新", "600872", "2018-03-28", "2020-09-02", 2.6242),
    ("汤臣倍健", "300146", "2019-12-02", "2021-05-25", 1.8092),
]


def main():  # noqa: PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    cl = z["CL"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点C1a"
    y = yoy_series("300347").set_index(["报告年", "报告期"])["同比"]
    assert abs(float(y.get((2017, "中报"), np.nan)) - 0.5307) < 0.005, "锚点C1a 泰格"
    print(f"锚点C1a ✓ {nt}×{ns};泰格同比 ✓  (CLS={os.path.basename(CLS)})", flush=True)

    t0 = time.time()
    raw = np.full((nt, ns), np.nan, np.float32)
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点C1a TTM"
    ind, names, nid = build_industry(codes, idx)
    fb = nid.get("食品饮料", -99)
    print(f"矩阵完成 ({time.time()-t0:.0f}s);食品饮料 id={fb}", flush=True)

    pos = {c: j for j, c in enumerate(codes)}
    ipos = pd.Index(idx)

    def dpos(d, how="bfill"):
        r = int(ipos.get_indexer([pd.Timestamp(d)], method=how)[0])
        return r if r >= 0 else None

    # ---- Part A:22 只代码验证 ----
    print("\n=== Part A 广发表4 逐只验证(A1:相对误差<25%)===", flush=True)
    ok_a, rows_a = [], []
    for nm, code, d0, d1, rep in GF_TABLE:
        if code is None:
            print(f"  {nm:8s} —— 港股,不在本 A 股面板,跳过")
            rows_a.append({"name": nm, "code": None, "status": "港股不在面板"})
            continue
        if code not in pos:
            print(f"  {nm:8s} {code} —— 不在面板,跳过")
            rows_a.append({"name": nm, "code": code, "status": "不在面板"})
            continue
        if pd.Timestamp(d0) < idx[0]:
            print(f"  {nm:8s} {code} 加速起点 {d0} 早于面板起点 {idx[0].date()},跳过")
            rows_a.append({"name": nm, "code": code, "status": "起点早于面板"})
            continue
        a, b = dpos(d0), dpos(d1, "ffill")
        j = pos[code]
        pa, pb = cl[a, j], cl[b, j]
        got = float(pb / pa - 1.0) if np.isfinite(pa) and np.isfinite(pb) and pa > 0 else np.nan
        rel = abs(got - rep) / abs(rep) if np.isfinite(got) else np.inf
        good = rel < 0.25
        ok_a.append((nm, code, a, b)) if good else None
        rows_a.append({"name": nm, "code": code, "rep": rep, "got": got,
                      "rel": rel, "A1": good, "status": "ok" if good else "误差超限"})
        print(f"  {nm:8s} {code} 研报 {rep*100:+8.1f}% | 面板 "
              f"{got*100:+8.1f}% | 相对误差 {rel:6.1%} {'✓' if good else '✗'}")
    print(f"A1 通过 {len(ok_a)}/22(港股与超前起点已剔除)", flush=True)

    # ---- Part B:起点当日三条件 ----
    print("\n=== Part B 加速起点当日的研报三条件(实测)===", flush=True)
    hit = {"dd30": 0, "pe20": 0, "yoy": 0, "all3": 0}
    rows_b = []
    for nm, code, a, _ in ok_a:
        j = pos[code]
        w = cl[max(0, a - 249):a + 1, j].astype(np.float64)
        dd = float(cl[a, j] / np.nanmax(w) - 1.0) if np.isfinite(np.nanmax(w)) else np.nan
        e = fm["eps_ttm"][a, j]
        pe = float(raw[a, j] / e) if np.isfinite(e) and e > 0 else np.nan
        ni, nip = fm["ni_ttm"][a, j], fm["ni_ttm"][max(0, a - 250), j]
        yy = float(ni / abs(nip) - 1.0) if np.isfinite(ni) and np.isfinite(nip) and nip != 0 else np.nan
        c1, c2, c3 = dd <= -0.30, (0 < pe <= 20), yy > 0
        hit["dd30"] += int(bool(c1))
        hit["pe20"] += int(bool(c2))
        hit["yoy"] += int(bool(c3))
        hit["all3"] += int(bool(c1 and c2 and c3))
        rows_b.append({"name": nm, "code": code, "dd250": dd, "pe_ttm": pe,
                      "ni_yoy_proxy": yy, "c_dd30": bool(c1), "c_pe20": bool(c2),
                      "c_yoy": bool(c3)})
        print(f"  {nm:8s} 回撤{dd:+7.1%} {'✓' if c1 else ' '} | "
              f"PE {pe:7.1f} {'✓' if c2 else ' '} | 同比(代理){yy:+8.1%} "
              f"{'✓' if c3 else ' '}")
    n = max(len(ok_a), 1)
    print(f"  命中率:回撤≤-30% {hit['dd30']}/{n}  PE≤20 {hit['pe20']}/{n}  "
          f"同比>0 {hit['yoy']}/{n}  **三条同时 {hit['all3']}/{n}**", flush=True)
    pd.DataFrame(rows_a).to_csv(f"{OUT}/davis_partA.csv", index=False)
    pd.DataFrame(rows_b).to_csv(f"{OUT}/davis_partB.csv", index=False)
    print(f"落库 {OUT}/davis_partA.csv, davis_partB.csv")
    np.savez_compressed(f"{OUT}/davis_mats.npz", raw=raw, ind=ind)
    print(f"中间矩阵缓存 {OUT}/davis_mats.npz")


if __name__ == "__main__":
    main()
