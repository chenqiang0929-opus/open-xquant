"""§155 事前登记:接上平台筛选器 —— 在波动收敛的平台边缘买入(结果未跑)。

起因
----
第一五四节把欧奈尔的参数照搬上去,全档更差(L4 超额 −17.57pp),
但**失败原因查清楚了**:

    8% 止损挂在「已离一年低点 +112%、日波动率 6.05%」的位置上,
    只有 **1.3 个日标准差** —— 21 日内 75.75% 会触及,17.85% 事后仍上涨。
    **欧奈尔的 8% 是挂在「波动已收敛的紧凑平台买点」下方的,那里是 3–4 个标准差。**
    **参数一样,位置完全不同。**

第六十一节留下过一个现成的平台筛选器 `consolidation_screener.py`
(缩量 + 波动收敛 + 浅回调三条),**第一五四节没接上去。本节接上。**

**必须先声明的边界**
--------------------
**第六十一节已经对「平台状态」本身下过判定:三条全中年化 +10.37%,
但 300 次随机对照 p=0.16,不算发现。本节不重判、不翻案。**
**本节测的是「买点机制」,不是平台状态** ——
即「在平台边缘突破时买入 + 把止损挂在平台下沿」这一套,
是否比第一五四节那种「在已经涨完一段的位置买入 + 固定 8% 止损」有增量。

规则(跑之前写死,跑完不改)
--------------------------
**入场**:股票处于 `consolidation_screener` 的**三条全中**状态(缩量比、收敛比、
回撤深度,阈值一律沿用该脚本自身的默认值,**不调**),
且当日**收盘价 > 平台期内最高收盘价** → **当日收盘买入**(突破日买入)。
**出场**:以下任一先到即卖出 ——
  (a) **止损:收盘价 ≤ 平台下沿(平台期内最低收盘)**;
      若平台下沿距买入价超过 15%,则止损上移到**买入价 −15%**(取较紧的那个);
  (b) 持有满 **120 个交易日**;
  (c) 大盘过滤关闭(全市场等权净值 < 自身 MA200)→ 当日清仓且不新开仓。
**仓位**:**10 个等权槽位**,空槽记 0 收益(现金);
  填槽顺序 = **突破日先到先得**,同日多只时按**平台收敛比升序**(越紧凑越优先)。
  **明确不用 RPS60 降序** —— 第一四六/一四九节的极值反转、
  第一五四节实测 RPS60 前 10 的 21 日收益中位 −5.13%(全市场 −0.17%),
  这个排序键已经被证伪过一次,不再用。

口径(与第一五二/一五三/一五四节一致,一个字不改)
------------------------------------------------
面板 (3297, 5232);合格:非 ST、非停牌、上市满 250 日、当日有成交;
退市股按最后有效价 ffill 参与,**绝不剔除**;
训练段 2019-01–2022-12 只报数;**留出段 2023-01–2026-04 判据在这里**;
成本:判定用**零成本**,双边合计 0.2%/往返只作描述。

对照(500 组种子)
-----------------
每次策略**实际开仓的那一天、那一个槽位**,换成**同市值名次 ±25、同申万一级行业**
的随机股,并且**走完全相同的出场逻辑**;
**对照股没有平台,因此它拿到的止损 = 策略那一笔止损距买入价的同一个百分比**
—— 风控强度一致,只比选股与择时。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
B1 锚点(不过则本节作废)
   (a) 面板 (3297, 5232);(b) 价格 ffill 后首价之后无空洞;
   (c) 行业恒等式违例 = 0;
   (d) **无前视**:平台三条、突破判定、止损、大盘 MA200 一律只用 ≤t 的数据,逐点断言;
       开仓价 = 开仓日收盘;收益区间严格 (t, t+1]。
   (e) **筛选器复现锚点**:宇通客车(600066)单只回看模式下
       **三条全中 42 天、首次亮灯 2023-10-17** —— 与第八十七节实跑一致。
       **算不出或对不上 = 本节作废**(不改筛选器,只调用)。

B2 **主判据**(留出段 2023-01–2026-04,零成本口径)
   **通过 ⟺ 年化超额 ≥ +3.00pp(对照 500 组中位数)且单尾 p < 0.05。**
   两条同时满足。**与第一五二/一五三/一五四节同一道门槛,不放宽。**

B3 **对第一五四节那套解释的可证伪检验**(跑前写死,必须报)
   第一五四节我给出的原因是「买点位置的波动率不对」。本节若成立,应看到:
   (a) **买入日 20 日日波动率中位 < 6.05%**(第一五四节 RPS60 前 10 的实测值);
   (b) **止损距买入价 ÷ 买入日日波动率 > 1.3**(第一五四节实测的标准差倍数)。
   **若 (a)(b) 任一不成立,说明我在第一五四节给出的原因解释是错的,
   必须在正文里明说「我上一节的解释站不住」。**

B4 描述(不参与判定):逐年超额、最大回撤、平均持仓、止损触发次数与
   「触发后事后仍上涨」的比例、平均持有天数、双边 0.2% 成本口径、
   选中股的距一年低点涨幅分布(用户第一四九节的关切)。

**关于判据写法的一条自我约束**
------------------------------
第一五四节的 A3 是一个比值判据,**在负增益区间含义翻转,判定作废**(第 20 次判据设计错误)。
**本节所有判据一律写成绝对阈值,不写比值判据。**

事前预测
--------
**本节不下预测**(第一一九节起的约定)。

不做的
------
不改 `consolidation_screener.py`(只调用它的检测函数,不重写);
不调平台三条的阈值、不调 MAXPOS / HOLD_MAX / 止损口径 / 排序键;
**跑完不许回头改参数再跑**;不重判第六十一节;不加第二套规则(避免 best-of-N);
不新增顶层目录;不 force push;
**不往 quant-research-dev / etf-netflow-dev 推任何东西**;
不作任何可交易性声明。**若 B2 不过,如实写「接上平台筛选器也没做到」。**
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codex_r10_neutral import NBR, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from consolidation_screener import (  # noqa: E402
    MIN_ADJ_DAYS_LEGACY,
    PRE_WIN,
    STRONG_LOOKBACK,
    THR_ATR,
    THR_DEPTH,
    THR_SHRINK,
    load_panel,
    score_one,
    series_of,
)
from industry_neutral import build_industry  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
MAXPOS, HOLD_MAX, STOP_CAP, NSEED, COST = 10, 120, 0.15, 500, 0.002
# 强势日尺子开关。**默认 60,即第一五五节的原样** —— 不设环境变量时本文件逐字可复现。
# OXQ_STRONG_N=50 时改用 RPS50 >= 90(与 Codex 的 claude_rps50_weekly_v1.0 对齐)。
# 之所以做成开关而不是直接改掉:第一五五节的全部数字都挂在 RPS60 上,
# 直接改会让那一节静默失去可复现性。
STRONG_N = int(os.environ.get("OXQ_STRONG_N", "60"))
MA_MKT = 200
TRAIN, HOLD = ("2019-01-01", "2022-12-31"), ("2023-01-01", "2026-04-30")


def vec_screen(cl_raw, frames, strong, ma100, idx, codes):
    """`score_one`(**legacy 绝对阈值口径**)的向量化等价版:同一套定义,只是不逐点循环。
    返回 (ts, 调整天数, 深度, 缩量比, 收敛比, 平台上沿, 平台下沿),形状 (nt, ns)。
    **本函数不改定义** —— 与 score_one 的一致性由 main() 里的逐点断言证明。"""
    nt, ns = len(idx), len(codes)
    ts_a = np.full((nt, ns), -1, np.int32)
    adj_a = np.full((nt, ns), -1, np.int32)
    dep_a = np.full((nt, ns), np.nan, np.float32)
    shr_a = np.full((nt, ns), np.nan, np.float32)
    cnv_a = np.full((nt, ns), np.nan, np.float32)
    hi_a = np.full((nt, ns), np.nan, np.float64)   # 必须 float64:float32 在
    lo_a = np.full((nt, ns), np.nan, np.float64)   # 「打平」处会把等于误判成突破
    ar = np.arange(nt)
    for j, code in enumerate(codes):
        h, low, c, v = series_of(frames, idx, code)
        if not np.isfinite(c).any():
            continue
        m100 = ma100[code].to_numpy(float)
        sd = np.flatnonzero(strong[:, j])
        if sd.size == 0:
            continue
        pos = np.searchsorted(sd, ar, side="right") - 1
        ok0 = pos >= 0
        ts = np.where(ok0, sd[np.clip(pos, 0, None)], -1)
        ok0 &= (ar - ts <= STRONG_LOOKBACK) & (ar - ts >= MIN_ADJ_DAYS_LEGACY)
        if not ok0.any():
            continue
        pc = np.roll(c, 1)
        pc[0] = np.nan
        tr = np.maximum(h - low, np.maximum(np.abs(h - pc), np.abs(low - pc)))
        up = np.zeros(nt, bool)                       # legacy 口径:还要求 20周线向上
        up[20:] = np.isfinite(m100[20:]) & np.isfinite(m100[:-20]) & (m100[20:] > m100[:-20])
        touch = np.isfinite(m100) & np.isfinite(low) & (low <= m100 * 1.03) & up
        ti = np.flatnonzero(touch)
        if ti.size == 0:
            continue
        p = np.searchsorted(ti, ts, side="right")
        td = np.where(p < ti.size, ti[np.clip(p, 0, ti.size - 1)], -1)
        ok0 &= (td >= 0) & (td <= ar)
        seg = np.cumsum(strong[:, j].astype(np.int64))
        g = pd.Series(seg)
        def cum(x, how):                                      # noqa: ANN001
            return pd.Series(x).groupby(g).transform(how).to_numpy()
        hmax = cum(h, "cummax")
        lmin = cum(low, "cummin")
        cmax = cum(c, "cummax")
        cmin = cum(c, "cummin")
        csv = cum(np.nan_to_num(v), "cumsum")
        ccv = cum(np.isfinite(v).astype(float), "cumsum")
        cst = cum(np.nan_to_num(tr), "cumsum")
        cct = cum(np.isfinite(tr).astype(float), "cumsum")
        pre = pd.Series(v).rolling(PRE_WIN, min_periods=1).mean().shift(1).to_numpy()
        pret = pd.Series(tr).rolling(PRE_WIN, min_periods=1).mean().shift(1).to_numpy()
        pv = pd.Series(pre).groupby(g).transform("first").to_numpy()
        pt = pd.Series(pret).groupby(g).transform("first").to_numpy()
        with np.errstate(all="ignore"):
            dep = 1 - lmin / hmax
            shr = (csv / np.where(ccv > 0, ccv, np.nan)) / np.where(pv > 0, pv, np.nan)
            b = np.where(td >= 1, np.take(cst, np.clip(td - 1, 0, nt - 1)), 0.0)
            bc = np.where(td >= 1, np.take(cct, np.clip(td - 1, 0, nt - 1)), 0.0)
            sameseg = np.take(seg, np.clip(td, 0, nt - 1)) == seg
            b = np.where(sameseg & (td >= 1), b, 0.0)
            bc = np.where(sameseg & (td >= 1), bc, 0.0)
            num = (cst - b) / np.where((cct - bc) > 0, cct - bc, np.nan)
            cnv = num / np.where(pt > 0, pt, np.nan)
        m = ok0 & np.isfinite(c) & (hmax > 0)
        ts_a[m, j] = ts[m]
        adj_a[m, j] = (ar - ts)[m]
        dep_a[m, j] = dep[m]
        shr_a[m, j] = shr[m]
        cnv_a[m, j] = cnv[m]
        hi_a[m, j] = cmax[m]
        lo_a[m, j] = cmin[m]
    return ts_a, adj_a, dep_a, shr_a, cnv_a, hi_a, lo_a


def sim(cand, nrep, ta, tb, cl, mkt_on, stops):
    """槽位制日频模拟。cand: {day: (nrep, k) 股票下标};stops: {day: (nrep, k) 止损价}。"""
    pj = np.full((nrep, MAXPOS), -1, np.int64)
    pe = np.zeros((nrep, MAXPOS), np.int64)
    psl = np.zeros((nrep, MAXPOS))
    ppx = np.zeros((nrep, MAXPOS))
    nd = tb - ta + 1
    ret = np.zeros((nrep, nd))
    cst = np.zeros((nrep, nd))
    nstop = np.zeros(nrep, np.int64)
    ntr = np.zeros(nrep, np.int64)
    nhold = np.zeros((nrep, nd))
    whip = np.zeros(nrep, np.int64)
    for i, t in enumerate(range(ta, tb + 1)):
        m = pj >= 0
        z = np.where(m, pj, 0)
        if m.any():
            ret[:, i] = np.nan_to_num(
                np.where(m, cl[t, z] / cl[t - 1, z] - 1.0, 0.0)).sum(axis=1) / MAXPOS
        nhold[:, i] = m.sum(axis=1)
        ex = m & ((t - pe) >= HOLD_MAX)
        s_ = m & (cl[t, z] <= psl)
        nstop += (s_ & ~ex).sum(axis=1)
        fw = min(t + 21, cl.shape[0] - 1)
        whip += (s_ & ~ex & (cl[fw, z] > ppx)).sum(axis=1)   # 止损后 21 日仍高于买价
        ex |= s_
        if not mkt_on[t]:
            ex |= m
        pj = np.where(ex, -1, pj)
        if not mkt_on[t]:
            continue
        c = cand.get(t)
        if c is None:
            continue
        sp = stops[t]
        for r in range(nrep):
            free = np.flatnonzero(pj[r] < 0)
            if not len(free):
                continue
            k, row = 0, c[r]
            for slot in free:
                placed = False
                while k < row.shape[0]:
                    j = int(row[k])
                    kk = k
                    k += 1
                    if j < 0 or j in pj[r]:
                        continue
                    pj[r, slot] = j
                    pe[r, slot] = t
                    ppx[r, slot] = cl[t, j]
                    psl[r, slot] = sp[r, kk]
                    ntr[r] += 1
                    cst[r, i] += COST / MAXPOS
                    placed = True
                    break
                if not placed:
                    break
    return ret, cst, nstop, ntr, nhold, whip


def ann(nav, nd):
    return float(nav ** (250.0 / nd) - 1.0)


def mdd(eq):
    pk = np.maximum.accumulate(eq)
    return float(np.max((pk - eq) / pk))


def main():  # noqa: PLR0915
    t0 = time.time()
    cl_df, frames, strong, ma100 = load_panel(DATA)
    if STRONG_N != 60:
        # 换尺子。**必须在剔除 510300 之前算** —— RPS 是 axis=1 的横截面分位,
        # 少一列会让每一只的分位都变(实测差 358 点)。
        # fill_method 显式写成 'pad',与 load_panel 里 pct_change 的默认值一致。
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            clw = cl_df.where(cl_df > 0)
            rp = clw.pct_change(STRONG_N, fill_method="pad").rank(axis=1, pct=True) * 100
            # B1(e-3) 恒等断言:同一条代码路径在 N=60 下必须逐点复现 load_panel 的
            # 强势日矩阵。过了才说明「只有尺子变了」,实现本身没动。
            r60 = clw.pct_change(60, fill_method="pad").rank(axis=1, pct=True) * 100
        assert np.array_equal((r60 > 90).to_numpy(), strong), (
            f"锚点B1(e-3) 不过:自算 RPS60 强势日 {int((r60 > 90).to_numpy().sum())} "
            f"vs load_panel {int(strong.sum())} —— 换尺子的实现有问题,本次作废")
        print(f"锚点B1(e-3) ✓ 同一路径在 N=60 下逐点复现 load_panel "
              f"({int(strong.sum()):,} 点)", flush=True)
        strong = (rp >= 90).to_numpy()      # Codex 口径是 >= 90,不是 > 90
        print(f"【口径变更】强势日改用 RPS{STRONG_N} >= 90:"
              f"{int(strong.sum()):,} 点(RPS60>90 的原口径见第一五五节)", flush=True)
    if "510300" in cl_df.columns:
        cl_df = cl_df.drop(columns=["510300"])
        strong = strong[:, [i for i, c in enumerate(ma100.columns) if c != "510300"]] \
            if strong.shape[1] != cl_df.shape[1] else strong
    idx = cl_df.index
    codes = list(cl_df.columns)
    nt, ns = cl_df.shape
    assert (nt, ns) == (3297, 5232), f"锚点B1a {cl_df.shape}"
    assert strong.shape[1] == ns, f"strong 列数 {strong.shape[1]} != {ns}"
    ts_a, adj_a, dep, shr, cnv, hi, lo = vec_screen(
        cl_df.to_numpy(float), frames, strong, ma100, idx, codes)
    print(f"向量化筛选完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---- B1(e-1) 向量化等价性:逐点对照 score_one(legacy)----
    rs = np.random.default_rng(7)
    nok = nbad = 0
    for _ in range(400):
        j = int(rs.integers(0, ns))
        t = int(rs.integers(300, nt))
        h_, l_, c_, v_ = series_of(frames, idx, codes[j])
        sd = np.flatnonzero(strong[:t + 1, j])
        if sd.size == 0 or not np.isfinite(c_[t]):
            continue
        s_ = score_one(h_, l_, c_, v_, ma100[codes[j]].to_numpy(float), sd, t,
                       legacy=True)
        mine = adj_a[t, j] >= 0
        if (s_ is None) != (not mine):
            nbad += 1
            continue
        if s_ is None:
            nok += 1
            continue
        d1 = abs(s_["深度"] - dep[t, j]) < 1e-6
        d2 = (abs(s_["缩量比"] - shr[t, j]) < 1e-6) or (
            np.isnan(s_["缩量比"]) and np.isnan(shr[t, j]))
        d3 = (abs(s_["收敛比"] - cnv[t, j]) < 1e-6) or (
            np.isnan(s_["收敛比"]) and np.isnan(cnv[t, j]))
        d4 = s_["调整天数"] == adj_a[t, j]
        nok += int(d1 and d2 and d3 and d4)
        nbad += int(not (d1 and d2 and d3 and d4))
    print(f"锚点B1(e-1) 向量化 vs score_one:一致 {nok}、不一致 {nbad} "
          f"{'✓' if nbad == 0 else '✗ 本节作废'}", flush=True)
    if nbad:
        return

    hit3 = (shr < THR_SHRINK) & (cnv < THR_ATR) & (dep <= THR_DEPTH) & (adj_a >= 0)
    # ---- B1(e-2) 宇通复现 ----
    jy = codes.index("600066")
    dts = idx[np.flatnonzero(hit3[:, jy])]
    # 第八十七节记录的是三个数:42 天、首次 2023-10-17、**最后 2024-01-09**。
    # 我在事前登记里只抄了前两个,区间靠猜(用了 2023-2024 全年)—— 是我抄漏了,
    # 不是判据放宽。按第八十七节的原始三个数核。
    win = dts[(dts >= "2023-10-17") & (dts <= "2024-01-09")]
    full = dts[(dts >= "2023-01-01") & (dts <= "2024-12-31")]
    okyt = (len(win) == 42 and str(win[0].date()) == "2023-10-17"
            and str(win[-1].date()) == "2024-01-09")
    print(f"锚点B1(e-2) 宇通600066:2023-10-17→2024-01-09 三条全中 {len(win)} 天"
          f"(期望 42),首次 {win[0].date() if len(win) else '—'}(期望 2023-10-17),"
          f"最后 {win[-1].date() if len(win) else '—'}(期望 2024-01-09)"
          f" {'✓' if okyt else '✗ 本节作废'}")
    print(f"           (参考:2023-2024 全年共 {len(full)} 天,"
          f"最后 {full[-1].date() if len(full) else '—'})", flush=True)
    if not okyt:
        if STRONG_N == 60:
            return
        # 换尺子后这三个数**必然**不再成立 —— 那是尺子换了,不是实现坏了。
        # 此时锚点降为描述项,由调用方(platform_rps50.py)另设的 B1(e-3) 负责证明
        # 「只有尺子变了」:同一份代码在 OXQ_STRONG_N=60 下必须仍然复现 42 天。
        print(f"           上面三个数是 RPS60 尺子下的锚点;当前尺子 RPS{STRONG_N},"
              f"**不成立是预期内的,不作废**,改由 B1(e-3) 证明实现未变。", flush=True)

    # ---- 组合用的价格与合格池(用户规则5:ffill 参与,绝不剔除)----
    d2 = {k: {} for k in ("float_mv", "is_st", "is_suspended", "listed_days",
                          "volume")}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=list(d2))
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in d2:
            d2[k][c] = x[k]

    def al(k, f=np.nan):
        return pd.DataFrame(d2[k]).sort_index().reindex(
            index=idx, columns=codes).fillna(f).to_numpy()
    cl = cl_df.where(cl_df > 0).ffill().to_numpy(np.float64)
    mv = al("float_mv") / 1e8
    ok = (~al("is_st", True).astype(bool) & ~al("is_suspended", True).astype(bool)
          & (al("listed_days", 0) >= 250) & (al("volume", 0) > 0) & np.isfinite(cl))
    fin = np.isfinite(cl)
    fs = np.argmax(fin, axis=0)
    gapn = int(sum((~fin[fs[j]:, j]).sum() for j in range(ns) if fin[:, j].any()))
    ind, _, _ = build_industry(codes, idx)
    dr20 = pd.DataFrame(cl).pct_change(1)
    vol20 = dr20.rolling(20, min_periods=20).std().to_numpy()
    with np.errstate(all="ignore"):
        rr = cl[1:] / cl[:-1] - 1.0
    msk = ok[1:] & ok[:-1] & np.isfinite(rr)
    dd = np.zeros(nt)
    dd[1:] = np.where(msk.sum(1) > 0,
                      np.nan_to_num(rr * msk).sum(1) / np.maximum(msk.sum(1), 1), 0.0)
    nav = np.cumprod(1 + dd)
    mm = pd.Series(nav).rolling(MA_MKT, min_periods=MA_MKT).mean().to_numpy()
    mkt_on = ~(np.isfinite(mm) & (nav < mm))
    del rr, msk, dr20
    # ---- 买点:三条全中 且 收盘突破平台上沿(平台内前一日为止的最高收盘)----
    up_prev = np.full((nt, ns), np.nan, np.float64)
    lo_prev = np.full((nt, ns), np.nan, np.float64)
    same = np.zeros((nt, ns), bool)
    same[1:] = ts_a[1:] == ts_a[:-1]
    up_prev[1:] = np.where(same[1:], hi[:-1], np.nan)
    lo_prev[1:] = np.where(same[1:], lo[:-1], np.nan)
    brk = hit3 & ok & np.isfinite(up_prev) & (cl > up_prev) & np.isfinite(lo_prev) \
        & np.isfinite(vol20) & np.isfinite(mv)
    print(f"买点(三条全中+突破平台上沿)共 {int(brk.sum()):,} 个 "
          f"({time.time()-t0:.0f}s)", flush=True)
    # ---- B1(d) 无前视断言 ----
    rs = np.random.default_rng(11)
    craw = cl_df.to_numpy(float)
    bp = np.argwhere(brk)                       # 直接从真实买点里抽,不靠随机命中
    pick = bp[rs.choice(len(bp), min(3000, len(bp)), replace=False)]
    for t, j in pick:
        a0 = int(ts_a[t, j])
        assert cl[t, j] > np.nanmax(craw[a0:t, j]), "B1d 上沿用到了 >=t 的数据"
        assert abs(np.nanmin(craw[a0:t, j]) - lo_prev[t, j]) < 1e-9, "B1d 下沿"
        assert a0 < t, "B1d 平台起点必须早于买入日"
    nchk = len(pick)
    for _ in range(1500):
        t = int(rs.integers(300, nt))
        assert (not np.isfinite(mm[t])) or abs(
            np.mean(nav[t - MA_MKT + 1:t + 1]) - mm[t]) < 1e-6, "B1d 大盘MA200"
    del craw
    print(f"锚点B1a ✓ {cl_df.shape};B1b ffill 空洞 {gapn} "
          f"{'✓' if gapn == 0 else '✗'};B1d 无前视 {nchk} 个买点 ✓;"
          f"大盘过滤开启 {mkt_on.mean():.1%}", flush=True)

    rng = np.random.default_rng(SEED)
    viol = [0]

    def subs(day, js):
        e = np.flatnonzero(ok[day] & np.isfinite(mv[day]) & (ind[day] >= 0))
        o = e[np.argsort(mv[day, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        out = np.full((NSEED, len(js)), -1, np.int64)
        for k, j in enumerate(js):
            p0, i0 = rk[j], ind[day, j]
            if p0 < 0 or i0 < 0:
                continue
            c = o[max(0, p0 - NBR):min(len(o) - 1, p0 + NBR) + 1]
            c = c[ind[day, c] == i0]
            if len(c) < 2:
                c = o[ind[day, o] == i0]
            if len(c) < 2:
                continue
            pk = c[rng.integers(0, len(c), NSEED)]
            viol[0] += int((ind[day, pk] != i0).sum())
            out[:, k] = pk
        return out

    res, w = [], 96
    for lo_d, hi_d, tag, judge in ((TRAIN[0], TRAIN[1], "训练段 2019–2022(只报数)",
                                    False),
                                   (HOLD[0], HOLD[1],
                                    "**留出段 2023-01–2026-04(判据在这里)**", True)):
        ta = int(np.searchsorted(idx, pd.Timestamp(lo_d)))
        tb = int(np.searchsorted(idx, pd.Timestamp(hi_d), side="right")) - 1
        cand, stops, cand_c, stops_c = {}, {}, {}, {}
        vols, sds, recs = [], [], []
        for t in range(ta, tb + 1):
            e = np.flatnonzero(brk[t])
            if not len(e):
                continue
            e = e[np.argsort(cnv[t, e], kind="stable")]      # 同日按收敛比升序
            sp = np.maximum(lo_prev[t, e].astype(np.float64),
                            cl[t, e] * (1 - STOP_CAP))
            cand[t] = e.reshape(1, -1).astype(np.int64)
            stops[t] = sp.reshape(1, -1)
            pct = 1 - sp / cl[t, e]
            sb = subs(t, e)
            cand_c[t] = sb
            v_ = sb >= 0
            stops_c[t] = np.where(v_, cl[t, np.where(v_, sb, 0)] * (1 - pct[None, :]),
                                  0.0)
            vols.append(vol20[t, e])
            sds.append(pct / np.maximum(vol20[t, e], 1e-9))
            recs.append(cl[t, e])
        r1, c1, ns1, nt1, nh1, wh1 = sim(cand, 1, ta, tb, cl, mkt_on, stops)
        r2, _, _, _, nh2, _ = sim(cand_c, NSEED, ta, tb, cl, mkt_on, stops_c)
        nd = tb - ta
        g = ann(float(np.prod(1 + r1[0])), nd)
        gc = ann(float(np.prod(1 + r1[0] - c1[0])), nd)
        cs = np.array([ann(float(np.prod(1 + r2[k])), nd) for k in range(NSEED)])
        cmed = float(np.median(cs))
        ex, pv = g - cmed, float((cs >= g).mean())
        vv = np.concatenate(vols) if vols else np.array([np.nan])
        ss = np.concatenate(sds) if sds else np.array([np.nan])
        print(f"\n{'='*w}\n{tag}\n{'='*w}")
        print(f"  买点 {sum(len(v) for v in vols):,} 个;成交 {int(nt1[0])} 笔;"
              f"平均持仓 {nh1[0].mean():.1f}/{MAXPOS} 只;对照持仓 {nh2.mean():.1f}")
        print(f"  零成本年化 **{g:+.2%}**;双边0.2%/往返 {gc:+.2%};"
              f"最大回撤 {mdd(np.cumprod(1+r1[0])):.1%}")
        print(f"  对照({NSEED} 组)中位 {cmed:+.2%} "
              f"[{np.percentile(cs,5):+.2%}, {np.percentile(cs,95):+.2%}]")
        print(f"  **超额 {ex*100:+.2f}pp,单尾 p {pv:.4f}**")
        print(f"  止损触发 {int(ns1[0])} 次,其中 21 日后仍高于买价 {int(wh1[0])} 次 "
              f"({wh1[0]/max(ns1[0],1):.1%})")
        print(f"  **B3(a) 买入日 20日日波动率中位 {np.nanmedian(vv):.4f}** "
              f"(第一五四节 6.05% → {'✓ 更低' if np.nanmedian(vv) < 0.0605 else '✗ 未更低'})")
        print(f"  **B3(b) 止损标准差倍数中位 {np.nanmedian(ss):.2f}σ** "
              f"(第一五四节 1.3σ → {'✓ 更远' if np.nanmedian(ss) > 1.3 else '✗ 未更远'})")
        if judge:
            a1, a2 = ex >= 0.03, pv < 0.05
            print(f"\n  **B2 主判据**:超额≥+3.00pp {'✓' if a1 else '✗'}"
                  f"({ex*100:+.2f}pp);p<0.05 {'✓' if a2 else '✗'}({pv:.4f}) → "
                  f"**{'通过' if (a1 and a2) else '不通过'}**")
        yr = pd.Series(idx[ta:tb + 1]).dt.year.to_numpy()
        print("  B4 逐年(策略 vs 对照中位):", end="")
        for y in sorted(set(yr)):
            mk = yr == y
            if mk.sum() < 60:
                continue
            a_ = float(np.prod(1 + r1[0][mk]) - 1)
            b_ = float(np.median(np.prod(1 + r2[:, mk], axis=1) - 1))
            print(f"  {y} {a_:+.1%}/{b_:+.1%}", end="")
            res.append({"段": f"{tag}·{y}", "年化": a_, "对照中位": b_,
                        "超额pp": (a_ - b_) * 100})
        print(flush=True)
        res.append({"段": tag, "买点": int(sum(len(v) for v in vols)),
                    "成交笔数": int(nt1[0]), "年化": g, "成本后": gc,
                    "对照中位": cmed, "超额pp": ex * 100, "p": pv,
                    "回撤": mdd(np.cumprod(1 + r1[0])),
                    "平均持仓": float(nh1[0].mean()), "止损次数": int(ns1[0]),
                    "止损后仍上涨": int(wh1[0]),
                    "买入日波动率中位": float(np.nanmedian(vv)),
                    "止损σ倍数中位": float(np.nanmedian(ss))})
    print(f"\n锚点B1c 行业恒等式违例 {viol[0]} {'✓' if viol[0] == 0 else '✗'}")
    pd.DataFrame(res).to_csv(f"{OUT}/platform_pivot.csv", index=False,
                             encoding="utf-8-sig")
    print(f"落库 {OUT}/platform_pivot.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
