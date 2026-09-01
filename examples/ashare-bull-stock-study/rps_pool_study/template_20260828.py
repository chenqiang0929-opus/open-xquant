"""§165 统一改用 RPS50 分档(承用户指令),用扩展到 2026-08-28 的面板,按 §163 已对齐的口径出当日模板清单。

**口径与 §163 逐字相同,只换三处:面板目录、锚点形状、输出区间。**
面板 = /home/user/oxq-panel-0828(2013-01-04 → 2026-08-28,含 19 个新交易日),
财务已按 publish_date 做 PIT 更新到中报。

原 §163 说明如下 ——


回函确认/修正的内容(逐条落实)
------------------------------
**A 我原来就对、不改的:**
- X01 四档与阈值(严格历史 listed_days≥250;反弹≥40%、120日收益≥10%、
  MA20持续度≥55%;RPS60≥80 标准确认、≥90 强确认;基础三条件满足但 RPS60<80 = 观察级)
- 平台 legacy 三项定义、触线日要求 **MA100 向上**、收敛比分子用**真实波幅 TR**
- **R09 核心质量分**:回函给出的正式 eligibility(净利率>0 / ROE>0 / ep_ttm>0 /
  ep&cfp>0 且 转换率>0)、1%/99% 缩尾后升序百分位、四项等权且任一缺失即整体缺失
  —— 与本地 `route_scores("R09")` + `wrank()` **逐行一致,不改**。
  **回函明确:案例表那一列没有严格复用正式 R09,是案例侧的问题,不与之对齐。**

**B 按回函修正的:**
1. **周线五态**改成他案例生成器的**有序判别**(先命中优先),与我原来的四象限不同:
   未知 → 多头趋势(周收>MA20周>MA60周) → 突破启动(周收≥MA20周 **且日线 ret_5 > 5%**)
   → 回踩修复(周收 ≥ 0.95×MA60周 **且** 周收 < MA20周) → 均线蓄势(|周收/MA20周−1| ≤ 5%)
   → 弱势结构。
   **两处关键差异**:突破启动要的是 `日线ret_5>5%` 而非 `MA20周≤MA60周`;
   回踩修复要的是 `周收 ≥ 0.95×MA60周` 而非 `MA20周>MA60周`。
2. **周边界**改成 `W-FRI` 重采样、取「周标签日期 ≤ 观察日」的最后一行 ——
   **周五收盘观察时使用当周**(我原来一律排除当周,是系统性差异)。
3. **RPS 横截面**改成他的口径:**5,217 只策略池中当日 N 日收益可算者**,
   **RPS 排名步骤不逐日剔除 ST/停牌/零成交**,最低横截面样本 100。
   (前复权与后复权的 N 日收益恒等,价格基准不影响 RPS。)
4. **成交量不做任何复权反调整**(回函:QFQ 只调 OHLC,volume 原样)——
   本地一直如此,不改;宁德时代的缩量比差异归因于底层面板数据本身,记录不强行对齐。
5. **字段命名按回函要求降级**:「质量等级」→ 案例展示分层;「周线五态」→ 案例辅助标签;
   另加**正式二元** `周线多头排列`(周收>MA20周>MA60周)。
6. 按回函第 8 节建议,新增 **首次触发日期 / 连续确认天数 / 触发状态**。

**C 回函未解决、本节保留原口径并标明的:**
- X01 提到「短历史最低要求 listed_days ≥ 121」,但未说明短历史下
  「距一年低点涨幅(250日)」如何计算。**本节一律用 ≥250**,与案例锚点一致;
  121 一档留待他补充。
- 胜宏科技 2023-09-12 的假突破,回函明确「不用于修改历史标签」,本节**不加过滤**。

锚点(不过则不出 2022 清单)
--------------------------
A. 面板 (3297, 5232);
B. 信号类型在 272 行可比案例上一致率 ≥ 95%(修正前 100.0%,不得退步);
C. 平台信号一致率 ≥ 95%(修正前 99.6%);
D. **周线五态一致率必须较修正前的 82.4% 有提升**(这是本节改动的直接检验)。
   质量等级**不再设锚点** —— 回函已确认案例侧那一列不是正式口径。

**本节不构成任何买入建议。平台信号仍为 WATCHLIST 研究状态。**
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codex_r10_neutral import CACHE  # noqa: E402
from codex_r10_replication import DATA as _D  # noqa: E402

DATA = os.environ.get("OXQ_PANEL_DIR", _D)
import codex_routes_rerun as _crr  # noqa: E402
from codex_routes_rerun import build_fund, route_scores  # noqa: E402
from panel_cache import cached  # noqa: E402
from platform_pivot import vec_screen  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
# 平台「强势日」的尺子。第一六八节按用户指令统一到 RPS50,故默认 50。
# 设 OXQ_STRONG_N=60 可退回原口径(第一五五节的全部数字挂在 RPS60 上)。
STRONG_N = int(os.environ.get("OXQ_STRONG_N", "50"))
PSTATE = f"{OUT}/platform_state.npz"
POOL = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/b8437f45-___20260831.xls")
CODEX50 = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/949ad4ba-____20260831_Claude_____RPS50.xlsx")
XL = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
      "0abc3d92-X01_____R09_________________v0.4____.xlsx")
MIN_XS = 100
COLS = ["样本类型", "观察日期", "股票代码", "股票名称", "收盘价", "统一信号",
        "信号类型", "信号理由", "首次触发日期", "连续确认天数", "触发状态",
        "平台信号", "周线多头排列", "案例展示分层_质量", "案例辅助标签_周线五态",
        "距一年低点涨幅", "近120日收益", "MA20持续度", "RPS50", "RPS60", "RPS250",
        "距一年高点价格差", "平台调整天数", "平台深度", "平台缩量比",
        "平台收敛比", "R09核心质量分"]


def weekly_wfri(cl, idx):
    """W-FRI 重采样;返回按「周标签 ≤ 观察日」映射到日频的 (周收, MA20周, MA60周)。"""
    df = pd.DataFrame(cl, index=idx)
    wk = df.resample("W-FRI").last()
    # 春节/国庆等整周无交易的周会生成全 NaN 行,min_periods=60 会被它一次打断
    # → 先剔掉"没有任何交易日的周",它们本来就不是周线上的一根 K。
    wk = wk.dropna(how="all")
    lab = wk.index
    wc = wk.to_numpy()
    m20 = wk.rolling(20, min_periods=20).mean().to_numpy()
    m60 = wk.rolling(60, min_periods=60).mean().to_numpy()
    k = np.searchsorted(lab.to_numpy(), idx.to_numpy(), side="right") - 1
    bad = k < 0
    k = np.clip(k, 0, len(lab) - 1)
    out = [wc[k], m20[k], m60[k]]
    for a in out:
        a[bad] = np.nan
    return out


def wstate5(wc, w20, w60, ret5):
    """Codex 案例生成器的有序五态判别(先命中优先)。"""
    s = np.full(wc.shape, "未知", object)
    ok = np.isfinite(w20) & np.isfinite(w60) & np.isfinite(wc)
    with np.errstate(all="ignore"):
        dv = wc / np.where(w20 > 0, w20, np.nan) - 1.0
    a = ok & (wc > w20) & (w20 > w60)
    s[a] = "多头趋势"
    b = ok & ~a & (wc >= w20) & np.isfinite(ret5) & (ret5 > 0.05)
    s[b] = "突破启动"
    c = ok & ~a & ~b & (wc >= 0.95 * w60) & (wc < w20)
    s[c] = "回踩修复"
    d = ok & ~a & ~b & ~c & np.isfinite(dv) & (np.abs(dv) <= 0.05)
    s[d] = "均线蓄势"
    s[ok & ~a & ~b & ~c & ~d] = "弱势结构"
    return s


def tier(rec, r120, mf, rps):
    base = (rec >= 0.40) & (r120 >= 0.10) & (mf >= 0.55)
    t = np.full(rec.shape, "无信号", object)
    t[base & (rps < 80)] = "观察级"
    t[base & (rps >= 80) & (rps < 90)] = "标准确认"
    t[base & (rps >= 90)] = "强确认"
    return t


def qtier(q):
    r = np.full(np.shape(q), "缺失", object)
    f = np.isfinite(q)
    r[f & (q < 0.30)] = "低"
    r[f & (q >= 0.30) & (q < 0.70)] = "中"
    r[f & (q >= 0.70)] = "高"
    return r


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]

    def _build_panel():
        cols = ["close", "volume", "is_st", "is_suspended", "listed_days"]
        d = {c: {} for c in cols}
        for c in codes:
            x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=cols)
            if getattr(x.index, "tz", None) is not None:
                x.index = x.index.tz_localize(None)
            for k in cols:
                d[k][c] = x[k]
        cldf_ = pd.DataFrame(d["close"]).sort_index()
        idx_ = cldf_.index

        def al_(k, f=np.nan):
            return pd.DataFrame(d[k]).sort_index().reindex(
                index=idx_, columns=cldf_.columns).fillna(f).to_numpy()
        cl_ = cldf_.where(cldf_ > 0).ffill().to_numpy(np.float64)
        # 源数据缺陷:mktdata_enriched 里 588 只科创板(688)的 listed_days 在近期为 0,
        # 其余三个板 0%。直接用会把整个科创板判成"上市不足250日"而全部剔除。
        # 修正:把 0 视为缺失、按股票前向填充最近一个有效值(仍不足则保持 0)。
        ld_ = al_("listed_days", 0)
        ldf_ = pd.DataFrame(ld_).replace(0, np.nan).ffill().fillna(0).to_numpy()
        print(f"listed_days 修正:末日为 0 的 {int((ld_[-1] == 0).sum())} 只 → "
              f"修正后 {int((ldf_[-1] == 0).sum())} 只", flush=True)
        okm_ = (~al_("is_st", True).astype(bool)
                & ~al_("is_suspended", True).astype(bool)
                & (ldf_ >= 250) & (al_("volume", 0) > 0)
                & np.isfinite(cl_))
        return {"idx": idx_.values.astype("datetime64[ns]"), "cl": cl_, "okm": okm_,
                "vol": al_("volume", np.nan)}
    _p = cached("panel", DATA, _build_panel)
    idx = pd.DatetimeIndex(_p["idx"])
    cl, okm = _p["cl"], _p["okm"]
    nt, ns = cl.shape
    assert (nt, ns) == (3316, 5232), f"锚点A {(nt, ns)}"
    px = pd.DataFrame(cl)
    lo250 = px.rolling(250, min_periods=250).min().to_numpy()
    hi250 = px.rolling(250, min_periods=250).max().to_numpy()
    ma20 = px.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        gap = cl / np.where(hi250 > 0, hi250, np.nan) - 1.0
        mfr = pd.DataFrame((cl > ma20).astype(np.float64)).where(
            np.isfinite(ma20)).rolling(120, min_periods=120).mean().to_numpy()
        r120 = px.pct_change(120).to_numpy()
        r60 = px.pct_change(60).to_numpy()
        r250 = px.pct_change(250).to_numpy()
        r50 = px.pct_change(50).to_numpy()
        ret5 = px.pct_change(5).to_numpy()
    del lo250, hi250, ma20

    # ---- 修正3:RPS 按 Codex 口径 —— 5,217 策略池、不逐日过滤、最低横截面 100 ----
    z = np.load(CACHE, allow_pickle=True)
    zc = list(z["codes"])
    zi = pd.DatetimeIndex(z["idx"])
    assert (zi == idx[:len(zi)]).all(), "R09 缓存日期前缀不一致"
    inpool = np.array([c in set(zc) for c in codes])
    print(f"RPS 池 {int(inpool.sum())} 只", flush=True)

    def rps_codex(r):
        v = np.where(inpool[None, :] & np.isfinite(r), r, np.nan)
        n = np.isfinite(v).sum(axis=1)
        out = pd.DataFrame(v).rank(axis=1, pct=True,
                                   method="average").to_numpy() * 100.0
        out[n < MIN_XS] = np.nan
        return out
    rps50, rps60, rps250 = rps_codex(r50), rps_codex(r60), rps_codex(r250)
    del r60, r250, r50
    print(f"RPS(Codex 口径,池 {int(inpool.sum()):,} 只)完成 "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 修正1&2:周线 W-FRI + 有序五态 ----
    wc, w20, w60 = weekly_wfri(cl, idx)
    wbull = np.isfinite(w20) & np.isfinite(w60) & (wc > w20) & (w20 > w60)
    w5 = wstate5(wc, w20, w60, ret5)
    del wc, w20, w60, ret5

    # 平台状态:缓存是旧面板(3297 行)的,必须在扩展面板上重算
    from consolidation_screener import (  # noqa: PLC0415
        THR_ATR,
        THR_DEPTH,
        THR_SHRINK,
        load_panel,
    )
    def _build_plat():
        pcl, pframes, pstrong, pma100 = load_panel(DATA)
        if STRONG_N != 60:
            # 第一六八节:平台「强势日」按用户指令统一到 RPS50 >= 90。
            # **必须在剔除 510300 之前算** —— RPS 是 axis=1 的横截面分位,
            # 少一列会让每一只的分位都变(实测差 358 点)。
            # fill_method 显式写 'pad',与 load_panel 里 pct_change 的默认值一致。
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                clw = pcl.where(pcl > 0)
                rp = (clw.pct_change(STRONG_N, fill_method="pad")
                      .rank(axis=1, pct=True) * 100)
                r60 = clw.pct_change(60, fill_method="pad").rank(axis=1, pct=True) * 100
            # 恒等断言:同一条路径在 N=60 下必须逐点复现 load_panel,
            # 过了才说明「只有尺子变了、实现没动」(第一六八节 C1(e-3) 同款)。
            assert np.array_equal((r60 > 90).to_numpy(), pstrong), (
                f"强势日恒等断言不过:自算 RPS60 {int((r60 > 90).to_numpy().sum())} "
                f"vs load_panel {int(pstrong.sum())}")
            pstrong = (rp >= 90).to_numpy()      # Codex 口径是 >= 90,不是 > 90
            print(f"【口径变更】平台强势日 RPS{STRONG_N} >= 90:"
                  f"{int(pstrong.sum()):,} 点(恒等断言已过)", flush=True)
        if "510300" in pcl.columns:
            keep = [i for i, c in enumerate(pma100.columns) if c != "510300"]
            pcl = pcl.drop(columns=["510300"])
            pstrong = pstrong[:, keep]
        assert list(pcl.columns) == codes, "平台面板列顺序与主面板不一致"
        assert (pcl.index == idx).all(), "平台面板日期与主面板不一致"
        a, b, c_, d_, e_, f_, g_ = vec_screen(
            pcl.to_numpy(float), pframes, pstrong, pma100, idx, codes)
        del pframes
        # dtype 必须原样带走:phi/plo 是 float64——第一五五节的真 bug 就是
        # float32 在「打平」处把等于误判成突破,虚增留出段约 4pp。
        return {"ts_a": a, "adj_a": b, "dep": c_, "shr": d_, "cnv": e_,
                "phi": f_, "plo": g_}
    _q = cached("platform", DATA, _build_plat, extra=f"rps{STRONG_N}")
    ts_a, adj_a = _q["ts_a"], _q["adj_a"]
    dep, shr, cnv, phi, plo = _q["dep"], _q["shr"], _q["cnv"], _q["phi"], _q["plo"]
    assert phi.dtype == np.float64 and plo.dtype == np.float64, "平台上下沿必须 float64"
    hit3 = ((shr < THR_SHRINK) & (cnv < THR_ATR) & (dep <= THR_DEPTH)
            & (adj_a >= 0))
    up_prev = np.full((nt, ns), np.nan, np.float64)
    lo_prev = np.full((nt, ns), np.nan, np.float64)
    same = np.zeros((nt, ns), bool)
    same[1:] = ts_a[1:] == ts_a[:-1]
    up_prev[1:] = np.where(same[1:], phi[:-1], np.nan)
    lo_prev[1:] = np.where(same[1:], plo[:-1], np.nan)
    brk = (hit3 & okm & np.isfinite(up_prev) & (cl > up_prev)
           & np.isfinite(lo_prev))
    print(f"平台状态重算完成:三条全中 {int((hit3 & okm).sum()):,} 点、"
          f"突破买点 {int(brk.sum()):,} 个 ({time.time()-t0:.0f}s)", flush=True)
    # 【口径变更】X01 分档改用 RPS50(用户指令,与 Codex 的
    # rule_version=claude_rps50_weekly_v1.0 对齐);RPS60 仍照常输出供交叉核对。
    # **平台强势日也已统一到 RPS50**(第一六八节):重跑第一五五节全套后
    # 留出段年化 +15.83%→+13.02%、超额 −2.46pp→−5.27pp、p 0.656→0.810,
    # 主判据仍不通过 —— 但 RPS60 下本来就不通过,结论未变。
    tt = tier(rec, r120, mfr, rps50)
    uni = np.isin(tt, ("标准确认", "强确认"))
    psig = np.where(brk, "平台突破（研究）", np.where(hit3, "平台观察", "无平台信号"))

    # ---- 修正6:首次触发日 / 连续确认天数 / 触发状态(日频)----
    run = np.zeros((nt, ns), np.int32)
    for t in range(1, nt):
        run[t] = np.where(uni[t], run[t - 1] + 1, 0)
    run[0] = uni[0].astype(np.int32)
    print(f"价格/周线/平台字段就绪 ({time.time()-t0:.0f}s)", flush=True)

    # ---- R09 正式口径(与回函一致,不改)----
    # R09 缓存止于 2026-08-03,且**它自己的最后一行是残的**
    # (合格 3,069 只 vs 前 250 日中位 4,788)—— 不能拿它填充,否则新增日全部被腰斩。
    # 处理:logcap/tmean 只用于 isfinite 过滤,从**最后一个干净行**延展;
    #       OK 直接在扩展面板上按同一定义重算(非ST、非停牌、上市满250日、有成交、价格有效)。
    okc = z["OK"]
    ssum = okc.sum(1)
    med = float(np.median(ssum[-250:]))
    clean = int(np.max(np.flatnonzero(ssum >= 0.9 * med)))
    print(f"R09 缓存末行合格 {ssum[-1]:,}(前250日中位 {med:,.0f}),"
          f"最后一个干净行 = {pd.Timestamp(z['idx'][clean]).date()}", flush=True)

    def _pad(a, src):
        if a.shape[0] >= nt:
            return a[:nt]
        return np.vstack([a, np.repeat(a[src:src + 1], nt - a.shape[0], axis=0)])
    logcap, tmean = _pad(z["LOGCAP"], clean), _pad(z["TMEAN"], clean)
    zcl = _pad(z["CL"], clean)
    zback = np.array([{c: i for i, c in enumerate(codes)}.get(c, -1) for c in zc])
    zok = np.zeros((nt, len(zc)), bool)
    gb = zback >= 0
    zok[:, gb] = okm[:, zback[gb]]          # 全程用扩展面板重算,不用缓存的 OK
    print(f"OK 改为按扩展面板重算:末日合格 {int(zok[nt-1].sum()):,} 只", flush=True)
    def _build_fund():
        raw_ = np.full((nt, len(zc)), np.nan, np.float32)
        for j, c in enumerate(zc):
            x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
            if getattr(x.index, "tz", None) is not None:
                x.index = x.index.tz_localize(None)
            raw_[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
                lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
        # 【修正】build_fund 用的是 codex_routes_rerun 的模块级 DATA(= 旧面板),
        # 不认 OXQ_PANEL_DIR。旧面板止于 2026-08-03,而中报在 8 月中旬才披露 ——
        # 实测抽样 494 只里 **334 只(67.6%)** 两张面板的末行 eps 不同
        # (000001:旧 0.67 一季报 vs 新 1.24 中报)。不改就是拿一季报算 R08/R09。
        _crr.DATA = DATA
        fm_, abad_ = build_fund(zc, idx)
        assert abad_ == 0, "TTM 恒等式不过"
        out = {"raw": raw_, "_abad": np.array([abad_])}
        out.update({f"fm_{k}": v for k, v in fm_.items()})
        return out
    _f = cached("fund", DATA, _build_fund)
    raw = _f["raw"]
    fm = {k[3:]: v for k, v in _f.items() if k.startswith("fm_")}
    assert int(_f["_abad"][0]) == 0, "TTM 恒等式不过(缓存)"
    zpos = {c: j for j, c in enumerate(zc)}
    zmap = np.array([zpos.get(c, -1) for c in codes])
    gm = zmap >= 0
    qcache = {}

    def qscore(t):
        if t not in qcache:
            e = np.flatnonzero(zok[t] & np.isfinite(logcap[t])
                               & np.isfinite(tmean[t]))
            v = np.full(len(zc), np.nan)
            if len(e):
                v[e] = route_scores("R09", t, e, fm, zcl, raw, logcap, tmean, "raw")
            o = np.full(ns, np.nan)
            o[gm] = v[zmap[gm]]
            qcache[t] = o
        return qcache[t]
    print(f"R09 因子面板就绪 ({time.time()-t0:.0f}s)", flush=True)

    pos = {c: j for j, c in enumerate(codes)}
    ip = pd.Index(idx)
    try:
        with open(f"{OUT}/code_name_map_wide.json", encoding="utf-8") as fh:
            nmap = json.load(fh)
    except OSError:
        nmap = {}

    def row(kind, t, j):
        q = float(qscore(t)[j])
        r = int(run[t, j])
        ft = idx[t - r + 1].date() if r > 0 else None
        return {"样本类型": kind, "观察日期": idx[t].date(), "股票代码": codes[j],
                "股票名称": nmap.get(codes[j], ""), "收盘价": cl[t, j],
                "统一信号": int(uni[t, j]), "信号类型": tt[t, j],
                "信号理由": "；".join(x for x in (
                    "反弹≥40%" if rec[t, j] >= .40 else "",
                    "120日收益≥10%" if r120[t, j] >= .10 else "",
                    "MA20持续度≥55%" if mfr[t, j] >= .55 else "",
                    f"RPS50={rps50[t, j]:.1f}" if np.isfinite(rps50[t, j]) else "",
                ) if x),
                "首次触发日期": ft, "连续确认天数": r,
                "触发状态": ("新触发" if r == 1 else "持续" if r > 1 else "未触发"),
                "平台信号": psig[t, j], "周线多头排列": bool(wbull[t, j]),
                "案例展示分层_质量": qtier(np.array([q]))[0],
                "案例辅助标签_周线五态": w5[t, j],
                "距一年低点涨幅": rec[t, j], "近120日收益": r120[t, j],
                "MA20持续度": mfr[t, j], "RPS50": rps50[t, j], "RPS60": rps60[t, j],
                "RPS250": rps250[t, j],
                "距一年高点价格差": gap[t, j],
                # 平台调整天数 = 当日距最近强势日的交易日数;-1 表示无平台
                "平台调整天数": (int(adj_a[t, j]) if adj_a[t, j] >= 0
                             else np.nan),
                "平台深度": dep[t, j],
                "平台缩量比": shr[t, j], "平台收敛比": cnv[t, j],
                "R09核心质量分": q}

    # ---- 锚点 ----
    # 【变更说明】原「信号类型 276/276」是在 **RPS60** 尺子下对 Codex v0.4 案例工作簿量的;
    # 本节分档改 RPS50 后该锚点必然不再成立,**不是退步**。
    # 平台与周线字段未变,仍对 v0.4 案例复核;信号类型改为与 Codex 2026-08-28
    # 的 662 只 RPS50 版结果逐只比对。
    ca = pd.read_excel(XL, sheet_name="案例摘要", header=3).dropna(how="all")
    ca["股票代码"] = ca["股票代码"].astype(str).str.split(".").str[0].str.zfill(6)
    ca["观察日期"] = pd.to_datetime(ca["观察日期"])
    ck = []
    for _, r in ca.iterrows():
        j, t = pos.get(r["股票代码"]), int(ip.searchsorted(r["观察日期"]))
        if j is None or t >= nt or idx[t] != r["观察日期"]:
            continue
        ck.append({"代码": r["股票代码"], "日期": r["观察日期"].date(),
                   "他_平台信号": r["平台信号"], "我_平台信号": psig[t, j],
                   "他_周线": r["周线结构"], "我_周线": w5[t, j]})
    ck = pd.DataFrame(ck)
    w = 92
    print(f"\n{'='*w}\n锚点一:v0.4 案例 {len(ck)} 行(平台与周线字段未变)\n{'='*w}")
    for k, thr in (("平台信号", 0.95), ("周线", 0.95)):
        m = ck[f"他_{k}"].astype(str) == ck[f"我_{k}"].astype(str)
        print(f"  {k:<8}{int(m.sum()):>4}/{len(ck)} = {m.mean():>6.1%}  "
              f"{'✓' if m.mean() >= thr else '✗'}")
    ck.to_csv(f"{OUT}/template_20260828_anchor.csv", index=False,
              encoding="utf-8-sig")
    try:
        cx = pd.read_excel(CODEX50, sheet_name="全部清单", header=1,
                           dtype={"股票代码": str})
        cx = cx[cx["股票代码"].notna()].copy()
        cx["股票代码"] = cx["股票代码"].astype(str).str.split(".").str[0].str.zfill(6)
        cmp_ = []
        for _, r in cx.iterrows():
            j = pos.get(r["股票代码"])
            if j is None:
                continue
            cmp_.append({"代码": r["股票代码"], "他": r["信号类型"],
                         "我": tt[nt - 1, j], "他RPS50": r.get("RPS50"),
                         "我RPS50": rps50[nt - 1, j],
                         "我合格": bool(okm[nt - 1, j])})
        cd = pd.DataFrame(cmp_)
        sg = cd[cd["他"].isin(["强确认", "标准确认", "观察级"])]
        m = (sg["他"] == sg["我"])
        print(f"\n{'='*w}\n锚点二:与 Codex 2026-08-28 RPS50 版逐只比对"
              f"(他池 {len(cd)} 只)\n{'='*w}")
        print(f"  他有信号 {len(sg)} 只,信号类型完全一致 {int(m.sum())} = {m.mean():.1%}")
        print(pd.crosstab(sg["他"], sg["我"]).to_string())
        a5 = pd.to_numeric(sg["他RPS50"], errors="coerce").to_numpy()
        b5 = sg["我RPS50"].to_numpy(float)
        g5 = np.isfinite(a5) & np.isfinite(b5)
        if g5.sum():
            print(f"  RPS50 数值:可比 {int(g5.sum())};中位|差| "
                  f"{np.median(np.abs(a5[g5]-b5[g5])):.3f};"
                  f"|差|<3 占 {(np.abs(a5[g5]-b5[g5]) < 3).mean():.1%}")
        cd.to_csv(f"{OUT}/template_20260828_vs_codex.csv", index=False,
                  encoding="utf-8-sig")
    except Exception as ex:                                    # noqa: BLE001
        print(f"\n锚点二跳过:{ex}")

    tl = nt - 1
    assert str(idx[tl].date()) == "2026-08-28", f"末日 {idx[tl].date()}"
    e = np.flatnonzero(okm[tl] & (np.isin(tt[tl], ("观察级", "标准确认", "强确认"))
                                  | (psig[tl] != "无平台信号")))
    rows = [row("当日观察", tl, int(j)) for j in e]
    print(f"\n观察日 {idx[tl].date()};全市场合格 {int(okm[tl].sum()):,} 只;"
          f"有信号 {len(rows):,} 只", flush=True)
    out = pd.DataFrame(rows)[COLS].sort_values(
        ["观察日期", "信号类型", "RPS60"], ascending=[True, True, False])
    out.to_csv(f"{OUT}/template_20260828.csv", index=False,
               encoding="utf-8-sig")

    # ---- 次新股池全量输出(含无信号与数据不足,对齐 Codex 的 662 行格式)----
    pl = pd.read_csv(POOL, sep="\t", encoding="gbk", dtype=str)
    pl = pl.rename(columns={pl.columns[0]: "代码"})
    pl["代码"] = (pl["代码"].astype(str).str.replace('="', "", regex=False)
                  .str.replace('"', "", regex=False).str.strip().str.zfill(6))
    pl = pl[pl["代码"].str.fullmatch(r"\d{6}")].drop_duplicates("代码")
    prows, miss = [], []
    for c in pl["代码"]:
        j = pos.get(c)
        if j is None:
            miss.append((c, "面板无此股"))
            continue
        r = row("池内观察", tl, int(j))
        why = []
        if not okm[tl, j]:
            why.append("不合格(ST/停牌/上市不足250日/无成交)")
        for nm_, v in (("距一年低点涨幅", rec[tl, j]), ("近120日收益", r120[tl, j]),
                       ("MA20持续度", mfr[tl, j]), ("RPS50", rps50[tl, j])):
            if not np.isfinite(v):
                why.append(f"{nm_}缺失")
        r["数据状态"] = "正常" if not why else "数据不足:" + "、".join(why)
        if why:
            r["统一信号"] = None
            r["信号类型"] = "数据不足"
        prows.append(r)
    for c, w_ in miss:
        prows.append({**{k: None for k in COLS}, "样本类型": "池内观察",
                      "观察日期": idx[tl].date(), "股票代码": c,
                      "信号类型": "数据不足", "数据状态": f"数据不足:{w_}"})
    po = pd.DataFrame(prows)[[*COLS, "数据状态"]]
    po.to_csv(f"{OUT}/pool_20260828.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'='*w}\n次新股池全量({len(pl)} 只)\n{'='*w}")
    print("  " + po["信号类型"].value_counts().to_string().replace("\n", "\n  "))
    ok_ = po[po["信号类型"] != "数据不足"]
    print(f"  可评估 {len(ok_)} 只;统一信号=1 共 "
          f"{int(pd.to_numeric(ok_['统一信号'], errors='coerce').sum())} 只")
    print("  平台信号:")
    print("    " + ok_["平台信号"].value_counts().to_string().replace("\n", "\n    "))
    print(f"  落库 {OUT}/pool_20260828.csv")
    print(f"\n{'='*w}\n2026-08-28 当日清单\n{'='*w}")
    print(f"  总行数 {len(out):,};涉及 {out['股票代码'].nunique():,} 只")
    for c in ("信号类型", "平台信号", "案例展示分层_质量",
              "案例辅助标签_周线五态", "触发状态"):
        print(f"\n  {c}:")
        print("    " + out[c].value_counts().to_string().replace("\n", "\n    "))
    print(f"\n  周线多头排列(正式二元)为真:{int(out['周线多头排列'].sum()):,} 行"
          f"({out['周线多头排列'].mean():.1%})")
    print(f"\n  统一信号=1:{int(out['统一信号'].sum()):,} 只")
    print(f"\n落库 {OUT}/template_20260828.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
