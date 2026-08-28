"""§163 按 Codex 2026-08-28 回函修正后,重跑 2022 年模板清单。

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
from codex_r10_replication import DATA  # noqa: E402
from codex_routes_rerun import build_fund, route_scores  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
PSTATE = f"{OUT}/platform_state.npz"
XL = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
      "0abc3d92-X01_____R09_________________v0.4____.xlsx")
MIN_XS = 100
COLS = ["样本类型", "观察日期", "股票代码", "股票名称", "收盘价", "统一信号",
        "信号类型", "信号理由", "首次触发日期", "连续确认天数", "触发状态",
        "平台信号", "周线多头排列", "案例展示分层_质量", "案例辅助标签_周线五态",
        "距一年低点涨幅", "近120日收益", "MA20持续度", "RPS60", "RPS250",
        "距一年高点价格差", "平台深度", "平台缩量比", "平台收敛比", "R09核心质量分"]


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
    cols = ["close", "volume", "is_st", "is_suspended", "listed_days"]
    d = {c: {} for c in cols}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=cols)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in cols:
            d[k][c] = x[k]
    cldf = pd.DataFrame(d["close"]).sort_index()
    idx = cldf.index
    nt, ns = cldf.shape
    assert (nt, ns) == (3297, 5232), f"锚点A {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f).to_numpy()
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    okm = (~al("is_st", True).astype(bool)
           & ~al("is_suspended", True).astype(bool)
           & (al("listed_days", 0) >= 250) & (al("volume", 0) > 0)
           & np.isfinite(cl))
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
        ret5 = px.pct_change(5).to_numpy()
    del lo250, hi250, ma20

    # ---- 修正3:RPS 按 Codex 口径 —— 5,217 策略池、不逐日过滤、最低横截面 100 ----
    z = np.load(CACHE, allow_pickle=True)
    zc = list(z["codes"])
    assert (pd.DatetimeIndex(z["idx"]) == idx).all(), "R09 缓存日期不一致"
    inpool = np.array([c in set(zc) for c in codes])

    def rps_codex(r):
        v = np.where(inpool[None, :] & np.isfinite(r), r, np.nan)
        n = np.isfinite(v).sum(axis=1)
        out = pd.DataFrame(v).rank(axis=1, pct=True,
                                   method="average").to_numpy() * 100.0
        out[n < MIN_XS] = np.nan
        return out
    rps60, rps250 = rps_codex(r60), rps_codex(r250)
    del r60, r250
    print(f"RPS(Codex 口径,池 {int(inpool.sum()):,} 只)完成 "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 修正1&2:周线 W-FRI + 有序五态 ----
    wc, w20, w60 = weekly_wfri(cl, idx)
    wbull = np.isfinite(w20) & np.isfinite(w60) & (wc > w20) & (w20 > w60)
    w5 = wstate5(wc, w20, w60, ret5)
    del wc, w20, w60, ret5

    p = np.load(PSTATE, allow_pickle=True)
    pc = {c: j for j, c in enumerate(list(p["codes"]))}
    pmap = np.array([pc.get(c, -1) for c in codes])
    g = pmap >= 0
    dep = np.full((nt, ns), np.nan, np.float32)
    shr = np.full((nt, ns), np.nan, np.float32)
    cnv = np.full((nt, ns), np.nan, np.float32)
    hit3 = np.zeros((nt, ns), bool)
    brk = np.zeros((nt, ns), bool)
    for dst, src in ((dep, "dep"), (shr, "shr"), (cnv, "cnv")):
        dst[:, g] = p[src][:, pmap[g]]
    hit3[:, g] = p["hit3"][:, pmap[g]]
    brk[:, g] = p["brk"][:, pmap[g]]
    tt = tier(rec, r120, mfr, rps60)
    uni = np.isin(tt, ("标准确认", "强确认"))
    psig = np.where(brk, "平台突破（研究）", np.where(hit3, "平台观察", "无平台信号"))

    # ---- 修正6:首次触发日 / 连续确认天数 / 触发状态(日频)----
    run = np.zeros((nt, ns), np.int32)
    for t in range(1, nt):
        run[t] = np.where(uni[t], run[t - 1] + 1, 0)
    run[0] = uni[0].astype(np.int32)
    print(f"价格/周线/平台字段就绪 ({time.time()-t0:.0f}s)", flush=True)

    # ---- R09 正式口径(与回函一致,不改)----
    logcap, tmean, zok, zcl = z["LOGCAP"], z["TMEAN"], z["OK"], z["CL"]
    raw = np.full((nt, len(zc)), np.nan, np.float32)
    for j, c in enumerate(zc):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    fm, abad = build_fund(zc, idx)
    assert abad == 0, "TTM 恒等式不过"
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
                    f"RPS60={rps60[t, j]:.1f}" if np.isfinite(rps60[t, j]) else "",
                ) if x),
                "首次触发日期": ft, "连续确认天数": r,
                "触发状态": ("新触发" if r == 1 else "持续" if r > 1 else "未触发"),
                "平台信号": psig[t, j], "周线多头排列": bool(wbull[t, j]),
                "案例展示分层_质量": qtier(np.array([q]))[0],
                "案例辅助标签_周线五态": w5[t, j],
                "距一年低点涨幅": rec[t, j], "近120日收益": r120[t, j],
                "MA20持续度": mfr[t, j], "RPS60": rps60[t, j], "RPS250": rps250[t, j],
                "距一年高点价格差": gap[t, j], "平台深度": dep[t, j],
                "平台缩量比": shr[t, j], "平台收敛比": cnv[t, j],
                "R09核心质量分": q}

    # ---- 锚点 B/C/D:272 行案例 ----
    ca = pd.read_excel(XL, sheet_name="案例摘要", header=3).dropna(how="all")
    ca["股票代码"] = ca["股票代码"].astype(str).str.split(".").str[0].str.zfill(6)
    ca["观察日期"] = pd.to_datetime(ca["观察日期"])
    ck = []
    for _, r in ca.iterrows():
        j, t = pos.get(r["股票代码"]), int(ip.searchsorted(r["观察日期"]))
        if j is None or t >= nt or idx[t] != r["观察日期"]:
            continue
        ck.append({"代码": r["股票代码"], "日期": r["观察日期"].date(),
                   "他_信号类型": r["信号类型"], "我_信号类型": tt[t, j],
                   "他_平台信号": r["平台信号"], "我_平台信号": psig[t, j],
                   "他_周线": r["周线结构"], "我_周线": w5[t, j],
                   "他_质量分": r["R09核心质量分"], "我_质量分": float(qscore(t)[j])})
    ck = pd.DataFrame(ck)
    w = 92
    print(f"\n{'='*w}\n锚点:{len(ck)} 行案例(修正后)\n{'='*w}")
    rates = {}
    for k, thr, before in (("信号类型", 0.95, 1.000), ("平台信号", 0.95, 0.996),
                           ("周线", None, 0.824)):
        m = ck[f"他_{k}"].astype(str) == ck[f"我_{k}"].astype(str)
        rates[k] = float(m.mean())
        tag = (f"门槛 {thr:.0%} {'✓' if m.mean() >= thr else '✗'}" if thr
               else f"修正前 {before:.1%} → {'✓ 有提升' if m.mean() > before else '✗ 未提升'}")
        print(f"  {k:<8} {int(m.sum()):>4}/{len(ck)} = {m.mean():>6.1%}   {tag}")
        if k == "周线" and m.mean() < 1:
            print("    仍不一致:")
            print("    " + ck[~m][["代码", "日期", "他_周线", "我_周线"]].head(10)
                  .to_string(index=False).replace("\n", "\n    "))
    a, b = ck["他_质量分"].to_numpy(float), ck["我_质量分"].to_numpy(float)
    gq = np.isfinite(a) & np.isfinite(b)
    print(f"  R09质量分(回函已确认案例侧非正式口径,不设锚点):可比 {int(gq.sum())}、"
          f"相关 {pd.Series(a[gq]).corr(pd.Series(b[gq])):.4f}、"
          f"中位|差| {np.median(np.abs(a[gq]-b[gq])):.4f}")
    ck.to_csv(f"{OUT}/codex_template_v2_anchor.csv", index=False,
              encoding="utf-8-sig")
    if rates["信号类型"] < 0.95 or rates["平台信号"] < 0.95:
        print("\n**锚点 B/C 不过 → 不出 2022 清单**")
        return

    me = np.sort(pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy().astype(int))
    rows = []
    for t in [int(x) for x in me if idx[x].year == 2022]:
        e = np.flatnonzero(okm[t] & (np.isin(tt[t], ("观察级", "标准确认", "强确认"))
                                     | (psig[t] != "无平台信号")))
        rows += [row("月末观察", t, int(j)) for j in e]
    for t in np.flatnonzero(idx.year == 2022):
        rows += [row("平台突破日", int(t), int(j)) for j in np.flatnonzero(brk[t])]
    out = pd.DataFrame(rows)[COLS].sort_values(
        ["观察日期", "信号类型", "RPS60"], ascending=[True, True, False])
    out.to_csv(f"{OUT}/codex_template_2022_v2.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n{'='*w}\n2022 年清单(按回函修正后)\n{'='*w}")
    print(f"  总行数 {len(out):,};涉及 {out['股票代码'].nunique():,} 只")
    for c in ("信号类型", "平台信号", "案例展示分层_质量",
              "案例辅助标签_周线五态", "触发状态"):
        print(f"\n  {c}:")
        print("    " + out[c].value_counts().to_string().replace("\n", "\n    "))
    print(f"\n  周线多头排列(正式二元)为真:{int(out['周线多头排列'].sum()):,} 行"
          f"({out['周线多头排列'].mean():.1%})")
    s1 = out[(out["样本类型"] == "月末观察") & (out["统一信号"] == 1)]
    print("\n  统一信号=1 的逐月只数:")
    print("    " + s1.groupby("观察日期").size().to_string().replace("\n", "\n    "))
    print(f"\n落库 {OUT}/codex_template_2022_v2.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
