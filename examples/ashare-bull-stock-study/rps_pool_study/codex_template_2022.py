"""§162 复现交付:完全按 Codex 监控模板 v0.4 的字段口径,在本地面板上跑 2022 年带信号的清单。

**本脚本是复现 + 名单交付,不是假设检验,不设通过/不通过判据。**
但有一个**硬锚点**:先用他自己的 276 行案例验证我对他口径的重建,重建不对就不出 2022。

他没有写死、由我从案例数据反解的三处口径(必须标明)
--------------------------------------------------
1. **观察级**:字段字典只写「无信号、观察级、标准确认、强确认」,没给门槛。
   实测案例:观察级 RPS60 ∈ [8.5, 77.2]、标准确认 [84.4, 89.0]、强确认 [90.2, 99.9],
   且观察级的反弹/120日收益/MA20持续度全部达标 →
   **观察级 = 三个观察条件全中 且 RPS60 < 80**。
2. **质量等级**:字典只写「按R09质量排名分层:高/中/低/缺失」。
   实测三档区间无重叠:低 [0.2124, 0.2181]、中 [0.3406, 0.6897]、高 [0.7090, 0.8647] →
   **≥0.70 高;0.30~0.70 中;<0.30 低;四因子任一缺失 → 缺失**。
3. **周线结构**:字典只列 5 个状态名,没给判别式。实测:
   多头趋势 C>MA20周 100%、MA20周>MA60周 100%;突破启动 C>MA20周 100%、MA20周>MA60周 **0%**;
   回踩修复 C>MA20周 **0%**、MA20周>MA60周 93%;均线蓄势 |MA20周乖离| ≤ 5% 且 C<MA60周 →
   **按上述四象限 + 乖离带重建**。周线一律只用**已完成周**,不含当周,避免前视。

其余口径逐字沿用他的字典
------------------------
- 标准确认:反弹≥40% 且 近120日收益≥10% 且 近120日站上MA20比例≥55% 且 RPS60≥80
- 强确认:同上,RPS60≥90;统一信号 = 标准确认或强确认记 1
- 平台信号:三条全中且严格突破平台上沿 → 平台突破(研究);三条全中 → 平台观察;否则无平台信号
- R09核心质量分:净利率/ROE/ROE同比变化/现金利润转换 四个横截面分位的等权综合,
  任一缺失则整体缺失(严格复用 `codex_routes_rerun.route_scores("R09")` 的冻结口径)

锚点(不过则不出 2022 清单)
--------------------------
A. 面板 (3297, 5232);
B. **信号类型在他 272 行可比案例上的一致率 ≥ 95%**;
C. **平台信号一致率 ≥ 95%**;
D. **质量等级一致率 ≥ 90%**(R09 因子面板口径差异容忍度略放宽,因财务缓存来源不同)。

产出
----
`codex_template_2022.csv` / `.xlsx`:2022 年**每月最后交易日 + 平台突破日**,
输出所有「有信号」的行(统一信号=1、或观察级、或有平台信号),
字段与他「案例摘要」页逐列对齐。

**本节不构成任何买入建议。**
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
from codex_r10_neutral import CACHE  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from codex_routes_rerun import build_fund, route_scores  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
PSTATE = f"{OUT}/platform_state.npz"
XL = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
      "0abc3d92-X01_____R09_________________v0.4____.xlsx")
COLS = ["样本类型", "观察日期", "股票代码", "股票名称", "收盘价", "统一信号",
        "信号类型", "平台信号", "质量等级", "距一年低点涨幅", "近120日收益",
        "MA20持续度", "RPS60", "RPS250", "距一年高点价格差", "周线结构",
        "平台深度", "平台缩量比", "平台收敛比", "R09核心质量分", "信号理由"]


def weekly(cl, idx):
    """已完成周的周收盘、MA20周、MA60周,前向映射到日频(不含当周,无前视)。"""
    wk = pd.Series(np.arange(len(idx)), index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last()
    wp = np.sort(wk.to_numpy().astype(int))
    wc = cl[wp]
    df = pd.DataFrame(wc)
    m20 = df.rolling(20, min_periods=20).mean().to_numpy()
    m60 = df.rolling(60, min_periods=60).mean().to_numpy()
    nt, ns = cl.shape
    out = [np.full((nt, ns), np.nan) for _ in range(3)]
    k = -1
    for t in range(nt):
        while k + 1 < len(wp) and wp[k + 1] < t:      # 严格 < t:只用已完成周
            k += 1
        if k >= 0:
            out[0][t], out[1][t], out[2][t] = wc[k], m20[k], m60[k]
    return out


def wstruct(c, m20, m60):
    """五态:按案例数据反解的四象限 + 乖离带。"""
    with np.errstate(all="ignore"):
        dev = c / np.where(m60 > 0, m60, np.nan) - 1.0
        dv20 = c / np.where(m20 > 0, m20, np.nan) - 1.0
    s = np.full(c.shape, "", object)
    ok = np.isfinite(c) & np.isfinite(m20) & np.isfinite(m60)
    a, b = c > m20, m20 > m60
    s[ok & a & b] = "多头趋势"
    s[ok & a & ~b] = "突破启动"
    s[ok & ~a & b] = "回踩修复"
    rest = ok & ~a & ~b
    s[rest & (np.abs(dv20) <= 0.05)] = "均线蓄势"
    s[rest & ~(np.abs(dv20) <= 0.05)] = "弱势结构"
    del dev
    return s


def tier(rec, r120, mf, rps):
    """X01 v1.0 分档,逐字按他的字典 + 反解的观察级门槛。"""
    base = (rec >= 0.40) & (r120 >= 0.10) & (mf >= 0.55)
    t = np.full(rec.shape, "无信号", object)
    t[base & (rps < 80)] = "观察级"
    t[base & (rps >= 80) & (rps < 90)] = "标准确认"
    t[base & (rps >= 90)] = "强确认"
    return t


def qtier(q):
    r = np.full(q.shape, "缺失", object)
    r[np.isfinite(q) & (q < 0.30)] = "低"
    r[np.isfinite(q) & (q >= 0.30) & (q < 0.70)] = "中"
    r[np.isfinite(q) & (q >= 0.70)] = "高"
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
    sus = al("is_suspended", True).astype(bool)
    vol = al("volume", 0)
    okm = (~al("is_st", True).astype(bool) & ~sus & (al("listed_days", 0) >= 250)
           & (vol > 0) & np.isfinite(cl))
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
    trad = ~sus & (vol > 0)
    rps60 = pd.DataFrame(np.where(trad & np.isfinite(r60), r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100.0
    rps250 = pd.DataFrame(np.where(trad & np.isfinite(r250), r250, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100.0
    del lo250, hi250, ma20, r60, r250
    wc, w20, w60 = weekly(cl, idx)
    wst = wstruct(wc, w20, w60)
    del wc, w20, w60
    p = np.load(PSTATE, allow_pickle=True)
    pc = {c: j for j, c in enumerate(list(p["codes"]))}
    pmap = np.array([pc.get(c, -1) for c in codes])
    dep = np.full((nt, ns), np.nan, np.float32)
    shr = np.full((nt, ns), np.nan, np.float32)
    cnv = np.full((nt, ns), np.nan, np.float32)
    hit3 = np.zeros((nt, ns), bool)
    brk = np.zeros((nt, ns), bool)
    g = pmap >= 0
    for dst, src in ((dep, "dep"), (shr, "shr"), (cnv, "cnv")):
        dst[:, g] = p[src][:, pmap[g]]
    hit3[:, g] = p["hit3"][:, pmap[g]]
    brk[:, g] = p["brk"][:, pmap[g]]
    tt = tier(rec, r120, mfr, rps60)
    uni = np.isin(tt, ("标准确认", "强确认")).astype(int)
    psig = np.where(brk, "平台突破（研究）",
                    np.where(hit3, "平台观察", "无平台信号"))
    print(f"价格/平台字段就绪 ({time.time()-t0:.0f}s)", flush=True)

    # ---- R09 核心质量分(严格复用冻结口径)----
    z = np.load(CACHE, allow_pickle=True)
    zc = list(z["codes"])
    assert (pd.DatetimeIndex(z["idx"]) == idx).all(), "R09 缓存日期不一致"
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
    print(f"R09 因子面板就绪 ({time.time()-t0:.0f}s)", flush=True)

    qcache = {}

    def qscore(t):
        if t in qcache:
            return qcache[t]
        e = np.flatnonzero(zok[t] & np.isfinite(logcap[t]) & np.isfinite(tmean[t]))
        v = np.full(len(zc), np.nan)
        if len(e):
            v[e] = route_scores("R09", t, e, fm, zcl, raw, logcap, tmean, "raw")
        out = np.full(ns, np.nan)
        gm = zmap >= 0
        out[gm] = v[zmap[gm]]
        qcache[t] = out
        return out
    pos = {c: j for j, c in enumerate(codes)}
    ip = pd.Index(idx)

    def row(kind, t, j):
        q = qscore(t)[j]
        return {"样本类型": kind, "观察日期": idx[t].date(), "股票代码": codes[j],
                "股票名称": "", "收盘价": cl[t, j], "统一信号": int(uni[t, j]),
                "信号类型": tt[t, j], "平台信号": psig[t, j],
                "质量等级": qtier(np.array([q]))[0],
                "距一年低点涨幅": rec[t, j], "近120日收益": r120[t, j],
                "MA20持续度": mfr[t, j], "RPS60": rps60[t, j],
                "RPS250": rps250[t, j], "距一年高点价格差": gap[t, j],
                "周线结构": wst[t, j], "平台深度": dep[t, j],
                "平台缩量比": shr[t, j], "平台收敛比": cnv[t, j],
                "R09核心质量分": q,
                "信号理由": ("；".join(
                    [x for x in ("反弹≥40%" if rec[t, j] >= .40 else "",
                                 "120日收益≥10%" if r120[t, j] >= .10 else "",
                                 "MA20持续度≥55%" if mfr[t, j] >= .55 else "",
                                 f"RPS60={rps60[t, j]:.1f}") if x]))}

    # ---- 锚点 B/C/D:用他自己的 276 行案例验证重建 ----
    ca = pd.read_excel(XL, sheet_name="案例摘要", header=3).dropna(how="all")
    ca["股票代码"] = ca["股票代码"].astype(str).str.split(".").str[0].str.zfill(6)
    ca["观察日期"] = pd.to_datetime(ca["观察日期"])
    chk = []
    for _, r in ca.iterrows():
        j, t = pos.get(r["股票代码"]), int(ip.searchsorted(r["观察日期"]))
        if j is None or t >= nt or idx[t] != r["观察日期"]:
            continue
        q = qscore(t)[j]
        chk.append({"他_信号类型": r["信号类型"], "我_信号类型": tt[t, j],
                    "他_平台信号": r["平台信号"], "我_平台信号": psig[t, j],
                    "他_质量等级": r["质量等级"], "我_质量等级": qtier(np.array([q]))[0],
                    "他_质量分": r["R09核心质量分"], "我_质量分": q,
                    "他_周线": r["周线结构"], "我_周线": wst[t, j],
                    "代码": r["股票代码"], "日期": r["观察日期"].date()})
    ck = pd.DataFrame(chk)
    w = 92
    print(f"\n{'='*w}\n锚点:用他自己的 {len(ck)} 行案例验证我对他口径的重建\n{'='*w}")
    rates = {}
    for k, thr in (("信号类型", 0.95), ("平台信号", 0.95), ("质量等级", 0.90),
                   ("周线", 0.90)):
        m = (ck[f"他_{k}"].astype(str) == ck[f"我_{k}"].astype(str))
        rates[k] = float(m.mean())
        print(f"  {k:<8} 一致 {int(m.sum()):>4}/{len(ck)} = {m.mean():>6.1%}  "
              f"(门槛 {thr:.0%}) {'✓' if m.mean() >= thr else '✗'}")
        if m.mean() < thr:
            print("    不一致样例:")
            print("    " + ck[~m][["代码", "日期", f"他_{k}", f"我_{k}"]].head(8)
                  .to_string(index=False).replace("\n", "\n    "))
    a = ck["他_质量分"].to_numpy(float)
    b = ck["我_质量分"].to_numpy(float)
    gq = np.isfinite(a) & np.isfinite(b)
    print(f"  R09质量分 可比 {int(gq.sum())};中位|差| "
          f"{np.median(np.abs(a[gq]-b[gq])):.4f};相关 "
          f"{pd.Series(a[gq]).corr(pd.Series(b[gq])):.4f}")
    ck.to_csv(f"{OUT}/codex_template_anchor.csv", index=False, encoding="utf-8-sig")
    if rates["信号类型"] < 0.95 or rates["平台信号"] < 0.95:
        print("\n**锚点 B/C 不过 → 重建不可靠,不出 2022 清单**")
        return

    # ---- 2022 年清单 ----
    yr = idx.year
    me = np.sort(pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy().astype(int))
    me22 = [int(t) for t in me if idx[t].year == 2022]
    rows = []
    for t in me22:
        e = np.flatnonzero(okm[t] & (np.isin(tt[t], ("观察级", "标准确认", "强确认"))
                                     | (psig[t] != "无平台信号")))
        for j in e:
            rows.append(row("月末观察", t, int(j)))
    for t in np.flatnonzero(yr == 2022):
        for j in np.flatnonzero(brk[t]):
            rows.append(row("平台突破日", int(t), int(j)))
    out = pd.DataFrame(rows)[COLS].sort_values(["观察日期", "信号类型", "RPS60"],
                                               ascending=[True, True, False])
    try:
        import json
        with open(f"{OUT}/code_name_map_wide.json", encoding="utf-8") as fh:
            nm = json.load(fh)
        out["股票名称"] = out["股票代码"].map(lambda c: nm.get(c, ""))
    except OSError:
        pass
    out.to_csv(f"{OUT}/codex_template_2022.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'='*w}\n2022 年带信号清单(完全按 Codex 模板字段)\n{'='*w}")
    print(f"  总行数 {len(out):,};涉及 {out['股票代码'].nunique():,} 只")
    print("\n  按信号类型:")
    print("    " + out["信号类型"].value_counts().to_string().replace("\n", "\n    "))
    print("\n  按平台信号:")
    print("    " + out["平台信号"].value_counts().to_string().replace("\n", "\n    "))
    print("\n  按质量等级:")
    print("    " + out["质量等级"].value_counts().to_string().replace("\n", "\n    "))
    print("\n  按月(统一信号=1 的只数):")
    s1 = out[(out["样本类型"] == "月末观察") & (out["统一信号"] == 1)]
    print("    " + s1.groupby("观察日期").size().to_string().replace("\n", "\n    "))
    print(f"\n落库 {OUT}/codex_template_2022.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
