"""§161 交叉核对:Codex 监控模板 v0.4 的 4 只案例 276 行,与本地面板逐字段比对。

**判据跑之前写死,跑完照判,不放宽。**

对象
----
`X01价格因子＋R09质量因子＋平台买点_v0.4.xlsx` 的「案例摘要」页:
4 只股票、276 行(月末观察 + 平台首次突破事件日),2021-01-01 → 2026-08-20。

**为什么值得核**:平台深度/缩量比/收敛比三个字段是他从本项目的
`platform_screener_report_20260828.md` 与 `platform_buypoints.csv` 拿去的。
**如果这三个对不上,一定有一边算错了。**

已知的口径差(不算违例,必须先扣掉)
----------------------------------
- 本地面板末日 **2026-08-03**,他的行情截止 **2026-08-20**,差 12 个交易日;
- 前复权锚点不同 → **绝对价格会差一个每只股票固定的倍数**,但**比值型字段不受影响**。

判据(写死)
----------
C1 锚点:面板 (3297, 5232);4 只案例全部在面板内;可比行数 ≥ 250。
C2 **比值型字段**(距一年低点涨幅、近120日收益、MA20持续度、距一年高点价格差、
   平台深度、平台缩量比、平台收敛比):**绝对差 < 0.005 的行占比 ≥ 95% 判一致。**
C3 **RPS60 / RPS250**(横截面百分位,依赖股票池口径):
   **绝对差 < 3.0 个百分点的行占比 ≥ 90% 判一致。**
C4 **收盘价**:前复权锚点不同只应带来每只股票**一个固定倍数** →
   **同一只股票内 (本地价/他的价) 的变异系数 < 0.5% 判一致**;
   若某只的变异系数超标,说明**复权处理有实质差异**,须定位到具体日期。

**任何一项不达标,如实报出并定位到行,不调容差。**

附带产出(不是判据)
------------------
他在「整合说明」里写明 600 池当前缺三项:**相对MA250位置、距一年高点天数、
行业内接近一年高点分位**。本地面板算得出,一并补给这 4 只案例。

不做的
------
不改他的数字、不改本地口径去迁就;不因为对不上就调容差;不作可交易性声明。
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
from codex_r10_replication import DATA  # noqa: E402
from industry_neutral import build_industry  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
XL = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
      "0abc3d92-X01_____R09_________________v0.4____.xlsx")
PSTATE = f"{OUT}/platform_state.npz"
RATIO = {"距一年低点涨幅": "rec250", "近120日收益": "r120", "MA20持续度": "mfrac",
         "距一年高点价格差": "gap_hi", "平台深度": "dep", "平台缩量比": "shr",
         "平台收敛比": "cnv"}
TOL_R, TOL_P, TOL_CV = 0.005, 3.0, 0.005


def main():  # noqa: PLR0915
    t0 = time.time()
    ca = pd.read_excel(XL, sheet_name="案例摘要", header=3).dropna(how="all")
    ca["股票代码"] = ca["股票代码"].astype(str).str.split(".").str[0].str.zfill(6)
    ca["观察日期"] = pd.to_datetime(ca["观察日期"])
    print(f"案例 {len(ca)} 行;股票 {sorted(ca['股票代码'].unique())};"
          f"{ca['观察日期'].min().date()} → {ca['观察日期'].max().date()}")
    print("样本类型:" + ", ".join(f"{k} {v}"
                               for k, v in ca["样本类型"].value_counts().items()))

    import glob
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
    assert (nt, ns) == (3297, 5232), f"C1 面板 {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f).to_numpy()
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    sus = al("is_suspended", True).astype(bool)
    vol = al("volume", 0)
    px = pd.DataFrame(cl)
    lo250 = px.rolling(250, min_periods=250).min().to_numpy()
    hi250 = px.rolling(250, min_periods=250).max().to_numpy()
    ma20 = px.rolling(20, min_periods=20).mean().to_numpy()
    ma250 = px.rolling(250, min_periods=250).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec250 = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        gap_hi = cl / np.where(hi250 > 0, hi250, np.nan) - 1.0
        rel250 = cl / np.where(ma250 > 0, ma250, np.nan) - 1.0
        mfrac = pd.DataFrame((cl > ma20).astype(np.float64)).where(
            np.isfinite(ma20)).rolling(120, min_periods=120).mean().to_numpy()
        r120 = px.pct_change(120).to_numpy()
        r60 = px.pct_change(60).to_numpy()
        r250 = px.pct_change(250).to_numpy()
    trad = ~sus & (vol > 0)
    rps60 = pd.DataFrame(np.where(trad & np.isfinite(r60), r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100.0
    rps250 = pd.DataFrame(np.where(trad & np.isfinite(r250), r250, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100.0
    # 距一年高点天数:当前收盘距 250 日内最高收盘出现日的间隔
    dh = np.full((nt, ns), np.nan)
    for j in range(ns):
        s = pd.Series(cl[:, j])
        am = s.rolling(250, min_periods=250).apply(lambda v: len(v) - 1 - v.argmax(),
                                                   raw=True)
        dh[:, j] = am.to_numpy()
    ind, ind_names, _ = build_industry(list(cldf.columns), idx)
    p = np.load(PSTATE, allow_pickle=True)
    pc = {c: j for j, c in enumerate(list(p["codes"]))}
    dep, shr, cnv = p["dep"], p["shr"], p["cnv"]
    hit3, brk = p["hit3"], p["brk"]
    print(f"面板与平台缓存就绪 ({time.time()-t0:.0f}s)", flush=True)

    pos = {c: j for j, c in enumerate(cldf.columns)}
    ip = pd.Index(idx)
    rows = []
    for _, r in ca.iterrows():
        c, dt = r["股票代码"], r["观察日期"]
        j = pos.get(c)
        t = int(ip.searchsorted(dt))
        if j is None or t >= nt or idx[t] != dt:
            continue
        jp = pc.get(c, -1)
        rows.append({
            "代码": c, "名称": r["股票名称"], "日期": dt, "样本类型": r["样本类型"],
            "他_收盘": r["收盘价"], "我_收盘": cl[t, j],
            "他_rec250": r["距一年低点涨幅"], "我_rec250": rec250[t, j],
            "他_r120": r["近120日收益"], "我_r120": r120[t, j],
            "他_mfrac": r["MA20持续度"], "我_mfrac": mfrac[t, j],
            "他_gap_hi": r["距一年高点价格差"], "我_gap_hi": gap_hi[t, j],
            "他_rps60": r["RPS60"], "我_rps60": rps60[t, j],
            "他_rps250": r["RPS250"], "我_rps250": rps250[t, j],
            "他_dep": r["平台深度"], "我_dep": dep[t, jp] if jp >= 0 else np.nan,
            "他_shr": r["平台缩量比"], "我_shr": shr[t, jp] if jp >= 0 else np.nan,
            "他_cnv": r["平台收敛比"], "我_cnv": cnv[t, jp] if jp >= 0 else np.nan,
            "他_平台信号": r["平台信号"],
            "我_三条全中": bool(hit3[t, jp]) if jp >= 0 else None,
            "我_突破买点": bool(brk[t, jp]) if jp >= 0 else None,
            "补_相对MA250": rel250[t, j], "补_距一年高点天数": dh[t, j],
            "补_行业": None if ind[t, j] < 0 else ind_names[ind[t, j]]})
    m = pd.DataFrame(rows)
    print(f"\n可比行 {len(m)}/{len(ca)}  C1 {'✓' if len(m) >= 250 else '✗'}")
    w = 96
    print(f"\n{'='*w}\nC2 比值型字段(判据:|差| < {TOL_R} 的行占比 ≥ 95%)\n{'='*w}")
    print(f"{'字段':<16}{'可比行':>7}{'一致行':>7}{'占比':>9}{'中位|差|':>11}"
          f"{'最大|差|':>11}  判定")
    res = []
    for zh, k in RATIO.items():
        a, b = m[f"他_{k}"].to_numpy(float), m[f"我_{k}"].to_numpy(float)
        g = np.isfinite(a) & np.isfinite(b)
        if g.sum() == 0:
            print(f"{zh:<16}{0:>7}  —  无可比行")
            continue
        dd = np.abs(a[g] - b[g])
        ok_ = (dd < TOL_R).mean()
        print(f"{zh:<16}{int(g.sum()):>7}{int((dd < TOL_R).sum()):>7}{ok_:>9.1%}"
              f"{np.median(dd):>11.6f}{dd.max():>11.6f}  "
              f"{'✓' if ok_ >= 0.95 else '✗'}")
        res.append({"字段": zh, "可比行": int(g.sum()), "一致占比": float(ok_),
                    "中位差": float(np.median(dd)), "最大差": float(dd.max()),
                    "判定": "一致" if ok_ >= 0.95 else "不一致"})
    print(f"\n{'='*w}\nC3 RPS(判据:|差| < {TOL_P} 个百分点的行占比 ≥ 90%)\n{'='*w}")
    for k, zh in (("rps60", "RPS60"), ("rps250", "RPS250")):
        a, b = m[f"他_{k}"].to_numpy(float), m[f"我_{k}"].to_numpy(float)
        g = np.isfinite(a) & np.isfinite(b)
        dd = np.abs(a[g] - b[g])
        ok_ = (dd < TOL_P).mean()
        print(f"  {zh:<8}可比 {int(g.sum()):>4};|差|<{TOL_P} 占 {ok_:>6.1%};"
              f"中位 {np.median(dd):.3f}、最大 {dd.max():.3f} "
              f"{'✓' if ok_ >= 0.90 else '✗'}")
        res.append({"字段": zh, "可比行": int(g.sum()), "一致占比": float(ok_),
                    "中位差": float(np.median(dd)), "最大差": float(dd.max()),
                    "判定": "一致" if ok_ >= 0.90 else "不一致"})
    print(f"\n{'='*w}\nC4 收盘价:同股票内 (我/他) 比值的变异系数 < {TOL_CV}\n{'='*w}")
    for c, g in m.groupby("代码"):
        rt = (g["我_收盘"] / g["他_收盘"]).to_numpy(float)
        rt = rt[np.isfinite(rt)]
        cv = float(np.std(rt) / np.mean(rt))
        print(f"  {c} {g['名称'].iloc[0]:<8}n={len(rt):>4}  比值中位 {np.median(rt):.6f}"
              f"  变异系数 {cv:.5f} {'✓' if cv < TOL_CV else '✗ 复权有实质差异'}")
        res.append({"字段": f"收盘价·{c}", "可比行": len(rt), "一致占比": np.nan,
                    "中位差": float(np.median(rt)), "最大差": cv,
                    "判定": "一致" if cv < TOL_CV else "不一致"})
    print(f"\n{'='*w}\n平台信号一致性(描述)\n{'='*w}")
    print(pd.crosstab(m["他_平台信号"], m["我_三条全中"], dropna=False).to_string())
    m.to_csv(f"{OUT}/case_crosscheck_codex.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res).to_csv(f"{OUT}/case_crosscheck_summary.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n落库 {OUT}/case_crosscheck_codex.csv(含补算的三个缺失字段)"
          f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
