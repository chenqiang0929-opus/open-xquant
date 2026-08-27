"""§147:换成月度滚动框架找「启动前」特征 —— 并补上此前没测的成交加速维度。

用户的问题
----------
「我给的股票,你难道没办法在启动之前识别出来吗」

**我之前一直用年度框架**(年初观察、当年翻倍),这可能就是问题所在:
启动可以发生在年中任何时候,年初那个截面根本看不到。
本节换框架:

- 观察点:**池内每月最后一个交易日**(2023-06 起,保证 250 日历史)
- **启动的定义**:未来 **60 个交易日**(约 3 个月)涨幅 ≥ **50%**
- 样本量从 2 个年度截面变成约 30 个月度截面

补上之前没测的一维:**成交/换手加速**
------------------------------------
前面所有节测的都是「价格已经涨了多少、涨了多久」,
**没有测过「量能是不是刚起来」**。本节新增:
  `turn_acc` = 20日均换手 ÷ 60日均换手 − 1
  `amt_acc`  = 20日均成交额 ÷ 60日均成交额 − 1
  `vol_ratio` = 20日波动率 ÷ 60日波动率 − 1

全部变量(11 个,观察日可算、无前视)
  RPS60、RPS250(全市场分位)、距一年低点、120日收益、MA20持续度、
  20日波动率、**换手加速**、**成交额加速**、**波动加速**、流通市值、20日均换手

评价
----
按池内分位切 低30/中40/高30,报各组**启动率**与 lift(相对池内启动基准)。
纪律 A:池内打乱标签 200 次,best-of-N 的 95 分位作噪音上界。
**同一只股票 60 日内不重复计事件**,避免同一段行情被重复计数。

**本节不设通过/不通过判据。** 不新增顶层目录;不 force push;不作可交易性声明。
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
from codex_r10_replication import DATA  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
XLS = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
       "f48a5b4d-___20260827.xls")
HOR, THR, GAP, N_PERM = 60, 0.50, 60, 200
L19 = ["001309", "688322", "688041", "301338", "301171", "688428", "688392",
       "603163", "688372", "688525", "603061", "301345", "688361", "603119",
       "688347", "301498", "603193", "301413", "001280"]


def main():  # noqa: PLR0915
    t0 = time.time()
    px = pd.read_excel(XLS, dtype=str)
    px = px.rename(columns={px.columns[1]: "名称"})
    px["代码"] = px["代码"].str.zfill(6)
    pool = dict(zip(px.代码, px.名称, strict=True))

    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "float_mv", "turnover", "amount", "volume", "is_st",
            "is_suspended", "listed_days"]
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
    assert (nt, ns) == (3297, 5232), f"锚点 {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f)
    mv = al("float_mv").to_numpy() / 1e8
    trn = al("turnover")
    amt = al("amount")
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)
    ok &= np.isfinite(cl)
    dfc = pd.DataFrame(cl)
    lo250 = dfc.rolling(250, min_periods=250).min().to_numpy()
    ma20 = dfc.rolling(20, min_periods=20).mean().to_numpy()
    t20 = trn.rolling(20, min_periods=10).mean().to_numpy()
    t60 = trn.rolling(60, min_periods=30).mean().to_numpy()
    a20 = amt.rolling(20, min_periods=10).mean().to_numpy()
    a60 = amt.rolling(60, min_periods=30).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        r120 = cl / np.roll(cl, 120, axis=0) - 1.0
        r120[:120] = np.nan
        r60 = cl / np.roll(cl, 60, axis=0) - 1.0
        r60[:60] = np.nan
        r250 = cl / np.roll(cl, 250, axis=0) - 1.0
        r250[:250] = np.nan
        lr = np.log(cl / np.roll(cl, 1, axis=0))
        lr[0] = np.nan
        turn_acc = t20 / np.where(t60 > 0, t60, np.nan) - 1.0
        amt_acc = a20 / np.where(a60 > 0, a60, np.nan) - 1.0
    ab = pd.DataFrame(cl > ma20).rolling(120, min_periods=120).mean().to_numpy()
    v20 = pd.DataFrame(lr).rolling(20, min_periods=20).std().to_numpy()
    v60 = pd.DataFrame(lr).rolling(60, min_periods=40).std().to_numpy()
    with np.errstate(all="ignore"):
        vol_acc = v20 / np.where(v60 > 0, v60, np.nan) - 1.0
    rps60 = pd.DataFrame(np.where(ok, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    rps250 = pd.DataFrame(np.where(ok, r250, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    fmax = pd.DataFrame(cl[::-1]).rolling(HOR, min_periods=1).max().to_numpy()[::-1]
    fwd = np.full_like(cl, np.nan)
    fwd[:-1] = fmax[1:]
    with np.errstate(all="ignore"):
        up = fwd / np.where(cl > 0, cl, np.nan) - 1.0
    print(f"面板就绪 ({time.time()-t0:.0f}s)", flush=True)

    varmap = {"RPS60": rps60, "RPS250": rps250, "距一年低点": rec,
              "120日收益": r120, "MA20持续度": ab, "20日波动率": v20,
              "**换手加速20/60**": turn_acc, "**成交额加速20/60**": amt_acc,
              "**波动加速20/60**": vol_acc, "流通市值(小→大)": mv,
              "20日均换手": t20}
    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy()
    cp = {c: j for j, c in enumerate(cldf.columns)}
    rows = []
    for t in me:
        t = int(t)
        if t < 260 or t > nt - HOR - 1 or idx[t] < pd.Timestamp("2023-06-01"):
            continue
        for c in pool:
            j = cp.get(c)
            if j is None or not ok[t, j] or not np.isfinite(up[t, j]):
                continue
            if not all(np.isfinite(m[t, j]) for m in varmap.values()):
                continue
            rows.append((int(t), int(j), c, bool(up[t, j] >= THR)))
    ev = pd.DataFrame(rows, columns=["t", "j", "code", "y"])
    ev = ev.sort_values(["code", "t"])
    keep, last = [], {}
    for r in ev.itertuples():
        if r.t - last.get(r.code, -10**9) >= GAP:
            keep.append(True)
            last[r.code] = r.t
        else:
            keep.append(False)
    ev = ev[keep]
    base = ev.y.mean()
    print(f"\n月度事件 {len(ev):,}(池内 {ev.code.nunique()} 只,"
          f"{idx[ev.t.min()].date()} → {idx[ev.t.max()].date()})")
    print(f"**启动基准率 = P(未来 {HOR} 日涨 ≥{THR:.0%}) = {base:.2%}**", flush=True)

    tv, jv, y = ev.t.to_numpy(), ev.j.to_numpy(), ev.y.to_numpy()
    print(f"\n{'='*92}\n各变量按池内分位分三组的启动率(基准 {base:.2%})\n{'='*92}")
    print(f"{'变量':<22}{'低30%':>21}{'中40%':>21}{'高30%':>21}")
    masks, out = {}, []
    for nm, mat in varmap.items():
        v = mat[tv, jv]
        q = pd.Series(v).rank(pct=True).to_numpy()
        line = f"{nm:<22}"
        for lo, hi, tag in ((0, .30, "低"), (.30, .70, "中"), (.70, 1.01, "高")):
            m = (q >= lo) & (q < hi)
            hr = y[m].mean()
            line += f"{int(m.sum()):>6d}只 {hr:>6.2%} lift{hr/base:>5.2f}"
            masks[(nm, tag)] = m
            out.append({"变量": nm, "组": tag, "n": int(m.sum()),
                        "启动率": float(hr), "lift": float(hr / base)})
        print(line)
    rg = np.random.default_rng(20260827)
    best = [max(rg.permutation(y)[m].mean() / base for m in masks.values())
            for _ in range(N_PERM)]
    hi95 = float(np.percentile(best, 95))
    df = pd.DataFrame(out)
    df["超上界"] = df.lift > hi95
    print(f"\n纪律A 噪音上界:打乱标签 {N_PERM} 次,best-of-{len(masks)} 的 lift "
          f"中位 {np.median(best):.2f}  **95分位 {hi95:.2f}**")
    win = df[df.超上界].sort_values("lift", ascending=False)
    print(f"\n**超出噪音上界的 {len(win)} 组**:")
    for _, r in win.iterrows():
        print(f"  {r['变量']} {r['组']}组  启动率 {r['启动率']:.2%}  "
              f"lift **{r['lift']:.2f}**  n={r['n']:,}")
    if not len(win):
        print("  **无**")

    # ---- 组合(只用超上界的变量,组合方式跑前写死:两两/三重取高30%)----
    def q30(mat):
        v = mat[tv, jv]
        return pd.Series(v).rank(pct=True).to_numpy() >= 0.70
    hi_rec, hi_v20 = q30(rec), q30(v20)
    hi_ta, hi_aa = q30(turn_acc), q30(amt_acc)
    hi_r250, hi_mv = q30(rps250), q30(mv)
    combos = {
        "距低点高 & 波动率高": hi_rec & hi_v20,
        "距低点高 & 换手加速高": hi_rec & hi_ta,
        "距低点高 & 成交额加速高": hi_rec & hi_aa,
        "RPS250高 & 换手加速高": hi_r250 & hi_ta,
        "距低点高 & 波动率高 & 换手加速高": hi_rec & hi_v20 & hi_ta,
        "距低点高 & 市值大 & 波动率高": hi_rec & hi_mv & hi_v20,
    }
    print(f"\n{'='*92}\n组合(各条均取池内高 30%);基准 {base:.2%},噪音上界 {hi95:.2f}\n{'='*92}")
    print(f"{'组合':<34}{'n':>7}{'覆盖率':>8}{'启动率':>9}{'lift':>7}{'召回率':>8}{'超上界':>7}")
    nb = int(y.sum())
    for nm, m in combos.items():
        if m.sum() < 50:
            print(f"{nm:<34}{int(m.sum()):>7}  样本<50")
            continue
        hr = y[m].mean()
        print(f"{nm:<34}{int(m.sum()):>7,}{m.mean():>8.1%}{hr:>9.2%}"
              f"{hr/base:>7.2f}{y[m].sum()/nb:>8.1%}"
              f"{'✓' if hr/base > hi95 else '':>7}")
        out.append({"变量": nm, "组": "组合", "n": int(m.sum()),
                    "启动率": float(hr), "lift": float(hr / base)})

    print(f"\n{'='*92}\n这 19 只在实际启动前一个月的特征(2023-06 起所有启动事件)\n{'='*92}")
    e19 = ev[ev.code.isin(L19) & ev.y]
    print(f"  19 只共 {len(e19)} 次启动事件(未来{HOR}日涨≥{THR:.0%})")
    if len(e19):
        print(f"{'名称':<10}{'观察日':<12}{'RPS60':>7}{'RPS250':>8}{'距低点':>8}"
              f"{'换手加速':>9}{'成交额加速':>11}{'后续涨幅':>9}")
        for r in e19.itertuples():
            print(f"{pool[r.code]:<10}{str(idx[r.t].date()):<12}"
                  f"{rps60[r.t,r.j]:>7.0f}{rps250[r.t,r.j]:>8.0f}"
                  f"{rec[r.t,r.j]:>8.0%}{turn_acc[r.t,r.j]:>9.0%}"
                  f"{amt_acc[r.t,r.j]:>11.0%}{up[r.t,r.j]:>9.0%}")
    # ---- 当前时点(面板末日)名单 ----
    tlast = nt - 1
    el = [c for c in pool if c in cp and ok[tlast, cp[c]]
          and all(np.isfinite(m[tlast, cp[c]]) for m in
                  (rec, v20, turn_acc, amt_acc, mv, rps250, rps60))]
    jn = np.array([cp[c] for c in el])

    def hi30(mat):
        v = mat[tlast, jn]
        return pd.Series(v).rank(pct=True).to_numpy() >= 0.70
    h_rec, h_v, h_t, h_m = hi30(rec), hi30(v20), hi30(turn_acc), hi30(mv)
    print(f"\n{'='*92}\n当前名单({idx[tlast].date()},池内合格 {len(el)});"
          f"结果未知,不构成买入建议\n{'='*92}")
    for nm, m in (("距低点高 & 市值大 & 波动率高(历史 lift 1.94)", h_rec & h_m & h_v),
                  ("距低点高 & 波动率高 & 换手加速高(历史 lift 1.85)",
                   h_rec & h_v & h_t),
                  ("距低点高 & 换手加速高(历史 lift 1.72,覆盖 11.3%)", h_rec & h_t)):
        sel = [el[i] for i in np.flatnonzero(m)]
        print(f"\n**{nm}** → {len(sel)} 只")
        for c in sorted(sel, key=lambda z: -rec[tlast, cp[z]]):
            j = cp[c]
            print(f"    {c} {pool[c]:<10} 距低点{rec[tlast,j]:>7.0%} "
                  f"波动{v20[tlast,j]:>6.2%} 换手加速{turn_acc[tlast,j]:>+6.0%} "
                  f"成交额加速{amt_acc[tlast,j]:>+6.0%} RPS60 {rps60[tlast,j]:>3.0f} "
                  f"RPS250 {rps250[tlast,j]:>3.0f}")
        out += [{"变量": "当前名单", "组": nm, "代码": c, "名称": pool[c]}
                for c in sel]
    df = pd.DataFrame(out)
    df.to_csv(f"{OUT}/pool_startup_monthly.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/pool_startup_monthly.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
