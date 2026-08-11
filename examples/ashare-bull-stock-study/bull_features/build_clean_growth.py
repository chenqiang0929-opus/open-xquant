"""构造正确的成长字段 —— 修复报告期错位污染

═══ 缺陷 ═══
面板的 `net_income` / `revenue` 是**本年累计(YTD)**,每年年初归零;
而 `ni_yoy_252` = `net_income / net_income.shift(252) - 1`(实测相关 1.0000)。
**252 个交易日 ≈ 1.04 年,会跨过报告期。**

茅台 2023-05-04 的 ni_yoy_252 = **-60.4%** —— 实际是拿 2023Q1 单季(208亿)
去比 2021 全年(524亿)。这个数没有意义。

**污染已量化**(横截面「净利同比>0」比例,按月份平均):
  4月 17.2% / 8月 42.3% / 10月 42.2%,**极差 25.1pp**
真实盈利增长不该随日历月份系统性摆动。

═══ 修法 ═══
1. **去累计**:同一财年内 单季 = YTD_t − YTD_{t-1};Q1 即 YTD 本身
   财年边界识别:YTD 相对上期**大幅下降**(新财年重新累计)
2. **C(欧奈尔口径)** = 本季单季 ÷ **去年同季** − 1
3. **TTM 同比** = 最近4季之和 ÷ 前4季之和 − 1(更稳健,作主口径)
4. 收入同样处理
5. 全部按**财报快照变化日**前推(ffill),不引入未来财报

═══ 硬自检(不过就不往下跑) ═══
新字段的「>0比例」按月份极差须 ≤ **5pp**(原 25.1pp),否则去累计没做对。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
MAX_MONTH_SPREAD = 0.05

t0 = time.time()


def decumulate(s: pd.Series) -> pd.DataFrame:
    """把 YTD 累计序列还原成 单季 / TTM,并按报告期对齐算同比。

    输入:逐日 ffill 的 YTD 值(每季报公布时跳变)
    输出:逐日 DataFrame,列 = q(单季)、ttm、q_yoy(当季同比)、ttm_yoy
    """
    v = s.dropna()
    if v.empty:
        return pd.DataFrame(index=s.index, columns=["q", "ttm", "q_yoy", "ttm_yoy"],
                            dtype=float)
    # 取每次跳变的首日 = 一个报告期快照
    chg = v.ne(v.shift())
    snap = v[chg]
    if len(snap) < 2:
        return pd.DataFrame(index=s.index, columns=["q", "ttm", "q_yoy", "ttm_yoy"],
                            dtype=float)
    vals = snap.to_numpy(float)
    dates = snap.index

    # 财年边界:**只用 YTD 相对上一期下降**判断,新财年会重新累计。
    #
    # **踩过的坑**:首版还加了「公布日年份变化」作为佐证 —— 但**年报在次年
    # 3-4 月才公布**,于是年报被误判成「新财年第一期」,单季 = 整年。
    # 茅台 2023-03-31 因此算出「单季 627 亿、同比 +263.7%」(实为 2022 全年)。
    # 自检抓到:C 当季同比的月份极差 10.2% > 5pp 门槛。
    prev = np.r_[np.nan, vals[:-1]]
    yr = dates.year.to_numpy()
    new_fy = np.abs(vals) < np.abs(prev) * 0.6
    new_fy[0] = True

    q = np.where(new_fy, vals, vals - prev)          # 单季
    # 报告期序号(1..4):同一财年内递增
    stage = np.zeros(len(vals), int)
    k = 0
    for i in range(len(vals)):
        k = 1 if new_fy[i] else min(k + 1, 4)
        stage[i] = k

    qs = pd.Series(q, index=dates)
    ttm = qs.rolling(4, min_periods=4).sum()

    # 当季同比:找去年同一 stage 的那一期
    q_yoy = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        tgt_y, tgt_s = yr[i] - 1, stage[i]
        m = np.flatnonzero((yr == tgt_y) & (stage == tgt_s))
        if m.size and np.isfinite(q[i]) and np.isfinite(q[m[-1]]) and abs(q[m[-1]]) > 0:
            q_yoy[i] = q[i] / abs(q[m[-1]]) - 1 if q[m[-1]] > 0 else np.nan
    ttm_yoy = (ttm / ttm.shift(4).abs() - 1).where(ttm.shift(4) > 0)

    out = pd.DataFrame({"q": qs, "ttm": ttm,
                        "q_yoy": pd.Series(q_yoy, index=dates),
                        "ttm_yoy": ttm_yoy})
    return out.reindex(s.index).ffill()


files = sorted(glob.glob(f"{DATA}/*.parquet"))
ni_q, ni_ttm, rev_ttm = {}, {}, {}
n_ok = 0
for f in files:
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    try:
        x = pd.read_parquet(f, columns=["net_income", "revenue"])
    except Exception:
        continue
    if x.empty:
        continue
    x.index = x.index.tz_localize(None)
    a = decumulate(pd.to_numeric(x["net_income"], errors="coerce"))
    b = decumulate(pd.to_numeric(x["revenue"], errors="coerce"))
    ni_q[k] = a["q_yoy"]          # C:当季净利同比
    ni_ttm[k] = a["ttm_yoy"]      # TTM 净利同比
    rev_ttm[k] = b["ttm_yoy"]     # TTM 收入同比
    n_ok += 1
    if n_ok % 1000 == 0:
        print(f"  已处理 {n_ok:,} 只  ({time.time()-t0:.0f}s)")

C_QYOY = pd.DataFrame(ni_q).sort_index()
NI_TTM = pd.DataFrame(ni_ttm).reindex_like(C_QYOY)
REV_TTM = pd.DataFrame(rev_ttm).reindex_like(C_QYOY)
print(f"完成 {n_ok:,} 只,面板 {C_QYOY.shape}  ({time.time()-t0:.0f}s)")

# ---------------- 单股手工核对 ----------------
print(f"\n{'='*100}\n验证1 单股核对:600519 去累计后的单季净利(亿元)\n{'='*100}")
x = pd.read_parquet(f"{DATA}/600519.parquet", columns=["net_income"])
x.index = x.index.tz_localize(None)
dec = decumulate(pd.to_numeric(x["net_income"], errors="coerce"))
s = dec.loc["2023-01-01":"2024-12-31"]
sn = s[s["q"].ne(s["q"].shift())].dropna(subset=["q"])
for d, r in sn.iterrows():
    print(f"  {d.date()}  单季 {r['q']/1e8:>8.1f}亿   TTM {r['ttm']/1e8:>9.1f}亿   "
          f"当季同比 {r['q_yoy']:>+7.1%}" if np.isfinite(r["q_yoy"]) else
          f"  {d.date()}  单季 {r['q']/1e8:>8.1f}亿   TTM "
          f"{r['ttm']/1e8 if np.isfinite(r['ttm']) else float('nan'):>9.1f}亿   当季同比      —")
print("  (茅台单季净利量级应在 130~250 亿之间;若出现 500+ 亿说明没去累计)")

# ---------------- 硬自检:月份极差 ----------------
print(f"\n{'='*100}\n验证2 硬自检:按月份的横截面「>0比例」极差须 ≤ 5pp(原 25.1pp)\n{'='*100}")
me = [d for d in C_QYOY.resample("ME").last().index if d >= pd.Timestamp("2015-01-01")]
rows = []
for m in me:
    p = C_QYOY.index.searchsorted(m, side="right") - 1
    if p < 260:
        continue
    rows.append({"月份": m.month,
                 "C当季>0": (C_QYOY.iloc[p] > 0).mean(),
                 "TTM净利>0": (NI_TTM.iloc[p] > 0).mean(),
                 "TTM双增长>0": ((NI_TTM.iloc[p] > 0) & (REV_TTM.iloc[p] > 0)).mean()})
G = pd.DataFrame(rows).groupby("月份").mean()
print(G.to_string(float_format=lambda v: f"{v:6.1%}"))
spreads = {c: G[c].max() - G[c].min() for c in G.columns}
print()
ok = True
for c, sp_ in spreads.items():
    flag = "✓" if sp_ <= MAX_MONTH_SPREAD else "✗"
    ok &= sp_ <= MAX_MONTH_SPREAD
    print(f"  {c:<14} 月份极差 **{sp_:.1%}**  {flag}")
print(f"\n  对照:原 ni_yoy_252 的极差 25.1%")
if not ok:
    print("\n  **自检未通过 —— 去累计有问题,不往下跑。**")
else:
    print("\n  **自检通过。**")

# ---------------- 覆盖率 ----------------
print(f"\n{'='*100}\n覆盖率(非空比例)\n{'='*100}")
for n, f_ in (("C 当季同比", C_QYOY), ("TTM 净利同比", NI_TTM), ("TTM 收入同比", REV_TTM)):
    print(f"  {n:<14} {f_.notna().mean().mean():>6.1%}")
old = pd.read_parquet(f"{DATA}/600519.parquet", columns=["ni_yoy_252"])
print(f"  {'(原 ni_yoy_252)':<14} 57.2%")

C_QYOY.to_parquet(f"{SP}/clean_growth_c_qyoy.parquet")
NI_TTM.to_parquet(f"{SP}/clean_growth_ni_ttm_yoy.parquet")
REV_TTM.to_parquet(f"{SP}/clean_growth_rev_ttm_yoy.parquet")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: clean_growth_{{c_qyoy,ni_ttm_yoy,rev_ttm_yoy}}.parquet")
