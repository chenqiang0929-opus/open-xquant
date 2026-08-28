"""§145:在用户的 663 只观察池**内部**扫变量 —— 什么能区分池内的牛股。

用户澄清
--------
「我这个 663 的池子就是一个观察池,就是希望通过这个观察池找出未来牛股或者启动的股票。」

第一四三节已证明:Codex 三档在这个池内没有增量
(2024 年三档全灭;2025 年 lift 1.02 / 0.73 / 0.77)。
**本节换个问法:在这个池子内部,哪些变量能区分当年翻倍与不翻倍?**

口径
----
- 池子:663 只中在本面板的 618 只;时点 = 上一年最后一个交易日;
  目标 = 下一自然年翻倍(标签取自 quant-research-dev 普查)。
- **可验证时点只有两个:2024 与 2025**(2023 时点池内合格数为 0)。
- **一切 lift 都相对池内基准算**,不用全市场基准。
- 每个变量在**池内**按分位切三组:低 30% / 中 40% / 高 30%。

变量(14 个,全部观察日可算、无前视)
------------------------------------
规模流动性:流通市值、20日均换手、Amihud
估值质量:EP_TTM、BP、CFP_TTM、SP_TTM、ROE、净利率
价格:距一年低点、120日收益、MA20持续度、RPS60、20日波动率

纪律
----
**14 个变量 × 2 组端点 = 28 次比较,必然出假阳性。**
照 `bull_feature_scan.py` 的纪律 A:**池内打乱标签 200 次,
每次记录所有变量所有分组里最高的 lift**,得到噪音上界;
**lift 必须超过该分布的 95 分位才算发现。**

**本节是探索性扫描,不设通过/不通过判据** —— 只有两个目标年,
统计效力本来就很弱,**任何结果都必须标注「n=2 年」这个前提**。

不做的
------
不新增顶层目录;不 force push;**不因为某个变量好看就宣称找到规则**
(要变规则须另开一节重新事前登记、加样本外);不作可交易性声明。
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
from codex_routes_rerun import build_fund  # noqa: E402
from startup_threshold_scan import load_labels  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
XLS = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
       "f48a5b4d-___20260827.xls")
N_PERM = 200


def main():  # noqa: PLR0915
    t0 = time.time()
    px = pd.read_excel(XLS, dtype=str)
    px = px.rename(columns={px.columns[1]: "名称"})
    px["代码"] = px["代码"].str.zfill(6)
    pool = dict(zip(px.代码, px.名称, strict=True))

    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "raw_close", "float_mv", "turnover", "amount", "volume",
            "is_st", "is_suspended", "listed_days"]
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
    turn = al("turnover").rolling(20, min_periods=10).mean().to_numpy()
    amt = al("amount").to_numpy()
    raw = al("raw_close").to_numpy()
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).to_numpy(np.float64)
    ok &= np.isfinite(cl)
    fm, abad = build_fund(list(cldf.columns), idx)
    assert abad == 0, "锚点 TTM"
    dfc = pd.DataFrame(cl)
    lo250 = dfc.rolling(250, min_periods=250).min().to_numpy()
    ma20 = dfc.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        r120 = cl / np.roll(cl, 120, axis=0) - 1.0
        r120[:120] = np.nan
        r60 = cl / np.roll(cl, 60, axis=0) - 1.0
        r60[:60] = np.nan
        lr = np.log(cl / np.roll(cl, 1, axis=0))
        lr[0] = np.nan
        amih = pd.DataFrame(np.abs(lr) / np.where(amt > 0, amt, np.nan)).rolling(
            20, min_periods=10).mean().to_numpy()
        capm = mv * 1e8
        ep = fm["eps_ttm"] / np.where(raw > 0, raw, np.nan)
        cfp = fm["ocf_ttm"] / capm if "ocf_ttm" in fm else np.full_like(cl, np.nan)
        spv = fm["rev_ttm"] / capm if "rev_ttm" in fm else np.full_like(cl, np.nan)
        bp = fm["bps"] / np.where(raw > 0, raw, np.nan) if "bps" in fm \
            else np.full_like(cl, np.nan)
        roe = fm["roe"] if "roe" in fm else np.full_like(cl, np.nan)
        marg = (fm["ni_ttm"] / np.where(fm["rev_ttm"] > 0, fm["rev_ttm"], np.nan)
                if "rev_ttm" in fm and "ni_ttm" in fm else np.full_like(cl, np.nan))
    ab = pd.DataFrame(cl > ma20).rolling(120, min_periods=120).mean().to_numpy()
    v20 = pd.DataFrame(lr).rolling(20, min_periods=20).std().to_numpy()
    rps60 = pd.DataFrame(np.where(ok, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    print(f"面板+财务就绪 ({time.time()-t0:.0f}s);fm 字段 {sorted(fm)}", flush=True)

    varmap = {"流通市值(小→大)": mv, "20日均换手": turn, "Amihud(低流动性大)": amih,
            "EP_TTM": ep, "BP": bp, "CFP_TTM": cfp, "SP_TTM": spv,
            "ROE": roe, "净利率": marg,
            "距一年低点": rec, "120日收益": r120, "MA20持续度": ab,
            "RPS60": rps60, "20日波动率": v20}
    l1s, _ = load_labels()
    cp = {c: j for j, c in enumerate(cldf.columns)}
    ip = pd.Index(idx)
    recs = []
    for ty, ds in ((2024, "2023-12-29"), (2025, "2024-12-31")):
        t = int(ip.get_indexer([pd.Timestamp(ds)], method="ffill")[0])
        elig = [c for c in pool if c in cp and ok[t, cp[c]]]
        js = np.array([cp[c] for c in elig])
        y = np.array([(ty, c) in l1s for c in elig])
        recs.append((ty, t, elig, js, y))
        print(f"\n{ty} 年:池内合格 {len(elig)},翻倍 {int(y.sum())} "
              f"(**池内基准 {y.mean():.2%}**)", flush=True)

    rows = []
    for ty, t, elig, js, y in recs:
        base = y.mean()
        if base <= 0:
            continue
        print(f"\n{'='*96}\n{ty} 年 —— 池内基准 {base:.2%},n={len(elig)}"
              f"\n{'='*96}")
        print(f"{'变量':<20}{'低30%':>22}{'中40%':>22}{'高30%':>22}")
        masks = {}
        for nm, mat in varmap.items():
            v = mat[t, js]
            gd = np.isfinite(v)
            if gd.sum() < 60:
                print(f"{nm:<20}{'可算样本<60,跳过':>22}")
                continue
            q = pd.Series(np.where(gd, v, np.nan)).rank(pct=True).to_numpy()
            line = f"{nm:<20}"
            for lo, hi, tag in ((0, .30, "低"), (.30, .70, "中"), (.70, 1.01, "高")):
                m = gd & (q >= lo) & (q < hi)
                if m.sum() < 20:
                    line += f"{'n<20':>22}"
                    continue
                hr = y[m].mean()
                line += f"{int(m.sum()):>5d}只 {hr:>6.1%} lift{hr/base:>5.2f}"
                masks[(nm, tag)] = m
                rows.append({"年": ty, "变量": nm, "组": tag, "n": int(m.sum()),
                             "翻倍率": float(hr), "lift": float(hr / base),
                             "池内基准": float(base)})
            print(line)
        rg = np.random.default_rng(20260827)
        best = []
        for _ in range(N_PERM):
            yy = rg.permutation(y)
            best.append(max(yy[m].mean() / base for m in masks.values()))
        hi95 = float(np.percentile(best, 95))
        top = max((r for r in rows if r["年"] == ty), key=lambda z: z["lift"])
        print(f"\n纪律A 噪音上界:池内打乱标签 {N_PERM} 次,best-of-{len(masks)} 的 lift "
              f"中位 {np.median(best):.2f}  **95分位 {hi95:.2f}**")
        print(f"  本年最高 lift **{top['lift']:.2f}**({top['变量']} {top['组']}组)"
              f" → {'**超出噪音上界**' if top['lift'] > hi95 else '**未超出,不能算发现**'}")
        for r in rows:
            if r["年"] == ty:
                r["噪音上界95"] = hi95
                r["超上界"] = r["lift"] > hi95

    df = pd.DataFrame(rows)
    print(f"\n{'='*96}\n两年都超噪音上界的(变量,组)\n{'='*96}")
    ok2 = df[df["超上界"]].groupby(["变量", "组"]).size()
    both = [k for k, v in ok2.items() if v == 2]
    print("  " + (", ".join(f"{a}·{b}" for a, b in both) if both else "**无**"))
    df.to_csv(f"{OUT}/pool_factor_scan.csv", index=False, encoding="utf-8-sig")
    print(f"\n落库 {OUT}/pool_factor_scan.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
