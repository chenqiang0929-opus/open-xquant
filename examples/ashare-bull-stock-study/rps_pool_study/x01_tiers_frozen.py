"""§142:把 Codex 的 X01 三档规则原样冻结,在本面板复跑 + 时间样本外。

起因
----
用户:「codex 都出结论了,你看看,你这边还没有任何结论。」
**这个批评成立。** 第一三五节我按 lift_ctrl 判,三档全空;
补记里我已承认「X01 要的是观察名单,该用途下 lift_base 才是相关指标」,
**却没有按那个口径把三档结论给出来** —— 只说了「你的口径够用」就停住。
本节补上,并做 Codex 自己在报告第 6 节说要做、但还没做的那一步:
**「冻结三档规则,再按年度滚动或未来完整年度检验。」**

Codex 的三档(逐字照抄 `X01_v1_阈值实证扫描_20260827.md` 第 5 节,一个数没改)
--------------------------------------------------------------------------
- **观察档**:距一年低点涨幅 ≥ 40% 且 120 日收益 ≥ 10%
- **标准启动档**:观察档 且 近120日站上MA20比例 ≥ 55% 且 RPS60 ≥ 80
- **强确认档**:观察档 且 近120日站上MA20比例 ≥ 55% 且 RPS60 ≥ 90

Codex 公布值(2019—2025):
  观察档 命中率 7.88% / Lift 1.56 / 覆盖牛股 36.0%
  标准档 命中率 8.04% / Lift 1.59 / 覆盖 13.4%
  强确认 命中率 8.76% / Lift 1.73 / 覆盖 9.3%

本节做什么
----------
1. **复现**:同口径(观察日=每年最后一个交易日,目标=下一自然年,
   标签取自 quant-research-dev 牛股普查)在本面板重算,与他的数字并列;
2. **两个 Lift 都报**:`lift_base`(与他可比)与 `lift_ctrl`
   (同市值名次±25 且同申万一级行业,500 组种子);
3. **时间样本外**(他没做):**目标年 2019–2022 为训练段、2023–2025 为留出段**,
   三档规则**在两段各跑一次**,看留出段是否保持。
   **注意:规则是他在 2012–2025 全样本上挑的,所以 2023–2025 对他不是真样本外;
   本节的留出只能检验「同一规则在后段是否退化」,不能当成干净的样本外。这句必须写进结论。**

判据
----
**本节不设通过/不通过判据** —— 这是复现 + 分段描述,不是假设检验。
锚点(不过则作废):面板 (3297, 5232);行业违例 0;无前视逐点断言;
基础率须与 Codex 的 5.06%(2019–2025)同量级。

不做的
------
不改 Codex 的任何阈值(不增删、不微调);不新增顶层目录;不 force push;
**不因为哪一档好看就宣称可以拿去交易**;不作可交易性声明。
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
from codex_r10_neutral import NBR, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from industry_neutral import build_industry  # noqa: E402
from startup_threshold_scan import load_labels  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NSEED = 500


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "float_mv", "volume", "is_st", "is_suspended", "listed_days"]
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
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).to_numpy(np.float64)
    ok &= np.isfinite(cl)
    ind, _, _ = build_industry(list(cldf.columns), idx)

    dfc = pd.DataFrame(cl)
    lo250 = dfc.rolling(250, min_periods=250).min().to_numpy()
    ma20 = dfc.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        r120 = cl / np.roll(cl, 120, axis=0) - 1.0
        r120[:120] = np.nan
        r60 = cl / np.roll(cl, 60, axis=0) - 1.0
        r60[:60] = np.nan
    ab = pd.DataFrame(cl > ma20).rolling(120, min_periods=120).mean().to_numpy()
    rps60 = pd.DataFrame(np.where(ok, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    rs = np.random.default_rng(9)
    n = 0
    for _ in range(2000):
        t = int(rs.integers(260, nt))
        j = int(rs.integers(0, ns))
        if np.isfinite(rec[t, j]):
            ref = cl[t, j] / np.nanmin(cl[t - 249:t + 1, j]) - 1.0
            assert abs(ref - rec[t, j]) < 1e-9, "无前视 rec"
            n += 1
    print(f"锚点 面板 {cldf.shape} ✓;无前视 {n} 点 ✓ ({time.time()-t0:.0f}s)", flush=True)

    l1s, l2s = load_labels()
    yend = pd.Series(np.arange(nt), index=idx).groupby(idx.year).last()
    rows = []
    for y, t in yend.items():
        ty = int(y) + 1
        if not (2019 <= ty <= 2025):
            continue
        e = np.flatnonzero(ok[t] & np.isfinite(rec[t]) & np.isfinite(r120[t])
                           & np.isfinite(ab[t]) & np.isfinite(rps60[t])
                           & np.isfinite(mv[t]) & (ind[t] >= 0))
        for j in e:
            c = cldf.columns[j]
            rows.append((ty, int(t), int(j), rec[t, j], r120[t, j], ab[t, j],
                         rps60[t, j], (ty, c) in l1s, (ty, c) in l2s))
    p = pd.DataFrame(rows, columns=["ty", "t", "j", "rec", "r120", "ab",
                                    "rps60", "L1", "L2"])
    print(f"样本 {len(p):,};基准 L1|L2 {p.L1.mean():.2%}(Codex 5.06%);"
          f"L2 {p.L2.mean():.2%}(Codex 1.13%)", flush=True)

    tv, jv = p.t.to_numpy(), p.j.to_numpy()
    pre = {}
    for t in np.unique(tv):
        e = np.flatnonzero(ok[t] & np.isfinite(mv[t]) & (ind[t] >= 0))
        o = e[np.argsort(mv[t, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        pre[t] = (o, rk)
    ch, off, lens = [], np.zeros(len(p), np.int64), np.zeros(len(p), np.int64)
    pos, keep = 0, np.ones(len(p), bool)
    for k in range(len(p)):
        t, j = int(tv[k]), int(jv[k])
        o, rk = pre[t]
        p0, i0 = rk[j], ind[t, j]
        a_, b_ = max(0, p0 - NBR), min(len(o) - 1, p0 + NBR)
        cand = o[a_:b_ + 1]
        cand = cand[ind[t, cand] == i0]
        if len(cand) < 2:
            cand = o[ind[t, o] == i0]
        if len(cand) < 2:
            keep[k] = False
            continue
        off[k], lens[k] = pos, len(cand)
        pos += len(cand)
        ch.append(cand)
    flat = np.concatenate(ch).astype(np.int64)
    rng = np.random.default_rng(SEED)
    pk = np.full((NSEED, len(p)), -1, np.int64)
    kk = np.flatnonzero(keep)
    for s0 in range(0, NSEED, 50):
        r = rng.random((50, len(kk)))
        pk[s0:s0 + 50, kk] = flat[off[kk][None, :]
                                  + (r * lens[kk][None, :]).astype(np.int64)]
    v = int((ind[tv[kk], pk[:, kk]] != ind[tv[kk], jv[kk]][None, :]).sum())
    print(f"锚点 行业违例 {v} 次 {'✓' if v == 0 else '✗ 作废'}", flush=True)
    assert v == 0
    lab1 = np.zeros((nt, ns), bool)
    lab1[tv[p.L1.to_numpy()], jv[p.L1.to_numpy()]] = True
    valid = np.zeros((nt, ns), bool)
    valid[tv, jv] = True

    base_m = (p.rec.to_numpy() >= 0.40) & (p.r120.to_numpy() >= 0.10)
    tiers = {
        "观察档": base_m,
        "标准启动档": base_m & (p.ab.to_numpy() >= 0.55) & (p.rps60.to_numpy() >= 80),
        "强确认档": base_m & (p.ab.to_numpy() >= 0.55) & (p.rps60.to_numpy() >= 90),
    }
    codex_pub = {"观察档": (7.88, 1.56, 36.0), "标准启动档": (8.04, 1.59, 13.4),
             "强确认档": (8.76, 1.73, 9.3)}

    def ev(m, seg, tag):
        sel = m & seg
        gi = np.flatnonzero(sel)
        if len(gi) < 50:
            return None
        sub = p[seg]
        b = float(sub.L1.mean())
        hr = float(p.L1.to_numpy()[gi].mean())
        h2 = float(p.L2.to_numpy()[gi].mean())
        cov = len(gi) / len(sub)
        rc = float(p.L1.to_numpy()[gi].sum() / max(sub.L1.sum(), 1))
        q = pk[:, gi]
        gg = q >= 0
        tq = tv[gi][None, :]
        cm = np.where(gg, lab1[tq, np.maximum(q, 0)], False)
        cv = np.where(gg, valid[tq, np.maximum(q, 0)], False)
        nv = cv.sum(1)
        cmed = float(np.nanmedian(np.where(nv > 0, cm.sum(1) / np.maximum(nv, 1),
                                           np.nan)))
        return {"段": tag, "档": None, "n": len(gi), "覆盖率": cov, "命中率": hr,
                "L2命中": h2, "召回": rc, "基准": b, "lift_base": hr / b,
                "lift_ctrl": hr / cmed if cmed > 0 else np.nan}

    res = []
    print("\n" + "=" * 104)
    print("一、复现 Codex 三档(2019–2025 全段),并列两个 Lift")
    print("=" * 104)
    print(f"{'档':<12}{'n':>7}{'覆盖率':>8}{'命中率':>8}{'Codex':>8}"
          f"{'lift_base':>11}{'Codex':>8}{'召回':>8}{'Codex':>8}{'lift_ctrl':>11}")
    allseg = np.ones(len(p), bool)
    for nm, m in tiers.items():
        r = ev(m, allseg, "2019–2025")
        r["档"] = nm
        res.append(r)
        c = codex_pub[nm]
        print(f"{nm:<12}{r['n']:>7,}{r['覆盖率']:>8.1%}{r['命中率']:>8.2%}"
              f"{c[0]:>7.2f}%{r['lift_base']:>11.2f}{c[1]:>8.2f}"
              f"{r['召回']:>8.1%}{c[2]:>7.1f}%{r['lift_ctrl']:>11.2f}")

    print("\n" + "=" * 104)
    print("二、分段:目标年 2019–2022 vs 2023–2025(规则不变)")
    print("=" * 104)
    print(f"{'档':<12}{'段':<12}{'n':>7}{'基准':>8}{'命中率':>8}"
          f"{'lift_base':>11}{'lift_ctrl':>11}{'召回':>8}")
    for nm, m in tiers.items():
        for lo, hi, tag in ((2019, 2022, "2019–2022"), (2023, 2025, "2023–2025")):
            seg = ((p.ty.to_numpy() >= lo) & (p.ty.to_numpy() <= hi))
            r = ev(m, seg, tag)
            if r is None:
                continue
            r["档"] = nm
            res.append(r)
            print(f"{nm:<12}{tag:<12}{r['n']:>7,}{r['基准']:>8.2%}"
                  f"{r['命中率']:>8.2%}{r['lift_base']:>11.2f}"
                  f"{r['lift_ctrl']:>11.2f}{r['召回']:>8.1%}")

    print("\n" + "=" * 104)
    print("三、逐年(观察档)")
    print("=" * 104)
    print(f"{'目标年':<8}{'基准':>8}{'n':>7}{'命中率':>8}{'lift_base':>11}{'lift_ctrl':>11}")
    for ty in range(2019, 2026):
        seg = p.ty.to_numpy() == ty
        r = ev(tiers["观察档"], seg, str(ty))
        if r is None:
            continue
        r["档"] = "观察档"
        res.append(r)
        print(f"{ty:<8}{r['基准']:>8.2%}{r['n']:>7,}{r['命中率']:>8.2%}"
              f"{r['lift_base']:>11.2f}{r['lift_ctrl']:>11.2f}")
    pd.DataFrame(res).to_csv(f"{OUT}/x01_tiers_frozen.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n落库 {OUT}/x01_tiers_frozen.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
