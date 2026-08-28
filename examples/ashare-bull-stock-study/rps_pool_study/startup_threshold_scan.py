"""§135 事前登记:X01「价格启动线索」五变量阈值扫描 + 同市值同行业对照(结果未跑)。

任务来源
--------
X01-v1 任务断点文档(2026-08-26)第「下一步直接执行」节要求:对
`recovery_from_low_250` / `close_to_ma_250` / `rps_60` / `vol_20` /
`above_ma20_share_120` 做阈值扫描,每档报「选择数量、覆盖率、L1/L2 命中率、
相对基础命中率的 Lift、牛股召回率、L2 命中率」,最后给
**观察档 / 标准启动档 / 强确认档**三档。

本节照做,并加一列(唯一的加法)
--------------------------------
X01 现在的 Lift 分母是**全样本基准命中率**。本节每一档**同时报两个 Lift**:
  - `lift_base` = 命中率 ÷ 全样本基准命中率(X01 口径,保留以便对照)
  - `lift_ctrl` = 命中率 ÷ **同市值名次±25 且同申万一级行业**对照的命中率中位数
                  (第一二五节口径,500 组种子)
**加这一列的理由是第一三三节的实例**:安信骨架的 B2 让命中率 2.81%→4.35%,
`lift_base` 看着有用,**但对照同步 3.63%→5.31%,`lift_ctrl` 增量为零**——
它只是把样本挪进了本来概率就高的一群股票。**不加这一列,五个变量都会长得很漂亮。**

口径
----
- 观察日:**每年最后一个交易日**;目标 = **下一个自然年**(与 X01 的 2025-12-31→2026 一致)。
- 合格样本(L0):观察日 非ST、非停牌、上市满 250 日、当日有成交(X01 口径)。
- **L1/L2 标签取自 quant-research-dev 牛股普查(只读)**:
  `annual_gt100` 三张表并集,(目标年, code) 命中即 `label_bull_any`(L1 或 L2);
  其中 `annual_return > 2.0`(三倍)记为 **L2**。
- 目标年区间:**全样本 2015–2025**(面板 2013-01 起,250 日特征需要 2014 年末观察日);
  **现代段 2019–2025**(X01 第一层训练区间),两段分别报。
- 特征全部在观察日当日计算,**只用 ≤t 的信息**。

阈值候选(照断点文档,不增不减)
--------------------------------
- `recovery_from_low_250`:0–20% / 20–40% / 40–60% / 60–100% / 100–200% / >200%
- `close_to_ma_250`:<0 / 0–10% / 10–20% / 20–40% / 40–60% / 60–100% / >100%
- `rps_60`:≥60 / ≥70 / ≥80 / ≥85 / ≥90 / ≥95
- `vol_20`:年度横截面 ≥50 / ≥60 / ≥70 / ≥80 / ≥90 分位
- `above_ma20_share_120`:≥50% / ≥55% / ≥60% / ≥65% / ≥70% / ≥75% / ≥80% / ≥90%

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
T1 锚点(不过则本节作废)
   (a) 面板 (3297, 5232);
   (b) **行业恒等式**:对照与被对照股同属一个申万一级行业,违例 > 0 即作废;
   (c) **无前视**:五个特征逐点重算断言;标签只来自**观察日之后**的自然年;
   (d) **标签一致性**:全样本基准命中率须与普查年度汇总对得上
       (本面板合格样本上的 L1/L2 比例,与普查同年翻倍股只数量级一致,逐年打印)。

T2 三档划分。**规则跑之前写死,跑完不许改,不许因为某档空着就放宽。**
   - **观察档**:`lift_ctrl` ≥ **1.2** 且 覆盖率 ≥ **20%**
   - **标准启动档**:`lift_ctrl` ≥ **1.5** 且 覆盖率 ≥ **5%**
   - **强确认档**:`lift_ctrl` ≥ **2.0** 且 选中样本数 ≥ **200**
   **三档一律以 `lift_ctrl` 为准,不用 `lift_base`。**
   某变量某档无阈值满足 → 明确输出「该档无满足阈值」,**不降标准凑一个出来**。

T3 噪音上界(纪律A,防 32 个阈值的多重比较)
   在每个目标年内**打乱 L1/L2 标签**(保留各年基准率)**200 次**,
   每次记录**所有阈值里最高的 `lift_ctrl`**,得到纯噪音下 best-of-N 的分布。
   **某阈值的 `lift_ctrl` 必须超过该分布的 95 分位,才允许写进三档。**
   这条与 `bull_features/bull_feature_scan.py` 的纪律A 同规格。

T4 描述项(不参与判定):按 X01 断点文档要求,逐档同时列出
   选中数量、覆盖率、L1/L2 命中率、L2 命中率、牛股召回率、`lift_base`、`lift_ctrl`。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。只登记判据。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;**不往 quant-research-dev 推任何东西**;
**不改断点文档给的阈值候选**(不增删、不微调边界);
**不因某个阈值好看就跨变量组合出一个新规则** —— 组合要另开一节重新事前登记;
**不基于本节结论做任何可交易性声明**。
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

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
CENSUS = ("/home/user/quant-research-dev/research/"
          "bull-stock-census-2010-2025/data")
NSEED, N_PERM = 500, 200
Y_ALL, Y_MOD = (2015, 2025), (2019, 2025)


def load_labels():
    fr = []
    for n in ("annual_gt100_main", "annual_gt100_listing_year",
              "annual_gt100_delisted"):
        x = pd.read_csv(f"{CENSUS}/{n}.csv", dtype={"code": str})
        x.columns = [c.lstrip("﻿") for c in x.columns]
        x["code"] = x["code"].str.zfill(6)
        fr.append(x[["year", "code", "annual_return"]])
    a = pd.concat(fr, ignore_index=True).drop_duplicates(["year", "code"])
    l1 = set(zip(a.year, a.code, strict=True))
    l2 = set(zip(a[a.annual_return > 2.0].year,
                 a[a.annual_return > 2.0].code, strict=True))
    return l1, l2


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
    assert (nt, ns) == (3297, 5232), f"锚点T1a {cldf.shape}"

    def al(k, fill=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(fill)
    mv = al("float_mv").to_numpy() / 1e8
    vol = al("volume", 0).to_numpy()
    st = al("is_st", True).astype(bool).to_numpy()
    sus = al("is_suspended", True).astype(bool).to_numpy()
    ld = al("listed_days", 0).to_numpy()
    cl = cldf.where(cldf > 0).to_numpy(np.float64)
    ind, _, _ = build_industry(list(cldf.columns), idx)
    ok = ~st & ~sus & (ld >= 250) & (vol > 0) & np.isfinite(cl)
    print(f"锚点T1a ✓ 面板 {cldf.shape};行业覆盖 {(ind>=0).mean():.1%} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 五个特征(观察日当日,只用 <=t)----
    dfc = pd.DataFrame(cl)
    lo250 = dfc.rolling(250, min_periods=250).min().to_numpy()
    ma250 = dfc.rolling(250, min_periods=250).mean().to_numpy()
    ma20 = dfc.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        c2ma = cl / np.where(ma250 > 0, ma250, np.nan) - 1.0
        r60 = cl / np.roll(cl, 60, axis=0) - 1.0
        r60[:60] = np.nan
        lr = np.log(cl / np.roll(cl, 1, axis=0))
        lr[0] = np.nan
    v20 = pd.DataFrame(lr).rolling(20, min_periods=20).std().to_numpy()
    ab = pd.DataFrame(cl > ma20).rolling(120, min_periods=120).mean().to_numpy()
    rps60 = pd.DataFrame(np.where(ok, r60, np.nan)).rank(
        axis=1, pct=True).to_numpy() * 100
    print(f"特征完成 ({time.time()-t0:.0f}s)", flush=True)

    # 锚点 T1c 无前视
    rs = np.random.default_rng(3)
    nchk = 0
    for _ in range(3000):
        t = int(rs.integers(260, nt))
        j = int(rs.integers(0, ns))
        if np.isfinite(rec[t, j]):
            ref = cl[t, j] / np.nanmin(cl[t - 249:t + 1, j]) - 1.0
            assert abs(ref - rec[t, j]) < 1e-9, "T1c rec"
            nchk += 1
        if np.isfinite(ab[t, j]):
            w = (cl[t - 119:t + 1, j] > ma20[t - 119:t + 1, j]).mean()
            assert abs(w - ab[t, j]) < 1e-9, "T1c above_ma20"
    print(f"锚点T1c 特征因果性 {nchk} 点一致 ✓", flush=True)

    # ---- 年末观察日 + 标签 ----
    l1s, l2s = load_labels()
    yend = pd.Series(np.arange(nt), index=idx).groupby(idx.year).last()
    rows = []
    for y, t in yend.items():
        ty = int(y) + 1
        if not (Y_ALL[0] <= ty <= Y_ALL[1]):
            continue
        e = np.flatnonzero(ok[t] & np.isfinite(rec[t]) & np.isfinite(c2ma[t])
                           & np.isfinite(v20[t]) & np.isfinite(ab[t])
                           & np.isfinite(rps60[t]) & np.isfinite(mv[t])
                           & (ind[t] >= 0))
        vq = pd.Series(v20[t, e]).rank(pct=True).to_numpy() * 100
        for k, j in enumerate(e):
            c = cldf.columns[j]
            rows.append((ty, int(t), int(j), c, rec[t, j], c2ma[t, j],
                         rps60[t, j], vq[k], ab[t, j],
                         (ty, c) in l1s, (ty, c) in l2s))
    p = pd.DataFrame(rows, columns=["ty", "t", "j", "code", "rec", "c2ma",
                                    "rps60", "vq20", "ab120", "L1", "L2"])
    print("\n锚点T1d 逐年基准命中率(合格样本 / L1|L2 / L2)", flush=True)
    for ty, g in p.groupby("ty"):
        print(f"  目标年 {ty}  合格 {len(g):5d}  L1|L2 {int(g.L1.sum()):4d} "
              f"({g.L1.mean():5.2%})  L2 {int(g.L2.sum()):3d} ({g.L2.mean():5.2%})")

    # ---- 对照:同市值名次 ±NBR 且同行业 ----
    pre = {}
    for t in p.t.unique():
        e = np.flatnonzero(ok[t] & np.isfinite(mv[t]) & (ind[t] >= 0))
        o = e[np.argsort(mv[t, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        pre[t] = (o, rk)
    tv, jv = p.t.to_numpy(), p.j.to_numpy()
    chunks, off, lens = [], np.zeros(len(p), np.int64), np.zeros(len(p), np.int64)
    pos_f = 0
    keep = np.ones(len(p), bool)
    for k in range(len(p)):
        t, j = int(tv[k]), int(jv[k])
        o, rk = pre[t]
        p0, i0 = rk[j], ind[t, j]
        if p0 < 0 or i0 < 0:
            keep[k] = False
            continue
        a_, b_ = max(0, p0 - NBR), min(len(o) - 1, p0 + NBR)
        cand = o[a_:b_ + 1]
        cand = cand[ind[t, cand] == i0]
        if len(cand) < 2:
            cand = o[ind[t, o] == i0]
        if len(cand) < 2:
            keep[k] = False
            continue
        off[k] = pos_f
        lens[k] = len(cand)
        pos_f += len(cand)
        chunks.append(cand)
    flat = np.concatenate(chunks).astype(np.int64)
    lab1 = np.zeros((nt, ns), bool)
    lab2 = np.zeros((nt, ns), bool)
    lab1[tv[p.L1.to_numpy()], jv[p.L1.to_numpy()]] = True
    lab2[tv[p.L2.to_numpy()], jv[p.L2.to_numpy()]] = True
    valid = np.zeros((nt, ns), bool)
    valid[tv, jv] = True
    print(f"对照候选就绪,剔除无对照 {int((~keep).sum())} 条 ({time.time()-t0:.0f}s)",
          flush=True)

    rng = np.random.default_rng(SEED)
    picks = np.empty((NSEED, len(p)), np.int64)
    kk = np.flatnonzero(keep)
    for s0 in range(0, NSEED, 50):
        r = rng.random((50, len(kk)))
        picks[s0:s0 + 50, kk] = flat[off[kk][None, :]
                                     + (r * lens[kk][None, :]).astype(np.int64)]
    picks[:, ~keep] = -1
    viol = int((ind[tv[kk], picks[:, kk]] != ind[tv[kk], jv[kk]][None, :]).sum())
    print(f"锚点T1b 行业违例 {viol} 次 {'✓' if viol == 0 else '✗ 作废'}", flush=True)
    assert viol == 0

    cands_all = {
        "recovery_from_low_250": [("0-20%", (0.0, 0.20)), ("20-40%", (0.20, 0.40)),
                                  ("40-60%", (0.40, 0.60)), ("60-100%", (0.60, 1.00)),
                                  ("100-200%", (1.00, 2.00)), (">200%", (2.00, 1e9))],
        "close_to_ma_250": [("<0", (-1e9, 0.0)), ("0-10%", (0.0, 0.10)),
                            ("10-20%", (0.10, 0.20)), ("20-40%", (0.20, 0.40)),
                            ("40-60%", (0.40, 0.60)), ("60-100%", (0.60, 1.00)),
                            (">100%", (1.00, 1e9))],
        "rps_60": [(f">={v}", (v, 1e9)) for v in (60, 70, 80, 85, 90, 95)],
        "vol_20_pct": [(f">={v}分位", (v, 1e9)) for v in (50, 60, 70, 80, 90)],
        "above_ma20_share_120": [(f">={int(v*100)}%", (v, 1e9))
                                 for v in (.50, .55, .60, .65, .70, .75, .80, .90)],
    }
    colmap = {"recovery_from_low_250": "rec", "close_to_ma_250": "c2ma",
            "rps_60": "rps60", "vol_20_pct": "vq20",
            "above_ma20_share_120": "ab120"}

    def scan(sub_mask, tag):
        sp = p[sub_mask]
        si = np.flatnonzero(sub_mask.to_numpy())
        base1, base2 = sp.L1.mean(), sp.L2.mean()
        nb = int(sp.L1.sum())
        print(f"\n{'='*112}\n{tag}:合格 {len(sp):,};基准 L1|L2 {base1:.2%};"
              f"L2 {base2:.2%}\n{'='*112}")
        print(f"{'变量':<24}{'阈值':<12}{'选中':>7}{'覆盖率':>8}{'命中率':>8}"
              f"{'L2命中':>8}{'召回':>7}{'lift_base':>10}{'lift_ctrl':>10}")
        out, masks = [], {}
        for var, cands in cands_all.items():
            col = sp[colmap[var]].to_numpy()
            for nm, (lo, hi) in cands:
                m = (col >= lo) & (col < hi)
                n1 = int(m.sum())
                if n1 < 30:
                    print(f"{var:<24}{nm:<12}{n1:>7}  样本<30,不判")
                    continue
                hr = float(sp.L1.to_numpy()[m].mean())
                h2 = float(sp.L2.to_numpy()[m].mean())
                rc = float(sp.L1.to_numpy()[m].sum() / max(nb, 1))
                cov = n1 / len(sp)
                gi = si[m]
                pk = picks[:, gi]
                good = pk >= 0
                cm = np.where(good, lab1[tv[gi][None, :], np.maximum(pk, 0)], False)
                cv = np.where(good, valid[tv[gi][None, :], np.maximum(pk, 0)], False)
                nv = cv.sum(1)
                cr = np.where(nv > 0, cm.sum(1) / np.maximum(nv, 1), np.nan)
                cmed = float(np.nanmedian(cr))
                lb = hr / base1 if base1 > 0 else np.nan
                lc = hr / cmed if cmed > 0 else np.nan
                print(f"{var:<24}{nm:<12}{n1:>7,}{cov:>8.1%}{hr:>8.2%}{h2:>8.2%}"
                      f"{rc:>7.1%}{lb:>10.2f}{lc:>10.2f}")
                out.append({"段": tag, "变量": var, "阈值": nm, "选中": n1,
                            "覆盖率": cov, "命中率": hr, "L2命中": h2, "召回": rc,
                            "lift_base": lb, "lift_ctrl": lc, "对照命中": cmed})
                masks[(var, nm)] = m
        # T3 噪音上界
        yv = sp.ty.to_numpy()
        lb1 = sp.L1.to_numpy()
        rg = np.random.default_rng(SEED + 7)
        best = []
        posy = {y: np.flatnonzero(yv == y) for y in np.unique(yv)}
        for _ in range(N_PERM):
            bb = np.zeros(len(sp), bool)
            for y, e in posy.items():
                k2 = int(lb1[e].sum())
                if k2:
                    bb[rg.choice(e, k2, replace=False)] = True
            bst = 0.0
            for key, m in masks.items():
                r0 = [o for o in out if (o["变量"], o["阈值"]) == key][0]
                if r0["对照命中"] > 0:
                    bst = max(bst, bb[m].mean() / r0["对照命中"])
            best.append(bst)
        hi95 = float(np.percentile(best, 95))
        print(f"\nT3 噪音上界:年内打乱标签 {N_PERM} 次,best-of-{len(masks)} 的 "
              f"lift_ctrl 中位 {np.median(best):.2f}  **95分位 {hi95:.2f}**")
        for o in out:
            o["超噪音上界"] = bool(o["lift_ctrl"] > hi95)
        # T2 三档
        print(f"\nT2 三档(一律以 lift_ctrl 为准,且须超噪音上界 {hi95:.2f})")
        for lab, (lmin, extra) in {
                "观察档   (lift_ctrl≥1.2 且 覆盖率≥20%)": (1.2, ("cov", 0.20)),
                "标准启动档(lift_ctrl≥1.5 且 覆盖率≥5%)": (1.5, ("cov", 0.05)),
                "强确认档 (lift_ctrl≥2.0 且 选中≥200)": (2.0, ("n", 200))}.items():
            hits = [o for o in out if o["lift_ctrl"] >= lmin and o["超噪音上界"]
                    and ((o["覆盖率"] >= extra[1]) if extra[0] == "cov"
                         else (o["选中"] >= extra[1]))]
            if not hits:
                print(f"  {lab}:**该档无满足阈值**")
            for o in sorted(hits, key=lambda z: -z["lift_ctrl"]):
                print(f"  {lab}:{o['变量']} {o['阈值']}  "
                      f"lift_ctrl {o['lift_ctrl']:.2f}  覆盖 {o['覆盖率']:.1%}  "
                      f"命中 {o['命中率']:.2%}")
        return out

    res = scan(pd.Series(np.ones(len(p), bool)), "全样本 2015–2025")
    res += scan((p.ty >= Y_MOD[0]) & (p.ty <= Y_MOD[1]), "现代段 2019–2025")
    pd.DataFrame(res).to_csv(f"{OUT}/startup_threshold_scan.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n落库 {OUT}/startup_threshold_scan.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
