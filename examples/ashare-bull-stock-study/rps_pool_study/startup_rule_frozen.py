"""§148 事前登记:冻结**一个**启动规则,全市场 2019 起,做时间样本外(结果未跑)。

起因
----
用户:「就没有一个相对统一的标准吗,有点乱,实在不行,把股票池推到 2019 年开始」

**「有点乱」这个批评成立。** 第一四七节我报了 9 个单变量 + 6 个组合,
没有给出一个可用的标准,而且那些组合是我在同一批数据上挑的。
本节**只冻结一个规则**,并按用户说的推到 2019、换成全市场,
做第一四七节欠着的那件事:**时间样本外**。

规则(**只有一条,跑之前写死,跑完不改**)
----------------------------------------
    距一年低点涨幅 ∈ 全市场当日前 30%
      且
    换手加速(20日均换手 ÷ 60日均换手 − 1)∈ 全市场当日前 30%

**为什么是这两条,必须说清楚**:
第一四七节在用户的 663 只池内,「距一年低点」是单变量 lift 最高的(1.45),
「换手加速」是此前从未测过的新维度且覆盖率合适(组合后覆盖 11.3%、lift 1.72)。
**这两条是我从那一节挑出来的,存在研究者自由度** —— 所以本节必须做样本外,
否则不算数。**不再试第二个组合;本节只判这一条。**

口径(与第一四七节一致,一个字不改)
----------------------------------
- 全市场 5,232 只;观察点 = **每月最后一个交易日**;区间 **2019-01 → 2026-04**
- 合格:非 ST、非停牌、上市满 250 日、当日有成交
- **启动 = 未来 60 个交易日涨幅 ≥ 50%**;同股 **60 日内不重复计事件**
- 分位在**全市场当日横截面**上取,不是池内

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
G1 锚点(不过则本节作废)
   (a) 面板 (3297, 5232);
   (b) **无前视**:距低点、换手加速逐点重算断言;前瞻窗口起点严格 > t。

G2 **核心判据 —— 时间样本外。**
   训练段 **2019-01 → 2022-12**:只报数,**不参与判定**(规则已在别处挑过,
   这一段没有信息价值)。
   **留出段 2023-01 → 2026-04:判据在这里。**
   **G2 通过 ⟺ 留出段 lift > 1.20 且 lift > 留出段自己的噪音上界 95 分位**
   (打乱标签 200 次,单规则,不取 best-of-N)。
   **两条同时满足才算通过。**

G3 逐年 lift(描述,不参与判定)。**第一三九节的教训:必须看逐年,
   合计数字会被少数年份主导。**

G4 同市值同行业对照(描述):留出段内,选中组 vs 同市值名次±25 且同申万一级
   行业对照的启动率,500 组种子。**报 lift_ctrl,不设门槛。**

事前预测
--------
**本节不下预测**(第一一九节起的约定)。

不做的
------
不改规则、不加第三个条件、不换阈值;不新增顶层目录;不 force push;
**不因为留出段不过就回头改规则** —— 那样本节就白做了;
不作任何可交易性声明。
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
HOR, THR, GAP, N_PERM, NSEED = 60, 0.50, 60, 200, 500
Q = 0.70


def main():  # noqa: PLR0915
    t0 = time.time()
    codes = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"{DATA}/*.parquet"))
             if os.path.basename(f)[:-8] != "510300"]
    cols = ["close", "float_mv", "turnover", "volume", "is_st", "is_suspended",
            "listed_days"]
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
    assert (nt, ns) == (3297, 5232), f"锚点G1a {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f)
    mv = al("float_mv").to_numpy() / 1e8
    trn = al("turnover")
    ok = (~al("is_st", True).astype(bool).to_numpy()
          & ~al("is_suspended", True).astype(bool).to_numpy()
          & (al("listed_days", 0).to_numpy() >= 250)
          & (al("volume", 0).to_numpy() > 0))
    cl = cldf.where(cldf > 0).to_numpy(np.float64)
    ok &= np.isfinite(cl)
    ind, _, _ = build_industry(list(cldf.columns), idx)
    lo250 = pd.DataFrame(cl).rolling(250, min_periods=250).min().to_numpy()
    t20 = trn.rolling(20, min_periods=10).mean().to_numpy()
    t60 = trn.rolling(60, min_periods=30).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        tacc = t20 / np.where(t60 > 0, t60, np.nan) - 1.0
    fmax = pd.DataFrame(cl[::-1]).rolling(HOR, min_periods=1).max().to_numpy()[::-1]
    fwd = np.full_like(cl, np.nan)
    fwd[:-1] = fmax[1:]
    with np.errstate(all="ignore"):
        up = fwd / np.where(cl > 0, cl, np.nan) - 1.0

    rs = np.random.default_rng(13)
    n1 = n2 = 0
    for _ in range(3000):
        t = int(rs.integers(260, nt - HOR - 1))
        j = int(rs.integers(0, ns))
        if np.isfinite(rec[t, j]):
            assert abs(cl[t, j] / np.nanmin(cl[t - 249:t + 1, j]) - 1.0
                       - rec[t, j]) < 1e-9, "G1b rec"
            n1 += 1
        if np.isfinite(up[t, j]):
            pk = np.nanmax(cl[t + 1:t + 1 + HOR, j]) / cl[t, j] - 1.0
            assert abs(pk - up[t, j]) < 1e-9, "G1b 前瞻窗口"
            n2 += 1
    print(f"锚点G1a ✓ {cldf.shape};G1b 无前视 rec {n1} 点、前瞻 {n2} 点 ✓ "
          f"({time.time()-t0:.0f}s)", flush=True)

    me = pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy()
    rows = []
    for t in me:
        t = int(t)
        if t > nt - HOR - 1 or idx[t] < pd.Timestamp("2019-01-01"):
            continue
        m = ok[t] & np.isfinite(rec[t]) & np.isfinite(tacc[t]) & np.isfinite(up[t])
        e = np.flatnonzero(m)
        if len(e) < 100:
            continue
        qr = pd.Series(rec[t, e]).rank(pct=True).to_numpy()
        qt = pd.Series(tacc[t, e]).rank(pct=True).to_numpy()
        hit = (qr >= Q) & (qt >= Q)
        for k, j in enumerate(e):
            rows.append((t, idx[t].year, int(j), bool(hit[k]),
                         bool(up[t, j] >= THR)))
    ev = pd.DataFrame(rows, columns=["t", "year", "j", "sel", "y"])
    ev = ev.sort_values(["j", "t"])
    keep, last = [], {}
    for r in ev.itertuples():
        if r.t - last.get(r.j, -10**9) >= GAP:
            keep.append(True)
            last[r.j] = r.t
        else:
            keep.append(False)
    ev = ev[keep].reset_index(drop=True)
    print(f"月度事件 {len(ev):,}({idx[ev.t.min()].date()} → "
          f"{idx[ev.t.max()].date()})", flush=True)

    def seg(lo, hi, tag, judge):
        s = ev[(ev.year >= lo) & (ev.year <= hi)]
        b = s.y.mean()
        m = s.sel.to_numpy()
        y = s.y.to_numpy()
        hr = y[m].mean()
        lf = hr / b
        rg = np.random.default_rng(SEED)
        perm = [rg.permutation(y)[m].mean() / b for _ in range(N_PERM)]
        h95 = float(np.percentile(perm, 95))
        print(f"\n{'='*88}\n{tag}(事件 {len(s):,},基准启动率 {b:.2%})\n{'='*88}")
        print(f"  选中 {int(m.sum()):,} 只次({m.mean():.1%} 覆盖率),"
              f"启动率 **{hr:.2%}**,**lift {lf:.2f}**,召回 {y[m].sum()/y.sum():.1%}")
        print(f"  噪音上界(打乱标签 {N_PERM} 次):中位 {np.median(perm):.2f},"
              f"**95分位 {h95:.2f}**")
        if judge:
            c1, c2 = lf > 1.20, lf > h95
            print(f"  **G2 判定**:lift>1.20 {'✓' if c1 else '✗'};"
                  f"lift>噪音上界 {'✓' if c2 else '✗'} → "
                  f"**{'通过' if (c1 and c2) else '不通过'}**")
        return {"段": tag, "事件": len(s), "基准": b, "选中": int(m.sum()),
                "覆盖率": float(m.mean()), "启动率": float(hr), "lift": float(lf),
                "召回": float(y[m].sum() / y.sum()), "噪音上界95": h95}

    res = [seg(2019, 2022, "训练段 2019–2022(只报数,不判定)", False),
           seg(2023, 2026, "**留出段 2023-01–2026-04(判据在这里)**", True)]

    print(f"\n{'='*88}\nG3 逐年(描述)\n{'='*88}")
    print(f"{'年':<8}{'事件':>8}{'基准':>9}{'选中':>8}{'启动率':>9}{'lift':>7}")
    for yy in range(2019, 2027):
        s = ev[ev.year == yy]
        if len(s) < 200:
            continue
        b = s.y.mean()
        m = s.sel.to_numpy()
        hr = s.y.to_numpy()[m].mean() if m.sum() else np.nan
        print(f"{yy:<8}{len(s):>8,}{b:>9.2%}{int(m.sum()):>8,}{hr:>9.2%}"
              f"{hr/b if b > 0 else np.nan:>7.2f}")
        res.append({"段": str(yy), "事件": len(s), "基准": float(b),
                    "选中": int(m.sum()), "启动率": float(hr),
                    "lift": float(hr / b) if b > 0 else np.nan})

    print(f"\n{'='*88}\nG4 留出段的同市值同行业对照(描述,不设门槛)\n{'='*88}")
    s = ev[(ev.year >= 2023) & ev.sel].reset_index(drop=True)
    tv, jv = s.t.to_numpy(), s.j.to_numpy()
    lab = np.zeros((nt, ns), bool)
    ally = ev[ev.year >= 2023]
    lab[ally.t.to_numpy()[ally.y.to_numpy()], ally.j.to_numpy()[ally.y.to_numpy()]] = True
    val = np.zeros((nt, ns), bool)
    val[ally.t.to_numpy(), ally.j.to_numpy()] = True
    pre = {}
    for t in np.unique(tv):
        e = np.flatnonzero(ok[t] & np.isfinite(mv[t]) & (ind[t] >= 0) & val[t])
        o = e[np.argsort(mv[t, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        pre[t] = (o, rk)
    ch, off, lens, kp = [], np.zeros(len(s), np.int64), np.zeros(len(s), np.int64), \
        np.ones(len(s), bool)
    pos = 0
    for k in range(len(s)):
        t, j = int(tv[k]), int(jv[k])
        o, rk = pre[t]
        p0, i0 = rk[j], ind[t, j]
        if p0 < 0 or i0 < 0:
            kp[k] = False
            continue
        a_, b_ = max(0, p0 - NBR), min(len(o) - 1, p0 + NBR)
        cand = o[a_:b_ + 1]
        cand = cand[ind[t, cand] == i0]
        if len(cand) < 2:
            cand = o[ind[t, o] == i0]
        if len(cand) < 2:
            kp[k] = False
            continue
        off[k], lens[k] = pos, len(cand)
        pos += len(cand)
        ch.append(cand)
    flat = np.concatenate(ch).astype(np.int64)
    kk = np.flatnonzero(kp)
    rng = np.random.default_rng(SEED)
    cp_, viol = [], 0
    for _ in range(0, NSEED, 50):
        r = rng.random((50, len(kk)))
        pick = flat[off[kk][None, :] + (r * lens[kk][None, :]).astype(np.int64)]
        viol += int((ind[tv[kk], pick] != ind[tv[kk], jv[kk]][None, :]).sum())
        cp_.extend(lab[tv[kk], pick].mean(1))
    cmed = float(np.median(cp_))
    hr2 = ev[(ev.year >= 2023) & ev.sel].y.mean()
    print(f"  行业违例 {viol} {'✓' if viol == 0 else '✗'}")
    print(f"  选中启动率 {hr2:.2%};同市值同行业对照中位 {cmed:.2%};"
          f"**lift_ctrl {hr2/cmed:.2f}**")
    res.append({"段": "留出段·同市值同行业对照", "启动率": float(hr2),
                "对照": cmed, "lift": float(hr2 / cmed)})
    # ---- 当前名单(面板末日)----
    tl = nt - 1
    m = ok[tl] & np.isfinite(rec[tl]) & np.isfinite(tacc[tl]) & np.isfinite(mv[tl])
    e = np.flatnonzero(m)
    qr = pd.Series(rec[tl, e]).rank(pct=True).to_numpy()
    qt = pd.Series(tacc[tl, e]).rank(pct=True).to_numpy()
    sel = e[(qr >= Q) & (qt >= Q)]
    print(f"\n{'='*88}\n当前名单({idx[tl].date()},全市场合格 {len(e):,})"
          f"\n{'='*88}")
    print(f"  规则选中 **{len(sel)} 只**(覆盖率 {len(sel)/len(e):.1%})")
    px2 = None
    try:
        xl = ("/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd/"
              "f48a5b4d-___20260827.xls")
        px2 = pd.read_excel(xl, dtype=str)
        px2 = px2.rename(columns={px2.columns[1]: "名称"})
        px2["代码"] = px2["代码"].str.zfill(6)
        nm2 = dict(zip(px2.代码, px2.名称, strict=True))
    except Exception:                                          # noqa: BLE001
        nm2 = {}
    cols_ = list(cldf.columns)
    inpool = [j for j in sel if cols_[j] in nm2]
    print(f"  其中落在用户 663 只池内的:**{len(inpool)} 只**")
    for j in sorted(inpool, key=lambda z: -tacc[tl, z]):
        c = cols_[j]
        print(f"    {c} {nm2.get(c,''):<10} 距低点{rec[tl,j]:>7.0%} "
              f"换手加速{tacc[tl,j]:>+7.0%} 流通市值{mv[tl,j]:>7.1f}亿")
    print(f"\n  全市场名单已落库(共 {len(sel)} 只)")
    pd.DataFrame([{"代码": cols_[j], "名称": nm2.get(cols_[j], ""),
                   "距低点": float(rec[tl, j]), "换手加速": float(tacc[tl, j]),
                   "流通市值亿": float(mv[tl, j]),
                   "在用户池内": cols_[j] in nm2} for j in sel]).to_csv(
        f"{OUT}/startup_rule_current.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res).to_csv(f"{OUT}/startup_rule_frozen.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n落库 {OUT}/startup_rule_frozen.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
