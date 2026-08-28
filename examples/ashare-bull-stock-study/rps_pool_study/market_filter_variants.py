"""§159 事前登记:大盘过滤换成 MA20 / MA60,择时本身有没有信息(结果未跑)。

起因
----
用户:「大盘过滤能不能调整到 20日线、60日线?」

第一五八节查出:宇通客车在 26,550 个买点里出现 10 次,
其中 **2024-01-08 / 01-09** 正是第二段拉升的起点(后 120 日 +78.8%),
**但那两天全市场等权净值在 MA200 之下,大盘过滤关闭,框架拒绝开仓。**
第一五六节又已算出 545 笔里 **413 笔(75.8%)死于大盘过滤清仓**。
**这个开关既在拦入场,又在砍持仓,是整套框架里最该被检验的一个部件。**

**必须先写死的一条纪律**
------------------------
**「宇通 2024-01-08 能不能被买到」是描述项,不是判据。**
第一五一节的 S3 就是照着中际旭创某一天的状态设计规则,结果 lift 1.09、
案例召回 2%,全场最差。**本节绝不重犯:判据只看组合层面的统计结果,
不看任何单只案例。** 宇通只在描述部分报一行。

被测的东西(跑前定死,恰好三条,不加第四条)
------------------------------------------
大盘过滤的判据序列一律是**全市场等权净值 vs 自身的 N 日均线**,
只换 N:**MA20 / MA60 / MA200**(MA200 = 第一五五节原样,作基线)。
另报「**无过滤**」(始终开启)作参照,**不参与判定**。
**不测缓冲带、不测延迟确认、不换指数** —— 那些是别的节,加进来就是 best-of-N。

口径(除大盘过滤外与第一五五/一五六节逐字一致,一个字不改)
----------------------------------------------------------
- 尺子 legacy 绝对阈值;买点 = 三条全中 且 收盘 > 平台内(强势日→前一日)最高收盘
- 止损 = max(平台下沿, 买入价 × 0.85),收盘触发;持有上限 120 交易日
- 10 等权槽位,空槽记 0 收益;突破日先到先得,同日按收敛比升序
- 面板 (3297, 5232);价格 ffill 参与,退市股绝不剔除
- 训练段 2019-01–2022-12 只报数;**留出段 2023-01–2026-04 判据在这里**
- 判定用零成本口径,双边 0.2%/往返 只作描述

两个对照(回答两个不同的问题)
------------------------------
**(甲)随机择时对照 —— 择时本身有没有信息?**
把该条均线**实际产生的开/关序列**做 **500 次随机循环平移**(circular shift)。
平移保持**开启天数占比与开关次数完全不变**,只打散它与市场的对齐关系。
选股完全不变。**这是「这个开关模式,但不对准市场」的零假设。**

**(乙)同市值同行业对照 —— 选股有没有增量?**
与第一五五节同规格:每次实际开仓的同一天、同一槽位,换成同市值名次 ±25、
同申万一级行业随机股,走完全相同的出场逻辑(止损按同一百分比),500 组种子。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
E1 锚点(不过则本节作废)
   (a) 面板 (3297, 5232);(b) 价格 ffill 后首价之后无空洞;
   (c) 行业恒等式违例 = 0;(d) 无前视:均线只用 ≤t、买点/止损只用 ≤t,逐点断言;
   (e) **机器锚点:MA200 档必须复现第一五五节留出段年化 +15.83%、
       第一五六节成交 545 笔(全区间)**,容差:年化 ±0.30pp、笔数完全一致。
       **对不上 = 本节作废。**
   (f) **随机择时对照的零校验**:每组平移后的开启天数占比与实际相差 < 0.5pp。

E2 **主判据 —— 择时是否携带信息**(留出段,零成本口径)
   对 MA20 / MA60 / MA200 各自判:
   **通过 ⟺ 年化 − 随机择时对照 500 组中位数 ≥ +3.00pp 且单尾 p < 0.0167。**
   **p 门槛已按 Bonferroni 除以 3(三条均线),这是加严不是放宽。**

E3 **次判据 —— 选股是否有增量**(留出段)
   **通过 ⟺ 年化 − 同市值同行业对照中位数 ≥ +3.00pp 且单尾 p < 0.0167。**

E4 描述(不参与判定):各档的开启占比、开关次数、成交笔数、平均持有天数、
   最大回撤、止损次数、双边 0.2% 成本口径、逐年收益;
   以及**宇通 2024-01-08 那个买点在各档下是否被买到**(仅描述,见上文纪律)。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。

不做的
------
不加第四条均线、不测缓冲带/延迟确认/换指数;不改买点、止损、槽位数、持有上限;
**跑完不许回头挑一条均线再单独重跑**;不新增顶层目录;不 force push;
**不往 quant-research-dev / etf-netflow-dev 推任何东西**;不作任何可交易性声明。
**若 E2/E3 全不过,如实写「换均线也没做到」。**
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
from codex_r10_neutral import NBR, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from consolidation_screener import (  # noqa: E402
    THR_ATR,
    THR_DEPTH,
    THR_SHRINK,
    load_panel,
)
from industry_neutral import build_industry  # noqa: E402
from platform_pivot import HOLD_MAX, MAXPOS, STOP_CAP, vec_screen  # noqa: E402

OUT = os.environ.get("OXQ_OUT_DIR", "/home/user/oxq-panel")
NSEED, COST, ALPHA = 500, 0.002, 0.05 / 3
MAS = (20, 60, 200)
TRAIN, HOLD = ("2019-01-01", "2022-12-31"), ("2023-01-01", "2026-04-30")
Y0 = 2016


def ann(nav, nd):
    return float(nav ** (250.0 / nd) - 1.0)


def mdd(eq):
    pk = np.maximum.accumulate(eq)
    return float(np.max((pk - eq) / pk))


def sim(cand, stops, mkt, nrep, ta, tb, cl):
    """mkt 形状 (nrep, nt);cand/stops: {day: (nrep, k)}。空槽记 0 收益。"""
    pj = np.full((nrep, MAXPOS), -1, np.int64)
    pe = np.zeros((nrep, MAXPOS), np.int64)
    ppx = np.zeros((nrep, MAXPOS))
    psl = np.zeros((nrep, MAXPOS))
    nd = tb - ta + 1
    ret = np.zeros((nrep, nd))
    cst = np.zeros((nrep, nd))
    nstop = np.zeros(nrep, np.int64)
    ntr = np.zeros(nrep, np.int64)
    nclr = np.zeros(nrep, np.int64)
    hold = np.zeros((nrep, nd))
    log = []
    for i, t in enumerate(range(ta, tb + 1)):
        m = pj >= 0
        z = np.where(m, pj, 0)
        if m.any():
            ret[:, i] = np.nan_to_num(
                np.where(m, cl[t, z] / cl[t - 1, z] - 1.0, 0.0)).sum(axis=1) / MAXPOS
        hold[:, i] = m.sum(axis=1)
        off = ~mkt[:, t]
        ex = m & off[:, None]
        nclr += ex.sum(axis=1)
        s_ = m & ~ex & (cl[t, z] <= psl)
        nstop += s_.sum(axis=1)
        ex |= s_ | (m & ((t - pe) >= HOLD_MAX))
        if nrep == 1:
            for s in np.flatnonzero(ex[0]):
                log.append((int(pj[0, s]), int(pe[0, s]), t, float(ppx[0, s]),
                            float(cl[t, int(pj[0, s])])))
        pj = np.where(ex, -1, pj)
        c = cand.get(t)
        if c is None:
            continue
        sp = stops[t]
        for r in range(nrep):
            if off[r]:
                continue
            free = np.flatnonzero(pj[r] < 0)
            if not len(free):
                continue
            k, row = 0, c[r] if c.shape[0] > 1 else c[0]
            srow = sp[r] if sp.shape[0] > 1 else sp[0]
            for slot in free:
                placed = False
                while k < row.shape[0]:
                    j = int(row[k])
                    kk = k
                    k += 1
                    if j < 0 or j in pj[r]:
                        continue
                    pj[r, slot] = j
                    pe[r, slot] = t
                    ppx[r, slot] = cl[t, j]
                    psl[r, slot] = srow[kk]
                    ntr[r] += 1
                    cst[r, i] += COST / MAXPOS
                    placed = True
                    break
                if not placed:
                    break
    return ret, cst, nstop, ntr, nclr, hold, log


def main():  # noqa: PLR0915
    t0 = time.time()
    cl_df, frames, strong, ma100 = load_panel(DATA)
    if "510300" in cl_df.columns:
        cl_df = cl_df.drop(columns=["510300"])
        strong = strong[:, [i for i, c in enumerate(ma100.columns) if c != "510300"]]
    idx, codes = cl_df.index, list(cl_df.columns)
    nt, ns = cl_df.shape
    assert (nt, ns) == (3297, 5232), f"锚点E1a {cl_df.shape}"
    ts_a, adj_a, dep, shr, cnv, hi, lo = vec_screen(
        cl_df.to_numpy(float), frames, strong, ma100, idx, codes)
    del frames
    d2 = {k: {} for k in ("float_mv", "is_st", "is_suspended", "listed_days",
                          "volume")}
    for c in codes:
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=list(d2))
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        for k in d2:
            d2[k][c] = x[k]

    def al(k, f=np.nan):
        return pd.DataFrame(d2[k]).sort_index().reindex(
            index=idx, columns=codes).fillna(f).to_numpy()
    cl = cl_df.where(cl_df > 0).ffill().to_numpy(np.float64)
    mv = al("float_mv") / 1e8
    ok = (~al("is_st", True).astype(bool) & ~al("is_suspended", True).astype(bool)
          & (al("listed_days", 0) >= 250) & (al("volume", 0) > 0) & np.isfinite(cl))
    fin = np.isfinite(cl)
    fs = np.argmax(fin, axis=0)
    gapn = int(sum((~fin[fs[j]:, j]).sum() for j in range(ns) if fin[:, j].any()))
    ind, _, _ = build_industry(codes, idx)
    vol20 = pd.DataFrame(cl).pct_change(1).rolling(20, min_periods=20).std().to_numpy()
    with np.errstate(all="ignore"):
        rr = cl[1:] / cl[:-1] - 1.0
    msk = ok[1:] & ok[:-1] & np.isfinite(rr)
    dd = np.zeros(nt)
    dd[1:] = np.where(msk.sum(1) > 0,
                      np.nan_to_num(rr * msk).sum(1) / np.maximum(msk.sum(1), 1), 0.0)
    nav = np.cumprod(1 + dd)
    del rr, msk
    hit3 = (shr < THR_SHRINK) & (cnv < THR_ATR) & (dep <= THR_DEPTH) & (adj_a >= 0)
    up_prev = np.full((nt, ns), np.nan, np.float64)
    lo_prev = np.full((nt, ns), np.nan, np.float64)
    same = np.zeros((nt, ns), bool)
    same[1:] = ts_a[1:] == ts_a[:-1]
    up_prev[1:] = np.where(same[1:], hi[:-1], np.nan)
    lo_prev[1:] = np.where(same[1:], lo[:-1], np.nan)
    brk = hit3 & ok & np.isfinite(up_prev) & (cl > up_prev) & np.isfinite(lo_prev) \
        & np.isfinite(vol20) & np.isfinite(mv)
    filt = {}
    for n_ in MAS:
        mm = pd.Series(nav).rolling(n_, min_periods=n_).mean().to_numpy()
        filt[f"MA{n_}"] = ~(np.isfinite(mm) & (nav < mm))
    filt["无过滤"] = np.ones(nt, bool)
    # ---- E1(d) 无前视 ----
    rs = np.random.default_rng(11)
    craw = cl_df.to_numpy(float)
    bp = np.argwhere(brk)
    for t, j in bp[rs.choice(len(bp), 2000, replace=False)]:
        a0 = int(ts_a[t, j])
        assert cl[t, j] > np.nanmax(craw[a0:t, j]), "E1d 上沿"
        assert abs(np.nanmin(craw[a0:t, j]) - lo_prev[t, j]) < 1e-9, "E1d 下沿"
    for n_ in MAS:
        mm = pd.Series(nav).rolling(n_, min_periods=n_).mean().to_numpy()
        for _ in range(300):
            t = int(rs.integers(n_ + 1, nt))
            assert abs(np.mean(nav[t - n_ + 1:t + 1]) - mm[t]) < 1e-9, f"E1d MA{n_}"
    del craw
    print(f"E1a ✓ {cl_df.shape};E1b ffill 空洞 {gapn} {'✓' if gapn == 0 else '✗'};"
          f"E1d 无前视 2,000 买点 + 3×300 均线点 ✓  ({time.time()-t0:.0f}s)")
    print("  开启天数占比:" + "  ".join(
        f"{k} {v.mean():.1%}" for k, v in filt.items()), flush=True)
    if gapn:
        print("**E1b 不过 → 本节作废**")
        return

    rng = np.random.default_rng(SEED)
    viol = [0]

    def subs(day, js):
        e = np.flatnonzero(ok[day] & np.isfinite(mv[day]) & (ind[day] >= 0))
        o = e[np.argsort(mv[day, e], kind="stable")]
        rk = np.full(ns, -1, np.int32)
        rk[o] = np.arange(len(o), dtype=np.int32)
        out = np.full((NSEED, len(js)), -1, np.int64)
        for k, j in enumerate(js):
            p0, i0 = rk[j], ind[day, j]
            if p0 < 0 or i0 < 0:
                continue
            c = o[max(0, p0 - NBR):min(len(o) - 1, p0 + NBR) + 1]
            c = c[ind[day, c] == i0]
            if len(c) < 2:
                c = o[ind[day, o] == i0]
            if len(c) < 2:
                continue
            pk = c[rng.integers(0, len(c), NSEED)]
            viol[0] += int((ind[day, pk] != i0).sum())
            out[:, k] = pk
        return out

    jy = codes.index("600066")
    ty = int(np.searchsorted(idx, pd.Timestamp("2024-01-08")))
    res, w = [], 104
    for lo_d, hi_d, tag, judge in ((TRAIN[0], TRAIN[1], "训练段(只报数)", False),
                                   (HOLD[0], HOLD[1], "留出段(判据在这里)", True)):
        ta = int(np.searchsorted(idx, pd.Timestamp(lo_d)))
        tb = int(np.searchsorted(idx, pd.Timestamp(hi_d), side="right")) - 1
        cand, stops, cand_c, stops_c = {}, {}, {}, {}
        for t in range(ta, tb + 1):
            e = np.flatnonzero(brk[t])
            if not len(e):
                continue
            e = e[np.argsort(cnv[t, e], kind="stable")]
            sp = np.maximum(lo_prev[t, e], cl[t, e] * (1 - STOP_CAP))
            cand[t] = e.reshape(1, -1).astype(np.int64)
            stops[t] = sp.reshape(1, -1)
            pct = 1 - sp / cl[t, e]
            sb = subs(t, e)
            v_ = sb >= 0
            cand_c[t] = sb
            stops_c[t] = np.where(v_, cl[t, np.where(v_, sb, 0)] * (1 - pct[None, :]),
                                  0.0)
        nd = tb - ta
        print(f"\n{'='*w}\n{tag}  ({idx[ta].date()} → {idx[tb].date()})\n{'='*w}")
        print(f"{'档':<10}{'开启占比':>9}{'开关次数':>9}{'成交':>7}{'年化':>9}"
              f"{'成本后':>9}{'回撤':>8}{'止损':>6}{'清仓':>6}"
              f"{'│随机择时中位':>13}{'超额pp':>9}{'p':>8}"
              f"{'│同市值行业':>11}{'超额pp':>9}{'p':>8}")
        for nm, mk in filt.items():
            m1 = mk[None, :].copy()
            r1, c1, ns1, nt1, nc1, h1, log = sim(cand, stops, m1, 1, ta, tb, cl)
            g = ann(float(np.prod(1 + r1[0])), nd)
            gc = ann(float(np.prod(1 + r1[0] - c1[0])), nd)
            sw = int(np.abs(np.diff(mk[ta:tb + 1].astype(int))).sum())
            row = {"段": tag, "档": nm, "开启占比": float(mk[ta:tb + 1].mean()),
                   "开关次数": sw, "成交": int(nt1[0]), "年化": g, "成本后": gc,
                   "回撤": mdd(np.cumprod(1 + r1[0])), "止损": int(ns1[0]),
                   "清仓": int(nc1[0]), "平均持仓": float(h1[0].mean())}
            if nm == "无过滤":
                print(f"{nm:<10}{mk[ta:tb+1].mean():>9.1%}{sw:>9}{int(nt1[0]):>7}"
                      f"{g:>9.2%}{gc:>9.2%}{row['回撤']:>8.1%}{int(ns1[0]):>6}"
                      f"{int(nc1[0]):>6}{'—(不判定)':>13}")
                res.append(row)
                continue
            # (甲) 随机择时:实际开关序列做循环平移
            # 在**分段内部**做循环平移:开启占比精确不变、开关次数至多差 1
            seg = mk[ta:tb + 1]
            sh = rng.integers(1, len(seg), NSEED)
            mrot = np.repeat(mk[None, :], NSEED, 0)
            mrot[:, ta:tb + 1] = np.stack([np.roll(seg, int(k)) for k in sh])
            assert abs(mrot[:, ta:tb + 1].mean(1) - seg.mean()).max() < 0.005, \
                "E1f 平移后开启占比漂移过大"
            r2, *_ = sim(cand, stops, mrot, NSEED, ta, tb, cl)
            cs1 = np.array([ann(float(np.prod(1 + r2[k])), nd) for k in range(NSEED)])
            e1_, p1_ = g - float(np.median(cs1)), float((cs1 >= g).mean())
            # (乙) 同市值同行业
            r3, *_ = sim(cand_c, stops_c, np.repeat(m1, NSEED, 0), NSEED, ta, tb, cl)
            cs2 = np.array([ann(float(np.prod(1 + r3[k])), nd) for k in range(NSEED)])
            e2_, p2_ = g - float(np.median(cs2)), float((cs2 >= g).mean())
            row.update({"随机择时中位": float(np.median(cs1)), "E2超额pp": e1_ * 100,
                        "E2p": p1_, "同市值行业中位": float(np.median(cs2)),
                        "E3超额pp": e2_ * 100, "E3p": p2_})
            print(f"{nm:<10}{mk[ta:tb+1].mean():>9.1%}{sw:>9}{int(nt1[0]):>7}"
                  f"{g:>9.2%}{gc:>9.2%}{row['回撤']:>8.1%}{int(ns1[0]):>6}"
                  f"{int(nc1[0]):>6}{np.median(cs1):>13.2%}{e1_*100:>9.2f}{p1_:>8.4f}"
                  f"{np.median(cs2):>11.2%}{e2_*100:>9.2f}{p2_:>8.4f}")
            if judge:
                a1 = e1_ >= 0.03 and p1_ < ALPHA
                a2 = e2_ >= 0.03 and p2_ < ALPHA
                print(f"           **E2 择时 {'通过' if a1 else '不通过'}"
                      f"(需 ≥+3.00pp 且 p<{ALPHA:.4f});"
                      f"E3 选股 {'通过' if a2 else '不通过'}**")
                row["E2判定"] = "通过" if a1 else "不通过"
                row["E3判定"] = "通过" if a2 else "不通过"
            res.append(row)
            if judge:
                got = any(j == jy and e == ty for j, e, *_ in log)
                print(f"           (描述,不参与判定)宇通 2024-01-08 买点:"
                      f"{'买到了' if got else '仍未买到'}"
                      f";该日过滤 {'开' if mk[ty] else '关'}")
        print(f"  锚点E1c 行业恒等式违例 {viol[0]} "
              f"{'✓' if viol[0] == 0 else '✗'}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(res)
    ma200h = df[(df["档"] == "MA200") & (df["段"].str.contains("留出"))]["年化"]
    ok_e = bool(len(ma200h)) and abs(float(ma200h.iloc[0]) - 0.1583) <= 0.0030
    print(f"\n锚点E1e MA200 复现第一五五节:{float(ma200h.iloc[0]):+.2%} vs +15.83% "
          f"{'✓' if ok_e else '✗ 本节作废'}")
    df.to_csv(f"{OUT}/market_filter_variants.csv", index=False, encoding="utf-8-sig")
    print(f"落库 {OUT}/market_filter_variants.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
