"""§153 事前登记:复现 Codex「统一价格启动信号」并给它做样本外(结果未跑)。

起因
----
用户:「你看看 codex,好歹也给我整了一个初稿出来吧,而你什么成果都没有啊」

**这个批评成立,而且是第二次(第一四二节同样的批评)。**
Codex 交的是 `990e9df9-...20260826.xlsx`:662 只次新股池、统一信号 1/0/NA、
51 只信号1(其中 41 只强确认)、规则表、勾稽检查页 —— **一份可以直接看的初稿**。
我这边有第一四八节的规则和 CSV,但**没有打包成同样能用的东西**,
而且第一四九至一五二节全是否定结论。

**本节做两件事,缺一不可:**
(1) 把 Codex 的规则在我的 13 年全市场面板上**复现并做样本外检验** ——
    这是只有我这边能做的(他那份 662 只里 44 只取不到行情,且只有一个快照日);
(2) **交付一份同格式的 Excel 初稿**,但每条规则旁边印上它的样本外成绩。

Codex 的规则(逐字抄自他的「信号规则」页,不改)
--------------------------------------------
    标准确认:距低点反弹 ≥ 40% 且 近120日收益 ≥ 10% 且
              近120日站上MA20比例 ≥ 55% 且 RPS60 ≥ 80
    强确认  :以上不变,但 RPS60 ≥ 90
    RPS 比较范围:同期本地可交易沪深A股(不是池内排名)
    统一信号1 = 标准确认 或 强确认

口径(与第一四八/一五二节一致,一个字不改)
------------------------------------------
- 全市场 5,232 只;观察点 = **每月最后一个交易日**;区间 **2019-01 → 2026-04**
- 合格:非 ST、非停牌、上市满 250 日、当日有成交
- **启动 = 未来 60 个交易日涨幅 ≥ 50%**;同股 **60 日内不重复计事件**
- 退市股按最后有效价 ffill 参与,**绝不剔除**
- 组合口径:月末等权买入全部选中股,持有到下月末;
  对照 = 同日、同市值名次 ±25、同申万一级行业随机抽同样只数,500 组种子

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
A1 锚点(不过则本节的检验部分作废,交付部分仍出)
   (a) 面板 (3297, 5232);
   (b) 价格 ffill 后首个有效价之后无空洞;
   (c) 无前视:距低点反弹、120日收益、MA20持续度、RPS60 逐点重算断言;
   (d) 组合收益区间严格 (t, t+1];行业恒等式违例 = 0。

A2 **口径反解**(描述,不是判据):我的面板末日是 **2026-08-03**,
   Codex 的观察日是 **2026-08-26**,差 15 个交易日,**数值必然对不上**。
   因此**不设「复现 Codex 数值」的锚点** —— 那样的锚点从设计上就不可能过
   (第一三一节 H2 就是这么栽的,不再犯第二次)。
   改为**反解他的「距低点反弹」用的是哪个回看窗**:
   分别用 250日低点 与 上市以来低点 算,报与他 662 只数值的 Spearman 相关与中位差,
   **取更接近的那个作为本节复现口径,并把选择理由写进正文**。

B1 **概率口径,时间样本外(判据一)**
   训练段 2019-01–2022-12:只报数,不判定。
   **留出段 2023-01–2026-04:B1 通过 ⟺ lift > 1.20 且 lift > 留出段自己的
   噪音上界 95 分位**(打乱标签 200 次,单规则,不取 best-of-N)。两条同时满足。

B2 **组合口径(判据二,与第一五二节同规格)**
   **留出段 2023-01–2026-04:B2 通过 ⟺ 零成本口径年化超额 ≥ +3.00pp
   (对照 500 组中位数)且单尾 p < 0.05。**两条同时满足。

B3 描述(不参与判定):逐年 lift、同市值同行业 lift_ctrl、双边 0.2%/月成本口径、
   最大回撤、覆盖率;**以及「距低点反弹」的分布** ——
   用户已在第一四九节明确否掉「1 年内已涨 5 倍还选进来」的做法,
   本节必须把 Codex 名单的这一分布如实报出来。

B4 **并排对照**:同一套机器同时跑第一四八节的规则(距低点前30% 且 换手加速前30%),
   两条规则的 B1/B2 数字并排,**让用户自己看哪条更值得用**。

事前预测
--------
**本节不下预测**(第一一九节起的约定)。

不做的
------
不改 Codex 的规则、不调他的阈值、不给他补条件 —— **原样复现,原样判**;
不因为 B1/B2 不过就修改规则再跑;不新增顶层目录;不 force push;
**不往 quant-research-dev / etf-netflow-dev 推任何东西**;
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
UP = "/root/.claude/uploads/e2d9b05a-8247-5772-8b9d-397e7f62f9fd"
XL = f"{UP}/990e9df9-____________20260826.xlsx"
HOR, THR, GAP, N_PERM, NSEED = 60, 0.50, 60, 200, 500
Q148, COST = 0.70, 0.002
TRAIN, HOLD = ("2019-01-01", "2022-12-31"), ("2023-01-01", "2026-04-30")


def ann(fac, nd):
    return float(np.prod(fac) ** (250.0 / nd) - 1.0)


def mdd(eq):
    pk = np.maximum.accumulate(eq)
    return float(np.max((pk - eq) / pk))


def load():
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
    return cldf, d


def main():  # noqa: PLR0915
    t0 = time.time()
    cldf, d = load()
    idx = cldf.index
    nt, ns = cldf.shape
    assert (nt, ns) == (3297, 5232), f"锚点A1a {cldf.shape}"

    def al(k, f=np.nan):
        return pd.DataFrame(d[k]).sort_index().reindex(
            index=idx, columns=cldf.columns).fillna(f)
    mv = al("float_mv").to_numpy() / 1e8
    trn = al("turnover")
    st = al("is_st", True).astype(bool).to_numpy()
    sus = al("is_suspended", True).astype(bool).to_numpy()
    vol = al("volume", 0).to_numpy()
    ld = al("listed_days", 0).to_numpy()
    cl = cldf.where(cldf > 0).ffill().to_numpy(np.float64)     # 用户规则5
    ok = ~st & ~sus & (ld >= 250) & (vol > 0) & np.isfinite(cl)
    fin = np.isfinite(cl)
    first = np.argmax(fin, axis=0)
    gapn = int(sum((~fin[first[j]:, j]).sum() for j in range(ns) if fin[:, j].any()))
    ind, ind_names, _ = build_industry(list(cldf.columns), idx)
    px = pd.DataFrame(cl)

    # ---- Codex 的五个字段 ----
    lo250 = px.rolling(250, min_periods=250).min().to_numpy()
    loall = px.cummin().to_numpy()
    ma20 = px.rolling(20, min_periods=20).mean().to_numpy()
    with np.errstate(all="ignore"):
        rec250 = cl / np.where(lo250 > 0, lo250, np.nan) - 1.0
        recall = cl / np.where(loall > 0, loall, np.nan) - 1.0
    above = pd.DataFrame((cl > ma20).astype(np.float64)).where(np.isfinite(ma20))
    mfrac = above.rolling(120, min_periods=120).mean().to_numpy()
    del above, lo250, loall
    r60 = px.pct_change(60).to_numpy()
    r120 = px.pct_change(120).to_numpy()
    trad = ~sus & (vol > 0) & np.isfinite(r60)                 # 同期本地可交易A股
    rr = np.where(trad, r60, np.nan)
    rps60 = pd.DataFrame(rr).rank(axis=1, pct=True).to_numpy() * 100.0
    rps250 = pd.DataFrame(np.where(trad, px.pct_change(250).to_numpy(),
                                   np.nan)).rank(axis=1, pct=True).to_numpy() * 100.0
    del rr
    # ---- 第一四八节的两个字段(B4 并排用)----
    t20 = trn.rolling(20, min_periods=10).mean().to_numpy()
    t60 = trn.rolling(60, min_periods=30).mean().to_numpy()
    with np.errstate(all="ignore"):
        tacc = t20 / np.where(t60 > 0, t60, np.nan) - 1.0
    del t20, t60
    # ---- 前瞻标签 ----
    fmax = pd.DataFrame(cl[::-1]).rolling(HOR, min_periods=1).max().to_numpy()[::-1]
    fwd = np.full_like(cl, np.nan)
    fwd[:-1] = fmax[1:]
    with np.errstate(all="ignore"):
        up = fwd / np.where(cl > 0, cl, np.nan) - 1.0
    del fmax, fwd

    # ---- A1(c) 无前视逐点重算 ----
    trnv = trn.to_numpy()
    rs = np.random.default_rng(13)
    n = [0, 0, 0, 0]
    for _ in range(3000):
        t = int(rs.integers(260, nt - HOR - 1))
        j = int(rs.integers(0, ns))
        if np.isfinite(rec250[t, j]):
            assert abs(cl[t, j] / np.nanmin(cl[t - 249:t + 1, j]) - 1
                       - rec250[t, j]) < 1e-9, "A1c 距低点"
            n[0] += 1
        if np.isfinite(r120[t, j]):
            assert abs(cl[t, j] / cl[t - 120, j] - 1 - r120[t, j]) < 1e-9, "A1c r120"
            n[1] += 1
        if np.isfinite(mfrac[t, j]):
            # 分两步断言:MA20 本身用容差(rolling 的累加顺序与重算在末位会差 1 ulp,
            # 边界上 close>MA20 会翻一天,恰好 1/120 —— 这是断言设计问题,不是数据问题);
            # 持续度则用面板自己的 MA20 验「窗口只覆盖 t-119..t」,即无前视。
            assert abs(np.mean(cl[t - 19:t + 1, j]) - ma20[t, j]) < 1e-6, "A1c MA20"
            assert abs(np.mean(cl[t - 119:t + 1, j] > ma20[t - 119:t + 1, j])
                       - mfrac[t, j]) < 1e-9, "A1c MA20持续度窗口"
            n[2] += 1
        if np.isfinite(up[t, j]):
            assert abs(np.nanmax(cl[t + 1:t + 1 + HOR, j]) / cl[t, j] - 1
                       - up[t, j]) < 1e-9, "A1c 前瞻窗口"
            n[3] += 1
        if np.isfinite(tacc[t, j]) and np.isfinite(trnv[t, j]):
            a, b = trnv[t - 19:t + 1, j], trnv[t - 59:t + 1, j]
            if np.isfinite(a).sum() >= 10 and np.isfinite(b).sum() >= 30:
                assert abs(np.nanmean(a) / np.nanmean(b) - 1 - tacc[t, j]) < 1e-9, \
                    "A1c 换手加速"
    print(f"锚点A1a ✓ 面板 {cldf.shape};A1b ffill 空洞 {gapn} "
          f"{'✓' if gapn == 0 else '✗'}")
    print(f"锚点A1c ✓ 无前视:距低点 {n[0]}、120日收益 {n[1]}、MA20持续度 {n[2]}、"
          f"前瞻 {n[3]} 点全部逐点重算一致({time.time()-t0:.0f}s)", flush=True)

    # ---- A2 反解 Codex 的「距低点反弹」回看窗(描述,不是判据)----
    cx = pd.read_excel(XL, sheet_name="股票池统一信号", dtype={"股票代码": str})
    cx["股票代码"] = cx["股票代码"].str.zfill(6)
    pos = {c: j for j, c in enumerate(cldf.columns)}
    cx["j"] = cx["股票代码"].map(pos)
    tl = nt - 1
    sub = cx[cx.j.notna() & cx["行情可用"].astype(bool)].copy()
    sub["j"] = sub["j"].astype(int)
    jj = sub.j.to_numpy()
    w = 92
    print(f"\n{'='*w}\nA2 反解 Codex 的「距低点反弹」回看窗(描述,不设判据)\n{'='*w}")
    print(f"  我的面板末日 {idx[tl].date()};Codex 观察日 2026-08-26 —— "
          f"差 15 个交易日,数值本就对不上,故不设复现锚点。")
    best = None
    for nm, arr in (("250日低点", rec250), ("上市以来低点", recall)):
        a = arr[tl, jj]
        b = sub["距低点反弹"].to_numpy(np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        sp = float(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))
        md = float(np.median(a[m] - b[m]))
        rel = float(np.median(np.abs(a[m] - b[m]) / (1 + b[m])))
        print(f"  {nm:<12} 可比 {int(m.sum()):>4} 只   Spearman {sp:.4f}   "
              f"中位差 {md:+.4f}   中位相对差 {rel:.2%}")
        if best is None or rel < best[1]:
            best = (nm, rel, arr)
    print(f"  → **本节复现口径取「{best[0]}」**(中位相对差更小)")
    rec = best[2]
    del recall

    # ---- 两条规则 ----
    codex_ok = ((rec >= 0.40) & (r120 >= 0.10) & (mfrac >= 0.55) & (rps60 >= 80))
    codex_strong = codex_ok & (rps60 >= 90)

    me = np.sort(pd.Series(np.arange(nt), index=idx).groupby(
        [idx.year, idx.month]).last().to_numpy().astype(int))
    rows = []
    for t in me:
        t = int(t)
        if t > nt - HOR - 1 or idx[t] < pd.Timestamp(TRAIN[0]) \
                or idx[t] > pd.Timestamp(HOLD[1]):
            continue
        m = ok[t] & np.isfinite(up[t]) & np.isfinite(mv[t])
        e = np.flatnonzero(m)
        if len(e) < 100:
            continue
        c1 = codex_ok[t, e] & np.isfinite(rec[t, e])
        c2 = codex_strong[t, e] & np.isfinite(rec[t, e])
        v3 = np.isfinite(rec[t, e]) & np.isfinite(tacc[t, e])
        qr = pd.Series(np.where(v3, rec[t, e], np.nan)).rank(pct=True).to_numpy()
        qt = pd.Series(np.where(v3, tacc[t, e], np.nan)).rank(pct=True).to_numpy()
        c3 = v3 & (qr >= Q148) & (qt >= Q148)
        for k, j in enumerate(e):
            rows.append((t, idx[t].year, int(j), bool(c1[k]), bool(c2[k]),
                         bool(c3[k]), bool(up[t, j] >= THR), float(rec[t, j])))
    ev = pd.DataFrame(rows, columns=["t", "year", "j", "codex", "strong",
                                     "r148", "y", "rec"]).sort_values(["j", "t"])
    keep, last = [], {}
    for r in ev.itertuples():
        if r.t - last.get(r.j, -10**9) >= GAP:
            keep.append(True)
            last[r.j] = r.t
        else:
            keep.append(False)
    ev = ev[keep].reset_index(drop=True)
    print(f"\n月度事件 {len(ev):,}({idx[ev.t.min()].date()} → "
          f"{idx[ev.t.max()].date()})  ({time.time()-t0:.0f}s)", flush=True)

    # ---- B1 概率口径,时间样本外 ----
    res = []
    for col, nm in (("codex", "Codex 统一信号1(标准+强确认)"),
                    ("strong", "Codex 强确认(RPS60≥90)"),
                    ("r148", "第一四八节规则(距低点前30% 且 换手加速前30%)")):
        print(f"\n{'='*w}\nB1 {nm}\n{'='*w}")
        for lo, hi, tag, judge in ((TRAIN[0], TRAIN[1], "训练段 2019–2022(只报数)",
                                    False),
                                   (HOLD[0], HOLD[1],
                                    "**留出段 2023-01–2026-04(判据在这里)**", True)):
            s = ev[(ev.year >= pd.Timestamp(lo).year)
                   & (ev.year <= pd.Timestamp(hi).year)]
            b = s.y.mean()
            m = s[col].to_numpy()
            y = s.y.to_numpy()
            if m.sum() < 30:
                print(f"  {tag}:选中 {int(m.sum())} 只次,不足 30,跳过")
                continue
            hr = y[m].mean()
            lf = hr / b
            rg = np.random.default_rng(SEED)
            perm = [rg.permutation(y)[m].mean() / b for _ in range(N_PERM)]
            h95 = float(np.percentile(perm, 95))
            print(f"  {tag}:事件 {len(s):,},基准 {b:.2%};选中 {int(m.sum()):,} 只次"
                  f"(覆盖 {m.mean():.1%}),启动率 **{hr:.2%}**,**lift {lf:.2f}**,"
                  f"召回 {y[m].sum()/y.sum():.1%};噪音上界95 {h95:.2f}")
            if judge:
                a1, a2 = lf > 1.20, lf > h95
                print(f"    **B1 判定**:lift>1.20 {'✓' if a1 else '✗'};"
                      f"lift>噪音上界 {'✓' if a2 else '✗'} → "
                      f"**{'通过' if (a1 and a2) else '不通过'}**")
            res.append({"规则": nm, "段": tag, "口径": "概率", "事件": len(s),
                        "基准": float(b), "选中": int(m.sum()),
                        "覆盖率": float(m.mean()), "启动率": float(hr),
                        "lift": float(lf), "噪音上界95": h95,
                        "召回": float(y[m].sum() / y.sum())})
        print("  逐年 lift:", end="")
        for yy in range(2019, 2027):
            s = ev[ev.year == yy]
            m = s[col].to_numpy()
            if len(s) < 200 or m.sum() < 5:
                continue
            b = s.y.mean()
            print(f"  {yy} {s.y.to_numpy()[m].mean()/b:.2f}", end="")
        print(flush=True)

    # ---- B3 距低点反弹分布(用户在第一四九节否掉过「已涨几倍还选」)----
    print(f"\n{'='*w}\nB3 选中股的「距低点反弹」分布(留出段;用户第一四九节的关切)\n{'='*w}")
    h = ev[ev.year >= 2023]
    print(f"{'规则':<44}{'中位':>9}{'75分位':>9}{'90分位':>9}{'>200%占比':>11}")
    for col, nm in (("codex", "Codex 统一信号1"), ("strong", "Codex 强确认"),
                    ("r148", "第一四八节规则")):
        a = h.loc[h[col].to_numpy(), "rec"].to_numpy()
        print(f"{nm:<44}{np.median(a):>9.0%}{np.percentile(a,75):>9.0%}"
              f"{np.percentile(a,90):>9.0%}{(a > 2.0).mean():>11.1%}")
        res.append({"规则": nm, "段": "留出段·距低点分布", "口径": "描述",
                    "中位": float(np.median(a)),
                    "p75": float(np.percentile(a, 75)),
                    "p90": float(np.percentile(a, 90)),
                    "超200%占比": float((a > 2.0).mean())})

    # ---- B2 组合口径(与第一五二节同规格)----
    print(f"\n{'='*w}\nB2 组合口径:月末等权买入,持有到下月末,"
          f"对照 = 同市值名次±25 同申万一级,{NSEED} 组\n{'='*w}")
    sel_rules = {"codex": [], "strong": [], "r148": []}
    months, univ = [], []
    for a, b in zip(me[:-1], me[1:], strict=True):
        a, b = int(a), int(b)
        if idx[a] < pd.Timestamp(TRAIN[0]) or idx[a] > pd.Timestamp(HOLD[1]):
            continue
        m = ok[a] & np.isfinite(mv[a])
        e = np.flatnonzero(m)
        if len(e) < 100:
            continue
        v3 = np.isfinite(rec[a, e]) & np.isfinite(tacc[a, e])
        qr = pd.Series(np.where(v3, rec[a, e], np.nan)).rank(pct=True).to_numpy()
        qt = pd.Series(np.where(v3, tacc[a, e], np.nan)).rank(pct=True).to_numpy()
        pick = {"codex": e[codex_ok[a, e] & np.isfinite(rec[a, e])],
                "strong": e[codex_strong[a, e] & np.isfinite(rec[a, e])],
                "r148": e[v3 & (qr >= Q148) & (qt >= Q148)]}
        months.append((a, b))
        univ.append(e)
        for k in sel_rules:
            sel_rules[k].append(pick[k])
    ent = np.array([idx[a] for a, _ in months])
    print(f"  调仓月 {len(months)} 个({ent[0].date()} → {idx[months[-1][1]].date()})")

    viol_all = 0
    for col, nm in (("codex", "Codex 统一信号1"), ("strong", "Codex 强确认"),
                    ("r148", "第一四八节规则")):
        sels = sel_rules[col]
        rng = np.random.default_rng(SEED)
        fac = np.ones(len(months))
        cfac = np.ones((NSEED, len(months)))
        nsel = np.zeros(len(months), int)
        for i, ((a, b), s, e) in enumerate(zip(months, sels, univ, strict=True)):
            nsel[i] = len(s)
            if len(s) < 3:                       # 空仓月:记 1.0(不持仓不赚不亏)
                continue
            fac[i] = float(np.mean(cl[b, s] / cl[a, s]))
            pool = e[ind[a, e] >= 0]
            o = pool[np.argsort(mv[a, pool], kind="stable")]
            rk = np.full(ns, -1, np.int32)
            rk[o] = np.arange(len(o), dtype=np.int32)
            flat, off, lens, use, p = [], [], [], [], 0
            for j in s:
                p0, i0 = rk[j], ind[a, j]
                if p0 < 0 or i0 < 0:
                    continue
                c = o[max(0, p0 - NBR):min(len(o) - 1, p0 + NBR) + 1]
                c = c[ind[a, c] == i0]
                if len(c) < 2:
                    c = o[ind[a, o] == i0]
                if len(c) < 2:
                    continue
                flat.append(c)
                off.append(p)
                lens.append(len(c))
                use.append(j)
                p += len(c)
            if not flat:
                cfac[:, i] = fac[i]
                continue
            flat = np.concatenate(flat).astype(np.int64)
            off, lens = np.asarray(off, np.int64), np.asarray(lens, np.int64)
            use = np.asarray(use, np.int64)
            g = rng.random((NSEED, len(use)))
            pk = flat[off[None, :] + (g * lens[None, :]).astype(np.int64)]
            viol_all += int((ind[a, pk] != ind[a, use][None, :]).sum())
            cfac[:, i] = (cl[b] / cl[a])[pk].mean(axis=1)
        print(f"\n  ── {nm} ── 每月选中中位 {int(np.median(nsel))} 只,"
              f"空仓月 {int((nsel < 3).sum())} 个")
        for lo, hi, tag, judge in ((TRAIN[0], TRAIN[1], "训练段(只报数)", False),
                                   (HOLD[0], HOLD[1], "**留出段(判据在这里)**",
                                    True)):
            mm = (ent >= pd.Timestamp(lo)) & (ent <= pd.Timestamp(hi))
            k = np.flatnonzero(mm)
            nd = int(months[k[-1]][1] - months[k[0]][0])
            g0 = ann(fac[mm], nd)
            gc = ann(fac[mm] * np.where(nsel[mm] >= 3, 1 - COST, 1.0), nd)
            cs = np.array([ann(cfac[z, mm], nd) for z in range(NSEED)])
            cmed = float(np.median(cs))
            exc, pv = g0 - cmed, float((cs >= g0).mean())
            eq = np.cumprod(fac[mm])
            print(f"    {tag}:零成本年化 **{g0:+.2%}**,双边0.2%/月 {gc:+.2%},"
                  f"对照中位 {cmed:+.2%},**超额 {exc*100:+.2f}pp,单尾 p {pv:.4f}**,"
                  f"回撤 {mdd(eq):.1%}")
            if judge:
                a1, a2 = exc >= 0.03, pv < 0.05
                print(f"      **B2 判定**:超额≥+3.00pp {'✓' if a1 else '✗'};"
                      f"p<0.05 {'✓' if a2 else '✗'} → "
                      f"**{'通过' if (a1 and a2) else '不通过'}**")
            res.append({"规则": nm, "段": tag, "口径": "组合", "零成本年化": g0,
                        "双边0.2%年化": gc, "对照年化中位": cmed,
                        "超额pp": exc * 100, "p": pv, "回撤": mdd(eq),
                        "每月中位只数": int(np.median(nsel[mm]))})
    print(f"\n  锚点A1d 行业恒等式违例 {viol_all} {'✓' if viol_all == 0 else '✗'}")
    pd.DataFrame(res).to_csv(f"{OUT}/codex_unified_signal.csv", index=False,
                             encoding="utf-8-sig")

    # ---- 当前名单落库(面板末日),供 Excel 初稿使用 ----
    nm_map = dict(zip(cx["股票代码"], cx["股票名称"], strict=True))
    innames = set(cx["股票代码"])
    m = ok[tl] & np.isfinite(rec[tl]) & np.isfinite(mv[tl])
    e = np.flatnonzero(m)
    v3 = np.isfinite(tacc[tl, e])
    qr = pd.Series(np.where(v3, rec[tl, e], np.nan)).rank(pct=True).to_numpy()
    qt = pd.Series(np.where(v3, tacc[tl, e], np.nan)).rank(pct=True).to_numpy()
    cols_ = list(cldf.columns)
    out = pd.DataFrame({
        "观察日期": idx[tl].date(), "股票代码": [cols_[j] for j in e],
        "股票名称": [nm_map.get(cols_[j], "") for j in e],
        "在次新股池内": [cols_[j] in innames for j in e],
        "Codex统一信号": codex_ok[tl, e].astype(int),
        "Codex强确认": codex_strong[tl, e].astype(int),
        "第一四八节信号": (v3 & (qr >= Q148) & (qt >= Q148)).astype(int),
        "两条都中": (codex_ok[tl, e] & v3 & (qr >= Q148) & (qt >= Q148)).astype(int),
        "距低点反弹": rec[tl, e], "近120日收益": r120[tl, e],
        "近60日收益": r60[tl, e], "MA20持续度": mfrac[tl, e],
        "RPS60": rps60[tl, e], "RPS250": rps250[tl, e],
        "换手加速": tacc[tl, e], "距低点分位": qr, "换手加速分位": qt,
        "流通市值亿": mv[tl, e],
        "申万一级": [None if ind[tl, j] < 0 else ind_names[ind[tl, j]]
                   for j in e]})
    out.to_csv(f"{OUT}/codex_unified_signal_current.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n当前名单({idx[tl].date()},合格 {len(e):,}):"
          f"Codex 信号1 {int(out['Codex统一信号'].sum())} 只、"
          f"强确认 {int(out['Codex强确认'].sum())} 只、"
          f"第一四八节 {int(out['第一四八节信号'].sum())} 只、"
          f"两条都中 {int(out['两条都中'].sum())} 只")
    sub2 = out[out["在次新股池内"]]
    print(f"  其中落在 Codex 的 662 只次新股池内:"
          f"合格 {len(sub2)} 只,Codex 信号1 {int(sub2['Codex统一信号'].sum())} 只、"
          f"强确认 {int(sub2['Codex强确认'].sum())} 只")
    print(f"\n落库 {OUT}/codex_unified_signal.csv、codex_unified_signal_current.csv"
          f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
