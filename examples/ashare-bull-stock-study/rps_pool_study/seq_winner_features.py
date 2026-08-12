"""三段各自的特征:什么把 13,731 个同形态事件里的赢家和输家分开

═══ 为什么这是本 session 最干净的一次对照 ═══
第五十八节把用户的形态(强势 → 深调20周线 → 买点)实现对了,验收通过
(宁德 +183.8%、胜宏 +665.8%、生益 +223.7% 全部抓到),
但同一形态触发 **13,731 次**,中位数 **-2.4%**、亏损 **52.4%**、翻倍仅 **6.5%**。

**形态被固定住了。** 剩下的差异就纯粹是「什么把 6.5% 的赢家和 52% 的输家分开」——
前面几十节做归因时,对照组总要费力构造;这一次对照组是天然的。

═══ 用户的三个问题,对应三段 ═══
① 第一段 RPS60→90+ 是什么推上去的?(肯定不是 RPS 自己)
② 第二段调整期能否量化?
③ 第三段最难,失败案例多,什么能提高胜率?

═══ 结果的两个口径(都报) ═══
  raw252  = 买点后 252 日的原始收益(用户看图时看到的那个)
  trade   = 规则A 实际交易结果(-10%止损、无止盈、252日)—— **决定能不能赚钱的是这个**

═══ 纪律(沿用第五十三节,不放宽) ═══
  A 每个特征**自己的零分布**(年内打乱标签 500 次)双侧 p < 0.05
  B lift > **公平 best-of-N 天花板**(只让命中≥300 的特征参与)
  C 2015-05 前后两段**方向一致**

═══ 最关键的设计:特征选择与验证分开 ═══
**特征选择只在 2013-2019 的事件上做,过关的特征拿到 2020-2026 上验证。**
前面几十节反复踩的坑是「在同一份数据上既选又验」。
这一次事前写死:
  - 选择集 = 买点日在 2019-12-31 之前的事件
  - 验证集 = 买点日在 2020-01-01 之后的事件(**选择时完全不看**)
  - 验证判据:组合级年化 ≥ **+7.22%**,且 **300 次**同选中率随机对照 p < 0.05
**验证集上不做任何调参。过不了就是过不了。**
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE, COST_PF = 0.003, 0.003
SLOTS, SEED, N_PERM, N_RAND = 10, 20260812, 500, 300
MIN_HITS_CEIL = 300
SPLIT = "2020-01-01"

t0 = time.time()
COLS = ["open", "high", "low", "close", "float_mv", "volume", "turnover",
        "bp_correct", "is_limit_up"]
d = {c: {} for c in COLS}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    try:
        x = pd.read_parquet(f, columns=COLS)
    except Exception:
        continue
    if x.empty:
        continue
    for c in COLS:
        d[c][k] = pd.to_numeric(x[c], errors="coerce")
CL = pd.DataFrame(d["close"]).sort_index(); CL.index = CL.index.tz_localize(None)


def al(k):
    fr = pd.DataFrame(d[k]).sort_index(); fr.index = fr.index.tz_localize(None)
    return fr.reindex(index=CL.index, columns=CL.columns)


OP, HI, LO = al("open"), al("high"), al("low")
MV, VO, TURN, BP, LU = al("float_mv"), al("volume"), al("turnover"), al("bp_correct"), al("is_limit_up")
CL = CL.where(CL > 0); OP = OP.where(OP > 0); HI = HI.where(HI > 0); LO = LO.where(LO > 0)
idx = CL.index
NT = len(idx)
A, OPa, HIa, LOa = CL.to_numpy(float), OP.to_numpy(float), HI.to_numpy(float), LO.to_numpy(float)
MVa, VOa, LUa = MV.to_numpy(float), VO.to_numpy(float), LU.to_numpy(float)
MA100a = CL.rolling(100, min_periods=100).mean().to_numpy(float)
codes = list(CL.columns)
col_of = {cd: i for i, cd in enumerate(codes)}
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
del d

RPS250 = (CL.pct_change(250).rank(axis=1, pct=True) * 100).to_numpy(float)
MV_PCT = MV.rank(axis=1, pct=True).to_numpy(float)
BP_PCT = BP.rank(axis=1, pct=True).to_numpy(float)
TURN_PCT = TURN.rolling(20, min_periods=10).mean().rank(axis=1, pct=True).to_numpy(float)
TR = np.maximum(HIa - LOa, np.maximum(np.abs(HIa - np.roll(A, 1, 0)),
                                      np.abs(LOa - np.roll(A, 1, 0))))
VOL50 = VO.rolling(50, min_periods=20).mean().to_numpy(float)
CQ = pd.read_parquet(f"{SP}/clean_growth_c_qyoy.parquet").reindex(
    index=idx, columns=CL.columns).to_numpy(float)
NI_TTM = pd.read_parquet(f"{SP}/clean_growth_ni_ttm_yoy.parquet").reindex(
    index=idx, columns=CL.columns).to_numpy(float)
RV_TTM = pd.read_parquet(f"{SP}/clean_growth_rev_ttm_yoy.parquet").reindex(
    index=idx, columns=CL.columns).to_numpy(float)
ACCEL = CQ - np.roll(CQ, 63, axis=0); ACCEL[:63] = np.nan
_mkt = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                     errors="coerce")
_mkt.index = _mkt.index.tz_localize(None)
mkt = _mkt.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()
BREADTH = (pd.DataFrame(RPS250) > 90).mean(axis=1).to_numpy()   # 全市场强势股占比
print(f"因子就绪  ({time.time()-t0:.0f}s)")

E = pd.read_csv(f"{SP}/seq_events_P1.csv", dtype={"code": str})
E = E[E.code.isin(col_of)].reset_index(drop=True)
print(f"事件 {len(E):,}(第五十八节 P1)")

# ── 逐事件算三段特征 ──
rows = []
for _, r in E.iterrows():
    j = col_of[r.code]
    ts, td, tb = int(r.t_strong), int(r.t_dip), int(r.dp)
    if tb + 252 >= NT or ts < 250:
        continue
    a = A[:, j]
    # 结果
    raw252 = a[tb + 252] / a[tb] - 1 if np.isfinite(a[tb]) and a[tb] > 0 else np.nan
    if not np.isfinite(raw252):
        continue
    # 交易结果(规则A)
    e = tb + 1
    entry = OPa[e, j]
    if not np.isfinite(entry) or entry <= 0:
        continue
    stop_px, last, ex = entry * 0.90, entry, None
    end = min(e + 252, NT - 1)
    for t in range(e, end + 1):
        if not np.isfinite(a[t]):
            continue
        last = a[t]
        if np.isfinite(LOa[t, j]) and LOa[t, j] <= stop_px:
            ex = OPa[t, j] if (np.isfinite(OPa[t, j]) and OPa[t, j] < stop_px) else stop_px
            break
    if ex is None:
        ex = a[end] if np.isfinite(a[end]) else last
    trade = ex / entry - 1

    def _m(arr, lo_, hi_):
        v = arr[lo_:hi_ + 1, j]
        v = v[np.isfinite(v)]
        return v.mean() if v.size else np.nan

    s_lo = max(ts - 60, 0)
    hi_s = HIa[ts - 60:ts + 1, j]; lo_s = LOa[ts - 60:ts + 1, j]
    seg_adj = slice(ts, tb + 1)
    hi_a = HIa[seg_adj, j]; lo_a = LOa[seg_adj, j]
    rows.append({
        "code": r.code, "dp": tb, "year": idx[tb].year, "date": idx[tb],
        "raw252": raw252, "trade": trade,
        # ── 第一段:强势期 ──
        "S_涨幅60": a[ts] / a[s_lo] - 1 if np.isfinite(a[s_lo]) and a[s_lo] > 0 else np.nan,
        "S_当季同比": CQ[ts, j],
        "S_盈利加速": ACCEL[ts, j],
        "S_双增长": 1.0 if (NI_TTM[ts, j] > 0 and RV_TTM[ts, j] > 0) else 0.0,
        "S_量能放大": (_m(VOa, ts - 60, ts) / _m(VOa, ts - 250, ts - 61)
                    if np.isfinite(_m(VOa, ts - 250, ts - 61)) and _m(VOa, ts - 250, ts - 61) > 0 else np.nan),
        "S_涨停次数": np.nansum(LUa[ts - 60:ts + 1, j]),
        "S_换手分位": TURN_PCT[ts, j],
        "S_市场广度": BREADTH[ts],
        # ── 第二段:调整期 ──
        "D_时长": tb - ts,
        "D_深度": (1 - np.nanmin(lo_a) / np.nanmax(hi_a)) if np.isfinite(np.nanmax(hi_a)) else np.nan,
        "D_缩量比": (_m(VOa, ts, tb) / _m(VOa, ts - 60, ts)
                  if np.isfinite(_m(VOa, ts - 60, ts)) and _m(VOa, ts - 60, ts) > 0 else np.nan),
        "D_波动收缩": (_m(TR, td, tb) / _m(TR, ts - 60, ts)
                   if np.isfinite(_m(TR, ts - 60, ts)) and _m(TR, ts - 60, ts) > 0 else np.nan),
        "D_破20周线天数占比": float(np.nanmean(A[seg_adj, j] < MA100a[seg_adj, j])),
        "D_触线到买点": tb - td,
        # ── 第三段:买点日 ──
        "B_量比": VOa[tb, j] / VOL50[tb, j] if np.isfinite(VOL50[tb, j]) and VOL50[tb, j] > 0 else np.nan,
        "B_RPS250": RPS250[tb, j],
        "B_距20周线": a[tb] / MA100a[tb, j] - 1 if np.isfinite(MA100a[tb, j]) and MA100a[tb, j] > 0 else np.nan,
        "B_大盘在MA200上": 1.0 if mkt_ok[tb] else 0.0,
        "B_市值分位": MV_PCT[tb, j],
        "B_BP分位": BP_PCT[tb, j],
    })
P = pd.DataFrame(rows)
P["win"] = P.raw252 > 1.0                      # 翻倍
P["win_trade"] = P.trade > 0                   # 交易赚钱
print(f"\n可用事件 {len(P):,}   翻倍率 **{P.win.mean():.2%}**   交易胜率 **{P.win_trade.mean():.2%}**"
      f"   ({time.time()-t0:.0f}s)")
print(f"  raw252 中位 {P.raw252.median():+.2%}   trade 净期望 {P.trade.mean()-COST_TRADE:+.2%}")

IN = P[P.date < SPLIT].reset_index(drop=True)
OUT = P[P.date >= SPLIT].reset_index(drop=True)
print(f"  **选择集(<{SPLIT}) {len(IN):,} 笔**,翻倍率 {IN.win.mean():.2%}")
print(f"  **验证集(≥{SPLIT}) {len(OUT):,} 笔**,翻倍率 {OUT.win.mean():.2%}  ← 选择时完全不看")

# ── 二元化(阈值取分布中位/常规值,事前定,不调) ──
def binarize(df):
    # df 里含字符串列 code,直接 df.quantile 会在 pyarrow 上报
    # ArrowNotImplementedError: 'quantile' has no kernel for large_string —— 只取数值列
    q = df.select_dtypes(include=[np.number]).quantile
    return {
        "① 强势期涨幅 高50%": df.S_涨幅60 >= q(.5).S_涨幅60,
        "① 当季同比 >25%": df.S_当季同比 > 0.25,
        "① 盈利加速 >0": df.S_盈利加速 > 0,
        "① 双增长": df.S_双增长 > 0,
        "① 强势期放量 >1.5×": df.S_量能放大 > 1.5,
        "① 强势期涨停 ≥3次": df.S_涨停次数 >= 3,
        "① 换手分位 高50%": df.S_换手分位 >= q(.5).S_换手分位,
        "① 市场广度 高50%": df.S_市场广度 >= q(.5).S_市场广度,
        "② 调整时长 长50%": df.D_时长 >= q(.5).D_时长,
        "② 调整深度 浅50%": df.D_深度 <= q(.5).D_深度,
        "② 调整期缩量 <0.8×": df.D_缩量比 < 0.8,
        "② 波动收缩 <0.8×": df.D_波动收缩 < 0.8,
        "② 未破20周线(占比<20%)": df.D_破20周线天数占比 < 0.20,
        "② 触线到买点 短50%": df.D_触线到买点 <= q(.5).D_触线到买点,
        "③ 买点日量比 ≥1.5": df.B_量比 >= 1.5,
        "③ 买点 RPS250 ≥90": df.B_RPS250 >= 90,
        "③ 买点贴近20周线 <10%": df.B_距20周线 < 0.10,
        "③ 大盘在MA200之上": df.B_大盘在MA200上 > 0,
        "③ 市值 小50%": df.B_市值分位 <= q(.5).B_市值分位,
        "③ BP 低50%(偏贵)": df.B_BP分位 <= q(.5).B_BP分位,
    }


def analyse(df, label_col, title):
    feats = binarize(df)
    b = df[label_col].to_numpy()
    BASE = b.mean()
    yr = df.year.to_numpy()
    rng = np.random.default_rng(SEED)
    perms = np.empty((N_PERM, len(b)), bool)
    for k in range(N_PERM):
        bb = b.copy()
        for yv in np.unique(yr):
            s = yr == yv
            bb[s] = rng.permutation(bb[s])
        perms[k] = bb
    out, nulls = [], {}
    early = df.date < "2019-01-01"
    for nm, mk in feats.items():
        m = mk.fillna(False).to_numpy().astype(bool)
        if m.sum() < 100:
            continue
        lf = b[m].mean() / BASE if BASE > 0 else np.nan
        nl = perms[:, m].mean(axis=1) / BASE
        nulls[nm] = nl
        p = float((np.abs(nl - 1) >= abs(lf - 1)).mean())

        def _lf(sel):
            mm = m & sel.to_numpy()
            return b[mm].mean() / b[sel.to_numpy()].mean() if mm.sum() >= 30 else np.nan
        e_, l_ = _lf(early), _lf(~early)
        out.append({"特征": nm, "命中": int(m.sum()), "P(赢|特征)": b[m].mean(),
                    "lift": lf, "p": p, "早": e_, "晚": l_,
                    "同向": bool(np.isfinite(e_) and np.isfinite(l_) and (e_ - 1) * (l_ - 1) > 0)})
    R = pd.DataFrame(out).sort_values("p")
    big = [o for o in out if o["命中"] >= MIN_HITS_CEIL]
    if len(big) >= 2:
        stack = np.vstack([nulls[o["特征"]] for o in big])
        q95 = float(np.quantile(stack.max(axis=0), 0.95))
    else:
        q95 = np.nan
    print(f"\n{'='*118}\n{title}   基准 {BASE:.2%}   公平噪音上界 {q95:.2f}\n{'='*118}")
    print(f"{'特征':<28}{'命中':>8}{'P(赢|特征)':>12}{'lift':>8}{'p':>9}{'早':>8}{'晚':>8}{'同向':>6}{'三条全过':>9}")
    for _, r in R.iterrows():
        ok = (r.p < 0.05) and r.同向 and np.isfinite(q95) and r.lift > q95
        print(f"{r.特征:<28}{r.命中:>8,}{r['P(赢|特征)']:>12.2%}{r.lift:>8.2f}{r.p:>9.4f}"
              f"{r.早:>8.2f}{r.晚:>8.2f}{'✓' if r.同向 else '✗':>6}{'**✓**' if ok else '✗':>9}")
    R["三条全过"] = [(r.p < 0.05) and r.同向 and np.isfinite(q95) and r.lift > q95
                  for _, r in R.iterrows()]
    return R, q95


R_win, q_win = analyse(IN, "win", "【选择集 2014-2019】用「翻倍」当标签")
R_tr, q_tr = analyse(IN, "win_trade", "【选择集 2014-2019】用「交易赚钱」当标签")
R_win.to_csv(f"{SP}/seq_feat_win.csv", index=False)
R_tr.to_csv(f"{SP}/seq_feat_trade.csv", index=False)

PASS = sorted(set(R_win[R_win.三条全过].特征) | set(R_tr[R_tr.三条全过].特征))
print(f"\n{'='*118}\n选择集上三条纪律全过的特征:**{len(PASS)} 个** {PASS}\n{'='*118}")
pd.Series(PASS).to_csv(f"{SP}/seq_feat_selected.csv", index=False, header=["特征"])
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: seq_feat_win.csv / seq_feat_trade.csv")
P.to_parquet(f"{SP}/seq_feature_panel.parquet")
print("Saved: seq_feature_panel.parquet(供 OOS 验证脚本使用)")
