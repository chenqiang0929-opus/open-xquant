"""步骤3-4:新定义下重选特征 + OOS 验证

═══ 先交代一个我事前推导错了的地方 ═══
计划里锁定 40% 的理由写的是「三条独立取 40% → 联合选中率 ≈ 0.4³ = 6.4%,
最接近选择集原本的 5.85%」。**实测新版选中率是 15~19%,不是 6.4%。**
错在**独立性假设**:缩量、波动收敛、浅回调在同一段整理里是同向变化的,
高度相关,所以联合远大于 0.4³。**这一步的相关系数本该事前测,我没测。**

按事前写死的规则,**不回头改分位数** ——
一旦开了「推导错了所以重来」的口子,和「结果不好所以重来」没有本质区别。
本脚本如实报告:判据②的选中率区间(5%~8%)**不通过**,同时报告它的实际含义。

═══ 判据(与计划完全一致,不放宽) ═══
第二关 工具级:
  - OOS 三条全中选中率落在 **5%~8%**,且逐年不再单调下滑
  - OOS 交易胜率相对同期基线提升 **≥ +4pp**
第三关 策略级:
  - 组合级年化 ≥ **+7.22%**
  - **300 次**同选中率随机对照,p < **0.0125**

═══ 三条纪律(选特征时,沿用第五十三/五十九节) ═══
  A 自身零分布(年内打乱 500 次)双侧 p < 0.05
  B lift > 公平 best-of-N 天花板(只让命中≥300 的参与)
  C 2015-05 前后两段方向一致
**选择只在 2014-2019,验证集 2020-2026 选择时完全不看。**
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
SPLIT = "2020-01-01"

t0 = time.time()
NEW = pd.read_parquet(f"{SP}/adaptive_events_new.parquet")
OLD = pd.read_parquet(f"{SP}/adaptive_events_old.parquet")
print(f"新版事件 {len(NEW):,}   旧版 {len(OLD):,}")

# ══════════ 诊断:三个指标的相关性(我事前该做而没做的检查) ══════════
print(f"\n{'='*104}\n诊断:为什么 0.4³=6.4% 的推导错了\n{'='*104}")
sub = NEW[["深度", "缩量比", "收敛比"]].dropna()
print("  三个指标两两 Spearman 相关:")
for a, b in (("深度", "缩量比"), ("深度", "收敛比"), ("缩量比", "收敛比")):
    print(f"    {a} vs {b}:  **{sub[a].rank().corr(sub[b].rank()):+.3f}**")
ind = 0.40 ** 3
act = (NEW.满足条数 == 3).mean()
print(f"  独立假设下的联合选中率 {ind:.2%}   **实测 {act:.2%}**（{act/ind:.1f} 倍）")
print("  → **三个指标高度正相关**:同一段整理里,缩量、波动收敛、浅回调是同向的。")
print("    我事前把它们当独立事件相乘,低估了联合命中率 —— 这是推导错误,不是数据问题。")

# ══════════ 面板(组合回测用) ══════════
o, h, l, c, mv = {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "high", "low", "close", "float_mv"])
    if x.empty:
        continue
    o[k] = pd.to_numeric(x["open"], errors="coerce")
    h[k] = pd.to_numeric(x["high"], errors="coerce")
    l[k] = pd.to_numeric(x["low"], errors="coerce")
    c[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
OP = pd.DataFrame(o).sort_index(); OP.index = OP.index.tz_localize(None)
LO = pd.DataFrame(l).set_axis(OP.index); CL = pd.DataFrame(c).set_axis(OP.index)
MV = pd.DataFrame(mv).set_axis(OP.index)
OP = OP.where(OP > 0); LO = LO.where(LO > 0); CL = CL.where(CL > 0)
idx = OP.index; NT = len(idx)
OPa, LOa, CLa, MVa = (OP.to_numpy(float), LO.to_numpy(float),
                      CL.to_numpy(float), MV.to_numpy(float))
col_of = {cd: i for i, cd in enumerate(OP.columns)}
_m = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])["close"],
                   errors="coerce")
_m.index = _m.index.tz_localize(None)
mkt = _m.reindex(idx).ffill()
mkt_ok = (mkt > mkt.rolling(200, min_periods=200).mean()).to_numpy()
print(f"面板就绪  ({time.time()-t0:.0f}s)")
del o, h, l, c, mv


def run_pf(evs, lo=200, hi=None, seed=SEED):
    hi = NT - 1 if hi is None else hi
    by_day = {}
    for cd, dp in zip(evs.code.to_numpy(), evs.dp.to_numpy()):
        by_day.setdefault(int(dp), []).append(cd)
    cash, holds = 1.0, {}
    equity = np.zeros(NT)
    for t in range(lo, hi + 1):
        for code in list(holds):
            hd = holds[code]; ci = hd["ci"]
            op_t, lo_t, cl_t = OPa[t, ci], LOa[t, ci], CLa[t, ci]
            ex = None
            if not np.isfinite(cl_t):
                ex = hd["last"]
            else:
                hd["last"] = cl_t
                if np.isfinite(lo_t) and lo_t <= hd["stop_px"]:
                    ex = op_t if (np.isfinite(op_t) and op_t < hd["stop_px"]) else hd["stop_px"]
                elif t - hd["t_in"] >= 252:
                    ex = cl_t
            if ex is not None and np.isfinite(ex) and ex > 0:
                cash += hd["shares"] * ex * (1 - COST_PF)
                del holds[code]
        cands = by_day.get(t - 1, [])
        free = SLOTS - len(holds)
        if cands and free > 0 and mkt_ok[t]:
            cands = [cd for cd in cands if cd not in holds
                     and np.isfinite(OPa[t, col_of[cd]]) and OPa[t, col_of[cd]] > 0]
            cands.sort(key=lambda cd: MVa[t, col_of[cd]]
                       if np.isfinite(MVa[t, col_of[cd]]) else np.inf)
            for cd in cands[:free]:
                alloc = cash / (SLOTS - len(holds)) if SLOTS > len(holds) else 0
                if alloc <= 0:
                    break
                px = OPa[t, col_of[cd]]
                holds[cd] = {"entry": px, "t_in": t, "last": px, "ci": col_of[cd],
                             "stop_px": px * 0.90, "shares": alloc * (1 - COST_PF) / px}
                cash -= alloc
        equity[t] = cash + sum(
            hd["shares"] * (CLa[t, hd["ci"]] if np.isfinite(CLa[t, hd["ci"]]) else hd["last"])
            for hd in holds.values())
    eq = pd.Series(equity[lo:hi + 1], index=idx[lo:hi + 1])
    eq = eq[eq > 0]
    if len(eq) < 100:
        return np.nan, np.nan
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
            float((eq / eq.cummax() - 1).min()))


# ══════════ 步骤3:三条纪律(选择集 2014-2019) ══════════
IN = NEW[NEW.date < SPLIT].reset_index(drop=True)
OUT = NEW[NEW.date >= SPLIT].reset_index(drop=True)
print(f"\n选择集 {len(IN):,} 笔(翻倍率 {(IN.raw252>1).mean():.2%}、交易胜率 {(IN.trade>0).mean():.2%})")
print(f"验证集 {len(OUT):,} 笔(交易胜率 {(OUT.trade>0).mean():.2%})  ← 选择时完全不看")

b = (IN.trade > 0).to_numpy()
BASE = b.mean()
yr = IN.year.to_numpy()
rng = np.random.default_rng(SEED)
perms = np.empty((N_PERM, len(b)), bool)
for k in range(N_PERM):
    bb = b.copy()
    for yv in np.unique(yr):
        s = yr == yv
        bb[s] = rng.permutation(bb[s])
    perms[k] = bb
FEATS = {
    "② 深度 最浅40%": IN["深度✓"].to_numpy(),
    "② 缩量比 最缩40%": IN["缩量比✓"].to_numpy(),
    "② 收敛比 最收敛40%": IN["收敛比✓"].to_numpy(),
    "② 三条全中": (IN.满足条数 == 3).to_numpy(),
    "【对照】调整天数 短50%": (IN.调整天数 <= IN.调整天数.median()).to_numpy(),
}
early = (IN.date < "2019-01-01").to_numpy()
print(f"\n{'='*104}\n步骤3 三条纪律(选择集,基准交易胜率 {BASE:.2%})\n{'='*104}")
print(f"{'特征':<24}{'命中':>8}{'P(赚钱|特征)':>13}{'lift':>8}{'p':>9}{'早':>7}{'晚':>7}{'同向':>6}")
nulls, res = {}, {}
for nm, m in FEATS.items():
    if m.sum() < 100:
        continue
    lf = b[m].mean() / BASE
    nl = perms[:, m].mean(axis=1) / BASE
    nulls[nm] = nl
    p = float((np.abs(nl - 1) >= abs(lf - 1)).mean())
    e_ = b[m & early].mean() / b[early].mean() if (m & early).sum() >= 30 else np.nan
    l_ = b[m & ~early].mean() / b[~early].mean() if (m & ~early).sum() >= 30 else np.nan
    same = np.isfinite(e_) and np.isfinite(l_) and (e_ - 1) * (l_ - 1) > 0
    res[nm] = {"命中": int(m.sum()), "lift": lf, "p": p, "同向": same}
    print(f"{nm:<24}{int(m.sum()):>8,}{b[m].mean():>13.2%}{lf:>8.2f}{p:>9.4f}"
          f"{e_:>7.2f}{l_:>7.2f}{'✓' if same else '✗':>6}")
big = [n for n in res if res[n]["命中"] >= 300]
if len(big) >= 2:
    q95 = float(np.quantile(np.vstack([nulls[n] for n in big]).max(axis=0), 0.95))
else:
    q95 = np.nan
print(f"\n  公平 best-of-{len(big)} 噪音上界 **{q95:.2f}**")
for nm, v in res.items():
    ok = v["p"] < 0.05 and v["同向"] and np.isfinite(q95) and v["lift"] > q95
    print(f"    {nm:<24} 三条纪律 {'**✓ 全过**' if ok else '✗'}")

# ══════════ 步骤4:OOS 验证 ══════════
print(f"\n{'='*104}\n步骤4 OOS 验证(2020-2026,选择时完全没看过)\n{'='*104}")
S0 = idx.searchsorted(pd.Timestamp(SPLIT))
FILT = {"【基线】全部OOS事件": np.ones(len(OUT), bool),
        "深度最浅40%": OUT["深度✓"].to_numpy(),
        "缩量比最缩40%": OUT["缩量比✓"].to_numpy(),
        "收敛比最收敛40%": OUT["收敛比✓"].to_numpy(),
        "**三条全中**": (OUT.满足条数 == 3).to_numpy()}
print(f"{'配置':<22}{'事件数':>8}{'选中率':>8}{'年均笔数':>9}{'胜率':>9}"
      f"{'净期望':>10}{'年化':>10}{'最大回撤':>10}")
out = {}
yrs_oos = (idx[NT - 1] - idx[S0]).days / 365.25
for nm, m in FILT.items():
    s = OUT[m]
    if len(s) < 30:
        continue
    a, dd = run_pf(s, S0, NT - 1)
    out[nm] = {"事件数": len(s), "选中率": m.mean(), "年均笔数": len(s) / yrs_oos,
               "胜率": (s.trade > 0).mean(), "净期望": s.trade.mean() - COST_TRADE,
               "年化": a, "回撤": dd}
    v = out[nm]
    print(f"{nm:<22}{len(s):>8,}{m.mean():>8.1%}{v['年均笔数']:>9.0f}{v['胜率']:>9.2%}"
          f"{v['净期望']:>+10.2%}{a:>+10.2%}{dd:>10.1%}   ({time.time()-t0:.0f}s)")

print(f"\n  随机对照 × {N_RAND}(从同期 OOS 事件里抽同样多):")
ALPHA = 0.05 / 4
for nm, m in FILT.items():
    if nm.startswith("【基线】") or nm not in out:
        continue
    k = out[nm]["事件数"]
    rg = np.random.default_rng(SEED)
    anns = np.array([run_pf(OUT.iloc[rg.choice(len(OUT), k, replace=False)], S0, NT - 1)[0]
                     for _ in range(N_RAND)])
    anns = anns[np.isfinite(anns)]
    p = float((anns >= out[nm]["年化"]).mean())
    out[nm]["p"] = p
    q = np.quantile(anns, [0.025, 0.975])
    print(f"    {nm:<22} 实际 {out[nm]['年化']:+.2%}  随机中位 {np.median(anns):+.2%}"
          f"  [{q[0]:+.2%}, {q[1]:+.2%}]  **p={p:.4f}**   ({time.time()-t0:.0f}s)")

print(f"\n{'='*104}\n判定(计划里事前写死,未放宽)\n{'='*104}")
base = out["【基线】全部OOS事件"]
tri = out.get("**三条全中**", {})
sel = tri.get("选中率", np.nan)
c2a = 0.05 <= sel <= 0.08
lift_pp = (tri.get("胜率", np.nan) - base["胜率"]) * 100
c2b = lift_pp >= 4.0
c3a = tri.get("年化", -1) >= 0.0722
c3b = tri.get("p", 1) < ALPHA
print(f"  第二关 工具级:")
print(f"    ② 选中率 5%~8%      → **{sel:.1%}**  {'✓' if c2a else '**✗**'}")
print(f"    ② 胜率提升 ≥+4pp     → {base['胜率']:.2%} → {tri.get('胜率', np.nan):.2%}"
      f"(**{lift_pp:+.1f}pp**)  {'✓' if c2b else '✗'}")
print(f"  第三关 策略级:")
print(f"    ③ 年化 ≥+7.22%      → {tri.get('年化', np.nan):+.2%}  {'✓' if c3a else '✗'}")
print(f"    ③ p < {ALPHA:.4f}        → {tri.get('p', np.nan):.4f}  {'✓' if c3b else '✗'}")
print(f"\n  **{'算发现' if (c2a and c2b and c3a and c3b) else '不算发现'}**")

pd.DataFrame(out).T.to_csv(f"{SP}/adaptive_features_oos.csv")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: adaptive_features_oos.csv")
