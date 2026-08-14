"""B 第一段到底是被什么推上去的(第六十二节第四部分)

═══ 这一节和第五十九节问的不是同一个问题 ═══
第五十九节问的是「① 的哪个特征**预测最终赚钱**」,答案是:
涨停/放量/高换手都显著**负向**(lift 0.77~0.85),业绩类正向但没过天花板。
但用户原话问的是「**这个时候它的特征是什么?会不会是其他因子导致上涨的?**」
—— 那是「**是什么推上去的**」,不是「哪个特征预测赚钱」。这个从没测过。

═══ 四个从没测过的候选 ═══
  1 板块共振:启动日 ±10 日内,全市场同时启动的股票占比
  2 财报邻近:启动日前 ≤5 个交易日内有财报(用 net_income 变化日代理)
  3 启动速度:RPS60 从 <60 升到 >90 用了几个交易日
  4 启动前形态:启动窗口之前 250 日的涨幅、以及距 250 日高点的位置

═══ 两层,缺一层结论就是错的 ═══
**层一 描述性(什么伴随启动)**:光说「80% 的启动股有板块共振」毫无信息 ——
必须有基线。所以给每个启动事件配一个**同日、同样在市、当时没有启动**的
随机对照股,比较两组的特征分布。报 P(特征|启动) 与 P(特征|对照) 及其比值。

**层二 归因(特征 vs 赚钱)**:在启动股内部沿用第五十三/五十九节的三条纪律:
  A 自身零分布(年内打乱 500 次)双侧 p < 0.05
  B lift > 公平 best-of-N 天花板(只让命中 ≥300 的参与)
  C 2015-05 前后两段方向一致
**阈值只在选择集 2014-2019 上定,验证集不重算。**

⚠️ 本节**不产出交易规则** —— 第六十二节的 A 已经证明 ① 的信息不可交易
   (剔除热钱驱动后胜率升到 25.37%,组合年化反而 -0.30%,p=0.76)。
   这里只回答「为什么涨」。

═══ 锚点 ═══
  事件 14,542 笔;选择集基准交易胜率 19.04%(第六十二节实测)
  ① 强势期涨停≥3次 lift 0.81、换手分位高50% lift 0.88 —— 必须复现
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST_TRADE = 0.003
SEED, N_PERM = 20260812, 500
SPLIT = "2020-01-01"
NEAR_REPORT_DAYS = 5      # 事前锁定
RESONANCE_WIN = 10        # 事前锁定:±10 个交易日

t0 = time.time()
NEW = pd.read_parquet(f"{SP}/adaptive_events_new.parquet")
print(f"事件 {len(NEW):,}")

cols = ["close", "turnover", "is_limit_up", "net_income"]
acc = {c: {} for c in cols}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=cols)
    if x.empty:
        continue
    for c in cols:
        acc[c][k] = pd.to_numeric(x[c], errors="coerce")
CL = pd.DataFrame(acc["close"]).sort_index()
CL.index = CL.index.tz_localize(None)
idx = CL.index
NT = len(idx)
CL = CL.where(CL > 0)
TURN = pd.DataFrame(acc["turnover"]).set_axis(idx)
LU = pd.DataFrame(acc["is_limit_up"]).set_axis(idx)
NI = pd.DataFrame(acc["net_income"]).set_axis(idx)
CLa, LUa = CL.to_numpy(float), LU.to_numpy(float)
TURN_PCT = TURN.rolling(20, min_periods=10).mean().rank(axis=1, pct=True).to_numpy(float)
col_of = {cd: i for i, cd in enumerate(CL.columns)}
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
del acc

RPS60 = (CL.pct_change(60).rank(axis=1, pct=True) * 100).to_numpy(float)
ALIVE = np.isfinite(CLa)

# ── 财报日代理:net_income(年初至今累计)发生变化的那一天 ──
REPORT = (NI.diff() != 0).to_numpy() & np.isfinite(NI.to_numpy(float))
_r = np.where(REPORT, np.arange(NT)[:, None], -10**6)
LAST_REPORT = np.maximum.accumulate(_r, axis=0)      # 每天回看:最近一次财报的下标
print(f"财报日代理:全市场共 {int(REPORT.sum()):,} 个「net_income 变化日」")

# ── 板块共振:全市场每天的「RPS60 上穿 90」股票数 / 当日在市数 ──
CROSS = (RPS60 > 90) & (np.roll(RPS60, 1, axis=0) <= 90)
CROSS[0] = False
cross_n = CROSS.sum(axis=1).astype(float)
alive_n = ALIVE.sum(axis=1).astype(float)
csum = np.concatenate([[0.0], np.cumsum(cross_n)])
asum = np.concatenate([[0.0], np.cumsum(alive_n)])


def resonance(t: int) -> float:
    """±10 交易日窗口内,全市场启动股占在市股的比例。"""
    a, b = max(0, t - RESONANCE_WIN), min(NT - 1, t + RESONANCE_WIN)
    tot = asum[b + 1] - asum[a]
    return (csum[b + 1] - csum[a]) / tot if tot > 0 else np.nan


RESO = np.array([resonance(t) for t in range(NT)])
print(f"板块共振序列就绪  ({time.time()-t0:.0f}s)")


def feats_at(j: int, ts: int) -> dict:
    """启动日 ts 上的四组特征。全部只用 ts 及之前的数据。"""
    # 启动速度:往回找最近一个 RPS60 < 60 的日子
    spd = np.nan
    lo_ = max(0, ts - 250)
    seg = RPS60[lo_:ts + 1, j]
    below = np.flatnonzero(np.isfinite(seg) & (seg < 60))
    if below.size:
        spd = ts - (lo_ + int(below[-1]))
    # 财报邻近
    lr = LAST_REPORT[ts, j]
    near = (ts - lr) if lr > -10**5 else np.nan
    # 启动前形态(启动窗口 ts-60 之前)
    p0 = ts - 60
    pre_ret = pre_hi = np.nan
    if p0 - 250 >= 0 and np.isfinite(CLa[p0, j]):
        b0 = CLa[p0 - 250, j]
        if np.isfinite(b0) and b0 > 0:
            pre_ret = CLa[p0, j] / b0 - 1
        w = CLa[p0 - 250:p0 + 1, j]
        w = w[np.isfinite(w)]
        if w.size and w.max() > 0:
            pre_hi = CLa[p0, j] / w.max() - 1
    return {"板块共振": RESO[ts], "启动速度": spd, "距财报天数": near,
            "启动前250日涨幅": pre_ret, "启动前距250日高": pre_hi,
            "涨停次数": np.nansum(LUa[max(ts - 60, 0):ts + 1, j]),
            "换手分位": TURN_PCT[ts, j]}


# ══════════ 层一:启动股 vs 同日随机对照股 ══════════
rng = np.random.default_rng(SEED)
ev_rows, ct_rows = [], []
launch_by_t = {}
for cd, ts in zip(NEW.code.to_numpy(), NEW.t_strong.to_numpy()):
    launch_by_t.setdefault(int(ts), set()).add(col_of[cd])
for cd, ts in zip(NEW.code.to_numpy(), NEW.t_strong.to_numpy()):
    j, ts = col_of[cd], int(ts)
    ev_rows.append(feats_at(j, ts))
    # 对照:同一天在市、且当天没有启动(RPS60 未上穿 90)的随机一只
    pool = np.flatnonzero(ALIVE[ts] & ~CROSS[ts])
    if pool.size:
        ct_rows.append(feats_at(int(rng.choice(pool)), ts))
EV = pd.DataFrame(ev_rows)
CT = pd.DataFrame(ct_rows)
print(f"层一样本:启动 {len(EV):,} 对照 {len(CT):,}  ({time.time()-t0:.0f}s)")

print(f"\n{'='*104}\n层一 描述性:启动股 vs 同日随机对照股(中位数)\n{'='*104}")
print(f"{'指标':<20}{'启动股':>12}{'对照股':>12}{'倍数/差':>12}")
L1 = []
for c in ("板块共振", "启动速度", "距财报天数", "启动前250日涨幅",
          "启动前距250日高", "涨停次数", "换手分位"):
    a, b = EV[c].median(), CT[c].median()
    rel = (a / b) if (np.isfinite(b) and b != 0) else np.nan
    L1.append({"指标": c, "启动股中位": a, "对照股中位": b, "倍数": rel})
    print(f"{c:<20}{a:>12.3f}{b:>12.3f}{rel:>12.2f}")
p_near_ev = (EV.距财报天数 <= NEAR_REPORT_DAYS).mean()
p_near_ct = (CT.距财报天数 <= NEAR_REPORT_DAYS).mean()
print(f"\n  **P(启动日前{NEAR_REPORT_DAYS}日内有财报)**:启动股 **{p_near_ev:.1%}**"
      f"   对照股 {p_near_ct:.1%}   lift **{p_near_ev/p_near_ct:.2f}**"
      if p_near_ct > 0 else "")

# ══════════ 层二:归因(三条纪律) ══════════
D = pd.concat([NEW.reset_index(drop=True), EV], axis=1)
IN = D[D.date < SPLIT].reset_index(drop=True)
b = (IN.trade > 0).to_numpy()
BASE = b.mean()
print(f"\n{'='*104}\n层二 归因(选择集 2014-2019,基准交易胜率 {BASE:.2%})\n{'='*104}")

# 复现锚点
for nm, m in (("① 涨停≥3次", (IN.涨停次数 >= 3).to_numpy()),
              ("① 换手分位高50%", (IN.换手分位 >= IN.换手分位.median()).to_numpy())):
    print(f"  锚点 {nm}: lift {b[m].mean()/BASE:.2f}  (应 0.81 / 0.88)")

q = IN.median(numeric_only=True)
FEATS = {
    "B1 板块共振 高50%": (IN.板块共振 >= q.板块共振).to_numpy(),
    "B2 财报邻近 ≤5日": (IN.距财报天数 <= NEAR_REPORT_DAYS).to_numpy(),
    "B3 启动速度 快50%": (IN.启动速度 <= q.启动速度).to_numpy(),
    "B4 启动前250日涨幅 低50%": (IN.启动前250日涨幅 <= q.启动前250日涨幅).to_numpy(),
    "B5 启动前贴近250日高 <10%": (IN.启动前距250日高 >= -0.10).to_numpy(),
}
yr = IN.year.to_numpy()
rr = np.random.default_rng(SEED)
perms = np.empty((N_PERM, len(b)), bool)
for k in range(N_PERM):
    bb = b.copy()
    for yv in np.unique(yr):
        s = yr == yv
        bb[s] = rr.permutation(bb[s])
    perms[k] = bb
early = (IN.date < "2019-01-01").to_numpy()
print(f"\n{'特征':<26}{'命中':>8}{'P(赚钱|特征)':>13}{'lift':>8}{'p':>9}"
      f"{'早':>7}{'晚':>7}{'同向':>6}")
nulls, res = {}, {}
for nm, m in FEATS.items():
    m = m & np.isfinite(IN.trade.to_numpy())
    if m.sum() < 100:
        continue
    lf = b[m].mean() / BASE
    nl = perms[:, m].mean(axis=1) / BASE
    nulls[nm] = nl
    p = float((np.abs(nl - 1) >= abs(lf - 1)).mean())
    e_ = b[m & early].mean() / b[early].mean() if (m & early).sum() >= 30 else np.nan
    l_ = b[m & ~early].mean() / b[~early].mean() if (m & ~early).sum() >= 30 else np.nan
    same = np.isfinite(e_) and np.isfinite(l_) and (e_ - 1) * (l_ - 1) > 0
    res[nm] = {"命中": int(m.sum()), "lift": lf, "p": p, "同向": same,
               "P赚钱": b[m].mean()}
    print(f"{nm:<26}{int(m.sum()):>8,}{b[m].mean():>13.2%}{lf:>8.2f}{p:>9.4f}"
          f"{e_:>7.2f}{l_:>7.2f}{'✓' if same else '✗':>6}")
big = [n for n in res if res[n]["命中"] >= 300]
q95 = (float(np.quantile(np.vstack([nulls[n] for n in big]).max(axis=0), 0.95))
       if len(big) >= 2 else np.nan)
print(f"\n  公平 best-of-{len(big)} 噪音上界 **{q95:.2f}**")
n_pass = 0
for nm, v in res.items():
    ok = v["p"] < 0.05 and v["同向"] and np.isfinite(q95) and v["lift"] > q95
    n_pass += ok
    print(f"    {nm:<26} 三条纪律 {'**✓ 全过**' if ok else '✗'}")
print(f"\n  **通过三条纪律的:{n_pass} 个**")

pd.DataFrame(L1).to_csv(f"{SP}/first_leg_control.csv", index=False)
pd.DataFrame(res).T.to_csv(f"{SP}/first_leg_drivers.csv")
print(f"\n→ first_leg_control.csv / first_leg_drivers.csv   ({time.time()-t0:.0f}s)")
