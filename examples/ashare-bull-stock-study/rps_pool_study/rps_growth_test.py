"""RPS 股池 + 净利润/收入双增长:是否真有超额收益

═══ 要检验的说法 ═══
用户:「2023-2026 每周拉 RPS>90 股池,只要净利润和收入增长率都>0,
投资收益并不低」。当时的做法是**每期等权买入全池、下期换仓**。

═══ 事前必须说清的先验 ═══
双增长条件保留了 A 池 68.5%、B 池 83.9% 的样本。
**一个保留 84% 样本的过滤器,数学上很难改变组合收益**,
除非被剔掉的那 16% 是灾难性的。
所以核心问题不是"双增长组赚不赚钱"(2024-09~2025 普涨,满仓必然赚),
而是**相对全池、相对同期全市场等权基准,有没有增量**。

═══ 口径(与用户当时做法一致,并堵住一个前视口子) ═══
快照日收盘后才拿到名单 → **次日开盘买入**,持有到下一快照日的**次日开盘**。
用当日收盘成交等于假设你在收盘前就知道名单。

═══ 缺失代码两种口径都报 ═══
A 缺 119 行(1.01%)、B 缺 34 行(0.45%),且缺失股的双增长占比明显更低
(A 48.7% vs 全池 68.5%)。直接剔除会**优先剔掉非双增长组的成员**,
若这些是退市股,等于替非双增长组洗掉了最差的名字。
故分别按「剔除」与「按 -100% 计入」报告。

═══ 判据(事前写死) ═══
以"相对全市场等权(同期同成本)的年化超额"为准:
  超额≥+5pp 且 双增长−非双增长 t>2 且 置换 p<0.05 → 说法成立
  超额为正但不显著 → 方向对、幅度落在噪音内
  超额≈0 → 收益来自 RPS 池或大盘β,财务过滤无增量
  全池已跑赢而双增长无额外增量 → 功劳属于 RPS,不属于财务过滤
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COSTS = [0.001, 0.003, 0.005]        # 单边
COST_MAIN = 0.003
N_PERM = 200
SEED = 20260811

t0 = time.time()

# ---------------- 价格面板 ----------------
op, cl, mv = {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue                      # 基准ETF,单独加载;其 schema 无 float_mv
    x = pd.read_parquet(f, columns=["open", "close", "float_mv"])
    if x.empty:
        continue
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
OP = pd.DataFrame(op).sort_index()
OP.index = OP.index.tz_localize(None)
CL = pd.DataFrame(cl).set_axis(OP.index)
MV = pd.DataFrame(mv).set_axis(OP.index)
OP = OP.where(OP > 0); CL = CL.where(CL > 0)
idx = OP.index
print(f"价格面板 {OP.shape}  {idx.min().date()} ~ {idx.max().date()}  ({time.time()-t0:.0f}s)")
del op, cl, mv

MKT = pd.to_numeric(pd.read_parquet(f"{DATA}/510300.parquet", columns=["open"])["open"],
                    errors="coerce")
MKT.index = MKT.index.tz_localize(None)
MKT = MKT.reindex(idx).ffill()


def next_pos(d):
    """快照日之后的第一个交易日位置(次日开盘入场)。"""
    p = idx.searchsorted(pd.Timestamp(d), side="right")
    return p if p < len(idx) else None


OPa, CLa = OP.to_numpy(), CL.to_numpy()
col_of = {c: i for i, c in enumerate(OP.columns)}


def period_rets(codes, e, x, missing_mode):
    """一期内每只股票的收益。missing_mode: 'drop' 剔除 / 'zero' 按-100%计入。

    走 numpy 索引而非 OP[c].iat —— 置换检验要跑 200 次,列查找会成为瓶颈。
    """
    out = []
    for c in codes:
        ci = col_of.get(c)
        if ci is None:
            if missing_mode == "zero":
                out.append(-1.0)
            continue
        a, b = OPa[e, ci], OPa[x, ci]
        if not np.isfinite(a) or a <= 0:
            continue                       # 入场日无价 → 根本买不进,不计
        if not np.isfinite(b) or b <= 0:
            # 期间停牌/退市:退到最后一个有效收盘价
            seg = CLa[e:x + 1, ci]
            seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                if missing_mode == "zero":
                    out.append(-1.0)
                continue
            b = seg[-1]
        out.append(b / a - 1)
    return np.array(out)


def run_pool(pool, tag, missing_mode="drop", cost=COST_MAIN):
    """按用户口径逐期推进,返回逐期结果表。"""
    snaps = sorted(pool.snap.unique())
    rows = []
    for i in range(len(snaps) - 1):
        s, s2 = snaps[i], snaps[i + 1]
        e, x = next_pos(s), next_pos(s2)
        if e is None or x is None or x <= e:
            continue
        g = pool[pool.snap == s]
        rec = {"snap": pd.Timestamp(s), "e": e, "x": x, "days": x - e,
               "n_all": len(g), "n_dual": int(g.dual.sum())}
        for name, mask in (("all", np.ones(len(g), bool)), ("dual", g.dual.to_numpy()),
                           ("nondual", ~g.dual.to_numpy()),
                           ("p_only", g.p_pos.to_numpy()), ("r_only", g.r_pos.to_numpy())):
            r = period_rets(g.code.to_numpy()[mask], e, x, missing_mode)
            rec[f"ret_{name}"] = r.mean() if len(r) else np.nan
            rec[f"cnt_{name}"] = len(r)
        # 基准:全市场等权(同期同口径)
        alive = OP.columns[np.isfinite(OP.iloc[e]) & np.isfinite(OP.iloc[x])]
        rec["ret_mkt_ew"] = (OP.iloc[x][alive] / OP.iloc[e][alive] - 1).mean()
        rec["ret_510300"] = MKT.iat[x] / MKT.iat[e] - 1
        rows.append(rec)
    R = pd.DataFrame(rows)
    # 换手与成本:等权全额换仓,按实际重合度计
    codes_by_snap = {s: set(pool[pool.snap == s].code) for s in snaps}
    tos = []
    for i in range(len(R)):
        s = R.snap.iat[i]
        prev = codes_by_snap[snaps[snaps.index(s) - 1]] if snaps.index(s) > 0 else set()
        cur = codes_by_snap[s]
        tos.append(1.0 if not prev else 1 - len(prev & cur) / max(len(cur), 1))
    R["turnover"] = tos
    R["cost"] = 2 * cost * R["turnover"]
    return R


def compound(r, days):
    """把逐期收益复利成年化。"""
    tot = np.prod(1 + r)
    yrs = days.sum() / 252.0
    return tot ** (1 / yrs) - 1 if tot > 0 and yrs > 0 else -1.0


def summarize(R, cost=COST_MAIN):
    out = {}
    for name in ("all", "dual", "nondual", "mkt_ew", "510300"):
        col = f"ret_{name}"
        r = R[col].to_numpy(float)
        ok = np.isfinite(r)
        c = R["cost"].to_numpy() if name in ("all", "dual", "nondual", "mkt_ew") else 0.0
        net = r[ok] - (c[ok] if np.ndim(c) else c)
        out[name] = {"年化": compound(net, R["days"].to_numpy()[ok]),
                     "逐期均值": net.mean(), "逐期胜率": (net > 0).mean(),
                     "期数": ok.sum()}
    return out


print(f"\n{'#'*118}")
print("说明:成本按**单边** 0.3%,并按实际换手率计(2 × 成本 × 换手)。")
print("     全市场等权基准同样按周度换仓扣成本(换手=1),否则对基准不公平。")
print(f"{'#'*118}")

results, panels = {}, {}
for tag in ("A", "B"):
    pool = pd.read_parquet(f"{SP}/rps_pool_{tag}.parquet")
    for mm in ("drop", "zero"):
        R = run_pool(pool, tag, missing_mode=mm)
        panels[(tag, mm)] = R
        if mm == "drop":
            results[tag] = R

    R = results[tag]
    span = f"{R.snap.min().date()} ~ {R.snap.max().date()}"
    print(f"\n{'='*118}")
    print(f"股池 {tag}  {span}  {len(R)} 期  中位持有 {R.days.median():.0f} 交易日  "
          f"中位换手 {R.turnover.median():.1%}")
    print(f"{'='*118}")
    print(f"{'组合':<26}{'年化':>10}{'逐期均值':>10}{'逐期胜率':>10}{'相对等权超额':>14}")
    s = summarize(R)
    base = s["mkt_ew"]["年化"]
    for name, disp in (("all", "全池(RPS筛选本身)"), ("dual", "**双增长子集**"),
                       ("nondual", "非双增长(对照)"),
                       ("mkt_ew", "全市场等权(同期同成本)"), ("510300", "510300")):
        v = s[name]
        ex = "" if name in ("mkt_ew", "510300") else f"{v['年化']-base:>+13.2f}pp"
        print(f"{disp:<26}{v['年化']:>+10.2%}{v['逐期均值']:>+10.3%}"
              f"{v['逐期胜率']:>10.1%}{ex:>14}")

    # ---- 增量检验:双增长 − 非双增长 ----
    d = (R["ret_dual"] - R["ret_nondual"]).dropna()
    t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan
    print(f"\n  逐期差(双增长 − 非双增长):均值 {d.mean():+.3%}  "
          f"期数 {len(d)}  **t值 {t:+.2f}**  胜率 {(d>0).mean():.1%}")
    dv = (R["ret_dual"] - R["ret_all"]).dropna()
    tv = dv.mean() / dv.std() * np.sqrt(len(dv)) if dv.std() > 0 else np.nan
    print(f"  逐期差(双增长 − 全池)    :均值 {dv.mean():+.3%}  **t值 {tv:+.2f}**"
          f"   ← 加了财务过滤到底值不值,看这一行")

    # ---- 缺失代码敏感性 ----
    Rz = panels[(tag, "zero")]
    sz = summarize(Rz)
    print(f"\n  缺失代码口径敏感性(年化):")
    for name, disp in (("all", "全池"), ("dual", "双增长"), ("nondual", "非双增长")):
        print(f"    {disp:<10} 剔除 {s[name]['年化']:+.2%}   按-100%计入 {sz[name]['年化']:+.2%}"
              f"   差 {sz[name]['年化']-s[name]['年化']:+.2f}pp")

    # ---- 成本敏感性 ----
    print(f"\n  成本敏感性(双增长子集年化):", end="")
    for c in COSTS:
        Rc = run_pool(pd.read_parquet(f"{SP}/rps_pool_{tag}.parquet"), tag, "drop", c)
        print(f"  单边{c:.1%} → {summarize(Rc, c)['dual']['年化']:+.2%}", end="")
    print()

    # ---- 拆开:利润>0 / 收入>0 单独 ----
    print(f"\n  拆开看(年化):", end="")
    for nm, disp in (("p_only", "仅利润>0"), ("r_only", "仅收入>0")):
        r = R[f"ret_{nm}"].to_numpy(float) - R["cost"].to_numpy()
        ok = np.isfinite(r)
        print(f"  {disp} {compound(r[ok], R['days'].to_numpy()[ok]):+.2%}", end="")
    print()

    # ---- 置换检验 ----
    pool = pd.read_parquet(f"{SP}/rps_pool_{tag}.parquet")
    rng = np.random.default_rng(SEED)
    snaps = sorted(pool.snap.unique())
    real = d.mean()
    null = []
    for _ in range(N_PERM):
        q = pool.copy()
        q["dual"] = q.groupby("snap")["dual"].transform(
            lambda z: rng.permutation(z.to_numpy()))
        Rp = run_pool(q, tag, "drop")
        null.append((Rp["ret_dual"] - Rp["ret_nondual"]).dropna().mean())
    null = np.array([v for v in null if np.isfinite(v)])
    p = float((np.abs(null) >= abs(real)).mean())
    print(f"\n  置换检验(每期池内打乱双增长标签 {len(null)} 次):")
    print(f"    真实逐期差 {real:+.3%}   纯噪音 2.5%~97.5% 分位 "
          f"[{np.quantile(null,.025):+.3%}, {np.quantile(null,.975):+.3%}]")
    print(f"    **双尾 p = {p:.3f}**  {'→ 显著' if p < 0.05 else '→ 与噪音不可区分'}"
          f"   ({time.time()-t0:.0f}s)")

    R.to_csv(f"{SP}/rps_growth_periods_{tag}.csv", index=False)

print(f"\n耗时 {time.time()-t0:.0f}s   Saved: rps_growth_periods_A.csv / _B.csv")
