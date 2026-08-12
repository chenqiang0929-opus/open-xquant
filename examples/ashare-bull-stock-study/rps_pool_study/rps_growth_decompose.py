"""独立复现 + 拆解:B 池 +82% 到底是什么

═══ 为什么必须做这一步 ═══
主检验结果:B 池(2025-01~2026-07)双增长子集年化 **+82.24%**,
同期全市场等权 +17.81%、510300 +22.03%,置换 p=0.000。
**这是本session出现过的最大超额。恰恰因为它太大,不能直接采信。**

前四十二节反复出现同一个教训:一个漂亮的数字,先找它可能来自哪里。
本脚本查四件事:

  1. **独立复现**:用我自己的价格面板重建 RPS>90 股池(不碰用户文件),
     同样口径跑一遍。能复现 → 是真规律;不能 → 快照文件里有我复现不了的东西。
  2. **与用户股池的重合度**:逐期算交集比例。若重合很低,说明用户的
     筛选条件不止 RPS,"RPS>90"这个描述本身就不准确。
  3. **拆解**:双增长组内按 流通市值 / RPS250 / 市盈 分三档,看收益是否单调。
     若全部来自小市值档,结论就是三十五节的「小市值」而非「双增长」。
  4. **集中度**:去掉表现最好的 3 期后还剩多少;逐期收益的最大回撤。
     若 +82% 靠几周撑着,那是运气不是规律。

═══ 复现用的定义(事前锁定,不调参) ═══
RPS250 = 250日收益率的横截面百分位 × 100,取 > 90(用户口径)。
双增长用面板自带的 ni_yoy_252(净利润同比)与 revenue 的 252 日同比。
**面板财务字段的 PIT 是我此前重建的,不如用户快照可靠**,
所以复现结果只用来判断"量级是否接近",不用来替代用户数据的结论。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST = 0.003

t0 = time.time()

op, cl, mv, niy, rev = {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "close", "float_mv", "ni_yoy_252", "revenue"])
    if x.empty:
        continue
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    niy[k] = pd.to_numeric(x["ni_yoy_252"], errors="coerce")
    rev[k] = pd.to_numeric(x["revenue"], errors="coerce")
OP = pd.DataFrame(op).sort_index(); OP.index = OP.index.tz_localize(None)
CL = pd.DataFrame(cl).set_axis(OP.index); MV = pd.DataFrame(mv).set_axis(OP.index)
NIY = pd.DataFrame(niy).set_axis(OP.index); REV = pd.DataFrame(rev).set_axis(OP.index)
OP = OP.where(OP > 0); CL = CL.where(CL > 0)
idx = OP.index
OPa, CLa = OP.to_numpy(), CL.to_numpy()
col_of = {c: i for i, c in enumerate(OP.columns)}
print(f"面板 {OP.shape}  ({time.time()-t0:.0f}s)")
del op, cl, mv, niy, rev

# 复现用:250日动量的横截面百分位 + 收入同比
MOM250 = CL.pct_change(250)
RPS250 = MOM250.rank(axis=1, pct=True) * 100

# ═══ 成长字段:改用修正后的口径(见 build_clean_growth.py 与第五十二节) ═══
# 原 `ni_yoy_252` = net_income/net_income.shift(252)-1,而 net_income 是 YTD 累计,
# 252 交易日 ≈ 1.04 年会跨报告期 —— 茅台 2023-05-04 得 -60.4%(单季比全年)。
# 污染量化:横截面「>0比例」按月份极差 25.1pp。
# 修正后(去累计 + 报告期对齐,已对官方财报核验 9 个单季全部吻合)极差 2.5~2.8pp。
def _load_clean_growth(index, columns):
    ni = pd.read_parquet(f"{SP}/clean_growth_ni_ttm_yoy.parquet").reindex(
        index=index, columns=columns)
    rv = pd.read_parquet(f"{SP}/clean_growth_rev_ttm_yoy.parquet").reindex(
        index=index, columns=columns)
    assert ni.notna().mean().mean() > 0.01, "clean_growth 净利字段几乎全空"
    assert rv.notna().mean().mean() > 0.01, "clean_growth 收入字段几乎全空"
    print(f"  成长字段 = **修正后 TTM 同比**(净利非空 {ni.notna().mean().mean():.1%}、"
          f"收入非空 {rv.notna().mean().mean():.1%})")
    return ni, rv

NIY, REV_YOY = _load_clean_growth(OP.index, OP.columns)


def next_pos(d):
    p = idx.searchsorted(pd.Timestamp(d), side="right")
    return p if p < len(idx) else None


def rets(codes, e, x):
    out = []
    for c in codes:
        ci = col_of.get(c)
        if ci is None:
            continue
        a, b = OPa[e, ci], OPa[x, ci]
        if not np.isfinite(a) or a <= 0:
            continue
        if not np.isfinite(b) or b <= 0:
            seg = CLa[e:x + 1, ci]; seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                continue
            b = seg[-1]
        out.append(b / a - 1)
    return np.array(out)


def compound(r, days):
    tot = np.prod(1 + r)
    yrs = np.sum(days) / 252.0
    return tot ** (1 / yrs) - 1 if tot > 0 and yrs > 0 else -1.0


for tag in ("A", "B"):
    pool = pd.read_parquet(f"{SP}/rps_pool_{tag}.parquet")
    snaps = sorted(pool.snap.unique())
    print(f"\n{'='*118}\n股池 {tag}:独立复现与拆解\n{'='*118}")

    rows = []
    for i in range(len(snaps) - 1):
        s, s2 = snaps[i], snaps[i + 1]
        e, x = next_pos(s), next_pos(s2)
        if e is None or x is None or x <= e:
            continue
        g = pool[pool.snap == s]
        user_codes = set(g.code)

        # ---- 1) 我自己重建的 RPS>90 池 ----
        r250 = RPS250.iloc[e - 1]                       # 快照日当天可得的信息
        alive = np.isfinite(OP.iloc[e]) & np.isfinite(OP.iloc[x])
        mine = set(OP.columns[(r250 > 90).fillna(False) & alive])
        # 加双增长(用面板财务)
        ny, ry = NIY.iloc[e - 1], REV_YOY.iloc[e - 1]
        mine_dual = set(OP.columns[(r250 > 90).fillna(False) & alive
                                   & (ny > 0).fillna(False) & (ry > 0).fillna(False)])

        rec = {"snap": pd.Timestamp(s), "days": x - e,
               "n_user": len(user_codes), "n_mine": len(mine), "n_mine_dual": len(mine_dual),
               "overlap": len(user_codes & mine) / max(len(user_codes), 1)}
        rec["ret_user_dual"] = rets(g.code.to_numpy()[g.dual.to_numpy()], e, x).mean() \
            if g.dual.any() else np.nan
        rec["ret_mine"] = rets(list(mine), e, x).mean() if mine else np.nan
        rec["ret_mine_dual"] = rets(list(mine_dual), e, x).mean() if mine_dual else np.nan

        # ---- 3) 拆解:双增长组内按市值/RPS/市盈分三档 ----
        gd = g[g.dual].copy()
        gd["fmv"] = [MV.iloc[e - 1].get(c, np.nan) for c in gd.code]
        for key, col in (("mv", "fmv"), ("pe", "pe"), ("rps", "RPS250")):
            v = pd.to_numeric(gd[col], errors="coerce")
            if v.notna().sum() < 15:
                continue
            try:
                q = pd.qcut(v.rank(method="first"), 3, labels=False)
            except ValueError:
                continue
            for k in range(3):
                cs = gd.code.to_numpy()[(q == k).to_numpy(na_value=False)]
                rr = rets(cs, e, x)
                rec[f"{key}_q{k+1}"] = rr.mean() if len(rr) else np.nan
        rows.append(rec)

    R = pd.DataFrame(rows)
    d = R["days"].to_numpy()

    print(f"\n【1+2】独立复现 与 重合度   ({len(R)} 期)")
    print(f"  用户股池每期只数 中位 {R.n_user.median():.0f}   "
          f"我重建的 RPS250>90 池 中位 {R.n_mine.median():.0f}")
    print(f"  **逐期重合度(用户池 ∩ 我的池 ÷ 用户池):中位 {R.overlap.median():.1%}**"
          f"  最低 {R.overlap.min():.1%}  最高 {R.overlap.max():.1%}")
    for nm, disp in (("ret_user_dual", "用户池·双增长(主检验结果)"),
                     ("ret_mine", "我重建的 RPS250>90 全池"),
                     ("ret_mine_dual", "我重建的 RPS250>90 + 双增长")):
        r = R[nm].to_numpy(float); ok = np.isfinite(r)
        if ok.sum() < 5:
            print(f"  {disp:<30} 样本不足")
            continue
        net = r[ok] - 2 * COST * 0.32          # 与主检验同口径:换手约32%
        print(f"  {disp:<30} 年化 {compound(net, d[ok]):>+9.2%}   "
              f"逐期均值 {net.mean():>+7.3%}   期数 {ok.sum()}")

    print(f"\n【3】双增长组内分档(逐期均值,看是否单调)")
    for key, disp in (("mv", "流通市值 小→大"), ("pe", "市盈 低→高"), ("rps", "RPS250 低→高")):
        cols = [f"{key}_q{k}" for k in (1, 2, 3) if f"{key}_q{k}" in R.columns]
        if len(cols) < 3:
            print(f"  {disp:<16} (该池无此字段)")
            continue
        vals = [R[c].mean() for c in cols]
        print(f"  {disp:<16} Q1 {vals[0]:+.3%}   Q2 {vals[1]:+.3%}   Q3 {vals[2]:+.3%}"
              f"   Q1-Q3 {vals[0]-vals[2]:+.3%}")

    print(f"\n【4】集中度:收益是否靠少数几期")
    r = R["ret_user_dual"].to_numpy(float); ok = np.isfinite(r)
    net = r[ok] - 2 * COST * 0.32
    full = compound(net, d[ok])
    order = np.argsort(net)[::-1]
    for k in (1, 3, 5):
        keep = np.ones(len(net), bool); keep[order[:k]] = False
        print(f"  去掉最好的 {k} 期 → 年化 {compound(net[keep], d[ok][keep]):>+9.2%}"
              f"   (完整 {full:+.2%})")
    eq = np.cumprod(1 + net)
    print(f"  逐期净值最大回撤 {(eq/np.maximum.accumulate(eq)-1).min():.2%}"
          f"   单期最好 {net.max():+.2%}  单期最差 {net.min():+.2%}")
    R.to_csv(f"{SP}/rps_decompose_{tag}.csv", index=False)

print(f"\n耗时 {time.time()-t0:.0f}s   Saved: rps_decompose_A.csv / _B.csv")
