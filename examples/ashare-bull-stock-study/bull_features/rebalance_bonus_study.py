"""再平衡收益能否落地 —— 组合构建层的系统检验

═══ 起点 ═══
同样是全市场股票、零预测,只是再平衡方式不同(2014-06-30 ~ 2026-08-03):
  ① 买入持有,从不再平衡   年化 **+6.77%**
  ② 月频再平衡等权         年化 **+12.25%**
  ③ 日频再平衡等权         年化 +14.28%
  ④ 510300(市值加权)      年化 +8.33%
**再平衡收益 = +5.5pp/年(月频),比本session 47 节找到的任何选股 alpha 都大,
且不需要任何预测能力。**

与已证实的事实自洽:两份研究都发现 A股动量失效、反转强 ——
强反转市场正是再平衡收割最多的地方。**动量赔钱与再平衡赚钱是同一枚硬币的两面。**

═══ 核心风险:它可能根本不可交易 ═══
本session的固定模式是:每个大数字先问"是不是靠少数不可交易的东西撑的"
(小市值 +26.67% → 去掉5个月剩 +11.65%;B池 +82% → 去掉5周剩 +27.7%)。
等价疑问:**再平衡收益是不是全来自买不进卖不出的小盘壳股?**
所以流动性检验放在第一步,不过就终止。

═══ 关键约束:池内一律随机抽样,不按因子挑 ═══
本方向的全部意义是"零预测"。一旦按因子挑股票就混入选股 alpha,无法归因。

═══ 事前写死的判据 ═══
第一步:"剔除最小50%市值"池的再平衡收益 ≥ +3pp/年 → 继续;
        ≤ +1pp → 收益来自不可交易的尾部,方向终止
第二步:N=20 与 N=50 的再平衡收益中位数 ≥ +2pp/年 且 25%分位 > 0
        → 散户可落地;否则记为"机构可用、散户不可用"
一律用**扣成本后**的数字下结论。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
START = "2014-06-30"
COSTS = [0.001, 0.003, 0.005]
COST_MAIN = 0.003
N_SEED = 50
SEED = 20260811

t0 = time.time()
d = {c: {} for c in ["close", "float_mv", "amount"]}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    try:
        x = pd.read_parquet(f, columns=["close", "float_mv", "amount"])
    except Exception:
        continue
    if x.empty:
        continue
    for c in d:
        d[c][k] = pd.to_numeric(x[c], errors="coerce")
CL = pd.DataFrame(d["close"]).sort_index()
CL.index = CL.index.tz_localize(None)


def _align(key):
    """索引统一成 tz-naive 再对齐(四十六节踩过:reindex_like 会全 NaN 且不报错)。"""
    f = pd.DataFrame(d[key]).sort_index()
    f.index = f.index.tz_localize(None)
    return f.reindex(index=CL.index, columns=CL.columns)


MV = _align("float_mv"); AMT = _align("amount")
CL = CL.where(CL > 0)
idx = CL.index
A = CL.to_numpy(); MVa = MV.to_numpy(); AMTa = AMT.to_numpy()
NT, NC = A.shape
for _n, _f in (("float_mv", MV), ("amount", AMT)):
    _r = _f.notna().mean().mean()
    assert _r > 0.01, f"{_n} 几乎全为 NaN(非空率 {_r:.4%})"
print(f"面板 {CL.shape}  {idx.min().date()} ~ {idx.max().date()}  ({time.time()-t0:.0f}s)")
del d

s0 = idx.searchsorted(pd.Timestamp(START))
eN = NT - 1
YRS = (idx[eN] - idx[s0]).days / 365.25
fin = np.isfinite(A)
lastv = np.where(fin.any(axis=0), NT - 1 - np.argmax(fin[::-1], axis=0), -1)

FREQS = {"日": None, "周": "W-FRI", "月": "ME", "季": "QE", "年": "YE"}


def rebal_dates(freq):
    if freq is None:
        return list(range(s0, eN + 1))
    ds = [x for x in CL.resample(freq).last().index if idx[s0] <= x <= idx[eN]]
    ps = sorted({idx.searchsorted(x, side="right") - 1 for x in ds} | {s0, eN})
    return [p for p in ps if s0 <= p <= eN]


def pool_at(p, kind):
    """p 日可投资的股票下标。kind 决定流动性/规模约束。"""
    ok = np.isfinite(A[p]) & (A[p] > 0)
    if kind == "全市场":
        return np.flatnonzero(ok)
    if kind.startswith("剔除最小"):
        q = float(kind.replace("剔除最小", "").replace("%市值", "")) / 100
        mv = np.where(ok, MVa[p], np.nan)
        th = np.nanquantile(mv, q)
        return np.flatnonzero(ok & (MVa[p] > th))
    if kind == "成交额前50%":
        am = np.nanmean(np.where(np.isfinite(AMTa[max(p - 19, 0):p + 1]),
                                 AMTa[max(p - 19, 0):p + 1], np.nan), axis=0)
        th = np.nanquantile(np.where(ok, am, np.nan), 0.5)
        return np.flatnonzero(ok & (am > th))
    raise ValueError(kind)


def buy_hold(kind, pick=None, rng=None):
    """起点等额买入,从不再平衡;退市按最后有效价清算后持币。"""
    ci = pool_at(s0, kind)
    if pick is not None and ci.size > pick:
        ci = rng.choice(ci, pick, replace=False)
    if ci.size == 0:
        return np.nan, 0
    sh = 1.0 / A[s0, ci]
    fv = np.array([A[min(lastv[c], eN), c] if lastv[c] >= s0 else A[s0, c] for c in ci])
    tot = float((sh * fv).sum() / ci.size)
    n_del = int((lastv[ci] < NT - 20).sum())
    return tot ** (1 / YRS) - 1, n_del


def rebalanced(kind, freq, cost, pick=None, rng=None, weight="等权"):
    """按 freq 再平衡。返回 (年化, 每期换手中位, 逐期净收益序列)。

    **踩过的坑**:首版用 1 日收益漂移权重,且没防 NaN —— 换手变 NaN,
    经 `pr - 2*cost*to` 传染成整条序列 NaN,年化直接返回 -100%。
    锚点自检抓住了它。现在:①用上次调仓到本次的整段收益漂移;②换手非有限时按 1 处理。
    """
    ps = rebal_dates(freq)
    rets, tos = [], []
    prev = None                      # (codes, weights, 上次调仓位置)
    for a, b in zip(ps[:-1], ps[1:]):
        ci = pool_at(a, kind)
        ci = ci[np.isfinite(A[b, ci]) & (A[b, ci] > 0)]
        if ci.size < 5:
            continue
        if pick is not None and ci.size > pick:
            ci = rng.choice(ci, pick, replace=False)
        r = A[b, ci] / A[a, ci] - 1
        if weight == "等权":
            w = np.full(ci.size, 1 / ci.size)
        elif weight == "inv_vol":
            win = A[max(a - 60, 0):a + 1, ci]
            with np.errstate(all="ignore"):
                v = np.nanstd(np.diff(np.log(win), axis=0), axis=0)
            iv = np.where(np.isfinite(v) & (v > 0), 1 / np.where(v > 0, v, np.nan), np.nan)
            iv = np.where(np.isfinite(iv), iv, np.nanmedian(iv))
            w = iv / iv.sum()
        else:
            mv = np.where(np.isfinite(MVa[a, ci]) & (MVa[a, ci] > 0), MVa[a, ci], np.nan)
            mv = np.where(np.isfinite(mv), mv, np.nanmedian(mv))
            w = mv / mv.sum()
        pr = float(np.nansum(w * r))
        # 换手:上次持仓漂移到 a 日的权重 vs 本次目标权重
        to = 1.0
        if prev is not None:
            pc, pw, pa = prev
            mult = A[a, pc] / A[pa, pc]
            mult = np.where(np.isfinite(mult) & (mult > 0), mult, 1.0)
            dv = pw * mult
            tot = dv.sum()
            if np.isfinite(tot) and tot > 0:
                cur = {c: v for c, v in zip(pc, dv / tot)}
                tgt = dict(zip(ci, w))
                to = 0.5 * sum(abs(tgt.get(k, 0.0) - cur.get(k, 0.0))
                               for k in set(cur) | set(tgt))
        if not np.isfinite(to):
            to = 1.0
        prev = (ci, w, a)
        tos.append(to)
        rets.append(pr - 2 * cost * to)
    if not rets:
        return np.nan, np.nan, np.array([])
    rr = np.array(rets, float)
    rr = rr[np.isfinite(rr)]
    if rr.size == 0:
        return np.nan, np.nan, np.array([])
    tot = float(np.prod(1 + rr))
    return (tot ** (1 / YRS) - 1 if tot > 0 else -1.0), float(np.median(tos)), rr


def mdd(r):
    eq = np.cumprod(1 + np.asarray(r, float))
    return float((eq / np.maximum.accumulate(eq) - 1).min()) if eq.size else np.nan


# ══════════════ 锚点自检 ══════════════
print(f"\n{'#'*104}\n锚点自检(须复现:买入持有 +6.77%、月频再平衡 +12.25%)\n{'#'*104}")
bh0, nd0 = buy_hold("全市场")
mo0, to0, r0 = rebalanced("全市场", "ME", 0.0)
print(f"  买入持有(全市场,不计成本)  {bh0:+.2%}   期末已退市 {nd0} 只")
print(f"  月频再平衡(全市场,不计成本) {mo0:+.2%}   换手中位 {to0:.2%}")
assert abs(bh0 - 0.0677) < 0.005 and abs(mo0 - 0.1225) < 0.005, "锚点未复现,口径漂移"
print("  ✓ 两个锚点均复现")

# ══════════════ 第一步:流动性约束 ══════════════
print(f"\n{'='*104}\n第一步:流动性约束(决定后面还做不做)\n{'='*104}")
print(f"{'股票池':<18}{'每期只数':>9}{'买入持有':>11}{'月频再平衡':>12}{'换手':>8}"
      f"{'**再平衡收益**':>15}")
POOLS = ["全市场", "剔除最小20%市值", "剔除最小50%市值", "成交额前50%"]
step1 = {}
for kind in POOLS:
    n = pool_at(idx.searchsorted(pd.Timestamp("2020-06-30")), kind).size
    bh, _ = buy_hold(kind)
    rb, to, rr = rebalanced(kind, "ME", COST_MAIN)
    step1[kind] = (bh, rb, rb - bh)
    print(f"{kind:<18}{n:>9,}{bh:>+11.2%}{rb:>+12.2%}{to:>8.2%}{(rb-bh)*100:>+14.1f}pp")
print("  (再平衡列已扣单边0.3%成本;买入持有无换手,不扣)")

gate = step1["剔除最小50%市值"][2]
print(f"\n  判据:剔除最小50%市值池的再平衡收益 = **{gate*100:+.1f}pp/年**")
if gate >= 0.03:
    print("  → **≥+3pp,通过,继续第二步**")
elif gate <= 0.01:
    print("  → **≤+1pp,收益来自不可交易的尾部 → 方向终止**")
else:
    print("  → 介于 +1~+3pp,灰区:继续但结论须降级")

if gate > 0.01:
    # ══════════════ 第二步:持仓数量 ══════════════
    print(f"\n{'='*104}\n第二步:持仓数量的衰减(池内**随机**抽,不按因子挑)\n{'='*104}")
    print(f"{'池':<16}{'N':>6}{'买入持有中位':>13}{'再平衡中位':>12}"
          f"{'**再平衡收益**':>15}{'25%分位':>10}{'75%分位':>10}")
    step2 = {}
    for kind in ("全市场", "剔除最小50%市值"):
        for N in (20, 50, 100, 300, None):
            bs, rs = [], []
            for k in range(N_SEED if N else 1):
                rng = np.random.default_rng(SEED + k)
                b, _ = buy_hold(kind, N, rng)
                rng = np.random.default_rng(SEED + k)
                r, _, _ = rebalanced(kind, "ME", COST_MAIN, N, rng)
                bs.append(b); rs.append(r)
            bs, rs = np.array(bs), np.array(rs)
            g = rs - bs
            step2[(kind, N)] = g
            print(f"{kind:<16}{(N if N else '全部'):>6}{np.median(bs):>+13.2%}"
                  f"{np.median(rs):>+12.2%}{np.median(g)*100:>+14.1f}pp"
                  f"{np.quantile(g,.25)*100:>+9.1f}pp{np.quantile(g,.75)*100:>+9.1f}pp")
        print()
    ok20 = np.median(step2[("剔除最小50%市值", 20)]) >= 0.02 and \
        np.quantile(step2[("剔除最小50%市值", 20)], .25) > 0
    ok50 = np.median(step2[("剔除最小50%市值", 50)]) >= 0.02 and \
        np.quantile(step2[("剔除最小50%市值", 50)], .25) > 0
    print(f"  判据(剔除最小50%池):N=20 {'✓' if ok20 else '✗'}   N=50 {'✓' if ok50 else '✗'}")
    print(f"  → **{'散户可落地' if (ok20 and ok50) else '机构可用、散户不可用或不充分'}**")

    # ══════════════ 第三步:频率 × 成本 ══════════════
    print(f"\n{'='*104}\n第三步:调仓频率 × 成本(剔除最小50%市值池,等权)\n{'='*104}")
    print(f"{'频率':<8}{'换手中位':>10}" + "".join(f"{f'单边{c:.1%}':>12}" for c in COSTS)
          + f"{'买入持有':>11}")
    bh_ref, _ = buy_hold("剔除最小50%市值")
    for fn, fq in FREQS.items():
        row, to = [], None
        for c in COSTS:
            a, to, _ = rebalanced("剔除最小50%市值", fq, c)
            row.append(a)
        print(f"{fn:<8}{to:>10.2%}" + "".join(f"{v:>+12.2%}" for v in row)
              + f"{bh_ref:>+11.2%}")

    # ══════════════ 第四步:加权方式 ══════════════
    print(f"\n{'='*104}\n第四步:加权方式(剔除最小50%市值池,月频,单边0.3%)\n{'='*104}")
    print(f"{'加权':<12}{'年化':>11}{'月度Sharpe':>12}{'最大回撤':>11}{'换手':>9}")
    for wt in ("等权", "inv_vol", "市值加权"):
        a, to, rr = rebalanced("剔除最小50%市值", "ME", COST_MAIN, weight=wt)
        sh = rr.mean() / rr.std() * np.sqrt(12) if rr.std() > 0 else np.nan
        print(f"{wt:<12}{a:>+11.2%}{sh:>12.3f}{mdd(rr):>11.2%}{to:>9.2%}")

    # ══════════════ 第五步:股票+现金 ══════════════
    print(f"\n{'='*104}\n第五步:股票+现金固定比例再平衡(现金按年化2%计,**这是假设**)\n{'='*104}")
    _, _, rr = rebalanced("剔除最小50%市值", "ME", COST_MAIN)
    cash_m = 0.02 / 12
    print(f"{'股票占比':<12}{'年化':>11}{'月度Sharpe':>12}{'最大回撤':>11}{'相对100%股票':>14}")
    base_a = float(np.prod(1 + rr)) ** (1 / YRS) - 1
    for w in (1.0, 0.8, 0.6, 0.4):
        mix = w * rr + (1 - w) * cash_m
        a = float(np.prod(1 + mix)) ** (1 / YRS) - 1
        sh = mix.mean() / mix.std() * np.sqrt(12) if mix.std() > 0 else np.nan
        print(f"{w:<12.0%}{a:>+11.2%}{sh:>12.3f}{mdd(mix):>11.2%}"
              f"{(a-base_a)*100:>+13.1f}pp")

print(f"\n耗时 {time.time()-t0:.0f}s")
