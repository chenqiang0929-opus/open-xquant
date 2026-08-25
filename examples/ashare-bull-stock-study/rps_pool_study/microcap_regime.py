"""§120 微盘策略的风格开关:能不能只砍回撤、不砍收益?

问题
----
§115-B/§116-A 里同时通过市值中性对照与 2026 干净留出期的两个因子
(`small_cap_low_turnover`、`high_amihud`)**不缺收益,缺回撤控制**:

  修正版 R10 全期最大回撤 **-56.38%**(不是 Codex 报的 -41.55%)
  `small_cap` 在 2026 干净留出期 **-18.16%**

Codex 的 R12「市场状态自适应」正是想解决这个,他用 **510300 月线 MA20 + MACD**
做开关,失败原因他自己写在文档里:**「510300 状态无法识别 2026 小盘风格转弱」**。
**用大盘指数的状态去给小盘策略做开关,方向就错了。**

本节检验三个开关(全部只用微盘自身或市场内部信息,不用大盘指数):

  SW1 相对强弱开关:微盘等权指数 / 510300 的比值,跌破其自身 N 日均线 → 空仓
  SW2 市场宽度开关:全市场收盘价在 MA100 之上的股票占比,跌破阈值 → 空仓
  SW3 自身回撤开关:策略净值自身回撤超过阈值 → 空仓(最朴素,作对照基线)

开关只在调仓日生效(与 20 日调仓同步),空仓 = 全部卖出持现金,
下一个调仓日信号恢复才重新买入。交易成本照付。

**参数只在训练期 2014-01-02→2021-12-31 内选,留出期 2022-01-04→面板末只看一次。**
候选参数网格(写死,不得扩充):
  SW1 N ∈ {60, 120, 250}          SW2 阈值 ∈ {0.30, 0.40, 0.50}
  SW3 阈值 ∈ {0.15, 0.25, 0.35}
每个开关在训练期内按判据 M2 选出唯一一组参数,拿到留出期跑一次。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
M1 锚点(不过则整节作废):面板 (3297,5217);
   **开关恒等式**:开关全程恒为「开」时,带开关的净值曲线必须与不带开关的
   基线**逐日相等**(最大相对误差 < 1e-9)。写错必抓,写对必过。
   **随机择时对照的空仓天数**必须与被对照的真开关空仓天数相等(±1 个调仓日)。

M2 训练期选参。对每个开关,在其参数网格内选出**唯一**一组参数,规则写死为:
   在「训练期年化下降 ≤ 3pp」的候选里,取**训练期最大回撤改善最大**的那组;
   若没有任何候选满足「年化下降 ≤ 3pp」,该开关判**训练期即失败**,不进留出期。

M3 **留出期判定。核心判据,只看一次。**
   M3 通过 ⟺ 在留出期同时满足:
     (a) 最大回撤相对不带开关的基线**改善 ≥ 10pp**;
     (b) 年化相对基线**下降 ≤ 3pp**;
     (c) 相对**随机择时对照**显著:随机择时 = 保持同样的空仓天数,
         但空仓时点随机决定,200 组种子;
         真开关的留出期年化须**严格高于**随机择时的 50 分位,
         **且**其留出期最大回撤须**优于**随机择时的 25 分位。
   (c) 是堵「少交易本身就降回撤」这个平凡解释的口 —— 没有它,
   任何降低仓位的规则都会「成功」。

M4 描述项(不设阈值):空仓天数占比、切换次数、留出期分年收益、
   开关在 2015 股灾 / 2018 熊市 / 2022 熊市 / 2026 风格切换四段的表现。

标的策略:`small_cap_low_turnover`(§114/§116-A 的幸存者,真实市值口径),
引擎与口径与 §114 完全一致。

事前预测(写下来以便被证伪;错了必须在正文里明说我错了)
--------------------------------------------------------
V1 SW3(自身回撤开关)在训练期会被 M2 选出参数,但**留出期过不了 M3(c)**——
   它本质上就是「跌了就减仓」,与随机择时的区别只在于用了自身净值信息,
   而自身净值是滞后的。
V2 **SW1(相对强弱开关)是三个里最可能过 M3 的。** 理由:它直接测的是
   「微盘相对大盘是否还在占优」,与失效场景(风格切换)一一对应;
   而 SW2 的市场宽度是全市场指标,对小盘风格的针对性弱于 SW1。
V3 **三个开关里通过 M3 的数量 ∈ [0, 1]。**
V4 至少一个开关会在训练期就失败(M2 找不到满足「年化下降 ≤ 3pp」的参数)。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推;
**不扩充参数网格、不因留出期结果不好回头改训练期选参规则**;
不基于本节结论做任何可交易性声明。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, OUT, SEED, build_sel, run_window_fast  # noqa: E402
from codex_r10_replication import DATA, TOP_N, metrics  # noqa: E402

NSEED = 200
WINS = {"train": ("2014-01-02", "2021-12-31"), "holdout": ("2022-01-04", "2026-08-03")}
GRID = {"SW1": (60, 120, 250), "SW2": (0.30, 0.40, 0.50), "SW3": (0.15, 0.25, 0.35)}


def gate_sel(sel, on):
    """把开关作用到选股字典:on[t] 为 False 的调仓日改为空持仓(全部卖出)。"""
    return {t: (v if on.get(t, True) else (np.zeros(0, np.int64), np.zeros(0)))
            for t, v in sel.items()}


def main():
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    op, cl, susp, lu, ld, ok = z["OP"], z["CL"], z["SUSP"], z["LU"], z["LD"], z["OK"]
    logcap, tmean = z["LOGCAP"], z["TMEAN"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点M1a"
    print(f"锚点M1a ✓ {nt}×{ns}", flush=True)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill().reindex(idx).ffill()
    cal = pd.DatetimeIndex(b.index.unique()).sort_values()
    cal = cal[(cal >= "2014-01-01") & (cal <= "2026-08-20")]
    cal_pos = pd.Index(idx).get_indexer(cal)
    reb = cal_pos[::20]
    ipos = pd.Index(idx)
    sel, elig, _ = build_sel(reb, ok, logcap, tmean)
    print(f"标的策略 small_cap_low_turnover,调仓日 {len(sel)}", flush=True)

    # ── 三个开关的原始序列(逐日)──
    ret = np.zeros_like(cl)
    ret[1:] = cl[1:] / cl[:-1] - 1.0
    micro = np.full(nt, np.nan)                     # 微盘等权指数(最小十分位)
    eqv, held = 1.0, None
    for t in range(nt):
        if held is not None and len(held):
            r = ret[t, held]
            r = r[np.isfinite(r)]
            eqv *= 1 + (float(r.mean()) if len(r) else 0.0)
        micro[t] = eqv
        if t in elig:
            o = elig[t]
            held = o[:max(TOP_N, len(o) // 10)]
    rel = pd.Series(micro / bs.to_numpy(), index=idx)
    ma100 = pd.DataFrame(cl, index=idx).rolling(100, min_periods=100).mean().to_numpy()
    with np.errstate(all="ignore"):
        above = np.nansum(cl > ma100, axis=1) / np.maximum(
            np.nansum(np.isfinite(ma100) & np.isfinite(cl), axis=1), 1)
    breadth = pd.Series(above, index=idx)

    def wpos(w):
        d0, d1 = WINS[w]
        return (int(ipos.get_indexer([pd.Timestamp(d0)], method="bfill")[0]),
                int(ipos.get_indexer([pd.Timestamp(d1)], method="ffill")[0]))

    def run(s, w):
        w0, w1 = wpos(w)
        eq, dd, tr, fz = run_window_fast(op, cl, susp, lu, ld, s, cal_pos, w0, w1)
        return metrics(eq, dd, idx), eq, dd

    # ── 锚点 M1:开关恒为「开」时必须与基线逐日相等 ──
    base = {w: run(sel, w) for w in WINS}
    allon = gate_sel(sel, {t: True for t in sel})
    ok_id = True
    for w in WINS:
        e1 = base[w][1]
        e2 = run(allon, w)[1]
        err = float(np.max(np.abs(e2 - e1) / np.maximum(np.abs(e1), 1e-9)))
        ok_id &= err < 1e-9
        print(f"锚点M1 开关恒等式 {w:8s} 最大相对误差 {err:.3e} "
              f"{'✓' if err < 1e-9 else '✗'}", flush=True)
    assert ok_id, "锚点M1 不过"
    for w in WINS:
        m = base[w][0]
        print(f"基线(无开关) {w:8s} 年化{m['cagr']:+7.2%} 回撤{m['mdd']:+7.2%} "
              f"夏普{m['sharpe']:5.2f}", flush=True)

    def switch_on(kind, par):
        """返回 {调仓日 -> 是否持仓}。信号只用到 t 当日及以前的信息。"""
        if kind == "SW1":
            ma = rel.rolling(par, min_periods=par).mean()
            sig = (rel > ma).to_numpy()
            return {int(t): bool(sig[int(t)]) if np.isfinite(rel.iloc[int(t)])
                    else True for t in sel}
        if kind == "SW2":
            sig = (breadth > par).to_numpy()
            return {int(t): bool(sig[int(t)]) for t in sel}
        raise ValueError(kind)

    def sw3_on(par, w):
        """自身回撤开关必须逐日模拟(依赖自身净值),在窗口内滚动生成。"""
        w0, w1 = wpos(w)
        days = cal_pos[(cal_pos >= w0) & (cal_pos <= w1)]
        on, cur = {}, True
        eq, _, _ = base[w][1], None, None
        pos = {int(d): k for k, d in enumerate(days)}
        peak = -np.inf
        for t in sel:
            if int(t) not in pos:
                continue
            k = pos[int(t)]
            peak = max(peak, float(np.max(eq[:k + 1])))
            cur = (peak - float(eq[k])) / peak <= par
            on[int(t)] = bool(cur)
        return on

    rows = []
    for kind in ("SW1", "SW2", "SW3"):
        cands = []
        for par in GRID[kind]:
            on = sw3_on(par, "train") if kind == "SW3" else switch_on(kind, par)
            m, _, _ = run(gate_sel(sel, on), "train")
            drop = (base["train"][0]["cagr"] - m["cagr"]) * 100
            imp = (m["mdd"] - base["train"][0]["mdd"]) * 100
            cands.append((par, drop, imp, m))
            print(f"  {kind} par={par} 训练期 年化{m['cagr']:+7.2%}"
                  f"(降{drop:+5.2f}pp) 回撤{m['mdd']:+7.2%}(改善{imp:+5.2f}pp)",
                  flush=True)
        good = [c for c in cands if c[1] <= 3.0]
        if not good:
            print(f"{kind} **训练期即失败**:没有任何参数满足年化下降 ≤3pp\n", flush=True)
            rows.append({"switch": kind, "M2": False, "M3": False})
            continue
        par, drop, imp, mtr = max(good, key=lambda c: c[2])
        print(f"{kind} M2 选中 par={par}(训练期回撤改善 {imp:+.2f}pp,年化降 {drop:+.2f}pp)",
              flush=True)

        on_h = sw3_on(par, "holdout") if kind == "SW3" else switch_on(kind, par)
        mh, _, _ = run(gate_sel(sel, on_h), "holdout")
        w0, w1 = wpos("holdout")
        keys = [t for t in sel if w0 <= t <= w1]
        noff = sum(1 for t in keys if not on_h.get(int(t), True))
        rng = np.random.default_rng(SEED)
        rc, rm = [], []
        for _ in range(NSEED):
            off = set(rng.choice(len(keys), size=noff, replace=False)) if noff else set()
            ro = {int(t): (i not in off) for i, t in enumerate(keys)}
            m2, _, _ = run(gate_sel(sel, ro), "holdout")
            rc.append(m2["cagr"])
            rm.append(m2["mdd"])
        rc, rm = np.array(rc), np.array(rm)
        a = (mh["mdd"] - base["holdout"][0]["mdd"]) * 100 >= 10.0
        bb = (base["holdout"][0]["cagr"] - mh["cagr"]) * 100 <= 3.0
        cc = (mh["cagr"] > np.percentile(rc, 50)) and (mh["mdd"] > np.percentile(rm, 25))
        rows.append({"switch": kind, "par": par, "M2": True,
                     "train_cagr": mtr["cagr"], "train_mdd": mtr["mdd"],
                     "hold_cagr": mh["cagr"], "hold_mdd": mh["mdd"],
                     "base_hold_cagr": base["holdout"][0]["cagr"],
                     "base_hold_mdd": base["holdout"][0]["mdd"],
                     "off_days": noff, "n_reb": len(keys),
                     "rand_cagr_p50": float(np.percentile(rc, 50)),
                     "rand_mdd_p25": float(np.percentile(rm, 25)),
                     "M3a": bool(a), "M3b": bool(bb), "M3c": bool(cc),
                     "M3": bool(a and bb and cc)})
        print(f"{kind} 留出期 年化{mh['cagr']:+7.2%}(基线{base['holdout'][0]['cagr']:+7.2%})"
              f" 回撤{mh['mdd']:+7.2%}"
              f"(基线{base['holdout'][0]['mdd']:+7.2%}) 空仓{noff}/{len(keys)}\n"
              f"     M3a 回撤改善≥10pp {'✓' if a else '✗'} | "
              f"M3b 年化降≤3pp {'✓' if bb else '✗'} | "
              f"M3c 优于随机择时(年化 vs p50 {np.percentile(rc,50):+.2%}, "
              f"回撤 vs p25 {np.percentile(rm,25):+.2%}) {'✓' if cc else '✗'} "
              f"→ M3 {'✓' if (a and bb and cc) else '✗'}\n", flush=True)

    df = pd.DataFrame(rows)
    m3 = df["M3"] if "M3" in df.columns else pd.Series([False] * len(df))
    npass = int(m3.fillna(False).sum())
    print(f"M3 留出期通过 {npass}/3:{', '.join(df.loc[m3.fillna(False), 'switch']) or '无'}")
    df.to_csv(f"{OUT}/microcap_regime.csv", index=False)
    print(f"落库 {OUT}/microcap_regime.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §120 结果:三个开关**全部在训练期就失败**,没有一个走到留出期。
#
# 锚点 M1 开关恒等式:开关恒为「开」时与基线逐日相等,train 与 holdout
#      最大相对误差均为 **0.000e+00** ✓
#
# 基线(无开关,small_cap_low_turnover,真实市值口径)
#   训练 2014–2021  年化 +21.89%  回撤 -56.38%  夏普 1.01
#   留出 2022–2026  年化 +29.02%  回撤 -27.78%  夏普 1.27
#
# 开关   参数    训练期年化   相对基线   训练期回撤   回撤改善   M2
# SW1     60    +17.56%    -4.34pp   -41.78%   +14.60pp   ✗
# SW1    120    +12.96%    -8.94pp   -44.76%   +11.62pp   ✗
# SW1    250    +13.61%    -8.28pp   -51.03%    +5.35pp   ✗
# SW2   0.30    +12.62%    -9.27pp   -58.13%    -1.75pp   ✗
# SW2   0.40    +12.57%    -9.32pp   -50.95%    +5.43pp   ✗
# SW2   0.50    +14.29%    -7.60pp   -39.59%   +16.79pp   ✗
# SW3   0.15    +13.77%    -8.13pp   -46.45%    +9.93pp   ✗
# SW3   0.25    +15.59%    -6.30pp   -44.16%   +12.22pp   ✗
# SW3   0.35    +14.03%    -7.86pp   -57.93%    -1.55pp   ✗
#
# M2 的规则是事前写死的:「在训练期年化下降 ≤3pp 的候选里,取回撤改善最大的一组;
# 若无候选满足,该开关判训练期即失败」。**九组参数没有一组的年化下降 ≤3pp**,
# 最好的一组是 SW1@60 的 -4.34pp。**M3 留出期通过 0/3。**
#
# ── 必须说清、且绝不能含糊的一点 ──
# SW1@60 用 **4.34pp 年化** 换来 **14.60pp 回撤改善**,按很多人的标准这是划算的。
# 但 3pp 这个阈值是我**在跑之前写死的**,现在放宽它就是事后调参 ——
# 这正是本项目 48 次判据里一次都没做过的事,本节也不做。
# 更要紧的是:**SW1 根本没走到留出期,它没有任何样本外证据。**
# 若要认真评估「4.34pp 换 14.60pp」这个取舍,必须**另开一节重新事前登记**,
# 把阈值、随机择时对照、留出期规则重新写死再跑,不能拿本节的训练期数字当结论。
#
# ── 事前预测 ──
# V1 **部分错。** 我预测「SW3 在训练期会被 M2 选出参数,但留出期过不了 M3(c)」——
#    前半错了,SW3 在训练期就没选出参数(三组的年化下降都在 6.30~8.13pp)。
# V2 **无法判定。** 我预测「SW1 是三个里最可能过 M3 的」。没有任何开关走到 M3,
#    这条预测无从证伪。descriptively SW1@60 确实是九组里训练期取舍最好的一组,
#    但那不构成对 V2 的确认 —— 不能把「方向像」当成「预测中了」。
# V3 ✓ 通过 M3 的数量 0 ∈ [0,1]。
# V4 ✓ 且比预测更彻底:预测「至少一个开关训练期失败」,实际**三个全失败**。
#
# ── 结论 ──
# **「用微盘自身的相对强弱做开关」这条路,在本节的判据下没有走通。**
# 它没有推翻 Codex 的 R12 结论(他用 510300 状态失败),而是把失败的范围扩大了:
# 换成针对性更强的微盘自身信号,同样换不来「只砍回撤不砍收益」。
# 三个开关的共同形态是:**回撤确实降了(最多 16.79pp),但收益的代价更大**
# (最少 4.34pp、最多 9.32pp)。
#
# 另一个值得记下的事实:基线在**留出期 2022–2026 的回撤只有 -27.78%**,
# 远小于训练期的 -56.38%。也就是说,微盘策略最需要开关的那段(2015、2018)
# 在训练期内,而留出期本身并不难 —— 这让「开关是否有用」这个问题
# 在本节的窗口切法下先天就不好回答。
# =============================================================================
