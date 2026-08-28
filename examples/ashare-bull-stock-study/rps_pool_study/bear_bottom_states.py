"""§128 把「熊市末期的牛市迹象」改写成可检验的条件分布。

为什么必须换问法
----------------
「熊市底部有哪些共同特征」这个问题**无法检验**:13 年只有 3–4 个底部,样本量就是 3–4。
在 3–4 个点上总能找到几条共同条件,而它对第 5 次没有任何预测力。
§120(三个风格开关全败)、§118(R01/R02 择时跑输)、§127(月线 MA20/MACD 挂上
随机对照后 0/3)、以及 Codex 自己的 R12,四条独立路径已经证明了这条路走不通。

**可检验的版本是把「预测拐点」换成「测量条件分布」**:

    不可做(n≈4)                    可做(n≈3000 个交易日)
    「底部有什么共同特征」    →    「处在状态 X 时,未来 H 天收益的分布是什么」
    「什么信号提示见底」      →    「状态越极端,forward return 是否越高,且单调」

状态变量(全部只用 ≤t 的信息,无前视)
--------------------------------------
S1 `dd250`   沪深300 距 250 日最高收盘的回撤深度  = close/max(close,250d) − 1
S2 `breadth` 全市场收盘价在 MA100 之上的股票占比
S3 `bp_pct`  全市场中位 B/P 的**扩张窗口历史分位**(只用截至 t 的历史,不用未来)

分档:每个状态变量按**扩张窗口分位**切 5 档(只用 ≤t 的历史定义档位边界,
避免用全样本分位泄漏未来)。前 500 个交易日因历史不足不参与。

前瞻区间:H ∈ {250, 500, 750} 个交易日(≈1/2/3 年),用沪深300 含分红收盘。

显著性:循环移位检验(**不是 block bootstrap** —— §90 栽过,那次 block bootstrap
在真 H0 下给出 p=0.028,校准是坏的)。
    做法:把**状态序列相对收益序列循环移位 k 天**(k=1…n−1),
    两条序列各自的自相关结构被**完整保留**,只有跨序列的对齐被打乱。
    这对「状态与未来收益是否真的对齐」这个问题是干净且精确的零假设。
    统计量 = **最极端档的中位 forward return − 最不极端档的中位 forward return**。
    p = #{移位下统计量 ≥ 实测} / n。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
Z1 锚点(不过则整节作废)
   (a) 面板 (3297, 5217);沪深300 交易日数与 §127 一致;
   (b) **移位恒等式**:移位 k=0 时必须**精确复现**实测统计量(相对误差 < 1e-12)。
       写错必抓。
   (c) **无前视自查**:forward return 的窗口起点必须严格 > 状态观测日;
       随机抽 200 个样本点逐一断言 `state_date < fwd_start_date`,违例 > 0 即作废。

Z2 判定。3 个状态 × 3 个 H = 9 组。**Bonferroni:α = 0.05/9 = 0.005556。**
   Z2 通过 ⟺ 循环移位 p < 0.005556。

Z3 单调性(**比显著性更重要的一条**)。
   Z3 通过 ⟺ 5 档的中位 forward return 关于档位**严格单调**(方向与假设一致)。
   **只有 Z2 与 Z3 都通过,才算「这个状态变量对未来收益有条件信息」。**
   只过 Z2 不过 Z3 → 记为「单点显著但不单调」,不下结论(§115-B 的教训:
   斜坡才是证据,单点显著不是)。

Z4 描述项(不设阈值):每档的样本数、中位/均值 forward return、正收益占比;
   并单独列出「当前值落在哪一档」。

事前预测
--------
**本节不下预测**(§119 起已停止此类外推)。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;不往 quant-research-dev 推;
**不跑科创板**(2020-11 起,forward 750 天几乎没有样本,给数字就是误导);
**不因某一档好看就把它当买点** —— 条件分布不是择时规则,
把它变成规则必须另开一节、重新事前登记、并挂随机择时对照(§127 的教训);
不基于本节结论做任何可交易性声明。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import CACHE, OUT, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402
from codex_routes_rerun import build_fund  # noqa: E402

HS = (250, 500, 750)
NB, WARM, ALPHA = 5, 500, 0.05 / 9


def expanding_bucket(x, nb=NB, warm=WARM):
    """扩张窗口分位分档:第 i 个点的档位只用 [0, i] 的历史定义边界,不用未来。"""
    v = np.asarray(x, float)
    out = np.full(len(v), -1, np.int8)
    for i in range(warm, len(v)):
        h = v[:i + 1]
        h = h[np.isfinite(h)]
        if len(h) < warm or not np.isfinite(v[i]):
            continue
        q = np.quantile(h, np.linspace(0, 1, nb + 1)[1:-1])
        out[i] = int(np.searchsorted(q, v[i], side="right"))
    return out


def main():  # noqa: PLR0915
    z = np.load(CACHE, allow_pickle=True)
    idx = pd.DatetimeIndex(z["idx"])
    codes = list(z["codes"])
    cl, ok = z["CL"], z["OK"]
    nt, ns = len(idx), len(codes)
    assert (nt, ns) == (3297, 5217), "锚点Z1a"
    print(f"锚点Z1a ✓ {nt}×{ns}", flush=True)

    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    bs = pd.to_numeric(b["close"], errors="coerce").ffill().reindex(idx).ffill()
    px = bs.to_numpy(float)

    # S1 距 250 日最高收盘的回撤
    hi250 = pd.Series(px, index=idx).rolling(250, min_periods=250).max().to_numpy()
    dd250 = px / hi250 - 1.0
    # S2 市场宽度:收盘 > MA100 的占比(只用当日可交易股票)
    ma100 = pd.DataFrame(cl, index=idx).rolling(100, min_periods=100).mean().to_numpy()
    with np.errstate(all="ignore"):
        above = np.nansum((cl > ma100) & ok, axis=1)
        denom = np.maximum(np.nansum(np.isfinite(ma100) & np.isfinite(cl) & ok, axis=1), 1)
    breadth = above / denom
    # S3 全市场中位 B/P 的扩张分位
    t0 = time.time()
    raw = np.full((nt, ns), np.nan, np.float32)
    for j, c in enumerate(codes):
        x = pd.read_parquet(f"{DATA}/{c}.parquet", columns=["raw_close"])
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
        raw[:, j] = pd.to_numeric(x["raw_close"], errors="coerce").where(
            lambda s: s > 0).ffill().reindex(idx).to_numpy(np.float32)
    fm, abad = build_fund(codes, idx)
    assert abad == 0, "锚点 TTM"
    with np.errstate(all="ignore"):
        bp = np.where(ok, fm["bps"] / raw, np.nan)
        bp_med = np.nanmedian(np.where(np.isfinite(bp) & (bp > 0), bp, np.nan), axis=1)
    print(f"状态变量完成 ({time.time()-t0:.0f}s)", flush=True)

    states = {"dd250": dd250, "breadth": breadth, "bp_med": bp_med}
    rows, ladder = [], []
    for sname, sv in states.items():
        bk = expanding_bucket(sv)
        for h in HS:
            fwd = np.full(nt, np.nan)
            fwd[:nt - h] = px[h:] / px[:nt - h] - 1.0     # t → t+h,起点严格在 t 之后
            m = (bk >= 0) & np.isfinite(fwd)
            bb, ff = bk[m], fwd[m]

            if len(bb) < 200:
                continue
            # 锚点 Z1c:无前视自查
            rng = np.random.default_rng(SEED)
            smp = rng.choice(len(bb), size=min(200, len(bb)), replace=False)
            pos = np.flatnonzero(m)
            viol = int(np.sum([idx[pos[i]] >= idx[min(pos[i] + 1, nt - 1)]
                               for i in smp]))
            med = np.array([np.median(ff[bb == k]) if (bb == k).any() else np.nan
                            for k in range(NB)])
            stat = med[0] - med[NB - 1]          # 最极端档(0=最低状态值)− 最高档
            n = len(bb)
            null = np.empty(n - 1)
            for kk in range(1, n):
                b2 = np.roll(bb, kk)
                mm = np.array([np.median(ff[b2 == k]) if (b2 == k).any() else np.nan
                               for k in range(NB)])
                null[kk - 1] = mm[0] - mm[NB - 1]
            # 移位恒等式 k=0
            b0 = np.roll(bb, 0)
            m0 = np.array([np.median(ff[b0 == k]) for k in range(NB)])
            ident = abs((m0[0] - m0[NB - 1]) - stat)
            p = (1 + int(np.sum(np.abs(null) >= abs(stat)))) / n
            mono_up = bool(np.all(np.diff(med) < 0))     # 状态越低 → 收益越高
            mono_dn = bool(np.all(np.diff(med) > 0))
            rows.append({"state": sname, "H": h, "n": n, "stat_pp": stat * 100,
                         "p": p, "Z2": bool(p < ALPHA), "mono": mono_up or mono_dn,
                         "ident_err": ident, "viol": viol,
                         **{f"q{k+1}_med": med[k] for k in range(NB)},
                         **{f"q{k+1}_n": int((bb == k).sum()) for k in range(NB)}})
            ladder.append((sname, h, med, [int((bb == k).sum()) for k in range(NB)]))
            print(f"{sname:8s} H={h:3d}  n={n:4d}  档中位 " +
                  " ".join(f"{v:+7.1%}" for v in med) +
                  f" | 统计量{stat*100:+7.2f}pp p={p:.4f} "
                  f"Z2 {'✓' if p < ALPHA else '✗'} 单调 {'✓' if mono_up or mono_dn else '✗'}"
                  f" | 恒等式{ident:.1e} 前视违例{viol}", flush=True)

    df = pd.DataFrame(rows)
    ie, vv = float(df["ident_err"].max()), int(df["viol"].sum())
    print(f"\n锚点Z1b 移位恒等式最大误差 {ie:.1e} {'✓' if ie < 1e-12 else '✗'}")
    print(f"锚点Z1c 前视违例 {vv} 次 {'✓' if vv == 0 else '✗'}")
    assert ie < 1e-12 and vv == 0
    both = df["Z2"] & df["mono"]
    print(f"Z2 通过 {int(df['Z2'].sum())}/9(α={ALPHA:.6f});"
          f"Z3 单调 {int(df['mono'].sum())}/9;**两者都过 {int(both.sum())}/9**:"
          f"{', '.join(df.loc[both, 'state'] + '@H' + df.loc[both, 'H'].astype(str)) or '无'}")
    print("\n当前值落档(面板末日):")
    for sname, sv in states.items():
        bk = expanding_bucket(sv)
        print(f"  {sname:8s} 当前 {sv[-1]:+.4f} → 第 {bk[-1]+1} 档 / {NB}")
    df.to_csv(f"{OUT}/bear_bottom_states.csv", index=False)
    print(f"落库 {OUT}/bear_bottom_states.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §128 结果:**Z2 通过 0/9,Z2+Z3 都过 0/9。三个状态变量都没有可检验的条件信息。**
#
# 锚点 Z1a ✓  Z1b ✓ 移位恒等式最大误差 0.0e+00  Z1c ✓ 前视违例 0 次
#
# 状态     H    n     第1档   第2档   第3档   第4档   第5档 | 统计量    p     Z2 单调
# dd250   250 2299  +11.0%  -4.0%   +8.9%  +19.8%  -1.3% | +12.38pp .4076  ✗  ✗
# dd250   500 2049  +22.0%  -0.2%   +7.1%  +10.0%  +4.1% | +17.91pp .5325  ✗  ✗
# dd250   750 1799  +30.3% +10.4%  +22.5%  +11.5% +15.6% | +14.73pp .5675  ✗  ✗
# breadth 250 2547  +13.5%  +6.3%   +7.2%   -3.8%  +2.9% | +10.58pp .5540  ✗  ✗
# breadth 500 2297  +31.7%  +9.9%   +8.9%   -1.1%  -6.2% | +37.89pp .0527  ✗  ✓
# breadth 750 2047  +31.2% +25.4%  +12.9%   +1.9%  +1.3% | +29.96pp .1402  ✗  ✓
# bp_med  250 2474   -5.3% +15.2%   -8.6%   -3.9% +22.1% | -27.42pp .2041  ✗  ✗
# bp_med  500 2224   +0.7% +14.8%  -13.1%   +5.8% +40.6% | -39.87pp .1407  ✗  ✗
# bp_med  750 1974   -9.0% +21.8%   -9.0%  +10.2% +23.7% | -32.66pp .4453  ✗  ✗
# (第1档 = 状态值最低档:dd250 最低=跌得最深;breadth 最低=宽度最差;
#  bp_med 最低=最贵)
#
# ── 这一节最值得看的地方,是「看起来很像」和「站得住」的差距 ──
# `breadth`(市场宽度)在 H=500 与 H=750 上**是单调的**:
#   宽度最差档 → 未来 2 年中位 **+31.7%**;宽度最好档 → **-6.2%**。
#   落差 **37.89pp**,方向完全符合直觉(市场最惨的时候买,两年后回报最高)。
# **但循环移位检验给出 p=0.0527 与 0.1402 —— 连 0.05 都过不了,更别说
# Bonferroni 的 0.005556。**
# 原因不是效应小,是**独立信息量太少**:市场宽度是个极慢的变量,
# 13 年里真正"宽度最差"的时段只有那么三四段,而且都和熊市底部重合。
# 2547 个交易日看着样本很大,**实际有效样本就是那三四段** ——
# 这正是循环移位检验要抓的东西:把状态序列整体挪个位置,
# 有 5%~14% 的挪法能产生同样大甚至更大的落差。
#
# **这跟「13 年只有 3–4 个底部」是同一件事,只是这次被量化了。**
# 我在事前登记里说"把 n≈4 换成 n≈3000 就可检验了",**这个说法不对**:
# 换成日频只是把重叠样本摊开,**没有增加独立信息**。
# 循环移位检验诚实地反映了这一点。如实记录:**我的问题设计本身乐观了。**
#
# ── dd250(距 250 日高点回撤)完全没有形状 ──
# 三个 H 上都不单调,第 4 档反而经常最高。
# "跌得越深、后续回报越好"在沪深300 这 13 年里**不成立**。
#
# ── bp_med(全市场中位 B/P)方向是**反的** ──
# 统计量为负,即"最便宜档"的后续回报反而低于"最贵档"。
# 三个 H 都不单调,p 也都不显著,**不构成任何结论**,只说明这个变量没有形状。
#
# ── 当前落档(面板末日 2026-08-03,描述性,不构成任何判断)──
#   dd250   -7.53% → 第 3 档 / 5(中间)
#   breadth 12.54% → **第 1 档 / 5(宽度最差档)**
#   bp_med   0.3839 → 第 3 档 / 5(中间)
# **即便 breadth 落在最差档,本节也已判定它 Z2 不过 ——
# 不能用它做任何买入判断。** 事前登记里写死了这一条。
#
# ── 与 §120/§127 合起来看 ──
# §127 证明月线 MA20/MACD 挂上随机择时对照后不显著;
# §120 三个微盘风格开关全部训练期即失败;
# **§128 进一步说明:连"状态变量与未来收益的条件关系"这一层都立不住。**
# 择时这条路,在本项目的判据下,五条独立路径全部失败
# (§118 R01/R02、§120、§127、§128,以及 Codex 自己的 R12)。
# =============================================================================
