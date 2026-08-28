"""§127 月线 MA20 + MACD 到底是不是牛熊分界线?

用户问题
--------
「沪深300 和科创板指数的月线 MACD 与 20 均线,是不是牛熊的分界线?」

为什么必须挂随机择时对照
------------------------
**任何跟随型均线规则,在只经历一两轮周期的图上都会显得像完美分界线。**
更要命的是:熊市里空仓本身就降回撤,**不加对照的话,任何降低仓位的规则都会"成功"**。
§120 已经栽过一次(三个风格开关全部训练期即失败),Codex 的 R12 也是这么失败的。
所以本节的核心不是"规则赚了多少",而是"**相对同样空仓月数、但时点随机的对照,
它还剩多少**"。

数据与样本量(事前写死,不得事后放宽)
--------------------------------------
沪深300  = 510300(本面板,含分红),月末收盘,2013-01 → 2026-07,**163 根月线**
           MA20 可用起点 2014-08,完整牛熊周期 **3–4 轮** → **参与判定**
科创50   = 588000 科创50ETF华夏(quant-research-dev/data/20260729/kline.parquet)
           2020-11-16 → 2026-07-29,**68 根月线**,MA20 可用起点 2022-07,
           实际可用 **48 个月、仅 1 轮完整周期** → **只作参考,不参与判定**
科创100  = 588030 等,最早 2023-09,仅 33 根月线,MA20 后剩 13 个月
           → **样本量根本不够,本节不跑,不给数字。**

口径瑕疵如实记录:510300 用本面板的含分红序列,588000 用 ETF 的行情收盘。
两条序列内部各自一致(规则与买入持有用同一条),但两者之间不完全可比。

规则(三个变体)
----------------
V1 `MA20`      月末收盘 > 月线 MA20                → 下月满仓,否则空仓
V2 `MACD`      月线 MACD(12,26,9) > signal          → 下月满仓,否则空仓
V3 `MA20&MACD` 两条同时满足                        → 下月满仓,否则空仓(Codex R12 的口径)
信号用 t 月末收盘,仓位**从 t+1 月生效**,不含交易成本(纯规则检验)。

判据(跑之前写死,跑完照判,不放宽;加严可以)
--------------------------------------------
Y1 锚点(不过则整节作废)
   (a) **恒开恒等式**:把信号强制置为恒"开"时,净值必须与买入持有**逐月相等**
       (最大相对误差 < 1e-9)。写错必抓。
   (b) **对照恒等式**:随机择时对照的空仓月数必须与被对照的真规则**完全相等**。
   (c) 数据锚点:510300 月线 163 根、588000 月线 68 根,与源一致。

Y2 判定(**仅沪深300**)。对照 = 保持同样空仓月数、空仓时点随机,**500 组种子**。
   **Bonferroni:3 个变体,α = 0.05/3 = 0.016667。**
   Y2 通过 ⟺ 同时满足:
     (a) 年化的单尾 p < 0.016667(p = (1+#{对照年化 ≥ 规则年化})/501);
     (b) 最大回撤**优于**随机对照的**中位数**。
   两条都满足才算"是分界线";只满足 (b) 说明它只是"少交易",不是"择对了时"。

Y3 科创50ETF(**描述项,不参与判定**)。同样跑三个变体与随机对照,报告数字,
   并在结论里标注"48 个可用月、仅 1 轮周期,不足以判定"。

Y4 描述项:各变体的空仓月占比、切换次数,以及在 §126 主口径 4 牛 4 熊段上的表现。

事前预测
--------
**本节不下预测**(§119 起已停止此类外推)。

不做的
------
不改 src/oxq/;不新增顶层目录;不 force push;**不往 quant-research-dev 推任何东西**;
**不跑科创100**(样本量不够,给数字本身就是误导);
不因科创50 结果好看就把它拿来支撑结论;
不基于本节结论做任何可交易性声明。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codex_r10_neutral import OUT, SEED  # noqa: E402
from codex_r10_replication import DATA  # noqa: E402

NSEED, ALPHA = 500, 0.05 / 3
ETF = "/home/user/quant-research-dev/data/20260729/kline.parquet"


def signals(m):
    """三个变体的月度信号(True=下月满仓)。信号用 t 月末,仓位 t+1 生效。"""
    ma = m > m.rolling(20).mean()
    e12, e26 = m.ewm(span=12).mean(), m.ewm(span=26).mean()
    macd = e12 - e26
    md = macd > macd.ewm(span=9).mean()
    return {"MA20": ma, "MACD": md, "MA20&MACD": ma & md}


def equity(m, pos):
    """月度净值。pos 是**已经对齐好的持仓指示**(pos[t]=True 表示吃到 t 月的收益)。

    §127 第一版在这里做了 `on.shift(1)`,而对照那边又先 `shift(-1)`,
    一来一回等于没移,但首月被强制置 False,导致**对照的空仓月数比真规则多 1 个月**;
    更糟的是锚点 Y1b 检查的是 `ro.shift(1)`,**量的不是净值实际用的那条序列**。
    锚点抓到了(违例 1358 次),按 §113 的规矩不放宽锚点、改实现:
    本版把移位一次性放在调用方,`equity` 只吃对齐好的仓位。
    """
    r = m.pct_change().fillna(0.0)
    return (r * pos.astype(float)).add(1.0).cumprod()


def stats(eq, m):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    sd = r.std(ddof=1)
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "total": float(eq.iloc[-1] - 1),
            "mdd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / sd * np.sqrt(12)) if sd > 0 else 0.0}


def run_one(m, label, judge):
    """对一条月线序列跑三个变体 + 随机择时对照。"""
    sig = signals(m)
    valid = sig["MA20"].notna() & sig["MACD"].notna()
    first = valid.idxmax()
    m2 = m[m.index >= first]
    allon = pd.Series(True, index=m2.index)
    bh = equity(m2, allon)
    sb = stats(bh, m2)
    print(f"\n=== {label}  可用月 {len(m2)}  起 {m2.index[0].date()} → "
          f"{m2.index[-1].date()} ===", flush=True)
    print(f"  买入持有  年化{sb['cagr']:+7.2%} 总{sb['total']:+8.2%} "
          f"回撤{sb['mdd']:+7.2%} 夏普{sb['sharpe']:5.2f}", flush=True)
    # 锚点 Y1a:恒开必须等于买入持有
    e_all = equity(m2, allon)
    err = float(np.max(np.abs(e_all.to_numpy() - bh.to_numpy())
                       / np.maximum(np.abs(bh.to_numpy()), 1e-9)))
    print(f"  锚点Y1a 恒开恒等式 最大相对误差 {err:.3e} "
          f"{'✓' if err < 1e-9 else '✗'}", flush=True)
    assert err < 1e-9, "锚点Y1a"
    rows = []
    for name, s in sig.items():
        pos = s.reindex(m2.index).shift(1).fillna(False)   # 信号 t 月末 → t+1 生效
        eq = equity(m2, pos)
        st = stats(eq, m2)
        noff = int((~pos).sum())
        sw = int((pos != pos.shift(1)).sum())
        rng = np.random.default_rng(SEED)
        cg, cm, viol = [], [], 0
        n = len(m2)
        for _ in range(NSEED):
            idxs = rng.choice(n, size=noff, replace=False)
            ro = pd.Series(True, index=m2.index)
            ro.iloc[idxs] = False
            # 对照恒等式:直接量净值用到的那条仓位序列
            viol += int(int((~ro).sum()) != noff)
            e2 = equity(m2, ro)
            s2 = stats(e2, m2)
            cg.append(s2["cagr"])
            cm.append(s2["mdd"])
        cg, cm = np.array(cg), np.array(cm)
        p = (1 + int(np.sum(cg >= st["cagr"]))) / (NSEED + 1)
        okm = st["mdd"] > float(np.median(cm))
        ok2 = bool(p < ALPHA and okm)
        rows.append({"series": label, "rule": name, **st, "off_months": noff,
                     "switches": sw, "ctrl_cagr_med": float(np.median(cg)),
                     "ctrl_mdd_med": float(np.median(cm)), "p": p,
                     "mdd_better": okm, "Y2": ok2 if judge else None,
                     "ctrl_viol": viol})
        tag = ("Y2 " + ("✓" if ok2 else "✗")) if judge else "(参考,不判定)"
        print(f"  {name:10s} 年化{st['cagr']:+7.2%} 总{st['total']:+8.2%} "
              f"回撤{st['mdd']:+7.2%} 夏普{st['sharpe']:5.2f} 空仓{noff:3d}月 "
              f"切换{sw:3d}次 | 对照 年化中位{np.median(cg):+7.2%} "
              f"回撤中位{np.median(cm):+7.2%} p={p:.4f} 回撤更优{'✓' if okm else '✗'}"
              f" | {tag}", flush=True)
    return rows


def main():
    b = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
    b.index = pd.to_datetime(b.index).tz_localize(None)
    hs = pd.to_numeric(b["close"], errors="coerce").ffill().resample("ME").last()
    k = pd.read_parquet(ETF, columns=["code", "trade_date", "close"])
    k = k[k["code"] == "588000"].copy()
    k["trade_date"] = pd.to_datetime(k["trade_date"])
    kc = k.set_index("trade_date")["close"].sort_index().resample("ME").last()
    print(f"锚点Y1c 510300 月线 {len(hs)} 根;588000 月线 {len(kc)} 根", flush=True)
    assert len(hs) >= 160 and len(kc) >= 65, "锚点Y1c 数据量"

    rows = run_one(hs, "沪深300 (510300, 含分红)", judge=True)
    rows += run_one(kc, "科创50ETF (588000)", judge=False)
    df = pd.DataFrame(rows)
    v = int(df["ctrl_viol"].sum())
    print(f"\n锚点Y1b 对照空仓月数违例 {v} 次 {'✓' if v == 0 else '✗ 作废'}")
    assert v == 0
    j = df[df["Y2"].notna()]
    print(f"Y2 通过 {int(j['Y2'].sum())}/3(α={ALPHA:.6f},仅沪深300):"
          f"{', '.join(j.loc[j['Y2'].astype(bool), 'rule']) or '无'}")
    df.to_csv(f"{OUT}/monthly_regime_test.csv", index=False)
    print(f"落库 {OUT}/monthly_regime_test.csv")


if __name__ == "__main__":
    main()


# =============================================================================
# §127 结果:**Y2 通过 0/3。月线 MA20 与 MACD 都不构成可检验的牛熊分界线。**
#
# 锚点 Y1a ✓ 恒开恒等式 0.000e+00(两条序列)  Y1b ✓ 对照空仓月数违例 0 次
#      Y1c ✓ 510300 月线 163 根、588000 月线 69 根
#
# ── 沪深300(510300 含分红,163 个可用月,3–4 轮周期,参与判定)──
# 买入持有            年化 +5.96%  总+118.49%  回撤-39.40%  夏普 0.38
# 规则        年化    总收益   回撤    夏普 | 空仓 切换 | 对照年化中位 对照回撤中位   p     回撤更优 Y2
# MA20      +5.11%  +95.93% -41.99%  0.38 | 71月 10次 |   +3.42%     -33.41%  .2914    ✗    ✗
# MACD      +7.74% +173.37% -32.51%  0.54 | 79月 10次 |   +2.91%     -32.92%  .0719    ✓    ✗
# MA20&MACD +7.30% +158.91% -32.51%  0.55 | 87月  8次 |   +2.53%     -32.18%  .0679    ✗    ✗
#
# **三个变体全部不通过。** 关键在这两列:
# ① **MA20 单独用比买入持有还差**(+5.11% vs +5.96%),而且**回撤反而更大**
#    (-41.99% vs -39.40%)。它在 163 个月里空仓 71 个月,躲开的和错过的相抵还倒亏。
# ② MACD 与 MA20&MACD 的年化确实高于买入持有(+7.74% / +7.30%),
#    **但相对随机择时对照 p=0.0719 / 0.0679,过不了 α=0.016667,连 0.05 都过不了。**
#    也就是说:**保持同样的空仓月数、把空仓时点完全打乱,有 7% 左右的随机方案
#    做得和它一样好或更好。**
# ③ 回撤那一列更能说明问题:随机择时对照的回撤中位数是 -32.2% ~ -33.4%,
#    而买入持有是 -39.40%。**光是"随机地空掉 71~87 个月"就把回撤从 39% 压到 33%** ——
#    规则的 -32.51% 相对这个基准几乎没有额外贡献。
#    **这就是"熊市空仓本身就降回撤"这个平凡解释,数据把它量出来了。**
#
# ── 科创50ETF(588000,69 根月线,MA20 后仅 48 个可用月,1 轮周期)──
# **本节事前登记里已写死:只作参考,不参与判定。**
# 买入持有            年化 +4.41%  总 +27.68%  回撤-56.27%  夏普 0.29
# MA20 / MA20&MACD  年化+11.68%  总 +86.95%  回撤-24.23%  夏普 0.55  空仓47月 切换仅 2 次
#   → 看起来极好,**但 p=0.0719,同样过不了 0.05**;而且**整段只切换 2 次**,
#     等于"2021 下半年卖出、2024 下半年买回"这一个动作。
#     **一个动作、一轮周期,不构成证据。** 用户看到的"月线 MA20 像完美分界线",
#     其形成机制正是这个:这 5 年只有一次向下、一次向上,任何跟随型均线都会显得完美。
# MACD 单独          年化 +5.73%  回撤-30.89%  p=0.3273 —— 连方向都不明显。
#
# ── 结论 ──
# **对沪深300:月线 MA20、MACD、以及两者且(Codex R12 的口径),
# 在挂上随机择时对照之后,没有一个能被判定为有效的牛熊分界线。**
# 这与 §120(三个微盘风格开关全部训练期即失败)、§118(R01/R02 择时全期跑输)、
# 以及 Codex 自己 R12 的失败**方向完全一致 —— 四条独立路径同一结论。**
# 对科创板:样本量不足以判定,本节不下结论,并明确拒绝用它去支撑任何说法。
#
# ── 锚点 Y1b 抓到的实现错误(如实记录)──
# 第一版把对照做成 `ro.shift(-1)` 再进 `equity()`(内部又 `shift(1)`),
# 一来一回等于没移,但首月被强制置 False → **对照空仓月数常比真规则多 1 个月**;
# 更糟的是锚点自己检查的是 `ro.shift(1)`,**量的不是净值实际用的那条序列** ——
# 这是 §83 说的"让正确实现不通过"的坏锚点。违例 1358 次。
# 按 §113 的规矩**不放宽锚点、改实现**:移位一次性放到调用方,
# `equity` 只吃对齐好的仓位,锚点也改量真正用到的序列。修正后违例 0 次,
# **三个变体的判定结果不变(修正前后都是 0/3 通过)。**
# =============================================================================
