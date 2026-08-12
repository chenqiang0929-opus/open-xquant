"""抽样方差:一个人只做 N 笔交易,会经历什么

═══ 为什么这是对用户问题最直接的回答 ═══
用户问:"为什么那么多投资者用陶博士/欧奈尔方法就成功了,我们跑出来这么差?"

我们的系统(四十二节 rule A):**胜率 18.6%、盈亏比 6.94、净期望 +4.61%/笔**。
这是个彩票型分布 —— **跑 70,124 笔得到的是期望值;
一个人一生跑 30-50 笔,得到的是一次抽样。**

在这种分布下,结果几乎完全取决于有没有碰上那一两笔大赢家。
本脚本把"有多少比例的人会成功"直接算出来,
**把"幸存者偏差"这句套话变成数字。**

═══ 方法 ═══
用真实逐笔收益(不是正态假设)有放回抽样,N=20/50/100/200,各 10,000 次,
按等额资金连续复利。分两组:全样本 vs **仅牛市年入场**(2015/2020/2025)——
因为大部分散户只在牛市活跃。

═══ 一个必须说明的保守方向 ═══
自助抽样假设各笔**独立**,而真实交易在时间上聚集(同一波行情里多笔同涨同跌),
**所以真实方差比这里算出来的更大**,成功者比例和失败者比例都会更极端。
本脚本的结论是保守的。

本检验**没有判据** —— 它是描述性的,目的是量化方差,不是判定优劣。
"""
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
COST = 0.003
NS = [20, 50, 100, 200]
N_SIM = 10_000
SEED = 20260811
BULL = (2015, 2020, 2025)

t0 = time.time()
rets = np.load(f"{SP}/oneil_baseline_trade_rets.npy")     # 基线逐笔毛收益
net = rets - COST
print(f"基线逐笔收益 {len(net):,} 笔")
print(f"  胜率 {(net>0).mean():.1%}   均盈 {net[net>0].mean():+.1%}   "
      f"均亏 {net[net<=0].mean():+.1%}   净期望 **{net.mean():+.2%}/笔**")
print(f"  分位:1% {np.quantile(net,.01):+.1%}   25% {np.quantile(net,.25):+.1%}   "
      f"中位 {np.median(net):+.1%}   75% {np.quantile(net,.75):+.1%}   "
      f"99% {np.quantile(net,.99):+.1%}   最大 {net.max():+.0%}")

# 牛市年子样本(按入场年份)
flags = pd.read_csv(f"{SP}/oneil_event_flags.csv")
print(f"\n事件标记 {len(flags):,} 行(与逐笔收益 {len(net):,} 笔略有差异:"
      f"入场日无价的事件被跳过)")


def simulate(pool, label):
    rng = np.random.default_rng(SEED)
    print(f"\n{'='*104}\n{label}  ({len(pool):,} 笔可抽)\n{'='*104}")
    print(f"{'交易笔数N':<10}{'中位倍数':>10}{'25%':>9}{'75%':>9}{'亏损比例':>10}"
          f"{'>2倍':>8}{'>3倍':>8}{'>10倍':>8}{'最好1%':>10}")
    out = []
    for N in NS:
        draws = rng.choice(pool, size=(N_SIM, N), replace=True)
        mult = np.prod(1 + draws, axis=1)
        out.append({"N": N, "中位": np.median(mult),
                    "q25": np.quantile(mult, .25), "q75": np.quantile(mult, .75),
                    "亏损比例": (mult < 1).mean(), ">2倍": (mult > 2).mean(),
                    ">3倍": (mult > 3).mean(), ">10倍": (mult > 10).mean(),
                    "最好1%": np.quantile(mult, .99)})
        r = out[-1]
        print(f"{N:<10}{r['中位']:>10.2f}{r['q25']:>9.2f}{r['q75']:>9.2f}"
              f"{r['亏损比例']:>10.1%}{r['>2倍']:>8.1%}{r['>3倍']:>8.1%}"
              f"{r['>10倍']:>8.1%}{r['最好1%']:>10.1f}")
    return pd.DataFrame(out).assign(样本=label)


res = [simulate(net, "全样本(2013-2026 所有突破)")]

# 牛市年:用事件年份对齐(逐笔收益按 groupby(code) 顺序生成,此处用年份比例近似不可行)
# → 改为重新按年份切分:利用 flags 里的 year 与逐笔顺序不一致,故仅在事件层面统计后
#   用"牛市年事件占比"做加权抽样不严谨。这里直接说明并跳过精确子样本。
bull_share = flags.year.isin(BULL).mean()
print(f"\n{'='*104}")
print(f"牛市年(2015/2020/2025)事件占全部的 {bull_share:.1%}")
print("**说明**:逐笔收益数组按 groupby(code) 顺序生成,与事件表的行序不对应,")
print("无法直接切出牛市子样本。四十二节已用同一套数据分年报过:")
print("  牛市年净期望 **+7.34%/笔**(rule A),熊市年 **-4.79%/笔**")
print("下面用这两个期望值做参数化对照(保持实际收益分布形状,仅平移到对应期望):")
print(f"{'='*104}")

for lbl, target in (("牛市年(期望 +7.34%/笔)", 0.0734), ("熊市年(期望 -4.79%/笔)", -0.0479)):
    shifted = net + (target - net.mean())      # 平移到目标期望,保持分布形状
    res.append(simulate(shifted, lbl))

R = pd.concat(res, ignore_index=True)
R.to_csv(f"{SP}/oneil_sampling_variance.csv", index=False)

print(f"\n{'='*104}\n结论(描述性,无判据)\n{'='*104}")
full = R[(R.样本.str.startswith("全样本")) & (R.N == 50)].iloc[0]
print(f"  全样本、做 50 笔交易的人:")
print(f"    中位数 {full['中位']:.2f} 倍   **{full['亏损比例']:.1%} 的人亏钱**   "
      f"**{full['>3倍']:.1%} 的人赚到 3 倍以上**   最好的 1% 拿到 {full['最好1%']:.1f} 倍")
print(f"\n  **注意:自助抽样假设各笔独立,真实交易在时间上聚集,")
print(f"  所以真实的方差比这里更大 —— 成功者和失败者都会更极端。本结论是保守的。**")
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: oneil_sampling_variance.csv")
