"""用户观察池(2022-08 前后上市)的信号表与 50 只候选筛选

═══ 用户的要求 ═══
「我会把 2022-8 月份上市的股票池作为观察标的,然后我需要你帮我出信号,
比如:RPS 的值、20周线和60周线是否金叉、成交量等等」
「然后通过一系列的规则帮我选出 50 只候选标的」

═══ 必须先声明的一件事(不影响照做,但必须写在产出里) ═══
本研究 §57 / §62 / §63 / §64 已四次独立确认:
**趋势、RPS、量能作为「选股因子」是负 alpha。**
§62 更直接:把胜率提到全研究最高时,组合年化转负。
**因此「按规则排序选出的 50 只」在期望上会跑输「同池随机 50 只」。**

**所以本脚本同时输出两份名单:**
  A. 规则筛选 50 只(用户要的)
  B. 同池随机 50 只(对照,固定种子可复现)
两份并列,差别由用户自己判断,**脚本不替用户做取舍**。

═══ 信号定义(全部按通行口径,不调参) ═══
  20 周线   MA100(交易日)
  60 周线   MA300(交易日)
  多头排列  MA100 > MA300
  金叉      MA100 上穿 MA300 发生在最近 60 个交易日内
  站上20周线 收盘 > MA100
  RPS_N     个股 N 日涨幅在**全市场 5,232 只**中的百分位 × 100(欧奈尔口径)
            N ∈ {50, 120, 250}
  量比      近 20 日均量 / 近 60 日均量
  距新高    收盘 / 250 日最高价 − 1

═══ 筛选阶梯(用户方案,原样实现) ═══
  ① 在池内且面板有数据
  ② 多头排列:MA100 > MA300
  ③ 站上 20 周线:收盘 > MA100
  ④ 量能不萎缩:量比 ≥ 1.0
  ⑤ 按 RPS250 降序取前 50

**每一层都报出剩余只数**,便于看清是哪一层砍掉了多少。

═══ 数据口径 ═══
  面板最后交易日 **2026-08-03**;用户 CSV 导出日 2026-08-14。
  **信号是 2026-08-03 的,不是 08-14 的** —— 相差约 8 个交易日,使用时须知。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
POOL_CSV = ("/root/.claude/uploads/95a7873e-a420-5ffc-8d4d-fc8fba4ec34e/"
            "8b6acb64-___20260814.csv")
OUT = f"{SP}/pool_signals_20260814.csv"
NPICK, SEED, XGOLD = 50, 20260814, 60

t0 = time.time()
pool = pd.read_csv(POOL_CSV, encoding="gbk", dtype=str)
pool.columns = [c.strip() for c in pool.columns]
pool["code"] = pool["代码"].str.zfill(6)
pool = pool.rename(columns={"名称(646)": "名称", "上市日期": "上市日",
                            "一二级行业": "行业", "细分行业": "细分"})
print(f"用户池 {len(pool)} 只  ({time.time()-t0:.0f}s)")

cl, vo = {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "volume"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    vo[k] = pd.to_numeric(x["volume"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
VO = pd.DataFrame(vo).set_axis(CL.index)
CL = CL.where(CL > 0)
NT, NS = CL.shape
print(f"面板 {CL.shape}  最后交易日 {CL.index[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

CLf = CL.ffill()
last = CLf.iloc[-1]
MA100 = CLf.rolling(100, min_periods=100).mean()
MA300 = CLf.rolling(300, min_periods=300).mean()
m100, m300 = MA100.iloc[-1], MA300.iloc[-1]
prev_up = (MA100 > MA300).iloc[-XGOLD - 1:-1]
gold = (m100 > m300) & (~prev_up.iloc[0].fillna(False))     # 60 日内上穿
hi250 = CLf.rolling(250, min_periods=100).max().iloc[-1]
v20 = VO.rolling(20, min_periods=10).mean().iloc[-1]
v60 = VO.rolling(60, min_periods=30).mean().iloc[-1]

# RPS:全市场百分位(只对当日有效价的标的排名)
alive = last.notna() & (last > 0)
RPS = {}
for n in (50, 120, 250):
    r = last / CLf.shift(n).iloc[-1] - 1
    r = r.where(alive)
    RPS[n] = r.rank(pct=True) * 100
print(f"信号计算完成  ({time.time()-t0:.0f}s)")

S = pool[["code", "名称", "上市日", "行业", "细分"]].copy()
S["在面板"] = S["code"].isin(CL.columns)
idxc = S["code"].where(S["在面板"])


def pick(series):
    return idxc.map(series).astype(float)


S["收盘"] = pick(last)
S["MA100"] = pick(m100)
S["MA300"] = pick(m300)
S["多头排列"] = (S["MA100"] > S["MA300"])
S["近60日金叉"] = idxc.map(gold).fillna(False).astype(bool)
S["站上20周线"] = (S["收盘"] > S["MA100"])
S["RPS50"] = pick(RPS[50])
S["RPS120"] = pick(RPS[120])
S["RPS250"] = pick(RPS[250])
S["量比20/60"] = pick(v20) / pick(v60)
S["距250日新高%"] = (S["收盘"] / pick(hi250) - 1) * 100

miss = (~S["在面板"]).sum()
print(f"\n面板覆盖 {len(S)-miss}/{len(S)},缺 {miss} 只(多为 2026 年新上市,面板未收)")
if miss:
    print("  缺:", ", ".join(S.loc[~S["在面板"], "code"].tolist()))

print(f"\n{'='*90}\n筛选阶梯(用户方案原样实现)\n{'='*90}")
f1 = S["在面板"] & S["收盘"].notna()
print(f"  ① 池内且面板有数据                {f1.sum():>4} 只")
f2 = f1 & S["多头排列"].fillna(False)
print(f"  ② 多头排列 MA100 > MA300          {f2.sum():>4} 只   (砍掉 {f1.sum()-f2.sum()})")
f3 = f2 & S["站上20周线"].fillna(False)
print(f"  ③ 收盘 > 20 周线                  {f3.sum():>4} 只   (砍掉 {f2.sum()-f3.sum()})")
f4 = f3 & (S["量比20/60"] >= 1.0)
print(f"  ④ 量比 ≥ 1.0                      {f4.sum():>4} 只   (砍掉 {f3.sum()-f4.sum()})")
cand = S[f4].sort_values("RPS250", ascending=False)
sel = cand.head(NPICK)
print(f"  ⑤ 按 RPS250 降序取前 {NPICK}          {len(sel):>4} 只")
if len(sel) < NPICK:
    print(f"  **⚠️ 不足 {NPICK} 只** —— 满足全部硬条件的只有 {len(cand)} 只。")

rng = np.random.default_rng(SEED)
univ = S.index[f1].to_numpy()
ctrl_idx = rng.choice(univ, min(NPICK, len(univ)), replace=False)
ctrl = S.loc[ctrl_idx]

S["规则选中"] = S.index.isin(sel.index)
S["随机对照"] = S.index.isin(ctrl.index)

print(f"\n{'='*110}\nA. 规则筛选 {len(sel)} 只(用户要的)\n{'='*110}")
print(f"{'#':<4}{'代码':<9}{'名称':<11}{'细分行业':<16}{'收盘':>9}"
      f"{'RPS250':>8}{'RPS120':>8}{'量比':>7}{'距新高%':>9}{'金叉':>6}")
for i, (_, r) in enumerate(sel.iterrows(), 1):
    print(f"{i:<4}{r['code']:<9}{str(r['名称'])[:9]:<11}{str(r['细分'])[:14]:<16}"
          f"{r['收盘']:>9.2f}{r['RPS250']:>8.1f}{r['RPS120']:>8.1f}"
          f"{r['量比20/60']:>7.2f}{r['距250日新高%']:>9.1f}"
          f"{'✓' if r['近60日金叉'] else '':>6}")

print(f"\n{'='*110}\nB. 同池随机 {len(ctrl)} 只(对照 · 种子 {SEED} 可复现)\n{'='*110}")
print("  " + "  ".join(f"{r['code']} {str(r['名称'])[:6]}" for _, r in ctrl.iterrows()))

print(f"\n{'='*90}\n两份名单的画像对比\n{'='*90}")
print(f"{'指标':<16}{'规则 50 只':>14}{'随机 50 只':>14}{'全池':>14}")
for c in ("RPS250", "RPS120", "量比20/60", "距250日新高%"):
    print(f"{c:<16}{sel[c].mean():>14.1f}{ctrl[c].mean():>14.1f}{S.loc[f1, c].mean():>14.1f}")
print(f"{'多头排列占比':<16}{sel['多头排列'].mean():>13.0%}"
      f"{ctrl['多头排列'].mean():>14.0%}{S.loc[f1,'多头排列'].mean():>14.0%}")

print(f"\n{'='*90}\n必须一起说的话\n{'='*90}")
print("  · 信号是 **2026-08-03** 的(面板最后交易日),用户 CSV 导出于 08-14,相差约 8 个交易日。")
print("  · 本研究 §57/§62/§63/§64 四次确认:RPS/趋势/量能作为**选股因子**是负 alpha。")
print("    **按期望,A 名单会跑输 B 名单。** 两份并列给出,取舍由用户决定。")
print("  · 用户池是「2022-08 前后上市」的**固定日期**池:这些标的现已上市约 4 年,")
print("    已**超出** §69 证实的 [1,3) 年窗口 —— 那个 2.73 倍的右尾优势不适用于本池。")

S.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n→ {OUT}   ({time.time()-t0:.0f}s)")
