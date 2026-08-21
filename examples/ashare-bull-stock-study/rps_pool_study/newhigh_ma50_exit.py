"""第一〇四节:首次新高后「10周线不破就持有」—— 用户的持有规则(事前登记)

═══ 用户的主张(原话)═══
> **「新高之后,10 周线不破就可以一直持有;跌破 10 周线,趋势就结束。」**

**主张有两半,都可证伪:**
① 持有半:10 周线不破时不该卖 —— 即用它离场**不会削掉右尾**。
② 结束半:跌破之后趋势结束 —— 即跌破后**不该再创新高**。

═══ 与 §42 的关系:不是重跑 ═══
§42 测过 10 周线止损,**结论是证伪**(纯 10 周线组合年化 +1.15%、Sharpe 0.165、
回撤 −69.97%,六种组合两种选股三种停牌处理下全部明显差于基准)。
**但那次是把它当「一般止损规则」测,判据是组合年化/Sharpe,且在 §77 改用右尾口径之前。**
**本节不同:入场限定在「首次创 250 日新高」事件上(§103 的 21,876 个),
判据用右尾口径,并且加测用户主张的第二半「跌破 = 趋势结束」——那一半从未测过。**

═══ 口径(事前锁定)═══
  **事件**  首次创 250 日新高(此前 120 日未创)—— 与 §103 完全相同,锚点② 恒等复现
  **入场**  首次新高日 **t0 收盘**(本节测持有规则,不是选股,故不等确认)
  **规则A(用户的)**  持有至 **收盘首次跌破 MA50(10 周线)** 当日收盘卖出;
            到面板末仍未破则按末日收盘计(单独标注占比)
  **规则B(对照)**  固定持有 **500 日(24 个月)**
  **规则C(对照)**  固定持有 120 日(6 个月),仅描述
  MA50 只用 t 及之前的收盘,**无前视**(锚点③ 截断校验)
  退市股 ffill 参与,绝不剔除;不含交易成本(换手次数一并输出)

═══ 锚点(不过则全节作废;四个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **§103 事件数恒等复现**:首次新高事件 **21,876** 个
  ③ **四只案例恒等复现**:宇通 **9** / 匠心 **1** / 嘉益 **2** / 泰格 **3** 次
  ④ **无前视校验**:截断到 2020-12-31 重算 MA50 与卖出日,该日前逐个相同

═══ 事前判据(跑之前写死,不放宽;Bonferroni **0.05/6 = 0.00833**)═══
  **前置条件**:可比事件(t0+500 在面板内)**< 1000** 则不判
  ① **实收不亏**:规则A 的**中位实收** ≥ 规则B 的中位实收
  ② **不削右尾(核心)**:规则A 的 **实收 ≥200% 比例 ≥ 规则B 的 0.9 倍**
     —— §62 的核心结论是「所有提高胜率的过滤器都在削右尾」,这一条直接测它
  ③ **「跌破 = 趋势结束」**:跌破 10 周线之后 **250 日内再创 250 日新高**的比例 **< 20%**

**①②③ 全过 = 用户的规则成立:10 周线既保住了右尾,跌破也确实标志趋势结束。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问**:止损天然降低平均亏损,①几乎必然通过 → **堵法:真正的判据是 ②**;
「趋势结束」若门槛设太松会恒真 → **堵法:20% 是明确阈值,且全市场任一时点
创 250 日新高的股票占比远低于此,若跌破后仍有 ≥20% 再创新高,「结束」就不成立**。
**反问**:样本不足 → 前置 n≥1000;
未破的事件被右截断 → **单独输出占比,且规则A/B 用同一批可比事件**;
锚点误杀 → **四个锚点全是恒等式,②③ 已在 §103 实测可达**。

═══ 事前预测(写下以便被证伪)═══
**① 通过、② 不通过、③ 不通过。**
理由:①止损本就降低平均亏损。**②我预测不过** —— §62 实测后 90% 交易亏钱、
利润全在前 10%,§63 九格「不止损」一致最好,§96 20 月线全样本年化 +2.1%→**−1.9%**,
§42 10 周线本身已被证伪。**③我预测不过** —— 跌破 10 周线后仍会有相当比例
(我猜 30%~50%)在一年内重新创新高,「趋势结束」是过强的说法。
**我已两次押「会通过」两次都错(§98、§103),本节回到「多数不过」的预测。
若 ② 通过,那是本项目第一次找到「不削右尾的离场规则」,我会明说我错了。**
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from consolidation_screener import load_panel  # noqa: E402

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
OUT = os.environ.get("OXQ_OUT_DIR", SP)
GAP, MA, H24, H6 = 120, 50, 500, 120
MIN_N, ALPHA, TAIL_R, TREND_MAX = 1000, 0.05 / 6, 0.90, 0.20
EXP_EV, EXP4 = 21876, {"600066": 9, "301061": 1, "301004": 2, "300347": 3}
CUT = "2020-12-31"

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    CL = CL.drop(columns=["510300"])
del frames, STRONG, MA100
idx = CL.index
NT, NS = CL.shape
codes = list(CL.columns)
print(f"面板 {CL.shape}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"
Fa = CL.where(CL > 0).ffill().to_numpy(float)


def newhi(px):
    hi = pd.DataFrame(px).rolling(250, min_periods=100).max().to_numpy(float)
    return np.isfinite(hi) & (px >= hi * 0.9999)


def ma(px, n):
    return pd.DataFrame(px).rolling(n, min_periods=n).mean().to_numpy(float)


NH, M50 = newhi(Fa), ma(Fa, MA)


def events(nh, lim):
    out = []
    for j in range(nh.shape[1]):
        col = nh[:, j]
        for t in np.flatnonzero(col[:lim]):
            if t < 250 or col[max(t - GAP, 0):t].any():
                continue
            out.append((int(t), j))
    return out


EV = events(NH, NT)
cnt = {}
for t, j in EV:
    cnt[codes[j]] = cnt.get(codes[j], 0) + 1
a2 = len(EV) == EXP_EV
a3 = all(cnt.get(c, 0) == v for c, v in EXP4.items())
print(f"首次新高事件 {len(EV):,}  {'✓' if a2 else '✗'} 锚点②(期望 {EXP_EV:,})")
print(f"  {'✓' if a3 else '✗'} 锚点③ 四只:" + " ".join(
    f"{c}{cnt.get(c,0)}/{v}" for c, v in EXP4.items()))


def exit_day(t, j, lim=None):
    """规则A:收盘首次跌破 MA50 的那一天(含);未破返回 None。"""
    hi = NT if lim is None else lim
    col, m = Fa[t + 1:hi, j], M50[t + 1:hi, j]
    w = np.flatnonzero(np.isfinite(m) & (col < m))
    return t + 1 + int(w[0]) if w.size else None


kc = int(np.searchsorted(idx, pd.Timestamp(CUT, tz=idx.tz), side="right"))
M50c = ma(Fa[:kc], MA)
same = True
for t, j in EV[:200000]:
    if t >= kc - 600:
        continue
    col, m = Fa[t + 1:kc, j], M50c[t + 1:kc, j]
    w = np.flatnonzero(np.isfinite(m) & (col < m))
    e1 = t + 1 + int(w[0]) if w.size else None
    e2 = exit_day(t, j, kc)
    if e1 != e2:
        same = False
        break
a4 = same
print(f"  {'✓' if a4 else '✗'} 锚点④ 无前视:截断到 {CUT} 重算卖出日一致")
print(f"预取完成  ({time.time()-t0:.0f}s)", flush=True)

rows = []
for t, j in EV:
    if t + H24 >= NT:
        continue
    p0 = Fa[t, j]
    if not np.isfinite(p0) or p0 <= 0:
        continue
    e = exit_day(t, j)
    held = e is None or e > t + H24
    ee = min(e, NT - 1) if e is not None else NT - 1
    r_a = Fa[ee, j] / p0 - 1
    pk_a = np.nanmax(Fa[t + 1:ee + 1, j]) / p0 - 1 if ee > t else np.nan
    r_b = Fa[t + H24, j] / p0 - 1
    pk_b = np.nanmax(Fa[t + 1:t + H24 + 1, j]) / p0 - 1
    r_c = Fa[t + H6, j] / p0 - 1
    # 跌破之后 250 日内是否再创 250 日新高
    again = np.nan
    if e is not None and e + 250 < NT:
        again = bool(NH[e + 1:e + 251, j].any())
    rows.append(dict(t=t, j=j, 年=idx[t].year, 未破=held, 持有日=(ee - t),
                     A实收=r_a, A峰值=pk_a, B实收=r_b, B峰值=pk_b, C实收=r_c, 再创新高=again))
D = pd.DataFrame(rows)
print(f"可比事件(t0+{H24} 在面板内){len(D):,}   "
      f"24 个月内未跌破 10 周线的占比 **{D['未破'].mean():.1%}**")

W = 96
print(f"\n{'='*W}\n规则对比(入场 = 首次新高日收盘)\n{'='*W}")
print(f"{'规则':<26}{'中位实收':>11}{'均值实收':>11}{'实收≥200%':>12}"
      f"{'实收>0':>10}{'中位持有日':>11}")
for nm, col, hold in (("A 10周线不破就持有(用户)", "A实收", D["持有日"].median()),
                      ("B 固定持有 500 日(24月)", "B实收", H24),
                      ("C 固定持有 120 日(6月)", "C实收", H6)):
    v = D[col].dropna()
    print(f"{nm:<26}{v.median():>11.1%}{v.mean():>11.1%}{(v>=2).mean():>12.2%}"
          f"{(v>0).mean():>10.1%}{hold:>11.0f}")
print(f"\n  峰值(未实现)口径:A 中位 {D['A峰值'].median():+.1%} / ≥200% "
      f"{(D['A峰值']>=2).mean():.2%}   B 中位 {D['B峰值'].median():+.1%} / ≥200% "
      f"{(D['B峰值']>=2).mean():.2%}")

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
for ok, nm in ((a2, "锚点② 事件数"), (a3, "锚点③ 四只案例"), (a4, "锚点④ 无前视")):
    print(f"  {'✓' if ok else '✗'} {nm}")
    if not ok:
        bad.append(nm)

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,Bonferroni {ALPHA:.5f})\n{'='*W}")
m_a, m_b = D["A实收"].median(), D["B实收"].median()
t_a, t_b = float((D["A实收"] >= 2).mean()), float((D["B实收"] >= 2).mean())
ag = D["再创新高"].dropna()
fr = float(ag.mean()) if len(ag) else np.nan
c1 = bool(m_a >= m_b)
c2 = bool(t_b > 0 and t_a >= TAIL_R * t_b)
c3 = bool(np.isfinite(fr) and fr < TREND_MAX)
print(f"  前置:可比事件 {len(D):,} ≥ {MIN_N}")
print(f"  {'✓' if c1 else '✗'} ① 实收不亏:A 中位 {m_a:+.2%} ≥ B 中位 {m_b:+.2%}")
print(f"  {'✓' if c2 else '✗'} ② 不削右尾:A 实收≥200% {t_a:.2%} ≥ "
      f"{TAIL_R:.0%}×B {t_b:.2%} = {TAIL_R*t_b:.2%}")
print(f"  {'✓' if c3 else '✗'} ③ 跌破=趋势结束:跌破后 250 日内再创 250 日新高的比例 "
      f"**{fr:.1%}** < {TREND_MAX:.0%}   (n={len(ag):,})")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:用户的规则成立 —— 10 周线既保住右尾,跌破也确实标志趋势结束。**")
    print("  **事前预测被证伪 —— 我错了。**")
elif c1 and not c2:
    print("  **结论:10 周线离场提高了实收中位,但削掉了右尾 —— §62 的又一次验证。**")
else:
    print("  **结论:用户的规则不成立,详见上表。**")

D.drop(columns=["t", "j"]).to_csv(f"{OUT}/newhigh_ma50_exit.csv", index=False)
print(f"\n→ {OUT}/newhigh_ma50_exit.csv   ({time.time()-t0:.0f}s)")
