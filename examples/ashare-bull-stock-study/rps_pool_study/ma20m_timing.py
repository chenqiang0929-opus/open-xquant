"""第九十五节:20 月线择时全样本 —— 慢速趋势止损是不是真的不削右尾(事前登记)

═══ 起因:用户指出我的基准不成立 ═══
我拿「买入持有」当基准比五次信号,用户指出 **2018 年那根是必须止损的**
(宇通 2018 年 −49.3%、年内回撤 57.0%),后又指出 2008 年同样跌破 20 月线。
**2008 无法验证 —— 数据源本身从 2013-01-04 起,2013 年之前 0 行。**

宇通单只实测(`case_yutong_ma20m.py`,已落库):

    买入持有   +541.9%  年化 +14.6%  最大回撤 **29.5%**  2018 −49.3%  持仓 100%
    MA20 月线  +528.7%  年化 +14.4%  最大回撤 **20.7%**  2018 −25.8%  持仓 60.4%

可查区间里 20 月线两次都起作用:2018 走了但不干净(5 月卖、6 月买回、7 月再卖,
一次 −12.7% 打脸);**2021-05 卖出 9.58 → 2023-03 买回 9.45,空仓 22 个月,
避开 −45.5% 的峰谷跌幅** —— 这次很干净。

**但 n=2、一只股票,且参数敏感:MA10/12/20/24 总收益 246%/466%/529%/408%,
用户提的 20 恰好是四个里最好的。本节上全样本。**

═══ 为什么这条值得单测:它和 §63 直接对撞 ═══
§63 的结论是「**不止损**在三路信号上一致最好」,§62 是「所有提高胜率的过滤器
都在削右尾」。**但那些测的是固定百分比止损和 252 天窗口,从来没测过慢速趋势线。**
20 月线的特点恰恰是**慢**:它不会在行情中途把赢家割掉(宇通 2014-09 买入一路
拿到 2017-01 拿满 +77.0%、2023-03 买入至今 +321.5%),只在趋势整体转向时离场。
**如果右尾结论是普适的,它照样会削;如果不是,那 §62/§63 的边界就找到了。**

═══ 口径(事前锁定)═══
  信号    20 月线 = 最近 20 个月末收盘均线;**月末收盘 > MA20 则下月满仓,否则空仓**
          信号取上月末,**无前视**(锚点③ 用截断面板证明)
  样本    全部 5232 只;退市股按最后有效价 ffill 参与,**绝不剔除**
          每只股票需 ≥ 20 个月历史才起算;有效月数 < 60 的股票不计入统计
  不含交易成本(A 股双边约 0.1%~0.2%;换手次数见输出,影响在正文里折算)
  稳健性  同时报 MA10 / MA12 / MA24,**仅描述,判据只压在用户提的 MA20 上**

═══ 锚点(不过则全节作废;三个都是恒等式)═══
  ① 面板 (3297, 5232)
  ② **宇通恒等复现单只跑**:MA20 总收益 **+528.7%**、最大回撤 **20.7%**、
     **11** 次交易(容差:收益 ±0.5pp、回撤 ±0.2pp、交易次数必须一致)
  ③ **无前视校验**:把面板截断到 2020-12-31 重算持仓序列,
     与全样本版本在该日之前**逐月相同**

═══ 事前判据(跑之前写死,不放宽)═══
  **前置条件**:有效月数 ≥ 60 的股票 < 1000 只则不判
  ① **降回撤**:「MA20 最大回撤 < 买入持有最大回撤」的股票占比 **≥ 80%**
  ② **不牺牲收益**:MA20 的**年化中位** ≥ 买入持有年化中位 **− 2pp**
  ③ **逐年一致**(§91 立的规矩):14 个年份里,「当年 MA20 回撤 < 持有回撤」
     的股票占比 ≥ 60% 的年份数 **≥ 80%**

**①②③ 全过 = 慢速趋势止损既降回撤又不削收益,§62/§63 的右尾结论存在边界。
① ③ 过而 ② 不过 = 它照样在削右尾,只是削得慢 —— 与 §62/§63 一致,不是例外。**

═══ 判据自查(§79 正问 + §83 反问)═══
**正问:什么会让它通过而不回答问题?**
→ ① 几乎必然通过(任何离场都降回撤)→ **堵法:真正的判据是 ②,①③ 只是前提**。
→ 只看总收益会被少数暴涨股主导 → **堵法:用中位数,并同时报占比**。
→ 幸存者偏差 → **堵法:退市股 ffill 参与,绝不剔除**。

**反问:什么会让它不通过而与问题无关?**
→ 参数选在宇通身上挑的 → **20 是用户按图指的,不是我调的;
  且 MA10/12/24 一并报出,若结论只在 20 上成立,必须在正文里说**。
→ 交易成本没算 → 换手次数一并输出,正文按双边 0.15% 折算后复核 ②。
→ 锚点误杀正确实现 → **三个锚点全是恒等式,② 已在单只实测可达**。

═══ 事前预测(写下以便被证伪)═══
**① 通过、③ 通过、② 不通过。**
理由:§62 实测 OOS 后 90% 的交易加起来是亏钱的、全部利润来自前 10%,
且「所有提高胜率的过滤器都在削右尾」;§63 在三路信号 × 三种离场的 9 格里
「不止损」一致最好。**20 月线虽然慢,但它在每一轮大跌里都会离场,
而右尾行情常常紧跟在大跌之后(宇通 2022-10 见底 → 2023-07 +133%),
我预计它会系统性地错过右尾起点,年化中位掉得比 2pp 多。**
**若 ② 通过,说明「慢」确实能把降回撤和不削右尾同时拿到,我错了 ——
那会是本项目第一次找到 §62/§63 结论的边界。**
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
WINS, MAIN = (10, 12, 20, 24), 20
MIN_MONTHS, MIN_STOCKS = 60, 1000
DD_FRAC, CAGR_TOL, YR_FRAC, YR_STOCK = 0.80, 0.02, 0.80, 0.60
CUT = "2020-12-31"

t0 = time.time()
CL, frames, STRONG, MA100 = load_panel(DATA)
if "510300" in CL.columns:
    CL = CL.drop(columns=["510300"])
del frames, STRONG, MA100
idx = CL.index
NT, NS = CL.shape
codes = list(CL.columns)
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点① 对不上 {(NT, NS)}"

Fa = CL.where(CL > 0).ffill().to_numpy(float)     # 退市股 ffill 参与,绝不剔除
ym = idx.to_period("M")
mp = ym.unique()
mend = np.array([int(np.flatnonzero(ym == q)[-1]) for q in mp])
MC = Fa[mend]                                      # (月数, 股票数) 月末收盘
NM = len(mp)
print(f"月末矩阵 {MC.shape}   ({time.time()-t0:.0f}s)", flush=True)


def mdd(eq):
    pk = np.maximum.accumulate(eq, axis=0)
    return np.nanmax((pk - eq) / pk, axis=0)


def hold_matrix(mc, win):
    """上月末 收盘 > MA(win) -> 本月持仓。信号滞后一期,无前视。"""
    ma = pd.DataFrame(mc).rolling(win).mean().to_numpy(float)
    sig = np.isfinite(ma) & (mc > ma)
    h = np.zeros_like(sig)
    h[1:] = sig[:-1]
    return h


def evaluate(mc, win):
    r = np.zeros_like(mc)
    r[1:] = mc[1:] / mc[:-1] - 1
    r[~np.isfinite(r)] = 0.0
    h = hold_matrix(mc, win)
    eq_t = np.cumprod(1 + r * h, axis=0)
    eq_b = np.cumprod(1 + r, axis=0)
    trades = np.abs(np.diff(h.astype(np.int8), axis=0)).sum(axis=0)
    return eq_t, eq_b, mdd(eq_t), mdd(eq_b), trades, h


valid = np.isfinite(MC).sum(axis=0) >= MIN_MONTHS
nv = int(valid.sum())
print(f"有效股票(≥{MIN_MONTHS} 个月)  **{nv:,}** / {NS:,}")

res = {}
for w in WINS:
    eq_t, eq_b, dd_t, dd_b, tr, h = evaluate(MC, w)
    yrs = np.isfinite(MC).sum(axis=0) / 12
    cg_t = np.where(yrs > 0, eq_t[-1] ** (1 / np.maximum(yrs, 1e-9)) - 1, np.nan)
    cg_b = np.where(yrs > 0, eq_b[-1] ** (1 / np.maximum(yrs, 1e-9)) - 1, np.nan)
    res[w] = dict(eq_t=eq_t, eq_b=eq_b, dd_t=dd_t, dd_b=dd_b, tr=tr, h=h,
                  cg_t=cg_t, cg_b=cg_b)
    print(f"  MA{w} 完成  ({time.time()-t0:.0f}s)", flush=True)

W = 100
print(f"\n{'='*W}\n全样本结果({nv:,} 只;中位数)\n{'='*W}")
print(f"{'口径':<10}{'总收益中位':>12}{'年化中位':>10}{'最大回撤中位':>13}"
      f"{'回撤更小占比':>13}{'年化更高占比':>13}{'交易次数中位':>13}{'持仓占比':>10}")
b = res[MAIN]
print(f"{'买入持有':<10}{np.nanmedian(b['eq_b'][-1][valid])-1:>+12.1%}"
      f"{np.nanmedian(b['cg_b'][valid]):>+10.1%}"
      f"{np.nanmedian(b['dd_b'][valid]):>13.1%}{'—':>13}{'—':>13}{1:>13}{'100.0%':>10}")
rows = []
for w in WINS:
    d = res[w]
    fdd = float(np.mean(d["dd_t"][valid] < d["dd_b"][valid]))
    fcg = float(np.mean(d["cg_t"][valid] > d["cg_b"][valid]))
    tag = f"MA{w} 月线" + (" ←主" if w == MAIN else "")
    print(f"{tag:<10}{np.nanmedian(d['eq_t'][-1][valid])-1:>+12.1%}"
          f"{np.nanmedian(d['cg_t'][valid]):>+10.1%}"
          f"{np.nanmedian(d['dd_t'][valid]):>13.1%}{fdd:>13.1%}{fcg:>13.1%}"
          f"{np.nanmedian(d['tr'][valid]):>13.0f}"
          f"{np.nanmean(d['h'][:, valid]):>10.1%}")
    rows.append(dict(口径=f"MA{w}", 总收益中位=float(np.nanmedian(d["eq_t"][-1][valid])) - 1,
                     年化中位=float(np.nanmedian(d["cg_t"][valid])),
                     回撤中位=float(np.nanmedian(d["dd_t"][valid])),
                     回撤更小占比=fdd, 年化更高占比=fcg,
                     交易次数中位=float(np.nanmedian(d["tr"][valid]))))

print(f"\n{'='*W}\n逐年:当年 MA{MAIN} 回撤 < 买入持有回撤 的股票占比\n{'='*W}")
yr = []
r_all = np.zeros_like(MC)
r_all[1:] = MC[1:] / MC[:-1] - 1
r_all[~np.isfinite(r_all)] = 0.0
h = b["h"]
for y in sorted({q.year for q in mp}):
    m = np.array([q.year == y for q in mp])
    if m.sum() < 6:
        continue
    et = np.cumprod(1 + r_all[m] * h[m], axis=0)
    eb = np.cumprod(1 + r_all[m], axis=0)
    ok = valid & np.isfinite(MC[m][0])
    f = float(np.mean(mdd(et)[ok] < mdd(eb)[ok])) if ok.sum() else np.nan
    yr.append(dict(年=y, n=int(ok.sum()), 占比=f))
    print(f"  {y}  n={int(ok.sum()):>5,}  回撤更小占比 {f:>6.1%}  "
          f"{'✓' if f >= YR_STOCK else '✗'}")
Y = pd.DataFrame(yr)
yfrac = float((Y["占比"] >= YR_STOCK).mean()) if len(Y) else np.nan

print(f"\n{'='*W}\n锚点核对(不过则全节作废)\n{'='*W}")
bad = []
print("  ✓ 锚点① 面板 (3297, 5232)")
J = codes.index("600066")
jr, jd, jt = b["eq_t"][-1][J] - 1, b["dd_t"][J], int(b["tr"][J])
a2 = abs(jr - 5.287) <= 0.005 and abs(jd - 0.207) <= 0.002 and jt == 11
print(f"  {'✓' if a2 else '✗'} 锚点② 宇通 MA20 恒等复现:总收益 {jr:+.1%}(期望 +528.7%)、"
      f"回撤 {jd:.1%}(期望 20.7%)、交易 {jt} 次(期望 11)")
if not a2:
    bad.append("锚点②")
kc = int(np.searchsorted([q.end_time.date() for q in mp], pd.Timestamp(CUT).date(),
                         side="right"))
a3 = bool(np.array_equal(hold_matrix(MC[:kc], MAIN), b["h"][:kc]))
print(f"  {'✓' if a3 else '✗'} 锚点③ 无前视校验:截断到 {CUT}(前 {kc} 个月)重算持仓一致")
if not a3:
    bad.append("锚点③")

print(f"\n{'='*W}\n事前判据 vs 实际(判据跑前写死并单独提交,未放宽)\n{'='*W}")
fdd = float(np.mean(b["dd_t"][valid] < b["dd_b"][valid]))
cg_t, cg_b = float(np.nanmedian(b["cg_t"][valid])), float(np.nanmedian(b["cg_b"][valid]))
print(f"  前置条件:有效股票 {nv:,} ≥ {MIN_STOCKS}")
c1 = fdd >= DD_FRAC
c2 = cg_t >= cg_b - CAGR_TOL
c3 = bool(np.isfinite(yfrac) and yfrac >= YR_FRAC)
print(f"  {'✓' if c1 else '✗'} 判据① 降回撤:回撤更小的股票占比 {fdd:.1%} ≥ {DD_FRAC:.0%}")
print(f"  {'✓' if c2 else '✗'} 判据② 不牺牲收益:年化中位 {cg_t:+.2%} ≥ "
      f"{cg_b:+.2%} − {CAGR_TOL:.0%} = {cg_b-CAGR_TOL:+.2%}")
print(f"  {'✓' if c3 else '✗'} 判据③ 逐年一致:合格年份占比 {yfrac:.1%} ≥ {YR_FRAC:.0%}")
print()
if bad:
    print(f"  **{bad} 不过:本节结论作废。**")
elif c1 and c2 and c3:
    print("  **结论:慢速趋势止损既降回撤又不削收益 —— §62/§63 的右尾结论存在边界。**")
    print("  **事前预测被证伪 —— 我错了。**")
elif c1 and c3:
    print("  **结论:20 月线降回撤有效且逐年一致,但年化中位掉得超过 2pp ——")
    print("     它照样在削右尾,只是削得慢。与 §62/§63 一致,不是例外。事前预测命中。**")
else:
    print("  **结论:20 月线择时在全样本上不成立。**")
print(f"\n  交易成本复核:交易次数中位 {np.nanmedian(b['tr'][valid]):.0f} 次,"
      f"按双边 0.15% 折算约 {np.nanmedian(b['tr'][valid])*0.0015:.2%},"
      f"摊到 {NM/12:.1f} 年约 {np.nanmedian(b['tr'][valid])*0.0015/(NM/12):.3%}/年")
print("  **参数说明:20 是用户按月线图指的,不是我调的;"
      "若结论只在 20 上成立而 10/12/24 不成立,以上表为准,必须在正文里说。**")

pd.DataFrame(rows).to_csv(f"{OUT}/ma20m_timing.csv", index=False)
Y.to_csv(f"{OUT}/ma20m_timing_yearly.csv", index=False)
print(f"\n→ {OUT}/ma20m_timing.csv + _yearly.csv   ({time.time()-t0:.0f}s)")
