"""第七十八节:右尾兑现率 —— 把 §77 的峰值口径换成实收口径

═══ 起因:用户的问题 ═══
§77 建立了「上市[1,3)年」把 P(未来250日内最大涨幅 ≥500%) 抬高 **1.58 倍**(p=0.0000)。
**但那是峰值,是上帝视角。** §70 实测浮盈兑现率只有 13%~55%,
§66 七只案例约 18%(泰格医药 峰值 +1076% → 实收 +198%)。

用户的问题:**右尾概率提高了,怎么把它变成实收?**

═══ 必须先修一个口径缺陷(本节的方法论要点) ═══
§70 自己把这条列为设计缺陷 ②:「**卖得越早,峰值越小,兑现率越好看**」。
§71 沿用了同一个实现(`run_trade` 在持仓循环**内部**累计 peak,
循环随离场而中断),因此**峰值是规则相关的**,六条规则的兑现率互不可比。

本节把峰值定义在**固定窗口**(入场后 H=750 日,等于最长持有),**与离场规则无关**。
于是 `兑现率 = 实收 / 固定窗口峰值` 在规则之间可比。

**代价必须先说清:本节的兑现率会系统性低于 §70/§71 报出的数**,
因为固定峰值 ≥ 规则相关峰值。**这不是结果变差,是原来那把尺子偏松。**

═══ A 部分:实收口径的 lift(与 §77 直接可比) ═══
沿用 §77 **完全相同**的机器(同月、同市值五分位内随机抽同样多只、NSEED 种子):
    lift = P(X ≥ G | 信号) / P(X ≥ G | 同月同市值档随机)
**唯一改变的是 X**:
    §77   X = 未来 H 日内**最大**累计涨幅   ← 峰值(上帝视角)
    §78   X = 按离场规则卖出后的**实收**    ← 实收(含双边成本)

**若 实收 lift@500% ≈ 峰值 lift@500% = 1.58,右尾能转成钱;
若塌回 1.0,离场规则把 §77 抬起来的东西又削掉了。**

═══ B 部分:按固定峰值分层的兑现率(A 部分的机制解释) ═══
每一笔按**固定窗口峰值**分四层:<100% / [100,200%) / [200,500%) / ≥500%,
逐层报兑现率与收益贡献占比。
§62 实测收益 **98.1% 来自 top-10% 的交易** ——
**平均兑现率高但右尾兑现率低的规则恰恰是最坏的**,它砍的正是全部利润的来源。

═══ 离场规则(全部来自 §70/§71,不新造、不调参) ═══
    H0  持有到窗口末(不卖)           ← 天花板参照
    A1  个股收盘破 20 周线(MA100)     ← §70 R1 / §71 A1(六条里最差)
    A2  个股收盘破 10 月线(MA200)     ← §70 R2 / §71 A2
    B2  指数收盘破 10 月线             ← §71 +1.8% p=0.005
    B3  指数月线 MACD 转负(**次月首日**,§71 修正后的无前视口径)← §71 最好 +3.3%
    C   基地计数(25 / 35% / 3 / 15%)  ← §71 +3.0% p=0.010,四参数照抄不动

未纳入 §71 的 A3 / B1:§71 已证明二者被同族规则支配(A3 p=0.250、B1 超额 −0.5%),
为控制运行时长本节不跑。**这是取舍,不是结果筛选** —— 事前写在这里。

═══ 事前判据(跑之前写死,不放宽) ═══
  ① **锚点**:面板 (3297, 5232);且本节用自己的入场口径(次月首日开盘)
     重算的**峰值 lift@≥500%(H=250、§77 同月集)与 §77 的 1.58 差 ≤ 0.25**。
     对不上说明入场口径或信号定义跑偏,**全节作废**。
  ② **主判据**:至少一条规则的 **实收 lift@≥500% ≥ 1.3 且 p < 0.05**
     (1.3 沿用 §77 判据 ②,**不新造门槛**)
  ③ **右尾不被亏待**:② 中 lift 最高的那条规则,其
     **≥500% 层的兑现率 ≥ 该规则的全体平均兑现率**
  ④ **诊断(非判据,但必须报出)**:四层兑现率方向、各层收益贡献占比、
     各规则**中位持有天数**(§71 限定 ⑤ 自认的漏项,本节补上)

  ②③都过 → 右尾概率能转成实收,且指出了具体规则
  ②过③不过 → 能转,但代价是右尾被削,需要换规则
  ②不过   → **右尾概率提高转不成实收**,这是对用户问题的直接否定回答

═══ C 部分:⚠️ 事后分解,不是检验 ═══
**首轮跑完之后才加的**,因此**没有事前判据,不计入判据计数**
(与 §71 限定 ② 对参数敏感性表的处理同规格:「那不是检验」)。
问题:B3 的 lift@500% 是离场规则的功劳,还是**红轴闸门换了个名字**?
§78 每月入场、B3 中位持有仅 23 日,样本是「绿轴入场即刻出场」与
「红轴入场长持」的混合。若 lift 几乎全部来自红轴入场,则 B3 不是离场规则 ——
而 §73 已证择时不创造收益(−0.70pp)、§77 已证红轴本身不提高右尾(lift 0.99),
那样 ② 的解释必须改写。
**分解只重切同一份逐月累加器,不改动任何计算,主表数字应逐位不变。**

**事前预测(写下以便被证伪)**:② **不通过**、③ **不通过**。
理由:§62「所有过滤器都在削右尾」在离场端的复现(§70 四条主动规则全灭);
§77 的 1.58 是峰值,而峰值要兑现必须**持有穿越整段行情**,
任何在途卖出的规则都会把 ≥500% 那一层打回低层。
**最可能的例外是 B3 与 C** —— §71 已证明它们的共同点是「不许早卖」。
**若 ② 通过,说明我又一次低估了慢规则,我错了。**

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
  §77 峰值 lift(AGE_YOUNG):≥100% 1.08 / ≥200% 1.15 / ≥500% **1.58**
"""
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd

# 面板含大量「尚未上市」的全 NaN 列,nanmax/除法必然告警;结果一律由 isfinite 兜住。
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
np.seterr(invalid="ignore", divide="ignore")

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
H, H_ANCHOR = 750, 250
GAINS = [1.0, 2.0, 5.0]
NQ, NSEED, SEED, COST = 5, 200, 20260814, 0.003
Y_LO, Y_HI = 365, 1095
MIN_BAND = 3                       # 某市值档内信号数少于此则该档不参与
BASE_MIN, BASE_MAXDD, N_BASE, FAIL_DD = 25, 0.35, 3, 0.15
BUCKETS = [(-np.inf, 1.0, "<100%"), (1.0, 2.0, "[100,200)"),
           (2.0, 5.0, "[200,500)"), (5.0, np.inf, "≥500%")]

t0 = time.time()
op, cl, ld, mv = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "close", "listed_days", "float_mv"])
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
OP = pd.DataFrame(op).sort_index()
OP.index = OP.index.tz_localize(None)
CL = pd.DataFrame(cl).set_axis(OP.index)
LD = pd.DataFrame(ld).set_axis(OP.index)
MV = pd.DataFrame(mv).set_axis(OP.index)
OP, CL = OP.where(OP > 0), CL.where(CL > 0)
idx = OP.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"

OPa, CLa = OP.to_numpy(float), CL.to_numpy(float)
LDa, MVa = LD.to_numpy(float), MV.to_numpy(float)
ALIVE = np.isfinite(CLa) & (CLa > 0)
CLf = pd.DataFrame(CLa).ffill().to_numpy(float)     # 退市股 ffill,绝不剔除
OPf = pd.DataFrame(OPa).ffill().to_numpy(float)
MA100 = pd.DataFrame(CLa).rolling(100, min_periods=100).mean().to_numpy(float)
MA200 = pd.DataFrame(CLa).rolling(200, min_periods=200).mean().to_numpy(float)
print(f"个股均线完成  ({time.time()-t0:.0f}s)")

ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
first_td = {p: int(np.flatnonzero(ym == p)[0]) for p in ym.unique()}
allm = sorted(last_td)

# ── 指数信号(与 §71 逐行一致) ──
mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
mkc = mk["close"].reindex(idx).ffill()
IDX = mkc.to_numpy(float)
IMA200 = mkc.rolling(200, min_periods=200).mean().to_numpy(float)
SELL_B2 = np.isfinite(IMA200) & (IDX < IMA200)

m_ = mkc.resample("ME").last()
d_ = m_.ewm(span=12, adjust=False).mean() - m_.ewm(span=26, adjust=False).mean()
ih = (d_ - d_.ewm(span=9, adjust=False).mean())
ih.index = ih.index.to_period("M")
SELL_B3 = np.zeros(NT, dtype=bool)          # 确认后次月首日,无前视
im = list(ih.index)
for i, p in enumerate(im[:-1]):
    if im[i + 1] in first_td and ih.iloc[i] < 0:
        SELL_B3[first_td[im[i + 1]]] = True
REG = {p: int(v > 0) for p, v in ih.items()}   # 红轴=1(§68/§71/§73 同一口径)
print(f"指数信号完成  ({time.time()-t0:.0f}s)")

RULES = ["H0 持有到窗口末", "A1 个股破20周线", "A2 个股破10月线",
         "B2 指数破10月线", "B3 指数月线MACD", "C  基地计数"]
NR = len(RULES)


def first_hit(cond):
    """cond (n, NS) 布尔;第 0 行(买入日)不允许卖。返回 (has, row)。"""
    cond = cond.copy()
    cond[0] = False
    has = cond.any(axis=0)
    return has, np.argmax(cond, axis=0)


def settle(has, row, entry, e, end):
    """按 (has,row) 结算:命中则次日开盘卖,否则持有到窗口末收盘。含双边成本。"""
    px = np.where(has, OPf[np.minimum(e + row + 1, NT - 1),
                          np.arange(NS)], CLf[end])
    bad = ~np.isfinite(px)
    if bad.any():                                     # 次日开盘缺失 → 用当日收盘
        px = np.where(bad, CLf[np.minimum(e + row, NT - 1), np.arange(NS)], px)
    ret = (px * (1 - COST)) / (entry * (1 + COST)) - 1
    hold = np.where(has, row + 1, end - e)
    return ret, hold


def run_window(e, end):
    """一个入场日上把 6 条规则一次算完(按时间循环、跨标的向量化)。"""
    entry = OPf[e]
    cw = CLf[e:end + 1]
    n = cw.shape[0]
    peak_fix = np.nanmax(cw, axis=0) / entry - 1
    peak_anc = np.nanmax(cw[:min(H_ANCHOR, n - 1) + 1], axis=0) / entry - 1
    out, hold = {}, {}

    out[0], hold[0] = settle(np.zeros(NS, bool), np.zeros(NS, int), entry, e, end)
    for r, mat in ((1, MA100), (2, MA200)):
        mw = mat[e:end + 1]
        out[r], hold[r] = settle(*first_hit(np.isfinite(mw) & (cw < mw)), entry, e, end)
    for r, sel_idx in ((3, SELL_B2), (4, SELL_B3)):
        out[r], hold[r] = settle(
            *first_hit(np.repeat(sel_idx[e:end + 1, None], NS, axis=1)), entry, e, end)

    # ── C 基地计数:顺序依赖,按时间循环、跨标的向量化(逐行照搬 §71 的次序) ──
    peak_c = entry.copy()
    nbase = np.zeros(NS, int)
    below = np.zeros(NS, int)
    exited = np.zeros(NS, bool)
    row_c = np.zeros(NS, int)
    for r in range(n):
        c = cw[r]
        ok = np.isfinite(c) & np.isfinite(peak_c) & (peak_c > 0)
        newhigh = ok & (c > peak_c)
        nbase += (newhigh & (below >= BASE_MIN)).astype(int)
        below = np.where(newhigh, 0, below + ok.astype(int))
        dd = np.where(ok, c / np.where(peak_c > 0, peak_c, np.nan) - 1, np.nan)
        hit = ok & ((dd <= -BASE_MAXDD) | ((nbase >= N_BASE) & (dd <= -FAIL_DD)))
        peak_c = np.where(newhigh, c, peak_c)
        fire = hit & ~exited & (r > 0)
        row_c = np.where(fire, r, row_c)
        exited |= fire
    out[5], hold[5] = settle(exited, row_c, entry, e, end)
    return out, hold, peak_fix, peak_anc


rng = np.random.default_rng(SEED)
# acc[key][G] = [(obs, rand[NSEED]) per month];key = 规则号 或 "PEAK"/"PEAK_ANC"
acc = {k: {g: [] for g in GAINS} for k in list(range(NR)) + ["PEAK", "PEAK_ANC"]}
strat = {r: {b[2]: [] for b in BUCKETS} for r in range(NR)}   # (实收, 峰值)
holds = {r: [] for r in range(NR)}
n_full = n_anc = 0


def bandwise(hit, sel, bands):
    """§77 的机器:逐市值档比对信号命中率与同档随机命中率。"""
    o, rr, nb = [], np.zeros(NSEED), 0
    for b in bands:
        si = b[sel[b]]
        if len(si) < MIN_BAND or len(b) <= len(si):
            continue
        o.append(hit[si].mean())
        rr += np.array([hit[rng.choice(b, len(si), replace=False)].mean()
                        for _ in range(NSEED)])
        nb += 1
    return (float(np.mean(o)), rr / nb) if nb else None


for mi, p in enumerate(allm[:-1]):
    t = last_td[p]
    e = first_td[allm[mi + 1]]
    if e >= NT - 5:
        continue
    base = ALIVE[t] & np.isfinite(OPa[e]) & (OPa[e] > 0)
    if base.sum() < 200:
        continue
    sel = base & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI)     # AGE_YOUNG
    if sel.sum() < 10:
        continue
    m = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(m[base], np.linspace(0, 1, NQ + 1)[1:-1])
    bands = []
    for i in range(NQ):
        lo = -np.inf if i == 0 else q[i - 1]
        hi = np.inf if i >= NQ - 1 else q[i]
        bands.append(np.flatnonzero(base & (m > lo) & (m <= hi)))

    red = REG.get(p, 0) == 1        # 入场依据月的轴色(与 §71 同:p 月末确认,次月首日买)

    # 锚点:§77 同月集(只需 250 日前瞻),峰值口径,不跑离场规则
    if e + H_ANCHOR < NT:
        C = CLf[e:e + H_ANCHOR + 1]
        pa = np.nanmax(C, axis=0) / OPf[e] - 1
        for g in GAINS:
            got = bandwise(np.where(np.isfinite(pa), pa, -9) >= g, sel, bands)
            if got:
                acc["PEAK_ANC"][g].append(got + (red,))
        n_anc += 1

    if e + H >= NT:                 # 主表要求完整 750 日窗口,不用截断窗口
        continue
    out, hold, peak_fix, peak_anc = run_window(e, e + H)
    for g in GAINS:
        got = bandwise(np.where(np.isfinite(peak_fix), peak_fix, -9) >= g, sel, bands)
        if got:
            acc["PEAK"][g].append(got + (red,))
        for r in range(NR):
            got = bandwise(np.where(np.isfinite(out[r]), out[r], -9) >= g, sel, bands)
            if got:
                acc[r][g].append(got + (red,))
    si = np.flatnonzero(sel)
    for r in range(NR):
        holds[r].append(hold[r][si])
        for lo, hi, nm in BUCKETS:
            k = si[(peak_fix[si] >= lo) & (peak_fix[si] < hi)
                   & np.isfinite(peak_fix[si]) & np.isfinite(out[r][si])]
            if len(k):
                strat[r][nm].append(np.c_[out[r][k], peak_fix[k]])
    n_full += 1
    if n_full % 20 == 0:
        print(f"  {p}  信号 {sel.sum()}  完整窗口月 {n_full}  ({time.time()-t0:.0f}s)",
              flush=True)
print(f"逐月完成:完整窗口 {n_full} 月 / 锚点月 {n_anc}  ({time.time()-t0:.0f}s)")


def summarize(key, g, only_red=None):
    """only_red=None 全部;True 仅红轴入场月;False 仅绿轴入场月。"""
    a = acc[key][g]
    if only_red is not None:
        a = [x for x in a if bool(x[2]) is only_red]
    if len(a) < 12:
        return None
    o = float(np.mean([x[0] for x in a]))
    r = np.mean([x[1] for x in a], axis=0)
    if r.mean() <= 0:
        return None
    lifts = o / np.where(r > 0, r, np.nan)
    return dict(n_mo=len(a), obs=o, rnd=float(r.mean()), lift=o / r.mean(),
                lo=float(np.nanpercentile(lifts, 5)),
                hi=float(np.nanpercentile(lifts, 95)),
                p=float((r >= o).mean()))


print(f"\n{'='*112}")
print("A 部分:实收口径 lift(信号=上市[1,3)年;对照=同月同市值五分位随机同样多只)")
print(f"{'='*112}")
print(f"{'口径':<22}{'≥100%':>10}{'p':>8}{'≥200%':>10}{'p':>8}{'≥500%':>10}{'p':>8}")
rows = []
for key, label in ([("PEAK_ANC", f"峰值 H={H_ANCHOR}(§77锚点)"),
                    ("PEAK", f"峰值 H={H}(本节窗口)")]
                   + [(r, RULES[r]) for r in range(NR)]):
    cells, line = [], f"{label:<22}"
    for g in GAINS:
        s = summarize(key, g)
        cells.append(s)
        line += (f"{s['lift']:>10.2f}{s['p']:>8.4f}" if s else f"{'—':>10}{'—':>8}")
    print(line)
    for g, s in zip(GAINS, cells):
        if s:
            rows.append(dict(部分="A", 口径=label, 门槛=f"≥{g:.0%}", 月数=s["n_mo"],
                             信号=s["obs"], 随机=s["rnd"], lift=s["lift"],
                             lift下界=s["lo"], lift上界=s["hi"], p=s["p"]))

print(f"\n{'='*112}")
print(f"B 部分:按**固定窗口峰值**(H={H},与规则无关)分层的兑现率")
print(f"{'='*112}")
print(f"{'规则':<18}{'层':>12}{'笔数':>9}{'占比':>8}{'实收':>10}"
      f"{'峰值':>10}{'兑现率':>9}{'收益贡献':>10}")
realz = {}
for r in range(NR):
    tot = np.concatenate([np.concatenate(strat[r][b[2]]) for b in BUCKETS
                          if strat[r][b[2]]])
    n_all = len(tot)
    sum_all = float(np.sum(tot[:, 0]))
    rr_all = float(np.mean(tot[:, 0]) / np.mean(tot[:, 1])) if np.mean(tot[:, 1]) > 0 else np.nan
    for lo, hi, nm in BUCKETS:
        if not strat[r][nm]:
            continue
        a = np.concatenate(strat[r][nm])
        rr = float(np.mean(a[:, 0]) / np.mean(a[:, 1])) if np.mean(a[:, 1]) > 0 else np.nan
        contrib = float(np.sum(a[:, 0]) / sum_all) if sum_all != 0 else np.nan
        if nm == "≥500%":
            realz[r] = rr
        print(f"{RULES[r]:<18}{nm:>12}{len(a):>9,}{len(a)/n_all:>8.1%}"
              f"{np.mean(a[:, 0]):>+10.1%}{np.mean(a[:, 1]):>+10.1%}"
              f"{rr:>9.0%}{contrib:>10.1%}")
        rows.append(dict(部分="B", 口径=RULES[r], 门槛=nm, 笔数=len(a),
                         占比=len(a) / n_all, 实收=float(np.mean(a[:, 0])),
                         峰值=float(np.mean(a[:, 1])), 兑现率=rr, 收益贡献=contrib))
    md = float(np.median(np.concatenate(holds[r])))
    print(f"{'':<18}{'全体':>12}{n_all:>9,}{1:>8.1%}{np.mean(tot[:, 0]):>+10.1%}"
          f"{np.mean(tot[:, 1]):>+10.1%}{rr_all:>9.0%}{'':>10}  中位持有 {md:.0f} 日")
    rows.append(dict(部分="B", 口径=RULES[r], 门槛="全体", 笔数=n_all, 占比=1.0,
                     实收=float(np.mean(tot[:, 0])), 峰值=float(np.mean(tot[:, 1])),
                     兑现率=rr_all, 中位持有=md))
    realz[str(r) + "_all"] = rr_all
    print()

print(f"{'='*112}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*112}")
anc = summarize("PEAK_ANC", 5.0)
c1 = anc is not None and abs(anc["lift"] - 1.58) <= 0.25
print("  ① 锚点 面板(3297,5232) ✓;峰值 lift@500%(H=250)vs §77 1.58 差 ≤0.25")
print(f"       实测 {anc['lift']:.2f}(差 {abs(anc['lift']-1.58):.2f},{n_anc} 月)"
      f"   {'✓' if c1 else '✗'}")
cand = [(r, summarize(r, 5.0)) for r in range(NR)]
cand = [(r, s) for r, s in cand if s and s["lift"] >= 1.3 and s["p"] < 0.05]
c2 = len(cand) > 0
print(f"  ② 实收 lift@500% ≥1.3 且 p<0.05                {len(cand)} 个"
      f"   {'✓' if c2 else '✗'}")
for r, s in cand:
    print(f"       {RULES[r]:<18} lift {s['lift']:.2f}  p={s['p']:.4f}")
if c2:
    best = max(cand, key=lambda x: x[1]["lift"])[0]
    c3 = realz.get(best, np.nan) >= realz.get(str(best) + "_all", np.nan)
    print(f"  ③ 最优规则 {RULES[best]} 的 ≥500% 层兑现率 ≥ 全体平均")
    print(f"       {realz.get(best, float('nan')):.0%} vs "
          f"{realz.get(str(best)+'_all', float('nan')):.0%}   {'✓' if c3 else '✗'}")
else:
    c3 = False
    print("  ③ 无候选规则,不适用                                    ✗")
print("  ④ 诊断已在 B 部分逐层报出(兑现率方向 / 收益贡献 / 中位持有天数)")

print()
if not c1:
    print("  **① 锚点不过:入场口径或信号定义跑偏,本节全部结论作废。**")
elif c2 and c3:
    print("  **结论:右尾概率能转成实收,且有具体规则。事前预测被证伪 —— 我错了。**")
elif c2:
    print("  **结论:能转,但右尾被削(③ 不过),需要换规则。事前预测部分被证伪。**")
else:
    print("  **结论:右尾概率提高转不成实收 —— 对用户问题的直接否定回答。**")
    print("  **事前预测命中(② 不通过)。**")

print(f"\n{'='*112}")
print("C 部分:按**入场月轴色**拆开(⚠️ 事后分解,**不是检验**,无判据)")
print(f"{'='*112}")
print("问:B3 的 lift@500% 是离场规则的功劳,还是红轴闸门换了个名字?")
print("   §78 每月入场,B3 中位持有仅 23 日 —— 样本是「绿轴入场即刻出场」")
print("   与「红轴入场长持」的混合。若 lift 几乎全部来自红轴入场,")
print("   则 B3 不是离场规则;而 §73 已证择时不创造收益(−0.70pp)、")
print("   §77 已证红轴本身不提高右尾(lift 0.99),那样解释完全不同。")
print()
print(f"{'口径':<22}{'轴色':>6}{'月数':>6}{'信号':>9}{'随机':>9}"
      f"{'lift':>8}{'lift区间':>16}{'p':>8}")
for key, label in [("PEAK", f"峰值 H={H}"), (4, RULES[4]), (0, RULES[0]), (5, RULES[5])]:
    for only, tag in ((None, "全部"), (True, "红轴"), (False, "绿轴")):
        s = summarize(key, 5.0, only)
        if not s:
            print(f"{label:<22}{tag:>6}{'—(月数不足12)':>20}")
            continue
        ci = "[{:.2f}, {:.2f}]".format(s["lo"], s["hi"])
        print(f"{label:<22}{tag:>6}{s['n_mo']:>6}{s['obs']:>9.3%}{s['rnd']:>9.3%}"
              f"{s['lift']:>8.2f}{ci:>16}{s['p']:>8.4f}")
        rows.append(dict(部分="C", 口径=label, 门槛="≥500%", 轴色=tag, 月数=s["n_mo"],
                         信号=s["obs"], 随机=s["rnd"], lift=s["lift"],
                         lift下界=s["lo"], lift上界=s["hi"], p=s["p"]))
    print()

b3r, b3g = summarize(4, 5.0, True), summarize(4, 5.0, False)
if b3r and b3g:
    print(f"  **判读**:红轴入场 lift {b3r['lift']:.2f}(p={b3r['p']:.4f})  vs  "
          f"绿轴入场 lift {b3g['lift']:.2f}(p={b3g['p']:.4f})")
    if b3g["lift"] >= 1.3 and b3g["p"] < 0.05:
        print("  → **两轴都成立:B3 不是红轴闸门的马甲,它自己有信息。**")
    elif b3r["lift"] >= 1.3 and b3r["p"] < 0.05:
        print("  → **只有红轴入场成立:B3 很可能就是红轴闸门换了名字,")
        print("     而 §73/§77 已证闸门不创造收益、不提高右尾 —— ② 的解释要改写。**")
    else:
        print("  → **拆开后两边都不显著:1.66 是合并样本才有的,须谨慎。**")

pd.DataFrame(rows).to_csv(f"{SP}/righttail_realization.csv", index=False)
print(f"\n→ {SP}/righttail_realization.csv   ({time.time()-t0:.0f}s)")
