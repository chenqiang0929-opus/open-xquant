"""右尾密度统一标尺:有没有任何日频信号能提高「出现极端右尾」的概率(第七十七节)

═══ 起因 ═══
用户把衡量目标换掉了:「我希望是**提高右尾的概率**,而不是一定要去计算它后期的收益,
你不是也说后期收益其实很难判断的。」

这个转向由 §75/§76 直接触发:
  §75 「金叉前夜」9,344 笔:≥100% 占比 **5.2%** vs 同市值随机 **5.4%** —— 右尾没动,中位数变差
  §76 用户四条件在三个时点:一胜两负,2024-01 那批 105 只里翻倍只有 4 只(3.8%)vs 随机 18.1%

而 §69 的年龄条件是 76 节里**唯一**把右尾密度抬起来过的东西
(红轴、同市值档内,≥500% 密度 **2.73 vs 1.36**,p=0.0000)。

**§76 只有三个横截面,且日期是用户挑的(选择偏差)。本节看 2013-2026 全样本。**

═══ 统一指标(所有信号共用,不因信号而变) ═══
    lift(信号, G) = P(未来 250 日内最大累计涨幅 ≥ G | 信号)
                  / P(同月、**同市值五分位**内随机抽同样多只)

  G ∈ {100%, 200%, 500%}  三个门槛全报,不挑
  逐月末取样;对照 NSEED 个种子,报中位、区间、p 值(§69 方法)
  退市股 ffill 参与,**不剔除**(否则检验自身就有幸存者偏差)

═══ A 部分:单独检验 ═══
每个信号各自 vs 同市值档随机。

═══ B 部分:增量检验 ═══
基线 = `REGIME_RED × AGE_YOUNG`(§68+§69,唯一通过完整控制的组合)。
在**基线样本内部**,信号组 vs 基线内同市值档随机抽同样多只。
**这一部分才回答「次新池里到底该看什么」—— 也就是仪表盘该显示什么。**

═══ 两个校准锚点(比多测几个信号更重要,任一不过则整表作废) ═══
  **正向** AGE_YOUNG 红轴 ≥500% 必须 ≈ **2.73 vs 1.36**(±0.15)—— 复现 §69
  **零**   SMALL_MV 全部门槛必须 ≈ **1.00**(±0.10)

零锚点是**一个必须失败的检验**:对照既然按市值五分位抽,
「最小五分位」这个信号在其所在档内部就是全体,lift 必然 ≈1.0。
若显著偏离,说明分档逻辑写错了。
全研究此前没有过这样的负对照,而 §68 恰恰在这里栽过
(把「>10年」当对照,随机中位就有 1.16 而非 1.00,等于把门槛设松了)。

═══ 事前判据(跑之前写死,不放宽) ═══
  ① 两个校准锚点都通过                      不过则整表作废
  ② A 部分:有信号 lift@500% ≥ 1.3 且 p<0.05
  ③ B 部分:基线之上有信号 lift@500% ≥ 1.2 且 p<0.05
  ④ 频率:通过 ③ 的信号,基线内月均触发 ≥ 50 只
     (否则组合层建不起 50 只 —— §69 算过 50 只才有 94% 概率抓到 ≥500%)
  ⑤ 案例锚点:USER_SETUP 必须检出宁德 2019-12 / 生益 2024-05 / 宇通 2024-01

**②③ 都不过 → 「没有任何日频信号能提高右尾密度」,这本身就是可交付结论**:
仪表盘该只显示闸门与年龄,其余全部标 ⛔。

**事前预测(写下以便被证伪)**:① 通过;② 除 SMALL_MV 外**全部不过**;③ **全部不过**。
理由:§59/§64 各类特征 lift 1.05/0.95/0.95、ML AUC 0.57;§75 已证买点信号不碰右尾。
**若 ③ 有信号通过,我错了,那是本研究最有价值的发现。**

═══ 必须带在结论旁边的限定 ═══
**右尾概率是峰值口径**(未来 250 日内**最大**累计涨幅),是上帝视角。
**右尾密度提高 ≠ 钱变多** —— §70 实测兑现率 13%~55%,§66 七只案例约 18%。

═══ 锚点 ═══
  面板 3,297 × 5,232、2013-01-04 ~ 2026-08-03
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
H, NQ, NSEED, SEED = 250, 5, 200, 20260814
GAINS = (1.0, 2.0, 5.0)
Y_LO, Y_HI = 365, 1095
MIN_N = 10                     # 单月单档内信号样本下限

t0 = time.time()
cl, vo, mv, ld, ni = {}, {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["close", "volume", "float_mv", "listed_days",
                                    "net_income"])
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    vo[k] = pd.to_numeric(x["volume"], errors="coerce")
    mv[k] = pd.to_numeric(x["float_mv"], errors="coerce")
    ld[k] = pd.to_numeric(x["listed_days"], errors="coerce")
    ni[k] = pd.to_numeric(x["net_income"], errors="coerce")
CL = pd.DataFrame(cl).sort_index()
CL.index = CL.index.tz_localize(None)
VO = pd.DataFrame(vo).set_axis(CL.index)
MV = pd.DataFrame(mv).set_axis(CL.index)
LD = pd.DataFrame(ld).set_axis(CL.index)
NI = pd.DataFrame(ni).set_axis(CL.index)
CL = CL.where(CL > 0)
idx = CL.index
NT, NS = CL.shape
print(f"面板 {CL.shape}  {idx[0].date()} ~ {idx[-1].date()}  ({time.time()-t0:.0f}s)")
assert (NT, NS) == (3297, 5232), f"锚点对不上 {(NT, NS)}"
cov = float(NI.notna().any().mean())
print(f"财务覆盖 {cov:.1%}(有财务标的占比)")
assert cov > 0.5, "财务列缺失 —— 先跑 data_prep/recover_panel.sh"

F = CL.ffill()
Fa = F.to_numpy(float)
ALIVE = CL.notna().to_numpy()
MA1 = F.rolling(100, min_periods=100).mean().to_numpy(float)
MA3 = F.rolling(300, min_periods=300).mean().to_numpy(float)
HI250 = F.rolling(250, min_periods=100).max().to_numpy(float)
V20 = VO.rolling(20, min_periods=10).mean().to_numpy(float)
V60 = VO.rolling(60, min_periods=30).mean().to_numpy(float)
MVa, LDa = MV.to_numpy(float), LD.to_numpy(float)
FMAX = pd.DataFrame(Fa[::-1]).rolling(H, min_periods=1).max().to_numpy(float)[::-1]
RET50 = np.full_like(Fa, np.nan)
RET50[50:] = Fa[50:] / Fa[:-50] - 1
RET250 = np.full_like(Fa, np.nan)
RET250[250:] = Fa[250:] / Fa[:-250] - 1
print(f"价格类指标完成  ({time.time()-t0:.0f}s)")

# 净利润同比加速:net_income 是**年初至今累计**,必须同报告期比,不能 shift(252)
# (§63 在这里踩过坑,README 已记录)。用「快照变化点」定位报告期。
NIv = NI.to_numpy(float)
NI_YOY = np.full_like(NIv, np.nan)
NI_ACC = np.zeros_like(NIv, dtype=bool)
for j in range(NS):
    s = NIv[:, j]
    ok = np.flatnonzero(np.isfinite(s))
    if len(ok) < 8:
        continue
    chg = ok[np.concatenate(([True], s[ok][1:] != s[ok][:-1]))]   # 快照跳变点
    if len(chg) < 6:
        continue
    vals = s[chg]
    yoy = np.full(len(chg), np.nan)
    yoy[4:] = np.where(np.abs(vals[:-4]) > 0, vals[4:] / np.abs(vals[:-4]) - 1, np.nan)
    acc = np.zeros(len(chg), bool)
    acc[5:] = np.isfinite(yoy[5:]) & np.isfinite(yoy[4:-1]) & (yoy[5:] > 0) \
        & (yoy[5:] > yoy[4:-1])
    for i, t in enumerate(chg):
        e = chg[i + 1] if i + 1 < len(chg) else NT
        NI_YOY[t:e, j] = yoy[i]
        NI_ACC[t:e, j] = acc[i]
print(f"财务同比加速完成  ({time.time()-t0:.0f}s)")

# 闸门:月末确认、次月首日生效(§73 修正口径,不用 §70 含前视的版本)
ym = idx.to_period("M")
last_td = {p: int(np.flatnonzero(ym == p)[-1]) for p in ym.unique()}
first_td = {p: int(np.flatnonzero(ym == p)[0]) for p in ym.unique()}
allm = sorted(last_td)
mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["close"])
mk.index = mk.index.tz_localize(None)
mkc = mk["close"].reindex(idx).ffill()
mm = mkc.resample("ME").last()
d = mm.ewm(span=12, adjust=False).mean() - mm.ewm(span=26, adjust=False).mean()
ih = d - d.ewm(span=9, adjust=False).mean()
ih.index = ih.index.to_period("M")
reg = {p: int(v > 0) for p, v in ih.items() if p in last_td}
RED_M = {p: bool(reg.get(allm[max(allm.index(p) - 1, 0)], 0)) for p in allm}
print(f"闸门完成,红轴月 {sum(RED_M.values())}/{len(RED_M)}  ({time.time()-t0:.0f}s)")


def rps(arr, t, base):
    v = np.where(base, arr[t], np.nan)
    return pd.Series(v).rank(pct=True).to_numpy(float) * 100


def signals(t, base, p):
    """返回 {名称: 布尔掩码},全部只用 t 及之前的信息。"""
    r50, r250 = rps(RET50, t, base), rps(RET250, t, base)
    m = np.where(base, MVa[t], np.nan)
    q20 = np.nanquantile(m[base], 0.2) if base.sum() else np.nan
    near = base & (Fa[t] >= HI250[t] * 0.90)
    up20 = base & (Fa[t] > MA1[t])
    bull_tol = base & (MA1[t] >= MA3[t] * 0.90)
    S = {
        "AGE_YOUNG 上市[1,3)年": base & (LDa[t] >= Y_LO) & (LDa[t] < Y_HI),
        "SMALL_MV 市值最小档": base & (m <= q20),
        "REGIME_RED 红轴": np.full(NS, RED_M.get(p, False)) & base,
        "RPS250>=90": base & (r250 >= 90),
        "RPS50>=95": base & (r50 >= 95),
        "MA_BULL 已多头排列": base & (MA1[t] > MA3[t]),
        "PRE_CROSS 金叉前夜": base & (MA1[t] < MA3[t]) & (MA1[t] >= MA3[t] * 0.90),
        "VOL_EXP 量比>=1.2": base & (V20[t] >= V60[t] * 1.2),
        "NEAR_HIGH 距新高<=10%": near,
        "NI_ACCEL 净利加速": base & NI_ACC[t],
        "USER_SETUP 用户四条件": base & (r50 >= 95) & up20 & near & bull_tol,
    }
    return {k: np.where(np.isfinite(MA3[t]) & np.isfinite(HI250[t]), v, False)
            for k, v in S.items()}


NAMES = list(signals(1000, ALIVE[1000], allm[30]).keys())
months = [p for p in allm if last_td[p] + H < NT]
print(f"信号 {len(NAMES)} 个,可用月 {len(months)} 个  ({time.time()-t0:.0f}s)")

rng = np.random.default_rng(SEED)
# acc[part][name][G] = [(obs_hit, rand_hit[NSEED]) per month]
acc = {"A": {n: {g: [] for g in GAINS} for n in NAMES},
       "B": {n: {g: [] for g in GAINS} for n in NAMES}}
case_hit = {}
CASES = [("300750", "2019-12-31"), ("688183", "2024-05-31"), ("600066", "2024-01-31")]
ci = {c: i for i, c in enumerate(CL.columns)}

for p in months:
    t = last_td[p]
    base = ALIVE[t] & np.isfinite(Fa[t]) & (Fa[t] > 0)
    if base.sum() < 200:
        continue
    S = signals(t, base, p)
    ratio = np.where(base, FMAX[min(t + 1, NT - 1)] / Fa[t] - 1, np.nan)
    m = np.where(base, MVa[t], np.nan)
    q = np.nanquantile(m[base], np.linspace(0, 1, NQ + 1)[1:-1])
    bands = []
    for i in range(NQ):
        lo = -np.inf if i == 0 else q[i - 1]
        hi = np.inf if i >= NQ - 1 else q[i]
        bands.append(np.flatnonzero(base & (m > lo) & (m <= hi)))
    baseline = S["REGIME_RED 红轴"] & S["AGE_YOUNG 上市[1,3)年"]

    for part, univ in (("A", base), ("B", baseline)):
        if univ.sum() < 50:
            continue
        ub = [b[univ[b]] for b in bands]
        for nm in NAMES:
            sel = S[nm] & univ
            if sel.sum() < MIN_N:
                continue
            for g in GAINS:
                hit = (ratio >= g)
                o, r = [], np.zeros(NSEED)
                nb = 0
                for i in range(NQ):
                    si = np.flatnonzero(sel & np.isin(np.arange(NS), ub[i])) \
                        if False else ub[i][sel[ub[i]]]
                    if len(si) < 3 or len(ub[i]) <= len(si):
                        continue
                    o.append(hit[si].mean())
                    r += np.array([hit[rng.choice(ub[i], len(si), replace=False)].mean()
                                   for _ in range(NSEED)])
                    nb += 1
                if nb:
                    acc[part][nm][g].append((float(np.mean(o)), r / nb))
    if len(acc["A"][NAMES[0]][1.0]) % 30 == 0 and acc["A"][NAMES[0]][1.0]:
        print(f"  {p}  ({time.time()-t0:.0f}s)", flush=True)

# 案例锚点
for code, ds in CASES:
    t = idx.get_indexer([pd.Timestamp(ds)], method="ffill")[0]
    base = ALIVE[t] & np.isfinite(Fa[t]) & (Fa[t] > 0)
    case_hit[code] = bool(signals(t, base, ym[t])["USER_SETUP 用户四条件"][ci[code]])
print(f"逐月累计完成  ({time.time()-t0:.0f}s)")


def summarize(part, nm, g):
    a = acc[part][nm][g]
    if len(a) < 12:
        return None
    o = np.mean([x[0] for x in a])
    r = np.mean([x[1] for x in a], axis=0)
    if r.mean() <= 0:
        return None
    lift = o / r.mean()
    lifts = o / np.where(r > 0, r, np.nan)
    return dict(n_mo=len(a), obs=o, rnd=float(r.mean()), lift=lift,
                lo=float(np.nanpercentile(lifts, 5)),
                hi=float(np.nanpercentile(lifts, 95)),
                p=float((r >= o).mean()))


rows = []
for part, title in (("A", "A 部分:单独检验(全市场,同市值档随机对照)"),
                    ("B", "B 部分:增量检验(红轴×上市[1,3)年 基线内部)")):
    print(f"\n{'='*112}\n{title}\n{'='*112}")
    print(f"{'信号':<26}{'门槛':<9}{'月数':>6}{'信号命中':>10}{'随机命中':>10}"
          f"{'lift':>8}{'lift区间':>18}{'p':>8}")
    for nm in NAMES:
        for g in GAINS:
            s = summarize(part, nm, g)
            if s is None:
                continue
            rows.append(dict(部分=part, 信号=nm, 门槛=f"≥{int(g*100)}%", **s))
            print(f"{nm:<26}{'≥'+str(int(g*100))+'%':<9}{s['n_mo']:>6}"
                  f"{s['obs']:>10.2%}{s['rnd']:>10.2%}{s['lift']:>8.2f}"
                  f"{f'[{s.lo:.2f}, {s.hi:.2f}]':>18}{s['p']:>8.4f}")
        print()

R = pd.DataFrame(rows)
print(f"{'='*112}\n事前判据 vs 实际(判据跑前写死,未放宽)\n{'='*112}")


def get(part, nm, g):
    x = R[(R.部分 == part) & (R.信号 == nm) & (R.门槛 == f"≥{int(g*100)}%")]
    return x.iloc[0] if len(x) else None


a_age = get("A", "AGE_YOUNG 上市[1,3)年", 5.0)
sm = [get("A", "SMALL_MV 市值最小档", g) for g in GAINS]
sm = [x for x in sm if x is not None]
anc_zero = all(abs(x["lift"] - 1.0) <= 0.10 for x in sm) if sm else False
print(f"  ① 零锚点 SMALL_MV lift ≈ 1.00±0.10   "
      + " / ".join(f"{x['lift']:.2f}" for x in sm) + f"   {'✓' if anc_zero else '✗'}")
if a_age is not None:
    print(f"     正向锚点 AGE_YOUNG ≥500%          "
          f"lift {a_age['lift']:.2f}  (信号 {a_age['obs']:.2%} / 随机 {a_age['rnd']:.2%})")

pass_a = R[(R.部分 == "A") & (R.门槛 == "≥500%") & (R.lift >= 1.3) & (R.p < 0.05)]
pass_b = R[(R.部分 == "B") & (R.门槛 == "≥500%") & (R.lift >= 1.2) & (R.p < 0.05)]
print(f"  ② A 部分 lift@500% ≥1.3 且 p<0.05     {len(pass_a)} 个"
      f"   {'✓ ' + ', '.join(pass_a.信号) if len(pass_a) else '✗'}")
print(f"  ③ B 部分 lift@500% ≥1.2 且 p<0.05     {len(pass_b)} 个"
      f"   {'✓ ' + ', '.join(pass_b.信号) if len(pass_b) else '✗'}")
print(f"  ⑤ 案例锚点 USER_SETUP 检出三只         "
      + " / ".join(f"{c} {'✓' if case_hit[c] else '✗'}" for c, _ in CASES))

if not anc_zero:
    print("\n  **零锚点不过:市值中性化没生效,整表作废。**")
elif len(pass_b):
    print(f"\n  **结论:在最好的基线之上,仍有信号能提高右尾密度 —— {', '.join(pass_b.信号)}**")
elif len(pass_a):
    print(f"\n  **结论:A 部分有信号通过({', '.join(pass_a.信号)}),但基线之上全部不通过 ——**")
    print("  **它们提高的那部分右尾,已经被「红轴×次新」这个基线包含了。**")
else:
    print("\n  **结论:没有任何日频信号能提高右尾密度。**")
    print("  仪表盘应只显示闸门与年龄,其余全部标 ⛔。")
print("\n  ⚠️ 右尾概率是**峰值口径**(上帝视角)。§70 实测兑现率 13%~55%,")
print("     §66 七只案例约 18%。**右尾密度提高 ≠ 钱变多。**")

R.to_csv(f"{SP}/righttail_lift.csv", index=False)
print(f"\n→ {SP}/righttail_lift.csv   ({time.time()-t0:.0f}s)")
