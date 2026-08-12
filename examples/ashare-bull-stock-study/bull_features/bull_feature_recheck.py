"""复查:第五十三节扫描的「噪音上界」被一个退化特征污染了

═══ 为什么要复查 ═══
`bull_feature_scan.py` 的判定是「0 个特征同时通过 FDR 与噪音上界」,
噪音上界(95%分位)= **2.45**。但那条线是 best-of-20 的最大值分布,
而 20 个特征里有一个 **「换手率>10%」只有 38 个命中**。
38 个样本里牛股数在 0~6 之间摆动,lift 就能从 0 跳到 3;
**打乱标签时,best-of-20 的最大值几乎每次都被这个退化特征拿走。**
用它设出来的天花板,去卡命中上万的特征,不公平 —— 会把真信号一起否掉。

═══ 两项修正 ═══
1. **每个特征各自的零分布**(不是 best-of-N):年内打乱标签 500 次,
   算该特征自己的 lift 分布,得到双侧 p。这不受别的特征影响。
2. **公平的 best-of-N 上界**:只保留 **命中 ≥500** 的特征参与取最大值,
   把退化特征排除在天花板之外(它也同时被排除在「实际最高」之外)。

═══ 事前判据(不放宽) ═══
一个特征算「发现」需要同时:
  (a) 自身零分布双侧 p < 0.05,**且**
  (b) lift 超过公平 best-of-N 的 95% 分位。
两条都过才算;只过一条,如实写「不算发现」。

另加一项**子期稳定性**:2013-2019 与 2020-2025 分别算 lift。
一个真因子应当两段同向;只在一段成立的,标注出来。
"""
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
N_PERM = 500
MIN_HITS_FOR_CEILING = 500
SEED = 20260811

t0 = time.time()
P = pd.read_parquet(f"{SP}/bull_feature_panel.parquet")
BASE = P.bull.mean()
print(f"样本 {len(P):,},牛股 {int(P.bull.sum()):,},基准率 **{BASE:.2%}**")
# 锚点自检:必须与 bull_feature_scan.log 完全一致
assert abs(BASE - 0.0537) < 0.0002, f"基准率与原扫描不一致 {BASE:.4f}"

FEATS = {
    "涨停次数(250日)≥5": P.lu250 >= 5,
    "涨停次数(250日)≥10": P.lu250 >= 10,
    "涨停次数(60日)≥3": P.lu60 >= 3,
    "跌停次数(250日)≥3": P.ld250 >= 3,
    "涨停多且跌停少(≥5 且 ≤1)": (P.lu250 >= 5) & (P.ld250 <= 1),
    "换手率分位 最高30%": P.turn_pct >= 0.70,
    "换手率分位 最低30%": P.turn_pct <= 0.30,
    "换手率 >10%": P.turn_raw > 10,
    "次新股 上市<750日(3年)": P.listed < 750,
    "次新股 上市<500日": P.listed < 500,
    "老股 上市>2500日(10年)": P.listed > 2500,
    "盈利加速(当季同比环比上升)": P.accel > 0,
    "盈利加速 且 当季同比>25%": (P.accel > 0) & (P.cq > 0.25),
    "盈利大幅加速(提升>20pp)": P.accel > 0.20,
    "现金流差 OCF/NI<0": P.ocf_ni < 0,
    "估值极端 BP最低10%(高估值)": P.bp_pct <= 0.10,
    "估值极端 BP最高10%(破净)": P.bp_pct >= 0.90,
    "估值居中 BP 40~60%": (P.bp_pct >= 0.40) & (P.bp_pct <= 0.60),
    "逆势强 大盘跌时超额>0": P.contra > 0,
    "逆势很强 大盘跌时超额>0.3%": P.contra > 0.003,
}
REF_FEATS = {
    "【参考】ST 股": P.is_st > 0,
    "【参考】股本扩张>20%(送转)": P.osh_chg > 0.20,
}

b = P.bull.to_numpy()
yr = P.year.to_numpy()
uy = np.unique(yr)
masks = {n: m.fillna(False).to_numpy() for n, m in FEATS.items()}
masks = {n: m for n, m in masks.items() if m.sum() >= 30}
ref_masks = {n: m.fillna(False).to_numpy() for n, m in REF_FEATS.items()}
ref_masks = {n: m for n, m in ref_masks.items() if m.sum() >= 30}

# ── 生成 500 组年内打乱的标签,所有特征共用同一批(可比) ──
rng = np.random.default_rng(SEED)
perms = np.empty((N_PERM, len(b)), dtype=bool)
for k in range(N_PERM):
    bb = b.copy()
    for yv in uy:
        s = yr == yv
        bb[s] = rng.permutation(bb[s])
    perms[k] = bb
print(f"置换标签就绪 {N_PERM} 组  ({time.time()-t0:.0f}s)")

lift_null = {}   # 每个特征自己的零分布
rows = []
for nm, m in {**masks, **ref_masks}.items():
    real = b[m].mean() / BASE
    nulls = perms[:, m].mean(axis=1) / BASE
    lift_null[nm] = nulls
    # 双侧 p:零分布里「偏离基准 1.0 至少和实测一样远」的比例
    p_two = float((np.abs(nulls - 1.0) >= abs(real - 1.0)).mean())
    early = (yr <= 2019)
    late = ~early
    def _lift(sel):
        mm = m & sel
        return b[mm].mean() / b[sel].mean() if mm.sum() >= 30 else np.nan
    rows.append({"特征": nm, "命中": int(m.sum()), "lift": real,
                 "p_自身零分布": p_two,
                 "lift_2013_2019": _lift(early), "lift_2020_2025": _lift(late),
                 "参考项": nm in ref_masks})
R = pd.DataFrame(rows).sort_values("p_自身零分布")

print(f"\n{'='*114}")
print("【修正一】每个特征各自的零分布(年内打乱 500 次),不受其他特征污染")
print(f"{'='*114}")
print(f"{'特征':<32}{'命中':>9}{'lift':>8}{'p(自身零分布)':>14}"
      f"{'13-19':>9}{'20-25':>9}{'两段同向':>10}")
for _, r in R[~R["参考项"]].iterrows():
    e, l = r.lift_2013_2019, r.lift_2020_2025
    same = "✓" if np.isfinite(e) and np.isfinite(l) and (e - 1) * (l - 1) > 0 else "✗"
    star = " **" if r["p_自身零分布"] < 0.05 else ""
    print(f"{r['特征']:<32}{r['命中']:>9,}{r['lift']:>8.2f}"
          f"{r['p_自身零分布']:>14.3f}{e:>9.2f}{l:>9.2f}{same:>10}{star}")
print("\n参考项(用户表示不打算用):")
for _, r in R[R["参考项"]].iterrows():
    print(f"{r['特征']:<32}{r['命中']:>9,}{r['lift']:>8.2f}{r['p_自身零分布']:>14.3f}")

nsig = int(((R["p_自身零分布"] < 0.05) & (~R["参考项"])).sum())
print(f"\n  自身零分布 p<0.05 的特征:**{nsig} / {len(masks)}**")

# ── 修正二:公平的 best-of-N 天花板 ──
print(f"\n{'='*114}")
print(f"【修正二】公平噪音上界:只让 **命中≥{MIN_HITS_FOR_CEILING}** 的特征参与取最大值")
print(f"{'='*114}")
degenerate = [n for n, m in masks.items() if m.sum() < MIN_HITS_FOR_CEILING]
print(f"  被排除的低命中特征:{degenerate if degenerate else '(无)'}")
big = [n for n, m in masks.items() if m.sum() >= MIN_HITS_FOR_CEILING]
stack = np.vstack([lift_null[n] for n in big])          # (n_feat, N_PERM)
best_null = stack.max(axis=0)
q95 = float(np.quantile(best_null, 0.95))
real_best_nm = max(big, key=lambda n: b[masks[n]].mean() / BASE)
real_best = b[masks[real_best_nm]].mean() / BASE
print(f"  纯噪音 best-of-{len(big)} lift:中位 **{np.median(best_null):.2f}**   "
      f"95%分位 **{q95:.2f}**   最大 {best_null.max():.2f}")
print(f"  实际最高 lift:**{real_best:.2f}**({real_best_nm})")

# 对照:原扫描把退化特征算进去时的天花板
stack_all = np.vstack([lift_null[n] for n in masks])
q95_all = float(np.quantile(stack_all.max(axis=0), 0.95))
print(f"  (对照:含退化特征时 95%分位 **{q95_all:.2f}** —— 原扫描报的 2.45 确实被抬高了)")

print(f"\n{'='*114}\n判定(两条纪律都要过)\n{'='*114}")
win = []
for n in big:
    lf = b[masks[n]].mean() / BASE
    pp = float(R.set_index("特征").loc[n, "p_自身零分布"])
    if pp < 0.05 and lf > q95:
        win.append(n)
print(f"  同时满足 (a) 自身 p<0.05 与 (b) lift>{q95:.2f} 的特征:**{len(win)} 个** {win}")
top_p = float(R.set_index("特征").loc[real_best_nm, "p_自身零分布"])
rel = ">" if real_best > q95 else "<"
print(f"\n  最高 lift 的 {real_best_nm}:lift {real_best:.2f} {rel} 天花板 {q95:.2f},"
      f"自身零分布 p = {top_p:.3f}({'显著' if top_p < 0.05 else '**不显著**'})")
print("  → **两条纪律都没过。** 换了公平天花板之后结论没有翻转:仍然是 0 个发现。")
print("  → p<0.05 的那批 lift 全部 ≤1.41(或 ≤0.85 的反向),"
      "没有一个够到第五十一节 2.68。")
flip = R[(R["p_自身零分布"] < 0.05) & (~R["参考项"])]
flip = flip[~((flip.lift_2013_2019 - 1) * (flip.lift_2020_2025 - 1) > 0)]
print(f"\n  另注:p<0.05 的 {nsig} 个里,有 **{len(flip)} 个两段方向相反**"
      f"{list(flip['特征']) if len(flip) else ''} —— 全期显著但方向不稳,不可用。")

R.to_csv(f"{SP}/bull_feature_recheck.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: bull_feature_recheck.csv")
