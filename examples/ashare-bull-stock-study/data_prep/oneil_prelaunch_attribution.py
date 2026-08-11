"""欧奈尔式"起涨前特征"归因:翻倍股在突破基底那一刻之前长什么样?

与本session此前的 winner_attribution.py 以及 DeepSeek 的 E 章的**关键
区别**:两者都用**年初快照**(1月第一个交易日)代理"起涨前",但一只2024年
翻倍的股票可能3月才启动,年初快照离真实起涨点差两个月。欧奈尔研究的是
**枢轴点**(突破基底那一刻)的特征。本脚本改用突破点定义。

起涨点定义(D):
  - close > 前60个交易日最高close           (60日新高突破)
  - 且突破前60日处于基底:(max-min)/min < 50%  (排除已在暴涨途中的)
  - 且距上一次突破 >= 60 个交易日            (同一波行情只记一次)
  特征全部取 **D-1 收盘**(突破前一天),严格无前视。

标签:D 之后 252 个交易日内,最大累计涨幅是否 > 100%。

**对照组是设计核心**:不是"全市场",而是**同样发生了60日突破、但没有
翻倍的股票**。这才隔离出"在所有突破里,什么区分了最终翻倍的那批"——
这正是欧奈尔的问题形式。

贯穿全程的认识:**归因 ≠ 可交易性**。符合牛股特征的股票里绝大多数不是
牛股(本session已实测:rmdd20 归因说深回撤出牛股,按此交易却亏6.15%)。
所以本脚本报的是 **精确率 P(翻倍|组合)** 而非"牛股里多少比例具备该特征",
且最终必须由回测检验(阶段5,另一脚本)。

搜索空间纪律(ETF阶段已量化:空间每扩大10倍,纯噪音best-of-N抬高约
1.16pp;扩大1000倍时噪音底噪会反超真实样本内最优):
  - 只用单因子已显示单调区分度的特征,**方向事前锁定**,不做方向搜索
  - 只测二元/三元组合,不做更高阶
  - 样本数<30 的组合不进排序
  - **打乱标签跑100次得到噪音底噪分布**,真实最佳必须显著超出才算发现
"""
import glob
import os
import time
from itertools import combinations

import numpy as np
import pandas as pd

SCRATCH = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = f"{SCRATCH}/oxq_stock_market_fixed"   # 已修复复权与float_mv缺陷,见三十二节

BREAKOUT_WIN = 60      # 60日新高
BASE_MAX_RANGE = 0.50  # 突破前60日振幅上限(基底)
MIN_GAP = 60           # 两次突破最小间隔
FWD_WIN = 252          # 突破后观察窗口
WIN_THRESHOLD = 1.00   # 翻倍
MIN_CELL = 30
N_SHUFFLE = 100

PERIODS = {
    "2013-2020": (2013, 2020),
    "2021-2023(样本内)": (2021, 2023),
    "2024-2026(样本外)": (2024, 2026),
}

t0 = time.time()
files = [f for f in sorted(glob.glob(f"{DATA_DIR}/*.parquet")) if not f.endswith("510300.parquet")]
print(f"个股文件: {len(files)}")

COLS = ["close", "high", "volume", "float_mv", "is_limit_up", "is_limit_down",
        "turnover", "net_income", "bp_correct", "eps"]
d = {c: {} for c in COLS}
for f in files:
    code = os.path.basename(f)[:-8]
    try:
        x = pd.read_parquet(f, columns=COLS)
    except Exception:
        continue
    if x.empty:
        continue
    for c in COLS:
        d[c][code] = pd.to_numeric(x[c], errors="coerce") if c not in ("is_limit_up", "is_limit_down") \
            else x[c].astype(float)

px = pd.DataFrame(d["close"]).sort_index()
vol = pd.DataFrame(d["volume"]).reindex_like(px)
fmv = pd.DataFrame(d["float_mv"]).reindex_like(px)
LU = pd.DataFrame(d["is_limit_up"]).reindex_like(px)
LD = pd.DataFrame(d["is_limit_down"]).reindex_like(px)
tovr = pd.DataFrame(d["turnover"]).reindex_like(px)
ni = pd.DataFrame(d["net_income"]).reindex_like(px)
bp = pd.DataFrame(d["bp_correct"]).reindex_like(px)
eps = pd.DataFrame(d["eps"]).reindex_like(px)
print(f"面板: {px.shape[0]} × {px.shape[1]}  ({time.time()-t0:.0f}s)")

# ---------- 特征(全部为"截至当日"的历史信息) ----------
print("计算特征...")
rets = px.pct_change()
roll_max = px.rolling(BREAKOUT_WIN, min_periods=BREAKOUT_WIN).max()
roll_min = px.rolling(BREAKOUT_WIN, min_periods=BREAKOUT_WIN).min()
base_range = (roll_max - roll_min) / roll_min.replace(0, np.nan)

mom_60 = px.pct_change(60)
mom_250 = px.pct_change(250)
rps_60 = mom_60.rank(axis=1, pct=True) * 100
rps_250 = mom_250.rank(axis=1, pct=True) * 100
fmv_pct = fmv.rank(axis=1, pct=True)
rvol_20 = rets.rolling(20, min_periods=10).std()
logp = np.log(px.where(px > 0))
rmdd_20 = logp.rolling(20).apply(lambda w: float(np.min(w - np.maximum.accumulate(w))), raw=True)
lu_252 = LU.rolling(252, min_periods=120).sum()
ld_252 = LD.rolling(252, min_periods=120).sum()
tovr_pct = tovr.rolling(20, min_periods=10).mean().rank(axis=1, pct=True)
vol_ratio = vol.rolling(20, min_periods=10).mean() / vol.rolling(60, min_periods=30).mean().replace(0, np.nan)
ni_yoy = (ni / ni.shift(252) - 1).replace([np.inf, -np.inf], np.nan)
ep = (eps / px.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
dist_high_252 = px / px.rolling(252, min_periods=120).max() - 1

FEATURES = {
    "BP(价值)": bp,
    "EP(盈利收益率)": ep,
    "C_净利润同比": ni_yoy,
    "L_RPS250": rps_250,
    "RPS60": rps_60,
    "N_距252日新高": dist_high_252,
    "S1_流通市值分位": fmv_pct,
    "S2_量比": vol_ratio,
    "波动率20日": rvol_20,
    "RMDD20": rmdd_20,
    "上年涨停占比": lu_252,
    "上年跌停占比": ld_252,
    "换手率分位": tovr_pct,
}
print(f"  {len(FEATURES)} 个特征  ({time.time()-t0:.0f}s)")

# ---------- 起涨点识别 ----------
print("\n识别起涨点(60日新高突破 + 基底约束)...")
prev_max = roll_max.shift(1)          # 前60日(不含当日)最高
prev_base = base_range.shift(1)
is_breakout = (px > prev_max) & (prev_base < BASE_MAX_RANGE)

fwd_max = px.rolling(FWD_WIN, min_periods=20).max().shift(-FWD_WIN)
fwd_gain = fwd_max / px - 1

idx = px.index
events = []
for code in px.columns:
    b = is_breakout[code]
    hits = np.flatnonzero(b.to_numpy())
    last = -10**9
    for pos in hits:
        if pos - last < MIN_GAP:
            continue
        if pos == 0:
            continue
        last = pos
        D = idx[pos]
        g = fwd_gain[code].iloc[pos]
        if pd.isna(g):
            continue
        rec = {"code": code, "D": D, "year": D.year, "fwd_gain": float(g),
               "winner": bool(g > WIN_THRESHOLD)}
        for fname, fdf in FEATURES.items():
            v = fdf[code].iloc[pos - 1]     # D-1,突破前一天
            rec[fname] = float(v) if pd.notna(v) else np.nan
        events.append(rec)

ev = pd.DataFrame(events)
ev.to_csv(f"{SCRATCH}/oneil_prelaunch_events_fixed.csv", index=False)
print(f"  突破事件: {len(ev):,}  其中翻倍: {int(ev['winner'].sum()):,} "
      f"({ev['winner'].mean():.2%})  ({time.time()-t0:.0f}s)")

print("\n逐年突破事件与翻倍率:")
yr = ev.groupby("year").agg(突破数=("winner", "size"), 翻倍数=("winner", "sum"))
yr["翻倍率"] = yr["翻倍数"] / yr["突破数"]
print(yr.to_string())

# ---------- 起涨点抽查(人工确认D在主升浪之前) ----------
print(f"\n{'='*100}\n抽查:随机3个翻倍事件,确认起涨点在主升浪之前\n{'='*100}")
w = ev[ev.winner].sample(3, random_state=7)
for _, r in w.iterrows():
    c, D = r["code"], r["D"]
    pos = idx.get_loc(D)
    p_before = px[c].iloc[max(0, pos - 60):pos].agg(["min", "max"])
    p_at = px[c].iloc[pos]
    p_after = px[c].iloc[pos:pos + FWD_WIN].max()
    print(f"  [{c}] 起涨日 {D.date()}  突破前60日区间 {p_before['min']:.2f}~{p_before['max']:.2f}  "
          f"突破日 {p_at:.2f}  之后252日最高 {p_after:.2f}  (涨幅 {p_after/p_at-1:+.1%})")

# ---------- 单因子 lift ----------
print(f"\n{'='*100}\n单因子在起涨点的区分度 (lift = P(翻倍|档位) ÷ 该时段突破样本翻倍率)\n{'='*100}")
single_rows = []
for pname, (ys, ye) in PERIODS.items():
    sub = ev[(ev.year >= ys) & (ev.year <= ye)]
    if len(sub) < 100:
        continue
    base_rate = sub["winner"].mean()
    for fname in FEATURES:
        v = sub[fname].dropna()
        if len(v) < 5 * MIN_CELL:
            continue
        try:
            q = pd.qcut(v.rank(method="first"), 5, labels=False)
        except ValueError:
            continue
        for qi in range(5):
            mem = v.index[q == qi]
            if len(mem) < MIN_CELL:
                continue
            wr = sub.loc[mem, "winner"].mean()
            single_rows.append({"period": pname, "feature": fname, "quantile": qi + 1,
                                "n": len(mem), "winner_rate": wr,
                                "lift": wr / base_rate if base_rate > 0 else np.nan})

sf = pd.DataFrame(single_rows)
sf.to_csv(f"{SCRATCH}/oneil_prelaunch_single_fixed.csv", index=False)

for pname in PERIODS:
    s = sf[sf.period == pname]
    if s.empty:
        continue
    piv = s.pivot(index="feature", columns="quantile", values="lift")
    piv.columns = [f"Q{c}" for c in piv.columns]
    piv["Q5-Q1"] = piv["Q5"] - piv["Q1"]
    print(f"\n  [{pname}]  基准翻倍率 {ev[(ev.year>=PERIODS[pname][0])&(ev.year<=PERIODS[pname][1])]['winner'].mean():.2%}")
    print(piv.sort_values("Q5-Q1", key=abs, ascending=False).round(2).to_string())

print(f"\n完成单因子 ({time.time()-t0:.0f}s)")
print(f"Saved: oneil_prelaunch_events.csv, oneil_prelaunch_single.csv")
