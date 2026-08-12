"""广度择时:把四十四节的"事后归纳"变成一个真正的样本外检验

═══ 假设从哪来(以及为什么不能在原地检验) ═══
四十四节发现方向型择时(MA200)失效,原因是结构性的:
B 期 69 期里 510300 有 100.0% 的时间在 MA200 之上,开关一次都没触发。
由此浮现的机制是"广度决定成败":

  2025 全年   普涨(等权 +49.56%)                 股池超额 **-18.2pp**
  2026 H1    极窄(仅29.3%上涨,中位 -14.6%)       股池超额 **+241.3pp**

RPS 动量筛选把仓位集中到少数正在跑的股票上 —— 普涨时"少数几只"跑不赢
"所有股票";极窄时只有那少数几只在涨。

**但这个假设是从 2023-10~2026-07 这三段里看出来的。**
在同一批数据上检验它、或在这三段上挑一个最合适的广度指标,都是自欺。

═══ 本脚本的核心设计:2017-2023 是干净的样本外 ═══
  OOS(唯一有资格下结论的) 2017-01 ~ 2023-09  我重建的股池
  IS (仅作参照,不作判据)  2023-10 ~ 2026-07  用户快照 A/B

重建股池可信的依据:四十三节实测**我算的 RPS 与用户的中位绝对差仅 0.07 分、
>90 阈值一致率 99.2%、各分档偏移均 <0.25 分**。

═══ 三个指标全部事前锁定、参数取常规值、结果全报不挑 ═══
  BR1 上涨股票占比(21日)      占比低  = 窄
  BR2 等权 − 510300(63日)    差值低  = 窄
  BR3 横截面收益离散度(21日)   离散度高 = 窄
阈值不设固定数值(最容易过拟合的地方),一律用该指标**过去3年滚动中位数**。

═══ 必须有的对照:随机开关 ═══
一个"有X%时间开仓"的开关本身就会改变收益。所以每个指标都再跑 200 次
**同开仓率的随机开关**。广度开关必须显著优于随机开关,
否则效果只来自"少在市场里待着",与广度无关。

═══ 判据(事前写死) ═══
以 OOS 为准,三条全满足才算成立:
  ① 相对等权基准的超额,加开关后改善 ≥ +5pp
  ② 优于同开仓率随机开关的 95% 分位(p<0.05)
  ③ 三个指标中至少两个同向(避免三选一的运气)
只满足①不满足② → 效果来自"减少暴露",不是广度。
OOS 不满足 → 假设被证伪,如实记录,不在 IS 上找补。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
COST = 0.003                      # 单边
N_RAND = 200
SEED = 20260811
OOS0, OOS1 = "2017-01-01", "2023-09-30"

t0 = time.time()
op, cl, niy, rev = {}, {}, {}, {}
for f in sorted(glob.glob(f"{DATA}/*.parquet")):
    k = os.path.basename(f)[:-8]
    if k == "510300":
        continue
    x = pd.read_parquet(f, columns=["open", "close", "ni_yoy_252", "revenue"])
    if x.empty:
        continue
    op[k] = pd.to_numeric(x["open"], errors="coerce")
    cl[k] = pd.to_numeric(x["close"], errors="coerce")
    niy[k] = pd.to_numeric(x["ni_yoy_252"], errors="coerce")
    rev[k] = pd.to_numeric(x["revenue"], errors="coerce")
OP = pd.DataFrame(op).sort_index(); OP.index = OP.index.tz_localize(None)
CL = pd.DataFrame(cl).set_axis(OP.index)
NIY = pd.DataFrame(niy).set_axis(OP.index); REV = pd.DataFrame(rev).set_axis(OP.index)
OP = OP.where(OP > 0); CL = CL.where(CL > 0)
idx = OP.index
OPa, CLa = OP.to_numpy(), CL.to_numpy()
col_of = {c: i for i, c in enumerate(OP.columns)}
print(f"面板 {OP.shape}  {idx.min().date()} ~ {idx.max().date()}  ({time.time()-t0:.0f}s)")
del op, cl, niy, rev

mk = pd.read_parquet(f"{DATA}/510300.parquet", columns=["open", "close"])
mk.index = mk.index.tz_localize(None)
MKO = pd.to_numeric(mk["open"], errors="coerce").reindex(idx).ffill()
MKC = pd.to_numeric(mk["close"], errors="coerce").reindex(idx).ffill()

RPS250 = CL.pct_change(250).rank(axis=1, pct=True) * 100

# ═══ 成长字段:改用修正后的口径(见 build_clean_growth.py 与第五十二节) ═══
# 原 `ni_yoy_252` = net_income/net_income.shift(252)-1,而 net_income 是 YTD 累计,
# 252 交易日 ≈ 1.04 年会跨报告期 —— 茅台 2023-05-04 得 -60.4%(单季比全年)。
# 污染量化:横截面「>0比例」按月份极差 25.1pp。
# 修正后(去累计 + 报告期对齐,已对官方财报核验 9 个单季全部吻合)极差 2.5~2.8pp。
def _load_clean_growth(index, columns):
    ni = pd.read_parquet(f"{SP}/clean_growth_ni_ttm_yoy.parquet").reindex(
        index=index, columns=columns)
    rv = pd.read_parquet(f"{SP}/clean_growth_rev_ttm_yoy.parquet").reindex(
        index=index, columns=columns)
    assert ni.notna().mean().mean() > 0.01, "clean_growth 净利字段几乎全空"
    assert rv.notna().mean().mean() > 0.01, "clean_growth 收入字段几乎全空"
    print(f"  成长字段 = **修正后 TTM 同比**(净利非空 {ni.notna().mean().mean():.1%}、"
          f"收入非空 {rv.notna().mean().mean():.1%})")
    return ni, rv

NIY, REVY = _load_clean_growth(OP.index, OP.columns)

# ---------------- 三个广度指标(事前锁定) ----------------
r1 = CL.pct_change(21)
BR1 = (r1 > 0).sum(axis=1) / r1.notna().sum(axis=1)          # 上涨股票占比
ew63 = CL.pct_change(63).mean(axis=1)
BR2 = ew63 - MKC.pct_change(63)                              # 等权 − 沪深300
BR3 = r1.std(axis=1)                                         # 横截面离散度
# "窄"的方向:BR1 低、BR2 低、BR3 高 → 统一成"数值高 = 窄"
BREADTH = {"BR1 上涨股票占比(21日)": -BR1,
           "BR2 等权−沪深300(63日)": -BR2,
           "BR3 横截面离散度(21日)": BR3}
THR = {k: v.rolling(756, min_periods=504).median() for k, v in BREADTH.items()}  # 过去3年
print(f"广度指标就绪  ({time.time()-t0:.0f}s)")


def rets(codes, e, x):
    out = []
    for c in codes:
        ci = col_of.get(c)
        if ci is None:
            continue
        a, b = OPa[e, ci], OPa[x, ci]
        if not np.isfinite(a) or a <= 0:
            continue
        if not np.isfinite(b) or b <= 0:
            seg = CLa[e:x + 1, ci]; seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                continue
            b = seg[-1]
        out.append(b / a - 1)
    return np.array(out)


def comp(r, d):
    r = np.asarray(r, float); d = np.asarray(d, float)
    ok = np.isfinite(r) & np.isfinite(d)
    if ok.sum() == 0:
        return np.nan
    t = np.prod(1 + r[ok]); y = d[ok].sum() / 252
    return t ** (1 / y) - 1 if t > 0 and y > 0 else -1.0


# ---------------- OOS:重建股池,周频 ----------------
fri = pd.Series(idx, index=idx).resample("W-FRI").last().dropna()
snaps = [d for d in fri if pd.Timestamp(OOS0) <= d <= pd.Timestamp(OOS1)]
print(f"OOS 快照 {len(snaps)} 期  {snaps[0].date()} ~ {snaps[-1].date()}")

rows = []
prev_pool = {"dual": set(), "rps": set()}
for i in range(len(snaps) - 1):
    e = idx.searchsorted(snaps[i], side="right")
    x = idx.searchsorted(snaps[i + 1], side="right")
    if e >= len(idx) or x >= len(idx) or x <= e:
        continue
    p = e - 1
    alive = np.isfinite(OP.iloc[e]) & np.isfinite(OP.iloc[x])
    hot = (RPS250.iloc[p] > 90).fillna(False) & alive
    dual = hot & (NIY.iloc[p] > 0).fillna(False) & (REVY.iloc[p] > 0).fillna(False)
    rec = {"snap": snaps[i], "e": e, "x": x, "days": x - e}
    for nm, mask in (("rps", hot), ("dual", dual)):
        codes = set(OP.columns[mask])
        r = rets(list(codes), e, x)
        rec[f"ret_{nm}"] = r.mean() if len(r) else np.nan
        rec[f"n_{nm}"] = len(codes)
        pv = prev_pool[nm]
        rec[f"to_{nm}"] = 1.0 if not pv else 1 - len(pv & codes) / max(len(codes), 1)
        prev_pool[nm] = codes
    av = OP.columns[alive]
    br = OP.iloc[x][av] / OP.iloc[e][av] - 1
    w = (1 + br) / (1 + br).sum()
    rec["bench"] = br.mean() - 2 * COST * (0.5 * np.abs(w - 1 / len(w)).sum())
    for k in BREADTH:
        v, th = BREADTH[k].iat[p], THR[k].iat[p]
        rec[k] = bool(np.isfinite(v) and np.isfinite(th) and v > th)   # True = 窄 = 开仓
    rows.append(rec)

O = pd.DataFrame(rows).dropna(subset=["ret_dual", "ret_rps"])
for nm in ("rps", "dual"):
    O[f"net_{nm}"] = O[f"ret_{nm}"] - 2 * COST * O[f"to_{nm}"]
print(f"OOS 有效期数 {len(O)}  ({time.time()-t0:.0f}s)")

print(f"\n{'#'*114}")
print("验证1 OOS 基线自检(不加择时)")
print(f"{'#'*114}")
bench = comp(O.bench, O.days)
for nm, disp in (("rps", "RPS250>90 全池"), ("dual", "RPS250>90 + 双增长")):
    a = comp(O[f"net_{nm}"], O.days)
    print(f"  {disp:<22} 年化 {a:>+8.2%}   每期只数中位 {O[f'n_{nm}'].median():>5.0f}   "
          f"换手中位 {O[f'to_{nm}'].median():>5.1%}   相对等权 {(a-bench)*100:>+6.1f}pp")
print(f"  {'全市场等权(基准)':<22} 年化 {bench:>+8.2%}")

print(f"\n{'='*114}\n验证2 各广度指标的开仓比例(OOS)\n{'='*114}")
for k in BREADTH:
    print(f"  {k:<26} 开仓期数占比 **{O[k].mean():>6.1%}**")

rng = np.random.default_rng(SEED)
print(f"\n{'='*114}")
print("OOS 主结果(判据:①超额改善≥+5pp ②优于同开仓率随机开关95%分位 ③三选二同向)")
print(f"{'='*114}")
print(f"{'配置':<30}{'基线年化':>11}{'择时后':>11}{'基线超额':>11}{'择时后超额':>12}"
      f"{'改善':>9}{'随机p':>8}")
verdict = {}
for nm, disp in (("dual", "RPS+双增长"), ("rps", "仅RPS(不依赖财务)")):
    base = comp(O[f"net_{nm}"], O.days)
    for k in BREADTH:
        m = O[k].to_numpy()
        timed = np.where(m, O[f"net_{nm}"], 0.0)
        a = comp(timed, O.days)
        gain = a - base
        rate = m.mean()
        null = []
        for _ in range(N_RAND):
            rm = rng.random(len(O)) < rate
            null.append(comp(np.where(rm, O[f"net_{nm}"], 0.0), O.days) - base)
        null = np.array([v for v in null if np.isfinite(v)])
        p = float((null >= gain).mean())
        verdict[(nm, k)] = {"base": base, "timed": a, "gain": gain, "p": p,
                            "ex0": base - bench, "ex1": a - bench}
        print(f"{disp+' / '+k:<30}{base:>+11.2%}{a:>+11.2%}{(base-bench)*100:>+10.1f}pp"
              f"{(a-bench)*100:>+11.1f}pp{gain*100:>+8.1f}pp{p:>8.3f}"
              f"{'  **' if (gain >= 0.05 and p < 0.05) else ''}")

print(f"\n{'='*114}\n判据判定(仅看 OOS)\n{'='*114}")
for nm, disp in (("dual", "RPS+双增长"), ("rps", "仅RPS")):
    passed = [k for k in BREADTH
              if verdict[(nm, k)]["gain"] >= 0.05 and verdict[(nm, k)]["p"] < 0.05]
    only1 = [k for k in BREADTH if verdict[(nm, k)]["gain"] >= 0.05]
    print(f"  [{disp}] 满足①的 {len(only1)}/3 个;**①②都满足的 {len(passed)}/3 个**"
          f"  → {'**成立**' if len(passed) >= 2 else ('三选一的运气,不算发现' if len(passed)==1 else '证伪')}")
    for k in BREADTH:
        v = verdict[(nm, k)]
        r1_ = "✓" if v["gain"] >= 0.05 else "✗"
        r2_ = "✓" if v["p"] < 0.05 else "✗"
        print(f"      {k:<26} ①改善{v['gain']*100:>+6.1f}pp {r1_}   ②随机p={v['p']:.3f} {r2_}")

O.to_csv(f"{SP}/breadth_timing_oos.csv", index=False)

# ---------------- IS:仅作参照 ----------------
print(f"\n{'='*114}")
print("IS 参照(2023-10~2026-07,用户快照)—— **假设诞生于此,不作判据**")
print(f"{'='*114}")
SEGS = [("A 期 2023-10~2024-12", "A", None, None),
        ("B·2025 全年", "B", "2025-01-01", "2025-12-31"),
        ("B·2026 H1", "B", "2026-01-01", "2026-12-31")]
IS = {}
for tag in ("A", "B"):
    P = pd.read_csv(f"{SP}/rps_timing_periods_{tag}.csv", parse_dates=["snap"])
    for k in BREADTH:
        P[k] = [bool(np.isfinite(BREADTH[k].iat[int(r.e) - 1])
                     and np.isfinite(THR[k].iat[int(r.e) - 1])
                     and BREADTH[k].iat[int(r.e) - 1] > THR[k].iat[int(r.e) - 1])
                for _, r in P.iterrows()]
    IS[tag] = P
print(f"{'期间':<22}{'指标':<26}{'开仓占比':>10}{'基线':>11}{'广度择时后':>12}"
      f"{'等权基准':>11}{'择时后超额':>12}")
for label, tag, d0, d1 in SEGS:
    P = IS[tag]
    g = P if d0 is None else P[(P.snap >= d0) & (P.snap <= d1)]
    b = comp(g.bench, g.days)
    base = comp(g.net_dual, g.days)
    for k in BREADTH:
        a = comp(np.where(g[k], g.net_dual, 0.0), g.days)
        print(f"{label:<22}{k:<26}{g[k].mean():>10.1%}{base:>+11.2%}{a:>+12.2%}"
              f"{b:>+11.2%}{(a-b)*100:>+11.1f}pp")

for tag in ("A", "B"):
    IS[tag].to_csv(f"{SP}/breadth_timing_is_{tag}.csv", index=False)
print(f"\n耗时 {time.time()-t0:.0f}s   Saved: breadth_timing_oos.csv, breadth_timing_is_A/B.csv")
