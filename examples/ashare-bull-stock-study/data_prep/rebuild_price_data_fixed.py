"""重建价格数据到 oxq_stock_market_fixed/,修复两个已确诊缺陷。

═══ 缺陷1:复权做了两次(致命) ═══
`build_oxq_stock_data.py:99` 做了 `hfq_close = close * factor`,但
`mktdata_enriched` 的 `close` **本身就已经是复权价**。证据:
  - 2016年684个"股本增加>40%"的送转事件,源表 close 当日收益中位数
    -1.22%,|收益|>25% 的比例 **0.0%** —— 未复权的价格必然在10转10当天
    腰斩,它没有,所以它是复权价
  - `adj_factor` 14年基本恒定(300122: 6.658~6.671),说明
    `hfq_close = close × 常数`,两列是同一序列的不同缩放
后果:每次送转都被凭空乘出一个向上跳空。全量实测 **3,479/5,232 = 66.5%**
的标的含"物理上不可能的日收益"(超过涨跌停限制),累计收益中位数虚高
**44.8个百分点**,89.1% 的标的收益被高估。

修复:`close/open/high/low` 直接用源表的值,不再重新复权。

═══ 缺陷2:float_mv 含前视(同样致命,但更隐蔽) ═══
源表 `float_mv = close × outstanding_share`,其中 close 是**前复权价**
(历史价被未来的送转按比例调低),而 outstanding_share 是**当时的真实
股本**。两者口径不一致。

实测 300122 在 2016-05-12:
  表内 float_mv = 8.38 × 4.267亿 = **35.8亿**
  真实市值     = 26.79 × 4.267亿 = **114.3亿**   (26.79 = amount/volume 实际成交均价)
**低估 3.2 倍,而这个倍数恰好等于该股此后累计送转的倍数。**

这是教科书式的前视:一只股票的历史市值取决于它**未来会不会高送转**。
而 A 股里高送转恰恰集中在已经大涨的股票上 —— 三十节"流通市值是13个
特征里唯一三段方向一致的锚"、三十一节"所有Top10组合都含小市值",
**很可能大部分是这个前视造出来的**。

修复:重建真实成交价 `raw_close = close / factor(t)`,其中 factor(t) =
t 之后所有送转/分红乘数的连乘;再用 `float_mv = raw_close × 股本`。

**验证方式**:`amount/volume` 是真实成交均价(amount 与 volume 都是未
复权的原始值),与重建的 raw_close 相互独立,可作为交叉验证。

═══ 输出 ═══
新目录 oxq_stock_market_fixed/,旧目录原封不动保留以便新旧对比。
列结构与旧文件一致(LocalMarketDataProvider 兼容),额外新增:
  raw_close        真实成交价(未复权)
  float_mv_legacy  旧的(错误的)float_mv,保留用于量化影响
财务列(eps/revenue/net_income/book_value_per_share/roe/operating_cash_flow)
从旧文件按日期索引搬运——它们由 merge_asof 生成,与价格无关,不受影响。
`bp_correct` 用 raw_close 重算。
横截面派生列(fmv_pct 等)本脚本不产出,由第二个脚本统一重算。
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
MKT = f"{SP}/mktdata_enriched"
OTH = f"{SP}/mktdata_enriched_others"
OLD = f"{SP}/oxq_stock_market_with_fundamentals"
OUT = f"{SP}/oxq_stock_market_fixed"
YEARS = list(range(2013, 2027))

FIN_COLS = ["eps", "revenue", "net_income", "book_value_per_share", "roe",
            "operating_cash_flow"]

os.makedirs(OUT, exist_ok=True)
t0 = time.time()

print("=" * 96)
print("阶段1: 读取源表")
print("=" * 96)
frames = [pd.read_parquet(f"{MKT}/{y}.parquet") for y in YEARS if os.path.exists(f"{MKT}/{y}.parquet")]
full = pd.concat(frames, ignore_index=True)
full["date"] = pd.to_datetime(full["date"])
full = full.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"])
print(f"  {len(full):,} 行, {full.code.nunique():,} 只标的  ({time.time()-t0:.0f}s)")

ca = pd.read_parquet(f"{OTH}/corporate_actions.parquet")
ca["ex_date"] = pd.to_datetime(ca["ex_date"])
# 同日多笔事件必须合并后再算乘数:2023-06-16 的 stock_dividend 0.4 +
# capitalization 0.1 是股本共 ×1.5,乘数应为 1/1.5;分开算成
# 1/1.4 × 1/1.1 = 1/1.54 会有 2.6% 的误差。
share_ev = (ca[ca.action_type.isin(["stock_dividend", "capitalization"])]
            .groupby(["code", "ex_date"])["ratio"].sum().rename("share_ratio"))
cash_ev = (ca[ca.action_type == "cash_dividend"]
           .groupby(["code", "ex_date"])["ratio"].sum().rename("cash_amt"))
ev = pd.concat([share_ev, cash_ev], axis=1).reset_index()
ev["share_ratio"] = ev["share_ratio"].fillna(0.0)
ev["cash_amt"] = ev["cash_amt"].fillna(0.0)
print(f"  公司行动: {len(ca):,} 条原始 → {len(ev):,} 个(标的,除权日)  "
      f"{ev.code.nunique():,} 只标的")
ev_by_code = {c: g.sort_values("ex_date") for c, g in ev.groupby("code", sort=False)}

print("\n" + "=" * 96)
print("阶段2: 逐标的重建")
print("=" * 96)

old_files = {os.path.basename(f)[:-8] for f in glob.glob(f"{OLD}/*.parquet")}
n_ok = n_nofin = n_err = 0
valid_rows = []

for code, g in full.groupby("code", sort=False):
    try:
        g = g.sort_values("date").reset_index(drop=True)
        dates = g["date"].to_numpy()
        close = g["close"].to_numpy(dtype=float)

        # ---- factor(t) = t 之后所有事件乘数的连乘 ----
        log_mult = np.zeros(len(g))
        e = ev_by_code.get(code)
        if e is not None and len(e):
            pos_arr = np.searchsorted(dates, e["ex_date"].to_numpy())
            for pos, sr, cash in zip(pos_arr, e["share_ratio"].to_numpy(),
                                     e["cash_amt"].to_numpy()):
                if pos >= len(log_mult):
                    continue
                m = 1.0 / (1.0 + sr) if sr > 0 else 1.0
                if cash > 0 and pos > 0:
                    ref = close[pos - 1]
                    if ref and ref > 0 and not np.isnan(ref):
                        m *= max(0.01, (ref - cash) / ref)
                if m > 0:
                    log_mult[pos] += np.log(m)
        # factor(t) = 严格晚于 t 的事件乘数之积
        factor = np.exp(log_mult.sum() - np.cumsum(log_mult))

        raw_close = close / factor          # 真实成交价
        shares = g["outstanding_share"].to_numpy(dtype=float)

        out = pd.DataFrame({
            "date": g["date"],
            # 复权价:直接用源表,不再二次复权
            "open": g["open"].to_numpy(dtype=float),
            "high": g["high"].to_numpy(dtype=float),
            "low": g["low"].to_numpy(dtype=float),
            "close": close,
            "volume": g["volume"], "amount": g["amount"], "turnover": g["turnover"],
            "outstanding_share": g["outstanding_share"],
            "float_mv": raw_close * shares,             # 修正:真实市值
            "float_mv_legacy": g["float_mv"].to_numpy(dtype=float),
            "raw_close": raw_close,
            "is_st": g["is_st"], "is_suspended": g["is_suspended"],
            "is_limit_up": g["is_limit_up"], "is_limit_down": g["is_limit_down"],
            "listed_days": g["listed_days"],
        }).set_index("date")
        out.index = out.index.tz_localize("UTC")
        out.index.name = "date"
        out = out[~out.index.duplicated(keep="last")]

        # ---- 搬运财务列(与价格无关,不受本次缺陷影响) ----
        if code in old_files:
            try:
                oldf = pd.read_parquet(f"{OLD}/{code}.parquet", columns=FIN_COLS)
                for c in FIN_COLS:
                    out[c] = pd.to_numeric(oldf[c], errors="coerce").reindex(out.index)
            except Exception:
                for c in FIN_COLS:
                    out[c] = np.nan
                n_nofin += 1
        else:
            for c in FIN_COLS:
                out[c] = np.nan
            n_nofin += 1

        # bp 用真实成交价(与每股净资产同口径)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bp_correct"] = out["book_value_per_share"] / out["raw_close"].replace(0, np.nan)

        out.to_parquet(f"{OUT}/{code}.parquet")
        n_ok += 1

        # ---- 独立验证:重建价 vs amount/volume 实际成交均价 ----
        vol = out["volume"].astype(float)
        vwap = (out["amount"].astype(float) / vol.where(vol > 0))
        rel = (vwap / out["raw_close"] - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rel) >= 100:
            valid_rows.append({"code": code, "n": len(rel),
                               "median_dev": float(rel.median()),
                               "p95_abs_dev": float(rel.abs().quantile(0.95))})
    except Exception as exc:
        n_err += 1
        if n_err <= 5:
            print(f"  [ERROR] {code}: {type(exc).__name__}: {exc}")
    if n_ok % 1000 == 0 and n_ok:
        print(f"  {n_ok} 只完成  ({time.time()-t0:.0f}s)")

print(f"\n完成: 成功 {n_ok}, 无财务列 {n_nofin}, 出错 {n_err}, 耗时 {time.time()-t0:.0f}s")

# ---------------- 验证 ----------------
v = pd.DataFrame(valid_rows)
v.to_csv(f"{SP}/rebuild_validation.csv", index=False)
print(f"\n{'='*96}\n验证1: 重建的 raw_close vs 实际成交均价(amount/volume)\n{'='*96}")
print("  两者相互独立:raw_close 来自公司行动表反推,VWAP 来自成交金额/成交量")
print(f"  标的数 {len(v):,}")
print(f"  偏差中位数的中位数 : {v.median_dev.median():+.3%}   (VWAP 天然略偏离收盘价,几个百分点内属正常)")
print(f"  |偏差|<10% 的标的  : {(v.p95_abs_dev<0.10).mean():.1%}")
print(f"  |偏差|>50% 的标的  : {(v.p95_abs_dev>0.50).mean():.1%}   ← 应接近0,否则复权因子仍有错")
if (v.p95_abs_dev > 0.50).any():
    print("  偏差最大的5只:")
    print(v.nlargest(5, "p95_abs_dev").to_string(index=False))

print(f"\n{'='*96}\n验证2: 新数据是否还有'不可能的日收益'\n{'='*96}")
bad_new = bad_old = tot = 0
rng = np.random.default_rng(7)
sample = rng.choice(sorted(glob.glob(f"{OUT}/*.parquet")), size=400, replace=False)
for f in sample:
    c = os.path.basename(f)[:-8]
    s = pd.to_numeric(pd.read_parquet(f, columns=["close"])["close"], errors="coerce").dropna()
    if len(s) < 50:
        continue
    r = s.pct_change().dropna()
    lim = pd.Series(0.115, index=r.index)
    if c.startswith(("300", "301", "688")):
        lim[r.index >= pd.Timestamp("2020-08-24", tz="UTC")] = 0.215
    bad_new += int((r.abs() > lim).sum())
    tot += len(r)
    op = f"{OLD}/{c}.parquet"
    if os.path.exists(op):
        so = pd.to_numeric(pd.read_parquet(op, columns=["close"])["close"], errors="coerce").dropna()
        ro = so.pct_change().dropna()
        lo = pd.Series(0.115, index=ro.index)
        if c.startswith(("300", "301", "688")):
            lo[ro.index >= pd.Timestamp("2020-08-24", tz="UTC")] = 0.215
        bad_old += int((ro.abs() > lo).sum())
print(f"  随机400只, {tot:,} 个交易日")
print(f"  旧数据不可能日收益: {bad_old:,}")
print(f"  新数据不可能日收益: {bad_new:,}   ← 目标接近0")

print(f"\n{'='*96}\n验证3: float_mv 修正幅度\n{'='*96}")
tot_r = []
for f in sample[:200]:
    d = pd.read_parquet(f, columns=["float_mv", "float_mv_legacy"])
    r = (d["float_mv"] / d["float_mv_legacy"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r):
        tot_r.append({"code": os.path.basename(f)[:-8], "median_ratio": float(r.median()),
                      "max_ratio": float(r.max())})
tr = pd.DataFrame(tot_r)
print(f"  新/旧 float_mv 比值中位数: {tr.median_ratio.median():.3f}")
print(f"  比值>1.5 的标的比例      : {(tr.median_ratio>1.5).mean():.1%}  (被旧口径低估50%以上)")
print(f"  比值>3.0 的标的比例      : {(tr.median_ratio>3.0).mean():.1%}")
print(f"\n全部完成,耗时 {time.time()-t0:.0f}s。输出目录: {OUT}")
