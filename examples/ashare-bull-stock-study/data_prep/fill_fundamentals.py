"""把 financials.parquet 的六列财务字段按 point-in-time 填进已重建的面板。

═══ 为什么不重跑 rebuild ═══
价格面板已经验证过:锚点 +4.61%/+6.34% 通过,且 §54 整张表逐位复现。
重跑整条链有可能引入新变量。**本脚本只加列,一个价格字节都不动**,
所以价格锚点必然保持不变。

═══ 时间语义(财务数据最容易在这里引入前视) ═══
源表 publish_date <= report_date 的比例实测 **0.0%**,发布滞后中位 62 天。
按 `publish_date` 做 merge_asof(direction=backward),
即**财报只在其公布日及之后可见**,不引入未来财报。

═══ 存原始值,不做清洗 ═══
面板里的 net_income/revenue 是**本年累计(YTD)**,这是原面板的状态,
§52 的 build_clean_growth.py 负责下游去累计。这里原样搬运,
否则 build_clean_growth 的自检口径会失去意义。

═══ 验收 ═══
  1. 价格列逐字节不变(脚本自检)
  2. build_clean_growth.py 的月份极差:清洗前应 ≈25.1pp、清洗后 ≤5pp
  3. §63 锚点 70,318 笔 / +4.61%
"""
import glob
import os
import time

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", "/home/user/oxq-panel")
DATA = f"{SP}/oxq_stock_market_fixed"
FIN_SRC = "/workspace/etf-netflow-dev/mktdata_enriched/others/financials.parquet"
FIN_COLS = ["eps", "revenue", "net_income", "book_value_per_share", "roe",
            "operating_cash_flow"]
PRICE_COLS = ["open", "high", "low", "close", "volume", "amount", "turnover",
              "outstanding_share", "float_mv", "raw_close"]

t0 = time.time()
fin = pd.read_parquet(FIN_SRC, columns=["code", "report_date", "publish_date"] + FIN_COLS)
fin["code"] = fin["code"].astype(str).str.zfill(6)
fin["publish_date"] = pd.to_datetime(fin["publish_date"])
fin["report_date"] = pd.to_datetime(fin["report_date"])
assert (fin["publish_date"] > fin["report_date"]).all(), "存在 publish<=report,前视风险"
# 同一公布日可能落多期报告,取报告期最新的那条
fin = (fin.sort_values(["code", "publish_date", "report_date"])
          .drop_duplicates(["code", "publish_date"], keep="last"))
by_code = {c: g for c, g in fin.groupby("code", sort=False)}
print(f"财务源 {len(fin):,} 行 / {len(by_code):,} 只  ({time.time()-t0:.0f}s)")

files = [f for f in sorted(glob.glob(f"{DATA}/*.parquet")) if not f.endswith("510300.parquet")]
print(f"面板文件 {len(files):,}")

n_fill = n_nofin = 0
chk_price = []
for i, f in enumerate(files):
    code = os.path.basename(f)[:-8]
    d = pd.read_parquet(f)
    before = {c: d[c].to_numpy(copy=True) for c in PRICE_COLS if c in d.columns}

    g = by_code.get(code)
    if g is None or g.empty:
        for c in FIN_COLS:
            d[c] = np.nan
        n_nofin += 1
    else:
        # 面板索引是 datetime64[ms],财务表是 [us] —— merge_asof 要求两边同精度
        left = pd.DataFrame({"_t": d.index.tz_localize(None).astype("datetime64[ns]")})
        left = left.sort_values("_t")
        right = g[["publish_date"] + FIN_COLS].copy()
        right["publish_date"] = right["publish_date"].astype("datetime64[ns]")
        right = right.sort_values("publish_date")
        m = pd.merge_asof(left, right, left_on="_t", right_on="publish_date",
                          direction="backward", allow_exact_matches=True)
        for c in FIN_COLS:
            d[c] = m[c].to_numpy()
        n_fill += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        d["bp_correct"] = d["book_value_per_share"] / d["raw_close"].replace(0, np.nan)

    # 自检:价格列必须逐值不变
    for c, v in before.items():
        now = d[c].to_numpy()
        if not np.array_equal(v, now, equal_nan=True):
            raise SystemExit(f"价格列被改动: {code} / {c}")
    chk_price.append(code)
    d.to_parquet(f)
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n完成: 有财务 {n_fill:,}, 无财务 {n_nofin:,}, "
      f"价格列自检通过 {len(chk_price):,}  ({time.time()-t0:.0f}s)")

# 覆盖率抽查
smp = pd.read_parquet(f"{DATA}/600519.parquet", columns=["net_income", "revenue", "close"])
nn = smp["net_income"].notna()
print(f"\n600519 net_income 非空 {nn.mean():.1%},首个非空 {smp.index[nn][0].date()}")
print(smp[nn].iloc[[0, -1]].to_string())
