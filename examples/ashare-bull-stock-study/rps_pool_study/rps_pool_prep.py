"""RPS 股池快照清洗对齐 —— 用户当时每周实时拉取并保存的两份快照

═══ 为什么这份数据和本session前四十二节都不同 ═══
前面所有 PIT(point-in-time)都是我**事后重建**的:用财报公告日错开、
用 fy_annual() 去累计、用 VWAP 反推真实成交价。重建再小心也有残余偏差。
**这两份是用户当时真拉下来存的快照,PIT 是天然的,不需要重建。**

已核实(read-only,见 plan):
  A 文件的「财务更新」日 **10,007 行中 0 行晚于快照日**,滞后中位 28 天
  → 财务字段确实是"当时能看到的",无前视。

═══ 两份是不同筛选,用户已明确要求分开看 ═══
  A  2023-10-16 ~ 2024-12-29  55期  每期中位218只  无RPS列
  B  2025-01-04 ~ 2026-07-09  72期  每期中位 90只  有RPS50/120/250

═══ 缺失代码不静默丢弃 ═══
A 缺 30 只 / B 缺 12 只(不在价格面板里)。**直接 dropna 等于制造幸存者
偏差**——买入后退市的股票正是最该计入亏损的那批。本脚本把它们单独标出,
由回测脚本按"剔除"与"按-100%计入"两种口径分别报告。
"""
import glob
import os

import numpy as np
import pandas as pd

SP = os.environ.get("OXQ_RESEARCH_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA = f"{SP}/oxq_stock_market_fixed"
IN = f"{SP}/rps_pool_input"

FILES = {"A": (f"{IN}/poolA_2023H2_2024.csv", "gbk"),
         "B": (f"{IN}/poolB_2025_2026.csv", "utf-8-sig")}


def to_num(s):
    """清洗数值列:'--  ' / 空串 → NaN;'286.44亿' → 286.44(单位统一为亿)。"""
    t = s.astype(str).str.strip()
    t = t.str.replace("亿", "", regex=False)
    t = t.replace({"--": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(t, errors="coerce")


def load(tag):
    path, enc = FILES[tag]
    d = pd.read_csv(path, encoding=enc, dtype={"代码": str})
    d["code"] = d["代码"].str.zfill(6)
    d["snap"] = pd.to_datetime(d["数据日期"].astype(str), format="%Y%m%d")
    d["profit_yoy"] = to_num(d["利润同比%"])
    d["rev_yoy"] = to_num(d["收入同比%"])
    d["pe"] = to_num(d["市盈(动)"])
    d["gross"] = to_num(d["毛利率%"])
    # 市值口径两份不同:A 只有总股本,B 有总市值 → 统一用面板的 float_mv,此处仅留原值备查
    d["mv_raw"] = to_num(d["总市值"]) if "总市值" in d.columns else to_num(d["总股本(亿)"])
    for c in ("RPS50", "RPS120", "RPS250"):
        d[c] = to_num(d[c]) if c in d.columns else np.nan
    d["ind"] = d["细分行业"].astype(str).str.strip() if "细分行业" in d.columns else ""
    # 用户的条件:净利润同比 > 0 且 收入同比 > 0
    d["dual"] = (d["profit_yoy"] > 0) & (d["rev_yoy"] > 0)
    d["p_pos"] = d["profit_yoy"] > 0
    d["r_pos"] = d["rev_yoy"] > 0
    return d[["code", "snap", "profit_yoy", "rev_yoy", "pe", "gross", "mv_raw",
              "RPS50", "RPS120", "RPS250", "ind", "dual", "p_pos", "r_pos"]]


if __name__ == "__main__":
    panel_codes = {os.path.basename(f)[:-8] for f in glob.glob(f"{DATA}/*.parquet")}
    out = {}
    for tag in FILES:
        d = load(tag)
        d["in_panel"] = d["code"].isin(panel_codes)
        out[tag] = d
        miss = d[~d.in_panel]
        print(f"\n{'='*92}\n股池 {tag}:{len(d):,} 行,{d.snap.nunique()} 期,"
              f"{d.snap.min().date()} ~ {d.snap.max().date()}")
        print(f"  每期只数 中位 {d.groupby('snap').size().median():.0f}")
        print(f"  双增长(利润>0 且 收入>0) **{d.dual.mean():.1%}**  "
              f"仅利润>0 {d.p_pos.mean():.1%}  仅收入>0 {d.r_pos.mean():.1%}")
        print(f"  不在价格面板中:{len(miss):,} 行 ({len(miss)/len(d):.2%})、"
              f"{miss.code.nunique()} 只")
        if len(miss):
            print(f"    代码:{sorted(miss.code.unique())[:15]}"
                  f"{' ...' if miss.code.nunique() > 15 else ''}")
            print(f"    这批的双增长占比 {miss.dual.mean():.1%}"
                  f"(与全池 {d.dual.mean():.1%} 对比,看缺失是否与条件相关)")
        d.to_parquet(f"{SP}/rps_pool_{tag}.parquet")
    print(f"\nSaved: rps_pool_A.parquet, rps_pool_B.parquet")
