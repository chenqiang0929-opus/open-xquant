"""把本地面板从 2026-08-03 扩到 2026-08-28,写到新目录(不改原面板)。

数据来源(均为 quant-research-dev 只读)
------------------------------------
- 行情:`mktdata_enriched/2026.parquet`(4,986 只,2026-01-05 → 2026-08-28)
- 财务:`mktdata_enriched/others/financials.parquet`(publish_date 至 2026-08-31)

已核验的两件事(先验后接)
------------------------
1. **新数据的 `close` 就是前复权,且与本地面板逐字相同** ——
   抽 334 只在重叠段(约 140 个交易日)比对:`新close/本地close` 的变异系数
   **334/334 全部 < 1e-6,比值中位 1.0**。因此直接 append,不需要重锚。
2. **`adj_factor` 是 `hfq_close/close` 的派生量**,因 close 只有 2 位小数
   而在 1e-4 量级抖动,不是真的每天变;本文件**没有不复权价列**。
   故 `raw_close` 按接缝比例外推:`raw_new = close_new × (raw本地/close本地)|接缝日`。
   **这在窗口内发生过除权除息的股票上会有偏差** —— 但 R09 四因子全部是价格无关的
   比率,价格只出现在 `ep>0 / cfp>0` 两个**符号门**上(价格恒正,符号不受影响),
   所以对本模板无实质影响;**若要跑 R08(估值三比值)则必须先取到真实不复权价。**

口径
----
- 本地面板缺席的 **247 只已退市股**:按最后有效价 ffill 保留,volume=0、停牌=True,
  **绝不剔除**(用户规则5)。
- 财务:对每个新交易日取 `publish_date ≤ 该日` 的最新一期累计值,填入
  eps / revenue / net_income / book_value_per_share / roe / operating_cash_flow。

产出:`/home/user/oxq-panel-0828/oxq_stock_market_fixed/*.parquet`
"""

from __future__ import annotations

import glob
import os
import shutil
import time

import numpy as np
import pandas as pd

SRC = "/home/user/oxq-panel/oxq_stock_market_fixed"
DST = "/home/user/oxq-panel-0828/oxq_stock_market_fixed"
MKT = "/home/user/quant-research-dev/mktdata_enriched/2026.parquet"
FIN = "/home/user/quant-research-dev/mktdata_enriched/others/financials.parquet"
SEAM = pd.Timestamp("2026-08-03")
END = pd.Timestamp("2026-08-28")
FCOL = {"eps": "eps", "revenue": "revenue", "net_income": "net_income",
        "book_value_per_share": "book_value_per_share", "roe": "roe",
        "operating_cash_flow": "operating_cash_flow"}


def main():  # noqa: PLR0915
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    m = pd.read_parquet(MKT)
    m["code"] = m["code"].astype(str).str.zfill(6)
    m = m[m["date"] > SEAM].copy()
    newd = sorted(m["date"].unique())
    print(f"新增交易日 {len(newd)} 个:{pd.Timestamp(newd[0]).date()} → "
          f"{pd.Timestamp(newd[-1]).date()}", flush=True)
    mg = {c: g.set_index("date").sort_index() for c, g in m.groupby("code")}

    fin = pd.read_parquet(FIN)
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    fin = fin[fin["publish_date"] <= END].sort_values(["code", "publish_date"])
    fg = {c: g for c, g in fin.groupby("code")}
    print(f"财务 {len(fin):,} 行 / {len(fg):,} 只(publish_date ≤ {END.date()}) "
          f"({time.time()-t0:.0f}s)", flush=True)

    files = sorted(glob.glob(f"{SRC}/*.parquet"))
    seam_bad, no_new, done, fin_upd = [], 0, 0, 0
    for f in files:
        c = os.path.basename(f)[:-8]
        a = pd.read_parquet(f)
        tz = getattr(a.index, "tz", None)
        a.index = pd.to_datetime(a.index).tz_localize(None)
        if a.index.max() != SEAM and c != "510300":
            pass
        cols = list(a.columns)
        b = mg.get(c)
        if b is None or c == "510300":
            no_new += 1
            last = a.iloc[-1]
            lc = float(last["close"]) if np.isfinite(last["close"]) else np.nan
            rows = []
            for d in newd:
                r = {k: np.nan for k in cols}
                for k in ("close", "open", "high", "low"):
                    if k in cols:
                        r[k] = lc
                if "raw_close" in cols:
                    r["raw_close"] = last.get("raw_close", np.nan)
                for k in ("volume", "amount", "turnover"):
                    if k in cols:
                        r[k] = 0.0
                for k in ("outstanding_share", "float_mv", "is_st", "listed_days",
                          *FCOL):
                    if k in cols:
                        r[k] = last.get(k, np.nan)
                if "is_suspended" in cols:
                    r["is_suspended"] = True
                for k in ("is_limit_up", "is_limit_down"):
                    if k in cols:
                        r[k] = False
                rows.append(r)
            ext = pd.DataFrame(rows, index=pd.DatetimeIndex(newd))
        else:
            ov = a.index.intersection(b.index)
            k = 1.0
            if len(ov) >= 20:
                q1 = a.loc[ov, "close"].to_numpy(float)
                q2 = b.loc[ov, "close"].to_numpy(float)
                g = np.isfinite(q1) & np.isfinite(q2) & (q1 > 0)
                if g.sum() >= 20:
                    rt = q2[g] / q1[g]
                    k = float(np.median(rt))
                    cv = float(np.std(rt) / np.mean(rt))
                    if cv > 1e-3:
                        seam_bad.append((c, cv, k))
            rr = (float(a.loc[SEAM, "raw_close"] / a.loc[SEAM, "close"])
                  if SEAM in a.index and np.isfinite(a.loc[SEAM, "close"])
                  and a.loc[SEAM, "close"] > 0 else np.nan)
            rows = []
            for d in newd:
                if d not in b.index:
                    continue
                s = b.loc[d]
                r = {kk: np.nan for kk in cols}
                for kk in ("open", "high", "low", "close"):
                    if kk in cols:
                        r[kk] = float(s[kk]) / k
                if "raw_close" in cols:
                    r["raw_close"] = float(s["close"]) / k * rr
                for kk in ("volume", "amount", "turnover", "outstanding_share",
                           "float_mv", "is_st", "is_suspended", "is_limit_up",
                           "is_limit_down", "listed_days"):
                    if kk in cols and kk in b.columns:
                        r[kk] = s[kk]
                rows.append(r)
            ext = pd.DataFrame(rows, index=pd.DatetimeIndex(
                [d for d in newd if d in b.index]))
        # ---- 财务 PIT ----
        fdf = fg.get(c)
        if fdf is not None and len(ext):
            for d in ext.index:
                sub = fdf[fdf["publish_date"] <= d]
                if not len(sub):
                    continue
                last = sub.iloc[-1]
                for src, dst in FCOL.items():
                    if dst in cols and pd.notna(last.get(src)):
                        ext.loc[d, dst] = float(last[src])
            fin_upd += 1
        if not len(ext):
            shutil.copy(f, f"{DST}/{c}.parquet")
            done += 1
            continue
        for kk in cols:
            if kk not in ext.columns:
                ext[kk] = np.nan
        out = pd.concat([a, ext[cols]])
        out = out[~out.index.duplicated(keep="first")].sort_index()
        if tz is not None:
            out.index = out.index.tz_localize("UTC")
        out.to_parquet(f"{DST}/{c}.parquet")
        done += 1
        if done % 1000 == 0:
            print(f"  {done}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n完成 {done} 只;新数据缺席按 ffill 保留 {no_new} 只;"
          f"财务更新 {fin_upd} 只")
    print(f"接缝比例非恒定(变异系数>1e-3)的 {len(seam_bad)} 只"
          f"{'' if not seam_bad else ':' + str(seam_bad[:5])}")
    chk = pd.read_parquet(f"{DST}/600066.parquet", columns=["close", "net_income"])
    chk.index = pd.to_datetime(chk.index).tz_localize(None)
    print(f"\n抽查 600066:行数 {len(chk)},末日 {chk.index.max().date()},"
          f"末值 close={chk['close'].iloc[-1]:.2f}")
    print(f"落库 {DST}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
